"""How rows of a table are told apart and documents flattened for comparison (review A1: moved from
``evaluation.metrics.field_accuracy``, which re-exports them). The served merge and the scorer pair rows by
the same identifiers, and calibration flattens a document the way the scorer does."""

from __future__ import annotations

from typing import Any

#: Fields that are containers, not values — scored by their rows, not directly.
_CONTAINER_TYPES = (list, dict)


def flatten_scalars(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten to scalar leaves, keeping list ROWS addressable.

    ``claims[0].paid`` rather than ``claims`` — per-value accuracy inside a row
    is meaningful, an aggregate over a whole table is not.

    **A list of scalars is one field, not a table.** ``line_of_business`` is a
    set of enum values (arch v2.1 §0b), so exploding it into
    ``line_of_business[0]`` would make position part of its identity — and since
    ``score_fields`` skips any path containing ``[``, it also made the field
    disappear from scoring entirely the moment it stopped being a scalar. Kept
    whole, ``values_match`` compares it as the set it is.

    **A canonical ``FieldValue`` envelope is one field, scored on its value.**
    Flattened as an object it would score ``raw``, ``page_ref`` and the
    pipeline's own ``confidence`` as though they were extracted fields; collapsed
    through :func:`common.canonical.values_view`, a canonical label and a
    canonical extraction compare exactly as two flat documents do.
    """
    from common.canonical import is_field_value, values_view

    out: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if is_field_value(value):
            out[path] = values_view(value)
        elif isinstance(value, dict):
            out.update(flatten_scalars(value, f"{path}."))
        elif isinstance(value, list):
            if value and all(is_field_value(item) for item in value):
                # A list of envelopes is a list of scalars: one field, a set.
                out[path] = values_view(value)
            elif any(isinstance(item, dict) for item in value):
                # A table: rows are addressable, and an empty one contributes
                # nothing rather than a phantom field.
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        out.update(flatten_scalars(item, f"{path}[{index}]."))
                    else:
                        out[f"{path}[{index}]"] = item
            else:
                out[path] = value
        else:
            out[path] = value
    return out

def _key_text(value: Any) -> str:
    """An identifier as compared: case, spacing and separators ignored.

    ``HO 00 03`` and ``HO-0003``, ``1HGCM82633A004352`` and ``1hgcm 8263 3a004352``
    name the same form and the same car. A number read as ``1`` and ``1.0`` too.
    """
    import re

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return re.sub(r"[\s\-./_]", "", str(value if value is not None else "")).casefold()

#: Fields that identify a row, strongest first. A table is matched on the first
#: one filled in for most of its expected rows. Loss Runs key on claim numbers;
#: the canonical policy tables on form numbers, VINs, numbered rows and names.
#: Knowing only the Loss Run keys, every policy table fell back to matching on
#: ALL its fields at once, so one wrong character anywhere dropped the whole row
#: from recall - a vehicle with a wrong premium counted as a missing vehicle,
#: and list recall read 0.0 on every smoke-run checkpoint.
ROW_IDENTIFIERS: tuple[str, ...] = (
    "claim_number", "policy_number", "form_number", "vin", "vin_or_hull_id",
    "hull_identification_number", "serial_number", "loan_number", "license_number",
    "docket_number", "vehicle_number", "driver_number", "unit_number", "motor_number",
    "item_number", "installment_number", "location_number", "structure_number",
    "residence_number", "object_number", "project_number", "agreement_number",
    "class_number", "question_number", "blanket_number", "coverage_code", "coverage_type",
    "coverage_name", "coverage_part", "discount_name", "exclusion_name", "benefit_name",
    "endorsement_name", "name", "individual_name", "entity_name", "identifier_type",
    "device_type", "device_description", "location_reference", "change_description",
    "field_changed", "exposure_type", "livestock_type", "plan_name", "option_name",
    "service_name", "vendor_name", "item_title", "rank", "description", "state", "label",
)

#: Identifiers that are only unique together with a companion: a location's
#: buildings are numbered 1, 2, … within each location.
#: A common-model coverage code names what is covered, not which unit: the same
#: code on two vehicles is two rows, told apart by the units they apply to.
_COMPANIONS: dict[str, tuple[str, ...]] = {
    "location_number": ("building_number",),
    "coverage_code": ("applies_to",),
}

#: Companions that count only where they hold references - a list of ids, or
#: of the units' own keys after ``comparable_view`` - as on a common-model line.
#: A self-contained line's ``applies_to`` is printed text ("Symbol 8"), and as a
#: key one wording of it against another unpaired rows whose codes agree.
_REFERENCE_COMPANIONS: frozenset[str] = frozenset({"applies_to"})

def _filled_share(rows: list[dict[str, Any]], field_name: str) -> float:
    filled = sum(1 for r in rows if _key_text(r.get(field_name)))
    return filled / len(rows) if rows else 0.0

def _holds_references(rows: list[dict[str, Any]], field_name: str) -> bool:
    """Whether every row that fills ``field_name`` holds a list there."""
    return all(isinstance(r.get(field_name), list) for r in rows if _key_text(r.get(field_name)))

def _infer_key_fields(rows: list[Any]) -> list[str]:
    """Pick identifying fields for row matching.

    The first :data:`ROW_IDENTIFIERS` entry filled in at least half the rows,
    with its companion when that is filled too (a reference companion only
    where it holds references, :data:`_REFERENCE_COMPANIONS`). Otherwise every
    scalar field any row carries, which makes matching strict rather than
    guessing at identity.

    Read across all rows: a label leaves an unstated value out of its row, so a
    first mortgagee with no name made a table whose other rows all had one fall
    back to matching on every field.
    """
    if not rows:
        return []
    # The non-dict guard comes FIRST. `candidate in rows[0]` was evaluated
    # before it, so a list of scalars — the exact case `_row_key` was hardened
    # for — raised TypeError from inside the scorer and took the whole eval run
    # down with it.
    dict_rows = [r for r in rows if isinstance(r, dict)]
    if not dict_rows:
        return []
    for candidate in ROW_IDENTIFIERS:
        if _filled_share(dict_rows, candidate) >= 0.5:
            companions = [c for c in _COMPANIONS.get(candidate, ())
                          if _filled_share(dict_rows, c) >= 0.5
                          and (c not in _REFERENCE_COMPANIONS or _holds_references(dict_rows, c))]
            return [candidate, *companions]
    return sorted({k for row in dict_rows for k, v in row.items()
                   if not isinstance(v, _CONTAINER_TYPES)})
