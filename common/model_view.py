"""A common-model (SPEC_21) line schema as the extraction model sees it.

The bundle a line overlay loads as (``common.schemas._bundle``) is the client's
contract: every value an envelope with ``confidence`` and ``flagged``, rules the
decoder cannot enforce, fields the pipeline fills, notes for schema authors.
Validation judges a served answer against all of it. The model is shown, and
held to, something narrower:

* every value as ``raw`` / ``parsed`` / ``page_ref`` - confidence and the review
  flag are the pipeline's to fill, from the logprobs and the thresholds;
* no field the pipeline fills itself (the document's type, modality, page count
  and file name, whether the policy is a package, a form's page range, a
  coverage's id) and no full-text tier;
* ``dependentRequired`` (a percentage limit names the coverage it is a
  percentage of) rewritten as ``anyOf`` variants the decoder can enforce; the
  client's ``if``/``then`` rules (a flat deductible has an amount, a sublimit a
  description) dropped, since a window can hold such a row without the printed
  value they require; and the keywords xgrammar refuses (``minProperties``,
  ``uniqueItems``, ...) dropped;
* the line's coverage codes, each with its meaning, as the only values a code
  field of the line's own coverages takes - an umbrella's underlying policy,
  and its limits, code another line's coverages and keep the client's string;
* descriptions written for the model where the client's are written for schema
  authors (``configs/model_view.yaml``).

Rendered into the prompt and handed to structured decoding alike, so the model is
both told and constrained the same thing.
"""

from __future__ import annotations

import copy
from functools import cache
from typing import Any

from common.config import CONFIG_DIR, load_yaml

#: Repo-owned overrides of the model view. The client's schema folder is theirs
#: and never edited, so a description written for schema authors is replaced
#: here, until the client's own text is rewritten for the model.
MODEL_VIEW_CONFIG = CONFIG_DIR / "model_view.yaml"

#: What each coverage code means, written by someone who knows the coverages
#: (review P2, decision 2026-11-24). A code is shown with its meaning - and its
#: ``tell_apart``, when given - once filled; until then with its name from the
#: line's code file, as before. Part of the prompt hash (common.prompts).
COVERAGE_CODE_MEANINGS = CONFIG_DIR / "coverage_code_meanings.yaml"

#: Fields the pipeline fills, never the model, per common-model definition.
#: Shown, they would teach the model to write values no page prints.
PIPELINE_FILLED: dict[str, tuple[str, ...]] = {
    "document": ("doc_type", "modality", "page_count", "source_file_name"),
    "policy": ("is_package",),
    "Form": ("page_range",),
    "Coverage": ("coverage_id",),
}

#: Fields that hold one of the line's coverage codes, per definition. Each is
#: held to the line's list (``CoverageCode``). An underlying policy's
#: ``coverage_code`` is not one: it codes a coverage of the policy an umbrella
#: sits over (an auto or home liability), another line's, which the umbrella's
#: own list cannot name. It keeps the client's plain string.
CODE_FIELDS: dict[str, tuple[str, ...]] = {
    "Coverage": ("coverage_code",),
    "Limit": ("basis_coverage_code",),
    "PremiumItem": ("coverage_code",),
}
CODE_LIST_FIELDS: dict[str, tuple[str, ...]] = {"Deductible": ("applies_to_coverages",)}

#: Plain strings that ARE printed, per definition: a coverage's form numbers,
#: copied as printed. Typed like the ids and codes, they rode with every window
#: that showed any of their row - teaching form numbers to windows that do not
#: show them - and the prompt called every plain string unprinted. The window
#: target places them by where they are printed
#: (data_pipeline.dataset_builder.policy_windows), and their description says
#: they are printed (configs/model_view.yaml).
PRINTED_STRING_FIELDS: dict[str, tuple[str, ...]] = {"Coverage": ("form_refs",)}

#: A table that gets its own copy of a shared definition: (definition, field)
#: -> (shared definition, the copy). An underlying policy's limits are the
#: client's Limit, but a percentage limit there is a percentage of the
#: underlying policy's coverage (40% of X_VEHICLE_LIABILITY) - another line's
#: code, like the policy's own ``coverage_code``. The copy is not in
#: :data:`CODE_FIELDS`, so its ``basis_coverage_code`` keeps the client's plain
#: string, while a coverage's own limit stays held to the line's list. Every
#: other step treats the copy as the definition it copies: it is split into the
#: same variants, and a description written for ``Limit`` describes it too.
OWN_COPIES: dict[tuple[str, str], tuple[str, str]] = {
    ("UnderlyingPolicy", "limits"): ("Limit", "UnderlyingLimit"),
}

COVERAGE_CODE_DEF = "CoverageCode"

#: Keywords the decoder cannot use: xgrammar refuses the first four outright
#: (vLLM's has_xgrammar_unsupported_json_features), and of the rest
#: ``dependentRequired`` is rewritten as variants first (:func:`_as_variants`),
#: the client's ``if``/``then`` rules are dropped as rules a window cannot always
#: satisfy, and the others carry nothing the model acts on.
_DROPPED_KEYWORDS = frozenset({
    "minProperties", "maxProperties", "uniqueItems", "propertyNames", "patternProperties",
    "format", "default", "allOf", "if", "then", "else", "dependentRequired",
})


class ModelViewError(RuntimeError):
    """Raised when a bundle cannot be turned into the model's view of it."""


@cache
def model_view_config() -> dict[str, Any]:
    return load_yaml(MODEL_VIEW_CONFIG)


@cache
def coverage_code_meanings() -> dict[str, Any]:
    return load_yaml(COVERAGE_CODE_MEANINGS) if COVERAGE_CODE_MEANINGS.exists() else {}


def code_meaning(code: str, lob: str | None) -> str | None:
    """``code``'s filled meaning for ``lob`` (the line's own entry, else the
    shared one), with its ``tell_apart`` after it; None while it is empty."""
    meanings = coverage_code_meanings()
    entry = (meanings.get(lob) or {}).get(code) if lob else None
    entry = entry or (meanings.get("shared") or {}).get(code) or {}
    meaning = str(entry.get("meaning") or "").strip()
    if not meaning:
        return None
    tell_apart = str(entry.get("tell_apart") or "").strip()
    return f"{meaning} {tell_apart}" if tell_apart else meaning


def model_view(bundle: dict[str, Any]) -> dict[str, Any]:
    """``bundle`` (whole or sliced) as the model is shown and constrained to it."""
    config = model_view_config()
    schema = copy.deepcopy(bundle)
    defs: dict[str, Any] = schema.setdefault("$defs", {})
    lob = schema.get("fideon:lob")
    # Each code by its meaning once one is written, else by its name (review P2).
    names: dict[str, str] = {code: code_meaning(code, lob if isinstance(lob, str) else None) or name
                             for code, name in (schema.get("fideon:coverage_code_names") or {}).items()}

    _drop_top_level(schema)
    for name, fields in PIPELINE_FILLED.items():
        _drop_fields(defs.get(name), fields)
    for name, defn in list(defs.items()):
        if _is_value_type(defn):
            defs[name] = _narrowed_value(defn)
    copies = _own_copies(defs)
    if names:
        defs[COVERAGE_CODE_DEF] = _coverage_code_def(names, config.get("coverage_code_style", "oneof"))
        _point_code_fields(defs)
    if lob and "LobPart" in defs and "lob" in defs["LobPart"].get("properties", {}):
        defs["LobPart"]["properties"]["lob"] = {"const": lob, "description": "Line of business of the part."}
    _apply_descriptions(defs, _with_copies(config.get("descriptions") or {}, copies))
    if lob:
        _apply_id_patterns(schema, defs, lob)
    for name, defn in list(defs.items()):
        variants = _as_variants(defn)
        if variants is not None:
            defs[name] = variants
    if isinstance(schema.get("required"), list):
        # A required LIST would force an empty one into every window without it;
        # objects stay required and are written as {} when nothing in them is stated.
        schema["required"] = [r for r in schema["required"] if not _is_array_node(
            (schema.get("properties") or {}).get(r), defs)]
        if not schema["required"]:
            del schema["required"]
    _relax_required_lists(defs)

    cleaned = _without_keywords(schema)
    cleaned = {k: v for k, v in cleaned.items() if not k.startswith("fideon:")}
    return _strip_fideon(_pruned(cleaned))


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


def _drop_top_level(schema: dict[str, Any]) -> None:
    """The file's own title and description, the full-text tier, and the
    ``fideon:`` properties (provenance): none is the model's to write."""
    schema.pop("title", None)
    schema.pop("description", None)
    properties = schema.get("properties") or {}
    for name in [n for n in properties if n == "text_sections" or n.startswith("fideon:")]:
        del properties[name]
    if isinstance(schema.get("required"), list):
        schema["required"] = [r for r in schema["required"] if r in properties]


def _drop_fields(defn: dict[str, Any] | None, fields: tuple[str, ...]) -> None:
    if not isinstance(defn, dict):
        return
    properties = defn.get("properties") or {}
    for field in fields:
        properties.pop(field, None)
    if isinstance(defn.get("required"), list):
        defn["required"] = [r for r in defn["required"] if r not in fields]


def _is_value_type(defn: Any) -> bool:
    return isinstance(defn, dict) and {"raw", "parsed", "page_ref"} <= set(defn.get("properties") or {})


def _narrowed_value(defn: dict[str, Any]) -> dict[str, Any]:
    """A value as the model writes it. ``raw`` is never null: a value the
    document does not state is omitted, not written empty."""
    out: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["raw", "parsed", "page_ref"],
        "properties": {
            "raw": {"type": "string"},
            "parsed": copy.deepcopy(defn["properties"]["parsed"]),
            "page_ref": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1},
        },
    }
    if defn.get("description"):
        out["description"] = defn["description"]
    return out


def _own_copies(defs: dict[str, Any]) -> dict[str, str]:
    """Give each table in :data:`OWN_COPIES` its own copy of the definition its
    rows share, before any code field is pointed at the line's list. Returns
    copy -> shared, for the copies made: a bundle or slice without the table,
    or whose rows are not the shared definition, gets none."""
    made: dict[str, str] = {}
    for (owner, field), (shared, copy_name) in OWN_COPIES.items():
        prop = ((defs.get(owner) or {}).get("properties") or {}).get(field)
        items = prop.get("items") if isinstance(prop, dict) else None
        if not (isinstance(items, dict) and items.get("$ref") == f"#/$defs/{shared}" and shared in defs):
            continue
        defs[copy_name] = copy.deepcopy(defs[shared])
        prop["items"] = {**items, "$ref": f"#/$defs/{copy_name}"}
        made[copy_name] = shared
    return made


def _with_copies(overrides: dict[str, str | None], copies: dict[str, str]) -> dict[str, str | None]:
    """``overrides`` with each one written for a shared definition repeated for
    its copies, so the model reads the same text on both."""
    out = dict(overrides)
    for copy_name, shared in copies.items():
        for path, text in overrides.items():
            name, dot, field = path.partition(".")
            if name == shared:
                out.setdefault(f"{copy_name}{dot}{field}", text)
    return out


def _coverage_code_def(names: dict[str, str], style: str) -> dict[str, Any]:
    if style == "enum":
        return {
            "type": "string",
            "enum": list(names),
            "description": "; ".join(f"{code}: {name}" for code, name in names.items()),
        }
    if style != "oneof":
        raise ModelViewError(f"coverage_code_style must be 'oneof' or 'enum', not {style!r}")
    return {"type": "string", "oneOf": [{"const": c, "description": n} for c, n in names.items()]}


def _point_code_fields(defs: dict[str, Any]) -> None:
    ref = {"$ref": f"#/$defs/{COVERAGE_CODE_DEF}"}
    for name, fields in CODE_FIELDS.items():
        properties = (defs.get(name) or {}).get("properties") or {}
        for field in fields:
            if field in properties:
                properties[field] = {**ref, **_described(properties[field])}
    for name, fields in CODE_LIST_FIELDS.items():
        properties = (defs.get(name) or {}).get("properties") or {}
        for field in fields:
            if field in properties:
                properties[field] = {"type": "array", "items": dict(ref), **_described(properties[field])}


def _described(node: dict[str, Any]) -> dict[str, Any]:
    return {"description": node["description"]} if node.get("description") else {}


def _apply_descriptions(defs: dict[str, Any], overrides: dict[str, str | None]) -> None:
    """``Def`` or ``Def.field`` -> the description the model reads (None: none)."""
    for path, text in overrides.items():
        node = _description_target(defs, path)
        if node is None:
            continue  # this bundle (or slice) does not carry it
        if text is None:
            node.pop("description", None)
        else:
            node["description"] = text


def _description_target(defs: dict[str, Any], path: str) -> dict[str, Any] | None:
    name, _, field = path.partition(".")
    node = defs.get(name)
    if not isinstance(node, dict):
        return None
    if not field:
        return node
    target = (node.get("properties") or {}).get(field)
    return target if isinstance(target, dict) else None


def _apply_id_patterns(schema: dict[str, Any], defs: dict[str, Any], lob: str) -> None:
    """Each table's structural id held to its own prefix (``veh_2``, not ``loc_2``),
    from the section map's ``ids``. The common model allows any ``word_N``."""
    from common.schema_sections import structural_ids

    for table, spec in structural_ids(lob).items():
        block = _resolved((schema.get("properties") or {}).get(table), defs)
        row = _resolved(block.get("items"), defs) if block.get("type") == "array" else {}
        field = (row.get("properties") or {}).get(spec["field"])
        if isinstance(field, dict):
            field["pattern"] = f"^{spec['prefix']}_[0-9]+$"


def _as_variants(defn: Any) -> dict[str, Any] | None:
    """A definition with a ``dependentRequired`` rule, as ``anyOf`` variants.

    One occurs in the common model (Limit: a percentage names the coverage it is
    a percentage of). Each dependency splits every variant in two: the property
    present with its dependants required, or absent altogether. A window can
    always satisfy it, because the dependant is a bare code that rides with its
    row into every window that shows any of it.

    The client's ``if``/``then`` rules are not rewritten (Limit: a sublimit
    names what it limits; Deductible: a flat one has an amount, a percentage
    one a percentage). Each requires a printed value, and a window can hold the
    row without it - the amount printed on another page, or not stated at all -
    so a variant requiring it would be a grammar that window's target cannot
    fit. The field they switch on keeps its plain list of values, and the rules
    are dropped with the other keywords the decoder cannot use, like the other
    client rules a window cannot satisfy. Validation still judges an answer
    against the client's own schema.

    Every object is closed (``additionalProperties: false``), so a property left
    out of a variant cannot be written in it. Descriptions are kept on the first
    variant only: the model reads them once.
    """
    if not isinstance(defn, dict) or "properties" not in defn:
        return None
    dependents: dict[str, list[str]] = dict(defn.get("dependentRequired") or {})
    if not dependents:
        return None

    properties: dict[str, Any] = defn["properties"]
    # Per variant: the properties it leaves out, and the ones it requires.
    branches: list[tuple[frozenset[str], list[str]]] = [(frozenset(), [])]
    for prop, needs in dependents.items():
        split: list[tuple[frozenset[str], list[str]]] = []
        for absent, extra in branches:
            split.append((absent, [*extra, prop, *needs]))
            split.append((absent | {prop}, list(extra)))
        branches = split

    base_required = list(defn.get("required") or [])
    variants = []
    for index, (absent, extra) in enumerate(branches):
        shown = {
            name: sub if index == 0 else {k: v for k, v in sub.items() if k != "description"}
            for name, sub in properties.items() if name not in absent
        }
        required = list(dict.fromkeys(r for r in [*base_required, *extra] if r in shown))
        variants.append({"type": "object", "additionalProperties": False,
                         "properties": shown, "required": required})
    kept = {k: v for k, v in defn.items() if k in ("description",)}
    return {**kept, "anyOf": variants}


def _resolved(node: Any, defs: dict[str, Any]) -> dict[str, Any]:
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 20:
        node = defs.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
        seen += 1
    return node if isinstance(node, dict) else {}


def _is_array_node(node: Any, defs: dict[str, Any]) -> bool:
    return _resolved(node, defs).get("type") == "array" if isinstance(node, dict) else False


def _relax_required_lists(defs: dict[str, Any]) -> None:
    """A top-level block that is itself a list (``lob_parts``) loses its minimum:
    the pipeline supplies the part when the model reads none."""
    lob_parts = defs.get("lob_parts")
    if isinstance(lob_parts, dict):
        lob_parts.pop("minItems", None)


def _without_keywords(node: Any) -> Any:
    if isinstance(node, list):
        return [_without_keywords(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in _DROPPED_KEYWORDS:
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            # Names here are field or definition names, never keywords: a field
            # called "default" or "format" must survive.
            out[key] = {name: _without_keywords(sub) for name, sub in value.items()}
        else:
            out[key] = _without_keywords(value)
    return out


def _strip_fideon(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip_fideon(item) for item in node]
    if not isinstance(node, dict):
        return node
    return {k: _strip_fideon(v) for k, v in node.items() if not str(k).startswith("fideon:")}


def _pruned(schema: dict[str, Any]) -> dict[str, Any]:
    """Only the definitions the schema's properties reach, transitively."""
    from common.schemas import _local_ref_names

    defs = schema.get("$defs") or {}
    reached = _local_ref_names(schema.get("properties") or {})
    pending = sorted(reached)
    while pending:
        for name in _local_ref_names(defs.get(pending.pop(), {})):
            if name not in reached:
                reached.add(name)
                pending.append(name)
    return {**schema, "$defs": {k: v for k, v in defs.items() if k in reached}}
