"""Merge a policy's section windows into one canonical document (arch v2.1 §7b).

A long policy is read as a cross product of section groups and page windows
(``configs/schema_sections.yaml``): every group over every page in several
windows, ``decl`` (and ``dtd``, where the line carries it) led by the declarations
pages. Each window returns its slice of the model form. This module
turns those partial answers into one document — and carries each value's token
spans with it, so confidence and ``flagged`` land on the value they describe.

**Deterministic, not model-mediated.** The same reason as
:mod:`serving.lossrun_merge`: asking the model to reconcile its own windows puts a
second sampling step between the extraction and the answer.

Rules, in order:

1. **Windows are merged in group order, then page order.** ``decl`` leads, so the
   declarations value wins any conflict by construction — the same rule
   ``policy.jinja`` states to the model.
2. **A value already present is kept.** A later window's differing value is a
   recorded conflict, never a silent overwrite.
3. **Arrays concatenate**, then rows are de-duplicated. A table crossing a window
   boundary legitimately comes back twice, so this is load-bearing, not tidying.
   Top-level arrays use the identifying key the section map declares
   (``array_key``), compared on normalised values; a row with no key value at all
   is kept rather than guessed into another. Nested arrays with no declared key
   collapse only rows whose values are identical.
4. **Duplicates keep the more complete row**, take any field only the other row
   has, and union the page references of fields both carry.

Spans travel by object identity rather than by path. A row's path changes when
rows are concatenated and de-duplicated; the envelope object does not.

**A common-model line** (SPEC_21) links rows by id, numbered within each window.
Its windows are merged with those ids namespaced per window; units (locations,
vehicles, drivers, ...) are joined on their own keys (VIN, location number),
every reference is rewritten to the unit it now is, the other tables are joined
on the line's own keys with references compared that way, and the whole
document's ids are then renumbered in printed order (common.structural_ids) -
the one numbering function training also uses.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from common.canonical import is_field_value, values_view
from common.normalize import normalize_text, normalize_value

log = logging.getLogger(__name__)


@dataclass
class PolicyWindow:
    """One window's answer: which group, over which pages, and what came back.

    ``extraction`` is the model form (sparse ``raw``/``parsed``/``page_ref``
    envelopes), dates already normalised. ``spans`` is keyed by values-view path,
    as :func:`common.canonical.collapse_spans` produces it.
    """

    group: str
    pages: list[int]
    extraction: dict[str, Any]
    spans: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


@dataclass
class MergedPolicy:
    """The merged model-form document, its spans, and what merging did."""

    extraction: dict[str, Any] = field(default_factory=dict)
    spans: dict[str, Any] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    #: Rows that came back without their identifier and were joined to the one
    #: identified row of their table. Right when the table has one row; flagged,
    #: because a second row whose identifier was never read would look the same.
    joined_unidentified: list[str] = field(default_factory=list)
    #: References (common-model lines) that named no row of the merged answer:
    #: a link the model wrote that the document cannot hold. Dropped, flagged.
    dangling_references: list[str] = field(default_factory=list)
    #: Unit tables (common-model lines) where a row was the same unit as two or
    #: more units known to differ - one fragment, two vehicles of one model. It
    #: was kept apart rather than joined to a guess, and is flagged.
    ambiguous_units: list[str] = field(default_factory=list)
    #: Unit tables (common-model lines) where a row stating no unit key was
    #: left a unit of its own, with two or more identified units it could have
    #: been part of: possibly a phantom unit, so flagged.
    unplaced_fragments: list[str] = field(default_factory=list)
    duplicates_collapsed: int = 0
    unkeyed_rows: int = 0

    @property
    def review_flags(self) -> list[str]:
        """A conflict is two windows reading one field differently. The merge
        kept one mechanically; a person should see which."""
        flags = [f"{path}:merge_conflict" for path in sorted({c.split(":", 1)[0] for c in self.conflicts})]
        flags += [f"{path}:joined_without_identifier" for path in sorted(set(self.joined_unidentified))]
        flags += [f"{table}:ambiguous_unit" for table in sorted(set(self.ambiguous_units))]
        flags += [f"{table}:unplaced_fragment" for table in sorted(set(self.unplaced_fragments))]
        return flags + [
            f"{ref.split('=', 1)[0]}:dangling_reference" for ref in sorted(set(self.dangling_references))
        ]


# --------------------------------------------------------------------------
# Spans by identity
# --------------------------------------------------------------------------


def _index_spans(node: Any, path: str, spans: Mapping[str, Any], out: dict[int, Any]) -> None:
    """Map each envelope object to the span the window recorded at its path."""
    if is_field_value(node):
        span = spans.get(path)
        if span is not None:
            out[id(node)] = span
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _index_spans(value, f"{path}.{key}" if path else key, spans, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _index_spans(item, f"{path}[{index}]", spans, out)


def _collect_spans(node: Any, path: str, by_id: Mapping[int, Any], out: dict[str, Any]) -> None:
    """Re-key spans to the merged document's values-view paths."""
    if is_field_value(node):
        span = by_id.get(id(node))
        if span is not None:
            out[path] = span
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _collect_spans(value, f"{path}.{key}" if path else key, by_id, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _collect_spans(item, f"{path}[{index}]", by_id, out)


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def _filled(node: Any) -> int:
    """How many stated values a node holds — "the more complete row"."""
    if is_field_value(node):
        return int(node.get("raw") is not None or node.get("parsed") is not None)
    if isinstance(node, dict):
        return sum(_filled(v) for v in node.values())
    if isinstance(node, list):
        return sum(_filled(v) for v in node)
    return int(node is not None)


def _union_pages(kept: dict[str, Any], other: dict[str, Any]) -> None:
    kept["page_ref"] = sorted({*kept.get("page_ref", []), *other.get("page_ref", [])})


def _same_value(a: dict[str, Any], b: dict[str, Any], path: str) -> bool:
    va, vb = values_view(a), values_view(b)
    if va == vb:
        return True
    return normalize_value(va, field_path=path) == normalize_value(vb, field_path=path)


def _merge_into(
    kept: Any, other: Any, path: str, report: MergedPolicy, *, top: bool,
    lob: str | list[str] | None = None,
) -> Any:
    """Merge ``other`` into ``kept``, keeping ``kept`` wherever both say something."""
    # An explicit null is an absence, not a value: the other window may state it.
    if kept is None:
        return other
    if other is None:
        return kept
    if is_field_value(kept) and is_field_value(other):
        if not _same_value(kept, other, path):
            report.conflicts.append(
                f"{path}: kept {values_view(kept)!r}, a later window read {values_view(other)!r}"
            )
        else:
            _union_pages(kept, other)
        return kept
    if isinstance(kept, dict) and isinstance(other, dict):
        for key, value in other.items():
            child = f"{path}.{key}" if path else key
            if key in kept:
                kept[key] = _merge_into(kept[key], value, child, report, top=False, lob=lob)
            else:
                kept[key] = value
        return kept
    if isinstance(kept, list) and isinstance(other, list):
        section = path if top else None
        return _dedupe(kept + other, path, section, report, lob=lob)
    if not isinstance(kept, (dict, list)) and not isinstance(other, (dict, list)):
        # Two bare values - a common-model code or type, read by two windows.
        # The same value is agreement, not a shape disagreement.
        if kept != other:
            report.conflicts.append(f"{path}: kept {kept!r}, a later window read {other!r}")
        return kept
    # A shape disagreement — one window wrote an object, another a value. The
    # earlier window's answer stands; the other is recorded, never dropped silently.
    report.conflicts.append(f"{path}: windows disagree on the shape of this field")
    return kept


def _row_key(row: Any, key_fields: tuple[str, ...], path: str) -> tuple | None:
    """A row's identity on normalised values, or ``None`` when it has none."""
    if not isinstance(row, dict) or not key_fields:
        return None
    parts = []
    for name in key_fields:
        value = values_view(row.get(name))
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            # A common-model reference list, already rewritten to the units it
            # names: the same set in any order is the same reference.
            parts.append(tuple(sorted(value)) or None)
            continue
        if isinstance(value, dict):
            leaves = sorted(
                normalize_text(v) or "" for v in _leaf_values(value) if v is not None
            )
            parts.append(tuple(leaves) or None)
        else:
            normalised = normalize_value(value, field_path=f"{path}[].{name}")
            # A value the typed normaliser cannot read is still a value. A form
            # edition such as `05/11` is not a parseable date, and dropping it
            # to None would make `HO 00 03 05/11` and `HO 00 03 10/00` — two
            # different forms — one row.
            if normalised is None and value is not None:
                normalised = normalize_text(value)
            parts.append(normalised)
    return tuple(parts) if any(p for p in parts) else None


def _leaf_values(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return [v for child in node.values() for v in _leaf_values(child)]
    if isinstance(node, list):
        return [v for child in node for v in _leaf_values(child)]
    return [node]


#: Common-model tables whose one row can stand on several units: a lienholder on
#: two vehicles, a multi-car discount. Their key leaves the units out.
_UNIT_SPANNING = ("interested_parties", "rating_modifiers")


def _dedupe(
    rows: list[Any], path: str, section: str | None, report: MergedPolicy,
    *, lob: str | list[str] | None = None,
) -> list[Any]:
    from common.schema_sections import array_key
    from common.schemas import is_common_model

    # A list of envelopes is a set of values, not a table: one entry per value.
    if rows and all(is_field_value(r) for r in rows):
        seen: dict[Any, dict[str, Any]] = {}
        out = []
        for row in rows:
            marker = normalize_value(values_view(row), field_path=path)
            if marker in seen:
                _union_pages(seen[marker], row)
                report.duplicates_collapsed += 1
                continue
            seen[marker] = row
            out.append(row)
        return out

    key_fields = array_key(section, lob) if section else ()
    # No declared key (the tables inside a line block: vehicles, units,
    # coverages): the identifier scoring matches rows on - VIN, unit number,
    # coverage name. Matching only identical rows left a vehicle read across two
    # windows as two half-rows. Rows sharing an identifier join only when no
    # field they both state disagrees; otherwise they are different rows.
    inferred = () if key_fields else _identifiers(rows)
    common_model = bool(key_fields) and is_common_model("policy", None, lob)
    out: list[Any] = []
    # Every row an identity names, not only the first: two real rows can share
    # an inferred identifier (two "Liability" coverages with different limits),
    # and a later window's re-read of the SECOND must join it, not be appended
    # again as a third.
    positions: dict[Any, list[int]] = {}
    for row in rows:
        if key_fields:
            identity = _row_key(row, key_fields, path)
        elif inferred:
            identity = _row_key(row, inferred, path)
        else:
            # Nothing identifies these rows: only an identical row is the same row.
            identity = ("exact", repr(values_view(row)))
        if identity is None:
            out.append(row)
            continue
        candidates = positions.setdefault(identity, [])
        # A declared key with a part missing - a common-model coverage whose
        # window could not name its unit - is no longer an identity on its own:
        # the same coverage code on two vehicles. Such rows join only when
        # nothing they both state differs, as rows on an inferred key do.
        # A common-model party or modifier is keyed without the units it stands
        # on (a window links only those it sees), so its key is as weak: the
        # rows join, their links united, unless a value both state differs.
        weak = common_model and (any(part is None for part in identity) or section in _UNIT_SPANNING)
        position = next(
            (i for i in candidates if not ((inferred or weak) and _disagree(out[i], row, path))), None
        )
        if position is None:
            candidates.append(len(out))
            out.append(row)
            continue
        kept, other = out[position], row
        if _filled(other) > _filled(kept):
            kept, other = other, kept
        out[position] = _merge_into(kept, other, f"{path}[{position}]", report, top=False, lob=lob)
        report.duplicates_collapsed += 1
    return out


#: Identifiers strong enough to join two windows' rows on. Scoring may match on
#: weaker ones (``description``, ``state``, ``label``, ``rank``) because a wrong
#: match there costs one comparison; a wrong JOIN merges two real rows into one
#: and loses the second, so the merge only trusts a value that names one row.
JOIN_IDENTIFIERS: frozenset[str] = frozenset({
    "claim_number", "policy_number", "form_number", "vin", "vin_or_hull_id",
    "hull_identification_number", "serial_number", "loan_number", "license_number",
    "docket_number", "vehicle_number", "driver_number", "unit_number", "motor_number",
    "item_number", "installment_number", "location_number", "building_number",
    "structure_number", "residence_number", "object_number", "agreement_number",
    "coverage_code", "coverage_name", "endorsement_name", "discount_name", "name",
    "individual_name", "entity_name",
})


def _attach_unidentified(node: Any, path: str, report: MergedPolicy, *, top: bool) -> Any:
    """Join rows that carry no identifier to the single identified row of their table.

    A row read across two windows comes back in two parts, and the part from the
    window that does not show the identifier (a vehicle's coverages, a page after
    its VIN) has nothing to be matched on. When the table holds exactly ONE
    identified row, the fragment is that row's; with two or more it is left as it
    is - which row owns it cannot be known. Tables inside a line block only: a
    top-level section's declared key already says what its rows are matched on.
    A fragment that states a different value for a field the row also states is
    never joined.
    """
    from evaluation.metrics.field_accuracy import ROW_IDENTIFIERS

    if is_field_value(node):
        return node
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = _attach_unidentified(value, f"{path}.{key}", report, top=False)
        return node
    if not (isinstance(node, list) and node and all(isinstance(r, dict) and not is_field_value(r) for r in node)):
        return node
    rows = [_attach_unidentified(row, f"{path}[{i}]", report, top=False) for i, row in enumerate(node)]
    if top:
        return rows

    def stated(row: dict[str, Any], name: str) -> bool:
        return is_field_value(row.get(name)) and values_view(row[name]) not in (None, "")

    name = next((n for n in ROW_IDENTIFIERS if n in JOIN_IDENTIFIERS and any(stated(r, n) for r in rows)), None)
    if name is None:
        return rows
    identified = [i for i, row in enumerate(rows) if stated(row, name)]
    if len(identified) != 1:
        return rows
    owner = rows[identified[0]]
    out = []
    for row in rows:
        if row is owner or _disagree(owner, row, path):
            out.append(row)
            continue
        # In place: the owner keeps its position, and its envelopes their spans.
        _merge_into(owner, row, path, report, top=False)
        report.joined_unidentified.append(path)
    return out


def _identifiers(rows: list[Any]) -> tuple[str, ...]:
    """The strong identifier these rows carry, as scoring infers it; ``()``
    when none is filled in most rows - those rows join only when identical."""
    from evaluation.metrics.field_accuracy import _infer_key_fields

    if not rows or not all(isinstance(r, dict) for r in rows):
        return ()
    keys = _infer_key_fields(values_view(rows))
    return tuple(k for k in keys if k in JOIN_IDENTIFIERS)


def _disagree(a: dict[str, Any], b: dict[str, Any], path: str) -> bool:
    """Whether two rows state a different value for any field both carry."""
    for key, value in a.items():
        other = b.get(key)
        if is_field_value(value) and is_field_value(other):
            stated = (values_view(value), values_view(other))
            if None not in stated and not _same_value(value, other, f"{path}[].{key}"):
                return True
    return False


def merge_policy_windows(
    windows: list[PolicyWindow], lob: str | list[str] | None = None
) -> MergedPolicy:
    """One canonical model-form document from a policy's windows.

    ``windows`` must arrive in group order, then page order —
    :func:`common.schema_sections.group_names` order, which puts ``decl`` first.
    ``lob`` selects the common-model merge for a SPEC_21 line.
    """
    from common.schema_sections import group_names
    from common.schemas import is_common_model

    # No line is the fallback, which composes the common model too.
    if is_common_model("policy", None, lob):
        return _merge_common_model(windows, lob)

    # The line named throughout: no line is the fallback, read with the
    # common-model profile, whose groups and row keys are not this line's.
    rank = {name: index for index, name in enumerate(group_names(lob))}
    ordered = sorted(
        windows, key=lambda w: (rank.get(w.group, len(rank)), min(w.pages or [0]))
    )

    by_id: dict[int, Any] = {}
    report = MergedPolicy()
    for window in ordered:
        _index_spans(window.extraction, "", window.spans, by_id)
        for key, value in window.extraction.items():
            if key in report.extraction:
                report.extraction[key] = _merge_into(
                    report.extraction[key], value, key, report, top=True, lob=lob
                )
            else:
                report.extraction[key] = (
                    _dedupe(value, key, key, report, lob=lob) if isinstance(value, list) else value
                )

    # Once, on the finished document: only then is it known how many rows a
    # table has, and so whether a fragment has exactly one row it can belong to.
    for key, value in report.extraction.items():
        report.extraction[key] = _attach_unidentified(value, key, report, top=True)

    _collect_spans(report.extraction, "", by_id, report.spans)
    # Counted once, on the result: arrays are re-deduplicated as each window
    # merges in, so counting inside _dedupe would count one row several times.
    from common.schema_sections import array_key

    report.unkeyed_rows = sum(
        1
        for section, rows in report.extraction.items()
        if isinstance(rows, list) and array_key(section, lob)
        for row in rows
        if _row_key(row, array_key(section, lob), section) is None
    )
    if report.conflicts:
        log.info("merged %d policy windows with %d conflict(s)", len(windows), len(report.conflicts))
    return report


# --------------------------------------------------------------------------
# Common-model lines (SPEC_21)
# --------------------------------------------------------------------------


#: Tables whose rows refer to units and are joined on the line's keys once every
#: reference names a merged unit. A row whose reference its window could not
#: write (the unit was read elsewhere) still joins the one row it matches.
_REFERRING_TABLES = ("coverages", "deductibles", "interested_parties", "rating_exposures",
                     "rating_modifiers", "underlying_insurance")


#: Joins one unit row into another, recording what the joined row held.
_Join = Callable[[dict[str, Any], dict[str, Any]], None]


def _drop_shared_page_rows(ordered: list[PolicyWindow], units: frozenset[str]) -> int:
    """Rows read twice because two windows of a group share a page (a split
    run overlaps, ``run_overlap_pages``), dropped from one of them; the number
    dropped.

    A row the later window holds only on the shared pages is the earlier
    window's row, read again - dropped when the earlier window holds a row
    agreeing on its values. Then a fragment the earlier window holds only on
    the shared pages (a coverage's name at the foot of the page) is dropped
    where the later window holds the row whole. Left in, the second copy has
    lost what the shared page does not show - the vehicle or boat its row
    applies to, read in the other window - and the merge cannot tell which
    unit's row it is: two boats' identical coverage lines stayed unmatched.
    Spans were taken from the windows already; dropping a row moves no other.

    Never a unit's row (``units``: a vehicle, a building): other rows of its
    window refer to it by id, and dropped it left them pointing at nothing.
    Units are joined on their own keys (:func:`_merge_units`), before this
    runs, so the links compared here name merged units.
    """
    dropped = 0
    groups: dict[str, list[PolicyWindow]] = {}
    for window in ordered:
        groups.setdefault(window.group, []).append(window)
    for windows in groups.values():
        for earlier, later in zip(windows, windows[1:], strict=False):
            shared = {int(p) for p in earlier.pages} & {int(p) for p in later.pages}
            if shared:
                dropped += (_drop_read_again(later, earlier, shared, units)
                            + _drop_read_again(earlier, later, shared, units))
    return dropped


def _drop_read_again(window: PolicyWindow, other: PolicyWindow, shared: set[int], units: frozenset[str]) -> int:
    dropped = 0
    for table, rows in list(window.extraction.items()):
        if not isinstance(rows, list) or table in units:
            continue
        others = [row for row in other.extraction.get(table) or [] if isinstance(row, dict)]
        # Compared with what the other row reads on the shared pages only: a
        # coverage of the same name printed for another vehicle on another page
        # is not this row.
        theirs = [_row_values(row, within=shared) for row in others]
        kept = []
        used: set[int] = set()
        for row in rows:
            mine = _row_values(row) if isinstance(row, dict) else None
            match = None
            if mine and _cited_within(row, shared):
                # One copy per row of the other window, and only a row whose
                # links agree - veh_3's and veh_4's identical coverage lines are
                # two rows - preferring the one that names the same units.
                fits = [(_link_fit(row, o), -i, o) for i, (o, their) in enumerate(zip(others, theirs, strict=True))
                        if id(o) not in used and not (mine - their)]
                fits = [fit for fit in fits if fit[0] is not None]
                match = max(fits, key=lambda fit: fit[:2])[2] if fits else None
            if match is not None:
                used.add(id(match))
                # The copy dropped may be the one that names the row's unit -
                # the window that shows the vehicle - which the kept copy could
                # not: its links go to the kept row.
                for key, value in row.items():
                    if match.get(key) in (None, "", []) and isinstance(value, (str, list)) and not any(
                            is_field_value(v) or isinstance(v, dict) for v in (value if isinstance(value, list) else [])):
                        match[key] = value
                dropped += 1
                continue
            kept.append(row)
        if len(kept) != len(rows):
            window.extraction[table] = kept
    return dropped


def _link_fit(row: dict[str, Any], other: dict[str, Any]) -> int | None:
    """How many links ``row`` and ``other`` both state alike; ``None`` when one
    they both state differs (they are rows of different units)."""
    same = 0
    for key, value in row.items():
        theirs = other.get(key)
        if key.endswith("_id") or not _is_link(value) or not _is_link(theirs) or value in (None, "", []) \
                or theirs in (None, "", []):
            continue                      # a row's own id is its window's numbering
        if value != theirs:
            return None
        same += 1
    return same


def _is_link(value: Any) -> bool:
    """A structural value: an id, a code or a list of them, never a printed one."""
    return isinstance(value, str) or (isinstance(value, list) and all(isinstance(v, str) for v in value))


def _row_values(row: Any, path: str = "", *, within: set[int] | None = None) -> Counter:
    """A row's printed values as ``(field path without positions, normalised
    text)`` - those citing a page of ``within``, when given."""
    out: Counter = Counter()
    if is_field_value(row):
        raw = row.get("raw") if row.get("raw") not in (None, "") else row.get("parsed")
        cited = {int(p) for p in row.get("page_ref") or [] if str(p).lstrip("-").isdigit()}
        if raw not in (None, "") and (within is None or cited & within):
            out[(path, normalize_text(str(raw)))] += 1
    elif isinstance(row, dict):
        for key, value in row.items():
            out += _row_values(value, f"{path}.{key}" if path else key, within=within)
    elif isinstance(row, list):
        for item in row:
            out += _row_values(item, f"{path}[]", within=within)
    return out


def _cited_within(node: Any, pages: set[int]) -> bool:
    """Whether every value in ``node`` cites only ``pages`` (and one at least does)."""
    refs: list[set[int]] = []

    def walk(item: Any) -> None:
        if is_field_value(item):
            refs.append({int(p) for p in item.get("page_ref") or [] if str(p).lstrip("-").isdigit()})
        elif isinstance(item, dict):
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)

    walk(node)
    return bool(refs) and all(ref and ref <= pages for ref in refs)


def _merge_common_model(windows: list[PolicyWindow], lob: str | list[str]) -> MergedPolicy:
    from common.schema_sections import group_names, references, structural_ids
    from common.structural_ids import renumber_structural_ids

    rank = {name: index for index, name in enumerate(group_names(lob))}
    ordered = sorted(windows, key=lambda w: (rank.get(w.group, len(rank)), min(w.pages or [0])))
    ids, refs = structural_ids(lob), references(lob)

    by_id: dict[int, Any] = {}
    report = MergedPolicy()
    for index, window in enumerate(ordered):
        _index_spans(window.extraction, "", window.spans, by_id)
        # Ids are numbered within a window: veh_1 of one window is not veh_1 of
        # the next. Namespaced, they cannot be mistaken for each other.
        _namespace(window.extraction, f"w{index}", ids, refs)

    # Units first, in the section map's order: a table referred to (lob_parts,
    # locations) is merged before the tables that refer to it.
    aliases: dict[str, str] = {}
    units = [table for table in ids if table != "coverages"]
    for table in units:
        rows = [row for w in ordered for row in w.extraction.get(table) or [] if isinstance(row, dict)]
        if not rows:
            continue
        for row in rows:
            _rewrite_references(row, aliases, refs)
        report.extraction[table] = _merge_units(table, rows, ids[table]["field"], lob, aliases, report)

    for window in ordered:
        _rewrite_references(window.extraction, aliases, refs)
    # With every link now naming a merged unit, a row read twice through a
    # shared page can be told from another unit's identical row.
    report.duplicates_collapsed += _drop_shared_page_rows(ordered, frozenset(units))
    for window in ordered:
        for key, value in window.extraction.items():
            if key in units:
                continue
            if key in report.extraction:
                report.extraction[key] = _merge_into(
                    report.extraction[key], value, key, report, top=True, lob=lob)
            else:
                report.extraction[key] = (
                    _dedupe(value, key, key, report, lob=lob) if isinstance(value, list) else value)

    for table in _REFERRING_TABLES:
        if isinstance(report.extraction.get(table), list):
            report.extraction[table] = _join_unreferenced(
                report.extraction[table], table, lob, refs, report)

    # One numbering for the whole document, in printed order, by the function
    # training numbers each window with. In place: spans follow the envelopes.
    _merged, numbered = renumber_structural_ids(
        report.extraction, lob, assign=("coverage_id",), in_place=True)
    report.dangling_references.extend(numbered.dangling)

    _collect_spans(report.extraction, "", by_id, report.spans)
    from common.schema_sections import array_key

    report.unkeyed_rows = sum(
        1
        for section, rows in report.extraction.items()
        if isinstance(rows, list) and array_key(section, lob)
        for row in rows
        if _row_key(row, array_key(section, lob), section) is None
    )
    return report


def _namespace(extraction: dict[str, Any], prefix: str, ids: dict[str, Any],
               refs: dict[str, tuple[str, ...]]) -> None:
    """Prefix every id and every reference in one window's answer, in place."""
    for table, spec in ids.items():
        for row in extraction.get(table) or []:
            if isinstance(row, dict) and isinstance(row.get(spec["field"]), str):
                row[spec["field"]] = f"{prefix}/{row[spec['field']]}"

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key in refs:
                    if isinstance(value, list):
                        node[key] = [f"{prefix}/{v}" if isinstance(v, str) else v for v in value]
                    elif isinstance(value, str):
                        node[key] = f"{prefix}/{value}"
                else:
                    walk(value)

    walk(extraction)


def _rewrite_references(node: Any, aliases: dict[str, str], refs: dict[str, tuple[str, ...]]) -> None:
    """Every reference rewritten to the merged row it now names, in place."""
    def final(value: Any) -> Any:
        seen = set()
        while isinstance(value, str) and value in aliases and value not in seen:
            seen.add(value)
            value = aliases[value]
        return value

    if isinstance(node, list):
        for item in node:
            _rewrite_references(item, aliases, refs)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key in refs:
                if isinstance(value, list):
                    node[key] = list(dict.fromkeys(final(v) for v in value))
                else:
                    node[key] = final(value)
            else:
                _rewrite_references(value, aliases, refs)


def _merge_units(table: str, rows: list[dict[str, Any]], id_field: str, lob: str | list[str],
                 aliases: dict[str, str], report: MergedPolicy) -> list[dict[str, Any]]:
    """One row per unit; a joined row's id becomes an alias of the kept row's.

    Two rows are one unit when the first unit key both state agrees and no
    value both state differs (:func:`_one_unit`). The rows are joined in two
    stages, so that page order never decides which unit a row joins:

    1. **On the keys that name a unit** - a VIN, a vehicle or location number,
       a building's place at its location - to a fixed point. A row that joins
       can make its unit match rows kept earlier - a VIN-only row and a
       number-only row meet once a row states both - so the grown row takes in
       every such row, until none is left.
    2. **Then on the keys that only describe one** - a year, make and model,
       which two vehicles can share - with every unit of stage 1 in view
       (:func:`_join_descriptive`). A fragment read before two 2020 Honda Civics
       sees both of them, as one read after them does.

    In either stage a row joins only when what it matches is one unit: a row
    matching two units known to differ is kept apart and flagged
    ``ambiguous_unit``. So ``#1 2020 Honda Civic`` joins the VIN-A vehicle
    numbered 1 in stage 1, and is never weighed against the VIN-B Civic it
    matches only on the model.

    A row stating no unit key is then placed where only one place fits
    (:func:`_join_keyless`), or joins the table's only keyed row, when there
    is exactly one; both flagged, as the self-contained merge does for a
    fragment. One left a unit of its own beside two or more keyed rows is
    flagged ``unplaced_fragment``.
    """
    from common.structural_ids import unit_key

    # The windows each kept row holds: its own, and every one a row joined
    # into it came from. Two rows holding one window are two rows that window
    # wrote, whatever was joined into them since.
    windows: dict[int, frozenset[str]] = {}
    for row in rows:
        window = _window_of(row, id_field)
        windows[id(row)] = frozenset({window} if window is not None else ())

    def join(kept: dict[str, Any], other: dict[str, Any]) -> None:
        windows[id(kept)] = windows[id(kept)] | windows[id(other)]
        _join_unit(kept, other, id_field, table, aliases, report, lob)

    merged: list[dict[str, Any]] = []
    for row in rows:
        candidates = _unit_candidates(table, row, merged, lob, descriptive=False)
        if not candidates:
            merged.append(row)
            if candidates is None:
                report.ambiguous_units.append(table)
            continue
        # One unit, or several rows not known to differ that this row shows to
        # be one (it states a key of each): all joined, and the grown row takes
        # in the rest.
        target = candidates[0]
        join(target, row)
        for other in candidates[1:]:
            join(target, other)
        merged[:] = [kept for kept in merged if not any(kept is other for other in candidates[1:])]
        _absorb(target, merged, table, lob, join)
    while True:
        before = len(merged)
        for target in list(merged):
            if any(kept is target for kept in merged):
                _absorb(target, merged, table, lob, join)
        if len(merged) == before:
            break

    merged = _join_descriptive(table, merged, lob, join, report)
    merged = _join_keyless(table, merged, id_field, lob, join, windows, report)

    keyed = [row for row in merged if unit_key(table, row, lob) is not None]
    if len(keyed) != 1:
        if len(keyed) > 1 and any(unit_key(table, row, lob) is None for row in merged):
            report.unplaced_fragments.append(table)
        return merged
    owner, out = keyed[0], []
    for row in merged:
        if row is not owner and unit_key(table, row, lob) is None and not _disagree(owner, row, table):
            join(owner, row)
            report.joined_unidentified.append(table)
        else:
            out.append(row)
    return out


def _one_unit(table: str, a: dict[str, Any], b: dict[str, Any], lob: str | list[str]) -> bool:
    """Whether two unit rows are known to be one unit: the first unit key both
    state agrees, and no value both state differs. A part is its line, so two
    parts of one line are one part even when their titles differ - the
    difference is a recorded conflict, never a second part."""
    from common.structural_ids import same_unit

    if not same_unit(table, a, b, lob):
        return False
    return table == "lob_parts" or not _disagree(a, b, table)


def _may_be_one(table: str, a: dict[str, Any], b: dict[str, Any], lob: str | list[str]) -> bool:
    """Whether two kept unit rows could still be one unit: no key both state
    differs, and no value both state differs."""
    from common.structural_ids import same_unit

    return same_unit(table, a, b, lob) is not False and not _disagree(a, b, table)


def _match(table: str, a: dict[str, Any], b: dict[str, Any], lob: str | list[str],
           *, descriptive: bool) -> int | None:
    """The rank of the unit key on which two rows are one unit (0 the
    strongest), when that key is of the kind asked for - one that describes a
    unit, or one that names it; ``None`` otherwise."""
    from common.structural_ids import deciding_key, descriptive_key

    if not _one_unit(table, a, b, lob):
        return None
    rank = deciding_key(table, a, b, lob)
    if rank is None or descriptive_key(table, rank, lob) != descriptive:
        return None
    return rank


def _unit_candidates(table: str, row: dict[str, Any], merged: list[dict[str, Any]], lob: str | list[str],
                     *, descriptive: bool) -> list[dict[str, Any]] | None:
    """The kept rows ``row`` is one unit with on this stage's keys, in kept
    order; ``None`` when they are units known to differ - which one it is
    cannot be known.

    A candidate matched on a weaker key than another candidate known to differ
    from it is dropped: the row states the stronger key's value, and the unit
    holding that value is not the weaker match. A row stating VIN-A and number
    1 is the VIN-A vehicle, not another vehicle numbered 1.
    """
    ranked = []
    for kept in merged:
        if kept is not row:
            rank = _match(table, kept, row, lob, descriptive=descriptive)
            if rank is not None:
                ranked.append((kept, rank))
    candidates = [kept for kept, rank in ranked
                  if not any(stronger < rank and not _may_be_one(table, other, kept, lob)
                             for other, stronger in ranked)]
    if any(not _may_be_one(table, a, b, lob) for i, a in enumerate(candidates) for b in candidates[i + 1:]):
        return None
    return candidates


def _absorb(target: dict[str, Any], merged: list[dict[str, Any]], table: str, lob: str | list[str],
            join: _Join) -> None:
    """Join into ``target``, in place in ``merged``, every other kept row it is
    now one unit with on a key that names a unit, until none is left. A row
    that is also one unit with a third kept row is not ``target``'s to take:
    which it is cannot be known."""
    def one(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return _match(table, a, b, lob, descriptive=False) is not None

    while True:
        other = next((row for row in merged if row is not target and one(target, row)
                      and not any(kept is not target and kept is not row and one(kept, row)
                                  for kept in merged)), None)
        if other is None:
            return
        join(target, other)
        merged[:] = [row for row in merged if row is not other]


def _join_descriptive(table: str, merged: list[dict[str, Any]], lob: str | list[str], join: _Join,
                      report: MergedPolicy) -> list[dict[str, Any]]:
    """Rows that are one unit only on a key that describes it (a year, make and
    model), joined with every unit in view.

    Each row's candidates are found among all the units the first stage left,
    before any is joined here, so the order rows were read in plays no part.
    Two rows join when each is the other's candidate and neither matches units
    known to differ: a 2020 Honda Civic fragment beside two 2020 Honda Civics
    joins neither, and neither of them takes it, whichever came first.
    """
    candidates = {id(row): _unit_candidates(table, row, merged, lob, descriptive=True)
                  for row in merged}
    for row in merged:
        if candidates[id(row)] is None:
            report.ambiguous_units.append(table)

    def joins(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return all(candidates[id(x)] is not None and any(c is y for c in candidates[id(x)])
                   for x, y in ((a, b), (b, a)))

    # The rows that join, as groups in kept order: each group is one unit.
    groups: list[list[dict[str, Any]]] = []
    for row in merged:
        linked = [group for group in groups if any(joins(member, row) for member in group)]
        for group in linked:
            groups.remove(group)
        groups.append([member for group in linked for member in group] + [row])
    out = list(merged)
    for group in groups:
        if len(group) < 2:
            continue
        group.sort(key=lambda r: next(i for i, kept in enumerate(merged) if kept is r))
        if any(not _may_be_one(table, a, b, lob) for i, a in enumerate(group) for b in group[i + 1:]):
            report.ambiguous_units.append(table)
            continue
        for other in group[1:]:
            join(group[0], other)
        out = [row for row in out if not any(row is other for other in group[1:])]
    return out


def _window_of(row: dict[str, Any], id_field: str) -> str | None:
    """The window a unit row came from: its id's namespace (``w2/bldg_1``)."""
    value = row.get(id_field)
    return value.split("/", 1)[0] if isinstance(value, str) and "/" in value else None


def _join_keyless(table: str, merged: list[dict[str, Any]], id_field: str, lob: str | list[str],
                  join_unit: _Join, windows: dict[int, frozenset[str]],
                  report: MergedPolicy) -> list[dict[str, Any]]:
    """Rows stating no unit key, joined where only one reading fits. Flagged.

    * Two read in different windows with the same values are one row re-read.
    * A table whose unit key holds a reference (a building is its location and
      its number): a dwelling rarely prints a building number, so the rows at
      one location join into one when none states a key, the location has no
      keyed row of the table, no two disagree, and no window wrote two of them
      - two from one window are two buildings.
    * A row naming no such reference joins the table's only other row, when
      nothing disagrees.

    Two rows from one window are never joined here: the window wrote two. A
    row's windows are every window it holds (``windows``), so a dwelling one
    window re-read from another is still that window's when the same window's
    second structure is placed.
    """
    from common.schema_sections import references, unit_keys
    from common.structural_ids import unit_key

    def keyless(row: dict[str, Any]) -> bool:
        return unit_key(table, row, lob) is None

    def join(kept: dict[str, Any], other: dict[str, Any]) -> None:
        join_unit(kept, other)
        report.joined_unidentified.append(table)
        merged[:] = [row for row in merged if row is not other]

    def apart(rows: list[dict[str, Any]]) -> bool:
        """Whether no two rows hold a window in common (a row whose window is
        unknown is taken as sharing one)."""
        held = [windows.get(id(row), frozenset()) for row in rows]
        return all(held) and len(frozenset().union(*held)) == sum(len(h) for h in held)

    def values(row: dict[str, Any]) -> Any:
        return values_view({k: v for k, v in row.items() if k != id_field})

    merged = list(merged)
    for row in [r for r in merged if keyless(r)]:
        # Into an earlier row: the unit keeps the place it was first read in.
        earlier = merged[:next(i for i, r in enumerate(merged) if r is row)]
        same = next((kept for kept in earlier if keyless(kept) and apart([kept, row])
                     and values(kept) == values(row)), None)
        if same is not None:
            join(same, row)

    refs = references(lob)
    ref_fields = list(dict.fromkeys(
        name for key in unit_keys(lob).get(table, ())
        for name in (key if isinstance(key, (list, tuple)) else (key,)) if name in refs))
    if not ref_fields:
        return merged

    def where(row: dict[str, Any]) -> tuple | None:
        place = tuple(row.get(name) for name in ref_fields)
        return place if all(isinstance(p, str) and p for p in place) else None

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in merged:
        if where(row) is not None:
            groups.setdefault(where(row), []).append(row)
    for group in groups.values():
        if (len(group) < 2 or not all(keyless(row) for row in group) or not apart(group)
                or any(_disagree(a, b, table) for i, a in enumerate(group) for b in group[i + 1:])):
            continue
        for other in group[1:]:
            join(group[0], other)

    for row in [r for r in merged if keyless(r) and where(r) is None]:
        others = [kept for kept in merged if kept is not row]
        if len(others) == 1 and apart([others[0], row]) and not _disagree(others[0], row, table):
            join(others[0], row)
    return merged


def _join_unit(kept: dict[str, Any], other: dict[str, Any], id_field: str, table: str,
               aliases: dict[str, str], report: MergedPolicy, lob: str | list[str]) -> None:
    kept_id, other_id = kept.get(id_field), other.get(id_field)
    if isinstance(other_id, str) and isinstance(kept_id, str) and other_id != kept_id:
        aliases[other_id] = kept_id
        other = {**other, id_field: kept_id}
    _merge_into(kept, other, table, report, top=False, lob=lob)
    report.duplicates_collapsed += 1


def _join_unreferenced(rows: list[Any], table: str, lob: str | list[str],
                       refs: dict[str, tuple[str, ...]], report: MergedPolicy) -> list[Any]:
    """A row that names no unit - its window did not show the unit, so wrote no
    reference - joins the one row that matches it on every other key and
    does name units. With two candidates which unit it is cannot be known, and
    it stays a row of its own."""
    from common.schema_sections import array_key

    keys = tuple(k for k in array_key(table, lob) if k not in refs)
    if not keys:
        return rows

    def referenced(row: Any) -> bool:
        return isinstance(row, dict) and any(row.get(name) for name in refs)

    out = list(rows)
    for row in [r for r in rows if isinstance(r, dict) and not referenced(r)]:
        identity = _row_key(row, keys, table)
        if identity is None:
            continue
        candidates = [other for other in out if other is not row and referenced(other)
                      and _row_key(other, keys, table) == identity and not _disagree(other, row, table)]
        if len(candidates) == 1:
            _merge_into(candidates[0], row, table, report, top=False, lob=lob)
            report.joined_unidentified.append(table)
            report.duplicates_collapsed += 1
            out = [other for other in out if other is not row]
    return out
