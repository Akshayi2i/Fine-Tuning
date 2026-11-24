"""Field-level accuracy, list F1, and list recall (arch §15).

Grouped in one module because they share a comparison core: every one of them
asks "does this extracted value match the golden value", and that question has
exactly one answer, supplied by ``common.normalize``. Splitting them would mean
three copies of the matching logic and three chances for them to drift.

**Normalized match, not string equality.** Insurance fields need it or you
systematically under-count correct extractions: `01/01/2026` vs `2026-01-01`,
`$1,200.00` vs `1200.0`, `Acme Mfg LLC` vs `ACME MANUFACTURING LLC`.

**List recall is separate from list F1 on purpose.** F1 over matched rows can
look healthy while whole rows are missing, and a missing row generates no tokens,
so per-field confidence is blind to it (arch §5). Recall is the metric that sees
a dropped Loss Run claim.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from common.normalize import values_match
from contracts.rows import (  # noqa: F401 - moved (review A1)
    _COMPANIONS,
    _CONTAINER_TYPES,
    _REFERENCE_COMPANIONS,
    ROW_IDENTIFIERS,
    _filled_share,
    _holds_references,
    _infer_key_fields,
    _key_text,
    flatten_scalars,
)

#: The normalizer for a field type the common model declares; an enum or an
#: undeclared field is left to the field's name.
_KIND_BY_TYPE = {"money": "currency", "number": "number", "date": "date", "identifier": "identifier"}


def scoring_kind(field_path: str) -> str | None:
    """How a scored field is compared, by the type its schema declares: a
    MoneyValue as money (``$831.00`` is 831.0), a YearValue or a count as a
    number. ``None`` leaves it to the name (``common.normalize.infer_field_kind``),
    which reads ``premium.total`` or ``year_built`` as text.

    Scoring only: the serving merge keeps its own reading of what a field is.
    Rows are written either way (``coverages[2]``, ``coverages[coverage_code=x]``).
    """
    from calibration.features import common_model_field_types

    declared = common_model_field_types().get(re.sub(r"\[[^\]]*\]", "", field_path))
    return _KIND_BY_TYPE.get(declared) if declared else None


def values_agree(expected: Any, got: Any, field_path: str) -> bool:
    """:func:`common.normalize.values_match` with the field's declared type."""
    return values_match(expected, got, field_path=field_path, kind=scoring_kind(field_path))


@dataclass
class FieldResult:
    """One field's outcome, kept per-field so failures are attributable."""

    field_path: str
    expected: Any
    got: Any
    correct: bool
    exact: bool


@dataclass
class AccuracyReport:
    """Scalar-field accuracy across a document or a corpus."""

    results: list[FieldResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def exact_match(self) -> float:
        return sum(r.exact for r in self.results) / self.total if self.total else 0.0

    @property
    def normalized_match(self) -> float:
        return sum(r.correct for r in self.results) / self.total if self.total else 0.0

    def failures(self) -> list[FieldResult]:
        return [r for r in self.results if not r.correct]

    def by_field(self) -> dict[str, float]:
        """Per-field accuracy — an aggregate hides which field is failing."""
        buckets: dict[str, list[bool]] = {}
        for result in self.results:
            buckets.setdefault(result.field_path, []).append(result.correct)
        return {path: sum(v) / len(v) for path, v in sorted(buckets.items())}


def score_fields(
    expected: dict[str, Any],
    got: dict[str, Any],
    *,
    skip_lists: bool = True,
) -> AccuracyReport:
    """Score scalar fields.

    List rows are excluded by default and scored by :func:`score_list_field`
    instead — mixing them would let a 40-row Loss Run dominate the document's
    accuracy over its half-dozen header fields.

    **A value the golden label does not state is scored as wrong.** Canonical
    targets are sparse — a field the document does not state is omitted — so a
    loop over the golden paths alone never saw an invented value: a checkpoint
    that hallucinated fields lost nothing, and checkpoint selection could prefer
    it. Flat schemas emit every key, so for them this changes nothing.
    """
    expected_flat = flatten_scalars(expected)
    got_flat = flatten_scalars(got)

    report = AccuracyReport()
    for path, expected_value in sorted(expected_flat.items()):
        if skip_lists and "[" in path:
            continue
        got_value = got_flat.get(path) if path in got_flat else _rows_at(got_flat, path)
        correct = values_agree(expected_value, got_value, path)
        report.results.append(
            FieldResult(
                field_path=path,
                expected=expected_value,
                got=got_value,
                correct=correct,
                # Never above the normalised match: 957 == 957.0 in Python.
                exact=correct and expected_value == got_value,
            )
        )
    for path, got_value in sorted(got_flat.items()):
        if path in expected_flat or (skip_lists and "[" in path) or _is_empty(got_value):
            continue
        report.results.append(
            FieldResult(field_path=path, expected=None, got=got_value, correct=False, exact=False)
        )
    return report


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _rows_at(flat: dict[str, Any], path: str) -> list[dict[str, Any]] | None:
    """The table a flattened extraction holds at ``path``, or ``None``.

    An empty expected table (``claims: []``) flattens to one field, while a
    populated one flattens to ``claims[0].paid`` ... with no ``claims`` key at all.
    Looking the field up by path alone therefore found ``None``, and ``[]`` vs
    ``None`` matches — so a model that invented claims for a document with none
    scored that field correct. Rebuilding the rows makes it compare as a
    non-empty table, which does not match an empty one.
    """
    marker = f"{path}["
    rows: dict[int, dict[str, Any]] = {}
    for key, value in flat.items():
        if not key.startswith(marker):
            continue
        index_text, _, rest = key[len(marker):].partition("]")
        if not index_text.isdigit():
            continue
        rows.setdefault(int(index_text), {})[rest.lstrip(".") or "value"] = value
    return [rows[i] for i in sorted(rows)] if rows else None


# --------------------------------------------------------------------------
# List fields
# --------------------------------------------------------------------------

@dataclass
class ListFieldReport:
    """Precision, recall and F1 for one repeating structure."""

    field_name: str
    expected_rows: int
    got_rows: int
    matched_rows: int
    row_field_accuracy: float = 0.0

    @property
    def precision(self) -> float:
        return self.matched_rows / self.got_rows if self.got_rows else 0.0

    @property
    def recall(self) -> float:
        """The metric that sees a dropped row.

        A missing row produces no tokens, so per-field confidence cannot flag it
        — recall is the only signal that catches it (arch §5).
        """
        return self.matched_rows / self.expected_rows if self.expected_rows else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def missed_rows(self) -> int:
        return max(0, self.expected_rows - self.matched_rows)


def _row_key(row: Any, key_fields: list[str]) -> tuple:
    """A row's identity. Non-object rows key on their own value.

    Calling ``.get`` unconditionally raised ``AttributeError`` on a list of
    scalars — a schema-invalid but perfectly possible generation, and scoring
    runs before schema validation — which destroyed every other document's score
    in the same run.
    """
    if not isinstance(row, dict):
        return ("scalar:", str(row).strip().casefold())
    return tuple(_key_text(row.get(f)) for f in key_fields)


def score_list_field(
    expected_rows: list[Any],
    got_rows: list[Any],
    field_name: str,
    *,
    key_fields: list[str] | None = None,
    common_model: bool = False,
) -> ListFieldReport:
    """Match rows by identity, then score.

    Rows are matched on an identifying subset (a claim number, say) rather than
    by position, because a model that drops row 2 would otherwise mis-align every
    row after it and score near zero for one omission.

    ``common_model``: the rows are a common-model document's, and pair as its
    field match pairs them (:func:`_pair_rows`). Keyed on the code and the link,
    a coverage whose link was lost between windows, or that got a sibling's
    code, counted as a missed row and an invented one, though field match
    reads the same row as one wrong value.
    """
    if common_model and key_fields is None:
        return _score_paired_rows(expected_rows, got_rows, field_name)
    keys = key_fields or _infer_key_fields(expected_rows)
    # Keys are counted, not deduplicated. Indexing rows into a dict collapsed
    # duplicates, so two claim rows sharing a claim number scored recall 0.5 on a
    # byte-perfect extraction — and if a key field was null in every row they all
    # collapsed to one. Both are ordinary in real Loss Runs.
    expected_index = _index_rows(expected_rows, keys)
    got_index = _index_rows(got_rows, keys)

    # Matched by multiplicity: three expected rows sharing a key and two
    # extracted ones count as two matches, not one. Counting distinct keys made
    # a perfect extraction of a duplicated row look like a 50% recall failure.
    matched_rows = 0
    field_scores: list[float] = []
    for key in sorted(set(expected_index) & set(got_index)):
        expected_group, got_group = expected_index[key], got_index[key]
        matched_rows += min(len(expected_group), len(got_group))
        for expected_row, got_row in zip(expected_group, got_group, strict=False):
            if not isinstance(expected_row, dict) or not isinstance(got_row, dict):
                field_scores.append(1.0 if expected_row == got_row else 0.0)
                continue
            report = score_fields(expected_row, got_row, skip_lists=False)
            if report.total:
                field_scores.append(report.normalized_match)

    return ListFieldReport(
        field_name=field_name,
        expected_rows=len(expected_rows),
        got_rows=len(got_rows),
        matched_rows=matched_rows,
        row_field_accuracy=sum(field_scores) / len(field_scores) if field_scores else 0.0,
    )


def _score_paired_rows(expected_rows: list[Any], got_rows: list[Any], field_name: str) -> ListFieldReport:
    """:func:`score_list_field` for a common-model table: rows paired by
    :func:`_pair_rows`."""
    mates, _unpaired = _pair_rows(expected_rows, got_rows)
    field_scores: list[float] = []
    for got_row, mate in zip(got_rows, mates, strict=True):
        if mate is None:
            continue
        if not isinstance(mate, dict) or not isinstance(got_row, dict):
            field_scores.append(1.0 if mate == got_row else 0.0)
            continue
        report = score_fields(mate, got_row, skip_lists=False)
        if report.total:
            field_scores.append(report.normalized_match)
    return ListFieldReport(
        field_name=field_name,
        expected_rows=len(expected_rows),
        got_rows=len(got_rows),
        matched_rows=sum(mate is not None for mate in mates),
        row_field_accuracy=sum(field_scores) / len(field_scores) if field_scores else 0.0,
    )


def _index_rows(rows: list[Any], keys: list[str]) -> dict[tuple, list[Any]]:
    """Group rows by identity, keeping every row that shares a key."""
    grouped: dict[tuple, list[Any]] = {}
    for row in rows:
        grouped.setdefault(_row_key(row, keys), []).append(row)
    return grouped


def rows_aligned_to(expected: Any, got: Any) -> Any:
    """``expected`` with every table's rows in the order ``got`` holds them.

    Rows are paired by identity (:func:`_infer_key_fields`, :func:`_row_key`),
    as list recall pairs them, so ``vehicles[2].vin`` in the result is the label
    row the model's ``vehicles[2]`` reads. A model row the label does not hold
    pairs with ``{}``; a label row the model left out has no place and is
    dropped. For comparing by path: by position, one omitted or reordered row
    shifted every row after it and every correct value in them read as wrong.
    Both sides as values (``values_view``).
    """
    if isinstance(got, dict):
        source = expected if isinstance(expected, dict) else {}
        aligned = {key: rows_aligned_to(source.get(key), value) for key, value in got.items()}
        return {**source, **aligned}
    if isinstance(got, list) and got and all(isinstance(row, dict) for row in got):
        rows = expected if isinstance(expected, list) else []
        keys = _infer_key_fields(rows) if rows else []
        pool = _index_rows(rows, keys)
        out = []
        for row in got:
            candidates = pool.get(_row_key(row, keys)) if rows else None
            out.append(rows_aligned_to(candidates.pop(0) if candidates else {}, row))
        return out
    return expected


def find_list_fields(obj: dict[str, Any], prefix: str = "") -> dict[str, list[dict[str, Any]]]:
    """Every repeating structure in a document, nested ones included.

    Keyed by dotted path (``auto.vehicles``, ``premium.taxes_and_fees``). A
    canonical policy keeps most of its tables inside sections, and a top-level
    scan found none of them — they were never scored at all.
    """
    found: dict[str, list[dict[str, Any]]] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, list) and value and isinstance(value[0], dict):
            found[path] = value
        elif isinstance(value, dict):
            found.update(find_list_fields(value, f"{path}."))
    return found


def _at(obj: dict[str, Any], dotted: str) -> Any:
    node: Any = obj
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def score_all_list_fields(
    expected: dict[str, Any], got: dict[str, Any], *, common_model: bool = False
) -> dict[str, ListFieldReport]:
    """Score every list field. Fields absent from the output score zero recall,
    which is the correct reading: the rows were expected and not produced.

    Both sides are compared on VALUES (``values_view``). A canonical row is a
    dict of envelopes, and keying rows on the envelope made two readings of one
    claim match only when raw text and page_ref were byte-identical — and never
    against a stored golden, which also carries confidence and flagged — while
    unkeyed tables matched on nothing, so any rows counted as the right rows.

    ``common_model``: a common-model document's rows pair as its field match
    pairs them (:func:`score_list_field`).
    """
    from common.canonical import values_view

    expected, got = values_view(expected), values_view(got)
    return {
        name: score_list_field(rows, _at(got, name) or [], name, common_model=common_model)
        for name, rows in find_list_fields(expected).items()
    }


def aligned_for_scoring(expected: Any, got: Any, *, fill_unpaired: bool = False) -> tuple[Any, Any]:
    """``(expected, got)`` with every table's rows paired by identity at one index.

    Like :func:`rows_aligned_to`, but for scoring values INSIDE rows, so it keeps
    what that drops: an expected row the answer left out goes after the answer's
    rows, where its values meet nothing and count as misses, and an answer row
    the label does not hold faces an empty row, where its values count as
    invented. Nested tables (a coverage's limits) are paired the same way.
    Rows pair as their values do (:func:`_pair_rows`), so two documents with
    their envelopes pair exactly as their values views would.

    ``fill_unpaired``: after that pairing, each answer row nothing paired takes
    the next label row nothing paired, in table order, and only the label rows
    left after that go at the end. For the metrics that ask whether a value was
    WRITTEN where the label has one (false nulls): a row the model wrote but
    misread is a wrong value there, never a null. Field match keeps the strict
    pairing, where such a row is one missed row and one invented one.
    """
    if isinstance(expected, dict) or isinstance(got, dict):
        left = expected if isinstance(expected, dict) else {}
        right = got if isinstance(got, dict) else {}
        out_left: dict[str, Any] = {}
        out_right: dict[str, Any] = {}
        for key in dict.fromkeys([*left, *right]):
            paired_left, paired_right = aligned_for_scoring(
                left.get(key), right.get(key), fill_unpaired=fill_unpaired)
            if key in left:
                out_left[key] = paired_left
            if key in right:
                out_right[key] = paired_right
        return out_left, out_right
    if _is_table(expected) or _is_table(got):
        rows = expected if isinstance(expected, list) else []
        answer = got if isinstance(got, list) else []
        mates, unpaired = _pair_rows(rows, answer)
        if fill_unpaired:
            free = iter(unpaired)
            mates = [mate if mate is not None else next(free, None) for mate in mates]
            unpaired = list(free)
        paired_left, paired_right = [], []
        for row, mate in zip(answer, mates, strict=True):
            a, b = aligned_for_scoring(mate if mate is not None else {}, row,
                                       fill_unpaired=fill_unpaired)
            paired_left.append(a)
            paired_right.append(b)
        for row in unpaired:
            a, _ = aligned_for_scoring(row, {}, fill_unpaired=fill_unpaired)
            paired_left.append(a)
        return paired_left, paired_right
    return expected, got


#: What a coverage prints: what it applies to and its name - where it pairs
#: before its code is trusted against a name that disagrees, and the identity
#: ``coverage_code_accuracy`` pairs on (``common_model._codes``). The code is a
#: choice the model makes from a list; one wrong pick is one wrong value, not
#: a missed row.
_PRINTED_COVERAGE_KEYS: tuple[str, ...] = ("applies_to", "coverage_name")

#: One pass of row pairing: the fields both rows must agree on, and a test of
#: an (answer row, label row) pair, as values, that must hold as well.
_Pass = tuple[list[str], Any]


def _pair_rows(rows: list[Any], answer: list[Any]) -> tuple[list[Any], list[Any]]:
    """Each answer row's label row (``None`` where none pairs), and the label
    rows nothing paired with, in table order. Only a common-model document's
    rows are paired here (field match, false nulls, auto-accept, list metrics).

    Rows pair on the table's identifiers (:func:`_infer_key_fields`), as values
    (``values_view``), each label row once, in passes from the strictest
    identity down; each pass runs over every answer row before the next, so a
    row that would pair on a weaker identity cannot take a row a stronger one
    claims. A row that misses on the whole key - a unit whose link the answer
    lost - still pairs on the identifier alone: the link is scored on its own
    (reference accuracy), not again here.

    A coverage is known by the name its page prints before the code, which is a
    choice the model makes from a list: a wrong code is most often a sibling
    coverage's, and on the code it would take that sibling's row. So the code
    pairs first only where the printed names agree, then the printed identity
    (:data:`_PRINTED_COVERAGE_KEYS`), then the name alone - a link lost between
    windows - and only then the code whatever the names say (a misread name).

    Where several label rows qualify, the one sharing the most values with the
    answer row is taken (premium, limits, deductibles ...), then the one nearest
    in the table: two Collision rows with their links lost are told apart by
    their premiums and deductibles, never by which comes first. Last, a row
    still unpaired takes the unpaired label row it shares the most values
    with, if that is at least half of that row's values: a misread VIN or
    limit amount then costs that value, not the row.
    """
    from common.canonical import values_view

    keys = _infer_key_fields(values_view(rows or answer))
    passes: list[_Pass] = [(keys, None)]
    if keys[:1] == ["coverage_code"]:
        passes = [
            (keys, _names_agree),
            (list(_PRINTED_COVERAGE_KEYS), _both_named),
            (["coverage_name"], _both_named),
            (keys, None),
            (["coverage_code"], None),
        ]
    elif len(keys) > 1:
        passes.append((keys[:1], None))
    return _pair_in_passes(rows, answer, passes, final=True)


def pair_coverages_by_name(rows: list[Any], answer: list[Any]) -> list[Any]:
    """Each answer coverage's label coverage by what its page prints alone:
    what it applies to and its name, then its name - never its code, which is
    what ``coverage_code_accuracy`` measures. A coverage that prints no name
    pairs with nothing (``None``)."""
    passes: list[_Pass] = [(list(_PRINTED_COVERAGE_KEYS), _both_named),
                           (["coverage_name"], _both_named)]
    return _pair_in_passes(rows, answer, passes, final=False)[0]


def _pair_in_passes(rows: list[Any], answer: list[Any], passes: list[_Pass], *,
                    final: bool) -> tuple[list[Any], list[Any]]:
    """Pair ``answer`` with ``rows`` pass by pass (:func:`_pair_rows`). Within a
    pass the pairs that share the most values go first, then the nearest in
    the table; ``final`` adds the last pass on shared values alone."""
    from common.canonical import values_view

    label_views = [values_view(row) for row in rows]
    answer_views = [values_view(row) for row in answer]
    label_values = [_stated_values(view) for view in label_views]
    answer_values = [_stated_values(view) for view in answer_views]
    mates: list[int | None] = [None] * len(answer)
    free = set(range(len(rows)))

    def assign(candidates: list[tuple[int, int, int, int]]) -> None:
        # (shared values, distance, answer position, label index), best first.
        for _shared, _distance, position, label in sorted(candidates, key=lambda c: (-c[0], *c[1:])):
            if mates[position] is None and label in free:
                mates[position] = label
                free.discard(label)

    for fields, qualifies in passes:
        candidates = []
        for position, view in enumerate(answer_views):
            if mates[position] is not None:
                continue
            key = _row_key(view, fields)
            for label in sorted(free):
                if _row_key(label_views[label], fields) != key:
                    continue
                if qualifies is not None and not qualifies(view, label_views[label]):
                    continue
                shared = sum((answer_values[position] & label_values[label]).values())
                candidates.append((shared, abs(label - position), position, label))
        assign(candidates)
    if final:
        scalars = {label: _stated_values(label_views[label], scalars_only=True) for label in free}
        candidates = []
        for position in range(len(answer)):
            if mates[position] is not None:
                continue
            mine = _stated_values(answer_views[position], scalars_only=True)
            for label in sorted(free):
                theirs = scalars[label]
                shared = sum((mine & theirs).values())
                if shared and 2 * shared >= sum(theirs.values()):
                    candidates.append((shared, abs(label - position), position, label))
        assign(candidates)
    return ([rows[m] if m is not None else None for m in mates],
            [row for index, row in enumerate(rows) if index in free])


def _stated_values(view: Any, *, scalars_only: bool = False) -> Counter:
    """A row's stated values, nested rows included, as ``(field, value)`` counts
    with row positions left out - so a coverage's limits in another order are
    the same values. ``scalars_only`` leaves out a list of values (a link's
    references): what a row applies to alone is no identity."""
    import re

    out: Counter = Counter()
    if not isinstance(view, dict):
        return out
    for path, value in flatten_scalars(view).items():
        if isinstance(value, list) and scalars_only:
            continue
        field_name = re.sub(r"\[\d+\]", "[]", path)
        for item in value if isinstance(value, list) else [value]:
            text = _key_text(item)
            if text:
                out[(field_name, text)] += 1
    return out


def _both_named(row: Any, label: Any) -> bool:
    """Whether both coverage rows print a name: without one, what a coverage
    applies to alone is no identity, and two unnamed coverages of a unit are
    not paired."""
    return _names_coverage(row) and _names_coverage(label)


def _names_agree(row: Any, label: Any) -> bool:
    """Whether two coverage rows print the same name, or either prints none."""
    if not (_names_coverage(row) and _names_coverage(label)):
        return True
    return _key_text(row.get("coverage_name")) == _key_text(label.get("coverage_name"))


def _names_coverage(row: Any) -> bool:
    """Whether a coverage row prints a name."""
    from common.canonical import values_view

    return isinstance(row, dict) and bool(_key_text(values_view(row.get("coverage_name"))))


def _is_table(node: Any) -> bool:
    return isinstance(node, list) and bool(node) and all(isinstance(row, dict) for row in node)
