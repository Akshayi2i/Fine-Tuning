"""Auto-accept error: of the values a user got WITHOUT a review flag, the share
that were wrong (arch v2.1 §15.2).

This is the number the review thresholds promise. The calibrate stage measures
it on the validation half the thresholds were chosen on, where it passes almost
by construction; here it is measured on the frozen golden set, through serving,
with the release's calibrators and thresholds - the documents the promise has
never seen.

The source of truth is the golden label. A value counts once, where it was
delivered:

* **Accepted** - its envelope says ``flagged: false``. A list of envelopes (a
  set field such as ``line_of_business``) is one value, accepted only when no
  element is flagged.
* **Wrong** - it does not match the label's value at the same place, compared
  as field match compares it - with the field's declared type
  (``field_accuracy.values_agree``: a MoneyValue's ``$831.00`` is 831.0) - or
  the label holds nothing there: an invented value accepted unreviewed is the
  worst case.
* **Table rows** are matched to the label's rows by their identifiers (VIN, form
  number, claim number ... the same matching as list recall), never by
  position: a reordered row is not wrong, and a row the label does not hold is
  invented in every value. A common-model document's rows pair as its field
  match pairs them (``field_accuracy._pair_rows``): a coverage with a wrong code
  or a lost link is still the coverage it read.

Not counted: values the model left out. An omission reaches nobody as an
accepted value; recall and ``false_null_rate`` measure it.

Beside the rate, reported and never gated: **coverage** - of the values a flag
decided, the share accepted (a release that flags everything has an error rate
of nothing and a coverage of 0) - and **selective accuracy** - the share of
accepted values that were right, with its Wilson 95% lower bound - each per
field type (``calibration.features.infer_field_type``) and overall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AutoAcceptTally:
    """Accepted values and wrong accepted values, summed over documents, overall
    and per field type (``by_type``: ``{type: [flags seen, accepted, wrong]}``)."""

    flags_seen: int = 0
    accepted: int = 0
    wrong: int = 0
    by_type: dict[str, list[int]] = field(default_factory=dict)

    def add(self, other: AutoAcceptTally) -> None:
        self.flags_seen += other.flags_seen
        self.accepted += other.accepted
        self.wrong += other.wrong
        for field_type, counts in other.by_type.items():
            mine = self.by_type.setdefault(field_type, [0, 0, 0])
            for index, count in enumerate(counts):
                mine[index] += count

    def count(self, field_type: str, *, accepted: bool, wrong: bool) -> None:
        """One value a flag decided: accepted or not, and if accepted, wrong or not."""
        counts = self.by_type.setdefault(field_type, [0, 0, 0])
        self.flags_seen += 1
        counts[0] += 1
        if accepted:
            self.accepted += 1
            counts[1] += 1
            if wrong:
                self.wrong += 1
                counts[2] += 1

    @property
    def rate(self) -> float | None:
        """The share of accepted values that were wrong. ``None`` - not
        measured - when no value carried a flag (a flat extraction) or when
        nothing was accepted: with no value accepted there is no error rate
        among accepted values, and the 0.0 this used to read let a release that
        flagged everything pass its 2% ceiling with room
        (``auto_accept_coverage`` says how much was accepted)."""
        if not self.flags_seen or not self.accepted:
            return None
        return self.wrong / self.accepted

    @property
    def coverage(self) -> float | None:
        """Of the values a flag decided, the share accepted; ``None`` when no
        value carried a flag."""
        return self.accepted / self.flags_seen if self.flags_seen else None

    @property
    def selective_accuracy(self) -> float | None:
        """The share of accepted (unflagged) values that were right; ``None``
        with none accepted."""
        return (self.accepted - self.wrong) / self.accepted if self.accepted else None

    @property
    def selective_accuracy_lower(self) -> float | None:
        """The Wilson 95% lower bound of :attr:`selective_accuracy`."""
        return _wilson_lower(self.accepted - self.wrong, self.accepted)

    def coverage_by_type(self) -> dict[str, float]:
        return {t: round(c[1] / c[0], 4) for t, c in sorted(self.by_type.items()) if c[0]}

    def selective_by_type(self) -> dict[str, dict[str, Any]]:
        """Per field type: values accepted, the share right, its Wilson 95%
        lower bound. Types with nothing accepted are left out."""
        return {
            t: {"accepted": c[1], "accuracy": round((c[1] - c[2]) / c[1], 4),
                "wilson_lower_95": round(_wilson_lower(c[1] - c[2], c[1]) or 0.0, 4)}
            for t, c in sorted(self.by_type.items()) if c[1]
        }


def _wilson_lower(right: int, total: int) -> float | None:
    """The lower end of the Wilson 95% interval of ``right / total``: one minus
    the upper end of the error share's (the interval is symmetric under swapping
    the two outcomes), the bound the review thresholds are set with."""
    from evaluation.bootstrap import binomial_upper_bound

    return 1.0 - binomial_upper_bound(total - right, total) if total else None


def score_auto_accept(expected: Any, got: Any, *, common_model: bool = False) -> AutoAcceptTally:
    """Tally one document: ``expected`` the golden label (either form),
    ``got`` the served extraction with its envelopes. ``common_model``: the
    document is on a common-model line, and its rows pair as field match pairs
    them."""
    from common.canonical import values_view

    tally = AutoAcceptTally()
    _walk(values_view(expected), got, "", tally, common_model)
    return tally


def _walk(expected: Any, got: Any, path: str, tally: AutoAcceptTally, common_model: bool) -> None:
    from common.canonical import is_field_value

    if is_field_value(got):
        _count(expected, got, [got], path, tally, common_model)
    elif isinstance(got, list) and got and all(is_field_value(item) for item in got):
        _count(expected, got, got, path, tally, common_model)
    elif isinstance(got, dict):
        sub = expected if isinstance(expected, dict) else {}
        for key, value in got.items():
            _walk(sub.get(key), value, f"{path}.{key}" if path else key, tally, common_model)
    elif isinstance(got, list):
        for index, (row, match) in enumerate(_match_rows(expected, got, common_model=common_model)):
            _walk(match, row, f"{path}[{index}]", tally, common_model)


def _count(expected: Any, node: Any, envelopes: list[dict[str, Any]], path: str,
           tally: AutoAcceptTally, common_model: bool = False) -> None:
    from calibration.features import infer_field_type
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import _is_empty, values_agree

    flags = [e.get("flagged") for e in envelopes]
    if any(not isinstance(f, bool) for f in flags):
        return  # no flag, no decision to measure
    value = values_view(node)
    if _is_empty(value):
        return  # nothing was delivered
    accepted = not any(flags)
    # Compared as field match compares it: with the field's declared type, so
    # a MoneyValue's "$831.00" is the label's 831.0 (field_accuracy.values_agree).
    wrong = accepted and (_is_empty(expected) or not values_agree(expected, value, path))
    tally.count(infer_field_type(path, value, common_model=common_model), accepted=accepted, wrong=wrong)


def _match_rows(expected: Any, got_rows: list[Any], *, common_model: bool = False) -> list[tuple[Any, Any]]:
    """Each served row with the label row it reads (``None`` when the label
    holds no such row), matched on identifiers as list recall matches them -
    on a common-model line as its field match pairs them."""
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import _infer_key_fields, _pair_rows, _row_key

    expected_rows = expected if isinstance(expected, list) else []
    if common_model:
        mates, _unpaired = _pair_rows(expected_rows, got_rows)
        return list(zip(got_rows, mates, strict=True))
    keys = _infer_key_fields(expected_rows) if expected_rows else []
    pool: dict[tuple, list[Any]] = {}
    for row in expected_rows:
        pool.setdefault(_row_key(row, keys), []).append(row)
    pairs = []
    for row in got_rows:
        candidates = pool.get(_row_key(values_view(row), keys)) if expected_rows else None
        pairs.append((row, candidates.pop(0) if candidates else None))
    return pairs
