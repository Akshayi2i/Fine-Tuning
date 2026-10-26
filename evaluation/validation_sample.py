"""The fixed validation sample: what in-training checks and checkpoint selection read.

The full validation split is ~3,300 rows on the delivered corpus. Measured on the
smoke run, one pass over it costs about an hour of evaluation loss and ~1.5 h of
generation per checkpoint, and at an evaluation every 50 steps the run spent
50-60 h validating - longer than training. Neither use needs every row: the
in-training checks only show a trend, and checkpoint selection only ranks a few
candidates against each other, on the same rows.

The sample is:

- **stratified** by line of business and reading mode, allocated in proportion
  with at least one row per group, so no line or regime drops out;
- **deterministic**: chosen by a stable hash of each row's identity, so the
  training checks and checkpoint selection - in different processes, on rows in
  different formats - pick the same rows, and a rerun picks them again;
- the whole split when it is no larger than the sample.

Three options, all off by default (``evaluation`` in the training config;
:func:`sample_options`), each changing which rows are chosen:

- ``validation_sample_by_empty_target``: stratified by whether the row's target
  states anything, too - a window whose answer is empty sections teaches and
  tests "say nothing", and a sample short of them never sees that;
- ``validation_rows_per_family_max``: at most that many rows of one family (a
  seed and its twins, a document's windows in every mode), so one long policy
  read in many windows does not fill the sample;
- ``validation_sample_mode_shares``: the reading modes weighted to these shares
  instead of to the split's (one row per mode per window, a third each).
  ``training`` takes the training mix (``modality_mix``): under the fast
  profile, image_only at 0.7, the mode production reads. A mode with no rows
  gives its share to the others.

What ships is still gated on the full frozen test set; and a near-tie between
the top two candidates is broken on the full validation split (checkpoint_eval).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Rows in the sample, unless configured otherwise.
SAMPLE_ROWS = 400


@dataclass(frozen=True)
class SampleOptions:
    """How the sample is drawn beyond line x mode (all off: as before them)."""

    by_empty_target: bool = False
    rows_per_family_max: int | None = None
    #: mode -> share of the sample; None: each mode's share of the split.
    mode_shares: dict[str, float] | None = None

    @property
    def active(self) -> bool:
        return self.by_empty_target or self.rows_per_family_max is not None or self.mode_shares is not None


def sample_options(evaluation: dict[str, Any] | None = None) -> SampleOptions:
    """The options configured in ``evaluation`` (default: the adapter's training config's)."""
    if evaluation is None:
        from common.adapter import ADAPTER
        from common.config import training_config

        evaluation = training_config(ADAPTER.training_config).get("evaluation") or {}
    cap = evaluation.get("validation_rows_per_family_max")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError(f"evaluation.validation_rows_per_family_max must be a whole number of at least 1, "
                         f"or null; got {cap!r}")
    shares = evaluation.get("validation_sample_mode_shares")
    if shares == "training":
        from training.data_mix import configured_modality_mix

        shares = configured_modality_mix()
    if shares is not None:
        if not isinstance(shares, dict) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in shares.values()) \
                or not sum(shares.values()) > 0:
            raise ValueError("evaluation.validation_sample_mode_shares must be null, 'training' or a mapping "
                             f"of reading modes to non-negative shares; got {shares!r}")
        shares = {str(mode): float(v) for mode, v in shares.items()}
    return SampleOptions(bool(evaluation.get("validation_sample_by_empty_target")), cap, shares)


def _stratum(row: dict[str, Any]) -> tuple[str, str]:
    lob = row.get("lob")
    lob = ",".join(sorted(map(str, lob))) if isinstance(lob, list) else str(lob or "unknown")
    return lob, str(row.get("modality_mode") or "unknown")


def target_state(row: dict[str, Any]) -> str:
    """``empty`` when the row's target states no value, ``stated`` when it
    states one, ``unknown`` when the row carries no readable target."""
    messages = row.get("messages") or []
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        return "unknown"
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    try:
        target = json.loads(content or "")
    except (TypeError, ValueError):
        return "unknown"
    return "stated" if _states(target) else "empty"


def _states(node: Any) -> bool:
    """Whether an answer states any value: a compact or full envelope with a
    value, or a plain value that is not empty."""
    if isinstance(node, dict):
        if "v" in node or "parsed" in node:                     # an envelope
            value = node.get("v", node.get("parsed"))
            return _states(value) if isinstance(value, (dict, list)) else value not in (None, "")
        return any(_states(value) for value in node.values())
    if isinstance(node, list):
        return any(_states(value) for value in node)
    return node not in (None, "")


def _family(row: dict[str, Any]) -> str:
    return str(row.get("group_id") or row.get("source_id"))


def _identity(row: dict[str, Any]) -> str:
    """What makes a validation row itself, independent of its file position."""
    key = [row.get("source_id"), row.get("sections"), row.get("modality_mode"),
           row.get("window_pages"), row.get("doc_type")]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def validation_sample(rows: Sequence[dict[str, Any]], size: int = SAMPLE_ROWS,
                      options: SampleOptions | None = None) -> list[dict[str, Any]]:
    """The sample of ``rows``, in their original order.

    ``options`` default to the configured ones (:func:`sample_options`), so
    the training checks and checkpoint selection, in different processes, draw
    the same sample."""
    rows = list(rows)
    if size <= 0 or len(rows) <= size:
        return rows
    options = sample_options() if options is None else options
    if options.active:
        return _sample_with(rows, size, options)

    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(_stratum(row), []).append(index)

    # Proportional, at least one per group, largest remainders take what is left.
    exact = {g: len(members) * size / len(rows) for g, members in groups.items()}
    quota = {g: max(1, int(share)) for g, share in exact.items()}
    spare = size - sum(quota.values())
    for group in sorted(exact, key=lambda g: (-(exact[g] - int(exact[g])), g)):
        if spare <= 0:
            break
        if quota[group] < len(groups[group]):
            quota[group] += 1
            spare -= 1

    chosen: list[int] = []
    for group, members in groups.items():
        ranked = sorted(members, key=lambda i: _identity(rows[i]))
        chosen.extend(ranked[: min(quota[group], len(members))])
    return [rows[i] for i in sorted(chosen)]


def _sample_with(rows: list[dict[str, Any]], size: int, options: SampleOptions) -> list[dict[str, Any]]:
    """:func:`validation_sample` with the options on.

    Strata are line x mode, and empty or stated target when asked. Each
    stratum's quota is proportional - to its share of its mode's rows times the
    mode's configured share, with ``mode_shares`` - with at least one per
    stratum and the largest remainders taking what is left. Rows are taken in
    one deterministic order (the hash of each row's identity) while their
    stratum is under quota and their family under the cap; what the cap left
    unfilled is then taken in the same order from any stratum.
    """
    def stratum(row: dict[str, Any]) -> tuple[str, ...]:
        base = _stratum(row)
        return (*base, target_state(row)) if options.by_empty_target else base

    groups: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(stratum(row), []).append(index)

    if options.mode_shares is None:
        exact = {g: len(members) * size / len(rows) for g, members in groups.items()}
    else:
        by_mode: dict[str, int] = {}
        for group, members in groups.items():
            by_mode[group[1]] = by_mode.get(group[1], 0) + len(members)
        # Modes with rows share the sample by their configured share; a mode
        # the configuration does not name, or one with no rows, takes nothing
        # of it (its strata keep the one row each below).
        weights = {mode: options.mode_shares.get(mode, 0.0) for mode in by_mode}
        total = sum(weights.values())
        if total <= 0:
            weights, total = {mode: float(n) for mode, n in by_mode.items()}, float(len(rows))
        exact = {g: size * weights[g[1]] / total * len(members) / by_mode[g[1]] for g, members in groups.items()}
    quota = {g: min(len(groups[g]), max(1, int(share))) for g, share in exact.items()}
    spare = size - sum(quota.values())
    while spare > 0:
        open_groups = [g for g in groups if quota[g] < len(groups[g])]
        if not open_groups:
            break
        for group in sorted(open_groups, key=lambda g: (-(exact[g] - quota[g]), g)):
            if spare <= 0:
                break
            quota[group] += 1
            spare -= 1

    order = sorted(range(len(rows)), key=lambda i: (_identity(rows[i]), i))
    of_group = {i: g for g, members in groups.items() for i in members}
    cap = options.rows_per_family_max
    taken: dict[tuple[str, ...], int] = {}
    per_family: dict[str, int] = {}
    chosen: set[int] = set()

    def take(index: int) -> None:
        chosen.add(index)
        taken[of_group[index]] = taken.get(of_group[index], 0) + 1
        family = _family(rows[index])
        per_family[family] = per_family.get(family, 0) + 1

    def room(index: int) -> bool:
        return cap is None or per_family.get(_family(rows[index]), 0) < cap

    for index in order:
        if taken.get(of_group[index], 0) < quota[of_group[index]] and room(index):
            take(index)
    for index in order:
        if len(chosen) >= size:
            break
        if index not in chosen and room(index):
            take(index)
    return [rows[i] for i in sorted(chosen)]
