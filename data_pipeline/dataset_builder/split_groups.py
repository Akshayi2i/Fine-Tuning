"""Train/val/test splitting at **group** level, before any expansion (arch v2.1 §8.2).

This is the single most consequential ordering rule in the corpus pipeline, and
v2.1 moves the unit of splitting from the document to the family.

**Why not source documents.** v1 split at ``source_id``, which stopped a
document's own modality variants spanning the split. It did nothing about
families: the same carrier's template, the same account renewed yearly, one
agency's four hundred near-identical certificates. Put one member in ``train``
and another in ``test`` and the eval number measures template memorisation. The
failure is invisible — nothing crashes, the metrics simply come back better than
the model deserves.

So: assign **groups** to splits first, then expand every document, task, window
and modality mode of a group inside whichever split it landed in.

Three things this adds beyond the group unit:

* **A held-out-carrier slice.** One carrier per line of business goes entirely
  to test once the line names ten or more carriers (fewer: every carrier
  trains), because "unseen template" is a different question from "unseen
  document" and only a held-out carrier answers it.
* **Validation halves.** Validation is split by group into a calibration half
  (fits calibrators, §5.3) and a threshold half (sets review thresholds, §5.4).
  Fitting both on the same data makes every threshold optimistic — the
  calibrator has already seen the errors the threshold is meant to price.
* **Synthetic documents are train-only.** A synthetic ACORD has perfect labels
  because it was generated from them; scoring against it measures the generator.

**The ratio** comes from the document type's volume (``common.constants.
SPLIT_RATIOS_BY_VOLUME``: 70/18/12 below 200 documents, 75/15/10 below 1000,
80/10/10 from 1000). It is per TYPE because the gate's metrics are per type: the
band answers "are 10% of this type's documents enough to measure it".

**Placement is per line of business** (a policy's ``lob``). Every group gets a
stable hash position; below the train edge it trains, below the val edge it is
validation, above it is test — the same rule in every line, so each line lands
near the type's ratio. Two per-line rules on top:

* a line with fewer than :data:`MIN_DOCS_TO_MEASURE_LINE` documents trains on
  everything it has. One or two documents cannot measure a line, and holding
  them out would leave the model with no example of it at all;
* a line with enough groups is repaired to have at least one val and one test
  group, so no measured line is absent from evaluation by the luck of the hash.

**On assignment stability.** Groups are assigned by hash *threshold*, not slice
position, so at a fixed ratio the corpus grows without reassigning anything.
Crossing a volume band changes the edges, and :func:`assert_bands_only_grow_train`
guarantees both edges only ever move UP (train edge 0.70 -> 0.75 -> 0.80, val
edge 0.88 -> 0.90 -> 0.90). So a band crossing only moves groups toward train —
test -> val, val -> train — and never moves a document a model trained on into
val or test. Two small-N rules can still move a group the other way, and both
say so in the log: the empty-split repair, and a line crossing
:data:`MIN_DOCS_TO_MEASURE_LINE`. Cross-version comparability comes from the
frozen golden eval set (§15.4), which no reassignment touches.

**A cycle that continues an adapter** (decision 2026-10-23) cannot afford even
those two moves: the continued adapter has memorised every document its chain
of runs trained on, so one of them in val would be scored as unseen - in
checkpoint selection, and in the calibration and threshold fits that read val
- and every one of those numbers would come out optimistic. Such a build passes
the documents the chain trained on (``trained``): their families are pinned to
train whatever the rules above would draw, ahead of the carrier hold-out and
the repair, and each is recorded with the split it would otherwise have taken
(``pinned_to_train``).
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal

from common.constants import SplitRatio, split_ratio_for

# A placeholder carrier ("N/A", "Unknown", "Not stated", blank) is no carrier:
# on a policy it never counts toward a line's carriers and is never the one
# held out. Read on the name's own words ("Other Mutual" names a carrier), and
# defined beside normalize_carrier, whose key must never turn a carrier's name
# into a placeholder.
from common.normalize import PLACEHOLDER_CARRIERS, is_placeholder_carrier  # noqa: F401

log = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")

ValHalf = Literal["calibration", "threshold"]

#: Below this many documents a line of business is not measured: all of it trains.
MIN_DOCS_TO_MEASURE_LINE = 5

#: A line needs this many groups before the repair forces a val and a test group:
#: with fewer, the repair would hand one line's whole evaluation to one family.
MIN_GROUPS_TO_REPAIR = 3

#: A line of business holds one carrier out of train and validation only when
#: its real documents name at least this many carriers (decision 2026-10-19):
#: below it, every carrier trains - one carrier of a few is too large a share
#: of the line's layouts to give up.
MIN_CARRIERS_PER_LINE_FOR_HELD_OUT = 10


class SplitError(RuntimeError):
    """Raised when a split assignment would leak or is otherwise unusable."""


@dataclass
class GroupSplitAssignment:
    """A deterministic ``group_id -> split`` mapping, and how it was made."""

    assignment: dict[str, Split] = field(default_factory=dict)

    #: Which half of validation a group is in. Empty for non-validation groups.
    val_half: dict[str, ValHalf] = field(default_factory=dict)

    #: Carriers placed entirely in test, per doc type.
    held_out_carriers: dict[str, list[str]] = field(default_factory=dict)
    #: Per doc type, the carrier held out of train and val for each line
    #: (Fideon SPEC_09 amendment item 4), and the lines with only one carrier,
    #: which cannot hold one out.
    held_out_carriers_by_line: dict[str, dict[str, str]] = field(default_factory=dict)
    single_carrier_lines: dict[str, list[str]] = field(default_factory=dict)
    #: Per doc type, lines with two or more carriers where none was held out,
    #: and why: every choice would have left the line no train or no
    #: validation family.
    lines_not_held_out: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Per doc type, lines none of whose documents names a carrier.
    lines_without_carrier: dict[str, list[str]] = field(default_factory=dict)
    #: Per doc type, lines naming fewer carriers than
    #: MIN_CARRIERS_PER_LINE_FOR_HELD_OUT, with how many they name: none held out.
    lines_below_carrier_minimum: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The eval set was frozen when this split was made: its test set is the
    #: frozen one, and the carriers held out are the ones recorded at freezing.
    frozen_eval_set: bool = False
    #: Families the hold-out moved into test from the split they were delivered in.
    moved_to_test: dict[str, int] = field(default_factory=dict)
    #: Every document of a held-out carrier: evaluation reports them apart.
    held_out_source_ids: list[str] = field(default_factory=list)
    #: The per-seed twin cap applied to train, and twins it dropped, per family.
    twin_cap: int | None = None
    twins_dropped: dict[str, int] = field(default_factory=dict)

    #: Groups per split per line of business, per doc type — what the ratio
    #: actually produced for each line.
    counts_by_line: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    #: Lines too small to measure, trained on whole.
    train_only_lines: dict[str, list[str]] = field(default_factory=dict)

    seed: int = 42
    ratios_by_doc_type: dict[str, dict[str, float]] = field(default_factory=dict)
    counts_by_doc_type: dict[str, dict[str, int]] = field(default_factory=dict)

    #: The run this corpus was built to continue (its adapter's chain trained
    #: on the documents pinned below), or None for a corpus built from the base.
    continues: str | None = None
    #: Per doc type, the families the continued chain trained on that the rules
    #: would have put outside train, and the split each would have taken: pinned
    #: to train, so the continued adapter is never validated on what it memorised.
    pinned_to_train: dict[str, dict[str, str]] = field(default_factory=dict)

    #: The split came with the data (``assign_delivered_splits``) rather than
    #: being drawn here. Synthetic documents may then sit in val and test: the
    #: delivery split by source document, so a test document's layout is one the
    #: model never trained on, and the gate reports real documents separately.
    delivered: bool = False

    def split_of(self, group_id: str) -> Split:
        try:
            return self.assignment[group_id]
        except KeyError:
            raise SplitError(
                f"group {group_id} has no split assignment. Every group must be assigned before "
                "expansion, or its documents could land in different splits."
            ) from None

    def half_of(self, group_id: str) -> ValHalf | None:
        return self.val_half.get(group_id)

    def groups_in(self, split: Split) -> list[str]:
        return sorted(g for g, s in self.assignment.items() if s == split)

    def as_dict(self) -> dict[str, Any]:
        return {
            "assignment": dict(sorted(self.assignment.items())),
            "val_half": dict(sorted(self.val_half.items())),
            "held_out_carriers": {k: sorted(v) for k, v in sorted(self.held_out_carriers.items())},
            "seed": self.seed,
            "ratios_by_doc_type": self.ratios_by_doc_type,
            "counts_by_doc_type": self.counts_by_doc_type,
            "counts_by_line": self.counts_by_line,
            "train_only_lines": {k: sorted(v) for k, v in sorted(self.train_only_lines.items())},
            "delivered": self.delivered,
            "held_out_carriers_by_line": {k: dict(sorted(v.items()))
                                          for k, v in sorted(self.held_out_carriers_by_line.items())},
            "single_carrier_lines": {k: sorted(v) for k, v in sorted(self.single_carrier_lines.items())},
            "lines_not_held_out": {k: dict(sorted(v.items()))
                                   for k, v in sorted(self.lines_not_held_out.items())},
            "lines_without_carrier": {k: sorted(v) for k, v in sorted(self.lines_without_carrier.items())},
            "lines_below_carrier_minimum": {k: dict(sorted(v.items()))
                                            for k, v in sorted(self.lines_below_carrier_minimum.items())},
            "frozen_eval_set": self.frozen_eval_set,
            "moved_to_test": dict(sorted(self.moved_to_test.items())),
            "held_out_source_ids": sorted(self.held_out_source_ids),
            "twin_cap": self.twin_cap,
            "twins_dropped": dict(sorted(self.twins_dropped.items())),
            "continues": self.continues,
            "pinned_to_train": {k: dict(sorted(v.items()))
                                for k, v in sorted(self.pinned_to_train.items())},
        }


@dataclass
class GroupRecord:
    """One family, as the splitter sees it."""

    group_id: str
    doc_type: str
    source_ids: list[str]
    carrier: str | None = None

    #: Generated documents never leave train: they have perfect labels because
    #: they were produced from them, so scoring against one measures the
    #: generator, not the model (arch v2.1 §4d).
    synthetic: bool = False

    #: A synthetic family that also holds a real document: a seed with its
    #: twins. It trains like any synthetic family, but its carrier (the real
    #: member's) is one of its line's carriers (:func:`held_out_carrier_per_line`).
    #: Read only when ``synthetic``.
    holds_real: bool = False

    #: The line of business the family belongs to (:func:`line_of`). ``None`` for
    #: types without one (ACORD, Loss Runs), which are then one line per type.
    line: str | None = None

    @property
    def size(self) -> int:
        return len(self.source_ids)


def line_of(lob: str | list[str] | None) -> str | None:
    """A document's line as a stratum key. A package policy's lines, sorted and
    joined, are one stratum of their own: they are read against the fallback
    schema, not either line's."""
    if lob is None or lob == [] or lob == "":
        return None
    if isinstance(lob, str):
        return lob.strip().lower()
    return "+".join(sorted(str(x).strip().lower() for x in lob))


def assert_bands_only_grow_train() -> None:
    """Both split edges must be non-decreasing as volume grows.

    That is what makes a band crossing move groups only toward train. An edge
    that fell — a later band with a smaller train share, say — would move groups
    a previous version trained on into val or test, and score it on them.
    """
    from common.constants import SPLIT_RATIOS_BY_VOLUME

    edges = [(r.train, r.train + r.val) for _t, r in SPLIT_RATIOS_BY_VOLUME]
    for (t0, v0), (t1, v1) in zip(edges, edges[1:], strict=False):
        if t1 < t0 - 1e-9 or v1 < v0 - 1e-9:
            raise ValueError(
                f"split bands {edges} move an edge DOWN between bands; a band crossing would "
                "move trained documents into val or test."
            )


assert_bands_only_grow_train()


def _stable_hash(group_id: str, seed: int, salt: str = "") -> float:
    """A deterministic value in [0, 1) for a group.

    Hash-based rather than shuffle-based so adding documents later never
    reshuffles the groups already assigned.
    """
    digest = hashlib.sha256(f"{seed}:{salt}:{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_group_splits(
    groups_by_doc_type: dict[str, list[GroupRecord]],
    *,
    seed: int = 42,
    ratios: SplitRatio | None = None,
    hold_out_carriers: bool = True,
    with_test: bool = True,
    trained: Collection[str] = (),
) -> GroupSplitAssignment:
    """Assign each group to a split, per document type.

    Splitting per type rather than globally keeps every type represented in every
    split — a global split on an unbalanced corpus can leave a type with no test
    documents at all, and its accuracy then silently unmeasured.

    ``with_test=False`` once the golden eval set is frozen: the frozen set IS the
    test set, so new documents split into train and val only, in the ratio's own
    train:val proportion, and no carrier is held out (the frozen set already
    holds the carriers the first split held out).

    With ``hold_out_carriers``, each line whose real documents name at least
    :data:`MIN_CARRIERS_PER_LINE_FOR_HELD_OUT` carriers holds one carrier out:
    every real family of it in that line goes to test (Fideon SPEC_09
    amendment item 4; decision 2026-10-19). The carrier is chosen by
    :func:`held_out_carrier_per_line` on the split the hash would draw, so the
    line keeps a train and a validation family. A policy carrier with any
    synthetic family is never the one: that family trains whatever the split,
    so the carrier is one the model has seen.

    ``trained``: source ids the adapter this cycle continues - and the runs it
    continued - trained on. A family holding one trains, whatever the hash, the
    line's size, the carrier hold-out or the empty-split repair would draw, and
    is recorded in ``pinned_to_train`` with the split it would have taken: the
    continued adapter memorised it, so in val it would make checkpoint
    selection and the calibration fits optimistic (module docstring).
    """
    result = GroupSplitAssignment(seed=seed)
    trained = frozenset(trained)

    for doc_type, records in sorted(groups_by_doc_type.items()):
        unique = sorted({r.group_id: r for r in records}.values(), key=lambda r: r.group_id)
        if not unique:
            continue
        pinned = _pinned_groups(unique, trained)

        # Ratios are chosen on DOCUMENT count, not group count: the volume bands
        # in §8.2 describe how much data exists, and one group of forty
        # certificates is still forty documents of training signal.
        document_count = sum(r.size for r in unique)
        ratio = ratios or split_ratio_for(document_count)
        if not with_test:
            share = ratio.train + ratio.val
            ratio = SplitRatio(ratio.train / share, ratio.val / share, 0.0)
        result.ratios_by_doc_type[doc_type] = {
            "train": ratio.train, "val": ratio.val, "test": ratio.test,
        }

        train_edge = ratio.train
        val_edge = ratio.train + ratio.val

        held_by_line: dict[str, str] = {}
        if hold_out_carriers and with_test:
            real = [r for r in unique if not r.synthetic]
            # A pinned family trains, so the choice must not count on it for
            # the line's validation. (A continued cycle gates on a frozen set,
            # which holds no carrier out, so the two rarely meet; where they
            # do, the pin wins and the family trains.)
            provisional = {
                r.group_id: "train" if r.group_id in pinned
                else _hash_split(r.group_id, seed, train_edge, val_edge) for r in real
            }
            # A policy's synthetic families too: each trains below, before the
            # hold-out is looked at, so its carrier must not be the one held out
            # (held_out_carrier_per_line counts and chooses from real ones only).
            choice = held_out_carrier_per_line(
                unique if doc_type == "policy" else real, seed, provisional,
                min_carriers=MIN_CARRIERS_PER_LINE_FOR_HELD_OUT)
            held_by_line = choice.held
            _record_hold_out(result, doc_type, choice)
        buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        by_line: dict[str | None, list[GroupRecord]] = defaultdict(list)
        for record in unique:
            by_line[record.line].append(record)

        for line, members in sorted(by_line.items(), key=lambda kv: kv[0] or ""):
            line_buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
            line_documents = sum(r.size for r in members)
            measured = line_documents >= MIN_DOCS_TO_MEASURE_LINE
            for record in members:
                held_out = False
                if record.synthetic:
                    split = "train"
                elif record.carrier and line is not None and held_by_line.get(line) == record.carrier:
                    split, held_out = "test", True
                elif not measured and line is not None:
                    split = "train"
                else:
                    split = _hash_split(record.group_id, seed, train_edge, val_edge)
                if split != "train" and record.group_id in pinned:
                    result.pinned_to_train.setdefault(doc_type, {})[record.group_id] = split
                    split, held_out = "train", False
                if held_out:
                    result.held_out_source_ids.extend(record.source_ids)
                line_buckets[split].append(record.group_id)

            if measured or line is None:
                _repair_empty_splits(
                    line_buckets, members, f"{doc_type}/{line or '-'}", seed,
                    starved=("test", "val") if with_test else ("val",), pinned=pinned,
                )
            elif not line_buckets["test"]:
                result.train_only_lines.setdefault(doc_type, []).append(line)
                log.info(
                    "%s/%s has %d document(s), fewer than %d: all of it trains, and the line is "
                    "not measured until it has more", doc_type, line, line_documents,
                    MIN_DOCS_TO_MEASURE_LINE,
                )
            else:
                # A held-out carrier's documents stay held out even in a small
                # line: training on them would make the held-out slice a
                # carrier the model has seen. So this line is NOT train-only,
                # and recording it as one said the opposite of what happened.
                log.warning(
                    "%s/%s has %d document(s), fewer than %d, and %d group(s) belong to a "
                    "held-out carrier: those are tested, not trained. %s",
                    doc_type, line, line_documents, MIN_DOCS_TO_MEASURE_LINE,
                    len(line_buckets["test"]),
                    "The line has NO training documents - the model never sees it."
                    if not line_buckets["train"] else
                    f"{len(line_buckets['train'])} group(s) train.",
                )
            if line is not None:
                result.counts_by_line.setdefault(doc_type, {})[line] = {
                    k: len(v) for k, v in line_buckets.items()
                }
            for split_name, ids in line_buckets.items():
                buckets[split_name].extend(ids)

        for split, members in buckets.items():
            for group_id in members:
                result.assignment[group_id] = split  # type: ignore[assignment]

        _assign_validation_halves(result, buckets["val"], seed)

        result.counts_by_doc_type[doc_type] = {s: len(m) for s, m in buckets.items()}
        _log_pinned(result, doc_type)

        if document_count < 10:
            log.warning(
                "%s has only %d documents in %d group(s) (%s). At this volume eval metrics are "
                "directional, not conclusive (arch §8.2).",
                doc_type, document_count, len(unique), result.counts_by_doc_type[doc_type],
            )

    return result


def _pinned_groups(records: list[GroupRecord], trained: frozenset[str]) -> set[str]:
    """The families holding a document the continued chain trained on.

    By family, not by document: a renewal that arrived since joins the family
    of the document it renews, and the family is the unit that must not span
    the split. Synthetic families train anyway."""
    if not trained:
        return set()
    return {r.group_id for r in records if not r.synthetic and trained.intersection(r.source_ids)}


def _log_pinned(result: GroupSplitAssignment, doc_type: str) -> None:
    pinned = result.pinned_to_train.get(doc_type)
    if pinned:
        log.warning(
            "%s: %d famil(ies) the continued adapter trained on would have moved out of train "
            "(%s); pinned to train, so it is never validated on documents it memorised",
            doc_type, len(pinned), dict(sorted(pinned.items())[:5]),
        )


def _hash_split(group_id: str, seed: int, train_edge: float, val_edge: float) -> str:
    position = _stable_hash(group_id, seed)
    return "train" if position < train_edge else "val" if position < val_edge else "test"


def _record_hold_out(result: GroupSplitAssignment, doc_type: str, choice: CarrierHoldOut) -> None:
    """Write one doc type's carrier hold-out onto the assignment."""
    if choice.held:
        result.held_out_carriers_by_line[doc_type] = dict(choice.held)
    if choice.single:
        result.single_carrier_lines[doc_type] = list(choice.single)
    if choice.below_minimum:
        result.lines_below_carrier_minimum[doc_type] = dict(choice.below_minimum)
    if choice.not_held_out:
        result.lines_not_held_out[doc_type] = dict(choice.not_held_out)
        log.warning("%s: no carrier held out of %s - %s", doc_type, sorted(choice.not_held_out),
                    next(iter(choice.not_held_out.values())))
    if choice.no_carrier:
        result.lines_without_carrier[doc_type] = list(choice.no_carrier)


#: The split names a delivery may carry, in a document's metadata ``split``.
DELIVERED_SPLITS = ("train", "val", "test")


#: Fideon SPEC_09 amendment item 4: at least one carrier per line is kept out
#: of train and validation, and reported apart in evaluation. Off for a
#: delivered split, by decision (2026-10-07): the SPEC_21 delivery already keeps
#: whole seeds - layouts the model never trains on - in validation and test,
#: and with 2 to 28 train seeds a line, holding a carrier out as well would
#: move 13 seeds' families out of training. Pass ``hold_out_carriers=True`` to
#: assign_delivered_splits to apply it.
HOLD_OUT_CARRIER_PER_LINE = False

#: Fideon SPEC_09 amendment item 5: at most this many synthetic twins of one
#: seed, per render mode, train.
MAX_TWINS_PER_SEED = 20


@dataclass
class CarrierHoldOut:
    """Which carrier each line holds out, and the lines that hold none out."""

    held: dict[str, str] = field(default_factory=dict)
    #: Lines with one carrier: none can be held out.
    single: list[str] = field(default_factory=list)
    #: Lines with two or more carriers but fewer than the minimum asked for,
    #: and how many they have: none held out.
    below_minimum: dict[str, int] = field(default_factory=dict)
    #: Lines where every carrier is the line's only train or only validation
    #: carrier, so holding any out would leave the line untrained or unvalidated.
    not_held_out: dict[str, str] = field(default_factory=dict)
    #: Lines none of whose documents names a carrier.
    no_carrier: list[str] = field(default_factory=list)


def held_out_carrier_per_line(records: list[GroupRecord], seed: int,
                              split_of: dict[str, str] | None = None, *,
                              min_carriers: int = MIN_CARRIERS_PER_LINE_FOR_HELD_OUT) -> CarrierHoldOut:
    """The carrier each line holds out of train and validation.

    Only in a line naming at least ``min_carriers`` carriers; a line with
    fewer holds none out (``below_minimum``).

    Only a carrier whose families can all go to test while the line keeps at
    least one train family and one validation family (``split_of``, the
    delivered split): a line that loses its only validation carrier has no
    validation at all. Among those, the one with the fewest train and
    validation documents - what holding it out costs; its test documents cost
    nothing - with ties broken by a seeded hash so the choice is reproducible.

    On a policy (decision 2026-10-22) carriers are counted from the families
    holding a real document - a seed with its twins among them, so a line's
    count is its real carriers - and a placeholder ("N/A", "Unknown", blank) is
    no carrier (:func:`is_placeholder_carrier`). A carrier with a synthetic
    family in the line is never chosen: twins train wherever they sit - a drawn
    split keeps them in train, a family holding a seed and its twins with them
    - so its layouts are seen in training. A family of twins alone counts for
    nothing: its carrier may be generated. Other doc types are read as given.
    """
    families: dict[str, dict[str, list[GroupRecord]]] = defaultdict(lambda: defaultdict(list))
    #: Per line, the policy carriers a synthetic family trains.
    trained: dict[str, set[str]] = defaultdict(set)
    lines: set[str] = set()
    for record in records:
        if not record.line:
            continue
        carrier = record.carrier
        if record.doc_type == "policy":
            carrier = None if is_placeholder_carrier(carrier) else carrier
            if record.synthetic:
                if carrier:
                    trained[record.line].add(carrier)
                if not record.holds_real:
                    continue
        lines.add(record.line)
        if carrier:
            families[record.line][carrier].append(record)
    result = CarrierHoldOut(no_carrier=sorted(lines - set(families)))
    split_of = split_of or {}

    def splits_without(carriers: dict[str, list[GroupRecord]], held: str) -> set[str]:
        return {split_of.get(r.group_id, "train") for c, rs in carriers.items() if c != held for r in rs}

    for line, carriers in sorted(families.items()):
        if len(carriers) < 2:
            result.single.append(line)
            continue
        if len(carriers) < min_carriers:
            result.below_minimum[line] = len(carriers)
            continue
        unseen = [c for c in carriers if c not in trained[line]]
        eligible = [c for c in unseen if {"train", "val"} <= splits_without(carriers, c)]
        if not eligible:
            result.not_held_out[line] = (
                "every carrier trains through a synthetic family, so none is unseen" if not unseen else
                "every carrier is the line's only train or only validation carrier, or trains through a "
                "synthetic family; holding one out would leave the line untrained or unvalidated, or "
                "hold out a carrier the model has seen")
            continue

        cost = {
            c: (sum(r.size for r in carriers[c] if split_of.get(r.group_id, "train") != "test"),
                _stable_hash(c, seed, f"carrier:{line}"))
            for c in eligible
        }
        result.held[line] = min(eligible, key=cost.__getitem__)
    return result


def cap_twins(documents: list[Any], assignment: GroupSplitAssignment, *,
              cap: int = MAX_TWINS_PER_SEED, seed: int = 42) -> list[Any]:
    """``documents`` with at most ``cap`` synthetic twins per seed and render mode in train.

    A seed is a document's family. The twins kept are chosen by a seeded hash
    of their ids, so the same corpus always keeps the same ones. Real documents
    and validation and test are never dropped. What was dropped is recorded on
    ``assignment`` and so in the corpus manifest.
    """
    counts: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for document in documents:
        if document.synthetic and assignment.assignment.get(document.family) == "train":
            counts[(document.family, getattr(document, "render_mode", None) or "default")].append(document)
    dropped: set[str] = set()
    assignment.twin_cap = cap
    for (family, _mode), twins in counts.items():
        if len(twins) <= cap:
            continue
        ordered = sorted(twins, key=lambda d: (_stable_hash(d.source_id, seed, "twin-cap"), d.source_id))
        for document in ordered[cap:]:
            dropped.add(document.source_id)
            assignment.twins_dropped[family] = assignment.twins_dropped.get(family, 0) + 1
    if dropped:
        log.info("twin cap: %d twin(s) above %d per seed and render mode left out of train",
                 len(dropped), cap)
    return [d for d in documents if d.source_id not in dropped]


def assign_delivered_splits(
    groups_by_doc_type: dict[str, list[GroupRecord]],
    split_of_group: dict[str, str],
    *,
    seed: int = 42,
    with_test: bool = True,
    hold_out_carriers: bool = HOLD_OUT_CARRIER_PER_LINE,
    min_carriers: int = MIN_CARRIERS_PER_LINE_FOR_HELD_OUT,
    trained: Collection[str] = (),
) -> GroupSplitAssignment:
    """Use the split the data arrived with, instead of drawing one.

    For a delivery that was split upstream — a synthetic set whose generator put
    each source document and all its twins in one split. The families here are
    the delivery's own (one per source document), so the split is taken as
    given; what this still enforces is that it is usable: no family in two
    splits, a non-empty val, and a non-empty test unless the eval set is frozen.
    Validation is halved by group exactly as for a drawn split.

    One exception to "as given": with ``hold_out_carriers`` (off by default),
    one carrier per line with at least ``min_carriers`` carriers is held out -
    every family of it in that line goes to test, whatever split it was
    delivered in (Fideon SPEC_09 amendment item 4) - unless that would leave the
    line no train or no validation family (:func:`held_out_carrier_per_line`). Not once the eval
    set is frozen: the frozen set is the test set and takes no new documents.

    Held out of its LINE: the carrier still trains in its other lines, so its
    test documents measure a line layout the model has not seen, not a carrier
    it has never seen. Recorded per line (``held_out_carriers_by_line``), never
    as a carrier held out of the whole doc type (``held_out_carriers``).

    The other exception: ``trained``, as for :func:`assign_group_splits`. A
    family holding a document the continued chain trained on trains, whatever
    split the delivery gave it this time, and is recorded in
    ``pinned_to_train``: a delivery re-split since (prepare_bundles draws its
    split per delivery) could otherwise put it in val.
    """
    result = GroupSplitAssignment(seed=seed, delivered=True)
    trained = frozenset(trained)
    for doc_type, records in sorted(groups_by_doc_type.items()):
        unique = sorted({r.group_id: r for r in records}.values(), key=lambda r: r.group_id)
        if not unique:
            continue
        pinned = _pinned_groups(unique, trained)
        held: dict[str, str] = {}
        if hold_out_carriers and with_test:
            # A pinned family trains, so the choice must not count on it for
            # the line's validation. (A continued cycle gates on a frozen set,
            # which holds no carrier out, so the two rarely meet; where they
            # do, the pin wins and the family trains.)
            delivered = {**split_of_group, **dict.fromkeys(pinned, "train")}
            choice = held_out_carrier_per_line(unique, seed, delivered, min_carriers=min_carriers)
            held = choice.held
            _record_hold_out(result, doc_type, choice)
        buckets: dict[str, list[str]] = {name: [] for name in DELIVERED_SPLITS}
        documents = dict.fromkeys(DELIVERED_SPLITS, 0)
        by_line: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(DELIVERED_SPLITS, 0))
        for record in unique:
            split = split_of_group.get(record.group_id)
            if split not in DELIVERED_SPLITS:
                raise SplitError(f"group {record.group_id} has no delivered split (got {split!r})")
            if record.group_id in pinned:
                if split != "train":
                    result.pinned_to_train.setdefault(doc_type, {})[record.group_id] = split
                split = "train"
            elif record.line in held and record.carrier == held[record.line]:
                if split != "test":
                    result.moved_to_test[doc_type] = result.moved_to_test.get(doc_type, 0) + 1
                split = "test"
                result.held_out_source_ids.extend(record.source_ids)
            if split == "test" and not with_test:
                raise SplitError(
                    f"group {record.group_id} is delivered as test, but the eval set is frozen: "
                    "a frozen yardstick takes no new documents. Deliver new documents as train or val."
                )
            buckets[split].append(record.group_id)
            documents[split] += record.size
            if record.line is not None:
                by_line[record.line][split] += 1
        if not buckets["val"]:
            raise SplitError(f"{doc_type}: the delivered split has no val documents")
        if with_test and not buckets["test"]:
            raise SplitError(f"{doc_type}: the delivered split has no test documents")
        for split, members in buckets.items():
            for group_id in members:
                result.assignment[group_id] = split  # type: ignore[assignment]
        _assign_validation_halves(result, buckets["val"], seed)
        _log_pinned(result, doc_type)
        total = sum(documents.values())
        result.ratios_by_doc_type[doc_type] = {s: round(n / total, 4) for s, n in documents.items()}
        result.counts_by_doc_type[doc_type] = {s: len(m) for s, m in buckets.items()}
        if by_line:
            result.counts_by_line[doc_type] = {line: dict(c) for line, c in sorted(by_line.items())}
    return result


def _repair_empty_splits(
    buckets: dict[str, list[str]], groups: list[GroupRecord], doc_type: str, seed: int,
    starved: tuple[str, ...] = ("test", "val"), pinned: Collection[str] = (),
) -> None:
    """Move one group into a starved split rather than ship an unusable corpus.

    An empty test split cannot be evaluated at all and an empty val split
    disables both early stopping and calibration. Fires only at very small N, and
    it is the one case where assignment is not stable under corpus growth.

    Synthetic groups are never moved out of train — that rule outranks this
    repair, because a test split made of generated documents measures the
    generator and reports it as model accuracy. Nor are ``pinned`` families
    (those a continued adapter trained on): moved to val, the continued model
    would be validated on documents it memorised.
    """
    if len(groups) < MIN_GROUPS_TO_REPAIR:
        return
    movable = {g.group_id for g in groups if not g.synthetic and g.group_id not in pinned}
    for split in starved:
        if buckets[split]:
            continue
        donor = max(buckets, key=lambda s: len([g for g in buckets[s] if g in movable]))
        candidates = [g for g in buckets[donor] if g in movable]
        if len(candidates) < 2:
            continue
        moved = sorted(candidates, key=lambda g: _stable_hash(g, seed))[-1]
        buckets[donor].remove(moved)
        buckets[split].append(moved)
        log.warning(
            "%s: group %r moved to %s so the split is not empty — an empty test split cannot be "
            "evaluated. This is the one case where assignment is not stable under growth.",
            doc_type, moved, split,
        )


def _assign_validation_halves(
    result: GroupSplitAssignment, val_groups: list[str], seed: int
) -> None:
    """Split validation by group into a calibration half and a threshold half.

    Fitting the calibrator and setting the review threshold on the same documents
    makes the threshold optimistic: the calibrator has already seen the errors the
    threshold is meant to price, so the measured error rate among auto-accepted
    fields is lower than production will deliver (arch v2.1 §5.3-5.4).

    Split by GROUP, for the same reason the outer split is: two renewals of one
    account across the halves is the same leak at a smaller scale.
    """
    for group_id in sorted(val_groups):
        half: ValHalf = (
            "calibration" if _stable_hash(group_id, seed, "valhalf") < 0.5 else "threshold"
        )
        result.val_half[group_id] = half


def assert_no_leakage(assignment: GroupSplitAssignment, rows: list[dict[str, Any]]) -> None:
    """Assert no group appears in more than one split.

    Run over the **expanded** rows, because that is where the failure would
    actually show up: the assignment itself is a dict and cannot contain a
    duplicate, but expansion into tasks, windows and modes is what could place
    them inconsistently.
    """
    splits_seen: dict[str, set[str]] = defaultdict(set)
    sources_seen: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        group_id, split = row.get("group_id"), row.get("split")
        source_id = row.get("source_id")
        if not group_id or not split:
            raise SplitError(
                f"row is missing group_id or split: source_id={source_id!r}. Every row carries "
                "its group, because the group is the unit that must not span the split."
            )
        splits_seen[group_id].add(split)
        if source_id:
            sources_seen[source_id].add(split)

    leaked = {g: sorted(s) for g, s in splits_seen.items() if len(s) > 1}
    if leaked:
        raise SplitError(
            f"LEAKAGE: {len(leaked)} group(s) appear in more than one split — "
            f"{dict(list(leaked.items())[:5])}. A family spanning the split means the model is "
            "evaluated on a template it trained on, which inflates every metric without "
            "reflecting generalisation (arch v2.1 §8.2)."
        )

    document_leaks = {s: sorted(v) for s, v in sources_seen.items() if len(v) > 1}
    if document_leaks:
        raise SplitError(
            f"LEAKAGE: {len(document_leaks)} source_id(s) span splits — "
            f"{dict(list(document_leaks.items())[:5])}. Split before task and modality "
            "expansion, not after."
        )

    for group_id, seen in splits_seen.items():
        expected = assignment.assignment.get(group_id)
        if expected and seen != {expected}:
            raise SplitError(
                f"group {group_id} was assigned to {expected!r} but its rows are in {sorted(seen)}"
            )


def assert_synthetic_is_train_only(rows: list[dict[str, Any]]) -> None:
    """Synthetic documents never appear in validation or test (arch v2.1 §4d).

    Their labels are perfect because they were generated from them, so a metric
    scored against one reports how faithfully the generator rendered its own
    input — and reports it as model accuracy.
    """
    escaped = sorted({
        str(row.get("source_id"))
        for row in rows
        if row.get("synthetic") and row.get("split") != "train" and row.get("source_id")
    })
    if escaped:
        raise SplitError(
            f"{len(escaped)} synthetic document(s) reached a non-train split: {escaped[:5]}. "
            "Scoring against generated labels measures the generator, not the model."
        )


def assert_single_tenant(rows: list[dict[str, Any]]) -> None:
    """Assert a corpus file carries exactly one tenant.

    Corpus composition **is** training data, so a file mixing tenants means a
    model trained on Broker A's data contains Broker B's (arch §8b).
    """
    tenants = {row.get("tenant_id") for row in rows}
    if len(tenants) > 1:
        raise SplitError(
            f"corpus rows span multiple tenants: {sorted(t for t in tenants if t)}. "
            "Cross-tenant mixing is prohibited — a model trained on one broker's data must "
            "never contain another's (arch §8b)."
        )
