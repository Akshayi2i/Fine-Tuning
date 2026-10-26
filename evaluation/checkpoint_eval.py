"""Checkpoint selection by generated field F1 (arch v2.1 §11.2).

**Validation loss does not select what ships.** It is logged, and it drives early
stopping, but it is a poor proxy for extraction accuracy: loss is averaged over
every token the model produces, so it is dominated by the easy copy tokens — the
schema keys, the JSON punctuation, the boilerplate — that the model gets right
from the first hundred steps. A checkpoint can improve on loss while getting
worse at the values, which is the only thing the gate reads.

Field F1 needs **generation**, and the training loop's own eval does not generate
efficiently for a VLM. So selection is a separate job, run after training:

    last 3 checkpoints + the best-loss checkpoint (+ the one nearest two-thirds
    of the final step, :func:`two_thirds_checkpoint`)
        -> generate on the validation split through vLLM
        -> score field F1 with the same code the gate uses
        -> the finalists on whole documents, compared on the gate mirror
           (:func:`choose_on_documents`)
        -> the winner is what ships; its answers are kept for calibration
           and its error tallies for the next fix (:func:`score_pass`)

**Checkpoints are loaded as a decoder LoRA, not merged.** vLLM applies one LoRA
per request, which is exactly enough for this — a per-checkpoint merge would cost
~16GB of disk and several minutes each, to answer a question that does not need
the weights folded in.

The candidate set is small on purpose. Scoring every checkpoint would multiply
the cost by the epoch count for a decision that, in practice, sits between the
last few and the one early stopping liked.

**On a pod with several GPUs the candidates are scored at once**, one worker
process and vLLM engine per GPU (:class:`ParallelScorer`), each with the engine
the in-process scorer uses; only how many run at a time changes. One engine on
one card took ~35 minutes a candidate on the smoke set, ~2.5 hours for four,
with three of four GPUs idle. Fewer candidates than GPUs - the two finalists -
are sharded across all of them instead, call for call (:func:`generate_sharded`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: How many trailing checkpoints to score, alongside the best-loss one.
TRAILING_CANDIDATES = 3

_CHECKPOINT_STEP = re.compile(r"checkpoint-(\d+)$")


class CheckpointEvalError(RuntimeError):
    """Raised when a checkpoint cannot be selected safely."""


class Scorer(Protocol):
    """Generates on the validation split and returns that checkpoint's metrics.

    Injected rather than imported so this module is testable without a GPU, and
    so the scorer is the **same** one the gate uses — a selector scoring by a
    different definition of "correct" picks a checkpoint the gate then rejects.
    """

    def __call__(self, checkpoint: str) -> dict[str, float]: ...


@dataclass
class CheckpointScore:
    """One candidate's result."""

    checkpoint: str
    step: int
    metrics: dict[str, float] = field(default_factory=dict)
    is_best_loss: bool = False
    #: The checkpoint nearest two-thirds of the final step (:func:`two_thirds_checkpoint`).
    is_two_thirds: bool = False

    @property
    def field_f1(self) -> float:
        """The selection metric — the gate's primary (arch v2.1 §15.2)."""
        return float(self.metrics.get("field_normalized_match", 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "step": self.step,
            "field_f1": round(self.field_f1, 4),
            "is_best_loss": self.is_best_loss,
            "is_two_thirds": self.is_two_thirds,
            "metrics": {k: round(v, 4) for k, v in sorted(self.metrics.items())},
        }


@dataclass
class SelectionReport:
    """Which checkpoint won, against what it was compared, and by how much."""

    selected: str | None = None
    scores: list[CheckpointScore] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: Set when the top two were within the tie-break margin on the sample and
    #: were scored again on the full validation split.
    tie_break: dict[str, Any] | None = None
    #: Set when the finalists were scored on whole validation documents, which
    #: decided (:func:`choose_on_documents`).
    document_choice: dict[str, Any] | None = None
    #: The two-thirds candidate, and whether it won or tied (:func:`two_thirds_outcome`).
    two_thirds: dict[str, Any] | None = None
    #: The selected checkpoint's saved finalist pass: where its answers and
    #: error tallies went (the checkpoint stage fills it in).
    finalist_pass: dict[str, Any] | None = None
    #: The gate's field match predicted from the gate mirror (:func:`predicted_gate_field_match`).
    early_warning: dict[str, Any] | None = None

    @property
    def best_loss_checkpoint(self) -> str | None:
        return next((s.checkpoint for s in self.scores if s.is_best_loss), None)

    @property
    def gate_mirror_accuracy(self) -> float | None:
        """The selected checkpoint's validation field match as the gate measures
        it (:data:`validation_generation.GATE_MIRROR_ACCURACY`), where its
        finalists' pass measured one."""
        finalists = (self.document_choice or {}).get("finalists") or {}
        value = (finalists.get(self.selected) or {}).get("gate_mirror_accuracy")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @property
    def margin(self) -> float:
        """Sample field F1 of the SELECTED checkpoint minus the best other
        candidate's (the sample's leader before anything is selected).

        Worth recording rather than just the winner: a margin inside ordinary
        eval noise means the selection was close to arbitrary, and a later
        regression is better read against that than against a bare filename.

        Of the selected checkpoint, not the sample's leader: whole validation
        documents (:func:`choose_on_documents`) or the full split
        (:func:`break_tie`) may pick one the sample ranks behind, and the
        leader's lead recorded beside it read as the selected one's - "+0.02"
        for a checkpoint 0.02 behind. Negative then: how far behind on the
        sample the decision that chose it went.
        """
        if len(self.scores) < 2:
            return 0.0
        chosen = next((s for s in self.scores if s.checkpoint == self.selected), None)
        if chosen is None:
            chosen = max(self.scores, key=lambda s: (s.field_f1, s.step))
        best_other = max(s.field_f1 for s in self.scores if s is not chosen)
        return round(chosen.field_f1 - best_other, 4)

    @property
    def loss_and_f1_disagreed(self) -> bool:
        """Whether the best-loss checkpoint is NOT the one with the best F1.

        The reason this job exists. When it fires, validation loss would have
        shipped a different model — so it is logged loudly rather than buried.
        """
        best = self.best_loss_checkpoint
        return bool(best and self.selected and best != self.selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "selection_metric": "field_normalized_match",
            # The selected checkpoint's sample field F1 minus the best other
            # candidate's: negative when a later decision chose one behind.
            "margin_over_runner_up": self.margin,
            "best_loss_checkpoint": self.best_loss_checkpoint,
            "loss_and_f1_disagreed": self.loss_and_f1_disagreed,
            "candidates": [s.as_dict() for s in self.scores],
            "skipped": [{"checkpoint": c, "reason": r} for c, r in self.skipped],
            "tie_break": self.tie_break,
            "document_choice": self.document_choice,
            "two_thirds_candidate": self.two_thirds,
            # The selected checkpoint's validation field match as the gate
            # measures it; beside the gate's own once that has run, it is the
            # early warning's evidence (predicted_gate_field_match).
            "validation_gate_mirror_accuracy": self.gate_mirror_accuracy,
            "early_warning": self.early_warning,
            "finalist_pass": self.finalist_pass,
        }


def checkpoint_step(path: str) -> int:
    """The global step from a ``checkpoint-{n}`` directory name.

    Raises rather than returning 0 on an unparseable name: ordering the
    candidates by step is what "the last three" means, and a silent 0 would put
    a real checkpoint at the front of the list.
    """
    match = _CHECKPOINT_STEP.search(str(path).rstrip("/"))
    if not match:
        raise CheckpointEvalError(
            f"cannot read a step number from {path!r}; expected a directory ending "
            "'checkpoint-{n}'. Ordering by step is what selects the last three."
        )
    return int(match.group(1))


def two_thirds_checkpoint(checkpoints: Sequence[str]) -> str | None:
    """The checkpoint directory on disk whose step is nearest two-thirds of the
    run's final step, or None (no checkpoints, or none on disk).

    Looked for beside the run's checkpoints, not only among them: the
    candidates are the checkpoints an evaluation fell on, and the one at
    two-thirds is usually a resume point between them. A tie goes to the later
    step, as every tie in selection does.

    Why it is a candidate: if it wins or ties, two epochs would ship the same
    model in two-thirds of the time - the evidence for moving to two.
    """
    if not checkpoints:
        return None
    ordered = sorted({str(c).rstrip("/") for c in checkpoints}, key=checkpoint_step)
    run_dir = Path(ordered[-1]).parent
    on_disk = {str(p) for p in run_dir.glob("checkpoint-*") if p.is_dir() and _CHECKPOINT_STEP.search(p.name)}
    on_disk |= {c for c in ordered if Path(c).is_dir()}
    if not on_disk:
        return None
    final = max(checkpoint_step(c) for c in [*on_disk, *ordered])
    target = final * 2 / 3
    return min(on_disk, key=lambda c: (abs(checkpoint_step(c) - target), -checkpoint_step(c)))


def two_thirds_outcome(report: SelectionReport, *, sample_margin: float) -> dict[str, Any] | None:
    """Whether the two-thirds candidate won or tied: ``won`` when it was
    selected; ``tied`` when it was not, but was within the margin of the
    decision that chose the winner - the guard margin on the finalists'
    measure when both were finalists, else ``sample_margin`` on the sample."""
    named = (report.two_thirds or {}).get("checkpoint")
    candidate = next((s for s in report.scores if s.is_two_thirds or s.checkpoint == named), None)
    if candidate is None:
        # Not scored (skipped) or not looked for: nothing won or tied.
        return report.two_thirds
    winner = next((s for s in report.scores if s.checkpoint == report.selected), None)
    out = {**(report.two_thirds or {}), "checkpoint": candidate.checkpoint, "step": candidate.step,
           "won": candidate.checkpoint == report.selected, "tied": False}
    if winner is None or out["won"]:
        return out
    choice = report.document_choice or {}
    finalists = choice.get("finalists") or {}
    measure = choice.get("field_match_measure") or DOCUMENT_ACCURACY
    if candidate.checkpoint in finalists and winner.checkpoint in finalists:
        mine, best = (finalists[candidate.checkpoint].get(measure), finalists[winner.checkpoint].get(measure))
        margin, decided_on = float(choice.get("guard_margin") or 0.0), measure
    else:
        mine, best, margin, decided_on = candidate.field_f1, winner.field_f1, sample_margin, "sample field_normalized_match"
    if isinstance(mine, (int, float)) and isinstance(best, (int, float)):
        out.update(tied=bool(best - mine <= margin), behind_by=round(float(best - mine), 4),
                   margin=margin, measured_on=decided_on)
    return out


#: Earlier cycles with both numbers before a prediction is printed.
EARLY_WARNING_MIN_CYCLES = 2


def predicted_gate_field_match(
    mirror: float | None, history: Sequence[dict[str, Any]], *, min_cycles: int = EARLY_WARNING_MIN_CYCLES,
) -> dict[str, Any]:
    """The golden gate's field match predicted from this cycle's validation
    gate mirror: the mirror plus the mean offset (gate minus mirror) over the
    earlier cycles that recorded both - ``history`` rows of
    ``{"version", "gate_mirror_accuracy", "gate_field_match"}``.

    An early warning, read before calibration and the gate run: a prediction
    below the 0.92 tier says so hours sooner. It predicts; it decides nothing.
    No prediction (``predicted`` None) with fewer than ``min_cycles`` such
    cycles, or no mirror this cycle. Scoring during training is out of scope:
    the mirror is measured once, on the finalists, after training.
    """
    pairs = [{"version": h.get("version"), "gate_mirror_accuracy": float(h["gate_mirror_accuracy"]),
              "gate_field_match": float(h["gate_field_match"])}
             for h in history
             if isinstance(h.get("gate_mirror_accuracy"), (int, float)) and isinstance(h.get("gate_field_match"), (int, float))]
    out: dict[str, Any] = {"validation_gate_mirror_accuracy": mirror, "cycles": len(pairs), "history": pairs,
                           "predicted": None, "mean_offset": None, "summary": None}
    if mirror is None or len(pairs) < min_cycles:
        out["note"] = ("no gate mirror this cycle" if mirror is None else
                       f"{len(pairs)} earlier cycle(s) with both numbers; a prediction needs {min_cycles}")
        return out
    offset = sum(p["gate_field_match"] - p["gate_mirror_accuracy"] for p in pairs) / len(pairs)
    predicted = mirror + offset
    out.update(predicted=round(predicted, 4), mean_offset=round(offset, 4),
               summary=f"predicted gate field match ≈ {predicted:.3f} (mean offset over {len(pairs)} cycles)")
    return out


def select_candidates(
    checkpoints: list[str],
    best_loss: str | None = None,
    *,
    trailing: int = TRAILING_CANDIDATES,
    two_thirds: str | None = None,
) -> list[CheckpointScore]:
    """The candidate set: the last ``trailing`` by step, plus the best-loss one,
    plus ``two_thirds`` (:func:`two_thirds_checkpoint`) when it is not already one.

    The best-loss checkpoint is included even when it is not among the last few,
    because that is precisely the interesting case — early stopping liked it and
    training continued past it.
    """
    if not checkpoints:
        raise CheckpointEvalError(
            "no checkpoints to select from. Training saves on `save_steps`; a run with none "
            "either crashed before its first save or had save_strategy disabled."
        )

    ordered = sorted({c.rstrip("/") for c in checkpoints}, key=checkpoint_step)
    chosen = ordered[-trailing:]
    if best_loss:
        best_loss = best_loss.rstrip("/")
        if best_loss not in chosen:
            chosen = [best_loss, *chosen]
    # Only when it is not one already: then it is scored as what it already is.
    two_thirds = two_thirds.rstrip("/") if two_thirds else None
    added = bool(two_thirds) and two_thirds not in chosen
    if added:
        chosen = [two_thirds, *chosen]

    return [
        CheckpointScore(
            checkpoint=path,
            step=checkpoint_step(path),
            is_best_loss=bool(best_loss) and path == best_loss,
            is_two_thirds=added and path == two_thirds,
        )
        for path in sorted(set(chosen), key=checkpoint_step)
    ]


def select_best(
    checkpoints: list[str],
    scorer: Scorer,
    *,
    best_loss: str | None = None,
    trailing: int = TRAILING_CANDIDATES,
    two_thirds: str | None = None,
) -> SelectionReport:
    """Score every candidate and return the one with the best field F1.

    A checkpoint whose scoring raises is **skipped and recorded**, not treated as
    a zero: a scorer that fell over on one candidate says nothing about that
    candidate's quality, and scoring it zero would silently remove it from
    contention.
    """
    report = SelectionReport()
    candidates = select_candidates(checkpoints, best_loss, trailing=trailing, two_thirds=two_thirds)
    if two_thirds:
        # Recorded either way: a run whose two-thirds checkpoint was already a
        # candidate (a short run) still says which it was.
        report.two_thirds = {"checkpoint": two_thirds.rstrip("/"), "step": checkpoint_step(two_thirds),
                             "final_step": max(checkpoint_step(c) for c in checkpoints),
                             "added": any(c.is_two_thirds for c in candidates)}
    # A scorer that can score several at once (ParallelScorer) is given them all.
    outcomes = _score_all(scorer, [c.checkpoint for c in candidates])
    for candidate in candidates:
        try:
            outcome = (outcomes[candidate.checkpoint] if outcomes is not None
                       else scorer(candidate.checkpoint))
            if isinstance(outcome, BaseException):
                raise outcome
            candidate.metrics = dict(outcome)
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not lose the rest
            report.skipped.append((candidate.checkpoint, f"{type(exc).__name__}: {exc}"))
            log.warning("could not score %s: %s", candidate.checkpoint, exc)
            continue
        report.scores.append(candidate)

    if not report.scores:
        raise CheckpointEvalError(
            "no checkpoint could be scored, so none can be selected. Merging an arbitrary one "
            f"would ship a model nobody measured. Skipped: {report.skipped}"
        )

    # Ties break toward the LATER step: it has seen more data, and picking the
    # earlier one on a tie would quietly prefer an under-trained checkpoint
    # whenever the metric saturates.
    winner = max(report.scores, key=lambda s: (s.field_f1, s.step))
    report.selected = winner.checkpoint

    if report.loss_and_f1_disagreed:
        log.warning(
            "validation loss and field F1 disagree: loss preferred %s, field F1 selected %s "
            "(margin %.4f). This is why selection does not read loss — it is averaged over "
            "every token, so easy copy tokens dominate it (arch v2.1 §11.2).",
            report.best_loss_checkpoint, report.selected, report.margin,
        )
    if 0 < report.margin < 0.005:
        log.warning(
            "the selected checkpoint beat the runner-up by only %.4f field F1, which is inside "
            "ordinary eval noise — treat this selection as close to arbitrary.", report.margin,
        )
    log.info("selected %s from %d candidates", report.selected, len(report.scores))
    return report


def generation_scorer(
    rows: list[dict[str, Any]], model: Any, *, repeat: bool = False, repeat_modes: Sequence[str] | None = None,
    finalists: bool = False, save_dir: Path | None = None, val_path: str | None = None,
) -> Scorer:
    """Score a checkpoint by generating the validation rows with it as a LoRA.

    Scored through :func:`evaluation.validation_generation.score_generations`,
    which is the gate's own ``build_report``. A selector scoring by a different
    definition of "correct" picks a checkpoint the gate rejects.

    ``repeat``: each checkpoint's rows are generated a second time, in the
    reverse order (``generate_validation(order="reversed")``), so the engine
    batches them differently, and its metrics carry how often the two readings
    agree (``document_run_to_run_agreement`` over whole documents).
    ``repeat_modes`` narrows that second reading to the rows read in those modes
    - the gate's own repeat mode (``golden_eval.run_to_run_mode``, image_only);
    none = every row, twice the generation.

    ``finalists``: this is the pass the finalists are judged on
    (:func:`score_pass`): its metrics carry the gate mirror, and with
    ``save_dir`` each checkpoint's answers and error tallies are saved there.
    Each setting can also be given after the scorer is made (``.configure``).
    """
    from evaluation.validation_generation import (
        assert_generations_usable,
        generate_validation,
        generation_settings,
    )

    if not rows:
        raise CheckpointEvalError(
            "the validation split is empty, so no checkpoint can be scored. Selecting by "
            "nothing would ship an arbitrary checkpoint."
        )

    def close() -> None:
        from inference_core.model_runner import release_model

        release_model(model)

    passes: list[dict[str, Any]] = []
    settings: dict[str, Any] = {"repeat": repeat, "repeat_modes": list(repeat_modes or []),
                                "finalists": finalists, "save_dir": save_dir, "val_path": val_path}
    saved: dict[str, dict[str, str]] = {}

    def configure(**changes: Any) -> None:
        unknown = set(changes) - set(settings)
        if unknown:
            raise TypeError(f"generation_scorer has no setting {sorted(unknown)}")
        settings.update(changes)

    def score(checkpoint: str) -> dict[str, float]:
        stats: dict[str, Any] = {}
        generations = generate_validation(rows, model, adapter=checkpoint, stats=stats)
        # What each pass cost, for the stage result: rows, calls, seconds.
        record = {"checkpoint": checkpoint, **stats}
        passes.append(record)
        # Raises for a broken pass, so select_best records this checkpoint as
        # unscorable instead of ranking it at 0.0 — and refuses the stage when
        # no checkpoint scored at all.
        assert_generations_usable(generations, what=f"checkpoint {checkpoint}")
        again = None
        twice = repeat_rows(rows, settings["repeat_modes"]) if settings["repeat"] else []
        if twice:
            # The same rows - those of the repeat mode - in the reverse order:
            # what run-to-run agreement compares the first reading with. Its
            # cost rides on the first's.
            repeat_stats: dict[str, Any] = {}
            again = generate_validation(twice, model, adapter=checkpoint, stats=repeat_stats, order="reversed")
            record["repeat"] = repeat_stats
            assert_generations_usable(again, what=f"checkpoint {checkpoint}'s second pass")
        return score_pass(generations, checkpoint=checkpoint, repeat=again, finalists=settings["finalists"],
                          save_dir=settings["save_dir"], val_path=settings["val_path"], saved=saved,
                          settings=generation_settings(getattr(model, "config", None)))

    # Called by the checkpoint stage once selection is done, to free the engine.
    score.close = close  # type: ignore[attr-defined]
    score.passes = passes  # type: ignore[attr-defined]
    score.configure = configure  # type: ignore[attr-defined]
    #: {checkpoint: where its finalists' pass was saved} (:func:`saved_pass_paths`).
    score.saved = saved  # type: ignore[attr-defined]
    return score


def repeat_rows(rows: Sequence[dict[str, Any]], modes: Sequence[str] | None) -> list[dict[str, Any]]:
    """The rows a pass reads a second time: those read in ``modes`` - each
    document whole in them - or every row when none are named."""
    if not modes:
        return list(rows)
    wanted = set(modes)
    return [row for row in rows if row.get("modality_mode") in wanted]


def pass_rows(rows: Sequence[dict[str, Any]], *, sample_rows: int, full: bool,
              full_modes: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The rows one scoring pass generates, in order, as :func:`vllm_scorer`
    builds it: every row with no sample; else the sample, or - for ``full``,
    the finalists' pass, when the sample is smaller than the split - the
    split's rows in ``full_modes``. One definition, so a parent and the
    workers it shards a pass across read the same rows in the same order."""
    if not sample_rows:
        return list(rows)
    from evaluation.validation_sample import validation_sample

    sample = validation_sample(rows, sample_rows)
    if len(sample) >= len(rows):
        return sample
    return rows_in_modes(rows, full_modes) if full else sample


def saved_pass_paths(save_dir: Path | str, checkpoint: str) -> dict[str, Path]:
    """Where a finalist's pass is saved under ``save_dir``: its answers
    (``generations``, :func:`validation_generation.write_generations`) and its
    error tallies (``tallies``). Named by the checkpoint, so a parent finds
    what a worker saved."""
    import hashlib

    name = f"{Path(str(checkpoint).rstrip('/')).name}-{hashlib.sha1(str(checkpoint).encode()).hexdigest()[:8]}"
    root = Path(save_dir)
    return {"generations": root / f"{name}.generations.jsonl", "tallies": root / f"{name}.validation_errors.json"}


def score_pass(
    generations: list[Any], *, checkpoint: str, repeat: list[Any] | None = None, finalists: bool = False,
    save_dir: Path | str | None = None, val_path: str | None = None, settings: dict[str, Any] | None = None,
    saved: dict[str, dict[str, str]] | None = None,
) -> dict[str, float]:
    """One checkpoint's metrics from its generations: the same for a pass
    generated in one process and one sharded across GPUs (:func:`generate_sharded`).

    The finalists' pass (``finalists``) also carries the gate mirror
    (:func:`validation_generation.gate_mirror`); with ``save_dir`` its answers,
    logprobs included, and its error tallies are written there
    (:func:`saved_pass_paths`) and noted in ``saved``: the selected one's are
    what calibration fits from and what the error tallies report. Either
    failing is logged, never fails the score.
    """
    from evaluation.validation_generation import (
        error_tallies,
        gate_mirror,
        score_generations,
        write_generations,
    )

    metrics ={k: float(v) for k, v in score_generations(generations, model_version=checkpoint, repeat=repeat).items()
               if isinstance(v, (int, float))}
    if not finalists:
        return metrics
    try:
        metrics.update(gate_mirror(generations, model_version=checkpoint))
    except Exception as exc:  # noqa: BLE001 - the finalists are then judged on the documents measure
        log.warning("could not score %s's gate mirror; its document field match stands in: %s", checkpoint, exc)
    if save_dir is None:
        return metrics
    files = saved_pass_paths(save_dir, checkpoint)
    try:
        header = {"version": 1, "checkpoint": checkpoint, "settings": dict(settings or {}), "val_path": val_path,
                  "modes": sorted({str(g.row.get("modality_mode") or "ocr_plus_image") for g in generations}),
                  "rows": len(generations), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_generations(files["generations"], header, generations)
        files["tallies"].write_text(json.dumps({"checkpoint": checkpoint, **error_tallies(generations)}),
                                    encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - calibration then generates again
        log.warning("could not save %s's finalist pass under %s: %s", checkpoint, save_dir, exc)
        return metrics
    if saved is not None:
        saved[checkpoint] = {name: str(path) for name, path in files.items()}
    return metrics


def saved_finalist_pass(scorer: Any, checkpoint: str | None) -> dict[str, str] | None:
    """Where ``checkpoint``'s finalist pass was saved, by the scorer that
    judged the finalists (``.full``, or the scorer itself); None when none was."""
    if not checkpoint:
        return None
    for candidate in (getattr(scorer, "full", None), scorer):
        found = (getattr(candidate, "saved", None) or {}).get(checkpoint)
        if found:
            return dict(found)
    return None


def generation_cost(scorer: Any) -> dict[str, Any]:
    """The generation passes a selection made - the sample's and ``.full``'s -
    and their totals: what selection's time went on. The parallel scorer's come
    back from its workers. Empty for a scorer that keeps none."""
    passes = [*getattr(scorer, "passes", []), *getattr(getattr(scorer, "full", None), "passes", [])]
    if not passes:
        return {}
    # A pass read twice for run-to-run agreement carries its second reading.
    readings = [reading for p in passes for reading in (p, p.get("repeat") or {})]
    return {
        "passes": passes,
        "rows": sum(int(r.get("generated", 0)) for r in readings),
        "seconds": round(sum(float(r.get("seconds", 0.0)) for r in readings), 1),
    }


def break_tie(report: SelectionReport, full_scorer: Scorer, margin: float) -> SelectionReport:
    """Re-decide a near-tie on the full validation split.

    The candidates were ranked on the validation sample. When the top two are
    closer than ``margin``, sample noise can order them either way, so both are
    scored again on every validation row and that result decides. Further apart,
    the sample's order stands.
    """
    if len(report.scores) < 2 or report.margin >= margin:
        return report
    first, second = sorted(report.scores, key=lambda s: (s.field_f1, s.step), reverse=True)[:2]
    outcomes = _score_all(full_scorer, [first.checkpoint, second.checkpoint])
    full: dict[str, float] = {}
    for candidate in (first, second):
        outcome = (outcomes[candidate.checkpoint] if outcomes is not None
                   else full_scorer(candidate.checkpoint))
        if isinstance(outcome, BaseException):
            raise outcome
        full[candidate.checkpoint] = float(dict(outcome).get("field_normalized_match", 0.0))
    winner = max((first, second), key=lambda s: (full[s.checkpoint], s.step))
    report.tie_break = {
        "margin_on_sample": report.margin, "threshold": margin,
        "full_validation_field_f1": {c: round(v, 4) for c, v in full.items()},
        "decided": winner.checkpoint, "changed_choice": winner.checkpoint != report.selected,
    }
    log.info("near-tie (%.4f < %.4f) broken on the full validation split: %s",
             report.margin, margin, winner.checkpoint)
    report.selected = winner.checkpoint
    return report


#: Document-level metrics the finalists are judged on (validation_generation.score_generations).
DOCUMENT_ACCURACY = "document_field_normalized_match"
#: Values no page prints: of the finalists close enough on accuracy, the one
#: inventing the fewest wins (decision 2026-10-24). Over every mode the pass
#: reads: an image_only answer is checked on its document's page text borrowed
#: from the same document's ocr_plus_image rows (decision 2026-10-26).
DOCUMENT_INVENTION = "document_hallucination_rate"
#: What that rate could not see, recorded beside it per finalist
#: (validation_generation._invention_coverage): document readings checked on
#: borrowed text, the pages they cite with none, and readings left unscored -
#: no text to borrow, counted, never as nothing invented.
DOCUMENT_INVENTION_COVERAGE = (
    "document_hallucination_borrowed_text_readings",
    "document_hallucination_cited_pages_without_text",
    "document_hallucination_unscored_readings",
)
#: Lower is better: a finalist more than the guard margin above the best
#: finalist's does not win.
DOCUMENT_GUARDS = ("document_false_null_rate",)
#: Higher is better: the same documents read twice, the second time in the
#: reverse order. A finalist more than the guard margin below the best
#: finalist's does not read the same twice, and does not win.
DOCUMENT_AGREEMENT = "document_run_to_run_agreement"
#: The finalists' field match as the gate measures it (validation_generation.gate_mirror):
#: image_only readings of real documents, whole, held to their borrowed page
#: text as served. The field-match measure of the decision when every
#: finalist has one; else :data:`DOCUMENT_ACCURACY` stands in, and the
#: decision says so.
GATE_MIRROR = "gate_mirror_accuracy"
#: What the gate mirror is over, recorded beside it per finalist.
GATE_MIRROR_COVERAGE = ("gate_mirror_documents", "gate_mirror_unscored_documents",
                        "gate_mirror_synthetic_documents", "gate_mirror_hallucination_rate")


def rows_in_modes(rows: list[dict[str, Any]], modes: Sequence[str] | None) -> list[dict[str, Any]]:
    """The validation rows read in ``modes``; every row when none are named.

    Filtered by reading mode, never by row: each document keeps all its windows
    in a mode, so the whole-document merge (validation_generation.merged_windows)
    sees every document whole. A filter that leaves nothing is refused - judging
    the finalists on no documents would decide nothing.
    """
    if not modes:
        return list(rows)
    wanted = set(modes)
    kept = [row for row in rows if row.get("modality_mode") in wanted]
    if not kept:
        raise CheckpointEvalError(
            f"no validation row is read in {sorted(wanted)}: the finalists would be judged on no "
            "documents. Check selection_document_modes against the split's modality_mode values."
        )
    from evaluation.run_eval import IMAGE_ONLY_MODE, TEXT_DONOR_MODE

    if IMAGE_ONLY_MODE in wanted and TEXT_DONOR_MODE not in wanted:
        # An image_only answer is checked for invention on page text borrowed
        # from its document's ocr_plus_image rows (decision 2026-10-26).
        log.warning("selection reads %s without %s: its image_only answers have no page text to borrow, "
                    "so their values are unscored for invention", sorted(wanted), TEXT_DONOR_MODE)
    return kept


def choose_on_documents(
    report: SelectionReport, full_scorer: Scorer | None, *, finalists: int, guard_margin: float,
    modes: Sequence[str] | None = None,
) -> SelectionReport:
    """Re-decide among the ``finalists`` best candidates on whole validation documents.

    The candidates were ranked window by window on the validation sample - a few
    hundred windows, a few dozen documents. The finalists are scored on every
    validation row (``full_scorer``; ``None`` when the sample already is the
    whole split, and its scores are used as they are) over whole documents -
    windows merged as serving merges them - and decided in three steps
    (decision 2026-10-24: the model must invent nothing, and read the same
    document the same twice):

    1. Guards. A finalist whose rate of printed values left empty is more than
       ``guard_margin`` above the best finalist's, or whose run-to-run
       agreement - where the scorer read the documents twice
       (``selection_run_to_run_check``) - is more than ``guard_margin`` below
       the best finalist's, does not win.
    2. Field match. Of the rest, those within ``guard_margin`` of the best
       one's field match go on. A finalist the guards excluded sets no
       bar: it cannot win. Field match is the gate mirror (:data:`GATE_MIRROR`):
       image_only readings of real documents, whole, held to their borrowed
       page text as served - what the gate measures. Where a finalist has none
       (no real image_only document with text to borrow was read), document
       accuracy over every mode read stands in for all of them, and the
       decision records the fallback (``field_match_fallback``).
    3. Invention. Of those, the one inventing the fewest values wins
       (:data:`DOCUMENT_INVENTION`); ties go to the higher field match, then to
       the later step, as in :func:`select_best`. A finalist with nothing
       checkable has no rate and ranks after one that has. The rate covers
       every mode read - image_only answers on page text borrowed from their
       document's ocr_plus_image rows - and what it could not check is
       recorded beside it (:data:`DOCUMENT_INVENTION_COVERAGE`).

    Guards that between them exclude every finalist exclude none. Without
    whole-document scores (no policy windows) the window-level metrics stand
    in, as they do for accuracy.

    ``modes`` names the reading modes ``full_scorer`` reads (``selection_document_modes``;
    none = all three), recorded with the decision.
    """
    ranked = sorted(report.scores, key=lambda s: (s.field_f1, s.step), reverse=True)[:finalists]
    if len(ranked) < 2:
        return report
    metrics: dict[str, dict[str, float]] = {}
    if full_scorer is None:
        metrics = {c.checkpoint: dict(c.metrics) for c in ranked}
    else:
        outcomes = _score_all(full_scorer, [c.checkpoint for c in ranked])
        for candidate in ranked:
            try:
                outcome = (outcomes[candidate.checkpoint] if outcomes is not None
                           else full_scorer(candidate.checkpoint))
                if isinstance(outcome, BaseException):
                    raise outcome
                metrics[candidate.checkpoint] = dict(outcome)
            except Exception as exc:  # noqa: BLE001 - a finalist that cannot be scored drops out
                report.skipped.append((candidate.checkpoint, f"document scoring: {type(exc).__name__}: {exc}"))
                log.warning("could not score %s on whole documents: %s", candidate.checkpoint, exc)
    scored = [c for c in ranked if c.checkpoint in metrics]
    if len(scored) < 2:
        return report

    def metric(c: CheckpointScore, name: str, stand_in: str | None = None) -> float | None:
        m = metrics[c.checkpoint]
        value = m.get(name, m.get(stand_in) if stand_in else None)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    # The gate mirror when every finalist has one: one measure for all, never mixed.
    mirror = {c.checkpoint: v for c in scored if (v := metric(c, GATE_MIRROR)) is not None}
    measure = GATE_MIRROR if len(mirror) == len(scored) else DOCUMENT_ACCURACY
    fallback = None if measure == GATE_MIRROR else (
        "no real image_only document with page text to borrow was read for "
        + ("every finalist" if mirror else "any finalist")
        + ": field match is document accuracy over every mode read (window gold, answers not held to "
        "page text, synthetic twins included), not the gate's measure")
    if fallback:
        log.warning("selection: %s", fallback)

    def accuracy(c: CheckpointScore) -> float:
        if measure == GATE_MIRROR:
            return mirror[c.checkpoint]
        return metric(c, DOCUMENT_ACCURACY, "field_normalized_match") or 0.0

    # 1. Guards: each against the best finalist on it.
    failed: dict[str, list[str]] = {}
    for guard in DOCUMENT_GUARDS:
        values = {c.checkpoint: v for c in scored if (v := metric(c, guard)) is not None}
        for checkpoint, value in values.items():
            if value > min(values.values()) + guard_margin:
                failed.setdefault(checkpoint, []).append(guard)
    agreement = {c.checkpoint: v for c in scored
                 if (v := metric(c, DOCUMENT_AGREEMENT, "run_to_run_agreement")) is not None}
    for checkpoint, value in agreement.items():
        if value < max(agreement.values()) - guard_margin:
            failed.setdefault(checkpoint, []).append(DOCUMENT_AGREEMENT)
    guarded = [c for c in scored if c.checkpoint not in failed] or scored
    # 2. Field match within the margin of the best that can win.
    bar = max(accuracy(c) for c in guarded) - guard_margin
    close = [c for c in guarded if accuracy(c) >= bar]

    # 3. The fewest invented values; then field match; then the later step.
    def rank(c: CheckpointScore) -> tuple[bool, float, float, int]:
        invented = metric(c, DOCUMENT_INVENTION, "hallucination_rate")
        return invented is not None, -(invented or 0.0), accuracy(c), c.step

    winner = max(close, key=rank)
    unscored = {c.checkpoint: int(v) for c in scored
                if (v := metric(c, "document_hallucination_unscored_readings", "hallucination_unscored_readings"))}
    if unscored:
        log.warning("some finalists' readings had no page text, own or borrowed, so their values were not "
                    "checked for invention (unscored, not clean): %s", unscored)
    report.document_choice = {
        "finalists": {c.checkpoint: {k: round(float(metrics[c.checkpoint][k]), 4)
                                     for k in (DOCUMENT_ACCURACY, GATE_MIRROR, DOCUMENT_INVENTION, *DOCUMENT_GUARDS,
                                               DOCUMENT_AGREEMENT, *DOCUMENT_INVENTION_COVERAGE,
                                               *GATE_MIRROR_COVERAGE)
                                     if k in metrics[c.checkpoint]}
                      for c in scored},
        "guard_margin": guard_margin,
        # What "within the guard margin on field match" read: the gate mirror,
        # or - with none to read - document accuracy, and why.
        "field_match_measure": measure,
        "field_match_fallback": fallback,
        # What the decision read: the whole-document pass's modes, or - when the
        # split is its own sample and no pass ran - the sample, in every mode.
        "modes": (list(modes) if modes else "all") if full_scorer is not None else "sample (all modes)",
        # Guards that failed every finalist excluded none (guarded is then all).
        "excluded_by_guard": [c.checkpoint for c in scored if c not in guarded],
        "guards_failed": {c.checkpoint: failed[c.checkpoint] for c in scored if c.checkpoint in failed},
        "behind_on_field_match": [c.checkpoint for c in guarded if c not in close],
        # Whether the finalists were read twice: off (selection_run_to_run_check),
        # or no pass to read twice, and the agreement guard did not apply.
        "run_to_run_measured": bool(agreement),
        "decided": winner.checkpoint,
        "changed_choice": winner.checkpoint != report.selected,
    }
    if winner.checkpoint != report.selected:
        log.info("whole validation documents changed the choice: %s -> %s", report.selected, winner.checkpoint)
    report.selected = winner.checkpoint
    return report


def vllm_scorer(
    *, client: Any, val_path: str, model: Any = None, images_root: str | None = None,
    sample_rows: int = 0, full_modes: Sequence[str] | None = None, repeat: bool = False,
    repeat_modes: Sequence[str] | None = None, save_dir: Path | str | None = None,
) -> Scorer:
    """A :func:`generation_scorer` over the stored validation split, on the base.

    The base is loaded in bf16 with LoRA enabled and each checkpoint applied as a
    decoder LoRA: one LoRA per request is exactly what vLLM supports, so no
    checkpoint needs merging to be scored.

    ``full_modes`` narrows ``.full`` - the finalists' whole-document pass and the
    near-tie - to those reading modes (:func:`rows_in_modes`); the sample keeps
    every mode. On the CGL run one mode was a third of the ~6,170 rows per finalist.

    ``repeat``: the pass the finalists are judged on (:func:`choose_on_documents`)
    reads its rows twice, the second time in reverse order, for the run-to-run
    guard (:func:`generation_scorer`): ``.full``, or - for a split that is its
    own sample, whose scores judge the finalists - the scorer itself - over the
    rows of ``repeat_modes`` (none = every row).

    That pass is the finalists' (:func:`score_pass`): its metrics carry the
    gate mirror, and with ``save_dir`` each finalist's answers and error
    tallies are saved there.
    """
    from evaluation.validation_generation import read_rows

    def scorer(part: list[dict[str, Any]]) -> Scorer:
        score = generation_scorer(part, model, repeat=True) if repeat else generation_scorer(part, model)
        configure = getattr(score, "configure", None)
        if callable(configure):
            configure(finalists=True, repeat_modes=list(repeat_modes or []),
                      save_dir=Path(save_dir) if save_dir is not None else None, val_path=val_path)
        return score

    rows = read_rows(client.read_text(val_path))
    # Checked before the engine loads: a mode filter that matches no row is a
    # configuration error, and an engine loaded first was never released.
    if sample_rows:
        rows_in_modes(rows, full_modes)
    if images_root is not None:
        # vLLM opens images by path; the rows hold Blob keys.
        from training.stage_data import localize_rows

        rows = localize_rows(rows, client, images_root)
    if model is None:  # pragma: no cover - needs a GPU
        from inference_core.model_runner import load_model

        model = load_model("base", client)
    if not sample_rows:
        return scorer(rows)
    # Candidates are ranked on the fixed validation sample; ``.full`` - the
    # finalists' whole-document pass and the near-tie (break_tie) - scores every
    # row of the named modes. One engine serves both. Only a split that is its
    # own sample has no ``.full``: its finalists are judged on the sample, which
    # holds every row already.
    from evaluation.validation_sample import validation_sample

    sample = validation_sample(rows, sample_rows)
    if len(sample) >= len(rows):
        return scorer(sample)
    score = generation_scorer(sample, model)
    score.full = scorer(rows_in_modes(rows, full_modes))  # type: ignore[attr-defined]
    return score


def _score_all(scorer: Any, checkpoints: list[str]) -> dict[str, Any] | None:
    """Every checkpoint's metrics (or the exception scoring it raised) from one
    call, when the scorer can score several at once; ``None`` when it cannot."""
    many = getattr(scorer, "score_many", None)
    return many(checkpoints) if callable(many) else None


def scoring_gpus() -> list[str]:
    """The GPUs this process may score on, as ``CUDA_VISIBLE_DEVICES`` ids: those it
    is limited to, or every one torch counts. Empty off a GPU machine."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [d.strip() for d in visible.split(",") if d.strip() and d.strip() != "-1"]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 - no torch / no CUDA: nothing to score on in parallel
        return []


def _spawn(command: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
    """Start one scoring worker, its output in ``log_path``. A seam for tests."""
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
    process.log_handle = handle  # type: ignore[attr-defined] - closed once the worker ends
    return process


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


@dataclass
class ParallelScorer:
    """Scores checkpoints at once, one worker process per GPU.

    Each worker (``evaluation.checkpoint_score_worker``) sees one GPU and builds
    the in-process scorer there - :func:`vllm_scorer`, the same engine, rows and
    scoring - so a candidate scores as it would alone. Candidates are dealt to
    the GPUs in turn; a worker that fails fails only the candidates it held, each
    with the end of its log.

    **A count of candidates that does not divide among the GPUs fills them all**
    - the finalists' pass is two candidates on a four-GPU pod, and dealt, two
    GPUs sat idle. Each candidate's pass is then sharded across every GPU
    (:func:`generate_sharded`): each worker makes its share of every
    candidate's calls - the calls, rows and order the pass makes unsharded -
    and this process scores the assembled generations with :func:`score_pass`,
    as a worker scores them. Sharded, a worker that fails fails every candidate
    whose calls it held.
    """

    val_path: str
    images_root: str | None
    sample_rows: int
    gpus: list[str]
    work_dir: Path
    #: Score every validation row, not the sample (the near-tie's ``.full``).
    full_split: bool = False
    #: The reading modes the full split is narrowed to (:func:`rows_in_modes`).
    full_modes: list[str] = field(default_factory=list)
    #: The finalists' pass is read twice, for run-to-run agreement (:func:`vllm_scorer`)...
    repeat: bool = False
    #: ...the second time only its rows in these modes (none = every row).
    repeat_modes: list[str] = field(default_factory=list)
    poll_seconds: float = 15.0
    #: What each worker's generation passes cost, as the workers report them
    #: (:func:`generation_cost`).
    passes: list[dict[str, Any]] = field(default_factory=list)
    #: This scorer's pass is the one the finalists are judged on (:func:`score_pass`).
    finalists: bool = False
    #: Where that pass saves each finalist's answers and error tallies.
    save_dir: Path | None = None
    #: The validation split's rows, as stored: what a sharded pass is put back
    #: together on. Without them every pass is dealt.
    rows: list[dict[str, Any]] | None = field(default=None, repr=False, compare=False)
    #: {checkpoint: where its finalist pass was saved} (:func:`saved_pass_paths`).
    saved: dict[str, dict[str, str]] = field(default_factory=dict)

    def __call__(self, checkpoint: str) -> dict[str, float]:
        outcome = self.score_many([checkpoint])[checkpoint]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        """Nothing to free here: each worker's engine ended with the worker."""

    def sharded(self, checkpoints: Sequence[str]) -> bool:
        """Whether ``checkpoints`` are sharded across the GPUs rather than dealt:
        when they do not divide evenly among them, so dealing would leave a GPU
        idle, or one GPU a candidate more than the rest."""
        if self.rows is None or len(self.gpus) < 2 or not checkpoints or len(checkpoints) % len(self.gpus) == 0:
            return False
        from evaluation.validation_generation import reading_key

        keys = [reading_key(row) for row in self._pass_rows()]
        return len(set(keys)) == len(keys)

    def _pass_rows(self) -> list[dict[str, Any]]:
        return pass_rows(self.rows or [], sample_rows=self.sample_rows, full=self.full_split,
                         full_modes=self.full_modes)

    def score_many(self, checkpoints: list[str]) -> dict[str, Any]:
        checkpoints = list(dict.fromkeys(checkpoints))
        if self.sharded(checkpoints):
            return self._score_sharded(checkpoints)
        lanes = self.gpus[: max(1, len(checkpoints))]
        shares = {gpu: checkpoints[index::len(lanes)] for index, gpu in enumerate(lanes)}
        self.work_dir.mkdir(parents=True, exist_ok=True)
        kind = "full" if self.full_split else "sample"
        running = {}
        for gpu, share in shares.items():
            if not share:
                continue
            out = self.work_dir / f"scores-{kind}-gpu{gpu}.json"
            out.unlink(missing_ok=True)
            command = [sys.executable, "-m", "evaluation.checkpoint_score_worker",
                       "--val-path", self.val_path, "--sample-rows", str(self.sample_rows),
                       "--out", str(out)]
            if self.images_root:
                command += ["--images-root", str(self.images_root)]
            if self.full_split:
                command.append("--full")
                for mode in self.full_modes:
                    command += ["--full-mode", mode]
            if self.repeat:
                command.append("--repeat")
                for mode in self.repeat_modes:
                    command += ["--repeat-mode", mode]
            if self.finalists and self.save_dir is not None:
                command += ["--save-dir", str(self.save_dir)]
            for checkpoint in share:
                command += ["--checkpoint", checkpoint]
            # One GPU each; the worker scores in place - it must not re-launch
            # itself into tmux as the pod's long jobs do.
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "FIDEON_NO_DETACH": "1"}
            log_path = self.work_dir / f"scores-{kind}-gpu{gpu}.log"
            running[gpu] = (share, out, log_path, _spawn(command, env, log_path))
        log.info("scoring %d checkpoint(s) on %d GPU(s) at once (%s rows); logs: %s",
                 len(checkpoints), len(running), kind, self.work_dir)

        outcomes: dict[str, Any] = {}
        while running:
            for gpu in [g for g, (*_, proc) in running.items() if proc.poll() is not None]:
                share, out, log_path, proc = running.pop(gpu)
                if getattr(proc, "log_handle", None) is not None:
                    proc.log_handle.close()
                try:
                    results = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
                except ValueError:
                    results = {}
                for checkpoint in share:
                    result = results.get(checkpoint)
                    if result and result.get("stats"):
                        self.passes.append({"checkpoint": checkpoint, "gpu": gpu, **result["stats"]})
                    if result and "metrics" in result:
                        outcomes[checkpoint] = result["metrics"]
                        log.info("scored %s on GPU %s: field F1 %.4f", checkpoint, gpu,
                                 float(result["metrics"].get("field_normalized_match", 0.0)))
                    else:
                        why = (result or {}).get("error") or (
                            f"the worker on GPU {gpu} exited {proc.returncode} without a score")
                        outcomes[checkpoint] = CheckpointEvalError(f"{why}\n{_tail(log_path)}")
            if running:
                time.sleep(self.poll_seconds)
        if self.finalists and self.save_dir is not None:
            # What each worker saved of its finalists' passes, found by name.
            for checkpoint in checkpoints:
                files = saved_pass_paths(self.save_dir, checkpoint)
                if not isinstance(outcomes.get(checkpoint), BaseException) and all(
                        path.is_file() for path in files.values()):
                    self.saved[checkpoint] = {name: str(path) for name, path in files.items()}
        return outcomes

    def _score_sharded(self, checkpoints: list[str]) -> dict[str, Any]:
        from evaluation.validation_generation import assert_generations_usable

        made = generate_sharded(
            rows=self._pass_rows(), val_path=self.val_path, images_root=self.images_root,
            sample_rows=self.sample_rows, full=self.full_split, full_modes=self.full_modes,
            checkpoints=checkpoints, gpus=self.gpus, work_dir=self.work_dir,
            kind="full" if self.full_split else "sample",
            # Read twice only where a worker's own scorer reads twice: the finalists' pass.
            repeat=self.repeat and self.finalists, repeat_modes=self.repeat_modes,
            poll_seconds=self.poll_seconds)
        outcomes: dict[str, Any] = {}
        for checkpoint in checkpoints:
            done = made[checkpoint]
            self.passes.extend({"checkpoint": checkpoint, **stats} for stats in done.stats)
            if done.error is not None:
                outcomes[checkpoint] = CheckpointEvalError(done.error)
                continue
            try:
                assert_generations_usable(done.generations, what=f"checkpoint {checkpoint}")
                if done.repeat is not None:
                    assert_generations_usable(done.repeat, what=f"checkpoint {checkpoint}'s second pass")
                outcomes[checkpoint] = score_pass(
                    done.generations, checkpoint=checkpoint, repeat=done.repeat, finalists=self.finalists,
                    save_dir=self.save_dir, val_path=self.val_path, settings=done.settings, saved=self.saved)
                log.info("scored %s across %d GPU(s): field F1 %.4f", checkpoint, len(self.gpus),
                         outcomes[checkpoint].get("field_normalized_match", 0.0))
            except Exception as exc:  # noqa: BLE001 - recorded per checkpoint, as select_best does
                outcomes[checkpoint] = exc
        return outcomes


@dataclass
class ShardedPass:
    """One checkpoint's pass, generated across GPUs and put back together."""

    checkpoint: str
    #: The pass's generations in its row order (:func:`pass_rows`); None when
    #: a shard is missing.
    generations: list[Any] | None = None
    #: The second reading of the rows of the repeat modes, in their order; None without one.
    repeat: list[Any] | None = None
    #: The generation settings every shard reported (validation_generation.generation_settings).
    settings: dict[str, Any] | None = None
    #: Each shard's cost: gpu, rows, calls, seconds, and its second reading's.
    stats: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def generate_sharded(
    *, rows: list[dict[str, Any]], val_path: str, images_root: str | None, sample_rows: int, full: bool,
    full_modes: Sequence[str] | None, checkpoints: Sequence[str], gpus: Sequence[str], work_dir: Path,
    kind: str, repeat: bool = False, repeat_modes: Sequence[str] | None = None,
    only_keys: Sequence[str] | None = None, prepare_adapters: bool = False, poll_seconds: float = 15.0,
) -> dict[str, ShardedPass]:
    """Each checkpoint's pass over ``rows`` generated across every GPU at once.

    ``rows`` is the pass as this process builds it (:func:`pass_rows` over the
    stored split, narrowed to the readings ``only_keys`` names when given -
    ``validation_generation.reading_key``); each worker builds the same from
    ``val_path`` and makes shard ``k`` of ``n`` of every checkpoint's calls
    (``generate_validation(shard=)``), and with ``repeat`` of the second
    reading's over the rows of ``repeat_modes``. The answers come back by
    reading, are checked against their rows (``row_digest``) and put back in
    row order: the generations the pass makes in one process. A checkpoint
    with a shard missing, a row unanswered, or shards that generated under
    different settings carries an ``error`` instead.
    """
    from evaluation.validation_generation import reading_key, restored_generation, row_digest

    keys = [reading_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise CheckpointEvalError("two validation rows are the same reading, so a pass cannot be sharded by reading")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    keys_file = None
    if only_keys is not None:
        keys_file = work_dir / f"shard-{kind}-keys.json"
        keys_file.write_text(json.dumps(sorted(set(only_keys))), encoding="utf-8")
    running: dict[str, tuple[Path, Path, Any]] = {}
    for index, gpu in enumerate(gpus):
        out = work_dir / f"shard-{kind}-gpu{gpu}.json"
        out.unlink(missing_ok=True)
        command = [sys.executable, "-m", "evaluation.checkpoint_score_worker", "--val-path", val_path,
                   "--sample-rows", str(sample_rows), "--out", str(out),
                   "--shard", str(index), "--shards", str(len(gpus))]
        if images_root:
            command += ["--images-root", str(images_root)]
        if full:
            command.append("--full")
            for mode in full_modes or []:
                command += ["--full-mode", mode]
        if repeat:
            command.append("--repeat")
            for mode in repeat_modes or []:
                command += ["--repeat-mode", mode]
        if keys_file is not None:
            command += ["--only-keys", str(keys_file)]
        if prepare_adapters:
            command.append("--prepare-adapter")
        for checkpoint in checkpoints:
            command += ["--checkpoint", checkpoint]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "FIDEON_NO_DETACH": "1"}
        log_path = work_dir / f"shard-{kind}-gpu{gpu}.log"
        running[gpu] = (out, log_path, _spawn(command, env, log_path))
    log.info("generating %d checkpoint(s)' %s pass of %d row(s), sharded across %d GPU(s); logs: %s",
             len(checkpoints), kind, len(rows), len(running), work_dir)

    finished: dict[str, tuple[dict[str, Any], Path, Any]] = {}
    while running:
        for gpu in [g for g, (*_, proc) in running.items() if proc.poll() is not None]:
            out, log_path, proc = running.pop(gpu)
            if getattr(proc, "log_handle", None) is not None:
                proc.log_handle.close()
            try:
                results = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
            except ValueError:
                results = {}
            finished[gpu] = (results, log_path, proc.returncode)
        if running:
            time.sleep(poll_seconds)

    position = {key: index for index, key in enumerate(keys)}
    twice = repeat_rows(rows, repeat_modes) if repeat else []
    twice_position = {reading_key(row): index for index, row in enumerate(twice)}

    def place(into: list[Any], on: list[dict[str, Any]], where: dict[str, int], records: Path) -> list[str]:
        problems = []
        for line in records.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            index = where.get(record.get("key"))
            if index is None or record.get("digest") != row_digest(on[index]):
                problems.append(f"an answer to a reading this pass does not hold: {record.get('key')}")
                continue
            into[index] = restored_generation(on[index], record)
        return problems

    passes: dict[str, ShardedPass] = {}
    for checkpoint in checkpoints:
        done = ShardedPass(checkpoint)
        first: list[Any] = [None] * len(rows)
        again: list[Any] = [None] * len(twice)
        problems: list[str] = []
        for gpu, (results, log_path, code) in sorted(finished.items()):
            entry = results.get(checkpoint) or {}
            if "first" not in entry:
                why = entry.get("error") or f"the worker on GPU {gpu} exited {code} without its share"
                problems.append(f"GPU {gpu}: {why}\n{_tail(log_path)}")
                continue
            done.stats.append({"gpu": gpu, **(entry.get("stats") or {}),
                               **({"repeat": entry["repeat_stats"]} if entry.get("repeat_stats") else {})})
            if done.settings is None:
                done.settings = entry.get("settings")
            elif entry.get("settings") != done.settings:
                problems.append(f"GPU {gpu} generated with other settings: {entry.get('settings')} != {done.settings}")
            try:
                problems += place(first, rows, position, Path(entry["first"]))
                if twice and entry.get("repeat"):
                    problems += place(again, twice, twice_position, Path(entry["repeat"]))
            except (OSError, ValueError) as exc:
                problems.append(f"GPU {gpu}: its answers could not be read: {exc}")
        missing = sum(g is None for g in first) + sum(g is None for g in again)
        if missing and not problems:
            problems.append(f"{missing} row(s) came back from no shard")
        if problems:
            done.error = f"{checkpoint}: the sharded pass is incomplete - " + "; ".join(problems[:3])
        else:
            done.generations, done.repeat = first, (again if twice else None)
        passes[checkpoint] = done
    return passes


def generate_shard(
    *, client: Any, val_path: str, images_root: str | None, sample_rows: int, full: bool,
    full_modes: Sequence[str] | None, checkpoints: Sequence[str], shard: int, shards: int, out: Path,
    repeat: bool = False, repeat_modes: Sequence[str] | None = None, only_keys: Sequence[str] | None = None,
    prepare_adapters: bool = False, model: Any = None,
) -> dict[str, Any]:
    """One worker's share of a sharded pass (:func:`generate_sharded`): the
    pass's rows built as the parent builds them, shard ``shard`` of ``shards``
    of each checkpoint's calls generated, and the answers written beside
    ``out`` as JSON lines (``validation_generation.generation_record``).
    ``out`` lists, per checkpoint, those files, what they cost and the
    generation settings - or why it has none. Rewritten after each checkpoint.
    ``prepare_adapters``: a checkpoint that is no local directory is a Blob
    prefix, copied down first (calibration ``--from-blob``)."""
    from evaluation.validation_generation import (
        generate_validation,
        generation_record,
        generation_settings,
        read_rows,
        reading_key,
    )

    out = Path(out)
    rows = pass_rows(read_rows(client.read_text(val_path)), sample_rows=sample_rows, full=full,
                     full_modes=full_modes)
    if only_keys is not None:
        wanted = set(only_keys)
        rows = [row for row in rows if reading_key(row) in wanted]
    if images_root is not None:
        from training.stage_data import localize_rows

        rows = localize_rows(rows, client, images_root)
    if model is None:  # pragma: no cover - needs a GPU
        from inference_core.model_runner import load_model

        model = load_model("base", client)
    twice = repeat_rows(rows, repeat_modes) if repeat else []

    def write(index: int, name: str, generations: list[Any]) -> str:
        path = out.with_name(f"{out.stem}-c{index}-{name}.jsonl")
        path.write_text("".join(json.dumps(generation_record(g)) + "\n" for g in generations), encoding="utf-8")
        return str(path)

    results: dict[str, Any] = {}
    try:
        for index, checkpoint in enumerate(checkpoints):
            try:
                adapter = _local_adapter(checkpoint, client, model, prepare=prepare_adapters)
                stats: dict[str, Any] = {}
                first = generate_validation(rows, model, adapter=adapter, stats=stats, shard=(shard, shards))
                entry = {"first": write(index, "first", first), "stats": stats,
                         "settings": generation_settings(getattr(model, "config", None))}
                if twice:
                    repeat_stats: dict[str, Any] = {}
                    again = generate_validation(twice, model, adapter=adapter, stats=repeat_stats,
                                                order="reversed", shard=(shard, shards))
                    entry.update(repeat=write(index, "repeat", again), repeat_stats=repeat_stats)
                results[checkpoint] = entry
            except Exception as exc:  # noqa: BLE001 - recorded per checkpoint, as select_best does
                results[checkpoint] = {"error": f"{type(exc).__name__}: {exc}"}
            # Written after each, so a worker that dies later keeps what it made.
            out.write_text(json.dumps(results), encoding="utf-8")
    finally:
        from inference_core.model_runner import release_model

        release_model(model)
    return results


def _local_adapter(checkpoint: str, client: Any, model: Any, *, prepare: bool) -> str:
    """The adapter as the engine is handed it: as named (selection: a checkpoint
    directory, handed over as it always was), or - ``prepare``, calibration
    ``--from-blob`` - copied down once from a Blob prefix and checked
    (``model_runner.prepare_adapter``)."""
    if not prepare or Path(checkpoint).is_dir():
        return checkpoint
    from inference_core.model_runner import prepare_adapter

    return prepare_adapter(checkpoint, client, model.config)


def parallel_vllm_scorer(
    *, client: Any, val_path: str, images_root: str | None, sample_rows: int, gpus: list[str],
    work_dir: Path, full_modes: Sequence[str] | None = None, repeat: bool = False,
    repeat_modes: Sequence[str] | None = None, save_dir: Path | str | None = None,
) -> ParallelScorer:
    """A :class:`ParallelScorer` over the stored validation split, with a ``.full``
    one for the near-tie when the sample is smaller than the split.

    The page images are fetched here, once, into the shared cache, so the workers
    only read them: four fetching the same missing page at once could each see
    another's half-written file.

    The finalists' pass - ``.full``, or the scorer itself for a split that is its
    own sample - is read twice over ``repeat_modes`` (``repeat``), carries the
    gate mirror and saves each finalist's pass under ``save_dir`` (:func:`score_pass`).
    """
    from evaluation.validation_generation import read_rows
    from evaluation.validation_sample import validation_sample

    rows = read_rows(client.read_text(val_path))
    if sample_rows:
        rows_in_modes(rows, full_modes)        # refused here, before any worker starts
    if images_root is not None:
        from training.stage_data import localize_rows

        localize_rows(rows, client, images_root)
    settings = dict(val_path=val_path, images_root=images_root, sample_rows=sample_rows,
                    gpus=list(gpus), work_dir=Path(work_dir), full_modes=list(full_modes or []),
                    repeat=repeat, repeat_modes=list(repeat_modes or []),
                    save_dir=Path(save_dir) if save_dir is not None else None, rows=rows)
    has_full = bool(sample_rows) and len(validation_sample(rows, sample_rows)) < len(rows)
    scorer = ParallelScorer(**settings, finalists=not has_full)
    if has_full:
        scorer.full = ParallelScorer(**settings, full_split=True, finalists=True)  # type: ignore[attr-defined]
    return scorer


def discover_checkpoints(output_dir: str) -> tuple[list[str], str | None]:
    """The checkpoints a finished run left on disk, and the best-loss one.

    ms-swift writes under a versioned subdirectory (``output_dir/v0-<timestamp>/``),
    so the search is recursive; when a directory holds several runs, the most
    recently written one is this run. The best-loss checkpoint comes from the HF
    Trainer's ``trainer_state.json``, which is where early stopping recorded it.

    Returns ``([], None)`` when the directory does not exist — a dry run, or a
    trainer that saved nothing, which checkpoint selection then reports.
    """
    import json
    from pathlib import Path

    root = Path(output_dir)
    if not root.is_dir():
        return [], None
    found = [p for p in root.rglob("checkpoint-*") if p.is_dir() and _CHECKPOINT_STEP.search(p.name)]
    if not found:
        return [], None

    latest_run = max({p.parent for p in found}, key=lambda d: d.stat().st_mtime)
    checkpoints = sorted(
        (str(p) for p in found if p.parent == latest_run), key=checkpoint_step
    )

    best_loss = None
    state = Path(checkpoints[-1]) / "trainer_state.json"
    if state.exists():
        trainer_state = json.loads(state.read_text(encoding="utf-8"))
        recorded = trainer_state.get("best_model_checkpoint")
        if recorded:
            best_loss = str(latest_run / Path(recorded).name)
        # Candidates are the checkpoints an evaluation fell on, and the last one.
        # The others are resume points saved between evaluations (hourly): scoring
        # them all would multiply selection time without a loss to compare them by.
        #
        # Read from the evaluations the run logged, where it logged any: a run
        # resumed on another GPU count was given another cadence
        # (training.train._checkpoint_cadence), and a filter on the LAST
        # cadence alone kept only the checkpoints that happened to be its
        # multiples - dropping every earlier evaluation checkpoint. The
        # recorded eval_steps is the fallback for a state with no log.
        last = checkpoint_step(checkpoints[-1])
        evaluated = {
            int(entry["step"]) for entry in trainer_state.get("log_history") or []
            if isinstance(entry, dict) and isinstance(entry.get("step"), int)
            and any(str(key).startswith("eval_") for key in entry)
        }
        at_evaluations = [c for c in checkpoints if checkpoint_step(c) in evaluated]
        eval_steps = int(trainer_state.get("eval_steps") or 0)
        save_steps = int(trainer_state.get("save_steps") or 0)
        if at_evaluations:
            checkpoints = [c for c in checkpoints
                           if checkpoint_step(c) in evaluated or checkpoint_step(c) == last]
        elif eval_steps and save_steps and eval_steps > save_steps:
            checkpoints = [c for c in checkpoints
                           if checkpoint_step(c) % eval_steps == 0 or checkpoint_step(c) == last]
    return checkpoints, best_loss
