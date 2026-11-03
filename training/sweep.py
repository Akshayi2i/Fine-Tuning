"""The bounded 3-phase hyperparameter sweep (IMPL-06 §6, arch §11a).

Bounded is the whole design. An unbounded search over learning rate × epochs ×
rank is 27 runs before it has told you anything; this is ~9–12, ordered so each
phase is conditioned on the previous winner:

===========  ==================================  ====================  ==========
phase        sweeps                              metric                budget
===========  ==================================  ====================  ==========
1 · lr       highest impact, so it runs first    validation loss       3 / type
2 · epochs   with the Phase 1 winner fixed       validation field F1   3 / type
3 · rank     **only if F1 plateaus** after 1–2   validation field F1   3
===========  ==================================  ====================  ==========

**Phase 2 changes metric on purpose.** Validation loss can fall while field
extraction gets worse — the model becomes more confident about a distribution
that is not the one being scored — and field F1 is what the promotion gate reads.
Selecting epochs on loss would optimise a proxy for the thing that matters.

**Every candidate writes a full run manifest** with ``is_sweep_run: true``. Sweep
runs are first-class registry entries, not untracked side experiments: a config
promoted to production has to be traceable to the run that justified it, and
"we tried a few and this looked best" is not traceable.

**When to run it.** After the IMPL-15 pilot passes, before the first production
run. Sweeping against 25–30 documents per type measures noise: the test split is
3–4 documents, and the accuracy difference between 1e-4 and 2e-4 is smaller than
the variance from which documents landed in it. :func:`assert_enough_data` makes
that a refusal rather than a footnote.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config import load_yaml
from registry_utils.models import RunManifest

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "sweeps"

PHASES: tuple[str, ...] = ("lr", "epochs", "rank")
PHASE_FILES: dict[str, str] = {
    "lr": "phase1_lr.yaml",
    "epochs": "phase2_epochs.yaml",
    "rank": "phase3_rank.yaml",
}

#: Below this many documents per type, the sweep is measuring the split rather
#: than the hyperparameter. arch §8 calls pilot metrics "directional"; a sweep
#: needs them to be conclusive.
MIN_DOCUMENTS_PER_TYPE = 200


class SweepError(RuntimeError):
    """Raised when a sweep cannot run, or would not mean anything if it did."""


@dataclass
class Candidate:
    """One configuration to train."""

    phase: str
    adapter_type: str          # "foundation" | "per_type"
    parameter: str
    value: Any
    fixed: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"sweep-{self.phase}-{self.adapter_type}-{self.parameter}-{self.value}"

    def overrides(self) -> dict[str, Any]:
        """The training config this candidate implies."""
        return {**self.fixed, self.parameter: self.value}


@dataclass
class CandidateResult:
    """One trained candidate and its validation score."""

    candidate: Candidate
    metrics: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    manifest: RunManifest | None = None

    def score(self, metric: str) -> float | None:
        value = self.metrics.get(metric)
        return float(value) if isinstance(value, (int, float)) else None


@dataclass
class PhaseResult:
    """One phase: every candidate, and the winner."""

    phase: str
    metric: str
    goal: str
    results: list[CandidateResult] = field(default_factory=list)
    winner: CandidateResult | None = None
    skipped: str = ""

    @property
    def unmeasured(self) -> list[str]:
        return [r.candidate.run_id for r in self.results if r.score(self.metric) is None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "metric": self.metric,
            "goal": self.goal,
            "skipped": self.skipped,
            "candidates": [
                {"run_id": r.candidate.run_id, "parameter": r.candidate.parameter,
                 "value": r.candidate.value, "score": r.score(self.metric)}
                for r in self.results
            ],
            "unmeasured": self.unmeasured,
            "winner": self.winner.candidate.run_id if self.winner else None,
            "winning_value": self.winner.candidate.value if self.winner else None,
        }


def load_phase(phase: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    if phase not in PHASE_FILES:
        raise SweepError(f"unknown phase {phase!r}; expected one of {list(PHASES)}")
    return load_yaml(config_dir / PHASE_FILES[phase])


def assert_enough_data(documents_per_type: dict[str, int]) -> None:
    """Refuse to sweep at a volume where the result would be noise.

    Not a warning. A sweep that ran and picked a winner produces a config that
    looks justified, gets promoted as the production training config, and carries
    a manifest saying it won — all of which is true and none of which means the
    hyperparameter was better.
    """
    if not documents_per_type:
        # Not a vacuous pass. An empty dict is what a caller that failed to count
        # produces, and treating "I know of no thin type" as "no type is thin" is
        # the same absence-is-success mistake the rest of the pipeline refuses.
        raise SweepError(
            "refusing to sweep: no per-type document counts were supplied, so there is nothing "
            "to check against the "
            f"{MIN_DOCUMENTS_PER_TYPE}-document floor. An empty count is a caller that did not "
            "count, not a corpus that is large enough — and a sweep run on unknown volume "
            "produces a winning config nobody can justify."
        )

    thin = {t: n for t, n in documents_per_type.items() if n < MIN_DOCUMENTS_PER_TYPE}
    if thin:
        raise SweepError(
            f"refusing to sweep: {thin} document(s) per type, below the {MIN_DOCUMENTS_PER_TYPE} "
            "needed for a validation score to separate hyperparameters from split variance. At "
            "pilot volume the test split is 3-4 documents and the difference between 1e-4 and "
            "2e-4 is smaller than the noise (arch §8, §11a). Run the IMPL-15 pilot first; the "
            "sweep belongs before the first *production* run, not before the pilot."
        )


def candidates_for(phase: str, *, adapter_type: str = "foundation",
                   fixed: dict[str, Any] | None = None,
                   config_dir: Path = CONFIG_DIR) -> list[Candidate]:
    """Every candidate in one phase, for one adapter type."""
    config = load_phase(phase, config_dir)
    grid = (config.get("grid") or {}).get(adapter_type) or {}
    if not grid:
        raise SweepError(f"phase {phase!r} defines no grid for {adapter_type!r}")

    candidates: list[Candidate] = []
    for parameter, values in grid.items():
        if len(values) < 2:
            raise SweepError(
                f"phase {phase!r} offers {len(values)} value(s) for {parameter!r}; a "
                "one-candidate grid is not a sweep"
            )
        candidates += [
            Candidate(phase=phase, adapter_type=adapter_type, parameter=parameter,
                      value=value, fixed=dict(fixed or {}))
            for value in values
        ]
    return candidates


def pick_winner(results: Sequence[CandidateResult], metric: str, goal: str) -> CandidateResult | None:
    """The best candidate, or ``None`` when nothing was measured.

    A candidate that produced no score is **excluded, not defaulted**. Ranking an
    unmeasured run as if it scored zero (or infinity) would let a crashed run win
    a minimise-loss phase outright.
    """
    scored = [(r, r.score(metric)) for r in results]
    # NaN is excluded alongside None. Every comparison against NaN is False, so
    # `min` never replaces it once it is the running best — a diverged run whose
    # loss went NaN wins a minimise phase outright, and `max` too. That is the
    # exact "a crashed run must not win" failure the docstring above promises to
    # prevent, arriving through a value that merely looks measured.
    measured = [
        (r, s) for r, s in scored
        if s is not None and not (isinstance(s, float) and math.isnan(s))
    ]
    diverged = [r.candidate for r, s in scored if isinstance(s, float) and math.isnan(s)]
    if diverged:
        log.warning(
            "excluding %d diverged candidate(s) whose %s was NaN: %s. A NaN is not a score.",
            len(diverged), metric, [c.run_id for c in diverged],
        )
    if not measured:
        return None
    return (min if goal == "minimize" else max)(measured, key=lambda pair: pair[1])[0]


def run_phase(
    phase: str,
    train: Callable[[Candidate], CandidateResult],
    *,
    adapter_type: str = "foundation",
    fixed: dict[str, Any] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> PhaseResult:
    """Train every candidate in a phase and pick the winner.

    ``train`` is injected — a stub in tests, the real launcher on a pod. The
    sweep's own logic (ordering, budgets, winner selection, manifests) is
    therefore verifiable without a GPU, which is what the IMPL-06 acceptance
    criterion asks for.
    """
    config = load_phase(phase, config_dir)
    metric = config["metric"]["name"]
    goal = config["metric"]["goal"]
    result = PhaseResult(phase=phase, metric=metric, goal=goal)

    for candidate in candidates_for(phase, adapter_type=adapter_type, fixed=fixed,
                                    config_dir=config_dir):
        trained = train(candidate)
        if trained.manifest is not None and not trained.manifest.is_sweep_run:
            raise SweepError(
                f"{candidate.run_id} produced a manifest with is_sweep_run=False. Sweep runs are "
                "first-class registry entries (IMPL-02); an untagged one would swamp default "
                "listings and be indistinguishable from a production run."
            )
        result.results.append(trained)

    if result.unmeasured:
        log.warning(
            "%d candidate(s) in phase %s produced no %s and are excluded from ranking: %s",
            len(result.unmeasured), phase, metric, result.unmeasured,
        )
    result.winner = pick_winner(result.results, metric, goal)
    if result.winner is None:
        raise SweepError(
            f"phase {phase!r} produced no measured {metric}, so it has no winner. Ranking on a "
            "metric nobody measured would promote a configuration chosen at random."
        )
    log.info("phase %s winner: %s = %r", phase, result.winner.candidate.parameter,
             result.winner.candidate.value)
    return result


def run_sweep(
    train: Callable[[Candidate], CandidateResult],
    *,
    documents_per_type: dict[str, int],
    adapter_type: str = "foundation",
    include_rank: bool = False,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    """The full protocol: phase 1, then 2 conditioned on its winner, then 3 if asked.

    Phase 3 is off by default. Rank is the least likely of the three to be the
    bottleneck, and the spec runs it **only if F1 plateaus** after phases 1 and 2
    — running it unconditionally is three GPU jobs spent on the least promising
    axis.
    """
    assert_enough_data(documents_per_type)

    phases: list[PhaseResult] = []
    fixed: dict[str, Any] = {}

    for phase in ("lr", "epochs"):
        outcome = run_phase(phase, train, adapter_type=adapter_type, fixed=dict(fixed),
                            config_dir=config_dir)
        phases.append(outcome)
        assert outcome.winner is not None                      # run_phase raises otherwise
        fixed[outcome.winner.candidate.parameter] = outcome.winner.candidate.value

    if include_rank:
        outcome = run_phase("rank", train, adapter_type=adapter_type, fixed=dict(fixed),
                            config_dir=config_dir)
        phases.append(outcome)
        assert outcome.winner is not None
        fixed[outcome.winner.candidate.parameter] = outcome.winner.candidate.value
    else:
        phases.append(PhaseResult(
            phase="rank", metric="field_normalized_match", goal="maximize",
            skipped="not run by default — rank is the least likely bottleneck and the spec "
                    "runs it only if F1 plateaus after phases 1 and 2",
        ))

    total_runs = sum(len(p.results) for p in phases)
    log.info("sweep complete: %d run(s); promoting %s", total_runs, fixed)
    return {
        "adapter_type": adapter_type,
        "total_runs": total_runs,
        "phases": [p.as_dict() for p in phases],
        # The configuration to promote as the production training config.
        "winning_config": fixed,
    }


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Bounded 3-phase hyperparameter sweep")
    parser.add_argument("--adapter-type", default="foundation", choices=["foundation", "per_type"])
    parser.add_argument("--include-rank", action="store_true",
                        help="phase 3; run only if field F1 has plateaued after phases 1 and 2")
    parser.add_argument("--out", type=Path, default=Path("sweep_result.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('training.sweep', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    raise SystemExit(
        "Wire the trainer here: pass a callable that applies a Candidate's overrides to the "
        "training config, launches it through training.train_foundation / train_adapter with "
        "is_sweep_run=True, and returns its validation metrics. The protocol itself — ordering, "
        "budgets, winner selection, the manifest requirement and the volume refusal — is complete "
        f"and tested against a stub trainer. ({args.adapter_type}, rank={args.include_rank})"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def write_result(result: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
