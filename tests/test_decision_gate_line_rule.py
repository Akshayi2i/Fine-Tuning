"""The per-line rule for continued runs (decision #14 of 2026-10-26).

* Each line is held to the PARENT run, margin 0.015, one-sided, with a Holm
  correction across the lines tested (family-wise alpha 0.05).
* A line is gated only on at least 30 shared real documents; below that it is
  warned, with the minimum detectable drop for its n.
* A cumulative drift check holds each line to the LAST FRESH run of the chain,
  margin 0.03, same minimum, same correction: small losses that add up.
* Every line's evidence (n, delta, bound, adjusted alpha, gated or warned) is
  recorded in the decision.

Synthetic fixtures only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.bootstrap import NonInferiorityTest, holm_step_down, minimum_detectable_drop
from evaluation.gating import LineComparison, LineEvidence, promotion_gate
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import PASSING_METRICS, _scored_on, make_context


def _gate(evidence: LineEvidence | None, **kwargs):
    return promotion_gate(dict(PASSING_METRICS), None, continued_from="policy-v4",
                          line_evidence=evidence, **kwargs)


def _line(line: str, before: list[float], after: list[float], reference: str = "policy-v4"):
    return LineComparison(line=line, reference=reference, reference_scores=list(before),
                          candidate_scores=list(after))


def test_the_margins_minimum_and_alpha_are_named_constants():
    from evaluation import gating

    assert gating.LINE_PARENT_MARGIN == 0.015
    assert gating.LINE_DRIFT_MARGIN == 0.03
    assert gating.LINE_MIN_DOCUMENTS == 30
    assert gating.LINE_FAMILY_ALPHA == 0.05


# --------------------------------------------------------------------------
# Holm's step-down
# --------------------------------------------------------------------------


def _test(name: str, at_margin: int) -> NonInferiorityTest:
    """A test whose resamples put ``at_margin`` of 10,000 differences at -0.02
    (beyond the 0.015 margin) and the rest at 0.0: its p-value is their share."""
    deltas = [-0.02] * at_margin + [0.0] * (10_000 - at_margin)
    return NonInferiorityTest(name, 0.0, 0.015, 40, at_margin / 10_000, deltas=deltas)


def test_holm_tests_the_smallest_p_at_alpha_over_m_and_stops_at_the_first_failure():
    tests = {"auto": _test("auto", 0), "wc": _test("wc", 300), "homeowners": _test("homeowners", 400)}
    # Each alone at 0.05 shows non-inferiority.
    assert all(t.shown(0.05) for t in tests.values())
    outcomes = holm_step_down(tests, 0.05)
    assert [(n, o.rank) for n, o in sorted(outcomes.items(), key=lambda i: i[1].rank)] == [
        ("auto", 1), ("wc", 2), ("homeowners", 3)]
    assert outcomes["auto"].adjusted_alpha == pytest.approx(0.05 / 3)
    assert outcomes["wc"].adjusted_alpha == pytest.approx(0.05 / 2)
    assert outcomes["homeowners"].adjusted_alpha == pytest.approx(0.05)
    assert outcomes["auto"].shown
    # wc's bound at 0.025 is -0.02: not shown, and the step-down stops there.
    assert not outcomes["wc"].shown and outcomes["wc"].bound == -0.02
    # homeowners clears its own level, but Holm rejects nothing after a failure.
    assert tests["homeowners"].shown(outcomes["homeowners"].adjusted_alpha)
    assert not outcomes["homeowners"].shown and outcomes["homeowners"].stopped_by == "wc"
    # Holm's adjusted p-values: the running maximum of (m - k + 1) p.
    assert outcomes["wc"].adjusted_p == pytest.approx(0.06)
    assert outcomes["homeowners"].adjusted_p == pytest.approx(0.06)


def _borderline() -> tuple[list[float], list[float]]:
    """40 documents, 4 of them 0.08 lower: one-sided bound just above -0.015 at
    alpha 0.05, just below at 0.025."""
    return [0.9] * 40, [0.82] * 4 + [0.9] * 36


def test_the_gate_corrects_across_the_lines_tested():
    before, after = _borderline()
    alone = _gate(LineEvidence(compared=[_line("auto", before, after)]))
    assert alone.passed, alone.report()

    together = _gate(LineEvidence(compared=[_line("auto", before, after), _line("wc", before, after)]))
    assert sorted(together.failed_gates) == ["line:auto", "line:wc"]
    verdicts = {v["line"]: v for v in together.line_verdicts}
    assert {v["holm_family"] for v in verdicts.values()} == {2}
    assert sorted(v["adjusted_alpha"] for v in verdicts.values()) == [0.025, 0.05]
    stopped = next(v for v in verdicts.values() if v["holm_stopped_by"])
    assert any("Holm's step-down stopped at" in r for r in together.blocking_reasons)
    assert stopped["bound"] > -0.015          # its own bound clears; Holm stopped before it


# --------------------------------------------------------------------------
# The 30-document minimum
# --------------------------------------------------------------------------


def test_a_line_under_30_shared_documents_is_warned_with_its_detectable_drop():
    before = [0.95, 0.90] * 15
    after = [0.80, 0.70] * 15
    thin = _gate(LineEvidence(compared=[_line("auto", before[:29], after[:29])]))
    assert thin.passed, thin.report()
    [verdict] = thin.line_verdicts
    assert verdict["status"] == "warned" and verdict["documents"] == 29
    assert verdict["non_inferior"] is None and verdict["bound"] is None
    expected = minimum_detectable_drop([a - b for a, b in zip(after[:29], before[:29], strict=True)])
    assert verdict["minimum_detectable_drop"] == pytest.approx(expected, abs=1e-6)
    assert any("auto (29, minimum detectable drop" in w for w in thin.warnings), thin.warnings

    gated = _gate(LineEvidence(compared=[_line("auto", before, after)]))
    assert gated.failed_gates == ["line:auto"]
    assert gated.line_verdicts[0]["status"] == "gated" and gated.line_verdicts[0]["documents"] == 30


def test_the_minimum_detectable_drop_shrinks_with_n_and_needs_two_documents():
    differences = [-0.1, 0.0, 0.05, -0.02] * 10
    assert minimum_detectable_drop(differences[:1]) is None
    assert minimum_detectable_drop(differences[:8]) > minimum_detectable_drop(differences)


def test_every_line_s_evidence_is_recorded():
    result = _gate(LineEvidence(compared=[_line("auto", [0.9] * 30, [0.91] * 30)]))
    [verdict] = result.as_dict()["line_verdicts"]
    assert {"line", "check", "held_to", "documents", "reference", "candidate", "delta", "margin",
            "bound", "adjusted_alpha", "status", "non_inferior", "p_value", "holm_adjusted_p",
            "minimum_detectable_drop"} <= set(verdict)
    assert (verdict["check"], verdict["status"], verdict["documents"]) == ("parent", "gated", 30)
    assert verdict["delta"] == pytest.approx(0.01) and verdict["margin"] == 0.015


# --------------------------------------------------------------------------
# The chain: the parent is the reference, the run from the base the drift one
# --------------------------------------------------------------------------


def _report(line_scores: dict[str, list[float]], served: dict[str, list[float]] | None = None) -> dict:
    """An eval report, as far as the line rule reads it: each real document's
    field match and line, documents ``<line>_0001`` and on - and, with
    ``served``, each one's served field match too (a report written after
    decision 2026-10-26)."""
    def by_source(by_line: dict[str, list[float]]) -> dict[str, float]:
        return {f"{line}_{i:04d}": s for line, values in by_line.items() for i, s in enumerate(values, 1)}

    scores = by_source(line_scores)
    per_document = {"field_normalized_match": scores}
    if served is not None:
        per_document["served_field_match"] = by_source(served)
    return {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
        "per_document": per_document,
        "per_document_weights": {name: dict.fromkeys(values, 1.0) for name, values in per_document.items()},
        "per_document_lines": {source: source.rsplit("_", 1)[0] for source in scores},
    }


@pytest.fixture
def chain(monkeypatch):
    """Candidate v5 continuing policy-v4, down to policy-v1 from the base; each
    report ``{version: {line: scores}}`` written by the returned function."""
    from orchestration import pipeline_dag
    from registry_utils import query_registry

    parents = {"policy-v4": "policy-v3", "policy-v3": "policy-v2", "policy-v2": "policy-v1",
               "policy-v1": None}

    def get(run_id, _client):
        if run_id not in parents:
            raise query_registry.RegistryQueryError(f"no manifest for {run_id}")
        return SimpleNamespace(parent_run_id=parents[run_id])

    monkeypatch.setattr(query_registry, "get", get)
    monkeypatch.setattr(pipeline_dag, "served_lines", lambda ctx: ["auto"])
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    controller = RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")
    ctx = make_context(client, controller, out_version="v5")

    def write(**reports: dict[str, list[float]] | dict):
        for version, lines in reports.items():
            body = lines if "per_document" in lines else _report(lines)
            client.write_json(paths.eval_report(version, scope="policy"), body)
        return ctx

    return write


def test_the_reference_is_the_parent_not_the_best_run_of_the_chain(chain):
    """v1 scored 0.97, the parent v4 0.95, the candidate 0.945: half a point
    under the parent, passed. Held to the best of the chain it would have been
    2.5 points down and blocked; the drift check (3 points) passes it too."""
    from orchestration.pipeline_dag import line_evidence

    ctx = chain(v1={"auto": [0.97] * 30}, v2={"auto": [0.96] * 30}, v3={"auto": [0.955] * 30},
                v4={"auto": [0.95] * 30}, v5={"auto": [0.945] * 30})
    evidence, why = line_evidence(ctx, "policy-v4")
    assert why is None
    assert [c.reference for c in evidence.compared] == ["policy-v4"]
    assert evidence.drift_reference == "policy-v1" and evidence.drift_reference_is_fresh
    result = _gate(evidence)
    assert result.passed, result.report()
    assert {(v["check"], v["held_to"]) for v in result.line_verdicts} == {
        ("parent", "policy-v4"), ("drift", "policy-v1")}


def test_cumulative_drift_blocks_a_slow_decline(chain):
    """0.01 a cycle: every step within the parent margin, 0.04 in all past the
    drift margin of 0.03, against the run the chain started from the base."""
    from orchestration.pipeline_dag import line_evidence

    ctx = chain(v1={"auto": [0.95] * 30}, v2={"auto": [0.94] * 30}, v3={"auto": [0.93] * 30},
                v4={"auto": [0.92] * 30}, v5={"auto": [0.91] * 30})
    evidence, _why = line_evidence(ctx, "policy-v4")
    result = _gate(evidence)
    assert result.failed_gates == ["line:auto"]
    [reason] = result.blocking_reasons
    assert "auto has drifted from policy-v1, the last run from the base" in reason
    assert "0.9500 -> 0.9100" in reason and "--fresh" in reason
    parent = next(v for v in result.line_verdicts if v["check"] == "parent")
    drift = next(v for v in result.line_verdicts if v["check"] == "drift")
    assert parent["non_inferior"] and not drift["non_inferior"]
    assert drift["margin"] == 0.03 and drift["delta"] == pytest.approx(-0.04)


def test_a_drift_reference_whose_report_is_gone_falls_back_and_says_so(chain):
    from orchestration.pipeline_dag import line_evidence

    ctx = chain(v2={"auto": [0.95] * 30}, v3={"auto": [0.94] * 30}, v4={"auto": [0.93] * 30},
                v5={"auto": [0.925] * 30})
    warnings: list[str] = []
    evidence, _why = line_evidence(ctx, "policy-v4", warnings)
    assert evidence.drift_reference == "policy-v2" and not evidence.drift_reference_is_fresh
    assert "policy-v1, the last run from the base, has no report on this yardstick" in evidence.drift_note
    result = _gate(evidence)
    assert any("oldest run of the chain on this yardstick" in w for w in result.warnings)
    assert result.passed, result.report()                     # 0.025 down: within 0.03


def test_a_parent_from_the_base_has_no_separate_drift_check(chain):
    from orchestration.pipeline_dag import line_evidence

    ctx = chain(v4={"auto": [0.95] * 30}, v5={"auto": [0.95] * 30})
    # policy-v4's manifest names policy-v3, which has no report and no
    # readable chain beyond: v4 is the oldest run on this yardstick.
    evidence, _why = line_evidence(ctx, "policy-v4")
    assert evidence.drift == [] and evidence.drift_reference is None
    assert evidence.drift_note.startswith("no cumulative drift check")


def test_lines_are_compared_on_the_served_field_match(chain):
    """Decision 2026-10-26: what users receive. The three-mode match held; the
    served one fell 5 points - and that blocks."""
    from orchestration.pipeline_dag import line_evidence

    ctx = chain(v1=_report({"auto": [0.95] * 30}, served={"auto": [0.95] * 30}),
                v2={"auto": [0.95] * 30}, v3={"auto": [0.95] * 30},
                v4=_report({"auto": [0.95] * 30}, served={"auto": [0.95] * 30}),
                v5=_report({"auto": [0.95] * 30}, served={"auto": [0.90] * 30}))
    evidence, _why = line_evidence(ctx, "policy-v4")
    assert [c.measure for c in evidence.compared + evidence.drift] == ["served_field_match"] * 2
    result = _gate(evidence)
    assert result.failed_gates == ["line:auto"]
    assert {v["measure"] for v in result.line_verdicts} == {"served_field_match"}
    assert all("served_field_match 0.9500 -> 0.9000" in r for r in result.blocking_reasons)


def test_a_reference_without_served_scores_falls_back_to_the_three_mode_match(chain):
    """A report written before served scores were kept: that reference's line is
    compared on field_normalized_match, said, and recorded; the other reference
    still on the served match."""
    from orchestration.pipeline_dag import line_evidence

    ctx = chain(v1={"auto": [0.95] * 30}, v2={"auto": [0.95] * 30}, v3={"auto": [0.95] * 30},
                v4=_report({"auto": [0.95] * 30}, served={"auto": [0.94] * 30}),
                v5=_report({"auto": [0.95] * 30}, served={"auto": [0.94] * 30}))
    warnings: list[str] = []
    evidence, _why = line_evidence(ctx, "policy-v4", warnings)
    assert [c.measure for c in evidence.compared] == ["served_field_match"]
    assert [c.measure for c in evidence.drift] == ["field_normalized_match"]
    assert any(w.startswith("auto against policy-v1: compared on field_normalized_match") for w in warnings)
    result = _gate(evidence, warnings=warnings)
    assert result.passed, result.report()
    assert {(v["check"], v["measure"]) for v in result.as_dict()["line_verdicts"]} == {
        ("parent", "served_field_match"), ("drift", "field_normalized_match")}


def test_the_decision_records_the_parent_and_the_drift_reference():
    import inspect

    from orchestration import pipeline_dag

    source = inspect.getsource(pipeline_dag.stage_evaluation_gate)
    for key in ('"parent": parent_run', '"drift_reference": evidence.drift_reference',
                '"drift_note": evidence.drift_note', '"tier_confidence"', '"tier_evidence"'):
        assert key in source, key
