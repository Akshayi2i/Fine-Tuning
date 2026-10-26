"""The per-line rule on a re-scored parent (decisions #14 and #15 of 2026-10-26).

A continued cycle's parent whose eval report is on another yardstick only
because the scoring code moved is re-scored from its stored answers - as the
continuation does before training - and its lines are gated against the
re-scored report, not blocked as missing. The line evidence records that the
reference was re-scored. Synthetic fixtures only (tests.test_decision_rescore).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.gating import promotion_gate
from tests.test_decision_rescore import (  # noqa: F401 - fast: the autouse format fixture
    LINE,
    _evaluate,
    _freeze,
    _scored_by_old_code,
    _serve,
    fast,
)
from tests.test_orchestration import PASSING_METRICS, make_context

#: 30 frozen documents: the fewest a line is gated on.
SOURCES = tuple(f"golden_p{i:02d}" for i in range(1, 31))


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller():
    from orchestration.runpod_controller import LocalBackend, RunPodController

    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def _continued_from_v1(client, controller, monkeypatch):
    """v2, continuing policy-v1 - a run from the base - on LINE."""
    from orchestration import pipeline_dag
    from registry_utils import query_registry

    monkeypatch.setattr(query_registry, "get",
                        lambda run_id, _client: SimpleNamespace(parent_run_id=None))
    monkeypatch.setattr(pipeline_dag, "served_lines", lambda ctx: [LINE])
    return make_context(client, controller, out_version="v2")


def test_a_parent_scored_by_old_code_is_rescored_and_its_line_gated(client, controller, monkeypatch):
    from orchestration.pipeline_dag import line_evidence

    _freeze(client, SOURCES)
    _serve(monkeypatch)
    _scored_by_old_code(monkeypatch, client, "v1")
    _evaluate(client, "v2")
    ctx = _continued_from_v1(client, controller, monkeypatch)

    warnings: list[str] = []
    evidence, why = line_evidence(ctx, "policy-v1", warnings)
    assert why is None, why
    assert set(evidence.rescored) == {"policy-v1"}
    assert "different scoring code" in evidence.rescored["policy-v1"]["difference"]
    assert evidence.rescored["policy-v1"]["rescored_from"]["fingerprint"]
    [comparison] = evidence.compared
    assert comparison.reference_rescored and len(comparison.candidate_scores) == 30
    assert any("were re-scored" in w for w in warnings), warnings

    result = promotion_gate(dict(PASSING_METRICS), None, continued_from="policy-v1",
                            line_evidence=evidence)
    assert result.passed, result.report()
    assert "line_regression_evidence" not in result.failed_gates
    [verdict] = result.line_verdicts
    assert (verdict["status"], verdict["held_to"], verdict["held_to_rescored"]) == (
        "gated", "policy-v1", True)


def test_a_parent_with_no_stored_answers_is_still_missing_and_says_why(client, controller, monkeypatch):
    from orchestration.pipeline_dag import line_evidence

    _freeze(client, SOURCES)
    _serve(monkeypatch)
    _scored_by_old_code(monkeypatch, client, "v1")
    _evaluate(client, "v2")
    key = paths.eval_report("v1", scope="policy")
    client.write_json(key, {k: v for k, v in client.read_json(key).items() if k != "answers_stored"})
    ctx = _continued_from_v1(client, controller, monkeypatch)

    evidence, why = line_evidence(ctx, "policy-v1")
    assert evidence is None
    assert why.startswith("policy-v1: the two were scored by different scoring code")
    assert "were not re-scored" in why
