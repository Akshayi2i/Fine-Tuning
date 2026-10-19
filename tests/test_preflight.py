"""The preflight: each check passes when it should, and fails with what to do."""

from __future__ import annotations

import json
from collections import namedtuple
from types import SimpleNamespace

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import preflight as P


def _statuses(results):
    return {c.name: c.status for c in results}


def test_azure_needs_write_overwrite_read_and_delete():
    results = []
    P.check_blob(BlobClient(backend=InMemoryBackend(), container="training-data"), results)
    assert _statuses(results) == {"azure": "PASS"}


def test_a_read_only_token_fails_the_azure_check(monkeypatch):
    client = BlobClient(backend=InMemoryBackend(), container="c")

    def refuse(*a, **k):
        raise PermissionError("AuthorizationPermissionMismatch")

    monkeypatch.setattr(client, "write_text", refuse)
    results = []
    P._run("azure", lambda r: P.check_blob(client, r), results)
    assert results[0].status == "FAIL" and "AuthorizationPermissionMismatch" in results[0].detail


def test_the_post_ocr_report_must_exist_and_pass(tmp_path):
    results = []
    P.check_ocr_report(tmp_path, results)
    assert results[-1].status == "FAIL" and "ocr_check" in results[-1].fix
    (tmp_path / "summary.json").write_text(json.dumps({"passed": False, "verdict": "gap 20%"}), encoding="utf-8")
    P.check_ocr_report(tmp_path, results)
    assert results[-1].status == "FAIL"
    (tmp_path / "summary.json").write_text(json.dumps({"passed": True, "verdict": "gap 1%"}), encoding="utf-8")
    P.check_ocr_report(tmp_path, results)
    assert results[-1].status == "PASS"


Plan = namedtuple("Plan", "documents built frozen delivered")


def _plan(documents, kept, rows, frozen=False):
    all_rows = [{"source_id": f"d{i}"} for i in range(kept)]
    built = SimpleNamespace(
        rows_by_split=rows, all_rows=all_rows,
        skipped=[(f"d{i}", "over budget: policy_schedule") for i in range(kept, documents)],
        cap_report=SimpleNamespace(as_dict=lambda: {"documents_rejected": documents - kept}),
    )
    return Plan([object()] * documents, built, frozen, True)


@pytest.mark.parametrize("documents,kept,rows,status", [
    (100, 99, {"train": [1], "val": [1], "test": [1]}, "PASS"),
    (100, 90, {"train": [1], "val": [1], "test": [1]}, "WARN"),     # 10% over the caps
    (100, 60, {"train": [1], "val": [1], "test": [1]}, "FAIL"),     # 40%: train on too little
    (100, 99, {"train": [1], "val": [1], "test": []}, "FAIL"),      # no test rows
])
def test_the_corpus_verdict(monkeypatch, documents, kept, rows, status):
    monkeypatch.setattr("orchestration.pipeline_dag.plan_corpus", lambda ctx: _plan(documents, kept, rows))
    results = []
    facts = P.check_corpus(SimpleNamespace(), results)
    assert results[-1].status == status
    assert facts["documents"] == documents and facts["kept"] == kept


def test_once_frozen_the_corpus_needs_no_test_rows(monkeypatch):
    monkeypatch.setattr("orchestration.pipeline_dag.plan_corpus",
                        lambda ctx: _plan(10, 10, {"train": [1], "val": [1], "test": []}, frozen=True))
    results = []
    P.check_corpus(SimpleNamespace(), results)
    assert results[-1].status == "PASS"


def test_an_unverified_base_model_fails(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("common.config.base_model_dir", lambda: tmp_path)
    monkeypatch.setattr("common.config.base_model_config",
                        lambda: {"model": {"model_id": "Qwen/X", "revision": "abc123", "local_dir": str(tmp_path)}})
    results = []
    P.check_base_model(results)
    assert results[-1].status == "FAIL"
    (tmp_path / ".fideon-revision").write_text("abc123\n", encoding="utf-8")
    P.check_base_model(results)
    assert results[-1].status == "PASS"


def test_a_check_that_raises_is_a_failed_check_not_a_crash():
    results = []
    P._run("boom", lambda r: 1 / 0, results)
    assert results == [P.Check("boom", "FAIL", "ZeroDivisionError: division by zero")]


def test_the_dataset_build_and_the_preflight_share_one_corpus_function():
    """So a preflight that passes has built the corpus the run will write."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "orchestration/pipeline_dag.py").read_text("utf-8")
    stage = source[source.index("def stage_dataset_build"):]
    assert "plan = plan_corpus(ctx)" in stage[:3000]


def _labels(*scores):
    return [("policy", f"policy_{i:04d}",
             {"double_annotated": score is not None, "agreement_score": score})
            for i, score in enumerate(scores)]


def test_agreement_is_reported_and_only_warns_with_no_minimum_set():
    from orchestration import preflight

    results = []
    preflight.check_label_agreement(None, results, labels=_labels(0.91, 0.72, None))
    [check] = results
    assert check.status == "WARN"
    assert "2 double-labelled" in check.detail and "lowest 0.720" in check.detail

    results = []
    preflight.check_label_agreement(None, results, labels=_labels(None, None))
    assert results[0].status == "WARN" and "unmeasured" in results[0].detail


def test_agreement_below_a_set_minimum_fails(monkeypatch):
    import common.constants as constants
    from orchestration import preflight

    monkeypatch.setattr(constants, "MIN_INTER_ANNOTATOR_AGREEMENT", 0.8)
    results = []
    preflight.check_label_agreement(None, results, labels=_labels(0.91, 0.72))
    assert results[0].status == "FAIL" and "policy_0001" in results[0].detail
    results = []
    preflight.check_label_agreement(None, results, labels=_labels(0.91))
    assert results[0].status == "PASS"


def test_every_model_drafted_label_needs_a_reviewer_before_training():
    from orchestration import preflight

    labels = [("policy", "policy_0001", {}),
              ("policy", "policy_0002", {"reviewer_id": "a", "review_date": "2026-10-20"}),
              ("policy", "policy_0003", {})]
    golds = {"policy_0001": {"fideon:provenance": {"provenance": "drafted by Claude Code"}},
             "policy_0003": {}}
    results = []
    preflight.check_reviewed_labels(None, results, labels, read_label=lambda _t, s: golds[s])
    assert results[0].status == "FAIL" and results[0].detail.endswith("['policy_0001']")

    results = []
    preflight.check_reviewed_labels(None, results, labels[1:], read_label=lambda _t, s: golds[s])
    assert results[0].status == "WARN"
