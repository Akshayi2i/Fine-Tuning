"""Re-scoring a release's stored answers when only the scoring code or settings
changed (decision 2026-10-26, #15). The old model is never run again
(decision 2026-10-22 stands).

The golden eval stores every reading's served answer, as scored, under the
tenant (``paths.eval_answers_key``), and its report records where. When the
promoted release's report - or a continued cycle's parent's - is on another
yardstick only because the scoring code or its settings moved, its stored
answers are scored again by the code running now, against the frozen golds
read as the gate reads them, and the gate compares again: non-inferiority and
the improvement rule apply, and the cycle continues its parent instead of
retraining from the base. Another frozen set, another tenant, or answers
missing or incomplete are refused, and the gate keeps judging on floors alone.
"""

from __future__ import annotations

import gzip
import json

import pytest

import common.config
from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation import golden_eval, rescore, run_eval
from evaluation.freeze_eval_set import manifest_key
from evaluation.rescore import load_answers, rescore_report, rescored_or_none
from evaluation.run_eval import REPEAT_ANSWER, eval_set_record, scoring_fingerprint, yardstick_difference

#: A self-contained line: its report scores the label as it is.
LINE = "cyber"
FROZEN_AT = "2026-10-01T00:00:00+00:00"
SOURCES = ("golden_p1", "golden_p2", "golden_p3")
#: What no running scoring code hashes to: the release was scored by other code.
OLD_CODE = "0ld5c0r1ngc0de00"


def _old_fingerprint(client: BlobClient, version: str = "v1") -> str:
    """The fingerprint the release recorded, scored by the old code."""
    recorded = client.read_json(paths.eval_report(version, scope="policy"))["eval_set"]["scoring_fingerprint"]
    assert recorded != scoring_fingerprint()
    return recorded


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """The shipped format profile, pinned."""
    monkeypatch.setattr(common.config, "shared_format_config", lambda: {"profile": "fast"})


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


def _number(source: str) -> str:
    return f"CY-{source[-1]}"


def _answer(number: str, effective: str = "01/01/2026") -> dict:
    return {"policy": {"policy_number": _env(number), "effective_date": _env(effective)}}


def _freeze(client: BlobClient, sources=SOURCES, *, frozen_at: str = FROZEN_AT) -> None:
    """A frozen set of reviewed golds, each printed on its one page."""
    root = paths.golden_eval_set_dir(None)
    for source in sources:
        client.write_json(f"{root}/{source}/metadata.json", {
            "doc_type": "policy", "lob": [LINE], "reviewer_id": "Dana Reviewer", "review_date": "2026-09-30"})
        client.write_bytes(f"{root}/{source}/page_1.png", b"\x89PNG fixture")
        client.write_text(f"{root}/{source}/page_1.md",
                          f"Declarations Policy {_number(source)} Effective 01/01/2026")
        client.write_json(f"{root}/{source}/golden.json", _answer(_number(source)))
    client.write_json(manifest_key(), {"frozen_at": frozen_at, "frozen_from_corpus": "v1",
                                       "source_ids": list(sources)})


def _serve(monkeypatch, *, wrong=(), fails=()) -> None:
    """Serving, faked: each reading answers its document's gold - with a wrong
    effective date for the documents in ``wrong`` - and those in ``fails``
    never come back. Sent no text (image_only), it records no printed values."""
    import serving.pipeline as pipeline
    from serving.pipeline import ExtractionResult

    def extract(request, model, classifier, calibration, **kwargs):
        if request.source_id in fails:
            raise RuntimeError("engine fell over")
        sent_text = request.modality_mode != "image_only"
        record = {"mode": "strict", "checked": 2, "unchecked": 0, "not_printed": [], "removed": True}
        return ExtractionResult(
            source_id=request.source_id, doc_type="policy", model_version="t", mode=request.modality_mode,
            schema_valid=True, overall_confidence=0.9,
            extraction=_answer(_number(request.source_id),
                               "02/02/2026" if request.source_id in wrong else "01/01/2026"),
            printed_check=record if sent_text else None,
        )

    monkeypatch.setattr(pipeline, "extract", extract)
    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})


def _evaluate(client: BlobClient, version: str = "v1", *, lines=(LINE,), calibrated: bool = False) -> dict:
    """The golden eval of ``version``; ``calibrated``: served through (stand-in) calibrators,
    as the gate serves a release that has them."""
    return golden_eval.evaluate_version(client, object(), version=version, corpus_version="v1",
                                        lines=list(lines) if lines is not None else None,
                                        calibrators=object() if calibrated else None)


def _scored_by_old_code(monkeypatch, client: BlobClient, version: str = "v1", **kwargs) -> dict:
    """The golden eval run under other scoring code: the fingerprint it records
    is another, as after any change to the scoring modules."""
    running = run_eval._scoring_fingerprint
    monkeypatch.setattr(run_eval, "_scoring_fingerprint", lambda: OLD_CODE)
    try:
        return _evaluate(client, version, **kwargs)
    finally:
        monkeypatch.setattr(run_eval, "_scoring_fingerprint", running)


def _stored_report(client: BlobClient, version: str = "v1") -> dict:
    return client.read_json(paths.eval_report(version, scope="policy"))


# --------------------------------------------------------------------------
# Storing the answers
# --------------------------------------------------------------------------


def test_the_golden_eval_stores_every_reading_it_scored_under_the_tenant(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch)
    body = _evaluate(client)

    stored = body["answers_stored"]
    assert stored["key"] == paths.eval_answers_key("policy-v1", None) == (
        "run-artifacts/default/policy-v1/eval_answers.jsonl.gz")
    # Under a tenant-scoped prefix: a tenant deletion reaches the answers.
    assert paths.tenant_of(stored["key"]) == "default"
    assert _stored_report(client)["answers_stored"] == stored
    assert stored["readings"] == 9 and stored["documents"] == 3          # 3 documents x 3 modes
    assert 0 < stored["bytes"] < stored["raw_bytes"]                       # gzip

    header, rows = load_answers(client, stored["key"], stored["sha256"])
    assert header["version"] == "v1" and header["tenant_id"] == "default"
    assert header["eval_set"] == body["eval_set"] and header["lines_scored"] == [LINE]
    assert sorted(header["golds"]) == list(SOURCES) and all(header["golds"].values())
    assert [(r["metadata"]["source_id"], r["metadata"]["modality_mode"]) for r in rows] == [
        (s, m) for s in SOURCES for m in golden_eval.GOLDEN_MODES]
    # What build_report reads rides with each answer: the printed-values
    # record, the text it was held to, the repeat answer, the confidences.
    image_only = [r["metadata"] for r in rows if r["metadata"]["modality_mode"] == "image_only"]
    assert all(REPEAT_ANSWER in m and m["printed_text"] == "borrowed" and m["printed_check"]
               for m in image_only)
    assert all({"field_confidence", "schema_valid", "synthetic", "lob"} <= set(r["metadata"]) for r in rows)
    assert rows[0]["answer"] == _answer("CY-1")


def test_answers_that_cannot_be_stored_never_fail_the_eval(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch)
    monkeypatch.setattr(paths, "eval_answers_key", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
    body = _evaluate(client)
    assert body["answers_stored"] == {"key": None, "error": "OSError: full"}
    assert body["gate_metrics"]["field_normalized_match"] == 1.0
    why = rescore_report(client, "v1")[1]
    assert "they could not be stored: OSError: full" in why


# --------------------------------------------------------------------------
# Re-scoring
# --------------------------------------------------------------------------


def test_the_same_scoring_code_scores_the_stored_answers_as_the_eval_did(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch, wrong={"golden_p1"})
    body = _evaluate(client)
    report, why = rescore_report(client, "v1")
    assert report is not None, why
    assert report["gate_metrics"] == body["gate_metrics"]
    assert report["per_document"] == body["per_document"]
    assert report["per_document_weights"] == body["per_document_weights"]
    assert report["invention_coverage"] == body["invention_coverage"]
    assert report["lines_scored"] == [LINE] and report["calibrated"] is False
    assert report["rescored_from"]["fingerprint"] == body["eval_set"]["scoring_fingerprint"]
    assert "re-scored under the scoring code running now" in why and "not run again" in why


def test_other_scoring_code_is_rescored_with_the_golds_read_as_the_gate_reads_them(client, monkeypatch):
    """A label rule changed since v1 was scored (its fingerprint is another):
    the golds are read again through load_golden_set, as the gate reads them,
    and the CURRENT build_report scores the old answers."""
    _freeze(client)
    _serve(monkeypatch, wrong={"golden_p1"})
    old = _scored_by_old_code(monkeypatch, client)
    assert old["eval_set"]["scoring_fingerprint"] != scoring_fingerprint()
    now = eval_set_record(client, None, [])
    assert "different scoring code" in yardstick_difference(old["eval_set"], now, None)
    assert old["gate_metrics"]["field_normalized_match"] < 1.0

    # The new rule: the effective date is no longer expected of the model.
    reading = golden_eval.verified_label
    monkeypatch.setattr(golden_eval, "verified_label", lambda label, texts, **kw: {
        "policy": {"policy_number": reading(label, texts, **kw)["policy"]["policy_number"]}})
    report, why = rescore_report(client, "v1")
    assert report is not None, why
    # Scored as an eval run now on the same answers would score them.
    fresh = _evaluate(client, "v9")
    assert report["gate_metrics"] == fresh["gate_metrics"] != old["gate_metrics"]
    assert report["per_document"] == fresh["per_document"]
    assert report["eval_set"]["scoring_fingerprint"] == scoring_fingerprint()
    assert yardstick_difference(report["eval_set"], now, None) is None
    tag = report["rescored_from"]
    assert tag["fingerprint"] == old["eval_set"]["scoring_fingerprint"] and tag["eval_set"] == old["eval_set"] and tag["at"]
    assert tag["answers"] == paths.eval_answers_key("policy-v1", None)
    # Nothing is written: the stored report stays as it was scored.
    assert _stored_report(client)["eval_set"]["scoring_fingerprint"] == _old_fingerprint(client)


def test_a_release_scored_before_answers_were_stored_is_not_rescored(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch)
    _evaluate(client)
    key = paths.eval_report("v1", scope="policy")
    client.write_json(key, {k: v for k, v in client.read_json(key).items() if k != "answers_stored"})
    report, why = rescore_report(client, "v1")
    assert report is None
    assert "stores no answers (scored before the golden eval stored them, decision 2026-10-26)" in why


@pytest.mark.parametrize("case, said", [
    ("frozen_again", "the set was frozen again"),
    ("other_tenant", "tenant 'default''s frozen set, and this is tenant 'acme'"),
    ("answers_missing", "its stored answers are missing"),
    ("answers_replaced", "not the ones its report recorded"),
    ("gold_changed", "golden_p2's frozen gold is not the one its answers were scored against"),
])
def test_another_set_or_tenant_or_missing_answers_are_refused(client, monkeypatch, case, said):
    _freeze(client)
    _serve(monkeypatch)
    body = _evaluate(client)
    key = body["answers_stored"]["key"]
    tenant = None
    if case == "frozen_again":
        _freeze(client, frozen_at="2026-10-20T00:00:00+00:00")
    elif case == "other_tenant":
        tenant = "acme"
    elif case == "answers_missing":
        client.delete(key)
    elif case == "answers_replaced":
        client.write_bytes(key, gzip.compress(b'{"kind": "golden_eval_answers"}\n'))
    else:
        client.write_json(f"{paths.golden_eval_set_dir(None)}/golden_p2/golden.json", _answer("CY-9"))
    report, why = rescore_report(client, "v1", tenant)
    assert report is None
    assert why.startswith("v1's stored answers were not re-scored: ") and said in why, why


def test_a_rescoring_that_breaks_is_a_refusal_not_a_crash(client, monkeypatch):
    """The callers fall back to what they did before re-scoring existed."""
    _freeze(client)
    _serve(monkeypatch)
    _evaluate(client)
    monkeypatch.setattr(run_eval, "build_report", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug")))
    report, why = rescore_report(client, "v1")
    assert report is None and "re-scoring failed (RuntimeError: bug)" in why


def test_answers_with_a_reading_that_failed_to_extract_are_incomplete(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch, fails={"golden_p3"})
    body = _evaluate(client)
    assert body["infrastructure_failures"]["count"] > 0
    report, why = rescore_report(client, "v1")
    assert report is None and "failed to extract twice" in why


def test_a_reading_the_settings_now_ask_for_that_was_never_made_is_refused(client, monkeypatch):
    """A larger noisy share, or another repeat mode, asks for readings the
    eval never made: refused, never approximated."""
    _freeze(client)
    _serve(monkeypatch)
    monkeypatch.setattr(golden_eval, "golden_noisy_share", lambda: 1e-9)
    _evaluate(client)
    monkeypatch.setattr(golden_eval, "golden_noisy_share", lambda: 1.0)
    report, why = rescore_report(client, "v1")
    assert report is None and "no noisy_ocr_image reading of golden_p1" in why

    client2 = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    _freeze(client2)
    _evaluate(client2)
    monkeypatch.setattr(golden_eval, "run_to_run_mode", lambda: "ocr_plus_image")
    report, why = rescore_report(client2, "v1")
    assert report is None and "golden_p1 was not read twice in ocr_plus_image" in why


def test_a_smaller_share_now_leaves_the_readings_it_no_longer_asks_for_out(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch)
    body = _evaluate(client)
    share = 0.5
    monkeypatch.setattr(golden_eval, "golden_noisy_share", lambda: share)
    kept = [s for s in SOURCES if golden_eval.in_share(s, share, "noisy")]
    report, why = rescore_report(client, "v1")
    assert report is not None, why
    assert report["golden_noisy_share"] == share == report["eval_set"]["golden_noisy_share"]
    assert report["rescored_from"]["readings"] == body["answers_stored"]["readings"] - (3 - len(kept))
    assert report["rescored_from"]["readings_left_out"] == 3 - len(kept)


def test_one_gate_rescores_a_release_once(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch)
    _scored_by_old_code(monkeypatch, client)
    loads = []
    reading = golden_eval.load_golden_set
    monkeypatch.setattr(golden_eval, "load_golden_set", lambda *a, **k: loads.append(1) or reading(*a, **k))
    first, _ = rescore_report(client, "v1")
    second, _ = rescore_report(client, "v1")
    assert first == second and first is not second and len(loads) == 1


def test_rescored_or_none_gives_the_report_on_the_yardstick_asked_for(client, monkeypatch):
    """The helper the gate's line rule calls for the runs it holds a line to."""
    _freeze(client)
    _serve(monkeypatch)
    _evaluate(client, "v2")
    notes: list[str] = []
    assert rescored_or_none(client, "v2", None, notes=notes) == _stored_report(client, "v2")
    assert notes == []                                       # already on it: nothing re-scored

    _scored_by_old_code(monkeypatch, client, "v1")
    report = rescored_or_none(client, "v1", None, _stored_report(client, "v2")["eval_set"], notes=notes)
    assert report is not None and report["rescored_from"]["fingerprint"] == _old_fingerprint(client)
    assert "different scoring code" in notes[-1] and "were re-scored" in notes[-1]

    key = paths.eval_report("v1", scope="policy")
    client.write_json(key, {k: v for k, v in client.read_json(key).items() if k != "answers_stored"})
    assert rescored_or_none(client, "v1", None, notes=notes) is None
    assert "stores no answers" in notes[-1]
    assert rescored_or_none(client, "v9", None, notes=notes) is None
    assert notes[-1].startswith("v9 has no eval report at ")


def test_the_stored_answers_are_json_lines_behind_gzip(client, monkeypatch):
    _freeze(client)
    _serve(monkeypatch)
    key = _evaluate(client)["answers_stored"]["key"]
    lines = gzip.decompress(client.read_bytes(key)).decode("utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["kind"] == "golden_eval_answers" and header["format"] == rescore.ANSWERS_FORMAT
    assert header["readings"] == len(lines) - 1


# --------------------------------------------------------------------------
# The gate's baseline
# --------------------------------------------------------------------------


@pytest.fixture
def controller():
    from orchestration.runpod_controller import LocalBackend, RunPodController

    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def _gate_context(client, controller, monkeypatch, **over):
    """v1 promoted, scored by other scoring code; v1.1 the candidate, scored
    (here: v1's answers scored now, so the two agree) on the yardstick as it
    stands. Both serve the frozen documents' line."""
    from orchestration import pipeline_dag
    from tests.test_orchestration import make_context, promote_in_index

    monkeypatch.setattr(pipeline_dag, "served_lines", lambda ctx: [LINE])
    promote_in_index(client, "v1")
    candidate = _evaluate(client, "v1.1", calibrated=True)
    _scored_by_old_code(monkeypatch, client, "v1", calibrated=True)
    return make_context(client, controller, out_version="v1.1", metrics_provider=None, **over), candidate


def test_the_gate_compares_a_promoted_release_rescored_from_its_answers(client, controller, monkeypatch):
    from orchestration.pipeline_dag import (
        baseline_not_comparable,
        baseline_rescoring,
        default_baseline_metrics,
        paired_document_scores,
    )

    _freeze(client)
    _serve(monkeypatch)
    ctx, candidate = _gate_context(client, controller, monkeypatch)
    assert default_baseline_metrics(ctx) == candidate["gate_metrics"]
    assert baseline_not_comparable(ctx) is None
    notes: list[str] = []
    paired = paired_document_scores(ctx, notes)
    assert paired and paired["field_normalized_match"][0] == [1.0, 1.0, 1.0], notes
    rescoring = baseline_rescoring(ctx)
    assert rescoring["rescored"] is True and "different scoring code" in rescoring["difference"]
    assert rescoring["rescored_from"]["fingerprint"] == _old_fingerprint(client)


def test_a_promoted_release_with_no_answers_stays_floors_only_and_says_why(client, controller, monkeypatch):
    from orchestration.pipeline_dag import baseline_not_comparable, baseline_rescoring, paired_document_scores

    _freeze(client)
    _serve(monkeypatch)
    ctx, _candidate = _gate_context(client, controller, monkeypatch)
    key = paths.eval_report("v1", scope="policy")
    client.write_json(key, {k: v for k, v in client.read_json(key).items() if k != "answers_stored"})
    why = baseline_not_comparable(ctx)
    assert why.startswith("the two were scored by different scoring code")
    assert f"(promoted report {key}). v1's stored answers were not re-scored: " in why
    assert baseline_rescoring(ctx)["rescored"] is False
    notes: list[str] = []
    assert paired_document_scores(ctx, notes) is None and "floors alone" in notes[-1]


def test_the_gate_decision_records_the_rescored_baseline(client, controller, monkeypatch):
    """The whole gate: the promoted release compared re-scored - not floors
    alone - and the decision says so."""
    from orchestration.pipeline_dag import run_stages, stages_for
    from tests.test_orchestration import seed_corpus

    seed_corpus(client)
    assert run_stages(_dry(client, controller, "v1"), stages_for("all"), command="all").ok
    _freeze(client)
    _serve(monkeypatch)
    ctx, _candidate = _gate_context(client, controller, monkeypatch, fresh=True)
    # A dry run selects no checkpoint to score: the candidate's report is
    # reused for the gate as one that names none.
    key = paths.eval_report("v1.1", scope="policy")
    client.write_json(key, {k: v for k, v in client.read_json(key).items() if k != "adapter_checkpoint"})
    run_stages(ctx, stages_for("all"), command="all")
    decision = client.read_json(paths.gate_decision("v1.1", scope="policy"))
    assert decision["baseline_not_comparable"] is None and decision["baseline_metrics"] is not None
    assert decision["baseline_rescored"] is True
    assert "different scoring code" in decision["baseline_rescoring"]["difference"]
    assert "were re-scored" in decision["baseline_rescoring"]["reason"]
    assert {v["basis"] for v in decision["verdicts"]} != {"floor_only"}


def _dry(client, controller, version):
    from tests.test_orchestration import make_context

    return make_context(client, controller, out_version=version)


# --------------------------------------------------------------------------
# Continuation
# --------------------------------------------------------------------------


def test_a_parent_on_other_scoring_code_is_rescored_and_continued(client, controller, monkeypatch):
    from orchestration.pipeline_dag import continuation, run_stages, stages_for
    from tests.test_incremental_training import write_adapter
    from tests.test_orchestration import make_context, promote_in_index, seed_corpus

    seed_corpus(client)
    assert run_stages(make_context(client, controller, out_version="v1"), stages_for("all"),
                      command="all").ok
    promote_in_index(client, "v1")
    write_adapter(client, "v1")
    _freeze(client)
    _serve(monkeypatch)
    _scored_by_old_code(monkeypatch, client, "v1", lines=None)

    ctx = make_context(client, controller, out_version="v2")
    assert continuation(ctx) == ("v1", None)
    assert ctx.parent_report_rescored["run_id"] == "policy-v1"
    assert "different scoring code" in ctx.parent_report_rescored["difference"]
    assert ctx.parent_report_rescored["rescored_from"]["fingerprint"] == _old_fingerprint(client)
    # And the training stage records it beside the parent it continued.
    finetune = run_stages(make_context(client, controller, out_version="v2"), stages_for("finetune"),
                          command="finetune")
    training = next(r for r in finetune.results if r.name == "training")
    assert training.data["continued"] == "policy-v1", training.detail
    assert training.data["parent_report_rescored"]["run_id"] == "policy-v1"

    # Scored before answers were stored: from the base, as before, and why.
    key = paths.eval_report("v1", scope="policy")
    client.write_json(key, {k: v for k, v in client.read_json(key).items() if k != "answers_stored"})
    parent, why = continuation(ctx)
    assert parent is None and ctx.parent_report_rescored is None
    assert why.startswith("policy-v1's eval report is on another yardstick, so the gate could not compare")
    assert "v1's stored answers were not re-scored: its eval report" in why


def test_a_parent_on_another_frozen_set_still_starts_fresh(client, controller, monkeypatch):
    from orchestration.pipeline_dag import continuation, run_stages, stages_for
    from tests.test_incremental_training import write_adapter
    from tests.test_orchestration import make_context, promote_in_index, seed_corpus

    seed_corpus(client)
    assert run_stages(make_context(client, controller, out_version="v1"), stages_for("all"),
                      command="all").ok
    promote_in_index(client, "v1")
    write_adapter(client, "v1")
    _freeze(client)
    _serve(monkeypatch)
    _evaluate(client, "v1", lines=None)
    _freeze(client, frozen_at="2026-10-20T00:00:00+00:00")
    parent, why = continuation(make_context(client, controller, out_version="v2"))
    assert parent is None and "the set was frozen again" in why
