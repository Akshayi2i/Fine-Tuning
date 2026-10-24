"""A release is judged on how often the model invents values and on whether the
same document reads the same twice (decision 2026-10-24): hallucination_rate
and run_to_run_agreement in the eval report, the gate and checkpoint selection."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evaluation.gating import (
    CONDITIONAL_METRICS,
    DEFAULT_DELTA,
    GATING_METRICS,
    NON_INFERIORITY_DELTA,
    PENDING_GATING_METRICS,
    PILOT_FLOORS,
    promotion_gate,
)
from evaluation.run_eval import PRINTED_CHECK, REPEAT_ANSWER, build_report, run_to_run_agreement

#: A self-contained line: its report scores the label as it is (as
#: tests/test_eval_metric_additions.py reads it).
LINE = "cyber"


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


def _answer(number="HO-1", effective="01/01/2026"):
    return {"policy": {"policy_number": _env(number), "effective_date": _env(effective)}}


def _check(checked, invented=0):
    """serving.printed_values' record (ExtractionResult.printed_check)."""
    return {"mode": "flag", "checked": checked, "unchecked": 0, "removed": False,
            "not_printed": [{"path": f"policy.field_{i}", "raw": f"Invented {i}", "pages": [1]}
                            for i in range(invented)]}


def _meta(source, mode="ocr_plus_image", **extra):
    return {"source_id": source, "doc_type": "policy", "lob": LINE, "modality_mode": mode, **extra}


# --------------------------------------------------------------------------
# hallucination_rate from serving's printed-values record
# --------------------------------------------------------------------------

def test_the_invented_rate_is_read_from_servings_record_and_weighted_by_values_checked():
    gold = _answer()
    triples = [
        # p1: two readings with a record, pooled; the image-only reading has none.
        (gold, _answer(), _meta("p1", **{PRINTED_CHECK: _check(10, 1)})),
        (gold, _answer(), _meta("p1", "noisy_ocr_image", **{PRINTED_CHECK: _check(10, 3)})),
        (gold, _answer(), _meta("p1", "image_only", **{PRINTED_CHECK: None})),
        (gold, _answer(), _meta("p2", **{PRINTED_CHECK: _check(30)})),
        # Nothing the text could settle, and no record at all: left out.
        (gold, _answer(), _meta("p3", **{PRINTED_CHECK: _check(0)})),
        (gold, _answer(), _meta("p4", **{PRINTED_CHECK: None})),
    ]
    body = build_report("v1", triples).as_dict()
    assert body["gate_metrics"]["hallucination_rate"] == pytest.approx(4 / 50)    # sum over sum
    scores, weights = body["per_document"]["hallucination_rate"], body["per_document_weights"]["hallucination_rate"]
    assert scores == {"p1": pytest.approx(0.2), "p2": 0.0}
    assert weights == {"p1": 20.0, "p2": 30.0}
    # The weighted mean the paired bootstrap tests is the rate the gate reads.
    mean = sum(scores[d] * weights[d] for d in scores) / sum(weights.values())
    assert mean == pytest.approx(body["gate_metrics"]["hallucination_rate"])
    [full] = [s for s in body["by_doc_type"]["policy"] if s["subset"] == "full"]
    assert full["printed_values"]["checked"] == 50 and full["printed_values"]["not_printed"] == 4
    assert full["printed_values"]["instances"][0]["source_id"] == "p1"


def test_a_set_whose_text_settles_nothing_has_no_invented_rate():
    triples = [(_answer(), _answer(), _meta("p1", "image_only", **{PRINTED_CHECK: None})),
               (_answer(), _answer(), _meta("p2", **{PRINTED_CHECK: _check(0)}))]
    metrics = build_report("v1", triples).gate_metrics()
    assert "hallucination_rate" not in metrics                    # not measured, never a perfect 0


def test_without_a_serving_record_the_text_is_searched_as_serving_searches_it():
    """A validation row: the same OCR-tolerant matcher over every page of the
    document, so selection reads the rate the gate reads."""
    ocr = "Policy HO-1 Effective 0l/01/2026"           # an l read for a 1
    triples = [(_answer(), _answer(), _meta("p1", ocr_text=ocr, ocr_pages={1: ocr})),
               (_answer(), _answer(number="ZZ-404-9"), _meta("p2", ocr_text=ocr, ocr_pages={1: ocr}))]
    metrics = build_report("v1", triples).gate_metrics()
    assert metrics["hallucination_rate"] == pytest.approx(1 / 4)   # only p2's invented number


def test_the_validation_count_is_servings_count():
    """Selection reads the rate the gate reads: on the same answer and pages,
    score_hallucinations counts what serving's printed-values check records."""
    import copy

    from evaluation.metrics.extraction_faults import score_hallucinations
    from serving.printed_values import hold_to_printed

    pages = {1: "Named Insured: Rivera Fabrication LLC  Policy NIPO115",
             2: "Coverage A Dwelling $219,000\nDeductible 1,000", 3: ""}
    answer = {
        "named_insured": {"primary_name": _env("Rivera Fabrication LLC")},
        "policy": {"policy_number": _env("NIP0115"),                    # an O read for a 0: printed
                   "program_name": _env("Nowhere Agency"),             # printed nowhere
                   "state": _env("NY"),                                # printed nowhere (now checked)
                   "agent": _env("Somebody Else", 3)},                 # cites a page with no text
        "coverages": [{"coverage_name": _env("Dwelling", 2), "limit": _env("$219,000", 2),
                       "premium": _env("$9,999.00", 2)}],              # printed nowhere
        "additional_fields": [{"label": "Note", "section_hint": "discounts",
                               "value": _env("Water Leak Detection")}],  # printed nowhere
    }
    record = hold_to_printed(copy.deepcopy(answer), {}, pages).record   # strict mode edits its input
    report = score_hallucinations([({}, answer, pages)])
    assert (report.opportunities, report.hits) == (record["checked"], len(record["not_printed"])) == (8, 4)


def test_a_policys_windows_are_looked_up_on_every_page_of_the_document():
    window_1 = _meta("p1", ocr_text="Policy HO-1", ocr_pages={1: "Policy HO-1"})
    window_2 = _meta("p1", ocr_text="Effective 01/01/2026", ocr_pages={2: "Effective 01/01/2026"})
    # Each window wrote both values; each value is printed on one of the two pages.
    windows = [(_answer(), _answer(), window_1), (_answer(), _answer(), window_2)]
    assert build_report("v1", windows).gate_metrics()["hallucination_rate"] == 0.0
    # Another document's pages are not this one's.
    apart = [(_answer(), _answer(), window_1), (_answer(), _answer(), {**window_2, "source_id": "p2"})]
    assert build_report("v1", apart).gate_metrics()["hallucination_rate"] == 0.5


# --------------------------------------------------------------------------
# run_to_run_agreement
# --------------------------------------------------------------------------

def test_agreement_is_the_share_of_fields_equal_both_ways_ignoring_nulls_in_both():
    first = {"policy": {"policy_number": _env("HO-1"), "effective_date": _env("01/01/2026"),
                        "expiration_date": None}}
    second = {"policy": {"policy_number": _env("HO-1"), "effective_date": _env("02/01/2026"),
                         "expiration_date": None}}
    assert run_to_run_agreement(first, second) == (0.5, 2)       # the null in both is not compared
    assert run_to_run_agreement(first, first) == (1.0, 2)
    assert run_to_run_agreement({"policy": {"expiration_date": None}}, {}) is None
    # A field only one reading wrote is a disagreement, whichever wrote it.
    share, compared = run_to_run_agreement(_answer(), {"policy": {"policy_number": _env("HO-1")}})
    assert (share, compared) == (0.5, 2)


def test_agreement_in_the_report_is_weighted_by_fields_compared():
    triples = [
        (_answer(), _answer(), _meta("p1", **{REPEAT_ANSWER: _answer(effective="02/01/2026")})),
        (_answer(), _answer(), _meta("p2", **{REPEAT_ANSWER: _answer()})),
        (_answer(), _answer(), _meta("p2", "noisy_ocr_image")),      # not repeated: not compared
    ]
    body = build_report("v1", triples).as_dict()
    assert body["gate_metrics"]["run_to_run_agreement"] == pytest.approx((0.5 * 2 + 1.0 * 2) / 4)
    assert body["per_document"]["run_to_run_agreement"] == {"p1": 0.5, "p2": 1.0}
    assert body["per_document_weights"]["run_to_run_agreement"] == {"p1": 2.0, "p2": 2.0}


# --------------------------------------------------------------------------
# The golden eval's repeat pass
# --------------------------------------------------------------------------

def _golden_docs():
    from evaluation.golden_eval import GoldenDocument

    return [GoldenDocument(source_id=s, doc_type="policy", golden=_answer(), lob=LINE,
                           image_keys=[f"processed/default/policy/{s}/page_1.png"],
                           page_texts={1: "Policy HO-1 Effective 01/01/2026"}) for s in ("p1", "p2")]


def _fake_extract(monkeypatch, calls, *, fail_repeat=False):
    import serving.pipeline as pipeline
    from serving.pipeline import ExtractionResult

    def extract(request, model, classifier, calibration, **kwargs):
        calls.append((request.source_id, request.modality_mode, request.send_order))
        if request.send_order == "reversed" and fail_repeat:
            raise RuntimeError("engine fell over")
        effective = "02/01/2026" if request.send_order == "reversed" else "01/01/2026"
        return ExtractionResult(
            source_id=request.source_id, doc_type="policy", model_version="t", mode=request.modality_mode,
            schema_valid=True, overall_confidence=0.9, extraction=_answer(effective=effective),
            printed_check=None if request.modality_mode == "image_only" else _check(2),
        )

    monkeypatch.setattr(pipeline, "extract", extract)
    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})


def test_the_main_mode_is_read_again_with_its_windows_reversed_after_the_normal_passes(monkeypatch):
    from evaluation import golden_eval

    calls: list[tuple[str, str, str]] = []
    _fake_extract(monkeypatch, calls)
    triples = golden_eval.evaluate(_golden_docs(), object(), object(), "/tmp", run_to_run=True)
    normal = [(s, m, "planned") for s in ("p1", "p2") for m in golden_eval.GOLDEN_MODES]
    assert calls == normal + [("p1", "ocr_plus_image", "reversed"), ("p2", "ocr_plus_image", "reversed")]
    assert len(triples) == 6                                     # the repeat adds no reading
    main = [meta for _e, _g, meta in triples if meta["modality_mode"] == "ocr_plus_image"]
    assert all(meta[REPEAT_ANSWER] == _answer(effective="02/01/2026") for meta in main)
    assert all(REPEAT_ANSWER not in meta for _e, _g, meta in triples if meta["modality_mode"] != "ocr_plus_image")
    assert [meta[PRINTED_CHECK] for _e, _g, meta in triples][:3] == [_check(2), _check(2), None]

    metrics = build_report("v1", triples).gate_metrics()
    assert metrics["run_to_run_agreement"] == 0.5                # the dates differ, the numbers agree
    assert metrics["hallucination_rate"] == 0.0


def test_no_repeat_when_the_check_is_off_and_a_failed_repeat_is_an_empty_answer(monkeypatch):
    from evaluation import golden_eval

    calls: list[tuple[str, str, str]] = []
    _fake_extract(monkeypatch, calls)
    golden_eval.evaluate(_golden_docs(), object(), object(), "/tmp", run_to_run=False)
    assert all(order == "planned" for _s, _m, order in calls)

    calls.clear()
    _fake_extract(monkeypatch, calls, fail_repeat=True)
    triples = golden_eval.evaluate(_golden_docs(), object(), object(), "/tmp", run_to_run=True)
    [p1] = [meta for _e, _g, meta in triples if meta["source_id"] == "p1" and meta["modality_mode"] == "ocr_plus_image"]
    assert p1[REPEAT_ANSWER] == {} and "engine fell over" in p1["repeat_error"]
    assert build_report("v1", triples).gate_metrics()["run_to_run_agreement"] == 0.0


def test_the_repeat_is_on_by_default_in_the_config():
    from common.config import training_config
    from evaluation.golden_eval import run_to_run_check

    assert training_config("unified")["evaluation"]["run_to_run_check"] is True
    assert run_to_run_check() is True


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def _metrics(**over):
    base = {name: 0.96 for name in GATING_METRICS}
    base.update(schema_validity_rate=1.0, ece_confidence=0.04, confusable_misattribution_rate=0.02,
                false_null_rate=0.03, auto_accept_error_rate=0.01, hallucination_rate=0.02,
                run_to_run_agreement=0.96)
    base.update(over)
    return base


def test_both_are_gating_with_their_margins_and_no_threshold_yet():
    assert GATING_METRICS["hallucination_rate"] == "lower_is_better"
    assert GATING_METRICS["run_to_run_agreement"] == "higher_is_better"
    assert "hallucination_rate" not in PENDING_GATING_METRICS
    assert NON_INFERIORITY_DELTA["hallucination_rate"] == NON_INFERIORITY_DELTA["false_null_rate"] == 0.010
    assert NON_INFERIORITY_DELTA.get("run_to_run_agreement", DEFAULT_DELTA) == DEFAULT_DELTA
    assert "hallucination_rate" not in PILOT_FLOORS and "run_to_run_agreement" not in PILOT_FLOORS
    assert {"hallucination_rate", "run_to_run_agreement"} <= CONDITIONAL_METRICS


def test_a_rise_in_invention_beyond_its_margin_is_blocked():
    blocked = promotion_gate(_metrics(hallucination_rate=0.05), _metrics())
    assert not blocked.passed and "hallucination_rate" in blocked.failed_gates
    assert promotion_gate(_metrics(hallucination_rate=0.025), _metrics()).passed      # within 0.010
    # Judged by the paired bootstrap when there are per-document scores.
    docs = 10
    paired = promotion_gate(
        _metrics(hallucination_rate=0.05), _metrics(),
        per_document={"hallucination_rate": ([0.02] * docs, [0.05] * docs, [20.0] * docs, [20.0] * docs)},
    )
    assert "hallucination_rate" in paired.failed_gates
    [verdict] = [v for v in paired.verdicts if v.name == "hallucination_rate"]
    assert verdict.basis == "paired_bootstrap"


def test_a_drop_in_agreement_beyond_its_margin_is_blocked():
    assert "run_to_run_agreement" in promotion_gate(_metrics(run_to_run_agreement=0.94), _metrics()).failed_gates
    assert promotion_gate(_metrics(run_to_run_agreement=0.95), _metrics()).passed      # within 0.015


def test_a_baseline_that_never_measured_them_is_compared_on_floors_only_with_a_warning():
    old = {k: v for k, v in _metrics().items() if k not in ("hallucination_rate", "run_to_run_agreement")}
    result = promotion_gate(_metrics(hallucination_rate=0.30, run_to_run_agreement=0.40), old)
    assert result.passed                                         # a first measurement never blocks
    for name in ("hallucination_rate", "run_to_run_agreement"):
        assert any(w.startswith(f"{name}: the promoted release's eval report does not measure it")
                   for w in result.warnings)
        [verdict] = [v for v in result.verdicts if v.name == name]
        assert verdict.basis == "floor_only" and verdict.floor is None


def test_absent_from_both_is_not_applicable_and_absent_after_a_measured_baseline_blocks():
    unmeasured = {k: v for k, v in _metrics().items() if k not in ("hallucination_rate", "run_to_run_agreement")}
    first = promotion_gate(unmeasured, None)
    assert first.passed
    assert {"hallucination_rate", "run_to_run_agreement"} <= set(first.as_dict()["not_applicable"])
    assert promotion_gate(unmeasured, unmeasured).passed
    stopped = promotion_gate(unmeasured, _metrics())
    assert {"hallucination_rate", "run_to_run_agreement"} <= set(stopped.failed_gates)


# --------------------------------------------------------------------------
# Checkpoint selection
# --------------------------------------------------------------------------

def _report(f1s):
    from evaluation.checkpoint_eval import CheckpointScore, SelectionReport

    report = SelectionReport(scores=[CheckpointScore(f"/ck/checkpoint-{s}", s, {"field_normalized_match": f})
                                     for s, f in f1s])
    report.selected = max(report.scores, key=lambda c: (c.field_f1, c.step)).checkpoint
    return report


def _documents(accuracy, invented, agreement=None, empty=0.40):
    metrics = {"field_normalized_match": accuracy, "document_field_normalized_match": accuracy,
               "document_hallucination_rate": invented, "document_false_null_rate": empty}
    if agreement is not None:
        metrics["document_run_to_run_agreement"] = agreement
    return metrics


def _choose(full, margin=0.02):
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60), (600, 0.58)][:len(full)])
    choose_on_documents(report, lambda ck: full[ck], finalists=3, guard_margin=margin)
    return report


def test_close_on_field_match_the_finalist_that_invents_less_wins():
    report = _choose({"/ck/checkpoint-280": _documents(0.60, 0.05), "/ck/checkpoint-560": _documents(0.59, 0.01)})
    assert report.selected == "/ck/checkpoint-560"
    assert report.document_choice["finalists"]["/ck/checkpoint-560"]["document_hallucination_rate"] == 0.01


def test_further_behind_on_field_match_inventing_less_does_not_win():
    report = _choose({"/ck/checkpoint-280": _documents(0.60, 0.05), "/ck/checkpoint-560": _documents(0.55, 0.01)})
    assert report.selected == "/ck/checkpoint-280"
    assert report.document_choice["behind_on_field_match"] == ["/ck/checkpoint-560"]


def test_an_invention_tie_goes_to_field_match_then_to_the_later_step():
    assert _choose({"/ck/checkpoint-280": _documents(0.60, 0.02),
                    "/ck/checkpoint-560": _documents(0.59, 0.02)}).selected == "/ck/checkpoint-280"
    assert _choose({"/ck/checkpoint-280": _documents(0.60, 0.02),
                    "/ck/checkpoint-560": _documents(0.60, 0.02)}).selected == "/ck/checkpoint-560"


def test_a_finalist_that_reads_the_same_document_differently_twice_does_not_win():
    report = _choose({"/ck/checkpoint-280": _documents(0.60, 0.01, agreement=0.90),
                      "/ck/checkpoint-560": _documents(0.59, 0.03, agreement=0.97)})
    assert report.selected == "/ck/checkpoint-560"
    choice = report.document_choice
    assert choice["excluded_by_guard"] == ["/ck/checkpoint-280"]
    assert choice["guards_failed"] == {"/ck/checkpoint-280": ["document_run_to_run_agreement"]}
    assert choice["run_to_run_measured"] is True
    # Within the margin of the best agreement, the guard lets it through.
    assert _choose({"/ck/checkpoint-280": _documents(0.60, 0.01, agreement=0.96),
                    "/ck/checkpoint-560": _documents(0.59, 0.03, agreement=0.97)}).selected == "/ck/checkpoint-280"


def test_the_false_null_guard_still_excludes_and_sets_no_bar():
    report = _choose({"/ck/checkpoint-280": _documents(0.70, 0.00, empty=0.50),
                      "/ck/checkpoint-560": _documents(0.60, 0.04, empty=0.40),
                      "/ck/checkpoint-600": _documents(0.59, 0.02, empty=0.41)})
    assert report.document_choice["guards_failed"] == {"/ck/checkpoint-280": ["document_false_null_rate"]}
    assert report.selected == "/ck/checkpoint-600"     # within 0.02 of 0.60, and inventing less
    assert report.document_choice["run_to_run_measured"] is False


def test_the_finalists_pass_is_generated_again_in_reverse_order(monkeypatch):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation.checkpoint_eval import generation_cost, generation_scorer
    from inference_core import model_runner
    from inference_core.model_runner import EchoBackend, load_model
    from tests.test_validation_generation import GOLDEN, _val_rows

    sent: list[list[tuple[str, ...]]] = []
    real = model_runner.generate_batch

    def images(messages):
        return tuple(part.get("image") for message in messages if isinstance(message.get("content"), list)
                     for part in message["content"] if isinstance(part, dict) and part.get("type") == "image")

    def recording(model, requests, **kwargs):
        sent.append([images(messages) for messages, _schema, _cap in requests])
        return real(model, requests, **kwargs)

    monkeypatch.setattr(model_runner, "generate_batch", recording)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load_model("base", client, backend_impl=EchoBackend(json.dumps(GOLDEN, separators=(",", ":"))))
    scorer = generation_scorer(_val_rows(3), model, repeat=True)
    metrics = scorer("/s/checkpoint-10")
    assert len(sent) == 2 and len(set(sent[0])) == 3
    assert sent[1] == list(reversed(sent[0]))                      # every row again, backwards
    assert metrics["run_to_run_agreement"] == 1.0                  # an echo reads the same twice
    cost = generation_cost(scorer)
    assert cost["rows"] == 6 and cost["passes"][0]["repeat"]["generated"] == 3


def test_the_repeat_goes_to_the_pass_the_finalists_are_judged_on(monkeypatch, tmp_path):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import checkpoint_eval as C
    from orchestration import pipeline_dag
    from tests.test_validation_cut import _rows

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in _rows()[:900]))
    built = []
    original = C.generation_scorer
    monkeypatch.setattr(C, "generation_scorer",
                        lambda rows, model, **kw: built.append((len(rows), kw)) or original(rows, model, **kw))
    scorer = C.vllm_scorer(client=client, val_path="corpus/val.jsonl", model=object(), sample_rows=400,
                           full_modes=["ocr_plus_image"], repeat=True)
    assert built == [(400, {}), (300, {"repeat": True})] and callable(scorer.full)

    started = []

    def spawn(command, env, log_path):
        started.append(command)
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(C, "_spawn", spawn)
    C.ParallelScorer(val_path="v", images_root=None, sample_rows=400, gpus=["0"], work_dir=tmp_path,
                     full_split=True, repeat=True, poll_seconds=0).score_many(["/out/checkpoint-1"])
    assert "--repeat" in started[0]
    # The configured selection reads its finalists twice.
    assert pipeline_dag._selection_settings(SimpleNamespace())["selection_run_to_run_check"] is True
    asked = []
    monkeypatch.setattr(C, "scoring_gpus", lambda: ["0"])
    monkeypatch.setattr(C, "vllm_scorer", lambda **kw: asked.append(kw) or "S")
    pipeline_dag._checkpoint_scorer(SimpleNamespace(client=object(), corpus="v1", tenant_id="t", out_version="v1"))
    assert asked[0]["repeat"] is True
