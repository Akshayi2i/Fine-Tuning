"""The gate's measurement, reviewed 2026-10-26: auto-accept compared as field
match compares, and not measured when nothing was accepted; the coverage and
selective accuracy beside it; field match without the model's codes; a reading
that fails twice blocks the gate instead of scoring as wrong answers; noisy
readings held to the clean text; deterministic shares; documents in flight."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import common.config
from evaluation.metrics.auto_accept import score_auto_accept
from evaluation.run_eval import PRINTED_CHECK, PRINTED_TEXT, build_report

LINE = "cyber"
PAGE_1 = "Declarations Policy HO-1 Effective 01/01/2026"


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(common.config, "shared_format_config", lambda: {"profile": "fast"})


def _env(value, *, flagged=False, raw=None, page=1):
    return {"raw": str(value) if raw is None else raw, "parsed": value, "page_ref": [page],
            "confidence": {"score": 0.99, "source": "vlm"}, "flagged": flagged}


# --------------------------------------------------------------------------
# #12 Auto-accept
# --------------------------------------------------------------------------

def test_an_accepted_money_value_is_compared_with_its_declared_type():
    """'$831.00' is the label's 831.0: field match says so (values_agree), and
    auto-accept used to compare it as text and count it wrong."""
    gold = {"premium": {"total": _env(831.0)}}
    got = {"premium": {"total": _env("$831.00", raw="$831.00")}}
    tally = score_auto_accept(gold, got)
    assert (tally.accepted, tally.wrong, tally.rate) == (1, 0, 0.0)
    assert score_auto_accept(gold, {"premium": {"total": _env("$813.00", raw="$813.00")}}).wrong == 1


def test_nothing_accepted_is_not_measured_and_coverage_says_so():
    gold = {"policy": {"policy_number": _env("HO-123"), "account_id": _env("A-1200")}}
    got = {"policy": {"policy_number": _env("HO-123", flagged=True), "account_id": _env("A-9", flagged=True)}}
    tally = score_auto_accept(gold, got)
    assert tally.rate is None                               # was 0.0: a perfect score for accepting nothing
    assert tally.coverage == 0.0 and tally.selective_accuracy is None
    meta = {"source_id": "p1", "doc_type": "policy", "lob": ["equipment_breakdown"], "modality_mode": "ocr_plus_image"}
    [full] = build_report("t", [(gold, got, meta)]).full_set()
    assert "auto_accept_error_rate" not in full.metrics
    assert full.metrics["auto_accept_coverage"] == 0.0
    assert full.metrics["auto_accept_counts"] == {"decided": 2, "accepted": 0, "wrong": 0}


def test_coverage_and_selective_accuracy_per_field_type_with_a_wilson_bound():
    gold = {"policy": {"policy_number": _env("HO-123"), "account_id": _env("A-1200")}}
    got = {"policy": {"policy_number": _env("HO-123"), "account_id": _env("A-9", flagged=True)}}
    tally = score_auto_accept(gold, got)
    assert tally.coverage == 0.5 and tally.selective_accuracy == 1.0
    assert 0.0 < tally.selective_accuracy_lower < 1.0      # one right of one: a wide interval
    by_type = tally.selective_by_type()
    [(field_type, row)] = by_type.items()
    assert row["accepted"] == 1 and row["accuracy"] == 1.0 and row["wilson_lower_95"] < 1.0
    assert sum(tally.coverage_by_type().values()) > 0
    assert field_type in tally.coverage_by_type()


def test_the_gate_reads_an_unmeasured_auto_accept_rate_as_its_missing_metric_policy_says():
    """No value accepted: absent from the metrics. auto_accept_error_rate is a
    conditional metric (gating.CONDITIONAL_METRICS): with no comparable
    promoted release that measured it, it is not applicable and does not
    block; after one that measured it, a candidate that accepts nothing is
    blocked as unmeasured (the policy kept as it was)."""
    from evaluation.gating import promotion_gate

    base = {"field_exact_match": 0.9, "field_normalized_match": 0.9}
    first = promotion_gate(dict(base), None, require_all_measured=False)
    assert "auto_accept_error_rate" not in first.failed_gates
    after = promotion_gate(dict(base), {**base, "auto_accept_error_rate": 0.01})
    assert "auto_accept_error_rate" in after.failed_gates
    assert any("auto_accept_error_rate was not measured" in r for r in after.blocking_reasons)
    on_another_yardstick = promotion_gate(dict(base), {**base, "auto_accept_error_rate": 0.01},
                                          baseline_not_comparable="another scoring code",
                                          require_all_measured=False)
    assert "auto_accept_error_rate" not in on_another_yardstick.failed_gates


# --------------------------------------------------------------------------
# #17 Field match without the model's codes
# --------------------------------------------------------------------------

def test_field_match_without_codes_leaves_out_bare_values_and_nothing_else():
    from evaluation.run_eval import model_assigned_paths

    gold = {"policy": {"policy_number": _env("HO-1")},
            "coverages": [{"coverage_code": "HO_COV_A", "coverage_name": _env("Dwelling")}]}
    got = {"policy": {"policy_number": _env("HO-1")},
           "coverages": [{"coverage_code": "HO_COV_B", "coverage_name": _env("Dwelling")}]}
    assert model_assigned_paths(gold, got) == {"coverages[].coverage_code"}
    # A flat document has no envelopes: nothing is told apart, nothing left out.
    assert model_assigned_paths({"a": "1"}, {"a": "2"}) == set()


def test_the_tier_reads_served_field_match_and_never_the_without_codes_figure():
    """Decision 2026-10-26: the tier reads what users receive, served_field_match
    - not the three-mode field_normalized_match, nor the without-codes figure."""
    from evaluation.gating import release_tier

    served = ([0.93] * 160, [1.0] * 160)
    assert release_tier({"served_field_match": 0.93, "field_normalized_match": 0.80,
                         "field_normalized_match_without_codes": 0.5}, served_documents=served) == "production"
    assert release_tier({"served_field_match": 0.91, "field_normalized_match": 0.99,
                         "field_normalized_match_without_codes": 0.99},
                        served_documents=([0.91] * 160, [1.0] * 160)) == "interim"
    assert release_tier({"field_normalized_match": 0.99}, served_documents=served) == "interim"


# --------------------------------------------------------------------------
# The gate's gold read as training checks it (label_verification, listed)
# --------------------------------------------------------------------------

def test_a_filled_figure_the_page_prints_is_expected_and_one_printed_nowhere_is_not():
    """Training keeps "$250,000.00" on a page printing "$250,000" (serving's
    figure-aware matcher); the gate's words-only reading dropped it, and a
    right answer scored as an extra value. A value no page prints is still not
    expected of the model - serving takes it out - so it leaves the gold."""
    from data_pipeline.dataset_builder.label_verification import VerificationReport
    from data_pipeline.dataset_builder.label_verification import verified_label as words_only
    from evaluation.golden_eval import verified_label

    def env(raw, parsed):
        return {"raw": raw, "parsed": parsed, "page_ref": [1]}

    label = {"premium": {"total": env("$250,000.00", 250000.0), "fee": env("$75.00", 75.0)},
             "fideon:filled": {"paths": ["premium.total", "premium.fee"]}}
    pages = ["Declarations Total premium $250,000 due"]
    report = VerificationReport()
    assert words_only(label, pages, report)["premium"] == {}                    # what the gate read before
    gold = verified_label(label, pages)
    assert gold["premium"] == {"total": env("$250,000.00", 250000.0)}
    assert label["premium"]["fee"] == env("$75.00", 75.0)                       # the frozen gold untouched
    answer = {"premium": {"total": env("$250,000", 250000.0)}}
    meta = {"source_id": "p1", "doc_type": "policy", "lob": ["equipment_breakdown"], "modality_mode": "ocr_plus_image"}
    [full] = build_report("t", [(gold, answer, meta)]).full_set()
    assert full.error_records == []


# --------------------------------------------------------------------------
# The golden eval, faked serving
# --------------------------------------------------------------------------

def _answer(number="HO-1", effective="01/01/2026"):
    return {"policy": {"policy_number": {"raw": number, "parsed": number, "page_ref": [1]},
                       "effective_date": {"raw": effective, "parsed": effective, "page_ref": [1]}}}


def _golden(source="p1", text=PAGE_1, carrier=None):
    from evaluation.golden_eval import GoldenDocument

    return GoldenDocument(source_id=source, doc_type="policy", golden=_answer(), lob=LINE,
                          image_keys=[f"processed/default/policy/{source}/page_1.png"], page_texts={1: text},
                          carrier=carrier)


def _fake_extract(monkeypatch, calls, *, fail=None):
    """``fail(request, attempt)`` -> True raises for that call."""
    import serving.pipeline as pipeline
    from serving.pipeline import ExtractionResult

    attempts: dict[tuple, int] = {}

    def extract(request, model, classifier, calibration, **kwargs):
        key = (request.source_id, request.modality_mode, request.send_order)
        attempts[key] = attempts.get(key, 0) + 1
        calls.append((*key, getattr(request, "printed_texts", None), dict(request.page_texts)))
        if fail and fail(request, attempts[key]):
            raise RuntimeError("engine fell over")
        record = {"mode": "strict", "checked": 2, "unchecked": 0, "not_printed": [], "removed": True}
        return ExtractionResult(
            source_id=request.source_id, doc_type="policy", model_version="t", mode=request.modality_mode,
            schema_valid=True, overall_confidence=0.9, extraction=_answer(),
            printed_check=None if request.modality_mode == "image_only" else record,
        )

    monkeypatch.setattr(pipeline, "extract", extract)
    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})


# --------------------------------------------------------------------------
# #16 Failed readings
# --------------------------------------------------------------------------

def test_a_reading_that_fails_once_is_read_again_and_scored(monkeypatch):
    from evaluation import golden_eval

    calls: list = []
    _fake_extract(monkeypatch, calls, fail=lambda r, n: r.modality_mode == "image_only" and n == 1)
    triples = golden_eval.evaluate([_golden()], object(), object(), "/tmp", run_to_run=False)
    assert len(triples) == 3 and triples.infrastructure_failures == []
    assert [(r["source_id"], r["mode"]) for r in triples.retried] == [("p1", "image_only")]


def test_a_reading_that_fails_twice_is_not_scored_and_is_recorded(monkeypatch):
    from evaluation import golden_eval

    calls: list = []
    _fake_extract(monkeypatch, calls, fail=lambda r, n: r.source_id == "p2" and r.modality_mode == "image_only")
    triples = golden_eval.evaluate([_golden("p1"), _golden("p2")], object(), object(), "/tmp", run_to_run=False)
    # Never an empty answer scored as wrong: left out, and recorded.
    assert [(m["source_id"], m["modality_mode"]) for _e, _g, m in triples] == [
        ("p1", "ocr_plus_image"), ("p1", "noisy_ocr_image"), ("p1", "image_only"),
        ("p2", "ocr_plus_image"), ("p2", "noisy_ocr_image")]
    assert [(f["source_id"], f["mode"], f["pass"]) for f in triples.infrastructure_failures] == [
        ("p2", "image_only", "first")]
    assert "engine fell over" in triples.infrastructure_failures[0]["error"]


def _client():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _version_setup(monkeypatch, documents):
    from evaluation import golden_eval, run_eval

    monkeypatch.setattr(golden_eval, "_refuse_interrupted_freeze", lambda *a, **k: None)
    monkeypatch.setattr(run_eval, "assert_eval_set_disjoint", lambda *a, **k: None)
    monkeypatch.setattr(golden_eval, "load_golden_set", lambda *a, **k: list(documents))


def test_infrastructure_failures_reach_the_gate_which_blocks_and_asks_for_a_rerun(monkeypatch):
    from artifact_registry import paths
    from common.adapter import ADAPTER
    from evaluation import golden_eval
    from evaluation.gating import INFRASTRUCTURE_FAILURES, is_gate, promotion_gate

    calls: list = []
    _fake_extract(monkeypatch, calls, fail=lambda r, n: r.source_id == "p2" and r.send_order == "reversed")
    _version_setup(monkeypatch, [_golden("p1"), _golden("p2")])
    client = _client()
    body = golden_eval.evaluate_version(client, object(), version="v1", corpus_version="v1",
                                        promoted_report={})
    assert body["infrastructure_failures"]["count"] == 1
    assert body["infrastructure_failures"]["documents"] == ["p2"]
    assert body["repeats_failed"] == 1 and body["documents_failed"] == 0
    assert body["gate_metrics"][INFRASTRUCTURE_FAILURES] == 1
    # Kept apart in Blob, so a re-run scores again rather than reading this back.
    stored = client.read_json(paths.eval_report("v1", scope=ADAPTER.name))
    assert stored["gate_metrics"] == {} and stored["gate_metrics_partial"][INFRASTRUCTURE_FAILURES] == 1

    result = promotion_gate(body["gate_metrics"], None)
    assert not result.passed and result.failed_gates == [INFRASTRUCTURE_FAILURES]
    assert "re-run the gate" in result.blocking_reasons[0]
    assert not is_gate(INFRASTRUCTURE_FAILURES)              # no override can name it
    # None failed: nothing of it in the metrics, nothing blocked for it.
    assert INFRASTRUCTURE_FAILURES not in promotion_gate({"field_normalized_match": 0.9}, None).failed_gates


# --------------------------------------------------------------------------
# #10 Noisy readings held to the clean text
# --------------------------------------------------------------------------

def test_a_noisy_reading_is_sent_the_corruption_and_held_to_the_clean_text(monkeypatch):
    import serving.pipeline as pipeline
    from evaluation import golden_eval
    from serving.pipeline import ExtractionRequest

    @dataclass
    class Request(ExtractionRequest):
        printed_texts: dict | None = None

    monkeypatch.setattr(pipeline, "ExtractionRequest", Request)
    calls: list = []
    _fake_extract(monkeypatch, calls)
    clean = "Declarations Policy HO-778812 Effective 01/01/2026 Premium $1,234.00 Insured Jane Smith " * 4
    triples = golden_eval.evaluate([_golden(text=clean)], object(), object(), "/tmp", run_to_run=False)
    [noisy] = [c for c in calls if c[1] == "noisy_ocr_image"]
    assert noisy[3] == {1: clean}                            # held to what the page prints
    assert noisy[4] != {1: clean}                            # sent the corruption to read
    assert [c[3] for c in calls if c[1] != "noisy_ocr_image"] == [None, None]
    [meta] = [m for _e, _g, m in triples if m["modality_mode"] == "noisy_ocr_image"]
    assert meta[PRINTED_TEXT] == "clean"


def test_without_printed_texts_in_serving_the_noisy_reading_says_what_it_was_held_to(monkeypatch):
    from evaluation import golden_eval

    monkeypatch.setattr(golden_eval, "_takes_printed_texts", lambda request_type: False)
    calls: list = []
    _fake_extract(monkeypatch, calls)
    triples = golden_eval.evaluate([_golden()], object(), object(), "/tmp", run_to_run=False)
    [meta] = [m for _e, _g, m in triples if m["modality_mode"] == "noisy_ocr_image"]
    assert meta[PRINTED_TEXT] == "corrupted"
    assert meta[PRINTED_CHECK]["checked"] == 2


# --------------------------------------------------------------------------
# #10 / #28 Deterministic shares, documents in flight
# --------------------------------------------------------------------------

def test_shares_pick_a_deterministic_nested_subset():
    from evaluation.golden_eval import in_share

    sources = [f"policy_{i:04d}" for i in range(400)]
    half = {s for s in sources if in_share(s, 0.5, "noisy")}
    quarter = {s for s in sources if in_share(s, 0.25, "noisy")}
    assert quarter <= half and 120 < len(half) < 280
    assert {s for s in sources if in_share(s, 0.5, "noisy")} == half          # every run, every machine
    assert all(in_share(s, 1.0, "repeat") for s in sources)
    assert {s for s in sources if in_share(s, 0.5, "repeat")} != half         # salted per use


def test_the_shares_narrow_the_noisy_and_repeat_readings(monkeypatch):
    from evaluation import golden_eval

    calls: list = []
    _fake_extract(monkeypatch, calls)
    docs = [_golden(f"p{i}") for i in range(40)]
    triples = golden_eval.evaluate(docs, object(), object(), "/tmp", run_to_run=True,
                                   noisy_share=0.5, repeat_share=0.25)
    noisy = {m["source_id"] for _e, _g, m in triples if m["modality_mode"] == "noisy_ocr_image"}
    repeated = {c[0] for c in calls if c[2] == "reversed"}
    assert noisy == {d.source_id for d in docs if golden_eval.in_share(d.source_id, 0.5, "noisy")}
    assert repeated == {d.source_id for d in docs if golden_eval.in_share(d.source_id, 0.25, "repeat")}
    assert len([m for _e, _g, m in triples if m["modality_mode"] == "image_only"]) == 40


def test_the_shares_are_configured_and_part_of_the_yardstick(monkeypatch):
    from evaluation import golden_eval
    from evaluation.run_eval import yardstick_difference, yardstick_settings

    settings = yardstick_settings()
    assert settings["golden_noisy_share"] == 1.0 and settings["run_to_run_share"] == 1.0
    record = {"tenant_id": "t", "frozen_at": "x", "frozen_from_corpus": "c", "scoring_fingerprint": "f"}
    assert "golden_noisy_share" in yardstick_difference({**record, "golden_noisy_share": 1.0},
                                                        {**record, "golden_noisy_share": 0.5}, "t")
    monkeypatch.setattr(common.config, "training_config",
                        lambda name: {"evaluation": {"golden_noisy_share": 0}})
    with pytest.raises(golden_eval.GoldenEvalError, match="golden_noisy_share"):
        golden_eval.golden_noisy_share()


def test_documents_in_flight_give_the_sequential_triples(monkeypatch):
    from evaluation import golden_eval

    calls: list = []
    _fake_extract(monkeypatch, calls)
    docs = [_golden(f"p{i}") for i in range(6)]
    one = golden_eval.evaluate(docs, object(), object(), "/tmp", run_to_run=True, in_flight=1)
    many = golden_eval.evaluate(docs, object(), object(), "/tmp", run_to_run=True, in_flight=4)
    assert [(m["source_id"], m["modality_mode"]) for _e, _g, m in one] == [
        (m["source_id"], m["modality_mode"]) for _e, _g, m in many]
    assert [g for _e, g, _m in one] == [g for _e, g, _m in many]
    assert build_report("v", one).as_dict()["gate_metrics"] == build_report("v", many).as_dict()["gate_metrics"]


def test_only_a_backend_that_answers_as_requests_finish_gets_several_in_flight(monkeypatch):
    from evaluation.golden_eval import DOCUMENTS_IN_FLIGHT, documents_in_flight

    assert documents_in_flight(object()) == 1
    in_process = SimpleNamespace(backend=SimpleNamespace(supports_submit=lambda: False))
    served = SimpleNamespace(backend=SimpleNamespace(supports_submit=lambda: True))
    assert documents_in_flight(in_process) == 1
    monkeypatch.setattr(common.config, "training_config", lambda name: {"evaluation": {}})
    assert documents_in_flight(served) == DOCUMENTS_IN_FLIGHT
    monkeypatch.setattr(common.config, "training_config",
                        lambda name: {"evaluation": {"golden_documents_in_flight": 8}})
    assert documents_in_flight(served) == 8
