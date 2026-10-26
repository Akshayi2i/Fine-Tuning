"""An image_only answer is held to the printed-values rule too (decision
2026-10-26). Production reads the page images alone and checks the fields on
MinerU's text, so evaluation checks an image_only answer on its document's page
text, borrowed page by page from the same document's ocr_plus_image reading. A
reading with no text to borrow - or a page no reading has text for - is
unscored and counted, never read as nothing invented. And the gate's
run-to-run repeat reads image_only, a mode that is part of the yardstick."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import common.config
from evaluation.run_eval import PRINTED_CHECK, REPEAT_ANSWER, build_report

#: A self-contained line: its report scores the label as it is.
LINE = "cyber"
PAGE_1 = "Declarations Policy HO-1 Effective 01/01/2026"


def _pin(monkeypatch, profile: str) -> None:
    monkeypatch.setattr(common.config, "shared_format_config", lambda: {"profile": profile})


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """The shipped profile, pinned; a test that needs standard pins it again."""
    _pin(monkeypatch, "fast")


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


def _answer(number="HO-1", effective="01/01/2026", page=1):
    return {"policy": {"policy_number": _env(number, page), "effective_date": _env(effective, page)}}


def _meta(source, mode, **extra):
    return {"source_id": source, "doc_type": "policy", "lob": LINE, "modality_mode": mode, **extra}


def _with_text(source, pages, mode="ocr_plus_image"):
    return _meta(source, mode, ocr_text="\n\n".join(pages.values()), ocr_pages=dict(pages))


def _subset(body, name):
    [subset] = [s for s in body["by_doc_type"]["policy"] if s["subset"] == name]
    return subset


# --------------------------------------------------------------------------
# Borrowed text, in the report every reader builds on
# --------------------------------------------------------------------------

def test_an_image_only_answer_inventing_a_value_is_counted_on_borrowed_text():
    gold = _answer()
    triples = [(gold, _answer(), _with_text("p1", {1: PAGE_1})),
               (gold, _answer(number="ZZ-404-9"), _meta("p1", "image_only"))]
    report = build_report("v1", triples)
    body = report.as_dict()
    # Four values checked - the image_only reading's two as well - one invented.
    assert body["gate_metrics"]["hallucination_rate"] == pytest.approx(1 / 4)
    image_only = _subset(body, "image_only")
    assert image_only["hallucination_rate"] == 0.5
    assert image_only["printed_values"]["instances"][0]["emitted"] == "ZZ-404-9"
    assert image_only["printed_values"]["borrowed_text"]["readings"] == 1
    assert image_only["printed_values"]["borrowed_text"]["from_mode"] == "ocr_plus_image"
    assert report.invention_coverage() == {"borrowed_text_readings": 1, "cited_pages_without_text": 0,
                                           "unscored_readings": 0}


def test_an_image_only_answer_with_no_text_to_borrow_is_unscored_not_clean():
    gold = _answer()
    triples = [
        (gold, _answer(), _with_text("p1", {1: PAGE_1})),
        # p2 has no ocr_plus_image reading; its noisy OCR text is not lent.
        (gold, _answer(number="ZZ-404-9"), _meta("p2", "image_only")),
        (gold, _answer(), _with_text("p2", {1: PAGE_1}, mode="noisy_ocr_image")),
    ]
    report = build_report("v1", triples)
    body = report.as_dict()
    assert body["per_document"]["hallucination_rate"] == {"p1": 0.0, "p2": 0.0}   # the noisy reading only
    assert body["per_document_weights"]["hallucination_rate"] == {"p1": 2.0, "p2": 2.0}
    image_only = _subset(body, "image_only")
    assert "hallucination_rate" not in image_only                                 # not measured, never 0
    assert image_only["printed_values"]["unscored"] == {"readings": 1, "by_mode": {"image_only": 1},
                                                        "documents": ["p2"]}
    assert report.invention_coverage()["unscored_readings"] == 1
    # Alone, nothing is measured: absent, not a perfect 0.
    alone = build_report("v1", [(gold, _answer(number="ZZ-404-9"), _meta("p2", "image_only"))])
    assert "hallucination_rate" not in alone.gate_metrics()
    assert alone.invention_coverage()["unscored_readings"] == 1


def test_a_page_no_reading_has_text_for_leaves_its_values_unchecked_not_invented():
    gold = _answer()
    unread = {"policy": {"policy_number": _env("HO-1"), "effective_date": _env("12/31/2027", 2)}}
    triples = [(gold, _answer(), _with_text("p1", {1: PAGE_1})), (gold, unread, _meta("p1", "image_only"))]
    image_only = _subset(build_report("v1", triples).as_dict(), "image_only")
    # The date cites page 2, which no reading carries: unchecked, and shown.
    assert (image_only["printed_values"]["checked"], image_only["printed_values"]["not_printed"]) == (1, 0)
    assert image_only["printed_values"]["borrowed_text"]["cited_pages_without_text"] == 1
    assert image_only["printed_values"]["borrowed_text"]["pages_without_text"] == ["p1 p2"]
    # Citing page 1, which has text, the same date is invented.
    cited = {"policy": {"policy_number": _env("HO-1"), "effective_date": _env("12/31/2027", 1)}}
    triples[1] = (gold, cited, _meta("p1", "image_only"))
    assert _subset(build_report("v1", triples).as_dict(), "image_only")["hallucination_rate"] == 0.5


# --------------------------------------------------------------------------
# Checkpoint selection: the fewest invented values over both modes
# --------------------------------------------------------------------------

PAGES = {1: "Declarations Policy Number HO-778812", 2: "Coverage A Dwelling $219,000"}
DECL = {"document": {}, "carrier": {}, "named_insured": {}, "policy": {"policy_number": _env("HO-778812")}}
ARRAYS = {"coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A",
                         "coverage_name": _env("Dwelling", 2)}]}
#: Each finalist's image_only declarations answer, wrong either way: one invents
#: a number no page prints, the other misreads a word page 1 does print.
IMAGE_ONLY_DECL = {"/ck/checkpoint-560": "ZZ-404-999", "/ck/checkpoint-280": "Declarations"}


def _row(sections, page, mode):
    content = [{"type": "image", "image": f"page_{page}.png"}]
    if mode == "ocr_plus_image":
        content.append({"type": "text", "text": f"<page {page} of 2>\n{PAGES[page]}"})
    return {"doc_type": "policy", "source_id": "s1", "lob": "homeowners", "sections": sections,
            "window_index": 0, "window_pages": [page], "modality_mode": mode,
            "messages": [{"role": "user", "content": content}]}


def _generations(checkpoint, modes):
    from evaluation.validation_generation import ValidationGeneration

    out = []
    for mode in modes:
        decl = DECL
        if mode == "image_only":
            decl = {**DECL, "policy": {"policy_number": _env(IMAGE_ONLY_DECL[checkpoint])}}
        out.append(ValidationGeneration(row=_row("decl", 1, mode), golden=DECL, extraction=decl))
        out.append(ValidationGeneration(row=_row("arrays", 2, mode), golden=ARRAYS, extraction=ARRAYS))
    return out


def _select(monkeypatch, profile):
    """Two finalists, judged on whole documents in the profile's configured modes."""
    from evaluation.checkpoint_eval import CheckpointScore, SelectionReport, choose_on_documents
    from evaluation.validation_generation import score_generations
    from orchestration import pipeline_dag

    _pin(monkeypatch, profile)
    modes = pipeline_dag._selection_settings(SimpleNamespace())["selection_document_modes"]
    report = SelectionReport(scores=[CheckpointScore("/ck/checkpoint-280", 280, {"field_normalized_match": 0.61}),
                                     CheckpointScore("/ck/checkpoint-560", 560, {"field_normalized_match": 0.62})])
    report.selected = "/ck/checkpoint-560"
    scored = {}

    def full(checkpoint):
        scored[checkpoint] = score_generations(_generations(checkpoint, modes), model_version=checkpoint)
        return scored[checkpoint]

    choose_on_documents(report, full, finalists=2, guard_margin=0.02, modes=modes)
    return report, scored, modes


def test_selection_prefers_the_finalist_inventing_less_across_both_modes(monkeypatch):
    report, scored, modes = _select(monkeypatch, "fast")
    assert modes == ["ocr_plus_image", "image_only"]
    late, early = scored["/ck/checkpoint-560"], scored["/ck/checkpoint-280"]
    # Equal on field match - one wrong value each - so only invention parts them.
    assert late["document_field_normalized_match"] == early["document_field_normalized_match"]
    assert late["document_hallucination_rate"] > early["document_hallucination_rate"] == 0.0
    assert report.selected == "/ck/checkpoint-280"
    finalist = report.document_choice["finalists"]["/ck/checkpoint-560"]
    assert finalist["document_hallucination_borrowed_text_readings"] == 1
    assert "document_hallucination_unscored_readings" not in finalist


def test_standard_profile_selection_is_unchanged(monkeypatch):
    """The standard profile's finalists are read in ocr_plus_image alone: no
    image_only answer is judged, nothing is borrowed, and the later step wins
    the tie as before."""
    report, scored, modes = _select(monkeypatch, "standard")
    assert modes == ["ocr_plus_image"]
    assert report.selected == "/ck/checkpoint-560"
    for metrics in scored.values():
        assert metrics["document_hallucination_rate"] == 0.0
        assert not any("borrowed" in name or "unscored" in name for name in metrics)


def test_selection_without_ocr_plus_image_warns_that_image_only_goes_unscored(caplog):
    from evaluation.checkpoint_eval import rows_in_modes
    from evaluation.validation_generation import score_generations

    rows = [_row("decl", 1, "ocr_plus_image"), _row("decl", 1, "image_only")]
    assert rows_in_modes(rows, ["image_only"]) == [rows[1]]
    assert "no page text to borrow" in caplog.text
    metrics = score_generations(_generations("/ck/checkpoint-560", ["image_only"]))
    assert "document_hallucination_rate" not in metrics
    assert metrics["document_hallucination_unscored_readings"] == 1


# --------------------------------------------------------------------------
# The release gate
# --------------------------------------------------------------------------

def _golden(source="p1", text=PAGE_1):
    from evaluation.golden_eval import GoldenDocument

    return GoldenDocument(source_id=source, doc_type="policy", golden=_answer(), lob=LINE,
                          image_keys=[f"processed/default/policy/{source}/page_1.png"], page_texts={1: text})


def _fake_extract(monkeypatch, calls, image_only_number="HO-1"):
    """Serving, faked: the image_only reading invents its number when told to,
    and - sent no text - comes back with no printed-values record."""
    import serving.pipeline as pipeline
    from serving.pipeline import ExtractionResult

    def extract(request, model, classifier, calibration, **kwargs):
        calls.append((request.source_id, request.modality_mode, request.send_order))
        image_only = request.modality_mode == "image_only"
        record = {"mode": "strict", "checked": 2, "unchecked": 0, "not_printed": [], "removed": True}
        return ExtractionResult(
            source_id=request.source_id, doc_type="policy", model_version="t", mode=request.modality_mode,
            schema_valid=True, overall_confidence=0.9,
            extraction=_answer(number=image_only_number if image_only else "HO-1"),
            printed_check=None if image_only else record,
        )

    monkeypatch.setattr(pipeline, "extract", extract)
    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})


def test_the_gates_image_only_reading_is_checked_on_the_documents_text(monkeypatch):
    from evaluation import golden_eval
    from evaluation.run_eval import BORROWED_TEXT, PRINTED_TEXT

    _fake_extract(monkeypatch, [], image_only_number="ZZ-404-9")
    triples = golden_eval.evaluate([_golden()], object(), object(), "/tmp", run_to_run=False)
    [image_only] = [meta for _e, _g, meta in triples if meta["modality_mode"] == "image_only"]
    assert image_only[PRINTED_TEXT] == BORROWED_TEXT
    assert [e["raw"] for e in image_only[PRINTED_CHECK]["not_printed"]] == ["ZZ-404-9"]
    report = build_report("v1", triples)
    assert _subset(report.as_dict(), "image_only")["hallucination_rate"] == 0.5
    # Before: the image_only reading was left out, and the rate read 0 of 4.
    assert report.gate_metrics()["hallucination_rate"] == pytest.approx(1 / 6)


def test_a_golden_document_with_no_page_text_is_unscored_not_clean(monkeypatch):
    from evaluation import golden_eval

    _fake_extract(monkeypatch, [], image_only_number="ZZ-404-9")
    triples = golden_eval.evaluate([_golden(text="")], object(), object(), "/tmp",
                                   modes=("image_only",), run_to_run=False)
    assert triples[0][2][PRINTED_CHECK] is None
    report = build_report("v1", triples)
    assert "hallucination_rate" not in report.gate_metrics()
    assert report.invention_coverage()["unscored_readings"] == 1


def test_the_gates_repeat_reads_image_only_and_the_report_records_it(monkeypatch):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import golden_eval, run_eval

    calls: list[tuple[str, str, str]] = []
    _fake_extract(monkeypatch, calls)
    monkeypatch.setattr(golden_eval, "_refuse_interrupted_freeze", lambda *a, **k: None)
    monkeypatch.setattr(run_eval, "assert_eval_set_disjoint", lambda *a, **k: None)
    monkeypatch.setattr(golden_eval, "load_golden_set", lambda *a, **k: [_golden("p1"), _golden("p2")])
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    body = golden_eval.evaluate_version(client, object(), version="v1", corpus_version="v1")
    assert [c for c in calls if c[2] == "reversed"] == [("p1", "image_only", "reversed"),
                                                         ("p2", "image_only", "reversed")]
    assert body["run_to_run_check"] is True and body["run_to_run_mode"] == "image_only"
    assert body["eval_set"]["run_to_run_mode"] == "image_only"
    assert body["gate_metrics"]["run_to_run_agreement"] == 1.0
    assert body["invention_coverage"] == {"borrowed_from": "ocr_plus_image", "borrowed_text_readings": 2,
                                          "cited_pages_without_text": 0, "unscored_readings": 0}


def test_the_repeat_mode_is_configured_and_refused_when_unknown(monkeypatch):
    from common.config import training_config
    from evaluation import golden_eval

    for profile in ("standard", "fast"):
        _pin(monkeypatch, profile)
        assert training_config("unified")["evaluation"]["run_to_run_mode"] == "image_only"
        assert golden_eval.run_to_run_mode() == "image_only"
    monkeypatch.setattr(common.config, "training_config", lambda name: {"evaluation": {"run_to_run_mode": "fax"}})
    with pytest.raises(golden_eval.GoldenEvalError, match="run_to_run_mode is 'fax'"):
        golden_eval.run_to_run_mode()
    monkeypatch.setattr(common.config, "training_config", lambda name: {"evaluation": {}})
    assert golden_eval.run_to_run_mode() == "image_only"                      # the default


def test_a_report_made_on_another_repeat_mode_is_another_yardstick(monkeypatch):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import golden_eval
    from evaluation.freeze_eval_set import manifest_key
    from evaluation.run_eval import eval_set_record, scoring_fingerprint, yardstick_difference
    from orchestration.pipeline_dag import _other_yardstick

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    client.write_json(manifest_key(), {"frozen_at": "2026-10-01T00:00:00+00:00", "frozen_from_corpus": "v1"})
    now = eval_set_record(client, None, [])
    assert now["run_to_run_mode"] == "image_only"
    monkeypatch.setattr(golden_eval, "run_to_run_mode", lambda: "ocr_plus_image")
    before = eval_set_record(client, None, [])
    assert before["scoring_fingerprint"] == scoring_fingerprint() != now["scoring_fingerprint"]
    # Compared (promoted against candidate), and reused (the gate's own report).
    assert "run-to-run repeat in different modes" in yardstick_difference(before, now, None)
    assert "fingerprint" in _other_yardstick(before, now)
    assert _other_yardstick(before, dict(before)) is None
    assert yardstick_difference(before, dict(before), None) is None


def test_the_repeat_answer_rides_on_the_image_only_reading(monkeypatch):
    from evaluation import golden_eval

    calls: list[tuple[str, str, str]] = []
    _fake_extract(monkeypatch, calls)
    triples = golden_eval.evaluate([_golden()], object(), object(), "/tmp", run_to_run=True)
    assert [m["modality_mode"] for _e, _g, m in triples if REPEAT_ANSWER in m] == ["image_only"]
    # A pass that does not read the repeat mode reads nothing twice.
    calls.clear()
    golden_eval.evaluate([_golden()], object(), object(), "/tmp", modes=("ocr_plus_image",), run_to_run=True)
    assert [c[2] for c in calls] == ["planned"]


def test_the_gate_scores_an_image_only_reading_as_it_is_served(monkeypatch):
    """Production serves an image_only read with every value no page prints
    taken out (strict): the gate scores that, not the model's raw answer
    (decision 2026-10-26). Invention is still counted from the record, and the
    repeat answer is held the same way, so the two compare as served."""
    from evaluation import golden_eval

    _fake_extract(monkeypatch, [], image_only_number="ZZ-404-9")
    triples = golden_eval.evaluate([_golden()], object(), object(), "/tmp", run_to_run=True)
    [(_gold, served, meta)] = [t for t in triples if t[2]["modality_mode"] == "image_only"]
    assert served["policy"]["policy_number"]["parsed"] is None              # invented, taken out
    assert served["policy"]["effective_date"]["parsed"] == "01/01/2026"     # printed, kept
    assert [e["raw"] for e in meta[PRINTED_CHECK]["not_printed"]] == ["ZZ-404-9"]
    assert meta[REPEAT_ANSWER]["policy"]["policy_number"]["parsed"] is None
    assert build_report("v1", triples).gate_metrics()["hallucination_rate"] == pytest.approx(1 / 6)
