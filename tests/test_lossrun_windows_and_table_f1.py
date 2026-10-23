"""A Loss Run longer than one window is read by window, merged and reconciled,
and evaluation reports its claims table F1 (Fideon SPEC_09 handoff item 8)."""

from __future__ import annotations

import json
import re

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from common.adapter import not_applicable_metrics
from evaluation.gating import CONDITIONAL_METRICS, GATING_METRICS, PILOT_FLOORS, promotion_gate
from evaluation.metrics.lossrun_table import score_claims, table_f1
from inference_core.model_runner import Generation, ModelBackend, load_model
from serving.doc_type_classifier import StaticClassifier
from serving.pipeline import LOSSRUN_MERGE_CONFLICT_FLAG, ExtractionRequest, extract, lossrun_windows

HEADER = {"carrier": "Sentinel Insurance Company", "policy_number": "WC-8842317-01",
          "insured_name": "Rivera Fabrication LLC", "effective_date": "2025-04-01",
          "expiration_date": "2026-04-01", "line_of_business": ["workers_comp"],
          "valuation_date": "2026-03-31", "total_claims_reported": 14, "line_of_business_other": []}


def _claim(page, n, total=None):
    amount = float(total if total is not None else page * 1000 + n)
    return {"claim_number": f"WC24-{page:02d}{n}", "loss_date": "2024-02-14", "status": "closed",
            "paid": amount, "reserved": 0.0, "total_incurred": amount, "description": f"claim {page}.{n}"}


def _page_text(page, rows=12):
    """A page whose table holds ``rows`` claim rows. Twelve on each of seven
    pages is 84 rows, more than one answer can hold, so the document is windowed."""
    body = "\n".join(f"| WC24-{page:02d}{n} | 02/14/2024 | {page * 1000 + n}.00 |" for n in range(1, rows + 1))
    return f"Loss Run page {page}\n\n| Claim | Loss date | Incurred |\n|---|---|---|\n{body}"


PAGES = 7
GOLD = {**HEADER, "claims": [_claim(p, n) for p in range(1, PAGES + 1) for n in (1, 2)]}


class _Windows(ModelBackend):
    """Reads each window: the header where page 1 is in it, the claims of its pages."""

    def __init__(self, misread=None):
        self.misread = misread or {}
        self.windows: list[list[int]] = []
        self.caps: list[int] = []

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        pages = [int(n) for n in re.findall(r"<page (\d+) of", json.dumps(messages[1]["content"]))]
        self.windows.append(pages)
        self.caps.append(config.max_new_tokens)
        header = HEADER if 1 in pages else {k: ([] if isinstance(v, list) else None) for k, v in HEADER.items()}
        claims = [_claim(p, n, self.misread.get((tuple(pages), p, n))) for p in pages for n in (1, 2)]
        text = json.dumps({**header, "claims": claims})
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens), finish_reason="stop")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _request(pages=PAGES, rows=12, **over):
    request = dict(
        source_id="lossrun_0001",
        image_paths=[f"processed/default/lossrun/lossrun_0001/page_{p}.png" for p in range(1, pages + 1)],
        page_texts={p: _page_text(p, rows) for p in range(1, pages + 1)},
        known_doc_type="lossrun",
    )
    return ExtractionRequest(**{**request, **over})


def _serve(client, backend, pages=PAGES, **over):
    model = load_model("base", client, backend_impl=backend)
    return extract(_request(pages, **over), model, StaticClassifier("lossrun"),
                   CalibrationParams(method="temperature", doc_type="lossrun", model_version="v1",
                                     temperature=1.0), strict_schema=False)


def test_windows_are_planned_from_row_density():
    assert lossrun_windows(_request()) == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    # Denser pages, smaller windows: 30 rows a page is a page a window.
    assert lossrun_windows(_request(pages=4, rows=30)) == [[1], [2], [3], [4]]


def test_a_loss_run_whose_rows_fit_one_answer_is_read_in_one_call(client):
    """Windows are a shape no training row has; they are only for what one call cannot hold."""
    assert lossrun_windows(_request(pages=4, rows=10)) == [[1, 2, 3, 4]]   # 40 rows fit
    backend = _Windows()
    _serve(client, backend, pages=4, rows=10)
    assert backend.windows == [[1, 2, 3, 4]]


def test_every_window_may_write_the_rows_it_was_planned_for(client):
    from serving.pipeline import lossrun_window_budget

    backend = _Windows()
    _serve(client, backend)
    assert len(backend.caps) == 3 and set(backend.caps) == {lossrun_window_budget()}


def test_without_page_texts_a_loss_run_takes_the_one_call_path_and_its_refusals(client):
    from serving.pipeline import PipelineError

    image_only = _request(page_texts={}, modality_mode="image_only")
    assert lossrun_windows(image_only) == [list(range(1, PAGES + 1))]
    with pytest.raises(PipelineError, match="single joined ocr_text"):
        _serve(client, _Windows(), page_texts={}, ocr_text="every page joined")


def test_a_multi_window_loss_run_is_read_merged_and_reconciled_end_to_end(client):
    backend = _Windows()
    result = _serve(client, backend)
    assert backend.windows == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    claims = result.extraction["claims"]
    assert len(claims) == 14                                         # pages 3 and 5 read twice, kept once
    assert {c["claim_number"] for c in claims} == {c["claim_number"] for c in GOLD["claims"]}
    assert result.extraction["carrier"] == HEADER["carrier"]         # from the window that holds page 1
    assert result.schema_valid and result.pages_used == list(range(1, PAGES + 1))
    assert result.reconciliation["status"] == "unverifiable"         # the schema carries no printed totals
    assert table_f1([(GOLD, result.extraction)]).f1 == 1.0


def test_merged_rows_keep_the_spans_of_the_window_rows_they_came_from():
    from serving.lossrun_merge import merge_extracted_windows

    first = ({**HEADER, "claims": [_claim(1, 1), _claim(3, 1)]},
             {"carrier": "s-carrier", "claims[0].paid": "s1", "claims[1].paid": "s3a"})
    second = ({**{k: None for k in HEADER}, "claims": [_claim(3, 1), _claim(4, 1)]},
              {"carrier": "s-null", "claims[0].paid": "s3b", "claims[1].paid": "s4"})
    extraction, spans, merged, _ = merge_extracted_windows([first, second])
    numbers = [c["claim_number"] for c in extraction["claims"]]
    assert numbers == ["WC24-011", "WC24-031", "WC24-041"] and merged.duplicates_collapsed == 1
    assert spans["carrier"] == "s-carrier"
    assert [spans[f"claims[{i}].paid"] for i in range(3)] == ["s1", "s3a", "s4"]


def test_two_windows_reading_one_claim_differently_is_flagged(client):
    backend = _Windows(misread={((3, 4, 5), 3, 1): 9999.0})
    result = _serve(client, backend)
    assert LOSSRUN_MERGE_CONFLICT_FLAG in result.review_flags


def test_a_loss_run_within_one_window_is_read_in_one_call(client):
    backend = _Windows()
    result = _serve(client, backend, pages=2)
    assert backend.windows == [[1, 2]] and len(result.extraction["claims"]) == 4
    assert result.reconciliation is not None


# --------------------------------------------------------------------------
# Table F1 and the gate
# --------------------------------------------------------------------------

def test_a_claim_counts_only_with_its_number_and_its_total_within_a_cent():
    got = {"claims": [_claim(1, 1), _claim(1, 2, total=1002.009), _claim(2, 1, total=2001.5),
                      {"claim_number": "WC24-999", "total_incurred": 5.0}]}
    gold = {"claims": [_claim(1, 1), _claim(1, 2), _claim(2, 1), _claim(2, 2)]}
    score = score_claims(gold, got)
    assert (score.matched, score.extracted, score.expected) == (2, 4, 4)
    assert score.f1 == 0.5


def test_a_loss_run_evaluation_reports_table_f1_and_the_gate_floors_it_at_088(client):
    from evaluation.golden_eval import evaluate  # noqa: F401 - the path that carries reconciliation
    from evaluation.run_eval import build_report

    result = _serve(client, _Windows())
    metadata = {"source_id": "lossrun_0001", "doc_type": "lossrun", "modality_mode": "ocr_plus_image",
                "page_count": PAGES, "reconciliation": result.reconciliation}
    metrics = build_report("v1", [(GOLD, result.extraction, metadata)]).gate_metrics()
    assert metrics["table_f1"] == 1.0
    assert "lossrun_totals_reconciliation_rate" not in metrics       # unverifiable: no evidence, no score
    assert GATING_METRICS["table_f1"] == "higher_is_better" and PILOT_FLOORS["table_f1"] == 0.88
    # Conditional: an eval set with no Loss Run has none to report, and the
    # policy adapter reads no Loss Run. Present, it still faces the floor.
    assert "table_f1" in CONDITIONAL_METRICS
    assert "table_f1" in not_applicable_metrics()
    passing = {name: 0.96 for name in GATING_METRICS}
    passing.update(schema_validity_rate=1.0, ece_confidence=0.04, confusable_misattribution_rate=0.02,
                   false_null_rate=0.03, auto_accept_error_rate=0.01)
    assert "table_f1" in promotion_gate({**passing, "table_f1": 0.87}, None).failed_gates
    assert "table_f1" not in promotion_gate({**passing, "table_f1": 0.89}, None).failed_gates


def test_the_reconciliation_report_reaches_the_evaluation(client, monkeypatch):
    """golden_eval copies the serving path's report into the scored metadata."""
    from evaluation import golden_eval
    from evaluation.golden_eval import GoldenDocument

    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})
    model = load_model("base", client, backend_impl=_Windows())
    doc = GoldenDocument(source_id="lossrun_0001", doc_type="lossrun", golden=GOLD,
                         image_keys=[f"processed/default/lossrun/lossrun_0001/page_{p}.png"
                                     for p in range(1, PAGES + 1)],
                         page_texts={p: _page_text(p) for p in range(1, PAGES + 1)})
    triples = golden_eval.evaluate([doc], model, client, "/tmp", modes=("ocr_plus_image",))
    assert triples[0][2]["reconciliation"]["status"] == "unverifiable"


def test_claims_without_a_number_match_on_their_loss_date():
    """A redacted Loss Run: a perfect answer scores 1.0, a wrong total still misses."""
    redacted = {"claims": [{"claim_number": None, "loss_date": "2024-02-14", "total_incurred": 100.0},
                           {"claim_number": None, "loss_date": "2024-03-01", "total_incurred": 0.0}]}
    assert score_claims(redacted, redacted).f1 == 1.0
    wrong = {"claims": [{**redacted["claims"][0], "total_incurred": 150.0}, redacted["claims"][1]]}
    assert score_claims(redacted, wrong).matched == 1
