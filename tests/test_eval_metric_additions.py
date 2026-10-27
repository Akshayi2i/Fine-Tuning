"""Precision, recall and F1 for single fields, table precision, accuracy by line and
weakest fields, and hallucination measured only where text was sent (evaluation.run_eval)."""

from __future__ import annotations

import pytest

from evaluation.run_eval import build_report


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


LABEL = {"policy": {"policy_number": _env("HO-1"), "effective_date": _env("01/01/2026"),
                    "expiration_date": _env("01/01/2027"), "account_id": _env("1200")},
         "forms_and_endorsements": [{"form_number": _env("HO 00 03")}, {"form_number": _env("HO 04 90")}]}


#: A self-contained line, scored on the label above as it is: a common-model
#: line's report narrows it to the common model's shape first.
LINE = "equipment_breakdown"


def _meta(source="p1", lob=LINE, ocr="HO-1 01/01/2026 01/01/2027 1200", mode="ocr_plus_image"):
    return {"source_id": source, "doc_type": "policy", "lob": lob, "modality_mode": mode,
            "ocr_text": ocr}


def _metrics(documents):
    [full] = build_report("t", documents).full_set()
    return full.metrics


def test_precision_recall_and_f1_separate_writing_wrong_from_leaving_out():
    got = {"policy": {"policy_number": _env("HO-1"),          # right
                      "effective_date": _env("02/02/2026"),   # written, wrong
                      "prior_policy_number": _env("HO-1")}}   # written, not in the label
    # expiration_date and account_id left out
    m = _metrics([(LABEL, got, _meta())])
    assert m["field_precision"] == pytest.approx(1 / 3)       # 1 right of 3 written
    assert m["field_recall"] == pytest.approx(1 / 4)          # 1 found of 4 in the label
    assert m["field_f1"] == pytest.approx(2 * 1 / (3 + 4))


def test_table_precision_counts_only_rows_the_model_wrote():
    got = {"forms_and_endorsements": [{"form_number": _env("HO 00 03")}, {"form_number": _env("XX 99")}]}
    m = _metrics([(LABEL, got, _meta())])
    assert m["list_field_precision"] == 0.5 and m["list_field_recall"] == 0.5


def test_accuracy_is_broken_down_by_line_and_by_field():
    perfect = {"policy": dict(LABEL["policy"])}
    no_premium = {"policy": {k: v for k, v in LABEL["policy"].items() if k != "account_id"}}
    m = _metrics([(LABEL, perfect, _meta("p1", LINE)),
                  (LABEL, no_premium, _meta("p2", LINE)),
                  (LABEL, {}, _meta("p3", "agriculture_farm"))])
    assert m["field_accuracy_by_lob"] == {LINE: 0.875, "agriculture_farm": 0.0}
    weakest = m["weakest_fields"]
    assert weakest[0]["field"] == "policy.account_id"          # right once in 3
    assert weakest[0]["accuracy"] == pytest.approx(1 / 3, abs=1e-4) and weakest[0]["scored"] == 3


def test_hallucination_is_measured_only_against_text_that_was_sent():
    """An image-only row sends no text; every value it reads off the image would
    otherwise count as invented."""
    read_off_image = {"policy": {"policy_number": _env("HO-7")}}
    m = _metrics([(LABEL, read_off_image, _meta("p1", ocr="", mode="image_only")),
                  (LABEL, {"policy": {"program_name": _env("Nowhere Agency")}}, _meta("p2"))])
    assert m["hallucination_rate"] == 1.0          # the one checked value (p2) is invented
    only_image = _metrics([(LABEL, read_off_image, _meta("p1", ocr="", mode="image_only"))])
    assert "hallucination_rate" not in only_image
