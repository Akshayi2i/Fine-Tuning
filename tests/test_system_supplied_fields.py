"""document.source_file_name and document.page_count: supplied by the system, never
taught to the model, never scored (common.canonical.SYSTEM_SUPPLIED_FIELDS)."""

from __future__ import annotations

from common.canonical import to_model_target, with_system_fields, without_system_fields
from common.schemas import is_valid
from evaluation.run_eval import build_report


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


LABEL = {"document": {"document_type": _env("Declarations"),
                      "source_file_name": _env("1- HOME_redacted.pdf"),
                      "page_count": _env(85)},
         "policy": {"policy_number": _env("HO-1")}}


def test_training_targets_never_carry_them():
    target = to_model_target(LABEL)
    assert "source_file_name" not in target["document"] and "page_count" not in target["document"]
    assert target["document"]["document_type"]["raw"] == "Declarations"
    assert "page_count" in LABEL["document"]                       # the label itself untouched


def test_window_targets_never_carry_them():
    from data_pipeline.dataset_builder.policy_windows import plan_windows, window_target

    for plan in plan_windows("gl", [1, 2, 3]):
        document = window_target(LABEL, "gl", plan).get("document", {})
        assert "source_file_name" not in document and "page_count" not in document


def test_they_are_not_scored_on_either_side():
    """A gold label still carrying them must not mark every answer wrong on them."""
    answer = {"document": {"document_type": _env("Declarations")}, "policy": {"policy_number": _env("HO-1")}}
    meta = {"source_id": "p1", "doc_type": "policy", "lob": "gl", "modality_mode": "ocr_plus_image"}
    [full] = build_report("t", [(LABEL, answer, meta)]).full_set()
    assert full.metrics["field_normalized_match"] == 1.0
    assert full.metrics["false_null_rate"] == 0.0


#: A line still on a self-contained schema, which declares these two as
#: envelopes. A common-model line - general liability, property and the
#: fallback among them - declares them as plain values
#: (tests/test_schema2_serving.py).
SELF_CONTAINED = "equipment_breakdown"


def test_serving_fills_them_from_the_request_into_a_valid_canonical_output():
    required = {"carrier": {}, "named_insured": {}, "policy": {}}      # the schema's required objects
    output = with_system_fields({"document": {}, **required}, page_count=85,
                                source_file_name="1- HOME_redacted.pdf", lob=SELF_CONTAINED)
    page_count = output["document"]["page_count"]
    assert page_count["parsed"] == 85 and page_count["confidence"] == {"score": 1.0, "source": "deterministic"}
    assert output["document"]["source_file_name"]["raw"] == "1- HOME_redacted.pdf"
    assert is_valid(output, "policy", None, SELF_CONTAINED)


def test_an_unknown_file_name_is_left_out():
    output = with_system_fields({}, page_count=3, source_file_name=None, lob=SELF_CONTAINED)
    assert "source_file_name" not in output["document"] and output["document"]["page_count"]["parsed"] == 3


def test_stripping_leaves_a_document_without_them_unchanged():
    plain = {"policy": {"policy_number": _env("HO-1")}}
    assert without_system_fields(plain) is plain
