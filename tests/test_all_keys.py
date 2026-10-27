"""Every served canonical JSON carries every key of its line's schema (common.canonical.with_all_keys).

A value the model did not extract is null, not absent, so the base model and a
trained one return the same keys for the same line; only values, and the number
of table rows found, differ.
"""

from __future__ import annotations

from common.canonical import SKELETON_SOURCE, empty_field_value, with_all_keys
from common.schemas import is_valid, load_schema


def _env(value, page=1, score=0.9):
    return {"raw": str(value), "parsed": value, "page_ref": [page],
            "confidence": {"score": score, "source": "vlm"}, "flagged": False}


def _keys(node, path=""):
    if isinstance(node, dict) and {"raw", "parsed", "page_ref"} <= set(node):
        return {path}
    out = set()
    if isinstance(node, dict):
        for key, value in node.items():
            out |= _keys(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for value in node:
            out |= _keys(value, path + "[]")
    return out


# A self-contained line: the common-model lines have their own case (Phase 6).
LINE = "equipment_breakdown"
SCHEMA = load_schema("policy", None, LINE)


def test_two_different_answers_get_the_same_keys():
    sparse = with_all_keys({"policy": {"policy_number": _env("HO-1")}}, SCHEMA)
    fuller = with_all_keys({"policy": {"policy_number": _env("HO-1"), "effective_date": _env("01/01/2026")},
                            "carrier": {"company_name": _env("Northfield Mutual")}}, SCHEMA)
    assert _keys(sparse) == _keys(fuller) and len(_keys(sparse)) > 100


def test_a_missing_value_is_an_empty_envelope_and_a_present_one_is_untouched():
    out = with_all_keys({"policy": {"policy_number": _env("HO-1")}}, SCHEMA)
    assert out["policy"]["policy_number"] == _env("HO-1")
    empty = out["policy"]["effective_date"]
    assert empty == empty_field_value()
    assert empty["raw"] is None and empty["confidence"] == {"score": 0.0, "source": SKELETON_SOURCE}
    assert empty["flagged"] is False


def test_a_missing_table_is_empty_and_each_row_found_gets_every_key():
    out = with_all_keys({"forms_and_endorsements": [{"form_number": _env("HO 00 03")}]}, SCHEMA)
    (row,) = out["forms_and_endorsements"]
    assert row["form_number"]["raw"] == "HO 00 03" and row["form_title"]["raw"] is None
    assert out["locations"] == []                 # rows are never invented


def test_keys_follow_schema_order():
    out = with_all_keys({"policy": {"expiration_date": _env("01/01/2027")}}, SCHEMA)
    # text_sections is the client's full-text tier: filled by the text path, never
    # by extraction (fideon:fsm_exclude), so the fill leaves it to that path.
    filled = [k for k, v in SCHEMA["properties"].items() if not v.get("fideon:fsm_exclude")]
    assert list(out)[: len(filled)] == filled


def test_the_filled_output_is_valid_against_the_full_schema():
    out = with_all_keys({"policy": {"policy_number": _env("HO-1")}}, SCHEMA)
    assert is_valid(out, "policy", None, LINE)


def test_serving_returns_every_key_for_a_canonical_policy():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_serving_pipeline import CALIBRATION

    answer = '{"policy": {"policy_number": {"raw": "HO-1", "parsed": "HO-1", "page_ref": [1]}}}'
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load_model("base", client, backend_impl=EchoBackend(answer))
    request = ExtractionRequest(source_id="p1", image_paths=["d/page_1.png"], page_texts={1: "HO-1"},
                                known_doc_type="policy", known_lob=LINE)
    result = extract(request, model, StaticClassifier("policy"), CALIBRATION, strict_schema=False)
    keys = _keys(result.extraction)
    assert "policy.effective_date" in keys and "carrier.company_name" in keys
    assert result.extraction["policy"]["effective_date"]["parsed"] is None
