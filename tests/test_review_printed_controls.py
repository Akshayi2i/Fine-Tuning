"""Serving's invention controls beyond "printed somewhere" (review 2026-10-26):
raw read against parsed (#18), right value in the wrong place or role (#19),
coverage form numbers (#20), rows of scattered fragments and the compact
yes/no envelope (#21), and the printed_texts a request may be held to (I6)."""

from __future__ import annotations

import copy
import json

import pytest

import serving.printed_values as printed_values
from common.canonical import expand_answer
from serving.printed_values import hold_result_to_printed, hold_to_printed


def _fv(raw, page=1, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [page]}


PAGES = {
    1: "DECLARATIONS\nGranite Mutual Insurance Company\nPolicy Number PA-2026-0042\n"
       "Effective Date: 04/01/2026\nExpiration Date: 10/01/2026\nTotal Premium $1,665.00",
    2: "SCHEDULE OF VEHICLES\nVeh 1 2019 Honda Accord VIN 1HGCV1F30KA000001\n"
       "Collision Deductible $500 Form CA 00 01",
    3: "",
}


@pytest.fixture
def flag_mode(monkeypatch):
    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)


# --------------------------------------------------------------------------
# #18 raw read against parsed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "raw", "parsed", "wrong"), [
    ("DateValue", "04/05/2026", "05/04/2026", True),          # day and month swapped
    ("DateValue", "04/05/2026", "04/05/2026", False),
    ("DateValue", "April 5, 2026", "04/05/2026", False),
    ("DateValue", "13/04/2026", "04/13/2026", False),         # read as it can only be read
    ("DateValue", "Full Prior Acts", None, False),
    ("DateValue", "01/01/2026 12:01 AM", "02/01/2026", False),  # not a clean date: nothing settled
    ("MoneyValue", "$1,000", 10000, True),
    ("MoneyValue", "$1,000", 1000.0, False),
    ("MoneyValue", "(109.00)", -109.0, False),                 # a credit, sign aside
    ("MoneyValue", "(109.00)", 109.0, False),
    ("MoneyValue", "$1M", 1000000, False),                     # not one clean figure
    ("MoneyValue", "100/300", 100000, False),
    ("MoneyValue", "Included", None, False),
    ("YearValue", "2019", 2021, True),
    ("YearValue", "2019", "2019", False),
    ("NumberValue", "1,250", 1250, False),
    ("PercentValue", "5%", 5, False),
    ("PercentValue", "5%", 0.05, False),
    ("PercentValue", "5%", 50, True),
    ("StateValue", "New York", "NY", False),
    ("StateValue", "New York", "NJ", True),
    ("StateValue", "N.J.", "NJ", False),
    ("StateValue", "Toledo, OH", "NY", False),                 # not a state alone
    ("PostalCodeValue", "12180-1234", "12180-1234", False),
    ("PostalCodeValue", "12180", "12181", True),
    ("PostalCodeValue", "12180-1234", "12180-9999", True),
    ("PostalCodeValue", "12180-1234", "12180", False),
])
def test_raw_is_read_by_its_type_and_only_a_clear_contradiction_counts(kind, raw, parsed, wrong):
    from serving.raw_parsed import contradicts

    assert contradicts(kind, raw, parsed) is wrong


def test_the_declared_type_decides_and_a_plain_field_is_read_by_its_name():
    from serving.raw_parsed import kind_of

    assert kind_of("policy.effective_date") == "DateValue"
    assert kind_of("coverages[3].limits[0].amount") == "MoneyValue"
    assert kind_of("locations[0].address.postal_code") == "PostalCodeValue"
    assert kind_of("vehicles[0].vin") is None                  # text: never compared
    assert kind_of("some_block.renewal_date") == "DateValue"   # by name, as the audit reads it


def test_strict_takes_out_a_parse_its_print_contradicts_and_keeps_its_row():
    answer = {
        "premium": {"total": _fv("$1,665.00", 1, 16650.0)},
        "policy": {"effective_date": _fv("04/01/2026", 1, "01/04/2026"), "policy_number": _fv("PA-2026-0042")},
        "coverages": [{"coverage_id": "cov_1", "coverage_code": "COLL", "coverage_name": _fv("Collision", 2),
                       "deductibles": [{"deductible_type": "flat", "amount": _fv("$500", 2, 5000)}]}],
    }
    held = hold_to_printed(answer, {"premium.total": "span"}, PAGES)

    assert held.extraction["premium"]["total"] is None and "premium.total" not in held.spans
    assert held.extraction["policy"]["effective_date"] is None
    assert held.extraction["policy"]["policy_number"]["raw"] == "PA-2026-0042"
    # Its print is on the page, so its row stays; the value alone goes.
    assert held.extraction["coverages"][0]["deductibles"][0]["amount"] is None
    assert {"premium.total:parsed_mismatch", "policy.effective_date:parsed_mismatch",
            "coverages[0].deductibles[0].amount:parsed_mismatch"} <= set(held.flags)
    assert held.record["parsed_mismatch"] == 3
    # Not an invention: the invented rate's terms are untouched.
    assert held.record["not_printed"] == []


def test_flag_mode_keeps_a_contradicted_parse_and_flags_it(flag_mode):
    held = hold_to_printed({"premium": {"total": _fv("$1,665.00", 1, 16650.0)}}, {}, PAGES)

    assert held.extraction["premium"]["total"]["parsed"] == 16650.0
    assert held.flags == ["premium.total:parsed_mismatch"] and held.record["parsed_mismatch"] == 1


def test_a_compact_answer_s_r_is_held_to_its_v():
    """r "$1,000" with v 10000: expanded, raw is the r - and v no longer ships unread."""
    window = {"premium": {"total": {"r": "$1,665.00", "v": 16650, "p": [1]}},
              "policy": {"policy_number": {"v": "PA-2026-0042", "p": [1]}}}
    held = hold_to_printed(expand_answer(window), {}, PAGES)

    assert held.extraction["premium"]["total"] is None
    assert "premium.total:parsed_mismatch" in held.flags
    assert held.extraction["policy"]["policy_number"]["raw"] == "PA-2026-0042"   # r left out: v is its print


def test_an_unreadable_raw_settles_nothing():
    answer = {"premium": {"total": _fv("Included", 1, 1665.0)}}
    held = hold_to_printed(answer, {}, {1: "Total Premium Included"})

    assert held.extraction["premium"]["total"]["parsed"] == 1665.0
    assert not any(flag.endswith(":parsed_mismatch") for flag in held.flags)


# --------------------------------------------------------------------------
# #19 printed off the cited pages; the wrong role
# --------------------------------------------------------------------------


def test_a_value_printed_only_off_its_cited_pages_is_kept_flagged_and_counted():
    answer = {"premium": {"total": _fv("$1,665.00", 2)}, "policy": {"policy_number": _fv("PA-2026-0042", 1)}}
    held = hold_to_printed(answer, {}, PAGES)

    assert held.extraction["premium"]["total"]["raw"] == "$1,665.00"
    assert held.flags == ["premium.total:printed_off_cited_page"]
    assert held.record["off_cited_page"] == 1 and held.record["not_printed"] == []


def test_a_blank_cited_page_says_nothing_about_the_citation():
    """Page 3 has no text: it may print the value too, so no flag."""
    held = hold_to_printed({"premium": {"total": _fv("$1,665.00", 3)}}, {}, PAGES)

    assert held.extraction["premium"]["total"]["raw"] == "$1,665.00"
    assert not any(flag.endswith(":printed_off_cited_page") for flag in held.flags)


def test_a_served_result_sends_an_off_page_value_to_review():
    from serving.pipeline import ExtractionResult

    served = {"premium": {"total": {**_fv("$1,665.00", 2), "confidence": {"score": 0.99, "source": "vlm"},
                                    "flagged": False}},
              "policy": {"policy_number": {**_fv("PA-2026-0042", 1), "confidence": {"score": 0.99, "source": "vlm"},
                                           "flagged": False}}}
    result = ExtractionResult(source_id="x", doc_type="policy", model_version="m", mode="image_only",
                              schema_valid=True, overall_confidence=0.99, extraction=served)
    hold_result_to_printed(result, PAGES)

    assert "premium.total:printed_off_cited_page" in result.review_flags
    assert result.extraction["premium"]["total"]["flagged"] is True
    assert result.extraction["policy"]["policy_number"]["flagged"] is False


def test_routing_sends_a_list_of_values_whole_and_never_a_value_taken_out():
    from serving.value_flags import route_served, routed_paths

    flags = ["vehicles[0].symbols[1]:near_tie", "policy.x[#0]:printed_off_cited_page", "policy.y:not_printed",
             "policy.z:role_mismatch"]
    assert routed_paths(flags) == {"vehicles[0].symbols[1]", "vehicles[0].symbols", "policy.z"}
    served = {"vehicles": [{"symbols": [{"raw": "A", "parsed": "A", "flagged": False},
                                        {"raw": "B", "parsed": "B", "flagged": False}]}],
              "policy": {"z": {"raw": "1", "parsed": 1, "flagged": False},
                         "y": {"raw": "2", "parsed": 2, "flagged": False},
                         "w": {"raw": None, "parsed": None, "flagged": False}}}
    assert route_served(served, [*flags, "policy.w:role_mismatch"]) == 3
    assert [s["flagged"] for s in served["vehicles"][0]["symbols"]] == [True, True]
    assert served["policy"]["z"]["flagged"] and not served["policy"]["y"]["flagged"]
    assert served["policy"]["w"]["flagged"] is False                 # a null serves nothing to review


@pytest.mark.parametrize(("served", "pages", "flagged"), [
    ("10/01/2026", {1: "Effective Date: 04/01/2026\nExpiration Date: 10/01/2026"}, True),
    ("04/01/2026", {1: "Effective Date: 04/01/2026\nExpiration Date: 10/01/2026"}, False),
    # An endorsement's own effective date beside the declarations': one line prints it.
    ("04/01/2026", {1: "Effective Date: 04/01/2026", 2: "Effective Date: 06/15/2026"}, False),
    # The label alone on its line, the value on the next.
    ("10/01/2026", {1: "Effective Date\n04/01/2026"}, True),
    # A table header: no value follows the label on its line - nothing to hold it to.
    ("10/01/2026", {1: "| Effective Date | Expiration Date |\n|---|---|\n| 04/01/2026 | 10/01/2026 |"}, False),
])
def test_a_value_served_under_a_label_printing_another_is_flagged(served, pages, flagged):
    from serving.conventions import role_mismatches

    extraction = {"policy": {"effective_date": _fv(served)}}
    flags = role_mismatches(extraction, pages, sorted(pages), ["policy"])
    assert flags == (["policy.effective_date:role_mismatch"] if flagged else [])


def test_the_role_check_compares_identifiers_and_amounts_as_what_they_read():
    from serving.conventions import role_mismatches

    pages = {1: "Policy Number: PA-2026-0042\nPrior Policy Number: PA-2025-0042\nTotal Premium: $1,665.00"}
    wrong = {"policy": {"policy_number": _fv("PA-2025-0042"), "prior_policy_number": _fv("PA-2025-0042")},
             "premium": {"total": _fv("$1,665", parsed=1665.0)}}
    assert role_mismatches(wrong, pages, [1], ["policy", "premium"]) == ["policy.policy_number:role_mismatch"]


# --------------------------------------------------------------------------
# #20 coverage form numbers
# --------------------------------------------------------------------------


def _with_forms():
    return {"coverages": [{"coverage_id": "cov_1", "coverage_code": "COLL", "coverage_name": _fv("Collision", 2),
                           "form_refs": ["CA 00 01", "ZZ 99 99"]}]}


def test_a_form_number_no_page_prints_leaves_its_list_counted_apart():
    held = hold_to_printed(_with_forms(), {}, PAGES)

    assert held.extraction["coverages"][0]["form_refs"] == ["CA 00 01"]
    assert "coverages[0].form_refs[#1]:not_printed_removed" in held.flags
    assert held.record["form_refs"] == {"checked": 2, "not_printed": [
        {"path": "coverages[0].form_refs[#1]", "raw": "ZZ 99 99", "pages": [2]}]}
    # The envelopes' counts - the invented rate - are as before.
    assert held.record["checked"] == 1 and held.record["not_printed"] == []


def test_flag_mode_keeps_a_form_number_and_flags_it(flag_mode):
    held = hold_to_printed(_with_forms(), {}, PAGES)

    assert held.extraction["coverages"][0]["form_refs"] == ["CA 00 01", "ZZ 99 99"]
    assert "coverages[0].form_refs[1]:not_printed" in held.flags


def test_a_served_result_loses_an_unprinted_form_number_from_its_rows_too():
    from serving.pipeline import ExtractionResult

    served = _with_forms()
    served["coverages"][0]["coverage_name"].update(confidence={"score": 0.9, "source": "vlm"}, flagged=False)
    result = ExtractionResult(source_id="x", doc_type="policy", model_version="m", mode="image_only",
                              schema_valid=True, overall_confidence=0.9, extraction=served,
                              list_fields={"coverages": {"rows": [{"coverage_id": "cov_1", "coverage_code": "COLL",
                                                                    "coverage_name": "Collision",
                                                                    "form_refs": ["CA 00 01", "ZZ 99 99"]}],
                                                         "extracted_rows": 1}})
    hold_result_to_printed(result, PAGES)

    assert result.extraction["coverages"][0]["form_refs"] == ["CA 00 01"]
    assert result.list_fields["coverages"]["rows"][0]["form_refs"] == ["CA 00 01"]


# --------------------------------------------------------------------------
# #21 rows of fragments; the compact yes/no
# --------------------------------------------------------------------------


def test_a_row_whose_printed_values_share_no_page_is_flagged_only():
    pages = {1: "Make Honda", 2: "VIN 1HGCV1F30KA000001", 3: "VIN 2T3P1RFV5MC000002 Make Toyota"}
    answer = {"vehicles": [
        {"unit_id": "veh_1", "vin": _fv("1HGCV1F30KA000001", 2), "make": _fv("Honda", 1)},
        {"unit_id": "veh_2", "vin": _fv("2T3P1RFV5MC000002", 3), "make": _fv("Toyota", 3)},
    ]}
    held = hold_to_printed(copy.deepcopy(answer), {}, pages)

    assert held.extraction == answer                                   # nothing taken out
    assert held.flags == ["vehicles[0]:row_not_colocated"]
    assert held.record["rows_not_colocated"] == 1


def test_a_row_is_judged_on_every_page_that_prints_its_values():
    """Each value cites its own page; both are printed together on page 3."""
    pages = {1: "Make Honda", 2: "VIN 1HGCV1F30KA000001", 3: "VIN 1HGCV1F30KA000001 Make Honda"}
    answer = {"vehicles": [{"unit_id": "veh_1", "vin": _fv("1HGCV1F30KA000001", 2), "make": _fv("Honda", 1)}]}
    held = hold_to_printed(answer, {}, pages)

    assert not any(flag.endswith(":row_not_colocated") for flag in held.flags)


def _bool_envelopes(node, out):
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict) and {"r", "v", "p"} <= set(props):
            out.append(node)
        for value in node.values():
            _bool_envelopes(value, out)
    elif isinstance(node, list):
        for value in node:
            _bool_envelopes(value, out)
    return out


def test_the_compact_schema_requires_r_for_a_yes_no_and_for_nothing_else():
    from common.canonical import _reads_yes_no, compact_schema
    from common.schemas import resolved_schema

    envelopes = _bool_envelopes(compact_schema(resolved_schema("policy", None, "personal_auto")), [])
    yes_no = [node for node in envelopes if _reads_yes_no(node["properties"]["v"])]
    others = [node for node in envelopes if not _reads_yes_no(node["properties"]["v"])]
    assert yes_no and others
    assert all((node.get("required") or [None])[0] == "r" for node in yes_no)
    assert not any("r" in (node.get("required") or ()) for node in others)


@pytest.mark.parametrize("envelope", [
    {"raw": "X", "parsed": True, "page_ref": [1]}, {"raw": "No", "parsed": False, "page_ref": [2]},
    {"raw": None, "parsed": None, "page_ref": []}, {"raw": "Yes", "parsed": None, "page_ref": [1]},
])
def test_every_training_target_writes_r_for_a_yes_no(envelope):
    """A yes/no never reads as its print, so training always writes its r: the
    required r refuses only answers no target holds."""
    from jsonschema import Draft202012Validator

    from common.canonical import compact_schema, to_compact_answer

    compact = to_compact_answer(envelope)
    assert "r" in compact and list(compact)[0] == "r"
    schema = compact_schema({"type": "object", "properties": {
        "raw": {"type": ["string", "null"]}, "parsed": {"type": ["boolean", "null"]},
        "page_ref": {"type": "array", "items": {"type": "integer"}}}, "required": ["raw", "parsed", "page_ref"]})
    assert schema["required"] == ["r", "v", "p"]
    assert not list(Draft202012Validator(schema).iter_errors(compact))
    assert list(Draft202012Validator(schema).iter_errors({"v": True, "p": [1]}))


# --------------------------------------------------------------------------
# Through extract: review routing and printed_texts (I6)
# --------------------------------------------------------------------------


def _accept_every_field(*, spans, **_kwargs):
    """A calibrator stand-in that accepts every field it can score: what is
    flagged then is what a review flag sent to review (without one, every field
    of an uncalibrated read is flagged, and nothing could be told apart)."""
    from calibration.apply_calibration import apply_calibration
    from calibration.logprob_confidence import field_confidences
    from tests.test_serving_pipeline import CALIBRATION

    return apply_calibration(field_confidences(spans), CALIBRATION, review_threshold=0.0)


def _serve(edit=None, *, printed_texts=None, monkeypatch=None):
    """The 3-page auto fixture served by replaying its training targets
    (tests.test_printed_values._serve), ``edit`` changing each window's answer.
    Given ``monkeypatch``, served with a calibrator that accepts every field
    (:func:`_accept_every_field`)."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from inference_core.model_runner import load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_printed_values import _texts
    from tests.test_schema2_serving import _compact_auto, _full_answer, _Replay, _replay_key
    from tests.test_serving_pipeline import CALIBRATION

    gold = _compact_auto()
    texts = _texts(gold, 3)
    images = [f"processed/default/policy/pa_1/page_{p}.png" for p in (1, 2, 3)]
    doc = SourceDocument(source_id="pa_1", doc_type="policy", golden_label=copy.deepcopy(gold),
                         ocr_pages=[texts[p] for p in (1, 2, 3)], image_paths=images, lob="personal_auto",
                         tenant_id="default")
    rows, _ = expand_document(doc, "train", modes=("ocr_plus_image",))
    answers = {}
    for row in rows:
        answer = _full_answer(row)
        answers[_replay_key(row)] = json.dumps(edit(answer) if edit else answer)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    request = ExtractionRequest(
        source_id="pa_1", image_paths=images, page_texts=texts, known_doc_type="policy",
        known_lob="personal_auto", source_file_name="auto_fixture.pdf", ocr_meta={"modality": "native_pdf"},
        printed_texts=printed_texts,
    )
    calibrated = {}
    if monkeypatch is not None:
        import serving.pipeline as pipeline

        monkeypatch.setattr(pipeline, "_feature_calibrated", _accept_every_field)
        calibrated = {"calibrators": object(), "thresholds": object()}
    return extract(request, load_model("base", client, backend_impl=_Replay(answers)),
                   StaticClassifier("policy"), CALIBRATION, strict_schema=False, **calibrated), texts


def _set(path_value):
    block, field, value = path_value

    def edit(answer):
        if isinstance(answer.get(block), dict) and isinstance(answer[block].get(field), dict):
            answer[block][field].update(value)
        return answer
    return edit


def test_extract_sends_a_value_under_the_wrong_label_to_review(monkeypatch):
    result, _texts_ = _serve(_set(("policy", "effective_date", {"raw": "10/01/2026", "parsed": "10/01/2026"})),
                             monkeypatch=monkeypatch)

    assert "policy.effective_date:role_mismatch" in result.review_flags
    assert result.extraction["policy"]["effective_date"]["raw"] == "10/01/2026"      # flagged, not changed
    assert result.extraction["policy"]["effective_date"]["flagged"] is True
    assert result.extraction["policy"]["expiration_date"]["flagged"] is False


def test_extract_takes_out_a_contradicted_parse():
    result, _texts_ = _serve(_set(("premium", "total", {"parsed": 16650.0})))

    assert "premium.total:parsed_mismatch" in result.review_flags
    assert result.extraction["premium"]["total"]["raw"] is None
    assert result.printed_check["parsed_mismatch"] == 1


def test_printed_texts_is_what_the_values_are_held_to():
    """The model reads one text and is held to another (the golden eval's
    noisy mode): the policy number is printed in the text it read, not in the
    text it is held to."""
    clean, texts = _serve()
    assert clean.extraction["policy"]["policy_number"]["raw"] == "PA-2026-0042"

    held_to = {page: text.replace("PA-2026-0042", "") for page, text in texts.items()}
    result, _texts_ = _serve(printed_texts=held_to)

    assert result.extraction["policy"]["policy_number"]["raw"] is None
    assert "policy.policy_number:not_printed_removed" in result.review_flags
    assert {entry["raw"] for entry in result.printed_check["not_printed"]} == {"PA-2026-0042"}


def test_extract_sends_a_value_cited_on_the_wrong_page_to_review(monkeypatch):
    """The conventions re-cite it to the page that prints it, as the labels
    are written; the doubt still goes to review."""
    result, _texts_ = _serve(_set(("premium", "total", {"page_ref": [2]})), monkeypatch=monkeypatch)

    assert result.extraction["premium"]["total"]["raw"] == "$1,665.00"
    assert result.extraction["premium"]["total"]["page_ref"] == [1]
    assert "premium.total:printed_off_cited_page" in result.review_flags
    assert result.extraction["premium"]["total"]["flagged"] is True
    assert result.extraction["policy"]["policy_number"]["raw"] == "PA-2026-0042"
    assert result.extraction["policy"]["policy_number"]["flagged"] is False
