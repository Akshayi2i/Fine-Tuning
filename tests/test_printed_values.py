"""Printed values only: a served policy holds no value its pages do not print
(serving.printed_values, decision 2026-10-24)."""

from __future__ import annotations

import copy
import json

import pytest

import serving.printed_values as printed_values
from common.canonical import values_view
from common.schemas import iter_validation_errors
from serving.printed_values import NO_PAGE_TEXT_FLAG, hold_to_printed


def _fv(raw, page=1, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [page]}


PAGES = {
    1: "DECLARATIONS\nGranite Mutual Insurance Company\nPolicy Number PA-2026-0042\n"
       "Policy Period 04/01/2026 to 10/01/2026  State NY\nAgency Code AG-7781",
    2: "SCHEDULE OF VEHICLES\nVeh 1 2019 Honda Accord VIN 1HGCV1F30KA000001\n"
       "Veh 2 2021 Toyota RAV4 VIN 2T3P1RFV5MC000002\nCollision Deductible $500",
}


def _answer():
    """A common-model answer as the merge leaves it: model form, ids numbered."""
    return {
        "carrier": {"name": _fv("Granite Mutual Insurance Company")},
        "policy": {"policy_number": _fv("PA-2026-0042"), "effective_date": _fv("04/01/2026"),
                   "rating_state": _fv("NY")},
        "vehicles": [
            {"unit_id": "veh_1", "vin": _fv("1HGCV1F30KA000001", 2), "year": _fv("2019", 2)},
            {"unit_id": "veh_2", "vin": _fv("5YJ3E1EA7KF000009", 2), "make": _fv("Tesla", 2)},
            {"unit_id": "veh_3", "vin": _fv("2T3P1RFV5MC000002", 2), "year": _fv("2021", 2)},
        ],
        "coverages": [
            {"coverage_id": "cov_1", "coverage_code": "COLL", "coverage_name": _fv("Collision", 2),
             "applies_to": ["veh_2"], "deductibles": [{"deductible_type": "flat", "amount": _fv("$500", 2)}]},
            {"coverage_id": "cov_2", "coverage_code": "COLL", "coverage_name": _fv("Collision", 2),
             "applies_to": ["veh_3"], "deductibles": [{"deductible_type": "flat", "amount": _fv("$2,000", 2)}]},
        ],
        "additional_fields": [
            {"label": "Account Number", "value": _fv("AC-0099")},
            {"label": "Agency Code", "value": _fv("AG-7781")},
        ],
    }


def _spans():
    return {path: f"span:{path}" for path in (
        "policy.policy_number", "vehicles[1].vin", "vehicles[2].vin", "vehicles[2].year",
        "coverages[1].deductibles[0].amount", "coverages[1].coverage_name",
        "additional_fields[0].value", "additional_fields[1].value")}


@pytest.fixture
def flag_mode(monkeypatch):
    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)


# --------------------------------------------------------------------------
# The mode
# --------------------------------------------------------------------------


def test_strict_is_the_configured_default():
    assert printed_values.configured_mode() == printed_values.STRICT


def test_an_unknown_mode_is_refused(monkeypatch):
    from common.config import ConfigError, serving_config

    monkeypatch.setitem(serving_config(), "printed_values", {"mode": "lenient"})
    with pytest.raises(ConfigError, match="lenient"):
        hold_to_printed(_answer(), {}, PAGES)
    monkeypatch.setitem(serving_config(), "printed_values", {})
    assert printed_values.configured_mode() == printed_values.STRICT        # unset: strict


# --------------------------------------------------------------------------
# What is checked, and what each outcome does
# --------------------------------------------------------------------------


def test_strict_takes_out_a_value_no_page_prints_and_keeps_its_key():
    answer = _answer()
    answer["policy"]["policy_number"] = _fv("PA-2026-9999")
    held = hold_to_printed(answer, _spans(), PAGES)

    assert "policy_number" in held.extraction["policy"] and held.extraction["policy"]["policy_number"] is None
    assert held.extraction["policy"]["effective_date"]["raw"] == "04/01/2026"
    assert "policy.policy_number" not in held.spans
    assert "policy.policy_number:not_printed_removed" in held.flags
    entry = next(e for e in held.record["not_printed"] if e["path"] == "policy.policy_number")
    assert entry == {"path": "policy.policy_number", "raw": "PA-2026-9999", "pages": [1]}
    assert held.record["mode"] == "strict" and held.record["removed"] is True
    assert set(held.record) == {"mode", "checked", "unchecked", "not_printed", "removed", "unread_page",
                                "parsed_mismatch", "off_cited_page", "rows_not_colocated", "form_refs"}


def test_flag_mode_keeps_the_value_and_flags_it(flag_mode):
    answer = _answer()
    held = hold_to_printed(answer, _spans(), PAGES)

    assert held.extraction["vehicles"][1]["vin"]["raw"] == "5YJ3E1EA7KF000009"
    assert len(held.extraction["vehicles"]) == 3 and len(held.extraction["additional_fields"]) == 2
    assert held.spans == _spans()
    assert "vehicles[1].vin:not_printed" in held.flags
    assert not any(flag.endswith("not_printed_removed") for flag in held.flags)
    assert held.record["removed"] is False
    assert {e["path"] for e in held.record["not_printed"]} == {
        "vehicles[1].vin", "vehicles[1].make", "coverages[1].deductibles[0].amount", "additional_fields[0].value"}


def test_an_extra_field_no_page_prints_leaves_its_list():
    held = hold_to_printed(_answer(), _spans(), PAGES)

    assert [entry["label"] for entry in held.extraction["additional_fields"]] == ["Agency Code"]
    assert held.spans["additional_fields[0].value"] == "span:additional_fields[1].value"
    assert "additional_fields[1].value" not in held.spans
    # Named by its place in the answer, marked: additional_fields[0] is the Agency Code now.
    assert "additional_fields[#0].value:not_printed_removed" in held.flags
    assert "additional_fields[0].value:not_printed_removed" not in held.flags
    assert held.record["not_printed"][-1]["path"] == "additional_fields[#0].value"


def test_a_row_with_nothing_printed_goes_and_the_rows_after_it_move_up_keeping_their_ids():
    held = hold_to_printed(_answer(), _spans(), PAGES)
    vehicles = held.extraction["vehicles"]

    # The invented vehicle is gone; the third is the second now, its id its own.
    assert [v["vin"]["raw"] for v in vehicles] == ["1HGCV1F30KA000001", "2T3P1RFV5MC000002"]
    assert [v["unit_id"] for v in vehicles] == ["veh_1", "veh_3"]
    # A link to it is dropped, and flagged; the link to the vehicle that moved is as it was.
    first, second = held.extraction["coverages"]
    assert "applies_to" not in first and second["applies_to"] == ["veh_3"]
    assert "coverages[0].applies_to:link_to_removed_row" in held.flags
    # Its spans go, and the next row's follow it up.
    assert held.spans["vehicles[1].vin"] == "span:vehicles[2].vin"
    assert held.spans["vehicles[1].year"] == "span:vehicles[2].year"
    assert "vehicles[2].vin" not in held.spans
    assert {"vehicles[#1].vin:not_printed_removed", "vehicles[#1].make:not_printed_removed"} <= set(held.flags)


def test_a_row_keeps_what_is_printed_and_loses_what_is_not():
    held = hold_to_printed(_answer(), _spans(), PAGES)
    coverage = held.extraction["coverages"][1]

    # Its name is printed: the coverage stays. Its invented deductible is a row
    # of its own, with nothing else printed in it: it goes.
    assert coverage["coverage_name"]["raw"] == "Collision" and coverage["deductibles"] == []
    assert "coverages[1].deductibles[0].amount" not in held.spans
    assert held.spans["coverages[1].coverage_name"] == "span:coverages[1].coverage_name"
    assert "coverages[1].deductibles[#0].amount:not_printed_removed" in held.flags


def test_unchecked_values_are_kept_and_counted():
    answer = {
        "policy": {"rating_state": _fv("NY"), "policy_number": _fv("PA-2026-0042"),
                   "subject_to_audit": _fv(True), "auditable": _fv("No")},
        # A page it cites has no text: the scan may print it where OCR read nothing.
        "premium": {"total": _fv("$9,999.00", 3)},
        # A row whose only value found nowhere sits beside one too short to look for.
        "vehicles": [{"unit_id": "veh_1", "vehicle_number": _fv("7", 2), "vin": _fv("JH4KA0000ZZ000000", 2)}],
    }
    held = hold_to_printed(answer, {}, {**PAGES, 3: ""})

    assert held.extraction["policy"]["rating_state"]["raw"] == "NY"     # two characters: found as a token
    assert held.extraction["policy"]["subject_to_audit"]["raw"] is True
    assert held.extraction["policy"]["auditable"]["raw"] == "No"
    assert held.extraction["premium"]["total"]["raw"] == "$9,999.00"
    assert held.extraction["vehicles"][0]["vehicle_number"]["raw"] == "7"
    assert held.extraction["vehicles"][0]["vin"] is None
    assert held.record["unchecked"] == 4 and held.record["checked"] == 3 and held.record["unread_page"] == 1
    # Kept on an unread page, never silently.
    assert "premium.total:unverified_unread_page" in held.flags


def test_system_fields_are_not_checked():
    answer = {
        "document": {"page_count": _fv("99"), "title": _fv("Personal Auto Policy Declarations Copy")},
        "policy": {"is_package": _fv("Package Policy Form 1234")},
    }
    held = hold_to_printed(answer, {}, PAGES)

    assert held.extraction["document"]["page_count"]["raw"] == "99"
    assert held.extraction["policy"]["is_package"]["raw"] == "Package Policy Form 1234"
    assert held.extraction["document"]["title"] is None                 # the one value checked
    assert held.record["checked"] == 1 and held.record["unchecked"] == 0


def test_a_parsed_value_behind_a_blank_raw_is_looked_for_too():
    """No parsed value is served unchecked: with raw empty or a placeholder, the
    forms parsed is printed in are looked for."""
    answer = {
        "named_insured": {"doing_business_as": {"raw": None, "parsed": "JR Hauling", "page_ref": [1]},
                          "primary_name": {"raw": "", "parsed": "JANE", "page_ref": [1]}},
        "carrier": {"name": {"raw": "-", "parsed": "Granite Mutual Insurance Company", "page_ref": [1]}},
        "policy": {"policy_number": {"raw": "", "parsed": "PA-2026-9999", "page_ref": [1]},
                   "effective_date": {"raw": "", "parsed": "2026-04-01", "page_ref": [1]},
                   "expiration_date": {"raw": "--", "parsed": "2027-04-01", "page_ref": [1]},
                   "subject_to_audit": {"raw": "", "parsed": True, "page_ref": [1]},
                   "auditable": {"raw": "-", "parsed": None, "page_ref": [1]}},
    }
    held = hold_to_printed(answer, {}, PAGES, lob="personal_auto")

    assert held.extraction["named_insured"] == {"doing_business_as": None, "primary_name": None}
    assert held.extraction["policy"]["policy_number"] is None
    assert held.extraction["policy"]["expiration_date"] is None          # no page prints 04/01/2027
    # Printed in another form of the same value: kept.
    assert held.extraction["carrier"]["name"]["parsed"] == "Granite Mutual Insurance Company"
    assert held.extraction["policy"]["effective_date"]["parsed"] == "2026-04-01"   # printed 04/01/2026
    # A reading, and a dash with nothing behind it: nothing to look for that a page settles.
    assert held.extraction["policy"]["subject_to_audit"]["parsed"] is True
    assert held.extraction["policy"]["auditable"]["raw"] == "-"
    listed = {entry["path"]: entry["raw"] for entry in held.record["not_printed"]}
    assert listed == {"named_insured.doing_business_as": "JR Hauling", "named_insured.primary_name": "JANE",
                      "policy.policy_number": "PA-2026-9999", "policy.expiration_date": "04/01/2027"}
    assert held.record["checked"] == 6 and held.record["unchecked"] == 1


def test_a_short_value_no_page_prints_is_taken_out():
    answer = {"policy": {"rating_state": _fv("NJ"), "state": _fv("NY")},
              "coverages": [{"coverage_id": "cov_1", "coverage_code": "COLL",
                             "deductibles": [{"deductible_type": "flat", "amount": _fv("$50", 2)},
                                             {"deductible_type": "flat", "amount": _fv("$500", 2)}]}]}
    held = hold_to_printed(answer, {}, PAGES)

    assert held.extraction["policy"] == {"rating_state": None, "state": _fv("NY")}
    assert [d["amount"]["raw"] for d in held.extraction["coverages"][0]["deductibles"]] == ["$500"]
    assert {"policy.rating_state:not_printed_removed",
            "coverages[0].deductibles[#0].amount:not_printed_removed"} <= set(held.flags)


# --------------------------------------------------------------------------
# A blank page among those a value cites (the rule in the module docstring)
# --------------------------------------------------------------------------


BLANK_3 = {**PAGES, 3: ""}


@pytest.mark.parametrize("cited", [[1], [1, 2], [9], []])
def test_printed_nowhere_and_every_cited_page_read_is_taken_out(cited):
    """Rule 3: on no page with text, and every page it cites has text (or is no
    page of the document, or it cites none): printed nowhere."""
    answer = {"carrier": {"name": {"raw": "Acme Phantom Insurance Co", "parsed": "x", "page_ref": cited}}}
    held = hold_to_printed(answer, {}, BLANK_3)

    assert held.extraction["carrier"]["name"] is None
    assert held.flags == ["carrier.name:not_printed_removed"]
    assert held.record["checked"] == 1 and held.record["unread_page"] == 0


@pytest.mark.parametrize("cited", [[3], [1, 3], [2, 3]])
def test_printed_on_no_page_with_text_while_a_cited_page_has_none_is_kept_and_flagged(cited):
    """Rule 4: a page it cites could not be read, and may print it: kept,
    flagged, and counted unchecked - in neither the checked count nor the
    not_printed list, the invented rate's two terms."""
    answer = {"carrier": {"name": {"raw": "Acme Phantom Insurance Co", "parsed": "x", "page_ref": cited}}}
    held = hold_to_printed(answer, {}, BLANK_3)

    assert held.extraction["carrier"]["name"]["raw"] == "Acme Phantom Insurance Co"
    assert held.flags == ["carrier.name:unverified_unread_page"]
    assert held.record == {"mode": "strict", "checked": 0, "unchecked": 1, "not_printed": [], "removed": True,
                           "unread_page": 1, "parsed_mismatch": 0, "off_cited_page": 0, "rows_not_colocated": 0,
                           "form_refs": {"checked": 0, "not_printed": []}}


def test_printed_on_another_page_than_a_blank_cited_one_is_printed():
    answer = {"carrier": {"name": _fv("Granite Mutual Insurance Company", 3)}}
    held = hold_to_printed(answer, {}, BLANK_3)
    assert held.flags == [] and held.record["checked"] == 1 and held.record["unread_page"] == 0


def test_flag_mode_flags_an_unread_page_too(flag_mode):
    answer = {"carrier": {"name": {"raw": "Acme Phantom Insurance Co", "parsed": "x", "page_ref": [1, 3]}}}
    held = hold_to_printed(answer, {}, BLANK_3)
    assert held.flags == ["carrier.name:unverified_unread_page"] and held.record["not_printed"] == []


# --------------------------------------------------------------------------
# A flag never names a served path that holds another value
# --------------------------------------------------------------------------


def _served_at(tree, path):
    try:
        return printed_values._at(tree, path)
    except (KeyError, IndexError, TypeError):
        return KeyError


def test_no_flag_names_a_served_path_holding_another_value():
    """M9: the rows after one taken out move up; a flag on a value taken out
    with its row names it by its place in the answer, marked, never by the
    served path the next row now has."""
    answer = _answer()
    # The first vehicle invented too: the third is served as vehicles[0].
    answer["vehicles"][0] = {"unit_id": "veh_1", "vin": _fv("ZZZZZ000000000000", 2), "make": _fv("Rivian", 2)}
    held = hold_to_printed(answer, {}, PAGES)
    served = held.extraction

    assert [v["unit_id"] for v in served["vehicles"]] == ["veh_3"]           # its id its own
    assert sorted(flag for flag in held.flags if flag.startswith("vehicles")) == [
        "vehicles[#0].make:not_printed_removed", "vehicles[#0].vin:not_printed_removed",
        "vehicles[#1].make:not_printed_removed", "vehicles[#1].vin:not_printed_removed"]
    assert {entry["path"] for entry in held.record["not_printed"] if entry["path"].startswith("vehicles")} == {
        "vehicles[#0].vin", "vehicles[#0].make", "vehicles[#1].vin", "vehicles[#1].make"}
    # A flag on a plain path names a value served there: a null where one was taken out.
    for flag in held.flags:
        path, _, kind = flag.rpartition(":")
        if kind == "not_printed_removed" and "[#" not in path:
            assert _served_at(served, path) is None, flag
    # The link to a vehicle taken out goes, flagged; the one to the vehicle left is as merged.
    assert "applies_to" not in served["coverages"][0] and served["coverages"][1]["applies_to"] == ["veh_3"]
    assert "coverages[0].applies_to:link_to_removed_row" in held.flags


@pytest.mark.parametrize("texts", [None, {}, {1: "", 2: "   \n"}])
def test_no_page_text_changes_nothing(texts):
    answer = _answer()
    held = hold_to_printed(answer, _spans(), texts)

    assert held.extraction == _answer() and held.spans == _spans()
    assert held.record is None and held.flags == [NO_PAGE_TEXT_FLAG]


def test_a_self_contained_answer_leaves_a_value_out_where_no_null_is_valid():
    """A self-contained line's schema allows no null value: the key is left out,
    which the every-key fill serves as the same null."""
    answer = {"carrier": {"company_name": _fv("Granite Mutual Insurance Company")},
              "policy": {"policy_number": _fv("WC-0000000-00"), "effective_date": _fv("04/01/2026")}}
    held = hold_to_printed(answer, {}, PAGES)

    assert held.extraction["policy"] == {"effective_date": _fv("04/01/2026")}
    assert "policy.policy_number:not_printed_removed" in held.flags


def test_the_pages_are_prepared_once_for_the_document(monkeypatch):
    from common import grounding

    built = []
    original = grounding.TolerantPage.__init__

    def counting(self, text):
        built.append(text)
        original(self, text)

    monkeypatch.setattr(grounding.TolerantPage, "__init__", counting)
    hold_to_printed(_answer(), _spans(), PAGES)
    assert len(built) == len(PAGES)


# --------------------------------------------------------------------------
# The flags raised before the check
# --------------------------------------------------------------------------


def test_the_flags_raised_before_follow_their_values():
    answer = _answer()
    answer["policy"]["policy_number"] = _fv("PA-2026-9999")
    held = hold_to_printed(answer, _spans(), PAGES)
    earlier = [
        "policy.policy_number:not_printed",              # the conventions': replaced
        "vehicles[1].vin:merge_conflict",                # on a row taken out: gone with it
        "vehicles[2].year:merge_conflict",               # on a row that moved up: moved too
        "additional_fields:not_printed_dropped",         # names no value
        "window_failed:decl:p3",
    ]
    joined = held.joined(earlier)

    assert "policy.policy_number:not_printed" not in joined
    assert "policy.policy_number:not_printed_removed" in joined
    assert not any(flag.startswith("vehicles[2]") for flag in joined)
    assert "vehicles[1].vin:merge_conflict" not in joined
    assert "vehicles[1].year:merge_conflict" in joined
    assert {"additional_fields:not_printed_dropped", "window_failed:decl:p3"} <= set(joined)


def test_flag_mode_does_not_repeat_the_conventions_flag(flag_mode):
    held = hold_to_printed(_answer(), {}, PAGES)
    joined = held.joined(["vehicles[1].vin:not_printed"])
    assert joined.count("vehicles[1].vin:not_printed") == 1


# --------------------------------------------------------------------------
# Served: extract, and fields first
# --------------------------------------------------------------------------


INVENTED_NUMBER = "PA-2026-9999"
INVENTED_VIN = "5YJ3E1EA7KF000009"


def _texts(gold, pages):
    """Each page's text printing every value its label cites there."""
    lines = {page: [] for page in range(1, pages + 1)}

    def walk(node, name):
        if isinstance(node, dict):
            if "raw" in node and "page_ref" in node:
                for page in node["page_ref"]:
                    lines[page].append(f"{name.replace('_', ' ').title()}: {node['raw']}")
                return
            for key, value in node.items():
                walk(value, key)
        elif isinstance(node, list):
            for value in node:
                walk(value, name)

    walk(gold, "")
    return {page: "\n".join(text) for page, text in lines.items()}


def _invent(answer):
    """The model's answer, with a policy number and a whole vehicle no page prints."""
    doc = json.loads(answer.replace("PA-2026-0042", INVENTED_NUMBER))
    if doc.get("vehicles"):
        doc["vehicles"].append({"unit_id": f"veh_{len(doc['vehicles']) + 1}",
                                "vin": _fv(INVENTED_VIN, 2), "make": _fv("Tesla", 2)})
    return json.dumps(doc)


def _serve(mode, modality="native_pdf", *, page_texts=True):
    """Serve the 3-page auto fixture by replaying its training targets, as
    ``tests.test_schema2_serving`` does, with page text that prints its values."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from inference_core.model_runner import load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_schema2_serving import _compact_auto, _full_answer, _Replay, _replay_key
    from tests.test_serving_pipeline import CALIBRATION

    gold = _compact_auto()
    texts = _texts(gold, 3)
    images = [f"processed/default/policy/pa_1/page_{p}.png" for p in (1, 2, 3)]
    doc = SourceDocument(source_id="pa_1", doc_type="policy", golden_label=copy.deepcopy(gold),
                         ocr_pages=[texts[p] for p in (1, 2, 3)], image_paths=images, lob="personal_auto",
                         tenant_id="default")
    rows, _ = expand_document(doc, "train", modes=(mode,))
    answers = {_replay_key(r): _invent(json.dumps(_full_answer(r))) for r in rows}
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    request = ExtractionRequest(
        source_id="pa_1", image_paths=images, page_texts=texts if page_texts else {}, modality_mode=mode,
        known_doc_type="policy", known_lob="personal_auto", source_file_name="auto_fixture.pdf",
        ocr_meta={"modality": modality},
    )
    result = extract(request, load_model("base", client, backend_impl=_Replay(answers)),
                     StaticClassifier("policy"), CALIBRATION, strict_schema=mode != "image_only")
    return result, texts


def _vins(result):
    return [values_view(v["vin"]) for v in result.extraction["vehicles"]]


@pytest.mark.parametrize("modality", ["native_pdf", "scanned_pdf"])
def test_extract_takes_out_what_the_model_invented(modality):
    result, _texts_ = _serve("ocr_plus_image", modality)

    listed = {entry["raw"]: entry for entry in result.printed_check["not_printed"]}
    assert {INVENTED_NUMBER, INVENTED_VIN, "Tesla"} == set(listed)
    assert listed[INVENTED_NUMBER]["path"] == "policy.policy_number" and listed[INVENTED_NUMBER]["pages"] == [1]
    assert result.printed_check["removed"] is True and result.printed_check["checked"] > 20

    # Served as the every-key fill's null; the invented vehicle gone, ids in order.
    assert result.extraction["policy"]["policy_number"]["raw"] is None
    assert _vins(result) == ["1HGCV1F30KA000001", "2T3P1RFV5MC000002"]
    assert [v["unit_id"] for v in result.extraction["vehicles"]] == ["veh_1", "veh_2"]
    assert "policy.policy_number" not in result.fields
    assert result.list_fields["vehicles"]["extracted_rows"] == 2
    # One flag for a value taken out: the conventions' not_printed is replaced.
    assert "policy.policy_number:not_printed_removed" in result.review_flags
    assert "policy.policy_number:not_printed" not in result.review_flags
    assert result.schema_valid
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == []


def test_fields_first_holds_the_fields_to_the_text_when_it_arrives():
    from serving.pipeline import NO_FULL_TEXT_FLAG, attach_full_text

    result, texts = _serve("image_only", page_texts=False)
    # Read from the images alone: nothing to check yet, and it says so.
    assert result.printed_check is None and NO_PAGE_TEXT_FLAG in result.review_flags
    assert result.extraction["policy"]["policy_number"]["raw"] == INVENTED_NUMBER
    assert INVENTED_VIN in _vins(result) and "policy.policy_number" in result.fields

    attach_full_text(result, texts, lob="personal_auto")

    assert NO_PAGE_TEXT_FLAG not in result.review_flags and NO_FULL_TEXT_FLAG not in result.review_flags
    assert {entry["raw"] for entry in result.printed_check["not_printed"]} == {INVENTED_NUMBER, INVENTED_VIN,
                                                                                "Tesla"}
    assert result.extraction["policy"]["policy_number"]["raw"] is None
    assert _vins(result) == ["1HGCV1F30KA000001", "2T3P1RFV5MC000002"]
    assert [v["unit_id"] for v in result.extraction["vehicles"]] == ["veh_1", "veh_2"]
    assert "policy.policy_number" not in result.fields
    rows = result.list_fields["vehicles"]
    assert rows["extracted_rows"] == len(rows["rows"]) == 2
    assert [row["vin"] for row in rows["rows"]] == _vins(result)
    assert "policy.policy_number:not_printed_removed" in result.review_flags
    assert not any(flag.startswith("vehicles[2]") and "removed" not in flag for flag in result.review_flags)
    # Every key is still there, the answer still valid, the full-text tier joined.
    assert result.schema_valid and result.extraction["text_sections"]
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == []

    # Checked once: the text arriving again changes nothing.
    checked = copy.deepcopy(result.printed_check)
    attach_full_text(result, texts, lob="personal_auto")
    assert result.printed_check == checked


def test_fields_first_in_flag_mode_keeps_and_flags(flag_mode):
    from serving.pipeline import attach_full_text

    result, texts = _serve("image_only", page_texts=False)
    attach_full_text(result, texts, lob="personal_auto")

    assert result.extraction["policy"]["policy_number"]["raw"] == INVENTED_NUMBER
    assert INVENTED_VIN in _vins(result)
    assert "policy.policy_number:not_printed" in result.review_flags
    assert result.printed_check["removed"] is False and NO_PAGE_TEXT_FLAG not in result.review_flags


def test_a_served_result_without_page_text_is_left_unchecked():
    result, _texts_ = _serve("image_only", page_texts=False)
    served = copy.deepcopy(result.extraction)
    printed_values.hold_result_to_printed(result, {1: "", 2: " "})

    assert result.printed_check is None and result.extraction == served
    assert NO_PAGE_TEXT_FLAG in result.review_flags


def test_overall_confidence_is_averaged_as_extract_averages_it():
    def scored(score, source="vlm", raw="x"):
        return {"raw": raw, "parsed": raw, "confidence": {"score": score, "source": source},
                "page_ref": [1], "flagged": False}

    served = {
        "policy": {"policy_number": scored(0.9), "effective_date": scored(0.5)},
        "document": {"page_count": scored(1.0, "deterministic")},     # the pipeline's, not read
        "carrier": {"name": scored(0.0, "structural", None)},         # the every-key fill
        "vehicles": [{"symbols": [scored(0.7), scored(0.7)]}],         # one field, scored once
    }
    flags = ["policy.effective_date:no_confidence"]                   # no confidence: not averaged
    assert printed_values._overall_confidence(served, flags) == pytest.approx((0.9 + 0.7) / 2)


def test_told_the_line_its_schema_decides_how_a_value_is_taken_out():
    """A self-contained line allows no null value, a common-model line does; told
    the line, nothing is read off the answer."""
    from common.schemas import is_common_model
    from serving.printed_values import _nullable

    assert not is_common_model("policy", None, "equipment_breakdown") and is_common_model("policy", None, "homeowners")
    assert _nullable({}, "equipment_breakdown") is False
    assert _nullable({}, "homeowners") is True
    assert _nullable({}, None) is True                   # no line: the fallback, on the common model
