"""serving/fields_first.py: the fields held to the pages they cite before they are served."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from common.canonical import values_view
from serving.fields_first import PENDING_FLAG, _pending, cited_pages, read_fields, verify_fields

PDF = b"%PDF-1.7 stand-in"
HONDA = "1HGCV1F30KA000001"
HONDA_LINES = ("Year: 2019", "Make: Honda", "Model: Accord", f"Vin: {HONDA}")


def test_the_pages_the_fields_cite_are_every_page_ref_within_the_document():
    extraction = {
        "policy": {"policy_number": {"raw": "PA-1", "page_ref": [1]}, "insured": None},
        "vehicles": [{"vin": {"raw": "1HGCM", "page_ref": [3, 4]}}, {"year": {"raw": "2019", "page_ref": [3]}}],
        "additional_fields": [{"label": "Agent", "raw": "J. Smith", "page_ref": [9, 120]}],
        "forms_and_endorsements": [{"form_number": {"raw": "PP 00 01", "page_ref": "7"}}],   # not a list: ignored
    }
    assert cited_pages(extraction, page_count=10) == [1, 3, 4, 9]
    assert cited_pages({}, page_count=10) == []


@pytest.mark.parametrize(("flag", "becomes"), [
    ("policy.policy_number:not_printed_removed", f"policy.policy_number:{PENDING_FLAG}"),
    ("vehicles[0].vin:not_printed", f"vehicles[0].vin:{PENDING_FLAG}"),
    ("vehicles[#0].vin:not_printed_removed", f"vehicles[#0].vin:{PENDING_FLAG}"),
    ("coverages[0].applies_to:link_to_removed_row", f"coverages[0].applies_to:{PENDING_FLAG}"),
    ("carrier.name:unverified_unread_page", "carrier.name:unverified_unread_page"),
    ("policy.policy_number:low_confidence", "policy.policy_number:low_confidence"),
    ("text_sections:no_page_text", "text_sections:no_page_text"),
])
def test_not_found_on_the_cited_pages_is_pending_not_printed_nowhere(flag, becomes):
    assert _pending(flag) == becomes


# --------------------------------------------------------------------------
# Nothing is served unchecked (L1)
# --------------------------------------------------------------------------


class _Reader:
    """A served MinerU: renders the pages, reads each page's given text."""

    def __init__(self, texts):
        self.texts, self.asked = dict(texts), []

    def read(self, pdf, pages_dir, *, name, with_text, publish_pages=None):
        images = [f"processed/default/policy/pa_1/page_{p}.png" for p in sorted(self.texts)]
        return {"images": images, "texts": {}, "links": None, "timings_s": {"render": 0.1, "links": None}}

    def ocr_text(self, pdf, *, name, pages=None, page_count=None, gate=None):
        self.asked.append(pages)
        texts = {page: self.texts.get(page, "") for page in pages or sorted(self.texts)}
        return {"texts": texts, "info": {"version": "x"}, "seconds": 0.1}


def _image_only_answer():
    return SimpleNamespace(
        extraction={"policy": {"policy_number": {"raw": "PA-2026-9999", "parsed": "PA-2026-9999", "page_ref": [1]}},
                    "carrier": {"name": {"raw": "Acme Phantom Insurance Co", "parsed": "x", "page_ref": [1]}}},
        review_flags=["text_sections:no_page_text"], printed_check=None, doc_type="policy", mode="image_only")


def test_fields_whose_cited_pages_have_no_text_are_not_verified():
    """MinerU read the cited page blank: nothing was checked, so nothing is verified."""
    checked = verify_fields(PDF, _image_only_answer(), name="scan.pdf", lob="personal_auto",
                            reader=_Reader({1: "", 2: "Schedule", 3: "Forms"}), page_count=3)
    assert checked["verified"] is False and checked["pages"] == [1]
    assert checked["served"].printed_check is None


def test_fields_whose_cited_pages_have_no_text_are_not_served_before_the_full_text(monkeypatch, tmp_path):
    import serving.pipeline as pipeline

    monkeypatch.setattr(pipeline, "extract", lambda request, *args, **kwargs: _image_only_answer())
    reader = _Reader({1: "", 2: "Schedule", 3: "Forms"})
    fields = read_fields(PDF, name="scan.pdf", lob="personal_auto", source_id="x", model=None, reader=reader,
                         pages_dir=tmp_path, classifier=None, calibration=None)
    assert fields["served"] is None and fields["verified"] is False
    # The page read stays read: the full text does not ask for it again.
    assert fields["cited_texts"] == {1: ""} and reader.asked == [[1]]

    # A cited page with text: checked, and served.
    reader = _Reader({1: "Policy Number PA-2026-9999\nAcme Phantom Insurance Co", 2: "Schedule", 3: "Forms"})
    fields = read_fields(PDF, name="scan.pdf", lob="personal_auto", source_id="x", model=None, reader=reader,
                         pages_dir=tmp_path, classifier=None, calibration=None)
    assert fields["verified"] is True and fields["served"] is not None
    assert fields["served"].printed_check["checked"] == 2


# --------------------------------------------------------------------------
# A served row's id and links: the same at the gate, at the end, and on the
# standard path (M8, M10)
# --------------------------------------------------------------------------


def _model(mode, edit):
    """The 3-page auto fixture's model, replaying its training targets for
    ``mode``, each changed by ``edit`` (the parsed answer, in place)."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from inference_core.model_runner import load_model
    from tests.test_printed_values import _texts
    from tests.test_schema2_serving import _compact_auto, _Replay

    gold = _compact_auto()
    texts = _texts(gold, 3)
    doc = SourceDocument(source_id="pa_1", doc_type="policy", golden_label=copy.deepcopy(gold),
                         ocr_pages=[texts[p] for p in (1, 2, 3)],
                         image_paths=[f"processed/default/policy/pa_1/page_{p}.png" for p in (1, 2, 3)],
                         lob="personal_auto", tenant_id="default")
    rows, _ = expand_document(doc, "train", modes=(mode,))
    answers = {}
    for row in rows:
        answer = json.loads(row["messages"][-1]["content"])
        edit(answer)
        answers[(row["messages"][0]["content"], tuple(row["window_pages"]))] = json.dumps(answer)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    return load_model("base", client, backend_impl=_Replay(answers)), gold


def _cite(node, page):
    if isinstance(node, dict):
        if isinstance(node.get("page_ref"), list):
            node["page_ref"] = [page]
        for value in node.values():
            _cite(value, page)
    elif isinstance(node, list):
        for value in node:
            _cite(value, page)


def _honda_cites(page):
    """The first vehicle with no vehicle number (nothing too short keeps its
    row), citing ``page``; the drivers, forms and extra fields page 2."""
    def edit(answer):
        for vehicle in answer.get("vehicles") or []:
            if (vehicle.get("vin") or {}).get("raw") == HONDA:
                vehicle.pop("vehicle_number", None)
                _cite(vehicle, page)
        for name in ("drivers", "forms_and_endorsements", "additional_fields"):
            _cite(answer.get(name) or [], 2)
    return edit


def _extract(model, mode, texts):
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_serving_pipeline import CALIBRATION

    request = ExtractionRequest(
        source_id="pa_1", image_paths=[f"processed/default/policy/pa_1/page_{p}.png" for p in (1, 2, 3)],
        page_texts=texts if mode != "image_only" else {}, modality_mode=mode, known_doc_type="policy",
        known_lob="personal_auto", source_file_name="scan.pdf", ocr_meta={"modality": "scanned_pdf"})
    return extract(request, model, StaticClassifier("policy"), CALIBRATION, strict_schema=False)


def _rows(result):
    """Each served row's id and links, by table; and list_fields' the same way."""
    view = values_view(result.extraction)

    def ids(rows):
        return [(row.get("unit_id") or row.get("coverage_id"), sorted(row.get("applies_to") or []),
                 row.get("vin") or row.get("coverage_code")) for row in rows]

    served = {name: ids(view[name]) for name in ("vehicles", "coverages", "interested_parties")}
    listed = {name: ids(result.list_fields[name]["rows"]) for name in served}
    return served, listed


def test_a_served_rows_id_and_links_are_the_same_when_the_fields_come_and_at_the_end():
    """The first vehicle is printed only on a page no value cites: held back at
    the gate, back at the end. The vehicle after it is veh_2 at both."""
    from serving.pipeline import attach_full_text
    from tests.test_printed_values import _texts
    from tests.test_schema2_serving import _compact_auto

    texts = _texts(_compact_auto(), 3)
    texts = {1: texts[1], 2: "\n".join([line for line in texts[2].splitlines() if line not in HONDA_LINES]
                                        + texts[3].splitlines()), 3: "\n".join(HONDA_LINES)}
    model, _gold = _model("image_only", _honda_cites(1))
    result = _extract(model, "image_only", texts)
    reader = _Reader(texts)
    checked = verify_fields(PDF, result, name="scan.pdf", lob="personal_auto", reader=reader, page_count=3)
    assert checked["verified"] and reader.asked == [[1, 2]]
    gate, _ = _rows(checked["served"])
    attach_full_text(result, texts, lob="personal_auto")
    end, listed = _rows(result)

    assert [vin for _id, _links, vin in gate["vehicles"]] == ["2T3P1RFV5MC000002"]       # the Honda held back
    assert [vin for _id, _links, vin in end["vehicles"]] == [HONDA, "2T3P1RFV5MC000002"]  # and back
    for table, rows in gate.items():
        for row_id, links, key in rows:
            at_end = next(row for row in end[table] if row[0] == row_id)
            if table == "coverages" and at_end[1] == ["veh_1"]:
                assert links == []                       # a link to the vehicle held back, held back with it
            else:
                assert (row_id, links, key) == at_end, (table, row_id)
    assert {"coverages[0].applies_to:pending_full_text", "vehicles[#0].vin:pending_full_text"} <= set(
        checked["served"].review_flags)
    assert not any(flag.startswith("vehicles[0].") and "pending" in flag for flag in checked["served"].review_flags)
    assert listed == end


def test_the_final_result_holds_ids_links_and_rows_as_the_standard_path_serves_them():
    """The first vehicle is printed nowhere: taken out at the end of fields
    first and by the standard path alike - the rows left keep their ids, a link
    to it goes, and list_fields follows the served JSON."""
    from serving.pipeline import attach_full_text
    from tests.test_printed_values import _texts
    from tests.test_schema2_serving import _compact_auto

    texts = _texts(_compact_auto(), 3)
    texts[2] = "\n".join(line for line in texts[2].splitlines() if line not in HONDA_LINES)

    model, _gold = _model("image_only", _honda_cites(2))
    fields_first = _extract(model, "image_only", texts)
    attach_full_text(fields_first, texts, lob="personal_auto")
    model, _gold = _model("ocr_plus_image", _honda_cites(2))
    standard = _extract(model, "ocr_plus_image", texts)

    served, listed = _rows(fields_first)
    assert served == _rows(standard)[0] == listed == _rows(standard)[1]
    assert [row[0] for row in served["vehicles"]] == ["veh_2"]                          # not renumbered
    assert [row[1] for row in served["coverages"]] == [[], [], ["veh_2"], ["veh_2"]]
    assert served["interested_parties"][0][1] == ["veh_2"]
    for result in (fields_first, standard):
        flags = set(result.review_flags)
        assert {"vehicles[#0].vin:not_printed_removed", "coverages[0].applies_to:link_to_removed_row"} <= flags
        assert not any(flag.startswith("vehicles[0].") and "printed" in flag for flag in flags)
        assert [entry["path"] for entry in result.printed_check["not_printed"]
                if entry["path"].startswith("vehicles")] == [
            "vehicles[#0].year", "vehicles[#0].make", "vehicles[#0].model", "vehicles[#0].vin"]


def test_an_image_only_read_is_looked_over_for_missed_fields_when_its_text_comes(monkeypatch):
    import serving.pipeline as pipeline
    from serving.pipeline import attach_full_text
    from tests.test_full_text import _compact_auto, _serve_replay

    result, _b, _a = _serve_replay(_compact_auto(), 3, strict=False)
    result.mode, result.printed_check = "image_only", {}               # read from images; check aside
    seen = []
    monkeypatch.setattr(pipeline, "_likely_missed",
                        lambda extraction, texts, lob: seen.append(sorted(texts)) or ["policy.insured:likely_missed"])
    attach_full_text(result, {1: "Declarations", 2: "Schedule page 2", 3: "Schedule page 3"}, lob="personal_auto")
    assert seen == [[1, 2, 3]] and "policy.insured:likely_missed" in result.review_flags
    # A read that had its text all along was looked over by extract already.
    result.mode, seen[:] = "ocr_plus_image", []
    attach_full_text(result, {1: "Declarations"}, lob="personal_auto")
    assert seen == []
