"""A rule-computed label value is trained on only if the page prints it
(data_pipeline.dataset_builder.label_verification).

Two thirds of the synthetic labels' values were added from the source's reviewed
gold by rules, and checked against the page only as an overall rate. One the
synthetic page does not print would teach the model to write what it cannot read.
"""

from __future__ import annotations

from data_pipeline.dataset_builder.label_verification import VerificationReport, verified_label


def _env(value, *pages):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


PAGES = ["Declarations. Carrier: Northfield Mutual. Policy HO-778812.",
         "Agent: Lakeside Agency, 12 Main Street. Premium 1,250.00",
         "Forms: HO 00 03. Agent phone on file."]


def _label(**added):
    label = {
        "policy": {"policy_number": _env("HO-778812", 1)},      # placed by the generator
        "carrier": {}, "producer": {}, "premium": {},
        "forms_and_endorsements": [{"form_number": _env("HO 00 03", 3)}],
    }
    paths = []
    for path, value in added.items():
        section, name = path.split(".")
        label[section][name] = value
        paths.append(path)
    label["fideon:filled"] = {"paths": paths}
    return label


def test_an_added_value_printed_on_the_page_it_cites_is_kept():
    label = _label(**{"carrier.company_name": _env("Northfield Mutual", 1)})
    report = VerificationReport()
    assert verified_label(label, PAGES, report) is label
    assert (report.checked, report.kept, report.dropped) == (1, 1, [])


def test_an_added_value_the_page_does_not_print_is_left_out():
    label = _label(**{"carrier.company_name": _env("Granite State Insurance", 1)})
    report = VerificationReport()
    out = verified_label(label, PAGES, report)
    assert "company_name" not in out["carrier"]
    assert report.dropped == ["carrier.company_name"]
    assert "company_name" in label["carrier"], "the stored label is never changed"


def test_an_added_value_on_one_other_page_cites_that_page():
    label = _label(**{"producer.agency_name": _env("Lakeside Agency", 1)})
    report = VerificationReport()
    out = verified_label(label, PAGES, report)
    assert out["producer"]["agency_name"]["page_ref"] == [2]
    assert report.repaged == ["producer.agency_name"]


def test_an_added_value_only_on_several_other_pages_is_left_out():
    """It is on the document, but nothing says which printing is this field's."""
    label = _label(**{"producer.contact_role": _env("Agent", 1)})
    out = verified_label(label, PAGES)
    assert "contact_role" not in out["producer"]


def test_a_value_nobody_computed_is_never_touched():
    """OCR may have missed it; it can still be on the image, the model's to read."""
    label = _label()
    label["policy"]["policy_number"] = _env("ZZ-000000", 1)
    assert verified_label(label, PAGES) is label


def test_a_value_too_short_to_check_is_kept():
    label = _label(**{"premium.installment_count": _env("4", 2)})
    report = VerificationReport()
    assert verified_label(label, PAGES, report) is label and report.too_short == 1


def test_without_ocr_text_nothing_is_verified_and_it_is_counted():
    label = _label(**{"carrier.company_name": _env("Granite State Insurance", 1)})
    report = VerificationReport()
    assert verified_label(label, ["", ""], report) is label
    assert verified_label(label, None, report) is label
    assert report.unverifiable == 2


def test_dropping_values_in_table_rows_keeps_the_other_rows_paths_right():
    label = {"forms_and_endorsements": [
        {"form_number": _env("HO 00 03", 3), "form_title": _env("Special Form", 3)},
        {"form_number": _env("HO 04 90", 3), "form_title": _env("Agent phone on file", 3)},
    ], "fideon:filled": {"paths": ["forms_and_endorsements[0].form_title",
                                   "forms_and_endorsements[1].form_number"]}}
    out = verified_label(label, PAGES)
    assert "form_title" not in out["forms_and_endorsements"][0]
    assert "form_number" not in out["forms_and_endorsements"][1]
    assert out["forms_and_endorsements"][1]["form_title"]["raw"] == "Agent phone on file"


def test_the_window_rows_carry_only_values_the_document_prints():
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, _policy_window_rows

    label = _label(**{"carrier.company_name": _env("Granite State Insurance", 1),
                      "producer.agency_name": _env("Lakeside Agency", 2)})
    document = SourceDocument(source_id="s1", doc_type="policy", golden_label=label,
                              ocr_pages=PAGES, image_paths=["p1.png", "p2.png", "p3.png"],
                              lob="homeowners")
    details: list[str] = []
    rows = _policy_window_rows(document, "train", "ocr_plus_image", list(PAGES), details)
    answers = " ".join(row["messages"][-1]["content"] for row in rows)
    # Listed, never deleted (review 2026-10-26): the window holding the value
    # no page prints is left out - with Lakeside Agency, which it also holds -
    # rather than taught to skip a value it may show.
    assert "Granite State Insurance" not in answers
    assert any("left out" in d and "carrier.company_name" in d for d in details), details
    assert any("not printed on the page" in d for d in details)
