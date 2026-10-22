"""The agreed label conventions, applied as bundles are prepared (data_pipeline.ingestion.label_rules)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from data_pipeline.ingestion.label_rules import PRINTED_NOWHERE, apply_label_rules, page_texts

#: Enough ordinary words that a page's text counts as a whole page.
FILLER = " ".join(f"word{i}" for i in range(60))
PAGES = {
    1: f"Mercury Casualty Company  Policy Number NYHP0000004000\n{FILLER}",
    2: "Policy Number NYHP0000004000\nNamed Insured Clemence Fairbank\nLocation: 2 TOWN RD MOUNT MARION NY\n"
       "Coverage A Dwelling $219,000  Protection Class 3  Paid By: Mortgagee  Escrow: Yes\n"
       f"Mortgagee M & T Bank\n{FILLER}",
    3: f"Policy Number NYHP0000004000\nSection II Medical Payments $219,000 aggregate\nPage 3 of 3\n{FILLER}",
    4: f"Policy Number NYHP0000004000\nWater Leak Detection: No\n{FILLER}",
}


def _v(raw, pages, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed,
            "confidence": {"score": 1.0, "source": "vlm"}, "page_ref": pages, "flagged": False}


def _gold(**extra):
    gold = {
        "policy": {"policy_number": _v("NYHP0000004000", [2])},
        "named_insured": {"primary_name": _v("Clemence Fairbank", [2])},
        "locations": [{"unit_id": "loc_1", "location_number": _v("2", [2], 2),
                       "address": {"street": _v("2 TOWN RD", [2])}}],
        "coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A",
                       "limits": [{"limit_type": "per_occurrence", "amount": _v("$219,000", [2], 219000.0)}]}],
        "interested_parties": [{"role": "mortgagee", "name": _v("M & T Bank", [2]),
                                "is_payor": _v("Yes", [2], True)}],
        "additional_fields": [],
    }
    gold.update(extra)
    return gold


def _extra(label, raw, pages):
    return {"label": label, "value": _v(raw, pages), "section_hint": "policy", "page_ref": pages}


def _kinds(notes):
    return [kind for kind, _ in notes]


def test_a_document_level_value_lists_every_page_that_prints_it():
    gold = _gold()
    notes = apply_label_rules(gold, PAGES)
    assert gold["policy"]["policy_number"]["page_ref"] == [1, 2, 3, 4]
    assert ("page list", "policy.policy_number [2] -> [1, 2, 3, 4]") in notes


def test_a_table_value_keeps_to_the_pages_it_cites():
    gold = _gold()
    apply_label_rules(gold, PAGES)     # $219,000 is also printed on page 3, for another coverage
    assert gold["coverages"][0]["limits"][0]["amount"]["page_ref"] == [2]


def test_a_page_that_does_not_print_the_value_is_taken_out():
    gold = _gold(policy={"policy_number": _v("NYHP0000004000", [2, 3]), "effective_date": _v("06/04/2025", [2, 3])},
                 premium={"total": _v("$1,234.00", [3], 1234.0), "change": _v("$56.78", [3], 56.78)})
    pages = {**PAGES, 2: PAGES[2] + "\nEffective 06/04/2025", 3: PAGES[3] + "\nTotal $1,234.00  Change $56.78"}
    apply_label_rules(gold, pages)
    assert gold["policy"]["effective_date"]["page_ref"] == [2]


def test_a_name_keeps_a_page_it_cites_whose_text_does_not_hold_it():
    gold = _gold(carrier={"name": _v("Leatherstocking Cooperative Insurance Company", [1, 4])})
    apply_label_rules(gold, {**PAGES, 4: PAGES[4] + "\nLeatherstocking Cooperative Insurance Company"})
    assert gold["carrier"]["name"]["page_ref"] == [1, 4]               # page 1 prints it in its logo


def test_a_phrase_is_added_where_it_heads_a_line_not_where_it_falls_in_a_sentence():
    gold = _gold(document={"title": _v("Policy Declarations", [1])})
    pages = {**PAGES, 1: PAGES[1] + "\nPolicy Declarations",
             3: PAGES[3] + "\nThe limits are as shown in the Policy Declarations for this policy.",
             4: PAGES[4] + "\nPOLICY DECLARATIONS (continued)"}
    apply_label_rules(gold, pages)
    assert gold["document"]["title"]["page_ref"] == [1, 4]


def test_a_page_whose_text_holds_the_words_but_not_the_amounts_settles_nothing():
    # Amounts drawn in a font that records no characters: every caption is in the text, no figure is.
    page = PAGES[2].replace("$219,000", "").replace("Protection Class 3", "Protection Class")
    gold = _gold(premium={"total": _v("$1,234.00", [2], 1234.0)})
    notes = apply_label_rules(gold, {**PAGES, 2: page})
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_a_page_whose_text_misses_most_values_settles_nothing():
    pages = {**PAGES, 2: f"Policy Number NYHP0000004000\n{FILLER}"}     # its table was an image
    gold = _gold(named_insured={"primary_name": _v("Clemence Fairbank", [2, 3])})
    notes = apply_label_rules(gold, pages)
    # Printed on no page now - but page 2 cannot settle that, so nothing is taken out.
    assert gold["named_insured"]["primary_name"]["page_ref"] == [2, 3]
    assert "location_number" in gold["locations"][0]                   # nor the number on page 2
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_an_extra_field_repeating_a_field_folds_into_it():
    gold = _gold(additional_fields=[_extra("Policy Number", "NYHP0000004000", [3])])
    notes = apply_label_rules(gold, PAGES)
    assert gold["additional_fields"] == []
    assert ("extra field repeats a field", "additional_fields[0] 'Policy Number' -> policy.policy_number") in notes


def test_a_common_amount_is_not_taken_for_a_repeat():
    gold = _gold(premium={"total": _v("$0.00", [2], 0.0)},
                 additional_fields=[_extra("Prior Annual Fire Fee", "$0.00", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nPrior Annual Fire Fee $0.00 Total $0.00"})
    assert [e["label"] for e in gold["additional_fields"]] == ["Prior Annual Fire Fee"]


def test_a_year_is_not_taken_for_a_repeat():
    gold = _gold(watercraft=[{"unit_id": "wc_1", "year": _v("2015", [2], 2015)}],
                 additional_fields=[_extra("Trailer Information Year", "2015", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nYear 2015  Trailer Information Year 2015"})
    assert [e["label"] for e in gold["additional_fields"]] == ["Trailer Information Year"]


def test_an_extra_field_repeating_a_table_value_gives_its_row_no_page():
    gold = _gold(additional_fields=[_extra("Modifies coverage(s) at renewal", "Coverage A - Dwelling", [4])])
    gold["coverages"][0]["coverage_name"] = _v("Coverage A - Dwelling", [2])
    apply_label_rules(gold, {**PAGES, 4: PAGES[4] + "\nModifies coverage(s) at renewal: Coverage A - Dwelling"})
    assert gold["additional_fields"] == []
    assert gold["coverages"][0]["coverage_name"]["page_ref"] == [2]     # mentioned on page 4, listed on 2


def test_the_same_extra_field_twice_becomes_one_with_both_pages():
    gold = _gold(additional_fields=[_extra("Water Leak Detection", "No", [4]),
                                    _extra("Water Leak Detection", "No", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nWater Leak Detection: No"})
    assert len(gold["additional_fields"]) == 1
    assert gold["additional_fields"][0]["value"]["page_ref"] == [2, 4]


def test_page_counters_and_prose_are_not_extra_fields():
    prose = "It is your responsibility to select and maintain adequate amounts of insurance on your dwelling"
    gold = _gold(additional_fields=[_extra("Page 3 of 3", "Page 3 of 3", [3]), _extra(prose, prose, [1]),
                                    _extra("Water Leak Detection", "No", [4])])
    notes = apply_label_rules(gold, PAGES)
    assert [e["label"] for e in gold["additional_fields"]] == ["Water Leak Detection"]
    assert _kinds(notes).count("extra field not a value") == 2


def test_a_long_label_printed_before_an_amount_is_a_value():
    label = "Your 12-month policy premium excluding billing fees and payment option discounts is"
    gold = _gold(additional_fields=[_extra(label, "$352.00", [2])])
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + f"\n{label} $352.00"})
    assert [e["label"] for e in gold["additional_fields"]] == [label]


def test_a_figure_of_policy_wording_or_hidden_text_is_not_an_extra_field():
    """A label written about where a value was found - a form's standard wording,
    or text the PDF hides - is no printed label: policy wording is never a value,
    and no image shows hidden text."""
    gold = _gold(additional_fields=[
        _extra('Form boilerplate amount (context: "...bail bonds up to")', "$250", [3]),
        _extra("Bail Bond Cap (CG 00 01 boilerplate)", "$250", [3]),
        _extra("Invisible text-layer remnant of another finance agreement (not shown on the page)", "1,234.00", [4]),
        _extra("Water Leak Detection", "No", [4]),
    ])
    notes = apply_label_rules(gold, PAGES)
    assert [e["label"] for e in gold["additional_fields"]] == ["Water Leak Detection"]
    assert _kinds(notes).count("extra field from wording") == 3


def test_wording_filed_as_an_extra_field_is_dropped_and_captions_stay():
    """Clauses, figures of standard wording and sentences picked up mid-way are
    no printed captions: taught, they are values to copy out of policy text."""
    wording = [
        "Form wording, page 15: limit of insurance", "Money printed on page 4",       # drafting notes
        "b. Up to", "a. The act resulted in insured losses in excess of", "2. We do not cover",  # clauses
        "for cost of bail bonds required", "policy premium or",                       # mid-sentence
        "Supplementary Payments bail bond limit", "Federal share of terrorism losses",
        "www.travelers.com, call our toll-free telephone number", "Deductible Liability Insurance threshold",
    ]
    captions = ["Terrorism Premium", "2. Policy Period From", "eBill", "Water Leak Detection"]
    gold = _gold(additional_fields=[_extra(label, "No", [4]) for label in wording + captions])
    notes = apply_label_rules(gold, PAGES)
    assert [e["label"] for e in gold["additional_fields"]] == captions
    assert _kinds(notes).count("extra field from wording") == len(wording)


def test_an_extra_field_naming_an_empty_field_moves_into_it():
    """A value with a field of its own goes in that field, never under the extra
    fields: the prompt says so, and a label saying otherwise teaches the opposite."""
    gold = _gold(carrier={"name": _v("Mercury Casualty Company", [1])}, additional_fields=[
        _extra("Business Description", "Machine shop", [2]),
        _extra("Form of Business", "Limited Liability Company", [2]),
        _extra("Billing Type", "Direct Bill", [3]),
        _extra("Claims Call Line", "1-800-555-0100", [4]),
        _extra("Minimum Earned Premium", "25%", [3]),
        _extra("Authorized Representative", "Pat Example", [4]),     # the gold has no countersignature
    ])
    notes = apply_label_rules(gold, PAGES)
    assert gold["named_insured"]["business_description"]["raw"] == "Machine shop"
    assert gold["named_insured"]["entity_type"]["raw"] == "Limited Liability Company"
    assert gold["billing"]["bill_type"]["page_ref"] == [3]
    assert gold["carrier"]["claims_phone"]["raw"] == "1-800-555-0100"
    assert gold["premium"]["minimum_earned_percent"]["parsed"] == 25.0
    assert "minimum_earned" not in gold["premium"]
    # A line block the gold lacks is never created: the line may have none.
    assert "countersignature" not in gold
    assert [e["label"] for e in gold["additional_fields"]] == ["Authorized Representative"]
    assert _kinds(notes).count("extra field moved to its field") == 5


def test_a_minimum_earned_amount_and_a_representative_go_to_their_own_fields():
    gold = _gold(countersignature={}, additional_fields=[
        _extra("Minimum Earned Premium", "$500.00", [3]),
        _extra("Authorized Representative", "Pat Example", [4]),
    ])
    apply_label_rules(gold, PAGES)
    assert gold["premium"]["minimum_earned"]["parsed"] == 500.0
    assert gold["countersignature"]["representative_name"]["raw"] == "Pat Example"
    assert gold["additional_fields"] == []


def test_an_extra_field_stays_when_its_field_is_filled_or_two_entries_name_it():
    gold = _gold(additional_fields=[
        _extra("Business Description", "Machine shop", [2]),
        _extra("Billing Type", "Direct Bill", [3]),
        _extra("Bill Type", "Agency Bill", [4]),
    ])
    gold["named_insured"]["business_description"] = _v("Welding and fabrication", [2])
    apply_label_rules(gold, PAGES)
    assert gold["named_insured"]["business_description"]["raw"] == "Welding and fabrication"
    assert "billing" not in gold
    assert [e["label"] for e in gold["additional_fields"]] == ["Business Description", "Billing Type", "Bill Type"]


def test_a_form_number_cites_every_page_its_header_or_footer_prints_it_on():
    """A form prints its number at the head or foot of each of its pages: a window
    over its third page shows the form. Named in the body - an endorsement saying
    which form it amends - it is not that page's form."""
    body = "\n".join(f"line {n} {FILLER}" for n in range(8))
    pages = {
        2: f"Forms schedule\nCG 00 01 Commercial General Liability Coverage Form\n{body}",
        5: f"COMMERCIAL GENERAL LIABILITY\nCG 00 01 04 13\n{body}",
        6: f"{body}\nCG 00 01 04 13 Insurance Services Office, Inc., 2012 Page 2 of 16",
        9: f"{body}\nThis endorsement modifies insurance provided under CG 00 01 as amended.\n{body}",
    }
    gold = {"forms_and_endorsements": [{"form_number": _v("CG 00 01", [2])}]}
    notes = apply_label_rules(gold, pages)
    assert gold["forms_and_endorsements"][0]["form_number"]["page_ref"] == [2, 5, 6]
    assert ("page list", "forms_and_endorsements[0].form_number [2] -> [2, 5, 6]") in notes


def test_a_carrier_s_short_form_code_is_found_at_a_page_s_edge_and_a_short_word_is_not():
    body = "\n".join(f"line {n} {FILLER}" for n in range(8))
    pages = {2: f"Forms schedule\nUFR 1 Utica Forms Rider\nPRIV Privacy Notice\n{body}",
             7: f"{body}\nUFR 1 (Ed. 01/10) Page 1 of 2",
             8: f"PRIV\n{body}"}
    gold = {"forms_and_endorsements": [{"form_number": _v("UFR 1", [2])}, {"form_number": _v("PRIV", [2])}]}
    apply_label_rules(gold, pages)
    assert gold["forms_and_endorsements"][0]["form_number"]["page_ref"] == [2, 7]
    assert gold["forms_and_endorsements"][1]["form_number"]["page_ref"] == [2]


def test_protection_class_and_a_paying_mortgagee_go_to_their_fields():
    gold = _gold(additional_fields=[_extra("Protection Class", "3", [2]), _extra("Paid By", "Mortgagee", [2])])
    notes = apply_label_rules(gold, PAGES)
    assert gold["additional_fields"] == []
    assert gold["locations"][0]["protection_class"]["raw"] == "3"
    assert _kinds(notes).count("extra field moved to its field") == 2      # is_payor was already set


def test_an_extra_field_with_figures_no_page_prints_is_listed_like_a_field():
    gold = _gold(additional_fields=[_extra("Water Leak Detection", "No", [4]),
                                    _extra("Terrorism Premium", "$1,234.00", [2]),
                                    _extra("Terrorism", "Rejected by insured", [2])])
    notes = apply_label_rules(gold, PAGES)
    assert len(gold["additional_fields"]) == 3                          # listed, not deleted
    assert (PRINTED_NOWHERE, "additional_fields[1].value") in notes
    assert (PRINTED_NOWHERE, "additional_fields[2].value") not in notes  # a phrase may be reworded


def test_an_amount_printed_in_another_format_is_printed():
    gold = _gold(additional_fields=[_extra("Fire Fee", "$1,234.00", [2])])
    notes = apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nFire Fee 1234.00"})
    assert [e["label"] for e in gold["additional_fields"]] == ["Fire Fee"]
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_a_street_number_is_not_a_location_number():
    gold = _gold()
    notes = apply_label_rules(gold, PAGES)                 # "Location: 2 TOWN RD" is the address
    assert "location_number" not in gold["locations"][0]
    assert ("number not printed", "locations[0].location_number '2'") in notes


def _one_location(number="1"):
    return [{"unit_id": "loc_1", "location_number": _v(number, [2], int(number)),
             "address": {"street": _v("2 TOWN RD", [2])}}]


def test_a_location_number_printed_as_one_stays():
    gold = _gold(locations=_one_location())
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nProperty: 1 of 1"})
    assert gold["locations"][0]["location_number"]["raw"] == "1"


def test_a_number_in_a_numbered_column_is_printed():
    gold = _gold(locations=_one_location())       # a table's cells come out of the text layer as lines
    apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nLoc. #\nLocation Description and Address\n1\n2 TOWN RD"})
    assert gold["locations"][0]["location_number"]["raw"] == "1"


def test_a_number_printed_on_a_page_it_does_not_cite_is_printed():
    gold = _gold(locations=_one_location())
    apply_label_rules(gold, {**PAGES, 3: PAGES[3] + "\nCoverage Information for Location 1 of 1"})
    assert gold["locations"][0]["location_number"]["raw"] == "1"


def test_a_field_value_with_figures_no_page_prints_is_listed_not_changed():
    gold = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2]), "effective_date": _v("06/04/2025", [2])})
    before = copy.deepcopy(gold["policy"])
    notes = apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nEffective 06/04/2025"})
    assert (PRINTED_NOWHERE, "policy.policy_number") in notes
    assert gold["policy"] == before


def test_one_missing_figure_on_a_page_of_few_settles_nothing():
    gold = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2])})   # 2 of its 3 figures printed
    assert PRINTED_NOWHERE not in _kinds(apply_label_rules(gold, PAGES))


def test_the_ocr_text_of_a_scanned_page_settles_nothing():
    gold = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2]), "effective_date": _v("06/04/2025", [2])},
                 locations=[{"unit_id": "loc_1", "location_number": _v("1", [2], 1)}])
    notes = apply_label_rules(gold, {**PAGES, 2: PAGES[2] + "\nEffective 06/04/2025"}, scanned={2})
    assert PRINTED_NOWHERE not in _kinds(notes)                # OCR may have misread it
    assert "location_number" in gold["locations"][0]


def test_scanned_pages_are_those_whose_text_is_drawn_invisibly(tmp_path):
    import pymupdf

    from data_pipeline.ingestion.label_rules import scanned_pages

    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "an OCR reading laid over a scan", render_mode=3)
        doc.new_page().insert_text((72, 72), "printed text")
        doc.save(pdf)
    assert scanned_pages(pdf) == {1}
    assert scanned_pages(tmp_path / "missing.pdf") == set()


def test_a_name_is_never_listed_as_printed_nowhere():
    gold = _gold(carrier={"name": _v("Leatherstocking Cooperative Insurance Company", [1])})
    notes = apply_label_rules(gold, PAGES)                  # printed in the logo, not the text layer
    assert PRINTED_NOWHERE not in _kinds(notes)


def test_without_a_text_layer_only_the_rules_needing_no_text_apply():
    gold = _gold(additional_fields=[_extra("Page 3 of 3", "Page 3 of 3", [3])])
    notes = apply_label_rules(gold, {})
    assert _kinds(notes) == ["extra field not a value"]
    assert gold["policy"]["policy_number"]["page_ref"] == [2]
    assert "location_number" in gold["locations"][0]


def test_page_texts_reads_each_pages_text_layer(tmp_path):
    import pymupdf

    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        for text in ("first page NYHP0000004000", "second page"):
            doc.new_page().insert_text((72, 72), text)
        doc.save(pdf)
    texts = page_texts(pdf)
    assert sorted(texts) == [1, 2] and "NYHP0000004000" in texts[1]
    assert page_texts(tmp_path / "missing.pdf") == {}


def test_the_corpus_build_leaves_out_windows_holding_a_value_printed_nowhere():
    from data_pipeline.dataset_builder.build_jsonl import _held_unprinted, _unprinted_values

    label = _gold(policy={"policy_number": _v("ZZ-9999-1234", [2])})
    unprinted = _unprinted_values(label, ["policy.policy_number"])
    assert unprinted == [("policy.policy_number", "policy", {2})]
    decl = SimpleNamespace(group="decl", pages=(1, 2, 3), single=True)
    arrays = SimpleNamespace(group="arrays", pages=(1, 2), single=False)
    assert _held_unprinted(decl, "homeowners", unprinted) == ["policy.policy_number"]
    assert _held_unprinted(arrays, "homeowners", unprinted) == []        # not its section

    label = _gold(additional_fields=[_extra("Terrorism Premium", "$1,234.00", [2])])
    unprinted = _unprinted_values(label, ["additional_fields[0].value"])
    assert unprinted == [("additional_fields[0].value", "additional_fields", {2})]
    assert _held_unprinted(arrays, "homeowners", unprinted) == ["additional_fields[0].value"]
    assert _held_unprinted(decl, "homeowners", unprinted) == []


def test_the_corpus_build_leaves_out_that_window_and_builds_the_others():
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, _policy_window_rows

    texts = [PAGES[number] for number in sorted(PAGES)]
    document = SourceDocument(source_id="s1", doc_type="policy", lob="homeowners", ocr_pages=texts,
                              golden_label=_gold(policy={"policy_number": _v("ZZ-9999-1234", [2])}),
                              image_paths=[f"p{number}.png" for number in sorted(PAGES)],
                              unprinted_values=["policy.policy_number"])
    details: list[str] = []
    rows = _policy_window_rows(document, "train", "ocr_plus_image", texts, details)
    assert any("left out" in d and "policy.policy_number" in d for d in details)
    assert rows and all("ZZ-9999-1234" not in row["messages"][-1]["content"] for row in rows)


def test_a_bundles_metadata_carries_the_values_printed_nowhere():
    from data_pipeline.ingestion.prepare_bundles import _gold_changed, _unprinted

    notes = [("renders identical", "x"), (PRINTED_NOWHERE, "policy.policy_number")]
    assert _unprinted(notes) == ["policy.policy_number"]
    assert not _gold_changed(notes)                       # the gold itself is as delivered
    assert _gold_changed([*notes, ("page list", "a [1] -> [1, 2]")])


def test_corrected_gold_runs_the_rules_on_the_text_layer(tmp_path):
    import pymupdf

    from data_pipeline.ingestion.label_rules import _envelopes
    from data_pipeline.ingestion.prepare_bundles import corrected_gold

    gold = json.loads(Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")
                      .read_text(encoding="utf-8"))
    # Page 1 prints every value the label cites there but the policy number.
    printed = [str(e["raw"]) for path, e in _envelopes(gold)
               if e.get("raw") and 1 in (e.get("page_ref") or []) and path != "policy.policy_number"]
    pdf = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=2000)
        for line, text in enumerate([*printed, FILLER[:200]]):
            page.insert_text((36, 40 + 14 * line), text, fontsize=9)
        doc.save(pdf)
    _fixed, notes, problem = corrected_gold(gold, "homeowners", carrier="example", text_pdf=pdf, recodes={})
    assert problem is None
    assert (PRINTED_NOWHERE, "policy.policy_number") in notes


def test_a_link_to_a_location_with_one_building_names_the_building():
    gold = _gold(buildings=[{"unit_id": "bldg_1", "location_ref": "loc_1", "year_built": _v("1990", [2], 1990)}])
    gold["coverages"][0]["applies_to"] = ["loc_1"]
    gold["interested_parties"][0]["applies_to"] = ["loc_1", "bldg_1"]
    notes = apply_label_rules(gold, {})
    assert gold["coverages"][0]["applies_to"] == ["bldg_1"]
    assert gold["interested_parties"][0]["applies_to"] == ["bldg_1"]           # named once
    assert ("link names the building", "coverages[0].applies_to ['loc_1'] -> ['bldg_1']") in notes
    assert gold["buildings"][0]["location_ref"] == "loc_1"                    # a building still sits at its location


def test_a_location_with_two_buildings_keeps_its_links():
    gold = _gold(buildings=[{"unit_id": "bldg_1", "location_ref": "loc_1"}, {"unit_id": "bldg_2", "location_ref": "loc_1"}])
    gold["coverages"][0]["applies_to"] = ["loc_1"]
    apply_label_rules(gold, {})
    assert gold["coverages"][0]["applies_to"] == ["loc_1"]


# --------------------------------------------------------------------------
# Form editions: one convention
# --------------------------------------------------------------------------


def _forms(*numbers, edition=None):
    rows = []
    for number in numbers:
        row = {"form_number": _v(number, [5])}
        if edition is not None:
            row["edition_date"] = edition
        rows.append(row)
    return {"forms_and_endorsements": rows}


def test_an_iso_edition_inside_a_form_number_moves_to_edition_date():
    gold = _forms("CG 00 01 04 13")
    notes = apply_label_rules(gold, {})
    row = gold["forms_and_endorsements"][0]
    assert row["form_number"]["raw"] == row["form_number"]["parsed"] == "CG 00 01"
    assert row["edition_date"]["raw"] == "04 13" and row["edition_date"]["parsed"] == "04/13"
    assert row["edition_date"]["page_ref"] == [5]
    assert row["edition_date"]["confidence"] == row["form_number"]["confidence"]
    assert ("form edition moved", "forms_and_endorsements[0].form_number 'CG 00 01 04 13' -> "
            "'CG 00 01', edition '04 13'") in notes


def test_a_bracketed_edition_moves_without_its_brackets():
    gold = _forms("IL0017(11-98)")
    apply_label_rules(gold, {})
    row = gold["forms_and_endorsements"][0]
    assert row["form_number"]["raw"] == "IL0017"
    assert (row["edition_date"]["raw"], row["edition_date"]["parsed"]) == ("11-98", "11/98")


def test_a_form_number_shaped_like_an_edition_is_left_alone():
    """'HO 04 35' is a real form number whose last groups look like MM YY."""
    gold = _forms("HO 04 35", "CG 00 01", "CG 20 10 04")
    before = copy.deepcopy(gold)
    assert not apply_label_rules(gold, {})
    assert gold == before


def test_a_filled_edition_date_is_never_overwritten():
    gold = _forms("CG 00 01 04 13", edition=_v("04 13", [5], "04/13"))
    before = copy.deepcopy(gold)
    apply_label_rules(gold, {})
    assert gold == before



def test_a_frozen_gold_is_scored_under_the_edition_rule_without_being_rewritten():
    """A set frozen before the rule is not rewritten in Blob; it is read under it."""
    from evaluation.golden_eval import verified_label

    gold = _forms("CG 00 01 04 13")
    before = copy.deepcopy(gold)
    read = verified_label(gold, ["page text"])
    assert read["forms_and_endorsements"][0]["edition_date"]["parsed"] == "04/13"
    assert gold == before


# --------------------------------------------------------------------------
# Where the edition rule holds: one predicate, read alike by the corpus and the gate
# --------------------------------------------------------------------------


import pytest  # noqa: E402 - the parametrised tests below


@pytest.mark.parametrize(("doc_type", "lob", "applies"), [
    ("policy", "homeowners", True), ("policy", ["personal_auto"], True), ("policy", "classic_auto", True),
    ("policy", "gl", True), ("policy", None, True),                       # no line: the fallback, common model
    ("policy", "auto", False), ("policy", "wc", False), ("policy", "umbrella", False),
    ("policy", "property", False),
    ("policy", "pet_insurance", True),                    # no schema of its own: the fallback, as serving reads it
    ("acord", None, False), ("lossrun", None, False),
])
def test_the_label_rules_hold_on_common_model_policy_lines_only(doc_type, lob, applies):
    """The lines prepare_bundles corrects and serving conforms an answer on."""
    from data_pipeline.ingestion.label_rules import label_rules_apply

    assert label_rules_apply(doc_type, lob) is applies


def _stored_gold(client, source_id, lob, number="CG 00 01 04 13"):
    """A gold stored before the edition rule, OCR'd, as the corpus build loads it."""
    from artifact_registry import paths

    client.write_json(paths.golden_label("policy", source_id), _forms(number))
    client.write_json(paths.label_metadata("policy", source_id),
                      {"lob": lob, "reviewer_id": "a.reviewer", "review_date": "2026-10-20"})
    client.write_json(paths.ocr_meta("policy", source_id), {"page_count": 1, "mineru_version": "3.4.5"})
    client.write_text(paths.processed_page("policy", source_id, 1, "md"), f"Forms {number}")
    client.write_bytes(paths.processed_page("policy", source_id, 1, "png"), b"\x89PNG")


@pytest.mark.parametrize(("lob", "moved"), [("homeowners", True), ("auto", False), ("wc", False)])
def test_the_corpus_build_reads_a_stored_gold_exactly_as_the_gate_scores_it(lob, moved):
    """Before, the corpus build trained a stored gold's edition inside its form
    number while the gate scored it split - and the gate split every line's."""
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation.golden_eval import verified_label
    from orchestration.pipeline_dag import load_labeled_documents

    client = BlobClient(backend=InMemoryBackend(), container="main")
    _stored_gold(client, "policy_0001", lob)
    ctx = SimpleNamespace(client=client, doc_types=["policy"], tenant_id=None)
    [document] = load_labeled_documents(ctx)
    stored = client.read_json(paths.golden_label("policy", "policy_0001"))
    gate = verified_label(stored, ["Forms CG 00 01 04 13"], doc_type="policy", lob=lob)
    assert document.golden_label == gate
    row = document.golden_label["forms_and_endorsements"][0]
    assert ("edition_date" in row) is moved
    assert row["form_number"]["raw"] == ("CG 00 01" if moved else "CG 00 01 04 13")
    assert stored == _forms("CG 00 01 04 13")                              # never rewritten in Blob


def test_the_corpus_builds_verification_applies_the_rule_when_told_the_line():
    from data_pipeline.dataset_builder.label_verification import verified_label

    gold = _forms("CG 00 01 04 13")
    assert verified_label(gold, ["text"]) is gold                         # as build_jsonl calls it today
    moved = verified_label(gold, ["text"], doc_type="policy", lob="homeowners")
    assert moved["forms_and_endorsements"][0]["edition_date"]["parsed"] == "04/13"
    assert verified_label(gold, ["text"], doc_type="policy", lob="auto") is gold
    assert gold == _forms("CG 00 01 04 13")
