"""A served answer held to the conventions its labels follow (serving.conventions)."""

from __future__ import annotations

from serving.conventions import EXTRA_DROPPED_FLAG, NOT_PRINTED_FLAG, conform

FILLER = " ".join(f"word{i}" for i in range(60))
PAGES = {
    1: f"Policy Number NYHP0000004000  Effective 06/04/2025\n{FILLER}",
    2: "Policy Number NYHP0000004000\nCoverage A Dwelling $219,000  Coverage B $21,900  Protection Class 3\n"
       f"Water Leak Detection: No  Total Premium $1,234.00\n{FILLER}",
}


def _v(raw, pages, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": pages}


def _extra(label, raw, pages):
    return {"label": label, "value": _v(raw, pages), "section_hint": "policy", "page_ref": pages}


def _answer(**extra):
    answer = {
        "policy": {"policy_number": _v("NYHP0000004000", [2]), "effective_date": _v("06/04/2025", [1])},
        "premium": {"total": _v("$1,234.00", [2], 1234.0)},
        "locations": [{"unit_id": "loc_1"}],
        "coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A",
                       "limits": [{"limit_type": "per_occurrence", "amount": _v("$219,000", [2], 219000.0)}]},
                      {"coverage_id": "cov_2", "coverage_code": "HO_COV_B",
                       "limits": [{"limit_type": "per_occurrence", "amount": _v("$21,900", [2], 21900.0)}]}],
    }
    answer.update(extra)
    return answer


def test_a_served_value_lists_every_page_that_prints_it():
    out = conform(_answer(), {}, PAGES, native=True)
    assert out.extraction["policy"]["policy_number"]["page_ref"] == [1, 2]


def test_an_extra_field_repeating_a_field_is_folded_and_the_spans_follow_the_rest():
    answer = _answer(additional_fields=[_extra("Policy Number", "NYHP0000004000", [2]),
                                        _extra("Water Leak Detection", "No", [2])])
    spans = {"additional_fields[0].value": "s-policy", "additional_fields[1].value": "s-leak",
             "policy.policy_number": "s-number"}
    out = conform(answer, spans, PAGES, native=True)
    assert [e["label"] for e in out.extraction["additional_fields"]] == ["Water Leak Detection"]
    assert out.spans["additional_fields[0].value"] == "s-leak" and "additional_fields[1].value" not in out.spans
    assert out.spans["policy.policy_number"] == "s-number"


def test_a_field_value_no_page_prints_is_flagged_and_kept():
    answer = _answer(policy={"policy_number": _v("ZZ-9999-1234", [2]), "effective_date": _v("06/04/2025", [1])})
    out = conform(answer, {}, PAGES, native=True)
    assert f"policy.policy_number:{NOT_PRINTED_FLAG}" in out.flags
    assert out.extraction["policy"]["policy_number"]["raw"] == "ZZ-9999-1234"


def test_an_extra_field_no_page_prints_is_dropped_on_a_digital_document():
    answer = _answer(additional_fields=[_extra("Water Leak Detection", "No", [2]),
                                        _extra("Terrorism Premium", "$4,321.00", [2])])
    out = conform(answer, {}, PAGES, native=True)
    assert [e["label"] for e in out.extraction["additional_fields"]] == ["Water Leak Detection"]
    assert out.flags == [EXTRA_DROPPED_FLAG]


def test_a_page_missing_several_of_the_answers_figures_settles_nothing():
    answer = _answer(policy={"policy_number": _v("ZZ-9999-1234", [2]), "effective_date": _v("06/04/2025", [1])},
                     additional_fields=[_extra("Terrorism Premium", "$4,321.00", [2])])
    out = conform(answer, {}, PAGES, native=True)      # two of page 2's five figures missing: its text, or the model?
    assert not out.flags and len(out.extraction["additional_fields"]) == 1


def test_a_scan_keeps_every_value_its_ocr_does_not_find():
    answer = _answer(additional_fields=[_extra("Terrorism Premium", "$4,321.00", [2])])
    out = conform(answer, {}, PAGES, native=False)
    assert len(out.extraction["additional_fields"]) == 1 and not out.flags


def test_a_moved_extra_field_takes_its_span_to_its_field():
    answer = _answer(additional_fields=[_extra("Protection Class", "3", [2])])
    out = conform(answer, {"additional_fields[0].value": "s-class"}, PAGES, native=True)
    assert "additional_fields" not in out.extraction
    assert out.extraction["locations"][0]["protection_class"]["raw"] == "3"
    assert out.spans["locations[0].protection_class"] == "s-class"


def test_without_page_text_only_the_rules_needing_none_apply():
    answer = _answer(additional_fields=[_extra("Page 3 of 3", "Page 3 of 3", [2])])
    out = conform(answer, {}, None, native=True)
    assert "additional_fields" not in out.extraction
    assert out.extraction["policy"]["policy_number"]["page_ref"] == [2]


def _missed(answer, text):
    from serving.conventions import likely_missed

    return likely_missed(answer, {1: text}, [1], ["document", "policy", "named_insured", "premium"])


def test_an_empty_field_whose_label_and_value_open_a_line_is_flagged():
    from serving.conventions import LIKELY_MISSED_FLAG

    text = "HOMEOWNERS DECLARATIONS\nTransaction Effective Date: 06/23/2025\nPolicy Number NYHP0000004000"
    policy = {"policy": {"policy_number": {"raw": "NYHP0000004000", "parsed": "NYHP0000004000"}}}
    assert _missed(policy, text) == [f"document.transaction_effective_date:{LIKELY_MISSED_FLAG}"]
    answered = {**policy, "document": {"transaction_effective_date": {"raw": "06/23/2025", "parsed": "06/23/2025"}}}
    assert _missed(answered, text) == []


def test_a_line_belongs_to_the_longest_label_it_opens_with():
    # "Policy Term Effective Date" is the effective date's label, not the term's.
    text = "Policy Term Effective Date: 03/03/2024, 12:01AM Standard Time"
    flags = _missed({}, text)
    assert not any(flag.startswith("policy.term_months") for flag in flags)


def test_a_label_followed_by_no_value_of_its_type_raises_no_flag():
    # A term in months is a number; a period printed after "Policy Term" is two dates.
    assert not any(f.startswith("policy.term_months") for f in _missed({}, "Policy Term:  From 02/15/2026 to 02/15/2027"))
    assert any(f.startswith("policy.term_months") for f in _missed({}, "Policy Term: 12 months"))


def test_a_label_two_fields_share_raises_no_flag():
    # The named insured's and every driver's: which one was missed?
    assert not any(f.startswith("named_insured.marital_status") for f in _missed({}, "Marital Status: M"))


def test_a_table_block_is_not_looked_in():
    from serving.conventions import declaration_objects

    assert "lob_parts" not in declaration_objects("homeowners") and "policy" in declaration_objects("homeowners")


def test_an_edition_moved_out_of_a_form_number_takes_the_numbers_span():
    answer = _answer(forms_and_endorsements=[{"form_number": _v("CG 00 01 04 13", [2])}])
    out = conform(answer, {"forms_and_endorsements[0].form_number": "s-form"}, PAGES, native=True)
    row = out.extraction["forms_and_endorsements"][0]
    assert row["form_number"]["raw"] == "CG 00 01" and row["edition_date"]["parsed"] == "04/13"
    assert out.spans["forms_and_endorsements[0].edition_date"] == "s-form"
    assert out.spans["forms_and_endorsements[0].form_number"] == "s-form"

