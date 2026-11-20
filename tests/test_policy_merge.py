"""serving.policy_merge — one canonical document from a policy's section windows.

A long policy is read as a cross product of section groups and page windows, so
the same value can come back from several windows. These tests pin the merge
rules that turn those partial answers into one document without a model in the
loop, and that keep each value's confidence evidence attached to it.
"""

from __future__ import annotations

from serving.policy_merge import PolicyWindow, merge_policy_windows


def _fv(value, page=1, parsed=None):
    return {"raw": value, "parsed": value if parsed is None else parsed, "page_ref": [page]}


def _loc(number, city, page):
    return {"location_number": _fv(number, page), "address": {"city": _fv(city, page)}}


def test_decl_wins_a_conflict_because_it_merges_first():
    """The declarations value wins, whatever order the windows arrive in — the
    same rule policy.jinja states to the model."""
    decl = PolicyWindow("decl", [1, 2, 3], {"policy": {"policy_number": _fv("HO-1")}})
    later = PolicyWindow("lineblk", [9], {"policy": {"policy_number": _fv("HO-9", 9)}})

    merged = merge_policy_windows([later, decl])

    assert merged.extraction["policy"]["policy_number"]["raw"] == "HO-1"
    assert merged.conflicts and "policy.policy_number" in merged.conflicts[0]
    assert "policy.policy_number:merge_conflict" in merged.review_flags


def test_the_same_value_from_two_windows_is_one_value_with_both_pages():
    a = PolicyWindow("lineblk", [4, 5], {"homeowners": {"coverage_a": _fv("300000", 5)}})
    b = PolicyWindow("lineblk", [6, 7], {"homeowners": {"coverage_a": _fv("$300,000", 6, "300000")}})

    merged = merge_policy_windows([a, b])

    leaf = merged.extraction["homeowners"]["coverage_a"]
    assert leaf["page_ref"] == [5, 6]
    assert not merged.conflicts


def test_a_row_spanning_a_window_boundary_collapses_to_one():
    """Training puts a boundary row in the gold of BOTH windows, so the duplicate
    is the expected case at serving, not an error."""
    first = PolicyWindow("arrays", [140, 141], {"locations": [_loc("1", "Toledo", 141)]})
    # The same row, read again by the next window with different case and
    # spacing: keys compare on normalised values.
    second = PolicyWindow("arrays", [141, 142], {"locations": [
        _loc("1", " TOLEDO ", 141), _loc("2", "Dayton", 142),
    ]})

    merged = merge_policy_windows([first, second])

    rows = merged.extraction["locations"]
    assert [r["address"]["city"]["raw"] for r in rows] == ["Toledo", "Dayton"]
    assert merged.duplicates_collapsed == 1
    assert not merged.conflicts


def test_the_more_complete_row_is_kept_and_gaps_are_filled():
    thin = {"form_number": _fv("HO 00 03", 10), "edition_date": _fv("05/11", 10),
            "premium": _fv("120", 10)}
    full = {"form_number": _fv("HO 00 03", 11), "edition_date": _fv("05/11", 11),
            "form_title": _fv("Special Form", 11), "is_included": _fv("Y", 11)}
    merged = merge_policy_windows([
        PolicyWindow("arrays", [10], {"forms_and_endorsements": [thin]}),
        PolicyWindow("arrays", [11], {"forms_and_endorsements": [full]}),
    ])
    (row,) = merged.extraction["forms_and_endorsements"]
    assert row["form_title"]["raw"] == "Special Form"     # from the fuller row
    assert row["premium"]["raw"] == "120"                 # a gap filled from the other
    assert row["form_number"]["page_ref"] == [10, 11]


def test_two_editions_of_one_form_stay_two_rows():
    """`05/11` is a form edition, not a parseable date. Letting it normalise to
    nothing would make two different editions of one form a single row."""
    merged = merge_policy_windows([
        PolicyWindow("arrays", [10], {"forms_and_endorsements": [
            {"form_number": _fv("HO 00 03", 10), "edition_date": _fv("05/11", 10)},
        ]}),
        PolicyWindow("arrays", [11], {"forms_and_endorsements": [
            {"form_number": _fv("HO 00 03", 11), "edition_date": _fv("10/00", 11)},
        ]}),
    ])
    assert len(merged.extraction["forms_and_endorsements"]) == 2
    assert merged.duplicates_collapsed == 0


def _form(number, edition, page, **more):
    row = {"form_number": _fv(number, page), **{k: _fv(v, page) for k, v in more.items()}}
    if edition is not None:
        row["edition_date"] = _fv(edition, page)
    return row


def _forms(*windows, lob=None):
    merged = merge_policy_windows([
        PolicyWindow("arrays", [rows[0]["form_number"]["page_ref"][0]], {"forms_and_endorsements": rows})
        for rows in windows
    ], lob)
    return merged, merged.extraction["forms_and_endorsements"]


def test_one_form_written_two_ways_by_two_windows_is_one_row():
    """Its schedule line and its own page: the edition inside the number in one,
    as a year in the other. Keyed on the number and edition as written, the
    first baseline kept 542 of 1,939 form rows twice."""
    merged, rows = _forms([_form("SP 30 23 10 18", "10/18", 13, premium="250")],
                          [_form("SP 30 23", "2018", 94, title="ACCOUNTS RECEIVABLE")])
    (row,) = rows
    assert row["premium"]["raw"] == "250" and row["title"]["raw"] == "ACCOUNTS RECEIVABLE"
    assert merged.duplicates_collapsed == 1


def test_a_form_whose_edition_one_window_did_not_read_joins_its_row():
    _, rows = _forms([_form("CG 00 01", "04/13", 3)], [_form("CG 00 01", None, 40, title="CGL Coverage Form")])
    assert len(rows) == 1 and rows[0]["edition_date"]["raw"] == "04/13"


def test_editions_that_contradict_each_other_stay_apart():
    _, by_month = _forms([_form("CG 00 01 04 13", None, 3)], [_form("CG 00 01", "12/07", 40)])
    _, by_year = _forms([_form("SP 30 23", "2018", 3)], [_form("SP 30 23", "2020", 40)])
    _, other_form = _forms([_form("SP 30 23", "10/18", 3)], [_form("SP 30 24", "10/18", 40)])
    assert len(by_month) == len(by_year) == len(other_form) == 2


def test_a_common_model_line_joins_a_form_read_twice_too():
    _, rows = _forms([_form("SP 30 18 10 18", "10/18", 13)], [_form("SP 30 18", "2018", 86)], lob="bopgl")
    assert len(rows) == 1


def test_rows_with_no_key_value_are_kept_rather_than_guessed_together():
    """A dropped row understates the policy and nothing downstream catches it;
    a duplicate is at worst a reviewable extra."""
    merged = merge_policy_windows([
        PolicyWindow("arrays", [3], {"deductibles": [{"notes": _fv("see schedule", 3)}]}),
        PolicyWindow("arrays", [4], {"deductibles": [{"notes": _fv("see schedule", 4)}]}),
    ])
    assert len(merged.extraction["deductibles"]) == 2
    assert merged.unkeyed_rows == 2


def test_spans_follow_their_values_through_reindexing():
    """Row 0 of the second window becomes row 1 of the merged table. Its span
    has to move with it, or confidence lands on the wrong location."""
    first = PolicyWindow(
        "arrays", [1], {"locations": [_loc("1", "Toledo", 1)]},
        spans={"locations[0].address.city": "span-toledo"},
    )
    second = PolicyWindow(
        "arrays", [2], {"locations": [_loc("2", "Dayton", 2)]},
        spans={"locations[0].address.city": "span-dayton"},
    )

    merged = merge_policy_windows([first, second])

    assert merged.spans["locations[0].address.city"] == "span-toledo"
    assert merged.spans["locations[1].address.city"] == "span-dayton"


def test_a_collapsed_duplicate_keeps_the_kept_rows_span():
    first = PolicyWindow(
        "arrays", [1], {"locations": [_loc("1", "Toledo", 1)]},
        spans={"locations[0].address.city": "span-first"},
    )
    second = PolicyWindow(
        "arrays", [2], {"locations": [_loc("1", "Toledo", 2)]},
        spans={"locations[0].address.city": "span-second"},
    )
    merged = merge_policy_windows([first, second])
    assert len(merged.extraction["locations"]) == 1
    assert merged.spans["locations[0].address.city"] == "span-first"


def test_sections_from_different_groups_combine_without_conflict():
    merged = merge_policy_windows([
        PolicyWindow("decl", [1], {"carrier": {"company_name": _fv("Granite Mutual")}}),
        PolicyWindow("arrays", [5], {"locations": [_loc("1", "Toledo", 5)]}),
        PolicyWindow("lineblk", [6], {"homeowners": {"coverage_a": _fv("300000", 6)}}),
    ])
    assert set(merged.extraction) == {"carrier", "locations", "homeowners"}
    assert not merged.conflicts


def test_a_self_contained_line_keys_its_rows_on_its_own_map():
    """No line is the fallback, read with the common-model profile since common
    model 1.1.0; a self-contained line's merge names its line, so its rows are
    keyed on its own map (a party is its name and type), not the common model's
    (its role and name)."""
    def party(kind, page):
        return {"name": _fv("FIRST BANK", page), "party_type": _fv(kind, page)}

    first = PolicyWindow("arrays", [3], {"interested_parties": [party("Mortgagee", 3)]})
    second = PolicyWindow("arrays", [4], {"interested_parties": [party("Loss Payee", 4)]})

    merged = merge_policy_windows([first, second], lob="property")

    assert len(merged.extraction["interested_parties"]) == 2
    assert not merged.conflicts


def test_the_fallback_joins_a_coverage_with_no_unit_only_when_nothing_differs():
    """The fallback has no units, so a coverage's key always lacks them: two
    windows' rows of one code are one row only when no value both state differs,
    as on a common-model line."""
    def coverage(premium, page):
        return {"coverage_code": "X_LIABILITY", "coverage_name": _fv("Liability", page),
                "premium": _fv(premium, page)}

    windows = [PolicyWindow("arrays", [3], {"coverages": [coverage("100", 3)]}),
               PolicyWindow("arrays", [4], {"coverages": [coverage("250", 4)]}),
               PolicyWindow("arrays", [5], {"coverages": [coverage("250", 5)]})]

    rows = merge_policy_windows(windows).extraction["coverages"]

    assert [row["premium"]["raw"] for row in rows] == ["100", "250"]
    assert rows[1]["premium"]["page_ref"] == [4, 5]


def test_merging_leaves_each_window_answer_as_the_model_wrote_it():
    """Scoring merges the same answers twice and saves them after (2026-11-11): a merge that folds rows and pages
    into the first window's answer changed what was saved, and a second merge started from the folded one."""
    import copy

    first = PolicyWindow("arrays", [140, 141], {"policy": {"policy_number": _fv("HO-1", 140)},
                                                "locations": [_loc("1", "Toledo", 141)]})
    second = PolicyWindow("arrays", [141, 142], {"policy": {"policy_number": _fv("HO-1", 142)},
                                                 "locations": [_loc("1", " TOLEDO ", 141), _loc("2", "Dayton", 142)]})
    before = copy.deepcopy([first.extraction, second.extraction])

    once = merge_policy_windows([first, second])
    again = merge_policy_windows([first, second])

    assert [first.extraction, second.extraction] == before
    assert once.extraction == again.extraction
    assert once.extraction["policy"]["policy_number"]["page_ref"] == [140, 142]


def test_one_form_spaced_two_ways_by_its_schedule_and_its_own_page_is_one_row():
    """v4's gate, Western World (2026-11-19): the schedule prints "IL0017 11_98", the form's page "IL 00 17 11 98"."""
    merged, rows = _forms([_form("IL0017", "11_98", 7, title="COMMON POLICY CONDITIONS")],
                          [_form("IL 00 17", "11 98", 10)])
    (row,) = rows
    assert row["title"]["raw"] == "COMMON POLICY CONDITIONS"
    assert merged.duplicates_collapsed == 1


def test_spacing_never_joins_two_different_forms_or_editions():
    _, numbers = _forms([_form("CG2139", "10/93", 7)], [_form("CG 21 38", "10/93", 54)])
    _, editions = _forms([_form("CG2147", "12/07", 7)], [_form("CG 21 47", "12/04", 55)])
    assert len(numbers) == len(editions) == 2


def test_a_wc_form_number_the_model_cut_in_two_is_joined_and_meets_its_own_page():
    """v4's gate, Merchants WC (2026-11-20): the schedule's "WC 00 04 14 A" came back as "WC 00 04" + edition "14 A"."""
    merged, rows = _forms([_form("WC 00 04", "14 A", 3), _form("MU 85 66", "10 08", 3)],
                          [_form("WC 00 04 14 A", None, 12, title="90-DAY REPORTING REQUIREMENT")])
    by_number = {row["form_number"]["raw"]: row for row in rows}
    assert set(by_number) == {"WC 00 04 14 A", "MU 85 66"}
    assert by_number["WC 00 04 14 A"]["title"]["raw"] == "90-DAY REPORTING REQUIREMENT"
    assert by_number["MU 85 66"]["edition_date"]["raw"] == "10 08"       # an edition that is one stays


def test_only_a_cut_wc_number_is_joined():
    _, rows = _forms([_form("WC 00 04", "04/84", 3), _form("CG 20 10", "04 13", 3), _form("WC 00 04", "04 04", 3)])
    assert sorted((row["form_number"]["raw"], row["edition_date"]["raw"]) for row in rows) == [
        ("CG 20 10", "04 13"), ("WC 00 04", "04 04"), ("WC 00 04", "04/84")]


def _limit(kind, amount=None, words=None, page=3):
    row = {"limit_type": kind}
    if amount is not None:
        row["amount"] = _fv(str(amount), page, amount)
    if words is not None:
        row["description"] = _fv(words, page)
    return row


def test_a_limits_table_read_in_two_halves_is_joined_row_by_row():
    """v4's gate, Merchants WC: amounts from one window, wording from another - six half-rows."""
    coverage = {"coverage_code": "WC_EMPLOYERS_LIABILITY", "coverage_name": _fv("Employers Liability", 3), "limits": [
        _limit("per_accident", 100000), _limit("policy_limit", 500000), _limit("each_employee", 100000),
        _limit("per_accident", words="Bodily Injury by Accident"),
        _limit("policy_limit", words="Bodily Injury by Disease"),
        _limit("each_employee", words="Bodily Injury by Disease")]}
    merged = merge_policy_windows([PolicyWindow("decl", [3], {"coverages": [coverage]})], "wc")
    (row,) = merged.extraction["coverages"]
    assert [(lim["amount"]["parsed"], lim["description"]["raw"]) for lim in row["limits"]] == [
        (100000, "Bodily Injury by Accident"), (500000, "Bodily Injury by Disease"), (100000, "Bodily Injury by Disease")]


def test_limits_halves_that_do_not_line_up_are_left_alone():
    from serving.policy_merge import join_split_limits

    extraction = {"coverages": [{"limits": [_limit("per_occurrence", 50000), _limit("other", words="12 months"),
                                            _limit("other", words="None")]}]}
    assert join_split_limits(extraction) == 0
    assert len(extraction["coverages"][0]["limits"]) == 3


def _rating(code, words, premium, page, **more):
    row = {"class_code": _fv(code, page), "class_description": _fv(words, page),
           "premium": _fv(f"${premium}", page, float(premium))}
    row.update({name: _fv(value, page) for name, value in more.items()})
    return row


def test_a_classification_read_twice_with_its_exposure_split_two_ways_is_one_row():
    """v4's gate, U.S. Underwriters: the schedule on page 2 and again on page 15, "6 Per Additional Insured" read as
    basis "6 Per Additional" in one and as amount 6 + basis "Per Additional" in the other - every class twice."""
    first = [_rating("49950", "Additional Insured - Designated Person", 300, 2, exposure_basis="6 Per Additional"),
             _rating("41670", "Membership Organization (Business)", 415, 2, rate="4.151")]
    again = [_rating("49950", "Additional Insured - Designated Person", 300, 15, exposure_basis="Per Additional",
                     exposure_amount="6"),
             _rating("41670", "Membership Organization (Business)", 415, 15)]
    merged = merge_policy_windows([PolicyWindow("decl", [2], {"rating_exposures": first}),
                                   PolicyWindow("decl", [15], {"rating_exposures": again})], "gl")
    rows = merged.extraction["rating_exposures"]
    assert [(r["class_code"]["raw"], r["exposure_basis"]["raw"] if r.get("exposure_basis") else None) for r in rows] \
        == [("49950", "6 Per Additional"), ("41670", None)]
    assert rows[1]["rate"]["raw"] == "4.151"


def test_two_classes_under_one_code_or_premium_stay_two_rows():
    from serving.policy_merge import join_repeated_rating_rows

    extraction = {"rating_exposures": [_rating("49950", "Additional Insured - Designated Person", 300, 2),
                                       _rating("49950", "Primary and Non-Contributory wording", 300, 2),
                                       _rating("49950", "Additional Insured - Designated Person", 350, 2)]}
    assert join_repeated_rating_rows(extraction) == 0
    assert len(extraction["rating_exposures"]) == 3
