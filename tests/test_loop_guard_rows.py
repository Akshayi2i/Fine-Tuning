"""A loop that writes one row again with another field changing is stopped too (2026-11-10).

On the pod (2026-11-09) v3 wrote policy_0071's form "APDEC 0919" 166 times, each with another title (64 different),
to the 12,288-token limit: production's guard, which needs the end to repeat exactly (``repeating_tail``), let it
run. The other four loops of that run repeated exactly and were caught at 8,000-14,000 characters.
"""

from __future__ import annotations

import json
import time

from inference_core.model_runner import STOPPED_REPEATING
from inference_core.openai_backend import (
    _read_stream,
    repeating_body,
    repeating_row,
    repeating_tail,
    repetition,
)

TITLES = [f"Package Endorsement Toolbox {kind} {n}" for n in range(40) for kind in ("Limit", "Premium")]


def _form(number: str, title: str, page: int) -> str:
    return json.dumps({"form_number": {"r": number, "v": number, "p": [page]}, "title": {"v": title, "p": [page]}},
                      separators=(",", ":"))


def _forms_loop(rows: int) -> str:
    """policy_0071's loop: the same form number, a new title each time."""
    return '{"forms_and_endorsements":[' + ",".join(_form("APDEC 0919", TITLES[i % len(TITLES)], 1 + i % 4)
                                                   for i in range(rows))


def test_the_same_form_number_with_titles_that_change_is_a_loop_the_exact_check_misses():
    text = _forms_loop(200)
    assert repeating_tail(text) is None
    assert repeating_row(text) == ("form_number", "APDEC 0919")


def test_a_real_forms_schedule_is_not_a_loop():
    text = '{"forms_and_endorsements":[' + ",".join(_form(f"BP {n:02d} {n % 12 + 1:02d} 18", TITLES[n % 80], n % 9 + 1)
                                               for n in range(300))
    assert repeating_row(text) is None


def test_the_same_coverages_at_each_location_are_not_a_loop():
    rows = []
    for location in range(1, 60):
        for name in ("Building", "Business Personal Property"):
            rows.append(json.dumps({"coverage_name": {"v": name, "p": [location]},
                                    "applies_to": [f"loc_{location}"]}, separators=(",", ":")))
    assert repeating_row('{"coverages":[' + ",".join(rows)) is None


def test_a_streamed_answer_writing_one_row_again_is_stopped_early():
    pieces = [_forms_loop(400)[i:i + 50] for i in range(0, len(_forms_loop(400)), 50)]
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}".encode() for piece in pieces]
    result = _read_stream(lines + [b"data: [DONE]"], started=time.perf_counter())
    assert result["finish_reason"] == STOPPED_REPEATING
    assert len(result["text"]) < len(_forms_loop(400)) / 2                # long before it would end


def test_a_real_page_naming_one_party_many_times_is_not_a_loop():
    """Review 2026-11-10 (finding 1): a single page cannot be read again in halves, so serving drops a window the
    guard stops there. The same mortgagee on 39 locations, the same coverage on each, must not look like a loop."""
    rows = [json.dumps({"name": {"v": "Wells Fargo Bank, N.A.", "p": [4]}, "loan_number": {"v": f"L-{n}", "p": [4]},
                        "applies_to": [f"loc_{n}"]}, separators=(",", ":")) for n in range(1, 40)]
    assert repeating_row('{"interested_parties":[' + ",".join(rows)) is None


def test_the_stop_is_logged_by_field_not_by_the_value_it_repeated(caplog):
    """Review 2026-11-10 (finding 8): a repeated name may be an insured's or a lender's."""
    import logging

    pieces = [_forms_loop(400)[i:i + 50] for i in range(0, len(_forms_loop(400)), 50)]
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}".encode() for piece in pieces]
    with caplog.at_level(logging.WARNING):
        _read_stream(lines + [b"data: [DONE]"], started=time.perf_counter())
    assert "form_number" in caplog.text and "APDEC" not in caplog.text


def _coverage(code: str, name: str, name_page: int, amount: str, amount_page: int, applies_to=None) -> str:
    row = {"coverage_code": code, "coverage_name": {"v": name, "p": [name_page]},
           "limits": [{"limit_type": "per_occurrence", "amount": {"r": amount, "v": float(amount.strip("$").replace(",", "")),
                                                                    "p": [amount_page]}}]}
    if applies_to:
        row["applies_to"] = applies_to
    return json.dumps(row)


def _body_loop(rows: int) -> str:
    """policy_0003 pages 15-20 (2026-11-11): one coverage, one $25,000 limit cited to page 16, written under each
    form title of pages 17-19 in turn - every row named apart, so neither other test sees it."""
    return '{"coverages": [' + ", ".join(_coverage("BOPGL_BI_WEBSITES", TITLES[i % len(TITLES)], 17 + i % 3,
                                                    "$25,000", 16) for i in range(rows))


def test_one_row_written_under_name_after_name_is_a_loop_the_other_tests_miss():
    text = _body_loop(200)
    assert repeating_tail(text) is None and repeating_row(text) is None
    assert repeating_body(text) is not None
    assert repetition(text)[0] == "repeating_body"


def test_a_few_rows_alike_but_their_names_are_allowed():
    assert repeating_body(_body_loop(39)) is None


def test_one_coverage_on_each_vehicle_is_not_a_loop():
    """The same code and limit on 80 vehicles: what each applies to, written after its limits, tells the rows apart
    (cycle 1's Liberty Mutual auto key holds one code 39 times)."""
    rows = [_coverage(f"X_VEHICLE_{name.upper()}", name, 3, amount, 3, [f"veh_{n}"]) for n in range(80)
            for name, amount in (("Liability", "$1,000,000"), ("Comprehensive", "$500"), ("Collision", "$1,000"))]
    text = '{"coverages": [' + ", ".join(rows)
    assert repeating_body(text) is None and repetition(text) is None


def test_real_lists_are_left_alone_by_every_test():
    forms = '{"forms_and_endorsements":[' + ",".join(_form(f"BP {n:02d} {n % 12 + 1:02d} 18", TITLES[n % 80], n % 9 + 1)
                                                     for n in range(300))
    per_location = [json.dumps({"coverage_name": {"v": name, "p": [loc]}, "applies_to": [f"loc_{loc}"]})
                    for loc in range(1, 60) for name in ("Building", "Business Personal Property")]
    coverages = [_coverage(f"BOPGL_{n}", f"Coverage {n}", 2 + n % 5, f"${(n + 1) * 5000:,}", 2 + n % 5)
                 for n in range(120)]
    for text in (forms, '{"coverages":[' + ",".join(per_location), '{"coverages": [' + ", ".join(coverages)):
        assert repetition(text) is None


def test_a_streamed_answer_writing_one_row_under_new_names_is_stopped_early():
    loop = _body_loop(400)
    pieces = [loop[i:i + 50] for i in range(0, len(loop), 50)]
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}".encode() for piece in pieces]
    result = _read_stream(lines + [b"data: [DONE]"], started=time.perf_counter())
    assert result["finish_reason"] == STOPPED_REPEATING
    assert len(result["text"]) < len(loop) / 2
