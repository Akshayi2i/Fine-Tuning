"""A loop that writes one row again with another field changing is stopped too (2026-11-10).

On the pod (2026-11-09) v3 wrote policy_0071's form "APDEC 0919" 166 times, each with another title (64 different),
to the 12,288-token limit: production's guard, which needs the end to repeat exactly (``repeating_tail``), let it
run. The other four loops of that run repeated exactly and were caught at 8,000-14,000 characters.
"""

from __future__ import annotations

import json
import time

from inference_core.model_runner import STOPPED_REPEATING
from inference_core.openai_backend import _read_stream, repeating_row, repeating_tail

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
