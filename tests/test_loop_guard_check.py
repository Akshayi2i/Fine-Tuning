"""scripts/loop_guard_check.py: a good answer the stream's loop stop would cut short is found (2026-11-11)."""

from __future__ import annotations

import json

from scripts.loop_guard_check import check, first_stop, main


def _env(value, page=1):
    return {"raw": value, "parsed": value, "page_ref": [page]}


def _forms(n: int, same: bool = False) -> dict:
    return {"forms_and_endorsements": [
        {"form_number": _env("APDEC 0919" if same else f"CG {i:02d} {i + 10:02d}"),
         "title": _env(f"Endorsement title number {i} for the policy"), "edition_date": _env("04/13")}
        for i in range(n)]}


def _write(tmp_path, records) -> str:
    path = tmp_path / "run.answers.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def _record(answer=None, unparsed=None, halves=()):
    return {"row": {"source_id": "policy_0001", "task": "policy_schedule", "modality_mode": "image_only",
                    "window_pages": [3, 4]},
            "answer": answer, "unparsed_text": unparsed, "halves": list(halves)}


def test_a_long_list_of_distinct_rows_is_not_stopped():
    assert first_stop(json.dumps(_forms(200))) is None


def test_one_row_name_written_again_and_again_is_stopped_early():
    text = json.dumps(_forms(400, same=True))
    at, test = first_stop(text)
    assert test.startswith("repeating_row") and at < len(text)


def test_the_report_counts_good_answers_stopped_and_where_cut_off_ones_would_have_ended(tmp_path, capsys):
    loop = json.dumps(_forms(400, same=True))[:-50]                      # cut off: no closing brace
    path = _write(tmp_path, [_record(answer=_forms(60)), _record(unparsed=loop, halves=[_record(answer=_forms(5))])])
    report = check(path)
    assert report["good_answers"] == 2 and report["good_answers_stopped"] == 0
    [cut] = report["cut_off_answers"]
    assert cut["at"] is not None and cut["at"] < cut["length"]
    assert main(["--answers", path]) == 0
    assert "real answers wrongly stopped: 0" in capsys.readouterr().out


def test_an_answer_that_ran_to_the_limit_but_parses_counts_as_cut_off_not_good(tmp_path):
    """Serving reads such a window again in halves, so stopping it early loses nothing."""
    record = {**_record(answer=_forms(400, same=True)), "cut_off": True}
    report = check(_write(tmp_path, [record]))
    assert report["good_answers"] == 0 and report["good_answers_stopped"] == 0
    [cut] = report["cut_off_answers"]
    assert cut["at"] is not None and cut["at"] < cut["length"]


def test_a_failed_reading_saved_with_an_empty_answer_is_not_a_good_answer(tmp_path):
    """A failed reading keeps {} as its answer beside its error; the unreadable text is the cut-off answer."""
    loop = json.dumps(_forms(400, same=True))[:-50]
    failed = {**_record(answer={}, unparsed=loop), "error": "stopped at the token limit", "cut_off": True}
    request_failed = {**_record(answer={}), "error": "ConnectionError: server gone"}
    report = check(_write(tmp_path, [failed, request_failed, _record(answer=_forms(3))]))
    assert report["good_answers"] == 1
    [cut] = report["cut_off_answers"]
    assert cut["length"] == len(loop) and cut["at"] is not None


def test_a_good_answer_the_stop_would_cut_fails_the_check(tmp_path):
    """A parsed answer that writes one row name over and over (the stop's own test) counts as wrongly stopped."""
    path = _write(tmp_path, [_record(answer=_forms(400, same=True))])
    assert check(path)["good_answers_stopped"] == 1
    assert main(["--answers", path]) == 1
