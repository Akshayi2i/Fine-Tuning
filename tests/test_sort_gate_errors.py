"""scripts/sort_gate_errors.py and scripts/show_eval_set.py: diagnosis read from stored records only."""

from __future__ import annotations

import json

from scripts.show_eval_set import describe, form_counts, render
from scripts.sort_gate_errors import (
    EDITION_TWIN,
    GOLD_ONLY,
    MODEL_ONLY,
    PAIRED,
    edition_twin,
    main,
    sort_records,
    summarise,
)


def _r(path, cls, expected=None, got=None, source="p1", mode="image_only"):
    return {"source_id": source, "modality_mode": mode, "field_path": path, "error_class": cls,
            "expected": expected, "got": got}


GOLDS = {"p1": {"forms_and_endorsements": [
    {"form_number": {"raw": "SSI SF 00 03 0123", "parsed": "SSI SF 00 03 0123", "page_ref": [8]}},
    {"form_number": {"raw": "CG 00 01", "parsed": "CG 00 01", "page_ref": [2]}},
    {"form_number": {"raw": "IL 00 17", "parsed": "IL 00 17", "page_ref": [3]}}]}}


def test_an_edition_twin_is_the_same_number_with_digits_added_or_taken_away():
    assert edition_twin("SSI SF 00 03", ["SSI SF 00 03 0123"]) == "SSI SF 00 03 0123"
    assert edition_twin("SSPJ-07", ["SSPJ-07-2021"]) == "SSPJ-07-2021"
    assert edition_twin("SS Privacy", ["SS Privacy 10 21"]) == "SS Privacy 10 21"
    assert edition_twin("SSI SF 00 03 0123", ["SSI SF 00 03"]) == "SSI SF 00 03"
    assert edition_twin("0025321173P1", ["SSI SF 00 03 0123"]) is None
    assert edition_twin("CG 00 01", ["CG 00 02 04 13"]) is None          # another form, not an edition
    assert edition_twin("SSI SF 00 03", ["SSI SF 00 03 ABCD"]) is None   # letters are not an edition


def test_rows_are_sorted_by_what_their_form_number_record_shows():
    records = [
        # model row 3: the same form as the gold's first, written without its edition
        _r("forms_and_endorsements[3].form_number", "invented", got="SSI SF 00 03"),
        _r("forms_and_endorsements[3].title", "invented", got="Signature Page"),
        # model row 4: a page code the gold holds nowhere
        _r("forms_and_endorsements[4].form_number", "invented", got="0025321173P1"),
        # gold row put after the answer's: the SSI form nothing paired with, its twin is in the model's rows
        _r("forms_and_endorsements[5].form_number", "omission", expected="SSI SF 00 03 0123"),
        _r("forms_and_endorsements[5].title", "omission", expected="Signature"),
        # gold row nothing paired with, no twin
        _r("forms_and_endorsements[6].form_number", "omission", expected="IL 00 17"),
        # a paired row with a wrong title
        _r("forms_and_endorsements[1].title", "schema_reasoning", expected="CGL Form", got="Other"),
        # outside forms
        _r("coverages[0].coverage_name", "invented", got="MED PAY"),
        _r("policy.policy_number", "omission", expected="X1"),
    ]
    causes = [r["cause"] for r in sort_records(records, GOLDS)]
    assert causes == [EDITION_TWIN, EDITION_TWIN, MODEL_ONLY, EDITION_TWIN, EDITION_TWIN, GOLD_ONLY, PAIRED,
                      "coverages: invented", "outside lists: omission"]


def test_rows_of_different_readings_are_kept_apart():
    records = [_r("forms_and_endorsements[3].form_number", "invented", got="SSI SF 00 03", mode="image_only"),
               _r("forms_and_endorsements[3].form_number", "schema_reasoning", expected="CG 00 01", got="CG 00 02",
                  mode="ocr_plus_image")]
    assert [r["cause"] for r in sort_records(records, GOLDS)] == [EDITION_TWIN, PAIRED]


def test_the_summary_and_the_command_line(tmp_path, capsys):
    records = [_r("forms_and_endorsements[3].form_number", "invented", got="SSI SF 00 03"),
               _r("policy.policy_number", "omission", expected="X1")]
    summary = summarise(sort_records(records, GOLDS))
    assert summary["by_cause"] == {EDITION_TWIN: 1, "outside lists: omission": 1}
    (tmp_path / "r.json").write_text(json.dumps({"x": {"error_records": records}}), encoding="utf-8")
    (tmp_path / "g.json").write_text(json.dumps(GOLDS), encoding="utf-8")
    assert main(["--file", str(tmp_path / "r.json"), "--golds", str(tmp_path / "g.json")]) == 0
    assert "2 failing field(s), sorted by cause" in capsys.readouterr().out


def test_the_eval_set_line_counts_forms_and_editions_as_stored():
    gold = {"forms_and_endorsements": [
        {"form_number": {"raw": "CG 00 01"}, "edition_date": {"raw": "04 13"}},
        {"form_number": {"raw": "SSI SF 00 03 0123"}, "edition_date": None},
        {"form_number": None}]}
    assert form_counts(gold) == (2, 1)
    row = describe("policy_0065", {"lob": "gl", "carrier": "Sutton", "held_out_carrier": True, "is_scanned": True},
                   gold, 73, {"original_filename": "sutto_national_cgl_policy.pdf"})
    assert row["gold_forms"] == 2 and row["names"] == {"original_filename": "sutto_national_cgl_policy.pdf"}
    text = render([row])
    assert "policy_0065" in text and "Sutton" in text and "sutto_national_cgl_policy.pdf" in text
