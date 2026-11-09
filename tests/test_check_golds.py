"""scripts/check_golds.py: every gold checked before training - what it misses, what no page prints, which rules it
breaks. Read-only (2026-11-10)."""

from __future__ import annotations

import copy
import json

from scripts.check_golds import (
    BREAKS_RULE,
    NOT_PRINTED,
    PRINTED_NOT_IN_GOLD,
    check_gold,
    main,
)
from scripts.missed_form_check import candidates


def _env(value, *pages):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


AMOUNTS = [1000 * n for n in range(1, 11)]
PAGES = {
    1: "DECLARATIONS\nPolicy Number BP-1001\n" + "\n".join(f"Limit {n} ${amount:,}" for n, amount in enumerate(AMOUNTS, start=1))
       + "\nForms: CG 00 01 04 13",
    2: "CG 21 06 05 14\nEXCLUSION - ACCESS OR DISCLOSURE OF CONFIDENTIAL INFORMATION\nwording wording wording\n"
       "Page 1 of 1",
    3: "PREMIUM CREDIT APPLICATION\nApplicant information\nHired Auto Liability",
}


def _gold():
    return {
        "policy": {"policy_number": _env("BP-1001", 1)},
        "coverages": [{"coverage_name": _env(f"Limit {n}", 1), "coverage_code": None,
                       "limits": [{"limit_type": "other", "amount": _env(float(a), 1)}]}
                      for n, a in enumerate(AMOUNTS, start=1)]
        + [{"coverage_name": _env("Each Occurrence Lim it", 1), "coverage_code": None,
            "limits": [{"limit_type": "per_occurrence", "amount": _env(999999.0, 1)}]},
           {"coverage_name": _env("Hired Auto Liability", 3), "coverage_code": None}],
        "forms_and_endorsements": [{"form_number": _env("CG 00 01 04 13", 1)}],
    }


def _findings(**kwargs):
    gold = _gold()
    before = copy.deepcopy(gold)
    found = check_gold(gold, PAGES, lob="gl", **kwargs)
    assert gold == before                                          # read-only: the gold is not changed
    return found


def test_a_form_attached_to_the_policy_that_the_gold_does_not_list_is_found():
    found = [f for f in _findings() if f["check"] == PRINTED_NOT_IN_GOLD]
    assert [f["detail"] for f in found] == ["CG 21 06 05 14"]


def test_a_gold_figure_no_page_prints_is_found_with_its_value():
    found = [f for f in _findings() if f["check"] == NOT_PRINTED]
    assert [(f["where"], f["detail"]) for f in found] == [("coverages[10].limits[0].amount", "gold value '999999.0'")]


def test_the_labelling_rules_a_gold_breaks_are_named():
    rules = {f["rule"] for f in _findings(name_repairs={"Each Occurrence Lim it": "Each Occurrence Limit"})
             if f["check"] == BREAKS_RULE}
    assert "rule 1: the edition is still inside the form number" in rules
    assert "rule 3: a coverage name is broken inside a word" in rules
    assert "rule 5: a form or coverage is read only from application pages" in rules


def test_a_line_the_rules_do_not_hold_on_gets_only_the_form_search():
    found = check_gold(_gold(), PAGES, lob="commercial_package")
    assert {f["check"] for f in found} == {PRINTED_NOT_IN_GOLD}


def test_addresses_and_phone_numbers_are_not_form_numbers():
    text = "CHESTER NY 10918-1059\nNORTH CAROLINA 27534\nFAX 570-823-2059\nCG 21 06 05 14"
    assert candidates(text) == ["CG 21 06 05 14"]


def test_the_command_checks_a_folder_and_writes_every_finding(tmp_path, capsys, monkeypatch):
    from data_pipeline.ingestion import label_rules

    folder = tmp_path / "gl__hybrid__carrier_policy"
    folder.mkdir()
    (folder / "golden.json").write_text(json.dumps(_gold()), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"lob": "gl"}), encoding="utf-8")
    monkeypatch.setattr(label_rules, "page_texts", lambda _pdf: dict(PAGES))
    monkeypatch.setattr(label_rules, "scanned_pages", lambda _pdf: set())
    out = tmp_path / "findings.csv"
    assert main(["--input", str(tmp_path), "--csv", str(out), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["golds"] == 1 and summary["by_check"][PRINTED_NOT_IN_GOLD] == 1
    assert summary["by_check"][NOT_PRINTED] == 1
    assert out.read_text(encoding="utf-8").splitlines()[0] == "document,check,rule,where,detail"
