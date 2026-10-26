"""Training targets and corpus rows (review 2026-10-26, items 1, 2, 6, 33, 38).

* 1 - a twin's rule-added values are checked once, on the digital render's
  clean text, with serving's figure-aware matcher; a value no page prints is
  listed (its windows left out), never deleted.
* 2 - on a common-model line a value citing no page is given every page that
  prints it; one no page prints costs its window, not the document; the audit
  reports it per gold and blocks a real gold only when it can be placed nowhere.
* 6 - validation policies under the fast profile are read in the two modes
  selection reads; every row says whether it is synthetic, and every val and
  test row whether its carrier was trained on.
* 33 - the manifest records rows, empty answers and values per answer, per
  line, group and mode, and what building the targets did.
* 38 - a window whose answer overruns its reservation is built as the halves
  serving retries it in, not set aside with its document.

Synthetic fixtures only.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from common.schemas import CANONICAL_ROOT
from data_pipeline.dataset_builder import build_jsonl
from data_pipeline.dataset_builder.build_jsonl import (
    SourceDocument,
    build_corpus,
    expand_document,
    share_twin_text,
    train_rows_by_epoch,
)
from data_pipeline.dataset_builder.label_verification import VerificationReport, verified_label
from data_pipeline.dataset_builder.policy_windows import window_halves, with_placed_pages
from data_pipeline.dataset_builder.split_groups import GroupSplitAssignment

HOME = json.loads((CANONICAL_ROOT / "common schema" / "examples" / "homeowners_minimal.json")
                  .read_text(encoding="utf-8"))


def _env(raw, pages):
    return {"raw": raw, "parsed": raw, "page_ref": list(pages)}


def _page_one(gold) -> str:
    """A page 1 that prints every value the gold cites on page 1."""
    from data_pipeline.audit import iter_envelopes

    return "\n".join(str(env["raw"]) for _, env in iter_envelopes(gold)
                     if env.get("raw") not in (None, "") and 1 in (env.get("page_ref") or []))


def _home(**changes):
    gold = copy.deepcopy(HOME)
    for path, value in changes.items():
        section, name = path.split(".")
        gold[section][name] = value
    return gold


def _document(source_id="h1", gold=None, pages=None, **fields):
    gold = gold if gold is not None else copy.deepcopy(HOME)
    pages = pages if pages is not None else [_page_one(gold), "Policy conditions.", "Endorsements."]
    return SourceDocument(source_id=source_id, doc_type="policy", golden_label=gold, ocr_pages=list(pages),
                          image_paths=[f"{source_id}/p{n}.png" for n in range(1, len(pages) + 1)],
                          lob="homeowners", tenant_id="default", **fields)


def _answers(rows) -> str:
    return " ".join(row["messages"][-1]["content"] for row in rows)


# --------------------------------------------------------------------------
# 1 - twins checked once, on the clean text, listed not deleted
# --------------------------------------------------------------------------


def _filled(label, *paths):
    label["fideon:filled"] = {"paths": list(paths)}
    return label


def test_an_amount_printed_without_cents_is_found_by_the_figure_aware_check():
    label = _filled({"premium": {"total": _env("$250,000.00", [1])}}, "premium.total")
    pages = ["Total premium $250,000", "Conditions."]
    words = VerificationReport()
    assert "total" not in verified_label(label, pages, words)["premium"]        # the gate's reading, unchanged
    listed = VerificationReport()
    assert verified_label(label, pages, listed, listed=True) is label
    assert (listed.kept, listed.unprinted) == (1, [])


def test_a_value_no_page_prints_is_listed_and_left_in_the_label():
    label = _filled({"carrier": {"name": _env("Granite State Insurance", [1])}}, "carrier.name")
    report = VerificationReport()
    out = verified_label(label, ["Declarations of Northfield Mutual"], report, listed=True)
    assert out["carrier"]["name"]["raw"] == "Granite State Insurance"
    assert report.unprinted == ["carrier.name"] and report.dropped == []


def test_a_value_printed_only_elsewhere_cites_where_it_is_printed():
    label = _filled({"policy": {"policy_number": _env("HX-0000-0001", [1])},
                     "coverages": [{"premium": _env("$1,210.00", [1])}]},
                    "policy.policy_number", "coverages[0].premium")
    pages = ["Declarations", "Policy HX-0000-0001 premium $1,210.00", "Policy HX-0000-0001 premium $1,210.00"]
    report = VerificationReport()
    out = verified_label(label, pages, report, listed=True)
    # A document-level value: every page that prints it (the value-once convention).
    assert out["policy"]["policy_number"]["page_ref"] == [2, 3]
    # A table row's figure on several other pages may be another row's: listed.
    assert report.repaged == ["policy.policy_number"] and report.unprinted == ["coverages[0].premium"]


def _twin(render, pages, gold):
    return _document(f"t1_{render}", gold=copy.deepcopy(gold), pages=pages, synthetic=True, twin_index=1,
                     render_mode=render, group_id="seed1")


def test_a_twin_is_checked_once_on_its_digital_text_and_both_renders_teach_it():
    gold = _filled(copy.deepcopy(HOME), "premium.total")              # "$1,425.00", page 1
    digital_text = _page_one(gold).replace("$1,425.00", "$1,425")
    scanned_text = digital_text.replace("$1,425", "$1,428")             # a digit misread: another number
    digital = _twin("digital", [digital_text, "Conditions.", "Endorsements."], gold)
    scanned = _twin("scanned", [scanned_text, "Conditions.", "Endorsements."], gold)

    alone = copy.deepcopy(scanned)
    verified_label(alone.golden_label, alone.ocr_pages, (own := VerificationReport()), listed=True)
    assert own.unprinted == ["premium.total"]                         # its own OCR would hold it out

    assert share_twin_text([digital, scanned]) == 1
    assert scanned.clean_ocr_pages == digital.ocr_pages
    counts = Counter()
    rows = {doc.render_mode: expand_document(doc, "train", modes=("ocr_plus_image",), counts=counts)[0]
            for doc in (digital, scanned)}
    for render_rows in rows.values():
        assert "1425" in _answers(render_rows).replace(",", "")
    # The scanned row still carries its own text: only its target is the twin's.
    assert any("$1,428" in json.dumps(row["messages"]) for row in rows["scanned"])
    assert counts["verify.kept"] == 2 and counts["verify.twin_renders_shared"] == 1


# --------------------------------------------------------------------------
# 2 - a value citing no page is placed; one no page prints costs a window
# --------------------------------------------------------------------------


def test_values_citing_no_page_are_given_every_page_that_prints_them():
    label = {
        "document": {"page_count": _env("3", [])},                    # the system's: never placed
        "policy": {"policy_number": _env("HX-0000-0001", []), "rating_state": _env("NY", []),
                   "prior_policy_number": _env("HX-9999-9999", []), "term_months": _env("12 Months", [])},
        "billing": {"is_payor": {"raw": "Yes", "parsed": True, "page_ref": []}},
        "additional_fields": [{"label": "Fire district", "value": _env("Troy FD", [])},
                              {"label": "Own page", "page_ref": [3], "value": _env("Elsewhere", [])}],
    }
    pages = ["Policy HX-0000-0001 State NY", "Fire district Troy FD", "Header HX-0000-0001"]
    out, placement = with_placed_pages(label, pages, skip=["policy.term_months"])
    assert out["policy"]["policy_number"]["page_ref"] == [1, 3]
    assert out["policy"]["rating_state"]["page_ref"] == [1]
    assert out["additional_fields"][0]["value"]["page_ref"] == [2]
    assert placement.not_printed == ["policy.prior_policy_number"]
    assert placement.unchecked == ["billing.is_payor"]
    assert "document.page_count" not in placement.placed and "policy.term_months" not in placement.placed
    assert out["additional_fields"][1]["value"]["page_ref"] == []      # the entry cites its own page
    assert label["policy"]["policy_number"]["page_ref"] == []          # a copy


def test_a_value_no_page_prints_costs_its_window_not_the_document():
    gold = _home(**{"policy.prior_policy_number": _env("HX-9999-9999", [])})
    pages = [_page_one(gold), "Policy conditions.", "Endorsements."]
    counts = Counter()
    rows, details = expand_document(_document(gold=gold, pages=pages), "train",
                                    modes=("ocr_plus_image",), counts=counts)
    assert rows, "the document is kept"
    assert counts["unpaged.windows_dropped"] >= 1 and counts["unpaged.not_printed"] == 1
    assert any("citing no page" in d and "policy.prior_policy_number" in d for d in details)
    assert "HX-9999-9999" not in _answers(rows)


def test_a_value_citing_no_page_is_taught_on_the_pages_that_print_it():
    gold = _home(**{"policy.policy_number": _env("HX-0000-0001", [])})
    pages = [_page_one(gold).replace("HX-0000-0001", ""), "Policy HX-0000-0001 conditions.", "Endorsements."]
    counts = Counter()
    rows, _ = expand_document(_document(gold=gold, pages=pages), "train", modes=("ocr_plus_image",),
                              counts=counts)
    assert counts["unpaged.placed_one_page"] == 1 and not counts["unpaged.windows_dropped"]
    taught = [json.loads(row["messages"][-1]["content"]).get("policy", {}).get("policy_number")
              for row in rows]
    assert [t["p"] for t in taught if t] == [[2]]


def test_build_corpus_keeps_a_common_model_document_with_an_unplaceable_value():
    gold = _home(**{"policy.prior_policy_number": _env("HX-9999-9999", [])})
    document = _document(gold=gold)
    result = build_corpus([document], GroupSplitAssignment(assignment={document.family: "train"}))
    assert not result.set_aside and result.rows_by_split["train"]
    assert result.target_counts["unpaged.documents_saved"] == 1


def _audit_folder(tmp_path: Path, name: str, gold: dict, lines: list[str], *, synthetic=False) -> Path:
    pdf_module = pytest.importorskip("pymupdf")
    folder = tmp_path / name
    folder.mkdir(parents=True)
    doc = pdf_module.open()
    page = doc.new_page(width=612, height=792)
    for i, line in enumerate(lines):
        page.insert_text((50, 60 + 16 * i), line, fontsize=9)
    doc.save(folder / "policy.pdf")
    (folder / "golden.json").write_text(json.dumps(gold), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({
        "lob": "homeowners", "synthetic": synthetic, "reviewer_id": "a.reviewer",
        "review_date": "2026-10-20"}), encoding="utf-8")
    return folder


def _no_page_ref(report, folder):
    return [(f.severity, f.detail) for f in report.findings if f.folder == folder and f.check == "no_page_ref"]


def test_the_audit_reports_values_citing_no_page_and_blocks_a_real_gold_it_cannot_place(tmp_path):
    from data_pipeline.audit import audit_folder

    lines = ["Homeowners Policy Declarations", "Policy HX-0000-0001 Effective 10/01/2026"]
    placeable = {"policy": {"policy_number": _env("HX-0000-0001", []), "effective_date": _env("10/01/2026", [1])}}
    nowhere = {"policy": {"policy_number": _env("HX-9999-9999", []), "effective_date": _env("10/01/2026", [1])}}
    _audit_folder(tmp_path, "placeable", placeable, lines)
    _audit_folder(tmp_path, "real", nowhere, lines)
    _audit_folder(tmp_path, "twin", nowhere, lines, synthetic=True)
    report = audit_folder(tmp_path)

    assert [s for s, _ in _no_page_ref(report, "placeable")] == ["warning"]
    assert sorted(s for s, _ in _no_page_ref(report, "real")) == ["blocker", "warning"]
    assert [s for s, _ in _no_page_ref(report, "twin")] == ["warning"]
    values = {(v.folder, v.path): v.result for v in report.values}
    assert values[("placeable", "policy.policy_number")] == "no_page_ref"
    assert values[("real", "policy.policy_number")] == "not_found"


# --------------------------------------------------------------------------
# 6 - validation rows under fast; synthetic and carrier_seen_in_train
# --------------------------------------------------------------------------


def _split_corpus():
    train = _document("tr1", carrier="example mutual")
    twin = _document("tr2", carrier="example mutual", synthetic=True, twin_index=1, render_mode="digital")
    val = _document("va1", carrier="Example Mutual")
    test = _document("te1", carrier="other carrier")
    assignment = GroupSplitAssignment(assignment={
        train.family: "train", twin.family: "train", val.family: "val", test.family: "test"})
    return build_corpus([train, twin, val, test], assignment)


def _modes(result, split):
    return {row["modality_mode"] for row in result.rows_by_split[split]}


def test_validation_policies_are_read_in_the_modes_selection_reads_under_fast():
    result = _split_corpus()
    assert _modes(result, "val") == {"ocr_plus_image", "image_only"}
    assert _modes(result, "test") == {"ocr_plus_image", "noisy_ocr_image", "image_only"}


@pytest.mark.format_profile("standard")
def test_validation_keeps_every_mode_under_standard():
    assert _modes(_split_corpus(), "val") == {"ocr_plus_image", "noisy_ocr_image", "image_only"}


def test_every_row_is_marked_synthetic_and_eval_rows_whether_their_carrier_trained():
    result = _split_corpus()
    assert all(isinstance(row["synthetic"], bool) for row in result.all_rows)
    assert {row["synthetic"] for row in result.rows_by_split["train"]} == {True, False}
    assert {row["carrier_seen_in_train"] for row in result.rows_by_split["val"]} == {True}
    assert {row["carrier_seen_in_train"] for row in result.rows_by_split["test"]} == {False}
    assert not any("carrier_seen_in_train" in row for row in result.rows_by_split["train"])


# --------------------------------------------------------------------------
# 33 - what the corpus teaches, per line, group and mode
# --------------------------------------------------------------------------


def test_the_manifest_counts_rows_empty_answers_and_values_per_mode():
    from data_pipeline.corpus_manifest import _target_counts, target_stats

    def row(mode, answer, sections="decl"):
        return {"lob": ["homeowners"], "sections": sections, "modality_mode": mode, "doc_type": "policy",
                "messages": [{"role": "assistant", "content": json.dumps(answer)}]}

    rows = {"train": [row("image_only", {"policy": {"policy_number": {"v": "HX-1", "p": [1]},
                                                    "rating_state": {"r": "N.Y.", "v": "NY", "p": [1]}}}),
                      row("image_only", {"policy": {}}),
                      row("ocr_plus_image", {"policy": {"policy_number": {"raw": "HX-1", "parsed": "HX-1",
                                                                          "page_ref": [1]}}})]}
    stats = target_stats(rows)["train"]["homeowners"]["decl"]
    assert stats["image_only"] == {"rows": 2, "empty_rows": 1, "mean_values": 1.0}
    assert stats["ocr_plus_image"] == {"rows": 1, "empty_rows": 0, "mean_values": 1.0}
    counts = _target_counts({"verify.kept": 3})
    assert counts["verify.kept"] == 3 and set(build_jsonl.TARGET_COUNTS) <= set(counts)


def test_build_counts_reach_the_manifest():
    from data_pipeline.corpus_manifest import build_manifest

    document = _document(gold=_home(**{"policy.prior_policy_number": _env("HX-9999-9999", [])}))
    result = build_corpus([document], GroupSplitAssignment(assignment={document.family: "train"}))
    manifest, _ = build_manifest(
        corpus_version="v1", tenant_id="default", rows_by_split=result.rows_by_split,
        golden_labels_by_source={document.source_id: document.golden_label}, provenance_by_source={},
        split_assignment={}, ocr_environment={}, doc_types=["policy"],
        lob_by_source={document.source_id: "homeowners"}, target_counts=result.target_counts)
    assert manifest["target_counts"]["unpaged.not_printed"] == 1
    assert "homeowners" in manifest["target_stats"]["train"]


# --------------------------------------------------------------------------
# 38 - an answer over its reservation is read in halves, as serving retries it
# --------------------------------------------------------------------------


def test_window_halves_are_serving_s_halves():
    from data_pipeline.dataset_builder.policy_windows import PolicyWindowPlan
    from serving.pipeline import _halves, _Planned

    plan = PolicyWindowPlan("decl", 2, (4, 5, 6, 7, 8), single=False, leads=True, homed=True)
    for declarations in (None, 4, 7):
        ours = [(list(half.pages), half.leads) for half in window_halves(plan, declarations)]
        theirs = [(piece.pages, piece.leads) for piece in _halves(_Planned("decl", [4, 5, 6, 7, 8], True),
                                                                   declarations)]
        assert ours == theirs
    assert {half.window_index for half in window_halves(plan, None)} == {2}


def _two_page_home():
    gold = copy.deepcopy(HOME)
    gold["premium"]["total"]["page_ref"] = [2]
    first = _page_one(gold)
    return gold, [first, f"Premium summary. Total {gold['premium']['total']['raw']}"]


def _overflow_when_two_pages_cited(task, answer, doc_type=None):
    return len(set(re.findall(r'"p": \[(\d+)', answer))) > 1


def test_an_answer_over_its_reservation_is_built_as_two_halves(monkeypatch):
    from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows

    monkeypatch.setattr(build_jsonl, "answer_overflows", _overflow_when_two_pages_cited)
    gold, pages = _two_page_home()
    document = _document(gold=gold, pages=pages)
    planned_before = plan_windows("homeowners", *pages_to_read(pages, 2), pages)
    counts = Counter()
    rows, details = expand_document(document, "train", modes=("ocr_plus_image",), counts=counts)
    halves = [row for row in rows if row.get("window_half")]
    assert halves and counts["halves.windows_split"] >= 1 and counts["halves.rows"] == len(halves)
    assert {tuple(row["window_pages"]) for row in halves} == {(1,), (2,)}
    assert any("read as two halves" in d for d in details)
    # The declarations lead the half holding their page, as serving's halves do.
    led = [row for row in halves if row["window_leads"]]
    assert led and all(row["window_pages"] == [1] for row in led)
    # The plan is serving's, unchanged: the halves are how its window is retried.
    assert plan_windows("homeowners", *pages_to_read(pages, 2), pages) == planned_before


def test_a_document_with_an_overflowing_window_is_kept(monkeypatch):
    monkeypatch.setattr(build_jsonl, "answer_overflows", _overflow_when_two_pages_cited)
    gold, pages = _two_page_home()
    document = _document(gold=gold, pages=pages)
    result = build_corpus([document], GroupSplitAssignment(assignment={document.family: "train"}))
    assert not result.set_aside
    assert result.cap_report.windows_halved >= 1 and result.target_counts["halves.documents_saved"] == 1
    by_epoch = train_rows_by_epoch(result)                               # no row twice
    assert all(by_epoch.values())
