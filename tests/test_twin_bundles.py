"""A SPEC_21 delivery turned into bundles: two renders per twin, one corrected gold
(data_pipeline.ingestion.prepare_bundles.prepare_twin_bundles)."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from data_pipeline.ingestion.prepare_bundles import (
    corrected_gold,
    delivery_line,
    prepare_bundles,
    prepare_twin_bundles,
    read_recodes,
)

pymupdf = pytest.importorskip("pymupdf")

EXAMPLE = Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")


def _pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    doc.new_page(width=612, height=792).insert_text((72, 72), text)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def _delivery(tmp_path: Path, golds: dict[str, dict], line_folder: str = "homeowners", lob: str = "homeowners"):
    """``golds``: twin number -> gold. One seed, every twin in train."""
    root = tmp_path / "delivery"
    rows = []
    for number, gold in golds.items():
        twin = f"{lob}__all_state__seed_a__twin_{number:03d}"
        digital = f"train/{line_folder}/Digital/{twin}.pdf"
        scanned = f"train/{line_folder}/Scanned/{twin}__scan.pdf"
        gold_path = f"train/{line_folder}/Gold_json/{twin}.json"
        _pdf(root / digital, "Allstate eBill: enrolled. Declarations page for the homeowners policy.")
        _pdf(root / scanned, "scan")
        (root / gold_path).parent.mkdir(parents=True, exist_ok=True)
        (root / gold_path).write_text(json.dumps(gold), encoding="utf-8")
        rows.append({"split": "train", "lob": lob, "carrier": "all_state", "seed": "seed_a", "twin": twin,
                     "digital": digital, "scanned": scanned, "gold": gold_path})
    with (root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _gold(**changes):
    gold = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            gold.pop(key, None)
        else:
            gold[key] = value
    return gold


def test_each_twin_becomes_a_digital_and_a_scanned_document_reading_one_gold(tmp_path):
    root = _delivery(tmp_path, {1: _gold(), 2: _gold()})
    out = tmp_path / "bundles"
    report = prepare_bundles(root, out)                                 # finds split_manifest.csv itself
    assert report.written == {("train", "synthetic"): 4}
    folder = out / "homeowners__all_state__seed_a__twin_002__scan"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    # The example gold's values are on no page of these one-line PDFs: a page whose
    # text misses most of what its label cites settles nothing, so none is listed.
    assert meta.pop("unprinted_values") == []
    assert meta == {"lob": "homeowners", "synthetic": True, "template_id": "homeowners/seed_a", "split": "train",
                    "carrier": "all_state", "source_system": "fideon_synth", "sample": 2, "render_mode": "scanned",
                    # The manifest records no review of the seed, so its twins have none.
                    "reviewer_id": None, "review_date": None, "draft_backend": None,
                    "review_inherited_from": None}
    assert report.unreviewed == 4 and "no review record: 4 document(s)" in report.describe()
    digital = json.loads((out / "homeowners__all_state__seed_a__twin_002" / "metadata.json").read_text("utf-8"))
    assert digital["render_mode"] == "digital" and digital["template_id"] == meta["template_id"]


def test_the_misspelt_motorcycle_line_is_read_as_motorcycle():
    assert delivery_line("motorcyle") == "motorcycle"
    assert delivery_line("classic_auto") == "personal_auto"


def test_a_coverage_code_in_no_list_is_recoded_as_the_changelogs_say():
    gold = _gold()
    gold["coverages"][1]["coverage_code"] = "HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS"
    fixed, notes, problem = corrected_gold(gold, "homeowners", carrier="example", text_pdf=Path("none.pdf"),
                                           recodes=read_recodes())
    assert problem is None and fixed["coverages"][1]["coverage_code"] == "X_MED_PAY"
    assert ("coverage code", "coverages[1] HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS -> X_MED_PAY") in notes
    assert gold["coverages"][1]["coverage_code"] == "HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS"   # input untouched


def test_a_premium_written_twice_keeps_its_coverage_home(tmp_path):
    """The example writes $1,210 and $190 in coverages[].premium and in premium.items."""
    gold = _gold()
    gold["premium"]["items"].append({**copy.deepcopy(gold["premium"]["items"][0]), "unit_type": "location",
                                     "coverage_code": None})
    gold["premium"]["items"][-1].pop("coverage_code")
    fixed, notes, problem = corrected_gold(gold, "homeowners", carrier="example", text_pdf=Path("none.pdf"),
                                           recodes={})
    assert problem is None
    assert [i.get("unit_type") for i in fixed["premium"]["items"]] == ["location"]   # the non-coverage item stays
    assert sum(1 for kind, _ in notes if kind == "premium written twice") == 2


def test_a_premium_item_that_totals_the_per_vehicle_rows_is_dropped_one_the_rows_lack_stays():
    gold = _gold()
    first = gold["coverages"][0]
    gold["coverages"] = [copy.deepcopy(first), copy.deepcopy(first)]    # two HO_COV_A rows, 1,210 each
    item = copy.deepcopy(gold["premium"]["items"][0])
    item["amount"] = {**item["amount"], "raw": "$2,420", "parsed": 2420.0}
    lone = copy.deepcopy(gold["premium"]["items"][0])
    lone["coverage_code"] = "HO_SPECIAL_LIMITS"                          # no coverage row: its only home
    gold["premium"]["items"] = [item, lone]
    fixed, notes, _problem = corrected_gold(gold, "homeowners", carrier="x", text_pdf=Path("none.pdf"), recodes={})
    assert [i["coverage_code"] for i in fixed["premium"]["items"]] == ["HO_SPECIAL_LIMITS"]


def test_a_missing_carrier_is_filled_as_the_document_prints_it(tmp_path):
    root = _delivery(tmp_path, {1: _gold(carrier=None)})
    out = tmp_path / "bundles"
    report = prepare_twin_bundles(root, out)
    gold = json.loads((out / "homeowners__all_state__seed_a__twin_001" / "golden.json").read_text("utf-8"))
    assert gold["carrier"]["name"]["raw"] == "Allstate" and gold["carrier"]["name"]["page_ref"] == [1]
    assert report.corrections["carrier missing"] == 1
    rows = list(csv.DictReader((out / "corrections.csv").open(encoding="utf-8")))
    assert {r["correction"] for r in rows} >= {"carrier missing", "premium written twice"}


def test_a_gold_still_outside_its_schema_is_left_out_with_the_reason(tmp_path):
    root = _delivery(tmp_path, {1: _gold(), 2: _gold(coverages=None)})
    report = prepare_twin_bundles(root, tmp_path / "bundles")
    assert report.written == {("train", "synthetic"): 2}
    assert any("'coverages' is a required property" in reason for reason in report.skipped)
    assert not (tmp_path / "bundles" / "homeowners__all_state__seed_a__twin_002").exists()


def test_a_dry_run_reports_everything_and_writes_nothing(tmp_path):
    root = _delivery(tmp_path, {1: _gold(carrier=None)})
    out = tmp_path / "bundles"
    report = prepare_twin_bundles(root, out, dry_run=True)
    assert report.written == {("train", "synthetic"): 2} and report.corrections["carrier missing"] == 1
    assert not out.exists() and "dry run" in report.describe()


def test_a_twin_whose_two_renders_are_the_same_file_is_one_scanned_document(tmp_path):
    """A scanned seed's twins are delivered as the same scan in both folders."""
    import shutil

    root = _delivery(tmp_path, {1: _gold()})
    row = next(csv.DictReader((root / "split_manifest.csv").open(encoding="utf-8")))
    shutil.copyfile(root / row["scanned"], root / row["digital"])
    out = tmp_path / "bundles"
    (out / Path(row["digital"]).stem).mkdir(parents=True)                # left by an earlier run
    report = prepare_twin_bundles(root, out)
    assert report.written == {("train", "synthetic"): 1} and report.corrections["renders identical"] == 1
    assert not (out / Path(row["digital"]).stem).exists()
    meta = json.loads((out / Path(row["scanned"]).stem / "metadata.json").read_text(encoding="utf-8"))
    assert meta["render_mode"] == "scanned"


def _originals(tmp_path: Path, gold: dict, seed: str = "seed_a") -> Path:
    root = tmp_path / "source data"
    _pdf(root / "pdfs" / "homeowners" / "All State" / f"{seed}.pdf",
         "Allstate ePolicy: enrolled. Declarations page for the homeowners policy.")
    path = root / "gold json" / "homeowners" / "All State" / f"{seed}.gold.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gold), encoding="utf-8")
    return root


def test_each_seed_is_a_real_document_in_its_twins_family_and_split(tmp_path):
    root = _delivery(tmp_path, {1: _gold()})
    originals = _originals(tmp_path, _gold(carrier=None))
    out = tmp_path / "bundles"
    report = prepare_twin_bundles(root, out, originals=originals)
    assert report.written == {("train", "synthetic"): 2, ("train", "real"): 1}
    folder = out / "homeowners__all_state__seed_a__original"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    twin = json.loads((out / "homeowners__all_state__seed_a__twin_001" / "metadata.json").read_text("utf-8"))
    assert meta["synthetic"] is False and meta["split"] == "train" and meta["template_id"] == twin["template_id"]
    gold = json.loads((folder / "golden.json").read_text(encoding="utf-8"))
    assert gold["carrier"]["name"]["raw"] == "Allstate"                   # the same corrections as its twins


def _with_review(root: Path, *records: dict) -> None:
    """Give the SPEC_21 manifest's rows (in order) review columns: the SEED's review."""
    path = root / "split_manifest.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row, record in zip(rows, records, strict=True):
        row.update(record)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*rows[0]])
        writer.writeheader()
        writer.writerows(rows)


def test_a_spec21_seed_and_its_twins_carry_the_seeds_review_record(tmp_path):
    from data_pipeline.labeling.export_golden_labels import has_review_record

    root = _delivery(tmp_path, {1: _gold(), 2: _gold()})
    record = {"reviewer_id": "a.reviewer", "review_date": "2026-10-20", "draft_backend": "label_studio"}
    _with_review(root, record, record)
    out = tmp_path / "bundles"
    report = prepare_twin_bundles(root, out, originals=_originals(tmp_path, _gold()))
    assert report.unreviewed == 0
    seed = json.loads((out / ORIGINAL / "metadata.json").read_text(encoding="utf-8"))
    assert {k: seed[k] for k in record} == record and has_review_record(seed)
    for name in ("homeowners__all_state__seed_a__twin_001", "homeowners__all_state__seed_a__twin_002__scan"):
        twin = json.loads((out / name / "metadata.json").read_text(encoding="utf-8"))
        assert {k: twin[k] for k in record} == record and twin["review_inherited_from"] == "homeowners/seed_a"
        assert has_review_record(twin)


def test_a_seeds_twins_stating_two_review_records_are_refused(tmp_path):
    from data_pipeline.ingestion.prepare_bundles import BundleError

    root = _delivery(tmp_path, {1: _gold(), 2: _gold()})
    _with_review(root, {"reviewer_id": "a.reviewer", "review_date": "2026-10-20"},
                 {"reviewer_id": "b.reviewer", "review_date": "2026-10-20"})
    with pytest.raises(BundleError, match="two review records"):
        prepare_twin_bundles(root, tmp_path / "bundles")


def test_a_seed_without_its_document_is_counted_not_guessed(tmp_path):
    root = _delivery(tmp_path, {1: _gold()})
    originals = _originals(tmp_path, _gold(), seed="another_seed")
    report = prepare_twin_bundles(root, tmp_path / "bundles", originals=originals)
    assert report.skipped["seed document or its gold not in the originals"] == 1
    assert ("train", "real") not in report.written


TWIN_1 = "homeowners__all_state__seed_a__twin_001"
ORIGINAL = "homeowners__all_state__seed_a__original"


def _corrections(monkeypatch, marker):
    """``corrected_gold`` changing nothing (``marker`` None), or adding ``marker``."""
    from data_pipeline.ingestion import prepare_bundles as module

    def corrected(gold, lob, **_kwargs):
        if marker is None:
            return gold, [], None
        return {**gold, "x_marker": marker}, [("test", "changed")], None

    monkeypatch.setattr(module, "corrected_gold", corrected)


def test_a_gold_corrected_on_a_rerun_is_a_new_file_and_the_delivered_gold_is_untouched(tmp_path, monkeypatch):
    """The first run links the uncorrected golds; writing the corrected one in
    place went through the link into the delivery."""
    root = _delivery(tmp_path, {1: _gold()})
    originals = _originals(tmp_path, _gold())
    out = tmp_path / "bundles"
    _corrections(monkeypatch, None)
    prepare_twin_bundles(root, out, originals=originals)
    delivered = [root / "train" / "homeowners" / "Gold_json" / f"{TWIN_1}.json",
                 originals / "gold json" / "homeowners" / "All State" / "seed_a.gold.json"]
    before = [path.read_bytes() for path in delivered]
    _corrections(monkeypatch, 1)
    prepare_twin_bundles(root, out, originals=originals)
    assert [path.read_bytes() for path in delivered] == before
    for name in (TWIN_1, f"{TWIN_1}__scan", ORIGINAL):
        assert json.loads((out / name / "golden.json").read_text(encoding="utf-8"))["x_marker"] == 1


def test_a_rerun_writes_new_files_and_a_batch_linked_to_the_bundles_keeps_its_own(tmp_path, monkeypatch):
    """A batch drawn from the bundles (the smoke batch) is hard links to their
    files: rewriting the metadata in place changed the batch's metadata - its
    printed-nowhere paths - under the batch's old gold."""
    import os

    from data_pipeline.ingestion import prepare_bundles as module
    from data_pipeline.ingestion.label_rules import PRINTED_NOWHERE

    root = _delivery(tmp_path, {1: _gold()})
    out = tmp_path / "bundles"
    _corrections(monkeypatch, None)
    prepare_twin_bundles(root, out)
    batch = tmp_path / "smoke" / TWIN_1
    batch.mkdir(parents=True)
    for name in ("golden.json", "metadata.json"):
        os.link(out / TWIN_1 / name, batch / name)
    before = {name: (batch / name).read_bytes() for name in ("golden.json", "metadata.json")}
    monkeypatch.setattr(module, "corrected_gold", lambda gold, lob, **_: (
        {**gold, "x_marker": 1}, [(PRINTED_NOWHERE, "policy.policy_number")], None))
    prepare_twin_bundles(root, out)
    meta = json.loads((out / TWIN_1 / "metadata.json").read_text(encoding="utf-8"))
    assert meta["unprinted_values"] == ["policy.policy_number"]
    assert {name: (batch / name).read_bytes() for name in before} == before


def test_move_mode_gives_both_renders_their_shared_gold(tmp_path, monkeypatch):
    root = _delivery(tmp_path, {1: _gold()})
    _corrections(monkeypatch, None)
    out = tmp_path / "bundles"
    report = prepare_twin_bundles(root, out, mode="move")
    assert report.written == {("train", "synthetic"): 2}
    for name in (TWIN_1, f"{TWIN_1}__scan"):
        assert (out / name / "golden.json").is_file() and (out / name / "document.pdf").is_file()
    assert not (root / "train" / "homeowners" / "Gold_json" / f"{TWIN_1}.json").exists()   # moved once


def test_a_rerun_removes_what_an_earlier_run_bundled_and_this_one_leaves_out(tmp_path):
    root = _delivery(tmp_path, {1: _gold(), 2: _gold()})
    out = tmp_path / "bundles"
    prepare_twin_bundles(root, out, originals=_originals(tmp_path, _gold()))
    (out / "another_delivery__document").mkdir()
    twin_2 = root / "train" / "homeowners" / "Gold_json" / "homeowners__all_state__seed_a__twin_002.json"
    twin_2.write_text(json.dumps(_gold(coverages=None)), encoding="utf-8")       # now outside its schema

    dry = prepare_twin_bundles(root, out, dry_run=True)                          # and no originals
    assert dry.removed == 3 and (out / ORIGINAL).is_dir()
    report = prepare_twin_bundles(root, out)
    assert report.removed == 3
    assert {p.name for p in out.iterdir() if p.is_dir()} == {TWIN_1, f"{TWIN_1}__scan", "another_delivery__document"}


def test_corrections_csv_is_rewritten_by_a_run_with_nothing_to_correct(tmp_path, monkeypatch):
    root = _delivery(tmp_path, {1: _gold()})
    out = tmp_path / "bundles"
    prepare_twin_bundles(root, out)                       # the example writes its premiums twice
    assert len((out / "corrections.csv").read_text(encoding="utf-8").splitlines()) > 1
    _corrections(monkeypatch, None)
    prepare_twin_bundles(root, out)
    assert (out / "corrections.csv").read_text(encoding="utf-8").splitlines() == ["twin,correction,detail"]


def test_a_dry_run_of_an_older_delivery_writes_nothing(tmp_path):
    root = tmp_path / "older"
    _pdf(root / "Train" / "pdfs" / "a.pdf", "a policy")
    (root / "Train" / "gold json").mkdir(parents=True)
    (root / "Train" / "gold json" / "a.json").write_text(json.dumps(_gold()), encoding="utf-8")
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "lob", "source", "kind", "pdf", "gold", "ok"])
        writer.writerow(["Train", "homeowners", "src/a.pdf", "synthetic", "Train/pdfs/a.pdf",
                         "Train/gold json/a.json", "true"])
    out = tmp_path / "bundles"
    report = prepare_bundles(root, out, dry_run=True)
    assert report.written == {("train", "synthetic"): 1} and not out.exists()
    assert (root / "Train" / "pdfs" / "a.pdf").is_file()
