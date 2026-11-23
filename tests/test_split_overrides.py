"""One insured's real documents share a split: a manifest.csv delivery that puts them in two is refused before
anything is written, and the owner's split overrides move a source document with its copies
(data_pipeline.ingestion.prepare_bundles; scripts.split_leaks proposes the rows). Cycle 2's delivery put
U.S. Underwriters' BOP property policies in test and train, sharing a policy number (2026-11-18)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from data_pipeline.ingestion.prepare_bundles import BundleError, prepare_bundles, read_split_overrides

COLUMNS = ["split", "lob", "source", "kind", "pdf", "gold", "ok", "carrier", "sample", "reviewer_id", "review_date"]
#: A gold the schema accepts (a manifest delivery's golds are corrected and validated), given another insured below.
EXAMPLE = Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")


def _cell(value: str) -> dict:
    return {"raw": value, "parsed": value, "confidence": {"score": 1.0, "source": "deterministic"}, "page_ref": [1],
            "flagged": False}


def _gold(name: str, number: str) -> dict:
    """The example gold as another insured's: its name and policy number, and no mailing address (the example's
    would make every gold one insured)."""
    gold = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    gold["named_insured"] = {"primary_name": _cell(name)}
    gold["policy"]["policy_number"] = _cell(number)
    gold["policy"].pop("prior_policy_number", None)
    return gold


def _delivery(tmp_path: Path, docs: list[tuple[str, str, dict, int]]) -> Path:
    """``docs``: (split, source, gold, copies) - each a real document and its copies in its split."""
    root = tmp_path / "delivery"
    rows = []
    for split, source, gold, copies in docs:
        for i in range(copies + 1):
            name = source if i == 0 else f"{source}__twin_{i:02d}"
            pdf, gold_rel = f"pdf/{name}.pdf", f"golds/{name}.gold.json"
            (root / "pdf").mkdir(parents=True, exist_ok=True)
            (root / "golds").mkdir(parents=True, exist_ok=True)
            (root / pdf).write_bytes(b"%PDF-1.7")
            body = gold if i == 0 else _gold(f"Copy {i} of {source}", f"COPY{i:04d}{len(source)}")
            (root / gold_rel).write_text(json.dumps(body), encoding="utf-8")
            rows.append({"split": split, "lob": "homeowners", "source": source, "kind": "real" if i == 0 else "synthetic",
                         "pdf": pdf, "gold": gold_rel, "ok": "true", "carrier": "", "sample": "",
                         "reviewer_id": "R", "review_date": "2026-11-23"})
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return root


LEAKY = [
    ("train", "acme_gl_policy_1", _gold("Acme Bakery LLC", "NBP3550236G"), 2),
    ("val", "acme_gl_policy_2", _gold("ACME BAKERY, L.L.C.", "XX0000001"), 2),   # the same insured, by name
    ("train", "other_gl_policy", _gold("Zenith Tools Inc", "ZT1234567"), 1),
]


def _splits(out: Path) -> dict[str, str]:
    return {folder.name: json.loads((folder / "metadata.json").read_text("utf-8"))["split"]
            for folder in out.iterdir() if folder.is_dir()}


@pytest.mark.parametrize("dry_run", [False, True])
def test_one_insured_in_two_splits_is_refused_before_anything_is_written(tmp_path, dry_run):
    out = tmp_path / "bundles"
    with pytest.raises(BundleError, match=r"1 insured\(s\).*acme_gl_policy_1.*acme_gl_policy_2.*split_overrides"):
        prepare_bundles(_delivery(tmp_path, LEAKY), out, dry_run=dry_run)
    assert not out.exists()


def test_documents_sharing_only_a_policy_number_are_one_insured(tmp_path):
    docs = [("train", "a_policy", _gold("North Bakery", "NBP3550236G"), 0),
            ("test", "b_policy", _gold("South Bakery", "NBP-355-0236G"), 0)]
    with pytest.raises(BundleError, match=r"test: \['b_policy'\], train: \['a_policy'\]"):
        prepare_bundles(_delivery(tmp_path, docs), tmp_path / "bundles")


def test_an_override_moves_the_source_and_its_copies(tmp_path):
    out = tmp_path / "bundles"
    report = prepare_bundles(_delivery(tmp_path, LEAKY), out, split_overrides={"acme_gl_policy_2": "train"})
    splits = _splits(out)
    assert splits["acme_gl_policy_2"] == "train"
    assert splits["acme_gl_policy_2__twin_01"] == splits["acme_gl_policy_2__twin_02"] == "train"
    assert report.moved == {"val -> train": 3}
    assert "split overridden: val -> train (3)" in report.describe()


def test_an_override_naming_no_document_is_refused(tmp_path):
    with pytest.raises(BundleError, match="no usable manifest row"):
        prepare_bundles(_delivery(tmp_path, LEAKY), tmp_path / "bundles",
                        split_overrides={"acme_gl_policy_2": "train", "acme_gl_policy_9": "train"})


def test_an_excluded_leftover_is_removed_only_when_nothing_is_refused(tmp_path):
    out = tmp_path / "bundles"
    (out / "other_gl_policy").mkdir(parents=True)                  # an earlier run's folder, now excluded
    delivery = _delivery(tmp_path, LEAKY)
    with pytest.raises(BundleError):
        prepare_bundles(delivery, out, exclusions={"other_gl_policy": "audit"})
    assert (out / "other_gl_policy").is_dir()                      # refused: nothing touched
    report = prepare_bundles(delivery, out, exclusions={"other_gl_policy": "audit"},
                             split_overrides={"acme_gl_policy_2": "train"})
    assert report.removed == 1 and not (out / "other_gl_policy").exists()


def test_read_split_overrides(tmp_path):
    path = tmp_path / "overrides.csv"
    assert read_split_overrides(path) == {}
    path.write_text("source,split,reason\nAcme/gl/a.pdf,Test,same insured\nb,train,\n,,\n", encoding="utf-8")
    assert read_split_overrides(path) == {"Acme/gl/a": "test", "b": "train"}
    path.write_text("source,split,reason\nb,holdout,x\n", encoding="utf-8")
    with pytest.raises(BundleError, match="not train, val or test"):
        read_split_overrides(path)
    path.write_text("source,split,reason\nb,train,x\nb,val,y\n", encoding="utf-8")
    with pytest.raises(BundleError, match="both train and val"):
        read_split_overrides(path)
    path.write_text("document,split\nb,train\n", encoding="utf-8")
    with pytest.raises(BundleError, match="source,split,reason"):
        read_split_overrides(path)


def test_split_leaks_proposes_rows_that_prepare_bundles_accepts(tmp_path, capsys):
    from scripts import split_leaks

    docs = [*LEAKY, ("test", "acme_gl_policy_3", _gold("Acme Bakery", "AB7777777"), 1)]
    delivery = _delivery(tmp_path, docs)
    proposed = tmp_path / "proposed.csv"
    assert split_leaks.main(["--input", str(delivery), "--overrides", str(tmp_path / "none.csv"),
                             "--write", str(proposed)]) == 1
    printed = capsys.readouterr().out
    assert "1 insured(s) in more than one split" in printed
    assert "2 move(s) proposed: 2 to test" in printed             # a group touching test goes to test
    assert read_split_overrides(proposed) == {"acme_gl_policy_1": "test", "acme_gl_policy_2": "test"}
    # Never over an existing file.
    assert split_leaks.main(["--input", str(delivery), "--overrides", str(tmp_path / "none.csv"),
                             "--write", str(proposed)]) == 1
    # With the rows agreed, nothing crosses and the bundles are written.
    assert split_leaks.main(["--input", str(delivery), "--overrides", str(proposed)]) == 0
    prepare_bundles(delivery, tmp_path / "bundles", split_overrides=read_split_overrides(proposed))
    assert _splits(tmp_path / "bundles")["acme_gl_policy_1__twin_02"] == "test"


def test_split_leaks_tie_goes_to_train():
    from scripts.split_leaks import proposed_split

    assert proposed_split({"train": ["a"], "val": ["b"]}) == "train"
    assert proposed_split({"train": ["a"], "val": ["b", "c"]}) == "val"
    assert proposed_split({"train": ["a", "b"], "test": ["c"]}) == "test"
