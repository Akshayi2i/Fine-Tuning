"""A manifest.csv delivery's golds are corrected as the other deliveries' are (prepare_bundles.corrected_gold): cycle 2's
delivery (data/cycle2_combined) linked every gold as delivered, so none of the label rules or the cycle 3 rules ran on
it (found 2026-11-23). And a PDF library that cannot load stops the step instead of reading every PDF as text-less."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from data_pipeline.ingestion import label_rules
from data_pipeline.ingestion.prepare_bundles import prepare_bundles

EXAMPLE = Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")


def _delivery(tmp_path: Path, golds: dict[str, dict]) -> Path:
    root = tmp_path / "delivery"
    (root / "pdf").mkdir(parents=True)
    (root / "golds").mkdir()
    rows = []
    for name, gold in golds.items():
        (root / "pdf" / f"{name}.pdf").write_bytes(b"%PDF-1.7")
        (root / "golds" / f"{name}.gold.json").write_text(json.dumps(gold), encoding="utf-8")
        rows.append({"split": "train", "lob": "homeowners", "source": name, "kind": "real", "pdf": f"pdf/{name}.pdf",
                     "gold": f"golds/{name}.gold.json", "ok": "true", "carrier": "", "sample": "",
                     "reviewer_id": "R", "review_date": "2026-11-23"})
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _with_edition_in_form_refs() -> dict:
    gold = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    gold["coverages"][0]["form_refs"] = ["HO 00 03 05 11"]
    return gold


def test_a_manifest_gold_is_corrected_and_the_delivery_left_as_it_was(tmp_path):
    delivery = _delivery(tmp_path, {"ho_policy": _with_edition_in_form_refs()})
    delivered = (delivery / "golds" / "ho_policy.gold.json").read_text(encoding="utf-8")
    out = tmp_path / "bundles"
    report = prepare_bundles(delivery, out)
    written = json.loads((out / "ho_policy" / "golden.json").read_text(encoding="utf-8"))
    assert written["coverages"][0]["form_refs"] == ["HO 00 03"]                 # R2: the number, without edition
    assert (delivery / "golds" / "ho_policy.gold.json").read_text(encoding="utf-8") == delivered
    assert not (out / "ho_policy" / "golden.json").samefile(delivery / "golds" / "ho_policy.gold.json")
    assert report.corrections["form reference edition removed"] == 1
    rows = list(csv.DictReader((out / "corrections.csv").open(encoding="utf-8")))
    assert ("ho_policy", "form reference edition removed") in {(r["document"], r["correction"]) for r in rows}
    assert json.loads((out / "ho_policy" / "metadata.json").read_text("utf-8"))["unprinted_values"] == []


def test_an_uncorrected_gold_is_linked_as_before(tmp_path):
    gold = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    gold["premium"].pop("items", None)            # the example writes two coverages' premiums twice
    delivery = _delivery(tmp_path, {"ho_policy": gold})
    out = tmp_path / "bundles"
    report = prepare_bundles(delivery, out)
    assert not report.corrections
    assert report.linked + report.copied == 2                                 # the PDF and the gold, placed


def test_a_gold_outside_its_schema_is_left_out_and_named(tmp_path):
    broken = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    del broken["carrier"]
    out = tmp_path / "bundles"
    report = prepare_bundles(_delivery(tmp_path, {"good": json.loads(EXAMPLE.read_text(encoding="utf-8")),
                                                  "broken": broken}), out)
    assert report.written == {("train", "real"): 1}
    assert not (out / "broken").exists()
    assert any("carrier" in reason for reason in report.skipped)
    rows = list(csv.DictReader((out / "corrections.csv").open(encoding="utf-8")))
    assert [r["correction"] for r in rows if r["document"] == "broken"] == ["left out"]


def test_a_pdf_library_that_cannot_load_stops_the_step(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError("DLL load failed while importing _mupdf: An Application Control policy has blocked "
                              "this file.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    with pytest.raises(label_rules.PdfReaderMissing, match="Application Control"):
        label_rules.page_texts(pdf)
    with pytest.raises(label_rules.PdfReaderMissing):
        label_rules.scanned_pages(pdf)
    assert label_rules.page_texts(tmp_path / "missing.pdf") == {}           # no file: nothing to read, no error
