"""A delivery split upstream is used as given: the synthetic-data route.

The generator puts each source document and its ten synthetic twins in one of
Train/Val/Test, so a test layout is one the model never trained on. The corpus
build takes that split (option A), refuses a source in two splits, lets synthetic
twins into val and test only then, and the gate reports real documents apart.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from data_pipeline.dataset_builder.build_jsonl import build_corpus
from data_pipeline.dataset_builder.split_groups import (
    GroupRecord,
    SplitError,
    assign_delivered_splits,
    assign_group_splits,
)
from tests.test_dataset_builder import _documents

# --------------------------------------------------------------------------
# The assignment
# --------------------------------------------------------------------------


def _groups(splits: dict[str, str], line="homeowners"):
    return {"policy": [GroupRecord(group_id=g, doc_type="policy", source_ids=[f"{g}-{i}" for i in range(11)],
                                   line=line, synthetic=True) for g in splits]}


def test_the_delivered_split_is_taken_as_given():
    splits = {"a": "train", "b": "train", "c": "val", "d": "val", "e": "test"}
    result = assign_delivered_splits(_groups(splits), splits)
    assert result.delivered
    assert result.assignment == splits
    assert set(result.val_half) == {"c", "d"}                     # validation still halved
    assert result.counts_by_doc_type["policy"] == {"train": 2, "val": 2, "test": 1}
    assert result.ratios_by_doc_type["policy"]["test"] == pytest.approx(0.2)
    assert result.as_dict()["delivered"] is True


@pytest.mark.parametrize("splits,message", [
    ({"a": "train", "b": "test"}, "no val"),
    ({"a": "train", "b": "val"}, "no test"),
    ({"a": "train", "b": "val", "c": "holdout"}, "no delivered split"),
])
def test_an_unusable_delivered_split_is_refused(splits, message):
    with pytest.raises(SplitError, match=message):
        assign_delivered_splits(_groups(splits), splits)


def test_once_frozen_a_delivery_takes_no_new_test_documents():
    splits = {"a": "train", "b": "val", "c": "test"}
    with pytest.raises(SplitError, match="frozen"):
        assign_delivered_splits(_groups(splits), splits, with_test=False)
    assert assign_delivered_splits(_groups({"a": "train", "b": "val"}), {"a": "train", "b": "val"},
                                   with_test=False).assignment == {"a": "train", "b": "val"}


# --------------------------------------------------------------------------
# Families: the source document, and all or none
# --------------------------------------------------------------------------


def _loaded(pairs):
    docs = _documents(len(pairs))
    for doc, (split, template) in zip(docs, pairs, strict=True):
        doc.delivered_split, doc.template_id = split, template
    return docs


def test_each_document_joins_its_source_family():
    from orchestration.pipeline_dag import delivered_split_of, use_delivered_families

    docs = _loaded([("train", "c/ho/s1"), ("train", "c/ho/s1"), ("test", "c/ho/s2")])
    assert use_delivered_families(docs)
    assert {d.family for d in docs} == {"delivered:c/ho/s1", "delivered:c/ho/s2"}
    assert delivered_split_of(docs) == {"delivered:c/ho/s1": "train", "delivered:c/ho/s2": "test"}


def test_a_source_in_two_splits_is_refused():
    from orchestration.pipeline_dag import PipelineError, delivered_split_of, use_delivered_families

    docs = _loaded([("train", "c/ho/s1"), ("test", "c/ho/s1")])
    use_delivered_families(docs)
    with pytest.raises(PipelineError, match="both train and test"):
        delivered_split_of(docs)


def test_all_or_none_of_the_documents_carry_a_split():
    from orchestration.pipeline_dag import PipelineError, use_delivered_families

    docs = _loaded([("train", "s1"), (None, "s2")])
    with pytest.raises(PipelineError, match="or none"):
        use_delivered_families(docs)
    assert not use_delivered_families(_loaded([(None, "s1"), (None, "s2")]))


def test_text_similar_families_across_the_split_are_reported_not_merged(caplog):
    from orchestration.pipeline_dag import use_delivered_families

    docs = _loaded([("train", "s1"), ("test", "s2")])
    for doc in docs:
        doc.group_id = "same-printed-form"             # what text similarity would have found
    with caplog.at_level("WARNING"):
        use_delivered_families(docs)
    assert "1 val/test document(s) share a printed form" in caplog.text
    assert docs[0].family != docs[1].family


# --------------------------------------------------------------------------
# Synthetic documents in val and test: only for a delivered split
# --------------------------------------------------------------------------


def test_synthetic_twins_may_be_evaluated_only_under_a_delivered_split():
    docs = _documents(12)
    for doc in docs:
        doc.synthetic = True
    splits = {d.source_id: ("train" if i < 8 else "val" if i < 10 else "test") for i, d in enumerate(docs)}
    groups = {"lossrun": [GroupRecord(group_id=d.source_id, doc_type="lossrun", source_ids=[d.source_id],
                                      synthetic=True) for d in docs]}
    built = build_corpus(docs, assign_delivered_splits(groups, splits))
    assert built.rows_by_split["test"] and all(r["synthetic"] for r in built.rows_by_split["test"])

    drawn = assign_group_splits(groups, hold_out_carriers=False)
    drawn.assignment.update(splits)                     # force synthetic into test on a drawn split
    with pytest.raises(SplitError, match="synthetic"):
        build_corpus(docs, drawn)


# --------------------------------------------------------------------------
# Import, and the real-only score
# --------------------------------------------------------------------------


def test_the_importer_refuses_an_unknown_split():
    from data_pipeline.ingestion.import_labeled_pdfs import LabeledPdf, check_bundle

    golden = json.loads((Path(__file__).parent / "fixtures/golden/policy_0001.golden.json").read_text("utf-8"))
    bundle = LabeledPdf(directory=Path("x"), pdf=Path("x.pdf"), golden=golden,
                        metadata={"lob": "workers_comp", "split": "holdout"})
    assert any("not train, val or test" in e for e in check_bundle(bundle, "policy").errors)


def test_the_gate_report_scores_real_documents_apart(monkeypatch):
    from evaluation import golden_eval

    class Report:
        def __init__(self, n):
            self.n = n

        def as_dict(self):
            return {"gate_metrics": {"documents": self.n}}

    monkeypatch.setattr("evaluation.run_eval.build_report", lambda version, triples, **k: Report(len(triples)))
    triples = [({}, {}, {"source_id": "r1", "synthetic": False}),
               ({}, {}, {"source_id": "s1", "synthetic": True}),
               ({}, {}, {"source_id": "s2", "synthetic": True})]
    section = golden_eval.real_only_section("v1", triples)
    assert section["composition"] == {"real_documents": 1, "synthetic_documents": 2}
    assert section["real_only"] == {"gate_metrics": {"documents": 1}}
    assert golden_eval.real_only_section("v1", triples[:1]) == {}        # all real: nothing extra


def test_the_frozen_set_and_the_eval_carry_the_synthetic_flag():
    root = Path(__file__).resolve().parent.parent
    assert '"synthetic": bool(metadata.get("synthetic", False))' in (root / "evaluation/freeze_eval_set.py").read_text("utf-8")
    assert '"synthetic": doc.synthetic' in (root / "evaluation/golden_eval.py").read_text("utf-8")


# --------------------------------------------------------------------------
# prepare_bundles: the generator's layout into import bundles
# --------------------------------------------------------------------------


def _delivery(tmp_path: Path) -> Path:
    root = tmp_path / "delivery"
    rows = []

    def add(split, lob, source, sample, kind, ok="True", write=True):
        tail = "original" if kind == "original" else f"synth_{sample:03d}"
        name = f"{lob}__{Path(source).stem}__{tail}"
        pdf, gold = f"{split}/pdfs/{name}.pdf", f"{split}/gold json/{name}.json"
        if write:
            for rel, body in ((pdf, b"%PDF-1.7"), (gold, b"{}")):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(body)
        rows.append({"split": split, "lob": lob, "carrier": "Acme", "source": source, "sample": sample,
                     "kind": kind, "pdf": pdf, "gold": gold, "pages": 3, "fields": 9, "ok": ok,
                     "seconds": 1, "problems": ""})

    add("Train", "homeowners", "Acme/homeowners/ho_1.pdf", 0, "original")
    add("Train", "homeowners", "Acme/homeowners/ho_1.pdf", 1, "synthetic")
    add("Test", "homeowners", "Acme/homeowners/ho_2.pdf", 1, "synthetic")
    add("Train", "flood", "Acme/flood/fl_1.pdf", 1, "synthetic", write=False)       # outside --lines, files gone
    add("Val", "homeowners", "Acme/homeowners/ho_3.pdf", 1, "synthetic", ok="False")  # flagged
    add("Val", "homeowners", "Acme/homeowners/ho_4.pdf", 1, "synthetic", write=False)  # missing files
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_prepare_bundles_writes_one_folder_per_document(tmp_path):
    from common.schemas import SNAPSHOT_LINES
    from data_pipeline.ingestion.prepare_bundles import prepare_bundles

    out = tmp_path / "bundles"
    report = prepare_bundles(_delivery(tmp_path), out, lines=frozenset(SNAPSHOT_LINES))
    assert report.documents == 3
    assert report.written == {("train", "real"): 1, ("train", "synthetic"): 1, ("test", "synthetic"): 1}
    assert report.skipped_lines == {"flood": 1}
    assert "outside --lines: flood 1" in report.describe()
    assert report.skipped == {"flagged by the generator (ok is false)": 1,
                              "files listed in the manifest are missing": 1}
    folder = out / "homeowners__ho_1__synth_001"
    assert sorted(p.name for p in folder.iterdir()) == ["document.pdf", "golden.json", "metadata.json"]
    meta = json.loads((folder / "metadata.json").read_text("utf-8"))
    assert meta == {"lob": "homeowners", "synthetic": True, "template_id": "Acme/homeowners/ho_1",
                    "split": "train", "carrier": "Acme", "source_system": "fideon_synth", "sample": 1,
                    "render_mode": None,
                    # Its seed's own row records no review, so the twin has none.
                    "reviewer_id": None, "review_date": None, "draft_backend": None,
                    "review_inherited_from": None}
    original = json.loads((out / "homeowners__ho_1__original/metadata.json").read_text("utf-8"))
    assert original["synthetic"] is False and original["template_id"] == meta["template_id"]


def test_a_twin_inherits_its_seeds_review_record_and_never_its_own(tmp_path):
    """Decision 2026-10-22: a twin counts as reviewed only when its seed was."""
    import csv as _csv

    from data_pipeline.ingestion.prepare_bundles import prepare_bundles
    from data_pipeline.labeling.export_golden_labels import has_review_record

    delivery = _delivery(tmp_path)
    with (delivery / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    for row in rows:
        if row["kind"] == "original":            # ho_1's seed: reviewed
            row.update(reviewer_id="a.reviewer", review_date="2026-10-20", draft_backend="label_studio")
        elif "ho_2" in row["source"]:           # ho_2's twin names a reviewer, its seed none
            row.update(reviewer_id="someone", review_date="2026-10-20", draft_backend="")
    with (delivery / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[*rows[0], "reviewer_id", "review_date", "draft_backend"])
        writer.writeheader()
        writer.writerows({"reviewer_id": "", "review_date": "", "draft_backend": "", **row} for row in rows)

    out = tmp_path / "bundles"
    report = prepare_bundles(delivery, out, lines=frozenset({"homeowners"}))

    def meta(name):
        return json.loads((out / name / "metadata.json").read_text("utf-8"))

    seed, twin, orphan = (meta(n) for n in ("homeowners__ho_1__original", "homeowners__ho_1__synth_001",
                                            "homeowners__ho_2__synth_001"))
    assert (seed["reviewer_id"], seed["review_date"], seed["draft_backend"]) == (
        "a.reviewer", "2026-10-20", "label_studio")
    assert {k: twin[k] for k in ("reviewer_id", "review_date", "review_inherited_from")} == {
        "reviewer_id": "a.reviewer", "review_date": "2026-10-20", "review_inherited_from": "Acme/homeowners/ho_1"}
    assert has_review_record(seed) and has_review_record(twin)
    assert orphan["reviewer_id"] is None and orphan["review_inherited_from"] is None
    assert not has_review_record(orphan) and report.unreviewed == 1


def test_prepare_bundles_is_safe_to_rerun_and_copies_when_asked(tmp_path):
    from data_pipeline.ingestion.prepare_bundles import prepare_bundles

    delivery, out = _delivery(tmp_path), tmp_path / "bundles"
    prepare_bundles(delivery, out, lines=frozenset({"homeowners"}), mode="copy")
    again = prepare_bundles(delivery, out, lines=frozenset({"homeowners"}), mode="copy")
    assert again.documents == 3 and again.copied == 0            # nothing re-copied
    assert (delivery / "Train/pdfs/homeowners__ho_1__original.pdf").is_file()   # the delivery is untouched


def test_the_audit_blocks_a_source_split_across_train_and_test():
    from data_pipeline.audit import DocumentAudit, delivered_split_findings

    docs = [DocumentAudit(folder="a", split="train", template_id="s1"),
            DocumentAudit(folder="b", split="test", template_id="s1"),
            DocumentAudit(folder="c", split="val", template_id="s2")]
    found = delivered_split_findings(docs)
    assert {f.folder for f in found} == {"a", "b"} and all(f.severity == "blocker" for f in found)
    partial = delivered_split_findings([DocumentAudit(folder="a", split="train"), DocumentAudit(folder="b")])
    assert [f.folder for f in partial] == ["b"]


def test_bundles_are_never_committed():
    ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "data/bundles/" in ignore


def test_excluded_documents_stay_out_and_an_earlier_folder_is_removed(tmp_path):
    from data_pipeline.ingestion.prepare_bundles import prepare_bundles, read_exclusions

    delivery, out = _delivery(tmp_path), tmp_path / "bundles"
    prepare_bundles(delivery, out, lines=frozenset({"homeowners"}), mode="copy")
    assert (out / "homeowners__ho_1__original").is_dir()
    exclusions_file = tmp_path / "exclusions.csv"
    exclusions_file.write_text("document,reason\nhomeowners__ho_1__original,duplicate\n", encoding="utf-8")
    report = prepare_bundles(delivery, out, lines=frozenset({"homeowners"}), mode="copy",
                             exclusions=read_exclusions(exclusions_file))
    assert report.skipped["excluded: duplicate"] == 1
    assert not (out / "homeowners__ho_1__original").exists()
    assert read_exclusions(tmp_path / "missing.csv") == {}


@pytest.mark.parametrize("parsed,flagged", [("05-17", False), ("5/2017", False), ("sometime", True)])
def test_a_month_and_year_is_not_a_date_error(parsed, flagged):
    from data_pipeline.audit import AuditReport, _check_formats

    report = AuditReport(root=".")
    _check_formats("doc", [("forms_and_endorsements[0].edition_date", {"raw": parsed, "parsed": parsed})], report)
    assert bool(report.formats) is flagged


def test_an_indicator_and_a_signed_return_are_not_amount_errors():
    from data_pipeline.audit import AuditReport, _check_formats

    report = AuditReport(root=".")
    _check_formats("doc", [
        ("document_type_detail.policy_change.additional_or_return_premium", {"raw": "Return", "parsed": "Return"}),
        ("document_type_detail.policy_change.net_change_amount", {"raw": "$109.00", "parsed": -109.0}),
    ], report)
    assert report.formats == []
    _check_formats("doc", [("premium.total_policy_premium", {"raw": "$47,250.00", "parsed": 4725.0})], report)
    assert [f.check for f in report.formats] == ["amount"]


def test_a_line_override_relabels_a_carriers_documents(tmp_path):
    from data_pipeline.ingestion.prepare_bundles import BundleError, prepare_bundles, read_lob_overrides

    overrides = tmp_path / "overrides.csv"
    overrides.write_text("source_prefix,lob,reason\nAcme/homeowners/ho_1,dwelling_fire,misfiled\n", encoding="utf-8")
    out = tmp_path / "bundles"
    report = prepare_bundles(_delivery(tmp_path), out, lines=frozenset({"homeowners", "dwelling_fire"}),
                             mode="copy", lob_overrides=read_lob_overrides(overrides))
    assert report.relined == {"homeowners -> dwelling_fire": 2}
    assert json.loads((out / "homeowners__ho_1__original/metadata.json").read_text("utf-8"))["lob"] == "dwelling_fire"
    assert json.loads((out / "homeowners__ho_2__synth_001/metadata.json").read_text("utf-8"))["lob"] == "homeowners"
    overrides.write_text("source_prefix,lob,reason\nAcme/,pet_insurance,x\n", encoding="utf-8")
    with pytest.raises(BundleError, match="canonical schema"):
        read_lob_overrides(overrides)
