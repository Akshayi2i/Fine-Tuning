"""The post-OCR value check, and the smoke run's selection and commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ocr_check import OcrCheckReport, check_document, run_check, verdict, write_report


def env(raw, page=1):
    return {"raw": raw, "parsed": raw, "confidence": {"score": 1.0}, "page_ref": [page], "flagged": False}


LABEL = {
    "named_insured": {"primary_name": env("Jane Fake")},
    "policy": {"policy_number": env("HO-999", page=2)},
    "homeowners": {"coverage_a": env("$300,000"), "coverage_b": env("$30,000")},
    "premium": {"total": env("$1,234.00", page=2)},
    "forms": [{"form_number": env("HO 00 03")}],
    "fideon:filled": {"paths": ["homeowners.coverage_b", "premium.total", "forms[0].form_number"]},
}
PAGES = ["DECLARATIONS\nNamed insured: JANE FAKE\nCoverage A $300,000\nPolicy HO-999",
         "Premium total $1,234.00\n| HO 00 03 | Homeowners 3 |"]


# --------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------


def test_each_value_is_judged_against_the_ocr_text_of_its_page():
    rows = {r.path: r for r in check_document("d1", LABEL, PAGES, kind="synthetic", line="homeowners")}
    assert rows["named_insured.primary_name"].result == "ok"
    assert rows["policy.policy_number"].result == "wrong_page"          # printed on page 1, cites page 2
    assert rows["homeowners.coverage_b"].result == "not_found"          # the fill-in added a value not there
    assert rows["premium.total"].result == "ok"
    assert rows["forms[0].form_number"].result == "wrong_page"          # cites page 1, printed on page 2


def test_added_fields_are_kept_apart_from_the_generators_own():
    rows = check_document("d1", LABEL, PAGES, kind="synthetic", line="homeowners")
    origin = {r.path: r.origin for r in rows}
    assert origin["homeowners.coverage_b"] == "added" and origin["named_insured.primary_name"] == "original"


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def _report(original_ok, original_n, added_ok, added_n):
    from data_pipeline.ocr_check import Row

    rows = [Row("d", f"o{i}", "x", "1", "ok" if i < original_ok else "not_found", "", "original", "synthetic", "h")
            for i in range(original_n)]
    rows += [Row("d", f"a{i}", "x", "1", "ok" if i < added_ok else "not_found", "", "added", "synthetic", "h")
             for i in range(added_n)]
    return OcrCheckReport(rows=rows, documents=1)


def test_the_gate_passes_when_added_fields_hold_up_like_the_original_ones():
    passed, reason = verdict(_report(90, 100, 88, 100))
    assert passed and "gap 2.0%" in reason


def test_the_gate_fails_when_added_fields_are_not_on_the_page():
    passed, reason = verdict(_report(90, 100, 60, 100))
    assert not passed and "Fix the labels before training" in reason


# --------------------------------------------------------------------------
# From Blob, end to end
# --------------------------------------------------------------------------


def test_the_check_reads_labels_and_ocr_from_blob_and_writes_its_report(tmp_path):
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    for sid, synthetic in (("s1", True), ("r1", False)):
        client.write_json(paths.golden_label("policy", sid, "smoke"), LABEL)
        client.write_json(paths.label_metadata("policy", sid, "smoke"), {"lob": "homeowners", "synthetic": synthetic})
        client.write_json(paths.ocr_meta("policy", sid, "smoke"), {"page_count": 2})
        for page, text in enumerate(PAGES, 1):
            client.write_text(paths.processed_page("policy", sid, page, "md", "smoke"), text)
    client.write_json(paths.golden_label("policy", "pending", "smoke"), LABEL)   # not OCR'd yet
    client.write_json(paths.label_metadata("policy", "pending", "smoke"), {"lob": "homeowners"})
    report = run_check(client, "policy", tenant_id="smoke")
    assert report.documents == 2 and report.skipped == {"not OCR'd yet": 1}
    summary = write_report(report, tmp_path)
    assert set(summary["by_kind"]) == {"real", "synthetic"}
    assert {"added", "original", "real documents"} == set(summary["by_origin_synthetic"])
    for name in ("summary.json", "value_checks.csv", "report.md"):
        assert (tmp_path / name).is_file()


# --------------------------------------------------------------------------
# The smoke run
# --------------------------------------------------------------------------


def _bundles(root: Path):
    specs = [("train", "homeowners", "A/homeowners/h1"), ("train", "homeowners", "A/homeowners/h2"),
             ("train", "personal_auto", "B/personal_auto/p1"), ("train", "motorcycle", "C/motorcycle/m1"),
             ("train", "dwelling_fire", "D/dwelling_fire/d1"), ("val", "homeowners", "A/homeowners/h3"),
             ("test", "personal_auto", "B/personal_auto/p2")]
    for split, lob, template in specs:
        for k in range(3):
            folder = root / f"{lob}__{template.rsplit('/', 1)[-1]}__synth_{k:03d}"
            folder.mkdir(parents=True)
            (folder / "metadata.json").write_text(json.dumps(
                {"split": split, "lob": lob, "template_id": template, "synthetic": k > 0}), encoding="utf-8")
            (folder / "document.pdf").write_bytes(b"%PDF")


def test_the_smoke_subset_takes_whole_families_across_lines(tmp_path):
    from orchestration.smoke_run import select_sources

    _bundles(tmp_path)
    chosen = select_sources(tmp_path, train_sources=4)
    metas = [json.loads((f / "metadata.json").read_text("utf-8")) for f in chosen]
    by_split = {s: {m["template_id"] for m in metas if m["split"] == s} for s in ("train", "val", "test")}
    assert len(by_split["train"]) == 4 and len(by_split["val"]) == 1 and len(by_split["test"]) == 1
    assert len({m["lob"] for m in metas if m["split"] == "train"}) == 4        # four different lines
    assert len(chosen) == 6 * 3                                                 # every twin of each source


def test_the_smoke_subset_needs_every_split(tmp_path):
    from orchestration.smoke_run import SmokeError, select_sources

    _bundles(tmp_path)
    with pytest.raises(SmokeError, match="val source"):
        select_sources(tmp_path, val_sources=2)


def test_the_smoke_subset_can_take_a_validation_and_a_test_source_of_every_line(tmp_path):
    from orchestration.smoke_run import parse_args, select_sources

    _bundles(tmp_path)
    for split, lob, template in [("val", "homeowners", "A/homeowners/h4"), ("val", "motorcycle", "C/motorcycle/m2"),
                                 ("test", "homeowners", "A/homeowners/h5"), ("test", "motorcycle", "C/motorcycle/m3")]:
        folder = tmp_path / f"{lob}__{template.rsplit('/', 1)[-1]}__synth_000"
        folder.mkdir()
        (folder / "metadata.json").write_text(json.dumps(
            {"split": split, "lob": lob, "template_id": template, "synthetic": False}), encoding="utf-8")
    args, _steps = parse_args(["--tenant", "smoke9", "--val-sources", "2", "--test-sources", "2"])
    chosen = select_sources(tmp_path, train_sources=args.train_sources, val_sources=args.val_sources,
                            test_sources=args.test_sources)
    metas = [json.loads((f / "metadata.json").read_text("utf-8")) for f in chosen]
    for split in ("val", "test"):
        assert len({m["lob"] for m in metas if m["split"] == split}) == 2        # one per line first
    assert parse_args(["--tenant", "smoke9"])[0].val_sources == 1                # as before by default


def test_the_smoke_subset_is_linked_not_moved(tmp_path):
    from orchestration.smoke_run import select_sources, stage_subset

    _bundles(tmp_path / "batch")
    chosen = select_sources(tmp_path / "batch")
    assert stage_subset(chosen, tmp_path / "subset") == len(chosen)
    assert all((tmp_path / "subset" / f.name / "document.pdf").is_file() for f in chosen)
    assert all((f / "document.pdf").is_file() for f in chosen)


def test_each_smoke_step_runs_in_its_own_environment_under_the_smoke_tenant(tmp_path):
    from orchestration.smoke_run import OCR_PYTHON, commands

    cmds = commands(batch_dir=tmp_path, subset_dir=tmp_path / "s", tenant="smoke", version="v0",
                    check_out=tmp_path / "c")
    assert cmds["ocr"][0] == str(OCR_PYTHON)
    assert all(c[c.index("--tenant") + 1] == "smoke" for c in cmds.values())
    assert "--out-version" in cmds["finetune"] and "v0" in cmds["finetune"]
    assert "freeze-eval-set" not in " ".join(sum(cmds.values(), []))


def test_an_empty_workspace_setting_means_the_default(monkeypatch):
    """.env carries `FIDEON_WORKSPACE=` (empty): the OCR interpreter must still be
    /workspace/venv-ocr/bin/python, not a path relative to wherever the run started."""
    import importlib

    import orchestration.preflight as preflight
    import orchestration.smoke_run as smoke

    monkeypatch.setenv("FIDEON_WORKSPACE", "")
    try:
        assert str(importlib.reload(smoke).OCR_PYTHON).replace("\\", "/") == "/workspace/venv-ocr/bin/python"
        assert str(importlib.reload(preflight).WORKSPACE).replace("\\", "/") == "/workspace"
    finally:
        monkeypatch.delenv("FIDEON_WORKSPACE")
        importlib.reload(smoke)
        importlib.reload(preflight)



def test_the_smoke_preflight_skips_the_corpus_build_only_when_finetune_builds_it():
    from pathlib import Path

    from orchestration.smoke_run import commands

    tmp = Path("/tmp")
    chained = commands(batch_dir=tmp, subset_dir=tmp, tenant="smoke", version="v0", check_out=tmp)
    assert "--skip-corpus" in chained["preflight"]
    alone = commands(batch_dir=tmp, subset_dir=tmp, tenant="smoke", version="v0", check_out=tmp,
                     steps=["preflight"])
    assert "--skip-corpus" not in alone["preflight"]
