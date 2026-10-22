"""A delivered batch travels through Blob, not git: staged under intake/, pulled onto the pod.

intake/ holds the delivered PDFs with their unredacted PII, so it sits in the raw
container under the raw layer's access rule. The pull is resumable and never
leaves a truncated file under its real name, and a batch pulled this way imports
exactly as a local folder does.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import AccessDeniedError, BlobClient, InMemoryBackend
from data_pipeline.ingestion import pull_intake as intake
from data_pipeline.ingestion.import_labeled_pdfs import import_batch
from tests.test_import_labeled_pdfs import GOLDEN, LINE


@pytest.fixture
def backend():
    return InMemoryBackend()


@pytest.fixture
def raw_client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ingestion")


def _stage(backend, batch="personal-v1", docs=("doc-a", "doc-b")):
    for name in docs:
        base = f"intake/{batch}/{name}"
        backend.write("raw", f"{base}/policy.pdf", f"%PDF-1.7 {name}".encode())
        backend.write("raw", f"{base}/golden.json", json.dumps(GOLDEN).encode())
        # With a review record: every gold needs one to import (decision 2026-10-22).
        backend.write("raw", f"{base}/metadata.json", json.dumps(
            {"lob": LINE, "reviewer_id": "a.reviewer", "review_date": "2026-10-20"}).encode())


# --------------------------------------------------------------------------
# Where the staging lives, and who may read it
# --------------------------------------------------------------------------


def test_intake_is_in_the_raw_container_under_the_raw_rule(backend):
    assert paths.requires_raw_container("intake/personal-v1/doc-a/policy.pdf")
    assert paths.requires_raw_container("intake")
    assert not paths.requires_raw_container("intake-reports/x")
    general = BlobClient(backend=backend, container="main", raw_container="raw")
    with pytest.raises(AccessDeniedError):
        general.list("intake/personal-v1")


@pytest.mark.parametrize("batch", ["", "../raw-documents", "a/b", "with space"])
def test_a_batch_name_cannot_escape_intake(batch):
    with pytest.raises(ValueError):
        paths.intake_batch_dir(batch)


# --------------------------------------------------------------------------
# The pull
# --------------------------------------------------------------------------


def test_a_batch_lands_folder_for_folder(backend, raw_client, tmp_path):
    _stage(backend)
    report = intake.pull_intake(raw_client, "personal-v1", tmp_path, workers=4)
    assert report.local_dir == tmp_path / "personal-v1"
    assert (report.downloaded, report.skipped, report.failed) == (6, 0, [])
    assert (report.local_dir / "doc-a" / "policy.pdf").read_bytes() == b"%PDF-1.7 doc-a"
    assert not list(report.local_dir.rglob("*.part"))


def test_a_rerun_skips_pdfs_but_refreshes_corrected_labels(backend, raw_client, tmp_path):
    _stage(backend)
    intake.pull_intake(raw_client, "personal-v1", tmp_path)
    backend.write("raw", "intake/personal-v1/doc-a/metadata.json", b'{"lob": "flood"}')   # re-synced fix
    report = intake.pull_intake(raw_client, "personal-v1", tmp_path)
    assert (report.downloaded, report.skipped) == (4, 2)
    assert json.loads((tmp_path / "personal-v1/doc-a/metadata.json").read_text())["lob"] == "flood"


def test_refresh_downloads_pdfs_again(backend, raw_client, tmp_path):
    _stage(backend)
    intake.pull_intake(raw_client, "personal-v1", tmp_path)
    assert intake.pull_intake(raw_client, "personal-v1", tmp_path, refresh=True).downloaded == 6


def test_a_neighbouring_batch_is_not_pulled(backend, raw_client, tmp_path):
    _stage(backend, "personal-v1", ["doc-a"])
    _stage(backend, "personal-v10", ["doc-z"])
    intake.pull_intake(raw_client, "personal-v1", tmp_path)
    assert not (tmp_path / "personal-v1" / "doc-z").exists()


def test_an_interrupted_download_leaves_no_file_under_its_real_name(backend, raw_client, tmp_path,
                                                                    monkeypatch):
    _stage(backend, docs=["doc-a"])
    real_read = raw_client.read_bytes

    def flaky(key):
        if key.endswith("golden.json"):
            raise OSError("connection reset")
        return real_read(key)

    monkeypatch.setattr(raw_client, "read_bytes", flaky)
    report = intake.pull_intake(raw_client, "personal-v1", tmp_path)
    assert [key for key, _ in report.failed] == ["intake/personal-v1/doc-a/golden.json"]
    assert not (tmp_path / "personal-v1/doc-a/golden.json").exists()
    assert (tmp_path / "personal-v1/doc-a/policy.pdf").is_file()


def test_nothing_staged_says_how_to_stage_it(raw_client, tmp_path):
    with pytest.raises(intake.IntakeError, match="azcopy sync"):
        intake.pull_intake(raw_client, "personal-v1", tmp_path)


def test_a_blob_name_that_climbs_out_is_refused(tmp_path):
    with pytest.raises(intake.IntakeError, match="outside"):
        intake._local_path(tmp_path, "doc/../../etc/passwd")


def test_the_default_destination_is_on_the_volume(monkeypatch):
    monkeypatch.delenv(intake.INTAKE_DIR_ENV, raising=False)
    assert intake.default_intake_dir().as_posix() == "/workspace/intake"


# --------------------------------------------------------------------------
# Pulled, it imports like a local folder
# --------------------------------------------------------------------------


def test_a_pulled_batch_imports(backend, raw_client, tmp_path, monkeypatch):
    from data_pipeline.ingestion import pull_raw_pdfs

    monkeypatch.setattr(pull_raw_pdfs, "inspect_pdf", lambda path: {
        "page_count": 2, "is_scanned": False, "text_chars_per_sampled_page": 900.0})
    _stage(backend)
    local = intake.pull_intake(raw_client, "personal-v1", tmp_path).local_dir
    client = BlobClient(backend=backend, container="main", raw_container="raw")
    report = import_batch(local, "policy", client, raw_client)
    assert not report.failed and len(report.imported) == 2
    source_id = report.imported["doc-a"]
    assert client.read_json(paths.golden_label("policy", source_id)) == GOLDEN


def test_the_importer_takes_a_blob_batch_instead_of_a_folder():
    import ast
    from pathlib import Path

    import data_pipeline.ingestion.import_labeled_pdfs as module

    text = Path(module.__file__).read_text(encoding="utf-8")
    assert '"--from-blob"' in text and "add_mutually_exclusive_group(required=True)" in text
    ast.parse(text)
