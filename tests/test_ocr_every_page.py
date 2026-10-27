"""Every page read by OCR (configs/shared/ocr.yaml read_every_page_by_ocr, decision
2026-10-27): a PDF with a text layer is read as a scan is, in training and in
serving, and the setting is part of the OCR pin."""

from __future__ import annotations

import pytest

from common.config import ocr_every_page
from scripts.served_adapter import ServedAdapterError, assert_ocr_pin

pymupdf = pytest.importorskip("pymupdf")

PIN = {"version": "4.0.12", "tier": "basic"}


def _native_pdf() -> bytes:
    doc = pymupdf.open()
    for number in range(2):
        doc.new_page().insert_text((72, 72), f"Policy Declarations page {number + 1} " + "text " * 40)
    return doc.tobytes()


def test_the_shipped_setting_reads_every_page_by_ocr():
    assert ocr_every_page() is True


def test_a_pdf_with_a_text_layer_takes_the_ocr_path_whatever_the_caller_says(monkeypatch):
    from data_pipeline.ocr.modality import NATIVE, reads_by_ocr
    from data_pipeline.ocr.served_ocr import ocr_mode_for

    pdf = _native_pdf()
    assert reads_by_ocr(pdf) and reads_by_ocr(pdf, modality=NATIVE, force_ocr=False)
    assert ocr_mode_for(pdf) == "ocr"
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)
    assert not reads_by_ocr(pdf) and ocr_mode_for(pdf) == "txt"          # the per-document rule, off


def test_a_served_reading_records_the_setting_beside_the_tier():
    from data_pipeline.ocr.served_ocr import ServedMinerUEngine

    meta = ServedMinerUEngine("https://mineru.example.test", "tok", render_workers=1).meta()
    assert meta == {"mineru_backend": "api", "mineru_tier": "basic", "read_every_page_by_ocr": True}


def test_a_stored_reading_under_the_other_setting_is_not_done(monkeypatch):
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.ocr.run_mineru import find_unprocessed

    # The OCR stage's login: it may read raw-documents/.
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="ocr")
    readings = {"policy_0001": {"read_every_page_by_ocr": True}, "policy_0002": {"read_every_page_by_ocr": False},
                # From before the setting: one read by OCR, one a hybrid read from its text layer.
                "policy_0003": {"read_by_ocr": True}, "policy_0004": {"read_by_ocr": False}}
    for source_id, recorded in readings.items():
        client.write_json(paths.raw_metadata("policy", source_id), {"source_id": source_id})
        client.write_json(paths.ocr_meta("policy", source_id),
                          {"mineru_version": "4.0.12", "mineru_tier": "basic", "page_count": 1, **recorded})
    # Read with the setting on: the reading that recorded it off is read again, and
    # so is one from before the setting that read a text layer - taken as done, it
    # trained on other text than serving reads, and no key told serving so.
    assert find_unprocessed(client, "policy", mineru_version="4.0.12") == ["policy_0002", "policy_0004"]
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)
    assert find_unprocessed(client, "policy", mineru_version="4.0.12") == ["policy_0001"]


def test_serving_refuses_a_release_trained_under_the_other_setting(monkeypatch):
    trained = {"mineru_version": "4.0.12", "mineru_tier": "basic"}
    assert_ocr_pin({**trained, "read_every_page_by_ocr": True}, PIN)
    assert_ocr_pin(trained, PIN)                       # a release from before the setting: version and tier
    with pytest.raises(ServedAdapterError, match="read_every_page_by_ocr False, but serving reads with True"):
        assert_ocr_pin({**trained, "read_every_page_by_ocr": False}, PIN)
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)
    with pytest.raises(ServedAdapterError, match="a PDF with a text layer is read another way"):
        assert_ocr_pin({**trained, "read_every_page_by_ocr": True}, PIN)


def test_promotion_holds_a_release_to_the_setting_too():
    from registry_utils.releases import ocr_pin_refusal

    trained = {"mineru_version": "4.0.12", "mineru_tier": "basic"}
    assert ocr_pin_refusal({**trained, "read_every_page_by_ocr": True}) is None
    assert "read another way" in ocr_pin_refusal({**trained, "read_every_page_by_ocr": False})


def test_a_corpus_read_under_the_other_setting_is_refused():
    """A release trained on it would be refused by serving at start-up: the
    corpus build refuses first, saying how to read the documents again."""
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from orchestration.pipeline_dag import PipelineError, load_labeled_documents
    from orchestration.runpod_controller import LocalBackend, RunPodController
    from tests.test_orchestration import make_context, seed_corpus, seeded_policies

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    source_id = seeded_policies(seed_corpus(client))[0]
    doc_type = source_id.rsplit("_", 1)[0]
    key = paths.ocr_meta(doc_type, source_id)
    ctx = make_context(client, RunPodController(backend=LocalBackend(), volume_id="v", git_commit="abc1234"))
    ctx.doc_types = [doc_type]
    client.write_json(key, {**client.read_json(key), "read_every_page_by_ocr": False})
    with pytest.raises(PipelineError, match=r"read with read_every_page_by_ocr False.*--force"):
        load_labeled_documents(ctx)
    client.write_json(key, {**client.read_json(key), "read_every_page_by_ocr": True})
    assert source_id in {d.source_id for d in load_labeled_documents(ctx)}
