"""Training on the pod that built the corpus reads the copy the build kept, never a copy that differs from Blob."""

from __future__ import annotations

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from training import stage_data


def _client():
    return BlobClient(backend=InMemoryBackend(), container="c")


def test_the_copy_kept_beside_the_upload_is_read_instead_of_downloading(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path / "volume"))
    (tmp_path / "volume").mkdir()
    client = _client()
    key = paths.corpus_epoch_file("v9", 1, None)
    text = '{"source_id": "policy_0001"}\n'
    client.write_text(key, text)
    stage_data.keep_corpus_copy(key, text)
    downloads = []
    real = client.read_text
    monkeypatch.setattr(client, "read_text", lambda k: downloads.append(k) or real(k))
    assert stage_data._read_rows(client, key) == [{"source_id": "policy_0001"}]
    assert downloads == []


def test_a_copy_that_is_not_the_blobs_bytes_is_never_used(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path / "volume"))
    (tmp_path / "volume").mkdir()
    client = _client()
    key = paths.corpus_epoch_file("v9", 1, None)
    stage_data.keep_corpus_copy(key, '{"source_id": "old"}\n')
    client.write_text(key, '{"source_id": "new"}\n')                  # the blob was written again since
    assert stage_data._read_rows(client, key) == [{"source_id": "new"}]


def test_no_volume_no_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path / "not-mounted"))
    stage_data.keep_corpus_copy(paths.corpus_epoch_file("v9", 1, None), "x\n")
    assert not (tmp_path / "not-mounted").exists()


def test_signatures_made_in_worker_processes_are_the_ones_made_one_by_one():
    """Document grouping (cycle 2: ~1 h+): the same MinHash signatures, made in parallel."""
    from data_pipeline.ingestion.dedup_and_group import minhash
    from orchestration.pipeline_dag import PARALLEL_MINHASH_FROM, _minhashes

    texts = [f"policy {i} declarations named insured acme {i % 7} limits of insurance each occurrence"
             for i in range(PARALLEL_MINHASH_FROM + 5)] + [""]
    assert _minhashes(texts, minhash) == [minhash(text) for text in texts]
