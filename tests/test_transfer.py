"""IMPL-02 §3 — the path-aware push/pull helpers.

These exist so no caller ever hand-builds a Blob path. That is not tidiness: the
only place that did build them inline published a Foundation run against the
`doc_type=None` "unified" merged and quantized paths, which exist only under
`--foundation-only`, so the manifest pointed at artifacts nobody had built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from artifact_registry import paths, transfer
from artifact_registry.blob_client import BlobClient, InMemoryBackend


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def artifact(tmp_path) -> Path:
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    (directory / "adapter_config.json").write_text('{"r": 64}', encoding="utf-8")
    return directory


def test_an_adapter_round_trips(client, artifact, tmp_path):
    prefix = transfer.push_adapter(artifact, "foundation", "v2", client=client)
    assert prefix == paths.adapter_dir("foundation", "v2")

    out = transfer.pull_adapter("foundation", "v2", tmp_path / "out", client=client)
    assert (out / "adapter_model.safetensors").read_bytes() == b"weights"


def test_a_per_type_adapter_must_name_its_type(client, artifact):
    """The path depends on it, so an unnamed one would land on the Foundation's."""
    with pytest.raises(transfer.TransferError, match="must name its doc_type"):
        transfer.push_adapter(artifact, "doc_type", "v2", None, client=client)


def test_the_unified_and_per_type_model_paths_are_distinct(client, artifact):
    """`doc_type=None` is the unified model, built only under --foundation-only.
    Conflating the two is what published a Foundation against paths that were
    never produced."""
    unified = transfer.push_merged_model(artifact, "v2", None, client=client)
    per_type = transfer.push_merged_model(artifact, "v2", "policy", client=client)
    assert unified != per_type


def test_pushing_an_absent_directory_is_refused(client, tmp_path):
    """An empty Blob prefix later reads as a published model."""
    with pytest.raises(transfer.TransferError, match="does not exist"):
        transfer.push_merged_model(tmp_path / "nope", "v2", client=client)


def test_quantized_models_are_kept_per_format(client, artifact):
    fp16 = transfer.push_quantized(artifact, "v2", "fp16", "policy", client=client)
    q5 = transfer.push_quantized(artifact, "v2", "q5_k_m", "policy", client=client)
    assert fp16 != q5


def test_a_corpus_version_round_trips(client, artifact, tmp_path):
    transfer.push_corpus_version(artifact, "v3", client=client)
    out = transfer.pull_corpus_version("v3", tmp_path / "corpus", client=client)
    assert (out / "adapter_config.json").exists()


def test_missing_calibration_raises_rather_than_returning_nothing(client):
    """Serving raw confidence as if calibrated makes every downstream review
    threshold meaningless."""
    transfer.push_calibration("v2", "policy", {"method": "temperature"}, client=client)
    assert transfer.pull_calibration("v2", "policy", client=client)["method"] == "temperature"

    with pytest.raises(transfer.TransferError, match="no calibration"):
        transfer.pull_calibration("v2", "lossrun", client=client)


def test_an_absent_eval_report_is_none_not_an_error(client):
    """A missing report is a fact the gate handles; only calibration must raise."""
    assert transfer.pull_eval_report("v9", client=client) is None
    transfer.push_eval_report({"field_exact_match": 0.9}, "v2", client=client)
    assert transfer.pull_eval_report("v2", client=client)["field_exact_match"] == 0.9


def test_the_golden_eval_set_has_its_own_prefix(client, artifact, tmp_path):
    """Frozen and versioned separately from the corpus — that constancy is the
    only reason model versions stay comparable over time."""
    prefix = transfer.push_golden_eval_set(artifact, client=client)
    assert prefix == paths.golden_eval_set_dir() == "golden-eval-set/default"
    assert "corpus" not in prefix

    out = transfer.pull_golden_eval_set(tmp_path / "eval", client=client)
    assert (out / "adapter_config.json").exists()


def test_each_tenant_pushes_and_pulls_only_its_own_golden_eval_set(client, artifact, tmp_path):
    """Each tenant is gated on its own frozen set, never on another's."""
    prefix = transfer.push_golden_eval_set(artifact, client=client, tenant_id="acme-2")
    assert prefix == "golden-eval-set/acme-2"
    assert paths.tenant_of(f"{prefix}/adapter_config.json") == "acme-2"

    # `golden-eval-set/acme` is a prefix of `golden-eval-set/acme-2`.
    transfer.pull_golden_eval_set(tmp_path / "acme", client=client, tenant_id="acme")
    assert not (tmp_path / "acme").exists() or not any((tmp_path / "acme").rglob("*"))
    out = transfer.pull_golden_eval_set(tmp_path / "acme-2", client=client, tenant_id="acme-2")
    assert (out / "adapter_config.json").exists()


def test_an_uncached_base_model_does_not_silently_fall_back(client, tmp_path):
    with pytest.raises(transfer.TransferError, match="not cached"):
        transfer.pull_base_model(tmp_path / "base", client=client, allow_hf=False)


def test_a_published_adapter_is_its_weights_and_config_without_the_resume_state(client, artifact):
    """A training checkpoint also holds ~1.5 GB of optimizer and scheduler state a
    resume needs; nothing that reads a published adapter opens it."""
    (artifact / "optimizer.pt").write_bytes(b"x" * 64)
    (artifact / "trainer_state.json").write_text("{}", encoding="utf-8")
    prefix = transfer.push_scoped_adapter(artifact, "casualty_fleet", "v1", client=client)
    assert sorted(k.rsplit("/", 1)[-1] for k in client.list(prefix)) == sorted(transfer.ADAPTER_FILES)


def test_a_folder_without_the_adapter_files_is_refused(client, tmp_path):
    folder = tmp_path / "checkpoint-9"
    folder.mkdir()
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(transfer.TransferError, match="adapter_config.json"):
        transfer.push_scoped_adapter(folder, "casualty_fleet", "v1", client=client)


def test_a_repush_sends_only_what_blob_does_not_already_hold(client, tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model-00001.safetensors").write_bytes(b"shard one")
    writes = []
    real = client._backend.write
    client._backend.write = lambda c, k, d: (writes.append(k), real(c, k, d))[1]
    transfer.push_merged_model(merged, "v1", client=client, scope="casualty_fleet")
    assert len(writes) == 2
    writes.clear()
    transfer.push_merged_model(merged, "v1", client=client, scope="casualty_fleet")
    assert writes == []                                                  # every byte already there
    (merged / "model-00001.safetensors").write_bytes(b"shard ONE")       # same size, other bytes
    transfer.push_merged_model(merged, "v1", client=client, scope="casualty_fleet")
    assert [k.rsplit("/", 1)[-1] for k in writes] == ["model-00001.safetensors"]


def test_a_blob_that_records_no_md5_is_sent_again(client, tmp_path):
    from artifact_registry.blob_client import file_md5

    path = tmp_path / "f.bin"
    path.write_bytes(b"data")
    client.write_bytes("merged-models/scope/x/v1/f.bin", b"data")
    assert client.holds(path, "merged-models/scope/x/v1/f.bin")
    assert file_md5(path) == __import__("hashlib").md5(b"data").digest()
    client._backend.md5 = lambda container, key: None                    # e.g. a block upload with none
    assert not client.holds(path, "merged-models/scope/x/v1/f.bin")
    assert not client.holds(path, "merged-models/scope/x/v1/missing.bin")


def test_upload_dir_refuses_a_named_file_that_is_not_there(client, tmp_path):
    from artifact_registry.blob_client import BlobError

    with pytest.raises(BlobError, match="no such file"):
        client.upload_dir(tmp_path, "adapters/x", names=["absent.bin"])
