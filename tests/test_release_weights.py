"""The weights that ship are the weights that were measured.

Merge folds the selected checkpoint into a pinned base; package uploads the real
directories rather than a placeholder JSON; training launches only where its
outputs can be found.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from training.merge import MergeError, MergePlan, merge


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _plan(tmp_path: Path, *, adapter: Path, base: str = "Qwen/Qwen3-VL-8B-Instruct@abc123"):
    return MergePlan(base_model=base, adapter=str(adapter), output_dir=str(tmp_path / "out"),
                     dtype="bf16")


def test_a_merge_needs_an_adapter_to_merge(tmp_path):
    with pytest.raises(MergeError, match="adapter_config.json"):
        merge(_plan(tmp_path, adapter=tmp_path / "root"))


def test_a_merge_refuses_a_floating_base_revision(tmp_path):
    adapter = tmp_path / "checkpoint-100"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MergeError, match="revision"):
        merge(_plan(tmp_path, adapter=adapter, base="Qwen/Qwen3-VL-8B-Instruct@PIN_ME"))


def test_a_scoped_push_lands_where_the_manifest_points(client, tmp_path):
    from artifact_registry.transfer import push_merged_model, push_scoped_adapter

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text('{"r": 64}', encoding="utf-8")
    prefix = push_scoped_adapter(adapter, "policy", "v2", client=client)
    assert prefix == paths.scoped_adapter_dir("policy", "v2")
    assert client.read_bytes(f"{prefix}/adapter_model.safetensors") == b"weights"

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    assert push_merged_model(merged, "v2", client=client, scope="policy") == (
        paths.merged_model_dir("v2", scope="policy")
    )


def test_pushing_an_absent_directory_is_refused(client, tmp_path):
    from artifact_registry.transfer import TransferError, push_scoped_adapter

    with pytest.raises(TransferError):
        push_scoped_adapter(tmp_path / "missing", "policy", "v2", client=client)


def test_training_refuses_to_launch_off_the_pod(monkeypatch):
    from training import train

    monkeypatch.delenv(train.OFF_POD_ENV, raising=False)
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", "/no/such/mount")
    with pytest.raises(train.TrainingError, match="refusing to launch"):
        train.assert_on_pod()


def test_the_off_pod_override_is_explicit(monkeypatch):
    from training import train

    monkeypatch.setenv(train.OFF_POD_ENV, "1")
    train.assert_on_pod()


# --------------------------------------------------------------------------
# The base model is loaded from the pod's local copy
# --------------------------------------------------------------------------


def _write_model(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


@pytest.mark.parametrize("layout", ["flat", "named", "hub_cache"])
def test_the_base_is_found_under_the_models_directory(tmp_path, monkeypatch, layout):
    from common.config import BASE_MODEL_DIR_ENV, base_model_dir, base_model_source

    target = {
        "flat": tmp_path,
        "named": tmp_path / "Qwen3-VL-8B-Instruct",
        "hub_cache": tmp_path / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / "abc123",
    }[layout]
    _write_model(target)
    monkeypatch.setenv(BASE_MODEL_DIR_ENV, str(tmp_path))
    assert base_model_dir() == target
    assert base_model_source() == str(target)


def test_training_and_serving_load_the_local_copy(tmp_path, monkeypatch):
    from common.config import BASE_MODEL_DIR_ENV
    from registry_utils.query_registry import resolve_model_version
    from training.train import build_training_config

    local = _write_model(tmp_path / "Qwen3-VL-8B-Instruct")
    monkeypatch.setenv(BASE_MODEL_DIR_ENV, str(tmp_path))
    swift, _ = build_training_config(corpus_paths=["e1.jsonl", "e2.jsonl", "e3.jsonl"], output_dir="out",
                                     val_paths=["val.jsonl"])
    assert swift.args["model"] == str(local)
    assert swift.args.get("model_revision") is None

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    assert resolve_model_version("base", client)["base_model"] == str(local)


def test_without_local_weights_the_hub_id_is_the_fallback(tmp_path, monkeypatch):
    from common.config import BASE_MODEL_DIR_ENV, base_model_config, base_model_source

    monkeypatch.setenv(BASE_MODEL_DIR_ENV, str(tmp_path / "empty"))
    assert base_model_source() == base_model_config()["model"]["model_id"]


def test_a_real_launch_is_refused_when_the_configured_base_is_missing(tmp_path, monkeypatch):
    from common.config import BASE_MODEL_DIR_ENV
    from training import train

    monkeypatch.delenv(train.OFF_POD_ENV, raising=False)
    monkeypatch.setenv(BASE_MODEL_DIR_ENV, str(tmp_path / "nothing-here"))
    with pytest.raises(train.TrainingError, match="no base model"):
        train.assert_on_pod()
