"""scripts/served_adapter.py: the promoted release's adapter, made ready for a vLLM we run no code on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from scripts import served_adapter as sa

PIN = {"mineru_version": "4.0.12", "ocr_device": "cuda"}
ADAPTER = "adapters/scope/policy/v3/run-1/final"


def _release(**changes):
    release = dict(release_id="v3", served_as="lora_on_base", adapter_dir=ADAPTER, ocr_pin=PIN,
                   calibrators="releases/v3/calibration/bf16/calibrators.json")
    return SimpleNamespace(**{**release, **changes})


@pytest.fixture
def client():
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    client.write_json(f"{ADAPTER}/adapter_config.json", {
        "r": 16, "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    })
    client.write_bytes(f"{ADAPTER}/adapter_model.safetensors", b"\0" * 1024)
    return client


def _promoted(monkeypatch, *releases):
    import serving.release_router as router

    monkeypatch.setattr(router, "build_serving_plan",
                        lambda client, tenant_id=None: SimpleNamespace(served=list(releases)))


def test_the_promoted_adapter_is_downloaded_with_the_flags_that_serve_it(client, monkeypatch, tmp_path):
    _promoted(monkeypatch, _release())
    ready = sa.prepare(client, tenant=None, out=tmp_path, pod_dir="/workspace/adapters/")
    assert ready["name"] == "policy-v3" and ready["files"] == 2 and ready["rank"] == 16
    assert (tmp_path / "policy-v3" / "adapter_config.json").is_file()
    assert ready["flags"] == ("--enable-lora --max-lora-rank 16 --max-loras 1 "
                              "--lora-modules policy-v3=/workspace/adapters/policy-v3")


def test_a_release_vllm_cannot_serve_as_an_adapter_is_refused(client, monkeypatch, tmp_path):
    _promoted(monkeypatch)
    with pytest.raises(sa.ServedAdapterError, match="no release is promoted"):
        sa.prepare(client, tenant=None, out=tmp_path, pod_dir="/workspace/adapters")
    _promoted(monkeypatch, _release(served_as="merged", adapter_dir=None))
    with pytest.raises(sa.ServedAdapterError, match="lora_on_base"):
        sa.prepare(client, tenant=None, out=tmp_path, pod_dir="/workspace/adapters")
    # Trained on the in-process MinerU 3.4.5's text, served with the API's 4.0.12: refused before any download.
    _promoted(monkeypatch, _release(ocr_pin={"mineru_version": "3.4.5"}))
    with pytest.raises(sa.ServedAdapterError, match="MinerU 3.4.5, but serving reads pages with MinerU 4.0.12"):
        sa.prepare(client, tenant=None, out=tmp_path, pod_dir="/workspace/adapters")
    assert not (tmp_path / "policy-v3").exists()


def test_only_a_language_model_lora_within_vllms_ranks_is_served():
    assert [sa.vllm_rank(r) for r in (1, 8, 12, 16, 64, 300)] == [1, 8, 16, 16, 64, 320]
    with pytest.raises(sa.ServedAdapterError, match="above vLLM's largest"):
        sa.vllm_rank(600)
    with pytest.raises(sa.ServedAdapterError, match="vision side"):
        sa.check_adapter_config({"r": 16, "target_modules": ["q_proj", "merger"]})
    with pytest.raises(sa.ServedAdapterError, match="whole modules"):
        sa.check_adapter_config({"r": 16, "target_modules": ["q_proj"], "modules_to_save": ["lm_head"]})
    assert sa.check_adapter_config({"r": 32, "target_modules": "all-linear"}) == 32


def test_the_runner_refuses_a_minerU_tier_the_model_was_not_trained_on(tmp_path, capsys):
    from testing import latency_check

    with pytest.raises(SystemExit):
        latency_check.main(["--model-url", "https://m", "--mineru-url", "https://o", "--tier", "flash",
                            "--doc", f"{tmp_path / 'a.pdf'}=personal_auto"])
    assert "not the pinned tier (basic)" in capsys.readouterr().err
