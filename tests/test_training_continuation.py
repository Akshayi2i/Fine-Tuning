"""Continuing an adapter: what the run trains with, and which checkpoints it resumes.

ms-swift 3.12 loads a continued run's adapter with ``from_pretrained(...,
is_trainable=True)``: the adapter's own ``adapter_config.json`` decides its
rank, alpha, dropout, rsLoRA, bias and targets, and the command's ``lora_*``
arguments are ignored. So a continued run whose adapter disagrees with the
config is refused - it would train with the adapter's values and record the
config's.

A checkpoint is OF the weights its run started from: the base, or an adapter,
named in the run fingerprint by the hash of its weights. A run from the base and
one continued from an adapter - by the pipeline or by hand - never resume each
other's checkpoints.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from training import train as T


def matching_adapter_config(**over) -> dict:
    """``adapter_config.json`` as PEFT saves it for an adapter trained with the
    shipped config's LoRA settings."""
    from common.adapter import ADAPTER
    from common.config import training_config

    lora = training_config(ADAPTER.training_config)["lora"]
    modules = list(lora["target_modules"]) + (["merger"] if lora.get("include_vision_projector") else [])
    return {"peft_type": "LORA", "r": lora["rank"], "lora_alpha": lora["alpha"],
            "lora_dropout": lora["dropout"], "use_rslora": bool(lora.get("use_rslora", False)),
            "bias": lora["bias"], "target_modules": sorted(modules), **over}


def _adapter(directory: Path, *, weights: bytes | None = b"weights-a", **config) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps(matching_adapter_config(**config)),
                                                   encoding="utf-8")
    if weights is not None:
        (directory / "adapter_model.safetensors").write_bytes(weights)
    return str(directory)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


#: The config's lora block, as training_config gives it.
LORA = {"rank": 64, "alpha": 128, "dropout": 0.05, "bias": "none", "use_rslora": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "include_vision_projector": False}

#: What PEFT saves for an adapter trained with LORA.
SAVED = {"peft_type": "LORA", "r": 64, "lora_alpha": 128, "lora_dropout": 0.05, "use_rslora": False,
         "bias": "none", "target_modules": ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj",
                                            "up_proj", "v_proj"]}


# --------------------------------------------------------------------------
# adapter_config_mismatch: the adapter's saved LoRA settings against the config
# --------------------------------------------------------------------------


def test_an_adapter_saved_with_the_config_s_settings_matches():
    assert T.adapter_config_mismatch(SAVED, LORA) is None
    # Floats compared with a tolerance, and an int where the config has a float.
    assert T.adapter_config_mismatch({**SAVED, "lora_dropout": 0.05000000000001, "lora_alpha": 128.0},
                                     LORA) is None
    # Missing use_rslora and bias mean PEFT's defaults, False and "none".
    defaults = {k: v for k, v in SAVED.items() if k not in ("use_rslora", "bias")}
    assert T.adapter_config_mismatch(defaults, LORA) is None
    assert T.adapter_config_mismatch({**SAVED, "use_rslora": None, "bias": None}, LORA) is None


def test_the_shipped_config_matches_what_peft_saves_for_it():
    from common.config import training_config

    assert T.adapter_config_mismatch(matching_adapter_config(), training_config("unified")["lora"]) is None


@pytest.mark.parametrize("saved, said", [
    ({"lora_dropout": 0.1}, "lora_dropout is 0.1 in the adapter, 0.05 in the config"),
    ({"use_rslora": True}, "use_rslora is True in the adapter, False in the config"),
    ({"bias": "lora_only"}, "bias is lora_only in the adapter, none in the config"),
    ({"r": 32}, "r is 32 in the adapter, 64 in the config"),
    ({"lora_alpha": 64}, "lora_alpha is 64 in the adapter, 128 in the config"),
    ({"r": None}, "r is missing from the adapter, 64 in the config"),
    ({"use_rslora": 1}, "use_rslora is 1 in the adapter, False in the config"),
])
def test_an_adapter_saved_with_other_settings_is_named_by_the_first_that_differs(saved, said):
    """The dropout case is the review's: the config moved 0.10 -> 0.05, the
    parent was saved at 0.10, and ms-swift would have kept training at 0.10."""
    assert T.adapter_config_mismatch({**SAVED, **saved}, LORA) == said


def test_rslora_or_bias_set_in_the_config_must_be_in_the_adapter_too():
    assert T.adapter_config_mismatch(SAVED, {**LORA, "use_rslora": True}) == (
        "use_rslora is False in the adapter, True in the config")
    defaults = {k: v for k, v in SAVED.items() if k != "bias"}
    assert T.adapter_config_mismatch(defaults, {**LORA, "bias": "all"}) == (
        "bias is none in the adapter, all in the config")


def test_target_modules_as_a_list_are_compared_as_a_set():
    reordered = {**SAVED, "target_modules": list(reversed(SAVED["target_modules"]))}
    assert T.adapter_config_mismatch(reordered, LORA) is None
    fewer = {**SAVED, "target_modules": ["q_proj", "v_proj"]}
    assert T.adapter_config_mismatch(fewer, LORA).startswith(
        "target_modules are ['q_proj', 'v_proj'] in the adapter, ['down_proj', ")
    # The vision projector adds "merger", which the adapter must then target too.
    vision = {**LORA, "include_vision_projector": True}
    assert "merger" in T.adapter_config_mismatch(SAVED, vision)
    assert T.adapter_config_mismatch({**SAVED, "target_modules": [*SAVED["target_modules"], "merger"]},
                                     vision) is None
    assert T.adapter_config_mismatch({k: v for k, v in SAVED.items() if k != "target_modules"},
                                     LORA).startswith("target_modules is missing from the adapter")


def test_target_modules_as_ms_swift_s_regex_must_name_every_configured_module():
    """ms-swift writes a regex for a multimodal model's targets, not a list."""
    pattern = (r"^(model\.language_model(?=\.).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj"
               r"|down_proj))$")
    assert T.adapter_config_mismatch({**SAVED, "target_modules": pattern}, LORA) is None
    narrower = r"^(model\.language_model(?=\.).*\.(q_proj|k_proj|v_proj|o_proj))$"
    said = T.adapter_config_mismatch({**SAVED, "target_modules": narrower}, LORA)
    assert said.endswith("which does not name ['down_proj', 'gate_proj', 'up_proj']"), said


# --------------------------------------------------------------------------
# train(): a continued run whose adapter disagrees with the config is refused
# --------------------------------------------------------------------------


def _as_if_on_the_pod(monkeypatch, tmp_path) -> list:
    """A real (non-dry) train() here, as tests/test_stage_data runs one: no CUDA
    and no ms-swift template, the volume a temporary folder, launch recorded."""
    from tests.test_stage_data import _stub_masking_encoder, _stub_tokens

    monkeypatch.setenv(T.OFF_POD_ENV, "1")
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path / "volume"))
    monkeypatch.setattr(T, "_masking_encoder", lambda swift: _stub_masking_encoder())
    monkeypatch.setattr(T, "_token_counter", lambda: _stub_tokens)
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    launched: list = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    return launched


def _train(client, **kwargs):
    from registry_utils.models import DataStats

    return T.train(corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
                   data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0), **kwargs)


def test_a_continued_run_whose_adapter_has_other_lora_settings_is_refused(client, monkeypatch, tmp_path):
    from registry_utils.query_registry import RegistryQueryError, get
    from tests.test_stage_data import _seed

    _seed(client)
    launched = _as_if_on_the_pod(monkeypatch, tmp_path)
    parent = _adapter(tmp_path / "parent", lora_dropout=0.05)        # the config says 0.10
    with pytest.raises(T.TrainingError) as refused:
        _train(client, dry_run=False, continue_from=parent, parent_run_id="policy-v8")
    message = str(refused.value)
    assert "lora_dropout is 0.05 in the adapter, 0.1 in the config" in message
    assert "--fresh" in message and "adapter_config.json" in message
    assert not launched
    # Refused before anything was recorded or staged.
    with pytest.raises((RegistryQueryError, KeyError, FileNotFoundError)):
        get("policy-v9", client)
    assert not Path(paths.scoped_staging_adapter_dir("policy", "v9")).exists()

    # The same adapter, saved with the config's settings, trains.
    _swift, manifest = _train(client, dry_run=False, continue_from=_adapter(tmp_path / "matching"),
                              parent_run_id="policy-v8")
    assert manifest.status == "trained" and launched[-1].args["adapters"] == [str(tmp_path / "matching")]


def test_an_adapter_config_that_cannot_be_read_is_refused(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(T.TrainingError, match="cannot be read"):
        _train(client, dry_run=False, continue_from=str(parent))
    (parent / "adapter_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(T.TrainingError, match="not a JSON object"):
        _train(client, dry_run=False, continue_from=str(parent))


def test_a_dry_run_reads_no_adapter(client, monkeypatch, tmp_path):
    """A dry run on an operator machine cannot see the pod's volume: it checks
    no adapter settings and hashes no weights."""
    from tests.test_stage_data import _seed

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path)
    hashed: list = []
    monkeypatch.setattr(T, "starting_weights", lambda continue_from: hashed.append(continue_from))
    parent = "/runpod-volume/staging/parent-adapters/policy/v9"
    swift, manifest = _train(client, dry_run=True, continue_from=parent, parent_run_id="policy-v8")
    assert swift.args["adapters"] == [parent] and manifest.continued_from == parent
    assert hashed == []


# --------------------------------------------------------------------------
# The fingerprint names the starting weights
# --------------------------------------------------------------------------


RECORDED = SimpleNamespace(lora_rank=64, lora_alpha=128, learning_rate=1e-4, lr_scheduler="cosine",
                           epochs=3, effective_batch_size=8, target_modules=["q_proj"])


def _fingerprint(continue_from=None, parent_run_id=None) -> dict:
    return T.run_fingerprint("v1", {"rows": 10}, RECORDED, parent_run_id, continue_from)


def test_the_fingerprint_names_the_weights_a_run_starts_from(tmp_path):
    fresh = _fingerprint()
    by_hand = _fingerprint(_adapter(tmp_path / "a"))
    pipeline = _fingerprint(_adapter(tmp_path / "a"), parent_run_id="policy-v8")
    other = _fingerprint(_adapter(tmp_path / "b", weights=b"weights-b"))
    assert fresh["starting_weights"] is None
    assert by_hand["starting_weights"].startswith("sha256:")
    assert len({str(f["starting_weights"]) for f in (fresh, by_hand, other)}) == 3
    # The weights, not where they are: the same adapter copied elsewhere is the same start.
    assert _fingerprint(_adapter(tmp_path / "copy"))["starting_weights"] == by_hand["starting_weights"]
    # The parent is still named, beside the weights.
    assert pipeline["parent_run_id"] == "policy-v8" and pipeline != by_hand


def test_with_no_weights_there_yet_the_start_is_the_normalised_directory(tmp_path):
    """A dry run, or a parent not yet staged: the directory names the start."""
    weightless = _adapter(tmp_path / "a", weights=None)
    named = _fingerprint(weightless)["starting_weights"]
    assert named == Path(weightless).as_posix()
    assert _fingerprint(weightless + "/")["starting_weights"] == named
    assert _fingerprint(f" {weightless}/./ ")["starting_weights"] == named


@pytest.mark.parametrize("written, resumed", [
    ("fresh", "by_hand"), ("by_hand", "fresh"), ("by_hand", "other"), ("fresh", "pipeline"),
])
def test_a_checkpoint_of_another_start_is_not_resumed(tmp_path, written, resumed):
    starts = {
        "fresh": lambda: _fingerprint(),
        "by_hand": lambda: _fingerprint(_adapter(tmp_path / "a")),
        "other": lambda: _fingerprint(_adapter(tmp_path / "b", weights=b"weights-b")),
        "pipeline": lambda: _fingerprint(_adapter(tmp_path / "a"), parent_run_id="policy-v8"),
    }
    run = tmp_path / "run"
    T.record_fingerprint(run, starts[written]())
    T.assert_resumable(run, starts[written]())                       # its own start resumes
    with pytest.raises(T.TrainingError, match="starting_weights|parent_run_id") as refused:
        T.assert_resumable(run, starts[resumed]())
    assert "a run from the base and one continued from an adapter" in str(refused.value)


def test_a_checkpoint_fingerprinted_before_the_start_was_recorded_is_not_resumed(tmp_path):
    """Its run may have started from the base or from an adapter: nothing says."""
    old = {k: v for k, v in _fingerprint().items() if k != "starting_weights"}
    T.record_fingerprint(tmp_path, old)
    with pytest.raises(T.TrainingError) as refused:
        T.assert_resumable(tmp_path, _fingerprint())
    message = str(refused.value)
    assert "written before ['starting_weights'] were recorded" in message
    assert "started from the base or from an adapter" in message


def test_fresh_does_not_resume_a_by_hand_continued_run_s_checkpoint(client, monkeypatch, tmp_path):
    """The review's case: v9 continued by hand from an adapter that learned a
    deleted document crashes; ``--fresh`` must not resume its checkpoint and
    record no parent. Nor, the other way, does ``--continue-from`` resume a
    fresh run's."""
    from tests.test_stage_data import _seed
    from tests.test_training_resume import _checkpoint

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path)
    adapter = _adapter(tmp_path / "hand")
    staging = Path(paths.scoped_staging_adapter_dir("policy", "v9"))
    monkeypatch.setattr(T, "launch", lambda config: (_ for _ in ()).throw(RuntimeError("GPU lost")))

    with pytest.raises(RuntimeError, match="GPU lost"):
        _train(client, dry_run=False, continue_from=adapter)              # by hand: no parent run
    _checkpoint(staging / "v0-x", 70)
    with pytest.raises(T.TrainingError, match="starting_weights"):
        _train(client, dry_run=False, fresh_reason="asked for (--fresh)")

    shutil.rmtree(staging)
    with pytest.raises(RuntimeError, match="GPU lost"):
        _train(client, dry_run=False, fresh_reason="asked for (--fresh)")
    _checkpoint(staging / "v0-y", 70)
    with pytest.raises(T.TrainingError, match="starting_weights"):
        _train(client, dry_run=False, continue_from=adapter)


# --------------------------------------------------------------------------
# continued_training: only optimization keys, positive where it matters
# --------------------------------------------------------------------------


OPTIMIZATION = {"num_train_epochs": 3, "learning_rate": 1e-4, "warmup_steps": 10}


@pytest.mark.parametrize("block, said", [
    ({"epochs": 1}, "continued_training.epochs is not a key of the optimization block"),
    ({"num_train_epochs": 0}, "continued_training.num_train_epochs must be a positive number"),
    ({"num_train_epochs": 1.5}, "continued_training.num_train_epochs must be a whole number"),
    ({"learning_rate": -1e-5}, "continued_training.learning_rate must be a positive number"),
    ({"learning_rate": "5e-5"}, "YAML reads 5e-5 as text"),
    ({"learning_rate": True}, "continued_training.learning_rate must be a positive number"),
    ([1, 2], "continued_training must map optimization keys to values"),
])
def test_a_continued_training_block_that_would_not_do_what_it_says_is_refused(block, said):
    with pytest.raises(ValueError, match=said):
        T.optimization_for({"optimization": OPTIMIZATION, "continued_training": block}, continuing=True)
    # Refused for a fresh run too: the config is wrong whichever run reads it.
    with pytest.raises(ValueError, match=said):
        T.optimization_for({"optimization": OPTIMIZATION, "continued_training": block}, continuing=False)


def test_a_continued_training_block_of_optimization_keys_is_accepted():
    cfg = {"optimization": OPTIMIZATION,
           "continued_training": {"num_train_epochs": 1, "learning_rate": None, "warmup_steps": 0}}
    assert T.optimization_for(cfg, continuing=True) == {
        "num_train_epochs": 1, "learning_rate": 1e-4, "warmup_steps": 0}
    assert T.optimization_for({"optimization": OPTIMIZATION, "continued_training": None},
                              continuing=True) == OPTIMIZATION


def test_validate_all_refuses_a_bad_continued_training_block_before_any_work(monkeypatch):
    from common import config
    from common.adapter import ADAPTER

    good = config.training_config(ADAPTER.training_config)
    for block, said in (({**good["continued_training"], "epochs": 1}, "continued_training.epochs"),
                        ({"num_train_epochs": 0}, "continued_training.num_train_epochs")):
        broken = {**good, "continued_training": block}
        monkeypatch.setattr(config, "training_config",
                            lambda name, broken=broken: broken if name == ADAPTER.training_config else good)
        with pytest.raises(config.ConfigError, match=said) as refused:
            config.validate_all()
        assert f"configs/training/{ADAPTER.training_config}.yaml" in str(refused.value)
