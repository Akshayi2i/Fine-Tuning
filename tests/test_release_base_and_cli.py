"""A release served on the base it trained on, extract --model vN, and the CLI's early refusals.

Since 2026-10-23 a release IS its adapter, served as a LoRA on the original
base model, so:

* **An adapter runs only on the base it trained on.** Bump the base revision
  pin and an older adapter still loads and answers, with no error, on weights
  it never saw. Promotion, cold start and ``extract --model vN`` refuse it; a
  merged release carries its own base and is not refused.
* **A staged version is the checkpoint its selection chose**, else the latest
  one on the volume - never the ms-swift output root, which holds none.
* **The HF backend can run ``--model vN``**: it binds the version's local
  adapter at load and answers a request with it, or with the base alone.
* **``--fresh`` and ``--from-stage`` are checked before the command detaches**:
  ``--fresh`` on a version already trained, or resuming after training, would
  change nothing; ``merge`` and ``quantize`` are no longer stages.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import run as cli
from registry_utils import query_registry as Q
from registry_utils import write_run_manifest as W
from registry_utils.models import Artifacts
from tests.test_registry import _bundle, _deps, _scoped

#: A base other than the configured one: the revision pin moved since.
OTHER_BASE = "Qwen/Qwen3-VL-8B-Instruct@" + "0" * 40


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _served_pins() -> dict:
    from common.prompts import prompt_hash
    from data_pipeline.dataset_builder.policy_windows import window_plan_id

    return {"prompt_hash": prompt_hash(), "window_plan_version": window_plan_id()}


# --------------------------------------------------------------------------
# Finding 6: a lora_on_base release runs only on the base it trained on
# --------------------------------------------------------------------------


def _candidate(client, version: str, *, base_model: str, served_as: str = "lora_on_base") -> None:
    """A gated candidate release, as package writes it: its bundle, index row
    and passed bf16 gate decision."""
    from registry_utils.releases import write_index

    artifact = (paths.scoped_adapter_dir("policy", version) if served_as == "lora_on_base"
                else paths.merged_model_dir(version, scope="policy"))
    bundle = _bundle(
        release_id=version, adapter=f"policy-{version}", tenant_scope=paths.default_tenant(),
        base_model=base_model, served_as=served_as,
        merged_model="" if served_as == "lora_on_base" else artifact,
        serving_formats={"bf16": artifact}, lines=["homeowners"],
        calibrators={"bf16": paths.release_calibrators(version, "bf16")},
        gate_reports={"bf16": paths.release_gate_decision(version, "bf16")}, **_served_pins(),
    )
    client.write_json(paths.release_bundle(version), json.loads(bundle.model_dump_json()))
    index_key = paths.release_index()
    rows = client.read_json(index_key) if client.exists(index_key) else []
    write_index(client, [r for r in rows if r["release_id"] != version] + [bundle.index_row()], None)
    client.write_json(paths.release_gate_decision(version, "bf16"), {"passed": True, "failed_gates": []})


def test_a_lora_release_trained_on_another_base_is_not_promoted(client):
    from registry_utils.releases import ReleaseError, promote_release

    _candidate(client, "v2", base_model=OTHER_BASE)
    with pytest.raises(ReleaseError, match="serving would refuse it at cold start") as refused:
        promote_release(client, "v2", promoted_by="A. Reviewer")
    message = str(refused.value)
    assert f"release v2 trained on base {OTHER_BASE}" in message
    assert f"configured base is {Q.configured_base_model()}" in message
    assert "an adapter only works on the base it was trained on; retrain (--fresh) or pin the base back" in message

    _candidate(client, "v3", base_model=Q.configured_base_model())
    assert promote_release(client, "v3", promoted_by="A. Reviewer") == []


def test_a_merged_release_carries_its_own_base_and_is_promoted(client):
    from registry_utils.releases import promote_release, serving_refusal

    _candidate(client, "v2", base_model=OTHER_BASE, served_as="merged")
    assert serving_refusal(client.read_json(paths.release_bundle("v2"))) is None
    assert promote_release(client, "v2", promoted_by="A. Reviewer") == []


def _promoted_lora(client, *, base_model: str) -> None:
    """A promoted lora_on_base release v3, on this code's prompt hash and window plan."""
    from tests.test_release_router import promote

    bundle = promote(client, "v3", lines=("homeowners",), base_model=base_model)
    bundle.update(served_as="lora_on_base", merged_model="",
                  serving_formats={"bf16": paths.scoped_adapter_dir("policy", "v3")}, **_served_pins())
    client.write_json(paths.release_bundle("v3"), bundle)


def test_cold_start_refuses_a_lora_release_on_another_base_before_pulling_its_adapter(client, tmp_path):
    from serving.vllm_entrypoint import ColdStartError, cold_start

    _promoted_lora(client, base_model=OTHER_BASE)
    model = SimpleNamespace(config=SimpleNamespace(enable_lora=True))
    with pytest.raises(ColdStartError, match=r"release v3 trained on base .*pin the base back"):
        cold_start("v3", client, model=model, adapter_root=tmp_path)
    assert not (tmp_path / "v3").exists()


def test_a_lora_release_on_the_configured_base_and_an_older_merged_one_are_served(client):
    from serving.release_router import build_serving_plan
    from serving.vllm_entrypoint import assert_base_model_served

    _promoted_lora(client, base_model=Q.configured_base_model())
    assert_base_model_served(build_serving_plan(client))
    # An older merged bundle on another base: its merged model is its own base.
    bundle = client.read_json(paths.release_bundle("v3"))
    merged = paths.merged_model_dir("v3", scope="policy")
    client.write_json(paths.release_bundle("v3"), {
        **bundle, "base_model": OTHER_BASE, "served_as": "merged", "merged_model": merged,
        "serving_formats": {"bf16": merged}})
    assert_base_model_served(build_serving_plan(client))


def test_extract_refuses_a_version_trained_on_another_base(client):
    from inference_core.model_runner import EchoBackend, load_model

    W.write_manifest(_scoped("policy-v3", "policy", ["policy"], dependencies=_deps(base_model=OTHER_BASE),
                             artifacts=Artifacts(status="published",
                                                 adapter_weights=paths.scoped_adapter_dir("policy", "v3"))),
                     client)
    with pytest.raises(Q.RegistryQueryError, match=r"version v3 cannot be loaded: policy-v3 trained on base"):
        Q.resolve_model_version("v3", client)
    with pytest.raises(Q.RegistryQueryError, match="pin the base back"):
        load_model("v3", client, backend_impl=EchoBackend())


# --------------------------------------------------------------------------
# Finding 12: a staged version is its selected checkpoint
# --------------------------------------------------------------------------


def _staged_run(client, tmp_path, monkeypatch) -> Path:
    """A staged policy-v3 and two checkpoints on the volume, as ms-swift leaves
    them; returns the run directory holding them."""
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"], artifacts=Artifacts(
        status="staged", staging_path=paths.scoped_staging_adapter_dir("policy", "v3"))), client)
    run = Path(paths.scoped_staging_adapter_dir("policy", "v3")) / "v0-20261023-120000"
    for step in (100, 200):
        (run / f"checkpoint-{step}").mkdir(parents=True)
        (run / f"checkpoint-{step}" / "adapter_config.json").write_text('{"r": 64}', encoding="utf-8")
    return run


def test_a_staged_version_loads_the_checkpoint_its_selection_chose(client, tmp_path, monkeypatch):
    from inference_core.model_runner import EchoBackend, load_model

    run = _staged_run(client, tmp_path, monkeypatch)
    chosen = (run / "checkpoint-100").as_posix()     # not the latest: the one selected
    client.write_json(paths.checkpoint_selection("v3", scope="policy"), {"selected": chosen})

    assert Q.resolve_model_version("v3", client)["foundation_adapter"] == chosen
    model = load_model("v3", client, backend_impl=EchoBackend())
    assert Path(model.default_adapter) == (run / "checkpoint-100").resolve()


def test_a_staged_version_with_no_selection_loads_its_latest_checkpoint(client, tmp_path, monkeypatch):
    run = _staged_run(client, tmp_path, monkeypatch)
    assert Path(Q.resolve_model_version("v3", client)["foundation_adapter"]) == run / "checkpoint-200"


def test_a_staged_version_with_no_checkpoint_anywhere_is_refused(client, tmp_path, monkeypatch):
    """Not the output root: it holds no adapter, and loading it fails later
    with a message about adapter_config.json that names the wrong problem."""
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"]), client)
    with pytest.raises(Q.RegistryQueryError, match="no checkpoint was selected for it"):
        Q.resolve_model_version("v3", client)


# --------------------------------------------------------------------------
# Finding 16: the HF backend runs --model vN
# --------------------------------------------------------------------------


class _Ids(list):
    """Token ids with the one tensor attribute generate reads."""

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def __getitem__(self, index):
        got = list.__getitem__(self, index)
        return _Ids(got) if isinstance(index, slice) else got


class _PeftModel:
    """Stands in for a PeftModel holding the version's adapter."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[bool] = []
        self._off = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self._off = True
        try:
            yield
        finally:
            self._off = False

    def generate(self, **_kw):
        self.calls.append(not self._off)          # whether the adapter answered
        return SimpleNamespace(sequences=[_Ids([1, 2, 3, 9])], scores=None)


class _Processor:
    def apply_chat_template(self, *_a, **_kw):
        return SimpleNamespace(to=lambda device: {"input_ids": _Ids([1, 2, 3])})

    def decode(self, _ids, skip_special_tokens=True):
        return "{}"


def test_the_hf_backend_binds_the_versions_local_adapter_and_answers_with_it_or_the_base(
        client, tmp_path, monkeypatch):
    from inference_core import model_runner
    from inference_core.model_runner import NO_ADAPTER, HFBackend, ModelRunnerError, generate, load_model

    monkeypatch.setattr(model_runner, "ADAPTER_CACHE", tmp_path / "cache")
    blob = paths.scoped_adapter_dir("policy", "v3")
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"], artifacts=Artifacts(
        status="published", adapter_weights=blob)), client)
    client.write_json(f"{blob}/adapter_config.json", {"r": 64, "lora_alpha": 128})
    client.write_bytes(f"{blob}/adapter_model.safetensors", b"weights")

    model = load_model("v3", client, backend="hf")
    backend = model.backend
    assert isinstance(backend, HFBackend)
    # Bound at load: the local copy, never the Blob prefix PEFT cannot open.
    assert backend.loaded_adapters() == [model.default_adapter]
    assert Path(model.default_adapter, "adapter_config.json").is_file()

    with pytest.raises(ModelRunnerError, match="binds adapters at load time"):
        generate(model, [], adapter="/adapters/another")

    peft = _PeftModel()
    backend._model, backend._processor = peft, _Processor()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=contextlib.nullcontext))
    generate(model, [], want_logprobs=False)                         # its adapter
    generate(model, [], adapter=model.default_adapter, want_logprobs=False)
    generate(model, [], adapter=NO_ADAPTER, want_logprobs=False)     # the base alone
    assert peft.calls == [True, True, False]


# --------------------------------------------------------------------------
# Finding 7: --fresh means nothing for a version that already trained
# --------------------------------------------------------------------------


def _args(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def _no_detach(monkeypatch) -> None:
    import orchestration.detach as detach

    monkeypatch.setattr(detach, "detach_module_if_needed",
                        lambda *a, **k: pytest.fail("detached before refusing"))


@pytest.mark.parametrize("stage", ["checkpoint_eval", "calibrate", "evaluation_gate", "package"])
def test_fresh_resuming_after_training_is_refused_before_the_detach(stage, monkeypatch, capsys):
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: pytest.fail("reached Blob"))
    _no_detach(monkeypatch)
    with pytest.raises(SystemExit) as exited:
        cli.main(["all", "--version", "v4", "--from-stage", stage, "--fresh"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert f"--from-stage {stage} resumes after training" in err
    assert "--fresh applies to a new cycle: run finetune/all without --version" in err


def test_fresh_resuming_at_training_or_before_is_left_to_the_run_manifest():
    for stage in ("ingestion", "dataset_build", "training"):
        assert cli.flag_problem(_args("finetune", "--version", "v4", "--from-stage", stage, "--fresh")) is None


def test_fresh_on_a_version_that_already_trained_continued_is_refused(client, monkeypatch, capsys):
    W.write_manifest(_scoped("policy-v4", "policy", ["policy"], parent_run_id="policy-v3",
                             continued_from=paths.staging_parent_adapter_dir("policy", "v4")), client)
    problem = cli.fresh_problem(_args("all", "--version", "v4", "--fresh"), client)
    assert problem == ("v4 already trained (continued from policy-v3); --fresh applies to a new cycle: "
                       "run finetune/all without --version. To resume v4 as it trained, drop --fresh")

    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: client)
    _no_detach(monkeypatch)
    with pytest.raises(SystemExit) as exited:
        cli.main(["all", "--version", "v4", "--fresh"])
    assert exited.value.code == 2 and "v4 already trained (continued from policy-v3)" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["evaluated", "promoted", "archived"])
def test_fresh_on_a_version_past_training_is_refused(client, status):
    from datetime import UTC, datetime

    from registry_utils.models import Promotion

    W.write_manifest(_scoped("policy-v4", "policy", ["policy"], status=status, fresh_reason="asked for (--fresh)",
                             promotion=Promotion(promoted_at=datetime.now(UTC), promoted_by="A. Reviewer")),
                     client)
    problem = cli.fresh_problem(_args("finetune", "--version", "v4", "--fresh"), client)
    assert problem and problem.startswith("v4 already trained (from the base: asked for (--fresh))")


def test_fresh_on_a_version_whose_training_is_complete_on_the_volume_is_refused(client, tmp_path, monkeypatch):
    from training.train import mark_training_complete

    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    mark_training_complete(paths.scoped_staging_adapter_dir("policy", "v4"), "policy-v4")
    problem = cli.fresh_problem(_args("all", "--version", "v4", "--fresh"), client)
    assert problem and "its training stage is complete on the staging volume" in problem


@pytest.mark.parametrize("status", [None, "training", "failed"])
def test_fresh_on_a_version_that_has_not_trained_is_accepted(client, tmp_path, monkeypatch, status):
    """No run yet, one still training (or interrupted), or one that failed:
    training runs, and --fresh decides where it starts."""
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    if status:
        W.write_manifest(_scoped("policy-v4", "policy", ["policy"], status=status), client)
    assert cli.fresh_problem(_args("all", "--version", "v4", "--fresh"), client) is None


def test_without_fresh_or_a_version_nothing_is_read(monkeypatch):
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: pytest.fail("reached Blob"))
    assert cli.fresh_problem(_args("all", "--version", "v4")) is None
    assert cli.fresh_problem(_args("all", "--fresh")) is None          # a new cycle: the next version
    assert cli.fresh_problem(_args("package", "--version", "v4")) is None


# --------------------------------------------------------------------------
# Finding 15: --from-stage is checked before the detach
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["merge", "quantize"])
@pytest.mark.parametrize("command", ["finetune", "all"])
def test_a_removed_stage_is_refused_before_the_detach(command, stage, monkeypatch, capsys):
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: pytest.fail("reached Blob"))
    _no_detach(monkeypatch)
    with pytest.raises(SystemExit) as exited:
        cli.main([command, "--version", "v3", "--from-stage", stage])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert (f"--from-stage {stage}: the {stage} stage was removed on 2026-10-23: nothing is merged or "
            "quantized; the release is its adapter") in err


def test_a_stage_the_command_does_not_run_is_refused_with_the_ones_it_does():
    problem = cli.flag_problem(_args("finetune", "--version", "v3", "--from-stage", "calibrate"))
    assert problem and "'calibrate' is not a stage of finetune" in problem and "checkpoint_eval" in problem
    assert "not a stage of all" in cli.flag_problem(_args("all", "--version", "v3", "--from-stage", "trian"))
    assert cli.flag_problem(_args("all", "--version", "v3", "--from-stage", "calibrate")) is None
