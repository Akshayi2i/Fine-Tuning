"""The Qwen3-VL LoRA fix for vLLM 0.11.0, against a stand-in vLLM."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from inference_core import vllm_patches


@dataclass
class _Keys:
    language_model: list = field(default_factory=list)
    connector: list = field(default_factory=list)
    tower_model: list = field(default_factory=list)

    @staticmethod
    def from_string_field(language_model=None, connector=None, tower_model=None):
        as_list = lambda v: [v] if isinstance(v, str) else list(v or [])  # noqa: E731
        return _Keys(as_list(language_model), as_list(connector), as_list(tower_model))


def _fake_vllm(monkeypatch, tower="model.visual."):
    class Qwen3VL:
        def get_mm_mapping(self):
            return _Keys.from_string_field("language_model", tower + "merger", tower)

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.models": types.ModuleType("vllm.model_executor.models"),
        "vllm.model_executor.models.qwen3_vl": types.SimpleNamespace(
            Qwen3VLForConditionalGeneration=Qwen3VL),
        "vllm.model_executor.models.module_mapping": types.SimpleNamespace(MultiModelKeys=_Keys),
    }
    modules["vllm.model_executor.models"].qwen3_vl = modules["vllm.model_executor.models.qwen3_vl"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv(vllm_patches.IN_PROCESS_ENV, "")      # so the value a test leaves is undone (see _clean_env)
    monkeypatch.delenv(vllm_patches.IN_PROCESS_ENV)
    return Qwen3VL


def test_the_vision_tower_is_excluded_by_the_engines_own_module_names(monkeypatch):
    """0.11.0 names the tower `model.visual.`; the engine's modules are `visual.`,
    so nothing was excluded and an image request failed in _lora_shrink."""
    cls = _fake_vllm(monkeypatch)
    assert vllm_patches.patch_qwen3_vl_lora("0.11.0") is True
    keys = cls().get_mm_mapping()
    assert keys.tower_model == ["visual."] and keys.connector == ["visual.merger"]
    assert keys.language_model == ["language_model"]


def test_the_engine_core_runs_in_process_so_the_fix_reaches_it(monkeypatch):
    import os

    _fake_vllm(monkeypatch)
    vllm_patches.patch_qwen3_vl_lora("0.11.0")
    assert os.environ[vllm_patches.IN_PROCESS_ENV] == "0"


def test_applying_it_twice_is_harmless(monkeypatch):
    cls = _fake_vllm(monkeypatch)
    vllm_patches.patch_qwen3_vl_lora("0.11.0")
    assert vllm_patches.patch_qwen3_vl_lora("0.11.0") is True
    assert cls().get_mm_mapping().tower_model == ["visual."]


def test_a_vllm_without_the_bug_is_left_alone(monkeypatch):
    import os

    cls = _fake_vllm(monkeypatch, tower="visual.")
    assert vllm_patches.patch_qwen3_vl_lora("0.11.1") is False
    assert vllm_patches.IN_PROCESS_ENV not in os.environ
    assert cls.get_mm_mapping is not vllm_patches._fixed_qwen3_vl_mapping


def test_unexpected_code_under_the_pinned_version_is_refused(monkeypatch):
    _fake_vllm(monkeypatch, tower="something.else.")
    with pytest.raises(vllm_patches.VllmPatchError, match="re-check"):
        vllm_patches.patch_qwen3_vl_lora("0.11.0")


def test_the_pinned_vllm_is_the_one_the_fix_targets():
    """Moving the pin off 0.11.0 means re-checking whether this fix is still needed."""
    from pathlib import Path

    lock = Path(__file__).resolve().parents[1] / "requirements-train.lock"
    pinned = [line.split("==")[1].strip() for line in lock.read_text().splitlines()
              if line.startswith("vllm==")]
    assert pinned and pinned[0] in vllm_patches._QWEN3_VL_BROKEN


# --------------------------------------------------------------------------
# Image parts, in the shape vLLM's chat parser accepts
# --------------------------------------------------------------------------


def _page(tmp_path):
    from PIL import Image

    path = tmp_path / "page_1.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return path


def test_ms_swift_image_parts_become_images_vllm_can_read(tmp_path):
    """vLLM refused `{"type": "image"}` ("Unknown part type: image") on every row,
    so checkpoint selection scored nothing."""
    from PIL import Image

    from inference_core.model_runner import to_vllm_messages

    path = _page(tmp_path)
    messages = [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": [{"type": "image", "image": str(path)},
                                     {"type": "image", "image": path.read_bytes()},
                                     {"type": "text", "text": "OCR text"}]},
    ]
    out = to_vllm_messages(messages)
    parts = out[1]["content"]
    assert [p["type"] for p in parts] == ["image_pil", "image_pil", "text"]
    assert all(isinstance(p["image_pil"], Image.Image) for p in parts[:2])
    assert out[0] == messages[0] and messages[1]["content"][0]["type"] == "image"   # input untouched


def test_an_image_url_stays_a_url():
    from inference_core.model_runner import to_vllm_messages

    out = to_vllm_messages([{"role": "user", "content": [
        {"type": "image", "image": "https://example.com/p.png"}]}])
    assert out[0]["content"][0] == {"type": "image_url", "image_url": {"url": "https://example.com/p.png"}}


def test_a_missing_page_is_an_error_not_a_silent_skip(tmp_path):
    from inference_core.model_runner import to_vllm_messages

    with pytest.raises(FileNotFoundError):
        to_vllm_messages([{"role": "user", "content": [
            {"type": "image", "image": str(tmp_path / "gone.png")}]}])


def test_the_vllm_backend_sends_converted_messages(tmp_path):
    from inference_core import model_runner as M

    seen = []

    class Engine:
        def chat(self, messages, params, lora_request=None):
            seen.append(messages)
            return []

    backend = M.VLLMBackend.__new__(M.VLLMBackend)
    backend._load = lambda config: Engine()
    backend._sampling_params = lambda config: None
    backend._lora = lambda adapter: None
    backend._to_generation = lambda output, config, latency: output
    msgs = [{"role": "user", "content": [{"type": "image", "image": str(_page(tmp_path))}]}]
    backend.generate(msgs, config=None)
    backend.generate_batch([msgs], [None])
    assert seen[0][0]["content"][0]["type"] == "image_pil"
    assert seen[1][0][0]["content"][0]["type"] == "image_pil"


# --------------------------------------------------------------------------
# Structured decoding without free whitespace, on xgrammar
# --------------------------------------------------------------------------


def _xgrammar_unsupported(obj) -> bool:
    """vLLM 0.11.0's has_xgrammar_unsupported_json_features, restated. Under
    backend "auto" such a schema falls back to another backend; with xgrammar
    named, it is refused instead."""
    if not isinstance(obj, dict):
        return False
    kind = obj.get("type")
    if kind in ("integer", "number") and "multipleOf" in obj:
        return True
    if kind == "array" and any(k in obj for k in ("uniqueItems", "contains", "minContains",
                                                  "maxContains")):
        return True
    if kind == "string" and "format" in obj:
        return True
    if kind == "object" and any(k in obj for k in ("minProperties", "maxProperties",
                                                   "propertyNames", "patternProperties")):
        return True
    for value in obj.values():
        items = value if isinstance(value, list) else [value]
        if any(_xgrammar_unsupported(item) for item in items):
            return True
    return False


def test_every_policy_schema_we_constrain_to_is_one_xgrammar_supports():
    """What decoding is handed is resolved_schema (serving.pipeline), not the
    client's file: for a common-model line the two differ by exactly the
    keywords xgrammar refuses, which the model view drops."""
    from common import schemas

    selectors = [sel for sel in schemas.schema_selectors() if sel[0] == "policy"]
    assert ("policy", None, None) in selectors and len(selectors) > 5
    unsupported = [sel for sel in selectors if _xgrammar_unsupported(schemas.resolved_schema(*sel))]
    assert not unsupported, f"xgrammar cannot compile {unsupported}"


@pytest.mark.xfail(strict=True, reason=(
    "Found 2026-10-06 when this check moved from the raw files to what decoding is handed: "
    "the flat ACORD and Loss Run schemas inline line_of_business from schemas/, an array "
    "with uniqueItems, which vLLM 0.11 refuses under the xgrammar backend. Not yet fixed: "
    "changing it moves those prompts."))
def test_every_flat_schema_we_constrain_to_is_one_xgrammar_supports():
    from common import schemas

    selectors = [sel for sel in schemas.schema_selectors() if sel[0] != "policy"]
    unsupported = [sel for sel in selectors if _xgrammar_unsupported(schemas.resolved_schema(*sel))]
    assert not unsupported, f"xgrammar cannot compile {unsupported}"


def test_the_engine_forbids_free_whitespace_on_a_backend_that_honours_it():
    from inference_core.model_runner import STRUCTURED_OUTPUTS_ENGINE

    assert STRUCTURED_OUTPUTS_ENGINE == {"backend": "xgrammar", "disable_any_whitespace": True}


# --------------------------------------------------------------------------
# page_ref bounded to the document's pages
# --------------------------------------------------------------------------


def test_page_refs_are_held_to_the_documents_pages():
    """A smoke-run answer counted page_ref on to 1528 and hit max_new_tokens."""
    from common.schemas import resolved_schema, with_page_bounds

    base = resolved_schema("policy", None, "homeowners", None)
    bounded = with_page_bounds(base, 12)
    ref = bounded["$defs"]["FieldValue"]["properties"]["page_ref"]
    assert ref["maxItems"] == 12
    assert ref["items"] == {"type": "integer", "minimum": 1, "maximum": 12}
    assert "maxItems" not in base["$defs"]["FieldValue"]["properties"]["page_ref"]  # cache untouched
    assert with_page_bounds(base, None) is base


def test_a_window_may_cite_only_its_own_pages():
    from common.schemas import resolved_schema, with_page_bounds

    base = resolved_schema("policy", None, "homeowners", "arrays")

    def ref(pages):
        return with_page_bounds(base, 40, pages=pages)["$defs"]["FieldValue"]["properties"]["page_ref"]

    assert ref([7, 5, 6])["items"] == {"type": "integer", "minimum": 5, "maximum": 7} and ref([5, 6, 7])["maxItems"] == 3
    assert ref([2, 9, 14])["items"] == {"type": "integer", "enum": [2, 9, 14]}
    assert ref(list(range(1, 80, 2)))["items"] == {"type": "integer", "minimum": 1, "maximum": 79}  # too long to list
    assert not _xgrammar_unsupported(with_page_bounds(base, 40, pages=[2, 9, 14]))


def test_the_pages_a_prompt_shows_are_read_from_its_markers():
    from inference_core.input_builder import shown_pages

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "<page 10 of 20>\n..."}, {"type": "text", "text": "<page 9 of 20>\nDECLARATIONS"}]}]
    assert shown_pages(messages) == [9, 10]
    assert shown_pages([{"role": "user", "content": "no markers"}]) == []


def test_the_bounded_schema_is_one_xgrammar_compiles():
    from common.schemas import resolved_schema, with_page_bounds

    assert not _xgrammar_unsupported(with_page_bounds(resolved_schema("policy", None, None, None), 7))


def test_the_page_count_is_read_from_the_prompts_markers():
    from inference_core.input_builder import page_total

    messages = [{"role": "user", "content": [
        {"type": "image", "image": "a"}, {"type": "text", "text": "<page 9 of 20>\nDECLARATIONS"},
        {"type": "text", "text": "<page 10 of 20>\n..."}]}]
    assert page_total(messages) == 20
    assert page_total([{"role": "user", "content": "no markers"}]) is None


def test_validation_generation_constrains_with_the_bound(monkeypatch):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation.validation_generation import generate_validation
    from inference_core import model_runner as M

    seen = []
    backend = M.EchoBackend('{"a":1}')
    real = backend.generate_batch
    backend.generate_batch = lambda msgs, configs, adapter=None: (
        seen.extend(c.json_schema for c in configs), real(msgs, configs, adapter))[1]
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = M.load_model("base", client, backend_impl=backend)
    row = {"source_id": "p1", "doc_type": "policy", "lob": "homeowners", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "<page 1 of 3>\ntext"}]},
        {"role": "assistant", "content": '{"a": 1}'}]}
    generate_validation([row], model, constrain=True)
    ref = seen[0]["$defs"]["FieldValue"]["properties"]["page_ref"]
    assert ref["maxItems"] == 1 and ref["items"]["maximum"] == 1       # it shows page 1 of 3


def test_repeatable_generation_turns_off_the_prefix_cache_and_asks_for_invariant_kernels(monkeypatch):
    import dataclasses

    from inference_core.model_runner import repeatable_engine_settings
    from inference_core.runner_config import RunnerConfig

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    assert repeatable_engine_settings(RunnerConfig()) == {}
    import os

    assert "VLLM_BATCH_INVARIANT" not in os.environ
    config = dataclasses.replace(RunnerConfig(), repeatable=True)
    assert repeatable_engine_settings(config) == {"enable_prefix_caching": False}
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
    assert config.fingerprint() != RunnerConfig().fingerprint()       # two runs that differ in it say so


def test_the_shipped_config_generates_with_the_prefix_cache():
    from inference_core.runner_config import load_runner_config

    assert load_runner_config().repeatable is False


# -- a served vLLM: the fix applied as each process starts (2026-11-10 review, finding 1) ------------------------


def test_a_served_vllm_is_patched_without_forcing_its_engine_into_one_process(monkeypatch):
    import os

    cls = _fake_vllm(monkeypatch)
    assert vllm_patches.patch_qwen3_vl_lora("0.11.0", in_process=False) is True
    assert cls().get_mm_mapping().tower_model == ["visual."]
    assert vllm_patches.IN_PROCESS_ENV not in os.environ


def _run_site(monkeypatch, flag):
    import runpy
    from pathlib import Path

    site = Path(vllm_patches.__file__).resolve().parent / "vllm_site" / "sitecustomize.py"
    monkeypatch.setattr(vllm_patches, "patch_qwen3_vl_lora",
                        lambda version=None, in_process=True: calls.append(in_process) or True)
    calls: list = []
    if flag is None:
        monkeypatch.delenv("FIDEON_VLLM_PATCH", raising=False)
    else:
        monkeypatch.setenv("FIDEON_VLLM_PATCH", flag)
    runpy.run_path(str(site), run_name="sitecustomize")
    return calls


def test_the_site_hook_applies_the_fix_in_a_process_started_with_the_flag(monkeypatch):
    assert _run_site(monkeypatch, "1") == [False]          # patched, the engine core left in its own process


def test_the_site_hook_does_nothing_without_the_flag(monkeypatch):
    assert _run_site(monkeypatch, None) == [] and _run_site(monkeypatch, "0") == []


def _python_with_hook(tmp_path, fake_patches: str, code: str, **env):
    """A real Python process started with the hook on its path and a stand-in ``vllm_patches`` beside it (the
    serving-pod layout): Python runs ``sitecustomize`` itself, as it does for vLLM's engine core."""
    import os
    import shutil
    import subprocess
    from pathlib import Path

    shutil.copy(Path(vllm_patches.__file__).resolve().parent / "vllm_site" / "sitecustomize.py", tmp_path)
    (tmp_path / "vllm_patches.py").write_text(fake_patches, encoding="utf-8")
    base = {k: v for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "FIDEON_VLLM_PATCH", "FIDEON_VLLM_PARENT_PID")}
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                          env={**base, "PYTHONPATH": str(tmp_path), **env}, check=False)


_FAILING = "def patch_qwen3_vl_lora(version=None, *, in_process=True):\n    raise RuntimeError('not the code it fixes')\n"
_WORKING = ("import os\ndef patch_qwen3_vl_lora(version=None, *, in_process=True):\n"
            "    os.environ['PATCHED'] = str(in_process)\n    return True\n")


def test_the_site_hook_stops_a_process_whose_needed_fix_cannot_be_applied(tmp_path):
    """A server or engine that would fail every page image must not start. Python on its own only prints an error
    raised in sitecustomize and runs on (exit 0, checked 2026-11-10), so the hook ends the process itself."""
    done = _python_with_hook(tmp_path, _FAILING, "print('started anyway')", FIDEON_VLLM_PATCH="1")
    assert done.returncode == 1 and "started anyway" not in done.stdout
    assert "could not be applied" in done.stderr and "not the code it fixes" in done.stderr


def test_a_process_started_with_the_hook_is_patched_and_runs(tmp_path):
    code = "import os; print(os.environ.get('PATCHED'))"
    done = _python_with_hook(tmp_path, _WORKING, code, FIDEON_VLLM_PATCH="1")
    assert done.returncode == 0 and done.stdout.strip() == "False"      # patched, its engine core left apart
    assert _python_with_hook(tmp_path, _WORKING, code).stdout.strip() == "None"


def test_only_a_direct_child_of_the_named_process_watches_for_its_parent(tmp_path):
    import os

    code = "import threading; print(any(t.name == 'fideon-exit-with-parent' for t in threading.enumerate()))"
    child = _python_with_hook(tmp_path, _WORKING, code, FIDEON_VLLM_PATCH="1",
                              FIDEON_VLLM_PARENT_PID=str(os.getpid()))
    assert child.stdout.strip() == "True"
    other = _python_with_hook(tmp_path, _WORKING, code, FIDEON_VLLM_PATCH="1", FIDEON_VLLM_PARENT_PID="1")
    assert other.stdout.strip() == "False"       # a grandchild (another parent from its start) is left alone


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does not re-parent an orphan, so its parent id stays")
def test_an_engine_core_ends_itself_when_its_parent_is_gone(tmp_path):
    """vLLM 0.11.0 does not end its engine core when the parent goes (a stall stop, a crash), and a core left
    behind keeps the GPU's memory from the run that resumes."""
    import os
    import time
    from pathlib import Path

    # The middle process starts a child with the hook, naming itself as the parent, prints the child's id and exits.
    middle = ("import os, subprocess, sys, time\n"
              "env = dict(os.environ, FIDEON_VLLM_PATCH='1', FIDEON_VLLM_PARENT_PID=str(os.getpid()))\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], env=env)\n"
              "print(child.pid, flush=True)\n"
              "time.sleep(1)\n")
    pid = int(_python_with_hook(tmp_path, _WORKING, middle).stdout.strip())
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split()[2] == "Z":     # gone, or ended and not yet reaped
            return
        time.sleep(0.5)
    os.kill(pid, 9)
    pytest.fail("the child outlived its parent by 20 s")


# -- the in-process engine with its engine core in its own process (2026-11-10, speed) --------------------------


def _clean_env(monkeypatch):
    # setenv first: monkeypatch records nothing for a variable absent when it is deleted, so one the code under test
    # sets afterwards would outlive the test (it did: later tests' processes started with the hook switched on).
    for name in ("PYTHONPATH", vllm_patches.PATCH_ENV, vllm_patches.PARENT_ENV, vllm_patches.IN_PROCESS_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_an_engine_core_in_its_own_process_is_patched_as_it_starts(monkeypatch):
    import os
    from pathlib import Path

    cls = _fake_vllm(monkeypatch)
    _clean_env(monkeypatch)
    assert vllm_patches.patch_for_engine_process("0.11.0") is True
    assert cls().get_mm_mapping().tower_model == ["visual."]      # this process too (a forked child inherits it)
    assert os.environ[vllm_patches.IN_PROCESS_ENV] == "1"
    assert os.environ[vllm_patches.PATCH_ENV] == "1" and os.environ[vllm_patches.PARENT_ENV] == str(os.getpid())
    first, root = os.environ["PYTHONPATH"].split(os.pathsep)[:2]
    assert Path(first) == vllm_patches.SITE_DIR and (Path(root) / "inference_core").is_dir()
    # A second engine in the same run adds nothing to the path.
    before = os.environ["PYTHONPATH"]
    vllm_patches.patch_for_engine_process("0.11.0")
    assert os.environ["PYTHONPATH"] == before
    # A real new process finds this hook first (no vLLM loaded: the flag is left out of the probe).
    assert Path(vllm_patches.site_hook_found()).resolve() == (vllm_patches.SITE_DIR / "sitecustomize.py").resolve()


def test_another_sitecustomize_found_first_is_refused(monkeypatch):
    _fake_vllm(monkeypatch)
    _clean_env(monkeypatch)
    monkeypatch.setattr(vllm_patches, "site_hook_found", lambda: "/usr/lib/python3.11/sitecustomize.py")
    with pytest.raises(vllm_patches.VllmPatchError, match="without the Qwen3-VL LoRA fix"):
        vllm_patches.patch_for_engine_process("0.11.0")


def test_a_vllm_without_the_bug_runs_its_engine_core_apart_with_no_hook(monkeypatch):
    import os

    _fake_vllm(monkeypatch, tower="visual.")
    _clean_env(monkeypatch)
    assert vllm_patches.patch_for_engine_process("0.11.1") is False
    assert os.environ[vllm_patches.IN_PROCESS_ENV] == "1" and vllm_patches.PATCH_ENV not in os.environ


def test_the_shipped_config_runs_the_engine_core_in_its_own_process():
    from inference_core.runner_config import RunnerConfig, load_runner_config

    assert load_runner_config().engine_in_own_process is True
    assert RunnerConfig().engine_in_own_process is False
    # It changes no setting of generation, so results stay comparable across the switch.
    assert RunnerConfig(engine_in_own_process=True).fingerprint() == RunnerConfig().fingerprint()


@pytest.mark.parametrize("own_process", [True, False])
def test_closing_ends_the_engine_core_process_only_when_it_has_one(own_process):
    from inference_core.model_runner import VLLMBackend
    from inference_core.runner_config import RunnerConfig

    stopped = []
    backend = VLLMBackend({"tag": "v3"}, RunnerConfig())
    backend._engine = types.SimpleNamespace(llm_engine=types.SimpleNamespace(
        engine_core=types.SimpleNamespace(shutdown=lambda: stopped.append(True))))
    backend._own_process = own_process
    backend.close()
    assert backend._engine is None and stopped == ([True] if own_process else [])


def test_a_stall_stop_ends_the_engine_core_process_too(monkeypatch):
    import multiprocessing

    from inference_core import stall_guard

    killed, exits = [], []
    child = types.SimpleNamespace(kill=lambda: killed.append(True))
    monkeypatch.setattr(multiprocessing, "active_children", lambda: [child])
    monkeypatch.setattr(stall_guard.faulthandler, "dump_traceback", lambda **k: None)
    monkeypatch.setattr("orchestration.notify.stalled", lambda reason: None)
    monkeypatch.setattr(stall_guard.os, "_exit", exits.append)
    stall_guard.abort("test")
    assert killed == [True] and exits == [stall_guard.EXIT_CODE]
