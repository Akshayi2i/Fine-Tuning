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
    monkeypatch.delenv(vllm_patches.IN_PROCESS_ENV, raising=False)
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


def test_the_site_hook_stops_a_process_whose_needed_fix_cannot_be_applied(monkeypatch):
    """A server that would fail every page image must not start."""
    import importlib.metadata
    import runpy
    from pathlib import Path

    _fake_vllm(monkeypatch, tower="model.something_else.")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.0")
    monkeypatch.setenv("FIDEON_VLLM_PATCH", "1")
    site = Path(vllm_patches.__file__).resolve().parent / "vllm_site" / "sitecustomize.py"
    with pytest.raises(vllm_patches.VllmPatchError):
        runpy.run_path(str(site), run_name="sitecustomize")
