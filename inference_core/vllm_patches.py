"""Fixes for the pinned vLLM, applied where the engine is built.

**Qwen3-VL LoRA over the vision tower (vLLM 0.11.0).** vLLM puts LoRA on the
language model only: it skips every module whose name starts with the model's
``get_mm_mapping()`` tower and connector prefixes. 0.11.0's Qwen3-VL returns
``model.visual.`` for them, the checkpoint's name, while the engine's own modules
are named ``visual.`` (its weight mapper renames them). Nothing matches, the
vision tower's linears are wrapped as LoRA layers, and the first request with an
image fails in ``_lora_shrink``: ``assert token_lora_mapping.size(0) == M`` -
the tower sees image patches, the mapping counts prompt tokens. Text-only
requests never reach the tower, which is how it passed the Phase 0 spike.
0.11.1 corrects the prefixes to ``visual.``; it also moves to torch 2.9, which
the training environment is pinned against, so the correction is applied here.

The engine core must run in this process for a patch here to reach it: vLLM V1
otherwise starts it in a child process, which is spawned (and so re-imports vLLM
unpatched) whenever CUDA is already initialised in the parent.

A served vLLM (``vllm serve``) runs its engine core in a child process, so there
the fix is applied in every process instead: ``inference_core/vllm_site`` holds a
``sitecustomize`` that applies it when Python starts, with ``in_process=False``
(2026-11-10 review, finding 1: production reads every page image through the
adapter, and a stock server did not have the fix).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

#: Setting vLLM reads to run the V1 engine core in the calling process.
IN_PROCESS_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"

#: vLLM versions whose Qwen3-VL mapping points at ``model.visual.``.
_QWEN3_VL_BROKEN = ("0.11.0",)


class VllmPatchError(RuntimeError):
    """Raised when a required vLLM fix could not be applied."""


def _fixed_qwen3_vl_mapping(self):
    from vllm.model_executor.models.module_mapping import MultiModelKeys

    return MultiModelKeys.from_string_field(
        language_model="language_model",
        connector="visual.merger",
        tower_model="visual.",
    )


def patch_qwen3_vl_lora(version: str | None = None, *, in_process: bool = True) -> bool:
    """Point Qwen3-VL's LoRA exclusion at the engine's own module names.

    Returns True when the fix was applied, False when this vLLM does not need
    it. Raises when it is needed and the code it corrects is no longer there.
    ``in_process`` also keeps the engine core in this process (the in-process
    engine); a served vLLM, patched in each of its processes, leaves it alone.
    """
    if version is None:
        from importlib.metadata import version as installed

        version = installed("vllm")
    if version not in _QWEN3_VL_BROKEN:
        return False
    if in_process:
        # Before vLLM is imported: its settings may be read once, at import.
        os.environ[IN_PROCESS_ENV] = "0"
    from vllm.model_executor.models import qwen3_vl

    cls = qwen3_vl.Qwen3VLForConditionalGeneration
    if cls.get_mm_mapping is not _fixed_qwen3_vl_mapping:
        current = cls.get_mm_mapping(None)
        if list(current.tower_model) != ["model.visual."]:
            raise VllmPatchError(
                f"vLLM {version}'s Qwen3-VL LoRA mapping is not the one this fix corrects "
                f"(tower {list(current.tower_model)}); re-check inference_core.vllm_patches."
            )
        cls.get_mm_mapping = _fixed_qwen3_vl_mapping
    log.info("vLLM %s: Qwen3-VL LoRA limited to the language model (engine core in process)",
             version)
    return True
