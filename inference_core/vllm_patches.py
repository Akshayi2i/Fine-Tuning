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

A patch here reaches only this process: vLLM V1 starts its engine core in a child
process unless told not to, and that child is spawned (and so re-imports vLLM
unpatched) whenever CUDA is already initialised in the parent. Two ways:

* :func:`patch_qwen3_vl_lora` keeps the engine core in this process
  (``serving.engine_in_own_process: false``) - the GPU then waits while every
  request of a call is prepared;
* :func:`patch_for_engine_process` lets it run in its own process and has the
  child patched as it starts (``serving.engine_in_own_process: true``, the
  shipped setting since 2026-11-10) - requests are prepared while it generates.

A served vLLM (``vllm serve``) runs its engine core in a child process, so there
the fix is applied in every process instead: ``inference_core/vllm_site`` holds a
``sitecustomize`` that applies it when Python starts, with ``in_process=False``
(2026-11-10 review, finding 1: production reads every page image through the
adapter, and a stock server did not have the fix).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

#: Setting vLLM reads to run the V1 engine core in the calling process.
IN_PROCESS_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"

#: The folder whose ``sitecustomize`` applies the fix as a process starts, the flag that turns it on, and the
#: process whose direct children end themselves when it is gone (``vllm_site/sitecustomize.py``).
SITE_DIR = Path(__file__).resolve().parent / "vllm_site"
PATCH_ENV = "FIDEON_VLLM_PATCH"
PARENT_ENV = "FIDEON_VLLM_PARENT_PID"

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
    log.info("vLLM %s: Qwen3-VL LoRA limited to the language model (engine core %s)",
             version, "in process" if in_process else "in its own process")
    return True


def patch_for_engine_process(version: str | None = None) -> bool:
    """The fix for an in-process engine whose engine core runs in a process of its own.

    Why: vLLM 0.11.0's ``LLM.chat`` prepares every request of a call (page images through the processor, about 2 s
    each on the H200) before the first is generated, and with the engine core in this process the GPU waits through
    all of it - about 200 s of a 96-request call (2026-11-09). With the engine core in its own process each request
    is sent as soon as it is prepared and the engine starts on it while the next is prepared
    (``vllm/v1/engine/core.py`` ``run_busy_loop``; ``LLMEngine.add_request`` prepares in this process).

    The fix is applied here (a forked child inherits it) and, for a spawned child, by
    ``inference_core/vllm_site/sitecustomize.py``: this puts that folder and the repository on ``PYTHONPATH``, sets
    ``FIDEON_VLLM_PATCH=1`` and this process's id, which the engine core reads when it starts. Refused when another
    ``sitecustomize`` would be found first, since the child would then start without the fix. Returns what
    :func:`patch_qwen3_vl_lora` returns; a vLLM without the bug needs no hook.
    """
    applied = patch_qwen3_vl_lora(version, in_process=False)
    # Before vLLM is imported: its settings may be read once, at import. An earlier in-process engine set it to 0.
    os.environ[IN_PROCESS_ENV] = "1"
    if not applied:
        return False
    hook = SITE_DIR / "sitecustomize.py"
    # One run loads several engines in turn (selection, calibration, the gate): the folders go in once.
    ours = [str(SITE_DIR), str(SITE_DIR.parents[1])]
    rest = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p not in ours]
    os.environ["PYTHONPATH"] = os.pathsep.join(ours + rest)
    os.environ[PATCH_ENV] = "1"
    os.environ[PARENT_ENV] = str(os.getpid())
    found = site_hook_found()
    if found is None or Path(found).resolve() != hook.resolve():
        raise VllmPatchError(
            f"a new Python process finds sitecustomize at {found}, not {hook}: the engine core would start without "
            "the Qwen3-VL LoRA fix. Remove the other one, or set serving.engine_in_own_process: false."
        )
    return True


def site_hook_found() -> str | None:
    """The ``sitecustomize`` a new Python process started with this environment imports, or None.

    Run without ``FIDEON_VLLM_PATCH``, so it does not load vLLM: under a second.
    """
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in (PATCH_ENV, PARENT_ENV)}
    probe = "import sys\ntry:\n import sitecustomize\nexcept ImportError:\n sys.exit(3)\nsys.stdout.write(sitecustomize.__file__ or '')"
    done = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=60,
                          check=False)
    return (done.stdout.strip() or None) if done.returncode == 0 else None
