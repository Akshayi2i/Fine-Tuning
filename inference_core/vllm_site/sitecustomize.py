"""The Qwen3-VL LoRA fix for a served vLLM 0.11.0, applied as each of its Python processes starts.

``inference_core.vllm_patches`` fixes vLLM 0.11.0's Qwen3-VL LoRA mapping, without which the first request carrying
a page image fails in ``_lora_shrink``. The in-process engine applies it itself; a served vLLM (``vllm serve``) runs
its engine core in a child process, which a patch in the parent does not reach when the child is spawned. Python
imports ``sitecustomize`` at the start of every process whose path holds it, so the fix is in each one before vLLM
loads the model (2026-11-10 review, finding 1).

Started as:

    PYTHONPATH=<this folder>:<the repository> FIDEON_VLLM_PATCH=1 vllm serve ...

On a serving pod without the repository, copy this file and ``inference_core/vllm_patches.py`` into one folder and
put that folder on ``PYTHONPATH``. Without ``FIDEON_VLLM_PATCH=1`` nothing is done, so the folder on the path of any
other process changes nothing. A fix that is needed and cannot be applied stops the process: a server that would
fail every page image must not start.
"""

import os

if os.environ.get("FIDEON_VLLM_PATCH") == "1":
    try:
        from inference_core import vllm_patches
    except ImportError:  # a serving pod without the repository: vllm_patches.py copied beside this file
        import vllm_patches  # type: ignore[no-redef]
    vllm_patches.patch_qwen3_vl_lora(in_process=False)
