"""The Qwen3-VL LoRA fix for a vLLM 0.11.0 whose engine runs in its own process, applied as each Python process starts.

``inference_core.vllm_patches`` fixes vLLM 0.11.0's Qwen3-VL LoRA mapping, without which the first request carrying
a page image fails in ``_lora_shrink``. A served vLLM (``vllm serve``), and the in-process engine with its engine
core in a process of its own (``serving.engine_in_own_process``), run the engine core in a child process, which a
patch in the parent does not reach when the child is spawned. Python imports ``sitecustomize`` at the start of every
process whose path holds it, so the fix is in each one before vLLM loads the model (2026-11-10 review, finding 1).

Started as:

    PYTHONPATH=<this folder>:<the repository> FIDEON_VLLM_PATCH=1 vllm serve ...

On a serving pod without the repository, copy this file and ``inference_core/vllm_patches.py`` into one folder and
put that folder on ``PYTHONPATH``. Without ``FIDEON_VLLM_PATCH=1`` nothing is done, so the folder on the path of any
other process changes nothing.

A fix that is needed and cannot be applied ends the process, exit status 1, with the reason on stderr: a server or
engine that would fail every page image must not start. Python itself only prints an error raised here and carries
on ("Error in sitecustomize"), so the exit is explicit.

``FIDEON_VLLM_PARENT_PID`` (set by ``vllm_patches.patch_for_engine_process``): a process whose parent is that
process ends itself when the parent is gone. vLLM 0.11.0 has no such watch, and an engine core left behind by a
parent that stopped (the stall guard, a crash) would keep the GPU's memory from the run that resumes it.
"""

import os

PARENT_CHECK_SECONDS = 5


def _exit_with_parent(parent: int) -> None:
    import threading
    import time

    def watch() -> None:
        while True:
            time.sleep(PARENT_CHECK_SECONDS)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=watch, name="fideon-exit-with-parent", daemon=True).start()


if os.environ.get("FIDEON_VLLM_PATCH") == "1":
    _parent = os.environ.get("FIDEON_VLLM_PARENT_PID", "")
    # Only the direct children of that process: a grandchild (a compile worker) has another parent from its start.
    if _parent.isdigit() and os.getppid() == int(_parent):
        _exit_with_parent(int(_parent))
    try:
        try:
            from inference_core import vllm_patches
        except ImportError:  # a serving pod without the repository: vllm_patches.py copied beside this file
            import vllm_patches  # type: ignore[no-redef]
        vllm_patches.patch_qwen3_vl_lora(in_process=False)
    except Exception as exc:  # noqa: BLE001 - every failure here means a process that cannot read page images
        import sys

        sys.stderr.write(f"FIDEON_VLLM_PATCH: the Qwen3-VL LoRA fix could not be applied, so this process stops: "
                         f"{type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        os._exit(1)
