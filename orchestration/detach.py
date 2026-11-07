"""Every long job on the pod runs detached, in tmux — whoever starts it, however.

A job started in an SSH terminal dies with that terminal: shutting the laptop,
letting it sleep or losing Wi-Fi ends the SSH session, the session's processes
get SIGHUP, and a training run hours in is gone. ``scripts/pod_run.sh`` avoids
that, but only if it is remembered every time. This makes it the only way:
on a RunPod pod, a long command that is not already inside tmux re-launches
itself through ``pod_run.sh`` and returns at once, telling you how to watch it.

Off the pod (a laptop, CI) nothing changes: commands run in the foreground as
they always did. Inside tmux, or inside a ``pod_run.sh`` run, they also run in
the foreground — they are already safe, and re-detaching would start a second
copy. ``FIDEON_NO_DETACH=1`` runs a command in the foreground on the pod anyway,
for the rare case that is wanted; it is never set by this code.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

POD_RUN = Path(__file__).resolve().parent.parent / "scripts" / "pod_run.sh"

#: Set by pod_run.sh inside every run it starts.
DETACHED_ENV = "FIDEON_DETACHED"
#: Opt out, for one command, on the pod.
NO_DETACH_ENV = "FIDEON_NO_DETACH"
#: RunPod sets this in every pod.
POD_ENV = "RUNPOD_POD_ID"
#: RunPod's copy of the pod environment, present even when a shell lacks it.
RUNPOD_ENV_FILE = Path("/etc/rp_environment")
#: The pod's network volume mount.
WORKSPACE = Path("/workspace")


def on_pod() -> bool:
    """Whether this is the RunPod pod.

    ``RUNPOD_POD_ID`` first. It is set in the pod's own environment, but an SSH
    session does not always inherit it, and a job started from a session that
    misses it would run attached and die with the connection. So also: RunPod's
    environment file, or — on Linux — ``/workspace`` MOUNTED as a volume on a
    machine with a GPU. A mount, not merely a folder: a laptop with a GPU and a
    ``workspace`` directory is not the pod.
    """
    from common.gpu import gpu_present

    if os.environ.get(POD_ENV) or RUNPOD_ENV_FILE.exists():
        return True
    return _is_linux() and os.path.ismount(WORKSPACE) and gpu_present()


def _is_linux() -> bool:
    import sys

    return sys.platform.startswith("linux")


def already_safe() -> bool:
    """Inside a pod_run.sh run, or inside tmux the operator started themselves."""
    return os.environ.get(DETACHED_ENV) == "1" or bool(os.environ.get("TMUX"))


def should_detach() -> bool:
    return on_pod() and not already_safe() and os.environ.get(NO_DETACH_ENV) != "1"


def run_name(hint: str) -> str:
    """``finetune-20261003-141502`` — unique per start, valid for pod_run.sh."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    # The process id too: four OCR shards started by one loop share a second,
    # and the second to claim the name was refused and never started.
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '-', hint).strip('-') or 'job'}-{stamp}-{os.getpid()}"


def detach(command: Sequence[str], hint: str) -> str:
    """Start ``command`` in a tmux session through pod_run.sh. Returns the run name.

    ``command`` is the full command line to run (``python -m ...``, ``bash ...``).
    """
    name = run_name(hint)
    subprocess.run(["bash", str(POD_RUN), "start", name, "--", *command], check=True)
    return name


def detach_if_needed(command: Sequence[str], hint: str) -> bool:
    """Re-launch ``command`` detached when on the pod and not already safe.

    Returns True when it did: the caller should exit at once, because the job now
    runs in tmux and running it here too would start it twice.
    """
    if not should_detach():
        return False
    name = detach(command, hint)
    print(
        f"\nOn the pod, long jobs run in tmux so a closed laptop or dropped Wi-Fi cannot stop "
        f"them. This one is running as '{name}'.\n"
        f"  status: bash scripts/pod_run.sh status {name}\n"
        f"  watch:  bash scripts/pod_run.sh attach {name}   (Ctrl-b then d to leave it running)\n"
        f"  log:    bash scripts/pod_run.sh tail {name}\n"
    )
    return True


#: The pod's training virtualenv (scripts/pod_bootstrap.sh); FIDEON_WORKSPACE moves it with the volume.
TRAINING_VENV = Path(os.environ.get("FIDEON_WORKSPACE") or str(WORKSPACE)) / "venv"


def detached_python() -> str:
    """The Python a detached job runs with: the caller's when it is a virtualenv's, else the training venv's.

    A shell with no venv active ran ``python`` as the pod's system Python, which has none of this repo's
    packages: ``data_health`` died at once on ``No module named 'pydantic'`` (2026-11-07). pod_run.sh falls
    back to the training venv only for a command that names no interpreter, and this one names it.
    """
    import sys

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return sys.executable
    venv_python = TRAINING_VENV / "bin" / "python"
    if venv_python.is_file():
        log.warning("no virtualenv is active; the job runs with the training venv's %s", venv_python)
        return str(venv_python)
    return sys.executable


def detach_module_if_needed(module: str, argv: Sequence[str] | None, hint: str | None = None) -> bool:
    """:func:`detach_if_needed` for ``python -m <module> <argv>``."""
    import sys

    if not should_detach():  # nothing moves, so no interpreter is chosen (and none is announced)
        return False
    args = list(argv) if argv is not None else sys.argv[1:]
    return detach_if_needed([detached_python(), "-m", module, *args], hint or module.rsplit(".", 1)[-1])


def detach_script_if_needed(script: str | Path, argv: Sequence[str] | None, hint: str) -> bool:
    """:func:`detach_if_needed` for ``python <script> <argv>``."""
    import sys

    if not should_detach():
        return False
    args = list(argv) if argv is not None else sys.argv[1:]
    return detach_if_needed([detached_python(), str(script), *args], hint)
