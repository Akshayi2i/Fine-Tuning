"""Turn a call that never returns into a stop that says where it was.

vLLM runs in this process, so an engine that wedges leaves the process alive, the GPU idle
and the log silent, with nothing to say why (checkpoint selection, cycle 1, 2026-11-01: 28
minutes of silence, one core at 100%, and no way to see the stack). A call that outlives its
limit now ends the process after writing every thread's stack to the log, so the next run is
told where it stood, and ``scripts/pod_run.sh`` can resume it (``POD_RUN_RETRY_STALLED``).
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import sys
import threading

#: Printed on the line that ends a stalled process. ``pod_run.sh`` looks for it, and for the
#: line faulthandler writes when the timer below could not run (``Timeout (``).
MARKER = "STALL-GUARD:"

#: The exit status of a process that stopped itself here: temporary failure, try again.
EXIT_CODE = 75

#: How long after the limit faulthandler ends a process whose timer thread never ran. The
#: timer needs the interpreter lock; a call stuck in native code that holds it keeps the timer
#: from firing, and faulthandler's own watchdog does not need it.
BACKSTOP_SECONDS = 60


def abort(reason: str) -> None:
    """Write ``reason`` and every thread's stack to stderr, then end the process.

    ``os._exit``: the engine is wedged, so nothing that would wait on it (atexit handlers, thread
    joins, destructors) is run.
    """
    sys.stderr.write(f"\n{MARKER} {reason}\n")
    sys.stderr.flush()
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    sys.stderr.flush()
    try:
        # Told before the process goes: a stall is exactly when nobody is watching the log. Five seconds
        # at most; a message that cannot be sent must not keep a wedged process alive.
        from orchestration.notify import stalled

        stalled(reason)
    except Exception:  # noqa: BLE001
        pass
    os._exit(EXIT_CODE)


@contextlib.contextmanager
def stall_guard(seconds: float | None, what: str):
    """End the process, with every stack written out, when the block runs past ``seconds``.

    ``None`` or 0 turns it off. One at a time: faulthandler's backstop is a single timer, so a
    guard inside a guard would disarm the outer one on exit.
    """
    if not seconds or seconds <= 0:
        yield
        return
    backstop = False
    try:
        faulthandler.dump_traceback_later(seconds + BACKSTOP_SECONDS, repeat=False, file=sys.stderr, exit=True)
        backstop = True
    except (OSError, ValueError):
        pass   # a stderr with no file descriptor (a test's capture): the timer below still guards the call
    timer = threading.Timer(
        seconds, abort, args=(f"{what} has run for {seconds:.0f} s without returning; stopping so the run "
                              "can be resumed (the stacks follow)",))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        if backstop:
            faulthandler.cancel_dump_traceback_later()
