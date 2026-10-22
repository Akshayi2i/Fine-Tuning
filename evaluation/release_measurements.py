"""Latency and GPU memory, measured where a release is evaluated (Fideon SPEC_09 handoff item 6).

Recorded in the eval report and copied into the release bundle; not gated
until SPEC_06 sets a target. Measured during the golden eval, which serves
every eval document through :func:`serving.pipeline.extract` on the target
GPU: the P95 of every generation batch - the windows of a policy or a Loss Run
that run together, or a whole short document read in one call - per adapter,
and the most GPU memory in use while the eval ran, sampled every few seconds.

What this is not: a load test with every adapter of every promoted release
loaded at once. The eval serves this release's merged model alone; the bundle
says so (``measured_with``).
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

log = logging.getLogger(__name__)

#: How a call served by the base model, with no LoRA, is labelled.
BASE_MODEL = "base model"


def p95(values: Sequence[float]) -> float | None:
    """The 95th percentile by the nearest-rank method; None for no values."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def latency_by_adapter(scored: Sequence[tuple[Any, Any, dict[str, Any]]], *,
                       served_as: str | None = None) -> dict[str, dict[str, Any]]:
    """Per adapter: the P95 latency of one generation batch in milliseconds, and how many.

    ``served_as`` names what a call with no adapter ran on: in the golden eval,
    the release's own merged model, not the base model.
    """
    calls: dict[str, list[float]] = defaultdict(list)
    for _expected, _got, metadata in scored:
        adapter = metadata.get("adapter") or served_as or BASE_MODEL
        calls[adapter].extend(float(v) for v in metadata.get("window_latencies_ms") or [] if v is not None)
    return {adapter: {"p95_ms": round(p95(values), 1), "calls": len(values)}
            for adapter, values in sorted(calls.items()) if values}


def peak_gpu_memory_mb() -> float | None:
    """The memory in use now on the busiest GPU, from ``nvidia-smi``; None without one.

    vLLM reserves its share of the card when it starts (gpu_memory_utilization),
    so this reads that reservation plus whatever else the process holds.
    :class:`GpuMemorySampler` takes the peak of these over a run.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("GPU memory not measured: %s", exc)
        return None
    readings = [float(line) for line in out.split() if line.strip().replace(".", "", 1).isdigit()]
    return max(readings) if readings else None


class GpuMemorySampler:
    """The peak of :func:`peak_gpu_memory_mb` while a block runs, every ``interval_s``.

    ``with GpuMemorySampler() as sampler: ...`` then ``sampler.peak_mb``. No
    sampling without a GPU (the first reading is None).
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        import threading

        self.interval_s = interval_s
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: Any = None

    def _sample(self) -> None:
        reading = peak_gpu_memory_mb()
        if reading is not None:
            self.peak_mb = reading if self.peak_mb is None else max(self.peak_mb, reading)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> GpuMemorySampler:
        import threading

        self._sample()
        if self.peak_mb is not None:
            self._thread = threading.Thread(target=self._run, name="gpu-memory-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)
            self._sample()


def measurements(scored: Sequence[tuple[Any, Any, dict[str, Any]]], *, served_as: str | None = None,
                 peak_mb: float | None = None) -> dict[str, Any]:
    return {
        "latency_p95_ms_by_adapter": latency_by_adapter(scored, served_as=served_as),
        "peak_gpu_memory_mb": peak_mb if peak_mb is not None else peak_gpu_memory_mb(),
        "measured_with": "the golden eval: this release's merged bf16 model alone, on the target GPU; "
                         "latency per generation batch, memory the peak sampled during the eval",
    }
