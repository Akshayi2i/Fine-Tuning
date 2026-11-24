"""MinerU version and device pinning (arch §8a).

The model is fine-tuned partly on **how MinerU formats its output** — table
markdown conventions, reading order, the shape of its errors. So the OCR tool is
not an interchangeable dependency: serving a model trained on MinerU v{n} against
documents processed by v{n+1} is **distribution shift**, and is treated as a
regression trigger rather than a routine dependency bump.

Device is pinned alongside version because OCR now runs on GPU (arch §14), and
GPU and CPU MinerU can select different model variants. If they format
differently, training on GPU-OCR and serving on CPU-OCR is the same distribution
shift §8a warns about. Whether they actually differ is an empirical question —
:func:`compare_devices` is how you answer it, and :data:`DEVICE_AFFECTS_OUTPUT`
is how you record the answer.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

log = logging.getLogger(__name__)

#: The only device OCR runs on. ``cpu`` is deliberately absent: MinerU's CPU path
#: uses lighter model variants and produces different markdown from the same PDF,
#: which the model would be learning as a second output format (arch §8a).
Device = Literal["cuda"]

#: What an ``ocr_meta.json`` may legitimately *contain*. Wider than ``Device``
#: because a corpus processed before this rule can still be read and reported on
#: — it just cannot be extended.
RecordedDevice = Literal["cuda", "cpu"]

#: Set from the Phase 0 spike. When MinerU's GPU and CPU output prove identical,
#: flip this to False and the device check relaxes to a warning. Defaulting to
#: True is the safe direction: a false alarm costs a reprocess, a missed
#: distribution shift costs a silently worse model.
DEVICE_AFFECTS_OUTPUT = True

UNKNOWN = "unknown"


class MinerUVersionError(RuntimeError):
    """Raised when the running MinerU differs from what a corpus pinned."""


@dataclass(frozen=True)
class OcrEnvironment:
    """The OCR environment that produced (or is about to produce) a corpus."""

    mineru_version: str
    device: Device
    gpu_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mineru_version": self.mineru_version,
            "ocr_device": self.device,
            "gpu_name": self.gpu_name,
        }


@lru_cache(maxsize=1)
def get_mineru_version() -> str:
    """The installed MinerU version, or ``"unknown"``.

    ``unknown`` is recorded rather than raised: a corpus built by an
    unidentifiable OCR version is still buildable, but the manifest then says so,
    which is the honest record.
    """
    try:
        from importlib.metadata import version

        # MinerU 3.x; 1.x was published as magic-pdf, and a corpus read by it
        # records a 1.x version - another distribution from 3.x's.
        return version("mineru")
    except Exception:
        pass
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import mineru.version as v; print(v.__version__)"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return out.stdout.strip().split()[-1] or UNKNOWN
    except Exception:
        return UNKNOWN


@lru_cache(maxsize=1)
def cuda_available() -> tuple[bool, str | None]:
    """Whether a CUDA device is usable, and its name."""
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, None


def resolve_device(requested: Device | None = None, *, strict: bool = True) -> Device:
    """Choose the OCR device. **GPU is the default** (arch §14).

    Args:
        requested: must be ``"cuda"`` or ``None``. ``"cpu"`` is refused.
        strict: retained for the call signature; GPU is required either way, so
            this no longer selects between raising and falling back. There is no
            fallback to select.

    Returns:
        Always ``"cuda"``.

    Raises:
        MinerUVersionError: if CPU was requested, or if no CUDA device is visible.
    """
    if requested == "cpu":
        raise MinerUVersionError(
            "MinerU runs on GPU only in this project — 'cpu' is not an available device. "
            "The CPU path uses lighter model variants, so it produces *different markdown* from "
            "the same PDF: different table splits, different cell boundaries. The model learns "
            "how MinerU formats its output (arch §8a), so a corpus built with a mix of devices "
            "is a corpus built from two distributions, and serving from the third. That is not a "
            "speed trade-off; it is a correctness one."
        )

    available, name = cuda_available()
    if not available:
        raise MinerUVersionError(
            "no CUDA device is visible, and OCR is GPU-only. Failing here is deliberate: a silent "
            "CPU fallback would finish the job and write markdown from a different distribution "
            "than the corpus was built on, with no error anywhere and no way to detect it later. "
            "Run this stage on a GPU pod — an L4/A10/L40S is sufficient, MinerU does not need the "
            "A100 (IMPL-13 §7)."
        )
    log.debug("MinerU on %s", name)
    return "cuda"


def current_environment(requested_device: Device | None = None, *, strict: bool = True) -> OcrEnvironment:
    """Describe the OCR environment for recording in ``ocr_meta.json``."""
    device = resolve_device(requested_device, strict=strict)
    _available, gpu_name = cuda_available()
    return OcrEnvironment(
        mineru_version=get_mineru_version(),
        device=device,
        gpu_name=gpu_name if device == "cuda" else None,
    )


def assert_version_matches(
    pinned: dict[str, Any],
    current: OcrEnvironment | None = None,
    *,
    check_device: bool | None = None,
) -> None:
    """Assert the running OCR environment matches what a corpus pinned.

    Called by the dataset builder (IMPL-05), the serving pipeline (IMPL-11) and
    the testing harness (IMPL-12). A mismatch is a **regression trigger**, not a
    warning to be dismissed — the remediation is stated in the message rather
    than left for the reader to work out.

    Args:
        pinned: a corpus manifest, or its ``dependencies`` block. Read for
            ``mineru_version`` and ``ocr_device``.
        current: the running environment; defaults to detecting it.
        check_device: override :data:`DEVICE_AFFECTS_OUTPUT`.
    """
    current = current or current_environment(strict=False)
    pinned_version = pinned.get("mineru_version")
    pinned_device = pinned.get("ocr_device")

    if pinned_version and pinned_version != UNKNOWN and pinned_version != current.mineru_version:
        raise MinerUVersionError(
            f"MinerU version mismatch: the corpus was built with {pinned_version}, this "
            f"environment has {current.mineru_version}. The model is fine-tuned partly on how "
            "MinerU formats its output, so this is distribution shift, not a dependency bump "
            "(arch §8a). Remediation: reprocess the affected documents with the pinned version, "
            "increment the corpus version, and retrain — or pin this environment to match."
        )

    should_check = DEVICE_AFFECTS_OUTPUT if check_device is None else check_device
    if should_check and pinned_device and pinned_device != current.device:
        raise MinerUVersionError(
            f"OCR device mismatch: the corpus was built on {pinned_device}, this environment is "
            f"{current.device}. GPU and CPU MinerU can select different model variants and format "
            "output differently, which is the same distribution shift §8a describes. If the Phase 0 "
            "spike shows the outputs are byte-identical, set DEVICE_AFFECTS_OUTPUT = False and this "
            "check relaxes."
        )


def assert_gpu_only(recorded_device: str | None) -> None:
    """Assert a recorded ``ocr_device`` is one this project still produces.

    A corpus carrying ``cpu`` predates the GPU-only rule. It stays readable, and
    it is not extendable: adding GPU-processed documents to it would put two
    different markdown formats in one training set, which is exactly what the
    version pin exists to prevent.
    """
    if recorded_device and recorded_device != "cuda":
        raise MinerUVersionError(
            f"this corpus records ocr_device={recorded_device!r}, but OCR is GPU-only. Documents "
            "processed on CPU carry different markdown formatting, so extending this corpus would "
            "mix two output formats in one training set. Reprocess it on GPU and increment the "
            "corpus version."
        )
