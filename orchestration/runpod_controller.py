"""RunPod control — ephemeral training pods and the persistent serving endpoint (IMPL-13 §7).

Two lifetimes, deliberately different (arch §14):

* **Training pods are ephemeral.** A pod clones the repo at a pinned commit,
  works, writes to the staging volume, and terminates. Code is never permanently
  resident on a pod, so "which code produced this model" is answered by the
  manifest rather than by whatever happened to be on a box.
* **The serving endpoint is persistent.** It is updated to a promoted version and
  rolled back, not launched and destroyed.

**Every pod gets the staging volume attached, and a launch without it is refused.**
That check is the whole reason this module owns pod creation rather than each
stage calling the API itself: a pod without the volume writes to pod-local disk,
terminates, and the work is gone — with no error anywhere, because every write
succeeded.

Stages 2, 4 and 5 all want a GPU, so ``finetune`` provisions **one pod for the
whole command** rather than shuttling a corpus between machines: cheaper than
three launches, and it removes two Blob round trips.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from common.config import env

log = logging.getLogger(__name__)

PodStatus = Literal["pending", "running", "terminated", "failed"]

#: GPU class per stage — fallbacks for ``orchestration/config/pipeline.yaml``,
#: keyed by the stage names the DAG uses. The deployment runs every GPU stage on
#: one H200 SXM (141 GB) pod, in process: moving a corpus between an OCR pod and
#: a training pod costs more than a cheaper OCR card saves.
GPU_CLASS_BY_STAGE: dict[str, str] = {
    stage: "H200-SXM"
    for stage in ("preprocessing", "dataset_build", "training", "checkpoint_eval",
                  "calibrate", "evaluation_gate")
}

DEFAULT_VOLUME_MOUNT = "/runpod-volume"

#: Applied to every log line before it leaves the pod (master §8). Insurance
#: documents carry names, addresses and policy numbers, and a training log that
#: echoes a prompt echoes all three into a third-party log store.
SCRUB_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]"),
    (r"\b\d{3}[.-]?\d{3}[.-]?\d{4}\b", "[PHONE]"),
    (r"(?i)\b(?:policy|certificate)\s*(?:no\.?|number|#)\s*:?\s*\S+", "[POLICY_NUMBER]"),
    (r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S+", "[SECRET]"),
)


class PodLaunchError(RuntimeError):
    """Raised when a pod cannot be launched safely."""


class EndpointError(RuntimeError):
    """Raised when the serving endpoint cannot be updated."""


def scrub(text: str) -> str:
    """Redact PII and secrets from a log line before it leaves the pod."""
    for pattern, replacement in SCRUB_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


# --------------------------------------------------------------------------
# The staging volume
# --------------------------------------------------------------------------


class StagingVolume:
    """The RunPod network volume, addressed by the paths in ``artifact_registry``.

    **Not a registry.** It is working storage with no durability guarantee: it can
    be reclaimed, and nothing about it is an artifact of record. That is exactly
    why ``finetune`` always writes its run manifest to Blob even while the weights
    stay here (IMPL-13 §3).

    The in-memory default is what lets the whole DAG run in CI with no pod. On a
    real pod, pass a backend that writes to the mount.
    """

    def __init__(self, mount: str = DEFAULT_VOLUME_MOUNT) -> None:
        self.mount = mount
        self._entries: dict[str, bytes] = {}

    def write(self, path: str, data: bytes | str = b"") -> None:
        self._entries[path] = data.encode("utf-8") if isinstance(data, str) else data

    def mark(self, path: str) -> None:
        """Record that a stage produced this path.

        The weights themselves are written by ms-swift, PEFT and llama.cpp
        directly to the mount; this records the same path so a resumed run can
        ask "is this already done?" identically whether the volume is the real
        mount or the in-memory model of it used off-pod.
        """
        self._entries.setdefault(path, b"")

    def read(self, path: str) -> bytes:
        if path not in self._entries:
            raise FileNotFoundError(path)
        return self._entries[path]

    def exists(self, path: str) -> bool:
        return path in self._entries or any(k.startswith(path.rstrip("/") + "/") for k in self._entries)

    def list(self, prefix: str) -> list[str]:
        return sorted(k for k in self._entries if k.startswith(prefix))

    def clear(self, prefix: str) -> int:
        """Drop everything under a prefix — what ``package`` does after a verified push."""
        doomed = self.list(prefix)
        for key in doomed:
            del self._entries[key]
        return len(doomed)


# --------------------------------------------------------------------------
# Pods
# --------------------------------------------------------------------------


@dataclass
class PodSpec:
    """One pod to launch. Inspectable before anything is billed."""

    name: str
    stage: str
    gpu_class: str
    volume_id: str | None
    volume_mount: str = DEFAULT_VOLUME_MOUNT
    image: str = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    #: The pod clones the repo fresh at this commit; code is never resident on a
    #: pod between jobs, so the manifest's commit is the only source of truth.
    git_commit: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def clone_command(self, repo: str) -> list[str]:
        if not self.git_commit or self.git_commit == "unknown":
            raise PodLaunchError(
                f"pod {self.name!r} has no resolvable git commit ({self.git_commit!r}). A pod that "
                "clones a moving branch "
                "produces a model whose code version is 'whatever main was that afternoon', which "
                "makes the run manifest's code_git_commit a guess (arch §12)."
            )
        return ["git", "clone", repo, "/workspace/repo", "&&",
                "git", "-C", "/workspace/repo", "checkout", self.git_commit]


@dataclass
class PodHandle:
    """A launched pod."""

    pod_id: str
    spec: PodSpec
    status: PodStatus = "pending"
    logs: list[str] = field(default_factory=list)


@dataclass
class JobResult:
    """What a pod job produced."""

    pod_id: str
    stage: str
    ok: bool
    value: Any = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)


class PodBackend(Protocol):
    """The RunPod API surface this module needs, narrowed to five calls."""

    def create(self, spec: PodSpec) -> str: ...
    def status(self, pod_id: str) -> PodStatus: ...
    def logs(self, pod_id: str) -> list[str]: ...
    def terminate(self, pod_id: str) -> None: ...


class LocalBackend:
    """Runs the job in this process instead of on a pod.

    Real, not a stub: it is how CI, the fixture runs, and any laptop-side dry run
    execute the whole DAG. What it does not do is provide a GPU, so a stage that
    genuinely needs one still fails at the stage's own boundary rather than here.
    """

    def __init__(self) -> None:
        self._pods: dict[str, PodStatus] = {}
        self._logs: dict[str, list[str]] = {}
        self._counter = 0

    def create(self, spec: PodSpec) -> str:
        self._counter += 1
        pod_id = f"local-{spec.stage}-{self._counter}"
        self._pods[pod_id] = "running"
        self._logs[pod_id] = [f"local pod for stage {spec.stage} on {spec.gpu_class}"]
        return pod_id

    def status(self, pod_id: str) -> PodStatus:
        return self._pods.get(pod_id, "terminated")

    def logs(self, pod_id: str) -> list[str]:
        return list(self._logs.get(pod_id, []))

    def terminate(self, pod_id: str) -> None:
        self._pods[pod_id] = "terminated"


class RunPodBackend:
    """The live RunPod API. Wired in the Phase 0 spike."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or env("RUNPOD_API_KEY")

    def _unavailable(self) -> None:
        raise NotImplementedError(
            "the live RunPod backend is not wired yet. It needs RUNPOD_API_KEY, the network "
            "volume created in Phase 0, and the pod image confirmed to install the [train] extra "
            "(flash-attn is the usual failure). Until then LocalBackend runs the same DAG in "
            "process, so nothing about the orchestration logic is waiting on this."
        )

    def create(self, spec: PodSpec) -> str:
        self._unavailable()
        raise AssertionError("unreachable")

    def status(self, pod_id: str) -> PodStatus:
        self._unavailable()
        raise AssertionError("unreachable")

    def logs(self, pod_id: str) -> list[str]:
        self._unavailable()
        raise AssertionError("unreachable")

    def terminate(self, pod_id: str) -> None:
        self._unavailable()


class RunPodController:
    """Launches, polls and tears down pods; manages the serving endpoint."""

    def __init__(
        self,
        *,
        backend: PodBackend | None = None,
        volume: StagingVolume | None = None,
        volume_id: str | None = None,
        volume_mount: str | None = None,
        git_commit: str | None = None,
    ) -> None:
        self.backend: PodBackend = backend or LocalBackend()
        self.volume_id = volume_id if volume_id is not None else env("RUNPOD_VOLUME_ID")
        self.volume_mount = volume_mount or env("RUNPOD_VOLUME_MOUNT", DEFAULT_VOLUME_MOUNT) or DEFAULT_VOLUME_MOUNT
        self.volume = volume or StagingVolume(self.volume_mount)
        self.git_commit = git_commit
        self._active: dict[str, PodHandle] = {}
        #: Deployment history, so a rollback has somewhere to roll back to.
        self._endpoint_versions: list[str] = []

    # -- pods ---------------------------------------------------------------

    def spec_for(self, stage: str, *, gpu_class: str | None = None, **over: Any) -> PodSpec:
        """The pod a stage needs, with the stage's configured GPU class.

        The class comes from ``config/pipeline.yaml`` so an operator can move a
        stage to a different card without editing code; the table below is the
        fallback when the file is unreadable.
        """
        from orchestration.settings import gpu_class_for, pod_image

        # `pod.image` was configured and never read, so PodSpec's hardcoded
        # default was always used and editing the config changed nothing.
        over.setdefault("image", pod_image() or PodSpec.image)
        return PodSpec(
            name=f"{stage}-pod",
            stage=stage,
            gpu_class=gpu_class or gpu_class_for(stage, GPU_CLASS_BY_STAGE.get(stage, "H200-SXM")),
            volume_id=self.volume_id,
            volume_mount=self.volume_mount,
            git_commit=self.git_commit,
            **over,
        )

    def launch(self, spec: PodSpec) -> PodHandle:
        """Launch one pod. Refuses without the staging volume attached."""
        # In process (LocalBackend) on the pod, there is no second pod whose disk
        # could vanish: the stage writes straight to the volume mounted here, so
        # the mount existing is the guarantee. Off the pod it does not exist, and
        # the refusal below stands.
        in_process_on_volume = (
            isinstance(self.backend, LocalBackend) and os.path.isdir(spec.volume_mount)
        )
        if not spec.volume_id and not in_process_on_volume:
            raise PodLaunchError(
                f"refusing to launch pod {spec.name!r} for stage {spec.stage!r} with no staging "
                "volume attached. Without it the pod writes to pod-local disk, terminates, and "
                "the work is gone — silently, because every write succeeded. Set RUNPOD_VOLUME_ID "
                f"(created in Phase 0) or pass volume_id explicitly (IMPL-13 §3); running in "
                f"this process, the staging mount {spec.volume_mount} must exist."
            )

        pod_id = self.backend.create(spec)
        handle = PodHandle(pod_id=pod_id, spec=spec, status="running")
        self._active[pod_id] = handle
        log.info("launched pod %s for stage %s on %s (volume %s at %s)",
                 pod_id, spec.stage, spec.gpu_class, spec.volume_id, spec.volume_mount)
        return handle

    def collect_logs(self, handle: PodHandle) -> list[str]:
        """Pull the pod's logs, **scrubbed** before they leave it (master §8)."""
        lines = [scrub(line) for line in self.backend.logs(handle.pod_id)]
        handle.logs = lines
        return lines

    def terminate(self, handle: PodHandle) -> None:
        self.backend.terminate(handle.pod_id)
        handle.status = "terminated"
        self._active.pop(handle.pod_id, None)
        log.info("terminated pod %s", handle.pod_id)

    @property
    def active_pods(self) -> list[str]:
        """Pods still running. Must be empty once a command returns — an
        orphaned pod bills by the hour and nothing else in the system notices."""
        return sorted(self._active)

    @contextmanager
    def pod(self, spec: PodSpec) -> Iterator[PodHandle]:
        """Launch, yield, and terminate — terminating even when the body raises.

        The ``finally`` is the point. A stage that raises mid-training must not
        leave a GPU pod running; that failure costs money silently and for as
        long as nobody looks.
        """
        handle = self.launch(spec)
        try:
            yield handle
        finally:
            # Logs are best-effort; termination is not. Fetching first meant a
            # transient API error on the log call leaked a billing A100 — and
            # run_job then reported pod_id="" so the caller could not even find
            # it to kill. This is the failure the `finally` exists to prevent.
            try:
                self.collect_logs(handle)
            except Exception as exc:  # noqa: BLE001 - never block teardown
                log.warning("could not collect logs for pod %s: %s", handle.pod_id, exc)
            self.terminate(handle)

    def run_job(self, spec: PodSpec, job: Callable[[PodHandle], Any]) -> JobResult:
        """Run one job on one pod and tear the pod down afterwards."""
        try:
            with self.pod(spec) as handle:
                value = job(handle)
                return JobResult(pod_id=handle.pod_id, stage=spec.stage, ok=True,
                                 value=value, logs=handle.logs)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            log.error("stage %s failed on pod: %s", spec.stage, exc)
            return JobResult(pod_id="", stage=spec.stage, ok=False, error=str(exc))

    @contextmanager
    def session_pod(self, stage: str = "training", *, gpu_class: str | None = None) -> Iterator[PodHandle]:
        """One pod for a whole command.

        ``finetune`` runs OCR, dataset build and training back to back, all
        GPU-bound. Three launches would mean three cold starts and two Blob round
        trips of the same corpus (IMPL-13 §7).
        """
        with self.pod(self.spec_for(stage, gpu_class=gpu_class)) as handle:
            yield handle

    # -- the serving endpoint ----------------------------------------------

    def deploy_endpoint(self, model_version: str, *, dry_run: bool = False) -> str:
        """Point the persistent vLLM endpoint at a promoted version."""
        if not model_version:
            raise EndpointError("deploy_endpoint needs a model version")
        previous = self._endpoint_versions[-1] if self._endpoint_versions else None
        log.info("serving endpoint -> %s (previous %s)%s",
                 model_version, previous, " [dry run]" if dry_run else "")
        if dry_run:
            # History records what was DEPLOYED. Appending before this check made
            # a preview claim the endpoint had moved, so health_check reported a
            # version that was never live and the next rollback popped it and
            # "restored" the version already running.
            return model_version

        self._endpoint_versions.append(model_version)
        raise NotImplementedError(
            "wire the RunPod Serverless vLLM endpoint update here once the Phase 0 spike "
            "confirms multi-LoRA hot-swap for Qwen3-VL. If it does not hold, IMPL-11 falls back "
            "to serving merged per-type models and this call deploys one of those instead."
        )

    def rollback_endpoint(self, *, dry_run: bool = False) -> str:
        """Restore the previously promoted version."""
        if len(self._endpoint_versions) < 2:
            raise EndpointError(
                "nothing to roll back to — this controller has deployed "
                f"{len(self._endpoint_versions)} version(s). Rollback restores the *previous* "
                "promoted version, so it needs a history; on a fresh process, read it from the "
                "registry index rather than assuming one exists."
            )
        # The history is read, not popped, until the rollback actually happens.
        # Popping first meant a preview permanently rewrote the recorded live
        # version: health_check then reported the old version while the new one
        # was still serving, and a subsequent real rollback refused with
        # "nothing to roll back to" — the same bug deploy_endpoint was fixed for.
        target = self._endpoint_versions[-2]
        log.warning("rolling serving endpoint back to %s", target)
        if dry_run:
            return target
        self._endpoint_versions.pop()
        raise NotImplementedError("wire the endpoint rollback alongside deploy_endpoint")

    def health_check(self) -> dict[str, Any]:
        return {
            "deployed_version": self._endpoint_versions[-1] if self._endpoint_versions else None,
            "history": list(self._endpoint_versions),
            "active_pods": self.active_pods,
        }
