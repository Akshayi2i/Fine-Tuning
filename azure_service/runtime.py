"""What the worker reads with, loaded once at start-up.

For each tenant it serves (``SERVED_TENANTS``):

1. the promoted release (:func:`serving.release_router.build_serving_plan`),
   refused unless it is served as a LoRA on the base (``lora_on_base``: the only
   kind the pod's stock vLLM can load), trained on the pinned MinerU's text
   (``scripts.served_adapter.assert_ocr_pin``), packaged with the prompts and
   the window plan this code renders, on the configured base
   (``serving.vllm_entrypoint.assert_prompt_hash``, ``assert_window_plan_served``,
   ``assert_base_model_served``);
2. the model: vLLM asked for ``policy-<release>`` (``scripts.served_adapter.lora_name``),
   which :class:`inference_core.openai_backend.OpenAIChatBackend` checks vLLM
   serves, on our base;
3. the release's calibrators, from Blob, as
   ``serving.vllm_entrypoint.load_release_runtimes`` reads them - without its
   adapter download: the adapter is vLLM's, on the pod's volume.

With nothing promoted, the worker refuses to start unless the base model is
configured as the fallback (``ALLOW_BASE_FALLBACK`` and ``BASE_MODEL_NAME``);
then a policy no release covers is read by the base model, flagged in its
result (``route: base_model_fallback``). Any failed check is a
:class:`serving.vllm_entrypoint.ColdStartError`: the container exits, and
Container Apps shows why in its log.

The MinerU reader is :class:`data_pipeline.ocr.served_ocr.ServedOcr`, pinned:
it refuses text from another MinerU version or tier.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from serving.vllm_entrypoint import ColdStartError, ReleaseRuntime

log = logging.getLogger(__name__)

#: The route of a policy read by the base model because no promoted release covers it.
BASE_FALLBACK_ROUTE = "base_model_fallback"
RELEASE_ROUTE = "release"


class JobRefused(RuntimeError):
    """A job this worker can never read as configured: failed at once, never retried."""


@dataclass(frozen=True)
class Reading:
    """How one job is read: the model, and the release's calibration when it has one."""

    model: Any
    release_id: str | None
    route: str
    calibrators: Any = None
    thresholds: Any = None

    def extract_options(self, tuning: dict[str, Any]) -> dict[str, Any]:
        """What :func:`serving.pipeline.extract` is given beside the request: the
        serving thresholds, and the release's calibrators when it has them."""
        options = dict(tuning)
        if self.calibrators is not None:
            options.update(calibrators=self.calibrators, thresholds=self.thresholds)
        return options


@dataclass
class TenantRuntime:
    """One tenant's promoted release and the models that read for it."""

    tenant: str
    plan: Any = None
    release_model: Any = None
    #: The base model with no LoRA, when the fallback is configured.
    base_model: Any = None
    release_runtime: ReleaseRuntime | None = None

    @property
    def release(self) -> Any:
        return self.plan.release if self.plan is not None else None

    def reading(self, lob: Any) -> Reading:
        """The model a policy of ``lob`` is read by, as the serving plan routes it:
        the release when it trained on the line, the base model otherwise."""
        release = self.release
        if release is not None and self.release_model is not None:
            routed = self.plan.route("policy", lob, allow_fallback=True)
            if routed.release is not None:
                runtime = self.release_runtime
                return Reading(self.release_model, release.release_id, RELEASE_ROUTE,
                               calibrators=runtime.calibrators if runtime else None,
                               thresholds=runtime.thresholds if runtime else None)
        if self.base_model is None:
            what = (f"release {release.release_id} serves lines {list(release.lines)}, not {lob!r}"
                    if release is not None else f"no release is promoted for tenant {self.tenant}")
            raise JobRefused(f"{what}, and this worker does not read with the base model "
                             "(ALLOW_BASE_FALLBACK is off)")
        return Reading(self.base_model, None, BASE_FALLBACK_ROUTE)


@dataclass
class ServingRuntime:
    """Everything a job is read with."""

    tenants: dict[str, TenantRuntime]
    reader: Any
    #: The serving thresholds (configs/inference/vllm_serving.yaml), for extract.
    tuning: dict[str, Any] = field(default_factory=dict)

    def tenant(self, name: str) -> TenantRuntime:
        runtime = self.tenants.get(name)
        if runtime is None:
            raise JobRefused(f"this worker serves tenants {sorted(self.tenants)}, not {name}")
        return runtime

    def describe(self) -> dict[str, Any]:
        return {name: {"release": t.release.release_id if t.release is not None else None,
                       "model": t.release_model.tag if t.release_model is not None else None,
                       "calibrated": bool(t.release_runtime and t.release_runtime.calibrators is not None),
                       "base_fallback": t.base_model is not None}
                for name, t in self.tenants.items()}


def identity_calibration(model: Any) -> Any:
    """Raw confidence: the v1 calibration ``extract`` falls back to when a
    release has no fitted calibrators (``Reading.extract_options``)."""
    from calibration.fit_calibration import CalibrationParams

    return CalibrationParams(method="temperature", doc_type="policy", model_version=model.tag, temperature=1.0)


def served_model(backend: Any, name: str) -> Any:
    """``backend`` as the pipeline loads a model, filed under ``name``
    (``policy-v3``, or ``base``). Nothing is read from Blob for it."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import load_model

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load_model("base", client, backend_impl=backend)
    # load_model files a model under its label only with an adapter of its own;
    # this one's adapter is the server's, so the name is set here.
    return dataclasses.replace(model, tag=name)


def load_calibrators(release: Any, registry: Any) -> ReleaseRuntime:
    """The release's fitted calibrators and thresholds
    (``serving.vllm_entrypoint.load_release_calibration``); ``adapter`` is
    None: vLLM applies it, by the name the model is asked for."""
    from serving.vllm_entrypoint import load_release_calibration

    return load_release_calibration(release, registry)


def check_release(plan: Any) -> None:
    """The vllm_entrypoint cold-start checks that apply to a release served by the pod's vLLM."""
    from common.config import served_mineru
    from scripts.served_adapter import ServedAdapterError, assert_ocr_pin
    from serving.vllm_entrypoint import (
        assert_base_model_served,
        assert_prompt_hash,
        assert_window_plan_served,
    )

    release = plan.release
    if release.served_as != "lora_on_base" or not release.adapter_dir:
        raise ColdStartError(
            f"release {release.release_id} is served as {release.served_as!r}: the pod's vLLM serves a release "
            "as a LoRA on the base (lora_on_base, every release packaged since 2026-10-23)")
    try:
        assert_ocr_pin(release.ocr_pin, served_mineru())
    except ServedAdapterError as exc:
        raise ColdStartError(str(exc)) from exc
    assert_prompt_hash(plan)
    assert_window_plan_served(plan)
    assert_base_model_served(plan)


def load_tenant(tenant: str, registry: Any, settings: Any, models: Callable[[str], Any]) -> TenantRuntime:
    """One tenant's release, model and calibrators; ``models(name)`` gives the served model ``name``."""
    from scripts.served_adapter import lora_name
    from serving.release_router import build_serving_plan

    plan = build_serving_plan(registry, tenant_id=tenant)
    runtime = TenantRuntime(tenant=tenant, plan=plan)
    if plan.release is not None:
        check_release(plan)
        runtime.release_model = models(lora_name(plan.release.release_id))
        runtime.release_runtime = load_calibrators(plan.release, registry)
    elif not settings.allow_base_fallback:
        raise ColdStartError(
            f"no release is promoted for tenant {tenant}, and the base model is not configured as the fallback "
            "(ALLOW_BASE_FALLBACK=true with BASE_MODEL_NAME): there is nothing to read with")
    if settings.allow_base_fallback:
        runtime.base_model = models(settings.base_model_name)
    return runtime


def openai_models(settings: Any, links: Any = None) -> Callable[[str], Any]:
    """Served models by name, from the vLLM at ``MODEL_URL``: one backend per name,
    shared by every tenant that reads with it."""
    from inference_core.openai_backend import OpenAIChatBackend

    made: dict[str, Any] = {}

    def model(name: str) -> Any:
        if name not in made:
            backend = OpenAIChatBackend(settings.model_url, settings.model_token, model=name,
                                        max_concurrency=settings.max_concurrency, images=settings.images,
                                        links=links)
            made[name] = served_model(backend, "base" if name == settings.base_model_name else name)
        return made[name]

    return model


def served_reader(settings: Any) -> Any:
    """The pinned MinerU at ``MINERU_URL``, in runs of ``RUN_PAGES`` pages."""
    from common.config import served_mineru
    from data_pipeline.ocr.served_ocr import ServedOcr

    pin = served_mineru()
    return ServedOcr(settings.mineru_url, settings.mineru_token, tier=pin["tier"],
                     expected_version=pin["version"], run_pages=settings.run_pages)


def load_runtime(settings: Any, registry: Any, *, models: Callable[[str], Any] | None = None,
                 reader: Any = None, links: Any = None) -> ServingRuntime:
    """Every tenant's release and model, the reader, the thresholds: the worker's cold start."""
    from common.config import serving_config
    from serving.vllm_entrypoint import assert_no_retired_settings, serving_thresholds

    assert_no_retired_settings(serving_config())
    models = models or openai_models(settings, links)
    tenants = {tenant: load_tenant(tenant, registry, settings, models) for tenant in settings.tenants}
    runtime = ServingRuntime(tenants=tenants, reader=reader or served_reader(settings), tuning=serving_thresholds())
    log.info("serving %s", runtime.describe())
    return runtime
