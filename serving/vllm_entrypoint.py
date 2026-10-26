"""The RunPod Serverless handler (IMPL-11 §1, arch §14).

A persistent endpoint, unlike the ephemeral training pods: it is updated to a
promoted version and rolled back, not launched and destroyed.

Everything here is a thin shell around :func:`serving.pipeline.extract` and the
IMPL-07 inference core, so a served response is byte-for-byte what eval and the
testing harness produce for the same input. Generation logic in this file would
be a second implementation, and the numbers in the registry would then describe
the wrong one.

Three guards live here rather than deeper down, because this is the boundary
where a request enters the system:

**The MinerU pin is asserted at startup, not per request.** The model learned how
MinerU formats its output; serving OCR from a different version — or a different
*device*, if the Phase 0 spike shows the output differs — is the distribution
shift arch §8a exists to catch. A mismatch fails the cold start rather than
degrading quietly across every request afterwards.

**Missing calibration raises.** Returning raw logprob confidence when no
calibration exists for the served version would hand downstream review routing
numbers that look like probabilities and are not.

**Nothing plaintext reaches the logs, and no raw body is persisted.** Insurance
documents carry names, TINs, addresses and financials; a handler that logs its
request for debugging has exported all of it (master §8).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry.blob_client import BlobClient
from common.config import serving_config
from common.constants import ACTIVE_DOC_TYPES
from serving.pipeline import ExtractionRequest, ExtractionResult, extract
from serving.release_router import build_serving_plan

log = logging.getLogger(__name__)


class ServingError(RuntimeError):
    """Raised when a request cannot be served."""


class ColdStartError(ServingError):
    """Raised when the endpoint cannot come up safely.

    Distinct from a per-request failure: a cold start that fails must not fall
    back to serving anyway, because everything it checks is a whole-endpoint
    property — the wrong OCR version is wrong for every request, not this one.
    """


#: Keys never echoed back or logged. The document itself and its OCR text are
#: the payload, so they are exactly what must not appear in a log line.
SENSITIVE_KEYS = frozenset({
    "ocr_text", "page_texts", "images", "image_bytes", "pdf", "pdf_bytes", "extraction",
})


def safe_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """What may be logged about a request: shape, never content.

    Sizes and counts are enough to debug a malformed request. The values are
    what the compliance posture is about.
    """
    summary: dict[str, Any] = {}
    for key, value in sorted(payload.items()):
        if key in SENSITIVE_KEYS:
            if isinstance(value, (list, dict)):
                summary[key] = f"<{len(value)} item(s), not logged>"
            elif isinstance(value, (bytes, str)):
                summary[key] = f"<{len(value)} bytes, not logged>"
            else:
                summary[key] = "<not logged>"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
        else:
            summary[key] = f"<{type(value).__name__}>"
    return summary


@dataclass
class EndpointState:
    """What a warm endpoint holds. Built once, on cold start."""

    model_version: str
    model: Any = None
    classifier: Any = None
    calibration: Any = None
    adapter_map: dict[str, Any] = field(default_factory=dict)
    corpus_manifest: dict[str, Any] = field(default_factory=dict)
    #: release_id -> what serving a document through that release needs: its
    #: adapter (when the release is applied as a LoRA on the base) and its fitted
    #: calibrators and thresholds. Built at cold start from the serving plan.
    release_runtimes: dict[str, Any] = field(default_factory=dict)
    #: The engine holds the base model (the release applied as a LoRA), so a policy
    #: routed to no release can be read by the base model alone.
    base_model_loaded: bool = True

    #: The tenant's promoted release and how a document is routed to it, or to
    #: the base model (serving.release_router); it refuses a type the release
    #: does not cover.
    plan: Any = None
    ready: bool = False

    def release_for(self, doc_type: str, lob: Any = None) -> Any:
        """The release serving this document. Raises ``UnservedDocType``."""
        if self.plan is None:
            return None
        return self.plan.release_for(doc_type, lob)


def assert_ocr_pin(corpus_manifest: dict[str, Any]) -> None:
    """Fail the cold start when serving OCR differs from the corpus pin.

    Checked at startup rather than per request: it is a property of the
    deployment, and a per-request check would turn one configuration mistake
    into a per-request cost while still not fixing it.
    """
    from data_pipeline.ocr.mineru_version import assert_version_matches

    if not corpus_manifest.get("mineru_version"):
        raise ColdStartError(
            "the corpus manifest records no mineru_version, so the serving OCR version cannot be "
            "checked against what the model trained on. That pin is the arch §8a guarantee; "
            "serving without it means an OCR upgrade becomes an unattributable accuracy drop."
        )
    from data_pipeline.ocr.mineru_version import UNKNOWN, MinerUVersionError, get_mineru_version

    if get_mineru_version() == UNKNOWN:
        # This endpoint runs no OCR: requests arrive with their page texts. The pin
        # binds wherever those texts are produced (data_pipeline.ocr), not here —
        # comparing it against a MinerU the serving pod does not have refused every
        # deployment.
        log.info(
            "no MinerU on this endpoint; page texts must come from MinerU %s (the corpus pin)",
            corpus_manifest.get("mineru_version"),
        )
        return
    # The manifest is passed whole: assert_version_matches reads both
    # mineru_version and ocr_device from it, and splitting them here would mean
    # two callers with two ideas of what the pin covers.
    try:
        assert_version_matches(corpus_manifest)
    except MinerUVersionError as exc:
        raise ColdStartError(str(exc)) from exc


def calibration_for(state: EndpointState, doc_type: str | None) -> Any:
    """The calibration to serve this document with, or ``None``."""
    if not isinstance(state.calibration, dict):
        return state.calibration
    return state.calibration.get(doc_type or "")


def assert_calibration_present(calibration: Any, model_version: str, doc_type: str) -> None:
    """Refuse to serve without calibration parameters.

    Raw logprob confidence is systematically overconfident, and the review
    routing built on it would send the wrong documents to humans. A mapping is
    accepted whole — which entry applies is decided after classification, inside
    `extract` — but an empty one is no calibration at all.
    """
    if isinstance(calibration, Mapping):
        if not calibration:
            raise ServingError(
                f"no calibration parameters loaded for {model_version}. Fit them against the "
                "frozen golden eval set and push them before serving (IMPL-09)."
            )
        return
    if calibration is None:
        raise ServingError(
            f"no calibration parameters for {model_version}/{doc_type}. Raw logprob confidence is "
            "systematically overconfident, so serving it would route the wrong documents to "
            "review (IMPL-09)."
        )


def _adapter_exists(adapter_path: str, client: BlobClient) -> bool:
    """Whether an adapter directory actually holds weights.

    A staging path lives on the pod's volume and a published one in Blob, so
    both are checked: the marker file PEFT always writes is what distinguishes a
    real adapter from a prefix nobody ever wrote to.
    """
    marker = f"{adapter_path.rstrip('/')}/adapter_config.json"
    if adapter_path.startswith("/"):
        return Path(marker).exists()
    try:
        return bool(client.exists(marker))
    except Exception:  # noqa: BLE001 - an unreachable store is not a present adapter
        log.warning("could not confirm %s exists; treating it as absent", adapter_path)
        return False



def build_adapter_map(model_version: str, client: BlobClient) -> dict[str, str]:
    """doc_type -> adapter path, resolved from the registry.

    ``resolved["adapters"]`` was read here and ``ResolvedModel`` defines no such
    key, so the map was always empty: every document served Foundation-only and
    carried a `routing:no_adapter_available` flag. Resolving each type on its own
    is what the registry is for, and a type with no promoted adapter is simply
    absent from the map — which the router already treats as Foundation-only.
    """
    from registry_utils.query_registry import RegistryQueryError, resolve_model_version

    adapter_map: dict[str, str] = {}
    for doc_type in ACTIVE_DOC_TYPES:
        try:
            resolved = resolve_model_version(model_version, client, doc_type=doc_type)
        except (RegistryQueryError, KeyError, FileNotFoundError):
            continue
        adapter = resolved.get("type_adapter")
        # `type_adapter` is constructed unconditionally by resolve_model_version
        # whenever a doc_type is passed — it is a path, not evidence that
        # anything was trained. Trusting it mapped all three types on a
        # --foundation-only build, so vLLM was handed a LoRARequest for a
        # directory that does not exist and EVERY request failed, instead of
        # serving Foundation-only with a routing flag as documented above.
        if adapter and _adapter_exists(adapter, client):
            adapter_map[doc_type] = adapter

    if not adapter_map:
        log.warning(
            "no per-type adapters resolved for %s — every document will serve Foundation-only "
            "and carry a routing review flag. That is correct during the pilot, when the "
            "Foundation may be the only model trained, and a defect at any other time.",
            model_version,
        )
    return adapter_map


@dataclass
class ReleaseRuntime:
    """The promoted release, as a request is served through it."""

    release_id: str
    #: Local LoRA directory applied per request, or ``None`` for the engine's own
    #: weights (the release's merged model IS the engine).
    adapter: str | None = None
    calibrators: Any = None
    thresholds: Any = None


def serves_base_with_loras(plan: Any) -> bool:
    """Whether the engine loads the BASE with the release as its LoRA.

    Always for a release served as a LoRA on the base (every release since
    2026-10-23: nothing is merged). For an older merged release, whenever a
    known line routes to no release: such a policy is read by the base model
    with no LoRA (``lob_fallback_used``), and an engine holding the merged model
    has no base to read it with. Otherwise the engine is that merged model.
    """
    from common.lob import known_lines

    served = list(plan.served)
    if not served:
        return False
    if any(getattr(release, "served_as", "merged") == "lora_on_base" for release in served):
        return True
    return any(plan.route("policy", line).lob_fallback_used for line in known_lines())


def line_classifier(model: Any, plan: Any) -> Any:
    """The classifier that reads a policy's line when no caller sent one: the
    zero-shot classifier on the engine's base weights, with no adapter.

    The line decides which release's adapter reads the policy, so it is read
    before any adapter is chosen - by the base model, answering a prompt it was
    never trained on (serving.doc_type_classifier). None when the engine holds
    no base model: a release's merged model would read the line with that
    release's weights while reporting the base.
    """
    from inference_core.model_runner import NO_ADAPTER, generate
    from serving.doc_type_classifier import ZeroShotClassifier

    if not (serves_base_with_loras(plan) or getattr(model, "is_base", False)):
        return None
    return ZeroShotClassifier(model, generate, adapter=NO_ADAPTER)


def classifier_for(classifier: Any, model: Any, plan: Any, *, detect_lob: bool) -> Any:
    """The classifier a warm endpoint holds.

    The one it was given, when it was given one. Otherwise, with line detection
    on (``routing.detect_lob``), the base-model line classifier
    (:func:`line_classifier`), which also reads the type of a document sent with
    none; with it off, none, as before detection existed - so turning detection
    on is that one setting, once its accuracy has been measured
    (scripts/measure_lob_detection.py).
    """
    if classifier is not None or not detect_lob:
        return classifier
    built = line_classifier(model, plan)
    if built is None:
        log.warning(
            "routing.detect_lob is on, but this engine holds a release's merged model, not the "
            "base: no line is read, and a policy sent with none is routed as with detection off. "
            "Serve the releases as LoRAs on the base model to read lines."
        )
    return built


def load_release_calibration(release: Any, client: BlobClient) -> ReleaseRuntime:
    """A release's fitted calibrators and thresholds, read from Blob; no adapter.

    Also what the Azure worker (azure_service.runtime) serves with: the pod's
    vLLM applies the adapter there, by name.
    """
    from calibration.feature_calibrator import CalibratorSet
    from calibration.thresholds import ThresholdSet

    runtime = ReleaseRuntime(release_id=release.release_id)
    if release.calibrators and client.exists(release.calibrators):
        body = client.read_json(release.calibrators)
        runtime.calibrators = CalibratorSet.from_dict(body["calibrators"])
        runtime.thresholds = ThresholdSet.from_dict(body["thresholds"])
    else:
        log.warning(
            "release %s has no calibrators at %s; its fields are served with the v1 fallback",
            release.release_id, release.calibrators,
        )
    return runtime


def load_release_runtimes(
    plan: Any,
    client: BlobClient,
    *,
    adapter_root: str | Path,
) -> dict[str, ReleaseRuntime]:
    """The served release's adapter and calibration, ready for requests.

    **A release served as a LoRA on the base** (``lora_on_base``, every release
    since 2026-10-23: nothing is merged): the engine loads the BASE with LoRA
    enabled and the release is its adapter, applied per request. **An older
    merged release**, as before: covering every known line, the engine loads its
    merged model and no adapter is applied; leaving lines to the base model
    (:func:`serves_base_with_loras`), the engine loads the base and the release
    is applied as its LoRA. The adapter is copied from Blob to ``adapter_root``
    once, here.
    """
    served = list(plan.served)
    as_lora = serves_base_with_loras(plan)
    runtimes: dict[str, ReleaseRuntime] = {}
    for release in served:
        runtime = load_release_calibration(release, client)
        if as_lora:
            if not release.adapter_dir:
                raise ColdStartError(
                    f"release {release.release_id} records no adapter, so it cannot be served as a "
                    "LoRA on the base model beside the base-model fallback"
                )
            prefix = release.adapter_dir
            local = Path(adapter_root) / release.release_id
            client.download_dir(prefix, local)
            if not (local / "adapter_config.json").is_file():
                raise ColdStartError(
                    f"release {release.release_id}: no adapter at {prefix} (no adapter_config.json). "
                    "Serving it on the base model needs its LoRA; run `package` for it."
                )
            runtime.adapter = str(local)
        runtimes[release.release_id] = runtime
    return runtimes


def load_calibrations(model_version: str, client: BlobClient) -> dict[str, Any]:
    """doc_type -> calibration parameters for the served version.

    Left unset, the endpoint came up reporting "warm" and then rejected **every**
    request at :func:`assert_calibration_present` — an endpoint that cannot serve
    a single document while its own health check says it is ready.
    """
    from calibration.apply_calibration import load_calibration

    loaded: dict[str, Any] = {}
    for doc_type in ACTIVE_DOC_TYPES:
        try:
            loaded[doc_type] = load_calibration(model_version, doc_type, client)
        except Exception as exc:  # noqa: BLE001 - a missing fit is reported per request
            log.warning("no calibration for %s/%s: %s", model_version, doc_type, exc)
    return loaded


def cold_start(
    model_version: str,
    client: BlobClient,
    *,
    corpus_version: str | None = None,
    tenant_id: str | None = None,
    model: Any = None,
    classifier: Any = None,
    adapter_root: str | Path | None = None,
) -> EndpointState:
    """Pull the promoted artifact and validate the deployment before serving.

    Order matters: the pin check runs **before** the model is loaded, so a
    misconfigured deployment fails in seconds rather than after pulling 16 GB.
    """
    from artifact_registry import paths
    from registry_utils.query_registry import resolve_model_version

    plan = build_serving_plan(client, tenant_id=tenant_id)
    if not plan.served:
        # No release bundle promoted yet: the pre-release path, served by version.
        # Resolved for its side effect, so an unknown version fails the cold start
        # here rather than on the first request. With a promoted release the plan
        # decides.
        resolve_model_version(model_version, client)

    manifest: dict[str, Any] = {}
    if corpus_version:
        manifest_key = paths.corpus_manifest(corpus_version, tenant_id)
        if not client.exists(manifest_key):
            raise ColdStartError(
                f"corpus manifest for {corpus_version} is missing, so the OCR pin cannot be "
                "verified. Serving a model whose training-time OCR version is unknown is the "
                "distribution-shift risk in arch §8a with no way to detect it."
            )
        manifest = client.read_json(manifest_key)
        assert_ocr_pin(manifest)

    config = serving_config()
    assert_no_retired_settings(config)
    if model is None:
        raise ColdStartError(
            f"no model was supplied and the vLLM backend is not wired yet. Resolve it with "
            f"inference_core.model_runner.load_model({model_version!r}, client) once the Phase 0 "
            "spike confirms vLLM multi-LoRA hot-swap for Qwen3-VL; if it does not hold, IMPL-11 "
            "falls back to serving merged per-type models and only this function changes."
        )

    assert_prompt_hash(plan)
    assert_window_plan_served(plan)
    assert_base_model_served(plan)
    runtimes = load_release_runtimes(
        plan, client, adapter_root=adapter_root or Path(paths.staging_root()) / "serving-adapters",
    )
    base_loaded = serves_base_with_loras(plan)
    if base_loaded and not getattr(getattr(model, "config", None), "enable_lora", True):
        raise ColdStartError(
            f"{len(runtimes)} release(s) are served as LoRAs on the base (lines left to the base "
            "model), but the model was loaded without LoRA support (enable_lora: false)."
        )
    state = EndpointState(
        model_version=model_version,
        model=model,
        classifier=classifier_for(
            classifier, model, plan, detect_lob=bool(serving_thresholds().get("detect_lob"))),
        adapter_map=build_adapter_map(model_version, client),
        calibration=load_calibrations(model_version, client),
        plan=plan,
        corpus_manifest=manifest,
        release_runtimes=runtimes,
        # Whether "no adapter" reads the base model: only when the engine is the
        # base, never a release's merged model.
        base_model_loaded=base_loaded or not plan.served,
        ready=True,
    )
    log.info(
        "endpoint warm: version=%s adapters=%s logprobs=%s",
        model_version, sorted(state.adapter_map), config.get("logprobs", True),
    )
    return state


#: Serving settings removed on 2026-10-22, and what replaced each.
RETIRED_ROUTING_SETTINGS: dict[str, str] = {
    "release_pins": "a rollback is `python -m orchestration.run promote --version vK --by NAME`",
    "lob_family_threshold": "it is `lob_uncertain_threshold` now",
}


def assert_no_retired_settings(config: dict[str, Any]) -> None:
    """Refuse a serving config that still carries a removed setting.

    Read by nothing, it would be ignored without a word: a rollback pin that no
    longer rolls back, a tuned threshold quietly back at its default.
    """
    routing = config.get("routing") or {}
    found = {key: why for key, why in RETIRED_ROUTING_SETTINGS.items() if key in routing}
    if found:
        raise ColdStartError(
            "configs/inference/vllm_serving.yaml routing holds removed setting(s) "
            + "; ".join(f"{key} ({why})" for key, why in found.items())
            + ". Remove them so the endpoint does what its config says."
        )


def assert_prompt_hash(plan: Any) -> None:
    """Refuse to serve a release trained on prompts this code does not render.

    Each release records the hash of every prompt input at packaging time. A
    template, schema or section-map edit since then means this deployment shows
    the model a prompt it was never trained on — the output degrades with every
    request still returning well-formed JSON. Checked at cold start, before the
    first document. A release with no hash (written before it was recorded) is
    warned about, not refused: there is nothing to compare.
    """
    from common.prompts import prompt_hash

    current = prompt_hash()
    for release in plan.served:
        if not release.prompt_hash:
            log.warning(
                "%s records no prompt hash, so prompt drift since it was packaged cannot be "
                "checked", release.release_id,
            )
        elif release.prompt_hash != current:
            raise ColdStartError(
                f"{release.release_id} was packaged with prompt hash {release.prompt_hash[:12]}, "
                f"and this code renders {current[:12]}: a template, schema or section map changed "
                "since it was trained. Serve the code it was packaged from, or retrain."
            )


def assert_window_plan_served(plan: Any) -> None:
    """Refuse to serve a policy release cut into other windows than this code reads.

    Checked at cold start with the prompt hash, which cannot tell: a window rule
    or a token budget changes where a policy's windows fall, not what any prompt
    says. A release serving policies records the window plan of the corpus it
    was trained on (``ReleaseBundle.window_plan_version``); one that records none
    was packaged before every page was read and is refused too.
    """
    from data_pipeline.dataset_builder.policy_windows import window_plan_id

    current = window_plan_id()
    for release in plan.served:
        if release.doc_types and "policy" not in release.doc_types:

            continue
        if release.window_plan_version != current:
            raise ColdStartError(
                f"{release.release_id} was trained on policies cut with window plan "
                f"{release.window_plan_version or 'unrecorded'}, and this code reads them with "
                f"{current}: serving it would show the model windows it never trained on. Serve the "
                "code it was packaged from, or rebuild the corpus and retrain."
            )


def assert_base_model_served(plan: Any) -> None:
    """Refuse a release served as a LoRA on the base whose adapter trained on another base.

    Checked at cold start with the prompt hash and the window plan, which cannot
    tell: a bumped base revision pin (configs/base_model.yaml) changes the
    weights under the adapter, not any prompt or window. The adapter would load
    and answer with no error. A merged release carries its own base, so only
    ``lora_on_base`` releases are checked (``query_registry.base_model_refusal``).
    """
    from registry_utils.query_registry import base_model_refusal

    for release in plan.served:
        if getattr(release, "served_as", "merged") != "lora_on_base":
            continue
        refusal = base_model_refusal(f"release {release.release_id}", release.base_model)
        if refusal:
            raise ColdStartError(refusal)


def build_request(payload: dict[str, Any]) -> ExtractionRequest:
    """Validate a request body into an :class:`ExtractionRequest`."""
    source_id = payload.get("source_id")
    images = payload.get("image_paths") or []
    if not source_id:
        raise ServingError("request is missing source_id, so its result could not be attributed")
    if not images:
        raise ServingError(
            "request supplies no page images. Both production modes need the image: image_only "
            "has nothing else, and ocr_plus_image arbitrates between the two (arch §6)."
        )

    mode = payload.get("modality_mode", "ocr_plus_image")
    ocr_text = payload.get("ocr_text")
    if mode == "image_only" and ocr_text:
        raise ServingError(
            "image_only was requested with OCR text attached. Serving OCR under the image-only "
            "prompt would mean the model is told no text exists while text is present, which is "
            "not a state it was trained on."
        )

    try:
        page_texts = {int(k): v for k, v in (payload.get("page_texts") or {}).items()}
    except (TypeError, ValueError) as exc:
        raise ServingError(f"page_texts keys must be page numbers: {exc}") from exc
    # One text per image, numbered 1..n. Anything else pairs a page's text with
    # another page's image — page 3's text read against page 4 — and the model
    # reports a value on the wrong page with full confidence.
    if page_texts and sorted(page_texts) != list(range(1, len(images) + 1)):
        raise ServingError(
            f"page_texts covers pages {sorted(page_texts)} but {len(images)} page image(s) were "
            f"sent; it must hold exactly pages 1..{len(images)}, one text per image."
        )

    return ExtractionRequest(
        source_id=str(source_id),
        image_paths=[str(p) for p in images],
        ocr_text=ocr_text,
        page_texts=page_texts,
        ocr_meta=payload.get("ocr_meta") or {},
        modality_mode=mode,
        known_doc_type=payload.get("doc_type"),
        known_acord_form=payload.get("acord_form"),
        known_lob=request_lob(payload),
        lob_hypothesis=lob_hypothesis(payload),
        allow_lob_fallback=bool(payload.get("allow_lob_fallback", False)),
        # Serving fills document.source_file_name from this (the model is never
        # asked for it); without it the served output never carried the field.
        source_file_name=_file_name(payload),
    )


def _file_name(payload: Mapping[str, Any]) -> str | None:
    """The document's file name as the caller supplies it, or ``None``."""
    name = payload.get("source_file_name", payload.get("file_name"))
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip():
        raise ServingError(f"source_file_name must be a non-empty string, got {name!r}")
    return name.strip()


def request_lob(payload: Mapping[str, Any]) -> str | list[str] | None:
    """The policy's line of business, as the caller (L1/L2) supplies it.

    ``lob`` (or ``line_of_business``): one line, or a list for a package policy.
    It selects the line's canonical schema. Nothing set it before, so every
    served policy was read against the canonical fallback while training had
    read it against its line's schema — the model was trained on one output
    shape and served another. A single line with no schema is refused: falling
    back quietly is the same mismatch with a name that looks right - unless the
    caller asked for the fallback (``allow_lob_fallback``).
    """
    from common.schemas import schema_selectors

    lob = payload.get("lob", payload.get("line_of_business"))
    if lob is None:
        return None
    lines = [lob] if isinstance(lob, str) else lob
    if not isinstance(lines, list) or not lines or not all(
        isinstance(line, str) and line.strip() for line in lines
    ):
        raise ServingError(f"lob must be a line name or a non-empty list of them, got {lob!r}")
    # Classic auto is read as personal auto (common.lob.MERGED_LINES) - here
    # too, or a caller naming it is refused for a line that has no schema of
    # its own.
    from common.lob import merge_line

    lines = merge_line([line.strip().lower() for line in lines])
    lob = lines[0] if isinstance(lob, str) else lines
    if len(lines) == 1 and not payload.get("allow_lob_fallback"):
        # Unless the caller chose the base-model fallback: then a line with no
        # schema is read against _fallback.json, flagged (ServingPlan.route).
        from common.schemas import LOB_SCHEMA_ALIASES

        line = lines[0]
        known = {q for doc_type, _, q in schema_selectors() if doc_type == "policy" and q}
        if LOB_SCHEMA_ALIASES.get(line, line) not in known:
            raise ServingError(
                f"no canonical policy schema for line {lob!r}; known lines: "
                f"{sorted(known | set(LOB_SCHEMA_ALIASES))}"
            )
    return lob


def lob_hypothesis(payload: Mapping[str, Any]) -> str | None:
    """The line L1/L2 suspects without being sure enough to send it as ``lob``.

    ``lob_hypothesis``: one line name, or absent. Reconciled with the line the
    classifier reads (``combine_lob_with_hypothesis``): agreement lifts an
    uncertain reading, a disagreement becomes a candidate. Nothing read it
    before, so the hint the pipeline reconciles never arrived from a caller.
    Normalised as ``lob`` is (classic auto is personal auto); a line with no
    schema is passed on rather than refused - a hint is evidence, not a
    selector, and the reconciliation ignores a line it does not know.
    """
    hint = payload.get("lob_hypothesis")
    if hint is None:
        return None
    if not isinstance(hint, str) or not hint.strip():
        raise ServingError(f"lob_hypothesis must be one line name, got {hint!r}")
    from common.lob import merge_line

    return str(merge_line(hint.strip().lower()))


def serving_thresholds() -> dict[str, Any]:
    """The tuneable thresholds, from configs/inference/vllm_serving.yaml.

    Each key was present in that file and read by no code, so an operator could
    edit it, redeploy, and see no change. Only keys `extract` accepts are
    returned, so an unknown one in the YAML is ignored rather than crashing a
    cold start.
    """
    config = serving_config()
    routing = config.get("routing") or {}
    confidence = config.get("confidence") or {}

    tuning: dict[str, Any] = {}
    for key, value in (
        ("classifier_threshold", routing.get("classifier_confidence_threshold")),
        ("review_threshold", confidence.get("review_threshold")),
        ("lob_confidence_threshold", routing.get("lob_confidence_threshold")),
        ("lob_uncertain_threshold", routing.get("lob_uncertain_threshold")),
    ):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tuning[key] = value
    if isinstance(routing.get("detect_lob"), bool):
        tuning["detect_lob"] = routing["detect_lob"]

    # An operator-chosen schema for a document nobody could identify. Absent by
    # default, and then an unroutable document is refused rather than extracted
    # against a schema nobody chose.
    fallback = routing.get("fallback_doc_type")
    if isinstance(fallback, str) and fallback.strip():
        tuning["fallback_doc_type"] = fallback.strip()
    return tuning


def handler(event: dict[str, Any], state: EndpointState) -> dict[str, Any]:
    """The RunPod Serverless entrypoint. One request in, the output contract out.

    Errors are returned as a structured payload rather than raised, because a
    serverless handler that raises returns an opaque platform error and the
    caller learns nothing about which of its inputs was wrong.
    """
    payload = event.get("input") or {}

    try:
        # Inside the try. `safe_log_payload` iterates the payload, so an `input`
        # that is not a JSON object — a bare string, a list, a number — raised
        # AttributeError straight out of the handler, and RunPod returned an
        # opaque platform error saying nothing about which input was wrong. That
        # is the exact failure this handler exists to prevent.
        log.info("request %s", safe_log_payload(payload))

        if not state.ready:
            return {"error": "endpoint is not warm", "retryable": True}

        request = build_request(payload)
        # The whole calibration map is handed to `extract`, which resolves the
        # right one once the classifier has said what the document is.
        # Resolving here from `request.known_doc_type` meant `.get("")` for
        # every request that did not name its own doc_type — which is every
        # classification-driven request, the endpoint's entire purpose — and
        # `assert_calibration_present` then refused it before extraction ran.
        runtimes = state.release_runtimes
        if not runtimes or not all(r.calibrators is not None for r in runtimes.values()):
            # The v1 per-version calibration is the fallback for releases without
            # their own calibrators; it has to exist when anything falls back.
            assert_calibration_present(
                state.calibration, state.model_version, request.known_doc_type or "any"
            )
        # The thresholds come from configs/inference/vllm_serving.yaml. They
        # were declared there and read by nothing, so raising review_threshold
        # and redeploying changed no behaviour at all — the hardcoded defaults
        # inside `extract` won every time, silently.
        tuning = serving_thresholds()
        result: ExtractionResult = extract(
            request,
            state.model,
            state.classifier,
            state.calibration,
            adapter_map=state.adapter_map,
            # The serving plan reaches `extract`, or the refusal it exists for
            # never happens: a document type no promoted release covers would be
            # extracted by a model that never trained on it, against a schema it
            # has never seen (arch v2.1 §12.3).
            plan=state.plan,
            release_runtimes=state.release_runtimes,
            base_model_loaded=state.base_model_loaded,
            **tuning,
        )
    except ServingError as exc:
        # The message names what was wrong with the request; it never echoes the
        # request, which is what keeps document content out of the logs.
        log.warning("request rejected: %s", exc)
        return {"error": str(exc), "retryable": False}
    except Exception as exc:  # noqa: BLE001 - a handler must never raise
        # Non-JSON output and schema-invalid output are the two most likely real
        # failures and both arrive as PipelineError, which is not a ServingError.
        # Letting them escape returned an opaque platform error carrying nothing
        # about which input was wrong — the exact thing this handler exists to
        # prevent. The exception type is named; the request never is.
        log.error("extraction failed (%s): %s", type(exc).__name__, exc)
        return {"error": f"{type(exc).__name__}: {exc}", "retryable": False}

    # `as_dict` is the master §9 output contract. Nothing raw is persisted here:
    # the caller owns the document, and the endpoint keeps no copy.
    return {"output": result.as_dict()}
