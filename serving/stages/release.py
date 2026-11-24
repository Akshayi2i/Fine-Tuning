"""Stage 2: the release that reads the document, its adapter and calibrators, and the schema
it is read against (``serving.pipeline.extract``, review A2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from serving import pipeline as P
from serving.stages.route import Routed


@dataclass
class Chosen:
    route: Any
    release: Any
    #: Read by the base model against the canonical fallback: no adapter for its line.
    lob_fallback_used: bool
    calibration: Any
    calibrators: Any
    thresholds: Any
    #: The line the schema is selected by (None on the fallback).
    lob: Any
    canonical: bool
    common_model: bool


def choose_release(routed: Routed, model: Any, calibration: Any, settings: Any) -> Chosen:
    request, route_, resolution = routed.request, routed.route, routed.resolution
    calibrators, thresholds = settings.calibrators, settings.thresholds
    release = None
    lob_fallback_used = fallback = False
    if settings.plan is not None:
        from serving.release_router import UnservedDocType

        try:
            # A policy no line could be found for is read by the base model
            # against the fallback, flagged - the caller need not opt in, since
            # the pipeline itself looked. A caller's unknown line is still refused
            # unless the caller asked for the fallback.
            routed_release = settings.plan.route(route_.doc_type, resolution.lob,
                                                 allow_fallback=request.allow_lob_fallback
                                                 or resolution.source == "undetected")
        except UnservedDocType as exc:
            raise P.PipelineError(str(exc)) from exc
        release = routed_release.release
        fallback = routed_release.lob_fallback_used
    elif route_.doc_type == "policy" and resolution.lob:
        # No serving plan - fields first, the golden eval, compare_models: a
        # line no schema has is refused as the plan refuses it, or read against
        # the fallback, flagged, when the caller asked. It used to be read
        # against the fallback with whatever adapter was routed, unflagged.
        from serving.release_router import _unroutable

        problem = _unroutable(resolution.lob)
        if problem and not request.allow_lob_fallback:
            raise P.PipelineError(f"{request.source_id}: {problem}")
        fallback = bool(problem)
    if fallback:
        if not settings.base_model_loaded:
            # The engine is one release's merged model: reading with "no
            # adapter" would be that release, against a schema it never
            # trained on, reported as the base model.
            raise P.PipelineError(
                f"{request.source_id} would be read by the base model against the fallback "
                "schema, but this endpoint's engine is the release's merged model, not the base: "
                "the release covers every known line, so the engine holds no base model. A "
                "policy with no line, a package or an undetected line cannot be read here; send "
                "its single line of business.")
        # The base model, no LoRA, against the canonical _fallback.json: no
        # adapter trained on this line's layout is promoted.
        lob_fallback_used = True
        from inference_core.model_runner import NO_ADAPTER

        route_ = P.replace(route_, adapter=NO_ADAPTER)
        # A release's calibrators were fitted to its adapter, not the base.
        calibrators = thresholds = None

    # Serve THROUGH the release the plan chose: its adapter and its calibrators.
    # The choice used to be a yes/no check and was then thrown away, so a
    # personal-lines release "answered" a homeowners policy that the unified
    # model then read, with the unified model's confidence.
    runtime = (settings.release_runtimes or {}).get(release.release_id) if release is not None else None
    if runtime is not None:
        if runtime.adapter is not None:
            route_ = P.replace(route_, adapter=runtime.adapter)
        if runtime.calibrators is not None:
            calibrators, thresholds = runtime.calibrators, runtime.thresholds

    # Calibration is per document type, and which type this is only becomes
    # known once the classifier has run — so it is selected here, not by the
    # caller. The endpoint used to resolve it from the caller-supplied
    # `doc_type`, which is absent on every classification-driven request; the
    # lookup fell through to `.get("")` and the request was refused before
    # extraction ever ran.
    # The v1 transform is needed only when no fitted calibrator set applies.
    if lob_fallback_used:
        # The base model has no fitted calibrators, and the endpoint may hold no
        # v1 parameters for the type: confidence is then raw. Every field is
        # flagged for review either way (no calibrator set below).
        try:
            calibration = P._calibration_for(calibration, route_.doc_type)
        except P.PipelineError:
            calibration = None
        if calibration is None:
            from calibration.fit_calibration import CalibrationParams

            # The identity: raw confidence, reported as a diagnostic only.
            calibration = CalibrationParams(method="temperature", doc_type=route_.doc_type,
                                            model_version=model.tag, temperature=1.0)
    else:
        calibration = P._calibration_for(calibration, route_.doc_type) if calibrators is None else None

    # The line selects the policy's canonical schema. It is the caller's to
    # supply; with none, `schema_key` selects the client's canonical fallback,
    # so a policy's output is canonical JSON either way.
    lob = None if lob_fallback_used else resolution.lob
    canonical = P.is_canonical(route_.schema_doc_type, route_.schema_acord_form, lob)
    common_model = canonical and route_.schema_doc_type == "policy" and P.is_common_model(
        route_.schema_doc_type, route_.schema_acord_form, lob)
    return Chosen(route=route_, release=release, lob_fallback_used=lob_fallback_used, calibration=calibration,
                  calibrators=calibrators, thresholds=thresholds, lob=lob, canonical=canonical,
                  common_model=common_model)
