"""How a document is read, beside the document and the model (review A2, 2026-11-24).

:func:`serving.pipeline.extract` took these as thirteen keyword arguments. They are one
object now, built once per entry point and handed to every stage of the reading
(:mod:`serving.stages`). ``extract`` keeps its keyword arguments, and builds one of these
from them, until every caller passes a ``ReadSettings`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from common.constants import DEFAULT_REVIEW_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class ReadSettings:
    """Everything a reading takes beside the request, the model, the classifier and
    the v1 calibration. The defaults are ``extract``'s."""

    adapter_map: Mapping[str, Any] | None = None
    #: The release bundle's fitted calibrators and thresholds for the serving
    #: format in use (arch v2.1 §5.3). Per FORMAT, always: quantization moves the
    #: logprob distribution, so a bf16 calibrator reports confidence for a
    #: distribution FP8 does not produce.
    calibrators: Any = None
    thresholds: Any = None
    classifier_threshold: float = 0.70
    review_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD
    #: Raise on a schema-invalid response rather than return it flagged
    #: ``schema:invalid``. Production reads it from ``schema.invalid_answers``
    #: (configs/inference/vllm_serving.yaml, review R7); True is extract's default.
    strict_schema: bool = True
    #: The endpoint's ``ServingPlan``: a document type no promoted release covers
    #: is refused before any generation.
    plan: Any = None
    fallback_doc_type: str | None = None
    release_runtimes: Mapping[str, Any] | None = None
    #: Read the line of a policy that arrives with none (``pipeline._resolve_lob``).
    detect_lob: bool = False
    base_model_loaded: bool = True
    lob_confidence_threshold: float = 0.90
    lob_uncertain_threshold: float = 0.60

    # ------------------------------------------------------------------
    # The named profiles: each entry point reads with one, never with flags
    # ------------------------------------------------------------------

    @classmethod
    def served(cls, options: Mapping[str, Any] | None = None) -> ReadSettings:
        """The production service (the Azure worker, through ``serving.fields_first``): an
        answer that fails the schema is served flagged ``schema:invalid``, not refused;
        ``options`` are the serving thresholds (``vllm_entrypoint.serving_thresholds``) and
        the release's calibrators and thresholds (``azure_service.runtime.extract_options``)."""
        return cls(**{"strict_schema": False, **dict(options or {})})

    @classmethod
    def gate(cls, *, calibrators: Any = None, thresholds: Any = None) -> ReadSettings:
        """The gate's reading of the frozen set (``evaluation.golden_eval``) and the review
        queue a release builds (``orchestration.label_noise``): the release's calibrators,
        an invalid answer kept and flagged so it is scored, not lost."""
        return cls(calibrators=calibrators, thresholds=thresholds, strict_schema=False)

    @classmethod
    def testing(cls) -> ReadSettings:
        """The testing tools (``testing.run_extraction``, ``compare_models``,
        ``latency_check``): no release calibrators, an invalid answer kept and flagged."""
        return cls(strict_schema=False)

    @classmethod
    def draft(cls) -> ReadSettings:
        """A pre-annotation draft (``data_pipeline.labeling.pre_annotate``): a rough draft is
        the point - review fixes it - so an invalid answer is kept, flagged."""
        return cls(strict_schema=False)

    @classmethod
    def endpoint(cls, *, adapter_map: Mapping[str, Any] | None, plan: Any, release_runtimes: Mapping[str, Any] | None,
                 base_model_loaded: bool, tuning: Mapping[str, Any] | None = None) -> ReadSettings:
        """The RunPod endpoint (``serving.vllm_entrypoint.handler``): its serving plan, release
        runtimes and engine, and the serving thresholds - ``schema.invalid_answers`` among
        them, as the worker reads it (review R7)."""
        return cls(adapter_map=adapter_map, plan=plan, release_runtimes=release_runtimes,
                   base_model_loaded=base_model_loaded, **dict(tuning or {}))
