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
    #: Raise on a schema-invalid response rather than returning it (the audit gate).
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
