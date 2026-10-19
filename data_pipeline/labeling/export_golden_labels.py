"""Writing verified golden labels, with validation and provenance (IMPL-04).

A golden label is the training target. Everything the model learns about field
semantics comes from these, so admission is gated: a label that validates against
the schema but names the wrong entity trains the model to make that exact mistake
confidently.

Three checks run before anything is written:

1. **Schema conformance** against the canonical Fideon SPEC_00 schema (arch §0a).
2. **``line_of_business`` present**, a list of valid values or explicitly ``[]`` (arch §0b).
3. **``field_provenance`` names no registered confusable** — a label claiming
   ``insured_name`` was found under "Certificate Holder" is rejected. This is the
   highest-value annotation check in the system, because that mistake is exactly
   what teaches the model to conflate distinct parties.

The **day-zero rule** (arch §4c) also lives here rather than in its own module:
it is a counter and a threshold, and it belongs beside the thing it gates.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any, Literal

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common import aliases
from common.canonical import field_paths
from common.constants import DAY_ZERO_MIN_LABELS_PER_TYPE
from common.lob import LobError, validate_lob
from common.schemas import SchemaError, is_canonical, is_valid, iter_validation_errors

log = logging.getLogger(__name__)

ReviewRequirement = Literal["full", "confidence_routed"]


class LabelValidationError(ValueError):
    """Raised when a golden label is not fit to enter the corpus."""


# --------------------------------------------------------------------------
# Day-zero bootstrap (arch §4c)
# --------------------------------------------------------------------------

def count_golden_labels(client: BlobClient, doc_type: str, tenant_id: str | None = None) -> int:
    """How many verified labels exist for a document type."""
    prefix = f"golden-labels/{paths._tenant(tenant_id)}/{doc_type}/"
    return sum(1 for key in client.list(prefix) if key.endswith("golden.json"))


def review_requirement(
    client: BlobClient,
    doc_type: str,
    tenant_id: str | None = None,
    *,
    minimum: int = DAY_ZERO_MIN_LABELS_PER_TYPE,
) -> ReviewRequirement:
    """Whether every draft still needs full human review.

    On day zero no fine-tuned Foundation exists, so drafts come from the *base*
    model and **every one is corrected by a human** until the corpus reaches
    ``minimum`` labels for this type. Confidence routing before that point would
    be routing on a model with no evidence it is trustworthy.

    Explicitly temporary: once Foundation v1.0 is promoted it becomes the
    pre-annotator and the base model leaves the production path.
    """
    return "full" if count_golden_labels(client, doc_type, tenant_id) < minimum else "confidence_routed"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_golden_label(
    label: dict[str, Any],
    doc_type: str,
    *,
    acord_form: str | None = None,
    field_provenance: dict[str, str] | None = None,
    lob: str | list[str] | None = None,
) -> None:
    """Run every admission check. Raises :class:`LabelValidationError`.

    ``lob`` selects a policy's canonical schema. A canonical label carries no
    top-level ``line_of_business`` — the client's schema has none — so for one
    the line is the one given here, from the label's metadata.
    """
    problems: list[str] = []

    # This check comes FIRST: schema *selection* depends on the form, so
    # validating without one raises a low-level SchemaError instead of the
    # actionable message the annotator needs.
    if doc_type == "acord" and not acord_form:
        raise LabelValidationError(
            "an ACORD label needs its acord_form (25 | 125 | 140). The form selects the schema, "
            "which is why classification is two-level (arch §4b) — without it there is no schema "
            "to validate against."
        )

    try:
        if not is_valid(label, doc_type, acord_form, lob):
            problems.extend(iter_validation_errors(label, doc_type, acord_form, lob))
        canonical = is_canonical(doc_type, acord_form, lob)
    except SchemaError as exc:
        problems.append(str(exc))
        canonical = False

    if canonical:
        # The line travels in the metadata, beside the client's label rather than
        # inside it. It names a canonical schema (flood, gl, cyber …), not a value
        # of the LOB enum the model outputs, so it is checked against the lines
        # that have a schema; absent is allowed, because the canonical fallback
        # exists for exactly that case.
        if lob:
            from common.scopes import known_lines, lob_lines

            unknown = sorted(lob_lines(lob) - known_lines())
            if not lob_lines(lob) or unknown:
                problems.append(
                    f"line of business {lob!r} names no canonical policy schema "
                    f"({unknown or 'empty'}); known lines: {sorted(known_lines())}"
                )
    elif "line_of_business" not in label:
        problems.append(
            "line_of_business is missing. It is required in every golden label for every "
            "document type, even when empty — the VLM is the fallback LoB detector when L1/L2 "
            "miss (arch §0b). An empty list means the document determines no line; that is a "
            "correct label, and it is not the same as omitting the field."
        )
    else:
        try:
            validate_lob(label["line_of_business"])
        except LobError as exc:
            problems.append(str(exc))

    # A canonical label nests its fields (`named_insured.primary_name`), so a
    # provenance entry names a path; a flat label's fields are its top level.
    present = field_paths(label)
    for field, surface_label in (field_provenance or {}).items():
        if field not in present:
            problems.append(
                f"field_provenance names {field!r}, which is not in the label"
            )
        # The alias registry is keyed by canonical field name, not by where the
        # field sits, so a nested path is checked by its leaf.
        elif aliases.is_confusable(doc_type, field.rsplit(".", 1)[-1], surface_label):
            problems.append(
                f"field_provenance says {field!r} was found under {surface_label!r}, which is a "
                f"registered CONFUSABLE for that field. Accepting this would train the model to "
                f"conflate two distinct parties — and it would train perfectly happily."
            )

    if problems:
        raise LabelValidationError(
            f"golden label for {doc_type}"
            + (f"/{acord_form}" if acord_form else "")
            + " rejected:\n  - " + "\n  - ".join(problems)
        )


#: A gold's own notes (``fideon:provenance``) naming a model as its drafter.
_MODEL_DRAFTED = re.compile(
    r"\b(?:drafted|generated|written|labell?ed|annotated|pre-annotated|extracted)\b[^.;|]{0,40}?"
    r"\bby\s+(?:an?\s+)?(?:claude|gpt|chatgpt|openai|anthropic|gemini|llama|mistral|copilot|qwen"
    r"|(?:ai|llm|language model)\b)",
    re.IGNORECASE,
)
#: ... or saying, in so many words, that nobody reviewed it.
_NOT_REVIEWED = re.compile(r"\bnot\s+reviewed\b|\bunreviewed\b", re.IGNORECASE)
#: ``draft_backend`` values that name no model: a person wrote the label.
_NO_MODEL = frozenset({"", "unknown", "human", "manual", "none"})


def review_problem(label: Any, metadata: dict[str, Any]) -> str | None:
    """Why a gold may not train or score yet; ``None`` when it may.

    A gold a model drafted counts only once a person has reviewed it: its
    metadata records ``reviewer_id`` and ``review_date`` (decision 2026-10-19 -
    frontier models are never used for gold, synthetic or training data, and a
    draft is never trusted as-is). Model-drafted is read from the metadata's
    ``draft_backend`` and from the gold's own notes: one naming a model as the
    drafter, or saying it was not reviewed (a synthetic twin's ``seed_review``).
    """
    if metadata.get("reviewer_id") and metadata.get("review_date"):
        return None
    evidence = _model_drafted(label, metadata)
    if evidence is None:
        return None
    return (f"{evidence}, and no reviewer is recorded: a person must review it and record "
            "reviewer_id and review_date in its metadata before it trains or scores")


def has_review_record(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("reviewer_id") and metadata.get("review_date"))


def _model_drafted(label: Any, metadata: dict[str, Any]) -> str | None:
    backend = str(metadata.get("draft_backend") or "").strip()
    if backend.lower() not in _NO_MODEL:
        return f"drafted by {backend}"
    notes = label.get("fideon:provenance") if isinstance(label, dict) else None
    values = notes.values() if isinstance(notes, dict) else [notes]
    texts = [value for value in values if isinstance(value, str)]
    if any(_NOT_REVIEWED.search(text) for text in texts):
        return "its notes say it was not reviewed"
    for text in texts:
        if match := _MODEL_DRAFTED.search(text):
            return f"its notes say it was {' '.join(match.group(0).split())}"
    return None


def export_golden_label(
    label: dict[str, Any],
    source_id: str,
    doc_type: str,
    client: BlobClient,
    *,
    acord_form: str | None = None,
    tenant_id: str | None = None,
    reviewer_id: str,
    draft_backend: str = "unknown",
    field_provenance: dict[str, str] | None = None,
    double_annotated: bool = False,
    agreement_score: float | None = None,
    accepted_without_review: bool = False,
    lob: str | list[str] | None = None,
) -> str:
    """Validate and write a golden label plus its provenance metadata.

    Args:
        accepted_without_review: the draft was taken as-is. Refused while the
            day-zero requirement is ``"full"``.

    Returns:
        The blob key of the written label.
    """
    requirement = review_requirement(client, doc_type, tenant_id)
    if accepted_without_review and requirement == "full":
        raise LabelValidationError(
            f"a draft was accepted without review, but {doc_type} has fewer than "
            f"{DAY_ZERO_MIN_LABELS_PER_TYPE} labels so every draft still requires full human "
            "correction (arch §4c). Drafts at this stage come from the untuned base model; "
            "trusting one would put its errors into the training target."
        )

    validate_golden_label(
        label, doc_type, acord_form=acord_form, field_provenance=field_provenance, lob=lob
    )

    label_key = paths.golden_label(doc_type, source_id, tenant_id)
    client.write_json(label_key, label)

    metadata = {
        "source_id": source_id,
        "doc_type": doc_type,
        "acord_form": acord_form,
        # Where the dataset build reads a policy's line from, to select the
        # canonical schema its prompt and training target use.
        "lob": lob,
        "reviewer_id": reviewer_id,
        "review_date": datetime.now(UTC).isoformat(),
        "draft_backend": draft_backend,
        "double_annotated": double_annotated,
        "agreement_score": agreement_score,
        "review_requirement": requirement,
        "accepted_without_review": accepted_without_review,
        # The observed surface label per canonical field. Golden keys are always
        # canonical; this records what the document actually said, which is what
        # lets IMPL-08 slice accuracy per surface variant (arch §0c).
        "field_provenance": field_provenance or {},
    }
    client.write_json(paths.label_metadata(doc_type, source_id, tenant_id), metadata)

    log.info("exported golden label for %s (reviewer=%s, requirement=%s)",
             source_id, reviewer_id, requirement)
    return label_key


def load_golden_label(
    source_id: str, doc_type: str, client: BlobClient, tenant_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a label and its metadata together — they are only meaningful as a pair."""
    return (
        client.read_json(paths.golden_label(doc_type, source_id, tenant_id)),
        client.read_json(paths.label_metadata(doc_type, source_id, tenant_id)),
    )


def list_labeled_source_ids(
    client: BlobClient, doc_type: str, tenant_id: str | None = None
) -> list[str]:
    """Every source_id with a verified golden label.

    This is what ``finetune`` builds the corpus from — and the complement of it
    is the unlabeled backlog the command reports (IMPL-13 §2).
    """
    prefix = f"golden-labels/{paths._tenant(tenant_id)}/{doc_type}/"
    return sorted(
        key.split("/")[3] for key in client.list(prefix) if key.endswith("golden.json")
    )


def inter_annotator_agreement(
    label_a: dict[str, Any], label_b: dict[str, Any]
) -> tuple[float, list[str]]:
    """Field-level agreement between two independent annotations.

    Run on 10-20% of documents. It measures the noise in your *labeling process*,
    which sets a realistic ceiling on model scores — some of the residual error at
    plateau is human disagreement, not model failure (arch §7).

    Returns ``(agreement, disagreeing_fields)``.
    """
    from common.canonical import has_envelopes, leaf_values
    from common.normalize import values_match

    # A canonical label nests every value in an envelope under an object, so
    # comparing its top-level keys would compare whole objects — and the
    # annotators' own confidence entries with them. Compared field by field on
    # values instead. A flat label keeps its top-level comparison.
    if has_envelopes(label_a) or has_envelopes(label_b):
        label_a, label_b = leaf_values(label_a), leaf_values(label_b)

    fields = sorted(set(label_a) | set(label_b))
    if not fields:
        return 1.0, []
    disagreements = [
        f for f in fields
        if not values_match(label_a.get(f), label_b.get(f), field_path=f)
    ]
    return (len(fields) - len(disagreements)) / len(fields), disagreements
