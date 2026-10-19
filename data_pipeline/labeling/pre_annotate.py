"""First-pass ("silver") drafts for human review (IMPL-04 §1, arch §7).

Nobody hand-types JSON from scratch. A corrected draft is faster and more
consistent than a blank form, and the draft improves every cycle as the model
does — which is what makes labeling cost fall over successive corpus versions
instead of staying flat.

Two backends, both self-hosted, in the order a project uses them:

======================  =========================================
backend                 when
======================  =========================================
``base_qwen3vl``        day zero, before any Foundation exists
``own_finetuned``       once v1 is promoted (the promoted version)
======================  =========================================

**No frontier model, ever** (decision 2026-10-19): frontier models are never
used for gold, synthetic or training data. There is no external backend, and
asking for one by name is refused - it is not a missing permission that a flag
could grant.

**The draft is never trusted as-is.** On day zero every draft is corrected by a
human — see :func:`export_golden_labels.review_requirement`, which is a counter
and a threshold rather than a policy anyone has to remember.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES

log = logging.getLogger(__name__)

Backend = Literal["base_qwen3vl", "own_finetuned"]

BACKENDS: tuple[Backend, ...] = ("base_qwen3vl", "own_finetuned")
DEFAULT_BACKEND: Backend = "base_qwen3vl"


class PreAnnotationError(RuntimeError):
    """Raised when a draft cannot be produced."""


def check_backend(backend: str) -> None:
    """Refuse any backend but the two self-hosted ones.

    A frontier model by name is refused with the decision behind it: it is not a
    backend this pipeline can be configured to use.
    """
    if backend in BACKENDS:
        return
    if "frontier" in str(backend) or "external" in str(backend):
        raise PreAnnotationError(
            f"{backend!r} is refused: frontier models are never used for gold, synthetic or "
            f"training data (decision 2026-10-19). Draft with {DEFAULT_BACKEND!r} or "
            "'own_finetuned', and have a person review every draft."
        )
    raise PreAnnotationError(f"unknown backend {backend!r}; expected one of {list(BACKENDS)}")


@dataclass
class Draft:
    """One pre-annotated document, with everything review needs to judge it."""

    source_id: str
    doc_type: str
    backend: Backend
    model_version: str
    extraction: dict[str, Any] = field(default_factory=dict)
    field_confidence: dict[str, float] = field(default_factory=dict)
    acord_form: str | None = None
    review_requirement: str = "full"
    schema_valid: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def trusted(self) -> bool:
        """Always ``False``. A draft is a starting point, never a label.

        Kept as an explicit property rather than left implicit so that any code
        reaching for "can I skip review on this one?" gets an answer instead of
        inventing a threshold.
        """
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "doc_type": self.doc_type,
            "acord_form": self.acord_form,
            "draft_backend": self.backend,
            "model_version": self.model_version,
            "review_requirement": self.review_requirement,
            "schema_valid": self.schema_valid,
            "created_at": self.created_at,
            "extraction": self.extraction,
            "field_confidence": self.field_confidence,
            "trusted": self.trusted,
        }


def resolve_backend_model(
    backend: Backend, client: BlobClient, *, doc_type: str | None = None
) -> str:
    """The model version a backend draws on."""
    if backend == "base_qwen3vl":
        return "base"
    if backend == "own_finetuned":
        from registry_utils.query_registry import latest_promoted

        promoted = latest_promoted(client, "foundation")
        if not promoted:
            raise PreAnnotationError(
                "no promoted Foundation exists, so there is nothing to draft with. This is the "
                f"day-zero case — use the {DEFAULT_BACKEND!r} backend, which is why it is the "
                "default (arch §4c)."
            )
        return promoted.replace("foundation-", "")
    check_backend(backend)
    raise PreAnnotationError(f"no model for backend {backend!r}")


def pre_annotate(
    source_id: str,
    doc_type: str,
    client: BlobClient,
    *,
    backend: Backend = DEFAULT_BACKEND,
    model: Any = None,
    classifier: Any = None,
    calibration: Any = None,
    tenant_id: str | None = None,
    acord_form: str | None = None,
) -> Draft:
    """Produce one draft for review.

    Extraction runs through the serving pipeline — the same path production uses
    — so a draft is exactly what the model would return in production, and a
    reviewer correcting it is correcting the real failure mode rather than an
    artefact of a separate drafting implementation.
    """
    if doc_type not in ACTIVE_DOC_TYPES:
        raise PreAnnotationError(f"unknown doc_type {doc_type!r}; expected one of {list(ACTIVE_DOC_TYPES)}")
    check_backend(backend)

    from data_pipeline.labeling.export_golden_labels import review_requirement
    from serving.pipeline import ExtractionRequest, extract

    model_version = resolve_backend_model(backend, client, doc_type=doc_type)
    requirement = review_requirement(client, doc_type, tenant_id)

    ocr_meta_key = paths.ocr_meta(doc_type, source_id, tenant_id)
    if not client.exists(ocr_meta_key):
        raise PreAnnotationError(
            f"{source_id} has not been OCR'd, so there is nothing to draft from. Run IMPL-03 "
            "preprocessing first."
        )
    ocr_meta = client.read_json(ocr_meta_key)
    page_count = int(ocr_meta.get("page_count", 1))
    render_only = bool(ocr_meta.get("render_only"))

    request = ExtractionRequest(
        source_id=source_id,
        image_paths=[
            paths.processed_page(doc_type, source_id, page, "png", tenant_id)
            for page in range(1, page_count + 1)
        ],
        page_texts={} if render_only else {
            page: client.read_text(
                paths.processed_page(doc_type, source_id, page, "md", tenant_id)
            )
            for page in range(1, page_count + 1)
        },
        ocr_meta=ocr_meta,
        modality_mode="image_only" if render_only else "ocr_plus_image",
        known_doc_type=doc_type,
        known_acord_form=acord_form,
    )

    if model is None:
        raise PreAnnotationError(
            "pre_annotate needs a loaded model. Resolve one with "
            f"inference_core.model_runner.load_model({model_version!r}, client) — which is still "
            "waiting on a live backend (Phase 0) — and pass it in. The drafting logic and the "
            "review rule around it are complete."
        )

    result = extract(
        request, model, classifier, calibration,
        strict_schema=False,  # a rough draft is the point; review fixes it
    )

    draft = Draft(
        source_id=source_id,
        doc_type=doc_type,
        backend=backend,
        model_version=model_version,
        extraction=result.extraction,
        field_confidence={
            path: value.get("confidence", 0.0)
            for path, value in result.fields.items()
            if isinstance(value, dict)
        },
        acord_form=acord_form,
        review_requirement=requirement,
        schema_valid=result.schema_valid,
    )
    log.info(
        "drafted %s with %s (%s); review requirement: %s",
        source_id, backend, model_version, requirement,
    )
    return draft


def write_draft(draft: Draft, client: BlobClient, tenant_id: str | None = None) -> str:
    """Store the draft next to where its golden label will land.

    Deliberately **not** at the ``golden.json`` path: a draft that could be
    mistaken for a verified label is one careless glob away from entering a
    corpus unreviewed.
    """
    key = f"{paths.golden_label_dir(draft.doc_type, draft.source_id, tenant_id)}/draft.json"
    client.write_json(key, draft.as_dict())
    return key


def pre_annotate_batch(
    source_ids: Iterable[str],
    doc_type: str,
    client: BlobClient,
    **kwargs: Any,
) -> tuple[list[Draft], list[tuple[str, str]]]:
    """Draft many documents. One unreadable document never stops the batch —
    but a refused backend stops all of them, because it applies to every one."""
    check_backend(kwargs.get("backend", DEFAULT_BACKEND))
    drafts: list[Draft] = []
    failed: list[tuple[str, str]] = []
    for source_id in source_ids:
        try:
            drafts.append(pre_annotate(source_id, doc_type, client, **kwargs))
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a batch
            failed.append((source_id, str(exc)))
            log.warning("draft failed for %s: %s", source_id, exc)
    return drafts, failed


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Pre-annotate documents for human review")
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--source-ids", nargs="+", required=True)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(BACKENDS))
    parser.add_argument("--tenant", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('data_pipeline.labeling.pre_annotate', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()

    drafts, failed = pre_annotate_batch(
        args.source_ids, args.doc_type, client,
        backend=args.backend, tenant_id=args.tenant,
    )
    for draft in drafts:
        write_draft(draft, client, args.tenant)
    print(json.dumps({"drafted": len(drafts), "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
