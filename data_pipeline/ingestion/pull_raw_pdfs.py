"""Ingest raw PDFs into the immutable ``raw-documents/`` layer (arch §18a).

The front of the whole pipeline, and the only component that writes this layer.
Three properties matter more than throughput:

**Immutable.** ``original.pdf`` is write-once. A corrected document is ingested
as a *new* ``source_id`` rather than replacing the old one, so every historical
training run stays reproducible against the exact bytes it trained on. The guard
lives in :mod:`artifact_registry.blob_client`; this module never tries to bypass it.

**Deduplicated by content.** Loss Runs and renewal policies are re-submitted
constantly with minor changes. Checksumming on arrival prevents paying to label
the same document twice and prevents corpus bloat from near-identical copies.

**Digital vs scanned recorded at ingest.** Detected here because it is cheap
here and expensive later, and because it defines the scanned eval subset that
feeds the ViT escalation gate (arch §3, §15).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, ImmutableBlobError, for_ingestion
from common.constants import ACTIVE_DOC_TYPES, UNCLASSIFIED
from common.ids import build_source_id, is_valid_source_id, parse_source_id

log = logging.getLogger(__name__)

#: Bucket for documents whose type is not yet known. The classifier resolves
#: them later; forcing a guess at ingest would bake a mistake into the source_id,
#: which is the one identifier everything downstream joins on.
#: Re-exported for the CLI and callers; defined in common.constants so
#: common/ids.py and artifact_registry/paths.py can honour it too — they
#: sit below data_pipeline and cannot import from it.
UNCLASSIFIED = UNCLASSIFIED

CHUNK = 1 << 20


class IngestionError(RuntimeError):
    """Raised on an unreadable PDF or an unresolvable document type."""


@dataclass
class IngestResult:
    """What one ingestion run did — reported, never inferred from logs."""

    ingested: list[str] = field(default_factory=list)
    skipped_duplicates: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.ingested)} ingested"]
        if self.skipped_duplicates:
            parts.append(f"{len(self.skipped_duplicates)} duplicates skipped")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pdf(path: Path) -> dict[str, Any]:
    """Page count and whether the PDF carries an embedded text layer.

    The digital/scanned distinction drives the scanned eval subset (arch §15) and
    is one of the two inputs to the ViT escalation gate. A PDF with no extractable
    text is scanned; a mostly-empty text layer means a scan with a thin OCR layer
    already applied, which is treated as scanned because the model still has to
    read the pixels.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional extra
        raise IngestionError(
            'pypdf is not installed. Install the [data] extra: pip install -e ".[data]"'
        ) from exc

    try:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        sampled = min(3, pages)
        text = "".join((p.extract_text() or "") for p in reader.pages[:sampled])
    except Exception as exc:
        # pypdf gives up on content PyMuPDF reads: an inline image whose
        # run-length data ends early stopped a whole import (one SPEC_21 seed, a
        # 9-page homeowners endorsement). PyMuPDF is the renderer the pipeline
        # uses for every page anyway, so its reading counts; only a PDF neither
        # can open is refused.
        try:
            import pymupdf

            with pymupdf.open(str(path)) as document:
                pages = document.page_count
                sampled = min(3, pages)
                text = "".join(document[i].get_text() for i in range(sampled))
        except Exception as fallback:
            raise IngestionError(f"could not read {path.name}: {exc}; PyMuPDF: {fallback}") from exc
        log.warning("%s: pypdf could not read it (%s); read with PyMuPDF instead", path.name, exc)

    chars_per_page = len(text.strip()) / max(1, sampled)
    return {
        "page_count": pages,
        "is_scanned": chars_per_page < 50,
        "text_chars_per_sampled_page": round(chars_per_page, 1),
    }


def _existing_checksums(client: BlobClient, doc_type: str, tenant_id: str | None) -> dict[str, str]:
    """Map checksum -> source_id for what is already ingested for this type."""
    prefix = f"raw-documents/{paths._tenant(tenant_id)}/{doc_type}/"
    seen: dict[str, str] = {}
    for key in client.list(prefix):
        if not key.endswith("metadata.json"):
            continue
        try:
            meta = client.read_json(key)
        except Exception:
            continue
        if checksum := meta.get("checksum_sha256"):
            seen[checksum] = meta.get("source_id", key)
    return seen


#: Every place a document's id is the key of something stored.
_ID_ROOTS = ("raw-documents", "processed", "golden-labels")


def taken_source_ids(client: BlobClient, doc_type: str, tenant_id: str | None) -> list[str]:
    """Every source_id already in use for this type, wherever it is in use.

    PDF ingestion counted the raw layer and the prepared import counted the
    labels, so each handed out ids the other had already given: ``policy_0001``
    twice, the second overwriting the first's pages and label. A root this
    client may not read (the raw layer is restricted to ingestion and OCR) is
    skipped - the other roots still hold every id that reached them.
    """
    out = set()
    for root in _ID_ROOTS:
        prefix = f"{root}/{paths._tenant(tenant_id)}/{doc_type}/"
        try:
            keys = list(client.list(prefix))
        except Exception:  # noqa: BLE001 - a root outside this client's reach
            continue
        for key in keys:
            parts = key.split("/")
            if len(parts) >= 4 and is_valid_source_id(parts[3]):
                out.add(parts[3])
    return sorted(out)


def _existing_source_ids(client: BlobClient, doc_type: str, tenant_id: str | None) -> list[str]:
    return taken_source_ids(client, doc_type, tenant_id)


def ingest_pdf(
    pdf_path: Path,
    doc_type: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    source_id: str | None = None,
    known_checksums: dict[str, str] | None = None,
    source_system: str = "manual",
) -> tuple[str | None, str]:
    """Ingest one PDF. Returns ``(source_id, status)``.

    ``status`` is ``"ingested"``, ``"duplicate"``, or ``"failed:<reason>"`` — the
    caller reports what happened rather than inferring it from log output.
    """
    checksum = sha256_file(pdf_path)

    if known_checksums is not None and checksum in known_checksums:
        original = known_checksums[checksum]
        log.info("skipping %s: identical content already ingested as %s", pdf_path.name, original)
        return original, "duplicate"

    if source_id is None:
        existing = _existing_source_ids(client, doc_type, tenant_id)
        indices = [parse_source_id(s).index for s in existing]
        source_id = build_source_id(doc_type, (max(indices) + 1) if indices else 1)

    inspected = inspect_pdf(pdf_path)
    metadata = {
        "source_id": source_id,
        "doc_type": doc_type,
        "tenant_id": paths._tenant(tenant_id),
        "ingested_at": datetime.now(UTC).isoformat(),
        "source_system": source_system,
        "original_filename": pdf_path.name,
        "checksum_sha256": checksum,
        "size_bytes": pdf_path.stat().st_size,
        **inspected,
        # Populated by the de-identification step if and when it is unblocked
        # (IMPL-05 §1). Recorded as unknown rather than omitted, so nothing
        # downstream can mistake absence for "no PII".
        "pii_flags": {"status": "not_assessed"},
        "retention_class": "standard",
    }

    try:
        client.upload_file(pdf_path, paths.raw_pdf(doc_type, source_id, tenant_id))
    except ImmutableBlobError:
        return source_id, "failed:already_exists_write_once"

    client.write_json(paths.raw_metadata(doc_type, source_id, tenant_id), metadata)
    if known_checksums is not None:
        known_checksums[checksum] = source_id
    log.info("ingested %s as %s (%d pages, %s)", pdf_path.name, source_id,
             inspected["page_count"], "scanned" if inspected["is_scanned"] else "digital")
    return source_id, "ingested"


def ingest_directory(
    input_dir: Path,
    doc_type: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    source_system: str = "manual",
) -> IngestResult:
    """Ingest every PDF in a directory. Idempotent and safe to re-run."""
    if doc_type != UNCLASSIFIED and doc_type not in ACTIVE_DOC_TYPES:
        raise IngestionError(
            f"unknown doc_type {doc_type!r}; expected one of {list(ACTIVE_DOC_TYPES)} "
            f"or {UNCLASSIFIED!r} when the type is not yet known"
        )

    result = IngestResult()
    known = _existing_checksums(client, doc_type, tenant_id)

    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        try:
            source_id, status = ingest_pdf(
                pdf_path, doc_type, client, tenant_id=tenant_id,
                known_checksums=known, source_system=source_system,
            )
        except (IngestionError, Exception) as exc:  # noqa: BLE001 - one bad PDF must not stop a batch
            result.failed.append((pdf_path.name, str(exc)))
            log.warning("failed ingesting %s: %s", pdf_path.name, exc)
            continue

        if status == "ingested":
            result.ingested.append(source_id or pdf_path.name)
        elif status == "duplicate":
            result.skipped_duplicates.append((pdf_path.name, source_id or ""))
        else:
            result.failed.append((pdf_path.name, status))

    log.info("ingestion complete: %s", result.summary())
    return result


def purge_plan(
    source_id: str, doc_type: str, tenant_id: str | None = None, *,
    client: BlobClient | None = None,
) -> dict[str, list[Any]]:
    """The full downstream cascade for a purge request (arch §18a).

    Returns the plan; it never deletes. Retaining derived data from a document
    that no longer legally exists is a real audit gap, so a purge has to reach
    ``processed/``, ``golden-labels/``, the frozen eval set and any corpus
    entries built from it — but corpus files are shared across documents, so
    removing a row from one is a rebuild, not a delete. That is why this reports
    rather than acts.

    A model trained on the document is the tenant's data too (arch §8b). With a
    ``client``, every run whose corpus listed the document is named, with its
    weights and the releases serving it: each needs a retrain or a delete. The
    weights' prefixes are not tenant-scoped yet, so the plan names them.
    """
    plan: dict[str, list[Any]] = {
        "raw": [paths.raw_document_dir(doc_type, source_id, tenant_id)],
        "processed": [paths.processed_dir(doc_type, source_id, tenant_id)],
        "golden_labels": [paths.golden_label_dir(doc_type, source_id, tenant_id)],
        "golden_eval_set": [f"{paths.golden_eval_set_dir(tenant_id)}/{source_id}"],
        # Result sets hold many documents' outputs; each one must be checked.
        "exports": [f"{paths.EXPORTS_PREFIX}/{paths._tenant(tenant_id)}"],
        "corpus_action": [
            f"rebuild any corpus version whose manifest lists source_id {source_id!r} — "
            "corpus files are shared, so the row cannot simply be deleted"
        ],
        "golden_eval_set_action": [
            "if the frozen eval set holds the document: remove it and update the set's "
            "manifest - the yardstick changes, so later scores are not comparable with earlier ones"
        ],
    }
    if client is None:
        plan["trained_weights"] = [
            "pass a client to list the runs trained on this document: each one's weights and "
            "releases need a retrain or a delete"
        ]
    else:
        plan["trained_weights"] = runs_trained_on(client, tenant_id=tenant_id, source_id=source_id)
    return plan


def tenant_purge_plan(tenant_id: str, client: BlobClient | None = None) -> dict[str, list[Any]]:
    """Everything a tenant deletion has to reach. Reports; never deletes.

    Built from :data:`artifact_registry.paths.TENANT_SCOPED`, so a prefix added
    there cannot be forgotten here, plus every run trained on the tenant's
    corpora - whose weights live under prefixes that are not tenant-scoped yet.
    """
    tenant = paths._tenant(tenant_id)
    plan: dict[str, list[Any]] = {
        "prefixes": sorted(f"{prefix}/{tenant}" for prefix in paths.TENANT_SCOPED),
    }
    if client is None:
        plan["trained_weights"] = [
            "pass a client to list the runs trained on this tenant: each one's weights and "
            "releases need a retrain or a delete"
        ]
    else:
        plan["trained_weights"] = runs_trained_on(client, tenant_id=tenant)
    return plan


def runs_trained_on(
    client: BlobClient, *, tenant_id: str | None = None, source_id: str | None = None,
) -> list[dict[str, Any]]:
    """The runs whose corpus held the tenant's data (or one document of it).

    Read from the run registry: a run of this tenant whose corpus manifest lists
    ``source_id`` in any split (any of the tenant's runs when ``source_id`` is
    None), with its weights and the tenant's releases built on it. A run that
    records no tenant, or whose corpus manifest is gone, is listed to check by
    hand rather than skipped: absent evidence is not a clean model.
    """
    from registry_utils.query_registry import RegistryQueryError, get, list_runs

    tenant = paths._tenant(tenant_id)
    releases: dict[str, list[str]] = {}
    index_key = paths.release_index(tenant)
    if client.exists(index_key):
        for row in client.read_json(index_key) or []:
            releases.setdefault(str(row.get("adapter")), []).append(str(row.get("release_id")))

    found: list[dict[str, Any]] = []
    for row in list_runs(client, include_sweeps=True):
        run_id = str(row.get("run_id"))
        try:
            manifest = get(run_id, client)
        except (RegistryQueryError, KeyError, FileNotFoundError):
            found.append({"run_id": run_id, "action": "check by hand: the manifest cannot be read"})
            continue
        if manifest.tenant_id is None:
            found.append({"run_id": run_id, "action": "check by hand: the run records no tenant"})
            continue
        if paths._tenant(manifest.tenant_id) != tenant:
            continue
        corpus = str(manifest.dependencies.corpus_version or "").removeprefix("corpus/")
        action = "needs retrain or delete"
        if source_id is not None:
            try:
                key = paths.corpus_manifest(corpus, tenant)
            except paths.PathError:
                key = None
            listed = client.read_json(key).get("source_ids_by_split") if key and client.exists(key) else None
            if listed is None:
                action = "check by hand: the run's corpus manifest is missing"
            elif not any(source_id in ids for ids in listed.values()):
                continue
        found.append({
            "run_id": run_id,
            "corpus_version": manifest.dependencies.corpus_version,
            "adapter_weights": manifest.artifacts.adapter_weights,
            "merged_model": manifest.artifacts.merged_model,
            "quantized_model": manifest.artifacts.quantized_model,
            "releases": sorted(releases.get(run_id, [])),
            "action": action,
        })
    return found


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Ingest raw PDFs into raw-documents/")
    parser.add_argument("--input", type=Path, help="directory of PDFs (required to ingest)")
    parser.add_argument("--doc-type", choices=[*ACTIVE_DOC_TYPES, UNCLASSIFIED],
                        help="required to ingest and for --purge-plan")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--source-system", default="manual")
    parser.add_argument("--purge-plan", metavar="SOURCE_ID",
                        help="print the downstream cascade for a purge request and exit")
    parser.add_argument("--tenant-purge-plan", dest="tenant_purge_plan", action="store_true",
                        help="print everything a deletion of --tenant must reach, and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.tenant_purge_plan and not args.tenant:
        parser.error("--tenant-purge-plan needs --tenant")
    if not args.tenant_purge_plan and not args.doc_type:
        parser.error("--doc-type is required")
    if not (args.purge_plan or args.tenant_purge_plan or args.input):
        parser.error("--input is required to ingest")
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('data_pipeline.ingestion.pull_raw_pdfs', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.tenant_purge_plan:
        print(json.dumps(tenant_purge_plan(args.tenant, BlobClient()), indent=2))
        return 0
    if args.purge_plan:
        print(json.dumps(
            purge_plan(args.purge_plan, args.doc_type, args.tenant, client=BlobClient()), indent=2))
        return 0

    result = ingest_directory(
        args.input, args.doc_type, for_ingestion(),
        tenant_id=args.tenant, source_system=args.source_system,
    )
    print(result.summary())
    for name, reason in result.failed:
        print(f"  FAILED {name}: {reason}")
    return 1 if result.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
