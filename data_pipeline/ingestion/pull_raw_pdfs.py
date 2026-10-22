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
from artifact_registry.blob_client import BlobClient, BlobError, ImmutableBlobError, for_ingestion
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


def _prefix(path: str) -> str:
    """A blob prefix that ends at a folder boundary.

    The path helpers return keys with no trailing ``/``, and as a key prefix
    ``golden-eval-set/acme/policy_1`` also matches ``policy_10/`` and
    ``corpus/acme`` matches ``corpus/acme_west/``: a purge run from the plan
    would delete another document's, or another tenant's, data.
    """
    return path.rstrip("/") + "/"


def _pod_volume(tenant: str, doc_type: str | None = None, source_id: str | None = None,
                folder: str | None = None) -> tuple[list[str], list[str]]:
    """Where the pod's volume holds copies of the tenant's documents: ``(paths, actions)``.

    The volume survives a pod stop, so a copy there outlives the purge of the
    Blob layers it came from. The paths are globs (``*``): the copies are keyed
    by batch, corpus version, scope and run, and every one of them counts.
    """
    from data_pipeline.ingestion.pull_intake import default_intake_dir
    from data_pipeline.ocr_check import DEFAULT_OUT as OCR_CHECK_OUT

    images = _prefix(paths.staging_train_images_dir("*", tenant))
    intake = default_intake_dir().as_posix()
    if source_id is not None and doc_type is not None:
        # The image cache stores each page under its own blob key, so one
        # document's pages are one folder in each corpus version's cache.
        images = _prefix(f"{images}{paths.processed_dir(doc_type, source_id, tenant)}")
        if folder:
            intake = f"{intake}/*/{folder}"
    found = [
        _prefix(intake),
        images,
        _prefix(paths.staging_train_data_dir("*", "*")),
        _prefix(f"{OCR_CHECK_OUT.as_posix()}*"),
    ]
    actions = [
        f"{found[0]}: delivered bundles (the PDF and its golden.json) pulled by pull_intake, "
        "and the smoke run's subsets copied from them - pull_intake never deletes a batch"
        + ("" if folder or source_id is None else
           "; the folder this document was imported from is not recorded, so check each batch by hand"),
        f"{found[1]}: page images cached per corpus version for training and evaluation",
        f"{found[2]}: each run's staged training rows (labels included), one folder per scope "
        "and version - delete the folders of the runs under trained_weights",
        f"{found[3]}: ocr_check reports, whose CSVs quote label values - delete them or run the "
        "check again after the purge (and any --out folder it was given)",
        "/workspace/logs/: diagnose_generation reports are named after the document they compare "
        "and hold its values (a compare folder holds its PDF too) - check by hand",
    ]
    return found, actions


def _intake_copies(client: BlobClient, folder: str) -> list[str] | None:
    """``intake/{batch}/{folder}/`` for every delivered batch holding that folder.

    ``None`` when this client cannot list ``intake/``: it is in the raw
    container, which a client outside ingestion and OCR may not read
    (AccessDeniedError) and a token without raw access cannot (a 403).
    """
    try:
        keys = client.list(_prefix(paths.INTAKE_PREFIX))
    except BlobError:  # AccessDeniedError is one
        return None
    found = set()
    for key in keys:
        parts = key.split("/")
        if len(parts) >= 4 and parts[0] == paths.INTAKE_PREFIX and parts[2] == folder:
            found.add(_prefix("/".join(parts[:3])))
    return sorted(found)


def _imported_from(client: BlobClient, doc_type: str, source_id: str, tenant: str) -> str | None:
    """The delivered folder a labeled import read the document from, if recorded.

    ``import_labeled_pdfs`` records the folder's name (``imported_from``) but
    not the batch it was delivered in.
    """
    key = paths.label_metadata(doc_type, source_id, tenant)
    try:
        meta = client.read_json(key) if client.exists(key) else None
    except (BlobError, ValueError):  # unreadable: the plan says check by hand
        return None
    folder = meta.get("imported_from") if isinstance(meta, dict) else None
    return str(folder) if folder else None


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

    The delivered copy in ``intake/`` (raw container) and the copies on the
    pod's volume are named too: nothing deletes them once they are imported.

    A model trained on the document is the tenant's data too (arch §8b). With a
    ``client``, every run whose corpus listed the document is named, with its
    weights and the releases serving it: each needs a retrain or a delete. The
    weights' prefixes are not tenant-scoped yet, so the plan names them.
    """
    tenant = paths._tenant(tenant_id)
    folder = _imported_from(client, doc_type, source_id, tenant) if client is not None else None
    on_volume, volume_actions = _pod_volume(tenant, doc_type, source_id, folder)
    plan: dict[str, list[Any]] = {
        "raw": [_prefix(paths.raw_document_dir(doc_type, source_id, tenant))],
        "processed": [_prefix(paths.processed_dir(doc_type, source_id, tenant))],
        "golden_labels": [_prefix(paths.golden_label_dir(doc_type, source_id, tenant))],
        "golden_eval_set": [_prefix(f"{paths.golden_eval_set_dir(tenant)}/{source_id}")],
        # Result sets hold many documents' outputs; each one must be checked.
        "exports": [_prefix(f"{paths.EXPORTS_PREFIX}/{tenant}")],
        "pod_volume": on_volume,
        "pod_volume_action": volume_actions,
        "corpus_action": [
            f"rebuild any corpus version whose manifest lists source_id {source_id!r} — "
            "corpus files are shared, so the row cannot simply be deleted"
        ],
        "golden_eval_set_action": [
            "if the frozen eval set holds the document: remove it and update the set's "
            "manifest - the yardstick changes, so later scores are not comparable with earlier ones. "
            "Every run gated on the set holds the document's expected values in its eval report's "
            "error_records (eval_report under trained_weights, and the tenant's other runs): "
            "remove them there too"
        ],
    }
    # intake/{batch}/{document}/: the PDF and golden.json exactly as delivered.
    copies = _intake_copies(client, folder) if folder else None
    if copies:
        plan["intake"] = copies
        plan["intake_action"] = [
            f"the import records the folder the document came from ({folder!r}), not its batch: "
            "these batches hold a folder of that name - confirm each is this document (its PDF's "
            "checksum is checksum_sha256 in the raw metadata.json) before deleting it"
        ]
    elif copies is not None:
        plan["intake"] = []
        plan["intake_action"] = [f"no batch under intake/ holds a folder named {folder!r}"]
    else:
        plan["intake"] = []
        plan["intake_action"] = [
            f"check every batch under intake/ in the raw container by hand for a folder named {folder!r} "
            "(the import records the folder, not the batch; this client cannot list intake/)"
            if folder else
            "check every batch under intake/ in the raw container by hand: nothing records which "
            "batch delivered this document" + ("" if client is not None else
                                                " (pass a client to look up the folder it was imported from)")
        ]
    if client is None:
        plan["trained_weights"] = [
            "pass a client to list the runs trained on this document: each one's weights and "
            "releases need a retrain or a delete"
        ]
    else:
        plan["trained_weights"] = runs_trained_on(client, tenant_id=tenant, source_id=source_id)
    return plan


def tenant_purge_plan(tenant_id: str, client: BlobClient | None = None) -> dict[str, list[Any]]:
    """Everything a tenant deletion has to reach. Reports; never deletes.

    Built from :data:`artifact_registry.paths.TENANT_SCOPED`, so a prefix added
    there cannot be forgotten here, plus every run trained on the tenant's
    corpora - whose weights live under prefixes that are not tenant-scoped yet.
    """
    tenant = paths._tenant(tenant_id)
    on_volume, volume_actions = _pod_volume(tenant)
    plan: dict[str, list[Any]] = {
        "prefixes": sorted(_prefix(f"{prefix}/{tenant}") for prefix in paths.TENANT_SCOPED),
        # intake/{batch}/ carries no tenant segment, so no prefix above reaches it.
        "intake_action": [
            "check every batch under intake/ in the raw container by hand: delivered batches "
            "are not filed by tenant, and nothing deletes one after import"
        ],
        "pod_volume": on_volume,
        "pod_volume_action": volume_actions,
    }
    if client is None:
        plan["trained_weights"] = [
            "pass a client to list the runs trained on this tenant: each one's weights and "
            "releases need a retrain or a delete"
        ]
    else:
        plan["trained_weights"] = runs_trained_on(client, tenant_id=tenant)
    return plan


def _corpus_lists(client: BlobClient, manifest: Any, tenant: str, source_id: str) -> bool | str:
    """Whether the run's corpus manifest lists ``source_id`` in any split.

    A string says why it cannot tell; the run is then listed to check by hand.
    """
    corpus = str(manifest.dependencies.corpus_version or "").removeprefix("corpus/")
    try:
        key = paths.corpus_manifest(corpus, tenant)
    except paths.PathError:
        return "the run's corpus manifest is missing"
    if not client.exists(key):
        return "the run's corpus manifest is missing"
    try:
        listed = client.read_json(key).get("source_ids_by_split")
    except (BlobError, ValueError, AttributeError) as exc:
        return f"the run's corpus manifest cannot be read ({type(exc).__name__})"
    if not isinstance(listed, dict):
        return "the run's corpus manifest is missing"
    return any(source_id in (ids or []) for ids in listed.values())


def _run_entry(manifest: Any, releases: dict[str, list[str]], action: str) -> dict[str, Any]:
    artifacts = manifest.artifacts
    return {
        "run_id": manifest.run_id,
        "corpus_version": manifest.dependencies.corpus_version,
        "adapter_weights": artifacts.adapter_weights,
        "merged_model": artifacts.merged_model,
        "quantized_model": artifacts.quantized_model,
        # The weights on the pod's volume while they are staged.
        "staging_path": artifacts.staging_path,
        # Both under shared prefixes (eval-reports/, calibration/), so no tenant
        # prefix reaches them: the report's error_records hold documents'
        # expected values, and the calibrator was fitted on the tenant's fields.
        "eval_report": artifacts.eval_report,
        "calibration_params": artifacts.calibration_params,
        "releases": sorted(releases.get(manifest.run_id, [])),
        "action": action,
    }


def runs_trained_on(
    client: BlobClient, *, tenant_id: str | None = None, source_id: str | None = None,
) -> list[dict[str, Any]]:
    """The runs whose corpus held the tenant's data (or one document of it).

    Read from the run registry: a run of this tenant whose corpus manifest lists
    ``source_id`` in any split (any of the tenant's runs when ``source_id`` is
    None), with its weights and the tenant's releases built on it. A run that
    records no tenant, or whose manifest or corpus manifest cannot be read, is
    listed to check by hand rather than skipped: absent evidence is not a clean
    model.

    Lineage counts as well as the corpus. A run that continued from a checkpoint
    carries what earlier corpora taught it though its own corpus may not list
    the document, and a per-type adapter is built on its Foundation's weights.
    """
    from registry_utils.query_registry import RegistryQueryError, get, list_runs

    tenant = paths._tenant(tenant_id)
    releases: dict[str, list[str]] = {}
    index_key = paths.release_index(tenant)
    if client.exists(index_key):
        for row in client.read_json(index_key) or []:
            releases.setdefault(str(row.get("adapter")), []).append(str(row.get("release_id")))

    found: list[dict[str, Any]] = []
    adapters: list[Any] = []
    for row in list_runs(client, include_sweeps=True):
        run_id = str(row.get("run_id"))
        try:
            manifest = get(run_id, client)
        # ValueError covers a manifest that fails validation (pydantic's
        # ValidationError: manifests forbid unknown fields), an index row that
        # builds no path (paths.PathError) and a manifest that is not JSON. One
        # bad manifest is that run's problem, not the whole plan's. The type is
        # named, never the message: a storage error can quote a URL.
        except (RegistryQueryError, BlobError, KeyError, FileNotFoundError, ValueError) as exc:
            found.append({"run_id": run_id,
                          "action": f"check by hand: the manifest cannot be read ({type(exc).__name__})"})
            continue
        if manifest.tenant_id is None:
            found.append({"run_id": run_id, "action": "check by hand: the run records no tenant"})
            continue
        if paths._tenant(manifest.tenant_id) != tenant:
            continue
        action = "needs retrain or delete"
        if source_id is not None:
            listed = _corpus_lists(client, manifest, tenant, source_id)
            if isinstance(listed, str):
                action = f"check by hand: {listed}"
            elif not listed:
                if manifest.continued_from:
                    # A checkpoint PATH (training.train --continue-from), not a
                    # run id, so the corpora behind it cannot be read from here.
                    action = (f"check by hand: continued from {manifest.continued_from}, "
                              "which may carry earlier corpora")
                elif manifest.run_type == "per_type_adapter":
                    adapters.append(manifest)
                    continue
                else:
                    continue
        found.append(_run_entry(manifest, releases, action))

    # A per-type adapter trained on a listed Foundation sits on weights that
    # learned the document, whatever its own corpus lists.
    listed_runs = {entry["run_id"]: entry["action"] for entry in found}
    for manifest in adapters:
        foundation = manifest.dependencies.foundation_version
        if foundation in listed_runs:
            verdict = ("needs retrain or delete" if listed_runs[foundation] == "needs retrain or delete"
                       else "check by hand with it")
            found.append(_run_entry(manifest, releases, f"built on {foundation}, listed here: {verdict}"))
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
        # An ingestion client, so the plan can look for the document's delivered
        # folder in intake/, in the raw container. It still only reads.
        print(json.dumps(
            purge_plan(args.purge_plan, args.doc_type, args.tenant, client=for_ingestion()), indent=2))
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
