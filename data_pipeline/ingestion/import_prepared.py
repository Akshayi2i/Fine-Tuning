"""Import documents that arrive already prepared (IMPL-03, IMPL-04, arch §8a).

The normal intake is a PDF: ingestion stores it, OCR renders pages and produces
MinerU markdown, and labeling writes the golden JSON. This importer is for the
case where all three already exist — page images, per-page markdown and a golden
JSON written against the canonical schema — and only need to be placed where the
pipeline reads them.

**It writes into the existing layout rather than a parallel one.** After an
import, ``_is_preprocessed`` reports those documents complete, ``stage_labeling``
counts them and the dataset build picks them up with no other change. A second
layout would mean a second set of readers, and one of them would eventually be
missed.

Two guards, both load-bearing:

* **The MinerU version is required.** The model learns how MinerU formats its
  output, so a version bump is distribution shift (arch §8a). The serving
  endpoint refuses to start when its OCR version does not match the corpus pin —
  and an import that left the pin blank would make that check unverifiable
  rather than satisfied.
* **The golden JSON is validated before it is written.** A malformed label that
  reaches ``golden-labels/`` is a training target nobody checked, and the first
  symptom is a model that learned the wrong shape.

Nothing is written to ``raw-documents/``: there is no PDF, so the importer never
touches the unredacted-PII layer at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES
from common.ids import next_source_id
from common.schemas import SchemaError, iter_validation_errors
from data_pipeline.ingestion.import_labeled_pdfs import twin_index

log = logging.getLogger(__name__)

_PAGE_RE = re.compile(r"page[_-]?(\d+)", re.I)

#: Image formats an import may supply. Everything is stored as PNG whatever
#: arrives: `processed/.../page_N.png` is what every reader asks for by name
#: (`input_builder.page_images_for`, the dataset build, pre-annotation), so a
#: stored .jpg would leave a document that looks complete and whose pages
#: nothing can find.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
STORED_SUFFIX = "png"


class ImportError_(RuntimeError):
    """Raised when a prepared bundle cannot be imported safely."""


@dataclass
class PreparedDocument:
    """One directory of prepared inputs."""

    directory: Path
    images: list[Path] = field(default_factory=list)
    markdown: list[Path] = field(default_factory=list)
    golden: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.images)


@dataclass
class ImportReport:
    """What an import run did, per document."""

    imported: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"imported {len(self.imported)} document(s)"]
        lines += [f"  skipped {name}: {why}" for name, why in self.skipped]
        return "\n".join(lines)


def _page_number(path: Path) -> int:
    match = _PAGE_RE.search(path.stem)
    if not match:
        raise ImportError_(
            f"{path.name} does not name its page. Files must be page_1.png / page_1.md: "
            "page order is what pairs an image with the text on it, and a wrong pairing "
            "teaches the model to read one page while looking at another."
        )
    return int(match.group(1))


def read_prepared(directory: Path) -> PreparedDocument:
    """Load one prepared directory, in page order."""
    doc = PreparedDocument(directory=directory)
    # One list sorted by PAGE NUMBER across every image extension. Sorting each
    # extension separately and concatenating put page_1.jpg after page_9.png, so
    # a mixed bundle paired images with another page's markdown — the exact
    # failure `_page_number` refuses an unnumbered file to prevent.
    doc.images = sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=_page_number,
    )
    doc.markdown = sorted((p for p in directory.glob("*.md")), key=_page_number)

    golden_path = directory / "golden.json"
    if not golden_path.exists():
        raise ImportError_(f"{directory.name} has no golden.json")
    doc.golden = json.loads(golden_path.read_text(encoding="utf-8"))

    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        doc.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not doc.images:
        raise ImportError_(f"{directory.name} has no page images")
    if doc.markdown and len(doc.markdown) != len(doc.images):
        raise ImportError_(
            f"{directory.name} has {len(doc.images)} image(s) and {len(doc.markdown)} markdown "
            "file(s). They are paired per page, so a mismatch means some page's text belongs to "
            "a different page's image."
        )
    return doc


def _as_png(image: Path) -> bytes:
    """The page as PNG bytes, converting only when it is not one already.

    Converted here rather than stored as-is because the whole pipeline addresses
    pages as ``page_N.png``. Lossless, and it happens once at import rather than
    on every read.
    """
    if image.suffix.lower() == ".png":
        return image.read_bytes()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a declared dependency
        raise ImportError_(
            f"{image.name} is not a PNG and Pillow is not installed to convert it. Pages are "
            "stored as page_N.png because that is what every reader asks for by name."
        ) from exc

    import io

    with Image.open(image) as opened:
        buffer = io.BytesIO()
        opened.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _checksum(doc: PreparedDocument) -> str:
    """SHA-256 over the page files, in order — there is no PDF to hash.

    Deduplication and the OCR-skip check both key on this, so it has to be
    stable: same pages in the same order, same digest.
    """
    digest = hashlib.sha256()
    for path in doc.images + doc.markdown:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def import_document(
    doc: PreparedDocument,
    doc_type: str,
    client: BlobClient,
    *,
    mineru_version: str,
    tenant_id: str | None = None,
    source_id: str | None = None,
    ocr_device: str = "unknown",
) -> str:
    """Place one prepared document where the pipeline reads it. Returns its id."""
    if doc_type not in ACTIVE_DOC_TYPES:
        raise ImportError_(f"unknown doc_type {doc_type!r}; active: {list(ACTIVE_DOC_TYPES)}")
    if not mineru_version:
        raise ImportError_(
            "an import must record the MinerU version that produced its markdown. The model "
            "learns how MinerU formats its output (arch §8a), and serving refuses to start when "
            "its OCR version does not match the corpus pin — a blank pin makes that check "
            "unverifiable rather than satisfied."
        )

    acord_form = doc.metadata.get("acord_form")
    from common.lob import merge_line

    # Classic auto is read as personal auto (common.lob.MERGED_LINES).
    lob = merge_line(doc.metadata.get("lob") or doc.golden.get("line_of_business"))
    try:
        errors = list(iter_validation_errors(doc.golden, doc_type, acord_form, lob))
    except SchemaError as exc:
        raise ImportError_(f"{doc.directory.name}: no schema to validate against — {exc}") from exc
    if errors:
        from common.schemas import old_shape_hint

        hint = old_shape_hint(doc.golden, doc_type, acord_form, lob)
        raise ImportError_(
            f"{doc.directory.name}: golden.json does not satisfy the {doc_type} schema — "
            f"{hint or errors[0]}. A label that reaches golden-labels/ is a training target nobody "
            "checked, and the first symptom is a model that learned the wrong shape."
        )
    from data_pipeline.labeling.export_golden_labels import review_problem

    problem = review_problem(doc.golden, doc.metadata)
    if problem:
        raise ImportError_(f"{doc.directory.name}: {problem}")

    # The next id unused ANYWHERE - raw PDFs, processed pages, labels - so an
    # imported document cannot take the id of one that was ingested as a PDF
    # (or the reverse) and overwrite its pages and label.
    from data_pipeline.ingestion.pull_raw_pdfs import taken_source_ids

    existing = taken_source_ids(client, doc_type, tenant_id)
    source_id = source_id or next_source_id(existing, doc_type)

    for page, image in enumerate(doc.images, start=1):
        client.write_bytes(
            paths.processed_page(doc_type, source_id, page, STORED_SUFFIX, tenant_id),
            _as_png(image),
        )
    for page, markdown in enumerate(doc.markdown, start=1):
        client.write_text(
            paths.processed_page(doc_type, source_id, page, "md", tenant_id),
            markdown.read_text(encoding="utf-8"),
        )

    client.write_json(paths.ocr_meta(doc_type, source_id, tenant_id), {
        "source_id": source_id,
        "doc_type": doc_type,
        "tenant_id": paths._tenant(tenant_id),
        "page_count": doc.page_count,
        "mineru_version": mineru_version,
        "ocr_device": ocr_device,
        "preprocessing_date": datetime.now(UTC).isoformat(),
        "resolution_cap_px": doc.metadata.get("resolution_cap_px"),
        "source_checksum": _checksum(doc),
        # Imported markdown carries no per-page table counts, so the IMPL-09
        # row-completeness cross-check has nothing to compare against for these
        # documents. Empty rather than zero: zero would read as "no rows found".
        "table_row_counts": {},
        "failed_pages": [],
        "imported": True,
        # An import that supplies no markdown is image-only by construction.
        "render_only": not doc.markdown,
    })

    client.write_json(paths.golden_label(doc_type, source_id, tenant_id), doc.golden)
    client.write_json(paths.label_metadata(doc_type, source_id, tenant_id), {
        "source_id": source_id,
        "doc_type": doc_type,
        "acord_form": acord_form,
        "lob": lob,
        # Which page each golden value came from. Without it the policy
        # decomposition has to apportion rows across windows instead of placing
        # them (§7b).
        "field_provenance": doc.metadata.get("field_provenance", {}),
        "document_kind": doc.metadata.get("document_kind"),
        "template_id": doc.metadata.get("template_id"),
        "carrier": doc.metadata.get("carrier"),
        "twin_index": twin_index(doc.metadata),
        "render_mode": doc.metadata.get("render_mode"),
        "synthetic": bool(doc.metadata.get("synthetic", False)),
        "reviewer_id": doc.metadata.get("reviewer_id"),
        "review_date": doc.metadata.get("review_date"),
        "draft_backend": doc.metadata.get("draft_backend"),
        "labeled_at": datetime.now(UTC).isoformat(),
        "imported_from": doc.directory.name,
    })

    if not doc.metadata.get("field_provenance"):
        log.warning(
            "%s has no field_provenance, so per-alias evaluation cannot score it until alias "
            "derivation backfills its surface labels.",
            source_id,
        )
    log.info("imported %s as %s (%d pages)", doc.directory.name, source_id, doc.page_count)
    return source_id


def import_batch(
    input_dir: Path,
    doc_type: str,
    client: BlobClient,
    *,
    mineru_version: str,
    tenant_id: str | None = None,
) -> ImportReport:
    """Import every prepared directory under ``input_dir``.

    One bad bundle is reported and skipped rather than failing the batch: a
    hundred-document import should not be lost to one malformed label.
    """
    report = ImportReport()
    for directory in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        try:
            doc = read_prepared(directory)
            report.imported.append(
                import_document(
                    doc, doc_type, client,
                    mineru_version=mineru_version, tenant_id=tenant_id,
                )
            )
        except (ImportError_, ValueError) as exc:
            report.skipped.append((directory.name, str(exc)))
            log.warning("skipping %s: %s", directory.name, exc)
    return report


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(
        description="Import prepared documents (page images + markdown + golden JSON)"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--mineru-version", required=True,
                        help="the MinerU version that produced the markdown (arch §8a)")
    parser.add_argument("--tenant", default=None)
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('data_pipeline.ingestion.import_prepared', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = import_batch(
        args.input, args.doc_type, BlobClient(),
        mineru_version=args.mineru_version, tenant_id=args.tenant,
    )
    print(report.describe())
    return 1 if report.skipped else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
