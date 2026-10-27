"""Import PDFs that already have golden labels (IMPL-03 §1, IMPL-04, arch §8a).

The delivery shape this exists for: a directory per document holding the source
**PDF**, its canonical **golden JSON**, and optional metadata. Both halves of a
training example arrive together, and neither the review tool nor MinerU needs to
have run first.

**OCR is not done here.** MinerU runs on a GPU, as pipeline stage 2, against the
PDFs this stores — so an import is cheap, runs anywhere, and leaves a document in
exactly the state a freshly-ingested one is in, plus its label. The next
``finetune`` run OCRs it on the pod and the dataset build picks it up. Rendering
pages locally would mean a second MinerU install whose version has to match the
pod's, and a version mismatch is distribution shift the model sees but nothing
reports (arch §8a).

**Validation reports, it does not stonewall.** Every problem with every document
is collected and printed, so a batch can be corrected in one pass rather than one
document per run. What it will not do is let an invalid label through: a bad
golden is a training target nobody checked, and the first symptom is a model that
learned the wrong shape.

``--validate-only`` runs the whole check and writes nothing — the mode to use
while the labels are still being corrected.

Compare :mod:`data_pipeline.ingestion.import_prepared`, which takes documents
that have ALREADY been through MinerU (page images + markdown) and therefore
skips ingestion and OCR entirely.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES
from common.schemas import SchemaError, iter_validation_errors, old_shape_hint

log = logging.getLogger(__name__)

#: How many schema errors to report per document. Enough to fix a label in one
#: pass; not so many that one broken file buries the rest of the batch.
MAX_ERRORS_REPORTED = 10


class LabeledPdfError(RuntimeError):
    """Raised when a labeled-PDF bundle cannot be read at all."""


@dataclass
class LabeledPdf:
    """One bundle: the source PDF and the label that goes with it."""

    directory: Path
    pdf: Path
    golden: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def acord_form(self) -> str | None:
        return self.metadata.get("acord_form")

    @property
    def lob(self) -> Any:
        from common.lob import merge_line

        # Classic auto is read as personal auto (common.lob.MERGED_LINES).
        return merge_line(self.metadata.get("lob") or self.golden.get("line_of_business"))


@dataclass
class DocumentCheck:
    """What is wrong with one bundle, if anything."""

    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ImportReport:
    """The outcome of one import or validation run."""

    checks: list[DocumentCheck] = field(default_factory=list)
    imported: dict[str, str] = field(default_factory=dict)      # directory -> source_id
    duplicates: dict[str, str] = field(default_factory=dict)    # directory -> existing id
    validated_only: bool = False

    @property
    def failed(self) -> list[DocumentCheck]:
        return [c for c in self.checks if not c.ok]

    @property
    def ok(self) -> bool:
        return not self.failed

    def describe(self) -> str:
        verb = "checked" if self.validated_only else "imported"
        lines = [
            f"{verb} {len(self.imported) or len(self.checks) - len(self.failed)} document(s); "
            f"{len(self.failed)} with problems"
        ]
        for name, existing in sorted(self.duplicates.items()):
            lines.append(f"  {name}: already ingested as {existing} — label refreshed, PDF kept")
        for check in self.checks:
            for warning in check.warnings:
                lines.append(f"  {check.name}: {warning}")
            for error in check.errors:
                lines.append(f"  {check.name}: ERROR {error}")
        return "\n".join(lines)


def read_bundle(directory: Path) -> LabeledPdf:
    """Load one bundle. Raises only when there is nothing to check."""
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise LabeledPdfError(f"{directory.name} holds no PDF")
    if len(pdfs) > 1:
        raise LabeledPdfError(
            f"{directory.name} holds {len(pdfs)} PDFs. One directory is one document, because "
            "one golden label describes one document — two PDFs means the label describes "
            "whichever was read first."
        )

    golden_path = directory / "golden.json"
    if not golden_path.exists():
        raise LabeledPdfError(f"{directory.name} has no golden.json")

    bundle = LabeledPdf(directory=directory, pdf=pdfs[0])
    bundle.golden = json.loads(golden_path.read_text(encoding="utf-8"))
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        bundle.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bundle


def check_bundle(bundle: LabeledPdf, doc_type: str) -> DocumentCheck:
    """Every problem with one bundle, collected rather than raised.

    Errors block the document; warnings do not, but each one names something the
    pipeline will do less well without.
    """
    check = DocumentCheck(name=bundle.directory.name)

    if not isinstance(bundle.golden, dict):
        check.errors.append("golden.json is not a JSON object")
        return check

    try:
        errors = list(iter_validation_errors(
            bundle.golden, doc_type, bundle.acord_form, bundle.lob
        ))
    except SchemaError as exc:
        check.errors.append(f"no schema to validate against — {exc}")
        return check

    hint = old_shape_hint(bundle.golden, doc_type, bundle.acord_form, bundle.lob) if errors else None
    if hint:
        check.errors.append(hint)
    check.errors.extend(errors[:MAX_ERRORS_REPORTED])
    if len(errors) > MAX_ERRORS_REPORTED:
        check.errors.append(f"... and {len(errors) - MAX_ERRORS_REPORTED} more")

    if not bundle.metadata.get("field_provenance"):
        check.warnings.append(
            "no field_provenance: per-alias evaluation cannot score this document until "
            "alias derivation backfills its surface labels"
        )
    # No policy_number warning: the schema REQUIRES it, so a label that reaches
    # here already carries one. A check that cannot fire reads as a guarantee
    # nobody is providing. It is recorded on the label metadata for grouping.
    if not bundle.lob:
        check.warnings.append("no line of business: the per-LOB schema cannot be selected")
    split = bundle.metadata.get("split")
    if split is not None and str(split).lower() not in ("train", "val", "test"):
        check.errors.append(f"metadata split {split!r} is not train, val or test")

    # One convention for form editions, on the lines the label rules hold for
    # (label_rules_apply): an edition left inside a form number the rules would
    # have moved means the bundle was never prepared. A self-contained line's
    # golds never go through them, so nothing is asked of those.
    from data_pipeline.ingestion.label_rules import edition_problems

    errors, warnings = edition_problems(bundle.golden, doc_type, bundle.lob)
    check.errors.extend(errors)
    check.warnings.extend(warnings)

    # Every gold needs a review record to train or score (decision 2026-10-22):
    # one without, or a twin whose seed nobody reviewed, is blocked.
    from data_pipeline.labeling.export_golden_labels import review_problem

    if problem := review_problem(bundle.golden, bundle.metadata):
        check.errors.append(problem)
    return check


def twin_index(metadata: dict[str, Any]) -> int | None:
    """Which twin of its seed a synthetic document is; ``None`` for a real one."""
    if not metadata.get("synthetic"):
        return None
    value = metadata.get("twin_index", metadata.get("sample"))
    return int(value) if isinstance(value, (int, str)) and str(value).strip().isdigit() else None


def import_history(
    client: BlobClient, key: str, folder: str, batch: str | None = None,
) -> list[dict[str, str | None]]:
    """Every delivered folder the document was imported from, this import last.

    Each entry is the folder's name and the intake batch it was pulled from,
    when the import pulled one (``--from-blob``), else ``None``. Read from the
    label metadata this import replaces: ``imported_from`` names only the last
    folder, and a re-delivery of the same PDF (a checksum duplicate, in another
    folder or batch) rewrote it, so the purge plan looked for the last copy
    alone. A record from before the history was kept starts it with its
    ``imported_from``.
    """
    try:
        previous = client.read_json(key) if client.exists(key) else {}
    except ValueError:  # not JSON: the history starts again from this import
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    recorded = previous.get("imported_from_history")
    history = [
        {"folder": str(entry["folder"]), "batch": str(entry["batch"]) if entry.get("batch") else None}
        for entry in (recorded if isinstance(recorded, list) else [])
        if isinstance(entry, dict) and entry.get("folder")
    ]
    if not history and previous.get("imported_from"):
        history = [{"folder": str(previous["imported_from"]), "batch": None}]
    entry = {"folder": folder, "batch": batch or None}
    return [*(e for e in history if e != entry), entry]


def _gold_policy_number(golden: dict) -> str | None:
    from data_pipeline.ingestion.prepare_bundles import gold_policy_number

    return gold_policy_number(golden)


def import_bundle(
    bundle: LabeledPdf,
    doc_type: str,
    client: BlobClient,
    raw_client: BlobClient,
    *,
    tenant_id: str | None = None,
    known_checksums: dict[str, str] | None = None,
    batch: str | None = None,
) -> tuple[str, str]:
    """Store the PDF and its label. Returns ``(source_id, status)``.

    The PDF goes through the normal ingestion path — same checksum dedup, same
    write-once raw layer — so an imported document is indistinguishable from one
    ingested any other way by the time OCR sees it. ``batch`` is the intake
    batch the bundle was pulled from, when the import pulled one.
    """
    from data_pipeline.ingestion.pull_raw_pdfs import ingest_pdf
    from data_pipeline.labeling.export_golden_labels import review_record

    source_id, status = ingest_pdf(
        bundle.pdf, doc_type, raw_client,
        tenant_id=tenant_id,
        known_checksums=known_checksums,
        source_system="labeled_import",
    )
    if source_id is None or status.startswith("failed"):
        raise LabeledPdfError(f"{bundle.directory.name}: ingestion failed ({status})")

    # Written through the general client: golden labels are not the PII layer,
    # and the importer holds the raw context only for the PDF itself.
    client.write_json(paths.golden_label(doc_type, source_id, tenant_id), bundle.golden)
    metadata_key = paths.label_metadata(doc_type, source_id, tenant_id)
    history = import_history(client, metadata_key, bundle.directory.name, batch)
    client.write_json(metadata_key, {
        "source_id": source_id,
        "doc_type": doc_type,
        "acord_form": bundle.acord_form,
        "lob": bundle.lob,
        "field_provenance": bundle.metadata.get("field_provenance", {}),
        "document_kind": bundle.metadata.get("document_kind"),
        # A canonical gold holds it at policy.policy_number; read at the top
        # level only, it was null for every one.
        "policy_number": bundle.metadata.get("policy_number") or _gold_policy_number(bundle.golden),
        "template_id": bundle.metadata.get("template_id"),
        # Fideon SPEC_09 amendment item 8: every example carries its carrier and,
        # for a synthetic twin, which twin of its seed it is (the bundle's
        # `sample`). The carrier also decides the per-line hold-out.
        "carrier": bundle.metadata.get("carrier"),
        "twin_index": twin_index(bundle.metadata),
        "render_mode": bundle.metadata.get("render_mode"),
        # Label values no page prints (prepare_bundles' label rules): the corpus
        # build leaves out the windows that hold them.
        "unprinted_values": list(bundle.metadata.get("unprinted_values") or []),
        # A test document whose carrier no training document has, as the bundler
        # marked it (prepare_bundles): the corpus build records it with the split
        # (pipeline_dag.delivered_held_out_ids) and the frozen set reports it apart.
        # Not copied, every delivered document read as a seen carrier's.
        "held_out_carrier": bool(bundle.metadata.get("held_out_carrier", False)),
        "synthetic": bool(bundle.metadata.get("synthetic", False)),
        # The delivery's own split (train/val/test), when it was split upstream:
        # the corpus build then uses it as given (split_groups.assign_delivered_splits).
        "split": str(bundle.metadata["split"]).lower() if bundle.metadata.get("split") else None,
        # Who reviewed the label and when, what drafted it, and for a twin the
        # seed whose record it inherited: every reader asks review_problem of
        # these before the gold trains, is frozen or scores.
        **review_record(bundle.metadata),
        "labeled_at": datetime.now(UTC).isoformat(),
        "imported_from": bundle.directory.name,
        # Every folder (and pulled batch) it was imported from, this one last:
        # each is a delivered copy a purge has to find (pull_raw_pdfs.purge_plan).
        "imported_from_history": history,
        # OCR has NOT run: MinerU runs on the GPU pod as stage 2. Recorded so a
        # later reader can tell an import awaiting OCR from a document whose OCR
        # failed.
        "awaiting_ocr": True,
    })
    return source_id, status


def import_batch(
    input_dir: Path,
    doc_type: str,
    client: BlobClient,
    raw_client: BlobClient,
    *,
    tenant_id: str | None = None,
    validate_only: bool = False,
    batch: str | None = None,
) -> ImportReport:
    """Check every bundle, then import the ones that passed.

    Checked **first, all of them**, so a batch is corrected in one pass. Nothing
    is written when any document fails validation in ``validate_only`` mode, and
    in a real import a failing document is skipped while the rest proceed.

    ``batch`` is the intake batch ``input_dir`` was pulled from (``--from-blob``):
    recorded with each document's folder, so a purge finds its delivered copy.
    """
    if doc_type not in ACTIVE_DOC_TYPES:
        raise LabeledPdfError(f"unknown doc_type {doc_type!r}; active: {list(ACTIVE_DOC_TYPES)}")

    report = ImportReport(validated_only=validate_only)
    bundles: list[LabeledPdf] = []

    for directory in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        try:
            bundle = read_bundle(directory)
        except (LabeledPdfError, ValueError) as exc:
            report.checks.append(DocumentCheck(name=directory.name, errors=[str(exc)]))
            continue
        check = check_bundle(bundle, doc_type)
        report.checks.append(check)
        if check.ok:
            bundles.append(bundle)

    if validate_only:
        return report

    # The same checksum map `ingest_directory` builds, so re-importing a
    # corrected batch re-labels the documents already stored rather than storing
    # a second copy of each PDF under a new source_id.
    from data_pipeline.ingestion.pull_raw_pdfs import IngestionError, _existing_checksums

    known = _existing_checksums(raw_client, doc_type, tenant_id)
    for bundle in bundles:
        try:
            source_id, status = import_bundle(
                bundle, doc_type, client, raw_client,
                tenant_id=tenant_id, known_checksums=known, batch=batch,
            )
        # An unreadable PDF is that document's failure, not the batch's: it ended
        # an import part-way, every later document unimported.
        except (LabeledPdfError, IngestionError, ValueError) as exc:
            next(c for c in report.checks if c.name == bundle.directory.name).errors.append(str(exc))
            continue
        report.imported[bundle.directory.name] = source_id
        if status == "duplicate":
            report.duplicates[bundle.directory.name] = source_id
    return report


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from artifact_registry.blob_client import for_ingestion

    parser = argparse.ArgumentParser(
        description="Import labeled PDFs (document.pdf + golden.json per directory). "
                    "MinerU runs later, on the GPU pod, as pipeline stage 2."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="a local folder of document folders")
    source.add_argument("--from-blob", metavar="BATCH",
                        help="a batch staged under intake/BATCH/ in the raw container: pulled onto "
                             "the volume first (data_pipeline.ingestion.pull_intake), then imported")
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--validate-only", action="store_true",
                        help="check every bundle and write nothing")
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('data_pipeline.ingestion.import_labeled_pdfs', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw_client = for_ingestion()
    input_dir = args.input
    if args.from_blob:
        from data_pipeline.ingestion.pull_intake import IntakeError, pull_intake

        try:
            pulled = pull_intake(raw_client, args.from_blob)
        except (IntakeError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        print(pulled.describe())
        if pulled.failed:
            print("Some files did not download; re-run to retry. Nothing was imported.", file=sys.stderr)
            return 1
        input_dir = pulled.local_dir
    report = import_batch(
        input_dir, args.doc_type, BlobClient(), raw_client,
        tenant_id=args.tenant, validate_only=args.validate_only, batch=args.from_blob,
    )
    print(report.describe())
    if report.ok and not args.validate_only:
        print(
            "\nNext: OCR runs on the GPU pod — "
            "python -m orchestration.run finetune (the cycle takes the next version)"
        )
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
