"""Run the same PDFs through the base model and through base + adapter, and compare both with gold.

    python -m testing.compare_models --input policy.pdf --gold policy_gold.json --lob homeowners \\
        --adapter /workspace/staging/.../checkpoint-280
    python -m testing.compare_models --input data/bundles/<document>/ --adapter <adapter folder>
    python -m testing.compare_models --input <folder of document folders> --adapter <adapter folder>

``--input`` is a PDF (with ``--gold`` and ``--lob``), a document folder
(``document.pdf`` + ``golden.json`` + ``metadata.json``, the bundle layout), or a
folder of document folders. Each document is imported and OCR'd under its own
tenant (``compare`` by default, never the real data's), then extracted twice
through the serving pipeline (``serving.pipeline.extract``) with ONE model load:
once by the base model alone, once with the adapter applied - not merged.

Output, one folder per document under ``--out``::

    <document>/document.pdf      the input
    <document>/gold.json         the gold label as supplied
    <document>/base.json         the base model's canonical JSON (every schema key)
    <document>/adapter.json      base + adapter's canonical JSON
    <document>/base.all_fields.json, adapter.all_fields.json
                                 every field of the line's schema, each with its value
                                 and its result against gold (testing.comparison.field_report)
    <document>/comparison.xlsx   Summary and Fields sheets (testing.comparison)
    summary.xlsx                 one row per document, and the totals

The finished folder is then uploaded to Blob under
``exports/{tenant}/comparisons/{run}/`` (the raw container: it holds policy data),
checked file by file, and removed from the pod (``--keep-on-pod`` keeps it). The
command prints the azcopy line that downloads it to the laptop, by default into
``D:\\Fine-Tuning\\testing\\results\\comparisons``, inside the laptop's repository
and git-ignored (``--download-to`` changes it).

Confidence is not calibrated here (no release calibrators), so every field is
flagged for review in both JSONs; the comparison is about the values.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TENANT = "compare"
COMPARISONS_ROOT = Path(__file__).resolve().parent / "results" / "comparisons"   # git-ignored


class CompareError(RuntimeError):
    """Raised when the comparison cannot run."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def resolve_inputs(source: Path, staging: Path, *, gold: Path | None = None,
                   lob: str | None = None) -> list[Path]:
    """The document folders to compare.

    A PDF becomes a document folder under ``staging`` (it needs ``gold`` and a
    line of business, from ``lob`` or the gold's ``line_of_business``). A folder
    holding a PDF is one document; a folder of such folders is several.
    """
    if source.is_file():
        if source.suffix.lower() != ".pdf":
            raise CompareError(f"{source} is not a PDF")
        if gold is None:
            raise CompareError("a PDF needs its gold label: pass --gold <gold.json>")
        label = json.loads(gold.read_text(encoding="utf-8"))
        line = lob or _first_line(label.get("line_of_business"))
        if not line:
            raise CompareError("no line of business: pass --lob (e.g. homeowners, personal_auto)")
        folder = staging / source.stem
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, folder / "document.pdf")
        (folder / "golden.json").write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "metadata.json").write_text(json.dumps({"lob": line}), encoding="utf-8")
        return [folder]
    if not source.is_dir():
        raise CompareError(f"{source} does not exist")
    if any(source.glob("*.pdf")):
        return [source]
    folders = sorted(p for p in source.iterdir() if p.is_dir() and any(p.glob("*.pdf")))
    if not folders:
        raise CompareError(f"{source} holds no PDF and no document folders")
    return folders


def _first_line(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value) if value else None


def import_and_ocr(folders: list[Path], *, tenant: str, doc_type: str = "policy") -> dict[str, str]:
    """Import each document folder, OCR what is not yet OCR'd. Returns folder name -> source_id.

    Re-running on the same PDF finds it by checksum and reuses its id and OCR.
    """
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, for_ingestion
    from data_pipeline.ingestion.import_labeled_pdfs import check_bundle, import_bundle, read_bundle
    from data_pipeline.ingestion.pull_raw_pdfs import _existing_checksums

    client, raw = BlobClient(), for_ingestion()
    known = _existing_checksums(raw, doc_type, tenant)
    ids: dict[str, str] = {}
    for folder in folders:
        bundle = read_bundle(folder)
        check = check_bundle(bundle, doc_type)
        for problem in check.errors[:3]:
            log.warning("%s: gold label: %s", folder.name, problem)
        if not bundle.lob:
            raise CompareError(f"{folder.name}: no line of business in metadata.json or the gold label")
        source_id, status = import_bundle(bundle, doc_type, client, raw, tenant_id=tenant, known_checksums=known)
        ids[folder.name] = source_id
        log.info("%s -> %s (%s)", folder.name, source_id, status)

    pending = [sid for sid in ids.values() if not client.exists(paths.ocr_meta(doc_type, sid, tenant))]
    if pending:
        from orchestration.smoke_run import OCR_PYTHON

        if not Path(OCR_PYTHON).is_file():
            raise CompareError(
                f"{len(pending)} document(s) need OCR, and the OCR environment ({OCR_PYTHON}) is not here. "
                "Run this on the pod.")
        command = [str(OCR_PYTHON), "-m", "data_pipeline.ocr.run_mineru", "--doc-type", doc_type,
                   "--source-ids", *pending, "--tenant", tenant]
        log.info("OCR: %s", " ".join(command))
        # Already inside this job's tmux session: OCR runs in place, then frees the GPU.
        if subprocess.run(command, env={**os.environ, "FIDEON_DETACHED": "1"}).returncode != 0:
            raise CompareError("OCR failed; see the output above")
    return ids


# --------------------------------------------------------------------------
# Both routes
# --------------------------------------------------------------------------


def _extract(model: Any, request: Any, doc_type: str) -> tuple[dict[str, Any], list[str], str | None]:
    """One route's canonical JSON and its review flags, or ``{}``, none and the error."""
    from calibration.fit_calibration import CalibrationParams
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract

    calibration = CalibrationParams(method="temperature", doc_type=doc_type,
                                    model_version=model.tag, temperature=1.0)
    try:
        result = extract(request, model, StaticClassifier(doc_type, request.known_acord_form), calibration,
                         strict_schema=False)
    except Exception as exc:  # noqa: BLE001 - the other route still runs
        log.error("%s on %s failed: %s: %s", model.tag, request.source_id, type(exc).__name__, exc)
        return {}, [], f"{type(exc).__name__}: {exc}"
    return result.extraction, list(result.review_flags), None


def _all_fields(answer: dict[str, Any], flags: list[str], gold: dict[str, Any] | None, request: Any,
                doc_type: str) -> dict[str, Any] | None:
    """testing.comparison.field_report for one route; ``None`` when it cannot be built."""
    from testing.comparison import field_report

    try:
        return field_report(answer, gold, doc_type=doc_type, acord_form=request.known_acord_form,
                            lob=request.known_lob, review_flags=flags)
    except Exception as exc:  # noqa: BLE001 - a report on the answer, never a reason to stop the run
        log.warning("%s: the all-fields report could not be built (%s: %s)",
                    request.source_id, type(exc).__name__, exc)
        return None


def compare_documents(base_model: Any, adapter_model: Any, client: Any, documents: list[tuple[str, str, Path]],
                      out: Path, *, tenant: str, doc_type: str = "policy", mode: str = "ocr_plus_image",
                      images_root: Path | None = None) -> list[tuple[str, str, Any]]:
    """Extract each ``(name, source_id, pdf)`` with both models; write its folder. Returns the comparisons."""
    from testing.comparison import compare, write_comparison, write_overview
    from testing.run_extraction import PAGE_CACHE, document_request

    compared: list[tuple[str, str, Any]] = []
    errors: dict[str, dict[str, str]] = {}
    for index, (name, source_id, pdf) in enumerate(documents, start=1):
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        request, gold = document_request(client, source_id, doc_type=doc_type, tenant_id=tenant, mode=mode,
                                         images_root=images_root or PAGE_CACHE)
        base, base_flags, base_error = _extract(base_model, request, doc_type)
        adapter, adapter_flags, adapter_error = _extract(adapter_model, request, doc_type)
        for label, error in (("base", base_error), ("adapter", adapter_error)):
            if error:
                errors.setdefault(name, {})[label] = error

        if pdf.is_file():
            shutil.copy2(pdf, folder / "document.pdf")
        for file_name, body in (("gold.json", gold or {}), ("base.json", base), ("adapter.json", adapter)):
            (folder / file_name).write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        for label, answer, flags in (("base", base, base_flags), ("adapter", adapter, adapter_flags)):
            report = _all_fields(answer, flags, gold, request, doc_type)
            if report is not None:
                (folder / f"{label}.all_fields.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        line = str(request.known_lob)
        comparison = compare(gold, base, adapter, doc_type=doc_type, acord_form=request.known_acord_form,
                             lob=request.known_lob)
        write_comparison(folder / "comparison.xlsx", comparison, document=name, line=line,
                         adapter_label=f"Base + adapter ({adapter_model.tag})")
        compared.append((name, line, comparison))
        log.info("[%d/%d] %s: base F1 %s, adapter F1 %s -> %s", index, len(documents), name,
                 _pct(comparison.base.f1), _pct(comparison.adapter.f1), folder)

    write_overview(out / "summary.xlsx", compared)
    if errors:
        (out / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    return compared


#: Where the download command points by default: the repository on the laptop,
#: under testing/results/ - git-ignored, so the policy data is never committed.
DEFAULT_DOWNLOAD_TO = r"D:\Fine-Tuning\testing\results\comparisons"


def export_run(out: Path, tenant: str, *, client: Any = None, keep_local: bool = False) -> tuple[str, int]:
    """Upload a finished run folder to ``exports/`` in Blob, check it, and remove the pod copy.

    Returns ``(blob prefix, files uploaded)``. The pod copy is removed only once
    every file is listed under the prefix; ``keep_local`` keeps it regardless.
    ``_inputs/`` (the PDF and gold as given, already in each document folder) is
    not uploaded.
    """
    from artifact_registry import paths
    from artifact_registry.blob_client import for_ingestion

    client = client or for_ingestion()
    prefix = paths.export_dir("comparisons", out.name, tenant)
    files = [f for f in sorted(out.rglob("*")) if f.is_file() and "_inputs" not in f.relative_to(out).parts]
    for file in files:
        client.write_bytes(f"{prefix}/{file.relative_to(out).as_posix()}", file.read_bytes())
    stored = set(client.list(prefix + "/"))
    missing = [f for f in files if f"{prefix}/{f.relative_to(out).as_posix()}" not in stored]
    if missing:
        raise CompareError(f"{len(missing)} file(s) did not reach {prefix}; the pod copy is kept at {out}")
    if not keep_local:
        shutil.rmtree(out)
    return prefix, len(files)


def download_command(prefix: str, *, container: str, account: str | None,
                     target: str = DEFAULT_DOWNLOAD_TO) -> str:
    """The azcopy command that copies an export to the laptop (the SAS is the operator's)."""
    host = f"https://{account or '<account>'}.blob.core.windows.net"
    return f'azcopy copy "{host}/{container}/{prefix}?<SAS>" "{target}" --recursive'


def _account_name() -> str | None:
    """The storage account's name - the name only, never the key or the token.

    From the SAS login's account URL when set, else from the connection string.
    """
    url = (os.environ.get("AZURE_STORAGE_ACCOUNT_URL") or "").strip()
    if url:
        host = url.split("://", 1)[-1].split("/", 1)[0]
        return host.split(".", 1)[0] or None
    for part in (os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or "").split(";"):
        if part.startswith("AccountName="):
            return part.split("=", 1)[1]
    return None


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "-"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over the functions above
    parser = argparse.ArgumentParser(description="Base vs base + adapter vs gold, per PDF")
    parser.add_argument("--input", required=True, type=Path,
                        help="a PDF (with --gold and --lob), a document folder, or a folder of them")
    parser.add_argument("--adapter", required=True,
                        help="LoRA adapter folder (checkpoint-N or the run's adapter folder; local or Blob)")
    parser.add_argument("--gold", type=Path, default=None, help="the gold JSON, when --input is a PDF")
    parser.add_argument("--lob", default=None, help="line of business, when --input is a PDF")
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help="where the documents are imported and OCR'd")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default under testing/results)")
    parser.add_argument("--label", default=None, help="name for the adapter in the outputs")
    parser.add_argument("--mode", choices=["ocr_plus_image", "image_only"], default="ocr_plus_image")
    parser.add_argument("--doc-type", dest="doc_type", default="policy")
    parser.add_argument("--keep-on-pod", dest="keep_on_pod", action="store_true",
                        help="keep the output folder on the pod after it is uploaded for download")
    parser.add_argument("--download-to", dest="download_to", default=DEFAULT_DOWNLOAD_TO,
                        help="the laptop folder the printed download command copies into")
    args = parser.parse_args(argv)
    if args.tenant in ("", "default"):
        parser.error("compare under its own tenant, never the real data's")

    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("testing.compare_models", argv, hint="compare"):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from artifact_registry.blob_client import BlobClient
    from inference_core.model_runner import load_model, release_model, without_adapter

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = args.out or COMPARISONS_ROOT / f"{args.label or 'compare'}-{stamp}"
    try:
        folders = resolve_inputs(args.input, out / "_inputs", gold=args.gold, lob=args.lob)
        ids = import_and_ocr(folders, tenant=args.tenant, doc_type=args.doc_type)
    except CompareError as exc:
        print(f"compare: {exc}")
        return 1

    client = BlobClient()
    adapter_model = load_model("base", client, adapter=args.adapter, label=args.label)
    base_model = without_adapter(adapter_model)          # the same engine, no adapter
    try:
        documents = [(folder.name, ids[folder.name], next(folder.glob("*.pdf"))) for folder in folders]
        compared = compare_documents(base_model, adapter_model, client, documents, out,
                                     tenant=args.tenant, doc_type=args.doc_type, mode=args.mode)
    finally:
        release_model(adapter_model)
    print(f"\n{len(compared)} document(s) compared.")
    try:
        prefix, files = export_run(out, args.tenant, keep_local=args.keep_on_pod)
    except CompareError as exc:
        print(f"compare: upload for download failed: {exc}")
        return 1
    from artifact_registry.blob_client import for_ingestion

    command = download_command(prefix, container=for_ingestion().raw_container, account=_account_name(),
                               target=args.download_to)
    print(f"{files} file(s) uploaded to {prefix}" + ("" if args.keep_on_pod else " (pod copy removed)"))
    print("\nOn your laptop, download them with (your SAS for the raw container, read + list):\n")
    print(f"  {command}\n")
    print("  per document: <name>/document.pdf, gold.json, base.json, adapter.json, comparison.xlsx;"
          " overview: summary.xlsx")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
