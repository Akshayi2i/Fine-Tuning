"""Audit a labeled-PDF training folder before it is uploaded or imported.

Read-only. Every document folder is opened, checked, and reported on; nothing is
written next to the data. The report says what would stop an import, which
labelled values the PDF does not support, which values are in the wrong format,
and whether there is enough of each line to train and to measure.

The checks, in five layers:

1. **Structure** (blocker): one PDF, a ``golden.json`` and a ``metadata.json``
   per folder; the PDF opens and has pages; no two folders hold the same PDF.
2. **Label shape** (blocker): the importer's own validation
   (:func:`data_pipeline.ingestion.import_labeled_pdfs.check_bundle`), so the
   audit and the import never disagree — plus a ``lob`` that names a canonical
   schema (and, with ``--lines``, is one of those lines), and every ``page_ref``
   inside the PDF.
3. **Values against the PDF** (per value): on a page with a text layer, each
   value's ``raw`` must appear on the page its ``page_ref`` names. Found on
   another page means a wrong ``page_ref``; found nowhere means a value to check.
   Pages without text (scans) are left for the check after OCR.
4. **Formats**: dates in ``parsed`` are readable (the build writes them MM/DD/YYYY); amounts in ``parsed`` agree
   with ``raw``; an effective date precedes its expiration date.
5. **Totals**: documents per line, digital vs scanned, page counts, lines too
   small to measure, and whether the test split can reach the 150 documents the
   golden eval set is frozen from.

Plus a seeded **spot-check sample** — 5% per line — for a person to verify by
hand, the only check that a value is *right* and not just present.

    python -m data_pipeline.audit                                  # the default folder
    python -m data_pipeline.audit --input "D:/data/personal" --lines homeowners,personal_auto

The CSVs quote label values: they hold the documents' personal data. They are
written to ``data/audit_report/`` (git-ignored) and must not leave the machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import grounding
from data_pipeline.ocr.run_mineru import TEXT_LAYER_MIN_CHARS as MIN_TEXT_CHARS

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO / "data" / "bundles"   # prepare_bundles output
DEFAULT_OUT = REPO / "data" / "audit_report"

#: Values this short ("Y", "1") match almost any page; they are not checked.
MIN_CHECKABLE_CHARS = grounding.MIN_CHECKABLE_CHARS
#: Share of each line drawn for the manual spot check.
SPOT_CHECK_SHARE = 0.05


@dataclass
class Finding:
    folder: str
    severity: str          # blocker | warning | info
    check: str
    detail: str


@dataclass
class ValueCheck:
    folder: str
    path: str
    raw: str
    page_ref: str
    result: str            # ok | wrong_page | not_found | no_page_ref | needs_ocr | skipped_short
    found_on: str = ""


@dataclass
class DocumentAudit:
    folder: str
    line: str = ""
    pages: int = 0
    text_pages: int = 0
    importable: bool = False
    values: list[ValueCheck] = field(default_factory=list)
    #: From metadata.json, when the delivery was split upstream (prepare_bundles).
    split: str | None = None
    synthetic: bool = False
    template_id: str | None = None

    @property
    def digital(self) -> bool:
        return self.pages > 0 and self.text_pages / self.pages >= 0.5


@dataclass
class AuditReport:
    root: str
    #: The lines documents must be of (``--lines``); None for every line.
    lines: frozenset[str] | None = None
    documents: list[DocumentAudit] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    formats: list[Finding] = field(default_factory=list)
    spot_check: list[str] = field(default_factory=list)

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def values(self) -> list[ValueCheck]:
        return [v for d in self.documents for v in d.values]


# --------------------------------------------------------------------------
# Text matching: common.grounding, the one definition the hallucination metric,
# the calibration feature and label verification share with the audit
# --------------------------------------------------------------------------

normalise = grounding.normalise
appears = grounding.appears
_Page = grounding.PageText
_appears = grounding.appears_on


def iter_envelopes(node: Any, path: str = ""):
    from common.canonical import is_field_value

    if is_field_value(node):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from iter_envelopes(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_envelopes(value, f"{path}[{index}]")


# --------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------


def document_folders(root: Path) -> list[Path]:
    """The folders the importer will read: the IMMEDIATE subfolders of ``root``.

    ``import_labeled_pdfs`` takes each direct subfolder of ``--input`` as one
    document, so that is what is audited — a layout the audit passes is one the
    importer can import. Documents nested deeper are reported by
    :func:`misplaced`.
    """
    return sorted(p for p in root.iterdir() if p.is_dir())


def misplaced(root: Path) -> list[Finding]:
    """PDFs the importer would never see: loose in ``root``, or below a subfolder."""
    found: list[Finding] = []
    loose = sorted({*root.glob("*.pdf"), *root.glob("*.PDF")})
    if loose:
        found.append(Finding(".", "blocker", "structure",
                             f"{len(loose)} PDF file(s) directly in the input folder; each document "
                             "needs its own folder (<input>/<document>/{pdf, golden.json, metadata.json})"))
    for folder in document_folders(root):
        nested = [p for p in folder.rglob("*") if p.suffix.lower() == ".pdf" and p.parent != folder]
        if nested:
            found.append(Finding(folder.name, "blocker", "structure",
                                 f"{len(nested)} PDF file(s) in subfolders below {folder.name}/; the "
                                 "importer reads only <input>/<document>/ — move each document up"))
    return found


def _read_pdf(pdf: Path) -> tuple[list[str], str | None]:
    """Page texts, or an error. Imported here: PyMuPDF is in the [data] group."""
    import pymupdf

    try:
        with pymupdf.open(pdf) as doc:
            if doc.needs_pass:
                return [], "PDF is password-protected"
            return [page.get_text() for page in doc], None
    except Exception as exc:  # noqa: BLE001 - any unreadable PDF is one finding
        return [], f"PDF cannot be opened ({type(exc).__name__}: {exc})"


def audit_document(folder: Path, report: AuditReport, checksums: dict[str, str]) -> DocumentAudit:
    from common.lob import known_lines, lob_lines, within_lines
    from data_pipeline.ingestion.import_labeled_pdfs import LabeledPdf, check_bundle

    name = str(folder.relative_to(report.root)) if Path(report.root) in folder.parents else folder.name
    audit = DocumentAudit(folder=name)

    def block(check: str, detail: str) -> None:
        report.findings.append(Finding(name, "blocker", check, detail))

    def warn(check: str, detail: str) -> None:
        report.findings.append(Finding(name, "warning", check, detail))

    # ---- 1. structure
    pdfs = sorted({*folder.glob("*.pdf"), *folder.glob("*.PDF")})
    if len(pdfs) != 1:
        block("structure", f"{len(pdfs)} PDF files (exactly one per folder)")
    label: Any = None
    metadata: dict[str, Any] = {}
    for filename in ("golden.json", "metadata.json"):
        path = folder / filename
        if not path.exists():
            block("structure", f"no {filename}")
            continue
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            block("structure", f"{filename} does not parse ({exc})")
            continue
        if filename == "golden.json":
            label = body
        elif isinstance(body, dict):
            metadata = body
        else:
            block("structure", "metadata.json is not a JSON object")

    page_texts: list[str] = []
    if len(pdfs) == 1:
        digest = hashlib.sha256(pdfs[0].read_bytes()).hexdigest()
        if digest in checksums:
            block("structure", f"same PDF as {checksums[digest]}")
        else:
            checksums[digest] = name
        page_texts, error = _read_pdf(pdfs[0])
        if error:
            block("structure", error)
        elif not page_texts:
            block("structure", "PDF has no pages")
    audit.pages = len(page_texts)
    has_text = [len(t.strip()) >= MIN_TEXT_CHARS for t in page_texts]
    audit.text_pages = sum(has_text)

    # ---- 2. label shape
    audit.split = str(metadata["split"]).lower() if metadata.get("split") else None
    audit.synthetic = bool(metadata.get("synthetic", False))
    audit.template_id = metadata.get("template_id")
    lob = metadata.get("lob")
    lines = lob_lines(lob)
    audit.line = "+".join(sorted(lines)) if lines else ""
    if not lines:
        block("label", "metadata.json has no lob (line of business)")
    else:
        unknown = sorted(lines - known_lines())
        if unknown:
            block("label", f"lob {lob!r}: {unknown} has no canonical schema")
        elif not within_lines(lob, report.lines):
            block("label", f"lob {lob!r} is outside the lines audited ({sorted(report.lines or ())})")

    if isinstance(label, dict) and len(pdfs) == 1:
        check = check_bundle(LabeledPdf(directory=folder, pdf=pdfs[0], golden=label,
                                        metadata=metadata), "policy")
        for error in check.errors:
            block("label", error)
        for warning in check.warnings:
            warn("label", warning)
    elif label is not None and not isinstance(label, dict):
        block("label", "golden.json is not a JSON object")

    envelopes = list(iter_envelopes(label)) if isinstance(label, dict) else []
    for path, envelope in envelopes:
        if not isinstance(envelope.get("page_ref") or [], list):
            # One malformed label is one finding, never a crash of the whole run.
            block("label", f"{path}: page_ref must be a list of page numbers, got "
                           f"{envelope.get('page_ref')!r}")
    # Only against a PDF that opened: an unreadable one is already a blocker, and
    # reporting every page_ref against "0 pages" would bury the real cause.
    for path, envelope in envelopes if audit.pages else ():
        for ref in _refs(envelope):
            if not isinstance(ref, int) or not 1 <= ref <= audit.pages:
                block("label", f"{path}: page_ref {ref} is not a page of this {audit.pages}-page PDF")

    audit.importable = not any(f.folder == name and f.severity == "blocker" for f in report.findings)

    # ---- 3. values against the PDF — only for a PDF that opened: an unreadable
    # one is its blocker, not N "needs OCR" rows.
    pages = [_Page(text) for text in page_texts]
    for path, envelope in envelopes if audit.pages else ():
        raw = envelope.get("raw")
        if raw in (None, "") or isinstance(raw, (dict, list)):
            continue
        raw = str(raw)
        refs = [r for r in _refs(envelope) if isinstance(r, int) and 1 <= r <= audit.pages]
        record = ValueCheck(name, path, raw, ",".join(map(str, refs)), "")
        if len(normalise(raw)) < MIN_CHECKABLE_CHARS:
            record.result = "skipped_short"
        elif refs and not any(has_text[r - 1] for r in refs) or not refs and not audit.text_pages:
            record.result = "needs_ocr"
        else:
            value = normalise(raw)
            on = [i + 1 for i, page in enumerate(pages) if has_text[i] and _appears(value, page)]
            record.found_on = ",".join(map(str, on[:5]))
            if not refs:
                record.result = "no_page_ref" if on else "not_found"
            elif set(on) & set(refs):
                record.result = "ok"
            elif on:
                record.result = "wrong_page"
            else:
                record.result = "not_found"
        audit.values.append(record)

    # ---- 4. formats
    _check_formats(name, envelopes, report)
    return audit


def _refs(envelope: dict) -> list:
    """``page_ref`` as a list; a malformed scalar is reported by the shape check."""
    refs = envelope.get("page_ref")
    if refs is None:
        return []
    return refs if isinstance(refs, list) else [refs]


#: A month and year with no day, as edition dates print it: 05-17, 5/2017, 05 11, 0699, (06/99), 00/00.
_MONTH_YEAR = re.compile(r"^\s*\(?\s*(0?[1-9]|1[0-2]|00)\s*[/. -]?\s*(\d{2}|\d{4})\s*\)?\s*$")
#: Two or more figures in one value: a split limit ("$100,000/$300,000", "100/300").
_FIGURES = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _check_formats(name: str, envelopes: list[tuple[str, dict]], report: AuditReport) -> None:
    from common.normalize import infer_field_kind, normalize_currency, normalize_date

    def flag(check: str, detail: str) -> None:
        report.formats.append(Finding(name, "warning", check, detail))

    dates: dict[str, dict[str, str]] = defaultdict(dict)
    for path, envelope in envelopes:
        parsed = envelope.get("parsed")
        raw = envelope.get("raw")
        if raw in (None, "") and parsed in (None, ""):
            continue
        kind = infer_field_kind(path)
        leaf = path.rsplit(".", 1)[-1]
        if kind == "date" and parsed not in (None, ""):
            # Any layout the pipeline can read is fine: the corpus build rewrites
            # every date to MM/DD/YYYY itself (common.canonical.with_output_dates).
            # A date it cannot read is the problem — it would train as written.
            text = str(parsed)
            # A month and year (a form's edition date, "05-17") has no day to
            # write MM/DD/YYYY with; the output formatter keeps it as written, in
            # the target and in serving alike, so it is consistent, not an error.
            if normalize_date(text) is None and not _MONTH_YEAR.match(text):
                flag("date_format", f"{path}: parsed {text!r} is not a readable date")
            elif leaf in ("effective_date", "expiration_date"):
                dates[path[: len(path) - len(leaf)]][leaf] = text
        if kind == "currency" and parsed not in (None, "") and raw not in (None, ""):
            # A field named like an amount that holds a word on both sides is an
            # indicator ("Additional" / "Return" premium), not a misread figure.
            if not any(ch.isdigit() for ch in f"{raw}{parsed}"):
                continue
            # A split limit is several amounts in one field; it has no single number.
            if len(_FIGURES.findall(str(parsed))) >= 2:
                continue
            try:
                number = float(str(parsed).replace(",", ""))
            except ValueError:
                flag("amount", f"{path}: parsed {parsed!r} is not a number")
                continue
            printed = normalize_currency(raw)
            # Magnitudes: a return or credit is printed "$109.00" and recorded -109.0.
            if printed is not None and abs(abs(float(printed)) - abs(number)) > 0.005:
                flag("amount", f"{path}: raw {raw!r} and parsed {parsed!r} disagree")
    for parent, pair in dates.items():
        start, end = (normalize_date(pair.get("effective_date")),
                      normalize_date(pair.get("expiration_date")))
        if start and end and end <= start:
            flag("period", f"{parent or '(top)'}: expiration {pair['expiration_date']} is not after "
                           f"effective {pair['effective_date']}")


# --------------------------------------------------------------------------
# The folder
# --------------------------------------------------------------------------


def audit_folder(root: Path, *, lines: frozenset[str] | None = None, seed: int = 42) -> AuditReport:
    report = AuditReport(root=str(root), lines=lines)
    checksums: dict[str, str] = {}
    report.findings.extend(misplaced(root))
    folders = document_folders(root)
    for index, folder in enumerate(folders, 1):
        report.documents.append(audit_document(folder, report, checksums))
        if index % 100 == 0:
            log.info("audited %d of %d folders", index, len(folders))
    report.findings.extend(delivered_split_findings(report.documents))
    report.findings.extend(output_shape_findings(report.documents))

    by_line: dict[str, list[str]] = defaultdict(list)
    for doc in report.documents:
        by_line[doc.line or "(no line)"].append(doc.folder)
    rng = random.Random(seed)
    for _line, names in sorted(by_line.items()):
        count = max(1, round(len(names) * SPOT_CHECK_SHARE))
        report.spot_check += sorted(rng.sample(sorted(names), min(count, len(names))))
    return report


def output_shape_findings(documents: list[DocumentAudit]) -> list[Finding]:
    """A blocker when the folder's single-line policies would teach the adapter
    two output shapes: lines on the SPEC_21 common model beside lines on
    self-contained schema files (``common.adapter.assert_one_output_shape``).

    Checked here, before import, because it is the last easy place to leave
    such documents out (``prepare_bundles --lines``): once imported, the corpus
    build takes every policy, and training refuses every cycle of the tenant.
    """
    from common.adapter import AdapterError, assert_one_output_shape
    from common.lob import known_lines

    lines = sorted({doc.line for doc in documents if doc.line and doc.line in known_lines()})
    try:
        assert_one_output_shape(lines)
    except AdapterError as exc:
        return [Finding(".", "blocker", "output shape",
                        f"{exc} Leave them out before import: prepare_bundles --lines ...")]
    return []


def delivered_split_findings(documents: list[DocumentAudit]) -> list[Finding]:
    """Blockers for a delivered split the corpus build would refuse.

    All or none of the documents carry one, and a source document with its twins
    sits in one split — otherwise a test document's layout was trained on.
    """
    carrying = [d for d in documents if d.split]
    if not carrying:
        return []
    found = [Finding(d.folder, "blocker", "split", "no split in metadata.json, but other documents have one")
             for d in documents if not d.split]
    splits_of: dict[str, set[str]] = defaultdict(set)
    for d in carrying:
        splits_of[d.template_id or d.folder].add(d.split)
    for d in carrying:
        splits = splits_of[d.template_id or d.folder]
        if len(splits) > 1:
            found.append(Finding(d.folder, "blocker", "split",
                                 f"source {d.template_id} is in {sorted(splits)}: a source and its twins "
                                 "must share one split"))
    return found


def summary(report: AuditReport) -> dict[str, Any]:
    from common.constants import split_ratio_for
    from data_pipeline.dataset_builder.split_groups import MIN_DOCS_TO_MEASURE_LINE
    from evaluation.freeze_eval_set import MIN_FROZEN_DOCS_PER_TYPE

    docs = report.documents
    importable = [d for d in docs if d.importable]
    results = Counter(v.result for v in report.values)
    lines = Counter(d.line or "(no line)" for d in importable)
    ratio = split_ratio_for(len(importable))
    expected_test = round(len(importable) * ratio.test)
    delivered = Counter((d.split, "synthetic" if d.synthetic else "real") for d in importable if d.split)
    if delivered:
        # The delivery's own split is what the corpus build will use, so report it
        # rather than what a drawn split would have produced.
        expected_test = sum(n for (split, _), n in delivered.items() if split == "test")
    buckets = Counter(
        "1-5" if d.pages <= 5 else "6-20" if d.pages <= 20 else "21-50" if d.pages <= 50
        else "51-100" if d.pages <= 100 else ">100"
        for d in importable
    )
    checked = sum(results[k] for k in ("ok", "wrong_page", "not_found", "no_page_ref"))
    return {
        "folders": len(docs),
        "importable": len(importable),
        "blockers": len(report.blockers),
        "documents_with_blockers": len({f.folder for f in report.blockers}),
        "digital": sum(1 for d in importable if d.digital),
        "scanned": sum(1 for d in importable if not d.digital),
        "values": dict(results),
        "values_found_on_their_page": round(results["ok"] / checked, 4) if checked else None,
        "format_findings": dict(Counter(f.check for f in report.formats)),
        "documents_per_line": dict(sorted(lines.items())),
        "lines_too_small_to_measure": sorted(
            line for line, n in lines.items() if n < MIN_DOCS_TO_MEASURE_LINE),
        "page_count_spread": {k: buckets[k] for k in ("1-5", "6-20", "21-50", "51-100", ">100")},
        "split_ratio": {"train": ratio.train, "val": ratio.val, "test": ratio.test},
        "delivered_split": {f"{s}/{k}": n for (s, k), n in sorted(delivered.items())} or None,
        "expected_test_documents": expected_test,
        "test_meets_freeze_minimum": expected_test >= MIN_FROZEN_DOCS_PER_TYPE,
        "spot_check_documents": len(report.spot_check),
    }


def write_report(report: AuditReport, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    facts = summary(report)

    def table(name: str, header: list[str], rows) -> None:
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    table("blockers.csv", ["folder", "check", "detail"],
          [(f.folder, f.check, f.detail) for f in report.blockers])
    table("warnings.csv", ["folder", "check", "detail"],
          [(f.folder, f.check, f.detail) for f in report.findings if f.severity != "blocker"])
    table("value_checks.csv", ["folder", "field", "raw", "page_ref", "result", "found_on"],
          [(v.folder, v.path, v.raw, v.page_ref, v.result, v.found_on)
           for v in report.values if v.result not in ("ok", "skipped_short")])
    table("formats.csv", ["folder", "check", "detail"],
          [(f.folder, f.check, f.detail) for f in report.formats])
    table("spot_check.csv", ["folder"], [(name,) for name in report.spot_check])
    (out / "summary.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render(report, facts), encoding="utf-8")
    return facts


def render(report: AuditReport, facts: dict[str, Any]) -> str:
    values = facts["values"]
    lines = [
        f"# Data audit: {report.root}",
        "",
        f"Lines: {', '.join(sorted(report.lines)) if report.lines else 'every line'} · "
        f"folders: {facts['folders']} · "
        f"importable: {facts['importable']} · digital {facts['digital']} / scanned {facts['scanned']}",
        "",
        f"## Blockers: {facts['blockers']} in {facts['documents_with_blockers']} document(s)",
        "Must be fixed before import — see blockers.csv.",
    ]
    for check, count in Counter(f.check for f in report.blockers).most_common():
        lines.append(f"- {check}: {count}")
    lines += [
        "",
        "## Values against the PDF",
        f"- found on their page: {values.get('ok', 0)}"
        + (f" ({facts['values_found_on_their_page']:.1%} of checkable)"
           if facts["values_found_on_their_page"] is not None else ""),
        f"- on another page (page_ref wrong): {values.get('wrong_page', 0)}",
        f"- not in the document (check the value): {values.get('not_found', 0)}",
        f"- no page_ref: {values.get('no_page_ref', 0)}",
        f"- on scanned pages, to check after OCR: {values.get('needs_ocr', 0)}",
        f"- too short to check: {values.get('skipped_short', 0)}",
        "Every non-ok value is listed in value_checks.csv.",
        "",
        "## Formats",
    ]
    lines += [f"- {k}: {v}" for k, v in facts["format_findings"].items()] or ["- none"]
    lines += ["", "## Totals", "| Line | Documents |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in facts["documents_per_line"].items()]
    lines += [
        "",
        (f"- delivered split (used as given): {facts['delivered_split']}" if facts["delivered_split"]
         else f"- split at this volume: {facts['split_ratio']}"),
        "- expected test documents: "
        f"{facts['expected_test_documents']} "
        f"({'enough' if facts['test_meets_freeze_minimum'] else 'BELOW'} the 150 needed to freeze)",
        f"- lines too small to measure (<5): {facts['lines_too_small_to_measure'] or 'none'}",
        f"- page counts: {facts['page_count_spread']}",
        "",
        f"## Spot check: {facts['spot_check_documents']} document(s) in spot_check.csv",
        "Verify these by hand against the PDF — the only check that values are right.",
        "",
        "> The CSVs quote label values: they hold personal data. Keep them on this machine.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a labeled-PDF training folder (read-only)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lines", default=None,
                        help="comma-separated lines documents must be of; every line by default")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.audit", argv):
        return 0
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.input.is_dir():
        print(f"no folder at {args.input}", file=sys.stderr)
        return 2
    from common.lob import LobError, parse_lines

    try:
        lines = parse_lines(args.lines)
    except LobError as exc:
        print(exc, file=sys.stderr)
        return 2
    report = audit_folder(args.input, lines=lines, seed=args.seed)

    facts = write_report(report, args.out)
    print((args.out / "report.md").read_text(encoding="utf-8"))
    print(f"report and CSVs in {args.out}")
    return 1 if facts["blockers"] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
