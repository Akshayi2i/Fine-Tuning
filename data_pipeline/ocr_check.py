"""After OCR: is every labelled value printed on the page the label says?

The pre-upload audit (``data_pipeline/audit.py``) compares values with a PDF's
own text layer; a scan has none, so it can only mark them "needs OCR". Once
MinerU has run on the pod, every page has text in ``processed/``, and this
repeats the comparison there — with the audit's own matching (case, punctuation
and line breaks ignored; a multi-word value may be split across columns).

It answers the question the synthetic labels leave open. Most of each synthetic
label was added from its source's reviewed gold (``fill_synthetic_labels``,
recorded under ``fideon:filled``), by rules rather than by reading the page. So
the report keeps the **added** fields apart from the **original** ones, and the
gate is relative: added fields must be found on their page about as often as
the fields the generator itself placed, on the same OCR. A gap means the rules
put values on the synthetic pages that are not there - training must wait.

    python -m data_pipeline.ocr_check --doc-type policy                 # report, gate
    python -m data_pipeline.ocr_check --doc-type policy --max-gap 0.05

Results per value: ``ok`` (on a page it cites), ``wrong_page``, ``not_found``,
``no_page_ref`` (found, cites no page), ``skipped_short`` (under 3 characters).
The CSVs quote label values: they stay on the pod's volume, never in git.

**Every document is held to a floor of its own**, not only the batch to one: a
few scans MinerU returned blank are 1 or 2 documents in 84, and the batch's found
rate hid them. A document fails when none of its pages has text, when too many
of its pages have none (:data:`MAX_BLANK_PAGE_SHARE`), or when too few of its
values are found on a page they cite - by the kind of document
(:data:`MIN_DOC_FOUND_RATE`: a hybrid PDF's labels were audited against its own
text layer and should be found, a scan's only against this OCR). The floors are
starting values, to be read against the first real OCR output;
``--min-doc-found`` sets them. Each failing document is named, with the way out:
read it, correct its gold, or leave it out (``<input>/exclusions.csv``). The
values for a person to look at - found on no page, or only on another - are in
``needs_review.csv``, real documents only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from data_pipeline.audit import MIN_CHECKABLE_CHARS, _appears, _Page, _refs, iter_envelopes, normalise

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("/workspace/ocr_check")
#: How much lower the added fields' found rate may be than the original fields'.
DEFAULT_MAX_GAP = 0.05
#: Below this share of ALL labelled values found on a page they cite, the OCR
#: text itself is suspect - empty pages, a wrong page numbering - whatever the
#: added-versus-original comparison says. Deliberately low: it is there to
#: catch OCR that did not work, not to grade labels.
MIN_FOUND_RATE = 0.50
CHECKED = ("ok", "wrong_page", "not_found", "no_page_ref")
#: The share of a document's labelled values found on a page they cite, below
#: which the document is refused, by kind (``document_kind`` of its label
#: metadata). Starting values, not measurements: lower for a scan, whose labels
#: were never audited against any text but this OCR. ``None``: kind not recorded.
MIN_DOC_FOUND_RATE: dict[str | None, float] = {"hybrid": 0.60, "scanned": 0.40, None: 0.40}
#: The share of a document's pages MinerU may return with no text. A scan
#: has blank pages (a blank back); a document of mostly blank pages was not read.
MAX_BLANK_PAGE_SHARE = 0.30


@dataclass
class Row:
    source_id: str
    path: str
    raw: str
    page_ref: str
    result: str
    found_on: str
    origin: str          # "added" (fideon:filled) or "original"
    kind: str            # "real" or "synthetic"
    line: str


@dataclass
class DocumentFacts:
    """One document, as its OCR and its values came out."""

    source_id: str
    line: str
    kind: str                       # "real" or "synthetic"
    document_kind: str | None       # "scanned" or "hybrid" (label metadata), when recorded
    pages: int
    blank_pages: int                # pages MinerU returned with no text
    values: int                     # checkable values
    found_rate: float | None


@dataclass
class OcrCheckReport:
    rows: list[Row] = field(default_factory=list)
    documents: int = 0
    skipped: Counter = field(default_factory=Counter)
    details: list[DocumentFacts] = field(default_factory=list)

    def rate(self, rows: list[Row]) -> float | None:
        """Share of checkable values found on a page they cite."""
        checked = [r for r in rows if r.result in CHECKED]
        return round(sum(r.result == "ok" for r in checked) / len(checked), 4) if checked else None

    def by(self, key) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[Row]] = defaultdict(list)
        for row in self.rows:
            groups[key(row)].append(row)
        return {name: {"values": len(rows), "found_rate": self.rate(rows),
                       "results": dict(Counter(r.result for r in rows))}
                for name, rows in sorted(groups.items())}

    def worst_documents(self, limit: int = 25) -> list[dict[str, Any]]:
        per: dict[str, list[Row]] = defaultdict(list)
        for row in self.rows:
            per[row.source_id].append(row)
        scored = [(self.rate(rows), sid, rows) for sid, rows in per.items() if self.rate(rows) is not None]
        return [{"source_id": sid, "found_rate": rate, "values": len(rows), "kind": rows[0].kind,
                 "line": rows[0].line}
                for rate, sid, rows in sorted(scored, key=lambda t: t[0])[:limit]]


def check_document(source_id: str, label: dict, page_texts: list[str], *, kind: str, line: str) -> list[Row]:
    """Every value of one label against its pages' OCR text."""
    filled = set(((label.get("fideon:filled") or {}).get("paths")) or [])
    pages = [_Page(text) for text in page_texts]
    has_text = [bool(text.strip()) for text in page_texts]
    rows = []
    for path, envelope in iter_envelopes(label):
        raw = envelope.get("raw")
        if raw in (None, "") or isinstance(raw, (dict, list)):
            continue
        raw = str(raw)
        refs = [r for r in _refs(envelope) if isinstance(r, int) and 1 <= r <= len(pages)]
        origin = "added" if path in filled else "original"
        value = normalise(raw)
        found_on: list[int] = []
        if len(value) < MIN_CHECKABLE_CHARS:
            result = "skipped_short"
        else:
            found_on = [i + 1 for i, page in enumerate(pages) if has_text[i] and _appears(value, page)]
            if not refs:
                result = "no_page_ref" if found_on else "not_found"
            elif set(found_on) & set(refs):
                result = "ok"
            elif found_on:
                result = "wrong_page"
            else:
                result = "not_found"
        rows.append(Row(source_id, path, raw, ",".join(map(str, refs)), result,
                        ",".join(map(str, found_on[:5])), origin, kind, line))
    return rows


def document_facts(source_id: str, rows: list[Row], page_texts: list[str], *, kind: str, line: str,
                   document_kind: str | None = None) -> DocumentFacts:
    """What one document's OCR and values came to (see :func:`document_problems`)."""
    checked = [r for r in rows if r.result in CHECKED]
    rate = round(sum(r.result == "ok" for r in checked) / len(checked), 4) if checked else None
    return DocumentFacts(source_id, line, kind, document_kind, len(page_texts),
                         sum(1 for text in page_texts if not text.strip()), len(checked), rate)


def document_problems(details: list[DocumentFacts], floors: dict[str | None, float] | None = None,
                      max_blank_share: float = MAX_BLANK_PAGE_SHARE) -> list[tuple[str, str]]:
    """``(source_id, what is wrong)`` for each document that fails its own floor.

    No page with text; more blank pages than ``max_blank_share`` allows; or a
    found rate under the floor for its kind. A document whose values could not be
    checked at all has no rate: only its pages are judged.
    """
    floors = {**MIN_DOC_FOUND_RATE, **(floors or {})}
    problems = []
    for d in details:
        if d.pages and d.blank_pages == d.pages:
            problems.append((d.source_id, f"none of its {d.pages} page(s) has any OCR text"))
            continue
        if d.pages and d.blank_pages / d.pages > max_blank_share:
            problems.append((d.source_id, f"{d.blank_pages} of its {d.pages} pages have no OCR text "
                                          f"(more than {max_blank_share:.0%})"))
            continue
        floor = floors.get(d.document_kind, floors[None])
        if d.found_rate is not None and d.found_rate < floor:
            problems.append((d.source_id, f"{d.found_rate:.1%} of its {d.values} values are found on a page "
                                          f"they cite (a {d.document_kind or 'document'} needs {floor:.0%})"))
    return problems


def run_check(client: BlobClient, doc_type: str, *, tenant_id: str | None = None,
              limit: int | None = None) -> OcrCheckReport:
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids, load_golden_label

    report = OcrCheckReport()
    for index, source_id in enumerate(list_labeled_source_ids(client, doc_type, tenant_id)):
        if limit is not None and report.documents >= limit:
            break
        meta_key = paths.ocr_meta(doc_type, source_id, tenant_id)
        if not client.exists(meta_key):
            report.skipped["not OCR'd yet"] += 1
            continue
        page_count = int(client.read_json(meta_key).get("page_count") or 0)
        if page_count < 1:
            report.skipped["OCR recorded no pages"] += 1
            continue
        label, metadata = load_golden_label(source_id, doc_type, client, tenant_id)
        texts = []
        for page in range(1, page_count + 1):
            key = paths.processed_page(doc_type, source_id, page, "md", tenant_id)
            texts.append(client.read_text(key) if client.exists(key) else "")
        lob = metadata.get("lob")
        kind = "synthetic" if metadata.get("synthetic") else "real"
        line = "+".join(lob) if isinstance(lob, list) else str(lob or "-")
        rows = check_document(source_id, label, texts, kind=kind, line=line)
        report.rows += rows
        report.details.append(document_facts(source_id, rows, texts, kind=kind, line=line,
                                             document_kind=metadata.get("document_kind")))
        report.documents += 1
        if (index + 1) % 100 == 0:
            log.info("checked %d documents", index + 1)
    return report


def verdict(report: OcrCheckReport, max_gap: float = DEFAULT_MAX_GAP,
            floors: dict[str | None, float] | None = None) -> tuple[bool, str]:
    """Pass when every document holds its floor and added fields are found about
    as often as original ones."""
    problems = document_problems(report.details, floors)
    if problems:
        named = "; ".join(f"{sid}: {what}" for sid, what in problems[:8])
        more = f" (and {len(problems) - 8} more)" if len(problems) > 8 else ""
        return False, (f"{len(problems)} document(s) fail their own floor - {named}{more}. Read each one's OCR "
                       "and its gold, then correct the gold, read the document again, or leave it out "
                       "(<input>/exclusions.csv).")
    # First, did OCR produce text the labels can be found in at all? A batch of
    # original documents has no added fields, so the comparison below had
    # nothing to compare and passed whatever the pages held - empty included.
    overall = report.rate(report.rows)
    if overall is None:
        return False, "no labelled value could be checked against the OCR text"
    if overall < MIN_FOUND_RATE:
        return False, (f"only {overall:.1%} of labelled values are found on a page they cite (under "
                       f"{MIN_FOUND_RATE:.0%}): the OCR text is empty or misnumbered, or the labels "
                       "cite the wrong pages. Read the worst documents before training.")
    synthetic = [r for r in report.rows if r.kind == "synthetic"]
    added = report.rate([r for r in synthetic if r.origin == "added"])
    original = report.rate([r for r in synthetic if r.origin == "original"])
    if added is None:
        return True, f"labelled values found {overall:.1%}; no rule-added fields to compare"
    if original is None:
        return False, "no original synthetic fields to compare the added ones with"
    gap = round(original - added, 4)
    if gap > max_gap:
        return False, (f"added fields are found on their page {added:.1%} of the time against {original:.1%} for "
                       f"the generator's own fields (gap {gap:.1%} > {max_gap:.0%}): the fill-in put values on "
                       "synthetic pages that are not there. Fix the labels before training.")
    return True, (f"added fields found {added:.1%}, original fields {original:.1%} (gap {gap:.1%} "
                  f"within {max_gap:.0%})")


def write_report(report: OcrCheckReport, out: Path, max_gap: float = DEFAULT_MAX_GAP,
                 floors: dict[str | None, float] | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    passed, reason = verdict(report, max_gap, floors)
    summary = {
        "documents": report.documents,
        "skipped": dict(report.skipped),
        "values": len(report.rows),
        "found_rate": report.rate(report.rows),
        "by_origin_synthetic": report.by(lambda r: r.origin if r.kind == "synthetic" else "real documents"),
        "by_kind": report.by(lambda r: r.kind),
        "by_line": report.by(lambda r: r.line),
        "worst_documents": report.worst_documents(),
        "failing_documents": [{"source_id": sid, "problem": what}
                              for sid, what in document_problems(report.details, floors)],
        "blank_pages": sum(d.blank_pages for d in report.details),
        "passed": passed,
        "verdict": reason,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "value_checks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_id", "path", "raw", "page_ref", "result", "found_on", "origin", "kind", "line"])
        writer.writerows([r.source_id, r.path, r.raw, r.page_ref, r.result, r.found_on, r.origin, r.kind, r.line]
                         for r in report.rows)
    # The values for a person to look at: found on no page, or only on another.
    # Real documents only - a synthetic twin's labels are checked as a group.
    with (out / "needs_review.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_id", "line", "path", "raw", "cited_pages", "result", "found_on"])
        writer.writerows([r.source_id, r.line, r.path, r.raw, r.page_ref, r.result, r.found_on]
                         for r in report.rows if r.kind == "real" and r.result in ("not_found", "wrong_page"))
    lines = [f"# Post-OCR value check — {'PASS' if passed else 'FAIL'}", "", reason, "",
             f"- documents checked: {report.documents}; skipped: {dict(report.skipped) or 'none'}",
             f"- values: {len(report.rows)}; found on a cited page: {summary['found_rate']}", "",
             "| Group | Values | Found on its page |", "|---|---|---|"]
    for group, facts in summary["by_origin_synthetic"].items():
        lines.append(f"| {group} | {facts['values']} | {facts['found_rate']} |")
    lines += ["", "| Line | Values | Found on its page |", "|---|---|---|"]
    lines += [f"| {k} | {v['values']} | {v['found_rate']} |" for k, v in summary["by_line"].items()]
    lines += ["", "Worst documents: summary.json `worst_documents`; every value: value_checks.csv.",
              "", "> The CSV quotes label values (personal data). Keep it on the pod's volume."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from artifact_registry.blob_client import for_ocr
    from common.constants import ACTIVE_DOC_TYPES

    parser = argparse.ArgumentParser(description="Check every labelled value against its page's OCR text")
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP,
                        help="how much lower the added fields' found rate may be than the original fields'")
    parser.add_argument("--limit", type=int, default=None, help="check only the first N documents")
    parser.add_argument("--min-doc-found", type=float, default=None, metavar="RATE",
                        help="every document's floor for the share of its values found on a page they cite "
                             "(default by kind: hybrid 0.60, scanned 0.40)")
    args = parser.parse_args(argv)
    floors = {kind: args.min_doc_found for kind in MIN_DOC_FOUND_RATE} if args.min_doc_found is not None else None
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.ocr_check", argv):
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run_check(for_ocr(), args.doc_type, tenant_id=args.tenant, limit=args.limit)
    summary = write_report(report, args.out, args.max_gap, floors)
    print((args.out / "report.md").read_text(encoding="utf-8"))
    print(f"report and CSV in {args.out}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
