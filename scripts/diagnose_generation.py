"""Show what vLLM actually returns for validation rows, and why a row fails to parse.

    python scripts/diagnose_generation.py --checkpoint <dir> [--source policy_0034] [--rows 8]
    python scripts/diagnose_generation.py --checkpoint <dir> --against-base

``--against-base`` scores the untouched base model and the checkpoint on every
validation row, the way checkpoint selection scores, and prints them side by
side: whether fine-tuning beat the model it started from.

Checkpoint selection logs only the parse error ("Unterminated string ..."), which
cannot tell a model that loops from a decoder that stops mid-value. This runs a
few validation rows through the same engine and records, per row and per mode
(schema-constrained, as scoring runs, and unconstrained), how generation ended,
how long the answer is, and its first and last characters. Needs the GPU to
itself: stop any running pipeline first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _describe(result, schema_used: bool) -> dict:
    if isinstance(result, BaseException):
        return {"constrained": schema_used, "error": f"{type(result).__name__}: {result}"}
    try:
        json.loads(result.text)
        parse = "ok"
    except ValueError as exc:
        parse = str(exc)
    return {
        "constrained": schema_used,
        "finish_reason": result.finish_reason,
        "tokens": len(result.tokens),
        "chars": len(result.text),
        "parse": parse,
        "head": result.text[:160],
        "tail": result.text[-240:],
        "last_tokens": result.tokens[-6:],
    }


def main(argv: list[str] | None = None) -> int:
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from common.schemas import resolved_schema, with_page_bounds
    from evaluation.validation_generation import split_prompt
    from inference_core.input_builder import page_total
    from inference_core.model_runner import generate_batch, load_model, release_model
    from training.stage_data import localize_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None, help="adapter directory to apply")
    parser.add_argument("--corpus", default="v0")
    parser.add_argument("--tenant", default="smoke")
    parser.add_argument("--source", default=None, help="only rows of this source_id")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--out", default=None, help="report path (default under /workspace/logs)")
    parser.add_argument("--against-base", dest="against_base", action="store_true",
                        help="score base and checkpoint on every validation row")
    parser.add_argument("--rescore", default=None, metavar="REPORT",
                        help="score a saved --against-base report again, without the GPU")
    parser.add_argument("--show-lists", dest="show_lists", default=None, metavar="REPORT",
                        help="print expected vs written table rows from a saved report")
    parser.add_argument("--errors", default=None, metavar="REPORT",
                        help="count every error type per model from a saved report")
    parser.add_argument("--audit-labels", dest="audit_labels", default=None, metavar="REPORT",
                        help="where gold and the fine-tuned model disagree, which one the page "
                             "text supports; writes a CSV to check against the PDFs")
    parser.add_argument("--document", default=None, metavar="SOURCE_ID",
                        help="with --report: gold vs base vs fine-tuned for one document "
                             "('list' shows the documents)")
    parser.add_argument("--report", default=None,
                        help="saved comparison (default: the one for --split)")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="validation rows (default) or the held-out test rows")
    parser.add_argument("--build-document", dest="build_document", default=None,
                        metavar="SOURCE_ID",
                        help="with --against-base: build this one document's rows from Blob "
                             "(label, OCR, page images) instead of reading a split - for a "
                             "document in any split, training included")
    parser.add_argument("--mode", default="ocr_plus_image",
                        choices=["ocr_plus_image", "noisy_ocr_image", "image_only"])
    args = parser.parse_args(argv)
    if args.audit_labels:
        return _print_audit(args)
    args.report = args.report or (
        _document_report_path(args.build_document) if args.build_document
        else _report_path(args.split))
    if args.document:
        return _print_document(args)
    if args.errors:
        breakdown = error_breakdown(args.errors)
        labels = list(breakdown)
        print(f"{'':<40}" + "".join(f"{label:>14}" for label in labels))
        keys = sorted({k for b in breakdown.values() for k in b["stats"]},
                      key=lambda k: ("table" in k, "unusable" in k or "lost" in k, k))
        for key in keys:
            print(f"{key:<40}" + "".join(f"{breakdown[x]['stats'].get(key, 0):>14}" for x in labels))
        print("errors by type:")
        for kind in ERROR_TYPES:
            print(f"  {kind:<38}" + "".join(f"{breakdown[x]['errors'][kind]:>14}" for x in labels))
        for label in labels:
            print(f"most affected fields ({label}):")
            for kind in ERROR_TYPES:
                top_fields = ", ".join(f"{f} ({n})" for f, n in breakdown[label]["top_fields"][kind])
                if top_fields:
                    print(f"  {kind}: {top_fields}")
        return 0
    if args.show_lists:
        for line in list_details(args.show_lists):
            print(line)
        return 0
    if args.rescore:
        print_comparison(rescore(args.rescore))
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --rescore is given")

    # On the pod, run detached in tmux: loading vLLM takes minutes.
    from orchestration.detach import detach_script_if_needed

    if detach_script_if_needed(__file__, argv, "diagnose"):
        return 0

    client = BlobClient()
    rows = (_document_rows(client, args.build_document, args.tenant) if args.build_document
            else _eval_rows(client, args))
    if args.source:
        rows = [r for r in rows if r.get("source_id") == args.source]
    if not args.against_base:
        rows = rows[: args.rows]
    if not rows:
        print("no validation rows matched", file=sys.stderr)
        return 1
    rows = localize_rows(rows, client, paths.staging_train_images_dir(args.corpus, args.tenant))
    if args.against_base:
        return _against_base(rows, client, args)
    args.out = args.out or "/workspace/logs/diagnose_generation.json"

    prompts = [split_prompt(row)[0] for row in rows]
    schemas = [with_page_bounds(
                   resolved_schema(r["doc_type"], r.get("acord_form"), r.get("lob"), r.get("sections")),
                   page_total(prompt))
               for r, prompt in zip(rows, prompts, strict=True)]
    model = load_model("base", client)
    report = []
    try:
        constrained = generate_batch(model, list(zip(prompts, schemas, strict=True)),
                                     adapter=args.checkpoint)
        free = generate_batch(model, [(p, None) for p in prompts], adapter=args.checkpoint)
        for row, c, f in zip(rows, constrained, free, strict=True):
            report.append({
                "source_id": row.get("source_id"),
                "sections": row.get("sections"),
                "lob": row.get("lob"),
                "modality_mode": row.get("modality_mode"),
                "with_schema": _describe(c, True),
                "without_schema": _describe(f, False),
            })
    finally:
        release_model(model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for entry in report:
        for mode in ("with_schema", "without_schema"):
            d = entry[mode]
            print(f"{entry['source_id']} {entry['sections'] or '-':<24} {mode:<15} "
                  f"finish={d.get('finish_reason')} tokens={d.get('tokens')} "
                  f"parse={'ok' if d.get('parse') == 'ok' else 'FAIL'}")
    print(f"full report: {args.out}")
    return 0


def _saved(generation) -> dict:
    """One generation as the report keeps it: the row's scoring metadata, the
    label, the answer. Page images are kept as a count, which is all scoring reads."""
    row = generation.row
    images = sum(1 for m in row.get("messages", []) if isinstance(m.get("content"), list)
                 for part in m["content"] if isinstance(part, dict) and part.get("type") == "image")
    slim = {k: row.get(k) for k in ("source_id", "doc_type", "acord_form", "lob", "sections",
                                    "modality_mode", "is_scanned")}
    # Images as a count; the OCR text the prompt carried, kept, so a rescore can
    # measure hallucination (a value against the text that was sent).
    text = generation.page_text
    slim["messages"] = [{"role": "user", "content": [{"type": "image"}] * images
                         + ([{"type": "text", "text": text}] if text else [])}]
    return {"row": slim, "golden": generation.golden, "extraction": generation.extraction,
            "error": generation.error, "failure_kind": generation.failure_kind}


def _scores(generations, label: str) -> dict:
    """The gate's numbers for one model, plus the breakdowns the gate does not read
    (accuracy per line of business, weakest fields)."""
    from evaluation.run_eval import build_report

    report = build_report(label, [(g.golden, g.extraction or {}, g.metadata) for g in generations])
    numeric = {k: v for k, v in report.gate_metrics().items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    details: dict = {}
    for subset in report.full_set():
        for key in ("field_accuracy_by_lob", "weakest_fields"):
            if subset.metrics.get(key):
                details[key] = subset.metrics[key]
    return {"metrics": numeric, "details": details,
            "unusable_json": sum(g.failure_kind == "output" for g in generations),
            "setup_failures": sum(bool(g.error) and g.failure_kind != "output" for g in generations),
            "rows": len(generations)}


def rescore(report_path: str) -> dict[str, dict]:
    """Score a saved comparison again with the current scorer. No GPU."""
    from evaluation.validation_generation import ValidationGeneration

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return {label: _scores([ValidationGeneration(**saved) for saved in entry["generations"]], label)
            for label, entry in report.items() if entry.get("generations")}


def list_details(report_path: str, limit: int = 6) -> list[str]:
    """For each table in a saved report: what the label holds and what each model wrote.

    One line per (model, row, table): the line of business, the identifier the
    scorer matched on, and the first identifiers on each side. No GPU.
    """
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import _at, _infer_key_fields, find_list_fields

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    lines: list[str] = []
    for label, entry in report.items():
        shown = 0
        for saved in entry.get("generations", []):
            expected = values_view(saved.get("golden") or {})
            got = values_view(saved.get("extraction") or {})
            for path, rows in find_list_fields(expected).items():
                keys = _infer_key_fields(rows)
                got_rows = _at(got, path) or []
                def ident(r, keys=keys):
                    return tuple(r.get(k) for k in keys) if isinstance(r, dict) else r

                lines.append(
                    f"{label:<10} {saved['row'].get('source_id')} lob={saved['row'].get('lob')} "
                    f"{path} key={keys}: expected {len(rows)} {[ident(r) for r in rows[:3]]} "
                    f"| wrote {len(got_rows)} {[ident(r) for r in got_rows[:3]]}")
                shown += 1
                if shown >= limit:
                    break
            if shown >= limit:
                break
    return lines


def _merge(into: dict, part: dict) -> dict:
    """One document's answer from its windows: each window writes its own sections."""
    for key, value in (part or {}).items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        elif key not in into:
            into[key] = value
    return into


#: Field outcomes in a document comparison, worst first.
OUTCOMES = ("missing (not written)", "written as null", "wrong value", "invented (not in gold)",
            "correct")


def _outcome(path: str, gold: dict, got: dict) -> str | None:
    from common.normalize import values_match
    from evaluation.metrics.extraction_faults import _is_empty

    in_gold = path in gold and not _is_empty(gold[path])
    if path not in got:
        return "missing (not written)" if in_gold else None
    if _is_empty(got[path]):
        return "written as null" if in_gold else None
    if not in_gold:
        return "invented (not in gold)"
    return "correct" if values_match(gold[path], got[path], field_path=path) else "wrong value"


def document_comparison(report_path: str, source_id: str, mode: str = "ocr_plus_image") -> dict:
    """Gold against each model's answer for ONE document, field by field. No GPU.

    The document's windows are merged back into one answer per model. Each gold
    field is marked correct, wrong value, missing (the model did not write the
    field at all) or written as null (it wrote the field, empty); a field the
    model wrote that gold does not have is marked invented. Table rows are
    compared by position; the table totals use the scorer's own row matching.
    """
    from collections import Counter

    from common.canonical import values_view, without_system_fields
    from evaluation.metrics.field_accuracy import flatten_scalars, score_all_list_fields, score_fields

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    gold: dict = {}
    answers: dict[str, dict] = {}
    unusable: Counter = Counter()
    for label, entry in report.items():
        answer: dict = {}
        for saved in entry.get("generations", []):
            row = saved["row"]
            if row.get("source_id") != source_id or row.get("modality_mode") != mode:
                continue
            if label == "base":
                _merge(gold, saved.get("golden") or {})
            if saved.get("error"):
                unusable[label] += 1
            _merge(answer, saved.get("extraction") or {})
        answers[label] = answer
    if not gold:
        raise SystemExit(f"no {mode} rows for {source_id} in {report_path}")
    # Scored as the evaluation scores: the system-supplied fields count nothing.
    gold = without_system_fields(gold)
    answers = {label: without_system_fields(answer) for label, answer in answers.items()}

    gold_flat = flatten_scalars(gold)
    flats = {label: flatten_scalars(answer) for label, answer in answers.items()}
    paths = sorted(set(gold_flat).union(*flats.values()))
    rows, summary = [], {}
    for label, flat in flats.items():
        outcomes = Counter(o for o in (_outcome(p, gold_flat, flat) for p in paths) if o)
        accuracy = score_fields(gold, answers[label])
        tables = score_all_list_fields(values_view(gold), values_view(answers[label]))
        summary[label] = {
            "accuracy": accuracy.normalized_match if accuracy.total else None,
            "fields_correct": sum(r.correct for r in accuracy.results),
            "fields_scored": accuracy.total,
            "outcomes": {o: outcomes.get(o, 0) for o in OUTCOMES},
            "table_rows": {name: f"{t.matched_rows} of {t.expected_rows} found, {t.got_rows} written"
                           for name, t in tables.items()},
            "unusable_windows": unusable.get(label, 0),
        }
    for path in paths:
        rows.append({"field": path, "gold": gold_flat.get(path),
                     **{f"{label}": flats[label].get(path) for label in flats},
                     **{f"{label} outcome": _outcome(path, gold_flat, flats[label]) or ""
                        for label in flats}})
    return {"source_id": source_id, "mode": mode, "summary": summary, "fields": rows,
            "gold": gold, "answers": answers}


#: Label-audit findings, most actionable first.
AUDIT_KINDS = (
    "gold likely missing it (model's value is on the page)",
    "gold value not in the page text (check the label)",
    "model likely invented it (value not on the page)",
)


def _text_of(saved: dict) -> str:
    return "\n".join(
        part.get("text", "") for message in saved["row"].get("messages", [])
        if isinstance(message.get("content"), list)
        for part in message["content"] if isinstance(part, dict) and part.get("type") == "text")


def _on_page(value, haystack: str) -> bool:
    """Whether a printed value appears in the OCR text, spacing and case ignored."""
    from common.normalize import normalize_text

    needle = normalize_text(value)
    return bool(needle) and needle in haystack


def label_audit(report_path: str, label: str = "checkpoint", mode: str = "ocr_plus_image") -> list[dict]:
    """Where the gold label and the model disagree, which side the page supports. No GPU.

    Per document (its windows merged), from the clean-OCR rows:

    - a value the model wrote that the gold does not have, printed in the OCR
      text of the pages sent: the gold is probably missing it;
    - the same, not in the text: the model probably invented it;
    - a gold value not in the text of the pages sent: a label to check (or a
      value OCR missed - the page image decides).

    Values are compared as printed (``raw``), with spacing and case ignored.
    Needs a report saved with the OCR text (every --against-base run since the
    hallucination measure was added).
    """
    from common.canonical import printed_view, without_system_fields
    from common.normalize import normalize_text
    from evaluation.metrics.extraction_faults import _is_empty
    from evaluation.metrics.field_accuracy import flatten_scalars

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    by_doc: dict[str, dict] = {}
    for saved in report.get(label, {}).get("generations", []):
        row = saved["row"]
        if row.get("modality_mode") != mode:
            continue
        doc = by_doc.setdefault(row["source_id"], {"gold": {}, "answer": {}, "text": []})
        _merge(doc["gold"], saved.get("golden") or {})
        _merge(doc["answer"], saved.get("extraction") or {})
        doc["text"].append(_text_of(saved))

    findings: list[dict] = []
    for source_id, doc in sorted(by_doc.items()):
        haystack = normalize_text("\n".join(doc["text"])) or ""
        if not haystack:
            continue
        gold = flatten_scalars(printed_view(without_system_fields(doc["gold"])))
        got = flatten_scalars(printed_view(without_system_fields(doc["answer"])))
        for path in sorted(set(gold) | set(got)):
            g, m = gold.get(path), got.get(path)
            if _is_empty(g) and not _is_empty(m):
                kind = AUDIT_KINDS[0] if _on_page(m, haystack) else AUDIT_KINDS[2]
            elif not _is_empty(g) and not _on_page(g, haystack):
                kind = AUDIT_KINDS[1]
            else:
                continue
            findings.append({"source_id": source_id, "field": path, "finding": kind,
                             "gold": "" if g is None else g, "model": "" if m is None else m})
    return findings


#: Error types, in the order they are reported.
ERROR_TYPES = ("left empty", "wrong value", "value from another field", "misread (near miss)",
               "invented (not in the label)")


def error_breakdown(report_path: str, top: int = 3) -> dict[str, dict]:
    """Every wrong field in a saved report, by type, per model. No GPU.

    Scored as checkpoint selection scores (``score_fields``), classified by the
    gate's own ``classify_error``, with its catch-all split in two: a value that
    belongs to another field of the same document, and a plain wrong value.
    Answers that were not usable JSON are counted apart, since every field in
    them is lost to the same cause.
    """
    from collections import Counter

    from common.canonical import values_view, without_system_fields
    from evaluation.metrics.field_accuracy import (
        _at,
        _infer_key_fields,
        _row_key,
        find_list_fields,
        flatten_scalars,
        score_fields,
    )
    from training.vit_gate import classify_error

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for label, entry in report.items():
        counts: Counter = Counter()
        fields: dict[str, Counter] = {t: Counter() for t in ERROR_TYPES}
        stats = Counter()
        for saved in entry.get("generations", []):
            stats["rows"] += 1
            golden = without_system_fields(saved.get("golden") or {})
            got = without_system_fields(saved.get("extraction") or {})
            if saved.get("error"):
                stats["unusable answers"] += 1
                stats["fields lost in unusable answers"] += len(flatten_scalars(golden))
                continue
            accuracy = score_fields(golden, got)
            stats["fields scored"] += accuracy.total
            stats["fields correct"] += sum(r.correct for r in accuracy.results)
            others = {str(v).strip() for v in flatten_scalars(golden).values() if v is not None}
            for r in accuracy.failures():
                kind = classify_error(r.expected, r.got, all_expected=golden)
                if kind == "omission":
                    name = "left empty"
                elif kind in ("invented", "unknown"):
                    name = "invented (not in the label)"
                elif kind == "perception":
                    name = "misread (near miss)"
                elif str(r.got).strip() in others:
                    name = "value from another field"
                else:
                    name = "wrong value"
                counts[name] += 1
                fields[name][r.field_path.split("[")[0]] += 1
            expected_v, got_v = values_view(golden), values_view(got)
            for path, rows in find_list_fields(expected_v).items():
                keys = _infer_key_fields(rows)
                got_rows = [r for r in (_at(got_v, path) or []) if isinstance(r, dict)]
                expected_keys = Counter(_row_key(r, keys) for r in rows)
                got_keys = Counter(_row_key(r, keys) for r in got_rows)
                found = sum((expected_keys & got_keys).values())
                stats["table rows expected"] += len(rows)
                stats["table rows found"] += found
                stats["table rows missed"] += len(rows) - found
                stats["table rows written with no/unknown ID"] += len(got_rows) - found
        out[label] = {"stats": dict(stats), "errors": {t: counts[t] for t in ERROR_TYPES},
                      "top_fields": {t: fields[t].most_common(top) for t in ERROR_TYPES}}
    return out


def _document_rows(client, source_id: str, tenant: str) -> list[dict]:
    """One document's rows in all three reading modes, built as the dataset build
    builds them: the stored label, the MinerU text of every page, the page images.

    For a document outside the validation and test files - a training document
    included. The fine-tuned model has then seen this document and its answer in
    training, so its score measures recall of training data, not extraction of a
    new document; the base model's does not change.
    """
    from artifact_registry import paths
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from data_pipeline.labeling.export_golden_labels import load_golden_label

    label, metadata = load_golden_label(source_id, "policy", client, tenant)
    page_count = int(client.read_json(paths.ocr_meta("policy", source_id, tenant)).get("page_count") or 0)
    if page_count < 1:
        raise SystemExit(f"{source_id} has no OCR'd pages")
    document = SourceDocument(
        source_id=source_id, doc_type="policy", golden_label=label,
        ocr_pages=[client.read_text(paths.processed_page("policy", source_id, page, "md", tenant))
                   for page in range(1, page_count + 1)],
        image_paths=[paths.processed_page("policy", source_id, page, "png", tenant)
                     for page in range(1, page_count + 1)],
        acord_form=metadata.get("acord_form"), lob=metadata.get("lob"), tenant_id=tenant,
        field_provenance=metadata.get("field_provenance", {}),
        synthetic=bool(metadata.get("synthetic", False)),
        delivered_split=metadata.get("split"), template_id=metadata.get("template_id"),
    )
    # "val": every reading mode, as the validation rows are built.
    rows, _details = expand_document(document, "val")
    print(f"{source_id}: {page_count} pages, {len(rows)} rows "
          f"(delivered split: {metadata.get('split')})")
    return rows


def _document_report_path(source_id: str) -> str:
    return f"/workspace/logs/base_vs_checkpoint_{source_id}.json"


def _eval_rows(client, args) -> list[dict]:
    """The adapter's rows of ``args.split``.

    Training writes the adapter's own copy of train and val only; test lives in
    the corpus-wide file. Without that copy, the file is read and narrowed to the
    adapter's document types.
    """
    from artifact_registry import paths
    from common.adapter import ADAPTER
    from evaluation.validation_generation import read_rows

    own = paths.corpus_scope_eval_split(args.corpus, args.split, ADAPTER.name, args.tenant)
    if client.exists(own):
        return read_rows(client.read_text(own))
    doc_types = set(ADAPTER.doc_types)

    rows = read_rows(client.read_text(paths.corpus_eval_split(args.corpus, args.split, args.tenant)))
    return [r for r in rows if r.get("doc_type") in doc_types]


def _print_audit(args) -> int:
    import csv
    from collections import Counter

    findings = label_audit(args.audit_labels, mode=args.mode)
    if not findings:
        print("no findings - or the report carries no OCR text (rerun --against-base)")
        return 0
    per_doc: dict[str, Counter] = {}
    for f in findings:
        per_doc.setdefault(f["source_id"], Counter())[f["finding"]] += 1
    short = ["gold missing", "gold not on page", "model invented"]
    print(f"{'document':<14}" + "".join(f"{h:>18}" for h in short))
    for source_id, counts in sorted(per_doc.items()):
        print(f"{source_id:<14}" + "".join(f"{counts.get(k, 0):>18}" for k in AUDIT_KINDS))
    totals = Counter(f["finding"] for f in findings)
    print(f"{'total':<14}" + "".join(f"{totals.get(k, 0):>18}" for k in AUDIT_KINDS))
    top = Counter(f["field"].split("[")[0] for f in findings if f["finding"] == AUDIT_KINDS[0])
    if top:
        print("fields the gold most often lacks: " + ", ".join(f"{k} ({n})" for k, n in top.most_common(8)))
    out = Path(args.out or str(Path(args.audit_labels).with_name(
        "label_audit_" + Path(args.audit_labels).stem + ".csv")))
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "field", "finding", "gold", "model"])
        writer.writeheader()
        writer.writerows(findings)
    print(f"every finding, to check against the PDFs: {out}")
    return 0


def _report_path(split: str) -> str:
    """Where a comparison over ``split`` is saved; the validation one keeps its old name."""
    suffix = "" if split == "val" else f"_{split}"
    return f"/workspace/logs/base_vs_checkpoint{suffix}.json"


def _print_document(args) -> int:
    """--document: list the documents, or compare one and save everything about it.

    The comparison goes to one folder: the PDF (when the pod still holds it), the
    gold JSON, each model's full JSON answer, the field-by-field CSV and the
    printed summary.
    """
    import csv
    import shutil

    if args.document == "list":
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        seen: dict[str, str] = {}
        for saved in next(iter(report.values()))["generations"]:
            seen.setdefault(saved["row"]["source_id"], str(saved["row"].get("lob")))
        for source_id, lob in sorted(seen.items()):
            print(f"{source_id}  lob={lob}{_describe_source(_provenance(source_id, args.tenant))}")
        return 0

    result = document_comparison(args.report, args.document, args.mode)
    meta = _provenance(args.document, args.tenant)
    labels = list(result["summary"])
    name = {"base": "base", "checkpoint": "fine-tuned"}
    lines = [f"document {args.document}{_describe_source(meta)}  (reading: {args.mode})",
             f"{'':<34}" + "".join(f"{name.get(x, x):>14}" for x in labels)]
    acc = [result["summary"][x]["accuracy"] for x in labels]
    lines.append(f"{'accuracy (single fields)':<34}" + "".join(
        f"{a:>14.1%}" if a is not None else f"{'-':>14}" for a in acc))
    lines.append(f"{'fields correct / scored':<34}" + "".join(
        f"{str(result['summary'][x]['fields_correct']) + ' / ' + str(result['summary'][x]['fields_scored']):>14}"
        for x in labels))
    for outcome in OUTCOMES:
        lines.append(f"{outcome:<34}" + "".join(
            f"{result['summary'][x]['outcomes'][outcome]:>14}" for x in labels))
    lines.append(f"{'unusable windows':<34}" + "".join(
        f"{result['summary'][x]['unusable_windows']:>14}" for x in labels))
    for x in labels:
        for table, text in result["summary"][x]["table_rows"].items():
            lines.append(f"  table {table} ({name.get(x, x)}): {text}")

    folder = Path(args.out or f"/workspace/logs/compare_{args.document}_{args.mode}")
    folder.mkdir(parents=True, exist_ok=True)
    columns = ["field", "gold"] + [c for x in labels for c in (x, f"{x} outcome")]
    with (folder / "fields.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in result["fields"]:
            writer.writerow({c: "" if row.get(c) is None else row.get(c) for c in columns})
    (folder / "gold.json").write_text(
        json.dumps(result["gold"], indent=2, ensure_ascii=False), encoding="utf-8")
    for x in labels:
        (folder / f"{name.get(x, x).replace('-', '_')}_output.json").write_text(
            json.dumps(result["answers"][x], indent=2, ensure_ascii=False), encoding="utf-8")
    pdf = _find_pdf(meta)
    if pdf:
        shutil.copyfile(pdf, folder / "document.pdf")
    (folder / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)
    print(f"saved in {folder}: " + ", ".join(sorted(f.name for f in folder.iterdir())))
    return 0


def _provenance(source_id: str, tenant: str) -> dict:
    """The label metadata the import wrote: the delivered bundle and whether it is synthetic."""
    try:
        from artifact_registry import paths
        from artifact_registry.blob_client import BlobClient

        return BlobClient().read_json(paths.label_metadata("policy", source_id, tenant)) or {}
    except Exception:  # noqa: BLE001 - provenance is a courtesy; the comparison stands without it
        return {}


def _describe_source(meta: dict) -> str:
    if not meta:
        return ""
    kind = "synthetic" if meta.get("synthetic") else "real"
    return f"  [{kind}, from {meta.get('imported_from')}]"


def _find_pdf(meta: dict, roots: tuple[str, ...] = ("/workspace/intake",)) -> Path | None:
    """The PDF the document was imported from, if the pod still holds its bundle."""
    folder = meta.get("imported_from") if meta else None
    if not folder:
        return None
    for root in roots:
        for candidate in sorted(Path(root).glob(f"*/{folder}/*.pdf")):
            return candidate
    return None


#: What the comparison prints, in order: (metric, plain name). The rest is in the report.
COMPARED = (
    ("field_normalized_match", "accuracy (correct fields, formatting ignored)"),
    ("field_exact_match", "exact match (character for character)"),
    ("field_precision", "precision (written values that are right)"),
    ("field_recall", "recall (label values found)"),
    ("field_f1", "F1 (precision and recall combined)"),
    ("false_null_rate", "left empty though on the page *"),
    ("hallucination_rate", "hallucination (value not in the text sent) *"),
    ("confusable_misattribution_rate", "value put in a similar wrong field *"),
    ("schema_validity_rate", "valid JSON for the schema"),
    ("image_only_accuracy", "accuracy, image only (no OCR)"),
    ("ocr_arbitration_accuracy", "accuracy, corrupted OCR"),
    ("list_field_recall", "table rows found (recall)"),
    ("list_field_precision", "table rows written that are real (precision)"),
    ("field_f1_list_fields", "table rows F1"),
)


def print_comparison(results: dict[str, dict]) -> None:
    """Base against checkpoint, every measure, then where each is won and lost."""
    base, tuned = results.get("base", {}), results.get("checkpoint", {})
    bm, tm = base.get("metrics", {}), tuned.get("metrics", {})
    print()
    print(f"{'measure':<48}{'base':>9}{'fine-tuned':>12}{'change':>10}")
    for name, label in COMPARED:
        b, t = bm.get(name), tm.get(name)
        if b is None or t is None:
            print(f"{label:<48}{'-':>9}{'-':>12}")
            continue
        print(f"{label:<48}{b:>9.1%}{t:>12.1%}{(t - b) * 100:>+8.1f}pt")
    b, t = bm.get("false_null_rate"), tm.get("false_null_rate")
    if b is not None and t is not None:
        print(f"{'completeness (1 - left empty)':<48}{1 - b:>9.1%}{1 - t:>12.1%}"
              f"{(b - t) * 100:>+8.1f}pt")
    for key in ("unusable_json", "setup_failures", "rows"):
        print(f"{key.replace('_', ' '):<48}{base.get(key, '-'):>9}{tuned.get(key, '-'):>12}")
    print("* lower is better")
    by_lob = [entry.get("details", {}).get("field_accuracy_by_lob", {}) for entry in (base, tuned)]
    lobs = sorted(set(by_lob[0]) | set(by_lob[1]))
    if lobs:
        print("accuracy by line of business:")
        for lob in lobs:
            cells = "".join(f"{v:>{w}.1%}" if v is not None else f"{'-':>{w}}"
                            for v, w in ((by_lob[0].get(lob), 9), (by_lob[1].get(lob), 12)))
            print(f"  {lob:<46}{cells}")
    for label, entry in (("base", base), ("fine-tuned", tuned)):
        weakest = entry.get("details", {}).get("weakest_fields") or []
        if weakest:
            print(f"weakest fields ({label}): " + ", ".join(
                f"{w['field']} {w['accuracy']:.0%} of {w['scored']}" for w in weakest[:6]))


def _against_base(rows: list[dict], client, args) -> int:
    """Base model and checkpoint on the same rows, scored by the gate's own report."""
    from evaluation.validation_generation import generate_validation
    from inference_core.model_runner import load_model, release_model

    model = load_model("base", client)
    results: dict[str, dict] = {}
    try:
        for label, adapter in (("base", None), ("checkpoint", args.checkpoint)):
            generations = generate_validation(rows, model, adapter=adapter)
            results[label] = {
                "adapter": adapter,
                **_scores(generations, label),
                # Every answer with its label, so a scoring question can be
                # answered again (--rescore) without the GPU.
                "generations": [_saved(g) for g in generations],
            }
    finally:
        release_model(model)

    out = Path(args.out or args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_comparison(results)
    print(f"full report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
