"""Time the extraction routine on PDFs, with a served model and a served MinerU.

    python -m testing.latency_check --model-url http://localhost:8000 --mineru-url http://localhost:8001 \\
        --doc /workspace/pdfs/a__scan.pdf=personal_auto --doc /workspace/pdfs/b__scan.pdf=homeowners

For a model already running behind ``vllm serve`` and a MinerU 4 API (on a pod,
on localhost or behind its RunPod proxy URL). Each document, one at a time:

1. its pages rendered by the shared renderer at the resolution cap - the pixels
   training and serving use - in several processes, while the served MinerU
   reads its text (``data_pipeline.ocr.served_ocr``);
2. read by :func:`serving.pipeline.extract` - production's routine, unchanged -
   with the served model (``inference_core.openai_backend``), each window
   taken as it is answered;
3. scored against its gold JSON, when one is found (``--gold-dir``, or the
   ``Gold_json`` folder beside the PDF's own), by the testing harness's scorer.

Each step is timed apart, with every window's prompt and answer tokens, seconds
and outcome (``window_stats``) and the bytes sent (``Sent MB``: the PDF, the
page links and the model's requests). Where the bytes go matters as much as the
GPU: a page image goes with every window that reads it, about four times a
page, so from a laptop through the proxy (about 1 MB/s) sending them is most of
the model time. Three things cut it, none changing what the model sees:

* a gray page goes as a grayscale PNG, about 60% of the bytes (always;
  ``--color-pages`` sends the pages as rendered, to compare);
* ``--images link`` uploads each page once to Blob, while MinerU reads the PDF,
  and sends the model a read-only link it fetches itself: a 114-page scan's
  871 MB of requests become about 105 MB of pages. It needs a storage login
  that may write (``AZURE_STORAGE_ACCOUNT_URL`` + ``AZURE_STORAGE_SAS_TOKEN``,
  or ``AZURE_STORAGE_CONNECTION_STRING``) and, unless that login holds the
  account key, a read-only SAS for the links (``--link-token`` or
  ``$PAGE_LINK_SAS_TOKEN``, ``sp=r``). The pages are deleted after each
  document (``--keep-links`` keeps them);
* ``--ocr-parts N`` sends the PDF to MinerU as N runs of pages, one after
  another, so it reads the first while the rest upload.

Next to the servers, ``--images file`` sends each image's path instead; the
vLLM server must then be started with ``--allowed-local-media-path`` covering
``--out``. A RunPod serverless vLLM endpoint is a ``--model-url`` too
(``https://api.runpod.ai/v2/<endpoint id>/openai``, not yet tried); its request
cap needs ``--images link``.

The token for both servers is ``--token`` or ``$FIDEON_REMOTE_TOKEN``;
``--mineru-token`` when MinerU's differs (a serverless model's is the RunPod API
key). Writes ``<out>/<document>.json`` (timings, window stats, accuracy and the
served canonical JSON) and ``<out>/summary.json`` / ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

#: Bearer token for the served model and MinerU.
TOKEN_ENV = "FIDEON_REMOTE_TOKEN"
#: A read-only SAS token for the page links (``--images link``).
LINK_TOKEN_ENV = "PAGE_LINK_SAS_TOKEN"

#: A scanned twin's name ends with this; its gold label is named without it.
SCAN_SUFFIX = "__scan"


def parse_doc(value: str) -> tuple[Path, str]:
    """``<pdf>=<line>`` - split at the last ``=``, since a path can hold one."""
    path, sep, lob = value.rpartition("=")
    if not sep or not path or not lob.strip():
        raise argparse.ArgumentTypeError(f"--doc takes <pdf>=<line of business>, got {value!r}")
    return Path(path), lob.strip()


def find_gold(pdf: Path, gold_dir: Path | None = None) -> Path | None:
    """The gold JSON of ``pdf``: ``<stem>.json``, a scan's without ``__scan``, in
    ``gold_dir`` or in the ``Gold_json`` folder beside the PDF's (the training
    data's ``<split>/<line>/Scanned|Digital`` layout)."""
    stem = pdf.stem.removesuffix(SCAN_SUFFIX)
    folders = [gold_dir] if gold_dir else []
    folders.append(pdf.resolve().parent.parent / "Gold_json")
    for folder in folders:
        path = folder / f"{stem}.json"
        if path.is_file():
            return path
    return None


def score(result: Any, gold_path: Path, lob: str) -> dict[str, Any]:
    """The served JSON against the gold, by the testing harness's scorer."""
    from testing.run_extraction import score_against_ground_truth

    metrics = score_against_ground_truth(result, json.loads(gold_path.read_text(encoding="utf-8")), lob=lob)
    f1 = metrics.get("list_field_f1") or {}
    return {
        "gold": str(gold_path),
        "field_exact_match_rate": metrics["field_exact_match_rate"],
        "field_normalized_match_rate": metrics["field_normalized_match_rate"],
        "mean_list_field_f1": round(sum(f1.values()) / len(f1), 4) if f1 else None,
        **{k: v for k, v in metrics.items() if k not in ("failures", "field_accuracy_by_field")},
        "failures": metrics["failures"][:50],
    }


def time_document(pdf: Path, lob: str, *, backend: Any, model: Any, reader: Any, mode: str, work: Path,
                  gold: Path | None = None, links: Any = None) -> dict[str, Any]:
    """One document through render and OCR, the routine and the scorer;
    with ``links`` (:class:`inference_core.page_links.PageLinks`) its pages go to
    the model as links, published while MinerU reads the PDF."""
    from calibration.fit_calibration import CalibrationParams
    from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from testing.extract_server import _extracted_counts, _flag_kinds, window_summary

    began = time.perf_counter()
    data = pdf.read_bytes()
    publish = (lambda paths: links.publish(paths, folder=pdf.stem)) if links is not None else None
    read = reader.read(data, work / pdf.stem / "pages", name=pdf.name, with_text=mode != "image_only",
                       publish_pages=publish)
    texts, paths = read["texts"], read["images"]

    request = ExtractionRequest(
        source_id=pdf.stem, image_paths=paths, page_texts=texts, modality_mode=mode,
        known_doc_type="policy", known_lob=lob, source_file_name=pdf.name,
    )
    every_page, declarations_page = pages_to_read(texts or None, len(paths))
    windows = len(plan_windows(lob, every_page, declarations_page, texts or None))
    sent_before = backend.sent_bytes
    model_started = time.perf_counter()
    result = extract(
        request, model, StaticClassifier("policy"),
        CalibrationParams(method="temperature", doc_type="policy", model_version=model.tag, temperature=1.0),
        strict_schema=False,
    )
    done = time.perf_counter()
    sent = {
        "pdf": round(len(data) / 2**20, 1) if mode != "image_only" else 0.0,
        "page_links": (read.get("links") or {}).get("mb", 0.0),
        "model_requests": round((backend.sent_bytes - sent_before) / 2**20, 1),
    }
    return {
        "document": pdf.name, "lob": lob, "mode": mode, "model": backend.model, "pages": len(paths),
        "windows_planned": windows, "model_calls": len(result.window_latencies_ms),
        "timings_s": {
            **read["timings_s"],
            "model": round(done - model_started, 2),
            "calls": [round(ms / 1000, 2) for ms in result.window_latencies_ms],
            "total": round(done - began, 2),
        },
        "sent_mb": round(sum(sent.values()), 1),
        "sent_mb_parts": sent,
        "images": backend.images,
        "ocr": read["ocr"],
        "windows": window_summary(result.window_stats),
        "schema_valid": result.schema_valid,
        "validation_errors": list(result.validation_errors)[:20],
        "review_flags": len(result.review_flags),
        "review_flag_kinds": _flag_kinds(result.review_flags),
        "text_coverage": result.text_coverage,
        "overall_confidence": result.overall_confidence,
        "accuracy": score(result, gold, lob) if gold else None,
        "extracted": _extracted_counts(result.extraction),
        "window_stats": result.window_stats,
        "extraction": result.extraction,
    }


def table(rows: list[dict[str, Any]]) -> str:
    head = ("| Document | Line | Pages | Windows | Pages+text s | Model s | Total s | Rounds | Splits | "
            "Answer tokens | Slowest window s | Sent MB | Schema valid | Text coverage | Exact match |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = [head]
    for row in rows:
        if "error" in row:
            out.append(f"| {row['document']} | {row['lob']} | FAILED: {row['error'][:90]} |"
                       + " |" * 12)
            continue
        t, w, accuracy = row["timings_s"], row["windows"], row["accuracy"] or {}
        slowest = (w["slowest_window"] or {}).get("seconds")
        out.append(f"| {row['document']} | {row['lob']} | {row['pages']} | {row['windows_planned']} | "
                   f"{t['pages_and_text']:.1f} | {t['model']:.1f} | {t['total']:.1f} | {w['rounds']} | "
                   f"{w['split']} | {w['answer_tokens']} | {slowest} | {row['sent_mb']} | "
                   f"{row['schema_valid']} | {row['text_coverage']} | "
                   f"{accuracy.get('field_exact_match_rate', '')} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-url", required=True, help="the served model's base URL (vllm serve)")
    parser.add_argument("--served-model", default=None,
                        help="the name to ask it for, when it serves more than one (a LoRA adapter)")
    parser.add_argument("--mineru-url", required=True, help="the served MinerU 4 API's base URL")
    parser.add_argument("--token", default=os.environ.get(TOKEN_ENV),
                        help=f"bearer token for both servers (default ${TOKEN_ENV})")
    parser.add_argument("--mineru-token", default=None,
                        help="MinerU's bearer token, when it is not --token's (default --token)")
    parser.add_argument("--doc", dest="docs", action="append", type=parse_doc, required=True, metavar="PDF=LOB")
    parser.add_argument("--mode", default="ocr_plus_image", choices=["ocr_plus_image", "image_only"])
    parser.add_argument("--tier", default="basic", help="the MinerU tier (flash, basic, standard, advanced)")
    parser.add_argument("--ocr-mode", default="auto", choices=["auto", "txt", "ocr"])
    parser.add_argument("--ocr-parts", type=int, default=1,
                        help="send each PDF to MinerU as this many runs of pages, one after another, "
                             "each read while the next uploads")
    parser.add_argument("--render-workers", type=int, default=None,
                        help="processes rendering pages at once (default: up to 8, one per CPU)")
    parser.add_argument("--images", default="inline", choices=["inline", "link", "file"],
                        help="page images in each request (inline); uploaded once to Blob and sent as "
                             "read-only links (link); or by path on the server (file: next to it only, "
                             "with vLLM's --allowed-local-media-path)")
    parser.add_argument("--link-token", default=os.environ.get(LINK_TOKEN_ENV),
                        help=f"a read-only SAS (sp=r) for the page links (default ${LINK_TOKEN_ENV}); "
                             "not needed when the storage login holds the account key")
    parser.add_argument("--keep-links", action="store_true",
                        help="leave each document's pages in Blob after it is read (they are deleted)")
    parser.add_argument("--color-pages", action="store_true",
                        help="send gray pages as rendered (RGB), not as grayscale PNGs (to compare)")
    parser.add_argument("--concurrency", type=int, default=128,
                        help="requests in flight at most (lower it through a proxy: 16)")
    parser.add_argument("--no-repeat-stop", action="store_true",
                        help="let an answer that repeats itself run to its token limit (to compare)")
    parser.add_argument("--gold-dir", type=Path, default=None,
                        help="where gold JSONs are (default: the Gold_json folder beside each PDF's)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where results go (default testing/results/latency_check/<time>)")
    args = parser.parse_args(argv)
    missing = [str(pdf) for pdf, _lob in args.docs if not pdf.is_file()]
    if missing:
        parser.error(f"no such PDF: {', '.join(missing)}")

    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("testing.latency_check", argv, "latency-check"):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.ocr.served_ocr import ServedOcr
    from inference_core.model_runner import load_model
    from inference_core.openai_backend import OpenAIChatBackend

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / "results" / "latency_check" / stamp
    out.mkdir(parents=True, exist_ok=True)
    links = None
    if args.images == "link":
        import secrets

        from inference_core.page_links import LINK_PREFIX, PageLinks

        # The real storage: the pages must be where the model server can fetch them.
        links = PageLinks(BlobClient(), f"{LINK_PREFIX}/{stamp}-{secrets.token_hex(3)}",
                          read_token=args.link_token, gray=not args.color_pages)
        links.check()
    backend = OpenAIChatBackend(args.model_url, args.token, model=args.served_model,
                                max_concurrency=args.concurrency, images=args.images, links=links,
                                gray_pages=not args.color_pages, stop_repeats=not args.no_repeat_stop)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load_model("base", client, backend_impl=backend, label=backend.model)
    reader = ServedOcr(args.mineru_url, args.mineru_token or args.token, tier=args.tier,
                       ocr_mode=args.ocr_mode, ocr_parts=args.ocr_parts, render_workers=args.render_workers)
    print(f"model {backend.model} at {args.model_url}; MinerU at {args.mineru_url} (tier {args.tier}, "
          f"{args.ocr_parts} part(s)); images {args.images}"
          f"{'' if args.color_pages else ', gray pages as grayscale PNG'}", flush=True)

    rows: list[dict[str, Any]] = []
    for pdf, lob in args.docs:
        gold = find_gold(pdf, args.gold_dir)
        print(f"reading {pdf.name} ({lob}){'' if gold else ', no gold JSON found'} ...", flush=True)
        try:
            row = time_document(pdf, lob, backend=backend, model=model, reader=reader, mode=args.mode,
                                work=out, gold=gold, links=links)
        except Exception as exc:  # noqa: BLE001 - one document's failure is reported, the rest still run
            log.exception("%s failed", pdf.name)
            row = {"document": pdf.name, "lob": lob, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if links is not None and not args.keep_links:
                links.discard()
        (out / f"{pdf.stem}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
        if "error" not in row:
            t, w = row["timings_s"], row["windows"]
            accuracy = row["accuracy"] or {}
            print(f"  {row['pages']} pages, {row['windows_planned']} windows: pages and text "
                  f"{t['pages_and_text']:.1f} s, model {t['model']:.1f} s, total {t['total']:.1f} s; "
                  f"{w['split']} split, {w['failed']} failed, {w['stopped_repeating']} stopped repeating; "
                  f"sent {row['sent_mb']} MB"
                  + (f"; exact match {accuracy['field_exact_match_rate']}" if accuracy else ""), flush=True)

    backend.close()
    summaries = [{k: v for k, v in row.items() if k not in ("extraction", "window_stats")} for row in rows]
    report = table(rows)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(report + "\n", encoding="utf-8")
    print("\n" + report + f"\n\nresults in {out}")
    return 0 if all("error" not in row for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
