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
key). Each ``--doc`` line is named as serving names it (``classic_auto`` is
``personal_auto``) and a line with no schema is refused. The setup - the
servers' addresses (HTTPS, or HTTP to this machine), the MinerU tier, the page
links' login and token - is checked before the command detaches on the pod,
against the environment the run will have (``scripts/pod_run.sh env``). Writes
``<out>/<document>.json`` per document and run (timings, window stats,
accuracy and the served canonical JSON; ``.fields-first.json`` for that run; a
name used twice, or ``summary``, takes ``-2``) and ``<out>/summary.json`` /
``summary.md``.

``--repeat`` reads every document twice on the server path: all of them once,
then all again (``<document>.repeat.json``), so other documents' windows sit
between a document's two readings as they would on a busy server. Serving
decodes greedily with ``repeatable: false``, and a near tie can still flip with
the batch around it; ``run_to_run.json`` and the summary give each document's
run-to-run agreement (the golden eval's ``run_to_run_agreement``) and the
overall one, pooled by fields compared.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from testing.extract_server import positive_int, tier_problem, unique_names

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

#: Bearer token for the served model and MinerU.
TOKEN_ENV = "FIDEON_REMOTE_TOKEN"
#: A read-only SAS token for the page links (``--images link``).
LINK_TOKEN_ENV = "PAGE_LINK_SAS_TOKEN"

#: A scanned twin's name ends with this; its gold label is named without it.
SCAN_SUFFIX = "__scan"


def parse_doc(value: str) -> tuple[Path, str]:
    """``<pdf>=<line>`` - split at the last ``=``, since a path can hold one.

    The line is checked and named as serving names it
    (``serving.vllm_entrypoint.request_lob``, as the job server does): a line
    with no schema is refused here, not read against the fallback schema and
    scored as if it were the line meant; ``classic_auto`` or ``Personal_Auto``
    is ``personal_auto``."""
    from serving.vllm_entrypoint import ServingError, request_lob

    path, sep, lob = value.rpartition("=")
    if not sep or not path or not lob.strip():
        raise argparse.ArgumentTypeError(f"--doc takes <pdf>=<line of business>, got {value!r}")
    try:
        line = request_lob({"lob": lob.strip()})
    except ServingError as exc:
        raise argparse.ArgumentTypeError(f"{value!r}: {exc}") from None
    return Path(path), str(line)


def run_environment(detaching: bool) -> dict[str, str]:
    """The environment the run will have.

    Detached on the pod, the run's settings are the repo's ``.env`` under the
    caller's own (``scripts/pod_run.sh``): asked of the launcher itself
    (``pod_run.sh env``), so a check made before the detach sees the storage
    login the run will have. Anywhere else, the caller's own environment:
    nothing loads ``.env`` off the pod."""
    env = dict(os.environ)
    if not detaching:
        return env
    from orchestration.detach import POD_RUN

    try:
        shown = subprocess.run(["bash", POD_RUN.as_posix(), "env"], capture_output=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):    # no launcher to ask: the caller's, as before
        return env
    return dict(item.split("=", 1) for item in shown.stdout.decode("utf-8", "replace").split("\0") if "=" in item)


@contextmanager
def _environment(env: dict[str, str]) -> Iterator[None]:
    """``os.environ`` as ``env`` for a moment (the storage login is read from it)."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def setup_problem(args: argparse.Namespace, env: dict[str, str]) -> str | None:
    """Why the servers' addresses or the page links cannot work; None when they can.

    Checked before the command detaches on the pod: the backend and the link
    store refuse the same, but only once the run has started, in tmux, with the
    operator's command already exited 0. ``env`` is the run's environment
    (:func:`run_environment`). Nothing is sent or written."""
    from common.transport import InsecureUrlError, assert_encrypted

    try:
        assert_encrypted(args.model_url, "the model server")
        assert_encrypted(args.mineru_url, "the MinerU server")
    except InsecureUrlError as exc:
        return str(exc)
    if args.images != "link":
        return None
    from artifact_registry.blob_client import BlobClient, BlobError
    from inference_core.page_links import LINK_PREFIX, PageLinks

    # --link-token's default was read from the caller's environment; the run reads its own.
    token = args.link_token or env.get(LINK_TOKEN_ENV) or None
    with _environment(env):
        try:
            PageLinks(BlobClient(), f"{LINK_PREFIX}/check", read_token=token).check()
        except (BlobError, ValueError) as exc:
            return f"--images link cannot make a page link: {exc}"
    return None


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


def _calibration(model: Any) -> Any:
    from calibration.fit_calibration import CalibrationParams

    return CalibrationParams(method="temperature", doc_type="policy", model_version=model.tag, temperature=1.0)


def _row(pdf: Path, lob: str, *, run: str, mode: str, backend: Any, result: Any, pages: int, windows: int,
         timings: dict[str, Any], sent: dict[str, float], ocr: dict[str, Any], gold: Path | None,
         accuracy_at_fields: dict[str, Any] | None = None) -> dict[str, Any]:
    """One document's line in the report: timings, windows, accuracy and the served JSON.

    ``accuracy`` is the result as finally served; ``accuracy_at_fields`` the
    fields as served when they came (fields first: before the page text could
    change them), else the same."""
    from testing.extract_server import _extracted_counts, _flag_kinds, window_summary

    accuracy = score(result, gold, lob) if gold else None
    return {
        "document": pdf.name, "lob": lob, "run": run, "mode": mode, "model": backend.model, "pages": pages,
        "windows_planned": windows, "model_calls": len(result.window_latencies_ms),
        "timings_s": {**timings, "calls": [round(ms / 1000, 2) for ms in result.window_latencies_ms]},
        "sent_mb": round(sum(sent.values()), 1),
        "sent_mb_parts": sent,
        "images": backend.images,
        "ocr": ocr,
        "windows": window_summary(result.window_stats),
        "schema_valid": result.schema_valid,
        "validation_errors": list(result.validation_errors)[:20],
        "review_flags": len(result.review_flags),
        "review_flag_kinds": _flag_kinds(result.review_flags),
        "text_coverage": result.text_coverage,
        "overall_confidence": result.overall_confidence,
        "accuracy": accuracy,
        "accuracy_at_fields": accuracy_at_fields if run == "fields_first" else accuracy,
        "extracted": _extracted_counts(result.extraction),
        "window_stats": result.window_stats,
        "extraction": result.extraction,
    }


def time_document(pdf: Path, lob: str, *, backend: Any, model: Any, reader: Any, mode: str, work: Path,
                  gold: Path | None = None, links: Any = None, key: str | None = None) -> dict[str, Any]:
    """One document through render and OCR, the routine and the scorer;
    with ``links`` (:class:`inference_core.page_links.PageLinks`) its pages go to
    the model as links, published while MinerU reads the PDF. ``key`` names its
    folder (default the PDF's stem; unique per run, :func:`main`)."""
    from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract

    key = key or pdf.stem
    began = time.perf_counter()
    data = pdf.read_bytes()
    publish = (lambda paths: links.publish(paths, folder=key)) if links is not None else None
    read = reader.read(data, work / key / "pages", name=pdf.name, with_text=mode != "image_only",
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
    result = extract(request, model, StaticClassifier("policy"), _calibration(model), strict_schema=False)
    done = time.perf_counter()
    sent = {
        "pdf": round(len(data) / 2**20, 1) if mode != "image_only" else 0.0,
        "page_links": (read.get("links") or {}).get("mb", 0.0),
        "model_requests": round((backend.sent_bytes - sent_before) / 2**20, 1),
    }
    total = round(done - began, 2)
    timings = {**read["timings_s"], "model": round(done - model_started, 2),
               # Fields and full text come together on this path.
               "fields": total, "full_text": total, "total": total}
    return _row(pdf, lob, run="standard", mode=mode, backend=backend, result=result, pages=len(paths),
                windows=windows, timings=timings, sent=sent, ocr=read["ocr"], gold=gold)


def time_fields_first(pdf: Path, lob: str, *, backend: Any, model: Any, reader: Any, work: Path,
                      gold: Path | None = None, links: Any = None, verify: bool = True,
                      key: str | None = None) -> dict[str, Any]:
    """One document fields first (``serving.fields_first``): the model reads the
    page images, the fields are held to the pages they cite, then MinerU's text
    of the rest holds them to the whole document and adds the full-text tier.
    ``fields`` is the time the gate counts (verification included);
    ``full_text`` how long the rest came after."""
    from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
    from serving.doc_type_classifier import StaticClassifier
    from serving.fields_first import add_full_text, read_fields

    key = key or f"{pdf.stem}.fields_first"
    began = time.perf_counter()
    data = pdf.read_bytes()
    pages_dir = work / key / "pages"
    publish = (lambda paths: links.publish(paths, folder=key)) if links is not None else None
    sent_before = backend.sent_bytes
    fields = read_fields(data, name=pdf.name, lob=lob, source_id=pdf.stem, model=model, reader=reader,
                         pages_dir=pages_dir, classifier=StaticClassifier("policy"),
                         calibration=_calibration(model), publish_pages=publish, verify=verify)
    result, count = fields["result"], len(fields["images"])
    served = fields["served"]
    every_page, declarations_page = pages_to_read(None, count)
    windows = len(plan_windows(lob, every_page, declarations_page, None))
    # What the gate serves, scored now: the page text that comes next may change it.
    at_fields = score(served, gold, lob) if gold and served is not None else None
    full = add_full_text(data, result, name=pdf.name, lob=lob, reader=reader, pages_dir=pages_dir,
                         page_count=count, known_texts=fields["cited_texts"])
    done = time.perf_counter()
    sent = {
        "pdf": round(len(data) / 2**20, 1),
        "page_links": (fields.get("links") or {}).get("mb", 0.0),
        "model_requests": round((backend.sent_bytes - sent_before) / 2**20, 1),
    }
    timings = {**fields["timings_s"], "ocr": full["timings_s"]["ocr"],
               "full_text": full["timings_s"]["full_text"], "total": round(done - began, 2)}
    row = _row(pdf, lob, run="fields_first", mode="image_only", backend=backend, result=result, pages=count,
               windows=windows, timings=timings, sent=sent, ocr=full["ocr"], gold=gold,
               accuracy_at_fields=at_fields)
    held = [flag for flag in (served.review_flags if served is not None else []) if flag.endswith(":pending_full_text")]
    row["fields_check"] = {"verified": fields["verified"], "served": served is not None,
                           "cited_pages": len(fields["cited_pages"]), "held_back": len(held),
                           "served_flags": sorted(served.review_flags) if served is not None else []}
    return row


#: The key a document's second reading is filed under (``--repeat``).
REPEAT_SUFFIX = ".repeat"


def agreement(first: Any, second: Any, lob: str) -> tuple[float, float] | None:
    """How alike two served answers to one document are: ``(share of fields
    equal, fields compared)``, or None when neither wrote a field - the golden
    eval's ``run_to_run_agreement``, on the answers as it scores them."""
    from common.lob import merge_line
    from evaluation.run_eval import _is_common_model_document, _scored_view, run_to_run_agreement

    common_model = _is_common_model_document("policy", {"lob": lob})
    line = merge_line(lob) if common_model else None
    selectors = ("policy", None, lob)
    return run_to_run_agreement(_scored_view(first or {}, selectors, line, common_model),
                                _scored_view(second or {}, selectors, line, common_model),
                                common_model=common_model, line=line)


def run_to_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each document read twice (``--repeat``): per document and run, the share
    of its fields the two readings agree on; overall, pooled by fields compared.
    A reading that failed leaves its document out, counted in ``unpaired``."""
    readings: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        document = (row.get("pdf") or row["document"], row.get("lob", ""), row.get("run", ""))
        readings.setdefault(document, {})[row.get("pass", 1)] = row
    documents: list[dict[str, Any]] = []
    agreed = compared = 0.0
    unpaired = 0
    for (pdf, _lob, run), passes in readings.items():
        first, second = passes.get(1), passes.get(2)
        if first is None or second is None or "error" in first or "error" in second:
            unpaired += 1
            continue
        result = agreement(first.get("extraction"), second.get("extraction"), first["lob"])
        entry = {"document": first["document"], "pdf": pdf, "run": run, "agreement": None, "fields_compared": 0}
        if result is not None:
            share, count = result
            entry.update(agreement=round(share, 4), fields_compared=round(count, 1))
            agreed += share * count
            compared += count
        documents.append(entry)
    return {"documents": documents, "overall": round(agreed / compared, 4) if compared else None,
            "fields_compared": round(compared, 1), "unpaired": unpaired}


def agreement_table(report: dict[str, Any]) -> str:
    out = ["| Document | Run | Run-to-run agreement | Fields compared |", "|---|---|---|---|"]
    for entry in report["documents"]:
        out.append(f"| {entry['document']} | {entry['run']} | {entry['agreement']} | {entry['fields_compared']} |")
    out.append(f"| **overall** | | {report['overall']} | {report['fields_compared']} |")
    if report["unpaired"]:
        out.append(f"\n{report['unpaired']} document(s) not read twice (a reading failed): left out.")
    return "\n".join(out)


def table(rows: list[dict[str, Any]]) -> str:
    head = ("| Document | Run | Pages | Windows | Fields s | Full text s | Model s | Rounds | Splits | "
            "Answer tokens | Slowest window s | Sent MB | Schema valid | Text coverage | Exact match at fields | "
            "Exact match |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = [head]
    for row in rows:
        if "error" in row:
            out.append(f"| {row['document']} | {row.get('run', '')} | FAILED: {row['error'][:90]} |"
                       + " |" * 13)
            continue
        t, w = row["timings_s"], row["windows"]
        accuracy, at_fields = row["accuracy"] or {}, row.get("accuracy_at_fields") or {}
        slowest = (w["slowest_window"] or {}).get("seconds")
        out.append(f"| {row['document']} | {row['run']} | {row['pages']} | {row['windows_planned']} | "
                   f"{t['fields']:.1f} | {t['full_text']:.1f} | {t['model']:.1f} | {w['rounds']} | "
                   f"{w['split']} | {w['answer_tokens']} | {slowest} | {row['sent_mb']} | "
                   f"{row['schema_valid']} | {row['text_coverage']} | "
                   f"{at_fields.get('field_exact_match_rate', '')} | {accuracy.get('field_exact_match_rate', '')} |")
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
    parser.add_argument("--runs", type=lambda v: [r.strip() for r in v.split(",") if r.strip()],
                        default=["standard"], metavar="RUN[,RUN]",
                        help="how each document is read, one row each: standard (OCR then the model, --mode) "
                             "and/or fields-first (the model reads the images, MinerU's text after: "
                             "serving.fields_first). E.g. --runs standard,fields-first to compare")
    parser.add_argument("--tier", default=None,
                        help="the MinerU tier (default the pinned one, configs/shared/ocr.yaml)")
    parser.add_argument("--ocr-mode", default=None, choices=["auto", "txt", "ocr"],
                        help="force MinerU's mode (default: decided per document as the OCR stage does)")
    parser.add_argument("--unverified-fields", action="store_true",
                        help="fields first: score the fields before any page text has checked them "
                             "(timing only; production serves them checked against the pages they cite)")
    parser.add_argument("--any-mineru", action="store_true",
                        help="accept text from a MinerU version or tier other than the pinned one "
                             "(timing only: the model is trained on the pinned one's text)")
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
    parser.add_argument("--concurrency", type=positive_int, default=128,
                        help="requests in flight at most (lower it through a proxy: 16)")
    parser.add_argument("--no-repeat-stop", action="store_true",
                        help="let an answer that repeats itself run to its token limit (to compare)")
    parser.add_argument("--repeat", action="store_true",
                        help="read every document twice - all documents once, then all again, so other "
                             "documents' windows sit between a document's two readings, as on a busy server "
                             "- and report each document's run-to-run agreement and the overall one "
                             "(repeatable: false decodes greedily, yet a near tie can flip with the batch)")
    parser.add_argument("--gold-dir", type=Path, default=None,
                        help="where gold JSONs are (default: the Gold_json folder beside each PDF's)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where results go (default testing/results/latency_check/<time>)")
    args = parser.parse_args(argv)
    from common.config import served_mineru

    pin = served_mineru()
    unknown = [run for run in args.runs if run not in ("standard", "fields-first")]
    if unknown or not args.runs:
        parser.error(f"--runs takes standard and/or fields-first, got {args.runs}")
    twice = sorted({run for run in args.runs if args.runs.count(run) > 1})
    if twice:
        # The second would be written over the first's file and counted twice in the table.
        parser.error(f"--runs names {', '.join(twice)} more than once: each run is one row per document")
    if problem := tier_problem(args.tier, args.any_mineru):
        parser.error(problem)
    args.tier = args.tier or pin["tier"]
    missing = [str(pdf) for pdf, _lob in args.docs if not pdf.is_file()]
    if missing:
        parser.error(f"no such PDF: {', '.join(missing)}")

    from orchestration.detach import detach_module_if_needed, should_detach

    # Before the detach below: a refusal after it is seen by nobody.
    if problem := setup_problem(args, run_environment(should_detach())):
        parser.error(problem)
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
                       ocr_mode=args.ocr_mode, ocr_parts=args.ocr_parts, render_workers=args.render_workers,
                       expected_version=None if args.any_mineru else pin["version"])
    print(f"model {backend.model} at {args.model_url}; MinerU at {args.mineru_url} (tier {args.tier}, "
          f"{'any version' if args.any_mineru else 'version ' + pin['version']}, "
          f"{args.ocr_parts} part(s)); images {args.images}"
          f"{'' if args.color_pages else ', gray pages as grayscale PNG'}", flush=True)

    # Each document's file per run, and its pages' folder: unique in the run (two
    # folders can each hold a policy.pdf; a summary.pdf is not summary.json).
    # --repeat: every document once, then every document again - the second pass
    # filed under <key>.repeat - so a document's two readings are apart, with the
    # other documents' windows between them.
    passes = (1, 2) if args.repeat else (1,)
    keys = iter(unique_names([pdf.stem + ("" if run == "standard" else f".{run}")
                              + ("" if attempt == 1 else REPEAT_SUFFIX)
                              for attempt in passes for pdf, _lob in args.docs for run in args.runs]))
    rows: list[dict[str, Any]] = []
    for attempt, pdf, lob in ((attempt, pdf, lob) for attempt in passes for pdf, lob in args.docs):
        gold = find_gold(pdf, args.gold_dir)
        for run in args.runs:
            key = next(keys)
            print(f"reading {pdf.name} ({lob}), {run}{'' if gold else ', no gold JSON found'}"
                  f"{f', reading {attempt} of 2' if args.repeat else ''} ...", flush=True)
            try:
                if run == "fields-first":
                    row = time_fields_first(pdf, lob, backend=backend, model=model, reader=reader, work=out,
                                            gold=gold, links=links, verify=not args.unverified_fields, key=key)
                else:
                    row = time_document(pdf, lob, backend=backend, model=model, reader=reader, mode=args.mode,
                                        work=out, gold=gold, links=links, key=key)
            except Exception as exc:  # noqa: BLE001 - one document's failure is reported, the rest still run
                log.exception("%s failed", pdf.name)
                row = {"document": pdf.name, "lob": lob, "run": run, "error": f"{type(exc).__name__}: {exc}"}
            finally:
                if links is not None and not args.keep_links:
                    links.discard()
            row = {**row, "pdf": str(pdf), "output": f"{key}.json", "pass": attempt}
            (out / f"{key}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            if "error" not in row:
                t, w = row["timings_s"], row["windows"]
                accuracy = row["accuracy"] or {}
                print(f"  {row['pages']} pages, {row['windows_planned']} windows: fields {t['fields']:.1f} s "
                      f"(model {t['model']:.1f} s), full text {t['full_text']:.1f} s, total {t['total']:.1f} s; "
                      f"{w['split']} split, {w['failed']} failed, {w['stopped_repeating']} stopped repeating; "
                      f"sent {row['sent_mb']} MB"
                      + (f"; exact match {accuracy['field_exact_match_rate']}" if accuracy else ""), flush=True)

    backend.close()
    summaries = [{k: v for k, v in row.items() if k not in ("extraction", "window_stats")} for row in rows]
    report = table(rows)
    if args.repeat:
        repeated = run_to_run(rows)
        (out / "run_to_run.json").write_text(json.dumps(repeated, indent=2), encoding="utf-8")
        report += "\n\n### Run-to-run agreement\n\n" + agreement_table(repeated)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(report + "\n", encoding="utf-8")
    print("\n" + report + f"\n\nresults in {out}")
    return 0 if all("error" not in row for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
