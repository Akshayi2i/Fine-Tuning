"""Send PDFs to the pod's extraction test server through its RunPod proxy URL, and report.

    python -m testing.extract_client --url https://<pod id>-8000.proxy.runpod.net --token <token> ^
        --doc "D:\\...\\a__scan.pdf=personal_auto" --doc "D:\\...\\b__scan.pdf=homeowners"

Each ``--doc`` is ``<pdf>=<line of business>``. The PDFs are posted one after
another and queue on the server, which reads them one at a time
(``testing/extract_server.py``); this polls until every job is done, saves each
served JSON, and prints a table of what came back and how long it took - the
server's own timings, so a document's wait in the queue and the upload are not
counted as its extraction time.

Writes ``<out>/<document>.json`` (the summary and the canonical JSON) and
``<out>/summary.json`` / ``summary.md``. ``testing/results/`` is git-ignored: the
served JSON quotes the document.

Standard library only, so it runs from the laptop's environment as it is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from testing.extract_server import TOKEN_ENV

ROOT = Path(__file__).resolve().parent

#: Uploads of the larger scans (84 MB) take a while through the proxy.
UPLOAD_TIMEOUT_S = 900
CALL_TIMEOUT_S = 60


class ClientError(RuntimeError):
    """The server refused a call, or could not be reached."""


def parse_doc(value: str) -> tuple[Path, str]:
    """``<pdf>=<line>`` - split at the last ``=``, since a Windows path can hold one."""
    path, sep, lob = value.rpartition("=")
    if not sep or not path or not lob:
        raise argparse.ArgumentTypeError(f"--doc takes <pdf>=<line of business>, got {value!r}")
    return Path(path), lob.strip()


def call(url: str, token: str, *, data: bytes | None = None, timeout: float = CALL_TIMEOUT_S) -> Any:
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/pdf",
        # Some proxies refuse a request with Python's default agent.
        "User-Agent": "fideon-extract-client/1",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise ClientError(f"{exc.code} from {urllib.parse.urlparse(url).path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ClientError(f"could not reach {url}: {exc.reason}") from exc


def submit(base: str, token: str, pdf: Path, lob: str, mode: str | None) -> dict[str, Any]:
    query = urllib.parse.urlencode({"lob": lob, "name": pdf.name, **({"mode": mode} if mode else {})})
    return call(f"{base}/jobs?{query}", token, data=pdf.read_bytes(), timeout=UPLOAD_TIMEOUT_S)


def wait(base: str, token: str, jobs: list[str], poll_s: float) -> dict[str, dict[str, Any]]:
    """Each job's final view, polling until none is queued or running."""
    views: dict[str, dict[str, Any]] = {}
    announced: set[str] = set()
    while True:
        for job in jobs:
            if views.get(job, {}).get("status") not in ("done", "failed"):
                views[job] = call(f"{base}/jobs/{job}", token)
        for job, view in views.items():
            if view["status"] == "fields_ready" and job not in announced:
                announced.add(job)
                print(f"  fields of {view['name']} ready after {view.get('fields_s')} s; its full text follows",
                      flush=True)
        pending = [v for v in views.values() if v["status"] in ("queued", "running", "fields_ready")]
        if not pending:
            return views
        running = next((v for v in pending if v["status"] == "running"), None)
        print(f"  {len(pending)} job(s) left"
              + (f"; reading {running['name']} for {running.get('running_s', 0):.0f} s" if running else ""),
              flush=True)
        time.sleep(poll_s)


def table(views: list[dict[str, Any]]) -> str:
    """The run as a markdown table, one row per document."""
    head = ("| Document | Line | Pages | Windows | Calls | OCR s | Model s | Fields s | Total s | Schema valid "
            "| Review flags | Text coverage |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for view in views:
        s = view.get("summary")
        if s is None:
            rows.append(f"| {view['name']} | {view['lob']} | FAILED: {view.get('error', '')[:80]} "
                        "| | | | | | | | | |")
            continue
        t = s["timings_s"]
        rows.append(f"| {s['document']} | {s['lob']} | {s['pages']} | {s['windows_planned']} | {s['model_calls']} "
                    f"| {t.get('ocr', 0):.1f} | {t['model']:.1f} | {view.get('fields_s') or t['total']:.1f} "
                    f"| {t['total']:.1f} | {s['schema_valid']} "
                    f"| {s['review_flags']} | {s['text_coverage']} |")
    return "\n".join([head, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the server's proxy URL, https://<pod id>-<port>.proxy.runpod.net")
    parser.add_argument("--token", default=os.environ.get(TOKEN_ENV),
                        help=f"the server's token (default ${TOKEN_ENV}); on the pod in token.txt")
    parser.add_argument("--doc", dest="docs", action="append", type=parse_doc, required=True,
                        metavar="PDF=LOB", help="a PDF and its line of business; repeat per document")
    parser.add_argument("--mode", default=None, choices=["ocr_plus_image", "image_only", "fields_first"],
                        help="how the server reads each PDF (default the server's: fields_first when it was "
                             "started with --fields-first)")
    parser.add_argument("--poll", type=float, default=15.0, help="seconds between polls")
    parser.add_argument("--out", type=Path, default=None,
                        help="where results go (default testing/results/proxy_check/<time>)")
    args = parser.parse_args(argv)
    if not args.token:
        parser.error(f"--token is required (or set ${TOKEN_ENV})")
    missing = [str(pdf) for pdf, _lob in args.docs if not pdf.is_file()]
    if missing:
        parser.error(f"no such PDF: {', '.join(missing)}")

    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("testing.extract_client", argv, "extract-client"):
        return 0

    base = args.url.rstrip("/")
    out = args.out or ROOT / "results" / "proxy_check" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    health = call(f"{base}/health", args.token)
    print(f"server ready: model {health['model']}")

    jobs: list[str] = []
    for pdf, lob in args.docs:
        size = pdf.stat().st_size / 2**20
        print(f"uploading {pdf.name} ({size:.1f} MB, {lob}) ...", flush=True)
        jobs.append(submit(base, args.token, pdf, lob, args.mode)["job"])
    views = wait(base, args.token, jobs, args.poll)

    ordered = [views[job] for job in jobs]
    for view in ordered:
        if view["status"] == "done":
            result = call(f"{base}/jobs/{view['job']}/result", args.token)
            (out / f"{Path(view['name']).stem}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for view in ordered:
        if view["status"] == "done" and view.get("error"):
            print(f"  {view['name']}: fields served, but {view['error']}", flush=True)
    report = table(ordered)
    (out / "summary.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(report + "\n", encoding="utf-8")
    print("\n" + report + f"\n\nresults in {out}")
    return 0 if all(v["status"] == "done" and not v.get("error") for v in ordered) else 1


if __name__ == "__main__":
    sys.exit(main())
