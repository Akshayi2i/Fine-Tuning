"""Send PDFs to the pod's extraction test server through its RunPod proxy URL, and report.

    python -m testing.extract_client --url https://<pod id>-8000.proxy.runpod.net --token <token> ^
        --doc "D:\\...\\a__scan.pdf=personal_auto" --doc "D:\\...\\b__scan.pdf=homeowners"

Each ``--doc`` is ``<pdf>=<line of business>``. Every line is checked before
anything is sent, as the server checks it (``serving.vllm_entrypoint.request_lob``),
and every PDF against the largest the server takes. The PDFs are posted one
after another and queue on the server, which reads them one at a time
(``testing/extract_server.py``); this polls until every job is done, saves each
served JSON, and prints a table of what came back and how long it took - the
server's own timings, so a document's wait in the queue and the upload are not
counted as its extraction time.

Writes ``<out>/<document>.json`` (the summary and the canonical JSON; a name
used twice, or ``summary``, takes ``-2``) and ``<out>/summary.json`` /
``summary.md`` - always, also when a document was refused or the server went
away: the jobs already queued are still collected. ``testing/results/`` is
git-ignored: the served JSON quotes the document. The exit code is 1 unless
every document was read in full.

Standard library only (the repo's own modules aside), so it runs from the
laptop's environment as it is; without the repo's dependencies the lines are
left to the server to check.
"""

from __future__ import annotations

import argparse
import http.client
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

from common.transport import InsecureUrlError, RedirectRefusedError, assert_encrypted, open_url
from testing.extract_server import MAX_PDF_BYTES, TOKEN_ENV, unique_names

ROOT = Path(__file__).resolve().parent

#: Uploads of the larger scans (84 MB) take a while through the proxy.
UPLOAD_TIMEOUT_S = 900
CALL_TIMEOUT_S = 60

#: A job in one of these is finished.
FINISHED = ("done", "failed")
#: A document this client did not get queued.
NOT_SENT = "not_sent"


class ClientError(RuntimeError):
    """The server refused a call, or could not be reached. ``status`` is the
    HTTP status of a refusal; None when no answer came."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def parse_doc(value: str) -> tuple[Path, str]:
    """``<pdf>=<line>`` - split at the last ``=``, since a Windows path can hold one."""
    path, sep, lob = value.rpartition("=")
    if not sep or not path or not lob.strip():
        raise argparse.ArgumentTypeError(f"--doc takes <pdf>=<line of business>, got {value!r}")
    return Path(path), lob.strip()


def checked_lines(docs: list[tuple[Path, str]]) -> tuple[list[tuple[Path, str]], list[str]]:
    """``docs`` with each line named as the server names it, and the refusals:
    a line the server would refuse is refused before any PDF is sent, not
    after the documents before it are queued. Without the repo's dependencies
    nothing is checked here (the server still checks each)."""
    try:
        from serving.vllm_entrypoint import ServingError, request_lob
    except ImportError:
        print("lines not checked here (the repo's dependencies are not installed): the server checks each",
              file=sys.stderr)
        return docs, []
    out, refused = [], []
    for pdf, lob in docs:
        try:
            out.append((pdf, str(request_lob({"lob": lob}))))
        except ServingError as exc:
            refused.append(f"--doc {pdf}={lob}: {exc}")
    return out, refused


def call(url: str, token: str, *, data: bytes | None = None, timeout: float = CALL_TIMEOUT_S) -> Any:
    """One call to the server, its JSON answer back. A redirect is refused, not
    followed (``common.transport.open_url``): it would take the token and the
    PDF to an address nobody checked. Every failure is a :class:`ClientError`."""
    path = urllib.parse.urlparse(url).path
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/pdf",
        # Some proxies refuse a request with Python's default agent.
        "User-Agent": "fideon-extract-client/1",
    })
    try:
        with open_url(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except OSError:
            detail = ""
        raise ClientError(f"{exc.code} from {path}: {detail}", status=exc.code) from exc
    except RedirectRefusedError as exc:
        raise ClientError(str(exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise ClientError(f"could not reach {url}: {exc.reason}") from exc
    except (OSError, http.client.HTTPException) as exc:
        # Reset, timed out, or closed before an answer: urllib passes these on unwrapped.
        raise ClientError(f"the connection broke during {path}: {type(exc).__name__}: {exc}") from exc
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise ClientError(f"{path} answered with something other than JSON: {body[:120]!r}") from exc


def submit(base: str, token: str, pdf: Path, lob: str, mode: str | None) -> dict[str, Any]:
    query = urllib.parse.urlencode({"lob": lob, "name": pdf.name, **({"mode": mode} if mode else {})})
    return call(f"{base}/jobs?{query}", token, data=pdf.read_bytes(), timeout=UPLOAD_TIMEOUT_S)


def wait(base: str, token: str, jobs: list[str], poll_s: float,
         views: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Each job's final view, polling until none is queued or running.

    ``views`` is updated as each poll answers, so a caller holds the last view
    of every job should a poll fail (:class:`ClientError`)."""
    views = {} if views is None else views
    announced: set[str] = set()
    while True:
        for job in jobs:
            if views.get(job, {}).get("status") not in FINISHED:
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


def read_in_full(view: dict[str, Any]) -> bool:
    """Whether a document came back whole: done, its full text too, its result saved."""
    return view.get("status") == "done" and not view.get("error") and not view.get("result_error")


def outcome(view: dict[str, Any]) -> str:
    """A document's status as the table shows it."""
    status, error = view.get("status"), view.get("error") or ""
    if status == "done" and error:
        # Fields first: the checked fields were served, MinerU's text never joined them.
        said = f"fields served, full text failed: {error.removeprefix('full text: ')[:80]}"
    elif status == "done":
        said = "done"
    elif status == "failed":
        said = f"FAILED: {error[:80]}"
    elif status == NOT_SENT:
        said = f"NOT SENT: {error[:80]}" if error else "NOT SENT"
    else:
        said = f"{status} when this client stopped"
    if view.get("result_error"):
        said += f"; result not saved: {view['result_error'][:60]}"
    return said.replace("|", "/")


def _seconds(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else ""


def table(views: list[dict[str, Any]]) -> str:
    """The run as a markdown table, one row per document. A timing the server
    did not report (a full text that failed has no total) is left blank."""
    head = ("| Document | Line | Status | Pages | Windows | Calls | OCR s | Model s | Fields s | Total s "
            "| Schema valid | Review flags | Text coverage |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for view in views:
        s = view.get("summary") or {}
        t = s.get("timings_s") or {}
        name = s.get("document") or view.get("name", "")
        output = view.get("output")
        if output and Path(output).stem != Path(name).stem:
            name += f" ({output})"                   # two documents of one name: which file is which
        cells = [name, s.get("lob") or view.get("lob", ""), outcome(view), s.get("pages"),
                 s.get("windows_planned"), s.get("model_calls"), _seconds(t.get("ocr")), _seconds(t.get("model")),
                 _seconds(view.get("fields_s") or t.get("total")), _seconds(t.get("total")),
                 s.get("schema_valid"), s.get("review_flags"), s.get("text_coverage")]
        rows.append("| " + " | ".join("" if cell is None else str(cell) for cell in cells) + " |")
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
    try:
        assert_encrypted(args.url, "the extraction test server")
    except InsecureUrlError as exc:
        parser.error(str(exc))
    missing = [str(pdf) for pdf, _lob in args.docs if not pdf.is_file()]
    if missing:
        parser.error(f"no such PDF: {', '.join(missing)}")
    docs, refused = checked_lines(args.docs)
    if refused:
        parser.error("; ".join(refused))

    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("testing.extract_client", argv, "extract-client"):
        return 0

    base = args.url.rstrip("/")
    out = args.out or ROOT / "results" / "proxy_check" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    # One view per document, in the order given: the server's once it is queued.
    views = [{"name": pdf.name, "lob": lob, "status": NOT_SENT, "file": str(pdf), "output": f"{name}.json"}
             for (pdf, lob), name in zip(docs, unique_names([pdf.stem for pdf, _lob in docs]), strict=True)]
    try:
        collect(base, args.token, docs, views, mode=args.mode, poll_s=args.poll, out=out)
    finally:
        report = table(views)
        (out / "summary.json").write_text(json.dumps(views, indent=2), encoding="utf-8")
        (out / "summary.md").write_text(report + "\n", encoding="utf-8")
        print("\n" + report + f"\n\nresults in {out}")
    return 0 if all(read_in_full(view) for view in views) else 1


def collect(base: str, token: str, docs: list[tuple[Path, str]], views: list[dict[str, Any]], *,
            mode: str | None, poll_s: float, out: Path) -> None:
    """Post ``docs``, wait for their jobs and save each result, updating ``views``
    as it goes. A failure is reported and stops what follows it, never what came
    before: the jobs already queued are still waited for and saved."""

    def stopped(what: str, exc: ClientError) -> None:
        print(f"extract_client: {what}: {exc}", file=sys.stderr, flush=True)

    try:
        health = call(f"{base}/health", token)
    except ClientError as exc:
        for view in views:
            view["error"] = str(exc)
        stopped("the server could not be reached", exc)
        return
    print(f"server ready: model {health.get('model')}")
    cap = int(health.get("max_pdf_bytes") or MAX_PDF_BYTES)
    over = [f"{pdf.name} ({pdf.stat().st_size / 2**20:.1f} MB)" for pdf, _lob in docs if pdf.stat().st_size > cap]
    if over:
        for view in views:
            view["error"] = f"over the server's limit of {cap / 2**20:.0f} MB: {', '.join(over)}"
        print(f"extract_client: nothing sent: the server takes a PDF of up to {cap / 2**20:.0f} MB; "
              f"over it: {', '.join(over)}", file=sys.stderr)
        return

    jobs: list[str] = []
    for (pdf, lob), view in zip(docs, views, strict=True):
        size = pdf.stat().st_size / 2**20
        print(f"uploading {pdf.name} ({size:.1f} MB, {lob}) ...", flush=True)
        try:
            posted = submit(base, token, pdf, lob, mode)
        except ClientError as exc:
            view["error"] = str(exc) if exc.status else (
                f"{exc} (a server or proxy may close the connection on a body it will not take: this server "
                f"takes up to {cap / 2**20:.0f} MB, a RunPod proxy about 100 MB)")
            stopped(f"{pdf.name} was not queued; the documents after it are not sent, those before it are "
                    "still collected", exc)
            break
        jobs.append(posted["job"])
        view.update({k: v for k, v in posted.items() if k not in ("poll", "result")})

    polled: dict[str, dict[str, Any]] = {}
    try:
        wait(base, token, jobs, poll_s, polled)
    except ClientError as exc:
        stopped("polling stopped; each job is reported as last seen", exc)
    for view in views:
        if view.get("job") in polled:
            view.update(polled[view["job"]])

    for view in views:
        if view["status"] != "done":
            continue
        try:
            result = call(f"{base}/jobs/{view['job']}/result", token)
        except ClientError as exc:
            view["result_error"] = str(exc)
            continue
        (out / view["output"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if view.get("error"):
            print(f"  {view['name']}: fields served, but {view['error']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
