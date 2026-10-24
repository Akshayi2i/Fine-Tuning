"""A served MinerU 4 and a served model that misbehave, against fake local
servers: pages left out, another MinerU mid-PDF, transient failures, malformed
answers, redirects, cut streams - and the tokens and pages that must not leak."""

from __future__ import annotations

import functools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from data_pipeline.ocr import mineru_api
from data_pipeline.ocr.mineru_api import MinerUApiError, ocr_via_api
from data_pipeline.ocr.served_ocr import ServedOcr, ServedOcrError
from inference_core.model_runner import STREAM_CUT, Generation, ModelRunnerError
from inference_core.runner_config import RunnerConfig

PIN = "4.0.12"


@pytest.fixture(autouse=True)
def _quick(monkeypatch):
    """Polls and retries in milliseconds, not seconds."""
    monkeypatch.setattr(mineru_api, "RETRY_WAITS_S", (0.01, 0.01, 0.01))
    monkeypatch.setattr(mineru_api, "ocr_via_api", functools.partial(mineru_api.ocr_via_api, poll_s=0.01))


def _serve(handler, host: str = "127.0.0.1"):
    httpd = ThreadingHTTPServer((host, 0), handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}"


def _pdf(pages: int) -> bytes:
    import pymupdf

    with pymupdf.open() as doc:
        for number in range(1, pages + 1):
            doc.new_page().insert_text((72, 72), f"Page {number}. Policy Number PA-1.", fontsize=12)
        return doc.tobytes()


def _texts(pdf: bytes) -> list[str]:
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_text().strip().split(".")[0] for page in doc]


class _Fake:
    """A MinerU 4 whose every answer a test can change. Each hook takes the task:
    ``{"id", "n" (1 for the first task posted), "pdf", "polls"}``."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.status = lambda task: "completed" if task["polls"] >= 1 else "running"
        self.version = lambda task: PIN
        self.tier = lambda task: "basic"
        #: The task's structured_content pages; by default one per page of its PDF.
        self.pages = lambda task: [{"page_idx": i, "blocks": [{"type": "text", "content": text}]}
                                   for i, text in enumerate(_texts(task["pdf"]))]
        #: ``(status, body)`` to answer with instead, ``"drop"`` to hang up, or None.
        self.post_reply = lambda task: None
        self.poll_reply = lambda task: None
        self.result_reply = lambda task: None
        self.tokens: list[str | None] = []

    def handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status: int, body: Any) -> None:
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _reply(self, reply: Any, default: Any) -> None:
                if reply == "drop":
                    self.close_connection = True              # no answer at all: the client sees the line drop
                elif reply is not None:
                    self._send(*reply)
                else:
                    self._send(*default)

            def do_POST(self):  # noqa: N802
                pdf = self.rfile.read(int(self.headers["Content-Length"]))
                with fake.lock:
                    task = {"id": f"job_{len(fake.tasks) + 1}", "n": len(fake.tasks) + 1, "pdf": pdf, "polls": 0}
                    fake.tasks[task["id"]] = task
                    fake.tokens.append(self.headers.get("Authorization"))
                self._reply(fake.post_reply(task), (202, {"task_id": task["id"], "status": "queued"}))

            def do_GET(self):  # noqa: N802
                fake.tokens.append(self.headers.get("Authorization"))
                task = fake.tasks[self.path.split("/v1/tasks/", 1)[1].split("/", 1)[0]]
                if self.path.endswith("result?response_format=json"):
                    self._reply(fake.result_reply(task), (200, {"tier": fake.tier(task), "files": [{
                        "parse": {"duration_ms": 1000, "parser_version": fake.version(task)},
                        "content": {"structured_content": {"pages": fake.pages(task)}}}]}))
                    return
                task["polls"] += 1
                self._reply(fake.poll_reply(task), (200, {"task_id": task["id"], "status": fake.status(task)}))

        return Handler


@pytest.fixture
def mineru():
    fake = _Fake()
    httpd, url = _serve(fake.handler())
    yield fake, url
    httpd.shutdown()


def _drop_last_page(task):
    return [{"page_idx": i, "blocks": [{"type": "text", "content": text}]}
            for i, text in enumerate(_texts(task["pdf"]))][:-1]


# --------------------------------------------------------------------------
# Every run held to its pages and to the pin
# --------------------------------------------------------------------------


def test_a_page_one_run_leaves_out_is_refused_not_read_as_blank(mineru):
    fake, url = mineru
    fake.pages = _drop_last_page
    reader = ServedOcr(url, run_pages=2, render_workers=1)
    # Runs read one at a time (a document's full text in the GPU's idle turns),
    with pytest.raises(MinerUApiError, match="returned no page 2 of the 2 it was sent"):
        reader.ocr_text(_pdf(6), page_count=6, gate=lambda: None)
    # runs read at once (ocr_parts), and the pages a document's fields cite.
    with pytest.raises(MinerUApiError, match="returned no page 2 of the 2"):
        ocr_via_api(_pdf(6), url, parts=3, poll_s=0.01)
    with pytest.raises(MinerUApiError, match="returned no page 2 of the 2"):
        reader.ocr_text(_pdf(8), pages=[3, 4, 5, 6, 7, 8], gate=lambda: None)
    # One task with a page missing from its middle: as many entries as pages
    # would not show it; the page numbers do.
    fake.pages = lambda task: [{"page_idx": i, "blocks": [{"type": "text", "content": "x"}]} for i in (0, 2)]
    with pytest.raises(MinerUApiError, match="returned no page 2 of the 2"):
        ocr_via_api(_pdf(3), url, poll_s=0.01)

    fake.pages = _Fake().pages                                            # every page read: accepted
    text = reader.ocr_text(_pdf(8), pages=[3, 4, 5, 6, 7, 8], gate=lambda: None)
    assert [text["texts"][n] for n in (3, 8)] == ["Page 3", "Page 8"]


def test_every_run_is_held_to_the_pinned_version_and_tier(mineru):
    fake, url = mineru
    fake.version = lambda task: PIN if task["n"] == 1 else "4.1.0"
    with pytest.raises(ServedOcrError, match=r"MinerU 4\.1\.0 \(tier basic\) \(run 2 of 2\)"):
        ServedOcr(url, ocr_parts=2, expected_version=PIN).ocr_text(_pdf(4), page_count=4)
    fake.tasks.clear()
    fake.version, fake.tier = (lambda task: PIN), (lambda task: "basic" if task["n"] < 3 else "flash")
    with pytest.raises(ServedOcrError, match=r"tier flash\) \(run 3 of 3\)"):
        ServedOcr(url, run_pages=2, expected_version=PIN).ocr_text(_pdf(6), page_count=6, gate=lambda: None)
    fake.tasks.clear()
    _pages, info = ocr_via_api(_pdf(6), url, run_pages=2, gate=lambda: None, poll_s=0.01)
    assert info["version_parts"] == [PIN] * 3 and info["tier_parts"] == ["basic", "basic", "flash"]
    assert info["version"] == PIN and info["tier"] == "basic / flash"


def test_a_reply_naming_no_tier_is_refused_under_the_pin(mineru):
    fake, url = mineru
    fake.tier = lambda task: None
    with pytest.raises(ServedOcrError, match="tier not reported"):
        ServedOcr(url, expected_version=PIN).ocr_text(_pdf(2), page_count=2)
    assert len(ServedOcr(url).ocr_text(_pdf(2), page_count=2)["pages"]) == 2      # no pin: a timing test


def test_the_tier_is_held_to_the_pin_not_to_the_tier_asked_for(mineru):
    from common.config import served_mineru

    fake, url = mineru
    assert served_mineru()["tier"] == "basic"
    with pytest.raises(ServedOcrError, match="tier flash is asked of MinerU.*tier basic.*--any-mineru"):
        ServedOcr(url, tier="flash", expected_version=PIN)
    fake.tier = lambda task: "flash"
    assert ServedOcr(url, tier="flash").ocr_text(_pdf(2))["info"]["tier"] == "flash"   # --any-mineru: anything
    with pytest.raises(ServedOcrError, match=r"tier flash\) at .*\(tier basic\)"):
        ServedOcr(url, expected_version=PIN).ocr_text(_pdf(2))            # asked for basic, read at flash
    # A pin of another tier is a deliberate one (ServedMinerUEngine(tier=...)).
    assert ServedOcr(url, tier="flash", expected_version=PIN, expected_tier="flash").ocr_text(_pdf(2))["pages"]


# --------------------------------------------------------------------------
# Polling: hiccups, failures, statuses, malformed answers
# --------------------------------------------------------------------------


def test_a_poll_or_a_result_fetch_survives_a_hiccup_but_an_upload_is_sent_once(mineru):
    fake, url = mineru
    fake.status = lambda task: "completed" if task["polls"] >= 5 else "running"
    hiccups = {1: (502, b"upstream connect error"), 2: (200, b"<html>Bad Gateway</html>"), 3: "drop"}
    fake.poll_reply = lambda task: hiccups.get(task["polls"])
    fetches = []
    fake.result_reply = lambda task: (503, b"busy") if not fetches.append(1) and len(fetches) == 1 else None
    pages, _info = ocr_via_api(_pdf(2), url, poll_s=0.01)
    assert pages == ["Page 1", "Page 2"] and len(fetches) == 2

    fake.poll_reply = lambda task: (502, b"down for good")                  # not a hiccup: said, after a few tries
    with pytest.raises(MinerUApiError, match="502 from /v1/tasks/job_2: down for good .tried 4 times"):
        ocr_via_api(_pdf(2), url, poll_s=0.01)
    fake.post_reply = lambda task: (502, b"upload failed")                  # a POST may have reached the server
    with pytest.raises(MinerUApiError, match="502 from /v1/tasks: upload failed"):
        ocr_via_api(_pdf(2), url, poll_s=0.01)
    assert len(fake.tasks) == 3                                             # the upload: sent once, not again
    fake.post_reply = lambda task: None
    fake.poll_reply = lambda task: (404, b"no such task")                   # an answer, not a hiccup
    started = time.perf_counter()
    with pytest.raises(MinerUApiError, match="404"):
        ocr_via_api(_pdf(2), url, poll_s=0.01)
    assert fake.tasks["job_4"]["polls"] == 1 and time.perf_counter() - started < 2


def test_the_first_run_to_fail_fails_the_pdf_at_once_and_stops_the_others(mineru):
    fake, url = mineru
    fake.status = lambda task: "failed" if task["n"] == 3 else "running"   # runs 1-2 would never end
    started = time.perf_counter()
    with pytest.raises(MinerUApiError, match="task job_3 failed"):
        ocr_via_api(_pdf(6), url, parts=3, poll_s=0.01, timeout_s=30)
    assert time.perf_counter() - started < 5
    polls = {task["id"]: task["polls"] for task in fake.tasks.values()}
    time.sleep(0.2)
    assert {task["id"]: task["polls"] for task in fake.tasks.values()} == polls      # nobody polls on


@pytest.mark.parametrize("status", ["cancelled", "error", "aborted"])
def test_a_task_ended_by_a_word_minerU_4_does_not_use_fails_at_once(mineru, status):
    fake, url = mineru
    fake.status = lambda task: status
    started = time.perf_counter()
    with pytest.raises(MinerUApiError, match=f"task job_1 {status}"):
        ocr_via_api(_pdf(2), url, poll_s=0.01, timeout_s=30)
    assert time.perf_counter() - started < 5


def test_a_status_not_known_but_not_an_ending_is_polled_on(mineru):
    fake, url = mineru
    fake.status = lambda task: "completed" if task["polls"] >= 3 else "Parsing"
    assert len(ocr_via_api(_pdf(2), url, poll_s=0.01)[0]) == 2


@pytest.mark.parametrize(("where", "reply", "said"), [
    ("poll", {"status": "failed", "files": []}, "task job_1 failed: no reason given"),
    ("poll", {"status": "failed", "files": None}, "task job_1 failed: no reason given"),
    ("poll", {"status": "failed", "error": "out of memory"}, "task job_1 failed: out of memory"),
    ("poll", {"task_id": "job_1"}, "names no status"),
    ("poll", ["completed"], "names no status"),
    ("post", {"id": "job_1"}, "named no task_id"),
    ("result", {"tier": "basic", "files": []}, "its result holds no file"),
    ("result", {"tier": "basic", "files": [{"content": {"structured_content": {"pages": 7}}}]},
     "structured_content this client cannot read"),
    ("result", {"tier": "basic", "files": [{"content": {"structured_content": "{}"}}]},
     "returned no structured_content"),
])
def test_an_answer_this_client_cannot_read_is_said_with_the_answer(mineru, where, reply, said):
    fake, url = mineru
    answer = (200, json.dumps(reply).encode())
    setattr(fake, f"{where}_reply", lambda task: answer)
    with pytest.raises(MinerUApiError, match=said):
        ocr_via_api(_pdf(2), url, poll_s=0.01, timeout_s=10)


def test_a_failed_rendering_stops_minerUs_reading(mineru, monkeypatch, tmp_path):
    from data_pipeline.ocr import render_only

    fake, url = mineru
    fake.status = lambda task: "running"                                   # a task that never ends

    def broken(*args, **kwargs):
        time.sleep(0.2)                                                  # MinerU has the PDF by now
        raise RuntimeError("the renderer broke")

    monkeypatch.setattr(render_only, "render_pdf_pages", broken)
    with pytest.raises(RuntimeError, match="the renderer broke"):
        ServedOcr(url, render_workers=1).read(_pdf(2), tmp_path / "pages")
    readers = [thread for thread in threading.enumerate() if thread.name == "served-ocr"]
    for thread in readers:
        thread.join(timeout=3)
    assert not any(thread.is_alive() for thread in readers)               # it no longer polls for an hour


# --------------------------------------------------------------------------
# Form fields, redirects, cut streams, page links
# --------------------------------------------------------------------------


def _form(pages: int) -> bytes:
    import pymupdf

    with pymupdf.open() as doc:
        for number in range(1, pages + 1):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {number} Named Insured:", fontsize=12)
            field = pymupdf.Widget()
            field.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
            field.field_name = f"insured_{number}"
            field.field_value = f"ACME HOLDINGS {number}"
            field.rect = pymupdf.Rect(220, 60, 420, 80)
            page.add_widget(field)
        return doc.tobytes()


def test_the_pages_a_documents_fields_cite_keep_their_form_fields():
    """MinerU renders a page with its form fields drawn from the AcroForm: a page
    sent without it has the field's box and none of its typed-in value."""
    import pymupdf

    from data_pipeline.ocr.mineru_api import select_pages, split_pdf

    for label, data in (("selected", select_pages(_form(4), [4, 2])), ("split", split_pdf(_form(4), 2)[1][1])):
        with pymupdf.open(stream=data, filetype="pdf") as picked:
            values = [[field.field_value for field in page.widgets()] for page in picked]
            assert picked.is_form_pdf == 2, label                         # the AcroForm came along
        assert values == ([["ACME HOLDINGS 4"], ["ACME HOLDINGS 2"]] if label == "selected"
                          else [["ACME HOLDINGS 3"], ["ACME HOLDINGS 4"]])
    with pytest.raises(MinerUApiError, match="no pages to read"):
        select_pages(_form(2), [])


class _Redirecting(BaseHTTPRequestHandler):
    """Answers everything with a redirect to ``target``."""

    target = ""

    def log_message(self, *args):
        pass

    def _redirect(self):
        self.send_response(307)
        self.send_header("Location", _Redirecting.target + self.path)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _redirect  # noqa: N815


def test_a_redirect_is_refused_and_the_token_never_follows_it(mineru):
    from inference_core.openai_backend import OpenAIChatBackend

    fake, elsewhere = mineru                                               # where the redirect would lead
    _Redirecting.target = elsewhere
    httpd, url = _serve(_Redirecting)
    try:
        with pytest.raises(MinerUApiError, match="redirected .307. to http://127.0.0.1:.* not followed"):
            ocr_via_api(_pdf(2), url, "SECRET", poll_s=0.01)
        with pytest.raises(ModelRunnerError, match="redirected .307.* not followed"):
            OpenAIChatBackend(url, "SECRET")                                 # GET /v1/models
    finally:
        httpd.shutdown()
    assert fake.tasks == {} and fake.tokens == []                          # nothing, and no token, went there


class _CutStream(BaseHTTPRequestHandler):
    """A model server whose answer stream breaks off: ``mode`` ``closed`` ends it
    with no finish reason and no [DONE]; ``chunked`` cuts it mid-chunk."""

    mode = "closed"

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        data = json.dumps({"data": [{"id": "served", "root": "/m", "parent": None}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        events = b"".join(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode()
                          for piece in ('{"policy": ', '{"named_insured": "AC'))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.close_connection = True
        if _CutStream.mode == "chunked":
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(events), events))
            self.wfile.write(b"400\r\ndata: {\"choi")                     # promises 1024 bytes, sends 14
        else:
            self.end_headers()
            self.wfile.write(events)


def test_a_stream_that_breaks_off_counts_as_cut_off_never_as_a_finished_answer(tmp_path):
    from inference_core.openai_backend import OpenAIChatBackend, _read_stream

    def sse(chunks, done):
        return iter([f"data: {json.dumps(c)}\n".encode() for c in chunks] + ([b"data: [DONE]\n"] if done else []))

    pieces = [{"choices": [{"delta": {"content": '{"a": 1}'}}]}]
    finished = [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    assert _read_stream(sse(pieces, done=False), started=0)["finish_reason"] == STREAM_CUT
    assert _read_stream(sse(pieces, done=True), started=0)["finish_reason"] == "stop"
    assert _read_stream(sse(pieces + finished, done=False), started=0)["finish_reason"] == "stop"
    assert Generation(text="{", finish_reason=STREAM_CUT).truncated()        # the window is read again, smaller

    httpd, url = _serve(_CutStream)
    try:
        backend = OpenAIChatBackend(url, "tok")
        messages = [{"role": "user", "content": "read it"}]
        _CutStream.mode = "closed"
        cut = backend.generate(messages, RunnerConfig(max_new_tokens=100))
        assert cut.finish_reason == STREAM_CUT and cut.truncated() and cut.text.endswith('"AC')
        _CutStream.mode = "chunked"
        with pytest.raises(ModelRunnerError, match="broke off.*IncompleteRead"):
            backend.generate(messages, RunnerConfig(max_new_tokens=100))
    finally:
        httpd.shutdown()


def test_a_link_token_must_be_read_only_and_for_one_container_or_blob():
    from artifact_registry.blob_client import BlobError, read_only_sas

    for token in ("?sv=2024-11-04&sr=c&sp=r&se=2026-10-25&sig=abc", "sv=2024-11-04&sr=b&sp=r&sig=abc"):
        assert read_only_sas(token) == token.lstrip("?")
    for token, said in (
        ("sv=2024-11-04&ss=b&srt=sco&sp=r&sig=abc", "account SAS"),          # every container, raw-documents too
        ("sv=2024&srt=sco&sp=r&sr=c&sig=abc", "account SAS"),
        ("sv=2024&SS=b&sp=r&sr=c&sig=abc", "account SAS"),
        ("sv=2024&sr=c&sp=r&sp=rwdl&sig=abc", "sp more than once"),
        ("sv=2024&sr=c&sp=rl&sig=abc", "read only"),
        ("sv=2024&sp=r&sig=abc", r"sr=c\) or one blob \(sr=b\), this one is for sr=\(none\)"),
        ("sv=2024&sr=d&sp=r&sig=abc", "this one is for sr=d"),
    ):
        with pytest.raises(BlobError, match=said):
            read_only_sas(token)


def test_a_gray_page_with_a_transparent_colour_goes_as_it_is_on_disk(tmp_path):
    from PIL import Image, ImageDraw

    from inference_core.page_links import sent_png

    image = Image.new("RGB", (40, 30), "white")
    ImageDraw.Draw(image).rectangle((5, 5, 20, 20), fill=(90, 90, 90))
    path = tmp_path / "page_001.png"
    image.save(path, transparency=(255, 255, 255))                          # a tRNS chunk: white is see-through
    assert sent_png(path) == path.read_bytes()
