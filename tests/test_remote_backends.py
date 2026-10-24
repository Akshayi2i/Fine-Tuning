"""A served model (inference_core.openai_backend) and a served MinerU 4
(data_pipeline.ocr.mineru_api), against fake local servers."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from data_pipeline.ocr.mineru_api import MinerUApiError, ocr_via_api, pages_from_structured_content
from inference_core.model_runner import STOPPED_REPEATING, ModelRunnerError
from inference_core.openai_backend import OpenAIChatBackend, chat_messages, repeating_tail
from inference_core.runner_config import RunnerConfig

ANSWER = '{"policy": {"policy_number": {"raw": "PA-1", "parsed": "PA-1", "page_ref": [1]}}}'
#: An answer stuck in a loop: the same row, on and on, to the token limit.
LOOP_ROW = '{"raw": "Comprehensive", "parsed": "Comprehensive", "page_ref": [3]}, '
LOOP_ROWS = 20000


def _serve(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


BASE_ONLY = [{"id": "Qwen3-VL-8B-Instruct", "root": "/workspace/models/Qwen3-VL-8B-Instruct", "parent": None}]


class _Vllm(BaseHTTPRequestHandler):
    seen: list = []
    #: What GET /v1/models lists: the base, and a LoRA with its base as parent.
    models: list = BASE_ONLY

    def log_message(self, *args):
        pass

    def _json(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self._json({"data": _Vllm.models})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Vllm.seen.append((self.headers.get("Authorization"), body))
        if body["max_tokens"] == 13:
            self._json({"error": "decoder prompt too long"}, status=400)
            return
        if body["max_tokens"] == 777:
            self._loop()
            return
        # Split into tokens, with the end-of-turn token the server scores too.
        tokens = [ANSWER[i:i + 5] for i in range(0, len(ANSWER), 5)] + ["<|im_end|>"]
        if not body.get("stream"):
            self._json({"choices": [{"message": {"content": ANSWER}, "finish_reason": "stop",
                                     "logprobs": {"content": [{"token": t, "logprob": -0.1} for t in tokens]}}]})
            return
        # As vLLM streams: a role chunk first, a chunk per token, the finish, [DONE].
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{"choices": [{"delta": {"role": "assistant"}}]}]
        chunks += [{"choices": [{"delta": {"content": "" if t == "<|im_end|>" else t},
                                 "logprobs": {"content": [{"token": t, "logprob": -0.1}]}}]} for t in tokens]
        chunks += [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks += [{"choices": [], "usage": {"prompt_tokens": 1234, "completion_tokens": len(tokens)}}]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _loop(self):
        """Streams the start of an answer, then one row over and over."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for piece in ['{"coverages": ['] + [LOOP_ROW] * LOOP_ROWS:
                chunk = {"choices": [{"delta": {"content": piece},
                                      "logprobs": {"content": [{"token": piece, "logprob": -0.1}]}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        except OSError:
            pass                                                         # the client hung up


@pytest.fixture
def vllm():
    _Vllm.seen, _Vllm.models = [], BASE_ONLY
    httpd, url = _serve(_Vllm)
    yield url
    httpd.shutdown()


def _messages(tmp_path):
    image = tmp_path / "page_001.png"
    image.write_bytes(b"\x89PNG fake")
    return [{"role": "system", "content": "rules"},
            {"role": "user", "content": [{"type": "image", "image": str(image)},
                                         {"type": "text", "text": "<page 1 of 1>\n\nPolicy Number PA-1"}]}]


@pytest.mark.parametrize("stream", [True, False])
def test_a_window_is_sent_as_a_chat_request_and_its_tokens_spell_the_answer(vllm, tmp_path, stream):
    backend = OpenAIChatBackend(vllm, "tok", stream=stream)
    assert backend.model == "Qwen3-VL-8B-Instruct"
    config = RunnerConfig(max_new_tokens=4096, json_schema={"type": "object"})
    result = backend.generate(_messages(tmp_path), config)
    assert result.text == ANSWER and "".join(result.tokens) == ANSWER     # the end token is dropped
    assert len(result.tokens) == len(result.token_logprobs) and result.finish_reason == "stop"
    auth, body = _Vllm.seen[-1]
    assert body.get("stream", False) is stream
    assert auth == "Bearer tok" and body["max_tokens"] == 4096 and body["temperature"] == 0.0
    assert body["logprobs"] is True and body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert set(body["mm_processor_kwargs"]) == {"min_pixels", "max_pixels"}
    assert backend.sent_bytes > 0
    if stream:   # the prompt's length comes in the last chunk; the first token's time is measured
        assert body["stream_options"] == {"include_usage": True}
        assert result.prompt_tokens == 1234 and result.first_token_ms is not None


def test_an_answer_repeating_itself_is_stopped_early_and_counts_as_cut_off(vllm, tmp_path):
    backend = OpenAIChatBackend(vllm, "tok")
    result = backend.generate(_messages(tmp_path), RunnerConfig(max_new_tokens=777))
    assert result.finish_reason == STOPPED_REPEATING and result.truncated()
    assert len(result.text) < 100 * len(LOOP_ROW) < LOOP_ROWS * len(LOOP_ROW)   # long before the end


def test_a_loop_is_told_from_rows_that_differ():
    rows = "".join(f'{{"raw": "Vehicle {n}", "page_ref": [{n % 9 + 1}]}}, ' for n in range(200))
    assert repeating_tail(rows) is None                                  # real rows differ
    assert repeating_tail(rows + LOOP_ROW * 60) == len(LOOP_ROW)
    assert repeating_tail(rows + " " * 3001) == 1                        # blank space to the limit
    assert repeating_tail(rows + LOOP_ROW * 3) is None                   # a few repeats are allowed


def test_page_images_can_go_by_their_path_on_the_server(tmp_path):
    messages = chat_messages(_messages(tmp_path), images="file")
    url = messages[1]["content"][0]["image_url"]["url"]
    assert url.startswith("file:///") and url.endswith("page_001.png")
    with pytest.raises(ValueError, match="images must be one of"):
        OpenAIChatBackend("http://127.0.0.1:9", model="served", images="ftp")


def _page(path, color=False):
    """A rendered page: RGB, gray unless ``color`` (a red stamp in one corner)."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (120, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Policy Number PA-1", fill=(40, 40, 40))
    draw.rectangle((20, 40, 100, 60), fill=(128, 128, 128))
    if color:
        draw.rectangle((100, 140, 119, 159), fill=(200, 0, 0))
    image.save(path)
    return path


def _decoded(url):
    import base64
    import io

    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_a_gray_page_goes_as_a_grayscale_png_with_the_same_pixels(tmp_path):
    from PIL import Image, ImageChops

    from inference_core.page_links import sent_png

    gray, color = _page(tmp_path / "page_001.png"), _page(tmp_path / "page_002.png", color=True)
    messages = [{"role": "user", "content": [{"type": "image", "image": str(gray)},
                                             {"type": "image", "image": str(color)}]}]
    first, second = (part["image_url"]["url"] for part in chat_messages(messages)[0]["content"])
    sent = _decoded(first)
    assert sent.mode == "L"
    # What the model reads: the server makes every image RGB - the same pixels as rendered.
    assert ImageChops.difference(sent.convert("RGB"), Image.open(gray).convert("RGB")).getbbox() is None
    # A page with any colour goes as it is on disk; so does every page with gray off.
    assert _decoded(second).mode == "RGB" and sent_png(color) == color.read_bytes()
    plain = chat_messages(messages, gray=False)[0]["content"][0]["image_url"]["url"]
    assert _decoded(plain).mode == "RGB"


def test_pages_go_once_as_read_only_links_and_are_deleted_after(vllm, tmp_path):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.page_links import PageLinks

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    links = PageLinks(client, "transient/page-links/run-1", workers=2)
    pages = [str(_page(tmp_path / f"page_00{n}.png")) for n in (1, 2)]
    published = links.publish(pages, folder="a__scan")
    assert published["pages"] == 2 and published["mb"] >= 0
    held = client.list("transient/page-links/run-1")
    assert held == ["transient/page-links/run-1/a__scan/page_001.png", "transient/page-links/run-1/a__scan/page_002.png"]
    assert client.read_bytes(held[0]).startswith(b"\x89PNG")                # the grayscale page, uploaded once

    backend = OpenAIChatBackend(vllm, "tok", images="link", links=links)
    messages = [{"role": "user", "content": [{"type": "image", "image": pages[1]},
                                             {"type": "image", "image": str(_page(tmp_path / "page_003.png"))},
                                             {"type": "text", "text": "<page 2 of 3>"}]}]
    backend.generate(messages, RunnerConfig(max_new_tokens=100))
    urls = [part["image_url"]["url"] for part in _Vllm.seen[-1][1]["messages"][0]["content"][:2]]
    assert urls[0] == "memory://main/transient/page-links/run-1/a__scan/page_002.png"
    assert urls[1].startswith("data:image/png;base64,")                     # a page with no link goes inline

    links.discard()
    assert client.list("transient/page-links/run-1") == [] and links.url(pages[0]) is None
    with pytest.raises(ValueError, match="needs links"):
        OpenAIChatBackend(vllm, model="served", images="link")


def test_a_link_carries_a_read_only_token_and_never_reaches_raw_documents():
    from artifact_registry.blob_client import (
        AccessDeniedError,
        BlobClient,
        BlobError,
        InMemoryBackend,
        read_only_sas,
    )

    assert read_only_sas("?sv=2024-11-04&sp=r&se=2026-10-24&sig=abc") == "sv=2024-11-04&sp=r&se=2026-10-24&sig=abc"
    for token in ("sv=2024&sp=rw&sig=abc", "sv=2024&sp=racwdl&sig=abc", "sv=2024&sig=abc"):
        with pytest.raises(BlobError, match="read only"):
            read_only_sas(token)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    with pytest.raises(BlobError, match="read only"):
        client.read_link("transient/page-links/x/page_001.png", read_token="sp=rw&sig=abc")
    with pytest.raises(AccessDeniedError, match="never handed out"):
        client.read_link("raw-documents/t/s/original.pdf")


def test_azure_signs_a_read_only_link_with_the_account_key_and_never_reuses_a_sas_login():
    pytest.importorskip("azure.storage.blob")
    import base64
    from urllib.parse import parse_qs, urlsplit

    from artifact_registry.blob_client import AzureBackend, BlobError

    key = base64.b64encode(b"k" * 64).decode()
    keyed = AzureBackend(f"DefaultEndpointsProtocol=https;AccountName=acct;AccountKey={key};"
                         "EndpointSuffix=core.windows.net")                    # signs offline: no request made
    link = urlsplit(keyed.read_link("main", "transient/page-links/r/a/page_001.png", expiry_s=600))
    assert (link.netloc, link.path) == ("acct.blob.core.windows.net", "/main/transient/page-links/r/a/page_001.png")
    query = parse_qs(link.query)
    assert query["sp"] == ["r"] and query["sr"] == ["b"] and "sig" in query       # this blob alone, read only

    login = AzureBackend(account_url="https://acct.blob.core.windows.net", credential="sv=2024&sp=racwdl&sig=LOGIN")
    with pytest.raises(BlobError, match="cannot sign"):
        login.read_link("main", "transient/page-links/r/a/page_001.png", expiry_s=600)
    given = login.read_link("main", "transient/page-links/r/a/page_001.png", expiry_s=600, read_token="sp=r&sig=RO")
    assert given.endswith("/main/transient/page-links/r/a/page_001.png?sp=r&sig=RO") and "LOGIN" not in given


def test_a_batch_runs_together_and_one_refused_window_keeps_its_own_slot(vllm, tmp_path):
    backend = OpenAIChatBackend(vllm, "tok", max_concurrency=4)
    configs = [RunnerConfig(max_new_tokens=n) for n in (100, 13, 100)]
    results = backend.generate_batch([_messages(tmp_path)] * 3, configs)
    assert [type(r).__name__ for r in results] == ["Generation", "ModelRunnerError", "Generation"]
    assert "400" in str(results[1])


def test_an_adapter_is_refused_and_an_unknown_block_is_said(tmp_path):
    backend = OpenAIChatBackend("http://127.0.0.1:9", model="served")      # never reached
    with pytest.raises(ModelRunnerError, match="no per-request adapter"):
        backend.generate(_messages(tmp_path), RunnerConfig(), adapter="checkpoint-280")
    with pytest.raises(ModelRunnerError, match="no chat form"):
        chat_messages([{"role": "user", "content": [{"type": "video", "video": "x"}]}])


# --------------------------------------------------------------------------
# MinerU 4
# --------------------------------------------------------------------------

STRUCTURED = {"pages": [
    {"page_idx": 0, "blocks": [
        {"type": "page_number", "content": "Page 1 of 2"},
        {"type": "doc_title", "level": 1, "content": "Personal Auto Declarations"},
        {"type": "header", "content": "Plymouth Rock"},
        {"type": "table", "content": "| A | B |\n| --- | --- |\n| 1 | 2 |",
         "captions": [{"bbox": [0.06, 0.11, 0.37, 0.13], "content": "Vehicles"}], "footnotes": []},
        {"type": "image", "content": "", "captions": ["Logo"]},
    ]},
    {"page_idx": 1, "blocks": [{"type": "paragraph_title", "level": 2, "content": "Conditions"},
                               {"type": "text", "content": "We will pay."}]},
]}


def test_structured_content_reads_as_the_ocr_stage_writes_a_page():
    first, second = pages_from_structured_content(STRUCTURED)
    assert first.split("\n\n") == ["Plymouth Rock", "# Personal Auto Declarations", "Vehicles",
                                   "| A | B |\n| --- | --- |\n| 1 | 2 |", "Logo", "Page 1 of 2"]
    assert second == "## Conditions\n\nWe will pay."


class _MinerU(BaseHTTPRequestHandler):
    polls = 0
    posted: list = []

    def log_message(self, *args):
        pass

    def _json(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        _MinerU.posted.append(self.path)
        failed = "broken" in self.path
        self._json({"task_id": "job_failed" if failed else "job_1", "status": "queued"}, status=202)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/v1/tasks/job_failed"):
            self._json({"task_id": "job_failed", "status": "failed", "files": [{"error": {"message": "bad pdf"}}]})
        elif self.path.endswith("result?response_format=json"):
            self._json({"tier": "basic", "files": [{"parse": {"duration_ms": 8250, "parser_version": "4.0.12"},
                                                    "content": {"structured_content": STRUCTURED}}]})
        else:
            _MinerU.polls += 1
            self._json({"task_id": "job_1", "status": "running" if _MinerU.polls < 2 else "completed"})


def test_a_pdf_is_read_through_the_task_api():
    httpd, url = _serve(_MinerU)
    try:
        pages, info = ocr_via_api(b"%PDF", url, "tok", filename="a.pdf", poll_s=0.01)
        assert len(pages) == 2 and pages[1].startswith("## Conditions")
        assert info == {"task_id": "job_1", "pages": 2, "server_seconds": 8.25, "version": "4.0.12", "tier": "basic"}
        with pytest.raises(MinerUApiError, match="failed"):
            ocr_via_api(b"%PDF", url, "tok", filename="broken.pdf", poll_s=0.01)
    finally:
        httpd.shutdown()


def _pdf(pages: int) -> bytes:
    import pymupdf

    with pymupdf.open() as doc:
        for number in range(1, pages + 1):
            doc.new_page().insert_text((72, 72), f"Page {number}. Policy Number PA-1.", fontsize=12)
        return doc.tobytes()


def test_a_pdf_can_be_read_as_runs_of_pages_at_once_and_joined_in_order():
    httpd, url = _serve(_MinerU)
    try:
        # Every task of the fake reads as STRUCTURED's two pages: two runs of two.
        pages, info = ocr_via_api(_pdf(4), url, "tok", filename="a.pdf", poll_s=0.01, parts=2)
        assert len(pages) == 4 and pages[0] == pages[2] and pages[1].startswith("## Conditions")
        assert info["parts"] == 2 and len(info["task_id"]) == 2 and info["server_seconds"] == 8.25
    finally:
        httpd.shutdown()


class _SlowUploadMinerU(_MinerU):
    """Takes a while to receive each PDF, as a laptop's upload does; records when."""

    events: list = []

    def do_POST(self):  # noqa: N802
        started = time.perf_counter()
        time.sleep(0.3)
        _SlowUploadMinerU.events.append(("post", started, time.perf_counter()))
        super().do_POST()

    def do_GET(self):  # noqa: N802
        _SlowUploadMinerU.events.append(("get", time.perf_counter(), None))
        super().do_GET()


def test_runs_of_pages_upload_one_after_another_and_each_is_read_while_the_next_uploads():
    _SlowUploadMinerU.events, _MinerU.polls = [], 0
    httpd, url = _serve(_SlowUploadMinerU)
    uploaded = threading.Event()
    try:
        pages, info = ocr_via_api(_pdf(6), url, "tok", filename="a.pdf", poll_s=0.01, parts=3, uploaded=uploaded)
    finally:
        httpd.shutdown()
    assert len(pages) == 6 and info["parts"] == 3 and uploaded.is_set()
    posts = [(start, end) for kind, start, end in _SlowUploadMinerU.events if kind == "post"]
    gets = [start for kind, start, _ in _SlowUploadMinerU.events if kind == "get"]
    # One upload at a time (each has the whole connection) ...
    assert all(later[0] >= earlier[1] for earlier, later in zip(posts, posts[1:], strict=False))
    # ... and the first run is polled while the last is still uploading.
    assert min(gets) < posts[-1][0]


def test_pages_are_rendered_while_minerU_reads_them_and_land_where_the_pipeline_looks(tmp_path):
    from data_pipeline.ocr.served_ocr import ServedOcr

    httpd, url = _serve(_MinerU)
    published = []
    try:
        reader = ServedOcr(url, "tok", render_workers=1)
        read = reader.read(_pdf(2), tmp_path / "pages", name="a.pdf",
                           publish_pages=lambda paths: published.append(paths) or {"seconds": 0.5, "mb": 0.1})
        assert [p.rsplit("page_", 1)[1] for p in read["images"]] == ["001.png", "002.png"]
        assert (tmp_path / "pages" / "page_002.md").read_text(encoding="utf-8").startswith("## Conditions")
        assert read["texts"][1].startswith("Plymouth Rock")
        assert set(read["timings_s"]) == {"render", "ocr", "ocr_server", "links", "pages_and_text"}
        # The rendered pages, published once the PDF was at MinerU.
        assert published == [read["images"]] and read["links"]["mb"] == 0.1 and read["timings_s"]["links"] == 0.5
        # As the job server's OcrCall: a PDF whose page count MinerU disagrees with fails the job.
        (tmp_path / "three.pdf").write_bytes(_pdf(3))
        refused = reader(tmp_path / "three.pdf", tmp_path / "three")
        assert refused["ok"] is False and "MinerU read 2 pages, the renderer 3" in refused["error"]
    finally:
        httpd.shutdown()


# --------------------------------------------------------------------------
# One MinerU for training and serving; HTTPS; the served adapter
# --------------------------------------------------------------------------


def test_the_corpus_is_read_by_the_served_minerU_and_records_its_pin(tmp_path):
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.ocr.run_mineru import process_document
    from data_pipeline.ocr.served_ocr import ServedMinerUEngine

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="ocr")
    client.write_bytes(paths.raw_pdf("policy", "policy_0001"), _pdf(2))
    client.write_json(paths.raw_metadata("policy", "policy_0001"),
                      {"source_id": "policy_0001", "doc_type": "policy", "checksum_sha256": "abc"})
    _MinerU.posted, _MinerU.polls = [], 0
    httpd, url = _serve(_MinerU)
    try:
        engine = ServedMinerUEngine(url, "tok", render_workers=1)     # no MinerU and no GPU here
        meta = process_document("policy", "policy_0001", client, engine)
        again = process_document("policy", "policy_0001", client, engine)
    finally:
        httpd.shutdown()
    assert meta["mineru_version"] == "4.0.12" and meta["mineru_backend"] == "api" and meta["mineru_tier"] == "basic"
    assert meta["ocr_device"] == "cuda" and meta["page_count"] == 2 and not meta["read_by_ocr"]
    assert "ocr_mode=txt" in _MinerU.posted[0]                         # a digital PDF: the OCR stage's rule
    assert client.read_text(paths.processed_page("policy", "policy_0001", 2, "md")).startswith("## Conditions")
    assert "_reprocessed" not in again and len(_MinerU.posted) == 1     # the same MinerU: not read again


def test_text_from_another_minerU_than_the_pinned_one_is_refused(tmp_path):
    from data_pipeline.ocr.served_ocr import ServedOcr, ServedOcrError

    httpd, url = _serve(_MinerU)
    try:
        with pytest.raises(ServedOcrError, match="read by MinerU 4.0.12 .* trained on text from MinerU 3.4.5"):
            ServedOcr(url, render_workers=1, expected_version="3.4.5").read(_pdf(2), tmp_path / "a")
        with pytest.raises(ServedOcrError, match="tier basic"):
            ServedOcr(url, tier="flash", render_workers=1, expected_version="4.0.12").read(_pdf(2), tmp_path / "b")
        read = ServedOcr(url, render_workers=1).read(_pdf(2), tmp_path / "c")    # a timing test: anything goes
        assert read["ocr"]["version"] == "4.0.12" and read["ocr"]["ocr_mode"] == "txt"
    finally:
        httpd.shutdown()


def test_the_ocr_stage_finds_the_served_minerU_in_the_environment(monkeypatch):
    from common.config import served_mineru
    from data_pipeline.ocr.served_ocr import ServedMinerUEngine

    monkeypatch.delenv("MINERU_URL", raising=False)
    assert ServedMinerUEngine.from_environment() is None
    monkeypatch.setenv("MINERU_URL", "https://pod-8001.proxy.runpod.net")
    monkeypatch.setenv("MINERU_TOKEN", "tok")
    engine = ServedMinerUEngine.from_environment()
    assert engine.reader.mineru_url.endswith("proxy.runpod.net") and engine.reader.token == "tok"
    assert engine.environment().mineru_version == served_mineru()["version"] == "4.0.12"


def test_pages_never_leave_this_machine_unencrypted():
    from common.transport import InsecureUrlError, assert_encrypted
    from data_pipeline.ocr.served_ocr import ServedOcr

    for url in ("https://pod-8000.proxy.runpod.net", "http://localhost:8000", "http://127.0.0.1:8001", "http://[::1]:9"):
        assert_encrypted(url, "a server")
    for url in ("http://pod-8000.proxy.runpod.net", "http://10.0.0.5:8000", "ftp://x"):
        with pytest.raises(InsecureUrlError, match="not HTTPS"):
            assert_encrypted(url, "a server")
    with pytest.raises(InsecureUrlError):
        OpenAIChatBackend("http://pod-8000.proxy.runpod.net", model="served")
    with pytest.raises(InsecureUrlError):
        ServedOcr("http://pod-8001.proxy.runpod.net")


def test_the_adapter_asked_for_must_be_served_and_on_our_base(vllm):
    lora = {"id": "policy-v3", "root": "/workspace/adapters/policy-v3", "parent": "Qwen3-VL-8B-Instruct"}
    _Vllm.models = [*BASE_ONLY, lora]
    assert OpenAIChatBackend(vllm, model="policy-v3").model == "policy-v3"
    with pytest.raises(ModelRunnerError, match="does not serve 'policy-v4'.*--lora-modules policy-v4="):
        OpenAIChatBackend(vllm, model="policy-v4")
    _Vllm.models = [{"id": "llama", "root": "/workspace/models/Llama-3.1-8B", "parent": None},
                    {**lora, "parent": "llama"}]
    with pytest.raises(ModelRunnerError, match="not on qwen3-vl-8b-instruct"):
        OpenAIChatBackend(vllm, model="policy-v3")
