"""A served model (inference_core.openai_backend) and a served MinerU 4
(data_pipeline.ocr.mineru_api), against fake local servers."""

from __future__ import annotations

import json
import threading
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


class _Vllm(BaseHTTPRequestHandler):
    seen: list = []

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
        self._json({"data": [{"id": "Qwen3-VL-8B-Instruct"}]})

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
    _Vllm.seen = []
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


def test_pages_are_rendered_while_minerU_reads_them_and_land_where_the_pipeline_looks(tmp_path):
    from data_pipeline.ocr.served_ocr import ServedOcr

    httpd, url = _serve(_MinerU)
    try:
        reader = ServedOcr(url, "tok", render_workers=1)
        read = reader.read(_pdf(2), tmp_path / "pages", name="a.pdf")
        assert [p.rsplit("page_", 1)[1] for p in read["images"]] == ["001.png", "002.png"]
        assert (tmp_path / "pages" / "page_002.md").read_text(encoding="utf-8").startswith("## Conditions")
        assert read["texts"][1].startswith("Plymouth Rock")
        assert set(read["timings_s"]) == {"render", "ocr", "ocr_server", "pages_and_text"}
        # As the job server's OcrCall: a PDF whose page count MinerU disagrees with fails the job.
        (tmp_path / "three.pdf").write_bytes(_pdf(3))
        refused = reader(tmp_path / "three.pdf", tmp_path / "three")
        assert refused["ok"] is False and "MinerU read 2 pages, the renderer 3" in refused["error"]
    finally:
        httpd.shutdown()
