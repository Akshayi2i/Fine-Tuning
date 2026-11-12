"""testing/write_speed_probe.py: how fast the server writes, as each of production's request settings is added."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from testing.write_speed_probe import VARIANTS, main, measure, request_body, window_schema

WRITTEN = 50


class _Server(BaseHTTPRequestHandler):
    """Lists one model and answers every chat request with WRITTEN tokens, streamed or not, as asked."""

    bodies: list[dict] = []

    def do_GET(self):  # noqa: N802 - http.server's name
        self._send(json.dumps({"data": [{"id": "policy-v3"}]}).encode(), "application/json")

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.bodies.append(body)
        usage = {"prompt_tokens": 40, "completion_tokens": WRITTEN}
        if body.get("stream"):
            chunks = [{"choices": [{"delta": {"content": "{"}, "finish_reason": None}]},
                      {"choices": [{"delta": {"content": "}"}, "finish_reason": "length"}]},
                      {"choices": [], "usage": usage}]
            payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            self._send(payload.encode(), "text/event-stream")
        else:
            reply = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": usage}
            self._send(json.dumps(reply).encode(), "application/json")

    def _send(self, data: bytes, kind: str):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _Server.bodies = []
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_each_variant_adds_one_of_productions_settings(server):
    from inference_core.openai_backend import OpenAIChatBackend

    client = OpenAIChatBackend(server, None, model="policy-v3")
    schema = window_schema("bopgl", "arrays")
    bodies = [json.loads(request_body(client, stream=s, scores=c, schema=schema if r else None, tokens=300))
              for _name, s, c, r in VARIANTS]
    plain, streamed, scored, shaped = bodies
    assert not plain.get("stream") and plain["logprobs"] is False and "response_format" not in plain
    assert streamed["stream"] and streamed["stream_options"] == {"include_usage": True}
    assert scored["logprobs"] is True and scored["top_logprobs"] == 2 and "response_format" not in scored
    assert shaped["response_format"]["json_schema"]["schema"] == schema
    assert all(b["max_tokens"] == 300 and b["model"] == "policy-v3" for b in bodies)


def test_the_tokens_are_the_servers_own_count_streamed_or_not(server):
    from inference_core.openai_backend import OpenAIChatBackend

    client = OpenAIChatBackend(server, None, model="policy-v3")
    for stream in (False, True):
        result = measure(server, request_body(client, stream=stream, scores=True, schema=None, tokens=50),
                         None, stream, requests=4)
        assert result["tokens"] == 4 * WRITTEN
        assert result["at_limit"] == (4 if stream else 0)


def test_the_window_schema_is_compact_and_bounded_to_six_pages():
    text = json.dumps(window_schema("bopgl", "arrays"))
    assert '"raw"' not in text and '"r"' in text and '"maximum": 6' in text


def test_the_command_prints_one_line_a_variant(server, capsys):
    assert main(["--url", server, "--requests", "3", "--tokens", "50"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name, *_ in VARIANTS) and "150 tokens" in out


def test_with_pages_the_last_round_runs_while_fresh_page_requests_arrive(server, tmp_path, capsys):
    from PIL import Image

    pages = tmp_path / "doc" / "pages"
    pages.mkdir(parents=True)
    for n in range(1, 13):
        Image.new("L", (100, 130), 255).save(pages / f"page_{n:03d}.png")
    assert main(["--url", server, "--requests", "2", "--tokens", "50", "--pages", str(pages)]) == 0
    out = capsys.readouterr().out
    assert "+ windows of new page images arriving alongside" in out and "2 page requests done after" in out
    image_bodies = [b for b in _Server.bodies if any(p.get("type") == "image_url"
                                                     for p in b["messages"][0]["content"])]
    urls = [p["image_url"]["url"] for b in image_bodies for p in b["messages"][0]["content"] if p["type"] == "image_url"]
    assert len(image_bodies) == 2 and len(urls) == 12 == len(set(urls))           # every page once, six a request
    assert all("/write_speed_probe/fresh_" in url.replace("\\", "/") for url in urls)  # copies, beside the pages
    with Image.open(urls[0].removeprefix("file://")) as copy:
        assert copy.size[0] < 100                                                  # not the image the server saw
