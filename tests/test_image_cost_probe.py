"""testing/image_cost_probe.py: what one request carrying page images costs the server, alone and together."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from testing.image_cost_probe import main, measure, request_body, shapes

SERVE_S = 0.05


class _OneAtATime(BaseHTTPRequestHandler):
    """A server that reads one request at a time, as the pod's did (2026-11-12)."""

    lock = threading.Lock()
    bodies: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            self.bodies.append(body)
            time.sleep(SERVE_S)
        reply = json.dumps({"choices": [{"message": {"content": "{"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _OneAtATime)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _OneAtATime.bodies = []
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def pages(tmp_path):
    folder = tmp_path / "doc" / "pages"
    folder.mkdir(parents=True)
    for n in range(1, 7):
        Image.new("L", (1000, 1300), 255).save(folder / f"page_{n:03d}.png")
    return folder


def test_requests_served_one_at_a_time_show_as_the_time_each_adds(server, pages):
    body = request_body("policy-v3", sorted(pages.glob("*.png")))
    result = measure(server, body, at_once=4)
    assert result["together_s"][-1] >= 4 * SERVE_S and result["per_request_s"] >= SERVE_S
    sent = _OneAtATime.bodies[0]
    assert sent["max_tokens"] == 1 and sent["model"] == "policy-v3"
    urls = [part["image_url"]["url"] for part in sent["messages"][0]["content"] if part["type"] == "image_url"]
    assert len(urls) == 6 and all(url.startswith("file://") and url.endswith(".png") for url in urls)


def test_the_three_shapes_are_as_served_half_the_pages_and_half_the_pixels(pages):
    images = sorted(pages.glob("*.png"))
    (_, served), (_, fewer), (_, smaller) = shapes(images, pages.parent / "image_cost_probe")
    assert served == images and fewer == images[:3]
    assert all(path.parent == pages.parent / "image_cost_probe" for path in smaller)  # beside the pages, not /tmp
    with Image.open(smaller[0]) as image:
        assert abs(image.size[0] * image.size[1] / (1000 * 1300) - 0.5) < 0.01


def test_the_command_prints_one_line_a_shape(server, pages, capsys):
    assert main(["--pages", str(pages), "--url", server, "--at-once", "2"]) == 0
    out = capsys.readouterr().out
    assert "6 pages, full size" in out and "3 pages, full size" in out and "6 pages, half the pixels" in out


def test_too_few_page_images_is_refused(pages):
    with pytest.raises(SystemExit):
        main(["--pages", str(pages), "--count", "9"])
