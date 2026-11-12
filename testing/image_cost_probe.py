"""How long the served model takes to take in page images: one request alone, then several at once. Read-only.

2026-11-12, pod (H200, vLLM 0.11.0, v3): a 224-page document's 135 windows started about 2.2 s apart, one after
another, whether vLLM ran one API server process or four, with 96 CPU cores mostly free and the GPU mostly idle;
fields were ready after 485 s three times over. One window-sized request alone (6 page images, one answer token)
took 4.73 s; 8 at once, the last after 17.88 s - 2.2 s a request again. This times requests that carry page images
and ask for one answer token, so only the server's preparing and reading of them is timed, in three shapes:

* ``--count`` pages at the size they were rendered (the window as served);
* half as many pages - the cost per page image;
* the same pages at half the pixels - the cost per pixel.

Run it against the served model on the pod, on pages a run of ``testing.latency_check --images file`` rendered:

    python -m testing.image_cost_probe --pages /workspace/latency_check_v3_probe/document.fields-first/pages

The half-size copies are written beside the pages (``<pages>/../image_cost_probe/``): vLLM reads image files only
under its ``--allowed-local-media-path``, which covers the pages' own folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: The share of each side a half-pixel copy keeps: 0.707 x 0.707 = half the pixels.
HALF_SIDE = 0.5 ** 0.5


def request_body(model: str, images: list[Path]) -> bytes:
    """A chat request with ``images`` (as local files) and one answer token: nothing but reading them is timed."""
    content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": f"file://{path}"}} for path in images]
    content.append({"type": "text", "text": "Read the pages."})
    return json.dumps({"model": model, "max_tokens": 1, "temperature": 0.0,
                       "messages": [{"role": "user", "content": content}]}).encode("utf-8")


def time_request(url: str, body: bytes, token: str | None = None, timeout: float = 600) -> float:
    """Seconds from sending ``body`` to the whole answer."""
    from common.transport import open_url

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    with open_url(urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", data=body, headers=headers,
                                         method="POST"), timeout=timeout) as response:
        response.read()
    return time.perf_counter() - started


def measure(url: str, body: bytes, at_once: int, token: str | None = None) -> dict[str, Any]:
    """One request alone, then ``at_once`` together: each one's seconds, and the last one's per request."""
    alone = time_request(url, body, token)
    with ThreadPoolExecutor(max_workers=at_once) as pool:
        together = sorted(pool.map(lambda _: time_request(url, body, token), range(at_once)))
    return {"alone_s": round(alone, 2), "together_s": [round(t, 2) for t in together],
            "per_request_s": round(together[-1] / at_once, 2)}


def half_pixels(images: list[Path], out: Path) -> list[Path]:
    """A copy of each image with half its pixels (each side x 0.707), in ``out``."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    copies = []
    for path in images:
        with Image.open(path) as image:
            width, height = image.size
            target = out / f"half_{path.name}"
            image.resize((max(1, round(width * HALF_SIDE)), max(1, round(height * HALF_SIDE)))).save(target)
        copies.append(target)
    return copies


def shapes(images: list[Path], out: Path) -> list[tuple[str, list[Path]]]:
    """The three request shapes: as served, half the pages, half the pixels."""
    count = len(images)
    return [(f"{count} pages, full size", images),
            (f"{max(1, count // 2)} pages, full size", images[:max(1, count // 2)]),
            (f"{count} pages, half the pixels", half_pixels(images, out))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, type=Path, help="a folder of rendered page images (*.png)")
    parser.add_argument("--url", default="http://localhost:8000", help="the served model's base URL")
    parser.add_argument("--model", default="policy-v3", help="the served model's name")
    parser.add_argument("--count", type=int, default=6, help="page images in a request (a window: 6)")
    parser.add_argument("--at-once", dest="at_once", type=int, default=8, help="requests sent together")
    parser.add_argument("--token", default=os.environ.get("FIDEON_REMOTE_TOKEN"))
    args = parser.parse_args(argv)
    images = sorted(args.pages.glob("*.png"))[: args.count]
    if len(images) < args.count:
        parser.error(f"{args.pages} holds {len(images)} page image(s); --count is {args.count}")
    from PIL import Image

    with Image.open(images[0]) as first:
        print(f"{len(images)} page images from {args.pages}, the first {first.size[0]} x {first.size[1]} pixels")
    for name, shape in shapes(images, args.pages.parent / "image_cost_probe"):
        result = measure(args.url, request_body(args.model, shape), args.at_once, args.token)
        print(f"{name}: alone {result['alone_s']} s; {args.at_once} at once {result['together_s']} s, "
              f"the last {result['per_request_s']} s per request", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
