"""How fast the served model writes answers, and which of production's request settings slows it. Read-only.

2026-11-13, pod (H200, vLLM 0.11.0, v3, image cache on): a 224-page document's fields took 400 s, 367 s of it in the
model, which wrote 78,061 answer tokens - about 213 tokens/s for all windows together - with the GPU mostly idle;
the server's own timers put each window at 46 s writing about 380 tokens (8 tokens/s). The CPU, the API server
process and the engine's queue were each ruled out. This sends ``--requests`` text-only requests at once, each
allowed ``--tokens`` answer tokens, four times, adding one of production's request settings each time:

1. plain - the whole answer in one reply, no confidence scores, no answer-shape rule;
2. + streaming (production streams every answer);
3. + confidence scores (``logprobs``, with the best alternative: top 2, as production asks);
4. + the answer-shape rule: a window's own JSON schema (a line's group, compact, page references bounded to six
   pages), as production sends it - every setting production uses.

The request bodies are built by production's own client (``OpenAIChatBackend.payload``); the tokens written are the
server's own count (``usage.completion_tokens``). Text only, so no page image is prepared and only writing is timed.

    python -m testing.write_speed_probe --url http://localhost:8000 --model policy-v3

Measured 2026-11-13 on the pod: 3,133 / 3,129 / 3,010 / 2,577 tokens/s together for the four - production's
settings cost at most 18%, against a real document's 213 tokens/s. A real document's windows each carry six page
images and keep arriving while earlier ones write; then (last night) each wrote 5-11 tokens/s, and the last three,
once every window had started, about 69. ``--pages`` repeats the last variant while windows of page images arrive
alongside: requests of six pages each, every page used once (as a document's first sight of a page), one answer
token each, sent as fast as the server takes them. Whether the writing slows while pages are being read is then
measured, not inferred:

    python -m testing.write_speed_probe --pages /workspace/latency_check_v3_probe/document.fields-first/pages
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

PROMPT = ("Write the coverage schedule of a made-up commercial property policy with 40 locations, as JSON: for each "
          "location, its building, business personal property and business income coverages, each with its limit "
          "and its deductible. Write every location in full.")

#: (name, streamed, confidence scores, answer-shape rule) - each adds one of production's settings to the last.
VARIANTS = [("plain", False, False, False), ("+ streaming", True, False, False),
            ("+ confidence scores", True, True, False), ("+ answer-shape rule (as production)", True, True, True)]


def window_schema(lob: str, group: str, pages: int = 6) -> dict[str, Any]:
    """The JSON schema a window of ``group`` on ``lob`` is decoded under: compact, page references bounded."""
    from common.canonical import compact_schema
    from common.schemas import resolved_schema, with_page_bounds

    return compact_schema(with_page_bounds(resolved_schema("policy", None, lob, group), pages,
                                           pages=list(range(1, pages + 1))))


def request_body(client: Any, *, stream: bool, scores: bool, schema: dict[str, Any] | None, tokens: int) -> bytes:
    """Production's request body for the prompt, with these settings."""
    from inference_core.runner_config import RunnerConfig

    config = replace(RunnerConfig(), max_new_tokens=tokens, logprobs=scores, top_logprobs=2 if scores else 1,
                     json_schema=schema)
    client.stream = stream
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    return json.dumps(client.payload(messages, config)).encode("utf-8")


def send(url: str, body: bytes, token: str | None, stream: bool, timeout: float = 900) -> tuple[int, float, str]:
    """``(answer tokens, seconds, finish reason)`` for one request, the tokens as the server counts them."""
    from common.transport import open_url

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    request = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", data=body, headers=headers,
                                     method="POST")
    with open_url(request, timeout=timeout) as response:
        if not stream:
            reply = json.loads(response.read().decode("utf-8"))
            finish = (reply.get("choices") or [{}])[0].get("finish_reason") or ""
            return int((reply.get("usage") or {}).get("completion_tokens") or 0), time.perf_counter() - started, finish
        written, finish = 0, ""
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data:"):])
            for choice in chunk.get("choices") or []:
                finish = choice.get("finish_reason") or finish
            if chunk.get("usage"):
                written = int(chunk["usage"].get("completion_tokens") or 0)
        return written, time.perf_counter() - started, finish


def page_requests(model: str, pages: list[Path], per_request: int = 6) -> list[bytes]:
    """One request per ``per_request`` pages, every page in one request only, one answer token each."""
    from testing.image_cost_probe import request_body as image_body

    return [image_body(model, pages[i:i + per_request]) for i in range(0, len(pages), per_request)]


def measure(url: str, body: bytes, token: str | None, stream: bool, requests: int,
            alongside: list[bytes] | None = None) -> dict[str, Any]:
    """``requests`` copies of ``body`` sent at once: tokens written, wall time, and the rate together and each.

    With ``alongside``, those requests (page images) are sent at the same time, 16 at a time, and the line reports
    how long they took too."""
    started = time.perf_counter()
    # Two pools: page requests queued in the writers' pool would hold its workers and start the writers late.
    with ThreadPoolExecutor(max_workers=16) as pages_pool, ThreadPoolExecutor(max_workers=requests) as pool:
        def page_request(other: bytes) -> float:
            send(url, other, token, False)
            return time.perf_counter() - started                     # when it was done, from the start

        side = [pages_pool.submit(page_request, other) for other in alongside or []]
        results = list(pool.map(lambda _: send(url, body, token, stream), range(requests)))
        side_s = max((future.result() for future in side), default=0.0)
    wall = time.perf_counter() - started
    tokens = sum(written for written, _s, _f in results)
    return {"tokens": tokens, "wall_s": round(wall, 1), "together_per_s": round(tokens / wall, 1) if wall else 0.0,
            "each_per_s": round(statistics.median(w / s for w, s, _f in results if s > 0), 1),
            "at_limit": sum(1 for _w, _s, finish in results if finish == "length"),
            "page_requests": len(alongside or []), "pages_done_s": round(side_s, 1)}


def fresh_pages(pages: list[Path], out: Path, nonce: int) -> list[Path]:
    """Copies of ``pages`` the server has not seen: each cropped by a few pixels (a run-unique number), so neither
    its image cache nor its prefix cache holds them - as a new document's pages arrive."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    trim = 1 + nonce % 7
    copies = []
    for path in pages:
        with Image.open(path) as image:
            width, height = image.size
            target = out / f"fresh_{nonce}_{path.name}"
            image.crop((0, 0, max(1, width - trim), height)).save(target)
        copies.append(target)
    return copies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="the served model's base URL")
    parser.add_argument("--model", default="policy-v3", help="the served model's name")
    parser.add_argument("--requests", type=int, default=32, help="requests sent at once (a document: 100+)")
    parser.add_argument("--tokens", type=int, default=1000, help="answer tokens each request may write")
    parser.add_argument("--lob", default="bopgl", help="the line whose window schema the last variant uses")
    parser.add_argument("--group", default="arrays", help="that window's group")
    parser.add_argument("--pages", type=Path, default=None,
                        help="a folder of rendered page images: repeat the last variant while windows of fresh copies "
                             "of them (six pages each, one answer token) arrive alongside")
    parser.add_argument("--token", default=os.environ.get("FIDEON_REMOTE_TOKEN"))
    args = parser.parse_args(argv)
    from inference_core.openai_backend import OpenAIChatBackend

    client = OpenAIChatBackend(args.url, args.token, model=args.model)
    schema = window_schema(args.lob, args.group)
    print(f"{args.requests} requests at once, up to {args.tokens} answer tokens each, text only; "
          f"the shape rule is {args.lob}/{args.group}'s window schema ({len(json.dumps(schema)):,} characters)")
    rounds: list[tuple[str, bool, bool, bool, list[bytes] | None]] = [(*variant, None) for variant in VARIANTS]
    if args.pages is not None:
        pages = sorted(args.pages.glob("*.png"))
        if not pages:
            parser.error(f"{args.pages} holds no page images (*.png)")
        fresh = fresh_pages(pages, args.pages.parent / "write_speed_probe", nonce=int(time.time()))
        rounds.append(("+ windows of new page images arriving alongside", True, True, True,
                       page_requests(args.model, fresh)))
    for name, stream, scores, shaped, alongside in rounds:
        body = request_body(client, stream=stream, scores=scores, schema=schema if shaped else None,
                            tokens=args.tokens)
        result = measure(args.url, body, args.token, stream, args.requests, alongside)
        side = (f"; {result['page_requests']} page requests done after {result['pages_done_s']} s"
                if alongside else "")
        print(f"{name}: {result['tokens']:,} tokens in {result['wall_s']} s = {result['together_per_s']} tokens/s "
              f"together, {result['each_per_s']} tokens/s each (median); {result['at_limit']} ran to the limit{side}",
              flush=True)
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
