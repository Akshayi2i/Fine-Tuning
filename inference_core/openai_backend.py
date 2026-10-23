"""A model served behind an OpenAI-compatible HTTP API (``vllm serve``), as a backend.

The in-process vLLM engine (``VLLMBackend``) is how training pods and the release
endpoint run the model. This reads with a model someone already serves - on a
pod, behind its RunPod proxy URL or on localhost - so the extraction routine
(:func:`serving.pipeline.extract`) runs unchanged: the same windows, prompts,
schema slices, answer limits and merge. Only the transport differs:

* every page image travels with every window that reads it, about four times a
  page: as a base64 PNG data URL (``images="inline"``, the default); as a
  read-only link to the page put in Blob once (``images="link"``, with
  :class:`inference_core.page_links.PageLinks`), which the server fetches
  itself; or - next to the server, which must then be started with
  ``--allowed-local-media-path`` - as a ``file://`` path the server reads from
  its own disk (``images="file"``). A gray page goes as a grayscale PNG
  (``gray_pages``, on by default): the same pixels once the server makes it
  RGB, about 60% of the bytes (``inference_core.page_links.sent_png``). From a
  laptop the inline upload is most of the time; ``sent_bytes`` counts what went
  out;
* each window is its own request (:meth:`OpenAIChatBackend.submit`), up to
  ``max_concurrency`` in flight, which vLLM schedules together as the engine's
  continuous batching does. The pipeline reads each answer as it arrives and
  sends a failed window's halves at once, instead of after the slowest window;
* the window's schema slice goes as ``response_format`` (``json_schema``), the
  pixel budget as ``mm_processor_kwargs``, and token logprobs as ``logprobs``;
* the answer is streamed (``stream``, on by default): a RunPod proxy closes a
  request that has sent nothing back for about 100 seconds, and a window queued
  behind others, then writing a long schedule, takes longer than that. Streamed,
  an answer whose end has become one passage repeated over and over
  (:func:`repeating_tail`) is stopped there (``stop_repeats``) rather than left
  to write to its token limit, minutes for one window; the pipeline then splits
  that window as it does one cut off at the limit.

What only the server sets is the server's: its logprobs mode (raw, for
calibration), xgrammar's no-whitespace rule and its context length. Start it as
``configs/inference/vllm_serving.yaml`` describes; a test of timing does not
depend on them.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from inference_core.model_runner import STOPPED_REPEATING, Generation, ModelBackend, ModelRunnerError
from inference_core.runner_config import RunnerConfig

log = logging.getLogger(__name__)

#: Content types for page images sent as data URLs.
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

#: How a page image reaches the server: in the request, by a link to its copy
#: in Blob, or by its path on the server's own disk.
IMAGE_TRANSPORTS = ("inline", "link", "file")

#: An answer is checked for a loop each time it has grown by this many characters.
REPEAT_CHECK_CHARS = 1000


def image_data_url(image: Any, *, gray: bool = True) -> str:
    """A page image - a path or bytes - as a data URL (PNG unless the path says
    otherwise); a gray page as a grayscale PNG (:func:`inference_core.page_links.sent_png`)."""
    from inference_core.page_links import sent_png

    if isinstance(image, bytes):
        data, kind = image, "image/png"
    else:
        path = Path(image)
        data, kind = sent_png(path, gray=gray), IMAGE_TYPES.get(path.suffix.lower(), "image/png")
    return f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}"


def image_url(image: Any, transport: str = "inline", *, links: Any = None, gray: bool = True) -> str:
    """The URL a page image is sent as: a data URL; under ``link`` its link in
    ``links`` (a page with none goes inline); under ``file`` the ``file://`` URL
    of its path. Bytes have no path, so they always go inline."""
    if not isinstance(image, bytes):
        if transport == "file":
            return Path(image).resolve().as_uri()
        if transport == "link":
            link = links.url(image) if links is not None else None
            if link:
                return link
            log.warning("%s has no link; sent in the request", Path(image).name)
    return image_data_url(image, gray=gray)


def chat_messages(messages: list[dict[str, Any]], images: str = "inline", *, links: Any = None,
                  gray: bool = True) -> list[dict[str, Any]]:
    """The pipeline's messages (``inference_core.input_builder``) as chat-completions messages."""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        parts = []
        for block in content:
            if block.get("type") == "image":
                url = image_url(block["image"], images, links=links, gray=gray)
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif block.get("type") == "text":
                parts.append({"type": "text", "text": block["text"]})
            else:
                raise ModelRunnerError(f"a content block of type {block.get('type')!r} has no chat form")
        out.append({"role": message["role"], "content": parts})
    return out


def repeating_tail(text: str, *, min_chars: int = 3000, min_repeats: int = 6,
                   max_period: int = 500) -> int | None:
    """The length of the passage the end of ``text`` repeats, or None.

    The end counts as a loop when its last ``min_chars`` characters (at least
    ``min_repeats`` passages) are one passage of at most ``max_period``
    characters over and over: a run of blank space, the same row again and
    again. A real answer's rows differ - another vehicle, another page - so it
    does not stay periodic for thousands of characters.
    """
    end = len(text)
    for period in range(1, max_period + 1):
        span = max(min_chars, period * min_repeats)
        if span + period > end:
            return None
        if text[end - 1] != text[end - 1 - period]:
            continue
        tail = text[end - span - period:]
        if tail[period:] == tail[:-period]:
            return period
    return None


def _read_stream(response: Any, *, started: float, stop_repeats: bool = True) -> dict[str, Any]:
    """One streamed chat completion (server-sent events), read to its end - or
    to the point its answer was found repeating itself."""
    content: list[str] = []
    tokens: list[str] = []
    logprobs: list[float] = []
    finish_reason = "stop"
    prompt_tokens: int | None = None
    first_token_ms: float | None = None
    length = checked_at = 0
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise ModelRunnerError(f"the server stopped the answer: {json.dumps(chunk['error'])[:400]}")
        if chunk.get("usage"):
            prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
        for choice in chunk.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content") or ""
            if piece and first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            content.append(piece)
            length += len(piece)
            for item in (choice.get("logprobs") or {}).get("content") or []:
                tokens.append(str(item["token"]))
                logprobs.append(float(item["logprob"]))
            finish_reason = choice.get("finish_reason") or finish_reason
        if stop_repeats and length - checked_at >= REPEAT_CHECK_CHARS:
            checked_at = length
            period = repeating_tail("".join(content))
            if period is not None:
                # Leaving the stream closes the connection, and the server
                # drops the request: its slot goes to the windows still waiting.
                log.warning("an answer repeated a %d-character passage after %d characters; stopped",
                            period, length)
                finish_reason = STOPPED_REPEATING
                break
    return {"text": "".join(content), "tokens": tokens, "logprobs": logprobs,
            "finish_reason": finish_reason, "prompt_tokens": prompt_tokens, "first_token_ms": first_token_ms}


class OpenAIChatBackend(ModelBackend):
    """``/v1/chat/completions`` of a served model, one request per window."""

    def __init__(self, base_url: str, token: str | None = None, model: str | None = None, *,
                 timeout: float = 1800, max_concurrency: int = 96, stream: bool = True,
                 images: str = "inline", stop_repeats: bool = True, links: Any = None,
                 gray_pages: bool = True) -> None:
        if images not in IMAGE_TRANSPORTS:
            raise ValueError(f"images must be one of {IMAGE_TRANSPORTS}, got {images!r}")
        if images == "link" and links is None:
            raise ValueError("images='link' needs links (inference_core.page_links.PageLinks)")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.stream = stream
        self.images = images
        #: Where a page's link is looked up (``images="link"``).
        self.links = links
        self.gray_pages = gray_pages
        self.stop_repeats = stop_repeats
        self.sent_bytes = 0
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self.model = model or self._served_model()

    def supports_logprobs(self) -> bool:
        return True

    def supports_submit(self) -> bool:
        return True

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _workers(self) -> ThreadPoolExecutor:
        """The requests in flight: one thread each, at most ``max_concurrency``."""
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.max_concurrency,
                                                thread_name_prefix="served-model")
            return self._pool

    def _request(self, path: str, body: bytes | None = None, *, stream: bool = False) -> Any:
        """The JSON answer to ``path``; with ``stream``, one chat completion's
        streamed chunks, read as :func:`_read_stream` reads them."""
        headers = {"User-Agent": "fideon-openai-backend/1", "Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(f"{self.base_url}{path}", data=body, headers=headers,
                                         method="POST" if body is not None else "GET")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if stream:
                    return _read_stream(response, started=started, stop_repeats=self.stop_repeats)
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ModelRunnerError(f"{exc.code} from {path}: {exc.read().decode('utf-8', 'replace')[:400]}") from exc
        except urllib.error.URLError as exc:
            raise ModelRunnerError(f"could not reach {self.base_url}{path}: {exc.reason}") from exc

    def _served_model(self) -> str:
        served = self._request("/v1/models").get("data") or []
        if len(served) != 1:
            raise ModelRunnerError(f"{self.base_url} serves {len(served)} models; name the one to use")
        return str(served[0]["id"])

    def payload(self, messages: list[dict[str, Any]], config: RunnerConfig) -> dict[str, Any]:
        from common.config import pixel_budget

        min_pixels, max_pixels = pixel_budget()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages(messages, self.images, links=self.links, gray=self.gray_pages),
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_new_tokens,
            "seed": config.seed,
            "logprobs": bool(config.logprobs),
            "mm_processor_kwargs": {"min_pixels": min_pixels, "max_pixels": max_pixels},
        }
        if config.json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "extraction", "schema": config.json_schema}}
        if self.stream:
            body["stream"] = True
            # The prompt's length comes in a last chunk of its own.
            body["stream_options"] = {"include_usage": True}
        return body

    def generate(self, messages: list[dict[str, Any]], config: RunnerConfig,
                 adapter: str | None = None) -> Generation:
        if adapter:
            raise ModelRunnerError("a served model is called as it is served: no per-request adapter here")
        body = json.dumps(self.payload(messages, config)).encode("utf-8")
        with self._lock:
            self.sent_bytes += len(body)
        started = time.perf_counter()
        if self.stream:
            answer = self._request("/v1/chat/completions", body, stream=True)
        else:
            reply = self._request("/v1/chat/completions", body)
            choice = reply["choices"][0]
            scored = (choice.get("logprobs") or {}).get("content") or []
            answer = {
                "text": choice["message"].get("content") or "",
                "tokens": [str(item["token"]) for item in scored],
                "logprobs": [float(item["logprob"]) for item in scored],
                "finish_reason": choice.get("finish_reason") or "stop",
                "prompt_tokens": (reply.get("usage") or {}).get("prompt_tokens"),
                "first_token_ms": None,
            }
        latency_ms = (time.perf_counter() - started) * 1000
        text, tokens, logprobs = answer["text"], answer["tokens"], answer["logprobs"]
        # The server scores the end-of-turn token (<|im_end|>) too, which the text
        # does not hold: left in, the tokens no longer spell the text and no field
        # can be given a confidence (inference_core.span_map).
        joined = "".join(tokens)
        while tokens and joined != text and joined.startswith(text):
            joined = joined[: len(joined) - len(tokens.pop())]
            logprobs.pop()
        first = answer["first_token_ms"]
        return Generation(
            text=text,
            tokens=tokens,
            token_logprobs=logprobs,
            finish_reason=answer["finish_reason"],
            generation_fingerprint=config.fingerprint(),
            latency_ms=round(latency_ms, 1),
            prompt_tokens=answer["prompt_tokens"],
            first_token_ms=round(first, 1) if first is not None else None,
        )

    def _answer(self, messages: list[dict[str, Any]], config: RunnerConfig,
                adapter: str | None) -> Generation | Exception:
        try:
            return self.generate(messages, config, adapter)
        except Exception as exc:  # noqa: BLE001 - one window's failure is its own result
            return exc

    def submit(self, messages: list[dict[str, Any]], config: RunnerConfig,
               adapter: str | None = None) -> Future:
        return self._workers().submit(self._answer, messages, config, adapter)

    def generate_batch(self, messages_list: list[list[dict[str, Any]]], configs: list[RunnerConfig],
                       adapter: str | None = None) -> list[Generation | Exception]:
        futures = [self.submit(m, c, adapter) for m, c in zip(messages_list, configs, strict=True)]
        return [future.result() for future in futures]
