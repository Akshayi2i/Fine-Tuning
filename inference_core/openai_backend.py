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
  pixel budget as ``mm_processor_kwargs``, and token logprobs as ``logprobs``
  with the best alternative to each token (``top_logprobs: 2``: the near-tie
  margin, ``span_map.FieldSpan.top2_margin``). The sampling settings a server
  would fill from the model's generation config - repetition penalty, top_k,
  presence and frequency penalties - are sent at their neutral values
  (``model_runner.PINNED_SAMPLING``), and the server is started with
  ``--generation-config vllm`` too;
* the answer is streamed (``stream``, on by default): a RunPod proxy closes a
  request that has sent nothing back for about 100 seconds, and a window queued
  behind others, then writing a long schedule, takes longer than that. Streamed,
  an answer whose end has become one passage repeated over and over
  (:func:`repeating_tail`) is stopped there (``stop_repeats``) rather than left
  to write to its token limit, minutes for one window; the pipeline then splits
  that window as it does one cut off at the limit. So too an answer whose
  stream ends before the server finished it (:data:`STREAM_CUT`: no finish
  reason, no ``[DONE]`` - a proxy closed the connection mid-answer); a stream
  that breaks off mid-chunk is a :class:`ModelRunnerError`, the window's failure;
* a redirect is refused, not followed (``common.transport.open_url``): it
  would send the request and its token to an address nobody checked.

What only the server sets is the server's: its logprobs mode (raw, for
calibration), xgrammar's no-whitespace rule, its context length and its
generation config (``--generation-config vllm``: vLLM's neutral defaults, not
the model's). Start it as ``configs/inference/vllm_serving.yaml`` describes; a
test of timing does not depend on them. The prompt's length the server reports
(``usage.prompt_tokens``) is compared per window with what the pages' pixel
budget implies (``inference_core.prompt_tokens``): a server resizing pages to
another budget is warned about, not trusted.
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

from inference_core.model_runner import (
    PINNED_SAMPLING,
    STOPPED_REPEATING,
    STREAM_CUT,
    TRANSPORT_STATUSES,
    Generation,
    ModelBackend,
    ModelRunnerError,
    ModelTransportError,
)
from inference_core.runner_config import RunnerConfig

log = logging.getLogger(__name__)

#: Content types for page images sent as data URLs.
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

#: How a page image reaches the server: in the request, by a link to its copy
#: in Blob, or by its path on the server's own disk.
IMAGE_TRANSPORTS = ("inline", "link", "file")

#: An answer is checked for a loop each time it has grown by this many characters.
REPEAT_CHECK_CHARS = 1000

#: Whitespace the server-settings probe pushes up (``OpenAIChatBackend.server_setting_problems``): those that
#: are one token in the served model's vocabulary.
_WHITESPACE_PROBES = (" ", "\n", "  ", "\n\n", "\t", "    ", " \n")
#: Words the logprobs probe pushes up as a reply to "OK": the first that is one token is used.
_MARKER_PROBES = (" banana", "banana", " zebra", "zebra", " purple", "purple")
#: A token pushed up by +100 that scores above this was scored after the bias: processed log-probabilities. Its raw
#: score as the reply to "Reply with the word OK." is far below.
PROCESSED_LOGPROB_FLOOR = -0.05


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


@cache
def _probe_page_png() -> str:
    """A small white grayscale page as base64 PNG, made here: what the image probe sends
    (``OpenAIChatBackend._image_reading_problem``). The server scales it to the page pixel budget."""
    import struct
    import zlib

    width = height = 64

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\xff" * width for _ in range(height))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode("ascii")


#: The keys that name a row: a loop writes the same row again under one of them.
LOOP_IDENTIFIER_KEYS = ("form_number", "coverage_name", "label", "name", "vin")
#: A key's value as written, in either answer form: ``"key":"X"`` or ``"key":{"r":"X"`` (``"v"``, ``"raw"``,
#: ``"parsed"`` too).
_KEY_VALUE = re.compile('"(' + "|".join(LOOP_IDENTIFIER_KEYS) + ')"'
                        + r':\s*(?:\{\s*"(?:r|v|raw|parsed)":\s*)?"((?:[^"\\]|\\.)*)"')


def repeating_row(text: str, *, tail_chars: int = 12000, min_count: int = 40,
                  share: float = 0.8) -> tuple[str, str] | None:
    """``(key, value)`` when the end of ``text`` writes one row again and again with other fields changing, or None.

    :func:`repeating_tail` needs the end to repeat exactly; a loop that writes the same form number with a new title
    each time never does (2026-11-09, policy_0071: "APDEC 0919" 166 times, 64 different titles, to the token
    limit). Here the end counts as a loop when, in its last ``tail_chars`` characters, one value of a row-naming key
    (:data:`LOOP_IDENTIFIER_KEYS`) is written at least ``min_count`` times and is at least ``share`` of that key's
    values there. A real list names its rows apart - another form, another coverage at another location - so no
    one name holds most of a long stretch of it; none of the 168 cycle-1 golds, in either form, trips this at any
    1,000-character step (checked 2026-11-10).

    The bar is 40, not lower: a window this stops is read again in halves, but a single page cannot be, and serving
    then drops it (``serving.pipeline``), so a real page that names one party or one coverage many times - the same
    mortgagee on 25 locations - must not look like a loop. The loops it was made for wrote the row 111 and 166 times.
    """
    tail = text[-tail_chars:]
    values: dict[str, list[str]] = {}
    for key, value in _KEY_VALUE.findall(tail):
        values.setdefault(key, []).append(value)
    for key, written in values.items():
        if len(written) < min_count:
            continue
        top = max(set(written), key=written.count)
        count = written.count(top)
        if count >= min_count and count >= share * len(written):
            return key, top
    return None


def _error_body(exc: Any) -> str:
    """A server's error answer, short and on one line. A proxy's HTML error page
    (RunPod's "Waiting for service to respond") is reduced to its title: whole,
    it filled the failure message of every window that met it."""
    body = exc.read().decode("utf-8", "replace")
    if body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        title = re.search(r"<title>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
        return f"an HTML error page ({' '.join(title.group(1).split())[:80] if title else 'no title'})"
    return " ".join(body.split())[:300]


def _read_stream(response: Any, *, started: float, stop_repeats: bool = True) -> dict[str, Any]:
    """One streamed chat completion (server-sent events), read to its end - or
    to the point its answer was found repeating itself.

    A stream that ends with no finish reason and no ``[DONE]`` was cut, not
    finished - a proxy or the server closed it mid-answer - and its answer, the
    start of one, is reported as :data:`STREAM_CUT` (cut off), never as a
    ``stop``: a cut that happened to land after a closing brace would parse."""
    content: list[str] = []
    tokens: list[str] = []
    logprobs: list[float] = []
    seconds: list[float | None] = []
    finish_reason: str | None = None
    done = False
    prompt_tokens: int | None = None
    first_token_ms: float | None = None
    length = checked_at = 0
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            done = True
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
                seconds.append(_second_best(item))
            finish_reason = choice.get("finish_reason") or finish_reason
        if stop_repeats and length - checked_at >= REPEAT_CHECK_CHARS:
            checked_at = length
            text = "".join(content)
            period = repeating_tail(text)
            row = None if period is not None else repeating_row(text)
            if period is not None or row is not None:
                # Leaving the stream closes the connection, and the server
                # drops the request: its slot goes to the windows still waiting.
                if period is not None:
                    log.warning("an answer repeated a %d-character passage after %d characters; stopped",
                                period, length)
                else:
                    # The field, not its value: a repeated name may be an insured's or a lender's.
                    log.warning("an answer wrote the same %s again and again after %d characters; stopped",
                                row[0], length)
                finish_reason = STOPPED_REPEATING
                break
    if finish_reason is None:
        finish_reason = "stop" if done else STREAM_CUT
        if not done:
            log.warning("an answer's stream ended after %d characters with no finish reason and no [DONE]; "
                        "counted as cut off", length)
    return {"text": "".join(content), "tokens": tokens, "logprobs": logprobs, "top2": seconds,
            "finish_reason": finish_reason, "prompt_tokens": prompt_tokens, "first_token_ms": first_token_ms}


def _second_best(item: dict[str, Any]) -> float | None:
    """One scored token's best alternative: the highest logprob among its
    ``top_logprobs`` that is not the token written; None when none is listed."""
    from inference_core.model_runner import second_best

    alternatives = item.get("top_logprobs") or []
    return second_best(((str(alt.get("token")), alt.get("logprob")) for alt in alternatives
                        if isinstance(alt, dict) and isinstance(alt.get("logprob"), (int, float))),
                       str(item.get("token")))


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
        from common.transport import assert_encrypted

        assert_encrypted(base_url, "the model server")
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
        if model:
            self._check_served(model)

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

    def _request(self, path: str, body: bytes | None = None, *, stream: bool = False,
                 timeout: float | None = None) -> Any:
        """The JSON answer to ``path``; with ``stream``, one chat completion's
        streamed chunks, read as :func:`_read_stream` reads them.

        A redirect is refused, and an answer that breaks off while it is read
        (a chunked stream cut mid-chunk: ``IncompleteRead``; a reset; a
        timeout) is a :class:`ModelRunnerError`, the window's own failure."""
        from common.transport import RedirectRefusedError, open_url

        headers = {"User-Agent": "fideon-openai-backend/1", "Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(f"{self.base_url}{path}", data=body, headers=headers,
                                         method="POST" if body is not None else "GET")
        started = time.perf_counter()
        try:
            with open_url(request, timeout=timeout or self.timeout) as response:
                if stream:
                    return _read_stream(response, started=started, stop_repeats=self.stop_repeats)
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            kind = ModelTransportError if exc.code in TRANSPORT_STATUSES else ModelRunnerError
            raise kind(f"{exc.code} from {path}: {_error_body(exc)}") from exc
        except RedirectRefusedError as exc:
            raise ModelRunnerError(str(exc.reason)) from exc
        except urllib.error.URLError as exc:
            raise ModelTransportError(f"could not reach {self.base_url}{path}: {exc.reason}") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise ModelTransportError(f"the answer from {self.base_url}{path} broke off after "
                                      f"{time.perf_counter() - started:.1f} s: {type(exc).__name__}: {exc}") from exc

    def _served_model(self) -> str:
        served = self._request("/v1/models").get("data") or []
        if len(served) != 1:
            raise ModelRunnerError(f"{self.base_url} serves {len(served)} models; name the one to use")
        return str(served[0]["id"])

    def _check_served(self, name: str) -> None:
        """Refuse a model name the server does not serve, or an adapter it
        serves on another base than ours: vLLM answers an unknown name with an
        error per window, and an adapter on the wrong base with plausible JSON.

        vLLM lists each LoRA it loaded (``--lora-modules name=path``) with its
        base as ``parent``. A server that cannot list its models is not checked.
        """
        from common.config import base_model_config

        try:
            served = self._request("/v1/models", timeout=30).get("data") or []
        except ModelRunnerError as exc:
            log.warning("could not list what %s serves, so %r is not checked: %s", self.base_url, name, exc)
            return
        entry = next((item for item in served if item.get("id") == name), None)
        if entry is None:
            raise ModelRunnerError(
                f"{self.base_url} does not serve {name!r} (it serves {[item.get('id') for item in served]}). "
                f"A policy adapter is loaded into vLLM with --enable-lora --lora-modules {name}=<its folder on "
                "the pod's volume>: scripts/served_adapter.py fetches it and prints the flags."
            )
        parent = entry.get("parent")
        if not parent:
            return
        base = next((item for item in served if item.get("id") == parent), {})
        ours = str(base_model_config()["model"]["model_id"]).rsplit("/", 1)[-1].lower()
        where = f"{parent} {base.get('root') or ''}".lower()
        if ours not in where:
            raise ModelRunnerError(
                f"{name!r} is served on {base.get('root') or parent}, not on {ours}: an adapter on another "
                "base reads every page wrongly and still writes valid JSON. Serve it on the pinned base "
                "(configs/base_model.yaml)."
            )

    def _single_tokens(self, texts: Sequence[str]) -> list[int]:
        """The token ids of those ``texts`` that are one token each (vLLM's ``/tokenize``)."""
        ids: list[int] = []
        for text in texts:
            body = json.dumps({"model": self.model, "prompt": text, "add_special_tokens": False}).encode("utf-8")
            tokens = self._request("/tokenize", body, timeout=30).get("tokens") or []
            if len(tokens) == 1:
                ids.append(int(tokens[0]))
        return ids

    def _probe(self, body: dict[str, Any]) -> dict[str, Any]:
        request = {"model": self.model, "temperature": 0.0, "stream": False, **body}
        answer = self._request("/v1/chat/completions", json.dumps(request).encode("utf-8"), timeout=60)
        return (answer.get("choices") or [{}])[0]

    def _image_reading_problem(self) -> str | None:
        """Why the served model cannot read a page image, or None: one small white page sent through the served
        name - the release's adapter - as a real window sends its pages, asking for one token.

        vLLM 0.11.0 wraps Qwen3-VL's vision tower as LoRA layers and fails the first request that carries an image
        (``inference_core.vllm_patches``); the in-process engine applies the fix, a stock ``vllm serve`` does not,
        and production reads every page image through the adapter (``serving.fields_first``). The text-only probes
        pass such a server (2026-11-10 review, finding 1). Run last, once the server has answered the others, so a
        failure here is the image, not the server being out of reach.
        """
        from common.config import pixel_budget

        min_pixels, max_pixels = pixel_budget()
        try:
            self._probe({
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _probe_page_png()}},
                    {"type": "text", "text": "Reply with the word OK."}]}],
                "max_tokens": 1, "mm_processor_kwargs": {"min_pixels": min_pixels, "max_pixels": max_pixels},
            })
        except ModelRunnerError as exc:
            return (f"the server cannot read a page image with {self.model!r} ({str(exc)[:200]}): vLLM 0.11.0 fails "
                    "Qwen3-VL's first image request through a LoRA unless the fix in inference_core.vllm_patches is "
                    "applied - serve with it applied, or with a vLLM that has the fix")
        return None

    def _served_length_problem(self) -> str | None:
        """Why the server's context is shorter than the windows were sized and the model trained for, or None.

        Training's ``max_length`` and every window's budget are ``sequence.max_seq_len`` (32,768); a server
        started with a shorter ``--max-model-len`` refuses the longest windows, which are then split or dropped
        (2026-11-10 review). vLLM 0.11 lists ``max_model_len`` on the base model's entry of ``/v1/models``; an
        adapter's entry names that base as its ``parent`` (``serving_models.show_available_models``). A server
        that cannot list its models, or does not say, is not refused on this.
        """
        from inference_core.runner_config import load_runner_config

        required = int(load_runner_config().max_seq_len)
        try:
            served = self._request("/v1/models", timeout=30).get("data") or []
        except ModelRunnerError as exc:
            log.warning("could not list what %s serves, so its context length is not checked: %s", self.base_url, exc)
            return None
        entry = next((item for item in served if item.get("id") == self.model), None) or {}
        base = next((item for item in served if item.get("id") == entry.get("parent")), entry)
        length = base.get("max_model_len")
        if isinstance(length, int) and length < required:
            return (f"the server's context is {length} tokens, shorter than the {required} the windows were sized "
                    f"and the model trained for: start it with --max-model-len {required}")
        return None

    def server_setting_problems(self) -> list[str]:
        """What this server is started with that the trained model and its calibration were not, as reasons;
        empty when it matches. Two settings only the server sets (``configs/inference/vllm_serving.yaml``):

        * ``disable_any_whitespace`` (``--structured-outputs-config``): vLLM 0.11's xgrammar takes it from the
          server's config alone, never from the request (``backend_xgrammar.py``). Probed by asking for an empty
          object with every whitespace token pushed up (``logit_bias``): with the rule the grammar still allows
          only ``{}``; without it, the push puts whitespace in the answer.
        * ``--logprobs-mode raw_logprobs``, vLLM's default: the calibration was fitted on raw log-probabilities.
          Probed with one unlikely token pushed up: a raw log-probability stays low (the bias is applied after
          it is taken), a processed one comes out near 0, and a logit is no log-probability at all.

        Raises :class:`ModelRunnerError` when the server cannot be asked (no ``/tokenize``, an error).
        """
        problems: list[str] = []
        if short := self._served_length_problem():
            problems.append(short)
        whitespace = self._single_tokens(_WHITESPACE_PROBES)
        if not whitespace:
            raise ModelRunnerError("no whitespace token could be found to probe the server's whitespace rule")
        empty = self._probe({
            "messages": [{"role": "user", "content": "Reply with an empty JSON object."}],
            "max_tokens": 8, "logit_bias": {str(token): 100 for token in whitespace},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "probe", "schema": {"type": "object", "properties": {}, "additionalProperties": False}}},
        })
        text = str((empty.get("message") or {}).get("content") or "")
        if text != "{}":
            problems.append(
                f"the server lets structured answers carry whitespace (asked for an empty object with whitespace "
                f"pushed up, it wrote {text[:40]!r}, not '{{}}'): start it with --structured-outputs-config "
                "'{\"backend\": \"xgrammar\", \"disable_any_whitespace\": true}'")
        marker = self._single_tokens(_MARKER_PROBES)[:1]
        if not marker:
            raise ModelRunnerError("no one-token word could be found to probe the server's logprobs mode")
        scored = self._probe({
            "messages": [{"role": "user", "content": "Reply with the word OK."}],
            "max_tokens": 1, "logprobs": True, "top_logprobs": 2, "logit_bias": {str(marker[0]): 100},
        })
        content = ((scored.get("logprobs") or {}).get("content") or [{}])[0]
        if "logprob" not in content:
            raise ModelRunnerError("the server returned no logprobs to the probe")
        written = float(content["logprob"])
        alternatives = [float(item["logprob"]) for item in content.get("top_logprobs") or [] if "logprob" in item]
        total = sum(math.exp(min(value, 50.0)) for value in alternatives)
        if written > 1e-6 or any(value > 1e-6 for value in alternatives) or total > 1.0 + 1e-3:
            problems.append(f"the server returns logits, not log-probabilities (the probe's token scored {written:.3g}): "
                            "start it with --logprobs-mode raw_logprobs")
        elif written > PROCESSED_LOGPROB_FLOOR:
            problems.append(
                f"the server returns processed log-probabilities (a token pushed up by logit_bias scored "
                f"{written:.3g}, its raw score is far lower): start it with --logprobs-mode raw_logprobs, which "
                "the calibration was fitted on")
        if image := self._image_reading_problem():
            problems.append(image)
        return problems

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
            # Every sampling setting the server would otherwise take from the
            # model's generation config, pinned (model_runner.PINNED_SAMPLING).
            **PINNED_SAMPLING,
            "logprobs": bool(config.logprobs),
            "mm_processor_kwargs": {"min_pixels": min_pixels, "max_pixels": max_pixels},
        }
        if config.logprobs and config.top_logprobs > 1:
            # The best alternative to each token written: its near-tie margin
            # (span_map.FieldSpan.top2_margin). Chat completions lists the top
            # alternatives beside each token; 1 would list only the token itself.
            body["top_logprobs"] = int(config.top_logprobs)
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
                "top2": [_second_best(item) for item in scored],
                "finish_reason": choice.get("finish_reason") or "stop",
                "prompt_tokens": (reply.get("usage") or {}).get("prompt_tokens"),
                "first_token_ms": None,
            }
        latency_ms = (time.perf_counter() - started) * 1000
        text, tokens, logprobs = answer["text"], answer["tokens"], answer["logprobs"]
        top2 = list(answer.get("top2") or [])
        # The server scores the end-of-turn token (<|im_end|>) too, which the text
        # does not hold: left in, the tokens no longer spell the text and no field
        # can be given a confidence (inference_core.span_map).
        joined = "".join(tokens)
        while tokens and joined != text and joined.startswith(text):
            joined = joined[: len(joined) - len(tokens.pop())]
            logprobs.pop()
            if len(top2) > len(tokens):
                top2.pop()
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
            token_top2_logprobs=top2 if len(top2) == len(tokens) and any(v is not None for v in top2) else [],
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
