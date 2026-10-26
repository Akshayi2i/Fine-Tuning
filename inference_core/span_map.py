"""Map generated JSON fields back to the tokens that produced them (arch §5).

Confidence starts here. The model returns text plus a logprob per token; this
module says *which tokens produced which field*, so calibration (IMPL-09) can
turn that into a per-field confidence. It contains no calibration logic itself —
it is pure, deterministic, and testable without a GPU.

Two design points that matter downstream:

**Unmappable fields are reported, never dropped.** A field whose span cannot be
located would otherwise get no confidence and look indistinguishable from a
clean, high-confidence extraction. That is the worst possible failure for a
signal whose entire job is flagging risk, so :class:`FieldSpan.mapped` exists and
callers are expected to check it.

**List rows are addressed individually** — ``claims[0].amount``, not ``claims``.
Per-value confidence within a row is meaningful; an aggregate over a whole table
is not. (Row *completeness* is a separate signal, because a missing row produces
no tokens at all — see IMPL-09.)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


class SpanMapError(ValueError):
    """Raised when the generated text is not parseable as JSON."""


@dataclass
class FieldSpan:
    """One field's location in the generated text, and its tokens."""

    field_path: str
    value: Any
    char_start: int
    char_end: int
    token_start: int = -1
    token_end: int = -1
    token_logprobs: list[float] = field(default_factory=list)
    mapped: bool = False
    reason: str | None = None      # why mapping failed, when it did
    #: The smallest margin over the value's tokens between the token written and
    #: the best other token (logprob, nats): near 0, the model all but wrote
    #: another value, and a run whose batch differs can write it
    #: (``serving.pipeline.near_tie_flags``). Negative when the decoder's
    #: schema masked the model's own first choice. None when the backend gave no
    #: second-best logprob for any of its tokens.
    top2_margin: float | None = None

    @property
    def token_count(self) -> int:
        return max(0, self.token_end - self.token_start)


# --------------------------------------------------------------------------
# JSON value scanning, with character offsets
# --------------------------------------------------------------------------
#
# The stdlib parser discards positions, so a small scanner records where each
# leaf value sits. The model is prompted to emit strict JSON with no fences, so
# this only needs to handle well-formed input — anything else is a schema-validity
# failure that IMPL-08 catches separately.

_WS = " \t\n\r"


class _Scanner:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in _WS:
            self.pos += 1

    def _expect(self, char: str) -> None:
        if self.pos >= len(self.text) or self.text[self.pos] != char:
            found = self.text[self.pos:self.pos + 1] or "<end of text>"
            raise SpanMapError(f"expected {char!r} at offset {self.pos}, found {found!r}")
        self.pos += 1

    def _scan_string(self) -> tuple[str, int, int]:
        start = self.pos
        self._expect('"')
        out: list[str] = []
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "\\":
                chunk = self.text[self.pos:self.pos + 6]
                decoded, consumed = self._decode_escape(chunk)
                out.append(decoded)
                self.pos += consumed
                continue
            if ch == '"':
                self.pos += 1
                return "".join(out), start, self.pos
            out.append(ch)
            self.pos += 1
        raise SpanMapError(f"unterminated string starting at offset {start}")

    @staticmethod
    def _decode_escape(chunk: str) -> tuple[str, int]:
        simple = {'"': '"', "\\": "\\", "/": "/", "b": "\b",
                  "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
        if len(chunk) < 2:
            raise SpanMapError("truncated escape sequence")
        marker = chunk[1]
        if marker in simple:
            return simple[marker], 2
        if marker == "u":
            return chr(int(chunk[2:6], 16)), 6
        raise SpanMapError(f"unknown escape sequence \\{marker}")

    def _scan_literal(self) -> tuple[Any, int, int]:
        start = self.pos
        for literal, value in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(literal, self.pos):
                self.pos += len(literal)
                return value, start, self.pos
        # number
        end = self.pos
        while end < len(self.text) and self.text[end] in "-+.eE0123456789":
            end += 1
        raw = self.text[self.pos:end]
        if not raw:
            raise SpanMapError(f"unexpected character at offset {self.pos}: {self.text[self.pos]!r}")
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise SpanMapError(f"invalid number {raw!r} at offset {self.pos}") from exc
        self.pos = end
        return value, start, end

    def scan(self, path: str = "") -> Iterator[FieldSpan]:
        """Yield a :class:`FieldSpan` for every leaf value, in document order."""
        self._skip_ws()
        if self.pos >= len(self.text):
            raise SpanMapError("empty generated text")

        char = self.text[self.pos]
        if char == "{":
            self.pos += 1
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == "}":
                self.pos += 1
                return
            while True:
                self._skip_ws()
                key, _ks, _ke = self._scan_string()
                self._skip_ws()
                self._expect(":")
                child = f"{path}.{key}" if path else key
                yield from self.scan(child)
                self._skip_ws()
                if self.pos < len(self.text) and self.text[self.pos] == ",":
                    self.pos += 1
                    continue
                self._expect("}")
                return

        if char == "[":
            self.pos += 1
            self._skip_ws()
            if self.pos < len(self.text) and self.text[self.pos] == "]":
                self.pos += 1
                return
            index = 0
            while True:
                yield from self.scan(f"{path}[{index}]")
                index += 1
                self._skip_ws()
                if self.pos < len(self.text) and self.text[self.pos] == ",":
                    self.pos += 1
                    continue
                self._expect("]")
                return

        if char == '"':
            value, start, end = self._scan_string()
            if end - start > 2:
                # The value's own characters, not its quotes: a quote shares its
                # token with the JSON around it (``": "``, ``",``), whose near-
                # certain logprob says the structure was predictable, not that
                # the value was read. An empty string keeps them: it has no other.
                start, end = start + 1, end - 1
        else:
            value, start, end = self._scan_literal()
        yield FieldSpan(field_path=path or "<root>", value=value, char_start=start, char_end=end)


def scan_json_spans(text: str) -> list[FieldSpan]:
    """Every leaf value in the generated JSON, with character offsets."""
    scanner = _Scanner(text)
    spans = list(scanner.scan())
    scanner._skip_ws()
    if scanner.pos != len(text):
        trailing = text[scanner.pos:scanner.pos + 40]
        raise SpanMapError(f"trailing content after the JSON value: {trailing!r}")
    return spans


# --------------------------------------------------------------------------
# Character offsets -> token indices
# --------------------------------------------------------------------------

def token_char_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    """Cumulative ``(start, end)`` character offsets for each token.

    Assumes the tokens concatenate to the generated text, which holds for the
    byte-level BPE tokenizers this model family uses. :func:`map_field_spans`
    verifies it rather than trusting it.
    """
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    return offsets


def _tokens_overlapping(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """Half-open token index range covering ``[start, end)``."""
    first, last = -1, -1
    for index, (t_start, t_end) in enumerate(offsets):
        if t_end <= start:
            continue
        if t_start >= end:
            break
        if first == -1:
            first = index
        last = index
    return (first, last + 1) if first != -1 else (-1, -1)


def top2_margin(logprobs: list[float], second: list[float | None] | None) -> float | None:
    """The smallest ``logprob - second-best logprob`` over a value's tokens;
    None when no token has a second-best."""
    if not second:
        return None
    margins = [first - other for first, other in zip(logprobs, second, strict=False) if other is not None]
    return round(min(margins), 6) if margins else None


def map_field_spans(
    generated_text: str,
    tokens: list[str],
    token_logprobs: list[float],
    *,
    strict: bool = False,
    token_top2_logprobs: list[float | None] | None = None,
) -> dict[str, FieldSpan]:
    """Map every generated field to its tokens and their logprobs.

    Args:
        generated_text: the model's raw output — strict JSON, no fences.
        tokens: generated token strings, in order.
        token_logprobs: one logprob per token, same order and length.
        strict: raise when the tokens do not reconstruct the text, instead of
            degrading to unmapped spans.
        token_top2_logprobs: per token, the best logprob of a token other than
            the one written (None where the backend gave none), same length -
            each span's :attr:`FieldSpan.top2_margin`. Left out, no span has one.

    Returns:
        ``field_path -> FieldSpan``. **Check ``mapped``**: an unmapped field has
        no confidence, and treating it as confident is exactly backwards.
    """
    if len(tokens) != len(token_logprobs):
        raise SpanMapError(
            f"{len(tokens)} tokens but {len(token_logprobs)} logprobs. They must align — "
            "a misaligned pair silently attributes one field's probability to another."
        )
    if token_top2_logprobs is not None and len(token_top2_logprobs) != len(tokens):
        # A second-best list that does not align is ignored, never misattributed.
        token_top2_logprobs = None

    spans = scan_json_spans(generated_text)
    offsets = token_char_offsets(tokens)
    reconstructed = "".join(tokens)
    tokens_align = reconstructed == generated_text

    if not tokens_align and strict:
        raise SpanMapError(
            "tokens do not reconstruct the generated text, so character offsets cannot be "
            f"trusted (text {len(generated_text)} chars, tokens {len(reconstructed)} chars). "
            "Check for a tokenizer that strips or normalises whitespace."
        )

    mapped: dict[str, FieldSpan] = {}
    for span in spans:
        if not tokens_align:
            span.reason = "tokens do not reconstruct the generated text"
            mapped[span.field_path] = span
            continue

        start, end = _tokens_overlapping(offsets, span.char_start, span.char_end)
        if start == -1:
            span.reason = f"no token overlaps characters [{span.char_start}, {span.char_end})"
        else:
            span.token_start, span.token_end = start, end
            span.token_logprobs = token_logprobs[start:end]
            span.top2_margin = top2_margin(
                span.token_logprobs, token_top2_logprobs[start:end] if token_top2_logprobs else None)
            span.mapped = True
        mapped[span.field_path] = span
    return mapped


def compact_envelope_paths(answer: Any, path: str = "") -> set[str]:
    """The paths, as :func:`scan_json_spans` writes them, of every compact
    envelope (``common.canonical.is_compact_value``) in a parsed answer."""
    from common.canonical import is_compact_value

    if is_compact_value(answer):
        return {path or "<root>"}
    found: set[str] = set()
    if isinstance(answer, dict):
        for key, value in answer.items():
            found |= compact_envelope_paths(value, f"{path}.{key}" if path else key)
    elif isinstance(answer, list):
        for index, value in enumerate(answer):
            found |= compact_envelope_paths(value, f"{path}[{index}]")
    return found


def map_compact_field_spans(
    generated_text: str,
    tokens: list[str],
    token_logprobs: list[float],
    *,
    strict: bool = False,
    token_top2_logprobs: list[float | None] | None = None,
) -> dict[str, FieldSpan]:
    """:func:`map_field_spans` for a compact answer (the fast format profile),
    keyed as the same answer written in full: an envelope's ``r`` span as its
    ``raw``, ``v`` as ``parsed``, ``p[i]`` as ``page_ref[i]``.

    Confidence then reads the value's own tokens, as it does on a full answer
    (``common.canonical.collapse_spans``): the printed text's, ``r``, where it
    was written; ``v`` where it was not - the printed text IS the value then,
    and ``v`` is where the model read it. Spans outside an envelope (a code, an
    id) keep their paths.
    """
    spans = map_field_spans(generated_text, tokens, token_logprobs, strict=strict,
                            token_top2_logprobs=token_top2_logprobs)
    try:
        envelopes = compact_envelope_paths(json.loads(generated_text))
    except ValueError as exc:
        raise SpanMapError(f"the generated text is not JSON: {exc}") from exc
    full = {"r": "raw", "v": "parsed", "p": "page_ref"}
    out: dict[str, FieldSpan] = {}
    for path, span in spans.items():
        head, _, tail = path.rpartition(".")
        key, bracket, index = tail.partition("[")
        if head in envelopes and key in full:
            span.field_path = f"{head}.{full[key]}{bracket}{index}"
        out[span.field_path] = span
    return out


def unmapped_fields(spans: dict[str, FieldSpan]) -> list[tuple[str, str]]:
    """``(field_path, reason)`` for every field that could not be mapped.

    Callers surface these rather than ignoring them: a field with no confidence
    must not be presented as a confident extraction.
    """
    return [(path, s.reason or "unknown") for path, s in spans.items() if not s.mapped]


def list_field_paths(spans: dict[str, FieldSpan]) -> dict[str, list[str]]:
    """Group list-row paths by their array, e.g. ``claims`` -> row field paths.

    Used by the row-completeness signal (IMPL-09), which needs to know how many
    rows were generated — the count logprobs cannot reveal, because a *missing*
    row emits no tokens at all.
    """
    grouped: dict[str, list[str]] = {}
    for path in spans:
        if "[" not in path:
            continue
        array_path = path.split("[", 1)[0]
        grouped.setdefault(array_path, []).append(path)
    return grouped


def row_count(spans: dict[str, FieldSpan], array_path: str) -> int:
    """How many rows of ``array_path`` were generated."""
    indices = set()
    prefix = f"{array_path}["
    for path in spans:
        if path.startswith(prefix):
            try:
                indices.add(int(path[len(prefix):].split("]", 1)[0]))
            except ValueError:
                continue
    return len(indices)
