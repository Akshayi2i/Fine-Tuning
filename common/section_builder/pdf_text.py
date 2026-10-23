"""
Character-level text of digital PDFs, from PyMuPDF's text trace.

Every character carries its page, box, baseline origin, font, size, colour and
render mode (0 visible, 3 invisible - e.g. an OCR text layer under a scan). This is
what the rewrite engine needs to put a replacement back exactly where and how the
original was printed, and what the checks use to read a twin back.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pymupdf

FLAG_ITALIC, FLAG_SERIF, FLAG_MONO, FLAG_BOLD = 2, 4, 8, 16


@dataclass
class Char:
    c: str
    bbox: tuple[float, float, float, float]
    origin: tuple[float, float]


@dataclass
class Span:
    page: int                      # 1-based
    font: str
    size: float
    color: tuple[float, ...]
    flags: int
    render_mode: int
    opacity: float
    chars: list[Char]
    seqno: int = 0
    dir: tuple[float, float] = (1.0, 0.0)

    @property
    def text(self) -> str:
        return "".join(ch.c for ch in self.chars)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs0 = [c.bbox[0] for c in self.chars]; ys0 = [c.bbox[1] for c in self.chars]
        xs1 = [c.bbox[2] for c in self.chars]; ys1 = [c.bbox[3] for c in self.chars]
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    @property
    def bold(self) -> bool:
        return bool(self.flags & FLAG_BOLD) or "bold" in self.font.lower()

    @property
    def visible(self) -> bool:
        return self.render_mode != 3 and self.opacity > 0


def spans(page: pymupdf.Page) -> list[Span]:
    out = []
    for s in page.get_texttrace():
        chars = [Char(chr(u) if u > 0 else "�", tuple(bb), tuple(org)) for u, _g, org, bb in s["chars"]]
        if not chars:
            continue
        out.append(Span(page.number + 1, s["font"], float(s["size"]), tuple(s.get("color") or (0, 0, 0)),
                        int(s.get("flags", 0)), int(s.get("type", 0)), float(s.get("opacity", 1.0)),
                        chars, int(s.get("seqno", 0)), tuple(s.get("dir", (1.0, 0.0)))))
    return out


@dataclass
class Line:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool
    refs: list[tuple[Span, int] | None] = field(default_factory=list)   # one per text char; None = inferred space

    @property
    def spans(self) -> list[Span]:
        seen, out = set(), []
        for r in self.refs:
            if r is not None and id(r[0]) not in seen:
                seen.add(id(r[0])); out.append(r[0])
        return out


def _segments(sp: list[Span]):
    """Cut spans into runs of characters on one baseline moving left to right."""
    segs = []
    for s in sp:
        cur = []
        for i, ch in enumerate(s.chars):
            if cur:
                pch = s.chars[cur[-1]]
                if abs(ch.origin[1] - pch.origin[1]) > max(1.0, 0.3 * s.size) or ch.origin[0] < pch.origin[0] - 0.5 * s.size:
                    segs.append((s, cur)); cur = []
            cur.append(i)
        if cur:
            segs.append((s, cur))
    return segs


def _split_interleaved(row: list) -> list:
    """Cut a segment at a gap where another segment of the same row starts. A content stream
    can draw '...AC DEC 04 07' and '2R10915' as one run with 'Policy #' drawn later into the
    gap between them, and a value redrawn into a longer span lands in the gap it left; sorted
    as whole segments, either would read after the run ('2R10915Policy #'). Segments that do
    not start inside another's gap (e.g. an OCR copy over the same text) are left as they are."""
    changed = True
    while changed:
        changed = False
        starts = [(id(g), g[0].chars[g[1][0]].bbox[0]) for g in row]
        for gi, (s, idx) in enumerate(row):
            for k in range(1, len(idx)):
                a, b = s.chars[idx[k - 1]], s.chars[idx[k]]
                if b.bbox[0] - a.bbox[2] <= 0.1 * max(s.size, 1):
                    continue
                if any(oid != id(row[gi]) and a.bbox[0] < x < b.bbox[0] for oid, x in starts):
                    row = row[:gi] + [(s, idx[:k]), (s, idx[k:])] + row[gi + 1:]
                    changed = True
                    break
            if changed:
                break
    return row


def lines(page: pymupdf.Page, visible_only: bool = True) -> list[Line]:
    """Visual lines: characters sharing a baseline, left to right, top to bottom.
    A space is inferred wherever the gap between two characters exceeds 0.2 em."""
    sp = [s for s in spans(page) if (s.visible or not visible_only)]
    segs = _segments(sp)
    segs = [g for g in segs if "".join(g[0].chars[i].c for i in g[1]).strip()]
    segs.sort(key=lambda g: (g[0].chars[g[1][0]].origin[1], g[0].chars[g[1][0]].origin[0]))
    rows: list[list] = []
    for g in segs:
        y = g[0].chars[g[1][0]].origin[1]
        if rows and abs(rows[-1][0][0].chars[rows[-1][0][1][0]].origin[1] - y) <= max(1.5, 0.3 * g[0].size):
            rows[-1].append(g)
        else:
            rows.append([g])
    out = []
    for r in rows:
        r = _split_interleaved(r)
        r.sort(key=lambda g: g[0].chars[g[1][0]].origin[0])
        refs: list[tuple[Span, int] | None] = []
        text = []
        prev = None
        for s, idx in r:
            for i in idx:
                ch = s.chars[i]
                if prev is not None:
                    ps, pi = prev
                    pch = ps.chars[pi]
                    gap = ch.bbox[0] - pch.bbox[2]
                    if ch.c != " " and pch.c != " " and gap > 0.2 * max(s.size, ps.size, 1):
                        sep = "   " if gap > 1.5 * max(s.size, 1) else " "
                        for c in sep:
                            text.append(c); refs.append(None)
                    elif ch.c != " " and pch.c == " " and gap > 1.5 * max(s.size, 1):
                        # a span ending in a space ("Coverage Summary ") before a far column: still a column gap
                        for c in "  ":
                            text.append(c); refs.append(None)
                text.append(ch.c); refs.append((s, i))
                prev = (s, i)
        boxes = [x[0].chars[x[1]].bbox for x in refs if x is not None]
        bb = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        main = max(r, key=lambda g: len(g[1]))[0]
        out.append(Line(page.number + 1, "".join(text), bb, main.size, main.bold, refs))
    return out


def page_text(page: pymupdf.Page, visible_only: bool = True) -> str:
    return "\n".join(l.text for l in lines(page, visible_only))


def norm(s: str) -> str:
    return " ".join((s or "").split())


@dataclass
class Hit:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    chars: list[tuple[Span, int]]          # (span, char index) for every matched char
    layer: str                              # "visible" | "invisible"


def _stream(page: pymupdf.Page):
    """All characters of a page in reading order, line by line, with the (span, index)
    of each real character; inferred spaces and line breaks carry None."""
    seq: list[tuple[str, tuple[Span, int] | None]] = []
    for ln in lines(page, visible_only=False):
        for c, r in zip(ln.text, ln.refs, strict=False):
            seq.append((c, r))
        seq.append(("\n", None))
    return seq


def num_bounds(n: str, zip_plus4: bool = False) -> tuple[str, str]:
    """Lookarounds that keep a value from matching inside a longer token: "$250" is not in
    "$250,000"; a value starting or ending with a digit is not part of a date ("02/15" is not in
    "02/15/2021"), while "$250" after "$1,000,000/" still is. With ``zip_plus4`` (leak counts only)
    a ZIP standing on its own is not the head of a ZIP+4 ("13360" is not in "13360-0329"): another
    party's ZIP+4 is no leak of a moved town, while C2 still finds a ZIP printed as its ZIP+4. The letter/digit edge applies only where the value itself
    starts or ends with a letter or digit, so "$35.17" glued to "50/100$35.17" is a whole token.
    The "/" guard applies only to a digit run short enough to be a date part (4 digits or fewer):
    "250000" and "500000" are whole in the split limit "250000/500000" (CR-PU-8)."""
    lead, trail = re.match(r"\d+", n), re.search(r"\d+$", n)
    pre = r"(?<!\d[.,/])" if lead and len(lead.group()) <= 4 else r"(?<!\d[.,])"
    post = r"(?![.,/]\d)" if trail and len(trail.group()) <= 4 else r"(?![.,]\d)"
    if zip_plus4 and re.search(r"(?:^|[\s,])\d{5}$", n):
        post = r"(?![.,/]\d|-\d{4}(?!\d))"
    if n[:1].isalnum():
        pre = r"(?<![A-Za-z0-9])" + pre
    if n[-1:].isalnum():
        post = r"(?![A-Za-z0-9])" + post
    return pre, post


def find(doc: pymupdf.Document, text: str, pages: Iterable[int] | None = None,
         case: bool = True, whole: bool = True) -> list[Hit]:
    """Every occurrence of ``text`` (whitespace-insensitive), with its box.
    ``whole`` requires the match not to be part of a longer alphanumeric token or number
    ('$500' is not found inside '$500,000')."""
    target = norm(text)
    if not target:
        return []
    words = [re.escape(w) for w in target.split(" ")]
    pat = r"\s+".join(words)
    if whole:
        # a value is not part of a longer number; one starting or ending with a digit is not part of
        # a date ("02/15" in "02/15/2021"), while "$250" after "$1,000,000/" still matches
        pre, post = num_bounds(target)
        pat = pre + pat + post
    rx = re.compile(pat, 0 if case else re.I)
    hits = []
    for pno in (pages or range(1, doc.page_count + 1)):
        seq = _stream(doc[pno - 1])
        s = "".join(c for c, _ in seq)
        for m in rx.finditer(s):
            refs = [seq[i][1] for i in range(m.start(), m.end()) if seq[i][1] is not None]
            if not refs:
                continue
            boxes = [r[0].chars[r[1]].bbox for r in refs]
            bb = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
            layer = "invisible" if all(not r[0].visible for r in refs) else "visible"
            hits.append(Hit(pno, m.group(0), bb, refs, layer))
    return hits


def chars_in(page: pymupdf.Page, bbox: Sequence[float], tol: float = 0.5) -> list[tuple[Span, int]]:
    """Characters whose centre lies inside ``bbox`` (grown by ``tol``)."""
    x0, y0, x1, y1 = bbox[0] - tol, bbox[1] - tol, bbox[2] + tol, bbox[3] + tol
    out = []
    for s in spans(page):
        for i, ch in enumerate(s.chars):
            cx = (ch.bbox[0] + ch.bbox[2]) / 2
            cy = (ch.bbox[1] + ch.bbox[3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                out.append((s, i))
    return out


def text_in(page: pymupdf.Page, bbox: Sequence[float], tol: float = 0.5) -> str:
    refs = chars_in(page, bbox, tol)
    refs.sort(key=lambda r: (round(r[0].chars[r[1]].origin[1]), r[0].chars[r[1]].origin[0]))
    out, prev = [], None
    for s, i in refs:
        ch = s.chars[i]
        if prev is not None:
            if abs(prev.origin[1] - ch.origin[1]) > 2:
                out.append(" ")
            elif ch.bbox[0] - prev.bbox[2] > 0.2 * s.size and not out[-1].endswith(" ") and ch.c != " ":
                out.append(" ")
        out.append(ch.c)
        prev = ch
    return norm("".join(out))


def has_text_layer(doc: pymupdf.Document, min_chars: int = 50) -> bool:
    return sum(len(p.get_text().strip()) for p in doc) >= min_chars


def split_by_line(h: Hit) -> list[Hit]:
    """A hit that wraps over several lines, as one hit per line (same layer). A
    variation-map occurrence must sit on one baseline (SPEC_21 §7.6)."""
    lines_: dict[float, list[tuple[Span, int]]] = {}
    for s, i in h.chars:
        y = round(s.chars[i].origin[1], 0)
        key = next((k for k in lines_ if abs(k - y) <= 1.5), y)
        lines_.setdefault(key, []).append((s, i))
    if len(lines_) <= 1:
        return [h]
    out = []
    for y in sorted(lines_):
        refs = sorted(lines_[y], key=lambda r: r[0].chars[r[1]].origin[0])
        boxes = [r[0].chars[r[1]].bbox for r in refs]
        txt, prev = [], None
        for s, i in refs:
            ch = s.chars[i]
            if prev is not None and ch.bbox[0] - prev.bbox[2] > 0.2 * s.size:
                txt.append(" ")
            txt.append(ch.c)
            prev = ch
        bb = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        out.append(Hit(h.page, norm("".join(txt)), bb, refs, h.layer))
    return out
