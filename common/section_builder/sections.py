"""
Deterministic section builder for ``text_sections`` (SPEC_21 §4.6, §7.2).

The same code builds the full-text tier for seed golds, for twin golds, in L1 and at
inference, so the structure is identical everywhere. Every printed line lands in
exactly one section, which is what makes the every-word rule (check C4) hold by
construction.

Input is a list of :class:`PageLines` - from a digital PDF (:func:`pages_from_pdf`),
from corrected OCR words (:func:`pages_from_ocr`) or from MinerU markdown
(:func:`pages_from_markdown`).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import pymupdf

from . import pdf_text as T

# -- page model -------------------------------------------------------------------------------


@dataclass
class L:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool = False


@dataclass
class Table:
    bbox: tuple[float, float, float, float]
    markdown: str


@dataclass
class PageLines:
    page: int
    width: float
    height: float
    lines: list[L]
    tables: list[Table] = field(default_factory=list)


def pages_from_pdf(doc: pymupdf.Document, tables: bool = True, visible_only: bool = False) -> list[PageLines]:
    """Lines of every text layer, visible and invisible, as pdfplumber reads the page at runtime and as C4
    counts the words (decision of 19 October 2026, row 18): a full-page image with an invisible text layer
    and one visible overlay line is sectioned from its text layer. This reads text a page already carries;
    it is not OCR. ``visible_only`` keeps the visible layer only (diagnostics)."""
    out = []
    for p in doc:
        ls = [L(l.text, l.bbox, l.size, l.bold) for l in T.lines(p, visible_only=visible_only)]
        tbs: list[Table] = []
        if tables:
            try:
                for t in p.find_tables().tables:
                    md = t.to_markdown(clean=False).strip()
                    if md and t.row_count >= 2 and t.col_count >= 2 and _table_is_printed(md, t.bbox, ls):
                        tbs.append(Table(tuple(t.bbox), md))
            except Exception:
                tbs = []
        out.append(PageLines(p.number + 1, p.rect.width, p.rect.height, ls, tbs))
    return out


def _table_is_printed(md: str, bbox, lines: list[L]) -> bool:
    """Full text is verbatim (§4.6): a detected table is kept only when every word of its markdown,
    counted with repeats, is printed inside the table's box. Merged cells copied into two columns,
    an empty column filled with a neighbour's "$0.00", or the clipped descenders of a line written
    as strikeout ("~~, y g y py~~") fail, and the region is written as its printed lines instead."""
    cells = re.sub(r"<br>|\*\*|~~|\|-{3,}|Col\d+", " ", md)
    need = Counter(re.findall(r"[A-Za-z0-9]+", cells))
    x0, y0, x1, y1 = bbox
    have = Counter(w for l in lines
                   if x0 - 1 <= (l.bbox[0] + l.bbox[2]) / 2 <= x1 + 1 and y0 - 1 <= (l.bbox[1] + l.bbox[3]) / 2 <= y1 + 1
                   for w in re.findall(r"[A-Za-z0-9]+", l.text))
    return bool(need) and all(have[w] >= c for w, c in need.items())


def pages_from_ocr(ocr: dict) -> list[PageLines]:
    """``ocr.json``: {"pages": [{"page": 1, "width": w, "height": h,
    "words": [{"text": "...", "bbox": [x0,y0,x1,y1], "conf": 0.98, "section_type": null}]}]}.
    Words are grouped into lines by vertical overlap."""
    out = []
    for pg in ocr.get("pages", []):
        words = sorted(pg.get("words", []), key=lambda w: ((w["bbox"][1] + w["bbox"][3]) / 2, w["bbox"][0]))
        rows: list[list] = []
        for w in words:
            cy = (w["bbox"][1] + w["bbox"][3]) / 2
            h = w["bbox"][3] - w["bbox"][1]
            if rows:
                r = rows[-1]
                rcy = sum((x["bbox"][1] + x["bbox"][3]) / 2 for x in r) / len(r)
                if abs(rcy - cy) <= 0.5 * h:
                    r.append(w)
                    continue
            rows.append([w])
        ls = []
        for r in rows:
            r.sort(key=lambda w: w["bbox"][0])
            parts, prev = [], None
            for w in r:
                if prev is not None:
                    gap = w["bbox"][0] - prev["bbox"][2]
                    parts.append("   " if gap > 2.5 * (prev["bbox"][3] - prev["bbox"][1]) else " ")
                parts.append(w["text"])
                prev = w
            bb = (min(w["bbox"][0] for w in r), min(w["bbox"][1] for w in r),
                  max(w["bbox"][2] for w in r), max(w["bbox"][3] for w in r))
            ls.append(L("".join(parts), bb, statistics.median(w["bbox"][3] - w["bbox"][1] for w in r), False))
        out.append(PageLines(pg["page"], pg.get("width", 2550), pg.get("height", 3300), ls))
    return out


def pages_from_markdown(md_pages: Sequence[str]) -> list[PageLines]:
    """MinerU markdown, one string per page. Headings ('#') become larger bold lines;
    markdown/HTML tables become tables."""
    out = []
    for i, md in enumerate(md_pages, 1):
        ls, tbs, y = [], [], 0.0
        buf: list[str] = []

        def flush_table():
            nonlocal buf, y
            if buf:
                # Called only inside the iteration that defines it, so `tbs` is the current page's list.
                tbs.append(Table((0, y, 600, y + 12 * len(buf)), "\n".join(buf)))  # noqa: B023
                y += 12 * len(buf)
                buf = []
        for raw in md.splitlines():
            s = raw.rstrip()
            if s.strip().startswith("|") or re.match(r"\s*<t(able|r|d)", s):
                buf.append(s)
                continue
            flush_table()
            if not s.strip():
                continue
            m = re.match(r"^(#+)\s*(.*)$", s)
            if m:
                ls.append(L(m.group(2), (0, y, 600, y + 14), 14 + max(0, 4 - len(m.group(1))), True))
            else:
                ls.append(L(s.strip(), (0, y, 600, y + 10), 10, False))
            y += 14
        flush_table()
        out.append(PageLines(i, 612, max(792, y + 40), ls, tbs))
    return out


# -- forms -------------------------------------------------------------------------------------

FORM_PATTERNS = [
    re.compile(r"\b([A-Z]{1,4}\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2})\b(?![-/]\d)"),            # ISO: HO 00 03 05 11 (not "...-01")
    # "Form 6489 NY (06/21)": the number with a following state code; the edition is read separately
    re.compile(r"\bForm\s*(?:No\.?|Number|#)?\s*:?\s*([A-Z0-9][A-Z0-9\-./]{3,24}(?:\s[A-Z]{2}\b(?!\s*\d{4,5}))?)", re.I),
    # form number + edition date. The date must be SEPARATED from the number (space, "(" or "Ed"):
    # "HPP00177450-01" (a policy number) and "NY 13335-1234" (state + ZIP) are not "form + edition".
    # "PL-50776 NY (11-23)": a state code may stand between the number and a bracketed edition
    re.compile(r"\b([A-Z]{1,5}[- ]?\d{2,6}(?:[- ]\d{2,6})?(?:\s[A-Z]{2}(?=\s*\())?)(?:\s+|\s*\(|\s*Ed\.?\s*)\(?(?:Ed\.?\s*)?\d{1,2}[/-]\d{2,4}\)?(?![\d/-])"),
    # carrier form code at the start of a line: ALL-21101-1106
    re.compile(r"^\s*([A-Z]{2,5}-\d{3,6}[A-Z]?-\d{2,6})\b"),
    # letter words + spaced edition, ISO style without a number: "AC DEC 04 07"
    re.compile(r"\b([A-Z]{2,4}(?:\s[A-Z]{2,4}){1,2}\s(?:0[1-9]|1[0-2])\s\d{2})\b(?!\s?[/-]?\d)"),
]
#: a letters-only form code is a form number only with its edition beside it: "LCIC(4/11)", "NCIC 07/14"
LETTER_FORM = re.compile(r"\b([A-Z]{3,6})\s?\((?:0?[1-9]|1[0-2])/\d{2,4}\)|\b([A-Z]{3,6})\s(?:0[1-9]|1[0-2])/\d{2}\s*$")
#: words printed before a date that are not form codes ("DATE 01/21", "EXP 05/27")
NOT_FORM = {"DATE", "DATED", "EXP", "EFF", "ISSUED", "PRINTED", "PAGE", "THRU", "FROM", "TERM", "ASOF"}
#: A candidate printed right after one of these labels is a policy, account or claim identifier,
#: not a form number ("Policy #: HPP00177450-01").
ID_LABEL = re.compile(r"\b(?:policy|pol|file|account|acct|claim|certificate|cert|loan|reference|ref|quote|"
                      r"binder|member|customer|invoice|agency|producer|agent|insured|id)\.?\s*"
                      r"(?:nos?\.?|numbers?|number\(s\)|num|#)?\s*[:#]?\s*$", re.I)
#: Only the header/footer zone and the first lines of a page name the page's form; a forms-schedule
#: row in the page body ("ML-216 06/99 Premises Alarm") does not.
FORM_HEAD_LINES = 5
EDITION = re.compile(r"(\d{1,2})[/-]?\s?(\d{2,4})\)?\s*$")

#: "policy view": the title of an agency-portal screen print (Allstate POLICY VIEW), printed smaller than
#: the section labels below it ("Contact Information"), which are not the document's title
TITLE_RX = re.compile(r"declarations?\b|coverage summary|policy summary|endorsement|amendatory|"
                      r"notice|application|policy jacket|coverage form|coverage part|policy view", re.I)

FORM_TYPES = [
    ("declarations", re.compile(r"declarations|coverage summary|policy summary|schedule of coverages", re.I)),
    ("endorsement", re.compile(r"endorsement|this endorsement changes|amendatory|rider", re.I)),
    ("schedule", re.compile(r"\bschedule\b", re.I)),
    ("notice", re.compile(r"notice|privacy|disclosure|important information|policyholder information", re.I)),
    ("application", re.compile(r"application|questionnaire", re.I)),
    ("correspondence", re.compile(r"^dear\b|thank you for", re.I)),
    ("coverage_form", re.compile(r"coverage form|coverage part|policy provisions|conditions|insuring agreement|policy jacket|definitions", re.I)),
]


def _norm_repeat(s: str) -> str:
    return re.sub(r"\d+", "#", " ".join(s.split()).lower())


def _find_form(text: str) -> str | None:
    for rx in FORM_PATTERNS:
        for m in rx.finditer(text):
            if ID_LABEL.search(text[: m.start(1)]):
                continue                      # an identifier after its label, not a form number
            f = " ".join(m.group(1).split())
            if len(re.sub(r"\D", "", f)) >= 3:
                return f
    for m in LETTER_FORM.finditer(text):
        f = m.group(1) or m.group(2)
        if f not in NOT_FORM and not ID_LABEL.search(text[: m.start()]):
            return f
    return None


def _edition(form: str | None, line: str = "") -> str | None:
    """Edition date: inside an ISO form number, in "(MM/YY)" after the number, or as "MM/YY"
    printed right after it on its line."""
    if not form:
        return None
    if line:
        m = re.search(re.escape(form) + r"\s*\(?\s*(?:Ed\.?\s*)?(\d{1,2})[/-](\d{2,4})\)?", line)
        if m:
            return "%s/%s" % (m.group(1), m.group(2))
    m = re.match(r"^[A-Z]{1,4}\s?\d{2}\s?\d{2}\s?(\d{2})\s?(\d{2})$", form) or \
        re.match(r"^[A-Z]{2,4}(?:\s[A-Z]{2,4}){1,2}\s(\d{2})\s(\d{2})$", form)
    if m:
        return "%s/%s" % (m.group(1), m.group(2))
    m = re.search(r"\((\d{1,2})[/-](\d{2,4})\)", form)
    if m:
        return "%s/%s" % (m.group(1), m.group(2))
    return None


# -- headings ----------------------------------------------------------------------------------

L1_RX = re.compile(r"^(SECTION|PART|ARTICLE|COVERAGE\s+[A-Z]\b|SCHEDULE\b|DEFINITIONS\b|CONDITIONS\b|EXCLUSIONS\b)", re.I)
L2_RX = re.compile(r"^([A-Z]\.|[IVX]{1,5}\.)\s+\S")
L3_RX = re.compile(r"^(\d{1,2}\.|\(\w{1,3}\))\s+\S")
#: a company name printed as a letterhead line ("American Reliable Insurance Company",
#: "Hagerty Insurance Agency, LLC."): who issued the page, not a section of it
ORG_RX = re.compile(r"^[A-Z][\w&.,' -]*\b(?:company|co|corp|corporation|llc|inc|incorporated|ltd|group|exchange|"
                    r"mutual|association|agency|underwriters)\.?$", re.I)
VALUE_RX = re.compile(r"\$\s*\d|\d{1,2}/\d{1,2}/\d{2,4}|\d{3}[-.)\s]\d{3}[-.]\d{4}|\d{5,}|"
                      r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b")


def _heading_level(l: L, body: float, values: frozenset[str] = frozenset()) -> int:
    """``values``: printed values the variation map changes (lower-cased); a line holding one is
    content, never a heading, so section keys do not change from twin to twin."""
    t = l.text.strip()
    letters = [c for c in t if c.isalpha()]
    if not letters or len(t) > 90:
        return 0
    if t[0].islower():                            # a continued sentence, not a heading
        return 0
    low = " ".join(t.lower().split())
    if any(v in low for v in values):
        return 0
    if ORG_RX.match(t) and not L1_RX.match(t):
        return 0
    if re.fullmatch(r"[A-Za-z]+:", t):            # one field label ("Date:", "Agent:"), its value elsewhere
        return 0
    if re.match(r"\d{2,6}\s+[A-Za-z]", t):        # a street line ("8667 E Hartford Dr Suite 225")
        return 0
    caps = sum(c.isupper() for c in letters) / len(letters) > 0.85 and len(letters) >= 3
    big = l.size >= 1.15 * body
    numbered = bool(L1_RX.match(t) or L2_RX.match(t) or L3_RX.match(t))
    emph = l.bold or big or (caps and numbered)
    if not emph:
        return 0
    # a line carrying a value (date, amount, long number) is a label/value line, not a heading
    if VALUE_RX.search(t) or sum(c.isdigit() for c in t) > 0.3 * len(t):
        return 0
    if "   " in t and not numbered:          # several columns on one line: a table row
        return 0
    if t.endswith((".", ",", ";")) and not (L2_RX.match(t) or L3_RX.match(t)) and len(t) > 40:
        return 0
    if L1_RX.match(t) or l.size >= 1.3 * body:
        return 1
    if L2_RX.match(t):
        return 2
    if L3_RX.match(t):
        return 3
    return 1 if big else 2


def _sub_token(t: str) -> str:
    m = L2_RX.match(t) or L3_RX.match(t)
    return m.group(1).rstrip(".").strip("()") if m else " ".join(t.split())[:60]


# -- builder -----------------------------------------------------------------------------------


def build(pages: list[PageLines], header_zone: float = 0.07, repeat_share: float = 0.25,
          values: Sequence[str] = ()) -> dict[str, dict]:
    """``values``: printed values of the variation map (a seed's occurrences, or a twin's new values);
    lines holding them are never headings or form titles."""
    vals = frozenset(" ".join(v.lower().split()) for v in values if len(" ".join(v.split())) >= 4)
    if not pages:
        return {}
    sizes = [l.size for p in pages for l in p.lines if l.text.strip()]
    body = statistics.median(sizes) if sizes else 10.0

    # header/footer: lines in the top or bottom band that repeat across pages
    def in_zone(l: L, p: PageLines) -> bool:
        cy = (l.bbox[1] + l.bbox[3]) / 2
        return cy < header_zone * p.height or cy > (1 - header_zone) * p.height

    zone_lines = Counter()
    for p in pages:
        seen = set()
        for l in p.lines:
            if in_zone(l, p):
                k = _norm_repeat(l.text)
                if k not in seen:
                    seen.add(k)
                    zone_lines[k] += 1
    min_rep = max(2, int(repeat_share * len(pages)))
    repeated = {k for k, c in zone_lines.items() if c >= min_rep}

    # form per page
    page_form: dict[int, str | None] = {}
    page_ed: dict[int, str | None] = {}
    prev = prev_ed = None
    for p in pages:
        found = ed = None
        zone = [l for l in p.lines if ((l.bbox[1] + l.bbox[3]) / 2) > (1 - 2 * header_zone) * p.height or ((l.bbox[1] + l.bbox[3]) / 2) < 2 * header_zone * p.height]
        for l in zone + p.lines[:FORM_HEAD_LINES]:
            found = _find_form(l.text)
            if found:
                ed = _edition(found, l.text)
                break
        page_form[p.page] = found or prev
        page_ed[p.page] = ed if found else prev_ed
        prev, prev_ed = page_form[p.page], page_ed[p.page]

    # form title and type: biggest non-repeated line on the form's first page
    form_meta: dict[str, dict] = {}
    for p in pages:
        f = page_form[p.page] or "NOFORM"
        if f in form_meta:
            continue
        cands = []
        for l in p.lines:
            # a line of several columns gives the column naming the document type, else its first one
            # ("Coverage Summary   Online Service", "North Country Insurance Company   Seasonal ... Declaration")
            segs = re.split(r"\s{3,}", l.text.strip())
            seg = next((x for x in segs if TITLE_RX.search(x)), segs[0])
            # a title followed by a value keeps the words before it ("... Policy Declarations Effective
            # 04/01/2021" -> "... Policy Declarations")
            m = VALUE_RX.search(seg)
            if m and TITLE_RX.search(seg[: m.start()]):
                seg = re.sub(r"(?:\s+(?:effective|eff\.?|dated|as of|on|for|from))*[\s:,-]*$", "",
                             seg[: m.start()], flags=re.I)
            low = " ".join(seg.lower().split())
            if in_zone(l, p) and _norm_repeat(l.text) in repeated or len(seg) <= 3 or VALUE_RX.search(seg) \
                    or any(v in low for v in vals) or ORG_RX.match(seg):
                continue
            if seg.endswith(".") and len(seg) > 60 or len(seg) > 90:     # a sentence (disclaimer), not a title
                continue
            if seg.endswith(":") or "#" in seg:            # a field label or a table header row
                continue
            if re.search(r"intentionally\s+(?:left\s+)?blank", seg, re.I):
                continue
            cands.append((l, seg))
        title = None
        if cands:
            # a line naming the document type in the top half of the page wins ("Personal Umbrella
            # Declarations Page" over a box label "Summary of"), then the largest line; a title wrapped
            # onto the next line joins it
            top, seg = max(cands, key=lambda c: (bool(TITLE_RX.search(c[1])) and c[0].bbox[1] < 0.5 * p.height,
                                                 round(c[0].size, 1), c[0].bold, -c[0].bbox[1]))
            title = " ".join(seg.split())
            cands = [c[0] for c in cands]
            nxt = [l for l in cands if l is not top and abs(l.size - top.size) < 0.3 and l.bold == top.bold
                   and 0 <= l.bbox[1] - top.bbox[3] < 0.8 * top.size and abs(l.bbox[0] - top.bbox[0]) < 3]
            if nxt:
                title += " " + " ".join(min(nxt, key=lambda l: l.bbox[1]).text.split())
        blob = " ".join(l.text for l in p.lines[:25])
        ftype = next((t for t, rx in FORM_TYPES if rx.search(title or "") or rx.search(blob)), "other")
        form_meta[f] = {"title": title, "type": ftype, "edition": page_ed[p.page] or _edition(page_form[p.page])}

    sections: dict[str, dict] = {}

    def add(key, text, page, stype, form, path, title):
        if key in sections and sections[key]["section_type"] != stype:
            key = key + "|" + stype
        s = sections.get(key)
        if s is None:
            meta = form_meta.get(form or "NOFORM", {})
            s = sections[key] = {
                "section_id": key, "form_number": form, "form_title": meta.get("title"),
                "edition_date": meta.get("edition"), "form_type": meta.get("type", "other"),
                "section_path": list(path), "section_title": title, "section_type": stype,
                "raw_text": "", "page_range": [page, page]}
        s["raw_text"] = (s["raw_text"] + "\n" + text) if s["raw_text"] else text
        s["page_range"] = [min(s["page_range"][0], page), max(s["page_range"][1], page)]

    stack: list[str | None] = [None, None, None]
    cur_form = object()
    for p in pages:
        form = page_form[p.page]
        if form != cur_form:
            stack = [None, None, None]
            cur_form = form
        fkey = form or "NOFORM"
        in_table = []
        for _ti, t in enumerate(p.tables):
            in_table.append(t)
        emitted_tables = set()
        for l in sorted(p.lines, key=lambda l: (l.bbox[1], l.bbox[0])):
            if not l.text.strip():
                continue
            # only inside the band: the insured's name repeated in the page header is also printed in the
            # body of page 1, and that line belongs to its section
            if in_zone(l, p) and _norm_repeat(l.text) in repeated:
                add("%s|HEADER_FOOTER|" % fkey, " ".join(l.text.split()), p.page, "header_footer", form,
                    ["HEADER_FOOTER"], None)
                continue
            cx, cy = (l.bbox[0] + l.bbox[2]) / 2, (l.bbox[1] + l.bbox[3]) / 2
            tix = next((i for i, t in enumerate(p.tables)
                        if t.bbox[0] - 1 <= cx <= t.bbox[2] + 1 and t.bbox[1] - 1 <= cy <= t.bbox[3] + 1), None)
            top = stack[0] or "(start)"
            sub = " > ".join(x for x in stack[1:] if x)
            if tix is not None:
                if tix not in emitted_tables:
                    emitted_tables.add(tix)
                    key = "%s|%s|%s" % (fkey, top, (sub + " > " if sub else "") + "table %d.%d" % (p.page, tix + 1))
                    add(key, p.tables[tix].markdown, p.page, "table", form, [top] + [x for x in stack[1:] if x] + ["table"], None)
                continue
            lvl = _heading_level(l, body, vals)
            text = " ".join(l.text.split())
            if lvl == 1:
                stack = [text[:80], None, None]
            elif lvl == 2:
                stack = [stack[0], _sub_token(text), None]
            elif lvl == 3:
                stack = [stack[0], stack[1], _sub_token(text)]
            top = stack[0] or "(start)"
            sub = " > ".join(x for x in stack[1:] if x)
            key = "%s|%s|%s" % (fkey, top, sub)
            add(key, text, p.page, "prose", form, [top] + [x for x in stack[1:] if x], text if lvl else None)
    # short label-like sections
    for s in sections.values():
        if s["section_type"] == "prose":
            ls = s["raw_text"].split("\n")
            if ls and sum(len(x) for x in ls) / len(ls) < 35 and not re.search(r"[a-z]{3,}[.;:]\s", s["raw_text"]):
                s["section_type"] = "label"
    return sections


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def coverage(pdf_words: list[str], tiers_text: str) -> float:
    """Share of the PDF's word tokens present in the output (multiset, order-free)."""
    if not pdf_words:
        return 1.0
    have = Counter(words(tiers_text))
    need = Counter(pdf_words)
    hit = sum(min(c, have.get(w, 0)) for w, c in need.items())
    return hit / sum(need.values())
