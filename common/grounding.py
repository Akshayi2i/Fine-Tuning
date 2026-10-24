"""Whether a value is printed, and on which pages: one answer for every caller.

The matching the data audit has always used (``data_pipeline.audit``): case,
punctuation and line breaks are ignored, and a value is on a page when it
appears there as a phrase on word boundaries or, across line breaks and table
columns, as every one of its words. A value shorter than
:data:`MIN_CHECKABLE_CHARS` ("1", "NY") is on almost every page and is not
checked at all - it is neither grounded nor ungrounded.

One definition, because four places ask the same question and each had its own
answer: the audit and label verification matched words, the hallucination
metric and the calibration feature matched a bare substring of the whole text -
so "1" was always "found", and a wrong amount was never compared in the form
the page prints. :func:`ground` adds the page: printed on a page the value
cites, printed only on others, printed on no page sent, or not checkable.

Pure: no I/O, no heavy imports, safe for serving.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: Values this short ("Y", "1") match almost any page; they are not checked.
MIN_CHECKABLE_CHARS = 3

_NON_WORD = re.compile(r"[^0-9a-z]+")

#: :class:`Grounding` statuses.
ON_CITED_PAGE = "on_cited_page"
ON_OTHER_PAGE = "on_other_page"
NOT_PRINTED = "not_printed"
UNCHECKED = "unchecked"


def normalise(text: Any) -> str:
    """Case, punctuation and line breaks removed; words separated by one space."""
    return " ".join(_NON_WORD.sub(" ", str(text).casefold()).split())


_FIGURE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
#: Only figure characters: an amount or a count, not a code that holds digits.
_AMOUNT = re.compile(r"[\s$()+\-.,%]*\d[\d,]*(?:\.\d+)?[\s$()%]*")


class PageText:
    """A page's text normalised once, for every value looked up on it."""

    def __init__(self, text: str | None) -> None:
        self.padded = f" {normalise(text or '')} "
        self.words = set(self.padded.split())
        #: Every amount the page prints, as a number: "$7,579.67" and "7579.67"
        #: are one. Not a bare digit or two ("2 TOWN RD", "Page 3"): "$2.00"
        #: would be printed on every page - but "$25" is an amount.
        self.numbers = {number for number in map(_as_number, (
            figure for figure in _FIGURE.findall(text or "")
            if len(figure) >= 3 or any(mark in figure for mark in "$.,"))) if number is not None}


def _as_number(text: Any) -> float | None:
    """An amount or a count as a number (2 decimals), when ``text`` is only that."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return round(float(text), 2)
    if not isinstance(text, str) or not _AMOUNT.fullmatch(text):
        return None
    figures = re.sub(r"[^\d.]", "", text)
    try:
        return round(float(figures), 2)
    except ValueError:
        return None


def appears_on(value: str, page: PageText) -> bool:
    """Whether an already-normalised ``value`` is on ``page``: as a phrase, or -
    a phrase broken across lines or table columns - as every one of its words.
    Not so for a code of short pieces: "HX-0000-0001" is the words "hx", "0000"
    and "0001", which any page holding "HX-0000-0000" and a "0001" also holds.
    Only a value with a word of three letters or more is matched word by word."""
    if not value:
        return False
    if f" {value} " in page.padded:
        return True
    words = value.split()
    return (len(words) > 1 and any(sum(ch.isalpha() for ch in word) >= 3 for word in words)
            and set(words) <= page.words)


def on_page(raw: Any, page: PageText) -> bool:
    """Whether ``raw`` is printed on ``page``: as a phrase or every one of its
    words, or - an amount or a count - as the same figure in another format."""
    if appears_on(normalise(raw), page):
        return True
    number = _as_number(raw)
    return number is not None and number in page.numbers


def appears(raw: Any, page_text: str | None) -> bool:
    """Whether ``raw`` is on the page, as a phrase or - across line breaks and
    columns - as every one of its words, or as the same figure."""
    return on_page(raw, PageText(page_text))


def checkable(raw: Any) -> bool:
    """Whether ``raw`` is long enough to be looked for at all."""
    return raw is not None and len(normalise(raw)) >= MIN_CHECKABLE_CHARS


@dataclass(frozen=True)
class Grounding:
    """Where a value is printed, against the pages it cites."""

    status: str
    #: The pages that print it, among those searched.
    pages: tuple[int, ...] = ()

    @property
    def printed(self) -> bool:
        return self.status in (ON_CITED_PAGE, ON_OTHER_PAGE)


def ground(raw: Any, cited: Iterable[Any] | None, pages: Mapping[int, str | PageText | None]) -> Grounding:
    """Where ``raw`` is printed among ``pages`` (page number -> its text).

    * :data:`ON_CITED_PAGE` - on a page it cites (or, citing none, on any page);
    * :data:`ON_OTHER_PAGE` - only on pages it does not cite;
    * :data:`NOT_PRINTED` - on no page searched;
    * :data:`UNCHECKED` - too short to look for, no page has text, or a page it
      cites has no text (a scan the OCR could not read may still print it).
    """
    value = normalise(raw) if raw is not None else ""
    if len(value) < MIN_CHECKABLE_CHARS:
        return Grounding(UNCHECKED)
    texts = {int(number): text if isinstance(text, PageText) else PageText(text)
             for number, text in pages.items()}
    if not any(text.words for text in texts.values()):
        return Grounding(UNCHECKED)
    found = tuple(sorted(number for number, text in texts.items() if on_page(raw, text)))
    wanted = {int(page) for page in cited or () if str(page).lstrip("-").isdigit()}
    if found:
        return Grounding(ON_CITED_PAGE if not wanted or wanted & set(found) else ON_OTHER_PAGE, found)
    if any(number in texts and not texts[number].words for number in wanted):
        return Grounding(UNCHECKED)
    return Grounding(NOT_PRINTED)


# --------------------------------------------------------------------------
# OCR-tolerant: what a scan's text can settle (decision 2026-10-24)
# --------------------------------------------------------------------------

#: Letters OCR reads for digits, and digits it reads for letters ("NIP0115" as
#: "NIPO115", "$5" as "$S"). Folded to the digit in every word that holds a
#: digit, on both sides, before the words are compared. A word with no digit is
#: never folded: "SOLD" is not "5010".
_DIGIT_LOOKALIKES = str.maketrans({"o": "0", "q": "0", "i": "1", "l": "1", "s": "5", "b": "8",
                                   "z": "2", "g": "6"})

#: A word of this many letters or more matches a page's word one character
#: away (one dropped, added or changed: "lnsurance", "Insuranee"). Shorter
#: words, and every word holding a digit, must match once folded: one digit
#: changed is another number.
FUZZY_MIN_CHARS = 5


def _fold(word: str) -> str:
    return word.translate(_DIGIT_LOOKALIKES) if any(ch.isdigit() for ch in word) else word


def _deletions(word: str) -> set[str]:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def _readings(figure: str) -> set[float]:
    """The amounts a figure may be once OCR has misread its separators: as
    written, every separator a thousands mark ("1.000.000" is 1000000), or the
    last one the decimal point ("1.000,50" is 1000.5). Never a figure with its
    decimal point moved: "$2,500" is not "$25.00"."""
    out: set[float] = set()
    if (number := _as_number(figure)) is not None:
        out.add(number)
    digits = re.sub(r"\D", "", figure)
    if not digits:
        return out
    marks = [at for at, ch in enumerate(figure) if ch in ".,"]
    if not marks or len(figure) - marks[-1] - 1 != 2:
        # A last mark with three digits after it, or none, is a thousands mark.
        out.add(float(digits))
    if marks:
        whole, cents = re.sub(r"\D", "", figure[:marks[-1]]), re.sub(r"\D", "", figure[marks[-1] + 1:])
        if whole and cents:
            out.add(round(float(f"{whole}.{cents}"), 2))
    return out


class TolerantPage:
    """A page's text, prepared once for :func:`ground_tolerant`: normalised,
    its digit-holding words folded, and every figure as a run of digits."""

    def __init__(self, text: str | None) -> None:
        self.exact = PageText(text)
        words = self.exact.padded.split()
        self.words = {_fold(word) for word in words}
        self.padded = f" {' '.join(_fold(word) for word in words)} "
        #: Every amount the page may print, each figure read every way its
        #: separators allow (:func:`_readings`), and without a leading 5 where it
        #: shows no "$": a "$" read as a "5" ("$37.47" printed, "537.47" read).
        self.amounts: set[float] = set()
        for figure in _FIGURE.findall(text or ""):
            body = figure.lstrip("$")
            if len(re.sub(r"\D", "", body)) < MIN_CHECKABLE_CHARS:
                continue
            self.amounts |= _readings(body)
            if not figure.startswith("$") and body.startswith("5") and len(body) > MIN_CHECKABLE_CHARS:
                self.amounts |= _readings(body[1:])
        self._near: dict[str, set[str]] | None = None

    def near(self, word: str) -> bool:
        """Whether a page word is at most one character from ``word``."""
        if self._near is None:
            self._near = {}
            for page_word in self.words:
                if len(page_word) >= FUZZY_MIN_CHARS - 1 and not any(ch.isdigit() for ch in page_word):
                    self._near.setdefault(page_word, set()).add(page_word)
                    for deleted in _deletions(page_word):
                        self._near.setdefault(deleted, set()).add(page_word)
        return word in self._near or any(deleted in self._near for deleted in _deletions(word))

    @property
    def has_text(self) -> bool:
        return bool(self.exact.words)


def tolerant_pages(pages: Mapping[int, str | TolerantPage | None]) -> dict[int, TolerantPage]:
    """Each page prepared for :func:`ground_tolerant`, once per document: a
    policy has hundreds of values to look up on the same pages."""
    return {int(number): text if isinstance(text, TolerantPage) else TolerantPage(text)
            for number, text in pages.items()}


def _tolerantly_on(value: str, raw: Any, page: TolerantPage) -> bool:
    if on_page(raw, page.exact):
        return True
    words = [_fold(word) for word in value.split()]
    if f" {' '.join(words)} " in page.padded:
        return True
    number = _as_number(raw)
    if number is not None and len(re.sub(r"\D", "", str(raw))) >= MIN_CHECKABLE_CHARS and number in page.amounts:
        return True
    if not words or not any(sum(ch.isalpha() for ch in word) >= 3 for word in words) and len(words) > 1:
        return False
    for word in words:
        if word in page.words:
            continue
        if len(word) >= FUZZY_MIN_CHARS and not any(ch.isdigit() for ch in word) and page.near(word):
            continue
        return False
    return True


def ground_tolerant(raw: Any, cited: Iterable[Any] | None,
                    pages: Mapping[int, str | TolerantPage | None]) -> Grounding:
    """:func:`ground`, tolerant of what OCR does to a scan.

    A scan's text misreads, so on its own word a value is never "printed
    nowhere" (``data_pipeline.ingestion.label_rules``). Where a value must be
    printed to be served (``serving.printed_values``, decision 2026-10-24) the
    search forgives the misreads OCR makes, and only them: digit lookalikes in a
    word holding a digit (:data:`_DIGIT_LOOKALIKES`), one character in a word of
    :data:`FUZZY_MIN_CHARS` letters or more, a figure's separators, and a "$"
    read as a "5". Every page is searched; the statuses are :func:`ground`'s.
    A ``bool`` is a reading, not a printed value: unchecked.
    """
    if isinstance(raw, bool):
        return Grounding(UNCHECKED)
    value = normalise(raw) if raw is not None else ""
    if len(value) < MIN_CHECKABLE_CHARS:
        return Grounding(UNCHECKED)
    texts = tolerant_pages(pages)
    if not any(text.has_text for text in texts.values()):
        return Grounding(UNCHECKED)
    found = tuple(sorted(number for number, text in texts.items() if _tolerantly_on(value, raw, text)))
    wanted = {int(page) for page in cited or () if str(page).lstrip("-").isdigit()}
    if found:
        return Grounding(ON_CITED_PAGE if not wanted or wanted & set(found) else ON_OTHER_PAGE, found)
    if any(number in texts and not texts[number].has_text for number in wanted):
        return Grounding(UNCHECKED)
    return Grounding(NOT_PRINTED)
