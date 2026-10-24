"""Whether a value is printed, and on which pages: one answer for every caller.

The matching the data audit has always used (``data_pipeline.audit``): case,
punctuation and line breaks are ignored, and a value is on a page when it
appears there as a phrase on word boundaries or, across line breaks and table
columns, as every one of its printed tokens - each found whole, so a code
("PAP-2026-0042") is never three words found apart. A figure is compared as a
figure: whole ("500" is not the end of "1,500"), and an amount written as one
("$2,019.00") only with an amount the page prints as one, never with a year or
a code. A value shorter than :data:`MIN_CHECKABLE_CHARS` ("1", "NY") is on
almost every page and is not checked at all - it is neither grounded nor
ungrounded; serving's search looks for a two-character one as a whole token.

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

#: Serving's search (:func:`ground_tolerant`) looks for a value this short
#: ("NJ", "50") as a whole token: a word, or a figure. One character is on
#: every page, and a reading (:data:`_READINGS`) is not printed: both unchecked.
MIN_TOKEN_CHARS = 2

#: A box marked or not, a yes or a no: what the model read, not a value a page
#: prints - like a ``bool``. Unchecked by :func:`ground_tolerant`.
_READINGS = frozenset({"y", "n", "yes", "no", "x", "true", "false"})

_NON_WORD = re.compile(r"[^0-9a-z]+")

#: :class:`Grounding` statuses.
ON_CITED_PAGE = "on_cited_page"
ON_OTHER_PAGE = "on_other_page"
NOT_PRINTED = "not_printed"
UNCHECKED = "unchecked"


def normalise(text: Any) -> str:
    """Case, punctuation and line breaks removed; words separated by one space."""
    return " ".join(_NON_WORD.sub(" ", str(text).casefold()).split())


#: A figure as a page prints it, whole: its digits with their separators, and a
#: "$" before them. Never a piece of a longer figure ("500" of "1,500.00") or of
#: a word or code glued to it ("000001" of "KA000001").
_FIGURE = re.compile(r"(?:\$\s?|(?<![0-9A-Za-z.,$]))\d(?:[\d,.]*\d)?(?![0-9A-Za-z]|[.,]\d)")
#: Only figure characters: an amount or a count, not a code that holds digits.
_AMOUNT = re.compile(r"[\s$()+\-.,%]*\d[\d,]*(?:\.\d+)?[\s$()%]*")
#: What makes a figure an amount: a "$", a thousands separator, or cents.
_MARKED = re.compile(r"\$|\d[.,]\d")
#: A run of digits led by a 0 ("07001", "0042"): a code, read as words - "0042"
#: is not the 42 of "Page 42".
_LEADING_ZERO = re.compile(r"\s*0\d+\s*")


def _figures(text: str) -> tuple[set[float], set[float]]:
    """The amounts a text prints - each figure written with a "$", a thousands
    separator or cents - and its bare whole numbers ("2019", "7781", "500")."""
    amounts: set[float] = set()
    integers: set[float] = set()
    for figure in _FIGURE.findall(text):
        number = _as_number(figure)
        if number is not None:
            (amounts if _MARKED.search(figure) else integers).add(number)
    return amounts, integers


class PageText:
    """A page's text normalised once, for every value looked up on it."""

    def __init__(self, text: str | None) -> None:
        self.padded = f" {normalise(text or '')} "
        self.words = set(self.padded.split())
        #: Every figure the page prints, as a number: "$7,579.67" and "7579.67"
        #: are one amount; "Year 2019" and "Code 7781" are whole numbers, not
        #: amounts (:func:`on_page`).
        self.amounts, self.integers = _figures(text or "")

    @property
    def has_text(self) -> bool:
        return bool(self.words)


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


def _number(raw: Any) -> float | None:
    """``raw`` as the figure it is compared as, or None for a value of words -
    a code of digits led by a 0 among them."""
    if isinstance(raw, str) and _LEADING_ZERO.fullmatch(raw):
        return None
    return _as_number(raw)


def _written_as_amount(raw: Any) -> bool:
    """Whether a figure is written as an amount: with a "$", a thousands
    separator or cents. A number given as a number carries no format."""
    return isinstance(raw, str) and bool(_MARKED.search(raw))


def _figure_on(raw: Any, number: float, amounts: set[float], integers: set[float]) -> bool:
    """Whether a figure is on a page that prints ``amounts`` and ``integers``.
    Written as an amount it is only an amount the page prints as one: "$2,019.00"
    is not the year 2019, "$7,781" not the agency code 7781. A bare figure is
    the same number printed any way."""
    if _written_as_amount(raw):
        return number in amounts
    return number in amounts or number in integers


def _tokens(raw: Any) -> list[str]:
    """What ``raw`` prints between spaces, each normalised: "PAP-2026-0042" is
    one token, "pap 2026 0042"."""
    return [token for token in map(normalise, str(raw).split()) if token]


def appears_on(value: Any, page: PageText) -> bool:
    """Whether ``value`` is on ``page`` as words: as a phrase, or - a phrase
    broken across lines or table columns - as every one of its printed tokens.

    A token is found whole: "PAP-2026-0042" is "pap 2026 0042", in that order,
    never a "PAP" here, a "2026" of the policy period there and a "0042" of the
    agency code elsewhere. Nor is a code of short pieces matched token by token:
    "PP 00 01" is the tokens "pp", "00" and "01", which any page holding them
    apart also holds. Only a value with a word of three letters or more is.

    ``value`` is normalised here. Given one normalised already, its tokens are
    its words: its codes are then matched word by word.
    """
    phrase = normalise(value)
    if not phrase:
        return False
    if f" {phrase} " in page.padded:
        return True
    words = phrase.split()
    if len(words) < 2 or not any(sum(ch.isalpha() for ch in word) >= 3 for word in words):
        return False
    tokens = _tokens(value)
    return len(tokens) > 1 and all(
        f" {token} " in page.padded if " " in token else token in page.words for token in tokens)


def on_page(raw: Any, page: PageText) -> bool:
    """Whether ``raw`` is printed on ``page``: a value of words as words
    (:func:`appears_on`), a figure as a figure - whole, the same number, and
    written as an amount only as an amount (:func:`_figure_on`)."""
    number = _number(raw)
    if number is None:
        return appears_on(raw, page)
    return _figure_on(raw, number, page.amounts, page.integers)


def appears(raw: Any, page_text: str | None) -> bool:
    """Whether ``raw`` is on the page, as a phrase or - across line breaks and
    columns - as every one of its tokens, or as the same figure."""
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
    #: Pages it cites that have no text, when no page with text prints it: a
    #: scan OCR could not read may print it there. Its status is then
    #: :data:`UNCHECKED` - the text cannot settle it - but it is not silent:
    #: serving flags it (``serving.printed_values``).
    unread: tuple[int, ...] = ()

    @property
    def printed(self) -> bool:
        return self.status in (ON_CITED_PAGE, ON_OTHER_PAGE)


def _placed(found: tuple[int, ...], cited: Iterable[Any] | None, blank: set[int]) -> Grounding:
    """The verdict on a value found on ``found`` among pages with text, ``blank``
    the pages searched that have none."""
    wanted = {int(page) for page in cited or () if str(page).lstrip("-").isdigit()}
    if found:
        return Grounding(ON_CITED_PAGE if not wanted or wanted & set(found) else ON_OTHER_PAGE, found)
    unread = tuple(sorted(wanted & blank))
    return Grounding(UNCHECKED, unread=unread) if unread else Grounding(NOT_PRINTED)


def ground(raw: Any, cited: Iterable[Any] | None, pages: Mapping[int, str | PageText | None]) -> Grounding:
    """Where ``raw`` is printed among ``pages`` (page number -> its text).

    Every page with text is searched - a blank page among those it cites does
    not stop the search, nor make a value printed elsewhere unchecked:

    * :data:`ON_CITED_PAGE` - on a page it cites (or, citing none, on any page);
    * :data:`ON_OTHER_PAGE` - only on pages it does not cite;
    * :data:`NOT_PRINTED` - on no page searched, and every page it cites has text;
    * :data:`UNCHECKED` - too short to look for, no page has text, or on no page
      with text while a page it cites has none (:attr:`Grounding.unread`: a scan
      the OCR could not read may still print it there).
    """
    value = normalise(raw) if raw is not None else ""
    if len(value) < MIN_CHECKABLE_CHARS:
        return Grounding(UNCHECKED)
    texts = {int(number): text if isinstance(text, PageText) else PageText(text)
             for number, text in pages.items()}
    blank = {number for number, text in texts.items() if not text.has_text}
    if len(blank) == len(texts):
        return Grounding(UNCHECKED)
    found = tuple(sorted(number for number, text in texts.items()
                         if number not in blank and on_page(raw, text)))
    return _placed(found, cited, blank)


# --------------------------------------------------------------------------
# What a value envelope prints
# --------------------------------------------------------------------------

_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_US_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def _date_forms(text: str) -> tuple[str, ...]:
    """The forms a page prints a date in, for a date written ISO or US style:
    "2026-04-01" is "04/01/2026", "4/1/2026", "2026-04-01" or "04/01/26"."""
    if match := _ISO_DATE.fullmatch(text.strip()):
        year, month, day = match.groups()
    elif match := _US_DATE.fullmatch(text.strip()):
        month, day, year = match.groups()
    else:
        return ()
    month_n, day_n = int(month), int(day)
    return tuple(dict.fromkeys((f"{month_n:02d}/{day_n:02d}/{year}", f"{month_n}/{day_n}/{year}",
                                f"{year}-{month_n:02d}-{day_n:02d}", f"{month_n:02d}/{day_n:02d}/{year[2:]}")))


def printed_forms(envelope: Mapping[str, Any]) -> tuple[Any, ...]:
    """What to look for on the pages for one value envelope: its ``raw``, as
    printed. Or - ``raw`` empty or a placeholder with no letter or digit ("",
    "-") while ``parsed`` holds a value - the forms ``parsed`` is printed in: a
    value served must be looked for whichever key holds it, and an invented
    ``parsed`` behind a blank ``raw`` is still invented. A date is looked for in
    each form a page prints one (:func:`_date_forms`); a number as the figure; a
    ``bool`` as the reading it is. Empty when the envelope holds nothing to look
    for - nothing, or a structure."""
    raw = envelope.get("raw")
    if isinstance(raw, (bool, int, float)) or isinstance(raw, str) and normalise(raw):
        return (raw,)
    parsed = envelope.get("parsed")
    if isinstance(parsed, str):
        return (_date_forms(parsed) or (parsed,)) if normalise(parsed) else ()
    return (parsed,) if isinstance(parsed, (bool, int, float)) else ()


# --------------------------------------------------------------------------
# OCR-tolerant: what a scan's text can settle (decision 2026-10-24)
# --------------------------------------------------------------------------

#: Letters OCR reads for digits, and digits it reads for letters ("NIP0115" as
#: "NIPO115", "$5" as "$S"). Folded to the digit in every word that holds a
#: digit, on both sides, before the words are compared. A word with no digit is
#: never folded: "SOLD" is not "5010".
_DIGIT_LOOKALIKES = str.maketrans({"o": "0", "q": "0", "i": "1", "l": "1", "s": "5", "b": "8",
                                   "z": "2", "g": "6"})

#: A "$" OCR read as an "S" before a figure ("S1,250.00").
_S_FOR_DOLLAR = re.compile(r"(?<![0-9A-Za-z])[Ss](?=\d)")

#: A word of this many letters or more matches a page's word one character
#: away (one dropped, added or changed: "lnsurance", "Insuranee"). Shorter
#: words, and every word holding a digit, must match once folded: one digit
#: changed is another number.
FUZZY_MIN_CHARS = 5


def _fold(word: str) -> str:
    return word.translate(_DIGIT_LOOKALIKES) if any(ch.isdigit() for ch in word) else word


def _ocr_figures(text: str) -> str:
    """``text`` with the misreads OCR makes in a figure undone where they can be
    told: a "$" read as an "S" before it, a letter read for a digit in a token
    that holds a digit ("1,25O.OO")."""
    text = _S_FOR_DOLLAR.sub("$", text)
    return " ".join(token.casefold().translate(_DIGIT_LOOKALIKES) if any(ch.isdigit() for ch in token) else token
                    for token in text.split())


def _deletions(word: str) -> set[str]:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def _readings(figure: str) -> set[float]:
    """The amounts a figure may be once OCR has misread its separators: as
    written, every separator a thousands mark when the last one has three
    digits after it ("1.000.000" is 1000000), or the last one the decimal point
    when it has one or two ("1.000,50" is 1000.5). Never a figure with its
    decimal point moved: "$2,500" is not "$25.00", nor "$2.50"."""
    out: set[float] = set()
    if (number := _as_number(figure)) is not None:
        out.add(number)
    digits = re.sub(r"\D", "", figure)
    if not digits:
        return out
    marks = [at for at, ch in enumerate(figure) if ch in ".,"]
    last = len(figure) - marks[-1] - 1 if marks else 0
    if not marks or last == 3:
        out.add(float(digits))
    if marks and last in (1, 2):
        whole, cents = re.sub(r"\D", "", figure[:marks[-1]]), re.sub(r"\D", "", figure[marks[-1] + 1:])
        if whole and cents:
            out.add(round(float(f"{whole}.{cents}"), 2))
    return out


class TolerantPage:
    """A page's text, prepared once for :func:`ground_tolerant`: normalised,
    its digit-holding words folded, and every figure as the numbers it may be."""

    def __init__(self, text: str | None) -> None:
        self.exact = PageText(text)
        #: The page's words in order, folded: a code is looked for as a run of them.
        self.sequence = [_fold(word) for word in self.exact.padded.split()]
        self.words = set(self.sequence)
        self.padded = f" {' '.join(self.sequence)} "
        #: Every amount the page may print, each figure read every way its
        #: separators allow (:func:`_readings`), and - one with a separator or
        #: cents, which shows no "$" - without a leading 5: a "$" read as a "5"
        #: ("$37.47" printed, "537.47" read). A bare whole number keeps its 5:
        #: "Form 5250" is not "$250".
        self.amounts: set[float] = set(self.exact.amounts)
        self.integers: set[float] = set(self.exact.integers)
        for figure in _FIGURE.findall(_ocr_figures(text or "")):
            body = figure.lstrip("$").strip()
            if not _MARKED.search(figure):
                self.integers.add(float(re.sub(r"\D", "", body)))
                continue
            self.amounts |= _readings(body)
            if not figure.startswith("$") and body.startswith("5") and body[1:2].isdigit():
                self.amounts |= _readings(body[1:])
        self._near: dict[str, set[str]] | None = None
        self._at: dict[str, list[int]] | None = None

    def _near_index(self) -> dict[str, set[str]]:
        if self._near is None:
            self._near = {}
            for page_word in self.words:
                if len(page_word) >= FUZZY_MIN_CHARS - 1 and not any(ch.isdigit() for ch in page_word):
                    self._near.setdefault(page_word, set()).add(page_word)
                    for deleted in _deletions(page_word):
                        self._near.setdefault(deleted, set()).add(page_word)
        return self._near

    def near(self, word: str) -> bool:
        """Whether a page word is at most one character from ``word``."""
        index = self._near_index()
        return word in index or any(deleted in index for deleted in _deletions(word))

    def like(self, word: str) -> set[str]:
        """The page's words ``word`` may be as OCR read it: itself, and - a word
        of :data:`FUZZY_MIN_CHARS` letters or more, no digit - each one
        character away."""
        found = {word} & self.words
        if len(word) >= FUZZY_MIN_CHARS and not any(ch.isdigit() for ch in word):
            index = self._near_index()
            for key in (word, *_deletions(word)):
                found |= index.get(key, set())
        return found

    def in_order(self, words: list[str]) -> bool:
        """Whether ``words`` (folded) are on the page one after another, each as
        :meth:`like` finds it: a code is one run of words, never pieces."""
        options = [self.like(word) for word in words]
        if not all(options):
            return False
        if len(words) == 1:
            return True
        if self._at is None:
            self._at = {}
            for position, word in enumerate(self.sequence):
                self._at.setdefault(word, []).append(position)
        last = len(self.sequence) - len(words)
        return any(all(self.sequence[start + i] in options[i] for i in range(1, len(words)))
                   for first in options[0] for start in self._at.get(first, ()) if start <= last)

    @property
    def has_text(self) -> bool:
        return bool(self.exact.words)


def tolerant_pages(pages: Mapping[int, str | TolerantPage | None]) -> dict[int, TolerantPage]:
    """Each page prepared for :func:`ground_tolerant`, once per document: a
    policy has hundreds of values to look up on the same pages."""
    return {int(number): text if isinstance(text, TolerantPage) else TolerantPage(text)
            for number, text in pages.items()}


def _tolerantly_on(raw: Any, page: TolerantPage) -> bool:
    number = _number(raw)
    if number is not None:
        return _figure_on(raw, number, page.amounts, page.integers)
    if on_page(raw, page.exact):
        return True
    tokens = [[_fold(word) for word in token.split()] for token in _tokens(raw)]
    words = [word for token in tokens for word in token]
    if not words:
        return False
    if f" {' '.join(words)} " in page.padded:
        return True
    if len(tokens) == 1:
        return page.in_order(tokens[0])
    # A phrase broken across lines or columns: each token found apart - but
    # only for a value with a word of three letters or more (appears_on).
    if not any(sum(ch.isalpha() for ch in word) >= 3 for word in words):
        return False
    return all(page.in_order(token) for token in tokens)


def ground_tolerant(raw: Any, cited: Iterable[Any] | None,
                    pages: Mapping[int, str | TolerantPage | None]) -> Grounding:
    """:func:`ground`, tolerant of what OCR does to a scan.

    A scan's text misreads, so on its own word a value is never "printed
    nowhere" (``data_pipeline.ingestion.label_rules``). Where a value must be
    printed to be served (``serving.printed_values``, decision 2026-10-24) the
    search forgives the misreads OCR makes, and only them: digit lookalikes in a
    word holding a digit (:data:`_DIGIT_LOOKALIKES`), one character in a word of
    :data:`FUZZY_MIN_CHARS` letters or more, a figure's separators, and a "$"
    read as a "5" or an "S". A code is still one run of words, a figure still
    whole. Every page is searched; the statuses are :func:`ground`'s.

    Short values are looked for too: one of :data:`MIN_TOKEN_CHARS` characters
    ("NJ", "50") as a whole token, an amount ("$5") as an amount. Unchecked: a
    ``bool`` or a yes/no (:data:`_READINGS`) - a reading, not a printed value -
    and a single character, on every page.
    """
    if isinstance(raw, bool):
        return Grounding(UNCHECKED)
    value = normalise(raw) if raw is not None else ""
    if not value or value in _READINGS:
        return Grounding(UNCHECKED)
    if len(value) < MIN_TOKEN_CHARS and not (_number(raw) is not None and _written_as_amount(raw)):
        return Grounding(UNCHECKED)
    texts = tolerant_pages(pages)
    blank = {number for number, text in texts.items() if not text.has_text}
    if len(blank) == len(texts):
        return Grounding(UNCHECKED)
    found = tuple(sorted(number for number, text in texts.items()
                         if number not in blank and _tolerantly_on(raw, text)))
    return _placed(found, cited, blank)
