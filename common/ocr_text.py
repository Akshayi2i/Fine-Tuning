"""OCR page text without pictures: a picture prints no text, so its markup is never page text.

MinerU drops a picture block to its captions already (``mineru_api.PICTURE_TYPES``,
``run_mineru.pages_from_content_list``), but a picture that sits inside a TABLE
comes through in the table's HTML as ``<img src="data:image/jpeg;base64,...">``.
Cycle 1 lost 8 Hartford policies to one blank "Hazardous Materials listing"
form: 14 small pictures in its table cells made 156,140 characters of page text
where the page prints 329, so the rows holding that page went over their token
budget and each document was set aside whole (pod, 2026-11-07).

Every writer of page text and every reader of stored page text calls
:func:`clean_page_text`, so text already stored with pictures is read clean
without a new OCR run, and serving (which reads MinerU's answer directly) sees
the same text training did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: An HTML ``<img ...>`` tag (MinerU's table cells), or a markdown image ``![alt](target)``.
#: Neither carries printed text: the picture itself is in the page image the model sees.
PICTURE_MARKUP = re.compile(r"<img\b[^>]*>|!\[[^\]\n]*\]\([^)\n]*\)", re.IGNORECASE)


@dataclass(frozen=True)
class CleanedText:
    """A page's text with its pictures taken out, and what was taken out."""

    text: str
    pictures: int
    characters_removed: int


def strip_pictures(text: str | None) -> CleanedText:
    """``text`` without picture markup, with how many pictures and characters went."""
    raw = text or ""
    pictures = 0

    def _drop(match: re.Match[str]) -> str:
        nonlocal pictures
        pictures += 1
        return ""

    cleaned = PICTURE_MARKUP.sub(_drop, raw)
    return CleanedText(cleaned, pictures, len(raw) - len(cleaned))


def clean_page_text(text: str | None) -> str:
    """``text`` without picture markup."""
    return strip_pictures(text).text
