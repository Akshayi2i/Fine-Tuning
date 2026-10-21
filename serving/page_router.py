"""Where a document's declarations are, and the old merge helpers (arch §7b).

**Every page is read, chunk by chunk, never selected pages** (decision
2026-10-19). Routing used to send a long policy's first page and the pages
scoring as a schema section, and a page that scored nothing was never read:
whatever it printed was silently missing. Now every page of every document
that reaches the model is read, in windows (``policy_windows.plan_windows``).

What this module still does is find the declarations page by its section
signals: the declarations group's first window is pages 1 to 5 plus that page,
read together, so a policy opening with a fax cover sheet still has its
declarations read as one block. Detecting a page labels it; it drops nothing.

Which pages fed the extraction is still **recorded** (``pages_used``), so a
low-confidence field can be traced to the page it came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Section markers, weighted by how strongly they say a page is the
#: declarations or a schedule. Only the declarations markers decide anything
#: now (:func:`plan_pages`); the scores are kept on the plan for tracing.
PAGE_SIGNALS: dict[str, float] = {
    "declaration": 10.0,
    "declarations page": 10.0,
    "policy declarations": 10.0,
    "common policy declarations": 10.0,
    "schedule": 5.0,
    "coverage schedule": 6.0,
    "coverage summary": 6.0,
    "deductible": 3.0,
    "named insured": 4.0,
    "policy number": 4.0,
    "policy period": 4.0,
    "premium": 3.0,
    "limits of insurance": 3.0,
    "endorsement": 2.0,
    "producer": 2.0,
}


@dataclass
class PageScore:
    page: int
    score: float
    matched: list[str] = field(default_factory=list)
    is_declarations: bool = False


@dataclass
class RoutingPlan:
    """The pages a document is read on - all of them - and its declarations page."""

    pages: list[int]
    scores: list[PageScore] = field(default_factory=list)
    declarations_page: int | None = None
    reason: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)


def score_page(text: str, page_number: int) -> PageScore:
    """Score one page by the section markers it contains."""
    lowered = (text or "").lower()
    score = 0.0
    matched: list[str] = []
    for signal, weight in PAGE_SIGNALS.items():
        if signal in lowered:
            score += weight
            matched.append(signal)

    is_declarations = any("declaration" in m for m in matched)
    return PageScore(page=page_number, score=score, matched=matched, is_declarations=is_declarations)


def plan_pages(page_texts: dict[int, str]) -> RoutingPlan:
    """Every page, in order, and the first page that reads as the declarations.

    Args:
        page_texts: page number -> OCR text, 1-based.
    """
    scores = [score_page(text, page) for page, text in sorted(page_texts.items())]
    declarations = next((s.page for s in scores if s.is_declarations), None)
    return RoutingPlan(
        pages=sorted(page_texts),
        scores=scores,
        declarations_page=declarations,
        reason=f"every page of {len(page_texts)} is read (decision 2026-10-19)",
    )


# --------------------------------------------------------------------------
# Merging scoped extractions
# --------------------------------------------------------------------------

#: Fields whose canonical value lives on the declarations page when pages
#: disagree. A schedule page can repeat a policy number in a different format;
#: the declarations page is the authority.
POLICY_LEVEL_FIELDS = frozenset({
    "insured_name", "insured_address", "policy_number", "effective_date",
    "expiration_date", "carrier", "producer", "total_premium", "line_of_business",
})


@dataclass
class MergeResult:
    """A merged document, and where each conflicting field came from."""

    extraction: dict[str, Any] = field(default_factory=dict)
    field_sources: dict[str, int] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)


def merge_page_extractions(
    per_page: dict[int, dict[str, Any]],
    *,
    declarations_page: int | None = None,
) -> MergeResult:
    """Combine per-page extractions into one document.

    **No longer on the serving path.** A routed request now sends its selected
    pages interleaved in a single call, so the model returns one document-level
    object and there is nothing to merge. The declarations-precedence rule this
    function implements is stated to the model directly instead (see
    ``prompts/doc_types/policy.jinja``), which keeps it in one place rather than
    two that can disagree.

    Kept because it is the only implementation of that conflict rule in code, and
    because any future path that does produce per-page results — a batched
    fallback, or a document too long for one context — needs exactly this.

    Conflict rules, in order:

    1. **The declarations page wins** for policy-level fields (arch §7).
    2. Otherwise the first non-null value in page order wins — earlier pages
       carry the document's identity.
    3. **List fields concatenate** rather than overwrite: a Loss Run's claims
       table spanning three pages must produce all its rows, and overwriting
       would silently drop two thirds of them.
    """
    result = MergeResult()

    for page in sorted(per_page):
        for key, value in (per_page[page] or {}).items():
            if value is None:
                continue

            if isinstance(value, list):
                existing = result.extraction.setdefault(key, [])
                if isinstance(existing, list):
                    existing.extend(value)
                    result.field_sources.setdefault(key, page)
                else:
                    # An earlier page gave a scalar and this one gives rows. The
                    # rows were previously dropped with no conflict recorded, so
                    # no review flag fired — the only disagreement path in this
                    # function that lost data silently.
                    result.conflicts.append(
                        f"{key}: page {result.field_sources.get(key)} gave a scalar "
                        f"{existing!r} but page {page} gave {len(value)} row(s); keeping the rows"
                    )
                    result.extraction[key] = list(value)
                    result.field_sources[key] = page
                continue

            if key not in result.extraction:
                result.extraction[key] = value
                result.field_sources[key] = page
                continue

            if result.extraction[key] == value:
                continue

            # A genuine disagreement between pages.
            result.conflicts.append(
                f"{key}: page {result.field_sources[key]} says {result.extraction[key]!r}, "
                f"page {page} says {value!r}"
            )
            if key in POLICY_LEVEL_FIELDS and page == declarations_page:
                result.extraction[key] = value
                result.field_sources[key] = page

    if result.conflicts:
        log.info("merged %d pages with %d field conflict(s)", len(per_page), len(result.conflicts))
    return result
