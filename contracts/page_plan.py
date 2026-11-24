"""Which pages of a policy hold what, from their text (review A1: moved from ``serving.page_router``, which
re-exports it). The window planner reads it to find the declarations page; serving reads every page in windows."""

from __future__ import annotations

from dataclasses import dataclass, field

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
