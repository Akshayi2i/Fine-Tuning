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

from contracts.page_plan import (  # noqa: F401 - moved (review A1)
    PAGE_SIGNALS,
    PageScore,
    RoutingPlan,
    plan_pages,
    score_page,
)

log = logging.getLogger(__name__)

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

    **Not on the serving path.** Every page is read in section windows that
    ``serving.policy_merge`` joins, so the model never returns per-page objects
    for this to merge. The declarations-precedence rule this
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
