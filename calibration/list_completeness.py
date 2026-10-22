"""Row completeness — the signal token logprobs structurally cannot provide (arch §5).

The failure this exists for: the model extracts 6 of 8 claims. The 2 missing rows
**generate no tokens**, so there is no low probability anywhere to flag. Per-field
confidence on the 6 extracted rows can all be 0.95 while the extraction is
silently incomplete. It is a recall failure, and token confidence is blind to it
by construction — no amount of calibration fixes a signal that has no input.

So list fields carry **two** confidence signals, not one:

1. per-value confidence, from logprobs, for the fields inside each extracted row;
2. row completeness, from cross-checks against the document itself.

Two independent cross-checks, either of which flags the list:

* the **document's own stated count** — a "Total Claims: 3" field;
* the **structure-derived count** — table rows MinerU detected on the relevant
  pages, recorded in ``ocr_meta.json`` at OCR time.

When either disagrees with the extracted row count, the **whole list** is flagged
regardless of how confident the individual values are. This matters most on Loss
Runs, where a missed claim row is both easy to produce and expensive to miss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Suffix on every row-completeness review flag, as ``"{field}:row_count_mismatch"``.
#:
#: Exported because `data_pipeline.labeling.active_learning` has to recognise
#: these flags to honour the "row completeness outranks every confidence score"
#: rule. It previously matched the prefixes "list:", "rows:" and "completeness:",
#: none of which this module has ever emitted, so the override was dead code and
#: an incomplete Loss Run with high per-value confidence was routed to a spot
#: check instead of full review. One constant, imported by both sides, is what
#: keeps a producer and its consumer from drifting apart in silence.
ROW_MISMATCH_SUFFIX = ":row_count_mismatch"


def is_row_completeness_flag(flag: str) -> bool:
    """Whether a review flag came from the row-completeness cross-check."""
    return flag.endswith(ROW_MISMATCH_SUFFIX)

#: A structure-derived count can legitimately differ by a row or two — a
#: continuation header, a totals row MinerU counted as data. Beyond this, the
#: disagreement is real.
STRUCTURE_TOLERANCE = 1


@dataclass
class CompletenessSignal:
    """Whether a list looks complete, and what the evidence was."""

    field_name: str
    extracted_rows: int
    stated_count: int | None = None
    detected_rows: int | None = None
    confidence: float = 1.0
    flagged: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def missing_rows(self) -> int:
        """Best estimate of how many rows are missing.

        The document's own stated count is preferred over the structural
        estimate: a printed "Total Claims: 8" is an assertion, whereas a detected
        row count is an inference from layout.
        """
        if self.stated_count is not None:
            return max(0, self.stated_count - self.extracted_rows)
        if self.detected_rows is not None:
            return max(0, self.detected_rows - self.extracted_rows)
        return 0

    def as_output(self) -> dict[str, Any]:
        return {
            "extracted_rows": self.extracted_rows,
            "stated_count": self.stated_count,
            "detected_rows": self.detected_rows,
            "row_completeness_confidence": round(self.confidence, 4),
            "flagged": self.flagged,
            "reasons": self.reasons,
        }


def _as_count(value: object) -> int | None:
    """A document-stated row count, however the model spelled it.

    ``8``, ``8.0`` and ``"8"`` are the same assertion. Accepting only ``int``
    meant a perfectly good stated count arriving as a JSON string turned the
    cross-check off without a word.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.isdigit():
            return int(text)
    return None


def check_completeness(
    field_name: str,
    extracted_rows: int,
    *,
    stated_count: int | None = None,
    detected_rows: int | None = None,
    structure_tolerance: int = STRUCTURE_TOLERANCE,
) -> CompletenessSignal:
    """Cross-check an extracted row count against the document's own evidence."""
    signal = CompletenessSignal(
        field_name=field_name,
        extracted_rows=extracted_rows,
        stated_count=stated_count,
        detected_rows=detected_rows,
    )

    # 1. The document's stated count — an assertion, so an exact comparison.
    if stated_count is not None and stated_count != extracted_rows:
        signal.flagged = True
        missing = stated_count - extracted_rows
        signal.reasons.append(
            f"the document states {stated_count} row(s) but {extracted_rows} were extracted "
            f"({abs(missing)} {'missing' if missing > 0 else 'extra'}). Missing rows generate no "
            "tokens, so per-value confidence cannot see this."
        )

    # 2. The structure-derived count — an inference, so tolerant.
    if detected_rows is not None and abs(detected_rows - extracted_rows) > structure_tolerance:
        signal.flagged = True
        signal.reasons.append(
            f"OCR detected {detected_rows} table row(s) but {extracted_rows} were extracted "
            f"(tolerance {structure_tolerance})."
        )

    if signal.flagged:
        # Scale confidence by how much is missing rather than zeroing it: losing
        # one row of forty is a different problem from losing thirty.
        # `or` treated a stated count of 0 as absent, so a document stating zero
        # claims against four hallucinated rows fell through to the extracted
        # count itself and reported confidence 1.0 — the pure-hallucination case
        # scoring highest, in a value that ships in the output contract.
        # The WORST disagreement, not the first available count. Preferring
        # stated_count meant a list flagged only by the structural check — the
        # document stating 6, the model returning 6, OCR having seen 9 — took
        # stated_count as its reference, found it equal to the extracted count,
        # and shipped row_completeness_confidence 1.0 on a list the very same
        # function had just flagged as incomplete.
        candidates = [c for c in (signal.stated_count, signal.detected_rows) if c is not None]
        reference = (
            max(candidates, key=lambda c: abs(c - extracted_rows))
            if candidates else max(extracted_rows, 1)
        )
        if reference == extracted_rows:
            signal.confidence = 1.0
        elif reference == 0:
            # Rows extracted where the document states none: entirely unsupported.
            signal.confidence = 0.0
        else:
            # Symmetric: over-extraction is as wrong as under-extraction, and the
            # min(1.0, ...) clamp used to report a surplus as perfect.
            signal.confidence = round(
                max(0.0, 1.0 - abs(extracted_rows - reference) / max(reference, 1)), 4
            )
        log.warning("list %r flagged incomplete: %s", field_name, "; ".join(signal.reasons))
    elif stated_count is None and detected_rows is None:
        # No cross-check available. Not the same as verified complete, and the
        # confidence says so.
        signal.confidence = 0.5
        signal.reasons.append(
            "no stated count and no detected row count — completeness could not be verified"
        )
    return signal


#: Where a document type states, in its own words, how many rows it holds — and
#: which field those rows live in. Only Loss Runs print a claim count today.
#: A type absent from this table has no self-reported count to cross-check
#: against, which is a real difference between document types rather than a gap:
#: a policy prints no "total coverages" line.
STATED_COUNTS: dict[str, tuple[str, str]] = {
    "lossrun": ("total_claims_reported", "claims"),
}


def stated_count_field(doc_type: str) -> str | None:
    """The field carrying the document's own row count, if the schema has one."""
    entry = STATED_COUNTS.get(doc_type)
    return entry[0] if entry else None


def stated_list_field(doc_type: str) -> str | None:
    """The list that a stated count is a count **of**.

    A stated count names one list. Comparing it against every list field flagged
    the others for disagreeing with a number that was never about them.
    """
    entry = STATED_COUNTS.get(doc_type)
    return entry[1] if entry else None


def check_document(
    extraction: dict[str, Any],
    doc_type: str,
    *,
    ocr_meta: dict[str, Any] | None = None,
    pages_used: list[int] | None = None,
) -> dict[str, CompletenessSignal]:
    """Check every list field in an extraction.

    Args:
        ocr_meta: the document's ``ocr_meta.json``, whose ``table_row_counts``
            supply the structure-derived count.
        pages_used: pages that actually fed this extraction, so a document is
            compared against the pages it read rather than the whole file.
    """
    detected_total = None
    if ocr_meta and (row_counts := ocr_meta.get("table_row_counts")):
        pages = [str(p) for p in pages_used] if pages_used else list(row_counts)
        detected_total = sum(int(row_counts.get(page, 0)) for page in pages)

    stated_field = stated_count_field(doc_type)
    stated = extraction.get(stated_field) if stated_field else None

    list_fields = {
        name: value for name, value in extraction.items()
        if isinstance(value, list) and not (value and not isinstance(value[0], dict))
    }

    # MinerU's row count is per page, not per field, so it can only be compared
    # against a list when the document has exactly one. Applying one
    # document-wide total to every list guaranteed a flag on any document with
    # two — a perfect policy_doc (5 coverage rows + 3 endorsements against a
    # detected 8) was flagged twice on every production request.
    comparable = detected_total if len(list_fields) == 1 else None
    if detected_total is not None and comparable is None:
        log.info(
            "%d list fields in this %s, so the page row count (%d) cannot be attributed to any "
            "one of them — the detected-row cross-check is skipped and the stated-count check "
            "still applies.",
            len(list_fields), doc_type, detected_total,
        )

    signals: dict[str, CompletenessSignal] = {}
    for name, value in list_fields.items():
        signals[name] = check_completeness(
            name, len(value),
            # The stated count names one list (IMPL-09 stated_count_field), so it
            # applies only to that one.
            # `_as_count`, not isinstance(int): the model emits JSON, and a
            # stated count arriving as the string "8" silently disabled the
            # whole stated-count cross-check for that document.
            stated_count=(
                _as_count(stated) if name == stated_list_field(doc_type) else None
            ),
            detected_rows=comparable,
        )
    return signals


def merge_review_flags(
    signals: dict[str, CompletenessSignal], existing_flags: list[str]
) -> list[str]:
    """Add completeness flags to a document's review routing.

    A flagged list goes to review **regardless of per-value confidence** — that
    is the whole point: the values present may be perfect while the list is
    incomplete.
    """
    flags = list(existing_flags)
    flags.extend(f"{name}{ROW_MISMATCH_SUFFIX}" for name, s in sorted(signals.items()) if s.flagged)
    return sorted(set(flags))
