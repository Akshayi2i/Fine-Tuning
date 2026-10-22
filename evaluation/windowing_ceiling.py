"""The ceiling reading a policy in windows puts on accuracy (the oracle merge).

Each document's GOLD label is sliced into its windows exactly as the dataset
build slices it (``policy_windows.window_target``), the slices are merged by the
serving merge (``serving.policy_merge``), and the result is compared with the
gold label as scoring uses it (``common.canonical.schema_label``). A perfect
model reproduces exactly those slices, so whatever this loses no model can win
back. Reported by scripts/diagnose_windowing.py and in the golden eval report.

Lost values get a reason: ``no_page_ref`` (no page recorded, group read over
several windows), ``unread_page`` (printed only on pages no window of its group
reads), ``row_unmatched`` (its row came back split or keyless) or ``merge``.
Values outside the schema are counted apart: no model can write them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

REASONS = ("no_page_ref", "unread_page", "row_unmatched", "merge")


@dataclass
class DocumentResult:
    source_id: str
    lob: str
    synthetic: bool
    pages: int
    windows: int = 0
    gold_values: int = 0
    outside_schema: int = 0
    recovered: int = 0
    lost: Counter = field(default_factory=Counter)
    gold_rows: int = 0
    merged_rows: int = 0
    conflicts: int = 0
    lost_values: list[tuple[str, str, str]] = field(default_factory=list)
    #: Common-model lines: links (a coverage to its vehicle) in the gold label,
    #: those the merged answer still makes, and references no window or merge
    #: could resolve.
    gold_references: int = 0
    recovered_references: int = 0
    dangling: int = 0

    @property
    def recall(self) -> float | None:
        return self.recovered / self.gold_values if self.gold_values else None


def oracle(source_id: str, label: dict, lob: Any, page_count: int,
           ocr_pages: list[str] | None, synthetic: bool = False) -> DocumentResult:
    """Slice ``label`` into its windows, merge the slices, compare with ``label``."""
    from common.canonical import schema_label, values_view, without_system_fields
    from common.label_mapping import map_label
    from data_pipeline.dataset_builder.policy_windows import (
        TargetReport,
        multi_window_sections,
        pages_to_read,
        plan_windows,
        unread_values,
        window_target,
        with_inferred_pages,
    )
    from serving.policy_merge import PolicyWindow, merge_policy_windows

    line = ", ".join(lob) if isinstance(lob, list) else str(lob)
    result = DocumentResult(source_id, line, synthetic, page_count)
    every_page, declarations_page = pages_to_read(ocr_pages, page_count)
    plans = plan_windows(lob, every_page, declarations_page, ocr_pages)
    report = TargetReport()
    # As the dataset build does: pages placed from the OCR text, when there is one.
    placed = with_inferred_pages(label, ocr_pages, report,
                                 sections=multi_window_sections(lob, plans))
    windows = [PolicyWindow(group=p.group, pages=list(p.pages),
                            extraction=window_target(placed, lob, p, report)) for p in plans]
    merged = merge_policy_windows(windows, lob=lob)
    result.windows, result.conflicts = len(plans), len(merged.conflicts)

    mapped = map_label(label, lob)
    scored = without_system_fields(schema_label(label, "policy", None, lob))
    result.outside_schema = _stated(without_system_fields(mapped)) - _stated(scored)
    answer = without_system_fields(merged.extraction)
    from common.schemas import is_common_model

    if is_common_model("policy", None, lob):
        # Ids are numbered by whoever writes them, so values are compared with
        # ids left out, and links - what a reference names - counted apart.
        from common.schema_sections import references
        from common.structural_ids import comparable_view, reference_pairs

        expected_links, found_links = Counter(reference_pairs(scored, lob)), Counter(reference_pairs(answer, lob))
        result.gold_references = sum(expected_links.values())
        result.recovered_references = sum((expected_links & found_links).values())
        result.dangling = len(set(report.dangling)) + len(merged.dangling_references)
        links = set(references(lob))
        scored, answer = (_without(comparable_view(doc, lob), links) for doc in (scored, answer))
    gold = values_view(scored)
    got = values_view(answer)
    reasons = {
        "no_page_ref": {_bare(p.split(":", 1)[1]) for p in report.unplaced},
        "unread_page": {_bare(p) for p in unread_values(map_label(placed, lob), lob, plans)},
    }

    def lose(path: str, value: Any, unmatched_row: bool) -> None:
        bare = _bare(path)
        if bare in reasons["no_page_ref"] or any(bare.startswith(r + ".") for r in reasons["no_page_ref"]):
            reason = "no_page_ref"
        elif bare in reasons["unread_page"]:
            reason = "unread_page"
        elif unmatched_row:
            reason = "row_unmatched"
        else:
            reason = "merge"
        result.lost[reason] += 1
        result.lost_values.append((path, str(value)[:80], reason))

    def compare(expected: Any, actual: Any, path: str, unmatched_row: bool = False) -> None:
        from evaluation.metrics.field_accuracy import _infer_key_fields, _is_empty, _row_key, values_agree

        if isinstance(expected, dict):
            sub = actual if isinstance(actual, dict) else {}
            for key, value in expected.items():
                compare(value, sub.get(key), f"{path}.{key}" if path else key, unmatched_row)
        elif isinstance(expected, list) and expected and all(isinstance(r, dict) for r in expected):
            rows = actual if isinstance(actual, list) else []
            result.gold_rows += len(expected)
            result.merged_rows += len(rows)
            keys = _infer_key_fields(expected)
            pool: dict[tuple, list[Any]] = defaultdict(list)
            for row in rows:
                pool[_row_key(row, keys)].append(row)
            for index, row in enumerate(expected):
                candidates = pool.get(_row_key(row, keys))
                found = candidates.pop(0) if candidates else None
                # Decided on the row found, not on what is left after taking it:
                # a row matched once was counted as unmatched.
                compare(row, found, f"{path}[{index}]", unmatched_row or found is None)
        elif not _is_empty(expected):
            result.gold_values += 1
            if values_agree(expected, actual, path):
                result.recovered += 1
            else:
                lose(path, expected, unmatched_row)

    compare(gold, got, "")
    return result


def _stated(node: Any) -> int:
    """Stated values in a label: envelopes with a raw or parsed value."""
    from common.canonical import is_field_value

    if is_field_value(node):
        return int(node.get("raw") is not None or node.get("parsed") is not None)
    if isinstance(node, dict):
        return sum(_stated(v) for v in node.values())
    if isinstance(node, list):
        return sum(_stated(v) for v in node)
    return 0


def _without(node: Any, names: set[str]) -> Any:
    if isinstance(node, dict):
        return {k: _without(v, names) for k, v in node.items() if k not in names}
    if isinstance(node, list):
        return [_without(v, names) for v in node]
    return node


def _bare(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def ceiling(results: list[DocumentResult]) -> dict[str, Any]:
    """The pooled figures a report carries."""
    gold = sum(d.gold_values for d in results)
    recovered = sum(d.recovered for d in results)
    lost = sum((d.lost for d in results), Counter())
    out = {
        "documents": len(results),
        "value_recall": round(recovered / gold, 4) if gold else None,
        "lost": {reason: lost[reason] for reason in REASONS if lost[reason]},
        "outside_schema_values": sum(d.outside_schema for d in results),
    }
    links = sum(d.gold_references for d in results)
    if links:
        out["reference_recall"] = round(sum(d.recovered_references for d in results) / links, 4)
        out["dangling_references"] = sum(d.dangling for d in results)
    return out
