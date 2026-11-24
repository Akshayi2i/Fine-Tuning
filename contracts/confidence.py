"""A document's overall confidence from its fields' (review A1: moved from ``serving.printed_values``, which
re-exports it). Serving and calibration score a result's overall confidence the same way."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _overall_confidence(extraction: Any, flags: Iterable[str]) -> float:
    """The mean confidence of the values the model read, as ``extract`` averages
    the calibrated fields: one score per field (a list of values is one), none
    for a field it had no confidence for (``<path>:no_confidence``), none for
    what the pipeline wrote (system fields, the every-key fill)."""
    from common.canonical import CONFIDENCE_SOURCE, is_field_value

    unscored = {flag.rsplit(":", 1)[0] for flag in flags if flag.endswith(":no_confidence")}
    scores: list[float] = []

    def walk(node: Any, path: str) -> None:
        if is_field_value(node):
            confidence = node.get("confidence")
            if (isinstance(confidence, dict) and confidence.get("source") == CONFIDENCE_SOURCE
                    and path not in unscored):
                scores.append(float(confidence.get("score") or 0.0))
        elif isinstance(node, dict):
            for key, value in node.items():
                if not str(key).startswith("fideon:"):
                    walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            if node and all(is_field_value(item) for item in node):
                walk(node[0], path)                     # one field, every value its score
            else:
                for index, item in enumerate(node):
                    walk(item, f"{path}[{index}]")

    walk(extraction, "")
    return round(sum(scores) / len(scores), 4) if scores else 0.0
