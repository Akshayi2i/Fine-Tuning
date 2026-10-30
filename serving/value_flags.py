"""Review flags on single values that send those values to review.

Most review flags only say the document needs a person (``needs_review`` is
any flag at all). These say one VALUE does, though it is served: its envelope
is ``flagged`` as a value below its review threshold is, so it is never an
auto-accepted value (``evaluation.metrics.auto_accept``) whatever its
confidence. The confidence reads how sure the tokens were, and each of these is
a doubt the tokens cannot show:

* ``<path>:printed_off_cited_page`` - printed, but only on pages it does not
  cite: the right figure, quite possibly from the wrong row or the wrong role
  (``serving.printed_values``);
* ``<path>:role_mismatch`` - a declarations line opens with this field's label,
  and the value printed after it is another (``serving.conventions.role_mismatches``);
* ``<path>:near_tie`` - a token of the value was all but another token
  (``inference_core.span_map.FieldSpan.top2_margin`` below
  ``serving.near_tie_margin``): greedy decoding with ``repeatable: false`` can
  write the other one on another run;
* ``<path>:date_order``, ``term_too_long``, ``implausible_date``, ``limit_order``,
  ``premium_below_items``, ``bad_postal_code``, ``bad_state``, ``bad_policy_no`` -
  values that cannot all be true together, or a value of a shape the field never
  has (``serving.consistency``): printed, so the text cannot say it is wrong;
* ``<path>:parsed_mismatch`` - its ``parsed`` is not what its ``raw`` reads as
  (flag mode keeps it; strict mode takes it out, and then there is nothing to
  send).

``<row>:row_not_colocated`` (a row whose values are printed on no common page)
is a document flag only: a row merged across windows is often printed so.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from serving.consistency import CONSISTENCY_FLAGS

OFF_CITED_PAGE_FLAG = "printed_off_cited_page"
ROLE_MISMATCH_FLAG = "role_mismatch"
NEAR_TIE_FLAG = "near_tie"
PARSED_MISMATCH_FLAG = "parsed_mismatch"
ROW_NOT_COLOCATED_FLAG = "row_not_colocated"

#: The flags that send the value they name to review: a value the page text cannot
#: settle, and the values that cannot all be true together (``serving.consistency``).
REVIEW_ROUTED = frozenset({OFF_CITED_PAGE_FLAG, ROLE_MISMATCH_FLAG, NEAR_TIE_FLAG, PARSED_MISMATCH_FLAG}
                          | CONSISTENCY_FLAGS)

#: ``serving.near_tie_margin`` when the config names none: nats.
DEFAULT_NEAR_TIE_MARGIN = 0.1

_LAST_INDEX = re.compile(r"\[\d+\]$")


def near_tie_margin() -> float:
    """``serving.near_tie_margin`` (configs/inference/vllm_serving.yaml)."""
    from common.config import serving_config

    value = (serving_config().get("serving") or {}).get("near_tie_margin", DEFAULT_NEAR_TIE_MARGIN)
    return float(value)


def routed_paths(flags: Iterable[str]) -> set[str]:
    """The value paths the flags send to review. A value of a list of values
    (``vehicles[0].symbols[1]``) sends its list too: a list of values is one
    field to calibration and to the reviewer. A value taken out (``[#0]``) has
    no path served."""
    out: set[str] = set()
    for flag in flags:
        path, _, kind = str(flag).rpartition(":")
        if kind not in REVIEW_ROUTED or not path or "[#" in path:
            continue
        out.add(path)
        if _LAST_INDEX.search(path):
            out.add(_LAST_INDEX.sub("", path))
    return out


def route_served(extraction: Any, flags: Iterable[str]) -> int:
    """Every served envelope of ``extraction`` (canonical, ``flagged`` in it)
    that ``flags`` send to review marked ``flagged``, in place; how many were
    not flagged before."""
    from common.canonical import is_field_value

    routed = routed_paths(flags)
    if not routed:
        return 0
    marked = 0

    def mark(node: dict[str, Any]) -> None:
        nonlocal marked
        if node.get("raw") is None and node.get("parsed") is None:
            return                                   # a null serves nothing to review
        if node.get("flagged") is not True:
            marked += 1
        node["flagged"] = True

    def walk(node: Any, path: str) -> None:
        if is_field_value(node):
            if path in routed:
                mark(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            whole = path in routed
            for index, item in enumerate(node):
                here = f"{path}[{index}]"
                if is_field_value(item) and (whole or here in routed):
                    mark(item)
                else:
                    walk(item, here)

    walk(extraction, "")
    return marked


def near_tie_flags(spans: Mapping[str, Any], margin: float | None = None) -> list[str]:
    """``<path>:near_tie`` for each value whose span's top-2 margin is below
    ``margin`` (default :func:`near_tie_margin`). Only a value served - a
    span with a value; nothing for a null or an empty string."""
    limit = near_tie_margin() if margin is None else margin
    out: list[str] = []
    for path, span in spans.items():
        gap = getattr(span, "top2_margin", None)
        value = getattr(span, "value", None)
        if gap is None or value is None or value == "" or not getattr(span, "mapped", False):
            continue
        if gap < limit:
            out.append(f"{path}:{NEAR_TIE_FLAG}")
    return sorted(out)
