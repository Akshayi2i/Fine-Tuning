"""A served value's ``raw`` read again by its declared type, against its ``parsed``.

The printed-values check looks up what a value PRINTS - its ``raw`` - and never
reads its ``parsed`` against it. The model writes the two apart: in the fast
profile's compact answer ``r`` (the printed form) and ``v`` (the value) are two
strings of tokens (``common.canonical.to_compact_answer``), and in the full form
``raw`` and ``parsed`` likewise. So ``{"r": "$1,000", "v": 10000}`` is served as
a printed $1,000 holding ten thousand, and ``{"r": "04/05/2026", "v":
"05/04/2026"}`` with its day and month swapped. The data audit reads a gold's
two against each other (``data_pipeline.audit``); serving did not.

Here ``raw`` is read as the field's declared type reads (``common.normalize``),
and compared with ``parsed``:

* DateValue - both read as dates (``normalize_date``: a US page's 04/05/2026 is
  April 5);
* MoneyValue, NumberValue, YearValue - a ``raw`` that is one figure and nothing
  else ("$1,000", "(250.00)", "2019"), against the number, sign aside (a credit
  is printed $109 and recorded -109);
* PercentValue - a figure with or without "%": 5% is 5, or 0.05;
* StateValue - a two-letter code or a state's name, against the code;
* PostalCodeValue - a ZIP or ZIP+4, against the ZIP (and the +4 when both have one).

A field the common model declares as a plain ``FieldValue`` is read by its name
instead, as the audit reads it: a date by name as a date, an amount by name as
money. Conservative throughout: a ``raw`` that does not read cleanly as its type
("Included", "$1M", "01/01/2026 12:01 AM", "100/300") settles nothing, and
nothing is done.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any

#: The typed values compared, by the common model's definition names.
TYPED = frozenset({"DateValue", "MoneyValue", "NumberValue", "YearValue", "PercentValue",
                   "StateValue", "PostalCodeValue"})
_FIGURE_TYPES = frozenset({"MoneyValue", "NumberValue", "YearValue"})

#: One figure and nothing else: a "$", thousands separators, cents, a sign or
#: brackets (a credit). "1M", "100/300", "1,000 each" are not.
_ONE_FIGURE = re.compile(r"^\s*[(\-]?\s*\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*\)?\s*$")
_PERCENT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")
_ZIP = re.compile(r"^\s*(\d{5})(?:\s*-?\s*(\d{4}))?\s*$")
_INDEX = re.compile(r"\[\d*\]")

STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN",
    "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "puerto rico": "PR", "guam": "GU", "american samoa": "AS",
    "u s virgin islands": "VI", "us virgin islands": "VI", "virgin islands": "VI",
    "northern mariana islands": "MP",
}
_CODES = frozenset(STATES.values())


@cache
def declared_value_types() -> dict[str, str]:
    """Field path (list markers removed) -> the typed value the common-model
    lines declare it as, for the types in :data:`TYPED` - one table for every
    common-model line, as the common model gives a path one type wherever it
    appears (``calibration.features.common_model_field_types``)."""
    from common.schemas import is_common_model, resolve_local, resolved_schema, schema_selectors

    out: dict[str, str] = {}

    def value_type(sub: Any, defs: dict[str, Any]) -> str:
        ref = sub.get("$ref") if isinstance(sub, dict) else None
        name = ref.rsplit("/", 1)[-1] if isinstance(ref, str) else ""
        node = defs.get(name)
        is_value = isinstance(node, dict) and {"raw", "parsed", "page_ref"} <= set(node.get("properties") or {})
        return name if is_value else ""

    def walk(node: dict[str, Any], defs: dict[str, Any], path: str, seen: frozenset[str]) -> None:
        for variant in [node, *(node.get("anyOf") or [])]:
            props = (variant.get("properties") or {}) if isinstance(variant, dict) else {}
            for key, sub in props.items():
                here = f"{path}.{key}" if path else key
                resolved = resolve_local(sub, defs)
                items = resolved.get("items") if isinstance(resolved, dict) else None
                name = value_type(sub, defs) or value_type(items, defs)
                if name:
                    if name in TYPED:
                        out.setdefault(here, name)
                    continue
                refs = {r.rsplit("/", 1)[-1] for r in (
                    sub.get("$ref") if isinstance(sub, dict) else None,
                    items.get("$ref") if isinstance(items, dict) else None) if isinstance(r, str)}
                if refs & seen:
                    continue
                for child in (resolved, resolve_local(items, defs) if items is not None else None):
                    if isinstance(child, dict) and (child.get("properties") or child.get("anyOf")):
                        walk(child, defs, here, seen | refs)

    for doc_type, form, lob in schema_selectors():
        if doc_type == "policy" and is_common_model(doc_type, form, lob):
            view = resolved_schema(doc_type, form, lob)
            walk(view, view.get("$defs") or {}, "", frozenset())
    return out


def kind_of(path: str) -> str | None:
    """The type ``path``'s value is read as: its declared typed value, else -
    by its name - ``DateValue`` for a date and ``MoneyValue`` for an amount;
    None for anything else (text, codes, yes/no readings, enums)."""
    from common.normalize import infer_field_kind

    bare = _INDEX.sub("", path)
    declared = declared_value_types().get(bare)
    if declared:
        return declared
    by_name = infer_field_kind(path)
    return {"date": "DateValue", "currency": "MoneyValue"}.get(by_name)


def _figure(value: Any) -> float | None:
    from common.normalize import normalize_currency

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _ONE_FIGURE.match(value):
        return normalize_currency(value)
    return None


def _state(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    letters = re.sub(r"[^a-z ]+", " ", value.casefold())
    letters = " ".join(letters.split())
    if letters.replace(" ", "").upper() in _CODES and len(letters.replace(" ", "")) == 2:
        return letters.replace(" ", "").upper()
    return STATES.get(letters)


def contradicts(kind: str | None, raw: Any, parsed: Any) -> bool:
    """Whether ``raw``, read as ``kind`` reads, is clearly another value than
    ``parsed``. False whenever either does not read cleanly: nothing is
    settled then."""
    from common.normalize import normalize_date

    if kind is None or parsed is None or raw is None or isinstance(parsed, bool) or isinstance(raw, bool):
        return False
    if isinstance(raw, str) and not raw.strip():
        return False
    if kind == "DateValue":
        printed, held = normalize_date(raw) if isinstance(raw, str) else None, normalize_date(parsed)
        return printed is not None and held is not None and printed != held
    if kind in _FIGURE_TYPES:
        printed, held = _figure(raw), _figure(parsed)
        return printed is not None and held is not None and abs(abs(printed) - abs(held)) > 0.005
    if kind == "PercentValue":
        match = _PERCENT.match(raw) if isinstance(raw, str) else None
        printed = float(match.group(1)) if match else _figure(raw)
        held = _figure(parsed)
        if printed is None or held is None:
            return False
        return not any(abs(printed - candidate) <= 0.005 for candidate in (held, held * 100, held / 100))
    if kind == "StateValue":
        printed, held = _state(raw), _state(parsed)
        return printed is not None and held is not None and printed != held
    if kind == "PostalCodeValue":
        printed = _ZIP.match(raw) if isinstance(raw, str) else None
        held = _ZIP.match(parsed) if isinstance(parsed, str) else None
        if not printed or not held:
            return False
        if printed.group(1) != held.group(1):
            return True
        return bool(printed.group(2) and held.group(2) and printed.group(2) != held.group(2))
    return False


def parsed_mismatch(path: str, envelope: dict[str, Any]) -> bool:
    """Whether the envelope at ``path`` holds a ``parsed`` its ``raw`` clearly
    does not read as (:func:`contradicts`, by :func:`kind_of`)."""
    return contradicts(kind_of(path), envelope.get("raw"), envelope.get("parsed"))
