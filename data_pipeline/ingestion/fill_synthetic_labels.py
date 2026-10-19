"""Complete synthetic labels from their source's reviewed gold, without regenerating anything.

A synthetic twin is its source document with identifying values replaced; its
label is what the generator could read and place, about 29% of the fields the
reviewed gold of that source holds. The model reads an omitted field as "not on
the page", so trained on those labels it learns to skip most of a policy.

"Reviewed" is what the source gold must be, not something this checks: a twin
filled from a seed nobody reviewed (its ``seed_review`` says so) is still
refused at import until a reviewer is recorded
(``export_golden_labels.review_problem``, decision 2026-10-19).

This adds the missing fields. It never changes a field the synthetic label
already has - those are the generator's values, which match the synthetic PDF.
Each field added comes from the source's reviewed gold, made true of the twin:

* **unchanged on the page** - coverage names, descriptions, form numbers, edition
  dates, the carrier's own name and address: added as they are;
* **amounts**: added only when every amount the two labels share is equal - some
  templates' amounts differ, and a scan cannot say which side is right;
* **full dates**: shifted by the twin's recorded ``date_shift_days``;
* **identifying values** (names of people and agencies, addresses, phones,
  emails, policy/account/licence numbers, VINs): added only when that exact
  value is known to have been replaced - a field both labels share maps old to
  new, and the generator replaces a value the same way everywhere. Otherwise it
  is left out: never guessed, never left real.

Table rows are matched to the twin's rows on values the generator does not
change (a coverage name, a form number, a make and year), never by position
alone; a matched row gains only its missing fields. A merged label must pass
the schema, or the twin keeps its label. After OCR on the pod, the audit checks
every value against the page, which is where anything this got wrong shows.

    python -m data_pipeline.ingestion.fill_synthetic_labels --input "data/training data"            # dry run
    python -m data_pipeline.ingestion.fill_synthetic_labels --input "data/training data" --apply
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from common.normalize import infer_field_kind, normalize_currency, normalize_date

DEFAULT_REVIEWED = Path("data") / "source data" / "gold json"

#: Recorded in a completed label's ``fideon:filled``, beside the paths added.
FILLED_FROM = "reviewed gold of the source (fill_synthetic_labels)"

#: Sections copied from neither side: free text the generator rewrites itself.
SKIP_KEYS = ("text_sections",)

_PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
_EMAIL = re.compile(r"\S+@\S+\.\w+")
_POBOX = re.compile(r"\bP\.?\s*O\.?\s*Box\b", re.I)
_STREET = re.compile(r"\b\d+\s+\w[\w .'-]*\b(Rd|Road|St|Street|Ave|Avenue|Ln|Lane|Dr|Drive|Ct|Court|Way|Blvd|"
                     r"Boulevard|Hwy|Highway|Route|Rte|Pl|Place|Pkwy|Parkway|Cir|Circle|Ter|Terrace|Trl|Trail)\b\.?", re.I)
_CITYLINE = re.compile(r",\s*[A-Z]{2}\s+\d{5}(-\d{4})?\b")
_VIN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
_MONTH_YEAR = re.compile(r"^\s*(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2}|\d{4})\s*$")

#: Field names that hold a party's contact details or an identifying number.
_IDENT_LEAF = re.compile(
    r"(^|_)(dba|address|addresses|street|line_1|line_2|line1|line2|city|postal|zip|phone|fax|email|"
    r"vin|policy_number|account|license|licence|loan|fein|tin|ssn|certificate_number|claim_number|file_number|"
    r"customer_number|member_number|hull|registration|serial|signature|signer)(_|$)"
)
#: A name is a party's only when it names a party role: the insured, a driver, an
#: agent, a lienholder... A coverage, discount or form name is the name of a thing.
_PARTY = re.compile(
    r"insured|driver|operator|owner|mortgagee|lienholder|interested_part|loss_payee|producer|agent|agency|"
    r"contact|applicant|representative|signer|signature|person|primary|secondary|first|last|middle|full|"
    r"employer|lessor|lessee|trustee|beneficiary|household|resident|member|occupant|company"
)


def _leaf(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1])


def is_identifying(path: str, value: Any) -> bool:
    """Is this a value the generator replaces: a party's identity or contact details?"""
    leaf = _leaf(path)
    if path.startswith("carrier.") and not re.search(r"phone|fax|email", leaf):
        return False                                   # the carrier's own name and address stay
    if leaf == "source_file_name":
        return False
    if _IDENT_LEAF.search(leaf):
        return True
    if leaf == "name" or leaf.endswith("_name") or leaf == "names":
        owner = leaf[: -len("_name")] if leaf.endswith("_name") else ""
        parents = re.sub(r"\[\d+\]", "", path).rsplit(".", 1)[0] if "." in path else ""
        if _PARTY.search(owner) or (not owner and _PARTY.search(parents.rsplit(".", 1)[-1])):
            return True
    text = "" if value is None else str(value)
    return bool(_PHONE.search(text) or _EMAIL.search(text) or _POBOX.search(text)
                or _STREET.search(text) or _CITYLINE.search(text) or _VIN.match(text.strip()))


def _is_envelope(node: Any) -> bool:
    return isinstance(node, dict) and "raw" in node and "parsed" in node


def _flat(node: Any, path: str = ""):
    if _is_envelope(node):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            if not key.startswith("fideon:") and key not in SKIP_KEYS:
                yield from _flat(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _flat(value, f"{path}[{index}]")


def _norm(value: Any) -> str:
    return " ".join(str(value).casefold().split()) if value is not None else ""


# --------------------------------------------------------------------------
# Dates, shifted in the format they are written in
# --------------------------------------------------------------------------

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]


def _render_like(original: str, new: date) -> str | None:
    """``new`` written the way ``original`` writes its date, or None if the layout is unknown."""
    s = original.strip()
    m = re.fullmatch(r"(\d{1,2})([/-])(\d{1,2})\2(\d{4}|\d{2})", s)
    if m:
        month = f"{new.month:02d}" if len(m.group(1)) == 2 else str(new.month)
        day = f"{new.day:02d}" if len(m.group(3)) == 2 else str(new.day)
        year = str(new.year) if len(m.group(4)) == 4 else f"{new.year % 100:02d}"
        return f"{month}{m.group(2)}{day}{m.group(2)}{year}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return new.isoformat()
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m.group(1).lower()[:3] in [x[:3] for x in _MONTHS]:
        word = m.group(1)
        name = _MONTHS[new.month - 1]
        name = name[:3] if len(word.rstrip(".")) <= 4 and word.lower() not in _MONTHS else name
        name = name.upper() if word.isupper() else name.capitalize() if word[0].isupper() else name
        comma = "," if "," in s else ""
        return f"{name} {new.day}{comma} {new.year}"
    return None


def shift_date(value: Any, days: int) -> Any:
    """A full date moved by ``days`` in its own layout; a month/year kept; None if unknown."""
    if value in (None, ""):
        return value
    text = str(value)
    if _MONTH_YEAR.match(text):
        return value                                   # edition dates: the generator leaves them
    iso = normalize_date(text)
    if not iso:
        return None
    try:
        moved = date.fromisoformat(iso) + timedelta(days=days)
    except ValueError:
        return None
    return _render_like(text, moved)


# --------------------------------------------------------------------------
# One twin
# --------------------------------------------------------------------------


@dataclass
class FillStats:
    added: Counter = field(default_factory=Counter)        # section -> fields added
    dropped: Counter = field(default_factory=Counter)      # reason -> fields left out
    rows_matched: int = 0
    rows_added: int = 0


@dataclass
class _Context:
    shift: int | None
    amounts_ok: bool
    replaced: dict[str, str]           # old raw (normalised) -> new raw
    replaced_parsed: dict[str, Any]    # old parsed (normalised) -> new parsed
    substrings: list[tuple[str, str]]  # (old, new), longest first, for text that quotes a party
    pages: int | None
    stats: FillStats
    #: The envelopes this fill added, kept as objects: a row's index can change
    #: between the source and the merged label, an object's identity cannot.
    added: list[dict] = field(default_factory=list)


def _context(source: dict, twin: dict, pages: int | None, stats: FillStats) -> _Context:
    src, syn = dict(_flat(source)), dict(_flat(twin))
    shift = (twin.get("fideon:provenance") or {}).get("date_shift_days")
    amounts_ok = True
    replaced: dict[str, str] = {}
    replaced_parsed: dict[str, Any] = {}
    for path in src.keys() & syn.keys():
        a, b = src[path], syn[path]
        if infer_field_kind(path) == "currency":
            x, y = normalize_currency(str(a.get("raw"))), normalize_currency(str(b.get("raw")))
            if x is not None and y is not None and abs(float(x) - float(y)) > 0.005:
                amounts_ok = False
        if is_identifying(path, a.get("raw")) and a.get("raw") not in (None, "") and _norm(a["raw"]) != _norm(b.get("raw")):
            replaced.setdefault(_norm(a["raw"]), b.get("raw"))
            if a.get("parsed") not in (None, ""):
                replaced_parsed.setdefault(_norm(a["parsed"]), b.get("parsed"))
    substrings = sorted(((old, str(new)) for old, new in replaced.items() if len(old) >= 5 and new),
                        key=lambda kv: -len(kv[0]))
    return _Context(shift if isinstance(shift, int) else None, amounts_ok, replaced, replaced_parsed,
                    substrings, pages, stats)


def _section(path: str) -> str:
    return re.split(r"[.\[]", path, maxsplit=1)[0]


def _transform(envelope: dict, path: str, ctx: _Context) -> dict | None:
    """The source's envelope made true of the twin, or None to leave the field out."""
    raw, parsed = envelope.get("raw"), envelope.get("parsed")
    refs = envelope.get("page_ref") or []
    if ctx.pages and any(isinstance(p, int) and p > ctx.pages for p in refs):
        ctx.stats.dropped["page beyond the twin"] += 1
        return None
    out = copy.deepcopy(envelope)
    kind = infer_field_kind(path)
    if raw in (None, "") and parsed in (None, ""):
        return out
    if kind == "date":
        # Not a full date ("01 06", "0699", "06/99"): an edition date or a
        # placeholder, which the generator leaves as printed.
        if not normalize_date(str(raw if raw not in (None, "") else parsed)):
            return out
        if ctx.shift is None:
            ctx.stats.dropped["date: no recorded shift"] += 1
            return None
        new_raw, new_parsed = shift_date(raw, ctx.shift), shift_date(parsed, ctx.shift)
        if (raw not in (None, "") and new_raw is None) or (parsed not in (None, "") and new_parsed is None):
            ctx.stats.dropped["date: layout not recognised"] += 1
            return None
        out["raw"], out["parsed"] = new_raw, new_parsed
        return out
    if kind == "currency" and not ctx.amounts_ok:
        ctx.stats.dropped["amount: this template's amounts differ"] += 1
        return None
    if is_identifying(path, raw):
        new = ctx.replaced.get(_norm(raw))
        if new is None:
            ctx.stats.dropped["identifying: replacement unknown"] += 1
            return None
        out["raw"] = new
        if parsed not in (None, ""):
            out["parsed"] = ctx.replaced_parsed.get(_norm(parsed), new if _norm(parsed) == _norm(raw) else None)
            if out["parsed"] is None:
                ctx.stats.dropped["identifying: replacement unknown"] += 1
                return None
        return out
    # Free text can quote a party replaced elsewhere ("... at 291 Poplar Point Road").
    for key in ("raw", "parsed"):
        value = out.get(key)
        if isinstance(value, str) and ctx.substrings:
            lowered = value.casefold()
            for old, new in ctx.substrings:
                if old in lowered:
                    value = re.sub(re.escape(old), new.replace("\\", "\\\\"), value, flags=re.I)
                    lowered = value.casefold()
            out[key] = value
    if isinstance(out.get("raw"), str) and is_identifying("text", out["raw"]) and _norm(out["raw"]) not in {
            _norm(v) for v in ctx.replaced.values()}:
        ctx.stats.dropped["text quoting an unknown party"] += 1
        return None
    return out


def _anchors(item: Any, prefix: str) -> set[tuple[str, str]]:
    """(field, value) pairs of a row the generator does not change."""
    found = set()
    for path, env in _flat(item, prefix):
        raw = env.get("raw")
        if raw in (None, "") or len(_norm(raw)) < 2:
            continue
        if infer_field_kind(path) == "date" or is_identifying(path, raw):
            continue
        found.add((_leaf(path), _norm(raw)))
    return found


def _merge(src: Any, syn: Any, path: str, ctx: _Context) -> Any:
    if _is_envelope(src):
        if syn is not None:
            return syn                                 # the twin's own value: never touched
        added = _transform(src, path, ctx)
        if added is not None:
            ctx.stats.added[_section(path)] += 1
            ctx.added.append(added)
        return added
    if isinstance(src, dict):
        result = dict(syn) if isinstance(syn, dict) else {}
        for key, value in src.items():
            if key.startswith("fideon:") or key in SKIP_KEYS:
                continue
            merged = _merge(value, result.get(key), f"{path}.{key}" if path else key, ctx)
            if merged is not None and merged != {} and merged != []:
                result[key] = merged
        return result
    if isinstance(src, list):
        return _merge_rows(src, syn if isinstance(syn, list) else [], path, ctx)
    return syn if syn is not None else src


def _merge_rows(src: list, syn: list, path: str, ctx: _Context) -> list:
    if not all(isinstance(i, dict) for i in src) or not all(isinstance(i, dict) for i in syn):
        return syn or src
    used: set[int] = set()
    matches: dict[int, int] = {}
    for index, item in enumerate(src):
        mine = _anchors(item, f"{path}[{index}]")
        match = next((j for j, other in enumerate(syn) if j not in used
                      and mine & _anchors(other, f"{path}[{j}]")), None)
        if match is None and len(src) == len(syn) and not mine and index not in used:
            match = index                              # rows with nothing unchanged: position, same count only
        if match is not None:
            used.add(match)
            matches[index] = match
    # A twin row no source row matched may be one of the unmatched source rows
    # read differently; adding those would list it twice. Rows are added only
    # when every row the twin has is accounted for.
    may_add = len(used) == len(syn)
    out = []
    for index, item in enumerate(src):
        if index in matches:
            ctx.stats.rows_matched += 1
            out.append(_merge(item, syn[matches[index]], f"{path}[{index}]", ctx))
        elif may_add:
            row = _merge(item, None, f"{path}[{index}]", ctx)
            if row:
                ctx.stats.rows_added += 1
                out.append(row)
        else:
            ctx.stats.dropped["table row: twin has rows that did not match"] += 1
    out += [row for j, row in enumerate(syn) if j not in used]
    return out


def fill(source: dict, twin: dict, *, pages: int | None = None) -> tuple[dict, FillStats]:
    """The twin's label with the source's missing fields added (see the module docstring)."""
    stats = FillStats()
    ctx = _context(source, twin, pages, stats)
    merged = _merge(source, twin, "", ctx)
    for key, value in twin.items():                   # fideon:* and free text stay as the twin had them
        if key.startswith("fideon:") or key in SKIP_KEYS:
            merged[key] = copy.deepcopy(value)
    absent = merged.get("fideon:absent")
    if isinstance(absent, list):
        present = {re.sub(r"\[\d+\]", "[]", p) for p, _ in _flat(merged)}
        merged["fideon:absent"] = [p for p in absent if p not in present]
    # Which values were computed here rather than read off the twin's page.
    # ocr_check compares exactly these with the page, and label_verification
    # keeps one for training only where the page prints it; unrecorded, both
    # had nothing to check and every computed value was trusted.
    mine = {id(node) for node in ctx.added}
    paths = [path for path, node in _flat(merged) if id(node) in mine]
    earlier = ((twin.get("fideon:filled") or {}).get("paths")) or []
    if paths or earlier:
        merged["fideon:filled"] = {
            "from": FILLED_FROM,
            "paths": list(dict.fromkeys([*earlier, *paths])),
        }
    return merged, stats


# --------------------------------------------------------------------------
# The delivery
# --------------------------------------------------------------------------


@dataclass
class DeliveryReport:
    documents: int = 0
    filled: int = 0
    refused: Counter = field(default_factory=Counter)
    fields_before: int = 0
    fields_after: int = 0
    added: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)
    rows_matched: int = 0
    rows_added: int = 0

    def describe(self) -> str:
        lines = [f"{self.filled} of {self.documents} synthetic label(s) completed; fields {self.fields_before} -> "
                 f"{self.fields_after} (x{self.fields_after / max(1, self.fields_before):.2f})",
                 f"  table rows: {self.rows_matched} matched, {self.rows_added} added"]
        lines += [f"  added {n:6}  {section}" for section, n in self.added.most_common(12)]
        lines += [f"  left out {n:6}  {reason}" for reason, n in self.dropped.most_common()]
        lines += [f"  refused {n}: {reason}" for reason, n in self.refused.most_common()]
        return "\n".join(lines)


def fill_delivery(delivery: Path, reviewed: Path, *, lines: frozenset[str] | None = None,
                  apply: bool = False) -> DeliveryReport:
    from common.schemas import is_common_model, iter_validation_errors

    report = DeliveryReport()
    with (delivery / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row.get("kind") != "synthetic" or (lines is not None and row["lob"] not in lines):
            continue
        gold_path = delivery / row["gold"]
        if not gold_path.is_file():
            continue
        report.documents += 1
        if is_common_model("policy", None, row["lob"]):
            # SPEC_21 twins are written complete by their generator (its check
            # C3). Copying a seed's values in would put the seed's ids and
            # values into a twin whose own differ.
            report.refused["common-model line: its twins are complete"] += 1
            continue
        source_gold = reviewed / Path(row["source"]).with_suffix(".json")
        if not source_gold.is_file():
            report.refused["no reviewed gold for the source"] += 1
            continue
        twin = json.loads(gold_path.read_text(encoding="utf-8"))
        source = json.loads(source_gold.read_text(encoding="utf-8"))
        pages = int(row["pages"]) if str(row.get("pages", "")).isdigit() else None
        merged, stats = fill(source, twin, pages=pages)
        errors = list(iter_validation_errors(merged, "policy", None, row["lob"]))
        before, after = sum(1 for _ in _flat(twin)), sum(1 for _ in _flat(merged))
        if errors:
            report.refused[f"schema: {errors[0][:80]}"] += 1
            report.fields_before += before
            report.fields_after += before
            continue
        report.filled += 1
        report.fields_before += before
        report.fields_after += after
        report.added.update(stats.added)
        report.dropped.update(stats.dropped)
        report.rows_matched += stats.rows_matched
        report.rows_added += stats.rows_added
        if apply:
            # In place: the bundles hard-link these files and see the change.
            with gold_path.open("r+", encoding="utf-8") as fh:
                fh.seek(0)
                fh.write(json.dumps(merged, indent=2, ensure_ascii=False))
                fh.truncate()
    return report


def main(argv: list[str] | None = None) -> int:
    from common.scopes import get_scope

    parser = argparse.ArgumentParser(description="Complete synthetic labels from their source's reviewed gold")
    parser.add_argument("--input", required=True, type=Path, help="the delivery: Train/Val/Test + manifest.csv")
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED,
                        help="reviewed gold, as <carrier>/<lob>/<source>.json")
    parser.add_argument("--scope", default="personal_lines")
    parser.add_argument("--apply", action="store_true", help="write the labels (default: report only)")
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.ingestion.fill_synthetic_labels", argv):
        return 0
    lines = None if args.scope == "none" else get_scope(args.scope).lines or None
    report = fill_delivery(args.input, args.reviewed, lines=lines, apply=args.apply)
    print(report.describe())
    print("\nwritten" if args.apply else "\ndry run: nothing written (add --apply)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
