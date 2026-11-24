"""Propose the field-type table (configs/field_types.yaml) for review.

    python scripts/propose_field_types.py --bundles data/bundles

A field's type decides its error target when accepted without review (money,
dates and identifiers 1%, enums 2%, names and addresses 5%, free text never),
so it has to be right. Guessed from the field name alone, 130 money, number and
identifier fields fell into free text. This proposes a type for EVERY field of
the canonical policy schemas from two independent signals - the name, and the
values the gold labels actually hold - and marks each field where they disagree,
or where the labels hold too few values to say, for a person to decide.

A field the common model declares a type for (a ``MoneyValue`` is money) is
not proposed: serving and fitting type it from the schema on the seven
common-model lines, and a row in the table - read first, for every line, by
bare path - would override that declaration and re-type the same path on the
self-contained lines. It is listed as a comment naming the declared type.

The output is a proposal: nothing reads it until it is reviewed and committed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LINES = ("homeowners", "personal_auto", "dwelling_fire", "recreational_vehicle",
         "personal_umbrella", "ocean_marine", "motorcycle")

MONEY_WORDS = ("premium", "limit", "amount", "deductible", "value", "cost", "fee", "price", "charge",
               "tax", "surcharge", "balance", "due", "paid", "reserve", "payment", "assessment",
               "remuneration", "sublimit", "retention", "benefit", "benefits", "expenses",
               "loss", "maximum", "reimbursement", "assistance", "limits", "price")
NUMBER_WORDS = ("percent", "percentage", "rate", "count", "number_of", "points", "year", "age",
                "miles", "mileage", "length", "horsepower", "displacement", "stories", "units",
                "days", "weeks", "months", "hours", "head_count", "factor")
ID_WORDS = ("number", "id", "vin", "hull", "serial", "code", "license", "docket", "naic", "fein")
# Names of closed sets. Few distinct values alone is not enough: the corpus
# repeats a handful of templates, so makes, models and form numbers look closed
# too - and a wrong enum lets a free-text field be auto-accepted.
ENUM_WORDS = ("gender", "basis", "use", "plan", "method", "party", "category", "answer",
              "causes_of_loss", "protection_class", "included", "present", "assessable",
              "renewal")
DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$|^\d{4}-\d{2}-\d{2}$")
NUMERIC = re.compile(r"^-?\$?\(?-?[\d,]+(\.\d+)?\)?%?$")


def schema_fields(lines: tuple[str, ...] = LINES) -> set[str]:
    """Every canonical policy field of ``lines``, as a path with list markers removed.

    A value is one field, never its ``raw``/``parsed``/``page_ref``: a
    ``FieldValue``, and on a common-model line each typed value too
    (``MoneyValue``, ``DateValue``, ...). Those are the money and date fields
    whose type matters most, so missing them leaves exactly those fields with no
    row to review. A row written as variants (a common-model ``Limit`` or
    ``Deductible``) holds the fields of every variant, as
    :func:`common.schemas.row_keys` reads it.
    """
    from common import schemas

    def walk(node, defs, path, out, depth=0):
        if depth > 25 or not isinstance(node, dict):
            return
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            node = defs.get(name, {})
            # A value type has raw, parsed and page_ref (common.model_view).
            if name == "FieldValue" or {"raw", "parsed", "page_ref"} <= set(node.get("properties") or {}):
                out.add(path)
                return
        if node.get("type") == "array":
            walk(node.get("items", {}), defs, path, out, depth + 1)
            return
        for key, value in (node.get("properties") or {}).items():
            walk(value, defs, f"{path}.{key}" if path else key, out, depth + 1)
        for variant in [*(node.get("anyOf") or []), *(node.get("oneOf") or [])]:
            walk(variant, defs, path, out, depth + 1)

    out: set[str] = set()
    for line in lines:
        schema = schemas.resolved_schema("policy", None, line, None)
        walk(schema, schema.get("$defs", {}), "", out)
    return out


def label_values(bundles: Path) -> dict[str, list]:
    """Every value the gold labels hold, by normalised path."""
    from common.canonical import values_view
    from common.label_mapping import map_label
    from evaluation.metrics.field_accuracy import flatten_scalars

    values: dict[str, list] = defaultdict(list)
    for golden in bundles.glob("*/golden.json"):
        meta_path = golden.parent / "metadata.json"
        lob = json.loads(meta_path.read_text(encoding="utf-8")).get("lob") if meta_path.exists() else None
        label = map_label(json.loads(golden.read_text(encoding="utf-8")), lob)
        for path, value in flatten_scalars(values_view(label)).items():
            if value not in (None, "", []):
                values[normalise(path)].append(value)
    return values


def has_word(leaf: str, words: tuple[str, ...]) -> bool:
    """Whether ``leaf`` holds any of ``words`` as whole ``_``-separated words:
    "count" must not match inside "county", nor "age" inside "mileage"."""
    padded = f"_{leaf}_"
    return any(f"_{w}_" in padded for w in words)


def normalise(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path)


def from_values(values: list) -> tuple[str | None, str]:
    """A type the values support, and how strongly."""
    if len(values) < 5:
        return None, f"only {len(values)} value(s) in the labels"
    text = [str(v).strip() for v in values]
    share = lambda test: sum(map(test, text)) / len(text)  # noqa: E731
    if share(lambda t: bool(DATE.match(t))) >= 0.8:
        return "date", "values are dates"
    numeric = share(lambda t: bool(NUMERIC.match(t.replace(" ", ""))))
    if numeric >= 0.8:
        return "numeric", "values are numbers"
    distinct = len(set(t.casefold() for t in text))
    # At least two values: a template's synthetic copies repeat one value, and
    # one repeated value says nothing about a closed set.
    if 2 <= distinct <= 12 and len(text) >= 20 and max(map(len, text)) <= 40:
        return "enum", f"{distinct} distinct short values"
    return None, "values are varied text"


def propose(path: str, values: list) -> dict:
    """The proposed type of ``path``, or - when the common model declares one -
    that type with ``declared`` set: not a row for the table, which would
    override the declaration on every line."""
    from calibration.features import common_model_field_types, infer_field_type

    declared = common_model_field_types().get(normalise(path))
    if declared:
        return {"type": declared, "by_name": declared, "values": "declared by the common model",
                "review": False, "declared": True, "examples": [str(v)[:30] for v in values[:3]]}
    leaf = path.rsplit(".", 1)[-1].casefold()
    by_name = infer_field_type(path)
    by_values, why = from_values(values)
    proposed = by_name
    if by_values == "date":
        proposed = "date"
    elif by_values == "numeric" and by_name == "date":
        # A form edition ("10 00" = October 2000) is printed as digits but read
        # as a code: matched exactly, never as an amount.
        proposed = "identifier"
    elif by_values == "numeric":
        # A unit wins over a money word: "rental_reimbursement_maximum_days" is
        # a count of days, "extended_replacement_cost_percentage" a percentage.
        if has_word(leaf, NUMBER_WORDS):
            proposed = "number"
        elif has_word(leaf, MONEY_WORDS):
            proposed = "money"
        elif has_word(leaf, ID_WORDS):
            proposed = "identifier"
        else:
            proposed = "number"
    elif by_values == "enum" and by_name == "free_text" and (
            has_word(leaf, ENUM_WORDS) or leaf.startswith("is_")):
        proposed = "enum"
    elif by_values is None and by_name == "free_text" and (
            has_word(leaf, ("vin", "serial")) or "hull_id" in leaf):
        proposed = "identifier"
    # A person decides where the name and the label values point different ways.
    # Where they agree - or the labels hold too few values to say - the name stands.
    review = proposed != by_name
    return {"type": proposed, "by_name": by_name, "values": why, "review": review,
            "declared": False, "examples": [str(v)[:30] for v in values[:3]]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundles", type=Path, default=Path("data/bundles"))
    parser.add_argument("--out", type=Path, default=Path("configs/field_types.proposed.yaml"))
    args = parser.parse_args(argv)

    values = label_values(args.bundles)
    proposals = {path: propose(path, values.get(path, [])) for path in sorted(schema_fields())}
    lines = ["# PROPOSED field types - review, then save as configs/field_types.yaml.",
             "# One line per canonical policy field. `review: true` = name and label values",
             "# disagree, or the labels hold too few values: decide those by hand.",
             "# Types: identifier money date number enum entity address free_text",
             "# (free_text is never accepted without review).",
             "# A field the common model types (`# path: declared ...`) is not tabled: a row",
             "# would override its declared type, and re-type the path on every line.", "fields:"]
    for path, p in proposals.items():
        if p["declared"]:
            lines.append(f"  # {path}: {p['type']}  (declared by the common model; not tabled)")
            continue
        note = f"name says {p['by_name']}; {p['values']}; e.g. {p['examples']}"
        flag = "  # REVIEW: " if p["review"] else "  # "
        lines.append(f"  {path}: {p['type']}{flag}{note}")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = sum(p["review"] for p in proposals.values())
    declared = sum(p["declared"] for p in proposals.values())
    counts: dict[str, int] = defaultdict(int)
    for p in proposals.values():
        if not p["declared"]:
            counts[p["type"]] += 1
    print(f"{len(proposals)} fields -> {args.out}; {declared} declared by the common model, "
          f"not tabled; {review} to review")
    print("by type:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
