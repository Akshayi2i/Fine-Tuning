"""Label rules applied as a bundle is prepared (accuracy plan, stage 1).

The delivered golds were drafted, not reviewed, and disagree with each other:
extra fields run from 1 to 221 rows per document and repeat values that have
their own field, a location number is written as "1" or taken from a street
address, page lists miss pages or name pages that do not print the value. The
model copies every one of those habits. The agreed convention:

* a value printed several times appears ONCE, in its proper field, its
  ``page_ref`` listing every page that prints it;
* extra fields hold printed values that have no proper field, one entry per
  label and value, also with every page that prints it;
* a value with its own field is not repeated in the extra fields - the pages
  the extra entry cites go into that field's ``page_ref``.

The rules, each logged as a correction (:func:`apply_label_rules`):

* **page list** - a value's pages become the pages that print it: a page that
  does not is taken out; a document-level value printed on more pages (a policy
  number in every page header) lists them all. A value in a table row keeps to
  the pages it cites: the same amount printed elsewhere may be another row's.
  A name or a phrase keeps every page it cites (it may be printed only in a
  logo), and another page is added only where it heads a line - a header, a
  footer, a "Named Insured: ..." line, whole or followed by a break - not where
  its words fall in a sentence ("as shown in the Policy Declarations"), nor
  where a wrapped sentence starts with them ("Individual members of ...").
* **extra field repeats a field** - folded into that field, as above. Only a
  value printed as itself (an identifier, a name, a large amount) that one
  field holds: "$0.00", "$500" or "2015" in two places is not one value, nor
  is a status or a span ("NOT COVERED", "30 days": :func:`_generic`). A
  table row's value gains no page this way: a row is listed where its table
  lists it, not where it is mentioned.
* **extra field merged** - the same label and value twice becomes one entry.
* **extra field not a value** - a page counter, a run of prose.
* **extra field moved to its field** - Protection Class and a "Paid By:
  Mortgagee" line, where the proper field is empty (and dropped where it holds
  them already).
* **form edition moved** - a form number that carries its edition in a shape
  no form number takes ("CG 00 01 04 13", "IL0017(11-98)"), with the form's
  ``edition_date`` empty, gives the edition to ``edition_date`` (parsed
  ``MM/YY``, the schema's code) and keeps the number. The labels did both, and
  the model learned both.
* **edition parsed** - every form's ``edition_date.parsed`` is derived from its
  ``raw`` as ``MM/YY`` (the schema's ``fideon:code_parsed``; ``None`` when the
  print holds no month and year): the first commercial set wrote it some 19
  ways.
* **link names the building** - an entry that applies to a location with one
  building names the building, as the client's own example does (Coverage A
  and the mortgagee of homeowners_minimal.json name ``bldg_1``). The labels
  named such a premises by its location in some seeds and its building in
  others, so the model learned a habit per layout.
* **number not printed** - a location or building number no page prints as
  one ("Location 1", "Property: 1", "Bldg 2", or a "Loc. #" column with the
  number in a cell of its own), never the house number of the street address.
* **printed nowhere** - a value with figures that no page prints, in a field
  or an extra field, is listed, not changed: the corpus build leaves out the
  windows that hold it (``build_jsonl._policy_window_rows``). Deleting the value
  would teach the model to skip a value the page may print in a form the
  search misses.

What is checked against a page's text is only what that text can settle. A
name is never "printed nowhere": it is often only in a logo, which the text
layer does not hold. A page whose text misses figures its label cites settles
nothing: nothing is taken out on its word. Figures are what the check counts -
a text layer can hold every word of a page and none of its amounts (figures
drawn in a font that records no characters, as on one carrier's declarations).
Nor does the OCR text laid invisibly over a scanned page (:func:`scanned_pages`):
it shows where a value is printed, never that one is not - OCR reads "NIP0115"
as "NIPO115" and "$37.47" as "537.47". Every check reads the digital render's
text layer, which a scanned twin shares with its digital twin through their one
gold; a document with no text layer gets only the rules that need no text.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from contracts.label_rules import (  # noqa: F401 - moved (review A1)
    _BRACKETED_EDITION,
    _CLAIMS_PHONE,
    _CLAUSE_LABEL,
    _DRAFTING_LABEL,
    _EDGE_LINES,
    _EDITION_COMPACT,
    _EDITION_MONTH_YEAR,
    _FIELD_LABELS,
    _FORM_NUMBER,
    _FULL_YEAR_EDITION,
    _GENERIC_WORDS,
    _HEADING_BREAK,
    _ISO_EDITION,
    _JOINED_EDITION,
    _MINIMUM_EARNED,
    _NUMBER_WORDS,
    _PAGE_COUNTER,
    _PAYOR_LABELS,
    _PROTECTION_CLASS,
    _SPACED_EDITION,
    _SYSTEM,
    _WORDING_FIGURE,
    _YEAR,
    PRINTED_NOWHERE,
    RELIABLE_PAGE_SHARE,
    _a_whole_number,
    _breaks_after,
    _building_links,
    _distinctive,
    _envelopes,
    _extra_fields,
    _field_moves,
    _form_code,
    _generic,
    _has_digits,
    _heads_a_line,
    _is_wording,
    _move_to_field,
    _moved_value,
    _not_a_value,
    _number,
    _numbers,
    _on_page_edge,
    _page_lists,
    _pages,
    _printed,
    _prints_number,
    _reliable_pages,
    _settles,
    _whole_number,
    apply_label_rules,
    edition_mm_yy,
    edition_rows,
    parse_form_editions,
    printed_nowhere,
    split_form_edition,
    split_form_editions,
)


class PdfReaderMissing(RuntimeError):
    """The PDF library (PyMuPDF) cannot be loaded on this machine. Every PDF would read as having no text, and the
    rules that need text would be skipped for every document without a word (a laptop whose application control
    blocked the library's DLL, 2026-11-23) - so this stops the step instead."""


def _pdf_library() -> Any:
    try:
        import pymupdf
    except Exception as exc:  # noqa: BLE001 - ImportError, or a DLL the system refuses to load
        raise PdfReaderMissing(f"the PDF library (PyMuPDF) cannot be loaded here ({type(exc).__name__}: {exc}); "
                               "run this where it loads (the pod) or allow it on this machine") from exc
    return pymupdf


def page_texts(pdf: Path) -> dict[int, str]:
    """Each page's text layer by page number; empty when the file has none or
    cannot be read - then only the rules that need no text apply. Raises
    :class:`PdfReaderMissing` when the PDF library itself cannot be loaded."""
    if not pdf.is_file():
        return {}
    pymupdf = _pdf_library()
    try:
        with pymupdf.open(pdf) as doc:
            return {number: page.get_text() for number, page in enumerate(doc, start=1)}
    except Exception:  # noqa: BLE001 - an unreadable PDF only means no text rules
        return {}


def scanned_pages(pdf: Path) -> set[int]:
    """The pages whose text is an OCR reading laid over a scan: most of their
    characters drawn invisibly (PDF text render mode 3), under the image."""
    if not pdf.is_file():
        return set()
    pymupdf = _pdf_library()
    try:
        out = set()
        with pymupdf.open(pdf) as doc:
            for number, page in enumerate(doc, start=1):
                drawn = invisible = 0
                for span in page.get_texttrace():
                    drawn += len(span["chars"])
                    invisible += len(span["chars"]) if span["type"] == 3 else 0
                if invisible * 2 > drawn > 0:
                    out.add(number)
        return out
    except Exception:  # noqa: BLE001 - unreadable: page_texts finds no text either
        return set()


# --------------------------------------------------------------------------
# Form editions
# --------------------------------------------------------------------------

#: A number ending in what may be a month and year, or may be its own groups
#: ("CG 20 10 04"): named for a person to look at, never changed.
_MAYBE_EDITION = re.compile(r"^(?P<form>.*\d.*?)[ /-](?P<mm>0[1-9]|1[0-2])[ /-](?P<yy>\d{2})$")


def may_hold_edition(text: Any) -> bool:
    """A form number that may end in its edition, in a shape a number takes too."""
    value = " ".join(str(text or "").split())
    return split_form_edition(value) is None and bool(_MAYBE_EDITION.match(value))


# --------------------------------------------------------------------------
# Coverage rows and names (user's rules, 2026-11-04). Labels only: serving keys each value's confidence by its row's
# place, so a served answer's rows are not moved; a model taught one row per coverage writes one.
# --------------------------------------------------------------------------

#: The fields of a coverage row that say which row it is, and so are not carried over when two rows become one.
_COVERAGE_ROW_IDENTITY = frozenset({"coverage_code", "coverage_id", "coverage_name", "limits", "applies_to"})


def _plain(cell: Any) -> Any:
    """A cell's value: an envelope's ``parsed``, else its ``raw``; a plain value as it is."""
    from common.canonical import values_view

    return values_view(cell)


def _blank(value: Any) -> bool:
    return value in (None, "", [], {})


def merge_coverage_limits(gold: dict[str, Any]) -> list[tuple[str, str]]:
    """One row per coverage, its limits inside ``limits[]``: IN PLACE; one note per row merged in.

    "EACH OCCURRENCE LIMIT" and "GENERAL AGGREGATE LIMIT" on two rows of one code are two limits of one coverage. Rows
    merge when they share a code and what they apply to, every one holds limits, and no limit type comes twice: two
    rows that each hold a per-claim and an aggregate limit are two coverages, and one with no limit may be anything.
    The first row stays and takes the others' limits, each keeping its printed line as its ``description``, so no
    printed word is lost. A field two rows both fill leaves them apart: nothing is chosen between two values.
    """
    rows = gold.get("coverages")
    if not isinstance(rows, list):
        return []
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        if isinstance(row, dict) and not _blank(_plain(row.get("coverage_code"))):
            key = (str(_plain(row.get("coverage_code"))), json.dumps(_plain(row.get("applies_to")), sort_keys=True))
            groups.setdefault(key, []).append(index)
    notes: list[tuple[str, str]] = []
    gone: set[int] = set()
    for (code, _applies), indexes in groups.items():
        if len(indexes) < 2 or not all(rows[i].get("limits") for i in indexes):
            continue
        limits = [limit for i in indexes for limit in rows[i]["limits"]]
        types = [str(_plain(limit.get("limit_type"))) if isinstance(limit, dict) else "" for limit in limits]
        if "" in types or len(types) != len(set(types)):
            continue
        first, *others = [rows[i] for i in indexes]
        moved: dict[str, Any] = {}
        clash = False
        for other in others:
            for name, value in other.items():
                if name in _COVERAGE_ROW_IDENTITY or _blank(_plain(value)):
                    continue
                if name in moved or not _blank(_plain(first.get(name))):
                    clash = True
                moved[name] = value
        if clash:
            continue
        first.update(moved)
        for other in others:
            taken = copy.deepcopy(other["limits"])
            name = other.get("coverage_name")
            if not _blank(_plain(name)) and _blank(_plain(taken[0].get("description"))):
                taken[0]["description"] = copy.deepcopy(name)
            first["limits"].extend(taken)
            notes.append(("coverage limits merged",
                          f"coverages: {code}: {_plain(name)!r} merged into {_plain(first.get('coverage_name'))!r}"))
        gone.update(indexes[1:])
    if gone:
        gold["coverages"] = [row for index, row in enumerate(rows) if index not in gone]
    return notes


def _name_shape(name: str) -> str:
    """A name with spaces around punctuation gone: printed spacing there is not a broken word."""
    return re.sub(r"\s*([^\w\s])\s*", r"\1", " ".join(name.split())).casefold()


def _spaced_as(name: str, clean: str) -> str:
    """``name`` keeping its own letters, case and punctuation, with only the spaces ``clean`` has inside words."""
    out: list[str] = []
    position = 0
    text = " ".join(name.split())
    for index, char in enumerate(text):
        if not char.isspace():
            out.append(char)
            position += 1
            continue
        beside_punctuation = not (text[index - 1].isalnum() and text[index + 1].isalnum())
        if beside_punctuation:
            out.append(char)
        elif position < len(clean) and clean[position] == " ":
            out.append(char)
            position += 1
    return "".join(out)


def split_name_repairs(names: Iterable[str]) -> dict[str, str]:
    """``{name: repaired}`` for names OCR broke inside their words, from the same name printed whole elsewhere.

    "each occurrence lim it" is repaired because "each occurrence limit" is in the folder, by the letters alone; a name
    no other label prints whole stays as it is, and "products / completed" beside "products/completed" are both left
    alone, since spacing around punctuation is how a page prints them.
    """
    shapes: dict[str, set[str]] = {}
    raw_names = {" ".join(str(name).split()) for name in names if name}
    for name in raw_names:
        shape = _name_shape(name)
        shapes.setdefault(re.sub(r"\s", "", shape), set()).add(shape)
    repairs: dict[str, str] = {}
    for name in raw_names:
        shape = _name_shape(name)
        variants = shapes[re.sub(r"\s", "", shape)]
        clean = min(variants, key=lambda variant: variant.count(" "))
        if shape != clean:
            repairs[name] = _spaced_as(name, clean)
    return repairs


def label_rules_apply(doc_type: str | None, lob: Any) -> bool:
    """Whether the label rules - the form-edition rule among them - hold for a
    gold of ``doc_type`` on line ``lob``: a policy on a common-model line (no
    line is the fallback, which is one).

    The one test of where the rules hold. They are the lines prepare_bundles
    corrects a gold on (``prepare_bundles.corrected_gold``) and serving holds an
    answer to them on (``serving.conventions.conform``); the importers' check,
    the corpus build, the gate and the harness read a gold under them there and
    nowhere else. A self-contained line keeps an edition
    in its form number - nothing ever moved it, in its golds or in its answers -
    so asking it of that line's golds asked what no step on the line produces.
    """
    if doc_type != "policy":
        return False
    from common.lob import merge_line
    from common.schemas import SchemaError, is_canonical, is_common_model

    line = merge_line(lob) or None
    try:
        return is_canonical("policy", None, line) and is_common_model("policy", None, line)
    except SchemaError:
        return False


def under_edition_rule(gold: dict[str, Any], doc_type: str | None, lob: Any) -> dict[str, Any]:
    """``gold`` read under the form-edition rule where it holds (:func:`label_rules_apply`):
    a copy with each edition moved out of its form number, or ``gold`` itself when
    the rule does not hold there or moves nothing.

    For golds stored before the rule: they are not rewritten in Blob, they are
    read under it - by the corpus build as by the gate, so a model is never
    trained on one convention and scored on the other.
    """
    if not isinstance(gold, dict) or not label_rules_apply(doc_type, lob):
        return gold
    if not any(split_form_edition(row["form_number"].get("raw")) for _, row in edition_rows(gold)):
        return gold
    moved = copy.deepcopy(gold)
    split_form_editions(moved)
    return moved


def edition_problems(gold: dict[str, Any], doc_type: str | None, lob: Any) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` for the form editions a gold to import still holds
    in its form numbers, where the rule holds (:func:`label_rules_apply`).

    An edition the rule moves, left in its number with ``edition_date`` empty,
    means the gold never went through the label rules: an error. A number that
    may end in its edition, in a shape a number takes too, is named for a person
    to check. On any other line nothing is asked of the gold.
    """
    if not label_rules_apply(doc_type, lob):
        return [], []
    unmoved, unsure = [], []
    for index, row in edition_rows(gold):
        raw = row["form_number"].get("raw")
        if split_form_edition(raw):
            unmoved.append(f"forms_and_endorsements[{index}] {raw!r}")
        elif may_hold_edition(raw):
            unsure.append(f"forms_and_endorsements[{index}] {raw!r}")
    errors = [
        f"{len(unmoved)} form number(s) still carry their edition with edition_date empty "
        f"({', '.join(unmoved[:3])}): re-run prepare_bundles, whose label rules move it"
    ] if unmoved else []
    warnings = [
        f"{len(unsure)} form number(s) may end in their edition, edition_date empty "
        f"({', '.join(unsure[:3])}): check them by hand"
    ] if unsure else []
    return errors, warnings


# --------------------------------------------------------------------------
# Values and where they are printed
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Cycle 3 rules (user, 2026-11-19: data/review_cycle2/LABELLING_RULES_DRAFT.md). Labels only, applied as a bundle is
# prepared (prepare_bundles) and reported by scripts.check_golds: they are not in apply_label_rules, which serving
# (serving.conventions) and the gate's golds also go through, so neither a served answer nor the yardstick changes.
# --------------------------------------------------------------------------

#: A broken character as a label holds it.
BROKEN_CHARACTER = "\ufffd"
#: The correction kind for a broken character no page text settles: listed for a person, never guessed.
BROKEN_CHARACTER_LEFT = "broken character left"


def _form_key(text: Any) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(text or "").upper())


def strip_form_ref_editions(gold: dict[str, Any]) -> list[tuple[str, str]]:
    """R2: each ``coverages[].form_refs`` entry written as the forms list's ``form_number``, without the edition.

    IN PLACE; one note per entry changed. An entry is changed only to a number the forms list holds: the entry is that
    number with the form's own edition after it ("SP 30 18 10 18" for "SP 30 18", edition "10 18"), or its number part
    (:func:`split_form_edition`) is that number spelled with other spacing. Anything else is left as it is.
    """
    forms: dict[str, str] = {}
    with_edition: dict[str, str] = {}
    for row in gold.get("forms_and_endorsements") or []:
        if not isinstance(row, dict):
            continue
        number = _printed(row.get("form_number") or {}) if isinstance(row.get("form_number"), dict) else None
        if not number:
            continue
        number = " ".join(str(number).split())
        forms.setdefault(_form_key(number), number)
        edition = row.get("edition_date")
        printed = _printed(edition) if isinstance(edition, dict) else None
        if printed:
            with_edition.setdefault(_form_key(f"{number}{printed}"), number)
    notes: list[tuple[str, str]] = []
    if not forms:
        return notes
    for index, row in enumerate(gold.get("coverages") or []):
        refs = row.get("form_refs") if isinstance(row, dict) else None
        if not isinstance(refs, list):
            continue
        for position, ref in enumerate(refs):
            text = ref.get("raw") if isinstance(ref, dict) else ref
            if not isinstance(text, str) or not text.strip():
                continue
            key = _form_key(text)
            if key in forms:
                target = forms[key]
            elif key in with_edition:
                target = with_edition[key]
            elif (split := split_form_edition(text)) and _form_key(split[0]) in forms:
                target = forms[_form_key(split[0])]
            else:
                continue
            if target == text:
                continue
            if isinstance(ref, dict):
                ref["raw"] = target
                if isinstance(ref.get("parsed"), str):
                    ref["parsed"] = target
            else:
                refs[position] = target
            notes.append(("form reference edition removed",
                          f"coverages[{index}].form_refs[{position}] {text!r} -> {target!r}"))
    return notes


def clear_code_descriptions(gold: dict[str, Any]) -> list[tuple[str, str]]:
    """R5: a ``rating_exposures[].class_description`` that is the row's own class code is emptied. IN PLACE."""
    notes: list[tuple[str, str]] = []
    for index, row in enumerate(gold.get("rating_exposures") or []):
        if not isinstance(row, dict):
            continue
        code, description = row.get("class_code"), row.get("class_description")
        if not (isinstance(code, dict) and isinstance(description, dict)):
            continue
        printed_code, printed_description = _printed(code), _printed(description)
        if printed_code and printed_description and _form_key(printed_code) == _form_key(printed_description):
            del row["class_description"]            # removed, not null: the schema holds a field, never null
            notes.append(("class code as description",
                          f"rating_exposures[{index}].class_description {printed_description!r} emptied"))
    return notes


def clear_premiums_without_amount(gold: dict[str, Any]) -> list[tuple[str, str]]:
    """R3: a coverage premium printed as a word ("Included", "Incl.", "N/A"), with no figure, is emptied. IN PLACE."""
    notes: list[tuple[str, str]] = []
    for index, row in enumerate(gold.get("coverages") or []):
        premium = row.get("premium") if isinstance(row, dict) else None
        if isinstance(premium, dict) and _printed(premium) is not None and not _has_digits(_printed(premium)):
            del row["premium"]                      # removed, not null: the schema holds a field, never null
            notes.append(("premium without an amount", f"coverages[{index}].premium {_printed(premium)!r} emptied"))
    return notes


def _broken_pattern(value: str) -> re.Pattern[str]:
    """``value`` with each run of broken characters standing for one or two printed characters each, spaces loose."""
    out = []
    for run in re.finditer(rf"{BROKEN_CHARACTER}+|\s+|[^\s{BROKEN_CHARACTER}]", value.strip()):
        text = run.group(0)
        if text.startswith(BROKEN_CHARACTER):
            out.append(rf"[^\s{BROKEN_CHARACTER}]{{1,{2 * len(text)}}}")
        elif text.isspace():
            out.append(r"\s+")
        else:
            out.append(re.escape(text))
    return re.compile("".join(out))


def repair_broken_characters(gold: dict[str, Any], pages: dict[int, str]) -> list[tuple[str, str]]:
    """R8: a value holding a broken character takes the characters its pages print there. IN PLACE.

    The value's own pages are searched first, then every page; it is repaired only where exactly one spelling fits.
    A value no text settles (the text layer holds the broken character itself, or two spellings fit) is reported as
    :data:`BROKEN_CHARACTER_LEFT` and left for a person.
    """
    notes: list[tuple[str, str]] = []
    for path, envelope in list(_envelopes(gold)):
        raw = envelope.get("raw")
        if not isinstance(raw, str) or BROKEN_CHARACTER not in raw:
            continue
        pattern = _broken_pattern(raw)
        found: set[str] = set()
        for numbers in (_pages(envelope), sorted(pages)):
            for number in numbers:
                found |= {" ".join(m.group(0).split()) for m in pattern.finditer(pages.get(number) or "")}
            if found:
                break
        if len(found) != 1:
            notes.append((BROKEN_CHARACTER_LEFT, f"{path} {raw!r}"))
            continue
        repaired = found.pop()
        if envelope.get("parsed") == raw:
            envelope["parsed"] = repaired
        envelope["raw"] = repaired
        notes.append(("broken character repaired", f"{path} {raw!r} -> {repaired!r}"))
    return notes


#: Words a title-cased value keeps lower case after its first word.
_SMALL_WORDS = frozenset({"and", "or", "of", "the", "a", "an", "in", "for", "to", "by", "with"})
#: An all-capitals word this long or shorter is an abbreviation (LLC, INC, LP) and stays as it is.
_ABBREVIATION_LETTERS = 3


def _title_case(text: str) -> str:
    """``"LIMITED LIABILITY COMPANY"`` -> ``"Limited Liability Company"``; an abbreviation (``LLC``) kept."""
    words = text.split(" ")
    out = []
    for index, word in enumerate(words):
        letters = re.sub(r"[^A-Za-z]", "", word)
        if not letters or len(letters) <= _ABBREVIATION_LETTERS and word.lower() not in _SMALL_WORDS:
            out.append(word)
        elif index and word.lower() in _SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append(word[:1] + word[1:].lower())
    return " ".join(out)


def normalise_value_shapes(gold: dict[str, Any]) -> list[tuple[str, str]]:
    """One shape per kind of ``parsed`` value (user, 2026-11-23), IN PLACE; ``raw`` stays as printed.

    * an amount or a number that is whole is written whole: ``1250.0`` -> ``1250`` (the audit found 18 fields
      written both ways - the model learns both);
    * a set value written as free text (``named_insured.entity_type``: "As printed: Individual, Corporation,
      LLC...") is written in title case when it was written in capitals: ``"CORPORATION"`` -> ``"Corporation"``.
      A value the schema holds to a list (``surplus_lines``, a state's ``NY``) is never in capitals of more than
      three letters, so it is left alone, as is a value in mixed case.
    """
    from calibration.features import infer_field_type

    notes: list[tuple[str, str]] = []
    for path, envelope in _envelopes(gold):
        parsed = envelope.get("parsed")
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float, str)):
            continue
        kind = infer_field_type(re.sub(r"\[\d+\]", "[]", path), common_model=True)
        if kind in ("money", "number") and isinstance(parsed, float) and parsed.is_integer():
            envelope["parsed"] = int(parsed)
            notes.append(("whole number written whole", f"{path} {parsed!r} -> {int(parsed)!r}"))
        elif kind == "enum" and isinstance(parsed, str) and parsed.isupper() and len(re.sub(r"[^A-Z]", "", parsed)) > 3:
            title = _title_case(parsed)
            if title != parsed:
                envelope["parsed"] = title
                notes.append(("set value in title case", f"{path} {parsed!r} -> {title!r}"))
    return notes


def apply_cycle3_rules(gold: dict[str, Any], pages: dict[int, str]) -> list[tuple[str, str]]:
    """The cycle 3 rules that need no person (R2's format, R3, R5, R8, one shape per value), IN PLACE;
    ``[(kind, detail)]``."""
    notes = strip_form_ref_editions(gold)
    notes += clear_code_descriptions(gold)
    notes += clear_premiums_without_amount(gold)
    notes += repair_broken_characters(gold, pages)
    notes += normalise_value_shapes(gold)
    return notes
