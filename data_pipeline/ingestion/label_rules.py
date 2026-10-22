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
  footer, a "Named Insured: ..." line - not where its words fall in a sentence
  ("as shown in the Policy Declarations").
* **extra field repeats a field** - folded into that field, as above. Only a
  value printed as itself (an identifier, a name, a large amount) that one
  field holds: "$0.00", "$500" or "2015" in two places is not one value. A
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
import re
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

from common import grounding

#: The correction kind that lists a value printed nowhere.
PRINTED_NOWHERE = "printed nowhere"

#: Fields the pipeline fills from the file, never printed values.
_SYSTEM = {"document.page_count", "document.source_file_name", "document.doc_type",
           "document.modality", "policy.is_package"}

#: A page settles whether a value is printed only when it prints at least this
#: share of the figures its label cites on it.
RELIABLE_PAGE_SHARE = 0.9

_PAGE_COUNTER = re.compile(r"^page \d+( of \d+)?$")
#: A label written about where a value was found, not printed before it: a figure
#: of a form's standard wording ("Form boilerplate amount (context: ...)", "Bail
#: Bond Cap (CG 00 01 boilerplate)", "Form wording, page 15: ...", "Money printed
#: on page 4") or text the page does not show ("Invisible text-layer remnant of
#: ..."). Policy wording is never a value, and no image shows the text a PDF
#: hides: taught, both are values to invent.
_DRAFTING_LABEL = re.compile(r"boilerplate|\(context:|text-layer|invisible text|^form wording|money printed on page",
                             re.IGNORECASE)
#: A clause of policy wording filed as a label: a list marker and then a sentence
#: ("b. Up to", "a. The act resulted in insured losses in excess of", "2. We do
#: not cover"). A numbered declarations caption ("2. Policy Period From") is
#: not one: what follows its marker is a caption, not a sentence.
_CLAUSE_LABEL = re.compile(r"^\(?[a-zA-Z0-9]{1,2}\)?\.\s+(?:[a-z]|Up to\b|We\b|The act\b|All expenses\b"
                           r"|Premium shown\b|The total of\b|The$)")
#: Figures of standard wording and notices: supplementary-payment caps, the
#: terrorism act's thresholds and shares, claims and service lines. A label
#: naming a premium ("Terrorism Premium") is a declarations value, not wording.
_WORDING_FIGURE = re.compile(
    r"\bbail bonds?\b|loss of earnings|\bearnings up to\b|terrorism risk insurance act|\btria\b|federal share"
    r"|program trigger|in excess of|act resulted|terrorist acts certified|toll[- ]?free|www\.|https?://|\bthreshold\b",
    re.IGNORECASE)
#: A form's number, printed in the header or footer of every page of the form.
_FORM_NUMBER = re.compile(r"^forms_and_endorsements\[\d+\]\.form_number$")
#: Lines at each end of a page that are its header and footer.
_EDGE_LINES = 4
_YEAR = re.compile(r"(?:19|20)\d\d")
#: Extra-field labels for a value that has a field of its own.
_PROTECTION_CLASS = {"protection class", "prot class", "fire protection class", "protection class code",
                     "public protection class", "ppc"}
_PAYOR_LABELS = ("paid by", "payor", "bill to", "billed to", "premium paid by")
#: Extra-field labels (normalised) for a value with a field of its own on every
#: common-model line: ``(block, field, labels)``. Moved only into an empty field,
#: and only when one entry of the label names it.
_FIELD_LABELS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("named_insured", "business_description", frozenset({
        "business description", "description of business", "description of operations", "operations description",
        "nature of business", "business operations", "description of insured s business", "insured s business"})),
    ("named_insured", "entity_type", frozenset({
        "form of business", "business type", "type of business", "entity type", "legal entity", "legal entity type",
        "business entity", "type of entity", "form of organization"})),
    ("billing", "bill_type", frozenset({"billing type", "bill type", "billing method", "type of billing"})),
    ("countersignature", "representative_name", frozenset({"authorized representative", "authorised representative"})),
)
#: A claims line: a label naming a phone for claims, and a value with a number.
_CLAIMS_PHONE = re.compile(r"\bclaims?\b.*\b(phone|call line|telephone|hotline)\b|^report (a )?claims?\b")
#: The premium kept on cancellation: an amount, or a percent of the premium.
_MINIMUM_EARNED = re.compile(r"^min(imum)? earned( premium)?( percent(age)?)?$")

#: How a location's or a building's own number is introduced on a page.
_NUMBER_WORDS = {
    "location_number": r"loc(?:ation)?|prem(?:ises)?|property|prop|risk|site",
    "building_number": r"bldg|bld|building|struct(?:ure)?|dwelling|dwg",
}


def page_texts(pdf: Path) -> dict[int, str]:
    """Each page's text layer by page number; empty when the file has none or
    cannot be read - then only the rules that need no text apply."""
    if not pdf.is_file():
        return {}
    try:
        import pymupdf

        with pymupdf.open(pdf) as doc:
            return {number: page.get_text() for number, page in enumerate(doc, start=1)}
    except Exception:  # noqa: BLE001 - an unreadable PDF only means no text rules
        return {}


def scanned_pages(pdf: Path) -> set[int]:
    """The pages whose text is an OCR reading laid over a scan: most of their
    characters drawn invisibly (PDF text render mode 3), under the image."""
    if not pdf.is_file():
        return set()
    try:
        import pymupdf

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


def apply_label_rules(gold: dict[str, Any], pages: dict[int, str], *,
                      scanned: Collection[int] = ()) -> list[tuple[str, str]]:
    """Apply the rules to ``gold`` IN PLACE; ``[(kind, detail)]``, one per change.

    ``scanned`` are the pages whose text is OCR over a scan (:func:`scanned_pages`):
    read for where a value is printed, never to decide one is not. A value
    printed nowhere is reported with kind :data:`PRINTED_NOWHERE` and its path
    as the detail.
    """
    texts = {number: grounding.PageText(text) for number, text in pages.items()}
    if not any(text.words for text in texts.values()):
        texts = {}
    reliable = _reliable_pages(gold, texts) - set(scanned) if texts else set()
    notes: list[tuple[str, str]] = split_form_editions(gold)
    _building_links(gold, notes)
    _extra_fields(gold, texts, reliable, notes)
    if texts:
        _numbers(gold, pages, reliable, notes)
        _page_lists(gold, pages, texts, reliable, notes)
        notes += [(PRINTED_NOWHERE, path) for path in printed_nowhere(gold, texts, reliable)]
    return notes


# --------------------------------------------------------------------------
# Form editions
# --------------------------------------------------------------------------

#: An edition printed as part of a form number, in shapes no form number takes:
#: ISO's number then month and year ("CG 00 01 04 13") ...
_ISO_EDITION = re.compile(r"^(?P<form>[A-Z]{2} [A-Z0-9]{2} \d{2}) (?P<mm>0[1-9]|1[0-2]) (?P<yy>\d{2})$")
#: ... or a month and year in brackets after the number ("IL0017(11-98)").
_BRACKETED_EDITION = re.compile(
    r"^(?P<form>.*?\S)\s*\(\s*(?P<mm>0[1-9]|1[0-2])\s*[-/ ]\s*(?P<yy>\d{4}|\d{2})\s*\)$")
#: A number ending in what may be a month and year, or may be its own groups
#: ("CG 20 10 04"): named for a person to look at, never changed.
_MAYBE_EDITION = re.compile(r"^(?P<form>.*\d.*?)[ /-](?P<mm>0[1-9]|1[0-2])[ /-](?P<yy>\d{2})$")


def split_form_edition(text: Any) -> tuple[str, str, str] | None:
    """``(form number, edition as printed, edition parsed MM/YY)`` when ``text``
    carries its edition in a shape no form number takes; else ``None``."""
    value = " ".join(str(text or "").split())
    if match := _ISO_EDITION.match(value):
        return match["form"], f"{match['mm']} {match['yy']}", f"{match['mm']}/{match['yy']}"
    if (match := _BRACKETED_EDITION.match(value)) and _form_code(match["form"]):
        printed = value[value.rindex("(") + 1:].rstrip(")").strip()
        return match["form"].strip(), printed, f"{match['mm']}/{match['yy'][-2:]}"
    return None


def may_hold_edition(text: Any) -> bool:
    """A form number that may end in its edition, in a shape a number takes too."""
    value = " ".join(str(text or "").split())
    return split_form_edition(value) is None and bool(_MAYBE_EDITION.match(value))


def edition_rows(gold: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(index, form row)`` for each form whose ``edition_date`` is empty."""
    from common.canonical import is_field_value

    for index, row in enumerate(gold.get("forms_and_endorsements") or []):
        if not isinstance(row, dict) or not is_field_value(row.get("form_number")):
            continue
        edition = row.get("edition_date")
        if edition is None or (isinstance(edition, dict) and edition.get("raw") in (None, "")):
            yield index, row


def split_form_editions(gold: dict[str, Any]) -> list[tuple[str, str]]:
    """Move each edition a form number carries to its form's ``edition_date``.

    IN PLACE; one note per form changed. Only where ``edition_date`` is empty:
    a filled one is the label's own reading and is never overwritten. The
    edition takes the number's pages and confidence: it is printed with it.
    """
    notes: list[tuple[str, str]] = []
    for index, row in edition_rows(gold):
        number = row["form_number"]
        split = split_form_edition(number.get("raw"))
        if split is None:
            continue
        form, printed, parsed = split
        before = number.get("raw")
        number["raw"] = form
        if isinstance(number.get("parsed"), str):
            number["parsed"] = form
        edition: dict[str, Any] = {"raw": printed, "parsed": parsed}
        if "confidence" in number:
            edition["confidence"] = copy.deepcopy(number["confidence"])
        edition["page_ref"] = list(number.get("page_ref") or [])
        if "flagged" in number:
            edition["flagged"] = number["flagged"]
        row["edition_date"] = edition
        notes.append(("form edition moved",
                      f"forms_and_endorsements[{index}].form_number {before!r} -> {form!r}, "
                      f"edition {printed!r}"))
    return notes


def label_rules_apply(doc_type: str | None, lob: Any) -> bool:
    """Whether the label rules - the form-edition rule among them - hold for a
    gold of ``doc_type`` on line ``lob``: a policy on a common-model line (no
    line is the fallback, which is one).

    The one test of where the rules hold. They are the lines prepare_bundles
    corrects a gold on (``prepare_bundles.corrected_gold``) and serving holds an
    answer to them on (``serving.conventions.conform``); the importers' check,
    the corpus build, the gate and the harness read a gold under them there and
    nowhere else. A self-contained line (auto, wc, umbrella) keeps an edition
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


def _envelopes(node: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every value envelope with its path (``coverages[3].limits[0].amount``)."""
    from common.canonical import is_field_value

    if is_field_value(node):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            if not str(key).startswith("fideon:"):
                yield from _envelopes(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _envelopes(item, f"{path}[{index}]")


def _pages(envelope: dict[str, Any]) -> list[int]:
    return sorted({int(p) for p in envelope.get("page_ref") or [] if str(p).isdigit()})


def _printed(envelope: dict[str, Any]) -> Any:
    raw = envelope.get("raw")
    return raw if raw not in (None, "") else None


def _has_digits(text: Any) -> bool:
    return bool(re.search(r"\d", str(text)))


def _distinctive(text: Any) -> bool:
    """A value printed as itself and nothing else: an identifier, a name or
    address, a date, an amount with four or more significant figures - not
    "Yes", "1", "$0.00", "$500" or a year, which many fields can hold (a boat
    and its trailer are both "2015")."""
    value = grounding.normalise(text)
    digits = re.sub(r"\D", "", value)
    letters = re.sub(r"[^a-z]", "", value)
    if digits and letters:
        return len(value) >= 6
    if digits:
        return len(digits.strip("0")) >= 4 and not _YEAR.fullmatch(value)
    return len(value) >= 10


def _reliable_pages(gold: dict[str, Any], texts: dict[int, grounding.PageText]) -> set[int]:
    """The pages whose text settles whether a value is printed: a page printing
    nearly every figure (amount, date, number, identifier) its label cites on
    it - nine in ten, or all but one of at least four - or, citing none, a page
    of text. A page whose text misses more (a table printed as an image,
    amounts in a font that records no characters) does not, and nothing is
    taken out on its word. Words are not counted: such a page's text can still
    hold every name and caption on it."""
    cited: dict[int, list[bool]] = {}
    for path, envelope in _envelopes(gold):
        printed = _printed(envelope)
        if path in _SYSTEM or not grounding.checkable(printed) or not _has_digits(printed):
            continue
        for page in _pages(envelope):
            if page in texts:
                cited.setdefault(page, []).append(grounding.on_page(printed, texts[page]))
    return {page for page, text in texts.items() if text.words and (
        _settles(cited[page]) if cited.get(page) else len(text.words) >= 50)}


def _settles(found: list[bool]) -> bool:
    hits = sum(found)
    return hits / len(found) >= RELIABLE_PAGE_SHARE or (len(found) - hits == 1 and hits >= 3)


def printed_nowhere(gold: dict[str, Any], texts: dict[int, grounding.PageText], reliable: set[int]) -> list[str]:
    """Paths of field values with figures (amounts, dates, numbers, identifiers)
    that no page prints, judged on reliable pages only. A name or a phrase is
    never listed: it may be printed only in a logo or reworded."""
    out = []
    for path, envelope in _envelopes(gold):
        printed = _printed(envelope)
        if path in _SYSTEM or path.startswith("additional_fields") or not _has_digits(printed):
            continue
        cited = _pages(envelope)
        if not set(cited or texts) <= reliable:
            continue
        if grounding.ground(printed, cited, texts).status == grounding.NOT_PRINTED:
            out.append(path)
    return out


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


def _page_lists(gold: dict[str, Any], pages: dict[int, str], texts: dict[int, grounding.PageText],
                reliable: set[int], notes: list) -> None:
    for path, envelope in _envelopes(gold):
        printed = _printed(envelope)
        if path in _SYSTEM or printed is None:
            continue
        cited = _pages(envelope)
        where = grounding.ground(printed, cited, texts)
        if not where.printed:
            continue
        found = set(where.pages)
        figures = _has_digits(printed)
        if not figures:
            # A name or a phrase is on a page it does not cite only where it
            # heads a line; in a sentence its words are the text's, not the value.
            value = grounding.normalise(printed)
            found = {page for page in found if page in cited or _heads_a_line(value, pages.get(page) or "")}
        # Once in its field, every page that prints it - for a value printed as
        # itself at the document's level (a header's policy number). In a table
        # the same amount on another page may be another row's.
        every_page = "[" not in path and _distinctive(printed)
        if where.status == grounding.ON_CITED_PAGE:
            new = found if every_page or not cited else set(cited) & found
        else:
            new = found if every_page or (len(found) == 1 and _distinctive(printed)) else set(cited)
        # A cited page whose text cannot settle it (no text, or figures of its
        # label missing) keeps its place; so does every page a name or a phrase
        # cites - it may be printed in a logo or an image the text does not hold.
        new |= {page for page in cited if page not in reliable or not figures}
        if _FORM_NUMBER.match(path) and _form_code(printed):
            # A form prints its number at the head or foot of each of its pages;
            # a window over its fourth page shows the form as surely as one over
            # the forms schedule does. Cited there only, the window was taught to
            # leave out a form it can see.
            value = grounding.normalise(printed)
            new |= {page for page, text in pages.items() if _on_page_edge(value, text or "")}
        new_pages = sorted(new)
        if new_pages and new_pages != cited:
            envelope["page_ref"] = new_pages
            notes.append(("page list", f"{path} {cited} -> {new_pages}"))


def _form_code(text: Any) -> bool:
    """Whether a form number is code enough to be taken for itself at a page's
    edge: a carrier's own short codes (``UFR 1``, ``L-783``, ``XCNTR``) are, though
    too short to be :func:`_distinctive` in a page's body; a short word
    (``PRIV``, ``ACD``) is not."""
    value = grounding.normalise(text)
    compact = value.replace(" ", "")
    return _distinctive(text) or (len(compact) >= 4 and any(c.isdigit() for c in compact)) or len(compact) >= 5


def _on_page_edge(value: str, text: str) -> bool:
    """Whether a normalised ``value`` stands as words in one of the first or last
    :data:`_EDGE_LINES` lines of ``text`` - the page's header or footer, never
    its body, where an endorsement may name another form it amends."""
    lines = [line for line in text.splitlines() if line.strip()]
    edge = lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]
    return any(f" {value} " in f" {grounding.normalise(line)} " for line in edge)


def _heads_a_line(value: str, text: str) -> bool:
    """Whether a normalised ``value`` begins a line of ``text``, or the part of
    a line after its label ("Named Insured: Clemence Fairbank")."""
    for line in text.splitlines():
        for part in (line, line.partition(":")[2]):
            words = grounding.normalise(part)
            if words == value or words.startswith(f"{value} "):
                return True
    return False


def _extra_fields(gold: dict[str, Any], texts: dict[int, grounding.PageText], reliable: set[int],
                  notes: list) -> None:
    entries = gold.get("additional_fields")
    if not isinstance(entries, list) or not entries:
        return
    core: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for path, envelope in _envelopes({k: v for k, v in gold.items() if k != "additional_fields"}):
        printed = _printed(envelope)
        if printed is not None and path not in _SYSTEM and _distinctive(printed):
            core.setdefault(grounding.normalise(printed), []).append((path, envelope))

    kept: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    moves = _field_moves(gold, entries)
    for index, entry in enumerate(entries):
        where = f"additional_fields[{index}]"
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "")
        value = entry.get("value") if isinstance(entry.get("value"), dict) else {}
        printed = _printed(value) if value else None
        label_n, value_n = grounding.normalise(label), grounding.normalise(printed or "")
        if _is_wording(label):
            notes.append(("extra field from wording", f"{where} {label[:60]!r}"))
            continue
        if index in moves:
            block, name = moves[index]
            gold.setdefault(block, {})[name] = _moved_value(name, value)
            notes.append(("extra field moved to its field", f"{where} {label!r} -> {block}.{name}"))
            continue
        if _not_a_value(label_n, value_n):
            notes.append(("extra field not a value", f"{where} {label!r}"))
            continue
        if _move_to_field(gold, label_n, value, notes, where):
            continue
        mates = core.get(value_n) if printed is not None and _distinctive(printed) else None
        if mates and len(mates) == 1:
            # One field holds this very value: the entry repeats it. Several do -
            # the value is not that field's alone - and the entry stays. A table
            # row keeps to the pages that list it: a coverage named again in an
            # endorsement's "modifies" line, a driver named in the vehicle table,
            # would be taught as a row of that page.
            path, envelope = mates[0]
            if "[" not in path:
                envelope["page_ref"] = sorted(set(_pages(envelope)) | set(_pages(value)))
            notes.append(("extra field repeats a field", f"{where} {label!r} -> {path}"))
            continue
        pair = (label_n, value_n)
        if pair in by_pair:
            first = by_pair[pair].get("value")
            if isinstance(first, dict):
                first["page_ref"] = sorted(set(_pages(first)) | set(_pages(value)))
            notes.append(("extra field merged", f"{where} {label!r}"))
            continue
        by_pair[pair] = entry
        kept.append(entry)
    if len(kept) != len(entries):
        gold["additional_fields"] = kept
    # Figures no page prints: listed as a field value is, on every page the
    # entry was merged from.
    for index, entry in enumerate(kept):
        value = entry.get("value") if isinstance(entry.get("value"), dict) else {}
        printed = _printed(value) if value else None
        cited = _pages(value)
        if (texts and _has_digits(printed) and set(cited or texts) <= reliable
                and grounding.ground(printed, cited, texts).status == grounding.NOT_PRINTED):
            notes.append((PRINTED_NOWHERE, f"additional_fields[{index}].value"))


def _not_a_value(label: str, value: str) -> bool:
    """A page counter, or a run of prose filed as a label - not a long label
    printed before an amount ("Your 12-month policy premium excluding billing
    fees and payment option discounts is $352.00")."""
    if _PAGE_COUNTER.match(label) or _PAGE_COUNTER.match(value):
        return True
    return (len(label) > 80 and not _has_digits(value)) or (label == value and len(label) > 40)


def _is_wording(label: str) -> bool:
    """Whether an extra field's label is policy wording or a drafting note, not a
    printed caption: a note on where the value was found, a clause, a figure of
    standard wording, or a sentence picked up mid-way - a label starting with a
    lowercase letter ("for cost of bail bonds required", "policy premium or");
    a camel-case name ("eBill") is a caption."""
    if _DRAFTING_LABEL.search(label) or _CLAUSE_LABEL.search(label):
        return True
    if _WORDING_FIGURE.search(label) and "premium" not in label.lower():
        return True
    return bool(label[:1].islower() and not (len(label) > 1 and label[1].isupper()))


def _field_moves(gold: dict[str, Any], entries: list[Any]) -> dict[int, tuple[str, str]]:
    """``{entry index: (block, field)}``: extra fields holding a value that has a
    field of its own, where that field is empty and no other entry names it.
    Two entries naming one field (a percent and an amount of the same name, two
    business descriptions) are left for a person to place. A block the gold
    does not have is created only where every common-model line has it
    (``premium``, ``billing``); a line block (``countersignature``) is filled
    only where the gold has it, so a line without one never gains it."""
    claims: dict[tuple[str, str], list[int]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("value"), dict):
            continue
        label = str(entry.get("label") or "")
        if _is_wording(label):
            continue
        label_n = grounding.normalise(label)
        printed = _printed(entry["value"]) or ""
        target = next(((block, name) for block, name, labels in _FIELD_LABELS if label_n in labels), None)
        if target is None and _CLAIMS_PHONE.search(label_n) and len(re.sub(r"\D", "", printed)) >= 7:
            target = ("carrier", "claims_phone")
        if target is None and _MINIMUM_EARNED.match(label_n) and _number(printed) is not None:
            percent = "%" in printed or "%" in label or "percent" in label_n
            target = ("premium", "minimum_earned_percent" if percent else "minimum_earned")
        if target is not None:
            claims.setdefault(target, []).append(index)
    moves = {}
    for (block, name), indices in claims.items():
        holder = gold.get(block)
        if holder is None and block not in ("premium", "billing"):
            continue
        current = holder.get(name) if isinstance(holder, dict) else None
        if len(indices) == 1 and (holder is None or isinstance(holder, dict)) and not (
                isinstance(current, dict) and current.get("raw") not in (None, "")):
            moves[indices[0]] = (block, name)
    return moves


def _number(text: str) -> float | None:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")
    return float(match.group().replace(",", "")) if match else None


def _moved_value(name: str, value: dict[str, Any]) -> dict[str, Any]:
    """The extra field's value as its field holds it: a number for the minimum
    earned premium (an amount, or a percent of the premium), as printed else."""
    moved = dict(value)
    if name in ("minimum_earned", "minimum_earned_percent"):
        moved["parsed"] = _number(_printed(value) or "")
    return moved


def _move_to_field(gold: dict[str, Any], label: str, value: dict[str, Any], notes: list, where: str) -> bool:
    """Move an extra field into the proper field it is (Protection Class, a
    mortgagee paying the premium); True when the entry is consumed."""
    if not value:
        return False
    if label in _PROTECTION_CLASS:
        locations = [row for row in gold.get("locations") or [] if isinstance(row, dict)]
        if len(locations) != 1:
            return False
        current = locations[0].get("protection_class")
        if isinstance(current, dict) and current.get("raw") not in (None, ""):
            notes.append(("extra field moved to its field", f"{where} {label!r}: already in locations[0]"))
            return True
        locations[0]["protection_class"] = dict(value)
        notes.append(("extra field moved to its field", f"{where} {label!r} -> locations[0].protection_class"))
        return True
    if label.startswith(_PAYOR_LABELS) and "mortgagee" in grounding.normalise(value.get("raw") or ""):
        parties = [row for row in gold.get("interested_parties") or []
                   if isinstance(row, dict) and str(row.get("role") or "").casefold() == "mortgagee"]
        if len(parties) != 1:
            return False
        current = parties[0].get("is_payor")
        if isinstance(current, dict) and current.get("parsed") is True:
            notes.append(("extra field moved to its field", f"{where} {label!r}: already in is_payor"))
            return True
        parties[0]["is_payor"] = {**value, "parsed": True}
        notes.append(("extra field moved to its field", f"{where} {label!r} -> interested_parties.is_payor"))
        return True
    return False


def _building_links(gold: dict[str, Any], notes: list) -> None:
    """Every ``applies_to`` link to a location with exactly one building
    rewritten to that building. A building with no ``location_ref`` belongs to
    the document's only location, when there is one."""
    locations = [row["unit_id"] for row in gold.get("locations") or [] if isinstance(row, dict) and row.get("unit_id")]
    homes: dict[str, list[str]] = {}
    for row in gold.get("buildings") or []:
        if not (isinstance(row, dict) and row.get("unit_id")):
            continue
        home = row.get("location_ref") or (locations[0] if len(locations) == 1 else None)
        if isinstance(home, str):
            homes.setdefault(home, []).append(row["unit_id"])
    only = {location: ids[0] for location, ids in homes.items() if len(ids) == 1}
    if not only:
        return

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
        elif isinstance(node, dict):
            links = node.get("applies_to")
            if isinstance(links, list) and any(link in only for link in links):
                rewritten = list(dict.fromkeys(only.get(link, link) for link in links))
                node["applies_to"] = rewritten
                notes.append(("link names the building", f"{path}.applies_to {links} -> {rewritten}"))
            for key, value in node.items():
                if key not in ("locations", "buildings", "applies_to"):
                    walk(value, f"{path}.{key}" if path else key)

    walk(gold, "")


def _numbers(gold: dict[str, Any], pages: dict[int, str], reliable: set[int], notes: list) -> None:
    """A location or building number only where a page prints it as one. Any
    page: a label's page for a number is often the summary that lists the
    location, while its own page says "Location 1 of 1"."""
    for table, field_name in (("locations", "location_number"), ("buildings", "building_number")):
        for index, row in enumerate(gold.get(table) or []):
            envelope = row.get(field_name) if isinstance(row, dict) else None
            if not isinstance(envelope, dict):
                continue
            number = _whole_number(envelope.get("parsed"), envelope.get("raw"))
            if number is None:
                continue
            if not set(_pages(envelope) or pages) <= reliable:
                continue                    # a page that cannot settle it
            street = ""
            address = row.get("address") if isinstance(row.get("address"), dict) else {}
            if isinstance(address.get("street"), dict):
                street = str(address["street"].get("raw") or "")
            if not any(_prints_number(text, _NUMBER_WORDS[field_name], number, street) for text in pages.values()):
                del row[field_name]
                notes.append(("number not printed", f"{table}[{index}].{field_name} {envelope.get('raw')!r}"))


def _whole_number(parsed: Any, raw: Any) -> int | None:
    for value in (parsed, raw):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and float(value).is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _prints_number(text: str, words: str, number: int, street: str) -> bool:
    """Whether ``text`` prints ``number`` as a location/building number:
    "Location 1", "Loc #: 001", "Property: 1 of 1", "Bldg 2" - not the house
    number of ``street`` ("Location: 2 TOWN RD" is an address) - or a table
    whose column or label names the number ("Loc. #", "Location Number:") with
    the number in a cell of its own: a text layer gives a table's cells as
    lines, the header's far from its value."""
    after_street = grounding.normalise(street).split()[1:2]
    pattern = re.compile(rf"\b(?:{words})\b[^0-9\n]{{0,15}}\n?[^0-9\n]{{0,15}}\b0*{number}\b(?P<rest>[^\n]{{0,40}})",
                         re.IGNORECASE)
    for match in pattern.finditer(text):
        rest = grounding.normalise(match.group("rest")).split()
        if after_street and rest[:1] == after_street:
            continue          # the number begins the street address
        return True
    header = re.compile(rf"\b(?:{words})\b\.?[ \t]*(?:#|no\b|num(?:ber)?\b)", re.IGNORECASE)
    return bool(header.search(text) and re.search(rf"(?m)^[ \t]*0*{number}[ \t]*$", text))
