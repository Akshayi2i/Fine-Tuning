"""What the field descriptions a prompt shows may quote (review P5, decision 2026-11-24).

The rule, relaxed by the user's decision: a description may show short examples of
the values and column headings its field holds ('Included', 'Pr/Co', 'Annual'), but
never a carrier's own wording and never an example of one particular line. The
common model's definitions are shared, so a description is shown to every line
that holds the field: a homeowners form number reached the general liability
prompt. Overrides live in ``configs/model_view.yaml``; the client's schema is not
edited.

Carrier names cannot be listed exhaustively here (the documents are not in the
repository): :data:`REMOVED_CARRIER_WORDING` holds what was taken out, so it cannot
come back, and the line checks below catch the rest of the examples that named a line.
"""

from __future__ import annotations

import re

import pytest

from common.schema_sections import groups_for
from common.schemas import resolved_schema
from testing.render_prompts import REFERENCE_WINDOW_LINES

#: Carrier wording removed on 2026-11-24 (carrier.claims_notice_recipient).
REMOVED_CARRIER_WORDING = ("Navigators",)

#: Words that name a line, and the form-number prefix of its own forms.
LINE_WORDS: dict[str, tuple[str, ...]] = {
    "homeowners": ("Homeowners",), "dwelling_fire": ("Dwelling",), "personal_auto": ("Personal Auto", "Auto Special"),
    "motorcycle": ("Motorcycle",), "recreational_vehicle": ("Recreational",), "ocean_marine": ("Ocean Marine",),
    "personal_umbrella": ("Umbrella",), "gl": ("General Liability",), "wc": ("Workers", "workers' compensation"),
    "boppr": ("Businessowners",), "bopgl": ("Businessowners",),
}
FORM_PREFIX: dict[str, str] = {"homeowners": "HO", "dwelling_fire": "DP", "personal_auto": "PP", "wc": "WC",
                               "gl": "CG", "boppr": "BP", "bopgl": "BP"}

#: Quoted examples that name a line's word but are a value the field holds, and why.
ALLOWED_EXAMPLES: dict[str, str] = {
    # Vehicle.rating_factors: the client's description quotes it as a rating factor printed
    # for a vehicle ("Covered by Workers' Compensation"). Cut at its apostrophe by the quote rule.
    "Covered by Workers": "a vehicle rating factor printed on auto policies, not a workers' compensation example",
}

_QUOTED = re.compile(r"'([^']{2,80})'")


def _descriptions(lob: str) -> set[str]:
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("description"), str):
                found.add(node["description"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for group in groups_for(lob):
        walk(resolved_schema("policy", None, lob, group))
    return found


def other_line_examples(lob: str, descriptions: set[str]) -> list[str]:
    """Quoted examples in ``lob``'s descriptions that name, or use the form prefix of, another line."""
    own_words = set(LINE_WORDS.get(lob, ()))
    own_prefix = FORM_PREFIX.get(lob)
    problems = []
    for text in sorted(descriptions):
        for example in _QUOTED.findall(text):
            if example in ALLOWED_EXAMPLES:
                continue
            for other, words in LINE_WORDS.items():
                if other == lob:
                    continue
                named = [w for w in words if w not in own_words and w.lower() in example.lower()]
                prefix = FORM_PREFIX.get(other)
                form = prefix and prefix != own_prefix and re.search(rf"\b{prefix} ?\d", example)
                if named or form:
                    problems.append(f"{lob}: '{example}' is a {other} example ({text[:80]})")
    return problems


@pytest.mark.parametrize("lob", REFERENCE_WINDOW_LINES)
def test_no_description_quotes_another_lines_example(lob: str) -> None:
    assert other_line_examples(lob, _descriptions(lob)) == []


@pytest.mark.parametrize("lob", REFERENCE_WINDOW_LINES)
def test_no_description_carries_a_carriers_wording(lob: str) -> None:
    hits = [text for text in _descriptions(lob) for word in REMOVED_CARRIER_WORDING if word in text]
    assert hits == []


def test_another_lines_example_is_seen() -> None:
    """Seen to fail on purpose: the descriptions the gl prompt showed before 2026-11-24."""
    old = {"Form number as printed, e.g. 'HO 00 03'.",
           "Document title as printed, e.g. 'Homeowners Policy Declarations'."}
    assert len(other_line_examples("gl", old)) == 2
    assert other_line_examples("homeowners", {"Form number as printed, e.g. 'HO 00 03'."}) == []
