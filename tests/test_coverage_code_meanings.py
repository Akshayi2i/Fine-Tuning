"""Coverage codes shown by their meaning, once a person has written it (review P2, 2026-11-24).

The prompt tells the model to choose a code "by meaning, not by the printed name",
then showed each code with one carrier's printed name. ``configs/coverage_code_meanings.yaml``
holds a meaning per code, to be written by someone who knows the coverages - never
generated. A filled meaning (and its ``tell_apart``) is what the model sees; an
empty one falls back to the name, as before.
"""

from __future__ import annotations

import inspect

import pytest

import common.model_view as model_view
import common.prompts as prompts
import common.schemas as schemas
from common.prompts import compact_fields_for_prompt, prompt_input_files

#: The lines cycle 2 trained on: theirs are the meanings to write first.
TRAINED_LINES = ("gl", "bopgl", "boppr", "wc")


def _clear_caches() -> None:
    for module in (schemas, prompts, model_view):
        for _name, value in inspect.getmembers(module):
            if callable(getattr(value, "cache_clear", None)):
                value.cache_clear()


@pytest.fixture
def meanings(monkeypatch):
    """A scratch copy of the meanings, in place of the file, caches cleared on both sides."""
    copy = {"shared": {}, "gl": {}}
    monkeypatch.setattr(model_view, "coverage_code_meanings", lambda: copy)
    _clear_caches()
    yield copy
    monkeypatch.undo()
    _clear_caches()


def _codes_text(lob: str) -> str:
    return compact_fields_for_prompt("policy", None, lob, "arrays")["codes"]


def test_a_filled_meaning_is_shown_and_an_empty_one_falls_back_to_the_name(meanings) -> None:
    names = schemas.load_schema("policy", lob="gl")["fideon:coverage_code_names"]
    code = next(code for code in names if code.startswith("GL_"))
    assert f"{code} = {names[code][:20]}" in _codes_text("gl")              # empty: the name, as before
    meanings["gl"][code] = {"meaning": "Liability for injury to others.", "tell_apart": "Not the medical payments."}
    _clear_caches()
    text = _codes_text("gl")
    assert f"{code} = Liability for injury to others. Not the medical payments." in text
    assert f"{code} = {names[code][:20]}" not in text


def test_a_shared_codes_meaning_serves_every_line_that_uses_it(meanings) -> None:
    meanings["shared"]["X_MED_PAY"] = {"meaning": "Medical expenses of others, regardless of fault."}
    _clear_caches()
    for lob in ("gl", "boppr"):
        assert "X_MED_PAY = Medical expenses of others, regardless of fault." in _codes_text(lob)


def test_the_meanings_file_is_prompt_input() -> None:
    """Writing a meaning changes what the model is shown, so it changes the prompt hash."""
    assert model_view.COVERAGE_CODE_MEANINGS in prompt_input_files()


def test_every_trained_lines_code_is_listed_in_the_meanings_file() -> None:
    listed = model_view.coverage_code_meanings()
    for lob in TRAINED_LINES:
        own = set(listed.get(lob) or {})
        shared = set(listed.get("shared") or {})
        codes = set(schemas.load_schema("policy", lob=lob)["fideon:coverage_code_names"])
        assert codes <= own | shared, sorted(codes - own - shared)


@pytest.mark.xfail(strict=True, reason="the meanings are written by a coverage expert and not delivered yet "
                                       "(review P2, decision 2026-11-24): remove this mark once they are")
@pytest.mark.parametrize("lob", TRAINED_LINES)
def test_every_code_a_trained_line_shows_has_a_meaning(lob: str) -> None:
    codes = schemas.load_schema("policy", lob=lob)["fideon:coverage_code_names"]
    empty = sorted(code for code in codes if model_view.code_meaning(code, lob) is None)
    assert not empty, f"{lob}: {len(empty)} code(s) still shown by their printed name"
