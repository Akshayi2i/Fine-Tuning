"""The reference prompts cover what is trained and served (review P7, 2026-11-24)."""

from __future__ import annotations

from common.schema_sections import groups_for
from testing.render_prompts import (
    PROMPTS_DIR,
    REFERENCE_MODES,
    REFERENCE_WINDOW_LINES,
    WHOLE_POLICY_NOTE,
    prompt_path,
)

#: The lines cycle 2 trained on: before P7 none had a reference file.
TRAINED_IN_CYCLE_2 = ("gl", "bopgl", "boppr", "wc")


def test_every_trained_line_has_every_window_prompt_in_both_modes() -> None:
    assert set(TRAINED_IN_CYCLE_2) <= set(REFERENCE_WINDOW_LINES)
    assert "image_only" in REFERENCE_MODES                   # the mode production reads first
    missing = [prompt_path("policy", mode, None, PROMPTS_DIR, lob, group).name
               for lob in REFERENCE_WINDOW_LINES for group in groups_for(lob) for mode in REFERENCE_MODES
               if not prompt_path("policy", mode, None, PROMPTS_DIR, lob, group).exists()]
    assert not missing, missing


def test_no_whole_schema_prompt_of_a_line_is_kept() -> None:
    """Serving reads every policy in windows: a line's whole-schema prompt is a
    request it never sends. Only the bare policy file stays, marked so."""
    whole = sorted(path.name for path in PROMPTS_DIR.glob("policy*.prompt.txt")
                   if not any(f".{group}." in path.name for group in ("decl", "arrays", "forms", "lineblk")))
    assert whole == ["policy.prompt.txt"]
    assert WHOLE_POLICY_NOTE in (PROMPTS_DIR / "policy.prompt.txt").read_text(encoding="utf-8")


def test_the_reference_files_are_not_prompt_input() -> None:
    """Regenerating them never changes the prompt hash: they are rendered from
    the prompt inputs, not read by the prompt."""
    from common.prompts import prompt_input_files

    assert not any("testing/prompts" in path.as_posix() for path in prompt_input_files())
