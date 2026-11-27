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


def test_the_prompt_hash_does_not_depend_on_line_endings(monkeypatch) -> None:
    """A Windows checkout (core.autocrlf) holds the prompt files with CRLF, the pod's with LF.
    Hashed as raw bytes, a release packaged on one was refused on the other: found 2026-11-25,
    when this laptop's hash for v4's own code (7.2.0) differed from v4's recorded fdcce708."""
    import pathlib

    from common.prompts import prompt_hash

    real = pathlib.Path.read_bytes

    def as_lf(self):
        return real(self).replace(b"\r\n", b"\n")

    def as_crlf(self):
        return as_lf(self).replace(b"\n", b"\r\n")

    monkeypatch.setattr(pathlib.Path, "read_bytes", as_lf)
    lf = prompt_hash()
    monkeypatch.setattr(pathlib.Path, "read_bytes", as_crlf)
    assert prompt_hash() == lf


def test_the_prompt_hash_tolerates_an_absent_meanings_file(monkeypatch) -> None:
    """coverage_code_meanings.yaml is optional (model_view reads it only when it exists).
    A checkout without it must hash a no-meanings state, not raise FileNotFoundError on every
    prompt_hash() call - corpus build, package, cold-start check (review follow-up 2026-11-27)."""
    import pathlib

    from common.model_view import COVERAGE_CODE_MEANINGS
    from common.prompts import prompt_hash

    real_exists = pathlib.Path.exists
    real_read = pathlib.Path.read_bytes
    absent = COVERAGE_CODE_MEANINGS.resolve()

    def exists(self):
        return False if self.resolve() == absent else real_exists(self)

    def read_bytes(self):
        if self.resolve() == absent:
            raise FileNotFoundError(self)                        # as a genuinely absent file would
        return real_read(self)

    monkeypatch.setattr(pathlib.Path, "exists", exists)
    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)
    assert len(prompt_hash()) == 64                              # does not raise
