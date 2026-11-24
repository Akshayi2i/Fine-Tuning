"""Every test a document names exists (review R6, 24 November 2026).

The specs, the README, the prompt templates' comments and the modules'
docstrings name tests as proof ("``test_prompt_parity`` asserts ..."). A name
that no longer exists is a claim with nothing behind it: the review found
three. This collects every ``tests/test_*.py`` path and every ``test_*`` name
those texts mention and fails on one that ``tests/`` does not define.

Names the texts use for something other than a test are listed in
:data:`NOT_TESTS`, each with why.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"

#: Words of the form ``test_*`` that are not test names, and why.
NOT_TESTS: dict[str, str] = {
    "test_data": "the folder testing/test_data/",
    "test_record": "the file test_record.csv",
    "test_examples": "a count in a corpus view (val_examples, test_examples)",
}

#: Stale names that cannot be fixed yet, and why. The prompt templates' bytes,
#: comments included, are hashed into ``prompt_hash()``: correcting a comment
#: there is a new prompt version, done with the cycle 3 prompt changes (review
#: Phase 3, P1-P5). :func:`test_held_names_are_still_stale` fails once they are
#: fixed, so this list cannot outlive them.
HELD_FOR_PROMPT_VERSION: dict[str, set[str]] = {
    "prompts/system_prompt_template.jinja": {"test_prompt_parity",
                                             "test_prompt_carries_descriptions_but_no_alias_strings"},
}

_PATH = re.compile(r"tests/(test_[A-Za-z0-9_]+\.py)")
_NAME = re.compile(r"(?<![A-Za-z0-9_./])(test_[a-z0-9_]{3,})\b(?!\.py|/)")


def _defined() -> tuple[set[str], set[str]]:
    files = {path.name for path in TESTS.glob("test_*.py")}
    names: set[str] = set()
    for path in TESTS.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
    return files, names


def _module_docstrings() -> list[tuple[str, str]]:
    texts = []
    for folder in ("common", "serving", "inference_core", "training", "evaluation", "calibration", "testing",
                   "data_pipeline", "orchestration", "registry_utils", "azure_service", "scripts",
                   "postprocessing", "artifact_registry", "pilot"):
        for path in sorted((REPO / folder).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    doc = ast.get_docstring(node, clean=False)
                    if doc:
                        texts.append((path.relative_to(REPO).as_posix(), doc))
            # Comments too: "# see test_x" is the same claim.
            comments = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                                 if line.lstrip().startswith("#"))
            if comments:
                texts.append((path.relative_to(REPO).as_posix(), comments))
    return texts


def _documents() -> list[tuple[str, str]]:
    texts = []
    sources = [REPO / "README.md", *sorted((REPO / "Documentation" / "Implementation MDs").glob("IMPL-*.md")),
               *sorted((REPO / "prompts").rglob("*.jinja")), *sorted((REPO / "azure_service").glob("*.md"))]
    for path in sources:
        texts.append((path.relative_to(REPO).as_posix(), path.read_text(encoding="utf-8")))
    return texts + _module_docstrings()


def missing_references() -> list[str]:
    files, names = _defined()
    missing: set[str] = set()
    for where, text in _documents():
        for name in _PATH.findall(text):
            if name not in files:
                missing.add(f"{where}: tests/{name}")
        for name in _NAME.findall(text):
            if name in names or name in NOT_TESTS or f"{name}.py" in files:
                continue
            if name in HELD_FOR_PROMPT_VERSION.get(where, set()):
                continue
            missing.add(f"{where}: {name}")
    return sorted(missing)


def test_every_named_test_exists() -> None:
    missing = missing_references()
    assert not missing, "documents name tests that do not exist:\n" + "\n".join(missing)


def test_held_names_are_still_stale() -> None:
    """A held name, once fixed, must leave :data:`HELD_FOR_PROMPT_VERSION`."""
    _files, names = _defined()
    for where, held in HELD_FOR_PROMPT_VERSION.items():
        text = (REPO / where).read_text(encoding="utf-8")
        for name in held:
            assert name in text and name not in names, f"{where}: {name} is fixed or exists - remove it from the list"


def test_a_missing_name_is_seen(monkeypatch) -> None:
    """Seen to fail on purpose: a text naming a test that does not exist."""
    import tests.test_doc_references as module

    monkeypatch.setattr(module, "_documents", lambda: [("x.md", "proved by test_nothing_by_this_name")])
    assert module.missing_references() == ["x.md: test_nothing_by_this_name"]


def test_each_specs_changelog_link_names_a_file() -> None:
    """A spec split into a current specification and a changelog (review R6,
    step 2) names its changelog; the file must be there."""
    folder = REPO / "Documentation" / "Implementation MDs"
    link = re.compile(r"`(changelog/[^`]+\.md)`")
    missing = [f"{spec.name}: {name}" for spec in sorted(folder.glob("IMPL-*.md"))
               for name in link.findall(spec.read_text(encoding="utf-8")) if not (folder / name).is_file()]
    assert not missing, missing
