"""``extract`` read in stages with one settings object (review A2, 2026-11-24).

No behaviour change is the rule: the serving, printed-values, fields-first and as-served
tests run the reading end to end through the stages. These hold the seams the split made.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

from serving.pipeline import extract
from serving.read_settings import ReadSettings

STAGES = ("route", "release", "answer", "printed", "confidence", "review", "output", "validate", "result")


def test_read_settings_holds_every_keyword_of_extract_with_its_default() -> None:
    """While both exist, a default changed in one and not the other would read differently
    through ``extract`` than through a caller holding its own ``ReadSettings``."""
    keywords = {name: p.default for name, p in inspect.signature(extract).parameters.items()
                if p.kind is inspect.Parameter.KEYWORD_ONLY and name != "settings"}
    fields = {f.name: f.default for f in dataclasses.fields(ReadSettings)}
    assert keywords == fields


def test_every_stage_is_a_module_of_its_own() -> None:
    for name in STAGES:
        module = importlib.import_module(f"serving.stages.{name}")
        assert module.__doc__ and "review A2" in module.__doc__


def test_the_stages_reach_the_pipelines_helpers_through_the_module(monkeypatch) -> None:
    """A test patching ``serving.pipeline.<helper>`` still reaches the reading."""
    import serving.pipeline as pipeline
    from serving.stages import output

    seen = []
    monkeypatch.setattr(pipeline, "_full_text_pages", lambda request: seen.append(request) or [])

    class _Chosen:
        route, lob, canonical, common_model = None, None, False, False

    class _Answer:
        extraction: dict = {}

    class _Reviewed:
        flags: list = []

    output.served_output("request", _Chosen(), _Answer(), None, _Reviewed())
    assert seen == ["request"]


def test_settings_and_keyword_options_together_are_refused() -> None:
    """One of the two would be ignored without a word."""
    import pytest

    with pytest.raises(TypeError, match="not both"):
        extract(None, None, None, None, strict_schema=False, settings=ReadSettings.testing())


def test_each_entry_points_profile() -> None:
    served = ReadSettings.served({"review_threshold": 0.8, "calibrators": "C", "thresholds": "T"})
    assert (served.strict_schema, served.review_threshold, served.calibrators) == (False, 0.8, "C")
    gate = ReadSettings.gate(calibrators="C", thresholds="T")
    assert (gate.strict_schema, gate.calibrators, gate.thresholds) == (False, "C", "T")
    assert ReadSettings.testing() == ReadSettings.draft() == ReadSettings(strict_schema=False)
    endpoint = ReadSettings.endpoint(adapter_map={}, plan="P", release_runtimes=None, base_model_loaded=True,
                                     tuning={"detect_lob": True})
    assert endpoint.strict_schema is True and endpoint.plan == "P" and endpoint.detect_lob is True


def test_no_caller_in_the_repository_passes_keyword_options() -> None:
    """Every entry point reads with a named profile (review A2): a keyword option passed to
    extract outside the tests is a reading nobody named."""
    import ast
    from pathlib import Path

    options = {f.name for f in dataclasses.fields(ReadSettings)}
    repo = Path(__file__).resolve().parents[1]
    found = []
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        if rel.startswith(("tests/", "data/", ".")) or "/." in rel:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "extract":
                passed = {k.arg for k in node.keywords} & options
                if passed:
                    found.append(f"{rel}:{node.lineno}: {sorted(passed)}")
    assert not found, found

