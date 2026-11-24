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
                if p.kind is inspect.Parameter.KEYWORD_ONLY}
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
