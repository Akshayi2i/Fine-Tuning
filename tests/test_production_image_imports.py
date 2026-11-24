"""The served image's code runs without the folders the image leaves out.

``azure_service/Dockerfile`` copies a list of folders; ``training/``,
``testing/``, ``pilot/`` and the rest stay out. Code inside the image that
imports one of them works on a laptop (where every folder is present) and
fails only in production - the prompt-length check did so, silently, until
review R4 (23 November 2026). These tests run the served code in a fresh
interpreter where every top-level package the Dockerfile does not copy cannot
be imported, so the laptop sees what the image sees.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "azure_service" / "Dockerfile"


def _copied_packages() -> set[str]:
    """Top-level folders the Dockerfile copies (``COPY name/ name/``)."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    return set(re.findall(r"^COPY\s+([A-Za-z_][\w]*)/\s", text, flags=re.MULTILINE))


def _left_out_packages() -> list[str]:
    """Top-level Python packages in the repo the image does not carry.
    ``scripts`` is copied in part (two files), so it is not blocked here."""
    copied = _copied_packages() | {"scripts"}
    return sorted(
        child.name for child in REPO.iterdir()
        if child.is_dir() and (child / "__init__.py").exists() and child.name not in copied
    )


def _run_as_image(body: str) -> subprocess.CompletedProcess[str]:
    blocked = _left_out_packages()
    script = textwrap.dedent(f"""
        import importlib.abc, sys
        BLOCKED = {blocked!r}

        class _NotInImage(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(f"{{name}} is not in the served image", name=name)
                return None

        sys.meta_path.insert(0, _NotInImage())
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True,
                          text=True, timeout=120, check=False)


def test_the_dockerfile_leaves_training_and_testing_out() -> None:
    """The premise of every test here: if the image starts carrying these, the
    blocker below stops meaning anything and this says so."""
    left_out = _left_out_packages()
    assert "training" in left_out
    assert "testing" in left_out
    assert "common" not in left_out and "serving" not in left_out


def test_the_blocker_blocks() -> None:
    """Seen to fail on purpose: an import of a left-out package fails."""
    done = _run_as_image("import training.length_check\n")
    assert done.returncode != 0
    assert "not in the served image" in done.stderr


def test_the_prompt_length_check_runs_in_the_image(tmp_path: Path) -> None:
    """Review R4: ``inference_core.prompt_tokens`` counts a page's tokens
    with no ``training`` package present, and gives the same count as
    training's own measure."""
    from PIL import Image

    page = tmp_path / "page.png"
    Image.new("L", (1240, 1754), color=255).save(page)
    done = _run_as_image(f"""
        from inference_core.prompt_tokens import check_prompt, expected_prompt
        messages = [{{"role": "user", "content": [
            {{"type": "image", "image": {str(page)!r}}},
            {{"type": "text", "text": "Read the page."}},
        ]}}]
        expected = expected_prompt(messages, budget=(200_704, 1_254_400), tokenizer=None)
        assert expected is not None, "the page's size could not be read"
        checked = check_prompt(expected["image_tokens"] + 20, expected)
        assert checked["plausible"], checked
        print(expected["image_tokens"])
    """)
    assert done.returncode == 0, done.stderr
    from training.length_check import image_tokens

    assert int(done.stdout.strip()) == image_tokens(1240, 1754, min_pixels=200_704, max_pixels=1_254_400)


def test_a_failed_prompt_check_is_warned_and_recorded(monkeypatch, caplog) -> None:
    """A check that cannot run leaves ``prompt_check_error`` in the window's
    stat and a warning, once per reason - never only a debug line."""
    import logging

    from inference_core import prompt_tokens
    from inference_core.model_runner import Generation
    from serving import pipeline

    def broken(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'training'")

    monkeypatch.setattr(prompt_tokens, "expected_prompt", broken)
    monkeypatch.setattr(pipeline, "_PROMPT_CHECK_FAILURES", set())
    result = Generation(text="{}", tokens=[], prompt_tokens=1000)
    messages = [{"role": "user", "content": "x"}]
    first: dict = {}
    second: dict = {}
    with caplog.at_level(logging.WARNING, logger=pipeline.log.name):
        pipeline._check_prompt_tokens("doc-1", first, result, messages, [])
        pipeline._check_prompt_tokens("doc-2", second, result, messages, [])
    assert "ModuleNotFoundError" in first["prompt_check_error"]
    assert second["prompt_check_error"] == first["prompt_check_error"]
    warnings = [r for r in caplog.records if "could not run" in r.getMessage()]
    assert len(warnings) == 1
