"""Fixtures shared across test modules."""

from __future__ import annotations

import pytest


@pytest.fixture
def no_window_overlap(monkeypatch):
    """Plan windows with no page shared between the windows of a split run.

    For tests of what happens at a window boundary - a fragment left without
    its identifier, a link to a row the window does not hold, two halves of a
    row joined by the merge. The configured overlap (``run_overlap_pages``)
    moves those boundaries; the mechanisms they test do not go away with it.
    """
    import common.schema_sections

    monkeypatch.setattr(common.schema_sections, "run_overlap", lambda lob=None: 0)


@pytest.fixture(autouse=True)
def _format_profile(request, monkeypatch):
    """``@pytest.mark.format_profile("standard")``: the test runs in that format
    profile instead of the shipped one (configs/shared/format.yaml, ``fast``).

    For tests of the standard format's own windows, prompts and answers, which
    stay as they were for any corpus built in it. Everything else runs in the
    shipped profile, the one production reads.
    """
    marker = request.node.get_closest_marker("format_profile")
    if marker is not None:
        import common.config

        profile = marker.args[0]
        monkeypatch.setattr(common.config, "shared_format_config", lambda: {"profile": profile})
