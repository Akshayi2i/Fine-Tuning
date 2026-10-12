"""Our log lines stay visible for the whole run.

On the CGL full run every INFO and WARNING line of ours stopped after training:
seven hours of checkpoint selection logged neither its progress nor its choice,
while vLLM kept printing through its own handler. common.logging_setup.reassert
undoes what a library did since setup().
"""

from __future__ import annotations

import io
import logging

from common import logging_setup


def test_reassert_brings_our_info_lines_back_after_a_library_silences_them():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    ours = logging.getLogger("evaluation.validation_generation")
    azure = logging.getLogger("azure")
    saved = (root.level, ours.level, azure.level)
    root.addHandler(handler)
    try:
        root.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
        azure.setLevel(logging.WARNING)
        # What a library's logging setup did: the handler and the root raised,
        # our logger disabled.
        handler.setLevel(logging.ERROR)
        root.setLevel(logging.ERROR)
        ours.disabled = True
        ours.info("lost")
        assert "lost" not in stream.getvalue()

        logging_setup.reassert()
        ours.info("validation generation: 32/400 row(s)")
        assert "validation generation: 32/400 row(s)" in stream.getvalue()
        assert handler.level == logging.INFO and root.level == logging.INFO
        # It never raises a level, and never lowers a library's own logger.
        assert azure.level == logging.WARNING
    finally:
        root.removeHandler(handler)
        root.setLevel(saved[0])
        ours.setLevel(saved[1])
        ours.disabled = False
        azure.setLevel(saved[2])


def test_reassert_restores_a_console_handler_a_library_removed(monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    logging_setup.reassert()
    assert len(root.handlers) == 1
    assert root.handlers[0].formatter._fmt == logging_setup.FORMAT
