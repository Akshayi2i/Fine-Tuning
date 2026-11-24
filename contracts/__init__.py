"""Rules both the served path and the build-time code hold to (review A1, 2026-11-24).

Imports only the standard library and ``common``. A piece the served path needs from
``data_pipeline``, ``evaluation``, ``scripts`` or ``orchestration`` moves here, and its
old module re-exports it, so callers on both sides keep working
(``tests/test_layering.py``).
"""
