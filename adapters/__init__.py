"""The served path's clients of things outside this code (review A1, 2026-11-24): the served MinerU
(``served_ocr``, ``mineru_api``), the PDF renderer (``pdf_pages``), the OCR environment
(``mineru_version``) and how a PDF carries its text (``modality``).

May import the standard library, third-party libraries, ``common``, ``contracts``,
``artifact_registry`` and itself - never build-time code (``tests/test_layering.py``). The old
``data_pipeline.ocr`` paths stay: the whole modules as aliases of these, the two that gave up
pieces as re-exports.
"""
