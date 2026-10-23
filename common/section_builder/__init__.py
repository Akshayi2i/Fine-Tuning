"""Fideon's section builder for the full-text tier, ``text_sections`` (SPEC_21 §4.6, §7.2).

``sections.py`` and ``pdf_text.py`` are COPIES of the team's files in FideonSLM
(``synth_core/fideon/synth_core/``), byte for byte, so L1 and this repository
build one structure: SPEC_21 has "L1 and L3 emit identical full-text structure".
``SOURCE.json`` records the commit they were copied from and each file's SHA-256.

Never edit them here. Change them in FideonSLM, copy them again, and update
``SOURCE.json``: tests/test_full_text.py fails when a copy here no longer matches
it, and ``python -m scripts.check_section_builder --fideon-slm <clone>`` says when
FideonSLM's own files have moved on. They are left out of ruff (pyproject.toml)
for the same reason.

The rest of the repository calls :mod:`common.full_text`, never these modules.
"""
