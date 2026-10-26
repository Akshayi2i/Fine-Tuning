"""Validation generation reads a fast-format row as serving reads it (the fast
format profile, 2026-10-25): its gold and its answer expanded from the compact
envelope, its decoding schema compact, its OCR text the pages' alone. On a
standard row every one of these is a no-op."""

from __future__ import annotations

import json

import pytest

from common.canonical import COMPACT_ANSWER_FORMAT
from evaluation import validation_generation as V
from evaluation.validation_generation import ValidationGeneration, split_prompt
from inference_core.input_builder import build_training_row
from inference_core.model_runner import Generation

COMPACT = {"policy": {"policy_number": {"v": "HO-1234567", "p": [1]},
                      "effective_date": {"r": "Jan 1, 2026", "v": "01/01/2026", "p": [1]}}}
FULL = {"policy": {"policy_number": {"raw": "HO-1234567", "parsed": "HO-1234567", "page_ref": [1]},
                   "effective_date": {"raw": "Jan 1, 2026", "parsed": "01/01/2026", "page_ref": [1]}}}
PAGE = "Policy Number HO-1234567 Effective Jan 1, 2026"


def _row(answer: dict, *, compact: bool, lob: str = "homeowners") -> dict:
    row = build_training_row("policy", "policy_0001", ["processed/default/policy/policy_0001/page_1.png"],
                             [PAGE], "ocr_plus_image", json.dumps(answer), lob=lob, sections="decl",
                             split="val")
    if compact:
        row["answer_format"] = COMPACT_ANSWER_FORMAT
    return row


def test_a_compact_gold_is_scored_in_the_full_form():
    assert split_prompt(_row(COMPACT, compact=True))[1] == FULL
    assert split_prompt(_row(FULL, compact=False))[1] == FULL          # a full gold is unchanged


def _envelope_keys(schema) -> set[str]:
    """Every key of every object in ``schema`` that declares a page reference."""
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and ({"page_ref", "p"} & set(props)):
                found.update(props)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return found


@pytest.mark.parametrize(("compact", "keys", "absent"), [(True, {"r", "v", "p"}, "raw"),
                                                         (False, {"raw", "parsed", "page_ref"}, "r")])
@pytest.mark.format_profile("standard")
def test_a_compact_row_is_decoded_against_the_compact_schema(monkeypatch, compact, keys, absent):
    schemas = []

    def generate_batch(model, requests, adapter=None):
        schemas.extend(schema for _messages, schema, _cap in requests)
        return [RuntimeError("not generated") for _ in requests]

    monkeypatch.setattr("inference_core.model_runner.generate_batch", generate_batch)
    model = type("M", (), {"config": type("C", (), {"structured_outputs": True})()})()
    V.generate_validation([_row(COMPACT if compact else FULL, compact=compact)], model, constrain=True)
    [schema] = schemas
    assert keys <= _envelope_keys(schema) and absent not in _envelope_keys(schema)


def test_a_compact_answer_is_expanded_and_its_spans_keyed_as_the_full_one():
    entry = ValidationGeneration(row=_row(COMPACT, compact=True), golden=FULL)
    text = json.dumps(COMPACT, separators=(",", ":"))
    V._finish(entry, Generation(text=text, tokens=list(text), token_logprobs=[-0.01] * len(text)))
    assert entry.error is None
    assert entry.extraction == FULL
    assert {"policy.policy_number", "policy.effective_date"} <= set(entry.logprobs_by_path)


def test_a_standard_answer_is_read_as_before():
    entry = ValidationGeneration(row=_row(FULL, compact=False), golden=FULL)
    text = json.dumps(FULL, separators=(",", ":"))
    V._finish(entry, Generation(text=text, tokens=list(text), token_logprobs=[-0.01] * len(text)))
    assert entry.extraction == FULL and "policy.policy_number" in entry.logprobs_by_path


def test_the_ocr_text_is_the_pages_not_the_request_block():
    row = _row(FULL, compact=False)
    before = ValidationGeneration(row=row, golden={}).page_text
    user = next(m for m in row["messages"] if m["role"] == "user")
    user["content"].append({"type": "text", "text": "\n\n# Request: decl - the declarations group"})
    after = ValidationGeneration(row=row, golden={}).page_text
    assert after == before and "Request" not in (after or "")
