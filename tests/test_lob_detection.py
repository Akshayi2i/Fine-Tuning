"""Reading a policy's line of business when no caller supplied it.

A policy that reaches L3 with no line - a scan L1/L2 could not read - is
classified for its line too, and routed by how sure that reading is: confidently
to its line's schema, uncertainly to its line's schema too (flagged), not at all
to the base model and the fallback schema (flagged). A caller's line always
wins.
"""

from __future__ import annotations

import json
import math

import pytest

from serving.doc_type_classifier import (
    Classification,
    classifier_messages,
    combine_lob_with_hypothesis,
    line_confidence,
    parse_classification,
)


def test_every_registered_line_has_a_meaning():
    from common.lob import known_lines
    from common.prompts import lob_meanings

    assert set(lob_meanings()) == set(known_lines())


def test_no_meaning_quotes_a_label_carriers_print():
    """The classifier chooses a line by what it IS (master §1.4)."""
    from common import aliases
    from common.prompts import lob_meanings
    from tests.test_schema2_model_view import _common_model_aliases

    text = " ".join(lob_meanings().values()).casefold()
    labels = {a for entry in aliases.load_registry("policy").values() for a in entry.aliases}
    for lob in ("homeowners", "personal_auto", "ocean_marine"):
        labels |= _common_model_aliases(lob)
    leaked = sorted(a for a in labels if " " in a and len(a) > 12 and a.casefold() in text)
    assert not leaked, leaked[:5]


def test_the_classifier_is_asked_for_the_line_with_every_lines_meaning():
    from common.prompts import lob_meanings, render_classifier_prompt

    prompt = render_classifier_prompt()
    assert '"lob": "<one of the lines above>" | null' in prompt
    for line, meaning in lob_meanings().items():
        assert f"`{line}` - {meaning}" in prompt


def test_a_line_only_question_asks_only_for_the_lines_given():
    from common.prompts import render_classifier_prompt

    prompt = render_classifier_prompt(lines=["homeowners", "dwelling_fire"], line_only=True)
    assert '{"lob": "homeowners" | "dwelling_fire"}' in prompt
    assert "personal_auto" not in prompt and "doc_type" not in prompt


@pytest.mark.parametrize("answer,expected", [
    ({"doc_type": "policy", "lob": "homeowners"}, "homeowners"),
    ({"doc_type": "policy", "lob": "classic_auto"}, "personal_auto"),     # read as personal auto
    ({"doc_type": "policy", "lob": "workers_comp"}, "wc"),               # the enum, as the file
    ({"doc_type": "policy", "lob": "pet_insurance"}, None),              # no such line
    ({"doc_type": "policy", "lob": None}, None),                         # a package
    ({"doc_type": "lossrun", "lob": "homeowners"}, None),                # no policy, no line
])
def test_a_line_is_read_as_a_registered_line_or_not_at_all(answer, expected):
    assert parse_classification(json.dumps({**answer, "confidence": 0.9})).lob == expected


def test_the_lines_confidence_is_its_own_tokens_probability():
    text = '{"doc_type": "policy", "lob": "homeowners", "confidence": 0.99}'
    tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
    logprobs = [-0.5 if '"homeowners"'.find(t) >= 0 or t in "homeowners" else -0.01 for t in tokens]
    confidence = line_confidence(text, tokens, logprobs)
    assert confidence is not None and 0 < confidence < 0.99          # never the model's own 0.99
    assert line_confidence(text, [], []) is None


def test_the_messages_are_the_same_for_training_and_serving():
    messages = classifier_messages(["p1.png", "p2.png", "p3.png"], "x" * 5000, lines=["homeowners"],
                                   line_only=True)
    images = [b for b in messages[1]["content"] if b["type"] == "image"]
    assert [b["image"] for b in images] == ["p1.png", "p2.png"]
    assert len(messages[1]["content"][-1]["text"]) == 4000


def test_an_agreeing_hint_lifts_the_line_and_a_disagreeing_one_becomes_a_candidate():
    agree = combine_lob_with_hypothesis(
        Classification("policy", None, 0.9, lob="homeowners", lob_confidence=0.4), "homeowners",
        confidence_threshold=0.9)
    assert agree.lob_confidence == 0.9 and agree.lob_hypothesis_agreed
    differ = combine_lob_with_hypothesis(
        Classification("policy", None, 0.9, lob="homeowners", lob_confidence=0.4), "dwelling_fire",
        confidence_threshold=0.9)
    assert differ.lob == "homeowners" and differ.lob_hypothesis_agreed is False
    assert ("dwelling_fire", 0.4) in differ.lob_candidates


# --------------------------------------------------------------------------
# Routing (serving.pipeline)
# --------------------------------------------------------------------------


class _Reads:
    """A classifier that read a policy as ``lob`` with ``confidence``."""

    def __init__(self, lob, confidence):
        self.result = Classification("policy", None, 0.95, method="zero_shot", lob=lob,
                                     lob_confidence=confidence, lob_stage="base")
        self.calls = 0

    def classify(self, image_paths, ocr_text):
        self.calls += 1
        return self.result


def _extract(classifier, **request):
    """Extract with detection on: it is off by default until it has been measured."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import EchoBackend, load_model
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_serving_pipeline import CALIBRATION

    backend = EchoBackend("{}")
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    result = extract(
        ExtractionRequest(source_id="p1", image_paths=["d/page_1.png"], page_texts={1: "Declarations"},
                          **request),
        load_model("base", client, backend_impl=backend), classifier, CALIBRATION, strict_schema=False,
        detect_lob=True,
    )
    schemas = [json.dumps(call["json_schema"]) for call in backend.calls]
    return result, schemas


def test_a_confidently_read_line_is_routed_as_the_callers_would_be():
    result, schemas = _extract(_Reads("homeowners", 0.97))
    assert result.route_info["lob_source"] == "detected"
    assert result.line_of_business == {"lines": ["homeowners"], "source": "detected", "confidence": 0.97,
                                       "candidates": [], "hypothesis_agreed": None}
    assert any("CoverageCode" in s for s in schemas)                     # the homeowners model view
    assert not [f for f in result.review_flags if f.startswith("lob:")]


def test_an_uncertain_line_is_kept_and_flagged():
    result, schemas = _extract(_Reads("homeowners", 0.7))
    assert result.route_info["lob_source"] == "detected_uncertain"
    assert "lob:uncertain" in result.review_flags
    assert "lob:uncertain_within_family" not in result.review_flags
    assert any("CoverageCode" in s for s in schemas)                     # its line's schema


def test_a_line_read_with_no_confidence_is_uncertain_not_detected():
    result, _ = _extract(_Reads("homeowners", None))
    assert result.route_info["lob_source"] == "detected_uncertain"


def test_an_unreadable_line_goes_to_the_fallback_flagged():
    result, schemas = _extract(_Reads("homeowners", 0.2))
    assert result.route_info["lob_source"] == "undetected"
    assert "lob:undetected" in result.review_flags
    assert not any("CoverageCode" in s for s in schemas)                 # the generic fallback


def test_the_callers_line_wins_and_a_confident_disagreement_is_flagged():
    classifier = _Reads("dwelling_fire", 0.99)
    result, _ = _extract(classifier, known_lob="homeowners")            # the classifier ran for the type
    assert result.route_info["lob_source"] == "caller"
    assert result.line_of_business["lines"] == ["homeowners"]
    assert "lob:caller_disagrees" in result.review_flags


def test_a_caller_that_names_the_type_still_gets_the_line_read():
    classifier = _Reads("homeowners", 0.97)
    result, _ = _extract(classifier, known_doc_type="policy")
    assert classifier.calls == 1 and result.route_info["lob_source"] == "detected"


def test_an_l1_hint_that_agrees_lifts_an_uncertain_reading():
    result, _ = _extract(_Reads("homeowners", 0.7), lob_hypothesis="homeowners")
    assert result.route_info["lob_source"] == "detected"
    assert result.line_of_business["hypothesis_agreed"] is True


def test_detection_can_be_turned_off():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import EchoBackend, load_model
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_serving_pipeline import CALIBRATION

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    result = extract(ExtractionRequest(source_id="p1", image_paths=["d/page_1.png"], page_texts={1: "x"}),
                     load_model("base", client, backend_impl=EchoBackend("{}")), _Reads("homeowners", 0.99),
                     CALIBRATION, strict_schema=False, detect_lob=False)
    assert result.route_info["lob_source"] is None


# --------------------------------------------------------------------------
# Training rows and scoring
# --------------------------------------------------------------------------


def test_a_classify_row_asks_for_the_line_among_every_known_line():
    from common.lob import known_lines
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, classify_rows

    doc = SourceDocument(source_id="h1", doc_type="policy", golden_label={}, ocr_pages=["Declarations"],
                         image_paths=["h/page_1.png"], lob="homeowners", tenant_id="default")
    rows = classify_rows(doc, "train", ("ocr_plus_image", "image_only"))
    assert [r["task"] for r in rows] == ["classify", "classify"]
    assert json.loads(rows[0]["messages"][-1]["content"]) == {"lob": "homeowners"}
    system = rows[0]["messages"][0]["content"]
    assert all(f'"{line}"' in system for line in known_lines())          # gl as well as dwelling fire
    assert not any(b["type"] == "text" for b in rows[1]["messages"][1]["content"])   # image only

    def other(**fields):
        return SourceDocument(**{"source_id": "g", "doc_type": "policy", "golden_label": {},
                                 "ocr_pages": ["x"], "image_paths": ["g/page_1.png"],
                                 "tenant_id": "default", **fields})

    assert classify_rows(other(doc_type="lossrun", lob="gl"), "train", ("ocr_plus_image",)) == []
    assert classify_rows(other(lob="pet_insurance"), "train", ("ocr_plus_image",)) == []    # no schema
    assert classify_rows(other(lob=["homeowners", "gl"]), "train", ("ocr_plus_image",)) == []  # a package


def test_line_detection_is_scored_per_line():
    from evaluation.metrics.lob_detection import score_lob_detection

    report = score_lob_detection(
        [("homeowners", "homeowners")] * 18 + [("homeowners", "dwelling_fire")] * 2
        + [("personal_auto", "auto")] * 3 + [("personal_auto", None)]
    )
    assert math.isclose(report.overall, 18 / 24)
    lines = report.by_line()
    assert lines["homeowners"] == {"documents": 20, "accuracy": 0.9, "undetected": 0}
    assert lines["personal_auto"] == {"documents": 4, "accuracy": 0.0, "undetected": 1}
    assert report.below_floor(0.95) == ["homeowners"]
