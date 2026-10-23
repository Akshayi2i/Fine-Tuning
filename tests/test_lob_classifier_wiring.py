"""The endpoint's line classifier: built at cold start when detection is on.

A policy's line decides whether the release's adapter or the base model reads
it, and against which schema, so the line is read first, by the base model with
no adapter. The endpoint builds that classifier
only when routing.detect_lob is on and the engine really holds the base model;
otherwise it holds what it held before detection existed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from inference_core.model_runner import NO_ADAPTER, Generation, ModelBackend
from serving.doc_type_classifier import ZeroShotClassifier

#: What the classifier answers for a homeowners policy.
LINE_ANSWER = '{"doc_type": "policy", "acord_form": null, "lob": "homeowners", "confidence": 0.9}'


class _ReadsTheLine(ModelBackend):
    """Answers the classifier's prompt with a line and every window with {}."""

    def __init__(self):
        self.calls = []

    def supports_logprobs(self):
        return True

    def generate(self, messages, config, adapter=None):
        from common.prompts import render_classifier_prompt

        classifier = messages[0]["content"] == render_classifier_prompt()
        self.calls.append({"classifier": classifier, "adapter": adapter,
                           "json_schema": json.dumps(config.json_schema or {})})
        text = LINE_ANSWER if classifier else "{}"
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(text=text, tokens=tokens, token_logprobs=[-0.001] * len(tokens),
                          generation_fingerprint=config.fingerprint())


def _base_model(backend, *, default_adapter=None):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import load_model

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load_model("base", client, backend_impl=backend)
    model.default_adapter = default_adapter
    return model


def test_the_line_classifier_reads_with_the_base_weights_alone():
    from serving.vllm_entrypoint import line_classifier

    classifier = line_classifier(_base_model(_ReadsTheLine()), SimpleNamespace(served=[]))
    assert isinstance(classifier, ZeroShotClassifier) and classifier.adapter == NO_ADAPTER


def test_an_engine_serving_the_release_as_a_lora_holds_the_base():
    """A release that leaves lines to the base model is served as a LoRA on it."""
    from serving.release_router import ServedRelease, ServingPlan
    from serving.vllm_entrypoint import line_classifier

    merged_or_lora = SimpleNamespace(is_base=False)
    some_lines = ServingPlan(ServedRelease("v3", "base@abc1234", "merged", doc_types=("policy",),
                                           lines=("homeowners",)))
    every_line = ServingPlan(ServedRelease("v3", "base@abc1234", "merged", doc_types=("policy",)))
    assert line_classifier(merged_or_lora, some_lines) is not None
    assert line_classifier(merged_or_lora, every_line) is None        # the engine is its merged model


def test_an_engine_holding_one_merged_release_reads_no_line():
    """No adapter on a merged model is that release's weights, not the base."""
    from serving.vllm_entrypoint import line_classifier

    assert line_classifier(SimpleNamespace(is_base=False), SimpleNamespace(served=[])) is None


def test_the_endpoint_builds_it_only_with_detection_on():
    from serving.vllm_entrypoint import classifier_for

    model, plan = _base_model(_ReadsTheLine()), SimpleNamespace(served=[])
    assert classifier_for(None, model, plan, detect_lob=False) is None
    built = classifier_for(None, model, plan, detect_lob=True)
    assert isinstance(built, ZeroShotClassifier) and built.adapter == NO_ADAPTER
    given = object()
    assert classifier_for(given, model, plan, detect_lob=True) is given


def test_detection_ships_off_so_the_endpoint_builds_nothing_yet():
    from serving.vllm_entrypoint import classifier_for, serving_thresholds

    detect = bool(serving_thresholds().get("detect_lob"))
    assert detect is False
    assert classifier_for(None, _base_model(_ReadsTheLine()), SimpleNamespace(served=[]),
                          detect_lob=detect) is None


def test_a_classifier_given_no_adapter_calls_generate_as_before():
    seen = {}

    def generate(model, messages, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text=LINE_ANSWER, tokens=[], token_logprobs=[])

    ZeroShotClassifier(SimpleNamespace(is_base=False), generate).classify(["p1.png"], "x")
    assert seen == {"want_logprobs": True}


def test_a_scanned_policy_with_no_line_is_read_routed_and_extracted_as_its_line():
    """End to end: the base model reads the line with no adapter - even on a model
    whose default adapter is a release's - and the windows are then decoded
    against that line's schema."""
    from serving.pipeline import ExtractionRequest, extract
    from serving.vllm_entrypoint import classifier_for
    from tests.test_serving_pipeline import CALIBRATION

    backend = _ReadsTheLine()
    model = _base_model(backend, default_adapter="release-lora")
    classifier = classifier_for(None, model, SimpleNamespace(served=[]), detect_lob=True)
    result = extract(
        ExtractionRequest(source_id="scan_1", image_paths=["d/page_1.png", "d/page_2.png"],
                          page_texts={1: "DECLARATIONS", 2: "SCHEDULE"}, known_doc_type="policy",
                          modality_mode="noisy_ocr_image"),
        model, classifier, CALIBRATION, strict_schema=False, detect_lob=True,
    )
    assert result.route_info["lob_source"] == "detected"
    assert result.line_of_business["lines"] == ["homeowners"]
    read = [c for c in backend.calls if c["classifier"]]
    assert len(read) == 1 and read[0]["adapter"] is None              # the base weights, no LoRA
    windows = [c for c in backend.calls if not c["classifier"]]
    # Decoded against the homeowners schema: its coverage list holds Coverage A.
    assert any("HO_COV_A" in c["json_schema"] for c in windows)
