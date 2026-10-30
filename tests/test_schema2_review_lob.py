"""Line-of-business detection after review: what it must never do, and what it must report.

Reading a policy's line is an optional lookup, so it must never fail a request
that would have been served without it - a wrapped answer, an answer that is no
JSON, no classifier at all. A classifier that never reads lines must not look
like one that looked and found nothing. Detection routes production policies
only once it has been measured, so it ships off and the measurement exists.
What trains the classifier is what serving sends it, the line hint can arrive
through the endpoint, and the prompt hash covers what the classifier is shown.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from serving.doc_type_classifier import (
    Classification,
    StaticClassifier,
    ZeroShotClassifier,
    classifier_messages,
    line_confidence,
)

ANSWER = '{"doc_type": "policy", "acord_form": null, "lob": "homeowners", "confidence": 0.95}'


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _model(client):
    from inference_core.model_runner import EchoBackend, load_model

    backend = EchoBackend("{}")
    return load_model("base", client, backend_impl=backend), backend


def _answering(text, *, tokens=None, logprobs=None, calls=None):
    """A generate function whose model answers ``text``, one token per character."""
    tokens = list(text) if tokens is None else tokens
    logprobs = [-0.01] * len(tokens) if logprobs is None else logprobs

    def generate(model, messages, want_logprobs=False):
        if calls is not None:
            calls.append(messages)
        return SimpleNamespace(text=text, tokens=tokens, token_logprobs=logprobs)

    return generate


def _request(**fields):
    from serving.pipeline import ExtractionRequest

    return ExtractionRequest(**{"source_id": "p1", "image_paths": ["d/page_1.png"],
                                "page_texts": {1: "Declarations"}, **fields})


# --------------------------------------------------------------------------
# L1: a wrapped answer never fails the request
# --------------------------------------------------------------------------

WRAPPED = {
    "fenced": f"```json\n{ANSWER}\n```",
    "think": f"<think>\nThe declarations insure a dwelling.\n</think>\n\n{ANSWER}",
    "trailing prose": f"{ANSWER}\nThe policy is homeowners.",
}


@pytest.mark.parametrize("wrapping", sorted(WRAPPED))
def test_a_wrapped_answer_is_read_with_its_lines_confidence(wrapping):
    """parse_classification tolerates the wrapping, and the confidence was read
    from the raw text as if it were bare JSON - SpanMapError escaped extract(),
    even for a request whose caller had sent the line."""
    text = WRAPPED[wrapping]
    classification = ZeroShotClassifier(SimpleNamespace(), _answering(text)).classify(["p.png"], "x")
    assert classification.lob == "homeowners"
    # One token per character at -0.01: the value's 10 characters, its quotes
    # left out (inference_core.span_map scores a string's own characters).
    assert classification.lob_confidence == round(math.exp(-0.10), 4)


def test_only_the_values_tokens_count_wherever_the_object_sits():
    """The span is found in the object and shifted by where the object starts."""
    text = WRAPPED["think"]
    value = text.index('"homeowners"')
    logprobs = [-0.01 if value <= i < value + len('"homeowners"') else -3.0 for i in range(len(text))]
    assert line_confidence(text, list(text), logprobs) == round(math.exp(-0.10), 4)     # its 10 characters


@pytest.mark.parametrize("tokens,logprobs", [
    (["{", "}"], [-0.1, -0.1]),                       # the tokens do not rebuild the text
    (list(ANSWER), [-0.1] * (len(ANSWER) - 1)),       # one logprob short
    ([], []),
])
def test_misaligned_tokens_give_no_confidence_not_an_error(tokens, logprobs):
    assert line_confidence(ANSWER, tokens, logprobs) is None


@pytest.mark.parametrize("text", [
    '{"doc_type": "policy", "confidence": 0.9}',                      # no lob
    '{"doc_type": "policy", "lob": "homeowners", "confidence": 0.9,}',  # does not scan
    "no object at all",
])
def test_a_value_that_cannot_be_located_gives_no_confidence(text):
    assert line_confidence(text, list(text), [-0.1] * len(text)) is None


def test_a_wrapped_answer_with_the_callers_line_is_served(client):
    from serving.pipeline import extract
    from tests.test_serving_pipeline import CALIBRATION

    model, _backend = _model(client)
    classifier = ZeroShotClassifier(SimpleNamespace(), _answering(WRAPPED["fenced"]))
    result = extract(_request(known_lob="homeowners"), model, classifier, CALIBRATION, strict_schema=False)
    assert result.route_info["lob_source"] == "caller"


# --------------------------------------------------------------------------
# L2: a failed or impossible read is never a failed request
# --------------------------------------------------------------------------

def test_an_answer_that_is_no_json_leaves_the_line_undetected(client):
    """The caller named the type, so on main the classifier never ran; the extra
    read for the line raised ClassificationError out of extract()."""
    from serving.pipeline import extract
    from tests.test_serving_pipeline import CALIBRATION

    model, _backend = _model(client)
    classifier = ZeroShotClassifier(SimpleNamespace(), _answering("I think this is a homeowners policy."))
    result = extract(_request(known_doc_type="policy"), model, classifier, CALIBRATION,
                     strict_schema=False, detect_lob=True)
    assert result.route_info["lob_source"] == "undetected"
    assert "lob:undetected" in result.review_flags


def test_a_failed_read_applies_no_hint():
    """A hint is too weak to route on alone, so a read that failed stays undetected."""
    from serving.adapter_router import Route
    from serving.pipeline import _resolve_lob

    class Broken:
        reads_lob = True

        def classify(self, image_paths, ocr_text):
            raise RuntimeError("generation failed")

    route_ = Route("policy", None, None, "policy", None,
                   classification=StaticClassifier("policy").classify([], None))
    found = _resolve_lob(_request(known_doc_type="policy", lob_hypothesis="homeowners"), Broken(), route_,
                         detect=True, line_threshold=0.9, uncertain_threshold=0.6)
    assert (found.lob, found.source, found.flags, found.hypothesis_agreed) == (
        None, "undetected", ["lob:undetected"], None)


def test_no_classifier_and_the_fallback_asked_for_reads_the_fallback_as_on_main(client):
    """cold_start wires no classifier by default; the read died with AttributeError."""
    from serving.pipeline import extract
    from tests.test_serving_pipeline import CALIBRATION

    model, backend = _model(client)
    result = extract(_request(known_doc_type="policy", allow_lob_fallback=True), model, None, CALIBRATION,
                     strict_schema=False, detect_lob=True)
    assert result.route_info["lob_source"] is None
    assert not [f for f in result.review_flags if f.startswith("lob:")]
    assert not any("CoverageCode" in json.dumps(call["json_schema"]) for call in backend.calls)


def test_no_classifier_and_a_plan_reads_the_fallback_or_refuses_cleanly(client):
    from common.schemas import SNAPSHOT_LINES
    from serving.pipeline import LOB_FALLBACK_FLAG, PipelineError, extract
    from serving.release_router import build_serving_plan
    from tests.test_release_router import promote

    promote(client, "v3", lines=SNAPSHOT_LINES)
    plan = build_serving_plan(client)
    model, _backend = _model(client)
    with pytest.raises(PipelineError, match="allow_lob_fallback"):
        extract(_request(known_doc_type="policy"), model, None, None, plan=plan, detect_lob=True)
    result = extract(_request(known_doc_type="policy", allow_lob_fallback=True), model, None, None,
                     plan=plan, strict_schema=False, detect_lob=True)
    assert result.route_info["lob_fallback_used"] and LOB_FALLBACK_FLAG in result.review_flags


def test_a_static_classifier_is_no_detection():
    """It never reads lines, so its None is no reading - it must not switch the
    automatic fallback on."""
    from serving.adapter_router import Route
    from serving.pipeline import _resolve_lob

    static = StaticClassifier("policy")
    route_ = Route("policy", None, None, "policy", None, classification=static.classify([], None))
    found = _resolve_lob(_request(known_doc_type="policy"), static, route_, detect=True,
                         line_threshold=0.9, uncertain_threshold=0.6)
    assert (found.lob, found.source, found.flags) == (None, None, [])


def test_a_classifier_that_never_asks_for_the_line_is_no_detection():
    from serving.adapter_router import Route
    from serving.pipeline import _resolve_lob

    class TypeOnly:
        def classify(self, image_paths, ocr_text):
            return Classification("policy", None, 0.95, method="trained")

    classifier = TypeOnly()
    route_ = Route("policy", None, None, "policy", None, classification=classifier.classify([], None))
    found = _resolve_lob(_request(), classifier, route_, detect=True, line_threshold=0.9, uncertain_threshold=0.6)
    assert found.source is None


def test_a_line_reader_that_found_no_line_is_undetected():
    from serving.adapter_router import Route
    from serving.pipeline import _resolve_lob

    reader = ZeroShotClassifier(SimpleNamespace(), _answering(ANSWER.replace('"homeowners"', "null")))
    route_ = Route("policy", None, None, "policy", None, classification=reader.classify([], None))
    found = _resolve_lob(_request(), reader, route_, detect=True, line_threshold=0.9, uncertain_threshold=0.6)
    assert (found.source, found.flags) == ("undetected", ["lob:undetected"])
    assert ZeroShotClassifier.reads_lob is True


# --------------------------------------------------------------------------
# L3: off until measured, and the measurement
# --------------------------------------------------------------------------

def test_detection_ships_off_in_the_pipeline_and_the_endpoint_config(client):
    from serving.pipeline import extract
    from serving.vllm_entrypoint import serving_thresholds
    from tests.test_lob_detection import _Reads
    from tests.test_serving_pipeline import CALIBRATION

    assert inspect.signature(extract).parameters["detect_lob"].default is False
    assert serving_thresholds()["detect_lob"] is False
    model, _backend = _model(client)
    reader = _Reads("homeowners", 0.99)
    result = extract(_request(known_doc_type="policy", allow_lob_fallback=True), model, reader, CALIBRATION,
                     strict_schema=False)
    assert reader.calls == 0 and result.route_info["lob_source"] is None


class _ReadsBySource:
    """A line reader whose answer depends on the document: ``{source: (lob, confidence)}``,
    keyed by the first image's folder. A missing source is a failed read."""

    reads_lob = True

    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def classify(self, image_paths, ocr_text):
        self.calls += 1
        source = image_paths[0].split("/")[0]
        if source not in self.answers:
            raise RuntimeError("generation failed")
        lob, confidence = self.answers[source]
        return Classification("policy", None, 0.95, method="zero_shot_base", lob=lob,
                              lob_confidence=confidence, lob_stage="base")


def _cases(spec):
    """``spec``: (source, true line, is_scanned, held_out_carrier) per document."""
    from evaluation.lob_detection_eval import DetectionCase

    return [DetectionCase(source, lob, _request(source_id=source, image_paths=[f"{source}/page_1.png"],
                                                known_lob=lob, lob_hypothesis=lob),
                          is_scanned=scanned, held_out_carrier=held_out)
            for source, lob, scanned, held_out in spec]


def _measure(cases, reader, **kwargs):
    from evaluation.lob_detection_eval import measure_lob_detection

    return measure_lob_detection(cases, reader, line_threshold=0.90, uncertain_threshold=0.60, **kwargs)


def test_the_measurement_resolves_as_serving_does_with_the_line_and_hint_withheld():
    cases = _cases([
        ("a", "homeowners", False, False),        # right, confident: detected
        ("b", "homeowners", True, False),         # wrong line, confident: routed with no flag
        ("c", "personal_auto", True, True),       # right, uncertain: its line, flagged
        ("d", "personal_auto", False, True),      # too unsure: the fallback
        ("e", "motorcycle", False, False),        # the read failed
        ("f", "homeowners", False, False),        # right at exactly the line threshold
    ])
    reader = _ReadsBySource({"a": ("homeowners", 0.97), "b": ("dwelling_fire", 0.95),
                             "c": ("personal_auto", 0.7), "d": ("personal_auto", 0.3),
                             "f": ("homeowners", 0.90)})
    body = _measure(cases, reader, min_support=1)

    assert body["documents"] == 6 and math.isclose(body["overall"], round(3 / 6, 4))
    assert body["confidently_wrong"] == 1 and math.isclose(body["confidently_wrong_rate"], round(1 / 6, 4))
    assert body["sources"] == {"detected": 3, "detected_uncertain": 1, "undetected": 2}
    assert body["read_failures"] == 1
    assert body["by_line"]["homeowners"] == {"documents": 3, "accuracy": round(2 / 3, 4), "undetected": 0}
    assert set(body["below_floor"]) == {"homeowners", "personal_auto", "motorcycle"}
    assert body["clears_floor"] is False
    assert (body["line_threshold"], body["uncertain_threshold"], body["floor"]) == (0.90, 0.60, 0.85)
    assert "family_threshold" not in body
    assert body["slices"]["scanned"]["documents"] == 2 and body["slices"]["native"]["documents"] == 4
    assert body["slices"]["held_out_carrier"]["confidently_wrong"] == 0
    assert body["slices"]["seen_carrier"]["confidently_wrong"] == 1
    json.dumps(body)                                                     # written as JSON


def test_the_sweep_replays_one_reading_per_document():
    cases = _cases([("a", "homeowners", None, None), ("b", "homeowners", None, None)])
    reader = _ReadsBySource({"a": ("homeowners", 0.92), "b": ("dwelling_fire", 0.55)})
    body = _measure(cases, reader)

    assert reader.calls == 2                                             # once each, whatever the sweep
    assert body["slices"] == {}                                          # the metadata said nothing
    by_thresholds = {(row["line_threshold"], row["uncertain_threshold"]): row for row in body["sweep"]}
    assert all(uncertain <= line for line, uncertain in by_thresholds)
    assert by_thresholds[(0.90, 0.50)]["detected_share"] == 0.5
    assert by_thresholds[(0.90, 0.50)]["uncertain_share"] == 0.5
    assert by_thresholds[(0.95, 0.60)]["detected_share"] == 0.0
    assert by_thresholds[(0.95, 0.60)]["undetected_share"] == 0.5


def test_a_measurement_that_clears_the_floor_says_so():
    cases = _cases([(f"d{i}", "homeowners", None, None) for i in range(3)])
    body = _measure(cases, _ReadsBySource({f"d{i}": ("homeowners", 0.99) for i in range(3)}))
    assert body["overall"] == 1.0 and body["clears_floor"] is True


def test_a_classifier_that_reads_no_lines_cannot_be_measured():
    with pytest.raises(ValueError, match="reads lines"):
        _measure(_cases([("a", "homeowners", None, None)]), StaticClassifier("policy"))


def test_the_cases_are_the_golden_policies_with_one_known_line():
    from evaluation.golden_eval import GoldenDocument
    from evaluation.lob_detection_eval import detection_cases

    def doc(source, doc_type, lob, **kw):
        return GoldenDocument(source, doc_type, {}, [f"{source}/page_1.png", f"{source}/page_2.png"],
                              page_texts={1: "Declarations 2026 policy 12345", 2: "Schedule 678"},
                              lob=lob, **kw)

    documents = [doc("h", "policy", "homeowners", is_scanned=True), doc("c", "policy", "classic_auto"),
                 doc("pkg", "policy", ["homeowners", "gl"]), doc("l", "lossrun", None),
                 doc("x", "policy", "pet_insurance")]
    local = {key: f"/cache/{key}" for d in documents for key in d.image_keys}
    cases = detection_cases(documents, local)
    assert [(c.source_id, c.lob, c.is_scanned) for c in cases] == [
        ("h", "homeowners", True), ("c", "personal_auto", False)]
    request = cases[0].request
    assert (request.known_doc_type, request.known_lob, request.lob_hypothesis) == ("policy", None, None)
    assert request.image_paths == ["/cache/h/page_1.png", "/cache/h/page_2.png"]
    assert detection_cases(documents, local, mode="image_only")[0].request.page_texts == {}
    noisy = detection_cases(documents, local, mode="noisy_ocr_image")[0].request.page_texts
    assert noisy != documents[0].page_texts


def test_the_measurement_uses_the_endpoints_thresholds():
    from evaluation.lob_detection_eval import serving_lob_thresholds
    from serving.vllm_entrypoint import serving_thresholds

    tuning = serving_thresholds()
    assert serving_lob_thresholds() == (tuning["lob_confidence_threshold"], tuning["lob_uncertain_threshold"])
    assert "lob_family_threshold" not in tuning


def test_the_measurement_script_answers_help_without_loading_a_model():
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_lob_detection.py"
    done = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "--corpus" in done.stdout


# --------------------------------------------------------------------------
# L4: a classify row is what serving sends
# --------------------------------------------------------------------------

PAGES = ["Declarations page one, policy HO-12345, premium 1,234.00",
         "Schedule page two, dwelling limit 350,000 deductible 1,000",
         "Conditions page three"]


def _document():
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument

    return SourceDocument(source_id="h1", doc_type="policy", golden_label={}, ocr_pages=list(PAGES),
                          image_paths=[f"h1/page_{n}.png" for n in (1, 2, 3)], lob="homeowners",
                          tenant_id="default")


def _served_content(mode, pages):
    from serving.pipeline import ExtractionRequest, _classifier_input

    request = ExtractionRequest(source_id="h1", image_paths=[f"h1/page_{n}.png" for n in (1, 2, 3)],
                                page_texts=dict(enumerate(pages, start=1)), modality_mode=mode)
    return classifier_messages(*_classifier_input(request))[1]["content"]


@pytest.mark.parametrize("mode", ["ocr_plus_image", "image_only"])
def test_a_classify_rows_input_is_what_serving_sends(mode):
    """The rows trained page 1's text alone while serving sent pages 1 and 2."""
    from data_pipeline.dataset_builder.build_jsonl import classify_rows

    [row] = classify_rows(_document(), "train", (mode,))
    assert row["messages"][1]["content"] == _served_content(mode, PAGES)


def test_a_noisy_classify_row_reads_its_own_corrupted_text():
    """The noisy row was byte-identical to the clean one."""
    from data_pipeline.dataset_builder.build_jsonl import classify_rows
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    clean, noisy = classify_rows(_document(), "train", ("ocr_plus_image", "noisy_ocr_image"))
    assert noisy["messages"][1]["content"] != clean["messages"][1]["content"]
    opening, _details = corrupt_ocr_pages(PAGES[:2], "h1#classify", seed=42)
    assert noisy["messages"][1]["content"] == _served_content("noisy_ocr_image", [*opening, PAGES[2]])
    assert classify_rows(_document(), "train", ("noisy_ocr_image",)) == [{**noisy, "mode_index": 0}]


def test_the_corpus_refuses_classify_rows_until_serving_asks_their_question():
    from data_pipeline.dataset_builder.build_jsonl import CorpusBuildError, build_corpus

    with pytest.raises(CorpusBuildError, match="serving does not ask"):
        build_corpus([_document()], None, with_classify_rows=True)


# --------------------------------------------------------------------------
# L5: a package line is a line
# --------------------------------------------------------------------------

def test_the_prompt_reports_a_package_line_as_that_line():
    from common.prompts import render_classifier_prompt

    prompt = render_classifier_prompt()
    assert "`commercial_package`" in prompt and "`business_owners`" in prompt
    assert "Report null when the policy is a package" not in prompt
    assert "A package policy written on one of the lines above is reported as that line" in prompt
    assert "Report null only when no single line above describes the policy as a whole." in prompt


# --------------------------------------------------------------------------
# L6: the line hint arrives through the endpoint
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hint,expected", [
    (None, None),
    ("homeowners", "homeowners"),
    (" Classic_Auto ", "personal_auto"),          # read as personal auto, as `lob` is
    ("pet_insurance", "pet_insurance"),           # passed on: the reconciliation ignores it
])
def test_the_endpoint_reads_the_line_hint(hint, expected):
    from serving.vllm_entrypoint import build_request

    payload = {"source_id": "p1", "image_paths": ["page_1.png"], "page_texts": {"1": "Declarations"}, "doc_type": "policy"}
    if hint is not None:
        payload["lob_hypothesis"] = hint
    assert build_request(payload).lob_hypothesis == expected


@pytest.mark.parametrize("hint", ["", "  ", 5, ["homeowners"], {"lob": "homeowners"}])
def test_a_hint_that_is_no_line_name_is_refused(hint):
    from serving.vllm_entrypoint import ServingError, build_request

    with pytest.raises(ServingError, match="lob_hypothesis"):
        build_request({"source_id": "p1", "image_paths": ["page_1.png"], "page_texts": {"1": "Declarations"},
                       "lob_hypothesis": hint})


def test_an_endpoint_hint_that_agrees_lifts_an_uncertain_reading(client):
    from serving.pipeline import extract
    from serving.vllm_entrypoint import build_request
    from tests.test_lob_detection import _Reads
    from tests.test_serving_pipeline import CALIBRATION

    request = build_request({"source_id": "p1", "image_paths": ["d/page_1.png"],
                             "page_texts": {"1": "Declarations"}, "lob_hypothesis": "homeowners"})
    model, _backend = _model(client)
    result = extract(request, model, _Reads("homeowners", 0.7), CALIBRATION, strict_schema=False,
                     detect_lob=True)
    assert result.route_info["lob_source"] == "detected"
    assert result.line_of_business["hypothesis_agreed"] is True


# --------------------------------------------------------------------------
# L7: the prompt hash covers what the classifier is shown
# --------------------------------------------------------------------------

@pytest.mark.format_profile("standard")
def test_the_prompt_hash_covers_the_classifiers_line_meanings():
    from common.config import CONFIG_DIR
    from common.prompts import prompt_input_files

    files = prompt_input_files()
    sections = files.index(CONFIG_DIR / "schema_sections.yaml")
    assert files[sections + 1:] == [CONFIG_DIR / "lob_meanings.yaml"]
    assert not any(path.name == "layout_families.yaml" for path in files)       # no families any more


@pytest.mark.parametrize("name", ["lob_meanings.yaml", "schema_sections.yaml"])
def test_an_edit_to_a_classifier_config_moves_the_prompt_hash(name, monkeypatch):
    from common.config import CONFIG_DIR
    from common.prompts import prompt_hash

    before = prompt_hash()
    target = CONFIG_DIR / name
    read_bytes = Path.read_bytes

    def edited(path):
        data = read_bytes(path)
        return data + b"\n# edited\n" if path == target else data

    monkeypatch.setattr(Path, "read_bytes", edited)
    assert prompt_hash() != before
