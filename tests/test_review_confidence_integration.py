"""Serving's side of the calibration work (review 2026-10-26, W3b's steps):
the mapped spans reach the feature builder (r_written, top2_margin), each
field's feature vector is kept on the result - never served - and fields first
scores the fields again on the page text with the release calibrators phase 1
used, the values a review flag sent to review staying flagged."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from calibration.thresholds import FieldTypeThreshold, ThresholdSet

PDF = b"%PDF-1.7 stand-in"


class _Calibrators:
    """Confident when the value is printed on a page it cites; records what it read."""

    def __init__(self):
        self.read = []

    def predict(self, features):
        self.read.append(features)
        return 0.99 if features.ocr_agreement == 1.0 else 0.40


_THRESHOLDS = ThresholdSet("r", "bf16", {kind: FieldTypeThreshold(kind, 0.9, 0.01)
                                         for kind in ("identifier", "date", "money", "entity", "free_text",
                                                      "enum", "number", "address")})


def test_the_feature_builder_reads_the_spans_themselves():
    """Step 1: FieldSpans, not logprob lists - whether r was written, and the
    runner-up's margin, reach the features; step 2: kept by path."""
    from common.canonical import collapse_spans, expand_answer
    from inference_core.span_map import map_compact_field_spans
    from serving.pipeline import _feature_calibrated

    text = ('{"policy": {"policy_number": {"v": "HO-1", "p": [1]}, '
            '"effective_date": {"r": "April 1, 2026", "v": "04/01/2026", "p": [1]}}}')
    tokens = [text[i:i + 3] for i in range(0, len(text), 3)]
    spans = collapse_spans(map_compact_field_spans(text, tokens, [-0.01] * len(tokens),
                                                   token_top2_logprobs=[-0.06] * len(tokens)))
    calibrators = _Calibrators()
    result = _feature_calibrated(extraction=expand_answer(json.loads(text)), spans=spans, calibrators=calibrators,
                                 thresholds=_THRESHOLDS, page_text=None, common_model=True,
                                 modality_mode="image_only")

    read = {features.field_path: features for features in calibrators.read}
    assert read["policy.policy_number"].r_written is False             # r left out: v is its print
    assert read["policy.effective_date"].r_written is True
    assert read["policy.policy_number"].top2_margin == pytest.approx(0.05)
    assert read["policy.policy_number"].modality_mode == "image_only"
    assert set(result.features) == {"policy.policy_number", "policy.effective_date"}
    assert result.features["policy.effective_date"] is read["policy.effective_date"]


def _read_from_images(*, calibrators, thresholds, edit=None):
    """The 3-page auto fixture read from its page images alone (fields first's
    phase 1), by replaying its image-only training targets."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from inference_core.model_runner import load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_printed_values import _texts
    from tests.test_schema2_serving import _compact_auto, _full_answer, _Replay, _replay_key
    from tests.test_serving_pipeline import CALIBRATION

    gold = _compact_auto()
    texts = _texts(gold, 3)
    images = [f"processed/default/policy/pa_1/page_{p}.png" for p in (1, 2, 3)]
    doc = SourceDocument(source_id="pa_1", doc_type="policy", golden_label=copy.deepcopy(gold),
                         ocr_pages=[texts[p] for p in (1, 2, 3)], image_paths=images, lob="personal_auto",
                         tenant_id="default")
    rows, _ = expand_document(doc, "train", modes=("image_only",))
    answers = {}
    for row in rows:
        answer = _full_answer(row)
        answers[_replay_key(row)] = json.dumps(edit(answer) if edit else answer)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    request = ExtractionRequest(source_id="pa_1", image_paths=images, page_texts={}, modality_mode="image_only",
                                known_doc_type="policy", known_lob="personal_auto",
                                source_file_name="auto_fixture.pdf")
    result = extract(request, load_model("base", client, backend_impl=_Replay(answers)),
                     StaticClassifier("policy"), CALIBRATION, strict_schema=False,
                     calibrators=calibrators, thresholds=thresholds)
    return result, texts


class _Reader:
    """A served MinerU reading each page's given text."""

    def __init__(self, texts):
        self.texts = dict(texts)

    def ocr_text(self, pdf, *, name, pages=None, page_count=None, gate=None):
        return {"texts": {page: self.texts.get(page, "") for page in pages or sorted(self.texts)},
                "info": {"version": "x"}, "seconds": 0.0}


def _wrong_effective_date(answer):
    policy = answer.get("policy")
    if isinstance(policy, dict) and isinstance(policy.get("effective_date"), dict):
        policy["effective_date"].update(raw="10/01/2026", parsed="10/01/2026")
    return answer


def test_the_feature_vectors_are_kept_on_the_result_and_never_served():
    calibrators = _Calibrators()
    result, _texts = _read_from_images(calibrators=calibrators, thresholds=_THRESHOLDS)

    assert "policy.policy_number" in result.confidence_features
    assert result.confidence_features["policy.policy_number"].ocr_agreement is None   # no text yet
    assert result.calibrated_with.calibrators is calibrators and result.calibrated_with.thresholds is _THRESHOLDS
    served = result.as_dict()
    assert "confidence_features" not in served and "calibrated_with" not in served
    assert "confidence_features" not in json.dumps(served)
    # Read without text, the policy number is not confident and goes to review.
    assert result.extraction["policy"]["policy_number"]["flagged"] is True
    assert "policy.policy_number:low_confidence" in result.review_flags


def test_the_full_text_scores_the_fields_again_with_phase_1_s_calibrators(tmp_path):
    """Step 3: after the full text and the printed-values hold, the fields are
    scored again with the calibrators the result was read with; a value a
    review flag sent to review (here, under a label printing another date)
    stays flagged whatever its new confidence."""
    from serving.fields_first import add_full_text

    result, texts = _read_from_images(calibrators=_Calibrators(), thresholds=_THRESHOLDS,
                                      edit=_wrong_effective_date)
    add_full_text(PDF, result, name="scan.pdf", lob="personal_auto", reader=_Reader(texts),
                  pages_dir=tmp_path / "pages", page_count=3)

    number = result.extraction["policy"]["policy_number"]
    assert number["confidence"]["score"] == 0.99 and number["flagged"] is False
    assert "policy.policy_number:low_confidence" not in result.review_flags
    assert result.confidence_features["policy.policy_number"].ocr_agreement == 1.0
    assert "policy.effective_date:role_mismatch" in result.review_flags
    assert result.extraction["policy"]["effective_date"]["confidence"]["score"] == 0.99
    assert result.extraction["policy"]["effective_date"]["flagged"] is True


def test_the_fields_served_at_the_gate_are_scored_on_the_pages_they_cite():
    from serving.fields_first import verify_fields

    result, texts = _read_from_images(calibrators=_Calibrators(), thresholds=_THRESHOLDS)
    checked = verify_fields(PDF, result, name="scan.pdf", lob="personal_auto", reader=_Reader(texts),
                            page_count=3)

    assert checked["served"].extraction["policy"]["policy_number"]["flagged"] is False
    # The model's answer, kept for the full text, is as it was read.
    assert result.extraction["policy"]["policy_number"]["flagged"] is True


def test_a_result_restored_from_its_checkpoint_is_scored_with_the_calibrators_given(tmp_path):
    """The Azure worker checkpoints phase 1 as JSON (dataclasses.asdict): the
    feature vectors come back as dicts and the calibrators not at all - they
    are passed to add_full_text, and the vectors read back."""
    from serving.fields_first import add_full_text
    from serving.pipeline import ExtractionResult

    calibrators = _Calibrators()
    result, texts = _read_from_images(calibrators=calibrators, thresholds=_THRESHOLDS)
    answer = json.loads(json.dumps(dataclasses.asdict(result), default=str))
    names = {f.name for f in dataclasses.fields(ExtractionResult)}
    restored = ExtractionResult(**{key: value for key, value in answer.items() if key in names})
    assert isinstance(restored.confidence_features["policy.policy_number"], dict)

    add_full_text(PDF, restored, name="scan.pdf", lob="personal_auto", reader=_Reader(texts),
                  pages_dir=tmp_path / "pages", page_count=3)
    assert restored.extraction["policy"]["policy_number"]["flagged"] is True        # no calibrators: unchanged

    restored = ExtractionResult(**{key: value for key, value in answer.items() if key in names})
    add_full_text(PDF, restored, name="scan.pdf", lob="personal_auto", reader=_Reader(texts),
                  pages_dir=tmp_path / "pages2", page_count=3, calibrators=calibrators, thresholds=_THRESHOLDS)
    assert restored.extraction["policy"]["policy_number"]["flagged"] is False


def test_without_calibrators_the_full_text_changes_no_confidence(tmp_path):
    from serving.fields_first import add_full_text

    result, texts = _read_from_images(calibrators=None, thresholds=None)
    assert result.calibrated_with is None and result.confidence_features == {}
    before = result.extraction["policy"]["policy_number"]["confidence"]
    add_full_text(PDF, result, name="scan.pdf", lob="personal_auto", reader=_Reader(texts),
                  pages_dir=tmp_path / "pages", page_count=3)
    assert result.extraction["policy"]["policy_number"]["confidence"] == before
