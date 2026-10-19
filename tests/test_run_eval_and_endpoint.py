"""IMPL-08 §3 and IMPL-11 §1 — the eval driver and the serving endpoint.

The two assertions here guard failures that leave no trace anywhere else.

**Eval-set leakage.** Train on the eval set and every number in the registry
becomes a measurement of memorisation: the loss curve looks healthy, the metrics
improve, and the promotion gate passes. Nothing in the pipeline says otherwise.

**PII in a serving log.** A handler that logs its request for debugging has
exported every name, TIN and address in the document, and it does so silently and
continuously rather than once.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.run_eval import (
    EVAL_SUBSETS,
    EvalSetLeakage,
    assert_eval_set_disjoint,
    build_report,
    subset_of,
)
from serving.vllm_entrypoint import (
    ColdStartError,
    EndpointState,
    ServingError,
    assert_calibration_present,
    assert_ocr_pin,
    build_request,
    handler,
    safe_log_payload,
)

GOLDEN = {
    "insured_name": "Rivera Fabrication LLC",
    "policy_number": "WC-8842317-01",
    "line_of_business": ["workers_comp"],
}


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def seed_corpus_rows(client: BlobClient, version: str, source_ids: list[str]) -> None:
    rows = "".join(
        json.dumps({"source_id": sid, "doc_type": "policy"}) + "\n" for sid in source_ids
    )
    client.write_text(paths.corpus_split(version, "policy", "train"), rows)


def seed_eval_set(client: BlobClient, source_ids: list[str]) -> None:
    for source_id in source_ids:
        client.write_json(f"{paths.golden_eval_set_dir()}/{source_id}/golden.json", GOLDEN)


# --------------------------------------------------------------------------
# The freeze guarantee
# --------------------------------------------------------------------------


def test_a_disjoint_eval_set_passes(client):
    seed_corpus_rows(client, "v1", ["policy_0001", "policy_0002"])
    seed_eval_set(client, ["policy_9001"])
    assert_eval_set_disjoint(client, "v1")  # does not raise


def test_an_eval_document_inside_the_corpus_fails_loudly(client):
    seed_corpus_rows(client, "v1", ["policy_0001", "policy_9001"])
    seed_eval_set(client, ["policy_9001"])

    with pytest.raises(EvalSetLeakage, match="policy_9001"):
        assert_eval_set_disjoint(client, "v1")


def test_the_leakage_message_says_which_side_to_change(client):
    """The eval set is the thing held constant across versions, so it is the
    corpus that changes."""
    seed_corpus_rows(client, "v1", ["policy_9001"])
    seed_eval_set(client, ["policy_9001"])

    with pytest.raises(EvalSetLeakage) as excinfo:
        assert_eval_set_disjoint(client, "v1")
    assert "Remove them from the corpus" in str(excinfo.value)
    assert "memorisation" in str(excinfo.value)


def test_leakage_is_its_own_error_type():
    """It invalidates results that already exist rather than blocking a run:
    every metric measured against a leaked eval set has to be discarded."""
    from evaluation.run_eval import EvalError

    assert issubclass(EvalSetLeakage, EvalError)


def test_an_eval_set_frozen_from_the_corpus_test_split_is_not_leakage(client):
    """Build v1, freeze v1's test split, gate the model trained on v1: the frozen
    documents are in v1's TEST split, which no model trains or selects on."""
    seed_corpus_rows(client, "v1", ["policy_0001"])
    client.write_text(paths.corpus_eval_split("v1", "test"),
                      json.dumps({"source_id": "policy_9001", "doc_type": "policy"}) + "\n")
    seed_eval_set(client, ["policy_9001"])
    assert_eval_set_disjoint(client, "v1")  # does not raise


def test_an_eval_document_in_the_validation_split_is_leakage(client):
    """Validation selects the checkpoint and fits calibration: overlap there is real."""
    client.write_text(paths.corpus_eval_split("v1", "val"),
                      json.dumps({"source_id": "policy_9001", "doc_type": "policy"}) + "\n")
    seed_eval_set(client, ["policy_9001"])
    with pytest.raises(EvalSetLeakage, match="policy_9001"):
        assert_eval_set_disjoint(client, "v1")


def test_an_empty_eval_set_does_not_pass_by_being_empty(client):
    """Nothing to overlap is not the same as verified disjoint — but it is also
    not a leak, so this stays quiet and the coverage warning lives elsewhere."""
    seed_corpus_rows(client, "v1", ["policy_0001"])
    assert_eval_set_disjoint(client, "v1")


# --------------------------------------------------------------------------
# Subset breakdown
# --------------------------------------------------------------------------


def test_a_document_can_belong_to_several_subsets():
    subsets = subset_of({
        "doc_type": "policy", "modality_mode": "image_only",
        "is_scanned": True, "page_count": 40,
    })
    assert set(subsets) == {"full", "image_only", "scanned", "long_policy"}


def test_a_short_digital_document_is_only_in_the_full_set():
    assert subset_of({"doc_type": "acord", "modality_mode": "ocr_plus_image", "page_count": 1}) == ["full"]


def test_every_specified_subset_is_reachable():
    assert set(EVAL_SUBSETS) == {"image_only", "scanned", "noisy_ocr", "long_policy"}


def test_the_report_breaks_down_per_doc_type_and_per_subset():
    documents = [
        (GOLDEN, GOLDEN, {"source_id": "p1", "doc_type": "policy",
                          "modality_mode": "ocr_plus_image", "page_count": 2}),
        (GOLDEN, GOLDEN, {"source_id": "p2", "doc_type": "policy",
                          "modality_mode": "image_only", "page_count": 2}),
        (GOLDEN, GOLDEN, {"source_id": "a1", "doc_type": "acord", "acord_form": "25",
                          "modality_mode": "ocr_plus_image", "page_count": 1}),
    ]
    report = build_report("v2", documents, corpus_version="v1")

    assert {s.subset for s in report.for_doc_type("policy")} == {"full", "image_only"}
    assert [s.subset for s in report.for_doc_type("acord")] == ["full"]


def test_an_image_only_regression_is_visible_as_its_own_gate_metric():
    """It can fall ten points while the overall number moves two, because
    image-only is a third of the corpus."""
    wrong = {**GOLDEN, "insured_name": "Someone Else"}
    documents = [
        (GOLDEN, GOLDEN, {"source_id": "p1", "doc_type": "lossrun",
                          "modality_mode": "ocr_plus_image", "page_count": 1}),
        (GOLDEN, wrong, {"source_id": "p2", "doc_type": "lossrun",
                         "modality_mode": "image_only", "page_count": 1}),
    ]
    metrics = build_report("v2", documents).gate_metrics()

    assert metrics["image_only_accuracy"] < metrics["field_normalized_match"]


def test_each_document_s_own_score_is_kept_for_the_gate_s_bootstrap():
    """The paired bootstrap needs per-document scores; read in one mode, their
    mean is the gate metric itself."""
    wrong = {**GOLDEN, "insured_name": "Someone Else"}
    documents = [
        (GOLDEN, GOLDEN if i % 2 else wrong,
         {"source_id": f"p{i}", "doc_type": "lossrun", "modality_mode": "ocr_plus_image",
          "page_count": 1})
        for i in range(6)
    ] + [
        (GOLDEN, wrong, {"source_id": "p0", "doc_type": "lossrun",
                         "modality_mode": "image_only", "page_count": 1}),
    ]
    report = build_report("v2", documents)
    per_document = report.per_document_scores()
    metrics = report.gate_metrics()

    assert set(per_document["field_normalized_match"]) == {f"p{i}" for i in range(6)}
    assert per_document["image_only_accuracy"] == {"p0": pytest.approx(2 / 3)}
    ocr_only = build_report("v2", documents[:6])
    scores = ocr_only.per_document_scores()["field_normalized_match"]
    assert sum(scores.values()) / len(scores) == pytest.approx(
        ocr_only.gate_metrics()["field_normalized_match"])
    assert metrics["image_only_accuracy"] == pytest.approx(2 / 3)
    assert report.as_dict()["per_document"]["field_normalized_match"]["p1"] == 1.0
    assert "per_document" not in report.full_set()[0].as_dict()


def test_error_records_are_kept_not_just_aggregates():
    """vit_gate needs the perception-vs-reasoning split, and failure-mode
    analysis needs real material."""
    wrong = {**GOLDEN, "policy_number": "WC-8842317-O1"}
    report = build_report("v2", [
        (GOLDEN, wrong, {"source_id": "p1", "doc_type": "lossrun",
                         "modality_mode": "ocr_plus_image", "page_count": 1}),
    ])
    records = report.full_set()[0].error_records
    assert records and records[0]["field_path"] == "policy_number"
    assert records[0]["error_class"]
    assert records[0]["source_id"] == "p1"


def test_classifier_accuracy_is_absent_rather_than_zero_when_unscored():
    """A zero would read as a measured catastrophe; the gate treats an absent
    metric as not-passed, which is the right answer for one nobody measured."""
    report = build_report("v2", [
        (GOLDEN, GOLDEN, {"source_id": "p1", "doc_type": "policy",
                          "modality_mode": "ocr_plus_image", "page_count": 1}),
    ])
    assert "doc_type_classifier_accuracy" not in report.gate_metrics()


def test_gate_metrics_are_document_weighted_across_doc_types():
    """A type with three eval documents must not carry the weight of one with
    thirty."""
    wrong = {**GOLDEN, "insured_name": "Wrong"}
    documents = [
        (GOLDEN, GOLDEN, {"source_id": f"p{i}", "doc_type": "lossrun",
                          "modality_mode": "ocr_plus_image", "page_count": 1})
        for i in range(9)
    ] + [
        (GOLDEN, wrong, {"source_id": "a1", "doc_type": "acord", "acord_form": "25",
                         "modality_mode": "ocr_plus_image", "page_count": 1}),
    ]
    metrics = build_report("v2", documents).gate_metrics()
    # 9 perfect policy docs and 1 acord doc scoring 2/3 -> much closer to 1.0
    # than the 0.83 an unweighted mean of the two doc types would give.
    assert metrics["field_normalized_match"] > 0.95


def test_reports_are_written_per_doc_type_plus_a_summary(client):
    from evaluation.run_eval import write_report

    report = build_report("v2", [
        (GOLDEN, GOLDEN, {"source_id": "p1", "doc_type": "policy",
                          "modality_mode": "ocr_plus_image", "page_count": 1}),
    ])
    written = write_report(report, client)
    assert any(key.endswith("summary.json") for key in written)
    assert any("policy" in key for key in written)


# --------------------------------------------------------------------------
# The serving endpoint
# --------------------------------------------------------------------------


def test_a_corpus_with_no_ocr_pin_fails_the_cold_start():
    """Serving a model whose training-time OCR version is unknown is the arch
    §8a distribution-shift risk with no way to detect it."""
    with pytest.raises(ColdStartError, match="no mineru_version"):
        assert_ocr_pin({})


def test_the_pin_check_is_a_cold_start_error_not_a_request_error():
    """The wrong OCR version is wrong for every request, so a cold start that
    fails must not fall back to serving anyway."""
    assert issubclass(ColdStartError, ServingError)


def test_missing_calibration_raises_rather_than_serving_raw_confidence():
    with pytest.raises(ServingError, match="overconfident"):
        assert_calibration_present(None, "v2", "policy")


def test_a_request_without_images_is_refused():
    """Both production modes need the image: image_only has nothing else, and
    ocr_plus_image arbitrates between the two."""
    with pytest.raises(ServingError, match="no page images"):
        build_request({"source_id": "p1", "image_paths": []})


def test_a_request_without_a_source_id_is_refused():
    with pytest.raises(ServingError, match="source_id"):
        build_request({"image_paths": ["page_1.png"]})


def test_image_only_with_ocr_attached_is_refused():
    """The model would be told no text exists while text is present — a state it
    was never trained on."""
    with pytest.raises(ServingError, match="image_only"):
        build_request({
            "source_id": "p1", "image_paths": ["page_1.png"],
            "modality_mode": "image_only", "ocr_text": "Named Insured: Rivera",
        })


def test_a_valid_request_becomes_an_extraction_request():
    request = build_request({
        "source_id": "policy_0001",
        "image_paths": ["processed/default/policy/policy_0001/page_1.png"],
        "ocr_text": "Named Insured: Rivera Fabrication LLC",
        "page_texts": {"1": "Named Insured: Rivera"},
        "doc_type": "policy",
    })
    assert request.source_id == "policy_0001"
    assert request.page_texts == {1: "Named Insured: Rivera"}
    assert request.known_doc_type == "policy"


def test_document_content_never_reaches_a_log_line():
    payload = {
        "source_id": "policy_0001",
        "ocr_text": "Named Insured: Rivera Fabrication LLC, 1420 Foundry Road",
        "page_texts": {"1": "TIN 12-3456789"},
        "modality_mode": "ocr_plus_image",
        "image_paths": ["page_1.png"],
    }
    logged = json.dumps(safe_log_payload(payload))

    assert "Rivera" not in logged
    assert "Foundry" not in logged
    assert "12-3456789" not in logged
    assert "policy_0001" in logged        # the id is how a request is traced
    assert "ocr_plus_image" in logged     # shape is debuggable, content is not


def test_a_cold_endpoint_reports_retryable_rather_than_serving():
    response = handler({"input": {"source_id": "p1"}}, EndpointState(model_version="v2"))
    assert response["retryable"] is True
    assert "output" not in response


def test_a_bad_request_returns_a_structured_error_not_an_opaque_crash():
    """A serverless handler that raises returns a platform error, and the caller
    learns nothing about which of its inputs was wrong."""
    state = EndpointState(model_version="v2", ready=True, calibration=object())
    response = handler({"input": {"source_id": "p1", "image_paths": []}}, state)

    assert response["retryable"] is False
    assert "no page images" in response["error"]


def test_the_error_response_does_not_echo_the_request():
    state = EndpointState(model_version="v2", ready=True, calibration=object())
    response = handler({"input": {
        "image_paths": ["page_1.png"], "ocr_text": "Named Insured: Rivera Fabrication LLC",
    }}, state)

    assert "Rivera" not in json.dumps(response)


def test_an_acord_document_with_no_form_is_counted_invalid_not_crashed():
    """The form is what selects the schema, so a formless ACORD cannot validate.
    Letting that escape would discard every other document's score with it."""
    report = build_report("v2", [
        (GOLDEN, GOLDEN, {"source_id": "a1", "doc_type": "acord",
                          "modality_mode": "ocr_plus_image", "page_count": 1}),
    ])
    assert report.full_set()[0].metrics["schema_validity_rate"] == 0.0


def test_subsets_are_assigned_from_document_metadata():
    assert subset_of({"modality_mode": "image_only"}) == ["full", "image_only"]
    assert "scanned" in subset_of({"modality_mode": "ocr_plus_image", "is_scanned": True})
    assert "noisy_ocr" in subset_of({"modality_mode": "noisy_ocr_image"})
    assert "long_policy" in subset_of({"doc_type": "policy", "page_count": 12})
    assert subset_of({"doc_type": "policy", "page_count": 3}) == ["full"]


def test_the_entrypoint_does_not_reimplement_generation():
    """Generation logic here would be a second implementation, and the registry
    numbers would then describe the wrong one."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "serving" / "vllm_entrypoint.py"
    ).read_text(encoding="utf-8")

    assert "from serving.pipeline import" in source
    for reimplemented in ("map_field_spans", "field_confidences", "build_messages", "json.loads("):
        assert reimplemented not in source, f"{reimplemented} is reimplemented in the entrypoint"
