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
# The yardstick an eval report records
# --------------------------------------------------------------------------


def test_two_reports_are_one_yardstick_only_when_set_tenant_and_scoring_code_agree():
    """The identity was (tenant, frozen_at, corpus): two reports of one frozen
    gold read under different label rules - the edition rule on every gold, or
    only where the rules hold - paired as one yardstick."""
    from evaluation.run_eval import EVAL_SET_IDENTITY, yardstick_difference

    assert "scoring_fingerprint" in EVAL_SET_IDENTITY
    record = {"tenant_id": "default", "frozen_at": "2026-10-01T00:00:00+00:00",
              "frozen_from_corpus": "v1", "scoring_fingerprint": "0123456789abcdef"}
    assert yardstick_difference(record, dict(record), None) is None
    assert "frozen again" in yardstick_difference({**record, "frozen_at": "2026-09-01"}, record, None)
    assert "tenant 'acme'" in yardstick_difference({**record, "tenant_id": "acme"}, record, None)
    assert "gating tenant 'acme'" in yardstick_difference(record, record, "acme")
    assert "different scoring code" in yardstick_difference(
        {**record, "scoring_fingerprint": "fedcba9876543210"}, record, None)
    before_fingerprints = {k: v for k, v in record.items() if k != "scoring_fingerprint"}
    assert "scoring_fingerprint missing" in yardstick_difference(before_fingerprints, record, None)
    assert "promoted report does not record" in yardstick_difference(None, record, None)


def test_a_label_rule_change_moves_the_fingerprint_and_a_comment_does_not(tmp_path):
    """Hashed from the syntax tree: a comment, a docstring or a CRLF checkout
    cannot change a score, so they leave the fingerprint alone; a change to the
    rule itself does not."""
    from evaluation.run_eval import code_fingerprint

    rules = tmp_path / "label_rules.py"
    rules.write_bytes(b'def applies(doc_type, lob):\n    return lob != "commercial_auto"\n')
    before = code_fingerprint(tmp_path, ["label_rules.py"])
    rules.write_bytes(b'# scoped to the policy lines\r\ndef applies(doc_type, lob):\r\n'
                      b'    """Where the edition rule holds."""\r\n\r\n'
                      b'    return lob != "commercial_auto"  # self-contained lines\r\n')
    assert code_fingerprint(tmp_path, ["label_rules.py"]) == before
    rules.write_bytes(b"def applies(doc_type, lob):\n    return True\n")
    assert code_fingerprint(tmp_path, ["label_rules.py"]) != before
    rules.unlink()
    assert code_fingerprint(tmp_path, ["label_rules.py"]) != before


def test_the_scoring_fingerprint_covers_real_modules():
    """A module renamed or moved would quietly stop being part of the yardstick."""
    from pathlib import Path

    from evaluation.run_eval import SCORING_CODE, scoring_fingerprint

    root = Path(__file__).resolve().parent.parent
    assert [p for p in SCORING_CODE if not list(root.glob(p))] == []
    assert {"data_pipeline/ingestion/label_rules.py", "evaluation/metrics/*.py",
            "common/normalize.py"} <= set(SCORING_CODE)
    assert len(scoring_fingerprint()) == 16


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


def _score_lists_as(monkeypatch, recalls: dict[str, list[float]]) -> None:
    """Score each document's list fields as ``recalls`` says, looked up by its
    insured name: one table per entry, so the number of tables is chosen too."""
    from evaluation.metrics import field_accuracy
    from evaluation.metrics.field_accuracy import ListFieldReport

    def score_all_list_fields(expected, got, *, common_model=False):
        return {
            f"table_{i}": ListFieldReport(f"table_{i}", expected_rows=20,
                                          got_rows=round(r * 20), matched_rows=round(r * 20))
            for i, r in enumerate(recalls[expected["insured_name"]])
        }

    monkeypatch.setattr(field_accuracy, "score_all_list_fields", score_all_list_fields)


def _read_in_modes(names: list[str], doc_type: str, modes: tuple[str, ...], **meta) -> list:
    return [
        ({**GOLDEN, "insured_name": name}, {**GOLDEN, "insured_name": name},
         {"source_id": f"{doc_type}_{i:02d}", "doc_type": doc_type, "modality_mode": mode,
          "page_count": 1, **meta})
        for i, name in enumerate(names) for mode in modes
    ]


def _weighted_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    return sum(scores[d] * weights[d] for d in scores) / sum(weights[d] for d in scores)


def _paired_from(current: dict, candidate: dict, name: str) -> tuple[list, list, list, list]:
    """One metric paired from two written reports, as the gate stage pairs them."""
    ids = sorted(current["per_document"][name])
    return (
        [current["per_document"][name][d] for d in ids],
        [candidate["per_document"][name][d] for d in ids],
        [current["per_document_weights"][name][d] for d in ids],
        [candidate["per_document_weights"][name][d] for d in ids],
    )


def test_list_recall_is_bootstrapped_as_the_gate_reports_it(monkeypatch):
    """The review's case: 15 documents with one table and 15 with four, each read
    in three modes. The baseline recalls 0.6 on the one-table documents and 1.0
    on the four-table ones; the candidate 1.0 and 0.85. Reported recall - a mean
    over every table, the number the floor reads - falls 0.92 -> 0.88. The
    bootstrap averaged documents instead, saw +0.125, and called it improved."""
    from evaluation.bootstrap import paired_bootstrap
    from evaluation.gating import GATING_METRICS, promotion_gate

    names = [f"Insured {i:02d}" for i in range(30)]
    documents = _read_in_modes(names, "lossrun", ("ocr_plus_image", "noisy_ocr_image", "image_only"))
    _score_lists_as(monkeypatch, {n: [0.6] if i < 15 else [1.0] * 4 for i, n in enumerate(names)})
    current = build_report("v1", documents)
    _score_lists_as(monkeypatch, {n: [1.0] if i < 15 else [0.85] * 4 for i, n in enumerate(names)})
    candidate = build_report("v2", documents)

    reported = {r.model_version: r.gate_metrics()["list_field_recall"] for r in (current, candidate)}
    assert reported == {"v1": pytest.approx(0.92), "v2": pytest.approx(0.88)}
    for report in (current, candidate):
        scores = report.per_document_scores()["list_field_recall"]
        weights = report.per_document_weights()["list_field_recall"]
        assert _weighted_mean(scores, weights) == pytest.approx(reported[report.model_version])

    paired = _paired_from(current.as_dict(), candidate.as_dict(), "list_field_recall")
    ci = paired_bootstrap("list_field_recall", paired[0], paired[1],
                          current_weights=paired[2], candidate_weights=paired[3])
    assert ci.observed == pytest.approx(-0.04, abs=1e-4), "the bootstrap tests the reported delta"
    assert not ci.improved and not ci.non_inferior(0.010)
    # What the unweighted per-document mean said: the opposite verdict.
    unweighted = paired_bootstrap("list_field_recall", paired[0], paired[1])
    assert unweighted.observed == pytest.approx(0.125, abs=1e-4) and unweighted.improved

    metrics = {name: 0.96 for name in GATING_METRICS}
    metrics.update(schema_validity_rate=1.0, ece_confidence=0.04, confusable_misattribution_rate=0.02,
                   false_null_rate=0.03, auto_accept_error_rate=0.01)
    result = promotion_gate(
        {**metrics, "list_field_recall": reported["v2"]},
        {**metrics, "list_field_recall": reported["v1"]},
        per_document={"list_field_recall": paired},
    )
    verdict = next(v for v in result.verdicts if v.name == "list_field_recall")
    assert verdict.meets_floor and not verdict.non_inferior and not verdict.improved
    assert "list_field_recall" in result.failed_gates


def test_each_per_document_metric_weighted_is_the_metric_the_gate_reports(monkeypatch):
    """Field match, exact match, schema validity, list recall and the subset
    accuracies: each weighted mean of per-document scores is the gate metric,
    across two doc types read in different numbers of modes and holding
    different numbers of tables."""
    lossrun = [f"Loss Run {i}" for i in range(4)]
    acord = [f"Certificate {i}" for i in range(3)]
    documents = _read_in_modes(lossrun, "lossrun", ("ocr_plus_image", "noisy_ocr_image", "image_only"))
    documents += _read_in_modes(acord, "acord", ("ocr_plus_image", "noisy_ocr_image"), acord_form="25")
    # Some readings wrong, so the documents' field match differ.
    for index in (1, 4, 5, 13):
        expected, got, meta = documents[index]
        documents[index] = (expected, {**got, "policy_number": "WRONG-1"}, meta)
    _score_lists_as(monkeypatch, {
        **{n: [0.5] * (i + 1) for i, n in enumerate(lossrun)},
        **{n: [0.25, 1.0][: 1 + (i == 2)] for i, n in enumerate(acord)},
    })
    report = build_report("v2", documents)
    metrics, scores, weights = report.gate_metrics(), report.per_document_scores(), report.per_document_weights()

    assert set(scores) >= {"field_normalized_match", "field_exact_match", "list_field_recall",
                           "schema_validity_rate", "image_only_accuracy", "ocr_arbitration_accuracy"}
    for name, by_source in scores.items():
        assert set(weights[name]) == set(by_source), name
        assert _weighted_mean(by_source, weights[name]) == pytest.approx(metrics[name]), name
    written = report.as_dict()
    assert set(written["per_document_weights"]) == set(written["per_document"])


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
