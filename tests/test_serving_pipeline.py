"""IMPL-10 + IMPL-11 + IMPL-12 — routing, page handling, the pipeline, and parity.

The parity test at the end is one of the four that guard silent failures: if the
testing harness ever stops calling the serving pipeline, its numbers describe a
system that is not the one in production, and no other test would notice.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from inference_core.model_runner import EchoBackend, load_model
from postprocessing.quantize import QuantizationError, plan_quantization
from serving.adapter_router import route
from serving.doc_type_classifier import (
    Classification,
    ClassificationError,
    StaticClassifier,
    parse_classification,
)
from serving.page_router import merge_page_extractions, plan_pages
from serving.pipeline import ExtractionRequest, PipelineError, extract
from testing.run_extraction import run_document, summarise
from training.merge import plan_merge


def _fv(raw, parsed=None, page=1):
    """One canonical leaf as the model writes it: raw, parsed, page_ref."""
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [page]}


#: A policy in the model's canonical form: the client's tree, sparse, each leaf
#: a raw/parsed/page_ref envelope. The pipeline adds confidence and flagged.
GOLDEN = {
    "carrier": {"company_name": _fv("Granite Mutual Insurance Co")},
    "named_insured": {"primary_name": _fv("Rivera Fabrication LLC")},
    "policy": {
        "policy_number": _fv("WC-8842317-01"),
        "effective_date": _fv("04/01/2026", "04/01/2026"),
    },
}
RESPONSE = json.dumps(GOLDEN)
CALIBRATION = CalibrationParams(
    method="temperature", doc_type="policy", model_version="v1", temperature=1.0
)

#: The line GOLDEN is read as. GOLDEN is in the self-contained shape
#: (carrier.company_name); the fallback and general liability compose the
#: common model since 1.1.0, so the pipeline's generic behaviour is checked on a
#: line still self-contained, and in no family part-way through that move.
LINE = "property"

#: The page GOLDEN is read from. It prints every value the fake model writes:
#: a served policy holds no value its pages do not print (serving.printed_values).
PAGE_TEXT = ("**Applicant** Rivera Fabrication LLC\nGranite Mutual Insurance Co\n"
             "Policy Number WC-8842317-01  Effective 04/01/2026")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def model(client):
    return load_model("base", client, backend_impl=EchoBackend(RESPONSE))


def _request(**over) -> ExtractionRequest:
    base = dict(
        source_id="policy_0001",
        image_paths=["processed/default/policy/policy_0001/page_1.png"],
        ocr_text=PAGE_TEXT,
        known_doc_type="policy",
        known_lob=LINE,
    )
    base.update(over)
    return ExtractionRequest(**base)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def test_classifier_response_is_parsed():
    result = parse_classification('{"doc_type":"lossrun","acord_form":null,"confidence":0.93}')
    assert result.doc_type == "lossrun" and result.confidence == 0.93


def test_fenced_json_is_recovered():
    """A parse failure would route a classifiable document to review for no
    reason."""
    result = parse_classification('```json\n{"doc_type":"policy","confidence":0.8}\n```')
    assert result.doc_type == "policy"


def test_unknown_doc_type_is_discarded():
    assert parse_classification('{"doc_type":"invoice","confidence":0.99}').doc_type is None


def test_acord_without_a_form_is_unusable_however_confident():
    """No form means no schema can be selected — confidence is irrelevant."""
    result = parse_classification('{"doc_type":"acord","acord_form":null,"confidence":0.99}')
    assert result.confidence == 0.0


def test_unparseable_response_raises():
    with pytest.raises(ClassificationError, match="not JSON"):
        parse_classification("I think this is a policy document.")


# --------------------------------------------------------------------------
# Routing — the low-confidence fallback
# --------------------------------------------------------------------------

def test_confident_classification_selects_the_adapter():
    result = route(
        Classification("lossrun", None, 0.95),
        adapter_map={"lossrun": "adapters/lossrun/v2"},
    )
    assert result.adapter == "adapters/lossrun/v2"
    assert not result.foundation_only
    assert not result.review_flags


def test_low_confidence_extracts_for_both_candidates():
    """v1 fell back to ONE guessed schema with a flag. That is worse than it
    sounds: a wrong-schema extraction is structurally valid and confidently
    wrong, so it passes the audit gate and reaches a user looking correct. Two
    answers plus the classification scores is what a router can actually review
    (arch v2.1 §4a)."""
    result = route(
        Classification("lossrun", None, 0.40, candidates=[("lossrun", 0.40), ("policy", 0.35)]),
        confidence_threshold=0.70,
        adapter_map={},
    )
    assert result.is_ambiguous
    assert [r.doc_type for r in result.candidate_routes] == ["lossrun", "policy"]
    assert all(r.schema_doc_type == r.doc_type for r in result.candidate_routes),         "each candidate is validated against its OWN schema"
    assert "routing:low_confidence" in result.review_flags
    assert result.needs_routing_review


def test_an_acord_candidate_without_a_form_is_not_offered():
    """It has no schema to validate against, so it cannot be one of the two
    answers presented for review."""
    result = route(
        Classification("acord", None, 0.40, candidates=[("acord", 0.40), ("policy", 0.38)]),
        confidence_threshold=0.70,
        fallback_doc_type="policy",
    )
    assert "acord" not in [r.doc_type for r in result.candidate_routes]


def test_a_confident_l1_l2_agreement_clears_the_threshold():
    """Two independent signals agreeing is worth more than either alone, and
    this is the case that lets a modest model confidence route cleanly."""
    from serving.doc_type_classifier import combine_with_hypothesis

    agreed = combine_with_hypothesis(Classification("policy", None, 0.55), "policy")
    assert agreed.hypothesis_agreed is True
    assert route(agreed).adapter is None and not route(agreed).is_ambiguous


def test_a_confident_model_overrides_the_l1_l2_hypothesis():
    """L1 is a carrier registry lookup and L2 is structural inference. Neither
    has seen the page."""
    from serving.doc_type_classifier import combine_with_hypothesis

    result = combine_with_hypothesis(Classification("lossrun", None, 0.95), "policy")
    assert result.doc_type == "lossrun"
    assert result.hypothesis_agreed is False


def test_low_confidence_still_keeps_the_best_guess_schema():
    """A generic extraction validated against a plausible schema beats one
    validated against nothing."""
    result = route(Classification("lossrun", None, 0.40), confidence_threshold=0.70)
    assert result.schema_doc_type == "lossrun"


def test_an_unclassifiable_document_is_refused_when_no_fallback_is_configured():
    """The fallback used to be "policy" by default, so an unidentifiable document
    was extracted against the policy schema whatever the deployment served — and
    in a deployment with no policy release, by a model that never saw one."""
    from serving.adapter_router import RoutingError

    with pytest.raises(RoutingError, match="could not be classified"):
        route(Classification(None, None, 0.0))

    with pytest.raises(RoutingError, match="ACORD form number is missing"):
        route(Classification("acord", None, 0.99))


def test_a_configured_fallback_still_gets_a_flag_rather_than_a_refusal():
    """An operator may still choose a fallback schema. Choosing it is the point:
    the flag then records that it was a fallback rather than a decision."""
    result = route(Classification(None, None, 0.0), fallback_doc_type="policy")
    assert result.foundation_only
    assert result.schema_doc_type == "policy"
    assert "routing:unclassified" in result.review_flags


def test_no_graduated_adapter_is_the_normal_path_not_a_flagged_one():
    """Under arch v2.1 §4.1 the merged unified model serves every type; a
    per-type adapter exists only after §4.2 graduation. Flagging the EXPECTED
    path would put every document in the review queue and teach everyone to
    ignore the flag."""
    result = route(Classification("policy", None, 0.99), adapter_map={})
    assert result.adapter is None
    assert result.foundation_only
    assert not result.review_flags, result.review_flags
    assert not result.needs_routing_review


def test_acord_form_reaches_the_schema_selection():
    result = route(Classification("acord", "25", 0.95), adapter_map={"acord": "a/25"})
    assert result.schema_acord_form == "25"


# --------------------------------------------------------------------------
# Pages read
# --------------------------------------------------------------------------

def test_every_page_of_a_long_document_is_read():
    """Every page, chunk by chunk, never selected pages (decision 2026-10-19):
    a page the section signals miss is read like any other."""
    pages = {i: "boilerplate exclusions and definitions" for i in range(1, 21)}
    pages[3] = "COMMON POLICY DECLARATIONS named insured policy number policy period"
    pages[7] = "COVERAGE SCHEDULE limits of insurance premium"

    plan = plan_pages(pages)
    assert plan.pages == list(range(1, 21))
    assert plan.declarations_page == 3


def test_a_layout_the_signals_do_not_recognise_is_read_in_full():
    plan = plan_pages({i: "unrecognisable layout" for i in range(1, 11)})
    assert plan.pages == list(range(1, 11))
    assert plan.declarations_page is None


def test_declarations_page_wins_policy_level_conflicts():
    """arch §7 — a schedule page can repeat a policy number differently; the
    declarations page is the authority."""
    merged = merge_page_extractions(
        {3: {"policy_number": "WC-8842317-01"}, 7: {"policy_number": "WC-8842317"}},
        declarations_page=3,
    )
    assert merged.extraction["policy_number"] == "WC-8842317-01"
    assert merged.conflicts


def test_list_fields_concatenate_across_pages():
    """A claims table spanning three pages must produce all its rows —
    overwriting would silently drop two thirds of them."""
    merged = merge_page_extractions({
        1: {"claims": [{"claim_number": "C1"}]},
        2: {"claims": [{"claim_number": "C2"}, {"claim_number": "C3"}]},
    })
    assert len(merged.extraction["claims"]) == 3


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def test_pipeline_produces_the_output_contract(model):
    """A policy comes back as the client's canonical JSON: every leaf a full
    FieldValue envelope, confidence and flagged filled by the pipeline."""
    result = extract(_request(), model, StaticClassifier("policy"), CALIBRATION)
    assert result.schema_valid
    assert result.doc_type == "policy"

    leaf = result.extraction["named_insured"]["primary_name"]
    assert set(leaf) == {"raw", "parsed", "confidence", "page_ref", "flagged"}
    assert leaf["parsed"] == "Rivera Fabrication LLC"
    assert leaf["confidence"]["source"] == "vlm"
    assert 0.0 <= leaf["confidence"]["score"] <= 1.0
    assert isinstance(leaf["flagged"], bool)
    assert leaf["page_ref"] == [1]

    assert "named_insured.primary_name" in result.fields
    assert set(result.fields["named_insured.primary_name"]) == {"value", "confidence"}
    assert result.pages_used


def test_a_canonical_leaf_carries_its_own_calibrated_confidence(model):
    """The envelope's score IS the calibrated field's, not a default: the two
    are one number reported in two places."""
    result = extract(_request(), model, StaticClassifier("policy"), CALIBRATION)
    leaf = result.extraction["policy"]["policy_number"]
    field = result.fields["policy.policy_number"]
    assert leaf["confidence"]["score"] == round(field["confidence"], 4)


def test_every_output_date_is_mm_dd_yyyy_whatever_the_model_wrote(client):
    """The format is enforced after generation, not merely requested: a model
    that writes ISO still returns MM/DD/YYYY, and raw stays as printed."""
    iso = json.dumps({
        **GOLDEN,
        "policy": {
            "policy_number": _fv("WC-8842317-01"),
            "effective_date": _fv("April 1, 2026", "2026-04-01"),
            "expiration_date": _fv("4/1/27", "2027-04-01"),
        },
    })
    result = extract(
        _request(ocr_text=f"{PAGE_TEXT}\nPolicy Period April 1, 2026 to 4/1/27"),
        load_model("base", client, backend_impl=EchoBackend(iso)),
        StaticClassifier("policy"), CALIBRATION,
    )
    policy = result.extraction["policy"]
    assert policy["effective_date"]["parsed"] == "04/01/2026"
    assert policy["expiration_date"]["parsed"] == "04/01/2027"
    assert policy["effective_date"]["raw"] == "April 1, 2026"


def test_a_flat_document_type_also_returns_mm_dd_yyyy(client):
    """ACORD and Loss Run keep their flat schemas; the date rule is the same."""
    response = json.dumps({
        "carrier": "Sentinel", "policy_number": "WC-1", "valuation_date": "2026-03-31",
        "line_of_business": ["workers_comp"], "total_claims_reported": 1,
        "claims": [{"claim_number": "C1", "loss_date": "2024-01-01", "status": "open"}],
    })
    result = extract(
        _request(source_id="lossrun_0001", known_doc_type="lossrun", known_lob=None),
        load_model("base", client, backend_impl=EchoBackend(response)),
        StaticClassifier("lossrun"),
        CalibrationParams(method="temperature", doc_type="lossrun",
                          model_version="v1", temperature=1.0),
        strict_schema=False,
    )
    assert result.extraction["valuation_date"] == "03/31/2026"
    assert result.extraction["claims"][0]["loss_date"] == "01/01/2024"


@pytest.mark.format_profile("standard")
def test_a_known_lob_selects_its_canonical_schema(client, monkeypatch):
    """The line is the caller's to give. It selects the schema the model is
    shown, constrained to and validated against."""
    import serving.pipeline as pipeline
    from common.schemas import resolved_schema, with_page_bounds
    from inference_core.input_builder import page_total

    # Sent in planned (group) order, so each call pairs with its group below.
    monkeypatch.setattr(pipeline, "_send_order", lambda pending, doc_type: list(range(len(pending))))

    # A line other than the default request's, so it is the caller's line that
    # selects the schema; self-contained like LINE, so GOLDEN is valid for it.
    backend = EchoBackend(RESPONSE)
    extract(
        _request(known_lob="cyber"), load_model("base", client, backend_impl=backend),
        StaticClassifier("policy"), CALIBRATION,
    )
    from common.schema_sections import groups_for

    # A policy is read in windows, each constrained to its slice of the LINE's
    # schema — so the cyber block reaches the model through `lineblk`.
    schemas = [call["json_schema"] for call in backend.calls]
    assert schemas == [
        with_page_bounds(resolved_schema("policy", None, "cyber", group),
                         page_total(call["messages"]))
        for group, call in zip(groups_for("cyber"), backend.calls, strict=True)
    ]
    assert any("cyber" in schema["properties"] for schema in schemas)


def test_a_policy_with_no_known_lob_still_returns_canonical_json(client):
    """No line means the client's canonical fallback, never the flat schema. The
    fallback composes the common model since 1.1.0: the carrier's name is
    `name`, and the document's page count is a plain value the pipeline sets."""
    fallback = json.dumps({
        "carrier": {"name": _fv("Granite Mutual Insurance Co")},
        "named_insured": {"primary_name": _fv("Rivera Fabrication LLC")},
        "policy": {
            "policy_number": _fv("WC-8842317-01"),
            "effective_date": _fv("04/01/2026"),
        },
    })
    result = extract(
        _request(known_lob=None), load_model("base", client, backend_impl=EchoBackend(fallback)),
        StaticClassifier("policy"), CALIBRATION,
    )
    assert result.schema_valid
    assert "confidence" in result.extraction["policy"]["effective_date"]
    assert result.extraction["document"]["page_count"] == 1


def test_schema_invalid_output_is_rejected_not_returned(client):
    """Mirrors the Fideon SPEC_07 Stage 3 audit gate — an invalid response is an
    error, not a result."""
    bad = load_model("base", client, backend_impl=EchoBackend('{"insured_name":"X"}'))
    with pytest.raises(PipelineError, match="failed schema validation"):
        extract(_request(), bad, StaticClassifier("policy"), CALIBRATION)


def test_unparseable_generation_fails_loudly(client):
    bad = load_model("base", client, backend_impl=EchoBackend("here is the policy data"))
    with pytest.raises(PipelineError, match="parseable JSON"):
        extract(_request(), bad, StaticClassifier("policy"), CALIBRATION)


def test_low_confidence_routing_reaches_the_result(model):
    result = extract(
        _request(known_doc_type=None), model,
        StaticClassifier("policy", confidence=0.3), CALIBRATION,
    )
    assert result.route_info["foundation_only"]
    assert "routing:low_confidence" in result.review_flags


def test_image_only_mode_runs_without_ocr(model):
    result = extract(
        _request(ocr_text=None, modality_mode="image_only"),
        model, StaticClassifier("policy"), CALIBRATION,
    )
    assert result.mode == "image_only"
    assert result.schema_valid


def test_list_completeness_flags_reach_the_result(client):
    """A dropped claim row is invisible to per-field confidence."""
    response = json.dumps({
        "carrier": "Sentinel", "policy_number": "WC-1", "valuation_date": "2026-03-31",
        "line_of_business": ["workers_comp"], "total_claims_reported": 8,
        "claims": [{"claim_number": f"C{i}", "loss_date": "2024-01-01", "status": "open"}
                   for i in range(6)],
    })
    lossrun_model = load_model("base", client, backend_impl=EchoBackend(response))
    result = extract(
        _request(source_id="lossrun_0001", known_doc_type="lossrun", known_lob=None),
        lossrun_model, StaticClassifier("lossrun"),
        CalibrationParams(method="temperature", doc_type="lossrun",
                          model_version="v1", temperature=1.0),
    )
    assert "claims:row_count_mismatch" in result.review_flags
    assert result.list_fields["claims"]["flagged"]


# --------------------------------------------------------------------------
# test == prod parity
# --------------------------------------------------------------------------

def test_the_harness_calls_the_serving_pipeline_not_a_copy(model):
    """THE parity test. Two extraction paths would be two systems, and the
    harness would be describing the wrong one."""
    direct = extract(_request(), model, StaticClassifier("policy"), CALIBRATION)
    via_harness, _metrics = run_document(
        _request(), model, StaticClassifier("policy"), CALIBRATION
    )
    assert via_harness.as_dict() == direct.as_dict()


def test_the_harness_does_not_reimplement_inference():
    """Enforced by inspection, not by discipline."""
    import inspect

    from testing import run_extraction

    source = inspect.getsource(run_extraction)
    assert "from serving.pipeline import" in source
    for forbidden in ("map_field_spans", "build_messages", "generate("):
        assert forbidden not in source, f"the harness reimplements {forbidden}"


def test_confidence_is_emitted_without_ground_truth(model):
    """The production case: on an unlabeled document confidence is still
    available to route review (arch §17)."""
    _result, metrics = run_document(_request(), model, StaticClassifier("policy"), CALIBRATION)
    assert metrics["overall_confidence"] > 0
    assert "not supplied" in metrics["ground_truth"]
    assert "field_normalized_match_rate" not in metrics


def test_ground_truth_adds_accuracy_metrics(model):
    _result, metrics = run_document(
        _request(), model, StaticClassifier("policy"), CALIBRATION, golden=GOLDEN
    )
    assert metrics["field_normalized_match_rate"] == 1.0
    assert all(f["correct"] for f in metrics["fields"].values())


def test_summary_aggregates_across_a_batch(model):
    results = [
        run_document(_request(source_id=f"policy_{i:04d}"), model,
                     StaticClassifier("policy"), CALIBRATION, golden=GOLDEN)
        for i in range(3)
    ]
    summary = summarise(results, "v1")
    assert summary.documents == 3
    assert summary.schema_validity_rate == 1.0
    assert summary.metrics["mean_field_normalized_match"] == 1.0


# --------------------------------------------------------------------------
# Merge and quantize
# --------------------------------------------------------------------------

def test_merge_plans_target_the_staging_volume():
    """The merged model is ~16GB and quantization also runs on RunPod — pushing
    it to Azure and back is a 32GB round trip for nothing."""
    plan = plan_merge(base_model="Qwen/Qwen3-VL-8B-Instruct", version="v2")
    assert plan.output_dir.startswith("/runpod-volume/")


def test_the_merge_defaults_to_bf16_not_fp16():
    """The adapter trained in bf16 against a bf16 base, and FP8 is quantized from
    this artifact (arch v2.1 §13a). Merging to fp16 would introduce a precision
    change between training and every serving format, for no reason."""
    assert plan_merge(base_model="m", version="v2").dtype == "bf16"


def test_the_merge_folds_in_the_selected_checkpoint_when_there_is_one():
    """Recorded because the merged weights do not say which checkpoint they came
    from, and once the staging volume is reclaimed nothing else does either."""
    chosen = "/runpod-volume/staging/adapters/foundation/v2/checkpoint-450"
    plan = plan_merge(base_model="m", version="v2", selected_checkpoint=chosen)

    assert plan.adapter == chosen
    assert plan.selected_checkpoint == chosen
    assert chosen in plan.describe()


def test_the_merge_falls_back_to_the_staged_adapter_directory():
    """What `--from-stage merge` does: no selection ran, so there is nothing to
    pick and the adapter directory is the honest source."""
    plan = plan_merge(base_model="m", version="v2")
    assert plan.selected_checkpoint is None
    assert plan.adapter.endswith("/v2")


def test_quantization_defaults_to_the_bf16_reference_alone():
    """FP8 is the plan, not a proven capability — Phase 0 spike item 9 verifies a
    decoder-only export loads in vLLM. Until then cycle 1 serves the merged bf16
    model (arch v2.1 §13a)."""
    plan = plan_quantization(version="v2")
    assert plan.formats == ["bf16"]
    assert plan.quantized_formats == [], "nothing is compressed in cycle 1"


def test_a_gguf_format_is_refused_on_the_serving_path():
    """GGUF is llama.cpp's format and the endpoint runs vLLM. Producing one per
    cycle spent conversion and eval compute on an artifact nothing could deploy."""
    with pytest.raises(QuantizationError, match="vLLM does not"):
        plan_quantization(version="v2", formats=["q5_k_m"])


def test_serving_and_edge_artifacts_do_not_share_a_prefix():
    """They are loaded by different programs, and one prefix invites deploying
    the wrong one."""
    from artifact_registry import paths

    assert "/vllm/" in paths.quantized_model_dir("v2", "fp8")
    assert "/gguf/" in paths.quantized_model_dir("v2", "q5_k_m")


def test_a_bf16_gguf_export_never_lands_in_the_vllm_directory():
    """bf16 is both a serving format and a GGUF format. Inferring the runtime from
    the name put a bf16 GGUF where the serving endpoint loads from."""
    from artifact_registry import paths
    from postprocessing.quantize import plan_gguf_export

    plan = plan_gguf_export(version="v2", formats=["bf16"], mmproj_verified=True)
    assert "/gguf/bf16" in plan.output_dirs["bf16"]
    assert "/vllm/" not in plan.output_dirs["bf16"]
    assert "/vllm/bf16" in paths.quantized_model_dir("v2", "bf16")
    with pytest.raises(paths.PathError, match="not a vllm format"):
        paths.quantized_model_dir("v2", "q5_k_m", runtime="vllm")


def test_unknown_quantization_format_is_refused():
    with pytest.raises(QuantizationError, match="unknown serving format"):
        plan_quantization(version="v2", formats=["int2"])



# --------------------------------------------------------------------------
# Loss Run window merge (arch v2.1 §7b)
# --------------------------------------------------------------------------

def test_the_overlap_joins_a_row_split_across_a_page_break():
    """The whole reason windows overlap. The first window saw the claim cut off
    at the page boundary and read no amount; the second saw it whole."""
    from serving.lossrun_merge import merge_windows

    first = [{"row_type": "claim", "claim_number": "CL-2", "claimant": "Smith",
              "total_incurred": None}]
    second = [{"row_type": "claim", "claim_number": "CL-2", "claimant": "Smith",
               "total_incurred": 50.0}]
    report = merge_windows([first, second])

    assert len(report.rows) == 1
    assert report.rows[0]["total_incurred"] == 50.0
    assert report.duplicates_collapsed == 1
    assert not report.conflicts, "filling an absent field is not a conflict"


def test_two_windows_reading_a_claim_differently_is_a_recorded_conflict():
    """The merge picks one. That choice is mechanical and auditable, but it is
    still a choice about a value nobody verified."""
    from serving.lossrun_merge import merge_windows

    report = merge_windows([
        [{"row_type": "claim", "claim_number": "CL-1", "total_incurred": 100.0}],
        [{"row_type": "claim", "claim_number": "CL-1", "total_incurred": 900.0,
          "claimant": "Smith"}],
    ])
    assert len(report.rows) == 1
    assert report.conflicts and report.flagged
    # The more complete reading wins: a window that saw the whole row has
    # strictly more evidence than one that saw it cut off.
    assert report.rows[0]["total_incurred"] == 900.0


def test_a_row_with_no_claim_number_falls_back_to_a_composite_key():
    """Ordinary on older Loss Runs and on reports that redact the number."""
    from serving.lossrun_merge import merge_windows

    row = {"row_type": "claim", "loss_date": "2024-03-01", "claimant": "Smith",
           "total_incurred": 100.0}
    report = merge_windows([[dict(row)], [dict(row)]])
    assert len(report.rows) == 1, "the same claim seen twice is one claim"


def test_an_unkeyable_row_is_kept_not_dropped():
    """A duplicate inflates a total, which reconciliation catches. A dropped row
    understates a loss history, which nothing catches."""
    from serving.lossrun_merge import merge_windows

    report = merge_windows([[{"row_type": "claim", "status": "open"}]])
    assert len(report.rows) == 1
    assert report.unkeyed_rows == 1


def test_a_subtotal_repeated_across_an_overlap_is_counted_once():
    """Counting it twice would double the figure reconciliation checks against."""
    from serving.lossrun_merge import merge_windows

    subtotal = {"row_type": "subtotal", "policy_period": "2024", "total_incurred": 150.0}
    report = merge_windows([[dict(subtotal)], [dict(subtotal)]])
    assert len(report.total_rows) == 1


def test_the_merged_claims_keep_document_order_and_are_reproducible():
    """The schema asks for claims in document order, and a claims list whose order
    depends on dict iteration is not comparable against its own previous
    extraction: first seen, window by window, row by row."""
    from serving.lossrun_merge import merge_windows

    rows = [
        {"row_type": "claim", "claim_number": f"CL-{i}", "total_incurred": float(i)}
        for i in (3, 1, 2)
    ]
    first = merge_windows([rows[:2], rows[1:]])
    assert [r["claim_number"] for r in first.rows] == ["CL-3", "CL-1", "CL-2"]
    assert merge_windows([rows[:2], rows[1:]]).rows == first.rows


def test_two_rows_with_one_key_in_one_window_are_two_claims():
    """Only the overlap repeats a claim: two $0 incidents on one date in one window
    are two claims, and the next window seeing both again leaves two."""
    from serving.lossrun_merge import merge_windows

    incident = {"row_type": "claim", "loss_date": "2024-03-01", "total_incurred": 0.0}
    report = merge_windows([[dict(incident), dict(incident)], [dict(incident), dict(incident)]])
    assert len(report.rows) == 2 and report.duplicates_collapsed == 2


def test_incident_claims_of_one_day_are_told_apart_by_their_description():
    from serving.lossrun_merge import merge_windows

    report = merge_windows([[
        {"row_type": "claim", "loss_date": "2024-03-01", "total_incurred": 0.0, "description": "Slip"},
    ], [
        {"row_type": "claim", "loss_date": "2024-03-01", "total_incurred": 0.0, "description": "Hail"},
    ]])
    assert len(report.rows) == 2


def test_a_row_cut_at_a_page_break_keeps_its_full_reading_without_a_conflict():
    from serving.lossrun_merge import merge_windows

    report = merge_windows([
        [{"row_type": "claim", "claim_number": "CL-1", "description": "Slip and fall"}],
        [{"row_type": "claim", "claim_number": "CL-1", "description": "Slip and fall in warehouse, back injury"}],
    ])
    assert report.rows[0]["description"] == "Slip and fall in warehouse, back injury"
    assert not report.conflicts


def test_merge_and_reconcile_closes_the_completeness_loop():
    """A merge whose result is never reconciled has no completeness signal at
    all — a missed row produces no tokens, so per-field confidence is
    structurally blind to it (§5.5)."""
    from serving.lossrun_merge import merge_and_reconcile

    windows = [
        [{"row_type": "claim", "claim_number": "CL-1", "total_incurred": 100.0}],
        [{"row_type": "claim", "claim_number": "CL-2", "total_incurred": 75.0},
         {"row_type": "subtotal", "policy_period": "2024", "total_incurred": 175.0}],
    ]
    merged, reconciliation = merge_and_reconcile(windows, totals_output={"total_incurred": 175.0})

    assert len(merged.rows) == 2
    assert reconciliation.reconciled, reconciliation.reasons()


def test_a_missed_row_fails_reconciliation():
    """The signal per-field confidence cannot produce."""
    from serving.lossrun_merge import merge_and_reconcile

    windows = [[
        {"row_type": "claim", "claim_number": "CL-1", "total_incurred": 100.0},
        {"row_type": "subtotal", "policy_period": "2024", "total_incurred": 175.0},
    ]]
    _merged, reconciliation = merge_and_reconcile(windows)
    assert not reconciliation.reconciled
    assert reconciliation.flagged


def test_unattributed_claims_reconcile_against_a_single_printed_period():
    """A single-period Loss Run commonly prints the period once in the header and
    not on every row. Flagging every such document would be a measure of the
    document's layout, not of the extraction."""
    from calibration.reconciliation import reconcile

    claims = [{"claim_number": "CL-1", "total_incurred": 100.0}]
    totals = [{"row_type": "subtotal", "policy_period": "2024", "total_incurred": 100.0}]
    assert reconcile(claims, totals).reconciled


def test_unattributed_claims_across_several_periods_are_not_guessed():
    """They cannot be assigned without guessing, and a guess here produces a
    reconciliation that means nothing."""
    from calibration.reconciliation import reconcile

    claims = [{"claim_number": "CL-1", "total_incurred": 100.0}]
    totals = [
        {"row_type": "subtotal", "policy_period": "2024", "total_incurred": 60.0},
        {"row_type": "subtotal", "policy_period": "2025", "total_incurred": 40.0},
    ]
    assert not reconcile(claims, totals).reconciled


# --------------------------------------------------------------------------
# Structured outputs and logprobs (arch v2.1 §13, §5.1)
# --------------------------------------------------------------------------

def test_structured_decoding_with_masked_logprobs_is_refused():
    """Constrained decoding masks invalid tokens, so the post-mask distribution
    is NOT the model's own. A calibrator fitted on it produces a number that
    looks like a probability, is not one, and cannot be told apart from one
    anywhere downstream (§5.1)."""
    from inference_core.runner_config import RunnerConfig

    unsafe = RunnerConfig(json_schema={"type": "object"}, logprobs_mode="processed_logprobs")
    with pytest.raises(ValueError, match="raw_logprobs"):
        unsafe.assert_logprobs_are_the_models_own()

    safe = RunnerConfig(json_schema={"type": "object"}, logprobs_mode="raw_logprobs")
    safe.assert_logprobs_are_the_models_own()


def test_unconstrained_generation_needs_no_logprobs_mode_guard():
    """Evaluation runs unconstrained too, as a training-health signal: whether
    the model learned the format on its own is a different question from whether
    the format is enforced."""
    from inference_core.runner_config import RunnerConfig

    RunnerConfig(json_schema=None, logprobs_mode="anything").assert_logprobs_are_the_models_own()


def test_the_serving_config_pins_raw_logprobs():
    from common.config import serving_config

    generation = serving_config()["generation"]
    assert generation["logprobs_mode"] == "raw_logprobs"
    assert generation["structured_outputs"] is True


# --------------------------------------------------------------------------
# The serving path uses the release bundle's calibrators (arch v2.1 §5.3)
# --------------------------------------------------------------------------

def _fitted_calibrators():
    """A calibrator set and thresholds fitted the way the calibrate stage does."""
    import random

    from calibration.feature_calibrator import fit_calibrators
    from calibration.features import build_features
    from calibration.thresholds import fit_thresholds

    rng = random.Random(23)

    def half(offset):
        rows = []
        for i in range(500):
            correct = rng.random() < 0.85
            logprobs = [-0.05 - rng.random() * 0.1] * 3 if correct else [-2.5 - rng.random()] * 3
            rows.append((
                build_features(
                    field_path="policy_number", value=f"WC-{i + offset}",
                    logprobs=logprobs, document={},
                    page_text=f"WC-{i + offset}" if correct else "absent",
                ),
                correct,
            ))
        return rows

    calibrators = fit_calibrators(half(0), release_id="release-2026.11.1", serving_format="bf16")
    scored: dict[str, list] = {}
    for features, correct in half(10_000):
        confidence = calibrators.predict(features)
        if confidence is not None:
            scored.setdefault(features.field_type, []).append((confidence, correct))
    thresholds = fit_thresholds(scored, release_id="release-2026.11.1", serving_format="bf16")
    return calibrators, thresholds


def test_the_serving_path_uses_the_fitted_calibrators(model):
    """Phase 10 built the replacement and it sat beside the serving path unused.
    This is the wire: confidence now comes from the per-field-type calibrator
    over the §5.2 feature vector, not from a length-biased minimum."""
    calibrators, thresholds = _fitted_calibrators()
    result = extract(
        _request(), model, StaticClassifier("policy"), CALIBRATION,
        calibrators=calibrators, thresholds=thresholds,
    )
    assert result.schema_valid
    assert result.fields, "no field carried a confidence at all"
    for body in result.fields.values():
        assert 0.0 <= body["confidence"] <= 1.0


def test_a_field_type_with_no_calibrator_routes_to_review(model):
    """Not a default number. A calibrator fitted on too little data produces
    values that look like probabilities and are not, and every §5.4 threshold is
    defined against a calibrated score."""
    from calibration.feature_calibrator import CalibratorSet
    from calibration.thresholds import ThresholdSet

    result = extract(
        _request(), model, StaticClassifier("policy"), CALIBRATION,
        calibrators=CalibratorSet(release_id="r", serving_format="bf16"),
        thresholds=ThresholdSet(release_id="r", serving_format="bf16"),
    )
    assert result.review_flags, "an uncalibrated release must flag everything"
    assert any("no_confidence" in f for f in result.review_flags)


def test_without_a_calibrator_set_the_v1_path_is_a_knowing_downgrade(model, caplog):
    """A release whose calibrate stage never ran has no per-field-type curves and
    no measured thresholds. Serving it through the v1 transform is a downgrade,
    not an equivalent, and the log says so rather than passing silently."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = extract(_request(), model, StaticClassifier("policy"), CALIBRATION)

    assert result.schema_valid
    assert any("length-biased" in r.getMessage() for r in caplog.records), caplog.text


@pytest.mark.format_profile("standard")
def test_serving_generation_is_constrained_to_the_routed_schema(client, monkeypatch):
    """structured_outputs and logprobs_mode sat in vllm_serving.yaml and reached
    nothing: generation ran unconstrained, so the §13b 100% schema-validity floor
    measured a model with no guarantee behind it."""
    import serving.pipeline as pipeline
    from common.schemas import resolved_schema, with_page_bounds
    from inference_core.input_builder import page_total
    from inference_core.runner_config import load_runner_config

    # Sent in planned (group) order, so each call pairs with its group below.
    monkeypatch.setattr(pipeline, "_send_order", lambda pending, doc_type: list(range(len(pending))))

    config = load_runner_config()
    assert config.structured_outputs is True
    assert config.logprobs_mode == "raw_logprobs"

    backend = EchoBackend(RESPONSE)
    constrained = load_model("base", client, backend_impl=backend)
    extract(_request(), constrained, StaticClassifier("policy"), CALIBRATION)

    assert backend.calls, "nothing was generated"
    from common.schema_sections import groups_for

    # Every window is constrained — each to its own slice of the routed schema.
    assert [c["json_schema"] for c in backend.calls] == [
        with_page_bounds(resolved_schema("policy", None, LINE, group), page_total(c["messages"]))
        for group, c in zip(groups_for(LINE), backend.calls, strict=True)
    ]


def test_structured_decoding_on_masked_logprobs_is_refused_at_load():
    import dataclasses

    from inference_core.runner_config import load_runner_config

    config = dataclasses.replace(load_runner_config(), logprobs_mode="processed_logprobs")
    with pytest.raises(ValueError, match="raw_logprobs"):
        config.assert_logprobs_are_the_models_own()
