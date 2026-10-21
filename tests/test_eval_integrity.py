"""What the gate, checkpoint selection and calibration measure — and what they used to miss.

Each test pins one way a number could be wrong while looking fine: an invented
field that cost nothing, a nested table nobody scored, a feature fitted on text
serving never sends, a broken generation pass scored as a model's answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.features import (
    build_document_features,
    infer_field_type,
    rule_checks,
    shown_ocr_text,
)
from common.tasks import CORPUS_TASKS
from evaluation.metrics.confusable import score_misattribution
from evaluation.metrics.coverage_metrics import score_lob
from evaluation.metrics.field_accuracy import find_list_fields, score_all_list_fields, score_fields
from evaluation.run_eval import build_report
from evaluation.validation_generation import (
    ValidationGeneration,
    ValidationGenerationError,
    assert_generations_usable,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _env(raw, parsed=None, page=1):
    return {"raw": raw, "parsed": parsed, "page_ref": [page]}


# --------------------------------------------------------------------------
# Field scoring
# --------------------------------------------------------------------------


def test_an_invented_field_is_scored_wrong():
    expected = {"policy": {"number": "P-1"}}
    got = {"policy": {"number": "P-1", "effective_date": "01/01/2026"}}
    report = score_fields(expected, got)
    assert report.total == 2
    assert report.normalized_match == 0.5


def test_an_empty_extra_field_costs_nothing():
    report = score_fields({"a": "x"}, {"a": "x", "b": None, "c": ""})
    assert report.normalized_match == 1.0


def test_a_window_with_nothing_to_score_is_not_a_zero():
    doc = {"doc_type": "policy", "sections": "decl"}
    right = {"policy": {"policy_number": "x"}}
    scored = build_report("v", [(right, right, doc), ({}, {}, doc)])
    assert scored.gate_metrics()["field_normalized_match"] == 1.0


def test_nested_tables_are_found_and_scored_on_values():
    expected = {"auto": {"vehicles": [{"vin": _env("1HGCM82633A004352")}]}}
    got = {"auto": {"vehicles": [{"vin": _env("1HGCM82633A004352", page=7)}]}}
    assert "auto.vehicles" in find_list_fields(expected)
    report = score_all_list_fields(expected, got)["auto.vehicles"]
    assert report.matched_rows == 1


def test_misattribution_is_seen_on_nested_paths():
    expected = {"named_insured": {"name": "Acme LLC"}, "producer": {"name": "Brokers Inc"}}
    got = {"named_insured": {"name": "Brokers Inc"}, "producer": {"name": "Brokers Inc"}}
    report = score_misattribution(expected, got, "policy")
    assert [c.field_path for c in report.cases] == ["named_insured.name"]


def test_lob_is_not_scored_for_a_label_that_carries_none():
    report = score_lob([({"policy": {"number": "P"}}, {"line_of_business": ["auto"]})])
    assert report.scored == 0


def test_every_corpus_task_is_declared():
    """The classifier metric is not applicable because no classify rows are built.
    The build names a flat row ``extract`` and a policy window by its section
    group's task; a task outside CORPUS_TASKS would silently mis-mark metrics."""
    from common.schema_sections import group_names, task_for

    built = {"extract"} | {task_for(group) for group in group_names()}
    assert built <= {str(task) for task in CORPUS_TASKS}


# --------------------------------------------------------------------------
# Calibration features
# --------------------------------------------------------------------------


def test_address_components_are_typed_by_their_path():
    assert infer_field_type("carrier.address.city") == "address"
    assert infer_field_type("named_insured.mailing_address.line_1") == "address"


def test_the_date_order_rule_reads_the_canonical_policy_period():
    document = {"policy": {"effective_date": "06/01/2026", "expiration_date": "01/01/2026"}}
    assert rule_checks("policy.effective_date", "06/01/2026", document) is False


def test_ocr_agreement_is_read_against_the_printed_form():
    extraction = {"policy": {"effective_date": _env("June 1, 2026", "06/01/2026")}}
    [features] = build_document_features(
        extraction=extraction, spans={}, page_text="Policy period June 1, 2026 to June 1, 2027"
    )
    assert features.field_path == "policy.effective_date"
    assert features.value == "06/01/2026"
    assert features.ocr_agreement == 1.0


def test_shown_ocr_text_strips_markers_and_is_none_without_text():
    assert shown_ocr_text(["<page 1 of 2>\nHello", "<page 2 of 2>\n"]) == "Hello"
    assert shown_ocr_text(["<page 1 of 1>"]) is None


def test_an_image_only_row_carries_no_ocr_text():
    row = {
        "modality_mode": "image_only",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "<page 1 of 1>"}, {"type": "image", "image": "k"},
        ]}],
    }
    assert ValidationGeneration(row=row, golden={}).page_text is None


# --------------------------------------------------------------------------
# Generation passes
# --------------------------------------------------------------------------


def _generations(failed: int, total: int) -> list[ValidationGeneration]:
    return [
        ValidationGeneration(row={}, golden={}, error="OSError: no image" if i < failed else None)
        for i in range(total)
    ]


def test_a_mostly_failed_pass_is_refused():
    with pytest.raises(ValidationGenerationError, match="3 of 10"):
        assert_generations_usable(_generations(3, 10), what="test")


def test_a_few_failures_are_tolerated():
    assert_generations_usable(_generations(1, 10), what="test")


def test_no_generations_is_refused():
    with pytest.raises(ValidationGenerationError):
        assert_generations_usable([], what="test")


def test_rows_are_localized_to_cached_files(client, tmp_path):
    from training.stage_data import localize_rows

    client.write_bytes("processed/p/page_1.png", b"\x89PNG fake")
    row = {"messages": [{"role": "user", "content": [
        {"type": "image", "image": "processed/p/page_1.png"}, {"type": "text", "text": "t"},
    ]}]}
    [local] = localize_rows([row], client, tmp_path)
    path = local["messages"][0]["content"][0]["image"]
    assert Path(path).read_bytes() == b"\x89PNG fake"
    assert row["messages"][0]["content"][0]["image"] == "processed/p/page_1.png"


# --------------------------------------------------------------------------
# Engines are released between stages
# --------------------------------------------------------------------------


def test_the_checkpoint_scorer_releases_its_engine(client):
    from evaluation.checkpoint_eval import generation_scorer
    from inference_core.model_runner import EchoBackend, load_model

    closed = []

    class Closing(EchoBackend):
        def close(self) -> None:
            closed.append(True)

    model = load_model("base", client, backend_impl=Closing())
    scorer = generation_scorer([{"messages": []}], model)
    scorer.close()
    assert closed == [True]


def test_release_model_ignores_objects_without_a_backend():
    from inference_core.model_runner import release_model

    release_model(object())


def test_scoped_quantization_plan_addresses_the_scope():
    from artifact_registry import paths
    from postprocessing.quantize import plan_quantization

    plan = plan_quantization(version="v1.0.0", formats=["bf16"], scope="policy")
    assert plan.merged_model == paths.staging_merged_model_dir("v1.0.0", scope="policy")


def test_golden_json_fixture_loads():
    """Guard for the fixture the golden-set tests read."""
    fixture = Path(__file__).parent / "fixtures" / "golden" / "policy_0001.golden.json"
    assert json.loads(fixture.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Corpus quality
# --------------------------------------------------------------------------


def test_one_insurer_written_three_ways_is_one_carrier():
    from common.normalize import normalize_carrier

    names = [
        "The Travelers Indemnity Company",
        "Travelers Casualty and Surety Company of America",
        "TRAVELERS",
    ]
    assert {normalize_carrier(n) for n in names} == {"travelers"}
    assert normalize_carrier("Great American Insurance Company") != normalize_carrier(
        "Great Northern Insurance Co."
    )


def test_cap_estimate_counts_digits_one_token_each():
    from data_pipeline.dataset_builder.cap_check import estimate_text_tokens

    assert estimate_text_tokens("04/01/2026") >= 8


def test_a_short_policy_reports_its_declarations_page():
    from serving.page_router import plan_pages

    plan = plan_pages({
        1: "Fax cover sheet", 2: "Billing notice", 3: "Privacy notice",
        4: "COMMON POLICY DECLARATIONS Named Insured Policy Period",
    })
    assert plan.pages == [1, 2, 3, 4]
    assert plan.declarations_page == 4


def test_character_noise_reaches_the_bottom_of_the_page():
    import random

    from data_pipeline.dataset_builder.noisy_ocr_augment import _corrupt_characters

    text = " ".join(f"POL{i:04d}0" for i in range(400))
    lowest = []
    for seed in range(40):
        corrupted, _ = _corrupt_characters(text, random.Random(seed), rate=0.08)
        changed = [i for i, (a, b) in enumerate(zip(text, corrupted, strict=False)) if a != b]
        if changed:
            lowest.append(max(changed))
    assert max(lowest) > len(text) // 2


def test_policy_lines_are_counted_from_metadata_not_checked_against_the_enum():
    """A policy's line is a schema name (flood, gl, cyber). Checked against the
    13-value LOB enum it raised, and the whole corpus build failed."""
    from data_pipeline.corpus_manifest import compute_lob_coverage, count_policy_lines

    labels = [{"policy": {}}, {"policy": {}}, {"policy": {}}, {"line_of_business": ["workers_comp"]}]
    lobs = ["flood", "gl", ["homeowners", "personal_auto"], None]
    shares, _ = compute_lob_coverage(labels, lobs)        # does not raise
    assert shares["workers_comp"] == 1.0                   # only the label that carries the field
    assert count_policy_lines(lobs) == {"flood": 1, "gl": 1, "homeowners": 1, "personal_auto": 1}


def test_a_long_policy_counts_once_not_once_per_window():
    """Two documents: a policy read as three windows, all right, and a certificate
    read once, all wrong. Per document that is 0.5; per row it was 0.75."""
    policy = {"doc_type": "policy", "sections": "decl", "source_id": "p1"}
    cert = {"doc_type": "policy", "sections": "decl", "source_id": "c1"}
    x, y = {"policy": {"policy_number": "x"}}, {"policy": {"policy_number": "y"}}
    rows = [(x, x, policy)] * 3 + [(x, y, cert)]
    assert build_report("v", rows).gate_metrics()["field_normalized_match"] == 0.5


def _unusable(bad: int, total: int, kind: str = "output") -> list[ValidationGeneration]:
    return [
        ValidationGeneration(row={}, golden={}, extraction={} if i < bad else {"a": 1},
                             error="JSONDecodeError: Unterminated string" if i < bad else None,
                             failure_kind=kind if i < bad else None)
        for i in range(total)
    ]


def test_unusable_json_is_a_wrong_answer_not_a_broken_pass():
    """19 of 132 looping answers refused every checkpoint of the smoke run; a
    checkpoint that loops more must LOSE, not drop out of the comparison."""
    assert_generations_usable(_unusable(19, 132), what="test")


def test_a_pass_that_is_mostly_unusable_json_is_still_refused():
    with pytest.raises(ValidationGenerationError, match="unusable JSON"):
        assert_generations_usable(_unusable(70, 132), what="test")


def test_setup_failures_keep_the_ten_percent_limit():
    with pytest.raises(ValidationGenerationError, match="19 of 132"):
        assert_generations_usable(_unusable(19, 132, kind="setup"), what="test")


def test_each_failure_is_classified_where_it_happens():
    from types import SimpleNamespace as NS

    from evaluation.validation_generation import _finish

    looped = ValidationGeneration(row={}, golden={})
    _finish(looped, NS(text='{"a": "xxxx', truncated=lambda: True, tokens=[], token_logprobs=[]))
    assert looped.failure_kind == "output" and "token limit" in looped.error

    refused = ValidationGeneration(row={}, golden={})
    _finish(refused, RuntimeError("engine died"))
    assert refused.failure_kind == "setup" and refused.extraction == {}


def test_the_warning_carries_the_whole_failure_record(caplog):
    """The log showed only the parse error, so an answer cut at the token limit
    read as one that stopped normally."""
    from types import SimpleNamespace as NS

    from evaluation.validation_generation import _finish

    entry = ValidationGeneration(row={"source_id": "p1"}, golden={})
    with caplog.at_level("WARNING"):
        _finish(entry, NS(text='{"a": "xxxx', finish_reason="length", truncated=lambda: True,
                          tokens=["x"] * 5, token_logprobs=[]))
    assert "token limit" in caplog.text and "finish=length" in caplog.text
