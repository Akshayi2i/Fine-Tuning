"""Scoring a common-model line: what the second review found the report still got wrong.

The first fixes paired a coverage with a wrong code on its printed name, but
only after the code had been tried, and only while the coverage kept its link
to its vehicle. A wrong code is most often a sibling coverage's, which took that
sibling's row; and a policy whose vehicles and coverages fall in different
windows is served with every link lost (tests/test_schema2_serving.py), so its
coverages paired on the code in table order. Each test below scores an answer
served by replaying its training targets, or the fixture itself, and checks
that one misread value costs that value, in every metric that reads it.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path

import pytest

from evaluation.run_eval import build_report
from tests.test_schema2_review_eval import SELF_CONTAINED
from tests.test_schema2_serving import AUTO, _compact_auto, _serve_replay

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
GOLD = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))
META = {"source_id": "pa_1", "doc_type": "policy", "lob": "personal_auto",
        "modality_mode": "ocr_plus_image"}


@cache
def _serving(name: str) -> tuple[str, str]:
    """``(gold, served)`` as JSON text: the compact fixture (links kept) or the
    6-page one (every coverage's link lost across windows)."""
    gold, pages = (_compact_auto(), 3) if name == "compact" else (AUTO, 6)
    result, _backend, _answers = _serve_replay(gold, pages)
    return json.dumps(gold), json.dumps(result.extraction)


def _served(name: str) -> tuple[dict, dict]:
    gold, served = _serving(name)
    return json.loads(gold), json.loads(served)


def _full(gold, got, **meta):
    [full] = build_report("t", [(gold, got, {**META, **meta})]).full_set()
    return full


def _misread(envelope, value):
    envelope.update(raw=str(value), parsed=value)


#: The answers each test scores: the fixture as written, and as served.
SOURCES = ("fixture", "compact", "auto6")


def _source(name: str) -> tuple[dict, dict]:
    return (copy.deepcopy(GOLD), copy.deepcopy(GOLD)) if name == "fixture" else _served(name)


# --------------------------------------------------------------------------
# Coverages pair on what the page prints, links or none
# --------------------------------------------------------------------------


@pytest.mark.format_profile("standard")
def test_the_six_page_policy_is_served_with_every_coverage_link_lost():
    """The case the rest of this file is about, as serving produces it."""
    _gold, served = _served("auto6")
    assert served["coverages"] and not any(c.get("applies_to") for c in served["coverages"])


@pytest.mark.parametrize("name", ["compact", "auto6"])
def test_a_perfect_served_answer_scores_perfectly_on_every_count(name):
    """With its links lost, no coverage paired for coverage_code_accuracy (it
    was not measured) and the list metrics read the coverage table as 0 of 4."""
    gold, served = _served(name)
    metrics = _full(gold, served).metrics
    assert metrics["field_normalized_match"] == 1.0
    assert metrics["coverage_code_accuracy"] == 1.0
    assert metrics["list_field_recall"] == 1.0 and metrics["field_f1_list_fields"] == 1.0
    assert metrics["false_null_rate"] == 0.0


@pytest.mark.parametrize("name", SOURCES)
@pytest.mark.parametrize("index,code", [(0, "X_COLLISION"), (1, "X_VEHICLE_LIABILITY"), (1, "PA_COMP")])
def test_a_wrong_code_costs_that_code_alone(name, index, code):
    """Even when it is a sibling coverage's code and its row comes first: on
    the code it took the sibling's row, and the sibling's row the next unit's."""
    gold, got = _source(name)
    got["coverages"][index]["coverage_code"] = code
    full = _full(gold, got)
    assert [r["field_path"] for r in full.error_records] == [f"coverages[{index}].coverage_code"]
    assert full.metrics["false_null_rate"] == 0.0
    assert full.metrics["list_field_recall"] == 1.0
    assert full.metrics["coverage_code_accuracy"] == 0.75
    assert full.metrics["auto_accept_error_rate"] == 0.0


@pytest.mark.parametrize("name", SOURCES)
@pytest.mark.parametrize("order", ["second_unit_first", "reversed"])
def test_coverages_in_another_order_pair_with_the_rows_they_read(name, order):
    """Two Collision rows with no links are told apart by their premiums and
    deductibles, never by which one comes first."""
    gold, got = _source(name)
    rows = got["coverages"]
    got["coverages"] = rows[2:] + rows[:2] if order == "second_unit_first" else rows[::-1]
    metrics = _full(gold, got).metrics
    assert metrics["field_normalized_match"] == 1.0
    assert metrics["false_null_rate"] == 0.0
    assert metrics["coverage_code_accuracy"] == 1.0


@pytest.mark.parametrize("name", SOURCES)
def test_a_misread_vin_costs_that_value_alone(name):
    """The VIN keys the vehicle table, and keyed on it alone the misread row was
    a missed vehicle and an invented one: five values wrong, five false nulls."""
    gold, got = _source(name)
    _misread(got["vehicles"][0]["vin"], "1HGCV1F30KA000007")
    full = _full(gold, got)
    assert [r["field_path"] for r in full.error_records] == ["vehicles[0].vin"]
    assert full.metrics["list_field_recall"] == 1.0
    assert full.metrics["false_null_rate"] == 0.0


def test_a_window_target_with_a_siblings_code_costs_that_code_alone():
    """A window's coverages carry no links at all, so its table keys on the code
    alone: one sibling code scored the window 0.34 with 29 errors."""
    from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target

    plan = next(p for p in plan_windows("personal_auto", [1, 2, 3, 4, 5, 6], 1)
                if window_target(GOLD, "personal_auto", p, TargetReport()).get("coverages"))
    target = window_target(GOLD, "personal_auto", plan, TargetReport())
    got = copy.deepcopy(target)
    got["coverages"][1]["coverage_code"] = "X_VEHICLE_LIABILITY"
    full = _full(target, got, sections=plan.group)
    assert [r["field_path"] for r in full.error_records] == ["coverages[1].coverage_code"]
    assert full.metrics["false_null_rate"] == 0.0


def test_coverage_code_accuracy_never_pairs_on_the_code():
    """What it measures cannot be what pairs the rows: a wrong code still pairs
    on the printed name, and two rows of one name pair by their values."""
    from evaluation.metrics.field_accuracy import pair_coverages_by_name

    rows = [{"coverage_code": "COLL", "coverage_name": "Collision", "premium": 380},
            {"coverage_code": "COLL", "coverage_name": "Collision", "premium": 420},
            {"coverage_code": "UM", "coverage_name": "Uninsured Motorist"}]
    answer = [{"coverage_code": "COLL", "coverage_name": "Collision", "premium": 420},
              {"coverage_code": "COLL", "coverage_name": "Uninsured Motorist"},
              {"coverage_code": "COLL", "premium": 380}]
    assert pair_coverages_by_name(rows, answer) == [rows[1], rows[2], None]


def test_rows_sharing_a_key_pair_by_their_values_then_their_place():
    """In any table: never simply the first row left with that key."""
    from evaluation.metrics.field_accuracy import _pair_rows

    rows = [{"description": "Fence", "amount": 100}, {"description": "Fence", "amount": 200},
            {"description": "Shed", "amount": 50}, {"description": "Shed", "amount": 50}]
    answer = [{"description": "Fence", "amount": 200}, {"description": "Fence", "amount": 999},
              {"description": "Shed", "amount": 50}, {"description": "Shed", "amount": 50}]
    mates, unpaired = _pair_rows(rows, answer)
    assert [next(i for i, row in enumerate(rows) if row is mate) for mate in mates] == [1, 0, 2, 3]
    assert unpaired == []


def test_a_row_sharing_under_half_its_values_is_not_paired():
    """The last pass pairs a misread row, not two different ones."""
    from evaluation.metrics.field_accuracy import _pair_rows

    rows = [{"vin": "A1", "make": "Honda", "model": "Accord", "year": 2019}]
    answer = [{"vin": "B2", "make": "Ford", "model": "Focus", "year": 2019}]
    assert _pair_rows(rows, answer) == ([None], rows)


def test_self_contained_list_scoring_still_keys_on_the_code():
    """The pairing above is the common model's: a self-contained line's list
    metrics are as they were."""
    from evaluation.metrics.field_accuracy import score_all_list_fields

    gold = {"coverages": [{"coverage_code": "BI", "coverage_name": "Bodily Injury"},
                          {"coverage_code": "PD", "coverage_name": "Property Damage"}]}
    got = copy.deepcopy(gold)
    got["coverages"][0]["coverage_code"] = "XX"
    assert score_all_list_fields(gold, got)["coverages"].recall == 0.5
    assert score_all_list_fields(gold, got, common_model=True)["coverages"].recall == 1.0


# --------------------------------------------------------------------------
# A row the model wrote holds no false nulls
# --------------------------------------------------------------------------


def test_a_misread_limit_amount_is_no_false_null():
    """A limit row has no identifier, so it keys on every value it holds."""
    got = copy.deepcopy(GOLD)
    _misread(got["coverages"][0]["limits"][1]["amount"], 30000)
    full = _full(copy.deepcopy(GOLD), got)
    assert full.metrics["false_null_rate"] == 0.0
    assert [r["field_path"] for r in full.error_records] == ["coverages[0].limits[1].amount"]


def test_a_row_misread_past_pairing_is_wrong_values_not_false_nulls():
    """Four of a vehicle's six values misread: no identity pairs it, and field
    match counts a missed row and an invented one - but every value of it was
    written, so none is a null."""
    got = copy.deepcopy(GOLD)
    vehicle = got["vehicles"][0]
    for name, value in (("vin", "9ZZZZ9Z99ZZ999999"), ("make", "Ford"), ("model", "Focus"), ("year", 2011)):
        _misread(vehicle[name], value)
    full = _full(copy.deepcopy(GOLD), got)
    assert full.metrics["field_normalized_match"] < 1.0
    assert full.metrics["false_null_rate"] == 0.0


def test_a_dropped_vehicle_still_counts_its_values_as_false_nulls():
    from common.canonical import without_bare_values
    from common.structural_ids import comparable_view
    from evaluation.metrics.extraction_faults import _is_empty, score_false_nulls
    from evaluation.metrics.field_accuracy import flatten_scalars
    from evaluation.run_eval import _read_values

    got = copy.deepcopy(GOLD)
    dropped = got["vehicles"].pop(1)
    report = score_false_nulls([_read_values(comparable_view(GOLD, "personal_auto"),
                                             comparable_view(got, "personal_auto"), True)])
    in_row = [v for v in flatten_scalars(without_bare_values(dropped)).values() if not _is_empty(v)]
    assert report.hits == len(in_row) > 0
    assert all(i["field"].startswith("vehicles[1].") for i in report.instances)


# --------------------------------------------------------------------------
# Calibration is fitted on the pairing the gate judges by
# --------------------------------------------------------------------------


def _arrays_window():
    from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target

    for plan in plan_windows("personal_auto", [1, 2, 3, 4, 5, 6], 1):
        target = window_target(GOLD, "personal_auto", plan, TargetReport())
        if target.get("coverages"):
            return plan, target
    raise AssertionError("no window holds the coverages")


def test_a_wrong_code_fits_no_calibrated_value_as_wrong():
    """The coverage's name, limits and premium were read right; field match and
    auto-accept count them right, and so must the labels its calibrators learn."""
    from evaluation.validation_generation import ValidationGeneration, calibration_samples

    plan, target = _arrays_window()
    got = copy.deepcopy(target)
    got["coverages"][2]["coverage_code"] = "X_COMPREHENSIVE"
    row = {"doc_type": "policy", "lob": "personal_auto", "val_half": "calibration",
           "sections": plan.group}
    samples = calibration_samples([ValidationGeneration(row=row, golden=target, extraction=got)])
    assert samples["calibration"]
    assert [f.field_path for f, correct in samples["calibration"] if not correct] == []


def test_a_self_contained_line_is_calibrated_on_its_old_labels(monkeypatch):
    from evaluation.metrics import field_accuracy
    from evaluation.validation_generation import ValidationGeneration, calibration_samples

    def refuse(*_args, **_kwargs):
        raise AssertionError("a self-contained line's labels were paired the common model's way")

    monkeypatch.setattr(field_accuracy, "aligned_for_scoring", refuse)
    value = {"raw": "GL-1", "parsed": "GL-1", "page_ref": [1]}
    extraction = {"policy": {"policy_number": value}}
    row = {"doc_type": "policy", "lob": SELF_CONTAINED, "val_half": "calibration"}
    [(_features, correct)] = calibration_samples(
        [ValidationGeneration(row=row, golden=extraction, extraction=extraction)])["calibration"]
    assert correct


# --------------------------------------------------------------------------
# Schema validity is serving's verdict where serving gave one
# --------------------------------------------------------------------------


def _without_declarations(answer):
    return json.dumps({k: v for k, v in json.loads(answer).items()
                       if k not in ("carrier", "named_insured", "policy")})


@pytest.mark.parametrize("edit,rate", [(None, 1.0), (_without_declarations, 0.0)])
def test_a_policy_serving_refused_is_invalid_in_the_report(monkeypatch, edit, rate):
    """No window wrote the declarations: serving refuses the policy, but the
    JSON it serves has every key filled, so checked again it passed."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import golden_eval
    from evaluation.golden_eval import GoldenDocument
    from inference_core.model_runner import load_model
    from tests.test_schema2_serving import _Replay

    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})
    gold = _compact_auto()
    _result, _backend, answers = _serve_replay(gold, 3, edit=edit)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    doc = GoldenDocument(
        source_id="pa_1", doc_type="policy", golden=gold, lob="personal_auto",
        image_keys=[f"processed/default/policy/pa_1/page_{p}.png" for p in (1, 2, 3)],
        page_texts={1: "Declarations", 2: "Schedule page 2", 3: "Schedule page 3"},
    )
    triples = golden_eval.evaluate([doc], load_model("base", client, backend_impl=_Replay(answers)),
                                   client, "/tmp", modes=("ocr_plus_image",))
    metadata = triples[0][2]
    assert metadata["schema_valid"] is (rate == 1.0)
    if not metadata["schema_valid"]:
        assert "a required section no window wrote" in metadata["validation_errors"][0]
    assert _full(*triples[0][:2], **metadata).metrics["schema_validity_rate"] == rate


def test_without_a_verdict_a_common_model_answer_is_validated_as_before():
    got = copy.deepcopy(GOLD)
    del got["coverages"][0]["coverage_id"]
    assert _full(copy.deepcopy(GOLD), got).metrics["schema_validity_rate"] == 0.0
    assert _full(copy.deepcopy(GOLD), copy.deepcopy(GOLD)).metrics["schema_validity_rate"] == 1.0


def test_a_merged_answer_is_judged_as_serving_returns_it():
    """Checkpoint selection holds the merged answer, not serving's verdict. What
    the pipeline adds - the document's system fields, a part when the model read
    none - must not fail it (it read as a false 0%); a section the model never
    wrote still does."""
    from evaluation.metrics.coverage_metrics import score_schema_validity

    merged = copy.deepcopy(GOLD)
    merged["document"] = {}
    del merged["lob_parts"]
    assert score_schema_validity([("pa_1", merged, "policy", None, "personal_auto")]).rate == 1.0
    unwritten = copy.deepcopy(merged)
    del unwritten["policy"]
    assert score_schema_validity([("pa_1", unwritten, "policy", None, "personal_auto")]).rate == 0.0


def test_a_self_contained_line_is_validated_whatever_its_metadata_says():
    from tests.test_schema2_review_eval import _coverage, _location, _old_style

    doc = _old_style([_location("1", "Akron", [_coverage("PREM", "Location 1", "1000000")])])
    for verdict in (True, False):
        metrics = _full(doc, copy.deepcopy(doc), lob=SELF_CONTAINED, schema_valid=verdict).metrics
        assert metrics["schema_validity_rate"] == 0.0


# --------------------------------------------------------------------------
# A classic-auto part is the personal-auto part training teaches
# --------------------------------------------------------------------------


def _classic(doc):
    out = copy.deepcopy(doc)
    out["lob_parts"][0]["lob"] = "classic_auto"
    return out


def test_a_classic_auto_part_is_scored_as_the_part_training_teaches():
    """The decoder can only write personal_auto there, and training teaches it."""
    full = _full(_classic(GOLD), copy.deepcopy(GOLD), lob="classic_auto")
    assert full.error_records == []
    assert full.metrics["field_normalized_match"] == 1.0
    assert full.metrics["list_field_recall"] == 1.0
    assert full.metrics["auto_accept_error_rate"] == 0.0


def test_training_and_scoring_read_a_part_through_one_normalisation():
    from common.canonical import schema_label

    gold = _classic(GOLD)
    label = schema_label(gold, "policy", None, "classic_auto")
    assert label["lob_parts"][0]["lob"] == "personal_auto"
    assert gold["lob_parts"][0]["lob"] == "classic_auto", "the caller's label was changed"
    # A part of another line is left for training to refuse, as before.
    foreign = copy.deepcopy(GOLD)
    foreign["lob_parts"][0]["lob"] = "homeowners"
    assert schema_label(foreign, "policy", None, "personal_auto")["lob_parts"][0]["lob"] == "homeowners"
