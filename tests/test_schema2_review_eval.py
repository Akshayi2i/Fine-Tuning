"""Scoring a common-model line: what review found the report got wrong.

Each test below is a measurement that read a correct answer as a wrong one, or a
wrong one as correct, on a common-model line (SPEC_21) - or a common-model
change that reached the self-contained lines, which must score, type and
calibrate exactly as they did. The last section pins a self-contained report.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.run_eval import build_report

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
GOLD = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))
META = {"source_id": "pa", "doc_type": "policy", "lob": "personal_auto",
        "modality_mode": "ocr_plus_image"}

#: A line whose schema is still self-contained, in a layout family none of whose
#: lines has moved yet. gl was this file's until it moved (common model 1.1.0).
SELF_CONTAINED = "property"


def _full(got, *, gold=GOLD, **meta):
    [full] = build_report("t", [(gold, got, {**META, **meta})]).full_set()
    return full


def _comparable(doc, lob="personal_auto"):
    from common.structural_ids import comparable_view

    return comparable_view(doc, lob)


# --------------------------------------------------------------------------
# Schema validity: what was produced, not the scored copy
# --------------------------------------------------------------------------


def test_a_perfect_common_model_answer_is_schema_valid():
    """The scored copy is narrowed to the model view and has its ids taken out,
    so checked against the client's schema it failed on every document."""
    assert _full(copy.deepcopy(GOLD)).metrics["schema_validity_rate"] == 1.0


def test_an_answer_missing_a_required_id_is_still_invalid():
    """Validity is still asked of the real output: a coverage with no
    coverage_id is not the client's tree."""
    got = copy.deepcopy(GOLD)
    del got["coverages"][0]["coverage_id"]
    assert _full(got).metrics["schema_validity_rate"] == 0.0


def test_each_window_target_is_valid_against_its_slice():
    """A window's answer is the model form of its slice: nothing the pipeline
    fills in after the merge (a coverage_id) is asked of it, so none is required."""
    from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target

    plans = plan_windows("personal_auto", [1, 2, 3, 4, 5, 6], 1)
    assert plans
    for plan in plans:
        target = window_target(GOLD, "personal_auto", plan, TargetReport())
        metrics = _full(copy.deepcopy(target), gold=target, sections=plan.group).metrics
        assert metrics["schema_validity_rate"] == 1.0, plan


def test_the_callers_metadata_is_left_as_it_was():
    metadata = dict(META)
    build_report("t", [(GOLD, copy.deepcopy(GOLD), metadata)])
    assert metadata == META


# --------------------------------------------------------------------------
# A wrong code is one wrong value, not a missing and an invented row
# --------------------------------------------------------------------------


def test_one_wrong_coverage_code_costs_that_value_alone():
    """The row still pairs on what it applies to and the name it prints, so its
    correctly read limits and premium count as read: a wrong code costs what one
    wrong policy number costs, 1/N of field match."""
    got = copy.deepcopy(GOLD)
    got["coverages"][2]["coverage_code"] = "X_COMPREHENSIVE"
    full = _full(got)
    one_wrong = copy.deepcopy(GOLD)
    one_wrong["policy"]["policy_number"] = {**one_wrong["policy"]["policy_number"],
                                            "raw": "PA-2026-0043", "parsed": "PA-2026-0043"}
    baseline = _full(one_wrong)
    assert len(baseline.error_records) == 1
    assert full.metrics["field_normalized_match"] == baseline.metrics["field_normalized_match"] < 1.0
    assert [r["field_path"] for r in full.error_records] == ["coverages[2].coverage_code"]
    assert full.metrics["coverage_code_accuracy"] == 0.75


def test_an_accepted_value_in_a_row_with_a_wrong_code_is_not_counted_wrong():
    """Auto-accept pairs a common-model row as field match does."""
    got = copy.deepcopy(GOLD)
    got["coverages"][2]["coverage_code"] = "X_COMPREHENSIVE"
    assert _full(got).metrics["auto_accept_error_rate"] == 0.0


def test_unnamed_coverages_are_not_paired_on_what_they_apply_to_alone():
    from evaluation.metrics.field_accuracy import _pair_rows

    rows = [{"coverage_code": "A", "applies_to": [["v1"]]}]
    answer = [{"coverage_code": "B", "applies_to": [["v1"]]}]
    mates, unpaired = _pair_rows(rows, answer)
    assert mates == [None] and unpaired == rows


# --------------------------------------------------------------------------
# A self-contained line's applies_to is text, never part of a row's identity
# --------------------------------------------------------------------------


def _env(value):
    return {"raw": value, "parsed": value, "page_ref": [1]}


def _additional_coverages(applies):
    return {"auto": {"additional_coverages": [
        {"coverage_code": _env("HIRED"), "applies_to": _env(applies[0]), "limit": _env("1000000")},
        {"coverage_code": _env("NONOWN"), "applies_to": _env(applies[1]), "limit": _env("1000000")},
    ]}}


def test_a_differently_worded_applies_to_does_not_unpair_old_style_rows():
    """As on main: one wording of free text against another is one wrong value
    inside the row, not every row of the table missed."""
    gold = _additional_coverages(["Symbol 8", "Symbol 9"])
    got = _additional_coverages(["Sym. 8 - Hired", "Sym. 9 - Non-Owned"])
    metrics = _full(got, gold=gold, lob="auto", source_id="x").metrics
    assert metrics["list_field_recall"] == 1.0
    assert metrics["field_f1_list_fields"] == 1.0 and metrics["list_field_precision"] == 1.0


def test_applies_to_keys_a_row_only_where_it_holds_references():
    from evaluation.metrics.field_accuracy import _infer_key_fields

    text = [{"coverage_code": "HIRED", "applies_to": "Symbol 8"}]
    links = [{"coverage_code": "LIAB", "applies_to": [["vin", "1HGCM82633A004352"]]}]
    assert _infer_key_fields(text) == ["coverage_code"]
    assert _infer_key_fields(links) == ["coverage_code", "applies_to"]


# --------------------------------------------------------------------------
# False nulls compare rows by identity, not position
# --------------------------------------------------------------------------


def test_coverages_in_another_order_are_no_false_nulls():
    got = copy.deepcopy(GOLD)
    got["coverages"].reverse()
    metrics = _full(got).metrics
    assert metrics["false_null_rate"] == 0.0
    assert metrics["field_normalized_match"] == 1.0


def test_a_dropped_coverage_counts_only_its_own_values_as_false_nulls():
    from common.canonical import without_bare_values
    from evaluation.metrics.extraction_faults import _is_empty, score_false_nulls
    from evaluation.metrics.field_accuracy import flatten_scalars
    from evaluation.run_eval import _read_values

    got = copy.deepcopy(GOLD)
    dropped = got["coverages"].pop(1)
    report = score_false_nulls([_read_values(_comparable(GOLD), _comparable(got), True)])
    in_row = [v for v in flatten_scalars(without_bare_values(dropped)).values() if not _is_empty(v)]
    assert report.hits == len(in_row) > 0
    # The row the answer left out goes after the answer's own rows.
    position = len(got["coverages"])
    assert all(i["field"].startswith(f"coverages[{position}].") for i in report.instances)


# --------------------------------------------------------------------------
# The testing harness scores a common-model line as the gate does
# --------------------------------------------------------------------------


def _harness_scores(got, lob="personal_auto", gold=GOLD):
    from testing.run_extraction import score_against_ground_truth

    result = SimpleNamespace(doc_type="policy", extraction=got)
    return score_against_ground_truth(result, gold, lob=lob)


def test_the_harness_scores_values_inside_coverage_rows():
    """Skipping every table path, it read a policy with every limit wrong as perfect."""
    got = copy.deepcopy(GOLD)
    for coverage in got["coverages"]:
        for limit in coverage.get("limits", []):
            limit["amount"] = {**limit["amount"], "raw": "$1", "parsed": 1}
    scores = _harness_scores(got)
    gate = _full(got).metrics["field_normalized_match"]
    assert scores["field_normalized_match_rate"] == round(gate, 4) < 1.0
    assert any(f["field"].startswith("coverages[") for f in scores["failures"])


def test_the_harness_reports_links_codes_and_overflow_for_a_common_model_line():
    scores = _harness_scores(copy.deepcopy(GOLD))
    assert scores["field_normalized_match_rate"] == 1.0
    assert scores["reference_accuracy"] == 1.0 and scores["coverage_code_accuracy"] == 1.0
    assert scores["additional_fields_recall"] == 1.0


def test_the_harness_scores_a_self_contained_line_as_before():
    gold = {"policy": {"policy_number": _env("GL-1")},
            "locations": [{"location_number": _env("1"), "occupancy_description": _env("Office")}]}
    got = copy.deepcopy(gold)
    got["locations"][0]["occupancy_description"] = _env("Warehouse")
    scores = _harness_scores(got, lob=SELF_CONTAINED, gold=gold)
    # Table paths are scored by list recall there, never by field match.
    assert scores["field_normalized_match_rate"] == 1.0 and scores["failures"] == []
    assert scores["list_field_recall"] == {"locations": 1.0}
    assert "reference_accuracy" not in scores


# --------------------------------------------------------------------------
# One line, read once; a code's labels are its own
# --------------------------------------------------------------------------


def test_a_coverage_code_is_split_by_the_labels_of_its_code():
    from evaluation.metrics.unseen_labels import common_model_aliases, label_split

    aliases = common_model_aliases("personal_auto")
    key = next(k for k in aliases if k.startswith("coverages.coverage_code=") and aliases[k])
    code, label = key.split("=", 1)[1], aliases[key][0]
    result = SimpleNamespace(field_path="coverages[0].coverage_code", expected=code, correct=True)
    assert label_split([result], f"Coverage: {label}", aliases, seen={label}) == {
        "seen": [True], "unseen": []}
    assert label_split([result], f"Coverage: {label}", aliases, seen=set()) == {
        "seen": [], "unseen": [True]}


def test_classic_auto_is_reported_as_personal_auto(monkeypatch):
    from evaluation import run_eval

    monkeypatch.setattr(run_eval, "WORST_LINE_MIN_VALUES", 1)
    metrics = _full(copy.deepcopy(GOLD), lob="classic_auto").metrics
    assert metrics["field_accuracy_by_lob"] == {"personal_auto": 1.0}
    assert metrics["worst_line_field_match"] == 1.0


def test_a_line_given_as_a_list_finds_its_aliases():
    """The unseen-label split read the alias file named by str(['personal_auto'])."""
    text = "Policy Number PA-2026-0042"
    metrics = _full(copy.deepcopy(GOLD), lob=["personal_auto"], ocr_text=text,
                    seen_labels=["Policy Number"]).metrics
    assert metrics["seen_label_field_match"] == 1.0


# --------------------------------------------------------------------------
# Calibration types: the common model's table only on a common-model line
# --------------------------------------------------------------------------

#: What main's infer_field_type returns for paths the self-contained lines and
#: ACORD 140 share with the common model - and what they must still return.
MAIN_TYPES = {
    "billing.amount_due": "free_text",
    "carrier.address.state": "address",
    "carrier.address.postal_code": "address",
    "named_insured.mailing_address.postal_code": "address",
    "locations[0].location_number": "identifier",
    "billing.installments[0].installment_number": "identifier",
    "policy.rating_state": "free_text",
    "interested_parties[0].rank": "free_text",
    "deductibles[0].percentage": "free_text",
    "buildings[0].year_built": "free_text",
    "buildings[0].building_number": "identifier",
    "additional_fields[0].value": "free_text",
    "premium.surcharges[0].percentage": "free_text",
    "interested_parties[0].is_payor": "free_text",
}


@pytest.mark.parametrize("path,kind", sorted(MAIN_TYPES.items()))
def test_outside_a_common_model_line_a_field_keeps_mains_type(path, kind):
    from calibration.features import infer_field_type

    assert infer_field_type(path, 12.0) == kind
    assert infer_field_type(path, 12.0, common_model=False) == kind


def test_outside_a_common_model_line_its_table_is_never_read(monkeypatch):
    from calibration import features

    def refuse():
        raise AssertionError("the common model's types were consulted")

    monkeypatch.setattr(features, "common_model_field_types", refuse)
    for path in MAIN_TYPES:
        features.infer_field_type(path, 12.0)


def test_on_a_common_model_line_a_shared_path_takes_its_declared_type():
    from calibration.features import common_model_field_types, infer_field_type

    assert common_model_field_types()["billing.amount_due"] == "money"
    assert infer_field_type("billing.amount_due", common_model=True) == "money"
    assert infer_field_type("additional_fields[0].value", 12.0, common_model=True) == "number"


def test_document_features_are_typed_by_the_documents_line():
    from calibration.features import build_document_features

    extraction = {"billing": {"amount_due": "$120.00"}}

    def types(**flag):
        return {f.field_path: f.field_type
                for f in build_document_features(extraction=extraction, spans={}, **flag)}

    assert types() == {"billing.amount_due": "free_text"}
    assert types(common_model=True) == {"billing.amount_due": "money"}


def test_serving_calibrates_with_the_lines_types():
    """Serving's feature calibration passes the switch on to the features it asks about."""
    from serving.pipeline import _feature_calibrated

    seen: dict[bool, str] = {}

    class Recorder:
        def __init__(self, flag):
            self.flag = flag

        def predict(self, features):
            seen[self.flag] = features.field_type
            return None

    for flag in (False, True):
        _feature_calibrated(extraction={"billing": {"amount_due": "$120.00"}}, spans={},
                            calibrators=Recorder(flag), thresholds=None, page_text=None,
                            common_model=flag)
    assert seen == {False: "free_text", True: "money"}


@pytest.mark.parametrize("lob,kind", [("personal_auto", "money"), (SELF_CONTAINED, "free_text")])
def test_calibrators_are_fitted_under_the_types_serving_uses(lob, kind):
    from evaluation.validation_generation import ValidationGeneration, calibration_samples

    extraction = {"billing": {"amount_due": _env("$120.00")}}
    generation = ValidationGeneration(
        row={"doc_type": "policy", "lob": lob, "val_half": "calibration"},
        golden=extraction, extraction=extraction,
    )
    [(features, correct)] = calibration_samples([generation])["calibration"]
    assert features.field_path == "billing.amount_due" and correct
    assert features.field_type == kind


# --------------------------------------------------------------------------
# A self-contained report, pinned
# --------------------------------------------------------------------------


def _flagged(value, flagged=False):
    return {"raw": value, "parsed": value, "page_ref": [1], "flagged": flagged,
            "confidence": {"score": 0.9, "source": "vlm"}}


def _old_style(locations, amount="$500.00"):
    return {
        "policy": {"policy_number": _flagged("GL-77"), "rating_state": _flagged("OH")},
        "billing": {"amount_due": _flagged(amount)},
        "locations": locations,
    }


def _location(number, city, coverages):
    return {"location_number": _flagged(number), "address": {"city": _flagged(city)},
            "coverages": coverages}


def _coverage(code, applies_to, limit):
    return {"coverage_code": _flagged(code), "applies_to": _flagged(applies_to),
            "limit_amount": _flagged(limit)}


def test_a_self_contained_report_scores_as_before():
    """Every number of an old-style report with a wrong value, a reordered
    table, a missing row and an invented one, as it scored before the
    common-model fixes above."""
    gold = _old_style([
        _location("1", "Akron", [_coverage("PREM", "Location 1", "1000000"),
                                 _coverage("PRODCO", "All locations", "2000000")]),
        _location("2", "Dayton", [_coverage("PREM", "Location 2", "1000000")]),
        _location("3", "Toledo", []),
    ])
    got = _old_style([
        _location("2", "Dayton", [_coverage("PREM", "Location 2", "1000000")]),
        _location("1", "Akron", [_coverage("PREM", "Location 1", "1000000"),
                                 _coverage("PRODCO", "All locations", "3000000")]),
        _location("9", "Canton", []),
    ], amount="$550.00")
    metadata = {"source_id": "gl-1", "doc_type": "policy", "lob": SELF_CONTAINED,
                "modality_mode": "ocr_plus_image",
                "ocr_text": "GL-77 OH $500.00 Akron Dayton Toledo Location 1 Location 2 PREM"}
    [full] = build_report("t", [(gold, got, metadata)]).full_set()
    metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in full.metrics.items()}
    assert metrics == PINNED_OLD_STYLE
    assert [r["field_path"] for r in full.error_records] == ["billing.amount_due"]


#: The report above as the scoring produced it before those fixes - except
#: hallucination_rate, redefined on purpose (common.grounding): a value too short
#: to look for ("OH", "1") is no longer counted as grounded (0.3333 before); and
#: the page-list measures added since (accuracy plan, stage 0.3); and the line
#: it is filed under, gl until gl moved to the common model (1.1.0), every
#: number unchanged by the move; and how many values the line's accuracy is
#: over, which a continued run's gate reads (decision 2026-10-23).
PINNED_OLD_STYLE = {
    "auto_accept_error_rate": 0.2222, "confusable_misattribution_rate": 0.0,
    "false_null_rate": 0.1667, "field_accuracy_by_lob": {SELF_CONTAINED: 0.6667},
    "field_scored_by_lob": {SELF_CONTAINED: 3},
    "field_exact_match": 0.6667, "field_f1": 0.6667, "field_f1_list_fields": 0.6667,
    "field_normalized_match": 0.6667, "field_precision": 0.6667, "field_recall": 0.6667,
    "hallucination_rate": 0.3571, "list_field_precision": 0.6667, "list_field_recall": 0.6667,
    "lob_accuracy_by_value": {}, "schema_validity_rate": 0.0,
    "page_ref_exact_rate": 1.0, "page_ref_precision": 1.0, "page_ref_recall": 1.0,
}
