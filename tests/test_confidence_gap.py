"""Closing the confidence gap: what may reach a user without review.

Three parts, each checked here:

* every field has the right TYPE, because the type sets the error a field may
  carry when accepted unreviewed (configs/field_types.yaml, proposed by
  scripts/propose_field_types.py);
* without fitted calibrators serving accepts nothing;
* the gate measures, on the golden set through serving, how many unreviewed
  values were wrong - against the golden label - and blocks above 2%.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from calibration import features
from calibration.features import FIELD_TYPES, field_type_table, infer_field_type
from calibration.thresholds import DEFAULT_ERROR_TARGETS
from evaluation.gating import (
    CONDITIONAL_METRICS,
    GATING_METRICS,
    PENDING_GATING_METRICS,
    PILOT_FLOORS,
)
from evaluation.metrics.auto_accept import score_auto_accept

ROOT = Path(__file__).resolve().parents[1]


def _proposer():
    spec = importlib.util.spec_from_file_location(
        "propose_field_types", ROOT / "scripts" / "propose_field_types.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(value, flagged=False):
    return {"raw": None if value is None else str(value), "parsed": value, "page_ref": [1],
            "confidence": {"score": 0.99, "source": "vlm"}, "flagged": flagged}


# --------------------------------------------------------------------------
# Field types
# --------------------------------------------------------------------------

def test_counts_years_and_percentages_have_a_type_and_a_target():
    assert "number" in FIELD_TYPES
    assert DEFAULT_ERROR_TARGETS["number"] == 0.01


def test_the_reviewed_table_outranks_the_name_heuristic(monkeypatch):
    monkeypatch.setattr(features, "field_type_table",
                        lambda: {"auto.vehicles.year": "number", "premium.policy_fee": "money"})
    assert infer_field_type("auto.vehicles[2].year") == "number"
    assert infer_field_type("premium.policy_fee") == "money"
    # Not in the table: the heuristic still answers.
    assert infer_field_type("carrier.address.city") == "address"


def test_a_typo_in_the_table_is_an_error_not_a_silent_free_text(tmp_path):
    table = tmp_path / "field_types.yaml"
    table.write_text("fields:\n  policy.policy_number: identifer\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field types"):
        field_type_table(table)


def test_the_reviewed_table_covers_every_canonical_field():
    if not features.FIELD_TYPE_TABLE.exists():
        pytest.skip("configs/field_types.yaml not reviewed yet")
    # A field the common model types itself is not tabled (propose_field_types):
    # its schema already gives it a calibrator, and a row would override that.
    declared = set(features.common_model_field_types())
    missing = _proposer().schema_fields() - set(field_type_table()) - declared
    assert not missing, f"fields with no reviewed type: {sorted(missing)[:20]}"


@pytest.mark.parametrize(("path", "values", "expected"), [
    # "count" inside "county" is not a count.
    ("locations.address.county_code", ["041"] * 10, "identifier"),
    # A form edition printed as digits is a code, matched exactly.
    ("premium.discounts_and_credits.edition_date", ["07 78"] * 10, "identifier"),
    # A unit wins over a money word.
    ("auto.physical_damage_coverages.rental_reimbursement_maximum_days", ["45"] * 10, "number"),
    ("auto.no_fault_benefits.maximum_monthly_work_loss", ["4000.0"] * 10, "money"),
    ("auto.vehicles.year", ["2019", "2021"] * 5, "number"),
    # Few distinct values alone is not a closed set: templates repeat.
    ("watercraft.watercraft.make", ["Manitou", "Grumman"] * 15, "free_text"),
    ("auto.drivers.gender", ["Male", "Female"] * 15, "enum"),
    # "hull" in a material is not a hull number.
    ("watercraft.watercraft.hull_material", [], "free_text"),
])
def test_proposed_types(path, values, expected):
    assert _proposer().propose(path, values)["type"] == expected


# --------------------------------------------------------------------------
# Auto-accept error, against the golden label
# --------------------------------------------------------------------------

GOLD = {
    "policy": {"policy_number": _env("HO-123"), "account_id": _env("A-1200")},
    "auto": {"vehicles": [{"vin": _env("1HGCM82633A004352"), "year": _env(2019)},
                          {"vin": _env("5TFDW5F11KX778685"), "year": _env(2021)}]},
}


def test_an_accepted_correct_value_is_not_an_error():
    tally = score_auto_accept(GOLD, {"policy": {"policy_number": _env("HO-123")}})
    assert (tally.accepted, tally.wrong, tally.rate) == (1, 0, 0.0)


def test_an_accepted_wrong_value_is_an_error():
    tally = score_auto_accept(GOLD, {"policy": {"account_id": _env("A-1300")}})
    assert (tally.accepted, tally.wrong) == (1, 1)


def test_a_flagged_wrong_value_reached_a_reviewer_not_a_user():
    tally = score_auto_accept(GOLD, {"policy": {"account_id": _env("A-1300", flagged=True)}})
    # Nothing accepted: no error rate among accepted values (None, not a
    # perfect 0.0), and a coverage of 0.
    assert (tally.accepted, tally.wrong, tally.rate, tally.coverage) == (0, 0, None, 0.0)


def test_an_accepted_value_the_label_does_not_hold_is_an_error():
    tally = score_auto_accept(GOLD, {"policy": {"policy_form": _env("HO-3")}})
    assert (tally.accepted, tally.wrong) == (1, 1)


def test_rows_are_matched_by_identifier_not_position():
    got = {"auto": {"vehicles": [{"vin": _env("5TFDW5F11KX778685"), "year": _env(2021)},
                                 {"vin": _env("1HGCM82633A004352"), "year": _env(2019)}]}}
    tally = score_auto_accept(GOLD, got)
    assert (tally.accepted, tally.wrong) == (4, 0)


def test_every_value_of_an_invented_row_is_an_error():
    got = {"auto": {"vehicles": [{"vin": _env("JH4KA7560MC000000"), "year": _env(1991)}]}}
    tally = score_auto_accept(GOLD, got)
    assert (tally.accepted, tally.wrong) == (2, 2)


def test_a_flat_extraction_without_flags_is_not_measured():
    assert score_auto_accept(GOLD, {"policy": {"policy_number": "HO-123"}}).rate is None


def test_an_omission_is_not_an_accepted_value():
    """Left out reaches nobody as accepted; recall and false nulls measure it."""
    tally = score_auto_accept(GOLD, {"policy": {"policy_number": _env(None)}})
    assert (tally.flags_seen, tally.accepted) == (0, 0)


def test_build_report_emits_the_rate():
    from evaluation.run_eval import build_report

    got = {"policy": {"policy_number": _env("HO-123"), "account_id": _env("A-999")}}
    # A self-contained line, whose policy block holds account_id: the common
    # model (general liability since 1.1.0, property since 1.4.0) has no such
    # field, so it would be scored on neither side and the wrong value would
    # never be seen.
    meta = {"source_id": "p1", "doc_type": "policy", "lob": ["equipment_breakdown"],
            "modality_mode": "ocr_plus_image"}
    [full] = build_report("t", [(GOLD, got, meta)]).full_set()
    assert full.metrics["auto_accept_error_rate"] == 0.5


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_the_gate_enforces_it_as_a_ceiling():
    assert GATING_METRICS["auto_accept_error_rate"] == "lower_is_better"
    assert "auto_accept_error_rate" not in PENDING_GATING_METRICS
    assert PILOT_FLOORS["auto_accept_error_rate"] == 0.02
    # Only canonical outputs carry flags; when measured it faces its ceiling.
    assert "auto_accept_error_rate" in CONDITIONAL_METRICS


def test_a_release_whose_unreviewed_values_are_3pct_wrong_is_blocked():
    from evaluation.gating import promotion_gate

    metrics = {name: 0.96 for name in GATING_METRICS}
    metrics.update(schema_validity_rate=1.0, ece_confidence=0.04,
                   confusable_misattribution_rate=0.02, false_null_rate=0.03,
                   auto_accept_error_rate=0.03)
    result = promotion_gate(metrics, None)
    assert not result.passed and "auto_accept_error_rate" in result.failed_gates


# --------------------------------------------------------------------------
# Serving without calibrators
# --------------------------------------------------------------------------

def test_without_calibrators_serving_accepts_nothing():
    """The v1 transform accepted a field above a fixed 0.70, an error rate
    nobody measured. Its confidence is still reported; nothing is accepted."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract
    from tests.test_serving_pipeline import CALIBRATION, RESPONSE, _request, model  # noqa: F401

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    loaded = load_model("base", client, backend_impl=EchoBackend(RESPONSE))
    result = extract(_request(), loaded, StaticClassifier("policy"), CALIBRATION)
    assert result.fields, "the response must hold fields for this test to mean anything"
    flagged = {flag.rsplit(":", 1)[0] for flag in result.review_flags}
    assert set(result.fields) <= flagged, set(result.fields) - flagged
