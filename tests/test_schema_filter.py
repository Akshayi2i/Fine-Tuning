"""Labels narrowed to what the line's schema can hold (common.canonical.within_schema).

The delivered labels carry annotation blocks (``fideon:*``, ``text_sections``,
``additional_fields``) and nested shapes the schema does not have - 22% of all
stated values in the delivery that introduced this. Decoding is held to the schema, so a key
it does not declare can never be written: in a training target it teaches a key
the grammar refuses, in a scored label it is a miss no model can fix.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from common.canonical import schema_label, training_target, within_schema

ROOT = Path(__file__).resolve().parents[1]


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


# A self-contained line: its line block is what labels get wrong. The common-model
# lines have no line block, and their own case comes with the target builder.
LINE = "equipment_breakdown"
LABEL = {
    "policy": {"policy_number": _env("CP-1"), "print_code": _env("X")},
    "equipment_breakdown": {
        "limits": {"business_income_limit": _env(1000000)},
        # A shape the labels invented: the schema has no `occurrences` list.
        "occurrences": [{"coverages": [{"coverage_name": _env("Premises"), "limit_amount": _env(250000)}]}],
    },
    "text_sections": {"page1_s1": "Declarations ..."},
    "fideon:provenance": {"source": "x"},
}


def test_keys_the_schema_does_not_declare_are_dropped_and_named():
    from common.schemas import resolved_schema

    kept, dropped = within_schema(LABEL, resolved_schema("policy", None, LINE))
    assert kept["policy"] == {"policy_number": LABEL["policy"]["policy_number"]}
    assert kept["equipment_breakdown"] == {"limits": {"business_income_limit": _env(1000000)}}
    assert {"policy.print_code", "equipment_breakdown.occurrences", "text_sections",
            "fideon:provenance"} <= set(dropped)


def test_an_envelope_is_kept_whole():
    from common.schemas import resolved_schema

    label = {"policy": {"policy_number": {**_env("CP-1"), "confidence": {"score": 1}, "flagged": False}}}
    kept, dropped = within_schema(label, resolved_schema("policy", None, LINE))
    assert kept == label and not dropped


def test_the_training_target_holds_only_writable_keys():
    target = training_target(LABEL, "policy", None, LINE)
    assert "occurrences" not in target.get("equipment_breakdown", {})
    assert "text_sections" not in target and "fideon:provenance" not in target


def test_the_window_target_holds_only_writable_keys():
    from data_pipeline.dataset_builder.policy_windows import plan_windows, window_target

    for plan in plan_windows(LINE, [1]):
        target = window_target(LABEL, LINE, plan)
        assert "occurrences" not in target.get("equipment_breakdown", {})
        assert "print_code" not in target.get("policy", {})


def test_a_gold_value_outside_the_schema_is_not_a_miss():
    from evaluation.run_eval import build_report

    got = {"policy": {"policy_number": _env("CP-1")},
           "equipment_breakdown": {"limits": {"business_income_limit": _env(1000000)}}}
    meta = {"source_id": "d1", "doc_type": "policy", "lob": LINE,
            "modality_mode": "ocr_plus_image"}
    [full] = build_report("t", [(LABEL, got, meta)]).full_set()
    assert full.metrics["field_normalized_match"] == 1.0


def test_flat_document_types_are_untouched():
    flat = {"insured_name": "X", "claims": [{"claim_number": "1"}]}
    assert schema_label(flat, "lossrun") == flat


def test_the_oracle_merge_reports_what_windowing_loses_and_what_the_schema_cannot_hold():
    spec = importlib.util.spec_from_file_location("diagnose_windowing",
                                                  ROOT / "scripts" / "diagnose_windowing.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)

    result = module.oracle("d1", LABEL, LINE, 1, None)
    assert result.recall == 1.0                      # one page: nothing lost to windows
    assert result.outside_schema == 3                # print_code, coverage_name, limit_amount
