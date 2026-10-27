"""Training targets for the common-model lines (SPEC_21), window by window.

A common-model gold links rows by id. A window is taught the rows it can see,
with ids counted from 1 in its own rows and no reference to a row it is not
shown; ids, codes and types travel only with a row that has a printed value on
the window's pages; and every target fits the window's own schema slice.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from common.schemas import CANONICAL_ROOT, resolved_schema
from common.structural_ids import renumber_structural_ids, resolve_references, unit_key
from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
HOME = json.loads((CANONICAL_ROOT / "common schema" / "examples" / "homeowners_minimal.json")
                  .read_text(encoding="utf-8"))
AUTO = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))


def _targets(gold, lob, pages, declarations=1):
    report = TargetReport()
    out = [(plan, window_target(gold, lob, plan, report))
           for plan in plan_windows(lob, pages, declarations)]
    return out, report


@pytest.mark.parametrize("gold,lob,pages", [(HOME, "homeowners", [1, 2, 3]),
                                            (AUTO, "personal_auto", [1, 2, 3, 4, 5, 6])])
def test_every_window_target_fits_its_slice(gold, lob, pages):
    targets, _ = _targets(gold, lob, pages)
    assert targets
    for plan, target in targets:
        view = resolved_schema("policy", None, lob, plan.group)
        errors = [e.message for e in Draft202012Validator(view).iter_errors(target)]
        assert not errors, (plan.group, plan.pages, errors[:3])


@pytest.mark.parametrize("gold,lob,pages", [(HOME, "homeowners", [1, 2, 3]),
                                            (AUTO, "personal_auto", [1, 2, 3, 4, 5, 6])])
def test_no_target_carries_what_the_model_never_writes(gold, lob, pages):
    targets, _ = _targets(gold, lob, pages)
    for _plan, target in targets:
        text = json.dumps(target)
        for key in ('"confidence"', '"flagged"', '"coverage_id"', '"page_count"', '"source_file_name"',
                    '"doc_type"', '"modality"', '"text_sections"', "fideon:", '"page_range"'):
            assert key not in text, key


def test_a_single_part_policy_is_taught_no_part():
    targets, _ = _targets(HOME, "homeowners", [1, 2, 3])
    assert all('"part"' not in json.dumps(target) for _plan, target in targets)


def test_ids_are_counted_from_one_in_each_window():
    """Vehicles on page 2 are veh_1 and veh_2 in the window that shows them."""
    targets, _ = _targets(AUTO, "personal_auto", [1, 2, 3, 4, 5, 6])
    (vehicles,) = [t["vehicles"] for p, t in targets if "vehicles" in t]
    assert [v["unit_id"] for v in vehicles] == ["veh_1", "veh_2"]
    assert all(v["garaging_location_ref"] == "loc_1" for v in vehicles)


@pytest.mark.format_profile("standard")
def test_a_reference_to_a_row_the_window_does_not_hold_is_left_out_and_counted(no_window_overlap):
    """The coverages on pages 4-5 apply to vehicles read in the window before."""
    targets, report = _targets(AUTO, "personal_auto", [1, 2, 3, 4, 5, 6])
    (coverages,) = [t["coverages"] for p, t in targets if "coverages" in t]
    assert len(coverages) == 4 and all("applies_to" not in c for c in coverages)
    assert len([d for d in report.dangling if "coverages" in d]) == 4
    assert any("interested_parties" in d for d in report.dangling)


def test_a_reference_inside_one_window_is_kept_and_renumbered():
    gold = copy.deepcopy(AUTO)
    for cov in gold["coverages"]:          # move every coverage onto the vehicle page
        for env in _envelopes(cov):
            env["page_ref"] = [2]
    targets, report = _targets(gold, "personal_auto", [1, 2, 3])
    (target,) = [t for p, t in targets if "coverages" in t]
    assert [c.get("applies_to") for c in target["coverages"]] == [["veh_1"], ["veh_1"], ["veh_2"], ["veh_2"]]
    assert not [d for d in report.dangling if "coverages" in d]


def test_ids_travel_only_with_a_row_the_window_shows():
    """A vehicle printed only on page 2 is not a structure-only row in the
    window over pages 4-6, though its unit_id is a value of the label."""
    targets, _ = _targets(AUTO, "personal_auto", [1, 2, 3, 4, 5, 6])
    later = [t for p, t in targets if p.group == "arrays" and 2 not in p.pages]
    assert later and all("vehicles" not in t and "locations" not in t for t in later)


def test_an_overflow_value_is_narrowed_and_trimmed_to_the_window():
    gold = copy.deepcopy(AUTO)
    gold["additional_fields"][0]["page_ref"] = [6, 9]
    gold["additional_fields"][0]["value"]["page_ref"] = [6, 9]
    targets, _ = _targets(gold, "personal_auto", [1, 2, 3, 4, 5, 6])
    (entries,) = [t["additional_fields"] for p, t in targets if "additional_fields" in t]
    assert entries == [{"label": "Good Student Discount",
                        "value": {"raw": "Applied", "parsed": "Applied", "page_ref": [6]},
                        "section_hint": "discounts", "page_ref": [6]}]


def test_only_a_date_value_is_reformatted():
    gold = copy.deepcopy(AUTO)
    gold["policy"]["effective_date"]["parsed"] = "2026-04-01"
    targets, _ = _targets(gold, "personal_auto", [1, 2, 3, 4, 5, 6])
    decl = next(t for p, t in targets if p.group == "decl")
    assert decl["policy"]["effective_date"]["parsed"] == "04/01/2026"
    forms = next(t for p, t in targets if p.group == "forms" and "forms_and_endorsements" in t)[
        "forms_and_endorsements"]
    assert forms[0]["edition_date"]["parsed"] == "09 18"            # a FieldValue, kept as printed


def _envelopes(node):
    if isinstance(node, dict):
        if "raw" in node and "page_ref" in node:
            yield node
        for value in node.values():
            yield from _envelopes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _envelopes(value)


# --------------------------------------------------------------------------
# common.structural_ids
# --------------------------------------------------------------------------


def _row(**fields):
    return {k: (v if k in ("unit_id", "garaging_location_ref", "applies_to", "coverage_code")
                else {"raw": str(v), "parsed": v, "page_ref": [1]}) for k, v in fields.items()}


def test_renumbering_follows_row_order_and_rewrites_every_reference():
    doc = {"vehicles": [_row(unit_id="veh_7", vin="A"), _row(unit_id="veh_3", vin="B")],
           "coverages": [_row(coverage_code="X_COLLISION", applies_to=["veh_3", "veh_7"])]}
    out, report = renumber_structural_ids(doc, "personal_auto")
    assert [v["unit_id"] for v in out["vehicles"]] == ["veh_1", "veh_2"]
    assert out["coverages"][0]["applies_to"] == ["veh_2", "veh_1"]
    assert report.renumbered["vehicles"] == {"veh_7": "veh_1", "veh_3": "veh_2"}
    assert doc["vehicles"][0]["unit_id"] == "veh_7"                     # a copy


def test_a_reference_to_no_row_is_dropped_and_reported():
    doc = {"vehicles": [_row(unit_id="veh_1", vin="A")],
           "coverages": [_row(coverage_code="X_COLLISION", applies_to=["veh_9"]),
                         _row(coverage_code="X_UM", applies_to=["veh_1", "veh_9"])]}
    out, report = renumber_structural_ids(doc, "personal_auto")
    assert "applies_to" not in out["coverages"][0]
    assert out["coverages"][1]["applies_to"] == ["veh_1"]
    assert len(report.dangling) == 2


def test_an_assigned_id_is_written_and_an_alias_resolves():
    doc = {"vehicles": [_row(unit_id="veh_1", vin="A")],
           "coverages": [_row(coverage_code="X_COLLISION", applies_to=["w2/veh_1"])]}
    out, report = renumber_structural_ids(doc, "personal_auto", extra_index={"w2/veh_1": "veh_1"},
                                          assign=("coverage_id",))
    assert out["coverages"][0]["coverage_id"] == "cov_1"
    assert out["coverages"][0]["applies_to"] == ["veh_1"] and not report.dangling


def test_a_unit_is_known_by_the_first_key_it_states():
    by_vin = _row(unit_id="veh_1", vin="1hgcv1f30ka000001", vehicle_number=1)
    by_number = _row(unit_id="veh_4", vehicle_number=1)
    assert unit_key("vehicles", by_vin, "personal_auto")[1] == ("vin",)
    assert unit_key("vehicles", by_number, "personal_auto")[1] == ("vehicle_number",)
    assert unit_key("vehicles", {"unit_id": "veh_2"}, "personal_auto") is None


def test_references_compare_by_the_units_own_keys():
    gold = {"vehicles": [_row(unit_id="veh_2", vin="AAA")],
            "coverages": [_row(coverage_code="X_COLLISION", applies_to=["veh_2"])]}
    answer = {"vehicles": [_row(unit_id="veh_1", vin="AAA")],
              "coverages": [_row(coverage_code="X_COLLISION", applies_to=["veh_1"])]}
    assert (resolve_references(gold, "personal_auto")["coverages"]
            == resolve_references(answer, "personal_auto")["coverages"])


# --------------------------------------------------------------------------
# Intake and corpus build
# --------------------------------------------------------------------------


def test_an_old_shape_gold_on_a_common_model_line_is_named_as_such():
    from common.schemas import old_shape_hint

    old = {"carrier": {"company_name": {"raw": "X", "parsed": "X"}}, "homeowners": {}, "terrorism": {}}
    hint = old_shape_hint(old, "policy", None, "homeowners")
    assert hint and "SPEC_21" in hint and "homeowners" in hint
    assert old_shape_hint(old, "policy", None, "equipment_breakdown") is None   # a self-contained line
    assert old_shape_hint(old, "policy", None, "gl")                        # gl.json composes the common model
    assert old_shape_hint(old, "policy", None, "property")                  # and property.json since 2.0.0
    assert old_shape_hint(HOME, "policy", None, "homeowners") is None


def test_the_twin_top_up_refuses_a_common_model_line(tmp_path):
    import csv

    from data_pipeline.ingestion.fill_synthetic_labels import fill_delivery

    (tmp_path / "Train/gold json").mkdir(parents=True)
    (tmp_path / "Train/gold json/ho.json").write_text(json.dumps(HOME), encoding="utf-8")
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "lob", "source", "kind", "gold", "pages"])
        writer.writeheader()
        writer.writerow({"split": "Train", "lob": "homeowners", "source": "A/ho.pdf", "kind": "synthetic",
                         "gold": "Train/gold json/ho.json", "pages": 3})
    report = fill_delivery(tmp_path, tmp_path / "reviewed")
    assert report.refused == {"common-model line: its twins are complete": 1} and report.filled == 0


def test_a_common_model_document_that_fails_to_expand_fails_the_build():
    from data_pipeline.dataset_builder.build_jsonl import CorpusBuildError, SourceDocument, build_corpus
    from data_pipeline.dataset_builder.split_groups import GroupSplitAssignment

    bad = SourceDocument(
        source_id="bad_1", doc_type="policy", golden_label={"policy": {"policy_number": "X"}},
        ocr_pages=["Declarations"], image_paths=["p/page_1.png"], lob="homeowners", tenant_id="default",
    )
    assignment = GroupSplitAssignment(assignment={bad.family: "train"})
    with pytest.raises(CorpusBuildError, match="common-model"):
        build_corpus([bad], assignment)
    result = build_corpus([bad], assignment, max_expansion_failures=1)
    assert result.set_aside["expansion"] == 1
