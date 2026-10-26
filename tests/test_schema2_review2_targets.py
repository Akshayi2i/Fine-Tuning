"""Second review of the common-model training targets and the model view.

A fragment of a coverage row is left out of a window only when nothing is lost
by it: every value it holds is taught elsewhere. A coverage split at a window
boundary - its name at the foot of one page, its amounts at the top of the
next - was taught in no window at all, and serving then gave its row another
vehicle's amounts. And an umbrella's underlying policy codes another line's
coverages in its limits as well as in its own code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from common import schemas as S
from common.canonical import CanonicalLabelError, values_view
from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
HOME = json.loads((S.CANONICAL_ROOT / "common schema" / "examples" / "homeowners_minimal.json")
                  .read_text(encoding="utf-8"))
AUTO = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))
AUTO_PAGES = [1, 2, 3, 4, 5, 6]


def _fv(raw, parsed, pages):
    return {"raw": raw, "parsed": parsed, "confidence": {"score": 1.0, "source": "deterministic"},
            "page_ref": pages, "flagged": False}


def _client_valid(gold, lob):
    assert list(S.iter_validation_errors(gold, "policy", None, lob)) == []


def _targets(gold, lob="personal_auto", pages=AUTO_PAGES):
    report = TargetReport()
    out = [(plan, window_target(gold, lob, plan, report)) for plan in plan_windows(lob, pages, 1)]
    return out, report


def _assert_writable(targets, lob="personal_auto"):
    for plan, target in targets:
        view = S.resolved_schema("policy", None, lob, plan.group)
        errors = [e.message for e in Draft202012Validator(view).iter_errors(target)]
        assert not errors, (plan.group, plan.pages, errors[:3])


def _arrays(targets, pages):
    (target,) = [t for p, t in targets if p.group == "arrays" and p.pages == pages]
    return target


def _named_on_page_3(index):
    """The fixture with one coverage's name printed at the foot of page 3, its
    amounts and premium at the top of page 4: the row breaks at the boundary
    between the arrays windows over pages 1-3 and 4-6."""
    gold = copy.deepcopy(AUTO)
    gold["coverages"][index]["coverage_name"]["page_ref"] = [3]
    _client_valid(gold, "personal_auto")
    return gold


# T1 - a fragment is an orphan only when its values are taught elsewhere ----


@pytest.mark.format_profile("standard")
def test_a_coverage_split_at_the_window_boundary_keeps_its_values(no_window_overlap):
    """The liability's limits and $410 premium are printed only on page 4. Left
    out of the 4-6 window as a nameless fragment, they were taught nowhere."""
    targets, report = _targets(_named_on_page_3(0))
    _assert_writable(targets)
    assert report.orphaned == []
    first = _arrays(targets, (1, 2, 3))
    assert [values_view(c["coverage_name"]) for c in first["coverages"]] == ["Bodily Injury / Property Damage"]
    second = _arrays(targets, (4, 5, 6))
    (fragment,) = [c for c in second["coverages"] if "coverage_name" not in c]
    assert fragment["coverage_code"] == "X_VEHICLE_LIABILITY"
    assert [values_view(limit["amount"]) for limit in fragment["limits"]] == [100000, 300000, 50000]
    assert values_view(fragment["premium"]) == 410.0


@pytest.mark.format_profile("standard")
def test_a_collision_split_at_the_window_boundary_keeps_its_deductible(no_window_overlap):
    targets, report = _targets(_named_on_page_3(1))
    _assert_writable(targets)
    assert report.orphaned == []
    (fragment,) = [c for c in _arrays(targets, (4, 5, 6))["coverages"] if "coverage_name" not in c]
    assert fragment["coverage_code"] == "X_COLLISION"
    assert [values_view(d["amount"]) for d in fragment["deductibles"]] == [500]
    assert values_view(fragment["premium"]) == 380.0


@pytest.mark.format_profile("standard")
def test_a_fragment_holding_one_value_printed_only_here_is_kept():
    """Collision's premium is also printed on page 4, where its row is taught
    whole; its deductible only on page 2. The deductible would be lost."""
    gold = copy.deepcopy(AUTO)
    collision = gold["coverages"][1]
    collision["premium"]["page_ref"] = [4, 2]
    collision["deductibles"][0]["amount"]["page_ref"] = [2]
    _client_valid(gold, "personal_auto")
    targets, report = _targets(gold)
    _assert_writable(targets)
    assert "arrays:coverages[1]" not in report.orphaned
    (fragment,) = _arrays(targets, (1, 2, 3))["coverages"]
    assert values_view(fragment["deductibles"][0]["amount"]) == 500
    assert values_view(fragment["premium"]) == 380.0


@pytest.mark.parametrize("index, code, own, other", [
    (0, "X_VEHICLE_LIABILITY", 410.0, 455.0),
    (1, "X_COLLISION", 380.0, 420.0),
])
@pytest.mark.format_profile("standard")
def test_a_coverage_split_at_the_window_boundary_is_served_with_its_own_values(index, code, own, other):
    """Replayed through serving, the row named on page 3 joins its own amounts
    from page 4. With the fragment orphaned it joined the other vehicle's row
    from page 5, and that vehicle's coverage was lost."""
    from tests.test_schema2_serving import _serve_replay

    result, _, _ = _serve_replay(_named_on_page_3(index), 6, strict=True)
    assert result.schema_valid
    coverages = result.extraction["coverages"]
    assert len(coverages) == 4
    rows = [c for c in coverages if c["coverage_code"] == code]
    assert sorted(values_view(c["premium"]) for c in rows) == [own, other]
    (veh_1,) = [c for c in rows if "veh_1" in (c.get("applies_to") or [])]
    assert values_view(veh_1["premium"]) == own
    if code == "X_VEHICLE_LIABILITY":
        assert [values_view(limit["amount"]) for limit in veh_1["limits"]] == [100000, 300000, 50000]
    else:
        assert [values_view(d["amount"]) for d in veh_1["deductibles"]] == [500]


# T2 - an underlying policy's limits name another line's coverage ----------


def _umbrella(basis="X_VEHICLE_LIABILITY", own_basis=None):
    """A personal umbrella over an auto policy whose property-damage limit is
    stated as 40% of its liability coverage - an auto code, which the
    umbrella's own list cannot name."""
    gold = {k: copy.deepcopy(HOME[k]) for k in ("document", "carrier", "producer", "named_insured", "policy",
                                                "lob_parts")}
    gold["lob_parts"][0]["lob"] = "personal_umbrella"
    gold["premium"] = {"total": copy.deepcopy(HOME["premium"]["total"])}
    limits = [{"limit_type": "per_occurrence", "amount": _fv("$1,000,000", 1000000, [1])}]
    if own_basis is not None:
        limits.append({"limit_type": "aggregate", "percentage": _fv("200%", 200, [1]),
                       "basis_coverage_code": own_basis})
    gold["coverages"] = [{"coverage_id": "cov_1", "coverage_code": "PU_LIABILITY",
                          "coverage_name": _fv("Personal Umbrella Liability", "Personal Umbrella Liability", [1]),
                          "limits": limits}]
    gold["underlying_insurance"] = [{
        "carrier_name": _fv("Acme", "Acme", [2]), "policy_number": _fv("PA-1", "PA-1", [2]),
        "coverage_code": "X_VEHICLE_LIABILITY", "coverage_name": _fv("Auto Liability", "Auto Liability", [2]),
        "limits": [{"limit_type": "per_person", "amount": _fv("$250,000", 250000, [2])},
                   {"limit_type": "property_damage", "percentage": _fv("40%", 40, [2]),
                    "basis_coverage_code": basis}],
    }]
    _client_valid(gold, "personal_umbrella")
    return gold


def test_an_underlying_limit_takes_another_lines_coverage_code_as_its_basis():
    targets, _ = _targets(_umbrella(), "personal_umbrella", [1, 2, 3])
    _assert_writable(targets, "personal_umbrella")
    (arrays,) = [t for p, t in targets if p.group == "arrays"]
    (_, percentage) = arrays["underlying_insurance"][0]["limits"]
    assert percentage["basis_coverage_code"] == "X_VEHICLE_LIABILITY"


def test_a_coverages_own_limit_still_names_one_of_the_lines_codes():
    with pytest.raises(CanonicalLabelError, match=r"coverages\[0\]\.limits\[1\]"):
        _targets(_umbrella(own_basis="X_VEHICLE_LIABILITY"), "personal_umbrella", [1, 2, 3])
    targets, _ = _targets(_umbrella(own_basis="PU_LIABILITY"), "personal_umbrella", [1, 2, 3])
    _assert_writable(targets, "personal_umbrella")


@pytest.mark.parametrize("group", [None, "arrays"])
def test_an_underlying_limit_is_the_limit_with_a_plain_basis(group):
    """Its own copy of Limit, split into the same variants with the same
    descriptions; only the basis differs. Kept by the arrays slice too."""
    defs = S.resolved_schema("policy", None, "personal_umbrella", group)["$defs"]
    assert defs["UnderlyingPolicy"]["properties"]["limits"]["items"] == {"$ref": "#/$defs/UnderlyingLimit"}
    shared, underlying = defs["Limit"], defs["UnderlyingLimit"]
    assert len(underlying["anyOf"]) == len(shared["anyOf"]) == 2
    assert underlying["anyOf"][0]["required"] == shared["anyOf"][0]["required"]
    assert "percentage" not in underlying["anyOf"][1]["properties"]
    for mine, theirs in zip(underlying["anyOf"], shared["anyOf"]):
        basis = mine["properties"].get("basis_coverage_code")
        if basis is not None:
            assert basis["type"] == "string" and "$ref" not in basis
            assert theirs["properties"]["basis_coverage_code"]["$ref"] == "#/$defs/CoverageCode"
        rest = {k: v for k, v in mine["properties"].items() if k != "basis_coverage_code"}
        assert rest == {k: v for k, v in theirs["properties"].items() if k != "basis_coverage_code"}


def test_a_description_written_for_the_limit_describes_the_underlying_limit_too():
    from common.model_view import _with_copies

    overrides = {"Limit": "A limit.", "Limit.amount": "Its amount.", "Coverage": "A coverage.",
                 "UnderlyingLimit.amount": "The underlying amount."}
    assert _with_copies(overrides, {"UnderlyingLimit": "Limit"}) == {
        **overrides, "UnderlyingLimit": "A limit.",
    }


def test_a_line_without_underlying_policies_has_no_copy():
    for lob in ("homeowners", "personal_auto"):
        assert "UnderlyingLimit" not in S.resolved_schema("policy", None, lob)["$defs"]
