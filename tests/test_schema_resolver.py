"""The SPEC_21 line overlays load as one self-contained schema each (the bundle).

An overlay is a thin file whose blocks ``$ref`` the common model in another file.
Every reader in this repo - the validator, the section slicer, the prompt, the
every-key fill - expects one self-contained schema, so the registry bundles the
overlay at load time. These tests pin what the bundle is and what it refuses.
"""

import json
import re

import pytest

from common import schemas as S

OVERLAY_LINES = (
    "homeowners", "personal_auto", "dwelling_fire", "ocean_marine",
    "motorcycle", "recreational_vehicle", "personal_umbrella",
)
EXAMPLE_GOLD = S.CANONICAL_ROOT / "common schema" / "examples" / "homeowners_minimal.json"


def _overlay(lob: str) -> dict:
    return json.loads((S.CANONICAL_DIR / f"{lob}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lob", OVERLAY_LINES)
def test_an_overlay_is_registered_as_a_common_model_line(lob):
    assert S.is_common_model("policy", lob=lob)


def test_the_self_contained_lines_are_not_common_model_lines():
    # The lines not yet moved to the common model keep their self-contained files.
    for lob in ("business_owners", "commercial_package", "crime_fidelity", "epli", "management_liability"):
        assert not S.is_common_model("policy", lob=lob), lob


def test_general_liability_and_the_fallback_compose_the_common_model():
    """gl.json is a SPEC_21 overlay since its 3.0.0, and _fallback.json - the
    schema of a policy of no known line - composes the common model too, both
    pinned at 1.4.0 since 23 October 2026."""
    assert S.is_common_model("policy", lob="gl")
    assert S.is_common_model("policy")  # the fallback
    common = S._common_model()[S.COMMON_MODEL_VERSION_KEY]
    assert S.schema_version("policy", lob="gl") == f"{_overlay('gl')['fideon:schema_version']}+common.{common}"
    assert S.schema_version("policy") == f"{_overlay('_fallback')['fideon:schema_version']}+common.{common}"


@pytest.mark.parametrize("lob", OVERLAY_LINES)
def test_a_bundle_has_no_reference_outside_its_own_definitions(lob):
    text = json.dumps(S.load_schema("policy", lob=lob))
    refs = re.findall(r'"\$ref": "([^"]+)"', text)
    assert refs and all(ref.startswith("#/$defs/") for ref in refs)
    definitions = S.load_schema("policy", lob=lob)["$defs"]
    assert {ref.split("/")[2] for ref in refs} <= set(definitions)


@pytest.mark.parametrize("lob", OVERLAY_LINES)
def test_a_bundle_version_names_the_overlay_and_the_common_model(lob):
    """The overlay's own version and the common model it was COMPOSED with -
    the one loaded, though the overlay may pin an earlier minor."""
    common = S._common_model()[S.COMMON_MODEL_VERSION_KEY]
    assert S.schema_version("policy", lob=lob) == f"{_overlay(lob)['fideon:schema_version']}+common.{common}"


def test_the_spec21_example_gold_validates_against_its_bundle():
    gold = json.loads(EXAMPLE_GOLD.read_text(encoding="utf-8"))
    assert list(S.iter_validation_errors(gold, "policy", lob="homeowners")) == []


def test_a_gold_in_the_old_shape_is_refused_with_readable_errors():
    envelope = {"raw": "X", "parsed": "X", "confidence": {"score": 1.0, "source": "vlm"},
                "page_ref": [1], "flagged": False}
    old = {"carrier": {"company_name": envelope}, "named_insured": {}, "policy": {},
           "homeowners": {"dwelling": {}}}
    errors = list(S.iter_validation_errors(old, "policy", lob="homeowners"))
    assert any("homeowners" in e for e in errors)
    assert any("company_name" in e for e in errors)


def test_a_bundle_carries_each_coverage_codes_meaning():
    names = S.load_schema("policy", lob="homeowners")["fideon:coverage_code_names"]
    assert list(names) == _overlay("homeowners")["fideon:coverage_codes"]
    assert names["HO_COV_A"] and names["X_MED_PAY"]


def test_a_reference_into_another_file_is_refused():
    overlay = _overlay("homeowners")
    overlay["properties"]["carrier"] = {"$ref": "../common/other_model.json#/$defs/carrier"}
    with pytest.raises(S.SchemaError, match="outside the common model"):
        S._bundle(overlay, S.CANONICAL_DIR / "homeowners.json")


def test_a_pointer_outside_the_definitions_is_refused():
    overlay = _overlay("homeowners")
    overlay["properties"]["carrier"] = {"$ref": "../common/common_model.json#/properties/x"}
    with pytest.raises(S.SchemaError, match="outside the common model"):
        S._bundle(overlay, S.CANONICAL_DIR / "homeowners.json")


def test_an_overlay_written_against_another_common_model_version_is_refused():
    overlay = _overlay("homeowners")
    overlay[S.COMMON_MODEL_VERSION_KEY] = "9.9.9"
    with pytest.raises(S.SchemaError, match="9.9.9"):
        S._bundle(overlay, S.CANONICAL_DIR / "homeowners.json")


def test_an_earlier_minor_composes_and_a_later_or_other_major_does_not():
    """A minor release of the common model is additions only (its change
    control), so a line pinned at 1.1.0 composes 1.4.0, as the schema owner's
    registry composes it; a later version or another major is refused."""
    assert S.composes_common("1.4.0", "1.4.0") and S.composes_common("1.1.0", "1.4.0")
    for wanted in ("1.5.0", "2.0.0", "0.9.0", None, "one"):
        assert not S.composes_common(wanted, "1.4.0"), wanted
    overlay = _overlay("homeowners")
    overlay[S.COMMON_MODEL_VERSION_KEY] = "1.0.0"
    assert S._bundle(overlay, S.CANONICAL_DIR / "homeowners.json")


def test_a_code_file_listing_other_codes_is_refused(tmp_path):
    overlay = _overlay("personal_umbrella")
    (tmp_path / overlay["fideon:coverage_codes_file"]).write_text(
        "codes:\n  - {code: PU_SOMETHING_ELSE, name: Something}\n", encoding="utf-8"
    )
    with pytest.raises(S.SchemaError, match="different coverage codes"):
        S._bundle(overlay, tmp_path / "personal_umbrella.json")


def test_a_code_without_a_meaning_is_refused(tmp_path):
    overlay = _overlay("personal_umbrella")
    codes = "".join(f"  - {{code: {c}}}\n" for c in overlay["fideon:coverage_codes"])
    (tmp_path / overlay["fideon:coverage_codes_file"]).write_text(f"codes:\n{codes}", encoding="utf-8")
    with pytest.raises(S.SchemaError, match="no name"):
        S._bundle(overlay, tmp_path / "personal_umbrella.json")


def test_lines_on_the_common_model_and_lines_on_self_contained_files_load_together():
    """Lines move to the common model one by one: general liability, workers'
    comp and the businessowners lines are on it while EPLI and management
    liability are not yet. Every schema still loads; one corpus holding both
    kinds is refused where the adapter is trained
    (common.adapter.assert_one_output_shape), not here."""
    S._sources.cache_clear()
    sources = S._sources()
    assert all(sources[f"policy:{lob}"].common_model for lob in ("gl", "wc", "bopgl", "boppr"))
    assert not any(sources[f"policy:{lob}"].common_model for lob in ("epli", "management_liability"))


def test_commercial_auto_reads_the_clients_auto_file():
    assert S.schema_key("policy", lob="commercial_auto") == "policy:auto"


def test_classic_auto_is_not_a_line_of_its_own():
    assert "policy:classic_auto" not in S._sources()
    assert S.schema_key("policy", lob="classic_auto") == "policy:personal_auto"


@pytest.mark.parametrize("lob", [None, "gl", "wc", "auto"])
def test_no_prompt_asks_for_the_full_text_tier(lob):
    from common.prompts import schema_json_for_prompt

    assert "text_sections" not in schema_json_for_prompt("policy", lob=lob)


def test_the_prompt_hash_covers_the_common_model_and_the_code_files():
    from common.prompts import prompt_input_files

    files = prompt_input_files()
    assert S.COMMON_MODEL in files
    # No layout families any more (decision 2026-10-22), so no file of them to hash.
    assert not any(path.name == "layout_families.yaml" for path in files)
    for lob in OVERLAY_LINES:
        assert S.CANONICAL_DIR / f"{lob}.coverage_codes.yaml" in files
