"""A common-model line's schema as the extraction model is shown and held to it.

The bundle is the client's contract and validation judges against all of it; the
model view (common.model_view) is what the prompt carries and the decoder
enforces. These tests pin the difference: what the model never writes, the rules
it is held to, the codes it may use, and what it must never see.
"""

from __future__ import annotations

import copy
import json
import re

import pytest
from jsonschema import Draft202012Validator

from common import schemas as S
from common.model_view import PIPELINE_FILLED, model_view_config

LINES = (
    "homeowners", "personal_auto", "dwelling_fire", "ocean_marine",
    "motorcycle", "recreational_vehicle", "personal_umbrella",
)
EXAMPLE_GOLD = S.CANONICAL_ROOT / "common schema" / "examples" / "homeowners_minimal.json"


def _view(lob):
    return S.resolved_schema("policy", None, lob)


def _text(lob):
    from common.prompts import schema_json_for_prompt

    return schema_json_for_prompt("policy", None, lob)


@pytest.mark.parametrize("lob", LINES)
def test_every_view_is_a_valid_schema_xgrammar_can_compile(lob):
    from tests.test_vllm_patches import _xgrammar_unsupported

    view = _view(lob)
    Draft202012Validator.check_schema(view)
    assert not _xgrammar_unsupported(view)
    assert not re.search(r'"(if|then|dependentRequired|minProperties|uniqueItems)"', json.dumps(view))


@pytest.mark.parametrize("lob", LINES)
def test_nothing_the_pipeline_fills_or_authors_annotate_reaches_the_model(lob):
    text = _text(lob)
    for marker in ("fideon:", "Confidence", '"confidence"', '"flagged"', "text_sections",
                   "TextSection", '"provenance"'):
        assert marker not in text, f"{lob}: {marker} reaches the model"
    defs = _view(lob)["$defs"]
    for name, fields in PIPELINE_FILLED.items():
        for field in fields:
            assert field not in (defs.get(name) or {}).get("properties", {}), f"{name}.{field}"


@pytest.mark.parametrize("lob", LINES)
def test_every_value_is_raw_parsed_and_its_pages(lob):
    for name, defn in _view(lob)["$defs"].items():
        props = defn.get("properties") or {}
        if "raw" in props:
            assert set(props) == {"raw", "parsed", "page_ref"}, name
            assert defn["required"] == ["raw", "parsed", "page_ref"] and not defn["additionalProperties"]
            assert props["page_ref"]["minItems"] == 1


@pytest.mark.parametrize("lob", LINES)
def test_a_coverage_code_is_one_of_the_lines_codes(lob):
    overlay = json.loads((S.CANONICAL_DIR / f"{lob}.json").read_text(encoding="utf-8"))
    code = _view(lob)["$defs"]["CoverageCode"]
    assert [c["const"] for c in code["oneOf"]] == overlay["fideon:coverage_codes"]
    assert all(c["description"] for c in code["oneOf"])
    coverage = _view(lob)["$defs"]["Coverage"]["properties"]["coverage_code"]
    assert coverage["$ref"] == "#/$defs/CoverageCode"


def test_the_line_of_a_part_is_the_lines_own():
    assert _view("ocean_marine")["$defs"]["LobPart"]["properties"]["lob"]["const"] == "ocean_marine"


@pytest.mark.parametrize("lob", LINES)
def test_only_objects_are_required_at_the_top(lob):
    assert S.required_fields("policy", None, lob) == ["document", "carrier", "named_insured", "policy"]


def _validates(lob, defn_name, row):
    view = _view(lob)
    schema = {"$defs": view["$defs"], "$ref": f"#/$defs/{defn_name}"}
    return Draft202012Validator(schema).is_valid(row)


def _v(raw, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [1]}


def test_a_sublimit_need_not_name_what_it_limits_in_a_window():
    """The client's rule (a sublimit has a description) is not the decoder's: a
    window can hold the sublimit without its description - printed on another
    page, or not stated - and a grammar requiring it would force one in."""
    bare = {"limit_type": "sublimit", "amount": _v("$1,500", 1500)}
    assert _validates("homeowners", "Limit", bare)
    assert _validates("homeowners", "Limit", {**bare, "description": _v("Theft of jewelry")})
    assert _validates("homeowners", "Limit", {"limit_type": "per_occurrence", "amount": _v("$1", 1)})
    assert not _validates("homeowners", "Limit", {**bare, "limit_type": "not_a_type"})


def test_a_percentage_limit_names_the_coverage_it_is_a_percentage_of():
    row = {"limit_type": "per_occurrence", "amount": _v("$175,000", 175000), "percentage": _v("50%", 50)}
    assert not _validates("homeowners", "Limit", row)
    assert _validates("homeowners", "Limit", {**row, "basis_coverage_code": "HO_COV_A"})
    assert not _validates("homeowners", "Limit", {**row, "basis_coverage_code": "NOT_A_CODE"})


def test_a_deductible_holds_whatever_its_window_shows_of_it():
    """Likewise a flat deductible's amount and a percentage one's percentage:
    either can be on another page than the row's type. The type keeps its list
    of values, and an unknown field is still refused."""
    assert _validates("homeowners", "Deductible", {"deductible_type": "flat"})
    assert _validates("homeowners", "Deductible", {"deductible_type": "flat", "amount": _v("$1,000", 1000)})
    assert _validates("homeowners", "Deductible", {"deductible_type": "percentage", "percentage": _v("2%", 2)})
    assert _validates("homeowners", "Deductible", {"deductible_type": "sir"})
    assert not _validates("homeowners", "Deductible", {"deductible_type": "not_a_type"})
    assert not _validates("homeowners", "Deductible", {"deductible_type": "flat", "note": _v("x")})


def _model_form(gold):
    """The SPEC_21 example gold as the model writes it."""
    out = copy.deepcopy(gold)
    for key in ("text_sections", "fideon:provenance"):
        out.pop(key, None)
    for name, fields in (("document", PIPELINE_FILLED["document"]), ("policy", PIPELINE_FILLED["policy"])):
        for field in fields:
            out.get(name, {}).pop(field, None)
    for cov in out.get("coverages", []):
        cov.pop("coverage_id", None)
        cov.pop("part", None)          # single-part policy: omitted, per its description
    for form in out.get("forms_and_endorsements", []):
        form.pop("page_range", None)

    def slim(node):
        if isinstance(node, dict):
            if {"raw", "parsed"} <= set(node):
                return {"raw": node["raw"], "parsed": node["parsed"], "page_ref": node["page_ref"]}
            return {k: slim(v) for k, v in node.items()}
        if isinstance(node, list):
            return [slim(v) for v in node]
        return node
    return slim(out)


def test_the_example_gold_as_the_model_writes_it_fits_the_view():
    gold = json.loads(EXAMPLE_GOLD.read_text(encoding="utf-8"))
    errors = [e.message for e in Draft202012Validator(_view("homeowners")).iter_errors(_model_form(gold))]
    assert not errors


@pytest.mark.parametrize("lob", LINES)
def test_the_view_stays_within_its_prompt_budget(lob):
    """About 8.8K tokens for homeowners, measured with the Qwen3-VL tokenizer;
    a model view that grows past this has started carrying what the model never
    writes. Common model 1.5.0 (2026-11-12) added optional fields to blocks every
    line uses: homeowners 42,460 characters, personal_auto 43,205, motorcycle
    41,114, recreational_vehicle 41,908, so the bound moved from 40,000 to 45,000."""
    assert len(_text(lob)) < 45_000


def test_alias_lists_never_reach_the_model():
    """The labels carriers print are corpus knowledge, never prompt text (master
    §1.4). A label may still appear where the schema itself says it - a coverage
    code's meaning, a description - but never as a list."""
    common = S._common_model()
    for lob in LINES:
        bundle = S.load_schema("policy", None, lob)
        aliases: set[str] = set()

        def collect(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "fideon:aliases" and isinstance(value, list):
                        aliases.update(value)
                    else:
                        collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)
        collect(bundle)
        collect(common.get("fideon:shared_coverage_codes"))
        own_words = json.dumps(_view(lob)["$defs"].get("CoverageCode", {})) + "".join(
            d for _, d in S.iter_described_fields(bundle, defs=bundle["$defs"]) if d)
        text = _text(lob)
        leaked = sorted(a for a in aliases
                        if len(a) >= 6 and f'"{a}"' in text and a not in own_words)
        assert not leaked, f"{lob}: alias strings in the model view: {leaked[:5]}"


def test_each_override_still_differs_from_the_clients_text():
    """An override that now matches the client's own text is stale: the client
    fixed it upstream, and the copy here would hide the next change they make."""
    defs = S._common_model()["$defs"]
    for path, text in (model_view_config().get("descriptions") or {}).items():
        name, _, field = path.partition(".")
        assert name in defs, f"{path}: the common model has no {name}"
        node = defs[name] if not field else (defs[name].get("properties") or {}).get(field)
        assert node is not None, f"{path}: the common model has no such field"
        assert node.get("description") != text, f"{path}: matches the client's text, remove it"


def _common_model_aliases(lob):
    """Every printed label the SPEC_21 sources record for this line: the
    overlay's, the common model's per field, and its coverage codes'."""
    from common.config import load_yaml

    found: set[str] = set()

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "fideon:aliases" and isinstance(value, list):
                    found.update(str(v) for v in value)
                elif key == "fideon:aliases" and isinstance(value, dict):
                    for labels in value.values():
                        found.update(str(v) for v in labels)
                else:
                    collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    overlay = json.loads((S.CANONICAL_DIR / f"{lob}.json").read_text(encoding="utf-8"))
    collect(overlay)
    collect(S.load_schema("policy", None, lob)["$defs"])
    for entry in load_yaml(S.CANONICAL_DIR / overlay["fideon:coverage_codes_file"]).get("codes") or []:
        found.update(str(a) for a in entry.get("aliases") or [])
    shared = S._common_model().get("fideon:shared_coverage_codes") or {}
    for code in overlay["fideon:coverage_codes"]:
        found.update(str(a) for a in (shared.get(code) or {}).get("aliases") or [])
    return found


@pytest.mark.parametrize("lob", LINES)
def test_no_common_model_alias_reaches_a_rendered_prompt(lob):
    """master §1.4, on every prompt a common-model line renders: whole and per
    window. A multi-word label may reach the prompt only where the schema itself
    says it (a description, a code's meaning); a one-word label collides with
    ordinary prose, as in the existing check."""
    from common.prompts import render_system_prompt
    from common.schema_sections import groups_for

    for sections in (None, *groups_for(lob)):
        rendered = render_system_prompt("policy", "ocr_plus_image", None, lob, sections)
        embedded = S.schema_text("policy", None, lob, sections)
        leaked = sorted(a for a in _common_model_aliases(lob)
                        if " " in a and a not in embedded and a in rendered)
        assert not leaked, f"{lob}/{sections}: {leaked[:5]}"
