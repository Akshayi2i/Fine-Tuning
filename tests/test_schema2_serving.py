"""Serving a common-model line (SPEC_21): merge, ids, and the answer's contract.

The strongest check is a replay: every window serving asks for is answered with
the target training built for that very window. A pipeline whose training and
serving agree, and whose merge is sound, then returns the gold label itself -
values, units, links and all, renumbered in printed order.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.canonical import values_view, with_system_fields, without_bare_values
from common.schemas import iter_validation_errors
from common.structural_ids import resolve_references
from inference_core.model_runner import Generation, ModelBackend, load_model
from serving.doc_type_classifier import StaticClassifier
from serving.pipeline import ExtractionRequest, extract
from serving.policy_merge import PolicyWindow, merge_policy_windows

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
AUTO = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))


def _on_page(node, page):
    if isinstance(node, dict):
        if "raw" in node and "page_ref" in node:
            node["page_ref"] = [page]
        for value in node.values():
            _on_page(value, page)
    elif isinstance(node, list):
        for value in node:
            _on_page(value, page)


def _compact_auto():
    """The fixture on three pages: vehicles and their coverages on page 2."""
    gold = copy.deepcopy(AUTO)
    for name in ("coverages", "interested_parties", "forms_and_endorsements", "additional_fields"):
        _on_page(gold[name], 2 if name in ("coverages", "interested_parties") else 3)
    gold["additional_fields"][0]["page_ref"] = [3]
    gold["document"]["page_count"] = 3
    return gold


def _request_of(content) -> str:
    """A window's own request: under the fast format profile the group's block,
    the user turn's last text (every group of a window shares the system prompt
    and the pages there); none under the standard one."""
    text = content if isinstance(content, str) else "".join(
        part.get("text", "") for part in content if isinstance(part, dict))
    at = text.rfind("\n\n# Request: ")
    return text[at:] if at >= 0 else ""


def _replay_key(row) -> tuple:
    """The key :class:`_Replay` finds a training row's window by."""
    return (row["messages"][0]["content"], _request_of(row["messages"][1]["content"]), tuple(row["window_pages"]))


def _full_answer(row) -> dict:
    """A training row's target in the full ``raw``/``parsed``/``page_ref`` form,
    whichever form the row was written in (a fast-profile row is compact); the
    pipeline reads a full answer as it is."""
    from common.canonical import expand_answer

    return expand_answer(json.loads(row["messages"][-1]["content"]))


class _Replay(ModelBackend):
    """Answers a window with the target training built for it, found by the
    window's system prompt, its own request and its page markers."""

    def __init__(self, answers):
        self.answers = answers
        self.asked: list[tuple[str, tuple[int, ...]]] = []

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        pages = tuple(int(n) for n in re.findall(r"<page (\d+) of", json.dumps(messages[1]["content"])))
        key = (messages[0]["content"], _request_of(messages[1]["content"]), pages)
        self.asked.append(key)
        text = self.answers[key]
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens))


def _serve_replay(gold, pages, lob="personal_auto", *, strict=False, edit=None):
    """Serve ``gold`` by replaying its training targets; ``edit`` changes the
    answers first, ``strict`` validates as the endpoint does.

    The page text prints none of the gold's values, and the printed-values
    check is not what these tests are about (tests/test_printed_values.py): it
    only flags them here."""
    from unittest import mock

    import serving.printed_values as printed_values
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from tests.test_serving_pipeline import CALIBRATION

    texts = ["Declarations"] + [f"Schedule page {p}" for p in range(2, pages + 1)]
    images = [f"processed/default/policy/pa_1/page_{p}.png" for p in range(1, pages + 1)]
    doc = SourceDocument(source_id="pa_1", doc_type="policy", golden_label=gold, ocr_pages=texts,
                         image_paths=images, lob=lob, tenant_id="default")
    rows, _ = expand_document(doc, "train", modes=("ocr_plus_image",))
    answers = {_replay_key(r): r["messages"][-1]["content"] for r in rows}
    if edit is not None:
        answers = {key: edit(answer) for key, answer in answers.items()}
    backend = _Replay(answers)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    request = ExtractionRequest(
        source_id="pa_1", image_paths=images, page_texts=dict(enumerate(texts, start=1)),
        known_doc_type="policy", known_lob=lob, source_file_name="auto_fixture.pdf",
        ocr_meta={"modality": "native_pdf"},
    )
    with mock.patch.object(printed_values, "configured_mode", return_value=printed_values.FLAG):
        result = extract(request, load_model("base", client, backend_impl=backend),
                         StaticClassifier("policy"), CALIBRATION, strict_schema=strict)
    return result, backend, answers


def _comparable(doc):
    """Values only, references by the units' own keys, ids and confidence gone."""
    from common.model_view import PIPELINE_FILLED

    resolved = resolve_references(doc, "personal_auto")

    def strip(node):
        if isinstance(node, dict):
            if "raw" in node and "page_ref" in node:
                return values_view(node)
            return {k: strip(v) for k, v in node.items()
                    if k not in ("unit_id", "part_id", "coverage_id", "fideon:provenance", "text_sections")
                    and v not in (None, [], {})}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    out = strip(resolved)
    for name, fields in PIPELINE_FILLED.items():
        if isinstance(out.get(name), dict):
            for field_name in fields:
                out[name].pop(field_name, None)
    return out


def _no_nulls(node):
    if isinstance(node, dict):
        if "raw" in node and "page_ref" in node:
            return node if node.get("raw") is not None else None
        out = {k: _no_nulls(v) for k, v in node.items()}
        return {k: v for k, v in out.items() if v not in (None, [], {})}
    if isinstance(node, list):
        return [v for v in (_no_nulls(i) for i in node) if v not in (None, [], {})]
    return node


def test_replaying_the_training_targets_serves_the_gold_back():
    gold = _compact_auto()
    result, backend, answers = _serve_replay(gold, 3)
    assert set(backend.asked) == set(answers), "serving asked for windows training never built"
    assert result.schema_valid
    # The JSON served, every key filled, and not only the answer before the
    # fill: what schema_valid reports must be true of what is returned.
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == []
    served = _no_nulls(result.extraction)
    assert _comparable(served) == _comparable(gold)
    assert [c["coverage_id"] for c in result.extraction["coverages"]] == ["cov_1", "cov_2", "cov_3", "cov_4"]
    assert result.extraction["coverages"][2]["applies_to"] == ["veh_2"]


def test_the_served_answer_carries_what_the_pipeline_supplies():
    result, _backend, _answers = _serve_replay(_compact_auto(), 3)
    document = result.extraction["document"]
    assert document["doc_type"] == "policy_check" and document["modality"] == "native_pdf"
    assert document["page_count"] == 3 and document["source_file_name"] == "auto_fixture.pdf"
    assert result.extraction["policy"]["is_package"] is False
    assert "fideon:provenance" not in result.extraction
    assert not [f for f in result.review_flags if f.endswith(":no_confidence")
                and re.search(r"(unit_id|applies_to|coverage_code|limit_type|ref)", f)]


@pytest.mark.format_profile("standard")
def test_a_link_its_window_could_not_see_is_lost_but_the_rows_are_not():
    """Vehicles on page 2, their coverages on pages 4-5: two schedule windows.
    Each coverage is served, without the vehicle it applies to - the cost the
    reference-recall measurement sizes."""
    result, _backend, _answers = _serve_replay(AUTO, 6)
    coverages = result.extraction["coverages"]
    assert len(coverages) == 4
    assert all(not c.get("applies_to") for c in coverages)
    assert result.schema_valid
    # Served as "the whole policy" by the client's reading: two vehicles, so each
    # is flagged for a person (serving.pipeline.unit_link_flags).
    assert {f"coverages[{i}].applies_to:unit_link_unknown" for i in range(4)} <= set(result.review_flags)
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == []


def _stated(raw, page):
    return {"raw": raw, "parsed": raw, "confidence": {"score": 1.0, "source": "deterministic"},
            "page_ref": [page], "flagged": False}


def test_a_flat_deductible_printing_no_amount_is_served_under_strict_validation():
    """The model view leaves a flat deductible's amount out of its rules - a
    window can hold the row without it - where the client's rule requires the
    key. The every-key fill supplies it as a null, so the answer judged is the
    one served: a policy whose deductible prints its peril and no amount is
    served, not refused for an amount nobody printed."""
    gold = copy.deepcopy(AUTO)
    deductible = next(c for c in gold["coverages"] if c.get("deductibles"))["deductibles"][0]
    page = deductible["amount"]["page_ref"][0]
    deductible["amount"] = {"raw": None, "parsed": None, "confidence": {"score": 1.0, "source": "deterministic"},
                            "page_ref": [], "flagged": False}
    deductible["peril"] = _stated("Collision", page)
    assert list(iter_validation_errors(gold, "policy", None, "personal_auto")) == []

    result, _backend, _answers = _serve_replay(gold, 6, strict=True)
    assert result.schema_valid and "schema:invalid" not in result.review_flags
    served = [d for c in result.extraction["coverages"] for d in c.get("deductibles") or []
              if values_view(d.get("peril")) == "Collision"]
    assert len(served) == 1 and values_view(served[0]["amount"]) is None
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == []


def test_a_section_no_window_wrote_still_fails_validation():
    """Judging the filled answer must not pass off a section nobody read: with
    no window writing the carrier, the insured or the policy block, the fill
    would supply all three as nulls - and the answer is invalid all the same."""
    from serving.pipeline import PipelineError

    def without_declarations(answer):
        return json.dumps({k: v for k, v in json.loads(answer).items()
                           if k not in ("carrier", "named_insured", "policy")})

    result, _backend, _answers = _serve_replay(_compact_auto(), 3, edit=without_declarations)
    assert not result.schema_valid and "schema:invalid" in result.review_flags
    assert {"carrier: a required section no window wrote",
            "policy: a required section no window wrote"} <= set(result.validation_errors)
    with pytest.raises(PipelineError, match="a required section no window wrote"):
        _serve_replay(_compact_auto(), 3, strict=True, edit=without_declarations)


# --------------------------------------------------------------------------
# The merge
# --------------------------------------------------------------------------


def _v(raw, page=1, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [page]}


def _merge(*answers, lob="personal_auto"):
    windows = [PolicyWindow("arrays", [i + 1], copy.deepcopy(a)) for i, a in enumerate(answers)]
    return merge_policy_windows(windows, lob=lob)


def test_a_vehicle_read_in_two_windows_is_one_and_its_links_follow_it():
    first = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-A"), "year": _v("2019", parsed=2019)}]}
    second = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-B")},
                           {"unit_id": "veh_2", "vin": _v("VIN-A", 2), "make": _v("Honda", 2)}],
              "coverages": [{"coverage_code": "X_COLLISION", "coverage_name": _v("Collision", 2),
                             "applies_to": ["veh_2"]}]}
    merged = _merge(first, second)
    vehicles = merged.extraction["vehicles"]
    assert [(v["unit_id"], v["vin"]["raw"]) for v in vehicles] == [("veh_1", "VIN-A"), ("veh_2", "VIN-B")]
    assert vehicles[0]["make"]["raw"] == "Honda" and vehicles[0]["vin"]["page_ref"] == [1, 2]
    assert merged.extraction["coverages"][0]["applies_to"] == ["veh_1"]     # followed VIN-A
    assert not merged.conflicts


def test_two_windows_both_numbering_from_one_do_not_collide():
    first = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-A")}],
             "coverages": [{"coverage_code": "X_UM", "coverage_name": _v("UM"), "applies_to": ["veh_1"]}]}
    second = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-B", 2)}],
              "coverages": [{"coverage_code": "X_UM", "coverage_name": _v("UM", 2), "applies_to": ["veh_1"]}]}
    merged = _merge(first, second).extraction
    assert [c["applies_to"] for c in merged["coverages"]] == [["veh_1"], ["veh_2"]]
    assert [c["coverage_id"] for c in merged["coverages"]] == ["cov_1", "cov_2"]


def test_one_code_on_two_vehicles_stays_two_coverages():
    answer = {"vehicles": [{"unit_id": "veh_1", "vin": _v("A")}, {"unit_id": "veh_2", "vin": _v("B")}],
              "coverages": [{"coverage_code": "X_COLLISION", "premium": _v("$1", parsed=1), "applies_to": ["veh_1"]},
                            {"coverage_code": "X_COLLISION", "premium": _v("$2", parsed=2), "applies_to": ["veh_2"]}]}
    assert len(_merge(answer).extraction["coverages"]) == 2


def test_a_coverage_its_window_could_not_link_joins_the_one_row_it_matches():
    first = {"vehicles": [{"unit_id": "veh_1", "vin": _v("A")}],
             "coverages": [{"coverage_code": "X_COLLISION", "coverage_name": _v("Collision"),
                            "applies_to": ["veh_1"]}]}
    second = {"coverages": [{"coverage_code": "X_COLLISION", "premium": _v("$380", 2, 380)}]}
    merged = _merge(first, second)
    (coverage,) = merged.extraction["coverages"]
    assert coverage["applies_to"] == ["veh_1"] and coverage["premium"]["parsed"] == 380
    assert "coverages:joined_without_identifier" in merged.review_flags


def test_the_same_code_read_twice_is_agreement_not_a_conflict():
    first = {"coverages": [{"coverage_code": "X_UM", "coverage_name": _v("UM"), "limits": [
        {"limit_type": "per_person", "amount": _v("$1", parsed=1)}]}]}
    second = {"coverages": [{"coverage_code": "X_UM", "coverage_name": _v("UM", 2), "limits": [
        {"limit_type": "per_person", "amount": _v("$1", 2, 1)}]}]}
    merged = _merge(first, second)
    assert len(merged.extraction["coverages"]) == 1 and not merged.conflicts


def test_a_reference_to_no_unit_is_dropped_and_flagged():
    answer = {"vehicles": [{"unit_id": "veh_1", "vin": _v("A")}],
              "coverages": [{"coverage_code": "X_UM", "coverage_name": _v("UM"), "applies_to": ["veh_9"]}]}
    merged = _merge(answer)
    assert "applies_to" not in merged.extraction["coverages"][0]
    assert "coverages[0].applies_to:dangling_reference" in merged.review_flags


def test_a_self_contained_line_merges_exactly_as_before():
    # Each line named: no line is the fallback, a common-model overlay since 1.1.0.
    answer = {"auto": {"vehicles": [{"vin": _v("A")}]}}
    assert merge_policy_windows([PolicyWindow("lineblk", [1], copy.deepcopy(answer))],
                                lob="property").extraction == answer
    assert merge_policy_windows([PolicyWindow("lineblk", [1], copy.deepcopy(answer))],
                                lob="commercial_auto").extraction == answer


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------


def test_system_fields_are_plain_values_on_a_common_model_line():
    out = with_system_fields({"policy": {}}, page_count=4, source_file_name="x.pdf",
                             lob="personal_auto", modality="scanned_pdf")
    assert out["document"] == {"doc_type": "policy_check", "modality": "scanned_pdf",
                               "page_count": 4, "source_file_name": "x.pdf"}
    assert out["lob_parts"] == [{"part_id": "part_1", "lob": "personal_auto"}]
    assert out["policy"]["is_package"] is False and out["coverages"] == []
    # A self-contained line (general liability composes the common model since 1.1.0).
    old = with_system_fields({}, page_count=4, source_file_name="x.pdf", lob="property")
    assert old["document"]["page_count"]["parsed"] == 4                     # an envelope, as before


def test_an_unknown_modality_is_left_out():
    out = with_system_fields({}, page_count=1, lob="homeowners", modality="ocr_plus_image")
    assert "modality" not in out["document"]


def test_calibration_sees_only_what_was_read():
    answer = {"coverages": [{"coverage_code": "X_UM", "applies_to": ["veh_1"], "premium": _v("$1", parsed=1),
                             "limits": [{"limit_type": "per_person", "amount": _v("$2", parsed=2)}]}]}
    assert without_bare_values(answer) == {"coverages": [{"premium": _v("$1", parsed=1),
                                                          "limits": [{"amount": _v("$2", parsed=2)}]}]}


@pytest.mark.parametrize("lob", ["homeowners", "personal_auto"])
def test_the_every_key_fill_adds_no_annotation_keys(lob):
    from common.canonical import with_all_keys
    from common.schemas import _strip_prefixed, load_schema

    filled = with_all_keys({"policy": {}}, _strip_prefixed(load_schema("policy", None, lob), ("fideon:",)))
    assert not [k for k in filled if k.startswith("fideon:")]
    assert "policy_number" in filled["policy"]


# --------------------------------------------------------------------------
# The windowing ceiling (evaluation/windowing_ceiling.py)
# --------------------------------------------------------------------------


@pytest.mark.format_profile("standard")
def test_the_ceiling_counts_links_apart_from_values(no_window_overlap):
    from evaluation.windowing_ceiling import ceiling, oracle

    texts = ["Declarations"] + [f"Schedule page {p}" for p in range(2, 7)]
    split = oracle("pa", AUTO, "personal_auto", 6, texts)
    compact = oracle("pa2", _compact_auto(), "personal_auto", 3, texts[:3])
    assert split.recall == 1.0 and compact.recall == 1.0          # every value, either way
    assert (split.recovered_references, split.gold_references) == (2, 7)   # vehicle -> location only
    assert (compact.recovered_references, compact.gold_references) == (7, 7)
    assert split.dangling == 5
    summary = ceiling([split, compact])
    assert summary["reference_recall"] == round(9 / 14, 4) and summary["dangling_references"] == 5


def test_a_one_dwelling_homeowners_policy_raises_no_unit_link_flag():
    """Its liability and its mortgagee apply to the whole policy, and with one
    dwelling the policy and the unit are the same thing."""
    import json as _json

    from serving.pipeline import unit_link_flags

    example = _json.loads(Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")
                          .read_text(encoding="utf-8"))
    unlinked = {key: [{k: v for k, v in row.items() if k != "applies_to"} if isinstance(row, dict) else row
                      for row in value] if isinstance(value, list) else value
                for key, value in example.items()}
    assert unit_link_flags(unlinked) == []
    one_vehicle = {"vehicles": [{"unit_id": "veh_1"}], "drivers": [{"unit_id": "drv_1"}, {"unit_id": "drv_2"}],
                   "coverages": [{"coverage_code": "X_COLLISION"}]}
    assert unit_link_flags(one_vehicle) == []                                   # drivers do not count
    partial = {"vehicles": [{"unit_id": "veh_1"}],
               "coverages": [{"coverage_code": "X_COLLISION", "applies_to": ["veh_1"]}, {"coverage_code": "X_COLLISION"}]}
    assert unit_link_flags(partial) == ["coverages[1].applies_to:unit_link_unknown"]   # its twin is linked
