"""The fast format profile (configs/shared/format.yaml): images first with one
window plan for every group, a compact prompt, a smaller pixel budget and
compact answers - written the same way into training rows and serving requests,
and refused across profiles.

Synthetic documents only: every text, label and answer here is made up, or the
repo's synthetic test fixture (tests/fixtures/schema2).
"""

from __future__ import annotations

import copy
import json
import re

import pytest
from jsonschema import Draft202012Validator

from common import config
from common.canonical import (
    COMPACT_ANSWER_FORMAT,
    collapse_spans,
    compact_schema,
    expand_answer,
    to_compact_answer,
)
from common.prompts import (
    FAST_TEMPLATE_DIR,
    prompt_hash,
    prompt_input_files,
    render_group_block,
    render_system_prompt,
)
from common.schema_sections import groups_for
from common.schemas import resolved_schema, with_page_bounds
from data_pipeline.dataset_builder.policy_windows import (
    assert_window_plan,
    pages_to_read,
    plan_windows,
    window_plan_id,
)
from inference_core.input_builder import build_messages
from inference_core.model_runner import Generation, ModelBackend
from inference_core.span_map import map_compact_field_spans

LINES = ("personal_auto", "homeowners")


def _profile(monkeypatch, name: str) -> None:
    monkeypatch.setattr(config, "shared_format_config", lambda: {"profile": name})


@pytest.fixture
def fast(monkeypatch):
    _profile(monkeypatch, "fast")


def _fv(raw, *pages, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": list(pages)}


def _images(count: int) -> list[str]:
    return [f"processed/default/policy/fp_1/page_{p}.png" for p in range(1, count + 1)]


def _texts(count: int) -> list[str]:
    return (["Personal Auto Declarations. Named Insured: JANE RIVERA. Policy Number PA-1."]
            + [f"Schedule page {p}. VIN 1HGCM82633A00{p:04d} Liability 100/300" for p in range(2, count + 1)])


# --------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------


def test_fast_is_the_shipped_profile():
    """The user's decision of 2026-10-26: the next corpus is built, trained and
    served in the fast profile."""
    assert config.load_yaml(config.SHARED_FORMAT_CONFIG)["profile"] == "fast"
    assert config.format_profile() == "fast" and config.fast_profile()


def test_a_config_naming_no_profile_reads_as_standard(monkeypatch):
    monkeypatch.setattr(config, "shared_format_config", lambda: {})
    assert config.format_profile() == "standard" and not config.fast_profile()


@pytest.mark.format_profile("standard")
def test_standard_writes_what_it_wrote_before():
    assert config.format_profile() == "standard" and not config.fast_profile()
    assert render_group_block("policy", None, "personal_auto", "decl") is None
    built = build_messages("policy", _images(2), _texts(2), "ocr_plus_image",
                           lob="personal_auto", sections="decl", check_resolution_parity=False)
    # Image then its text, page by page, and nothing after: as before.
    assert [b["type"] for b in built.messages[1]["content"]] == ["image", "text", "image", "text"]
    assert built.answer_format is None and "Schema:" in built.system_prompt()
    assert not [p for p in prompt_input_files() if FAST_TEMPLATE_DIR in p.parents]
    assert window_plan_id().startswith("4:")
    assert config.pixel_budget() == (200704, 2483200)


def test_an_unknown_profile_is_refused(monkeypatch):
    _profile(monkeypatch, "fastest")
    with pytest.raises(config.ConfigError, match="fastest"):
        config.format_profile()


# --------------------------------------------------------------------------
# 1. Images first, the same windows for every group
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lob", LINES)
@pytest.mark.parametrize("mode", ["ocr_plus_image", "image_only"])
def test_every_group_of_a_window_sends_the_same_tokens_up_to_its_block(fast, lob, mode):
    """The prefix the engine's cache reuses: for one window, the serialized
    messages of every group are identical up to the final group block."""
    texts = None if mode == "image_only" else _texts(3)
    serialized = {}
    for group in groups_for(lob):
        messages = build_messages("policy", _images(3), texts, mode, lob=lob, sections=group,
                                  page_numbers=[1, 2, 3], total_pages=9,
                                  check_resolution_parity=False).messages
        content = messages[1]["content"]
        # Every page image, then its marker and text, as in standard; the block last.
        assert [b["type"] for b in content] == ["image", "text"] * 3 + ["text"]
        assert content[-1]["text"].startswith("\n\n# Request: ")
        serialized[group] = (json.dumps([messages[0], {"role": "user", "content": content[:-1]}]),
                             json.dumps(content[-1]))
    prefixes = {prefix for prefix, _block in serialized.values()}
    assert len(prefixes) == 1, "a group's messages differ before its block"
    assert len({block for _prefix, block in serialized.values()}) == len(serialized)


def test_training_rows_share_the_prefix_too(fast):
    """The rows the corpus builds for one window, one per group: the same
    system message and pages, each its own block last - as served."""
    rows = _rows(9)
    by_window: dict[tuple[int, ...], list[dict]] = {}
    for row in rows:
        by_window.setdefault(tuple(row["window_pages"]), []).append(row)
    assert {len(group_rows) for group_rows in by_window.values()} == {len(groups_for("personal_auto"))}
    for group_rows in by_window.values():
        heads = {json.dumps(r["messages"][:1] + [{"content": r["messages"][1]["content"][:-1]}])
                 for r in group_rows}
        assert len(heads) == 1


@pytest.mark.parametrize("pages", [3, 9, 31, 120])
def test_every_group_reads_the_same_windows(fast, pages):
    texts = _texts(pages)
    every, declarations = pages_to_read(texts, pages)
    plans = plan_windows("personal_auto", every, declarations, texts)
    by_group: dict[str, list[tuple[int, ...]]] = {}
    for plan in plans:
        by_group.setdefault(plan.group, []).append(plan.pages)
    assert len({tuple(windows) for windows in by_group.values()}) == 1
    windows = next(iter(by_group.values()))
    assert set().union(*map(set, windows)) == set(every)
    # The declarations lead - and only the declarations group marks it so.
    led = [plan for plan in plans if plan.leads]
    assert {plan.group for plan in led} <= {"decl"} and len(led) <= 1
    assert led and set(led[0].pages) <= {1, 2, 3, 4, 5}
    if pages > 5:
        # The page after the leading ones is read beside the last of them.
        assert any({5, 6} <= set(window) for window in windows)


def test_the_declarations_page_found_later_leads_the_windows(fast):
    texts = [f"Notice page {p}." for p in range(1, 13)]
    texts[8] = "COMMON POLICY DECLARATIONS named insured policy number policy period"
    every, declarations = pages_to_read(texts, 12)
    assert declarations == 9
    plans = plan_windows("personal_auto", every, declarations, texts)
    decl = [plan for plan in plans if plan.group == "decl"]
    assert decl[0].pages == (1, 2, 3, 4, 5, 9) and decl[0].leads
    assert {plan.pages for plan in plans if plan.group == "arrays"} == {plan.pages for plan in decl}


@pytest.mark.format_profile("standard")
def test_the_fast_windows_hold_more_pages_than_the_standard_ones(monkeypatch):
    from data_pipeline.dataset_builder.expand_tasks import pages_per_extraction_call

    standard = {g: pages_per_extraction_call("policy", None, "personal_auto", g) for g in groups_for("personal_auto")}
    _profile(monkeypatch, "fast")
    fast_ = {g: pages_per_extraction_call("policy", None, "personal_auto", g) for g in groups_for("personal_auto")}
    # The binding group (a schedule's answer reservation) gains the most.
    assert min(fast_.values()) > min(standard.values())
    assert all(fast_[g] >= standard[g] for g in standard)


@pytest.mark.parametrize("dense", [False, True])
def test_training_and_serving_plan_the_same_windows(fast, dense):
    texts = _texts(9)
    if dense:
        texts = [texts[0]] + [t + " " + " ".join(["Liability 100/300 Premium 120.00"] * 900) for t in texts[1:]]
    rows = _rows(9, texts=texts)
    trained = sorted((row["sections"], tuple(row["window_pages"])) for row in rows)
    backend = _Scripted(lambda group, pages: "{}")
    _extract(backend, _request(texts))
    assert sorted((group, tuple(pages)) for group, pages in backend.calls) == trained


# --------------------------------------------------------------------------
# 2. The compact prompt
# --------------------------------------------------------------------------


@pytest.mark.format_profile("standard")
@pytest.mark.parametrize("lob", LINES)
def test_the_group_block_lists_every_field_compactly(monkeypatch, lob):
    """A short field list in place of the schema JSON, every key of the slice
    named. Measured 2026-10-24 (characters): personal_auto decl 31,997 ->
    11,313 group block + a 10,227 shared system message; arrays 33,106 ->
    11,501; forms 13,399 -> 1,218; homeowners decl 33,145 -> 12,136, arrays
    33,243 -> 12,021, forms 13,399 -> 1,218."""
    standard = {g: render_system_prompt("policy", "ocr_plus_image", None, lob, g) for g in groups_for(lob)}
    _profile(monkeypatch, "fast")
    shared = {render_system_prompt("policy", "ocr_plus_image", None, lob, g) for g in groups_for(lob)}
    assert len(shared) == 1
    (system,) = shared
    assert "Schema:" not in system and '"$defs"' not in system
    for group in groups_for(lob):
        block = render_group_block("policy", None, lob, group)
        assert "$ref" not in block and '"properties"' not in block
        # The non-shared part of a request is well under half the standard prompt.
        assert len(block) < 0.4 * len(standard[group]), group
        for key in resolved_schema("policy", None, lob, group).get("properties") or {}:
            assert f"`{key}`" in block


def test_the_field_list_names_kinds_codes_and_meanings(fast):
    block = render_group_block("policy", None, "personal_auto", "arrays")
    assert "coverages: [Coverage] - Every coverage and every limit." in block
    assert "  coverage_code: CoverageCode - " in block
    assert "X_COLLISION = Collision" in block
    assert re.search(r"^date: string - A date\.", block, re.M)
    assert "  unit_id: id - " in block
    # A union of row variants is listed once, with its whole meaning.
    assert "Limit - A limit. Always inside the coverage it limits. Split limits" in block


# --------------------------------------------------------------------------
# 3. The fast pixel budget
# --------------------------------------------------------------------------


def test_the_fast_pixel_budget_is_read_by_the_corpus_the_trainer_and_the_engine(fast):
    from data_pipeline.dataset_builder.cap_check import PIXELS_PER_VISUAL_TOKEN, estimate_visual_tokens

    assert config.pixel_budget() == (200704, 1254400)
    assert 1254400 % PIXELS_PER_VISUAL_TOKEN == 0
    assert estimate_visual_tokens(1, "policy_schedule") == 1225
    # The thumbnail task keeps its own budget.
    assert config.vision_for_task("classify")["max_pixels"] == 262144
    from training.train import _pixel_budget

    env = _pixel_budget()
    assert env["MAX_PIXELS"] == "1254400" and env["IMAGE_MAX_TOKEN_NUM"] == "1225"


# --------------------------------------------------------------------------
# 4. The compact answer
# --------------------------------------------------------------------------


def _gold_answer() -> dict:
    """A synthetic model-form answer exercising every envelope case."""
    return {
        "policy": {
            "policy_number": _fv("PA-2026-0042", 1),
            "effective_date": _fv("April 1, 2026", 1, parsed="04/01/2026"),
            "term_months": _fv("12", 1, parsed=12),
            "subject_to_audit": _fv("Yes", 1, parsed=True),
        },
        "premium": {"total": _fv("$1,240.50", 1, parsed=1240.5), "deposit": _fv("100", 1, parsed=100)},
        "coverages": [
            {"coverage_code": "X_COLLISION", "applies_to": ["veh_1"],
             "coverage_name": _fv("Collision", 2),
             "limits": [{"limit_type": "per_accident", "amount": _fv("ACV", 2, parsed=None)}],
             "premium": _fv("0", 2, parsed=0.0)},
        ],
        "additional_fields": [{"label": "Garaging ZIP", "value": _fv("12180", 2, 3), "page_ref": [2, 3]}],
    }


def test_full_to_compact_to_expanded_is_the_full_answer():
    full = _gold_answer()
    compact = to_compact_answer(copy.deepcopy(full))
    assert expand_answer(compact) == full
    policy = compact["policy"]
    assert policy["policy_number"] == {"v": "PA-2026-0042", "p": [1]}
    assert policy["effective_date"] == {"r": "April 1, 2026", "v": "04/01/2026", "p": [1]}
    assert policy["term_months"] == {"v": 12, "p": [1]}           # "12" reads as 12
    assert list(policy["subject_to_audit"]) == ["r", "v", "p"]     # a yes/no keeps its mark
    assert compact["premium"]["total"]["r"] == "$1,240.50"
    assert compact["coverages"][0]["premium"]["r"] == "0"           # 0.0 prints as 0.0, not "0"
    assert compact["coverages"][0]["coverage_code"] == "X_COLLISION"
    # Shorter even on this answer of mostly reformatted values: 900 -> 672 characters.
    assert len(json.dumps(compact)) < 0.8 * len(json.dumps(full))


@pytest.mark.format_profile("standard")
def test_the_compact_answers_are_shorter_on_a_whole_policy():
    """Measured 2026-10-24 on the synthetic 3-page auto fixture's window
    targets: 5,252 -> 3,707 characters (-29%); its 6-page form 6,648 -> 4,724."""
    full = [row["messages"][-1]["content"] for row in _rows(3)]
    compact = [json.dumps(to_compact_answer(json.loads(text)), ensure_ascii=False) for text in full]
    assert sum(map(len, compact)) < 0.75 * sum(map(len, full))


@pytest.mark.parametrize("lob", LINES)
def test_no_schema_object_could_be_taken_for_a_compact_value(lob):
    """``is_compact_value`` reads an object of keys among v, r, p as a value:
    no object of any line's schema may have only such keys."""
    def walk(node, seen):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and props and set(props) <= {"v", "r", "p"}:
                seen.append(sorted(props))
            for value in node.values():
                walk(value, seen)
        elif isinstance(node, list):
            for value in node:
                walk(value, seen)

    for group in (None, *groups_for(lob)):
        seen: list[list[str]] = []
        walk(resolved_schema("policy", None, lob, group), seen)
        assert not seen, (group, seen)


def test_every_compact_training_target_is_writable_by_the_compact_grammar(fast):
    """The rows' compact answers validate against the decoding schema serving
    sends: the window's slice, page-bounded, compacted."""
    rows = _rows(3)
    assert rows
    for row in rows:
        assert row["answer_format"] == COMPACT_ANSWER_FORMAT
        answer = json.loads(row["messages"][-1]["content"])
        assert '"raw"' not in row["messages"][-1]["content"]
        schema = compact_schema(with_page_bounds(resolved_schema("policy", None, "personal_auto", row["sections"]),
                                                 3, pages=row["window_pages"]))
        errors = list(Draft202012Validator(schema).iter_errors(answer))
        assert not errors, (row["sections"], errors[0].message)


@pytest.mark.format_profile("standard")
def test_the_compact_targets_are_the_standard_targets_written_short(monkeypatch):
    standard = {(r["sections"], tuple(r["window_pages"])): json.loads(r["messages"][-1]["content"])
                for r in _rows(3)}
    _profile(monkeypatch, "fast")
    for row in _rows(3):
        key = (row["sections"], tuple(row["window_pages"]))
        if key in standard:
            assert expand_answer(json.loads(row["messages"][-1]["content"])) == standard[key]


def _tokens_for(text: str, logprob_of) -> tuple[list[str], list[float]]:
    tokens = [text[i:i + 3] for i in range(0, len(text), 3)]
    offsets, logprobs = 0, []
    for token in tokens:
        logprobs.append(logprob_of(offsets, offsets + len(token)))
        offsets += len(token)
    return tokens, logprobs


@pytest.mark.parametrize("written", ["r", "v"])
def test_confidence_reads_the_printed_value_s_own_tokens(written):
    """``r`` where it is written - the printed text, the reading - else ``v``."""
    if written == "r":
        text = '{"policy": {"effective_date": {"r": "April 1, 2026", "v": "04/01/2026", "p": [1]}}}'
        reading = "April 1, 2026"
    else:
        text = '{"policy": {"policy_number": {"v": "PA-2026-0042", "p": [1]}}}'
        reading = "PA-2026-0042"
    start = text.index(reading)
    end = start + len(reading)
    tokens, logprobs = _tokens_for(text, lambda a, b: -2.0 if a < end and b > start else -0.01)
    spans = map_compact_field_spans(text, tokens, logprobs)
    field = "policy.effective_date" if written == "r" else "policy.policy_number"
    assert f"{field}.parsed" in spans and f"{field}.page_ref[0]" in spans
    assert (f"{field}.raw" in spans) == (written == "r")
    collapsed = collapse_spans(spans)
    assert set(collapsed) == {field}
    assert collapsed[field].value == reading
    assert collapsed[field].mapped and min(collapsed[field].token_logprobs) == -2.0


def test_ids_and_codes_keep_their_paths_in_a_compact_answer():
    text = json.dumps({"coverages": [{"coverage_code": "X_UM", "premium": {"r": "$5", "v": 5, "p": [2]}}]})
    tokens, logprobs = _tokens_for(text, lambda a, b: -0.1)
    spans = map_compact_field_spans(text, tokens, logprobs)
    assert {"coverages[0].coverage_code", "coverages[0].premium.raw", "coverages[0].premium.parsed",
            "coverages[0].premium.page_ref[0]"} == set(spans)


# --------------------------------------------------------------------------
# Serving: compact answers expanded before the merge and every check
# --------------------------------------------------------------------------


class _Replay(ModelBackend):
    """Answers a fast-layout window with the target training built for it,
    found by its group block and page markers; checks every answer against
    the decoding schema it was sent with, as the grammar would hold it."""

    def __init__(self, answers):
        self.answers = answers
        self.asked: list[tuple[str, tuple[int, ...]]] = []

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config_, adapter=None) -> Generation:
        content = messages[1]["content"]
        pages = tuple(int(n) for n in re.findall(r"<page (\d+) of", json.dumps(content)))
        key = (content[-1]["text"], pages)
        self.asked.append(key)
        text = self.answers[key]
        if config_.json_schema is not None:
            errors = list(Draft202012Validator(config_.json_schema).iter_errors(json.loads(text)))
            assert not errors, errors[0].message
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens))


INVENTED = "PA-2026-9999"


def _serve_fast(*, invent: bool = False, mode: str = "ocr_plus_image"):
    """The synthetic 3-page auto fixture served under the fast profile by
    replaying its compact training targets, on page text printing its values."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from inference_core.model_runner import load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from tests.test_printed_values import _texts as printed_texts
    from tests.test_schema2_serving import _compact_auto
    from tests.test_serving_pipeline import CALIBRATION

    gold = _compact_auto()
    texts = printed_texts(gold, 3)
    images = [f"processed/default/policy/pa_1/page_{p}.png" for p in (1, 2, 3)]
    doc = SourceDocument(source_id="pa_1", doc_type="policy", golden_label=copy.deepcopy(gold),
                         ocr_pages=[texts[p] for p in (1, 2, 3)], image_paths=images, lob="personal_auto",
                         tenant_id="default")
    rows, _ = expand_document(doc, "train", modes=(mode,))
    answers = {}
    for row in rows:
        answer = row["messages"][-1]["content"]
        if invent:
            answer = answer.replace("PA-2026-0042", INVENTED)
        answers[(row["messages"][1]["content"][-1]["text"], tuple(row["window_pages"]))] = answer
    backend = _Replay(answers)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    request = ExtractionRequest(
        source_id="pa_1", image_paths=images, page_texts=texts, modality_mode=mode,
        known_doc_type="policy", known_lob="personal_auto", source_file_name="auto_fixture.pdf",
        ocr_meta={"modality": "native_pdf"},
    )
    result = extract(request, load_model("base", client, backend_impl=backend), StaticClassifier("policy"),
                     CALIBRATION, strict_schema=True)
    return result, backend, answers, gold


def test_replaying_the_compact_targets_serves_the_gold_back(fast):
    from common.schemas import iter_validation_errors
    from tests.test_schema2_serving import _comparable, _no_nulls

    result, backend, answers, gold = _serve_fast()
    assert set(backend.asked) == set(answers), "serving asked for windows training never built"
    assert result.schema_valid
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == []
    assert _comparable(_no_nulls(result.extraction)) == _comparable(gold)
    # Expanded before anything read it: the served JSON has full envelopes,
    # and the printed values were all found.
    number = result.extraction["policy"]["policy_number"]
    assert number["raw"] == "PA-2026-0042" and number["page_ref"] == [1]
    assert result.printed_check["not_printed"] == [] and result.printed_check["checked"] > 20
    # Confidence reached the values through the compact answers' spans.
    assert "policy.policy_number" in result.fields


def test_an_invented_value_in_a_compact_answer_is_still_taken_out(fast):
    result, _backend, _answers, _gold = _serve_fast(invent=True)
    listed = {entry["raw"]: entry for entry in result.printed_check["not_printed"]}
    assert set(listed) == {INVENTED}
    assert listed[INVENTED]["path"] == "policy.policy_number" and listed[INVENTED]["pages"] == [1]
    assert result.printed_check["removed"] is True
    assert result.extraction["policy"]["policy_number"]["raw"] is None
    assert "policy.policy_number:not_printed_removed" in result.review_flags
    # The invented rate the evaluation reads (not_printed / checked) counts it.
    assert len(result.printed_check["not_printed"]) / result.printed_check["checked"] > 0


# --------------------------------------------------------------------------
# Send order: reversed still sends in reverse and merges as planned
# --------------------------------------------------------------------------


class _Scripted(ModelBackend):
    """Answers each window with ``answer(group, pages)`` - compact JSON - and
    records the order windows were sent in."""

    def __init__(self, answer, lob="personal_auto"):
        self.answer, self.lob = answer, lob
        self.calls: list[tuple[str | None, list[int]]] = []

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config_, adapter=None) -> Generation:
        block = messages[1]["content"][-1]["text"]
        group = next((g for g in groups_for(self.lob)
                      if block == "\n\n" + render_group_block("policy", None, self.lob, g)), None)
        pages = [int(n) for n in re.findall(r"<page (\d+) of \d+>", json.dumps(messages[1]["content"]))]
        self.calls.append((group, pages))
        text = self.answer(group, pages)
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens))


def _request(texts, **over):
    from serving.pipeline import ExtractionRequest

    return ExtractionRequest(source_id="fp_1", known_doc_type="policy", known_lob="personal_auto",
                             page_texts=dict(enumerate(texts, start=1)), image_paths=_images(len(texts)), **over)


def _extract(backend, request):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from calibration.fit_calibration import CalibrationParams
    from inference_core.model_runner import load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract

    calibration = CalibrationParams(method="temperature", doc_type="policy", model_version="v1", temperature=1.0)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    return extract(request, load_model("base", client, backend_impl=backend), StaticClassifier("policy"),
                   calibration, strict_schema=False)


def _renewal_texts():
    """Page 1 a renewal notice printing the PRIOR policy number; the
    declarations, with the policy's own, on page 9 - outside the leading pages."""
    texts = [f"Page {p} wording." for p in range(1, 13)]
    texts[0] = "Renewal notice. Your prior policy WC-OLD expires soon."
    texts[8] = ("COMMON POLICY DECLARATIONS named insured policy number policy period\n"
                "Named Insured: Rivera Fabrication LLC\nPolicy Number WC-NEW")
    return texts


def _policy_number(group, pages):
    if group != "decl":
        return "{}"
    if 9 in pages:
        out = {"policy": {"policy_number": {"v": "WC-NEW", "p": [9]}}}
    elif 1 in pages:
        out = {"policy": {"policy_number": {"v": "WC-OLD", "p": [1]}}}
    else:
        out = {}
    return json.dumps(out)


@pytest.mark.parametrize("order", ["planned", "reversed"])
def test_reversed_sends_backwards_and_merges_in_plan_order(fast, order):
    texts = _renewal_texts()
    backend = _Scripted(_policy_number)
    result = _extract(backend, _request(texts, send_order=order))
    # The windows the declarations lead (pages 1-5 and 9) win, whatever the order.
    number = result.extraction["policy"]["policy_number"]
    assert number["raw"] == "WC-NEW" and number["page_ref"] == [9]

    from serving.pipeline import _send_order

    every, declarations = pages_to_read(texts, len(texts))
    plans = plan_windows("personal_auto", every, declarations, texts)
    planned = [(plan.group, list(plan.pages)) for plan in plans]
    expected = [planned[p] for p in _send_order([(g, p) for g, p in planned], "policy")]
    assert backend.calls == (expected if order == "planned" else expected[::-1])
    # A window's groups are sent together, so its pages are prefilled once.
    seen, current = [], None
    for _group, pages in expected:
        if pages != current:
            assert pages not in seen, "a window's groups were not sent together"
            seen.append(pages)
            current = pages


# --------------------------------------------------------------------------
# Identity: a profile is never served, trained or packaged in the other
# --------------------------------------------------------------------------


@pytest.mark.format_profile("standard")
def test_the_profile_is_part_of_the_prompt_hash_and_the_window_plan(monkeypatch):
    from serving.release_router import ServedRelease, ServingPlan
    from serving.vllm_entrypoint import ColdStartError, assert_prompt_hash, assert_window_plan_served

    standard_hash, standard_plan = prompt_hash(), window_plan_id()
    _profile(monkeypatch, "fast")
    fast_hash, fast_plan = prompt_hash(), window_plan_id()
    assert fast_hash != standard_hash and fast_plan != standard_plan
    assert fast_plan.startswith(f"4+fast.aligned-1.{COMPACT_ANSWER_FORMAT}:")
    assert [p for p in prompt_input_files() if FAST_TEMPLATE_DIR in p.parents]

    # Trained in standard, refused in fast - serving, promote and training alike.
    with pytest.raises(ValueError, match="Rebuild the corpus"):
        assert_window_plan({"window_plan_version": standard_plan})
    trained = ServedRelease("v9", "base@abc1234", "merged", doc_types=("policy",),
                            prompt_hash=standard_hash, window_plan_version=standard_plan)
    with pytest.raises(ColdStartError, match="prompt hash"):
        assert_prompt_hash(ServingPlan(trained))
    with pytest.raises(ColdStartError, match="window plan"):
        assert_window_plan_served(ServingPlan(trained))

    # Trained in fast, refused in standard.
    _profile(monkeypatch, "standard")
    with pytest.raises(ValueError, match="Rebuild the corpus"):
        assert_window_plan({"window_plan_version": fast_plan})
    with pytest.raises(ColdStartError, match="prompt hash"):
        assert_prompt_hash(ServingPlan(ServedRelease("v10", "base@abc1234", "merged", doc_types=("policy",),
                                                     prompt_hash=fast_hash, window_plan_version=fast_plan)))


# --------------------------------------------------------------------------
# 5. The modality mix and checkpoint selection
# --------------------------------------------------------------------------


@pytest.mark.format_profile("standard")
def test_the_fast_profile_trains_and_selects_on_more_image_only(monkeypatch):
    from common.adapter import ADAPTER
    from training.data_mix import configured_modality_mix

    standard = config.training_config(ADAPTER.training_config)
    assert "format_profiles" not in standard
    assert standard["evaluation"]["selection_document_modes"] == ["ocr_plus_image"]
    standard_mix = configured_modality_mix()

    _profile(monkeypatch, "fast")
    fast_config = config.training_config(ADAPTER.training_config)
    assert configured_modality_mix() == {"ocr_plus_image": 0.25, "noisy_ocr_image": 0.05, "image_only": 0.7}
    assert configured_modality_mix() != standard_mix
    assert fast_config["evaluation"]["selection_document_modes"] == ["ocr_plus_image", "image_only"]
    # Everything else is the standard config.
    assert fast_config["lora"] == standard["lora"] and fast_config["optimization"] == standard["optimization"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _rows(pages: int, *, texts: list[str] | None = None, mode: str = "ocr_plus_image") -> list[dict]:
    """The training rows of a synthetic personal-auto policy of ``pages`` pages."""
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from tests.test_schema2_serving import _compact_auto

    texts = texts or _texts(pages)
    document = SourceDocument(source_id="fp_1", doc_type="policy", golden_label=_compact_auto(),
                              ocr_pages=texts, image_paths=_images(pages), lob="personal_auto",
                              tenant_id="default")
    rows, _ = expand_document(document, "train", modes=(mode,))
    return rows
