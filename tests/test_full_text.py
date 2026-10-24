"""The full-text tier, ``text_sections``: built from the page text by Fideon's
section builder, never by the model (SPEC_21 §4.6, common.full_text).

With the fields and ``additional_fields`` it puts every printed word somewhere
in the served JSON - SPEC_21's every-word rule - and each served document says
how much of its page text reached the JSON.
"""

from __future__ import annotations

import json
import shutil

import pytest

from common.full_text import build_text_sections, text_coverage
from common.schemas import iter_validation_errors
from serving.pipeline import NO_FULL_TEXT_FLAG
from tests.test_schema2_serving import _compact_auto, _serve_replay

PAGES = [
    "# Homeowners Policy Declarations\nPolicy Number HO-123456\nNamed Insured: Rivera Fabrication LLC\n"
    "| Coverage | Limit |\n|---|---|\n| Dwelling | $500,000 |\n| Personal Property | $250,000 |",
    "SECTION I - CONDITIONS\nA. Insurable Interest\nWe will not pay more than the insurable interest.\n"
    "B. Duties After Loss\nGive prompt notice to us or our agent.",
]


# --------------------------------------------------------------------------
# The copied builder
# --------------------------------------------------------------------------


def test_the_copied_builder_matches_its_recorded_source():
    """common/section_builder holds FideonSLM's files byte for byte: an edit made
    here would quietly make L3's full text differ from L1's."""
    from scripts.check_section_builder import local_differences, source

    assert local_differences() == []
    assert set(source()["files"]) == {"sections.py", "pdf_text.py"}


def test_a_moved_on_fideonslm_copy_is_reported(tmp_path):
    from scripts.check_section_builder import BUILDER, main, source, upstream_differences

    clone = tmp_path / "FideonSLM"
    folder = clone / source()["path"]
    folder.mkdir(parents=True)
    for name in source()["files"]:
        shutil.copy(BUILDER / name, folder / name)
    assert upstream_differences(clone) == [] and main(["--fideon-slm", str(clone)]) == 0
    # A Windows checkout of the same commit is the same file (ours may be one already).
    lf = (BUILDER / "pdf_text.py").read_bytes().replace(b"\r\n", b"\n")
    (folder / "pdf_text.py").write_bytes(lf.replace(b"\n", b"\r\n"))
    assert upstream_differences(clone) == []
    (folder / "sections.py").write_text("# changed upstream\n", encoding="utf-8")
    assert [d.split(":")[0] for d in upstream_differences(clone)] == ["sections.py"]
    assert main(["--fideon-slm", str(clone)]) == 1


# --------------------------------------------------------------------------
# Building the tier
# --------------------------------------------------------------------------


def test_every_word_of_the_page_text_lands_in_a_section():
    sections = build_text_sections(PAGES)
    assert sections
    for section in sections.values():
        assert {"section_id", "raw_text", "page_range"} <= set(section)
        assert 1 <= section["page_range"][0] <= section["page_range"][1] <= len(PAGES)
    assert text_coverage(PAGES, {"text_sections": sections}) == 1.0
    # The wording is organised by its headings, and a table is kept as a table.
    assert any("CONDITIONS" in key for key in sections)
    assert any(s["section_type"] == "table" and "$500,000" in s["raw_text"] for s in sections.values())


def test_a_markdown_table_is_written_and_its_anchor_never_is():
    """FideonSLM's builder writes a table only when a page line falls inside it,
    and MinerU's tables have none: every table was dropped. The anchor that
    fixes it is never written - not even when the same table opens every page,
    where a repeated line would read as a page header."""
    table = "| Coverage | Limit |\n|---|---|\n| Dwelling | $500,000 |"
    pages = [f"{table}\nPage {n} wording follows here." for n in (1, 2, 3)]
    sections = build_text_sections(pages)
    tables = [s for s in sections.values() if s["section_type"] == "table"]
    assert sorted(s["page_range"][0] for s in tables) == [1, 2, 3]
    assert text_coverage(pages, {"text_sections": sections}) == 1.0
    assert "anchor" not in json.dumps(sections)


def test_no_page_text_is_no_full_text_and_no_measure():
    assert build_text_sections([None, "", "   "]) == {}
    assert text_coverage([None, ""], {"policy": {"policy_number": "HO-1"}}) is None


def test_coverage_counts_words_with_their_repeats_and_never_keys():
    pages = ["Dwelling Dwelling Personal Property"]
    assert text_coverage(pages, {"Dwelling": "Personal Property"}) == pytest.approx(2 / 4)
    assert text_coverage(pages, {"a": ["Dwelling", {"b": "Dwelling Personal Property"}]}) == 1.0


# --------------------------------------------------------------------------
# Served
# --------------------------------------------------------------------------


def test_a_served_policy_carries_its_full_text_and_it_is_valid():
    result, _backend, _answers = _serve_replay(_compact_auto(), 3, strict=True)
    sections = result.extraction["text_sections"]
    assert sections and all(s["raw_text"] for s in sections.values())
    assert result.schema_valid
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == []
    assert result.text_coverage == 1.0
    assert NO_FULL_TEXT_FLAG not in result.review_flags
    assert result.as_dict()["text_coverage"] == 1.0


def test_the_full_text_tier_can_join_the_fields_after_them(monkeypatch):
    """Fields first: the fields are served before MinerU has read the PDF, and its
    text adds the tier afterwards - as extract would have built it."""
    import serving.pipeline as pipeline
    from serving.pipeline import FULL_TEXT_INVALID_FLAG, attach_full_text

    served, _backend, _answers = _serve_replay(_compact_auto(), 3, strict=True)
    texts = {1: "Declarations", 2: "Schedule page 2", 3: "Schedule page 3"}      # what _serve_replay read

    def fields_only():
        result, _b, _a = _serve_replay(_compact_auto(), 3, strict=True)
        result.extraction = {**result.extraction, "text_sections": None}
        result.review_flags = sorted({*result.review_flags, NO_FULL_TEXT_FLAG})
        result.text_coverage = None
        return result

    joined = attach_full_text(fields_only(), texts, lob="personal_auto")
    assert joined.extraction["text_sections"] == served.extraction["text_sections"]
    assert NO_FULL_TEXT_FLAG not in joined.review_flags and joined.text_coverage == served.text_coverage == 1.0
    assert list(iter_validation_errors(joined.extraction, "policy", None, "personal_auto")) == []

    # A tier that fails the client's schema is left out: the fields stand, flagged.
    later = fields_only()
    monkeypatch.setattr(pipeline, "build_text_sections", lambda pages: {"bad": {"raw_text": 7}})
    refused = attach_full_text(later, texts, lob="personal_auto")
    assert refused.extraction["text_sections"] is None and FULL_TEXT_INVALID_FLAG in refused.review_flags
    # No text: nothing changes.
    assert attach_full_text(refused, {}, lob="personal_auto").extraction["text_sections"] is None


def test_fields_that_already_break_the_schema_still_get_their_full_text():
    """The tier is judged by the errors it brings, not by the fields' own."""
    from serving.pipeline import FULL_TEXT_INVALID_FLAG, attach_full_text

    result, _b, _a = _serve_replay(_compact_auto(), 3, strict=False)
    fields = {k: v for k, v in result.extraction.items() if k != "text_sections"}
    fields["policy"] = {**fields["policy"], "policy_number": 12345}          # not a field object
    result.extraction, result.text_coverage, result.printed_check = fields, None, {}
    result.review_flags = sorted({*result.review_flags, NO_FULL_TEXT_FLAG, "schema:invalid"})
    errors_before = list(iter_validation_errors(fields, "policy", None, "personal_auto"))
    attach_full_text(result, {1: "Declarations", 2: "Schedule page 2", 3: "Schedule page 3"}, lob="personal_auto")
    assert result.extraction["text_sections"] and FULL_TEXT_INVALID_FLAG not in result.review_flags
    assert list(iter_validation_errors(result.extraction, "policy", None, "personal_auto")) == errors_before


def test_a_request_with_no_page_text_is_flagged_and_not_measured(monkeypatch):
    """An image-only request carries no page text: there is nothing to section,
    and its wording is in no part of the JSON - said, not served as complete."""
    import serving.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_full_text_pages", lambda request: [])
    result, _backend, _answers = _serve_replay(_compact_auto(), 3)
    assert result.extraction.get("text_sections") is None
    assert NO_FULL_TEXT_FLAG in result.review_flags
    assert result.text_coverage is None


def test_the_full_text_is_built_from_the_page_text_as_the_request_carried_it():
    from serving.pipeline import ExtractionRequest, _full_text_pages

    paged = ExtractionRequest(source_id="p", image_paths=["a", "b"], page_texts={2: "two", 1: None})
    assert _full_text_pages(paged) == ["", "two"]                # no placeholder for a blank page
    single = ExtractionRequest(source_id="p", image_paths=["a"], ocr_text="only page")
    assert _full_text_pages(single) == ["only page"]
    assert _full_text_pages(ExtractionRequest(source_id="p", image_paths=["a", "b"])) == []


def test_the_full_text_is_never_scored():
    """Scoring reads what the model read: the tier serving adds after the merge
    changes no accuracy number."""
    from testing.run_extraction import score_against_ground_truth

    gold = _compact_auto()
    result, _backend, _answers = _serve_replay(gold, 3)
    assert result.extraction["text_sections"]
    with_text = score_against_ground_truth(result, gold, lob="personal_auto")
    result.extraction = {k: v for k, v in result.extraction.items() if k != "text_sections"}
    assert json.dumps(with_text, sort_keys=True, default=str) == json.dumps(
        score_against_ground_truth(result, gold, lob="personal_auto"), sort_keys=True, default=str)


def test_the_run_summary_reports_the_every_word_rule():
    from testing.run_extraction import FULL_TEXT_COVERAGE, summarise

    full, _b, _a = _serve_replay(_compact_auto(), 3)
    short, _b, _a = _serve_replay(_compact_auto(), 3)
    short.source_id, short.text_coverage = "pa_2", FULL_TEXT_COVERAGE - 0.05
    unread, _b, _a = _serve_replay(_compact_auto(), 3)
    unread.source_id, unread.text_coverage = "pa_3", None
    unread.review_flags = [*unread.review_flags, NO_FULL_TEXT_FLAG]

    metrics = summarise([(full, {}), (short, {}), (unread, {})], "v1").as_dict()
    assert metrics["min_text_coverage"] == FULL_TEXT_COVERAGE - 0.05
    assert metrics["mean_text_coverage"] == round((1.0 + FULL_TEXT_COVERAGE - 0.05) / 2, 4)
    assert metrics["documents_below_full_text"] == ["pa_2"]
    assert metrics["documents_without_full_text"] == ["pa_3"]
