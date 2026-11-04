"""data_pipeline.near_copies and scripts/label_conventions.py: what a delivery holds, read-only."""

from __future__ import annotations

import csv
import json

import pymupdf
import pytest

from data_pipeline.near_copies import (
    SAME_INSURED,
    SAME_POLICY,
    TEMPLATE,
    TWIN,
    Document,
    groups_of,
    load_documents,
    main,
    near_copy_pairs,
    page_hashes,
    pair_kind,
)
from scripts.label_conventions import form_edition_place, spacing_variants, tally


def _doc(name, pages, identity=(), **over):
    return Document(name=name, pages=set(pages), page_count=len(pages), identity=set(identity), **over)


def test_a_pair_is_named_by_what_its_labels_share():
    assert pair_kind(_doc("a", "x", ["policy:1", "name:n"]), _doc("b", "x", ["policy:1"])) == SAME_POLICY
    assert pair_kind(_doc("a", "x", ["name:n"]), _doc("b", "x", ["name:n", "policy:2"])) == SAME_INSURED
    assert pair_kind(_doc("a", "x", ["policy:1"]), _doc("b", "x", ["policy:2"])) == TEMPLATE
    assert pair_kind(_doc("a", "x", synthetic=True, source="s"), _doc("b", "x", source="s")) == TWIN


def test_two_real_documents_of_one_source_key_are_not_twins():
    assert pair_kind(_doc("a", "x", source="s"), _doc("b", "x", source="s")) == TEMPLATE


def test_pairs_are_those_sharing_enough_of_the_smaller_document():
    documents = [_doc("big", "abcdefghij", split="train"), _doc("small", "abcd", split="test"), _doc("other", "xyz")]
    [pair] = near_copy_pairs(documents, 0.5)
    assert (pair["a"], pair["b"], pair["shared_pages"], pair["share"]) == ("big", "small", 4, 1.0)
    assert pair["crosses_splits"] is True
    assert near_copy_pairs([_doc("a", "abcd"), _doc("b", "defg")], 0.5) == []


def test_groups_join_documents_transitively():
    pairs = [{"a": "a", "b": "b"}, {"a": "b", "b": "c"}, {"a": "x", "b": "y"}]
    assert groups_of(pairs) == [["a", "b", "c"], ["x", "y"]]


def _pdf(path, texts):
    document = pymupdf.open()
    for text in texts:
        document.new_page().insert_text((72, 72), text)
    document.save(path)


@pytest.fixture
def delivery(tmp_path):
    def folder(name, texts, gold):
        (tmp_path / name).mkdir()
        _pdf(tmp_path / name / "document.pdf", texts)
        (tmp_path / name / "golden.json").write_text(json.dumps(gold), encoding="utf-8")
        (tmp_path / name / "metadata.json").write_text("{}", encoding="utf-8")

    number = {"policy": {"policy_number": {"raw": "POL-123456", "parsed": "POL-123456"}}}
    folder("one", ["Page A", "Page B", "Page C"], number)
    folder("two", ["Page A", "Page B", "Page D"], number)
    folder("three", ["Page X"], {})
    with (tmp_path / "split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["document", "split"])
        writer.writeheader()
        writer.writerows([{"document": "one", "split": "train"}, {"document": "two", "split": "test"},
                          {"document": "three", "split": "train"}])
    return tmp_path


def test_identical_pages_hash_alike_and_different_pages_do_not(delivery):
    one, two = page_hashes(delivery / "one" / "document.pdf"), page_hashes(delivery / "two" / "document.pdf")
    assert one[:2] == two[:2] and one[2] != two[2]


def test_a_delivery_is_read_with_its_splits_and_labels(delivery):
    documents = load_documents(delivery, log=lambda _text: None)
    [pair] = near_copy_pairs(documents, 0.5)
    assert (pair["a"], pair["b"], pair["kind"], pair["crosses_splits"]) == ("one", "two", SAME_POLICY, True)


def test_the_cache_means_a_second_run_renders_nothing(delivery, tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "hashes.json"
    load_documents(delivery, cache, log=lambda _text: None)
    monkeypatch.setattr("data_pipeline.near_copies.page_hashes", lambda _pdf: pytest.fail("rendered again"))
    assert len(load_documents(delivery, cache, log=lambda _text: None)) == 3


def test_the_command_line_prints_and_writes_the_pairs(delivery, tmp_path, capsys):
    out = tmp_path / "pairs.json"
    assert main(["--input", str(delivery), "--cache", str(tmp_path / "c.json"), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "1 pair(s) share at least 50%" in text and "One policy number on more than one document" in text
    assert json.loads(out.read_text(encoding="utf-8"))[0]["kind"] == SAME_POLICY


def test_where_a_form_holds_its_edition():
    assert form_edition_place({"form_number": "CG 00 01", "edition_date": "04/13"}) == "field"
    for number in ("CG 00 01 04 13", "SSI SF 00 03 0123", "SSPJ-07-2021", "IL0017(11-98)"):
        assert form_edition_place({"form_number": number}) == "number", number
    assert form_edition_place({"form_number": "SSI-SOP-001"}) == "none"
    assert form_edition_place({"form_number": ""}) is None


def test_names_that_differ_only_by_spacing_are_found():
    assert spacing_variants(["Each Occurrence Limit", "each occurrence lim it", "Fire Legal"]) == [
        ["each occurrence lim it", "each occurrence limit"]]


def test_the_tally_counts_forms_and_coverages():
    golds = {
        "a": {"forms_and_endorsements": [{"form_number": "CG 00 01", "edition_date": "04/13"},
                                         {"form_number": "SSI SF 00 03 0123"}],
              "coverages": [{"coverage_name": "General Aggregate Limit", "coverage_code": "GL_BI_PD"},
                            {"coverage_name": "Each Occurrence", "coverage_code": "GL_BI_PD"}],
              "additional_fields": [{}, {}]},
        "b": {"forms_and_endorsements": [{"form_number": "SSI-SOP-001"}], "coverages": []},
    }
    counts = tally(golds)
    assert counts["form_edition_place"] == {"field": 1, "number": 1, "none": 1}
    assert counts["documents_mixing_form_conventions"] == ["a"]
    assert counts["coverage_rows"] == 2 and counts["coverage_rows_named_as_limits"] == 1
    assert counts["documents_with_no_coverage_rows"] == ["b"]
    assert counts["codes_with_most_names"][0]["names"] == 2
    assert counts["additional_fields_per_document"] == {"min": 0, "median": 1.0, "max": 2}
