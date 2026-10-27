"""The one policy adapter (common.adapter, decision 2026-10-22): one output shape
per corpus, and the ``--lines`` filter the data commands share (common.lob)."""

from __future__ import annotations

import dataclasses

import pytest

from common.adapter import ADAPTER, AdapterError, assert_one_output_shape
from common.lob import LobError, known_lines, parse_lines, within_lines
from common.schemas import SNAPSHOT_LINES, is_common_model

# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


def test_the_adapter_is_fixed_not_configured():
    assert ADAPTER.name == "policy"
    assert ADAPTER.doc_types == ("policy",)
    assert ADAPTER.training_config == "unified"
    assert ADAPTER.run_id("v3") == "policy-v3"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ADAPTER.name = "unified"  # type: ignore[misc]


# --------------------------------------------------------------------------
# One output shape
# --------------------------------------------------------------------------


def test_the_personal_lines_alone_train_one_shape():
    """Every personal line is on the SPEC_21 common model."""
    assert all(is_common_model("policy", None, line) for line in SNAPSHOT_LINES)
    assert_one_output_shape(SNAPSHOT_LINES)


def test_self_contained_lines_alone_train_one_shape():
    lines = ["equipment_breakdown", "epli", "business_owners"]
    assert not any(is_common_model("policy", None, line) for line in lines)
    assert_one_output_shape(lines)


def test_a_corpus_with_nothing_in_it_has_no_shape_to_mix():
    assert_one_output_shape([])


def test_a_common_model_line_and_a_self_contained_line_are_refused():
    """Homeowners answers in the common model's shape, equipment breakdown in
    its own file's: one adapter trained on both learns neither well."""
    assert not is_common_model("policy", None, "equipment_breakdown")
    with pytest.raises(AdapterError, match="two output shapes") as caught:
        assert_one_output_shape(["homeowners", "equipment_breakdown"])
    message = str(caught.value)
    assert "read against the common model (homeowners)" in message
    assert "self-contained schema files (equipment_breakdown)" in message


def test_a_package_or_a_policy_with_no_line_is_judged_by_the_fallback_it_is_read_against():
    """A package of lines, and a policy with no line, are read against
    _fallback.json, which is on the common model: with equipment breakdown
    alone they are two shapes, with homeowners one."""
    assert_one_output_shape(["homeowners", ["epli", "equipment_breakdown"], None])
    with pytest.raises(AdapterError, match="two output shapes") as caught:
        assert_one_output_shape(["equipment_breakdown", ["epli", "equipment_breakdown"]])
    assert "epli+equipment_breakdown" in str(caught.value)
    with pytest.raises(AdapterError, match="no line"):
        assert_one_output_shape(["equipment_breakdown", None])


def test_general_liability_with_a_self_contained_line_is_refused_too():
    """General liability composes the common model, and auto does since
    auto.json 2.0.0; a commercial package does not yet."""
    assert_one_output_shape({"gl": 40, "auto": 3})
    with pytest.raises(AdapterError, match="two output shapes"):
        assert_one_output_shape({"gl": 40, "commercial_package": 3})   # a line-count mapping reads as its lines


# --------------------------------------------------------------------------
# --lines
# --------------------------------------------------------------------------


def test_no_lines_means_every_line():
    assert parse_lines(None) is None
    assert parse_lines("") is None
    assert parse_lines("  ") is None


def test_lines_are_read_as_canonical_names():
    assert parse_lines("homeowners, personal_auto") == frozenset({"homeowners", "personal_auto"})
    # The enum spelling and a merged line read as the schema's line.
    assert parse_lines("workers_comp,classic_auto") == frozenset({"wc", "personal_auto"})
    assert parse_lines("general_liability,") == frozenset({"gl"})


def test_a_line_with_no_schema_is_refused():
    """A misspelt line would filter out every document without a word."""
    with pytest.raises(LobError, match="no canonical policy schema") as caught:
        parse_lines("homeowners,homeowner")
    assert "homeowner'" in str(caught.value) and "pet_insurance" not in str(caught.value)
    with pytest.raises(LobError):
        parse_lines("pet_insurance")
    assert "homeowner" not in known_lines()


def test_within_lines_passes_everything_without_a_filter():
    assert within_lines("wc", None)
    assert within_lines(None, None)
    assert within_lines(["homeowners", "wc"], frozenset())


def test_within_lines_needs_every_line_of_the_document():
    lines = frozenset({"homeowners", "personal_auto", "wc"})
    assert within_lines("homeowners", lines)
    assert within_lines("workers_comp", lines)                  # read as wc
    assert within_lines("classic_auto", lines)                  # read as personal_auto
    assert not within_lines("gl", lines)
    assert not within_lines(["wc", "gl"], lines)                # one line outside: outside


def test_a_document_with_no_line_is_outside_any_filter():
    assert not within_lines(None, frozenset({"homeowners"}))
    assert not within_lines([], frozenset({"homeowners"}))


def test_a_package_of_common_model_lines_is_outside_any_filter():
    """A package is read against the fallback schema, so the adapter never
    learned its shape for those lines: the base model reads it."""
    lines = frozenset({"homeowners", "personal_auto"})
    assert not within_lines(["homeowners", "personal_auto"], lines)
    # Self-contained lines carry no such shape: a package of them passes.
    assert within_lines(["equipment_breakdown", "epli"], frozenset({"equipment_breakdown", "epli"}))
