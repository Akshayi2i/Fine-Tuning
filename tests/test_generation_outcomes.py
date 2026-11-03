"""Rows with no usable answer are counted beside the accuracy they lower."""

from __future__ import annotations

from evaluation.validation_generation import ValidationGeneration, _generation_outcomes


def _g(kind=None, error=None):
    entry = ValidationGeneration(row={}, golden={})
    entry.error, entry.failure_kind = error, kind
    return entry


def test_looped_and_failed_rows_are_counted_apart():
    rows = [_g(), _g(), _g("output", "stopped at the token limit"), _g("setup", "no assistant turn")]
    assert _generation_outcomes(rows) == {
        "unusable_output_rate": 0.25, "generation_failed_rate": 0.25,
        "unusable_output_rows": 1, "generation_failed_rows": 1,
    }


def test_clean_rows_report_zero_and_no_rows_report_nothing():
    assert _generation_outcomes([_g(), _g()])["unusable_output_rate"] == 0.0
    assert _generation_outcomes([]) == {}
