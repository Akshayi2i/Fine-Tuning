"""Classic auto is personal auto: one line of business (common.lob.MERGED_LINES).

Wherever a line is read - a label's line_of_business, bundle and label metadata,
schema selection - classic_auto is read as personal_auto, so it never selects a
schema, a split, a routed line or a report row of its own.
"""

from __future__ import annotations

from common.lob import lob_values, merge_line, normalize_lob, validate_lob
from common.schemas import schema_key


def test_classic_auto_is_read_as_personal_auto():
    assert merge_line("classic_auto") == "personal_auto"
    assert merge_line(["classic_auto", "personal_auto"]) == ["personal_auto"]
    assert merge_line(None) is None and merge_line("homeowners") == "homeowners"


def test_it_selects_the_personal_auto_schema():
    assert schema_key("policy", None, "classic_auto") == schema_key("policy", None, "personal_auto")


def test_it_is_not_a_line_the_model_can_name():
    assert "classic_auto" not in lob_values()
    # A stored label that still says classic_auto reads, as personal auto.
    assert validate_lob(["classic_auto"]) == ["personal_auto"]
    assert normalize_lob("classic_auto") == ["personal_auto"]


def test_a_source_document_carries_the_merged_line():
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument

    doc = SourceDocument(source_id="h1", doc_type="policy", golden_label={}, ocr_pages=[],
                         image_paths=[], lob="classic_auto")
    assert doc.lob == "personal_auto"


def test_there_is_one_auto_line_to_route_and_snapshot():
    from common.lob import known_lines, lob_lines
    from common.schemas import SNAPSHOT_LINES

    assert "personal_auto" in known_lines() and "classic_auto" not in known_lines()
    assert lob_lines("classic_auto") == frozenset({"personal_auto"})
    assert "personal_auto" in SNAPSHOT_LINES and "classic_auto" not in SNAPSHOT_LINES
