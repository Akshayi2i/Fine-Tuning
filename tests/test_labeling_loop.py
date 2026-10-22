"""IMPL-04 §1 and §6 — pre-annotation and the confidence-routed review queue.

Two guards carry this file. **No frontier model drafts a label**: frontier
models are never used for gold, synthetic or training data (decision
2026-10-19), so the only backends are self-hosted and a frontier one is refused
by name. The **day-zero routing refusal** is the same shape in a different
place: routing by confidence before a model has earned trust sends its own
mistakes past a reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.list_completeness import ROW_MISMATCH_SUFFIX
from data_pipeline.labeling import active_learning, pre_annotate
from data_pipeline.labeling.active_learning import (
    SPOT_CHECK_THRESHOLD,
    ActiveLearningError,
    assert_routing_permitted,
    build_queue,
)
from data_pipeline.labeling.pre_annotate import (
    DEFAULT_BACKEND,
    Draft,
    PreAnnotationError,
    resolve_backend_model,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def seed_labels(client: BlobClient, doc_type: str, count: int) -> None:
    for i in range(1, count + 1):
        client.write_json(paths.golden_label(doc_type, f"{doc_type}_{i:04d}"),
                          {"line_of_business": ["workers_comp"]})


# --------------------------------------------------------------------------
# No frontier model drafts a label
# --------------------------------------------------------------------------


def test_there_is_no_frontier_backend():
    assert pre_annotate.BACKENDS == ("base_qwen3vl", "own_finetuned")


def test_a_frontier_backend_is_refused_by_name(client):
    """Not a missing permission a flag could grant: the decision is recorded in
    the refusal."""
    with pytest.raises(PreAnnotationError, match="never used for gold"):
        pre_annotate.pre_annotate("policy_0001", "policy", client, backend="external_frontier")
    with pytest.raises(PreAnnotationError, match="never used for gold"):
        resolve_backend_model("external_frontier", client)  # type: ignore[arg-type]


def test_a_refused_backend_stops_the_whole_batch(client):
    with pytest.raises(PreAnnotationError, match="never used for gold"):
        pre_annotate.pre_annotate_batch(
            ["policy_0001", "policy_0002"], "policy", client, backend="external_frontier",
        )


def test_the_self_hosted_backend_needs_no_permission():
    assert DEFAULT_BACKEND == "base_qwen3vl"
    assert resolve_backend_model("base_qwen3vl", client=None) == "base"  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Backend resolution
# --------------------------------------------------------------------------


def test_own_finetuned_refuses_before_a_foundation_is_promoted(client):
    """The day-zero case, which is why the self-hosted base is the default."""
    with pytest.raises(PreAnnotationError, match="day-zero"):
        resolve_backend_model("own_finetuned", client)


def test_an_unknown_backend_is_refused(client):
    with pytest.raises(PreAnnotationError, match="unknown backend"):
        pre_annotate.pre_annotate("policy_0001", "policy", client, backend="gpt")  # type: ignore[arg-type]


def test_an_undocumented_document_cannot_be_drafted(client):
    with pytest.raises(PreAnnotationError, match="not been OCR"):
        pre_annotate.pre_annotate("policy_0001", "policy", client)


# --------------------------------------------------------------------------
# The draft itself
# --------------------------------------------------------------------------


def test_a_draft_is_never_trusted():
    draft = Draft(source_id="p1", doc_type="policy", backend="base_qwen3vl", model_version="base")
    assert draft.trusted is False
    assert draft.as_dict()["trusted"] is False


def test_a_draft_is_not_written_where_a_golden_label_lives(client):
    """One careless glob away from entering a corpus unreviewed."""
    draft = Draft(source_id="p1", doc_type="policy", backend="base_qwen3vl", model_version="base")
    key = pre_annotate.write_draft(draft, client)

    assert key != paths.golden_label("policy", "p1")
    assert key.endswith("draft.json")
    assert not client.exists(paths.golden_label("policy", "p1"))


# --------------------------------------------------------------------------
# Confidence routing — the day-zero refusal
# --------------------------------------------------------------------------


def test_routing_is_refused_below_the_day_zero_threshold(client):
    seed_labels(client, "policy", 3)
    with pytest.raises(ActiveLearningError, match="day-zero"):
        assert_routing_permitted(client, ["policy"])


def test_routing_is_refused_when_any_single_type_is_thin(client):
    """Routing ACORD by confidence while ACORD has four labels is the mistake,
    even if Policy has three hundred."""
    seed_labels(client, "policy", 40)
    seed_labels(client, "acord", 4)
    with pytest.raises(ActiveLearningError, match="acord"):
        assert_routing_permitted(client, ["policy", "acord"])


def test_routing_is_permitted_once_every_type_clears_the_threshold(client):
    for doc_type in ("policy", "acord", "lossrun"):
        seed_labels(client, doc_type, 25)
    assert_routing_permitted(client, ["policy", "acord", "lossrun"])  # does not raise


# --------------------------------------------------------------------------
# Queue ordering
# --------------------------------------------------------------------------


@dataclass
class _Result:
    """The parts of an ExtractionResult the queue reads."""

    source_id: str
    doc_type: str = "policy"
    overall_confidence: float = 0.5
    fields: dict[str, Any] = field(default_factory=dict)
    review_flags: list[str] = field(default_factory=list)


def _fields(**scores: float) -> dict[str, Any]:
    return {name: {"value": "x", "confidence": score} for name, score in scores.items()}


def test_the_queue_is_ordered_by_ascending_confidence():
    """The reviewer's first hour goes where it changes the most."""
    queue = build_queue([
        _Result("a", overall_confidence=0.95, fields=_fields(insured_name=0.95)),
        _Result("b", overall_confidence=0.30, fields=_fields(insured_name=0.30)),
        _Result("c", overall_confidence=0.60, fields=_fields(insured_name=0.60)),
    ])
    assert [item.source_id for item in queue.items] == ["b", "c", "a"]


def test_row_completeness_outranks_every_confidence_score():
    """A missing row has no confidence to be low — its absence is the problem,
    and no per-value number can express it."""
    queue = build_queue([
        _Result("low", overall_confidence=0.20, fields=_fields(insured_name=0.20)),
        _Result("incomplete", overall_confidence=0.99, fields=_fields(insured_name=0.99),
                # The flag string the pipeline ACTUALLY emits, taken from the
                # module that emits it. This test used to hand-write
                # "list:claims_row_count_mismatch" — a string nothing in the
                # codebase produces — so it passed while the override it guards
                # was dead code for every real document.
                review_flags=[f"claims{ROW_MISMATCH_SUFFIX}"]),
    ])
    assert queue.items[0].source_id == "incomplete"
    assert queue.items[0].routing == "full_review"


def test_the_queue_reads_the_flags_list_completeness_actually_emits():
    """A seam test. Both sides of this contract were written independently and
    disagreed in silence: the emitter used '{field}:row_count_mismatch', the
    consumer matched the prefixes 'list:'/'rows:'/'completeness:'. Asserting the
    real emitted string is what keeps them from drifting apart again."""
    from calibration.list_completeness import check_completeness, merge_review_flags

    signal = check_completeness("claims", 6, stated_count=8)
    assert signal.flagged, "a 6-of-8 list should be flagged"
    emitted = merge_review_flags({"claims": signal}, [])
    assert emitted, "list_completeness emitted no flag for a flagged list"

    queue = build_queue([
        _Result("incomplete", overall_confidence=0.99,
                fields=_fields(insured_name=0.99), review_flags=emitted),
    ])
    assert queue.items[0].completeness_flags == sorted(emitted)
    assert queue.items[0].routing == "full_review"


def test_a_confident_document_gets_only_a_spot_check():
    queue = build_queue([
        _Result("a", overall_confidence=0.97, fields=_fields(insured_name=0.98, policy_number=0.96)),
    ])
    assert queue.items[0].routing == "spot_check"
    assert queue.spot_check_share == 1.0


def test_low_confidence_is_reported_per_field_not_per_document():
    """Flagging a whole 40-row Loss Run because one date was uncertain wastes
    the reviewer on 39 correct rows."""
    queue = build_queue([
        _Result("a", overall_confidence=0.80,
                fields=_fields(insured_name=0.99, effective_date=0.31, policy_number=0.97)),
    ])
    item = queue.items[0]
    assert item.routing == "field_review"
    assert [name for name, _score in item.low_confidence_fields] == ["effective_date"]


def test_a_field_with_no_confidence_is_not_a_confident_field():
    """It means the span could not be mapped — a reason to look, not to skip."""
    queue = build_queue([
        _Result("a", overall_confidence=0.9, fields={"insured_name": {"value": "x"}}),
    ])
    assert queue.items[0].low_confidence_fields == [("insured_name", 0.0)]


def test_the_spot_check_share_is_the_number_that_shows_cost_falling():
    queue = build_queue([
        _Result("a", overall_confidence=0.99, fields=_fields(x=0.99)),
        _Result("b", overall_confidence=0.99, fields=_fields(x=0.99)),
        _Result("c", overall_confidence=0.20, fields=_fields(x=0.20)),
        _Result("d", overall_confidence=0.20, fields=_fields(x=0.20)),
    ])
    assert queue.spot_check_share == 0.5
    assert queue.by_routing == {"field_review": 2, "spot_check": 2}


def test_the_spot_check_threshold_is_above_the_review_threshold():
    """A document only needs a light touch when it is well clear of the line
    that would have sent individual fields to review."""
    from common.constants import DEFAULT_REVIEW_CONFIDENCE_THRESHOLD

    assert SPOT_CHECK_THRESHOLD > DEFAULT_REVIEW_CONFIDENCE_THRESHOLD


def test_an_empty_queue_reports_no_saving_rather_than_dividing_by_zero():
    assert build_queue([]).spot_check_share == 0.0


def test_the_queue_is_written_where_reviewers_look(client):
    queue = build_queue([_Result("a", fields=_fields(x=0.9))], model_version="v2")
    key = active_learning.write_queue(queue, client)
    assert key.endswith("v2.json")
    assert client.read_json(key)["documents"] == 1


def test_no_external_pre_annotation_setting_remains():
    """The external backend is gone, so is the permission that guarded it."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = (root / "data_pipeline" / "labeling" / "pre_annotate.py").read_text(encoding="utf-8")
    template = (root / ".env.example").read_text(encoding="utf-8")
    for name in ("ALLOW_EXTERNAL_PREANNOTATION", "EXTERNAL_PREANNOTATION_ENDPOINT"):
        assert name not in source and name not in template


# --------------------------------------------------------------------------
# Every gold needs a review record (decision 2026-10-22)
# --------------------------------------------------------------------------


REVIEWED = {"reviewer_id": "a.person", "review_date": "2026-10-20"}


def _gold(note):
    return {"fideon:provenance": {"provenance": note}} if note else {}


@pytest.mark.parametrize("note", [
    # Notes the old guess missed - and notes it blocked wrongly. Neither decides:
    # without a record every one is blocked, with one every one passes.
    "Claude-drafted seed gold", "drafted with Claude Code", "Created by the generator", "AI-generated",
    "not drafted by AI", "Keyed in by Claude Dupont", "Claudette's labels", "not reviewed by carrier", None,
])
def test_a_gold_trains_on_its_review_record_never_on_its_notes(note):
    from data_pipeline.labeling.export_golden_labels import review_problem

    problem = review_problem(_gold(note), {})
    assert problem and "reviewer_id is missing" in problem and "review_date is missing" in problem
    assert review_problem(_gold(note), REVIEWED) is None


@pytest.mark.parametrize("backend", ["label_studio", "base_qwen3vl", "human", None])
def test_what_drafted_a_gold_is_information_not_the_gate(backend):
    from data_pipeline.labeling.export_golden_labels import review_problem

    assert review_problem({}, {"draft_backend": backend}) is not None
    assert review_problem({}, {**REVIEWED, "draft_backend": backend}) is None


@pytest.mark.parametrize("reviewer", [" ", "", "pending", "Pending", "none", "N/A", "n/a", "TBD", "unknown",
                                      "-", "null", 7, ["a.person"]])
def test_a_placeholder_reviewer_names_nobody(reviewer):
    from data_pipeline.labeling.export_golden_labels import review_problem

    missing = review_problem({}, {**REVIEWED, "reviewer_id": reviewer}).split(". A person")[0]
    assert "reviewer_id" in missing and "review_date" not in missing


@pytest.mark.parametrize("reviewer", ["李明", "Иван Петров", "José Álvarez", "محمد علي", "a.reviewer",
                                      "Claude Dupont", "Ai Tanaka", "annotator_1", "reviewer 2"])
def test_a_person_s_name_in_any_script_is_a_reviewer(reviewer):
    """Names were read on a-z and 0-9 alone: "李明" was "", a name of nobody."""
    from data_pipeline.labeling.export_golden_labels import has_review_record

    assert has_review_record({**REVIEWED, "reviewer_id": reviewer})


@pytest.mark.parametrize("reviewer", ["claude", "Claude", "system", "auto", "fideon_synth", "gpt", "GPT-4o",
                                      "chatgpt", "qwen", "Qwen3-VL-8B-Instruct", "base_qwen3vl", "label_studio",
                                      "Label Studio", "pre_annotate", "pre-annotation", "auto reviewer", "AI",
                                      "bot", "pipeline"])
def test_a_model_system_or_generator_is_no_reviewer(reviewer):
    """Only a person's name shows a person checked a gold (frontier models are
    never used for data)."""
    from data_pipeline.labeling.export_golden_labels import review_problem

    missing = review_problem({}, {**REVIEWED, "reviewer_id": reviewer}).split(". A person")[0]
    assert "names a model, system or generator, not a person" in missing and "review_date" not in missing


@pytest.mark.parametrize("when", ["TBD", "pending", "20/10/2026", "2026-13-01", "", 20261020, "2099-01-01",
                                  "2099-01-01T00:00:00+00:00"])
def test_a_review_date_must_be_an_iso_date_not_in_the_future(when):
    from data_pipeline.labeling.export_golden_labels import review_problem

    missing = review_problem({}, {**REVIEWED, "review_date": when}).split(". A person")[0]
    assert "review_date" in missing and "reviewer_id" not in missing


@pytest.mark.parametrize("when", ["2026-10-20", "2026-10-20T09:30:00", "2026-10-20T09:30:00Z",
                                  "2026-10-20T09:30:00+05:30"])
def test_an_iso_date_or_date_time_is_a_review_date(when):
    from data_pipeline.labeling.export_golden_labels import has_review_record

    assert has_review_record({**REVIEWED, "review_date": when})


def test_a_review_dated_today_in_the_reviewers_zone_is_not_in_the_future():
    from datetime import UTC, datetime, timedelta

    from data_pipeline.labeling.export_golden_labels import has_review_record

    ahead = (datetime.now(UTC) + timedelta(hours=13)).date().isoformat()     # already that day at UTC+13
    assert has_review_record({**REVIEWED, "review_date": ahead})
    later = (datetime.now(UTC) + timedelta(days=2)).date().isoformat()
    assert not has_review_record({**REVIEWED, "review_date": later})


def test_a_twin_is_reviewed_only_through_its_seed():
    from data_pipeline.labeling.export_golden_labels import has_review_record, review_problem

    own = {"synthetic": True, **REVIEWED}
    assert "inherits its record from" in review_problem({}, own)                  # its own record: not enough
    assert has_review_record({**own, "review_inherited_from": "homeowners/seed_1"})
    assert not has_review_record({"synthetic": True, "review_inherited_from": "homeowners/seed_1"})
