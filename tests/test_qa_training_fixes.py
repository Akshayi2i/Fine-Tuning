"""Regressions for the training-side QA findings (2026-10-24).

* **H6** - an eval report is reused for the gate only when it was scored on
  the frozen set as it stands and by the scoring code running now.
* **M12** - a cycle that continues an adapter never validates (or tests) on a
  document the adapter's chain of runs trained on.
* **L3** - the selection margin is the selected checkpoint's, negative when a
  later decision chose one the sample ranks behind.
* Hardening: NaN gating metrics fail closed; a fault rate with nothing counted
  is absent, not a perfect 0.0; ECE refuses confidences that are not
  probabilities; an all-null label is set aside; unprinted-value paths survive
  label verification's list removals; a run resumed on another GPU count keeps
  its own checkpoint cadence and its earlier evaluation checkpoints.

Synthetic fixtures only: the training data is being replaced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration.pipeline_dag import PipelineError, run_stages, stages_for, stages_from
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import (
    PASSING_METRICS,
    _scored_on,
    freeze_as_scored,
    make_context,
    promote_in_index,
    seed_corpus,
    seed_policy_copy,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


@pytest.fixture
def released(client, controller) -> BlobClient:
    """Release v1 packaged and promoted, everything a cycle needs to continue
    it (as tests/test_incremental_training.py's fixture of the same name)."""
    from tests.test_incremental_training import FROZEN_AT, _report, write_adapter

    seed_corpus(client)
    report = run_stages(make_context(client, controller, out_version="v1"), stages_for("all"),
                        command="all")
    assert report.ok, report.render()
    promote_in_index(client, "v1")
    freeze_as_scored(client, frozen_at=FROZEN_AT)
    client.write_json(paths.eval_report("v1", scope="policy"), _report({"auto": [0.95] * 6}, policy=200))
    write_adapter(client, "v1")
    return client


# --------------------------------------------------------------------------
# H6: a report scored on another yardstick is scored again, never reused
# --------------------------------------------------------------------------

T1, T2 = "2026-10-01T00:00:00+00:00", "2026-10-20T00:00:00+00:00"


def _stale_report(client: BlobClient, eval_set: dict) -> None:
    client.write_json(paths.eval_report("v1", scope="policy"), {
        "gate_metrics": {**PASSING_METRICS, "field_normalized_match": 0.99}, "eval_set": eval_set})
    # The lines the release serves, from its corpus: the scoring needs them.
    client.write_json(paths.corpus_manifest("v1"), {"policy_line_counts": {"homeowners": 1}})


@pytest.mark.parametrize("frozen_now, eval_set, why", [
    (T2, _scored_on(frozen_at=T1), "the set was frozen again"),
    (T1, _scored_on(fingerprint="0000old0scorer00"), "other scoring code"),
    (None, _scored_on(frozen_at=T1), "no set is frozen now"),
    (T1, {**_scored_on(), "frozen_at": None}, "does not record which freeze"),
])
def test_a_report_on_another_yardstick_is_scored_again(client, controller, frozen_now, eval_set, why):
    """The candidate's report was scored on the set frozen at T1; the set was
    then deleted and frozen again with corrected golds (or the scoring code
    changed). Reused, the gate passed on the old numbers and recorded the new
    set beside them."""
    from orchestration.pipeline_dag import eval_report_metrics

    if frozen_now:
        freeze_as_scored(client, frozen_at=frozen_now)
    _stale_report(client, eval_set)

    with pytest.raises(PipelineError, match=f"not reused.*{why}.*metrics nobody measured"):
        eval_report_metrics(make_context(client, controller, metrics_provider=None))

    class Loaded(Exception):
        pass

    def loader(_ctx, _fmt):
        raise Loaded
    # Off a dry run it is scored again, as when there is no report at all.
    with pytest.raises(Loaded):
        eval_report_metrics(make_context(client, controller, metrics_provider=None, dry_run=False,
                                         serving_model_loader=loader))


def test_a_report_on_today_s_yardstick_is_still_reused(client, controller):
    from orchestration.pipeline_dag import eval_report_metrics

    freeze_as_scored(client, frozen_at=T2)
    _stale_report(client, _scored_on(frozen_at=T2))
    assert eval_report_metrics(make_context(client, controller, metrics_provider=None))[
        "field_normalized_match"] == 0.99


def test_the_gate_does_not_pass_a_first_release_on_a_refrozen_set_s_report(client, controller):
    """The gate stage itself: no decision recorded as passed on stale numbers."""
    seed_corpus(client)
    freeze_as_scored(client, frozen_at=T2)
    client.write_json(paths.eval_report("v1", scope="policy"), {
        "gate_metrics": {**PASSING_METRICS, "field_normalized_match": 0.99},
        "eval_set": _scored_on(frozen_at=T1)})

    report = run_stages(make_context(client, controller, metrics_provider=None), stages_for("all"),
                        command="all")
    assert report.failed_at == "evaluation_gate"
    assert "frozen again" in report.results[-1].detail
    assert not client.exists(paths.gate_decision("v1", scope="policy"))


# --------------------------------------------------------------------------
# M12: what the continued chain trained on stays in train
# --------------------------------------------------------------------------


def _records(n: int, other: int = 50):
    from data_pipeline.dataset_builder.split_groups import GroupRecord

    recs = [GroupRecord(group_id=f"ho_{i:03d}", doc_type="policy", source_ids=[f"ho_{i:03d}"],
                        carrier=f"carrier_{i}", line="homeowners") for i in range(n)]
    recs += [GroupRecord(group_id=f"pa_{i:03d}", doc_type="policy", source_ids=[f"pa_{i:03d}"],
                         carrier=f"pc_{i % 4}", line="personal_auto") for i in range(other)]
    return {"policy": recs}


def test_a_line_crossing_the_measure_minimum_keeps_what_the_chain_trained_on_in_train():
    """Corpus N: homeowners has 4 documents, all train. Corpus N+1: a fifth
    arrives, the line is measured, and ho_000 hashes to val - a document the
    continued adapter memorised, now selecting its checkpoint."""
    from data_pipeline.dataset_builder.split_groups import assign_group_splits

    before = assign_group_splits(_records(4), seed=42, with_test=False)
    trained = {g for g, split in before.assignment.items() if split == "train"}
    assert "ho_000" in trained

    drawn = assign_group_splits(_records(5), seed=42, with_test=False)
    assert drawn.split_of("ho_000") == "val"            # what the rules alone draw

    after = assign_group_splits(_records(5), seed=42, with_test=False, trained=trained)
    assert all(after.split_of(g) == "train" for g in trained if g in after.assignment)
    assert after.pinned_to_train["policy"]["ho_000"] == "val"
    assert "ho_000" not in after.val_half
    record = after.as_dict()
    assert record["pinned_to_train"] == {"policy": after.pinned_to_train["policy"]}
    # Families the chain never saw split as before.
    assert {g: s for g, s in after.assignment.items() if g not in trained} == {
        g: s for g, s in drawn.assignment.items() if g not in trained}


def test_a_pinned_family_beats_the_carrier_hold_out_and_the_repair():
    from data_pipeline.dataset_builder.split_groups import (
        GroupRecord,
        _repair_empty_splits,
        assign_group_splits,
    )

    groups = [GroupRecord(group_id=f"g{i}", doc_type="policy", source_ids=[f"g{i}"], line="ho")
              for i in range(4)]
    buckets = {"train": ["g0", "g1", "g2", "g3"], "val": [], "test": []}
    _repair_empty_splits(buckets, groups, "policy/ho", 42, starved=("val",),
                         pinned={"g0", "g1", "g2", "g3"})
    assert buckets["val"] == []                          # nothing movable: nothing moved
    _repair_empty_splits(buckets, groups, "policy/ho", 42, starved=("val",), pinned={"g0", "g1"})
    assert buckets["val"] and buckets["val"][0] in {"g2", "g3"}

    # Every family of every carrier trained before: none is held out to test.
    recs = [GroupRecord(group_id=f"h{i:02d}", doc_type="policy", source_ids=[f"h{i:02d}"],
                        carrier=f"carrier_{i}", line="homeowners") for i in range(40)]
    everything = {r.source_ids[0] for r in recs}
    pinned = assign_group_splits({"policy": recs}, seed=42, trained=everything)
    assert set(pinned.assignment.values()) == {"train"}
    assert not pinned.held_out_source_ids
    assert set(pinned.pinned_to_train["policy"].values()) <= {"val", "test"}


def test_a_delivered_split_keeps_what_the_chain_trained_on_in_train():
    from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_delivered_splits

    recs = [GroupRecord(group_id=f"s{i}", doc_type="policy", source_ids=[f"s{i}"], line="ho")
            for i in range(4)]
    # A continued cycle gates on a frozen set (its parent's report must be on
    # it), so the delivery takes no test: a trained document re-delivered as
    # test is pinned, not refused as a new test document.
    delivered = {"s0": "train", "s1": "val", "s2": "val", "s3": "test"}
    result = assign_delivered_splits({"policy": recs}, delivered, with_test=False, trained={"s1", "s3"})
    assert result.split_of("s1") == "train" and result.split_of("s3") == "train"
    assert result.split_of("s2") == "val"
    assert result.pinned_to_train == {"policy": {"s1": "val", "s3": "test"}}
    # With every validation family trained, there is no validation left: refused.
    from data_pipeline.dataset_builder.split_groups import SplitError

    with pytest.raises(SplitError, match="no val"):
        assign_delivered_splits({"policy": recs}, delivered, with_test=False,
                                trained={"s1", "s2", "s3"})


def test_a_continuing_cycle_s_corpus_pins_what_its_chain_trained_on(released, controller):
    """Through the pipeline: the build asks the question training asks
    (continuation), reads the chain's train documents and pins them."""
    from orchestration.pipeline_dag import plan_corpus

    # Families of their own, so the line is measured and some hash to val.
    copies = [seed_policy_copy(released, f"policy_{100 + i:04d}", own_family=True) for i in range(10)]
    # Say v1's corpus trained every document the next build loads, so any the
    # rules would put in val must be pinned.
    key = paths.corpus_manifest("v1")
    corpus = released.read_json(key)
    every = sorted({s for ids in corpus["source_ids_by_split"].values() for s in ids} | set(copies))
    released.write_json(key, {**corpus, "source_ids_by_split": {**corpus["source_ids_by_split"],
                                                               "train": every}})

    fresh = plan_corpus(make_context(released, controller, out_version="v2", fresh=True))
    held = {g: s for g, s in fresh.assignment.assignment.items() if s != "train"}
    assert held, "the fixture build puts no family outside train; nothing to pin"
    assert fresh.assignment.continues is None and not fresh.assignment.pinned_to_train

    plan = plan_corpus(make_context(released, controller, out_version="v2"))
    assert plan.assignment.continues == "policy-v1"
    assert set(plan.assignment.assignment.values()) == {"train"}
    assert plan.assignment.pinned_to_train == {"policy": held}
    assert not plan.built.rows_by_split["val"] and not plan.built.rows_by_split["test"]

    # Written on the corpus manifest, with the split.
    report = run_stages(make_context(released, controller, out_version="v2"),
                        stages_for("finetune"), command="finetune")
    assert report.ok, report.render()
    split = released.read_json(paths.corpus_manifest("v2"))["split_assignment"]
    assert split["continues"] == "policy-v1" and split["pinned_to_train"] == {"policy": held}


def test_a_corpus_validating_on_what_the_chain_trained_on_trains_from_the_base(released, controller):
    """A corpus built for another chain, or reused with --corpus-version: its val
    holds a document v1 trained on, so continuing v1 on it would select the
    checkpoint on a memorised document. The cycle starts from the base, and
    says why."""
    from registry_utils.query_registry import get

    key = paths.corpus_manifest("v1")
    corpus = released.read_json(key)
    seen = corpus["source_ids_by_split"]["train"][0]
    released.write_json(key, {**corpus, "source_ids_by_split": {
        **corpus["source_ids_by_split"], "val": [*corpus["source_ids_by_split"].get("val", []), seen]}})

    ctx = make_context(released, controller, out_version="v3", corpus_version="v1")
    report = run_stages(ctx, stages_from("training"), command="finetune")
    assert report.ok, report.render()
    manifest = get("policy-v3", released)
    assert manifest.parent_run_id is None
    assert "memorised" in manifest.fresh_reason and seen in manifest.fresh_reason


def test_a_chain_whose_documents_cannot_be_read_is_not_continued(released, controller):
    from orchestration.pipeline_dag import continuation

    released.delete(paths.corpus_manifest("v1"))
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None
    assert "chain of runs trained on cannot be read" in why and "memorised" in why


# --------------------------------------------------------------------------
# L3: the recorded margin is the selected checkpoint's
# --------------------------------------------------------------------------


def test_the_margin_is_the_selected_checkpoint_s_when_documents_pick_the_runner_up():
    from evaluation.checkpoint_eval import choose_on_documents, select_best

    sample = {"ck/checkpoint-100": {"field_normalized_match": 0.90},
              "ck/checkpoint-200": {"field_normalized_match": 0.88}}
    full = {"ck/checkpoint-100": {"document_field_normalized_match": 0.80,
                                  "document_hallucination_rate": 0.10},
            "ck/checkpoint-200": {"document_field_normalized_match": 0.79,
                                  "document_hallucination_rate": 0.01}}
    report = select_best(list(sample), lambda c: sample[c])
    assert report.margin == pytest.approx(0.02)          # the leader, selected
    report = choose_on_documents(report, lambda c: full[c], finalists=2, guard_margin=0.02)
    assert report.selected == "ck/checkpoint-200"
    assert report.margin == pytest.approx(-0.02)
    assert report.as_dict()["margin_over_runner_up"] == pytest.approx(-0.02)


def test_the_margin_follows_a_tie_break_that_changed_the_choice():
    from evaluation.checkpoint_eval import break_tie, select_best

    sample = {"ck/checkpoint-100": {"field_normalized_match": 0.901},
              "ck/checkpoint-200": {"field_normalized_match": 0.900}}
    full = {"ck/checkpoint-100": {"field_normalized_match": 0.80},
            "ck/checkpoint-200": {"field_normalized_match": 0.85}}
    report = break_tie(select_best(list(sample), lambda c: sample[c]), lambda c: full[c], 0.005)
    assert report.selected == "ck/checkpoint-200"
    assert report.tie_break["margin_on_sample"] == pytest.approx(0.001)
    assert report.margin == pytest.approx(-0.001)


# --------------------------------------------------------------------------
# Hardening
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name, value", [
    ("hallucination_rate", float("nan")),
    ("run_to_run_agreement", float("nan")),
    ("run_to_run_agreement", float("inf")),
    ("field_normalized_match", float("nan")),
])
def test_a_metric_that_is_not_a_number_fails_closed_on_a_first_release(name, value):
    """No floor and no baseline: NaN lost every comparison, so nothing compared
    it, and a first release passed."""
    from evaluation.gating import promotion_gate

    result = promotion_gate({**PASSING_METRICS, "hallucination_rate": 0.0,
                             "run_to_run_agreement": 1.0, name: value}, None)
    assert not result.passed
    assert name in result.failed_gates
    assert any(f"{name} is" in r and "not a finite number" in r for r in result.blocking_reasons)
    verdict = next(v for v in result.verdicts if v.name == name)
    assert verdict.not_a_number and not verdict.meets_floor and not verdict.non_inferior
    assert "NOT A NUMBER" in verdict.describe()
    assert verdict.as_dict()["meets_floor"] is False


def test_finite_first_measurements_still_pass_on_no_floor():
    from evaluation.gating import promotion_gate

    result = promotion_gate({**PASSING_METRICS, "hallucination_rate": 0.0,
                             "run_to_run_agreement": 1.0}, None)
    assert result.passed, result.blocking_reasons


def test_a_fault_rate_with_nothing_counted_is_absent_not_perfect():
    from evaluation.metrics.extraction_faults import FaultReport, score_false_nulls

    empty = score_false_nulls([({"policy": {"term": None}}, {})])
    assert empty.opportunities == 0 and empty.rate is None
    assert empty.as_dict()["rate"] is None
    assert FaultReport("x", hits=1, opportunities=4).rate == 0.25


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.2, float("inf")])
def test_ece_refuses_a_confidence_that_is_not_a_probability(bad):
    """Out of [0, 1] or NaN fell in no bin but counted in the total: ECE came
    out smaller - better - than it was."""
    from evaluation.metrics.coverage_metrics import expected_calibration_error

    with pytest.raises(ValueError, match="not probabilities in"):
        expected_calibration_error([0.9, bad, 0.4], [True, False, True])


def test_ece_reads_float_rounding_at_the_edge_as_the_edge():
    from evaluation.metrics.coverage_metrics import expected_calibration_error

    rounded = expected_calibration_error([1.0 + 1e-9, -1e-9], [True, False])
    assert rounded == expected_calibration_error([1.0, 0.0], [True, False]) == 0.0
    assert expected_calibration_error([0.95] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.45)


def _all_null(node):
    from common.canonical import is_field_value

    if is_field_value(node):
        return {**node, "raw": None, "parsed": None}
    if isinstance(node, dict):
        return {k: _all_null(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_all_null(v) for v in node]
    return node


def test_a_label_whose_every_value_is_null_is_set_aside_with_its_reason():
    """It passed the flat-label guard (it has envelopes) and trained targets of
    empty sections: 'this policy states nothing'."""
    from data_pipeline.dataset_builder.build_jsonl import build_corpus, states_nothing
    from tests.test_dataset_builder import _policy_documents, assign_splits

    docs = _policy_documents(6)
    docs[0].golden_label = _all_null(docs[0].golden_label)
    assert states_nothing(docs[0].golden_label)
    assert not states_nothing(docs[1].golden_label)
    assert not states_nothing({"policy": {"term": "flat"}})       # flat: the other guard's

    result = build_corpus(docs, assign_splits({"policy": [d.source_id for d in docs]}))
    assert result.set_aside["empty_label"] == 1
    assert any(sid == docs[0].source_id and "every value" in why for sid, why in result.skipped)
    assert docs[0].source_id not in {r["source_id"] for r in result.all_rows}


def test_unprinted_paths_are_read_in_the_verified_label_s_numbering():
    from data_pipeline.dataset_builder.label_verification import paths_after_removal, without_paths

    assert paths_after_removal(
        ["a.b[2]", "a.b[1].x", "a.b[0]", "c[3].d[2]", "a.b[5]", "e"],
        ["a.b[1]", "a.b[3]", "c[3].d[0]"],
    ) == ["a.b[1]", "a.b[0]", "c[3].d[1]", "a.b[3]", "e"]
    label = {"a": {"b": ["x0", "x1", "x2"]}, "k": {"v": 1, "w": 2}}
    assert without_paths(label, ["a.b[0]", "a.b[2]", "k.w", "nowhere[3]"]) == {
        "a": {"b": ["x1"]}, "k": {"v": 1}}
    assert label["a"]["b"] == ["x0", "x1", "x2"]           # a copy


def _applicable(raw: str, page: int) -> dict:
    return {"raw": raw, "parsed": raw, "confidence": {"score": 1.0, "source": "vlm"},
            "page_ref": [page], "flagged": False}


def test_a_value_no_page_prints_is_held_out_after_verification_drops_a_member_before_it():
    """applicable_coverages[1] was rule-added and is not printed: verification
    drops it, and [2] becomes [1]. The unprinted [2], recorded against the gold,
    resolved to nothing and its window was taught a value no page prints."""
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, _policy_window_rows
    from tests.test_label_rules import PAGES, _gold

    unprinted = "Quillfeather Unprinted Endorsement"
    gold = _gold(document={"applicable_coverages": [
        _applicable("Mercury Casualty Company", 1),
        _applicable("Zyxwvut Phantom Coverage Part", 2),
        _applicable(unprinted, 2)]})
    gold["fideon:filled"] = {"paths": ["document.applicable_coverages[1]"]}
    texts = [PAGES[number] for number in sorted(PAGES)]
    document = SourceDocument(source_id="s1", doc_type="policy", lob="homeowners", ocr_pages=texts,
                              golden_label=gold, image_paths=[f"p{n}.png" for n in sorted(PAGES)],
                              unprinted_values=["document.applicable_coverages[2]"])
    details: list[str] = []
    rows = _policy_window_rows(document, "train", "ocr_plus_image", texts, details)
    assert any("left out" in d and "document.applicable_coverages[1]" in d for d in details), details
    assert all(unprinted not in row["messages"][-1]["content"] for row in rows)


# --------------------------------------------------------------------------
# Resume on another GPU count
# --------------------------------------------------------------------------


def _state(checkpoint: Path, **state) -> None:
    (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")


def test_a_resume_keeps_the_cadence_the_run_started_with(tmp_path):
    """1 GPU saved every 70 steps and evaluated every 280, keeping 4 x 4 + 1.
    On 4 GPUs the trainer still saves every 70 (it restores the cadence from the
    checkpoint) but the new arguments kept 4 x 1 + 1: rotation deleted the
    earlier evaluation checkpoints selection compares."""
    from training import train as T

    one = T._checkpoint_cadence(280, {"checkpoint_every_steps": 70, "save_total_limit": 4})
    four = T._checkpoint_cadence(280, {"checkpoint_every_steps": 70, "save_total_limit": 4}, gpus=4)
    assert one["save_total_limit"] == 17 and four["save_total_limit"] == 5

    checkpoint = tmp_path / "checkpoint-140"
    checkpoint.mkdir()
    _state(checkpoint, global_step=140, save_steps=one["save_steps"], eval_steps=one["eval_steps"])
    assert T.resumed_cadence(checkpoint, {"save_total_limit": 4}) == one
    _state(checkpoint, global_step=140)
    assert T.resumed_cadence(checkpoint, {"save_total_limit": 4}) is None


def test_a_resumed_run_launches_on_its_own_cadence(client, monkeypatch, tmp_path):
    from tests.test_incremental_training import _as_if_on_the_pod, _train
    from tests.test_stage_data import _seed
    from tests.test_training_resume import _checkpoint
    from training import train as T

    _seed(client)
    launched = _as_if_on_the_pod(monkeypatch, tmp_path / "volume")
    monkeypatch.setattr(T, "launch", lambda config: (_ for _ in ()).throw(RuntimeError("GPU lost")))
    with pytest.raises(RuntimeError):
        _train(client, dry_run=False)
    staging = Path(paths.scoped_staging_adapter_dir("policy", "v9"))
    checkpoint = _checkpoint(staging / "v0-x", 7)
    _state(checkpoint, global_step=7, save_steps=7, eval_steps=21)
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    _train(client, dry_run=False)
    args = launched[-1].args
    assert args["resume_from_checkpoint"] == str(checkpoint)
    assert (args["save_steps"], args["eval_steps"]) == (7, 21)
    limit = T.training_config(T.ADAPTER.training_config)["evaluation"]["save_total_limit"]
    assert args["save_total_limit"] == int(limit) * 3 + 1


def test_selection_compares_every_checkpoint_an_evaluation_fell_on(tmp_path):
    """Evaluations at 105 and 210 (the first pod's cadence), then at 300 (the
    resumed one's 100). Filtered on the last cadence alone, 105 and 210 were
    dropped from selection."""
    from evaluation.checkpoint_eval import discover_checkpoints

    run = tmp_path / "v0-x"
    for step in (35, 70, 105, 140, 175, 210, 300, 350):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    log_history = [{"loss": 1.0, "step": 35}, {"eval_loss": 0.9, "step": 105},
                   {"eval_loss": 0.8, "step": 210}, {"eval_loss": 0.7, "step": 300},
                   {"loss": 0.6, "step": 350}]
    _state(run / "checkpoint-350", global_step=350, eval_steps=100, save_steps=50,
           log_history=log_history)
    checkpoints, _best = discover_checkpoints(str(tmp_path))
    assert [int(c.rsplit("-", 1)[1]) for c in checkpoints] == [105, 210, 300, 350]


def test_nan_is_never_an_auto_pass_in_float_comparisons():
    """Why the gate needs the explicit check: every comparison with NaN is False."""
    nan = float("nan")
    assert not (nan <= 0.02) and not (nan > -0.01) and math.isnan(nan)


# --------------------------------------------------------------------------
# Train/serve parity: the window the declarations lead merges first
# --------------------------------------------------------------------------


def _decl_window(index: int, pages: list[int], number: str, *, leads: bool):
    from evaluation.validation_generation import ValidationGeneration

    answer = {"document": {}, "carrier": {}, "named_insured": {},
              "policy": {"policy_number": {"raw": number, "parsed": number, "page_ref": [pages[0]]}}}
    row = {"doc_type": "policy", "source_id": "s1", "lob": "homeowners", "sections": "decl",
           "window_index": index, "window_pages": pages, "window_leads": leads,
           "modality_mode": "image_only", "messages": []}
    return ValidationGeneration(row=row, golden=answer, extraction=answer)


def test_a_split_declarations_window_s_lead_piece_wins_in_the_validation_merge_as_in_serving():
    """Split by its text, the declarations window's first piece holds page 1 -
    a renewal notice printing the PRIOR policy number - and the piece the
    declarations lead holds page 3. Serving merges the lead piece first, so its
    number wins; checkpoint selection merged by first page, and judged the
    other answer."""
    from evaluation.validation_generation import merged_windows
    from serving.policy_merge import PolicyWindow, merge_policy_windows

    notice = _decl_window(0, [1, 2], "HO-PRIOR-0001", leads=False)
    declarations = _decl_window(1, [3, 4], "HO-CURRENT-0002", leads=True)
    served = merge_policy_windows([
        PolicyWindow(group="decl", pages=[1, 2], extraction=notice.extraction),
        PolicyWindow(group="decl", pages=[3, 4], extraction=declarations.extraction, leads=True),
    ], lob="homeowners")
    assert served.extraction["policy"]["policy_number"]["raw"] == "HO-CURRENT-0002"

    document = merged_windows([notice, declarations])[0]
    assert document.answer["policy"]["policy_number"]["raw"] == "HO-CURRENT-0002"
    assert document.gold["policy"]["policy_number"]["raw"] == "HO-CURRENT-0002"


def test_training_rows_record_which_window_the_declarations_lead():
    """What the validation merge reads back (window_leads), from the plan the
    rows were cut from."""
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, _policy_window_rows
    from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
    from tests.test_label_rules import PAGES, _gold

    texts = [PAGES[number] for number in sorted(PAGES)]
    document = SourceDocument(source_id="s1", doc_type="policy", lob="homeowners", ocr_pages=texts,
                              golden_label=_gold(), image_paths=[f"p{n}.png" for n in sorted(PAGES)])
    rows = _policy_window_rows(document, "train", "ocr_plus_image", texts, [])
    every_page, declarations_page = pages_to_read(texts, len(texts))
    plans = {(p.group, p.window_index): p.leads
             for p in plan_windows("homeowners", every_page, declarations_page, texts)}
    assert rows and all(row["window_leads"] == plans[(row["sections"], row["window_index"])]
                        for row in rows)
    assert any(row["window_leads"] for row in rows if row["sections"] == "decl")


def test_an_invented_parsed_behind_a_blank_raw_counts_as_serving_counts_it():
    """Serving looks a value up by its printed forms (raw, else parsed's), so an
    invented parsed behind a blank raw is held back there; validation looked up
    raw only and never saw it, so selection read a lower rate than the gate."""
    import copy

    from evaluation.metrics.extraction_faults import score_hallucinations
    from serving.printed_values import hold_to_printed

    pages = {1: "Named Insured: Rivera Fabrication LLC  Policy NIP0115"}
    answer = {"named_insured": {"primary_name": {"raw": "Rivera Fabrication LLC",
                                                 "parsed": "Rivera Fabrication LLC", "page_ref": [1]}},
              "policy": {"program_name": {"raw": "", "parsed": "Invented Program Name",
                                          "page_ref": [1]}}}
    record = hold_to_printed(copy.deepcopy(answer), {}, pages).record
    report = score_hallucinations([({}, answer, pages)])
    assert (report.opportunities, report.hits) == (record["checked"], len(record["not_printed"])) == (2, 1)
    assert report.instances[0]["field"] == "policy.program_name"
