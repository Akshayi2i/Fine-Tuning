"""What checkpoint selection measures and records (review #30, #34, #39, #40).

#30: the checkpoint nearest two-thirds of the final step is a candidate, and
whether it won or tied is reported. #34: the winner's validation error tallies
go to Blob in the agreed format. #39: the finalists are compared on the gate
mirror - image_only, whole documents, held to their borrowed page text as
served, real documents only - and the fewest-invented rule decides among those
close on it. #40: the gate's field match is predicted from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from evaluation import checkpoint_eval as C
from evaluation.checkpoint_eval import CheckpointScore, SelectionReport, choose_on_documents
from evaluation.validation_generation import ValidationGeneration, error_tallies, gate_mirror
from tests.test_review_selection_sharding import DOCUMENTS, answer, env, model_for, row, rows


@pytest.fixture
def client():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")


@pytest.fixture
def small_calls(monkeypatch):
    """Two rows a call."""
    from evaluation import validation_generation

    monkeypatch.setattr(validation_generation, "configured_batch_rows", lambda: 2)


def generation(source, mode, got, *, synthetic=False, gold=None):
    r = row(source, mode, synthetic=synthetic)
    return ValidationGeneration(row=r, golden=gold or answer(DOCUMENTS[source][0]), extraction=got)


# --------------------------------------------------------------------------
# #39 - the gate mirror
# --------------------------------------------------------------------------

def test_the_gate_mirror_holds_image_only_answers_to_borrowed_text_and_reads_real_documents_only():
    invented = answer("ZZ-404-9")
    generations = [
        generation("d1", "ocr_plus_image", answer("HO-1")),
        generation("d1", "image_only", invented),                       # invented: taken out as served
        generation("d2", "ocr_plus_image", answer("HO-2")),
        generation("d2", "image_only", answer("HO-2")),
        generation("d3", "image_only", answer("HO-3"), synthetic=True),  # a twin: counts toward nothing
    ]
    mirror = gate_mirror(generations)
    assert mirror["gate_mirror_documents"] == 2.0 and mirror["gate_mirror_synthetic_documents"] == 1.0
    # d1's invented number was removed (held to its page text), so it is a
    # missed value: 1 of d1's 2 fields right, both of d2's -> 0.75.
    assert mirror["gate_mirror_accuracy"] == pytest.approx(0.75)
    assert mirror["gate_mirror_hallucination_rate"] == pytest.approx(0.25)


def test_an_image_only_reading_with_no_text_to_borrow_is_unscored_and_no_document_means_no_mirror():
    alone = [generation("d1", "image_only", answer("HO-1")), generation("d2", "ocr_plus_image", answer("HO-2"))]
    assert gate_mirror(alone) == {}                       # d1 has nothing to borrow: nothing scored
    lent = [*alone, generation("d1", "ocr_plus_image", answer("HO-1"))]
    assert gate_mirror(lent)["gate_mirror_unscored_documents"] == 0.0
    assert gate_mirror([generation("d1", "ocr_plus_image", answer("HO-1"))]) == {}


def _finalists(**by_step):
    report = SelectionReport(scores=[CheckpointScore(f"/ck/checkpoint-{s}", s, {"field_normalized_match": 0.7})
                                     for s in sorted(int(k[1:]) for k in by_step)])
    report.selected = report.scores[-1].checkpoint
    for score in report.scores:
        score.metrics.update(by_step[f"s{score.step}"])
    return report


def _metrics(document, invented, mirror=None):
    out = {"document_field_normalized_match": document, "document_hallucination_rate": invented,
           "document_false_null_rate": 0.1}
    if mirror is not None:
        out.update(gate_mirror_accuracy=mirror, gate_mirror_documents=10.0)
    return out


def test_the_finalists_are_compared_on_the_gate_mirror():
    # Within 0.02 on document accuracy both go on, and the one inventing less
    # (280) would win; on the gate mirror 280 is 0.06 behind and cannot.
    report = _finalists(s280=_metrics(0.80, 0.01, mirror=0.70), s560=_metrics(0.79, 0.03, mirror=0.76))
    choose_on_documents(report, None, finalists=2, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-560"
    assert report.document_choice["field_match_measure"] == "gate_mirror_accuracy"
    assert report.document_choice["field_match_fallback"] is None
    assert report.document_choice["behind_on_field_match"] == ["/ck/checkpoint-280"]
    assert report.gate_mirror_accuracy == 0.76


def test_among_finalists_close_on_the_gate_mirror_the_fewest_inventions_still_win():
    report = _finalists(s280=_metrics(0.80, 0.01, mirror=0.75), s560=_metrics(0.80, 0.03, mirror=0.76))
    choose_on_documents(report, None, finalists=2, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-280"                     # decision 2026-10-24 kept


def test_without_a_gate_mirror_for_every_finalist_document_accuracy_stands_in_and_says_so():
    report = _finalists(s280=_metrics(0.80, 0.01, mirror=0.70), s560=_metrics(0.79, 0.03))
    choose_on_documents(report, None, finalists=2, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-280"
    choice = report.document_choice
    assert choice["field_match_measure"] == "document_field_normalized_match"
    assert "not the gate's measure" in choice["field_match_fallback"]


# --------------------------------------------------------------------------
# #34 - error tallies
# --------------------------------------------------------------------------

def test_error_tallies_count_values_wrong_missed_and_invented_as_the_report_counts_errors():
    from evaluation.run_eval import build_report
    from evaluation.validation_generation import merged_documents

    gold = {"policy": {"policy_number": env("HO-1")}}
    generations = [
        generation("d1", "ocr_plus_image", answer("HO-1"), gold=gold),          # effective date invented
        generation("d2", "ocr_plus_image", {"policy": {}}),                     # both missed
        generation("d3", "image_only", answer("HO-9")),                         # number wrong
    ]
    tallies = error_tallies(generations)
    assert set(tallies) == {"by_field", "by_line", "by_group", "by_mode"}
    number, effective = tallies["by_field"]["policy.policy_number"], tallies["by_field"]["policy.effective_date"]
    assert number["wrong"] == 1 and number["missed"] == 1
    assert effective["invented"] == 1 and effective["missed"] == 1
    assert tallies["by_line"]["cyber"]["values"] == sum(t["values"] for t in tallies["by_field"].values())
    assert set(tallies["by_group"]) == {"decl"} and set(tallies["by_mode"]) == {"ocr_plus_image", "image_only"}
    # Every error the report records is a tallied one, field by field.
    report = build_report("v", merged_documents(generations))
    records = [r for s in report.full_set() for r in s.error_records]
    for path, tally in tallies["by_field"].items():
        assert tally["wrong"] + tally["missed"] + tally["invented"] == sum(r["field_path"] == path for r in records)


def test_list_indexes_are_folded_in_the_field_paths_and_grouped_by_window_group():
    """On a common-model line, whose field match scores the values in rows."""
    gold = {"policy": {"policy_number": env("HO-1")}, "locations": [{"address": env("1 Main St")},
                                                                   {"address": env("2 Main St")}]}
    got = {"policy": {"policy_number": env("HO-1")}, "locations": [{"address": env("1 Main St")}]}
    read = generation("d1", "ocr_plus_image", got, gold=gold)
    read.row["lob"] = ["homeowners"]
    tallies = error_tallies([read])
    assert tallies["by_field"]["locations[].address"] == {"values": 2, "wrong": 0, "missed": 1, "invented": 0}
    assert set(tallies["by_group"]) == {"decl", "arrays"} and set(tallies["by_line"]) == {"homeowners"}


# --------------------------------------------------------------------------
# #30 - the two-thirds candidate
# --------------------------------------------------------------------------

def _run(tmp_path, steps):
    run = tmp_path / "v0-20261026"
    for step in steps:
        (run / f"checkpoint-{step}").mkdir(parents=True)
    return [str(run / f"checkpoint-{step}") for step in steps]


def test_the_two_thirds_checkpoint_is_the_one_on_disk_nearest_two_thirds_of_the_final_step(tmp_path):
    on_disk = _run(tmp_path, (70, 140, 200, 210, 280, 300))
    evaluated = [c for c in on_disk if not c.endswith(("-200", "-210"))]   # 200 is a resume point
    assert C.two_thirds_checkpoint(evaluated) == on_disk[2]                # step 200 = 2/3 of 300
    assert C.two_thirds_checkpoint([]) is None


def test_a_tie_goes_to_the_later_step(tmp_path):
    on_disk = _run(tmp_path, (100, 300))
    assert C.two_thirds_checkpoint(on_disk) == on_disk[1]                  # 100 and 300 both 100 from 200


def test_it_is_added_once_and_flagged(tmp_path):
    on_disk = _run(tmp_path, (50, 100, 150, 200, 250, 300))
    candidates = C.select_candidates(on_disk, trailing=3, two_thirds=on_disk[3])
    assert [c.step for c in candidates] == [200, 250, 300] and not any(c.is_two_thirds for c in candidates)
    candidates = C.select_candidates(on_disk[:2] + on_disk[4:], trailing=3, two_thirds=on_disk[3])
    assert [c.step for c in candidates if c.is_two_thirds] == [200]


@pytest.mark.parametrize("f1,won,tied", [(0.66, True, False), (0.645, False, True), (0.60, False, False)])
def test_whether_it_won_or_tied_is_reported(tmp_path, f1, won, tied):
    on_disk = _run(tmp_path, (100, 200, 250, 300))
    scores = {on_disk[1]: f1, on_disk[2]: 0.62, on_disk[3]: 0.65, on_disk[0]: 0.5}
    report = C.select_best(on_disk[2:], lambda c: {"field_normalized_match": scores[c]}, two_thirds=on_disk[1])
    outcome = C.two_thirds_outcome(report, sample_margin=0.015)
    assert outcome["checkpoint"] == on_disk[1] and outcome["added"] is True
    assert (outcome["won"], outcome["tied"]) == (won, tied)
    assert report.as_dict()["candidates"][0]["is_two_thirds"] is True


# --------------------------------------------------------------------------
# #40 - the early warning
# --------------------------------------------------------------------------

def test_the_gate_field_match_is_predicted_from_two_earlier_cycles():
    history = [{"version": "v1", "gate_mirror_accuracy": 0.90, "gate_field_match": 0.92},
               {"version": "v2", "gate_mirror_accuracy": 0.88, "gate_field_match": 0.92},
               {"version": "v3", "gate_mirror_accuracy": None, "gate_field_match": 0.93}]
    warning = C.predicted_gate_field_match(0.89, history)
    assert warning["cycles"] == 2 and warning["mean_offset"] == pytest.approx(0.03)
    assert warning["predicted"] == pytest.approx(0.92)
    assert warning["summary"] == "predicted gate field match ≈ 0.920 (mean offset over 2 cycles)"
    one = C.predicted_gate_field_match(0.89, history[:1])
    assert one["predicted"] is None and one["summary"] is None and "needs 2" in one["note"]


# --------------------------------------------------------------------------
# The checkpoint stage: all of it recorded
# --------------------------------------------------------------------------

class _Scorer:
    """Sample scores that are the finalists' too (a split that is its own sample)."""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, checkpoint):
        return dict(self.metrics[checkpoint])


def _stage_ctx(client, checkpoints, scorer, version="v3"):
    from orchestration.pipeline_dag import StageContext

    return StageContext(client=client, controller=None, out_version=version,  # type: ignore[arg-type]
                        checkpoints=list(checkpoints), checkpoint_scorer=scorer)


def test_the_selection_summary_reports_the_two_thirds_candidate_and_the_predicted_gate(client, tmp_path):
    from orchestration.pipeline_dag import stage_checkpoint_eval

    for version, mirror, gate in (("v1", 0.90, 0.92), ("v2", 0.88, 0.92)):
        client.write_json(paths.checkpoint_selection(version, scope="policy"),
                          {"validation_gate_mirror_accuracy": mirror})
        client.write_json(paths.gate_decision(version, scope="policy"),
                          {"candidate_metrics": {"field_normalized_match": gate}})
    on_disk = _run(tmp_path, (100, 200, 250, 300))
    metrics = {c: {"field_normalized_match": f, **_metrics(f, 0.02, mirror=f - 0.01)}
               for c, f in zip(on_disk, (0.50, 0.648, 0.62, 0.65), strict=True)}
    result = stage_checkpoint_eval(_stage_ctx(client, on_disk[2:], _Scorer(metrics)))
    assert result.data["selected"] == on_disk[3]
    assert "two-thirds candidate checkpoint-200 (step 200 of 300): tied" in result.detail
    assert "predicted gate field match ≈ 0.670 (mean offset over 2 cycles)" in result.detail
    record = client.read_json(paths.checkpoint_selection("v3", scope="policy"))
    assert record["validation_gate_mirror_accuracy"] == pytest.approx(0.64)
    assert record["two_thirds_candidate"]["tied"] is True
    assert record["early_warning"]["predicted"] == pytest.approx(0.67)
    assert record["finalist_pass"]["saved"] is False                   # this scorer saves nothing


def test_the_selected_finalists_answers_and_error_tallies_are_kept_in_blob(client, small_calls, tmp_path):
    from orchestration.pipeline_dag import stage_checkpoint_eval

    on_disk = _run(tmp_path, (10, 20))
    scorer = C.generation_scorer(rows(("ocr_plus_image", "image_only")), model_for(client), finalists=True,
                                 save_dir=tmp_path / "finalists")
    result = stage_checkpoint_eval(_stage_ctx(client, on_disk, scorer, version="v1"))
    winner = result.data["selected"]
    assert winner == on_disk[0]                         # checkpoint-20 invents d2's number from the image
    kept = result.data["finalist_pass"]
    assert kept["saved"] is True and kept["checkpoint"] == winner
    assert kept["blob"] == paths.validation_generations_key("policy-v1")
    errors = client.read_json(paths.validation_errors_key("policy-v1"))
    assert set(errors) == {"version", "run_id", "checkpoint", "by_field", "by_line", "by_group", "by_mode"}
    assert (errors["version"], errors["run_id"], errors["checkpoint"]) == (1, "policy-v1", winner)
    assert set(errors["by_field"]["policy.policy_number"]) == {"values", "wrong", "missed", "invented"}
    assert set(errors["by_mode"]) == {"ocr_plus_image", "image_only"}
    saved = client.read_text(kept["blob"]).splitlines()
    assert json.loads(saved[0])["header"]["checkpoint"] == winner and len(saved) == 1 + 6
    assert Path(kept["local"]).is_file()


def test_the_keys_are_tenant_scoped_beside_the_run():
    key = paths.validation_errors_key("policy-v3", "broker_a")
    assert key == "run-artifacts/broker_a/policy-v3/validation_errors.json"
    assert paths.tenant_of(key) == "broker_a" and paths.is_tenant_scoped(key)
    assert paths.validation_generations_key("policy-v3", "broker_a").startswith("run-artifacts/broker_a/policy-v3/")
    with pytest.raises(paths.PathError):
        paths.validation_errors_key("not a run")
