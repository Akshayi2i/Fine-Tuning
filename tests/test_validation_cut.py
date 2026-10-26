"""The validation cut: a fixed stratified sample for in-training checks and checkpoint
selection, and a near-tie broken on the full validation split."""

from __future__ import annotations

import pytest

from evaluation.checkpoint_eval import CheckpointScore, SelectionReport, break_tie
from evaluation.validation_sample import validation_sample


def _rows():
    rows = []
    for lob, n in (("homeowners", 1000), ("personal_auto", 300), ("motorcycle", 30)):
        for i in range(n):
            for mode in ("ocr_plus_image", "noisy_ocr_image", "image_only"):
                rows.append({"source_id": f"{lob}-{i}", "lob": [lob], "modality_mode": mode,
                             "sections": "decl", "doc_type": "policy"})
    return rows


def test_the_sample_is_its_size_and_keeps_every_line_and_mode():
    sample = validation_sample(_rows(), 400)
    assert len(sample) == 400
    groups = {(r["lob"][0], r["modality_mode"]) for r in sample}
    assert len(groups) == 9                                   # 3 lines x 3 modes, none dropped
    homeowners = sum(r["lob"][0] == "homeowners" for r in sample)
    assert 280 <= homeowners <= 320                           # roughly its 75% share


def test_the_same_rows_are_chosen_whatever_their_order_or_format():
    """The training checks and checkpoint selection read rows in different files
    and processes; they must pick the same rows."""
    rows = _rows()
    first = validation_sample(rows, 400)
    reshuffled = validation_sample(list(reversed(rows)), 400)
    identity = lambda r: (r["source_id"], r["modality_mode"])  # noqa: E731
    assert sorted(map(identity, first)) == sorted(map(identity, reshuffled))


def test_a_small_split_is_used_whole():
    rows = _rows()[:120]
    assert validation_sample(rows, 400) == rows


def _report(f1s):
    report = SelectionReport(scores=[CheckpointScore(f"/ck/checkpoint-{s}", s, {"field_normalized_match": f})
                                     for s, f in f1s])
    report.selected = max(report.scores, key=lambda c: (c.field_f1, c.step)).checkpoint
    return report


def test_a_near_tie_is_decided_on_the_full_validation_split():
    report = _report([(280, 0.610), (560, 0.620), (600, 0.500)])      # margin 0.010 < 0.015
    full = {"/ck/checkpoint-280": 0.64, "/ck/checkpoint-560": 0.61}
    break_tie(report, lambda ck: {"field_normalized_match": full[ck]}, 0.015)
    assert report.selected == "/ck/checkpoint-280"
    assert report.tie_break["changed_choice"] is True
    assert set(report.tie_break["full_validation_field_f1"]) == set(full)   # only the top two


def test_a_clear_winner_on_the_sample_stands():
    report = _report([(280, 0.55), (560, 0.62)])                      # margin 0.07
    calls = []
    break_tie(report, lambda ck: calls.append(ck) or {}, 0.015)
    assert report.selected == "/ck/checkpoint-560" and calls == [] and report.tie_break is None


def test_staging_narrows_only_the_validation_rows(tmp_path):
    import json

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from training.stage_data import stage_training_data

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "{}"}],
           "task": "extract", "doc_type": "policy"}
    client.write_text("corpus/e1.jsonl", "\n".join(json.dumps(row) for _ in range(5)))
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(row) for _ in range(5)))
    staged = stage_training_data(["corpus/e1.jsonl"], "corpus/val.jsonl", client, tmp_path,
                                 val_filter=lambda rows: rows[:2])
    assert staged.val_rows == 2
    assert len((tmp_path / "train" / "epoch_1.jsonl").read_text().splitlines()) == 5


def test_checkpoint_selection_ranks_on_the_sample_and_keeps_the_full_split_for_ties():
    import json

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import checkpoint_eval as C

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = _rows()[:900]
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    seen = []
    original = C.generation_scorer

    def recording(rows_, model):
        seen.append(len(rows_))
        return original(rows_, model)

    C.generation_scorer = recording
    try:
        scorer = C.vllm_scorer(client=client, val_path="corpus/val.jsonl", model=object(), sample_rows=400)
    finally:
        C.generation_scorer = original
    assert seen == [400, 900] and callable(scorer.full)


def _document_metrics(accuracy, invented=0.10, empty=0.40):
    return {"field_normalized_match": accuracy, "document_field_normalized_match": accuracy,
            "document_hallucination_rate": invented, "document_false_null_rate": empty}


def test_the_finalists_are_judged_on_whole_documents():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60), (600, 0.55), (640, 0.40)])
    full = {"/ck/checkpoint-280": _document_metrics(0.50), "/ck/checkpoint-560": _document_metrics(0.58),
            "/ck/checkpoint-600": _document_metrics(0.52)}
    asked = []
    choose_on_documents(report, lambda ck: asked.append(ck) or full[ck], finalists=3, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-560" and report.document_choice["changed_choice"] is True
    assert sorted(asked) == sorted(full)                         # the top three only


def test_a_finalist_that_invents_more_does_not_win_on_accuracy():
    """Within the guard margin on field match, the fewest invented values win
    (decision 2026-10-24); invention is the ranking there, not a guard."""
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60)])
    full = {"/ck/checkpoint-280": _document_metrics(0.57, invented=0.10),
            "/ck/checkpoint-560": _document_metrics(0.58, invented=0.16)}     # 6 points more invented
    choose_on_documents(report, lambda ck: full[ck], finalists=3, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-280"
    assert report.document_choice["excluded_by_guard"] == []


def test_a_sample_that_is_the_whole_split_is_judged_without_scoring_again():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60)])
    report.scores[0].metrics.update(_document_metrics(0.50))
    report.scores[1].metrics.update(_document_metrics(0.58))
    choose_on_documents(report, None, finalists=2, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-560"


def test_one_candidate_needs_no_document_choice():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62)])
    choose_on_documents(report, lambda ck: 1 / 0, finalists=3, guard_margin=0.02)
    assert report.selected == "/ck/checkpoint-280" and report.document_choice is None


def test_the_in_training_loss_reads_its_own_smaller_sample():
    """The loss only draws the progress curve, so its checks read loss_sample_rows;
    selection keeps validation_sample_rows. Unset, the loss reads the selection sample."""
    from training.train import loss_sample_rows

    assert loss_sample_rows({"validation_sample_rows": 400, "loss_sample_rows": 160}) == 160
    assert loss_sample_rows({"validation_sample_rows": 400}) == 400
    assert loss_sample_rows({}) == 0
    sample = validation_sample(_rows(), 160)
    assert len(sample) == 160
    assert {r["modality_mode"] for r in sample} == {"ocr_plus_image", "noisy_ocr_image", "image_only"}


def test_the_configured_loss_sample_is_no_larger_than_the_selection_sample():
    from common.config import training_config

    evaluation = training_config("unified")["evaluation"]
    assert 0 < evaluation["loss_sample_rows"] <= evaluation["validation_sample_rows"]


def test_the_finalists_whole_document_pass_reads_only_the_named_modes():
    """selection_document_modes narrows .full by reading mode, never by row: every
    document is scored whole in that mode, while the sample keeps all three."""
    import json

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import checkpoint_eval as C

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = _rows()[:900]                                   # 300 documents x 3 modes
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    seen = []
    original = C.generation_scorer

    def recording(rows_, model):
        seen.append(rows_)
        return original(rows_, model)

    C.generation_scorer = recording
    try:
        scorer = C.vllm_scorer(client=client, val_path="corpus/val.jsonl", model=object(), sample_rows=60,
                               full_modes=["ocr_plus_image"])
    finally:
        C.generation_scorer = original
    sample, full = seen
    assert callable(scorer.full)
    assert {r["modality_mode"] for r in sample} == {"ocr_plus_image", "noisy_ocr_image", "image_only"}
    assert {r["modality_mode"] for r in full} == {"ocr_plus_image"}
    assert sorted(r["source_id"] for r in full) == sorted({r["source_id"] for r in rows})


def test_a_mode_filter_that_matches_no_row_is_refused():
    from evaluation.checkpoint_eval import CheckpointEvalError, rows_in_modes

    rows = _rows()[:9]
    assert rows_in_modes(rows, None) == rows and rows_in_modes(rows, []) == rows
    with pytest.raises(CheckpointEvalError, match="no validation row"):
        rows_in_modes([r for r in rows if r["modality_mode"] != "image_only"], ["image_only"])


def test_the_document_choice_records_the_modes_it_read():
    from evaluation.checkpoint_eval import choose_on_documents

    full = {"/ck/checkpoint-280": _document_metrics(0.55), "/ck/checkpoint-560": _document_metrics(0.58)}
    report = _report([(280, 0.62), (560, 0.60)])
    choose_on_documents(report, lambda ck: full[ck], finalists=2, guard_margin=0.02, modes=["ocr_plus_image"])
    assert report.document_choice["modes"] == ["ocr_plus_image"]
    report = _report([(280, 0.62), (560, 0.60)])
    choose_on_documents(report, lambda ck: full[ck], finalists=2, guard_margin=0.02)
    assert report.document_choice["modes"] == "all"


def test_an_unknown_selection_mode_is_refused(monkeypatch):
    import copy
    from types import SimpleNamespace

    import common.config as config
    from orchestration import pipeline_dag

    real = config.training_config

    def with_fax(name):
        cfg = copy.deepcopy(real(name))            # never the cached config itself
        cfg["evaluation"]["selection_document_modes"] = ["fax"]
        return cfg

    monkeypatch.setattr(config, "training_config", with_fax)
    with pytest.raises(pipeline_dag.PipelineError, match="fax"):
        pipeline_dag._selection_settings(SimpleNamespace())


@pytest.mark.format_profile("standard")
def test_the_configured_selection_judges_two_finalists_in_valid_modes():
    from common.config import training_config
    from common.constants import MODALITY_MODES

    evaluation = training_config("unified")["evaluation"]
    assert evaluation["selection_document_finalists"] == 2
    assert evaluation["selection_document_modes"] == ["ocr_plus_image"]
    assert set(evaluation["selection_document_modes"]) <= set(MODALITY_MODES)


def test_a_split_between_one_and_three_samples_still_gets_its_whole_document_pass():
    """900 rows, 300 per mode, sample 400: the sample is not the whole split, so the
    finalists are still judged on whole documents - the 300 rows of the main mode."""
    import json

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import checkpoint_eval as C

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = _rows()[:900]
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    seen = []
    original = C.generation_scorer

    def recording(rows_, model):
        seen.append(rows_)
        return original(rows_, model)

    C.generation_scorer = recording
    try:
        scorer = C.vllm_scorer(client=client, val_path="corpus/val.jsonl", model=object(), sample_rows=400,
                               full_modes=["ocr_plus_image"])
    finally:
        C.generation_scorer = original
    assert callable(scorer.full)
    assert [len(r) for r in seen] == [400, 300]
    assert {r["modality_mode"] for r in seen[1]} == {"ocr_plus_image"}


def test_a_mode_filter_that_matches_nothing_is_refused_before_an_engine_loads(monkeypatch):
    import json

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import checkpoint_eval as C
    from inference_core import model_runner

    loaded = []
    monkeypatch.setattr(model_runner, "load_model", lambda *a, **k: loaded.append(True))
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = [r for r in _rows()[:900] if r["modality_mode"] != "image_only"]
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    with pytest.raises(C.CheckpointEvalError, match="no validation row"):
        C.vllm_scorer(client=client, val_path="corpus/val.jsonl", sample_rows=400, full_modes=["image_only"])
    assert loaded == []                                   # no engine left holding the card


def test_a_decision_made_on_the_sample_says_so():
    from evaluation.checkpoint_eval import choose_on_documents

    report = _report([(280, 0.62), (560, 0.60)])
    report.scores[0].metrics.update(_document_metrics(0.50))
    report.scores[1].metrics.update(_document_metrics(0.58))
    choose_on_documents(report, None, finalists=2, guard_margin=0.02, modes=["ocr_plus_image"])
    assert report.document_choice["modes"] == "sample (all modes)"
