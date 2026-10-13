"""Checkpoint candidates scored at once, one worker and vLLM engine per GPU
(evaluation.checkpoint_eval.ParallelScorer, evaluation.checkpoint_score_worker)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import checkpoint_eval as C
from evaluation.checkpoint_eval import CheckpointScore, SelectionReport, break_tie, select_best

CHECKPOINTS = [f"/out/checkpoint-{s}" for s in (18, 24, 27, 30)]


class _Many:
    """A scorer that scores several at once, and records each call."""

    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def __call__(self, checkpoint):               # never used when score_many exists
        raise AssertionError("scored one at a time")

    def score_many(self, checkpoints):
        self.calls.append(list(checkpoints))
        return {c: self.scores[c] for c in checkpoints}


def test_selection_hands_every_candidate_to_a_scorer_that_scores_several_at_once():
    scores = {c: {"field_normalized_match": f} for c, f in zip(CHECKPOINTS, (0.50, 0.61, 0.66, 0.64), strict=True)}
    scores[CHECKPOINTS[1]] = RuntimeError("worker died")
    scorer = _Many(scores)
    report = select_best(CHECKPOINTS, scorer, best_loss=CHECKPOINTS[0])
    assert scorer.calls == [CHECKPOINTS]                                    # one call, all four
    assert report.selected == CHECKPOINTS[2]
    assert report.skipped == [(CHECKPOINTS[1], "RuntimeError: worker died")]  # recorded, not a zero


def test_a_near_tie_scores_both_finalists_in_one_call():
    report = SelectionReport(scores=[
        CheckpointScore("/out/checkpoint-27", 27, {"field_normalized_match": 0.660}),
        CheckpointScore("/out/checkpoint-30", 30, {"field_normalized_match": 0.655}),
    ], selected="/out/checkpoint-27")
    full = _Many({"/out/checkpoint-27": {"field_normalized_match": 0.61},
                  "/out/checkpoint-30": {"field_normalized_match": 0.63}})
    break_tie(report, full, 0.015)
    assert full.calls == [["/out/checkpoint-27", "/out/checkpoint-30"]]
    assert report.selected == "/out/checkpoint-30"


def test_a_split_no_bigger_than_the_sample_is_not_scored_again_for_a_tie():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = [{"source_id": f"d{i}", "lob": ["homeowners"], "modality_mode": "ocr_plus_image",
             "sections": "decl", "doc_type": "policy"} for i in range(338)]
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    scorer = C.vllm_scorer(client=client, val_path="corpus/val.jsonl", model=object(), sample_rows=400)
    assert getattr(scorer, "full", None) is None


class _Worker:
    """A finished worker: writes its scores (or nothing) when started."""

    def __init__(self, command, env, log_path, fail):
        self.command, self.env, self.returncode = command, env, (1 if fail else 0)
        out = Path(command[command.index("--out") + 1])
        shares = [command[i + 1] for i, a in enumerate(command) if a == "--checkpoint"]
        if fail:
            log_path.write_text("Traceback\nCUDA out of memory\n", encoding="utf-8")
        else:
            out.write_text(json.dumps({c: {"metrics": {"field_normalized_match": 0.5}} for c in shares}))

    def poll(self):
        return self.returncode


def test_candidates_are_dealt_to_the_gpus_each_worker_seeing_one(tmp_path, monkeypatch):
    started = []

    def spawn(command, env, log_path):
        worker = _Worker(command, env, log_path, fail=env["CUDA_VISIBLE_DEVICES"] == "3")
        started.append(worker)
        return worker

    monkeypatch.setattr(C, "_spawn", spawn)
    scorer = C.ParallelScorer(val_path="corpus/val.jsonl", images_root="/imgs", sample_rows=400,
                              gpus=["0", "1", "2", "3"], work_dir=tmp_path, poll_seconds=0)
    outcomes = scorer.score_many(CHECKPOINTS)
    by_gpu = {w.env["CUDA_VISIBLE_DEVICES"]: w for w in started}
    assert sorted(by_gpu) == ["0", "1", "2", "3"]
    assert all(w.env["FIDEON_NO_DETACH"] == "1" and "--full" not in w.command for w in started)
    assert [a for a in by_gpu["0"].command if a.startswith("/out/")] == [CHECKPOINTS[0]]
    assert outcomes[CHECKPOINTS[0]] == {"field_normalized_match": 0.5}
    failed = outcomes[CHECKPOINTS[3]]                                        # GPU 3's only
    assert isinstance(failed, C.CheckpointEvalError) and "CUDA out of memory" in str(failed)
    assert sum(isinstance(o, Exception) for o in outcomes.values()) == 1


def test_two_gpus_take_two_candidates_each(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(C, "_spawn", lambda c, e, p: started.append(_Worker(c, e, p, False)) or started[-1])
    scorer = C.ParallelScorer(val_path="v", images_root=None, sample_rows=0, gpus=["0", "1"],
                              work_dir=tmp_path, full_split=True, poll_seconds=0)
    outcomes = scorer.score_many(CHECKPOINTS)
    assert len(started) == 2 and all("--full" in w.command for w in started)
    assert all(sum(a == "--checkpoint" for a in w.command) == 2 for w in started)
    assert set(outcomes) == set(CHECKPOINTS)


def test_the_gpus_scored_on_are_the_ones_this_process_may_use(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert C.scoring_gpus() == ["2", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert C.scoring_gpus() == []


def test_the_parallel_scorer_fetches_the_pages_once_and_keeps_a_full_scorer_only_when_needed(monkeypatch,
                                                                                                tmp_path):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from training import stage_data

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = [{"source_id": f"d{i}", "lob": ["homeowners"], "modality_mode": "ocr_plus_image",
             "sections": "decl", "doc_type": "policy"} for i in range(900)]
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    fetched = []
    monkeypatch.setattr(stage_data, "localize_rows", lambda rows_, client_, root: fetched.append(root) or rows_)
    scorer = C.parallel_vllm_scorer(client=client, val_path="corpus/val.jsonl", images_root="/imgs",
                                    sample_rows=400, gpus=["0", "1"], work_dir=tmp_path)
    assert fetched == ["/imgs"] and scorer.full.full_split is True
    small = C.parallel_vllm_scorer(client=client, val_path="corpus/val.jsonl", images_root=None,
                                   sample_rows=1000, gpus=["0", "1"], work_dir=tmp_path)
    assert getattr(small, "full", None) is None


def test_a_worker_writes_each_checkpoints_score_or_why_it_has_none(monkeypatch, tmp_path):
    from evaluation import checkpoint_score_worker as W

    closed = []

    def scorer(checkpoint):
        if checkpoint.endswith("24"):
            raise ValueError("no answers")
        return {"field_normalized_match": 0.7}

    scorer.close = lambda: closed.append(True)
    scorer.full = lambda checkpoint: {"field_normalized_match": 0.9}
    monkeypatch.setattr(C, "vllm_scorer", lambda **kwargs: scorer)
    monkeypatch.setattr("artifact_registry.blob_client.BlobClient", lambda: object())   # no account here
    out = tmp_path / "scores.json"
    assert W.main(["--val-path", "v", "--checkpoint", "/out/checkpoint-18", "--checkpoint",
                   "/out/checkpoint-24", "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == {
        "/out/checkpoint-18": {"metrics": {"field_normalized_match": 0.7}},
        "/out/checkpoint-24": {"error": "ValueError: no answers"},
    }
    assert closed == [True]
    W.main(["--val-path", "v", "--checkpoint", "/out/checkpoint-18", "--out", str(out), "--full"])
    assert json.loads(out.read_text())["/out/checkpoint-18"]["metrics"]["field_normalized_match"] == 0.9


@pytest.mark.parametrize("gpus,parallel", [(["0", "1", "2", "3"], True), (["0"], False)])
def test_selection_scores_in_parallel_when_training_had_several_gpus(monkeypatch, gpus, parallel):
    from common.scopes import get_scope
    from orchestration import pipeline_dag
    from training import train

    built = []
    monkeypatch.setattr(C, "scoring_gpus", lambda: gpus)
    monkeypatch.setattr(train, "training_gpus", lambda distributed: 4)
    monkeypatch.setattr(C, "parallel_vllm_scorer", lambda **kw: built.append(("parallel", kw)) or "P")
    monkeypatch.setattr(C, "vllm_scorer", lambda **kw: built.append(("single", kw)) or "S")
    ctx = SimpleNamespace(client=object(), corpus="v1", scope=get_scope("personal_lines"), tenant_id="smoke4x",
                          out_version="v1")
    assert pipeline_dag._checkpoint_scorer(ctx) == ("P" if parallel else "S")
    kind, kwargs = built[0]
    assert kind == ("parallel" if parallel else "single")
    assert kwargs["sample_rows"] == 400 and "smoke4x" in kwargs["val_path"]
    assert kwargs["full_modes"] == ["ocr_plus_image"]
    if parallel:
        assert kwargs["gpus"] == gpus and kwargs["work_dir"].name == "checkpoint_scores"


def test_the_full_pass_workers_are_told_which_modes_to_read(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(C, "_spawn", lambda c, e, p: started.append(_Worker(c, e, p, False)) or started[-1])
    full = C.ParallelScorer(val_path="v", images_root=None, sample_rows=400, gpus=["0", "1"], work_dir=tmp_path,
                            full_split=True, full_modes=["ocr_plus_image"], poll_seconds=0)
    full.score_many(CHECKPOINTS[:2])
    for worker in started:
        assert worker.command[worker.command.index("--full-mode") + 1] == "ocr_plus_image"
    started.clear()
    sample = C.ParallelScorer(val_path="v", images_root=None, sample_rows=400, gpus=["0"], work_dir=tmp_path,
                              full_modes=["ocr_plus_image"], poll_seconds=0)
    sample.score_many(CHECKPOINTS[:1])
    assert "--full-mode" not in started[0].command                   # the sample reads every mode


def test_a_worker_narrows_its_full_pass_to_the_modes_it_is_given(monkeypatch, tmp_path):
    from evaluation import checkpoint_score_worker as W

    asked = []

    def scorer(checkpoint):
        return {"field_normalized_match": 0.7}

    scorer.close = lambda: None
    monkeypatch.setattr(C, "vllm_scorer", lambda **kwargs: asked.append(kwargs) or scorer)
    monkeypatch.setattr("artifact_registry.blob_client.BlobClient", lambda: object())
    W.main(["--val-path", "v", "--checkpoint", "/out/checkpoint-18", "--out", str(tmp_path / "s.json"),
            "--full", "--full-mode", "ocr_plus_image"])
    assert asked[0]["full_modes"] == ["ocr_plus_image"]


def test_the_workers_report_what_their_passes_cost(monkeypatch, tmp_path):
    from evaluation import checkpoint_score_worker as W

    def scorer(checkpoint):
        scorer.passes.append({"checkpoint": checkpoint, "generated": 400, "seconds": 12.5})
        return {"field_normalized_match": 0.7}

    scorer.passes = []
    scorer.close = lambda: None
    monkeypatch.setattr(C, "vllm_scorer", lambda **kwargs: scorer)
    monkeypatch.setattr("artifact_registry.blob_client.BlobClient", lambda: object())
    out = tmp_path / "scores.json"
    W.main(["--val-path", "v", "--checkpoint", "/out/checkpoint-18", "--out", str(out)])
    assert json.loads(out.read_text())["/out/checkpoint-18"]["stats"] == {"generated": 400, "seconds": 12.5}

    def spawn(command, env, log_path):
        # A finished worker that reports its pass's cost with its score.
        worker = _Worker(command, env, log_path, False)
        out = Path(command[command.index("--out") + 1])
        shares = [command[i + 1] for i, a in enumerate(command) if a == "--checkpoint"]
        out.write_text(json.dumps({c: {"metrics": {"field_normalized_match": 0.5},
                                       "stats": {"generated": 400, "seconds": 9.0}} for c in shares}))
        return worker

    monkeypatch.setattr(C, "_spawn", spawn)
    parallel = C.ParallelScorer(val_path="v", images_root=None, sample_rows=400, gpus=["0"], work_dir=tmp_path,
                                poll_seconds=0)
    parallel.score_many(CHECKPOINTS[:1])
    cost = C.generation_cost(parallel)
    assert cost["rows"] == 400 and cost["seconds"] == 9.0 and cost["passes"][0]["gpu"] == "0"
