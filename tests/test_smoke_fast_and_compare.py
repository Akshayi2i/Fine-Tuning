"""The smoke run's cut-down checkpoint selection and its base-against-adapter look."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from orchestration import pipeline_dag, smoke_run


def test_a_smoke_run_cuts_checkpoint_selection_down(monkeypatch):
    monkeypatch.setenv(pipeline_dag.FAST_SELECTION_ENV, "1")
    settings = pipeline_dag._selection_settings(SimpleNamespace())
    assert settings["selection_document_finalists"] == 0
    assert settings["selection_run_to_run_check"] is False
    assert settings["selection_two_thirds_candidate"] is False
    assert settings["selection_repeat_modes"] == []
    assert 0 < settings["validation_sample_rows"] <= pipeline_dag.FAST_SELECTION_ROWS


def test_a_real_cycle_keeps_the_whole_selection(monkeypatch):
    monkeypatch.delenv(pipeline_dag.FAST_SELECTION_ENV, raising=False)
    settings = pipeline_dag._selection_settings(SimpleNamespace())
    assert settings["selection_document_finalists"] >= 2
    assert settings["selection_run_to_run_check"] is True
    assert settings["validation_sample_rows"] > pipeline_dag.FAST_SELECTION_ROWS


def test_only_the_smoke_runs_finetune_step_asks_for_it():
    source = Path(smoke_run.__file__).read_text(encoding="utf-8")
    assert 'step == "finetune" and not args.full_selection' in source
    args, steps = smoke_run.parse_args(["--batch", "b"])
    assert args.full_selection is False and steps[-1] == "compare"
    assert smoke_run.parse_args(["--batch", "b", "--full-selection"])[0].full_selection is True


def test_the_last_checkpoint_of_the_newest_run_is_the_one_compared(tmp_path):
    root = tmp_path / "staging" / "adapters" / "scope" / "policy" / "v0.1"
    for run, steps in (("v0-20261028-100000", (2, 4)), ("v0-20261028-140700", (2, 8, 18))):
        for step in steps:
            (root / run / f"checkpoint-{step}").mkdir(parents=True)
    assert smoke_run.latest_checkpoint(tmp_path, "v0.1") == root / "v0-20261028-140700" / "checkpoint-18"
    assert smoke_run.latest_checkpoint(tmp_path, "v9") is None


def test_the_scores_are_read_from_the_summary_the_run_names(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"mean_field_normalized_match": 0.43, "mean_list_field_recall": 0.7}),
                       encoding="utf-8")
    output = f"INFO something\nINFO testing.run_extraction summary -> {summary}\n{{}}\n"
    assert smoke_run.read_scores(output) == {"field_match": 0.43, "list_recall": 0.7}
    assert smoke_run.read_scores("no summary here") is None


def test_the_compare_step_never_fails_the_run_when_nothing_is_there(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke_run, "WORKSPACE", tmp_path)
    smoke_run.compare_with_base(SimpleNamespace(version="v0.1", tenant="smoke"))
    assert "nothing to compare" in capsys.readouterr().out
