"""scripts/score_on_training_docs.py: a release's score on documents it trained on, beside selection's.

The command runs the real row reading, image copying, generation and scoring; only the engine is a stub, so a wrong
field or metric name fails here rather than on the pod.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.checkpoint_eval import CheckpointScore, SelectionReport
from inference_core.input_builder import build_training_row
from inference_core.model_runner import EchoBackend, ModelRunnerError
from inference_core.model_runner import load_model as real_load_model
from scripts.score_on_training_docs import SHOWN, choose_documents, comparison, main, selected_metrics

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "golden" / "policy_0001.golden.json").read_text(encoding="utf-8"))
OCR = (FIXTURES / "ocr" / "policy_0001_page_1.md").read_text(encoding="utf-8")
LOB = "equipment_breakdown"


def _rows():
    lines = {"gl": ["policy_0001", "policy_0002", "policy_0003", "policy_0004"], "wc": ["policy_0005"], "boppr": ["policy_0006"]}
    return [{"source_id": source, "lob": line, "doc_type": "policy"} for line, sources in lines.items()
            for source in sources for _window in range(2)]


def test_documents_are_drawn_line_by_line_so_a_small_line_is_not_left_out():
    chosen = choose_documents(_rows(), 3)
    assert len(chosen) == 3 and {"policy_0005", "policy_0006"} <= set(chosen)


def test_the_draw_is_fixed_by_the_seed_and_asks_for_no_more_than_there_are():
    assert choose_documents(_rows(), 3, seed=1) == choose_documents(_rows(), 3, seed=1)
    assert len(choose_documents(_rows(), 99)) == 6


def test_a_row_whose_line_is_a_list_is_still_drawn():
    rows = [{"source_id": "a", "lob": ["gl"]}, {"source_id": "b", "lob": ["gl", "wc"]}]
    assert choose_documents(rows, 2) == ["a", "b"]


def test_the_table_shows_both_scores():
    text = comparison({"document_field_normalized_match": 0.9}, {"document_field_normalized_match": 0.7})
    assert "0.9000" in text and "0.7000" in text and "trained documents" in text


def test_the_selected_checkpoints_metrics_come_from_a_real_report_shape():
    report = SelectionReport(selected="/x/checkpoint-857", scores=[
        CheckpointScore("/x/checkpoint-688", 688, {"field_normalized_match": 0.5}),
        CheckpointScore("/x/checkpoint-857", 857, {"field_normalized_match": 0.7}),
    ], document_choice={"finalists": {"/x/checkpoint-857": {"document_field_normalized_match": 0.695}}})
    assert selected_metrics(report.as_dict()) == {"field_normalized_match": 0.7, "document_field_normalized_match": 0.695}
    assert selected_metrics({}) == {}


@pytest.fixture
def pod(tmp_path, monkeypatch):
    """A corpus and a selection report in a stand-in Blob, and an engine that answers with the gold."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    rows = []
    for number in range(1, 5):
        image = f"processed/default/policy/policy_{number:04d}/page_1.png"
        client.write_bytes(image, b"image")
        rows.append(build_training_row("policy", f"policy_{number:04d}", [image], [OCR], "ocr_plus_image",
                                       json.dumps(GOLDEN), lob=LOB, split="train"))
    client.write_text(paths.corpus_epoch_file("v3", 1), "".join(json.dumps(row) + "\n" for row in rows))
    report = SelectionReport(selected="c", scores=[CheckpointScore("c", 857, {"field_normalized_match": 0.7})])
    client.write_json(paths.checkpoint_selection("v3", scope="policy"), report.as_dict())

    answer = json.dumps(GOLDEN, separators=(",", ":"))
    monkeypatch.setattr("artifact_registry.blob_client.BlobClient", lambda: client)
    monkeypatch.setattr("inference_core.model_runner.load_model",
                        lambda tag, c, adapter=None: real_load_model(tag, c, adapter=adapter, backend_impl=EchoBackend(answer)))
    monkeypatch.setattr("artifact_registry.paths.staging_train_images_dir", lambda corpus, tenant=None: str(tmp_path / "images"))
    monkeypatch.setattr("orchestration.detach.detach_module_if_needed", lambda *_a, **_k: False)
    return client


def _adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    return adapter


def test_the_command_scores_the_chosen_documents_with_the_real_scorer(pod, tmp_path, capsys):
    out = tmp_path / "result.json"
    assert main(["--version", "v3", "--corpus-version", "v3", "--adapter", str(_adapter(tmp_path)), "--documents", "2", "--out", str(out)]) == 0

    result = json.loads(out.read_text(encoding="utf-8"))
    assert len(result["documents"]) == 2 and result["rows"] == 2
    # The stub answers with the gold, so a working chain scores it perfect.
    assert result["trained"]["field_normalized_match"] == 1.0
    assert result["trained"]["generation_failed_rate"] == 0.0
    assert result["selection"]["field_normalized_match"] == 0.7
    # Flat rows make no document_* scores, and this gold holds no list rows: every other name is the scorer's own.
    expected = {name for name in SHOWN if not name.startswith("document_") and "list_field" not in name}
    assert expected <= set(result["trained"])
    printed = capsys.readouterr().out
    assert "1.0000" in printed and "0.7000" in printed


def test_window_rows_are_scored_as_whole_documents_too(pod, tmp_path):
    rows = [json.loads(line) for line in pod.read_text(paths.corpus_epoch_file("v3", 1)).splitlines()]
    for row in rows:
        row.update(sections="decl", window_pages=[1], window_index=0, task="policy_declarations")
    pod.write_text(paths.corpus_epoch_file("v3", 1), "".join(json.dumps(row) + "\n" for row in rows))

    out = tmp_path / "result.json"
    assert main(["--version", "v3", "--corpus-version", "v3", "--adapter", str(_adapter(tmp_path)), "--documents", "2", "--out", str(out)]) == 0

    trained = json.loads(out.read_text(encoding="utf-8"))["trained"]
    assert {"document_field_normalized_match", "document_field_f1", "document_hallucination_rate"} <= set(trained)
    assert trained["generation_failed_rate"] == 0.0


def test_a_folder_with_no_adapter_is_refused_before_the_bare_base_answers(pod, tmp_path):
    empty = tmp_path / "not_an_adapter"
    empty.mkdir()
    with pytest.raises(ModelRunnerError, match="adapter_config.json"):
        main(["--version", "v3", "--corpus-version", "v3", "--adapter", str(empty)])


def test_a_corpus_with_no_policy_rows_says_so(pod, tmp_path, capsys):
    pod.write_text(paths.corpus_epoch_file("v9", 1), json.dumps({"source_id": "x", "doc_type": "other"}) + "\n")
    assert main(["--version", "v9", "--corpus-version", "v9", "--adapter", str(_adapter(tmp_path))]) == 1
    assert "no training rows" in capsys.readouterr().err


def test_with_no_corpus_named_it_reads_the_one_the_run_trained_on(pod, tmp_path, monkeypatch):
    monkeypatch.setattr("orchestration.label_noise.corpus_of", lambda client, version: "v3" if version == "v8" else "wrong")
    out = tmp_path / "result.json"
    assert main(["--version", "v8", "--adapter", str(_adapter(tmp_path)), "--documents", "1", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["rows"] == 1
