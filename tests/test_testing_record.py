"""The testing record (testing/testing_record.py): one row per document an
extraction run read - round, date, split, carrier, line, document, pages, field
match, latency and tokens - written by the run and copied into the team's
workbook without overwriting anything typed there."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

openpyxl = pytest.importorskip("openpyxl")

from testing.testing_record import (  # noqa: E402
    COLUMNS,
    RecordError,
    append_csv,
    append_to_workbook,
    read_csv,
    record_row,
)

GOLD = {"carrier": {"name": {"raw": "Harborline Mutual", "parsed": "Harborline Mutual", "page_ref": [1]}}}


def _result(**overrides):
    stats = [{"prompt_tokens": 12_000, "answer_tokens": 900}, {"prompt_tokens": 11_500, "answer_tokens": 450},
             {"prompt_tokens": None, "answer_tokens": None}]           # a window that failed to answer
    return SimpleNamespace(**{"source_id": "policy_0007", "latency_ms": 41_250.0, "window_stats": stats,
                              "model_version": "base", "mode": "image_only", **overrides})


def test_a_row_carries_every_column_of_the_sheet():
    row = record_row(_result(), {"field_normalized_match_rate": 0.83333},
                     metadata={"split": "test", "lob": "gl"}, golden=GOLD, pages=24,
                     round_label="Baseline", date="2026-10-28")
    assert [name for name in row if name in COLUMNS] == list(COLUMNS)
    assert row == {**row, "Training Round": "Baseline", "Training/Testing Date": "2026-10-28", "Test/Val": "test",
                   "Carrier": "Harborline Mutual", "LOB": "gl", "Doc ID": "policy_0007", "# Pages": 24,
                   "F1 Score Accuracy": 0.8333, "Latency": 41.2,
                   "Input Tokens": 23_500, "Output Tokens": 1_350}     # summed over the windows read


def test_an_unscored_or_untimed_document_leaves_those_cells_empty():
    row = record_row(_result(latency_ms=None, window_stats=[]), {}, metadata={"carrier": "From Metadata"},
                     golden=None, pages=3, round_label="v1")
    assert row["F1 Score Accuracy"] is None and row["Latency"] is None
    assert row["Input Tokens"] is None and row["Output Tokens"] is None
    assert row["Carrier"] == "From Metadata"                 # no gold: the label metadata's record


def _workbook(path):
    book = openpyxl.Workbook()
    ws = book.active
    ws.title = "Testing Record "                             # the team's sheet name, as typed
    # The team's headers, some with trailing spaces as typed in the sheet.
    ws.append(["Training Round ", "Training/Testing Date", "Test/Val", "Carrier", "LOB", "Doc ID ", "# Pages ",
               "F1 Score Accuracy", "Latency ", "Input Tokens", "Output Tokens "])
    for _ in range(2):
        ws.append(["Baseline "])                             # rows pre-typed for the baseline
    ws.append(["Notes", None, None, None, None, None, None, None, None, None, None])
    ws["B4"] = "kept by hand"
    book.save(path)


def test_rows_fill_the_pre_typed_round_rows_then_new_ones_and_never_twice(tmp_path):
    xlsx = tmp_path / "record.xlsx"
    _workbook(xlsx)
    rows = [record_row(_result(source_id=f"policy_000{i}"), {"field_normalized_match_rate": 0.5},
                       metadata={"split": "val", "lob": "wc"}, golden=GOLD, pages=10,
                       round_label="Baseline", date="2026-10-28") for i in (1, 2, 3)]
    assert append_to_workbook(rows, xlsx) == {"written": 3, "skipped": 0}

    ws = openpyxl.load_workbook(xlsx).worksheets[0]
    assert [ws.cell(row=r, column=6).value for r in (2, 3, 5)] == ["policy_0001", "policy_0002", "policy_0003"]
    assert (ws["A4"].value, ws["B4"].value) == ("Notes", "kept by hand")     # a hand-typed row is left alone
    assert (ws["G2"].value, ws["H2"].value, ws["J2"].value) == (10, 0.5, 23_500)

    assert append_to_workbook(rows, xlsx) == {"written": 0, "skipped": 3}    # running it again adds nothing


def test_a_sheet_missing_a_column_is_refused_by_name(tmp_path):
    xlsx = tmp_path / "record.xlsx"
    book = openpyxl.Workbook()
    book.active.title = "Testing Record"
    book.active.append(["Training Round", "Doc ID"])
    book.save(xlsx)
    with pytest.raises(RecordError, match="no column"):
        append_to_workbook([], xlsx)


def test_the_csv_round_trips_into_the_workbook_as_numbers(tmp_path):
    path = tmp_path / "test_record.csv"
    row = record_row(_result(), {"field_normalized_match_rate": 0.75}, metadata={"split": "test", "lob": "gl"},
                     golden=GOLD, pages=24, round_label="Baseline", date="2026-10-28")
    append_csv([row], path)
    append_csv([row], path)                                  # appended, one header
    with path.open(encoding="utf-8") as handle:
        assert sum(1 for line in csv.reader(handle) if line and line[0] == "Training Round") == 1
    xlsx = tmp_path / "record.xlsx"
    _workbook(xlsx)
    assert append_to_workbook(read_csv(path), xlsx) == {"written": 1, "skipped": 1}
    ws = openpyxl.load_workbook(xlsx).worksheets[0]
    assert (ws["G2"].value, ws["H2"].value, ws["I2"].value, ws["K2"].value) == (24, 0.75, 41.2, 1_350)


def test_the_extraction_run_writes_a_row_per_document(tmp_path):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import EchoBackend, load_model
    from testing.run_extraction import run_batch
    from tests.test_extract_with_adapter import ANSWER, _imported_document

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    _imported_document(client)
    model = load_model("base", client, backend_impl=EchoBackend(ANSWER))
    run_batch(model, client, ["policy_0001"], tenant_id="t1", images_root=tmp_path / "pages",
              root=tmp_path / "out", round_label="Baseline")
    rows = read_csv(tmp_path / "out" / "metrics" / model.tag / "test_record.csv")
    assert [(r["Training Round"], r["Doc ID"], r["LOB"], r["# Pages"]) for r in rows] == [
        ("Baseline", "policy_0001", "homeowners", "2")]
    metrics = json.loads((tmp_path / "out" / "metrics" / model.tag / "policy_0001.metrics.json")
                         .read_text(encoding="utf-8"))
    assert float(rows[0]["F1 Score Accuracy"]) == round(metrics["field_normalized_match_rate"], 4)
