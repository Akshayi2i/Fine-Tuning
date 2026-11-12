"""scripts/archive_storage.py: the frozen eval set and some documents' labels moved aside, copied before deleted."""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from scripts import archive_storage
from scripts.archive_storage import apply, plan

ARCHIVE = "archive/test"


@pytest.fixture
def client():
    blob = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    eval_dir = paths.golden_eval_set_dir()
    blob.write_text(f"{eval_dir}/manifest.json", json.dumps({"frozen_from_corpus": "v1", "source_ids": ["policy_0003"]}))
    blob.write_text(f"{eval_dir}/policy/policy_0003/golden.json", "{}")
    blob.write_text(f"{paths.golden_eval_set_dir()}2/manifest.json", "{}")              # another tenant: default2
    for source_id, metadata in (
            ("policy_0010", {"imported_from": "gl__hybrid__hartford_cgl_renewal"}),
            ("policy_0011", {"imported_from": "later_folder",
                             "imported_from_history": [{"folder": "wc__scanned__hartford_wc_dec_1", "batch": None}]}),
            ("policy_0100", {"imported_from": "bopgl__hybrid__hartford_bopgl_policy_1"}),
            ("policy_01000", {"imported_from": "something_else"})):
        blob.write_text(paths.golden_label("policy", source_id), "{}")
        blob.write_text(paths.label_metadata("policy", source_id), json.dumps(metadata))
    return blob


def test_a_dry_plan_names_the_eval_set_and_the_documents_from_those_folders(client):
    moving = plan(client, eval_set=True, folders=["gl__hybrid__hartford_cgl_renewal", "wc__scanned__hartford_wc_dec_1"])
    assert len(moving.groups["frozen eval set"]) == 2                                    # not default2's
    labelled = [name for name in moving.groups if name.startswith("labels of")]
    assert sorted(name.split()[2] for name in labelled) == ["policy_0010", "policy_0011"]  # history counts too
    assert moving.unmatched == []
    assert client.exists(f"{paths.golden_eval_set_dir()}/manifest.json")                  # nothing moved


def test_apply_copies_checks_then_deletes_and_leaves_the_rest(client):
    moving = plan(client, eval_set=True, folders=["gl__hybrid__hartford_cgl_renewal"])
    keys = moving.keys()
    originals = {key: client.read_bytes(key) for key in keys}
    assert apply(client, moving, ARCHIVE) == len(keys) == 4
    for key, data in originals.items():
        assert not client.exists(key) and client.read_bytes(f"{ARCHIVE}/{key}") == data
    assert client.exists(f"{paths.golden_eval_set_dir()}2/manifest.json")                 # other tenant untouched
    assert client.exists(paths.golden_label("policy", "policy_0100"))                     # not a prefix match
    assert client.exists(paths.golden_label("policy", "policy_01000"))


def test_a_folder_no_document_came_from_moves_nothing(client, monkeypatch, capsys):
    monkeypatch.setattr("artifact_registry.blob_client.BlobClient", lambda *a, **k: client)
    code = archive_storage.main(["--eval-set", "--labels-from", "no_such_folder", "--apply"])
    assert code == 1 and "NOT FOUND" in capsys.readouterr().out
    assert client.exists(f"{paths.golden_eval_set_dir()}/manifest.json")


def test_without_apply_nothing_moves(client, monkeypatch, capsys):
    monkeypatch.setattr("artifact_registry.blob_client.BlobClient", lambda *a, **k: client)
    assert archive_storage.main(["--eval-set"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert client.exists(f"{paths.golden_eval_set_dir()}/manifest.json")
