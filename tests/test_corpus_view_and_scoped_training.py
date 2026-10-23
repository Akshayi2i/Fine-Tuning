"""The policy adapter's training run: its corpus view, its budget, its manifest, its merge.

One adapter trains each cycle (decision 2026-10-22), and everything it must get
right is here:

* it reads a filtered **view** of the one corpus, not a corpus of its own -
  always written beside the corpus under ``train/scope/policy/``;
* it spends only the sequence budget its own tasks and types need;
* it records what it covers, under its own run id and staging path;
* its merge folds ITS adapter, not whichever one is staged at that version.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.adapter import ADAPTER
from registry_utils.models import DataStats
from training.corpus_view import CorpusViewError, materialize
from training.train import build_manifest, build_training_config, corpus_max_length


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


EPOCH_PATHS = [f"corpus/default/v1/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)]


def _row(source_id: str, doc_type: str, epoch: int | None = None) -> dict:
    row = {"source_id": source_id, "doc_type": doc_type, "modality_mode": "ocr_plus_image"}
    if doc_type == "policy":
        row["lob"] = "homeowners"       # the view trains single-line policies only
    if epoch is not None:
        row["epoch"] = epoch
    return row


def seed_corpus(client: BlobClient, version: str = "v1") -> None:
    """A corpus holding all three types, as the dataset build leaves it."""
    for epoch in (1, 2, 3, 4):
        rows = [
            _row("policy_0001", "policy", epoch),
            _row("policy_0002", "policy", epoch),
            _row("lossrun_0001", "lossrun", epoch),
            _row("acord_0001", "acord", epoch),
        ]
        client.write_text(
            paths.corpus_epoch_file(version, epoch),
            "".join(json.dumps(r) + "\n" for r in rows),
        )
    client.write_text(
        paths.corpus_eval_split(version, "val"),
        "".join(json.dumps(r) + "\n" for r in
                [_row("policy_0003", "policy"), _row("lossrun_0002", "lossrun")]),
    )


# --------------------------------------------------------------------------
# The corpus view
# --------------------------------------------------------------------------


def test_the_view_is_always_written_beside_the_corpus_as_the_policy_view(client):
    """There is no "the corpus already is the view" case any more: the adapter
    reads policies only, so its epoch and validation files are always written,
    under the policy path segment."""
    seed_corpus(client)
    view = materialize("v1", client)

    assert view.adapter == "policy"
    assert view.epoch_files == [paths.corpus_scope_epoch_file("v1", e, "policy") for e in (1, 2, 3, 4)]
    assert view.val_path == paths.corpus_scope_eval_split("v1", "val", "policy")
    assert view.epoch_files[0] != EPOCH_PATHS[0]
    assert all(client.exists(path) for path in [*view.epoch_files, view.val_path])


def test_the_view_drops_every_row_that_is_not_a_policy(client):
    """Filtered, never rebuilt. A separate BUILD would draw its own split, so a
    document could be train in one and test in another."""
    seed_corpus(client)
    view = materialize("v1", client)

    assert view.rows_by_epoch == {1: 2, 2: 2, 3: 2, 4: 2}   # two policies per epoch
    assert view.dropped_rows == 8                            # one lossrun + one acord x 4
    assert view.val_rows == 1

    for path in [*view.epoch_files, view.val_path]:
        written = [json.loads(line) for line in client.read_text(path).splitlines()]
        assert {r["doc_type"] for r in written} == {"policy"}


def test_the_view_trains_single_line_policies_only(client):
    """A package (several lines) and a policy with no line are read against
    _fallback.json, a shape the adapter is not taught, and serving sends them to
    the base model (ServedRelease.covers_lob): they do not train. A line is read
    by its schema name (workers_comp is wc)."""
    rows = [
        {**_row("policy_0001", "policy", 1), "lob": "homeowners"},
        {**_row("policy_0002", "policy", 1), "lob": ["homeowners", "personal_auto"]},
        {**_row("policy_0003", "policy", 1), "lob": ["auto", "wc"]},
        {**_row("policy_0004", "policy", 1), "lob": None},
        {**_row("policy_0005", "policy", 1), "lob": "workers_comp"},
        {**_row("policy_0006", "policy", 1), "lob": "not_a_line"},
    ]
    for epoch in (1, 2, 3, 4):
        client.write_text(paths.corpus_epoch_file("v1", epoch),
                          "".join(json.dumps({**r, "epoch": epoch}) + "\n" for r in rows))
    client.write_text(paths.corpus_eval_split("v1", "val"), json.dumps(rows[0]) + "\n")
    view = materialize("v1", client)

    written = [json.loads(line) for line in client.read_text(view.epoch_files[0]).splitlines()]
    assert {r["source_id"] for r in written} == {"policy_0001", "policy_0005"}
    assert set(view.examples_by_line["train"]) == {"homeowners", "wc"}


def test_a_run_refuses_a_view_whose_documents_read_two_output_shapes():
    """Judged on the documents the view trains, each by the schema it is read
    against: homeowners and a package (_fallback.json) are one shape, workers'
    comp alone is another."""
    from common.adapter import AdapterError, assert_one_output_shape
    from training.corpus_view import CorpusView, trained_lobs

    view = CorpusView(examples_by_line={"train": {"homeowners": {}, "auto,wc": {}, "unknown": {}}})
    assert trained_lobs(view) == ["homeowners", ["auto", "wc"], None]
    assert_one_output_shape(trained_lobs(view))
    view.examples_by_line["train"]["wc"] = {}
    with pytest.raises(AdapterError, match="two output shapes"):
        assert_one_output_shape(trained_lobs(view))


def test_the_view_keeps_every_epoch_so_three_epochs_still_means_three_passes(client):
    seed_corpus(client)
    view = materialize("v1", client)

    per_epoch = [
        sorted(json.loads(line)["source_id"] for line in client.read_text(path).splitlines())
        for path in view.epoch_files
    ]
    assert len(per_epoch) == 4
    assert all(epoch == per_epoch[0] for epoch in per_epoch), "an epoch lost a document"


def test_a_corpus_with_no_policy_rows_is_refused(client):
    """Training on nothing produces an adapter that reports success and learned
    nothing, and every downstream metric would describe the base model."""
    for epoch in (1, 2, 3, 4):
        client.write_text(
            paths.corpus_epoch_file("v1", epoch),
            json.dumps(_row("acord_0001", "acord", epoch)) + "\n",
        )
    client.write_text(paths.corpus_eval_split("v1", "val"), "")

    with pytest.raises(CorpusViewError, match="no training rows"):
        materialize("v1", client)


def test_the_view_cannot_invent_a_corpus_it_was_never_given(client):
    with pytest.raises(CorpusViewError, match="does not build one of its own"):
        materialize("v9", client)


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_the_adapter_spends_the_sequence_budget_its_own_tasks_need():
    """The cap decides which card a run fits on. The policy extraction override
    (32768) is the adapter's own: its tasks and its document type."""
    assert corpus_max_length() == 32768


def test_the_trainer_config_takes_its_cap_from_the_adapter():
    swift, _recorded = build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert swift.args["max_length"] == 32768


# --------------------------------------------------------------------------
# What the run records
# --------------------------------------------------------------------------


def _manifest():
    _swift, recorded = build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")
    return build_manifest(
        run_id=ADAPTER.run_id("v2"),
        corpus_version="v2",
        corpus_manifest={},
        training_cfg=recorded,
        data_stats=DataStats(train_examples=10, val_examples=2, test_examples=2),
        staging_path=paths.scoped_staging_adapter_dir(ADAPTER.name, "v2"),
    )


def test_the_run_records_the_policy_adapter_and_its_coverage():
    """Filed where the policy runs always were, so an older policy run and a
    new one read alike."""
    manifest = _manifest()

    assert manifest.run_id == "policy-v2"
    assert manifest.run_type == "scoped"
    assert manifest.scope == "policy" and manifest.doc_types == ["policy"]
    assert manifest.artifacts.staging_path.endswith("adapters/scope/policy/v2")


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def test_the_merge_folds_the_adapter_of_its_own_scope():
    """Two runs can share a version, so an unscoped merge would fold whichever
    adapter happened to be staged at that tag."""
    from training.merge import plan_merge

    policy = plan_merge(base_model="base@rev", version="v2", scope="policy")
    unified = plan_merge(base_model="base@rev", version="v2")

    assert policy.adapter.endswith("adapters/scope/policy/v2")
    assert policy.output_dir.endswith("merged-models/scope/policy/v2")
    assert unified.adapter.endswith("adapters/foundation/v2")
    assert unified.output_dir.endswith("merged-models/unified/v2")


def test_the_selected_checkpoint_still_wins_over_the_scope_directory():
    """§11.2 selection picks a checkpoint inside the staged run; the scope only
    decides which run that is."""
    from training.merge import plan_merge

    plan = plan_merge(
        base_model="base@rev", version="v2", scope="policy",
        selected_checkpoint="/staging/adapters/scope/policy/v2/v0-x/checkpoint-40",
    )
    assert plan.adapter.endswith("checkpoint-40")
    assert plan.output_dir.endswith("merged-models/scope/policy/v2")
