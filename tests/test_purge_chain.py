"""A purge names every copy of weights that learned the document (decision 2026-10-23).

A continued run carries everything its parents learned, so the chain of
continued runs is followed to a run read as clean. One whose chain never gets
there is listed to check by hand, with every run continued from it: a parent
no longer in the registry (its row removed once a purge reached it, or lost),
another tenant's run, a chain that loops. Dropping it reads absent evidence as
a clean model.

The pod's volume keeps copies of the weights outside Blob too: the parent
adapters staged for continued training, and the adapters a serving cold start
copies. Both plans name them, and each run names the run it continued and the
folder it started from.
"""

from __future__ import annotations

import glob
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ingestion import pull_raw_pdfs as ingest

#: Why a run whose parent cannot be read is checked by hand.
MISSING = "whose manifest is not in the registry (it may have learned the document)"
LOOPS = "whose chain of parents loops back on itself, so it never reaches a run read as clean"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _plan(client) -> dict:
    """The purge plan's runs for policy_0001 of broker_a, by run id."""
    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    return {entry["run_id"]: entry for entry in plan["trained_weights"]}


def _purge_runs(client, *runs: tuple[str, str, str | None], corpora: dict[str, list[str]],
                tenant: str = "broker_a") -> dict:
    """``(run_id, corpus, parent_run_id)`` runs of ``tenant``, written in the
    order given, and each corpus's train split; returns the purge plan's runs
    for policy_0001 of broker_a. As tests/test_incremental_training's helper."""
    from common.run_ids import version_of
    from registry_utils.write_run_manifest import write_manifest
    from tests.test_registry import _deps, _scoped

    for corpus, ids in corpora.items():
        client.write_json(paths.corpus_manifest(corpus, tenant), {"source_ids_by_split": {"train": ids}})
    for run_id, corpus, parent in runs:
        version = version_of(run_id)
        manifest = _scoped(
            run_id, "policy", ["policy"], tenant_id=tenant, parent_run_id=parent,
            continued_from=paths.staging_parent_adapter_dir("policy", version) if parent else None,
            dependencies=_deps(corpus_version=f"corpus/{corpus}"))
        manifest.artifacts.adapter_weights = paths.scoped_adapter_dir("policy", version)
        write_manifest(manifest, client)
    return _plan(client)


def _drop_rows(client, *run_ids: str) -> None:
    """Remove the runs' registry rows, as an operator does once a purge reached them."""
    index = client.read_json(paths.registry_index())
    index["runs"] = [row for row in index["runs"] if row["run_id"] not in run_ids]
    client.write_json(paths.registry_index(), index)


# --------------------------------------------------------------------------
# A parent the plan cannot read
# --------------------------------------------------------------------------


def test_a_run_continued_from_a_parent_no_longer_in_the_registry_is_checked_by_hand(client):
    """policy-v1 trained on the document and policy-v2 continued it. Once the
    plan was applied to v1 and its row removed, a second plan returned nothing,
    though v2 still holds what v1 learned."""
    runs = _purge_runs(client, ("policy-v1", "v1", None), ("policy-v2", "v2", "policy-v1"),
                       corpora={"v1": ["policy_0001"], "v2": ["policy_0002"]})
    assert runs["policy-v2"]["action"] == "continued policy-v1, listed here: needs retrain or delete"

    _drop_rows(client, "policy-v1")
    runs = _plan(client)
    assert "policy-v1" not in runs
    assert runs["policy-v2"]["action"] == f"check by hand: continued policy-v1, {MISSING}"
    assert runs["policy-v2"]["adapter_weights"] == paths.scoped_adapter_dir("policy", "v2")


def test_a_parent_whose_row_was_lost_is_no_all_clear(client):
    """The registry never held policy-v1's row: the same, with nothing applied."""
    runs = _purge_runs(client, ("policy-v2", "v2", "policy-v1"), corpora={"v2": ["policy_0002"]})
    assert runs["policy-v2"]["action"] == f"check by hand: continued policy-v1, {MISSING}"


def test_the_runs_continued_from_one_with_a_missing_parent_are_checked_with_it(client):
    """Written newest first: v4 continued v3, which continued v2, whose parent
    v1 is gone. Each level down is checked by hand with the one above it."""
    _purge_runs(client, ("policy-v4", "v4", "policy-v3"), ("policy-v3", "v3", "policy-v2"),
                ("policy-v2", "v2", "policy-v1"), ("policy-v1", "v1", None),
                corpora={"v1": ["policy_0001"], "v2": ["policy_0002"], "v3": ["policy_0003"],
                         "v4": ["policy_0004"]})
    _drop_rows(client, "policy-v1")
    runs = _plan(client)
    assert runs["policy-v2"]["action"] == f"check by hand: continued policy-v1, {MISSING}"
    assert runs["policy-v3"]["action"] == "continued policy-v2, listed here: check by hand with it"
    assert runs["policy-v4"]["action"] == "continued policy-v3, listed here: check by hand with it"


def test_a_run_continued_from_a_clean_parent_is_still_left_out(client):
    """v1 was read and never saw the document, so neither v2 nor v3, continued
    down from it, did - though v5's parent, elsewhere in the registry, is gone."""
    runs = _purge_runs(
        client, ("policy-v3", "v3", "policy-v2"), ("policy-v2", "v2", "policy-v1"),
        ("policy-v1", "v1", None), ("policy-v5", "v5", "policy-v4"),
        corpora={"v1": ["policy_0009"], "v2": ["policy_0002"], "v3": ["policy_0003"],
                 "v5": ["policy_0005"]})
    assert set(runs) == {"policy-v5"}
    assert runs["policy-v5"]["action"] == f"check by hand: continued policy-v4, {MISSING}"


def test_a_run_continued_from_another_tenant_s_run_is_checked_by_hand(client):
    """The plan does not read another tenant's runs, so one of them is no clean
    parent: it may itself have continued one of this tenant's runs."""
    _purge_runs(client, ("policy-v1", "v1", None), corpora={"v1": ["policy_0001"]}, tenant="broker_b")
    runs = _purge_runs(client, ("policy-v2", "v2", "policy-v1"), corpora={"v2": ["policy_0002"]})
    assert "policy-v1" not in runs, "another tenant's run was listed"
    assert runs["policy-v2"]["action"] == (
        "check by hand: continued policy-v1, another tenant's run, whose lineage this plan does "
        "not follow (it may have learned the document)")


def test_a_per_type_adapter_whose_foundation_is_gone_is_checked_by_hand(client):
    """The same holds for an adapter on a Foundation: one no longer read is no
    clean one."""
    from registry_utils.models import RunManifest
    from registry_utils.write_run_manifest import write_manifest
    from tests.test_registry import _DS, _deps, _tc

    client.write_json(paths.corpus_manifest("v2", "broker_a"),
                      {"source_ids_by_split": {"train": ["policy_0003"]}})
    write_manifest(RunManifest(
        run_id="policy-adapter-v1", run_type="per_type_adapter", doc_type="policy",
        tenant_id="broker_a", training_config=_tc(), data_stats=_DS,
        dependencies=_deps(corpus_version="corpus/v2", foundation_version="extractor-v1"),
    ), client)
    runs = _plan(client)
    assert runs["policy-adapter-v1"]["action"] == f"check by hand: built on extractor-v1, {MISSING}"


# --------------------------------------------------------------------------
# A chain that loops
# --------------------------------------------------------------------------


@pytest.mark.parametrize("loop", [
    (("policy-v2", "v2", "policy-v3"), ("policy-v3", "v3", "policy-v2")),
    (("policy-v2", "v2", "policy-v2"),),
], ids=["two runs", "a run its own parent"])
def test_a_chain_that_loops_ends_and_is_checked_by_hand(client, loop):
    """No run in a loop is read as clean, and the plan must still end: the runs
    in it, and v4 continued from one, are checked by hand."""
    done: dict = {}

    def plan() -> None:
        try:
            done["runs"] = _purge_runs(
                client, *loop, ("policy-v4", "v4", "policy-v2"),
                corpora={"v2": ["policy_0002"], "v3": ["policy_0003"], "v4": ["policy_0004"]})
        except BaseException as exc:  # noqa: BLE001 - re-raised below, in the test's thread
            done["error"] = exc

    worker = threading.Thread(target=plan, daemon=True)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), "the purge plan hung on a chain that loops"
    if "error" in done:
        raise done["error"]
    runs = done["runs"]
    for run_id, _corpus, parent in loop:
        assert runs[run_id]["action"] == f"check by hand: continued {parent}, {LOOPS}"
    assert runs["policy-v4"]["action"] == f"check by hand: continued policy-v2, {LOOPS}"


# --------------------------------------------------------------------------
# The copies on the volume, and where each run started from
# --------------------------------------------------------------------------


def _reached(patterns) -> set[str]:
    return {Path(found).as_posix() for pattern in patterns
            for found in glob.glob(pattern + "**", recursive=True) if Path(found).is_file()}


def _serving_adapter_root(monkeypatch) -> Path:
    """Where serving.vllm_entrypoint.cold_start copies the releases' adapters by
    default, read from the cold start itself so the plan cannot drift from it."""
    from serving import vllm_entrypoint as entry

    class Reached(Exception):
        pass

    seen: dict = {}

    def load(plan, client, *, adapter_root):
        seen["root"] = Path(adapter_root)
        raise Reached

    monkeypatch.setattr(entry, "build_serving_plan",
                        lambda client, tenant_id=None: SimpleNamespace(served=[object()]))
    monkeypatch.setattr(entry, "assert_prompt_hash", lambda plan: None)
    monkeypatch.setattr(entry, "assert_window_plan_served", lambda plan: None)
    monkeypatch.setattr(entry, "load_release_runtimes", load)
    with pytest.raises(Reached):
        entry.cold_start("v3", BlobClient(backend=InMemoryBackend(), container="main",
                                          raw_container="raw"), model=object())
    return seen["root"]


def test_both_plans_name_the_staged_parents_and_the_serving_cache(tmp_path, monkeypatch):
    """The parent's adapter is copied to the volume before a continued cycle
    trains and cleared only after a successful package; a cold start serving a
    release as a LoRA copies its adapter there too. Neither plan named them."""
    from orchestration.pipeline_dag import staging_parent_adapter_dir

    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    staged = Path(staging_parent_adapter_dir("v2"), "adapter_model.safetensors")
    served = Path(_serving_adapter_root(monkeypatch), "v3", "adapter_model.safetensors")
    for weights in (staged, served):
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"weights")

    root = paths.staging_root()
    for plan in (ingest.purge_plan("policy_0001", "policy", "broker_a"),
                 ingest.tenant_purge_plan("broker_a")):
        volume = plan["pod_volume"]
        assert {staged.as_posix(), served.as_posix()} <= _reached(volume)
        assert f"{root}/parent-adapters/*/*/" in volume and f"{root}/serving-adapters/" in volume
        actions = plan["pod_volume_action"]
        parents = next(a for a in actions if a.startswith(f"{root}/parent-adapters/*/*/: "))
        assert "continued training" in parents and "continued_from" in parents
        assert "delete the folders of the runs under trained_weights" in parents
        cache = next(a for a in actions if a.startswith(f"{root}/serving-adapters/: "))
        assert "one folder per release id" in cache
        assert "delete the folders of the releases under trained_weights" in cache


def test_each_run_names_the_run_it_continued_and_where_it_started_from(client):
    runs = _purge_runs(client, ("policy-v1", "v1", None), ("policy-v2", "v2", "policy-v1"),
                       corpora={"v1": ["policy_0001"], "v2": ["policy_0002"]})
    tenant = {entry["run_id"]: entry for entry in
              ingest.tenant_purge_plan("broker_a", client)["trained_weights"]}
    for listed in (runs, tenant):
        assert listed["policy-v2"]["parent_run_id"] == "policy-v1"
        assert listed["policy-v2"]["continued_from"] == paths.staging_parent_adapter_dir("policy", "v2")
        assert listed["policy-v1"]["parent_run_id"] is None
        assert listed["policy-v1"]["continued_from"] is None
