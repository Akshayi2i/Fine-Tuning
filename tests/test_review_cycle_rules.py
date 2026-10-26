"""Review 2026-10-26 #cycle: when a cycle starts from the base instead of continuing.

``continued_training`` in orchestration/config/pipeline.yaml, every rule off by
default (``pipeline_dag.fresh_by_rule``, asked at the end of ``continuation``):
fresh when the new documents exceed a share, every N cycles, or when a gold the
chain trained on changed - compared by the gold hashes each corpus now records.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import pipeline_dag, settings
from orchestration.runpod_controller import LocalBackend, RunPodController

OFF = dict(settings.CONTINUED_TRAINING_DEFAULTS)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _ctx(client: BlobClient) -> SimpleNamespace:
    return SimpleNamespace(client=client, tenant_id=None, doc_types=["policy"])


def _golds(client: BlobClient, n: int) -> list[str]:
    ids = []
    for i in range(n):
        source_id = f"policy_{i:04d}"
        client.write_json(paths.golden_label("policy", source_id), {"policy": {"policy_number": f"P-{i}"}})
        ids.append(source_id)
    return ids


def _corpus(client: BlobClient, version: str, train: list[str], *, hashes: bool = True) -> dict:
    body = {"corpus_version": version, "source_ids_by_split": {"train": train, "val": [], "test": []}}
    if hashes:
        body["gold_sha256"] = {s: pipeline_dag.gold_hash(client.read_json(paths.golden_label("policy", s)))
                               for s in train}
    return body


def _rules(monkeypatch, **on) -> None:
    monkeypatch.setattr(settings, "continued_training_rules", lambda config=None: {**OFF, **on})


def _chain(monkeypatch, chain: list) -> None:
    monkeypatch.setattr(pipeline_dag, "_chain_corpora", lambda ctx, run_id: chain)


# --------------------------------------------------------------------------
# The config
# --------------------------------------------------------------------------


def test_every_rule_ships_off():
    assert settings.continued_training_rules() == OFF


@pytest.mark.parametrize("block", [
    {"fresh_when_new_share_above": 40},          # meant 40%: would never fire
    {"fresh_when_new_share_above": "0.4"},
    {"fresh_every_n_cycles": 0},
    {"fresh_every_n_cycles": 2.5},
    {"fresh_when_golds_changed": "yes"},
    {"fresh_every_cycles": 4},                   # a key spelt otherwise: silently unused
])
def test_a_rule_that_would_not_do_what_it_says_is_refused(block):
    with pytest.raises(settings.SettingsError):
        settings.continued_training_rules({"continued_training": block})


def test_set_rules_are_read():
    rules = settings.continued_training_rules({"continued_training": {
        "fresh_when_new_share_above": 0.4, "fresh_every_n_cycles": 4, "fresh_when_golds_changed": True}})
    assert rules == {"fresh_when_new_share_above": 0.4, "fresh_every_n_cycles": 4, "fresh_when_golds_changed": True}


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


def test_with_every_rule_off_nothing_is_read(client, monkeypatch):
    _rules(monkeypatch)
    monkeypatch.setattr(pipeline_dag, "_chain_corpora", lambda *a: pytest.fail("read the chain"))
    assert pipeline_dag.fresh_by_rule(_ctx(client), "policy-v3", set()) is None


def test_every_nth_cycle_starts_from_the_base(client, monkeypatch):
    chain = [("policy-v3", {}), ("policy-v2", {}), ("policy-v1", {})]
    _chain(monkeypatch, chain)
    _rules(monkeypatch, fresh_every_n_cycles=4)
    assert pipeline_dag.fresh_by_rule(_ctx(client), "policy-v3", set()) is None      # v4 continues
    _rules(monkeypatch, fresh_every_n_cycles=3)
    why = pipeline_dag.fresh_by_rule(_ctx(client), "policy-v3", set())
    assert "fresh_every_n_cycles is 3" in why and "policy-v1 from the base" in why


def test_a_cycle_mostly_new_documents_starts_from_the_base(client, monkeypatch):
    ids = _golds(client, 10)
    _chain(monkeypatch, [("policy-v1", {})])
    trained = set(ids[:5])
    _rules(monkeypatch, fresh_when_new_share_above=0.6)
    assert pipeline_dag.fresh_by_rule(_ctx(client), "policy-v1", trained) is None
    _rules(monkeypatch, fresh_when_new_share_above=0.4)
    why = pipeline_dag.fresh_by_rule(_ctx(client), "policy-v1", trained)
    assert "5 of the 10 documents" in why and "50%" in why


def test_the_frozen_eval_set_is_not_counted_new(client, monkeypatch):
    from evaluation import freeze_eval_set

    ids = _golds(client, 10)
    _chain(monkeypatch, [("policy-v1", {})])
    monkeypatch.setattr(freeze_eval_set, "is_frozen", lambda c, tenant_id=None: True)
    monkeypatch.setattr(freeze_eval_set, "frozen_manifest", lambda c, tenant_id=None: {"source_ids": ids[5:9]})
    _rules(monkeypatch, fresh_when_new_share_above=0.4)
    # 6 documents can train, 1 of them new.
    assert pipeline_dag.fresh_by_rule(_ctx(client), "policy-v1", set(ids[:5])) is None


def test_a_changed_gold_the_chain_trained_on_starts_from_the_base(client, monkeypatch):
    ids = _golds(client, 4)
    _chain(monkeypatch, [("policy-v2", _corpus(client, "v2", ids[:3])), ("policy-v1", _corpus(client, "v1", ids[:2]))])
    _rules(monkeypatch, fresh_when_golds_changed=True)
    ctx = _ctx(client)
    assert pipeline_dag.fresh_by_rule(ctx, "policy-v2", set(ids[:3])) is None

    # A gold not trained on changes: nothing to do with the chain.
    client.write_json(paths.golden_label("policy", ids[3]), {"policy": {"policy_number": "changed"}})
    assert pipeline_dag.fresh_by_rule(ctx, "policy-v2", set(ids[:3])) is None

    client.write_json(paths.golden_label("policy", ids[1]), {"policy": {"policy_number": "corrected"}})
    why = pipeline_dag.fresh_by_rule(ctx, "policy-v2", set(ids[:3]))
    assert "1 gold(s)" in why and ids[1] in why and "fresh_when_golds_changed" in why


def test_a_deleted_gold_the_chain_trained_on_starts_from_the_base(client, monkeypatch):
    ids = _golds(client, 2)
    _chain(monkeypatch, [("policy-v1", _corpus(client, "v1", ids))])
    _rules(monkeypatch, fresh_when_golds_changed=True)
    client.delete(paths.golden_label("policy", ids[0]))
    assert ids[0] in pipeline_dag.fresh_by_rule(_ctx(client), "policy-v1", set(ids))


def test_a_chain_corpus_without_gold_hashes_cannot_show_a_change_and_goes_fresh(client, monkeypatch):
    ids = _golds(client, 2)
    _chain(monkeypatch, [("policy-v2", _corpus(client, "v2", ids)), ("policy-v1", _corpus(client, "v1", ids, hashes=False))])
    _rules(monkeypatch, fresh_when_golds_changed=True)
    why = pipeline_dag.fresh_by_rule(_ctx(client), "policy-v2", set(ids))
    assert "policy-v1" in why and "records no gold hashes" in why


def test_the_corpus_build_records_each_kept_golds_hash(monkeypatch):
    from orchestration.pipeline_dag import run_stages, stages_for
    from tests.test_orchestration import make_context, seed_corpus, seeded_policies

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    controller = RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")
    policies = seeded_policies(seed_corpus(client))
    run_stages(make_context(client, controller), stages_for("finetune"), command="finetune")
    manifest = client.read_json(paths.corpus_manifest("v1"))
    kept = {s for split in manifest["source_ids_by_split"].values() for s in split}
    assert set(manifest["gold_sha256"]) == kept and kept <= set(policies)
    for source_id, digest in manifest["gold_sha256"].items():
        assert digest == pipeline_dag.gold_hash(client.read_json(paths.golden_label("policy", source_id)))


def test_the_corpus_manifest_carries_what_building_the_targets_did(monkeypatch):
    """The target counts reach the manifest (they were all zeros while unpassed)."""
    import data_pipeline.corpus_manifest as cm
    import data_pipeline.dataset_builder.build_jsonl as bj
    from orchestration.pipeline_dag import run_stages, stages_for
    from tests.test_orchestration import make_context, seed_corpus

    built, passed = [], []
    real_build, real_manifest = bj.build_corpus, cm.build_manifest

    def build(*args, **kwargs):
        result = real_build(*args, **kwargs)
        result.target_counts["verify.checked"] += 7          # a count the build made
        built.append(result)
        return result

    def manifest(**kwargs):
        passed.append(kwargs.get("target_counts"))
        return real_manifest(**kwargs)

    monkeypatch.setattr(bj, "build_corpus", build)
    monkeypatch.setattr(cm, "build_manifest", manifest)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    controller = RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")
    seed_corpus(client)
    run_stages(make_context(client, controller), stages_for("finetune"), command="finetune")

    assert passed == [dict(built[-1].target_counts)]
    counts = client.read_json(paths.corpus_manifest("v1"))["target_counts"]
    assert counts == cm._target_counts(passed[0]) and counts["verify.checked"] >= 7
