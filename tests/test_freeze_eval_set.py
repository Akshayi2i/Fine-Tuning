"""The test split becomes the frozen golden eval set, once, and stays out of training.

Before this nothing read the test split and nothing populated the eval set the
gate scores, so the gate had no documents and the test documents were held out
for nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits
from evaluation.freeze_eval_set import FreezeError, freeze_eval_set, is_frozen, manifest_key
from evaluation.golden_eval import load_golden_set
from evaluation.run_eval import eval_set_source_ids
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import make_context, seed_corpus


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def _built_corpus(client, controller):
    from orchestration.pipeline_dag import run_stages, stages_from

    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_from("dataset_build")[:1])
    assert report.ok, report.render()
    return ctx


def _test_ids(client, ctx):
    import json

    text = client.read_text(paths.corpus_eval_split(ctx.corpus, "test", None))
    return {json.loads(line)["source_id"] for line in text.splitlines() if line.strip()}


def test_freezing_copies_the_test_split_in_the_layout_the_gate_reads(client, controller):
    ctx = _built_corpus(client, controller)
    test_ids = _test_ids(client, ctx)
    if not test_ids:
        pytest.skip("the fixture corpus drew no test document")

    manifest = freeze_eval_set(client, ctx.corpus, allow_small=True)
    assert set(manifest["source_ids"]) == test_ids
    assert eval_set_source_ids(client) == test_ids
    documents = load_golden_set(client)
    assert {d.source_id for d in documents} == test_ids
    assert all(d.image_keys and d.golden for d in documents)
    assert client.exists(manifest_key())


def test_freezing_twice_is_refused(client, controller):
    ctx = _built_corpus(client, controller)
    if not _test_ids(client, ctx):
        pytest.skip("the fixture corpus drew no test document")
    freeze_eval_set(client, ctx.corpus, allow_small=True)
    with pytest.raises(FreezeError, match="already frozen"):
        freeze_eval_set(client, ctx.corpus, allow_small=True)


def test_a_corpus_without_a_test_split_cannot_be_frozen(client):
    with pytest.raises(FreezeError, match="no test split"):
        freeze_eval_set(client, "v404", allow_small=True)
    assert not is_frozen(client)


def test_a_frozen_document_and_its_family_stay_out_of_the_corpus(client):
    from orchestration.pipeline_dag import exclude_eval_families

    client.write_json(f"{paths.golden_eval_set_dir()}/eval-1/golden.json", {})

    def doc(source_id, family):
        return SimpleNamespace(source_id=source_id, family=family, carrier=None)

    kept, excluded = exclude_eval_families(client, [
        doc("eval-1", "fam-a"), doc("renewal-of-eval-1", "fam-a"), doc("other", "fam-b"),
    ])
    assert [d.source_id for d in kept] == ["other"]
    assert excluded == ["eval-1", "renewal-of-eval-1"]


def test_once_frozen_new_documents_split_into_train_and_val_only():
    groups = {"acord": [
        GroupRecord(group_id=f"g{i}", doc_type="acord", source_ids=[f"s{i}"],
                    carrier=f"carrier-{i % 5}")
        for i in range(60)
    ]}
    assignment = assign_group_splits(groups, with_test=False)
    counts = assignment.counts_by_doc_type["acord"]
    assert counts["test"] == 0
    assert counts["train"] > 0 and counts["val"] > 0
    assert not assignment.held_out_carriers
    ratio = assignment.ratios_by_doc_type["acord"]
    assert ratio["test"] == 0.0
    assert abs(ratio["train"] + ratio["val"] - 1.0) < 1e-9


def test_a_rebuild_after_freezing_trains_on_none_of_the_eval_set(client, controller):
    import json

    from orchestration.pipeline_dag import stage_dataset_build

    ctx = _built_corpus(client, controller)
    if not _test_ids(client, ctx):
        pytest.skip("the fixture corpus drew no test document")
    frozen = set(freeze_eval_set(client, ctx.corpus, allow_small=True)["source_ids"])

    rebuilt = make_context(client, controller, corpus_version="v2")
    stage_dataset_build(rebuilt)
    in_corpus = set()
    for epoch in (1, 2, 3, 4):
        key = paths.corpus_epoch_file("v2", epoch, None)
        if client.exists(key):
            in_corpus |= {json.loads(ln)["source_id"]
                          for ln in client.read_text(key).splitlines() if ln.strip()}
    val = client.read_text(paths.corpus_eval_split("v2", "val", None))
    in_corpus |= {json.loads(ln)["source_id"] for ln in val.splitlines() if ln.strip()}
    assert in_corpus and not in_corpus & frozen
    assert not client.read_text(paths.corpus_eval_split("v2", "test", None)).strip()


def test_a_pilot_sized_set_is_not_frozen_by_accident(client, controller):
    ctx = _built_corpus(client, controller)
    if not _test_ids(client, ctx):
        pytest.skip("the fixture corpus drew no test document")
    with pytest.raises(FreezeError, match="allow-small"):
        freeze_eval_set(client, ctx.corpus)
    assert not is_frozen(client)


# --------------------------------------------------------------------------
# One frozen set per tenant
# --------------------------------------------------------------------------


#: A review record: a gold is frozen and scored only with one (decision 2026-10-22).
REVIEWED = {"reviewer_id": "a.reviewer", "review_date": "2026-10-20", "draft_backend": "label_studio"}


def _seed_test_split(client, tenant, corpus, labels, metadata=None):
    """A tenant's corpus test split, and what a freeze copies for each document."""
    import json

    rows = "".join(json.dumps({"source_id": sid, "doc_type": "policy"}) + "\n" for sid in labels)
    client.write_text(paths.corpus_eval_split(corpus, "test", tenant), rows)
    for source_id, label in labels.items():
        client.write_json(paths.golden_label("policy", source_id, tenant), label)
        client.write_json(paths.label_metadata("policy", source_id, tenant),
                          {"lob": "homeowners", **REVIEWED, **(metadata or {}).get(source_id, {})})
        client.write_json(paths.ocr_meta("policy", source_id, tenant), {"page_count": 1})
        client.write_bytes(paths.processed_page("policy", source_id, 1, "png", tenant), b"png")


def test_an_unreviewed_gold_is_never_frozen_and_nothing_is_copied(client):
    _seed_test_split(client, "t", "v1", {"policy_0001": {}, "policy_0002": {}},
                     metadata={"policy_0002": {"reviewer_id": "", "review_date": None}})
    with pytest.raises(FreezeError, match="1 of the 2 test document.*policy_0002: no review record"):
        freeze_eval_set(client, "v1", tenant_id="t", allow_small=True)
    assert not client.list(paths.golden_eval_set_dir("t"))


def test_the_frozen_metadata_carries_the_review_record_the_gate_checks(client):
    twin = {"synthetic": True, "review_inherited_from": "homeowners/seed_1"}
    _seed_test_split(client, "t", "v1", {"policy_0001": {}, "policy_0002": {}}, metadata={"policy_0002": twin})
    freeze_eval_set(client, "v1", tenant_id="t", allow_small=True)
    root = paths.golden_eval_set_dir("t")
    real, frozen_twin = (client.read_json(f"{root}/{sid}/metadata.json") for sid in ("policy_0001", "policy_0002"))
    assert {k: real[k] for k in REVIEWED} == REVIEWED and real["review_inherited_from"] is None
    assert frozen_twin["review_inherited_from"] == "homeowners/seed_1" and frozen_twin["synthetic"] is True
    assert [d.source_id for d in load_golden_set(client, tenant_id="t")] == ["policy_0001", "policy_0002"]


def test_a_set_frozen_before_the_rule_is_refused_until_its_records_are_copied_in(client):
    """The gate refuses, never narrows: skipping a frozen gold would score a
    smaller yardstick than earlier versions were scored on."""
    from evaluation.freeze_eval_set import mark_reviewed
    from evaluation.golden_eval import GoldenEvalError

    _seed_test_split(client, "t", "v1", {"policy_0001": {}, "policy_0002": {}})
    freeze_eval_set(client, "v1", tenant_id="t", allow_small=True)
    root = paths.golden_eval_set_dir("t")
    for sid in ("policy_0001", "policy_0002"):                         # as an older freeze wrote it
        frozen = client.read_json(f"{root}/{sid}/metadata.json")
        client.write_json(f"{root}/{sid}/metadata.json", {k: v for k, v in frozen.items() if k not in REVIEWED})
    with pytest.raises(GoldenEvalError, match="2 frozen document.*no review record"):
        load_golden_set(client, tenant_id="t")

    # One label still has no record of its own: the copy leaves it listed.
    client.write_json(paths.label_metadata("policy", "policy_0002", "t"), {"lob": "homeowners"})
    assert list(mark_reviewed(client, tenant_id="t")) == ["policy_0002"]
    with pytest.raises(GoldenEvalError, match="1 frozen document.*policy_0002"):
        load_golden_set(client, tenant_id="t")
    client.write_json(paths.label_metadata("policy", "policy_0002", "t"), {"lob": "homeowners", **REVIEWED})
    golden_before = client.read_json(f"{root}/policy_0002/golden.json")
    assert mark_reviewed(client, tenant_id="t") == {}
    assert len(load_golden_set(client, tenant_id="t")) == 2
    assert client.read_json(f"{root}/policy_0002/golden.json") == golden_before   # only the record changed


def test_two_tenants_freeze_independently_and_each_reads_only_its_own(client):
    """Source ids are numbered per tenant, so both tenants have a policy_0001.
    Freezing one must neither freeze nor fill the other."""
    _seed_test_split(client, "personal", "v1", {
        "policy_0001": {"insured_name": "Personal One"}, "policy_0002": {"insured_name": "P2"},
    })
    _seed_test_split(client, "cgl", "v1", {
        "policy_0001": {"insured_name": "CGL One"}, "policy_0003": {"insured_name": "C3"},
    })

    freeze_eval_set(client, "v1", tenant_id="personal", allow_small=True)
    assert is_frozen(client, tenant_id="personal")
    assert not is_frozen(client, tenant_id="cgl")
    assert eval_set_source_ids(client, tenant_id="cgl") == set()

    manifest = freeze_eval_set(client, "v1", tenant_id="cgl", allow_small=True)  # not "already frozen"
    assert manifest["tenant_id"] == "cgl"
    assert manifest["source_ids"] == ["policy_0001", "policy_0003"]
    assert client.exists(manifest_key(tenant_id="cgl"))

    assert eval_set_source_ids(client, tenant_id="personal") == {"policy_0001", "policy_0002"}
    assert eval_set_source_ids(client, tenant_id="cgl") == {"policy_0001", "policy_0003"}
    for tenant, insured in (("personal", "Personal One"), ("cgl", "CGL One")):
        root = paths.golden_eval_set_dir(tenant)
        assert client.read_json(f"{root}/policy_0001/golden.json") == {"insured_name": insured}
        documents = load_golden_set(client, tenant_id=tenant)
        assert all(key.startswith(f"{root}/") for d in documents for key in d.image_keys)
    # Freezing twice is still refused, per tenant.
    with pytest.raises(FreezeError, match="golden-eval-set/cgl/ is already frozen"):
        freeze_eval_set(client, "v1", tenant_id="cgl", allow_small=True)


def test_a_tenant_never_reads_a_tenant_whose_name_extends_its_own(client):
    """`golden-eval-set/acme` is a prefix of `golden-eval-set/acme-2`, and listing
    is by bare prefix."""
    client.write_json(f"{paths.golden_eval_set_dir('acme-2')}/policy_0001/golden.json", {})
    client.write_json(f"{paths.golden_eval_set_dir('acme-2')}/policy_0001/metadata.json",
                      {"doc_type": "policy"})
    client.write_bytes(f"{paths.golden_eval_set_dir('acme-2')}/policy_0001/page_1.png", b"png")
    assert eval_set_source_ids(client, tenant_id="acme") == set()
    assert load_golden_set(client, tenant_id="acme") == []
    assert eval_set_source_ids(client, tenant_id="acme-2") == {"policy_0001"}


def test_an_id_frozen_in_one_tenant_is_not_frozen_in_another(client, controller):
    """The other tenant's policy_0001 is a different document: it stays in this
    tenant's corpus, and this tenant still draws its own test split."""
    import json

    from evaluation.run_eval import assert_eval_set_disjoint
    from orchestration.pipeline_dag import exclude_eval_families, plan_corpus

    seeded = seed_corpus(client)                       # the default tenant's documents
    colliding = seeded[0]
    other = paths.golden_eval_set_dir("other")
    client.write_json(f"{other}/{colliding}/golden.json", {})
    client.write_json(manifest_key(tenant_id="other"), {"source_ids": [colliding]})

    # The corpus build: the default tenant is not frozen, and keeps the document.
    plan = plan_corpus(make_context(client, controller))
    assert not plan.frozen
    assert colliding in {d.source_id for d in plan.documents}

    doc = SimpleNamespace(source_id=colliding, family="fam-a", carrier=None)
    kept, excluded = exclude_eval_families(client, [doc], tenant_id=None)
    assert kept == [doc] and excluded == []
    kept, excluded = exclude_eval_families(client, [doc], tenant_id="other")
    assert kept == [] and excluded == [colliding]

    # The leakage check: the same id in the default tenant's train split is no leak...
    client.write_text(paths.corpus_epoch_file("v9", 1, None),
                      json.dumps({"source_id": colliding, "doc_type": "policy"}) + "\n")
    assert_eval_set_disjoint(client, "v9", None)
    # ...but it is one in the tenant that froze it.
    client.write_text(paths.corpus_epoch_file("v9", 1, "other"),
                      json.dumps({"source_id": colliding, "doc_type": "policy"}) + "\n")
    from evaluation.run_eval import EvalSetLeakage

    with pytest.raises(EvalSetLeakage, match=colliding):
        assert_eval_set_disjoint(client, "v9", "other")


def test_a_set_frozen_at_the_unscoped_root_is_refused_with_where_to_move_it(client):
    """Ignored, its tenant would read as unfrozen and could freeze a new set,
    changing the yardstick its earlier versions were gated on."""
    from evaluation.freeze_eval_set import frozen_manifest, partial_freeze
    from evaluation.run_eval import assert_eval_set_disjoint

    client.write_json("golden-eval-set/manifest.json",
                      {"tenant_id": "personal", "source_ids": ["policy_0001"]})
    client.write_json("golden-eval-set/policy_0001/golden.json", {})
    _seed_test_split(client, "personal", "v1", {"policy_0002": {"insured_name": "P2"}})

    guidance = r"golden-eval-set/manifest\.json.*Move the root set.*under golden-eval-set/personal/"
    with pytest.raises(FreezeError, match=guidance):
        is_frozen(client, tenant_id="personal")
    # Every tenant, not only the one it came from: no tenant can tell it is not theirs.
    with pytest.raises(FreezeError, match=guidance):
        is_frozen(client, tenant_id="cgl")
    with pytest.raises(FreezeError, match=guidance):
        partial_freeze(client, tenant_id="personal")
    with pytest.raises(FreezeError, match=guidance):
        freeze_eval_set(client, "v1", tenant_id="personal", allow_small=True)
    with pytest.raises(FreezeError, match=guidance):
        assert_eval_set_disjoint(client, "v1", "personal")
    # The gate too, although this tenant's own prefix is empty.
    from common.scopes import get_scope
    from evaluation.golden_eval import evaluate_version

    with pytest.raises(FreezeError, match=guidance):
        evaluate_version(client, object(), version="v1", corpus_version="v1",
                         scope=get_scope("policy"), tenant_id="personal")
    assert not client.exists(manifest_key(tenant_id="personal"))   # nothing was frozen

    # Moved under its tenant, it is that tenant's frozen set again.
    client.delete("golden-eval-set/manifest.json")
    client.delete("golden-eval-set/policy_0001/golden.json")
    client.write_json(manifest_key(tenant_id="personal"),
                      {"tenant_id": "personal", "source_ids": ["policy_0001"]})
    client.write_json(f"{paths.golden_eval_set_dir('personal')}/policy_0001/golden.json", {})
    assert is_frozen(client, tenant_id="personal")
    assert frozen_manifest(client, tenant_id="personal")["source_ids"] == ["policy_0001"]
    assert eval_set_source_ids(client, tenant_id="personal") == {"policy_0001"}


def test_a_root_manifest_naming_no_tenant_still_says_what_to_do(client):
    client.write_json("golden-eval-set/manifest.json", {"source_ids": ["policy_0001"]})
    with pytest.raises(FreezeError, match=r"golden-eval-set/\{tenant\}/, for the tenant it was frozen from"):
        is_frozen(client)


def _test_documents(client, ctx):
    import json

    text = client.read_text(paths.corpus_eval_split(ctx.corpus, "test", None))
    return sorted({(json.loads(line)["source_id"], json.loads(line)["doc_type"])
                   for line in text.splitlines() if line.strip()})


def _mark_in_label_metadata(client, source_id, doc_type):
    key = paths.label_metadata(doc_type, source_id)
    client.write_json(key, {**client.read_json(key), "held_out_carrier": True})


def test_a_document_the_delivery_marks_as_an_unseen_carriers_is_frozen_as_one(client, controller):
    ctx = _built_corpus(client, controller)
    documents = _test_documents(client, ctx)
    if not documents:
        pytest.skip("the fixture corpus drew no test document")
    source_id, doc_type = documents[0]
    _mark_in_label_metadata(client, source_id, doc_type)

    manifest = freeze_eval_set(client, ctx.corpus, allow_small=True)
    root = paths.golden_eval_set_dir(None)
    assert client.read_json(f"{root}/{source_id}/metadata.json")["held_out_carrier"] is True
    assert manifest["held_out_source_ids"] == [source_id]


def test_an_existing_frozen_set_is_marked_without_changing_anything_else(client, controller):
    """A set frozen while the importer dropped the mark: re-import, then mark it.
    Only the flag changes, so the yardstick stays the one every version was
    scored on."""
    from evaluation.freeze_eval_set import mark_held_out

    ctx = _built_corpus(client, controller)
    documents = _test_documents(client, ctx)
    if not documents:
        pytest.skip("the fixture corpus drew no test document")
    freeze_eval_set(client, ctx.corpus, allow_small=True)
    root = paths.golden_eval_set_dir(None)
    source_id, doc_type = documents[0]
    golden_before = client.read_json(f"{root}/{source_id}/golden.json")
    assert client.read_json(f"{root}/{source_id}/metadata.json")["held_out_carrier"] is False

    _mark_in_label_metadata(client, source_id, doc_type)       # what the re-import now writes
    assert mark_held_out(client) == [source_id]
    assert client.read_json(f"{root}/{source_id}/metadata.json")["held_out_carrier"] is True
    assert client.read_json(f"{root}/{source_id}/golden.json") == golden_before
    assert client.read_json(manifest_key())["held_out_source_ids"] == [source_id]
    for other, _ in documents[1:]:
        assert client.read_json(f"{root}/{other}/metadata.json")["held_out_carrier"] is False


def test_marking_needs_a_frozen_set(client):
    from evaluation.freeze_eval_set import mark_held_out

    with pytest.raises(FreezeError, match="no frozen eval set"):
        mark_held_out(client)
