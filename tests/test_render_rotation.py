"""One render per twin per epoch in the adapter's view of the corpus
(training.corpus_view.plan_render_rotation and materialize)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from training import corpus_view
from training.corpus_view import line_repeats, materialize, plan_render_rotation


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _seed_rows(line: str, seed: str, twins: int, *, scan_only: tuple[int, ...] = ()) -> list[dict]:
    """A seed's original and its twins, each twin a digital and a scanned render
    (one scan for a twin in ``scan_only``), one row per document."""
    family = f"delivered:{line}/{seed}"

    def row(source_id, synthetic, twin, scanned, mode):
        return {"source_id": source_id, "group_id": family, "doc_type": "policy", "lob": line,
                "synthetic": synthetic, "twin_index": twin, "is_scanned": scanned, "render_mode": mode,
                "modality_mode": "ocr_plus_image"}

    rows = [row(f"{line}__{seed}__original", False, None, False, None)]
    for t in range(1, twins + 1):
        name = f"{line}__{seed}__twin_{t:03d}"
        if t not in scan_only:
            rows.append(row(name, True, t, False, "digital"))
        rows.append(row(f"{name}__scan", True, t, True, "scanned"))
    return rows


def _trained(rows, plan, epoch):
    return {r["source_id"] for r in rows} - plan.rested.get(epoch, set())


def test_each_twin_trains_once_an_epoch_in_alternating_renders_and_the_original_every_epoch():
    rows = _seed_rows("homeowners", "a", 6)
    plan = plan_render_rotation(rows, seed=42)
    epochs = [_trained(rows, plan, e) for e in (1, 2, 3, 4)]
    for trained in epochs:
        assert "homeowners__a__original" in trained
        assert len(trained) == 7                                          # the original + one render of 6 twins
        assert sum(1 for s in trained if s.endswith("__scan")) == 3      # half and half
    for t in range(1, 7):
        name = f"homeowners__a__twin_{t:03d}"
        renders = [name in trained for trained in epochs]               # digital this epoch?
        assert renders in ([True, False, True, False], [False, True, False, True])
    assert plan.lines["homeowners"] == {"documents": 13, "documents_per_epoch": 7, "twins_rotated": 6,
                                        "both_renders": False}


def test_a_twin_with_one_render_trains_every_epoch():
    rows = _seed_rows("motorcycle", "a", 3, scan_only=(2,))
    plan = plan_render_rotation(rows, seed=42)
    assert all("motorcycle__a__twin_002__scan" in _trained(rows, plan, e) for e in (1, 2, 3, 4))
    assert plan.lines["motorcycle"]["twins_rotated"] == 2


def test_the_rotation_is_reproducible_and_follows_the_seed():
    rows = _seed_rows("homeowners", "a", 8)
    assert plan_render_rotation(rows, seed=42).rested == plan_render_rotation(rows, seed=42).rested
    assert len({frozenset(plan_render_rotation(rows, seed=s).rested[1]) for s in range(10)}) > 1


def test_a_line_rotation_would_take_under_the_balance_floor_keeps_both_renders():
    big = [r for s in "abcd" for r in _seed_rows("homeowners", s, 15)]           # 124 documents, 64 an epoch
    small = _seed_rows("personal_umbrella", "a", 15)                              # 31 documents, 16 an epoch
    plan = plan_render_rotation(big + small, seed=42, min_documents=100)
    assert plan.target == 64
    assert plan.lines["homeowners"]["documents_per_epoch"] == 64
    assert plan.lines["personal_umbrella"] == {"documents": 31, "documents_per_epoch": 31, "twins_rotated": 0,
                                               "both_renders": True}
    assert not any(s.startswith("personal_umbrella") for s in plan.rested[1])


def test_the_balance_target_is_fixed_before_a_small_line_takes_both_renders():
    """Un-rotated, the small line outgrows the rotated large one; counted
    afresh, line balance would then repeat the large line."""
    large = _seed_rows("homeowners", "a", 15)                                     # 16 an epoch
    small = _seed_rows("motorcycle", "a", 12)                                     # 13 an epoch, 25 unrotated
    rows = large + small
    plan = plan_render_rotation(rows, seed=42, min_documents=100)
    assert plan.target == 16 and plan.lines["motorcycle"]["both_renders"]
    first = [r for r in rows if r["source_id"] not in plan.rested[1]]
    assert line_repeats(first, min_documents=100, max_repeat=3, target=plan.target) == {
        "homeowners": 1, "motorcycle": 1}


def test_without_line_balance_every_line_rotates():
    plan = plan_render_rotation(_seed_rows("personal_umbrella", "a", 15), seed=42, min_documents=0)
    assert plan.lines["personal_umbrella"]["documents_per_epoch"] == 16


def _seed_corpus(client, train, val):
    for epoch in (1, 2, 3, 4):
        client.write_text(paths.corpus_epoch_file("v1", epoch), "".join(json.dumps(r) + "\n" for r in train))
    client.write_text(paths.corpus_eval_split("v1", "val"), "".join(json.dumps(r) + "\n" for r in val))


def _epoch_ids(client, view):
    return [[json.loads(line)["source_id"] for line in client.read_text(f).splitlines() if line.strip()]
            for f in view.epoch_files]


def test_the_view_rotates_renders_and_keeps_every_render_for_validation(client):
    train = _seed_rows("homeowners", "a", 4) + _seed_rows("homeowners", "b", 4)
    val = _seed_rows("homeowners", "v", 3)
    _seed_corpus(client, train, val)
    view = materialize("v1", client)
    epochs = _epoch_ids(client, view)
    assert all(len(ids) == len(set(ids)) == 10 for ids in epochs)            # 2 originals + 8 twins, once each
    for t in range(1, 5):
        for seed in "ab":
            name = f"homeowners__{seed}__twin_{t:03d}"
            assert (name in epochs[0]) != (name in epochs[1])                 # the other render next epoch
    assert view.rows_by_epoch == {1: 10, 2: 10, 3: 10, 4: 10}
    assert view.examples_by_line["train"]["homeowners"]["documents"] == 10
    assert view.val_rows == len(val) == 7                                     # every render
    assert view.render_rotation["enabled"] and view.render_rotation["lines"]["homeowners"]["twins_rotated"] == 8


def test_render_rotation_off_trains_both_renders_every_epoch(client, monkeypatch):
    monkeypatch.setattr(corpus_view, "_rotation_enabled", lambda: False)
    train = _seed_rows("homeowners", "a", 4)
    _seed_corpus(client, train, _seed_rows("homeowners", "v", 1))
    view = materialize("v1", client)
    assert all(len(ids) == 9 for ids in _epoch_ids(client, view))
    assert view.render_rotation == {}


def test_the_adapter_rotates_renders():
    assert corpus_view._rotation_enabled()


def test_the_model_card_shows_the_rotation_per_line():
    from registry_utils.model_card import _mix

    stats = SimpleNamespace(data_mix={"settings": {}, "lines": {}, "rested_documents": 0, "render_rotation": {
        "enabled": True, "balance_target": 64,
        "lines": {"homeowners": {"documents": 124, "documents_per_epoch": 64, "twins_rotated": 60,
                                 "both_renders": False},
                  "personal_umbrella": {"documents": 31, "documents_per_epoch": 31, "twins_rotated": 0,
                                        "both_renders": True}}}})
    card = "\n".join(_mix(stats))
    assert "Render rotation: on" in card
    assert "| homeowners | 124 | 64 | 60 | no |" in card
    assert "| personal_umbrella | 31 | 31 | 0 | yes (under the balance floor) |" in card
    off = "\n".join(_mix(SimpleNamespace(data_mix={"settings": {}, "lines": {}})))
    assert "Render rotation: off" in off


def test_every_corpus_row_says_which_render_it_is():
    from data_pipeline.dataset_builder.build_jsonl import build_corpus
    from data_pipeline.dataset_builder.split_groups import GroupSplitAssignment
    from tests.test_dataset_builder import _documents

    docs = _documents(6)
    for index, doc in enumerate(docs):
        doc.render_mode = "scanned" if index % 2 else "digital"
    assignment = GroupSplitAssignment(assignment={d.family: "train" for d in docs[:4]}
                                      | {docs[4].family: "val", docs[5].family: "test"})
    built = build_corpus(docs, assignment)
    by_id = {d.source_id: d.render_mode for d in docs}
    rows = [row for split_rows in built.rows_by_split.values() for row in split_rows]
    assert rows and all(row["render_mode"] == by_id[row["source_id"]] for row in rows)
