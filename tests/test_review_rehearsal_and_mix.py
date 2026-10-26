"""Review items #25, #31, #34 and #37: rehearsal of a continued run, twin
rotation, error-driven repeats and line balance in the adapter's view of the
corpus (training.corpus_view, training.data_mix). Synthetic rows only."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from training import corpus_view, data_mix
from training.corpus_view import (
    error_rate,
    field_key,
    hard_fields,
    holds_any,
    line_repeats,
    materialize,
    plan_rehearsal,
    target_fields,
)
from training.data_mix import DataMixError, MixSettings, parse_settings, plan_twin_rotation

MODES = {1: "ocr_plus_image", 2: "image_only", 3: "noisy_ocr_image", 4: "image_only"}


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _answer(vin: str | None = None) -> str:
    """A compact (fast profile) window answer: a vehicle table and a policy number."""
    vehicles = [{"vin": {"v": vin, "p": [2]}}] if vin else []
    return json.dumps({"policy_info": {"policy_number": {"v": "P-1", "p": [1]}}, "vehicles": vehicles})


def _row(source_id: str, line: str, *, family: str | None = None, synthetic: bool = False,
         twin: int | None = None, carrier: str = "Acme", render: str | None = None,
         answer: str | None = None) -> dict:
    return {"source_id": source_id, "group_id": family or source_id, "doc_type": "policy", "lob": line,
            "synthetic": synthetic, "twin_index": twin, "carrier": carrier,
            "is_scanned": render == "scanned", "render_mode": render, "sections": "decl",
            "messages": [{"role": "user", "content": "read"},
                         {"role": "assistant", "content": answer or _answer()}]}


def _seed(line: str, seed: str, twins: int, *, renders: tuple[str, ...] = ("digital",)) -> list[dict]:
    family = f"{line}/{seed}"
    rows = [_row(f"{line}__{seed}", line, family=family)]
    for t in range(1, twins + 1):
        for render in renders:
            suffix = "" if render == "digital" else "__scan"
            rows.append(_row(f"{line}__{seed}__twin_{t:03d}{suffix}", line, family=family, synthetic=True,
                             twin=t, render=render))
    return rows


def _write_corpus(client: BlobClient, train: list[dict], val: list[dict] | None = None) -> None:
    for epoch in (1, 2, 3, 4):
        body = "".join(json.dumps({**r, "epoch": epoch, "modality_mode": MODES[epoch]}) + "\n" for r in train)
        client.write_text(paths.corpus_epoch_file("v1", epoch), body)
    val = val or [_row("val-1", "homeowners")]
    client.write_text(paths.corpus_eval_split("v1", "val"), "".join(json.dumps(r) + "\n" for r in val))


def _epoch_rows(client: BlobClient, view) -> list[list[dict]]:
    return [[json.loads(line) for line in client.read_text(f).splitlines() if line.strip()]
            for f in view.epoch_files]


@pytest.fixture
def plain(monkeypatch):
    """Today's defaults, set explicitly, whatever the config says."""
    monkeypatch.setattr(corpus_view, "_rotation_enabled", lambda: False)
    monkeypatch.setattr(corpus_view, "_balance_settings", lambda: {"min_documents": 0, "max_repeat": 1})
    monkeypatch.setattr(corpus_view, "_balance_options", lambda: {"count": "documents", "distinct_repeats": False})
    monkeypatch.setattr(corpus_view, "_rehearsal_settings", lambda: (None, False))
    monkeypatch.setattr(corpus_view, "_hard_settings", lambda: None)
    monkeypatch.setattr(data_mix, "mix_settings", lambda: MixSettings())


# --------------------------------------------------------------------------
# Defaults: everything new is off
# --------------------------------------------------------------------------


def test_the_shipped_config_leaves_every_new_option_off():
    from common.config import continued_training_problem, training_config

    cfg = training_config("unified")
    assert cfg["continued_training"]["rehearsal_old_share"] is None
    assert cfg["continued_training"]["rehearsal_weight_by_line_errors"] is False
    assert cfg["continued_training"]["num_train_epochs"] is None
    assert cfg["continued_training"]["learning_rate"] is None
    assert cfg["data_mix"]["twins_per_seed_per_epoch"] is None
    assert cfg["hard_examples"] == {"top_fields": None, "max_repeat": 2, "min_values": 30}
    assert corpus_view._balance_options() == {"count": "documents", "distinct_repeats": False}
    assert corpus_view._rehearsal_settings() == (None, False)
    assert corpus_view._hard_settings() is None
    assert data_mix.mix_settings().twins_per_seed_per_epoch is None
    assert continued_training_problem(cfg) is None


def test_the_rehearsal_keys_are_data_not_optimizer_values():
    from common.config import continued_training_problem
    from training.train import optimization_for

    cfg = {"optimization": {"num_train_epochs": 3, "learning_rate": 1e-4},
           "continued_training": {"num_train_epochs": 1, "rehearsal_old_share": 0.4,
                                  "rehearsal_weight_by_line_errors": True}}
    assert continued_training_problem(cfg) is None
    assert optimization_for(cfg, continuing=True) == {"num_train_epochs": 1, "learning_rate": 1e-4}
    bad = {**cfg, "continued_training": {"rehearsal_old_share": 1.5}}
    assert "rehearsal_old_share" in continued_training_problem(bad)


# --------------------------------------------------------------------------
# #25 Rehearsal
# --------------------------------------------------------------------------


def _rehearsal_rows() -> tuple[list[dict], set[str]]:
    rows, old = [], set()
    for line, n_old, n_new in (("homeowners", 30, 4), ("personal_auto", 10, 2)):
        for i in range(n_old):
            rows.append(_row(f"{line}-old-{i}", line, carrier=f"C{i % 3}"))
            old.add(f"{line}-old-{i}")
        rows += [_row(f"{line}-new-{i}", line) for i in range(n_new)]
    return rows, old


def test_each_epoch_trains_every_new_document_and_a_stratified_rotated_share_of_the_old():
    rows, old = _rehearsal_rows()
    plan = plan_rehearsal(rows, old, 0.4, seed=42)
    every = {r["source_id"] for r in rows}
    new = every - old
    seen: set[str] = set()
    for epoch in (1, 2, 3, 4):
        trained = every - plan.rested[epoch]
        assert new <= trained                                           # every new document, every epoch
        olds = trained & old
        by_line = Counter(s.split("-")[0] for s in olds)
        assert by_line == {"homeowners": 12, "personal_auto": 4}       # 0.4 of 30 and of 10
        carriers = Counter(next(r["carrier"] for r in rows if r["source_id"] == s)
                           for s in olds if s.startswith("homeowners"))
        assert set(carriers.values()) <= {3, 4, 5}                      # 12 over 3 carriers of 10
        seen |= olds
        if epoch == 3:
            assert seen == old                                          # ceil(1 / 0.4) epochs see them all
    assert plan.rested[1] != plan.rested[2]
    assert plan.record["new"] == sorted(new)
    assert plan.record["rehearsed_by_epoch"][1] == sorted((every - plan.rested[1]) & old)
    assert plan.record["lines"]["homeowners"]["old_per_epoch"] == 12
    assert plan_rehearsal(rows, old, 0.4, seed=42).rested == plan.rested           # reproducible
    assert plan_rehearsal(rows, old, 0.4, seed=7).rested[1] != plan.rested[1]      # and follows the seed


def test_line_errors_weight_the_rehearsal_toward_the_lines_the_parent_got_wrong():
    rows, old = _rehearsal_rows()
    plan = plan_rehearsal(rows, old, 0.4, seed=42,
                          line_rates={"homeowners": 0.05, "personal_auto": 0.30})
    lines = plan.record["lines"]
    assert lines["personal_auto"]["share"] == 1.0                       # capped at every old document
    assert lines["personal_auto"]["old_per_epoch"] == 10
    assert lines["homeowners"]["share"] < 0.4
    assert plan.record["weighted_by_line_errors"] is True


def test_a_continued_run_rehearses_and_a_fresh_one_trains_everything(client, plain, monkeypatch):
    rows, old = _rehearsal_rows()
    _write_corpus(client, rows)
    monkeypatch.setattr(corpus_view, "_rehearsal_settings", lambda: (0.4, False))
    monkeypatch.setattr(corpus_view, "chain_trained_documents", lambda c, run, t=None: (old, None))

    fresh = materialize("v1", client)
    assert fresh.rehearsal["enabled"] is False and fresh.rows_by_epoch[1] == len(rows)

    view = materialize("v1", client, parent_run_id="policy-v1", out_version="v2")
    assert view.rehearsal["enabled"] and view.rehearsal["parent_run_id"] == "policy-v1"
    assert view.rehearsal["new_documents"] == 6 and view.rehearsal["old_documents"] == 40
    for epoch, written in enumerate(_epoch_rows(client, view), start=1):
        ids = {r["source_id"] for r in written}
        assert {r["source_id"] for r in rows} - old <= ids
        assert len(ids & old) == 16
        assert sorted(ids & old) == view.rehearsal["rehearsed_by_epoch"][epoch]


def test_an_unreadable_chain_turns_rehearsal_off_rather_than_guess(client, plain, monkeypatch):
    rows, old = _rehearsal_rows()
    _write_corpus(client, rows)
    monkeypatch.setattr(corpus_view, "_rehearsal_settings", lambda: (0.4, False))
    monkeypatch.setattr(corpus_view, "chain_trained_documents", lambda c, run, t=None: (old, "a manifest is gone"))
    view = materialize("v1", client, parent_run_id="policy-v1")
    assert view.rehearsal["enabled"] is False and "a manifest is gone" in view.rehearsal["reason"]
    assert view.rows_by_epoch[1] == len(rows)


# --------------------------------------------------------------------------
# #31 Twin rotation
# --------------------------------------------------------------------------


def test_twin_rotation_takes_k_twins_a_seed_each_epoch_and_every_twin_over_the_epochs():
    rows = _seed("homeowners", "a", 6, renders=("digital", "scanned")) + _seed("homeowners", "b", 5)
    plan = plan_twin_rotation(rows, 2, seed=42, epochs=4, line_of=corpus_view._line_of)
    every = {r["source_id"] for r in rows}
    seen: set[int] = set()
    for epoch in (1, 2, 3, 4):
        trained = every - plan.rested[epoch]
        assert {"homeowners__a", "homeowners__b"} <= trained            # the seeds always
        twins_a = {s.split("twin_")[1][:3] for s in trained if s.startswith("homeowners__a__twin")}
        assert len(twins_a) == 2                                        # 2 twins, both renders of each
        assert sum(s.startswith("homeowners__b__twin") for s in trained) == 2
        seen |= {int(t) for t in twins_a}
        if epoch == 3:
            assert seen == set(range(1, 7))
    assert plan.lines["homeowners"]["twins"] == 11
    assert plan.lines["homeowners"]["twins_seen_by_epoch"][3] == 11
    again = plan_twin_rotation(rows, 2, seed=42, epochs=4, line_of=corpus_view._line_of)
    assert again.rested == plan.rested


def test_twins_per_seed_per_epoch_is_a_whole_number_or_null():
    assert parse_settings({"twins_per_seed_per_epoch": 3}).twins_per_seed_per_epoch == 3
    assert parse_settings({}).twins_per_seed_per_epoch is None
    assert not parse_settings({"twins_per_seed_per_epoch": 3}).steers       # not a share
    for bad in (0, 1.5, True, "2"):
        with pytest.raises(DataMixError, match="twins_per_seed_per_epoch"):
            parse_settings({"twins_per_seed_per_epoch": bad})


def test_the_view_rotates_twins_and_render_rotation_still_picks_the_render(client, plain, monkeypatch):
    monkeypatch.setattr(corpus_view, "_rotation_enabled", lambda: True)
    monkeypatch.setattr(data_mix, "mix_settings", lambda: MixSettings(twins_per_seed_per_epoch=2))
    rows = _seed("homeowners", "a", 6, renders=("digital", "scanned"))
    _write_corpus(client, rows)
    view = materialize("v1", client)
    for written in _epoch_rows(client, view):
        ids = [r["source_id"] for r in written]
        assert "homeowners__a" in ids and len(ids) == 3                 # the seed + 2 twins, one render each
    assert view.twin_rotation["enabled"] and view.twin_rotation["twins_per_seed_per_epoch"] == 2
    assert view.mix_settings["twins_per_seed_per_epoch"] == 2


# --------------------------------------------------------------------------
# #34 Hard examples
# --------------------------------------------------------------------------


ERRORS = {
    "version": 1, "run_id": "policy-v1", "checkpoint": "checkpoint-700",
    "by_field": {
        "vehicles[].vin": {"values": 120, "wrong": 30, "missed": 10, "invented": 2},
        "policy_info.policy_number": {"values": 400, "wrong": 4, "missed": 0, "invented": 0},
        "drivers[].license": {"values": 12, "wrong": 9, "missed": 0, "invented": 0},     # too few values
        "vehicles[].year": {"values": 120, "wrong": 0, "missed": 0, "invented": 0},     # no error
    },
    "by_line": {"homeowners": {"values": 100, "wrong": 5, "missed": 0, "invented": 0}},
    "by_group": {}, "by_mode": {},
}


def test_hard_fields_rank_by_error_rate_over_enough_values():
    assert error_rate(ERRORS["by_field"]["vehicles[].vin"]) == (0.35, 120)
    assert [f["field"] for f in hard_fields(ERRORS, 5)] == ["vehicles[].vin", "policy_info.policy_number"]
    assert [f["field"] for f in hard_fields(ERRORS, 1)] == ["vehicles[].vin"]
    assert hard_fields(ERRORS, 5, min_values=500) == []
    assert field_key("vehicles[3].coverages[0].limit") == "vehicles[].coverages[].limit"


def test_a_target_holds_the_fields_it_states_a_value_for():
    row = _row("d", "personal_auto", answer=_answer(vin="1HGCM"))
    assert target_fields(row) == {"policy_info.policy_number", "vehicles[].vin"}
    assert "vehicles[].vin" not in target_fields(_row("e", "personal_auto"))
    assert target_fields({"messages": []}) == set()
    assert holds_any({"vehicles[].vin"}, {"vehicles[].vin"})
    assert holds_any({"vehicles[].vin"}, {"personal_auto.vehicles[].vin"})     # rooted differently
    assert not holds_any({"vehicles[].year"}, {"vehicles[].vin"})


def test_hard_windows_are_repeated_from_other_epoch_files(client, plain, monkeypatch):
    rows = [_row(f"pa-{i}", "personal_auto", answer=_answer(vin=f"V{i}" if i < 2 else None)) for i in range(5)]
    _write_corpus(client, rows)
    monkeypatch.setattr(corpus_view, "_hard_settings",
                        lambda: {"top_fields": 1, "max_repeat": 3, "min_values": 30})
    monkeypatch.setattr(paths, "validation_errors_key",
                        lambda run_id, tenant_id=None: f"registry/{run_id}/validation_errors.json", raising=False)
    client.write_text("registry/policy-v1/validation_errors.json", json.dumps(ERRORS))
    view = materialize("v1", client, parent_run_id="policy-v1", out_version="v2")
    assert view.hard_examples["enabled"] and view.hard_examples["errors_run"] == "policy-v1"
    assert [f["field"] for f in view.hard_examples["fields"]] == ["vehicles[].vin"]
    first = _epoch_rows(client, view)[0]
    copies = [r for r in first if r["source_id"] == "pa-0"]
    assert [r["modality_mode"] for r in copies] == [MODES[1], MODES[2], MODES[3]]   # each its own draw
    assert sum(r["source_id"] == "pa-4" for r in first) == 1                        # no hard field
    assert view.hard_examples["rows_repeated_by_epoch"] == {1: 4, 2: 4, 3: 4, 4: 4}
    assert view.rows_by_epoch[1] == 5 + 4


def test_without_a_record_nothing_is_repeated_and_the_view_says_why(client, plain, monkeypatch):
    rows = [_row(f"pa-{i}", "personal_auto", answer=_answer(vin="V")) for i in range(3)]
    _write_corpus(client, rows)
    monkeypatch.setattr(corpus_view, "_hard_settings",
                        lambda: {"top_fields": 1, "max_repeat": 2, "min_values": 30})
    monkeypatch.delattr(paths, "validation_errors_key", raising=False)
    view = materialize("v1", client, parent_run_id="policy-v1")
    assert view.hard_examples["enabled"] is False and "validation_errors_key" in view.hard_examples["reason"]
    assert view.rows_by_epoch[1] == 3

    monkeypatch.setattr(paths, "validation_errors_key",
                        lambda run_id, tenant_id=None: f"registry/{run_id}/validation_errors.json", raising=False)
    monkeypatch.setattr(corpus_view, "promoted_run", lambda c, t, v: None)
    fresh = materialize("v1", client, out_version="v2")
    assert fresh.hard_examples["enabled"] is False and fresh.hard_examples["errors_run"] is None
    assert fresh.rows_by_epoch[1] == 3


def test_a_fresh_run_reads_the_promoted_release_s_errors(client, plain, monkeypatch):
    rows = [_row(f"pa-{i}", "personal_auto", answer=_answer(vin="V")) for i in range(3)]
    _write_corpus(client, rows)
    monkeypatch.setattr(corpus_view, "_hard_settings",
                        lambda: {"top_fields": 1, "max_repeat": 2, "min_values": 30})
    monkeypatch.setattr(paths, "validation_errors_key",
                        lambda run_id, tenant_id=None: f"registry/{run_id}/validation_errors.json", raising=False)
    monkeypatch.setattr(corpus_view, "promoted_run", lambda c, t, v: "policy-v4")
    client.write_text("registry/policy-v4/validation_errors.json", json.dumps(ERRORS))
    view = materialize("v1", client, out_version="v5")
    assert view.hard_examples["errors_run"] == "policy-v4" and view.rows_by_epoch[1] == 6


# --------------------------------------------------------------------------
# #37 Line balance: count families, fill repeats from other epoch files
# --------------------------------------------------------------------------


def test_a_line_of_few_seeds_and_many_twins_is_small_counted_in_families():
    rows = [r for s in "abcde" for r in _seed("motorcycle", s, 20)]                  # 105 documents, 5 families
    rows += [_row(f"h{i}", "homeowners") for i in range(100)]
    assert line_repeats(rows, min_documents=100, max_repeat=3) == {"motorcycle": 1, "homeowners": 1}
    assert line_repeats(rows, min_documents=100, max_repeat=3, count="families") == {
        "motorcycle": 3, "homeowners": 1}
    assert line_repeats(rows, min_documents=100, max_repeat=3, count="real") == {
        "motorcycle": 3, "homeowners": 1}
    only_twins = [r for r in rows if r.get("synthetic")] + rows[-100:]
    assert line_repeats(only_twins, min_documents=100, max_repeat=3, count="real")["motorcycle"] == 3
    with pytest.raises(corpus_view.CorpusViewError):
        line_repeats(rows, min_documents=100, max_repeat=3, count="pages")


def test_distinct_repeats_come_from_the_other_epoch_files(client, plain, monkeypatch):
    rows = [_row(f"h{i}", "homeowners") for i in range(9)] + [_row("m0", "motorcycle")]
    _write_corpus(client, rows)
    monkeypatch.setattr(corpus_view, "_balance_settings", lambda: {"min_documents": 9, "max_repeat": 3})
    same = materialize("v1", client)
    copies = [r for r in _epoch_rows(client, same)[0] if r["source_id"] == "m0"]
    assert len(copies) == 3 and len({json.dumps(r, sort_keys=True) for r in copies}) == 1   # today: identical

    monkeypatch.setattr(corpus_view, "_balance_options", lambda: {"count": "documents", "distinct_repeats": True})
    view = materialize("v1", client)
    for epoch, written in enumerate(_epoch_rows(client, view), start=1):
        copies = [r for r in written if r["source_id"] == "m0"]
        expected = [(epoch - 1 + k) % 4 + 1 for k in range(3)]
        assert [r["epoch"] for r in copies] == expected                # its own, then the next two files
        assert sum(r["source_id"].startswith("h") for r in written) == 9
    assert view.rows_by_epoch == same.rows_by_epoch == {1: 12, 2: 12, 3: 12, 4: 12}
    assert view.line_balance["distinct_repeats"] is True


def test_with_every_option_off_the_view_writes_what_it_always_wrote(client, plain, monkeypatch):
    """The default path is the old one, byte for byte: each kept row, repeated in place."""
    rows = [_row(f"h{i}", "homeowners") for i in range(9)] + [_row("m0", "motorcycle")]
    _write_corpus(client, rows)
    monkeypatch.setattr(corpus_view, "_balance_settings", lambda: {"min_documents": 9, "max_repeat": 3})
    view = materialize("v1", client)
    source = client.read_text(paths.corpus_epoch_file("v1", 2)).splitlines()
    expected = [line for line in source for _ in range(3 if '"m0"' in line else 1)]
    assert client.read_text(view.epoch_files[1]).splitlines() == expected
    assert view.rehearsal == view.twin_rotation == view.hard_examples == {}
