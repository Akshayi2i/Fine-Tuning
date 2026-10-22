"""Carrier and twin index on rows, one carrier held out per line, at most 20 twins per seed
(Fideon SPEC_09 amendment items 4, 5 and 8; handoff item 2)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from data_pipeline.dataset_builder.split_groups import (
    MAX_TWINS_PER_SEED,
    GroupRecord,
    GroupSplitAssignment,
    assign_delivered_splits,
    cap_twins,
    held_out_carrier_per_line,
)


def _record(group, carrier, line="homeowners", size=11, synthetic=False, doc_type="policy"):
    # Real families unless a test says otherwise: on a policy, carriers are
    # counted and chosen from real families only (decision 2026-10-22).
    return GroupRecord(group_id=group, doc_type=doc_type, source_ids=[f"{group}-{i}" for i in range(size)],
                       carrier=carrier, line=line, synthetic=synthetic)


# --------------------------------------------------------------------------
# Carrier hold-out per line
# --------------------------------------------------------------------------

def test_the_carrier_costing_train_and_val_least_is_held_out_and_single_carrier_lines_listed():
    records = [_record("a", "big co", size=30), _record("b", "small co", size=5), _record("c", "big co"),
               _record("v", "big co"), _record("d", "only co", line="motorcycle")]
    splits = {"a": "train", "b": "train", "c": "train", "v": "val", "d": "train"}
    choice = held_out_carrier_per_line(records, seed=42, split_of=splits, min_carriers=2)
    assert choice.held == {"homeowners": "small co"} and choice.single == ["motorcycle"]


def test_test_documents_cost_nothing_when_choosing():
    # zenith has the most documents, but most are already test: holding it out costs 22.
    records = [_record("a", "acme", size=10), _record("a2", "acme", size=15),
               _record("t1", "zenith", size=40), _record("t2", "zenith", size=40),
               _record("tr", "zenith"), _record("v", "zenith")]
    splits = {"a": "train", "a2": "val", "t1": "test", "t2": "test", "tr": "train", "v": "val"}
    assert held_out_carrier_per_line(records, seed=42, split_of=splits,
                                     min_carriers=2).held == {"homeowners": "zenith"}


def test_a_line_keeps_its_only_validation_carrier():
    """motorcycle on the delivered data: Progressive was its only validation carrier."""
    records = [_record("m1", "allstate", line="motorcycle"), _record("m2", "progressive", line="motorcycle"),
               _record("m3", "progressive", line="motorcycle")]
    splits = {"m1": "train", "m2": "val", "m3": "test"}
    choice = held_out_carrier_per_line(records, seed=42, split_of=splits, min_carriers=2)
    assert choice.held == {} and "motorcycle" in choice.not_held_out      # either choice empties train or val
    records.append(_record("m4", "honda", line="motorcycle"))
    splits["m4"] = "train"
    choice = held_out_carrier_per_line(records, seed=42, split_of=splits, min_carriers=2)
    assert choice.held["motorcycle"] in {"allstate", "honda"}            # never progressive


def test_a_held_out_carrier_goes_to_test_whatever_split_it_was_delivered_in():
    splits = {"a": "train", "b": "val", "c": "val", "d": "test", "z": "train", "e": "train"}
    groups = {"policy": [_record("a", "acme"), _record("b", "acme"), _record("c", "zenith", size=40),
                         _record("d", "zenith", size=40), _record("z", "zenith", size=40),
                         _record("e", "solo", line="motorcycle")]}
    result = assign_delivered_splits(groups, splits, hold_out_carriers=True, min_carriers=2)
    assert result.held_out_carriers_by_line["policy"] == {"homeowners": "acme"}
    assert result.assignment["a"] == "test" and result.assignment["b"] == "test"     # moved out of train and val
    assert result.assignment["e"] == "train"                                         # single-carrier line: untouched
    assert result.single_carrier_lines["policy"] == ["motorcycle"]
    assert result.moved_to_test["policy"] == 2
    assert set(result.held_out_source_ids) == {f"a-{i}" for i in range(11)} | {f"b-{i}" for i in range(11)}
    recorded = result.as_dict()
    assert recorded["held_out_carriers_by_line"] == {"policy": {"homeowners": "acme"}}
    # Held out of its line, not of the doc type: it is not recorded as a carrier
    # placed entirely in test, which the frozen-set warning reads.
    assert not result.held_out_carriers


def test_lines_that_hold_none_out_or_name_no_carrier_are_recorded():
    splits = {"m1": "train", "m2": "val", "m3": "test", "n1": "train", "n2": "val"}
    groups = {"policy": [_record("m1", "allstate", line="motorcycle"),
                         _record("m2", "progressive", line="motorcycle"),
                         _record("m3", "allstate", line="motorcycle"),
                         _record("n1", None, line="ocean_marine"), _record("n2", None, line="ocean_marine")]}
    result = assign_delivered_splits(groups, splits, hold_out_carriers=True, min_carriers=2)
    assert "motorcycle" in result.lines_not_held_out["policy"]
    assert result.lines_without_carrier["policy"] == ["ocean_marine"]
    assert result.assignment["m2"] == "val"                                          # its validation kept


def test_asked_for_on_a_delivered_split_the_ten_carrier_minimum_still_applies():
    splits = {"a": "train", "b": "val", "c": "test", "d": "train"}
    groups = {"policy": [_record("a", "acme"), _record("b", "zenith"), _record("c", "zenith"),
                         _record("d", "zenith")]}
    result = assign_delivered_splits(groups, splits, hold_out_carriers=True)
    assert result.assignment == splits and not result.held_out_carriers_by_line
    assert result.lines_below_carrier_minimum == {"policy": {"homeowners": 2}}


def test_a_delivered_split_is_kept_as_delivered_unless_the_hold_out_is_asked_for():
    """Off by default (decided 2026-10-07): the delivery already keeps whole
    seeds out of training. Asked for, it is still skipped once the eval set is
    frozen."""
    splits = {"a": "train", "b": "val", "c": "test"}
    groups = {"policy": [_record("a", "acme"), _record("b", "zenith", size=40), _record("c", "zenith", size=40)]}
    result = assign_delivered_splits(groups, splits)
    assert result.assignment == splits and not result.held_out_carriers_by_line
    assert assign_delivered_splits(groups, splits, hold_out_carriers=False).assignment == splits
    frozen = {"a": "train", "b": "val"}
    result = assign_delivered_splits({"policy": groups["policy"][:2]}, frozen, with_test=False,
                                     hold_out_carriers=True)
    assert result.assignment == frozen and not result.held_out_carriers_by_line


# --------------------------------------------------------------------------
# The hold-out leaks no seen carrier (decision 2026-10-22, policies only)
# --------------------------------------------------------------------------


def _ten_carriers(line="homeowners"):
    """Ten real carriers of one line, each with a train, a val and a test family."""
    records, splits = [], {}
    for index in range(10):
        for split in ("train", "val", "test"):
            group = f"c{index}-{split}"
            records.append(_record(group, f"carrier {index}", line=line, size=1 + index))
            splits[group] = split
    return records, splits


def test_a_carrier_with_a_synthetic_family_is_never_held_out():
    """Twins train wherever they sit, so their carrier is one the model has seen.
    carrier 0 is the cheapest to hold out; with a twin family it is not chosen."""
    records, splits = _ten_carriers()
    assert held_out_carrier_per_line(records, seed=42, split_of=splits).held == {"homeowners": "carrier 0"}
    twins = records + [_record("c0-twins", "carrier 0", size=40, synthetic=True)]
    choice = held_out_carrier_per_line(twins, seed=42, split_of={**splits, "c0-twins": "train"})
    assert choice.held == {"homeowners": "carrier 1"}


def test_a_drawn_split_never_trains_the_held_out_carriers_real_document():
    """A family holding a real seed and its twins is synthetic and trains before
    the hold-out is looked at; were its carrier held out, that carrier's real
    document would train while its others were scored as unseen."""
    from data_pipeline.dataset_builder.split_groups import assign_group_splits

    records = [_record(f"c{i}-{n}", f"carrier {i}", size=5) for i in range(1, 12) for n in range(4)]
    records += [_record("c0-real", "carrier 0", size=1), _record("c0-real2", "carrier 0", size=1)]
    # Chosen from the real families alone, carrier 0 - the cheapest - is held out.
    assert assign_group_splits({"policy": records}, seed=42).held_out_carriers_by_line == {
        "policy": {"homeowners": "carrier 0"}}
    records.append(_record("seed-and-twins", "carrier 0", size=1, synthetic=True))
    result = assign_group_splits({"policy": records}, seed=42)
    held = result.held_out_carriers_by_line["policy"]["homeowners"]
    assert held != "carrier 0" and result.assignment["seed-and-twins"] == "train"
    trained = {r.carrier for r in records if result.assignment[r.group_id] == "train"}
    assert held not in trained


def test_a_synthetic_only_carrier_neither_counts_nor_is_held_out_on_a_delivered_split():
    records, splits = _ten_carriers()
    nine = [r for r in records if not r.group_id.startswith("c9-")]
    nine_splits = {g: s for g, s in splits.items() if not g.startswith("c9-")}
    twins = [_record(f"t-{split}", "twin carrier", size=1, synthetic=True) for split in ("train", "val", "test")]
    groups = {"policy": nine + twins}
    twin_splits = {**nine_splits, **{f"t-{s}": s for s in ("train", "val", "test")}}
    result = assign_delivered_splits(groups, twin_splits, hold_out_carriers=True)
    assert result.lines_below_carrier_minimum == {"policy": {"homeowners": 9}}     # the twins' carrier: no count
    assert result.assignment == twin_splits                                      # and nothing moved
    # With a tenth real carrier the line holds one out - never the twins' carrier.
    groups = {"policy": records + twins}
    result = assign_delivered_splits(groups, {**splits, **twin_splits}, hold_out_carriers=True)
    assert result.held_out_carriers_by_line["policy"]["homeowners"] not in ("twin carrier",)
    assert all(result.assignment[f"t-{s}"] == s for s in ("train", "val", "test"))


def test_a_carrier_seen_only_with_its_twins_counts_toward_the_minimum_but_is_never_held_out():
    """Ten real carriers, the tenth's real document delivered in one family with
    its twins: the line names ten carriers, so it holds one out - never that one."""
    from orchestration.pipeline_dag import group_records

    records, splits = _ten_carriers()
    nine = [r for r in records if not r.group_id.startswith("c9-")]
    nine_splits = {g: s for g, s in splits.items() if not g.startswith("c9-")}

    def doc(source_id, carrier, synthetic):
        return SimpleNamespace(source_id=source_id, doc_type="policy", family="seed-9", carrier=carrier,
                               synthetic=synthetic, lob="homeowners")

    # A twin listed first, under a generated carrier: the family takes its seed's.
    (family,) = group_records([doc("t1", "Generated Mutual", True), doc("s9", "carrier 9", False),
                               doc("t2", "Generated Mutual", True)])["policy"]
    assert family.synthetic and family.holds_real and family.carrier == "carrier 9"

    choice = held_out_carrier_per_line(nine + [family], seed=42, split_of={**nine_splits, "seed-9": "train"})
    assert not choice.below_minimum and choice.held == {"homeowners": "carrier 0"}
    # Carrier 0, the cheapest, seen only through such a family: counted, never held out.
    family.carrier = "carrier 0"
    without_c0 = [r for r in records if not r.group_id.startswith("c0-")]
    choice = held_out_carrier_per_line(without_c0 + [family], seed=42, split_of=splits)
    assert choice.held == {"homeowners": "carrier 1"}
    family.holds_real = False                                      # twins alone count for nothing
    family.carrier = "carrier 9"
    choice = held_out_carrier_per_line(nine + [family], seed=42, split_of={**nine_splits, "seed-9": "train"})
    assert choice.below_minimum == {"homeowners": 9} and not choice.held


@pytest.mark.parametrize("placeholder", ["N/A", "n a", "Unknown", "none", "TBD", "various", "", "  "])
def test_a_placeholder_carrier_neither_counts_nor_is_held_out(placeholder):
    records, splits = _ten_carriers()
    nine = [r for r in records if not r.group_id.startswith("c9-")]
    stand_in = [_record(f"p-{split}", placeholder, size=1) for split in ("train", "val", "test")]
    stand_in_splits = {f"p-{s}": s for s in ("train", "val", "test")}
    nine_splits = {g: s for g, s in splits.items() if not g.startswith("c9-")}
    choice = held_out_carrier_per_line(nine + stand_in, seed=42, split_of={**nine_splits, **stand_in_splits})
    assert choice.below_minimum == {"homeowners": 9} and not choice.held
    cheapest = [_record(f"p-{split}", placeholder, size=0) for split in ("train", "val", "test")]
    choice = held_out_carrier_per_line(records + cheapest, seed=42, split_of={**splits, **stand_in_splits})
    assert choice.held == {"homeowners": "carrier 0"}                         # never the stand-in


def test_a_doc_type_with_no_line_holds_out_as_before():
    """ACORD and Loss Runs are left as they were: no line, no hold-out, and their
    placeholder carriers and synthetic families read as given."""
    records = [_record(f"a{i}", "N/A" if i == 0 else f"insurer {i}", line=None, doc_type="acord")
               for i in range(12)]
    assert held_out_carrier_per_line(records, seed=42).held == {}
    # Ten carriers only counting the stand-in and the synthetic one, the
    # stand-in the cheapest: an ACORD line holds it out, as it always did.
    lined = [_record(f"l{i}-{s}", "N/A" if i == 0 else f"insurer {i}", line="commercial_auto",
                     size=0 if i == 0 else 1, doc_type="acord", synthetic=(i == 1))
             for i in range(10) for s in ("train", "val", "test")]
    splits = {r.group_id: r.group_id.rsplit("-", 1)[1] for r in lined}
    assert held_out_carrier_per_line(lined, seed=42, split_of=splits).held == {"commercial_auto": "N/A"}


# --------------------------------------------------------------------------
# Twin cap
# --------------------------------------------------------------------------

def _doc(source_id, family, synthetic=True, mode=None):
    return SimpleNamespace(source_id=source_id, family=family, synthetic=synthetic, render_mode=mode)


def test_at_most_twenty_twins_of_one_seed_train_per_render_mode():
    assignment = GroupSplitAssignment(assignment={"f": "train", "v": "val"})
    docs = ([_doc("f-orig", "f", synthetic=False)]
            + [_doc(f"f-{i}", "f") for i in range(25)]
            + [_doc(f"f-s{i}", "f", mode="scanned") for i in range(22)]
            + [_doc(f"v-{i}", "v") for i in range(30)])
    kept = cap_twins(docs, assignment, seed=42)
    ids = {d.source_id for d in kept}
    assert MAX_TWINS_PER_SEED == 20
    assert "f-orig" in ids                                                   # real documents always stay
    assert sum(1 for i in range(25) if f"f-{i}" in ids) == 20                # digital twins capped
    assert sum(1 for i in range(22) if f"f-s{i}" in ids) == 20               # scanned twins capped apart
    assert all(f"v-{i}" in ids for i in range(30))                           # validation never capped
    assert assignment.twins_dropped == {"f": 7} and assignment.twin_cap == 20
    again = cap_twins(docs, GroupSplitAssignment(assignment={"f": "train", "v": "val"}), seed=42)
    assert {d.source_id for d in again} == ids                               # the same twins every build


# --------------------------------------------------------------------------
# Carrier and twin index travel from the bundle to the row
# --------------------------------------------------------------------------

def test_the_import_records_carrier_and_twin_index():
    from data_pipeline.ingestion.import_labeled_pdfs import twin_index

    assert twin_index({"synthetic": True, "sample": "7"}) == 7
    assert twin_index({"synthetic": True, "twin_index": 3}) == 3
    assert twin_index({"synthetic": False, "sample": "7"}) is None


def test_the_delivered_carrier_outranks_the_one_the_label_names():
    from orchestration.pipeline_dag import _document_carrier

    label = {"carrier": {"company_name": {"raw": "Label Carrier Inc", "parsed": "Label Carrier Inc", "page_ref": [1]}}}
    assert _document_carrier({"carrier": "Delivered Carrier Co"}, label) != _document_carrier({}, label)
    assert _document_carrier({}, label)


def test_every_row_carries_carrier_and_twin_index():
    from data_pipeline.dataset_builder.build_jsonl import build_corpus
    from tests.test_dataset_builder import _documents

    docs = _documents(6)
    for index, doc in enumerate(docs):
        doc.carrier = f"carrier-{index % 2}"
        doc.synthetic, doc.twin_index = (True, index) if index in (1, 2) else (False, None)   # train twins
    assignment = GroupSplitAssignment(assignment={d.family: "train" for d in docs[:4]}
                                      | {docs[4].family: "val", docs[5].family: "test"})
    rows = [row for split_rows in build_corpus(docs, assignment).rows_by_split.values() for row in split_rows]
    by_id = {doc.source_id: doc for doc in docs}
    assert rows
    for row in rows:
        doc = by_id[row["source_id"]]
        assert row["carrier"] == doc.carrier and row["twin_index"] == doc.twin_index
    assert {row["twin_index"] for row in rows if not by_id[row["source_id"]].synthetic} == {None}


def test_the_run_manifest_records_the_split_policy():
    from orchestration.pipeline_dag import split_policy
    from registry_utils.models import DataStats

    policy = split_policy({"held_out_carriers_by_line": {"policy": {"homeowners": "acme"}},
                           "single_carrier_lines": {"policy": ["motorcycle"]}, "twin_cap": 20,
                           "twins_dropped": {"f": 5, "g": 2}})
    stats = DataStats(train_examples=1, val_examples=1, test_examples=1, split_policy=policy)
    assert stats.split_policy["held_out_carriers_by_line"]["policy"]["homeowners"] == "acme"
    assert stats.split_policy["twins_dropped"] == 7


# --------------------------------------------------------------------------
# Evaluation reports held-out carriers apart
# --------------------------------------------------------------------------

def test_held_out_carrier_documents_are_a_reported_subset_with_their_own_match():
    from evaluation.gating import GATING_METRICS
    from evaluation.run_eval import build_report, subset_of

    assert "held_out_carrier" in subset_of({"doc_type": "lossrun", "held_out_carrier": True})
    golden = {"insured_name": "A", "policy_number": "P-1"}
    seen = {"source_id": "s1", "doc_type": "lossrun", "modality_mode": "ocr_plus_image", "page_count": 1}
    unseen = {**seen, "source_id": "s2", "held_out_carrier": True}
    metrics = build_report("v", [(golden, golden, seen), (golden, {"insured_name": "A"}, unseen)]).gate_metrics()
    assert metrics["held_out_carrier_match"] == 0.5
    assert "held_out_carrier_match" not in GATING_METRICS                    # reported, never gated


def test_after_freezing_a_carrier_held_out_of_a_line_is_watched_in_that_line(caplog):
    from orchestration.pipeline_dag import warn_on_held_out_carriers

    manifest = {"held_out_carriers": {}, "held_out_carriers_by_line": {"policy": {"homeowners": "acme"}}}
    other_line = SimpleNamespace(carrier="acme", doc_type="policy", lob="personal_auto")
    warn_on_held_out_carriers(manifest, [other_line])
    assert "held out" not in caplog.text                                     # it trains in its other lines
    same_line = SimpleNamespace(carrier="acme", doc_type="policy", lob="homeowners")
    warn_on_held_out_carriers(manifest, [same_line])
    assert "acme (homeowners)" in caplog.text


def test_the_render_mode_comes_from_the_manifest_or_the_gold(tmp_path):
    import json as _json

    from data_pipeline.ingestion.prepare_bundles import render_mode

    gold = tmp_path / "g.json"
    gold.write_text(_json.dumps({"fideon:provenance": {"mode": "scanned_from_digital"}}), encoding="utf-8")
    assert render_mode({"mode": "Native"}, gold) == "native"                 # the manifest column first
    assert render_mode({}, gold) == "scanned_from_digital"                   # else the gold's provenance
    gold.write_text("{}", encoding="utf-8")
    assert render_mode({}, gold) is None and render_mode({}, tmp_path / "missing.json") is None


def test_forty_twins_of_one_seed_in_two_modes_all_train():
    """SPEC_21 §10: the cap is per seed AND render mode."""
    assignment = GroupSplitAssignment(assignment={"s": "train"})
    docs = ([_doc(f"n{i}", "s", mode="native") for i in range(20)]
            + [_doc(f"c{i}", "s", mode="scanned_from_digital") for i in range(20)])
    assert len(cap_twins(docs, assignment, seed=42)) == 40 and not assignment.twins_dropped
