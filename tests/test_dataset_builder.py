"""IMPL-05 — splitting, modality expansion, corruption, and the corpus manifest.

The split-leakage test here is one of the four that guard silent failures: if
expansion moved before the split, every eval number in the project would be
inflated and nothing else would notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.constants import MODALITY_MODES, split_ratio_for
from data_pipeline.corpus_manifest import (
    build_manifest,
    compute_alias_coverage,
    compute_lob_coverage,
    count_confusable_examples,
)
from data_pipeline.dataset_builder.build_jsonl import (
    CorpusBuildError,
    SourceDocument,
    build_corpus,
    expand_document,
    train_rows_by_epoch,
    write_jsonl,
)
from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr
from data_pipeline.dataset_builder.split_groups import (
    GroupRecord,
    SplitError,
    assert_no_leakage,
    assert_single_tenant,
    assert_synthetic_is_train_only,
    assign_group_splits,
)


def _groups(source_ids_by_doc_type, **over):
    """One group per document, which is the v1 behaviour and the right baseline
    for the tests that are about split ratios rather than about families."""
    return {
        doc_type: [
            GroupRecord(group_id=sid, doc_type=doc_type, source_ids=[sid], **over)
            for sid in sids
        ]
        for doc_type, sids in source_ids_by_doc_type.items()
    }


def assign_splits(source_ids_by_doc_type, **kwargs):
    """Shim so the ratio/stability tests keep reading as they did."""
    return assign_group_splits(_groups(source_ids_by_doc_type), **kwargs)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _documents(n: int = 12) -> list[SourceDocument]:
    """Flat Loss Run documents: exactly one row per input mode each.

    The corpus rules these tests pin — one draw per epoch, the split, leakage —
    are properties of the build, not of a document type. A flat type keeps the
    row arithmetic one-per-mode; canonical policies, which expand into windows,
    have their own tests below.
    """
    golden = json.loads((FIXTURES / "golden/lossrun_0001.golden.json").read_text(encoding="utf-8"))
    ocr = (FIXTURES / "ocr/lossrun_0001_page_1.md").read_text(encoding="utf-8")
    return [
        SourceDocument(
            source_id=f"lossrun_{i:04d}", doc_type="lossrun",
            golden_label=golden, ocr_pages=[ocr],
            image_paths=[f"processed/default/lossrun/lossrun_{i:04d}/page_1.png"],
            tenant_id="default",
        )
        for i in range(1, n + 1)
    ]


def _policy_documents(n: int = 6, *, pages: int = 1, lob: str | None = "workers_comp"):
    """Canonical policies. Page 1 carries the fixture's OCR; any further page is
    a copy of it, so the router has a declarations page to find."""
    golden = json.loads((FIXTURES / "golden/policy_0001.golden.json").read_text(encoding="utf-8"))
    ocr = (FIXTURES / "ocr/policy_0001_page_1.md").read_text(encoding="utf-8")
    return [
        SourceDocument(
            source_id=f"policy_{i:04d}", doc_type="policy",
            golden_label=golden, ocr_pages=[ocr] * pages,
            image_paths=[
                f"processed/default/policy/policy_{i:04d}/page_{p}.png" for p in range(1, pages + 1)
            ],
            lob=lob, tenant_id="default",
        )
        for i in range(1, n + 1)
    ]


# --------------------------------------------------------------------------
# Splitting — the ordering rule
# --------------------------------------------------------------------------

def test_split_covers_every_document_exactly_once():
    docs = _documents(12)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    assert len(assignment.assignment) == 12
    assert set(assignment.assignment.values()) == {"train", "val", "test"}


def test_split_is_deterministic_under_the_same_seed():
    ids = [f"policy_{i:04d}" for i in range(1, 21)]
    assert assign_splits({"policy": ids}, seed=42).assignment == assign_splits({"policy": ids}, seed=42).assignment


def test_growth_does_not_reshuffle_documents_at_a_fixed_ratio():
    """Threshold assignment: a document's hash is fixed, so with fixed edges the
    corpus can grow without migrating anything already assigned.

    Slicing would fail this — every boundary shifts as the corpus grows, moving
    documents near it between splits.
    """
    from common.constants import SplitRatio

    ratio = SplitRatio(0.80, 0.10, 0.10)
    first = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 101)]}, ratios=ratio)
    second = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 501)]}, ratios=ratio)
    for source_id, split in first.assignment.items():
        assert second.assignment[source_id] == split


def test_crossing_a_volume_band_does_reassign_some_documents():
    """The honest limit of stability.

    Ratios scale with volume by design (arch §8), so crossing a band moves the
    thresholds and reassigns a minority of documents. Cross-version comparability
    comes from the **frozen golden eval set**, which is versioned separately and
    held constant — not from the corpus test split.
    """
    pilot = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 101)]})     # 70/18/12
    at_scale = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 301)]})  # 75/15/15

    assert pilot.ratios_by_doc_type["policy"] != at_scale.ratios_by_doc_type["policy"]
    moved = [sid for sid, split in pilot.assignment.items() if at_scale.assignment[sid] != split]
    assert moved, "crossing a band is expected to reassign some documents"
    assert len(moved) < len(pilot.assignment) * 0.25, "but only a minority of them"


def test_empty_split_is_repaired_at_small_volume():
    """At very small N a split can come up empty by chance. An empty test split
    cannot be evaluated at all, so one document is moved to fill it."""
    assignment = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 21)]})
    assert all(assignment.counts_by_doc_type["policy"][s] >= 1 for s in ("train", "val", "test"))


def test_split_ratio_scales_with_volume():
    """Pilot volume gets ~70/18/12; at scale 80/10/10 (arch §8)."""
    pilot = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 31)]})
    assert pilot.ratios_by_doc_type["policy"]["train"] == pytest.approx(0.70)

    at_scale = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 1501)]})
    assert at_scale.ratios_by_doc_type["policy"]["train"] == pytest.approx(0.80)


def test_every_split_gets_at_least_one_document():
    """A corpus with an empty test split cannot be evaluated at all."""
    assignment = assign_splits({"policy": [f"policy_{i:04d}" for i in range(1, 6)]})
    counts = assignment.counts_by_doc_type["policy"]
    assert counts["train"] >= 1 and counts["val"] >= 1 and counts["test"] >= 1


def test_types_are_split_independently():
    """A global split on an unbalanced corpus can leave a type with no test
    documents, and its accuracy then silently unmeasured."""
    assignment = assign_splits({
        "policy": [f"policy_{i:04d}" for i in range(1, 21)],
        "lossrun": [f"lossrun_{i:04d}" for i in range(1, 6)],
    })
    for doc_type in ("policy", "lossrun"):
        assert all(assignment.counts_by_doc_type[doc_type][s] >= 1 for s in ("train", "val", "test"))


# --------------------------------------------------------------------------
# Expansion, and the leakage guard
# --------------------------------------------------------------------------

def test_one_document_expands_to_three_modality_rows():
    rows, _details = expand_document(_documents(1)[0], "train")
    assert len(rows) == 3
    assert {r["modality_mode"] for r in rows} == set(MODALITY_MODES)
    assert all(r["split"] == "train" for r in rows)


def test_image_only_row_carries_no_ocr_text():
    """The guarantee is "no OCR", not "no text at all": each page still carries
    its `<page N of M>` marker, which is document metadata rather than content
    read off the page."""
    rows, _ = expand_document(_documents(1)[0], "train")
    image_only = next(r for r in rows if r["modality_mode"] == "image_only")
    clean = next(r for r in rows if r["modality_mode"] == "ocr_plus_image")

    texts = [b["text"] for b in image_only["messages"][1]["content"] if b["type"] == "text"]
    assert all(t.startswith("<page ") and "\n" not in t for t in texts)

    ocr_body = "".join(
        b["text"] for b in clean["messages"][1]["content"] if b["type"] == "text"
    ).replace("<page 1 of 1>", "").strip()
    assert ocr_body and ocr_body not in "".join(texts)


def test_noisy_row_keeps_the_correct_golden_target():
    """The mismatch IS the training signal: the model can only get it right by
    reading the page (arch §5, §6)."""
    rows, _ = expand_document(_documents(1)[0], "train")
    clean = next(r for r in rows if r["modality_mode"] == "ocr_plus_image")
    noisy = next(r for r in rows if r["modality_mode"] == "noisy_ocr_image")

    assert noisy["messages"][2]["content"] == clean["messages"][2]["content"]   # same target
    clean_text = clean["messages"][1]["content"][-1]["text"]
    noisy_text = noisy["messages"][1]["content"][-1]["text"]
    assert noisy_text != clean_text                                            # different input


def test_noisy_and_clean_rows_share_one_system_prompt():
    """The noise lives in the data, not the instruction — at inference nothing
    announces that OCR is bad."""
    rows, _ = expand_document(_documents(1)[0], "train")
    clean = next(r for r in rows if r["modality_mode"] == "ocr_plus_image")
    noisy = next(r for r in rows if r["modality_mode"] == "noisy_ocr_image")
    assert clean["messages"][0] == noisy["messages"][0]


def test_no_source_id_crosses_splits():
    """THE leakage test. If expansion moved before the split, this fails."""
    docs = _documents(12)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    result = build_corpus(docs, assignment)
    assert_no_leakage(assignment, result.all_rows)

    by_source: dict[str, set[str]] = {}
    for row in result.all_rows:
        by_source.setdefault(row["source_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_source.values())


def test_leakage_is_detected_when_it_is_injected():
    """Mutation check — the guard must actually fire."""
    docs = _documents(6)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    result = build_corpus(docs, assignment)

    rows = list(result.all_rows)
    rows[0] = {**rows[0], "split": "test" if rows[0]["split"] != "test" else "train"}
    with pytest.raises(SplitError, match="LEAKAGE"):
        assert_no_leakage(assignment, rows)


def test_cross_tenant_mixing_is_refused():
    """The one live tenancy rule: corpus composition IS training data."""
    rows = [
        {"source_id": "policy_0001", "split": "train", "tenant_id": "broker_a"},
        {"source_id": "policy_0002", "split": "train", "tenant_id": "broker_b"},
    ]
    with pytest.raises(SplitError, match="multiple tenants"):
        assert_single_tenant(rows)


def test_rebuild_with_the_same_seed_is_byte_identical():
    """A corpus that changes between builds cannot be compared across model
    versions."""
    docs = _documents(8)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    first = write_jsonl(build_corpus(docs, assignment, seed=42).all_rows)
    second = write_jsonl(build_corpus(docs, assignment, seed=42).all_rows)
    assert first == second


def test_every_epoch_file_holds_every_train_document_exactly_once():
    """The concatenated epoch files are the whole run. A document missing from
    one, or in one twice, trains on something the epoch count does not describe."""
    docs = _documents(30)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    built = build_corpus(docs, assignment)

    by_epoch = train_rows_by_epoch(built)
    train_ids = {r["source_id"] for r in built.rows_by_split["train"]}
    assert sorted(by_epoch) == [1, 2, 3, 4]
    for rows in by_epoch.values():
        assert sorted(r["source_id"] for r in rows) == sorted(train_ids)


def test_an_epoch_missing_a_document_is_refused():
    docs = _documents(30)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    built = build_corpus(docs, assignment)
    dropped = next(r for r in built.rows_by_split["train"] if r["epoch"] == 2)
    built.rows_by_split["train"].remove(dropped)

    with pytest.raises(CorpusBuildError, match="epoch 2"):
        train_rows_by_epoch(built)


def test_val_and_test_keep_all_three_variants():
    """So image-only and noisy-OCR accuracy are measured on the full eval
    population, not a sample of it."""
    docs = _documents(30)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    sampled = build_corpus(docs, assignment)

    for split in ("val", "test"):
        rows = sampled.rows_by_split[split]
        per_source: dict[str, set[str]] = {}
        for row in rows:
            per_source.setdefault(row["source_id"], set()).add(row["modality_mode"])
        assert all(modes == set(MODALITY_MODES) for modes in per_source.values())


# --------------------------------------------------------------------------
# Corruption
# --------------------------------------------------------------------------

def test_corruption_is_seeded_and_always_does_something():
    """A 'noisy' row identical to the clean one teaches nothing, and would
    silently dilute the arbitration signal below its intended share."""
    text = (FIXTURES / "ocr/lossrun_0001_page_1.md").read_text(encoding="utf-8")
    a = corrupt_ocr(text, "lossrun_0001", seed=42)
    b = corrupt_ocr(text, "lossrun_0001", seed=42)
    c = corrupt_ocr(text, "lossrun_0001", seed=99)

    assert a.text == b.text
    assert a.text != c.text
    assert a.was_corrupted and c.was_corrupted
    assert a.text != text


def test_corruptions_are_recorded_for_eval_slicing():
    """So ocr_arbitration_accuracy scores on documents where OCR and image
    genuinely disagree, rather than on the whole corpus."""
    text = (FIXTURES / "ocr/lossrun_0001_page_1.md").read_text(encoding="utf-8")
    result = corrupt_ocr(text, "lossrun_0001", seed=42)
    assert result.corruptions and result.details


# --------------------------------------------------------------------------
# Manifest and coverage
# --------------------------------------------------------------------------

def test_lob_under_coverage_names_the_thin_values():
    labels = [{"line_of_business": ["workers_comp"]}] * 19 + [{"line_of_business": ["property"]}]
    shares, warnings = compute_lob_coverage(labels)
    assert shares["workers_comp"] > 0.9
    assert warnings and "property" in warnings[0]


def test_alias_under_coverage_names_the_thin_labels():
    """95 documents saying 'Named Insured' and 5 saying 'Applicant' yield a model
    shaky on 'Applicant' however large the corpus is."""
    provenance = {f"policy_{i:04d}": {"insured_name": "Named Insured"} for i in range(1, 20)}
    provenance["policy_0020"] = {"insured_name": "Applicant"}
    counts, warnings = compute_alias_coverage(provenance)

    assert counts["insured_name"]["Named Insured"] == 19
    assert counts["insured_name"]["Applicant"] == 1
    assert warnings and "Applicant" in warnings[0]


def test_alias_coverage_is_quiet_when_balanced():
    provenance = {
        f"policy_{i:04d}": {"insured_name": "Applicant" if i % 2 else "Named Insured"}
        for i in range(1, 21)
    }
    _counts, warnings = compute_alias_coverage(provenance)
    assert not warnings


def test_zero_confusable_examples_is_reported_as_a_defect():
    """Without them the model learns the mapping but never the boundary."""
    labels = {"policy_0001": {"insured_name": "Rivera Fabrication LLC"}}
    count, warnings = count_confusable_examples(labels, "policy")
    assert count == 0
    assert warnings and "corpus defect" in warnings[0]


def test_confusable_co_occurrence_is_counted():
    labels = {
        "policy_0001": {
            "insured_name": "Rivera Fabrication LLC",
            "producer": "Hanover Risk Partners",
        }
    }
    count, warnings = count_confusable_examples(labels, "policy")
    assert count == 1
    assert not warnings


def test_manifest_records_every_pin_and_measurement():
    docs = _documents(12)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    result = build_corpus(docs, assignment)

    manifest, report = build_manifest(
        corpus_version="v1",
        tenant_id="default",
        rows_by_split=result.rows_by_split,
        golden_labels_by_source={d.source_id: d.golden_label for d in docs},
        provenance_by_source={d.source_id: {"insured_name": "Applicant"} for d in docs},
        split_assignment=assignment.as_dict(),
        ocr_environment={"mineru_version": "1.4.2", "ocr_device": "cuda"},
        doc_types=["policy"],
        git_commit="deadbee",
    )

    # Reproducibility pins — a change to any forces a rebuild and retrain.
    assert manifest["mineru_version"] == "1.4.2"
    assert manifest["ocr_device"] == "cuda"
    assert manifest["prompt_template_version"]
    assert manifest["schema_versions"]["policy"]
    assert manifest["builder_git_commit"] == "deadbee"
    assert manifest["seed"] == 42

    # Composition and coverage. Not a fixed number any more: under arch v2.1
    # §6.1 a train document contributes one row per EPOCH (four, always
    # materialized) while val and test keep all three modality regimes so
    # image-only accuracy is measured on the full eval population.
    from data_pipeline.dataset_builder.sample_modes import EPOCH_FILES

    expected = (
        len(result.rows_by_split["train"])
        + len(result.rows_by_split["val"])
        + len(result.rows_by_split["test"])
    )
    assert manifest["total_rows"] == expected
    assert len(result.rows_by_split["train"]) % EPOCH_FILES == 0, (
        "every train document contributes exactly one row per epoch"
    )
    assert set(manifest["modality_mix"]) == set(MODALITY_MODES)
    assert manifest["lob_coverage_target"] == 0.20
    assert "alias_coverage" in manifest

    # De-identification is blocked — recorded honestly, not omitted (IMPL-05 §1).
    assert manifest["deidentified"] is False
    assert manifest["image_redaction"] == "unresolved"
    assert isinstance(report.warnings, list)


def test_the_corruption_budget_is_per_document_not_per_page():
    """`corrupt_ocr` spends its budget per call, so calling it once per page
    would multiply the noise by the page count. A 20-page policy with every page
    mangled teaches "OCR is always garbage" rather than arbitration — and would
    degrade the 50% ocr_plus_image regime, which never sees noise at all."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    pages = [f"Policy ABC{i}234 issued 2026-01-0{i} amount 1{i},500.00" for i in range(1, 21)]
    corrupted, details = corrupt_ocr_pages(pages, "policy_0001", max_corruptions=2)

    assert len(corrupted) == len(pages)
    changed = sum(a != b for a, b in zip(pages, corrupted, strict=True))
    assert 1 <= changed <= 2, f"{changed} of 20 pages corrupted — the budget is not being honoured"
    assert len(details) <= 2


def test_a_noisy_document_is_never_identical_to_the_clean_one():
    """A noisy row that matches its clean twin teaches nothing and silently
    dilutes the 20% arbitration signal."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    for source_id in (f"policy_{i:04d}" for i in range(1, 15)):
        pages = ["Insured Rivera Fabrication LLC", "Policy WC-8842317-01 premium 47250.00"]
        corrupted, details = corrupt_ocr_pages(pages, source_id)
        assert corrupted != pages, f"{source_id} produced an uncorrupted noisy row"
        assert details


def test_page_corruption_is_seeded_and_reproducible():
    """A corpus that changes between builds cannot be compared across versions."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    pages = [f"page {i} Policy ABC1234 amount 12,500.00" for i in range(1, 6)]
    a, da = corrupt_ocr_pages(pages, "policy_0001", seed=42)
    b, db = corrupt_ocr_pages(pages, "policy_0001", seed=42)
    c, _dc = corrupt_ocr_pages(pages, "policy_0002", seed=42)

    assert a == b and da == db
    assert a != c or da != _dc, "different documents must corrupt differently"


def test_the_details_name_the_page_the_noise_landed_on():
    """IMPL-08's ocr_arbitration_accuracy slices on these; a detail that does not
    say where the change happened cannot be matched to a field."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages

    pages = [f"Policy ABC{i}234 amount 1{i},500.00" for i in range(1, 9)]
    _corrupted, details = corrupt_ocr_pages(pages, "policy_0003")
    assert all(d.startswith("page ") for d in details), details


def test_multi_character_confusions_are_reachable():
    """`rn`->`m` and `cl`->`d` are the two most realistic confusions in print,
    and scanning one character at a time left both as dead table entries — so
    the noise the corpus taught arbitration on was narrower than it looked."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import (
        CHAR_CONFUSIONS,
        _confusable_positions,
    )

    multi = [k for k in CHAR_CONFUSIONS if len(k) > 1]
    assert multi, "the table declares no multi-character confusions"
    for source in multi:
        token = f"AB{source}CD"
        assert (2, source) in _confusable_positions(token), f"{source!r} is unreachable"


def test_a_longer_confusion_wins_over_an_overlapping_shorter_one():
    """Applying a single-character rule inside `rn` would consume half of it and
    leave the other half stranded."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import _confusable_positions

    positions = _confusable_positions("Warner")
    assert (2, "rn") in positions
    starts = [i for i, _s in positions]
    assert len(starts) == len(set(starts)), "two rules claim the same position"


def test_every_confusion_entry_is_applied_at_its_full_width():
    """A multi-character source must replace all of its characters, not one."""
    from data_pipeline.dataset_builder.noisy_ocr_augment import CHAR_CONFUSIONS

    token = "Warner"
    index, source = 2, "rn"
    corrupted = token[:index] + CHAR_CONFUSIONS[source] + token[index + len(source):]
    assert corrupted == "Wamer", corrupted


# --------------------------------------------------------------------------
# Group-aware splits (arch v2.1 §8.2)
# --------------------------------------------------------------------------

def _family(group_id, n, doc_type="policy", **over):
    return GroupRecord(
        group_id=group_id,
        doc_type=doc_type,
        source_ids=[f"{group_id}_{i:03d}" for i in range(n)],
        **over,
    )


def test_a_family_never_spans_the_split():
    """The whole reason the unit moved from document to group. One carrier's
    template in train and another copy of it in test measures how well the model
    memorised a layout, and nothing crashes to say so."""
    families = [_family(f"grp{i}", 8) for i in range(20)]
    assignment = assign_group_splits({"policy": families})

    for fam in families:
        splits = {assignment.split_of(fam.group_id)}
        assert len(splits) == 1, f"{fam.group_id} spans {splits}"


def test_leakage_is_detected_on_the_expanded_rows():
    """The assignment is a dict and cannot contain a duplicate. Expansion into
    tasks, windows and modes is what could place them inconsistently, so the
    check runs there."""
    assignment = assign_group_splits({"policy": [_family("grp0", 4)]})
    rows = [
        {"group_id": "grp0", "source_id": "grp0_000", "split": "train"},
        {"group_id": "grp0", "source_id": "grp0_001", "split": "test"},
    ]
    with pytest.raises(SplitError, match="LEAKAGE"):
        assert_no_leakage(assignment, rows)


def test_a_row_without_its_group_is_refused():
    """Every row carries its group, because the group is the unit that must not
    span the split. A row that does not name one cannot be checked at all."""
    assignment = assign_group_splits({"policy": [_family("grp0", 2)]})
    with pytest.raises(SplitError, match="missing group_id"):
        assert_no_leakage(assignment, [{"source_id": "grp0_000", "split": "train"}])


def test_a_line_with_ten_or_more_carriers_holds_exactly_one_out_entirely():
    """'Unseen template' is a different question from 'unseen document', and only
    a carrier held out in full answers it: one per line (decision 2026-10-19)."""
    families = [
        _family(f"grp{i}", 4, carrier=f"carrier_{i % 12}", line="homeowners") for i in range(48)
    ]
    assignment = assign_group_splits({"policy": families})
    held = assignment.held_out_carriers_by_line.get("policy", {})

    assert list(held) == ["homeowners"], held
    for fam in families:
        if fam.carrier == held["homeowners"]:
            assert assignment.split_of(fam.group_id) == "test"
    assert not assignment.held_out_carriers, "held out of its line, not of the doc type"


def test_no_carrier_is_held_out_of_a_line_with_fewer_than_ten():
    """Below ten carriers every carrier trains: one of a few is too large a share
    of the line's layouts to give up."""
    families = [_family(f"grp{i}", 4, carrier=f"carrier_{i % 9}", line="homeowners") for i in range(36)]
    assignment = assign_group_splits({"policy": families})
    assert not assignment.held_out_carriers_by_line.get("policy")
    assert assignment.lines_below_carrier_minimum["policy"] == {"homeowners": 9}
    assert assignment.as_dict()["lines_below_carrier_minimum"] == {"policy": {"homeowners": 9}}


def test_each_line_counts_its_own_carriers():
    ten = [_family(f"h{i}", 3, carrier=f"c{i % 10}", line="homeowners") for i in range(30)]
    nine = [_family(f"m{i}", 3, carrier=f"c{i % 9}", line="motorcycle") for i in range(27)]
    assignment = assign_group_splits({"policy": ten + nine})
    assert set(assignment.held_out_carriers_by_line["policy"]) == {"homeowners"}
    assert assignment.lines_below_carrier_minimum["policy"] == {"motorcycle": 9}


def test_the_largest_carrier_is_not_held_out():
    """The carrier held out is the one costing train and validation least:
    removing the one that contributes most of the training data is not a
    generalisation test."""
    families = [_family(f"big{i}", 20, carrier="dominant", line="homeowners") for i in range(10)]
    families += [_family(f"grp{i}", 2, carrier=f"carrier_{i % 11}", line="homeowners")
                 for i in range(22)]
    assignment = assign_group_splits({"policy": families})
    assert assignment.held_out_carriers_by_line["policy"]["homeowners"] != "dominant"


def test_validation_is_halved_by_group():
    """Fitting the calibrator and setting the review threshold on the same
    documents makes the threshold optimistic — the calibrator has already seen
    the errors the threshold is meant to price (arch v2.1 §5.3-5.4)."""
    families = [_family(f"grp{i}", 3) for i in range(60)]
    assignment = assign_group_splits({"policy": families})

    val_groups = assignment.groups_in("val")
    assert val_groups, "no validation groups to halve"
    halves = {assignment.half_of(g) for g in val_groups}
    assert halves == {"calibration", "threshold"}, f"validation not halved: {halves}"

    for group_id in assignment.groups_in("train"):
        assert assignment.half_of(group_id) is None, "only validation groups carry a half"


def test_synthetic_families_stay_in_train():
    """Their labels are perfect because they were generated from them, so a
    metric scored against one reports how faithfully the generator rendered its
    own input — and reports it as model accuracy (arch v2.1 §4d)."""
    families = [_family(f"real{i}", 4, carrier=f"c{i}") for i in range(10)]
    families += [_family(f"synth{i}", 4, synthetic=True) for i in range(10)]
    assignment = assign_group_splits({"policy": families})

    for fam in families:
        if fam.synthetic:
            assert assignment.split_of(fam.group_id) == "train"


def test_a_synthetic_row_outside_train_is_refused():
    rows = [{"group_id": "s0", "source_id": "s0_000", "split": "test", "synthetic": True}]
    with pytest.raises(SplitError, match="synthetic"):
        assert_synthetic_is_train_only(rows)


def test_ratios_follow_document_count_not_group_count():
    """The §8.2 volume bands describe how much data exists. One group of forty
    certificates is still forty documents of training signal, and treating it as
    a single unit would pick the pilot ratios for a corpus well past pilot."""
    few_big = {"policy": [_family(f"grp{i}", 40) for i in range(8)]}   # 320 documents
    many_small = {"policy": [_family(f"grp{i}", 1) for i in range(8)]}  # 8 documents

    big = assign_group_splits(few_big).ratios_by_doc_type["policy"]
    small = assign_group_splits(many_small).ratios_by_doc_type["policy"]
    assert big != small, "group count, not document count, drove the ratio band"


def test_assignment_is_stable_as_the_corpus_grows():
    """A group that was in test stays in test, so eval numbers stay comparable
    without reshuffling on every ingest."""
    first = [_family(f"grp{i}", 3) for i in range(40)]
    grown = first + [_family(f"grp{i}", 3) for i in range(40, 60)]

    before = assign_group_splits({"policy": first}, ratios=split_ratio_for(500))
    after = assign_group_splits({"policy": grown}, ratios=split_ratio_for(500))

    moved = [
        f.group_id for f in first
        if before.split_of(f.group_id) != after.split_of(f.group_id)
    ]
    assert not moved, f"{len(moved)} group(s) moved on growth: {moved[:5]}"


# --------------------------------------------------------------------------
# Per-epoch mode sampling (arch v2.1 §6.1)
# --------------------------------------------------------------------------

def test_each_document_appears_once_per_epoch():
    """v1 expanded every document into three rows, so a "3 epoch" run was nine
    passes over the corpus and the manifest's epoch count described something
    other than what ran."""
    from data_pipeline.dataset_builder.sample_modes import sample_modes

    ids = [f"doc_{i:03d}" for i in range(200)]
    assignment = sample_modes(ids, epochs=4)

    for source_id in ids:
        assert len(assignment.modes_seen_by(source_id)) == 4


def test_the_realised_mix_approaches_the_target_not_thirty_three_each():
    """The v1 three-rows-per-document expansion produced 33/33/33, and a
    downstream sampler discarded rows to correct it — throwing away labelled data
    to fix a shape problem."""
    from common.constants import MODALITY_MIX
    from data_pipeline.dataset_builder.sample_modes import assert_mix_is_close, sample_modes

    assignment = sample_modes([f"doc_{i:04d}" for i in range(500)], epochs=4)
    assert_mix_is_close(assignment)

    realised = assignment.realised_mix()
    assert abs(realised["ocr_plus_image"] - MODALITY_MIX["ocr_plus_image"]) < 0.05
    assert realised["ocr_plus_image"] > realised["image_only"] > realised["noisy_ocr_image"]


def test_a_document_is_shown_in_more_than_one_regime_over_a_run():
    """The whole point of per-epoch sampling: a document only ever seen with
    clean OCR teaches nothing about arbitration."""
    from data_pipeline.dataset_builder.sample_modes import sample_modes

    assignment = sample_modes([f"doc_{i:03d}" for i in range(200)], epochs=4)
    varied = [
        sid for sid in {s for s, _ in assignment.draws}
        if len(set(assignment.modes_seen_by(sid))) > 1
    ]
    assert len(varied) > 100, "epochs are drawing the same mode every time"


def test_mode_draws_are_stable_as_the_corpus_grows():
    """Adding a document must not change the regime an existing one was shown
    in, or every rebuild reshuffles what the model already learned."""
    from data_pipeline.dataset_builder.sample_modes import sample_modes

    before = sample_modes([f"doc_{i:03d}" for i in range(50)])
    after = sample_modes([f"doc_{i:03d}" for i in range(80)])

    for key, mode in before.draws.items():
        assert after.draws[key] == mode


def test_a_mix_that_does_not_sum_to_one_is_refused():
    """Whichever regime falls off the end of the cumulative thresholds is
    silently starved."""
    from data_pipeline.dataset_builder.sample_modes import ModeSamplingError, sample_modes

    with pytest.raises(ModeSamplingError, match="sums to"):
        sample_modes(["a"], mix={"ocr_plus_image": 0.5, "noisy_ocr_image": 0.2, "image_only": 0.2})


# --------------------------------------------------------------------------
# Cap checking (arch v2.1 §7a)
# --------------------------------------------------------------------------

def test_an_over_cap_row_is_rejected_not_truncated():
    """A truncated example is not a smaller example, it is a wrong one: cut the
    tail off a Loss Run assistant span and the target becomes JSON that stops
    after eleven claims, so the model is trained to stop after eleven claims."""
    from data_pipeline.dataset_builder.cap_check import CapReport, check_row, estimate_row

    report = CapReport()
    estimate = estimate_row(
        task="extract",
        system_prompt="x" * 4000,
        ocr_pages=["y" * 40000] * 12,
        page_count=12,
        target_json={"claims": [{"n": i} for i in range(200)]},
        doc_type="acord",
    )
    assert not check_row(estimate, source_id="big_001", report=report)
    assert report.rejected and report.accepted == 0


def test_an_over_budget_output_is_rejected_even_when_the_total_fits():
    """The case that actually produces a clipped assistant span — and the one a
    single total-length check would miss."""
    from data_pipeline.dataset_builder.cap_check import check_row, estimate_row

    estimate = estimate_row(
        task="lossrun_totals",
        system_prompt="short",
        ocr_pages=None,
        page_count=1,
        target_json="z" * 60000,
    )
    assert estimate.output_tokens > 1024
    assert not check_row(estimate, source_id="totals_001")


def test_a_normal_row_fits_its_budget():
    from data_pipeline.dataset_builder.cap_check import CapReport, check_row, estimate_row

    report = CapReport()
    estimate = estimate_row(
        task="extract",
        system_prompt="You are an insurance document extraction system. " * 40,
        ocr_pages=["page text " * 300] * 2,
        page_count=2,
        target_json={"insured_name": "Rivera Fabrication LLC", "claims": []},
        doc_type="acord",
    )
    assert check_row(estimate, source_id="ok_001", report=report)
    assert report.accepted == 1 and not report.rejected


def test_the_estimate_is_pessimistic_about_page_cost():
    """Assumes every page fills its pixel budget. Over-estimating costs a
    rejection an operator sees; under-estimating costs silent truncation."""
    from data_pipeline.dataset_builder.cap_check import estimate_visual_tokens

    # >4x: extraction's pixels are halved under the fast format profile.
    assert estimate_visual_tokens(1, "extract") > estimate_visual_tokens(1, "classify") * 4
    assert estimate_visual_tokens(3, "extract") == 3 * estimate_visual_tokens(1, "extract")


def test_the_rejection_warning_says_the_documents_are_not_the_problem():
    from data_pipeline.dataset_builder.cap_check import CapReport, TokenEstimate

    report = CapReport()
    report.reject("d1", "extract", TokenEstimate(task="extract", doc_type="policy"), 24576)
    assert "contribute NOTHING to training" in report.warning()


# --------------------------------------------------------------------------
# Task decomposition (arch v2.1 §7b)
# --------------------------------------------------------------------------

def test_dense_lossrun_pages_get_a_single_page_window():
    """A dense page overflows the OUTPUT budget, not the input one — which is why
    the window shrinks as row density rises."""
    from data_pipeline.dataset_builder.expand_tasks import plan_windows

    assert plan_windows([40] * 6, output_budget=8000) == [[1], [2], [3], [4], [5], [6]]


def test_sparse_lossrun_pages_get_a_three_page_window_with_overlap():
    """Overlap so a row split across a page break is seen whole by one window."""
    from data_pipeline.dataset_builder.expand_tasks import plan_windows

    windows = plan_windows([8] * 6, output_budget=8000)
    assert windows[0] == [1, 2, 3]
    assert windows[1][0] == 3, "windows overlap by one page"


def test_the_planner_shrinks_a_window_the_page_count_alone_would_allow():
    """Three sparse pages holding ninety rows between them still overflow a
    budget sized for sixty-five, and the band table cannot see that."""
    from data_pipeline.dataset_builder.expand_tasks import plan_windows

    assert plan_windows([30, 30, 30], output_budget=4000) == [[1], [2], [3]]


def _lossrun_examples():
    from data_pipeline.dataset_builder.expand_tasks import expand_lossrun

    return expand_lossrun(
        source_id="lr_001",
        golden_label={
            "carrier": "Acme",
            "claims": [{"claim_number": f"C{i}"} for i in range(30)],
            "totals": {"incurred": 184200.0},
        },
        page_markdown=["| a | b |\n|---|---|\n" + "| 1 | 2 |\n" * 10] * 4,
        output_budget=8000,
    )


def test_a_lossrun_with_no_totals_in_its_schema_gets_no_totals_example():
    """The schema declares no totals, so a totals target would always be {}:
    it would teach the model to ignore printed totals."""
    from common.tasks import Task

    tasks = [e.task for e in _lossrun_examples()]
    assert tasks[0] == Task.LOSSRUN_HEADER
    assert Task.LOSSRUN_ROWS in tasks
    assert Task.LOSSRUN_TOTALS not in tasks


def test_a_lossrun_expands_into_header_windows_and_totals(monkeypatch):
    """Printed totals sit at the END of the report and per policy period, not on
    pages 1-2 where v2.0 looked for them (v2.1 correction)."""
    from common.tasks import Task
    from data_pipeline.dataset_builder import expand_tasks

    monkeypatch.setattr(expand_tasks, "lossrun_schema_has_totals", lambda: True)
    examples = _lossrun_examples()
    tasks = [e.task for e in examples]
    assert tasks[0] == Task.LOSSRUN_HEADER
    assert tasks[-1] == Task.LOSSRUN_TOTALS
    assert Task.LOSSRUN_ROWS in tasks

    totals = examples[-1]
    assert totals.pages == [3, 4], "totals read the LAST pages, not the first"


def test_every_lossrun_example_shares_the_documents_identity():
    """All of them share the document's group, so all of them share its split.
    A header in train and its rows in test is the same leak as any other."""
    from data_pipeline.dataset_builder.expand_tasks import expand_lossrun

    examples = expand_lossrun(
        source_id="lr_002",
        golden_label={"carrier": "Acme", "claims": [], "totals": {}},
        page_markdown=["| a |\n|---|\n| 1 |\n"] * 3,
        output_budget=8000,
    )
    assert {e.source_id for e in examples} == {"lr_002"}


def test_a_long_policy_is_read_on_every_page():
    """Every page, chunk by chunk, never selected pages (decision 2026-10-19): a
    200-page policy's windows cover all 200, the declarations leading."""
    from data_pipeline.dataset_builder.expand_tasks import plan_policy_windows

    pages = list(range(1, 201))
    windows = plan_policy_windows(pages, pages_per_window=5, leading=[1, 2, 3])
    assert [p for w in windows for p in w] == pages, "windowing lost a page"
    assert windows[0] == [1, 2, 3], "the declarations lead their own window"
    assert all(len(w) <= 5 for w in windows)


def test_a_run_of_pages_is_not_split_until_it_has_to_be():
    """A table printed across consecutive pages is one table; splitting it at an
    arbitrary boundary hands the model half a table with no header."""
    from data_pipeline.dataset_builder.expand_tasks import plan_policy_windows

    assert plan_policy_windows(
        [1, 2, 3, 10, 11, 12, 40], pages_per_window=4, leading=[1, 2, 3]
    ) == [[1, 2, 3], [10, 11, 12, 40]]
    # A run too long for one window gets windows of its own, never mixed.
    assert plan_policy_windows(
        [1, 10, 11, 12, 13, 14, 40], pages_per_window=3, leading=[1]
    ) == [[1], [10, 11, 12], [13, 14], [40]]


def test_scattered_pages_share_a_window_rather_than_each_paying_a_prompt():
    """Keyword routing returns scattered single pages. A window per page would
    spend a whole system prompt — some eleven thousand tokens — on each one."""
    from data_pipeline.dataset_builder.expand_tasks import plan_policy_windows

    windows = plan_policy_windows([1, 7, 12, 19, 25, 31], pages_per_window=5)
    assert windows == [[1, 7, 12, 19, 25], [31]]


def test_how_many_pages_a_call_holds_depends_on_the_line():
    """The prompt carries the whole canonical schema, so a line with a larger one
    leaves less room for pages. One constant cannot express that."""
    from data_pipeline.dataset_builder.expand_tasks import pages_per_extraction_call

    larger = pages_per_extraction_call("policy", None, "management_liability")
    smaller = pages_per_extraction_call("policy", None, "equipment_breakdown")
    assert 1 <= larger < smaller, (
        f"management_liability {larger} should fit fewer than equipment_breakdown {smaller}"
    )


def test_train_documents_get_one_row_per_epoch_not_one_per_regime():
    """v1 expanded every train document into all three regimes, so a 3-epoch run
    was nine passes. There is no longer any path that does that for train."""
    docs = _documents(30)
    assignment = assign_splits({"lossrun": [d.source_id for d in docs]})
    built = build_corpus(docs, assignment)

    per_doc: dict[str, int] = {}
    for row in built.rows_by_split["train"]:
        per_doc[row["source_id"]] = per_doc.get(row["source_id"], 0) + 1
    assert per_doc and set(per_doc.values()) == {4}


def test_every_acord_form_in_the_corpus_is_pinned_not_just_form_25():
    """The pin is what forces a rebuild when a schema changes. Recording only
    ACORD 25 meant a corpus holding 125s and 140s pinned no version for them, so
    those schemas could change underneath a corpus claiming reproducibility."""
    from data_pipeline.corpus_manifest import _schema_pins

    pins = _schema_pins(["acord", "policy"])

    assert {"acord:25", "acord:125", "acord:140", "policy"} <= set(pins)
    assert all(version for version in pins.values())
    assert "lossrun" not in pins, "a type the corpus does not hold is not pinned"


def test_a_routed_group_does_not_set_the_first_pages_apart():
    """Only the declarations group leads with its first pages. Another group packing
    them apart spent a whole call on pages that may hold none of its tables."""
    from data_pipeline.dataset_builder.expand_tasks import plan_policy_windows

    assert plan_policy_windows([1, 2, 3, 10, 11], pages_per_window=5) == [[1, 2, 3, 10, 11]]


def test_the_declarations_page_is_read_with_the_leading_pages():
    """A policy behind a fax cover sheet has its declarations on page 5; that page
    is read in the same window as pages 1-3, not in a window of its own."""
    from data_pipeline.dataset_builder.policy_windows import plan_windows

    plans = plan_windows("homeowners", [1, 2, 3, 5, 9, 10], declarations_page=5)
    decl = [list(p.pages) for p in plans if p.group == "decl"]
    assert decl[0] == [1, 2, 3, 5]
    # ... and then every other page, as every group reads them.
    assert {page for pages in decl for page in pages} == {1, 2, 3, 5, 9, 10}
