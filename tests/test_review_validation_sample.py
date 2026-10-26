"""Review item #6 (part): the validation sample stratified by empty targets,
capped per family, and weighted to the training mode mix under the fast
profile (evaluation.validation_sample). Synthetic rows only."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from evaluation.validation_sample import SampleOptions, sample_options, target_state, validation_sample

MODES = ("ocr_plus_image", "noisy_ocr_image", "image_only")


def _rows(*, empty_every: int = 0, big_family_windows: int = 0):
    """3 lines x documents x windows x 3 modes; every ``empty_every``-th window
    answers nothing; one family of ``big_family_windows`` windows besides."""
    rows = []

    def add(source_id, lob, family, window, empty):
        answer = {"policy_info": {"policy_number": {"v": None if empty else f"P-{source_id}"}}}
        for mode in MODES:
            rows.append({"source_id": source_id, "group_id": family, "lob": [lob], "modality_mode": mode,
                         "sections": "decl", "window_pages": [window], "doc_type": "policy",
                         "messages": [{"role": "user", "content": "x"},
                                      {"role": "assistant", "content": json.dumps(answer)}]})

    n = 0
    for lob, docs in (("homeowners", 300), ("personal_auto", 100), ("motorcycle", 20)):
        for i in range(docs):
            for window in (1, 2):
                n += 1
                add(f"{lob}-{i}", lob, f"{lob}-{i}", window, bool(empty_every) and n % empty_every == 0)
    for window in range(big_family_windows):
        add("homeowners-big", "homeowners", "homeowners-big", 100 + window, False)
    return rows


def test_the_shipped_config_draws_the_sample_as_before():
    options = sample_options()
    assert options == SampleOptions() and not options.active
    rows = _rows(empty_every=5)
    assert validation_sample(rows, 400) == validation_sample(rows, 400, SampleOptions())


def test_target_state_reads_compact_and_full_answers():
    def row(answer):
        return {"messages": [{"role": "assistant", "content": json.dumps(answer)}]}

    assert target_state(row({"a": {"v": None, "p": [1]}, "b": []})) == "empty"
    assert target_state(row({"a": {"v": "x"}})) == "stated"
    assert target_state(row({"a": {"raw": "1", "parsed": 1, "page_ref": [2]}})) == "stated"
    assert target_state(row({"t": [{"a": {"parsed": None}}]})) == "empty"
    assert target_state({"messages": []}) == "unknown"


def test_empty_targets_are_sampled_in_their_share_of_every_stratum():
    rows = _rows(empty_every=5)                                          # a fifth of the windows answer nothing
    sample = validation_sample(rows, 400, SampleOptions(by_empty_target=True))
    assert len(sample) == 400
    for lob in ("homeowners", "personal_auto"):
        for mode in MODES:
            stratum = [r for r in sample if r["lob"] == [lob] and r["modality_mode"] == mode]
            empty = sum(target_state(r) == "empty" for r in stratum)
            assert abs(empty - len(stratum) / 5) <= 1
    assert {(r["lob"][0], r["modality_mode"], target_state(r)) for r in sample} == {
        (lob, mode, state) for lob in ("homeowners", "personal_auto", "motorcycle") for mode in MODES
        for state in ("empty", "stated")}


def test_no_family_fills_the_sample_past_the_cap():
    rows = _rows(big_family_windows=200)                                 # 600 rows of one policy
    uncapped = Counter(r["group_id"] for r in validation_sample(rows, 400, SampleOptions(by_empty_target=True)))
    assert uncapped["homeowners-big"] > 20
    sample = validation_sample(rows, 400, SampleOptions(rows_per_family_max=4))
    assert len(sample) == 400
    assert max(Counter(r["group_id"] for r in sample).values()) <= 4


def test_the_fast_profile_s_training_mix_weights_image_only():
    from training.data_mix import configured_modality_mix

    shares = configured_modality_mix()                                   # fast: image_only 0.7
    options = sample_options({"validation_sample_mode_shares": "training"})
    assert options.mode_shares == shares
    sample = validation_sample(_rows(), 400, options)
    assert len(sample) == 400
    by_mode = Counter(r["modality_mode"] for r in sample)
    for mode in MODES:
        assert abs(by_mode[mode] - 400 * shares[mode]) <= 3              # one row per line stratum, at most
    assert {r["lob"][0] for r in sample if r["modality_mode"] == "noisy_ocr_image"} == {
        "homeowners", "personal_auto", "motorcycle"}                     # no stratum drops out


def test_a_mode_with_no_rows_gives_its_share_to_the_others():
    rows = [r for r in _rows() if r["modality_mode"] != "image_only"]
    sample = validation_sample(rows, 400, SampleOptions(mode_shares={"ocr_plus_image": 0.25,
                                                                     "noisy_ocr_image": 0.05,
                                                                     "image_only": 0.7}))
    by_mode = Counter(r["modality_mode"] for r in sample)
    assert len(sample) == 400 and abs(by_mode["ocr_plus_image"] - 400 * 0.25 / 0.3) <= 3


def test_the_options_still_pick_the_same_rows_whatever_their_order():
    rows = _rows(empty_every=5, big_family_windows=50)
    options = SampleOptions(by_empty_target=True, rows_per_family_max=4,
                            mode_shares={"ocr_plus_image": 0.25, "noisy_ocr_image": 0.05, "image_only": 0.7})
    first = validation_sample(rows, 400, options)
    again = validation_sample(list(reversed(rows)), 400, options)

    def identity(r):
        return r["source_id"], r["modality_mode"], tuple(r["window_pages"])

    assert sorted(map(identity, first)) == sorted(map(identity, again))


def test_a_misconfigured_option_is_refused():
    with pytest.raises(ValueError, match="rows_per_family_max"):
        sample_options({"validation_rows_per_family_max": 0})
    with pytest.raises(ValueError, match="mode_shares"):
        sample_options({"validation_sample_mode_shares": "fast"})
