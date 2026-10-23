"""The nine findings of the training and extraction code review, one test each."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from serving.policy_merge import PolicyWindow, merge_policy_windows


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


# 1 - a request naming classic auto is read as personal auto, not refused
def test_a_classic_auto_request_is_served_as_personal_auto():
    from serving.vllm_entrypoint import request_lob

    assert request_lob({"lob": "classic_auto"}) == "personal_auto"
    assert request_lob({"lob": ["Classic_Auto"]}) == ["personal_auto"]
    assert request_lob({"lob": "homeowners"}) == "homeowners"


# 2 - rows and requests written before the merge still read as personal auto
def test_a_classic_auto_row_is_read_as_personal_auto():
    from common.lob import lob_lines, parse_lines, within_lines
    from training.corpus_view import _line_of

    assert lob_lines("classic_auto") == lob_lines("personal_auto")
    # A --lines filter naming personal auto keeps a row written before the merge.
    assert within_lines(["classic_auto"], frozenset({"personal_auto"}))
    assert parse_lines("classic_auto") == frozenset({"personal_auto"})
    assert _line_of({"lob": ["classic_auto"]}) == _line_of({"lob": ["personal_auto"]})


# 3 - a re-read of the SECOND row sharing an identifier joins it
def test_a_reread_of_the_second_row_sharing_a_name_is_not_a_third_row():
    def coverage(limit, **more):
        return {"coverage_name": _env("Liability"), "limit_amount": _env(limit),
                **{k: _env(v) for k, v in more.items()}}

    first = [coverage(100000), coverage(300000)]          # two real rows, one name
    reread = [coverage(300000, premium=42.0)]             # the second, read again
    merged = merge_policy_windows([
        PolicyWindow("lineblk", [1], {"auto": {"additional_coverages": first}}),
        PolicyWindow("lineblk", [2], {"auto": {"additional_coverages": reread}}),
    ])
    rows = merged.extraction["auto"]["additional_coverages"]
    assert [r["limit_amount"]["parsed"] for r in rows] == [100000, 300000]
    assert rows[1]["premium"]["parsed"] == 42.0


# 4 - the request carries the file name serving fills the output with
def test_the_request_carries_the_source_file_name():
    from serving.vllm_entrypoint import ServingError, _file_name

    assert _file_name({"source_file_name": " policy.pdf "}) == "policy.pdf"
    assert _file_name({"file_name": "a.pdf"}) == "a.pdf" and _file_name({}) is None
    with pytest.raises(ServingError):
        _file_name({"source_file_name": ""})


# 5 - a value citing a page with no OCR text cannot be verified, so it is kept
def test_an_added_value_citing_a_page_with_no_ocr_text_is_kept():
    from data_pipeline.dataset_builder.label_verification import VerificationReport, verified_label

    label = {"carrier": {"company_name": _env("Northfield Mutual", 2)},
             "fideon:filled": {"paths": ["carrier.company_name"]}}
    report = VerificationReport()
    pages = ["Declarations page with text", ""]            # page 2: a scan OCR could not read
    assert verified_label(label, pages, report) is label
    assert (report.unverifiable, report.dropped) == (1, [])


# 6 - a checkpoint of another corpus or configuration is not resumed
def _fingerprint(corpus="v1", manifest=None, rank=64):
    from training.train import run_fingerprint

    recorded = SimpleNamespace(lora_rank=rank, lora_alpha=128, learning_rate=1e-4,
                               lr_scheduler="cosine", epochs=3, effective_batch_size=8,
                               target_modules=["q_proj"])
    return run_fingerprint(corpus, manifest or {"rows": 10}, recorded)


def test_a_run_resumes_only_checkpoints_of_the_same_corpus_and_settings(tmp_path):
    from training.train import TrainingError, assert_resumable, record_fingerprint

    assert _fingerprint()["adapter"] == "policy"
    record_fingerprint(tmp_path, _fingerprint())
    assert_resumable(tmp_path, _fingerprint())                          # the same run
    with pytest.raises(TrainingError, match="corpus_manifest_sha256"):
        assert_resumable(tmp_path, _fingerprint(manifest={"rows": 11}))  # a rebuilt corpus
    with pytest.raises(TrainingError, match="lora_rank"):
        assert_resumable(tmp_path, _fingerprint(rank=32))


def test_a_run_started_before_fingerprints_resumes_with_a_warning(tmp_path, caplog):
    from training.train import assert_resumable

    with caplog.at_level(logging.WARNING):
        assert_resumable(tmp_path, _fingerprint())
    assert "without checking" in caplog.text


# 8 - the classifier reads the first pages in page order, with their text
def test_the_classifier_gets_the_first_pages_in_order_with_their_text():
    from serving.pipeline import ExtractionRequest, _classifier_input

    request = ExtractionRequest(
        source_id="p", image_paths=["d/page_10.png", "d/page_2.png", "d/page_1.png"],
        page_texts={1: "Declarations", 2: "Schedule", 10: "Endorsement"},
    )
    images, text = _classifier_input(request)
    assert images == ["d/page_1.png", "d/page_2.png"]
    assert text == "Declarations\n\nSchedule"

    image_only = ExtractionRequest(source_id="p", image_paths=["d/page_1.png"],
                                   page_texts={1: ""}, modality_mode="image_only")
    assert _classifier_input(image_only)[1] is None


# 9 - line balance is applied in the adapter's view, never silently skipped. (It
# was skipped, with a warning, for the unified scope; the one adapter's view of
# the corpus now always applies it.)
def test_the_adapter_s_view_applies_the_configured_line_balance():
    from common.config import training_config
    from training.corpus_view import _balance_settings, line_repeats

    configured = training_config("unified")["line_balance"]
    assert _balance_settings() == {"min_documents": int(configured["min_documents"]),
                                   "max_repeat": int(configured["max_repeat"])}
    rows = ([{"source_id": f"h{i}", "lob": "homeowners"} for i in range(120)]
            + [{"source_id": "m0", "lob": "motorcycle"}])
    repeats = line_repeats(rows, **_balance_settings())
    assert repeats["homeowners"] == 1 and repeats["motorcycle"] == int(configured["max_repeat"])


# 7 - a small line owned by a held-out carrier is tested, and recorded as such
def test_a_small_line_of_a_held_out_carrier_is_not_recorded_as_train_only(monkeypatch, caplog):
    from data_pipeline.dataset_builder import split_groups as S
    from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits

    monkeypatch.setattr(S, "held_out_carrier_per_line",
                        lambda *a, **k: S.CarrierHoldOut(held={"motorcycle": "Held Out Co"}))
    groups = [GroupRecord(group_id=f"h{i}", doc_type="policy", source_ids=[f"h{i}"],
                          carrier=f"Carrier {i % 7}", line="homeowners") for i in range(60)]
    # A two-document line, both from the held-out carrier.
    groups += [GroupRecord(group_id=f"m{i}", doc_type="policy", source_ids=[f"m{i}"],
                           carrier="Held Out Co", line="motorcycle") for i in range(2)]
    with caplog.at_level(logging.WARNING):
        result = assign_group_splits({"policy": groups})
    assert {result.assignment["m0"], result.assignment["m1"]} == {"test"}
    assert "motorcycle" not in result.train_only_lines.get("policy", [])
    assert "NO training documents" in caplog.text
