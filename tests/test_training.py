"""IMPL-06 — masking, the ViT gate, early stopping, and training configuration.

The masking tests here are the highest-value in the suite. Broken masking trains
the model to reproduce its own prompt, converges to a normal-looking loss curve,
and shows no symptom at all until evaluation — by which point a training run has
been spent. So the guard is **mutation-checked**: a deliberately broken masking
implementation must fail it.
"""

from __future__ import annotations

import pytest

from training import train as T
from training.callbacks.early_stopping import EarlyStoppingState
from training.data_collator import (
    IGNORE_INDEX,
    MaskingError,
    assert_masking_correct,
    find_assistant_span,
    reference_labels,
    verify_batch,
)
from training.vit_gate import (
    ErrorRecord,
    classify_error,
    evaluate_gate,
    summarise_error_mix,
)

#: The four materialized epoch files, as the dataset build writes them.
EPOCH_PATHS = [f"corpus/default/v1/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)]

# A toy sequence: 10 prompt tokens, then 6 assistant tokens.
INPUT_IDS = list(range(100, 116))
ASSISTANT_START, ASSISTANT_END = 10, 16


# --------------------------------------------------------------------------
# Label masking — the critical guard
# --------------------------------------------------------------------------

def test_correct_masking_passes():
    labels = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    report = assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)
    assert report.is_correct
    assert report.supervised_tokens == 6
    assert report.masked_tokens == 10


def test_supervising_prompt_tokens_is_caught():
    """MUTATION CHECK. This is the failure the module exists to prevent: the
    model trains on reproducing its own system prompt, and the loss curve looks
    entirely normal while it happens."""
    labels = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    labels[3] = INPUT_IDS[3]                       # leak one prompt token
    with pytest.raises(MaskingError, match="PROMPT token"):
        assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)


def test_masking_out_assistant_tokens_is_caught():
    """The opposite failure: part of the JSON the model must produce is not
    trained on."""
    labels = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    labels[12] = IGNORE_INDEX
    with pytest.raises(MaskingError, match="ASSISTANT token"):
        assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)


def test_fully_masked_batch_is_caught():
    """Training would complete successfully and produce an adapter identical to
    its initialisation."""
    labels = [IGNORE_INDEX] * len(INPUT_IDS)
    with pytest.raises(MaskingError, match="NO tokens are supervised"):
        assert_masking_correct(labels, ASSISTANT_START, ASSISTANT_END)


def test_masking_everything_is_caught():
    """No masking at all — the whole sequence supervised."""
    with pytest.raises(MaskingError, match="PROMPT token"):
        assert_masking_correct(list(INPUT_IDS), ASSISTANT_START, ASSISTANT_END)


def test_batch_verification_across_rows():
    rows = [
        reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END),
        reference_labels(INPUT_IDS, 8, 16),
    ]
    report = verify_batch({"labels": rows}, [(ASSISTANT_START, ASSISTANT_END), (8, 16)])
    assert report.supervised_tokens == 6 + 8


def test_batch_without_labels_is_caught():
    """Without labels nothing is supervised and training silently does nothing."""
    with pytest.raises(MaskingError, match="no `labels`"):
        verify_batch({"input_ids": INPUT_IDS}, [(0, 1)])


def test_one_bad_row_fails_the_whole_batch():
    good = reference_labels(INPUT_IDS, ASSISTANT_START, ASSISTANT_END)
    bad = list(INPUT_IDS)
    with pytest.raises(MaskingError, match=r"row 1"):
        verify_batch({"labels": [good, bad]}, [(ASSISTANT_START, ASSISTANT_END), (ASSISTANT_START, ASSISTANT_END)])


def test_assistant_span_is_located_from_header_tokens():
    ids = [1, 2, 3, 77, 78, 40, 41, 42, 99]
    start, end = find_assistant_span(ids, assistant_header_ids=[77, 78], end_token_id=99)
    assert (start, end) == (5, 9)
    assert ids[start:end] == [40, 41, 42, 99]


def test_the_span_includes_the_end_of_turn_token():
    """The model has to be taught to stop. Excluding the end token left EOS
    unsupervised, so generation runs to max_new_tokens — which the runner
    reports as truncation, i.e. as silently dropped rows."""
    ids = [1, 2, 77, 78, 40, 99]
    start, end = find_assistant_span(ids, assistant_header_ids=[77, 78], end_token_id=99)
    assert ids[end - 1] == 99

    labels = reference_labels(ids, start, end)
    assert labels[-1] == 99, "the end-of-turn token is masked out of the loss"


def test_the_span_takes_in_the_templates_suffix_after_the_end_token():
    """ms-swift's ChatML suffix is `<|im_end|>\\n` and it supervises the newline.
    Ending the span at `<|im_end|>` refused a correct ms-swift row before launch."""
    ids = [1, 2, 77, 78, 40, 99, 198]
    labels = [IGNORE_INDEX] * 4 + [40, 99, 198]
    start, end = find_assistant_span(ids, assistant_header_ids=[77, 78], end_token_id=99,
                                     end_suffix_ids=[198])
    assert (start, end) == (4, 7)
    verify_batch({"labels": [labels]}, [(start, end)])
    with pytest.raises(MaskingError, match=r"after the assistant turn.*token ids \[198\]"):
        verify_batch({"labels": [labels]}, [find_assistant_span(ids, [77, 78], 99)])


def test_only_the_exact_suffix_is_taken_in():
    """Anything else supervised after the end token is still a masking error."""
    ids = [1, 2, 77, 78, 40, 99, 55, 56]
    labels = [IGNORE_INDEX] * 4 + [40, 99, 55, 56]
    span = find_assistant_span(ids, [77, 78], 99, end_suffix_ids=[198])
    assert span == (4, 6)
    with pytest.raises(MaskingError, match="2 token"):
        verify_batch({"labels": [labels]}, [span])


def test_a_span_past_a_truncated_row_is_a_masking_error():
    """Spans computed before truncation do not survive it. A bare IndexError
    gives none of the diagnosis this module exists to provide."""
    with pytest.raises(MaskingError, match="does not fit a row"):
        verify_batch({"labels": [[IGNORE_INDEX] * 8]}, [(4, 20)])


def test_a_missing_assistant_turn_is_caught():
    """Either the chat template changed or the header ids are wrong — both would
    mask the wrong span."""
    with pytest.raises(MaskingError, match="assistant turn was not found"):
        find_assistant_span([1, 2, 3], assistant_header_ids=[77, 78])


# --------------------------------------------------------------------------
# The ViT gate (arch §3, §16c)
# --------------------------------------------------------------------------

def _perception_errors(n: int = 10, modality: str = "image_only") -> list[ErrorRecord]:
    """Errors from the image-only subset by default: accuracy is gated on
    image-only and scanned documents, so the mix that decides comes from there."""
    return [
        ErrorRecord(f"policy_{i:04d}", "policy_number", "WC-8842317-01", "WC-8842317-0l",
                    error_class="perception", modality_mode=modality)
        for i in range(n)
    ]


def _reasoning_errors(n: int = 10, modality: str = "image_only") -> list[ErrorRecord]:
    return [
        ErrorRecord(f"policy_{i:04d}", "insured_name", "Rivera Fabrication LLC",
                    "Meridian Property Group LLC", error_class="schema_reasoning",
                    modality_mode=modality)
        for i in range(n)
    ]


def test_gate_refuses_to_decide_on_pilot_volume():
    """arch §16c — at pilot volume low accuracy is a data artifact, and thin data
    produces exactly the perception-shaped errors the gate keys on. Firing here
    would buy a retrain for a problem more documents would solve."""
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=9, scanned_documents=7,     # pilot volume
        errors=_perception_errors(),
    )
    assert decision.decision == "insufficient_data"
    assert not decision.should_train_vit
    assert "data-volume artifact" in decision.rationale
    assert "resolution cap" in decision.recommendation   # the cheap fix comes first


def test_gate_holds_when_accuracy_meets_target():
    decision = evaluate_gate(
        image_only_accuracy=0.88, scanned_accuracy=0.85,
        image_only_documents=50, scanned_documents=40,
        errors=_perception_errors(3),
    )
    assert decision.decision == "hold"
    assert "train_vit: false" in decision.recommendation


def test_gate_holds_when_errors_are_reasoning_not_perception():
    """THE load-bearing distinction. If the model reads the character correctly
    and puts it in the wrong field, unfreezing the ViT will not help."""
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=50, scanned_documents=40,
        errors=_reasoning_errors(),
    )
    assert decision.decision == "hold"
    assert "schema_reasoning" in decision.rationale
    assert not decision.should_train_vit


def test_gate_fires_only_on_low_accuracy_and_perception_errors():
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=50, scanned_documents=40,
        errors=_perception_errors(),
    )
    assert decision.decision == "fire"
    assert decision.should_train_vit
    assert "Never a full fine-tune" in decision.recommendation


def test_even_when_firing_the_cheap_fixes_come_first():
    """Resolution cap and modality rebalance need no new labeling; a ViT LoRA
    costs a full Foundation retrain."""
    decision = evaluate_gate(
        image_only_accuracy=0.50, scanned_accuracy=0.55,
        image_only_documents=60, scanned_documents=60,
        errors=_perception_errors(20),
    )
    assert decision.decision == "fire"
    assert "resolution cap" in decision.recommendation
    assert "modality mix" in decision.recommendation


def test_a_subset_nobody_scored_is_not_a_subset_that_passed():
    """Skipping the count check on a None fell through to the accuracy check,
    where `value is not None` skipped it again — so the gate returned `hold`
    with a rationale asserting a target was met that nobody had measured."""
    decision = evaluate_gate(
        image_only_accuracy=None, scanned_accuracy=None,
        image_only_documents=0, scanned_documents=0,
        errors=[],
    )
    assert decision.decision == "insufficient_data"
    assert "never scored" in decision.rationale


def test_one_scored_subset_does_not_carry_an_unscored_one():
    decision = evaluate_gate(
        image_only_accuracy=0.92, scanned_accuracy=None,
        image_only_documents=60, scanned_documents=0,
        errors=[],
    )
    assert decision.decision == "insufficient_data"


def test_the_deciding_error_mix_comes_from_the_gated_subsets_only():
    """Accuracy is read from image-only and scanned documents, so the mix has to
    be too. Over every error, an ocr_plus_image majority — the largest subset,
    and where perception is least implicated — drowns a genuine image-only
    perception failure under the 60% threshold."""
    errors = (
        _perception_errors(9, modality="image_only")
        + _reasoning_errors(30, modality="ocr_plus_image")
    )
    decision = evaluate_gate(
        image_only_accuracy=0.55, scanned_accuracy=0.60,
        image_only_documents=50, scanned_documents=40,
        errors=errors,
    )
    assert decision.decision == "fire", "the ocr_plus_image majority outvoted the gated subsets"
    assert decision.error_mix["perception"] == pytest.approx(1.0)
    assert decision.overall_error_mix["perception"] < 0.60, "the overall mix is still reported"


def test_a_scanned_error_counts_even_when_its_modality_is_ocr_plus_image():
    """Scanned is a gated subset in its own right; a scanned document normally
    still carries OCR."""
    from training.vit_gate import in_gated_subsets

    scanned = ErrorRecord("policy_0001", "policy_number", "A", "4",
                          error_class="perception", modality_mode="ocr_plus_image",
                          is_scanned=True)
    plain = ErrorRecord("policy_0002", "policy_number", "A", "4",
                        error_class="perception", modality_mode="ocr_plus_image")
    assert in_gated_subsets([scanned, plain]) == [scanned]


def test_the_gate_reads_the_keys_an_eval_report_actually_emits():
    """It read `eval_metrics`/`subset_counts`/`error_records`, none of which
    EvalReport.as_dict emits — so every well-formed report produced None
    accuracies and zero documents, and the gate decided from nothing."""
    from evaluation.run_eval import EvalReport, SubsetReport
    from training.vit_gate import evaluate_from_report

    report = EvalReport(model_version="foundation-v1", corpus_version="v1", subsets=[
        SubsetReport("policy", "full", documents=90,
                     metrics={"field_normalized_match": 0.91}),
        SubsetReport("policy", "image_only", documents=50,
                     metrics={"field_normalized_match": 0.55},
                     error_records=[{"source_id": "policy_0001", "field_path": "policy_number",
                                     "expected": "WC-8842317-01", "got": "WC-8842317-0l",
                                     "error_class": "perception"}] * 10),
        SubsetReport("policy", "scanned", documents=40,
                     metrics={"field_normalized_match": 0.60}),
    ]).as_dict()

    decision = evaluate_from_report(report)

    assert decision.image_only_accuracy == pytest.approx(0.55)
    assert decision.scanned_accuracy == pytest.approx(0.60)
    assert decision.image_only_documents == 50 and decision.scanned_documents == 40
    assert decision.error_mix["perception"] == pytest.approx(1.0)
    assert decision.decision == "fire"


def test_a_payload_that_is_not_an_eval_report_is_refused():
    """Guessing at the shape is how the gate came to decide from all-None."""
    from training.vit_gate import evaluate_from_report

    with pytest.raises(ValueError, match="not an EvalReport payload"):
        evaluate_from_report({"eval_metrics": {"image_only_accuracy": 0.5}})


def test_error_classification_separates_misreading_from_misplacing():
    # A near-miss on the expected string: misread character.
    assert classify_error("WC-8842317-01", "WC-8842317-0l") == "perception"
    # A different field's value: read correctly, placed wrongly.
    assert classify_error(
        "Rivera Fabrication LLC", "Meridian Property Group LLC",
        all_expected={"insured_name": "Rivera Fabrication LLC",
                      "certificate_holder": "Meridian Property Group LLC"},
    ) == "schema_reasoning"
    # Nothing produced.
    assert classify_error("Rivera Fabrication LLC", None) == "omission"


def test_error_mix_is_summarised_as_shares():
    mix = summarise_error_mix(_perception_errors(6) + _reasoning_errors(4))
    assert mix["perception"] == pytest.approx(0.6)
    assert mix["schema_reasoning"] == pytest.approx(0.4)


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------

def test_early_stopping_tracks_field_f1_not_loss():
    """Loss can improve while extraction gets worse — loss rewards fluent JSON,
    and a confidently wrong value is fluent."""
    state = EarlyStoppingState(patience=2)
    assert not state.update(100, field_f1=0.70, eval_loss=0.45)
    assert not state.update(200, field_f1=0.78, eval_loss=0.38)
    assert not state.update(300, field_f1=0.77, eval_loss=0.35)
    assert state.update(400, field_f1=0.76, eval_loss=0.31)      # stop, despite falling loss
    assert state.best_score == 0.78 and state.best_step == 200


def test_improvement_resets_patience():
    state = EarlyStoppingState(patience=2)
    state.update(100, 0.70)
    state.update(200, 0.69)
    assert not state.update(300, 0.75)
    assert state.evaluations_without_improvement == 0


# --------------------------------------------------------------------------
# Training configuration (arch v2.1 §4.1, §9, §11.1)
# --------------------------------------------------------------------------

def test_training_is_one_lora_run_on_a_bf16_base():
    """One adapter, trained fresh on the base, across every task. vLLM applies ONE
    LoRA per request, so v1's Foundation + per-type stack could never both be
    active — it was unservable, not merely awkward."""
    swift, recorded = T.build_training_config(
        corpus_paths=EPOCH_PATHS, output_dir="/tmp/out"
    )
    assert swift.args["train_type"] == "lora"
    # bf16 emits no quantization flag at all: ms-swift 3's quant_bits defaults to
    # None, and the 2.x "quantization_bit" is not an argument it accepts.
    assert "quant_bits" not in swift.args and "quantization_bit" not in swift.args
    assert "bnb_4bit_quant_type" not in swift.args
    assert recorded.technique == "LoRA"
    assert recorded.base_quantization == "bf16_frozen_base"
    assert recorded.lora_rank == 64 and recorded.lora_alpha == 128


def test_both_the_vit_and_the_mergers_are_frozen():
    """A reversal of v1, which targeted the merger with LoRA. Tower and connector
    LoRA in vLLM is experimental with known mixed-adapter batching risks, and
    arbitration is learned in the DECODER, where image tokens and OCR text tokens
    attend to each other — the mergers never see OCR text (arch v2.1 §9a)."""
    swift, recorded = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert swift.args["freeze_vit"] is True
    assert swift.args["freeze_aligner"] is True
    assert "merger" not in swift.args["target_modules"]
    assert "merger" not in recorded.target_modules
    assert set(recorded.target_modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def test_the_memory_settings_that_make_the_largest_cap_affordable_reach_the_trainer():
    """Without use_logits_to_keep the LM head produces a 151k-vocabulary
    distribution at every position of a 32k sequence, which dominates activation
    memory on its own (arch v2.1 §9.3)."""
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert swift.args["use_logits_to_keep"] is True
    assert swift.args["padding_free"] is True
    assert swift.args["group_by_length"] is False
    assert swift.args["gradient_checkpointing"] is True


def test_max_length_is_the_largest_task_cap():
    """ms-swift takes ONE max_length and the corpus interleaves every task, so
    anything smaller would truncate the longest one rather than reject it — and a
    clipped assistant span trains the model to stop early (arch v2.1 §7a)."""
    swift, recorded = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")
    largest = T.corpus_max_length()

    assert swift.args["max_length"] == largest
    assert recorded.max_seq_len == largest
    assert largest > 8192, "the v1 single cap could not hold a page plus its schema"


def test_warmup_reaches_the_trainer_in_steps_not_as_a_ratio():
    """At pilot volume a 0.03 ratio over a handful of steps rounds to zero, and
    the first optimizer step lands at full learning rate on a freshly-initialised
    adapter (arch v2.1 §11.1)."""
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert swift.args["warmup_steps"] >= 1
    assert "warmup_ratio" not in swift.args


def test_deepspeed_is_absent_unless_asked_for():
    """Under LoRA, ZeRO-2 shards ~1.5GB of adapter optimizer state and is close
    to a no-op. Sequence parallelism is the first reach for VRAM (§9.3)."""
    without, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")
    assert "deepspeed" not in without.args

    with_zero, _ = T.build_training_config(
        corpus_paths=EPOCH_PATHS, output_dir="/tmp/out", deepspeed="zero3"
    )
    assert with_zero.args["deepspeed"].endswith("zero3.json")


def test_four_bit_is_a_live_flag_not_a_hardcoded_constant(monkeypatch):
    """`load_in_4bit` sat in the YAML being read by nothing while the trainers
    hardcoded `quantization_bit: 4`, so the QLoRA-vs-LoRA decision could not be
    A/B'd without a code change — and the manifest recorded a value that had no
    effect on the run."""
    base = T.base_model_config()
    quantized = {**base, "quantization": {**base["quantization"], "load_in_4bit": True}}
    monkeypatch.setattr(T, "base_model_config", lambda: quantized)

    swift, recorded = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert swift.args["quant_method"] == "bnb" and swift.args["quant_bits"] == 4
    assert swift.args["bnb_4bit_quant_type"] == "nf4"
    assert recorded.technique == "QLoRA"
    assert recorded.base_quantization == "nf4_double_quant_bfloat16_compute"


def test_the_manifest_cannot_default_its_way_into_claiming_qlora():
    """These three carried "QLoRA" / NF4 / paged-8-bit as model defaults, so a
    bf16 run that did not pass them recorded a technique it never used. The
    record is the whole basis for attributing a regression, and base precision is
    exactly the kind of change that causes one."""
    from registry_utils.models import TrainingConfig

    with pytest.raises(ValueError):
        TrainingConfig(
            lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3,
            gradient_accumulation_steps=8, effective_batch_size=8,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=24576, seed=42,
        )


def test_training_the_vit_is_recorded_as_lora_never_full_fine_tune():
    """The §3 escalation adds a ViT LoRA. Full fine-tuning risks the pretrained
    document/OCR capability the image-only path relies on."""
    swift, recorded = T.build_training_config(
        corpus_paths=EPOCH_PATHS, output_dir="/tmp/out", train_vit=True
    )
    assert swift.args["freeze_vit"] is False
    assert recorded.vit_trainable is True
    assert recorded.vit_method == "lora"


def test_the_run_reads_one_file_per_epoch_once_each():
    """Four epoch files are materialized, and a 3-epoch run reads three of them
    ONCE. Passing num_train_epochs=3 as well made ms-swift loop the three files
    three times — nine passes, recorded as three (arch v2.1 §6.1)."""
    swift, recorded = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert swift.args["dataset"] == EPOCH_PATHS[: recorded.epochs]
    assert swift.args["num_train_epochs"] == 1
    assert recorded.epochs == 3


def test_fewer_epoch_files_than_epochs_is_refused():
    with pytest.raises(T.TrainingError, match="epoch file"):
        T.build_training_config(corpus_paths=EPOCH_PATHS[:2], output_dir="/tmp/out")


def test_manifest_example_counts_are_summed_not_cast():
    """The corpus manifest nests counts by type and mode; int() on that is a
    TypeError, which is how the training CLI failed before it started."""
    assert T.count_examples({"policy": {"image_only": 3, "ocr_plus_image": 5}, "acord": {"x": 2}}) == 10
    assert T.count_examples(None) == 0


def test_the_validation_split_never_enters_the_training_set():
    """Passing both to --dataset made ms-swift treat validation as training data
    and then carve its own eval split out of the union, so the selected
    checkpoint was chosen on documents the model had memorised — and the
    promotion gate read that number."""
    swift, _ = T.build_training_config(
        corpus_paths=EPOCH_PATHS,
        val_paths=["corpus/default/v1/val/val.jsonl"],
        output_dir="/tmp/out",
    )
    assert swift.args["val_dataset"] == ["corpus/default/v1/val/val.jsonl"]
    assert not set(swift.args["dataset"]) & set(swift.args["val_dataset"])


def test_explicit_evaluation_settings_win_over_the_early_stopping_defaults():
    """The helper is unpacked FIRST so the explicit keys win. Unpacking it last
    silently overrode this config's metric_for_best_model and
    load_best_model_at_end with the helper's own defaults."""
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")

    assert swift.args["metric_for_best_model"] == "eval_loss"
    assert swift.args["greater_is_better"] is False
    # Off by the config, deliberately: checkpoint selection chooses what ships, and
    # on, the trainer refuses to save resume points between evaluations.
    assert swift.args["load_best_model_at_end"] is False


def test_checkpoint_selection_does_not_happen_here():
    """Field F1 needs generation, which the training loop's eval does not do
    efficiently for a VLM. Loss selects for EARLY STOPPING only; a separate vLLM
    job selects what ships (arch v2.1 §11.2)."""
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")
    assert swift.args["save_total_limit"] >= 4, (
        "checkpoint_eval scores the last 3 plus the best-loss checkpoint; fewer "
        "retained would discard a candidate before it is scored"
    )


def test_swift_cli_renders_flags_correctly():
    """Booleans render as `--flag false`, not as a dropped presence flag — which
    silently disabled every option whose correct value is False. Lists render one
    argv element per item, or nargs="+" collapses them into one token."""
    cli = T.SwiftConfig(args={
        "freeze_vit": False, "dataset": ["a.jsonl", "b.jsonl"], "lora_rank": 64, "skip": None,
    }).to_cli()

    assert cli[:2] == ["swift", "sft"]
    assert "--freeze_vit" in cli and cli[cli.index("--freeze_vit") + 1] == "false"
    assert cli[cli.index("--dataset") + 1:cli.index("--dataset") + 3] == ["a.jsonl", "b.jsonl"]
    assert "--skip" not in cli


def test_no_code_path_can_enable_full_vit_fine_tuning():
    """The §3 escalation is 'add a ViT LoRA', never 'unfreeze and train the
    encoder'."""
    import inspect

    source = inspect.getsource(T)
    for forbidden in ('train_type="full"', "full_finetune", "freeze_vit=False,  # full"):
        assert forbidden not in source, f"{T.__name__} has a full fine-tune path"


def test_a_trainable_vit_must_declare_lora():
    """The manifest refuses the combination outright, so a run cannot record a
    ViT escalation it did not perform by any method."""
    from registry_utils.models import TrainingConfig

    with pytest.raises(ValueError, match="vit_method"):
        TrainingConfig(
            technique="LoRA", base_quantization="bf16_frozen_base", optimizer="adamw_torch",
            lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3,
            gradient_accumulation_steps=8, effective_batch_size=8,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=24576, seed=42,
            vit_trainable=True, vit_method="frozen",
        )


def _blob():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _pending_manifest():
    from registry_utils.models import (
        Artifacts,
        DataStats,
        Dependencies,
        RunManifest,
        TrainingConfig,
    )

    return RunManifest(
        run_id="extractor-v1", run_type="unified",
        dependencies=Dependencies(base_model="qwen3-vl-8b-instruct@abc1234",
                                  corpus_version="corpus/v1", code_git_commit="abc1234"),
        training_config=TrainingConfig(
            technique="LoRA", base_quantization="bf16_frozen_base", optimizer="adamw_torch",
            lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3,
            gradient_accumulation_steps=8, effective_batch_size=8,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=24576, seed=42),
        data_stats=DataStats(train_examples=10, val_examples=2, test_examples=2),
        artifacts=Artifacts(status="staged", staging_path="/runpod-volume/staging/x"),
        status="training",
    )


def test_a_run_that_never_launched_is_not_recorded_as_trained(monkeypatch):
    """The manifest is written before ms-swift starts — the run_id has to be
    reserved and the config captured even for a run that dies. But a pod that
    OOMs at step 40 must not leave a registry entry asserting trained weights and
    a staging path holding nothing."""
    client, manifest = _blob(), _pending_manifest()
    monkeypatch.setattr(T, "launch", lambda _c: None)

    T.launch_and_record(T.SwiftConfig(args={}), manifest, client)
    assert manifest.status == "trained"


def test_a_crashed_run_is_recorded_as_failed(monkeypatch):
    client, manifest = _blob(), _pending_manifest()

    def boom(_config):
        raise RuntimeError("CUDA out of memory at step 40")

    monkeypatch.setattr(T, "launch", boom)
    with pytest.raises(RuntimeError):
        T.launch_and_record(T.SwiftConfig(args={}), manifest, client)

    assert manifest.status == "failed"
    from registry_utils.query_registry import get
    assert get("extractor-v1", client).status == "failed"


def test_a_registry_run_id_is_refused_where_a_checkpoint_path_belongs():
    """`continue_from` reaches ms-swift as an adapter directory. A run-id finds
    no adapter, trains from base, and the manifest records `continued_from` — a
    lineage that never happened, later read as evidence for how much regression
    testing a promotion needs. Refused however it is spelled."""
    for run_id in ("extractor-v3", "extractor-v3/", "policy-v2/", "./extractor-v3"):
        with pytest.raises(T.TrainingError, match="checkpoint path"):
            T.assert_checkpoint_path(run_id)

    T.assert_checkpoint_path("/runpod-volume/staging/adapters/foundation/v3")   # fine


def test_a_real_run_refuses_a_directory_with_no_adapter_in_it(tmp_path):
    with pytest.raises(T.TrainingError, match="adapter_config.json"):
        T.assert_checkpoint_path(str(tmp_path), must_exist=True)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    T.assert_checkpoint_path(str(tmp_path), must_exist=True)


def test_continuing_starts_a_new_run_from_the_adapter():
    """resume_from_checkpoint restores the old optimizer and global_step, so a
    continuation could run zero steps and exit "trained" with the old weights."""
    swift, _ = T.build_training_config(
        corpus_paths=EPOCH_PATHS, output_dir="/tmp/out", resume_from="/ckpt/checkpoint-150"
    )
    assert swift.args["adapters"] == ["/ckpt/checkpoint-150"]
    assert "resume_from_checkpoint" not in swift.args


def test_the_command_uses_ms_swift_3_names_only():
    """One ms-swift version, throughout. Its parser refuses an unknown flag, so a
    2.x name in a 3.x command kills the run at argument parsing."""
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")
    for old in ("model_type", "model_id_or_path", "lora_target_modules", "quantization_bit",
                "length_grouped_sampling", "early_stopping_patience", "resume_from_checkpoint"):
        assert old not in swift.args, f"{old} is not an ms-swift 3 argument"
    assert swift.args["model"] == T.base_model_config()["model"]["model_id"]
    assert swift.args["use_hf"] is True
    assert swift.args["split_dataset_ratio"] == 0.0
    assert swift.args["report_to"] == "none"
    assert swift.args["attn_impl"] == "flash_attn"


def test_the_evaluation_interval_is_sized_to_the_run():
    """A fixed 50 steps gave a ~56-step pilot run one evaluation, and checkpoint
    selection nothing to choose between - and the delivered corpus ~50
    evaluations, 50-60 h of validating. The validation cut divides every run into
    evaluations_per_run checks."""
    small, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o", train_rows=450)
    steps = -(-450 // 8)
    assert steps // small.args["eval_steps"] >= T.MIN_EVALUATIONS
    big, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o", train_rows=20_000)
    big_steps = -(-20_000 // 8)
    assert big_steps // big.args["eval_steps"] in (9, 10)          # ~10 checks, not ~50
    assert big.args["eval_steps"] % big.args["save_steps"] == 0    # each check lands on a save


def test_train_vit_is_refused_until_the_vision_targets_are_known(monkeypatch):
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    with pytest.raises(T.TrainingError, match="train-vit"):
        T.train(corpus_version="v1", out_version="v9", client=None, corpus_manifest={},
                data_stats=None, train_vit=True, dry_run=True)


# --------------------------------------------------------------------------
# Classification of nested misplacement
# --------------------------------------------------------------------------


def test_a_value_lifted_from_the_wrong_list_row_is_not_a_misread():
    """A loss run's confusables live inside `claims[]`. Reading only top-level
    fields sent the canonical misplacement error to the near-miss check, where
    two similar amounts read as a misread character — and that classification is
    what decides whether a Foundation retrain is justified."""
    golden = {
        "insured_name": "Rivera Fabrication LLC",
        "claims": [
            {"claim_number": "CLM-00417", "amount": "12500.00"},
            {"claim_number": "CLM-00418", "amount": "12800.00"},
        ],
    }
    # Row 1's amount answered with row 2's: read correctly, placed wrongly.
    assert classify_error("12500.00", "12800.00", all_expected=golden) == "schema_reasoning"
    # A genuine misread of a value that appears nowhere else stays perception.
    assert classify_error("12500.00", "125OO.00", all_expected=golden) == "perception"


def test_the_rank_phase_has_a_grid_for_both_adapter_types():
    """Every other phase does. Sweeping rank for the Foundation only meant
    `--adapter-type per_type` raised at phase 3 — after six runs were spent."""
    from training.sweep import candidates_for, load_phase

    for phase in ("lr", "epochs", "rank"):
        for adapter_type in ("foundation", "per_type"):
            assert candidates_for(phase, adapter_type=adapter_type), \
                f"{phase} has no grid for {adapter_type}"

    per_type = [c.value for c in candidates_for("rank", adapter_type="per_type")]
    assert 16 in per_type, "the per-type grid should bracket the rank-16 default"
    assert max(per_type) < max(load_phase("rank")["grid"]["foundation"]["lora_rank"])


def test_an_empty_document_count_is_not_enough_data():
    """An empty dict is a caller that did not count, not a corpus that is large
    enough — the same absence-is-success mistake the pipeline refuses elsewhere."""
    from training.sweep import SweepError, assert_enough_data

    with pytest.raises(SweepError, match="no per-type document counts"):
        assert_enough_data({})


def test_an_evaluation_with_no_loss_records_no_loss():
    """`or 0.0` wrote a loss of zero — a perfect model — for an evaluation that
    reported none, and that 0.0 then read as the best value in the history."""
    state = EarlyStoppingState(patience=2)
    state.update(100, field_f1=0.70)
    state.update(200, field_f1=0.75, eval_loss=0.31)

    assert "eval_loss" not in state.history[0]
    assert state.history[1]["eval_loss"] == 0.31


def test_max_length_covers_the_per_doc_type_overrides():
    """Iterating only the bare tasks gave 24576 while the policy extraction
    override is 32768, so every routed policy would have been TRUNCATED — the
    exact failure the §7a caps exist to prevent, and one nothing would have
    reported except unexplained row loss on long documents."""
    from common.config import seq_cap_for_task

    assert T.corpus_max_length() >= seq_cap_for_task("extract", "policy")
    assert T.corpus_max_length() == 32768


def test_dataset_preprocessing_runs_on_several_processes_capped_at_the_cpu_count(monkeypatch):
    """ms-swift's dataset preprocessing gets several processes, never more than
    the pod has. (group_by_length, now off, made it encode every row before step 1.)"""
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")
    assert swift.args["dataset_num_proc"] == 8
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/tmp/out")
    assert swift.args["dataset_num_proc"] == 4
