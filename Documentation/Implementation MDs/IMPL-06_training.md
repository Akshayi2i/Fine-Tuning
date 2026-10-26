# SPEC 06 — Training (ms-swift LoRA: ONE Adapter, for Policies)

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-05.
>
> **Architecture refs:** `finetuning-architecture-v2.7.docx` §3 (ViT escalation gate — **LoRA, never full FT**), §4.1 (**one policy adapter**), §4.2 (**incremental training**: each cycle continues the promoted adapter), §6.1 (per-epoch modality sampling), §7b (policy windows over every page), §8b (tenancy per Fideon SPEC_12; de-identification per Fideon SPEC_11 — Option A, none, see IMPL-05 §1), **§9.2 (bf16 base vs 4-bit)**, §9.3 (memory settings), §9.4 / §9a (frozen ViT and mergers; LoRA targets), §10 (**locked three-layer trainer stack**; §10.1 fallback, §10.2 parity tests), §11 (hyperparameters), §11.2 (checkpoint selection), §11a (**sweep methodology**), §12 (versioning + run registry).

## Goal

Fine-tune Qwen3-VL-8B-Instruct with **LoRA on a bf16 base** via the locked trainer stack. **One adapter**, with the ViT *and* the mergers frozen, and every run — sweeps included — recorded in the registry. Since the decision of 2026-10-22 it is the **policy adapter** (`common/adapter.py`): policy documents of every line, the policy tasks, trained each cycle over all the policy data so far, new and existing. Since the decision of 2026-10-23 a cycle **continues the promoted adapter** rather than starting from the base (§1, *Versioning rule*), and nothing is merged: the release is the adapter. There are no training scopes and no layout families; Fideon SPEC_09's five policy adapter groups by layout family are knowingly not followed. The code for ACORD and Loss Run documents is kept and not trained.

Two reversals from v1, both forced by how vLLM serves:

* **One adapter, not a Foundation plus per-type stack.** vLLM applies one LoRA per request, so the two could never both be active — the v1 topology was unservable, not merely awkward. It also did not fit the data: at 25–30 documents per type a rank-16 adapter memorises its own training set.
* **The mergers are not a LoRA target** (`freeze_aligner: true`). Tower and connector LoRA in vLLM is experimental with known mixed-adapter batching risks, and image-vs-OCR arbitration is learned in the **decoder**, where image tokens and OCR text tokens attend to each other. The mergers only map visual features and never see OCR text at all.

## The locked stack (arch §10) — not a choice to re-make

```
Layer 3   ms-swift                    ← what you invoke (CLI / config)
              │ wraps and configures ↓
Layer 2   swift.trainers.Seq2SeqTrainer   ← the real loop, on the HF Trainer
              │ (optimizer, backprop, grad accumulation, checkpointing,
              │  eval hooks, early stopping, DeepSpeed/Accelerate)
              │ built on ↓
Layer 1   PyTorch + Transformers + PEFT + Accelerate (bitsandbytes only under load_in_4bit)
```

TRL's `SFTTrainer` is **not** in this stack: ms-swift builds its own `Seq2SeqTrainer` on the HF Trainer, and TRL is a dependency for its RLHF trainers only. `training/train.py` is a **thin wrapper that assembles the ms-swift config and launches it**, not a custom training loop.

**Fallback clause (arch §10.1):** migrating to TRL `SFTTrainer` is permitted **only** if ms-swift lacks a specific Qwen3-VL capability needed at implementation time. It is a framework migration, not a layer swap — the template, collator and masking must be re-implemented against the §10.2 parity tests. Document it if invoked. (Dropping to ms-swift's own `Seq2SeqTrainer` is not the fallback: it is already Layer 2.)

## Deliverables

### 1. `training/train.py` — the adapter's run

Replaces v1's `train_foundation.py` and `train_adapter.py`. One entrypoint. Every run is the policy adapter's (`common.adapter.ADAPTER`): run type `scoped`, `scope: policy`, run id `policy-v{n}` (`ADAPTER.run_id`), filed under `registry/scope/policy/` — one independent adapter on the base, never stacked. Older runs (`extractor-v{n}`, run type `unified`) still load.

- **Config:** `configs/training/unified.yaml` + `configs/base_model.yaml` (`ADAPTER.training_config`). There are no per-type or per-scope training configs: `unified.yaml` holds the `data_mix` too (`configs/training/personal_lines.yaml` is gone).
- **Base precision** from `training/base_precision.py`. Default is a **bf16 frozen base**: no quantization argument at all (ms-swift 3 has no `quantization_bit`), and every `bnb_4bit_*` argument suppressed so the run log never carries settings describing nothing the run did. Under `load_in_4bit` it renders `quant_method: bnb`, `quant_bits: 4` and the `bnb_4bit_*` set. `load_in_4bit: true` in `configs/base_model.yaml` switches to NF4 with double quantization and bf16 compute. The manifest's `technique` (`LoRA`/`QLoRA`) and `base_quantization` are derived from the same config, never defaulted.
- **LoRA:** rank 64 / alpha 128, dropout **0.10** at pilot scale (0.05 from ≥200 docs per type), `bias: none`. Targets are the attention and MLP projections of the **language-model decoder only** — `q/k/v/o_proj`, `gate/up/down_proj`. No `merger`.
- **Frozen:** `freeze_vit: true` (unless `--train-vit`, the §3 escalation — LoRA-on-ViT, never full fine-tune) and `freeze_aligner: true` always.
- **Optimisation (§11.1):** AdamW (`adamw_torch`), LR **1e-4**, cosine, **`warmup_steps: 10`** (steps, not a ratio — a ratio rounds to zero warmup at pilot volume), weight decay **0.0** (decay on a low-rank adapter is a shrinkage prior on the update, not regularisation), max grad norm 1.0, **effective batch 8** (`batch.effective_batch_size`) at per-device batch 1, the accumulation derived from the GPU count (`_accumulation_steps`: effective ÷ (per-device × GPUs), refused when it does not divide), gradient checkpointing, bf16. `dataset_num_proc` 8 for ms-swift's preprocessing, capped at the pod's CPU count. A continued run uses the non-null values of the `continued_training:` block (`num_train_epochs`, `learning_rate`) instead of `optimization`'s (`training.train.optimization_for`). Both are null today, so a continued run trains as a fresh one; set them from a measured continued run, not by guess. The block is checked by `optimization_for` and, before any work, by `common.config.validate_all` (`continued_training_problem`): a key the `optimization` block does not have, or an epoch count or learning rate that is not a positive number (epochs a whole number), is refused.
- **Memory (§9.3):** `use_logits_to_keep`, `padding_free`. **`group_by_length` is false** (`memory.length_grouped_sampling: false`): with one row per device and padding-free batching there is no padding to save, and with it on ms-swift encoded every row before step 1 to learn its length (~1.5 h on the CGL run). Rows are visited in a plain random order: same data, same steps. `max_length` is the largest cap of **the adapter's tasks**, including the `by_doc_type` overrides of the types it trains (`corpus_max_length()`; 32,768 for the policy adapter). If a cap does not fit one card, **sequence parallelism comes first**; DeepSpeed is off by default, because under LoRA ZeRO-2 shards ~1.5 GB of adapter optimizer state and is close to a no-op.
- **Datasets.** **Epochs are files, and ms-swift runs one pass over them** (§6.1). `dataset` is the first `num_train_epochs` (3) of the adapter's epoch files — its view of the corpus, `corpus/{tenant}/v{n}/train/scope/policy/epoch_{1..4}.jsonl` — and ms-swift is given **`num_train_epochs: 1`**. Each file already holds every train document once in that epoch's modality draw, so the concatenation *is* the N-epoch run; also telling ms-swift N looped the N files N times — nine passes for a "3 epoch" run. The manifest records the logical count. Fewer epoch files than epochs is refused. `val_dataset` is the adapter's validation file (`corpus/{tenant}/v{n}/val/scope/policy/val.jsonl`), staged as the fixed loss sample (see *Evaluation*) and passed separately (see below). The corpus is read for `--tenant`, never silently from the default tenant. (ms-swift shuffles across the concatenation, so epoch-2 rows can precede epoch-1 rows; each document is still seen exactly N times, each time in its drawn regime.)
- **Evaluation during training** is validation **loss** on a fixed stratified sample of **`evaluation.loss_sample_rows` (160)** validation rows (`train.loss_sample_rows`, falling back to `validation_sample_rows`; chosen by `evaluation.validation_sample.validation_sample`, stratified by line × reading mode, by a stable hash). It only draws the progress curve: **no early stopping is passed to ms-swift** and **`load_best_model_at_end: false`**. The run is divided into **`evaluations_per_run` (10)** checks (`_eval_interval`: ⌈optimizer steps ÷ 10⌉; without the setting, the configured `eval_steps` shrunk so a run gets at least 5). `_checkpoint_cadence` then saves a resume point at least every **`checkpoint_every_steps` (70)** steps — multiplied by the GPU count, since n GPUs run n× the steps in the same hour — evaluates on every k-th save (`eval_steps` = `save_steps` × k), and sets `save_total_limit` to the configured 4 × k + 1, so the limit counts evaluation checkpoints and resume points never push a candidate out. `metric_for_best_model: eval_loss`, `greater_is_better: false` let the trainer record the best-loss checkpoint, one selection candidate. **Loss never selects what ships** — see §3 below.
- **Staging:** the adapter and its `checkpoint-*` directories go to the RunPod staging volume under `staging/adapters/scope/policy/v{n}/`. Pushed to Blob by `package` (the checkpoint selection chose), or earlier with `finetune --push-adapters`, at the end of checkpoint selection (the same selected checkpoint) — the adapter's weights and config only (`push_scoped_adapter`), never a checkpoint's resume state. Each push records the checkpoint it pushed (`pushed` in the checkpoint selection), which `package --from-blob` checks (IMPL-13). A continued cycle first copies its parent's adapter from Blob to `staging/parent-adapters/policy/v{n}/` and starts from it.
- **Manifest:** a `RunManifest` (IMPL-02) is written at status **`training`** before launch; `launch_and_record` flips it to `trained` when ms-swift returns, or **`failed`** when it raises, so the registry never claims weights a crashed run never wrote. It records the full config, data stats, LoB coverage, seed, git commit, corpus version, MinerU version, schema/prompt template versions, and the de-identification flag (recorded, not asserted: `false` under Option A, no de-identification — IMPL-05 §1). A model card is written beside it (`registry_utils/model_card.py`).
- **The YAML's `evaluation:` block** is read in three places. `build_training_config` passes `eval_strategy`, `save_strategy`, `metric_for_best_model`, `greater_is_better` and `load_best_model_at_end` directly, and derives `eval_steps`, `save_steps` and `save_total_limit` from `evaluations_per_run`, `checkpoint_every_steps` and `save_total_limit`. Staging reads `loss_sample_rows`. Checkpoint selection reads `validation_sample_rows` and the `selection_*` keys (§3). `early_stopping_patience` is still in the YAML and passed nowhere: ms-swift 3 accepts none.

Flags: `--corpus vN`, `--out-version vN`, `--tenant`, `--deepspeed zero2|zero3`, `--train-vit` (refused, see *Current implementation*), `--continue-from <checkpoint dir>`, `--dry-run`. There is no `--scope` (decision 2026-10-22): the run always trains the policy adapter. Through `orchestration.run` the version is the cycle's (IMPL-13).

**Versioning rule (arch §12) — enforced in code:**
- **Every fine-tuning cycle continues the tenant's promoted release** (decision 2026-10-23; until then each cycle was a fresh LoRA from the base): its adapter's weights, trained further on new and existing data, with a fresh optimizer and learning-rate schedule (ms-swift `adapters`). The base stays the original HF base model, never a merged model. `orchestration.pipeline_dag.continuation` decides, and the cycle starts from the base instead when `--fresh` is given (after a data deletion, or to reset drift), nothing is promoted, the parent's run manifest cannot be read, it trained on another base model, it has another LoRA rank, alpha or target modules, its adapter (`adapter_config.json` or `adapter_model.safetensors`) is not in Blob, its `adapter_config.json` cannot be read or was saved with other LoRA settings than the config (`adapter_config_mismatch`: dropout, rsLoRA and bias as well as rank, alpha and target modules; ms-swift would train with the adapter's own), its release has no bundle, or its release was packaged with another prompt hash or window plan (a template, schema or section-map change), or its eval report is missing or on another yardstick (the set frozen again, or other scoring code), which the gate's line rule would block only after training. The reason is logged and recorded on the run manifest (`fresh_reason`) and in the stage result; the parent is `RunManifest.parent_run_id`. The promoted release's bundle is read by its release id (`promoted_release`), so a release with an old-style name (`release-YYYY.M.N`) is continued too. Each version still has its own run, checkpoints, calibration, gate and release; nothing is merged or quantized. The first cycle on the new training set and canonical schema will start fresh (the prompt hash changes); later ones continue.
- **A continued run is held, line by line, to the runs before it** (IMPL-08): continued training compounds drift, so each line is held to the best run on it since the last run from the base, not only to the parent. This replaces the "cross-type regression evidence" rule of the old multi-type Foundation design. `--continue-from` on `training.train` (by hand, from a directory) sets `continued_from` but names no parent run, so the gate blocks it.
- **A continued run that stopped part-way resumes its own checkpoint**, not the parent's adapter again; the run fingerprint names the parent and the weights it starts from (`starting_weights`), so a checkpoint of another start is refused (*Current implementation*, Resume).
- **A real run checks the adapter it continues.** `train()` reads the adapter's `adapter_config.json` (the staged parent's, or `--continue-from`'s) and refuses it when its LoRA settings differ from the config (`assert_adapter_matches_config`), because ms-swift trains a continued run with the adapter's own settings. A dry run reads no adapter.
- **`--continue-from` takes a checkpoint DIRECTORY, never a registry run-id.** It reaches ms-swift 3 as `adapters: [<dir>]` (it used to be `resume_from_checkpoint`, which restored the old run's optimizer, schedule and step), which reads a path; the directory must hold an `adapter_config.json`; given a run-id, ms-swift finds nothing, trains from base, and the manifest records a lineage that never happened. `assert_checkpoint_path` refuses any bare run-id of the shapes `common.run_ids` defines (`policy-v2`, `extractor-v2`, `extractor-v2.1`, …), with or without a trailing slash or `./` prefix.

**What the operator should know about incremental training:**
- Old documents train again every cycle, so over many cycles they are seen more often than new ones — a memorisation risk. The lever is fewer epochs or a lower rate for continued cycles (`continued_training`); the gate on the frozen eval set catches what it costs.
- A wrong label, learned once, carries forward until it is fixed in the data.
- Reproducing a version needs its chain of parents (`parent_run_id`, back to a run from the base).
- Start fresh (`--fresh`) after a data deletion and from time to time to reset drift. A schema, prompt, LoRA-settings or base-model change starts fresh by itself, and so does a change to the scoring code or a set frozen again. The scoring code changed on 2026-10-24, so a cycle continuing a release scored before then starts from the base instead; none is promoted yet.

### 2. `training/merge.py`

`plan_merge` / `merge` fold **one** adapter — the checkpoint §11.2 selected, recorded on the plan — into the bf16 base with PEFT `merge_and_unload()`, on the staging volume. See IMPL-10 §1. **No longer called by the pipeline** (decision 2026-10-23): the release is the adapter, served on the base, and there is no merge stage.

### 3. Checkpoint selection — `evaluation/checkpoint_eval.py` (arch §11.2)

Runs as the `checkpoint_eval` stage, after training and before calibration (there is no merge stage since 2026-10-23).

- **Candidates:** the last 3 checkpoints by step (`TRAILING_CANDIDATES`), plus the best-loss one. After a real run, `discover_checkpoints` reads them from ms-swift's versioned output directory (`…/v{n}/v0-<timestamp>/checkpoint-*`, the most recent run), keeping only checkpoints an evaluation fell on plus the last (resume points saved between evaluations are not candidates), and the best-loss checkpoint from `trainer_state.json`.
- **Scoring:** each candidate is applied as a **decoder LoRA** on the bf16 base in vLLM and generates over a fixed stratified sample of **`validation_sample_rows` (400)** rows of the adapter's validation view, in all three reading modes, through `evaluation/validation_generation.py` — the row's own prompt with the golden answer removed, constrained as serving is, `validation_batch_rows` (96) rows per vLLM call — and is scored by the gate's own `build_report`. The winner has the best `field_normalized_match`; ties go to the later step. A candidate whose scoring raises is skipped and recorded, never scored zero.
- **Whole documents decide among the finalists** (`choose_on_documents`): the `selection_document_finalists` (2) best are scored on every validation row of `selection_document_modes` (`[ocr_plus_image]`; filtered by mode so every document stays whole) and judged on whole documents, windows merged as serving merges them. Then, in three steps (decision 2026-10-24): a finalist whose `document_false_null_rate` is more than `selection_guard_margin` (0.02) above the best finalist's, or whose `document_run_to_run_agreement` is more than it below the best finalist's, cannot win; of the rest, those within the margin of the best `document_field_normalized_match` go on; and the lowest `document_hallucination_rate` wins (ties: the higher field match, then the later step). Agreement is measured by generating the pass a second time with its rows in reverse order (`selection_run_to_run_check`, on; only with 2 or more finalists; one more pass per finalist). With fewer than 2 finalists configured, a near-tie under `selection_tie_break_margin` (0.015) is re-decided on the full split instead (`break_tie`). A mode filter matching no row is refused before the engine loads. A whole common-model document's schema validity is judged as serving would return it (`coverage_metrics.score_schema_validity` → `serving.pipeline.served_schema_verdict`: enveloped, system fields and every key filled, then checked), so a merged answer is not failed for what the pipeline supplies — which had made `document_schema_validity_rate` read a false 0%.
- **On a multi-GPU pod** the candidates are scored at once, one worker and vLLM engine per GPU (`parallel_vllm_scorer`), up to the run's training GPU count.
- **Why not loss:** averaged over every token, it is dominated by schema keys and JSON punctuation the model gets right in the first hundred steps. A checkpoint can improve on loss while getting worse at the values.

### 4. Per-type adapters — only through the §4.2 graduation gate

There is **no per-type training entrypoint** in the default topology. A per-type adapter returns only when a type has **≥500 labeled documents AND a measured >2 pp win outside the paired-bootstrap 95% CI**. Even then it trains on the **merged** foundation, decoder-only, and is applied at serving time as the request's one LoRA — never stacked, never merged. When the adapter moves to a new major version, every graduated adapter must be re-validated; `query_registry.adapters_depending_on()` (IMPL-02) produces the work list. **ACORD** stays one type with per-form schemas (arch §4b); per-form adapters are not considered until a form has 1000+ examples.

### 5. `training/data_collator.py`

**Override hook only.** ms-swift provides multimodal collation and `-100` label masking (system, image and OCR tokens masked; **loss only on the assistant JSON**). This file:
- documents that the framework default is in use;
- contains `assert_masking_correct` / `verify_batch`, which verify masking on a sample batch — the highest-value correctness check in the system, because broken masking trains the model to reproduce its own prompt and is invisible in loss curves;
- implements custom masking only if a genuine need arises (`custom_collator` raises until then).

### 6. `training/callbacks/early_stopping.py`

**Not in the launch path.** `swift_early_stopping_args` (which emitted `early_stopping_patience`) is no longer called: ms-swift 3 accepts no patience argument, and stopping early would only remove candidates from checkpoint selection, which chooses by generated field accuracy over the evaluation checkpoints. `EarlyStoppingState` (a field-F1 tracker) remains in the module, exercised by tests only; neither the launch path nor `checkpoint_eval` imports it.

### 7. `training/vit_gate.py` (decision helper — arch §3)

Given an eval report (IMPL-08) for a frozen-ViT run, decide whether the ViT escalation fires:

```
1. Train with the ViT frozen.
2. Evaluate on (a) the image-only eval subset and (b) the scanned-PDF eval subset.
3. IF image-only OR scanned accuracy is below target
   AND the errors are perception errors (misread characters, missed checkboxes)
   rather than schema/reasoning errors:
      → apply LoRA to the ViT and retrain.
4. ELSE keep it frozen.
```

- The **error-type distinction is the load-bearing part**: a character read correctly but placed in the wrong field is a decoder problem, and touching the ViT won't help. `classify_error` separates perception from schema/reasoning errors, and the gate reads that — not the accuracy number alone.
- **When the ViT is trained, LoRA is applied — never full fine-tuning** (arch §3). `--train-vit` configures a ViT LoRA; no code path enables full ViT fine-tuning.
- Emits a recommendation and rationale. **Does not auto-train.**

### 8. `training/sweep.py` — built; **execution waits for production volume**

The protocol is implemented and tested against a stub trainer. What waits is *running* it: a 9–12 run sweep against 25–30 documents per type mostly measures noise. `assert_enough_data` refuses below **200 documents per type**. It is deliberately **not a pipeline stage** — nine to twelve training runs is an operator decision.

| Phase | Sweeps (the adapter's run — `foundation` grid key) | Held fixed | Metric | Budget |
|---|---|---|---|---|
| **1 — Learning rate** (first, highest impact) | `{5e-5, 1e-4, 2e-4}`, 1 epoch each | everything else | validation loss | 3 runs |
| **2 — Epochs** | `{2, 3, 4}` document passes, patience 2 | Phase 1 winner | validation field F1 | 3 runs |
| **3 — LoRA rank** (**only if F1 plateaus**; off by default) | `{32, 64, 128}`, rsLoRA on | Phases 1–2 winners | validation field F1 | 3 runs |

- The four materialized epoch files exist so the epoch phase compares the same data draws rather than regenerating them.
- The `per_type` grids in `configs/sweeps/` are dormant until a type graduates (§4 above).
- **Every sweep run writes a full `run_manifest.json`** with `is_sweep_run: true`. A candidate that produced no score is excluded from ranking, not defaulted.

## Constraints
- ms-swift is the entrypoint (locked). TRL `SFTTrainer` is a documented migration fallback only.
- **One adapter.** No code path trains a per-type adapter outside the §4.2 graduation gate, and none stacks two LoRAs.
- **ViT and mergers frozen** unless `--train-vit` — and then LoRA-on-ViT, never full fine-tune. The mergers are never a LoRA target.
- Every run writes a manifest — **no silent training, sweeps included**.
- Runs record (not assert) de-identification status — `false` under Option A (IMPL-05 §1); every run asserts its corpus is single-tenant.
- Vision budgets and sequence caps come from `configs/shared/` and must match data prep and serving (the §10.2 parity tests assert this; token parity is measured on the pod, see *Current implementation*).
- Seed is fixed, logged, and recorded in the manifest.

## Datasets, and the split that must stay split

The trainer receives **`--dataset` (the epoch files) and `--val_dataset` (val) as separate arguments**, with `split_dataset_ratio: 0.0`. Passing both to `--dataset` makes ms-swift treat the validation set as training data and carve its own eval split out of the union, so the validation loss — and the best-loss candidate — reads documents the model memorised, and nothing downstream can detect it.

**Argument rendering is part of this contract.** Two rendering rules are load-bearing:

- A `False` value renders as `--flag false`, never as an omitted flag. Dropping it silently disabled every option whose correct value is `False` — `freeze_vit=False` meant `--train-vit` never reached the trainer while the manifest recorded `vit_trainable: true`.
- A list renders as one argv element **per item**. Joining them gives `HfArgumentParser`'s `nargs="+"` one token containing every value, so the epoch files become one nonexistent filename and `lora_target_modules` matches no module.

## Acceptance checklist
- [ ] The adapter's run trains end-to-end on a tiny sample corpus and produces a loadable LoRA adapter (GPU milestone).
- [ ] `dataset` is the first `num_train_epochs` epoch files, ms-swift gets `num_train_epochs: 1`, and fewer files than epochs is refused.
- [ ] Every file training reads exists after a real dataset build (`tests/test_orchestration.py`).
- [ ] Train and val are passed as **separate** datasets; no path puts the validation split into `--dataset`.
- [ ] Under the default config the rendered CLI carries no quantization and no `bnb_4bit_*` argument; `load_in_4bit: true` switches to NF4 (`quant_method bnb`, `quant_bits 4`), and the manifest's `technique` follows.
- [ ] Target modules contain no `merger`; `freeze_aligner` is always true; `--train-vit` renders `--freeze_vit false` and records a ViT **LoRA**.
- [ ] `max_length` covers the largest cap of the adapter's tasks, including the `by_doc_type` overrides of the types it trains.
- [ ] A corpus whose policies mix lines on the common model with lines on self-contained schema files is refused before anything is staged (`assert_one_output_shape`, on the single-line policies the view trains).
- [ ] The view trains policies of one known line only: a package and a policy with no line are left out, and a line is keyed by its schema name (`workers_comp` as `wc`).
- [ ] `group_by_length` renders `false`; a run gets `evaluations_per_run` evaluations, resume points no further apart than `checkpoint_every_steps` × GPUs, and a `save_total_limit` that keeps the last evaluation checkpoints; `load_best_model_at_end` is `false`.
- [ ] The trainer's validation file is the `loss_sample_rows` sample; checkpoint selection ranks on its own `validation_sample_rows` sample and decides among finalists on whole documents.
- [ ] The assistant span **includes** the end-of-turn token, so EOS is supervised.
- [ ] **Label masking verified**: loss only on assistant tokens; a mutated masking implementation fails the test.
- [ ] Every flag in the rendered command is one ms-swift 3's own parser accepts (Phase 0 spike `check_ms_swift` parses the exact command with `HfArgumentParser(TrainArguments)`).
- [ ] On the pod, `tests/test_parity.py::test_template_token_parity` (`pytest -m gpu`) passes: ms-swift's encoder and a vLLM engine on the serving pixel budget give identical prompt token ids and identical per-image token counts, each equal to the resize rule, and the processor's pixels per visual token equal `cap_check.PIXELS_PER_VISUAL_TOKEN`. Off the pod it skips, naming what is missing.
- [ ] The adapter's tasks hold no `page_select`: the task is gone (`Task.PAGE_SELECT` removed).
- [ ] `--continue-from` refuses a run-id; a run continued by hand from a directory names no parent run, so the gate blocks it.
- [ ] A cycle continues the promoted release unless `--fresh` is given or it cannot be continued; the reason is logged and recorded as `fresh_reason`, the parent as `parent_run_id`. A continued run trains with `continued_training`'s non-null values, and its fingerprint names its parent and its starting weights; a checkpoint of another start, or one fingerprinted before the start was recorded, is not resumed.
- [ ] A parent adapter saved with other LoRA settings than the config (dropout, rsLoRA and bias included) is not continued by the pipeline and is refused by `train()`; so is an `adapter_config.json` that cannot be read. A `continued_training` key the `optimization` block lacks, or a non-positive epoch count or learning rate, is refused before any work.
- [ ] Checkpoints are discovered from the versioned output directory after training, and selection scores them with the gate's metric, not loss.
- [ ] Adapter lands on the staging volume; the RunManifest lands in **Blob** with `artifacts.status: "staged"`, LoB coverage, and seed recorded.
- [ ] `vit_gate` fires only when errors are perception errors, never on schema/reasoning errors alone.
- [ ] `sweep.py` refuses below production volume, writes one manifest per candidate with `is_sweep_run: true`, conditions each phase on the previous winner, excludes unscored candidates, and leaves phase 3 off by default.

---

## Current implementation (2026-10-21)

**ms-swift 3 command** (`build_training_config`; `pyproject.toml` pins `ms-swift>=3.9,<4`): `model` (the
local base directory — see below — else the Hub id, with `model_revision` only for a Hub id), `use_hf`,
`dataset` (the first N epoch files), `val_dataset`, `train_type: lora`, `lora_rank`/`lora_alpha`/
`lora_dropout` (64 / 128 / 0.10), `target_modules` (`q/k/v/o_proj`, `gate/up/down_proj`), `use_rslora`
(LoRA `bias` is never passed, which is harmless while it is `none`),
`attn_impl: flash_attn`, `torch_dtype: bfloat16`, `num_train_epochs: 1`, `gradient_accumulation_steps` for
the GPU count, `dataset_num_proc`, `use_logits_to_keep`, `padding_free`, `group_by_length: false`,
`freeze_vit: true`, `freeze_aligner: true`, `split_dataset_ratio: 0.0` (validation is ours, never carved
from train), `eval_strategy`/`save_strategy: steps` with the derived `eval_steps`/`save_steps`/
`save_total_limit`, `load_best_model_at_end: false`, `metric_for_best_model: eval_loss`, `report_to: none`,
`adapters` for a continued run (the parent's staged adapter, or `--continue-from`), `truncation_strategy: delete` with `strict: true` (a row over `max_length`
is an error, never silently shortened, and never silently swapped for another row). No `model_type`
(ms-swift infers it). `--train-vit` is **refused**: the LoRA targets are decoder projection names and
ms-swift applies `freeze_vit=false` only when expanding `all-linear`, so it trained the same decoder-only
adapter while recording `vit_trainable`.

**Several GPUs** (`distributed.gpus: auto` = every visible GPU, or a number; `training_gpus`): data-parallel,
each GPU taking whole rows. The launch env gets `NPROC_PER_NODE=n` (ms-swift starts one process per GPU) and
the command `ddp_find_unused_parameters: false`; accumulation shrinks so the effective batch, the step count
and the schedule are those of one GPU.

**Resume.** An unfinished run picks up where it stopped. `resume_point` finds the latest complete
`checkpoint-*` (trainer state, adapter and optimizer state) under the staging directory of a run without
`training_complete.json` — the marker `launch_and_record` writes only when ms-swift returns successfully, and
what "trained" means to the pipeline (ms-swift creates the directory when training starts).
`assert_resumable` refuses the resume when `run_fingerprint.json` differs: corpus version and manifest hash,
the adapter's name, the parent run (`parent_run_id`; a run continued from v3 and one from the base are not one run), the starting weights (`starting_weights`: the sha256 of the parent adapter's `adapter_model.safetensors`, else its directory, `None` from the base, so a run continued by hand is not taken for one from the base), LoRA rank/alpha, learning rate and schedule, epochs, effective batch, target modules, and a hash of the
training config (minus `distributed` and the per-device batch) and the shared vision and sequence configs.
A fingerprint that lacks a key this run records (one written before `starting_weights`) is refused too; a run
with no fingerprint file resumes, with a warning. A dry run computes no fingerprint.
Resumed, ms-swift gets `resume_from_checkpoint`, the old run directory as `output_dir` and
`add_version: false` — the same run carried on, unlike `--continue-from`, which starts a new run from the
weights. A continued cycle resumed this way drops `adapters`: its own checkpoint already started from the
parent's weights.

**Staging** (`training/stage_data.py`): the corpus rows are rewritten into ms-swift's own format —
string content with `<image>` placeholders and an `images` list — as local JSONL under
`staging/train-data/policy/v{n}/`, and their page images are fetched once into a per-corpus-version
cache (`staging/train-images/{tenant}/{corpus}/`, atomic unique `.part` downloads). The validation rows are
narrowed to the loss sample before conversion (`val_filter`, which needs the line and mode the conversion
drops); every row's output reservation is kept beside the files for the length check (`output_caps`); a row
whose task has another pixel budget than the run's is refused before any download. Older corpus versions'
caches are pruned (the current one and the most recent other are kept). A run is refused if its run id
already exists in any state but `training`/`failed`, and a real run is refused when the adapter's view has
no validation rows.

**Before launch, on the pod**:
1. **Length** (`_assert_rows_fit`): every staged row is measured with the real tokenizer and the real resize
   rule (smart_resize, factor 32) against `max_length` **and** against its task's reserved output tokens
   (`training/length_check.py`); any over is refused with the rows named.
2. **Masking** (`_assert_masking`): the first staged rows are encoded with ms-swift's own template
   (`_masking_encoder`, ms-swift imported in-process) and `data_collator.verify_staged_rows` asserts exactly
   the assistant content plus `<|im_end|>` is supervised. ms-swift's import replaces the root log handlers,
   so `common.logging_setup.reassert()` follows it; without it every later stage in the process — checkpoint
   selection included — logged nothing.
3. **Where**: `assert_on_pod` refuses a real launch without CUDA (`common/gpu.py`), without the staging
   volume, or when `model.local_dir` is configured and holds no model. `FIDEON_ALLOW_OFF_POD=1` opts a
   non-RunPod GPU host in.

Earlier, before anything is staged: `training.data_mix.assert_corpus_mix` refuses a training config whose
input-mode mix differs from the corpus's `modality_mix_target`, and a corpus that would teach the adapter two
output shapes is refused (`common.adapter.assert_one_output_shape`): policies read against the SPEC_21 common
model beside policies read against self-contained schema files. `training.train` checks the documents the
view really trains (`corpus_view.trained_lobs`), before anything is staged: single-line policies only, since
the view trains no package and no policy with no line (below). The corpus's `policy_line_counts` would count
a package's lines one by one, so it is not used for this. Today the lines on the common model are the seven
personal lines and `gl`; a corpus holding `auto` or `wc` policies beside them is refused until those lines
move onto the common model or their documents are left out of the training set. This replaces the per-family
check. The same check runs earlier too: the pre-upload audit blocks such a folder (`output shape`, IMPL-04),
where `prepare_bundles --lines` can still leave the documents out, and the preflight fails on the tenant's
imported policies (IMPL-13).

**Pixel budget**: one budget per run from `common.config.pixel_budget(ADAPTER.tasks)`, passed to ms-swift as
both `MAX_PIXELS`/`MIN_PIXELS` and `IMAGE_MAX_TOKEN_NUM`/`IMAGE_MIN_TOKEN_NUM` (whole 32×32 tokens), the
same function the vLLM engine takes its budget from.

**Train/serve token parity, measured on the pod** (arch §10.2): `tests/test_parity.py::test_template_token_parity`
(marked `gpu`) encodes ten fixed rows — or the first rows of the staged epoch file `$FIDEON_PARITY_CORPUS`
names — with the trainer's own ms-swift encoder (`_masking_encoder` over `build_training_config`) and with a
vLLM engine built on the serving pixel budget (`pixel_budget()`), and asserts identical prompt token ids up to
the assistant header and identical per-image visual-token counts, each equal to the resize rule the sequence
caps use (`training.length_check.image_tokens`). It also measures pixels per visual token from the processor
(patch × merge, squared) against `cap_check.PIXELS_PER_VISUAL_TOKEN` (1,024). Without a CUDA GPU, the pinned
base weights, ms-swift or vLLM it skips, naming what is missing. What is compared lives in `training/parity.py`
(`prompt_ids`, `image_pad_runs`, `from_swift_row`, `flat_ids`) and is tested without a GPU; the Phase 0 spike
(`check_swift_row_format`) compares token ids the same way, not decoded text. It passes a plain-string
system prompt as one text block (the pinned processor reads every content as a list of blocks; Qwen3-VL's
template renders the two alike), and any error tokenizing the serving prompt is a FAIL, never "unknown". Serving output parity
(`test_serving_output_parity`) is still a placeholder that skips.

**Every page is read** (decision 2026-10-19): a policy trains on its windows over every page (IMPL-05), so a
long policy has more windows than when pages were selected and a run over the same documents is longer. A
corpus built before this change must be rebuilt and the adapter retrained before the next gate; earlier gate
numbers are not comparable with new ones. The corpus holds no `page_select` rows: the task, with its sequence
and vision budgets, is gone (`Task.PAGE_SELECT` removed), so no task list holds it.

**Base model**: loaded from `model.local_dir` in `configs/base_model.yaml` (default `/workspace/models`;
the directory itself, a folder named after the model, or a Hugging Face cache snapshot), overridable with
`FIDEON_BASE_MODEL_DIR`. The manifest still records `model_id@revision` as the identity.

**Environment**: the ms-swift subprocess gets the pod environment minus credentials it has no use for
(Azure, RunPod, anything named `*SECRET*`, `*PASSWORD*`, `*API_KEY*`, …); `HF_TOKEN` is kept.

**Manifest**: `data_stats.train_examples` is the actual row count of the epochs used; `tenant_id` is
recorded. The run records its view's `val_examples` and `test_examples`, `examples_by_line`, and
`data_mix` (the shares configured and reached per line, documents rested, the render rotation applied);
the model card shows the same.

**On the pod** `training.train` and `training.sweep` run detached in tmux like every long job.

**The adapter's view of the corpus** (`training/corpus_view.py`, `materialize(corpus_version, client)`):
always written, under `corpus/{tenant}/{v}/train/scope/policy/` and `val/scope/policy/`, holding the
corpus's **policy rows of one known line** (`corpus_view._filter`, `common.lob.single_line`; decision
2026-10-22; there is no longer a path that reads the corpus files directly). A package (several lines) and a
policy with no line are left out: they are read against `_fallback.json`, a shape the adapter is not taught,
and serving sends them to the base model (`ServedRelease.covers_lob`, IMPL-11). Before the code review of
2026-10-23 a package of self-contained lines trained. There is no other line filter. Lines are keyed by their
schema name (`corpus_view._line_of`: `workers_comp` is `wc`, `classic_auto` is `personal_auto`), so the data
mix's per-line settings and the release's lines match them. The split and the group assignment are the corpus's own; train, val (hence
checkpoint selection and calibration) and the test count all follow the view. On the train rows, from the
first epoch's documents and the same for every epoch, in this order:

1. **Data mix** (`training/data_mix.py`, `data_mix:` in `configs/training/unified.yaml`): per
   line, `synthetic_fraction` and `scanned_share` of its train documents. Every real document trains;
   synthetic ones are taken one twin of every seed before a second of any (seeded); when a scanned or digital
   pool runs short, fewer synthetic documents are taken rather than the scanned share missed. A per-line
   `null` takes the default; a default of `null` leaves the share unsteered. All values are `null`
   today, so every document trains. A `scanned_share` on rows without `is_scanned` is refused.
2. **Render rotation** (`render_rotation: true`, `plan_render_rotation`): a twin delivered digital and
   scanned with one gold trains in one render per epoch and the other the next; half of a seed's twins start
   in each render. Real documents and single-render twins train every epoch; a line that rotation would take
   under the line-balance target keeps both renders.
3. **Line balance** (`line_balance: {min_documents: 100, max_repeat: 3}`, `line_repeats`): a line with fewer
   documents per epoch than the target (`min_documents`, capped at the largest line) has its rows repeated
   ⌈target ÷ documents⌉ times, at most 3, in every epoch file.

Validation and test are never sampled, rotated or repeated. The CGL documents still train in their own tenant
(`--tenant`), now as the policy adapter: one line, real documents only, so the three change nothing there. The
architecture records that the CGL adapter's v1 was trained and scored on golds nobody reviewed; a gold with no
review record no longer reaches a corpus or the gate (decision 2026-10-22; IMPL-05, IMPL-08), so they are
reviewed before it trains again.

## Changes since 2026-10-05

- Added the alignment line (commit f5668a4).
- Run type and id: scoped runs (`scoped`, `{lineage}-v{n}`) beside unified `extractor-v{n}`;
  `assert_checkpoint_path` refuses every `common.run_ids` shape, scoped ids included.
- Config: scopes name their training config; `personal_lines.yaml` extends unified with its `data_mix`.
- Optimisation and memory: accumulation derived from the GPU count; `dataset_num_proc`; `group_by_length` is
  now false (`length_grouped_sampling: false`), with the reason; `max_length` scoped to the scope's tasks.
- Datasets: scoped epoch and validation files; the staged validation file is the loss sample.
- Evaluation: replaced "≥5 evaluations, `eval_steps` = `save_steps`, `load_best_model_at_end`" with
  `evaluations_per_run` 10, `_checkpoint_cadence` (`checkpoint_every_steps` 70 × GPUs, `save_total_limit`
  counted in evaluation checkpoints), `load_best_model_at_end: false`, no early stopping, and
  `loss_sample_rows` 160 for the trainer's validation file; corrected the "every evaluation key reaches
  ms-swift" bullet (`early_stopping_patience` is unused).
- `--push-adapters` publishes adapter files only; model card noted; `--scope` flag listed.
- §3 checkpoint selection: candidates exclude resume points; ranking on the 400-row sample; whole-document
  finalists (2, `ocr_plus_image`) with hallucination/false-null guards; near-tie fallback; parallel scoring.
- §6: `EarlyStoppingState` is not used by `checkpoint_eval` (tests only).
- Current implementation: full ms-swift argument list, multi-GPU launch, resume with fingerprint and
  completion marker, staging details, `_assert_rows_fit`/`_assert_masking` with `logging_setup.reassert()`,
  `assert_corpus_mix` and `assert_one_output_shape`, scoped manifest stats, and the scope view's data mix,
  render rotation and line balance; casualty_fleet added.
- Checklist: scoped `max_length`, evaluation cadence and the loss/selection samples.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx` (§7b, §10.1, §10.2 named); de-identification is Option A (none), not "blocked".
- Locked stack: the fallback clause cites arch §10.1 — a migration to TRL `SFTTrainer` — and says ms-swift's `Seq2SeqTrainer` is Layer 2, not the fallback.
- §1 manifest and Constraints: the de-identification flag is recorded as `false` under Option A; token parity is measured on the pod.
- §3: checkpoint selection judges a whole common-model document's schema validity with `served_schema_verdict`.
- Acceptance checklist: the GPU token-parity test; no scope declares `page_select`.
- Current implementation: token parity on the pod (`training/parity.py`, `test_template_token_parity`, GPU-marked, the spike comparing token ids); every page read, so more windows per policy, a longer run, a corpus rebuild and retrain of both adapters, and no `page_select` rows; casualty_fleet v1's unreviewed golds.

## Changes of 2026-10-22 (code review)

- Phase 0 spike: a plain-string content goes in as one text block; a tokenizing error is a FAIL with its traceback.
- casualty_fleet: a gold with no review record reaches neither a corpus nor the gate (decision 2026-10-22).
- Flags: name a policy scope while only policies are targeted (decision 2026-10-22).
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.

## Changes of 2026-10-22 (one adapter)

- One adapter (decision 2026-10-22; Fideon SPEC_09's layout-family groups knowingly not followed): every run is `policy-v{n}`, trained fresh on the base each cycle; no `--scope` and no `personal_lines.yaml` (`unified.yaml` holds the data mix); the view of the corpus is always written, policy rows only, with no line filter (a package of common-model lines left out); `assert_one_output_shape` refuses training documents read against both shapes, replacing the per-family check; casualty_fleet and the scope-by-line text removed.
- Code review of 2026-10-23: the view keeps policies of one known line only (a package and a policy with no line are left out, as serving sends them to the base model) and keys lines by schema name; the output-shape check reads those single-line policies, and the audit and the preflight run it earlier. New acceptance item.

## Changes of 2026-10-23 (incremental training)

- Incremental training (the user's decision of 2026-10-23), in the goal and §1: a cycle continues the promoted release's adapter on new and existing data, on the original base, with a fresh optimizer and schedule, and starts from the base with `--fresh` or when it cannot continue (`fresh_reason`); the parent is `parent_run_id`, its adapter staged under `parent-adapters/`; the `continued_training` block (null today) for continued runs; a continued run resumes its own checkpoint, its fingerprint naming the parent.
- The cross-type regression rule is gone: a continued run is held line by line to the runs before it (IMPL-08), and one continued by hand from a directory is blocked. What the operator should know: memorisation of old documents, wrong labels carried forward, the chain of parents, when to start fresh.
- §2 and §3: `training/merge.py` is no longer called by the pipeline; checkpoint selection is followed by calibration. Acceptance items updated.
- Code review of the incremental training: a cycle also starts from the base when the parent's adapter files are not in Blob, its `adapter_config.json` cannot be read or holds other LoRA settings, or its release has no bundle, and a release with an old-style name is continued; `train()` checks the adapter it continues (`assert_adapter_matches_config`); the fingerprint names the starting weights; `continued_training` is validated; `--push-adapters` pushes the selected checkpoint at the end of checkpoint selection; LoRA `bias` is not passed to ms-swift. Acceptance items updated.

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.

## Changes of 2026-10-24 (printed values only and consistency)

- §3 checkpoint selection (the user's decisions of 2026-10-24: nothing invented, consistent output): guards on false nulls and on run-to-run agreement (the finalists' pass generated again in reverse order, `selection_run_to_run_check`), then the finalists within the guard margin on field match, then the fewest invented values; `document_hallucination_rate` is no longer a guard. Invented values are now looked for on every page of the document with the OCR-tolerant matcher serving uses (IMPL-08).
- Continuing: the scoring code changed, so a cycle continuing a release scored before 2026-10-24 starts from the base; none is promoted yet.

## Changes of 2026-10-24 (QA fixes)

- A resumed run keeps the checkpoint's own save and eval cadence and `save_total_limit` (`training.train.resumed_cadence`): on more GPUs the limit shrank and rotation deleted earlier evaluation checkpoints. Checkpoint discovery reads eval steps from `log_history`.
- A cycle starts fresh, the reason recorded, when the parent chain's training documents cannot be read, or when its corpus would validate on documents the chain trained on (IMPL-05).

## Changes of 2026-10-24 (architecture v2.7)

- Header: aligned with `finetuning-architecture-v2.7.docx`, which records the decisions of 2026-10-23 and 2026-10-24: printed values only, consistent output, fields first, one MinerU, production on Azure.

## Changes of 2026-10-25 (fast format profile, Azure service)

- Under the fast format profile (`configs/shared/format.yaml`, IMPL-01) the trainer resizes pages to 1,254,400 px (the trainer env reads `vision_for_task`), the corpus is drawn with 70% of its rows `image_only` (IMPL-05), and checkpoint selection reads `ocr_plus_image` and `image_only` (`selection_document_modes`). Validation decodes compact rows against the compact schema and expands them before scoring (`evaluation.validation_generation`, IMPL-08). An adapter trained in one profile is refused by promote and serving in the other.
