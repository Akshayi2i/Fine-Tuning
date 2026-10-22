# SPEC 06 — Training (ms-swift LoRA: ONE Unified Adapter)

> Aligned with the code of 2026-10-21.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-05.
>
> **Architecture refs:** `finetuning-architecture-v2.4.docx` §3 (ViT escalation gate — **LoRA, never full FT**), §4.1 (**one unified adapter**), §4.2 (graduation gate for per-type adapters), §6.1 (per-epoch modality sampling), §7b (policy windows over every page), §8b (tenancy per Fideon SPEC_12; de-identification per Fideon SPEC_11 — Option A, none, see IMPL-05 §1), **§9.2 (bf16 base vs 4-bit)**, §9.3 (memory settings), §9.4 / §9a (frozen ViT and mergers; LoRA targets), §10 (**locked three-layer trainer stack**; §10.1 fallback, §10.2 parity tests), §11 (hyperparameters), §11.2 (checkpoint selection), §11a (**sweep methodology**), §12 (versioning + run registry).

## Goal

Fine-tune Qwen3-VL-8B-Instruct with **LoRA on a bf16 base** via the locked trainer stack. **One adapter**, across every document type and every task, with the ViT *and* the mergers frozen, and every run — sweeps included — recorded in the registry.

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

### 1. `training/train.py` — the unified extractor run

Replaces v1's `train_foundation.py` and `train_adapter.py`. One entrypoint. The unified scope's run type is `unified` with run id `extractor-v{n}`; a narrower scope (`configs/scopes.yaml`) records run type `scoped` and run id `{lineage}-v{n}` — still one independent adapter on the base, never stacked.

- **Config:** `configs/training/unified.yaml` + `configs/base_model.yaml`. There are no per-type training configs. A scope names its training config (`training_config` in `configs/scopes.yaml`); `configs/training/personal_lines.yaml` declares `extends: unified` and adds only its `data_mix`, merged over unified by `common.config.training_config`, so the two cannot drift on LoRA, optimisation, batching or evaluation.
- **Base precision** from `training/base_precision.py`. Default is a **bf16 frozen base**: no quantization argument at all (ms-swift 3 has no `quantization_bit`), and every `bnb_4bit_*` argument suppressed so the run log never carries settings describing nothing the run did. Under `load_in_4bit` it renders `quant_method: bnb`, `quant_bits: 4` and the `bnb_4bit_*` set. `load_in_4bit: true` in `configs/base_model.yaml` switches to NF4 with double quantization and bf16 compute. The manifest's `technique` (`LoRA`/`QLoRA`) and `base_quantization` are derived from the same config, never defaulted.
- **LoRA:** rank 64 / alpha 128, dropout **0.10** at pilot scale (0.05 from ≥200 docs per type), `bias: none`. Targets are the attention and MLP projections of the **language-model decoder only** — `q/k/v/o_proj`, `gate/up/down_proj`. No `merger`.
- **Frozen:** `freeze_vit: true` (unless `--train-vit`, the §3 escalation — LoRA-on-ViT, never full fine-tune) and `freeze_aligner: true` always.
- **Optimisation (§11.1):** AdamW (`adamw_torch`), LR **1e-4**, cosine, **`warmup_steps: 10`** (steps, not a ratio — a ratio rounds to zero warmup at pilot volume), weight decay **0.0** (decay on a low-rank adapter is a shrinkage prior on the update, not regularisation), max grad norm 1.0, **effective batch 8** (`batch.effective_batch_size`) at per-device batch 1, the accumulation derived from the GPU count (`_accumulation_steps`: effective ÷ (per-device × GPUs), refused when it does not divide), gradient checkpointing, bf16. `dataset_num_proc` 8 for ms-swift's preprocessing, capped at the pod's CPU count.
- **Memory (§9.3):** `use_logits_to_keep`, `padding_free`. **`group_by_length` is false** (`memory.length_grouped_sampling: false`): with one row per device and padding-free batching there is no padding to save, and with it on ms-swift encoded every row before step 1 to learn its length (~1.5 h on the CGL run). Rows are visited in a plain random order: same data, same steps. `max_length` is the largest cap of **the scope's tasks**, including the `by_doc_type` overrides of the types it trains (`corpus_max_length(scope)`; 32,768 for any policy scope). If a cap does not fit one card, **sequence parallelism comes first**; DeepSpeed is off by default, because under LoRA ZeRO-2 shards ~1.5 GB of adapter optimizer state and is close to a no-op.
- **Datasets.** **Epochs are files, and ms-swift runs one pass over them** (§6.1). `dataset` is the first `num_train_epochs` (3) of the scope's epoch files — `corpus/{tenant}/v{n}/train/epoch_{1..4}.jsonl` for the unified scope, the view under `train/scope/{name}/` otherwise — and ms-swift is given **`num_train_epochs: 1`**. Each file already holds every train document once in that epoch's modality draw, so the concatenation *is* the N-epoch run; also telling ms-swift N looped the N files N times — nine passes for a "3 epoch" run. The manifest records the logical count. Fewer epoch files than epochs is refused. `val_dataset` is the scope's validation file (`corpus/{tenant}/v{n}/val/val.jsonl`, or `val/scope/{name}/val.jsonl`), staged as the fixed loss sample (see *Evaluation*) and passed separately (see below). The corpus is read for `--tenant`, never silently from the default tenant. (ms-swift shuffles across the concatenation, so epoch-2 rows can precede epoch-1 rows; each document is still seen exactly N times, each time in its drawn regime.)
- **Evaluation during training** is validation **loss** on a fixed stratified sample of **`evaluation.loss_sample_rows` (160)** validation rows (`train.loss_sample_rows`, falling back to `validation_sample_rows`; chosen by `evaluation.validation_sample.validation_sample`, stratified by line × reading mode, by a stable hash). It only draws the progress curve: **no early stopping is passed to ms-swift** and **`load_best_model_at_end: false`**. The run is divided into **`evaluations_per_run` (10)** checks (`_eval_interval`: ⌈optimizer steps ÷ 10⌉; without the setting, the configured `eval_steps` shrunk so a run gets at least 5). `_checkpoint_cadence` then saves a resume point at least every **`checkpoint_every_steps` (70)** steps — multiplied by the GPU count, since n GPUs run n× the steps in the same hour — evaluates on every k-th save (`eval_steps` = `save_steps` × k), and sets `save_total_limit` to the configured 4 × k + 1, so the limit counts evaluation checkpoints and resume points never push a candidate out. `metric_for_best_model: eval_loss`, `greater_is_better: false` let the trainer record the best-loss checkpoint, one selection candidate. **Loss never selects what ships** — see §3 below.
- **Staging:** the adapter and its `checkpoint-*` directories go to the RunPod staging volume under `staging/adapters/foundation/v{n}/` (unified; `staging/adapters/scope/{scope}/v{n}/` for a scoped run). Pushed to Blob by `package` (the checkpoint selection chose), or immediately with `finetune --push-adapters` (the best-loss checkpoint, else the last, before selection) — the adapter's weights and config only (`push_scoped_adapter`), never a checkpoint's resume state.
- **Manifest:** a `RunManifest` (IMPL-02) is written at status **`training`** before launch; `launch_and_record` flips it to `trained` when ms-swift returns, or **`failed`** when it raises, so the registry never claims weights a crashed run never wrote. It records the full config, data stats, LoB coverage, seed, git commit, corpus version, MinerU version, schema/prompt template versions, and the de-identification flag (recorded, not asserted: `false` under Option A, no de-identification — IMPL-05 §1). A model card is written beside it (`registry_utils/model_card.py`).
- **The YAML's `evaluation:` block** is read in three places. `build_training_config` passes `eval_strategy`, `save_strategy`, `metric_for_best_model`, `greater_is_better` and `load_best_model_at_end` directly, and derives `eval_steps`, `save_steps` and `save_total_limit` from `evaluations_per_run`, `checkpoint_every_steps` and `save_total_limit`. Staging reads `loss_sample_rows`. Checkpoint selection reads `validation_sample_rows` and the `selection_*` keys (§3). `early_stopping_patience` is still in the YAML and passed nowhere: ms-swift 3 accepts none.

Flags: `--corpus vN`, `--out-version vN`, `--tenant`, `--scope NAME` (defaults to unified), `--deepspeed zero2|zero3`, `--train-vit` (refused, see *Current implementation*), `--continue-from <checkpoint dir>`, `--dry-run`.

**Versioning rule (arch §12) — enforced in code:**
- **Major corpus expansion** → retrain **from the original HF base model** on the full accumulated corpus. Continued training on an existing LoRA compounds drift across cycles.
- **Minor incremental patch** → continuing from the current checkpoint is acceptable, **but promotion requires cross-type regression evidence** against the frozen golden eval set (IMPL-08). `--continue-from` sets `continued_from` on the manifest, which the gate reads.
- **`--continue-from` takes a checkpoint DIRECTORY, never a registry run-id.** It reaches ms-swift 3 as `adapters: [<dir>]` (it used to be `resume_from_checkpoint`, which restored the old run's optimizer, schedule and step), which reads a path; the directory must hold an `adapter_config.json`; given a run-id, ms-swift finds nothing, trains from base, and the manifest records a lineage that never happened. `assert_checkpoint_path` refuses any bare run-id of the shapes `common.run_ids` defines (`extractor-v2`, `extractor-v2.1`, scoped ids such as `policy-v2`, …), with or without a trailing slash or `./` prefix.

### 2. `training/merge.py`

`plan_merge` / `merge` fold **one** adapter — the checkpoint §11.2 selected, recorded on the plan — into the bf16 base with PEFT `merge_and_unload()`, on the staging volume. See IMPL-10 §1.

### 3. Checkpoint selection — `evaluation/checkpoint_eval.py` (arch §11.2)

Runs as the `checkpoint_eval` stage, between training and merge.

- **Candidates:** the last 3 checkpoints by step (`TRAILING_CANDIDATES`), plus the best-loss one. After a real run, `discover_checkpoints` reads them from ms-swift's versioned output directory (`…/v{n}/v0-<timestamp>/checkpoint-*`, the most recent run), keeping only checkpoints an evaluation fell on plus the last (resume points saved between evaluations are not candidates), and the best-loss checkpoint from `trainer_state.json`.
- **Scoring:** each candidate is applied as a **decoder LoRA** on the bf16 base in vLLM and generates over a fixed stratified sample of **`validation_sample_rows` (400)** rows of the scope's validation view, in all three reading modes, through `evaluation/validation_generation.py` — the row's own prompt with the golden answer removed, constrained as serving is, `validation_batch_rows` (96) rows per vLLM call — and is scored by the gate's own `build_report`. The winner has the best `field_normalized_match`; ties go to the later step. A candidate whose scoring raises is skipped and recorded, never scored zero.
- **Whole documents decide among the finalists** (`choose_on_documents`): the `selection_document_finalists` (2) best are scored on every validation row of `selection_document_modes` (`[ocr_plus_image]`; filtered by mode so every document stays whole) and judged on whole documents, windows merged as serving merges them. The best `document_field_normalized_match` wins among finalists whose `document_hallucination_rate` and `document_false_null_rate` are within `selection_guard_margin` (0.02) of the best finalist's. With fewer than 2 finalists configured, a near-tie under `selection_tie_break_margin` (0.015) is re-decided on the full split instead (`break_tie`). A mode filter matching no row is refused before the engine loads. A whole common-model document's schema validity is judged as serving would return it (`coverage_metrics.score_schema_validity` → `serving.pipeline.served_schema_verdict`: enveloped, system fields and every key filled, then checked), so a merged answer is not failed for what the pipeline supplies — which had made `document_schema_validity_rate` read a false 0%.
- **On a multi-GPU pod** the candidates are scored at once, one worker and vLLM engine per GPU (`parallel_vllm_scorer`), up to the run's training GPU count.
- **Why not loss:** averaged over every token, it is dominated by schema keys and JSON punctuation the model gets right in the first hundred steps. A checkpoint can improve on loss while getting worse at the values.

### 4. Per-type adapters — only through the §4.2 graduation gate

There is **no per-type training entrypoint** in the default topology. A per-type adapter returns only when a type has **≥500 labeled documents AND a measured >2 pp win outside the paired-bootstrap 95% CI**. Even then it trains on the **merged** foundation, decoder-only, and is applied at serving time as the request's one LoRA — never stacked, never merged. When the unified run moves to a new major version, every graduated adapter must be re-validated; `query_registry.adapters_depending_on()` (IMPL-02) produces the work list. **ACORD** stays one type with per-form schemas (arch §4b); per-form adapters are not considered until a form has 1000+ examples.

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

| Phase | Sweeps (the unified run — `foundation` grid key) | Held fixed | Metric | Budget |
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
- [ ] The unified run trains end-to-end on a tiny sample corpus and produces a loadable LoRA adapter (GPU milestone).
- [ ] `dataset` is the first `num_train_epochs` epoch files, ms-swift gets `num_train_epochs: 1`, and fewer files than epochs is refused.
- [ ] Every file training reads exists after a real dataset build (`tests/test_orchestration.py`).
- [ ] Train and val are passed as **separate** datasets; no path puts the validation split into `--dataset`.
- [ ] Under the default config the rendered CLI carries no quantization and no `bnb_4bit_*` argument; `load_in_4bit: true` switches to NF4 (`quant_method bnb`, `quant_bits 4`), and the manifest's `technique` follows.
- [ ] Target modules contain no `merger`; `freeze_aligner` is always true; `--train-vit` renders `--freeze_vit false` and records a ViT **LoRA**.
- [ ] `max_length` covers the largest cap of the scope's tasks, including the `by_doc_type` overrides of the types it trains.
- [ ] `group_by_length` renders `false`; a run gets `evaluations_per_run` evaluations, resume points no further apart than `checkpoint_every_steps` × GPUs, and a `save_total_limit` that keeps the last evaluation checkpoints; `load_best_model_at_end` is `false`.
- [ ] The trainer's validation file is the `loss_sample_rows` sample; checkpoint selection ranks on its own `validation_sample_rows` sample and decides among finalists on whole documents.
- [ ] The assistant span **includes** the end-of-turn token, so EOS is supervised.
- [ ] **Label masking verified**: loss only on assistant tokens; a mutated masking implementation fails the test.
- [ ] Every flag in the rendered command is one ms-swift 3's own parser accepts (Phase 0 spike `check_ms_swift` parses the exact command with `HfArgumentParser(TrainArguments)`).
- [ ] On the pod, `tests/test_parity.py::test_template_token_parity` (`pytest -m gpu`) passes: ms-swift's encoder and a vLLM engine on the serving pixel budget give identical prompt token ids and identical per-image token counts, each equal to the resize rule, and the processor's pixels per visual token equal `cap_check.PIXELS_PER_VISUAL_TOKEN`. Off the pod it skips, naming what is missing.
- [ ] No scope declares `page_select`: a task list naming it is refused at load as an unknown task.
- [ ] `--continue-from` refuses a run-id and marks the manifest so the gate demands cross-type regression evidence.
- [ ] Checkpoints are discovered from the versioned output directory after training, and selection scores them with the gate's metric, not loss.
- [ ] Adapter lands on the staging volume; the RunManifest lands in **Blob** with `artifacts.status: "staged"`, LoB coverage, and seed recorded.
- [ ] `vit_gate` fires only when errors are perception errors, never on schema/reasoning errors alone.
- [ ] `sweep.py` refuses below production volume, writes one manifest per candidate with `is_sweep_run: true`, conditions each phase on the previous winner, excludes unscored candidates, and leaves phase 3 off by default.

---

## Current implementation (2026-10-21)

**ms-swift 3 command** (`build_training_config`; `pyproject.toml` pins `ms-swift>=3.9,<4`): `model` (the
local base directory — see below — else the Hub id, with `model_revision` only for a Hub id), `use_hf`,
`dataset` (the first N epoch files), `val_dataset`, `train_type: lora`, `lora_rank`/`lora_alpha`/
`lora_dropout` (64 / 128 / 0.10), `target_modules` (`q/k/v/o_proj`, `gate/up/down_proj`), `use_rslora`,
`attn_impl: flash_attn`, `torch_dtype: bfloat16`, `num_train_epochs: 1`, `gradient_accumulation_steps` for
the GPU count, `dataset_num_proc`, `use_logits_to_keep`, `padding_free`, `group_by_length: false`,
`freeze_vit: true`, `freeze_aligner: true`, `split_dataset_ratio: 0.0` (validation is ours, never carved
from train), `eval_strategy`/`save_strategy: steps` with the derived `eval_steps`/`save_steps`/
`save_total_limit`, `load_best_model_at_end: false`, `metric_for_best_model: eval_loss`, `report_to: none`,
`adapters` for `--continue-from`, `truncation_strategy: delete` with `strict: true` (a row over `max_length`
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
scope, LoRA rank/alpha, learning rate and schedule, epochs, effective batch, target modules, and a hash of the
training config (minus `distributed` and the per-device batch) and the shared vision and sequence configs.
Resumed, ms-swift gets `resume_from_checkpoint`, the old run directory as `output_dir` and
`add_version: false` — the same run carried on, unlike `--continue-from`, which starts a new run from the
weights.

**Staging** (`training/stage_data.py`): the corpus rows are rewritten into ms-swift's own format —
string content with `<image>` placeholders and an `images` list — as local JSONL under
`staging/train-data/{scope}/v{n}/`, and their page images are fetched once into a per-corpus-version
cache (`staging/train-images/{tenant}/{corpus}/`, atomic unique `.part` downloads). The validation rows are
narrowed to the loss sample before conversion (`val_filter`, which needs the line and mode the conversion
drops); every row's output reservation is kept beside the files for the length check (`output_caps`); a row
whose task has another pixel budget than the run's is refused before any download. Older corpus versions'
caches are pruned (the current one and the most recent other are kept). A run is refused if its run id
already exists in any state but `training`/`failed`, and a real run is refused when its scope has no
validation rows.

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

Earlier, before anything is staged: `training.data_mix.assert_corpus_mix` refuses a scope whose input-mode
mix differs from the corpus's `modality_mix_target`, and the training stage refuses a scope that would train
one layout family on two output shapes (`common.scopes.assert_one_output_shape`, read against the corpus's
`policy_line_counts`: a family part-way onto the common model, such as casualty_fleet with gl moved).

**Pixel budget**: one budget per run from `common.config.pixel_budget(scope.tasks)`, passed to ms-swift as
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
corpus built before this change must be rebuilt and both adapters retrained before the next gate; earlier gate
numbers are not comparable with new ones. The corpus holds no `page_select` rows: the task, with its sequence
and vision budgets, is gone (`Task.PAGE_SELECT` removed), so no scope declares it.

**Base model**: loaded from `model.local_dir` in `configs/base_model.yaml` (default `/workspace/models`;
the directory itself, a folder named after the model, or a Hugging Face cache snapshot), overridable with
`FIDEON_BASE_MODEL_DIR`. The manifest still records `model_id@revision` as the identity.

**Environment**: the ms-swift subprocess gets the pod environment minus credentials it has no use for
(Azure, RunPod, anything named `*SECRET*`, `*PASSWORD*`, `*API_KEY*`, …); `HF_TOKEN` is kept.

**Manifest**: `data_stats.train_examples` is the actual row count of the epochs used; `tenant_id` is
recorded. A scoped run records its view's `val_examples` and `test_examples`, `examples_by_line`, and
`data_mix` (the shares configured and reached per line, documents rested, the render rotation applied);
the model card shows the same.

**On the pod** `training.train` and `training.sweep` run detached in tmux like every long job.

**Scopes by line of business** (`training/corpus_view.py`, `materialize`): a scope with `lines` filters the
one corpus by line as well as type — a row is kept only when all of its lines are in scope
(`Scope.covers_lob`: a personal + commercial package is not a personal-lines document, nor is a package of
common-model lines; a row with no line is outside the scope). The view is written under
`train/scope/{name}/` and `{val,test}/scope/{name}/`; train, val (hence checkpoint selection and
calibration) and the test count all follow. On the train rows, from the first epoch's documents and the same
for every epoch, in this order:

1. **Data mix** (`training/data_mix.py`, `data_mix:` in the scope's config — `personal_lines.yaml`): per
   line, `synthetic_fraction` and `scanned_share` of its train documents. Every real document trains;
   synthetic ones are taken one twin of every seed before a second of any (seeded); when a scanned or digital
   pool runs short, fewer synthetic documents are taken rather than the scanned share missed. A per-line
   `null` takes the scope default; a scope default of `null` leaves the share unsteered. All values are `null`
   today, so every document trains. A `scanned_share` on rows without `is_scanned` is refused.
2. **Render rotation** (`render_rotation: true`, `plan_render_rotation`): a twin delivered digital and
   scanned with one gold trains in one render per epoch and the other the next; half of a seed's twins start
   in each render. Real documents and single-render twins train every epoch; a line that rotation would take
   under the line-balance target keeps both renders.
3. **Line balance** (`line_balance: {min_documents: 100, max_repeat: 3}`, `line_repeats`): a line with fewer
   documents per epoch than the target (`min_documents`, capped at the largest line) has its rows repeated
   ⌈target ÷ documents⌉ times, at most 3, in every epoch file.

Validation and test are never sampled, rotated or repeated. None of the three applies to the unified scope,
which reads the corpus files as they are and warns when they are configured. **casualty_fleet** (`lines:
[gl]`, `training_config: unified`) is the CGL adapter, trained in its own tenant on the CGL bundles
(`--tenant`): one line, real documents only, so the three change nothing there. The architecture records that
its v1 adapter was trained and scored on golds nobody reviewed; a gold with no review record no longer
reaches a corpus or the gate (decision 2026-10-22; IMPL-05, IMPL-08), so they are reviewed before it trains
again. A scope without `lines`
(`policy`, `lossrun`) filters by type alone.

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
