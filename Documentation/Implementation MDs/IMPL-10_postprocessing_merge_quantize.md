# SPEC 10 — Postprocessing (Merge + Serving-Format Quantization)

> Aligned with the code of 2026-10-21.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-02, IMPL-06.
>
> **Architecture refs:** `finetuning-architecture-v2.4.docx` §13 steps 7–8, §13a (**vLLM-native serving formats**), §13b (quantization quality thresholds, absolute percentage points), §18 (artifact paths).

## Goal

Fold the trained adapter into the base model, and produce the quantized formats **the serving endpoint can actually load**. Quality thresholds are specified here and enforced before any quantized format is served.

## What changed from v1, and why

v1 produced `fp16` + `q5_k_m` GGUF files every cycle. GGUF is **llama.cpp's** format and the endpoint runs **vLLM**, so the pipeline spent conversion and eval compute on an artifact nothing could deploy — and there was no small format the server *could* open, leaving bf16 or nothing.

v1's own module docstring had the right of it: *"GGUF is the portable/edge path and vLLM on the merged model is the primary one."* It never followed through, and left `DEFAULT_FORMATS = ("fp16", "q5_k_m")` making a GGUF the per-cycle deliverable.

**GPU only** (decision 2026-10-19). The merge, every quantized export and the vLLM engine that serves and scores them run on CUDA and refuse to start without it (`common/gpu.py` `require_cuda`); nothing falls back to the CPU. Every deployment target is a vLLM format on a GPU. GGUF stays an **optional edge export** (§2a), never a deployment target.

## 1. `training/merge.py`

- Loads the base in **bf16** and folds in **one** adapter via PEFT `merge_and_unload()` → one standalone bf16 model.
- **One adapter, one merge.** v1 applied a Foundation LoRA and then a per-type LoRA on top, producing a model per document type — and the order mattered, because reversing it produced a different model from the one that was evaluated. That whole ordering problem is gone with the topology that created it (arch §4.1).
- A graduated per-type adapter (§4.2) is **never merged**. It is applied at serving time on top of these merged weights, one LoRA per request — the only shape vLLM can serve.
- Folds in the **checkpoint the §11.2 selector chose**, recorded on the plan (`MergePlan.selected_checkpoint`). `orchestration.pipeline_dag.stage_merge` reads it through `selected_checkpoint(ctx)`: this process's `checkpoint_eval` result, else the choice recorded in Blob at `eval-reports/v{n}/[scope/{scope}/]checkpoint_selection.json`. The merged weights do not say which checkpoint they came from, and once the staging volume is reclaimed nothing else does either.
- `dtype` defaults to **bf16, not fp16**: the adapter trained in bf16 against a bf16 base, and FP8 is quantized from this artifact. Merging to fp16 would insert a precision change between training and every serving format for no reason.
- Writes to the **staging volume**, per scope (`plan_merge(scope=)`): `staging/merged-models/{unified|scope/{scope}}/v{n}/`. The merged model is ~16 GB and quantization also runs on RunPod; pushing it to Azure and pulling it back is a 32 GB round trip for nothing (master §12a).
- **Runs on the GPU** into the **pinned local base** (see *Current implementation*); written to `<output>.partial` and renamed into place only once `config.json` and the safetensors exist.

## 2. `postprocessing/quantize.py` — serving formats

| Format | Tool | Hardware | Role |
|---|---|---|---|
| **bf16 merged** | PEFT merge | Any CUDA GPU | Reference, and cycle 1's serving format |
| **FP8 (W8A8)** | llm-compressor | Native FP8 on Ada/Hopper (L4, L40S, H100, H200); weight-only on A100 | Default serving format from cycle 2 |
| **AWQ INT4 (W4A16)** | llm-compressor | Modern NVIDIA | VRAM-constrained serving only, if it meets threshold |

```bash
# fp8 is refused until it is verified (below); --formats bf16 is the default
python -m orchestration.run package --version v2 --release-id release-2026.11.1 --formats bf16 fp8
```

**bf16 is never re-exported.** It *is* the merged model; a copy would be a second 16 GB artifact identical to the first.

**With `--formats bf16` — the default (`defaults.formats` in `orchestration/config/pipeline.yaml`) — quantize is a no-op.** `_is_quantized` reports the stage complete as soon as merge is, and `quantize()` returns the merged model as the bf16 entry without exporting anything.

**FP8 is refused until verified.** `plan_quantization` defaults to bf16 alone and `quantize()` raises on `fp8` unless `fp8_verified=True` (`StageContext.fp8_verified`), which is to be set once Phase 0 spike item 9 has verified the export; no CLI flag sets it. An unverified serving format is a deployment that fails at load — or worse, one that loads and reads badly.

**The vision path is never quantized.** `NEVER_QUANTIZED = visual.*, mergers, aligners, lm_head`, passed to llm-compressor as its ignore list (arch §13a). Compressing it produces a model that loads cleanly and reads pages badly — very hard to notice, because it still emits well-formed JSON.

**Every serving format gets its own calibrator fit and its own gate run.** Quantization degrades exactly what was fine-tuned in, so an FP8 release inherits nothing from bf16's result. Today the evaluation gate scores bf16 only (its decision is `release_gate_decision(release, "bf16", tenant)`); a quantized format with no gate decision of its own leaves the release bundle at status `gated`, never `promoted`.

## 2a. GGUF — the edge export, on request only

`plan_gguf_export` / `export_gguf` produce a GGUF for an **offline or edge** deployment. Not per-cycle, not on the serving path, not a deployment target (the deployment is GPU only), and **validated separately in llama.cpp** against the §13b AWQ INT4 column — the serving gate does not cover it, because the serving endpoint never loads one. No command calls it, and past its guards `export_gguf` raises `NotImplementedError`: the llama.cpp conversion is not wired.

The v1 mmproj guard is carried over **unchanged**: Qwen3-VL is multimodal, so a GGUF needs the quantized LLM **plus a separate `mmproj`** file for the vision encoder and projector. Exporting without a verified one produces a model that loads and cannot see, which fails silently on every image-only document.

Serving and edge artifacts are stored under **different runtime prefixes** — `quantized-models/{scope}/v{n}/vllm/{format}/` and `.../gguf/{format}/`. They are loaded by different programs, and one prefix invites deploying the wrong one.

## 3. `postprocessing/quant_thresholds.py`

**Absolute percentage points against bf16, per field class** (arch §13b):

| Metric | FP8 max drop | AWQ INT4 max drop |
|---|---|---|
| Exact match — identifiers, money, dates | 0.5 pp | 1.0 pp |
| Match — names, addresses | 1.0 pp | 2.0 pp |
| List-field row recall | 0.5 pp | 1.0 pp |
| Confusable misattribution (increase) | 0.5 pp | 1.0 pp |
| ECE (increase) | 0.01 | 0.02 |
| Schema validity (constrained) | 100% | 100% |

`field_normalized_match` is judged on the **names, addresses** row and `field_exact_match` on the **identifiers, money, dates** row, each against its own allowance.

Three things about this table are deliberate:

- **Absolute, not relative.** v1 allowed a 2% *relative* field-F1 drop, which is 1.9 pp at F1 0.95 and 1.4 pp at 0.70 — so the rule got **stricter as the model got worse**, which is backwards. The same number should mean the same thing whatever the baseline.
- **Per field class.** A wrong policy number is a wrong extraction; a slightly-off entity name is usually still matchable. One allowance would let the forgiving class lend its slack to the unforgiving one.
- **Schema validity is a floor, not a margin.** Structured decoding guarantees it (§13), so anything below 100% means the guarantee is not working — an error, not a degraded result.

The reference is **bf16**. v1 measured every format against fp16, a GGUF format, so the reference itself was an artifact the endpoint could not load.

A format exceeding **any** threshold is not served. Asking the serving gate about a GGUF format raises and says where it *is* validated, rather than reading as "unsupported".

## 4. `postprocessing/validate_quant.py`

- Judges each produced format's metrics — from the extraction routine (IMPL-12) against the **frozen golden eval set** — against the table above. It scores nothing itself: the metrics are supplied (`StageContext.quant_metrics`), and with none supplied the quantize stage logs a warning instead of judging. No CLI path supplies them yet; a bf16-only cycle has nothing to compare, because bf16 *is* the reference.
- Also checks **row recall** and **confusable misattribution** on their own margins: a quantized model that keeps its field accuracy while dropping claim rows has lost exactly what a Loss Run is for, and an aggregate field score would not show it.
- Writes `quant_threshold_results` into the quantize stage result (`ValidationReport.manifest_entry()`) and **refuses promotion of an over-threshold format** (`assert_servable`) — no override flag.

## 5. Where the gates sit in `package`

`package` runs stages 8–11 in order (`orchestration.pipeline_dag.STAGES`): **quantize** (the §4 threshold gate at its end, when per-format metrics are supplied) → **calibrate** (IMPL-09) → **evaluation_gate** → **package** (the push).

- **The promotion gate** (IMPL-08) scores the merged bf16 model on the frozen golden eval set through serving, with the release's bf16 calibrators. It holds every floor; judges non-inferiority by a **paired bootstrap** over the per-document scores of the candidate's and the promoted release's eval reports (`paired_document_scores`; a point comparison, with a logged warning, against a report written before per-document scores were kept); and blocks a release that improves no primary metric unless `--fixes-defect "<ticket>"` names the defect it fixes (on `package` and `all`; recorded in the decision). A passing release is **production** at `field_normalized_match` ≥ 0.92 and **interim** below it — and **interim whatever its score** when the frozen set was accepted small (`small_set_accepted`), with a warning in the decision.
- A blocked gate stops the release: the merged model stays on the staging volume, nothing is published, and the command exits non-zero with the per-metric verdicts. The push re-reads the recorded decision (`assert_gate_passed`) and refuses a version that did not pass, so a standalone `package` cannot publish it either. The release bundle carries the bf16 decision's tier.
- **Latency is recorded, not gated** (no latency SLA; decision 2026-10-19): `record_release_measurements` copies P95 latency per adapter and peak GPU memory from the eval report's `release_measurements` onto the bundle, and a missing measurement is a warning, never a block.

## Constraints

- Merged models never overwrite adapters (separate Blob paths).
- Every **served** quantized format passes threshold validation first — no exceptions, no override.
- The bf16 reference is always produced: every §13b threshold is a margin against it, so a run producing only a quantized format has nothing to measure its drop from.
- Record all produced formats in the manifest.
- Never emit a GGUF whose multimodal projector support is unverified.

## Acceptance checklist

- [ ] `merge` produces one loadable bf16 model from one adapter, folding in the selected checkpoint.
- [ ] `plan_quantization` defaults to **bf16 alone**, and refuses a GGUF format with an error naming the runtime mismatch.
- [ ] `quantize` refuses FP8 until `fp8_verified`, and never re-exports bf16.
- [ ] With `--formats bf16`, the quantize stage is complete once merge is and exports nothing.
- [ ] A serving plan without the bf16 reference is refused outright.
- [ ] `NEVER_QUANTIZED` covers the vision tower, every merger and `lm_head`.
- [ ] `export_gguf` refuses an unverified mmproj, and is reachable only on request.
- [ ] Serving and GGUF artifacts land under different runtime prefixes.
- [ ] `quant_thresholds` returns pass/fail correctly **at each format's exact boundary** — a format landing precisely on its allowance passes.
- [ ] Margins behave identically at F1 0.95 and 0.70, which is what "absolute" means.
- [ ] `validate_quant` refuses an over-threshold format **with no override path**, and refuses validation without the bf16 reference.
- [ ] A format that was not measured does not pass.
- [ ] **`assert_servable` refuses a serving format that produced no verdict at all.** `validate_quant` only returns a result for formats it was given metrics for, so a format nobody scored appears in neither `servable_formats` nor `blocked_formats`. Checking `blocked_formats` alone let it sail through to push with zero measurements — the precise inverse of the rule this module exists to enforce.
- [ ] The threshold gate sits inside `package`, between quantize and calibrate.
- [ ] `package` publishes an adapter as its `adapter_config.json` and `adapter_model.safetensors` only, never a checkpoint's resume state.
- [ ] The merge and every quantized export refuse to run without CUDA; nothing falls back to the CPU.
- [ ] The promotion gate runs inside `package`, after calibrate and before the push; the push refuses a version whose recorded gate decision did not pass.
- [ ] The release bundle's tier is the bf16 gate decision's: `interim` below 0.92, and `interim` whatever the score when the frozen set was accepted small.
- [ ] P95 latency and peak GPU memory are recorded on the release bundle and never block a release.

---

## Current implementation (2026-10-21)

**Merge** (`training/merge.py`, run by `orchestration.pipeline_dag.stage_merge`), implemented:
- refuses an adapter directory without `adapter_config.json` (a run's output root is not an adapter), and
  a base revision still `PIN_ME` when the base would come from the Hub;
- loads the base with `AutoModelForImageTextToText` in the merge dtype on **`device_map="cuda"`** — GPU
  only (`require_cuda`); every engine is released before merge runs — from the pod's local copy
  (`model.local_dir`) when present, applies the adapter with PEFT, `merge_and_unload()`, saves safetensors
  (5 GB shards) **and the processor** (vLLM reads the chat template and image-processor config from the
  model directory), then renames `.partial` into place;
- merges the checkpoint `checkpoint_eval` selected (`selected_checkpoint`), read from Blob on a resumed run;
  with nothing selected a real run refuses with a `PipelineError` rather than merging the output root. The
  same reader decides which adapter `package` publishes, so the adapter published is the one merged;
- `finetune` merges in bf16; `all` takes `--dtype` (default `bf16`, `defaults.dtype`).

**Quantization plan is scope-aware** (`plan_quantization(..., scope=)`): a scoped run quantizes its own
merged model into its own paths. With `--formats bf16` the stage does nothing. The llm-compressor export
(`_export`) refuses without CUDA and is not wired yet — it raises `NotImplementedError` naming the recipe and
the ignore list — and needs its **own environment** (`requirements-quantize.txt`; PyPI name `llmcompressor`):
it requires `datasets>=4` where ms-swift 3 needs `<4`, and a transformers range vLLM 0.11 excludes.

**Package uploads real weights** (`orchestration.pipeline_dag._push_weights`, through
`artifact_registry/transfer.py`), after the release bundle and its routing check are built (IMPL-13 §4):
- the selected adapter, by `push_scoped_adapter`, to `adapters/foundation/v{n}/` (unified) or
  `adapters/scope/{scope}/v{n}/` — **only** `adapter_config.json` and `adapter_model.safetensors`
  (`ADAPTER_FILES`), never the checkpoint's optimizer and scheduler state (~1.5 GB of a rank-64 checkpoint).
  `finetune --push-adapters` publishes the same two files;
- the merged model to `merged-models/{unified|scope/{scope}}/v{n}/`;
- every non-bf16 format to `quantized-models/{unified|scope/{scope}}/v{n}/vllm/{format}/`.

The adapter and merged-model pushes skip every file whose bytes Blob already holds, compared by the MD5
recorded with each upload (`BlobClient.upload_dir(..., skip_identical=True)`), so re-packaging a release
sends nothing; a blob that records no MD5 is sent again. A dry run still records where each would go. The run
manifest's `quantized_model` points at the first quantized format — never bf16, which is not an export.

## Changes since 2026-10-05

- Added the alignment line and dated *Current implementation* 2026-10-13.
- §1: the merge reads the selection through `selected_checkpoint` (process result, else `checkpoint_selection.json` in Blob); the staging output path is per scope.
- §2: `--formats bf16` (the default) makes quantize a no-op; `fp8_verified` has no CLI setter; the FP8 example notes it is refused until verified.
- §4: `validate_quant` judges supplied metrics only; no CLI supplies them yet, so the stage warns.
- *Current implementation*: the published adapter is `adapter_config.json` + `adapter_model.safetensors` only (also for `--push-adapters`); adapter and merged-model pushes skip identical files by MD5; the bundle and routing check come before any push; the llm-compressor export is not wired.
- Acceptance checklist: added the bf16 no-op and the two-file adapter publish.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference is `finetuning-architecture-v2.4.docx`.
- What changed from v1: GPU only (decision 2026-10-19); GGUF an optional edge export, never a deployment target.
- §2: the bf16 reference runs on any CUDA GPU, not any hardware.
- §2a: GGUF is not a deployment target; no command calls `export_gguf`, and the llama.cpp conversion is not wired.
- §5 (new): the stage order in `package`; the promotion gate's paired bootstrap, `--fixes-defect` and the interim cap for a small eval set; the push re-checks the decision; the bundle's tier; latency recorded, not gated (no SLA).
- Acceptance checklist: GPU only; the gate's place and the push's check; the bundle's tier and the small-set cap; latency never blocks.
