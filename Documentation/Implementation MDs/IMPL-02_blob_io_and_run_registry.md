# SPEC 02 — Azure Blob I/O + Training Run Registry

> Aligned with the code of 2026-10-22.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01. This is the ONLY module that talks to the Azure SDK directly. Everything else goes through it.
>
> **Architecture refs:** `finetuning-architecture-v2.5.docx` §8b (tenant isolation per Fideon SPEC_12), §12 (versioning + training run registry), §13b (quantization thresholds recorded), §15 (metrics recorded), §18/§18a (Blob layout, raw-document access control).

## Goal

Provide a clean, typed interface for all Azure Blob reads/writes, and implement the training run registry that records **every** Foundation, per-type, per-tenant, and **sweep** training run.

## Deliverables

### 1. `artifact_registry/paths.py` (build this first)
- The **single place** the Blob layout (master §4) is expressed. Every other function builds paths through it.
- **Reserves the tenant prefix** (arch §8b): `TENANT_SCOPED` — `raw-documents/`, `processed/`, `golden-labels/`, `corpus/`, `releases/`, `exports/` and `golden-eval-set/` — are `{tenant_id}`-prefixed; `SHARED` — `base-models/`, `adapters/`, `merged-models/`, `quantized-models/`, `registry/`, `eval-reports/`, `calibration/` — are never prefixed (the module records that trained weights are tenant data under v2.1 and that moving `adapters/`, `merged-models/` and `quantized-models/` under the tenant is still a pending migration; until then a tenant deletion cannot remove trained weights by prefix, so the purge plans name each run trained on the tenant's corpora, its weights and its releases — IMPL-03 §1). `tenant_id` **defaults from `DEFAULT_TENANT_ID`** (else `default`) and is validated by `paths._tenant` (lowercase alphanumeric, `-`/`_`, at most 63 characters: it becomes a path segment). Every ingestion, OCR, import and pipeline CLI takes `--tenant`, and separate tenants are in real use: the CGL documents train under a tenant of their own, smoke runs under `smoke` / `smoke-cgl`. Each tenant has its own documents, labels, corpus versions, frozen eval set (`golden-eval-set/{tenant}/`) and releases.
- Distinguishes the adapter lineages: `adapters/foundation/v{n}/` (unified), `adapters/scope/{scope}/v{n}/` (a scoped run) and `adapters/{doc_type}/v{n}/` (a §4.2 graduated per-type adapter). A per-tenant lineage is **not built** until a broker requires one.
- **Also expresses the RunPod staging paths** (master §12a) — `/runpod-volume/staging/{adapters,merged-models,eval-reports,run_manifests}/` — deliberately mirroring the Blob layout so `package` (IMPL-13 command 2) copies rather than translates. Staging paths and Blob paths are built by separate functions that must never be interchanged: one is working storage, the other is the artifact of record.

### 2. `artifact_registry/blob_client.py`
- Thin wrapper around `azure-storage-blob`, initialized from env (`AZURE_BLOB_CONTAINER`, `AZURE_RAW_CONTAINER`, and a login). `azure_from_environment` prefers a **SAS token** — `AZURE_STORAGE_ACCOUNT_URL` + `AZURE_STORAGE_SAS_TOKEN`, short-lived and limited to the permissions it was issued with — when both are set, and falls back to `AZURE_STORAGE_CONNECTION_STRING`, which carries the account key; half a SAS login (one of the two) is refused, and the token never appears in an error. `AZURE_RAW_SAS_TOKEN` is the raw container's own token: `AzureBackend` routes each call by container (`_service_for`), so the main token can be a container SAS that cannot read raw PDFs. Without it the one token serves both containers and must be an account SAS. It is refused beside a connection string alone (half a SAS login) and when there is no separate raw container. Two backends behind `BlobBackend`: `AzureBackend` (SDK imported lazily) and `InMemoryBackend` (tests; same write-once and container rules). The Azure SDK's own logger is held at WARNING.
- Methods: `read_bytes` / `write_bytes`, `read_text` / `write_text`, `read_json` / `write_json`, `exists`, `list(prefix)`, `delete`, `upload_file`, `download_file`, `upload_dir`, `download_dir`, `holds`.
  - `upload_dir(local_dir, prefix, *, names=None, skip_identical=False)` returns the files written. `names` uploads only those paths relative to the folder (a missing one raises `BlobError`); `skip_identical` leaves any blob that already holds the file's bytes.
  - `holds(local, key)` compares the MD5 recorded with the blob (`BlobBackend.md5`) with the local file's (`file_md5`, read in 8 MB chunks). A blob that records no MD5 does not count, so it is sent again. The properties call is retried like any read.
  - `AzureBackend.write` records `Content-MD5` on every upload (Azure stores one itself only for a single-shot upload, and a multi-gigabyte shard goes up in blocks); `AzureBackend.md5` reads it back from the blob properties, `InMemoryBackend.md5` computes it, the base class returns `None` (nothing is then skipped).
  - `download_dir` matches the prefix at a `/` boundary, so `corpus/default/v1` never pulls `v10`.
- Retries with exponential backoff (`max_retries`, default 3); a rule violation (`ImmutableBlobError`, `AccessDeniedError`) is never retried; clear errors on auth/missing-container.
- **Separate-container support (arch §18a):** keys under `raw-documents/`, `intake/` and `exports/` (`paths.requires_raw_container`) are routed to the raw container automatically, and only a client built with context `ingestion` or `ocr` (`for_ingestion()`, `for_ocr()`) may touch them; any other context gets `AccessDeniedError`, including for the bare prefix.
- **Immutability guard:** writing to (or deleting) an existing `raw-documents/.../original.pdf` raises `ImmutableBlobError`. Raw documents are write-once (arch §18a).

### 3. `artifact_registry/transfer.py`
*(One module, not the `push_to_blob.py` / `pull_from_blob.py` pair originally sketched: a push and its pull must agree on the path, and two files are two places for them to drift.)*
- High-level, path-aware helpers keyed to the Blob layout. Tenant-scoped helpers accept an optional `tenant_id` that defaults from env:
  - `push_adapter(local_dir, kind, version, doc_type=None)` → foundation / per-type path; `pull_adapter(kind, version, local_dir, doc_type=None)`
  - `push_scoped_adapter(local_dir, scope, version)` → `paths.scoped_adapter_dir` (see Current implementation)
  - `push_merged_model(local_dir, version, doc_type=None, scope=None)` / `pull_merged_model(...)`, `push_quantized(local_dir, version, fmt, doc_type=None, scope=None)` / `pull_quantized(...)`
  - `pull_base_model(local_dir)` (from `base-models/`, or from HF at the pinned revision if absent, then cache to Blob)
  - `push_corpus_version(local_dir, version, tenant_id=None)`, `pull_corpus_version(version, local_dir, tenant_id=None)`
  - `push_calibration(version, doc_type, params)` / `pull_calibration(...)` (IMPL-09 store; a missing calibration raises)
  - `push_eval_report(...)` / `pull_eval_report(...)`, `push_golden_eval_set(local_dir, tenant_id=None)` / `pull_golden_eval_set(local_dir, tenant_id=None)` — one frozen set per tenant, under `golden-eval-set/{tenant}/`
- Every push refuses a missing source directory (`TransferError`) rather than creating an empty prefix that would read as a published artifact.
- No standalone pull CLI: the pipeline stages call these helpers directly (training stages its own data through `training/stage_data.py`). `pull_base_model` from the Hub is not wired; the pod reads the base from `/workspace/models` (`configs/base_model.yaml` `local_dir`).

**Reading the index is not optional.** `_index` raises `RegistryQueryError` when the index blob exists but cannot be read. It must never degrade to an empty list: an empty registry and an unreadable one are different facts, and conflating them makes a throttled or 403'd Azure call look like a first-ever run — `latest_promoted` returns `None`, the promotion gate sets `is_first_version` and has nothing to regress against, and a candidate whose accuracy collapsed is promoted. The same swallow silently cleared the arch §12 Foundation cascade block.

### 4. `registry_utils/models.py`
Pydantic model `RunManifest` capturing the arch §12 manifest **plus the v1 additions**:

- `run_id`, `run_type` (`unified` | `foundation` | `per_type_adapter` | `scoped`), `doc_type` (nullable — the §4.2 graduated adapter a run *is*), `scope` (nullable, reads as `unified`) and `doc_types` (what a scoped run covers; required for `scoped`), `tenant_id` (nullable, defaulted), `status` (`training|trained|evaluated|promoted|archived|failed`), `created_at`, `continued_from` (a Foundation run continued from a checkpoint).
  - `unified` / `foundation` runs may name neither a `doc_type` nor a narrower scope; a `scoped` run must name its scope and `doc_types` and no `doc_type`.
  - **`training` exists because the manifest is written *before* ms-swift is launched** — the run_id has to be reserved and the configuration captured even for a run that dies. Without it the pre-launch manifest defaulted to `trained`, so a pod that OOM'd at step 40 left a registry entry asserting a trained adapter over a staging path holding nothing, and `package` published Blob locations for weights that were never written. `training.train.launch_and_record` flips it to `trained` when ms-swift returns and to `failed` when it does not; **`stage_push` publishes only `trained`, `evaluated` or `promoted` runs.**
- **`is_sweep_run`** (bool) + `sweep_id` / `sweep_phase` — reserved fields so sweep runs are first-class registry entries when sweeps are run (arch §11a; deferred past the IMPL-15 pilot).
- `dependencies`:
  - `base_model` (`qwen3-vl-8b-instruct@<hf_revision_pin>`)
  - `foundation_version` (nullable — which Foundation this adapter sits on)
  - `corpus_version`
  - `code_git_commit`
  - **`mineru_version`** and `ocr_device` (arch §8a) — read from the corpus manifest the run trains on (`training/train.py`), so the run records the MinerU that read its documents (the corpus build refuses documents read by more than one version, IMPL-03)
  - **`schema_version`** and **`prompt_template_version`** (arch §7)
- `training_config`: **`technique` (`LoRA` | `QLoRA`) and `base_quantization` (`bf16_frozen_base`, or `<quant_type>_<double|single>_quant_<dtype>_compute` under 4-bit) — both REQUIRED, never defaulted**. They previously carried QLoRA/NF4 defaults on the model, so a bf16 run that did not pass them recorded a technique it never used; base precision is exactly the kind of change that causes a regression, and the manifest is what makes one attributable. Then lora_rank/alpha/dropout, bias, lr, lr_scheduler, warmup_ratio, epochs, **optimizer (`adamw_torch`, also required)**, betas, eps, weight_decay, max_grad_norm, per_device_batch, grad_accum, effective_batch_size, gradient_checkpointing, mixed_precision, target_modules, **`vit_trainable`** with `vit_method` (`frozen` | `lora`; when trainable it must be **LoRA-on-ViT, never full fine-tune** — arch §3), resolution_cap_px, `pixel_budget` (the image resize budget the trainer was actually given), max_seq_len, **`seed`**.
- `data_stats`: train/val/test counts, `modality_mix` and `modality_mix_target`, **`lob_coverage`** (per-LoB-value share, arch §0b), `alias_coverage`, `confusable_example_count`, `tenant_ids` contributing, `split_policy` (carriers held out per line, single-carrier lines, lines naming fewer than the 10 carriers a hold-out needs (`lines_below_carrier_minimum`), families moved into test, twins dropped by the per-seed cap), `data_mix` (the scope's configured `synthetic_fraction` / `scanned_share` and what each line's train documents came to), `examples_by_line` (split → line → documents, real, synthetic, rows; scoped runs), `deidentified: true|false` and `image_redaction` (recorded; `false` and `unresolved` under Option A, no de-identification, which the architecture records as the current state — IMPL-00 §8, IMPL-05 §1).
- `eval_metrics` (all against the frozen golden eval set, arch §15):
  `field_exact_match`, `field_normalized_match`, `field_f1_list_fields`, `list_field_recall`, `schema_validity_rate`, `ece_confidence`, `ocr_arbitration_accuracy`, `image_only_accuracy`, `scanned_accuracy`, `doc_type_classifier_accuracy`, **`lob_detection_accuracy`** (overall, and `lob_accuracy_by_value`), `alias_accuracy` (reported), **`confusable_misattribution_rate`** (gating), `table_f1` and `lossrun_totals_reconciliation_rate` (Loss Runs), `held_out_carrier_match` (reported, not gated), `latency_ms_per_doc`.
- `artifacts`: **`status`** (`staged` | `published`), **`staging_path`** (set while `staged`), `adapter_weights`, `merged_model`, `quantized_model`, `quantized_formats[]`, `eval_report`, `calibration_params`.
  - `finetune` (IMPL-13 command 1) writes the manifest with `status: "staged"` and weights still on the RunPod volume; `package` (command 2) flips it to `"published"` and fills the Blob paths. A manifest is written **either way**, so a reclaimed staging volume never means an untracked training run.
- `promotion`: `gated_against`, `beat_previous_on_all_gates`, `failed_gates`, `promoted_by`, `promoted_at`, `tier` (`production` | `interim`, `None` when not gated or blocked).

### 5. `registry_utils/write_run_manifest.py`
- `write_manifest(manifest: RunManifest)` → writes `registry/foundation/{run_id}/run_manifest.json` (unified and foundation runs share the slot), `registry/adapters/{doc_type}/{run_id}/…` or `registry/scope/{scope}/{run_id}/…` (`paths.run_manifest`) to Blob, rewrites the run's **model card** beside it (`model_card.md`, `registry_utils/model_card.py`, rendered from the manifest alone; a card that fails to write is logged, never fails the run), and updates `registry/registry_index.json` (a flat table: run_id, run_type, doc_type, scope, doc_types, status, artifact_status, corpus/foundation version, commit, created_at, is_sweep_run, key metrics).
- `capture_git_commit()` to fill `code_git_commit`, `is_dirty_worktree()` so callers can warn; `mineru_version` comes from the corpus manifest the training stage reads (IMPL-05), not from a capture here. `mark_promoted` and `mark_published` (below and IMPL-13).
- `data_stats.deidentified` is **recorded, not enforced**: under Option A (no de-identification, IMPL-00 §8) every run records `false`. Should a de-identification option (B or C) be chosen later, the refusal belongs here — it is the right enforcement point.
- `log_to_tracker` mirrors params and numeric metrics to MLflow when `MLFLOW_TRACKING_URI` is set (no-op otherwise; W&B is not wired) — the tracker gives the queryable UI, the Blob manifest is the durable record that travels with the artifacts (arch §12).

### 6. `registry_utils/query_registry.py`
CLI + functions:
- `get(run_id)` → RunManifest.
- `list_runs(run_type=None, doc_type=None, status=None, include_sweeps=False)`.
- `adapters_depending_on(foundation_version)` → adapter run_ids — this makes the arch §12 dependency-upgrade rule **a query, not a manual audit**: when Foundation moves to v3.0, this is the exact list needing re-validation.
- `latest_promoted(kind, doc_type)` → the version currently serving.
- `resolve_model_version(tag)` → given a user tag, return the concrete artifact paths. Used by IMPL-07, IMPL-11, IMPL-12, IMPL-13.
  - **`"base"` is a valid tag** and resolves to the pinned HF base model with **no adapter** — the zero-shot path used by `extract --model base` (IMPL-13), the pilot baseline (IMPL-15), and day-zero pre-annotation (IMPL-04).
  - `"v2"` resolves the merged model, plus a graduated per-type adapter (§4.2) when the routed type has one. Never a stack: one LoRA per request is what vLLM serves.
  - Resolves from Blob for `published` versions and from the staging volume for `staged` ones, so `extract` works on a model that has not been packaged yet.
- `diff_manifests(run_a, run_b)` → field-by-field delta. This is the arch §12 regression-debugging workflow: when a doc type regresses, diff the new manifest against the last-good one and see immediately whether the corpus, foundation version, MinerU version, or a hyperparameter moved.

## Constraints
- Only this package imports `azure-storage-blob`.
- All Blob paths built via `paths.py` — the layout lives in one place and the tenant prefix is not optional.
- `raw-documents/` is write-once; it, `intake/` and `exports/` sit in the raw container and are reachable only from ingestion/OCR clients.
- No PII in logs.
- Every training/eval/quantize job writes a manifest — **no silent runs, sweeps included**.

## Promotion and the index

- **`mark_promoted` requires the affirmative gate result**, not merely the
  absence of a failure. `promotion.beat_previous_on_all_gates` defaults to
  `None` and only the gate ever sets it, so rejecting just `False` let a run that
  was **never evaluated** promote cleanly — and the index could not tell it from
  one that actually passed.
- **A Foundation run owns its adapter and nothing else.** In a per-type build the
  merged and quantized models are produced per doc type, and the `doc_type=None`
  "unified" paths exist only under `--foundation-only`; publishing them anyway
  pointed the manifest at artifacts that were never built.
- **`registry_index.json` is a shared read-modify-write.** Concurrent runs each
  read N rows and write N+1, so the later write drops the earlier run's row — and
  the index is exactly what `adapters_depending_on` and `latest_promoted` read,
  so a lost row makes a run invisible to the queries the registry exists for. The
  write re-reads and re-checks; a store with conditional writes is the real fix
  before jobs run in parallel.
- **Version tags match exactly.** A substring match makes `v1` match `v10`, so
  `resolve_model_version("v1")` can read v10's artifact status and return a
  staging path for a model published in Blob.

## Acceptance checklist
- [ ] `blob_client` round-trips a file and a directory against a real/emulated container.
- [ ] Overwriting an existing `original.pdf` raises (write-once guard).
- [ ] `RunManifest` validates the arch §12 example (extended with the v1 fields) and rejects malformed input.
- [ ] `write_run_manifest` creates the manifest and updates `registry_index.json`.
- [ ] `adapters_depending_on("foundation-v2.0")` returns the right list from seeded manifests.
- [ ] `resolve_model_version("v2")` returns concrete adapter/merged paths; `resolve_model_version("base")` returns the pinned base model with no adapter.
- [ ] A `staged` manifest resolves to staging-volume paths; a `published` one resolves to Blob paths.
- [ ] `diff_manifests` surfaces a changed `corpus_version` / `mineru_version` / hyperparameter.
- [ ] Unit tests mock the Blob client (no live Azure needed for CI).
- [ ] A published scoped adapter holds `adapter_config.json` and `adapter_model.safetensors` only; a folder missing either is refused.
- [ ] A re-push of the same adapter or merged model writes nothing; a blob that records no MD5 is sent again; `upload_dir(names=...)` refuses a named file that is not there.
- [ ] Each tenant pushes and pulls only its own golden eval set (`golden-eval-set/{tenant}/`).
- [ ] With `AZURE_STORAGE_ACCOUNT_URL` and `AZURE_STORAGE_SAS_TOKEN` both set the client logs in with the SAS token; with one of them alone it refuses, naming the missing one; with neither it uses `AZURE_STORAGE_CONNECTION_STRING`. With `AZURE_RAW_SAS_TOKEN` set too, calls to the raw container use it and calls to the main container do not; it is refused without the URL and main token, or with no separate raw container, and no message holds a token.
- [ ] A `BlobClient` built from the environment sends read, list and exists calls on `raw-documents/`, `intake/` and `exports/` keys to the raw container with the raw token, and every other key to the main container with the main token.

---

## Current implementation (2026-10-21)

**Paths beyond the original layout** (all in `artifact_registry/paths.py`):

| Path | Holds |
|---|---|
| `adapters/scope/{scope}/v{n}/`, `merged-models/scope/{scope}/v{n}/`, `quantized-models/scope/{scope}/v{n}/vllm/{format}/` | a scoped run's own adapter and models (the unified scope keeps the pre-scope keys: `adapters/foundation/`, `merged-models/unified/`) |
| `eval-reports/v{n}/scope/{scope}/{summary,gate_decision,checkpoint_selection}.json` | a scope's report, verdict and chosen checkpoint (`paths.eval_report`, `gate_decision`, `checkpoint_selection`; not tenant-prefixed) |
| `registry/{foundation\|adapters/{doc_type}\|scope/{scope}}/{run_id}/model_card.md` | the run's model card, beside its `run_manifest.json` (`paths.model_card`) |
| `releases/{tenant}/{release_id}/bundle.json`, `.../calibration/{format}/calibrators.json`, `.../gate/{format}/gate_decision.json`, `releases/{tenant}/release_index.json` | the release bundle — the unit of promotion |
| `golden-eval-set/{tenant}/{source_id}/{golden.json,metadata.json,page_N.png,page_N.md}`, `golden-eval-set/{tenant}/manifest.json` | one tenant's frozen eval set (IMPL-08). A set left at the old root (`paths.legacy_golden_eval_manifest()`, `golden-eval-set/manifest.json`) is refused by `evaluation.freeze_eval_set` with the move to make, never read |
| `intake/{batch}/{document}/` (raw container) | a delivered batch of bundle folders, staged with azcopy before import (`paths.intake_batch_dir`, IMPL-03) |
| `exports/{tenant}/{kind}/{name}` (raw container) | result sets made for download to a laptop, e.g. a model comparison (`paths.export_dir`) |
| staging: `{mount}/staging/train-data/{scope}/v{n}/`, `{mount}/staging/train-images/{tenant}/{corpus}/` | a run's staged JSONL; one page-image cache per corpus version, shared by every run on it |

The staging root is `$RUNPOD_VOLUME_MOUNT/staging` (default `/runpod-volume`; set `/workspace` when the pod
mounts its volume there).

**`transfer.py`** has scope-aware helpers — `push_scoped_adapter(local, scope, version)`,
`push_merged_model(..., scope=)`, `push_quantized(..., scope=)` — which `package` uses to upload the real
adapter, merged model and quantized formats. Each refuses a missing source directory instead of creating
an empty prefix that would read as a published model.

- **A published adapter is its weights and config only.** `push_scoped_adapter` uploads
  `ADAPTER_FILES` = (`adapter_config.json`, `adapter_model.safetensors`) — never a checkpoint's optimizer and
  scheduler state (~1.5 GB of a ~2 GB rank-64 checkpoint) — and refuses a folder missing either file. Both
  `package` and the training stage's `--push-adapters` publish through it.
- **A re-push sends nothing it does not have to.** `push_scoped_adapter` and `push_merged_model` (~18 GB of
  bf16 shards) call `upload_dir(..., skip_identical=True)`: a file whose MD5 matches the one recorded with the
  blob is left in place. Blobs written before MD5s were recorded are sent once more. `push_quantized`,
  `push_adapter`, `push_corpus_version` and `push_golden_eval_set` upload everything.

**Run manifest** records `tenant_id` and the manifest's `quantized_model` points at the first *quantized*
format — bf16 is the merged model, never an export. `quantized_formats` lists every exported format, and the
purge plans name each one's directory (IMPL-03).

**Release bundle** (`registry_utils/models.py::ReleaseBundle`) also records `tier` (`production` at field
match ≥ 0.92, `interim` when it passed every floor below that, or when its scope was scored on fewer than
150 frozen documents of a type it covers; `evaluation.gating.release_tier`), and — recorded, not gated, since no latency SLA is agreed
(decision 2026-10-19) — `latency_p95_ms_by_adapter`, `peak_gpu_memory_mb`, `measured_with`, and `routing` (where
every line of business routes once the release is promoted), plus `scope`, `doc_types` and `lines` (empty
means every type / every line).

**Blob `upload_dir`/`download_dir`** are what staging, localisation and package use; images are fetched
into the per-corpus cache atomically (unique `.part` names, renamed into place).

## Changes since 2026-10-05

- Tenant prefix: `golden-eval-set/` is now tenant-scoped (one frozen set per tenant, legacy root set refused); listed the full `TENANT_SCOPED` / `SHARED` sets and `paths._tenant` validation; replaced "single-tenant" with the tenants in real use (CGL, smoke).
- `BlobClient`: documented `holds()`, `upload_dir(names=, skip_identical=)`, `file_md5`, `Content-MD5` recorded on every Azure write, `BlobBackend.md5()`, and the raw-container routing of `intake/` and `exports/`.
- `transfer.py`: `push_scoped_adapter` publishes only `ADAPTER_FILES` and skips identical files; `push_merged_model` skips shards Blob already holds; corrected helper signatures; golden eval set push/pull take `tenant_id`.
- `RunManifest`: added `unified`/`scoped` run types, `scope`, `doc_types`, `ocr_device`, `vit_method`, `pixel_budget`, the new `data_stats` fields (`split_policy`, `data_mix`, `modality_mix_target`, `examples_by_line`), new metrics (`table_f1`, `lossrun_totals_reconciliation_rate`, `held_out_carrier_match`) and `promotion.tier`.
- `write_manifest`: scope manifest path, model card written beside every manifest; removed the non-existent `capture_mineru_version()` (the MinerU pin comes from the corpus manifest); tracker is MLflow only.
- Current implementation: paths table gained model cards, per-tenant eval set, `intake/`, `exports/`; release bundle tier, latency, GPU memory and routing fields.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx`.
- §1: until trained weights are tenant-prefixed, the purge plans name the runs trained on a tenant's documents (IMPL-03).
- §2: the SAS login (`AZURE_STORAGE_ACCOUNT_URL` + `AZURE_STORAGE_SAS_TOKEN`) is preferred over the connection string; half a SAS login is refused. New acceptance item.
- §4/§5: de-identification is Option A (none), recorded, not "blocked"; `split_policy` records `lines_below_carrier_minimum`.
- Current implementation: the tier is also `interim` for a release gated on a set frozen with `--allow-small`; latency is recorded, not gated (no SLA).

## Changes of 2026-10-22 (code review)

- §2: the raw container's own SAS token (`AZURE_RAW_SAS_TOKEN`), routed per call; two new refusals. New acceptance item.
- Release bundle tier: `interim` follows what the release's scope was scored on (fewer than 150 frozen documents of a type it covers), not the whole set's `--allow-small` (IMPL-08).
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.
- The release bundle records `window_plan_version`, the window plan of the corpus behind a policy release; serving refuses a mismatch (IMPL-11).
- Acceptance: a `BlobClient` routes raw-layer calls to the raw login and the rest to the main one; the purge plans name every quantized format's directory.
