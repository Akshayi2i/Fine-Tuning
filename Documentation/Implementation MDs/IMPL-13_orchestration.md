# SPEC 13 — Orchestration (Command Surface + RunPod Controller + Pipeline DAG)

> Aligned with the code of 2026-10-22.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–12).
>
> **Architecture refs:** `finetuning-architecture-v2.5.docx` §0e (decisions of 2026-10-19), §13 (**end-to-end pipeline, 13 stages, idempotency, the gate**), §13c (**operator command surface**), §14 (**RunPod: ephemeral training pods vs. persistent serving endpoint**), §12 (every cycle recorded), §17 (extraction routine), §11a (sweep orchestration — deferred).

## Goal

Expose the whole lifecycle through **three operator commands plus one umbrella command**, so a cycle is run by a person who knows the version tag and nothing else about paths, pods, or stage wiring.

## 1. The command surface

Single entrypoint, `orchestration/run.py`, with four lifecycle subcommands (plus `deploy-endpoint`, `rollback-endpoint` and `freeze-eval-set`; see *Current implementation*):

| # | Command | Covers arch §13 stages | Ends with |
|---|---|---|---|
| **1** | `finetune` | 1 → 7 (ingest, OCR, labeling, dataset build, train, checkpoint eval, merge) | Adapters + merged model on the **RunPod staging volume** |
| **2** | `package` | 8 → 11 (quantize, calibrate, **gate**, bundle) | A promoted **release bundle** in Azure Blob |
| **3** | `extract` | §17 extraction routine | Extracted JSON + confidence + metrics, locally |
| **—** | `all` | `finetune` then `package` | Same as command 2 |

**`all` never includes `extract`.** Extraction is a separate concern — you run it against a chosen model version whenever you want, including against models trained weeks earlier and against the untuned base. Folding it into the build would conflate "produce a model" with "use a model."

```bash
# 1 — ingest through merge. Artifacts staged on RunPod, not yet in Blob.
python -m orchestration.run finetune \
       --input ./intake \
       --scope personal_lines \
       --corpus-version v2 \
       --out-version v2

# 2 — quantize, calibrate, gate, then publish the release bundle.
python -m orchestration.run package \
       --version v2 \
       --scope personal_lines \
       --release-id release-2026.11.1 \
       --formats bf16 \
       [--fixes-defect "<ticket>"]          # only when the release exists to fix a defect

# 3 — extraction on imported, OCR'd documents, model chosen by the operator.
python -m orchestration.run extract --model base --tenant T --split test --corpus v2
python -m orchestration.run extract --model v2   --tenant T --source-ids policy_0001 policy_0002
python -m orchestration.run extract --model base --adapter <staged checkpoint-N> --tenant T --split val --corpus v2

# all — commands 1 and 2 back to back. Extraction excluded by design.
python -m orchestration.run all \
       --input ./intake --scope personal_lines --out-version v2 --release-id release-2026.11.1 --formats bf16
```

Each subcommand is a thin argument-parsing shell over `pipeline_dag.py` stage functions (§4). No orchestration logic lives in the CLI layer.

## 2. `finetune` — command 1 (stages 1 → 7)

**`finetune` produces artifacts; `package` judges and publishes them.** The gate moved into `package` under arch v2.1 §13, because it scores the merged model in each serving format — and neither the merge nor the formats exist until `finetune` has finished.

Runs, in order: **ingest → OCR → labeling → dataset build → train (ONE adapter per scope) → select checkpoint → merge.**

**The human-labeling precondition.** Stage 3 (labeling) is human work and cannot be inside an automated command. `finetune` therefore:

1. Ingests and OCRs **everything** in `--input` (OCR in practice runs first, in the OCR environment; see *Current implementation*).
2. Builds the corpus from **only the source_ids that have a validated `golden.json`** (IMPL-04) — and refuses the run, naming them, if any gold has no review record (`load_labeled_documents`; no real `reviewer_id` and `review_date` in its metadata, a twin's inherited from its seed; decision 2026-10-22).
3. **Reports the unlabeled backlog** — a count and a `source_id` list written to the run report — and continues with what is labeled.
4. Aborts before training only if the labeled set is empty, or below `--min-labels-per-type` (default 25, matching the day-zero rule in IMPL-04).

This is the honest shape: you ingest 500 documents, 200 are labeled, you train on 200, and the command tells you 300 are waiting for a reviewer.

**The corpus is refused before anything trains on too little of it.** `stage_dataset_build` refuses, with nothing written, a corpus that set aside more than 25% of its loaded documents (`FAIL_REJECTED_SHARE`, measured by `corpus_set_aside` — the same helper preflight's corpus check reads), a corpus missing a split it needs, and documents read by more than one MinerU version.

**No training fan-out.** One `finetune` run launches **one** training job per scope. v1 launched 1 + N sequentially — a Foundation, then one adapter per document type stacked on it — and that topology is unservable: vLLM applies one LoRA per request, so the two could never both be active (arch v2.1 §4.1). A per-type adapter returns only through the §4.2 graduation gate, trained on the merged foundation and never stacked. With several `--scope`s, stages 1–4 build the corpus once over the union of their document types and stages 5–11 run once per scope, each with its own context (`run_multi_scope`); a block in one scope does not stop the others, and the exit code is non-zero if any failed.

**Where the gate's numbers come from.** The gate (in `package`) reads the candidate's scores from this scope's frozen-eval-set report (`eval-reports/v{n}/[scope/{scope}/]summary.json`, IMPL-08) — producing it, when absent, by serving the frozen set through the serving pipeline with the merged bf16 model and the release's bf16 calibrators — and the baseline from this tenant's currently promoted version's own report for the same scope (`latest_promoted(tenant=)`). A candidate report is reused only when its `eval_set` is this tenant's; otherwise the frozen set is scored again. Each report also keeps every document's scores and weights (`per_document`, `per_document_weights`) and the frozen set it was scored on (`eval_set`, IMPL-08); `paired_document_scores` pairs the two on the documents both scored, so non-inferiority is a **paired bootstrap**, weighted so it tests the metric the report states. It pairs only when both reports are on one yardstick: the same frozen set of this tenant, scored by the same code (`scoring_fingerprint`). A baseline on another yardstick — another frozen set, another tenant, other scoring code (`scoring_fingerprint`), or a report that does not record its yardstick — is not compared: the candidate is judged on the floors alone, without the improvement rule (decision 2026-10-22). On one yardstick, reports that cannot be paired — no shared documents, no per-document scores — are compared on point estimates, the improvement rule cannot be enforced, and the decision's warnings say so and why. Neither report is optional and neither is fabricated: with no report a dry run fails rather than gating on nothing, and with no promoted version the baseline is genuinely `None` — a first version — rather than a lookup that quietly failed.

**The gate is a hard stop inside `package`.** It runs after merge, quantize and calibrate (arch v2.1 §13), so a regression stops the **release**, not the build: merging still happens, nothing is packaged, and the staged artifacts sit on a volume that will be reclaimed. It exits non-zero with the per-metric verdicts — floor, interval and basis, not a bare delta. A passing release is classed **production** (field normalized match ≥ 0.92) or **interim** — below that, or whenever the scope was scored on fewer than 150 real frozen documents of a type it covers (twins do not count, decision 2026-10-22), with a warning. The count is the real documents the scope was scored on (`scope_eval_set_size`: the candidate report's own real counts, else the frozen manifest's real counts per line or per type; an older manifest without them is unknown, so interim), not the whole set's `small_set_accepted`. The `tier` is recorded in the gate decision and the bundle.

**A release must improve something, or name the defect it fixes.** With per-document scores, a release on which no primary metric (`field_normalized_match`, `list_field_recall`) improves (its interval's lower bound above zero) is blocked at `improvement`, unless `package` / `all` is given **`--fixes-defect "<ticket>"`** (`StageContext.fixes_defect`; `scope=<ticket>` per scope when several run, as `--release-id`; a blank or placeholder value is refused before any work). That stands in for an improvement and nothing else: floors and non-inferiority still apply, and the text is recorded in the gate decision. The first release is exempt.

There is no `--force`. There **is** an override (§15.5), requiring a named person, a written reason of at least 20 characters and the gates it waives, all recorded in the gate decision and the release bundle. It is given on `package` and `all` with `--override-approver`, `--override-reason` and `--override-gate` (repeatable): all three together, checked before any work, for one scope per run, naming real gates. It lifts only gates that failed, and they are recorded as waived; one that waived nothing is a warning. v1 had none at all, on the reasoning that a waivable gate is a suggestion — which was right about the risk and wrong about the remedy, because the v1 gate demanded improvement on twelve metrics within 0.001 and could not be passed at pilot volume at all.

**Flags:** `--input`, `--doc-types`, `--corpus-version`, `--out-version`, `--scope NAME` (repeatable; `configs/scopes.yaml`, default the unified scope — while only policies are targeted, name a policy scope: `policy` or a line scope, decision 2026-10-22; a unified release scored on no ACORD or Loss Run documents is capped at interim), `--gpu`, `--commit <sha>`, `--from-stage <name>` (resume), `--skip-ingest`, `--train-vit`, `--min-labels-per-type`, `--push-adapters` (see §3), `--tenant` (optional, defaults from env), `--dry-run`. `--foundation-only` is gone: one adapter per scope is the only topology.

## 3. The RunPod staging volume

> **Terminology, to prevent a real bug.** In this spec set "**registry**" means the durable run-manifest registry in Azure Blob (IMPL-02). The RunPod side is the "**staging volume**" and is never called a registry. They have different lifetimes and different guarantees.

RunPod **pods are ephemeral** — a pod clones the repo, works, pushes, and terminates (arch §14). Persistence between commands 1 and 2 therefore comes from a **RunPod network volume**, mounted at `/runpod-volume` and attached to every pod the controller launches.

```
$RUNPOD_VOLUME_MOUNT/staging/            # default /runpod-volume; /workspace on the pod (pod_bootstrap.sh)
  adapters/foundation/v{n}/              # unified run: ms-swift output, v*-<timestamp>/checkpoint-* inside
  adapters/scope/{scope}/v{n}/           # a scoped run (personal_lines, casualty_fleet, ...)
    checkpoint_scores/                   # multi-GPU checkpoint selection's worker output
  merged-models/{unified|scope/{scope}}/v{n}/
  quantized-models/{unified|scope/{scope}}/v{n}/vllm/{format}/
  train-data/{scope}/v{n}/               # the run's staged ms-swift JSONL
  train-images/{tenant}/{corpus}/        # page images, one cache per tenant and corpus version
  eval-reports/v{n}/[scope/{scope}/]
  serving-adapters/{release_id}/         # a serving cold start's release LoRAs
```

The layout deliberately mirrors the Blob layout (master §4) so `package` copies rather than translates.

**Why stage instead of pushing straight to Blob:** the merged model is ~16 GB. Quantization also runs on RunPod. Pushing the merged model to Azure at the end of command 1 and pulling it back at the start of command 2 costs a 32 GB round trip for no benefit.

**Durability rule — the one thing command 1 always sends to Blob.** A network volume is working storage, not an artifact of record. If command 2 never runs, or the volume is reclaimed, an untracked training run has happened. So `finetune` **always writes the run manifest to Blob** (IMPL-02) even though the weights stay staged, with `artifacts.status: "staged"` and `artifacts.staging_path` set. `package` later flips it to `"published"` and fills the real Blob paths. The manifest is a few KB; the lineage guarantee in arch §12 is worth that.

**Optional belt-and-braces:** `finetune --push-adapters` also pushes the LoRA adapter to Blob right after training — `adapter_config.json` and `adapter_model.safetensors` of the best-loss checkpoint (or the last), never the checkpoint's resume state — leaving only the merged model staged. `package` later publishes the selected checkpoint's two files to the same prefix, skipping any file Blob already holds. Off by default, per the staging design; recommended if a cycle's commands 1 and 2 may be separated by more than a day.

**Config:** `RUNPOD_VOLUME_ID` and `RUNPOD_VOLUME_MOUNT` (default `/runpod-volume`) in `.env`. The controller attaches the volume to every pod it launches; a launch without it is refused rather than silently writing to pod-local disk.

## 4. `package` — command 2 (stages 8 → 11)

Runs: **quantize → calibrate → evaluation & gate → release bundle (with its routing check) → push the adapter, merged model and quantized model(s) to Azure Blob → update the run manifests → write the bundle.**

- Reads `--version v{n}` from the staging volume. If that version is not staged, **fail loudly** naming the expected path and the remediation (re-run `finetune --from-stage merge`, or `--from-blob` if adapters were pushed with `--push-adapters`).
- Quantizes per `--formats` (default: **bf16 alone** — it is the merged model, so quantize is a no-op; FP8 only once the spike has verified it, per IMPL-10). GGUF, and its `mmproj`, is an on-request edge export, never a `package` output.
- Pushes **all three artifact classes** to their respective Blob locations (master §4), through `artifact_registry/transfer.py`:

| Artifact | Blob destination |
|---|---|
| The scope's LoRA adapter — `adapter_config.json` + `adapter_model.safetensors` of the selected checkpoint only | `adapters/foundation/v{n}/` (unified), `adapters/scope/{scope}/v{n}/` |
| Merged bf16 model | `merged-models/{unified\|scope/{scope}}/v{n}/` |
| Each quantized (non-bf16) serving format | `quantized-models/{unified\|scope/{scope}}/v{n}/vllm/{format}/` |
| Eval report | `eval-reports/v{n}/[scope/{scope}/]` |

  The adapter and merged-model pushes skip every file whose bytes Blob already holds (the MD5 recorded with each upload), so re-packaging a release sends nothing.
- Updates the run manifest: `artifacts.status` → `"published"`, real Blob paths, `quantized_formats[]`.
- `--keep-staging` retains the volume copy (default: clear **this scope's** staged paths after at least one manifest was published, so the volume doesn't fill).
- Quantization threshold validation (IMPL-10 / arch §13b) runs inside the quantize stage when per-format metrics are supplied (`StageContext.quant_metrics`); no CLI path supplies them yet, so a bf16-only cycle — bf16 *is* the reference — passes with a warning.

**Flags:** `--version`, **`--release-id release-YYYY.M.N`** (required; repeatable as `scope=release-YYYY.M.N` pairs when several scopes are named), `--formats`, `--skip-quantize` (push adapters + merged only), `--keep-staging`, `--from-blob`, `--dtype`, `--fixes-defect "<ticket>"` (repeatable as `scope=<ticket>`; also on `all`; §2), `--override-approver` / `--override-reason` / `--override-gate` (the written override, all three together; also on `all`; §2), and on a standalone `package` **`--scope`**, **`--doc-types`**, **`--tenant`**, `--corpus-version` (read from the run manifest when omitted). Every scope's `--release-id` and `--fixes-defect`, and the override, are checked in `main()` before the command detaches on the pod (`run.flag_problem`), so a bad flag is refused at once, not in a log.

**`--release-id` is checked before any stage runs** — under `all`, before training. Calibrators, gate decisions and the bundle are all addressed by it, and an invalid one used to surface only when calibrate tried to save what it had already fitted. It is named by the operator, never derived: a derived id would move between a failed run and its `--from-stage` resume and split one release across two ids. A missing or malformed id fails with the next free id for the month as the suggested fix; pass the same id when resuming. Each scope produces its own release, so one bare id given for several scopes is refused (`release_id_for`).

**Where the package stages get their inputs.** Nothing is hand-supplied in a real cycle:

- **Checkpoints** — after training, `discover_checkpoints` reads the `checkpoint-*` directories ms-swift wrote under the staged adapter directory (the most recent `v*-<timestamp>/` run), and the best-loss one from `trainer_state.json`. Selection (`stage_checkpoint_eval`) scores each as a decoder LoRA by generated field accuracy on a stratified sample of the scope's own validation view (`validation_sample_rows`: 400) through `evaluation/validation_generation.py`, whose metrics are the gate's own `build_report`. The best `selection_document_finalists` (2) are then generated on every validation row in `selection_document_modes` (`[ocr_plus_image]`) and judged on whole documents — windows merged as serving merges them — among finalists whose rates of invented and of missed printed values are within `selection_guard_margin` (0.02) of the best one's; with 0 or 1 finalists the near-tie rule (`selection_tie_break_margin`) decides. With several GPUs, one vLLM engine per GPU scores the candidates at once (`parallel_vllm_scorer`). Generation sends `validation_batch_rows` (96) rows per vLLM call. The stage result carries the selection and the generation cost (`generation`: rows and seconds); the choice is written to `eval-reports/v{n}/[scope/{scope}/]checkpoint_selection.json`.
- **Calibration samples** — when none are supplied, calibrate generates the validation split **with each staged serving format**, constrained as serving is, and labels every field's features correct or not against the golden label, split by the `val_half` the dataset build stamped. There is no shared or wildcard sample set. The stage records the expected calibration error per field type and overall.
- **Gate decision** — written under `releases/{tenant}/{release_id}/gate/bf16/`. The candidate metrics are the merged bf16 model's, so that is bf16's gate run and no other format's. Besides the metrics, the verdicts (each with its basis: `paired_bootstrap`, `point_estimate`, or `floor_only` with no baseline or one on another yardstick), the failed and waived gates, the override and the tier, it records `paired_documents` (how many documents each bootstrapped metric paired), `baseline_not_comparable` and `baseline_metrics_not_compared`, `fixes_defect`, `eval_set` (`small_set_accepted`, `documents_by_doc_type`, `frozen_from_corpus`, `frozen_at`, and the count that decided the interim cap: `scope_documents_by_doc_type`, `scope_documents_counted_from`, `minimum_per_doc_type`, `small_for_scope`) and the gate's `warnings`.

**The release bundle** (`releases/{tenant}/{release_id}/bundle.json`, plus a row in `release_index.json`) pins the scope, its `lines` and served `doc_types`, the adapter run, merged and per-format model paths, per-format calibrators and gate decisions, the release `tier`, a hash of every prompt input (`common.prompts.prompt_hash`: templates, schemas, the section map), schema versions, the OCR pin, and hashes of `configs/shared/vision.yaml`, `configs/inference/vllm_serving.yaml` and the dependency lock (`requirements-train.lock`; `pyproject.toml` only when the lock is missing). It also records the p95 latency per adapter and the peak GPU memory from the eval report's `release_measurements`, and the **routing table**. Its status is **`promoted` only when every serving format has its own calibrator and its own gate run**; otherwise it is written as **`gated`** with the missing pieces listed in the stage result. Until a per-format gate run exists for quantized formats, an FP8 release is `gated`.

**The bundle is built, and its routing check run, before anything is pushed** (`record_release_measurements`, `serving/routing_check.py`): every line in `configs/layout_families.yaml` is routed through the serving plan as it would stand with this release promoted (with the serving config's `release_pins`). A line that cannot be routed, or one this release was trained for that would not reach it, stops the release with a `PipelineError` — no weights in Blob, no manifest marked published.

`--scope`, `--doc-types` and `--tenant` must match the `finetune` run that produced the version. `finetune` decides *which* models get built; `package` publishes their locations, so packaging with the wrong ones publishes paths for artifacts that were never built — an empty Blob prefix that later reads as a published model — or another scope's paths under this one's release. (Under `all` they come from the finetune flag set; adding them twice is an argparse conflict.)

**`stage_push` publishes only runs that finished.** Manifests are selected by version *and status*: a run left at `training` or marked `failed` has no weights, and flipping it to `published` advertises a Blob path that serving will fetch and find empty.

## 5. `extract` — command 3 (arch §17)

A wrapper over `testing/run_extraction.py` (IMPL-12), which itself wraps the serving pipeline (IMPL-11) so test == prod. It reads documents already imported and OCR'd, from Blob.

**Model selection is the point of this command.** `--model` accepts:

| Value | Resolves to |
|---|---|
| `base` | Untuned `Qwen/Qwen3-VL-8B-Instruct` at the pinned revision, **no adapter** |
| `base --adapter <folder>` | The base with that LoRA (a training `checkpoint-N`, or a run's adapter folder; local or Blob) applied per request, **not merged** |
| `v1`, `v2`, … | That version's merged model, plus a graduated per-type adapter where the routed type has one |
| `v2 --format fp8` | That version's FP8 serving weights |

`--model base` is not a curiosity — it is the **zero-shot baseline** of the pilot protocol (IMPL-15) and the day-zero pre-annotation path (IMPL-04). `pilot/zero_shot_baseline.py` calls the same `testing.run_extraction.run_document` rather than a parallel implementation.

**Flags:** `--model`, `--adapter`, `--label`, `--source-ids …` | `--split train|val|test` (with `--corpus`), `--doc-type` (default `policy`), `--mode ocr_plus_image|image_only`, `--no-score`, `--format`, `--limit`, `--tenant`.

Outputs are those of IMPL-12: `results/{version}/` (each result with `all_fields`), and `metrics/{version}/` (per-document metrics and `summary.json`).

## 6. `all` — commands 1 + 2

Runs `finetune`, then `package`, sharing `--out-version`. Halts if the gate fails; nothing is published on a failed gate. Accepts the union of both flag sets; `--release-id` is checked before training starts. **Does not run `extract`.**

## 7. `orchestration/runpod_controller.py`

**Training — ephemeral pods (arch §14):**
- **Deployment: one RunPod H200 SXM pod runs every GPU stage in process**, inside tmux — OCR, training, checkpoint selection, merge, calibration and the gate share the cards, each vLLM engine released before the next loads. On a multi-GPU pod training is data-parallel over every GPU (`distributed.gpus: auto`, the effective batch unchanged), checkpoint selection runs one vLLM engine per GPU, and OCR can be sharded (`run_mineru --shard I/N`). Classes per stage stay configurable (`orchestration/config/pipeline.yaml`) for a future per-stage launcher; the original plan put OCR on a cheaper L4/L40S card.
- **Stages 2, 4 and 5 all want a GPU**, so `finetune` provisions **one pod for the whole command** rather than shuttling a corpus between machines. That is cheaper than three pod launches and removes two Blob round trips.
- **Training code lives in the private git repo; the pod clones it fresh at job start** at a pinned commit — code is never permanently resident on a pod.
- Every launched pod has the staging volume attached (§3).
- Pod job: pull base model + corpus from Blob → run → write to the staging volume → **terminate**.
- Poll status, stream logs (**PII-scrubbed**, master §8), handle failures and retries.

**Serving — persistent endpoint (arch §14):**
- Manage the **RunPod Serverless vLLM endpoint** (IMPL-11): deploy/update to a promoted version, health-check, roll back.

**No separate controller CLI.** Single steps run through `orchestration.run <command> --from-stage <stage>`; the endpoint through `orchestration.run deploy-endpoint` / `rollback-endpoint`.

## 8. `orchestration/pipeline_dag.py`

The 13 stages (arch v2.1 §13) as reusable, individually addressable stage functions — this is the implementation `run.py` composes, and the same functions an Airflow DAG or GitHub Actions workflow can schedule.

| # | Stage | Command | Writes to |
|---|---|---|---|
| 1 | **Ingestion** — checksum-deduped, immutable | `finetune` | `raw-documents/` (Blob) |
| 2 | **Preprocessing (GPU)** — MinerU OCR (3.4.5, pipeline backend, in the OCR environment) + page rendering at the resolution cap | `finetune` | `processed/` (Blob) |
| 3 | **Labeling** — human review | *(outside the CLI)* | `golden-labels/` (Blob) |
| 4 | **Dataset build** — compile JSONL, inject schema, modality split, train/val/test split | `finetune` | `corpus/v{n}/` (Blob) |
| 5 | **Training** — ONE LoRA per scope on a bf16 base | `finetune` | staging volume |
| 6 | **Checkpoint eval** — vLLM generation on a validation sample, finalists judged on whole documents | `finetune` | staging volume + `checkpoint_selection.json` (Blob) |
| 7 | **Merge** — PEFT `merge_and_unload()`, the selected checkpoint into bf16 | `finetune` | staging volume |
| 8 | **Quantize** — no-op for `[bf16]`; FP8/AWQ, vLLM-native, once verified | `package` | staging volume |
| 9 | **Calibrate** — per-field-type calibrators and risk-controlled thresholds, per serving format | `package` | release bundle |
| 10 | **Evaluation & gate** — frozen golden eval set, per serving format; production or interim tier | `package` | gate decision |
| 11 | **Package** — release bundle pinning adapter, prompt, schema, calibrators, OCR pin, serving config and the routing table; adapter files, merged model and formats pushed | `package` | Blob |
| 12 | **Serving** — endpoint pulls the promoted release bundle | `deploy-endpoint` | endpoint |
| 13 | **Feedback loop** — low-confidence output, human corrections, and a random 5% of auto-accepted documents → labeling queue | `extract` + IMPL-04 | next corpus version |

**Pipeline properties that are requirements, not aspirations (arch §13):**
- Every stage reads from and writes to Azure Blob or the staging volume — never pod-local disk.
- Every training, evaluation, and quantization job produces a **run manifest** (IMPL-02).
- **Stages are idempotent and resumable.** `--from-stage` re-enters mid-pipeline; a completed stage re-run is a no-op, not a duplicate.
  - **Ingestion is the exception, and always runs.** There is no cheap correct completion check for it — comparing an input directory against ingested checksums costs the same as ingesting — and the approximate one that used to be here ("has anything ever been ingested?") was true from cycle two onward, so a second `finetune --input ./new_batch` skipped the stage and built the corpus from the previous batch. `ingest_directory` is checksum-deduped, so re-running it writes nothing and reports duplicates. An approximate guard on cheap idempotent work buys nothing and can be wrong in the direction that loses documents.
- **The gate (stage 10) is a hard stop.** Promotion requires every gating metric to pass. There is no `--force-deploy`; the only exception is the recorded written override (§2), given with `--override-approver`, `--override-reason` and `--override-gate`. `package` reads the recorded gate decision and refuses a version whose gate did not pass — running `package` after a blocked gate used to publish it.
- **Every run says where its time went.** Each stage that runs records its `wall_seconds` (every attempt included) in its result, and the run report is written after every stage (*Current implementation*).

## 9. `orchestration/config/`
- GPU class per stage, corpus/version tags, staging volume id and mount, schedule, notification hooks, retry policy (`retry.max_attempts: 2`, `backoff_seconds: 30` × the attempt number).
- **Sweep orchestration (arch §11a) — deferred with IMPL-06 `sweep.py`:** when enabled, the 3-phase sweep runs as a scheduled job sequence (~9–12 runs before the first **production** run), each writing a manifest, winner promoted to the production training config. Not part of the first cycle.
- **Foundation-upgrade cascade (arch §12):** when a new unified major version passes its gate, generate the re-validation work list from `query_registry.adapters_depending_on()` (IMPL-02) and schedule retraining of every dependent GRADUATED adapter (§4.2) **before** it becomes production. Dormant until a type graduates — the default topology produces no dependents at all. The DAG produces the work list and waits for it to pass rather than silently promoting.

## Constraints
- Four lifecycle subcommands, one entrypoint. `all` = `finetune` + `package`, never `extract`.
- Training pods **ephemeral**; the staging volume is the only thing that persists between them, and it is not an artifact of record.
- `finetune` always writes its run manifest to Blob even when weights stay staged.
- Orchestration runs on cheap CPU infra **outside** RunPod; only GPU-bound work runs on RunPod (arch §14).
- Secrets from env only. Logs PII-scrubbed before leaving the pod.
- No flag around the evaluation gate; only a recorded, attributed override (§2), whose three flags build that record and nothing else.

## Acceptance checklist
- [ ] `run finetune` executes stages 1→7 on a tiny fixture set and leaves adapters + merged model on the staging volume.
- [ ] `finetune` reports the unlabeled backlog and trains on the labeled subset; it aborts when labels are below `--min-labels-per-type`.
- [ ] `finetune` launches **one** training job per scope (one unified adapter, or one per named scope).
- [ ] The dataset build refuses a corpus that set aside more than 25% of its documents, before writing anything — also when preflight ran with `--skip-corpus`.
- [ ] **A failed gate stops `package` before anything is published, exits non-zero, and prints per-metric verdicts (floor, interval and basis);** `package` run on its own refuses a version whose recorded gate did not pass.
- [ ] With per-document scores in both eval reports, non-inferiority is a paired bootstrap and the decision records `paired_documents`; a release that improves no primary metric is blocked at `improvement` unless `--fixes-defect` names the defect, which the decision records. A baseline on another yardstick (frozen set, tenant, scoring code, or unrecorded) is judged on the floors alone; reports on one yardstick that cannot be paired (no shared documents, no per-document scores) are compared on point estimates; either way the reason is in the decision's warnings. `--fixes-defect` is checked and given per scope, before the command detaches.
- [ ] A release whose scope was scored on fewer than 150 real frozen documents of a type it covers is at most `interim`, with a warning; the decision records the `eval_set` and the count it was judged on.
- [ ] `--override-approver`, `--override-reason` and `--override-gate` build a validated written override only when all three are given, for one scope, naming real gates; the decision and the bundle record it as a waiver.
- [ ] Training and packaging refuse a corpus built under another review rule, or holding policies with a `window_plan_version` that is not `policy_windows.window_plan_id()`; the release bundle records the window plan.
- [ ] The corpus build refuses a gold with no review record (a twin's is its seed's); preflight names every such label (FAIL) and reports annotation agreement (WARN while no minimum is set).
- [ ] `freeze-eval-set --mark-reviewed` copies review records into an existing frozen set only where the frozen gold is unchanged, changing nothing else, and exits 1 while any frozen gold still has none.
- [ ] `finetune` writes a run manifest to Blob with `artifacts.status: "staged"` even though weights are not pushed.
- [ ] `run package --version v2` pushes the **real** selected adapter (its `adapter_config.json` and `adapter_model.safetensors` only), the merged model and each non-bf16 serving format to their Blob paths, then flips the manifest to `"published"`.
- [ ] A release whose routing check fails pushes nothing.
- [ ] `package` fails loudly with remediation when the requested version is not on the staging volume.
- [ ] `run extract --model base` runs the untuned base model with no adapter; `--model base --adapter <checkpoint>` applies it per request; `--model v2` resolves that version's merged model, plus a graduated per-type adapter where the routed type has one.
- [ ] `run all` chains 1 and 2, **never runs extraction**, and does not reach `package` when the gate fails.
- [ ] `--from-stage` resumes mid-pipeline; re-running a completed stage is a no-op — **except ingestion**, which always runs and relies on checksum dedup.
- [ ] A completion check covers **every** target it claims to, not just the first: a quantize run that failed halfway must not read as complete on resume.
- [ ] The staging volume is cleared only after at least one manifest was actually published.
- [ ] A pod launched without the staging volume attached is refused.
- [ ] Each stage that runs records `wall_seconds`, and the run report is rewritten after every stage.
- [ ] **Cross-type regression evidence carries both sides.** `StageContext` exposes `cross_type_evidence` as `{doc_type: {"current": {...}, "candidate": {...}}}` — the shape `promotion_gate` reads — and it is passed through verbatim. It is a **separate field** from `revalidation_evidence`, which holds per-doc-type booleans for the arch §12 cascade; reusing one field for both produced an entry with no `"current"` key, which the gate correctly rejected as empty. A continued Foundation could then not pass by any input: supply metrics and it read as empty evidence, supply the booleans and it read as no evidence at all.
- [ ] **A deterministic failure is not retried.** The retry policy exists for transient faults — a throttled Blob read, a pod that dropped. `PipelineError`, `PathError` and the gate's own refusals (`GoldenEvalError`, `EvalError`, `FreezeError`, `GateError`) mean the stage cannot run at all with these inputs (below the day-zero label floor, missing `--input`, a malformed `--out-version`, a checkpoint scorer that cannot start), so a second attempt re-lists everything and fails with the identical message after a pointless backoff. They stop the run immediately, like a gate block.
- [ ] **A dry-run `rollback-endpoint` does not move the endpoint.** Popping the deployment history before checking `dry_run` meant a preview permanently rewrote the recorded live version: `health_check` reported the previous version while the current one was still serving, and the next real rollback refused with "nothing to roll back to".
- [ ] A pod leaves no persistent process after completion.
- [ ] `deploy-endpoint --model vN` updates the serving endpoint; `rollback-endpoint` restores the previous promoted version.
- [ ] *(deferred)* Quantization thresholds gate `package`; the sweep sequence runs Phase 1 → 2 and promotes the winning config.
- [ ] A Foundation version bump produces the dependent-adapter re-validation work list and blocks promotion until it passes.

---

## Current implementation (2026-10-21)

**Commands** (`orchestration/run.py`): `finetune`, `package`, `all`, `extract`, `deploy-endpoint`,
`rollback-endpoint`, and **`freeze-eval-set --corpus vN [--tenant T] [--allow-small]`**, or **`freeze-eval-set --tenant T --mark-held-out`**
on an already frozen set (marks the documents the delivery calls unseen carriers', after a re-import, and records
their carriers by line; nothing else changes), or **`freeze-eval-set --tenant T --mark-reviewed`** (copies each
frozen document's review record from its label metadata, which the gate needs, only where the frozen gold is
still the current one; nothing else changes; exit 1 while any frozen gold still has none). The two mark flags
run together when both are given, and either is refused with `--corpus` or `--allow-small`
(IMPL-08): the first copies that
tenant's corpus test split into `golden-eval-set/{tenant}/` — each tenant freezes its own, from its own corpus,
and is gated on it. It refuses fewer than 150 real test documents per type (twins do not count) unless `--allow-small` (recorded as
`small_set_accepted` in the set's manifest; the gate caps at interim a release whose scope was scored on fewer
than 150 of its documents of a type), refuses a second
freeze, refuses a test split holding a gold with no review record, and refuses while a set frozen before sets were per tenant still sits at the root. After a freeze the
tenant's corpus builds leave the frozen documents and their families out and split into train and val only.

**Scopes** (`configs/scopes.yaml`, `--scope`, repeatable): `unified`, `policy`, `lossrun`, `personal_lines`
(policies of the seven personal lines) and `casualty_fleet` — the casualty-and-fleet family's adapter, trained
for now on commercial general liability alone (`lines: [gl]`), from the CGL source documents under a tenant of
its own (`--tenant`). Training refuses a scope that would train a layout family on two output shapes
(`common.scopes.assert_one_output_shape`): `auto` and `wc` are still self-contained schemas while `gl` is on the
common model, so they join `casualty_fleet` only once they move. The run report and staged paths are per scope,
so two scopes or tenants can share a version tag.

**Every long job survives a disconnect** (`orchestration/detach.py`, `scripts/pod_run.sh`):
- On the pod — `RUNPOD_POD_ID` set, or RunPod's `/etc/rp_environment`, or Linux with `/workspace` mounted
  and a GPU present (a laptop with a GPU and a `workspace` folder is not the pod) — every long entry point
  re-launches itself in a tmux session through `pod_run.sh start <name> -- <command>` and returns at once with
  the job's name. Covered: every `orchestration.run` command except the endpoint switches,
  `orchestration.preflight`, `orchestration.smoke_run`, `training.train`, `training.sweep`, `evaluation.run_eval`,
  the checkpoint-score workers, OCR and rendering, `data_pipeline.ocr_check`, the audit, the intake pull, the
  importers and bundle preparation, pre-annotation, active learning, `testing.run_extraction`,
  `testing.compare_models`, the Phase 0 spike, `setup_pod.sh` and `pod_bootstrap.sh`. The guard runs after
  argument parsing, so `--help` and bad arguments answer at once.
- Inside tmux, or inside a `pod_run.sh` run (`FIDEON_DETACHED=1`), a command runs in place — never a second
  copy. Off the pod nothing changes. `FIDEON_NO_DETACH=1` opts one command out.
- `pod_run.sh start <name> -- <command>` (or `start <name> <orchestration.run arguments>`),
  `attach <name>`, `status [<name>]`, `tail <name>`, `list`, `stop <name>`: logs and exit codes in
  `$FIDEON_LOG_DIR` (default `/workspace/logs`); one run per name; nothing ends a run except `stop`, which asks
  for the run's name while it is still running. It cannot survive the pod itself stopping — `status` then reports
  the run as interrupted and `--from-stage` resumes.
- `tests/test_detach.py` fails if a new entry point is neither guarded nor listed as quick.
- **A run keeps its caller's environment**: each run sources the repo's `.env` and records the caller's `PATH`,
  `VIRTUAL_ENV` and `CUDA_VISIBLE_DEVICES` (set or unset), because a tmux session otherwise takes them from the
  tmux server (started by whichever job came first); training finds `swift` next to the running Python before
  searching `PATH`. The bootstrap builds its venvs with Python >= 3.11 (`FIDEON_PYTHON` overrides), since
  `python3` on Ubuntu 22.04 can be the system 3.10.

**One command, OCR to frozen eval set.** On the pod, with the training environment active, one detached run
chains the steps, each running in place inside it:

```bash
bash scripts/pod_run.sh start cgl-v1 -- bash -c '
  /workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy --all-unprocessed --tenant T &&
  python -m data_pipeline.ocr_check --doc-type policy --tenant T &&
  python -m orchestration.preflight --scope casualty_fleet --tenant T --corpus-version v1 --out-version v1 --skip-corpus &&
  python -m orchestration.run finetune --scope casualty_fleet --skip-ingest --tenant T --corpus-version v1 --out-version v1 &&
  python -m orchestration.run freeze-eval-set --corpus v1 --tenant T --allow-small'
```

`preflight --skip-corpus` because `finetune` builds the same corpus and refuses the same >25% set-aside itself, so
the corpus is built once.

**Stage behaviour** (`orchestration/pipeline_dag.py::run_stages`):
- **Logging is re-asserted before every stage** (`common/logging_setup.py`): `orchestration.run` installs a
  `ConsoleHandler` — a `StreamHandler` subclass, which ms-swift's and modelscope's logger setup leave alone — and
  `reassert()` lowers whatever a library raised and re-enables this repository's loggers.
- **Each stage that runs records `wall_seconds`** (every attempt included) in its `StageResult` data; a stage
  skipped as already complete records none.
- **A progress report is written after every stage** — and on a block or failure — to `$FIDEON_LOG_DIR`
  (default `/workspace/logs`) when that folder exists, as `run-<command>-<scope>-<tenant>-<version>.json`
  (`progress_report_name`), so a run that dies part-way still says what each finished stage took. A report that
  cannot be written never stops the run.
- **Retries**: a failed stage is retried up to `retry.max_attempts` (2) after `backoff_seconds` × attempt. Never
  a gate block; never a `PipelineError`, `PathError`, `GoldenEvalError`, `EvalError` (`EvalSetLeakage`
  included), `FreezeError` or `GateError` (the run stops at once); never the deterministic training
  errors (`TrainingError`, `StagingError`, `LengthCheckError`, `CorpusViewError`, `CalledProcessError`). A
  checkpoint scorer that cannot start (a split or setting it cannot use) raises `PipelineError` and is not retried.
- `checkpoint_eval`'s result includes the generation cost (`generation`: rows and seconds).
- Training raises when a real run leaves no checkpoint; `checkpoint_eval` rediscovers checkpoints on a
  resumed run; merge reads the selection from Blob when resuming, and refuses a real run with none.
- Engines are released between stages (checkpoint selection → merge → calibration per format → golden eval),
  so each finds the card free.
- `package` requires a passed recorded gate, finds calibrators in Blob on resume, builds the bundle and its
  routing check before pushing, and uploads real weights through `artifact_registry/transfer.py` (IMPL-10):
  adapter files only, identical files skipped by MD5.
- Training and packaging read the corpus manifest and refuse a corpus this code would not build today
  (`pipeline_dag._assert_corpus_pins`): one built under another review rule (`review_rule_version`, so a
  reused corpus cannot train golds the dataset build never checked), and one holding policies whose
  `window_plan_version` is not `policy_windows.window_plan_id()` — the prompt hash is computed at packaging,
  so it cannot catch an adapter trained on another window plan. The release bundle records the window plan,
  and serving checks it at cold start (IMPL-11).
- `--push-adapters` uploads the best-loss checkpoint's `adapter_config.json` and `adapter_model.safetensors`
  after training (non-dry runs), not a placeholder and not the checkpoint's resume state.
- The dataset build excludes the tenant's frozen eval documents and their families once its set is frozen
  (IMPL-05), refuses documents read by more than one MinerU, refuses a >25% set-aside, and refuses a gold
  with no review record (`load_labeled_documents`, through `export_golden_labels.review_problem`: no real
  `reviewer_id` and `review_date` in its metadata, or a twin naming no seed it inherited them from; the message
  says they "have no review record").
- The run manifest's `data_stats.split_policy` (`pipeline_dag.split_policy`) records the carriers held out per
  line and the lines naming fewer than the 10 carriers a hold-out needs (`lines_below_carrier_minimum`), which
  hold none out (IMPL-05).
- The stage wiring carries no page selection: there is no `page_select` task and no `long_doc_types`; every
  policy is planned in windows over every page (decision 2026-10-19, IMPL-05/11).
- The gate stage (`stage_evaluation_gate`) passes `promotion_gate` the per-document scores
  (`StageContext.per_document`, injected by tests; otherwise `paired_document_scores` reads both eval reports,
  and only against the promoted report — an injected baseline has none to pair with — and only when both name
  the same frozen set of this tenant), the reasons it could not pair as `warnings`, `fixes_defect`, the
  `--override-*` record as `override`, and `small_eval_set` from what the scope was scored on
  (`scope_eval_set_size`).

**Preflight** (`python -m orchestration.preflight --scope S [--tenant T] [--skip-corpus]`): PASS/WARN/FAIL per
check, exit 1 on any FAIL, report in `/workspace/logs/preflight.json`; detached on the pod. Checks: every visible
GPU at least 90% free and the training GPU count dividing the batch; packages (torch, transformers 4.57, vLLM,
flash-attn, ms-swift, peft, deepspeed; `swift` next to the Python); the base model verified at the pinned revision;
config invariants; 100 GB free; Azure write, overwrite, read and delete; every labelled document OCR'd and at
least `--min-labels-per-type`; **reviewed labels** (`check_reviewed_labels`: FAIL naming each label with no
review record, the labels the corpus build would refuse, with the first one's reason); **label agreement** (`check_label_agreement`: how many documents are double-labelled
and their mean and lowest `agreement_score`; WARN, since `common.constants.MIN_INTER_ANNOTATOR_AGREEMENT` is
`None` — no spec states a minimum — and a FAIL below it once one is set); the **post-OCR report** (`--ocr-check`, default `/workspace/ocr_check`) passed; and
the **corpus**, built in memory by `plan_corpus` (rows per split, documents set aside and why, WARN above 5%,
FAIL above 25% via `corpus_set_aside`). `--skip-corpus` skips that build, for a chain in which `finetune` follows.

**Pod setup** (`scripts/setup_pod.sh ocr|train|serve|quantize|dev`): installs the role's requirements file,
builds flash-attn against the installed torch, installs tmux, checks packages and CUDA.

**Pod bootstrap** (`scripts/pod_bootstrap.sh [--pdf <policy.pdf>]`): one command for a fresh H200 SXM pod,
detached in tmux and safe to re-run. In order: GPU (warns if not an H200; reads every card of a multi-GPU pod),
a driver supporting CUDA 12.8, `/workspace` mounted, the repo on the volume, Python >= 3.11, 150 GB free
(`FIDEON_MIN_FREE_GB`); `.env` keys `RUNPOD_VOLUME_MOUNT=/workspace`, `HF_HOME`,
`MINERU_TOOLS_CONFIG_JSON=/workspace/mineru.json` (replacing a MinerU 1.x `magic-pdf.json` value),
`MINERU_DEVICE_MODE=cuda`, `MINERU_MODEL_SOURCE=local` and `HF_HUB_ENABLE_HF_TRANSFER=0`, filled only when unset,
blank or the template's default; a Blob login — `AZURE_STORAGE_ACCOUNT_URL` + `AZURE_STORAGE_SAS_TOKEN`, or
`AZURE_STORAGE_CONNECTION_STRING` — with a refusal if the connection string (`;`) or a token (`&`,
`AZURE_STORAGE_SAS_TOKEN` or `AZURE_RAW_SAS_TOKEN`) is unquoted, each line read as bash reads it (`export`
and indented lines, an unquoted trailing ` # comment`, an escaped `\&`/`\;` and a trailing CR set aside), and
half a SAS login (blank or CR-only values count as empty, as Python strips them) or a raw token with no
separate raw container listed as still to do; `/workspace/venv`
(`setup_pod.sh train`) and `/workspace/venv-ocr` (`setup_pod.sh ocr`, MinerU 3.4.5), rebuilt only when their
requirements change; MinerU's pinned pipeline weights and config (`scripts/download_mineru_models.sh`, checked by
`data_pipeline.ocr.mineru_config`); the base model at the pinned revision into `/workspace/models` (an existing
copy is verified by size and SHA-256, not re-downloaded); the spike in the training environment
(`--skip-mineru`) and, with `--pdf`, its MinerU checks in the OCR environment (`--only-mineru`). It ends with
what is still to do.

**OCR before `finetune`**: `stage_preprocessing` refuses with a `PipelineError` naming the OCR environment's
command when documents still need OCR and MinerU (`mineru`) is not importable (the training environment). With
nothing pending it needs no MinerU.

**RunPod backend**: `RunPodBackend` (launching pods through the API) is still unimplemented, and so is the
endpoint update behind `deploy-endpoint` / `rollback-endpoint` (a dry run records only); `LocalBackend` runs the
DAG in-process on whatever machine starts it — on the pod, inside tmux.

**Post-OCR check and smoke run.** `data_pipeline/ocr_check.py` checks every labelled value against its page's OCR
text in `processed/` (the audit's matching), reporting `fideon:filled` fields apart from the generator's own, real
apart from synthetic and per line; it exits non-zero when added fields are found on their page more than
`--max-gap` (default 0.05) less often than original ones. `orchestration/smoke_run.py` runs a few whole source
families (4 train, 1 val, 1 test by default, one per line first) through the steps `select`, `import`, `ocr` (in
`/workspace/venv-ocr`), `check`, `preflight` and `finetune` (`--steps` picks them), under its own tenant (`smoke`
by default; never the real data's) and version (`v0`), with `--scope` choosing the adapter (`personal_lines`, or
`casualty_fleet` for the CGL batch) and `--min-labels-per-type 1`. Its preflight gets `--skip-corpus` only when
`finetune` runs too. A run starting at `select` refuses a tenant or version an earlier smoke run used; a failed run
resumes from the failed step. It never freezes an eval set or packages a release: a smoke subset is no yardstick.

## Changes since 2026-10-05

- Added the alignment line and dated *Current implementation* 2026-10-13.
- §1: `finetune` covers ingest → merge with checkpoint eval and no gate; examples use `--scope` and `--formats bf16`; `extract` examples use `--tenant` with `--split`/`--corpus`, `--source-ids` or `--adapter` instead of `--input`/`--ground-truth`.
- §2: one job per scope and `run_multi_scope`; the dataset build's >25% set-aside guard; the gate's report is per scope and produced through the serving pipeline; production/interim tier; flags gained `--scope`, `--train-vit`, `--dry-run`.
- §3: staging layout per scope and tenant (`checkpoint_scores/`, `serving-adapters/`); `--push-adapters` pushes the two adapter files only.
- §4: package stage order with bundle and routing check before the push; Blob table corrected (scope paths, `vllm/{format}`, no GGUF, adapter files only, MD5 skip); quant-threshold gate runs only on supplied metrics; flags (`scope=id` release ids, `--scope`, `--corpus-version`, no `--foundation-only`); checkpoint selection (sample, whole-document finalists, guard margin, one engine per GPU, generation cost); bundle contents (tier, lines, prompt-input hash, lock-file hash, latency/memory, routing table).
- §5–§8: extract flags and outputs; `all` checks the release id before training; multi-GPU deployment note; removed the non-existent controller CLI; stage table rows 2, 5–8, 10, 11; per-stage wall time.
- *Current implementation*: freeze-eval-set per tenant; scopes incl. CGL-only `casualty_fleet` in its own tenant; detach coverage, `pod_run.sh` usage and `CUDA_VISIBLE_DEVICES`; the one-command chain; logging re-assert, `wall_seconds`, progress report naming, retries and deterministic errors; preflight checks and `--skip-corpus`; bootstrap for MinerU 3.4.5 (`mineru.json`, pinned weights); endpoint deploy not wired; smoke-run steps and the `--skip-corpus` rule; the smoke run's reason for not freezing.
- `freeze-eval-set --mark-held-out` (commit f5668a4).

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx` (§0e added).
- §1: the `package` example shows `--fixes-defect`.
- §2: the corpus build refuses an unreviewed model-drafted gold; the gate pairs per-document scores (paired bootstrap; a point comparison with a warning for older reports); the interim cap for a set frozen with `--allow-small`; the improvement rule is enforced, `--fixes-defect` standing in for an improvement.
- §4: `--fixes-defect` flag; the gate decision's new fields (`paired_documents`, `fixes_defect`, `eval_set`, `warnings`, each verdict's basis).
- Acceptance checklist: per-metric verdicts rather than deltas; new items for the paired bootstrap and `--fixes-defect`, the interim cap, and reviewed labels.
- *Current implementation*: `small_set_accepted` read by the gate; the dataset build's reviewed-gold refusal; `split_policy.lines_below_carrier_minimum`; no page selection in the stage wiring; what the gate stage passes `promotion_gate`; preflight's `check_reviewed_labels` and `check_label_agreement`; the bootstrap's SAS or connection-string login.

## Changes of 2026-10-22 (code review)

- Bootstrap: the quoting check reads `export` and indented lines, covers `AZURE_RAW_SAS_TOKEN`, and lists half a SAS login as still to do.
- §2 and §4: the paired bootstrap is weighted and pairs only reports scored on the same frozen set of the tenant, with every fallback a warning in the decision; the interim cap follows what the scope was scored on (`scope_eval_set_size`); `--fixes-defect` is checked and given per scope; the written override on the command line (`--override-approver`, `--override-reason`, `--override-gate`, all three, one scope); the decision's new `eval_set` fields.
- Training and packaging refuse a corpus that holds policies built with another window plan (`pipeline_dag._assert_window_plan`).
- Acceptance checklist: the pairing fallbacks and per-scope `--fixes-defect`, the per-scope interim cap, the override flags, the window-plan refusal.
- *Current implementation*: the freeze's cap per scope; what the gate stage passes `promotion_gate`; the window-plan refusal.
- §2 and *Current implementation*: the corpus build refuses every gold with no review record (decision 2026-10-22; a twin's is its seed's) and says so; preflight's `check_reviewed_labels` fails on the same labels and no longer warns.
- `freeze-eval-set --mark-reviewed`: review records copied into an existing frozen set, exit 1 while any is missing; the freeze refuses a gold with no record.
- Acceptance checklist: the review-record refusal and `--mark-reviewed`.
- Flags: name a policy scope while only policies are targeted (decision 2026-10-22); the examples already use `personal_lines`.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.
- Training and packaging also refuse a corpus built under another review rule (`_assert_corpus_pins`); the bundle records `window_plan_version`.
- Bootstrap: an unquoted value is judged as bash reads it (a trailing comment, escaped `\&`/`\;` and a CR set aside), and the half-login check trims each value first.
- `--mark-reviewed` copies only onto an unchanged frozen gold; `--mark-held-out` records the delivered carriers.
- The baseline is this tenant's promoted run, and a candidate report is reused only when it is this tenant's; a baseline on another yardstick is judged on floors alone.
- Flags are checked before the command detaches; override gate names are checked; the gate's own refusals are not retried; `--mark-held-out` and `--mark-reviewed` run together.
- The interim cap and the freeze's minimum count real documents only.
