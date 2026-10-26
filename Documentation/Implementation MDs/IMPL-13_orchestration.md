# SPEC 13 — Orchestration (Command Surface + RunPod Controller + Pipeline DAG)

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–12).
>
> **Architecture refs:** `finetuning-architecture-v2.7.docx` §0e (decisions of 2026-10-19), §13 (**end-to-end pipeline, 11 stages, idempotency, the gate**), §13c (**operator command surface**), §14 (**RunPod: ephemeral training pods vs. persistent serving endpoint**), §12 (every cycle recorded), §17 (extraction routine), §11a (sweep orchestration — deferred).

## Goal

Expose the whole lifecycle through **three operator commands plus one umbrella command**, so a cycle is run by a person who knows the version tag and nothing else about paths, pods, or stage wiring.

## 1. The command surface

Single entrypoint, `orchestration/run.py`, with four lifecycle subcommands (plus `promote`, `versions`, `deploy-endpoint`, `rollback-endpoint` and `freeze-eval-set`; see §4 and *Current implementation*). Every cycle trains the one policy adapter (`common/adapter.py`; decision 2026-10-22) and takes the next version, which names its run (`policy-v3`), adapter and release:

| # | Command | Covers arch §13 stages | Ends with |
|---|---|---|---|
| **1** | `finetune` | 1 → 6 (ingest, OCR, labeling, dataset build, train, checkpoint eval) | The adapter on the **RunPod staging volume** |
| **2** | `package` | 7 → 9 (calibrate, **gate**, bundle and push) | A promoted **release bundle** in Azure Blob; the release it replaces archived |
| **3** | `extract` | §17 extraction routine | Extracted JSON + confidence + metrics, locally |
| **—** | `all` | `finetune` then `package` | Same as command 2 |

**`all` never includes `extract`.** Extraction is a separate concern — you run it against a chosen model version whenever you want, including against models trained weeks earlier and against the untuned base. Folding it into the build would conflate "produce a model" with "use a model."

```bash
# 1 — ingest through checkpoint selection. The adapter is staged on RunPod, not yet in Blob.
#     A new cycle takes the next version itself and says so ("this cycle is v3"),
#     and continues the tenant's promoted adapter unless it has to start from the base.
python -m orchestration.run finetune --input ./intake

# start from the base model instead (after a data deletion, or to reset drift)
python -m orchestration.run finetune --input ./intake --fresh

# resume a cycle, or rerun one that failed: name its version
python -m orchestration.run finetune --version v3 --from-stage training

# 2 — calibrate, gate, then publish and promote the release bundle (releases/{tenant}/v3/).
python -m orchestration.run package \
       --version v3 \
       [--fixes-defect "<ticket>"]          # only when the release exists to fix a defect

# the registry, and a rollback
python -m orchestration.run versions
python -m orchestration.run promote --version v2 --by "<person>"

# 3 — extraction on imported, OCR'd documents, model chosen by the operator.
python -m orchestration.run extract --model base --tenant T --split test --corpus v2
python -m orchestration.run extract --model v2   --tenant T --source-ids policy_0001 policy_0002
python -m orchestration.run extract --model base --adapter <staged checkpoint-N> --tenant T --split val --corpus v2

# all — commands 1 and 2 back to back. Extraction excluded by design.
python -m orchestration.run all --input ./intake
```

Each subcommand is a thin argument-parsing shell over `pipeline_dag.py` stage functions (§4). No orchestration logic lives in the CLI layer.

## 2. `finetune` — command 1 (stages 1 → 6)

**`finetune` produces artifacts; `package` judges and publishes them.** The gate moved into `package` under arch v2.1 §13, because it scores what is served, with the calibrators `package` fits first. Since 2026-10-23 what is served is the base with the selected adapter applied: nothing is merged or quantized.

Runs, in order: **ingest → OCR → labeling → dataset build → train (the ONE policy adapter, continuing the promoted release unless it starts from the base) → select checkpoint.**

**The human-labeling precondition.** Stage 3 (labeling) is human work and cannot be inside an automated command. `finetune` therefore:

1. Ingests and OCRs **everything** in `--input` (OCR in practice runs first, in the OCR environment; see *Current implementation*).
2. Builds the corpus from **only the source_ids that have a validated `golden.json`** (IMPL-04) — and refuses the run, naming them, if any gold has no review record (`load_labeled_documents`; no real `reviewer_id` and `review_date` in its metadata, a twin's inherited from its seed; decision 2026-10-22).
3. **Reports the unlabeled backlog** — a count and a `source_id` list written to the run report — and continues with what is labeled.
4. Aborts before training only if the labeled set is empty, or below `--min-labels-per-type` (default 25, matching the day-zero rule in IMPL-04).

This is the honest shape: you ingest 500 documents, 200 are labeled, you train on 200, and the command tells you 300 are waiting for a reviewer.

**The corpus is refused before anything trains on too little of it.** `stage_dataset_build` refuses, with nothing written, a corpus that set aside more than 25% of its loaded documents (`FAIL_REJECTED_SHARE`, measured by `corpus_set_aside` — the same helper preflight's corpus check reads), a corpus missing a split it needs, and documents read by more than one MinerU version.

**No training fan-out.** One `finetune` run launches **one** training job: the policy adapter, a LoRA on the bf16 base over all the policy data so far, new and existing (decision 2026-10-22). There are no training scopes and no layout families (Fideon SPEC_09's five adapter groups are knowingly not followed); `--scope` and `run_multi_scope` are gone. v1 launched 1 + N sequentially — a Foundation, then one adapter per document type stacked on it — and that topology is unservable: vLLM applies one LoRA per request, so the two could never both be active (arch v2.1 §4.1). A per-type adapter returns only through the §4.2 graduation gate, trained on the merged foundation and never stacked.

**Continue or start fresh** (`pipeline_dag.continuation`; the user's decision of 2026-10-23). A cycle continues the tenant's promoted release: its adapter's weights, with a fresh optimizer and learning-rate schedule (ms-swift `adapters`), on the original base model — never a merged model. It starts from the base instead when `--fresh` is given (after a data deletion, or to reset drift), nothing is promoted, the parent's run manifest cannot be read, it trained on another base model, it has another LoRA rank, alpha or target modules, its adapter is not in Blob (`adapter_config.json` or `adapter_model.safetensors` missing), its `adapter_config.json` cannot be read or was saved with other LoRA settings than the config (`training.train.adapter_config_mismatch`: dropout, rsLoRA and bias as well as rank, alpha and target modules), its release has no bundle, or its release was packaged with another prompt hash or window plan (a template, schema or section-map change), or its eval report is missing or on another yardstick (the set frozen again, or other scoring code), which the gate's line rule would block only after training. The promoted release's bundle is read by its release id (`promoted_release`), so a release with an old-style name (`release-YYYY.M.N`) is continued too. The reason is logged and recorded on the run manifest (`fresh_reason`) and in the training stage's result (`continued`, `fresh_reason`); the parent is `RunManifest.parent_run_id`, also in the registry index row. The parent's adapter is copied to staging (`parent-adapters/policy/vN`). A continued run uses `continued_training`'s non-null values (IMPL-06). A continued run that stopped part-way resumes its own checkpoint; the run fingerprint names the parent and the weights it starts from (IMPL-06). The first cycle on the new training set and canonical schema will start fresh (the prompt hash changes); later ones continue.

**What the operator should know about continuing:**
- Old documents train again every cycle, so over many cycles they are seen more often than new ones — a memorisation risk. Fewer epochs or a lower rate for continued cycles (`continued_training`, set from a measured continued run) is the lever, and the gate on the frozen set catches what it costs.
- A wrong label, learned once, carries forward until it is fixed in the data.
- Reproducing a version needs its chain of parents (`parent_run_id`).
- Start fresh (`--fresh`) after a data deletion and from time to time to reset drift; a schema, prompt, LoRA-settings or base-model change starts fresh by itself, and so does a change to the scoring code or a set frozen again. The scoring code changed on 2026-10-24, so a cycle continuing a release scored before then starts from the base instead; none is promoted yet.

**The version.** A new `finetune` or `all` takes the next version itself: one more than the highest number used by the adapter's runs (the registry index) or by any tenant's corpora (`corpus/{tenant}/vN`) and releases (`releases/{tenant}/vN`) (`registry_utils.releases.next_cycle_version`, `artifact_registry.paths.next_version`). One sequence for every tenant: runs, weights and eval reports are not kept per tenant, so two tenants on one number would write one run's artifacts over the other's (before the code review of 2026-10-23 only this tenant's corpora and releases counted). `--version vN` names one only to resume (`--from-stage`, which refuses to run without it: the next free number after a first attempt that wrote anything would be a new cycle) or to rerun a failed cycle. The corpus defaults to the cycle's version (`--corpus-version` names one already built).

**Where the gate's numbers come from.** The gate (in `package`) reads the candidate's scores from the adapter's frozen-eval-set report (`eval-reports/v{n}/scope/policy/summary.json`, IMPL-08) — producing it, when absent, by serving the frozen set through the serving pipeline with the base and the selected adapter applied (as served) and the release's bf16 calibrators, each document's main mode read a second time for `run_to_run_agreement` (`run_to_run_check`, on) — and the baseline from this tenant's promoted release's own report: the release marked `promoted` in the tenant's release index (`pipeline_dag.promoted_version`). Before 2026-10-22 it read a run-manifest status `promoted` that no real run ever set, so every candidate was gated as a first version. The frozen set is scored on the lines the release serves only (`evaluate_version(lines=served_lines(ctx))`; the report records `lines_scored`): serving sends every other policy — another line, a package, no line — to the base model. A run with no trained line fails the gate step before the model loads. A candidate report is reused only when its `eval_set` is this tenant's and it was scored on the same lines and the same checkpoint (`adapter_checkpoint`: the selected one, else the staged adapter directory); otherwise the frozen set is scored again. Each report also keeps every document's scores and weights (`per_document`, `per_document_weights`) and the frozen set it was scored on (`eval_set`, IMPL-08); `paired_document_scores` pairs the two on the documents both scored, so non-inferiority is a **paired bootstrap**, weighted so it tests the metric the report states. It pairs only when both reports are on one yardstick: the same frozen set of this tenant, scored by the same code (`scoring_fingerprint`). A baseline on another yardstick — another frozen set, another tenant, other scoring code (`scoring_fingerprint`), or a report that does not record its yardstick — is not compared: the candidate is judged on the floors alone, without the improvement rule (decision 2026-10-22). So is a promoted report scored on other lines than the candidate serves, or without its calibrators (`calibrated: false`, as an ad-hoc `run_eval` writes it to the same key). A promoted `release-YYYY.M.N` row whose adapter is not a `policy-vN` run is no baseline (a warning), rather than the policy adapter's report at that number. On one yardstick, reports that cannot be paired — no shared documents, no per-document scores — are compared on point estimates, the improvement rule cannot be enforced, and the decision's warnings say so and why. Neither report is optional and neither is fabricated: with no report a dry run fails rather than gating on nothing, and with no promoted version the baseline is genuinely `None` — a first version — rather than a lookup that quietly failed.

**The gate is a hard stop inside `package`.** It runs after calibrate (arch v2.1 §13; there is no merge or quantize stage since 2026-10-23), so a regression stops the **release**, not the build: the adapter is trained and selected, nothing is packaged, and the staged artifacts sit on a volume that will be reclaimed. It exits non-zero with the per-metric verdicts — floor, interval and basis, not a bare delta. A passing release is classed **production** (field normalized match ≥ 0.92) or **interim** — below that, or whenever the release was scored on fewer than 150 real frozen policies (twins do not count, decision 2026-10-22), with a warning. The count is the real documents the candidate was scored on, of the lines it serves (`eval_set_size`: the candidate report's own real counts, else the frozen manifest's `real_documents_by_doc_type`; an older manifest without them is unknown, so interim), not the whole set's `small_set_accepted`. The `tier` is recorded in the gate decision and the bundle.

**A continued run is held, line by line, to the runs before it** (decision 2026-10-23; IMPL-08). Each line the candidate serves is held to the best run on it since the last run from the base: the parent, its parent and so on, down `parent_run_id`. The best run has the highest weighted field match on the line's real documents that both reports scored, with at least 5 such documents (`gating.LINE_MIN_DOCUMENTS`); on a tie, the nearer run. A paired bootstrap over those documents blocks the line (gate `line:<line>`) unless the lower bound of the 95% interval of (candidate − reference) is above −0.015 (`DEFAULT_DELTA`). A line with fewer shared documents, a new line, and a run with no line to compare get a warning only. A line the parent's release served that the candidate no longer serves blocks at `line:<line>`. No comparable parent report — missing, on another frozen set or scoring code, or without `per_document_lines` — blocks (gate `line_regression_evidence`); the remedy is the next cycle from the base (`--fresh`, without `--version`), or keeping the parent's report on the frozen set. A small line has a wide interval, so it may block until it has more documents; it can be waived as `line:<line>`. The usual comparison with the promoted release still applies. A run continued by hand from a directory (`training.train --continue-from`) names no parent and is blocked.

**A release must improve something, or name the defect it fixes.** With per-document scores, a release on which no primary metric (`field_normalized_match`, `list_field_recall`) improves (its interval's lower bound above zero) is blocked at `improvement`, unless `package` / `all` is given **`--fixes-defect "<ticket>"`** (`StageContext.fixes_defect`; one value; a blank or placeholder value is refused before any work). That stands in for an improvement and nothing else: floors and non-inferiority still apply, and the text is recorded in the gate decision. The first release is exempt.

There is no `--force`. There **is** an override (§15.5), requiring a named person, a written reason of at least 20 characters and the gates it waives, all recorded in the gate decision and the release bundle. It is given on `package` and `all` with `--override-approver`, `--override-reason` and `--override-gate` (repeatable): all three together, checked before any work, naming real gates. It lifts only gates that failed, and they are recorded as waived; one that waived nothing is a warning. v1 had none at all, on the reasoning that a waivable gate is a suggestion — which was right about the risk and wrong about the remedy, because the v1 gate demanded improvement on twelve metrics within 0.001 and could not be passed at pilot volume at all.

**Flags:** `--input`, `--corpus-version` (defaults to the cycle's version), `--version` (only to resume or rerun a cycle), `--gpu`, `--commit <sha>`, `--from-stage <name>` (resume; needs `--version`), `--skip-ingest`, `--fresh` (start from the base, not the promoted adapter), `--train-vit`, `--min-labels-per-type`, `--push-adapters` (see §3), `--tenant` (optional, defaults from env), `--dry-run`. `--scope`, `--doc-types` and `--out-version` are gone (decision 2026-10-22): the adapter fixes the types, and the cycle its version. `--foundation-only` is gone too: one adapter is the only topology.

**Flags are checked before the command detaches** on the pod. `--from-stage` names a stage of the command (`run.from_stage_problem`): `merge` and `quantize` are refused as "removed on 2026-10-23: nothing is merged or quantized; the release is its adapter". `--fresh` with a `--from-stage` after training is refused (`run.flag_problem`), and so is `--fresh --version vN` when vN already trained: its run status is `trained`, `evaluated`, `promoted` or `archived`, its artifacts are `published`, or the training completion marker is on the staging volume (`run.fresh_problem`). Either way nothing would train from the base: `--fresh` is for a new cycle.

## 3. The RunPod staging volume

> **Terminology, to prevent a real bug.** In this spec set "**registry**" means the durable run-manifest registry in Azure Blob (IMPL-02). The RunPod side is the "**staging volume**" and is never called a registry. They have different lifetimes and different guarantees.

RunPod **pods are ephemeral** — a pod clones the repo, works, pushes, and terminates (arch §14). Persistence between commands 1 and 2 therefore comes from a **RunPod network volume**, mounted at `/runpod-volume` and attached to every pod the controller launches.

```
$RUNPOD_VOLUME_MOUNT/staging/            # default /runpod-volume; /workspace on the pod (pod_bootstrap.sh)
  adapters/scope/policy/v{n}/            # the adapter: ms-swift output, v*-<timestamp>/checkpoint-* inside
    checkpoint_scores/                   # multi-GPU checkpoint selection's worker output
  parent-adapters/policy/v{n}/           # the adapter cycle v{n} continues (its parent's), copied from Blob
  train-data/policy/v{n}/                # the run's staged ms-swift JSONL
  train-images/{tenant}/{corpus}/        # page images, one cache per tenant and corpus version
  eval-reports/v{n}/scope/policy/
  serving-adapters/{release_id}/         # a serving cold start's release LoRA
```

The layout deliberately mirrors the Blob layout (master §4) so `package` copies rather than translates.

**Why stage instead of pushing straight to Blob:** staging was sized for the ~16 GB merged model, which is no longer built (decision 2026-10-23). Commands 1 and 2 both run on the pod: calibration and the gate read the staged checkpoints and the selected adapter there, and `package` publishes the adapter only once its gate has passed.

**Durability rule — the one thing command 1 always sends to Blob.** A network volume is working storage, not an artifact of record. If command 2 never runs, or the volume is reclaimed, an untracked training run has happened. So `finetune` **always writes the run manifest to Blob** (IMPL-02) even though the weights stay staged, with `artifacts.status: "staged"` and `artifacts.staging_path` set. `package` later flips it to `"published"` and fills the real Blob paths. The manifest is a few KB; the lineage guarantee in arch §12 is worth that.

**Optional belt-and-braces:** `finetune --push-adapters` also pushes the LoRA adapter to Blob at the end of checkpoint selection — `adapter_config.json` and `adapter_model.safetensors` of the selected checkpoint, never the checkpoint's resume state — so the adapter survives a reclaimed volume (`package --from-blob`, §4). Every push, by `--push-adapters` or by `package`, records the checkpoint it pushed in the checkpoint selection in Blob (`"pushed": <checkpoint>`). `package` later publishes the selected checkpoint's two files to the same prefix, skipping any file Blob already holds. Off by default, per the staging design; recommended if a cycle's commands 1 and 2 may be separated by more than a day.

**Config:** `RUNPOD_VOLUME_ID` and `RUNPOD_VOLUME_MOUNT` (default `/runpod-volume`) in `.env`. The controller attaches the volume to every pod it launches; a launch without it is refused rather than silently writing to pod-local disk.

## 4. `package` — command 2 (stages 7 → 9)

Runs: **calibrate → evaluation & gate → release bundle (with its routing check) → push the adapter to Azure Blob → update the run manifests → write the bundle → promote it, archiving the release it replaces.**

**Nothing is merged or quantized** (decision 2026-10-23): the `merge` and `quantize` stages are removed. The release **is** the adapter (`ReleaseBundle.served_as = "lora_on_base"`; older bundles say `merged`), served on the bf16 base, and calibration and the gate score the base with the selected adapter applied (`load_model("base", adapter=...)`) — what is served. `training/merge.py` and `postprocessing/quantize.py` are no longer called by the pipeline (IMPL-10).

**A released version is not packaged again.** Before any work, `package` — and `all` when it reaches package — refuses a version whose release is already `promoted` or `archived` for the tenant (`pipeline_dag.assert_not_released`), with the command to serve it again: `promote --version vN --by "<person>"`. Packaging it again re-ran its gate against the release live now: a pass promoted it with no named person, and a failure overwrote the decision a rollback needs. A `gated` release may be packaged again; a new model is a new cycle.

- Reads `--version v{n}` from the staging volume. If that version is not staged, **fail loudly** naming the expected path and the remediation (re-run `finetune --version <it> --from-stage training`, or `--from-blob` if the selected checkpoint was pushed, by `finetune --push-adapters` or an earlier `package`).
- **`--from-blob`** scores and releases the adapter already in Blob (`adapters/scope/policy/v{n}/`), with no upload: calibration and the gate load it from there (`pipeline_dag.serving_adapter`). It is refused unless both adapter files are in Blob and the checkpoint selection's `pushed` is the selected checkpoint (`_assert_in_blob`), so calibration, the gate and the release are all that one checkpoint. A dry run checks nothing.
- Serves **bf16 only**: `--formats`, `--skip-quantize` and `--dtype` are gone, and so are `formats` and `dtype` from the defaults in `orchestration/config/pipeline.yaml`. GGUF, and its `mmproj`, is an on-request edge export, never a `package` output (IMPL-10).
- Publishes to Blob (master §4), through `artifact_registry/transfer.py`:

| Artifact | Blob destination |
|---|---|
| The adapter — `adapter_config.json` + `adapter_model.safetensors` of the selected checkpoint only: the whole release | `adapters/scope/policy/v{n}/` |
| Eval report | `eval-reports/v{n}/scope/policy/` |

  Older releases keep their merged models and quantized formats under `merged-models/` and `quantized-models/`; nothing new is written there. The adapter push skips every file whose bytes Blob already holds (the MD5 recorded with each upload), so re-packaging a `gated` release sends nothing.
- Updates the run manifest: `artifacts.status` → `"published"`, the adapter's real Blob path.
- `--keep-staging` retains the volume copy (default: clear **this version's** staged paths after at least one manifest was published, so the volume doesn't fill).
- There is no quantize stage, so no quantization threshold validation runs (IMPL-10 / arch §13b).

**Flags:** **`--version vN`** (required), `--keep-staging`, `--from-blob`, `--fixes-defect "<ticket>"` (one value; also on `all`; §2), `--override-approver` / `--override-reason` / `--override-gate` (the written override, all three together; also on `all`; §2), and on a standalone `package` **`--tenant`** and `--corpus-version` (read from the run manifest when omitted). `--fixes-defect` and the override are checked in `main()` before the command detaches on the pod (`run.flag_problem`), so a bad flag is refused at once, not in a log.

**The release id is the version** (decision 2026-10-22): `releases/{tenant}/v3/` holds v3's calibrators, gate decisions and bundle (`StageContext.release_id` is the cycle's version). `--release-id`, `release_id_for` and `paths.next_release_id` are gone; `release-YYYY.M.N` is only still read, so an older index loads. One number names the run, the adapter and the release, so a `--from-stage` resume of `--version v3` can never split one release across two ids.

**Where the package stages get their inputs.** Nothing is hand-supplied in a real cycle:

- **Checkpoints** — after training, `discover_checkpoints` reads the `checkpoint-*` directories ms-swift wrote under the staged adapter directory (the most recent `v*-<timestamp>/` run), and the best-loss one from `trainer_state.json`. Selection (`stage_checkpoint_eval`) scores each as a decoder LoRA by generated field accuracy on a stratified sample of the adapter's own validation view (`validation_sample_rows`: 400) through `evaluation/validation_generation.py`, whose metrics are the gate's own `build_report`. The best `selection_document_finalists` (2) are then generated on every validation row in `selection_document_modes` (`[ocr_plus_image]`) and judged on whole documents — windows merged as serving merges them (decision 2026-10-24): a finalist more than `selection_guard_margin` (0.02) worse than the best finalist on printed values left empty, or on run-to-run agreement (the pass generated a second time in reverse order, `selection_run_to_run_check`, on), cannot win; of the rest, those within the margin of the best field match go on, and the one inventing the fewest values wins; with 0 or 1 finalists the near-tie rule (`selection_tie_break_margin`) decides. With several GPUs, one vLLM engine per GPU scores the candidates at once (`parallel_vllm_scorer`). Generation sends `validation_batch_rows` (96) rows per vLLM call. The stage result carries the selection and the generation cost (`generation`: rows and seconds); the choice is written to `eval-reports/v{n}/scope/policy/checkpoint_selection.json`.
- **Calibration samples** — when none are supplied, calibrate generates the validation split **with the model as served** (the base with the selected adapter applied; bf16 only), constrained as serving is, and labels every field's features correct or not against the golden label, split by the `val_half` the dataset build stamped. There is no shared or wildcard sample set. The stage records the expected calibration error per field type and overall. The calibrators record the checkpoint they were fitted on (`adapter_checkpoint`); calibrators of another checkpoint than the one selected now are refused ("Run package again from calibrate").
- **Gate decision** — written under `releases/{tenant}/v{n}/gate/bf16/`. The candidate metrics are the served model's — the bf16 base with the selected adapter — so that is bf16's gate run and no other format's. Besides the metrics, the verdicts (each with its basis: `paired_bootstrap`, `point_estimate`, or `floor_only` with no baseline or one on another yardstick), the failed and waived gates, the override and the tier, it records `paired_documents` (how many documents each bootstrapped metric paired), `baseline_not_comparable` and `baseline_metrics_not_compared`, `fixes_defect`, `eval_set` (`small_set_accepted`, `documents_by_doc_type`, `frozen_from_corpus`, `frozen_at`, and the count that decided the interim cap: `scored_documents_by_doc_type`, `scored_documents_counted_from`, `minimum_per_doc_type`, `small`), the `adapter` and its `doc_types`, the gate's `warnings` and `blocking_reasons`, and for a continued run `line_comparison` (IMPL-08: the runs compared, each line's verdict, and the lines not compared, dropped and added, or why no comparison could be made).

**The release bundle** (`releases/{tenant}/v{n}/bundle.json`, plus a row in `release_index.json`, kept sorted by version number) pins the adapter's name (`scope: policy`), the `lines` it trained on (`pipeline_dag.trained_lines`: the single lines of the run's train view, else the corpus's `policy_line_counts`; a line seen only inside a package was not trained) and its served `doc_types`, the adapter run and how it is served (`served_as: lora_on_base`, `serving_formats.bf16` the adapter's Blob directory, no merged model), per-format calibrators and gate decisions, the release `tier`, a hash of every prompt input (`common.prompts.prompt_hash`: templates, schemas, the section map), schema versions, the OCR pin, and hashes of `configs/shared/vision.yaml`, `configs/inference/vllm_serving.yaml` and the dependency lock (`requirements-train.lock`; `pyproject.toml` only when the lock is missing). It also records the p95 latency per adapter and the peak GPU memory from the eval report's `release_measurements`, and the **routing table**. Its status is **`promoted` only when every serving format has its own calibrator and its own gate run**; otherwise it is written as **`gated`** with the missing pieces listed in the stage result. Until a per-format gate run exists for quantized formats, an FP8 release is `gated`. A release with no trained line is refused (`build_release_bundle`; a dry run warns): an empty `lines` reads as every line at serving.

**Promotion and rollback** (`registry_utils/releases.py`; decision 2026-10-22). The bundle is written `gated` first; then `package` promotes a release whose gate passed (`promote_release`), and archives the tenant's previously promoted release: its index row and bundle status become `archived`. A dry run never promotes: it pushed placeholders, not weights, so its bundle stays `gated` (`write_release_bundle`). The run manifests mirror it (`promoted` / `archived`). So a tenant has one promoted release, the one serving reads (IMPL-11) and the next candidate is gated against. Nothing is deleted.
- `python -m orchestration.run promote --version vK --by "<person>" [--tenant T]` is a rollback: vK becomes the promoted release again and the current one is archived. Refused unless vK's bf16 gate decision passed, its bundle records the lines it serves, and serving would start with it: `promote_release` runs the cold-start prompt-hash, window-plan and base-model checks (`registry_utils.releases.serving_refusal`; a `lora_on_base` release whose adapter trained on another base than the configured one is refused), so a rollback must be promoted from the code it was packaged with, and that code served. Promoting the release already promoted changes nothing (its run keeps who promoted it and when). `--by` must name a person, not a system identity (checked before anything runs).
- `python -m orchestration.run versions [--tenant T]` prints the registry table: one row per version, with its run status, release status and tier (`registry_utils.releases.version_table`).
- `routing.release_pins` is gone from `configs/inference/vllm_serving.yaml`: a promotion replaces it. Cold start refuses a config that still holds it (IMPL-11).

**The bundle is built, and its routing check run, before anything is pushed** (`record_release_measurements`, `serving/routing_check.py`): every known line (the canonical policy schemas, `common.lob.known_lines`) is routed through the serving plan as it would stand with this release promoted. A line that cannot be routed, or one this release was trained for that would not reach it, stops the release with a `PipelineError` — no weights in Blob, no manifest marked published.

`--tenant` must match the `finetune` run that produced the version. `finetune` decides *which* models get built; `package` publishes their locations, so packaging with the wrong tenant publishes paths for artifacts that were never built — an empty Blob prefix that later reads as a published model. (Under `all` it comes from the finetune flag set; adding it twice is an argparse conflict.)

**`stage_push` publishes only runs that finished.** Manifests are selected by version *and status*: a run left at `training` or marked `failed` has no weights, and flipping it to `published` advertises a Blob path that serving will fetch and find empty.

## 5. `extract` — command 3 (arch §17)

A wrapper over `testing/run_extraction.py` (IMPL-12), which itself wraps the serving pipeline (IMPL-11) so test == prod. It reads documents already imported and OCR'd, from Blob.

**Model selection is the point of this command.** `--model` accepts:

| Value | Resolves to |
|---|---|
| `base` | Untuned `Qwen/Qwen3-VL-8B-Instruct` at the pinned revision, **no adapter** |
| `base --adapter <folder>` | The base with that LoRA (a training `checkpoint-N`, or a run's adapter folder; local or Blob) applied per request, **not merged** |
| `v1`, `v2`, … | The policy adapter's run of that version (`policy-v2`): the base with that version's adapter applied per request, staged or published — `registry_utils.query_registry.resolve_model_version` returns kind `base_plus_adapter` and `load_model` applies it as the default adapter; nothing is merged (decision 2026-10-23). A staged version is the checkpoint its selection chose, else the latest on the staging volume (with a warning), else it is refused. A version trained on another base than the configured one is refused. The HF backend runs it too (IMPL-07). For an older unified or Foundation run, its merged model plus a graduated per-type adapter where the routed type has one |
| `v2 --format fp8` | Refused for a version of the policy adapter (bf16 only); an older run's FP8 serving weights |

`--model base` is not a curiosity — it is the **zero-shot baseline** of the pilot protocol (IMPL-15) and the day-zero pre-annotation path (IMPL-04). `pilot/zero_shot_baseline.py` calls the same `testing.run_extraction.run_document` rather than a parallel implementation.

**Flags:** `--model`, `--adapter`, `--label`, `--source-ids …` | `--split train|val|test` (with `--corpus`), `--doc-type` (default `policy`), `--mode ocr_plus_image|image_only`, `--no-score`, `--format`, `--limit`, `--tenant`.

Outputs are those of IMPL-12: `results/{version}/` (each result with `all_fields`), and `metrics/{version}/` (per-document metrics and `summary.json`).

## 6. `all` — commands 1 + 2

Runs `finetune`, then `package`, sharing the cycle's version (the next one, unless `--version` names one to resume). Halts if the gate fails; nothing is published on a failed gate. Accepts the union of both flag sets; `--fixes-defect` and the override are checked before training starts. **Does not run `extract`.**

## 7. `orchestration/runpod_controller.py`

**Training — ephemeral pods (arch §14):**
- **Deployment: one RunPod H200 SXM pod runs every GPU stage in process**, inside tmux — OCR, training, checkpoint selection, calibration and the gate share the cards, each vLLM engine released before the next loads. On a multi-GPU pod training is data-parallel over every GPU (`distributed.gpus: auto`, the effective batch unchanged), checkpoint selection runs one vLLM engine per GPU, and OCR can be sharded (`run_mineru --shard I/N`). Classes per stage stay configurable (`orchestration/config/pipeline.yaml`) for a future per-stage launcher; the original plan put OCR on a cheaper L4/L40S card.
- **Stages 2, 4 and 5 all want a GPU**, so `finetune` provisions **one pod for the whole command** rather than shuttling a corpus between machines. That is cheaper than three pod launches and removes two Blob round trips.
- **Training code lives in the private git repo; the pod clones it fresh at job start** at a pinned commit — code is never permanently resident on a pod.
- Every launched pod has the staging volume attached (§3).
- Pod job: pull base model + corpus from Blob → run → write to the staging volume → **terminate**.
- Poll status, stream logs (**PII-scrubbed**, master §8), handle failures and retries.

**Serving — persistent endpoint (arch §14):**
- Manage the **RunPod Serverless vLLM endpoint** (IMPL-11): deploy/update to a promoted version, health-check, roll back.

**No separate controller CLI.** Single steps run through `orchestration.run <command> --from-stage <stage>`; the endpoint through `orchestration.run deploy-endpoint` / `rollback-endpoint`.

## 8. `orchestration/pipeline_dag.py`

The 11 stages (arch v2.1 §13 has 13; its merge and quantize stages were removed on 2026-10-23) as reusable, individually addressable stage functions — this is the implementation `run.py` composes, and the same functions an Airflow DAG or GitHub Actions workflow can schedule.

| # | Stage | Command | Writes to |
|---|---|---|---|
| 1 | **Ingestion** — checksum-deduped, immutable | `finetune` | `raw-documents/` (Blob) |
| 2 | **Preprocessing (GPU)** — MinerU OCR (3.4.5, pipeline backend, in the OCR environment) + page rendering at the resolution cap | `finetune` | `processed/` (Blob) |
| 3 | **Labeling** — human review | *(outside the CLI)* | `golden-labels/` (Blob) |
| 4 | **Dataset build** — compile JSONL, inject schema, modality split, train/val/test split | `finetune` | `corpus/v{n}/` (Blob) |
| 5 | **Training** — the ONE policy LoRA on the bf16 base, continuing the promoted adapter unless it starts from the base | `finetune` | staging volume |
| 6 | **Checkpoint eval** — vLLM generation on a validation sample, finalists judged on whole documents | `finetune` | staging volume + `checkpoint_selection.json` (Blob) |
| 7 | **Calibrate** — per-field-type calibrators and risk-controlled thresholds, on the base with the selected adapter (bf16) | `package` | release bundle |
| 8 | **Evaluation & gate** — frozen golden eval set, the base with the selected adapter; a continued run also held on each line to the best run since the last run from the base; production or interim tier | `package` | gate decision |
| 9 | **Package** — release bundle pinning adapter, prompt, schema, calibrators, OCR pin, serving config and the routing table; the adapter pushed (it is the release); a passing release promoted, the one it replaces archived | `package` | Blob |
| 10 | **Serving** — endpoint pulls the promoted release bundle | `deploy-endpoint` | endpoint |
| 11 | **Feedback loop** — low-confidence output, human corrections, and a random 5% of auto-accepted documents → labeling queue | `extract` + IMPL-04 | next corpus version |

**Pipeline properties that are requirements, not aspirations (arch §13):**
- Every stage reads from and writes to Azure Blob or the staging volume — never pod-local disk.
- Every training, evaluation, and quantization job produces a **run manifest** (IMPL-02).
- **Stages are idempotent and resumable.** `--from-stage` re-enters mid-pipeline; a completed stage re-run is a no-op, not a duplicate.
  - **Ingestion is the exception, and always runs.** There is no cheap correct completion check for it — comparing an input directory against ingested checksums costs the same as ingesting — and the approximate one that used to be here ("has anything ever been ingested?") was true from cycle two onward, so a second `finetune --input ./new_batch` skipped the stage and built the corpus from the previous batch. `ingest_directory` is checksum-deduped, so re-running it writes nothing and reports duplicates. An approximate guard on cheap idempotent work buys nothing and can be wrong in the direction that loses documents.
- **The gate (stage 8) is a hard stop.** Promotion requires every gating metric to pass. There is no `--force-deploy`; the only exception is the recorded written override (§2), given with `--override-approver`, `--override-reason` and `--override-gate`. `package` reads the recorded gate decision and refuses a version whose gate did not pass — running `package` after a blocked gate used to publish it.
- **Every run says where its time went.** Each stage that runs records its `wall_seconds` (every attempt included) in its result, and the run report is written after every stage (*Current implementation*).

## 9. `orchestration/config/`
- GPU class per stage, corpus/version tags, staging volume id and mount, schedule, notification hooks, retry policy (`retry.max_attempts: 2`, `backoff_seconds: 30` × the attempt number).
- **Sweep orchestration (arch §11a) — deferred with IMPL-06 `sweep.py`:** when enabled, the 3-phase sweep runs as a scheduled job sequence (~9–12 runs before the first **production** run), each writing a manifest, winner promoted to the production training config. Not part of the first cycle.
- **Foundation-upgrade cascade (arch §12):** when a new major version of the adapter passes its gate, generate the re-validation work list from `query_registry.adapters_depending_on()` (IMPL-02) and schedule retraining of every dependent GRADUATED adapter (§4.2) **before** it becomes production. Dormant until a type graduates — the default topology produces no dependents at all. The DAG produces the work list and waits for it to pass rather than silently promoting.

## Constraints
- Four lifecycle subcommands, one entrypoint. `all` = `finetune` + `package`, never `extract`.
- Training pods **ephemeral**; the staging volume is the only thing that persists between them, and it is not an artifact of record.
- `finetune` always writes its run manifest to Blob even when weights stay staged.
- Orchestration runs on cheap CPU infra **outside** RunPod; only GPU-bound work runs on RunPod (arch §14).
- Secrets from env only. Logs PII-scrubbed before leaving the pod.
- No flag around the evaluation gate; only a recorded, attributed override (§2), whose three flags build that record and nothing else.

## Acceptance checklist
- [ ] `run finetune` executes stages 1→6 on a tiny fixture set and leaves the adapter on the staging volume; no stage merges or quantizes.
- [ ] `finetune` reports the unlabeled backlog and trains on the labeled subset; it aborts when labels are below `--min-labels-per-type`.
- [ ] `finetune` launches **one** training job: the policy adapter, continuing the tenant's promoted release, or from the base with `--fresh` or when the promoted one cannot be continued; the reason is logged and recorded (`fresh_reason`), the parent as `parent_run_id`, and the parent's adapter is staged under `parent-adapters/`. A new cycle takes the next version (one more than any run, or any tenant's corpus or release, uses); `--from-stage` without `--version` is refused.
- [ ] The dataset build refuses a corpus that set aside more than 25% of its documents, before writing anything — also when preflight ran with `--skip-corpus`.
- [ ] **A failed gate stops `package` before anything is published, exits non-zero, and prints per-metric verdicts (floor, interval and basis);** `package` run on its own refuses a version whose recorded gate did not pass.
- [ ] With per-document scores in both eval reports, non-inferiority is a paired bootstrap and the decision records `paired_documents`; a release that improves no primary metric is blocked at `improvement` unless `--fixes-defect` names the defect, which the decision records. A baseline on another yardstick (frozen set, tenant, scoring code, or unrecorded) is judged on the floors alone; reports on one yardstick that cannot be paired (no shared documents, no per-document scores) are compared on point estimates; either way the reason is in the decision's warnings. `--fixes-defect` is one value, checked before the command detaches.
- [ ] A release scored on fewer than 150 real frozen policies is at most `interim`, with a warning; the decision records the `eval_set` and the count it was judged on.
- [ ] The gate's baseline is the tenant's promoted release in its release index. A passing `package` promotes its release and archives the one it replaces (index, bundle, run manifest); `promote --version vK --by <person>` rolls back, refused unless vK's gate passed, its bundle records lines and serving would start with it; `versions` prints the registry table.
- [ ] `package` refuses a version whose release is already promoted or archived, before any work; a release with no trained line is refused; a dry run never promotes; the gate scores only the frozen policies of the lines the release serves.
- [ ] Ingestion refuses an intake holding `acord/` or `lossrun/` but no `policy/` folder; with a `policy/` folder the other type folders are logged as not ingested.
- [ ] `--override-approver`, `--override-reason` and `--override-gate` build a validated written override only when all three are given, naming real gates; the decision and the bundle record it as a waiver.
- [ ] Training and packaging refuse a corpus built under another review rule, or holding policies with a `window_plan_version` that is not `policy_windows.window_plan_id()`; the release bundle records the window plan.
- [ ] The corpus build refuses a gold with no review record (a twin's is its seed's); preflight names every such label (FAIL) and reports annotation agreement (WARN while no minimum is set).
- [ ] `freeze-eval-set --mark-reviewed` copies review records into an existing frozen set only where the frozen gold is unchanged, changing nothing else, and exits 1 while any frozen gold still has none.
- [ ] `finetune` writes a run manifest to Blob with `artifacts.status: "staged"` even though weights are not pushed.
- [ ] `run package --version v2` pushes the **real** selected adapter (its `adapter_config.json` and `adapter_model.safetensors` only) — the whole release, `served_as: lora_on_base` — to its Blob path, then flips the manifest to `"published"`. `package` takes no `--formats`, `--skip-quantize` or `--dtype`.
- [ ] A release whose routing check fails pushes nothing.
- [ ] `package` fails loudly with remediation when the requested version is not on the staging volume.
- [ ] `package --from-blob` releases the adapter in Blob with no upload, refused unless both adapter files are there and the recorded push is the selected checkpoint; `finetune --push-adapters` pushes the selected checkpoint at the end of checkpoint selection, and every push records it.
- [ ] `run extract --model base` runs the untuned base model with no adapter; `--model base --adapter <checkpoint>` applies it per request; `--model v2` loads the base with the policy adapter's v2 adapter applied (`policy-v2`, not merged; a non-bf16 `--format` refused), or an older run's merged model plus a graduated per-type adapter where the routed type has one. A version trained on another base is refused, and a staged one loads its selected checkpoint, else its latest with a warning.
- [ ] `run all` chains 1 and 2, **never runs extraction**, and does not reach `package` when the gate fails.
- [ ] `--from-stage` resumes mid-pipeline; re-running a completed stage is a no-op — **except ingestion**, which always runs and relies on checksum dedup.
- [ ] `--from-stage merge` or `quantize` is refused before the command detaches, saying the stage was removed; `--fresh` with a `--from-stage` after training, or with `--version` of a version that already trained, is refused before the detach.
- [ ] A completion check covers **every** target it claims to, not just the first: a stage that failed halfway must not read as complete on resume.
- [ ] The staging volume is cleared only after at least one manifest was actually published.
- [ ] A pod launched without the staging volume attached is refused.
- [ ] Each stage that runs records `wall_seconds`, and the run report is rewritten after every stage.
- [ ] **A continued run is held line by line to the runs before it.** The gate stage walks the chain from the parent (`pipeline_dag.line_evidence`) and holds each served line to the best run on it since the last run from the base, by a paired bootstrap on at least 5 shared real documents; a line whose lower bound is not above −0.015, or a dropped line, blocks at `line:<line>`; no comparable parent report — or a run continued by hand from a directory — blocks at `line_regression_evidence`; a thin or new line only warns; the decision records `line_comparison`. `StageContext.cross_type_evidence`, the old cross-type rule's input, is gone; `revalidation_evidence` still holds the arch §12 cascade's per-doc-type booleans.
- [ ] **A deterministic failure is not retried.** The retry policy exists for transient faults — a throttled Blob read, a pod that dropped. `PipelineError`, `PathError` and the gate's own refusals (`GoldenEvalError`, `EvalError`, `FreezeError`, `GateError`) mean the stage cannot run at all with these inputs (below the day-zero label floor, missing `--input`, a malformed `--version`, a checkpoint scorer that cannot start), so a second attempt re-lists everything and fails with the identical message after a pointless backoff. They stop the run immediately, like a gate block.
- [ ] **A dry-run `rollback-endpoint` does not move the endpoint.** Popping the deployment history before checking `dry_run` meant a preview permanently rewrote the recorded live version: `health_check` reported the previous version while the current one was still serving, and the next real rollback refused with "nothing to roll back to".
- [ ] A pod leaves no persistent process after completion.
- [ ] `deploy-endpoint --model vN` updates the serving endpoint; `rollback-endpoint` restores the previous promoted version.
- [ ] *(deferred)* The sweep sequence runs Phase 1 → 2 and promotes the winning config. (Quantization thresholds no longer gate `package`: nothing is quantized since 2026-10-23.)
- [ ] A Foundation version bump produces the dependent-adapter re-validation work list and blocks promotion until it passes.

---

## Current implementation (2026-10-21)

**Commands** (`orchestration/run.py`): `finetune`, `package`, `all`, `extract`, `promote`, `versions` (§4),
`deploy-endpoint`, `rollback-endpoint`, and **`freeze-eval-set --corpus vN [--tenant T] [--allow-small]`**, or **`freeze-eval-set --tenant T --mark-held-out`**
on an already frozen set (marks the documents the delivery calls unseen carriers', after a re-import, and records
their carriers by line; nothing else changes), or **`freeze-eval-set --tenant T --mark-reviewed`** (copies each
frozen document's review record from its label metadata, which the gate needs, only where the frozen gold is
still the current one; nothing else changes; exit 1 while any frozen gold still has none). The two mark flags
run together when both are given, and either is refused with `--corpus` or `--allow-small`
(IMPL-08): the first copies that
tenant's corpus test split into `golden-eval-set/{tenant}/` — each tenant freezes its own, from its own corpus,
and is gated on it. It refuses fewer than 150 real test documents per type (twins do not count) unless `--allow-small` (recorded as
`small_set_accepted` in the set's manifest; the gate caps at interim a release scored on fewer than 150 real
frozen policies), refuses a second
freeze, refuses a test split holding a gold with no review record, and refuses while a set frozen before sets were per tenant still sits at the root. After a freeze the
tenant's corpus builds leave the frozen documents and their families out and split into train and val only.

**One adapter** (`common/adapter.py`; decision 2026-10-22). There are no training scopes: `configs/scopes.yaml`
and `common/scopes.py` are gone, `--scope` is gone from every CLI (`orchestration.run`,
`orchestration.preflight`, `orchestration.smoke_run`, `evaluation.run_eval`, `scripts/diagnose_generation.py`,
`training.train`), and `gpu_class_by_scope` is gone from `orchestration/config/pipeline.yaml`. The code for ACORD
and Loss Run documents is kept and not trained. The CGL source documents train the same adapter under a tenant of
their own (`--tenant`). The adapter trains on, is scored on and serves **single-line policies of a known line
only** (`common.lob.single_line`): a package and a policy with no line are left out of its view
(`training.corpus_view._filter`) and of the gate, and serving sends them to the base model against
`_fallback.json`. `pipeline_dag.trained_lines` counts single-line keys only: a line seen only inside a package
was not trained. Training refuses a view whose single-line policies mix lines on the common model (today the
seven personal lines and `gl`) with lines on self-contained schema files (`common.adapter.assert_one_output_shape`
on `corpus_view.trained_lobs`); the audit (before import) and the preflight check the same earlier.
Run ids (`policy-vN`) are not tenant-prefixed, so every tenant's cycles share one version sequence
(`registry_utils.releases.used_versions` reads every run of the adapter and every tenant's corpora and
releases); the run report is named per tenant and version.

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
bash scripts/pod_run.sh start cgl-vN -- bash -c '
  /workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy --all-unprocessed --tenant T &&
  python -m data_pipeline.ocr_check --doc-type policy --tenant T &&
  python -m orchestration.preflight --tenant T --corpus-version vN --version vN --skip-corpus &&
  python -m orchestration.run finetune --skip-ingest --tenant T --version vN &&
  python -m orchestration.run freeze-eval-set --corpus vN --tenant T --allow-small'
```

`vN` is a version no run, corpus or release of any tenant has used yet (the one a new `finetune` would take), named here
because the freeze needs it.

`preflight --skip-corpus` because `finetune` builds the same corpus and refuses the same >25% set-aside itself, so
the corpus is built once.

**Stage behaviour** (`orchestration/pipeline_dag.py::run_stages`):
- **Logging is re-asserted before every stage** (`common/logging_setup.py`): `orchestration.run` installs a
  `ConsoleHandler` — a `StreamHandler` subclass, which ms-swift's and modelscope's logger setup leave alone — and
  `reassert()` lowers whatever a library raised and re-enables this repository's loggers.
- **Each stage that runs records `wall_seconds`** (every attempt included) in its `StageResult` data; a stage
  skipped as already complete records none.
- **Ingestion reads one folder per document type** (`<input>/policy/`), or a folder of one type read flat
  (`stage_ingestion`). An intake holding `acord/` or `lossrun/` but no `policy/` is refused: read flat, every
  PDF under it would be stored as a policy in the write-once raw layer. With a `policy/` folder, the other type
  folders are logged as not ingested.
- **A progress report is written after every stage** — and on a block or failure — to `$FIDEON_LOG_DIR`
  (default `/workspace/logs`) when that folder exists, as `run-<command>-<tenant>-<version>.json`
  (`progress_report_name`), so a run that dies part-way still says what each finished stage took. A report that
  cannot be written never stops the run.
- **Retries**: a failed stage is retried up to `retry.max_attempts` (2) after `backoff_seconds` × attempt. Never
  a gate block; never a `PipelineError`, `PathError`, `GoldenEvalError`, `EvalError` (`EvalSetLeakage`
  included), `FreezeError` or `GateError` (the run stops at once); never the deterministic training
  errors (`TrainingError`, `StagingError`, `LengthCheckError`, `CorpusViewError`, `CalledProcessError`). A
  checkpoint scorer that cannot start (a split or setting it cannot use) raises `PipelineError` and is not retried.
- `checkpoint_eval`'s result includes the generation cost (`generation`: rows and seconds).
- Training raises when a real run leaves no checkpoint; `checkpoint_eval` rediscovers checkpoints on a
  resumed run; calibration, the gate and the push read the selection from Blob when resuming
  (`selected_checkpoint`).
- Training first decides whether to continue (`continuation`): it continues the promoted release, copying its
  adapter to `parent-adapters/policy/vN` on the volume, or starts from the base, and its result records
  `continued` and `fresh_reason`.
- Engines are released between stages (checkpoint selection → calibration → golden eval), so each finds the
  card free.
- `package` refuses a version already promoted or archived (`assert_not_released`, a precondition checked
  before any stage), requires a passed recorded gate, finds calibrators in Blob on resume, builds the bundle
  and its routing check before pushing, and uploads real weights through `artifact_registry/transfer.py` (IMPL-10):
  adapter files only, identical files skipped by MD5.
- Training and packaging read the corpus manifest and refuse a corpus this code would not build today
  (`pipeline_dag._assert_corpus_pins`): one built under another review rule (`review_rule_version`, so a
  reused corpus cannot train golds the dataset build never checked), and one holding policies whose
  `window_plan_version` is not `policy_windows.window_plan_id()` — the prompt hash is computed at packaging,
  so it cannot catch an adapter trained on another window plan. The release bundle records the window plan,
  and serving checks it at cold start (IMPL-11).
- `--push-adapters` uploads the selected checkpoint's `adapter_config.json` and `adapter_model.safetensors`
  at the end of `checkpoint_eval` (non-dry runs), not a placeholder and not the checkpoint's resume state.
  Every push (`--push-adapters` or `package`) records `"pushed": <checkpoint>` in the checkpoint selection in
  Blob, which `package --from-blob` checks (`_assert_in_blob`).
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
  `--override-*` record as `override`, and `small_eval_set` from the real frozen policies the candidate was
  scored on (`eval_set_size`). For a continued run it also passes `continued_from`, `line_evidence` and
  `line_evidence_missing` (`line_evidence`, from the candidate's eval report and those of the runs it
  continued, back to the last run from the base).
  The golden eval it runs gets `lines=served_lines(ctx)` and `adapter_checkpoint` (`eval_report_metrics`).
  It takes no scope; the not-applicable metrics come from
  `common.adapter.not_applicable_metrics()` and the floors are `PILOT_FLOORS` (IMPL-08).

**Preflight** (`python -m orchestration.preflight [--tenant T] [--corpus-version vN] [--version vN] [--skip-corpus]`; no `--scope`, and `--version` replaces `--out-version`): PASS/WARN/FAIL per
check, exit 1 on any FAIL, report in `/workspace/logs/preflight.json`; detached on the pod. Checks: every visible
GPU at least 90% free and the training GPU count dividing the batch; packages (torch, transformers 4.57, vLLM,
flash-attn, ms-swift, peft, deepspeed; `swift` next to the Python); the base model verified at the pinned revision;
config invariants; 100 GB free; Azure write, overwrite, read and delete; every labelled document OCR'd and at
least `--min-labels-per-type`; **reviewed labels** (`check_reviewed_labels`: FAIL naming each label with no
review record, the labels the corpus build would refuse, with the first one's reason); **output shape**
(`check_output_shape`: FAIL when the tenant's imported single-line policies mix lines on the common model with
self-contained lines, which training would refuse after staging, and every later cycle of the tenant too);
**label agreement** (`check_label_agreement`: how many documents are double-labelled
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
by default; never the real data's) and version (`v0`), training the policy adapter as a real cycle does
(`--batch cgl-v1 --tenant smoke-cgl` for the CGL batch; no `--scope`), with `--min-labels-per-type 1`. Its preflight gets `--skip-corpus` only when
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

## Changes of 2026-10-22 (one adapter)

- One adapter, one version per cycle (decision 2026-10-22; Fideon SPEC_09's adapter groups knowingly not followed): no `--scope`, `--doc-types`, `--out-version` or `--release-id`, and no `run_multi_scope`; a fresh cycle takes the next version, `--version` resumes or reruns one; the release id is the version; `package` promotes a release that passed and archives the one it replaces, `promote` rolls back, `versions` prints the registry; the gate's baseline is the promoted release in the index; `--fixes-defect` and the override are one value; the routing check routes every known line, with no `release_pins`; `extract --model vN` finds the policy adapter's runs; run report, preflight, smoke run and the CGL chain without scopes. New acceptance items.
- Code review of 2026-10-23: one version sequence for every tenant (every tenant's corpora and releases count, §2); single-line policies only, in the view, the gate and serving (*Current implementation*); the gate scores the lines the release serves, and a baseline scored on other lines, uncalibrated, or of another model is not compared (§2); a released version is not packaged again (`assert_not_released`), a release with no trained line is refused, a dry run never promotes, and `promote` refuses a release serving would refuse (§4); ingestion refuses an intake of other types with no `policy/` folder; the preflight's `output shape` check. New acceptance items.

## Changes of 2026-10-23 (incremental training)

- Incremental training (the user's decision of 2026-10-23), §1–§3: a cycle continues the tenant's promoted release (`continuation`), or starts from the base with `--fresh` or a recorded reason (`fresh_reason`); the parent is `parent_run_id`, its adapter staged under `parent-adapters/policy/vN`; what the operator should know (memorisation, wrong labels carried forward, the chain of parents, when to start fresh).
- No merge and no quantize (§1, §2, §4, §5, §7, §8): 11 stages — `finetune` 1–6, `package` 7–9 (calibrate, gate, push), serving 10, feedback 11; the release is the adapter (`served_as: lora_on_base`), calibrated and gated as served; `--formats`, `--skip-quantize`, `--dtype` and the `formats`/`dtype` defaults gone; `extract --model vN` is the base plus that version's adapter.
- The gate holds a continued run line by line to the runs before it (`line:<line>`, `line_regression_evidence`), replacing the cross-type evidence item (§2, acceptance, *Current implementation*).
- Code review of the incremental training: each line is held to the best run on it since the last run from the base, by a paired bootstrap on at least 5 real documents, and a dropped line blocks (§2, §8, acceptance); the decision records `line_comparison` and `blocking_reasons` (§4); an eval report or calibrators of another checkpoint are not reused (§2, §4); `continuation` also starts from the base on missing or unreadable adapter files, other saved LoRA settings or a missing bundle, and continues a release with an old-style name (§2); `--push-adapters` pushes the selected checkpoint at the end of checkpoint selection, every push records `pushed`, and `package --from-blob` releases the adapter in Blob (§3, §4); `promote` and `extract --model vN` refuse an adapter trained on another base, and a staged `vN` is its selected checkpoint (§4, §5); `--from-stage` and `--fresh` are checked before the detach (§2). New acceptance items.

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.

## Changes of 2026-10-24 (one served MinerU)

- Preprocessing reads with the served MinerU when `$MINERU_URL` is set (`ServedMinerUEngine.from_environment`, IMPL-03): it then runs in any environment, and both the stage and its done-check count a document another MinerU read as still to read. Without it, the in-process MinerU of the OCR environment, as before.

## Changes of 2026-10-24 (printed values only and consistency)

- Checkpoint selection (the user's decisions of 2026-10-24: nothing invented, consistent output): guards on printed values left empty and on run-to-run agreement, then the finalists within the guard margin on field match, then the fewest invented values; the finalists' pass is read twice for the agreement guard when there are 2 or more finalists (`selection_run_to_run_check` in `_selection_settings`, on unless the config says false) (§4).
- The gate's eval reads each document's main mode a second time for `run_to_run_agreement` (`evaluation.run_to_run_check` in `configs/training/unified.yaml`, on) (§2).
- The scoring code changed, so a cycle continuing a release scored before 2026-10-24 starts from the base (another yardstick); none is promoted yet (§2).

## Changes of 2026-10-24 (QA fixes)

- `plan_corpus` asks `continuation()` and passes the parent chain's training documents to the split (IMPL-05); `stage_training` starts from the base when a reused corpus validates on them.
- `orchestration.run extract` refuses `--split` without `--corpus`, and `--adapter` with a model other than `base`, before the detach. `scripts/pod_run.sh`: a value set on the caller's command line beats `.env` for every setting `.env`/`.env.example` names, detached runs included (written to an owner-only `<name>.env` beside the log, sourced after `.env`, then deleted; a value equal to the pod's own is not the caller's), and `bash scripts/pod_run.sh env` prints the environment a run would get.

## Changes of 2026-10-24 (architecture v2.7)

- Header: aligned with `finetuning-architecture-v2.7.docx`, which records the decisions of 2026-10-23 and 2026-10-24: printed values only, consistent output, fields first, one MinerU, production on Azure.

## Changes of 2026-10-26 (pipeline review fixes)

- `label-noise --version vN [--corpus-version V] [--limit N]`: the release reads the train documents; values it reads that the page prints but the gold lacks are queued for a person (`golden-labels/<tenant>/_review_queue/`), never written into a gold. The `active_learning` CLI is wired.
- `--ocr-workers N`, `--allow-dirty`; stage times logged to MLflow (one run per run id) when configured; webhooks (HTTPS) on a stage failure, a gate block and a finished run (`orchestration/notify.py`).
- Fresh-run rules (off by default): go fresh when new documents exceed a share of train, every N cycles, or when a gold the chain trained on changed (`gold_sha256` in the corpus manifest); the reason goes in `fresh_reason`.
- The OCR pin is checked at corpus build, preflight and promotion (IMPL-03). Image staging is cached by content.

## Changes of 2026-10-26 (gate and release decisions)

- Stage 9 writes the release `gated` and archives nothing (#44); the run summary prints the next steps. `shadow --version vN --model-url URL --mineru-url URL [--documents 50]` (detached on the pod) reads about 50 frozen documents, lines in turn, real first, through the Azure worker's two phases with the candidate and the promoted release, scores both with build_report and writes `releases/{tenant}/{vN}/shadow/shadow_report.json`. Pass: served field match no more than 0.02 below the promoted release's, invented values no more than 0.005 above, no failures, schema validity not lower; with nothing promoted, no failures and schema validity >= 0.98.
- `promote --version vN --by NAME [--accept-interim] [--skip-shadow --reason ...]`: serving and OCR-pin refusals first; a rollback to a release promoted before needs nothing else; an interim release needs `--accept-interim`; a gated release needs a passing shadow report for the bundle as packaged, against the release promoted now, or a written reason (>= 20 characters). `ReleaseBundle.promotions` records each one.
- The gate's baseline goes through `promoted_baseline_report`: a promoted report on another yardstick only because of scoring code or settings is re-scored and compared (`baseline_rescored`). `continuation` re-scores a parent that differs only in yardstick and continues (`parent_report_rescored`); the per-line references are re-scored the same way.
