# SPEC 01 — Project Scaffold, Configs, Schemas, Prompts

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: none. This spec creates the repo skeleton and all static configuration that later modules read.
>
> **Architecture refs:** `finetuning-architecture-v2.6.docx` §0a (schema contract), §0b (LoB), §6 (modality instructions), §7 (dataset format, schema registry, prompt template), §9/§9a (LoRA targets), §11 (full hyperparameter spec), §11a (sweeps), §13a/§13b (quantization), §19 (repo structure).

## Goal

Scaffold the `insurance-extraction-finetuning` repo and create every config file, JSON schema, and prompt template that the rest of the system depends on.

## Deliverables

### 1. Repo root
- `README.md` — project overview, setup steps, how to run each pipeline stage (stub with section headings + the module map from master §5). State prominently that this repo implements the **L3 VLM layer** of the Fideon pipeline and that its output schema is owned by **Fideon SPEC_00** (master §1.1).
- `pyproject.toml` — Python 3.11+. **The only place versions are declared**, in dependency groups: `[data]`, `[ocr]` (MinerU 3.4, pipeline backend), `[train]`, `[serve]`, `[quantize]`, `[eval]`, `[track]`, `[dev]`. See *Current implementation* below for the pinned versions and why each group is separate.
- `requirements.txt` and `requirements-{ocr,train,serve,quantize}.txt` — one per machine, each only `-e .[groups]`: they choose groups, never versions. Installed on a pod by `scripts/setup_pod.sh <role>`, constrained by `requirements-{ocr,train,serve,quantize}.lock`.
- `.env.example` — every required env var with placeholder: a Blob login — `AZURE_STORAGE_ACCOUNT_URL` + `AZURE_STORAGE_SAS_TOKEN` (preferred), or `AZURE_STORAGE_CONNECTION_STRING` — then `AZURE_BLOB_CONTAINER`, `AZURE_RAW_CONTAINER` (separately-permissioned raw container: raw documents, intake batches, exports — master §4, §8), `AZURE_RAW_SAS_TOKEN` (the raw container's own SAS token; left empty, the one token serves both containers and must be an account SAS), `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_VOLUME_ID`, `RUNPOD_VOLUME_MOUNT` (default `/runpod-volume`, the staging volume — master §12a), `HF_TOKEN`, `HF_MODEL_REVISION`, `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `MLFLOW_TRACKING_URI` / `WANDB_API_KEY` (optional), `DEFAULT_TENANT_ID` (the default tenant), `FIDEON_BASE_MODEL_DIR` (overrides the local base-model directory), `FIDEON_ALLOW_OFF_POD` (train on a non-RunPod GPU host), `FIDEON_NO_DETACH` (run one command in the foreground on the pod), `FIDEON_INTAKE_DIR`, `FIDEON_WORKSPACE`, and MinerU's `MINERU_TOOLS_CONFIG_JSON` (its `mineru.json`), `MINERU_DEVICE_MODE` (cuda) and `MINERU_MODEL_SOURCE` (local: the pinned weights). Comment each. `tests/test_repo_contracts.py` fails when the code reads a variable the template does not list, or the template lists one nothing reads (library-owned and runtime-set names excepted).
- `.gitignore` — Python caches, `.env`, model weights (`*.safetensors`, `*.gguf`, `*.bin`, `*.ckpt`), `*.pdf` (except under `tests/fixtures/`), `testing/{results,metrics,ocr_cache,test_data}/`, `calibration/calibration_store/`, `pilot/reports/`, the client data folders under `data/` (deliveries, bundles, CGL source data, audit reports, exclusion and override CSVs), `configs/field_types.proposed.yaml`, and `Documentation/`.

### 2. `configs/`
- **`base_model.yaml`** — `model_id: Qwen/Qwen3-VL-8B-Instruct`, `revision: <pin>`, **`local_dir: /workspace/models`** (where the pod keeps the weights; every loader reads them from there), quantization block (**`load_in_4bit: false` by default — the base is held in bf16**; the `bnb_4bit_*` keys stay populated so flipping the flag on a VRAM-constrained pod needs no other edit, and are read only when it is true, per arch §9.2), `attn_implementation: flash_attention_2`, resolution cap (`max_image_long_side_px: 1792`, valid range 1536–2048 per arch §11), `train_vit: false`, `max_seq_len` (comment: **set from the 95th-percentile token count measured on the real corpus**, not guessed; the per-task caps that training and serving actually use are in `configs/shared/sequence.yaml`).
- **`configs/shared/vision.yaml`** (per-task `max_pixels`/`min_pixels`, whole 32×32 visual tokens) and **`configs/shared/sequence.yaml`** (per-task `max_seq_len` and reserved `max_output_tokens`: `policy_schedule` 12,288, `lossrun_rows` 8,192, `policy_declarations` 4,096 …) — one file each, read by dataset build, training and serving.
- **`configs/training/unified.yaml`** (was `foundation.yaml` under v1's two-level design) — the full parameter set from arch §11, at the values the code reads now:
  - LoRA: rank 64, alpha 128, dropout 0.10 (0.05 at ≥200 documents per type), `bias: none`, the seven decoder projections of master §2; `include_vision_projector: false` (mergers not targeted); `use_rslora: false` (on for the rank sweep only).
  - Optimization: LR `1e-4`, cosine schedule, `warmup_steps: 10` (steps, not a ratio, so pilot volume still warms up), 3 epochs (document passes), `optim: adamw_torch`, β₁ 0.9, β₂ 0.999, ε 1e-8, weight decay 0.0, max grad norm 1.0.
  - Continued training (decision 2026-10-23): `continued_training: {num_train_epochs: null, learning_rate: null}`. A cycle that continues the promoted adapter uses its non-null values instead of `optimization`'s (`training.train.optimization_for`); null trains as a fresh run. Set them from a measured continued run, not by guess.
  - Batch/memory: per-device train batch 1, grad accumulation 8, **effective batch 8** at pilot volume (kept when training data-parallel on several GPUs: accumulation shrinks), gradient checkpointing on, bf16, `dataset_num_proc: 8`; `use_logits_to_keep`, `padding_free`, and `length_grouped_sampling: false` (with one row per device and `padding_free` there is no padding to save, and on it made ms-swift encode every row before step 1).
  - Vision: `train_vit: false` (the §3 gate default), resolution cap inherited from `base_model.yaml`.
  - Eval/checkpointing: eval and save every 50 steps, `save_total_limit: 4` (the candidates checkpoint selection scores), a resume point every 70 steps (`checkpoint_every_steps`), `load_best_model_at_end: false`; validation loss tracked (`metric_for_best_model: eval_loss`, `greater_is_better: false`) for the progress curve only — `early_stopping_patience` is in the file but not passed, since ms-swift 3 takes none. The validation cut: `evaluations_per_run: 10`, each in-training loss check on `loss_sample_rows: 160`; checkpoint selection on `validation_sample_rows: 400` in all modes, then `selection_document_finalists: 2` judged on whole validation documents in `selection_document_modes: [ocr_plus_image]` with `selection_guard_margin: 0.02`, and a near-tie (`selection_tie_break_margin: 0.015`) re-scored. `selection_run_to_run_check: true` reads the finalists' whole documents a second time in reverse order, and `run_to_run_check: true` repeats the golden eval's main mode the same way (run-to-run agreement, decision 2026-10-24, IMPL-08). What ships is chosen by generated accuracy (IMPL-06 §3), never by loss. Fixed recorded `seed: 42`, `logging_steps: 10`.
  - Corpus view: `line_balance` (`min_documents: 100`, `max_repeat: 3`) and `render_rotation: true` (one render per twin per epoch), applied in the adapter's view of the corpus.
  - Data mix: `data_mix` with `synthetic_fraction` and `scanned_share`, a default and optional per-line values under `lines:`, all `null` (unsteered) until set; the file may also set its own `modality_mix`. It is the one adapter's training config (`common.adapter.ADAPTER.training_config`).
  - Distributed: `deepspeed_config: null`, `sequence_parallel_size: 1`, `gpus: auto` (data-parallel on every GPU the pod has).
  - Top comment: "hyperparameters are sweep starting points (arch §11/§11a) — tune on the validation set, not fixed truth."
- **No scope or layout-family files** (decision 2026-10-22): `configs/training/personal_lines.yaml`, `configs/scopes.yaml` and `configs/layout_families.yaml` are gone. What a training run covers is fixed in code (`common/adapter.py`, §5), not configured; there is no LOB → layout-family table (Fideon SPEC_09's adapter groups are knowingly not followed). No task list holds `page_select`: every page is read (decision 2026-10-19).
- **`configs/schema_sections.yaml`** — the policy section groups, page rules and identifying keys the window planner reads in training and serving; a default map (decl, arrays, lineblk, dtd; `forms` declared empty) for self-contained lines and a `common_model` profile (decl, arrays, forms, lineblk; each table's id field and prefix, reference fields, `array_keys`, unit keys, `run_overlap_pages: 1`); `pages.declarations_leading: 5` for every line. Every group reads every page; the page rule says only how its first window is formed: `every` (arrays, forms, lineblk — pages packed into windows by runs) or `declarations` (decl, dtd — pages 1–5 and the detected declarations page read together first, the rest as for `every`).
- **`configs/model_view.yaml`** — repo-owned replacements for common-model descriptions written for schema authors, and how coverage codes are offered (`oneof` or `enum`); part of the prompt.
- **`configs/lob_meanings.yaml`** — one role-based sentence per registered line, for the line classifier (no carrier wording).
- **`configs/label_mappings.yaml`** — labels written for one line's schema used under another's (applied when targets are built and labels scored, never to stored labels).
- **`configs/coverage_code_recodes.yaml`** — coverage codes re-coded when bundles are prepared; **`configs/coverage_code_meanings.yaml`** — a template of neutral meanings per coverage code, to be filled in; not yet applied to the model view or the prompt.
- **`configs/canonical schema/`** — the client's schemas, read-only (§3b).
- **Not created under v2.1:** `configs/training/{acord,lossrun,policy}_adapter.yaml`. There is one adapter and no per-type adapter (arch §4.1); a per-type config arrives only with a type that passes the §4.2 graduation gate. The original v1 description, for reference — LoRA rank 16 / alpha 32 / dropout 0.05, LR `7e-5` (range 5e-5–1e-4), epochs 4 (range 3–5), **effective batch 16–32**; each references its `doc_type`, its schema file, and `foundation_version: <set at runtime>`.
  - **One shared `acord` adapter only** (arch §4b Option 1). Per-form adapter configs (`acord25`/`acord125`/`acord140`) are **not created** — with limited data per form, one adapter learning shared "ACORD-ness" generalizes better than several data-starved ones. Add them only when a form has 1000+ examples *and* eval shows the shared adapter underperforming. The per-form **schemas** still exist, because the classifier must select the right schema regardless.
- **`configs/sweeps/`** *(arch §11a — **author the configs now, run them after the IMPL-15 pilot**)* — the bounded 3-phase sweep definitions, as W&B Sweep / MLflow configs. A 9–12 run sweep against 25–30 docs/type mostly measures noise; arch §11a scopes it to "before the first **production** run", not before the pilot:
  - `phase1_lr.yaml` — Foundation `{5e-5, 1e-4, 2e-4}`, per-type `{2e-5, 5e-5, 1e-4}`; 1 epoch per candidate; metric = validation loss; 3 runs per adapter type.
  - `phase2_epochs.yaml` — Foundation `{2, 3, 4}`, per-type `{3, 4, 5}` with early stopping patience 2; metric = validation field-level F1; 3 runs per adapter type.
  - `phase3_rank.yaml` — Foundation rank `{32, 64, 128}`; **secondary, not run by default** — only if F1 plateaus.
  - Comment the total budget: **~9–12 training runs before the first production run**, and that **every sweep run writes a full run manifest** (IMPL-02) so sweeps are first-class registry entries, not untracked side experiments.
- **`configs/deepspeed/zero2.json`** and **`zero3.json`** — ZeRO-2 default, ZeRO-3 for VRAM-constrained multi-GPU.
- **`configs/inference/vllm_serving.yaml`** — resolution cap (**must match `base_model.yaml`**, `common.config.assert_resolution_parity`), `max_seq_len: 32768` (at least the largest per-task cap); `generation`: greedy (`temperature: 0.0`), `max_new_tokens: 12288` (every task's full reservation), `logprobs: true` with `logprobs_mode: raw_logprobs`, `structured_outputs: true`, `repeatable: false`, and `validation_batch_rows: 96` (rows per vLLM call when a whole split is generated, longest expected answer first); `serving`: backend, served model tag (resolved from the registry), `enable_lora`, `max_loras`, `max_lora_rank`; `routing`: `classifier_confidence_threshold: 0.70`, `detect_lob: false` with `lob_confidence_threshold: 0.90` and `lob_uncertain_threshold: 0.60` (no `release_pins`: a rollback is `promote --version vK`, IMPL-13; cold start refuses a config still holding `release_pins` or the removed `lob_family_threshold`, `vllm_entrypoint.assert_no_retired_settings`); **review confidence threshold (`confidence.review_threshold`, default 0.7, arch §5)**; `printed_values.mode`: `strict` (a value no page prints is taken out of the served JSON) or `flag` (kept and flagged), strict by default (decision 2026-10-24, IMPL-11), any other value refused. Adapter paths are **not** listed here — they are resolved through the registry and the promoted release. There is no page threshold anywhere: every page is read, in windows planned by the function the corpus build uses (`policy_windows.plan_windows`), and `long_documents` holds only `conflict_resolution: declarations_page_wins`.

### 3. `schemas/` — JSON Schemas (Draft 2020-12)

**These are not invented schemas — they are the Fideon SPEC_00 Pydantic models serialised to JSON Schema** (master §1.1). Generate them from the Fideon models where available; hand-authored versions are a temporary stand-in and must be reconciled before the first real corpus build.

- `acord25.schema.json` — Certificate of Liability: insured, insurer(s), policy numbers, effective/expiration dates, coverage limits (GL, auto, umbrella), certificate holder.
- `acord125.schema.json` — Commercial Application: applicant, business info, premises, coverages requested.
- `acord140.schema.json` — Property section (per arch §4b two-level ACORD classification).
- `lossrun.schema.json` — carrier, policy number, valuation date, and a **`claims` array** (repeating rows: claim number, loss date, status, paid, reserved, total incurred, description). This is the key list-field type and the one where a **missed row** is both easy and expensive (arch §5).
- `policy_doc.schema.json` — the legacy flat policy schema (named insured, policy number, effective/expiration, coverage schedule, premium, endorsements). Kept with its example, but **no policy is extracted into it**: the registry key `policy` resolves to the client's canonical `_fallback.json` (§3b).
- `common_fields.json` — shared definitions `$ref`'d by the flat schemas.

**Every flat schema must additionally include:**
- **`line_of_business`** (arch v2.1 §0b) — a **list** of enum values (`$ref: lob.enum.json#/$defs/line_of_business`, `uniqueItems`, default `[]`), required in every flat schema; an empty list means the document determines no line. The enum lives once, in `schemas/lob.enum.json` (12 values: five commercial lines and seven personal lines), and lines outside it go to the list `line_of_business_other`. Canonical policy schemas carry no `line_of_business`: a policy's line comes with the request (master §1.3).
- **A `description` on every field** — see the semantic-gloss rules below (master §1.4). This is not documentation; it is injected into the prompt and is part of what the model is trained on.
- Explicit `null` unions for optional fields — "absent" must be representable as `null`, never as a missing key or a hallucinated value.
- Date fields as `string` with an `MM/DD/YYYY` format note — the one output date format (`common.normalize.OUTPUT_DATE_LABEL`), applied to every output by `common.canonical.with_output_dates`.
- `required` vs optional marked deliberately.

**Field descriptions — semantic glosses with exclusions (master §1.4)**

The same field appears under many surface labels across documents (*Insured Name*, *Named Insured*, *Applicant*, *Name of Applicant*, …). The model has to map all of them onto one canonical key. Its main aid is the field's `description`, so write these as **definitions with explicit exclusions**, never as label lists:

```json
"insured_name": {
  "type": ["string", "null"],
  "description": "The party purchasing the insurance policy. Not the certificate
                  holder, the producer/agency, or an additional insured."
}
```

The exclusion clause is the part doing the real work — it teaches the **boundary** between confusable entities, which insurance documents are dense with (certificate holder, producer, additional insured, loss payee, carrier). A definition generalises to phrasings nobody listed; a list of aliases does not, and gives the model a lexical prior that makes confusable errors *worse*.

Rules for writing them:
- Define what the field **is**, in the document's own domain terms.
- Add an exclusion clause naming the confusable entities it must not be.
- One or two lines. These ride on every training row and every production request.
- **Never enumerate surface labels here.** Those live in the alias registry below, which never enters the prompt.

### 3a. `schemas/aliases/{doc_type}.aliases.json` — the alias registry

A versioned sidecar per document type, recording the surface forms actually observed and the entities that must not be confused with them:

```json
{
  "insured_name": {
    "aliases": ["Insured Name", "Named Insured", "Applicant", "Name of Applicant",
                "Applicant Name", "Name Insured", "Insured"],
    "confusables": ["Certificate Holder", "Producer", "Agency",
                    "Additional Insured", "Loss Payee", "Mortgagee", "Carrier"]
  },
  "policy_number": {
    "aliases": ["Policy Number", "Policy No.", "Policy #", "Certificate Number"],
    "confusables": ["Quote Number", "Claim Number", "Binder Number"]
  }
}
```

**Three consumers, and one hard prohibition:**

| Consumer | Use |
|---|---|
| IMPL-04 | Annotator rulebook — so three reviewers apply one consistent mapping, and a label naming a confusable is rejected |
| IMPL-05 | Alias coverage counting in the corpus manifest |
| IMPL-08 | Per-alias eval slicing and the confusable-misattribution metric |

**It is never rendered into the prompt and never consulted at inference** (master §1.4 anti-pattern). Adding a newly observed surface form is a registry edit, **not** a schema change — it triggers no corpus rebuild. That asymmetry is deliberate: the registry can grow freely as annotators encounter new phrasings, while the glosses stay stable.

`schemas/aliases/policy.aliases.json` is built from the canonical schemas by `scripts/derive_aliases_from_canonical.py`, one entry per canonical field, reading each common-model line through its bundle. The SPEC_21 schemas also carry the labels carriers print; those do not reach a rendered prompt either, beyond what a field's own description says (a test checks every common-model prompt, whole and per window). Instead `scripts/check_alias_coverage.py` counts, per common-model line, how many training documents print each recorded label and lists those none does, and evaluation splits field accuracy into labels training showed and labels it never did (`evaluation/metrics/unseen_labels.py`, reported, not gated).

Each schema must be loadable by `jsonschema` and validate a correct example. Include one valid example JSON per schema under `schemas/examples/{type}.example.json`, each exercising a populated `line_of_business` and at least one `null` field.

**Schema versioning:** each schema declares a version and the registry exposes it (`common.schemas.schema_version`): a top-level `version` in `schemas/` (e.g. `lossrun` 1.1.0), `fideon:source.version` in a self-contained canonical file (e.g. `property` 1.5.0), and for a common-model line the bundle version `<overlay>+common.<common model>` (e.g. `homeowners` 1.0.0+common.1.1.0, `gl` 3.0.0+common.1.1.0). Master §7 — a schema change forces a corpus rebuild and a new training cycle.

### 3b. `configs/canonical schema/` — the client's policy schemas (read-only)

Every policy is extracted into the client's canonical JSON for its line. These files are the client's: never edited here; what this repository adds lives in separate files (`configs/model_view.yaml`, `configs/schema_sections.yaml`, `configs/label_mappings.yaml`).

- **`LOB Schema/*.json`** — 34 files. 31 lines are registered as `policy:<line>` (`common.schemas._sources`), plus `_fallback.json` as the key `policy` (a policy with no single line, or a package). `_common.json` and `classic_auto.json` are not registered (classic auto is personal auto). With the three ACORD forms and the Loss Run, the registry holds **36 schema keys**. Request line names reach a file through `LOB_SCHEMA_ALIASES` (`general_liability`/`cgl` → `gl`, `workers_comp` → `wc`, `commercial_auto` → `auto`) and `common.lob.MERGED_LINES`.
- **The Fideon SPEC_21 common model** — `common schema/common_model.json` (1.1.0). Nine line files are thin overlays that `$ref` it block by block: the seven personal lines, `gl` and `_fallback`. The registry **bundles** each overlay at load (`common.schemas._bundle`): references into the common model are made local and the definitions they reach copied in, so the validator, the slicer, the prompt and the every-key fill read one self-contained schema. A bundle is refused when it references any other file, was written against another common-model version, or names coverage codes its `<line>.coverage_codes.yaml` does not. Training refuses a corpus whose policies mix lines on the common model with lines on self-contained files (`common.adapter.assert_one_output_shape`): one adapter would learn two output shapes. Each document is judged by the schema it is read against; a package, or a policy with no line, is read against `_fallback.json`, on the common model. `common.schemas.is_common_model` is the one switch every difference keys off.
- **The model view** (`common/model_view.py`) is what the prompt carries and the decoder enforces for a common-model line, while validation keeps judging the whole bundle: every printed value `raw`/`parsed`/`page_ref` (`raw` never null); no field the pipeline fills (document type, modality, page count, file name, `is_package`, a coverage's id); no full-text tier or provenance; each code field held to the line's coverage codes with their meanings; keywords xgrammar refuses dropped; author-facing descriptions replaced from `configs/model_view.yaml`.
- Each common-model line (the personal lines and `gl`) also ships `<line>.coverage_codes.yaml` (a prompt input), `<line>.crosswalk.yaml` and `<line>.fields.md`; `common schema/lob_schema_map.yaml` maps the 48 L1 codes to schemas and families.

### 4. `prompts/`
- **`system_prompt_template.jinja`** — rendered with `doc_type`, `acord_form`, `modality_mode`, `schema_json`, the output outline, the window's required keys, and two switches decided by the schema registry: `canonical` (a policy's `{raw, parsed, page_ref}` envelopes, absent fields omitted) and `common_model` (a SPEC_21 line). It includes `prompts/doc_types/{acord,lossrun,policy}.jinja`, each type's own structure and failure modes. Must:
  - State the doc type and inject the **exact schema from the registry** — never hand-typed per example (arch §7). For a policy window it is that window's slice of the schema (`sections`); on a common-model line, the model view.
  - Give extraction rules: valid JSON only, no markdown fences, absent fields → `null` (omitted in the canonical shape), the output date format rendered from `OUTPUT_DATE_LABEL` (`MM/DD/YYYY`), how to handle repeating table rows. On a common-model line also: each value written once, its `page_ref` listing every page that prints it; identifiers, references, codes and types as plain strings; rows only from the pages given; `page_ref` the only source reference; and, for the window holding the additional fields, which top-level sections other windows read.
  - Include the **evidence block for the modality**, explicitly declared rather than silently omitted (arch §6):
    - `ocr_plus_image`: OCR text and page images of one document; the OCR text is the primary source for dense text, identifiers and amounts, the images are the authority on layout, and where the two disagree on a character the image wins.
    - `noisy_ocr_image`: identical to `ocr_plus_image` (`common.constants.PROMPT_MODE_ALIASES`) — **the noise lives in the data, not the instruction**.
    - `image_only`: page images and no OCR text; read the values from the pixels, and output null rather than a partly guessed identifier or amount.
  - Be versioned by `common.prompts.PROMPT_TEMPLATE_VERSION` (now **7.2.0**), a constant recorded in the corpus manifest (IMPL-05), not text in the prompt.
- **`doc_type_classifier_prompt.jinja`** — zero-shot prompt asking the model to name the doc type (`acord`/`policy`/`lossrun`) and, if acord, the form number — the **two-level classification** of arch §4b — and, when lines are passed, a policy's line of business from the registered lines, each with its meaning from `configs/lob_meanings.yaml`. Output a small fixed JSON: `{"doc_type": "...", "acord_form": "...", "lob": "..." | null, "confidence": 0.0}` (without `lob` when no lines are passed). A `line_only` variant asks the line alone, among the lines passed. Line detection ships off (`routing.detect_lob: false`).

**Non-negotiable:** `common.prompts.render_system_prompt(...)` is the single renderer used by dataset build (IMPL-05), evaluation (IMPL-08), serving (IMPL-11), and testing (IMPL-12); `render_classifier_prompt(...)` renders the classifier. Training-time and inference-time prompts must render identically (arch §7).

- The template renders each field's **`description` alongside its type**, so the semantic glosses (§3) reach the model. No structural change is needed — descriptions ride along with the already-injected schema.
- The template **must not** reference the alias registry (master §1.4).
- **`common.prompts.prompt_hash()`** is a SHA-256 over `prompt_input_files()`: every template (`prompts/**/*.jinja`), every schema in `schemas/` and `configs/canonical schema/LOB Schema/`, the common model, the coverage-code files, `configs/model_view.yaml`, `configs/schema_sections.yaml` and `configs/lob_meanings.yaml`. Serving refuses at cold start a release packaged with a different hash. Since `layout_families.yaml` left the hash (2026-10-22) the window-plan pin has changed, so the corpus must be rebuilt (the new training set needs that anyway). The lines whose policy prompts are snapshotted are the seven personal lines (`common.schemas.SNAPSHOT_LINES`, was `CANONICAL_FAMILY`).
- `testing/prompts/*.prompt.txt` (45 files) are reference renderings, regenerated by `python -m testing.render_prompts` and checked in CI with `--check`; they include every window prompt a common-model line sends. The self-contained lines', ACORD and Loss Run prompts are frozen as fingerprints in `tests/fixtures/old_style_prompt_fingerprints.json`.

### 5. `common/` (shared library — create this package)
- `common/config.py` — YAML plus environment, with the cross-file invariants enforced: `load_yaml(path)`, `env(name)`, one loader per file (`base_model_config`, `serving_config`, `training_config(name)` which merges an `extends:` chain), the shared budgets (`pixel_budget`, `seq_cap_for_task`, `answer_cap`), and the checks (`assert_resolution_parity`, `assert_task_budgets_are_coherent`, `assert_effective_batch`, `assert_model_revision_pinned`, `validate_all`).
- `common/schemas.py` — the schema registry: `schema_key(doc_type, acord_form, lob, sections)` selects a schema (a policy by line, the fallback otherwise), loads and bundles canonical overlays, validates (`validate`, `iter_validation_errors`), slices a schema per window, lists required fields, exposes `schema_version`, `is_canonical`, `is_common_model`, `with_page_bounds` (a window's `page_ref` held to its own pages).
- `common/prompts.py` — `render_system_prompt(doc_type, modality_mode, acord_form, lob, sections)`, `render_classifier_prompt(lines, line_only)`, `PROMPT_TEMPLATE_VERSION`, `prompt_input_files()` / `prompt_hash()`.
- `common/ids.py` — `source_id` helpers: build/parse `{doc_type}_{index}`, validate format.
- `common/lob.py` *(new)* — the LOB enum, validation, the list form (`normalize_lob`), merged lines (`MERGED_LINES`, `merge_line`: classic auto → personal auto, watercraft → ocean marine), a per-value coverage counter used by IMPL-05 and IMPL-08, and a policy's lines (`lob_lines`, `known_lines` — the canonical policy schemas, `common_model_package`; `single_line`, the one line a policy is of, `None` for a package or a policy with no line — the adapter trains on, is scored on and serves only those; `parse_lines` and `within_lines` for the data tools' `--lines`, which compare schema names).
- `common/canonical.py` — the client's canonical `FieldValue` JSON and the shapes around it: the training target (`training_target`, `to_model_target`), the system fields the pipeline fills (`with_system_fields`), the every-key fill (`with_all_keys`), output dates (`with_output_dates`).
- `common/model_view.py` — a common-model line's schema as the model sees it (§3b), with `configs/model_view.yaml`.
- `common/structural_ids.py` — structural ids on a common-model document: unit keys, references resolved to them, renumbering in printed order, and `comparable_view` (what scoring compares instead of ids).
- `common/grounding.py` — whether a value is printed, and on which pages: `ground`, one matcher for the label rules, the audit and the OCR-agreement feature; and `ground_tolerant`, the same search forgiving OCR's misreads of a scan, for the printed-values check at serving and the hallucination metric (decision 2026-10-24).
- `common/schema_sections.py` — which canonical sections one extraction call asks for (`configs/schema_sections.yaml`), `pages_for(group, pages)` (every page, for every group), `leading_pages(group, pages, declarations_page=)` (the pages a declarations group reads together in its first window; empty for the others), and `run_overlap` (pages two windows of a split run share).
- `common/adapter.py` — the one adapter (`ADAPTER`: name and lineage `policy`, doc types `("policy",)`, tasks classify, extract and the three policy window tasks, training config `unified`; `run_id("v3")` is `policy-v3`), `not_applicable_metrics()` (the gate metrics nothing in its corpus can produce) and `assert_one_output_shape`. It replaces `common/scopes.py` (decision 2026-10-22). `common/tasks.py` — the `Task` enum (no `PAGE_SELECT`) and `CORPUS_TASKS` (the tasks the corpus builds rows for: `extract` and the three policy window tasks).
- `common/label_mapping.py` — `configs/label_mappings.yaml` applied when targets are built and labels scored.
- `common/gpu.py` — model compute never falls back to the CPU; `common/run_ids.py` — the one run-id shape.
- `common/logging_setup.py` — the run's logging: `setup()` installs a `ConsoleHandler` (a `StreamHandler` subclass that ms-swift's and modelscope's logger setup leave alone), and `reassert()` restores our levels, handlers and loggers before every stage.
- `common/normalize.py` *(moved here from IMPL-08)* — per-field-type normalizers so comparison reflects correctness rather than formatting: dates (`01/01/2026` == `2026-01-01`), currency (`$1,200.00` == `1200.0`), entity names (`Acme Mfg LLC` ≈ `ACME MANUFACTURING LLC`), policy/claim number separators. **Built here rather than in IMPL-08 because three specs need it and the earliest is IMPL-04**: alias derivation (IMPL-04), the promotion gate (IMPL-08), and the testing routine (IMPL-12). Pure and dependency-free. One definition of "matches" shared by all three — divergence silently changes what "correct" means between them.
- `common/aliases.py` *(new)* — loads `schemas/aliases/{doc_type}.aliases.json`; `aliases_for(doc_type, field)`, `confusables_for(doc_type, field)`, `canonical_for(doc_type, surface_label)` (reverse lookup, **for labeling and eval only**), `is_confusable(doc_type, field, label)`. Mirrors the `common/lob.py` pattern. Its module docstring must state that no serving or inference path may import it.
- `common/constants.py` — `ACTIVE_DOC_TYPES = ["acord", "policy", "lossrun"]`, **`UNCLASSIFIED = "unclassified"`** (the holding bucket for documents whose type is not yet known — it lives here, not in `data_pipeline`, because `common/ids.py` and `artifact_registry/paths.py` both have to honour it and both sit below `data_pipeline`), the **`doc_type` → Fideon SPEC_00 canonical model mapping table (master §1.2)** (`DOC_TYPE_TO_CANONICAL`), ACORD form list, modality modes + mix ratios (`MODALITY_MIX`, `PROMPT_MODE_ALIASES`), resolution cap default, split-ratio table by volume (`SPLIT_RATIOS_BY_VOLUME`), confidence review threshold, and `MIN_INTER_ANNOTATOR_AGREEMENT` (`None`: no spec states a minimum, so the preflight warns on agreement and blocks only once it is set).

## Constraints
- No business logic yet — scaffold + static assets + tiny shared helpers.
- Everything imports cleanly (`python -c "import common"`).
- **The doc_type ↔ canonical-model mapping exists in exactly one place** (`common/constants.py`).
- The LOB enum exists in exactly one place (`schemas/lob.enum.json`, surfaced by `common/lob.py`).
- Follow all global conventions in master §7.

## Acceptance checklist
- [ ] Repo tree matches master §5 (empty dirs get a `.gitkeep`).
- [ ] `pip install -e .[dev]` succeeds (or documents which heavy deps are RunPod-only).
- [ ] All schemas load and validate their example JSON via `jsonschema`.
- [ ] **Every flat schema requires `line_of_business` as a list of enum values** (an empty list valid, meaning undetermined), and rejects an out-of-enum value.
- [ ] `common.prompts.render_system_prompt(...)` produces a correct system prompt for each of the 3 modality modes, and `noisy_ocr_image` renders **identically** to `ocr_plus_image`.
- [ ] Every common-model overlay loads as one self-contained bundle; a bundle referencing another file, or written against another common-model version, is refused (`tests/test_schema_resolver.py`).
- [ ] The self-contained lines', ACORD and Loss Run prompts render exactly as frozen (`tests/test_old_style_prompts_frozen.py`), and `python -m testing.render_prompts --check` passes.
- [ ] **Every field in every schema has a non-empty `description`**, and each description for a confusable-prone field carries an exclusion clause.
- [ ] The rendered prompt **contains** the field descriptions and **contains no alias strings**.
- [ ] `schemas/aliases/{doc_type}.aliases.json` loads for every active doc type; no string appears in both `aliases` and `confusables` for the same field.
- [ ] `common.aliases.canonical_for("policy", "Applicant")` returns `insured_name`; `is_confusable("policy", "insured_name", "Certificate Holder")` is true.
- [ ] `common.constants` resolves `lossrun → loss_run/LossRunDocument` and the other two active types.
- [ ] `configs/training/unified.yaml` contains every parameter in arch §11's full specification tables (not just the summary table).
- [ ] `configs/sweeps/` defines the 3 phases with the documented candidate sets and budgets (authored, not executed this cycle).
- [ ] `configs/inference/vllm_serving.yaml` resolution cap equals `configs/base_model.yaml` — assert this in a test.
- [ ] `.env.example` lists every env var used anywhere in the design.
- [ ] `ruff` and `mypy` pass on `common/`.

---

## Current implementation (2026-10-21)

**Dependencies** (`pyproject.toml`; resolved for Linux with `uv` — the resolver found three failures that
would have broken the pods, fixed below):

| Group | Holds | Why separate |
|---|---|---|
| `data` | azure-storage-blob, pypdf, pillow `>=11,<12`, PyMuPDF `>=1.24,<1.25` | Blob and page rendering, everywhere |
| `ocr` | `mineru[pipeline]>=3.4.5,<3.5`, `six>=1.16`, torch `>=2.8,<2.9`, torchvision `>=0.23,<0.24` | MinerU 3.4's pipeline backend (layout, PP-OCRv6 OCR, tables; no VLM); the code imports `mineru` and records its version on every page. `six` because MinerU 3.4.5's OCR imports it without declaring it |
| `train` | torch `>=2.8,<2.9`, transformers `>=4.57,<4.58`, peft, bitsandbytes, accelerate, deepspeed, ms-swift `>=3.9,<4`, qwen-vl-utils | torch pinned to the minor vLLM 0.11 pins: the training pod also runs vLLM |
| `serve` | vLLM `==0.11.0`, transformers `>=4.57,<4.58` | exact, and the same on both pods: calibration is fitted on the training pod's vLLM |
| `quantize` | `llmcompressor>=0.8` (its PyPI name), torch `>=2.8,<2.9` | needs datasets `>=4` (ms-swift 3 needs `<4`) and a transformers range vLLM 0.11 excludes: its own environment |
| `eval`, `track`, `dev` | numpy/scikit-learn/openpyxl, mlflow, pytest/pytest-cov/ruff/mypy/packaging/pyarrow | |

flash-attn is in no group: it must be built against the installed torch with `--no-build-isolation`,
which `scripts/setup_pod.sh train` does as a second step. `tests/test_dependencies.py` fails when the code
imports a package no group installs (MinerU was exactly that), or a requirements file names an unknown group.

**Pod roles** (`scripts/setup_pod.sh ocr|train|serve|quantize|dev`): installs in order, builds flash-attn,
installs tmux, checks every expected package and that torch sees CUDA. On the pod it runs itself in tmux.

**Two environments, not one.** MinerU's dependency set (its own runtimes for layout, OCR and tables) is kept
apart from the training stack, so `ocr` and `train` are separate environments: `scripts/pod_bootstrap.sh`
builds `/workspace/venv` (train, which includes serve) and `/workspace/venv-ocr` (ocr) on the volume and
records the requirements and lock each was built from, so a re-run reinstalls only what changed. MinerU's
pipeline weights are a separate, pinned download (`scripts/download_mineru_models.sh`: PDF-Extract-Kit-1.0 at a
fixed revision into `HF_HOME`, and `mineru.json` written to point at them); OCR refuses to run without that
config, off the GPU, with a remote model source, or with a MinerU 1.x (`magic-pdf.json`) config
(`data_pipeline.ocr.mineru_config`).

**Lock files** (`requirements-{train,ocr,serve,quantize}.lock`, from `scripts/lock_requirements.sh` with uv, for
Linux x86-64 / Python 3.11): `setup_pod.sh` installs each role with `-c <lock>`, so every package is pinned.
Resolving them found what ranges alone hid: the serving environment took **transformers 5.x** (vLLM 0.11.0 sets
no upper bound) and the OCR and quantize environments took the newest torch, built for a CUDA newer than the
pod's. Now `transformers>=4.57,<4.58` in train and serve, `torch>=2.8,<2.9` in every environment, `pillow<12`
and `PyMuPDF<1.25` everywhere (one renderer and one resizer for the OCR environment's training pages and for
serving), and the serve lock is resolved inside the train lock so serving runs exactly what calibration saw.
`flash-attn<3`. The OCR role installs `libgl1`/`libglib2.0-0` (OpenCV). The release bundle's `lockfile_hash` is
the train lock's. `tests/test_dependency_locks.py` fails on a stale lock or drift between environments, and
keeps `six` in the OCR lock. Locked today: ms-swift 3.12.6, transformers 4.57.6, torch 2.8.0, vLLM 0.11.0
(train and serve); mineru 3.4.5 with the same torch, transformers 4.57.6, pillow 11.3.0 and PyMuPDF 1.24.14
(ocr); llmcompressor 0.9 (quantize).

**Base model revision** pinned to `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (Hub commit of 2025-10-15, four
safetensors shards, 17.5 GB); the bootstrap downloads exactly that into `local_dir`.

**`.env.example`**: the Azure connection string is quoted — the file is sourced by bash on the pod, which
would cut an unquoted value at its first `;`. A SAS login (`AZURE_STORAGE_ACCOUNT_URL` +
`AZURE_STORAGE_SAS_TOKEN`) is preferred and, when both are set, used instead: the connection string carries the
account key, which reaches everything and never expires. The token is quoted too (bash reads each `&` as "run in
the background"). `AZURE_RAW_SAS_TOKEN` is the raw container's own token: a container SAS reaches one
container, so without it the main token must be an account SAS, which reaches the raw PDFs from every pod that
holds it. There is no external pre-annotation setting: frontier models are never used for gold,
synthetic or training data (decision 2026-10-19). `HF_HOME` is documented (on the pod,
`/workspace/.cache/huggingface`); `MINERU_TOOLS_CONFIG_JSON` names MinerU 3.x's `mineru.json` and may be an
absolute path (the bootstrap sets `/workspace/mineru.json`, replacing a MinerU 1.x `magic-pdf.json`);
`MINERU_DEVICE_MODE=cuda` and `MINERU_MODEL_SOURCE=local` are set by the bootstrap and applied before MinerU
is imported.

**Vision budget** (`configs/shared/vision.yaml`): budgets are whole 32×32 visual tokens — `max_pixels`
`2483200` (was `2483712`, a Qwen2.5 28×28 figure that made `MAX_PIXELS` and `IMAGE_MAX_TOKEN_NUM` name
different budgets). `common.config.pixel_budget()` is the one source for the trainer's env and the vLLM
engine's `mm_processor_kwargs`, and refuses a non-multiple.

**Section map** (`configs/schema_sections.yaml`): the policy section groups (decl, arrays, forms, lineblk, dtd),
their page rules and identifying keys, read by the window planner in training and serving. Common-model lines
read the `common_model` profile instead (decl: the policy-level objects, `lob_parts`, premium, billing; arrays:
every unit table and everything that refers to a unit, plus `additional_fields`; forms: the forms list, a job of
its own since 2026-10-23; lineblk: the remainder, today general liability's claims-made terms and
countersignature, and nothing on the other lines), with each table's id field and prefix, the reference fields,
identifying keys, unit keys and `run_overlap_pages: 1`. The self-contained lines keep their forms in `arrays`, so
`forms` is empty on them. Every group reads every page (decision 2026-10-19). The page rules are `every`
(arrays, forms, lineblk) and `declarations` (decl, dtd): `pages.declarations_leading: 5` makes the declarations
group's first window the first five pages plus a declarations page found later, read together, and every page
after them follows in windows.

**Schema 2 configs** (all prompt or target inputs, never the client's files): `configs/model_view.yaml`
(description replacements; a test fails once the client's own text matches an entry),
`configs/lob_meanings.yaml` (a test keeps recorded carrier labels out and requires every registered line),
`configs/label_mappings.yaml`, `configs/coverage_code_recodes.yaml` (27 codes re-coded at bundling), and
`configs/coverage_code_meanings.yaml` (the accuracy plan's template of neutral code meanings, to be filled in;
not yet read by the model view).

## Changes since 2026-10-05

- Added the alignment line.
- §1: `[ocr]` is MinerU 3.4 (pipeline backend); requirements install through locks; `.env.example` list completed (MinerU 3.x vars, HF vars, intake/workspace, external pre-annotation endpoint); `.gitignore` list matches the file (data folders, `Documentation/`).
- §2: `unified.yaml` values corrected to the file (dropout 0.10, LR 1e-4, `warmup_steps`, `adamw_torch`, weight decay 0.0, effective batch 8, mergers not targeted) and the time-cut keys added (`length_grouped_sampling: false`, `loss_sample_rows: 160`, `selection_document_finalists: 2`, `selection_document_modes: [ocr_plus_image]`, `selection_guard_margin`); added `shared/`, `personal_lines.yaml`, `scopes.yaml` (incl. `casualty_fleet`), `layout_families.yaml`, `schema_sections.yaml`, `model_view.yaml`, `lob_meanings.yaml`, `label_mappings.yaml`, the coverage-code files; `vllm_serving.yaml` described as it is (no adapter map, `validation_batch_rows: 96`, `detect_lob: false`, LoB thresholds, release pins).
- §3/§3a/§3b: `line_of_business` is a list over the 12-value enum, flat schemas only; output dates `MM/DD/YYYY`; `policy_doc.schema.json` is legacy (key `policy` → `_fallback.json`); schema versions as the registry reads them; new §3b on the canonical schemas, the SPEC_21 common model 1.1.0 overlays and bundling, partly migrated families, and the model view; alias derivation and coverage checks for SPEC_21 lines.
- §4: the template's real switches (`canonical`, `common_model`), common-model rules, evidence blocks as rendered, `PROMPT_TEMPLATE_VERSION` 7.2.0, classifier `lob` output, renderer names, prompt hash inputs, reference snapshots and frozen fingerprints.
- §5: `common/` described as built (config loaders, registry functions, `canonical`, `model_view`, `structural_ids`, `grounding`, `schema_sections`, `scopes`/`tasks`, `label_mapping`, `gpu`, `run_ids`, `logging_setup`).
- Current implementation: MinerU 3.4.5 dependencies, pinned weights and config refusals, locked versions, the `common_model` section profile and the Schema 2 config files.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx`.
- §1 `.env.example`: the SAS login (`AZURE_STORAGE_ACCOUNT_URL`, `AZURE_STORAGE_SAS_TOKEN`) beside the connection string; `ALLOW_EXTERNAL_PREANNOTATION` and `EXTERNAL_PREANNOTATION_ENDPOINT` are gone (no frontier model, decision 2026-10-19).
- §2: scopes declare no `long_doc_types` and no `page_select` task; the section map's page rules are `every` and `declarations`; no page threshold in `vllm_serving.yaml` or anywhere else (every page is read).
- §5: `schema_sections.pages_for` / `leading_pages`; no `Task.PAGE_SELECT`; `DEFAULT_LONG_DOC_PAGE_THRESHOLD` removed from `common.constants`, `MIN_INTER_ANNOTATOR_AGREEMENT` added.
- Current implementation: the SAS token in `.env.example`; the section map's every-page rule.

## Changes of 2026-10-22 (code review)

- `.env.example`: `AZURE_RAW_SAS_TOKEN`, and the account-SAS note for a single token.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.

## Changes of 2026-10-22 (one adapter)

- One adapter (decision 2026-10-22): `personal_lines.yaml`, `scopes.yaml` and `layout_families.yaml` are gone, and `unified.yaml` holds the data mix (§2); `common/adapter.py` replaces `common/scopes.py`, the scope and family loaders left `common/config.py`, and `common/lob.py` gained the line helpers (§5); no overlay-family check at schema load, `assert_one_output_shape` reads the corpus's lines (§3b); `lob_uncertain_threshold` and no `release_pins` (§2); the prompt hash no longer covers `layout_families.yaml`, and `SNAPSHOT_LINES` replaces `CANONICAL_FAMILY` (§4).
- Code review of 2026-10-23: cold start refuses a serving config still holding `routing.release_pins` or `routing.lob_family_threshold` (§2); `common/lob.py` gained `single_line`, the one line a policy is of (§5).

## Changes of 2026-10-23 (forms job)

- `configs/schema_sections.yaml` declares a `forms` group (Forms and endorsements; `policy_schedule`, `every`), empty in the default map and `[forms_and_endorsements]` in the `common_model` profile, ahead of `lineblk`. Reference prompts `policy_<line>.forms.prompt.txt` replace the `lineblk` ones (same text).
- `common/section_builder/` holds FideonSLM's `sections.py` and `pdf_text.py` byte for byte (commit 3d40d77, SHA-256 in `SOURCE.json`; left out of ruff), used only through `common/full_text.py`; `scripts/check_section_builder.py --fideon-slm <clone>` reports a copy edited here or moved on there.

## Changes of 2026-10-23 (incremental training)

- §2 `unified.yaml`: the `continued_training` block (`num_train_epochs`, `learning_rate`, both null) a continued cycle trains with where set (decision 2026-10-23, IMPL-06).

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.

## Changes of 2026-10-24 (printed values only and consistency)

- §2 `vllm_serving.yaml`: `printed_values.mode` (strict by default). `unified.yaml`'s evaluation block: `selection_run_to_run_check` and `run_to_run_check`, both on (the run-to-run agreement of checkpoint selection and the gate, IMPL-08).
- §5 `common/grounding.py`: `ground_tolerant` beside `ground`.
