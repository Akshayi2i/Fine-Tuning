# SPEC 00 — Master Context (Read First)

> Aligned with the code of 2026-10-22.

> **Purpose of this file.** This is the shared context every other spec assumes. When you run any individual spec (IMPL-01 … IMPL-15) in Claude Code, that spec references this file for global conventions. Read this once; each module spec then adds only what's specific to it.
>
> **Source of truth.** This spec set and the architecture document **`Documentation/finetuning-architecture-v2.5.docx`** are **in sync** as of this revision (v2.4, v2.3, v2.1 and v1 are kept for history only). Where a spec cites a section like "(arch §4)", it refers to that document.
>
> **Precedence.** The architecture carries the *why* — design rationale, trade-offs, the reasoning behind each decision. These specs carry the *how* — module boundaries, file paths, function signatures, acceptance criteria. Neither outranks the other: they describe the same system at different altitudes. **If they disagree, that is a bug in one of them.** Fix the disagreement rather than picking a winner, and say which one you changed.
>
> Anything genuinely build-level — a CLI flag, a module name, a test — lives here and is not expected to appear in the architecture.
>
> **Revision:** aligned to **`finetuning-architecture-v2.5.docx`** (its Revision v2.5, 2026-10-22), which records the decisions of 2026-10-19 and 2026-10-22 below. The section numbers are unchanged from v2.1.
>
> **Current target.** **Policy documents only, for now** (decision 2026-10-22): train and gate a policy scope (`--scope policy`, or a line scope such as `personal_lines`); the ACORD and Loss Run code is kept, unchanged, but not targeted. A new training set and canonical schema are to be delivered, and the corpus, the frozen eval set and the adapters are rebuilt from them.
>
> What v2.5 changed (decisions 2026-10-22, after a code review of v2.4), and what it means for these specs:
>
> | Change | Why | Specs affected |
> |---|---|---|
> | **Policy documents only, for now.** A policy scope is trained and gated; the ACORD and Loss Run code stays as it is | The current target. A unified release scored on no ACORD or Loss Run documents is capped at interim, since its scope covers types it was not scored on | 00, 06, 13, 15 |
> | **Every gold needs a review record** (`reviewer_id`, `review_date`), read from its metadata alone; a twin carries its seed's | Only a person's name shows a person checked a gold; notes can be read either way | 00, 03, 04, 05, 08, 12, 13 |
> | **Policy windows sized by their pages' own text**; the corpus pins `window_plan_version` | Dense wording overflowed the sequence cap and set whole documents aside | 04, 05, 11, 13 |
> | **A new training set and canonical schema are to be delivered** | Counts and schema details in these specs describe the older data and the current schema; they change when the new ones arrive | all |
>
> What v2.4 changed (decisions 2026-10-19), and what it means for these specs:
>
> | Change | Why | Specs affected |
> |---|---|---|
> | **Every page, chunk by chunk, never selected pages.** Every window group reads every page; the declarations group's first window is still pages 1–5 plus the detected declarations page. Page routing, the page threshold, `long_doc_types` and the `page_select` task are gone | A page no group read was a page whose values were silently missing | 00, 01, 05, 06, 07, 08, 11, 12, 14 |
> | **No latency SLA is agreed** — latency is recorded, not gated | No target has been set | 00 |
> | **No frontier model for gold, synthetic or training data**; every gold needs a review record (decision 2026-10-22); a twin inherits its seed's | The compliance position. It overrides Fideon SPEC_21's model-drafted seed golds | 00, 01, 03, 04, 05, 08, 12, 13, 15 |
> | **One carrier held out per line of business, only in a line naming 10 or more carriers** | Replaces "2 carriers per doc type once 6 exist" | 05, 08, 15 |
> | **The production tier stays at field match 0.92** | Unchanged | 08 |
> | **GPU only**; GGUF stays an optional edge export | The endpoint runs vLLM on a GPU | 10 |
>
> The first row and the form-edition label rule (§13.2) change training: more windows per policy and new targets, so the corpus must be rebuilt and the adapters retrained, and earlier gate numbers are not comparable with new ones. The carrier rule takes effect when a corpus is next split. The same code also wires the paired-bootstrap gate, caps a release scored on a small eval set at interim, takes the pilot targets from the gate's floors, and adds SAS login and fuller purge plans (§13).
>
> What v2.1 changed, and what it means for these specs:
>
> | Change | Why | Specs affected |
> |---|---|---|
> | **ONE unified LoRA**, merged; per-type only via the §4.2 graduation gate | vLLM applies one LoRA per request, so the v1 Foundation-plus-per-type stack was **unservable** | 00, 02, 06, 07, 10, 11, 12, 13, 15 |
> | **Per-task token budgets** (4k–32k), `max_pixels` not a long-side cap | 8192 could not hold a one-page ACORD with schema, image, OCR and output | 01, 05, 06 |
> | **Task decomposition**: classify, extract, lossrun header/rows/totals, policy section windows (v2.1's `page_select` was removed in v2.4: every page is read) | Long Loss Runs and Policies do not fit one call | 05, 11, 12 |
> | **Group-level splits** (carrier, template, account) + held-out carriers | Source-document splits leak renewals and shared templates | 05, 08 |
> | **Per-epoch modality sampling** | Three rows per document produced 33/33/33, not 50/20/30, and tripled effective epochs | 05, 06 |
> | **Gate**: absolute floors + paired-bootstrap non-inferiority + a recorded written override, given with all three `--override-*` flags | "Beat previous on every metric within 0.001" could not be passed at pilot volume, and had no override | 08, 13 |
> | **Confidence**: per-field-type feature calibrators, risk-controlled thresholds | Min-logprob is length-biased; a fixed 0.70 was tied to no measured error rate | 09, 11 |
> | **Serving formats**: bf16 → FP8 on vLLM; GGUF demoted to an edge export | GGUF is llama.cpp's format and the endpoint runs vLLM | 10, 12, 13 |
> | **13 pipeline stages**, gate after merge/quantize/calibrate | The gate scores the merged model in each serving format, and reads a metric that does not exist until thresholds are chosen | 13 |
> | **Trainer stack corrected**: ms-swift's own `Seq2SeqTrainer` on the HF Trainer, not TRL | Factual correction — TRL is an ms-swift dependency for RLHF trainers, not the SFT loop | 06 |
>
> This supersedes the earlier spec set derived from `qwen3vl-insurance-extraction-finetuning-architecture.md`.
>
> **Implementation status (2026-10-21).** Where a statement below was wrong it is corrected in
> place; what the code does that the original spec set never described is in **§13 Current
> implementation** at the end of this file, and each module spec carries its own "Current
> implementation" section. Read §13 before trusting any version number, split ratio or path in an
> older note.

---

## 0. Read this before anything else — two naming hazards

### 0.1 "SPEC_xx" means two different things

The architecture document references **Fideon pipeline specs** that are *external to this repository*, and they happen to use the same `SPEC_nn` numbering as these implementation specs. They are not the same documents.

| Reference in the architecture doc | What it actually is | Our equivalent |
|---|---|---|
| **Fideon SPEC_00** | The canonical output-schema spec — Pydantic models in `fideon/schemas/` | Consumed by our IMPL-01 (`schemas/`) |
| **Fideon SPEC_01** | L0→L1→L2→L3 routing spec (where the VLM sits in the pipeline) | Context only; see §1.1 |
| **Fideon SPEC_07 Stage 3** | The production **audit gate** that schema-validates every VLM output | Mirrored by our IMPL-08 `schema_validity` + our IMPL-11 output validation |
| **Fideon SPEC_11** | Presidio **de-identification** step | **Not used**: the architecture records Option A (no de-identification) as the current state — see IMPL-05 §1 and §8 below |
| **Fideon SPEC_12** | **Multi-tenant deployment** model | Implemented for corpus, eval sets and releases in our IMPL-02/05/06/08 |
| **Fideon SPEC_05** §4, §4.1, §9 | OCR: modality, `force_ocr` for drawn figures, time budgets | `data_pipeline/ocr/modality.py`, `data_pipeline/ocr/mineru_runner.py` (IMPL-03) |
| **Fideon SPEC_06** §9a / **SPEC_09** §2.1a | Line of business → layout family | `configs/layout_families.yaml`, read by training and serving (IMPL-05/11) |
| **Fideon SPEC_09** | Fine-tuning requirements (amendment and handoff items: carrier hold-out, twin cap, data mix, model card, release tier, latency/routing on releases) | IMPL-05, 08, 13 |
| **Fideon SPEC_21** | The canonical schema set: `common_model.json` plus thin per-line overlays | `configs/canonical schema/`, loaded by `common/schemas.py` (IMPL-01) |

**Convention for this spec set:** files in this folder are **"IMPL-nn"** (renamed from SPEC_nn on 5 October 2026, because the Fideon specs reuse the same numbers). The external pipeline specs are always **"Fideon SPEC_nn"**. So `IMPL-07` is our inference core and `Fideon SPEC_07` is the audit gate; a bare `SPEC_nn` should not appear.

### 0.2 Architecture section numbers changed in v1

The previous spec set cited the old architecture numbering. Everything has been re-cited against v1. For anyone cross-reading old notes:

| Topic | Old § | **v1 §** |
|---|---|---|
| Pipeline integration / schema contract / LoB | *(did not exist)* | **§0, §0a, §0b** |
| Qwen3-VL recap + ViT escalation gate | §2 | **§3** |
| Adapter strategy (ONE unified LoRA; per-type only via the §4.2 graduation gate) | §3 | **§4** |
| Doc-type classifier | §3a | **§4a** |
| ACORD sub-types | §3b | **§4b** |
| Day-zero bootstrap | *(did not exist)* | **§4c** |
| Confidence (per-field-type feature calibrators, risk-controlled thresholds) | §4 | **§5** |
| Dual-input-mode training | §5 | **§6** |
| Dataset format / golden JSON / long docs | §6 | **§7** |
| Corpus management + GROUP-level split strategy | §7 | **§8** |
| MinerU version pinning | *(did not exist)* | **§8a** |
| Multi-tenant corpus isolation | *(did not exist)* | **§8b** |
| Fine-tuning technique (LoRA on a bf16 base) | §8 | **§9** |
| LoRA target-module justification | *(did not exist)* | **§9a** |
| Trainer stack | §9 | **§10** |
| Hyperparameters | §10 | **§11** |
| Hyperparameter sweep methodology | *(did not exist)* | **§11a** |
| Versioning + run registry | §11 | **§12** |
| End-to-end pipeline | §12 | **§13** |
| Serving-format quantization (bf16 → FP8; GGUF demoted to an edge export) | §12a | **§13a** |
| Quantization quality thresholds | *(did not exist)* | **§13b** |
| RunPod infrastructure | §13 | **§14** |
| Evaluation framework | §14 | **§15** |
| Pilot validation protocol | *(did not exist)* | **§16** |
| Extraction / testing routine | §15 | **§17** |
| Azure Blob layout | §16 | **§18, §18a** |
| Repo structure | §17 | **§19** |

---

## 1. What is being built

A production-grade fine-tuning and extraction system for **insurance document extraction** using **Qwen3-VL-8B-Instruct**. The system ingests insurance PDFs (digital and scanned), runs OCR (MinerU), and uses a fine-tuned vision-language model to extract structured **JSON with per-field confidence scores**.

**Active document types (only these three for now):** `acord`, `policy`, `lossrun`.
Quote and Endorsement are deferred — design for easy addition but do not implement them.

**Two production inference modes, one model:**
1. `ocr_plus_image` — MinerU OCR text + page image, both passed to the model.
2. `image_only` — page image only, no OCR (model must still extract).

### 1.1 Where this sits: the L3 VLM layer (arch §0)

This system is the **L3 layer** of the Fideon document-extraction pipeline. L0 → L1 → L2 → L3 routing is defined in **Fideon SPEC_01**. L3 is invoked when the cheaper upstream layers cannot handle a document (scanned input, unknown carrier, structural inference failure).

Three contractual obligations follow, and every spec downstream assumes they hold:

1. **The training target JSON is the canonical Fideon SPEC_00 schema — not an internal format.** Our `schemas/` registry (IMPL-01) is the Fideon SPEC_00 Pydantic models serialised to JSON Schema; for policies it is the client's own per-line canonical schemas in `configs/canonical schema/`, read-only here (§1.2). Field names, types, nesting, and null representation must match exactly.
2. **Any Fideon SPEC_00 schema change requires a corpus rebuild and a new training cycle.** Treat a schema bump like a breaking dependency upgrade, not a patch.
3. **Every golden label is validated against the Fideon SPEC_00 JSON Schema before admission to the corpus** (IMPL-04), and every production inference output is validated by the Fideon SPEC_07 Stage 3 audit gate. Corpus/schema drift therefore surfaces as a production validation failure, not as a silent quality regression.

### 1.2 Document-type vocabulary and the canonical model mapping

The architecture uses two vocabularies: the Fideon SPEC_00 **canonical model names** (arch §0a) and the **corpus/adapter `doc_type` tags** used everywhere else (arch §7 record fields, §17 CLI, §18 Blob layout, §19 repo tree). **Resolved decision for implementation:** `doc_type` stays short and stable (`acord|policy|lossrun`), and a single mapping table resolves it to the canonical schema. Implement the mapping once in `common/constants.py`; never hardcode either name elsewhere.

| `doc_type` (ours — corpus, adapters, paths, CLI) | Fideon SPEC_00 canonical key | Fideon SPEC_00 Pydantic model | Status |
|---|---|---|---|
| `lossrun` | `loss_run` | `LossRunDocument` | active |
| `policy` | `policy_check` | `PolicyCheckDocument` | active |
| `acord` | `acord_mapping` | `ACORDMappingDocument` | active (+ `acord_form` sub-type) |
| `quote` | `quote_gen` | `QuoteGenDocument` | **deferred — do not implement** |

> If Fideon SPEC_00 keys its models differently from the table above, the mapping table is the **only** place to change. Flag it rather than renaming paths.

A **policy's** output schema is chosen one level further down, by line of business: every policy is extracted into the client's canonical JSON for its line (`configs/canonical schema/`, selected by `common.schemas.schema_key("policy", lob=...)`), and a policy with no single line uses the client's `_fallback.json`. ACORD forms and Loss Runs keep the flat schemas in `schemas/` (§13.1).

### 1.3 Line of Business is a first-class VLM output (arch §0b)

L1 (carrier registry) and L2 (structural inference) both attempt LoB detection first. When L3 is invoked, **the VLM must detect and output `line_of_business` itself** for the flat document types (ACORD, Loss Run).

- Valid values: the 12-value enum in `schemas/lob.enum.json` — five commercial lines (`workers_comp`, `general_liability`, `commercial_auto`, `property`, `umbrella`) and seven personal lines (`homeowners`, `personal_auto`, `dwelling_fire`, `ocean_marine`, `motorcycle`, `recreational_vehicle`, `personal_umbrella`).
- `line_of_business` is a **list** of enum values (a certificate or package covers several lines); an **empty list** means the document determines no line. Lines outside the enum go to `line_of_business_other`.
- Emitted in the confidence-bearing shape `{"line_of_business": {"value": ["workers_comp"], "confidence": 0.92}}` (the minimum over the listed values).
- **Present in every flat golden JSON, for every flat document type, even when empty** (IMPL-04 rejects labels missing the key).
- **Corpus coverage target: ≥20% of training examples per LoB value** in the real document population (IMPL-05 reports actual coverage in the manifest; under-coverage is a loud warning, not a silent pass).
- **Reported as its own metric, per LoB value, and it is a gating metric** (IMPL-08). It is never averaged into overall field accuracy — a rare class must not hide inside a healthy aggregate.

**Policies are different.** A policy's line arrives with the request (`lob`, from L1/L2) and selects the line's canonical schema; a canonical policy label carries no `line_of_business` for the model to detect (its line is the label metadata's `lob`), so `lob_detection_accuracy` is structurally not applicable to policy-only scopes (`common.scopes.structural_not_applicable`). A policy's line is a canonical schema name (`gl`, `auto`, `wc`, `homeowners` …): `general_liability`/`cgl` read as `gl`, `workers_comp` as `wc`, `commercial_auto` as `auto` (`common.schemas.LOB_SCHEMA_ALIASES`), and two lines are merged everywhere a line is read — `classic_auto` is `personal_auto` and `watercraft` is `ocean_marine` (`common.lob.MERGED_LINES`).

**Reading a policy's line when no caller sent one exists but ships OFF.** The classifier prompt can ask for the line from every registered line, each described by its role (`configs/lob_meanings.yaml`), with the line's own token probability as its confidence (`serving/doc_type_classifier.py`). `routing.detect_lob` in `configs/inference/vllm_serving.yaml` is `false` until `scripts/measure_lob_detection.py` has measured it on held-out policies against the 0.85 floor and its thresholds (`lob_confidence_threshold` 0.90, `lob_family_threshold` 0.60) are fitted. Off, a policy with no line is refused, or read by the base model against `_fallback.json` when the caller sends `allow_lob_fallback`.

### 1.4 Canonical field mapping — the model does the semantic work

The same real-world field appears under many surface labels across documents. The party purchasing the policy shows up as *Insured Name*, *Named Insured*, *Applicant*, *Name of Applicant*, *Applicant of Insured*, or bare *Name*. **The golden JSON always uses the one canonical key** (`insured_name`), and so does the inference output.

**The contract:**

1. **Golden labels are canonical.** Annotators map whatever the document says onto the canonical key. Surface labels never appear as keys in a golden JSON.
2. **Inference output is canonical.** The extraction returns `insured_name` regardless of how the document phrased it.
3. **The VLM performs the mapping.** This is a core thing the unified extractor LoRA is trained to do (arch §4, "insurance terminology & abbreviations") — recognising that *Applicant* on this page denotes the same field as *Named Insured* on that one.

**How the model is taught it — three mechanisms, all required:**

| Mechanism | Where | What it contributes |
|---|---|---|
| **Semantic gloss in the schema** | IMPL-01 — every field carries a `description` defining what it means **and what it excludes** | A semantic anchor ("find the party purchasing the insurance"), so unseen phrasings resolve. Also lifts base-model pre-annotation quality from day one. |
| **Corpus coverage across variants** | IMPL-05 — alias coverage tracked per canonical field | The mapping is learned from examples; a variant appearing twice is learned weakly |
| **Confusable co-occurrence examples** | IMPL-05 — required edge case | Teaches the **boundary**, not just the mapping |

> ### Anti-pattern — no runtime alias lookup, ever
>
> It is tempting to map extracted labels onto canonical keys with a lookup table at inference. **Do not.** A lookup caps the system at the list someone wrote, cannot handle the phrasing nobody anticipated, and defeats the entire reason for fine-tuning a VLM. The alias registry (IMPL-01) is **training, labeling, and evaluation material only** — it is never consulted at request time and never appears in the prompt.

**Why the prompt carries a gloss and not an alias list.** An alias list gives the model a *lexical* prior, which makes confusable errors worse — tell it `insured_name` may appear as "Insured" and a document containing "Additional Insured" substring-matches. It also never closes the long tail, and since schema text is prompt text, every newly discovered alias would trigger a corpus rebuild and retrain (§7). A gloss encodes the **boundary** ("not the certificate holder, the producer, or an additional insured"), which an alias list structurally cannot, and it is written once.

**The discrimination risk this exists to manage.** Insurance documents are dense with name-like fields — certificate holder, producer/agency, additional insured, loss payee, mortgagee, carrier. A corpus that only ever teaches "name-ish label → `insured_name`" collapses them, producing confident, well-formed, wrong extractions. Confusable misattribution is therefore a **gating metric** (IMPL-08), not a nice-to-have.

---

## 2. Core architectural decisions (do not re-litigate — implement as stated)

| Decision | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-8B-Instruct`, pinned HF revision (`configs/base_model.yaml`). **Loaded from the pod's local copy** under `model.local_dir` (default `/workspace/models`) by training, the tokenizer, the merge and vLLM — never from the Hub at run time (§13) |
| Fine-tuning technique | **LoRA on a bf16 base** — bf16 frozen base weights, LoRA adapters in bf16. Both serving paths hold the base in bf16/fp16 (merged model per IMPL-10, vLLM LoRA hot-swap per IMPL-11), so training in bf16 means the adapter is applied to exactly the weights it trained against. **4-bit NF4 QLoRA remains a live flag** (`quantization.load_in_4bit`) for VRAM-constrained pods — arch §9.2 |
| Trainable components | **LLM decoder** via LoRA. The vision-language mergers are **not** targeted (`lora.include_vision_projector: false`, arch v2.1 §9a). **Vision Encoder (ViT) frozen by default** — escalated only via the eval gate (arch §3). **When the ViT is trained it gets a LoRA — never a full fine-tune** (arch §3). |
| Adapter strategy | **One LoRA per training scope** (`configs/scopes.yaml`; rank 64, alpha 128), each an independent adapter on the bf16 base with its own run, merge, calibration, gate and release. The `unified` scope trains across all document types, all tasks and all modality regimes; the line-scoped `personal_lines` and `casualty_fleet` scopes are layout-family adapters (`configs/layout_families.yaml`). The task, type and schema are always in the prompt, so cross-type interference is controlled by conditioning rather than by separate weights. Per-type adapters return only through the §4.2 **graduation gate** (≥500 labeled docs of that type AND a measured >2pp win outside the bootstrap CI), trained on the merged foundation, decoder-only, **never stacked** — vLLM applies one LoRA per request, so the v1 Foundation-plus-per-type stack was unservable. |
| Trainer stack | **Layer 3 ms-swift** (what you invoke) → **Layer 2 ms-swift's `Seq2SeqTrainer`** (the real loop) → **Layer 1 PyTorch/Transformers/PEFT/bitsandbytes/Accelerate+DeepSpeed**. Locked, one option per layer (arch §10). **Pinned:** ms-swift `>=3.9,<4` (locked 3.12.6), transformers `>=4.57,<4.58` (locked 4.57.6), torch 2.8.0 (§13). |
| Trainer fallback | Migrating to TRL `SFTTrainer` is a **contingency**, permitted only if ms-swift lacks a required Qwen3-VL capability at implementation time — a framework migration (template, collator and masking re-implemented against the parity tests), not a parallel option (arch §10.1, IMPL-06). |
| Collator | ms-swift provides multimodal collation and `-100` masking; `training/data_collator.py` is an **override hook only** (arch §10). |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in every language-model decoder layer; the mergers are not targeted (justified per-module in arch v2.1 §9a). |
| Attention impl | `flash_attention_2`. |
| Confidence | Per-token logprobs of the **printed value's own tokens** (`raw`'s span, without its quotes) → **per-field feature vectors** (min/mean/first-token logprob, token count, OCR agreement of the value *as printed* on the page it cites, rule checks) → **per-field-type calibrators**, fitted on windowed policies merged as they are served → **risk-controlled review thresholds** chosen on a separate validation half (arch v2.1 §5). Min-token-prob alone was length-biased and a fixed 0.7 was tied to no measured error rate; both are gone. List fields also get a **row-completeness** signal. |
| Quantization | **vLLM-native serving formats**: bf16 merged (reference, and cycle 1's serving format) then **FP8 W8A8** via llm-compressor from cycle 2; AWQ INT4 only under VRAM constraint. The vision tower, mergers and `lm_head` stay bf16 in every format. GGUF is an on-request **edge** export, validated in llama.cpp, never the serving path — it is llama.cpp's format and the endpoint runs vLLM. Thresholds are absolute percentage points against bf16, per field class (arch v2.1 §13a-b). |
| Artifact storage | **Azure Blob** — all weights/corpus/PDFs. Repo holds code only. |
| Compute | **RunPod** — ephemeral training pods + a persistent Serverless vLLM inference endpoint. Business logic lives outside RunPod. **GPU only** (decision 2026-10-19): every step that runs a model refuses to start without CUDA rather than fall back to the CPU (§13); GGUF stays an optional edge export, not a deployment target. **On the pod every long job runs detached in tmux**, so a closed laptop or dropped Wi-Fi cannot stop it (§13). |
| Schema contract | Target JSON = Fideon SPEC_00 canonical schema; for policies the client's per-line canonical schemas, nine of which (the seven personal lines, `gl` and `_fallback`) compose the Fideon SPEC_21 common model 1.1.0; audit gate Fideon SPEC_07 Stage 3 validates every inference call (§1.1, §13.1). |
| LoB detection | VLM outputs `line_of_business` (a list over the 12-value enum) for ACORD and Loss Runs; ≥20% corpus coverage per value; own gating metric (§1.3). For a **policy**, the line is supplied by the caller (`lob` on the request) and selects the line's canonical schema; the LoB metric is not applicable to policy scopes. Reading a policy's line in serving exists but **ships off** (`routing.detect_lob: false`). |
| Tenant isolation | `tenant_id` path prefix per Fideon SPEC_12. Tenants keep corpora apart (the CGL data, the smoke run and comparisons each run under their own); each tenant has its own corpus, frozen golden eval set and releases. No cross-tenant mixing in a corpus file (§8 below). |
| De-identification | **Option A — none.** Arch §8b rejects text-only de-identification (it corrupts the training signal) and records Option A — real values, protected by tenancy, RBAC and encryption — as the current state for every tenant's build; `deidentified: false` and `image_redaction: unresolved` are recorded. See §8 and IMPL-05 §1. |
| MinerU | **MinerU 3.4.5, pipeline backend** (layout, PP-OCRv6 OCR and table models, no VLM), on the GPU, on pipeline weights pinned to a fixed PDF-Extract-Kit-1.0 revision. Version **pinned per corpus version**; inference must use the same version as the corpus, and a corpus holding documents read by two MinerU versions is refused (§7 below, arch §8a). |
| Prompt template | Versioned alongside the schema registry; **training-time and inference-time prompts must render identically**; a template change forces a corpus rebuild (arch §7). |
| Page reading | **Every page, chunk by chunk, never selected pages** (decision 2026-10-19). A policy is read as section group × page window over all its pages, planned by one function in training and serving; the declarations group's first window is pages 1–5 plus the detected declarations page, read together (§13.1). |
| Gold and training data | **No frontier model, ever** (decision 2026-10-19): drafts come from the self-hosted Qwen3-VL only, and every gold needs a review record to train, be frozen or score — `reviewer_id` and `review_date` in its metadata, a twin's inherited from its seed (decision 2026-10-22; IMPL-04). |
| Latency | Recorded per release (P95 per adapter, peak GPU memory), **not gated**: no latency SLA is agreed (decision 2026-10-19). |
| Release tier | **production** at field match ≥ 0.92 (kept, decision 2026-10-19), **interim** below that or when the scope was scored on fewer than 150 frozen documents of a type it covers (IMPL-08). |

---

## 3. The `source_id` join key (critical convention)

Every source document has a stable `source_id` (e.g. `acord_0001`, `lossrun_0042`, `policy_0007`). This **same id** appears at every layer — raw PDF, OCR output, golden label, every compiled training row, and every extraction result. It is the traceability spine. Never generate data at any layer without carrying its `source_id`.

Naming: `{doc_type}_{zero_padded_index}`, e.g. `acord_0001`. ACORD form number is tracked as a separate field (`acord_form: "25" | "125" | "140"`), not baked into the doc_type.

Worked traceability chain (arch §7):

```
raw-documents/{tenant}/acord/acord_0001/original.pdf
processed/{tenant}/acord/acord_0001/page_1.png + page_1.md + ocr_meta.json
golden-labels/{tenant}/acord/acord_0001/golden.json + label_metadata.json
corpus/{tenant}/v1/train/epoch_{1..4}.jsonl  →  in each epoch file, the document's rows
                                                  (one per task/window), all source_id
                                                  "acord_0001", in ONE mode drawn for that epoch
corpus/{tenant}/v1/{val,test}/{val,test}.jsonl → its rows in all three modes
```

A training document appears **once per epoch** with its modality drawn per epoch (§10), so N training documents give N documents' rows per epoch, not 3N. A long policy contributes one row per section window (§13.1). Validation and test documents keep all three modes.

---

## 4. Azure Blob layout (the data/artifact side — code never stores these locally)

```
azure-blob://insurance-extraction/
  # ---- tenant-partitioned (PII-bearing) ----
  raw-documents/{tenant_id}/{doc_type}/{source_id}/original.pdf, metadata.json   # immutable, tightest RBAC
  intake/{batch}/{document}/*.pdf, golden.json, metadata.json      # raw container: a delivered batch, staged by azcopy
  processed/{tenant_id}/{doc_type}/{source_id}/page_*.png, page_*.md, ocr_meta.json
  golden-labels/{tenant_id}/{doc_type}/{source_id}/golden.json, label_metadata.json
  corpus/{tenant_id}/v{n}/train/epoch_{1..4}.jsonl                # one modality draw per epoch
  corpus/{tenant_id}/v{n}/train/scope/{scope}/epoch_{1..4}.jsonl    # a scope's view (scoped runs)
  corpus/{tenant_id}/v{n}/{val,test}/{val,test}.jsonl               # all doc types together
  corpus/{tenant_id}/v{n}/{val,test}/scope/{scope}/{val,test}.jsonl # a scope's view (scoped runs)
  corpus/{tenant_id}/v{n}/manifest.json
  golden-eval-set/{tenant_id}/{source_id}/golden.json, metadata.json, page_N.png, page_N.md
  golden-eval-set/{tenant_id}/manifest.json                          # frozen ONCE per tenant (§13)
  releases/{tenant_id}/{release_id}/bundle.json                      # the unit of promotion
  releases/{tenant_id}/{release_id}/calibration/{format}/calibrators.json
  releases/{tenant_id}/{release_id}/gate/{format}/gate_decision.json
  releases/{tenant_id}/release_index.json
  exports/{tenant_id}/{kind}/{name}/                                 # raw container: e.g. comparisons

  # ---- shared / no tenant data ----
  base-models/qwen3-vl-8b-instruct/                                  # cached from HF, pinned revision
  adapters/foundation/v{n}/                                          # the unified adapter
  adapters/scope/{scope}/v{n}/                                       # a scoped run's adapter
                     # published adapter = adapter_config.json + adapter_model.safetensors only
  merged-models/{unified|scope/{scope}}/v{n}/                        # bf16, post merge_and_unload
  quantized-models/{unified|scope/{scope}}/v{n}/vllm/{format}/       # fp8 / awq (gguf only as edge export)
  registry/foundation/{run_id}/run_manifest.json, model_card.md
  registry/scope/{scope}/{run_id}/run_manifest.json, model_card.md
  registry/adapters/{doc_type}/{run_id}/run_manifest.json
  registry/registry_index.json
  calibration/{version}/{doc_type}.json
  eval-reports/v{n}/summary.json                                     # EvalReport.as_dict()
  eval-reports/v{n}/{doc_type}/report.json
  eval-reports/v{n}/gate_decision.json                              # the gate's verdict, NOT the report
  eval-reports/v{n}/scope/{scope}/{summary,gate_decision,checkpoint_selection}.json
```

Every upload records its MD5; the adapter and merged-model pushes (`artifact_registry/transfer.py`) skip a file whose bytes Blob already holds (`BlobClient.upload_dir(..., skip_identical=True)`), so a re-push of an unchanged merged model sends nothing.

**`gate_decision.json` is a separate key on purpose.** The promotion gate writes its verdict — pass/fail, per-metric deltas, failed gates — and the scored `EvalReport` writes `summary.json`. Sharing one key meant the gate overwrote the report it had just read, taking `by_doc_type` and every per-document error record with it; `vit_gate` then saw zero image-only and zero scanned documents and returned `insufficient_data` for ever. Two writers, two keys.

**Tenant partitioning rule (arch §8b, §18):** everything holding tenant document data is prefixed by `tenant_id` — `raw-documents`, `processed`, `golden-labels`, `corpus`, `golden-eval-set`, `releases`, `exports` (`artifact_registry.paths.TENANT_SCOPED`). A frozen eval set is the tenant's documents, and source ids are numbered per tenant, so each tenant is gated on its own set; a set left at the old unscoped root is refused with the move to make. Shared artifacts addressed by scope and version (`base-models/`, `adapters/`, `merged-models/`, `registry/`) stay un-prefixed.

**Scope for the current build:** `tenant_id` defaults from `DEFAULT_TENANT_ID` and is not a required argument on every CLI (`--tenant` names another). Tenants are used to keep corpora apart — the CGL data that trains `casualty_fleet`, the smoke run (`smoke`), base-vs-adapter comparisons (`compare`) — each with its own corpus, frozen eval set and releases. The rule that is enforced at build time is **no cross-tenant mixing** — a corpus file must never contain rows from two tenants, because that is a training-data-composition rule, not plumbing.

**Access-control split (arch §18a):** `raw-documents/` lives in a separate container (`AZURE_RAW_CONTAINER`) from everything else — it is the layer holding unredacted PII. `intake/` (delivered batches staged by azcopy) and `exports/` (result sets that quote policy data, such as base-vs-adapter comparisons) sit in the same container under the same rule (`artifact_registry.paths.requires_raw_container`). No training or serving component ever reads it directly.

---

## 5. Repository structure (code side — as built on 2026-10-21)

```
insurance-extraction-finetuning/
├── pyproject.toml  requirements*.txt  requirements-*.lock  .env.example  README.md
├── configs/          base_model.yaml  scopes.yaml  layout_families.yaml  schema_sections.yaml
│   │                 model_view.yaml  lob_meanings.yaml  label_mappings.yaml
│   │                 coverage_code_recodes.yaml  coverage_code_meanings.yaml (template)
│   ├── canonical schema/LOB Schema/     34 client schema files: 31 lines (registered), _fallback,
│   │                                    _common, classic_auto (read as personal_auto) + coverage
│   │                                    codes, crosswalks, notes
│   ├── canonical schema/common schema/  common_model.json (1.1.0), lob_schema_map.yaml, inventory
│   ├── shared/       vision.yaml (pixel budgets)  sequence.yaml (per-task caps)
│   ├── training/     unified.yaml  personal_lines.yaml (extends unified)
│   ├── sweeps/       phase1_lr  phase2_epochs  phase3_rank  README
│   ├── inference/    vllm_serving.yaml
│   └── deepspeed/    zero2.json  zero3.json
├── schemas/          acord25/125/140, lossrun, policy_doc (legacy) schemas,
│                     common_fields.json, lob.enum.json, aliases/, examples/
├── prompts/          system_prompt_template.jinja, doc_type_classifier_prompt.jinja,
│                     doc_types/{acord,lossrun,policy}.jinja
├── common/           aliases canonical config constants gpu grounding ids label_mapping
│                     lob logging_setup model_view normalize prompts run_ids
│                     schema_sections schemas scopes structural_ids tasks
├── artifact_registry/  blob_client  paths  transfer
├── registry_utils/   models  model_card  query_registry  write_run_manifest
├── data_pipeline/    audit  ocr_check  corpus_manifest
│   ├── ingestion/    pull_intake  pull_raw_pdfs  prepare_bundles  label_rules
│   │                 fill_synthetic_labels  import_labeled_pdfs  import_prepared
│   │                 dedup_and_group
│   ├── ocr/          run_mineru  mineru_config  mineru_runner  mineru_version
│   │                 modality  render_only
│   ├── labeling/     pre_annotate  export_golden_labels  derive_aliases
│   │                 active_learning  review_tool/
│   └── dataset_builder/  split_groups  sample_modes  expand_tasks  policy_windows
│                     noisy_ocr_augment  cap_check  label_verification  build_jsonl
├── training/         train  stage_data  length_check  corpus_view  data_mix
│                     base_precision  data_collator  merge  sweep  vit_gate  parity
│                     callbacks/early_stopping
├── inference_core/   model_runner  input_builder  span_map  runner_config  vllm_patches
├── evaluation/       run_eval  golden_eval  freeze_eval_set  checkpoint_eval
│   │                 checkpoint_score_worker  validation_generation  validation_sample
│   │                 gating  bootstrap  release_measurements  lob_detection_eval
│   │                 windowing_ceiling
│   └── metrics/      field_accuracy  common_model  confusable  coverage_metrics
│                     extraction_faults  auto_accept  lossrun_table  lob_detection
│                     unseen_labels
├── calibration/      features  feature_calibrator  thresholds  fit_calibration
│                     apply_calibration  logprob_confidence  list_completeness
│                     reconciliation
├── postprocessing/   quantize  quant_thresholds  validate_quant
├── serving/          vllm_entrypoint  pipeline  policy_merge  lossrun_merge
│                     conventions  page_router  doc_type_classifier  adapter_router
│                     release_router  routing_check
├── orchestration/    run  pipeline_dag  preflight  smoke_run  runpod_controller
│                     detach  settings  config/pipeline.yaml
├── scripts/          pod_bootstrap.sh  setup_pod.sh  pod_run.sh  download_mineru_models.sh
│                     lock_requirements.sh  phase0_spike  combine_specs
│                     derive_aliases_from_canonical  check_alias_coverage
│                     propose_field_types  measure_lob_detection
│                     diagnose_generation  diagnose_windowing
├── testing/          run_extraction  render_prompts  compare_models  comparison
│                     prompts/ (45 reference prompt snapshots)
├── pilot/            zero_shot_baseline  smoke_test  pilot_run  pilot_report
└── tests/            122 test modules + conftest.py; fixtures/ (synthetic, no PII)
```

`Documentation/` (these specs and the architecture documents) is listed in `.gitignore`, so new files there are not picked up; the files already tracked, these specs among them, remain under version control.

**Key separation principle:** the repo never stores model weights, corpus data, or PDFs. Everything data/artifact-related is pulled from / pushed to Azure Blob at runtime via `artifact_registry/`.

---

## 6. Spec build order & dependency graph

Build in this order — later specs import from earlier ones. **IMPL-07 (inference core) is deliberately placed before evaluation/calibration/serving/testing because all four share it** — this breaks what would otherwise be a circular dependency between them.

```
IMPL-01  Project scaffold, configs, schemas, prompts, common   (no deps)
IMPL-02  Azure Blob I/O + training run registry                (deps: 01)
IMPL-03  Ingestion + MinerU OCR (+ version pinning)            (deps: 01, 02)
IMPL-04  Labeling + golden JSON + day-zero bootstrap gate      (deps: 01, 02, 03)
IMPL-05  Dataset builder (JSONL, modality, split)              (deps: 01, 02, 04)
IMPL-06  Training (ms-swift LoRA, one adapter per scope)       (deps: 01, 02, 05)
IMPL-07  Inference core (shared model-runner primitive)        (deps: 01, 02)
IMPL-08  Evaluation (metrics + gating)                         (deps: 01, 02, 06, 07)
IMPL-09  Confidence calibration                                (deps: 01, 02, 07, 08)
IMPL-10  Postprocessing (merge + FP8/AWQ quantize + thresholds) (deps: 01, 02, 06)
IMPL-11  Serving (vLLM endpoint, classifier, routing)          (deps: 01, 02, 07, 09, 10)
IMPL-12  Testing / extraction routine (CLI harness)            (deps: 01, 02, 03, 07, 08, 09, 11)
IMPL-13  Orchestration (RunPod controller, pipeline DAG)       (deps: 01-12)
IMPL-14  Unit tests + CI fixtures                              (deps: 01-13)
IMPL-15  Pilot validation protocol (operational runbook)       (deps: 01-14; executed, not imported)
```

**Deferred until after IMPL-15 passes** (they are real work, just not first-cycle work):
- **Hyperparameter sweep execution** (arch §11a) — 9–12 runs on 25–30 docs/type mostly measures noise. `configs/sweeps/` is authored in IMPL-01; running it belongs *after* the pilot and *before* production-scale training.
- **Quantized-format validation** (arch §13b) — the primary serving path is vLLM on the merged fp16/bf16 model. Nothing is quantized in the first cycle, so nothing needs threshold validation yet.

**Note on cross-references that point "forward" but are not build-order violations:**
- `postprocessing/validate_quant.py` (IMPL-10) calls the testing routine (IMPL-12). `validate_quant` is a *gate you run after IMPL-12 exists*, not something merge/quantize needs. Build merge+quantize in IMPL-10; wire `validate_quant` once IMPL-12 is done.
- `evaluation/run_eval.py` (IMPL-08) can compute classifier accuracy only once the classifier (IMPL-11) exists. IMPL-08 runs *without* it (supply the true doc_type); classifier metrics fill in on a later pass.
- IMPL-15 is a **runbook**, not a library. It orders the pilot experiments (arch §16) and is executed against the built system; nothing imports it.

Everything else is strictly bottom-up.

---

## 7. Global conventions every spec must follow

- **Language/runtime:** Python 3.11+. Type hints throughout. `pydantic` v2 for config/data models.
- **Config:** all runtime config via YAML in `configs/` + environment variables (`.env`, loaded with `python-dotenv`). Never hardcode paths, model ids, or credentials.
- **Secrets:** Azure + RunPod credentials from env vars only (including `RUNPOD_VOLUME_ID` / `RUNPOD_VOLUME_MOUNT` for the staging volume, §12a). `.env.example` lists every required var with placeholders. Never commit real secrets.
- **Logging:** stdlib `logging`, set up once per run by `common/logging_setup.py` (`setup()`; format `%(levelname)s %(name)s %(message)s`). Its console handler is a `ConsoleHandler` subclass of `StreamHandler`, which ms-swift's and modelscope's logger setup leave alone, and `reassert()` runs before every pipeline stage and after the in-process ms-swift import to undo what any library did to our loggers. **Never log PII** (raw OCR text, extracted field values, image bytes) at INFO or above.
- **Blob access:** all Azure Blob reads/writes go through `artifact_registry/` (IMPL-02). No module opens the Azure SDK directly except that module.
- **Determinism:** set and log random seeds for any stochastic step (splitting, training, augmentation, generation). The seed is recorded in the run manifest.
- **CLIs:** every runnable module exposes an `argparse`/`typer` CLI with `--help`. Long-running scripts print progress.
- **Error handling:** fail loudly with clear messages on missing config, missing Blob artifacts, or schema-invalid data. Don't silently continue.
- **No network assumptions:** code may run inside a RunPod pod with a restricted egress allowlist; make external endpoints configurable, not hardcoded.
- **Idempotency and resumability (arch §13):** every pipeline stage is idempotent and resumable — a failed stage must not corrupt state and must be safe to simply re-run. Key on `source_id` + content checksum; never duplicate.
- **Tenant scoping:** Blob paths are constructed only via `artifact_registry/paths.py`, which applies the `tenant_id` prefix. `tenant_id` defaults from `DEFAULT_TENANT_ID` and is not a required CLI argument (`--tenant` names another). Each tenant has its own corpus, frozen golden eval set and releases. The live rule is **no cross-tenant mixing in a corpus file** (a training-data-composition rule, `assert_single_tenant` in the corpus build).
- **MinerU version pinning (arch §8a):** the MinerU version (now 3.4.5) and device used to preprocess a corpus version are recorded on every `ocr_meta.json` and in that corpus manifest, and pinned; so are MinerU's pipeline weights (`scripts/download_mineru_models.sh`, a fixed PDF-Extract-Kit-1.0 revision, read locally). **Inference must use the same MinerU version as the training corpus.** A mismatch is treated as **distribution shift and a regression trigger**, not a routine dependency bump — an upgrade means reprocessing the affected documents (`run_mineru --all-unprocessed` treats a document another MinerU read as unprocessed) and incrementing the corpus version *before* the next training cycle; the corpus build refuses documents read by two versions. This matters because the model is fine-tuned partly on *how MinerU formats its output* (table markdown conventions, reading order, error patterns — and, since 3.4, page headers and footers kept in each page's text).
- **Prompt/schema versioning (arch §7):** the system prompt template is versioned alongside the schema registry and both versions are recorded in the corpus manifest. **A change to either forces a corpus rebuild and a new training cycle.** This includes **field `description` text** — the semantic glosses of §1.4 are injected into the prompt with the schema, so editing a gloss is a schema change, not a documentation tweak. Settle the glosses before the first real corpus build. (The **alias registry** is not prompt material and is exempt: adding a newly observed surface form is a registry edit and triggers no rebuild.) Training-time and inference-time prompts must render identically — this divergence is the single most common cause of post-fine-tuning degradation and is invisible in training metrics because it only manifests at serving time.

## 8. PII, tenancy, and de-identification (non-negotiable, applies to every spec)

Insurance documents contain PII (names, TINs/SSNs, addresses, financials).

- `raw-documents/` is the most sensitive layer — separate container, tighter RBAC, encryption at rest, immutable, Cool tier once processed, never Archive (a corpus rebuild after an OCR pin change needs it readable without hours of rehydration). Only `data_pipeline/ingestion/` writes it and only `data_pipeline/ocr/` reads it.
- **Cross-tenant mixing is prohibited.** Corpus, processed output, and golden labels are `tenant_id`-prefixed (§4). Assert that a corpus file never contains rows from two tenants.
- **No de-identification (Option A, arch §8b)** — see the note below and IMPL-05 §1. PII protection rests on tenancy, access control and encryption, and that is stated to the compliance owner rather than left implicit.
- Never send PII to third-party APIs. Pre-annotation (IMPL-04) uses the self-hosted Qwen3-VL only: frontier models are never used for gold, synthetic or training data (decision 2026-10-19).
- Never persist raw request/response bodies or OCR text in plaintext logs (affects IMPL-11 serving, IMPL-12 testing).
- Scrub/access-control eval reports and error dumps — a logged prompt or a failed extraction printed to a log can leak PII into low-security observability tooling.

> ### De-identification — why not text-only, and the current state (Option A)
>
> Text-only Presidio de-identification is not used (arch §8b). It de-identifies the **text**, not the **page images** the vision encoder reads, and that is not merely an incomplete privacy control — it **corrupts the training signal**:
>
> - **`image_only` (30% of the Foundation corpus):** the image shows "John Smith", the target says `PERSON_1`. The target is **not derivable from the input**. That is an unlearnable example, and 30% of the corpus made of them is pure hallucination pressure.
> - **`ocr_plus_image` (50%):** OCR says `PERSON_1`, image says "John Smith", target says `PERSON_1` → teaches **trust-OCR-over-image**, the exact inverse of what the decoder LoRA and the 20% `noisy_ocr_image` regime exist to teach (arch §3, §6).
>
> **Half-de-identifying is worse than either extreme.** De-identification must be consistent across OCR text, page image and target JSON. The architecture lists three options — **A**, no de-identification (real values; tenancy, RBAC and encryption); **B**, consistent replacement in the image, the OCR text and the target; **C**, a PII-free synthetic layer for any shared model — and records **Option A** as the current state for every tenant's build. The corpus manifest records `deidentified: false` and `image_redaction: "unresolved"`, and this is stated to the compliance owner explicitly.

## 9. Data contracts (shared shapes used across specs)

**Training example (one JSONL row):**
```json
{
  "doc_type": "lossrun",
  "acord_form": null,
  "modality_mode": "ocr_plus_image | noisy_ocr_image | image_only",
  "source_id": "lossrun_0042",
  "lob": ["workers_comp"],
  "tenant_id": "broker_a",
  "split": "train | val | test",
  "deidentified": false,
  "messages": [
    {"role": "system", "content": "<prompt: schema + modality instructions>"},
    {"role": "user", "content": [
      {"type": "image", "image": "<page 1 image>"},
      {"type": "text",  "text": "<page 1 of N>

<page 1 MinerU markdown>"},
      {"type": "image", "image": "<page 2 image>"},
      {"type": "text",  "text": "<page 2 of N>

<page 2 MinerU markdown>"}
    ]},
    {"role": "assistant", "content": "<target JSON string, incl. line_of_business>"}
  ]
}
```
`doc_type`, `acord_form`, `modality_mode`, `source_id`, `split` are the required top-level keys from arch §7; `tenant_id` and `deidentified` are added by the tenancy and de-identification rules (arch §8b) — `deidentified` is `false` on every row under Option A (§8). The corpus build adds what later stages read: `lob`, `task`, the window's `sections`, `window_index` and `window_pages` (a policy is one row per section window), `epoch` (train), `group_id` (the family), `synthetic`, `carrier`, `twin_index` (a synthetic twin's number, else null), `render_mode` (digital / scanned render of a twin), `is_scanned`, and `val_half` (calibration or threshold half of validation).

**The user turn is interleaved: each page's image is immediately followed by that page's own markdown, prefixed `<page N of M>`.** Three reasons, and all three were live defects before it was:

1. **Correspondence.** With all images first and one joined text blob, nothing said where page 1's text ended. The evidence rule — *trust the image where the two disagree* — requires knowing which image, and on a 40-page policy the model cannot recover that by content matching.
2. **Markdown integrity.** A blank line terminates a Markdown table, so joining pages split every table crossing a page boundary in two, the second half without its header. Loss Runs and long policies are exactly where row completeness matters most.
3. **One shape, not two.** A window of a long document (arch §7) is the same structure with fewer pairs — `[img_9][txt_9][img_10][txt_10]` — instead of N separate single-page calls asking for a whole-document JSON from one page, a shape no training row ever had.

`image_only` carries the marker and no markdown: the guarantee is *no OCR text*, not *no text at all*, and without it an image-only window cannot tell the model it holds pages 9 and 10 of 20 rather than a two-page document. A gap in the numbering is itself the signal that absent fields live on pages the model was not given.

**Loss is computed only on the assistant tokens** — system, image, and OCR-text tokens are masked with label `-100`.

**Extraction output (per document, at inference/testing):**
```json
{
  "source_id": "abcLossRun",
  "doc_type": "lossrun",
  "model_version": "v2",
  "mode": "ocr_plus_image",
  "schema_valid": true,
  "overall_confidence": 0.88,
  "line_of_business": {"value": ["workers_comp"], "confidence": 0.92},
  "extraction": { "<the JSON itself: every key of the schema, null where not extracted>" },
  "fields": {
    "<field>": {"value": "...", "confidence": 0.94}
  },
  "list_fields": {
    "<list_field>": {"rows": [...], "row_completeness_confidence": 0.7}
  },
  "pages_used": [1, 2, 3, 4, 5],
  "review_flags": ["valuation_date:low_confidence", "claims:row_count_mismatch"],
  "route": {"adapter": "...", "foundation_only": false, "classifier_confidence": 0.97,
            "classifier_method": "...", "layout_family": null, "lob_fallback_used": false,
            "lob_source": "..."},
  "reconciliation": { "<Loss Runs only: claims against printed totals>" }
}
```

`ExtractionResult.as_dict()` in `serving/pipeline.py` is this contract. `extraction` is the served JSON: for a policy the client's canonical JSON with every key of the line's schema (a post-process that fills absent keys, never trained), values in `{raw, parsed, confidence, page_ref, flagged}` envelopes; `fields` and `list_fields` summarise it. Policy review flags include `<path>:not_printed` and `<path>:likely_missed` from the serving conventions (§13.5). The testing routine adds `all_fields` to each results JSON: every field the document's schema declares, with its value, status and, against gold, its result (`testing.comparison.field_report`).

**Run manifest (registry, per training run):** see IMPL-02 for the full schema.

## 10. Modality mix for training data (used by IMPL-05)

Each training document appears **once per epoch**, its regime drawn per epoch from a seeded generator against the target mix (`data_pipeline/dataset_builder/sample_modes.py`, `common.constants.MODALITY_MIX`; arch v2.1 §6.1); the corpus always writes four epoch files. Validation and test rows take all three regimes. A scope's training config may set its own `modality_mix`; the corpus records the mix it was drawn with, and a run on a corpus drawn with another mix is refused.

| Regime | Share | Purpose |
|---|---|---|
| `ocr_plus_image` — image + real MinerU OCR | ~50% | Baseline dual-modality behavior |
| `noisy_ocr_image` — image + deliberately imperfect OCR, **correct** golden target | ~20% | Teaches image-over-OCR arbitration |
| `image_only` — image, no OCR block at all | ~30% | Satisfies the must-work-without-OCR requirement |

The system prompt **explicitly declares which mode is active** rather than silently omitting the OCR block — that explicit signal is what lets the model switch behavior reliably instead of guessing which mode it is in.

**Split at FAMILY (group) level BEFORE modality expansion** to prevent leakage — renewals and same-template documents stay in one split (arch v2.1 §8.2). Split ratio scales with per-type volume (arch §8), `common/constants.py::SPLIT_RATIOS_BY_VOLUME`:

| Data volume per doc type | Train / Val / Test |
|---|---|
| under 200 (pilot, ~25–30/type) | 70 / 18 / 12 — metrics are directional, not final |
| 200–999 | 75 / 15 / 10 |
| 1000+ (target state) | 80 / 10 / 10 |

The band is chosen per **document type** (the gate's metrics are per type) and applied **per line of business** (IMPL-05 §Current implementation): a line under 5 documents trains whole; a measured line always reaches val and test. The band table is checked at import to only ever raise both split edges, so crossing a band never moves a trained document into val or test. Once the tenant's golden eval set is frozen (§13), new documents split into **train and val only**.

**A delivered split is used as given.** When every document carries a split (the SPEC_21 delivery's `split_manifest.csv`; the CGL bundles, whose split is drawn once at bundling and delivered in their metadata), `split_groups.assign_delivered_splits` takes it: one family per seed, a seed in two splits refused, synthetic twins allowed in val and test. The per-line carrier hold-out (Fideon SPEC_09 amendment item 4; decision 2026-10-19) holds **one carrier per line of business** out of train and validation, and only in a line whose real documents name **at least 10 carriers** (`split_groups.MIN_CARRIERS_PER_LINE_FOR_HELD_OUT`); a line with fewer holds none out. It applies to a drawn split and is **off** for a delivered split (`HOLD_OUT_CARRIER_PER_LINE = False`; the delivery already holds whole seeds out; asked for, the same minimum applies). At most 20 twins of one seed per render mode train (`MAX_TWINS_PER_SEED`).

**What a scope's training view adds** (`training/corpus_view.py`, line-scoped views only): one render per twin per epoch (`render_rotation: true` — a twin's digital and scanned PDFs share one gold, so each epoch trains one and the next the other), small lines repeated up to `line_balance.max_repeat` toward `min_documents`, and for `personal_lines` a per-line `synthetic_fraction` / `scanned_share` (`configs/training/personal_lines.yaml`, all null until set; `training/data_mix.py`).

---

## 11. How to use these specs with Claude Code

1. Start a Claude Code session at the repo root.
2. Run `IMPL-01` first to scaffold the project.
3. Run each subsequent spec **in numeric order (01 → 14)**. Each spec is self-contained but assumes earlier specs are done. The order is dependency-correct — no spec references code from a higher-numbered spec as a build prerequisite (the two "forward" mentions in IMPL-08 and IMPL-10 are explicitly non-blocking; see §6).
4. Each spec ends with an **Acceptance checklist** — verify it before moving on.
5. **IMPL-15 is executed, not built** — run it after IMPL-14 as the pilot validation protocol before committing to full corpus annotation.
6. This master file is the context; when a spec says "per master context", it means this file.

---

## 12. The operator command surface (IMPL-13)

A cycle is run through **three commands plus one umbrella command**, all from `orchestration/run.py`. Anyone running the pipeline needs the version tag and nothing else — no paths, no pod management, no stage wiring.

| # | Command | Stages (`orchestration.pipeline_dag.STAGES`) | Ends with |
|---|---|---|---|
| **1** | `finetune` | 1 → 7: ingestion, preprocessing (OCR), labeling, dataset_build, training, checkpoint_eval, merge | Adapter + merged model on the **RunPod staging volume**; run manifest in Blob as `staged` |
| **2** | `package` | 8 → 11: quantize, calibrate, evaluation_gate, package (push) | Adapter files, merged + quantized model, calibrators and the release bundle in **Azure Blob** |
| **—** | `deploy-endpoint` / `rollback-endpoint` | 12: serving | The endpoint pointed at a promoted version (or the previous one) |
| **3** | `extract` | 13 (§17 extraction routine) | JSON + confidence + metrics, locally |
| **—** | `all` | `finetune` then `package` | Same as command 2 |

```bash
python -m orchestration.run finetune --input ./intake --corpus-version v1 --out-version v2 [--scope personal_lines] [--tenant T]
python -m orchestration.run package  --version v2 --release-id release-2026.10.1 --formats bf16 [--fixes-defect "<ticket>"]
python -m orchestration.run package  --version v2 --release-id release-2026.10.1 --override-approver "A. Reviewer" --override-reason "<why, 20+ characters>" --override-gate <failed gate>   # a written waiver, one scope
python -m orchestration.run extract  --model base|v1|v2 --tenant T --source-ids policy_0001 policy_0002
python -m orchestration.run extract  --model base --adapter <adapter folder> --tenant T --split test --corpus v1
python -m orchestration.run all      --input ./intake --out-version v2 --release-id release-2026.10.1
python -m orchestration.run freeze-eval-set --corpus v1 [--tenant T] [--allow-small]   # once per tenant
python -m orchestration.preflight --scope personal_lines [--skip-corpus]              # checks before finetune (incl. reviewed labels)
python -m orchestration.smoke_run [--scope casualty_fleet]                            # small end-to-end run
```

`extract` reads documents already imported and OCR'd (`--source-ids`, or a corpus split with `--split`/`--corpus`), scores them against their labels unless `--no-score`, and writes JSON per document to `testing/results/<model>/` and metrics to `testing/metrics/<model>/`. `--adapter` applies a LoRA folder (a checkpoint or a run's adapter) to the base model per request, unmerged.

On the pod every one of these (except the seconds-long endpoint switches) starts itself in tmux and
returns with the job's name; see §13.

**`all` never includes `extract`.** Extraction is a separate concern from building a model — it runs against any chosen version, including models trained weeks earlier and the untuned base.

**Three facts this surface has to respect, and does:**

1. **Labeling is human work inside the span of command 1.** `finetune` ingests and OCRs everything, builds the corpus from only the source_ids that have a validated `golden.json`, and **reports the unlabeled backlog**. It aborts before training only when the labeled set is below `--min-labels-per-type` (default 25, matching IMPL-04's day-zero rule).
2. **The gate is a hard stop inside command 2.** `package` quantizes and calibrates first — the gate reads `auto_accept_error_rate`, which does not exist until thresholds are chosen — then runs the evaluation gate on the frozen golden set through the serving pipeline. Non-inferiority is a paired bootstrap over the documents the candidate and the promoted release both scored — weighted as the metric counts them, and only when both were scored on the same frozen set of the tenant; otherwise a point comparison, with the reason in the decision's warnings — and a release that improves no primary metric is blocked unless it names the defect it fixes (`--fixes-defect`, per scope when several run). The only way past a failed gate is a written override, given with `--override-approver`, `--override-reason` and `--override-gate` (all three, one scope); there is no `--force`. A failed gate stops before anything is pushed and exits non-zero, so `all` never publishes a model that failed it. The release's routing check also runs before the push.
3. **Command 1 launches one training job per scope.** Stages 1–4 build the corpus once and every scope shares it (`SHARED_STAGES`); stages 5–11 run once per scope (`PER_SCOPE_STAGES`, `--scope` is repeatable), each an independent adapter. v1 fanned out into 1 + N sequentially, because each per-type adapter sat on top of the Foundation. That topology is gone (arch v2.1 §4.1): vLLM applies one LoRA per request, so a Foundation LoRA and a per-type LoRA could never both be active on a call.

Every stage records its wall time (`wall_seconds`), and the run report so far is written after every stage to `$FIDEON_LOG_DIR` (default `/workspace/logs`, when that folder exists) as `run-<command>-<scope>-<tenant>-<version>.json`, so two scopes or tenants trained under one version tag keep their own.

### 12a. Staging volume vs. registry — do not conflate these

| | **Staging volume** | **Registry** |
|---|---|---|
| Where | RunPod network volume at `$RUNPOD_VOLUME_MOUNT/staging/` (default `/runpod-volume`; set `/workspace` when the pod mounts the volume there) | Azure Blob `registry/` |
| Holds | Working copies of adapters, merged model, eval reports between commands 1 and 2 | Durable `run_manifest.json` per run |
| Lifetime | Cleared after a verified push | Permanent |
| Guarantee | **None** — working storage | Lineage of record (arch §12) |

RunPod pods are ephemeral (arch §14), so persistence between commands 1 and 2 comes from a network volume attached to every launched pod. Staging exists because the merged model is ~16 GB and quantization also runs on RunPod — pushing it to Azure at the end of command 1 and pulling it back at the start of command 2 is a 32 GB round trip for no benefit.

**But a volume is not an artifact of record.** So command 1 **always writes its run manifest to Blob** with `artifacts.status: "staged"`, even though the weights stay on the volume; command 2 flips it to `"published"` with real Blob paths. Without that, a reclaimed volume means a training run that happened and left no trace.

---

## 13. Current implementation (2026-10-21)

What the build does now that the sections above did not describe. Each module spec has its own
"Current implementation" section with the detail; this is the index.

### 13.1 Output, schemas and prompts
- **Policies output the client's canonical JSON** (`configs/canonical schema/`): FieldValue envelopes
  `{raw, parsed, confidence{score,source}, page_ref, flagged}`. The model writes only `raw`, `parsed` and
  `page_ref`, and only for values the document states; confidence and flags are added after, and the
  served JSON then carries **every key of the line's schema**, null where nothing was extracted (a
  post-process, never trained).
- **Two shapes of line schema.** Nine canonical files compose the **Fideon SPEC_21 common model 1.1.0**
  (`canonical schema/common schema/common_model.json`): the seven personal lines (homeowners,
  personal_auto, dwelling_fire, ocean_marine, motorcycle, recreational_vehicle, personal_umbrella), `gl`
  (3.0.0) and `_fallback` (the schema for a policy with no single line, registry key `policy`). The
  registry bundles each overlay at load (`common.schemas._bundle`; version `<overlay>+common.1.1.0`) and
  `common.schemas.is_common_model` is the one switch every difference keys off. The other lines keep
  self-contained files. A layout family part-way through the move (`casualty_fleet`: `gl` moved; `auto`,
  `wc`, `umbrella` not) warns at load, and training refuses a scope whose corpus holds both shapes of one
  family (`common.scopes.assert_one_output_shape`). `classic_auto.json` and `_common.json` are not
  registered: 36 schema keys in all (31 lines, the fallback, three ACORD forms, the Loss Run).
- **The model view** (`common/model_view.py`, `configs/model_view.yaml`) is what the prompt carries and the
  decoder enforces on a common-model line: every printed value raw/parsed/page_ref; identifiers, references,
  codes and types as plain strings; no pipeline-filled field; each coverage code with its meaning; the
  keywords xgrammar refuses dropped. Validation still judges the client's whole schema, on the JSON served.
  **Structural ids** (`veh_N`, `loc_N`, `cov_N` …; `common/structural_ids.py`) are numbered per window and
  renumbered once, in printed order, after the merge; scoring never compares ids, only what they name.
- **Dates are `MM/DD/YYYY`** in every output (`common/canonical.py::with_output_dates`), for any field with
  the word `date`/`dates`/`dated`/`dob` in its name — a test runs over all LOB schemas.
- **Policies are read as windows over every page**: section group (`configs/schema_sections.yaml`: decl,
  arrays, lineblk, dtd for self-contained lines; a `common_model` profile with decl, arrays, lineblk) × page
  window, planned by ONE function shared by training and serving
  (`data_pipeline/dataset_builder/policy_windows.py`) and merged at serving (`serving/policy_merge.py`:
  units joined on their own keys — VIN, number, then year/make/model — references rewritten, rows read
  twice through a shared page dropped). **Every group reads every page, chunk by chunk, never selected
  pages** (decision 2026-10-19): `policy_windows.pages_to_read` returns every page and the declarations page
  `serving.page_router.plan_pages` detects; the declarations group's (and `dtd`'s) first window is pages 1–5
  (`pages.declarations_leading`) plus that page, read together, and the pages after it follow in windows as
  every other group's do. The windows of a split page run **share one page** (`run_overlap_pages: 1`,
  common-model lines); `policy_schedule` reserves 12,288 output tokens. Windows run as one vLLM batch and a
  failed window is contained. A long document of another type is read in one call, or, for a Loss Run, in
  its windows.
- **Long Loss Runs** are read window by window only when one answer cannot hold their rows, then merged
  and reconciled (`serving/lossrun_merge.merge_and_reconcile`).
- **Prompt** `PROMPT_TEMPLATE_VERSION = "7.2.0"`. On common-model lines it says: each value once, its
  `page_ref` listing every page that prints it; rows only from the pages given; a value with a field never
  repeated under the additional fields; the arrays window is told which sections other windows read. The
  self-contained lines, ACORD and Loss Run prompts are frozen as fingerprints
  (`tests/fixtures/old_style_prompt_fingerprints.json`). The **prompt hash** (`common.prompts.prompt_hash`)
  covers every template, every schema (the common model and the coverage-code files included),
  `model_view.yaml`, the section map, `lob_meanings.yaml` and `layout_families.yaml`; serving refuses at
  cold start to serve a release packaged with different prompt inputs.

### 13.2 Data, labels and splits
- **Bundles before import** (`data_pipeline/ingestion/prepare_bundles.py`): `prepare_twin_bundles` reads the
  SPEC_21 delivery (`split_manifest.csv`; each twin a digital and a scanned document reading one gold; each
  seed bundled as a real document in its twins' family and split); `prepare_original_bundles --lob` reads real
  documents of one line with no twins (the CGL source data; one family per insured; split drawn once and
  delivered in the metadata; a test document of a carrier no training document has is marked
  `held_out_carrier`). Each gold is corrected in `corrected_gold`, every change listed in `corrections.csv`:
  coverage codes re-coded (`configs/coverage_code_recodes.yaml`, a code file's `replaces_codes`), duplicate
  premium items dropped, a printed carrier filled, then the **label conventions** (`label_rules.apply_label_rules`):
  a value printed several times appears once, in its own field, its `page_ref` listing every page that prints
  it; additional fields hold only values with no field of their own; policy wording and drafting notes filed as
  extra fields are dropped; a form's edition printed inside its form number in an unambiguous shape moves to
  the form's empty `edition_date` (parsed `MM/YY`; `label_rules.split_form_edition`), which changes training
  targets; a value with figures no page prints is listed in the metadata (`unprinted_values`) and the windows
  holding one are left out of the corpus. These rules hold on a common-model policy line only
  (`label_rules.label_rules_apply`), where serving applies them too; there a gold stored before the edition
  rule is read under it (`load_golden_label`), by the corpus build as by the gate and the harness. Each
  bundle's `metadata.json` carries its review record: a real document's own, a twin's its seed's
  (`review_inherited_from`; `reviews.csv` for real documents alone).
- **Golds and who drafted them** (decision 2026-10-19): frontier models are never used for gold, synthetic or
  training data — pre-annotation has only the self-hosted `base_qwen3vl` and `own_finetuned` backends, and a
  frontier backend asked for by name is refused. **Every gold needs a review record** (decision 2026-10-22):
  a real `reviewer_id` and an ISO `review_date` not in the future, read from its metadata alone
  (`export_golden_labels.review_problem`); a twin inherits its seed's. The import, the pre-upload audit, the
  preflight, the corpus build, the freeze and the gate's golden eval refuse a gold without one; the harness
  extracts it but does not score it.
- **What a window is taught** (`policy_windows`): its target cites only its own pages; a common-model row
  travels only with a printed value on those pages, its ids renumbered from 1 and a reference to a row the
  window cannot see left out (counted as dangling); a coverage row is taught only on windows that show its
  **home** — its name, a limit's or deductible's amount or percentage, or its premium (`HOME_FIELDS`) — so
  windows of policy wording that print only a caption are not taught the row; a coverage's form numbers only
  where its pages print them; `text_sections` never.
- **Splits**: group-level, band per type, placement per line; lines under 5 documents train whole (§10). A
  drawn split holds one carrier out per line naming 10 or more carriers (lines under the minimum are recorded
  as `lines_below_carrier_minimum`; on a policy only real carriers count — a placeholder such as "N/A" is no
  carrier, and a carrier any synthetic family trains is never held out); a delivered split is used as given (no carrier hold-out by default); twin
  cap 20 per seed per render mode;
  rows carry `carrier`, `twin_index`, `render_mode`, `is_scanned`. The corpus build counts each answer with
  the base model's tokenizer (`cap_check.target_tokens`; a 1.25-padded estimate where it is not at hand), sets
  aside a document whose row exceeds its budget, and refuses a build that sets aside more than 25% (the same
  helper the preflight reads).
- **The golden eval set is frozen once per tenant** at `golden-eval-set/{tenant}/` from that tenant's corpus
  test split (`python -m orchestration.run freeze-eval-set --corpus vN [--tenant T]`); refused twice, refused
  below 150 test documents per type (arch §15.4) unless `--allow-small`, and a set left at the old unscoped
  root is refused with the move to make. A gold with no review record is refused before anything is copied;
  each frozen document carries its record, and the gate refuses a frozen gold without one
  (`freeze-eval-set --mark-reviewed` copies records into a set frozen earlier). Later builds exclude its
  documents and their families and split new documents into train/val only.
- **Scopes** (`configs/scopes.yaml`): `unified`, `policy`, `lossrun`, and two line-scoped policy adapters.
  `personal_lines` covers homeowners, personal_auto (classic auto is read as personal auto), dwelling_fire,
  ocean_marine, motorcycle, recreational_vehicle and personal_umbrella — **not flood**, and not a package of
  common-model lines; `configs/training/personal_lines.yaml` adds its data mix. `casualty_fleet` covers `gl`
  alone, trained on the CGL documents in their own tenant: serving routes by a release's lines, so `auto`,
  `wc` and `umbrella` join only once they are on the common model and in the corpus. A line-scoped release
  refuses a policy of another line, or with no `lob`.
- A policy's line is a canonical schema name (`flood`, `gl`, `cyber` …), validated against the lines that
  have a schema — not against the 12-value LOB enum, which is for labels that carry `line_of_business`.
- Render-only and zero-page documents are skipped from training; a build with no train rows is refused,
  and a real build with no val rows, or no test rows while the eval set is not yet frozen.
- **Purge plans** report and never delete (`python -m data_pipeline.ingestion.pull_raw_pdfs --purge-plan
  SOURCE_ID`, `--tenant-purge-plan`): a document's raw, processed, label and frozen-eval-set prefixes, the
  tenant's exports, the delivered copies in `intake/` and on the pod's volume, the corpus versions to rebuild,
  and every run trained on it — with its weights, staged copy, eval report, calibrator and releases, "needs
  retrain or delete" — since trained weights are not tenant-prefixed yet; a tenant's plan covers every prefix
  in `paths.TENANT_SCOPED`. Every prefix ends in `/`, so `policy_1` never reaches `policy_10`. A run that
  continued from a checkpoint, or a per-type adapter on a listed Foundation, is listed too; a manifest that
  cannot be read is listed "check by hand", naming the error type only.

### 13.3 Training, selection, merge, packaging
- ms-swift 3 CLI; staged rows in ms-swift's own format on the pod volume; before launch the real tokenizer
  and resize rule measure every row against `max_length` and each task's output reservation, and label
  masking is verified on encoded rows. Data-parallel on every GPU the pod has (`distributed.gpus: auto`) at
  the same effective batch; `length_grouped_sampling: false` (with one row per device and `padding_free` it
  saved nothing, and made ms-swift encode every row before step 1); a resume point every 70 steps.
- **In-training loss checks** (10 per run) read `evaluation.loss_sample_rows: 160`; loss only draws the
  progress curve and never selects what ships.
- **Checkpoint selection** (`evaluation/checkpoint_eval.py`): every saved candidate generates on the fixed
  validation sample (`validation_sample_rows: 400`, all three modes); the best two
  (`selection_document_finalists: 2`) are then scored on **whole validation documents** — windows merged by
  the serving merge — in `selection_document_modes: [ocr_plus_image]` only, and the best document accuracy
  wins among finalists whose rates of invented values and of printed values left empty are within
  `selection_guard_margin: 0.02` of the best finalist's (`choose_on_documents`). With two or more GPUs each
  candidate scores in its own worker and vLLM engine (`evaluation/checkpoint_score_worker.py`).
- **Generating a split** (`evaluation/validation_generation.py`): `generation.validation_batch_rows: 96` rows
  per vLLM call, longest expected answer first; a call that fails is retried in halves until the bad row is
  alone; a mode filter that matches no row is refused before an engine loads. Each pass reports what it cost.
- **Calibration** is fitted on windowed policies merged and held to the label conventions as they are
  served, on one validation half, thresholds on the other; the calibrate stage reports expected calibration
  error per field type.
- PEFT `merge_and_unload` on the GPU into the pinned local base.
- **More windows per policy** since every page is read (decision 2026-10-19): a longer training run, and a
  corpus rebuild and retrain of both adapters before the next gate.
- **`package`**: quantize → calibrate → evaluation gate on the tenant's frozen set through the serving
  pipeline → push. The gate pairs the candidate's per-document scores with the promoted release's
  (`pipeline_dag.paired_document_scores`) when both were scored on the same frozen set of the tenant, so
  non-inferiority is a paired bootstrap, weighted as each metric counts its documents, and the improvement
  rule is enforced (`--fixes-defect "<ticket>"`, per scope, stands in for an improvement and nothing else;
  a blank or placeholder value is refused). Reports that cannot be paired get a point comparison, with the
  reason in the decision's warnings. The release bundle
  records the gate's metrics and tier (**production** at field match ≥ 0.92, **interim** below, or whenever
  the scope was scored on fewer than 150 frozen documents of a type it covers; none when blocked), latency P95 per adapter and peak GPU
  memory (recorded, not gated: no latency SLA is agreed), and a routing table built before anything is
  pushed: a line that cannot be routed, or a line the release trained for that would not reach it, stops the
  release. The published adapter is
  `adapter_config.json` + `adapter_model.safetensors` only (also for `finetune --push-adapters`); adapter and
  merged-model pushes skip files whose MD5 Blob already records. A model card (`registry_utils/model_card.py`)
  is written beside every run manifest.
- **Run visibility**: `common/logging_setup.py` keeps our log lines visible for the whole run; every stage
  records `wall_seconds`; the run report `run-<command>-<scope>-<tenant>-<version>.json` is rewritten after
  every stage in `/workspace/logs` (§12).

### 13.4 The pod
- **Dependencies** (`pyproject.toml` groups; `requirements-*.txt` per pod, installed through
  `requirements-*.lock`; `scripts/setup_pod.sh <role>`): training pod = ms-swift 3.12.6 + torch 2.8.0 +
  transformers 4.57.6 + **vLLM 0.11.0** (checkpoint selection and calibration run vLLM in the same process) +
  flash-attn built against that torch; serving pod = the same **vLLM 0.11.0**; OCR pod = **MinerU 3.4.5**
  (`mineru[pipeline]`, plus `six`, which its OCR imports undeclared; wired in
  `data_pipeline/ocr/run_mineru.py`); quantization = `llmcompressor` in its own environment (its
  `datasets`/`transformers` ranges conflict with ms-swift and vLLM).
- **Base model** in `/workspace/models` (`model.local_dir`, or `FIDEON_BASE_MODEL_DIR`); a real launch is
  refused when it is configured and absent. Pinned to revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- **H200 SXM pods (one GPU or several) with a network volume at `/workspace`**, prepared by
  `scripts/pod_bootstrap.sh`: GPU (every GPU counted)/CUDA 12.8 driver/mount/150 GB checks, `.env` keys
  (`MINERU_TOOLS_CONFIG_JSON=/workspace/mineru.json`, `MINERU_DEVICE_MODE=cuda`,
  `MINERU_MODEL_SOURCE=local`; a Blob login, either `AZURE_STORAGE_ACCOUNT_URL` + `AZURE_STORAGE_SAS_TOKEN`
  (with `AZURE_RAW_SAS_TOKEN` for the raw container) or `AZURE_STORAGE_CONNECTION_STRING`, an unquoted value
  refused, `export` and indented lines included, half a SAS login listed as still to do), **two environments** on the volume (`/workspace/venv` for training and
  serving; `/workspace/venv-ocr` for MinerU, whose dependency set is kept apart from the training stack),
  MinerU's pinned pipeline weights (`scripts/download_mineru_models.sh`, which writes `mineru.json`; OCR
  refuses to run without it, and a MinerU 1.x config is refused), the pinned base model, the spike. OCR runs
  first in the OCR environment; `finetune` then finds nothing to OCR, and refuses with that command if
  documents are still pending.
- **Before a real run**: `python -m data_pipeline.ocr_check` (labelled values against the OCR text),
  `python -m orchestration.preflight` (GPU, packages, base model, config invariants, disk, Azure, OCR done,
  every label with a review record, annotation agreement reported, and an in-memory corpus build;
  `--skip-corpus` when `finetune` follows in the same chain), and
  `python -m orchestration.smoke_run` (tenant `smoke`, version `v0`, `--scope`; a few sources per split with
  their twins; select → import → OCR → check → finetune, never freezing or packaging).
- **Blob login** (`artifact_registry.blob_client.azure_from_environment`): a SAS token
  (`AZURE_STORAGE_ACCOUNT_URL` + `AZURE_STORAGE_SAS_TOKEN`) when both are set, else
  `AZURE_STORAGE_CONNECTION_STRING`; half a SAS login is refused. `AZURE_RAW_SAS_TOKEN` is the raw
  container's own token; without it the one token serves both containers, so it must be an account SAS.
- **Train/serve token parity is measured on the pod** (`tests/test_parity.py::test_template_token_parity`,
  GPU-marked; it skips without a GPU, the base weights, ms-swift and vLLM): the trainer's token ids against
  vLLM's prompt token ids, and each image's token count against the resize rule the sequence caps use.
- **GPU only** (`common/gpu.py`): training, merge, quantization, the vLLM engine, the HF backend and MinerU
  refuse to start without CUDA; models load with `device_map="cuda"`, never `"auto"` (which offloads to CPU).
- **Jobs survive a disconnect** (`orchestration/detach.py`, `scripts/pod_run.sh`): on the pod
  (`RUNPOD_POD_ID`, `/etc/rp_environment`, or Linux with `/workspace` mounted and a GPU) every long entry
  point re-launches itself in tmux and returns; logs and exit codes live in `/workspace/logs`. Nothing
  stops a job except `pod_run.sh stop <name>`, which asks for confirmation. A test fails if a new
  long-running entry point lacks the guard.

### 13.5 Evaluation, serving guards and routing
- **Typed scoring**: a field is compared by the type its schema declares
  (`evaluation.metrics.field_accuracy.values_agree`, `scoring_kind`); two numbers compare as numbers.
  **Grounding** (`common/grounding.py`) is one matcher for whether a value is printed and on which pages,
  shared by the label rules, the audit, the OCR-agreement feature and the hallucination metric, which looks
  for the printed value on the pages it cites. Reported, not gated: `page_ref` exact rate, an "invented"
  error class, `error_summary` (errors by class, line and field), `hallucination_rate` and
  `additional_fields_hallucination_rate`.
- **Common-model lines** are scored through `common.structural_ids.comparable_view`
  (`evaluation/metrics/common_model.py`): rows
  paired by what the page prints, values inside rows counted, plus `reference_accuracy`,
  `coverage_code_accuracy` / `coverage_code_recall` and additional-fields recall/precision.
- **Gate** (`evaluation/gating.py`): absolute floors plus non-inferiority, a paired bootstrap over the
  per-document scores the eval report keeps (`EvalReport.per_document_scores()`), weighted so it tests the
  metric the report states (`per_document_weights()`), and paired only on the same frozen set of the tenant;
  a point comparison, with the reason in the decision's warnings, when the reports cannot be paired or fewer
  than two documents measure a metric. Conditional metrics — exempt only
  from "was it measured" — include `table_f1` (floor 0.88, Loss Run scopes), `worst_line_field_match`
  (floor 0.85, once a line has 200 values scored) and `scanned_accuracy` (an eval set with no scans, such as
  the all-digital CGL originals, is not blocked on it; one the frozen baseline measured is required). The
  report adds a `real_only` section and the held-out carriers apart (`held_out_carrier_match`, not gated).
  The pilot's criteria take their thresholds from the same floors (`PILOT_FLOORS`, IMPL-15). A merged answer's
  schema validity is judged as serving returns it (`serving.pipeline.served_schema_verdict`). No ceiling on
  invented values is gated: `hallucination_rate` is reported, no spec states a ceiling, and the gate path does
  not measure it — an open item.
- **Serving guards**: `page_ref` is bounded to the pages a request shows (`common.schemas.with_page_bounds`);
  on a common-model line `serving/conventions.py` holds the merged answer to the label conventions against every page's text — a
  value no page prints is flagged `<path>:not_printed` (an extra field like it dropped on a born-digital
  document), a form's edition left in its number is moved to `edition_date` with the number's span, and an
  empty declarations field whose own label and a value of its type open a line of any page is flagged
  `<path>:likely_missed`. A coverage that may have lost its unit is flagged.
- **Confidence** reads the printed value: a string's own tokens (not its quotes), `raw`'s span, and OCR
  agreement on the page it cites (`calibration.features.page_agreement`: 1.0 there, 0.5 printed only
  elsewhere, 0.0 nowhere).
- **Routing**: serving goes through the release the plan chooses — its adapter and its calibrators. One
  promoted release: the engine serves its merged model. Several, or any line that would fall back: the engine
  loads the base with LoRA enabled and each release is its own LoRA, applied per request. An operator pin
  (rollback) outranks everything; then a policy reaches the release of its line's **layout family**
  (`configs/layout_families.yaml`); an unrestricted (`unified`/`policy`) release reads an unrouted policy
  before the base model does; otherwise the base model reads it with no LoRA against `_fallback.json`,
  flagged, with `lob_fallback_used` in the result's route. Line detection ships off (§1.3).

## Changes since 2026-10-05

- Added the alignment line; the architecture reference now names `finetuning-architecture-v2.3.docx`.
- §0.1: added the Fideon SPEC_05, SPEC_06 §9a / SPEC_09 §2.1a, SPEC_09 and SPEC_21 references the code now implements.
- §1.1/§1.2/§1.3: policies select a per-line canonical schema (the client's files, read-only); the LOB enum is 12 values and `line_of_business` a list; policy line aliases and merged lines (classic auto, watercraft); policy line detection exists but ships off.
- §2: decoder-only LoRA (mergers not targeted); one adapter per scope incl. `personal_lines` and `casualty_fleet`; locked versions; confidence on the printed value's tokens and cited page; SPEC_21 common model in the schema contract; MinerU 3.4.5 pipeline backend with pinned weights; tenants in use.
- §3/§9/§10: per-epoch modality draw replaces "3 rows per document"; row metadata keys; extraction output carries `extraction`, `route`, list-valued `line_of_business`, `all_fields` in test outputs; delivered split, carrier hold-out off, twin cap, render rotation, data mix.
- §4: per-tenant `golden-eval-set/{tenant}/`, scoped train views, scoped registry with model cards, `exports/`, two-file published adapter, MD5 skip on push; the full tenant-scoped prefix list; `intake/` and `exports/` in the raw container; tenants actually in use.
- §8: the de-identification blocker names the decoder LoRA (the mergers are not trained).
- §5: repository tree regenerated from the current file list (122 test modules); `Documentation/` is in `.gitignore` (already-tracked files stay tracked).
- §6: the IMPL-06 and IMPL-10 one-line summaries (one adapter per scope; FP8/AWQ rather than GGUF).
- §7: logging via `common/logging_setup.py`; tenant scoping; MinerU pinning incl. weights, re-OCR and mixed-version refusal.
- §12: stage table corrected (the gate is in `package`, stages 8–11); real `extract` flags; `freeze-eval-set --tenant/--allow-small`; preflight and smoke run; one training job per scope; per-stage wall time and run report naming.
- §13: rewritten for Schema 2 / SPEC_21, label conventions and window teaching rules (incl. `HOME_FIELDS`), casualty_fleet, MinerU 3.4.5, the accuracy-plan measures and serving guards, checkpoint choice on whole documents, and the 2026-10-12/13 time cuts.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; the architecture reference is `finetuning-architecture-v2.4.docx`; a "What v2.4 changed" table of the decisions of 2026-10-19 (every page read, no latency SLA, no frontier model and reviewed golds, one carrier per line from 10 carriers, production tier kept at 0.92, GPU only). The v2.1 table's task list no longer names `page_select`.
- §0.1, §2, §8, §9: de-identification is Option A (none) as the architecture records, not "blocked"; the trainer fallback is a migration to TRL `SFTTrainer` (arch §10.1); `raw-documents/` is Cool tier, never Archive; no third-party pre-annotation; new §2 rows for page reading, gold and training data, latency and the release tier; the interleaving note and the `pages_used` example describe windows over every page.
- §5: `training/parity.py` in the tree; the 34 schema files counted as 31 lines plus `_fallback`, `_common` and `classic_auto`.
- §10: the carrier hold-out is one carrier per line, only in a line naming 10 or more carriers.
- §11 now sits between §10 and §12, so the sections run in order.
- §12: `--fixes-defect`; the gate's paired bootstrap; preflight checks reviewed labels.
- §13: every page read and the declarations group's first window; the form-edition label rule; the frontier ban and reviewed golds; the per-line carrier rule; purge plans; the paired-bootstrap gate, `--fixes-defect` and the interim cap for a small eval set; SAS login; token parity on the pod; preflight's label checks; the served schema verdict; pilot targets from `PILOT_FLOORS`; no invented-value ceiling (open item).

## Changes of 2026-10-22 (code review)

- Purge plans: prefixes end in `/`; the delivered copies in `intake/` and on the pod's volume; each run's staged copy, eval report and calibrator; lineage (`continued_from`, per-type adapters); unreadable manifests listed by error type.
- Blob login: `AZURE_RAW_SAS_TOKEN` for the raw container; the bootstrap's quoting and half-login checks.
- Every page: a window whose pages' own text would not fit its call is split (decision 2026-10-22); the corpus manifest pins `window_plan_version`, and training and packaging refuse a policy corpus built with another.
- Gate: per-document weights in the paired bootstrap; pairing only on the same frozen set and tenant; the written override on the command line; the interim cap per scope; `--fixes-defect` checked and per scope (IMPL-08, IMPL-13).
- Gate: a metric fewer than two paired documents measure is a point estimate, with a warning; an unenforced improvement rule, and why reports were not paired, are warnings in the decision (IMPL-08).
- Every gold needs a review record (decision 2026-10-22), in the v2.4 table, §2 and §13.2: a twin inherits its seed's; the freeze and the gate refuse a gold without one, and the harness does not score it.
- §13.2: the label rules, the form-edition rule among them, hold on common-model policy lines only (`label_rules_apply`); a stored gold is read under the edition rule by the corpus build as by the gate; bundles carry the review record.
- §13.2: on a policy the carrier hold-out counts real carriers only — a placeholder is no carrier, and a carrier a synthetic family trains is never held out; `freeze-eval-set --mark-reviewed`.
- §13.4 and §13.5: the preflight's review-record check; serving's conventions on common-model lines.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22; the current target (policy documents only) and a table of what v2.5 changed.
