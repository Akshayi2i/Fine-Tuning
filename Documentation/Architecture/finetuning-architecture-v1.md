> # ⚠ SUPERSEDED BY v2.1
>
> **Update 2026-09-27:** v2.1 now carries a *Revision v2.2 — Implementation Update* section (canonical LOB
> output with `MM/DD/YYYY` dates, policy windows for 200+ page documents, the frozen golden eval set, the
> per-line split, the pinned dependency stack, GPU-only model compute, tmux-detached pod jobs); the
> module-level detail is in `Implementation MDs/Fideon SPEC_00_MASTER_CONTEXT.md` §13.
>
> **This document is historical.** The current architecture is
> **`finetuning-architecture-v2.1.docx`**, and the codebase implements that one.
>
> Kept because the v2.1 revision tables reference what v1 said and why it
> changed — a decision log is worth less without the decision it reversed.
> **Do not implement from this file.** Its central claims are no longer true:
>
> | v1 said | v2.1 |
> |---|---|
> | Foundation LoRA + per-type adapters stacked | ONE unified LoRA, merged. vLLM applies one LoRA per request, so the stack was unservable |
> | `max_seq_len: 8192` for everything | Per-task caps, 4k–32k. 8192 could not hold a one-page ACORD with its schema, image, OCR and output |
> | LoRA on the vision projector (`merger`) | ViT **and** mergers frozen; arbitration is learned in the decoder |
> | Gate: beat previous on every metric within 0.001, no override | Absolute floors + paired-bootstrap non-inferiority + a named, recorded override |
> | Confidence: minimum token logprob, threshold 0.70 | Per-field-type feature calibrators, thresholds tied to a measured error bound |
> | Split at `source_id` | Split at group level — carrier, template, account |
> | GGUF (`q5_k_m`) as the serving target | bf16 → FP8 on vLLM; GGUF is an edge export |
> | Trainer: TRL `SFTTrainer` | ms-swift's own `Seq2SeqTrainer` on the HF Trainer — v1 was simply wrong about this |
> | 11 pipeline stages, gate before merge | 13 stages, gate after merge/quantize/calibrate |

- [Qwen3-VL-8B-Instruct Fine-Tuning Architecture](#qwen3-vl-8b-instruct-fine-tuning-architecture)
  - [Insurance Document Extraction (ACORD, Policy, Loss Run)](#insurance-document-extraction-acord-policy-loss-run)
  - [0. Pipeline Integration and Schema Contract](#pipeline-integration-and-schema-contract)
  - [1. Recommendations Summary](#recommendations-summary)
  - [2. Problem Restatement & Design Constraints](#problem-restatement-design-constraints)
  - [3. Qwen3-VL-8B Architecture Recap — and What It Means for Fine-Tuning](#qwen3-vl-8b-architecture-recap-and-what-it-means-for-fine-tuning)
  - [4. Adapter Strategy — Recommended: Hybrid (Foundation + Per-Type)](#adapter-strategy-recommended-hybrid-foundation-per-type)
  - [5. Confidence Score — Recommended: Logprob Extraction + Calibration](#confidence-score-recommended-logprob-extraction-calibration)
  - [6. Dual-Input-Mode Training (OCR+Image, and Image-Only)](#dual-input-mode-training-ocrimage-and-image-only)
  - [7. Dataset Format](#dataset-format)
  - [8. Data Corpus Management & Versioning](#data-corpus-management-versioning)
  - [9. Fine-Tuning Technique — LoRA on a bf16 Base](#fine-tuning-technique-lora-on-a-bf16-base)
  - [10. Trainer & Framework — The Three-Layer Stack](#trainer-framework-the-three-layer-stack)
  - [11. Hyperparameters](#hyperparameters)
  - [12. Fine-Tuning Cycle & Versioning Strategy](#fine-tuning-cycle-versioning-strategy)
  - [13. End-to-End Pipeline](#end-to-end-pipeline)
  - [14. RunPod Infrastructure — Training vs. Serving](#runpod-infrastructure-training-vs.-serving)
  - [15. Evaluation Framework — Metric Definitions](#evaluation-framework-metric-definitions)
  - [16. Pilot Validation Protocol — De-Risking Before Full Annotation](#pilot-validation-protocol-de-risking-before-full-annotation)
  - [17. Extraction / Testing Routine (Post-Fine-Tuning)](#extraction-testing-routine-post-fine-tuning)
  - [18. Azure Blob Folder Structure](#azure-blob-folder-structure)
  - [19. Project Repository Folder Structure](#project-repository-folder-structure)
  - [20. Summary Checklist](#summary-checklist)
  - [21. Glossary — What Each Tool Is, and Why It’s Here](#glossary-what-each-tool-is-and-why-its-here)

# Qwen3-VL-8B-Instruct Fine-Tuning Architecture

## Insurance Document Extraction (ACORD, Policy, Loss Run)

> **Revision — synced to the implementation spec set (IMPL-00 … IMPL-15).**
> This document and the specs in `Documentation/Implementation MDs/` are now in agreement. Changes carried in from implementation:
>
> | Change | Where |
> |---|---|
> | **Canonical field mapping** — surface labels to canonical keys, semantic glosses, alias registry | new §0c |
> | **Operator command surface** — three commands plus one umbrella | new §13c |
> | **RunPod staging volume** — persistence between build and package | §14, §18 |
> | **De-identification BLOCKED** — text-only Presidio corrupts the training signal | §8b |
> | **Deferred to post-pilot** — hyperparameter sweep, quantization threshold enforcement | §11a, §13b |
> | **Removed** — per-tenant adapter lineage, per-form ACORD adapter configs | §8b, §4b |
> | New metrics — alias accuracy, confusable misattribution | §15, §16 |
>
> **Maintenance rule:** the Implementation MDs carry build-level detail this document deliberately does not. When they disagree, the discrepancy is a bug in one of them — fix it rather than choosing a winner, and keep this banner current.

## 0. Pipeline Integration and Schema Contract

This architecture specifies the **L3 VLM layer** of the Fideon document extraction pipeline. L0 → L1 → L2 → L3 routing is defined in Fideon SPEC_01. This section states the contractual obligations between the VLM and the rest of the pipeline — everything downstream in this document assumes these contracts hold.

### 0a. Canonical Output Schema (Fideon SPEC_00)

The training target JSON is **not an internal format** — it is the canonical schema defined in Fideon SPEC_00 (`fideon/schemas/`). The schema registry in §7 is the Fideon SPEC_00 Pydantic model serialised to JSON Schema. **Any Fideon SPEC_00 schema change requires a corpus rebuild and a new training cycle.** The golden label JSON for every training example is validated against the Fideon SPEC_00 JSON Schema before being admitted to the corpus.

The target JSON for each document type maps directly to its Fideon SPEC_00 Pydantic model:

| Document type   | Fideon SPEC_00 Pydantic model |
|-----------------|------------------------|
| `loss_run`      | `LossRunDocument`      |
| `policy_check`  | `PolicyCheckDocument`  |
| `quote_gen`     | `QuoteGenDocument`     |
| `acord_mapping` | `ACORDMappingDocument` |

All field names, data types, nesting structure, and null representation must match Fideon SPEC_00 exactly. The audit gate (Fideon SPEC_07 Stage 3) validates every VLM output against the Fideon SPEC_00 schema on every inference call — so a schema drift between the corpus and Fideon SPEC_00 surfaces as a production validation failure, not as a silent quality regression.

**Two vocabularies, one mapping table.** This document uses the Fideon SPEC_00 canonical model names above, while §7, §17, §18, and §19 use short `doc_type` tags. These are the same things. The implementation resolves them in exactly one place (`common/constants.py`):

| `doc_type` (corpus, adapters, paths, CLI) | Fideon SPEC_00 canonical key | Fideon SPEC_00 model | Status |
|---|---|---|---|
| `lossrun` | `loss_run` | `LossRunDocument` | active |
| `policy` | `policy_check` | `PolicyCheckDocument` | active |
| `acord` | `acord_mapping` | `ACORDMappingDocument` | active (+ `acord_form`) |
| `quote` | `quote_gen` | `QuoteGenDocument` | deferred |

**Every field in every schema carries a `description`** — a semantic gloss with explicit exclusions. This is not documentation: the schema is injected into the system prompt, so the gloss is training input. See §0c.

### 0b. Line of Business (LoB) in VLM Output

L1 (carrier registry) and L2 (structural inference) both attempt LoB detection before the document reaches L3. When L3 is invoked — because L1/L2 missed, or because the document is scanned — **the VLM is expected to detect and output** `line_of_business` **as part of its extraction.**

Valid values follow the LOB enum in Fideon SPEC_00: `workers_comp`, `general_liability`, `commercial_auto`, `property`, `umbrella`. The field is output as `null` when LoB cannot be determined from the document.

    {
      "line_of_business": {"value": "workers_comp", "confidence": 0.92},
      ...
    }

LoB is explicitly represented in **every** training example — every golden JSON includes a `line_of_business` field, even when null. Corpus coverage target: at least **20% of training examples per LoB value** in the real document population — read as a floor on *starvation*, not as an exact quota. There are five non-null LoB values, so a flat 20% is precisely the uniform share and only a perfectly uniform corpus could satisfy it; the coverage check therefore applies three quarters of the uniform share, which scales if a value is added to the enum and never exceeds the configured target. A value with **no** documents is a different problem and is reported separately — there is nothing to be weak at and no per-value accuracy to measure. LoB extraction accuracy is reported as a dedicated metric in the evaluation framework (§15).

### 0c. Canonical Field Mapping — the VLM does the semantic work

The same real-world field appears under many surface labels. The party purchasing the policy shows up as *Insured Name*, *Named Insured*, *Applicant*, *Name of Applicant*, *Applicant of Insured*, or bare *Name*. **The golden JSON always uses the one canonical key** (`insured_name`), and so does the inference output.

**The contract:**

1. **Golden labels are canonical.** Annotators map whatever the document says onto the canonical key. Surface labels never appear as keys.
2. **Inference output is canonical.** The extraction returns `insured_name` regardless of the document's phrasing.
3. **The VLM performs the mapping.** This is core Foundation-LoRA behavior (§4) — recognising that *Applicant* here denotes the same field as *Named Insured* there.

This is not really a design choice: the Fideon SPEC_00 contract and the §0a audit gate require canonical keys downstream. Every alternative needs a mapping layer somewhere, and the model is the only component that can handle phrasings nobody anticipated — and the only one that works in **image-only mode**, where there is no OCR text for a rule to read.

**Three mechanisms, all required:**

| Mechanism | What it contributes |
|---|---|
| **Semantic gloss in the schema** — every field's `description` defines what it means **and what it excludes** | A semantic anchor ("find the party purchasing the insurance"), so unseen phrasings resolve. Also lifts base-model pre-annotation quality on day zero |
| **Corpus coverage across variants** — alias coverage tracked per canonical field | The mapping is learned from examples; a variant seen twice is learned weakly |
| **Confusable co-occurrence examples** — documents where a field and its confusables both appear | Teaches the **boundary**, not just the mapping |

**Write glosses as definitions with exclusions, never as label lists:**

    "insured_name": {
      "type": ["string", "null"],
      "description": "The party purchasing the insurance policy. Not the
                      certificate holder, the producer/agency, or an
                      additional insured."
    }

The exclusion clause is the load-bearing part. An alias list can only say what a field is *called*; a definition can say what it is *not* — which is the only thing separating `insured_name` from the four other name-shaped entities on the same page.

**Why alias lists do not go in the prompt.** They give the model a *lexical* prior that makes confusable errors worse (tell it `insured_name` may appear as "Insured" and a page containing "Additional Insured" substring-matches). They never close the long tail. And since schema text is prompt text, every newly discovered alias would trigger a corpus rebuild and retrain.

**The alias registry** (`schemas/aliases/{doc_type}.aliases.json`) records observed surface forms and their confusables. It is **derived from labeled data**, not hand-written: given (document, golden JSON) pairs, the golden JSON supplies the value, the OCR supplies the text, and the label is whatever introduces that value on the page. It feeds annotator guidance, corpus coverage, and evaluation slicing.

> **Anti-pattern — no runtime alias lookup, ever.** Mapping extracted labels to canonical keys with a lookup table at inference caps the system at the list someone wrote, cannot handle unanticipated phrasing, and defeats the reason for fine-tuning a VLM. The registry is **training, labeling, and evaluation material only.**

**The discrimination risk.** Insurance documents are dense with name-like fields — certificate holder, producer/agency, additional insured, loss payee, mortgagee, carrier. A corpus that only teaches "name-ish label → `insured_name`" collapses them, producing confident, well-formed, wrong extractions. **Confusable misattribution is therefore a gating metric** (§15).

## 1. Recommendations Summary

You asked me to make the call on two open design questions. Here's the recommendation, with reasoning expanded in the relevant sections below.

| Decision              | Recommendation                                                                                                                                                                                 |
|:----------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Adapter strategy      | **Hybrid: one shared "Foundation LoRA" + small per-document-type adapters stacked on top**                                                                                                     |
| Confidence score      | **Token-logprob extraction + post-hoc calibration** (not a separate verifier model, not pure self-reported)                                                                                    |
| Fine-tuning technique | **LoRA on a bf16 base** (LoRA on LLM decoder + vision-language projector; ViT frozen *initially*, unfrozen via an eval gate if scanned/image-only accuracy demands it). 4-bit QLoRA stays available behind `load_in_4bit` for VRAM-constrained pods — see §9.2 |
| Trainer stack         | **ms-swift (entrypoint) → TRL** `SFTTrainer` **(training loop) → PyTorch/PEFT/bitsandbytes/DeepSpeed (foundation)**                                                                            |
| Versioning strategy   | **Foundation retrained from the raw HF base model on major corpus expansions; per-type adapters always retrained fresh from the current Foundation, not from the previous adapter checkpoint** |
| RunPod usage          | **Two separate concerns — ephemeral training pods, and a persistent Serverless vLLM inference endpoint. Your orchestration/business logic lives outside RunPod and calls the endpoint.**       |

### 1a. Architecture Summary — Locked Specification

The table above records *decisions*; this one records the resulting **committed specification** in the form the implementation team works from:

| Component                  | Specification                                                                                                                                                                                                                                           |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base model                 | `Qwen/Qwen3-VL-8B-Instruct`, pinned Hugging Face revision                                                                                                                                                                                               |
| Fine-tuning technique      | LoRA — bf16 frozen base, LoRA adapters in bf16. Both serving paths hold the base in bf16/fp16, so training in bf16 means the adapter is applied to exactly the weights it trained against (§9.2). 4-bit NF4 (QLoRA) remains available via `load_in_4bit` |
| Trainable components       | Vision-language projector and LLM decoder via LoRA. Vision Encoder (ViT) frozen by default, unfrozen only through the evaluation gate in §3                                                                                                             |
| Adapter strategy           | Shared Foundation LoRA across all document types and modality regimes, with per-document-type LoRA adapters stacked on top                                                                                                                              |
| Trainer stack              | ms-swift entrypoint over TRL `SFTTrainer`, on PyTorch, PEFT, bitsandbytes, Accelerate/DeepSpeed                                                                                                                                                         |
| Confidence                 | Per-token logprob extraction with post-hoc calibration; list fields carry an additional row-completeness signal                                                                                                                                         |
| Quantization               | GGUF, user-selectable format (`fp16`, `bf16`, `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m`)                                                                                                                                                                       |
| Artifact storage           | Azure Blob Storage for all weights, corpora, and source documents                                                                                                                                                                                       |
| Compute                    | RunPod — ephemeral training pods plus a persistent Serverless vLLM inference endpoint                                                                                                                                                                   |
| Line of Business detection | VLM outputs `line_of_business` per the Fideon SPEC_00 LOB enum when L1/L2 fail to detect it. Values: `workers_comp`, `general_liability`, `commercial_auto`, `property`, `umbrella`; `null` when undetermined. Corpus coverage target ≥20% per LoB value (§0b) |
| Schema contract            | Target JSON is the Fideon SPEC_00 canonical schema; audit gate Fideon SPEC_07 Stage 3 validates every inference call (§0a)                                                                                                                                            |
| Tenant isolation           | Corpus partitioned by `tenant_id` per Fideon SPEC_12; Foundation trained only on Presidio de-identified data per Fideon SPEC_11 (§8b)                                                                                                                                 |

Active document types are ACORD, Policy, and Loss Run, each with a distinct target JSON schema. Two inference modes are supported by a single model: **OCR-plus-image**, which supplies MinerU OCR text alongside the page image, and **image-only**, which supplies the page image without OCR.

## 2. Problem Restatement & Design Constraints

- Document types in scope: **ACORD forms (25, 125, 140, etc.), Policy documents, and Loss Runs** — each with a **distinct target JSON schema**. *(Quote and Endorsement are on the overall roadmap but deferred; the current build and the testing routine in §17 cover the three active types. The Foundation + per-type adapter design extends to them later without rework.)*
- Input PDFs: both **digital/native** and **scanned**.
- Two production modes must both work from **one model**:
  1.  MinerU OCR output (markdown/raw text) + original page image, both passed in.
  2.  Image only, no OCR — for cases where OCR is skipped or unavailable.
- Output: strict JSON matching a per-document-type schema, **with a confidence score per extracted field (or per document)**.
- Infra: RunPod GPUs for compute-heavy work, Azure Blob for all artifact storage (adapters, merged models, quantized models).
- You explicitly want the "why," not just the "what" — so each section includes the underlying reasoning tied to Qwen3-VL's actual architecture.

## 3. Qwen3-VL-8B Architecture Recap — and What It Means for Fine-Tuning

Referencing the official architecture diagram you shared:

| Component                            | What it does                                                                                                                                                                            | Fine-tuning implication                                                                                                                                                                                                                                                                                       |
|:-------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Vision Encoder (ViT)**             | Native-resolution input — patch count scales with image size (e.g., ~11,427 tokens for a full-resolution screenshot vs. ~8 tokens for a tiny 32×256 image in the diagram's own example) | Image token count is **not fixed** — it's a direct function of page resolution. This is your single biggest cost/latency lever. You must standardize a resolution cap per page (see §11).                                                                                                                     |
| **DeepStack**                        | Injects multi-level ViT features into multiple LLM decoder blocks (not just the final layer)                                                                                            | This is a *pretrained* fusion mechanism already tuned for fine-grained visual detail (small text, table gridlines, checkboxes) — exactly what insurance forms need. Freezing the ViT does **not** disable DeepStack; the injected features are still useful. No need to unfreeze ViT just to "activate" this. |
| **Interleaved-MRoPE**                | Full-frequency positional encoding across time/width/height, mainly built for video                                                                                                     | For static multi-page PDFs this mostly matters for **multi-image ordering** (page 1 vs page 2 of the same document) — keep page order consistent and explicit in your input structure.                                                                                                                        |
| **Vision-Language Projector/Merger** | Maps ViT output dimensionality into the LLM's embedding space                                                                                                                           | This is the layer literally responsible for **fusing** image evidence with the language model's reasoning. It's small, cheap to train, and is the highest-leverage LoRA target for teaching "trust the image over the OCR text when they disagree."                                                           |
| **Qwen3 LM Decoder (Dense, 8B)**     | Autoregressive generation over the fused token stream                                                                                                                                   | This is where schema-following, field-naming discipline, and JSON structure get learned. Primary LoRA target: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.                                                                                                                                |

**Net architectural decision — staged, not absolute:** start with the ViT frozen, apply LoRA to the projector + full LLM decoder. But this is a **staged decision with an explicit escalation gate**, because two of your requirements push directly against a permanently-frozen ViT:

- **Scanned / low-quality PDFs** — degraded faxes, skewed scans, and low-DPI documents demand more from the visual pathway than clean digital renders. The pretrained ViT may not read them well enough out of the box.
- **Image-only mode (no OCR)** — this is the critical case. When there's no OCR text, the ViT *is* your OCR — it's the only thing reading characters off the page. In OCR+image mode a weak ViT is backstopped by MinerU's text; in image-only mode there is no backstop.

So the freeze is a **starting hypothesis**, not a permanent commitment. The escalation gate:

    1. Train Foundation with ViT frozen.
    2. Evaluate specifically on: (a) the image-only eval subset, (b) the scanned-PDF eval subset.
    3. IF image-only OR scanned accuracy is below target
       AND the errors are perception errors (misread characters, missed checkboxes)
       rather than schema/reasoning errors:
          → unfreeze the ViT (or apply LoRA to it) and retrain Foundation.
    4. ELSE keep it frozen (cheaper, faster, and already sufficient).

The distinction in step 3 matters: if the model reads the character correctly but puts it in the wrong JSON field, that's an LLM/projector problem and unfreezing the ViT won't help. Only unfreeze when the evidence shows the *visual reading itself* is the bottleneck. This keeps you from paying the cost of ViT training unless your own eval data proves you need it — which is the right default given the ViT's strong document pretraining, while still honoring the reality that scanned + image-only extraction may demand it.

**When the ViT *is* trained, LoRA is applied to it — full fine-tuning of the vision encoder is not used.** Full fine-tuning risks degrading the encoder’s pretrained document and OCR capabilities, which are the very capabilities the image-only pathway depends on. The escalation is therefore “add a ViT LoRA,” not “unfreeze and train the encoder.”

## 4. Adapter Strategy — Recommended: Hybrid (Foundation + Per-Type)

### Why not a single unified adapter

ACORD 25, a Loss Run, and a Policy Declaration page have structurally different schemas, layouts, and failure modes (e.g., Loss Runs are dense repeating tables; ACORD forms are checkbox/field-grid heavy; Policy documents are long-form prose with embedded schedules). A single LoRA trained across all of them risks:

- Cross-type interference (gradients from one schema nudging behavior on another).
- All-or-nothing redeployment — fixing a Loss Run bug means retraining and re-validating everything, including document types that were already working fine.

### Why not fully separate adapters per type (no shared foundation)

- You'd re-learn "generic insurance document understanding" (terminology, table parsing, checkbox reading, OCR-error correction patterns) from scratch for every type — wasteful given you have 1000+ examples per type.
- Slower to bring up a **new** document type in the future (e.g., a Binder or Certificate of Insurance) — you'd start from zero instead of inheriting a strong base.

### Recommended structure: Foundation → Per-Type

    Qwen3-VL-8B-Instruct (frozen base, bf16)
            │
            ▼
    Foundation LoRA  (rank 64–128, trained on ALL doc types combined)
      - Insurance terminology & abbreviations
      - Table / grid / checkbox reading
      - OCR-error correction behavior (image vs. text arbitration)
      - JSON structural discipline (valid syntax, null-handling, schema obedience)
      - Modality-dropout training (OCR+image AND image-only — see §6)
            │
            ▼
    Per-Type LoRA  (rank 16–32, small, trained on top of Foundation)
      - ACORD-25-Adapter, ACORD-125-Adapter, PolicyDoc-Adapter, LossRun-Adapter
        (Quote / Endorsement adapters added later when those types are activated)
      - Only learns the final schema-mapping specialization for that type

**At inference:** a lightweight document-type classifier (can be a cheap text/image classifier, or a zero-shot prompt to the Foundation model itself) picks the right per-type adapter, which gets stacked on top of the Foundation for that request. This is supported natively by PEFT's multi-adapter loading and by vLLM's LoRA hot-swap serving.

**Why this wins for your situation specifically:** you have 1000+ examples per type — enough to make per-type adapters genuinely learn sharp, specialized behavior — but you also have real value in shared cross-type patterns (OCR arbitration behavior, table reading) that a Foundation layer captures once instead of re-learning per adapter. It also directly answers your versioning question: adding a new document type later (Quote, Endorsement, or anything else) means training one small new adapter on top of the existing Foundation, not retraining everything.

### 4a. Document-Type Classifier — a Load-Bearing Component, Specified

Both the adapter routing above and the testing routine (§17) depend on **correctly identifying the document type first**. This is not a minor preprocessing step — it's load-bearing: if classification is wrong, you load the wrong adapter *and* the wrong prompt *and* the wrong schema, and the extraction fails no matter how good the model is. It deserves explicit design rather than being hand-waved as "a cheap classifier."

**Routing flow:**

    document ──> classifier ──> confidence >= threshold ? ──> yes ──> select adapter + prompt + schema
                                         │                              (acord -> + form number)
                                         └── no ──> Foundation-only extraction + human routing review

**What it is:** a dedicated classifier that maps an input document → one of {acord, policy, lossrun} (see ACORD sub-type note below). Three viable implementations, in increasing cost/accuracy:

| Option                                    | How                                                                                                                                                                                                 | Trade-off                                                                                                       |
|:------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
| **A. Text-feature classifier**            | Lightweight model (e.g., logistic regression / small transformer) over MinerU's OCR text — insurance doc types have strong lexical signatures ("Loss Run", "ACORD 25", "Declarations", "Quotation") | Cheapest, fast, but fails in image-only mode (no OCR text) and on ambiguous/mislabeled headers                  |
| **B. Vision classifier**                  | Small image classifier over the first-page render — doc types have distinct visual layouts                                                                                                          | Works in image-only mode; needs labeled layout data; robust to missing/garbled headers                          |
| **C. Zero-shot via the Foundation model** | Prompt the already-loaded Foundation model to name the doc type before extraction                                                                                                                   | No extra model to train/serve; reuses the model you already have; slightly higher latency (an extra generation) |

**Recommendation:** start with **Option C** (zero-shot via Foundation) for the pilot — it's zero extra infrastructure and the Foundation model already understands these document types. Move to a dedicated **Option B (vision) classifier** if classification accuracy becomes a measured bottleneck, because it works in both OCR and image-only modes. Whichever you pick, the classifier is a **first-class component with its own requirements**:

- **Its own eval + accuracy target** — measured on the golden eval set, reported alongside extraction metrics. A classifier at 92% caps your whole system at 92% regardless of extraction quality.
- **A low-confidence fallback path** — when the classifier isn't confident, don't silently guess. Fall back to **Foundation-only extraction** (no per-type adapter, using a generic schema-agnostic prompt) and flag the document for human routing review. A wrong-adapter extraction is worse than a slightly-generic one.
- **Runs before adapter selection** in both the serving path and the testing routine (§17 step 2).

### 4b. ACORD Sub-Types — Resolve the Granularity Explicitly

"ACORD" is not one form. ACORD 25 (Certificate of Liability), 125 (Commercial Application), 140 (Property), etc. have genuinely different layouts and field sets. This is a real fork you must decide, because it cascades into the classifier granularity, the schema registry, and the adapter count:

- **Option 1 — One ACORD adapter, multiple schemas:** a single `acord` adapter, but the classifier identifies the specific form number and selects the matching schema/prompt. Simpler adapter management; relies on one adapter generalizing across ACORD layouts.
- **Option 2 — Per-form ACORD adapters:** `acord25`, `acord125`, `acord140` each get their own per-type adapter on top of Foundation. Sharper specialization; more adapters to train and version.

**Recommendation:** start with **Option 1** (one ACORD adapter + per-form schemas) for the pilot, because with limited data per specific form, one adapter learning shared "ACORD-ness" generalizes better than several data-starved per-form adapters. **Per-form adapter configs are not created** until a form crosses the threshold below; the per-form *schemas* do exist, because the classifier must select the right schema regardless. Split to **Option 2** for any specific form once you have 1000+ examples of that form *and* evaluation shows the unified adapter underperforming on it. Either way, the **classifier must identify the specific form number** (not just "ACORD") so the correct schema is always selected — so treat ACORD form detection as a two-level classification: type = acord, then form = 25/125/140/…

### 4c. Day-Zero Bootstrap — Before a Foundation Model Exists

The recommendation above defaults to zero-shot classification *via the Foundation model* — but on day zero no fine-tuned Foundation adapter exists yet. The bootstrap path used during the initial labeling sprint (§16) is:

1.  Use the **base** `Qwen3-VL-8B-Instruct` with a zero-shot classification prompt to identify document type and ACORD form number.
2.  Run extraction with the **base** model using the schema-injected prompts from §7.
3.  **All pre-annotation output from the base model receives 100% human review** — no draft is accepted without correction until at least **25 labeled examples per document type** exist.

Once Foundation v1.0 is trained and promoted, it becomes the zero-shot classifier and the base model is retired from the production inference path. This path is explicitly temporary: it applies only during the initial labeling sprint and is superseded on promotion of Foundation v1.0.

## 5. Confidence Score — Recommended: Logprob Extraction + Calibration

**End-to-end confidence flow:**

    Generation ──> token logprobs ──> map to field spans ──> aggregate per field
                                                                  │
                                                            raw confidence
                                                                  │
                                                  post-hoc calibration transform
                                                  (temperature / isotonic, per type)
                                                                  │
                                                         calibrated confidence
                                                           │              │
                                                  scalar fields      list fields
                                                           │              │
                                                  threshold check   + row-completeness
                                                           │              │
                                                           └──> review routing <──┘

### The mechanism

1.  At generation time, request `logprobs` for every output token (supported by vLLM and HF `generate(output_scores=True)`).
2.  Map each field's value span in the generated JSON back to its underlying tokens.
3.  Aggregate per field — a reasonable default is the **minimum token probability within the span** (most sensitive to the weakest link, which is what you want for flagging risky fields); the exact aggregation (min vs. mean vs. geometric mean) is an implementation tuning detail to settle empirically, not an architectural decision.
4.  This gives you a raw confidence score per field, at zero extra training or inference cost.

### The problem with raw logprobs

Fine-tuned generative models — especially after LoRA fine-tuning on a narrow task — tend to become **overconfident**: correct and incorrect outputs often carry similarly high token probabilities, because the model has learned to be fluent in the target format even when it's wrong about the content.

### The fix: post-hoc calibration

After training, run inference on a **held-out labeled validation set** (not used in training) and compare raw confidence scores against actual field-level correctness (exact match / normalized match). Fit a calibration function — **temperature scaling** (simple, one parameter, works well for this) or **isotonic regression** (more flexible, handles non-monotonic miscalibration) — per field or per document type. Store this as a small lookup/transform applied at serving time, after the model call, before the JSON is returned to your application.

### Why not a separate verifier model

A second model that re-reads the extraction and scores it adds real cost: another model to train, version, evaluate, and keep in sync with every Foundation/adapter version, plus doubled inference latency. It's a reasonable escalation path **if** calibrated logprobs prove insufficient in evaluation (see §15) — but start with the cheap, well-understood approach first.

### Output shape

    {
      "policy_number": {"value": "ABC-1234567", "confidence": 0.94},
      "effective_date": {"value": "2026-01-01", "confidence": 0.97},
      "named_insured": {"value": "Acme Manufacturing LLC", "confidence": 0.81},
      "line_items": [
        {"description": "General Liability", "premium": 4500.00, "confidence": 0.62}
      ]
    }

Low-confidence fields (below a tuned threshold, e.g. 0.7) are exactly what you route to human review in production — this is the practical payoff of doing calibration properly.

### List fields need a second confidence signal (missed rows have no logprobs)

Per-field logprob confidence works for scalar fields, but **list fields** (Loss Run claims, line items, ACORD schedule rows) have a failure mode that logprobs can't see: a **missed row**. If the model extracts 6 of 8 claims, the 2 missing claims have no generated tokens, so there's no low probability to flag — the per-field confidence on the 6 rows it *did* extract can all be high while the extraction is silently incomplete. This is a recall failure, and token confidence is blind to it.

Handle list fields with **two confidence signals, not one:**

1.  **Per-value confidence** (as above) — for the fields within each extracted row.
2.  **Row-count / completeness confidence** — a separate signal for "did we get all the rows?" Practical approaches: cross-check the extracted row count against a count derivable from the document (e.g., a "total claims: N" field or the number of table rows MinerU detected), and/or calibrate a separate confidence for list completeness against ground-truth row counts on the validation set. When the model's row count disagrees with the document's own stated/detected count, flag the whole list for review regardless of per-value confidence.

This matters specifically for Loss Runs, where a missed claim row is both easy to make and expensive to miss. Note it in the Loss Run schema/eval design and measure list-field **recall** (not just per-value accuracy) in the eval framework (§15).

## 6. Dual-Input-Mode Training (OCR+Image, and Image-Only)

Both production modes must be learned by the **same** Foundation adapter — this is a modality-dropout training strategy, not two separate models.

### Training mix (within the Foundation corpus)

| Regime                                                              | Share | Purpose                                                     |
|:--------------------------------------------------------------------|:------|:------------------------------------------------------------|
| Image + real, clean-ish MinerU OCR text                             | ~50%  | Baseline dual-modality behavior                             |
| Image + deliberately-imperfect real OCR text, corrected target JSON | ~20%  | Teaches image-over-OCR arbitration when they disagree       |
| **Image only, no OCR text block at all**                            | ~30%  | Directly satisfies your "must work without OCR" requirement |

### Implementation detail

The system prompt should explicitly declare which mode is active rather than silently omitting the OCR block:

    system: ... "OCR text: [not provided — extract directly from the image]" ...

vs.

    system: ... "OCR text is provided below; use it as your primary source for dense text 
    and numbers, and use the image to verify layout and correct OCR errors." ...

This explicit signal — rather than an ambiguous missing field — is what lets the model reliably switch behavior between modes instead of guessing which mode it's in.

## 7. Dataset Format

### Chat-format JSONL, one example per line

    {
      "doc_type": "loss_run",
      "modality_mode": "ocr_plus_image",
      "source_id": "lossrun_2024_0091_p1",
      "messages": [
        {
          "role": "system",
          "content": "You are an insurance document extraction system. Document type: Loss Run. Extract into this exact JSON schema: {schema injected here}. OCR text is provided; use it as primary source and the image to verify/correct. Assign a confidence 0-1 per field internally via your best judgment is NOT required — output only the JSON values, confidence is computed externally from your generation."
        },
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "lossrun_2024_0091_p1.png"},
            {"type": "text", "text": "<page 1 of 3>

<page 1 MinerU markdown>"},
            {"type": "image", "image": "lossrun_2024_0091_p2.png"},
            {"type": "text", "text": "<page 2 of 3>

<page 2 MinerU markdown>"},
            {"type": "image", "image": "lossrun_2024_0091_p3.png"},
            {"type": "text", "text": "<page 3 of 3>

<page 3 MinerU markdown>"}
          ]
        },
        {
          "role": "assistant",
          "content": "{\"carrier\": \"...\", \"policy_number\": \"...\", \"claims\": [...]}"
        }
      ]
    }

**Page pairing.** The user turn interleaves each page's image with that page's own markdown, prefixed `<page N of M>`. It is not a formatting preference — the earlier "all images, then one joined text block" shape carried three defects:

| Defect | Consequence |
|---|---|
| No page boundary in the joined text | The §6 evidence rule *(trust the image where they disagree)* needs to know **which** image. On a long policy the model cannot recover that by content matching, so the rule silently degrades to guessing. |
| A blank line ends a Markdown table | Joining page N to page N+1 split every table crossing a page boundary in two, and the second half lost its header. This hit Loss Runs and long policies hardest — the documents where row completeness is the whole point. |
| Page routing sent one call per page | A routed request asked for a complete document-level JSON from a single page, a shape no training row ever had, and prevented the model from seeing that a table on page 9 continues on page 14. |

Pairing fixes all three, and makes a routed request a **subsequence of the full document** — `[img_1][txt_1][img_9][txt_9]` — rather than a different structure. Selected pages therefore go in **one** call, not one per page, so the §7 per-page merge step is no longer on the serving path; the declarations-precedence rule it implemented is stated to the model directly in the Policy prompt block instead, which keeps it in one place rather than two that can disagree.

`image_only` carries the marker and no markdown. The mode's guarantee is *no OCR text*, not *no text at all*; the marker is document metadata, and a gap in the numbering is what tells the model that absent fields live on pages it was not shown.

**Unverified dependency:** that ms-swift and vLLM both accept interleaved image/text content lists for Qwen3-VL. The model family supports interleaved multimodal input; confirm it in the §16 pilot's dependency spike before the first corpus build.

### Required record fields

Every JSONL record carries the following top-level keys, in addition to `messages`:

| Field           | Purpose                                                                                                                    |
|-----------------|----------------------------------------------------------------------------------------------------------------------------|
| `doc_type`      | `acord` \| `policy` \| `lossrun` — drives prompt, schema, and adapter selection                                            |
| `acord_form`    | ACORD form number (`25`, `125`, `140`, …) or `null` for non-ACORD types — required for the two-level classification in §4b |
| `modality_mode` | `ocr_plus_image` \| `noisy_ocr_image` \| `image_only` (§6)                                                                 |
| `source_id`     | Join key back to `raw-documents/`, `processed/`, and `golden-labels/`                                                      |
| `split`         | `train` \| `val` \| `test` — assigned at source-document level before modality expansion (§8)                              |

Loss is computed **only** on the assistant tokens; system, image, and OCR-text tokens are masked with label `-100`.

### Schema registry

Maintain each document type's JSON schema as its own versioned file (`schemas/acord25.schema.json`, `schemas/lossrun.schema.json`, ...) and inject it into the system prompt at dataset-build time — never hand-type the schema per example. This keeps schema changes a one-file edit that regenerates the whole corpus consistently.

### System prompt template — versioned alongside the schema

The system prompt is constructed at dataset-build time by injecting the versioned schema and the modality instruction. **Training-time and inference-time prompts must be identical** — divergence between them is the single most common cause of post-fine-tuning performance degradation, and it is invisible in training metrics because it only manifests at serving time.

    --- OCR-plus-image and noisy-OCR mode ---
    You are an insurance document extraction model. Extract all fields from the
    provided {document_type} document and return a single valid JSON object
    conforming exactly to the schema below. The OCR text is the primary source;
    use the page image to verify and correct OCR errors. Where a field is absent
    from the document, output null.

    Schema: {schema_json}

    --- Image-only mode ---
    You are an insurance document extraction model. Extract all fields from the
    provided {document_type} document images and return a single valid JSON
    object conforming exactly to the schema below. No OCR text is provided;
    extract directly from the images. Where a field is absent, output null.

    Schema: {schema_json}

The prompt template is versioned alongside the schema registry in the corpus manifest. **Changes to the template require a corpus rebuild and a new training cycle, identical to a schema change.**

### Required edge cases in the corpus

- Multi-page documents (multiple `image` blocks + concatenated OCR text in one example)
- Rotated / skewed scans, low-quality faxes
- Handwritten annotations on otherwise digital forms
- Fields legitimately absent (must produce `null`, not hallucinate)
- Repeating table rows of variable length (Loss Run claims, ACORD schedule lines)

### Long / Multi-Page Documents — Context-Window Strategy (Policy Docs Especially)

Policy documents can run 50+ pages. Each page at full resolution can cost 1,000–2,000+ vision tokens, so a naive "put every page image + all OCR text in one prompt" approach will blow even Qwen3-VL's large context window on long policies — and waste enormous compute on pages that contain no extractable fields. This needs an explicit strategy, not silent reliance on the context limit:

**Recommended: relevant-page selection + page-scoped extraction, then merge.**

1.  **Page routing** — for a multi-page document, first identify which pages contain the target fields. Most policy schemas draw from a minority of pages (declarations page, schedule pages, specific endorsement pages). Use a cheap first pass — keyword/section detection over MinerU's per-page OCR text, or a lightweight page-classifier — to select the pages that matter for the schema.
2.  **Scoped extraction** — run the model only on the selected pages (their images + OCR), not the whole document. This keeps each inference within a comfortable context budget and cuts cost dramatically.
3.  **Merge** — combine per-page/per-section extractions into the final document JSON, with a defined conflict-resolution rule when the same field appears on multiple pages (e.g., declarations page wins for policy-level fields).

**Simpler fallback for short docs:** ACORD forms and most Loss Runs are short (1–few pages) — for these, single-pass whole-document extraction is fine and the routing step is skipped. Apply the page-selection strategy only when a document exceeds a page-count threshold (e.g., \>5 pages). Record which pages fed each extraction in the output metadata, so a low-confidence field can be traced to the specific page it came from.

This is primarily a **Policy-document** concern; note it explicitly in the Policy schema/prompt design and build long multi-page policies into the corpus and eval set so the strategy is actually tested, not assumed.

### Worked Example: Mapping Your Pilot PDFs (e.g. 30 ACORD / 30 Loss Run / 30 Policy)

Walking one document through every layer, using ACORD document \#1 as the example (`source_id = acord_0001`):

    1. Original PDF (untouched, immutable)
       azure-blob://insurance-extraction/raw-documents/acord/acord_0001/original.pdf
       azure-blob://insurance-extraction/raw-documents/acord/acord_0001/metadata.json

    2. MinerU OCR output + rendered page image(s)
       azure-blob://insurance-extraction/processed/acord/acord_0001/
           page_1.png                    # resolution-capped, matches training + inference config
           page_1.md                     # MinerU markdown/raw text output
           (page_2.png, page_2.md, ... if multi-page)

    3. Golden JSON (human-verified target output)
       azure-blob://insurance-extraction/golden-labels/acord/acord_0001/
           golden.json                   # the verified target JSON for this document
           label_metadata.json           # annotator id, review date, source of pre-annotation, agreement score

    4. Compiled training example(s) — ONE source document generates MULTIPLE
       JSONL rows, one per modality-mode variant (§6), all referencing source_id: "acord_0001"
       azure-blob://insurance-extraction/corpus/v1/acord/train.jsonl
       →  {"doc_type": "acord", "modality_mode": "ocr_plus_image",   "source_id": "acord_0001", "messages": [...]}
       →  {"doc_type": "acord", "modality_mode": "noisy_ocr_image",  "source_id": "acord_0001", "messages": [...]}
       →  {"doc_type": "acord", "modality_mode": "image_only",       "source_id": "acord_0001", "messages": [...]}

The same `source_id` — `acord_0001` — appears identically at every layer (`raw-documents/`, `processed/`, `golden-labels/`, and every generated row in `corpus/`). That's the join key that lets you go from a training example all the way back to the original PDF, and it's what makes the split-leakage rule below enforceable.

For a pilot batch across the three active types, that gives you:

    raw-documents/acord/acord_0001..00NN/
    raw-documents/lossrun/lossrun_0001..00NN/
    raw-documents/policy/policy_0001..00NN/

...each producing 1 `processed/` entry, 1 `golden-labels/` entry, and (with the 3-regime modality split from §6) roughly 3 compiled JSONL rows per document — so N source PDFs yield on the order of ~3N training examples across the `corpus/v1/{doc_type}/` files, split across `train.jsonl`/`val.jsonl`/`test.jsonl` per the strategy below.

### Golden JSON Generation Mechanism (When You Don't Have Labels Yet)

For an unknown/unlabeled PDF, don't hand-type JSON from scratch — use a **bootstrap pre-annotation + human review** workflow, which is both faster and produces more consistent labels than starting blank:

1.  **OCR pass** — run MinerU to get markdown/text + rendered page image(s), same as production (`processed/`).
2.  **Draft ("silver") JSON via pre-annotation** — generate a first-pass JSON automatically, using one of:
    - The **base, not-yet-fine-tuned** Qwen3-VL-8B-Instruct with a well-crafted zero-/few-shot prompt containing the schema (works reasonably for early batches).
    - A stronger frontier model (e.g., GPT-4V/Claude with vision) as a one-time pre-annotator if base Qwen3-VL's zero-shot accuracy on insurance forms is too rough to be a useful starting point — this is a one-time bootstrap cost, not a permanent dependency. **PII caveat:** insurance documents contain PII (names, TINs/SSNs, addresses, financials). Sending them to a third-party API may violate your compliance posture or data-processing agreements — **confirm this is permitted before using any external model for pre-annotation**, and prefer a zero-retention/enterprise API tier or on-prem model if there's any doubt. If external pre-annotation isn't permissible, use the self-hosted base Qwen3-VL for bootstrapping instead, accepting rougher first-pass drafts.
    - **Once you have a fine-tuned v1**, switch pre-annotation to your own model — it will already outperform generic zero-shot on your document types, which compounds: each cycle's labeling gets faster than the last.
3.  **Human review and correction** — a reviewer with insurance domain knowledge compares the draft JSON against the actual PDF (side-by-side, in a labeling tool such as Label Studio or Argilla, or a lightweight custom review UI) and corrects every field. This produces the actual **golden** JSON — the draft is never trusted as-is.
4.  **Double-annotation on a sample** — for roughly 10–20% of documents, have a second reviewer label independently, then adjudicate disagreements. This gives you an inter-annotator agreement measure, which tells you how much noise/ambiguity exists in your own labeling process — useful context when the model's eval scores plateau below 100%, since some of that ceiling is human disagreement, not model error.
5.  **Store provenance, not just the label** — `label_metadata.json` (per the worked example above) should record who reviewed it, when, which model produced the draft, and the agreement score if double-annotated. This matters later for debugging — if a document type's accuracy regresses, you want to be able to check whether the *labels* for that batch were lower quality, not just suspect the model.
6.  **Prioritize review effort using confidence, once you have a model in the loop** — after v1 exists, run it over new unlabeled PDFs and use the calibrated confidence score (§5) to route: high-confidence extractions get a light spot-check, low-confidence fields get full manual review. This is the active-learning loop referenced in §13 step 11, and it's what makes labeling cost drop over successive corpus versions instead of staying flat.

## 8. Data Corpus Management & Versioning

    azure-blob://insurance-extraction/corpus/
      v1/
        lossrun/{train,val,test}.jsonl
        acord25/{train,val,test}.jsonl
        ...
      v2/
        ... (expanded corpus, new edge cases, new examples)

- Each version is immutable once used for a training run — never edit in place.
- Maintain one **frozen golden evaluation set**, human-double-verified, held constant across corpus versions so you can compare model versions apples-to-apples over time (see §15).
- New labeled data appends into the next corpus version; a lightweight metadata manifest (`manifest.json` per version) records example counts per type/modality-mode for auditability.

### Train/Val/Test Split Strategy — and Why Not a Flat 70/20/10

**Two rules matter more than the exact percentages:**

1.  **Split at the source document level, before modality-mode expansion — never after.** Because each source PDF generates 3 JSONL rows (OCR+image, noisy-OCR, image-only — §6), splitting *after* expansion risks the same document appearing in both `train` and `val`/`test` under a different modality mode. That's data leakage — the model would effectively be evaluated on a document it already saw during training, inflating your eval metrics without reflecting real generalization. Always split `source_id`s first, then generate all 3 modality variants for each `source_id` into whichever split it landed in.
2.  **Split ratio should depend on how much data you actually have per type, not a fixed default.** A flat 70/20/10 is a fine default at scale, but it breaks down at small volumes:

| Data volume per doc type                    | Recommended split                                                                             | Why                                                                                                                                                                                                                                                                                                                                               |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Small pilot batch (~25–30/type)**         | ~70% train / ~18% val / ~12% test, i.e. roughly **17-21 train / 4-5 val / 3-4 test** per type | At this size, a strict 70/20/10 leaves only 2–3 test documents per type — too few to trust any single metric. Treat this batch as a **bootstrap/pilot run**: enough to validate the pipeline end-to-end (data format, training loop, eval harness all work), but not enough to draw strong conclusions about model quality per document type yet. |
| **200–1000 examples/type**                  | 75/15/10 or 80/10/10                                                                          | Val/test sets are now large enough to give stable metrics; shift a bit more into train since the pipeline is already validated.                                                                                                                                                                                                                   |
| **1000+ examples/type** (your target state) | 80/10/10                                                                                      | Standard split; 10% val and 10% test are both large enough at this volume to be statistically meaningful, and you want to maximize train data once the process is proven.                                                                                                                                                                         |

**My call for your situation specifically:** don't over-invest in getting the pilot batch's split ratio perfect — its real job is proving the pipeline works (folder structure, JSONL generation, collator masking, training loop, eval harness all functioning correctly end-to-end) before you scale up labeling to 1000+/type. Use roughly 70/18/12 for this batch, treat its eval numbers as directional rather than final, and re-baseline properly once you're at real scale.

### 8a. MinerU Version Pinning

The MinerU version used to preprocess each corpus version is **recorded in the corpus manifest and pinned for that version**. The inference pipeline must use the same MinerU version as the training corpus.

- When MinerU is upgraded, affected documents are reprocessed and the corpus version is incremented **before** the next training cycle.
- Serving a model trained on MinerU v{n} output against documents processed by MinerU v{n+1} constitutes **distribution shift**, and is treated as a regression trigger — not as a routine dependency bump.
- Additional corpus manifest fields: `mineru_version` (string, e.g. `"1.4.2"`) and `preprocessing_date` (ISO 8601 timestamp, per document).

This matters because the model is fine-tuned partly on *how MinerU formats its output* — table markdown conventions, reading order, error patterns. A silent OCR upgrade changes the input distribution the model was trained to arbitrate against.

### 8b. Multi-Tenant Corpus Isolation

Raw documents contain insurance PII. Corpus partitioning follows the tenant isolation model defined in **Fideon SPEC_12 (Multi-Tenant Deployment)**:

- Corpus paths are prefixed by broker `tenant_id`:

      azure-blob://insurance-extraction/corpus/{tenant_id}/v{n}/{doc_type}/{split}.jsonl

- **Cross-tenant mixing is prohibited** — a model trained on Broker A data must never contain examples from Broker B.

- The corpus manifest records contributing `tenant_id`s and de-identification status per example.

**Scope for the current build.** The `tenant_id` path prefix is reserved so no migration is needed later, but the system runs **single-tenant**: `tenant_id` defaults from configuration and is not a required argument on every CLI. The one rule that is live and enforced is **no cross-tenant mixing in a corpus file**, because corpus composition is training data. **Per-tenant adapter lineages are not built** until a broker actually requires one.

> ### De-identification — BLOCKED, not skipped
>
> The original requirement here was that the shared Foundation LoRA train only on Presidio de-identified data (Fideon SPEC_11). **Do not implement it as specified.** It de-identifies **text** but says nothing about the **page images** the vision encoder reads, and that combination corrupts the training signal:
>
> | Regime | Share of Foundation corpus | What the model is taught |
> |---|---|---|
> | `image_only` | **30%** | Image shows "John Smith"; target says `PERSON_1`. **The target is not derivable from the input** — 30% of the corpus becomes unlearnable examples and pure hallucination pressure |
> | `ocr_plus_image` | **50%** | OCR says `PERSON_1`, image says "John Smith", target says `PERSON_1` → teaches **trust-OCR-over-image**, the exact inverse of what the projector LoRA and the 20% noisy-OCR regime exist to teach (§3, §6) |
>
> **Half-de-identifying is worse than either extreme.** Two coherent resolutions, to be chosen by the compliance owner:
>
> 1. **Redact page images consistently with the text** — input and target then agree and the arbitration signal survives. Costs an image-redaction pipeline this architecture does not currently specify.
> 2. **De-identify nothing; rely on tenancy plus access control** — training data keeps real values and the input/target contract stays coherent. Requires explicit sign-off that PII in `corpus/` carries the same posture as `raw-documents/`.
>
> Until resolved, the corpus manifest records `deidentified: false` and `image_redaction: "unresolved"`, PII protection rests on access control and tenancy, and **that limitation is stated to the compliance owner rather than left implicit**. Whichever resolution is chosen, one rule holds: de-identification must be consistent across the OCR text, the page image, and the target JSON.

## 9. Fine-Tuning Technique — LoRA on a bf16 Base

**Two decisions, argued separately.** An earlier revision of this section justified "QLoRA" with a single bullet titled *why QLoRA over full fine-tuning* — but every clause in it (behavior adaptation rather than knowledge injection, a fraction of the VRAM, many training cycles) argued for **low-rank adaptation**, and none of it distinguished a 4-bit base from a bf16 one. The 4-bit half was never actually argued. It is now, and it came out the other way.

### 9.1 Why LoRA rather than a full fine-tune

This task is behavior/format adaptation (schema discipline, OCR-vs-image arbitration) on top of a model that already has strong document/OCR pretraining — not new-domain-knowledge injection. A full fine-tune of an 8B model needs roughly 16 GB of weights plus 16 GB of gradients plus 32–64 GB of optimizer state; low-rank adaptation reaches the same place on a fraction of that, with much faster iteration, which matters when a cycle is one Foundation run plus three per-type runs.

### 9.2 Why a bf16 base rather than 4-bit (QLoRA)

**The default is LoRA on a bf16 base.** `configs/base_model.yaml` sets `quantization.load_in_4bit: false`.

1. **Train/serve alignment — the decisive reason.** *Both* serving paths hold the base in bf16/fp16: the merged model (§13 step 7, IMPL-10) and vLLM's LoRA hot-swap (§11, IMPL-11). An adapter trained against a 4-bit base learns a delta that partly compensates for quantization error in weights it is then **never served against**. Training in bf16 removes that gap: the adapter is applied to exactly the weights it saw. This compounds — every per-type adapter stacks on a Foundation that would otherwise carry it.
2. **ZeRO-3 becomes a usable escape hatch.** Under LoRA, ZeRO-2 shards ~1.5 GB of adapter optimizer state and is close to a no-op; ZeRO-3 shards the ~16 GB base, which is where the saving actually is. Two GPUs put ~8 GB of base on each — better headroom than holding the whole base in 4-bit on one card. bitsandbytes 4-bit params do not partition cleanly under stage 3, which is precisely what made this escape hatch awkward under QLoRA.
3. **Faster steps.** NF4 weights dequantize on every matmul; expect 20–40% slower steps under QLoRA. (The old §9 claim of "much faster iteration" was true against *full fine-tuning*, and backwards against LoRA.)
4. **One fewer approximation in the optimizer.** At rank 64 only ~150–200M params train, so fp32 AdamW state costs ~1.4 GB against ~0.35 GB for 8-bit. Affordable — so the optimizer is `adamw_torch`, not `paged_adamw_8bit`.

**4-bit remains a live flag, not a deleted path.** Set `load_in_4bit: true` to save ~11 GB of base weights (~16 GB → ~5.5 GB), accepting the mismatch and the slowdown. It is the right answer on a VRAM-constrained pod, and the manifest records which way the run went (`technique`, `base_quantization`), so the two are never confused after the fact.

**Per pod class** (peak VRAM, Foundation, rank 64 — *estimates pending the §9.3 measurement*):

| Pod | QLoRA (NF4) | LoRA (bf16) | Verdict |
|:----|:------------|:------------|:--------|
| A100 80 GB (Foundation) | ~22 GB | ~33 GB | bf16 comfortable, ~47 GB spare |
| L40S 48 GB (per-type) | ~22 GB | ~33 GB | bf16 fine, ~15 GB margin |
| A100 40 GB (per-type) | ~22 GB | ~33 GB | **tight** — prefer L40S, ZeRO-3, or `load_in_4bit: true` |

Base weights are 16 GB in bf16 against ~5.5 GB in NF4; adapter, gradients and optimizer state add ~1.5 GB; the rest is activations, dominated at `max_seq_len: 8192` by the LM-head logit tensor (Qwen3's ~151k vocab × 8192 positions is ~2.5 GB in bf16 and roughly double when cross-entropy upcasts to fp32) plus ~2.4 GB of layer-boundary checkpoints and the frozen ViT forward at 1792 px.

### 9.3 Open item — measure before trusting the table

The activation term above is arithmetic, not a measurement, and it is the term that decides whether the 40 GB pod still works. `scripts/phase0_spike.py` records the card's total VRAM but never peak allocation during a step; `check_model_loads` is where a `torch.cuda.max_memory_allocated()` probe over one forward/backward at full `max_seq_len` belongs. Two further reasons this stays open: `max_seq_len: 8192` is itself a placeholder pending the Phase 10 re-measurement on the real corpus (a higher value shrinks the margin), and the §3 ViT escalation adds trainable params on top.

**Worth measuring once, not assumed:** the actual field-F1 delta between a 4-bit and a bf16 Foundation. The evaluation framework and promotion gate already exist, so the cost is one extra Foundation run. The gap is currently assumed to be zero, which is the one thing it certainly is not.

### 9.4 Settled parameters

- **Base precision:** bf16 frozen base weights, LoRA adapters in bf16. 4-bit NF4 with double quantization available via `load_in_4bit`.
- **Frozen (initially):** entire Vision Encoder (ViT) — subject to the escalation gate in §3. If image-only or scanned-PDF evaluation shows the visual reading itself is the bottleneck, unfreeze the ViT (or apply LoRA to it) and retrain.
- **LoRA targets:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in every LLM decoder layer, plus the vision-language projector/merger layer.
- **LoRA rank/alpha:**
  - Foundation: rank 64, alpha 128 (broader behavior space, needs more capacity)
  - Per-type: rank 16, alpha 32 (narrow specialization, small and fast)
- **LoRA dropout:** 0.05
- **Attention implementation:** `flash_attention_2` — essential given multi-image, high-token-count inputs.

### 9a. LoRA Target Modules — What Each One Does and Why It’s Targeted

LoRA is applied to the attention and feed-forward projection matrices in each decoder layer, plus the vision-language projector. Each target serves a distinct function, and the selection is not arbitrary:

| Target                    | Function and fine-tuning effect                                                                                                                                                                                                 |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `q_proj`                  | Query projection in self-attention. Produces the query vectors that determine which other tokens each position attends to. Adapting it changes **what the model looks for** when reading the fused image and text sequence.     |
| `k_proj`                  | Key projection in self-attention. Produces the key vectors matched against queries to compute attention weights. Adapting it changes **how tokens are matched**, which affects alignment between field labels and their values. |
| `v_proj`                  | Value projection in self-attention. Produces the value vectors aggregated according to attention weights. Adapting it changes **what information is carried forward** once attention has been assigned.                         |
| `o_proj`                  | Output projection in self-attention. Recombines the attention heads into the model dimension. Adapting it changes **how the multi-head result is integrated** into the residual stream.                                         |
| `gate_proj`               | Gating projection in the feed-forward network. Controls the gated activation selecting which features pass through the block. Adapting it changes **nonlinear feature selection** per token.                                    |
| `up_proj`                 | Up projection in the feed-forward network. Expands the hidden representation into the larger intermediate dimension. Adapting it changes **the feature space** in which per-token transformations occur.                        |
| `down_proj`               | Down projection in the feed-forward network. Contracts the intermediate representation back to the model dimension. Adapting it changes **how expanded features are projected back** into the residual stream.                  |
| Vision-language projector | Maps Vision Encoder output into the LLM embedding space. Adapting it changes **how image evidence is fused** with the language representation — the primary locus of OCR-versus-image arbitration.                              |

In short: adapting **attention** projections changes how the model routes and aligns information across the sequence; adapting **feed-forward** projections changes per-token feature transformations; adapting the **projector** changes cross-modal fusion. Together these cover exactly the three behaviors fine-tuning must learn — schema adherence, field alignment, and modality arbitration. The Vision Encoder is excluded by default under the policy in §3.

## 10. Trainer & Framework — The Three-Layer Stack

**Clarifying the common confusion:** “SFTTrainer” and “ms-swift” are **not competing choices** — they’re different layers of one stack. `SFTTrainer` is not skipped; it sits underneath the framework you invoke.

    Layer 3   ms-swift                          ← what you actually invoke (CLI / config)
                  │   wraps and configures ↓
    Layer 2   TRL SFTTrainer                    ← the real training loop
                  │   (optimizer, backprop, gradient accumulation,
                  │    checkpointing, eval hooks, early stopping, DeepSpeed/Accelerate)
                  │   built on ↓
    Layer 1   PyTorch + Transformers + PEFT + bitsandbytes + Accelerate/DeepSpeed

When you run an ms-swift training command, it constructs a TRL `SFTTrainer` loop — with the Qwen3-VL chat template, LoRA/QLoRA wiring, and the multimodal data collator already set up. You get the full training loop, including DeepSpeed and Accelerate integration, callbacks, and resumable checkpoints, without hand-writing the boilerplate.

### The Decision — Locked Stack

This is the committed choice for the project, one option per layer (no open alternatives):

| Layer                       | Chosen                                                                  | Role                                                                                                                                                                         | Why this one                                                                                                                                                                                                                                                                                                                      |
|-----------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Layer 3 — Entrypoint**    | **ms-swift (ModelScope-Swift)**                                         | What you invoke; builds config, launches training, manages adapters                                                                                                          | Most current first-class Qwen3-VL multimodal support; natively handles interleaved image+text collation, the `-100` label masking, and multi-adapter (Foundation + per-type) workflows — a direct match to this architecture. Chosen over LLaMA-Factory to standardize on one framework and avoid the raw-collation masking risk. |
| **Layer 2 — Training loop** | **TRL** `SFTTrainer`                                                    | Optimizer step, backprop, gradient accumulation, checkpointing, eval hooks, early stopping                                                                                   | Not a separate decision — it’s what ms-swift builds on. `SFTTrainer` brings a mature, well-tested training loop with DeepSpeed/Accelerate integration, callbacks, and resumable checkpoints for free.                                                                                                                             |
| **Layer 1 — Foundation**    | **PyTorch · Transformers · PEFT · bitsandbytes · Accelerate/DeepSpeed** | Tensors/autograd (PyTorch), model + Qwen3-VL classes (Transformers), LoRA adapters (PEFT), optional 4-bit base (bitsandbytes, only under `load_in_4bit`), multi-GPU/sharding (Accelerate + DeepSpeed ZeRO) | Standard, battle-tested stack that every Layer-2/3 tool sits on; fixed regardless of anything above it. |

**What this means concretely:**

- You invoke **ms-swift** (CLI/config). It internally runs TRL `SFTTrainer`, which runs on the **PyTorch/PEFT/bitsandbytes/DeepSpeed** foundation.

- `training/train_foundation.py` and `training/train_adapter.py` are thin wrappers that assemble the ms-swift config (model, corpus version, LoRA params, base precision, DeepSpeed config) and launch it — not custom training loops. Base precision comes from `training/base_precision.py`, one helper shared by both so they cannot drift apart on how the base is held.

- The **data collator** (system/image/OCR tokens masked to `-100`, loss only on assistant JSON) is provided by ms-swift; you don’t hand-write it. `training/data_collator.py` exists only as an override hook if you ever need to customize masking behavior.

- **Fallback clause:** the only condition under which you’d drop to invoking TRL `SFTTrainer` directly at Layer 3 is if ms-swift lacks support for a specific Qwen3-VL capability you need at implementation time. That’s a contingency, not a parallel option — the committed path is ms-swift.

- **Distributed training:** DeepSpeed ZeRO-2 (or ZeRO-3 if VRAM-constrained) for multi-GPU RunPod instances — configured through ms-swift, executed at the Accelerate/DeepSpeed layer, independent of the Layer-3 choice. Note what ZeRO-2 does *not* do: under LoRA it shards only ~1.5 GB of adapter optimizer state and is close to a no-op. ZeRO-3 shards the bf16 base and is the stage that actually saves VRAM (§9.2).

## 11. Hyperparameters

**These are starting points for a sweep, not settled values.** The specific numbers below (especially learning rates and epoch counts) are sensible initializations based on LoRA norms for this model size and task — but treat them as the center of a small hyperparameter search, tuned against your validation set, not as fixed truth. The Foundation-vs-per-type differentiation reflects the right *direction* (per-type builds on a stable base, so lower LR / more epochs on less data), but the exact ranges should be confirmed empirically.

| Parameter                        | Foundation phase                                                                                                                      | Per-type phase                                                     |
|:---------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
| Learning rate                    | 1e–4 to 2e–4                                                                                                                          | 5e–5 to 1e–4 (lower — building on a stable base)                   |
| LR schedule                      | Cosine, warmup ratio 0.03–0.05                                                                                                        | Same                                                               |
| Epochs                           | 2–3 over full mixed corpus                                                                                                            | 3–5 (smaller data, can afford more passes)                         |
| Early stopping                   | On val loss + field-level F1, patience 2 evals                                                                                        | Same                                                               |
| Weight decay                     | 0.01                                                                                                                                  | 0.01                                                               |
| Effective batch size             | 32–64 (via grad accumulation)                                                                                                         | 16–32                                                              |
| Gradient checkpointing           | On                                                                                                                                    | On                                                                 |
| Max image resolution (long side) | Cap at 1536–2048px                                                                                                                    | Same — **must match** whatever cap you use in production inference |
| Max sequence length              | Set from your 95th-percentile token count across image + OCR text + JSON output (measure this on your actual corpus before fixing it) | Same                                                               |

**On resolution capping specifically:** the architecture diagram you shared shows a single high-resolution image can cost over 11,000 vision tokens. For dense insurance forms you want enough resolution to read small print and checkboxes clearly, but an uncapped native-resolution policy will blow out both training cost and inference latency unpredictably across a mixed corpus of scan qualities. Standardize a resolution cap, apply it identically in training data prep and production inference preprocessing, and only increase it if evaluation shows small-text/checkbox misses tied to resolution.

### Full parameter specification

The summary table above gives the shape of the decision; the tables below are the complete configuration the training entrypoints assemble.

**Core optimization**

| Parameter         | Foundation                     | Per-Type                       |
|-------------------|--------------------------------|--------------------------------|
| Learning rate     | 1e-4 to 2e-4                   | 5e-5 to 1e-4                   |
| LR scheduler      | Cosine                         | Cosine                         |
| Warmup ratio      | 0.03 to 0.05                   | 0.03 to 0.05                   |
| Epochs            | 2 to 3                         | 3 to 5                         |
| Optimizer         | `adamw_torch` (fp32 state)     | `adamw_torch` (fp32 state)     |
| Adam β₁           | 0.9                            | 0.9                            |
| Adam β₂           | 0.999                          | 0.999                          |
| Adam ε            | 1e-8                           | 1e-8                           |
| Weight decay      | 0.01                           | 0.01                           |
| Max gradient norm | 1.0                            | 1.0                            |

**Batch and memory**

| Parameter                   | Foundation                                                     | Per-Type                              |
|-----------------------------|----------------------------------------------------------------|---------------------------------------|
| Per-device train batch size | 1 to 2                                                         | 1 to 2                                |
| Gradient accumulation steps | Set to reach the effective batch size                          | Set to reach the effective batch size |
| Effective batch size        | 32 to 64                                                       | 16 to 32                              |
| Gradient checkpointing      | Enabled                                                        | Enabled                               |
| Mixed precision             | bf16                                                           | bf16                                  |
| Max sequence length         | 95th percentile of image + OCR + JSON tokens across the corpus | Same                                  |

**LoRA**

| Parameter         | Foundation                                                                      | Per-Type                                     |
|-------------------|---------------------------------------------------------------------------------|----------------------------------------------|
| Rank (r)          | 64                                                                              | 16                                           |
| Alpha             | 128                                                                             | 32                                           |
| Dropout           | 0.05                                                                            | 0.05                                         |
| Target modules    | Attention and feed-forward projections plus the vision-language projector (§9a) | Same                                         |
| Bias              | none                                                                            | none                                         |
| Base precision    | bf16 frozen base (`load_in_4bit: false`, §9.2)                                  | Same                                         |

**Vision and sequence**

| Parameter                        | Value                                                          |
|----------------------------------|----------------------------------------------------------------|
| Max image resolution (long side) | 1536 to 2048 px, standardized across training and inference    |
| ViT trainable                    | `False` by default, subject to the §3 gate (LoRA, not full FT) |
| Attention implementation         | `flash_attention_2`                                            |

**Evaluation and checkpointing**

| Parameter                | Value                                                  |
|--------------------------|--------------------------------------------------------|
| Evaluation strategy      | Per fixed step interval                                |
| Early stopping criterion | Validation loss and field-level F1                     |
| Early stopping patience  | 2 evaluations                                          |
| Save strategy            | Per fixed step interval, retaining the best checkpoint |
| Metric for best model    | Field-level F1                                         |
| Random seed              | Fixed and recorded per run                             |
| Logging interval         | Per fixed step interval                                |

### 11a. Hyperparameter Sweep Methodology

> **Deferred until after the pilot (§16).** A 9–12 run sweep against 25–30 documents per type mostly measures noise — §8 itself calls pilot metrics directional. The sweep configurations are authored up front so the protocol is settled, but execution belongs *after* the pilot passes and *before* production-scale training. The wording below already scopes it to "before the first **production** run".

The values above are starting points, and the sweep that turns them into settled values is itself specified so it doesn’t become an unbounded search:

**Phase 1 — Learning rate** (highest impact, run first). Hold all other parameters fixed. Sweep `{5e-5, 1e-4, 2e-4}` for Foundation and `{2e-5, 5e-5, 1e-4}` for per-type. One epoch per candidate. Metric: validation loss. Budget: 3 runs per adapter type.

**Phase 2 — Epochs.** With the best learning rate fixed, sweep `{2, 3, 4}` for Foundation and `{3, 4, 5}` for per-type, with early stopping (patience 2). Metric: field-level F1 on the validation set. Budget: 3 runs per adapter type.

**Phase 3 — LoRA rank** (only if F1 plateaus). Sweep rank `{32, 64, 128}` for Foundation. Secondary sweep, not run by default. Budget: 3 runs.

**Sweep management:** Weights & Biases Sweeps or MLflow hyperparameter tracking. Each sweep run produces a full `run_manifest.json` (§12), so sweep runs are first-class registry entries rather than untracked side experiments. The best configuration by validation field-level F1 is promoted as the production training run. Total budget: approximately **9 to 12 training runs before the first production run**.

## 12. Fine-Tuning Cycle & Versioning Strategy

This directly answers your question: *"should we take v1 as the base for training new data, or always start from the original base model?"*

**Rule: it depends on which layer you're retraining, and how big the change is.**

    Major corpus expansion ──> retrain Foundation from HF base on full corpus
    Minor patch ──────────────> continue Foundation, mandatory regression test

      foundation-v{n}
           │  (each adapter always retrained fresh from current Foundation)
           ├──> acord-adapter-v{n}
           ├──> policy-adapter-v{n}
           └──> lossrun-adapter-v{n}

    Foundation advances ──> all dependent adapters re-validated and retrained

### Foundation LoRA

- **Major corpus expansion** (significant new data volume, new edge cases, or a new document type folded into the shared foundation): retrain from the **original HF base model**, using the full accumulated corpus (old + new data together), not by continuing training on top of the previous Foundation checkpoint.
  - *Why:* continued training on top of an existing LoRA compounds drift across cycles — small biases and overfitting artifacts from v1 get baked in and amplified rather than corrected when v2 trains on top of it. Starting fresh from base with the full corpus each major version is slower per-cycle but far more reproducible and easier to debug when something regresses.
- **Minor incremental patch** (small batch of new examples fixing a narrow gap, no structural corpus change): continuing training on top of the current Foundation checkpoint is acceptable, **but** must be validated against the frozen golden eval set for regression on the *other* document types before promotion — never assume a patch is safe without measuring it.

### Per-Type Adapters

- **Always retrain fresh from the current Foundation version** using the full accumulated per-type dataset — never continue-train a per-type adapter on top of its own previous checkpoint.
  - *Why:* per-type adapters are cheap and small (rank 16–32, hours not days to train), so there's no real cost advantage to incremental training, and fresh training from a fixed, well-evaluated Foundation avoids stacking small errors across cycles. This also means every per-type adapter version has a clean, explicit dependency on exactly one Foundation version — critical for debugging "why did this document type regress" later.

### Versioning scheme

    foundation-v2.0
    lossrun-adapter-v2.1  (built on foundation-v2.0)
    acord25-adapter-v2.0  (built on foundation-v2.0)
    ...

Every per-type adapter's version tag records which Foundation version it depends on. When Foundation moves to v3.0, all per-type adapters must be re-validated (and likely retrained) against it before it becomes the production Foundation — treat this like a dependency upgrade, not an automatic cascade.

### Training Run Registry — Yes, Every Cycle Is Recorded (Foundation *and* Per-Type)

This directly answers your question: **yes**, keep a registry entry for every single training run, for both the Foundation LoRA and every Per-Type Adapter. Without it you cannot answer basic production questions later ("which corpus version produced lossrun-adapter-v2.1?", "why did ACORD accuracy drop between v2 and v3?", "what hyperparameters made the best Loss Run adapter?"). This is the single source of truth linking *code version + data version + config + resulting metrics + artifact location* for every run.

**Recommended: use MLflow (or Weights & Biases) as the experiment tracker, backed by a registry manifest in Azure Blob.** MLflow gives you the queryable UI/API over runs; the Blob manifest is the durable, human-readable record that travels with the artifacts.

**Registry structure in Blob:**

    azure-blob://insurance-extraction/registry/
      foundation/
        foundation-v1.0/run_manifest.json
        foundation-v2.0/run_manifest.json
        ...
      adapters/
        lossrun/
          lossrun-adapter-v1.0/run_manifest.json
          lossrun-adapter-v2.1/run_manifest.json
        acord/
          acord-adapter-v1.0/run_manifest.json
        ...
      registry_index.json            # top-level table: every run, its status, and pointers

**What each** `run_manifest.json` **records:**

    {
      "run_id": "lossrun-adapter-v2.1",
      "run_type": "per_type_adapter",
      "doc_type": "loss_run",
      "status": "promoted",                        // training | trained | evaluated | promoted | archived | failed
                                                   // `training` = manifest written, ms-swift not yet returned.
                                                   // A crashed run ends at `failed`, never `trained`, so the
                                                   // registry cannot advertise weights nobody wrote.
      "created_at": "2026-02-14T09:20:00Z",
      "dependencies": {
        "base_model": "qwen3-vl-8b-instruct@<hf_revision_pin>",
        "foundation_version": "foundation-v2.0",   // which foundation this adapter sits on
        "corpus_version": "corpus/v3",
        "code_git_commit": "a1b2c3d"               // exact repo commit that ran this
      },
      "training_config": {
        "technique": "LoRA",                       // "LoRA" | "QLoRA" — required, never defaulted
        "base_quantization": "bf16_frozen_base",   // or nf4_double_quant_bfloat16_compute under 4-bit
        "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "learning_rate": 7e-5, "epochs": 4,
        "effective_batch_size": 24,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","projector"],
        "vit_frozen": true,
        "resolution_cap_px": 1792, "max_seq_len": 8192
      },
      "data_stats": {
        "train_examples": 3100, "val_examples": 390, "test_examples": 310,
        "modality_mix": {"ocr_plus_image": 0.5, "noisy_ocr_image": 0.2, "image_only": 0.3}
      },
      "eval_metrics": {                            // against the frozen golden eval set (§15)
        "field_exact_match": 0.923,
        "field_f1_list_fields": 0.887,
        "schema_validity_rate": 0.998,
        "ece_confidence": 0.041,
        "ocr_arbitration_accuracy": 0.905,
        "image_only_accuracy": 0.861
      },
      "artifacts": {
        "adapter_weights": "adapters/loss_run/v2.1/",
        "merged_model": "merged-models/loss_run/v2.1/",
        "quantized_model": "quantized-models/loss_run/v2.1/gguf/",   // one subfolder per exported format
        "quantized_formats": ["fp16", "q6_k", "q5_k_m", "q4_k_m"],
        "eval_report": "eval-reports/v2.1/loss_run/"
      },
      "promotion": {
        "gated_against": "lossrun-adapter-v2.0",
        "beat_previous_on_all_gates": true,
        "promoted_by": "<reviewer>", "promoted_at": "2026-02-14T15:00:00Z"
      }
    }

**Why this matters concretely:**

- **Reproducibility** — `code_git_commit` + `corpus_version` + `training_config` together let you re-run any historical training exactly, which you'll need the first time a run you can't reproduce turns out to be your best model.
- **Regression debugging** — when a doc type regresses, you diff the new manifest against the last-good one: did the corpus change? the foundation version? a hyperparameter? The answer is right there instead of reconstructed from memory.
- **Dependency graph** — `foundation_version` in every adapter manifest means you can instantly list every adapter that depends on `foundation-v2.0` and therefore needs re-validation when Foundation moves to v3.0 (the §12 dependency-upgrade rule becomes a query, not a manual audit).
- **Lineage for compliance** — in a regulated domain, being able to show exactly which data and code produced a model that made a business decision is not optional.

The `registry_index.json` at the top is a flat table of all runs (run_id, type, status, key metrics, created_at) so you can see the whole history at a glance without opening individual manifests.

## 13. End-to-End Pipeline

    1. Ingestion         → raw PDFs land in Azure Blob  /raw-documents/ (immutable,
                             checksum-deduped, see §18a)
    2. Preprocessing      → MinerU OCR (markdown+layout) + page image rendering
                             (resolution-capped, matching training config)
                             → Azure Blob /processed/
    3. Labeling           → human-verified target JSON per document
                             (review tool / QA queue) → /golden-labels/
    4. Dataset build      → compile JSONL per doc type, inject schema,
                             apply 3-regime modality split, train/val/test split
                             → Azure Blob /corpus/v{n}/
    5. Training (RunPod)  → pull base model (cached from HF) + corpus from Blob
                             → LoRA train (Foundation or per-type)
                             → checkpoints + eval logs → /adapters/.../v{n}/
    6. Evaluation & Gate   → run against frozen golden eval set
                             → must beat previous version on all gating metrics
                             to be promoted (see §15)
    7. Merge              → PEFT merge_and_unload() → full fp16/bf16 model
    8. Quantize           → GGUF export, user-selected format(s): FP16 | BF16 |
                             Q8_0 | Q6_K | Q5_K_M | Q4_K_M (see §13a) — one or many
    9. Push to Azure Blob  → (a) raw adapter weights (b) merged full model
                             (c) quantized GGUF(s) — versioned, one subfolder per format (see §18)
    10. Serving           → RunPod Serverless vLLM endpoint pulls the promoted
                             artifact, serves via OpenAI-compatible API
    11. Feedback loop      → low-confidence / human-corrected production outputs
                             flow back into the labeling queue → next corpus version

**Pipeline properties.** Each stage reads from and writes to Azure Blob, and each training, evaluation, and quantization job produces a run manifest. Stages are **idempotent and resumable**, so a failed stage does not corrupt state and can simply be re-run. **Stage 6 (the evaluation gate) is a hard stop:** a candidate is promoted only if it matches or exceeds the current production version on every gating metric — there is no manual override path that skips it.

### 13a. GGUF Quantization — User-Selectable Format Matrix

Quantization is a **configurable export step**, not a single fixed choice: after merging the adapter into the base model, the user selects which GGUF format(s) to produce via a CLI flag. You can export one format or several from the same merged model in one run, since they all derive from the same fp16 GGUF conversion.

**Pipeline:** merged fp16/bf16 model → convert to base GGUF (`convert_hf_to_gguf.py` from llama.cpp) → quantize to each requested format (`llama-quantize`).

    python postprocessing/quantize.py --model v2 \
           --formats fp16 q6_k q5_k_m q4_k_m      # export several at once
    python postprocessing/quantize.py --model v2 --formats q4_k_m   # or just one

**Format matrix — what each buys you:**

| Format     | Bits/weight (approx) | Relative size¹ | Quality              | Typical use                                                                             |
|:-----------|:---------------------|----------------|:---------------------|:----------------------------------------------------------------------------------------|
| **FP16**   | 16                   | 100% (~16 GB)  | Reference / lossless | Accuracy baseline; the model you measure all quantized variants against                 |
| **BF16**   | 16                   | 100% (~16 GB)  | Reference / lossless | Same size as FP16, wider dynamic range; preferred baseline on hardware with native bf16 |
| **Q8_0**   | 8                    | ~53%           | Near-lossless        | When you want maximum quality but half the memory of fp16                               |
| **Q6_K**   | ~6.6                 | ~44%           | Very high            | Strong quality/size balance; good default when accuracy matters                         |
| **Q5_K_M** | ~5.7                 | ~38%           | High                 | Balanced; noticeably smaller with minimal quality loss                                  |
| **Q4_K_M** | ~4.8                 | ~32%           | Good                 | Most memory-efficient serving; the standard "ship it" choice when latency/cost dominate |

¹ Relative to the fp16 merged model (~16 GB for an 8B model). Actual sizes vary; measure on your merged model.

**Which formats to routinely produce and validate:** supporting all six is fine as a capability, but you don't need to *generate and re-validate all six every cycle* — that wastes eval compute. In practice, converge on **fp16 as the accuracy baseline + one serving format (likely Q5_K_M or Q4_K_M)** as the formats you produce and validate every cycle. Generate the others (Q8_0, Q6_K, BF16) on demand when a specific deployment target calls for them, not as a standing part of the pipeline.

**Critical caveat for a fine-tuned extraction model:** lower-bit quantization (especially Q4) can degrade exactly the behaviors you fine-tuned in — strict JSON structure, precise field values (dates, currency, policy numbers), and confidence calibration. **Every quantized format must be re-run through the extraction/testing routine (§17) against the golden eval set before it's promoted for serving** — never assume Q4_K_M preserves the accuracy you measured on fp16. It's common to find fp16 and Q6_K statistically indistinguishable while Q4_K_M drops a point or two on field-exact-match; whether that trade is acceptable is a per-deployment decision, which is exactly why you're keeping all formats available rather than hardcoding one.

**A note on the VL model specifically:** Qwen3-VL is multimodal, so the GGUF export must handle the vision encoder + projector as well as the LLM. In the llama.cpp ecosystem this typically means producing the quantized LLM GGUF plus a separate multimodal projector file (`mmproj`), served together. Confirm current llama.cpp support for the exact Qwen3-VL version before committing to a GGUF-only serving path — if vision-side GGUF support lags, keep the vLLM path (serving the merged fp16/bf16 or an AWQ/FP8 variant) as the primary serving route and treat GGUF as the portable/edge/offline option. This is worth verifying at implementation time rather than assuming, since multimodal GGUF support evolves quickly.

### 13b. Quantization Quality Thresholds

> **Enforcement deferred; the table stands as reference.** The primary serving path is vLLM on the merged fp16/bf16 model (§13a), so the first cycle ships nothing quantized and there is nothing to threshold-validate yet. Build the threshold check and wire the validation gate when a GGUF format is actually going to be served.

“Re-validate before promotion” needs numbers attached, or it degrades into a judgement call per release. Every format intended for serving is re-run against the frozen golden eval set, and the **acceptable degradation relative to the fp16 reference** is:

| Format | Max field F1 drop | Max ECE increase | Min JSON validity |
|--------|-------------------|------------------|-------------------|
| FP16   | Reference (0%)    | Reference        | 100%              |
| BF16   | ≤ 0.5%            | ≤ 0.005          | 100%              |
| Q8_0   | ≤ 1.0%            | ≤ 0.010          | 100%              |
| Q6_K   | ≤ 1.5%            | ≤ 0.015          | ≥ 99.5%           |
| Q5_K_M | ≤ 2.0%            | ≤ 0.020          | ≥ 99.5%           |
| Q4_K_M | ≤ 4.0%            | ≤ 0.030          | ≥ 99.0%           |

A format exceeding **any** threshold is not promoted to serving. **Q5_K_M is the default serving target**; Q4_K_M is used only under VRAM constraint and only when it meets threshold. These thresholds are initial targets for the pilot cycle and are revised after the first full evaluation cycle, once absolute F1 values are known — a 2% relative drop means something different at F1 0.95 than at 0.70.

### 13c. Operator Command Surface

The eleven stages above are driven by **three commands plus one umbrella command**, so a cycle is run by someone who knows the version tag and nothing else about paths, pods, or stage wiring.

| # | Command | Stages | Ends with |
|---|---|---|---|
| **1** | `finetune` | 1 to 7 (ingest, OCR, dataset build, train, evaluate, **gate**, merge) | Adapters + merged model on the **RunPod staging volume** (§14) |
| **2** | `package` | 8 to 9 (quantize, push) | Adapters + merged + quantized model in **Azure Blob** |
| **3** | `extract` | §17 extraction routine | JSON + confidence + metrics, locally |
| **—** | `all` | `finetune` then `package` | Same as command 2 |

    python -m orchestration.run finetune --input ./intake --out-version v2 --gpu a100-80
    python -m orchestration.run package  --version v2 --formats fp16 q5_k_m
    python -m orchestration.run extract  --model base|v1|v2 --input testing/test_data/
    python -m orchestration.run all      --input ./intake --out-version v2

**`all` never includes `extract`.** Extraction is a separate concern from building a model — it runs against any chosen version, including models trained weeks earlier and the untuned base.

**Three realities this surface respects:**

1.  **Labeling is human work inside the span of command 1.** `finetune` ingests and OCRs everything, builds the corpus from only the source documents that have a validated golden label, and **reports the unlabeled backlog**. It aborts before training only when the labeled set falls below a minimum per type (default 25, matching §4c).
2.  **The gate is a hard stop inside command 1.** A failed evaluation stops `finetune` *before* merge and exits non-zero, so `all` never reaches `package` on a failed gate.
3.  **Command 1 fans out into 1 + N GPU jobs** — Foundation first, then one per active document type, sequentially, because a per-type adapter cannot start before the Foundation it depends on has trained and passed evaluation (§12).

**`--model base` is a first-class tag** for command 3: it resolves to the pinned base model with no adapter. That is the same path used by the §16a zero-shot baseline and §4c day-zero pre-annotation, so all three share one implementation.

## 14. RunPod Infrastructure — Training vs. Serving

**Direct answer to your question: don't put your whole application inside RunPod. Split it into two scoped concerns.**

    Training (ephemeral pod)                 Serving (persistent endpoint)
    ─────────────────────────                ─────────────────────────────
    clone repo @ commit                      vLLM + promoted model
    pull base + corpus (Blob)                        │
    run LoRA training                        request ──> classify doc type
    push adapter + manifest (Blob)                   │
    terminate                                 select + hot-swap adapter
                                                     │
    External orchestrator                     inference + logprobs
    triggers pods via RunPod API                     │
                                              calibrate confidence
    Application logic (CPU infra,                     │
    outside RunPod): prompt                   return JSON + confidence
    assembly, endpoint calls,
    confidence calibration

    Preprocessing (GPU pod)
    MinerU OCR + page rendering

### Training — ephemeral pods

- On-demand RunPod GPU Pods (A100 80GB for Foundation training given the larger effective batch/sequence needs; **L40S 48GB for the smaller per-type adapter runs**). A100 40GB is marginal under the bf16 base — see the §9.2 table — so prefer L40S, or reach for ZeRO-3 or `load_in_4bit: true` when a 40GB card is what is available.
- Training **code lives in your private git repo**, versioned normally — the pod clones it fresh at job start, rather than code living permanently resident on a pod.
- A pod's job: pull base model + corpus version from Azure Blob → run training → push resulting adapter + eval report to Azure Blob → terminate. Trigger this via RunPod's API from a lightweight orchestrator (a simple Python controller, or GitHub Actions / Airflow if you want scheduling and audit trails).
- Nothing about training needs to be "always on" — keep these pods ephemeral to control cost.

### The staging volume — persistence between commands 1 and 2

Pods are ephemeral, so persistence between `finetune` and `package` (§13c) comes from a **RunPod network volume** mounted at `/runpod-volume` and attached to every pod the controller launches:

    /runpod-volume/staging/
      adapters/foundation/v{n}/
      adapters/{doc_type}/v{n}/
      merged-models/{doc_type|unified}/v{n}/
      eval-reports/v{n}/summary.json            # the scored EvalReport
      eval-reports/v{n}/{doc_type}/report.json
      eval-reports/v{n}/gate_decision.json      # the gate's verdict — a separate key,
                                                # so it cannot overwrite the report it read

The layout deliberately mirrors the Blob layout (§18) so `package` copies rather than translates. It exists because the merged model is ~16 GB and quantization also runs on RunPod — pushing it to Azure at the end of command 1 and pulling it back at the start of command 2 is a 32 GB round trip for nothing.

> **Terminology.** "Registry" in this architecture means the durable run-manifest registry in Azure Blob (§12). The RunPod side is the **staging volume** and is never called a registry — different lifetimes, different guarantees. Staging is working storage with **no durability guarantee**; the registry is the lineage of record.

**Because a volume is not an artifact of record, command 1 always writes its run manifest to Blob** with `artifacts.status: "staged"`, even though the weights stay on the volume; command 2 flips it to `"published"` with real Blob paths. Without that, a reclaimed volume would mean a training run that happened and left no trace.

### Preprocessing runs on GPU

**MinerU is GPU-accelerated, not CPU.** Its layout detection, table and formula
recognition, and OCR models are all GPU-bound; on CPU they fall back to lighter
variants, so the GPU path is both several times faster in wall-clock *and* higher
quality. Ingestion itself (checksum, dedup, immutable write) stays CPU — only the
OCR and page-render step needs the GPU.

**Pod class.** MinerU does not need an A100. An L4, A10 or L40S is sufficient and
is the right default for this stage; reserve the A100 for Foundation training.
Cost usually *falls* against CPU despite the higher hourly rate, because the job
finishes in a fraction of the wall-clock time.

**One pod for the whole of command 1.** Stages 2, 4 and 5 (OCR → dataset build →
train) all now want a GPU, so the natural shape is a single pod for the whole
`finetune` command rather than shuttling a corpus between machines (§13c).

**Device may be part of the corpus pin.** §8a exists because the model learns *how
MinerU formats its output*. If GPU and CPU MinerU produce different markdown,
then training on GPU-OCR and serving on CPU-OCR is the same distribution shift
§8a warns about. `ocr_meta.json` and the corpus manifest therefore record
`ocr_device` alongside `mineru_version`, and the version check compares both.
Confirm empirically whether the outputs differ rather than assuming either way.

**Open question for the serving path.** Inference also runs OCR. If MinerU needs
a GPU at request time, the boundary above — business logic on cheap CPU infra,
RunPod scoped to GPU work — weakens, and OCR either bundles into the serving pod
or becomes its own endpoint. The training path does not depend on the answer.

### Serving — persistent endpoint

- **RunPod Serverless Endpoint** running vLLM, hosting the currently-promoted merged/quantized model, exposed as an OpenAI-compatible HTTPS API.
- Your **application/orchestration logic** (assembling the prompt, calling this endpoint, applying the confidence calibration transform, returning final JSON) should run in **your own service** — Azure Function, App Service, or AKS, not inside RunPod. This keeps business logic on cheap CPU infra, keeps it portable if you ever change GPU providers, and keeps RunPod scoped purely to GPU-bound work.
- Adapter hot-swapping (per-document-type) is handled at the vLLM layer via LoRA adapter loading — your application just tells the endpoint which adapter to apply per request (typically via a request parameter after your document-type classification step).

## 15. Evaluation Framework — Metric Definitions

Defines each metric used by both the extraction/testing routine (§17) and the promotion gate. Run every candidate version against the **frozen golden eval set** before promotion:

| Metric                                                | What it catches                                                                                                                                           |
|:------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Document-type classifier accuracy                     | Whether the right adapter/prompt/schema is even being selected (§4a) — caps whole-system accuracy                                                         |
| Field-level exact match / normalized match            | Core extraction accuracy                                                                                                                                  |
| Field-level F1 (for list fields — line items, claims) | Precision/recall on repeating structures                                                                                                                  |
| **List-field recall (row completeness)**              | Whether whole rows are being missed (§5) — a Loss Run claim dropped silently, invisible to per-value confidence                                           |
| JSON schema validity rate                             | Structural reliability (parseable, all required keys present)                                                                                             |
| Expected Calibration Error (ECE) on confidence scores | Whether your confidence numbers are trustworthy, not just your extractions                                                                                |
| OCR-vs-image disagreement resolution accuracy         | Specifically evaluated on the deliberately-noisy-OCR eval subset — did the model correctly override bad OCR using the image?                              |
| Image-only mode accuracy                              | Evaluated on the no-OCR eval subset — confirms the second production pathway actually works, not just the primary one (feeds the ViT-unfreeze gate in §3) |
| Scanned-PDF accuracy                                  | Evaluated on the scanned-document eval subset — the other input to the ViT-unfreeze gate (§3)                                                             |
| Latency / token cost per document                     | Production feasibility, not just accuracy                                                                                                                 |

### Additional metrics — canonical field mapping (§0c)

| Metric | What it catches | Gating? |
|---|---|---|
| **Alias accuracy** | Field accuracy sliced by the **observed surface label**, joined through the `field_provenance` recorded at labeling time. Turns "field accuracy is 0.87" into "0.94 on *Named Insured*, 0.61 on *Applicant*" — which names the documents to go collect | Reported, **not gating** — rare aliases have too little support for a stable gate |
| **Confusable misattribution rate** | How often a confusable entity's value is returned as the canonical field — a certificate holder emitted as `insured_name` | **Gating** |

Misattribution is isolated from ordinary field accuracy because it is **systematic rather than random**: it means the model has collapsed two distinct entities, it will keep doing so, and the output is fluent and well-formed enough to pass every structural check.

### Additional metric — Line-of-Business detection accuracy

Because the VLM is the fallback LoB detector when L1/L2 miss (§0b), `line_of_business` accuracy is reported as its own metric rather than being averaged into overall field accuracy. It is measured per LoB value, so a class that is rare in the corpus can’t hide inside a healthy-looking aggregate, and it is a gating metric for promotion.

**Normalized match** matters for insurance fields specifically: dates (`01/01/2026` vs `2026-01-01`), currency (`$1,200.00` vs `1200.0`), and entity names (`Acme Mfg LLC` vs `ACME MANUFACTURING LLC`) should be compared after normalization, not as raw strings, or you'll under-count correct extractions. Define a per-field-type normalizer and apply it consistently in both the testing routine and the promotion gate.

A new Foundation or per-type adapter version is only promoted if it **matches or beats** the current production version on every gating metric for the relevant document type(s) — no silent regressions.

## 16. Pilot Validation Protocol — De-Risking Before Full Annotation

Before committing to full corpus annotation, three sequential experiments de-risk the architecture in order of increasing investment. Each one answers a different question, and each is cheap relative to the one after it.

### 16a. Zero-Shot Baseline (Week 1 — no annotation cost)

Run the **base** `Qwen3-VL-8B-Instruct`, untuned, against **10 de-identified real documents per document type**, using the §7 prompt template with the Fideon SPEC_00 schema injected. Compute field-level F1 manually against ground-truth annotations.

This tells you how much the base model extracts for free, and — more usefully — *where* it fails: schema adherence, list-row recall, LoB detection, or OCR arbitration. Outcome thresholds:

| Zero-shot field F1 | Interpretation                                                            | Decision                                                                                                     |
|--------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| \> 0.70            | Strong base-model prior; fine-tuning expected to reach production quality | Proceed to pilot corpus                                                                                      |
| 0.40 to 0.70       | Fine-tuning will materially improve; specific failure modes identified    | Proceed, with targeted corpus coverage for the weak areas                                                    |
| \< 0.40            | Base model likely insufficient as framed                                  | Review prompt design, schema complexity, and document difficulty **before** committing annotation investment |

### 16b. Pipeline Smoke Test (Weeks 1–2 — 5 annotated documents)

Annotate exactly **5 documents per document type**. Train to *intentional overfit*: no train/val split, 5 epochs, learning rate at the upper end of the sweep range. Confirm that:

- Training loss reaches below **0.05 within 2 epochs**.
- The model reproduces the 5 training examples with **field F1 \> 0.95 on the training set**.
- The ms-swift training loop, data collator, LoRA adapter loading, adapter save and push to Azure Blob, vLLM multi-adapter hot-swap, and run manifest generation all complete without error, end to end.

This proves **the code pipeline is correct** — not that the architecture generalises. It is a sanity check, not a proof of concept.

**Why 5 documents and not 1:** a single source document generates only three modality variants (three training examples), which is not enough to exercise the full data collator pipeline. Five documents per type covers multi-page, OCR-failure, and image-only edge cases.

### 16c. Pilot Training Run (Weeks 2–6 — 25 to 30 documents per type)

Annotate 25 to 30 documents per document type following §7, using the pilot split ratios from §8. Train Foundation and per-type adapters. Evaluate on the pilot test split.

This is the **minimum experiment that tests the architecture’s generalisation claim**: whether the Foundation captures cross-type behavior, whether per-type adapters specialise correctly, whether modality-dropout arbitration functions, whether row-completeness detection fires on unseen documents, and whether confidence calibration is meaningful with held-out data.

Pilot success criteria — **directional targets, not production thresholds**:

| Metric                              | Pilot threshold                               |
|-------------------------------------|-----------------------------------------------|
| Field F1 (non-list fields)          | \> 0.80                                       |
| List-field recall                   | \> 0.75                                       |
| JSON schema validity                | 100%                                          |
| Image-only field F1                 | Within 15% of OCR-plus-image F1               |
| Row-completeness detection accuracy | \> 80% of documents with missing rows flagged |
| LoB detection accuracy              | \> 0.85                                       |
| Alias generalization                | Field accuracy on a **held-out surface label** within 15% of that field's dominant label |
| Confusable misattribution rate      | \< 5%                                         |

A failed criterion signals a **specific architectural issue to diagnose** before scaling the corpus — not a reason to abandon the architecture. Diagnose the failure-mode distribution, expand corpus coverage for the failing case, and re-run the pilot.

**Alias generalization is measured by deliberate hold-out.** Pick one surface label per confusable-prone field (say *Applicant* for `insured_name`), keep every document using it out of `train`, and place them in the pilot test split. Scoring well on them shows the model learned the semantic mapping rather than memorising label strings — the central claim of §0c, and the one thing no plumbing test can prove.

**Two measurement caveats at pilot scale**, both consequences of volume rather than design:

- **Alias generalization may be unmeasurable.** With five documents using a rare label and a ~70/18/12 split, you are testing on one or two documents. Treat the criterion as directional here and enforce it only at real scale.
- **Image-only will lag, and that must not fire the §3 ViT gate.** Image-only is ~30% of the corpus, so at pilot it carries roughly nine documents of signal per type — for the hardest regime, where the ViT reads with no OCR backstop. Low accuracy there is a **data-volume artifact**, and thin data produces exactly the perception-shaped errors the gate keys on. Unfreezing the ViT on pilot evidence would buy an expensive retrain to fix a problem more documents would have solved. **The gate should not fire on pilot data.**

## 17. Extraction / Testing Routine (Post-Fine-Tuning)

This is the CLI harness you run **after** a model version is trained, to test it on real documents, inspect the JSON quality, and generate the metrics that tell you whether the last fine-tuning cycle actually improved things. It mirrors the exact production inference path (MinerU → model → JSON), so what you test is what you'll ship.

### Folder layout

The `testing/` module lives in the main repository structure — see §19 for its placement and file listing (`test_data/`, `prompts/`, `ocr_cache/`, `results/{version}/`, `metrics/{version}/`, `extraction_registry.json`, `run_extraction.py`). The rest of this section covers how the routine behaves.

### CLI usage — model is user-selectable per your requirement

    # User decides which model version to run — v1, v2, v3, ...
    python testing/run_extraction.py --model v2 --input testing/test_data/

    # Single file, image-only mode (no OCR), to test the no-OCR pathway
    python testing/run_extraction.py --model v2 --input testing/test_data/abcLossRun.pdf --mode image_only

    # Full batch with OCR (default mode), with ground-truth for metric scoring
    python testing/run_extraction.py --model v3 --input testing/test_data/ \
           --mode ocr_plus_image --ground-truth testing/golden/

- `--model v2` resolves to the correct artifact from the registry (foundation-v2 + the right per-type adapter, or the merged/quantized v2 model) — the user never has to know paths, just the version tag.
- `--mode` selects `ocr_plus_image` (default) or `image_only`, so you can test both production pathways from the same harness.
- `--ground-truth` is optional: if supplied, metrics are computed against golden JSON; if omitted, the routine still runs and produces JSON + confidence scores (just no accuracy metrics, since there's nothing to compare against).

### What the routine does, step by step

    For each PDF in --input:
      1. OCR              → MinerU produces markdown + rendered page image(s) → ocr_cache/
                            (skipped if --mode image_only, image still rendered)
      2. Doc-type detect  → classify the document as acord | policy | lossrun
                            (cheap classifier or a zero-shot prompt to the model itself)
      3. Prompt select    → load the matching prompts/{doc_type}.prompt.txt
                            (this is why you keep one prompt file per type — see below)
      4. Assemble input   → system = prompt file (schema + instructions)
                            user   = [image] (+ OCR markdown, unless image_only)
      5. Model inference  → selected --model version generates structured JSON
                            + token logprobs
      6. Confidence       → per-field confidence from logprobs, calibration transform applied (§5)
      7. Validate         → check JSON parses + conforms to that doc type's schema
      8. Write result     → results/{version}/{doc_stem}.json
      9. Metrics          → if ground truth available, compute per-field + overall metrics
                            → metrics/{version}/{doc_stem}.metrics.json
     10. Register         → append to extraction_registry.json

### Per-doc-type prompt files

You asked for a separate `prompt.txt` per doc type — this is the right design and here's why: each document type has a different schema, different field semantics (a Loss Run's "valuation date" vs. an ACORD's "policy effective date"), and different extraction edge cases. Keeping them in separate files means:

- The doc-type detection step in the routine directly maps to a prompt file — detect `lossrun` → load `lossrun.prompt.txt`.
- You can tune one document type's prompt without risk to the others.
- Each file contains: the role framing, the injected JSON schema for that type (pulled from the §19 schema registry so it stays in sync), the extraction rules (null-handling, date formats, how to treat repeating table rows), and the modality-mode instruction line (§6). These are the **same prompts** used at production serving time — testing and production must not drift.

### Metrics produced (per your "per-field and overall confidence" requirement)

**Per-field, per-document** (`metrics/{version}/{doc}.metrics.json`):

    {
      "document": "abcLossRun.pdf",
      "model_version": "v2",
      "doc_type": "loss_run",
      "schema_valid": true,
      "overall_confidence": 0.88,                    // aggregate over all fields
      "fields": {
        "carrier":        {"value": "...", "confidence": 0.97, "correct": true},
        "policy_number":  {"value": "...", "confidence": 0.94, "correct": true},
        "valuation_date": {"value": "...", "confidence": 0.71, "correct": false},   // flagged: low conf + wrong
        "claims[0].amount": {"value": 12000, "confidence": 0.63, "correct": true}
      },
      "field_exact_match_rate": 0.91,                // this document's field-level accuracy (needs ground truth)
      "list_field_f1": 0.88                          // for repeating structures (claims, line items)
    }

**Batch summary** (`metrics/{version}/_run_summary.json`) — aggregates across all test docs for that model version: mean field-exact-match, mean list-field F1, schema-validity rate, ECE (confidence calibration quality), per-doc-type breakdown, and mean latency/document. This is the file you actually compare across model versions to answer "did v3 beat v2?" — it's the testing-time mirror of the promotion gate in §15.

**On the confidence-without-ground-truth case:** even when you have no golden JSON (a brand-new unknown document), the routine still emits `overall_confidence` and per-field confidence from the calibrated logprobs. That's the whole point of the calibration work in §5 — confidence is available at inference time on documents you've never labeled, which is exactly what you use to route low-confidence extractions to human review in production.

### Extraction registry — provenance of every extracted result

`testing/extraction_registry.json` records which document was extracted by which model, so an output JSON is never ambiguous about its origin:

    {
      "extractions": [
        {
          "document": "abcLossRun.pdf",
          "doc_type": "loss_run",
          "model_version": "v2",
          "mode": "ocr_plus_image",
          "result_path": "results/v2/abcLossRun.json",
          "metrics_path": "metrics/v2/abcLossRun.metrics.json",
          "overall_confidence": 0.88,
          "schema_valid": true,
          "extracted_at": "2026-02-15T11:03:00Z"
        }
      ]
    }

Combined with the `results/{version}/` folder layout you specified, anyone can see at a glance that `results/v2/abcLossRun.json` came from model v2 — both from the path itself and from the registry entry. This ties back to the training run registry in §12: the `model_version` here (`v2`) is the same tag that resolves to a `run_manifest.json`, so you can trace an extracted result all the way back to the corpus version and git commit that produced the model that generated it.

## 18. Azure Blob Folder Structure

    azure-blob://insurance-extraction/
      raw-documents/                                      # original source PDFs — see §18a for detail
      processed/                                          # MinerU output + rendered page images
      golden-labels/                                      # human-verified target JSON, keyed to raw-documents
      base-models/qwen3-vl-8b-instruct/                   # cached from HF, pinned revision
      corpus/v{n}/{doc_type}/{train,val,test}.jsonl
      adapters/
        foundation/v{n}/
        {doc_type}/v{n}/                                  # tagged with dependent foundation version
      merged-models/{doc_type|"unified"}/v{n}/            # fp16/bf16
      quantized-models/{doc_type|"unified"}/v{n}/gguf/{fp16|bf16|q8_0|q6_k|q5_k_m|q4_k_m}/
      registry/                                           # training run manifests (see §12)
      eval-reports/v{n}/summary.json            # the scored EvalReport
      eval-reports/v{n}/{doc_type}/report.json
      eval-reports/v{n}/gate_decision.json      # the gate's verdict — a separate key,
                                                # so it cannot overwrite the report it read
      golden-eval-set/                                     # frozen, versioned separately, rarely changes

**Staging vs. Blob:** during a cycle, adapters and the merged model live first on the RunPod **staging volume** (§14) and reach the Blob paths above only when `package` (§13c) runs. The run manifest is written to Blob either way — a staged run is still a recorded run.

**Tenant partitioning:** under the multi-tenant model in §8b, `corpus/` (and `raw-documents/`, `processed/`, `golden-labels/`) are prefixed by broker `tenant_id` — e.g. `azure-blob://insurance-extraction/corpus/{tenant_id}/v{n}/{doc_type}/{split}.jsonl`. Shared artifacts that contain no tenant data (`base-models/`, the Foundation adapter trained on de-identified data, `registry/`) remain un-prefixed.

### 18a. Where the Original Source PDFs Live — Detail

This deserves its own treatment since everything downstream (OCR, labels, corpus) traces back to it, and insurance documents carry real PII (names, SSNs/TINs on some ACORD forms, addresses, financial/premium data) — so this isn't just a storage question, it's a compliance one.

**Location and structure**

    azure-blob://insurance-extraction/raw-documents/
      {doc_type}/
        {source_id}/
          original.pdf                        # untouched, exactly as received
          metadata.json                        # ingestion timestamp, source system, checksum, PII flags

- `source_id` is the same identifier used in the dataset JSONL (`source_id` field, §7), in `processed/`, and in `golden-labels/` — this is the join key that lets you trace any training example all the way back to its original PDF for audit or debugging.
- Store the PDF **immutably, exactly as received** — never overwrite. If a corrected version of a document arrives, ingest it as a new `source_id` (or a versioned sub-path) rather than replacing the original, so historical training runs remain reproducible against the exact bytes they were trained on.

**Storage tier and access control**

- Use a separate Blob container (or at minimum separate access policy) for `raw-documents/` versus everything else — it's the only layer holding unredacted PII, so it should have tighter RBAC (fewer identities with read access) than `corpus/`, `adapters/`, or `eval-reports/`, which downstream training/serving jobs need broader access to.
- Enable encryption at rest (default for Azure Blob) and consider customer-managed keys if your compliance requirements call for it — reasonable given the sensitivity of the data, but confirm against whatever regulatory framework applies to your data (state insurance regulations, SOC 2 scope, etc.) rather than assuming.
- Cool/Archive tier is appropriate for `raw-documents/` once a document has been processed and labeled — it's accessed rarely (mainly for audit/reprocessing), unlike `processed/` and `corpus/` which get pulled repeatedly by training jobs.

**Retention**

- Define an explicit retention policy for raw PDFs (this is a legal/compliance decision your team needs to make, not a purely technical one) — insurance data often has multi-year retention requirements, but you should also have a documented deletion path for cases where a source document must be purged (e.g., a client requests removal).
- If a raw PDF is deleted, its downstream `processed/`, `golden-labels/`, and any `corpus/` entries built from it should be flagged or removed too — otherwise your training corpus retains derived data from a document that no longer legally exists, which is a real audit gap in a regulated domain.

**Deduplication**

- Store a content checksum (e.g., SHA-256) in each `metadata.json` and check it at ingestion time — insurance documents (especially Loss Runs and renewal policies) are frequently re-submitted with only minor changes, and de-duping prevents redundant labeling effort and corpus bloat from near-identical documents.

**Access pattern in the pipeline**

- `data_pipeline/ingestion/pull_raw_pdfs.py` (from the repo structure in §19) writes into `raw-documents/` and is the **only** pipeline component that touches this layer directly.
- `data_pipeline/ocr/run_mineru.py` reads from `raw-documents/`, writes to `processed/` — it never writes back into `raw-documents/`.
- No training or serving component should ever need direct access to `raw-documents/` — by the time data reaches `corpus/`, it's already been through OCR, labeling, and dataset compilation. Keeping training/serving scoped away from raw PDFs is both a security good-practice and a way to enforce that the corpus-build step is the single place PII handling policy gets applied (e.g., redaction rules, if your legal team requires masking certain fields before they ever reach a training example).

**PII beyond storage — it flows through the whole pipeline, not just the raw layer** Storage encryption (above) covers data at rest, but PII also moves through processing steps that need their own handling:

- **Pre-annotation (§7)** — do not send PII-bearing documents to third-party model APIs unless contractually permitted; see the PII caveat in §7.
- **Training data** — the corpus itself contains PII in both inputs (OCR text, images) and targets (extracted names, policy numbers). It inherits the same access-control and encryption posture as `raw-documents/`; it is not "safe" just because it's been reformatted into JSONL.
- **Logs and eval reports** — inference logs, error dumps, and eval reports can inadvertently capture PII (a logged prompt, a failed extraction printed to a log). Scrub or access-control these; don't let PII leak into low-security observability tooling.
- **Inference-time (serving)** — documents flowing through the live endpoint carry PII in-flight. Ensure the serving path (RunPod endpoint + your orchestration service) meets the same compliance bar, and avoid persisting raw request/response bodies in plaintext logs.
- **Model memorization** — fine-tuning on PII means the model can, in principle, memorize and regurgitate specific values. This is usually low-risk for a schema-extraction model (it's learning *format*, not *facts*), but worth noting if the model is ever exposed beyond your controlled environment.

## 19. Project Repository Folder Structure

This is the **codebase** structure (git repo) — separate from the Azure Blob artifact layout in §17, which holds data/model outputs, not code. Keep these cleanly separated: the repo is what gets cloned into a RunPod training pod at job start; Blob is everything the repo reads from and writes to.

    insurance-extraction-finetuning/
    ├── README.md
    ├── pyproject.toml / requirements.txt
    ├── .env.example                          # Azure Blob + RunPod credentials template (never commit real secrets)
    │
    ├── configs/
    │   ├── base_model.yaml                   # HF model id, revision pin, quantization config
    │   ├── training/
    │   │   ├── foundation.yaml               # LR, epochs, LoRA rank/alpha, target modules, etc.
    │   │   ├── acord25_adapter.yaml
    │   │   ├── acord125_adapter.yaml
    │   │   ├── lossrun_adapter.yaml
    │   │   └── policy_adapter.yaml           # (quote/endorsement adapter yamls added when activated)
    │   ├── deepspeed/
    │   │   └── zero2.json / zero3.json
    │   └── inference/
    │       └── vllm_serving.yaml             # resolution cap, max_seq_len, adapter routing config
    │
    ├── schemas/                              # JSON schema registry (source of truth, versioned)
    │   ├── acord25.schema.json                # every field carries a `description` gloss (§0c)
    │   ├── acord125.schema.json
    │   ├── acord140.schema.json
    │   ├── lossrun.schema.json
    │   ├── policy_doc.schema.json            # (quote/endorsement schemas added when activated)
    │   ├── lob.enum.json                     # one definition of the LOB enum, $ref'd by all
    │   ├── aliases/{doc_type}.aliases.json   # NEVER rendered into a prompt (§0c)
    │   └── examples/{type}.example.json
    │
    ├── common/                               # shared helpers used by every stage
    │   ├── config.py, schemas.py, prompts.py, ids.py, constants.py
    │   ├── lob.py                            # LOB enum + per-value coverage counting
    │   ├── aliases.py                        # registry loader — labeling/eval only, never serving
    │   └── normalize.py                      # one definition of "matches", shared by §15/§17/alias derivation
    │
    ├── prompts/
    │   ├── system_prompt_template.jinja      # injects schema + modality-mode instructions
    │   └── doc_type_classifier_prompt.jinja
    │
    ├── data_pipeline/
    │   ├── ingestion/
    │   │   └── pull_raw_pdfs.py              # Azure Blob raw/ ingestion
    │   ├── ocr/
    │   │   └── run_mineru.py                 # MinerU batch OCR + page image rendering
    │   ├── labeling/
    │   │   ├── pre_annotate.py               # silver drafts; external backends off by default
    │   │   ├── derive_aliases.py             # builds the alias registry from labeled docs (§0c)
    │   │   ├── review_tool/                  # (or config pointing to external labeling tool)
    │   │   ├── active_learning.py            # confidence-routed review queue (§13 step 11)
    │   │   └── export_golden_labels.py       # schema validation + field_provenance
    │   ├── dataset_builder/
    │   │   ├── build_jsonl.py                # assembles chat-format examples
    │   │   ├── modality_dropout.py           # applies the 50/20/30 modality-mode split
    │   │   ├── noisy_ocr_augment.py          # generates deliberate OCR-error examples
    │   │   └── split_train_val_test.py
    │   └── corpus_manifest.py                # writes manifest.json per corpus version
    │
    ├── training/
    │   ├── train_foundation.py               # entrypoint: builds ms-swift config → launches training
    │   │                                     #   (Layer 3; real loop is TRL SFTTrainer, §10)
    │   ├── train_adapter.py                  # per-doc-type adapter entrypoint (same ms-swift stack)
    │   ├── data_collator.py                  # override hook only; ms-swift provides -100 masking by default (§10)
    │   └── callbacks/
    │       └── early_stopping.py
    │
    ├── evaluation/
    │   ├── run_eval.py                       # runs a model/adapter version against golden eval set
    │   ├── metrics/
    │   │   ├── field_exact_match.py
    │   │   ├── field_f1.py
    │   │   ├── schema_validity.py
    │   │   ├── calibration_error.py          # ECE computation
    │   │   └── ocr_arbitration_accuracy.py
    │   └── gating.py                         # promotion pass/fail logic vs. previous version
    │
    ├── calibration/
    │   ├── fit_calibration.py                # temperature scaling / isotonic regression fit
    │   └── calibration_store/                # saved calibration params per field/doc-type/version
    │
    ├── postprocessing/
    │   ├── merge_adapter.py                  # PEFT merge_and_unload()
    │   └── quantize.py                       # GGUF export, --formats fp16|bf16|q8_0|q6_k|q5_k_m|q4_k_m (§13a)
    │
    ├── artifact_registry/
    │   ├── push_to_blob.py                   # pushes adapters/merged/quantized to Azure Blob
    │   └── pull_from_blob.py                 # used by training pods and serving endpoint
    │
    ├── serving/
    │   ├── vllm_entrypoint.py                # RunPod Serverless handler
    │   ├── doc_type_classifier.py            # identifies doc type + ACORD form number (§4a/§4b)
    │   ├── adapter_router.py                 # classifier result → adapter/prompt/schema selection
    │   ├── page_router.py                    # long-doc page selection for Policy docs (§7)
    │   └── confidence_postprocess.py         # calibration transform + list-field completeness check (§5)
    │
    ├── testing/                             # post-fine-tuning extraction/test harness (§17)
    │   ├── run_extraction.py                 # CLI entrypoint: --model v2 --input ... --mode ...
    │   ├── test_data/                        # INPUT: drop test PDFs here (scanned or digital, mixed)
    │   │   ├── abcLossRun.pdf
    │   │   ├── xyz_acord25.pdf
    │   │   └── policy_sample_01.pdf
    │   ├── prompts/                          # one prompt file per doc type
    │   │   ├── acord.prompt.txt
    │   │   ├── policy.prompt.txt
    │   │   └── lossrun.prompt.txt              # (quote/endorsement prompts added when activated)
    │   ├── ocr_cache/                        # MinerU output cached per doc: {doc_stem}/page_*.md, page_*.png
    │   ├── results/                          # OUTPUT: extracted JSON, organized BY MODEL VERSION
    │   │   ├── v1/abcLossRun.json
    │   │   ├── v2/abcLossRun.json            # /results/v2/abcLossRun.json ← requested layout
    │   │   └── v3/...
    │   ├── metrics/                          # OUTPUT: per-run metric reports, by version
    │   │   └── v2/
    │   │       ├── abcLossRun.metrics.json   # per-document, per-field metrics + confidence
    │   │       └── _run_summary.json         # aggregate over the whole test batch for this version
    │   └── extraction_registry.json          # provenance: which doc → which model version, when
    │
    ├── registry_utils/                     # training run registry read/write (§12)
    │   ├── write_run_manifest.py
    │   └── query_registry.py                 # e.g. "list adapters depending on foundation-v2.0"
    │
    ├── orchestration/
    │   ├── run.py                            # the §13c command surface: finetune | package | extract | all
    │   ├── runpod_controller.py              # ephemeral training pods + staging volume, via RunPod API
    │   └── pipeline_dag.py                   # (Airflow/GH Actions) full pipeline trigger, stage 1→11 in §13
    │
    └── tests/
        ├── test_data_collator.py
        ├── test_schema_validity.py
        ├── test_calibration.py
        └── fixtures/                         # small sample PDFs/JSONL for CI

**Key separation principle:** the repo never stores model weights, corpus data, or PDFs directly — everything data/artifact-related is pulled from and pushed to Azure Blob at runtime via `artifact_registry/`. This keeps the repo lightweight, makes RunPod pod startup fast (clone repo, pull only what that specific job needs from Blob), and keeps corpus/model versioning fully decoupled from code versioning.

## 20. Summary Checklist

- [ ] ViT frozen *initially*; unfreeze via eval gate if scanned/image-only accuracy shows visual-reading is the bottleneck (§3)
- [ ] LoRA on LLM decoder + projector
- [ ] LoRA on a bf16 base, bf16 adapters, flash-attention 2 (§9.2); `load_in_4bit` available for VRAM-constrained pods, and the manifest records which way each run went
- [ ] Trainer stack locked: ms-swift entrypoint → TRL SFTTrainer loop → PyTorch/PEFT/bitsandbytes/DeepSpeed; framework provides the `-100` collator masking
- [ ] Foundation LoRA (rank 64/alpha 128) trained on full mixed corpus, all doc types + all 3 modality regimes
- [ ] Per-type LoRA adapters (rank 16/alpha 32) trained fresh from current Foundation, never from previous adapter checkpoint
- [ ] Document-type classifier as a first-class component with its own accuracy target + low-confidence fallback to Foundation-only extraction (§4a); two-level for ACORD form detection (§4b)
- [ ] Long/multi-page strategy for Policy docs: page selection → scoped extraction → merge (§7)
- [ ] Major Foundation updates retrain from raw HF base with full accumulated corpus; minor patches may continue-train with mandatory regression testing
- [ ] Standardized image resolution cap, identical in training prep and production inference
- [ ] Confidence via logprob extraction + post-hoc calibration; list fields get a second row-completeness signal (§5)
- [ ] Hyperparameters treated as sweep starting points, tuned on validation set (§11)
- [ ] Frozen golden eval set gating every promotion; includes classifier accuracy, list-field recall, image-only + scanned subsets
- [ ] RunPod split into ephemeral training pods + persistent Serverless vLLM endpoint; business logic lives outside RunPod
- [ ] Azure Blob versioned artifact structure for corpus, adapters, merged models, quantized models, eval reports
- [ ] PII handled across the whole pipeline, not just storage: pre-annotation (no unpermitted 3rd-party APIs), training data, logs, inference in-flight (§18a)
- [ ] GGUF quantization user-selectable (FP16/BF16/Q8_0/Q6_K/Q5_K_M/Q4_K_M); routinely produce/validate only fp16 baseline + serving format (Q5_K_M or Q4_K_M), others on demand; verify Qwen3-VL mmproj support at implementation time
- [ ] Training run registry (MLflow/W&B + Blob `run_manifest.json`) for EVERY Foundation and Per-Type run — records code commit, corpus version, config, metrics, artifact paths, promotion status
- [ ] Split at source-document level before modality expansion (no leakage); ratio scaled to data volume (~70/18/12 pilot → 80/10/10 at scale)
- [ ] **Every band's ratio triple sums to exactly 1.0**, asserted in code. The splitter assigns by hash threshold — train below `train`, val below `train + val`, test above — so a triple summing to 1.05 silently gives the test split 10% while the corpus manifest records 15%, and nothing else notices.
- [ ] Golden JSON via bootstrap pre-annotation + human review; provenance stored per label; switch pre-annotation to own model once v1 exists
- [ ] Target JSON is the Fideon SPEC_00 canonical schema; golden labels validated against it before corpus admission; schema change ⇒ corpus rebuild + new training cycle (§0a)
- [ ] `line_of_business` output by the VLM per the Fideon SPEC_00 LOB enum, present in every golden JSON (null when undetermined), ≥20% corpus coverage per LoB value, gated as its own metric (§0b)
- [ ] Day-zero bootstrap defined: base model classifies + pre-annotates with 100% human review until 25 labeled examples/type exist (§4c)
- [ ] System prompt template versioned with the schema registry; training and inference prompts identical (§7)
- [ ] MinerU version pinned per corpus version; upgrade ⇒ reprocess + corpus increment; version mismatch treated as a regression trigger (§8a)
- [ ] **OCR runs on GPU only — no CPU path, no fallback.** MinerU's CPU path uses lighter model variants and emits different markdown from the same PDF, so a corpus spanning both devices is built from two distributions. With no GPU visible the stage fails rather than falling back, because a fallback would finish the job and report success while writing the wrong distribution.
- [ ] Tenancy per Fideon SPEC_12: `tenant_id` path prefix reserved, **no cross-tenant mixing in a corpus file** (the live rule); single-tenant build, per-tenant adapter lineages not built (§8b)
- [ ] **De-identification BLOCKED** — text-only Presidio corrupts the training signal; resolve image redaction or de-identify nothing before implementing (§8b)
- [ ] LoRA target modules specified and justified per module (§9a); ViT escalation uses LoRA, never full fine-tuning (§3)
- [ ] Hyperparameter sweep specified as a bounded 3-phase protocol (LR → epochs → rank), ~9-12 runs, each producing a run manifest — **execution deferred until after the pilot** (§11a)
- [ ] Quantization quality thresholds defined per format; Q5_K_M default serving target — **enforcement deferred**, first cycle serves merged fp16/bf16 via vLLM (§13b)
- [ ] Pilot validation protocol executed before full annotation: zero-shot baseline → 5-doc smoke test → 25-30-doc pilot, each with explicit pass thresholds (§16)
- [ ] Extraction/testing routine (`testing/run_extraction.py`) with user-selectable `--model` (including `base`), per-doc-type prompt files, results in `results/{version}/`, per-field + overall confidence metrics, and an extraction registry for output provenance
- [ ] **Canonical field mapping** (§0c): golden JSON and inference output both canonical; every schema field carries a semantic gloss with exclusions; alias registry derived from labeled data and **never used at runtime**
- [ ] **Confusable discrimination**: co-occurrence documents required in the corpus; misattribution rate is a gating metric; alias accuracy reported per surface label (§15)
- [ ] **Operator command surface** (§13c): `finetune` → `package` → `extract`, plus `all` = 1+2; extraction never folded into the build
- [ ] **RunPod staging volume** (§14) holds adapters and the merged model between commands 1 and 2; the run manifest goes to Blob either way, so a staged run is still a recorded run

## 21. Glossary — What Each Tool Is, and Why It’s Here

Plain-language definitions for every named technology in this document. Each entry answers two questions: *what is it* and *why does this architecture use it*.

### Training stack

| Term                                         | What it is                                                                                                                                                                              | Why it’s used here                                                                                                                                                                                           |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **PyTorch**                                  | The deep-learning framework that stores tensors, runs them on the GPU, and computes gradients (autograd). Everything else in the stack is written in it.                                | Qwen3-VL is a PyTorch model, and every tool above (Transformers, PEFT, TRL, ms-swift, vLLM) is built on PyTorch. Not a live choice — it’s the ground the stack stands on.                                    |
| **Transformers** (Hugging Face)              | A library of ready-made model architectures — the Qwen3-VL classes, tokenizer, image processor, and chat template are all defined here.                                                 | Gives us Qwen3-VL as loadable code plus the correct chat template, instead of reimplementing the architecture.                                                                                               |
| **TRL** (Transformer Reinforcement Learning) | Hugging Face’s library of fine-tuning trainers for language models, including `SFTTrainer`.                                                                                             | Supplies the training loop we use.                                                                                                                                                                           |
| **`SFTTrainer`**                             | *Supervised Fine-Tuning Trainer* — the training loop itself: it runs forward passes, computes loss, backpropagates, steps the optimizer, checkpoints, and evaluates.                    | This is the loop that actually trains the adapters. “Supervised fine-tuning” means learning from labeled input→output pairs, which is exactly our setup (document in, golden JSON out).                      |
| **ms-swift** (ModelScope-Swift)              | A higher-level fine-tuning framework you drive by CLI or config file. It assembles the model, dataset, collator, and LoRA settings, then launches `SFTTrainer`.                         | Chosen for its first-class Qwen3-VL multimodal support: it handles interleaved image+text collation, `-100` label masking, and multi-adapter workflows out of the box, so we don’t hand-write that plumbing. |
| **PEFT** (Parameter-Efficient Fine-Tuning)   | Hugging Face library that implements LoRA and friends — it inserts the small trainable matrices, saves/loads adapters, and merges them back into the base model.                        | Provides the LoRA machinery for both the Foundation and per-type adapters, plus `merge_and_unload()` in the pipeline’s merge step.                                                                           |
| **bitsandbytes**                             | A library of low-precision GPU operations — 4-bit/8-bit weight storage and 8-bit optimizers. | Only used when `load_in_4bit: true` (§9.2). Under the bf16 default neither the 4-bit base nor a paged 8-bit optimizer is in play, but the dependency stays so flipping the flag needs no environment change. |
| **Accelerate**                               | Hugging Face’s device/distribution layer — decides what runs on which GPU and handles mixed precision.                                                                                  | Lets the same training script run on one GPU or many without code changes.                                                                                                                                   |
| **DeepSpeed / ZeRO**                         | Microsoft’s distributed-training library. ZeRO (“Zero Redundancy Optimizer”) shards optimizer state, gradients, and optionally weights across GPUs instead of duplicating them on each. | Cuts per-GPU memory on multi-GPU RunPod instances. ZeRO-2 shards optimizer state and gradients; ZeRO-3 also shards weights, for tighter VRAM budgets.                                                        |
| **flash-attention-2**                        | A memory-efficient, fused implementation of the attention operation.                                                                                                                    | Our inputs are long (thousands of image tokens plus OCR text). Standard attention memory grows with the square of sequence length; flash-attention makes these sequences affordable.                         |

### Fine-tuning concepts

| Term                                   | What it is                                                                                                                                                                     | Why it’s used here                                                                                                                                              |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Fine-tuning**                        | Continuing to train an already-pretrained model on your own data so it adapts to your task.                                                                                    | Teaches Qwen3-VL insurance-document conventions and strict JSON schema output.                                                                                  |
| **LoRA** (Low-Rank Adaptation)         | Instead of updating billions of weights, freeze them and train a small pair of low-rank matrices alongside each targeted layer. The result is an “adapter” file of tens of MB. | Cheap to train, fast to iterate, and small to store and version — which is what makes one Foundation plus several per-type adapters practical.                  |
| **QLoRA**                              | LoRA applied on top of a base model held in 4-bit precision. | **Not the default here** (§9.2): the base is held in bf16 so the adapter trains against the same weights it is served against. Available via `load_in_4bit` on a VRAM-constrained pod, where it saves ~11 GB of base weights at the cost of that alignment and ~20-40% slower steps. |
| **Rank / alpha**                       | Rank is the size (capacity) of the LoRA matrices; alpha scales how strongly the adapter influences the base model.                                                             | Foundation uses rank 64 / alpha 128 because it learns broad behavior; per-type adapters use rank 16 / alpha 32 because they learn only a narrow schema mapping. |
| **NF4 / double quantization**          | NF4 is a 4-bit number format designed for the way neural-network weights are distributed; double quantization also compresses the quantization constants. | The base-model storage format when `load_in_4bit` is enabled — opt-in, not the default (§9.2). |
| **Adapter**                            | The trained LoRA weights saved as a standalone file, loaded on top of an unchanged base model.                                                                                 | The unit we version, promote, store in Azure Blob, and hot-swap per document type at serving time.                                                              |
| **Modality dropout**                   | Deliberately removing one input channel (here, the OCR text) for a share of training examples.                                                                                 | Trains a *single* model to work both with OCR text and image-only, instead of maintaining two models.                                                           |
| **`-100` masking**                     | The label value PyTorch’s loss function ignores. Tokens marked `-100` contribute nothing to the loss.                                                                          | Applied to system, image, and OCR-text tokens so the model is only graded on the JSON it generates, not on reproducing its own prompt.                          |
| **Epoch / batch size / learning rate** | One epoch is one full pass over the training data; batch size is how many examples are processed before a weight update; learning rate is how large each update is.            | The three knobs with the biggest effect on training outcome — hence the sweep protocol in §11a.                                                                 |
| **Gradient checkpointing**             | Discards intermediate activations during the forward pass and recomputes them during the backward pass.                                                                        | Trades roughly 30% extra compute for a large VRAM saving — necessary with high image-token counts.                                                              |
| **Catastrophic drift / compounding**   | The gradual accumulation of small biases when you repeatedly train on top of a previously fine-tuned checkpoint.                                                               | The reason major Foundation versions retrain from the raw base model and per-type adapters always retrain fresh (§12).                                          |

### Model architecture

| Term                                                 | What it is                                                                                                                                      | Why it matters here                                                                                                                                     |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **VLM** (Vision-Language Model)                      | A model that takes both images and text as input and produces text.                                                                             | Qwen3-VL is a VLM — it can read a page image directly, which is what makes the image-only pathway possible.                                             |
| **ViT** (Vision Transformer)                         | The vision encoder — it cuts the page image into patches and turns them into tokens the model can reason over.                                  | Effectively the model’s own OCR. Frozen by default; unfrozen (via LoRA) only if the eval gate in §3 shows visual reading is the bottleneck.             |
| **DeepStack**                                        | Qwen3-VL’s mechanism for injecting ViT features into *multiple* decoder layers rather than only at the input.                                   | Preserves fine visual detail — small print, table gridlines, checkboxes — which insurance forms are full of. Works even with a frozen ViT.              |
| **Interleaved-MRoPE**                                | Qwen3-VL’s positional encoding across time, width, and height.                                                                                  | Governs page ordering in multi-page documents; the reason page order must be explicit and consistent in the input.                                      |
| **Vision-language projector (merger)**               | The small layer that maps ViT output into the LLM’s embedding space.                                                                            | The literal fusion point between image and text — and therefore the highest-leverage LoRA target for teaching “trust the image when the OCR disagrees.” |
| **Decoder / `q_proj`, `k_proj`, `v_proj`, `o_proj`** | The transformer’s attention projections: query, key, value, and output. Together they decide which parts of the input each position attends to. | Primary LoRA targets, because schema adherence is fundamentally about aligning field labels to the right values (§9a).                                  |
| **`gate_proj`, `up_proj`, `down_proj`**              | The feed-forward network’s projections, which transform each token’s representation individually.                                               | The other LoRA targets — they carry per-token feature transformation (§9a).                                                                             |
| **Token / vision token**                             | The unit a model processes. Text is split into word-pieces; an image is split into patches, each becoming a vision token.                       | Image token count scales with resolution, so a resolution cap is the main lever on cost and latency (§11).                                              |
| **Context window**                                   | The maximum number of tokens a model can process in one call.                                                                                   | Why 50-page policy documents need page selection rather than a single whole-document prompt (§7).                                                       |

### Inference and serving

| Term                                                        | What it is                                                                                                       | Why it’s used here                                                                                                                             |
|-------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| **vLLM**                                                    | A high-throughput inference server for LLMs, with an OpenAI-compatible API and native LoRA adapter hot-swapping. | Serves the promoted model on RunPod and lets us switch per-document-type adapters per request without reloading the base model.                |
| **MinerU**                                                  | An open-source PDF parsing/OCR tool that produces markdown plus layout and page images.                          | Supplies the OCR text half of the OCR-plus-image mode. Its version is pinned per corpus (§8a) because the model learns its output conventions. |
| **Logprobs**                                                | The model’s own probability for each token it generated, returned alongside the output.                          | The raw material for per-field confidence scores — free, requiring no second model (§5).                                                       |
| **Calibration (temperature scaling / isotonic regression)** | Statistical corrections fitted on held-out data that map raw model confidence onto real-world accuracy.          | Fine-tuned models are systematically overconfident; calibration is what makes “0.7” actually mean 70% (§5).                                    |
| **ECE** (Expected Calibration Error)                        | A metric measuring the gap between stated confidence and observed accuracy.                                      | Tells us whether the confidence numbers are trustworthy, not just whether the extractions are correct (§15).                                   |
| **Quantization**                                            | Storing model weights at lower numeric precision to shrink size and speed up inference.                          | Reduces serving cost; the format matrix and quality thresholds are in §13a/§13b.                                                               |
| **GGUF / llama.cpp**                                        | GGUF is a portable quantized model file format; llama.cpp is the runtime that executes it, including on CPU.     | The portable/edge/offline serving option, alongside the primary vLLM path.                                                                     |
| **mmproj**                                                  | The separate multimodal-projector file that a GGUF export of a vision model requires.                            | Qwen3-VL is multimodal, so a GGUF export is incomplete without it — flagged for verification at implementation time.                           |
| **Merge (`merge_and_unload`)**                              | Folding LoRA adapter weights into the base weights to produce one standalone model.                              | Required before quantization and for single-model serving.                                                                                     |

### Data, storage, and operations

| Term                                | What it is                                                                                                     | Why it’s used here                                                                                                |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **JSONL**                           | JSON Lines — a text file with one complete JSON object per line.                                               | The standard training-data format; lets the trainer stream examples without loading the whole corpus into memory. |
| **JSON Schema**                     | A formal, machine-checkable description of a JSON document’s required fields, types, and structure.            | Every model output is validated against it — the structural half of the quality gate (§0a).                       |
| **Pydantic**                        | A Python library that defines data models as classes and validates data against them; it can emit JSON Schema. | Fideon SPEC_00 is written as Pydantic models, and the schema registry is those models serialised (§0a).                  |
| **Golden labels / golden eval set** | Human-verified correct outputs. The eval set is frozen so scores stay comparable across model versions.        | The ground truth for training targets and the promotion gate (§15).                                               |
| **Silver / pre-annotation**         | Machine-generated draft labels, corrected by a human before use.                                               | Faster and more consistent than labeling from blank, and the reason labeling cost falls each cycle (§7).          |
| **Inter-annotator agreement**       | How often two independent human labelers agree on the same document.                                           | Sets a realistic ceiling on model scores — some of the residual error is human disagreement, not model failure.   |
| **Data leakage**                    | When information from evaluation data reaches training, inflating scores without real improvement.             | The reason splitting happens at source-document level *before* modality expansion (§8).                           |
| **Azure Blob Storage**              | Microsoft’s object store for large files.                                                                      | Holds every artifact: raw PDFs, corpora, adapters, merged and quantized models, registries, eval reports.         |
| **RunPod**                          | A GPU rental provider offering both ephemeral pods and serverless endpoints.                                   | Ephemeral pods for training (pay only while training), a persistent serverless endpoint for inference (§14).      |
| **MLflow / Weights & Biases**       | Experiment-tracking platforms that log every run’s config, metrics, and artifacts with a queryable UI.         | Backs the training run registry, alongside the durable manifest in Blob (§12).                                    |
| **Presidio**                        | Microsoft’s open-source PII detection and de-identification toolkit.                                           | Removes PII before data enters Foundation training, per Fideon SPEC_11 (§8b).                                            |
| **PII**                             | Personally Identifiable Information — names, SSNs/TINs, addresses, financial details.                          | Present throughout insurance documents; drives the access-control, tenancy, and pre-annotation rules (§8b, §18a). |
| **Label Studio / Argilla**          | Open-source annotation tools with side-by-side document and label review.                                      | Candidate UIs for the human review step in golden-label creation (§7).                                            |
| **Idempotent**                      | An operation that produces the same result whether run once or many times.                                     | Why a failed pipeline stage can simply be re-run without corrupting state (§13).                                  |
