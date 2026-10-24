# SPEC 12 — Testing / Extraction Routine (CLI Harness)

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-03 (OCR), IMPL-07 (inference core), IMPL-08 (metrics), IMPL-09 (calibration), IMPL-11 (serving pipeline).
>
> **Naming hazard (master §0.1):** the architecture's "**Fideon SPEC_12**" is the external multi-tenant deployment spec. This file is our testing harness. Different documents.
>
> **Architecture refs:** `finetuning-architecture-v2.7.docx` §17 (**extraction/testing routine**), §19 (folder placement), §13a (serving format: the adapter on the bf16 base), §5 (confidence without ground truth).

## Goal

A CLI harness to test any fine-tuned model version — or the base model with a training checkpoint applied — on real documents post-training, producing structured JSON + per-field/overall confidence + quality metrics, with version-organized outputs. The documents are ones the pipeline has already imported and OCR'd, read exactly as serving reads one (page images + MinerU page text). **It mirrors the exact production inference path (MinerU text → model → JSON) by reusing the serving pipeline (IMPL-11), so what you test is what you ship.**

## Deliverables

### 1. Folder layout under `testing/`
```
testing/
├── run_extraction.py            # CLI entrypoint (also reached as `orchestration.run extract`)
├── compare_models.py            # base vs base + adapter vs gold, per document (§6)
├── comparison.py                # field-by-field comparison, the all-fields report, Excel workbooks
├── render_prompts.py            # writes prompts/ from common.prompts
├── prompts/                     # reference renders of the prompts serving sends (generated)
│   ├── acord_25.prompt.txt, acord_125.prompt.txt, acord_140.prompt.txt
│   ├── lossrun.prompt.txt
│   ├── policy.prompt.txt, policy_<line>.prompt.txt               # + *.image_only.prompt.txt
│   └── policy_<line>.{decl,arrays,forms}.prompt.txt              # window prompts, common-model lines
├── ocr_cache/                   # page images fetched from processed/ in Blob, kept between runs
├── results/{version}/           # OUTPUT: canonical JSON per document, organized BY MODEL VERSION
│   └── v2/policy_0001.json
├── results/comparisons/         # compare_models runs, downloaded from Blob (§6)
├── metrics/{version}/           # OUTPUT: per-doc metrics + summary.json
└── test_data/                   # git-ignored; read by no code: documents enter by import + OCR
```
`extraction_registry.json` (§5) is appended by `run_batch` (through `append_registry`) after each document's
results and metrics are written; a document that failed writes no entry.

### 2. `testing/prompts/*.prompt.txt`
One per ACORD form (25, 125, 140), one for Loss Runs, one for the bare `policy` (a package policy, an unclassified line) and one per snapshot line (`common.schemas.SNAPSHOT_LINES`: the seven personal lines) — each in `ocr_plus_image` and `image_only` — plus, for every common-model line, one per section group (`decl`, `arrays`, `lineblk`): the window prompts serving actually sends. Each contains the role framing, the **injected JSON schema for that selection pulled from the IMPL-01 registry** (so it stays in sync), the extraction rules (null-handling, date formats, repeating table rows), and the modality-mode instruction line. They are generated (`python -m testing.render_prompts --write`), never hand-edited.

**Why one file per type, form and line (arch §17):** each has a different schema, different field semantics (a Loss Run's "valuation date" vs. an ACORD's "policy effective date"), and different edge cases. Separate files let a reviewer read exactly what each is asked, and one prompt can be tuned without risk to the others.

**These are the same prompts serving uses.** The harness does not read the files: it renders through `common.prompts` inside `serving.pipeline.extract`, as serving does. The files must resolve to the identical rendered prompt — `test_prompt_files_match_the_renderer` asserts this equality in a test rather than maintaining it by discipline.

**Canonical field mapping needs no change here** (master §1.4): the semantic glosses are schema `description` fields, so they propagate automatically through the existing schema injection. The one requirement is negative — **the extraction path must never import `common.aliases`**. Mapping surface labels to canonical keys is the model's job; a runtime lookup would cap the system at a hand-written list. `test_no_runtime_aliases` (IMPL-14) enforces this.

### 3. `testing/run_extraction.py` — the CLI

```bash
# User decides which model to run — base, v1, v2, v3, ... — on imported, OCR'd documents
python -m testing.run_extraction --model v2   --tenant T --split test --corpus v1
python -m testing.run_extraction --model base --tenant T --split test --corpus v1   # zero-shot baseline

# The base model with a training checkpoint applied per request, not merged
python -m testing.run_extraction --model base \
       --adapter /workspace/staging/adapters/scope/policy/v1/<run>/checkpoint-280 \
       --tenant T --split val --corpus v1

# Or through the operator command surface (IMPL-13), which wraps this:
python -m orchestration.run extract --model v2 --tenant T --source-ids policy_0001 policy_0002

# Image-only mode (no OCR text), to test the no-OCR pathway
python -m testing.run_extraction --model v2 --tenant T --source-ids policy_0001 --mode image_only

# Extract without scoring against the gold labels
python -m testing.run_extraction --model v2 --tenant T --split test --corpus v1 --no-score
```

- `--model` resolves via IMPL-02 `resolve_model_version` — **the user never has to know paths, just the tag.** Accepted values:

  | Value | Resolves to |
  |---|---|
  | `base` | Untuned `Qwen/Qwen3-VL-8B-Instruct` at the pinned revision, **no adapter** |
  | `base` + `--adapter <folder>` | The base with that LoRA applied on every request, **not merged** — a `checkpoint-N` folder or a run's adapter folder, local or a Blob prefix (copied once to `/workspace/adapters`). Refused with any other `--model`, without `adapter_config.json`, above the engine's `max_lora_rank`, or with `enable_lora: false` |
  | `v1`, `v2`, … | The policy adapter's run of that version (`policy-v2`): the base with that version's adapter applied per request, staged or published — nothing is merged (decision 2026-10-23). A staged version loads the checkpoint its selection chose, else its latest checkpoint on the staging volume (with a warning), else it is refused. A version trained on another base than the configured `model_id@revision` is refused. For an older unified or Foundation run, its merged model plus a graduated per-type adapter where the routed type has one |
  | `v2` + `--format fp8` | Refused for a version of the policy adapter: it is a LoRA on the bf16 base, so there is no FP8 model. For an older run, that version's FP8 serving weights (refused while the version is still staged). A GGUF format here is a category error: llama.cpp loads those, and the endpoint runs vLLM |

  **`base` is not a curiosity** — it is the zero-shot baseline of the pilot protocol (IMPL-15) and the day-zero pre-annotation path (IMPL-04). Supporting it here means those two have one implementation, not three.
- **Which documents:** `--source-ids …` or `--split train|val|test` with `--corpus vN` (that corpus manifest's `source_ids_by_split`) — mutually exclusive; with neither, every labelled document of `--doc-type` (default `policy`) for `--tenant`. `--limit N` keeps the first N.
- `--mode ocr_plus_image` (default) | `image_only` — both production pathways testable from one harness.
- **Scoring is on by default**: a document with a gold label in `golden-labels/` is scored against it; one without, or whose gold has no review record (decision 2026-10-22), is extracted and not scored — the metrics file says why (`ground_truth: not scored: …`) and `summary.json` lists it under `not_scored`. `--no-score` extracts without comparing.
- Additional: `--label` (the name results are filed under; default the tag, or `base+<run>-<checkpoint>` with `--adapter`), `--format`, `--limit`, `--tenant`.

**What it does, step by step (arch §17):**
```
For each source_id:
  1. Read             → page count from processed/…/ocr_meta.json; every page image (cached in
                        ocr_cache/) and MinerU's per-page markdown as page_texts (none under
                        image_only); the label metadata (acord_form, lob) and the gold label
  2. Doc type         → known: --doc-type with the label's ACORD form (StaticClassifier);
                        the policy's line from the label metadata (classic auto → personal auto)
  3. Prompt select    → rendered by common.prompts for that type, form, line and window group
  4. Assemble input   → system = prompt (schema + instructions)
                        user   = [page image + page marker (+ OCR markdown, unless image_only)] per page
  5. Model inference  → the selected model generates JSON + token logprobs (policies: in windows over
                        every page, merged)
  6. Confidence       → per-field confidence from logprobs; no release calibrators are loaded, so the
                        identity transform applies and every field is flagged for review
  7. Validate         → serving's audit verdict (serving.pipeline.schema_verdict): JSON parses +
                        conforms to that selection's schema (strict_schema=False: an invalid
                        answer is kept and flagged schema:invalid)
  8. Write result     → results/{version}/{source_id}.json, with all_fields
  9. Metrics          → scored against the gold label → metrics/{version}/{source_id}.metrics.json
 10. Summarise        → metrics/{version}/summary.json; a failed document is listed, the batch carries on
```

**Steps 2–7 are executed by calling `serving/pipeline.py` (IMPL-11)** — the routine is a thin CLI wrapper over the serving pipeline plus document reading and metrics. It must **NOT re-implement inference**; doing so breaks test == prod, which is the entire point of the harness.

### 4. Metrics output

**Results `results/{version}/{doc}.json`** — the serving output contract (`ExtractionResult.as_dict()`: `extraction` is the answer — for a policy, the canonical JSON with every key of its line's schema — then `fields`, `list_fields`, `line_of_business`, `pages_used`, `review_flags`, `route`, and `reconciliation` for a Loss Run) with **`all_fields`** inserted after `list_fields`: every field of the selection's schema (`testing.comparison.field_report`) — those neither the answer nor the gold holds too — each with its `value` and `status` (`extracted`, `null`, `no_rows`, `not_in_output`, `system`, `structural`), a written value's `confidence` and `page_ref`, its review `flags`, and against a gold label its `gold` value and `result` (`correct`, `wrong`, `missed`, `invented`, `empty`); plus a `summary` of counts and the `document_flags`.

**Per-doc `metrics/{version}/{doc}.metrics.json`** (arch §17):
```json
{
  "document": "policy_0001",
  "model_version": "v2",
  "doc_type": "policy",
  "mode": "ocr_plus_image",
  "schema_valid": true,
  "overall_confidence": 0.88,
  "pages_used": [1, 2, 3, 4, 5, 6, 7, 8, 9],
  "review_flags": ["coverages[3].limit:merge_conflict", "window_failed:arrays:p9"],
  "fields": {
    "policy.policy_number":   {"value": "...", "confidence": 0.94, "correct": true},
    "policy.expiration_date": {"value": "...", "confidence": 0.71, "correct": false}
  },
  "reference_accuracy": 0.91,
  "coverage_code_accuracy": 0.95,
  "field_exact_match_rate": 0.91,
  "field_normalized_match_rate": 0.93,
  "field_accuracy_by_field": {"...": "..."},
  "list_field_f1": {"coverages": 0.88},
  "list_field_recall": {"coverages": 0.83},
  "missed_rows": {"coverages": 2},
  "failures": [{"field": "policy.expiration_date", "expected": "...", "got": "..."}]
}
```
All correctness uses IMPL-08 metrics — `score_fields`, `score_all_list_fields`, and `values_agree` for each field's `correct` (dates, amounts and names normalised, numbers compared by their declared type) — the same definition of "correct" as the promotion gate. The gold label is narrowed to what the line's schema can hold (`common.canonical.schema_label`) and the system-filled fields are left out of both sides. On a **common-model line** both sides are narrowed, ids are compared through what their links name (`common.structural_ids.comparable_for`), table values are scored (`core_for_scoring`, as `run_eval.score_subset` does), and links, codes and overflow get numbers of their own (`CommonModelTally`: `reference_accuracy`, `coverage_code_accuracy`, `coverage_code_recall`, `additional_fields_recall`, `additional_fields_precision`). Without a gold label the file says `"ground_truth": "not supplied — confidence reported, accuracy not measurable"`.

**Batch summary `metrics/{version}/summary.json`** — aggregates across the batch for that model version: document count, schema-validity rate, mean confidence, mean latency per document, documents needing review, per-doc-type and per-mode counts, the failed documents with their errors, and — over the scored documents — mean field normalized match and mean list-field recall. **This is the file compared across model versions to answer "did v3 beat v2?"** — the testing-time mirror of the IMPL-08 promotion gate.

**On the confidence-without-ground-truth case (arch §17):** even with no golden JSON — a brand-new unknown document — the routine still emits `overall_confidence` and per-field confidence. That is the whole payoff of the calibration work: confidence is available at inference time on documents you have never labeled, which is exactly what routes low-confidence extractions to human review in production. (The harness loads no release calibrators, so its numbers are the identity transform of the raw signal, and every field is flagged; a release's calibrated confidence is measured by the golden eval, IMPL-08.)

### 5. `testing/extraction_registry.json`
One entry per extraction, so an output JSON is never ambiguous about its origin — written by `append_registry`, which `run_batch` calls after each document's results and metrics are written (§1):
```json
{
  "extractions": [
    {
      "document": "policy_0001",
      "doc_type": "policy",
      "model_version": "v2",
      "quant_format": null,
      "mode": "ocr_plus_image",
      "result_path": "results/v2/policy_0001.json",
      "metrics_path": "metrics/v2/policy_0001.metrics.json",
      "overall_confidence": 0.88,
      "schema_valid": true,
      "review_flags": [],
      "extracted_at": "2026-02-15T11:03:00Z"
    }
  ]
}
```
Combined with the `results/{version}/` layout, anyone can see at a glance that `results/v2/policy_0001.json` came from model v2 — from the path *and* the registry. `model_version` here is the same tag that resolves to a `run_manifest.json` (IMPL-02), so an extracted result traces all the way back to the corpus version and git commit that produced the model that generated it.

### 6. `testing/compare_models.py` — base vs base + adapter
```bash
python -m testing.compare_models --input policy.pdf --gold policy_gold.json --lob homeowners \
    --reviewer-id <reviewer> --review-date YYYY-MM-DD --adapter <checkpoint-N>
python -m testing.compare_models --input data/bundles/<document>/ --adapter <adapter folder>
python -m testing.compare_models --input <folder of document folders> --adapter <adapter folder>
```
- `--input` is a PDF (with `--gold` and `--lob`), a bundle folder (`document.pdf` + `golden.json` + `metadata.json`), or a folder of them. Each document is imported and OCR'd (in `/workspace/venv-ocr`) under its own tenant — `compare` by default, never `default` or the real data's — and a re-run finds it by checksum. A gold is compared only with a review record (decision 2026-10-22): a PDF's `--gold` takes `--reviewer-id` and `--review-date`, written into its `metadata.json`; a bundle folder carries its own. Without one the answers are compared with no gold.
- One model load (`load_model("base", adapter=…)`, and `without_adapter` for the base on the same engine); each document is extracted twice through `serving.pipeline.extract`, by the base alone and with the adapter applied, not merged. No calibrators: every field is flagged in both.
- Per document under `--out` (default `testing/results/comparisons/<label>-<stamp>/`): `document.pdf`, `gold.json`, `base.json`, `adapter.json` (canonical JSON, every schema key), `base.all_fields.json`, `adapter.all_fields.json`, and `comparison.xlsx` (Summary and Fields sheets: each field's gold, base and adapter values, results, and `fixed`/`broke`). Table rows are matched by their identifier (a VIN, a form number, a coverage name), not by position. Across documents: `summary.xlsx` (one row per document, pooled totals) and `errors.json` when a route failed.
- The finished folder is uploaded to Blob under `exports/{tenant}/comparisons/{run}/` (the raw container: it holds policy data), checked file by file, and removed from the pod (`--keep-on-pod` keeps it). The command prints the `azcopy` line that downloads it to the laptop, by default into `D:\Fine-Tuning\testing\results\comparisons` (git-ignored; `--download-to` changes it).

## Constraints
- Confidence emitted even without ground truth.
- **Same prompts and same pipeline as serving — reuse, never reimplement.**
- Documents are read from the OCR the pipeline wrote (`processed/`), produced by the pinned MinerU (IMPL-03); the corpus build refuses documents read by more than one MinerU. The harness does not re-check the pin.
- No PII in logs; `results/`, `metrics/`, `ocr_cache/` and `test_data/` are gitignored, and comparison runs leave the pod through the raw container only.

## Acceptance checklist
- [ ] `run_extraction --model v2 --split test --corpus v1` writes `results/v2/*.json`.
- [ ] `--model base` runs the untuned base model with no adapter and writes `results/base/*.json`.
- [ ] `--model base --adapter <checkpoint>` applies the LoRA per request without merging, and files results under `base+<run>-<checkpoint>` (or `--label`); `--adapter` with any other `--model` is refused.
- [ ] `orchestration.run extract` produces byte-identical output to calling `run_extraction.py` directly.
- [ ] `--mode image_only` runs with no OCR and still extracts.
- [ ] Per-field + overall confidence present **with and without** ground truth.
- [ ] `line_of_business` present in every result, with confidence, `null` when undetermined.
- [ ] With a gold label that has a review record, per-doc + summary metrics computed, including list-field recall; `--no-score` skips them; a gold with no review record is not scored, and the metrics file and `summary.json` (`not_scored`) say why.
- [ ] Every result carries `all_fields`: every field of its schema, with its value and, against gold, its result.
- [ ] `--model vN` loads the base with that version's adapter applied, not merged; a non-bf16 `--format` is refused for it (a GGUF format is a category error here: the harness runs vLLM). A staged version loads its selected checkpoint, else its latest one with a warning, else it is refused; a version trained on another base than the configured one is refused.
- [ ] `extraction_registry.json` links result → model version → run manifest.
- [ ] `testing/prompts/*.prompt.txt` render identically to `common.prompts` for the same inputs (asserted, not assumed).
- [ ] **Output equals the serving pipeline's output on the same input** (parity test).
- [ ] `compare_models` writes base and adapter JSON, both `all_fields` reports and `comparison.xlsx` per document, `summary.xlsx` across them, under its own tenant, and uploads the run to `exports/{tenant}/comparisons/`.

---

## Current implementation (2026-10-21)

- Both entries run the extraction: `python -m testing.run_extraction …` and
  `python -m orchestration.run extract …`, which passes the same arguments to `run_extraction.main`. Extraction
  goes through `serving.pipeline.extract` — windows and the window merge for policies, the serving guards,
  the printed-values check, every-key output, dates in `MM/DD/YYYY` — so test output is production output.
- **Every page is read** (decision 2026-10-19, IMPL-11 §4): a policy of any length is read in windows over
  all its pages, never on selected pages, so `pages_used` lists every page and a long policy takes more
  windows than it did. Results written before 2026-10-21 read a policy longer than five pages on its selected
  pages only. Nothing in the harness selects pages: there is no page threshold and no `long_doc_types`.
- Documents are read from Blob as the pipeline wrote them (`document_request`): `processed/` page images and
  MinerU markdown, `golden-labels/` gold and metadata. A document not yet OCR'd is refused with what to run;
  one failing document is recorded in `summary.json` and the batch carries on (exit 1 only when every document
  failed).
- A policy's line comes from its label metadata (`lob`); without it the fallback canonical schema (the common
  model) is used. No serving plan is passed, so nothing is refused for its line.
- No calibrators are loaded: confidence is the identity transform and every field is flagged for review. This
  run measures extraction, not a release's review thresholds.
- `schema_valid` and the `schema:invalid` flag are serving's own verdict (`serving.pipeline.schema_verdict`, IMPL-11
  §5): on a common-model line the served answer, every key filled, is what is judged.
- The gold is read from `golden-labels/` under the form-edition rule where it holds — a common-model policy
  line (`document_request` → `label_rules.under_edition_rule`) — as the gate's golden eval reads the frozen set
  (IMPL-08): the served answer there has a form's edition moved out of its number into `edition_date` (IMPL-11),
  and a gold imported before that rule is not rewritten in Blob, so scored as stored it would count the
  convention wrong. A self-contained line's gold is scored as stored: nothing on it moves an edition.
- A gold with no review record (`export_golden_labels.review_problem`, decision 2026-10-22; a twin inherits its
  seed's) is not scored: `document_request` returns no gold and logs why, the extraction still runs and is
  written, the metrics file records `ground_truth: not scored: …`, and `summary.json` lists it under
  `not_scored`. `compare_models` reads golds the same way.
- On the pod `run_extraction` and `compare_models` run detached in tmux like every long job; they survive a
  closed laptop.
- The gate's numbers do not come from this harness: they come from the frozen golden eval set
  (IMPL-08 `golden_eval.evaluate_version`), through the same serving pipeline.

## Changes since 2026-10-05

- Added the alignment line and dated *Current implementation* 2026-10-13.
- Goal and §1: documents are imported and OCR'd ones read from Blob, not PDFs dropped into `test_data/` (now unread); added `compare_models.py`, `comparison.py`, `render_prompts.py`, `results/comparisons/`, `summary.json`; the registry file is not written by the CLI.
- §2: real prompt file names (per ACORD form, per line, image-only variants, window prompts per group); the harness renders through `common.prompts` rather than reading the files.
- §3: replaced `--input`/`--ground-truth` with `--source-ids` / `--split` + `--corpus` / `--tenant` / `--doc-type` / `--no-score` / `--label`; added `--model base --adapter <checkpoint>`; corrected `--format` (FP8 serving weights, not GGUF); rewrote the step list (read from Blob, static classification, identity calibration, `strict_schema=False`, all_fields, summary).
- §4: results JSON carries `all_fields`; per-doc metrics keys as written (typed `values_agree`, common-model scoring and tally metrics); summary is `summary.json` with the aggregates actually computed (no ECE or per-value LoB accuracy).
- §5: noted that `append_registry` is not called by `run_batch`.
- §6 (new): `compare_models` — inputs, compare tenant, outputs (`base/adapter.json`, `*.all_fields.json`, `comparison.xlsx`, `summary.xlsx`), upload to `exports/{tenant}/comparisons/{run}/` and the azcopy download.
- Constraints and acceptance checklist updated to match.
- `run_batch` now writes `extraction_registry.json` (commit f5668a4).

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx`.
- §3: a policy is read in windows over every page; validation is serving's audit verdict (`schema_verdict`).
- §4: the `pages_used` example lists every page.
- §5: `run_batch` writes the registry (the stale "does not call today" removed).
- *Current implementation*: every page read, no page selection, results before 2026-10-21 read selected pages; `schema_valid` is serving's verdict; golds read under the form-edition rule, as the golden eval reads them.

## Changes of 2026-10-22 (code review)

- §2 and *Current implementation*: a gold with no review record (decision 2026-10-22) is extracted and not scored, and the metrics file and `summary.json` (`not_scored`) say why.
- *Current implementation*: golds are read under the form-edition rule only on common-model policy lines (`label_rules.under_edition_rule`).
- §6: `compare_models --reviewer-id --review-date` give a PDF's `--gold` its review record; without one the answers are compared with no gold.
- Acceptance checklist: scoring needs a reviewed gold.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.

## Changes of 2026-10-22 (one adapter)

- One adapter (decision 2026-10-22): `--model vN` resolves the policy adapter's run (before, only unified/Foundation runs were found); the adapter path example is `scope/policy/`; the line prompts snapshotted are those of `SNAPSHOT_LINES`.

## Changes of 2026-10-23 (full text)

- Each document's metrics carry `text_coverage` (the share of its page text's words found in the served JSON, SPEC_21's every-word rule); `summary.json` adds `mean_text_coverage`, `min_text_coverage`, `documents_below_full_text` (under `FULL_TEXT_COVERAGE` = 0.99, chosen, not measured) and `documents_without_full_text` (flagged `text_sections:no_page_text`). The full-text tier is never scored: scoring narrows the answer to the model view, which has no `text_sections`.
- `testing/extract_server.py` (pod) and `testing/extract_client.py` (laptop): a PDF posted through the pod's RunPod proxy URL is OCR'd by MinerU in the OCR environment (`data_pipeline/ocr/ocr_worker.py`, kept loaded) and read by `serving.pipeline.extract` with the base model or `--adapter`; jobs answered at once and polled (the proxy's ~100 s limit), one at a time, token-guarded; each reports pages, windows planned, model calls, OCR/model/total seconds, schema validity, review flags by kind and `text_coverage`. `EXTRACT_SERVER_TOKEN` in `.env.example`.
- `testing/latency_check.py`: times the extraction routine on PDFs against a model already served (`vllm serve`, `inference_core/openai_backend.py`: one chat request per window, run concurrently; page images as base64 data URLs; the schema slice as `response_format`; the end-of-turn token dropped from the scored tokens) and a served MinerU 4 API (`data_pipeline/ocr/mineru_api.py`: `/v1/tasks`, `structured_content` written per page as the OCR stage writes it). Reports render, OCR (and the server's own OCR seconds), model and total seconds, model calls and MB sent per document; run it next to the servers, since page images go with every window. `FIDEON_REMOTE_TOKEN` in `.env.example`. A served MinerU 4 is not the corpus pin (3.4.5): for timing and checking, not training data.

## Changes of 2026-10-23 (served model speed)

- `data_pipeline/ocr/served_ocr.py` (`ServedOcr`): renders a PDF's pages in several processes (`render_pdf_pages(..., workers=)`, the same bytes as one process) while a served MinerU reads it, optionally as several runs of pages at once (`ocr_via_api(..., parts=)`, `split_pdf`); writes `page_NNN.png`/`.md`; also the job server's OCR call.
- `testing/latency_check.py` reads through it, sends each window as it can (`--concurrency`, default 128), stops repeating answers (`--no-repeat-stop` to compare), can send images by path (`--images file`) or ask for a served adapter (`--served-model`), and scores each document against its gold JSON when one is found (`--gold-dir`, or `Gold_json` beside the PDF's folder; `testing.run_extraction.score_against_ground_truth`). Each document's JSON carries `window_stats` and a `windows` summary (rounds, splits, failures, cut off, stopped repeating, prompt and answer tokens, the slowest window); the table adds pages-and-text seconds, rounds, splits, answer tokens, the slowest window and exact match.
- `testing/extract_server.py` can use the pod's own servers (`--model-url`, `--mineru-url`, `--remote-token`, `--images`, `--tier`, `--ocr-parts`, `--served-model`): it then loads neither vLLM nor MinerU, and only the PDF crosses the proxy. Its summary adds `windows` and the rendering/OCR timings; the result JSON adds `window_stats`.

## Changes of 2026-10-23 (page bytes)

- `testing/latency_check.py` from a laptop through the proxy: gray pages go as grayscale PNGs (the same pixels to the model; `--color-pages` to compare); `--images link` uploads each page once to Blob while MinerU reads the PDF - after the PDF is at MinerU, so the two do not share the connection (`ServedOcr.read(publish_pages=)`, `ocr_via_api(uploaded=)`) - and sends the model read-only links (`--link-token` / `$PAGE_LINK_SAS_TOKEN`, checked at start; `--keep-links`); `--ocr-parts N` uploads the runs of pages one after another, each polled from the moment it is in, so MinerU reads the first while the rest upload; `--mineru-token` when MinerU's token is not the model's (a RunPod serverless endpoint's is the API key). `Sent MB` is now everything that left: the PDF, the page links and the model's requests (`sent_mb_parts`).

## Changes of 2026-10-23 (incremental training)

- Nothing is merged (decision 2026-10-23): `--model vN` is the base with that version's adapter applied per request (`query_registry` kind `base_plus_adapter`), and a non-bf16 `--format` is refused for it (§3 table, the registry example, acceptance).
- Code review of the incremental training: `--model vN` refuses a version trained on another base than the configured one, and a staged version loads its selected checkpoint, else its latest with a warning, else it is refused (§3 table, acceptance).

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.

## Changes of 2026-10-24 (one served MinerU)

- `testing/latency_check.py` and `testing/extract_server.py` read with the pinned MinerU (`configs/shared/ocr.yaml`): `--tier` defaults to the pinned tier, every task's version and tier are checked, and `--any-mineru` accepts another (timing only); `latency_check` refuses a `--tier` other than the pinned one without it. `--ocr-mode` defaults to the OCR stage's rule, per document, rather than MinerU's `auto`.

## Changes of 2026-10-24 (fields first)

- `testing/latency_check.py --runs standard,fields-first` reads each document both ways, one row each (`time_fields_first`): `fields` (the gate) and `full_text` (when the tier joined) next to the standard run's time, with exact match against gold for both - the measurement that decides whether image-only fields can be the production path.
- `testing/extract_server.py --fields-first` (needs `--mineru-url`): a job becomes `fields_ready` once the model has read the page images (its result already written, `/result` answers), and a second worker adds MinerU's text in the GPU's idle turns before `done`; a failed full text leaves the fields standing, reported as the job's error. `ExtractService.submit` takes the server's default mode. `testing/extract_client.py` waits through `fields_ready`, says when each document's fields came, and reports `Fields s`.
- The window summary adds `first_token_s_max` (windows queued, or put back for want of memory) and `by_group`: each job's answer lengths (median, p95, longest), to set the answer limits from.
- Code review of 2026-10-24: `latency_check` scores a fields-first document twice - when the fields come (`accuracy_at_fields`, the `Exact match at fields` column: what the gate serves) and as finally served, after the page text could change it. The job server writes each result whole (`write_whole`: a temporary file swapped in), since a `fields_ready` result is served while its full text is added; takes `--run-pages` (20) and `--text-max-wait` (600 s); keeps a job's fields for its second phase only once its first has succeeded. The client reports a document whose fields came but whose full text failed, and exits 1 for it. From the laptop with `--images link`, fields first puts the page upload before the model (nothing to hide it behind, unlike OCR): about 1.5-2 minutes for 114 pages at 1 MB/s; negligible on Azure.
- Fields first serves its fields only once held to the pages they cite (IMPL-11): the job server reports `fields_served`, `fields_verified` and `cited_pages`, becomes `fields_ready` only with checked fields, and a job whose full text then fails is `done` with its checked fields, or `failed` when none were served. `latency_check` scores the fields as served at the gate (held-back values included) and the final result, counts `fields_check.held_back`, and times `verify` inside `fields`; `--unverified-fields` scores unchecked fields, for timing only. Tests: a value cited on the wrong page is held back, then served once found; an invented value is never served; fields whose pages cannot be read are not served before the full text.

## Changes of 2026-10-24 (printed values only and consistency)

- Test outputs are served outputs, so they hold printed values only (IMPL-11; the user's decision of 2026-10-24): in strict mode, the default, a value no page prints is null in `extraction` and in `all_fields`, flagged `<path>:not_printed_removed`; against gold, a removed value the gold holds counts as missed. `printed_values.mode: flag` in `configs/inference/vllm_serving.yaml` keeps such values and flags them `<path>:not_printed` instead. What the check found (`ExtractionResult.printed_check`) is not written to the results JSON.
- Fields first: the fields served at `fields_ready` are held to the pages they cite (`serving.fields_first.verify_fields`, IMPL-11); a value not found on them yet is held back (`pending_full_text`), and the whole document is checked when MinerU's full text arrives, before the full-text tier. So `latency_check`'s `Exact match at fields` scores the fields as first served, and its final score the fully checked ones.

## Changes of 2026-10-24 (QA fixes)

- The served MinerU client's QA fixes (IMPL-03) apply to the test runners; a non-pinned `--tier` is refused unless `--any-mineru` (the job server too). See IMPL-11 for the merge, printed-values and transport fixes.
- `extract_client`: a Status column (`done`, `fields served, full text failed: ...`, `FAILED: ...`, `NOT SENT: ...`); a timing the server did not report is left blank, never a crash; `summary.*` is always written; exit 1 unless every document was read in full. Lines and PDF sizes are checked before anything is uploaded (`/health` reports `max_pdf_bytes`); the documents already queued are still collected after a refusal; `--url` must be HTTPS or HTTP to this machine, redirects are refused, and server errors are reported, not raised. Output names never clash (`-2` added; `summary` reserved). The job server answers 401 for a non-ASCII token and 400 for a bad Content-Length, lists its jobs under a lock, and refuses a non-pinned `--tier` and plain-HTTP server URLs before the detach. `latency_check` canonicalises each `--doc` line and refuses an unknown one, refuses a run named twice and `--concurrency` below 1, and checks the URLs and the link login before the detach (`setup_problem`, against the environment the run will have).

## Changes of 2026-10-24 (architecture v2.7)

- Header: aligned with `finetuning-architecture-v2.7.docx`, which records the decisions of 2026-10-23 and 2026-10-24: printed values only, consistent output, fields first, one MinerU, production on Azure.
