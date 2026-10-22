# SPEC 11 — Serving (vLLM Endpoint + Classifier + Routing)

> Aligned with the code of 2026-10-22.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-07 (inference core), IMPL-09 (calibration), IMPL-10 (postprocessing).
>
> **Naming hazard (master §0.1):** the architecture's "**Fideon SPEC_11**" is the external Presidio de-identification spec (implemented in our IMPL-05). This file is our serving layer. Different documents.
>
> **Architecture refs:** `finetuning-architecture-v2.5.docx` §0a (audit gate contract), §0b (LoB output), §0e (decisions of 2026-10-19: every page read), §4a (**classifier as a load-bearing component**), §4b (ACORD two-level), §4c (day-zero classifier), §5 (confidence), §7 (long documents: every page, read in windows), §14 (**RunPod split: what runs where**).

## Goal

Serve the promoted model on a RunPod Serverless vLLM endpoint with per-release adapter routing, document-type classification, windowed reading of every page of a long document, and calibrated confidence post-processing — built on the shared inference core so **serving == eval == testing**.

## The infrastructure boundary (arch §14) — get this right or the design leaks

```
Training (ephemeral pod)              Serving (persistent endpoint)
─────────────────────────             ─────────────────────────────
clone repo @ commit                   vLLM + promoted model
pull base + corpus (Blob)                     │
run LoRA training                     request ──> classify doc type
push adapter + manifest (Blob)                │
terminate                             select + hot-swap adapter
                                              │
External orchestrator                 inference + logprobs
triggers pods via RunPod API                  │
                                       calibrate confidence
Application logic (CPU infra,                 │
outside RunPod): OCR, prompt          return JSON + confidence
assembly, endpoint calls,
confidence calibration
```

**RunPod is scoped purely to GPU-bound work.** Application/orchestration logic — calling MinerU, assembling the prompt, calling the endpoint, applying the calibration transform, returning final JSON — runs in **your own service** (Azure Function, App Service, or AKS), not inside RunPod. This keeps business logic on cheap CPU infra, keeps it portable if the GPU provider changes, and keeps the RunPod handler small. Document this boundary in the module docstring; it is an architectural constraint, not a deployment preference.

## Deliverables

### 1. `serving/vllm_entrypoint.py`
- RunPod Serverless handler wrapping vLLM, serving the promoted merged/quantized model over an OpenAI-compatible API, **with logprobs enabled** (calibration depends on them).
- **One LoRA per request, never a reload.** With one promoted release covering every line, the engine is that release's merged model and no adapter is applied. With several releases — or one that leaves lines to the base model — the engine loads the **base with LoRA enabled** and each release is its adapter, applied per request (`serves_base_with_loras`, `load_release_runtimes`). A §4.2 graduated per-type adapter would be applied the same way, from the classifier result. One LoRA per request is what vLLM supports, which is exactly why the v1 Foundation-plus-per-type stack could never have been served.
- Pulls the promoted releases' adapters and calibrators from Blob on cold start (IMPL-02); version configurable. The model itself is handed to `cold_start` — loading it there is not wired yet, and `cold_start` refuses without one.
- Uses IMPL-07 `model_runner` / `input_builder` for generation so output matches eval and testing exactly.
- **Asserts the serving MinerU version matches the training corpus pin** (IMPL-03 `assert_version_matches`) wherever MinerU is installed — a mismatch is distribution shift and a regression trigger (arch §8a), not a warning to ignore. A serving pod without MinerU does no OCR; it logs the pin its requests' page texts must come from.
- **Every tuneable in `configs/inference/vllm_serving.yaml` must have a reader.** `serving_thresholds()` lifts `routing.classifier_confidence_threshold`, `confidence.review_threshold`, `routing.detect_lob`, `routing.lob_confidence_threshold`, `routing.lob_family_threshold` and `routing.fallback_doc_type` out of the config and passes them into `extract`; `release_pins()` reads `routing.release_pins`. (There is no page threshold to read: every page is read, in windows planned by the function the corpus build planned its training rows with, so nothing in serving's config can re-shape documents away from what was trained. The `long_documents` block keeps only `conflict_resolution: declarations_page_wins`, which names the rule the window merge applies by group order; no code reads it.) They were declared in that file and read by no code, so an operator could raise `review_threshold`, redeploy, and change nothing at all — the hardcoded defaults inside `extract` won silently every time. A test asserts each key the helper emits is a parameter `extract` accepts.
- **Calibration is selected after classification, not before.** The handler hands the whole `{doc_type: params}` map to `extract`, which picks the entry once the classifier has said what the document is. Resolving it in the handler from the caller-supplied `doc_type` meant a lookup on `""` for every request that did not name its own type — which is every classification-driven request, the endpoint's entire purpose — and the request was refused before extraction ran. `assert_calibration_present` accepts the map and checks it is non-empty; an empty one is still no calibration. It is checked only when no release is served, or some served release has no calibrators of its own.
- **The adapter map holds adapters that exist.** `resolve_model_version(..., doc_type=…)` constructs `type_adapter` unconditionally — it is a path, not evidence anything was trained — so `build_adapter_map` confirms the directory holds an `adapter_config.json` before mapping a doc type to it. Trusting the path mapped all three types on a `--foundation-only` build, and vLLM was then handed a `LoRARequest` for a directory that does not exist, failing every request instead of serving Foundation-only with a `routing:no_adapter_available` flag.
- **No plaintext PII in logs**; do not persist raw request/response bodies (master §8). The payload log call sits **inside** the handler's `try`: `safe_log_payload` iterates the payload, so an `input` that is not a JSON object raised straight out of the handler and RunPod returned an opaque platform error — the exact failure the structured-error contract exists to prevent.

### 2. `serving/doc_type_classifier.py`
**This is load-bearing, not preprocessing (arch §4a).** If classification is wrong, you load the wrong adapter *and* the wrong prompt *and* the wrong schema, and the extraction fails no matter how good the model is.

```
document ──> classifier ──> confidence >= threshold ? ──> yes ──> select adapter + prompt + schema
                                     │                              (acord -> + form number)
                                     └── no ──> Foundation-only extraction + human routing review
```

- Identifies doc type ∈ {`acord`, `policy`, `lossrun`} **and the ACORD form number** — the two-level classification of arch §4b. Returns label + confidence.
- Reads a document's **first two pages** — their images by page number and their text joined, cut at 4,000 characters (`classifier_input`, `classifier_messages`) — the same rule a training row is built by.
- **Default implementation: zero-shot via the Foundation model** (arch §4a Option C) using `doc_type_classifier_prompt.jinja` — zero extra infrastructure, and the Foundation already understands these document types.
- **Also reads a policy's line of business** when asked (`Classification.lob`), with the probability of the line's own tokens as its confidence (`line_confidence`) — never a number the model writes about itself. Used only with `routing.detect_lob` on, which ships **off** (§6, *Current implementation*). With detection on and no classifier supplied, the endpoint builds `ZeroShotClassifier` on the engine's base weights with no adapter (`line_classifier`) — only when the engine holds the base; an engine holding one release's merged model reads no line. The line decides which release's adapter reads the policy, so no adapter can have been chosen yet.
- Interface is swappable for a **dedicated vision classifier** (Option B) later, which is the right escalation if classification accuracy becomes a *measured* bottleneck — it works in both OCR and image-only modes, unlike a text-feature classifier.
- **Day-zero (arch §4c):** before Foundation v1.0 exists, the classifier runs against the **base** `Qwen3-VL-8B-Instruct`. This path is explicitly temporary and is superseded on promotion of Foundation v1.0, after which the base model is retired from the production inference path.
- **Has its own accuracy target and is measured on the golden eval set** alongside extraction metrics (IMPL-08 `classifier_accuracy`) — a classifier at 92% caps the whole system at 92%.
- A request that names its `doc_type` (and `acord_form`) skips classification (`StaticClassifier`); with no classifier held and detection off, a request without `doc_type` has nothing to classify it.

### 3. `serving/adapter_router.py`
- Maps classifier result → (adapter path, prompt file, schema). Deterministic selection.
- **Low-confidence fallback (arch §4a):** when the classifier isn't confident, **don't silently guess**. Fall back to **Foundation-only extraction** with a generic schema-agnostic prompt and flag the document for human routing review. A wrong-adapter extraction is worse than a slightly-generic one.
- Runs **before** adapter selection in both the serving path and the testing routine.
- For ACORD, resolves the form number to the matching schema even when a single shared `acord` adapter is in use (arch §4b Option 1) — the classifier must identify the specific form so the correct schema is always selected.

### 4. `serving/page_router.py`
Long-document handling, primarily a **Policy-document** concern (arch §7). Policy documents can run 50+ pages; each page at full resolution costs 1,000–2,000+ vision tokens, so a whole policy cannot go into one prompt. **Every page is read, chunk by chunk, never selected pages** (decision 2026-10-19, arch §0e). Page selection was removed: a page that scored nothing was never read, at training or at serving, and whatever it printed was silently missing.

1. **Declarations page** — `plan_pages` returns **every page**, in order, and the first page whose MinerU text matches a declarations signal (`declarations_page`). The other `PAGE_SIGNALS` (coverage schedule and coverage summary 6; schedule 5; named insured, policy number, policy period 4; premium, deductible, limits of insurance 3; endorsement, producer 2) stay on the plan's scores for tracing and decide nothing: detecting a page labels it and drops nothing. There is no page threshold, no selection threshold and no `RoutingPlan.routed`. `data_pipeline.dataset_builder.policy_windows.pages_to_read` wraps it — `(every page, declarations page)` — so training and serving plan by one rule; with no OCR text (`image_only`) every page is read and no declarations page is detected.
2. **Windowed extraction (policies)** — every policy (every policy's output is canonical) is read as **section group × page window**, planned by `policy_windows.plan_windows` over every page, the same function the corpus build expanded its training rows with. **Every group reads every page** (`common.schema_sections.pages_for`; the page rules in `configs/schema_sections.yaml` are `every` and `declarations`). `decl` (and `dtd` where a line carries it) takes as its first window pages 1–5 (`pages.declarations_leading: 5`) plus the detected declarations page, read together (`leading_pages`); the pages after them follow in windows, as every other group's do. Windows hold `pages_per_extraction_call` pages, with split runs overlapping by `run_overlap_pages` (1 page on the common model), and a window whose pages' own OCR text would not fit its call is split into windows that do (`expand_tasks.fit_windows_to_text`, the rule the corpus build sizes its rows with; `image_only` windows by page count alone). Each window's prompt and decoding schema are its group's slice. More windows per policy than under page selection, so a longer read. See §6.
3. **Other types** — there is no routed single call. A Loss Run is windowed only when its estimated rows overflow one answer; anything else (an ACORD form, a Loss Run that fits) is read in one call over every page, interleaved `[img_1][<page 1 of N> + md_1][img_2][<page 2 of N> + md_2]…` (IMPL-07). One call per page was the earlier design and was wrong twice over: it asked the model to produce a complete document-level JSON from one page — a shape no training row ever had — and it prevented the model from seeing that a table on page 9 continues on page 10. Interleaving makes a policy window a **subsequence of the full document**, so the served shape is the trained shape with fewer pairs.
4. **Merge** — a policy's windows are merged by `serving/policy_merge.merge_policy_windows`, a long Loss Run's by `serving/lossrun_merge.merge_extracted_windows` (§6). `merge_page_extractions` (the declarations-page-wins rule for per-page results) stays in `page_router.py` as code only; it is not on the serving path, and the declarations rule is stated to the model in the Policy prompt and applied by the window merge's group order.

The page numbers in the markers are the document's real numbers, so a window's numbering tells the model it is holding a fragment: fields living on pages it was not shown are absent, not missing. `image_only` carries the markers without markdown — the guarantee is *no OCR text*, not *no text at all*.

- **No document is page-routed** — a policy of any length is read over every page, in windows; there is no per-scope list of routed types (`long_doc_types` is gone).
- **Record which pages fed each extraction** in the output metadata (`pages_used`, master §9), so a low-confidence field can be traced to the specific page it came from. A policy's `pages_used` is every page.

### 5. Confidence post-processing — composed in `serving/pipeline.py`
*(`calibration/apply_calibration.py` + `calibration/list_completeness.py`, called from the pipeline, rather than a `serving/confidence_postprocess.py` wrapper. The steps below are all implemented; a serving-side wrapper around two calibration modules would be a second place for the output contract to be shaped, and the contract has to have exactly one.)*
- Applies IMPL-09 calibration to raw logprob confidence (the release's per-field-type calibrators and thresholds when it has them; otherwise the v1 transform, with every field flagged `:uncalibrated`), adds the **list-completeness** signal, shapes the final output contract (master §9), and flags sub-threshold fields for review.
- Emits `line_of_business`: for a policy, the line resolution `{lines, source, confidence, candidates, hypothesis_agreed}` (`source` ∈ `caller`, `detected`, `detected_uncertain`, `undetected`); otherwise the standard `{value, confidence}` shape (arch §0b) collapsed from the line spans, `null` when undetermined.
- **Validates the final JSON against the doc_type schema before returning** — this mirrors the Fideon SPEC_07 Stage 3 audit gate (arch §0a). A schema-invalid response is an error, not a returned result. On a common-model line the served JSON (every key filled) is what is judged, and a required section no window wrote, or an object written empty (`minProperties`), still fails. The judgement is one function, `serving.pipeline.schema_verdict` (a `SchemaVerdict`: the answer as judged, the filled answer, `valid`, `errors`), shared with evaluation: `served_schema_verdict` envelopes a merged common-model answer and fills its system fields first, then judges it the same way, so checkpoint selection (`evaluation/metrics/coverage_metrics.score_schema_validity`) judges a whole common-model document as serving would return it. Checked as written, the merged answer lacked what the pipeline adds and `document_schema_validity_rate` read a false 0%.
- **Fills every key of the line's schema** (`common.canonical.with_all_keys`, without the client's `fideon:` annotation properties) after validation and confidence: a value not extracted is `null`, not absent, so every served policy JSON — base model or trained — has the same keys. A post-process, never trained.
- Never returns raw (uncalibrated) confidence as if calibrated — IMPL-09 raises when calibration is missing.

### 6. `serving/pipeline.py` (request orchestrator)
Per request: classify (or take the caller's `doc_type`) → route adapter/prompt/schema → resolve the policy's line (caller's; detected only with `detect_lob` on) → route through the serving plan (release, layout family, or base model + `_fallback.json`) → read: policy windows / Loss Run windows / one call → merge → serving guards (common-model lines) → IMPL-09 confidence → list completeness → canonical envelope → schema validation → every-key fill → final JSON.

**This is the canonical pipeline the testing harness (IMPL-12) reuses**, which is what guarantees test == prod. Testing must call it, never re-implement it. The golden eval (IMPL-08) reads through it too, and checkpoint selection's whole-document pass merges its windows with the same `serving/policy_merge.py`.

## Constraints
- Non-GPU orchestration runs **outside** RunPod; the RunPod handler stays scoped to inference (arch §14). Document the boundary.
- Deterministic adapter/prompt/schema selection.
- Calibrated (never raw) confidence in responses.
- Every response schema-validated before return.
- MinerU version parity with the training corpus asserted.
- No PII in logs; no persisted raw request/response bodies.
- Line-of-business detection stays off (`routing.detect_lob: false`) until it has been measured.

## Acceptance checklist
- [ ] Endpoint serves a request end-to-end → schema-shaped JSON + calibrated confidence + `line_of_business`.
- [ ] Where a type has graduated, the correct per-type adapter is hot-swapped from the classification result; otherwise the merged model serves it with no adapter, and that is NOT flagged for review — flagging the expected path would queue every document.
- [ ] ACORD classification returns both `doc_type` and `acord_form`, and the form selects the schema.
- [ ] Low classifier confidence → **Foundation-only fallback + review flag**, never a silent guess.
- [ ] A policy of any length is read in windows over **every page**, and its `pages_used` is every page; a page no section signal matches is read like any other.
- [ ] The declarations group's (and `dtd`'s) first window is pages 1–5 plus the detected declarations page, read together; a declarations page behind a cover sheet joins it; every later page follows in windows.
- [ ] A window's user turn is a **subsequence** of the full document's — the same blocks, fewer pairs. Comparing system prompts alone does not catch a divergence here.
- [ ] `pages_used` reflects the pages actually sent, including for `image_only` documents that carry no per-page OCR text.
- [ ] Nothing in serving's config or a release's scope selects pages: there is no page threshold and no `long_doc_types`.
- [ ] A window that fails (unparseable, or cut at the token limit) is split in half and retried; one that still fails alone is dropped and flagged `window_failed:<group>:p<page>`, and the rest of the document is returned.
- [ ] A served policy JSON carries every key of its line's schema, `null` where nothing was extracted.
- [ ] A policy with no line, detection off, is refused unless the caller sends `allow_lob_fallback`; a known line no promoted release covers is read by the base model against `_fallback.json`, flagged `route:lob_fallback_used`.
- [ ] A schema-invalid generation is rejected, not returned.
- [ ] Checkpoint selection judges a merged common-model answer with serving's verdict (`served_schema_verdict`), not a stricter copy of it.
- [ ] On a common-model line, a form's edition left in its form number in an unambiguous shape is moved to the form's empty `edition_date` (`MM/YY`) and takes the number's span; a filled `edition_date` is never overwritten.
- [ ] Missing calibration for the served version raises rather than returning raw confidence.
- [ ] MinerU version mismatch against the corpus pin fails loudly.
- [ ] No PII in logs; no raw bodies persisted.
- [ ] `serving/pipeline.py` is the single path reused by IMPL-12.

---

## Current implementation (2026-10-21)

**Canonical policies are read as windows over every page** (`serving/pipeline.py::_extract_policy_windows`):
every section group × page window planned by `data_pipeline/dataset_builder/policy_windows.plan_windows`
over every page (`pages_to_read`: every page, and the declarations page `serving.page_router.plan_pages`
finds in the same text the training rows were planned on), whatever the document's length. The routed
single-call branch, the page threshold and `long_doc_types` are gone from `extract`. Each round's windows
go to vLLM in **batches of at most 32** (`MAX_WINDOWS_PER_BATCH`, chosen, not measured; `generate_batch`), so a
200-page policy's page images are not all decoded at once and one refused prompt cannot fail every window; each window's answer limit is its section task's reserved budget
(`common.config.answer_cap(task_for(group))`), its page references are bounded to its own pages
(`common.schemas.with_page_bounds`), and an answer cut at the token limit is refused even if it parses. A
window that fails is **split in half and retried** in the next round (decoding is greedy, so an unchanged
retry would write the same thing); one that still fails at a single page is dropped and flagged
`window_failed:<group>:p<page>`. Only when no window succeeds is the request an error. `window_latencies_ms`
holds one entry per round. This is how **200+ page policies** are served.

**The window merge** (`serving/policy_merge.py::merge_policy_windows`):
- *Self-contained lines* (the frozen non-SPEC_21 schemas): windows merged in group order then page order, so
  `decl` wins a conflict; a later differing value is kept out and flagged `<path>:merge_conflict`; arrays
  concatenate and are de-duplicated on the `configs/schema_sections.yaml` `array_keys`, a duplicate keeping
  the more complete row and the union of page references.
- *Common-model lines* (the seven personal lines, `gl`, and the fallback — no line): each window's ids are
  namespaced; **units** (locations, buildings, vehicles, drivers, watercraft, scheduled items, LOB parts) are
  joined on their own keys (`unit_keys`: VIN, location number, ...) and every reference rewritten to the
  merged unit; a row read twice because two windows share an overlap page is dropped from one of them — never
  a unit's row (`_drop_shared_page_rows`); the other tables are joined on the line's keys with references
  compared through the units they name, and a row whose reference its window could not see joins the one row
  it matches (`_join_unreferenced`); the whole document is then renumbered in printed order
  (`common.structural_ids.renumber_structural_ids`, the numbering training uses). Flags:
  `:joined_without_identifier`, `:ambiguous_unit`, `:unplaced_fragment`, `:dangling_reference`.

**Serving guards on common-model lines** (`serving/conventions.py`, after the merge):
- `conform` applies the label rules (`data_pipeline.ingestion.label_rules.apply_label_rules`) against every
  page's text: each value once, citing every page that prints it; an extra field moved into its own empty
  field. A value no page prints is flagged `<path>:not_printed`; on a born-digital document
  (`ocr_meta.modality` native) an extra field no page prints is dropped, flagged
  `additional_fields:not_printed_dropped`. A scan's OCR text settles nothing, so on a scan every value stays.
- `conform` also moves a form's edition left inside its form number to the form's empty `edition_date`, as the
  labels now do (`label_rules.split_form_edition`): only an unambiguous shape — ISO's `AA 99 99 MM YY`, or a
  month and year in brackets after the number — parsed `MM/YY`, the number kept without it; a filled
  `edition_date` is never overwritten. The moved edition takes the number's span, since it was read in the
  number's tokens.
- `likely_missed` flags `<block>.<field>:likely_missed` where a declarations field was left empty while a line
  of any page opens with its label and a value of its type (`_likely_missed` passes every page: the
  declarations windows read every page).
- `unit_link_flags` flags `coverages[i].applies_to:unit_link_unknown` for a coverage served without its unit
  when the answer holds two or more vehicles, boats or buildings, or another row of its code does carry a link.
- Empty optional objects and empty rows are dropped from the answer (`_without_empty`).

**Long Loss Runs** (`lossrun_windows`, `_extract_lossrun_windows`): windowed only when the estimated rows of
the whole document overflow one answer (`answer_cap("lossrun_rows")`, 8,192 tokens), with the same split-and-
retry rule; merged by `serving/lossrun_merge.merge_extracted_windows` and reconciled against the printed totals.
Flags `claims:merge_conflict`, `claims:totals_mismatch`; the output carries `reconciliation`.

**The request** (`serving/vllm_entrypoint.py::build_request`):
- `lob` (or `line_of_business`): one line, or a list for a package policy, normalised to lower case with classic
  auto read as `personal_auto` (`common.lob.merge_line`). It selects the line's canonical schema; a single line
  with no schema is **refused** unless the caller sends `allow_lob_fallback`.
- `lob_hypothesis`: a line L1/L2 suspects without being sure; reconciled with the line the classifier reads.
- `source_file_name` (or `file_name`): written into the canonical output by the system, never asked of the model.
- `page_texts` — MinerU's per-page markdown, from MinerU 3.4.5's pipeline backend at the corpus pin — must number
  exactly pages `1..n` against the `n` images sent, or the request is refused; a joined `ocr_text` is accepted for
  a single page only. Images are paired **by page number** (`page_N.` by file name, else position); a page beyond
  the images is an error. `ocr_meta` carries the document's modality.

**Line of business** (`serving/pipeline.py::_resolve_lob`): the caller's line wins (flagged
`lob:caller_disagrees` when a confident classifier reads another). **Detection ships off**
(`routing.detect_lob: false`) until `scripts/measure_lob_detection.py` has measured it on held-out policies; off,
a policy with no line is refused, or read against the fallback when the caller sends `allow_lob_fallback`. On: a
line read at or above `lob_confidence_threshold` (0.90) routes as the caller's would; at or above
`lob_family_threshold` (0.60) it routes to that line's family, flagged `lob:uncertain_within_family`; below, the
base model reads the policy against the fallback, flagged `lob:undetected`.

**Routing by layout family** (`serving/release_router.py::ServingPlan.route`, `configs/layout_families.yaml`:
`personal_lines` = homeowners, personal_auto, dwelling_fire, ocean_marine, motorcycle, recreational_vehicle,
personal_umbrella; `casualty_fleet` = gl, auto, wc):
- a rollback pin (`routing.release_pins`) takes every document of its type;
- a policy whose line is missing or in no family goes to an unrestricted release (unified, policy) still serving
  policies; with none, to the base-model fallback when the caller allowed it (or detection found no line);
  otherwise it is refused;
- a routable line goes to the narrowest promoted release whose `lines` cover all of its lines (newest on a tie;
  a package of common-model lines never reaches a line-scoped release);
- a known line no promoted release covers is read by the **base model with no LoRA against `_fallback.json`**,
  flagged `route:lob_fallback_used`, with the release's calibrators dropped (every field goes to review). It is
  refused when the engine holds a release's merged model rather than the base (`base_model_loaded`).
`route` in the output records `layout_family`, `lob_fallback_used` and `lob_source`. The release step records
the same routing as a table on the bundle and fails when a line cannot be routed (`serving/routing_check.py`,
IMPL-13 §4).

**Serving through the chosen release** (`serving/vllm_entrypoint.load_release_runtimes`, `extract(...,
release_runtimes=)`): cold start builds, per served release, its adapter and its fitted calibrators and
thresholds (`releases/.../calibration/{format}/calibrators.json`). With **one** release covering every line the
engine serves its merged model and no adapter is applied; with **several**, or one that leaves a known line to
the base model (`serves_base_with_loras`), the engine loads the base with LoRA enabled and each release's adapter
is downloaded to `staging/serving-adapters/<release_id>/` and applied per request — cold start refuses a model
loaded without LoRA support (`enable_lora: false`), or a release whose adapter is missing. `extract` routes each
request through the release `route` returned. The v1 per-version calibration is needed only for a release
without its own calibrators. An operator **pin** outranks line-scoped routing, so a rollback applies to every
document of the type. Cold start does not require a unified run when releases are promoted. No scope lists
page-routed types any more: `long_doc_types` and `long_doc_types_for` were removed, and every served type
reads every page.

**Parity with training**: the same blank-page placeholder, the same pixel budget (engine
`mm_processor_kwargs`), the same window plan over every page (`pages_to_read`, `plan_windows`), dates in `MM/DD/YYYY` by the same word rule,
the same classifier input, and the calibration page text by the same definition
(`calibration.features.shown_ocr_text`).

**Cold start** (`assert_prompt_hash`): each release records the hash of **every prompt input** — all
templates (`prompts/**/*.jinja`, the per-doc-type ones included), every schema (ours and the client's
canonical files) and the section map. A release packaged with different prompt inputs is **refused**; one
with no recorded hash is warned about.

**Engine**: vLLM `==0.11.0` (the build calibration was fitted with); refuses to start without CUDA; loads the
merged model, or the base from the pod's local copy.

**OCR pin at cold start**: the corpus manifest must still record `mineru_version`, but the version comparison
runs only where MinerU is installed. A serving pod does no OCR (requests carry page texts produced by the pinned
MinerU), and comparing against a MinerU it does not have refused every deployment. A real mismatch is a
`ColdStartError`.

## Changes since 2026-10-05

- Added the alignment line and dated *Current implementation* 2026-10-13.
- §1: base + LoRA per request when several releases are promoted or a release leaves lines to the base model; `serving_thresholds()` also reads `detect_lob`, the two line thresholds and `fallback_doc_type`; `release_pins()`; the model is supplied to `cold_start`.
- §2: the classifier's two-page input; line-of-business reading with token-probability confidence, used only with detection on (ships off); the base-model line classifier.
- §4: replaced "one call for the selected pages, no merge" for policies with routed-page windows (`plan_windows`: decl over 5 leading pages + declarations page, arrays/lineblk over routed pages, 1-page overlap); documented the page signals, the ≤5-page rule and that short documents are still scored; the single interleaved call is now the path for other long types; merges are `policy_merge` / `lossrun_merge`.
- §5/§6: line-resolution output shape; validation of the filled answer on common-model lines; every-key post-process; updated request flow.
- *Current implementation*: window batching, answer caps, page bounds, split-and-retry, `window_failed`; common-model merge (units, joins, overlap duplicates, renumbering) and its flags; serving guards (`not_printed`, extra-field drop on born-digital, `likely_missed`, `unit_link_unknown`, empty-object drop); Loss Run windows; new request keys (`lob_hypothesis`, `allow_lob_fallback`, `source_file_name`, classic auto → personal auto, MinerU 3.4.5 page texts); line resolution with detection off; routing by layout family with the base-model fallback (replacing "refused"); the release routing table.
- Acceptance checklist: replaced the single-call policy item; added window retry, every-key output and fallback routing items.

## Changes since 2026-10-13

- Header and goal: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx` (§0e added); "page routing" replaced by windowed reading of every page.
- §1: no page threshold to read (`DEFAULT_LONG_DOC_PAGE_THRESHOLD` is gone); `long_documents` keeps only `conflict_resolution`, read by no code.
- §4: rewritten for every page read (decision 2026-10-19): `plan_pages` returns every page and only detects the declarations page; `pages_to_read` replaces `routed_pages`; every group reads every page, the declarations group's (and `dtd`'s) first window pages 1–5 plus the declarations page; the routed single-call branch, `SELECTION_THRESHOLD`, `RoutingPlan.routed`, the page cap and `long_doc_types` removed.
- §5: the schema judgement is `schema_verdict`, shared with evaluation through `served_schema_verdict`.
- Acceptance checklist: every-page and declarations-window items replace the routed-page and page-cap items; new items for the served schema verdict and form editions.
- *Current implementation*: windows over every page; `conform` moves a form's edition to `edition_date` with the number's span; `likely_missed` reads every page; `long_doc_types_for` removed; parity names the window plan, not a page threshold.

## Changes of 2026-10-22 (code review)

- Windows split by their pages' own OCR text, by the rule the corpus build uses (`fit_windows_to_text`); `image_only` windows by page count alone.
- A round's windows go to vLLM in batches of at most 32 (`MAX_WINDOWS_PER_BATCH`, chosen, not measured), one latency entry per batch.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.
