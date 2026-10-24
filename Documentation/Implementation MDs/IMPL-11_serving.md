# SPEC 11 — Serving (vLLM Endpoint + Classifier + Routing)

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-07 (inference core), IMPL-09 (calibration), IMPL-10 (postprocessing).
>
> **Naming hazard (master §0.1):** the architecture's "**Fideon SPEC_11**" is the external Presidio de-identification spec (implemented in our IMPL-05). This file is our serving layer. Different documents.
>
> **Architecture refs:** `finetuning-architecture-v2.7.docx` §0a (audit gate contract), §0b (LoB output), §0e (decisions of 2026-10-19: every page read), §4a (**classifier as a load-bearing component**), §4b (ACORD two-level), §4c (day-zero classifier), §5 (confidence), §7 (long documents: every page, read in windows), §14 (**RunPod split: what runs where**).

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
- RunPod Serverless handler wrapping vLLM, serving the promoted release over an OpenAI-compatible API, **with logprobs enabled** (calibration depends on them). A release packaged since 2026-10-23 is its adapter (`served_as: lora_on_base`), served on the bf16 base; nothing is merged or quantized.
- **One LoRA per request, never a reload.** A tenant has one promoted release (decision 2026-10-22). A `lora_on_base` release always runs with the engine holding the **base with LoRA enabled** and the release's adapter applied per request (`serves_base_with_loras` is True for it; `load_release_runtimes`). An older merged release: when it covers every known line, the engine is that release's merged model and no adapter is applied; when it leaves lines to the base model, the engine loads the base with LoRA enabled and the release's adapter is applied per request. A §4.2 graduated per-type adapter would be applied the same way, from the classifier result. One LoRA per request is what vLLM supports, which is exactly why the v1 Foundation-plus-per-type stack could never have been served.
- Pulls the promoted release's adapter and calibrators from Blob on cold start (IMPL-02); version configurable. The model itself is handed to `cold_start` — loading it there is not wired yet, and `cold_start` refuses without one.
- Uses IMPL-07 `model_runner` / `input_builder` for generation so output matches eval and testing exactly.
- **Asserts the serving MinerU version matches the training corpus pin** (IMPL-03 `assert_version_matches`) wherever MinerU is installed — a mismatch is distribution shift and a regression trigger (arch §8a), not a warning to ignore. A serving pod without MinerU does no OCR; it logs the pin its requests' page texts must come from.
- **Every tuneable in `configs/inference/vllm_serving.yaml` must have a reader.** `serving_thresholds()` lifts `routing.classifier_confidence_threshold`, `confidence.review_threshold`, `routing.detect_lob`, `routing.lob_confidence_threshold`, `routing.lob_uncertain_threshold` and `routing.fallback_doc_type` out of the config and passes them into `extract`. (`routing.release_pins` and `release_pins()` are gone: a rollback is `promote --version vK`, IMPL-13. A config still holding `release_pins` or the removed `lob_family_threshold` is refused at cold start, `assert_no_retired_settings`.) (There is no page threshold to read: every page is read, in windows planned by the function the corpus build planned its training rows with, so nothing in serving's config can re-shape documents away from what was trained. The `long_documents` block keeps only `conflict_resolution: declarations_page_wins`, which names the rule the window merge applies by group order; no code reads it.) They were declared in that file and read by no code, so an operator could raise `review_threshold`, redeploy, and change nothing at all — the hardcoded defaults inside `extract` won silently every time. A test asserts each key the helper emits is a parameter `extract` accepts.
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
- **Also reads a policy's line of business** when asked (`Classification.lob`), with the probability of the line's own tokens as its confidence (`line_confidence`) — never a number the model writes about itself. Used only with `routing.detect_lob` on, which ships **off** (§6, *Current implementation*). With detection on and no classifier supplied, the endpoint builds `ZeroShotClassifier` on the engine's base weights with no adapter (`line_classifier`) — only when the engine holds the base; an engine holding one release's merged model reads no line. The line decides whether the release's adapter or the base model reads the policy, so no adapter can have been chosen yet.
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

- **No document is page-routed** — a policy of any length is read over every page, in windows; there is no list of routed types (`long_doc_types` is gone).
- **Record which pages fed each extraction** in the output metadata (`pages_used`, master §9), so a low-confidence field can be traced to the specific page it came from. A policy's `pages_used` is every page.

### 5. Confidence post-processing — composed in `serving/pipeline.py`
*(`calibration/apply_calibration.py` + `calibration/list_completeness.py`, called from the pipeline, rather than a `serving/confidence_postprocess.py` wrapper. The steps below are all implemented; a serving-side wrapper around two calibration modules would be a second place for the output contract to be shaped, and the contract has to have exactly one.)*
- Applies IMPL-09 calibration to raw logprob confidence (the release's per-field-type calibrators and thresholds when it has them; otherwise the v1 transform, with every field flagged `:uncalibrated`), adds the **list-completeness** signal, shapes the final output contract (master §9), and flags sub-threshold fields for review.
- Emits `line_of_business`: for a policy, the line resolution `{lines, source, confidence, candidates, hypothesis_agreed}` (`source` ∈ `caller`, `detected`, `detected_uncertain`, `undetected`); otherwise the standard `{value, confidence}` shape (arch §0b) collapsed from the line spans, `null` when undetermined.
- **Validates the final JSON against the doc_type schema before returning** — this mirrors the Fideon SPEC_07 Stage 3 audit gate (arch §0a). A schema-invalid response is an error, not a returned result. On a common-model line the served JSON (every key filled) is what is judged, and a required section no window wrote, or an object written empty (`minProperties`), still fails. The judgement is one function, `serving.pipeline.schema_verdict` (a `SchemaVerdict`: the answer as judged, the filled answer, `valid`, `errors`), shared with evaluation: `served_schema_verdict` envelopes a merged common-model answer and fills its system fields first, then judges it the same way, so checkpoint selection (`evaluation/metrics/coverage_metrics.score_schema_validity`) judges a whole common-model document as serving would return it. Checked as written, the merged answer lacked what the pipeline adds and `document_schema_validity_rate` read a false 0%.
- **Attaches the full-text tier** on a common-model policy (`common.full_text.build_text_sections`): `text_sections` — every printed word, by form and section — built from the request's page text by Fideon's deterministic section builder (`common/section_builder`, FideonSLM's `sections.py` and `pdf_text.py` copied byte for byte, recorded in `SOURCE.json`; `scripts/check_section_builder.py` reports drift), never by the model (SPEC_21 §4.6). Before validation, so the client's schema judges it. A request with no page text (image-only) gets none and is flagged `text_sections:no_page_text`. Each result carries `text_coverage`, the share of the page text's words found in the served JSON (SPEC_21's every-word rule, check C4; `None` with no page text). Our wrapper anchors each MinerU table so the builder writes it: the copied builder drops a markdown table otherwise (to be fixed in FideonSLM).
- **Fills every key of the line's schema** (`common.canonical.with_all_keys`, without the client's `fideon:` annotation properties) after validation and confidence: a value not extracted is `null`, not absent, so every served policy JSON — base model or trained — has the same keys. A post-process, never trained.
- Never returns raw (uncalibrated) confidence as if calibrated — IMPL-09 raises when calibration is missing.

### 6. `serving/pipeline.py` (request orchestrator)
Per request: classify (or take the caller's `doc_type`) → route adapter/prompt/schema → resolve the policy's line (caller's; detected only with `detect_lob` on) → route through the serving plan (the promoted release, or base model + `_fallback.json`) → read: policy windows / Loss Run windows / one call → merge → serving guards (common-model lines) → printed values only (every policy) → IMPL-09 confidence → list completeness → canonical envelope → schema validation → every-key fill → final JSON.

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
- [ ] Where a type has graduated, the correct per-type adapter is hot-swapped from the classification result; otherwise the release serves it — its adapter on the base, or an older release's merged model with no adapter — and that is NOT flagged for review — flagging the expected path would queue every document.
- [ ] ACORD classification returns both `doc_type` and `acord_form`, and the form selects the schema.
- [ ] Low classifier confidence → **Foundation-only fallback + review flag**, never a silent guess.
- [ ] A policy of any length is read in windows over **every page**, and its `pages_used` is every page; a page no section signal matches is read like any other.
- [ ] The declarations group's (and `dtd`'s) first window is pages 1–5 plus the detected declarations page, read together; a declarations page behind a cover sheet joins it; every later page follows in windows.
- [ ] A window's user turn is a **subsequence** of the full document's — the same blocks, fewer pairs. Comparing system prompts alone does not catch a divergence here.
- [ ] `pages_used` reflects the pages actually sent, including for `image_only` documents that carry no per-page OCR text.
- [ ] Nothing in serving's config or a release selects pages: there is no page threshold and no `long_doc_types`.
- [ ] A window that fails (unparseable, or cut at the token limit) is split in half and retried; one that still fails alone is dropped and flagged `window_failed:<group>:p<page>`, and the rest of the document is returned.
- [ ] A served policy JSON carries every key of its line's schema, `null` where nothing was extracted.
- [ ] In strict mode a value no page prints is taken out of a served policy JSON (its key null; an extra field, or a row with nothing printed, leaves its list, the rows after it renumbered) and flagged `<path>:not_printed_removed`; flag mode keeps and flags it `<path>:not_printed`; an unknown mode is refused; with no page text nothing is checked and the result is flagged `printed_values:no_page_text`.
- [ ] A served common-model policy with page text carries `text_sections` from the section builder, valid against the client's schema, with every word of its page text somewhere in the JSON (`text_coverage` 1.0 on the test fixtures); one with none is flagged `text_sections:no_page_text`. The copied builder matches `SOURCE.json`.
- [ ] A policy with no line, detection off, is refused unless the caller sends `allow_lob_fallback`; a known line the promoted release did not train on, and a package even of lines it trained on, is read by the base model against `_fallback.json`, flagged `route:lob_fallback_used`.
- [ ] One promoted release serves a tenant; if the index holds several, the newest that can be served serves, with a warning naming it. Cold start reads the release's `window_plan_version` (`ServedRelease`), and refuses a config still holding `routing.release_pins` or `routing.lob_family_threshold`. It also refuses a `lora_on_base` release whose adapter trained on another base than the configured `model_id@revision`; an older merged release is not checked.
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
go to vLLM in **calls of at most 96** (`MAX_WINDOWS_PER_BATCH`, the validation's `validation_batch_rows`; not yet measured in serving; `generate_batch`), the longest expected answers first (`_send_order`: the larger answer reservation, then more pages; `ExtractionRequest.send_order="reversed"` sends them the other way, as the golden eval's repeat pass does) and read back in planned order, so a
69-page policy's 81 windows are one call and a 200-page policy's page images are not all decoded at once. vLLM fails every prompt of a call when it refuses one, so a call whose windows all failed with one error is read again in halves (`_batch_in_halves`) until the window that caused it is alone, and only that window fails. Each window's answer limit is its section task's reserved budget
(`common.config.answer_cap(task_for(group))`), its page references are bounded to its own pages
(`common.schemas.with_page_bounds`), and an answer cut at the token limit is refused even if it parses. A
window that fails is **split in half and retried** in the next round (decoding is greedy, so an unchanged
retry would write the same thing); one that still fails at a single page is dropped and flagged
`window_failed:<group>:p<page>`. Only when no window succeeds is the request an error. `window_latencies_ms`
holds one entry per batch (and per window read alone after a whole-batch failure). This is how **200+ page
policies** are served. At cold start `assert_window_plan_served` refuses a release serving policies whose
`window_plan_version` is not this code's `window_plan_id()` — one that records none was packaged before every
page was read.

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
  `additional_fields:not_printed_dropped`. A scan's OCR text settles nothing for the label rules, so here a
  scan's values all stay; the printed-values check below judges scans too, and in strict mode its
  `not_printed_removed` replaces `not_printed`.
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

**Printed values only** (`serving/printed_values.py`, decision 2026-10-24). After the merge and `conform`,
every canonical policy's values are looked up on the document's pages with
`common.grounding.ground_tolerant` (IMPL-08 §1), on scans too: each envelope with a non-empty `raw`, extra
fields and table rows included; system fields are skipped.
- Printed on some page: kept. Unchecked (too short, a yes/no reading, a cited page with no text): kept and
  counted.
- Printed nowhere, `strict` mode (the default; `printed_values.mode` in `configs/inference/vllm_serving.yaml`,
  read by `configured_mode`, which refuses an unknown mode): taken out of the JSON. Its key stays, null on a
  common-model line; on a self-contained line it is left out, and the every-key fill serves it as null
  (`hold_to_printed(..., lob=)`). An extra field leaves its list. A table row goes when nothing in it is
  found printed (a value too short to check keeps it); the rows after it move up, and the structural ids are
  numbered again as the merge numbers them. Spans go, or move, with their values. Each value is flagged
  `<path>:not_printed_removed`, on the path it had in the model's answer; that flag replaces `conform`'s
  `<path>:not_printed`, and other flags follow their values or go with them.
- Printed nowhere, `flag` mode: kept and flagged `<path>:not_printed`.

The record is `ExtractionResult.printed_check`: `{mode, checked, unchecked, not_printed: [{path, raw, pages}],
removed}`, `not_printed` listed before removal. It is not in `as_dict()`; the golden eval reads it (IMPL-08).
With no page text (image only) nothing is checked: `printed_check` is None and the result is flagged
`printed_values:no_page_text`. Fields first serves its fields only once held to the pages they cite
(`serving.fields_first.verify_fields`: `hold_result_to_printed` on a copy, over those pages' text; a value
not found yet is held back, `pending_full_text`), and checks the whole result once MinerU's full text arrives
(`attach_full_text` → `hold_result_to_printed`): for every policy, before the full-text tier, only when no
check has run yet; `fields`, `list_fields`, `overall_confidence`, the flags and the schema verdict follow
the JSON. The trade-off the user accepted: a value MinerU misread beyond what the search forgives is removed
too, and comes back null with the flag.

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
`lob_uncertain_threshold` (0.60, was `lob_family_threshold`) it is read with that line's schema, flagged
`lob:uncertain` (was `lob:uncertain_within_family`) for a person to check; below, the base model reads the
policy against the fallback, flagged `lob:undetected`. The LOB detection report no longer has
`family_accuracy`.

**One promoted release per tenant** (`serving/release_router.py`, `build_serving_plan`, `ServingPlan.route`;
decision 2026-10-22). The release marked `promoted` in the tenant's release index serves; should the index
hold several, the newest that can be served serves and the warning names it. There are no layout families
(`configs/layout_families.yaml` is gone; Fideon SPEC_06 §9a / SPEC_09's family adapters are knowingly not
followed). The release records the lines it trained on (`lines`: the single lines of the run's train view,
else the corpus's `policy_line_counts`; never none for a new release, since packaging refuses one, IMPL-13):
- a policy of ONE line the release covers goes to it (`ServedRelease.covers_lob`, `common.lob.single_line`);
- any other known line, and a package even of lines the release covers (it is read against `_fallback.json`,
  a shape the adapter is not taught) — every line, while nothing is promoted — is read by the **base model
  with no LoRA against `_fallback.json`**, flagged `route:lob_fallback_used`, with the release's calibrators
  dropped (every field goes to review). Before the code review of 2026-10-23 a package of self-contained
  lines went to the release. It is refused when the engine holds an older release's merged model rather than
  the base (`base_model_loaded`; a `lora_on_base` engine always holds the base); the error says a policy with no line, a package or an undetected line cannot be
  read there, and to send the policy's single line;
- a policy with no line, or a line with no schema, is refused unless the caller sends `allow_lob_fallback` (or
  detection found no line); an older promoted release that records no lines reads it instead;
- a document of a type the release does not cover is refused (`UnservedDocType`).
`route` in the output records `lob_fallback_used` and `lob_source` (no `layout_family`). The release step
records the same routing as a table on the bundle and fails when a line cannot be routed, or a line the
release trained on would not reach it; it routes every known line, the canonical policy schemas
(`serving/routing_check.py`, IMPL-13 §4).

**Serving through the release** (`serving/vllm_entrypoint.load_release_runtimes`, `extract(...,
release_runtimes=)`): cold start builds the release's adapter and its fitted calibrators and
thresholds (`releases/.../calibration/{format}/calibrators.json`). A `lora_on_base` release (every release
since 2026-10-23) always has the engine load the base with LoRA enabled (`serves_base_with_loras` is True for
it): its adapter (`ServedRelease.adapter_dir`, the bundle's `serving_formats.bf16`) is downloaded to
`staging/serving-adapters/<release_id>/` and applied per request. For an older merged release, when every
known line reaches it the engine serves its merged model and no adapter is applied; when it leaves a known
line to the base model, the engine loads the base with LoRA enabled and applies the release's adapter the same
way — cold start refuses a model loaded without LoRA support (`enable_lora: false`), or a release whose
adapter is missing. `extract` routes each request through the release `route` returned. The v1 per-version
calibration is needed only for a release without its own calibrators. A rollback is a promotion
(`promote --version vK`, IMPL-13), not a pin. Cold start does not require a unified run when a release is
promoted. **Fixed (2026-10-22):** `ServedRelease` now carries `window_plan_version`, which
`assert_window_plan_served` reads at cold start; without it, any promoted policy release raised
`AttributeError`. No release lists page-routed types any more: `long_doc_types` and `long_doc_types_for` were
removed, and every served type reads every page.

**Parity with training**: the same blank-page placeholder, the same pixel budget (engine
`mm_processor_kwargs`), the same window plan over every page (`pages_to_read`, `plan_windows`), dates in `MM/DD/YYYY` by the same word rule,
the same classifier input, and the calibration page text by the same definition
(`calibration.features.shown_ocr_text`).

**Cold start** (`assert_prompt_hash`): each release records the hash of **every prompt input** — all
templates (`prompts/**/*.jinja`, the per-doc-type ones included), every schema (ours and the client's
canonical files) and the section map. A release packaged with different prompt inputs is **refused**; one
with no recorded hash is warned about. A release served as a LoRA on the base (`lora_on_base`) is also
refused when its adapter trained on another base than the configured `model_id@revision`
(`configs/base_model.yaml`; `assert_base_model_served`, through `query_registry.base_model_refusal`): a
moved revision pin changes the weights under the adapter, which would still load and answer with no error.
An older merged release carries its own base and is not checked. `promote` runs the same prompt-hash,
window-plan and base-model checks (`registry_utils.releases.serving_refusal`), so a release this endpoint
would refuse is never promoted (IMPL-13). Cold start also refuses a serving config still holding a removed setting, `routing.release_pins`
or `routing.lob_family_threshold` (`vllm_entrypoint.assert_no_retired_settings`): read by nothing, it would be
ignored without a word.

**Engine**: vLLM `==0.11.0` (the build calibration was fitted with); refuses to start without CUDA; loads the
base from the pod's local copy (for every `lora_on_base` release), or an older release's merged model.

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
- A batch whose windows all failed with one error is read again window by window; latency is one entry per batch.
- Cold start refuses a policy release whose recorded window plan is not this code's (`assert_window_plan_served`).

## Changes of 2026-10-22 (one adapter)

- One promoted release per tenant (decision 2026-10-22): no layout families and no `release_pins` (a rollback is `promote`); a policy of a line the release trained on goes to it, any other known line to the base model against `_fallback.json`; `route` has no `layout_family`; the routing check routes every known line; `lob_uncertain_threshold` and `lob:uncertain` replace the family names, and the LOB detection report drops `family_accuracy`; fixed: `ServedRelease` carries `window_plan_version`, which the cold-start check reads. New acceptance item.
- Code review of 2026-10-23: `covers_lob` covers a policy of ONE line the release trained on, so a package of trained lines now goes to the base model against `_fallback.json`, flagged `lob_fallback_used`; cold start refuses a config still holding `routing.release_pins` or `routing.lob_family_threshold` (`assert_no_retired_settings`); `promote` runs the cold-start prompt-hash and window-plan checks; the several-promoted warning names the release served; the merged-engine error no longer suggests a remedy that does not exist. Acceptance items updated.

## Changes of 2026-10-23 (window calls, forms job, full text)

- A round's windows go to vLLM in calls of at most 96 (`MAX_WINDOWS_PER_BATCH`, was 32): at 32 a call, about one call in four of the CGL checkpoint selection waited ~140 s on one long window with the GPU nearly idle. A 69-page policy (81 windows) is one call. Not yet measured in serving.
- Sent longest expected answer first (`_send_order`), read back in planned order: the windows, the prompts and what the merge is given are unchanged.
- A call refused whole is read again in halves (`_batch_in_halves`), not window by window; latency is one entry per call.
- The forms list is read by a `forms` group of its own on the common-model lines (configs/schema_sections.yaml): the same windows as before on the personal lines, one more group (the claims-made terms and countersignature) on general liability; merged after `arrays`.
- The full-text tier is served (decision 2026-10-23): `text_sections` from the page text by Fideon's section builder (`common/full_text.py` over `common/section_builder`, FideonSLM 3d40d77 copied byte for byte), before validation; `text_sections:no_page_text` without page text; `text_coverage` on every common-model result. The copied builder dropped every MinerU table; the wrapper anchors them.

## Changes of 2026-10-23 (served model speed)

- A backend that answers each request on its own (`ModelBackend.supports_submit`; the served model, `inference_core/openai_backend.py`) has a policy's windows read as they are answered (`_read_as_answered`): every window sent at once, longest first, and a failed window's halves sent the moment it fails rather than after the slowest window of its round. The answers are merged in the rounds' order (by round, then planned position), so the merge is the same whichever window finished first; the in-process engine still reads in rounds of calls (`_read_in_rounds`). One `window_latencies_ms` entry for the whole reading.
- A streamed answer whose last 3,000 characters are one passage of at most 500 repeated (`repeating_tail`) is stopped and its connection closed, finish reason `repetition` (`STOPPED_REPEATING`), which counts as cut off: the window is split as at the token limit, instead of writing to 12,288 tokens.
- Every window read is recorded in `ExtractionResult.window_stats` (group, pages, attempt, outcome, seconds, time to first token, prompt and answer tokens, finish reason), not in the served JSON; `Generation` carries `prompt_tokens` and `first_token_ms`.
- Page images can go to a served model by path (`images="file"`, a `file://` URL; vLLM needs `--allowed-local-media-path`) instead of as base64.

## Changes of 2026-10-23 (page bytes)

- A served model is sent a gray page (an RGB render whose channels are equal everywhere) as an 8-bit grayscale PNG (`inference_core.page_links.sent_png`, `OpenAIChatBackend(gray_pages=True)`, the default): vLLM converts every image to RGB, so the model reads the same pixels, at 66% of the bytes (all 114 pages of a scan checked); a page with any colour, or one PIL cannot read, goes as it is on disk. Each page is encoded once however many windows send it.
- `images="link"` (`inference_core.page_links.PageLinks`): each page is uploaded once to Blob (`transient/page-links/<run>/<document>/`) and every window's request carries a read-only link the server fetches itself, instead of the image about four times a page; a page with no link goes inline. The link's token is a read-only SAS given (`PAGE_LINK_SAS_TOKEN`, refused unless `sp=r`) or one signed for the blob alone from the account key (`BlobClient.read_link`, IMPL-02). The pages are deleted after the document; the links expire (1 hour when signed here).

## Changes of 2026-10-23 (incremental training)

- Nothing is merged (decision 2026-10-23): a release is its adapter (`served_as: lora_on_base`), so the engine always holds the bf16 base and applies the release's adapter per request (`serves_base_with_loras` is True for it); an older merged release is served as before (§1, acceptance, *Current implementation*: serving through the release, the engine).
- Code review of the incremental training: cold start refuses a `lora_on_base` release whose adapter trained on another base than the configured one (`assert_base_model_served`), and `promote` runs the same check (*Current implementation*, cold start; acceptance).

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.

## Changes of 2026-10-24 (production on Azure)

- The routine runs on Azure and RunPod serves only a stock vLLM and the MinerU 4 API (no code of ours there). Every served document's text comes from the pinned MinerU (`configs/shared/ocr.yaml`, IMPL-03): `ServedOcr(expected_version=)` refuses a task another version or tier read.
- `OpenAIChatBackend(model=name)` checks at start that the server lists `name` and, for a LoRA, that its `parent` is our base model (`configs/base_model.yaml`); a server that cannot list its models is not checked. `scripts/served_adapter.py` makes the promoted release's adapter ready for the pod's network volume: it refuses a release not served as `lora_on_base`, one whose OCR pin is not the served MinerU's, and an adapter vLLM cannot serve (LoRA on the vision side, `modules_to_save`, a rank above 512), downloads it as `policy-<version>`, and prints the copy command and the vLLM flags (`--enable-lora --max-lora-rank --max-loras 1 --lora-modules`).
- Model and MinerU URLs must be HTTPS, or HTTP to this machine only (`common.transport.assert_encrypted`); page links expire after an hour. Outside the code: the Azure region near the RunPod data centre, RunPod Secure Cloud under a data processing agreement, vLLM's request logging off.

## Changes of 2026-10-24 (fields first)

- One H100 serves both the model and MinerU, and the gate is a 200-page scan in 2 minutes (decision 2026-10-24); MinerU alone takes 4-5 minutes for it. `serving/fields_first.py` reads a document in two phases: `read_fields` renders the pages and the model reads their images (`image_only`) - the fields, served flagged `text_sections:no_page_text` - and `add_full_text` has MinerU read the PDF afterwards and adds the full-text tier (`serving.pipeline.attach_full_text`: Fideon's section builder over the page text, the result checked against the client's schema with it, `text_coverage` measured; a tier that fails the schema is left out, flagged `text_sections:invalid`). Only a result flagged `no_page_text` gets one: `extract` flags exactly the documents that have the tier and were read without text.
- `GpuTurns` orders the one GPU: MinerU starts a run of pages only while no document's fields are being read (`ocr_via_api(gate=)`: the runs then read one at a time, each to its end), so a document arriving while another's text is read waits at most one run.
- Whether image-only fields are accurate enough is measured before relying on them (IMPL-12, `latency_check --runs standard,fields-first`); the training changes that bring a 200-page scan under 2 minutes (page images first with the same windows for every job, a compact prompt, smaller images, a compact answer) wait for the next cycle.
- Code review of 2026-10-24: MinerU's turns are runs of at most 20 pages whatever the PDF's length (`ServedOcr.run_pages`, `split_pdf(run_pages=)`), so a document arriving mid full text waits about 25 s, not a whole document's OCR; a run waits for its turn at most `text_max_wait_s` (600 s) and then goes regardless (`GpuTurns.overdue_runs`), so fields arriving back to back cannot hold full texts back for ever. The turn is held from the start of a document's fields, rendering included: a run started while the pages render would still be reading when the model starts. `attach_full_text` judges the tier by the schema errors it adds, so fields that already break the schema keep their full text.
- Printed values only, at the gate too (requirement of 2026-10-24: no value the PDF does not print is served): before fields first serves the fields, MinerU reads only the pages they cite (`cited_pages`, `ServedOcr.ocr_text(pages=)`, `select_pages`; a few dozen of a 200-page policy, ~15-40 s, within the document's own GPU turn) and `verify_fields` holds a copy of the result to that text (`serving.printed_values.hold_result_to_printed`): a value not found there is held back as `<path>:pending_full_text`, not removed as printed nowhere - the model may have cited the wrong page - and `likely_missed` flags declarations fields left empty that those pages print. The model's own answer is kept, and when the remaining pages are read (the cited ones are not read again) it is held to the whole document: a held-back value printed elsewhere comes back, one printed nowhere is removed (`not_printed_removed`), as on the standard path. `attach_full_text` looks an image-only read over for missed fields too. Should the cited pages fail to be read, nothing is served before the full text; `verify=False` (`--unverified-fields`) serves unchecked fields flagged `fields:unverified`, for timing tests only.

## Changes of 2026-10-24 (printed values only and consistency)

- Printed values only (the user's decision of 2026-10-24: nothing invented in the served JSON): `serving/printed_values.py` holds every canonical policy to its pages after the merge and `conform`, with the OCR-tolerant `common.grounding.ground_tolerant`, on scans too. `printed_values.mode` in `configs/inference/vllm_serving.yaml`: `strict` (the default) takes a value printed nowhere out and flags it `<path>:not_printed_removed`, replacing `conform`'s `<path>:not_printed`; `flag` keeps it and flags it `<path>:not_printed`; any other value is refused. The record is `ExtractionResult.printed_check`; with no page text nothing is checked, flagged `printed_values:no_page_text`. Fields first runs the check in `attach_full_text` (`hold_result_to_printed`) once MinerU's text arrives (*Current implementation*, §6, acceptance).
- `ExtractionRequest.send_order` (`planned`, the default, or `reversed`; anything else is refused): the order a policy's windows go to the model, on both reading paths. The merge keeps planned order. The golden eval's repeat pass uses `reversed` to measure `run_to_run_agreement` (IMPL-08).

## Changes of 2026-10-24 (QA fixes)

- **Merge:** within its group, the window holding the declarations page (`PolicyWindow.leads`) merges first, so a declarations window split by its text or retried in halves still gives the declarations value (the half holding the declarations page inherits the lead; else the first half). A section in the wrong JSON type fails its window. Page references are held to the window's pages: one outside them is dropped and the value flagged `<path>:page_ref_outside_window`; "12" is page 12. `extract` refuses a request with no images or with `page_texts` not numbered 1..n; with no serving plan, a line with no schema is refused unless `allow_lob_fallback` is sent (then the base model against `_fallback.json`, flagged `route:lob_fallback_used`).
- **Printed values only, the rule:** a value is looked up on the cited pages that have text, then on every page with text. Found anywhere: printed. Not found, and every cited page has text (or it cites none): removed in strict mode. Not found while a cited page has no text (an image MinerU could not read): kept and flagged `<path>:unverified_unread_page`, counted as unchecked (record key `unread_page`). Unchecked only when no page has text. Codes match as one run of words, never pieces found apart; figures match whole, and a money amount only an amount-like printed figure (currency sign, separator or cents) - never a year or a code; two-character values match as whole tokens; yes/no readings and single characters stay unchecked; a `parsed` value behind a blank `raw` is looked for in its printed forms.
- **Removed rows:** ids are never renumbered after a removal (gaps are possible); a link to a removed row is dropped and flagged `link_to_removed_row`. A value removed with its row is named by its list and its position in the model's answer, `list[#N]`, so no flag names a served path that holds another value; `list_fields` follow the served JSON. A served row's id and links are the same at `fields_ready`, at `done` and on the standard path.
- **Fields first:** nothing is served when no cited page has text (`fields_verified` false, no `fields_ready`).
- **Transport:** redirects are refused (`common.transport.open_url`), so a token never follows one; a stream that ends with no finish reason and no `[DONE]` is `stream_cut`, cut off, and its window split; a stream broken mid-chunk is a `ModelRunnerError`. A page link's SAS must be a container or blob SAS (`sr=c`/`sr=b`, `sp=r`): an account SAS is refused. A transparent PNG is sent as it is on disk.
- A window's truncation message names its finish reason: the token limit, an answer stopped for repeating itself (`repetition`), or a stream that broke off (`stream_cut`).

## Changes of 2026-10-24 (architecture v2.7)

- Header: aligned with `finetuning-architecture-v2.7.docx`, which records the decisions of 2026-10-23 and 2026-10-24: printed values only, consistent output, fields first, one MinerU, production on Azure.
