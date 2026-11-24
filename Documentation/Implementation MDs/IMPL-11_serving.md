# IMPL-11 — Serving (vLLM Endpoint + Classifier + Routing)

> Current specification, aligned with the code of 2026-11-24. Dated history: `changelog/IMPL-11_serving.changelog.md`.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-03 (OCR), IMPL-07 (inference core), IMPL-09 (calibration), IMPL-10 (postprocessing). The production caller is IMPL-16 (`azure_service/`).
>
> **Naming hazard (master §0.1):** the architecture's "**Fideon SPEC_11**" is the external Presidio de-identification spec (implemented in our IMPL-05). This file is our serving layer. Different documents.
>
> **Architecture:** `Documentation/Architecture/finetuning-architecture-v2.8.docx`.

## Goal

Read one document with the promoted release and return canonical JSON with calibrated confidence and review flags. Every page is read, in windows. A value no page prints is not served (strict mode). One function, `serving.pipeline.extract`, does it for production, the golden eval (IMPL-08), checkpoint selection and the testing harness (IMPL-12), so **serving == eval == testing**.

## The infrastructure boundary

```
Azure Container Apps (our code, CPU)            RunPod (no code of ours, one GPU)
─────────────────────────────────────           ─────────────────────────────────
API: PDF + lob in, job id out                   stock vLLM: bf16 base + the release's
worker: render pages, plan windows,  ──HTTPS──>   LoRA, asked for as policy-<release>
  call the model, merge, hold values            MinerU 4 API (the pinned version)
  to the printed text, calibrate,    ──HTTPS──>
  validate, write the result to Blob
```

- RunPod runs only GPU work: a stock vLLM and the MinerU API. Prompt building, merging, the printed-values check, calibration and validation run in the Azure worker (IMPL-16), on CPU.
- Model and MinerU URLs must be HTTPS, or HTTP to this machine only (`common.transport.assert_encrypted`). Redirects are refused (`common.transport.open_url`).
- `serving/vllm_entrypoint.py` (a RunPod Serverless handler around `extract`) is not the production entry point. Its cold-start checks are reused by the Azure worker and by `promote`.

## Deliverables

### 1. `serving/vllm_entrypoint.py` — cold-start checks, request validation, handler
- **A release is served as a LoRA on the bf16 base** (`served_as: lora_on_base`, every release since 2026-10-23); nothing is merged or quantized. `serves_base_with_loras` is True for such a release; `load_release_runtimes` downloads its adapter (`ServedRelease.adapter_dir`) to `staging/serving-adapters/<release_id>/` and applies it per request. An older `merged` release is still served as its merged model, or as a LoRA on the base when it leaves a known line to the base model. Cold start refuses an engine loaded without LoRA support when it must serve the base.
- **Cold-start refusals** (`cold_start`, each a `ColdStartError`):
  - a release packaged with other prompt inputs (`assert_prompt_hash`: templates, schemas, section map; a release with no hash is warned about);
  - a policy release cut with another window plan (`assert_window_plan_served` against `policy_windows.window_plan_id`, which also names the format profile, so a release of the other profile is refused);
  - a `lora_on_base` release whose adapter trained on another base than `configs/base_model.yaml` (`assert_base_model_served`);
  - a serving config still holding `routing.release_pins` or `routing.lob_family_threshold` (`assert_no_retired_settings`);
  - when a corpus version is given: a missing corpus manifest, one with no `mineru_version`, or a MinerU on this machine that differs from it (`assert_ocr_pin`); with no MinerU installed it only logs the pin;
  - no model given: `cold_start` does not load one itself.
- `promote` runs the same prompt-hash, window-plan and base-model checks (`registry_utils.releases.serving_refusal`), so a release serving would refuse is never promoted (IMPL-13).
- **Every tuneable has a reader.** `serving_thresholds()` passes `routing.classifier_confidence_threshold`, `confidence.review_threshold`, `routing.detect_lob`, `routing.lob_confidence_threshold`, `routing.lob_uncertain_threshold` and `routing.fallback_doc_type` (unset in the shipped config) into `extract`. `long_documents.conflict_resolution` is read by no code; it names the rule the window merge applies by group order.
- **The request** (`build_request`): `source_id` and one image per page are required. `page_texts` (prompt text, `ocr_plus_image`) and `printed_texts` (the same MinerU text held back from an `image_only` prompt and used only to check the answer) must each number exactly pages 1..n. A request with no MinerU text at all is refused (decision 2026-10-30: extraction never skips MinerU). `lob` (or `line_of_business`) is one line or a list; it is lower-cased, classic auto is read as `personal_auto` (`common.lob.merge_line`), and a single line with no schema is refused unless `allow_lob_fallback` is sent (a business-owners name is told to send `boppr` or `bopgl`). `lob_hypothesis` is a line hint; `source_file_name` (or `file_name`) is written into the output by the system.
- **Calibration is chosen after classification.** The handler passes the whole `{doc_type: params}` map to `extract`. `assert_calibration_present` runs only when no release is served or a served release has no calibrators of its own; an empty map is no calibration.
- `build_adapter_map` maps a doc type to a per-type adapter only when its directory holds `adapter_config.json`.
- **No document content in logs.** `safe_log_payload` logs shapes, never values, and runs inside the handler's `try`. Errors come back as `{"error", "retryable"}`; the output is `{"output": ExtractionResult.as_dict()}`. Nothing is persisted.

### 2. `serving/doc_type_classifier.py`
- Two levels: doc type in `acord`, `policy`, `lossrun` (`common.constants.ACTIVE_DOC_TYPES`) and the ACORD form. A wrong type means the wrong prompt and schema, so the classifier is measured on the golden set (IMPL-08).
- Reads the first two pages (`CLASSIFIER_PAGES = 2`): their images by page number and their text joined, cut at 4,000 characters (`classifier_input`, `classifier_messages`). Training rows are built by the same functions.
- `ZeroShotClassifier` prompts the loaded model with `doc_type_classifier_prompt.jinja`. It can also read a policy's line (`Classification.lob`); the line's confidence is the probability of its own tokens (`line_confidence`), never a number the model writes.
- A request that names its `doc_type` skips classification (`StaticClassifier`). The Azure worker always sends `policy` with the caller's line, so production runs no classifier.
- The endpoint holds a classifier only when one is given to `cold_start`, or when `routing.detect_lob` is on: then `line_classifier` reads lines with the base weights and no adapter, and only on an engine that holds the base.

### 3. `serving/adapter_router.py` — type to adapter, prompt and schema
- `route` picks adapter, schema type and ACORD form from one classification. No adapter mapped is the normal case and is not flagged.
- Below `classifier_confidence_threshold` (0.70) the document is still read as the classified type, with that type's mapped adapter if any, and flagged `routing:low_confidence`. `route` also lists the top two types as `candidate_routes`; no serving code reads them.
- An unclassifiable document, or an ACORD form with no form number, is refused unless `routing.fallback_doc_type` is set; with it, it is read against that type with no adapter, flagged `routing:unclassified` or `routing:acord_form_unknown`.

### 4. `serving/release_router.py` — one promoted release per tenant
- `build_serving_plan` serves the release marked `promoted` in the tenant's release index. If several are promoted, the newest one that can be served is used and a warning names it. A rollback is `promote --version vK` (IMPL-13). There are no layout families.
- `ServingPlan.route` for a policy: a single line the release trained on goes to it (`ServedRelease.covers_lob`). Any other known line, a package of lines, or any line while nothing is promoted is read by the **base model with no LoRA against `_fallback.json`**, flagged `route:lob_fallback_used`, with the release's calibrators dropped. No line, or a line with no schema, is refused unless the caller sent `allow_lob_fallback` (or detection found no line); an older release recording no lines reads it instead.
- `extract` refuses the fallback when the engine holds a merged model, not the base (`base_model_loaded`).
- A document type the release does not cover is refused (`UnservedDocType`).
- The release step routes every known line the same way and records the table on the bundle (`serving/routing_check.py`, IMPL-13).

### 5. Reading every page — `serving/page_router.py` and the window plan
- **No document is page-routed** (decision 2026-10-19). `page_router.plan_pages` returns every page and the first page whose text matches a declarations signal; the other `PAGE_SIGNALS` scores are kept for tracing and decide nothing. `policy_windows.pages_to_read` wraps it for training and serving alike; with no text (`image_only`) no declarations page is found.
- **A policy is read in windows grouped by section**: section group × page window, planned by `data_pipeline.dataset_builder.policy_windows.plan_windows`, the function the corpus build plans training rows with. Every group reads every page (`common.schema_sections.pages_for`). A declarations-led group (`decl`, and `dtd` where a line has it) opens with pages 1-5 (`pages.declarations_leading`) plus the detected declarations page, read together (`leading_pages`). Split runs overlap by `run_overlap_pages` (1 page on common-model lines; none on lines whose map sets none). A window whose pages' OCR text would not fit its call is split (`expand_tasks.fit_windows_to_text`); `image_only` windows are sized by page count.
- **The fast format profile is the configured one** (`configs/shared/format.yaml`). Under it every group reads the same windows (`policy_windows._plan_aligned`), sized for the most demanding group, and the model answers in the compact form (`{"r","v","p"}`). `_parse_generation` expands every answer to full envelopes (`common.canonical.expand_answer`) before anything else reads it. Decoding uses the compact schema (`common.canonical.compact_schema`).
- Each window's prompt and decoding schema are its group's slice. Page references are bounded to the window's pages (`common.schemas.with_page_bounds`). Pages are interleaved `[image][<page k of N> + text]…` with the document's real page numbers; `image_only` keeps the markers without text.
- **Loss Runs** are windowed only when their estimated rows overflow one answer (`lossrun_windows`, `answer_cap("lossrun_rows")`, 8,192 tokens), merged by `serving/lossrun_merge.merge_extracted_windows` and reconciled against printed totals (`claims:merge_conflict`, `claims:totals_mismatch`, `reconciliation` in the output). Anything else is one call over every page.
- `pages_used` records the pages read; for a policy it is every page.

### 6. Sending the windows — `serving/pipeline.py`
- A served model that answers each request on its own (`model_runner.can_submit`) gets every window at once (`_read_as_answered`); a failed window's halves go out as soon as it fails. The in-process engine reads in rounds of calls of at most 96 windows (`MAX_WINDOWS_PER_BATCH`, `_read_in_rounds`), and a call refused whole is read again in halves (`_batch_in_halves`).
- Send order: longest expected answer first (`_send_order`); under the fast profile a window's groups go together, so later groups reuse the pages' prefill from vLLM's prefix cache. `ExtractionRequest.send_order="reversed"` sends them the other way (the golden eval's repeat pass). Answers are always merged in planned order.
- Each window's answer limit is its section task's budget (`common.config.answer_cap`). An answer cut off (token limit, `repetition`, `stream_cut`, `deadline`) is refused even if it parses. A repeating answer is stopped early (`inference_core.openai_backend.repetition`; the in-process engine through `model_runner.watched_chat`).
- **A window that fails** (unparseable, cut off, a section in the wrong JSON type) is split in half and retried; the half holding the declarations page keeps the lead (`_halves`). A window still unreadable at one page is dropped and flagged `window_failed:<group>:p<page>`; the rest of the document is served. Only when no window succeeds is the request an error.
- **A window that never reached the model** (`ModelTransportError`: unreachable, broken off, 408/425/429/5xx) is resent once as it is; failing again, it fails the whole document, so the Azure worker retries the job whole. A one-page window still being written at its request deadline also fails the document. Any other 4xx is a plain failure, split and dropped as above.
- A page reference outside the window's pages is dropped and the value flagged `<path>:page_ref_outside_window`.
- Each window read is recorded in `ExtractionResult.window_stats` (group, pages, attempt, outcome, seconds, time to first token, prompt and answer tokens, finish reason), not in the served JSON. `_check_prompt_tokens` holds the server's prompt-token count to what the pages imply (`inference_core.prompt_tokens`, with the token math in `common/token_math.py`) and records `prompt_check`; when the check cannot run it records `prompt_check_error` and warns once.

### 7. Merge and serving guards
- **Merge** (`serving/policy_merge.merge_policy_windows`): windows in group order, `decl` first, the declarations-led window first within its group, then page order. A later differing value is kept out and flagged `<path>:merge_conflict`. On a common-model line ids are namespaced per window, units are joined on their keys, references rewritten, rows doubled by an overlap page dropped, and the whole document renumbered in printed order (`common.structural_ids.renumber_structural_ids`). Flags: `joined_without_identifier`, `ambiguous_unit`, `unplaced_fragment`, `dangling_reference`.
- **Label conventions on common-model lines** (`serving.conventions.conform`, with page text): the training label rules applied to the merged answer. A value no page prints is flagged `<path>:not_printed`; on a born-digital document an extra field no page prints is dropped (`additional_fields:not_printed_dropped`). A form edition left in an unambiguous shape inside its form number moves to an empty `edition_date` (`MM/YY`) and takes the number's span. A value re-cited to pages it did not cite is flagged `printed_off_cited_page`.
- `likely_missed` flags a declarations field left empty while a page line opens with its label and a value of its type. `role_mismatches` flags a value served under a label that prints another value. `unit_link_flags` flags a coverage with no `applies_to` when there are two or more vehicles, boats or buildings, or another row of its code is linked. Empty optional objects and empty rows are dropped (`_without_empty`).

### 8. Printed values only — `serving/printed_values.py` (decision 2026-10-24)
- After the merge, every policy's values (extra fields, table rows and coverage `form_refs` included) are looked up on the page text with the OCR-tolerant `common.grounding.ground_tolerant`, scans too: the cited pages with text first, then every page with text. The text is `ExtractionRequest.printed_texts` when sent, else the text the model read.
- Printed: kept. Printed nowhere while every cited page has text: in `strict` mode (`printed_values.mode`, the default; `configured_mode` refuses an unknown mode) taken out and flagged `<path>:not_printed_removed`; in `flag` mode kept and flagged `<path>:not_printed`. A key stays null; an extra field leaves its list; a table row goes when nothing in it is still printed. Rows left are never renumbered; a link to a removed row is dropped and flagged `link_to_removed_row`. A value removed with its row is named `list[#N]`, its place in the model's answer.
- Not found while a cited page has no text: kept, flagged `<path>:unverified_unread_page`. Printed only off its cited pages: kept, flagged `printed_off_cited_page`. `raw` that does not read as `parsed` (`serving/raw_parsed.py`): taken out in strict mode, flagged `parsed_mismatch`. A row whose printed values share no page: `row_not_colocated` (flag only).
- The record is `ExtractionResult.printed_check` (not in `as_dict()`; the golden eval reads it). With no page text nothing is checked: `printed_check` is None and the result is flagged `printed_values:no_page_text`.

### 9. Confidence and the output contract — `serving/pipeline.py`
- With the release's calibrators (`releases/.../calibration/{format}/calibrators.json`), each field is scored by `_feature_calibrated`; a field type with no calibrator or no threshold goes to review (`:no_confidence`, `:low_confidence`). Without them, the v1 transform (`calibration.apply_calibration`) is used and every field goes to review (`:uncalibrated` on each one the transform would have accepted); a doc type with no v1 parameters is refused. The base-model fallback has no fitted calibrators: its confidence is raw, as a diagnostic, and every field goes to review.
- List completeness (`calibration.list_completeness.check_document`) and `near_tie` (top-1/top-2 margin below `serving.near_tie_margin`, 0.1 nats) are added. `serving.consistency` flags values that cannot all be true together (date order, a term over 24 months, implausible dates, limit order, premium below items, malformed postal code, state or policy number). `serving.value_flags.route_served` sends a value with `printed_off_cited_page`, `role_mismatch`, `near_tie`, `parsed_mismatch` or a consistency flag to review (`flagged` in its envelope).
- The canonical envelope gets the system fields (`common.canonical.with_system_fields`). A common-model policy gets the full-text tier `text_sections`, built from the page text by Fideon's section builder (`common.full_text.build_text_sections` over `common/section_builder`, copied byte for byte and recorded in `SOURCE.json`; `scripts/check_section_builder.py` reports drift), never by the model; with no page text it is flagged `text_sections:no_page_text`. `text_coverage` is the share of page-text words found in the served JSON.
- Keys the client's schema does not name are removed and flagged `<path>:unknown_key_removed` (`_without_unknown_keys`).
- **Schema validation mirrors the audit gate.** `schema_verdict` judges the answer against the client's full schema; on a common-model line it judges the every-key-filled answer, and a required section no window wrote or an object written empty (`minProperties`) still fails. With `strict_schema` (the default) an invalid answer is an error; otherwise it is flagged `schema:invalid`. `served_schema_verdict` is the same judgement for checkpoint selection (`evaluation/metrics/coverage_metrics.score_schema_validity`).
- After validation, every key of the line's schema is filled, `null` where nothing was extracted (`common.canonical.with_all_keys`, without `fideon:` annotations).
- `line_of_business` is the line resolution `{lines, source, confidence, candidates, hypothesis_agreed}` for a policy (`source`: `caller`, `detected`, `detected_uncertain`, `undetected`), else `{value, confidence}` from the line spans. `route` records the adapter, the classifier's confidence and method, `lob_fallback_used` and `lob_source`.

### 10. Line of business — `serving.pipeline._resolve_lob`
- The caller's line wins; a confident classifier reading another adds `lob:caller_disagrees`.
- Detection ships **off** (`routing.detect_lob: false`) until `scripts/measure_lob_detection.py` has measured it. Off, a policy with no line is refused, or read against the fallback when the caller sends `allow_lob_fallback`. On: at or above `lob_confidence_threshold` (0.90) the line routes as the caller's would; at or above `lob_uncertain_threshold` (0.60) it is used and flagged `lob:uncertain`; below, or when the read fails, the base model reads the policy against the fallback, flagged `lob:undetected`.

### 11. Fields first — `serving/fields_first.py` (production on Azure)
- One H100 serves both the model and MinerU. A document is read in two phases (IMPL-16 runs them).
- **Phase 1, `read_fields`:** the pages are rendered and the model reads their images (`image_only`) through `extract` with the release's calibrators (`extract_options`). Then `verify_fields` has MinerU read the pages the fields cite — at most `fields_first.verify_page_budget` (40) pages, chosen to cover the most values (`pages_to_verify`), or pages 1-5 when the fields cite none — and holds a copy of the result to that text (`printed_values.hold_result_to_printed`). A value not found there is held back as `<path>:pending_full_text`, not removed. `likely_missed` and `role_mismatches` run on those pages, and confidence is scored again on the text (`rescore_on_text`). If no cited page has text, or the check fails, nothing is served before the full text. `verify=False` serves unchecked fields flagged `fields:unverified` (timing tests only).
- **Phase 2, `add_full_text`:** MinerU reads the remaining pages, and `attach_full_text` holds the model's answer to the whole document: a held-back value printed elsewhere comes back, one printed nowhere is removed as on the standard path. It looks an image-only read over for `likely_missed` and `role_mismatch`, adds the full-text tier (a tier that adds schema errors is left out, flagged `text_sections:invalid`), and confidence is scored again.
- `GpuTurns` orders the GPU: MinerU starts a run of pages (`ServedOcr.run_pages`, 20) only while no document's fields are being read, from rendering through verification; a run waits at most `text_max_wait_s` (600 s) and then goes anyway (`overdue_runs`).
- MinerU text comes from the pinned served MinerU (`configs/shared/ocr.yaml`: 4.0.12, tier `basic`); `ServedOcr(expected_version=)` refuses text from another version or tier.
- Every result the Azure worker serves carries a `pins` block: `base_model`, `release`, `prompt_hash`, `window_plan`, `format_profile`, `ocr_version`, `ocr_tier`, `calibration` (`azure_service.runtime.served_pins`) and `schema_version` (`schema_pin`, per job).

### 12. The request path — `serving.pipeline.extract`
Pages numbered 1..n (`_numbered_pages`) → classify, or take the caller's `doc_type` → route adapter/schema → resolve the line → route through the serving plan (the release, or the base model + `_fallback.json`) → the release's adapter and calibrators → read: policy windows / Loss Run windows / one call → merge → conventions, `likely_missed` and `role_mismatches` (common-model lines with page text) → printed values only (every policy) → unit links → confidence → list completeness and near ties → consistency → envelope and system fields → full-text tier → unknown keys removed → schema verdict → every-key fill → `text_coverage` → `ExtractionResult`.

IMPL-12 (testing) and IMPL-08 (golden eval) call this function and never re-implement it. Checkpoint selection merges with the same `serving/policy_merge.py` and judges with `served_schema_verdict`.

## Constraints
- Non-GPU work runs outside RunPod; RunPod serves only a stock vLLM and the MinerU API.
- One promoted release per tenant, served as a LoRA on the bf16 base; nothing merged.
- Every page read, in windows planned by the training function; nothing in the serving config selects pages.
- No value served that no page prints (strict mode); at the fields-first gate, no field served before it is held to the pages it cites.
- Calibrated confidence where the release has calibrators; otherwise every field goes to review.
- Every response schema-validated before return.
- MinerU parity with the training corpus pin; prompt-hash, window-plan and base-model parity at cold start.
- No document content in logs or error messages; no raw request or response persisted by the endpoint.
- Line-of-business detection stays off until measured.

## Acceptance checklist
- [ ] A policy of any length is read in windows over every page; `pages_used` is every page; a page no signal matches is read like any other.
- [ ] The declarations-led window is pages 1-5 plus the detected declarations page, read together; a declarations page behind a cover sheet joins it.
- [ ] Under the fast profile every group reads the same windows, and the compact answer is expanded before the merge, the printed-values check and calibration.
- [ ] A window that fails is split and retried; one unreadable at one page is dropped and flagged `window_failed:<group>:p<page>`, and the rest is served.
- [ ] A window that never reached the model is resent once; failing again, the document fails and is not served.
- [ ] A single line the promoted release trained on is read by the release; any other known line, or a package, by the base model against `_fallback.json`, flagged `route:lob_fallback_used`; no line is refused unless `allow_lob_fallback`.
- [ ] Below the classifier threshold the document is flagged `routing:low_confidence`; an unclassifiable one is refused while `routing.fallback_doc_type` is unset.
- [ ] Cold start refuses a release with another prompt hash, window plan or base, and a config holding a retired routing setting; `promote` refuses the same releases.
- [ ] In strict mode a value no page prints is removed and flagged `<path>:not_printed_removed`; rows left keep their ids; flag mode keeps it flagged `<path>:not_printed`; an unknown mode is refused; with no page text the result is flagged `printed_values:no_page_text`.
- [ ] Fields first serves only fields held to the pages they cite; a value not found there is held back as `pending_full_text` and settled by the full text.
- [ ] A served policy JSON carries every key of its line's schema, `null` where nothing was extracted, and no key the schema does not name.
- [ ] A common-model policy with page text carries `text_sections` from the section builder, valid against the client's schema; one without is flagged `text_sections:no_page_text`.
- [ ] A form edition in an unambiguous shape inside its form number moves to an empty `edition_date`; a filled one is never overwritten.
- [ ] A schema-invalid answer is an error with `strict_schema`; checkpoint selection judges with `served_schema_verdict`.
- [ ] Without release calibrators every field is flagged for review; raw confidence is never presented as calibrated.
- [ ] A request with no MinerU text is refused.
- [ ] Every Azure result carries the `pins` block.
- [ ] No document content in logs; `serving.pipeline.extract` is the single path IMPL-08 and IMPL-12 reuse.
