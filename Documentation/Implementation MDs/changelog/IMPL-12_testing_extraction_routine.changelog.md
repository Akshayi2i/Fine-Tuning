# IMPL-12 — Testing / Extraction Routine: changelog

The dated history of `../IMPL-12_testing_extraction_routine.md`, oldest first, moved here on 2026-11-24 (review R6) so that file says only what is true now. Entries are not rewritten: an entry describes the code of its date. The *Current implementation (2026-10-21)* section below is the old file's, kept as it was.

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
  one failing document is recorded in `summary.json` and the batch carries on. **The run exits 1 when any
  document failed** (review R3, decision 2026-11-24; until then only when every document failed), unless
  `--allow-failures N` allows up to N. The last line printed is `read X, scored Y, failed Z, not scored W`.
  `latency_check` and `extract_client` already exit 1 on any failure.
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

## Changes of 2026-10-26 (fast format shipped)

- `testing.render_prompts` renders a fast-format window's reference file as what the model reads: the shared system prompt, a line standing for the window's pages, then the group's request (`render_group_block`). The policy window references under `testing/prompts/` are regenerated in the fast format.

## Changes of 2026-10-26 (pipeline review fixes)

- `latency_check --repeat` reads every document twice, interleaved, on the server path and writes `run_to_run.json`.

## Changes of 2026-11-24 (extraction routine review, Phase 1)

- Title is IMPL-12 (it read "SPEC 12", the name the naming hazard above warns about).
- Goal: says what the harness does not run (fields first, the cited-page check, calibration) instead of
  "mirrors the exact production path" (review R1; the served reading comes in Phase 2).
- One folder per run, `results/{version}/{run_id}/` and `metrics/{version}/{run_id}/`, each with `run.json`;
  the registry moved to `results/extraction_registry.json`, written whole; `test_record.csv` gained a
  `Run ID` column, and an older file is rewritten once with it (review R2).
- Exit 1 when any document failed, `--allow-failures N`; `summary.json` carries `counts` beside its means;
  a closing line prints them (review R3).
- §2: the window groups are `decl`, `arrays`, `forms` (+ `lineblk` where a line has fields outside them).

## Changes of 2026-11-24 (extraction routine review, Phase 2)

- `--as-served` (review R1, decision 2026-11-24: opt-in): each document read as production serves it -
  image_only, the release's calibrators and thresholds (`--release`, default `--model`), the answer held to
  the stored page text - by the gate's own reading (`evaluation.golden_eval.read_as_served`, split out of
  `_read`, which the gate still calls). Beside it, the fields-first result by production's own
  `serving.fields_first.verify_fields`, the stored text standing in for MinerU (`StoredText`):
  `<doc>.fields_first.json`, `fields_first` in the metrics and the summary. `run.json` and every metrics file
  record `path` (`harness` or `as_served`) and `calibration`. `orchestration.run release-check --version vN
  --corpus C` runs it by name. Proof: `tests/test_as_served.py` - the same answer object as the gate's reading
  of the same stored document; the default image_only path served a value no page prints (0.96 confidence)
  that both the gate and `--as-served` take out.
- Reference prompts (review P7): window prompts for every line of `REFERENCE_WINDOW_LINES` (the seven personal
  lines plus `gl`, `bopgl`, `boppr`, `wc`), every group, in both modes; the per-line whole-schema files
  removed (serving never sends them), `policy.prompt.txt` kept and marked so. `Documentation/Prompts/` removed
  for a pointer to `testing/prompts/`.
- The spec split into this changelog and the current specification (review R6, step 2).
