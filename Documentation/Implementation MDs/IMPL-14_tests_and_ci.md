# SPEC 14 — Unit Tests + CI Fixtures

> Aligned with the code of 2026-10-22.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: all prior specs (01–13).
>
> **Architecture refs:** `finetuning-architecture-v2.5.docx` §0a/§0b (schema + LoB contracts), §5 (list-field recall), §7 (prompt identity, split leakage), §8a/§8b (MinerU pin, tenancy), §10 (collator masking, §10.2 parity tests), §15 (gating).

## Goal

A test suite + CI that verifies the correctness-critical pieces without GPUs or live Azure, plus small fixtures. The tests exist to catch the failure modes the architecture calls out as silent — the ones that don't show up in a loss curve.

## Deliverables

### 1. `tests/fixtures/`
- Tiny synthetic stand-ins per active doc type, MinerU-style OCR markdown, and matching golden JSON — CI-small, **containing no real PII** (`tests/fixtures/golden/`, `tests/fixtures/ocr/`). Test PDFs are drawn at test time with PyMuPDF rather than committed.
- Fixtures covering the required corpus edge cases (arch §7): a multi-page document, a scanned/low-quality page, a document with a legitimately-absent field, and a Loss Run with a variable-length `claims` array. A 6-page common-model personal auto document (`tests/fixtures/schema2/personal_auto_6page.json`) splits vehicles from their coverages across windows.
- Flat golden fixtures (ACORD, Loss Run) **carry `line_of_business`** as a list, covering more than one LoB value; canonical policy fixtures carry none (their line is the label metadata's `lob`); the empty-list case is built inline by the schema tests.
- Label-metadata fixtures carry **`field_provenance`**, with the same canonical field appearing under **two different surface labels** across fixtures (`named_insured.primary_name` as "Applicant" in `policy_0001`, "Named Insured" in `policy_0002`) so alias slicing has something to slice.
- At least one **confusable co-occurrence** fixture: a policy page carrying both a named insured and a certificate holder, with the golden JSON resolving them to different canonical keys.
- Seeded synthetic logprob/generation samples for calibration + confidence tests.
- An **in-memory mock Blob backend** (`artifact_registry.blob_client.InMemoryBackend`; no live Azure).
- A seeded fake corpus manifest with `mineru_version`, `schema_version`, `prompt_template_version` for the pinning tests.
- `tests/fixtures/old_style_prompt_fingerprints.json` — the frozen prompts, decoding schemas and required fields of every self-contained line, ACORD form and Loss Run slice.
- `tests/conftest.py` — the `no_window_overlap` fixture, for tests of what happens at a window boundary (the configured one-page overlap moves the boundaries; the mechanisms remain).

### 2. Core tests (mirror each spec's acceptance checks)

**Highest-value — these guard silent failures:**

| Test | Guards |
|---|---|
| `test_training.py` (+ staged-row masking in `test_stage_data.py`) | **Label masking**: loss only on assistant JSON tokens; system/image/OCR masked to `-100` (IMPL-06). Broken masking trains the model on its own prompt and is invisible in loss curves. **Mutation-check it** — a deliberately broken masking implementation must fail this test. |
| `test_dataset_builder.py`, `test_split_by_line.py` | No `source_id` crosses train/val/test; modality expansion happens **after** the split (IMPL-05). Leakage inflates every eval number the project trusts. |
| `test_inference_core.py` | `build_messages` identical across eval/serving/testing contexts; `map_field_spans` correct for scalar, `null`, and list-row fields (IMPL-07). **Guards test == prod at the primitive level.** |
| `test_parity.py::test_template_token_parity` (`gpu`, run on the pod) | **Train/serve token parity, measured**: on ten fixed rows (or the first rows of `$FIDEON_PARITY_CORPUS`, a staged epoch file) the trainer's own ms-swift encoder and a vLLM engine built on the serving pixel budget give identical prompt token ids and identical per-image token counts, each equal to the resize rule the sequence caps use; the processor's pixels per visual token equal `cap_check`'s (arch §10.2, `training/parity.py`). It skips without a GPU, the base weights, ms-swift and vLLM, so CI never runs it. `test_serving_output_parity` (HF and vLLM values agreeing) is still a placeholder that skips. |
| `test_serving_pipeline.py` | Testing output == serving pipeline output: the harness calls `serving/pipeline.py` and does not re-implement inference (`test_the_harness_calls_the_serving_pipeline_not_a_copy`, `test_the_harness_does_not_reimplement_inference`; IMPL-11/12). |
| `test_parity.py`, `test_serving_parity.py` | The rendered prompt at corpus-build time (IMPL-05) is **byte-identical** to the one at inference time (IMPL-07) and to `testing/prompts/*.prompt.txt` (IMPL-12), for every doc_type × modality mode and every common-model window prompt. Training/inference prompt drift is the most common cause of post-fine-tuning degradation and shows up in no training metric. |
| `test_old_style_prompts_frozen.py` | The self-contained lines, ACORD forms and Loss Run render exactly as frozen in `old_style_prompt_fingerprints.json`: common-model work runs through code every policy uses and must not move their prompts. |
| `test_schema2_serving.py` | **Replay**: every window serving asks for is answered with the target training built for it, and the served answer is the gold label back — values, units and links, renumbered in printed order. Guards that training targets, the merge and serving agree. |

**Contract and correctness:**

| Test | Guards |
|---|---|
| `test_schema_contract.py` | Schemas load; valid examples pass; malformed/wrong-type fail (IMPL-01/08) |
| `test_schema_contract.py`, `test_labeling.py` | Every flat schema requires `line_of_business` as a list of enum values; out-of-enum values rejected; an empty list accepted (undetermined); a golden label missing the key is rejected by IMPL-04; per-value LoB accuracy computed correctly by IMPL-08 |
| `test_schema_resolver.py`, `test_schema2_model_view.py` | Each SPEC_21 overlay bundles into one self-contained schema, and a bundle naming another file, another common-model version or an unlisted coverage code is refused; the model view holds what the model writes (no pipeline-filled field, codes from the line's list, no alias text) |
| `test_dataset_builder.py` | 50/20/30 holds; `image_only` rows omit OCR; `noisy_ocr_image` renders the same prompt as `ocr_plus_image` (IMPL-05) |
| `test_dataset_builder.py` | No corpus file mixes tenants — the one live tenancy rule, because corpus composition is training data (IMPL-05) |
| `test_ingestion_ocr.py` | A MinerU version mismatch between corpus manifest and runtime raises in dataset build, serving, and testing; a corpus read by two MinerU versions is refused; a document another MinerU read counts as unprocessed (IMPL-03) |
| `test_evaluation_calibration.py`, `test_eval_integrity.py` | Normalized match (dates/currency/names), list F1, and **list recall** correct (IMPL-08) |
| `test_evaluation_calibration.py` | **Non-inferiority within each metric's margin δ** (`NON_INFERIORITY_DELTA`, 0.005 to 0.015 per metric), judged on its own for **any single** gating metric — LoB accuracy and confusable misattribution included: with per-document scores a paired bootstrap blocks when its 95% CI lower bound falls below −δ (`test_the_bootstrap_decides_when_per_document_scores_are_supplied`, `test_a_real_regression_is_not_non_inferior`; the bootstrap alone calls one document inconclusive, and the gate compares a metric fewer than two documents measure as a point estimate, with a warning); without them, a point drop beyond δ blocks and one inside δ does not. Not "beat the previous release" on every metric: a release whose intervals show no primary metric improved is blocked at `improvement` unless it names a fixed defect (`fixes_defect`), and without intervals that rule does not fire, which the decision's warnings say. A blank or placeholder `fixes_defect` is refused. An override needs a named approver, a written reason and the gates it waives; a `--continue-from` candidate lacking cross-type regression evidence is blocked (IMPL-08) |
| `test_evaluation_calibration.py` | Temperature + isotonic reduce ECE; params persist/reload; `apply_calibration` **raises** on missing params; list-completeness flags dropped rows even when per-value confidence is high (IMPL-09) |
| `test_registry.py` | `RunManifest` validates; `adapters_depending_on` + `resolve_model_version` + `diff_manifests` correct; each tenant has its own golden eval set path (IMPL-02) |
| `test_serving_pipeline.py`, `test_release_router.py` | Low classifier confidence → an extraction for both candidates, each against its own schema, flagged `routing:low_confidence`; an unclassifiable document refused unless a fallback is configured (then flagged `routing:unclassified`); ACORD returns doc_type **and** form; deterministic release selection (IMPL-11) |
| `test_layout_family_routing.py`, `test_lob_detection.py`, `test_schema2_review_lob.py`, `test_lob_classifier_wiring.py` | A policy reaches its line's layout-family release, else an unrestricted release, else the base model against `_fallback.json` with `lob_fallback_used` and a flag; the family table agrees with the delivered `lob_schema_map.yaml`; line detection never fails a request, ships off, and builds its classifier at cold start only when on and the engine holds the base (IMPL-11) |
| `test_serving_pipeline.py`, `test_policy_windows.py`, `test_schema_sections.py`, `test_page_router_signals.py` | **Every page is read**: the page plan holds every page, a page no section signal matches included; every group reads every page in windows, the declarations group's first window pages 1–5 plus the detected declarations page (found behind a cover sheet too); `image_only` reads every page; `pages_used` recorded; the declarations-page conflict rule resolves duplicates (IMPL-11) |
| `test_training.py` | Fires on perception-type errors below target; **does not** fire on schema/reasoning errors; no code path enables full ViT fine-tuning (IMPL-06) |
| `test_ids_and_run_ids.py` | id build/parse/validate round-trips (IMPL-01) |
| `test_schema_contract.py` | Registry loads for every doc type; **every schema field has a non-empty `description`**; no string appears in both `aliases` and `confusables` for one field; the rendered prompt contains descriptions and **contains no alias strings** (IMPL-01) |
| `test_labeling.py` | A label whose `field_provenance` names a registered confusable is **rejected** by `export_golden_labels`; provenance round-trips through review-tool export (IMPL-04) |
| `test_evaluation_calibration.py` | Misattribution scored correctly on synthetic cases and distinguished from an ordinary wrong value; a rise in it beyond its margin, alone, blocks promotion (IMPL-08) |
| `test_no_runtime_aliases.py` | **No module under `serving/`, `inference_core/`, or `testing/` imports `common.aliases`** — the master §1.4 anti-pattern, enforced by an import check rather than by discipline |
| `test_orchestration.py` | `all` = `finetune` + `package` and **never** invokes extraction; `finetune` ends at merge and `package` owns the gate (quantize → calibrate → gate → push), so a failed gate stops before anything is pushed; no bypass flag exists, and the written override's three flags work only together, for one scope; `--from-stage` resumes and a completed stage re-runs as a no-op; every stage records its wall time and the run report is written after each, named per scope and tenant; a corpus that set aside more than 25% is refused by the build; with per-document scores in both eval reports the gate is a paired bootstrap, records `paired_documents`, blocks a release that improves nothing and passes it with `--fixes-defect` (checked, per scope); reports scored on another frozen set or tenant are not paired, and the decision says why; a scope scored on too few frozen documents of a type is capped at interim, whatever the set's own flag; a gold with no review record never trains (IMPL-13) |
| `test_registry.py` | `resolve_model_version("base")` returns the pinned base with no adapter; a unified run resolves by its tag; a `staged` manifest resolves to volume paths and a `published` one to Blob paths (IMPL-02/13) |
| `test_orchestration.py` | `finetune` writes a Blob manifest with `status: "staged"` even though weights stay on the volume; `package` flips it to `"published"` with real paths and fails loudly when the version is not staged (IMPL-13) |

### 3. CI
- `.github/workflows/ci.yml`, on every push and pull request: install `.[data,eval,dev]`, `ruff check .`, `mypy` over the code packages (`common artifact_registry registry_utils data_pipeline inference_core training evaluation calibration serving postprocessing testing orchestration pilot`), `python -m testing.render_prompts --check` (the checked-in reference prompts are current), then `pytest -q -m "not gpu and not azure"` with coverage. GPU and live-Azure tests are marked (`gpu`, `azure` in `pyproject.toml`) so CI is **CPU-only and fast**.
- **Coverage gate on the correctness-critical modules** (`--cov-fail-under=85`): `training.data_collator`, `inference_core.input_builder`, `inference_core.span_map`, `common.prompts`, `common.normalize`, `evaluation.gating`, `calibration.apply_calibration`, `registry_utils.models` and the split code, `data_pipeline.dataset_builder.split_groups`.
- A **secret/PII scan** job: SSN-shaped strings in `tests/fixtures/` and `schemas/examples/`, and credential-shaped strings (`AccountKey=`, SAS signatures, private keys) anywhere in the repo, fail the build.

## Constraints
- Runs without GPU or live Azure (mock/stub heavy deps).
- Fixtures contain no real PII.
- Deterministic (seeded) everywhere.

## Acceptance checklist
- [ ] `pytest` passes green on CPU-only CI.
- [ ] Collator masking test **fails** when masking is deliberately broken (mutation-checked: `test_training.py::test_masking_out_assistant_tokens_is_caught` and its neighbours; `test_stage_data.py::test_a_run_whose_masking_supervises_the_prompt_is_refused`).
- [ ] Split-leakage test **fails** if expansion is moved before the split.
- [ ] The prompt-parity tests (`test_parity.py::test_the_serving_prompt_is_the_corpus_prompt`, `test_the_reference_prompts_match_the_renderer`) fail when the corpus-build prompt and the inference prompt diverge by a single character.
- [ ] `test_inference_core.py` + `test_parity.py` pass on fixtures.
- [ ] On the pod, `pytest -m gpu tests/test_parity.py::test_template_token_parity` passes: training and serving tokenize each fixed row identically and give each page the token count the resize rule predicts.
- [ ] `test_evaluation_calibration.py::test_a_release_that_is_merely_no_worse_does_not_pass` fails if the improvement rule stops being enforced on per-document scores; `test_orchestration.py::test_per_document_scores_in_both_reports_gate_on_a_paired_bootstrap` fails if the gate stage stops pairing the two eval reports; `test_run_eval_and_endpoint.py::test_list_recall_is_bootstrapped_as_the_gate_reports_it` fails if the bootstrap stops testing the metric the report states; `test_orchestration.py::test_reports_that_cannot_be_paired_say_so_in_the_decision` fails if reports scored on another frozen set or tenant are paired.
- [ ] The every-page tests (`test_serving_pipeline.py::test_every_page_of_a_long_document_is_read`, `test_schema_sections.py::test_every_group_reads_every_page`, `test_policy_windows.py::test_every_group_is_asked_over_every_page`) fail if any page of a document goes unread.
- [ ] `test_schema_contract.py::test_line_of_business_is_a_required_list_of_enum_values` fails when a flat schema drops `line_of_business`.
- [ ] `test_dataset_builder.py::test_cross_tenant_mixing_is_refused` fails when a corpus mixes tenants.
- [ ] `test_orchestration.py::test_no_cli_flag_can_override_the_gate` fails if a bypass flag (`--force`, `--skip-gate` and the like) is introduced; the only override is a recorded written one (named approver, reason, waived gates — `test_evaluation_calibration.py::test_an_override_must_be_attributable`), and its flags work only all three together, for one scope (`test_orchestration.py::test_an_override_flag_alone_is_no_way_round_the_gate`).
- [ ] `test_orchestration.py::test_all_never_reaches_package_on_a_failed_gate` and `test_all_never_includes_extraction` fail if `all` ever pushes past a failed gate, or invokes extraction.
- [ ] `test_orchestration.py::test_finetune_writes_a_blob_manifest_while_weights_stay_staged` fails if `finetune` can complete without writing a manifest to Blob.
- [ ] `test_schema_contract.py` fails when any schema field is missing a `description` (`test_every_field_has_a_description`, `test_every_canonical_field_has_a_description`), or when an alias string leaks into a rendered prompt (`test_no_alias_string_reaches_the_prompt`).
- [ ] `test_no_runtime_aliases` fails the moment a serving or inference module imports `common.aliases`.
- [ ] `tests/test_module_seams.py` passes — **the seam suite**.
- [ ] `ruff` + `mypy` clean.

**Seam tests: run both sides, never hand-write one of them.** A full-tree review found 34 defects while the suite was entirely green, and every one lived at a boundary between two modules that had each been tested against a hand-written fixture rather than against what its neighbour actually produces. Three tests passed *because* they encoded the bug they were meant to guard: one asserted a review flag string nothing emits, one asserted a dry run moves the endpoint, one asserted a dry-run pipeline can publish.

A unit test proves a module is self-consistent. Only a test that runs both sides proves they agree. The suite covers: every `GATING_METRICS` name is one `score_subset` emits (the defect that made the promotion gate unpassable); the review-queue consumer recognises the flags `list_completeness` emits; every `vllm_serving.yaml` threshold reaches a parameter `extract` accepts; both trainers honour their YAML over helper defaults; the double-annotation sample is not a subset of `train`; and the gate's verdict key differs from the eval report's.

**Fix a pair, test the pair.** The trainer tests are parameterised over *both* trainers because the same class of bug — the validation-split leak, then the argument-ordering override — was twice fixed in one of two and reported as fixed in both.

---

## Current implementation (2026-10-21)

2,955 tests collected in 123 test modules (`pytest --collect-only -q`), CPU-only, no live Azure; the `gpu`
tests skip off the pod. Test files added since this spec, each guarding a failure that was real:

| Test | Guards |
|---|---|
| `test_eval_integrity.py` | invented fields scored wrong; nested tables; per-document pooling; failure-rate refusal; image localisation; calibration features on canonical paths; engine release; carrier normalisation; digit-aware estimate; per-window noise spread |
| `test_serving_parity.py` | `lob` on the request; page/image pairing; one blank-page text; one pixel budget; **every date field in every LOB schema** formatted; no page threshold in serving (none exists: every page is read); prompt hash over every prompt input, refused at cold start |
| `test_freeze_eval_set.py` | freeze layout the gate reads; refused twice; refused below 150/type without `--allow-small`; a rebuild after freezing trains on none of the eval set or its families; two tenants freeze independently and read only their own set; an id frozen in one tenant is not frozen in another; a set at the old unscoped root is refused with where to move it |
| `test_split_by_line.py` | band per type, placement per line; small lines train whole; measured lines reach val and test; bands only raise edges; crossing a band never moves a trained family into evaluation |
| `test_release_weights.py` | merge refusals; scoped push paths; off-pod refusal; the base found under `/workspace/models` in all three layouts |
| `test_dependencies.py` | every third-party import installed by some group; requirements files name real groups; the training pod has vLLM |
| `test_detach.py`, `test_pod_run.py` | on the pod every long entry point detaches; already-safe jobs do not re-detach; every entry point classified; the launcher never stops the pod or a run unasked; LF line endings |
| `test_personal_lines_scope.py` | a scope narrowed by line: coverage rules, corpus view, golden eval and serving route by line and refuse other lines |
| `test_mineru_engine.py` | one markdown string per page from MinerU 3.4's content list (a 1.x content list still reads); HTML table rows counted; OCR vs text mode; the engine reads in one pinned way; GPU only |
| `test_review_fixes.py` | both end-to-end reviews: MinerU runs only on the pinned weights and on the GPU, a MinerU 1.x config refused and replaced, mixed PDFs OCR'd whole, serving OCR pin, policy-only line counts, resumable freeze, narrow keeps lines, audit layout / malformed page_ref / unreadable PDF; serving through the chosen release (adapter + calibrators, LoRA when several), rollback pin, freeze guard on types with no test documents, scanned from MinerU's classification, HTML row headers and nested tables, explicit threshold targets, scoped pulls, classify in every scope |
| `test_data_audit.py` | the pre-upload audit on real synthetic PDFs: every blocker, wrong-page and missing values, scans left for OCR, format and period checks, totals, report files, audit output git-ignored |
| `test_gpu_only.py` | model loaders require CUDA; no `device_map="cpu"`/`"auto"`; OCR refuses CPU; pod detection without `RUNPOD_POD_ID`, and look-alikes rejected |

Test files added for Schema 2, the CGL adapter, MinerU 3.4, the accuracy plan and the time cuts:

| Test | Guards |
|---|---|
| `test_schema_resolver.py` | SPEC_21 overlays bundle into one self-contained schema; what a bundle refuses |
| `test_schema2_model_view.py` | the model view: what the model never writes, the rules and codes it is held to, what it must never see |
| `test_schema2_targets.py`, `test_schema2_review_targets.py`, `test_schema2_review2_targets.py` | common-model window targets: rows a window can see, ids from 1, no reference to an unseen row, every target fits its slice; a split coverage's values taught somewhere; a two-part or off-line label refused |
| `test_schema2_serving.py`, `test_schema2_review_merge.py`, `test_schema2_review2_merge.py` | the replay (served answer = gold); the unit merge independent of page order (naming keys before describing keys); the every-key fill never breaks the client's schema; self-contained merges unchanged |
| `test_schema2_eval.py`, `test_schema2_review_eval.py`, `test_schema2_review2_eval.py` | common-model scoring: rows paired by what the page prints, ids never scored, links, codes and overflow scored apart; a merged common-model answer judged as serving returns it (`served_schema_verdict`, `test_a_merged_answer_is_judged_as_serving_returns_it`); self-contained lines score as before |
| `test_schema2_review_lob.py`, `test_lob_detection.py`, `test_lob_classifier_wiring.py`, `test_schema2_review2_misc.py` | line detection: never fails a request, ships off, confidence from the line's own token, cold-start classifier only when on; a section written empty fails; the detection measurement corrupts the pages the classifier reads |
| `test_schema2_review_scripts.py` | the alias and field-type scripts read a common-model line through its bundle |
| `test_old_style_prompts_frozen.py` | self-contained lines, ACORD and Loss Run prompts unchanged |
| `test_layout_family_routing.py` | layout-family routing, base-model fallback, the table against `lob_schema_map.yaml` |
| `test_release_measurements_and_routing.py`, `test_release_tier.py` | latency per adapter and peak GPU memory on every release; a routing table that routes as the endpoint will (rollback pins included), stops a release whose own lines would not reach it or that sits on another base; production at field match ≥ 0.92, interim below, and interim with a warning when the release was scored on too few eval documents of a type |
| `test_lossrun_windows_and_table_f1.py` | long Loss Runs read by window, merged and reconciled; `table_f1` |
| `test_carrier_holdout_and_twin_cap.py`, `test_data_mix_and_model_card.py`, `test_render_rotation.py` | carrier and twin index on rows, the per-line carrier hold-out (one carrier, only in a line naming ten or more; the same minimum when asked for on a delivered split), twin cap per seed and render mode; per-line data mix and the model card; one render per twin per epoch |
| `test_twin_bundles.py`, `test_cgl_originals.py` | SPEC_21 delivery → bundles (two renders, one corrected gold); CGL real documents → bundles with a delivered split, `held_out_carrier`, the `casualty_fleet` scope covering only `gl`, the smoke run taking `--scope` |
| `test_label_rules.py`, `test_grounding.py` | the label conventions applied at bundling (each value once, every page that prints it, wording dropped, unprinted values listed, a form's edition moved out of its number); one matcher for printed values |
| `test_additional_fields_and_page_trim.py`, `test_window_rows_and_prompts.py`, `test_window_home_pages.py` | window targets cite only their pages; additional fields placed by page; `text_sections` never taught; form numbers only where printed; a coverage row taught only on windows showing its home (`HOME_FIELDS`) |
| `test_page_router_signals.py`, `test_answer_token_count.py` | every page read whatever it prints, the declarations found behind a cover sheet; answers counted with the real tokenizer at corpus build |
| `test_scoring_types.py`, `test_eval_measures.py`, `test_field_report.py`, `test_serving_conventions.py` | typed comparison; page-aware hallucination and the new reported measures; `all_fields` in test outputs; served answers held to the label conventions (`not_printed`, `likely_missed`, a moved edition taking the number's span) |
| `test_mineru_force_ocr_and_timeouts.py`, `test_inspect_pdf_fallback.py` | MinerU takes modality and `force_ocr` from its caller, drawn figures force OCR (a table of thin border rectangles does not), a searchable scan reads as a scan, time budgets, every failure a `MinerUError`; a PDF pypdf cannot read is read with PyMuPDF |
| `test_parallel_checkpoint_scoring.py`, `test_validation_cut.py`, `test_validation_generation.py` | one worker and vLLM engine per GPU; in-training loss on its own 160-row sample; two finalists judged on whole documents in the named modes, a mode filter matching nothing refused before an engine loads, a small split still getting its whole-document pass; 96 rows per call, longest answer first, failed calls retried in halves, each pass's cost reported |
| `test_logging_setup.py` | our log lines survive a library silencing the root handlers; one console however often `setup()` runs; every package of the repository is one of ours |

Tests changed with the time cuts as well: `test_transfer.py` (a published adapter is its weights and config;
a re-push sends only what Blob does not hold, by MD5), `test_orchestration.py` (per-stage wall time, run reports
per scope and tenant, the build's >25% set-aside refusal) and `test_ocr_check_and_smoke.py` (the smoke chain's
preflight skips its corpus build only when `finetune` builds it).

Tests rewritten for **every page read** (2026-10-21): `test_serving_pipeline.py` (the routing tests became
`test_every_page_of_a_long_document_is_read` and `test_a_layout_the_signals_do_not_recognise_is_read_in_full`),
`test_schema_sections.py` (`test_every_group_reads_every_page`,
`test_the_declarations_lead_together_and_follow_the_declarations_page`), `test_dataset_builder.py`
(`test_a_long_policy_is_read_on_every_page`; the page-selection and thumbnail-chunk tests went with
`select_policy_pages` and `page_select_chunks`), `test_release_router.py` (the `long_doc_types_for` test removed),
`test_page_router_signals.py`, `test_policy_windows.py` (`test_every_group_is_asked_over_every_page`,
`test_a_declarations_page_behind_a_cover_sheet_joins_the_first_window`, `test_image_only_reads_every_page`,
`test_no_value_is_left_on_a_page_no_window_reads`) and `test_inference_core.py`
(`test_a_windows_pages_come_back_in_page_order`).

Tests added since 2026-10-13:

| Test | Guards |
|---|---|
| `test_dataset_builder.py`, `test_carrier_holdout_and_twin_cap.py` | the carrier rule: a line naming ten or more carriers holds exactly one out entirely, a line with fewer holds none and is recorded (`lines_below_carrier_minimum`), each line counts its own carriers, the same minimum on a delivered split asked for a hold-out; the Loss Run totals example only when the schema declares `totals` |
| `test_labeling_loop.py`, `test_import_labeled_pdfs.py`, `test_preflight.py`, `test_orchestration.py` | reviewed golds: no frontier backend, one asked for by name refused, no external pre-annotation setting left; every gold needs a review record whatever its notes or `draft_backend` say, a twin its seed's (decision 2026-10-22); the import refuses one without and copies the record into the metadata; preflight names every one; the corpus build refuses the run |
| `test_label_rules.py`, `test_import_labeled_pdfs.py`, `test_serving_conventions.py` | form editions: an ISO or bracketed edition moves to an empty `edition_date`, a number shaped like an edition is left alone, a filled `edition_date` is never overwritten; the import blocks an unmoved edition on a common-model line and warns on a doubtful one; a frozen gold is scored under the rule without being rewritten; a served edition takes the number's span |
| `test_pilot.py` | the pilot's floors are the gate's (`PILOT_FLOORS`: threshold, direction, inclusive comparison); schema validity below the gate's floor fails the pilot |
| `test_registry.py`, `test_pod_bootstrap.py` | SAS login: preferred to the connection string, the connection string alone still works, half a SAS login refused without naming the token, no login refused; the bootstrap takes either login and refuses an unquoted token |
| `test_ingestion_ocr.py` | purge plans: a document's plan names every model trained on it; a tenant's reaches every tenant-scoped prefix and its runs |
| `test_run_eval_and_endpoint.py`, `test_release_tier.py`, `test_orchestration.py` | each document's own score kept in the eval report for the bootstrap; the gate stage's paired bootstrap and `--fixes-defect`; too few eval documents of a type cap a production score at interim with a warning |
| `test_preflight.py` | annotation agreement reported, a warning while no minimum is set, a failure below one once set |
| `test_schema2_review2_eval.py` | a merged common-model answer judged as serving returns it |
| `test_parity.py` | train/serve token parity measured on the pod (above), and what it compares (`training/parity.py`) checked without a GPU |

Gate tests added on 2026-10-22 (code review):

| Test | Guards |
|---|---|
| `test_run_eval_and_endpoint.py` (`test_list_recall_is_bootstrapped_as_the_gate_reports_it`, `test_each_per_document_metric_weighted_is_the_metric_the_gate_reports`) | the review's case: reported list recall falling 0.92 → 0.88 is bootstrapped as −0.04 and blocks, where the unweighted mean said +0.125 and improved; each per-document metric's weighted mean is the reported one, across doc types |
| `test_evaluation_calibration.py` (`test_weights_make_each_side_a_weighted_mean`, `test_a_metric_one_document_measures_is_a_point_estimate_not_a_permanent_block`, `test_the_improvement_rule_left_unenforced_is_said_in_the_decision`, `test_a_blank_or_placeholder_defect_does_not_lift_the_improvement_rule`, `test_a_ticket_id_names_a_defect`) | the weighted bootstrap and its refusals; one paired document is a point estimate with a warning, and a real drop on it still blocks; an unenforced improvement rule is a warning; a blank or placeholder defect is refused, a ticket id kept |
| `test_orchestration.py` (`test_reports_that_cannot_be_paired_say_so_in_the_decision`, `test_an_old_report_without_weights_still_pairs`, `test_the_interim_cap_follows_what_the_scope_was_scored_on`, `test_a_written_override_can_be_given_on_the_command_line`, `test_an_override_flag_alone_is_no_way_round_the_gate`, `test_one_override_cannot_cover_several_scopes`, `test_fixes_defect_is_checked_and_given_per_scope`) | reports of another frozen set or tenant, with no shared documents or no recorded set, are not paired, and the decision says why; an old report pairs at weight one; the interim cap by what the scope was scored on (a GL scope's 9 policies of a large set; a policy scope's 200 in a set flagged small); the written override from the command line, recorded in the decision and the bundle, refused unless all three flags are given, and for several scopes; `--fixes-defect` checked and per scope |

Review-record, form-edition and carrier tests added on 2026-10-22 (code review):

| Test | Guards |
|---|---|
| `test_labeling_loop.py` (`test_a_gold_trains_on_its_review_record_never_on_its_notes`, `test_what_drafted_a_gold_is_information_not_the_gate`, `test_a_placeholder_reviewer_names_nobody`, `test_a_review_date_must_be_an_iso_date_not_in_the_future`, `test_an_iso_date_or_date_time_is_a_review_date`, `test_a_review_dated_today_in_the_reviewers_zone_is_not_in_the_future`, `test_a_twin_is_reviewed_only_through_its_seed`), `test_labeling.py` (`test_a_draft_accepted_without_review_is_refused_after_day_zero_too`, `test_an_export_naming_no_reviewer_writes_nothing`) | the review's cases: notes the old guess missed ("Claude-drafted", "AI-generated") and notes it blocked wrongly ("not drafted by AI", "Claude Dupont", "Claudette") decide nothing; `label_studio` passes with a record; placeholder reviewers, non-ISO and future dates refused; a twin is reviewed only through its seed; the export writes nothing unreviewed |
| `test_import_labeled_pdfs.py`, `test_import_prepared.py`, `test_data_audit.py`, `test_preflight.py`, `test_orchestration.py` (`test_a_gold_with_no_review_record_never_trains`) | both importers and the audit refuse a gold with no record, and a twin with only its own; they copy the record and `review_inherited_from`; the edition blocker on common-model lines only (auto, wc, umbrella, property left as printed), `import_prepared`'s included; preflight fails on every label without a record; the corpus build refuses the run |
| `test_freeze_eval_set.py` (`test_an_unreviewed_gold_is_never_frozen_and_nothing_is_copied`, `test_the_frozen_metadata_carries_the_review_record_the_gate_checks`, `test_a_set_frozen_before_the_rule_is_refused_until_its_records_are_copied_in`, `test_the_cli_can_mark_a_frozen_set_reviewed`), `test_extract_with_adapter.py`, `test_compare_models.py` | the freeze refuses an unreviewed gold and copies nothing, and carries the record; the golden eval refuses an older set until `mark_reviewed` copies the records, changing nothing else; `--mark-reviewed`; the harness extracts but does not score an unreviewed gold and says why; a self-contained line's edition is scored as stored; a PDF's gold carries the review record `compare_models` is given |
| `test_twin_bundles.py`, `test_delivered_split.py`, `test_cgl_originals.py`, `test_fill_synthetic_labels.py` | bundles carry the review record: a twin inherits its seed's, never its own; a SPEC_21 row's columns are its seed's, and two records for one seed are refused; `reviews.csv` for real documents alone; a placeholder carrier is never an unseen one; no twin is filled from a seed with no record, and a filled twin names its seed, review and notes |
| `test_label_rules.py` (`test_the_label_rules_hold_on_common_model_policy_lines_only`, `test_the_corpus_build_reads_a_stored_gold_exactly_as_the_gate_scores_it`, `test_the_corpus_builds_verification_applies_the_rule_when_told_the_line`) | one predicate for where the edition rule holds; the corpus build's gold equals the gate's on homeowners, auto and wc; the stored gold is never rewritten |
| `test_carrier_holdout_and_twin_cap.py` (`test_a_carrier_with_a_synthetic_family_is_never_held_out`, `test_a_drawn_split_never_trains_the_held_out_carriers_real_document`, `test_a_synthetic_only_carrier_neither_counts_nor_is_held_out_on_a_delivered_split`, `test_a_carrier_seen_only_with_its_twins_counts_toward_the_minimum_but_is_never_held_out`, `test_a_placeholder_carrier_neither_counts_nor_is_held_out`, `test_a_doc_type_with_no_line_holds_out_as_before`) | on a policy, a carrier a synthetic family trains is never held out; a seed family's real carrier counts toward the 10, a twins-only one does not; a placeholder carrier neither counts nor is held out; ACORD and Loss Runs as before |

Contract tests also check that `.env.example` documents every variable the code reads (and nothing it
does not), and that `IMPL_ALL_COMBINED.md` is regenerated, never hand-edited
(`python -m scripts.combine_specs --write`).

**The coverage gate covers the split code.** It names `data_pipeline.dataset_builder.split_groups` (98% at
commit f5668a4; 95.9% over all the gated modules), where it used to name the removed
`split_train_val_test`, which pytest-cov only warned about.

**Not testable off the pod**, and left to the pod — the Phase 0 spike (`scripts/phase0_spike.py`, which compares
token ids, not decoded text) and `pytest -m gpu`: train/serve token parity (`test_template_token_parity`), real
tmux across a disconnect, a real merge, the masking encoder on ms-swift's template, vLLM 0.11 loading Qwen3-VL
with the pixel budget, MinerU 3.4.5's pipeline backend reading a page on the GPU with its pinned weights, the multi-GPU checkpoint-scoring workers on
real vLLM engines (the tests fake the engine), and a real upload.

## Changes since 2026-10-05

- Added the alignment line.
- §1: fixtures described as they are (PDFs drawn at test time, flat vs canonical goldens, provenance in label metadata, the 6-page common-model fixture, frozen prompt fingerprints, `no_window_overlap`).
- §2: added the frozen-prompt and replay guards, the SPEC_21 bundle/model-view tests and the routing tests; corrected rows that described the old behaviour (`line_of_business` as a list; low classifier confidence extracts both candidates; the gate sits in `package`; no "Foundation + per-type" resolution).
- §3: CI described as in `ci.yml` (install groups, mypy scope, `render_prompts --check`, the gate's module list, the secrets scan).
- Acceptance checklist: placeholder test names replaced by the real test functions; the gate check is "no CLI override", with the recorded written override as the only exception.
- Current implementation: 2,765 tests in 122 modules; MinerU 3.4 and per-tenant freeze rows; a table of the test files added since; the stale split module in the coverage gate; MinerU 3.4.5 and multi-GPU scoring added to what only the pod can test.
- The CI coverage gate names `split_groups` instead of the removed `split_train_val_test` (commit f5668a4).

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx`.
- §2: `test_parity.py` measures train/serve token parity on the pod (GPU-marked; skips off it), and testing == serving is `test_serving_pipeline.py`'s; the gate rows say non-inferiority within δ by a paired bootstrap (a point comparison without per-document scores), not "beat the previous release", with the improvement rule and `fixes_defect`; the long-document row is every page read; the orchestration row gained the paired gate, the interim cap and the reviewed-gold refusal.
- Acceptance checklist: the token parity test on the pod, the improvement rule and paired gate, the every-page tests.
- *Current implementation*: 2,801 tests in 122 modules; the tests rewritten for every page read; a table of the tests added since (carrier rule, reviewed golds, form editions, pilot floors, SAS login, purge plans, the paired gate and interim cap, annotation agreement, the served schema verdict, token parity); rows for the schema verdict, the tier, the carrier rule, label rules, router signals and conventions updated; token parity added to what only the pod tests.

## Changes of 2026-10-22 (code review)

- §2: the gate rows: a metric fewer than two documents measure is a point estimate; an unenforced improvement rule is a warning; a placeholder defect is refused; unpaired reports and the per-scope interim cap; the written override's three flags.
- Acceptance checklist: the weighted bootstrap and pairing tests; the bypass-flag check names the written override's flags.
- *Current implementation*: a table of the gate tests added in the code review; the tier rows say the per-scope interim cap.
- *Current implementation*: a table of the review-record, form-edition and carrier tests added in the code review; the reviewed-gold, form-edition and orchestration rows say the new rules.
- *Current implementation*: 2,955 tests in 123 modules.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.
