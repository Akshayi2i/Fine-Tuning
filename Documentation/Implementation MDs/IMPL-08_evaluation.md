# SPEC 08 — Evaluation (Metrics + Promotion Gating)

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-06, IMPL-07. Uses the shared inference core (IMPL-07) for all model runs — no forward dependency on serving or testing.
>
> **Architecture refs:** `finetuning-architecture-v2.6.docx` §15 (**metric definitions + LoB metric**), §0b (LoB gating), §3 (ViT gate inputs), §5 (list-field recall), §13 step 6 (the gate is a hard stop), §12 (gate result recorded in the manifest).

## Goal

Evaluate any model/adapter version against the **frozen golden eval set** and gate promotion so no regression ships.

## The gate is a hard stop (arch §13)

A candidate is promoted only if it passes **every** gating metric for the relevant document type(s): the absolute floors, and paired-bootstrap non-inferiority against the tenant's promoted release (arch v2.1 §15). **There is no `--force` flag.** The one exception path is a **recorded written override** (arch v2.1 §15.5): a named approver, a written reason and the gates it waives, stored in the gate decision and the release bundle. It is given on `package` and `all` with `--override-approver`, `--override-reason` and `--override-gate`: all three together. There is still no bypass. `package` reads the recorded decision and refuses a version whose gate did not pass.

## Deliverables

### 1. `common/normalize.py` (shared normalizer — **built in IMPL-01**, consumed here)
> **Moved to IMPL-01.** `common/normalize.py` is now built in IMPL-01 alongside the other `common/` helpers, because **IMPL-04 `derive_aliases` needs it too** — and IMPL-04 builds before IMPL-08. It is a pure, dependency-free string utility, so it belongs with the rest of `common/`. This spec **consumes** it; it no longer creates it.

Field-type normalizers make comparison reflect correctness rather than formatting (arch §15). Insurance fields specifically need this or you under-count correct extractions:
- **Dates**: `01/01/2026` == `2026-01-01`
- **Currency**: `$1,200.00` == `1200.0`
- **Entity names**: `Acme Mfg LLC` ≈ `ACME MANUFACTURING LLC`
- Policy/claim numbers: whitespace and separator normalization; a number read as `1.0` is the identifier `1`.
- Two numbers compare as numbers whatever the field is called (`957` == `957.0`).

**Typed comparison.** Every scoring call site compares through `evaluation.metrics.field_accuracy.values_agree(expected, got, field_path)`: `common.normalize.values_match` with the type the schema declares for that field (`scoring_kind` — a common-model `MoneyValue` as money, a `YearValue` or count as a number, a date, an identifier), falling back to the field's name only where nothing is declared. Exact match never exceeds the normalised match.

One definition of "matches", applied **consistently in the promotion gate, checkpoint selection, calibration fitting, the testing routine (IMPL-12 — `testing.comparison.field_report`, written into test outputs as `all_fields`, grades every schema field with `values_agree`), and alias derivation (IMPL-04)** — divergence silently changes what "correct" means between them.

**Grounding** — whether a value is printed, and on which pages — has one definition too, `common/grounding.py`, shared by the data audit, the OCR-agreement calibration feature (IMPL-09) and the hallucination metric: case, punctuation and line breaks ignored; a value is on a page as a phrase on word boundaries, as every one of its words (across lines and table columns), or — an amount or a count — as the same figure in another format. `ground(raw, cited, pages)` returns `on_cited_page`, `on_other_page`, `not_printed`, or `unchecked` (shorter than `MIN_CHECKABLE_CHARS` = 3, no page text, or a cited page with no text).

### 2. `evaluation/metrics/` — one module per metric, each pure and testable

| Where (`evaluation/…`) | Metric | What it catches (arch §15) |
|---|---|---|
| `metrics/field_accuracy.score_fields` | Exact + normalized match, per-field and aggregate, by the schema's declared type (`values_agree`) | Core extraction accuracy |
| `metrics/field_accuracy.score_list_field` | Precision/recall/F1 for list fields (claims, schedule rows) | Precision/recall on repeating structures |
| `metrics/field_accuracy.score_list_field` (recall) | **Row completeness** — extracted row count vs ground truth; missed-row rate | Whole rows silently missed — a dropped Loss Run claim is invisible to per-value confidence (arch §5) |
| `metrics/coverage_metrics.score_schema_validity` | Parse + validate against the doc_type schema; a whole common-model document is judged as serving returns it (`serving.pipeline.served_schema_verdict`) | Structural reliability; mirrors the Fideon SPEC_07 Stage 3 audit gate |
| `metrics/coverage_metrics.expected_calibration_error` | Expected Calibration Error on confidence vs correctness | Whether the confidence numbers are trustworthy, not just the extractions |
| `run_eval` — the `noisy_ocr` subset | Scored on the **deliberately-noisy-OCR subset** | Did the model correctly override bad OCR using the image? |
| `run_eval` — `image_only` / `scanned` subsets; `training/vit_gate.classify_error` on every error record | Separate accuracy for the `image_only` and `scanned` subsets | Confirms the second production pathway works; **feeds the ViT escalation gate** (IMPL-06) |
| `doc_type_classifier_accuracy` — not applicable while the corpus builds no classify rows | Doc-type + ACORD-form classification accuracy | Whether the right adapter/prompt/schema is even selected — **a classifier at 92% caps the whole system at 92%** (arch §4a) |
| `metrics/coverage_metrics.score_lob` | **`line_of_business` detection accuracy, overall and per LoB value** | The VLM is the fallback LoB detector when L1/L2 miss (arch §0b) |
| `metrics/confusable.score_alias_accuracy` (not yet called by `score_subset`); `metrics/unseen_labels.label_split` | Field accuracy **sliced by the observed surface label**; on common-model lines, `seen_label_field_match` / `unseen_label_field_match` (labels training showed vs never showed) | Whether the canonical mapping generalises across phrasings, or only works on the dominant one (master §1.4) |
| `metrics/confusable.score_misattribution` | Rate at which a **confusable entity's value is returned as the canonical field** | The failure that produces confident, well-formed, wrong extractions — a certificate holder returned as `insured_name` |
| `release_measurements` (golden eval) | P95 latency per adapter of each generation round, peak GPU memory | Production feasibility, not just accuracy — recorded in the eval report and the release bundle, not gated (there is no latency SLA; decision 2026-10-19) |
| `metrics/extraction_faults` — `score_false_nulls`, `score_hallucinations`, `score_page_refs` | False-null rate, hallucination rate (schema fields; `additional_fields` apart), page_ref exact rate / precision / recall | A value on the page emitted as null; a value printed on no page sent (looked for on the pages it cites); a right value citing the wrong pages |
| `metrics/common_model.CommonModelTally` | `reference_accuracy` (links), `coverage_code_accuracy`, `coverage_code_recall`, `additional_fields_recall` / `_precision` | Common-model lines: which unit a coverage applies to, whether it got the right code, the overflow scored apart from the core fields |
| `metrics/lossrun_table.table_f1` | Claims matched by claim number (or, unnumbered, by loss date) with total incurred within a cent, micro-averaged | Loss Run table reading as a whole |
| `metrics/auto_accept.score_auto_accept` | `auto_accept_error_rate`: of the values delivered without a review flag, the share that were wrong | What the release's review thresholds promise, measured on documents they never saw |
| `run_eval.score_subset` | `field_precision` / `field_recall` / `field_f1`, `field_accuracy_by_lob` (with `field_scored_by_lob`), `weakest_fields`, `worst_line_field_match`, `error_summary` | Writing little but well vs everything half wrong; where accuracy is won and lost; one line hiding behind the others' volume |

**Alias accuracy specifics (master §1.4).** Joins predictions to each eval document's `field_provenance` (IMPL-04) and reports accuracy per canonical field × surface label. This turns an unhelpful aggregate into an actionable one: *0.94 on "Named Insured", 0.61 on "Applicant"* tells you the mapping is not generalising and names the documents to go collect. **Reported, not gating** — rare aliases have too little support for a stable gate, and gating on them would block promotion on noise. The report currently carries the seen/unseen-label split (computed when the eval metadata carries the training set's seen labels and the page text); `score_alias_accuracy` exists but `score_subset` does not call it.

**`confusable.py` specifics.** For each canonical field, check whether the returned value matches the document's value for one of that field's registered **confusables** instead. Requires the eval golden labels to carry the confusable entities' values, so the frozen eval set must include the confusable co-occurrence documents from IMPL-05. **This is a gating metric.** Ordinary field accuracy already penalises a wrong value — but misattribution is worth isolating because it is systematic rather than random: it means the model has collapsed two distinct entities, it will keep doing so, and the output is fluent and confident enough to pass every structural check.

**LoB accuracy specifics (arch §15):** LoB accuracy is reported as **its own metric, measured per LoB value**, and is **never averaged into overall field accuracy** — a class that is rare in the corpus must not hide inside a healthy-looking aggregate. It is a gating metric. It scores the extracted `line_of_business` field, which a canonical policy label does not carry, so it is not applicable to the policy adapter. Reading a policy's line when no caller supplied one is a different number, measured apart (*Current implementation*).

**Mode accuracy specifics:** must also classify errors as **perception** (misread characters, missed checkboxes) vs **schema/reasoning** (right value, wrong field), because IMPL-06's `vit_gate` needs that distinction, not just the accuracy number. `classify_error` also names **omission** (nothing produced) and **invented** (a value where the label has none and no other field holds it); invented errors are counted in the report's error totals and left out of the ViT error mix.

### 3. `evaluation/run_eval.py`
- Given `--model vN` (resolved via IMPL-02) + the frozen golden eval set, run inference **through the serving pipeline** (`golden_eval.evaluate_version` → `serving.pipeline.extract`, built on the IMPL-07 inference core — so windows, merge, date formatting and calibrated confidence are what is measured; serving reads every page of a document, window by window, never a selection of pages) across all eval docs and all three modes, compute every metric, and write the report where the gate reads it (`paths.eval_report(version, scope=ADAPTER.name)` — `eval-reports/{version}/scope/policy/summary.json`). Every frozen document of the adapter's types (policies) is scored; `run_eval` has no `--scope`. Broken down **per doc_type and per subset** inside it (`by_doc_type`). A version is the base model with its adapter applied, as it is served: nothing is merged (decision 2026-10-23). The full-set subsets record `field_scored_by_lob` (how many values each line was scored on) next to `field_accuracy_by_lob`. The report also records each real document's line (`per_document_lines`, `{source_id: line}`; synthetic twins left out), and `run_eval.line_document_scores` groups each document's field match and weight by that line for a continued run's gate (§4).
- Runs the eval subsets explicitly: `image_only`, `scanned`, `noisy_ocr`, `long_policy`, plus the full set — and `held_out_carrier` when the set holds documents of a carrier held out of train and val (reported as `held_out_carrier_match`, never gated).
- Records per-document error records (field, expected, got, error class, mode, scanned, line) so `vit_gate` and failure-mode analysis have real material rather than aggregates, and totals them per subset (`error_summary`: by class, by line, the 15 fields that fail most).
- **Keeps each document's own score** for the gate's paired bootstrap (`SubsetReport.per_document`, gathered by `EvalReport.per_document_scores()` and written as the report's `per_document`, `{metric: {source_id: score}}`): `field_normalized_match`, `field_exact_match`, `list_field_recall` and `schema_validity_rate`, plus the `image_only`, `scanned` and `noisy_ocr` subsets' field match as `image_only_accuracy`, `scanned_accuracy` and `ocr_arbitration_accuracy`. A document's score is the mean of its readings (one per mode), or of its list fields for `list_field_recall`. **Each document also has a weight** (`SubsetReport.per_document_items`, gathered by `EvalReport.per_document_weights()` and written as `per_document_weights`): how much it counts in the reported metric — its list fields for `list_field_recall`, its readings for the others, scaled by its doc type's share. The weighted mean of the scores is the reported metric, so the bootstrap tests the number the floor reads; an unweighted mean once called a fall in list recall from 0.92 to 0.88 an improvement. The report also records `eval_set` (`run_eval.eval_set_record`): the tenant, the frozen set's `frozen_at` and `frozen_from_corpus`, the documents scored per type, the real ones per type (`real_documents_by_doc_type`, twins excluded), and the scoring code's fingerprint (`scoring_fingerprint`: a hash of the syntax of the modules that score, `run_eval.SCORING_CODE` — label rules and verification, the noisy-OCR corruption, the golden eval, `run_eval`, the metrics and the normalisers they use — so a comment, docstring or line ending does not move it, and a rule change does). Other gating metrics carry no per-document scores.
- **The golden eval set is frozen and versioned separately, one per tenant, human-double-verified, and held constant across corpus versions** so model versions compare apples-to-apples over time (arch §8). Never train on it — assert no eval `source_id` appears in the tenant's corpus train or val split (`assert_eval_set_disjoint`).
- **Every gating metric must be emitted by `score_subset`.** `gating.GATING_METRICS` is checked with `require_all_measured=True`, so a metric the scorer never computes blocks *every* candidate, permanently, and the gate has no override by design. This is not hypothetical: `ece_confidence` and `confusable_misattribution_rate` were both gating metrics that `run_eval` never produced — `score_misattribution` had zero production callers and ECE had none at all — which made the gate unpassable under any input. The names must match exactly too: `list_field_recall` and `lob_detection_accuracy` are the gate's spellings, and emitting `list_recall` or `lob_accuracy` instead means the metric silently never arrives. **A test asserts the two sets agree** (`tests/test_module_seams.py`), because a hand-written metrics dict in a unit test proves nothing about what the scorer emits.
- ECE needs per-field confidences: `score_subset` reads them from each document's `metadata["field_confidence"]` and skips fields that carry none, rather than assuming a value. An unmeasured field is not evidence in either direction.
- A metric with nothing to measure is **absent, not zero** (`None` values are dropped from the subset report).
- Note: classifier accuracy uses the classifier from IMPL-11 once it exists; until then `run_eval` scores extraction with the true doc_type supplied, and classifier metrics fill in on a later pass. This keeps eval runnable before serving is built.

### 4. `evaluation/gating.py`
- `promotion_gate(candidate_metrics, current_metrics, *, per_document=None, floors=None, continued_from=None, line_evidence=None, line_evidence_missing=None, require_all_measured=True, fixes_defect=None, override=None, small_eval_set=False, warnings=None, baseline_not_comparable=None)` → `GateResult` (per-metric verdicts, failed gates, blocking reasons, tier, warnings). It takes no scope. Three conditions, all of which must hold (arch v2.1 §15.5):
  1. **Absolute floors** (`PILOT_FLOORS`; `floors_for(scope)` is gone) — applied to the first release too: `field_exact_match` / `field_normalized_match` 0.85, `list_field_recall` 0.75, `schema_validity_rate` 0.98, `doc_type_classifier_accuracy` 0.95, `lob_detection_accuracy` 0.85, `confusable_misattribution_rate` ≤ 0.05, `lossrun_totals_reconciliation_rate` 0.80, `table_f1` 0.88, `auto_accept_error_rate` ≤ 0.02, `worst_line_field_match` 0.85. `PILOT_FLOORS` is also the one source of the pilot's thresholds: `pilot.pilot_run` builds its field match, list recall, schema validity, LoB detection and confusable-rate criteria from it and `GATING_METRICS`, with the same inclusive comparison (IMPL-15).
  2. **Non-inferiority** against the tenant's promoted release: the 95% CI lower bound of (candidate − current) from a paired bootstrap over documents above −δ, with a per-metric δ (`NON_INFERIORITY_DELTA`, default 0.015). The gate stage supplies `per_document`: `orchestration.pipeline_dag.paired_document_scores` reads this tenant's promoted release's eval report — the release marked `promoted` in the tenant's release index (`pipeline_dag.promoted_version`, decision 2026-10-22; it used to read a run-manifest status `promoted` that no real run ever set, so every candidate was gated as a first version) — and the candidate's, and pairs each metric's per-document scores, with their weights, on the documents both scored (a mismatch is a warning). It pairs only when both reports are on one yardstick — the same frozen set of this tenant, scored by the same code (`eval_set`: tenant, `frozen_at`, `frozen_from_corpus`, `scoring_fingerprint`; `run_eval.yardstick_difference`): source ids are numbered per tenant and reused when a set is frozen again, and the report path names no tenant. A metric with no per-document scores gets a point comparison against δ, recorded as `basis: point_estimate` and weaker evidence. So does a metric fewer than two paired documents measure, with a warning naming it: one document would widen the interval to the whole range and block every later release. **A baseline on another yardstick is not compared at all** (decision 2026-10-22) — another frozen set, another tenant, other scoring code (`scoring_fingerprint`), or a report that does not record its yardstick; also a promoted report scored on other lines than the candidate serves (`lines_scored`), or without its calibrators (`calibrated: false`, as an ad-hoc `run_eval` writes it to the same key: every field flagged, so `auto_accept_error_rate` reads 0.0): the candidate is judged on the floors alone, the improvement rule is not applied, it is not called a first release, and the reason is a warning (`promotion_gate(baseline_not_comparable=)`). A promoted `release-YYYY.M.N` row whose adapter is not a `policy-vN` run is no baseline at all (a warning): the policy adapter's report at that number would be another model's. Every report written before the fingerprint lacks it, so the first candidate after it is judged on floors once. On one yardstick, every metric gets a point comparison when the reports cannot be paired — no shared documents, or no per-document scores — and the reason is a warning in `GateResult.warnings` and the decision. When either report lacks weights for a metric, both sides are paired unweighted, with a warning; `paired_bootstrap` refuses weights given for one side only.
  3. **Improvement**, enforced: at least one primary metric (`field_normalized_match`, `list_field_recall`) improves with its CI lower bound above zero, or the release is blocked on `improvement`. Naming the defect the release fixes (`--fixes-defect "<ticket>"` on `package` or `all`, `StageContext.fixes_defect`, recorded in the decision) stands in for the improvement and nothing else: floors and non-inferiority still apply. The value is checked (`gating.documented_defect`): a blank or placeholder one (`" "`, `-`, `n/a`, `fix`) is refused, and a ticket id such as `TICKET-12` or a sentence is kept. It is one value (no `scope=TICKET` pairs). The first release is exempt. When neither primary metric has paired per-document scores, the rule cannot be evaluated and is not enforced, and the gate says so in its warnings (`GateResult.warnings`, also in the decision).
- A passing release is tiered by `release_tier`: **production** when `field_normalized_match` ≥ `PRODUCTION_FIELD_MATCH` (0.92, kept by the decision of 2026-10-19), **interim** below that; a blocked release has no tier. **Too few eval documents cap the tier**: a release scored on fewer than `MIN_FROZEN_DOCS_PER_TYPE` (150) real frozen policies is passed as `promotion_gate(small_eval_set=True)`, which tiers a passing release `interim` whatever its score and adds a warning (`GateResult.warnings`, also in the decision). The count is the **real** documents the candidate was scored on (decision 2026-10-22: synthetic twins do not count; `pipeline_dag.eval_set_size`), so only those of the lines it serves: the candidate report's own `eval_set` `real_documents_by_doc_type`, else the frozen manifest's `real_documents_by_doc_type`, of the adapter's types. A manifest frozen before those were recorded counts as unknown, so the release is interim, with a warning saying why. The whole set's `small_set_accepted` does not decide it: a policy release scored on 200 policies is not capped because the set holds 9 Loss Runs. No frozen policies counts as too few, and so does a count nothing records.

**Gating metric set** (`GATING_METRICS`): field exact/normalized match, list-field F1 (`field_f1_list_fields`), **list-field recall**, schema validity rate, ECE, OCR-arbitration accuracy, image-only accuracy, scanned accuracy, doc-type classifier accuracy, **LoB detection accuracy**, **confusable misattribution rate**, false-null rate, Loss Run totals reconciliation rate, `table_f1`, `auto_accept_error_rate`, `worst_line_field_match`.

- **Not applicable vs not measured.** `not_applicable_for()` excuses what the adapter cannot produce (`common.adapter.not_applicable_metrics()`, derived from what it trains, never declared: no Loss Runs → no reconciliation or `table_f1`; every policy's line comes from metadata → no LoB detection metric; no classify rows built → no doc-type classifier metric; no list field in any policy schema → no list metrics) plus `CONDITIONAL_METRICS` (`lossrun_totals_reconciliation_rate`, `table_f1`, `auto_accept_error_rate`, `worst_line_field_match`, `scanned_accuracy`), which only exist when the eval set holds the relevant documents. A conditional metric the baseline measured is required. Anything else unmeasured blocks.
- `PENDING_GATING_METRICS` — `list_field_precision`, `hallucination_rate` — are reported, and not enforced until their producers are complete. A metric moves into `GATING_METRICS` in the same commit as the code that computes it. (`page_selection_recall` is gone: every page is read, so there is no selection to measure.)
- **No ceiling on invented values is gated** — an open item. `hallucination_rate` is reported where rows carry their OCR text, no spec states a ceiling, and the golden eval does not carry the pages' text in its metadata, so on the gate path it is not measured.

`alias_accuracy` and `held_out_carrier_match` are **reported alongside but not gated** — see the note above.

A frozen document is an unseen carrier's when the split held its carrier out or the delivery marked it
(`label_metadata.json` `held_out_carrier`; `evaluation.freeze_eval_set.held_out_carrier`); the manifest lists
them as `held_out_source_ids`. Each frozen document's `carrier` is the corpus build's key (`document_carrier`),
and the carrier of each document the delivery marked is added to `held_out_carriers_by_line[doc_type][line]` (a
list when a line holds out more than one; placeholders and documents with no line skipped), so a later build
warns when it trains. `mark_held_out` (`freeze-eval-set --mark-held-out`) marks a set frozen before the importer
carried the flag, and records those carriers; nothing else changes. Which carriers a drawn split holds out is IMPL-05's rule: one
per line of business, only in a line whose real documents name 10 or more carriers
(`split_groups.MIN_CARRIERS_PER_LINE_FOR_HELD_OUT`). A set already frozen keeps the carriers its split held out:
a frozen set draws no new test documents, and splits made after freezing hold no carrier out.

- **A continued run is held, line by line, to the runs before it** (decision 2026-10-23). This replaces the old "cross-type regression evidence" rule of the multi-type Foundation design; `StageContext.cross_type_evidence` and the `cross_type:<doc_type>` gates are gone. Continued training compounds drift: a line that slips a little every cycle never fails one comparison with the promoted release, or with its parent. So each line the candidate serves is held to the **best run on it since the last run from the base** (`_require_line_non_regression`):
  - **The reference.** The chain is walked from the parent by `parent_run_id` (IMPL-06) down to the run that started from the base. The best run is the one with the highest weighted field match on the line's real documents that both reports scored, with at least 5 such documents (`gating.LINE_MIN_DOCUMENTS`, the same number as `split_groups.MIN_DOCS_TO_MEASURE_LINE`). On a tie, the nearer run wins.
  - **The comparison.** A paired bootstrap over those shared documents. The line passes when the lower bound of the 95% interval of (candidate − reference) is above −`DEFAULT_DELTA` (0.015; `ConfidenceInterval.non_inferior` reads the bound rounded to six places); otherwise it blocks as `line:<line>`.
  - **Warnings, not blocks.** A line with fewer than 5 shared real documents is "thin" and not compared, and a new line has no earlier run to compare with. Each gets a warning, and so does a run in which no line could be compared.
  - **A dropped line blocks** as `line:<line>`: a line the parent's release served (its report's `lines_scored`) that the candidate no longer serves, whose policies would go to the base model.
  - **No comparable parent report blocks** as `line_regression_evidence`: the parent's report is missing, on another yardstick, or has no `per_document_lines` (written before reports recorded them); so does a candidate report without them. Other runs on the chain with no usable report are left out with a warning, and the chain stops, with a warning, at a run whose manifest cannot be read. The remedy is to train the next cycle from the base (`--fresh`, without `--version`), or to keep the parent's report on the frozen set.
  - **A small line** has a wide interval, so it may block until it has more documents. It can be waived as `line:<line>` (a written override, below).

  The gate stage builds the comparison (`pipeline_dag.line_evidence`, a `gating.LineEvidence` of `LineComparison`s) and passes `continued_from`, `line_evidence` and `line_evidence_missing`. `GateResult.line_verdicts` holds one entry per compared line: `line`, `held_to`, `reference`, `candidate`, `delta`, `non_inferior` and `interval`. A run continued by hand from a directory (`training.train --continue-from`) names no parent run, so it is blocked at `line_regression_evidence`. The usual comparison with the promoted release still applies.
- Writes the decision, tier, per-metric verdicts, metrics gated and `gated_against` into the candidate's `RunManifest` (IMPL-02, `apply_to_manifest`); a run recorded as failed is not marked evaluated. The gate decision the stage writes (`paths.gate_decision`, and under the release) also records `paired_documents` (how many documents each bootstrapped metric paired; a metric compared any other way is not listed), `baseline_not_comparable` and `baseline_metrics_not_compared` (why the baseline was on another yardstick, and its metrics; `baseline_metrics` is then null), `fixes_defect`, `eval_set` (`small_set_accepted`, documents and real documents per type, the corpus it was frozen from and when; the real documents per type the candidate was scored on, where that count was read, the minimum, and whether it was too small: `scored_documents_by_doc_type`, `scored_documents_counted_from`, `minimum_per_doc_type`, `small`), `adapter`, `doc_types`, `warnings`, `blocking_reasons` and `line_comparison`. For a continued run `line_comparison` is `{continued, runs_compared, lines, not_compared, dropped, added}` (`lines` the line verdicts, `not_compared` each thin line with the most documents it shared), or `{continued, missing}` when no comparison could be made; for a run from the base it is `null`.
- **No bypass flag.** A failed gate is a failed gate unless a written, attributed override is recorded with it (`GateOverrideRecord`: a named person, not a system identity; a reason of at least 20 characters; the gates it waives). It is given on `package` and `all` with `--override-approver`, `--override-reason` and `--override-gate` (repeatable): all three together, checked before any work by the record's own `validate()` (`run.gate_override_from`). It lifts only the named gates that actually failed, and they are recorded as waived, never as passed. The gates it names must be real (`GateOverrideRecord.validate`, through `gating.is_gate`: a gating metric, `improvement`, `line_regression_evidence` or `line:<line>`), and an override that waived nothing is a warning in the decision. There is no `--force`.

## Constraints
- Golden eval set is frozen + versioned separately; never train on it — assert it.
- Metrics deterministic + unit-testable on synthetic inputs.
- All model runs go through IMPL-07 — no bespoke inference here.
- `common/normalize.py` (through `values_agree`) is the single definition of "correct" shared with IMPL-12, and `common/grounding.py` the single definition of "printed".
- No PII in eval reports beyond what access control permits (master §8) — scrub or restrict.

## Acceptance checklist
- [ ] Each metric returns correct values on hand-constructed synthetic cases.
- [ ] Normalized match handles dates, currency, and entity names via `common.normalize`, and numbers by the schema's declared type.
- [ ] `list_field_recall` flags a deliberately dropped claim row.
- [ ] `lob_detection_accuracy` / `lob_accuracy_by_value` report per-LoB-value accuracy and do not fold LoB into the aggregate field score.
- [ ] `alias_accuracy` slices field accuracy by observed surface label from `field_provenance`, and a field scoring well on its dominant alias but poorly on a rare one is visible in the report.
- [ ] `confusable_misattribution` scores a prediction that returns the certificate holder's value as `insured_name` as a failure, distinctly from an ordinary wrong value.
- [ ] A regression on confusable misattribution **alone** blocks promotion.
- [ ] `classify_error` separates perception errors from schema/reasoning errors on seeded inputs.
- [ ] `run_eval` produces a per-doc-type report with mode breakdowns (`image_only`, `scanned`, `noisy_ocr`, `long_policy`), using IMPL-07.
- [ ] An eval `source_id` appearing in a corpus train or val split causes a loud failure.
- [ ] The freeze refuses a gold with no review record before copying anything and carries the record into the frozen metadata; the golden eval refuses a frozen gold without one; `mark_reviewed` copies records into an older set only where the frozen gold is still the current one, and changes nothing else; a delivered held-out carrier is recorded by line.
- [ ] `gating` blocks a candidate that fails **any single** gate metric's floor or non-inferiority test — including LoB accuracy alone.
- [ ] The eval report keeps each document's score and weight (`per_document`, `per_document_weights`) and the frozen set it was scored on (`eval_set`); the weighted mean of the scores is the reported metric. With both reports on the same yardstick (frozen set, tenant and scoring code), the gate judges non-inferiority by a paired bootstrap (`basis: paired_bootstrap`) and records `paired_documents`. A baseline on another yardstick is judged on the floors alone, with a warning; on one yardstick, reports that cannot be paired, and a metric fewer than two paired documents measure, are compared on point estimates, with the reason in the warnings.
- [ ] A release whose primary metrics did not improve is blocked on `improvement`, unless `--fixes-defect` names the defect it fixes (one value; a blank or placeholder value is refused), which is recorded in the decision. An improvement rule left unenforced is a warning in the decision.
- [ ] The gate scores only the frozen policies whose single line the release serves, and the report records `lines_scored`; a report scored on other lines is not reused; a baseline scored on other lines, or without its calibrators, is judged on floors alone; a promoted legacy release of another model is no baseline.
- [ ] A release scored on fewer than 150 real frozen policies (twins not counted) caps a passing release at `interim`, with a warning in `GateResult.warnings` and the decision, which records the count; the whole set's `small_set_accepted` does not decide it.
- [ ] A merged common-model answer's schema validity is judged as serving returns it (`served_schema_verdict`), not failed for the system fields the pipeline adds.
- [ ] A continued run holds each line it serves to the best run on it since the last run from the base, on at least 5 real documents both reports scored (the nearer run on a tie). A line whose paired 95% lower bound is not above −0.015 is blocked at `line:<line>`, and so is a line the parent's release served and the candidate dropped; a thin or new line only warns; no comparable parent report, or a run continued by hand from a directory, is blocked at `line_regression_evidence`. The report records `per_document_lines`, the full-set report `field_scored_by_lob`, and the decision `line_comparison` and `blocking_reasons`.
- [ ] No code path waives a failed gate except a validated, attributed `GateOverrideRecord`, recorded as a waiver; the `--override-*` flags build that record and nothing else.
- [ ] Promotion decision + per-metric verdicts written to the RunManifest.

---

## Current implementation (2026-10-21)

**The frozen golden eval set** (`evaluation/freeze_eval_set.py`,
`python -m orchestration.run freeze-eval-set --corpus vN [--allow-small]`):
- copies one corpus build's **test split** — golden label, metadata, every page image and OCR page — into
  `golden-eval-set/{tenant}/{source_id}/`, with `golden-eval-set/{tenant}/manifest.json` (source corpus, commit,
  documents per type and per line of business, held-out carriers overall and per line). Each document's metadata
  records its carrier, whether that carrier was held out of train and val, and its review record (`reviewer_id`,
  `review_date`, `draft_backend`, `review_inherited_from`). `golden.json` is written last per
  document, so an interrupted freeze leaves nothing half-copied;
- **one set per tenant**: each tenant's releases are gated on its own documents, and source ids are numbered per
  tenant. A set still at the root (`golden-eval-set/`, frozen before sets were per tenant) is refused, never
  ignored (`refuse_unscoped_set`);
- **refused a second time** (the set is the yardstick every version is compared on), and **refused when
  any type would freeze fewer than 150 real documents** (`MIN_FROZEN_DOCS_PER_TYPE`, arch §15.4; synthetic
  twins do not count, decision 2026-10-22) unless
  `--allow-small`: the set cannot grow once frozen, and at 150 documents a rate near 80% is known to about ±3
  points. A set accepted small records `small_set_accepted` in its manifest, with `real_documents_by_doc_type` and
  `real_policies_by_line`. The gate caps at `interim` a release scored on fewer than 150 real frozen
  policies, whatever that flag says (§4);
- after freezing, every corpus build excludes the frozen documents and their families and splits new
  documents into train/val only (IMPL-05);
- **refused, before anything is copied, while any gold has no review record** (`review_problem`, decision
  2026-10-22): a gold nobody reviewed is never frozen or scored. A set frozen before the freeze carried the
  record gets it with `freeze-eval-set --tenant T --mark-reviewed` (`mark_reviewed`): only the review keys
  change, copied from each document's label metadata only while `golden-labels/` (read under the edition rule)
  still holds the frozen gold. A gold changed or gone since the freeze is listed ("gold changed since the
  freeze … delete the set and freeze again"), since its record is the new gold's; the command exits 1 while
  any frozen gold still has none.

**The golden eval** (`evaluation/golden_eval.py::evaluate_version`): refuses a partial set and asserts the set
is disjoint from the corpus; refuses, never narrows, a set holding a frozen gold with no review record in its
frozen metadata (`load_golden_set`, decision 2026-10-22: skipping one would score a smaller yardstick than
earlier versions were scored on) — checked with an interrupted freeze before the model loads
(`assert_scoreable`), and never retried: `GoldenEvalError`, `EvalError`, `FreezeError` and `GateError` are
deterministic; loads every one of the tenant's frozen documents of the adapter's doc types (policies) and,
given `lines=` (the gate's: the lines the release serves), keeps only the policies whose single line is one of
them — `None`, as `run_eval`'s command line passes, keeps every frozen document; none left is refused ("lines
this release serves"); each label passed through label verification (a rule-added value no page prints is not
expected of the model, as in its training targets) and the form-edition label rule (`verified_label`, through the corpus build's own
`label_verification.verified_label` with the document's `doc_type` and line: on a common-model policy line, where
the label rules hold (`label_rules.label_rules_apply`), an edition printed inside a form number moves to the
form's empty `edition_date`, as training labels and serving's `conform` have it; the frozen gold in Blob is read
under the rule, not rewritten; a self-contained line's numbers are scored as stored), and classic auto read as personal auto; runs every document
in **all three modes** (`GOLDEN_MODES`; `noisy_ocr_image` with the corpus build's own seeded corruption)
through `serving.pipeline.extract`, by the base model with the release's adapter applied (as served; decision 2026-10-23), with the release's bf16 calibrators; scores failures as `{}` with the error
recorded; and writes the report where the gate reads it. A common-model policy carries serving's own schema
verdict (`schema_valid`, from `serving.pipeline.schema_verdict`), which `schema_validity_rate` reads. The report adds `documents_failed`, `lines_scored` (the lines it was scored on, `None` for every frozen document), `adapter_checkpoint` (the checkpoint scored: the selected one, else the staged adapter directory; `pipeline_dag.adapter_checkpoint`), `calibrated`
(a report made without calibrators is not reused once the release has them, nor one whose `eval_set` is not this tenant's, nor one scored on other lines than the release serves, nor one made from another checkpoint than the one selected now — `pipeline_dag.eval_report_metrics`; a dry run says "not reused" and why),
`release_measurements` (P95 latency per adapter, peak GPU memory sampled through the eval; recorded, not gated)
and `windowing_ceiling` (what the oracle merge of the gold loses before any model runs, over the same windows on
every page — `policy_windows.pages_to_read`; reported, never gated), `per_document`, `per_document_weights` and `per_document_lines`, and `eval_set` (§3). The gate stage
produces the report when none exists. Since every page is read (decision 2026-10-19), a policy is read in more
windows than before: the corpus must be rebuilt and the adapter retrained, and gate numbers from before the
change are not comparable with new ones.

**Validation-based scoring** (checkpoint selection, calibration): `evaluation/validation_generation.py`.
- `generate_validation(rows, model, *, adapter, constrain, batch_rows, stats)` generates every row once with the
  row's own prompt (assistant turn removed), constrained as serving is (`model.config.structured_outputs`) to
  the row's schema slice with `page_ref` bounded to the pages it shows (`with_page_bounds`), and capped at the
  row's task answer budget (`common.config.answer_cap`). Rows go `configured_batch_rows()` per `generate_batch`
  call — `generation.validation_batch_rows` (96) in `configs/inference/vllm_serving.yaml`, else `BATCH_ROWS`
  (32) — **longest expected answer first within a call**, so the long tail runs beside the short answers. A
  call that fails together (one unreadable page image refuses the batch) is **retried in halves** until the bad
  row is alone. `stats` is filled with rows, rows generated, rows per call, calls, seconds and the slowest call.
- Answers are parsed as serving parses them (`collapse_spans(map_field_spans(...))`, `with_output_dates`). A
  row with nothing usable back is a `setup` failure; an answer that is not a JSON object is an `output` failure,
  scored as a wrong answer, with how generation ended recorded.
- `assert_generations_usable` refuses a pass in which more than **10%** of rows had a setup failure
  (`MAX_GENERATION_FAILURE_RATE`), or more than **50%** came back as unusable JSON (`MAX_UNUSABLE_OUTPUT_RATE`) —
  scoring a broken setup turned it into a checkpoint "chosen" at 0.0 and a calibrator fitted on nothing, while a
  checkpoint that loops on some documents has to lose, not be dropped.
- `score_generations` is the gate's `build_report` over the rows, window by window, plus the same metrics over
  **whole documents** (`document_*`): each policy's windows merged per reading mode by the serving merge
  (`merged_windows` → `serving.policy_merge.merge_policy_windows`), its window targets into the gold and its
  generations into the answer, so windowing loss is out of both and the score reads the model alone. A whole
  common-model document's schema validity is judged as serving would return it
  (`serving.pipeline.served_schema_verdict`: enveloped, system fields and every key filled, then the client's
  schema), so `document_schema_validity_rate` no longer reads a false 0% for what the pipeline adds.
- Images are localised to the pod cache first (vLLM opens paths, the rows hold Blob keys).

**Checkpoint selection** (`evaluation/checkpoint_eval.py`, run by `orchestration.pipeline_dag.stage_checkpoint_eval`;
settings in the training config's `evaluation` block):
- **Candidates**: `discover_checkpoints` finds the latest run's `checkpoint-*` directories and the best-loss one
  (`trainer_state.json`'s `best_model_checkpoint`); when `eval_steps` exceeds `save_steps` only the checkpoints an
  evaluation fell on, and the last, are kept. `select_candidates` takes the last `TRAILING_CANDIDATES` (3) by
  step plus the best-loss one.
- **Ranking on the validation sample**: each candidate is applied as a LoRA per request on one base engine
  (`vllm_scorer`, `load_model("base")`) and scored on `validation_sample_rows` (400) rows of the adapter's
  validation split — `evaluation/validation_sample.py`, stratified by line of business and reading mode (at
  least one row per group), deterministic by a hash of each row's identity, the whole split when it is no
  larger. `select_best` picks the best `field_normalized_match`, ties to the later step; a candidate whose
  scoring raises is skipped and recorded, not scored 0; loss/F1 disagreement and a margin under 0.005 are
  logged.
- **Whole documents decide** (`choose_on_documents`): the best `selection_document_finalists` (2) are scored on
  the full split's `.full` pass and judged on `document_field_normalized_match`, among those whose
  `document_hallucination_rate` and `document_false_null_rate` are within `selection_guard_margin` (0.02) of the
  best finalist's; ties go to the later step. `selection_document_modes` (`[ocr_plus_image]`) narrows that pass
  to those reading modes — **by mode, never by row** (`rows_in_modes`), so every document stays whole; the
  sample still ranks in all three modes, and the golden gate scores all three. A mode filter matching no row is
  refused before the engine loads (an unknown mode name by `_selection_settings`), as a `PipelineError` that is
  not retried. `.full` exists whenever the sample is not the whole split; without it the finalists are judged on
  the sample's own scores. `document_choice` records the finalists' metrics, the guard margin, the modes read
  (or `"sample (all modes)"` when no whole-document pass ran), those excluded by a guard, and whether the choice
  changed.
- **Near-tie**: with fewer than two finalists configured, `break_tie` re-scores the top two on `.full` when they
  are within `selection_tie_break_margin` (0.015) on the sample.
- **Several GPUs**: with more than one of the GPUs training used, `ParallelScorer` (`parallel_vllm_scorer`) deals
  the candidates to one worker process and vLLM engine per GPU (`python -m evaluation.checkpoint_score_worker`,
  `--full` and `--full-mode` for the whole-document pass), each building the same in-process scorer; the parent
  fetches the page images once, a failed worker fails only its candidates (with the end of its log), and each
  worker sends back what each generation pass cost.
- **Cost**: `generation_cost(scorer)` totals the passes (sample and `.full`: rows, calls, seconds) into the stage
  result; the selection report is written to `paths.checkpoint_selection(version, scope=ADAPTER.name)`.

**Scoring corrections** (`evaluation/metrics/`):
- a value the golden label does not state is **scored wrong** (canonical targets are sparse, so a loop over
  golden paths alone never saw an invented field); empty extra values cost nothing;
- fields compare by their **declared type** (`values_agree`), so `957` and `957.0` agree and an identifier read
  as `1.0` is `1`;
- **tables are found at any depth** (`auto.vehicles`) and rows compared on **values**, not envelopes; rows are
  **paired by identity**, never by position (`rows_aligned_to`, and for common-model lines `aligned_for_scoring`:
  passes from the strictest identity down — a coverage by its printed name and what it applies to before its
  code — then shared values), so one omitted or reordered row does not make every later row wrong;
- **common-model lines** (`metrics/common_model.py`): both sides go through `comparable_view`, so structural ids
  are never scored and links are compared by what they name; values inside rows count; links, codes and
  `additional_fields` are scored apart; schema validity is asked of the answer as it was produced, never of the
  scored copy: one window against its slice's model view, a whole document as serving returns it (serving's own
  verdict on the gate path, `served_schema_verdict` where no verdict exists);
- **hallucination** looks for the value as printed (`raw`) on the pages it cites (`common.grounding`); printed
  only on another page is a wrong page, not an invention; a value too short to look for is left out; a right
  value is never a hallucination; only rows that were sent text count;
- **false nulls** pair an answer row nothing paired with the next unpaired label row (`fill_unpaired`), so a row
  the model wrote but misread is wrong values, never nulls; codes and links are neither false nulls nor
  hallucinations;
- **misattribution** is checked on flattened nested paths;
- **LoB** is not scored for a label that carries none (canonical policies), and the LoB metric is not
  applicable to the policy adapter;
- field accuracy is **pooled per document reading** (source × mode), then averaged over documents — a
  60-window policy counts once, not 60 times; a window with nothing to score adds nothing;
- `worst_line_field_match` counts a common-model line only once it has `WORST_LINE_MIN_VALUES` (200) values
  scored;
- the classifier metric is not applicable while the corpus builds no classify rows (`common.tasks.CORPUS_TASKS`).

**The gate scores the lines the release serves** (code review of 2026-10-23). There are no line scopes
(decision 2026-10-22), but serving sends every other policy — another line, a package, a policy with no line —
to the base model, so scoring them on the adapter would measure something nobody is served. The gate stage
passes `lines=served_lines(ctx)` (the lines the release trained on, `pipeline_dag.trained_lines`) and
`evaluate_version` scores only the frozen policies whose single line is one of them (`common.lob.single_line`).
A run with no trained line fails the gate step before the model loads. Double annotation of the frozen
documents (arch §15.4) remains a manual step before the first production gate.

**Freeze guard covers every corpus type**: a document type the corpus holds but the test split drew none
of counts as 0 and blocks the freeze (it used to pass unchecked and be missing from the frozen set for good).

**Leakage check reads train and val, not test** (`run_eval.corpus_source_ids`): the frozen set is compared
with the splits a model learns from and is selected on. The test split is what it is frozen FROM, so the
sequence *build v1 → freeze v1's test → gate the model trained on v1* is valid; counting test made it fail as
leakage for every frozen document. The family split keeps a frozen document's relatives out of train and val.

**Frozen = the manifest exists.** An interrupted freeze (documents copied, no manifest) is not frozen: it can be
resumed from the same corpus, a partial set from another corpus is refused with instructions, and the golden
eval refuses to gate on a partial set.

**Line detection is measured apart** (`evaluation/lob_detection_eval.py`, `scripts/measure_lob_detection.py`,
`metrics/lob_detection.py`): each frozen policy with one known line is read as serving would read it with no
line, and resolved by serving's own `_resolve_lob` at its thresholds; reported per line against
`DETECTION_FLOOR` (0.85, per line from `PER_LINE_MIN_SUPPORT` = 20 documents), with the confidently-wrong rate,
scanned/native and held-out/seen carriers apart, and a threshold sweep. Not a release gate (the detector is the
base model, which a release cannot move); `routing.detect_lob` stays off until it has been run.

### Real documents scored apart (2026-09-29)

The frozen set records each document's `synthetic` flag (`freeze_eval_set`), the golden eval carries it into
every scored document's metadata, and when the set mixes real and synthetic documents the eval report adds
`composition` (real and synthetic document counts) and `real_only.gate_metrics` — the same report over the real
documents alone (`golden_eval.real_only_section`). Reported beside the gate, never instead of it: the synthetic
twins of held-out sources are unseen layouts and a fair test, but their labels came from a generator, and the
gap between the two scores is how much it flatters the model. A set with no synthetic documents reports nothing
extra.

## Changes since 2026-10-05

- §1: added typed comparison (`values_agree`, `scoring_kind`, numbers as numbers, identifiers) and the shared grounding matcher (`common/grounding.py`).
- §2 table: latency row now points at `release_measurements`; alias row corrected (`score_alias_accuracy` uncalled, seen/unseen split reported); `score_by_mode` replaced by `classify_error`; added page-ref, common-model, `table_f1`, auto-accept and precision/recall/error-summary rows; error classes include omission and invented.
- §3: report path is the scoped `summary.json` written by the golden eval; added the `held_out_carrier` subset, error totals, per-tenant frozen set, train/val-only leakage check.
- §4: replaced the stale "match or beat on every metric" gate with floors + paired-bootstrap non-inferiority + improvement, release tiers (0.92), the current gating/conditional/pending metric sets and the attributed override; checklist items corrected to match.
- Current implementation: frozen set is per tenant with carrier fields; golden eval details (label verification, served schema verdict, `calibrated`, release measurements, windowing ceiling); validation generation (96 rows per call, longest first, halves retry, stats, page bounds, answer caps, two failure rates, document metrics); new checkpoint-selection section (candidates, 400-row stratified sample, two finalists on whole documents with guards, `selection_document_modes`, `.full`, `document_choice`, `break_tie`, `ParallelScorer`, `generation_cost`); scoring corrections for typed comparison, identity pairing, common-model scoring, page-aware hallucinations, false nulls; line-detection measurement.
- Frozen held-out documents: the delivery's mark, `held_out_source_ids`, `mark_held_out` (commit f5668a4).

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference is `finetuning-architecture-v2.4.docx`.
- §2 table: schema validity judges a whole common-model document as serving returns it (`served_schema_verdict`); latency is recorded, not gated (no SLA); `score_page_selection` and page-selection recall removed.
- §3: serving reads every page; the eval report keeps per-document scores (`per_document`, `EvalReport.per_document_scores()`).
- §4: non-inferiority is a paired bootstrap on per-document scores the gate stage pairs (`paired_document_scores`), with a point comparison and warning for a report written before they were kept; the improvement rule is enforced, `--fixes-defect` stands in for it; a set accepted small caps a release at `interim` with a warning; production tier stays 0.92; `PILOT_FLOORS` is the pilot's one source; `page_selection_recall` gone from the conditional and pending sets; no invented-value ceiling (open item); the per-line carrier rule and the frozen set's held-out carriers; what the gate decision records.
- Acceptance checklist: per-document scores and the paired bootstrap, the enforced improvement rule and `--fixes-defect`, the small-set cap, the served schema verdict.
- Current implementation: `small_set_accepted`; frozen golds read under the form-edition rule (`split_form_editions`); serving's verdict from `schema_verdict`; the windowing ceiling over every page and old gate numbers not comparable; checkpoint selection's whole-document schema validity as served.

## Changes of 2026-10-22 (code review)

- §3: the eval report keeps each document's weight (`per_document_weights`) and the frozen set it was scored on (`eval_set`).
- §4: the paired bootstrap is weighted, so it tests the metric the report states; it pairs only reports scored on the same frozen set of the tenant, and every fallback to a point comparison is a warning in the decision; a metric fewer than two paired documents measure is a point estimate, with a warning; an unenforced improvement rule is a warning; `--fixes-defect` is checked and given per scope; the interim cap follows what the scope was scored on (`scope_eval_set_size`); the written override on the command line (`--override-approver`, `--override-reason`, `--override-gate`); the decision's new `eval_set` fields.
- Acceptance checklist and *Current implementation*: the same.
- The freeze refuses a gold with no review record (decision 2026-10-22) before copying anything, and carries `reviewer_id`, `review_date`, `draft_backend` and `review_inherited_from` into the frozen metadata; `freeze-eval-set --mark-reviewed` (`mark_reviewed`) gives an older set its records and exits 1 while any is missing.
- The golden eval refuses a frozen set holding a gold with no review record, never skipping it: a set frozen before 2026-10-22 is marked or frozen again before the gate scores it.
- Frozen golds are read under the form-edition rule only on common-model policy lines (`label_rules_apply`), through the corpus build's own `label_verification.verified_label`.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.
- `mark_reviewed` copies a record only while the frozen gold equals the current one; the freeze and `mark_held_out` record a delivered held-out carrier by line, under the build's carrier key.
- A baseline on another yardstick (frozen set, tenant, scoring code, or unrecorded) is judged on floors alone, with a warning; `eval_set` records `scoring_fingerprint` and `real_documents_by_doc_type`; missing weights mean both sides unweighted.
- The production minimum and the freeze's minimum count real documents only (decision 2026-10-22); an older manifest counts as unknown.
- `paired_documents` lists only bootstrapped metrics; override gate names are checked; review records are checked before the model loads and the gate's own errors are not retried; a candidate report is reused only when it is this tenant's.

## Changes of 2026-10-22 (one adapter)

- One adapter, no scopes (decision 2026-10-22): the baseline is the tenant's promoted release in the release index (`promoted_version`; before, every candidate was gated as a first version); `promotion_gate` takes no scope, floors are `PILOT_FLOORS` (`floors_for` gone), the not-applicable metrics come from `common.adapter.not_applicable_metrics()`; the interim cap counts the real frozen policies scored (`eval_set_size`; decision keys `scored_documents_by_doc_type`, `scored_documents_counted_from`, `small`); `--fixes-defect` and the override are one value per run; every frozen policy is gated (the line-scoped gate is gone); `run_eval` has no `--scope`.
- Code review of 2026-10-23: the gate scores only the frozen policies of the lines the release serves (`evaluate_version(lines=)`, the report's `lines_scored`), replacing "every frozen policy is gated"; a report or baseline scored on other lines is not reused or compared; a baseline report scored without calibrators is not compared; a promoted legacy release of another model is no baseline; a run with no trained line fails the gate before the model loads. New acceptance item.

## Changes of 2026-10-23 (incremental training)

- §4: the cross-type regression rule, `cross_type_evidence` and the `cross_type:<doc_type>` gates are gone; a continued run (decision 2026-10-23) is held line by line to the runs before it — `line:<line>` blocks a line that lost ground, `line_regression_evidence` blocks when no comparable parent report exists (the rule as built: the code review bullet below). `promotion_gate` takes `line_evidence` and `line_evidence_missing`; override gate names follow.
- §3 and *Current implementation*: a version is scored as served, the base with its adapter applied; the full-set report records `field_scored_by_lob`. Acceptance item updated.
- Code review of the incremental training (§3, §4, acceptance, *Current implementation*): each line is held to the best run on it since the last run from the base, by a paired bootstrap on at least 5 shared real documents (`LINE_MIN_DOCUMENTS`), where it was compared with the parent alone on 200 values (`LINE_MIN_VALUES` is gone); a dropped line blocks; the report records `per_document_lines` and `adapter_checkpoint`, and a report of another checkpoint is scored again; the gate result has `line_verdicts`, and the decision `line_comparison` and `blocking_reasons`.

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.
