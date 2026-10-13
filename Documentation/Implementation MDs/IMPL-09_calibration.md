# SPEC 09 — Confidence Calibration

> Aligned with the code at commit f5668a4 (2026-10-13).

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-07, IMPL-08. Uses the inference core (IMPL-07) for logprobs/spans and the evaluation normalizer (IMPL-08) for the correctness signal. No forward dependency on serving/testing.
>
> **Architecture refs:** `finetuning-architecture-v2.3.docx` §5 (**confidence flow, calibration, list-field second signal**), §0b (LoB carries confidence too), §15 (ECE as a gating metric).

## Goal

Turn raw token logprobs into trustworthy per-field confidence via post-hoc calibration, plus the list-field completeness signal that logprobs are structurally blind to.

## The end-to-end flow being implemented (arch §5)

```
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
```

(This is v1's flow. Under arch v2.1 the "aggregate per field" and "calibration transform" steps are a feature vector per field and a per-field-type calibrator, and the threshold is fitted per field type — see *Current implementation*.)

**Why calibration is not optional:** fine-tuned generative models — especially after LoRA fine-tuning on a narrow task — become **overconfident**. Correct and incorrect outputs carry similarly high token probabilities, because the model has learned to be fluent in the target format even when it is wrong about the content. Raw logprobs alone are not a usable confidence signal.

**Why not a separate verifier model:** another model to train, version, evaluate, and keep in sync with every Foundation/adapter version, plus doubled inference latency. It stays a documented escalation path **if** calibrated logprobs prove insufficient in evaluation (IMPL-08 ECE) — start with the cheap, well-understood approach.

## Deliverables

### 1. `calibration/logprob_confidence.py`
- Consumes field spans + logprobs from IMPL-07 `span_map` and aggregates to a raw per-field confidence.
- **Superseded as the confidence itself (arch v2.1 §5.1).** The minimum token probability is length-biased — the minimum of *n* draws falls as *n* grows, so a long correct policy number scores below a short wrong year. It is now **one feature among several** handed to a per-field-type calibrator; see *Current implementation* below. Serving reads this raw aggregate only when it holds no calibrator set, and then flags every field.
- Handles nested fields and list fields (per-value confidence within each row).
- **`line_of_business` gets confidence like any other field** (arch §0b) and is emitted in the same `{value, confidence}` shape.
- Costs nothing extra at training or inference time — the logprobs are already there.

### 2. `calibration/fit_calibration.py`
- On a **held-out labeled validation set — never training data** — compare raw confidence against actual field-level correctness (exact / normalized match via `common.normalize` from IMPL-08) and fit a transform:
  - **`temperature` scaling** (default: simple, one parameter, works well here)
  - **`isotonic` regression** (more flexible; handles non-monotonic miscalibration)
- Fit **per doc_type**, optionally per field-type. Fit for a specific model version — calibration is not transferable across versions.
- Persist to `calibration/{version}/{doc_type}.json` in Blob (`paths.calibration_params`, IMPL-02). This is a small lookup/transform applied at serving time **after** the model call, before the JSON is returned.
- Report the ECE before and after fitting, so a fit that fails to improve calibration is visible rather than assumed.

### 3. `calibration/apply_calibration.py`
- Load the fitted transform for a model version + doc_type; map raw → calibrated confidence at inference time. Used by serving (IMPL-11) and testing (IMPL-12).
- **Fails loudly when no calibration exists for a version/doc_type** rather than silently passing raw (overconfident) values through as if calibrated. A missing calibration is an operational error, not a fallback.
- Applies the review threshold (default 0.7, `DEFAULT_REVIEW_CONFIDENCE_THRESHOLD` — arch §5) and emits `review_flags` for sub-threshold fields, which is the practical payoff of doing calibration properly. On serving's v1 fallback path no field is accepted on it (no measured error rate stands behind 0.7).

### 4. `calibration/list_completeness.py`
The second confidence signal that per-token logprobs structurally cannot provide (arch §5).

**The failure mode:** if the model extracts 6 of 8 claims, the 2 missing claims have **no generated tokens**, so there is no low probability to flag. Per-field confidence on the 6 extracted rows can all be high while the extraction is silently incomplete. This is a recall failure, and token confidence is blind to it.

Implement **two independent cross-checks**, either of which flags the list:
1. **Document-stated count** — a `total claims: N` style field extracted from the document.
2. **Structure-derived count** — the number of table rows MinerU detected on the relevant pages (recorded in `ocr_meta.json` by IMPL-03).

When the model's row count disagrees with either, **flag the whole list for review regardless of per-value confidence**.

**The flag format is a contract, so it is a constant.** Flags are `f"{field}{ROW_MISMATCH_SUFFIX}"` — `"claims:row_count_mismatch"` — and `is_row_completeness_flag()` is the only supported way to recognise one. Both are exported from this module and imported by `data_pipeline.labeling.active_learning`, which has to recognise them to honour the "row completeness outranks every confidence score" rule. The consumer previously matched hand-written prefixes (`list:`, `rows:`, `completeness:`) that this module has never emitted, so the override was dead code and an incomplete Loss Run with high per-value confidence was routed to a spot check.

**A stated count is read however the model spelled it.** `8`, `8.0` and `"8"` are the same assertion; accepting only `int` turned the whole stated-count cross-check off whenever the model emitted a JSON string.

**A flagged list is never fully confident.** `row_completeness_confidence` is scored against whichever signal disagrees *most*, not the first one available — preferring the stated count meant a list flagged solely by the structural check (document says 6, model returns 6, OCR saw 9) shipped confidence `1.0` on a list this function had just flagged. Optionally calibrate a separate `row_completeness_confidence` against ground-truth row counts on the validation set.

This matters most for **Loss Runs**, where a missed claim row is both easy to make and expensive to miss.

## Constraints
- Calibration fit ONLY on held-out data — assert the validation `source_id`s are disjoint from training.
- Params versioned per model version + doc_type (v2.1: per release + serving format); never reused across versions.
- Correctness signal reuses IMPL-08's definition of "correct" — `common.normalize` through `evaluation.metrics.field_accuracy.values_agree` (by the field's declared type) — the same definition as the promotion gate.
- Raw confidence is never returned to callers as if it were calibrated.
- No PII in logs (confidence records reference field paths, not field values).

## Acceptance checklist
- [ ] Raw per-field confidence extracted from a sample generation + logprobs (via IMPL-07 spans), including a `claims[i].amount` row field and `line_of_business`.
- [ ] Aggregation strategy is configurable and the choice is recorded.
- [ ] Temperature and isotonic both reduce ECE on a synthetic miscalibrated set; before/after ECE is reported.
- [ ] Fitted params persist/reload and apply deterministically.
- [ ] `apply_calibration` **raises** when no calibration exists for a version/doc_type.
- [ ] `list_completeness` flags a list when extracted rows < the document-stated count, and separately when < the MinerU-detected row count.
- [ ] A flagged list is routed to review even when every per-value confidence is high.
- [ ] Calibrated confidence is available with **no ground truth** — the inference-time case that makes production review routing possible.
- [ ] Fitting on data overlapping the training split fails the assertion.

---

## Current implementation (2026-10-13)

This spec's §1–§3 describe v1's single-number calibration. What runs now (arch v2.1 §5):

1. **Features per field** (`calibration/features.py::build_document_features`): min / mean / first-token
   logprob, token count, **OCR agreement**, **rule checks**, null flag, cross-mode agreement, field type. A
   feature that was not computed encodes as a neutral 0.5 plus a presence flag, never as 0.
   - *The logprob features read the printed value's own tokens*: a string's characters without its quotes
     (`inference_core.span_map`), and for a canonical envelope the span of `raw` — the reading of the page —
     with `parsed`'s standing in only when nothing was read (`common.canonical.collapse_spans`). `parsed` is
     the same value reformatted, written after `raw` and near-certain whether the reading was right or not.
   - *OCR agreement* compares the page text against the value **as printed** (a canonical envelope's
     `raw`, via `printed_view`), not the normalised `parsed` — a date rewritten to `MM/DD/YYYY` is not on the
     page.
   - *Grounded on the cited page*: given each page's text (`page_texts`, page number → text), agreement is
     `page_agreement` — `common.grounding.ground` against the pages the value's envelope cites: **1.0** printed
     on a cited page, **0.5** printed only on other pages (a wrong page, or another row's figure), **0.0**
     printed on no page, `None` when a cited page has no text (a scan OCR could not read), the value is too
     short to look for (under 3 characters), or there is no text (image_only). Serving and calibration
     fitting both pass each page's text. Without `page_texts`, `ocr_agreement` searches the whole shown text
     with the same matching — word boundaries, every word across line breaks, figures in any format — so "1"
     no longer agrees with every page holding a 1.
   - *Page text* has ONE definition for fitting and serving (`shown_ocr_text`): the OCR text of the pages
     the model was shown, page markers and the blank-page placeholder removed, `None` for `image_only`.
   - *Rule checks* read siblings beside the field (`policy.effective_date` vs `policy.expiration_date`),
     so they fire on canonical policies; totals against their rows; identifier shape.
   - *Field type* (`infer_field_type`) — one of `identifier`, `money`, `date`, `number`, `entity`, `address`,
     `enum`, `free_text`: the reviewed table `configs/field_types.yaml` first; then, **for a common-model
     document only**, the type its schema declares (`common_model_field_types` over `VALUE_TYPE_FIELD_TYPES`:
     `MoneyValue` money, `DateValue` date, year/number/percent number, NAIC and postal code identifier,
     state and the enum-like values enum) and an `additional_fields` value by what was parsed; then the name
     (`common.normalize.infer_field_kind`; an address by any path segment, `carrier.address.city`). The
     self-contained lines and ACORD keep the name-based types for the paths they share with the common model.
   - Features are built on the **value view** of the extraction (envelopes collapsed), keyed like serving.
     A field whose span could not be located is kept with `mapped=False` (no confidence, reviewed), never
     dropped.
2. **Per-field-type calibrators** (`calibration/feature_calibrator.py`): logistic regression per field type
   over those features (pure Python, L2-regularised), fitted on the **calibration half** of validation. A type
   with fewer than `MIN_INSTANCES` (300) labelled instances, or with only one outcome, gets no enforced
   calibrator, and every field of it routes to review. Isotonic above `ISOTONIC_THRESHOLD` (2,000) is
   recorded, not implemented.
3. **Risk-controlled thresholds** (`calibration/thresholds.py`): per field type, the lowest threshold on the
   0.50–0.99 grid whose auto-accepted error rate, as the **upper bound of a 95% binomial interval**, meets the
   target with at least `MIN_ACCEPTED_SAMPLE` (30) accepted fields, chosen on the **threshold half**.
   `DEFAULT_ERROR_TARGETS`: identifier, money, date, number 1%; enum 2%; entity, address 5%; free text has no
   target and is never auto-accepted. A type with no calibrator, or no threshold meeting the target, routes
   to review rather than carrying an unmeasured number.
4. **Per serving format**: each format gets its own calibrators from its own generations (quantization
   moves the logprob distribution), stored at
   `releases/{tenant}/{release}/calibration/{format}/calibrators.json` (`paths.release_calibrators`; the
   calibrators and the thresholds together).
5. **Loss Run reconciliation** (`calibration/reconciliation.py`) per policy period, alongside row
   completeness.

**Where the samples come from.** `orchestration.pipeline_dag.collect_calibration_samples` reads the scope's
validation split, localises its page images, and for each serving format in turn (`bf16` alone with
`--skip-quantize`) loads the staged merged or quantized model, generates **every validation row** with
`evaluation.validation_generation.generate_validation` (constrained as serving is, `page_ref` bounded to each
row's pages, the row's task answer cap, `validation_batch_rows` rows per call — IMPL-08), releases the engine
before the next format loads, and logs what the pass cost: rows, seconds, calls, rows per call and the slowest
call. A pass with too many failed rows is refused (`assert_generations_usable`); a format with no evidence gets
no entry. A resumed `package` finds the calibrators in Blob.

**How a sample is labelled** (`validation_generation.calibration_samples`), by the `val_half` the dataset build
stamped on each row (a row without one is left out, with a warning):
- a policy read in windows is fitted **as it is served**: its windows and their spans merged
  (`merged_windows`, the serving merge), held to the label conventions (`serving.conventions.conform`), then
  featured against each page's text and scored against the merged window targets — not one window at a time;
- every other row is fitted as generated; a row that failed to generate has no features;
- correctness is `values_agree` (IMPL-08's typed comparison), with the label's rows **paired by identity** to
  the answer's (`aligned_for_scoring` on a common-model line, `rows_aligned_to` otherwise), so one omitted or
  reordered row does not make every later row's correct values "wrong";
- a common-model line is featured without its ids, codes and links (`without_bare_values`) and typed as its
  schema declares — what serving calibrates (`serving.pipeline._feature_calibrated`).

**The calibrate stage** (`stage_calibrate`) fits calibrators on the calibration half and thresholds on the
threshold half per format, writes them to the release, and reports per format the unenforced types, the
guarantees, `auto_accept_error_rate`, and the **expected calibration error per field type and overall**
(`expected_calibration_error`, keyed `all`) — both measured on the threshold half, which the calibrators were
not fitted on.

**Thresholds** load back for serving (`ThresholdSet.from_dict`). An explicit target table that leaves a
field type out now means *no promise* — every field of that type is reviewed — instead of silently falling
back to the default target.

## Changes since 2026-10-05

- Noted that the flow diagram is v1's and that serving's v1 path (no calibrator set) flags every field.
- Corrected the v1 persistence path to `calibration/{version}/{doc_type}.json` (`paths.calibration_params`).
- Constraints: correctness is `values_agree` (typed comparison), and v2.1 params are per release and serving format.
- Features: logprob features read the value's own tokens (no quotes; `raw`'s span via `collapse_spans`); OCR agreement grounded on the cited pages (`page_agreement`: 1.0 / 0.5 / 0.0 / none) with the shared `common.grounding` matcher; field types include `number` and the common model's declared value types (common-model documents only); neutral encoding of missing features.
- Calibrators and thresholds: added `MIN_INSTANCES` 300 and the single-outcome rule, the threshold grid, binomial upper bound, `MIN_ACCEPTED_SAMPLE` 30 and the per-type targets; the calibrators file name.
- Samples: described `collect_calibration_samples` (every val row per format, merged model, engine released between formats, rows/seconds/calls logged) and `calibration_samples` (windowed policies merged and conformed as served, rows paired by identity, typed correctness, common-model fields as serving calibrates them).
- Calibrate stage: reports ECE per field type and overall on the threshold half.
