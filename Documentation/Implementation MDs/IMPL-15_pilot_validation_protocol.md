# SPEC 15 — Pilot Validation Protocol (De-Risking Before Full Annotation)

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01–IMPL-14 built.
>
> **This spec is executed, not imported.** It is an operational runbook, not a library — nothing else in the repo depends on it. It exists because architecture v1 added §16, which has no home in the module specs.
>
> **Architecture refs:** `finetuning-architecture-v2.7.docx` §16 (**pilot validation protocol**: 16.1 zero-shot baseline, 16.2 smoke test, 16.3 pilot run, 16.4 exit decision), §0d (the floors the pilot targets come from), §0e (decisions of 2026-10-19), §8 (pilot split ratios), §8b (de-identification), §4c (day-zero bootstrap), §15 (metrics).

## Goal

Before committing to full corpus annotation, run three sequential experiments that de-risk the architecture in order of increasing investment. Each answers a different question, and each is cheap relative to the one after it.

**Run these in order. Do not skip to Experiment C.**

**Policy documents only, for now** (decision 2026-10-22): wherever this protocol says *per document type*, read policies, and train and gate the one policy adapter (`common/adapter.py`; there are no training scopes). The code for ACORD and Loss Run documents is kept, and not trained.

## Deliverables

### 1. `pilot/zero_shot_baseline.py` — Experiment A (Week 1, no annotation cost)

Run the **base, untuned** `Qwen3-VL-8B-Instruct` against **10 real documents per document type**, using the IMPL-01 prompt template with the Fideon SPEC_00 schema injected (for a policy, its line's canonical schema). Compute field-level F1 against ground-truth annotations.

**Implement this as a thin wrapper over the existing extraction command**, not as a parallel path. The documents are imported and OCR'd first (under their own tenant, with their labels); `extract` then reads them by source id and scores them against those labels:

```bash
python -m orchestration.run extract --model base --tenant pilot \
       --source-ids policy_0001 policy_0002 ...        # or: --split test --corpus v1
```

`pilot/zero_shot_baseline.py` holds the scoring and the decision table (`run_baseline`, `summarise_baseline`, `classify_baseline`) over `testing.run_extraction.run_document`; its own CLI only points to the extraction command (it prints the import, OCR and `extract --model base --tenant pilot --source-ids …` commands to run).

IMPL-13's `--model base` and IMPL-02's `resolve_model_version("base")` exist precisely so the baseline, day-zero pre-annotation (IMPL-04), and this experiment all run the same code. A separate implementation here would measure something subtly different from what production runs.

This tells you how much the base model extracts for free and — more usefully — **where it fails**: schema adherence, list-row recall, LoB detection, or OCR arbitration.

**Outcome thresholds and what each means:**

| Zero-shot field F1 | Interpretation | Decision |
|---|---|---|
| **> 0.70** | Strong base-model prior; fine-tuning expected to reach production quality | Proceed to pilot corpus |
| **0.40 – 0.70** | Fine-tuning will materially improve; specific failure modes identified | Proceed, **with targeted corpus coverage for the weak areas** |
| **< 0.40** | Base model likely insufficient as framed | **Review prompt design, schema complexity, and document difficulty *before* committing annotation investment** |

The "field F1" this table reads is `BaselineReport.field_f1`: each document's field normalized match (`score_fields(...).normalized_match`), averaged per type and weighted over every document presented, so a document the pipeline refused counts as zero. The report also records it as `field_normalized_match`, the metric the pilot and the gate use.

- Output: `pilot/reports/zero_shot_baseline.json` — per-doc-type F1, per-metric breakdown (schema validity, list recall, LoB accuracy, OCR arbitration), and the failure-mode distribution.
- Uses IMPL-07 inference core and IMPL-08 metrics — the same code the production gate uses, so the numbers are comparable later.

### 2. `pilot/smoke_test.py` — Experiment B (Weeks 1–2, 5 annotated documents)

Annotate exactly **5 documents per document type**. Train to **intentional overfit**: no train/val split, 5 epochs, learning rate at the upper end of the sweep range.

**This proves the code pipeline is correct — not that the architecture generalises.** It is a sanity check, not a proof of concept.

Confirm all of:
- Training loss reaches **below 0.05 within 2 epochs**.
- The model reproduces the 5 training examples with **field F1 > 0.95 on the training set**.
- End-to-end, without error: the ms-swift training loop, the data collator, LoRA adapter loading, adapter save and push to Azure Blob, vLLM multi-adapter hot-swap, and run manifest generation.

**Why 5 documents and not 1:** a single source document generates only three modality variants (three training examples), which isn't enough to exercise the full data collator pipeline. Five documents per type covers multi-page, OCR-failure, and image-only edge cases.

- Output: `pilot/reports/smoke_test.json` — loss curve, train-set F1, and a pass/fail per pipeline component.
- **A failure here is a code bug, not an architecture problem.** Fix it before spending annotation budget.

### 3. Experiment C — Pilot Training Run (Weeks 2–6, 25–30 documents per type)

Annotate 25–30 documents per document type per IMPL-04 — **reviewed golds only**: no frontier model drafts a gold, and every gold needs a review record — `reviewer_id` and `review_date` — to train or score, a twin its seed's (decisions 2026-10-19 and 2026-10-22; the preflight and the corpus build refuse the run otherwise). Use the **pilot split ratios (~70/18/12)** from IMPL-05 (or the split a delivery arrives with — master §10). Train the ONE policy adapter with the §11.1 pilot hyperparameters (`configs/training/unified.yaml`). The pilot's first cycle starts from the base, since nothing is promoted yet; a later cycle continues the promoted adapter unless given `--fresh` (decision 2026-10-23, IMPL-13 §2). Evaluate on the pilot test split, in every modality regime, with **every page of every document read**, chunk by chunk, never selected pages (decision 2026-10-19).

This is the **minimum experiment that tests the architecture's generalisation claim**: whether one adapter captures cross-type behaviour without the types interfering, whether modality-dropout arbitration functions, whether totals reconciliation catches a missed row on unseen documents, and whether the per-field-type calibrators have enough data to be enforced at all — at pilot volume several field types will not clear the 300-instance floor and will route everything to review, which is the system working rather than failing.

**Pilot success criteria — directional targets, not production thresholds** (`pilot.pilot_run.PILOT_CRITERIA`):

| Metric | Pilot threshold | Source |
|---|---|---|
| Field normalized match (`field_normalized_match`) | ≥ 0.85 | gate floor |
| List-field recall (`list_field_recall`) | ≥ 0.75 | gate floor |
| JSON schema validity (`schema_validity_rate`) | ≥ 0.98 | gate floor |
| Image-only field F1 (`image_only_gap`) | Within 15% of OCR-plus-image F1 (gap < 0.15) | pilot's own |
| Row-completeness detection accuracy | > 80% of documents with missing rows flagged | pilot's own |
| LoB detection accuracy (`lob_detection_accuracy`) | ≥ 0.85 (ACORD and Loss Run documents; see below for policies) | gate floor |
| **Alias generalization** (`alias_generalization_gap`) | Field accuracy on a **held-out surface label** within 15% of accuracy on that field's dominant label (gap < 0.15) | pilot's own |
| **Confusable misattribution rate** | ≤ 5% | gate floor |

**The five gate-floor targets are the gate's own** (arch §0d): `_from_floor` builds each from `evaluation.gating.PILOT_FLOORS` (the threshold) and `GATING_METRICS` (the direction), and compares inclusively (≥ / ≤), as the gate does — so a model exactly at a floor passes both the gate and the pilot. A second copy of the thresholds in the pilot had drifted (field F1 0.80, schema validity 0.9999); there is now one table. The three relative criteria have no gate floor and keep strict comparisons.

**LoB detection on a policy-only pilot.** A canonical policy label carries no `line_of_business` — the line comes with the request — so `lob_detection_accuracy` is structurally not applicable to the policy adapter (`common.adapter.not_applicable_metrics`) and the gate does not ask for it. `pilot/pilot_run.py` records it as not applicable when the pilot holds no ACORD or Loss Run documents (`Criterion.applies_to`; listed under `not_applicable` in the report), so it neither passes nor fails; for such a pilot, read it from `scripts/measure_lob_detection.py --corpus vN [--tenant T]` (the base model reading the line of each policy in the tenant's frozen eval set zero-shot, its line withheld, at serving's thresholds, against the 0.85 floor), and record that it stands in. Line detection stays off in serving until that measurement passes (master §1.3).

**A failed criterion signals a specific architectural issue to diagnose — not a reason to abandon the architecture.** Diagnose the failure-mode distribution, expand corpus coverage for the failing case, and re-run the pilot. Map each failure to its likely cause:

| Failed criterion | First place to look |
|---|---|
| Field match low | Prompt/schema design; corpus coverage for that doc type; LR/epochs on the adapter's run (a targeted retrain, not the deferred sweep) |
| List-field recall low | Loss Run corpus row-length variety; row-completeness signal (IMPL-09); consider list-specific prompt rules |
| Schema validity < 0.98 | Prompt rules (no fences, null-handling); JSON structural discipline in the training corpus |
| Image-only F1 gap > 15% | **The ViT escalation gate (IMPL-06 `vit_gate`)** — but only if errors are perception-type, not schema/reasoning |
| Row-completeness misses | Cross-check sources in IMPL-09: is MinerU's table row count reaching the signal? |
| LoB accuracy low | **LoB corpus coverage (IMPL-05 `corpus_manifest`)** — check the ≥20%-per-value target before blaming the model. For a policy's line (the zero-shot measurement): the line meanings in `configs/lob_meanings.yaml`, and the per-line floor in the measurement's report |
| Alias generalization gap > 15% | **`alias_coverage` in the corpus manifest** — the held-out label is probably under-represented. Also check the field's semantic gloss: a vague `description` gives the model nothing to generalise from |
| Confusable misattribution > 5% | **`confusable_example_count`** — near zero means the corpus taught the mapping but never the boundary. Also check the gloss carries an explicit exclusion clause (IMPL-01) |

**Alias generalization is measured by deliberate hold-out.** Pick one surface label per confusable-prone field (say, *Applicant* for `insured_name`), keep every document using it out of `train`, and place them in the pilot test split. If the model scores well on them it has learned the semantic mapping rather than memorising label strings — which is the whole claim of the canonical-field design (master §1.4) and the one thing no plumbing test can prove.

- Output: `pilot/reports/pilot_run.json` + a `pilot/pilot_report.py` summary that puts all three experiments side by side.

### 4. `pilot/pilot_report.py`
- Aggregates the three experiment reports into one human-readable go/no-go document: baseline → smoke → pilot, each with its thresholds, actuals, and pass/fail.
- States explicitly which criteria failed and the diagnosis path from the table above.
- **Every pilot training run still writes a normal `RunManifest`** (IMPL-02) — pilot runs are registry entries, not untracked experiments.
- **This is the gate the deferred work waits on.** The hyperparameter sweep (IMPL-06 `sweep.py`, arch §11a) runs *after* this protocol passes and *before* production-scale training — sweeping on 25–30 docs/type measures noise, not signal.

## Constraints
- Run the experiments **in order**; each gates the next.
- The pilot is not an exemption from the PII rules. De-identification is **Option A — none**, the state the architecture records for every tenant's build (arch §8b, IMPL-05 §1): pilot data is protected by tenancy, access control and encryption, the corpus manifest records `deidentified: false` and `image_redaction: "unresolved"`, and that is acknowledged to the compliance owner before real documents are used.
- Pilot eval numbers are **directional**. At 3–4 test documents per type, no single metric is trustworthy on its own; the batch's real job is proving the pipeline works end to end before scaling annotation to 1000+/type.
- Re-baseline properly once at real scale — do not carry pilot thresholds forward as production gates.
- Choosing between the de-identification options (A / B / C) stays a compliance open item, to be closed before the first production release (arch §8b). Until another option is chosen, pilot corpora are built under Option A and that limitation is stated, not assumed away.
- Golds are reviewed golds: no frontier model is used for gold, synthetic or training data, and every gold needs a review record, a twin's inherited from its seed (decision 2026-10-22, IMPL-04).

## Acceptance checklist
- [ ] `zero_shot_baseline` runs the base model over 10 docs/type and emits per-type F1 plus a failure-mode distribution.
- [ ] The baseline decision table produces the correct proceed/review recommendation on seeded scores.
- [ ] `smoke_test` trains to overfit on 5 docs/type and reports loss < 0.05 by epoch 2 and train F1 > 0.95.
- [ ] The smoke test exercises and reports pass/fail for every listed pipeline component, including Blob push and vLLM hot-swap.
- [ ] Pilot run evaluates against all eight pilot criteria (`pilot.pilot_run.PILOT_CRITERIA`) and names any that fail, or were not measured, with its diagnosis path.
- [ ] The five criteria with a gate floor take threshold and direction from `PILOT_FLOORS` / `GATING_METRICS` and are met at exactly the floor (`field_normalized_match` 0.85, `schema_validity_rate` 0.98 pass).
- [ ] Every pilot training run produces a `RunManifest`.
- [ ] `pilot_report` renders the three experiments as one go/no-go summary.

---

## Current implementation (2026-10-21)

- **Pilot split**: under 200 documents per type the band is 70/18/12, applied **per line of business**; a
  line with fewer than 5 documents trains whole and is not measured (listed in the manifest's
  `train_only_lines`). At pilot volume many lines fall below that — their pilot numbers do not exist, by
  design, rather than being noise.
- **A delivered split is used as given** (`split_groups.assign_delivered_splits`): the SPEC_21 delivery puts
  each seed and its twins in one split, so test layouts are ones the model never trained on, and the gate
  report adds a `real_only` section (the same metrics over the real documents alone). The per-line carrier
  hold-out (decision 2026-10-19) applies to a drawn split: a line whose real documents name **10 or more
  carriers** holds **one** carrier out of train and validation; a line with fewer holds none out
  (`split_groups.MIN_CARRIERS_PER_LINE_FOR_HELD_OUT`; recorded as `lines_below_carrier_minimum` in the split and
  the run manifest). The hold-out is off for a delivered split. The CGL bundler marks a test document of a
  carrier no training document has as `held_out_carrier`, and evaluation reports such documents apart
  (`held_out_carrier_match`, not gated).
- **Alias hold-out** (above) is still done by choosing documents deliberately; it is not automatic — the
  hash split does not know which surface label to hold out. For common-model lines,
  `scripts/check_alias_coverage.py` lists the recorded labels no training document prints, and evaluation
  splits field accuracy into labels training showed and labels it never did (`unseen_labels`, reported).
- **Freezing a pilot eval set** is per tenant: `freeze-eval-set --corpus vN [--tenant T]` writes
  `golden-eval-set/{tenant}/`, and needs `--allow-small` under 150 test documents per type: the set is then
  directional (its manifest records `small_set_accepted`, and a release scored on fewer than 150 real frozen
  policies is capped at the `interim` tier, with a warning, whatever that flag says), and once frozen it is the yardstick for every later version of that
  tenant. Prefer to freeze from the first build at real scale; if a pilot set is frozen, replacing it later is
  a deliberate delete-and-refreeze after which older scores are not comparable.
- **A pilot whose eval set holds no scans** (the CGL originals are all digital) is not blocked on
  `scanned_accuracy`: it is a conditional gate metric, required only when the eval set holds scanned documents
  — and required whenever the frozen baseline measured it, since its absence then means something broke.
- **Pilots by tenant** (no training scopes since 2026-10-22): the SPEC_21 personal-lines delivery and the CGL
  documents (`gl`, real documents, no twins) each train the one policy adapter, the CGL documents in their own
  tenant, bundled with `python -m data_pipeline.ingestion.prepare_bundles --input <PDFs/ + Gold JSON/> --lob gl --out <dir>`
  (`cgl` reads as `gl`; one family per insured; split drawn once and delivered in the metadata). Each tenant has its
  own corpus, frozen set and promoted release; versions are numbered across every tenant's runs, corpora and
  releases, so two tenants never share a number (IMPL-13). A pilot trains, is gated and is served on
  single-line policies of its known lines only: a package or a policy with no line is left out of training and
  of the gate, and read by the base model (IMPL-06, IMPL-08).
- **Pilot cycles continue** (decision 2026-10-23): a pilot cycle run after a release of its tenant is promoted
  continues that release's adapter and is gated line by line against the best run on each line since the
  last run from the base (IMPL-08); give `--fresh` when the
  pilot is meant to measure a run from the base. Nothing is merged: the pilot's release is its adapter,
  served on the bf16 base.
- **Every page is read** (decision 2026-10-19): each window group reads every page of a policy, chunk by
  chunk, and the declarations group's first window is pages 1–5 plus the detected declarations page. That is
  more windows per policy, so a longer training run; a corpus built before the change must be rebuilt and the
  adapter retrained, and pilot or gate numbers from before it are not comparable with new ones.
- **Before the pilot's `finetune`**: `python -m data_pipeline.ocr_check` (labelled values against the OCR
  text), `python -m orchestration.preflight [--tenant T]` (it also fails on a label with no review
  record and on imported policies that would teach two output shapes, and reports annotation agreement, warning only; add `--skip-corpus` when `finetune` follows in the
  same chain: the build then runs once, and refuses a >25% set-aside itself), and
  `python -m orchestration.smoke_run` (tenant `smoke`, version `v0`, the policy adapter, 4 train / 1 val / 1 test
  sources with their twins; never freezes or packages). The smoke run proves the plumbing; it is not
  Experiment B. Train/serve token parity is measured on the pod with
  `pytest tests/test_parity.py::test_template_token_parity` (GPU-marked; it skips without a GPU, the base
  weights, ms-swift and vLLM), so run it in the training environment there.
- **Reading the results**: `extract` writes per-document JSON (with `all_fields`: every schema field, its value
  and its result against gold) and metrics; `python -m testing.compare_models` runs the same documents through
  the base model and base + adapter and writes both, with a workbook per document. Every stage of a run records
  its wall time in `run-<command>-<tenant>-<version>.json` under `/workspace/logs`.
- **Pod runs**: every pilot command started on the pod runs detached in tmux (IMPL-13); install each pod with
  `bash scripts/pod_bootstrap.sh` (both environments, MinerU 3.4.5's pinned pipeline weights, the base model
  and the spike), and run OCR in `/workspace/venv-ocr` (MinerU 3.4.5, pipeline backend) before any pipeline
  command. A corpus read by two MinerU versions is refused, so documents read under MinerU 1.x are re-read
  first (`run_mineru --all-unprocessed`).

## Changes since 2026-10-05

- Added the alignment line.
- Experiment A: the extraction command as it exists (`--tenant`, `--source-ids` or `--split`/`--corpus`, documents imported and OCR'd first); where the baseline's scoring lives, and that its CLI hint is stale.
- Experiment C: one adapter per pilot scope; a delivered split may replace the drawn one.
- Criteria: LoB detection is not applicable to policy-only scopes; how to stand in `scripts/measure_lob_detection.py` for it, and that `pilot_run` records it as not applicable; diagnosis row for a policy's line.
- Acceptance: eight criteria, not six.
- Current implementation: delivered split and held-out carriers, alias coverage for common-model lines, per-tenant freeze, conditional `scanned_accuracy`, the `personal_lines` and `casualty_fleet` pilots, the pre-`finetune` checks (OCR check, preflight `--skip-corpus`, smoke run), result tools (`all_fields`, `compare_models`, run report), and MinerU 3.4.5 on the pod.
- `pilot_run` records a criterion the pilot's document types cannot measure as not applicable, and the zero-shot baseline prints the current commands (commit f5668a4).

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; the architecture reference is `finetuning-architecture-v2.4.docx` (§16.1–16.4, §0d, §0e, §8b); "do not skip to Experiment C".
- Experiment A: what the baseline's "field F1" is (mean field normalized match over every document presented).
- Experiment C: reviewed golds only; every page read.
- Pilot criteria: field match is `field_normalized_match` ≥ 0.85 (was field F1 > 0.80), schema validity ≥ 0.98 (was 100%), list recall ≥ 0.75, LoB detection ≥ 0.85 and confusable rate ≤ 5%, all taken from `PILOT_FLOORS` / `GATING_METRICS` with the gate's inclusive comparison; the diagnosis table follows.
- Constraints: de-identification is Option A (none) as the architecture records, not "blocked"; the option choice stays a compliance open item; reviewed golds.
- Acceptance checklist: the gate-floor criteria are met at exactly the floor.
- Current implementation: one carrier held out per line, only in a line naming 10 or more carriers; a set frozen with `--allow-small` caps the release at interim; every page read (longer run, rebuild and retrain, earlier numbers not comparable); the preflight's label checks; token parity on the pod.

## Changes of 2026-10-22 (code review)

- Freezing a pilot eval set: the interim cap follows what the pilot's scope was scored on (fewer than 150 frozen documents of a type it covers), not the set's `small_set_accepted` (IMPL-08).
- Reviewed golds: every gold needs a review record (decision 2026-10-22), a twin its seed's; the preflight fails on any without one.
- Goal: the pilot covers policy documents only, for now (decision 2026-10-22).
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.

## Changes of 2026-10-22 (one adapter)

- One adapter (decision 2026-10-22): the pilot trains and gates the one policy adapter, not a scope; LoB detection is not applicable to it; the interim cap counts real frozen policies; pilots by tenant, not by scope; preflight, smoke run and run report without `--scope`.
- Code review of 2026-10-23: versions are numbered across every tenant's runs, corpora and releases; a pilot trains, is gated and is served on single-line policies only; the preflight also fails on imported policies teaching two output shapes.

## Changes of 2026-10-23 (incremental training)

- Experiment C and *Current implementation*: a pilot's first cycle starts from the base; a later one continues the promoted adapter unless `--fresh`, and is gated line by line against it; the release is the adapter, nothing merged (decision 2026-10-23).
- Code review of the incremental training: a continued pilot cycle is gated, line by line, against the best run on each line since the last run from the base (IMPL-08).

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.

## Changes of 2026-10-24 (architecture v2.7)

- Header: aligned with `finetuning-architecture-v2.7.docx`, which records the decisions of 2026-10-23 and 2026-10-24: printed values only, consistent output, fields first, one MinerU, production on Azure.
