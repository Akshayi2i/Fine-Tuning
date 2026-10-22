# LOB identification for scanned and L1-failed documents: classifier cascade

6 October 2026. For the pipeline developer (main repository). Specs: SPEC_06 amendment of 6 October 2026 (items 1–7), SPEC_00 amendment of 6 October 2026.

## Design in one paragraph

When L1 gives no LOB (every scanned PDF, and native PDFs where L1 fails), a CPU classifier reads the MinerU markdown of the first pages and scores the 48 L1 codes. If it is confident and no package is likely, the LOB is fixed without a VLM call. Otherwise the existing base-model LOB-ID pass runs, with the classifier's top three codes as its hint. The fixed LOB selects the adapter (`LOB_TO_LAYOUT_FAMILY`) and the prompt schema and FSM (`lob_schema_map.yaml`). After extraction the classifier checks the extracted LOB evidence; a disagreement sends the document to review.

## Inputs that already exist

- `config/lob_registry.yaml`: the 48 L1 codes (labels).
- The corpus of real policies, foldered by ACORD LOB code (47 codes). ACORD → L1 crosswalk: comments of `LOB_TO_LAYOUT_FAMILY` in SPEC_06 §9a (CPL and PPKGE → homeowners, AGLIA → agriculture_farm, OTHER → specialty, BOAT → watercraft, and so on).
- MinerU output per page (SPEC_05), signed-off golds with `forms_and_endorsements[].form_number`.

## Items

| # | Change | Needed before |
|---|---|---|
| 1 | Labelled document table and split | everything else |
| 2 | Form-prefix table from the corpus | classifier |
| 3 | Title and keyword signals | classifier |
| 4 | Classifier and package model, calibrated | cascade |
| 5 | `fideon/lob_id/` inference module | cascade |
| 6 | Cascade in L3, envelope fields, LOB-ID hint | serving |
| 7 | Consistency check after extraction | serving |
| 8 | Evaluation report and thresholds | going live |

---

### 1. Labelled document table and split

**Prompt.**
> Build data/lob_id/documents.csv with one row per real policy document of the corpus: path, acord_code (folder), l1_code (via the crosswalk in SPEC_06 §9a LOB_TO_LAYOUT_FAMILY comments; fail on any code with no mapping), carrier, modality (native_pdf | scanned_pdf), is_package (true if the document's gold has more than one lob_parts entry or the folder marks it as a package; else false), page_count. Exclude synthetic twins: they share text with their seed and would inflate accuracy. Split by carrier: for every L1 code with two or more carriers hold out at least one carrier for test, and keep a validation set by carrier for calibration; write the split to data/lob_id/split.csv and the counts per code and split to data/lob_id/split_report.md, listing codes with one carrier. Add a test that no carrier appears in two splits for the same code.

**Accepted when.** The table covers every real document; the split report lists counts for all codes.

### 2. Form-prefix table

**Prompt.**
> From the train split, collect every form number (from signed-off golds' forms_and_endorsements, and from MinerU text of the first 3 pages with the form-number pattern of synth_core.sections). Reduce each to its prefix (letters plus the first number group, e.g. "HO 00 03" → "HO 00", "CG 00 01" → "CG"; keep both levels). For each prefix count documents per L1 code. Write config/lob_id/form_prefixes.yaml with, per prefix: the code distribution, the share of the top code, and the document count. Mark a prefix as "decisive" only where one code holds nearly all its documents and the count is not trivial; propose the cut-offs and list the decisive prefixes in a review table for the project owner; do not hard-code prefixes from outside the corpus. Mark IL-family prefixes and "common policy declarations" titles as package signals.

**Accepted when.** The YAML and the review table exist; the project owner approves the decisive list.

### 3. Title and keyword signals

**Prompt.**
> From the train split, find for each L1 code the phrases of the first 3 pages (MinerU markdown, lower-cased, numbers masked) that are most distinctive for that code against all others (for example by chi-square or log-odds on word 1–3-grams, minimum document frequency 3). Write the top candidates per code to config/lob_id/signals.yaml with their train precision and support, as a proposal; mark the confusable sets (homeowners / dwelling_fire / mobile home; personal_umbrella / umbrella; personal_property / inland_marine; auto) and list the phrases that separate them. Stop for review of the list before it is used as a hand-weighted feature.

**Accepted when.** The YAML is reviewed; the classifier of item 4 also learns these phrases directly, so the list is used for explanation and as a feature, not as the only signal.

### 4. Classifier and package model

**Prompt.**
> In fideon/lob_id/train.py train on the first 3 pages of MinerU markdown per document: TF-IDF word 1–2-grams and character 3–5-grams, plus binary features for decisive form prefixes (item 2) and approved signals (item 3), into a multinomial logistic regression over the 48 L1 codes (codes with no documents are reported, not trained). Train a separate binary model for is_package on the same features plus package signals. Calibrate both on the validation split (isotonic or sigmoid, whichever has the lower Brier score) and save to models/lob_id/<version>/ with the feature config, label list and training report; track with the model registry, not Git. Keep the code CPU-only; report inference time per document.

**Accepted when.** Models load from the registry; the training report gives validation accuracy and calibration per code.

### 5. Inference module

**Prompt.**
> Write fideon/lob_id/identify.py: identify(pages: list[str], max_pages: int = 3) -> LobIdResult with fields top: list[(code, probability)] (top 3), package_probability, signals_fired (form prefixes and phrases with their codes), model_version. Pure function of its inputs, deterministic, CPU. Add tests on fixtures: a homeowners declarations text returns homeowners first; an IL-forms common declarations text gives package_probability above the package threshold; an empty input returns no code.

**Accepted when.** Tests pass; p95 time per document is reported.

### 6. Cascade in L3

**Prompt.**
> In the L3 orchestration for policy_check, when L1 gives no LOB: run identify() on the MinerU pages; if top-1 probability >= accept_threshold and package_probability < package_threshold (config/lob_id/thresholds.yaml), fix the LOB with lob_source="classifier"; otherwise run the LOB-ID pass of SPEC_06 §9a with the classifier's top three codes and fired signals as the hint (change build_lob_id_prompt to take them in place of the L1 hint), lob_source="lob_id_pass"; if it returns no code, lob_source="unknown" and the 5 October fallback applies. Set ExtractionMeta.lob_source and lob_confidence (SPEC_00 amendment of 6 October). Then route by LOB_TO_LAYOUT_FAMILY and lob_schema_map.yaml as SPEC_06 specifies. Native PDFs with an L1 LOB are unchanged. Add tests for each branch, including a package going to the LOB-ID pass even when one code is confident.

**Accepted when.** Tests pass; every policy_check envelope carries lob_source.

### 7. Consistency check

**Prompt.**
> After extraction, run identify() on a short text built from the extracted document LOB text, lob_parts[].title and forms_and_endorsements[].form_number. If its top code differs from the routed LOB (for native PDFs: from L1's LOB), set ExtractionMeta.lob_consistency="mismatch" and route the document to review (SPEC_07 review flag, SPEC_11); otherwise "ok". Never re-extract automatically. Add tests for both outcomes.

**Accepted when.** Tests pass; mismatches appear in the review queue with the two codes.

### 8. Evaluation and thresholds

**Prompt.**
> On the held-out carrier test split, report per code and overall: classifier top-1 and top-3 accuracy, layout-family accuracy, package precision and recall, expected calibration error; the same for the LOB-ID pass alone (base model, no hint) and for the full cascade at a range of accept_threshold values; for each threshold, the share of documents decided without a VLM call and the cascade's top-1 accuracy. Report the confusable sets separately. Write docs/lob_id_evaluation.md and propose accept_threshold and package_threshold with the trade-off table. Do not set the thresholds in config until the project owner approves.

**Accepted when.** The report is reviewed and the approved thresholds are committed.

---

Not in scope: changes to L1 detection; training a LoRA for LOB identification (considered only if item 8 shows the cascade below the accuracy the project owner requires).
