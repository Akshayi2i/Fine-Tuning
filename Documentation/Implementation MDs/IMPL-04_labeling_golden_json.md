# SPEC 04 — Labeling, Golden JSON, and the Day-Zero Bootstrap

> Aligned with the code at commit f5668a4 (2026-10-13).

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01 (schemas, alias registry, `common.normalize`, `common.aliases`), IMPL-02, IMPL-03 (MinerU output — `derive_aliases` reads it).
>
> **Architecture refs:** `finetuning-architecture-v2.3.docx` §0a (schema contract), §0b (LoB in every label), §4c (**day-zero bootstrap**), §7 "Golden JSON Generation Mechanism", §13 step 11 (feedback loop).

## Goal

Turn OCR'd documents into human-verified **golden JSON** labels via a bootstrap pre-annotation + human-review workflow, with provenance tracking — and enforce the day-zero rule that no machine draft is accepted without correction until the corpus reaches a defined size.

## Deliverables

### 1. `data_pipeline/labeling/pre_annotate.py`
Generates a first-pass ("silver") JSON draft for a document using a configurable annotator backend. **Don't hand-type JSON from scratch** — a corrected draft is both faster and more consistent than starting blank (arch §7).

| Backend | When | Guard |
|---|---|---|
| `base_qwen3vl` (**default**) | Day zero, before any fine-tuned Foundation exists | none — self-hosted |
| `own_finetuned` | Once v1 is promoted; drafts improve every cycle, so labeling cost falls instead of staying flat | resolves the promoted version via IMPL-02 |
| `external_frontier` | One-time bootstrap only, if base Qwen3-VL zero-shot is too rough to be a useful starting point | **disabled by default** — requires both `--allow-external` AND the `ALLOW_EXTERNAL_PREANNOTATION` env flag, plus a configured zero-retention/enterprise endpoint |

**PII caveat (arch §7, master §8):** insurance documents contain names, TINs/SSNs, addresses, and financials. Sending them to a third-party API may violate the compliance posture or data-processing agreements. The external backend must **refuse to run** rather than warn, unless both permissions are present. If external pre-annotation isn't permissible, use the self-hosted base model and accept rougher first-pass drafts.

The draft is **never trusted as-is** — it is only a starting point for review.

### 2. Day-zero bootstrap rule *(arch §4c — implemented inside `export_golden_labels.py`, not as its own module)*
The day-zero path exists because on day zero **no fine-tuned Foundation adapter exists yet**:

1. Use the **base** `Qwen3-VL-8B-Instruct` with the zero-shot classification prompt (IMPL-01) to identify document type **and ACORD form number**.
2. Run extraction with the **base** model using the schema-injected prompts from IMPL-01.
3. **All pre-annotation output from the base model receives 100% human review** — no draft is accepted without correction until at least **25 labeled examples per document type** exist.

Implement this as an enforced rule, not a guideline — it is a counter and a threshold, so it lives in `export_golden_labels.py` rather than a dedicated module:
- `review_requirement(doc_type)` → `"full"` while `count(golden labels for doc_type) < 25`, otherwise `"confidence_routed"` (§5 active learning below).
- `export_golden_labels` refuses to write a label marked as accepted-without-review while the requirement is `"full"`.
- The gate is **explicitly temporary**: once Foundation v1.0 is trained and promoted, it becomes the zero-shot classifier and the base model is retired from the production inference path. Record the transition in the label metadata.

### 3. `data_pipeline/labeling/derive_aliases.py` — build the registry from labeled documents

The alias registry is **derived from existing (document, golden JSON) pairs**, not hand-written. Given documents you have already labeled canonically, the surface label for each field is recoverable by alignment: the golden JSON supplies the **value**, the OCR supplies the **text**, and the label is what introduces that value on the page.

```bash
python -m data_pipeline.labeling.derive_aliases --doc-type policy \
       --source-ids all --out schemas/aliases/policy.aliases.json --propose
```

**Algorithm — value-anchored label discovery.** For each `(canonical_field, value)` in each golden JSON:

1. **Locate the value in the OCR text.** Not exact match — the golden JSON is canonical (`2026-03-31`, `12400.00`) while the page is not (`03/31/2026`, `12,400.00`). Match through **`common.normalize`** (IMPL-01), the same normalizer evaluation uses, so "found in the document" means the same thing here as "correct" does at eval time.
2. **Skip low-entropy values.** A value of `"CA"` or `"3"` matches everywhere and yields noise. Require a minimum length / uniqueness threshold; record skipped fields rather than guessing.
3. **Read backward from the match** for a label, applying layout patterns in priority order against MinerU's markdown:

   | Pattern | Example | Priority |
   |---|---|---|
   | Inline key-value | `**Named Insured:** Rivera Fabrication LLC` | highest |
   | Bold label, adjacent value | `**Policy Number** WC-8842317-01` | high |
   | Table row label | `\| Applicant \| Rivera Fabrication LLC \|` | high |
   | Table column header | header cell above the value's column | medium |
   | Stacked form layout | label on one line, value on the next | medium |
   | Dotted leaders | `Applicant .......... Rivera Fabrication LLC` | medium |

4. **Normalize the candidate label** — strip punctuation and leaders, collapse whitespace, title-case — so `NAMED INSURED:` and `Named insured` converge.
5. **Aggregate across the corpus** — cluster candidates per canonical field and count supporting documents.

**Deriving confusables, which is the part you cannot hand-write reliably.** While walking the document, also collect **every** label→value pair found, including ones matching no canonical field. Then for each canonical field, a label is a **confusable candidate** when it is unmapped, its value is type-compatible with the canonical field's value (both organisation names, both dates, both currency), and it co-occurs in the same document. That is precisely how *Certificate Holder* surfaces as a confusable for `insured_name` — same shape of value, same page, different entity. Evidence-derived confusables beat guessed ones, because they are the distractors your real documents actually contain.

**Output — a proposal with evidence, never a silent overwrite:**

```json
{
  "insured_name": {
    "aliases": [
      {"label": "Named Insured", "documents": 41, "confidence": 0.98,
       "pattern": "inline_kv", "examples": ["policy_0003", "policy_0007"]},
      {"label": "Applicant", "documents": 5, "confidence": 0.91,
       "pattern": "stacked_form", "examples": ["policy_0044"]}
    ],
    "confusables": [
      {"label": "Certificate Holder", "documents": 23, "basis": "unmapped_type_compatible"}
    ],
    "unresolved": [
      {"source_id": "policy_0061", "reason": "value_not_found_in_ocr"}
    ]
  }
}
```

`--propose` writes to `*.aliases.proposed.json` for review; promoting to the live registry is an explicit confirm step. Treat this as **high-quality candidate generation, not ground truth** — a human confirms once, which is minutes of work against a blank-page alternative measured in hours.

**Three things this gives you beyond the registry itself:**

- **Retroactive `field_provenance`.** The same alignment fills in the provenance map (§5) for documents already labeled — so per-alias eval slicing works on your existing corpus without re-annotating anything.
- **`alias_coverage` before training.** You learn that 41 documents say *Named Insured* and 5 say *Applicant* while there is still time to go collect more *Applicant* documents, rather than discovering it from a disappointing eval.
- **A free QA pass over the golden labels.** Every `unresolved` entry means the value is nowhere in the document — so either the label is wrong or OCR failed on that page. Both are worth knowing, and neither is visible any other way.

**Honest limits.** Values appearing in several places (an insured name in both header and signature block) produce ambiguous anchors — the tool reports the ambiguity rather than picking. Checkbox and derived fields have no text anchor and are skipped. Heavily OCR-mangled values will not match and land in `unresolved`. None of these are silent: every skip and ambiguity is in the report.

### 4. `data_pipeline/labeling/review_tool/`
Integration config for an external labeling tool (Label Studio or Argilla) OR a lightweight custom review UI:
- Presents the page image(s) + OCR + draft JSON side-by-side. A reviewer with insurance domain knowledge corrects every field.
- **Surfaces each field's semantic gloss** (the schema `description`, IMPL-01) next to the field being reviewed, and the alias registry behind it. This is the labeling rulebook: without it, three reviewers will disagree about whether a document's *Applicant* is `insured_name`, and that inconsistency trains directly into the model as noise.
- **Records the observed surface label** for each canonical field it fills — see `field_provenance` below. This is one extra click per field and it is what makes per-alias diagnosis possible later.
- **`line_of_business` is a mandatory review field on every document type** (arch §0b) — the reviewer must set it to a valid LOB enum value or explicitly to `null`. It cannot be left unset.
- Support **double-annotation** on a configurable sample (default 10–20%): a second reviewer labels independently, disagreements are adjudicated, and an **inter-annotator agreement** score is produced. This measures how much noise/ambiguity exists in the labeling process itself — useful context when eval scores plateau below 100%, since some of that ceiling is human disagreement, not model error.
- Provide a task-export/import adapter so labels round-trip cleanly.

### 5. `data_pipeline/labeling/export_golden_labels.py`
- Writes verified labels to `golden-labels/{tenant_id}/{doc_type}/{source_id}/golden.json` and `label_metadata.json`.
- **`label_metadata.json` records provenance, not just the label** (arch §7 step 5): reviewer id, review date, which annotator backend produced the draft, base-model-vs-finetuned draft source, double-annotated flag, agreement score, review requirement in force at time of labeling. This matters later for debugging — if a document type's accuracy regresses, you want to check whether the *labels* for that batch were lower quality rather than only suspecting the model.
- **`field_provenance` — the observed surface label per canonical field** (master §1.4). Golden JSON keys are always canonical; this records what the document actually said:

  ```json
  "field_provenance": {
    "insured_name":  "Applicant",
    "policy_number": "Policy No.",
    "carrier":       "Insurer"
  }
  ```

  This is what lets IMPL-08 slice accuracy **per surface variant** — the difference between "field accuracy is 0.87" and "we score 0.94 on *Named Insured* and 0.61 on *Applicant*, go find more Applicant documents."

  For documents labeled **before** provenance capture existed, `derive_aliases.py` (§3) backfills this map by alignment. New labels capture it directly from the review tool; historic ones are recovered. Either way no document needs re-annotating.

- **Validates every `golden.json` against its Fideon SPEC_00 schema (IMPL-01) before writing** — reject invalid labels (arch §0a). Validation includes:
  - schema conformance (types, required keys, null representation),
  - **presence of `line_of_business`** with a valid enum value or explicit `null`,
  - `acord_form` set for ACORD documents (arch §4b two-level requirement),
  - **`field_provenance` does not name a registered confusable** for that field — a label claiming `insured_name` was found as "Certificate Holder" is rejected outright. This is the single highest-value annotation check in the system, because it catches the exact mistake that trains the model to conflate distinct entities.

**New surface forms.** When a reviewer meets a phrasing not in the registry, they append it to `schemas/aliases/{doc_type}.aliases.json`. That is a **registry edit, not a schema edit** — it triggers no corpus rebuild (master §7). Only editing a field's `description` does.
- Idempotent; re-export overwrites only with a new `label_metadata` version.

### 6. `data_pipeline/labeling/active_learning.py`
Once a model version exists (arch §7 step 6, §13 step 11):
- Run the promoted model over new unlabeled docs, and use the **calibrated confidence** (IMPL-09) to route: high-confidence extractions get a light spot-check, low-confidence fields get full manual review.
- Emits a prioritized review queue ordered by ascending confidence, with per-field flags rather than whole-document flags where possible.
- Also surfaces **list-field row-completeness flags** (IMPL-09) — a Loss Run with a row-count mismatch goes to full review regardless of per-value confidence.
- This is what makes labeling cost drop over successive corpus versions instead of staying flat.
- **Disabled while `review_requirement` is `"full"`** — confidence routing is meaningless before there are 25 labels/type.

## Constraints
- External pre-annotation OFF unless explicitly permitted by flag **and** env.
- Golden labels must be schema-valid and must carry `line_of_business` (for a canonical policy label, whose schema has no such key, the line is `lob` in its label metadata).
- Day-zero: 100% human review until ≥25 labeled examples per doc type.
- Store provenance for every label.
- Labels are tenant-scoped; never mix tenants in a review batch.
- No PII in logs.

## Acceptance checklist
- [ ] `pre_annotate` produces a draft with the self-hosted backend; the external backend **refuses** without both the flag and the env permission.
- [ ] `review_requirement` returns `"full"` below 25 labels/type and `"confidence_routed"` at/above it.
- [ ] `export_golden_labels` rejects a label missing `line_of_business`, one with an out-of-enum LoB value, and one failing schema validation.
- [ ] An ACORD label without `acord_form` is rejected.
- [ ] A label whose `field_provenance` names a registered confusable (`insured_name` found as "Certificate Holder") is **rejected**.
- [ ] `field_provenance` is captured for every filled field and round-trips through the review tool export.
- [ ] Appending a new surface form to the alias registry does not invalidate an existing corpus version.
- [ ] `derive_aliases` recovers `Named Insured` and `Applicant` as aliases of `insured_name` from a fixture set where each appears under a different layout pattern.
- [ ] It matches through `common.normalize`, so a golden `2026-03-31` anchors to a page reading `03/31/2026`.
- [ ] It proposes `Certificate Holder` as a confusable from a document where that label carries a type-compatible value mapping to no canonical field.
- [ ] It skips low-entropy values rather than guessing, and lists them in the report.
- [ ] A golden value absent from the OCR lands in `unresolved` — never silently dropped, never guessed.
- [ ] `--propose` writes `*.aliases.proposed.json` and never overwrites the live registry.
- [ ] Running it over already-labeled documents backfills `field_provenance` without re-annotation.
- [ ] Review workflow exports a corrected golden JSON that passes schema validation.
- [ ] Double-annotation path produces an agreement score on the sampled subset.
- [ ] `label_metadata.json` captures full provenance including draft backend and the review requirement in force.
- [ ] `active_learning` orders a queue by ascending confidence, surfaces row-completeness flags, and refuses to run while `review_requirement` is `"full"`.
- [ ] At bundling, a document-level value lists every page that prints it, a table value keeps to the pages it cites, and a form number cites every page whose header or footer prints it.
- [ ] An extra field repeating a field folds into it; wording and drafting notes filed as extra fields are dropped while captions stay; an extra field naming an empty field moves into it, and stays when the field is filled or two entries name it.
- [ ] A value with figures no page prints is listed in `unprinted_values`, never changed, and the corpus build leaves out the windows holding it; the OCR text over a scanned page never settles that a value is missing.
- [ ] A coverage code in no list of its line is re-coded (code file's `replaces_codes`, then the repository's table, which wins); a package policy filed under one common-model line is left out with the reason.

---

## Current implementation (2026-10-13)

- **Policy labels are the client's canonical JSON** (`configs/canonical schema/LOB Schema/`, 34 `.json`
  files: 31 lines are registered, plus `_fallback`; `_common` is merged by the client's own registry and
  `classic_auto` is read as `personal_auto`). Nine of them — the seven personal-lines overlays (homeowners,
  personal_auto, dwelling_fire, ocean_marine, motorcycle, recreational_vehicle, personal_umbrella), `gl` and
  `_fallback` — are thin SPEC_21 overlays on the **common model**
  (`configs/canonical schema/common schema/common_model.json`, `fideon:common_model_version` **1.1.0**), each
  loaded as one self-contained bundle (`common.schemas._bundle`; an overlay written against another
  common-model version is refused). The other lines are self-contained files read as they are. The fallback
  is used when the line is unknown or the policy spans several lines, and is itself a common-model schema
  with no line part of its own. Every leaf is a FieldValue envelope `{raw, parsed, confidence{score,source},
  page_ref, flagged}` (typed variants on the common model: `DateValue`, `MoneyValue`, `PercentValue` …);
  `raw` is the text as printed, `parsed` the normalised value, `page_ref` every page that prints it. On the
  common model, units carry their own ids (`unit_id`, e.g. `bldg_1`), entries link to them through
  `applies_to`, a coverage row's `coverage_code` comes from the line's list (`fideon:coverage_codes`,
  `<line>.coverage_codes.yaml`), and `additional_fields` is the overflow tier (`label`, `value`) for printed
  values with no field of their own. The training target (`common.canonical.to_model_target`) keeps only
  `raw`, `parsed`, `page_ref` and drops what the document does not state; for a common-model line it is read
  against the model view, so ids, references and codes stay bare strings as the schema declares them.
- **Dates are written `MM/DD/YYYY`** in `parsed` — in the training target and in every output
  (`with_output_dates`). On a common-model line a field is a date when the schema types it `DateValue` (whose
  `parsed` pattern is MM/DD/YYYY), so a form edition printed `05 11` is never read as a date because of its
  name. On a self-contained line a field is a date when a whole word of its name is `date`, `dates`, `dated`
  or `dob` (so `date_of_birth`, `date_licensed`, `report_due_dates` are; `candidate`, `update_reason` are not);
  `tests/test_serving_parity.py` checks every date field in all LOB schemas.
- **`page_ref` matters now**: long policies are trained and served as page windows, and a value is taught
  only in the window that holds its page. The agreed convention (applied at bundling, below) is that a value
  appears **once**, in its proper field, citing **every page that prints it** — and a form number cites every
  page whose header or footer prints it, so a window over a form's fourth page is taught the form it shows. A
  value with no `page_ref` in a multi-window group is reported as *unplaced* and a value on pages no window
  reads as *unread* — both logged per document at corpus build. A window whose target would hold a value no
  page prints (`unprinted_values`) is left out of training whole (`build_jsonl._held_unprinted`).
- **Line of business** for a policy comes from `label_metadata.json` (`lob`), not from the label; it
  selects the canonical schema and is what the corpus manifest's LoB coverage counts. Merged lines are read
  as the line they belong to wherever a line is read (`common.lob.MERGED_LINES`: `classic_auto` →
  `personal_auto`, `watercraft` → `ocean_marine`), and other spellings map to the schema's file name
  (`common.schemas.LOB_SCHEMA_ALIASES`: `cgl` and `general_liability` → `gl`, `workers_comp` → `wc`,
  `commercial_auto` → `auto`).
- **Synthetic documents** (`metadata.synthetic: true`) pin their whole family to train (IMPL-05) — unless the
  delivery carries its own split, which the corpus build then takes as given; synthetic twins may then sit in
  val and test, and the gate reports real documents apart.
- **What the importer records** (`import_labeled_pdfs.import_bundle` → `label_metadata.json`): `source_id`,
  `doc_type`, `acord_form`, `lob` (merged), `field_provenance`, `document_kind`, `policy_number`,
  `template_id` (the family), `carrier`, `twin_index` (a synthetic twin's number, from `sample`),
  `render_mode`, `unprinted_values`, `synthetic`, `split` (the delivery's, when it has one), `labeled_at`,
  `imported_from`, `awaiting_ocr: true` and `held_out_carrier` (the bundler's unseen-carrier mark, which the
  corpus build and the freeze read; IMPL-03).

**Labels corrected at bundling** (`data_pipeline/ingestion/prepare_bundles.py::corrected_gold`, for SPEC_21
twins and seeds and for real documents alone). On a common-model line, in this order, each change listed in
the bundles' `corrections.csv` and the delivery itself never changed:

- **coverage code** in no list of the line → the listed code that replaces it: the `replaces_codes` of the
  line's `<line>.coverage_codes.yaml` (gl 3.0.0 moved three endorsement codes to shared `X_` codes) and this
  repository's table, `configs/coverage_code_recodes.yaml` (27 codes from the overlays' changelogs), which
  wins where both name a code. The decoder only writes listed codes;
- **premium written twice** — a `premium.items` entry for a coverage whose own `premium` holds the same amount
  (or, for a code on several rows, their sum) is dropped, so a coverage premium has one home;
- **carrier missing** (SPEC_21) — filled with the delivery's carrier as the document prints it, citing those
  pages;
- the **label rules** (`data_pipeline/ingestion/label_rules.py::apply_label_rules`), on the digital render's
  text layer:
  - *page list* — a value's pages become the pages that print it; a document-level value printed as itself
    (a policy number in every header) lists them all; a table row's value keeps to the pages it cites; a name
    or phrase keeps every page it cites (it may be only in a logo) and gains a page only where it heads a line;
    a form number gains every page whose first or last four lines print it, carriers' short codes included;
  - *extra field repeats a field* — an extra field holding a distinctive value that exactly one field holds is
    folded into that field (its pages join the field's `page_ref`, except for a table row); the same label and
    value twice becomes one entry; page counters and prose filed as labels are dropped;
  - *extra field from wording* — policy wording filed as an extra field is dropped: drafting notes ("Form
    boilerplate amount (context: …)", "Invisible text-layer remnant …", "Form wording, page 15: …"), clauses
    after a list marker, figures of standard wording and notices (bail bonds, the terrorism act's thresholds,
    toll-free lines, URLs — a label naming a premium stays), and sentences picked up mid-way (a label starting
    lowercase; a camel-case name is a caption);
  - *extra field moved to its field* — into its own **empty** field when exactly one entry names it: business
    description, entity type, bill type, claims phone, authorized representative (only where the gold has a
    countersignature block) and the minimum earned premium (amount or percent); also Protection Class (one
    location) and a "Paid By: Mortgagee" line (`interested_parties[].is_payor`);
  - *link names the building* — an `applies_to` link to a location with one building names the building;
  - *number not printed* — a location or building number no page prints as one is removed (never taken from a
    street address);
  - *printed nowhere* — a value with figures that no page prints, in a field or an extra field, is listed, not
    changed: the bundle's metadata carries it in `unprinted_values`.

  Only what the text can settle is settled: a page counts only when it prints nearly every figure its label
  cites there (`RELIABLE_PAGE_SHARE` 0.9), the OCR text laid invisibly over a scanned page shows where a value
  is printed but never that it is not (`scanned_pages`), a name is never "printed nowhere", and a PDF with no
  text layer gets only the rules that need none. Matching is `common.grounding` (an amount matches the same
  figure in another format; short codes only as a phrase).

A gold still outside its line's schema afterwards, or a package policy (two parts, or a part of another line)
filed under one common-model line, is not bundled; the reason is listed.

**A policy's line** (`label_metadata.lob`) is validated against the lines that have a canonical schema
(`common.scopes.known_lines`, schema spellings and enum spellings both accepted) — not against the 13-value
LOB enum, which rejected `flood`, `cyber`, `professional_eo`, `gl`, `wc` and failed real labels.

**Pre-upload data audit** (`python -m data_pipeline.audit [--input <folder>] [--scope personal_lines]`, default
input `data/bundles`, the `prepare_bundles` output; `--scope casualty_fleet` for the CGL bundles): read-only, on
the laptop, before anything is uploaded. Five layers — structure
(one PDF + `golden.json` + `metadata.json` per folder, PDF opens, no duplicate PDFs), label shape (the
importer's own `check_bundle`, `lob` has a schema and is in scope, `page_ref` inside the PDF, a delivered split
that keeps each source in one split), values against
the PDF (each `raw` found on the page its `page_ref` names, with the `common.grounding` matcher the label rules
and the evaluation also use; wrong page / not found / needs OCR for scans),
formats (readable dates — the build rewrites them MM/DD/YYYY — amounts agreeing with `raw`, period order) and
totals (documents per line, digital vs scanned, page spread, lines under 5, whether the test split reaches the
150 needed to freeze), plus a 5%-per-line spot-check sample. Output in `data/audit_report/` (git-ignored: the
CSVs quote label values); the command exits non-zero while blockers remain.

The audit reads the **importer's layout**: each immediate subfolder of the input is one document. PDFs loose in
the input folder, or nested below a subfolder (e.g. `<input>/homeowners/<doc>/`), are blockers — the importer
would never see them. A malformed `page_ref` is one blocker, never a crash; an unreadable PDF is one cause, not a
row per value. Page text is normalised once per page.

## Changes since 2026-10-05

- Canonical schemas: documented the common model 1.1.0 (`configs/canonical schema/common schema/common_model.json`) and the nine overlays loaded as bundles (seven personal lines, `gl`, `_fallback`); the fallback is a common-model schema; unit ids, `applies_to`, coverage-code lists and the `additional_fields` overflow tier; corrected the "34 files" line (31 registered lines plus the fallback).
- Dates: on common-model lines a date is a `DateValue` by schema type, not by field name.
- `page_ref`: the value-once-on-every-page convention, form numbers on every header/footer page, and windows holding an unprinted value left out of training.
- Line of business: merged lines (`classic_auto`, `watercraft`) and spelling aliases (`cgl` → `gl` …).
- Synthetic documents: train-only unless the delivery carries its own split.
- Added what the importer writes to `label_metadata.json` (`carrier`, `twin_index`, `render_mode`, `unprinted_values`, `split`, `awaiting_ocr` …), and `held_out_carrier`, which the importer carries since commit f5668a4.
- New section on labels corrected at bundling: `corrected_gold` (recodes incl. `replaces_codes`, premium written twice, carrier missing) and `label_rules` (page lists, folded extra fields, wording dropped, extra field moved into its empty field, building links, unprinted numbers, printed nowhere).
- Audit: default input is `data/bundles` (not `data/training data`), split check, shared `common.grounding` matcher; new acceptance items for the label rules.
