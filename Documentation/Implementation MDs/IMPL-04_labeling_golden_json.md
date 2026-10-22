# SPEC 04 — Labeling, Golden JSON, and the Day-Zero Bootstrap

> Aligned with the code of 2026-10-21.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01 (schemas, alias registry, `common.normalize`, `common.aliases`), IMPL-02, IMPL-03 (MinerU output — `derive_aliases` reads it).
>
> **Architecture refs:** `finetuning-architecture-v2.4.docx` §0a (schema contract), §0b (LoB in every label), §0e (decisions of 2026-10-19: **no frontier model**, reviewed golds only), §4c (**day-zero bootstrap**), §7c "Golden Label Generation", §13 step 11 (feedback loop).

## Goal

Turn OCR'd documents into human-verified **golden JSON** labels via a bootstrap pre-annotation + human-review workflow, with provenance tracking — and enforce the day-zero rule that no machine draft is accepted without correction until the corpus reaches a defined size. Drafts come from the self-hosted Qwen3-VL only, and every gold needs a review record to train or score (decision 2026-10-22); a twin inherits its seed's.

## Deliverables

### 1. `data_pipeline/labeling/pre_annotate.py`
Generates a first-pass ("silver") JSON draft for a document using a configurable annotator backend. **Don't hand-type JSON from scratch** — a corrected draft is both faster and more consistent than starting blank (arch §7).

| Backend | When | Guard |
|---|---|---|
| `base_qwen3vl` (**default**) | Day zero, before any fine-tuned Foundation exists | none — self-hosted |
| `own_finetuned` | Once v1 is promoted; drafts improve every cycle, so labeling cost falls instead of staying flat | resolves the promoted version via IMPL-02 |

These are the only two backends (`pre_annotate.BACKENDS`), both self-hosted.

**No frontier model, ever (decision 2026-10-19, arch §0e, §7c):** frontier models are never used for gold, synthetic or training data. There is no external backend: `check_backend` refuses any backend outside `BACKENDS`, and one asked for by name (`external_frontier`, or any name containing `frontier` or `external`) is refused with the decision in the message — it is not a missing permission a flag could grant. `--allow-external`, `ALLOW_EXTERNAL_PREANNOTATION` and `EXTERNAL_PREANNOTATION_ENDPOINT` are removed; `.env.example` says there is nothing to configure. `pre_annotate_batch` checks the backend before the first document, so a refused backend stops the whole batch, while one unreadable document does not. Insurance documents carry names, TINs/SSNs, addresses and financials, so no document leaves the tenancy boundary for drafting either (master §8).

The draft is **never trusted as-is** — it is only a starting point for review.

### 2. Day-zero bootstrap rule *(arch §4c — implemented inside `export_golden_labels.py`, not as its own module)*
The day-zero path exists because on day zero **no fine-tuned Foundation adapter exists yet**:

1. Use the **base** `Qwen3-VL-8B-Instruct` with the zero-shot classification prompt (IMPL-01) to identify document type **and ACORD form number**.
2. Run extraction with the **base** model using the schema-injected prompts from IMPL-01.
3. **All pre-annotation output from the base model receives 100% human review** — no draft is accepted without correction until at least **25 labeled examples per document type** exist. Past day zero a draft is still never a gold until a person has reviewed it (§5, *Reviewed golds only*).

Implement this as an enforced rule, not a guideline — it is a counter and a threshold, so it lives in `export_golden_labels.py` rather than a dedicated module:
- `review_requirement(doc_type)` → `"full"` while `count(golden labels for doc_type) < 25`, otherwise `"confidence_routed"` (§5 active learning below).
- `export_golden_labels` refuses to write a label marked as accepted-without-review: while the requirement is `"full"` with the day-zero reason, and after it too, since such a draft has no review record and could never train or score (decision 2026-10-22).
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
- Support **double-annotation** on a configurable sample (default 10–20%): a second reviewer labels independently, disagreements are adjudicated, and an **inter-annotator agreement** score is produced (`export_golden_labels.inter_annotator_agreement`, field level). This measures how much noise/ambiguity exists in the labeling process itself — useful context when eval scores plateau below 100%, since some of that ceiling is human disagreement, not model error.
- **The preflight reports agreement before training** (`orchestration.preflight.check_label_agreement`): how many labels are double-annotated (`double_annotated`, `agreement_score` in `label_metadata.json`), with their mean and lowest agreement. It is a **warning only**: no spec states a minimum, so `common.constants.MIN_INTER_ANNOTATOR_AGREEMENT` is `None`; with none double-labelled it warns that label noise is unmeasured. Setting the constant turns a document below it into a FAIL.
- Provide a task-export/import adapter so labels round-trip cleanly.

### 5. `data_pipeline/labeling/export_golden_labels.py`
- Writes verified labels to `golden-labels/{tenant_id}/{doc_type}/{source_id}/golden.json` and `label_metadata.json`.
- **`label_metadata.json` records provenance, not just the label** (arch §7 step 5): `reviewer_id` (required by `export_golden_label`, and refused when blank or a placeholder), `review_date` (the time of export), `draft_backend` (which backend produced the draft — base model or fine-tuned; `unknown` when not given), `double_annotated`, `agreement_score`, `review_requirement` in force at time of labeling, `accepted_without_review`, `field_provenance`. This matters later for debugging — if a document type's accuracy regresses, you want to check whether the *labels* for that batch were lower quality rather than only suspecting the model.
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

**Every gold needs a review record (decision 2026-10-22; decision 2026-10-19, arch §0e, §7c).** A gold trains, is frozen into the eval set or is scored only when its metadata records who reviewed it and when. `export_golden_labels.review_problem(label, metadata)` returns why a gold may not, or `None`; `has_review_record(metadata)` is the same test as a yes or no. It reads the metadata alone:
- **`reviewer_id`** — a string, not blank and not a placeholder ("pending", "none", "N/A", "TBD", "unknown", "-", "null" …, case and punctuation aside);
- **`review_date`** — an ISO date or date-time, not in the future (a date with no zone is in the future only once it is ahead in every zone, UTC+14);
- **a synthetic twin** is reviewed only when its seed was: its record is the seed's, inherited when the bundle is prepared, and `review_inherited_from` names that seed (IMPL-03). A twin naming no seed is not reviewed, whatever record it holds of its own.

The gold's own notes (`fideon:provenance`) and `draft_backend` decide nothing: a note can be read either way, and only a person's name says a person checked a gold. `draft_backend` is kept as information. The message names what is missing.

It is enforced at every door: `import_labeled_pdfs.check_bundle` makes it an error (so both the import and the pre-upload audit block it; IMPL-03), `import_prepared` refuses it, the preflight names every such label (`check_reviewed_labels`: FAIL), the corpus build refuses the run on one already in Blob (`pipeline_dag.load_labeled_documents`), the freeze refuses to copy one into the eval set and the gate's golden eval refuses a frozen one (IMPL-08), and the harness extracts but does not score one (IMPL-12). Both importers copy `reviewer_id`, `review_date`, `draft_backend` and `review_inherited_from` from the bundle's metadata into `label_metadata.json` (`REVIEW_KEYS`, `review_record`). A label written through `export_golden_label` records `reviewer_id` (a required argument, checked before anything is written) and `review_date`.

This **overrides Fideon SPEC_21's model-drafted seed golds** (SPEC_21 P2, §5.3, where a frontier model drafts each seed gold, schema and variation map for a reviewer to sign off): no gold here is drafted by a frontier model, and no delivered gold trains without a review record.

### 6. `data_pipeline/labeling/active_learning.py`
Once a model version exists (arch §7 step 6, §13 step 11):
- Run the promoted model over new unlabeled docs, and use the **calibrated confidence** (IMPL-09) to route: high-confidence extractions get a light spot-check, low-confidence fields get full manual review.
- Emits a prioritized review queue ordered by ascending confidence, with per-field flags rather than whole-document flags where possible.
- Also surfaces **list-field row-completeness flags** (IMPL-09) — a Loss Run with a row-count mismatch goes to full review regardless of per-value confidence.
- This is what makes labeling cost drop over successive corpus versions instead of staying flat.
- **Disabled while `review_requirement` is `"full"`** — confidence routing is meaningless before there are 25 labels/type.

## Constraints
- No frontier or external pre-annotation, ever: only the self-hosted `base_qwen3vl` and `own_finetuned` backends exist.
- Every gold needs a review record (`reviewer_id`, `review_date`) to train, be frozen or score (decision 2026-10-22); a twin inherits its seed's.
- Golden labels must be schema-valid and must carry `line_of_business` (for a canonical policy label, whose schema has no such key, the line is `lob` in its label metadata).
- Day-zero: 100% human review until ≥25 labeled examples per doc type.
- Store provenance for every label.
- Labels are tenant-scoped; never mix tenants in a review batch.
- No PII in logs.

## Acceptance checklist
- [ ] `pre_annotate` produces a draft with the self-hosted backend; `BACKENDS` is exactly `base_qwen3vl`, `own_finetuned`; `external_frontier` asked for by name is **refused** with the decision in the message, and stops a whole batch; no external pre-annotation setting remains in the code or `.env.example`.
- [ ] `review_problem` refuses every gold without a real `reviewer_id` (blank and placeholders refused) and an ISO `review_date` not in the future, and a twin naming no seed it inherited its record from, whatever its notes or `draft_backend` say; `check_bundle`, `import_prepared`, the preflight, the corpus build, the freeze and the gate's golden eval all refuse it, and the harness does not score it; `export_golden_label` refuses a draft accepted without review and a placeholder reviewer, writing nothing.
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
- [ ] Double-annotation path produces an agreement score on the sampled subset; the preflight reports the count, mean and lowest agreement and warns (no minimum is set).
- [ ] `label_metadata.json` captures full provenance including draft backend and the review requirement in force.
- [ ] `active_learning` orders a queue by ascending confidence, surfaces row-completeness flags, and refuses to run while `review_requirement` is `"full"`.
- [ ] At bundling, a document-level value lists every page that prints it, a table value keeps to the pages it cites, and a form number cites every page whose header or footer prints it.
- [ ] An extra field repeating a field folds into it; wording and drafting notes filed as extra fields are dropped while captions stay; an extra field naming an empty field moves into it, and stays when the field is filled or two entries name it.
- [ ] A value with figures no page prints is listed in `unprinted_values`, never changed, and the corpus build leaves out the windows holding it; the OCR text over a scanned page never settles that a value is missing.
- [ ] A coverage code in no list of its line is re-coded (code file's `replaces_codes`, then the repository's table, which wins); a package policy filed under one common-model line is left out with the reason.
- [ ] `split_form_edition` moves "CG 00 01 04 13" to number `CG 00 01` and `edition_date` `04/13`, and a bracketed edition after a form code likewise; a filled `edition_date` is never overwritten; "HO 04 35" and "CG 20 10 04" are left alone, the second warned on at import; on a common-model policy line a bundle still carrying a movable edition is refused at import and audit, a self-contained line's numbers are left as printed, and a gold stored before the rule is read under it by the corpus build (`load_golden_label`) as by the gate.

---

## Current implementation (2026-10-21)

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
- **`page_ref` matters now**: policies are trained and served as page windows over **every page**, chunk by
  chunk, never selected pages (decision 2026-10-19), and a value is taught only in the window that holds its
  page. The agreed convention (applied at bundling, below) is that a value
  appears **once**, in its proper field, citing **every page that prints it** — and a form number cites every
  page whose header or footer prints it, so a window over a form's fourth page is taught the form it shows. A
  value with no `page_ref` is taught in its group's only window, or in the window the declarations lead; in
  any other multi-window group it is reported as *unplaced*, and a value on pages no window reads as *unread* — both logged per document at corpus build. A window whose target would hold a value no
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
  `render_mode`, `unprinted_values`, `synthetic`, `split` (the delivery's, when it has one), `reviewer_id`,
  `review_date`, `draft_backend` and `review_inherited_from` (from the bundle's metadata; what `review_problem`
  reads), `labeled_at`,
  `imported_from`, `awaiting_ocr: true` and `held_out_carrier` (the bundler's unseen-carrier mark, which the
  corpus build and the freeze read; IMPL-03). `import_prepared` records the same four review keys.

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
  - *form edition moved* (first, and needing no text) — a form number that carries its edition in a shape no
    form number takes, with the form's `edition_date` empty, gives the edition to `edition_date` and keeps the
    number (`split_form_editions`, row by row through `split_form_edition`). Two shapes only: ISO's
    `AA 99 99 MM YY` ("CG 00 01 04 13" → `CG 00 01`, edition `04 13`) and a month and year in brackets after a
    form code ("IL0017(11-98)"). The edition is parsed `MM/YY` (the schema's code) and takes the number's
    `page_ref`, confidence and flag. A filled `edition_date` is never overwritten; a number that only *may*
    end in its edition ("CG 20 10 04", `may_hold_edition`) is left alone, and the import warns on it. The
    labels did both, and the model learned both — so this **changes training targets**: the bundles are prepared again and the adapters
    retrained. The rule holds where the label rules hold — a policy on a common-model line
    (`label_rules.label_rules_apply`) — and nowhere else. There the import and the audit refuse a bundle that
    still carries such an edition (`check_bundle`, `import_prepared`, IMPL-03); serving's `conventions.conform`
    applies the same rule to an answer and carries the number's span to the moved edition (IMPL-11); and a gold
    stored before the rule is read under it without being rewritten (`label_rules.under_edition_rule`) — by the
    corpus build (`export_golden_labels.load_golden_label`), the gate's golden eval (`golden_eval.verified_label`,
    IMPL-08) and the harness alike;
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
importer's own `check_bundle` (so a form's edition left in its number on a common-model line and a gold with
no review record block here too), `lob` has a schema and is in scope, `page_ref` inside the PDF, a delivered
split that keeps each source in one split), values against
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

## Changes since 2026-10-13

- Header and goal: aligned with the code of 2026-10-21; the architecture reference is `finetuning-architecture-v2.4.docx` (§0e decisions, §7c golden label generation); drafts come from the self-hosted model only.
- §1 `pre_annotate`: no frontier model, ever — `external_frontier`, `--allow-external`, `ALLOW_EXTERNAL_PREANNOTATION` and `EXTERNAL_PREANNOTATION_ENDPOINT` removed; `check_backend` refuses a frontier backend by name.
- §2 Day zero: a draft is never a gold until reviewed, past day zero too.
- §4 Review tool: the preflight reports annotation agreement, warning only (`MIN_INTER_ANNOTATOR_AGREEMENT = None`).
- §5 Export: the metadata keys as written; new *Reviewed golds only* (`review_problem`, `has_review_record`, `_model_drafted`; enforced at import, audit, preflight and corpus build), overriding Fideon SPEC_21's model-drafted seed golds.
- Constraints and acceptance checklist: the frontier ban, the reviewed-gold rule, agreement in the preflight, the form-edition rule.
- Current implementation: windows over every page; the importer records `reviewer_id`, `review_date`, `draft_backend`; the *form edition moved* label rule (`split_form_edition`, `split_form_editions`), which changes training targets; the audit's label check includes both new blockers.

## Changes of 2026-10-22 (code review)

- A value with no `page_ref` is taught in its group's only window or the window the declarations lead; elsewhere it is still reported `unplaced`.
- §5: every gold needs a review record (decision 2026-10-22), read from its metadata alone — a real `reviewer_id`, an ISO `review_date` not in the future, and for a twin the seed it inherited the record from (`review_inherited_from`); the note-reading `_model_drafted` is gone and `draft_backend` is information only; the freeze, the gate's golden eval and the harness enforce it too.
- §2 and §5: `export_golden_label` refuses a draft accepted without review after day zero too, and a placeholder reviewer, writing nothing.
- The form-edition rule holds on common-model policy lines only (`label_rules_apply`); a stored gold is read under it by `load_golden_label`, so the corpus build and the gate read it alike.
- What the importer records: `review_inherited_from` beside the review record.
- Constraints and acceptance checklist: the review-record rule and the form-edition rule's lines.
