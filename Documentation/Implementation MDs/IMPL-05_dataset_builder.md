# SPEC 05 — Dataset Builder (JSONL, Modality, Split)

> Aligned with the code of 2026-10-22.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02, IMPL-04.
>
> **Architecture refs:** `finetuning-architecture-v2.5.docx` §0e (decisions of 2026-10-19), §6 (modality dropout), §7 (dataset format, schema registry, prompt template versioning, long documents), §8 (corpus versioning, split strategy), §8a (MinerU pin in the manifest), §8b (tenant isolation per Fideon SPEC_12; **de-identification per Fideon SPEC_11 — Option A, none, see §1**), §0b (LoB coverage).

## Goal

Compile processed docs + golden labels into versioned, chat-format JSONL training corpora — applying the 3-regime modality split and leakage-safe train/val/test splitting, with a manifest that pins everything the corpus depends on.

## Deliverables

### 1. De-identification — **Option A (none); no `data_pipeline/deidentify/` module**

Arch §8b rejects text-only Presidio de-identification (Fideon SPEC_11) and records **Option A — no de-identification** as the current state for every tenant's build. No de-identification module is built.

**Why not text-only: it is a training bug, not only a compliance gap.** Text-only de-identification changes the **text**, not the **page images** the vision encoder reads. If the text is de-identified and the image is not:

| Regime | Share of Foundation corpus | What the model is taught |
|---|---|---|
| `image_only` | **30%** | Image shows "John Smith"; target says `PERSON_1`. **The target is not derivable from the input.** These are unlearnable examples — 30% of the corpus becomes pure hallucination pressure. |
| `ocr_plus_image` | **50%** | OCR says `PERSON_1`, image says "John Smith", target says `PERSON_1` → teaches **trust-OCR-over-image**, the exact inverse of what the decoder LoRA and the 20% `noisy_ocr_image` regime exist to teach (arch §3, §6). |

**Half-de-identifying is worse than either extreme.** The architecture lists three options; choosing between them is the compliance owner's open item before the first production release (arch §8b):

- **A — no de-identification (the current state).** Real values, protected by tenancy, RBAC and encryption; the input/target contract stays coherent, and PII in `corpus/` carries the same posture as `raw-documents/`.
- **B — consistent replacement.** Presidio entity spans plus OCR bounding boxes: the value replaced with realistic fake text in the image, the OCR text and the target. Needs an image-redaction pipeline.
- **C — a PII-free synthetic layer** (arch §4d) for any shared model; real data only in tenant-scoped models.

Under Option A the corpus manifest records `deidentified: false` and `image_redaction: "unresolved"`, every row carries `deidentified: false`, the run registry records it without enforcing it (IMPL-02), and **the limitation is stated to the compliance owner rather than left implicit**. If option B is chosen this section becomes the Presidio + image-redaction module.

**Whichever is chosen, one rule holds:** de-identification must be **consistent across the OCR text, the page image, and the target JSON**. Any inconsistency between input and target teaches the model to hallucinate.

### 2. `data_pipeline/dataset_builder/split_groups.py`
- **Split at GROUP level, BEFORE task and modality expansion** (arch v2.1 §8.2). The dataset stage first assigns every document a family `group_id` with `data_pipeline/ingestion/dedup_and_group.py`, within its doc type, from: the source checksum OCR recorded (`ocr_meta.source_checksum` — the dataset build may not read `raw-documents/`), a MinHash over the OCR text, and the declared carrier + `template_id` + insured. Splitting per document would put one renewal of an account in `train` and the next in `test`, and eval would measure template memorisation.
- **Not yet computed:** the page-1 layout perceptual hash. No image-hash dependency is installed, so same-template documents with different text group only through a declared `template_id`.
- Ratio scales with per-type volume (arch §8), configurable, seeded, logged:

| Volume per doc type | Train / Val / Test | Note |
|---|---|---|
| under 200 (pilot, ~25–30/type) | 70 / 18 / 12 | ≈17–21 train / 4–5 val / 3–4 test. **Treat metrics as directional** — the batch's job is proving the pipeline works, not measuring model quality. |
| 200–999 | 75 / 15 / 10 | Val/test now large enough for stable metrics |
| 1000+ | 80 / 10 / 10 | Target state: 10% is ≥100 documents per split |

**Every triple must sum to 1.0**, and `common.constants` asserts it at import. The splitter assigns by hash threshold, so a triple summing to 1.05 gives the test split 10% while `ratios_by_doc_type` — copied verbatim into the corpus manifest — records 15%. Every downstream reader then believes the eval population is half again as large as it is. **Both split edges must also only rise from band to band** (`split_groups.assert_bands_only_grow_train`, at import), so crossing a band moves groups only toward train.

- Output: a deterministic `source_id → split` assignment, saved for reproducibility.
- **A split delivered with the data is used as given** (`assign_delivered_splits`; see *Delivered split* below) instead of being drawn here.
- **Splitting is tenant-scoped** — never build a split spanning tenants.

### 3. Modality expansion — `data_pipeline/dataset_builder/build_jsonl.py`
- **Train: one row per document per epoch** (arch v2.1 §6.1) — for a windowed policy, one row per window per epoch, all in that epoch's mode. `sample_modes` draws each train document's regime for each of the 4 epochs (`EPOCH_FILES`), seeded, against the target mix: the `modality_mix` of the adapter's training config (`configs/training/unified.yaml`) when it sets one, else `common.constants.MODALITY_MIX` (**50 / 20 / 30**). The target is recorded as `modality_mix_target` in the manifest, and a training run whose config wants another mix is refused (`training.data_mix.assert_corpus_mix`). It is the only sampling step: v1's `sample_to_target_mix` down-sampler is gone, because running it over epoch rows would drop documents from epochs. Twins over the per-seed cap are dropped before the draw (`cap_twins`), so the mix check measures what trains.
- **Val / test: all 3 variants** per document, so image-only and noisy-OCR accuracy are measured on the full eval population.
- The mix check (`assert_mix_is_close`, 5-point tolerance) is enforced from **900 draws**; below that it warns. At pilot volume (30 train docs × 4 epochs = 120 draws) one regime's share has a ~4.6-point standard deviation, so an enforced check would fail correct corpora.
- `image_only` rows omit the OCR text block entirely and use the image-only system prompt — the explicit "no OCR provided" declaration, not a silently missing field (arch §6).
- `noisy_ocr_image` rows use the **same instruction as `ocr_plus_image`** — the noise lives in the data, not the prompt.

### 4. `data_pipeline/dataset_builder/noisy_ocr_augment.py`
- For `noisy_ocr_image` rows: take the real MinerU OCR and inject **realistic** corruptions — digit↔letter confusions (O/0, l/1, S/5), merged/split table cells, dropped headers, reading-order scrambles — while keeping the **golden JSON as the correct target**. This teaches image-over-OCR arbitration (arch §5, §6).
- Corruptions parameterized and seeded. Mirror MinerU's real failure modes; **random noise teaches the wrong lesson**.
- Tag each corrupted row with the corruption types applied, so IMPL-08's `ocr_arbitration_accuracy` can score specifically on documents where OCR and image genuinely disagree.

### 5. `data_pipeline/dataset_builder/build_jsonl.py`
Orchestrates: for each tenant, split, and doc_type, assemble chat-format rows using the exact data contract (master §9):
- `system` = rendered prompt (schema + modality instruction) via **`common.prompts`** — the same renderer serving and testing use. **Never hand-type the schema per example**; inject it from the registry at build time so a schema change is a one-file edit that regenerates the whole corpus consistently (arch §7).
- `user` = image block(s) (+ OCR text unless `image_only`). **Multi-page documents produce multiple ordered `image` blocks, each followed by that page's own markdown, in one example** (see *Page pairing* below).
- `assistant` = for a flat type (ACORD, Loss Run), the golden JSON via `common.canonical.training_target` (dates written MM/DD/YYYY), including `line_of_business`; for a canonical policy, the window's target (see *Current implementation*).
- Carry `doc_type`, `acord_form`, `lob`, `modality_mode`, `source_id`, `tenant_id`, `split`, `deidentified` on every row, plus `group_id`, `synthetic`, `carrier`, `twin_index`, `render_mode`, `is_scanned` (what the adapter's data mix and render rotation read, IMPL-06), `val_half` on validation rows, `epoch` on train rows, and `sections`, `task`, `window_index`, `window_pages` on a policy window row.
- Write `corpus/{tenant_id}/v{n}/train/epoch_{1..4}.jsonl` and `corpus/{tenant_id}/v{n}/{val,test}/{val,test}.jsonl` to Blob — all doc types the build was given together, in one file (§8.1); the pipeline gives it the adapter's types, today policies only. These are exactly the paths `training/train.py` reads; `tests/test_orchestration.py` asserts that on a real build. Every epoch file holds every train document exactly once (`train_rows_by_epoch` refuses otherwise; a policy's windows may differ between epochs, its documents may not). Rows are written sorted by `(source_id, modality_mode, sections, window_index)` (`write_jsonl`). The adapter's training run reads a filtered view of these files (`training/corpus_view.py`, `train/scope/policy/` and `val/scope/policy/`: policies of one known line only — a package and a policy with no line are left out, IMPL-06), never a corpus of its own. The corpus itself still holds them; its `policy_line_counts` counts a package once for each of its lines.
- Reference images/OCR by Blob URI or a training-time-resolvable path (document the choice; it must resolve identically inside a RunPod pod).

**Required edge cases the corpus must actually contain (arch §7)** — assert coverage and report counts in the manifest:
- Multi-page documents (multiple image blocks in one example)
- Rotated / skewed scans and low-quality faxes
- Handwritten annotations on otherwise digital forms
- Fields legitimately absent — must produce `null`, not a hallucination
- Repeating table rows of variable length (Loss Run claims, ACORD schedule lines)
- **Long multi-page policies**, so the window planner over every page (`policy_windows.plan_windows`, shared with serving, IMPL-11) is tested rather than assumed
- **Confusable co-occurrence** *(master §1.4)* — documents where a canonical field and at least one of its registered confusables both appear on the page (a named insured **and** a certificate holder; an insurer **and** a producer), with the golden JSON resolving them to different canonical keys. Without these the model learns the mapping but never the **boundary**, and will happily return the certificate holder as `insured_name` with high confidence. Track the count in the manifest and treat a zero count as a corpus defect, not an acceptable state.

### 6. `data_pipeline/corpus_manifest.py`
Writes `manifest.json` per corpus version — this is what training and eval read to know exactly what a corpus version contains, and what pins it:
- Example counts per `doc_type` × `split` × `modality_mode`; `source_id` lists per split.
- Split seed, ratios, builder git commit; the whole `split_assignment` (counts by line, train-only lines, held-out carriers, twin cap, `delivered`, `frozen_eval_set`).
- `modality_mix` over train rows and `modality_mix_target`, the mix the modes were drawn against.
- **`mineru_version`**, **`ocr_device`**, and per-document `preprocessing_date` (arch §8a). Device is pinned alongside version because the model learns MinerU's output conventions, and GPU and CPU MinerU may not produce identical markdown.
- **`schema_version`** and **`prompt_template_version`** (arch §7) — a change to either forces a corpus rebuild and a new training cycle.
- `tenant_ids` contributing; `deidentified: false` and `image_redaction: "unresolved"` under Option A (§1).
- **`lob_coverage`** (arch §0b) — computed here, not as a separate module: counts `line_of_business` per doc_type and split, compares against the **≥20%-per-value target**, and writes the actual distribution. Under-coverage emits a **loud warning naming the under-represented values**; it does not block the build, because the remedy is targeted document collection, not a build-time fix.
- **`alias_coverage`** (master §1.4) — same pattern, same module: counts examples per canonical field × observed surface label, read from each label's `field_provenance` (IMPL-04). **Alias coverage matters more than document count** — 95 documents saying *Named Insured* and 5 saying *Applicant* produce a model that is shaky on *Applicant* no matter how large the corpus is. Warn loudly naming variants below a configurable floor (default: fewer than 3 source documents). Non-blocking, for the same reason as LoB.
- **`confusable_example_count`** — how many source documents satisfy the confusable co-occurrence edge case, per canonical field. **Zero is a corpus defect** and warns accordingly.
- Edge-case coverage counts (multi-page, scanned, handwritten, null-field, long-policy).

**Corpus versions are immutable once used for a training run — never edit in place.** New labeled data appends into the next corpus version.

## Constraints
- Zero split leakage — assert no `source_id` appears in more than one split.
- Zero cross-tenant mixing — assert every row in a corpus shares its `tenant_id` prefix. This is the one live tenancy rule, because corpus composition is training data.
- Every row schema-checkable (a flat assistant JSON validates against the doc_type schema, including `line_of_business`; a common-model window target is validated against its window's slice of the model view, and a target that fails raises instead of being taught — `policy_windows._check_writable`).
- Deterministic + seeded; re-running with the same inputs yields byte-identical corpora.
- MinerU version at build time must match the version recorded on the processed documents (IMPL-03 `assert_version_matches`).

## Page pairing and the noise budget

- `SourceDocument.ocr_pages` holds **one markdown string per page**, never a joined blob. The corpus row's user turn is interleaved per page (master §9), so a training row and a serving request — including one window of a long policy — have the same structure.
- `noisy_ocr_image` corruption is budgeted **per document, not per page** (`corrupt_ocr_pages`) — and for a windowed policy, **per window**: each window row gets the budget over its own pages, seeded by window, after the windows are planned on the clean text (otherwise one budget over a 200-page policy left almost every noisy window byte-identical to its clean twin). Calling the single-page corrupter once per page multiplies the noise by the page count: a 20-page policy would take up to 40 corruptions instead of 2, and the "nothing changed" fallback would fire on every page. That is not a louder version of the same signal — a corpus where every page is garbage teaches the model to ignore OCR entirely, degrading the 50% `ocr_plus_image` regime that never sees noise.
- Corruption details record **which page** the noise landed on, so IMPL-08's `ocr_arbitration_accuracy` can match a corruption to the field it affected.

## Acceptance checklist
- [ ] No `source_id` crosses splits (automated assertion).
- [ ] No corpus file contains rows from more than one `tenant_id`.
- [ ] Each train document appears once per epoch file with the drawn modality (the training config's mix, default 50/20/30, across draws); val/test documents carry all three modes.
- [ ] `noisy_ocr_image` rows have corrupted OCR but **correct** golden JSON targets, tagged with corruption types.
- [ ] `image_only` rows contain no OCR text and use the image-only prompt; `noisy_ocr_image` renders the same prompt as `ocr_plus_image`.
- [ ] Corpus rows interleave each page image with that page's own markdown; no two pages' markdown are joined.
- [ ] The noise budget is per document: a 20-page document takes no more corruptions than a 1-page one, and no noisy row is identical to its clean twin.
- [ ] Every flat assistant JSON validates against its schema and contains `line_of_business`; every common-model window target validates against its slice of the model view.
- [ ] A window target cites only the window's own pages; no window holding a value no page prints is built; no coverage row is taught on a multi-window window that shows none of its home pages.
- [ ] Every page of a policy is in a window of every group planned for its line, whatever its length; the declarations group's (and `dtd`'s) first window is pages 1–5 plus the detected declarations page; the corpus holds no `page_select` rows.
- [ ] A drawn split holds one carrier out of train and val per line only where the line's real documents name at least 10 carriers; a line with fewer holds none out and is recorded in `lines_below_carrier_minimum`. A delivered split stays as delivered. On a policy only real carriers count: a placeholder ("N/A", "Unknown", "Unknown Insurance Company", blank) is no carrier ("Other Mutual" is one), a carrier any synthetic family trains is never held out, and a seed family's real carrier still counts toward the 10 (`GroupRecord.holds_real`); a delivered split asked for a hold-out counts and chooses the same way.
- [ ] A gold with no review record (`reviewer_id`, `review_date`; decision 2026-10-22), or a twin whose seed has none, stops the corpus build.
- [ ] On a common-model policy line the corpus build reads a stored gold exactly as the gate scores it — a form's edition moved out of its number (`load_golden_label`) — and on a self-contained line as stored.
- [ ] No `lossrun_totals` example is built while the Loss Run schema declares no `totals`.
- [ ] `corpus_manifest` warns loudly and names under-represented LoB values below the 20% target.
- [ ] `corpus_manifest` reports `alias_coverage` per canonical field × surface label and warns on variants below the floor.
- [ ] `confusable_example_count` is reported per canonical field, and a zero count warns.
- [ ] `manifest.json` counts match actual JSONL row counts, and record `mineru_version`, `schema_version`, `prompt_template_version`, tenant list, de-id status, and edge-case coverage.
- [ ] Rebuild with the same seed produces byte-identical output.

---

## Current implementation (2026-10-21)

A document's `lob` is read through `common.lob.merge_line` (`SourceDocument.__post_init__`): classic auto is
personal auto everywhere.

**Policy windows.** A canonical policy is built as one row per *(section group × page window)* per reading
mode, planned by `policy_windows.plan_windows` — the same function `serving.pipeline` uses, so every served
window is a shape a training row had. **Every page is read, chunk by chunk, never selected pages** (decision
2026-10-19): `policy_windows.pages_to_read` returns every page of the document, whatever its length, with the
declarations page `serving.page_router.plan_pages` detects in the OCR text the row is planned on (the clean
text for `ocr_plus_image` and `noisy_ocr_image` — a noisy window is corrupted after planning — and none for
`image_only`, which has no text to search). There is no page threshold, no routed subset and no `page_select`
row. The groups are `common.schema_sections.groups_for(lob)` over `configs/schema_sections.yaml`:

- **self-contained lines:** `decl`, `arrays`, `lineblk` (every section no other group claims), and `dtd`
  where the line carries it;
- **common-model lines** (`common.schemas.is_common_model`: the seven personal-lines overlays, `gl` and
  `_fallback`, common model 1.1.0): the `common_model` profile — `decl` (document, carrier, producer,
  named_insured, policy, lob_parts, premium, billing), `arrays` (every unit table and every table that refers
  to a unit — coverages, deductibles, interested_parties, locations, buildings, vehicles, drivers, watercraft,
  scheduled_items, rating_exposures, rating_modifiers, underlying_insurance — plus `additional_fields`),
  `forms` (forms_and_endorsements, a job of its own: decision 2026-10-23) and `lineblk` (the remainder:
  general liability's claims_made_terms and countersignature; empty, so not planned, on the other lines).
  `text_sections` is asked of no group. The self-contained lines keep their forms in `arrays` (their prompts
  are frozen), so `forms` is empty on them.

A group with no sections on the line is not planned at all.

Every group reads every page (`common.schema_sections.pages_for(group, pages)`); the page rule — `every`
for `arrays`, `forms` and `lineblk`, `declarations` for `decl` and `dtd` — says only how the first window is formed.
For `decl` and `dtd`, `schema_sections.leading_pages(group, pages, declarations_page=)` gives the first
`pages.declarations_leading` = **5** pages plus the detected declarations page when it falls later; they are
read together in the group's first window (split evenly only if they exceed one window), and every page after
them follows in windows as for the other groups. A long policy therefore has more windows, and the corpus more
rows, than when pages were selected: a corpus built before this change must be rebuilt and the adapters
retrained, and earlier gate numbers are not comparable with new ones (IMPL-06).
`expand_tasks.plan_policy_windows` packs whole page runs while they fit and splits a longer run evenly; on a
common-model line consecutive windows of a split run share `run_overlap_pages: 1` page
(`common.schema_sections.run_overlap`, `_split_overlapping`), so a row printed across the boundary is whole in
one window. A window's capacity is `expand_tasks.pages_per_extraction_call(..., group)`: the group task's
`max_seq_len` minus its reserved `max_output_tokens` (`policy_declarations` 4,096, `policy_schedule` 12,288 in
`configs/shared/sequence.yaml`) minus the sliced prompt, per page at the task's pixel budget plus 700 OCR
tokens. A window whose pages' **own** text would not fit that budget (policy-form wording runs to twice the
700 tokens a page is charged) is split into the fewest windows that fit, as evenly as they allow
(`expand_tasks.fit_windows_to_text`, `page_tokens`, `window_page_room`; decision 2026-10-22): the answer is
reserved whole, as serving reserves it, and each page costs its image plus its marker and OCR text, counted as
the cap check counts a row. A window that fits is left as planned. `image_only` rows carry no text and are sized
by page count alone; a noisy row is planned on the clean text, since its corruption is drawn per window after
the plan. A run split with overlapping windows is refitted as one run, each window again opening on the
last `run_overlap_pages` of the one before, so no page break goes unread (`_fit_overlapping`). The corpus
manifest pins `window_plan_version` = `policy_windows.window_plan_id()`: the rule's version
(`WINDOW_PLAN_VERSION`, now `"3"`) and a fingerprint of everything the planner sizes windows from — the
section map, each policy task's sequence budget and visual tokens per page, the per-page estimates and the
prompt hash — so a budget edit that moves windows changes the pin. It also pins `review_rule_version`
(`corpus_manifest.REVIEW_RULE_VERSION`, now `"2"`: every gold reviewed). Training and packaging refuse a
corpus whose pins differ from the code's (IMPL-13).

**Window targets** (`window_target`): the group's sections, narrowed to values printed on the window's pages.

- A label written for another line's schema is first moved into this line's block
  (`common.canonical.schema_label`, `configs/label_mappings.yaml`).
- `page_ref` is narrowed to the window's own pages, in document page numbers, on a copy (the label is cached
  per document and read by every window and mode). A value printed on pages 1 and 30 is cited `[1]` by the
  window showing page 1.
- `text_sections` is never in a target (`EXCLUDED_FROM_TARGETS`). Each `additional_fields` entry is placed by
  its own `page_ref` — on a common-model line, by its own pages and its value's — trimmed the same way, and
  kept in SPEC_21's shape (label, value, section_hint, page_ref).
- A value with no `page_ref`, in a group read over several windows (`multi_window_sections`, the
  declarations group included), is first given a page by `with_inferred_pages` when the clean OCR prints it
  (at least 4 characters, on word boundaries) on exactly one page. One still without a page is taught in the
  window that takes such values (`PolicyWindowPlan.holds_unpaged`): a group's only window, or the window its
  declarations pages lead (`decl`, and `dtd` where the line carries it) — the one holding the declarations
  page when those pages are split (`_home_window`). In a group the declarations do not lead it is reported
  `unplaced`. A value no page prints (`unprinted_values`) with no page recorded is taught nowhere in a group
  read over several windows; only a group's only window is left out for holding one. Values printed only on pages no window of their group reads are reported `unread`;
  with every page read, that is now only a value citing a page the document does not have.

On a **common-model** line, additionally:

- a label with two parts, or a part of another line, is refused as a mislabel (`with_own_part`; a part read
  as the line, such as classic auto, is rewritten as the line), and `part` is dropped on a single-part policy;
- ids, codes and types (values no page prints) ride only with a row that keeps a printed value on the
  window's pages, so no window is taught a bare row skeleton. A coverage's `form_refs`
  (`common.model_view.PRINTED_STRING_FIELDS`) ride only on windows whose pages print them
  (`printed_pages_in`, looked up once per document over the clean text); an item the text shows on no page
  stays with its row, since OCR misreads form numbers on scans;
- a reference to a table another group reads is left out (`common.schemas.cross_group_references`), the
  window's ids are renumbered from 1 (`common.structural_ids.renumber_structural_ids`), and references to
  rows the window does not hold are counted as `dangling`;
- **orphan rule:** only a printed value identifies a row (`_printed_identifiers`). A fragment that kept none
  of its row's identifiers is left out (`orphaned`) unless it holds identified nested rows or a value no page
  outside the window cites (`_cited_only_here`): a coverage split at a window boundary is taught, a premium
  printed again on a summary page is not;
- **home rule** (`HOME_FIELDS`): in a group read over several windows, a `coverages` row is taught only on
  windows that show one of its home pages — the pages that print its `coverage_name`, a limit's or
  deductible's `amount` or `percentage`, its `premium` or `coinsurance_percent` (nested rows included). A row
  with a home is left out of every other window (`TargetReport.away`, "away from home" in the build notes); a
  row with none stays wherever its values are cited. Windows of policy wording that print only a caption
  ("Each Occurrence Limit") or "Included" are not taught coverage rows;
- the finished target is written in model form (`to_model_target`), narrowed to the window's slice
  (`within_schema`), put in the grammar's key order, and validated against the window's slice of the model
  view (`_check_writable`): a target the decoder could not write raises `CanonicalLabelError`.

**Golds the build reads.**

- **Every gold needs a review record** (decision 2026-10-22; frontier models are never used for gold,
  synthetic or training data, decision 2026-10-19): `pipeline_dag.load_labeled_documents` passes each gold's
  metadata to `export_golden_labels.review_problem`. A gold with no real `reviewer_id` and `review_date`, or a
  twin naming no seed it inherited its record from, stops the build (`PipelineError`: they "have no review
  record", naming one and up to five others). The gold's notes and `draft_backend` decide nothing. The import,
  the pre-upload audit, the preflight, the freeze and the gate refuse the same golds (IMPL-04, IMPL-08,
  IMPL-13).
- **One form-edition convention**: the label rules `prepare_bundles` applies to a common-model gold
  (`corrected_gold` → `label_rules.apply_label_rules`) move an edition printed inside a form number in a shape
  no form number takes (ISO's `AA 99 99 MM YY`, or a month and year in brackets after the number) to the
  form's empty `edition_date`, parsed `MM/YY`, with the number's pages and confidence
  (`label_rules.split_form_edition`); a filled `edition_date` is never overwritten. The import and the
  pre-upload audit refuse a bundle on a common-model line that still carries one (`check_bundle`,
  `import_prepared`) and warn on a number that only may end in one; the rule holds there and nowhere else
  (`label_rules.label_rules_apply`), so a self-contained line's numbers are trained as printed. A gold stored
  before the rule is read under it, not rewritten (`export_golden_labels.load_golden_label` →
  `label_rules.under_edition_rule`), so the build reads it exactly as the gate scores it. This changed training
  targets: bundles prepared before it are prepared again and the corpus rebuilt.

**Label checks before slicing** (`build_jsonl._policy_window_rows`, once per document, on the clean OCR text
whatever the row's mode — the target is the same answer in every mode):

- `label_verification.verified_label`: a value a rule added to a synthetic label (`fideon:filled`) is kept
  where the page prints it, re-paged when exactly one other page prints it, and left out otherwise; with no
  OCR text it is kept and counted unverifiable; given the gold's `doc_type` and line (`doc_type=`, `lob=`) it
  first reads the gold under the form-edition rule where that holds — the gate reads its frozen golds this way;
- **unprinted values**: bundle metadata `unprinted_values` (written by `prepare_bundles`' label rules) lists
  label values no page prints; a window whose sections and pages hold one is **left out whole**
  (`_held_unprinted`) — taught, it is a value to invent; deleted, a printed value the search missed would be
  taught as one to skip.

What every check left out — `unread`, `unplaced`, `orphaned`, `away from home`, windows left out, the dangling
count, the verification counts — is kept per document on `BuildResult.window_notes` and logged.

**Expansion failures.** A document that fails to expand is set aside (`set_aside["expansion"]`), except on a
common-model line, where more than `max_expansion_failures` (default 0) such documents fail the build.
`classify_rows` (the adapter's line question, among every known line) exists, but
`build_corpus(with_classify_rows=True)` is refused until serving asks that question.

**Loss Runs** train as one whole-document `extract` row. `expand_tasks.expand_lossrun` (header, row windows,
totals; arch §7b) has no production caller yet, and its `lossrun_totals` example is built only when the Loss
Run schema declares `totals` (`expand_tasks.lossrun_schema_has_totals`). It declares none today; without the
check every such target was `{}`, which teaches the model to ignore printed totals.

**Blank pages** get one placeholder text in training and serving (`input_builder.EMPTY_PAGE_TEXT`) instead
of an empty block in training and a placeholder at serving.

**Size estimate** (`cap_check.estimate_row`): prompt and OCR text by `estimate_text_tokens` — digits one token
each (Qwen splits numbers into digits), the rest at 3.5 characters per token; pages at the task's
`max_pixels` / 1,024 visual tokens each; the **answer** by `target_tokens` — the base model's own tokenizer
when its files are on the machine (`base_model_dir()` with a `tokenizer.json`, loaded as
`training.length_check` loads it), else the estimate × `TARGET_ESTIMATE_FACTOR` (1.25). A row over its task's
`max_seq_len`, or with an answer over its `max_output_tokens`, sets its **whole document** aside (`budget`);
so does text containing a trainer tag (`trainer_tag`, `common.constants.TRAINER_SPECIAL_TAGS`). The real
tokenizer re-measures every staged row before launch (IMPL-06).

**Split, per line of business** (`split_groups.assign_group_splits`):
- the band (table above) is chosen on the **document type's** volume; placement is by stable hash **within
  each line of business** (`GroupRecord.line`, from the policy's `lob`), so every line lands near the ratio;
- a line with **fewer than 5 documents trains whole** and is listed in `train_only_lines`; a measured line
  with 3+ families always gets at least one val and one test family;
- **any family with a synthetic member stays in train** (the old rule — synthetic only if *every* member
  was — let a mixed family carry generated labels into test);
- **carriers are compared normalised** (`common.normalize.normalize_carrier`: "The Travelers Indemnity
  Company" = "TRAVELERS"; an underscore separates words, so `dryden_mutual` is Dryden Mutual; a name word is
  kept where dropping it would leave a placeholder, so "Other Mutual" stays a carrier; a bundle's own `carrier`
  first, else the label's), so a held-out carrier is not still in train under another spelling;
- **held-out carrier, one per line** (Fideon SPEC_09 amendment item 4; decision 2026-10-19): a line whose real
  documents name at least `MIN_CARRIERS_PER_LINE_FOR_HELD_OUT` (10) carriers holds one carrier out of train
  and val — every real family of it in that line goes to test (`held_out_carriers_by_line`,
  `held_out_source_ids`). It is chosen by `held_out_carrier_per_line(..., min_carriers=)` on the split the hash
  would draw: of the carriers whose removal still leaves the line a train and a val family, the one with the
  fewest train and val documents, ties broken by a seeded hash. A line naming fewer carriers holds none out
  (`CarrierHoldOut.below_minimum`, recorded per doc type as `lines_below_carrier_minimum` with each line's
  carrier count); so does a line with one carrier (`single_carrier_lines`), none named
  (`lines_without_carrier`), or no carrier it can spare (`lines_not_held_out`). This replaces "2 carriers per
  doc type once 6 exist" and takes effect when a corpus is next split (a frozen eval set draws no new test
  documents);
- **on a policy, real carriers only** (decision 2026-10-22): a placeholder carrier
  (`common.normalize.is_placeholder_carrier`, re-exported by `split_groups`: "N/A", "Unknown", "Unknown Insurance
  Company", "Multiple Insurers", "Not Stated", "To be determined", "Undisclosed", "none", "TBD", "various", blank …,
  read on the name's own words, never its key; "Other Mutual" and "Pending Mutual" are carriers) is no carrier — it
  never counts toward the 10 and is never held out; a carrier any synthetic family trains in the line is never
  the one held out (twins train wherever they sit, so its layouts are seen), though a family holding a real seed
  and its twins counts its real carrier (`GroupRecord.holds_real`, set by `pipeline_dag.group_records`, which
  names the family by its real member's carrier); a twins-only family counts for nothing. A delivered split
  asked for a hold-out counts and chooses the same way. ACORD and Loss Runs are read as before;
- `counts_by_line`, `train_only_lines` and the hold-out records are in the manifest's `split_assignment`; the
  run manifest's `split_policy` carries `lines_below_carrier_minimum` too.

**Twin cap** (`split_groups.cap_twins`, drawn and delivered splits alike): at most `MAX_TWINS_PER_SEED` (20)
synthetic twins of one seed (family) per render mode train, chosen by a seeded hash, before modes are drawn;
recorded as `twin_cap` and `twins_dropped`. Real documents, validation and test are never dropped.

Example — 1,920 personal-lines policies: ≈1,523 train / 206 val / 191 test, each measured line near 80/10/10.

**After the eval set is frozen** (IMPL-08; each tenant has its own, `golden-eval-set/{tenant}/`, and only
the building tenant's counts): the build drops the frozen documents **and every document in their
families**, splits new documents into **train and val only** (train:val kept in proportion, no carrier held
out), records the frozen set's held-out carriers on the split (`frozen_eval_set: true`), and warns when a
carrier the frozen set held out — per doc type or per line, by the split or by the delivery; a line may list
several — starts training. Both sides are compared as `normalize_carrier` keys.

**Guards**: render-only and zero-page documents are skipped with a reason; documents read by more than one
MinerU version are refused (`assert_one_mineru_version`); a build with no train rows is refused before
anything is written, and a real (non-dry) build with no val rows — or no test rows while the eval set is not
frozen — is refused too. A real build that sets aside more than 25% of its loaded documents
(`orchestration.preflight.FAIL_REJECTED_SHARE`, measured by `corpus_set_aside`, the figure preflight reads
too) is refused before anything is written. The stage result reports `documents_set_aside` by reason
(`budget`, `trainer_tag`, `expansion`) with the first examples.

**Manifest**: LoB coverage reads each document's line from metadata (canonical policy labels carry none);
the `modality_mix` counts **train rows only** (`modality_mix_basis: "train rows"`), since val/test carry
every mode by design; `modality_mix_target` is the mix the modes were drawn against.

**Line coverage in the manifest**: `lob_coverage` (the ≥20% enum target) counts only labels that carry
`line_of_business` (ACORD, Loss Run); policy lines are counted from metadata in `policy_line_counts`
(canonical schema names). Checking policy lines against the enum raised and failed the whole build.

Only **policies** contribute to `policy_line_counts`; ACORD and Loss Run documents stay in the enum coverage
even when the importer copied their label's `line_of_business` into metadata.

### Delivered split

A delivery split upstream is used as given. `data_pipeline/ingestion/prepare_bundles.py` writes one folder
per document with `metadata.json` (`lob`, `synthetic`, `template_id` = the family the split was drawn by,
`split`, `carrier`, `render_mode`, `unprinted_values`, `held_out_carrier`) from three kinds of delivery:

- fideon_synth's `Train|Val|Test` + `manifest.csv` (`prepare_bundles`);
- a SPEC_21 delivery's `split_manifest.csv` (`prepare_twin_bundles`): each twin becomes a digital and a
  scanned document reading one corrected gold (one document when both renders are the same file), its seed
  is the family, and each seed document is bundled as a real document in its twins' family and split;
- real documents alone, of one line (`prepare_original_bundles`, e.g. the CGL source data with `--lob gl`):
  one family per insured (`_insured_family`), the split drawn by `assign_group_splits` exactly as the build
  would draw it and written into the metadata (`split.csv` lists it for review); a test document of a
  carrier no training document has is marked `held_out_carrier`.

The importer carries `split` into the label metadata and refuses any value other than train/val/test. When
**every** document carries a split (all or none, else the build stops), `pipeline_dag.use_delivered_families`
makes each document's family its source document (`delivered:<template_id>`), `delivered_split_of` refuses a
source in two splits, and `split_groups.assign_delivered_splits` takes the assignment as given (validation
still halved by group; a split with no val, or no test before the eval set is frozen, is refused; after
freezing, no document may be delivered as test). The per-line carrier hold-out
(`held_out_carrier_per_line`: the cheapest carrier whose families can all go to test while the line keeps a
train and a validation family) is **off** for a delivered split (`HOLD_OUT_CARRIER_PER_LINE = False`; pass
`hold_out_carriers=True` to apply it, with the same minimum: `min_carriers`, default
`MIN_CARRIERS_PER_LINE_FOR_HELD_OUT`): the delivery already keeps whole seeds out of training. Documents
delivered with `held_out_carrier` that sit in test are added to `held_out_source_ids`
(`delivered_held_out_ids`), so evaluation reports them apart. Families this pipeline detects by text
similarity that cross the delivered split — several source documents of one carrier share a printed form —
are reported with a count of val/test documents on a form seen in training, not merged.
`assert_synthetic_is_train_only` applies to drawn splits only; under a delivered split the synthetic twins of
held-out sources are evaluated, and the gate reports real documents apart (IMPL-08). The corpus manifest's
split record carries `delivered: true`, and the run manifest a `split_policy` summary (held-out carriers,
lines holding none out, lines below the carrier minimum, frozen set, families moved to test, twin cap).

## Changes since 2026-10-05

- Added the alignment line (commit f5668a4).
- §3: the modality target is the scope's `modality_mix` (default 50/20/30), recorded as `modality_mix_target`
  and checked by training; twins are capped before the draw; windowed policies draw one mode per epoch for
  all their windows.
- §5: assistant target split into flat-type JSON vs policy window target; user turn described as interleaved
  per page; row bookkeeping fields listed (`group_id`, `synthetic`, `carrier`, `twin_index`, `render_mode`,
  `is_scanned`, `val_half`, `epoch`, window fields); sorted output and scope views noted.
- §6: the manifest's `split_assignment` contents and `modality_mix_target` listed.
- Constraints and checklist: common-model targets checked against the model view (`_check_writable`); new
  checks for page-trimmed targets, unprinted values and the coverage home rule.
- Current implementation rewritten for Schema 2 windowing: `decl`/`arrays`/`lineblk` (and `dtd`) groups,
  the common-model profile, 5 leading declarations pages (was 1–3), `run_overlap_pages: 1`, window capacity
  from the group's reservation; target trimming, `additional_fields` placement, `text_sections` exclusion,
  structural ids and `form_refs` placement, renumbering and dangling references, the printed-identifier
  orphan rule with `_cited_only_here`, the `HOME_FIELDS` home rule ("away from home"), `_check_writable`,
  label verification, `with_inferred_pages`, unprinted-value windows left out, common-model expansion
  failures, refused `classify_rows`.
- Size estimate: answers counted by the base model's tokenizer when local, else estimate × 1.25 (was the
  plain estimate).
- Split: drawn-split held-out carriers, the twin cap, per-tenant frozen sets, the MinerU-version guard and
  the >25% set-aside refusal; delivered split extended to SPEC_21 twin bundles, seed documents, insured
  families for real-only deliveries, the per-line carrier hold-out (off by default) and `held_out_carrier`.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx` (with §0e, the decisions of 2026-10-19).
- §1: de-identification is Option A (none), as the architecture records, not "blocked"; options A/B/C listed; "projector LoRA" corrected to the decoder LoRA; the manifest's `deidentified: false` / `image_redaction: "unresolved"` explained under Option A. §6 says the same.
- §5 edge cases and *Page pairing*: long policies test the every-page window planner, not page selection; noisy windows are corrupted after planning.
- Acceptance checklist: every page in a window of every group, the declarations group's first window, no `page_select` rows; the one-carrier-per-line rule from 10 carriers; unreviewed model-drafted golds stop the build; no `lossrun_totals` example while the schema has no totals.
- Current implementation, policy windows: every page read (`policy_windows.pages_to_read`, `schema_sections.pages_for` / `leading_pages`, page rules `every` / `declarations`); no page threshold, routed pages or `page_select`; more windows per policy, so a rebuild and retrain; `unread` now only means a page the document does not have.
- Current implementation, golds: `load_labeled_documents` refuses an unreviewed model-drafted gold (`review_problem`); the form-edition label rule applied by `prepare_bundles` (`split_form_edition`).
- Current implementation, Loss Runs: the `lossrun_totals` example only when the schema declares `totals`.
- Split: one carrier per line, only in a line naming 10 or more carriers (`MIN_CARRIERS_PER_LINE_FOR_HELD_OUT`, `held_out_carrier_per_line`, `below_minimum`, `lines_below_carrier_minimum`), replacing 2 per doc type once 6 exist; the delivered split's opt-in hold-out takes the same minimum.

## Changes of 2026-10-22 (code review)

- Windows sized by their pages' own text (decision 2026-10-22): a window that would not fit its call is split into the fewest that fit, as evenly as they allow (`fit_windows_to_text`); a dense policy is read, never set aside. More, smaller windows on text-heavy policies, so a longer run.
- A value with no `page_ref` is taught in the window the declarations lead, or a group's only window (`PolicyWindowPlan.holds_unpaged`), not dropped from every window.
- The corpus manifest pins `window_plan_version` (`policy_windows.WINDOW_PLAN_VERSION`, `"2"`); training and packaging refuse a policy corpus built with another plan (IMPL-13).
- Golds: every gold needs a review record to train (decision 2026-10-22), a twin its seed's; the build's refusal says they "have no review record".
- Golds: the corpus build reads a stored gold under the form-edition rule where it holds (`load_golden_label`), exactly as the gate scores it; on a self-contained line the edition stays in the number. `label_verification.verified_label` takes `doc_type=` and `lob=` to do the same.
- Split, policies only: a placeholder carrier is no carrier; a carrier any synthetic family trains is never held out; a seed family's real carrier counts toward the 10 (`GroupRecord.holds_real`); a delivered split counts and chooses from real carriers.
- Acceptance checklist: the review-record refusal, the corpus build reading golds as the gate does, real carriers only.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.
- A split keeps its runs' shared pages; the declarations-led window is the one holding the declarations page; declarations values get a page from the text too; a value no page prints no longer drops the declarations window.
- `window_plan_version` is `window_plan_id()` (version `"3"` plus a fingerprint of the planner's inputs); the manifest also pins `review_rule_version`.
- `normalize_carrier` reads an underscore as a space and keeps a name word where dropping it would leave a placeholder; `is_placeholder_carrier` moved to `common.normalize` with more variants.
- The frozen-carrier warning covers carriers the delivery held out, several per line, compared as keys.

## Changes of 2026-10-22 (one adapter)

- No training scopes or layout families (decision 2026-10-22): the modality target is the adapter's training config's; the adapter's view (`train/scope/policy/`, policy rows) is what trains; `classify_rows` asks among every known line, not a family's.
- Code review of 2026-10-23: the adapter's view keeps policies of one known line only; a package and a policy with no line stay in the corpus and out of the view (§5).

## Changes of 2026-10-23 (forms job)

- The forms list is a job of its own on the common-model lines: group `forms` (`forms_and_endorsements`; task `policy_schedule`, page rule `every`, as `lineblk` had). `lineblk` keeps the remainder: general liability's claims_made_terms and countersignature, and nothing on the other lines, which are therefore read in three groups (`decl`, `arrays`, `forms`) with the windows and prompt text they had. General liability is read in four. The self-contained lines are unchanged (forms in `arrays`). The section map is part of the window plan's fingerprint: rebuild the corpus and retrain.
