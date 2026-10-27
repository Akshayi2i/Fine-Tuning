# Common model changelog

`common_model.json` is the shared envelope and building blocks that every
policy_check LOB schema composes (SPEC_21 §4.2). Changes need the approval of
the common-model owner. Adding a block or field is a minor version. Renaming or
removing one is a major version, and every LOB schema that uses it needs a
crosswalk entry.

**What goes in this file:** a field, block or coverage code belongs here when
two or more LOBs print it. Anything printed by one LOB only goes in that LOB's
own block or `coverage_codes.yaml`.

## 1.4.0 — 23 October 2026 (six optional fields printed by two or more lines)

DECISIONS_2026-10-19 rows 115 and 116 (Sagar: release now; the fine-tuning side follows). Additions only: six optional `FieldValue`
properties of existing blocks; nothing renamed, removed, retyped or made required, so every LOB schema pinned at 1.0.0 to 1.3.0 composes
this version unchanged (`tools/check_common_compat.py`: compatible). Evidence rule (above): two or more lines print the concept; V3 counts
are documents / carrier folders of POLICIES-FINAL-V3 for a caption printed in 5 or more different documents of 2 or more carriers (no page
text or value is stored).

| Path | Printed wording | Printed by |
|---|---|---|
| `policy.effective_time`, `policy.expiration_time` | the time of day of the policy period (`12:01 a.m. Standard Time ...`) | gl (25 documents, up to 6 carriers), boppr (the Leatherstocking declarations print it beside both dates), wc (the pre-SPEC_21 wc 1.6.0 had it) |
| `carrier.company_type` | `A Stock Company`, `A Mutual Company` beside the company name | gl (34 documents, up to 9 carriers), wc (7 documents, 4 carriers) |
| `document.copy_designation` | `Agent Copy`, `Insured Copy` | gl (8 documents, 3 to 4 carriers), bopgl (8 documents, 2 carriers) |
| `named_insured.naics_code`, `named_insured.sic_code` | `NAICS`, `SIC`, `SIC Code` | wc (Hartford and AmTrust seeds, 17+ documents), health seeds; gl and BOP declarations (CR-WC-02) |

**Effect:** the fields are in the `document`, `carrier`, `policy` and `named_insured` blocks that every line uses, so every line's prompt
schema, section files, the manifest and the FSM hashes change (as in 1.1.0 for `carrier.financial_rating`). The fine-tuning repository does
not render them yet: `fideon/vlm/tests/conftest.py` renders this repository's side of the three parity modules without exactly these six
fields (and gl from its frozen 3.0.0), so any other drift still fails; the hand-off is `docs/migration/HANDOFF_common_model_1.4.0_fine_tuning.md`.
`tools/prompt_schema_families.py` lists the families whose adapters need a recheck.

Regenerated: `common_model.fields.md`, `policy_check/_fallback.json`, every prompt schema and the manifest, the section files, the FSM hash
fixture. `lob_schema_map.yaml` `common_model_version` is 1.4.0. The lines gl 3.1.0, wc 2.1.0, bopgl 2.1.0 and boppr 2.1.0 pin 1.4.0.

## 1.3.0 — 23 October 2026 (G1 review of POLICIES-FINAL-V3: shared business-liability coverage codes)

DECISIONS_2026-10-19 rows 114 and 115 (Sagar). Additions only: five entries of `fideon:shared_coverage_codes`; no field, block, definition
or enum value changes, nothing renamed, removed, retyped or made required, so every LOB schema pinned at 1.0.0, 1.1.0 or 1.2.0 composes this
version unchanged (`tools/check_common_compat.py`: compatible) and no line's prompt schema, section file or FSM hash moves
(`tools/build_prompt_schemas.py --check`: current; only the manifest's and the fallback's common-model version string changed).

| Code | Printed by | Default limit types |
|---|---|---|
| `X_BI_PD_LIABILITY` | gl, bopgl, boppr (and commercial_fire, property) as `GL_BI_PD`, `BOPGL_LIABILITY`, `BOPPR_LIABILITY` (Coverage L) | per_occurrence, aggregate |
| `X_PRODUCTS_COMPLETED_OPS` | gl, bopgl, boppr (Coverage N) | products_completed_ops_aggregate |
| `X_DAMAGE_TO_PREMISES_RENTED` | gl, bopgl, boppr (Coverage O, fire legal liability) | per_occurrence, per_location |
| `X_PERSONAL_ADVERTISING_INJURY` | gl, bopgl, boppr (Coverage P) | per_occurrence, aggregate |
| `X_LIQUOR_LIABILITY` | gl (`GL_LIQUOR_LIABILITY`), bopgl (`BOPGL_LIQUOR_LIABILITY`) | per_occurrence, aggregate |

Requests applied: CR-GL-03 follow-up, CR-BOPGL-01, CR-BOPPR-01. The three liability lines keep their own codes in this release: renaming a
code is a major version of the line (SPEC_21 4.1) and the registry entries' declared paths, golds and varmaps name the old codes. L1 matches a
printed name to a line's own code first (`fideon/l1/spec21.py` `_codes`), so no row of those lines changes. Aliases are exact-name only (a name
already claimed by another shared code, or by a line's own file, is not repeated). The moves to the shared codes (gl 4.0.0, bopgl 3.0.0,
boppr 3.0.0, each with crosswalk entries) are for the project owner to schedule together.

**Proposals NOT taken** (from the gl and wc review, `gl_wc_common_proposals.md`): the officer inclusion / exclusion schedule (the captions
printed by gl and bopgl are policy wording and an ACORD certificate field, so only wc prints the schedule), `Deductible.basis` (the captions
found are deductible amounts, not a basis), the embedded certificate insurers, the terrorism disclosure block (statutory boilerplate; its
premium already has `X_TERRORISM`), and the shared codes for accounts receivable, valuable papers, employee dishonesty, debris removal and
spoilage (renames for bopgl and boppr, and a name match that would change what L1 codes in other lines; they stay counterpart codes between
bopgl and boppr). Single-line candidates (employee count on a rating exposure, rating-modifier premium, commission percentage) stay
`additional_fields`.

Regenerated: `common_model.fields.md`, `policy_check/_fallback.json` (version strings only), `prompt/manifest.json`. `lob_schema_map.yaml`
`common_model_version` raised to 1.3.0. The `fideon/schemas/tests/test_common_model.py` version assertion follows. The FSM hash fixture moves
only for bopgl and boppr (their own 2.1.0 releases).

## 1.2.0 — 15 October 2026 (value states and absence reasons)

From the SPEC amendment of 15 October 2026 ("Explainable gaps, reader failures,
L1 grounding, visible text, value states, registry validation"), items 1 and 5.
Additions only: two string-enum `$defs` that no LOB schema property references,
so every LOB schema pinned at 1.0.0 or 1.1.0 composes this version unchanged
(`tools/check_common_compat.py`: compatible) and the prompt schemas, the
manifest and the FSM hashes are byte-identical
(`tools/build_prompt_schemas.py --check`: 25 of 25 current). LOB schemas keep
their pin.

| Definition | Values | Where it is used |
|---|---|---|
| `ValueState` | `amount`, `zero`, `included`, `excluded`, `not_covered`, `see_form`, `see_schedule`, `statutory`, `unlimited`, `not_applicable`, `not_stated` | the envelope's `value_states` (canonical path -> state, and the form a `see_form` names); interpreted only by `fideon/schemas/value_states.py` |
| `AbsenceReason` | `not_printed`, `page_not_read`, `page_unreadable`, `refers_to_other_document`, `not_applicable`, `schedule_not_read` | the envelope's `field_gaps` (canonical path -> why the value is missing) |

The FieldValue leaf is unchanged: `parsed` of a MoneyValue stays a number or
null. A FieldValue-level state is open item O8 of the amendment (a 2.0 change,
once the fine-tuning repository's model view can carry it).

Regenerated: `common_model.fields.md`, `policy_check/_fallback.json` (its
version strings only; its prompt schema does not change). `lob_schema_map.yaml`
`common_model_version` raised to 1.2.0.

## 1.1.0 — 8 October 2026 (change requests from gl; accepted by the owner)

The first batch of change requests under SPEC_21 v0.9 §4.2.7, from
`policy_check/CHANGE_REQUESTS_gl.md`, as the common-model owner accepted it:
one minor release, so each adapter is rechecked once. Additions only: no field
is renamed, removed, retyped or made required, so every LOB schema pinned at
1.0.0 still composes this version unchanged
(`tools/check_common_compat.py`: compatible).

Release order the owner set:
1. the CR-GL-19 checks;
2. a short wait for auto, wc and property findings that would affect CR-GL-16 or
   add requests;
3. this release;
4. gl 3.0.0 on it;
5. the CGL seed golds against gl 3.0.0.

| Request | Decision | Change |
|---|---|---|
| CR-GL-02 | accepted | `RatingExposure.rating_component`, optional, enum `RatingComponent`: `premises_operations` / `products_completed_operations`. A classification whose schedule prints an `All Other` and a `Pr/Co` rate and premium is two `rating_exposures[]` rows with the same `class_code`, `class_description` and `location_ref`, one per column. A row without it keeps its 1.0.0 meaning (one rate). 8 or more carrier folders print the split. |
| CR-GL-12 | accepted | `carrier.financial_rating` (FieldValue; aliases `A.M. Best Rating`, `AM Best Rating`, `A.M Best Rated`) and `premium.minimum_earned_percent` (PercentValue; alias `Minimum Earned Premium Percentage`). `premium.minimum_earned` now says it holds an amount and points a percentage to the new field. Evidence: CGL 12 documents in 4 folders (rating) and 17 in 3 (percentage). The Sterling flood registry also maps a printed `Minimum Earned Premium Percentage` (to `additional_fields` today), a second line for the percentage. |
| CR-GL-17 | accepted, parsed by code | `Form.edition_date` carries `fideon:code_parsed: MM/YY`. The pipeline derives `parsed` from `raw` (`04 13`, `0413`, `04-13`, `4/2013` all give `04/13`; null when the print has no month and year) in `schema_registry.conform_document`, which L1, L3 and the delivered envelope all pass through. The model's description is unchanged, so this request changes no prompt (the 8 October decision on code-parsed values, `HANDOFF_parsed_agreement_lanes.md`). |
| CR-GL-03 | accepted; homeowners left alone | Shared codes `X_ADDITIONAL_INSURED`, `X_PRIMARY_NONCONTRIBUTORY`, `X_WAIVER_OF_SUBROGATION` (no default limit type: an endorsement-coverage usually prints no limit). gl renames to them in 3.0.0. Homeowners keeps `HO_ADDITIONAL_INSURED` until its own next major version, so the personal-lines golds do not change. |
| CR-GL-18 | accepted | `lob_schema_map.yaml`: gl `routing_key: gl` (was `cgl`). SPEC_06 §9a has keyed the line by its L1 code since 5 October and has no `cgl` key. |
| CR-GL-15 | decided: `gl` | A package part's `lob` is its L1 code (`gl`), which is also the schema id and routing key. The SPEC_21 §4.2.3 and §4.2.6 examples that wrote `cgl` are corrected. No model change. |
| CR-GL-06 | confirmed | The gl `blocks_used` edit below, released with this version. |
| CR-GL-16 | held | `Deductible.basis` waits until the batch is cut: one binder of one carrier prints it, too little training signal. The auto and wc drafters look for it; it joins if a second line prints it. |
| CR-GL-19 | accepted, built first | `tools/check_lob_defs.py`, `tools/check_common_compat.py`, `tools/prompt_schema_families.py` with `config/canonical_schema/prompt/manifest.json` (written by `tools/build_prompt_schemas.py`), `CONTRIBUTING_common.md`; run in CI (`ci-pipeline-api.yml`, "Common-model change control"). No schema change. |

**Map:** `lob_schema_map.yaml` `version` and `common_model_version` are 1.1.0.

**Regenerated:**
- `common_model.fields.md`, which now shows a code-parsed field's layout in its type column;
- `policy_check/_fallback.json`;
- every prompt schema and the manifest;
- `common/examples/homeowners_minimal.json` (the form's edition `parsed` is `05/11`).

CR-GL-12 reaches every line's prompt schema and CR-GL-02 gl's only. Under §4.2.7 item 5 every adapter whose
prompt schema changed is rechecked. That includes the personal_lines adapter if it has already been trained;
its training status is for the developer to confirm. `tools/prompt_schema_families.py` lists the families.

**Code changes that go with the release:**
- `fideon/vlm/model_view.py`: the bundle step, ported from the fine-tuning
  repository, refused a LOB schema pinned at any version other than the loaded
  common model's. It now accepts the same major version with a pin no newer than
  the model, as `SchemaRegistry` does. Without this every prompt build and L3
  read failed on the lines pinned at 1.0.0. The fine-tuning repository
  needs the same change when it takes 1.1.0.
- `fideon/l1/spec21.py`: L1 dropped a coverage name two codes claimed, so the new
  `X_` codes would have unmatched homeowners' `Additional insured` rows. A name
  the line's own `coverage_codes.yaml` claims now wins over a shared code. This
  also settles four names that clashed before 1.1.0 and were unmatched:
  `Additional PIP` and `OBEL` (personal_auto), `Optional Basic Economic Loss`
  (personal_auto, motorcycle, recreational_vehicle) and `Liability` (homeowners,
  now `X_PERSONAL_LIABILITY`).
- `fideon/schemas/schema_registry.py`: the MM/YY edition parser (CR-GL-17).

**FSM cache:** `vlm_server/fsm/tests/fixtures/schema_hashes.json` regenerated: every
`policy_check` hash moved, so the deployed FSM cache is invalidated on the next
deploy.

**Fine-tuning parity:** the recorded fixtures in `fideon/vlm/tests/fixtures/`
(`finetuning_model_view.json`, `finetuning_policy_input.json`,
`finetuning_section_windows.json`, `finetuning_gl_derived.json`) hold the
fine-tuning repository's rendering of common model 1.0.0 (and gl 2.0.0). They
were not re-derived here: that would compare this repository with itself and
hide the real gap. Until that repository takes 1.1.0, its training prompts
differ from the prompts served here, and the parity tests fail. They pass again
once the fixtures are re-recorded there (`record_finetuning_gl.py`,
`record_finetuning_windows.py`).

**Not in this release** (still open in `CHANGE_REQUESTS_gl.md`):
- CR-GL-16 (held, above).
- The aliases offered in CR-GL-04 and CR-GL-09. `Premium Basis` is already an alias of
  `rating_exposures[].exposure_basis`, and CR-GL-09 offers it for `exposure_amount`.
- The decisions asked in CR-GL-07, -08, -10, -11, -13 and -14.

## lob_schema_map.yaml — released in 1.1.0

**14 October 2026 (builders_risk):** the builders_risk entry's `schema_lob` is `builders_risk`, its `blocks_used`
`[locations, buildings, countersignature]`, the blocks the new `builders_risk.json` 3.0.0 (SPEC_21 draft) composes, and
`source_pdfs` is `{folder: bldrk, carriers: 3, files: 5}`. `lob_code`, `coverage_code_prefix` (the codes use `BR_`,
`CHANGE_REQUESTS_builders_risk.md` CR-BR-01), `routing_key` and `seeds` are unchanged. `common_model.json` is
unchanged.

**14 October 2026 (flood):** the flood entry's `blocks_used` is `[locations, buildings, countersignature]`, the
blocks the new `flood.json` 2.0.0 (SPEC_21 draft) composes; `source_pdfs` (`{folder: flood, carriers: 3, files: 34}`),
`lob_code`, `coverage_code_prefix` and `seeds` are unchanged (the codes use `FL_`, `CHANGE_REQUESTS_flood.md`
CR-FL-01). A comment records what the schema was built from. `common_model.json` is unchanged.

**14 October 2026 (directors_officers):** the directors_officers entry is unchanged in its values; its comment records
that `directors_officers.json` 2.1.0 adds three coverage-part codes and a 9th seed (filed in the epli folder,
`CHANGE_REQUESTS_directors_officers.md` CR-DO-09). `common_model.json` is unchanged.

**10 October 2026 (commercial_fire):** a new entry for the new L1 code `commercial_fire` (Commercial Fire), added to
`config/lob_registry.yaml` by the project owner's ruling of 10 October 2026 (CR-BOPPR-06: Leatherstocking's 'Commercial Fire
and Liability' product is a line of its own). It reads the new `commercial_fire.json` 1.0.0 (SPEC_21 draft; property_pkg;
`blocks_used: [locations, buildings]`), `source_pdfs: {folder: TBD, carriers: 1, files: 7}` (the seven documents are filed
in three folders of one carrier), `lob_code` and `coverage_code_prefix` TBD (codes use `CF_`,
`CHANGE_REQUESTS_commercial_fire.md` CR-CFIRE-02). The map now has 49 entries; the hold-out open item names the
line (one carrier). `common_model.fields.md` regenerated. `common_model.json` is unchanged.

**10 October 2026 (auto):** the auto entry's `blocks_used` adds `countersignature`, the blocks the new
`auto.json` 2.0.0 (SPEC_21 draft, commercial auto) composes: `[locations, vehicles, drivers, rating_exposures,
rating_modifiers, countersignature]`. `source_pdfs` (`{folder: autob, carriers: 5, files: 11}`), `lob_code` (AU),
`coverage_code_prefix` (AU_) and `seeds` are unchanged. `common_model.json` is unchanged; auto's requests to the
owner are in `policy_check/CHANGE_REQUESTS_auto.md` (CR-AU-01 to CR-AU-04).

**10 October 2026 (property):** the property entry is unchanged in its values (`blocks_used: [locations, buildings,
rating_modifiers]`, `source_pdfs: {folder: prop, carriers: 3, files: 11}`, `lob_code` PR, prefix `PR_`); a comment
records what `property.json` 2.0.0 (SPEC_21 draft) was built from. `common_model.json` is unchanged.

**10 October 2026 (health):** the health entry's `blocks_used` is `[benefit_plans]`, the block the new
`health.json` 1.0.0 (SPEC_21 draft; the line had no schema) composes, and `source_pdfs` is `{folder: health,
carriers: 3, files: 7}`. The L1 code `statutory_disability` now reads `health.json` (`schema_lob: health`, as
`watercraft` reads `ocean_marine`): the corpus files the New York statutory disability (DBL) and paid family leave
(PFL) policies under health, and L1 keeps routing them as `statutory_disability` (CHANGE_REQUESTS_health.md
CR-HEALTH-01). `common_model.json` is unchanged.

**10 October 2026 (bopgl):** the bopgl entry's `blocks_used` is `[locations, buildings]`, the blocks the new
`bopgl.json` 2.0.0 (SPEC_21 draft) composes, and `source_pdfs` is `{folder: bopgl, carriers: 5, files: 13}` (the
single-LOB folder; package folders not counted). `lob_code`, `coverage_code_prefix`, `routing_key` and `seeds`
are unchanged (owner to assign, `CHANGE_REQUESTS_bopgl.md` CR-BOPGL-03). `common_model.json` is unchanged.

**10 October 2026 (inland_marine):** the inland_marine entry's `blocks_used` adds `countersignature`, the blocks
the new `inland_marine.json` 2.0.0 (SPEC_21 draft) composes: `[locations, scheduled_items, countersignature]`.
`source_pdfs` (`{folder: inmrc, carriers: 4, files: 10}`), `lob_code`, `coverage_code_prefix`, `routing_key` and
`seeds` are unchanged (owner to assign; the codes use `IM_`, `CHANGE_REQUESTS_inland_marine.md` CR-INMRC-01).
`common_model.json` is unchanged.

**11 October 2026 (professional_eo):** the professional_eo entry is unchanged in its values (`blocks_used:
[claims_made_terms]`, `source_pdfs: {folder: eo, ...}`, `lob_code` and `coverage_code_prefix` TBD); a comment
records that `professional_eo.json` 2.0.0 (SPEC_21 draft) was built from the pl folder's 16 single-LOB declarations
and that its codes use `PL_` (`CHANGE_REQUESTS_professional_eo.md` CR-PL-01). `common_model.json` is unchanged.

**12 October 2026 (professional_eo):** the professional_eo entry's values are unchanged; its comment records that
`professional_eo.json` 2.0.0 now also has 13 seeds of the eo folder (CR-PL-05 for the two that are not).
`common_model.json` is unchanged.

**12 October 2026 (directors_officers):** the directors_officers entry is unchanged in its values (`blocks_used:
[claims_made_terms, insuring_agreements]`, `source_pdfs: {folder: do, carriers: 7, files: 9}`, `lob_code` and
`coverage_code_prefix` TBD); a comment records that `directors_officers.json` 2.0.0 (SPEC_21 draft) was built from 8
of the folder's 9 documents and that its codes use `DO_` (`CHANGE_REQUESTS_directors_officers.md` CR-DO-01).
`common_model.json` is unchanged; `common_model.fields.md` regenerated.

**12 October 2026 (surety):** the surety entry's `blocks_used` is `[bond]`, the block the new `surety.json` 4.0.0
(SPEC_21 draft) composes, and `source_pdfs` is `{folder: sure, carriers: 1, files: 4}`. `lob_code`,
`coverage_code_prefix` (the codes use `SU_`, `CHANGE_REQUESTS_surety.md` CR-SU-01), `routing_key` and `seeds` are
unchanged. `common_model.json` is unchanged; `common_model.fields.md` regenerated.

**11 October 2026 (cyber):** the cyber entry is unchanged in its values (`blocks_used: [claims_made_terms,
insuring_agreements]`, `source_pdfs: {folder: cyber, ...}`, `lob_code` and `coverage_code_prefix` TBD); a comment
records that `cyber.json` 2.0.0 (SPEC_21 draft) was built from 9 of the folder's 11 documents and that its codes use
`CY_` (`CHANGE_REQUESTS_cyber.md` CR-CY-01). `common_model.json` is unchanged; `common_model.fields.md` regenerated.

**11 October 2026 (umbrella):** the umbrella entry's `blocks_used` adds `countersignature`, the blocks the new
`umbrella.json` 2.0.0 (SPEC_21 draft) composes: `[underlying_insurance, attachment, countersignature]`. `source_pdfs`,
`lob_code` (UM), `coverage_code_prefix` (UM_) and `seeds` are unchanged. `common_model.json` is unchanged.

**10 October 2026 (boppr):** the boppr entry's `blocks_used` is `[locations, buildings]`, the blocks the new
`boppr.json` 2.0.0 (SPEC_21 draft) composes, and `source_pdfs` is `{folder: boppr, carriers: 3, files: 5}` (the
single-LOB folder; package folders not counted). `lob_code`, `coverage_code_prefix`, `routing_key` and `seeds`
are unchanged (owner to assign, `CHANGE_REQUESTS_boppr.md` CR-BOPPR-03). `common_model.fields.md` regenerated.
`common_model.json` is unchanged.

**10 October 2026 (wc):** the wc entry's `blocks_used` adds `countersignature`, the blocks the new `wc.json`
2.0.0 (SPEC_21 draft, workers compensation) composes: `[locations, rating_exposures, rating_modifiers,
countersignature]`; `wc.json` also defines one LOB-local block (`wc`, `$defs/WorkersCompensationTerms`; passes
`tools/check_lob_defs.py`). `source_pdfs`, `lob_code` (WC), `coverage_code_prefix` (WC_) and `seeds` are unchanged.
`common_model.json` is unchanged; wc's requests to the owner are in `policy_check/CHANGE_REQUESTS_wc.md`.

**8 October 2026 (layout families from SPEC_06 §9a / SPEC_09 §2.1a; to be confirmed by the common-model
owner):** the 13 L1 codes the spec table assigns but the map still had at `TBD` now carry their family and
adapter: `personal_property` personal_lines; `product_liability`, `construction` casualty_fleet;
`builders_risk`, `commercial_package`, `equipment_breakdown`, `agriculture_farm` property_pkg (AGLIA);
`medical_malpractice`, `management_liability` exec_specialty; `annuities`, `long_term_care`, `disability`,
`statutory_disability` benefits_misc. Only aviation, energy, environmental, reinsurance and title stay
without an adapter (base model + `_fallback`). `schema_lob`, `lob_code` and prefixes stay `TBD`: none of
these lines has a SPEC_21 schema, so they are still read against `_fallback` until one exists. Open: the
SPEC_06 amendment of 6 October 2026, item 4 keeps the L1 code `ocean_marine` (commercial hull and cargo) on
property_pkg, while the 3 October decision above made that code the personal boat line (personal_lines);
left as it is pending the owner's decision. `common_model.json` is unchanged.

**8 October 2026 (project-owner approval, CR-GL-06):** the gl entry's `blocks_used` is now
`[locations, rating_exposures, claims_made_terms, countersignature]`, the blocks `gl.json` composes:
no CGL document of either corpus prints a rating modifier, and CGL documents print claims-made terms
(a retroactive date) and a countersignature. A comment records that gl 1.1.0 also reads the
`policy_CGL` corpus. `common_model.fields.md` regenerated (`scripts/build_common_model_fields_md.py`):
`rating_modifiers` is no longer listed as used by gl; `claims_made_terms` and `countersignature` are.
`common_model.json` is unchanged.

**4 October 2026 (project decision):** classic auto shares one canonical schema with personal_auto.
The L1 code `classic_auto` keeps routing "Classic Automobile" documents (`config/lob_registry.yaml`), and
its `schema_lob` is now `personal_auto` (batch code PA, prefix `PA_`, the personal_auto blocks, seeds in
`autop`), as `watercraft` reads `ocean_marine`. The personal_auto Phase 2 schema covers classic-auto
concepts (guaranteed / agreed value as `vehicles[].stated_amount`, spare parts, collector-vehicle
coverages) and its Phase 3 crosswalk migrates `policy/hagerty/classic_auto.yaml` and
`policy/essentia/classic_auto.yaml`. Until that lands, the legacy `policy_check/classic_auto.json` still
takes precedence in SchemaRegistry (a code's own file wins over its `schema_lob`); it is removed in the
same change that lands the personal_auto SPEC_21 schema (SPEC21_TEAM_PROMPT.md §7.3). No common-model
field changes.

## 1.0.0 — 3 October 2026 (field list approved by the project owner)

**lob_schema_map.yaml, 3 October 2026 (project-owner decisions):**
- Watercraft is part of the ocean_marine line, not a separate line. The `ocean_marine` entry now
  carries the personal boat line (schema `ocean_marine`, `personal_lines`, batch code OM, prefix `OM_`,
  seed folder `boat`); the L1 code `watercraft` still routes to the same schema. The open item
  "ocean_marine (L1 code) has no schema" is closed.
- The seed PDFs under `synth_core/fideon/synth_core/data/source data/` are the de-identified seeds
  (SPEC_21 D2).
- The common-model field list is approved.

This is the first version for LOB developers to build on, delivered as the
Common Model Developer Brief asks. LOB developers start their step 3 once the
project owner approves the field list and the open items in
`lob_schema_map.yaml`.

**Files:**
- `common_model.json`
- `common_model.fields.md`, generated by
  `scripts/build_common_model_fields_md.py` from the JSON and the map
- `lob_schema_map.yaml`: one entry per L1 code (48), with open items
- `examples/personal_lines_example.schema.json`
- `examples/homeowners_minimal.json`
- `test_report.txt`: the `-v` run of `fideon/schemas/tests/test_common_model.py`

**HANDOFF 1.0.0 fixes (3 October 2026):**

- **Item 1 (D-A, D-B):** `lob_registry.yaml` is replaced by
  `lob_schema_map.yaml`, keyed by the 48 codes of `config/lob_registry.yaml`
  in that file's order.
  - `watercraft` reads against the `ocean_marine` schema.
  - The L1 code `ocean_marine` (commercial hulls) has no schema yet.
  - 20 codes have no layout family.
  - SPEC_06 `group` is an open item.
  - The columns `source_pdfs` and `holdout_carrier` from the replaced file
    are kept.
  - `fideon/schemas/schema_registry.py` reads the map's `schema_lob` column.
    Without this, a code whose schema has another name (`watercraft`) fell
    back to `_fallback`, because no `watercraft.json` exists.
- **Item 2:** the test file reports into `test_report.txt`. The "every date
  leaf rejects 2025-06-04" check now swaps every date in the example one at
  a time; the example gained a print date and a billing due date for it.
- **Item 3:** `field_inventory.csv` was built, and its gap list approved
  (3 October 2026).
  - **Source:** 26,175 rows. 5,077 come from the 113 personal-lines PDFs in
    `synth_core/fideon/synth_core/data/source data/`: every "Label: value" and every table column
    header, with value type, page and carrier. 21,098 come from the
    captions and `canonical_path` values of the 292 policy registry YAMLs.
  - **Deviations from the HANDOFF prompt:**
    - The seed PDFs were read in place from their folder, not copied to
      `data/seeds/`; that folder holds each seed's gold, varmap and review.
    - A `canonical_path` column was added.
    - Labels that carry a printed date were dropped: a committed file
      derived from the seeds holds labels only, never policy data.
  - **Gap rule:** 2 or more LOBs and 2 or more carriers. Duplicate carrier
    spellings were merged ("All State" and `allstate`), and
    `ocean_marine`/`watercraft` counted as one boat line. 241 candidate gaps
    became 145 after the approved changes; the rest are listed below.
  - **New fields:**
    - `policy.policy_type`: 14 LOBs, 48 carriers.
    - `policy.original_inception_date`: 3 LOBs, 3 carriers.
    - `Vehicle.engine_displacement`: 3 LOBs, 2 carriers.
    - `producer.web_address`: 2 LOBs, 2 carriers.
  - **Garaging:** option (a). A vehicle's printed garaging ZIP or state is a
    `locations[]` entry, which `vehicles[].garaging_location_ref` points to.
    The labels "Garaging Address", "Garaging ZIP Code", "Garaging State" and
    "Garage" are on `locations[].address`.
  - **About 90 printed labels added to existing fields:** policy number and
    dates, rating state, transaction, carrier, producer, insured, premium
    total and change, fees, discounts, payment method, vehicle, driver,
    location, building, valuation, forms. Also added to the blocks:
    coverages, vehicles, interested parties, forms. And to the shared codes
    `X_VEHICLE_LIABILITY`, `X_MED_PAY` and `X_PIP`.
  - **Not added:**
    - Printed by one carrier only: Number of Times Renewed, Tier, Interest
      Type, Pay in Full. "Policyholder Since" was kept only as a label.
    - Boat-only, so they belong in the watercraft LOB block: Total
      Horsepower, Propulsion Type.
    - "Basement Type": its printed values are not yes/no.
    - Policy-condition headings: go to `text_sections`.
    - Generic words and letter text.
    - Coverage-specific limits and deductibles: go to each LOB's coverage
      codes.
- **Item 4 (reversed):** the 419 source PDFs were moved from
  `synth_core/fideon/synth_core/data/source data/` to
  `data/raw/source_pdfs/`. On 3 October 2026 they were moved back to
  `synth_core/fideon/synth_core/data/source data/` at the developer's
  request, so item 4 is open again.
  - The PDFs are not in Git there either: the root `.gitignore` ignores
    every `*.pdf`.
  - **DVC tracking was not done.** DVC is not installed here.
  - **Correction (3 October 2026):** the HANDOFF called these PDFs "not
    de-identified" and "not seeds". The developer confirmed they are the
    seed data, which SPEC_21 decision D2 states arrive de-identified, so
    they can be tracked through DVC as `SYNTHETIC_DATA_PROCESS.md`
    describes.
  - None of them was ever committed, so there is no Git history to clean.
- **Item 5:** the brief, the review and the HANDOFF moved to
  `Documentation/Synthetic data generation docs/`.
- **Items 6–7 (not done):** wait for `synth_core_0.1.1.patch`, which is not in
  the repository.

**Generic printed labels removed (3 October 2026):** L1 matches aliases against
the captions printed on the page, so a one-word label claims every value
printed under it.

| Field | Removed | Why |
|---|---|---|
| `countersignature.date` | "Date" | Replaced by "Countersignature Date" and "Date Countersigned". |
| `carrier.address` | "Mailing Address" | It is the insured's address, and the label already belongs to `named_insured.mailing_address`. |
| `carrier.claims_email` | "Email" | Would catch the insured's or the agent's email. |
| `carrier.claims_address` | "Claims" | Heads claim history and claim instructions too. |
| `carrier.name` | "Company" | On umbrella policies it heads the underlying insurer column. |

A test now rejects any alias that is a single generic word ("Date", "Name",
"Email", "Claims", "Company", "Number", "Amount" and a few more).

**Content:**
- The `FieldValue` leaf and typed variants of it.
- Every envelope block (SPEC_21 §4.2.1).
- Every building block in the §4.2.2 tables, including those no
  personal-lines LOB uses.
- `lob_parts`, `additional_fields` and `text_sections`.
- 19 shared `X_` coverage codes.
- Four default audit rules.

The fields and codes chosen by the two-LOB rule are listed under the 0.2.0
and 0.3.0 drafts below.

**Changed from the 0.3.0 draft to meet the brief:**

| Brief | Change |
|---|---|
| Rule 1 | `BooleanValue` renamed `BoolValue`. |
| Rule 7 | Address `line_1` and `line_2` renamed `street` and `street_2`. synth_core reads `named_insured.mailing_address.street`. |
| Rule 4 | `attachment.retained_limit` renamed `retained_amount`; the printed label "Retained Limit" stays as an alias. It is an amount the insured retains, not a limit of the policy. |
| Rule 6 | `provenance` closed, listing the keys `synth_core/engine.py` writes plus `provenance`, the free-text note on a seed gold. |
| Rule 8 | `fideon:tier` is now only `core` or `overflow`. `text_sections` is marked by `fideon:fsm_exclude` alone. |
| Deliverables | Every field has a meaning (`description`). The build fails if one does not. |

**L1 registry check:** all 293 policy registry YAMLs were read for the table
columns they map (vehicle, endorsement and schedule columns). 44 column fields
are used by 2 or more LOBs; 35 already had a home.
- Two needed only a printed label: "Base Form" (on
  `lob_parts[].coverage_part_form`) and "Drivers" (on `drivers[].name`).
- "Limit / Amount" on a forms schedule is not a field: a form that carries a
  limit is recorded as a coverage whose `form_refs` names the form, so limits
  stay in `coverages[].limits[]`.
- The endorsement obligation columns (duty, mandatory, timeframe) describe
  wording inside an endorsement and belong in `text_sections`.
- The rest are printed by one carrier only.

**Changes from the SPEC_21 review (Lakshman, 3 October 2026):**

- **#19** — `part` added to every risk unit (locations, buildings, vehicles,
  drivers, watercraft, scheduled items). Coverages, forms and premium items
  already had it. Without it, a package policy cannot be split per LOB.
- **#6** — "at least one coverage with a limit" became the default audit rule
  `len(coverages[].limits[]) >= 1`; a mandatory-field list cannot express it.
- **#12** — `text_sections` is the only open-ended map, and it is
  FSM-excluded. `additional_fields` stays in the training target (SPEC_21
  §10.2): it is a list of label and value pairs, not a map.
- **`lob_registry.yaml`** (since replaced by `lob_schema_map.yaml`) gains
  three fields per LOB:
  - `l1_code`: the pipeline's LOB code, which names the schema file;
  - `source_pdfs`: the folder, carriers and files for the LOB;
  - `holdout_carrier`: TBD (#22), plus open items for the LOBs that cannot
    hold a carrier out.
- **Loader (`fideon/schemas/schema_registry.py`):**
  - A SPEC_21 LOB schema is no longer merged with the old `_common.json`
    (#17). The common-model definitions it uses are inlined into it, so every
    reader gets one self-contained schema.
  - Loading is refused when the common model is missing, when its version
    does not match, when a `$ref` points to an unknown file, or when a LOB
    definition shadows a common one.
  - The FSM generation schema now drops `fideon:fsm_exclude` properties (#12).
  - Mandatory fields written as leaf paths and selectors (`carrier.name`)
    are now resolved by the AuditGate and by L1's coverage rate. Before, the
    first SPEC_21 schema would have failed every document.

**Known departures from the brief:**
- **Two limit fields outside `coverages`.** `underlying_insurance[].limits[]`
  stays because SPEC_21 §4.2.2 puts it there: the limits belong to the
  underlying policy, not this one. `claims_made_terms.defense_within_limits`
  is a yes/no flag, not an amount. The test lists both as named exceptions.
- **The examples reference `../common_model.json`, not
  `../common/common_model.json`.** They live in `common/examples/`, and
  references resolve by file location. A real LOB schema in
  `policy_check/<lob>.json` uses `../common/common_model.json`.
- **The test lives in `fideon/schemas/tests/test_common_model.py`, not a
  top-level `tests/`.** CI runs `pytest fideon`, so a test anywhere else would
  never run.
- **The seed folder the brief names (`Synthetic Data Generator
  Test/Input/`) is not in the repository.** `field_inventory.csv` was built
  from the source PDFs in `synth_core/fideon/synth_core/data/source data/` and the registry YAMLs
  instead (see HANDOFF item 3 above).

## 0.3.0 — 3 October 2026 (draft, not yet approved)

This version comes from a gap scan. Every printed label (text followed by
":") was collected from the first 4 pages of all 419 source PDFs. Labels used
in 2 or more LOB folders, by 2 or more carriers, were then checked against the
field names and aliases already in the model.

**New fields:**

| Where | Added | LOBs seen in |
|---|---|---|
| `document` | `mailing_date` (starts the notice period on cancellation and nonrenewal notices) | autop, home, flood, boppr_cgl |
| `carrier` | `fax`, `claims_email`, `claims_address` (where notices are sent) | wc, pl, eo, epli, cumbr, home, autop and others |
| `producer` | `fax`, `contract_number` | eo, pl, home, bopgl, inmrc |
| `named_insured` | `fein`, `business_description` | bopgl, cgl_prop, wc, dfire |
| `policy` | `certificate_number` | cgl, cgl_prop |
| locations | `fire_district`, `insured_interest` (owner, deeded owner, tenant, LLC member) | home, dfire, boppr_cgl, cgl_prop, bopgl_boppr |
| buildings | `basement` | home, flood, cgl_prop |
| interested parties | `is_payor` (the mortgagee pays the premium) | dfire, flood, home |

**Aliases:** carriers print some existing fields under other labels. These
labels were added as aliases, not as new fields:
- "Inception Date": effective date
- "Zone" / "Rating Zone": territory
- "Protection": protection class
- "Underwritten by", "Insuring Company", "Issued by": carrier name
- "Home Office", "Administrative Office": carrier address
- "Amended Date": transaction date
- "Process Date", "Printed": print date
- "Policy Issued": issue date
- "Loan" / "Loan #": loan number
- "Prior or Pending Date": pending/prior litigation date
- "Square Footage": area in square feet
- "Attn": address line 2
- "Endorsement No": endorsement number
- "Representative": countersignature name
- surplus lines tax, stamping fee and fire fee labels: taxes and fees
- protective-device labels (smoke alarm, extinguisher, opening protection):
  protective devices

**Left out:**
- "Account ID": printed only by Coterie and its carrier Benchmark, so it is
  one company's form.
- About 30 labels that are headings, letter text or marketing copy.

**Limitation:** this scan sees only `Label: value` text. Labels used only as
table column headers are not covered.

## 0.2.0 — 3 October 2026 (draft, not yet approved)

Additions chosen by the rule above, from a scan of the 419 source PDFs
(first 4–6 pages of each).

**New fields:**

| Where | Added | LOBs seen in |
|---|---|---|
| `document` | `transaction_date`, `transaction_effective_date`, `transaction_reason`, `endorsement_number` | home, dfire, pumbr, autop, recv, flood, cyber, bopgl, boppr, cgl_prop, inmrc, health |
| `carrier` | `claims_phone` | autop, home, pumbr, recv, autob, cgl, cgl_prop, bopgl, wc |
| `named_insured` and additional named insureds | `date_of_birth`, `gender`, `marital_status`, `occupation` | home, dfire, pumbr, autop |
| `policy` | `rating_state` | autop, cycle, recv |
| `policy` | `audit_period`, `subject_to_audit` | bopgl, pl, cgl, cgl_prop, prop, autob |
| `premium` | `minimum` (separate from `minimum_earned`), `change` (negative for a return premium) | wc, cgl, inmrc, bldrk, other; most lines print a premium change |
| fees, taxes and charges | `fully_earned` | flood, cgl, do, eo, cumbr |
| `billing` | `bill_type`, `payment_method`, `account_number`, `amount_due`, `due_date` | autop, home, pumbr, cycle, recv, autob, prop, cgl_prop, bldrk |
| drivers | `age`, `gender`, `marital_status` | autop, cycle, recv, boat |
| vehicles | `annual_mileage` | autop, cycle, recv |
| vehicles | `commute_miles_one_way` | cycle, recv |
| vehicles | `attached_to_ref`, so a trailer is a vehicle linked to what tows it | boat, recv, cycle, autob |
| buildings | `heating_type`, `heating_fuel`, `heating_year_updated` | home, dfire |
| coverages | `covered_auto_symbols` | autob, package auto parts |
| new optional block | `countersignature` | many lines |

**Shared coverage codes:** `fideon:shared_coverage_codes` lists 19 `X_`
codes, each seen in 2 to 17 LOB folders: vehicle liability, medical payments,
personal liability, UM, UIM, PIP, collision, comprehensive, towing, rental,
loss of use, ordinance or law, water backup, equipment breakdown, identity
theft, cyber, EPL, hired and non-owned auto, and terrorism. Terrorism is a
coverage, so whether it was accepted or rejected goes in `included`; it does
not need a block of its own.

**New rules:**
- An empty block `{}` is rejected.
- A value that was not printed (`raw` null) must have `parsed` null too.
- A sublimit needs a `description`.
- A percentage limit needs a `basis_coverage_code`.
- A flat deductible needs an `amount`; a percentage deductible needs a
  `percentage`.
- New typed values: `StateValue` (USPS two-letter code), `PostalCodeValue`
  (5 digits or ZIP+4), `NaicValue` (5 digits) and `YearValue` (1900–2100).
- `lob_parts` must have at least one part.
- `text_sections` keys must have the form `form|heading|sub-path`.

**Default audit rules:** `fideon:default_audit_rules` are cross-field rules
that every LOB copies into its own `fideon:audit_rules`. A rule is skipped
when the values it needs are absent.
- Effective date is before expiration date.
- The term in months matches the policy dates.
- The change effective date falls within the policy period.
- The retroactive date is on or before the effective date.

**Deliberately left out:**
- **Flood zone:** only flood prints it, so it goes in the flood LOB block.
- **Date licensed and distance to coast:** no source PDF prints them.
- **Employee benefits liability:** printed under one LOB only.
- **Old schemas as evidence:** 13 of the old per-LOB schemas share one
  generated block, so a field appearing in many of them is not evidence that
  many LOBs print it.

**Single-carrier fields:** `heating_*` (Dryden Mutual) and
`commute_miles_one_way` (Allstate) meet the two-LOB rule, but each comes from
one carrier's form. Confirm them with a second carrier before approval.

## 0.1.0 — 3 October 2026 (draft, not yet approved)

First draft, written from SPEC_21 §4.2–4.6. It still needs to be checked
against a sample of seed PDFs, and the owner has not signed it off.

- **Envelope:** `document`, `carrier`, `producer`, `named_insured`, `policy`,
  `lob_parts`, `coverages`, `deductibles`, `interested_parties`, `premium`,
  `billing`, `forms_and_endorsements`, `additional_fields`, `text_sections`.
- **Building blocks:** `locations`, `buildings`, `vehicles`, `drivers`,
  `watercraft`, `scheduled_items`, `rating_exposures`, `rating_modifiers`,
  `claims_made_terms`, `insuring_agreements`, `underlying_insurance`,
  `attachment`, `bond`, `benefit_plans`.
- **Typed values:** typed versions of `FieldValue` (`DateValue`, `MoneyValue`,
  `PercentValue`, `NumberValue`, `BooleanValue`, and versions limited to a
  fixed list of values) restrict `parsed`:
  - `DateValue.parsed` must be MM/DD/YYYY (D1). It is null when the printed
    text is not a date, e.g. a retroactive date printed "Full Prior Acts".
  - `MoneyValue.parsed` must be a number. It is null when the printed text is
    not an amount ("Included", "Not Covered").
- **Structure:**
  - Every block rejects fields it does not define, so a LOB schema cannot add
    fields to a shared block.
  - Every risk unit requires a `unit_id`.
  - A printed value (`raw` set) must name at least one page (`page_ref`).
- **Not included (deliberately):**
  - Location boxes: these stay in `varmap.json` and `ocr.json`, not in
    `FieldValue`.
  - `printed_lines` (removed by SPEC_21 §4.2.6).
  - From the old `_common.json`: terrorism, loss history, state notices, claim
    reporting and signature blocks. Their values go to `additional_fields` and
    their wording to `text_sections`. Any of them that turns up across carriers
    is a candidate to add back as a core field.
