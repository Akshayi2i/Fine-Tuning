# Changelog - gl canonical schema

## 3.2.0 - 26 October 2026 (additive; G1 scanned and hybrid seeds, change request approved by the project owner)

Source: the coverage rows the 56 cgl seed golds (G1 scanned and hybrid folders) printed that no gl code named. Printed names only; no document
value is kept. Nothing is removed or renamed, so the registry `canonical_paths`, the prompt files and the 3.1.0 golds keep working.

- **Aliases on existing codes:** `GL_BI_PD` (Liability and Medical Expenses Limit; General Aggregate other than Products/Completed Operations; General Aggregate Limit Per Location;
  Coverage A), `GL_PRODUCTS_COMPLETED_OPS` (Products/Completed Operations Aggregate), `GL_PERSONAL_ADVERTISING_INJURY` (Personal And Advertising Injury; Coverage B),
  `GL_DAMAGE_TO_PREMISES_RENTED` (Fire Legal Liability), `X_HIRED_NONOWNED_AUTO` (Hired Auto and Non-Owned Auto Liability - New York).
- **New codes (66):** the Spectrum businessowners coverage extensions and optional coverages that Hartford packages print inside documents filed as cgl (property extension rows,
  business income extensions, equipment breakdown, identity recovery, data breach, cyber virus, employment practices, umbrella, contractor's pollution liability and others).
  The gl line has no property block; these rows are coverages of the package, coded here so they stay in `coverages[]` instead of `additional_fields` (CR-GL-ALIAS).

## 3.1.0 - 23 October 2026 (additive; decisions 114/115, POLICIES-FINAL-V3 recurring wording)

Source: the distinct originals of POLICIES-FINAL-V3 (185 CGL documents), read in place for recurring form wording only. A caption
counts when it prints in at least 5 different documents of at least 3 different carriers (lower-cased, digits as 9);
lines with a digit run of 4 or more, street, ZIP, company-name or agency-name patterns, or unfamiliar words are dropped.
Counts only; no document text or value is kept, in the repository or in the schema. Nothing is removed or renamed, so the L1 registry
paths, the repaired seeds and golds and the prompt and section files keep working (the derived files are regenerated, not
edited, by the common-model owner's build).

- **Common model pin:** `fideon:common_model_version` raised from 1.1.0 to 1.4.0 (the newest release; 1.2.0 to 1.4.0 are additive: value-state definitions, shared liability codes, six optional fields), which this line inherits without a path change. The FSM hash and the prompt schema are regenerated.
- **New coverage codes (2):** `GL_SUPPLEMENTARY_PAYMENTS` (Supplementary Payments - Coverages A and B; 30 documents of 10
  carriers print the heading; the bail-bond and loss-of-earnings caps are `sublimit` and `per_day` limits of it; the earlier
  gold labels `Supplementary Payments Aggregate Limit`, `Bail Bond Cap` and `Daily Earnings Cap` belonged here) and
  `GL_OWNERS_CONTRACTORS_PROTECTIVE` (OCP liability coverage part; 25 documents of 6 carriers). The code list in `gl.json`
  and `gl.coverage_codes.yaml` stays in one order (20 codes).
- **Aliases (no field change):** `document.issue_date` + `Date Issued`; `named_insured.entity_type` + `Form of Business`;
  `policy.certificate_number` + `Certificate Number`; `policy.policy_type` + `Policy Type`; `policy.prior_policy_number` +
  `Renewal Of`; `premium.taxes_fees[].amount` + `Policy Fee`, `Inspection Fee`; `premium.total` + `Policy Premium`,
  `Total Policy Premium`; `producer.agency_name` + `Agency`, `Producer`.
- **No new field and no LOB block:** every recurring concept without a home that the V3 CGL documents print is also printed
  by another line (policy period time, carrier company type, the insurer's President and Secretary names, insured copy
  marking, the terrorism federal-share / trigger / cap figures, the liability deductible basis, the insurer letters of an
  embedded certificate, package crime and property coverages), so SPEC_21 4.2.4 puts them in the common model. They are
  requested in CR-GL-22; gl takes them through its envelope at the next common-model release. Until then they travel as
  `additional_fields` as today.
- **Crosswalk:** `to_schema` 3.1.0; no mapping changes (no legacy path gained a home).
- **Exclusion endorsements need no new home:** the communicable disease (two titles), nuclear energy liability,
  employment-related practices, cyber incident, PFAS, asbestos, punitive damages, fungi or bacteria and total pollution
  exclusions are `forms_and_endorsements[]` rows (title and form number); the property, inland marine, auto, equipment
  breakdown and professional coverage-part titles are `lob_parts[]` titles.
- **Derived files not regenerated here:** `prompt/policy_check/gl.prompt.json`, `prompt/sections/gl/coverages.json` and the
  prompt manifest list the code enumeration; the common-model owner's build regenerates them.
- **Seeds (decision needed):** seed mode (read-only) over the 51 gl seeds with a PDF: the 39 that passed before still read the same, and
  the only new problems are the version stamps (`C13 review schema version 3.0.0, current 3.1.0` and `varmap schema version 3.0.0, run uses
  3.1.0`): the engine compares the stamps for equality, so any minor release needs either a same-major-and-not-newer compatibility rule in
  `checks.c13_signoff` / `engine._check_varmap` and the `signoff_problems` of the line, or a re-stamp by the seed reviewer.
- **Checks:** see the implementation log of 23 October 2026.

## Aliases - 15 October 2026 (Dryden Mutual commercial application; no field or code change)

`gl.coverage_codes.yaml`, the liability part of Dryden Mutual's Commercial Application (read by `dryden_mutual/gl.yaml`): `X_MED_PAY` + `Coverage M – Premises Medical Payments` (printed with an en dash). The code list, `gl.json` and the prompt schema are unchanged.

## Aliases - 12 October 2026 (package policies, PKG2; no field or code change)

`gl.coverage_codes.yaml`, the liability part of the Dryden Mutual and North Country package declarations: `GL_BI_PD` + `Cov L - Bodily Injury & Property Damage`; `GL_PRODUCTS_COMPLETED_OPS` + `Coverage N - Products/Completed Operations`, `Cov N - Products/Completed Operations`; `GL_PERSONAL_ADVERTISING_INJURY` + `Coverage P - Personal Injury & Advertising Liability`; `X_MED_PAY` + `Cov M - Medical Payments`; `GL_COVERAGE_EXTENSION` + `Business General Liability - Extra Coverage`, `General Liability Coverage (Business General Liability Extra Coverage)`; `X_WAIVER_OF_SUBROGATION` + `Waiver Of Subrogation`. The code list (and `gl.json` `fideon:coverage_codes`) is unchanged.

## Aliases - 11 October 2026 (no version change)

`gl.coverage_codes.yaml`, printed names the CGL seed documents print and L1 did not code (completeness programme):
`GL_PRODUCTS_COMPLETED_OPS` + `Products-Completed Ops Aggregate Limit`; `GL_PERSONAL_ADVERTISING_INJURY` +
`Personal/Advertising Limit`; `X_MED_PAY` + `Medical Expense Limit (Any one person)`; `X_ADDITIONAL_INSURED` +
`Vendors`, `Lessor of Premises`, `Owner of Leased Land`, `Owners, Lessees or Contractors Completed Work - Automatic
Status` and `State or Political Subdivision - Premises Permits` (the additional-insured coverages of Utica First's
`OPTIONAL LIABILITY COVERAGE LIST`, each with its own premium). No field, code or limit type changes, so the schema
version stays 3.0.0 and the prompt schema is unchanged.

## Training parity - 09 October 2026 (no version change)

The fine-tuning repository (c8df4bb) now carries this schema natively (CR-GL-01 resolved) and took the
`rating_exposures` section-map entry with the key `[class_code, location_ref, rating_component]` (CR-GL-05).
The serving port follows it: prompt template 7.2.0 and the model-view descriptions change the gl prompt schema
and system prompt (`config/canonical_schema/prompt/policy_check/gl.prompt.json` regenerated); the schema itself
is unchanged, so the version stays 3.0.0.

## Aliases - 08 October 2026 (no version change)

`gl.coverage_codes.yaml`: `X_ADDITIONAL_INSURED` gains the aliases `Additional Insured Designated Person or
Organization` and `Designated Person or Organization`, the printed names of Utica First's AAIS form AP 0301 in its
forms inventory and in its `OPTIONAL LIABILITY COVERAGE LIST` (`policy/utica_first/gl.yaml`). Aliases are printed
names collected from the registry captions (SPEC_21 4.2.4); no field, code or limit type changes, so the schema
version stays 3.0.0.

## 3.0.0 - 08 October 2026 (draft for the owner's approval; adopts common model 1.1.0)

The common-model owner accepted the gl change requests and released them as common model 1.1.0. gl 3.0.0 takes
them up before any CGL seed gold is drafted, so the renames cost no re-coding. It replaces approving 2.0.0: approving
2.0.0 and then moving to 3.0.0 would have meant signing the schema off twice.
- **Codes (CR-GL-03):** `GL_ADDITIONAL_INSURED`, `GL_PRIMARY_NONCONTRIBUTORY` and `GL_WAIVER_OF_SUBROGATION` are
  renamed `X_ADDITIONAL_INSURED`, `X_PRIMARY_NONCONTRIBUTORY` and `X_WAIVER_OF_SUBROGATION` (`shared: true`,
  `replaces_codes`). Same names, aliases, carrier counts and position in the list. 18 codes, 6 of them shared.
- **Fields (CR-GL-02, CR-GL-12):** the values 2.0.0 sent to `additional_fields` have fields now.
  - The `Pr/Co` rate and premium of a classification are a second `rating_exposures[]` row:
    `rating_component: products_completed_operations`, beside the `premises_operations` row. `Pr/Co`,
    `Prod/Ops Rate` and `Prod./Comp. Ops` become labels of `rating_exposures[].rate`, and
    `Products-Completed Operations Premium` a label of `rating_exposures[].premium`.
  - `premium.minimum_earned_percent`.
  - `carrier.financial_rating`. The common model carries the A.M. Best labels.
- `fideon:common_model_version` 1.1.0. The description, status and `fideon:lob_block` say so.

Major version: renamed coverage codes (SPEC_21 4.1). Core leaves 170 -> 172 (`rating_component` is a row
classifier, not a printed value). Crosswalk `to_schema` 3.0.0:
- the 2.0.0 codes map to the `X_` codes;
- the 1.1.0-1.2.0 block paths and the legacy `premium_products_completed_operations`,
  `rate_products_completed_operations` and `policy.minimum_earned_premium_percentage` are `new` paths, no longer
  `overflow`;
- the block's join key `class_code`, a `drop` in 2.0.0, is now the products-completed row's own `class_code`.

`registry_migrate --lob gl --check`: no registry path changes. L1 codes the renamed coverages through gl's own
coverage-code file (`coverage_code_for("gl", "Additional Insured")` is `X_ADDITIONAL_INSURED`).

**Pipeline:** `gl.prompt.json` regenerated, along with the prompt-schema manifest and the `schema_hashes.json` gl hash.
casualty_fleet has no adapter yet, so no recheck. The gl fine-tuning parity fixture still holds the fine-tuning
repository's 2.0.0 rendering, so it is re-recorded there.

**Open:** CR-GL-16 (`Deductible.basis`) is held by the owner for a second line's evidence. The deductible basis still
travels as `additional_fields` (`Deductible Basis`).

## 2.0.0 - 08 October 2026 (SPEC_21 v0.9 alignment)

SPEC_21 v0.9 (8 October 2026) allows a LOB block only for single fields unique to the LOB, requires a LOB that
prints rating exposures to use the common block, and makes a new field in an existing common block a change
request rather than a LOB-schema field (4.2, 4.2.4). The 1.1.0 `gl` block broke all three, so it is removed:
- `gl.products_completed_ops_rating[]` (a repeating classification table) -> `additional_fields`
  (`Products-Completed Operations Rate` / `Premium`, `section_hint` the classification); CR-GL-02 reopened as a
  request for optional `RatingExposure` fields.
- `gl.minimum_earned_premium_percent`, `gl.carrier_financial_rating` -> `additional_fields`
  (`Minimum Earned Premium Percentage`, `Carrier Financial Rating`); CR-GL-12 stands.

Major version: removing core fields (SPEC_21 4.1). Core leaves 175 -> 170. Crosswalk: `to_schema` 2.0.0; the three
legacy paths that 1.1.0 mapped to the block are `overflow` again, and the five 1.1.0-1.2.0 block paths have entries
(four `overflow`, the join key `class_code` a `drop`, which needs reviewer approval). No registry YAML used the block
(`registry_lint --lob gl` clean). Coverage codes, aliases and blocks unchanged. New CR-GL-15: `cgl` vs `gl` as a
package part's `lob`. CR-GL-06: the common-model owner confirms the 1.1.0 edits to its files (v0.9 4.2.5).

**Review fixes (8 October 2026):** audit rule 8 removed (not evaluable in the synth_core relation language;
CR-GL-21); the meaning of rule 5 and of the mandatory `coverages[].limits[].amount` written down (at least one coverage
has a limit); the endorsement-coverage convention (one coverage per endorsement form, printed title, `form_refs`)
documented in `gl.fields.md` and `gl.coverage_codes.yaml`; the per-fire limit documented, with the L1 fix requested
(CR-GL-20) instead of a second default limit type, which would turn every unqualified damage-to-premises limit into
`other`.

**Change requests (8 October 2026, after review):** CR-GL-02 revised to one optional `RatingExposure.rating_component`; CR-GL-12 marked to send before the property_pkg team drafts; CR-GL-03 evidence updated; new CR-GL-16 (`Deductible.basis`), CR-GL-17 (edition-date format), CR-GL-18 (`routing_key: gl`), CR-GL-19 (the handoff's CI checks). A send list heads CHANGE_REQUESTS_gl.md. No schema change: the common model is the owner's (SPEC_21 v0.9 4.2.7).

**Pipeline:** `gl.prompt.json` regenerated (the `gl` block leaves the prompt; v0.9 4.2.7 adapter recheck applies to
casualty_fleet, which has no adapter yet); `finetuning_gl_derived.json` re-derived with this repository's port;
`schema_hashes.json` gl hash updated. Tests: the gl LOB-block assertions restored to "no LOB block".

## 1.2.0 - 08 October 2026 (approved by the project owner)

**Built from** 105 CGL documents: 1.1.0's 99 plus six CGL documents of the `to_sakshi` set that the corpus
did not hold (compared by content and policy number): Main Street America's Commercial Portfolio
declarations, endorsement and renewal (Midvale Indemnity Company, NGM Insurance Company), a Scottsdale
Insurance Company surplus-lines policy, a USLI declarations and a Utica First Contractors Special renewal.
They are added to `policy_CGL/` (`The Main Street America Group`, `Scottsdale Insurance Company`, `USLI`,
`Utica First Insurace Company`). The set's other 17 files were already covered, are not CGL (North Country
special multi-peril) or are paperwork.

**Changes** (additive only; no field, block or limit type added):
- Coverage codes 17 -> 18: `GL_SNOW_PLOW_OPERATIONS` (`Snow Plow Operations Coverage`, `Snow Removal Coverage`,
  `Snow Removal Operations Coverage`; Utica prints it with its own premium; 3 carrier folders).
- Coverage-code aliases for captions the six documents print and L1 did not code (`fideon/l1/spec21.py
  coverage_code_for`): `GL_BI_PD` + `Per Occurrence Limit`; `GL_PRODUCTS_COMPLETED_OPS` + `PRODUCTS - COMP/OP AGG`;
  `GL_PERSONAL_ADVERTISING_INJURY` + `PERSONAL & ADV INJURY`; `X_MED_PAY` + `Medical Payments (limit per person)`,
  `MED EXP (Any one person)`; `GL_DAMAGE_TO_PREMISES_RENTED` + 4 (incl. `Broadened Coverage for Damage to Premises
  Rented to You`); `GL_POLLUTION_LIABILITY` + `Pesticide or Herbicide Applicator - Limited Pollution Coverage`;
  `GL_ADDITIONAL_INSURED` + 2 (owners, lessees or contractors); `GL_WAIVER_OF_SUBROGATION` + `Waiver of Transfer of
  Rights of Recovery Against Others to Us`; `GL_COVERAGE_EXTENSION` + `Contractors Extension Endorsement`. Every
  caption of the six documents now codes. `carriers` recounted over 23 carrier folders (`GL_LIQUOR_LIABILITY`
  reads lower because its meaning now names its captions, not because a document was lost).
- `gl.crosswalk.yaml`: `to_schema` 1.2.0 (no path changes).
- Open, recorded as change requests: CR-GL-07 (building data on contractors packages) gains the Utica renewal;
  CR-GL-13 (certificates of insurance inside a policy packet); CR-GL-14 (the property and inland-marine parts of
  a commercial package have no SPEC_21 schema).

**Pipeline (same day):** `prompt/policy_check/gl.prompt.json` regenerated; `finetuning_gl_derived.json`
re-derived with this repository's port (re-record with `record_finetuning_gl.py`); `schema_hashes.json` gl hash
updated.

## 1.1.0 - 08 October 2026 (approved by the project owner)

**Built from** 99 CGL documents: the 49 seed documents of 1.0.0 and 50 CGL documents of 10 carrier
folders of the `policy_CGL` corpus (FORTEGRA, Great American, Hiscox, Indium, Johnson & Johnson, Liberty
Mutual, McNeil & Company, Philadelphia Insurance Companies, Russell Bond, XS Broker), read in place (first
20 pages: a renewal packet can print its CGL declarations late). Not counted, and removed from
`policy_CGL`: Businessowners (Coterie `dec_01`, both folders), professional liability (Coterie
`dec_02`/`renewal_01`, Coterie Insurance `dec_02`), D&O (Philadelphia `policy_01`), paperwork (FORTEGRA
`renewal_01`, Philadelphia `policy_02`, Russell Bond `endo_01`), five text-less scans, the unreadable XS
Broker `policy_02`, and Chubb's personal excess policy. New writing companies: Arch, Atlantic Casualty,
Century Surety, Evanston, Hiscox, Markel, Mesa Underwriters Specialty, Ohio Security, Palomar Specialty,
Philadelphia Indemnity (identified from the documents' text; Indium's are Travelers).

**Changes** (additive only; nothing removed or renamed):
- **LOB block `gl`** (SPEC_21 4.2.4), for what 3 or more of the 20 carrier folders print and no building block
  holds: `products_completed_ops_rating[]` (`class_code`, `location_ref`, `rate`, `premium`: the Pr/Co figures
  of each classification, 35 documents / 8 folders; CR-GL-02 resolved for gl), `minimum_earned_premium_percent`
  (17 / 3) and `carrier_financial_rating` (12 / 4; both CR-GL-12). Core leaves 170 -> 175.
- Coverage codes 14 -> 17: `GL_COVERAGE_EXTENSION` (broadening / extension endorsements: `Commercial General
  Liability Extension`, `Contractors Extender`, `Contractors Custom Protector Coverages`...; 20 documents / 7
  folders), `GL_EMPLOYEE_BENEFITS_LIABILITY` (meaning names its captions, `Each Employee Limit`, `Aggregate
  Limit`, as the limit-carrying codes' meanings do since the live run of 8 October 2026) (limits `each_employee`, `aggregate`; XS
  Broker) and `GL_POLLUTION_LIABILITY` (a pollution liability coverage part or limited pollution coverage
  written with the CGL; Johnson & Johnson, Liberty Mutual, McNeil). Both are in the prompt schema's code list.
- Coverage-code aliases: `GL_BI_PD` + `General Aggregate`, `General Aggregate Limit (Other Than
  Products-Completed Ops)`, `Designated Construction Project(s) General Aggregate Limit` (a per-project
  aggregate is a `GL_BI_PD` `aggregate` limit whose `description` names the project basis; no new limit
  type); `GL_PRODUCTS_COMPLETED_OPS` + 2; `GL_PERSONAL_ADVERTISING_INJURY` + 2; `X_MED_PAY` + 2;
  `GL_DAMAGE_TO_PREMISES_RENTED` + `Fire Damage`, `Fire Damage (Any One Fire)`; `X_HIRED_NONOWNED_AUTO` +
  `Hired and Non-Owned Auto`. `carriers` recounted over the 20 carrier folders.
- `fideon:aliases` (labels the common model does not already list): `policy.audit_period` (`Audit Period
  (if applicable)`), `policy.subject_to_audit` (`Auditable`, `Non-Auditable`),
  `rating_exposures[].exposure_amount` (`Exposure`), `rating_exposures[].exposure_basis` (`Exposure
  Basis`), `rating_exposures[].class_code` (`Classification Code No.`, `Classification Code`, `Class
  Code`), `carrier.name` (`Company`), `deductibles[].amount` (`Deductible Limit of Liability`),
  `claims_made_terms.retroactive_date` (2). `Form of Business` and `Audit Period` are already common-model
  aliases and were not repeated.
- `gl.fields.md`: header, source note, and the label and carrier columns of the fields above.
- `gl.crosswalk.yaml`: `to_schema` 1.1.0; the legacy `classification_schedule[].rate_products_completed_operations`,
  `premium_products_completed_operations` and `policy.minimum_earned_premium_percentage` move from overflow to the
  LOB block (`registry_migrate.py --lob gl --check`: no registry change).
- `lob_schema_map.yaml`: CR-GL-06 applied (see CHANGELOG_common.md).

**Pipeline (same day):** `prompt/policy_check/gl.prompt.json` regenerated (`tools/build_prompt_schemas.py`;
only gl changed). The parity fixture `fideon/vlm/tests/fixtures/finetuning_gl_derived.json` was re-derived
with this repository's port of the fine-tuning code because the fine-tuning repository was not available;
re-record it with `record_finetuning_gl.py` at the next fine-tuning checkout.

**Golds:** five real-document golds (kept locally in `cgl_gold/`, not in the repository: customer data) validate against 1.1.0:
Mesa Underwriters binder, Travelers Business Pac Plus, Atlantic Casualty binder, Ohio Security policy change,
Philadelphia Indemnity renewal. Overflow (`additional_fields`) across them: 6 values, none recurring: two
Businessowners property coverages printed 'NOT COVERED' (another line's), commission (2 folders), quote number
and underwriter (1 folder each).

**Tests updated for the LOB block:** `test_gl_schema.py::test_mandatory_fields_and_blocks` (the block's
fields instead of 'no LOB block') and `test_section_windows_parity.py::test_gl_fixture_is_the_ported_commit`
(the `lineblk` group now ends with `gl`).

## 1.0.0 - 08 October 2026 (SPEC_21; no seed golds yet; awaiting project-owner approval)

First SPEC_21 schema of the line (commercial general liability, code GL). It replaces the pre-SPEC_21 `gl.json` (`fideon.canonical.general_liability` 1.5.0), which was generated from carrier-shaped blocks (a `general_liability` line block with `limits_of_insurance`, `classification_schedule`, `claims_made_provisions`, `additional_insureds`, ...) and did not compose the common model, so `SchemaRegistry` read every CGL document against `_fallback` (E011). That file is not carried over (SPEC_21 4.1); every one of its 377 leaves has a crosswalk entry. The same legacy file is the fine-tuning repository's `configs/canonical schema/LOB Schema/gl.json` (commit 9679ec8): the training side has no SPEC_21 gl schema (CHANGE_REQUESTS_gl.md, CR-GL-01).

**Built from** what the line's real CGL documents print: 49 documents of 10 carriers (Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group) under `synth_core/fideon/synth_core/data/source data/<Carrier>/cgl/`, read in place (text layer, first 8 pages; no file copied); the legacy `gl.json` 1.5.0; the line's L1 registry YAMLs (`config/document_registry/policy/*/gl.yaml`). The folder's Chubb file is a personal excess liability policy and was not used. No seed gold exists for gl, so no gold was validated (check C1 has nothing to run on); a synthetic gl document validates (`fideon/schemas/tests/test_gl_schema.py`).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `locations`, `rating_exposures` (the classification schedule), `claims_made_terms` and `countersignature`.
  - `rating_modifiers` (listed for gl in `lob_schema_map.yaml`) is left out: no document prints an IRPM, schedule-rating or experience factor.
  - `claims_made_terms` and `countersignature` are composed although the map does not list them: the legacy schema carries claims-made provisions, Gemini's declarations print a retroactive date, and Diamond State, Utica First and USLI print a countersignature or its date. The map is not edited here (its derived `common_model.fields.md` is the common-model owner's): CR-GL-06.
- 170 core leaves (budget 80-300); 95 of them carry a label the documents print (label scan, `gl.fields.md`). No LOB block: no concept the documents print lacks a common-model home, except the two change requests CR-GL-02 and CR-GL-07, which travel as `additional_fields` meanwhile.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount` (any coverage with a limit: a liquor-only coverage part prints no Coverage A).
- `fideon:audit_rules`: the common model's five default rules (the claims-made rule kept), plus: each-occurrence <= general aggregate on `GL_BI_PD` and each-common-cause <= aggregate on `GL_LIQUOR_LIABILITY` (ISO CG 00 01 / CG 00 33 Section III); a classification names a scheduled location or none.
- `fideon:aliases`: printed labels per core field that the common model does not already list (`gl.fields.md`).
- 14 coverage codes (11 LOB, 3 shared `X_`) in `gl.coverage_codes.yaml`; every limit is inside `coverages[]`:
  - ISO declarations print a CGL's limits as captions, not coverage names, so each caption is an alias of the coverage it limits: `Each Occurrence Limit` and `General Aggregate Limit (Other Than Products-Completed Operations)` are `GL_BI_PD` (Coverage A, limits `per_occurrence` and `aggregate`; the general aggregate is shared by Coverages A-C, CR-GL-08); `Products-Completed Operations Aggregate Limit` is `GL_PRODUCTS_COMPLETED_OPS` (Utica First and Security Mutual print it as Coverage N; limit `products_completed_ops_aggregate`); `Personal and Advertising Injury Limit` is `GL_PERSONAL_ADVERTISING_INJURY` (`per_person`); `Medical Expense Limit` is the shared `X_MED_PAY` (`per_person`); `Damage to Premises Rented to You Limit`, `Fire Damage Limit` and `Fire Legal Liability Limit` are `GL_DAMAGE_TO_PREMISES_RENTED` (`per_location`).
  - Further codes, each from printed labels: `GL_LIQUOR_LIABILITY`, `GL_ABUSE_MOLESTATION`, `GL_PROFESSIONAL_LIABILITY`, `GL_CARE_CUSTODY_CONTROL`, `GL_ADDITIONAL_INSURED`, `GL_PRIMARY_NONCONTRIBUTORY`, `GL_WAIVER_OF_SUBROGATION`, and the shared `X_HIRED_NONOWNED_AUTO`, `X_TERRORISM`.
  - A sublimit printed under the limits (`Independent Contractors/Subcontractors Sublimit`) is a `sublimit` limit of `GL_BI_PD`; a self-insured retention is a top-level `deductibles[]` entry of type `sir`.
- Crosswalk `gl.crosswalk.yaml`: every leaf of the legacy schema and every path the registry YAMLs declare, 380 paths: 273 new, 102 overflow, 5 drop (document kind, carrier folder, source system, signature flag, the line itself: computed values, not printed ones; reviewer approval needed). The registry YAMLs needed no rewrite: the one entry that declares paths (`utica_first/gl.yaml`) already names SPEC_21 envelope paths and `additional_fields`.

**Pipeline (same day):** prompt schema `config/canonical_schema/prompt/policy_check/gl.prompt.json` (`tools/build_prompt_schemas.py`); the section map's common-model profile reads `rating_exposures` in the `arrays` group with its `array_keys` (this repository's addition, CR-GL-05); `SchemaRegistry` resolves gl to this schema (no longer `_fallback`). After the first live run (the base model wrote every limit caption as overflow), the meanings of the six limit-carrying codes name the captions their limits are printed under (`GL_BI_PD`: "... (its limits: Each Occurrence Limit, General Aggregate Limit)"); a code's meaning is the prompt text the model reads beside the code.

**Open for the project owner:**
- Approve the field list (`gl.fields.md`) and the coverage codes.
- The change requests in `CHANGE_REQUESTS_gl.md`, above all CR-GL-01 (training parity: the fine-tuning repository's gl is still the legacy file, and it refuses a partly migrated layout family).
- Seed golds for gl (SPEC_21 5): none exist; the field list was read off the documents, not drafted golds.
