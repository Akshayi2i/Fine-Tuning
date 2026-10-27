# Changelog - homeowners canonical schema

## Aliases - 14 October 2026 (Madison Mutual homeowners declarations; no field change)

`homeowners.coverage_codes.yaml`: one alias, the coverage name as the Madison Mutual Insurance Company homeowners declarations print it (read by `madison_mutual/homeowners.yaml`, GAP-A): `HO_CREDIT_CARD_FORGERY` + `HO 2530 01 06 Incidental Property Coverages - Higher Limits - Credit Card, Electronic Fund Transfer Card Or Access Device, Forgery And Counterfeit Money` (the form's coverage row, its three printed lines joined). No new code; `homeowners.json` and the prompt schema are unchanged (`tools/build_prompt_schemas.py --check`: homeowners current).

## Aliases - 12 October 2026 (Leatherstocking Landlords Package declarations; no field change)

`homeowners.coverage_codes.yaml`: two aliases, the coverage names as the Leatherstocking Cooperative Insurance Company Landlords Package declarations print them (read by `leatherstocking/homeowners.yaml`, CR-PROP-01): `X_LOSS_OF_USE` + `Increased Limit (Coverage D)` (the row that raises Coverage D's limit) and `X_WATER_BACKUP` + `ML-WD Water Damage` (the water-damage endorsement row of the coverage schedule). No new code; `homeowners.json` and the prompt schema are unchanged (`tools/build_prompt_schemas.py --check`: current).

## Common model 1.1.0 - 09 October 2026 (no version change)

Re-pinned to `common_model.json` 1.1.0: `fideon:common_model_version` "1.0.0" -> "1.1.0". No field change: `fideon:schema_version` stays 1.0.0, and the prompt schema (`config/canonical_schema/prompt/policy_check/homeowners.prompt.json`) is unchanged by the pin. Common model 1.x is additive (SPEC_21 4.2.7), so the schema composes 1.1.0 as it composed 1.0.0. The fine-tuning repository's copy of this schema carries the same pin since its commit c8df4bb (9 October 2026); with it the two files are byte-identical again, and both repositories' loaders compose it.

## 1.0.0 - 05 October 2026 (SPEC_21 Phase 2; awaiting project-owner approval)

First SPEC_21 schema of the line. It replaces the pre-SPEC_21 `homeowners.json` (1.6.0, pre-SPEC_21), which was generated from carrier-shaped blocks and did not compose the common model; that file is not carried over (SPEC_21 4.1).

**Built from** all 37 seed golds of 11 carriers (Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers), the draft schema and coverage codes used for the SPEC_21 engine-test twins of this line (synth_core 0.1.2, batches TEST-SYN-PDF-PC-HO-901/-902), and the line's L1 registry YAMLs. Every gold of the line validates against this schema (check C1).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `locations`, `buildings`, `scheduled_items`, `rating_modifiers`. 
- 177 core leaves (budget 80-300); 122 of them are printed on the current seeds. No LOB block: no concept the seeds print lacks a common-model home.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount`.
- `fideon:audit_rules`: the common model's default rules without `claims_made_terms.retroactive_date <= policy.effective_date` (no claims-made terms on a personal line), without the `policy.term_months` rule (no seed prints a term), plus: `len(premium.items[]) == 0 or abs(premium.total - (sum(premium.items[].amount) + sum(premium.discounts[].amount) + sum(premium.surcharges[].amount) + sum(premium.taxes_fees[].amount))) <= 1`
- `fideon:aliases`: printed labels per core field, read from every seed (`homeowners.fields.md`).
- 46 coverage codes (38 LOB, 8 shared `X_`) in `homeowners.coverage_codes.yaml`; every limit is inside `coverages[]`.
- Crosswalk `homeowners.crosswalk.yaml` from the registry paths: 104 new, 58 overflow, 1 drop.

**Changes from the engine-test draft:**
- Written into `config/canonical_schema/policy_check/` as the line's schema (the engine-test drafts lived in a scratch copy of `canonical_schema/` and said 'one-seed draft, not for policy_check').
- Description, status and `fideon:built_from` state the real basis (all seeds of the line); the draft text that named one or two seeds is gone.
- The default rule `claims_made_terms.retroactive_date <= policy.effective_date` is removed (no claims-made terms on a personal line, and the block is not composed).
- `fideon:aliases` added: printed labels per core field from every seed of the line.
- Coverage codes: one code per printed coverage; draft duplicates merged (`replaces_draft_codes` in the codes file), aliases taken from every seed's printed coverage names.
- `fideon:coverage_codes` is a plain list of the codes; the per-code detail lives only in the coverage codes file.
- The coverage codes built from form numbers (`HO_HO_2550_01_06_...`, `HO_MM_HO_5039_09_19_...`) are replaced by the concept codes they duplicate; the form number belongs in `coverages[].form_refs`.
- `HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS` and `HO_ORDINANCE_OR_LAW` are the shared `X_MED_PAY` and `X_ORDINANCE_LAW`.
- No Coverage C <= Coverage A rule: a multi-location seed (Midstate) prints one Coverage A per location, and the rule language compares single values only.
- The `hob_schema` copy of the draft (ho_t/hob_schema, 18:22) was compared with the main draft (23:07): the coverage-codes YAML is identical and the schema differs only by the main draft's added `fideon:coverage_codes` block, so the main draft already held every addition; both are merged here.

**Seed golds to re-code at review** (the coverage code a gold uses differs from the canonical code; the gold still validates, the code only changes the comparison key):

- `Madison Mutual/madison_mutual_home_3`: HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS -> X_MED_PAY; HO_HO_2550_01_06_LOSS_ASSESSMENT_COVERAGE_D -> HO_LOSS_ASSESSMENT; HO_HO_2584_01_06_AUTOMATIC_ADJUSTMENT_OF_LI -> HO_AUTOMATIC_INCREASE; HO_HO_2708_07_11_WATER_BACK_UP_AND_SUMP_DIS -> X_WATER_BACKUP; HO_MM_HO_5039_09_19_DISAPPEARING_DEDUCTIBLE -> HO_DISAPPEARING_DEDUCTIBLE; HO_MM_HO_5140_09_19_ADDITIONAL_INSURED_SPEC -> HO_ADDITIONAL_INSURED; HO_MM_HO_5346_09_19_EQUIPMENT_BREAKDOWN_ENH -> X_EQUIPMENT_BREAKDOWN
- `Madison Mutual/madison_mutual_home_4`: HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS -> X_MED_PAY; HO_HO_2550_01_06_LOSS_ASSESSMENT_COVERAGE_D -> HO_LOSS_ASSESSMENT; HO_HO_2584_01_06_AUTOMATIC_ADJUSTMENT_OF_LI -> HO_AUTOMATIC_INCREASE; HO_HO_2708_07_11_WATER_BACK_UP_AND_SUMP_DIS -> X_WATER_BACKUP; HO_MM_HO_5039_09_19_DISAPPEARING_DEDUCTIBLE -> HO_DISAPPEARING_DEDUCTIBLE; HO_MM_HO_5140_09_19_ADDITIONAL_INSURED_SPEC -> HO_ADDITIONAL_INSURED; HO_MM_HO_5346_09_19_EQUIPMENT_BREAKDOWN_ENH -> X_EQUIPMENT_BREAKDOWN
- `Madison Mutual/madison_mutual_home_5`: HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS -> X_MED_PAY; HO_HO_2584_01_06_AUTOMATIC_ADJUSTMENT_OF_LI -> HO_AUTOMATIC_INCREASE; HO_HO_4855_01_06_REPLACEMENT_COST_LOSS_SETT -> HO_PERSONAL_PROPERTY_REPLACEMENT_COST; HO_LIMITS_CREDIT_CARD_ELECTRONIC_FUND_TRANS -> HO_CREDIT_CARD_FORGERY; HO_MM_HO_5140_09_19_ADDITIONAL_INSURED_SPEC -> HO_ADDITIONAL_INSURED
- `Madison Mutual/madison_mutual_home_6`: HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS -> X_MED_PAY; HO_HO_2584_01_06_AUTOMATIC_ADJUSTMENT_OF_LI -> HO_AUTOMATIC_INCREASE; HO_HO_4855_01_06_REPLACEMENT_COST_LOSS_SETT -> HO_PERSONAL_PROPERTY_REPLACEMENT_COST; HO_LIMITS_CREDIT_CARD_ELECTRONIC_FUND_TRANS -> HO_CREDIT_CARD_FORGERY
- `Madison Mutual/madison_mutual_home_7`: HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS -> X_MED_PAY; HO_HO_2584_01_06_AUTOMATIC_ADJUSTMENT_OF_LI -> HO_AUTOMATIC_INCREASE; HO_HO_4855_01_06_REPLACEMENT_COST_LOSS_SETT -> HO_PERSONAL_PROPERTY_REPLACEMENT_COST; HO_LIMITS_CREDIT_CARD_ELECTRONIC_FUND_TRANS -> HO_CREDIT_CARD_FORGERY; HO_MM_HO_5140_09_19_ADDITIONAL_INSURED_SPEC -> HO_ADDITIONAL_INSURED
- `Madison Mutual/madison_mutual_home_8`: HO_COVERAGE_M_MEDICAL_PAYMENTS_TO_OTHERS -> X_MED_PAY; HO_HO_2584_01_06_AUTOMATIC_ADJUSTMENT_OF_LI -> HO_AUTOMATIC_INCREASE; HO_HO_4855_01_06_REPLACEMENT_COST_LOSS_SETT -> HO_PERSONAL_PROPERTY_REPLACEMENT_COST; HO_LIMITS_CREDIT_CARD_ELECTRONIC_FUND_TRANS -> HO_CREDIT_CARD_FORGERY
- `Millennial Specialty Insurance/millennial_home`: HO_ORDINANCE_OR_LAW -> X_ORDINANCE_LAW

**Open for the project owner:**
- Approve the field list (`homeowners.fields.md`).
- Approve the dwelling characteristics request (CR-HO-01) or confirm overflow for them.
- L1 registry: the homeowners registry declares 164 paths; 13 YAMLs need the Phase 3 migration in one change with this schema (SPEC21_TEAM_PROMPT 7.3).
