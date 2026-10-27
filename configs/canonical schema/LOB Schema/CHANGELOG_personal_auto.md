# Changelog - personal_auto canonical schema

## Coverage code aliases - 15 October 2026 (no version change)

Printed names added as aliases in `personal_auto.coverage_codes.yaml`, as agency-management-system printouts (`VEHICLE COVERAGES` grid; Allstate and NYCM paper) print them, each under the code the line's draft golds use: `Personal Injury Protection/Basic Economic Loss`, `Extended PIP`, `Work Loss Benefit` (`X_PIP`); `Comprehensive (Full Glass applies)` (`X_COMPREHENSIVE`); `Collision Coverage`, `Collision Coverage (Full Glass applies)` (`X_COLLISION`); `Automobile Death Indemnity Benefits`, `Death Benefits` (`PA_ADDITIONAL_DEATH_BENEFITS`); `Apip - Other Expense` (`PA_ADDITIONAL_PIP`); `Supplementary Uninsured Motorist` (`X_UIM`); `Towing & Labor`, `Travel Protection - Roadside` (`X_TOWING`); `Travel Protection - Transportation Expenses` (`X_RENTAL`); `Travel Protection - Trip Interruption` (`PA_TRIP_INTERRUPTION`). No code, field or default limit type changes; the schema and its prompt schema are unchanged. Each alias names one code only.

## Coverage code alias - 14 October 2026 (no version change)

`Optional Basic Economical Loss` added as an alias of `PA_OPTIONAL_BASIC_ECONOMIC_LOSS`, as Travelers' automobile policy continuation declarations (form PL-7782, Item 4, coverage W) print it. No code, field or default limit type changes; the schema and its prompt schema are unchanged. The alias names one code only.

## Coverage code aliases - 13 October 2026 (no version change)

Three printed names added as aliases in `personal_auto.coverage_codes.yaml`, as NYCM's automobile endorsement summary (form NYC 79A) prints them in its `VEHICLE COVERAGES` grid: `FULL GLASS` (`PA_FULL_GLASS`), `OEM` (`PA_OEM_PARTS`) and `PERSONAL ARTICLES` (`PA_PERSONAL_ARTICLES`). No code, field or default limit type changes; the schema (`fideon:schema_version` 1.0.0) and its prompt schema are unchanged. Each alias names one code only (no clash with another code of the line or a shared `X_` code).

## Common model 1.1.0 - 09 October 2026 (no version change)

Re-pinned to `common_model.json` 1.1.0: `fideon:common_model_version` "1.0.0" -> "1.1.0". No field change: `fideon:schema_version` stays 1.0.0, and the prompt schema (`config/canonical_schema/prompt/policy_check/personal_auto.prompt.json`) is unchanged by the pin. Common model 1.x is additive (SPEC_21 4.2.7), so the schema composes 1.1.0 as it composed 1.0.0. The fine-tuning repository's copy of this schema carries the same pin since its commit c8df4bb (9 October 2026); with it the two files are byte-identical again, and both repositories' loaders compose it.

## 1.0.0 - 05 October 2026 (SPEC_21 Phase 2; awaiting project-owner approval)

First SPEC_21 schema of the line. It replaces the pre-SPEC_21 `personal_auto.json` (1.5.0, pre-SPEC_21), which was generated from carrier-shaped blocks and did not compose the common model; that file is not carried over (SPEC_21 4.1).

**Built from** all 20 seed golds of 9 carriers (AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers), the draft schema and coverage codes used for the SPEC_21 engine-test twins of this line (synth_core 0.1.2, batches TEST-SYN-PDF-PC-PA-901/-902), and the line's L1 registry YAMLs. Every gold of the line validates against this schema (check C1).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `locations`, `vehicles`, `drivers`, `rating_modifiers`. Classic auto is part of this line (L1 code `classic_auto` reads this schema per `lob_schema_map.yaml`): guaranteed / agreed value is `vehicles[].stated_amount`, spare parts and collector coverages are coverage codes.
- 180 core leaves (budget 80-300); 106 of them are printed on the current seeds. No LOB block: no concept the seeds print lacks a common-model home.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount`.
- `fideon:audit_rules`: the common model's default rules without `claims_made_terms.retroactive_date <= policy.effective_date` (no claims-made terms on a personal line), plus: `premium.total == sum(premium.items[].amount) + sum(premium.taxes_fees[].amount) ± 1`; ``coverages[coverage_code=X_VEHICLE_LIABILITY].limits[limit_type=per_person].amount` <= `coverages[coverage_code=X_VEHICLE_LIABILITY].limits[limit_type=per_accident].amount``; ``coverages[coverage_code=X_UIM].limits[limit_type=per_person].amount` <= `coverages[coverage_code=X_UIM].limits[limit_type=per_accident].amount``
- `fideon:aliases`: printed labels per core field, read from every seed (`personal_auto.fields.md`).
- 33 coverage codes (24 LOB, 9 shared `X_`) in `personal_auto.coverage_codes.yaml`; every limit is inside `coverages[]`.
- Crosswalk `personal_auto.crosswalk.yaml` from the registry paths: 42 new, 26 overflow, 2 drop.

**Changes from the engine-test draft:**
- Written into `config/canonical_schema/policy_check/` as the line's schema (the engine-test drafts lived in a scratch copy of `canonical_schema/` and said 'one-seed draft, not for policy_check').
- Description, status and `fideon:built_from` state the real basis (all seeds of the line); the draft text that named one or two seeds is gone.
- The default rule `claims_made_terms.retroactive_date <= policy.effective_date` is removed (no claims-made terms on a personal line, and the block is not composed).
- `fideon:aliases` added: printed labels per core field from every seed of the line.
- Coverage codes: one code per printed coverage; draft duplicates merged (`replaces_draft_codes` in the codes file), aliases taken from every seed's printed coverage names.
- `fideon:coverage_codes` is a plain list of the codes; the per-code detail lives only in the coverage codes file.
- Classic auto merged: the Hagerty seeds and the two `classic_auto` registry YAMLs are covered by this schema and its crosswalk (project decision of 4 October 2026, CHANGELOG_common Unreleased).
- The empty `personal_auto` LOB block of the one-seed draft is removed.
- Duplicate draft codes merged (`PA_SUPP_SPOUSAL_LIAB`, `PA_OEM`, `PA_MEDIA`, `PA_PET_INJURY_COVERAGE`, `PA_PERSONAL_ARTICLES_COVERAGE`, `PA_TRAVEL_TRIP_INTERRUPTION`, `PA_ADDED_PIP`, `PA_ADDITIONAL_PERSONAL_INJURY_PROTECTION`, `PA_TOTAL_NO_FAULT_BENEFITS`, `PA_FULL_COVERAGE_WINDOW_GLASS`); rental and towing/roadside coverages use the shared `X_RENTAL` and `X_TOWING`.
- Supplementary uninsured/underinsured motorists (SUM) printed names are aliases of `X_UIM`, as the common model assigns them; basic UM stays `X_UM`.
- The policy-term rule is kept (five seeds print the term).

**Seed golds to re-code at review** (the coverage code a gold uses differs from the canonical code; the gold still validates, the code only changes the comparison key):

- `Foremost Insurance Company/foremost_autop`: PA_RENTAL_VEHICLE -> X_RENTAL (x3); PA_TOWING_LABOR -> X_TOWING (x3)
- `Hagerty Insurance/hagerty_insurance_autop`: PA_RENTAL_VEHICLE -> X_RENTAL; X_PIP -> PA_ADDITIONAL_PIP; X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS
- `Hagerty Insurance/hagerty_insurance_autop_2`: PA_RENTAL_VEHICLE -> X_RENTAL; PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY; X_PIP -> PA_ADDITIONAL_PIP; X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS
- `Hagerty Insurance/hagerty_insurance_autop_3`: X_PIP -> PA_AGGREGATE_NO_FAULT_BENEFITS_AVAILABLE
- `Hagerty Insurance/hagerty_insurance_autop_4`: PA_ADDED_PIP -> PA_ADDITIONAL_PIP; PA_TOTAL_NO_FAULT_BENEFITS -> PA_AGGREGATE_NO_FAULT_BENEFITS_AVAILABLE
- `Mercury Insurance Company/mercury_insurance_company_autop`: PA_RENTAL_VEHICLE -> X_RENTAL (x2)
- `NYCM Insurance/nyc_autop`: PA_ADDITIONAL_PERSONAL_INJURY_PROTECTION -> PA_ADDITIONAL_PIP (x3); PA_FULL_COVERAGE_WINDOW_GLASS -> PA_FULL_GLASS (x3); PA_PET_INJURY_COVERAGE -> PA_PET_INJURY (x3); PA_TRAVEL_TRIP_INTERRUPTION -> PA_TRIP_INTERRUPTION (x3)
- `NYCM Insurance/nyc_autop_1`: PA_ADDITIONAL_PERSONAL_INJURY_PROTECTION -> PA_ADDITIONAL_PIP (x3); PA_PET_INJURY_COVERAGE -> PA_PET_INJURY (x3); PA_RENTAL_VEHICLE -> X_RENTAL (x3); PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY (x3)
- `NYCM Insurance/nyc_autop_2`: PA_OEM -> PA_OEM_PARTS; PA_PET_INJURY_COVERAGE -> PA_PET_INJURY; PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY; X_PIP -> PA_ADDITIONAL_DEATH_BENEFITS; X_PIP -> PA_ADDITIONAL_PIP (x2); X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS; X_PIP -> PA_OUT_OF_STATE_PERSONAL_INJURY_PROTECTION
- `NYCM Insurance/nyc_autop_3`: PA_MEDIA -> PA_TAPES_DISCS_MEDIA; PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY; X_PIP -> PA_ADDITIONAL_DEATH_BENEFITS; X_PIP -> PA_ADDITIONAL_PIP (x2); X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS; X_PIP -> PA_OUT_OF_STATE_PERSONAL_INJURY_PROTECTION
- `NYCM Insurance/nycm_insurance_autop_endo`: PA_PET_INJURY_COVERAGE -> PA_PET_INJURY; PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY; X_PIP -> PA_ADDITIONAL_DEATH_BENEFITS; X_PIP -> PA_ADDITIONAL_PIP (x2); X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS; X_PIP -> PA_OUT_OF_STATE_PERSONAL_INJURY_PROTECTION
- `NYCM Insurance/nycm_insurance_autop_renewal`: PA_ADDITIONAL_PERSONAL_INJURY_PROTECTION -> PA_ADDITIONAL_PIP; PA_FULL_COVERAGE_WINDOW_GLASS -> PA_FULL_GLASS; PA_PERSONAL_ARTICLES_COVERAGE -> PA_PERSONAL_ARTICLES; PA_PET_INJURY_COVERAGE -> PA_PET_INJURY
- `Plymouth/plymouth_rock_pauto`: PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY (x2); X_PIP -> PA_ADDITIONAL_PIP (x2); X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS (x2)
- `Progressive/progressive_autop_2`: PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY; X_PIP -> PA_ADDITIONAL_PIP; X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS
- `Travelers/travelers_autop_endo`: PA_SUPP_SPOUSAL_LIAB -> PA_SUPPLEMENTAL_SPOUSAL_LIABILITY; X_PIP -> PA_ADDITIONAL_PIP; X_PIP -> PA_OPTIONAL_BASIC_ECONOMIC_LOSS
- `Travelers/travelers_autop_renewal`: PA_RENTAL_VEHICLE -> X_RENTAL; PA_ROADSIDE_ASSISTANCE -> X_TOWING

**Open for the project owner:**
- Approve the field list (`personal_auto.fields.md`).
- Remove the legacy `classic_auto.json` once this schema and its crosswalk land (CHANGELOG_common, Unreleased): while it exists SchemaRegistry and `registry_lint.py` keep reading `classic_auto` documents against it. Owner decision; not done here.
- CR-PA-01 (split X_PIP) decides whether OBEL / additional PIP keep LOB codes.
