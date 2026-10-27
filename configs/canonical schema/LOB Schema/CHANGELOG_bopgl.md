# Changelog - bopgl canonical schema

## 2.3.0 - 26 October 2026 (additive; G1 scanned and hybrid seeds, change request approved by the project owner)

Source: the coverage rows the bopgl seed golds (Hartford, Liberty/Scottsdale) printed in `additional_fields` because no code named them (CR-BG-ALIAS-HRW-01..03, CR-BG-ALIAS-LM2-01/02).
Printed names only. Nothing is removed or renamed.

- **Aliases on existing codes:** `BOPGL_LIABILITY`, `BOPGL_PRODUCTS_COMPLETED_OPS`, `BOPGL_PERSONAL_INJURY`, `X_HIRED_NONOWNED_AUTO`, the business income and equipment rows that already had a code.
- **New codes (33):** coverage extensions and optional coverages of the Spectrum businessowners policy (arson and theft reward, brands and labels, fine arts, paved surfaces, data breach,
  cyber virus, employment practices, umbrella and others); counterparts on boppr where that line codes the same row.

## 2.2.0 - 25 October 2026 (additive; G1/bop review)

No path, code or alias is removed or renamed, so the 2.1.0 golds, varmaps, the registry entries' `canonical_paths` and the prompt/section files keep working. A seed signed at 2.0.0 or 2.1.0 stays valid (same major version, newer minor: DECISIONS row 117).

- **Evidence:** the `G1/bop` folder, 6 documents read in place: 2 businessowners declarations (Utica First, Businessowners Special Policy form BP 0200 with its Maximizer package endorsement; NYCM, form BP 00 03, a scanned file read by OCR) and 4 documents that are not businessowners (3 Utica First Contractors Special Policy documents, form AP 100, and 1 USLI commercial package policy; CR-BOPGL-07). Labels and codes only: no page text, name, address, number or amount is stored. BOP stays two parts, `bopgl` and `boppr` (DECISIONS row 89); no `bop` line is created.
- **Composition unchanged:** `common_model.json` 1.4.0, blocks `locations` and `buildings`. The review found no concept without a home in the common model that two or more lines print, so the common model is not changed (CR-BOPGL-10).
- **Aliases on existing codes (19).** A printed name of a concept the line already codes became an alias of that code: `Each Occurrence Limit`, the general aggregate caption and the NYCM `Liability and Medical Expenses` headings on `BOPGL_LIABILITY`; the products-completed operations aggregate caption; `Personal & Advertising Injury Limit`; `Fire Legal Liability Limit` on `BOPGL_DAMAGE_TO_PREMISES_RENTED`; `BUILDING COVERAGE`; `Additional Debris Removal`; `Fire Protective Devices Recharge`; `Lock Replacement`; `Glass Coverage`; `Refrigerated Food Products` on `BOPGL_SPOILAGE`; `Increased Cost - Ordinance or Law`; `Business Income - Extended Period of Indemnity`; `EQUIPMENT BREAKDOWN COVERAGE`; and on the shared entries `Nonowned Hired Auto Liability` (`X_HIRED_NONOWNED_AUTO`), `Lessor of Premises` and `Additional Insured - Lessor of Leased Equipment` (`X_ADDITIONAL_INSURED`).
- **New coverage codes (26, 82 codes now).**
  - Ten concepts no code covered on either line, each with a counterpart code on boppr: `BOPGL_BPP_OF_OTHERS`, `BOPGL_COMPUTER_COVERAGE`, `BOPGL_CREDIT_CARD_RECEIPTS`, `BOPGL_CREDIT_CARD_FORGERY`, `BOPGL_DEMOLITION`, `BOPGL_HAZARDOUS_SUBSTANCES`, `BOPGL_LIMITED_FUNGUS`, `BOPGL_LOSS_OF_EARNINGS`, `BOPGL_SIGNS_OFF_PREMISES`, `BOPGL_UTILITY_INTERRUPTION`.
  - Fifteen counterparts of boppr codes that bopgl lacked, with boppr's printed names cross-copied (the 2.1.0 rule: every row both lines print has a counterpart on the other): `BOPGL_ANTENNAS_SIGNS`, `BOPGL_BPP_OFF_PREMISES`, `BOPGL_BUILDING_OFF_PREMISES`, `BOPGL_DATA_RECORDS`, `BOPGL_DATA_RESTORATION`, `BOPGL_EXPEDITING_EXPENSES`, `BOPGL_IDENTITY_RECOVERY`, `BOPGL_INVENTORY_APPRAISAL`, `BOPGL_NEWLY_ACQUIRED_BPP`, `BOPGL_NEWLY_ACQUIRED_BUILDINGS`, `BOPGL_POLLUTANT_CLEANUP`, `BOPGL_SEWER_BACKUP`, `BOPGL_SIGNS`, `BOPGL_TRANSIT`, `BOPGL_TREES_SHRUBS`.
  - One shared entry listed for the line: `X_CYBER` (`Cyber Liability`, `Cyber Insurance Endorsement - New York`).
- **Counterpart table:** 43 entries before, 72 now (`counterpart` key, both lines).
- **Package endorsements are not coverages.** The Maximizer package endorsement, `Select Choice Plus`, `Policy Protector Pack` and `NYCM BOP Enhancement` print a premium beside a form number: `forms_and_endorsements[].premium`, as the coverage-code file's header already says for form-number rows (CR-BOPGL-09). The coverages they bundle are coded one by one.
- **Field captions (`fideon:aliases`): 18 labels added, 38 to 47 caption paths** (for example `policy.term_months`, `document.transaction_effective_date`, `locations[].distance_to_hydrant`, `locations[].distance_to_fire_station`, `buildings[].area_sq_ft`, `named_insured.entity_type`, `named_insured.business_description`, `billing.bill_type`, `carrier.name`). Each is a distinctive multi-word caption; generic one-word captions (`Agent`, `Code`, `Type`, `Email`) were left out.
- **Checked, nothing moved.** L1's printed-name table (`fideon.l1.spec21._codes`) and caption index (`registry_extractor._canonical_paths`) were taken before and after: no existing name changed code and no existing caption changed path; 66 names are new.
- **Not changed, and why:** default limit types of existing codes (L1 types a limit from the default only when a code lists exactly one type); the liability and property codes stay `BOPGL_` (the move to the shared `X_` codes is a major version, CR-BOPGL-01); the blocks used (classification rows: CR-BOPGL-08).
- **Derived files regenerated:** `prompt/policy_check/bopgl.prompt.json`, `prompt/sections/bopgl/*`, `prompt/manifest.json` (`tools/build_prompt_schemas.py`, `tools/build_section_schemas.py`), the FSM hash fixture.


## 2.1.0 - 23 October 2026 (additive; DECISIONS_2026-10-19 rows 114 and 115)

No path, code or alias is removed or renamed, so the 2.0.0 golds, varmaps, the registry entries' `canonical_paths` and the prompt/section files keep working. A seed's review stamp (`schema_version`) is 2.0.0 and must be renewed by the seed reviewer for 2.1.0 (list in the implementation log).

- **Composition:** `common_model.json` 1.4.0 (was 1.1.0). 1.2.0 added only the `ValueState` and `AbsenceReason` definitions, which no line property references; 1.4.0 adds six optional fields (`policy.effective_time`, `policy.expiration_time`, `carrier.company_type`, `document.copy_designation`, `named_insured.naics_code`, `named_insured.sic_code`) that this line inherits from the blocks it composes, and 1.3.0 the shared liability coverage codes `X_BI_PD_LIABILITY`, `X_PRODUCTS_COMPLETED_OPS`, `X_DAMAGE_TO_PREMISES_RENTED`, `X_PERSONAL_ADVERTISING_INJURY` and `X_LIQUOR_LIABILITY`; this line does not rename its own codes to them (that is a major version, 3.0.0, with crosswalk entries; CR-BOPGL-01).
- **Section I and Section II on one declarations page (CR-BOPGL-01, the property/liability split).** bopgl and boppr print the same businessowners declarations: Section I property beside Section II liability. Each row both lines print now has a counterpart code on the other line: the new `counterpart` key of `bopgl.coverage_codes.yaml` (43 pairs) names it, and the printed names of a row are aliases of both codes (the printed names of boppr's rows copied into bopgl's, and the reverse; 64 aliases added here). A document is coded under the prefix of the line it is read as; the shared X_ codes (`X_MED_PAY`, `X_EQUIPMENT_BREAKDOWN`, ...) are the same on both. No existing alias moved to another code.
- **New coverage codes (9, all additive):** `BOPGL_COLLAPSE`, `BOPGL_GLASS_EXPENSE`, `BOPGL_LEASE_ASSESSMENT`, `BOPGL_THEFT_DAMAGE_BUILDING`, `BOPGL_SPOILAGE`, `BOPGL_UTILITY_SERVICES_DIRECT`, `BOPGL_WATER_DAMAGE` (the ISO businessowners additional coverages whose headings print in 8 or more of the 57 bopgl documents, 2 or more carriers each, in the recurring-wording review below) and `BOPGL_AUTO_INCREASE_BUILDING`, `BOPGL_SEASONAL_INCREASE` (the percentage rows the Coterie seeds print as labelled values). 56 codes now.
- **New aliases:** `Business Income and Extra Expense` and `Business Income` on `BOPGL_LOSS_OF_INCOME`, `Extra Expense` on `BOPGL_ADDITIONAL_EXPENSE`, and the cross-copied row names above.
- **New field captions** (`fideon:aliases`): `billing.account_number` + `Account ID`; `deductibles[].percentage` + the windstorm or hail percentage deductible caption; `interested_parties[].loan_number` + `Loan Number` (the field already existed; the mortgagee address is `interested_parties[].address`, also existing); `producer.contact_name` + `CONTACT NAME`.
- **Not changed, and why:** the default limit types of every existing code stay as they are. L1 types a limit from the code's default only when the code lists exactly one type (`fideon/l1/spec21.py` `limits_from`), so a second default would turn every bare figure of that code into `other`. The Pollutant Cleanup, Fire Legal Liability and Money and Securities off-premises limit types of the Leatherstocking Preferred form therefore remain a reader rule (registry entry), see CHANGELOG_boppr.md.
- **Recurring-wording review (privacy: counts only).** `POLICIES-FINAL-V3` (original documents, local only): 57 distinct bopgl documents scanned for visible text; a line was kept only when it prints in 5 or more different documents of 2 or more carrier folders, has no digit run of 4 or more, and each of its words is used in 10 or more carrier folders. 290 title-case lines survived, 151 of them had no home in the schema, the codes or the registry captions; they are policy-form headings (no field), endorsement titles (`forms_and_endorsements[].title`), ISO additional-coverage headings (the codes above) and ACORD certificate captions (a different document type). Nothing else needs a new path.
- **Captions with no schema home after this release:** none that two or more lines print. One-carrier items stay in `additional_fields` (program administrator name and address, business class, covered premises and projects: Coterie only).
- **Derived files regenerated:** `prompt/policy_check/bopgl.prompt.json`, `prompt/sections/bopgl/*`, `prompt/manifest.json` (`tools/build_prompt_schemas.py`, `tools/build_section_schemas.py`), the FSM hash fixture.

## Codes and aliases - 12 October 2026 (package policies, PKG2/PKGF; no field change)

`bopgl.coverage_codes.yaml` (and the same list in `bopgl.json` `fideon:coverage_codes`, kept in alphabetical order), for the business liability part of The Hartford (Spectrum Business Owner's Policy) and Midstate Mutual businessowners package declarations (CR-PKG-06). New codes (9): `BOPGL_AGGREGATE_PER_PROJECT`, `BOPGL_BLANKET_ADDITIONAL_INSURED`, `BOPGL_CONTRACTORS_POLLUTION`, `BOPGL_CONTRACTOR_BROADENING`, `BOPGL_LIQUOR_LIABILITY`, `BOPGL_PERSONAL_INJURY`, `BOPGL_TENANTS_LEGAL_LIABILITY`, `X_ADDITIONAL_INSURED`, `X_WAIVER_OF_SUBROGATION`. New aliases: `BOPGL_LIABILITY` + `Cov L-Bodily Injury & Property Damage`; `BOPGL_PRODUCTS_COMPLETED_OPS` + `Food or Beverages for Consumption on Premises`; `X_MED_PAY` + `Cov M-Premises Medical Payments`. Aliases are the coverage names as the declarations print them. The prompt schemas are regenerated with `tools/build_prompt_schemas.py`.

## 2.0.0 - 10 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (businessowners liability, L1 code and schema id `bopgl`). It replaces the
pre-SPEC_21 `bopgl.json` 1.5.0, a re-base of `fideon.canonical.policy_core` with the repo's own
`business_liability_coverage_part`, `other_liability_coverages` and `common_forms` blocks, which was modelled
without a sample document and did not compose the common model. A new major version because every path changes
(SPEC_21 4.1); the legacy file is not carried over.

**Built from** the 13 single-LOB documents of the source-data `bopgl` folder, read in place, every field of the
legacy 1.5.0 file and the line's registry YAMLs (none declared canonical paths). Eight documents have a native
text layer (coterie 6, Leatherstocking Cooperative Insurance Company 2) and are the seeds; the other five (four
agency-system prints and one coterie policy) are page images under an invisible OCR text layer, so they are scans
and get no gold. Package folders (`bopgl_boppr`) are not read: package policies are out of scope.

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope (document, carrier, producer,
  named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing,
  forms_and_endorsements, additional_fields, text_sections, provenance) and the building blocks `locations` and
  `buildings`. Every bopgl document prints the whole businessowners declarations: the premises and buildings with
  their Section I property limits beside the Section II liability limits. No LOB block and no local `$defs`
  (`tools/check_lob_defs.py` passes).
- **Coverage codes:** `bopgl.coverage_codes.yaml`, 38 codes, 35 prefixed `BOPGL_` and 3 shared (`X_MED_PAY`,
  `X_HIRED_NONOWNED_AUTO`, `X_EQUIPMENT_BREAKDOWN`). Aliases are the coverage names and limit
  captions the seeds print, so L1 codes a printed row by its name. The business liability rows use `BOPGL_` codes
  until shared codes exist (CR-BOPGL-01).
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule, and an each-occurrence liability limit
  never exceeds its aggregate. No premium-sum rule: the coterie declarations print one policy premium and no
  coverage premiums.
- **Crosswalk:** `bopgl.crosswalk.yaml` maps all 325 legacy leaves (234 to a SPEC_21 path, 82 to an
  `additional_fields` label, 9 dropped as not values): the policy_core paths as `boppr.crosswalk.yaml` (then
  `gl.crosswalk.yaml`) maps them where the target is in `bopgl.json`, the line's own blocks by hand.
  `tools/registry_migrate.py --lob bopgl` had no declared path to rewrite.

**Registry:** `policy/coterie/bopgl.yaml` is new (the coterie businessowners declarations, both templates). The
`business_owners` entries are left as they are (see CHANGE_REQUESTS_bopgl.md CR-BOPGL-02).

**Seeds:** DRAFT golds (`synth_core/fideon/synth_core/data/seeds/bopgl/`, not in Git), pending reviewer sign-off
(Shachindra Thakar).

**Pipeline:** `config/canonical_schema/prompt/policy_check/bopgl.prompt.json` regenerated with
`tools/build_prompt_schemas.py`, with the prompt manifest. property_pkg has no adapter yet, so no recheck. The
fine-tuning repository still holds the legacy file (CR-BOPGL-04).
