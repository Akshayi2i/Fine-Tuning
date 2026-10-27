# Changelog - boppr canonical schema

## 2.3.0 - 26 October 2026 (additive; G1 scanned and hybrid seeds, change request approved by the project owner)

- **Aliases:** `X_HIRED_NONOWNED_AUTO` (Hired/Non-Owned Auto Liability per occurrence/aggregate), `BOPPR_PERIOD_OF_LOSS_EXTENSION` (Extended Business Income - Extended Number of Days).
- **New codes (5):** `BOPPR_DO_LIABILITY`, `BOPPR_EPL`, `BOPPR_FIDUCIARY_LIABILITY` (the management liability part of a U.S. Underwriters package) and the counterparts of new bopgl codes.

## 2.2.0 - 25 October 2026 (additive; G1/bop review)

No path, code or alias is removed or renamed, so the 2.1.0 golds, varmaps, the registry entries' `canonical_paths` and the prompt/section files keep working. A seed signed at 2.0.0 or 2.1.0 stays valid (same major version, newer minor: DECISIONS row 117).

- **Evidence:** the `G1/bop` folder, 6 documents read in place: 2 businessowners declarations (Utica First, Businessowners Special Policy form BP 0200 with its Maximizer package endorsement; NYCM, form BP 00 03, a scanned file read by OCR) and 4 documents that are not businessowners (3 Utica First Contractors Special Policy documents, form AP 100, and 1 USLI commercial package policy; CR-BOPPR-10). Labels and codes only: no page text, name, address, number or amount is stored. BOP stays two parts, `bopgl` and `boppr` (DECISIONS row 89); no `bop` line is created.
- **Composition unchanged:** `common_model.json` 1.4.0, blocks `locations` and `buildings`. The common model is not changed (CR-BOPPR-08; the review is in CR-BOPGL-10).
- **Aliases on existing codes (23).** The same names as bopgl for the shared rows (the `Liability and Medical Expenses` headings and `Each Occurrence Limit` on `BOPPR_LIABILITY`, the products-completed operations, personal and advertising injury and `Fire Legal Liability Limit` captions, `BUILDING COVERAGE`, `Additional Debris Removal`, `Fire Protective Devices Recharge`, `Lock Replacement`, `Glass Coverage`, `Refrigerated Food Products`, `Increased Cost - Ordinance or Law` on `X_ORDINANCE_LAW`, `EQUIPMENT BREAKDOWN COVERAGE`), and for rows boppr already coded: `Backup of Sewer and Drains` (`BOPPR_SEWER_BACKUP`), `Property In Transit` (`BOPPR_TRANSIT`), `Business Property At Newly Acquired Locations` and `Business Property - Newly Acquired` (`BOPPR_NEWLY_ACQUIRED_BPP`), `Business Property - Off Premises` (`BOPPR_BPP_OFF_PREMISES`), `EXPEDITING EXPENSE`, `IDENTITY RECOVERY ENDORSEMENT`, `Business Income - Extended Period of Indemnity` (`BOPPR_PERIOD_OF_LOSS_EXTENSION`).
- **New coverage codes (13, 109 codes now).** The ten concepts listed in CHANGELOG_bopgl.md (`BOPPR_BPP_OF_OTHERS`, `BOPPR_COMPUTER_COVERAGE`, `BOPPR_CREDIT_CARD_RECEIPTS`, `BOPPR_CREDIT_CARD_FORGERY`, `BOPPR_DEMOLITION`, `BOPPR_HAZARDOUS_SUBSTANCES`, `BOPPR_LIMITED_FUNGUS`, `BOPPR_LOSS_OF_EARNINGS`, `BOPPR_SIGNS_OFF_PREMISES`, `BOPPR_UTILITY_INTERRUPTION`), and three shared entries listed for the line: `X_HIRED_NONOWNED_AUTO`, `X_ADDITIONAL_INSURED`, `X_CYBER`.
- **Counterpart table:** 43 entries before, 72 now; the 15 boppr codes that had no bopgl counterpart now name one.
- **Package endorsements are not coverages** (Maximizer, `Select Choice Plus`, `Policy Protector Pack`, `NYCM BOP Enhancement`): `forms_and_endorsements[].premium` (CR-BOPPR-09).
- **Field captions (`fideon:aliases`): 16 labels added, 33 to 40 caption paths.** Same rule as bopgl: distinctive captions only.
- **Checked, nothing moved.** L1's printed-name table and caption index were taken before and after: no existing name changed code and no existing caption changed path; 41 names are new.
- **Not changed:** default limit types, the `BOPPR_` liability codes (CR-BOPPR-01), the blocks used (CR-BOPPR-10).
- **Derived files regenerated:** `prompt/policy_check/boppr.prompt.json`, `prompt/sections/boppr/*`, `prompt/manifest.json`, the FSM hash fixture; `boppr.fields.md` updated.


## 2.1.0 - 23 October 2026 (additive; DECISIONS_2026-10-19 rows 114 and 115)

No path, code or alias is removed or renamed, so the 2.0.0 golds, varmaps, the registry entries' `canonical_paths` and the prompt/section files keep working. A seed's review stamp (`schema_version`) is 2.0.0 and must be renewed by the seed reviewer for 2.1.0 (list in the implementation log).

- **Composition:** `common_model.json` 1.4.0 (was 1.1.0). 1.2.0 added only the `ValueState` and `AbsenceReason` definitions, which no line property references; 1.4.0 adds six optional fields (`policy.effective_time`, `policy.expiration_time`, `carrier.company_type`, `document.copy_designation`, `named_insured.naics_code`, `named_insured.sic_code`) that this line inherits from the blocks it composes, and 1.3.0 the shared liability coverage codes `X_BI_PD_LIABILITY`, `X_PRODUCTS_COMPLETED_OPS`, `X_DAMAGE_TO_PREMISES_RENTED`, `X_PERSONAL_ADVERTISING_INJURY` and `X_LIQUOR_LIABILITY`; this line does not rename its own codes to them (that is a major version, 3.0.0, with crosswalk entries; CR-BOPPR-01).
- **Leatherstocking Preferred form (SF-10P), settled for the schema.** Its rows were added as aliases on 22 October (DECISIONS row 87). Decisions: (1) Money and Securities on and off premises is ONE coverage, `BOPPR_MONEY_SECURITIES`, with two limits: the on-premises figure `per_occurrence` and the off-premises figure a `sublimit` whose description is the printed row name (as the four Preferred seeds' golds have it); no second code. (2) Pollutant Cleanup is `per_occurrence` and Fire Legal Liability `per_location` on that form (the golds), while the code defaults stay `aggregate` and `per_occurrence` for the other carriers: a default change is not additive, because L1 types a bare figure from the code's single default (`limits_from`), so every other boppr document would change. The Preferred-form types are therefore a reader rule for the registry entry `leatherstocking/boppr.yaml` (the L1 owner), not the code list. (3) Loss payee / mortgagee: the schema already has `interested_parties[].address` and `interested_parties[].loan_number`; the open gap (the address block and the `#` prefix of the loan number) is a reader rule too. `interested_parties[].loan_number` + `Loan Number` is added as a field caption.
- **Section I and Section II on one declarations page (CR-BOPPR-01, the property/liability split).** Each row both lines print has a counterpart code on the other line (`counterpart` key, 43 pairs) and the same printed names (36 aliases added here from bopgl's list: the `Liability and Medical Expenses` limit heading, `Damage to Premises Rented To You`, the Classic Coverage M rows and others). A document is coded under the prefix of the line it is read as.
- **New coverage code (1):** `BOPPR_DATA_RESTORATION` (the data restoration row of the cyber insurance endorsement, 7 of 19 documents, 3 carriers). 96 codes now.
- **New aliases:** `Business Income and Extra Expense` on `BOPPR_LOSS_OF_INCOME`, `Extra Expense` on `BOPPR_ADDITIONAL_EXPENSE`, and the cross-copied names above.
- **built_from:** 8 seed golds (the four Leatherstocking Preferred-form documents moved from property on DECISIONS rows 87 and 89).
- **Recurring-wording review (privacy: counts only).** `POLICIES-FINAL-V3`: 19 distinct boppr documents; same rule as for bopgl; 48 title-case lines survived, 19 without a home: endorsement titles (`forms_and_endorsements[].title`), page furniture, and the printed `Incl.` value (a value state, `value_states`). Nothing else needs a new path. Captions with no schema home: none.
- **Derived files regenerated:** `prompt/policy_check/boppr.prompt.json`, `prompt/sections/boppr/*`, `prompt/manifest.json`, the FSM hash fixture; `boppr.fields.md` updated.

## Codes and aliases - 12 October 2026 (package policies, PKG2/PKGF; no field change)

`boppr.coverage_codes.yaml` (and the same list in `boppr.json` `fideon:coverage_codes`, kept in alphabetical order), for the property part of The Hartford (Spectrum Business Owner's Policy) and Midstate Mutual businessowners package declarations (CR-PKG-06). New codes (49): `BOPPR_ADDED_WATER_DAMAGE`, `BOPPR_APPURTENANT_STRUCTURES`, `BOPPR_ARSON_THEFT_REWARD`, `BOPPR_BI_OFF_PREMISES_OPERATIONS`, `BOPPR_BI_OFF_PREMISES_UTILITY`, `BOPPR_BI_WEBSITES`, `BOPPR_BRANDS_LABELS`, `BOPPR_BUILDING_PROPERTY_OF_OTHERS`, `BOPPR_CIVIL_AUTHORITY`, `BOPPR_CLAIM_EXPENSE`, `BOPPR_COLLAPSE`, `BOPPR_COMPUTERS_WORLDWIDE`, `BOPPR_COMPUTER_OPERATIONS`, `BOPPR_CONTRACT_PENALTIES`, `BOPPR_CYBER_VIRUS_MALWARE`, `BOPPR_ELECTRONIC_DATA`, `BOPPR_EMPLOYEE_DISHONESTY`, `BOPPR_EXPEDITING_EXPENSES`, `BOPPR_FINE_ARTS`, `BOPPR_FRAUDULENT_TRANSFER`, `BOPPR_GLASS_EXPENSE`, `BOPPR_IDENTITY_RECOVERY`, `BOPPR_LEASEHOLD_IMPROVEMENTS`, `BOPPR_LEASE_ASSESSMENT`, `BOPPR_LOST_KEYS`, `BOPPR_MONEY_SECURITIES`, `BOPPR_NEWLY_ACQUIRED_PROPERTY`, `BOPPR_NON_OWNED_TRAILERS`, `BOPPR_OUTDOOR_PROPERTY`, `BOPPR_OUTDOOR_SIGNS`, `BOPPR_PAIRS_SETS`, `BOPPR_PAVED_SURFACES`, `BOPPR_PRESERVATION_OF_PROPERTY`, `BOPPR_PROPERTY_AWAY`, `BOPPR_PROPERTY_OFF_PREMISES`, `BOPPR_SALESPERSONS_SAMPLES`, `BOPPR_SEWER_BACKUP`, `BOPPR_SIGNS`, `BOPPR_SPOILAGE`, `BOPPR_SPRINKLER_LEAKAGE`, `BOPPR_STRETCH_BLANKET`, `BOPPR_SUMP_OVERFLOW`, `BOPPR_THEFT_DAMAGE_BUILDING`, `BOPPR_TRANSIT`, `BOPPR_TRANSIT_BI`, `BOPPR_UNAUTHORIZED_CARD_USE`, `BOPPR_UTILITY_SERVICES_DIRECT`, `BOPPR_VALUATION_CHANGES_STOCK`, `BOPPR_WATER_DAMAGE`. New aliases: `BOPPR_AUTO_INCREASE_BUILDING` + `BUILDING INFLATION PROTECTION`; `BOPPR_BPP` + `Cov B - Business Property`; `BOPPR_SEASONAL_INCREASE` + `SEASONAL VARIATION`; `X_ORDINANCE_LAW` + `Ordinance and Law Coverage`. Aliases are the coverage names as the declarations print them. The prompt schemas are regenerated with `tools/build_prompt_schemas.py`.

## 2.0.0 - 10 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (businessowners property, L1 code and schema id `boppr`). It replaces the
pre-SPEC_21 `boppr.json` 1.5.0, a re-base of `fideon.canonical.policy_core` 1.4.0 with three repo-made blocks
(`common_forms`, `locations`, `property_coverage_part`) that was modelled without a sample document
(`fideon:verification.registry_entries: 0`) and did not compose the common model. A new major version because
every path changes (SPEC_21 4.1); the legacy file is not carried over.

**Built from** the 5 single-LOB documents of the source-data `boppr` folder, read in place (Leatherstocking
Cooperative Insurance Company 2, Madison Mutual 2, Reinsure 1), every field of the legacy 1.5.0 file and the
line's one L1 registry YAML (`policy/leatherstocking/boppr.yaml`, which declared no canonical paths). Package
folders (`bopgl_boppr`, `boppr_cgl`) are not read: package policies are out of scope.

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope (document, carrier, producer,
  named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing,
  forms_and_endorsements, additional_fields, text_sections, provenance) and the property-line building blocks
  `locations` and `buildings` (as dwelling_fire uses them). No LOB block and no local `$defs`
  (`tools/check_lob_defs.py` passes). 170 core leaves.
- **Coverage codes:** `boppr.coverage_codes.yaml`, 46 codes prefixed `BOPPR_`, 3 of them shared
  (`X_MED_PAY` with this line's own default limit type `per_person`, `X_EQUIPMENT_BREAKDOWN`,
  `X_ORDINANCE_LAW`). Aliases are the coverage names the seeds print, so L1 codes a printed row by its name.
  The business liability rows a businessowners declarations prints (Coverage L, N, O, P) are coded here under
  `BOPPR_` codes until shared codes exist (CR-BOPPR-01).
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule; an each-occurrence liability limit
  never exceeds its aggregate; the premium rule (coverage, form, item, surcharge, discount and tax/fee amounts
  add up to the total), which holds on all 5 seeds.
- **Crosswalk:** `boppr.crosswalk.yaml` maps all 333 legacy leaves (policy_core paths as `gl.crosswalk.yaml` maps
  them, except the buildings fields and the gl-only blocks; the 28 boppr-only paths by hand). `tools/registry_migrate.py
  --lob boppr` had no path to rewrite (the registry YAML declared none).

**Seeds:** 5 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/boppr/`, not in Git), pending reviewer sign-off
(Shachindra Thakar). One of them (Reinsure) is an agency-system print whose text layer is invisible over a drawn
page: its modality is a question for the project owner (see its review.md).

**Pipeline:** `config/canonical_schema/prompt/policy_check/boppr.prompt.json` is new (regenerated with
`tools/build_prompt_schemas.py`), with the prompt manifest. property_pkg has no adapter yet, so no recheck. The
fine-tuning repository still holds the legacy file (CR-BOPPR-02).
