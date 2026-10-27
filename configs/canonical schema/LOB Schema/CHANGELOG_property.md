# Changelog - property canonical schema

## Aliases - 15 October 2026 (Dryden Mutual commercial application; no field or code change)

`property.coverage_codes.yaml`, the property part of Dryden Mutual's Commercial Application (read by `dryden_mutual/property.yaml`): `PR_BUILDING` + `Building Coverage`; `PR_BPP` + `Contents Coverage`; `PR_LOSS_OF_INCOME` + `Loss of Earnings`. The code list, `property.json` and the prompt schema are unchanged.

## Codes and aliases - 12 October 2026 (package policies, PKG2; no field change)

`property.coverage_codes.yaml` (and the same list in `property.json` `fideon:coverage_codes`), for the property part of the Dryden Mutual (`Commercial Fire & Liability Policy`, `Craft 12 - Contractors Package Policy`) and North Country (`Commercial Special Multi-Peril Policy Declaration`) package declarations. New codes: `PR_LOSS_OF_RENTS` (`Loss Of Rents`), `PR_CONDOMINIUM_LOSS` (`Condominium Loss`), `PR_CRAFTS_PAK_BASIC` (`Craft12 - Basic Coverage`), `PR_CREDIT_CARD_FORGERY` (the Crafts Pak credit card / forgery / counterfeit money rows, per occurrence and per policy period), `PR_PROPERTY_OF_OTHERS`, `PR_TOOLS_EQUIPMENT`. New aliases: `PR_BUILDING` + `Coverage A - Building`; `PR_BPP` + `Coverage B - Business Property`, `Cov B - Business Property`, `Coverage B Business Property`; `PR_PROPERTY_AWAY` + `Business Property Off Premises`; `PR_UTILITY_SERVICE` + `Utility Services Interruption - For Time Element Coverages Including Transmission Lines`; `PR_LOSS_OF_INCOME` + `Loss of Income`; `X_EQUIPMENT_BREAKDOWN` + `Equipment Breakdown Enhancement Endorsement`. The prompt schemas are regenerated centrally by the PKG2 agent.

## 2.0.0 - 10 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (commercial property, L1 code and schema id `property`, batch code PR). It replaces
the pre-SPEC_21 `property.json` (`fideon.canonical.commercial_property` 1.5.0, a policy_core re-base with the repo's
`commercial_property` block), which did not compose the common model and was never resolved. A new major version
because every path changes (SPEC_21 4.1).

**Built from** the 11 single-LOB documents of the source-data `prop` folder, read in place. Six print a commercial
property or businessowners declarations and are the seeds: Leatherstocking's Businessowner's Policy with the SF-10P
Preferred supplement (4) and its Commercial Fire and Liability declarations (1), and Merchants' businessowners
declarations (1). Four are the Leatherstocking Landlords Package / Student Housing product, which
`policy/leatherstocking/homeowners.yaml` reads as homeowners (two homeowners seeds are the same product), and one
(North Country's Commercial Special Multi-Peril declarations) is a property-and-liability package read by
`policy/north_country/commercial_package.yaml`; neither group gets a property gold (CR-PROP-01).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `locations`, `buildings`
  and `rating_modifiers` blocks (the lob_schema_map entry). No LOB block and no local `$defs`.
- **Coverage codes:** `property.coverage_codes.yaml`, 40 codes: 33 prefixed `PR_` and 7 shared (`X_ORDINANCE_LAW`, `X_EPL`,
  `X_EQUIPMENT_BREAKDOWN`, `X_MED_PAY`, `X_CYBER`, `X_HIRED_NONOWNED_AUTO`). The liability rows the seeds print beside
  their property use `PR_` codes until shared codes exist (CR-PROP-02).
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule.
- **Crosswalk:** `property.crosswalk.yaml` maps all 426 legacy leaves (287 to a SPEC_21 path with the two
  whole-array paths `travelers/property.yaml` declares, 135 to an `additional_fields` label, 6 dropped).
  `tools/registry_migrate.py --lob property`: no declared path needed rewriting (40 entries).

**Registry:** no property entry reads a seed: the Leatherstocking Businessowner's documents stay declined pending
the product ruling (CR-BOPPR-06), the Commercial Fire and Liability declarations is the Commercial Fire (C-Fire)
line's (owner ruling, 10 October 2026; read by the `commercial_fire` entry), and Merchants has no property entry.

**Seeds:** 6 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/property/`, not in Git), pending reviewer sign-off.

**Pipeline:** `config/canonical_schema/prompt/policy_check/property.prompt.json` regenerated with
`tools/build_prompt_schemas.py`. property_pkg has no adapter yet. The fine-tuning repository still holds the legacy
file (CR-PROP-03).
