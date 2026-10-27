# Changelog - wc canonical schema

## 2.2.0 - 26 October 2026 (additive; G1 scanned and hybrid seeds)

- **Alias:** `WC_EMPLOYERS_LIABILITY` gains `Limits Emp Liability` (the Guard proposal prints the Part Two limits under that caption). No path or code is removed or renamed.

## 2.1.0 - 23 October 2026 (additive; decisions 114/115, POLICIES-FINAL-V3 recurring wording)

Source: the distinct originals of POLICIES-FINAL-V3 (83 workers compensation documents), read in place for recurring form wording only. A caption
counts when it prints in at least 5 different documents of at least 3 different carriers (lower-cased, digits as 9);
lines with a digit run of 4 or more, street, ZIP, company-name or agency-name patterns, or unfamiliar words are dropped.
Counts only; no document text or value is kept, in the repository or in the schema. Nothing is removed or renamed.

- **Common model pin:** `fideon:common_model_version` raised from 1.1.0 to 1.4.0 (the newest release; 1.2.0 to 1.4.0 are additive: value-state definitions, shared liability codes, six optional fields), which this line inherits without a path change. The FSM hash and the prompt schema are regenerated.
- **Two additions to the `wc` block (`$defs/WorkersCompensationTerms`):** `total_estimated_cost` (`Total Estimated Cost`;
  8 documents of 3 carriers; the premium plus the state assessment and other surcharges, which `premium.total` does not hold)
  and `officers[]` (name, title, ownership_percent, included, class_code, remuneration; `Name of Officer(s)`, 5 documents of 3
  carriers). Both were `additional_fields` labels before; the six legacy `workers_compensation.officers_and_owners[]` paths
  now map to `wc.officers[]` in the crosswalk (CR-WC-05 resolved locally). `tools/check_lob_defs.py` reports no duplicate
  definition. Core leaves 173 -> 180.
- **Aliases (no field change):** `billing.due_date` + `Premium Due Date`; `countersignature.representative_name` +
  `Authorized Representative`; `premium.surcharges[].amount` + `New York State Assessment`;
  `rating_exposures[].exposure_amount` + `Premium Basis`; the new fields' own aliases.
- **Crosswalk:** `to_schema` 2.1.0; the six `officers_and_owners[]` entries move from `overflow` to `new`.
- **No coverage code added:** the V3 originals print Part One, Part Two and Part Three only; the New York Construction
  Classification Premium Adjustment Program, premium due date, interest and service charge, catastrophe and state assessment
  endorsements are premium lines and forms already held by `premium.discounts[]`, `billing`, `premium.taxes_fees[]`,
  `premium.items[]`, `premium.surcharges[]` and `forms_and_endorsements[]`.
- **Common-model requests (not applied here):** CR-WC-08 to CR-WC-10 in `CHANGE_REQUESTS_wc.md`; the concepts two or more lines
  print (policy period time, carrier company type, insurer officer names, NAICS / SIC, terrorism disclosure figures) go to the
  common model, and wc takes them through its envelope.
- **Seeds (decision needed):** the same version-stamp effect as gl 3.1.0 (CHANGELOG_gl.md): the 17 wc seeds that passed seed mode read the same,
  the only new problems are `C13 review schema version 2.0.0, current 2.1.0` and `varmap schema version 2.0.0, run uses 2.1.0`.
- **Derived files not regenerated here:** `prompt/policy_check/wc.prompt.json`, `prompt/sections/wc/*.json` and the prompt
  manifest; the common-model owner's build regenerates them.

## 2.0.0 amendment - 19 October 2026 (M2.9 fix 4; no version change)

- **Mandatory fields:** `coverages[].limits[].amount` is replaced by `coverages[].coverage_name`: a coverage
  row, with or without an amount. The New York State Insurance Fund's Information Page prints no Part Two
  limits, only the Workers Compensation and Employers Liability policy it attaches to, so every NYSIF read
  failed the line's own mandatory check once the audit gate was given the line. `nysif/wc.yaml` now emits
  the Part One and Part Two rows (`WC_PART_ONE`, `WC_EMPLOYERS_LIABILITY`) from that title, with no amount.
  Rule 5 (`len(coverages[].limits[]) >= 1`, a warn) is unchanged.
- Derived files regenerated: `prompt/sections/wc/catch_all.json`, `canonical_schema_compressed.json`
  (`tools/build_section_schemas.py`); the prompt schema is unchanged (`build_prompt_schemas.py --check`).

## 2.0.0 - 10 October 2026 (draft for the owner's approval; first SPEC_21 version, common model 1.1.0)

Replaces the self-contained pre-SPEC_21 `wc.json` 1.6.0 (`fideon.canonical.workers_compensation`, generated
from `workers_compensation.schema.json` by `scripts/build_canonical_policy.py`). Major version: every path of
1.6.0 moves (SPEC_21 4.1); `wc.crosswalk.yaml` maps all 382 leaf paths of 1.6.0 and the registry's declared
paths (266 `new`, 113 `overflow`, 5 `drop`).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the envelope (document, carrier, producer, named_insured,
  policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements,
  additional_fields, text_sections) plus `locations`, `rating_exposures`, `rating_modifiers` and
  `countersignature`. 173 core leaves.
- **LOB-local block `wc`** (`$defs/WorkersCompensationTerms`, SPEC_21 4.2.4): `part_one_states[]` (Item 3.A),
  `other_states_insurance` (Item 3.C wording), `ncci_company_number`, `bureau_file_number` (risk id / R.B. file
  number). `tools/check_lob_defs.py`: no duplicate definition.
- **Coverages** (`wc.coverage_codes.yaml`): `WC_PART_ONE` (statutory, no limit), `WC_EMPLOYERS_LIABILITY` (three
  limits: `per_accident`, `policy_limit`, `each_employee`, each limit's `description` its printed caption),
  `WC_OTHER_STATES`. The 1.6.0 `workers_compensation.item_3_coverage` fields map to these.
- **Classification schedule:** `rating_exposures[]` (class code, description, remuneration as
  `exposure_amount`, rate per $100, premium, state, location_ref); experience, schedule and merit rating are
  `rating_modifiers[]`; the expense constant, terrorism and catastrophe charges are `premium.items[]`, the state
  assessment / total surcharges `premium.surcharges[]`.
- **Overflow (no core home, in `additional_fields`):** NAICS / SIC codes, number of employees, total standard
  premium, total estimated cost, officers included or excluded, rate components. Candidates for the common model
  are in `CHANGE_REQUESTS_wc.md`.
- **Audit rules:** the common model's five default rules plus "Part Two disease each-employee limit <= disease
  policy limit".
- **Mandatory fields:** the common model's minimum (carrier name, named insured, policy number, effective and
  expiration dates, a coverage limit).

**Registry:** the 38 `policy/*/wc.yaml` entries pass `tools/registry_lint.py --lob wc --strict`;
`tools/registry_migrate.py --lob wc --check` reports no path to change (only `utica_first/wc.yaml` declares
paths, already 2.0.0 paths or overflow). `hartford/wc.yaml` and `amtrust/wc.yaml` gained SPEC_21 field maps
(sectioned declarations) checked against the DRAFT seed golds (C14).

**Seeds:** 31 DRAFT golds in `synth_core/fideon/synth_core/data/seeds/wc/` (Hartford 19, AmTrust 4, NYSIF 8),
pending reviewer sign-off (Shachindra Thakar). The 8 documents whose text is an invisible OCR layer over
vector-drawn pages (Hartford `hartford_wc`, `hartford_wc_dec`, `hartford_wc_endo`, `hartford_wc_endo_2`,
`the_hartford_wc_3`; AmTrust `amtrust_wc_endo`, `_endo_1`, `_endo_2`) and the scanned Berkshire Hathaway GUARD
policy are not drafted (SPEC21_TEAM_PROMPT Phase 0: the owner decides scanned_pdf or leave out).

**Pipeline:** `config/canonical_schema/prompt/policy_check/wc.prompt.json` generated (new; wc was served from
`_fallback` before), the prompt-schema manifest and `_fallback.json` regenerated. casualty_fleet has no adapter
yet, so no recheck. Training parity: see CR-WC-01.
