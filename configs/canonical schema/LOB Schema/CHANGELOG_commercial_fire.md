# Changelog - commercial_fire canonical schema

## 1.2.0 - 27 October 2026 (additive; commercial-line template audit: Leatherstocking Student Housing, SMP and Manufacturers and Contractors declarations)

Composes common model 1.5.0 (was 1.1.0); the common model's new optional fields are available to this line without a path change.

- New code CF_ROOF_SURFACE_ACV (form row, no limit); alias `SF-312A - Loss of Income/Rents - Special Limitations` (CF_LOSS_OF_RENTS); the block `rating_exposures` is adopted (the rating-information lines: class code, description, exposure basis and amount).
- Derived files regenerated: `prompt/policy_check/commercial_fire.prompt.json`, `prompt/sections/commercial_fire/*`, `prompt/manifest.json`, the FSM hash fixture.

## 1.1.0 - 26 October 2026 (additive; Leatherstocking Student Housing, SMP and Manufacturers and Contractors declarations)

Printed names only; nothing is removed or renamed, the common model (1.1.0) is untouched. These rows coded to nothing, so `require_coded_coverages` had to be switched off in `commercial_fire.yaml`.

- **New codes (14 + 1 shared):** `CF_RELATED_PRIVATE_STRUCTURE`, `CF_ACCOUNTS_RECEIVABLE`, `CF_ADDITIONAL_EXPENSE`, `CF_EMPLOYEE_DISHONESTY`, `CF_SIGNS`, `CF_MONEY_SECURITIES`, `CF_VALUABLE_PAPERS`, `CF_PROPERTY_AWAY` (also `SF-133 Business Property - While Away From The Insured Premise`), `CF_BUILDING_VMM`, `CF_LOSS_OF_RENTS_BUILDING` (`per_occurrence`); `CF_LOSS_OF_RENTS`, `CF_SEASONAL_VARIATION`, `CF_SPRINKLER_LEAKAGE` (printed `***`, no default limit type); `CF_PRODUCTS_COMPLETED_OPS` (`per_occurrence`, `products_completed_ops_aggregate`; `LS-43 Products/Completed Operations` and both `Coverage N - Products (...)` rows); the shared `X_EQUIPMENT_BREAKDOWN` (`Equipment Breakdown`, `SF-345 Equipment Breakdown`, `SF-345 - Equipment Breakdown`).
- **Aliases:** `Personal Injury` (`CF_PERSONAL_INJURY`), `SM-26 Automatic Increase, ACV` (`CF_AUTO_INCREASE`).
- **Not coverages, no code:** `Claim Free Renewal Credit`, `Leatherstocking Credit` (discounts), `Student Housing Surcharge` (surcharge), `SF-124`, `SF-312A - Loss of Income/Rents - Special Limitations` and the `SF-10D` note (form or note rows), the `Coverage L / M / N - Manufacturers and Contractors, Classification 1` rows (premium.items of `CF_LIABILITY` / `X_MED_PAY` / `CF_PRODUCTS_COMPLETED_OPS`, a registry matter). `Fire Legal Liability` is already `X_DAMAGE_TO_PREMISES_RENTED`.
- **Derived files regenerated:** `prompt/policy_check/commercial_fire.prompt.json`, `prompt/sections/commercial_fire/*`, `prompt/manifest.json`, the FSM hash fixture, the fine-tuning parity fixtures (re-derived locally; the fine-tuning repository must regenerate from the same canonical files).

## 1.0.0 - 10 October 2026 (SPEC_21; draft for the project owner's approval)

First schema of a new line. The project owner ruled on 10 October 2026 that Leatherstocking Cooperative Insurance
Company's "Commercial Fire and Liability" product (`DECLARATION, Commercial Fire and Liability` on the LCIC(4/11)
declarations form) is a line of business of its own, Commercial Fire, and not Businessowners Property, Commercial
Property or a package (resolving `CHANGE_REQUESTS_boppr.md` CR-BOPPR-06), and approved new canonical schemas.
L1 code and schema id `commercial_fire`, display "Commercial Fire" (`config/lob_registry.yaml`,
`fideon.schemas.fields.LOB_DISPLAY`). There is no legacy schema, so there is no crosswalk.

**Built from** the seven documents of the source data that print the banner, read in place: one filed under `boppr`
(a boppr DRAFT seed until this ruling), one under `prop` (a property DRAFT seed until this ruling) and five under
`cgl_prop`, all of one carrier. Same form set and layout; three templates of it (a three-page print with the OLT
classifications on page 1 and no Coverage B; a print whose insured name sits on its caption's line; the rest).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope (document, carrier, producer,
  named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing,
  forms_and_endorsements, additional_fields, text_sections, provenance) and the property-line building blocks
  `locations` and `buildings`, as boppr and property use them. No LOB block and no local `$defs`
  (`tools/check_lob_defs.py` passes).
- **Coverage codes:** `commercial_fire.coverage_codes.yaml`, 11 codes prefixed `CF_`, one shared (`X_MED_PAY`, with
  this line's default limit type `per_person`). Section I: building fire and extended coverage (Coverage A), business
  property fire, extended coverage, vandalism and SF-4 special form (Coverage B). Section II: premises liability
  (Coverage L, `CF_LIABILITY`, per occurrence and aggregate) and medical payments (Coverage M). Optional items: LS-49
  personal injury, SM-26 automatic increase, SF-27B replacement cost. The liability rows use `CF_` codes until shared
  business liability codes exist (CR-CFIRE-01). Exclusion rows priced against a form (LS-59, LS-89, LS-118) are form
  premiums; the OLT classification rows are `premium.items` of Coverage L and M.
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule; Coverage L occurrence <= aggregate;
  Coverage M per person <= per accident; the premium rule (coverage, form, item, surcharge, discount and tax/fee
  amounts add up to the total), which holds on all seven seeds.

**Seeds:** 7 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/commercial_fire/`, not in Git), pending reviewer
sign-off (Shachindra Thakar). The boppr and property DRAFT golds of the two documents filed under those lines were
moved here (redrafted against this schema) and removed from those lines' inventories.

**Registry:** `config/document_registry/policy/leatherstocking/commercial_fire.yaml` (new) claims the banner
(`lob_signals`) and reads all seven at L1; the reading rules came from `leatherstocking/boppr.yaml`, which no longer
carries them.

**Pipeline:** `config/canonical_schema/prompt/policy_check/commercial_fire.prompt.json` is new (regenerated with
`tools/build_prompt_schemas.py`), with the prompt manifest. property_pkg has no adapter yet, so no recheck; the
fine-tuning repository has no file for the line (CR-CFIRE-03).
