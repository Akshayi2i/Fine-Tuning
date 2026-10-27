# Changelog - auto canonical schema (commercial auto)

## Alias - 15 October 2026 (Merchants Mutual commercial auto policy; no field change)

`auto.coverage_codes.yaml`: `X_UIM` + `Supplementary Uninsured/Underinsured`, the SUM row name as the first line of Merchants Mutual's Item Two schedule (MU 72 50) prints it in a full commercial automobile policy (read by `merchants_mutual/auto.yaml`). No new code; `auto.json` is unchanged.

## Alias - 12 October 2026 (Merchants Mutual policy change endorsement; no field change)

`auto.coverage_codes.yaml`: `AU_FULL_GLASS` + `Full Safety Glass Coverage`, the coverage name as Merchants Mutual's schedule of autos and related coverages (MU 72 51) prints it in a commercial automobile policy change packet (read by `merchants_mutual/auto.yaml`, now a measured entry). No new code; `auto.json` and the prompt schema are unchanged (`tools/build_prompt_schemas.py --check`: current).

## 2.0.0 - 10 October 2026 (SPEC_21; draft for the owner's approval)

`auto.json` is rewritten as a SPEC_21 schema on common model 1.1.0. It replaces the generated pre-SPEC_21 file
(`fideon.canonical.commercial_auto` 1.5.0, built by `scripts/build_canonical_policy.py` from a supplied template),
which was self-contained and did not compose the common model.

- **Envelope and blocks:** the common envelope plus `locations` (garaging), `vehicles`, `drivers`,
  `rating_exposures`, `rating_modifiers` and `countersignature`, all by `$ref` to `common/common_model.json`. The
  vehicles and drivers blocks are the ones personal_auto uses. No LOB block and no local `$defs`
  (`tools/check_lob_defs.py`: no duplicate definitions).
- **Built from** the labels the line's 11 single-LOB seed documents of 5 carriers print (Glatfelter / National
  Union, The Hartford, Merchants Mutual, Preferred Mutual, Progressive; `source data/<Carrier>/autob/`, read in
  place), every field of the legacy schema and the line's registry YAMLs (`*/auto.yaml`, `progressive/autob.yaml`).
- **How a business auto policy is held:**
  - Item Two (schedule of coverages and covered autos) and Progressive's outline of coverage: one `coverages[]`
    entry per coverage, with `covered_auto_symbols`, `limits`, `deductibles`, `premium` and `included`.
  - Item Three (schedule of covered autos you own): one `vehicles[]` entry per auto; the auto's own limits,
    deductibles and INCL / No Coverage marks are `coverages[]` entries with `applies_to` the vehicle; the auto's
    premium by coverage is `premium.items[]` (`unit_type: coverage`, `coverage_code`, `applies_to`), its total a
    `unit_type: vehicle` item. Item Two's premiums are the policy totals, so the per-auto premiums are their
    breakdown and are not counted a second time in `coverages[]`.
  - Items Four and Five (hired and non-owned autos): an `AU_HIRED_AUTO_LIABILITY` /
    `AU_NON_OWNERSHIP_LIABILITY` coverage plus a `rating_exposures[]` row (cost of hire, rate, number of
    employees, premium).
  - A vehicle's printed garaging town, state, ZIP and territory are a `locations[]` entry
    (`garaging_location_ref`).
- **Coverage codes** (`auto.coverage_codes.yaml`, 23): the shared `X_VEHICLE_LIABILITY`, `X_PIP`, `X_MED_PAY`,
  `X_UM`, `X_UIM`, `X_COMPREHENSIVE`, `X_COLLISION`, `X_TOWING`, `X_RENTAL`, `X_ADDITIONAL_INSURED`,
  `X_WAIVER_OF_SUBROGATION`, and `AU_` codes for the New York no-fault parts printed as their own coverages
  (`AU_ADDITIONAL_PIP`, `AU_OPTIONAL_BASIC_ECONOMIC_LOSS`, `AU_AGGREGATE_NO_FAULT_BENEFITS`, as personal_auto
  keeps them apart), `AU_SUPPLEMENTAL_SPOUSAL_LIABILITY`, `AU_SPECIFIED_CAUSES_OF_LOSS`,
  `AU_HIRED_AUTO_LIABILITY`, `AU_NON_OWNERSHIP_LIABILITY`, `AU_HIRED_AUTO_PHYSICAL_DAMAGE`,
  `AU_DOWNTIME_RENTAL_TOWING`, `AU_FULL_GLASS`, `AU_COVERAGE_EXTENSION` and `AU_OTHER_AUTO_COVERAGES`. Aliases are
  the printed coverage names; L1 codes a printed row by them (`fideon/l1/spec21.py coverage_code_for`).
- **Mandatory fields and audit rules:** the common model's minimum mandatory fields; `fideon:default_audit_rules`
  without the claims-made rule (commercial auto is occurrence-based).

Major version: a new schema shape (SPEC_21 4.1). Crosswalk `auto.crosswalk.yaml` (`from_schema` 1.5.0,
`to_schema` 2.0.0): all 496 legacy leaves plus the two registry paths that are not legacy leaves, 498 entries
(352 `new`, 140 `overflow`, 6 `drop`). The envelope paths the legacy file shares with the legacy gl.json 1.5.0
(same generator) are mapped as `gl.crosswalk.yaml` maps them; `coverages[].coverage_type` is dropped as the line
itself. Drops awaiting reviewer approval: `document.document_type`, `document.source_carrier_folder`,
`document.source_system`, `signature.has_signature`, `coverages[].coverage_type`,
`auto.no_fault_benefits.is_applicable`.

`registry_migrate --lob auto`: no registry path changes (the line's 40 entries declare 9 canonical paths, in
`utica_first/auto.yaml`, all at the same SPEC_21 path). `registry_lint --lob auto --strict`: OK on all 40.

**Seeds:** 11 DRAFT seed golds (`seeds/auto/<Carrier>/`), pending reviewer sign-off (Shachindra Thakar).

**Pipeline:** `auto.prompt.json` generated (new), with the prompt-schema manifest and a new
`policy_check/auto` hash in `vlm_server/fsm/tests/fixtures/schema_hashes.json`. casualty_fleet has no adapter
yet. Training parity: CR-AU-01.
