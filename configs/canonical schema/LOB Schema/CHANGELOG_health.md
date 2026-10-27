# Changelog - health canonical schema

## 1.0.0 - 10 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First schema of the line (health, L1 code and schema id `health`); no `health.json` existed before, so health
documents were read against `_fallback`. The L1 code `statutory_disability` now reads this schema too
(`common/lob_schema_map.yaml`, `schema_lob: health`), because the corpus files the New York statutory disability
(DBL) and paid family leave (PFL) policies under health; its pre-SPEC_21 `statutory_disability.json` 3.1.0 was never
resolved (legacy files are not usable) and is the `from_schema` of the crosswalk.

**Built from** the 7 single-LOB documents of the source-data `health` folder, read in place. Four have a native text
layer and are the seeds: two Shelterpoint DBL/PFL policy packets and two Glatfelter Insurance Group (VFIS program,
National Union) volunteer accident and sickness policies. The other three (one Glatfelter, two agency-system prints)
are page images under an invisible OCR text layer: scans, no gold.

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `benefit_plans` block
  (a plan's printed benefit formula, maximum, waiting period, contribution and rate). No LOB block and no local
  `$defs` (`tools/check_lob_defs.py` passes).
- **Coverage codes:** `health.coverage_codes.yaml`, 37 codes prefixed `HEALTH_`: the DBL and PFL benefits and the
  35 benefits a volunteer accident and sickness schedule prints. Aliases are the printed benefit names without the
  schedule's outline numbering.
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule.
- **Crosswalk:** `health.crosswalk.yaml` maps all 651 leaves of the legacy `statutory_disability.json` (the paths the
  statutory_disability registry entries were written against): 225 to a SPEC_21 path, 421 to an `additional_fields`
  label, 5 dropped.

**Registry:** `policy/shelterpoint/statutory_disability.yaml` rewritten against `health.json` (the Policy Summary,
Benefit Summary, policy jacket and changes rider, application, DB-820/829 certificates and notices);
`policy/glatfelter/health.yaml` is new (the VFIS schedule of coverage, forms and policyholders).

**Seeds:** 4 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/health/`, not in Git), pending reviewer sign-off
(Shachindra Thakar).

**Pipeline:** `config/canonical_schema/prompt/policy_check/health.prompt.json` is new (regenerated with
`tools/build_prompt_schemas.py`), with the prompt manifest. benefits_misc has no adapter and no parity test covers
health (CR-HEALTH-03).
