# Change requests - health (to the common-model owner, the training-side owner, the registry owner and the project owner)

Raised while building the health SPEC_21 schema 1.0.0 (10 October 2026). Nothing here is applied to the common
model: the schema composes `common_model.json` 1.1.0 unchanged. Labels only; no document values.

**CR-HEALTH-01. statutory_disability reads health.json (project owner).**
The corpus files the New York statutory disability (DBL) and paid family leave (PFL) policies under health, and L1
routes them as `statutory_disability` (the carrier's own title). `lob_schema_map.yaml` now gives that code
`schema_lob: health` (as `watercraft` reads `ocean_marine`), so both registry entries of the code
(`shelterpoint`, `hartford`) are read and linted against `health.json`. If the owner prefers a schema of its own for
statutory disability, the crosswalk and the Shelterpoint entry move to it unchanged in shape.

**CR-HEALTH-02. `lob_schema_map.yaml` health entry.**
`blocks_used: [benefit_plans]` and `source_pdfs: {folder: health, carriers: 3, files: 7}` are filled; `lob_code`,
`routing_key` and `coverage_code_prefix` stay as the owner sets them (the files use `HEALTH_` meanwhile). The SPEC_06
open item (the `group` routing key, health or group_accident_sickness) is unchanged.

**CR-HEALTH-03. Training parity (fine-tuning repository owner).**
The fine-tuning repository has no health schema. benefits_misc has no adapter and no parity test; health is served
by the base model until the family is migrated.

**CR-HEALTH-04. Benefit plan types (common-model owner).**
`BenefitPlan.plan_type` lists life, add, std, ltd, medical, dental, vision and other. A statutory DBL plan is drafted
as `std` and a PFL plan as `other`; the owner may add `statutory_disability` and `paid_family_leave` (two carriers'
DBL/PFL documents would justify it under the inclusion rule once a second carrier prints them).

**CR-HEALTH-05. Weekly limits (common-model owner).**
DBL, PFL and the volunteer weekly disability benefits are weekly amounts; `LimitType` has `per_day` but no
`per_week`, so they are `other` with the printed qualifier as the description. Adding `per_week` would type them.

**CR-HEALTH-06. Scans filed as native (project owner).**
Three health documents (one Glatfelter policy, two agency-system prints) are page images under an invisible OCR
text layer although the manifest records them as `native_pdf`. L1 does not read them.
