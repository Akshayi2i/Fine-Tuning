# Changelog - ocean_marine canonical schema

## Common model 1.1.0 - 09 October 2026 (no version change)

Re-pinned to `common_model.json` 1.1.0: `fideon:common_model_version` "1.0.0" -> "1.1.0". No field change: `fideon:schema_version` stays 1.0.0, and the prompt schema (`config/canonical_schema/prompt/policy_check/ocean_marine.prompt.json`) is unchanged by the pin. Common model 1.x is additive (SPEC_21 4.2.7), so the schema composes 1.1.0 as it composed 1.0.0. The fine-tuning repository's copy of this schema carries the same pin since its commit c8df4bb (9 October 2026); with it the two files are byte-identical again, and both repositories' loaders compose it.

## 1.0.0 - 05 October 2026 (SPEC_21 Phase 2; awaiting project-owner approval)

First SPEC_21 schema of the line. It replaces the pre-SPEC_21 `ocean_marine.json` (3.1.0, pre-SPEC_21), which was generated from carrier-shaped blocks and did not compose the common model; that file is not carried over (SPEC_21 4.1).

**Built from** all 4 seed golds of 3 carriers (Markel American Insurance Company, Progressive, Travelers), the draft schema and coverage codes used for the SPEC_21 engine-test twins of this line (synth_core 0.1.2, batches TEST-SYN-PDF-PC-OM-901/-902), and the line's L1 registry YAMLs. Every gold of the line validates against this schema (check C1).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `watercraft`, `drivers`, `rating_modifiers`. Watercraft is part of this line: the L1 code `watercraft` reads this schema (`lob_schema_map.yaml`); boat operators are `drivers[]`.
- 167 core leaves (budget 80-300); 79 of them are printed on the current seeds. No LOB block: no concept the seeds print lacks a common-model home.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount`.
- `fideon:audit_rules`: the common model's default rules without `claims_made_terms.retroactive_date <= policy.effective_date` (no claims-made terms on a personal line), plus: `premium.total == sum(coverages[].premium) ± 1`
- `fideon:aliases`: printed labels per core field, read from every seed (`ocean_marine.fields.md`).
- 22 coverage codes (16 LOB, 6 shared `X_`) in `ocean_marine.coverage_codes.yaml`; every limit is inside `coverages[]`.
- Crosswalk `ocean_marine.crosswalk.yaml` from the registry paths: 21 new, 2 overflow, 0 drop.

**Changes from the engine-test draft:**
- Written into `config/canonical_schema/policy_check/` as the line's schema (the engine-test drafts lived in a scratch copy of `canonical_schema/` and said 'one-seed draft, not for policy_check').
- Description, status and `fideon:built_from` state the real basis (all seeds of the line); the draft text that named one or two seeds is gone.
- The default rule `claims_made_terms.retroactive_date <= policy.effective_date` is removed (no claims-made terms on a personal line, and the block is not composed).
- `fideon:aliases` added: printed labels per core field from every seed of the line.
- Coverage codes: one code per printed coverage; draft duplicates merged (`replaces_draft_codes` in the codes file), aliases taken from every seed's printed coverage names.
- `fideon:coverage_codes` is a plain list of the codes; the per-code detail lives only in the coverage codes file.
- The `allOf` enum on `coverages[].coverage_code` is removed: `fideon.schemas.schema_registry.conform_document` and `registry_lint.py` follow `$ref` only, so an `allOf` wrapper hid the coverages from both. The code list is the coverage codes file and `fideon:coverage_codes`.
- `fideon:common_model_change_requests` moved to `CHANGE_REQUESTS_ocean_marine.md`.

**Open for the project owner:**
- Approve the field list (`ocean_marine.fields.md`).
- CR-OM-01: add `vehicles` (boat trailer) to ocean_marine blocks_used.
