# Changelog - personal_umbrella canonical schema

## Common model 1.1.0 - 09 October 2026 (no version change)

Re-pinned to `common_model.json` 1.1.0: `fideon:common_model_version` "1.0.0" -> "1.1.0". No field change: `fideon:schema_version` stays 1.0.0, and the prompt schema (`config/canonical_schema/prompt/policy_check/personal_umbrella.prompt.json`) is unchanged by the pin. Common model 1.x is additive (SPEC_21 4.2.7), so the schema composes 1.1.0 as it composed 1.0.0. The fine-tuning repository's copy of this schema carries the same pin since its commit c8df4bb (9 October 2026); with it the two files are byte-identical again, and both repositories' loaders compose it.

## 1.0.0 - 05 October 2026 (SPEC_21 Phase 2; awaiting project-owner approval)

First SPEC_21 schema of the line. It replaces the pre-SPEC_21 `personal_umbrella.json` (3.2.0, pre-SPEC_21), which was generated from carrier-shaped blocks and did not compose the common model; that file is not carried over (SPEC_21 4.1).

**Built from** all 4 seed golds of 2 carriers (NYCM Insurance, Plymouth), the draft schema and coverage codes used for the SPEC_21 engine-test twins of this line (synth_core 0.1.2, batches TEST-SYN-PDF-PC-PU-901/-902), and the line's L1 registry YAMLs. Every gold of the line validates against this schema (check C1).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `underlying_insurance`. `attachment` and `rating_modifiers` (listed for personal_umbrella in `lob_schema_map.yaml`) are left out: no seed fills them; the printed retention is a `retention` deductible of the umbrella coverage. Residences and exposure counts (Khushi's CR-PU-4) stay in `additional_fields` until the project owner decides.
- 149 core leaves (budget 80-300); 60 of them are printed on the current seeds. No LOB block: no concept the seeds print lacks a common-model home.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[coverage_code=PU_LIABILITY].limits[limit_type=per_occurrence].amount`.
- `fideon:audit_rules`: the common model's default rules without `claims_made_terms.retroactive_date <= policy.effective_date` (no claims-made terms on a personal line), plus: `premium.total == sum(coverages[].premium) ± 1`; `len(underlying_insurance[]) >= 1`; ``coverages[coverage_code=PU_LIABILITY].deductibles[deductible_type=retention].amount` <= `coverages[coverage_code=PU_LIABILITY].limits[limit_type=per_occurrence].amount``
- `fideon:aliases`: printed labels per core field, read from every seed (`personal_umbrella.fields.md`).
- 3 coverage codes (3 LOB, 0 shared `X_`) in `personal_umbrella.coverage_codes.yaml`; every limit is inside `coverages[]`.
- Crosswalk `personal_umbrella.crosswalk.yaml` from the registry paths: 17 new, 4 overflow, 2 drop.

**Changes from the engine-test draft:**
- Written into `config/canonical_schema/policy_check/` as the line's schema (the engine-test drafts lived in a scratch copy of `canonical_schema/` and said 'one-seed draft, not for policy_check').
- Description, status and `fideon:built_from` state the real basis (all seeds of the line); the draft text that named one or two seeds is gone.
- The default rule `claims_made_terms.retroactive_date <= policy.effective_date` is removed (no claims-made terms on a personal line, and the block is not composed).
- `fideon:aliases` added: printed labels per core field from every seed of the line.
- Coverage codes: one code per printed coverage; draft duplicates merged (`replaces_draft_codes` in the codes file), aliases taken from every seed's printed coverage names.
- `fideon:coverage_codes` is a plain list of the codes; the per-code detail lives only in the coverage codes file.
- `attachment` and `rating_modifiers` removed: no seed fills them (the one-seed draft composed them).
- The mandatory limit is the umbrella limit itself (`PU_LIABILITY` per occurrence), which every seed prints.
- `fideon:draft_coverage_codes` moved to `personal_umbrella.coverage_codes.yaml`.
- `lob_parts` is required again (every gold carries it; the draft dropped it).

**Open for the project owner:**
- Approve the field list (`personal_umbrella.fields.md`).
- Khushi's CR-PU-4 (locations, rating_exposures) and CR-PU-3 (required underlying limits): decisions change this schema in a minor version.
- Reconcile with Khushi's uncommitted draft (177 leaves, `PU_UMBRELLA_LIABILITY`): see the reconciliation notes handed to the owner.
