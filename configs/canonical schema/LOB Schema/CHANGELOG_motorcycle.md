# Changelog - motorcycle canonical schema

## Common model 1.1.0 - 09 October 2026 (no version change)

Re-pinned to `common_model.json` 1.1.0: `fideon:common_model_version` "1.0.0" -> "1.1.0". No field change: `fideon:schema_version` stays 1.0.0, and the prompt schema (`config/canonical_schema/prompt/policy_check/motorcycle.prompt.json`) is unchanged by the pin. Common model 1.x is additive (SPEC_21 4.2.7), so the schema composes 1.1.0 as it composed 1.0.0. The fine-tuning repository's copy of this schema carries the same pin since its commit c8df4bb (9 October 2026); with it the two files are byte-identical again, and both repositories' loaders compose it.

## 1.0.0 - 05 October 2026 (SPEC_21 Phase 2; awaiting project-owner approval)

First SPEC_21 schema of the line. It replaces the pre-SPEC_21 `motorcycle.json` (1.5.0, pre-SPEC_21), which was generated from carrier-shaped blocks and did not compose the common model; that file is not carried over (SPEC_21 4.1).

**Built from** all 9 seed golds of 3 carriers (All State, Progressive, RT Specialty), the draft schema and coverage codes used for the SPEC_21 engine-test twins of this line (synth_core 0.1.2, batches TEST-SYN-PDF-PC-MC-901/-902), and the line's L1 registry YAMLs. Every gold of the line validates against this schema (check C1).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `locations`, `vehicles`, `drivers`, `rating_modifiers`. 
- 180 core leaves (budget 80-300); 79 of them are printed on the current seeds. No LOB block: no concept the seeds print lacks a common-model home.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount`.
- `fideon:audit_rules`: the common model's default rules without `claims_made_terms.retroactive_date <= policy.effective_date` (no claims-made terms on a personal line), plus: `premium.minimum <= premium.total`; `premium.total == sum(premium.items[unit_type=coverage].amount) + sum(coverages[].premium) ± 1`
- `fideon:aliases`: printed labels per core field, read from every seed (`motorcycle.fields.md`).
- 16 coverage codes (8 LOB, 8 shared `X_`) in `motorcycle.coverage_codes.yaml`; every limit is inside `coverages[]`.
- Crosswalk `motorcycle.crosswalk.yaml` from the registry paths: 0 new, 0 overflow, 0 drop.

**Changes from the engine-test draft:**
- Written into `config/canonical_schema/policy_check/` as the line's schema (the engine-test drafts lived in a scratch copy of `canonical_schema/` and said 'one-seed draft, not for policy_check').
- Description, status and `fideon:built_from` state the real basis (all seeds of the line); the draft text that named one or two seeds is gone.
- The default rule `claims_made_terms.retroactive_date <= policy.effective_date` is removed (no claims-made terms on a personal line, and the block is not composed).
- `fideon:aliases` added: printed labels per core field from every seed of the line.
- Coverage codes: one code per printed coverage; draft duplicates merged (`replaces_draft_codes` in the codes file), aliases taken from every seed's printed coverage names.
- `fideon:coverage_codes` is a plain list of the codes; the per-code detail lives only in the coverage codes file.
- Supplementary uninsured/underinsured motorists names filed under `X_UM` in the golds are `X_UIM` (common model alias 'Supplementary Uninsured/Underinsured Motorists'); OBEL names filed under `X_PIP` are `MC_OPTIONAL_BASIC_ECONOMIC_LOSS`.
- The policy-term rule is kept (four seeds print the term).

**Seed golds to re-code at review** (the coverage code a gold uses differs from the canonical code; the gold still validates, the code only changes the comparison key):

- `All State/all_state_cycle`: X_UM -> X_UIM
- `All State/all_state_recv_snowmobile_5`: X_UM -> X_UIM
- `RT Specialty/rt_specialty_cycle_2`: X_PIP -> MC_OPTIONAL_BASIC_ECONOMIC_LOSS (x2)

**Open for the project owner:**
- Approve the field list (`motorcycle.fields.md`).
- L1 routing of the motorcycle seeds (Suraj's CR-1, to the L1 owner): Progressive and All State seeds route to auto; American Reliable needs the page-one pattern; until fixed, C14 differences are expected.
- The motorcycle registry YAMLs declare no `canonical_path`; Phase 3 has nothing to migrate, but `registry_lint --strict` will report every produced field without a path.
