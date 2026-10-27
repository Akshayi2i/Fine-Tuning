# Changelog - dwelling_fire canonical schema

## Coverage codes, default limit types - 10 October 2026 (no version change)

`dwelling_fire.coverage_codes.yaml`: `default_limit_types` of three codes, the kind L1 gives a printed limit when nothing printed in its cell or its coverage name decides (`fideon.l1.spec21.limits_from`; no other reader uses the field):
- `DF_COV_L` `[per_occurrence, aggregate]` -> `[per_occurrence]`. Two defaults never decide, so every unqualified premises-liability limit came out `other`; every carrier of the line prints an aggregate with its own qualifier (`Aggregate Limit`), and the bare figure is the occurrence limit (Leatherstocking `(Each Occurrence)` beside the cell, North Country `Occ`).
- `X_LOSS_OF_USE` `[per_occurrence, other]` -> `[per_occurrence]`, as every seed of the line records it.
- `X_MED_PAY` `[per_occurrence, per_person, per_accident]` -> `[per_person, per_accident]`: the order of SPEC_21 4.2.2's split limit, so North Country's `$5,000 Per/$10,000 Acc` reads as per person / per accident. A qualifier printed in the name or cell still decides first (Leatherstocking's `- Per Occurrence` row stays per occurrence).

No field, code or alias changes, so `fideon:schema_version` stays 1.0.0 and the prompt schema is unchanged (it does not carry the codes file). The fine-tuning repository's copy of the codes file is now behind by these three lines: CR-DF-03.

## Common model 1.1.0 - 09 October 2026 (no version change)

Re-pinned to `common_model.json` 1.1.0: `fideon:common_model_version` "1.0.0" -> "1.1.0". No field change: `fideon:schema_version` stays 1.0.0, and the prompt schema (`config/canonical_schema/prompt/policy_check/dwelling_fire.prompt.json`) is unchanged by the pin. Common model 1.x is additive (SPEC_21 4.2.7), so the schema composes 1.1.0 as it composed 1.0.0. The fine-tuning repository's copy of this schema carries the same pin since its commit c8df4bb (9 October 2026); with it the two files are byte-identical again, and both repositories' loaders compose it.

## 1.0.0 - 05 October 2026 (SPEC_21 Phase 2; awaiting project-owner approval)

First SPEC_21 schema of the line. It replaces the pre-SPEC_21 `dwelling_fire.json` (1.5.0, pre-SPEC_21), which was generated from carrier-shaped blocks and did not compose the common model; that file is not carried over (SPEC_21 4.1).

**Built from** all 16 seed golds of 4 carriers (Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company), the draft schema and coverage codes used for the SPEC_21 engine-test twins of this line (synth_core 0.1.2, batches TEST-SYN-PDF-PC-DF-901/-902), and the line's L1 registry YAMLs. Every gold of the line validates against this schema (check C1).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `locations`, `buildings`. `rating_modifiers` (listed for dwelling_fire in `lob_schema_map.yaml`) is left out: no dwelling fire seed fills it; the printed credits and surcharges are `premium.discounts[]`/`surcharges[]`.
- 168 core leaves (budget 80-300); 95 of them are printed on the current seeds. No LOB block: no concept the seeds print lacks a common-model home.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount`.
- `fideon:audit_rules`: the common model's default rules without `claims_made_terms.retroactive_date <= policy.effective_date` (no claims-made terms on a personal line), plus: ``coverages[coverage_code=DF_COV_C].limits[limit_type=per_occurrence].amount` <= `coverages[coverage_code=DF_COV_A].limits[limit_type=per_occurrence].amount``; `premium.total == sum(coverages[].premium) + sum(forms_and_endorsements[].premium) + sum(premium.items[].amount) + sum(premium.surcharges[].amount) + sum(premium.taxes_fees[].amount) ± 0.01`
- `fideon:aliases`: printed labels per core field, read from every seed (`dwelling_fire.fields.md`).
- 17 coverage codes (14 LOB, 3 shared `X_`) in `dwelling_fire.coverage_codes.yaml`; every limit is inside `coverages[]`.
- Crosswalk `dwelling_fire.crosswalk.yaml` from the registry paths: 19 new, 4 overflow, 0 drop.

**Changes from the engine-test draft:**
- Written into `config/canonical_schema/policy_check/` as the line's schema (the engine-test drafts lived in a scratch copy of `canonical_schema/` and said 'one-seed draft, not for policy_check').
- Description, status and `fideon:built_from` state the real basis (all seeds of the line); the draft text that named one or two seeds is gone.
- The default rule `claims_made_terms.retroactive_date <= policy.effective_date` is removed (no claims-made terms on a personal line, and the block is not composed).
- `fideon:aliases` added: printed labels per core field from every seed of the line.
- Coverage codes: one code per printed coverage; draft duplicates merged (`replaces_draft_codes` in the codes file), aliases taken from every seed's printed coverage names.
- `fideon:coverage_codes` is a plain list of the codes; the per-code detail lives only in the coverage codes file.
- `rating_modifiers` removed (no seed fills it).
- The Dryden Mutual application gold used `HO_COV_A/B/C` and `X_PERSONAL_LIABILITY` for 'Premises Liability' are mapped to `DF_COV_A/B/C` and `DF_COV_L`.
- The policy-term rule is kept: NYCM prints the term.

**Seed golds to re-code at review** (the coverage code a gold uses differs from the canonical code; the gold still validates, the code only changes the comparison key):

- `Dryden Mutual/dryden_mutual_dfire_app`: HO_COV_A -> DF_COV_A (x2); HO_COV_B -> DF_COV_B (x2); HO_COV_C -> DF_COV_C (x2); X_PERSONAL_LIABILITY -> DF_COV_L (x2)

**Open for the project owner:**
- Approve the field list (`dwelling_fire.fields.md`).
- `rating_modifiers` in `lob_schema_map.yaml` blocks_used: drop it for dwelling_fire, or name the documents that print rating modifiers (common-model owner edits the map).
- L1 routes only the Leatherstocking seeds to dwelling_fire (registry_inventory_dwelling_fire.csv); Dryden Mutual, NYCM and North Country have no dwelling_fire routing yet (Phase 3, L1 owner).
