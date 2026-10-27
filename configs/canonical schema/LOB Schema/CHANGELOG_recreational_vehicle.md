# Changelog - recreational_vehicle canonical schema

## Coverage code aliases - 15 October 2026 (no version change)

Printed names added as aliases in `recreational_vehicle.coverage_codes.yaml`, as agency-management-system printouts (`VEHICLE COVERAGES` grid) print them, under the code the line's golds use: `Collision Coverage` (`X_COLLISION`), `GL` (`RV_GUEST_LIABILITY`, the guest liability row of an all-terrain vehicle's grid) and `Personal Injury Protection/Basic Economic Loss` (`X_PIP`). No code, field or default limit type changes; the schema and its prompt schema are unchanged. Each alias names one code only.

## Coverage code alias - 14 October 2026 (no version change)

`Supplementary Uninsured / Underinsured` added as an alias of `X_UIM`, as American Modern's motorsports declarations (form MS-CW-D-0001, Policy Coverages) print the coverage name with its last word (`Motorist`) wrapped onto the next line. No code, field or default limit type changes; the schema and its prompt schema are unchanged. The alias names one code only.

## Common model 1.1.0 - 09 October 2026 (no version change)

Re-pinned to `common_model.json` 1.1.0: `fideon:common_model_version` "1.0.0" -> "1.1.0". No field change: `fideon:schema_version` stays 1.0.0, and the prompt schema (`config/canonical_schema/prompt/policy_check/recreational_vehicle.prompt.json`) is unchanged by the pin. Common model 1.x is additive (SPEC_21 4.2.7), so the schema composes 1.1.0 as it composed 1.0.0. The fine-tuning repository's copy of this schema carries the same pin since its commit c8df4bb (9 October 2026); with it the two files are byte-identical again, and both repositories' loaders compose it.

## 1.0.0 - 05 October 2026 (SPEC_21 Phase 2; awaiting project-owner approval)

First SPEC_21 schema of the line. It replaces the pre-SPEC_21 `recreational_vehicle.json` (1.5.0, pre-SPEC_21), which was generated from carrier-shaped blocks and did not compose the common model; that file is not carried over (SPEC_21 4.1).

**Built from** all 22 seed golds of 4 carriers (All State, American Modern, Foremost Insurance Company, Progressive), the draft schema and coverage codes used for the SPEC_21 engine-test twins of this line (synth_core 0.1.2, batches TEST-SYN-PDF-PC-RV-901/-902), and the line's L1 registry YAMLs. Every gold of the line validates against this schema (check C1).

**Contents:**
- Composes `common_model.json` 1.0.0 by `$ref`: the envelope (document, carrier, producer, named_insured, policy, lob_parts, coverages, deductibles, interested_parties, premium, billing, forms_and_endorsements) plus `locations`, `vehicles`, `drivers`, `rating_modifiers`. `watercraft` (listed for recreational_vehicle in `lob_schema_map.yaml`) is left out: no RV seed insures a boat (Sakshi's Q-RV-01 asks the same).
- 180 core leaves (budget 80-300); 95 of them are printed on the current seeds. No LOB block: no concept the seeds print lacks a common-model home.
- `fideon:mandatory_fields`: `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount`.
- `fideon:audit_rules`: the common model's default rules without `claims_made_terms.retroactive_date <= policy.effective_date` (no claims-made terms on a personal line), plus: `premium.total == sum(coverages[].premium) ± 1`
- `fideon:aliases`: printed labels per core field, read from every seed (`recreational_vehicle.fields.md`).
- 26 coverage codes (18 LOB, 8 shared `X_`) in `recreational_vehicle.coverage_codes.yaml`; every limit is inside `coverages[]`.
- Crosswalk `recreational_vehicle.crosswalk.yaml` from the registry paths: 6 new, 4 overflow, 0 drop.

**Changes from the engine-test draft:**
- Written into `config/canonical_schema/policy_check/` as the line's schema (the engine-test drafts lived in a scratch copy of `canonical_schema/` and said 'one-seed draft, not for policy_check').
- Description, status and `fideon:built_from` state the real basis (all seeds of the line); the draft text that named one or two seeds is gone.
- The default rule `claims_made_terms.retroactive_date <= policy.effective_date` is removed (no claims-made terms on a personal line, and the block is not composed).
- `fideon:aliases` added: printed labels per core field from every seed of the line.
- Coverage codes: one code per printed coverage; draft duplicates merged (`replaces_draft_codes` in the codes file), aliases taken from every seed's printed coverage names.
- `fideon:coverage_codes` is a plain list of the codes; the per-code detail lives only in the coverage codes file.
- `watercraft` removed: no RV seed insures a boat.
- Supplementary uninsured/underinsured motorists names filed under `X_UM` are `X_UIM`; 'Mandatory Pedestrian Personal Injury Protection' filed under `X_PIP` is `RV_PEDESTRIAN_PIP`.
- The policy-term rule is kept (ten seeds print the term).

**Seed golds to re-code at review** (the coverage code a gold uses differs from the canonical code; the gold still validates, the code only changes the comparison key):

- `All State/all_state_recv_snowmobile`: X_UM -> X_UIM (x2)
- `All State/all_state_recv_snowmobile_1`: X_UM -> X_UIM (x2)
- `All State/all_state_recv_snowmobile_2`: X_UM -> X_UIM (x3)
- `All State/all_state_recv_snowmobile_3`: X_UM -> X_UIM (x4)
- `All State/all_state_recv_snowmobile_4`: X_UM -> X_UIM (x3)
- `All State/all_state_recv_snowmobile_hawksoft`: X_UM -> X_UIM (x2)
- `American Modern/american_modern_autop_snowmobil_motosport`: X_UM -> X_UIM
- `American Modern/american_modern_recv_snowmobile`: X_UM -> X_UIM
- `American Modern/american_modern_recv_snowmobile_1`: X_UM -> X_UIM
- `American Modern/american_modern_recv_snowmobile_2`: X_UM -> X_UIM
- `American Modern/american_modern_snowmobile`: X_UM -> X_UIM
- `Foremost Insurance Company/foremost_rv`: X_UM -> X_UIM
- `Progressive/progressive_autop`: X_PIP -> RV_PEDESTRIAN_PIP; X_UM -> X_UIM
- `Progressive/progressive_recv`: X_UM -> X_UIM
- `Progressive/progressive_recv_3`: X_PIP -> RV_PEDESTRIAN_PIP; X_UM -> X_UIM

**Open for the project owner:**
- Approve the field list (`recreational_vehicle.fields.md`).
- `watercraft` in `lob_schema_map.yaml` blocks_used for recreational_vehicle: drop it (Sakshi's Q-RV-01).
- Seed PDF `All State/recv/all_state_recv.pdf` was renamed on disk to `All State/recv/progressive_recv.pdf` (same document, policy number matches the gold); the gold, varmap, review and run configs still name `all_state_recv`. Refile per Sakshi's Q-RV-02 and update the artefacts together.
- Reconcile with Sakshi's uncommitted draft (180 leaves, 8 shared + 15 RV codes, watercraft left out too).
