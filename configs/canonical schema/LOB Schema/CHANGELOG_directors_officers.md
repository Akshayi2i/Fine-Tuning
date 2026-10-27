# Changelog - directors_officers canonical schema

## 2.1.0 - 14 October 2026 (minor; draft for the project owner's approval)

**Why:** the source-data `epli` folder of ARC Excess & Surplus holds CNA's Investment Management Liability Solutions
policy (declarations CNA88758XX, insurer Columbia Casualty Company), not an employment practices policy. It is a
management liability policy with three coverage parts, read with this schema (CR-DO-09).

- **Coverage codes** (`directors_officers.coverage_codes.yaml`, 10 -> 13 codes, added at the end):
  `DO_INVESTMENT_ADVISER_PROFESSIONAL`, `DO_INVESTMENT_ADVISER_MANAGEMENT`, `DO_FUND_MANAGEMENT` (the three coverage
  parts; aggregate limits). New aliases: `Policy Aggregate Limit of Liability` on `DO_POLICY_AGGREGATE`;
  `Shareholder Derivative Demand Investigation Costs` (and the same with `Sublimit of Liability`) on
  `DO_DERIVATIVE_DEMAND`. No code renamed or removed; no path changes, so every 2.0.0 document stays valid.
- **Schema:** `fideon:schema_version` 2.1.0, `fideon:coverage_codes` lists the three new codes; description, status and
  `fideon:built_from` (9 seed golds) updated. Generated from the schema spec with the program's generator, as 2.0.0.
- **Registry:** `columbia_casualty/directors_officers.yaml` (new carrier): the declarations (Items 1-9), the cover
  letter and the endorsement fill-ins (private funds schedule, extended reporting options, vital executive expenses
  sublimit); 1 seed.
- **Seeds:** 1 DRAFT gold (`seeds/directors_officers/ARC Excess & Surplus, LLC/`, not in Git), pending reviewer
  sign-off. The 8 existing golds are unchanged and valid.
- **Pipeline:** `prompt/policy_check/directors_officers.prompt.json` and the schema hash fixture need regenerating
  (`tools/build_prompt_schemas.py`; `vlm_server.fsm.tests.test_compiler --update-fixture`).

## 2.0.0 - 12 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (directors and officers liability, L1 code and schema id `directors_officers`). It
replaces the pre-SPEC_21 `directors_officers.json` (1.5.0, `_common`-shaped with a `directors_officers` block), which
did not compose the common model. A new major version because every path changes (SPEC_21 4.1).

**Built from** the single-LOB documents of the source-data `do` folder, read in place: Wesco's Elements policy with
its private company D&O coverage element (filed under AmTrust), the Zurich Select Plus private company declarations
(ARC Excess & Surplus), Federal's personal director's liability declarations (Chubb), Hudson Excess' Private Defender
declarations (CNA Companies), Philadelphia Indemnity's FlexiPlus Five not-for-profit declarations and United States
Liability's community association D&O declarations in three terms (RT Specialty). Each of these schedules only its
D&O part; the other parts the forms offer are printed as not granted. RPS's document is a broker's proposal of
insurance (CR-DO-02).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `claims_made_terms` and
  `insuring_agreements` blocks. No LOB block.
- **Coverage codes:** `directors_officers.coverage_codes.yaml`, 10 codes with a provisional `DO_` prefix (CR-DO-01):
  the policy aggregate, the D&O coverage part (element), the insured person, company reimbursement and company
  insuring clauses (sides A, B, C), derivative demand investigation costs, adverse media event loss, the additional
  side A and defense costs limits, and retired independent directors.
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults, the claims-made rule kept.
- **Crosswalk:** `directors_officers.crosswalk.yaml` maps all 364 legacy leaves (221 to a SPEC_21 path, 136 to an
  `additional_fields` label, 7 dropped). `tools/registry_migrate.py --lob directors_officers`: no declared path
  needed rewriting.

**Registry** (`config/document_registry/policy/`):
- `amtrust/directors_officers.yaml`: was an unmeasured placeholder (sample_count 0); now a sectioned map of the
  Elements general declarations, the D&O coverage element declarations and the elements schedule, with the line
  signal of the element's title; 1 of 1 seed.
- `zurich/directors_officers.yaml` (new): the Select Plus private company declarations and form schedule; opts in to
  revise the `entry_fingerprint` line source by its coverage schedule heading; 1 seed.
- `chubb_federal/directors_officers.yaml`: was a fingerprint-only entry; now a sectioned map of the personal
  director's liability declarations (the 49 organizations read in page order with the new opt-in `index_width`);
  1 seed.
- `hudson/directors_officers.yaml` (new carrier): the Private Defender declarations; 1 seed.
- `philadelphia_indemnity/directors_officers.yaml` (new carrier): the common policy declarations, the FlexiPlus
  Five declarations and both forms schedules; 1 seed.
- `usli/directors_officers.yaml` (new): the policy declarations, forms extension and community association D&O
  coverage part declarations; 3 of 3 seeds.
- Every new or rewritten entry carries the shared `policy_fields` block.

**L1:** `document_fields: {pattern, all: true, index_width: N}` (opt-in, `fideon/l1/registry_extractor.py`): the
repeat index of every reading is zero-padded so readings under one `additional_fields` label keep page order; test
`fideon/l1/tests/test_sectioned_every_match_order.py`.

**Seeds:** 8 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/directors_officers/`, not in Git), pending
reviewer sign-off.

**Pipeline:** `config/canonical_schema/prompt/policy_check/directors_officers.prompt.json` generated with
`tools/build_prompt_schemas.py`; the schema hash fixture holds its hash. The fine-tuning repository still holds the
legacy file (CR-DO-06).
