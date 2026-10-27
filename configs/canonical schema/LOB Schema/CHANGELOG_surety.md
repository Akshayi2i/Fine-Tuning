# Changelog - surety canonical schema

## 4.0.0 - 12 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (surety bonds, L1 code and schema id `surety`). It replaces the pre-SPEC_21
`surety.json` (3.1.0, `_common`-shaped with a `surety` block), which did not compose the common model. A new major
version because every path changes (SPEC_21 4.1).

**Built from** the single-LOB documents of the source-data `sure` folder, read in place: Western Surety Company's New
York liquor license or permit bonds (Bond Form L-9) with their powers of attorney, 4 documents of 2 pages each. Every
page is a scan under an OCR text layer, so no seed gold was drafted and L1 does not read them (CR-SU-02).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `bond` block (bond number,
  type, penal sum, principal, obligee, term). No LOB block.
- **Coverage codes:** `surety.coverage_codes.yaml`, 3 codes with a provisional `SU_` prefix (CR-SU-01): the license or
  permit bond the sources print, and the contract and other commercial bond categories of the legacy schema. A bond
  is one coverage whose penal sum is its `policy_limit`.
- **Mandatory fields:** `carrier.name` (the surety), `named_insured.primary_name` (the principal),
  `policy.policy_number` (the bond number) and `coverages[].limits[].amount` (the penal sum).
- **Audit rules:** the bond's own: its term in order, the bond number equal to the policy number, the penal sum equal
  to the coverage limit, at least one limit. The policy-period and claims-made defaults do not apply.
- **Crosswalk:** `surety.crosswalk.yaml` maps all 660 legacy leaves (220 to a SPEC_21 path, 435 to an
  `additional_fields` label, 5 dropped). `tools/registry_migrate.py --lob surety`: the line's one registry YAML
  (`western_surety/surety.yaml`) declares no canonical_paths, so nothing needed rewriting.

**Registry:** unchanged. `western_surety/surety.yaml` identifies the carrier; each of the 4 bonds escalates to L3 at
the plausibility gate (coverage 0.20-0.40 below 0.80).

**Seeds:** none (scans; SPEC_21 5.3 would need corrected transcriptions, and L1 never reads a scan).

**Pipeline:** `config/canonical_schema/prompt/policy_check/surety.prompt.json` generated with
`tools/build_prompt_schemas.py`; the schema hash fixture holds its hash. The fine-tuning repository still holds the
legacy file (CR-SU-04).
