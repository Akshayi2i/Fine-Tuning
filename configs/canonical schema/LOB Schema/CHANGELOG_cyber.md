# Changelog - cyber canonical schema

## 2.0.0 - 11 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (cyber and privacy liability, L1 code and schema id `cyber`). It replaces the
pre-SPEC_21 `cyber.json` (1.5.0, `_common`-shaped with a `cyber` block), which did not compose the common model. A
new major version because every path changes (SPEC_21 4.1).

**Built from** the single-LOB documents of the source-data `cyber` folder, read in place: At-Bay's cyber insurance
policy declarations in three editions (filed under AtBay and ARC Excess & Surplus), Coalition's cyber policy
declarations in three forms (its own paper, the fronting insurers' SP forms, the Coalition Active quota-share form)
and the Corvus Smart Cyber declarations on Travelers Excess and Surplus Lines paper. CRC's document is an
agency-system print, not a declarations page (CR-CY-02); Travelers' other document is a lawyers professional
liability policy, read under `professional_eo` (CR-CY-03).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `claims_made_terms` and
  `insuring_agreements` blocks. No LOB block.
- **Coverage codes:** `cyber.coverage_codes.yaml`, 37 codes with a provisional `CY_` prefix (CR-CY-01): the policy
  aggregate, the third-party agreements (privacy, network security, regulatory, PCI, media, funds transfer),
  first-party response and recovery, extortion, business interruption and its system-failure, voluntary-shutdown and
  contingent variants, digital asset restoration, bricking, reputational harm, court attendance, criminal reward,
  the fraud agreements (social engineering, computer, service and invoice fraud, impersonation repair), pre-claim
  assistance, and the shared `X_TERRORISM`.
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults, the claims-made rule kept.
- **Crosswalk:** `cyber.crosswalk.yaml` maps all 400 legacy leaves (259 to a SPEC_21 path, 134 to an
  `additional_fields` label, 7 dropped). `tools/registry_migrate.py --lob cyber`: no declared path needed rewriting
  (every cyber entry is written in SPEC_21 paths).

**Registry** (`config/document_registry/policy/`):
- `atbay/cyber.yaml` (new): the At-Bay AB-CYB-004 declarations (09/2023, 12/2024 and 12/2025 editions): insurer,
  producer, insured, period, premium and fees, the insuring agreements with their limits and retentions, the system
  failure and reputational harm enhancements, claims-made dates and the forms schedule; 3 of 3 seeds.
- `coalition/cyber.yaml` (new): Coalition's CYUSP-00DC-1022 declarations, the SP 17 226 / SP 14 797 declarations and
  the Coalition Active declarations (five insurers in quota share); 4 of 4 seeds routed to Coalition.
- `swiss_re/cyber.yaml` (new): the same field map for the SP declarations issued on Swiss Re Corporate Solutions
  paper (the router names the issuing insurer); 1 seed (CR-CY-06).
- `travelers/cyber.yaml`: was an unmeasured placeholder (sample_count 0); now a sectioned map of the Corvus Smart
  Cyber declarations (ITEM 1 to ITEM 9), the line signal of their title and the issuing insurer's name as a
  fingerprint; 1 of 1 seed.
- Every new entry carries the shared `policy_fields` block.

**Seeds:** 9 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/cyber/`, not in Git), pending reviewer sign-off.

**Pipeline:** `config/canonical_schema/prompt/policy_check/cyber.prompt.json` generated with
`tools/build_prompt_schemas.py`; the schema hash fixture holds its hash. The fine-tuning repository still holds the
legacy file (CR-CY-07).
