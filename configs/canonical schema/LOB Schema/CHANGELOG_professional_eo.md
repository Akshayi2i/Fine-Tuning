# Changelog - professional_eo canonical schema

## 2.0.0, extended - 12 October 2026 (the eo folder; still a draft)

The schema's paths, codes and rules are unchanged; its description, `fideon:built_from` and the
`PL_PROFESSIONAL` aliases now name the `eo` folder's errors and omissions declarations, read in place: Mount Vernon
Fire Insurance Company's (USLI) specified professions renewals (11: renewal certificate, forms extension, coverage
part declarations; one a certificate alone), Swiss Re Corporate Solutions' insurance agencies professional
liability declarations (1) and Navigators' accountants professional liability declarations (1, through ARC Excess &
Surplus). Not seeds (CR-PL-05): an agency's proposal of insurance and a scanned lawyers professional liability
application.

**Registry** (`config/document_registry/policy/`):
- `mount_vernon/professional_eo.yaml`: a sectioned map of the renewal certificate (UPC), its forms extension (EOD)
  and the coverage part declarations (SP 150), the line signals of the coverage part and the shared
  `policy_fields` block; the proposal it was also credited with prints no declarations and escalates.
- `usli/professional_eo.yaml`: the same sectioned map (CR-PL-06): the router names USLI for the renewals whose front
  bundle prints the USLI companies' legal names. Its provenance note now says what it was verified on.
- `swiss_re/professional_eo.yaml`: a sectioned map of the SP 4 604 O NY declarations, the named insured
  endorsement and the shared `policy_fields` block.
- `navigators/professional_eo.yaml`: a sectioned map of the NAV APL NIC DEC NY declarations and the shared
  `policy_fields` block.

**Seeds:** 13 more DRAFT golds (30 in all), pending reviewer sign-off. Through `POST /extract`, every seed of the
eo folder reads at 100% judged accuracy and 100% completeness.

## 2.0.0 - 11 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (professional liability and errors and omissions, L1 code and schema id
`professional_eo`). It replaces the pre-SPEC_21 `professional_eo.json` (1.5.0, `_common`-shaped with a
`professional_liability` block), which did not compose the common model. A new major version because every path
changes (SPEC_21 4.1).

**Built from** the single-LOB professional liability declarations of the source-data `pl` folder, read in place,
all with a native text layer on their declarations pages: Benchmark's and Coterie's (Providence Washington)
professional liability errors and omissions declarations (1 each), CNA's Epack Extra media liability declarations
(4), Hartford's Premier Choice Tax Preparer (3) and Design Professionals (3) declarations, Scottsdale's (Nationwide)
Business and Management Indemnity declarations scheduling only the miscellaneous professional services section (3)
and Travelers' New York lawyers professional liability declarations (1, and an identical copy filed under
`cyber`). The `eo` folder's errors and omissions
declarations extend it. Not seeds (CR-PL-02): two management-liability packages filed under `pl` (ARC Excess &
Surplus's financial institutions package, Hartford's Private Choice Premier, 3 documents), and endorsements,
cancellation notices and a cover letter (10 documents).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `claims_made_terms`
  block (the lob_schema_map entry). No LOB block and no local `$defs`.
- **Coverage codes:** `professional_eo.coverage_codes.yaml`, 22 codes: 20 prefixed `PL_` (professional liability,
  the financial-institutions sections, personal injury, media, pollution, rectification, disciplinary and
  regulatory proceedings, supplementary payments, subpoena, pre-claim, crisis management, reputation, non-profit
  D&O, client fee assistance, publishing, pro bono, retired reporting) and the shared `X_CYBER` and `X_TERRORISM`.
  The prefix is provisional (CR-PL-01).
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults, the claims-made rule kept.
- **Crosswalk:** `professional_eo.crosswalk.yaml` maps all 362 legacy leaves (228 to a SPEC_21 path, 127 to an
  `additional_fields` label, 7 dropped). `tools/registry_migrate.py --lob professional_eo`: no declared path needed
  rewriting (18 entries declared none).

**Registry** (`config/document_registry/policy/`):
- `coterie/professional_eo.yaml` (new): both Coterie templates (CTPL CW PL on Benchmark paper, CTPL NY PL on
  Providence Washington paper); 2 of 2 seeds read at L1.
- `hartford/professional_eo.yaml`: a sectioned map of the Premier Choice declarations (Tax Preparer and Design
  Professionals), their line signals, the letterhead fingerprint and a content anchor at ITEM 1; 6 of 6 seeds.
- `continental_casualty/professional_eo.yaml`: a sectioned map of CNA's Epack Extra declarations; 4 of 4 seeds.
- `scottsdale/professional_eo.yaml` (new): the Business and Management Indemnity declarations, the scanned copy on
  page 1 passed over; 3 of 3 seeds.
- `travelers/professional_eo.yaml`: a sectioned map of the New York lawyers professional liability declarations
  and its line signals; 1 of 1 seed and its copy filed under cyber.
- The commercial-package field maps those entries already carried (the synthesized fixtures) are unchanged; the
  new entries carry the shared `policy_fields` block.

**Seeds:** 17 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/professional_eo/`, not in Git), pending
reviewer sign-off.

**Pipeline:** `config/canonical_schema/prompt/policy_check/professional_eo.prompt.json` generated with
`tools/build_prompt_schemas.py`; the schema hash fixture holds its hash. exec_specialty has no adapter yet. The
fine-tuning repository still holds the legacy file (CR-PL-03).
