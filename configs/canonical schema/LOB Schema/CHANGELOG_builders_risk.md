# Changelog - builders_risk canonical schema

## 3.0.0 - 14 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (builders risk, L1 code and schema id `builders_risk`). It replaces the pre-SPEC_21
`builders_risk.json` (2.2.0, `_common`-shaped with a `builders_risk` block), which did not compose the common model.
A new major version because every path changes (SPEC_21 4.1).

**Built from** the documents of the source-data `bldrk` folder, read in place (5 documents, 3 carrier folders):

- US Assure (3): a builders risk RENEWAL application printed from the carrier's web portal (`BUILDERS RISK
  RENEWAL`), not declarations. One print is signed and carries the e-signature service's Document Completion
  Certificate (pages 2-3); the other two are the same unsigned one-page print (identical text). The form prints the
  policy number (with and without its prefix), the insured, the property location, the current policy dates, the
  project's expected completion date, an e-mail address and the signatures; no coverage, limit, deductible,
  premium or form schedule. These are the line's seeds.
- Vacant Express (1): a premium invoice (`Invoice for Insurance Premium`, a billing statement) on Diamond State
  Insurance Company paper, not declarations. No seed (CR-BR-04).
- Zurich (1): a builders risk policy with its New York supplemental declarations (forms U-GU / HBIS); its value
  pages are an OCR text layer over a scan. No seed; its supplemental-declarations captions gave coverage aliases
  (CR-BR-05).

Changes:

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `locations`,
  `buildings` and `countersignature` blocks. No LOB block: the project site is a `locations[]` element, the
  project's building facts a `buildings[]` element, and the project facts with no core home are `additional_fields`
  labels (CR-BR-03).
- **Coverage codes:** `builders_risk.coverage_codes.yaml`, 38 codes: 35 with a provisional `BR_` prefix (CR-BR-01)
  from the legacy limits, deductibles and time-element blocks (building under construction, soft costs, delay in
  completion, rental income, loan interest, temporary storage, transit, debris removal, pollutant clean-up,
  scaffolding, trees and shrubs, theft of materials, expediting expense, model home contents ...) and the Zurich
  supplemental declarations' additional coverages and endorsements (collapse, re-erection of scaffolding, fire
  department service charge, valuable papers, the three ordinance or law parts, preservation of property, rewards,
  claim preparation, contract penalties, business income, extra expense, civil authority, fences walls and signs,
  green building ...), and the common model's shared `X_ORDINANCE_LAW`, `X_WATER_BACKUP` and `X_TERRORISM`.
- **Mandatory fields:** `carrier.name`, `named_insured.primary_name`, `policy.policy_number`,
  `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount` (the default list).
- **Audit rules:** the common model's defaults 1-4 (the claims-made rule left out: builders risk is written on an
  occurrence basis).
- **Crosswalk:** `builders_risk.crosswalk.yaml` maps all 654 legacy leaves (283 to a SPEC_21 path, 361 to an
  `additional_fields` label, 10 dropped) plus the 4 property-location paths the registry YAML declares (658
  entries). Dropped: the e-signature certificate's Document History event log (an audit trail, not a document
  value), its participant row numbers, the application's printed certification wording, and the reference marks and
  sub-item captions the shared rules drop. `tools/registry_migrate.py --lob builders_risk --check`: 0 changes in
  both registry YAMLs.

**Registry:** `us_assure/builders_risk.yaml` rewritten to SPEC_21 paths (30 declared paths). The carrier's name is
only in the drawn logo; the entry declares it among its fingerprints, so the portal host the reader takes for the
insurer is completed to it (no OCR). All 3 US Assure prints read at L1 with 100% judged accuracy and 100%
completeness against their seed golds. `zurich/builders_risk.yaml` is unchanged; the Zurich scan and the Vacant
Express invoice escalate to L3 at the plausibility gate (no coverage limit or premium read).

**Seeds:** `seeds/builders_risk/US Assure/` (3 DRAFT golds, pending reviewer sign-off). The renewal application
prints no coverage: the varmaps waive `coverages[].limits[].amount` (`mandatory_waivers`), and the golds' carrier
name is an image value (the logo).

**Pipeline:** the prompt schema `config/canonical_schema/prompt/policy_check/builders_risk.prompt.json`, the schema
hash fixture, `common/lob_schema_map.yaml` and `common/CHANGELOG_common.md` are to be updated by the program
coordinator (CR-BR-06). The fine-tuning repository still holds the legacy file (CR-BR-07).
