# Change requests - flood (to the common-model owner, the registry owner and the project owner)

Raised while building the flood SPEC_21 schema 2.0.0 (14 October 2026). Nothing here is applied to the common
model. Labels only; no document values.

**CR-FL-01. Batch code and coverage-code prefix (project owner).**
`lob_schema_map.yaml` leaves `lob_code` and `coverage_code_prefix` TBD for flood. The codes use `FL_` until the owner
sets them; a different prefix is a rename in `flood.coverage_codes.yaml`, the two entries' declared paths and the
golds.

**CR-FL-02. Documents that are not readable declarations (registry owner).**
1. Sterling's renewal quote (`Quote #...`, "does not imply that coverage is bound") is a proposal, not declarations:
   the entry's title does not match it and it escalates (L3) at the coverage gate. No seed gold.
2. IAT Insurance Group's declarations is a scan under an OCR text layer: no registry entry, it escalates (L3), no
   seed gold. A text-layer copy is needed before the carrier can have an L1 seed.

**CR-FL-03. Flood rating facts in the common model (common-model owner).**
The `locations` and `buildings` blocks have no fields for the flood zone, the base flood and lowest floor elevations
and their difference, the first floor height and how it was determined, the enclosure, the elevator, the NFIP policy
number, the replacement cost value, the prior NFIP claims count or the primary-residence flag. They are
`additional_fields` labels. Decide which belong in `Location` / `Building` (homeowners and dwelling fire print some
of them too).

**CR-FL-04. Lloyd's coverholder declarations (project owner).**
Sterling Surplus Underwriters is the coverholder and surplus-lines broker; the insurer is Lloyd's. The 2024-2026
declarations print the insurer after the caption `Carrier`; the 2021-2023 layouts print no insurer name (only the
Lloyd's form number and the service-of-suit designee), so their golds hold no `carrier.name` (a C3 note on those
seeds). The coverholder's claims box is `carrier.claims_address` / `claims_email`; its name and licences are
`additional_fields`. Confirm, or add a coverholder / broker field to the common model.

**CR-FL-05. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `flood.json`. Decide with CR-BOPPR-02 for the property_pkg family.
