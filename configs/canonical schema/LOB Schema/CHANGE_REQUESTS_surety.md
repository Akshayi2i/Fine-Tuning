# Change requests - surety (to the common-model owner, the registry owner and the project owner)

Raised while building the surety SPEC_21 schema 4.0.0 (12 October 2026). Nothing here is applied to the common
model. Labels only; no document values.

**CR-SU-01. Coverage code prefix (project owner).**
`lob_schema_map.yaml` leaves the surety `lob_code` and `coverage_code_prefix` TBD. The schema uses `SU_` for its 3
codes; confirm the prefix.

**CR-SU-02. Scanned bonds (registry owner).**
All 4 documents of the sure folder (Western Surety Company license or permit bonds and their powers of attorney) are
scans under an OCR text layer whose words are partly misread. L1 never reads a scan: each escalates to L3 at the
plausibility gate, and no seed gold was drafted. A text-layer source document is needed before the line can have an
L1 seed.

**CR-SU-03. Bond facts in the common model (common-model owner).**
The `bond` block has no fields for the bond form and the authority that prescribes it, the principal's trade name and
county, the application or license number, or the attorney-in-fact; they are `additional_fields` labels. Decide
whether any of them belongs in the block (crime and fidelity bonds print some of them too).

**CR-SU-04. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `surety.json`. Decide with the exec_specialty family's other lines.
