# Change requests - inland_marine (to the common-model owner, the registry owner and the project owner)

Raised while building the inland_marine SPEC_21 schema 2.0.0 (10 October 2026). Nothing here is applied to the
common model. Labels only; no document values.

**CR-INMRC-01. Batch code and coverage-code prefix (project owner).**
`lob_schema_map.yaml` leaves `lob_code` and `coverage_code_prefix` TBD for inland_marine. The codes use `IM_` (the
routing key) until the owner sets them; a different prefix is a rename in `inland_marine.coverage_codes.yaml`, the
three entries' declared paths and the golds.

**CR-INMRC-02. The RT Specialty agency-system prints (registry owner).**
1. The two RT Specialty documents are HawkSoft AL3 Viewer prints of a scheduled equipment policy (the print states it
   cannot be used to determine coverage). The carrier's name is printed cut short at the cell's edge.
   `american_reliable/inland_marine.yaml` lists both as its samples, but its fingerprint (another carrier's name) is
   printed on neither, so the entry never matches and both documents escalate (L3), as before.
2. On one of the two the named insured is printed twice over the same place (two text runs overlap), so a reading by
   position cannot give the name the text layer holds. No entry was built: one carrier entry would read both prints.
   Decide whether agency-system prints are read at L1 (one entry per print system, not per carrier) and correct the
   carrier id of the existing entry; the golds are drafted for that.

**CR-INMRC-03. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `inland_marine.json`. Decide with CR-BOPPR-02 for the property_pkg family.

**CR-INMRC-04. The unmeasured zurich entry (registry owner).**
`zurich/inland_marine.yaml` has no measured sample (`sample_count: 0`, its one corpus document is outside the
inmrc folder), so `lookup` declines it; it is unchanged.
