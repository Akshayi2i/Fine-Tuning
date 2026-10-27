# Change requests - cyber (to the common-model owner, the registry owner and the project owner)

Raised while building the cyber SPEC_21 schema 2.0.0 (11 October 2026). Nothing here is applied to the common
model. Labels only; no document values.

**CR-CY-01. Coverage code prefix (project owner).**
`lob_schema_map.yaml` leaves the cyber `lob_code` and `coverage_code_prefix` TBD. The schema uses `CY_` for its 37
codes; confirm the prefix (or assign another and the codes are renamed in one pass).

**CR-CY-02. CRC's document (registry owner).**
The CRC document in the cyber folder is a print from an agency management system (account summary), not an insurer's
declarations page. No carrier entry and no gold; it stays L3.

**CR-CY-03. Travelers' lawyers professional liability policy (registry owner).**
One document of the Travelers cyber folder is a New York lawyers professional liability policy. It is read under
`professional_eo` (its gold is in that line's seeds) and is not a cyber seed.

**CR-CY-04. Quota-share policies without a single insurer (common-model owner).**
The Coalition Active declarations name five insurers, each with a participation percentage; `carrier.name` is
mandatory and holds one name. The gold records each insurer and its participation as `additional_fields` and drafts
no `carrier.name` (C3 reports it). Decide whether the common model needs a participating-insurers list.

**CR-CY-05. Text-layer artefacts (synth_core owner).**
Two Coalition SP documents print a bold value twice, run together, in the text layer (the part title and one
agreement name); C2 reports the value as not printed. One At-Bay producer town is split at a ligature and is left out
of the gold. Decide whether C2 should read these runs.

**CR-CY-06. Duplicate field map (registry owner).**
`swiss_re/cyber.yaml` repeats the Coalition SP field map because the router names the issuing insurer. Decide whether
a program-level entry (one map, several issuing insurers) should replace the copy.

**CR-CY-07. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `cyber.json`. Decide with the exec_specialty family's other lines.

**CR-CY-08. System-failure and sub-limits (common-model owner).**
Direct and contingent system-failure limits are coded as `aggregate` and `other` on one coverage; waiting periods,
periods of indemnity and endorsement limits go to `additional_fields`. Decide whether `Limit` needs a waiting-period
or indemnity-period field shared with business interruption in other lines.
