# Change requests - professional_eo (to the common-model owner, the registry owner and the project owner)

Raised while building the professional_eo SPEC_21 schema 2.0.0 (10-11 October 2026). Nothing here is applied to
the common model. Labels only; no document values.

**CR-PL-01. Batch code and coverage-code prefix (project owner).**
`lob_schema_map.yaml` leaves `lob_code` and `coverage_code_prefix` TBD for professional_eo, and the corpus files the
line under two folders (`pl`, `eo`). The codes use `PL_` until the owner sets them; a different prefix is a rename
in `professional_eo.coverage_codes.yaml`, the entries' declared paths and the golds.

**CR-PL-02. Management-liability packages and non-declarations filed under pl (project owner).**
1. ARC Excess & Surplus's financial institutions management and professional liability policy (management
   liability, EPL, fiduciary and three professional liability sections on one declarations) and Hartford's Private
   Choice Premier declarations (D&O, EPL, fiduciary, crime and kidnap coverage parts; 3 documents) are packages:
   out of the single-LOB scope, not drafted, unchanged (L3). `management_liability` has no schema
   (`schema_lob: TBD`); decide whether these are read under it or under directors_officers.
2. Hartford's policy-change endorsements (3), Nationwide's cancellation notices (3, one of them a premium-finance
   company's notice with an unreadable text layer) and name/address endorsements (3, a font without a Unicode map),
   and CNA's producer cover letter (1) are not declarations; they stay L3.

**CR-PL-03. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `professional_eo.json`. Decide with the exec_specialty family's other
lines (directors_officers, cyber, epli).

**CR-PL-04. Shared codes (common-model owner).**
The financial-institutions and Hartford packages print EPL and fiduciary sections; `X_EPL` exists, a shared
fiduciary code does not. Settle with CR-PL-02 and the directors_officers schema.

**CR-PL-05. Not seeds in the eo folder (registry owner).**
RPS's proposal of insurance (a quote filed under the Mount Vernon entry's samples) prints no declarations and
escalates; Berkley's document is a lawyers professional liability application, scanned under an OCR text layer, and
escalates. Neither has a gold.

**CR-PL-06. Duplicate field map (registry owner).**
`usli/professional_eo.yaml` repeats the `mount_vernon/professional_eo.yaml` map because the router names USLI when a
Mount Vernon renewal's front bundle prints the USLI companies' legal names (one of them United States Liability
Insurance Company). Decide whether a group-level entry should replace the copy, or whether the router should prefer
the declarations page's letterhead.
