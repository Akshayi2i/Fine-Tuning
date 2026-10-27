# Change requests - directors_officers (to the common-model owner, the registry owner and the project owner)

Raised while building the directors_officers SPEC_21 schema 2.0.0 (12 October 2026). Nothing here is applied to the
common model. Labels only; no document values.

**CR-DO-01. Coverage code prefix (project owner).**
`lob_schema_map.yaml` leaves the directors_officers `lob_code` and `coverage_code_prefix` TBD. The schema uses `DO_`
for its 10 codes; confirm the prefix (or assign another and the codes are renamed in one pass).

**CR-DO-02. RPS's document (registry owner).**
The RPS document in the do folder is a wholesale broker's proposal of insurance (quote, premium summary and required
forms list) for a non-profit D&O renewal, not an insurer's declarations page. No carrier entry and no gold; it
escalates to L3 (the plausibility gate).

**CR-DO-03. Coverage parts not granted (reviewer).**
Four of the forms print every coverage part they offer (employment practices, fiduciary, crime, public company,
workplace violence, internet liability) with a Yes/No grant or a blank limit, and grant only the D&O part. The golds
record the granted part only; a part printed as not granted, `N/A` or blank is not drafted. Decide whether a not-granted
part should be a coverage with `included: not_purchased`.

**CR-DO-04. Retentions by claim type (common-model owner).**
`Deductible` has no caption or claim-type field. A retention printed per insuring clause is that clause's coverage
deductible; a retention printed by claim type (`Each Securities Claim`, `All other Claims`) or as a split cell
(per insured person / aggregate) goes to `additional_fields` under its caption. Decide whether `Deductible` needs a
`description` like `Limit`.

**CR-DO-05. Named insured drawn as overlapping glyphs (synth_core owner).**
On Wesco's declarations the named insured is drawn as a run of glyphs set on top of each other (unreadable on the
page, a misspelt run in the text layer). The gold takes the name the endorsement schedules print legibly. A
reviewer confirms.

**CR-DO-06. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `directors_officers.json`. Decide with the exec_specialty family's other
lines.

**CR-DO-07. Organizations of a personal director's liability policy (common-model owner).**
Federal's declarations list the organizations the insured person serves (Item 4) - 49 on the seed. They are not
named insureds; the gold records each as an `additional_fields` entry labelled `Organization`. Decide whether the
common model needs an outside-entity list.

**CR-DO-08. Stale RT Specialty provenance (registry owner).**
`american_reliable/directors_officers.yaml` names the three RT Specialty documents as its samples. They are United
States Liability's paper and route to `usli/directors_officers.yaml`; the american_reliable entry has no field map
and is left unchanged. Decide whether its samples should be removed.


**CR-DO-09. Investment management liability read as D&O (project owner, reviewer).**
Raised with 2.1.0 (14 October 2026). ARC Excess & Surplus' `epli` sample is CNA's Investment Management Liability
Solutions policy (CNA88758XX, Columbia Casualty Company): investment adviser professional liability, investment
adviser management liability and fund management and professional liability coverage parts under one policy
aggregate. There is no investment management schema; it is read with directors_officers (management liability) and
its parts get three new `DO_` codes. Decide (a) whether the professional liability parts belong in this line or in
professional_eo (a combined management/professional policy), and (b) whether a sublimit an endorsement adds to the
declarations' limit item (Vital Executive Expenses) should be a coverage rather than an `additional_fields` entry,
as the draft gold records it. The tracker's `EPLI` label for the row is a filing name, not the line.
