# Change requests - wc (to the common-model owner and the training-side owner)

Raised while building the wc SPEC_21 schema 2.0.0 (10 October 2026). Nothing here is applied to the common
model: the schema composes `common_model.json` 1.1.0 unchanged, and each concept below stays in
`additional_fields` or in the LOB block `wc` until a request is accepted. Labels only; no document values.

**CR-WC-01. Training parity (fine-tuning repository owner).**
Mirrors CR-GL-01. The fine-tuning repository's `configs/canonical schema/LOB Schema/wc.json` is the pre-SPEC_21
self-contained file (`fideon.canonical.workers_compensation` 1.6.0) that 2.0.0 replaces, and its
`common.schemas._check_overlay_families` refuses a layout family that is only partly on the common model. gl is
already on the common model there (c8df4bb); wc, auto and umbrella are the rest of casualty_fleet. Decide: take
this wc overlay, its coverage-code file and its LOB block into the fine-tuning repository (with gl and the other
casualty_fleet lines), or allow wc alone. Until then wc is served by the base model (no casualty_fleet adapter).
The fine-tuning parity tests in this repository (`fideon/vlm/tests/test_*parity*.py`) cover the seven personal
lines and gl only and are unaffected.

**CR-WC-02. Industry codes on the named insured.**
The Hartford Information Page prints `NAICS` and `SIC` for the insured (every Hartford seed), AmTrust prints
`SIC Code`; the legacy 1.6.0 had `named_insured.sic_code` / `naics_code`, and GL and BOP declarations print the
same codes. Proposal: `named_insured.naics_code` and `named_insured.sic_code` (FieldValue). Meanwhile overflow
(`NAICS`, `SIC`, `SIC Code`).

**CR-WC-03. Employee count on a rating exposure.**
AmTrust's item 4 prints `# of Emps` per classification and Hartford's Schedule of Operations `NO. OF EMPL` per
location. Proposal: `RatingExposure.number_of_employees` (NumberValue). Meanwhile overflow (`Class <code> # of
Emps`, `NO. OF EMPL`).

**CR-WC-04. Promote the LOB block if a second line needs it.**
`wc.part_one_states`, `wc.other_states_insurance`, `wc.ncci_company_number` and `wc.bureau_file_number` are
local (SPEC_21 4.2.4). The employers liability part of a BOP or a commercial package's WC part would need the
same fields; if so they are promoted unchanged (SPEC_21 4.2.7 item 4 is respected: no new single-LOB block in the
common model meanwhile).

**CR-WC-05. Officers and owners included or excluded.**
The legacy 1.6.0 carried `workers_compensation.officers_and_owners[]`. NYSIF's sole-proprietor / LLC member
endorsements name the persons included, and the NY executive-officer exclusion endorsement names excluded
officers on other carriers' forms. Proposal (owner to decide whether common or local): a list of persons with
`included` (IncludedValue), title and ownership percentage. Meanwhile text_sections (and overflow where the
legacy crosswalk sends it).

**CR-WC-06. Code-parsed premium amounts with a credit suffix.**
NYSIF prints credits as `3,341.84CR`. The seed golds parse them as negative amounts; the common model's
MoneyValue description does not say how a printed credit suffix parses. Proposal: one sentence in
`MoneyValue` ("a printed credit marker - CR, parentheses, a leading minus - parses as a negative amount").

**CR-WC-07. `Deductible.basis` (CR-GL-16) - no second-line evidence from wc.**
The wc documents print no deductible program; CR-GL-16 stays held as far as wc is concerned.

## V3 recurring-wording review (23 October 2026, decisions 114/115)

Counts of documents / carriers from POLICIES-FINAL-V3 (captions printed in at least 5 documents of at least 3 carriers); no
text or values. Labels only.

**CR-WC-08. Concepts the V3 workers compensation documents print that need a common-model field (to the common-model owner).**
Each is printed by wc and at least one other line.
1. `policy.effective_time` (FieldValue) - the time of day the policy period starts and ends, as on the gl declarations (the 1.6.0
   legacy schema had `policy.effective_time`, `policy.time_zone`).
2. `carrier.company_type` (FieldValue) - `A Stock Company` (7 documents of 4 carriers; CGL 25 of 5).
3. `insurer_officers[]` (name, title) - `President`, `Secretary` signature lines (7 of 4 and 6 of 3; CGL 34 and 37 of 13).
4. Terrorism disclosure figures (federal share, trigger, annual cap): the `Policyholder Disclosure Notice` (17 documents of 7
   carriers) and the `Disclosure Endorsement` (23 of 3).
5. `named_insured.naics_code`, `named_insured.sic_code` (CR-WC-02): printed by Hartford and AmTrust; gl and BOP declarations print
   them (the V3 filter drops the acronyms, so no V3 count).
CR-WC-03 (employees per classification) stays open: Hartford and AmTrust print it, no V3 caption survives the filter.

**CR-WC-09. Rating modifier premium (to the common-model owner, single-line evidence).** Hartford prints the premium effect of
the experience and schedule rating beside the factor (14 and 3 seed documents). `RatingModifier` has factor and percent only.
Proposal: `RatingModifier.premium` (MoneyValue). Held for a second line.

**CR-WC-10. Resolved in wc 2.1.0 (no common-model change):** `wc.total_estimated_cost` and `wc.officers[]` (CR-WC-05: wc is the
only line among those read that prints the officers schedule at 3 or more carriers, so it is a LOB block; promoted unchanged if a
second line prints it, CR-WC-04 rule). A group number and a total standard premium are printed by 2 carriers only in V3 and stay
in `additional_fields`.

**CR-WC-ALIAS. `Limits Emp Liability` alias. Approved by the project owner and resolved in wc 2.2.0 (26 October 2026).**
