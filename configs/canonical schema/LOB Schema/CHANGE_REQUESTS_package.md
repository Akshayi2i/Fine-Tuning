# Change requests — package policies (L1, 12 October 2026)

No document values are recorded here.

Raised while making L1 read package policies per SPEC_02 §2a/§4, SPEC_21 §4.2.3 and owner decision D4
(12 October 2026: per-part escalation stays). Each needs an owner decision or a spec amendment; the code
follows the choice stated here until then.

**Decision, 12 October 2026 (relayed by Sagar Katekhaye):** CR-PKG-01 to CR-PKG-09 are approved as stated below.
CR-PKG-10 onwards are still open.

**CR-PKG-01. SPEC_02 §2a all-or-nothing vs per-part escalation (spec owner).**
SPEC_02 §2a sends the whole package to L3 when any detected line lacks a registry entry (Path B). D4 keeps
per-part escalation: parts L1 reads stay L1; an unread part is escalated alone
(`l3_service.read_escalated_parts`). Code: a package with NO part read at L1 goes to L3 whole with
`routing.reason` starting `package_policy` and `meta.is_package` set. SPEC_02 §2a (and its §9 Path B criteria)
needs an amendment recording D4.

**CR-PKG-02. `coverage_part_anchor` options beyond SPEC_02 §4 (spec owner).**
`fideon/l1/package_parts.py` reads `patterns` and `max_section_pages` as §4 writes them (regex, not the literal
substring the ISO reader uses) and three more keys: `end_patterns` (the section ends before the first later page
matching one), `search_pages` (how far the anchor is looked for, default 40) and `read: false` (detection only:
the part is located and escalated, the entry's field map does not read it — for an entry measured on single-line
documents only). SPEC_02 §4 should document them.

**CR-PKG-03. `part` on blocks that have none (common-model owner).**
SPEC_21 §4.2.3 says every coverage, unit, form and premium item carries its `part`. The common model has no
`part` on `Deductible`, `InterestedParty`, `Charge` (taxes, fees, surcharges, discounts), `AdditionalField` or
`RatingModifier`, so a part's deductible, mortgagee or fee cannot be attributed. Until then L1 and the golds
leave them untagged (shared by every part) and `additional_fields[].section_hint` names the section.

**CR-PKG-04. Dryden Mutual "Commercial Fire & Liability" and "Contractors Package" policies (project owner).**
Read as packages with parts `property` (Property Coverage Section), `gl` (Liability Coverage Section) and
`inland_marine` (Inland Marine Coverage Section), each section with its own premium on the master page. Not
`boppr` (the corpus folder name) and not `commercial_fire` (the ruling names the Leatherstocking product only,
CR-CFIRE-04).

**CR-PKG-05. North Country "Commercial Special Multi-Peril" (project owner).**
One coverage schedule carries both parts: Cov A/B rows -> `property`, Cov L/M/N rows -> `gl` (SMP sections I
and II). Not `commercial_package` (no schema, outside the program's 16 lines) and not `boppr`/`bopgl` (the
corpus folder name of two of the four documents).

**CR-PKG-06. Businessowners policies are packages: `boppr` + `bopgl` (project owner).**
SPEC_21 §4.2.3 lists BOP among packages, one part per line. Hartford Spectrum and Midstate businessowners
policies are read with a `boppr` part and a `bopgl` part. The finished Leatherstocking businessowners gold reads
the same kind of policy as ONE `boppr` part with liability codes inside it; the owner should rule which is the
convention and the other should be migrated.

**CR-PKG-07. Package golds (seed reviewer).**
One gold per package document under `seeds/package/<Carrier>/`, the whole package as one canonical JSON
(`lob_parts[]`, `part` on every element that has the property), validated per part (each part's subset against its
LOB schema) and scored per part. DRAFT, pending Shachindra Thakar.

**CR-PKG-08. Parts outside the program lines (project owner).**
Hartford Private Choice Premier (fiduciary, crime, kidnap and ransom; D&O and EPL printed NOT COVERED) and the
VFIS portfolio's crime, portable equipment and management liability parts have no program line or no L1 code
(fiduciary, kidnap and ransom). They are not read at L1.

**CR-PKG-09. `meta.line_of_business` of a package (spec owner).**
SPEC_00 has one `line_of_business`. A package reports the first part read at L1 there, every part in
`lines_detected`, and the L1 codes in `detected_lobs` (`confirmed_lobs` stays L3's).

**CR-PKG-10. Gemini package gold (seed reviewer; the gold is signed off).**
`seeds/gl/Gemini Insurance Company/policy_01.gold.json` pairs two of the common declarations' tax and fee
amounts with the wrong captions (it followed text-layer order; the page prints each amount on the row of the
caption after the one the gold names, and the caption the gold uses for the first has no amount). It also tags one
endorsement that modifies the Professional Liability Coverage Part with the general liability part, and codes the
professional liability coverage with a gl code instead of a code from `professional_eo.coverage_codes.yaml`. Until
it is corrected and re-signed, Gemini's entries are detection-only and the package goes to L3 whole.

**CR-PKG-11. Whole-policy audit rules on a part (schema owner).**
A rule such as "the total premium equals the sum of the printed premiums" holds for the package, not for one part:
a businessowners part prints only some of the premiums that make up the package total. Audit rules of a part's LOB
schema that sum to the policy total should apply to the whole package document, not to the part.

**CR-PKG-12. `epli.json` does not declare `fideon:provenance` (schema owner).**
Every SPEC_21 gold carries `fideon:provenance`; the epli schema's closed envelope rejects it, so the per-part check of
the two Hartford Spectrum packets with an Employment Practices Liability part reports one error on that part (its
own values validate). The epli part is detection-only at L1 and is escalated on its own.

**CR-PKG-13. Hartford Spectrum entries were built with the golds as locators (project owner).**
`hartford/boppr.yaml` and `hartford/bopgl.yaml` were generated by a scratch helper that found each gold value on its
printed page and wrote a pattern for that row: coverage names and captions are literal, every amount and figure is
generalised, and no insured name, address, amount or policy number is in either file (checked against all 25 package
golds). The golds themselves were drafted from the PDF text layer only. Because the field maps were tuned against
the same golds they are scored on, the Hartford result is not an independent measure; the reviewer's sign-off of the
golds, or a held-out Hartford packet, is what makes it one. The program rule (golds independent of L1, L1 config not
generated from gold values) should say whether locating rows by gold values is allowed.

**CR-PKG-14. A gold edited by an entry helper (seed reviewer).**
The Dryden entry helper changed one structural tag in the 10 Dryden package golds (`locations[loc_1]` gained
`part: part_1`, the property part, where the location's rating facts are printed), recorded under Corrections in
each review. No value was changed and none came from L1 output; the reviewer should confirm the tag.

**CR-PKG-15. Hartford Spectrum Policy Change packets go to L3 (project owner).**
Four Policy Change packets in the Hartford businessowners folder were read at L1 by the
removed combined `hartford/business_owners.yaml`. With per-part entries they fail the SPEC_02 §7 coverage gate (a
change prints only the changed schedule) and go to L3. A Policy Change reading (an endorsement-type read per part)
needs its own entries and golds.

**CR-PKG-16. An ISO-form package that no part entry reads (spec owner).**
D4 (CR-PKG-01) sends a package with no part read at L1 to L3 whole. That rule is applied to packages found by a
carrier's own `coverage_part_anchor` entries. A package found only by the ISO master-declarations reader (a
`... Coverage Part` premium table and a TOTAL, no carrier anchor) whose parts no per-part entry reads keeps the
reading it had before per-part packages: the SPEC_20 master-declarations path at L1, `policy_package` beside
`policy_check` (`pipeline_api/tests/integration/test_dispatcher.py`). The owner should confirm the exception or ask
for such a package to go to L3 whole as well.
