# Change requests - boppr (to the common-model owner, the training-side owner and the engine owner)

Raised while building the boppr SPEC_21 schema 2.0.0 (10 October 2026). Nothing here is applied to the common
model: the schema composes `common_model.json` 1.1.0 unchanged, and each concept below stays in a boppr coverage
code or in `additional_fields` as named until a request is accepted. Labels only; no document values.

**CR-BOPPR-01. Shared coverage codes for business liability.**
A businessowners declarations prints its liability coverages beside the property ones: bodily injury and property
damage (Coverage L), products-completed operations (Coverage N), fire legal liability / damage to premises rented
(Coverage O) and personal and advertising injury (Coverage P); gl prints the same concepts as `GL_BI_PD`,
`GL_PRODUCTS_COMPLETED_OPS`, `GL_DAMAGE_TO_PREMISES_RENTED` and `GL_PERSONAL_ADVERTISING_INJURY`, and bopgl will.
By the common model's inclusion rule (printed by two or more LOBs) they belong in `fideon:shared_coverage_codes`.
Until decided this line uses `BOPPR_LIABILITY`, `BOPPR_PRODUCTS_COMPLETED_OPS`, `BOPPR_FIRE_LEGAL_LIABILITY` and
`BOPPR_PERSONAL_INJURY`; once shared codes are released, boppr renames them in a major version with crosswalk
entries (SPEC_21 4.1), as gl did for CR-GL-03. To be raised together with the bopgl owner.

*Status 23 October 2026 (DECISIONS rows 114, 115):* released in common model 1.3.0 as shared codes (`X_BI_PD_LIABILITY`,
`X_PRODUCTS_COMPLETED_OPS`, `X_DAMAGE_TO_PREMISES_RENTED`, `X_PERSONAL_ADVERTISING_INJURY`). boppr 2.1.0 keeps `BOPPR_LIABILITY` and the
others (a rename is a major version, 3.0.0), and adds the counterpart-code table shared with bopgl (see CHANGE_REQUESTS_bopgl.md CR-BOPGL-01).
Awaiting Sagar's decision on moving bopgl, boppr and gl together.

**CR-BOPPR-02. Training parity (fine-tuning repository owner).**
The fine-tuning repository (c8df4bb) holds the legacy `configs/canonical schema/LOB Schema/boppr.json` (policy_core
re-base 1.5.0, the same content as the file this schema replaces). property_pkg has no adapter and no parity test
covers boppr. Decide, as for CR-GL-01: migrate the property_pkg family (property, boppr, flood, inland_marine...) to
the common model together, taking this overlay and `boppr.coverage_codes.yaml`, or allow boppr alone. Until then
boppr is served by the base model.

**CR-BOPPR-03. `lob_schema_map.yaml` boppr entry.**
The map entry was filled for this schema (`blocks_used: [locations, buildings]`, `source_pdfs: {folder: boppr,
carriers: 3, files: 5}`); `lob_code`, `routing_key`, `coverage_code_prefix` and `seeds` stay as the owner sets them
(the seeds are DRAFT, not signed off): the batch code is TBD (no `SYN-PDF-PC-<CODE>` batch ID can be issued for boppr until it is assigned),
and the schema files use the prefix `BOPPR_` meanwhile. Owner to confirm or assign.

**CR-BOPPR-04. synth_core: comma-grouped split limits and repeated coverage codes (engine owner).**
1. C2's whole-token test (`pdf_text.num_bounds`) refuses `1,000,000` inside the printed split limit
   `1,000,000/2,000,000` (the `/` guard applies to a trailing three-digit group), while SPEC_21 4.2.2 stores a
   split limit as one `limits[]` entry per figure. CR-PU-8 fixed the case without thousands separators. Seen on the
   Reinsure seed (flagged values).
2. C14 (`paths.keyed`) pairs array elements by a key only when the key is unique in the array; coverages of a
   policy with two buildings repeat their codes (one per building, told apart by `applies_to`), so they are paired
   by position. `tools/measure_policy_extraction` pairs them by code and units. Request: let `keyed` combine
   `coverage_code` with `applies_to`. Meanwhile the two-building Leatherstocking gold lists its coverages in the
   order L1 emits them.

**CR-BOPPR-05. L1: one reading into a list of readings; comma streets (registry owner).**
`buildings[].protective_devices` is a list of FieldValues; a single reading declared onto it is spilled by
`fideon.l1.spec21` under a made-up label, so the Leatherstocking entry sends it to `additional_fields` under its
printed caption (`Protective Safeguard Credit (SF-53)`). And `spec21._address` refuses a street that prints a comma
(`<number> <street>, 3rd Floor`), so a mortgagee address read whole stays one overflow value. Both cost
completeness only, never a wrong value.

**CR-BOPPR-06. One line per Leatherstocking product (project owner).**
Leatherstocking prints two businessowners-style products on the one LCIC(4/11) declarations form, and the corpus
files each under several lines:
- `DECLARATION, Commercial Fire and Liability` (Coverage A building and B business property by peril, Coverage L and M
  liability, OLT classifications): 1 document under boppr (a seed), 5 under cgl_prop (CGL+property package), 1 under
  prop. Same form set and layout; nothing the page prints tells them apart. The boppr entry
  (`policy/leatherstocking/boppr.yaml`) does not claim this banner, so all seven keep escalating as before; its
  reading rules were measured on the boppr seed and are kept in the entry.
- `DECLARATION, Businessowner's Policy`: 1 document under boppr (a seed, read at L1), 2 under bopgl and 4 under prop.
  The bopgl and prop ones print supplemental schedules the boppr map was not measured on (an SF-10P property
  supplement; a products-completed operations credit and two further liability rows); the entry's opt-in
  `require_coded_coverages` makes it step aside on them, so they escalate as before.
Decide the line of each product (boppr, bopgl, business_owners as the whole policy, property, or a package), and
the corpus labels follow; the entry is then widened or split accordingly.
*Ruled 10 October 2026 (project owner, for the first product):* `DECLARATION, Commercial Fire and Liability` is
the Commercial Fire line (`commercial_fire`, a line of its own; not boppr, property or a package). The seven
documents are read by `policy/leatherstocking/commercial_fire.yaml` against `commercial_fire.json`; their
reading rules moved there from the boppr entry, and the boppr seed `leatherstocking_boppr` moved to
`seeds/commercial_fire/` (boppr now has 4 seeds). The Businessowner's Policy product stays open.

**CR-BOPPR-07. Leatherstocking Preferred form: limit types and the mortgagee block (registry owner; opened 23 October 2026).**
Schema side settled in boppr 2.1.0 (CHANGELOG_boppr.md): Money and Securities off premises is a `sublimit` of `BOPPR_MONEY_SECURITIES`; the Pollutant
Cleanup and Fire Legal Liability limit types of that form (`per_occurrence`, `per_location`) are not made code defaults, because a changed default
changes every other boppr document. Needed in `leatherstocking/boppr.yaml`: a row rule that types those three rows, the mortgagee address block
(`interested_parties[].address`) and the loan number without its `#` prefix (`interested_parties[].loan_number`). Not done here (L1 and the registry
are not touched by the schema release).

**CR-BOPPR-08. Common-model review for the G1/bop release (common-model owner; opened 25 October 2026).**
No change to the common model: see CR-BOPGL-10 (the review covers both businessowners lines, which print the same declarations).

**CR-BOPPR-09. Package endorsements print a premium (registry owner).**
Same as CR-BOPGL-09: `Maximizer`, `Select Choice Plus`, `Policy Protector Pack` and `NYCM BOP Enhancement` are `forms_and_endorsements[].premium`.

**CR-BOPPR-10. Classification and exposure rows; documents in `G1/bop` that are not businessowners; registry entries.**
See CR-BOPGL-07, CR-BOPGL-08 and CR-BOPGL-11: the same findings apply to boppr.

**CR-BOPPR-11. Coverage-code aliases from the G1 scanned and hybrid seeds. Approved by the project owner and resolved in boppr 2.3.0 (26 October 2026).** Management liability codes and two aliases; see CHANGELOG_boppr.md.
