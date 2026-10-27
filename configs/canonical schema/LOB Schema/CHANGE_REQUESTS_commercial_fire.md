# Change requests - commercial_fire (to the common-model owner, the training-side owner and the project owner)

Raised while building the commercial_fire SPEC_21 schema 1.0.0 (10 October 2026). Nothing here is applied to the
common model: the schema composes `common_model.json` 1.1.0 unchanged, and each concept below stays in a
commercial_fire coverage code or in `additional_fields` as named until a request is accepted. Labels only; no
document values.

**CR-CFIRE-01. Shared coverage codes for premises liability (with CR-BOPPR-01, CR-BOPGL-01, CR-PROP-02).**
A commercial fire declarations prints premises liability (Coverage L, bodily injury and property damage, per
occurrence and aggregate) and a personal injury option (LS-49) beside its property; gl prints the same concepts as
`GL_BI_PD` and `GL_PERSONAL_ADVERTISING_INJURY`, and boppr, bopgl and property print them too. By the common model's
inclusion rule they belong in `fideon:shared_coverage_codes`. Until decided this line uses `CF_LIABILITY` and
`CF_PERSONAL_INJURY` (medical payments already uses the shared `X_MED_PAY`); once shared codes are released,
commercial_fire renames them in a major version with crosswalk entries (SPEC_21 4.1).

**CR-CFIRE-02. `lob_schema_map.yaml` commercial_fire entry and the line's codes (project owner).**
The entry was added with `layout_family: property_pkg`, `blocks_used: [locations, buildings]` and
`source_pdfs: {folder: TBD, carriers: 1, files: 7}`; `lob_code`, `coverage_code_prefix`, `routing_key` and
`holdout_carrier` are TBD. The batch code is needed before a `SYN-PDF-PC-<CODE>` batch ID can be issued; the schema
files use the prefix `CF_` meanwhile. The seven documents sit in three corpus folders of one carrier (boppr 1,
prop 1, cgl_prop 5), so no single source folder can be named for the synthetic-twin workspace; the seed inventory
lists them. One carrier only: SPEC_21 10.3 cannot hold a carrier out until a second carrier's commercial fire
documents arrive. Owner to assign the codes and confirm the corpus labels of the seven documents.

**CR-CFIRE-03. Training parity (fine-tuning repository owner).**
The fine-tuning repository has no schema for this line. property_pkg has no adapter and no parity test covers it.
Add `commercial_fire.json` and its coverage codes when the property_pkg family is migrated to the common model
(CR-BOPPR-02, CR-PROP-03). Until then the line is served by the base model.

**CR-CFIRE-04. Other carriers' "Commercial Fire & Liability" policies (project owner).**
Dryden Mutual prints `Commercial Fire & Liability Policy Declarations` on documents the corpus files as packages
(`boppr_cgl` 4, `cgl_prop` 2), beside a separate general liability part. The ruling names the Leatherstocking product
only, so the line's signals match the Leatherstocking banner (`DECLARATION, Commercial Fire and Liability`) and
nothing looser; those documents route as before. Decide whether they are commercial fire too.
