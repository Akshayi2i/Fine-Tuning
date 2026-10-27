# Change requests - bopgl (to the common-model owner, the training-side owner, the registry owner and the project owner)

Raised while building the bopgl SPEC_21 schema 2.0.0 (10 October 2026). Nothing here is applied to the common
model: the schema composes `common_model.json` 1.1.0 unchanged, and each concept below stays in a bopgl coverage
code or in `additional_fields` as named until a request is accepted. Labels only; no document values.

**CR-BOPGL-01. Shared coverage codes for business liability (with CR-BOPPR-01).**
The bopgl declarations print the liability rows gl codes as `GL_BI_PD`, `GL_PRODUCTS_COMPLETED_OPS` and
`GL_DAMAGE_TO_PREMISES_RENTED`, and boppr as `BOPPR_LIABILITY`, `BOPPR_PRODUCTS_COMPLETED_OPS` and
`BOPPR_FIRE_LEGAL_LIABILITY`. Printed by three lines, they meet the common model's inclusion rule for
`fideon:shared_coverage_codes`. Until decided bopgl uses `BOPGL_LIABILITY`, `BOPGL_PRODUCTS_COMPLETED_OPS` and
`BOPGL_DAMAGE_TO_PREMISES_RENTED`; once shared codes are released, bopgl renames them in a major version with
crosswalk entries (SPEC_21 4.1). The Section I property codes (`BOPGL_BUILDING`, `BOPGL_BPP`, the property
coverage extensions) repeat boppr's; the same request covers them if the owner prefers one businessowners code set.

*Status 23 October 2026 (DECISIONS rows 114, 115):* the shared codes are released in common model 1.3.0 as `X_BI_PD_LIABILITY`,
`X_PRODUCTS_COMPLETED_OPS`, `X_DAMAGE_TO_PREMISES_RENTED` and `X_PERSONAL_ADVERTISING_INJURY` (and `X_LIQUOR_LIABILITY`). bopgl 2.1.0 keeps
its own `BOPGL_` codes, because renaming a code is a major version (3.0.0, with crosswalk entries) and the registry entries' declared paths, the
golds and the varmaps name them. Open for Sagar: do bopgl, boppr and gl move to the shared codes together in their next major versions? The
property side is settled additively: every Section I / Section II row both businessowners lines print has a counterpart code on the other line
(`counterpart` key) and the same printed names.

**CR-BOPGL-02. The `business_owners` registry entries (registry owner / project owner).**
Eight `policy/*/business_owners.yaml` entries (american_reliable, coterie, hartford, madison_mutual, midstate,
mount_vernon, nycm, utica_mutual) were generated from the carrier-by-line matrix; seven declare no field map, and
their schema is the legacy `business_owners.json`. Six list a bopgl document as a sample. Decision taken here:
they are left unchanged. The coterie bopgl documents now route to the new `policy/coterie/bopgl.yaml` through its
own `lob_signals` (opted in to revise the stated `business_owners` banner line), so `coterie/business_owners.yaml`
is no longer reached by any document of the corpus; the four agency-system prints the other stubs name are page
images under an OCR text layer, which L1 does not read. Retiring the stubs (or pointing `business_owners` at
`bopgl.json`) is the owner's decision; nothing in the bopgl reads depends on it.

**CR-BOPGL-03. `lob_schema_map.yaml` bopgl entry.**
The entry was filled for this schema (`blocks_used: [locations, buildings]`, `source_pdfs: {folder: bopgl,
carriers: 5, files: 13}`); `lob_code`, `routing_key`, `coverage_code_prefix` and `seeds` stay as the owner sets
them (the seeds are DRAFT). The schema files use the prefix `BOPGL_` meanwhile. Owner to confirm or assign.

**CR-BOPGL-04. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `bopgl.json`. property_pkg has no adapter and no parity test covers
bopgl. Decide with CR-BOPPR-02 (migrate the property_pkg family together, or allow bopgl alone). Until then bopgl
is served by the base model.

**CR-BOPGL-05. One line per Leatherstocking product (project owner; with CR-BOPPR-06).**
The two Leatherstocking `DECLARATION, Businessowner's Policy` documents filed under bopgl are the product the
boppr entry reads for its own seed. They stay declined (the boppr entry's `require_coded_coverages` steps aside on
their supplemental rows, and no bopgl entry claims them) until the owner decides the product's line; their DRAFT
golds against bopgl.json are drafted so either decision can be checked.

**CR-BOPGL-06. Scans filed as native (project owner).**
Five bopgl documents (the four agency-system prints and one coterie policy) are page images under an invisible
OCR text layer although the corpus manifest records them as `native_pdf`. L1 never reads them; their modality
should be recorded as scanned.

**CR-BOPGL-07. Documents in `G1/bop` that are not businessowners policies (corpus owner / project owner; opened 25 October 2026).**
The folder holds 6 documents. Two are businessowners declarations (Utica First `Businessowners Special Policy`, NYCM `BP 00 03`). Three are Utica First
`Contractors Special Policy` documents (form AP 100: a general liability coverage part with an optional liability list and a tools and equipment list), the
same product as the Utica First files of `G1/cgl`, although their names say `bopgl`. One, named `bop quote`, is a USLI commercial package policy (a
Commercial Liability coverage part and a Commercial Property coverage part, not a businessowners form). None of the four was used for this release. They route
to gl and to the commercial package line; the folder (or the file names) should be corrected, and any gold drafted against bopgl for them redone.

**CR-BOPGL-08. Classification and exposure rows (common-model owner / project owner).**
NYCM prints the rated class, its description, a liability class group and a liability exposure base; Utica First prints a class description. The home is the
common model's `rating_exposures` block (gl and wc compose it); bopgl composes only `locations` and `buildings`. Adding the block to bopgl is additive but
changes its prompt schema and section map, and property_pkg has no adapter, so it is the owner's call. Until then those values go to `additional_fields`.

**CR-BOPGL-09. Package endorsements print a premium (registry owner).**
`Maximizer`, `Select Choice Plus`, `Policy Protector Pack` and `NYCM BOP Enhancement` bundle coverages under one form and print one premium (or `Incl.`).
The premium belongs to `forms_and_endorsements[].premium`; the bundled coverages are coded individually. A registry field map for these carriers must not
route that premium to a coverage.

**CR-BOPGL-10. Common-model review (common-model owner).**
Every concept the two declarations print has a home in the common model (fire district, distance to hydrant and fire station, building area, policy term,
producer fields, taxes and fees, bill type, interested parties). Shared coverage codes were considered under the 1.3.0 rule (a code is shared when two or
more lines print it and the owner has not ruled against it): the new rows are printed by bopgl and boppr only, which 1.3.0 left as counterpart codes (spoilage,
debris removal and similar). So `common_model.json` stays 1.4.0 and `CHANGELOG_common.md` has no new entry. Names of shared concepts (hired and non-owned
auto, additional insured, cyber, equipment breakdown, ordinance or law) were added as aliases on the line's own entries for those shared codes. If the owner
prefers them in the common model, they are alias-only additions (a minor version).

**CR-BOPGL-11. Registry entries for the three carriers (registry owner).**
`policy/nycm/business_owners.yaml` is a stub with no declared paths; Utica First has no businessowners entry (auto, gl, property, umbrella, wc) and USLI has
none either. New-carrier entries and their field maps wait for the de-identified V3 samples (DECISIONS row 144), so none was written here. The schema side is
ready: the codes, aliases and captions above are what such an entry's rows resolve against.

**CR-BOPGL-12. Coverage-code aliases from the G1 scanned and hybrid seeds (registry owner). Approved by the project owner and resolved in bopgl 2.3.0 (26 October 2026).** 33 codes and 14 aliases; see CHANGELOG_bopgl.md.
