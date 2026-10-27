# Change requests - builders_risk (to the common-model owner, the registry owner and the project owner)

Raised while building the builders_risk SPEC_21 schema 3.0.0 (14 October 2026). Nothing here is applied to the
common model. Labels only; no document values.

**CR-BR-01. Coverage code prefix (project owner).**
`lob_schema_map.yaml` leaves the builders_risk `lob_code` and `coverage_code_prefix` TBD. The schema uses `BR_` for
its 35 own codes (plus the shared `X_ORDINANCE_LAW`, `X_WATER_BACKUP`, `X_TERRORISM`); confirm the prefix.

**CR-BR-02. No declarations among the sources (project owner, registry owner).**
None of the bldrk folder's 5 documents is a builders risk declarations page that L1 can read: US Assure's are a
renewal APPLICATION printed from the carrier's portal (with an e-signature completion certificate on the signed
print), Vacant Express's is a premium invoice and Zurich's is a scan. The coverage codes and the coverage aliases
come from the legacy schema and the Zurich supplemental declarations' captions; no seed gold exercises them. A
digital builders risk declarations (a coverage schedule with limits) is needed before the codes can be measured.
US Assure's application reads at L1 because the entry declares what it prints; its golds have no coverage
(`mandatory_waivers` for `coverages[].limits[].amount`). Decide whether the application is an accepted policy_check
document type for the line.

**CR-BR-03. Project facts in the common model (common-model owner).**
A builders risk policy describes a project, and the common model has no fields for the project's name and type,
construction start and expected completion dates, completed value, hard and soft costs, percentage complete at
inception, general contractor, owner, renovation-or-new-construction flag, reporting basis or rate; they are
`additional_fields` labels (`Expected completion date of project` is the one the seeds print). Decide whether a
`project` block (or `locations[]` / `buildings[]` fields) should hold them; inland marine installation floaters
print some of them too.

**CR-BR-04. Vacant Express premium invoice (registry owner).**
The Vacant Express document (`Invoice for Insurance Premium`, on Diamond State Insurance Company paper) is a billing
statement for a builders risk renewal offer, not declarations: no coverage, limit or deductible is printed. It is a
clean decline: it escalates to L3 at the plausibility gate (no coverage limit or premium read while the page prints
currency amounts). No seed gold.

**CR-BR-05. Zurich scan (registry owner).**
The Zurich builders risk policy's declarations and supplemental declarations pages are an invisible OCR text layer
over a scan. L1 never reads a scan: it escalates to L3 at the plausibility gate, and no seed gold was drafted. Its
supplemental-declarations captions (forms U-GU and HBIS) gave the coverage aliases.

**CR-BR-06. Generated and shared files (program coordinator).**
Not edited here: the `builders_risk` entry of `common/lob_schema_map.yaml` (`schema_lob`, `blocks_used`,
`source_pdfs`), `common/CHANGELOG_common.md`, the prompt schema (`tools/build_prompt_schemas.py`) and the schema hash
fixture.

**CR-BR-07. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `builders_risk.json`. Decide with the property_pkg family's other lines.
