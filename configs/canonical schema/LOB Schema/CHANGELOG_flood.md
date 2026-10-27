# Changelog - flood canonical schema

## 2.0.0 - 14 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (flood, L1 code and schema id `flood`). It replaces the pre-SPEC_21 `flood.json`
(1.6.0, `_common`-shaped with a `flood` block), which did not compose the common model. A new major version because
every path changes (SPEC_21 4.1).

**Built from** the single-LOB documents of the source-data `flood` folder (3 carrier folders, 34 documents), read in
place:
- Sterling Surplus Underwriters' private flood on Lloyd's paper (29): one- and two-page declarations of three yearly
  layouts and electronic policy sets that open with the broker's cover letter and bind a certification of adequacy,
  the excess line declinations and the notice of excess line placement. One of the 29 is a renewal quote, not
  declarations (CR-FL-02).
- Foremost Insurance Company's National Flood Insurance Program declarations (4, two layouts).
- IAT Insurance Group's declarations (1): a scan under an OCR text layer (CR-FL-02).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `locations`, `buildings`
  and `countersignature` blocks (the lob_schema_map entry). No LOB block and no local `$defs`: each flood coverage is
  a coverage with its limit, deductible and premium; the insured property is a `locations[]` element and its building
  facts a `buildings[]` element; the flood rating facts the common model has no field for are `additional_fields`
  labels (CR-FL-03).
- **Coverage codes:** `flood.coverage_codes.yaml`, 7 codes: 6 prefixed `FL_` (building or dwelling, contents, other
  structures, basement coverage, basement personal property, increased cost of compliance) and the shared
  `X_LOSS_OF_USE`. The prefix is provisional (CR-FL-01).
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule.
- **Crosswalk:** `flood.crosswalk.yaml` maps all 747 legacy leaves and the paths the line's registry entry declared
  before 2.0.0 (271 to a SPEC_21 path, 476 to an `additional_fields` label, 5 dropped).
  `tools/registry_migrate.py --lob flood --check`: no declared path needed rewriting.

**Registry** (`config/document_registry/policy/`):
- `sterling_flood/flood.yaml`: rewritten. The three declarations layouts are read by their own geometry (captions
  beside or below their values, side-by-side party blocks read by column bands, the coverage and premium columns,
  the forms paragraph, the mortgagee boxes), and so are the policy set's cover letter, certification of adequacy,
  declinations and notice of excess line placement. The fingerprint `PC License #...` is now quoted (unquoted, YAML
  read everything from ` #` as a comment, leaving the generic `PC License`). 28 of 28 seeds read at L1; the renewal
  quote's title is not matched and it escalates, as before.
- `foremost/flood.yaml`: new, the two Foremost NFIP layouts; `own_line_only`, so it never answers for another
  Foremost line. 4 of 4 seeds read at L1.
- Both carry the shared `policy_fields` block.
- L1 opt-in used (inert for any entry that does not declare it): `row_lists.<list>.town_only_address` (a party
  printed with no street), with `fideon/l1/tests/test_sectioned_party_town_only.py`.

**Seeds:** 32 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/flood/`, not in Git), pending reviewer sign-off.

**Pipeline:** `config/canonical_schema/prompt/policy_check/flood.prompt.json` generated with
`tools/build_prompt_schemas.py`; the schema hash fixture holds its hash. property_pkg has no adapter yet. The
fine-tuning repository still holds the legacy file (CR-FL-05).
