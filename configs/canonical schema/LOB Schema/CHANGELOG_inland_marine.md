# Changelog - inland_marine canonical schema

## Aliases - 12 October 2026 (package policies, PKG2; no field or code change)

`inland_marine.coverage_codes.yaml`, the inland marine part of Dryden Mutual's `Craft 12 - Contractors Package Policy`: `IM_CONTRACTORS_EQUIPMENT` + `Tools Equipment Coverage` (form DMIC MR-79). The code list is unchanged.

## 2.0.0 - 10 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (commercial inland marine, L1 code and schema id `inland_marine`). It replaces the
pre-SPEC_21 `inland_marine.json` (1.0.0, `_common`-shaped with an `inland_marine` block), which did not compose the
common model. A new major version because every path changes (SPEC_21 4.1).

**Built from** the 10 single-LOB documents of the source-data `inmrc` folder (4 folders), read in place, all with a
native text layer: Hartford's commercial articles policies (3), Leatherstocking's contractors equipment declarations
(4), Arch's apparatus and equipment declarations filed under McNeil & Company (1) and two agency-system prints
(HawkSoft AL3 Viewer) of a scheduled equipment policy filed under RT Specialty (2).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `locations`,
  `scheduled_items` and `countersignature` blocks (the lob_schema_map entry). No LOB block and no local `$defs`: a
  covered class of property is a coverage with its limit and deductible, each scheduled article, apparatus or piece
  of equipment a `scheduled_items[]` element.
- **Coverage codes:** `inland_marine.coverage_codes.yaml`, 7 codes: 6 prefixed `IM_` (commercial articles,
  contractors equipment, emergency apparatus, portable equipment, other property, scheduled equipment) and the shared
  `X_TERRORISM`. The prefix is provisional (CR-INMRC-01).
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule.
- **Crosswalk:** `inland_marine.crosswalk.yaml` maps all 391 legacy leaves and the paths the line's registry entries
  declared before 2.0.0 (242 to a SPEC_21 path, 143 to an `additional_fields` label, 6 dropped).
  `tools/registry_migrate.py --lob inland_marine` was run on the five entries before their field maps were written:
  no declared path needed rewriting.

**Registry** (`config/document_registry/policy/`):
- `leatherstocking/inland_marine.yaml`: the LCIC(4/11) declarations read by its own layout (header, premises,
  optional items, rating information, forms paragraph, loss payee, the schedule of coverages with one item per
  centred row group); 4 of 4 seeds read at L1.
- `hartford/inland_marine.yaml`: the line signal of Form MS 18 and a field map of the declarations, the terrorism
  schedule and the commercial articles schedule; 3 of 3 seeds read at L1.
- `arch/inland_marine.yaml`: the line signal and a field map of the DEC MA 0500 declarations, the endorsement
  schedule, the long named insured endorsement, the schedule of apparatus and the loss payable, automatic increase
  and guaranteed replacement cost schedules; 1 of 1 seed read at L1.
- Every entry carries the shared `policy_fields` block. `american_reliable/inland_marine.yaml` and
  `zurich/inland_marine.yaml` are unchanged (CR-INMRC-02).
- L1 opt-ins used (inert for any entry that does not declare them): `merge_gap`, `merge_columns`, `unclipped`
  (row lists), `verbatim_values` (sectioned read), `x_min`/`x_max` and `all` (document fields).

**Seeds:** 10 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/inland_marine/`, not in Git), pending reviewer
sign-off.

**Pipeline:** `config/canonical_schema/prompt/policy_check/inland_marine.prompt.json` regenerated with
`tools/build_prompt_schemas.py`; the schema hash fixture holds the new hash. property_pkg has no adapter yet. The
fine-tuning repository still holds the legacy file (CR-INMRC-03).
