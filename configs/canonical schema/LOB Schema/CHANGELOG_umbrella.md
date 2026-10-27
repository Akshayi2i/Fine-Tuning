# Changelog - umbrella canonical schema

## 2.0.0 - 11 October 2026 (SPEC_21 Phase 2; draft for the project owner's approval)

First SPEC_21 schema of the line (commercial umbrella and excess liability, L1 code and schema id `umbrella`, batch
code UM). It replaces the pre-SPEC_21 `umbrella.json` (1.6.0, `_common`-shaped with an `umbrella` block), which did
not compose the common model. A new major version because every path changes (SPEC_21 4.1).

**Built from** the single-LOB documents of the source-data `cumbr` folder, read in place: Utica First's commercial
umbrella/excess liability declarations (2) and CNA's follow-form excess insurance declarations (1, over a directors
and officers policy). Sutton National's commercial excess declarations page is a scan under an OCR text layer: not a
seed, it stays L3 (CR-UM-02).

- **Composition:** `common_model.json` 1.1.0 by `$ref`: the SPEC_21 4.2.1 envelope and the `underlying_insurance`,
  `attachment` and `countersignature` blocks (the lob_schema_map entry adds `countersignature`). No LOB block.
- **Coverage codes:** `umbrella.coverage_codes.yaml`, 6 codes: the umbrella and excess layers (`UM_UMBRELLA_LIABILITY`,
  `UM_EXCESS_LIABILITY`), three codes for scheduled underlying policies (`underlying_insurance[].coverage_code`:
  general liability, the followed policy, an underlying excess layer) and the shared `X_TERRORISM`.
- **Mandatory fields:** the common model's five plus `coverages[].limits[].amount`.
- **Audit rules:** the common model's defaults without the claims-made rule.
- **Crosswalk:** `umbrella.crosswalk.yaml` maps all 401 legacy leaves (225 to a SPEC_21 path, 171 to an
  `additional_fields` label, 5 dropped). `tools/registry_migrate.py --lob umbrella`: no declared path needed rewriting.

**Registry** (`config/document_registry/policy/`):
- `utica_first/umbrella.yaml`: a sectioned map of the ULCDNY declarations (insured, agent, period, policy and
  retained limits, premium, countersignature; the limits of insurance and the schedule of underlying insurance; the
  forms inventory), the line signal of their title and an override of the `entry_fingerprint` line source (this
  carrier's gl entry claimed the documents by fingerprint); 2 of 2 seeds read at L1.
- `continental_casualty/umbrella.yaml` (new): CNA's excess insurance declarations and the underlying excess schedule;
  1 of 1 seed. It carries the shared `policy_fields` block.

**Seeds:** 3 DRAFT golds (`synth_core/fideon/synth_core/data/seeds/umbrella/`, not in Git), pending reviewer
sign-off.

**Pipeline:** `config/canonical_schema/prompt/policy_check/umbrella.prompt.json` generated with
`tools/build_prompt_schemas.py`; the schema hash fixture holds its hash. The fine-tuning repository still holds the
legacy file (CR-UM-03).
