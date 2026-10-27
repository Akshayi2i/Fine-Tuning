# Changing the common model

`common_model.json` is shared by every policy_check LOB schema. Family teams
(personal_lines, casualty_fleet, property_pkg, ...) draft their schemas in
parallel, so the model changes only through the owner and only in ways that keep
every LOB schema working (SPEC_21 v0.9 §4.2.7).

## Asking for a change

A LOB schema never adds to or renames a common block. When your line prints a
concept the common model has no home for, open a change request:

1. Open a GitHub issue labelled `common-model`, and record the request in your
   line's `policy_check/CHANGE_REQUESTS_<lob>.md`.
2. State:
   - the concept, in domain words;
   - the LOBs that print it: two or more, or the reason a single line still
     needs it in the common model;
   - example labels as the seeds print them;
   - the block and field you propose, with its type.
3. Until the request is released, the value travels as `additional_fields`
   (label and value), or in your LOB's own block if no other LOB prints it.

A unit block only your LOB prints goes in your LOB file's own `$defs`
(LOB-local, §4.2.4). Check the common blocks first: Location, Building, Vehicle,
Driver, ScheduledItem, RatingExposure, RatingModifier, underlying insurance,
claims-made terms, insuring agreements and benefit plans are reused, never
redefined.

## What a release may change

Within 1.x a release may only:

- add a block;
- add an optional property;
- add an enum value or a shared coverage code;
- reword a description.

Renaming, removing, retyping or making a field required needs version 2.0,
planned with every family team. The owner batches accepted requests into one
minor release, so adapters are rechecked once rather than once per request.

## Checks (CI, `ci-pipeline-api.yml`)

| Command | Fails when |
|---|---|
| `python tools/check_lob_defs.py` | A LOB file defines a block named like a common one, or two definitions (two LOB files, or a LOB file and the common model) share half or more of their field names. |
| `python tools/check_common_compat.py --base-ref <previous release>` | Something was removed, renamed or retyped, a field became required, an enum value was dropped, or the file changed without a higher version. |
| `python tools/build_prompt_schemas.py --check` | A prompt schema or `prompt/manifest.json` is stale. |
| `python tools/prompt_schema_families.py --base-ref <previous release>` | Never fails. Lists the layout families whose prompt schemas changed: the adapters to recheck (SPEC_09 §4.1). |

Run `check_lob_defs.py` before drafting golds against a LOB-local block.

## Release checklist (owner)

1. Edit `common_model.json` and raise `fideon:common_model_version`, by the minor
   number for a 1.x release.
2. Regenerate the generated files:
   - `python scripts/build_common_model_fields_md.py`
   - `python tools/build_fallback_schema.py`
   - `python tools/build_prompt_schemas.py` (prompt schemas and manifest)
   - `python -m vlm_server.fsm.tests.test_compiler --update-fixture`. Keep the
     fixture's `_note` and add a sentence for the release; the helper drops it.
3. Raise `common_model_version` in `lob_schema_map.yaml`.
4. Run `tools/check_common_compat.py` and `tools/check_lob_defs.py`.
5. Add a release section to `CHANGELOG_common.md` listing the requests it
   contains. Mark each request applied in its `CHANGE_REQUESTS_<lob>.md`.
6. Run `tools/prompt_schema_families.py` and tell each listed family that its
   adapter needs a recheck. The fine-tuning repository takes the release, makes
   any port changes and re-records the parity fixtures.

LOB schemas keep the version they pin. A LOB adopts a new field in its own
release, raising its `fideon:common_model_version`.
