# SPEC_21 — LOB Canonical Schema and Synthetic Digital-Twin Generation

**Status:** v0.6 — 3 October 2026. v0.6 aligns §6.2, §6.4, §7.1 and §7.10 with the delivered `synth_core` 0.1.0 (see its README) and records one open decision on scanned read-back (D11, §16). v0.5 — all decisions D1–D10 resolved (§16). Affected specs amended the same day (§15). v0.4 added the hand-off package (§1.1), the run config (§7.10) and the phased Claude Code workflow (§14). v0.5 replaces §4.2 with the cross-LOB common model (envelope, building blocks, limits inside coverages, package parts, path selectors) and provides `synth_core` ready-built.
**Owner:** ML Engineering / Data Engineering (one developer per LOB)
**Depends on:** SPEC_00 (FieldValue, TextSection, schema location), SPEC_02 / SPEC_19 (L1 carrier registry), SPEC_04 (chunking), SPEC_05 (MinerU), SPEC_09 (LoRA SFT), SPEC_17 / SPEC_20 (evaluation), `SYNTHETIC_DATA_PROCESS.md` (batch IDs, Presidio, DVC)
**Supersedes:** SPEC_16 §4 and §5 (synthetic generators), the `fideon_synth` package as the production generator, every existing `config/canonical_schema/policy_check/<lob>.json`, and every seed gold produced by `seedkit auto` (see §15)
**Language:** Python 3.11 · PyMuPDF · MinerU (SPEC_05) · Pydantic v2 · jsonschema · PyYAML · OpenCV / Pillow (scanned twins)
**Audience:** Developers working in Claude Code. Each section states what is built, where, and how it is accepted.

---

## 1. Purpose

Each line of business (LOB) receives:

1. A **canonical schema** that describes every policy of that LOB from any carrier, in domain terms rather than in any one carrier's labels.
2. A **crosswalk** that keeps the L1 carrier registry (SPEC_02) pointing at that schema.
3. A set of **human-verified seed golds**: one canonical JSON per seed PDF, digital or scanned, describing that PDF completely.
4. A **LOB-specific synthetic generator** that turns every seed into N digital twins. A twin has the seed's look but different values and, where declared, a different number of repeating items (vehicles, drivers, locations, scheduled items, endorsements). Each twin comes with a gold JSON that describes it exactly.

The twins supplement verified real examples for LoRA SFT training (SPEC_09). The canonical JSON is also the input to policy comparison, so every word the policy prints must be present in the output JSON (§4.6).

### 1.1 Assignment and hand-off package

Each developer is assigned **one LOB** and works in Claude Code. The developer receives:

| Item | Source |
|---|---|
| This spec (SPEC_21) | Project owner |
| The LOB name (e.g. `homeowners`) and its batch LOB code (§11) | Project owner |
| The seed PDFs for that LOB, from several carriers, digital and/or scanned, already de-identified | Project owner; placed under `data/seeds/<lob>/<Carrier>/` |
| The name of the seed reviewer (§5.4) | Project owner |
| The common model `config/canonical_schema/common/common_model.json` and `lob_registry.yaml` (§4.2), and the name of the common-model owner | Common-model owner |
| `fideon/synth_core` (§7.1), already built and tested | Repository |
| Repository access: `config/document_registry/policy_check/*/<lob>.yaml` (L1 registry YAMLs for the LOB), the current `config/canonical_schema/policy_check/<lob>.json` (reference only), `fideon_synth` (source for porting), `tools/registry_lint.py` | Repository |
| SPEC_00, SPEC_02, SPEC_04, SPEC_05, SPEC_09, SPEC_16 §2, `SYNTHETIC_DATA_PROCESS.md` | `VLM Pipeline - Implementation Specs/` |
| A GPU environment with the production MinerU configuration (SPEC_05), for scanned seeds and SFT packaging | ML Ops |

From these, the developer produces, for the assigned LOB:

1. the canonical schema (§4);
2. the crosswalk and migrated L1 registry YAMLs (§4.7);
3. for every seed PDF: a verified seed gold, a variation map and a review record (§5, §6);
4. the LOB generator (§7);
5. for every seed PDF: N digital twins, each with its gold JSON (§8), from a committed run config (§7.10).

**Shared core.** `fideon/synth_core` (§7.1) is provided ready-built with the hand-off package. Developers build their LOB generators on top of it and do not modify it locally. Fixes and extensions found by any developer go to `synth_core` by pull request to its owner (§14).

---

## 2. Why this spec exists

A review of `fideon_synth` (October 2026) on the Chubb homeowners and AEIC personal-auto test outputs found the following. The PDF rewrite mechanics worked: twins rendered cleanly, no replaced value leaked, and the gold validated against the schema. The training data was still unusable:

| Finding | Evidence |
|---|---|
| Seed gold produced by code, not verified | Provenance `"seed built by code (seedkit auto)"`. Chubb `coverage_a_dwelling_limit` = `"$1"`; the printed dwelling limit is $2,100,000 (page 20). `protection_class` = `"100 North"`. |
| Seed gold nearly empty in structured fields | Chubb: 39 structured fields filled out of ~1,550 schema leaves; forms schedule, locations, coverages and deductibles empty. AEIC: 31 fields; vehicles had no VIN, year, make, model or drivers. |
| Completeness check defeated | A printed value counted as "in the gold" if it appeared anywhere, including the verbatim page text. Empty structured fields passed. |
| Too little variation | Chubb twin: 7 fields changed. Insured, address, policy number, policy dates and premium unchanged. |
| Same value replaced inconsistently | AEIC policy number printed as `268-22-38-40` and `268-22-38-40A` became two unrelated numbers in the twin. |
| Protected text damaged | Carrier URL `https://www2.Chubb.com/us-en/privacy.aspx` became `northfieldridge.aspx`. |
| Relationships broken | Contents limit (50% of dwelling) changed while dwelling did not; AEIC vehicle values all scaled by one factor. |
| Schema too large and carrier-shaped | `homeowners.json` v1.6.0: ~1,550 leaves including `printed_lines`. |

The root causes are: an unverified seed, value semantics inferred by regex, and an oversized schema. This spec fixes all three.

---

## 3. Principles (non-negotiable)

**P1 — No gold without human verification.** A seed gold is a draft until a named reviewer signs it off (§5.4). The generator refuses unsigned seeds.

**P2 — Model understands, code mutates.** Claude (in the developer's Claude Code session) drafts the schema, the seed gold and the variation map. Deterministic Python performs every PDF, image and gold change. No LLM runs inside the generation loop.

**P3 — Three tiers in every canonical JSON:**

| Tier | Holds | Filled at inference by | Trained / gated |
|---|---|---|---|
| Core fields | Domain concepts the business uses (§4.3) | Extraction model (or L1) | Yes / Yes |
| Overflow (`additional_fields`) | Every other printed value, as label + value + page | Extraction model (or L1) | Yes / Loosely |
| Full text (`text_sections`) | Every printed word, organised by form and section | MinerU + section builder (deterministic) | No / Coverage check only |

**P4 — One value, one replacement, everywhere.** Every printed form of the same value (with suffixes, reformatted, abbreviated, split across lines) belongs to one equivalence group and receives one consistent replacement.

**P5 — Twins stay plausible.** Relationships, formats and domain rules that hold in the seed hold in every twin.

**P6 — Nothing unchecked is written.** Every twin passes every check in §9 or is discarded and reported.

**P7 — One canonical shape for every layer.** L1, L2 and L3 emit the same canonical JSON (SPEC_00 §5.4). The L1 registry maps into this schema; no layer has a private shape.

---

## 4. Canonical schema per LOB

### 4.1 Location and versioning

```
config/canonical_schema/common/common_model.json         # shared envelope and building blocks (§4.2), central owner
config/canonical_schema/policy_check/<lob>.json        # JSON Schema (draft 2020-12), the contract; composes the common model
config/canonical_schema/policy_check/<lob>.fields.md   # one row per core field: definition, examples, carriers seen
config/canonical_schema/policy_check/<lob>.coverage_codes.yaml  # coverage codes, names, aliases (§4.2.4)
config/canonical_schema/policy_check/<lob>.crosswalk.yaml  # old schema paths -> new schema paths (§4.7)
config/canonical_schema/policy_check/CHANGELOG_<lob>.md
```

- `<lob>` uses the existing folder names: `homeowners`, `dwelling_fire`, `personal_auto`, `motorcycle`, `recreational_vehicle`, `personal_umbrella`, `ocean_marine`, and others as assigned.
- `fideon:schema_version` uses semantic versioning:
  - adding a core field is a minor version;
  - renaming or removing one is a major version and requires a crosswalk entry, regenerated golds, and updated L1 registry YAMLs for that LOB.
- The schema is written fresh for this spec. Existing schemas may be consulted for field ideas but are not copied.

### 4.2 Common model — one structure for all LOBs

Fideon covers 38 LOBs across five layout-family adapters (SPEC_09). Every LOB schema is composed from **one central common model**, so the same concept has the same name and shape in every LOB. Developers do not design their own fields for a concept the common model already covers.

```
config/canonical_schema/common/common_model.json      # envelope + building blocks ($defs); central owner only
config/canonical_schema/common/CHANGELOG_common.md
config/canonical_schema/policy_check/<lob>.json       # composes the common model via $ref, adds code lists and the LOB block
config/canonical_schema/policy_check/<lob>.coverage_codes.yaml
```

The common model has its own semantic version. A LOB schema records the common-model version it composes (`fideon:common_model_version`). The **common-model owner** (named by the project owner) approves every change to it. A developer who needs a new building block, or a new field in an existing one, raises a change request rather than adding it to the LOB schema.

#### 4.2.1 Envelope (every LOB, every document)

| Block | Content |
|---|---|
| `document` | `doc_type`, `modality`, `transaction_type` (new, renewal, endorsement, cancellation, reinstatement, quote, binder, application, declarations_only), title, issue date, print date, page count, source file name |
| `carrier` | Writing company, NAIC code, group, address, phone, web address, admitted status where printed |
| `producer` | Agency, producer code, address, contact |
| `named_insured` | Primary name, additional named insureds, DBA, entity type, mailing address, contact |
| `policy` | Policy number, prior policy number, effective and expiration dates, term, coverage trigger (`occurrence`, `claims_made`, `claims_made_and_reported`, `not_applicable`), package flag |
| `lob_parts[]` | One entry per LOB in the document (§4.2.3) |
| `coverages[]` | Every coverage and every limit (§4.2.2) |
| `deductibles[]` | Deductibles that apply across coverages or perils (§4.2.2) |
| `interested_parties[]` | Role (mortgagee, loss payee, additional insured, lienholder, certificate holder, additional interest), name, address, loan number, applies-to units |
| `premium` | Total, deposit, minimum earned; `items[]` (per coverage, unit or part), `taxes_fees[]`, `surcharges[]`, `discounts[]` |
| `billing` | Payment plan, installments |
| `forms_and_endorsements[]` | Form number, edition date, title, form type, part, applies-to, premium, page range |
| `additional_fields[]` | Overflow tier (§4.5) |
| `text_sections` | Full-text tier (§4.6) |

#### 4.2.2 Shared building blocks

A LOB schema includes the blocks its documents carry and no others. Field names inside a block are fixed by the common model.

**Coverages and limits — one place for every limit.** A limit is always stored inside the coverage it limits. No LOB block holds a limit.

```json
"coverages": [{
  "coverage_id": "cov_3",                       // structural id, unique in the document
  "coverage_code": "HO_COV_C",                  // from <lob>.coverage_codes.yaml
  "coverage_name": {FieldValue},                // as printed
  "part": "part_1",                             // lob_parts[].part_id
  "applies_to": ["bldg_1"],                     // unit_ids; empty = whole policy
  "included": {FieldValue},                     // included / excluded / not purchased
  "limits": [{
    "limit_type": "per_occurrence",             // enum below
    "amount": {FieldValue},
    "percentage": {FieldValue},                 // when stated as % of another coverage
    "basis_coverage_code": "HO_COV_A",          // the coverage the percentage refers to
    "description": {FieldValue}                 // printed qualifier, e.g. "theft of jewelry"
  }],
  "deductibles": [{deductible}],
  "premium": {FieldValue},
  "valuation": {FieldValue},                    // replacement cost, actual cash value, agreed value, stated amount
  "coinsurance_percent": {FieldValue},
  "form_refs": ["HO 04 90 05 11"]
}]
```

- **`limit_type` values:** `per_occurrence`, `aggregate`, `products_completed_ops_aggregate`, `per_person`, `per_accident`, `property_damage`, `combined_single`, `per_claim`, `each_employee`, `policy_limit`, `per_location`, `per_item`, `per_day`, `per_disablement`, `sublimit`, `blanket`, `scheduled_total`, `other`.
- **Split limits** (auto 100/300/50) are three `limits[]` entries of one coverage (`per_person`, `per_accident`, `property_damage`).
- **Sublimits** (jewelry theft $1,500) are `sublimit` entries with a `description`, inside the coverage they restrict.

**Deductible shape** (inside a coverage, or in top-level `deductibles[]` when it applies across coverages or perils):

```json
{"deductible_type": "flat | percentage | waiting_period | sir | retention | franchise",
 "amount": {FieldValue}, "percentage": {FieldValue}, "percentage_basis": {FieldValue},
 "peril": {FieldValue}, "applies_to_coverages": ["HO_COV_A"], "applies_to": ["loc_1"]}
```

**Risk units.** Every unit carries a structural `unit_id` (e.g. `veh_2`) that coverages, interested parties and premium items refer to.

| Block | Main fields | Typical LOBs |
|---|---|---|
| `locations[]` | location number, address, county, territory, protection class, distance to fire station/hydrant | property, BOP, homeowners, dwelling fire, GL, WC |
| `buildings[]` | location ref, building number, construction, occupancy, year built, area, stories, roof, protective devices, sprinklered | property, BOP, homeowners, dwelling fire |
| `vehicles[]` | vehicle number, year, make, model, VIN, body type, use, garaging location ref, cost new, stated amount, symbol(s) | personal auto, commercial auto, truckers, motorcycle, RV |
| `drivers[]` | name, date of birth, licence state and number, relationship, status (rated, excluded) | personal auto, commercial auto, motorcycle |
| `watercraft[]` | HIN, year, make, model, length, hull type, hull value, motor(s), navigation area | watercraft, ocean marine, RV |
| `scheduled_items[]` | category, description, serial, value, valuation, location ref | inland marine, valuables, contractors equipment, homeowners scheduled property |

**Rating.**

| Block | Main fields | Typical LOBs |
|---|---|---|
| `rating_exposures[]` | state, location ref, class code, class description, exposure basis (payroll, sales, units, area, receipts), exposure amount, rate, rate basis, premium | WC, GL, commercial auto, garage, liquor |
| `rating_modifiers[]` | modifier type (experience mod, schedule, IRPM, multi-policy, protective device), factor or percent, applies-to | WC, GL, property, auto, personal lines |

**Line-type blocks.**

| Block | Main fields | Typical LOBs |
|---|---|---|
| `claims_made_terms` | retroactive date, prior/pending litigation date, continuity date, extended reporting period (length, premium %), defense within limits | D&O, E&O, cyber, EPLI, professional liability |
| `insuring_agreements[]` | agreement code and name, coverage refs, retention | D&O (Side A/B/C), cyber, crime |
| `underlying_insurance[]` | carrier, policy number, coverage code, limits[], dates | umbrella, excess |
| `attachment` | retained limit, SIR, attachment point, follow-form flag | umbrella, excess |
| `bond` | bond type, bond number, penal sum, principal, obligee, dates | surety, fidelity, crime bonds |
| `benefit_plans[]` | plan id and name, plan type (life, AD&D, STD, LTD, medical, dental, vision), eligible class, benefit amount or formula, maximum, waiting period, contribution, rate and rate basis | life, group medical/accident |

#### 4.2.3 Package policies — `lob_parts[]`

```json
"lob_parts": [{"part_id": "part_1", "lob": "cgl", "title": {FieldValue}, "coverage_part_form": {FieldValue}, "premium": {FieldValue}}]
```

- A single-LOB policy has one part.
- A package (CPP, BOP, farm) has one part per LOB. Every coverage, unit, form and premium item carries its `part`.
- This matches the multi-pass package extraction of SPEC_06/SPEC_09 (`PerLOBCanonical`). The per-LOB result is the subset of the canonical JSON whose `part` equals that LOB's part.

#### 4.2.4 LOB block and coverage codes

- **`<lob>` block:** only concepts unique to the LOB that no building block covers (e.g. homeowners policy form HO-3/HO-5, personal auto PIP state option). Always small; any field that recurs in two LOBs is moved to the common model.
- **`<lob>.coverage_codes.yaml`:** the LOB's coverage list. Each entry has `code`, `name`, `aliases` (printed names across carriers) and `default_limit_types`.
  - Codes are prefixed by LOB (`HO_COV_A`, `PA_BI`).
  - A coverage shared across LOBs uses the shared prefix `X_` (e.g. `X_PERSONAL_LIABILITY`, `X_MED_PAY`), so comparison matches it across LOBs.
  - Policy comparison aligns coverages by code, not by printed name.

#### 4.2.5 Building blocks by layout family

| Layout family (adapter) | LOBs (SPEC_09) | Blocks used besides the envelope |
|---|---|---|
| `personal_lines` (first adapter; seeds available) | homeowners, dwelling fire, personal auto, motorcycle, recreational vehicle, personal umbrella, ocean marine / watercraft | locations, buildings, vehicles, drivers, watercraft, scheduled_items, rating_modifiers, underlying_insurance and attachment (personal umbrella) |
| `casualty_fleet` | CGL, commercial auto, truckers, WC, umbrella | locations, vehicles, drivers, rating_exposures, rating_modifiers, underlying_insurance, attachment |
| `property_pkg` | property, BOP, inland marine, flood, earthquake | locations, buildings, scheduled_items, rating_modifiers, lob_parts (BOP) |
| `exec_specialty` | D&O, E&O, cyber, EPLI, crime, bonds | claims_made_terms, insuring_agreements, bond |
| `benefits_misc` | life, group medical/accident, specialty | benefit_plans |

The exact list of 38 LOBs and their codes is maintained by the common-model owner in `config/canonical_schema/common/lob_registry.yaml` (LOB name, layout family, LOB code, blocks used). When seeds for a later adapter arrive, the owner first checks that the common model holds their concepts. Gaps are closed in the common model before developers of that adapter start.

#### 4.2.6 Paths and selectors

Paths into arrays may select an element by a key, so a mapping can name one coverage or one limit:

```
coverages[coverage_code=HO_COV_A].limits[limit_type=per_occurrence].amount
vehicles[].vin
lob_parts[lob=cgl].premium
```

- `[]` means every element (one value per element, as in SPEC_02).
- `[key=value]` selects the element whose key equals the value; L1 creates the element if it does not exist.
- These paths are used by the crosswalk (§4.7), registry `canonical_path` (SPEC_02), `fideon:audit_rules` and varmap `field_paths`.

`printed_lines` is removed. Tables and labels it held go into `text_sections` (§4.6).

### 4.3 Core field rules

A field enters the core tier only if at least one of these holds:

- It is printed by **at least two carriers'** seeds in the developer's set, or
- It is a concept policy comparison needs (limits, deductibles, coverage grants and exclusions referenced by form, insured parties, dates, premium), even if only one seed prints it, or
- An existing L1 registry YAML for the LOB maps a carrier field to it and the field passes the first two tests on review (§4.7).

Constraints:

- **Budget:** 80–300 core leaf fields per LOB, counted over the envelope, the building blocks the LOB uses and its LOB block. Exceeding 300 requires a written justification in `<lob>.fields.md` and project-owner approval.
- **Names describe the domain concept**, never a carrier label. Field names inside the envelope and building blocks are fixed by the common model; LOB developers name only coverage codes and LOB-block fields.
- **Aliases:** each core field carries `"fideon:aliases"`, the printed labels observed across carriers (e.g. `["Dwelling", "Coverage A - Dwelling", "Cov A"]`). Aliases are collected from the seeds **and** from the captions in the L1 registry YAMLs for the LOB. They feed L1 mappings and the training prompt.
- **Repeating units are arrays of objects** using the building blocks (vehicles, drivers, locations, buildings, watercraft, scheduled items), never numbered fields (`vehicle_1_vin`).
- **Mandatory fields:** `fideon:mandatory_fields` lists the core fields every policy of the LOB must have (at minimum: carrier name, named insured, policy number, effective and expiration dates, at least one coverage with a limit).
- **Relationships** are declared in `fideon:audit_rules` (e.g. `"coverages[coverage_code=HO_COV_C].limits[limit_type=per_occurrence].amount <= coverages[coverage_code=HO_COV_A].limits[limit_type=per_occurrence].amount"`, `"premium.total == sum(premium.items[*].amount) ± 1"`, `"len(vehicles) == len(premium.items[unit_type=vehicle])"`). The generator (§7.5), the Audit Gate (SPEC_07) and the L1 plausibility gate all read these.

### 4.4 Leaf type

Every leaf is a SPEC_00 `FieldValue`:

```json
{"raw": "$2,100,000", "parsed": 2100000, "confidence": {"score": 1.0, "source": "deterministic"},
 "page_ref": [20], "flagged": false}
```

- `raw` is exactly as printed (one occurrence; the first by page order).
- `page_ref` lists every page where the value is printed. It must be non-empty whenever `raw` is non-null.
- `parsed` types:
  - money → number;
  - percentage → number (5% → 5);
  - **date → `MM/DD/YYYY`** (resolved D1);
  - boolean → `true`/`false`;
  - everything else → string.
- Absent fields are omitted, not written as null leaves. Absence is recorded in the side log (§8.3).

### 4.5 Overflow tier

```json
"additional_fields": [
  {"label": "Hurricane Shutters Credit", "value": {FieldValue}, "section_hint": "discounts", "page_ref": [3]}
]
```

Every printed value with no core home goes here. Each entry is also written to `unmapped.log` (§8.3). A label appearing in overflow across two or more carriers is a candidate for promotion to core in the next schema version.

### 4.6 Full-text tier — every word in the output

Policy comparison needs the complete wording, organised so that the same form and section can be aligned across two policies. `text_sections` is a map keyed by a stable section key:

```json
"text_sections": {
  "HO 00 03 05 11|SECTION I - EXCLUSIONS|A.2": {
    "form_number": "HO 00 03 05 11",
    "form_title": "Homeowners 3 - Special Form",
    "edition_date": "05/11",
    "form_type": "coverage_form",
    "section_path": ["SECTION I - PERILS INSURED AGAINST", "EXCLUSIONS", "A.2"],
    "section_title": "Earth Movement",
    "section_type": "prose",
    "raw_text": "…verbatim text…",
    "page_range": [12, 13]
  }
}
```

- **Key:** `<form_number or 'NOFORM'>|<top heading>|<sub-path>`, stable across carriers that use the same form.
- **`form_type`:** one of `declarations`, `schedule`, `coverage_form`, `endorsement`, `notice`, `application`, `correspondence`, `other`.
- **`section_type`:** one of `prose`, `table`, `label`, `header_footer`, `handwriting`, `stamp`. Tables are written as markdown tables in `raw_text`.
- **Every-word rule:** every word printed in the PDF appears in at least one of: a core field `raw`, an `additional_fields` entry, or a `text_sections.raw_text`. Values may appear both in a field and in the text; that duplication is expected.
  - **Digital PDFs:** all visible text layers count.
  - **Scanned PDFs:** the verified transcription (§5.3) counts.
- **`fideon:fsm_exclude: true`:** at inference the full text is assembled from MinerU output by the section builder, not generated by the model (SPEC_00). The training target excludes it (§10.2).

### 4.7 Alignment with the L1 carrier registry

The current L1 registry YAMLs (`config/document_registry/policy_check/<carrier_id>/<lob>.yaml`, SPEC_02) map every carrier field to a schema path through `canonical_path`. Replacing the schema would orphan those paths, so each developer keeps the registry aligned for their LOB.

1. **Inventory.** List every `canonical_path` used by the LOB's registry YAMLs, with carrier and caption.
2. **Crosswalk.** `<lob>.crosswalk.yaml` maps every old path to one of:
   - `new: <new path>` — same concept, new location;
   - `overflow: <label>` — kept as an `additional_fields` entry with that label;
   - `drop: <reason>` — not a value (e.g. a mis-mapped caption). Requires reviewer approval.

   ```yaml
   lob: homeowners
   from_schema: "1.6.0"
   to_schema: "1.0.0"   # new schema line
   paths:
     homeowners.section_i_property_coverages.coverage_a_dwelling_limit: {new: "coverages[coverage_code=HO_COV_A].limits[limit_type=per_occurrence].amount"}
     named_insured.primary_name: {new: named_insured.primary_name}
     document_type_detail.quote_proposal.quote_reference_number: {overflow: "Quote Reference Number"}
   ```

3. **Registry update.**
   - Rewrite `canonical_path` in every registry YAML of the LOB using the crosswalk. The tool is `tools/registry_migrate.py --lob <lob>`, which applies the crosswalk and refuses any path without an entry.
   - Overflow targets use `canonical_path: "additional_fields.<label>"` (SPEC_02 §5.1 "nothing is dropped" rule).
   - Array paths use `[]` notation as SPEC_02 already defines.
4. **Lint.** `python tools/registry_lint.py` must pass for every YAML of the LOB against the new schema. It fails on any `canonical_path` that does not resolve, and on any field missing `canonical_path`.
5. **Full text from L1.** L1 already emits `text_sections` (SPEC_02, SPEC_00). The registry's text-section extraction is updated to produce the §4.6 keys and attributes, using the shared section builder (§7.2) so L1 and L3 emit identical full-text structure.
6. **Seed cross-check (C14, §9).** For every seed whose carrier has a registry YAML, run L1 on the seed. Every core value L1 returns must equal the seed gold value at the same path. A difference means either the seed gold or the registry is wrong. The reviewer resolves it before sign-off.
7. **Ownership.** The LOB developer owns the crosswalk and the registry updates for that LOB (D8). Changes go through the existing CI registry checks (`ci-registry-ops.yml`). SPEC_19's YAML generator is updated to emit paths from the new schema (§15).

---

## 5. Seed golds

### 5.1 Inputs per developer

```
data/seeds/<lob>/<Carrier>/<doc_stem>.pdf
```

- Seeds are already de-identified (resolved D2). No further de-identification is needed before a Claude session. The Presidio scan of outputs (§11) still runs.
- Seeds may be **digital** (text layer) or **scanned** (image only). Step 0 of SPEC_01 (`detect_modality`) decides the modality; it is recorded in the seed gold as `document.modality`.

### 5.2 Artifacts per seed

```
data/seeds/<lob>/<Carrier>/<doc_stem>.gold.json      # canonical JSON of the seed (all three tiers)
data/seeds/<lob>/<Carrier>/<doc_stem>.varmap.json    # variation map (§6)
data/seeds/<lob>/<Carrier>/<doc_stem>.review.md      # reviewer sign-off (§5.4)
data/seeds/<lob>/<Carrier>/<doc_stem>.ocr.json       # scanned seeds only: verified word boxes (§5.3)
```

### 5.3 Drafting workflow (Claude Code)

**Digital seeds**

1. Extract every page's text by layer with PyMuPDF (`get_text("rawdict")`), keeping page, bounding box, font and size.
2. Claude drafts the gold from that text **and** the rendered page images. Every core value carries its `page_ref`, measured from the text layer, not guessed.

**Scanned seeds**

1. Run the production MinerU configuration (SPEC_05) and keep word-level boxes. Also render each page at 300 DPI.
2. Claude reviews the OCR against the page images and drafts a corrected transcription (`ocr.json`: page, word, box, confidence, corrected flag). Handwriting, signatures and stamps are marked with their box and `section_type`.
3. Claude drafts the gold from the corrected transcription and the images. Every core value carries its `page_ref` and the box of each occurrence.

**All seeds**

1. Build `text_sections` with the deterministic section builder (§7.2), from the text layer (digital) or the corrected transcription (scanned). Not by Claude, so the same code runs at inference.
2. Run `seed check` (§9, seed mode). Fix all errors.
3. Claude drafts the variation map (§6), including structures (§6.4).
4. A reviewer verifies (§5.4).

### 5.4 Review and sign-off

The reviewer checks, against the rendered PDF:

- every core field value and page reference;
- that no printed value that belongs in a core field sits in overflow instead;
- every equivalence group, relationship and structure in the variation map;
- the protected list (§6.3);
- for scanned seeds, the corrected transcription on a sample of at least 10% of pages plus every page holding a core value;
- every C14 difference with the L1 registry (§4.7).

`review.md` records the reviewer's name, date, schema version, modality, and the counts of core fields filled, overflow entries and corrections made. The generator refuses a seed without a `review.md` whose schema version matches the current schema.

---

## 6. Variation map (`varmap.json`)

### 6.1 Purpose

The varmap states, for one seed, exactly what may change and how. It replaces regex inference. The generator acts only on what the varmap lists.

### 6.2 Value groups and relations

```json
{
  "seed": "chubb_home",
  "schema_version": "1.0.0",
  "modality": "native_pdf",
  "groups": [
    {
      "id": "g_policy_number",
      "kind": "policy_number",
      "field_paths": ["policy.policy_number"],
      "value": "268-22-38-40",
      "occurrences": [
        {"page": 1, "text": "268-22-38-40", "bbox": [72.0, 140.2, 151.8, 151.0]},
        {"page": 4, "text": "268-22-38-40A", "bbox": [410.5, 62.0, 494.1, 72.8], "form": "suffix", "suffix": "A"}
      ]
    },
    {
      "id": "g_insured_name",
      "kind": "person_name",
      "field_paths": ["named_insured.primary_name"],
      "value": "MEGHAN & PETER FUENTES",
      "parts": {"given": ["MEGHAN", "PETER"], "surname": "FUENTES"},
      "occurrences": [
        {"page": 1, "text": "MEGHAN & PETER FUENTES", "bbox": []},
        {"page": 7, "text": "Fuentes, Meghan", "bbox": [], "form": "surname_first"}
      ]
    }
  ],
  "relations": [
    {"expr": "g_cov_c == 0.5 * g_cov_a", "source": "printed"},
    {"expr": "g_expiration == add_months(g_effective, 12)", "source": "printed"}
  ],
  "structures": [],
  "protected": ["Chubb Insurance Company of New Jersey", "https://www2.Chubb.com/us-en/privacy.aspx", "Form no. 0800029"],
  "notes": "Form print timestamps (e.g. '10/07/20 8:41:41') are protected: they identify the form edition."
}
```

Every occurrence carries a `bbox` (PDF points for digital seeds, pixels at 300 DPI for scanned seeds). Empty boxes in the example above are filled by the drafting tool, not by hand.

### 6.3 Kinds and protected content

**Kinds** (extendable per LOB in the LOB package):

| Category | Kinds |
|---|---|
| Parties | `person_name`, `org_name` |
| Address | `address_street`, `address_city_state_zip` (one group: city, state, ZIP and county move together) |
| Identifiers | `policy_number`, `identifier`, `vin`, `hin`, `serial`, `license_number`, `loan_number` |
| Contact | `phone`, `email`, `url` |
| Dates | `date`, `date_of_birth` |
| Numbers | `money`, `percentage`, `count`, `year` (model year, year built) |
| Text values | `enum_text` (a value from a fixed set, e.g. construction type) |

**Always protected:**
- carrier names, addresses, phones and URLs;
- form numbers, edition dates and form print timestamps;
- statutory and boilerplate wording;
- page numbers;
- signatures, handwriting and stamps on scanned seeds;
- anything the reviewer adds to `protected`.

### 6.4 Structures (repeating items and optional forms)

Structural variation (resolved D6: in scope) changes how many repeating items a twin has. Each structure is declared in the varmap; the generator never discovers structure by itself.

```json
"structures": [
  {
    "id": "s_vehicles",
    "kind": "table_rows",
    "schema_array": "vehicles",
    "region": {"page": 8, "bbox": [36, 210, 576, 640]},
    "rows": [
      {"id": "row_veh1", "page": 8, "bbox": [36, 230, 576, 262], "groups": ["g_veh1_vin", "g_veh1_amount"], "array_index": 0},
      {"id": "row_veh2", "page": 8, "bbox": [36, 262, 576, 294], "groups": ["g_veh2_vin", "g_veh2_amount"], "array_index": 1}
    ],
    "row_template": "row_veh1",
    "min": 1, "max": 6,
    "linked": ["s_vehicle_premium_rows"]
  },
  {
    "id": "s_forms_schedule",
    "kind": "table_rows",
    "schema_array": "forms_and_endorsements",
    "region": {"page": 2, "bbox": [36, 400, 576, 560]},
    "rows": [{"id": "row_form_HO0490", "page": 2, "bbox": [36, 420, 576, 434], "groups": [], "array_index": 5},
             {"id": "row_form_HO0416", "page": 2, "bbox": [36, 434, 576, 448], "groups": [], "array_index": 6}]
  },
  {
    "id": "s_optional_endorsements",
    "kind": "optional_forms",
    "schema_array": "forms_and_endorsements",
    "forms": {"HO 04 90": [41, 42], "HO 04 16": [43]},
    "schedule_rows": {"HO 04 90": "row_form_HO0490", "HO 04 16": "row_form_HO0416"},
    "library": "data/seeds/<lob>/_form_library/",
    "min_kept": 0
  }
]
```

- `rows` are objects; `array_index` ties a row to its element of `schema_array`; `row_template` names the row copied for additions.
- A table that serves as a forms schedule changes only together with its optional forms.
- Totals over a table are written as relations with `column(<structure>, <j>)`, the j-th group of every row the twin keeps or adds, e.g. `{"expr": "g_total_premium == sum(column(s_vehicle_premium_rows, 0))", "derive": "g_total_premium"}`.
- Field names are fixed by `synth_core/models.py`; unknown keys are rejected at load time.
- Not yet in `synth_core` 0.1.0: rows of one structure spread over several pages, continuation pages (added rows are capped at the region), and insertion of library forms (removal works). The seed check reports a structure whose rows span several pages.

| Structure kind | What changes | Rules |
|---|---|---|
| `table_rows` | Rows added or removed in a declared table region (vehicles, drivers, locations, scheduled items, interested parties, claims) | A new row is drawn from the row template with fresh values for its groups. Content below the row moves up or down by the row height **inside the declared region only**. If added rows exceed the region, a continuation page is created from the declared `continuation_template` when one exists; otherwise the row count is capped. Linked structures (e.g. per-vehicle premium rows on another page) change together. |
| `optional_forms` | Optional endorsements removed or added | Removing a form deletes its schedule row and its pages. Adding a form inserts its schedule row and its pages from the LOB form library (forms taken from other verified seeds of the same LOB, with their verified `text_sections`). Page numbers, page references and printed "Page x of y" are renumbered. |
| `list_items` | Items in a bulleted or numbered list of values (e.g. discounts applied) | Same rules as table rows, within the list region. |

After any structural change:
- the gold arrays, page references and `text_sections` are rebuilt;
- totals and counts are recalculated through relations (§7.5);
- check C15 (§9) verifies integrity.

---

## 7. LOB generator

### 7.1 Package layout

```
fideon/synth_core/                  # shared; owned jointly; changes by pull request with tests
  pdf_rewrite.py                    # digital: text-layer rewrite in place (port of fideon_synth.digital.textlayer/contentstream, bug-fixed)
  image_rewrite.py                  # scanned: erase and redraw values in page images (§7.8)
  structure_ops.py                  # row insert/remove, region shift, page insert/delete, renumbering (§6.4)
  fonts.py                          # glyph availability, width measurement, fallback; font matching for scans
  sections.py                       # deterministic section builder for text_sections (§7.2); also used at inference and by L1
  scan_effects.py                   # scan look: skew, blur, noise, JPEG (§7.9)
  checks.py                         # every check in §9
  varmap.py                         # varmap load/validate (Pydantic models)
  render_diff.py                    # pixel diff outside changed boxes
  sidelog.py                        # unmapped.log / absent.log writers
  # also delivered: models.py, paths.py, relations.py, forms.py, values_base.py, plugin.py, engine.py,
  # runconfig.py, ocr.py, tools.py, cli_lob.py — full list and status in synth_core/README.md
fideon/synth/<lob>/                 # one per developer
  __main__.py                       # CLI (§7.10)
  values.py                         # realistic LOB value generators
  relations.py                      # LOB relationship rules beyond fideon:audit_rules
  profile.yaml                      # variation ranges, structure ranges, mandatory varied kinds
  tests/
tools/registry_migrate.py           # applies <lob>.crosswalk.yaml to registry YAMLs (§4.7)
```

Each developer writes the LOB generator. The PDF and image rewrite, structure operations, section builder and checks are shared (D3) so that document-level bugs are fixed once. The parts of `fideon_synth` worth porting are `textlayer`, `contentstream`, the leak/fidelity/page-ref checks and the run manifest. The generic `generator.py`/`structure.py` engine and its regex value inference are not ported.

### 7.2 Section builder (`synth_core/sections.py`)

- **Input:**
  - text with layout (PyMuPDF for digital PDFs);
  - corrected word boxes for scanned seeds;
  - MinerU output at inference.
- **Output:** `text_sections` per §4.6.
- **How:**
  1. Detect form boundaries from form-number footers/headers.
  2. Detect headings by font weight, size and numbering patterns.
  3. Join lines into paragraphs.
  4. Render tables as markdown.
- **Determinism:** same input, same output.
- **Twins:**
  - digital twins: `text_sections` are rebuilt from the twin PDF;
  - scanned twins: rebuilt from the seed's corrected transcription after applying the twin's occurrence-level replacements and structure operations. Each replacement is tied to a varmap occurrence box, never a free-text search.

### 7.3 Value generation (`values.py`, `profile.yaml`)

- **Realistic per LOB.** For example:
  - homeowners: dwelling limits by region and construction, coverage C as a percentage of A, deductible options;
  - personal auto: VINs with valid check digit and a make/model/year consistent with the WMI, driver dates of birth giving ages 16–90, licence formats by state;
  - watercraft: HIN format.
- **No single scale factor.** Each money value is drawn independently within its profile range; then relations (§7.5) are re-derived.
- **Places:** city, state, ZIP and county drawn together from a real geographic table covering the states seen in the LOB's seeds. Phone area codes match the drawn state.
- **Names and organisations:** invented. They must not match any name in the seed set.
- **Dates (all `MM/DD/YYYY` in `parsed`):**
  - all policy dates shift by the same whole number of months, with the term length preserved;
  - dates of birth shift independently within age bounds;
  - protected dates never shift.
- **Structure counts:** drawn per structure within `min`/`max` and the distribution in `profile.yaml` (e.g. personal auto vehicles: 1 → 30%, 2 → 40%, 3 → 20%, 4+ → 10%).
- **Determinism:** `(run_seed, seed_stem, twin_index)` fully determines the output.

### 7.4 Minimum variation per twin

These must change in every twin unless the reviewer has listed them as protected:

- every `person_name` and `org_name` group except carrier;
- named insured address;
- policy number (all occurrences);
- effective and expiration dates;
- every `vin`, `hin` or `serial`;
- at least 60% of `money` groups (`profile.yaml` may raise this).

Structural changes are additionally required in at least 40% of a seed's twins when the varmap declares any structure (D10; `profile.yaml` may raise this).

A twin below the minimum is discarded. The manifest records the variation rate (changed groups ÷ variable groups) and the structure changes made.

### 7.5 Relations

1. Every relation in the varmap and in `fideon:audit_rules` is re-evaluated after drawing and after structure operations.
2. Dependent values are computed, not drawn: totals, counts, percentages of other limits, term-derived dates.
3. If a relation cannot be satisfied within the profile ranges, the twin is redrawn, up to 10 attempts, then reported.

### 7.6 Text fitting (digital)

1. Replacement text is measured with the original font and size.
2. If it is wider than the original box plus 2% tolerance, the generator first draws a shorter value of the same kind.
3. Failing that, it reduces the font size by at most 10%.
4. Failing that, the twin is discarded.

Missing glyphs in a subset font trigger a fallback to a metrically compatible font. The fallback is recorded in the manifest.

### 7.7 Digital twins

The digital-seed path is the in-place text-layer rewrite (`pdf_rewrite.py`) plus structure operations on content streams. Output is a digital PDF with the seed's fonts, layout and hidden layers.

### 7.8 Scanned twins (resolved D7: in scope)

Scanned seeds produce scanned twins, so the scanned extraction path is trained on scanned input. For each changed occurrence box:

1. **Erase.**
   - Fill the box with the local background, estimated from a margin ring around it.
   - Use inpainting (OpenCV Telea) where the box overlaps lines or shading.
   - Table rules crossing the box are redrawn from detected line positions.
2. **Redraw.** Render the new text with:
   - a font matched from a library of common print fonts by stroke width, x-height and character widths measured from the original glyphs;
   - size matched to the original cap height;
   - ink colour sampled from the original strokes.
3. **Blend.** Apply the page's measured blur and noise to the redrawn patch so it does not stand out.

Rules:
- Handwriting, signatures and stamps are never altered (always protected).
- Structure operations on scans move image bands inside declared regions; page insertion and deletion work as for digital twins.
- Output is an image-only PDF at the seed's resolution, with no text layer, as the seed.

### 7.9 Render modes

Each twin is produced in one of two modes, set per run:

| Mode | Source | Output |
|---|---|---|
| `native` | Digital or scanned seed | Same modality as the seed |
| `scanned_from_digital` | Digital seed | The digital twin rasterised at 200–300 DPI, then scan effects applied (skew ±1.5°, blur, noise, JPEG quality 60–90, optional fax/photocopy profile), saved image-only |

`scanned_from_digital` gives scanned training data for carriers whose seeds are digital. The gold is the digital twin's gold with `document.modality = "scanned_pdf"` and a scan-effects record in provenance.

### 7.10 Run config and CLI

Every generation run is described by one **run config** file, committed to Git **before** the run. Its commit hash is the single record of how a batch was made, and is entered in the Tracker (`SYNTHETIC_DATA_PROCESS.md`, Phase A).

**Location and name:** `config/synth/<batch_id>.yaml`, e.g. `config/synth/SYN-PDF-PC-HO-001.yaml`. A new batch (new seeds, new count, new profile, regeneration after a failed gate) gets a new batch ID and a new file. Committed run configs are never edited.

```yaml
batch_id: SYN-PDF-PC-HO-001
lob: homeowners
schema:
  path: config/canonical_schema/policy_check/homeowners.json
  version: "1.0.0"                       # must equal fideon:schema_version in the file
generator:
  package: fideon.synth.homeowners
  version: "0.3.0"                       # package __version__ at the commit used
  synth_core_version: "0.1.0"
profile: fideon/synth/homeowners/profile.yaml
run_seed: 1
render_modes:
  native: {count_per_seed: 20}
  scanned_from_digital: {count_per_seed: 20, applies_to: digital_seeds}
seeds:                                   # every seed in the batch, with its sign-off
  - path: data/seeds/homeowners/Chubb/chubb_home.pdf
    modality: native_pdf
    gold: data/seeds/homeowners/Chubb/chubb_home.gold.json
    varmap: data/seeds/homeowners/Chubb/chubb_home.varmap.json
    review: {reviewer: "<name>", date: "10/05/2026", schema_version: "1.0.0"}
  - path: data/seeds/homeowners/Dryden Mutual/dryden_mutual_home_dec.pdf
    modality: scanned_pdf
    gold: data/seeds/homeowners/Dryden Mutual/dryden_mutual_home_dec.gold.json
    varmap: data/seeds/homeowners/Dryden Mutual/dryden_mutual_home_dec.varmap.json
    ocr: data/seeds/homeowners/Dryden Mutual/dryden_mutual_home_dec.ocr.json
    review: {reviewer: "<name>", date: "10/06/2026", schema_version: "1.0.0"}
output: data/twins/homeowners/SYN-PDF-PC-HO-001
splits:                                  # by seed (§10.3); filled when the batch is packaged
  holdout_carriers: ["Dryden Mutual"]
```

**Rules:**
- The generator reads only the run config. Every value above that is also passed on the command line causes an error.
- Before generating, it verifies that:
  - the schema file's version equals `schema.version`;
  - each `review` matches the seed's `review.md`;
  - the profile file exists;
  - the run config itself is committed with no local changes (`git status --porcelain` for the file is empty).

  Any mismatch stops the run.
- The run config's path and Git blob hash are written into every twin's provenance (§8.1) and into `manifest.csv`.

**CLI:**

```
python -m fideon.synth.<lob> --config config/synth/<batch_id>.yaml [--only <seed_stem>] [--check-seeds]
```

- `--check-seeds` runs seed mode of §9 on the seeds listed in the run config and stops.
- `--only` limits a run to some seeds for debugging. Output of an `--only` run is never packaged for training.
- No absolute paths in code. The schema root comes from the run config.

---

## 8. Outputs

### 8.1 Per twin

```
<out>/pdfs/<lob>__<carrier>__<seed_stem>__twin_<nnn>.pdf
<out>/gold/<lob>__<carrier>__<seed_stem>__twin_<nnn>.json
```

The gold holds all three tiers plus:

```json
"fideon:provenance": {"generator": "fideon.synth.<lob>", "generator_version": "x.y.z", "batch_id": "...",
  "run_config": "config/synth/<batch_id>.yaml", "run_config_git_blob": "<sha1>",
  "seed": "<carrier>/<seed_stem>", "seed_review": "<reviewer>, <date>", "schema_version": "1.0.0",
  "seed_modality": "scanned_pdf", "twin_modality": "scanned_pdf", "mode": "native",
  "twin_index": 1, "run_seed": 1, "attempt": 0, "variation_rate": 0.87,
  "structure_changes": [{"structure": "s_vehicles", "from": 4, "to": 2}], "scan_effects": null,
  "synthetic": true}
```

### 8.2 Per run

- `manifest.csv`: one row per twin, holding seed, carrier, modality, mode, twin index, variation rate, structure changes, font fallbacks, attempts, seconds, pass/fail, problems.
- `report.json`: problems per twin and per seed.

### 8.3 Side logs (not inside the gold)

| Log | Meaning |
|---|---|
| `unmapped.log` | Printed values placed in `additional_fields` (no core home): label, value, page, carrier. Input to schema promotion. |
| `absent.log` | Core fields the seed does not fill. For review only. |

---

## 9. Checks

All checks are blocking. **Seed mode** runs on seed + seed gold. **Twin mode** runs on every twin. For scanned twins, "printed" means "read back by OCR of the twin" with the thresholds given.

| # | Check | Seed | Twin | Rule |
|---|---|---|---|---|
| C1 | Schema | ✓ | ✓ | Validates against `<lob>.json` |
| C2 | Page references | ✓ | ✓ | Each core and overflow value is printed on every page in `page_ref`, and on no other page |
| C3 | Core coverage | ✓ | ✓ | Every `fideon:mandatory_fields` entry filled. Every core field filled in the seed is filled in the twin (except arrays reduced by a structure operation). Seed mode: each value-like token (money, date, policy-number shape, VIN, phone) printed in the PDF maps to a core field or an overflow entry; matches inside `text_sections` do **not** count |
| C4 | Every word | ✓ | ✓ | Digital: ≥ 99.5% of normalised word tokens and 100% of numeric tokens covered by the three tiers. Scanned: same against the corrected transcription; twin OCR read-back agrees with the gold full text at ≥ 98% word level (D9) |
| C5 | Leak | — | ✓ | No original value of a varied group is printed (digital) or read back by OCR (scanned) anywhere in the twin |
| C6 | Group consistency | — | ✓ | Every occurrence of a group prints its one new value in its declared form (suffixes, case and format kept) |
| C7 | Gold = PDF | — | ✓ | Every core and overflow `raw` in the twin gold is printed on its `page_ref` pages. Scanned: OCR read-back of each changed occurrence matches exactly, or the twin is discarded |
| C8 | Relations | ✓ | ✓ | All varmap relations and `fideon:audit_rules` hold |
| C9 | Protected | — | ✓ | Every protected string is printed exactly as in the seed; protected image regions are pixel-identical |
| C10 | Render | — | ✓ | Outside changed boxes and moved regions, the rendered page differs from the seed by < 0.5% of pixels. No text overlaps. No missing glyphs. Scanned: each redrawn patch passes an edge-artefact test (no visible box boundary) |
| C11 | Plausibility | ✓ | ✓ | VIN check digit valid; ZIP/city/state consistent; area code matches state; dates valid and ordered; ages within bounds |
| C12 | Variation | — | ✓ | §7.4 minimums met |
| C13 | Sign-off | ✓ | — | `review.md` present and schema version matches |
| C14 | L1 agreement | ✓ | ✓ | Where the carrier has a registry YAML: L1 output on the seed equals the seed gold at every mapped core path (seed mode). On digital twins, L1 output equals the twin gold at every mapped path (twin mode). Any difference blocks |
| C15 | Structure integrity | — | ✓ | Array lengths in the gold equal printed row counts; linked structures have equal counts; removed forms have no pages, schedule rows or `text_sections` left; inserted forms have all three; "Page x of y" and page references are consistent after renumbering |

---

## 10. Packaging for SFT

### 10.1 User content

1. Run each twin PDF through the production MinerU configuration (SPEC_05), not the PDF text layer. Training input must match inference input.
2. Chunk with SPEC_04 using the same token budget as inference, counting image tokens where page images are attached.
3. Attach page images per chunk exactly as SPEC_06 does at inference.

### 10.2 Assistant content (training target)

- Core fields and `additional_fields` only, restricted to the chunk's pages.
- `text_sections` is **excluded** from the target. It is attached by the section builder at inference, which keeps the target within the token budget and keeps the model from regenerating text it can copy.
- Dates in the target use `MM/DD/YYYY` in `parsed`.

### 10.3 Splits and mix

- **Split by seed:** all twins of one seed (both render modes) go to the same split (train / val / test). Twins of a seed never span splits.
- **Carrier hold-out:** at least one carrier per LOB is held out entirely from train and val. It measures accuracy on unseen carriers.
- **Cap per seed:** at most 20 twins per seed per render mode in train, so that no single template dominates.
- **Modality balance:** the share of scanned twins in train matches the expected production share per LOB (set in `profile.yaml`).
- **Mix:** verified real golds remain part of every adapter's training set. The ratio is the per-adapter, per-LOB `synthetic_fraction` parameter defined in the SPEC_09 amendment (D4).

### 10.4 Format

SPEC_16 §2 JSONL, with `"source": "synthetic"`, `adapter`, `layout_family`, `modality`, plus `seed`, `carrier`, `twin_index`.

---

## 11. Process

Batch IDs, Presidio scan, DVC tracking and the quality gate follow `SYNTHETIC_DATA_PROCESS.md`. LOB codes for personal lines are added there: `HO` homeowners, `DF` dwelling fire, `PA` personal auto, `MC` motorcycle, `RV` recreational vehicle, `PU` personal umbrella, `OM` ocean marine.

---

## 12. Deliverables per developer (definition of done)

1. **Schema:** `<lob>.json` composing the current common model, `<lob>.coverage_codes.yaml`, `<lob>.fields.md`, `CHANGELOG_<lob>.md`; core leaf count within budget; every core field and coverage code with aliases; change requests raised for any concept the common model lacks.
2. **Registry alignment:** `<lob>.crosswalk.yaml`; every registry YAML of the LOB migrated with `registry_migrate.py`; `registry_lint.py` clean; L1 text sections in the §4.6 structure.
3. **Seeds:** for every assigned seed, `gold.json`, `varmap.json` (with structures where the seed has repeating items or optional forms), `review.md`, and `ocr.json` for scanned seeds, all passing seed mode of §9.
4. **Generator:** `fideon/synth/<lob>/` package with tests (§13).
5. **Runs:**
   - a committed run config `config/synth/<batch_id>.yaml` listing every assigned seed (§7.10);
   - 20 `native` twins per seed;
   - 20 `scanned_from_digital` twins per digital seed;
   - 100% of written twins pass §9;
   - `manifest.csv`, `report.json` and `unmapped.log` delivered.
6. **Note:**
   - variation rate and structure-change rate per seed;
   - font fallbacks;
   - seeds that could not be varied and why;
   - C14 differences found and how they were resolved;
   - proposed schema promotions from `unmapped.log`.

---

## 13. Tests (minimum)

| Test | Asserts |
|---|---|
| Suffix group | A value printed as `X` and `XA` receives one replacement, printed as `Y` and `YA` |
| Name forms | "Carol B Mitchell" and "Mitchell, Carol B" change together |
| Protected URL | Carrier URL unchanged after 20 twins |
| Relation | Coverage C stays at its declared percentage of coverage A in every twin |
| Term | Expiration − effective is unchanged in every twin; all `parsed` dates are `MM/DD/YYYY` |
| Determinism | Same run seed → byte-identical gold and identical page text |
| Leak | An injected unchanged occurrence fails C5 |
| Every word | Removing one paragraph from `text_sections` fails C4 |
| Core coverage | Moving a core value into `text_sections` only fails C3 |
| Fit | An over-long replacement is redrawn or the twin is discarded; never overflows |
| Sign-off | A seed without `review.md` is refused |
| Row removal | Removing one vehicle removes its row, its linked premium row, its gold array entry, and recalculates total premium |
| Row addition | An added driver row prints fresh values, moves following content inside the region only, and appears in the gold |
| Form removal | Removing an optional endorsement removes its schedule row, its pages and its `text_sections`; "Page x of y" and every `page_ref` are renumbered |
| Form insertion | An inserted library form brings its pages and verified `text_sections`; schedule and gold agree |
| Scanned redraw | A changed value on a scanned page reads back exactly by OCR; the original value does not; pixels outside the box are unchanged |
| Scanned protected | A signature region is pixel-identical after 20 twins |
| Scan effects | `scanned_from_digital` output has no text layer and routes as `scanned_pdf` through SPEC_01 Step 0 |
| Registry migration | A registry YAML with a `canonical_path` absent from the crosswalk makes `registry_migrate.py` fail |
| L1 agreement | A deliberately wrong seed value at a registry-mapped path fails C14 |

---

## 14. Instructions for Claude Code sessions

### 14.1 Phases

Each phase ends with a commit and its exit criterion. A phase does not start until the previous one has met its exit criterion, except where noted.

| Phase | Work | Output | Exit criterion |
|---|---|---|---|
| 0 | Set up: place seeds under `data/seeds/<lob>/<Carrier>/`; add the LOB `CLAUDE.md` (§14.3); run SPEC_01 Step 0 modality detection on every seed | `data/seeds/<lob>/seed_inventory.csv` (carrier, file, pages, modality) | Every seed listed with modality |
| 1 | Registry inventory (§4.7 step 1) and seed reading: extract text (digital) or MinerU OCR (scanned) for every seed | `registry_inventory_<lob>.csv`; per-seed text dumps (not committed if large) | Every registry `canonical_path` listed |
| 2 | Canonical schema (§4): compose the common model; Claude proposes coverage codes, the building blocks used and LOB-block fields from all seeds and registry captions together, never from one seed at a time | `<lob>.json`, `<lob>.coverage_codes.yaml`, `<lob>.fields.md`, `CHANGELOG_<lob>.md`; common-model change requests | Validates as JSON Schema; core count within budget; project owner approves the field list; common-model owner approves any change request |
| 3 | Crosswalk and registry migration (§4.7) | `<lob>.crosswalk.yaml`; migrated YAMLs | `registry_lint.py` clean |
| 4 | Seed golds, one seed per session (§5) | `*.gold.json`, `*.ocr.json` (scanned) | Seed mode of §9 passes, except C13 (sign-off pending) |
| 5 | Variation maps and structures, one seed per session (§6) | `*.varmap.json` | Validates against the varmap Pydantic model |
| 6 | Review (§5.4) | `*.review.md` per seed | Every seed signed off; C13 and C14 pass |
| 7 | LOB generator on `synth_core` (§7). May start after Phase 2 in parallel with Phases 4–6, using one signed seed as the first fixture | `fideon/synth/<lob>/` with tests (§13) | All §13 tests pass |
| 8 | Run config and generation (§7.10, §8) | `config/synth/<batch_id>.yaml` committed; twins, golds, manifest, report, side logs | 100% of written twins pass §9; minimums of §7.4 met |
| 9 | SFT packaging (§10) and hand-over note (§12 item 6) | JSONL per SPEC_16 §2; note | Presidio scan clean; DVC tracked per `SYNTHETIC_DATA_PROCESS.md` |

### 14.2 Rules for every session

1. **One seed per session** in Phases 4–6, so the session's context holds one document at a time. The schema, `fields.md` and `CLAUDE.md` are loaded at the start of every session.
2. **Values:**
   - Never invent a value in a seed gold. A value not visible in the PDF is left absent.
   - Uncertain readings, especially on scanned seeds, are marked `"flagged": true` and listed in `review.md` for the reviewer.
3. **Code rules:**
   - Rebuild `text_sections` from the twin (digital) or from occurrence-level replacements on the corrected transcription (scanned). Never by free-text search-and-replace.
   - No hardcoded paths or carrier-specific branches in `synth_core`. LOB packages may hold LOB rules; carrier-specific rules belong in the varmap.
4. **Discuss before changing:**
   - Changes to `synth_core` go through a pull request with tests, reviewed by another developer.
   - Ambiguities in this spec are raised with the project owner rather than resolved silently.
5. **Schema changes after Phase 2.** A core field found missing while drafting a later seed is added as a minor schema version. Already-signed seeds are re-checked with seed mode of §9: those that print the new field are re-opened and re-signed. The crosswalk and registry YAMLs are updated in the same commit.
6. **Commit per phase step,** with messages naming the phase and seed (e.g. `phase4: gold chubb_home`). Seeds, golds, varmaps and reviews are committed through DVC pointers as `SYNTHETIC_DATA_PROCESS.md` requires; code, schemas, crosswalks and run configs are committed to Git directly.

### 14.3 `CLAUDE.md` for the LOB workspace

Each developer places this file at the repository root of the LOB workspace, with the placeholders filled. Claude Code reads it at the start of every session.

```markdown
# Fideon synthetic data — LOB: <lob>

Authoritative spec: VLM Pipeline - Implementation Specs/SPEC_21_LOB_Canonical_Schema_and_Synthetic_Twin_Generation.md
Read SPEC_21 §3 (principles), §4 (schema), §14 (phases and rules) before any work.

LOB: <lob>   LOB code: <code>   Seed reviewer: <name>
Seeds: data/seeds/<lob>/   Schema: config/canonical_schema/policy_check/<lob>.json
Generator: fideon/synth/<lob>/   Shared core: fideon/synth_core/ (changes by pull request only)

Current phase: <0-9>   Current seed: <carrier>/<stem>

Hard rules:
- Never invent a value. A value not visible in the PDF is absent; uncertain readings are flagged.
- Every core value carries page_ref measured from the text layer or OCR boxes.
- Parsed dates are MM/DD/YYYY.
- text_sections are built by synth_core/sections.py, never written by hand or by the model.
- The generator acts only on what the varmap lists; no regex inference of values.
- No hardcoded paths. No carrier-specific code in synth_core.
- A twin that fails any SPEC_21 §9 check is discarded, not repaired by hand.
- Generation runs only from a committed config/synth/<batch_id>.yaml.
- Technical documents and code comments use impersonal language (no we/our/you/your).
- Field names in the envelope and building blocks come from common_model.json; never rename or add them in the LOB schema. Raise a change request instead.
- Every limit lives in coverages[].limits[]; no limit in the LOB block.
- Ask the developer before changing the schema, the crosswalk, or anything in synth_core.
```

---

## 15. Impact on existing documents

| Document | Status | Action |
|---|---|---|
| SPEC_16 §4 (Faker generators, `fideon/data_gen/` layout) and §5 (CLI) | **Obsolete** for policy_check LOBs — marked in SPEC_16 | §4.2 / §4.3 retained for the `loss_run` task adapter (D5) |
| SPEC_16 §2 (SFT JSONL format) | **Amended 3 Oct 2026** | Assistant content excludes `text_sections`; add `modality`, `seed`, `carrier`, `twin_index` |
| SPEC_16 §7 (edge-case coverage) | **Amended 3 Oct 2026** | Express coverage in seeds, carriers, modalities and structure ranges per LOB |
| `fideon_synth` package (generic `synthesize` engine, `seedkit auto`) | **Obsolete** as production generator | Port the rewrite engine and checks into `synth_core`; archive the rest |
| Seed golds produced by `seedkit auto` | **Obsolete** | Not usable as gold; may serve as drafts only |
| `config/canonical_schema/policy_check/<lob>.json` (current, e.g. homeowners v1.6.0) | **Obsolete** | Replaced by per-LOB schemas under §4, with a crosswalk (§4.7) |
| `absent_fields.log` semantics | **Obsolete** | Replaced by `unmapped.log` and `absent.log` (§8.3) |
| SPEC_00 | **Amended 3 Oct 2026** | Extend `TextSection` (§4.6 fields, including `handwriting` and `stamp` section types); remove `printed_lines`; add `fideon:tier`, `fideon:aliases`; fix parsed date format to `MM/DD/YYYY` |
| SPEC_02 | **Amended 3 Oct 2026** | `canonical_path` supports `[key=value]` selectors (§4.2.6); values follow the new schemas via the crosswalk; text-section extraction uses the shared section builder and §4.6 keys; registry lint runs against the new schemas |
| SPEC_06 (package routing) | **Amended 3 Oct 2026** | `PerLOBCanonical` is the subset of the canonical JSON whose `part` equals the LOB's `lob_parts[].part_id` (§4.2.3) |
| SPEC_19 | **Amended 3 Oct 2026** | YAML generator emits paths from the new schemas and uses `fideon:aliases` to match captions |
| SPEC_07 | **Amended 3 Oct 2026** | Required fields read from `fideon:mandatory_fields`; relations from `fideon:audit_rules` |
| SPEC_09 | **Amended 3 Oct 2026** | Training target = core + overflow; split by seed; carrier hold-out; modality balance; real/synthetic mix |
| SPEC_17, SPEC_20 | **Amended 3 Oct 2026** | Field F1 on core tier; separate overflow metric; report carrier hold-out and scanned results separately; full-text coverage metric (C4) |
| SPEC_18 | **Amended 3 Oct 2026** | Promotion of recurring `unmapped.log` labels into core; registry-miss events reference new schema paths |
| `SYNTHETIC_DATA_PROCESS.md` | **Amended 3 Oct 2026** | Personal-lines LOB codes (§11) |

---

## 16. Decisions (D1–D10 resolved 3 October 2026; D11 open)

| ID | Decision |
|---|---|
| D1 | Parsed dates use `MM/DD/YYYY` |
| D2 | Seeds are already de-identified; no further de-identification before Claude sessions |
| D3 | Shared `fideon/synth_core` (rewrite, structure operations, section builder, checks) plus one LOB-specific generator per developer |
| D4 | Real : synthetic ratio is the per-adapter, per-LOB `synthetic_fraction` training parameter (SPEC_09 amendment); verified real golds always included |
| D5 | Loss-run synthetic generation (SPEC_16 §4.2–4.3) continues unchanged and is outside SPEC_21 |
| D6 | Structural variation (repeating rows, optional forms) is in scope (§6.4) |
| D7 | Scanned seeds are in scope (§5.3, §7.8) |
| D8 | The LOB developer owns the crosswalk and registry YAML migration for that LOB |
| D9 | Scanned twins: OCR read-back agrees with the gold full text at ≥ 98% word level; changed values read back exactly |
| D10 | When a varmap declares structures, at least 40% of that seed's twins carry a structural change |
| D11 | **Open.** Scanned read-back (C4, C7). `synth_core` 0.1.0 implements D9 literally (`ocr_readback: exact`) and, as an option, `seed_baseline`: a twin may not read back worse than the seed itself does with the same OCR engine. Tests showed Tesseract misreading some seed values (for example a leading `1`), so the literal rule rejects every twin of such a seed. Decision needed on which rule §9 states |
