# Change requests - gl (to the common-model owner and the training-side owner)

Raised while building the gl SPEC_21 schema 1.0.0 (08 October 2026). Nothing here is applied to the common model: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields`, in a gl coverage code or in this repository's section map as named, until a request is accepted. Labels only; no document values.

**To send to the common-model owner now (8 October 2026), before the property_pkg team drafts its schemas**
(SPEC_21 v0.9 4.2.7: within 1.x the owner releases additive changes in batches; HANDOFF_lob_local_blocks_parallel_families.md):

| Request | Change to the common model or its files | Additive? | Prompt schemas that change |
|---|---|---|---|
| CR-GL-02 | `RatingExposure.rating_component` (enum) | yes (optional property) | gl only |
| CR-GL-12 | `carrier.financial_rating`, `premium.minimum_earned_percent` | yes (optional properties) | every LOB |
| CR-GL-16 | `Deductible.basis` | yes (optional property) | every LOB |
| CR-GL-17 | one parsed format for `forms_and_endorsements[].edition_date` | rule in the field description | every LOB (description) |
| CR-GL-18 | `lob_schema_map.yaml` gl entry: `routing_key: gl`; confirm the gl edits of CR-GL-06 | map only | none |
| CR-GL-03 | `X_ADDITIONAL_INSURED`, `X_PRIMARY_NONCONTRIBUTORY`, `X_WAIVER_OF_SUBROGATION` in `fideon:shared_coverage_codes` | yes | gl (codes renamed in a gl major version) |
| CR-GL-19 | `tools/check_lob_defs.py`, `tools/check_common_compat.py`, prompt-schema manifest (handoff Part A2-A4) | tooling | none |

**Owner's decisions (8 October 2026; `common/CHANGELOG_common.md` 1.1.0):**
- **Accepted into common model 1.1.0:** CR-GL-02, CR-GL-03 (homeowners left alone), CR-GL-12, CR-GL-17 (parsed
  by code, not by the model) and CR-GL-18, with the CR-GL-06 map edit.
- **Decided, no model change:** CR-GL-15 (`gl`).
- **Built first:** CR-GL-19.
- **Held:** CR-GL-16, for a second line's evidence.

gl 3.0.0 (draft) adopts CR-GL-02, CR-GL-03 and CR-GL-12. The CGL seed golds are drafted against it once it is
approved.

**CR-GL-01. Training parity (fine-tuning repository owner).**
The fine-tuning repository (commit 9679ec8) has no SPEC_21 gl schema: its `configs/canonical schema/LOB Schema/gl.json` is the pre-SPEC_21 self-contained file (`fideon.canonical.general_liability` 1.5.0) this schema replaces, and adapters for casualty_fleet would be trained on that shape. Its `common.schemas._check_overlay_families` also refuses a layout family that is only partly on the common model, so it cannot take gl without auto, wc and umbrella. Decide one of: (a) migrate the whole casualty_fleet family (gl, auto, wc, umbrella) to the common model together, taking this gl overlay, its coverage-code file and CR-GL-05; (b) allow gl alone. Until then gl is served by the base model (no casualty_fleet adapter exists). What is held equal today: the fine-tuning repository's own code, run on this overlay and this section map in memory, renders byte-identical whole and sliced prompts, schemas, page bounds, capacities and window plans (`fideon/vlm/tests/fixtures/record_finetuning_gl.py`, `finetuning_gl_derived.json`).
**Resolved upstream (9 October 2026).** The fine-tuning repository at c8df4bb carries this gl overlay (3.0.0, common model 1.1.0) and registers it natively ("Read general liability and the fallback as common model 1.1.0 lines"). `finetuning_gl_derived.json` is now recorded from that repository directly, and the live parity checks compare against it with no in-memory registration.

**CR-GL-02. Products-completed operations rate and premium per classification.**
The ISO premium schedule (Diamond State, USLI, Western World: `Rate` / `Advance Premium`, each split `Pr/Co` and `All Other`) prints two rates and two premiums per classification. `RatingExposure` has one `rate` and one `premium`. Proposal: `rate_products_completed_ops` and `premium_products_completed_ops` (or a `hazard` qualifier: premises-operations / products-completed operations). Until decided: `rate` and `premium` hold the `All Other` (premises-operations) figures; the `Pr/Co` figures go to `additional_fields` (`Products-Completed Operations Rate`, `Products-Completed Operations Premium`, `section_hint` the classification).
**Open (2.0.0).** gl 1.1.0-1.2.0 held these figures in a LOB block (`gl.products_completed_ops_rating[]`). SPEC_21 v0.9 (4.2, 4.2.4) does not allow that: a LOB that prints rating exposures uses the common block, and a new field in an existing common block is a change request. Removed in 2.0.0; the figures travel as `additional_fields` again. Request (revised 8 October 2026 after review): add one optional property to `RatingExposure`, `rating_component`, enum `premises_operations` | `products_completed_operations`. The products-completed operations figures of a classification are a second rating column of the same row, so they become a second `rating_exposures[]` row with the same `class_code`, `class_description` and `location_ref`, `rating_component: products_completed_operations` and that column's `rate` and `premium`; the `All Other` row carries `premises_operations`. A row without the property keeps today's meaning (one rate for the classification). This replaces the earlier two-field proposal: no field is duplicated, and the HANDOFF_lob_local_blocks_parallel_families.md B1 rule (reuse the common block) holds. An additive 1.x change (v0.9 4.2.7). Only gl composes `rating_exposures`, so only gl's prompt schema would change (checked: the homeowners and personal_auto prompt schemas do not carry `RatingExposure`). *1.1.0 (8 October 2026):* the `policy_CGL` corpus adds five more carrier folders printing the split (Indium/Travelers, Johnson & Johnson, McNeil, Philadelphia, XS Broker); the Atlantic Casualty and Ohio Security golds (local `cgl_gold/`, not in the repository) carry their `Prod/Ops Rate` figures in `additional_fields` (in the LOB block under 1.1.0-1.2.0).
**Applied in common model 1.1.0 (8 October 2026)** as proposed: `RatingExposure.rating_component`, enum `RatingComponent`. gl 3.0.0 uses it (a second `rating_exposures[]` row per classification).

**CR-GL-03. Shared coverage codes for the casualty endorsements.**
Additional insured, primary and noncontributory, and waiver of subrogation are printed on CGL, BOP liability, commercial umbrella and auto policies alike, and homeowners already carries `HO_ADDITIONAL_INSURED`. By the common model's inclusion rule (printed by two or more LOBs) they belong in `fideon:shared_coverage_codes` as `X_ADDITIONAL_INSURED`, `X_PRIMARY_NONCONTRIBUTORY`, `X_WAIVER_OF_SUBROGATION`. This schema uses `GL_ADDITIONAL_INSURED`, `GL_PRIMARY_NONCONTRIBUTORY`, `GL_WAIVER_OF_SUBROGATION` meanwhile. *1.1.0:* the codes are now printed by 12, 4 and 3 of 20 carrier folders (`gl.coverage_codes.yaml`); `GL_POLLUTION_LIABILITY` (new in 1.1.0) is a candidate for the same treatment once the environmental lines have schemas. Likewise `GL_LIQUOR_LIABILITY` (BOP liability prints liquor liability) and `GL_PROFESSIONAL_LIABILITY` (the professional_eo line, once it has a schema). *1.2.0:* printed by 14, 5 and 6 of 23 carrier folders. Once the owner releases the `X_` codes, gl renames its three codes in a major version with crosswalk entries (SPEC_21 4.1).
**Applied in common model 1.1.0 (8 October 2026).** The three `X_` codes carry the generic printed names only; carrier and ISO form titles stay in `gl.coverage_codes.yaml`. gl 3.0.0 renames its three codes to them (`replaces_codes`, crosswalk entries). Homeowners keeps `HO_ADDITIONAL_INSURED` until its own next major version; meanwhile L1 codes a name both `HO_` and `X_` claim to the `HO_` code, because a line's own coverage-code file wins over a shared code (`fideon/l1/spec21.py`).

**CR-GL-04. Aliases for shared codes.**
The CGL documents print `Medical Expense Limit`, `Medical Expenses Limit`, `Coverage C Medical Payments`, `Medical Payments (Coverage M)` (X_MED_PAY), `Nonowned Hired Auto Liability` (X_HIRED_NONOWNED_AUTO) and `Terrorism Coverage`, `Certified Terrorism Loss` (X_TERRORISM). They are in `gl.coverage_codes.yaml`; the common model's shared aliases may take them.

**CR-GL-05. Section map: `rating_exposures` beside the locations (fine-tuning repository owner).**
The common-model profile of `configs/schema_sections.yaml` reads `locations` in the `arrays` group but does not name `rating_exposures`, so it fell to `lineblk` and its `location_ref` was cut as a cross-group reference: a classification could never name its location. This repository's `config/canonical_schema/common/schema_sections.yaml` adds `rating_exposures` to `arrays` and `array_keys.rating_exposures: [class_code, class_description, location_ref]` (header note in the file), and `fideon/vlm/policy_merge.py` joins an unreferenced classification row like the other referring tables. No source line composes `rating_exposures`, so the seven personal lines are unchanged (parity tests). To be taken upstream before a gl adapter is trained.
**Taken upstream (fine-tuning c8df4bb, 9 October 2026)** with a different key: `arrays` lists `rating_exposures`, `array_keys.rating_exposures: [class_code, location_ref, rating_component]` (one class code at one location per rating column), and the merge lists it among its referring tables. This repository's section map and merge now carry the source's version (re-port of 9 October 2026).

**CR-GL-06. `lob_schema_map.yaml` blocks for gl.**
The map lists `[locations, rating_exposures, rating_modifiers]`. The documents print no rating modifier and do print claims-made terms and a countersignature. Proposal: `blocks_used: [locations, rating_exposures, claims_made_terms, countersignature]`. **Applied 8 October 2026** (project-owner approval) with `common_model.fields.md` regenerated; see CHANGELOG_common.md. SPEC_21 v0.9 4.2.5 makes `lob_schema_map.yaml` the common-model owner's file: the owner confirms this edit, the CHANGELOG_common.md entry and the regenerated `common_model.fields.md`.

**CR-GL-07. Contractors packages filed as gl.**
Utica First's `Contractors Special Policy` and Security Mutual's `Artisan Policy` print, beside the liability limits, property and inland-marine coverages (`Tools & Equipment`, `Installation Floater`, `Accounts Receivable Coverage`, `Valuable Papers & Records Coverage`, `Computer Coverage`, `Employee Dishonesty`; building limits with construction and valuation). L1 files them as gl. They have no gl code and travel as `additional_fields`. Decide: read such a policy as a package (`lob_parts` gl + inland_marine / property) or add the codes.
*1.2.0 (8 October 2026):* a further Utica First Contractors Special renewal prints the same building data on its declarations (building number, year constructed, construction, county, valuation, cause of loss). Building construction or year built is printed by 20 documents of the corpus. `gl.json` composes no `buildings` block, so these stay in `additional_fields` until this is decided.

**CR-GL-08. The general aggregate.**
ISO CG 00 01 Section III: the general aggregate limit caps Coverages A, B and C together (products-completed operations aside), and the each-occurrence limit caps A and C. SPEC_21 keeps every limit inside one coverage, so this schema places both on `GL_BI_PD` (Coverage A). Confirm, or add a policy-level limit concept to the common model.

**CR-GL-09. Labels as aliases.**
The documents print `Insurer`, `Insurance Company`, `Insuring Company` (carrier.name), `Agents Name and Address`, `Servicing Agent`, `Producer Name` (producer.agency_name), `Master Policy Number` (policy.policy_number, on certificates under a master policy), `Coverage Period` (policy.effective_date), `Code No.` (rating_exposures[].class_code), `Premium Basis` (rating_exposures[].exposure_amount), `Advance Premium` (rating_exposures[].premium). They are in this schema's `fideon:aliases`; the common model may take them.

**CR-GL-10. Surplus-lines charges as named `taxes_fees` entries (1.1.0).**
Non-admitted CGL placements in the `policy_CGL` corpus (FORTEGRA, Johnson & Johnson, XS Broker) print a
policy fee, inspection fee, broker or agency fee, stamping (office) fee and surplus lines tax, each with its
own amount. They fit `premium.taxes_fees[]` (description + amount) and the golds record them so. Proposal for
the common-model owner: a recommended description vocabulary (`Policy Fee`, `Inspection Fee`, `Broker Fee`,
`Stamping Fee`, `Surplus Lines Tax`) so a consumer can total them without reading carrier wording. Nothing
changes in the schema meanwhile.

**CR-GL-11. Per-project and per-location general aggregate (1.1.0).**
ISO CG 25 03 / CG 25 04 and carrier equivalents (`Designated Construction Project(s) General Aggregate
Limit`, Liberty Mutual's `Construction Project(s)-General Aggregate Limit (Per Project)` and `Each Location
General Aggregate Limit`) apply the general aggregate per project or per location. `LimitType` has no such
value, so 1.1.0 records it as a `GL_BI_PD` `aggregate` limit whose `description` names the basis. Decide:
keep that, or add `per_project_aggregate` / `per_location_aggregate` to `LimitType` (a common-model change).

**CR-GL-12. Minimum earned premium percentage and carrier financial rating to the common model.**
Non-admitted binders and quotes print the minimum earned premium as a percentage (`Minimum Earned Premium: 25%`;
17 documents, 3 folders) and the writing company's A.M. Best rating (`A+ XV`; 12 documents, 4 folders). Neither is
CGL-specific: any surplus-lines binder prints them. `premium.minimum_earned` is a `MoneyValue` and cannot hold a
percentage, and `carrier` has no rating field. gl 1.1.0-1.2.0 held them in its LOB block; SPEC_21 v0.9 (4.2, 4.2.4) requires a change request instead, so 2.0.0 removes them and they travel as `additional_fields` (`Minimum Earned Premium Percentage`, `Carrier Financial Rating`). Proposal: `premium.minimum_earned_percent` (PercentValue) and `carrier.financial_rating` (FieldValue) in the common model. `premium` and `carrier` are in every LOB's prompt schema, so this changes every line's prompt and, under v0.9 4.2.7, triggers a recheck of each adapter (and the fine-tuning parity fixtures). *8 October 2026:* send now: an A.M. Best rating and a percentage minimum earned premium are printed across lines (surplus-lines binders of any LOB) and the property_pkg team will meet both while drafting in parallel; a property-local copy would fail the duplicate check (v0.9 4.2.7 item 3) and have to be promoted later. Commission percentage
(2 folders) would go with them if a second line prints it.
**Applied in common model 1.1.0 (8 October 2026)** as proposed: `carrier.financial_rating`, `premium.minimum_earned_percent`. gl 3.0.0 uses both. A second line for the percentage: the Sterling flood registry maps a printed `Minimum Earned Premium Percentage`.

**CR-GL-13. Certificates of insurance inside a policy packet (1.2.0).**
Main Street America's policy packets carry ACORD 25 (liability) and ACORD 24 (property) certificates of
insurance beside the declarations (5 documents of the corpus print an ACORD 25). A certificate restates the
policy's limits and adds its own facts: certificate number, certificate date, certificate holder, the
`ADDL INSR` / `SUBR WVD` marks and `GEN'L AGGREGATE LIMIT APPLIES PER: POLICY / PROJECT / LOC`. The certificate
holder fits `interested_parties[]` (`role: certificate_holder`) and the limits restate `coverages[]`, but
`policy.certificate_number` is defined for a certificate under a master policy, not a certificate of insurance,
and the aggregate basis has no field (CR-GL-11). Decide: treat embedded certificates as `form_type: other`
pages whose facts go to `additional_fields`, or give the common model a certificate-of-insurance block.

**CR-GL-14. The property and inland-marine parts of a commercial package (1.2.0).**
Main Street America's Commercial Portfolio and Utica First's Contractors Special print, beside the general
liability, property (building and business personal property limits, causes of loss, valuation, coinsurance,
property deductible, business income) and inland marine (scheduled tools and equipment) coverages, each with its
own premium. The gl schema holds the general liability part and each part's premium (`lob_parts[]`). The other
parts belong to the property and inland-marine schemas, and neither has a SPEC_21 schema yet (nor does BOP or
commercial package). Until one exists, the package flow has nowhere to put them.

**CR-GL-15. The `lob` value of a CGL package part (SPEC_21 v0.9).**
SPEC_21 4.2.3 and 4.2.6 write a CGL package part as `"lob": "cgl"` (`lob_parts[lob=cgl].premium`), while
`lob_schema_map.yaml` gives the line the code `gl` (`routing_key: cgl`) and gl.json fixes `lob_parts[].lob` to `gl`.
Confirm which value a package part carries: the L1 code (`gl`, as now) or the routing key (`cgl`, as the spec's
examples).
**Decided by the owner (8 October 2026): `gl`**, the L1 code, which is also the schema id and the routing key
(CR-GL-18). SPEC_21 4.2.3 and 4.2.6 now write `"lob": "gl"` and `lob_parts[lob=gl]`.

**CR-GL-16. A basis for a deductible (8 October 2026).**
XS Broker's Mesa binder prints `DEDUCTIBLE: $5,000 per occurrence BI/PD`. The amount and the coverage it applies to
have homes (`coverages[coverage_code=GL_BI_PD].deductibles[].amount`); the basis (per occurrence, per claim, per
person) does not: `Deductible` has `deductible_type`, `amount`, `percentage`, `percentage_basis`, `peril`,
`applies_to_coverages` and `applies_to`, and `peril` is for perils (wind, hail). Request: optional `Deductible.basis`
(FieldValue, as printed: `per occurrence`, `per claim`, `each common cause`). Meanwhile the basis travels as
`additional_fields` (`Deductible Basis`). Printed across lines (liability deductibles of CGL, auto and E&O alike).
**Held by the owner (8 October 2026)** until the batch is cut: one binder of one carrier prints it, so the field
would have almost no training signal. The auto and wc drafters look for it. It is added if a second line prints it;
meanwhile the basis stays in `additional_fields`.

**CR-GL-17. One parsed format for form edition dates (8 October 2026).**
`forms_and_endorsements[].edition_date` is a `FieldValue` whose `parsed` repeats `raw`, so the same edition prints and
parses four ways across carriers: `04 13`, `0413`, `04-13`, `0318` (one binder holds `0318`, `03-18` and `04 13`).
Editions are then not comparable across carriers or between two policies. Request: the field description fixes
`parsed` as `MM/YY` (e.g. `04/13`), `raw` unchanged. Every LOB's forms list is affected, so the owner decides it once.
**Applied in common model 1.1.0 (8 October 2026), parsed by code.** The owner decided that this deterministic parse is
the pipeline's, not the model's (decision of 8 October on code-parsed values). `edition_date` carries
`fideon:code_parsed: MM/YY`, and `schema_registry.conform_document` derives `parsed` from `raw` for L1, L3 and
the delivered document. The model's description is unchanged, so no prompt changes. A gold whose `parsed` is
not MM/YY is corrected at seed review.

**CR-GL-18. `routing_key` of the gl entry of `lob_schema_map.yaml` (8 October 2026).**
The map gives gl `routing_key: cgl` (HANDOFF_common_model_1.0.0_fixes.md item 1, 3 October). SPEC_06 §9a
`LOB_TO_LAYOUT_FAMILY` was rewritten on 5 October to the 48 L1 codes and has a `gl` key and no `cgl` key, and the
SPEC_06 amendment of 6 October routes L3 by schema LOB id (`gl`). `routing_key: cgl` therefore names no key. Request:
`routing_key: gl`. The map is the owner's file (SPEC_21 4.2.5), so the owner makes the edit; it also confirms the gl
edits of CR-GL-06. CR-GL-15 (the `lob` value of a package part) is decided with it.
**Applied in common model 1.1.0 (8 October 2026):** `routing_key: gl`, and the CR-GL-06 edit released with it. CR-GL-15 is
decided with it (`gl`).

**CR-GL-19. The CI checks of the parallel-family handoff (8 October 2026).**
HANDOFF_lob_local_blocks_parallel_families.md Part A asks the owner for `tools/check_lob_defs.py` (A2, duplicate local
definitions across LOB files and against the common model), `tools/check_common_compat.py` (A3, 1.x stays additive)
and a prompt-schema manifest with a family-diff tool (A4, which adapters to recheck, SPEC_09 §4.1). None exists in the
repository yet. gl 2.0.0 defines no local `$defs`, so it passes A2 by construction; casualty_fleet and property_pkg
both need A2 before either drafts golds against a local block.
**Built (8 October 2026), accepted by the owner and built before the release:** `tools/check_lob_defs.py`,
`tools/check_common_compat.py` (with `common/CONTRIBUTING_common.md`), and `tools/prompt_schema_families.py` with
`prompt/manifest.json`. All run in CI (`ci-pipeline-api.yml`, "Common-model change control"), and their tests are in
`fideon/schemas/tests/test_common_model_change_control.py`.

**CR-GL-20. L1: read "any one fire" as a per-occurrence qualifier (L1 owner, 8 October 2026).**
`fideon/l1/spec21.py printed_limit_types` returns no type for `Fire Damage (Any One Fire)`, so L1 types the limit by
the code's default, `per_location`. A per-fire limit is `per_occurrence`. Adding `per_occurrence` as a second default
of `GL_DAMAGE_TO_PREMISES_RENTED` is not the fix: `limits_from` uses a default only when it is the single one, so every
unqualified damage-to-premises limit would become `other`. Request: map `any one fire` (and `each fire`) to
`per_occurrence` among the printed qualifiers, with a test.

**CR-GL-21. A coded check: a classification names a scheduled location or none (Audit Gate / synth_core owner).**
Removed from `fideon:audit_rules` in 2.0.0: `all()` and `in` are outside the synth_core relation language
(`relations.py`), so C8 reported the rule as not evaluable on every gold. Request either a coded Audit Gate check
(every `rating_exposures[].location_ref` is a `locations[].unit_id`), or membership support in the relation language
(a synth_core change by pull request to its owner), after which the rule can return.

## V3 recurring-wording review (23 October 2026, decisions 114/115)

Evidence below is counts of documents / carriers from POLICIES-FINAL-V3 (captions printed in at least 5 documents of at least
3 carriers); no text or values. Labels only.

**CR-GL-22. Concepts the V3 CGL documents print that no gl or common-model field holds (to the common-model owner).**
Each is printed by gl and at least one other line, so by the inclusion rule it belongs in the common model, not in a gl block.
1. `policy.effective_time` (FieldValue) - the time of day the period starts and ends, `12:01 a.m. standard time at the insured's
   mailing address` (7 CGL documents of 4 carriers; the wc legacy schema had `policy.effective_time` and `policy.time_zone`).
2. `carrier.company_type` (FieldValue) - `A Stock Company`, `A Mutual Company` (CGL 25 documents of 5 carriers and 9 of 4; wc 7 of 4).
3. `insurer_officers[]` (name, title) - the President and Secretary signature lines (CGL 34 and 37 documents of 13 carriers; wc 7
   of 4 and 6 of 3).
4. `document.copy_designation` (FieldValue) - `Agent Copy`, `Insured Copy` (CGL 8 documents of 4 and 3 carriers; bopgl `Insured
   Copy` 5 of 3).
5. A terrorism disclosure block: federal share percentage, program trigger, annual loss cap (CGL `Notice of Terrorism Insurance
   Coverage` 8 of 4 and `Cap on Losses from Certified Acts of Terrorism` 25 of 6; wc `Policyholder Disclosure Notice` 17 of 7
   and `Disclosure Endorsement` 23 of 3; bopgl cap 13 of 4; boppr 6 of 3). `X_TERRORISM` keeps the premium.
6. `Deductible.basis` - per claim / per occurrence (reopens CR-GL-16, second line: CGL `Deductible Liability Insurance` 21
   documents of 5 carriers; bopgl `Amount of Deductible` 8 of 3).
7. `insurers[]` (letter, name, NAIC) - the insurer A to F letters of an embedded certificate of insurance (CGL 9 documents of 3
   carriers; `Certificate Number` CGL 10 of 3, bopgl 10 of 3) - the common-model answer to CR-GL-13.
8. Shared package coverage codes `X_ACCOUNTS_RECEIVABLE`, `X_VALUABLE_PAPERS_RECORDS`, `X_EMPLOYEE_DISHONESTY`, `X_DEBRIS_REMOVAL`
   (CGL 13 / 6, 9 / 4, 14 / 5, 13 / 6; bopgl 15 / 5, 16 / 4, 11 / 4, 12 / 3) - the common-model answer to CR-GL-07 and CR-GL-14
   for the property and crime coverages of a contractors package filed as gl.
**Not requested:** exclusion endorsements (forms rows), coverage-part titles (lob_parts), privacy, OFAC, fraud and
service-of-suit notices (text sections).

**Status of earlier requests after V3:** CR-GL-07 and CR-GL-14 - item 8; CR-GL-13 - item 7; CR-GL-16 - item 6 supplies the
second line; CR-GL-11 - no V3 caption survives the filter (the per-project / per-location aggregate stays held).

**CR-GL-23. Resolved in gl 3.1.0 (no common-model change):** Supplementary Payments (bail bond and daily loss-of-earnings caps)
and the owners and contractors protective coverage part have gl codes.

**CR-GL-ALIAS. Coverage-code aliases from the G1 scanned and hybrid seeds. Approved by the project owner and resolved in gl 3.2.0 (26 October 2026).** 66 codes and 8 aliases; see CHANGELOG_gl.md.
