# Fine-tuning repository: code changes to match the specs

5 October 2026. For the fine-tuning repository developer. Source: `Fideon_Specs_Gap_Report.docx` (repository version 5993a3c) and the specs amended the same day: SPEC_04, SPEC_05, SPEC_06, SPEC_09, SPEC_10, SPEC_11, SPEC_14 (amendment blocks at the top of each).

Where the repository's method was the better choice, the specs now describe it, and nothing below asks for it to change: ms-swift, one LoRA config (rank 64, alpha 128, dropout 0.10), lr 1e-4, generated-answer evaluation and checkpoint choice, the multi-floor gate, Azure Blob release registry, page windows from the serving planner, keyword page routing, H200 target, offline review loop. The items below are the places where the code must change.

Each item gives the spec reference, a Claude Code prompt and the acceptance test. Commit each item separately.

| # | Change | Spec | Needed before |
|---|---|---|---|
| 1 | Train `additional_fields` against the SPEC_21 schemas | SPEC_09 amendment 3 Oct, item 1 | personal_lines full run |
| 2 | Carrier and twin_index on rows; carrier hold-out per LOB; 20-twin cap | items 4, 5, 8; SPEC_09 §10 | personal_lines full run |
| 3 | Per-LOB `synthetic_fraction` and `scanned_share`; model card | items 6, 7; SPEC_09 §3, §7 | personal_lines full run |
| 4 | Routing by `LOB_TO_LAYOUT_FAMILY`; base-model fallback; retire `unified` and `policy` | SPEC_06 §9a and amendments; SPEC_09 amendment 5 Oct | first release with routing |
| 5 | Production / interim release in the promotion gate | SPEC_09 §6 | first release |
| 6 | Release measurements and routing check | SPEC_09 §7 items 6–8 | first release |
| 7 | `force_ocr`, caller modality, timeouts | SPEC_05 amendment | serving |
| 8 | Wire loss-run merge and reconciliation | SPEC_04 amendment item 5; SPEC_09 §6 | loss_run adapter |
| 9 | Rename spec references in code | — | any time |
| 10 | Rank-32 comparison run | SPEC_09 §2.2 | after personal_lines set is complete |

---

## 1. Train additional_fields

**Gap.** None of the 33 schema files in the repository defines `additional_fields`, so it is dropped from targets. The SPEC_21 schemas are not in the repository.

**Prompt.**
> Bring the released SPEC_21 schema files into this repository from the main repository: config/canonical_schema/common/common_model.json, config/canonical_schema/policy_check/<lob>.json for every LOB with a schema, and policy_check/_fallback.json. Load them with $ref resolution (synth_core.schema.Schema or an equivalent resolver). Change the target builder so the assistant target for policy_check is core fields plus additional_fields of the SPEC_21 schema, restricted to the window's pages (page references outside the window are trimmed), and never contains text_sections. Keep the 33 existing schema files only for document types that SPEC_21 does not cover, and list which ones remain. Add tests: a gold with additional_fields keeps them in the target; a field whose page is outside the window is absent from that window's target; no target contains text_sections.

**Accepted when.** Tests pass; a regenerated personal_lines training file has `additional_fields` in the targets where the gold has them.

## 2. Row metadata, carrier hold-out, twin cap

**Gap.** Rows have no `carrier` and no `twin_index`. The split actually used holds out no carrier (the drawn split holds out two per document type, not per LOB). No cap on twins per seed.

**Prompt.**
> Add carrier and twin_index to every training and validation row (twin_index null for real documents; seed stays group_id). In the split builder used for training, hold out at least one carrier per LOB for every LOB with two or more carriers, keep all twins of one seed in one split, and write to the run manifest the held-out carriers per LOB and the LOBs with only one carrier. Cap train at 20 twins per seed per render mode, chosen deterministically from the run seed, and record dropped counts in the run manifest; fold this into the planned copy-rotation setting if that is where copies are chosen. Evaluation reports held-out carrier results separately per LOB. Add tests for each rule.

**Accepted when.** Tests pass; the run manifest of a personal_lines run lists a held-out carrier for every LOB with two or more carriers.

## 3. Per-LOB synthetic_fraction and scanned_share; model card

**Gap.** One global input-mode mix; no per-LOB scanned share; no `synthetic_fraction`; no model card.

**Prompt.**
> In config/training/<adapter>.yaml add per-LOB synthetic_fraction and scanned_share, each with a scope-level default. The data builder samples synthetic and scanned examples to these shares (verified real golds are always included). Keep the existing input-mode mix (50% OCR text + images, 20% corrupted OCR + images, 30% images only) as a separate global default that a scope may override. Write a model card per run (markdown in the run folder of the registry) with: examples per LOB and split, real and synthetic counts, synthetic_fraction and scanned_share used, input-mode mix, carrier hold-out, hyperparameters, gate results. Add tests: the built set matches the configured shares within one example per LOB; the model card lists every configured value.

**Accepted when.** Tests pass; a personal_lines run produces a model card.

## 4. Routing by layout family; base-model fallback

**Gap.** The release router chooses by document type and LOB; `configs/layout_families.yaml` declares four families with no LOBs; `unified` and `policy` adapters exist.

**Prompt.**
> Implement LOB_TO_LAYOUT_FAMILY exactly as SPEC_06 §9a (48 L1 codes of config/lob_registry.yaml; five codes map to None) and populate configs/layout_families.yaml from it, or replace that file with it; there must be one source. The router derives layout_family from the LOB, selects the scope adapter of that family from the release bundle, and when the value is None or the release has no adapter for that family, calls the base model with no LoRA against policy_check/_fallback.json and sets lob_fallback_used. A missing LOB or a LOB not in the L1 list is never routed silently: it raises unless the fallback path is chosen explicitly. Remove the unified and policy scopes from training configs, the release bundle and the router; keep their past runs in the registry, marked retired. Add tests: keys equal the L1 codes; personal_auto → personal_lines adapter; title → base model with _fallback; a family without a trained adapter → base model; agriculture_farm → property_pkg; the router and lob_schema_map.yaml (layout_family column) agree once that file carries values.

**Accepted when.** Tests pass; no code path references the unified or policy adapters.

## 5. Production and interim releases

**Gap.** The gate has one field-match floor (0.85).

**Prompt.**
> In the promotion gate keep every existing floor (field match 0.85, list recall 0.75, schema validity 0.98, auto-accept error ≤ 2%, no regression against the previous release) and add: a release is "production" only if field match ≥ 0.92 for every adapter in it; a release that passes all floors with field match between 0.85 and 0.92 is promoted as "interim", and the release manifest records the status. Add tests: field match 0.89 → interim, not production; 0.93 → production; 0.84 → blocked.

**Accepted when.** Tests pass; the release manifest shows the status.

## 6. Release measurements and routing check

**Gap.** P95 latency and GPU memory are not measured; no routing check.

**Prompt.**
> Extend the release step: after smoke_run and compare_models, measure P95 latency per chunk for each adapter on the target GPU with the base model and every adapter of the release loaded, and peak GPU memory, and write both to the release manifest. Add a routing check that, for each of the 48 L1 codes, resolves the adapter or the base-model fallback through the router of item 4 and writes the table to the release manifest. The release step fails if the routing check fails; latency and memory are recorded, not gated, until SPEC_06 sets a new target.

**Accepted when.** A release manifest contains the latency, memory and routing tables.

## 7. MinerU: force_ocr, caller modality, timeouts

**Gap.** Mode is detected automatically, so a native PDF with drawn figures goes to text mode; no timeouts.

**Prompt.**
> Change the MinerU entry point to take modality and force_ocr from the caller, as SPEC_05 §4 specifies; detect modality only when the caller passes none (offline tools), using the same detection function as the upstream stage. Implement the drawn-figure check of SPEC_05 §4 ("The third case"; data_is_drawn in the spec) if the repository has no equivalent, and pass force_ocr=True when it is true. In the serving path run MinerU with the native and scanned timeouts of SPEC_05 (force_ocr uses the scanned timeout) and raise the spec's MinerU error on timeout. Add tests: a native fixture with drawn figures goes through OCR; a timeout raises; offline detection gives the same modality as the upstream function.

**Accepted when.** Tests pass.

## 8. Loss-run merge and reconciliation

**Gap.** `expand_lossrun`, `merge_windows` and `merge_and_reconcile` exist but only tests call them; the totals metric in the gate is never filled.

**Prompt.**
> Wire expand_lossrun, merge_windows and merge_and_reconcile into the loss-run extraction path for documents longer than one window, and fill table_f1 (claims matched by claim number, total incurred within $0.01) and the totals reconciliation result in evaluation. The gate applies table_f1 ≥ 0.88 to loss_run adapters. Add an end-to-end test on a multi-window loss run.

**Accepted when.** Test passes; a loss_run evaluation reports both metrics. Not needed for personal lines.

## 9. Spec references in code

**Gap.** Code comments use "SPEC_10", "SPEC_11", "SPEC_14" for the repository's own architecture sections (merge and quantisation, serving pipeline, unit tests), which clash with the Fideon specs of those numbers.

**Prompt.**
> Rename every reference in code, comments and docs that points to the repository architecture document's sections as SPEC_xx to a distinct prefix (for example ARCH-10). References to Fideon specs keep the SPEC_xx form. List every renamed reference in the commit message.

**Accepted when.** `grep -rn "SPEC_1[0-4]"` in the repository returns only references to Fideon specs.

## 10. Rank-32 comparison (after the personal_lines set is complete)

**Prompt.**
> Train personal_lines with rank 32, alpha 64 on the same split and data as the current rank-64 run, evaluate both with the promotion gate metrics, and write a comparison to the run manifests. Do not change the default; report the result.

**Accepted when.** The comparison is reported; the project owner decides the default.

---

## Not in scope of this handoff

Package multi-pass (SPEC_06 §9a), review-queue service and correction log (SPEC_11), Presidio (compliance decision pending), continuous learning, drift monitoring and DAPT (SPEC_14), sentence encoder (SPEC_10, deferred), quote_gen adapter (deferred).
