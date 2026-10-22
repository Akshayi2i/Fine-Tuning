# Value grounding (Audit Gate rule G1) and field locations

6 October 2026. For the pipeline developer (main repository) and, for item 5, the fine-tuning repository developer. Specs: SPEC_07 amendment of 6 October 2026 (items 1–5), SPEC_00 amendment of 6 October 2026 (field locations). Background: `Document_Extraction_Inference_Quality_Matrix_v2.xlsx`, rows "Hallucination risk" and "Traceability".

## Why

The model can return a value that is not printed on the page it cites. Schema-constrained decoding guarantees structure, not truth. G1 checks that every extracted `raw` string exists in the page text of its `page_ref`, which is the cross-check a hybrid OCR + VLM stack depends on. The same match gives each field a box on the page for reviewers, without changing what the model outputs.

| # | Change | Repository |
|---|---|---|
| 1 | Text normalisation and page-text provider | main |
| 2 | `grounding_checker.py` (G1) and flags | main |
| 3 | Field locations on the envelope | main |
| 4 | Routing rows and configuration | main |
| 5 | G1 on golds before training; grounding rate in evaluation | fine-tuning + main |
| 6 | Evaluation of thresholds | main |

---

### 1. Normalisation and page text

**Prompt.**
> Write fideon/audit/text_norm.py: normalise(s) applies Unicode NFKC, case-fold, maps every dash variant (U+2010–U+2015, U+2212, U+FE58, U+FE63, U+FF0D) to "-", every quote variant to ' or ", and collapses all whitespace including line breaks to one space; it returns the normalised string and an index map from normalised positions back to source positions. Write fideon/audit/page_text.py: for L3 documents, page_text(page) returns the MinerU per-page markdown with markdown syntax removed (table pipes, emphasis, heading marks) and keeps the span list with boxes for that page; for L1 documents, it returns the PDF text layer of the page from PyMuPDF words with their boxes. Both return text, the per-character source span index, and the boxes in PDF points (origin top-left, unrotated page). Add tests: the U+2010 hyphen case ("1‐607‐762‐6579" equals "1-607-762-6579"), a value wrapped over two lines, a markdown table cell.

**Accepted when.** Tests pass.

### 2. Rule G1

**Prompt.**
> Write fideon/audit/grounding_checker.py implementing SPEC_07 amendment of 6 October 2026, items 1 and 2: walk every FieldValue (including FieldValue subtypes such as DateValue and MoneyValue) of the canonical JSON; skip raw = null and confidence.source in {deterministic, audit}; for each page in page_ref, match normalise(raw) against normalise(page_text): exact substring for native_pdf without force_ocr, otherwise the best fuzzy match over windows of the same length (character-level similarity, e.g. rapidfuzz partial_ratio / 100) accepted at or above grounding_min_similarity. On a miss add FieldFlag(field_path=<path with indices>, flag_type="not_grounded", detail="<raw> not found on page(s) <page_ref>; best similarity <s>", severity="fail" if the path matches an entry of the schema's fideon:mandatory_fields (with [] matching any index) else "warn"), and set that FieldValue's flagged to true. Register the checker in gate.py after the schema checker. Add AuditResult.not_grounded_count. Add tests: a value present on its page passes; a value present only on another page fails; a mandatory field miss is "fail", an optional one "warn"; a deterministic field is skipped; a scanned value with one OCR character error passes at the configured similarity.

**Accepted when.** Tests pass; every AuditResult reports not_grounded_count.

### 3. Field locations

**Prompt.**
> Add FieldLocation and DocumentEnvelope.field_locations as SPEC_00 amendment of 6 October 2026 specifies. In the grounding checker, for each grounded field, map the matched normalised range back to source characters, collect the spans (L3) or words (L1) it covers, and union their boxes into one bbox per page; write FieldLocation(page, bbox, source, match, similarity) under the field's concrete path. Do not write anything into the canonical JSON. Add tests: a single-line value gets one box covering exactly its words; a wrapped value gets one box per line unioned per page; field_locations keys equal the paths of grounded fields.

**Accepted when.** Tests pass; the envelope JSON of a sample native and a sample scanned policy contains field_locations for every grounded field.

### 4. Routing and configuration

**Prompt.**
> Add to config/audit.yaml: grounding_min_similarity and grounding_max_warns, both with a provisional value clearly marked "provisional until item 6" in a comment. In the STP routing of SPEC_07 §9, send to HITL any document with a not_grounded fail, or with not_grounded warns at or above grounding_max_warns. Add tests for both routes and for a clean document staying STP.

**Accepted when.** Tests pass.

### 5. Golds and evaluation

**Prompt (fine-tuning repository).**
> Before building any training file, run the G1 function of the main repository (import it, do not copy it) on every seed gold and twin gold against its own PDF's page text. Write a report of misses per LOB and seed; a miss in a gold is a gold error to fix, except values the gold marks as derived. The training build fails if any unfixed miss remains. Add a test with one gold containing a value absent from its cited page.

**Prompt (main repository, evaluation).**
> In the SPEC_17 evaluation, report per adapter and per LOB: grounding rate (grounded fields / checked fields), not_grounded fail and warn counts, and the share of documents routed to HITL by G1.

**Accepted when.** The gold report exists and shows no unfixed misses; the evaluation report contains the grounding columns.

### 6. Threshold evaluation

**Prompt.**
> On a held-out set of real documents with signed-off golds, scanned and native, run G1 on (a) the golds and (b) the model outputs. For grounding_min_similarity from 0.80 to 0.98 in steps of 0.02, report for scanned documents: share of gold values accepted (should be near all) and share of known-wrong model values rejected (values that differ from the gold). Report the distribution of not_grounded warns per document for correct outputs. Propose grounding_min_similarity and grounding_max_warns with the trade-off table in docs/grounding_evaluation.md; commit them to config/audit.yaml only after the project owner approves.

**Accepted when.** The report is reviewed and approved values are committed.

---

Not in scope: changing the canonical schema, the prompt schemas or the training targets (locations are post-extraction only); model-predicted boxes (Qwen3-VL grounding output), which may be evaluated later against these text-matched boxes.
