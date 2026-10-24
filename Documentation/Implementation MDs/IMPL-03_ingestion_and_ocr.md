# SPEC 03 — Ingestion + MinerU OCR (GPU, with Version + Device Pinning)

> Aligned with the code of 2026-10-23.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02.
>
> **Architecture refs:** `finetuning-architecture-v2.7.docx` §0e (decisions of 2026-10-19: GPU only, reviewed golds), §7 (page images, multi-page), §7c (delivered golds), §8a (**MinerU version pinning**), §11 (resolution cap), §13 steps 1–2, §18a (raw-document storage, dedup, retention, access pattern), §18.2–18.3 (storage tiers, purge cascade).

## Goal

Ingest raw insurance PDFs into the immutable, tenant-scoped `raw-documents/` layer, then run MinerU OCR + page-image rendering into `processed/` — recording the exact MinerU version used, because the model is fine-tuned partly on *how MinerU formats its output*.

This is the front of both the training-data pipeline and (reused) the inference pipeline.

## Deliverables

### 1. `data_pipeline/ingestion/pull_raw_pdfs.py`
For each PDF from a local folder or source location:
- Assign/validate a `source_id` (`{doc_type}_{index}`, four digits: `policy_0001`; doc_type either provided or deferred — allow an `--unclassified` bucket resolved later by the classifier). **`build_source_id` and `paths.raw_pdf` must accept it**, not only the guard in `ingest_directory` — accepting it in one place and rejecting it in the others made every document fail one at a time into `failed`, so a broken mode read as bad input. Ids are numbered per tenant and doc type, past every id already in use in `raw-documents/`, `processed/` or `golden-labels/` (`taken_source_ids`), so the PDF importer and the prepared importer never hand out the same id.
- Compute a **SHA-256 checksum** and **dedup** (arch §18a): if the checksum was already ingested, skip and log. Insurance documents — Loss Runs and renewal policies especially — are frequently re-submitted with only minor changes; de-duping prevents redundant labeling effort and corpus bloat.
- Write `raw-documents/{tenant_id}/{doc_type}/{source_id}/original.pdf` — **immutable, exactly as received, never overwritten**. A corrected version of a document is ingested as a **new `source_id`**, so historical training runs remain reproducible against the exact bytes they trained on.
- Write `metadata.json`: ingestion timestamp, `tenant_id`, source system, original filename, checksum, size, page count, a first **digital-vs-scanned flag**, PII flags placeholder (`not_assessed`), retention class. The ingest flag (`inspect_pdf`) reads the extractable text of up to three pages with pypdf — PyMuPDF when pypdf cannot read the file, so only a PDF neither library opens is refused — and calls a document scanned below 50 characters per page. It counts any extractable text, a searchable scan's invisible OCR layer included, so it is a first look only: the flag the scanned eval subset (IMPL-08) and the ViT gate (IMPL-06) read is the OCR stage's `ocr_meta.is_scanned` (§3), which counts visible text only.
- Idempotent and re-runnable. CLI: `--input` (required to ingest), `--doc-type` (an active type or `unclassified`; required to ingest and for `--purge-plan`), `--tenant` (optional, defaults from env), `--source-system`, `--purge-plan SOURCE_ID` and `--tenant-purge-plan` (below; both print a plan and delete nothing). There is no container flag: the raw container is `AZURE_RAW_CONTAINER`, reached through `for_ingestion()`.
- **Purge plans** (arch §18.3) report what a purge must reach and never delete:
  - `purge_plan(source_id, doc_type, tenant_id, client=)` (`--purge-plan SOURCE_ID`): the document's `raw`, `processed` and `golden_labels` prefixes, `golden-eval-set/{tenant}/{source_id}` (with the action: remove it from the frozen set and update its manifest — later scores are then not comparable with earlier ones), `exports/{tenant}` with `exports_action` (each result set holds many documents' outputs, so each set is checked and the document's files removed, never the prefix; `compare_models` files its copies under tenant `compare` with another source id), `intake` (every `intake/{batch}/{folder}/` holding a folder the document was imported from — each entry of `imported_from_history`, with its batch for a `--from-blob` import, or `imported_from` on an older record — to confirm against the raw `checksum_sha256`; a lookup that finds nothing reads "none found by name; check by hand", never an all-clear), `pod_volume` (the pulled intake copies, the cached page images, the staged training rows, the golden-eval cache under `golden-{version}`, the testing harness's page cache, `testing/results/*/{sid}.json` and `testing/metrics/*/{sid}.metrics.json` — filed by source id, not tenant, so each is confirmed by hand — the adapter cache `/workspace/adapters/`, the parent adapters staged for continued training (`<staging>/parent-adapters/*/*/`), the adapters a serving cold start copied (`<staging>/serving-adapters/`), `ocr_check` reports, and `testing/results/comparisons/`, `extraction_registry.json` and `/workspace/logs/` to check by hand), the corpus versions to rebuild (corpus files are shared, so a row cannot simply be deleted), and `trained_weights`. Every prefix ends in `/`, so `policy_1` does not match `policy_10/`. `--purge-plan` uses an ingestion client, so it can list `intake/`; it still only reads.
  - `tenant_purge_plan(tenant_id, client)` (`--tenant-purge-plan`, needs `--tenant`): every prefix in `paths.TENANT_SCOPED` for the tenant, built from that set so a prefix added there cannot be missed, plus the pod volume (the tenant's page caches, the harness's results and metrics folders, the adapter cache, the staged parent adapters and the serving cold start's adapters), `trained_weights`, and a note that `intake/` batches — the pod's intake folder too — hold every tenant's deliveries, so they must be checked by hand.
  - `trained_weights` comes from `runs_trained_on(client, tenant_id=, source_id=)`, which reads the run registry: each run of the tenant (for a document, each whose corpus manifest lists it in any split) with its corpus version, adapter, merged weights and every quantized format's directory (`quantized_models`; `quantized_model` is the first), staged copy, eval report, calibrator and the tenant's releases built on it, marked **"needs retrain or delete"**. A run whose manifest cannot be read (any registry, storage, validation or JSON error, named by type only), that records no tenant, or (for a document) whose corpus manifest is missing or unreadable is listed "check by hand", never skipped. A run whose corpus does not list the document but that continued a listed run (`parent_run_id`; decision 2026-10-23) carries what its parent learned, so the plan follows the chain down: each such run is listed "continued <parent>, listed here: needs retrain or delete" (or "check by hand with it" when its parent is to be checked by hand). A run whose chain never reaches a run read as clean is listed "check by hand", with every run continued from it, since it may have learned the document: its parent's manifest is not in the registry (the row was removed once a purge reached it, or lost), its parent is another tenant's run, or the chain loops. Retraining after a deletion is a `--fresh` cycle without the document (IMPL-13). A run continued by hand from a checkpoint directory is listed "check by hand" (`continued_from` is a path, so the corpora behind it are not traced), a per-type adapter built on a listed Foundation is listed with it, and one whose Foundation is not in the registry is listed "check by hand". Each run entry names its `parent_run_id` and `continued_from`. A known limit: another tenant's run that continued this tenant's run is not listed. Trained weights are not tenant-prefixed yet (`paths.SHARED`), so a deletion cannot reach them by prefix and the plans name them instead. A storage call that fails (the corpus manifest check, the release index, the run registry) is listed "check by hand", naming the error type only, and the rest of the plan still prints; an unreadable run registry is one `trained_weights` entry with `run_id: null`. Without a client the plan says to pass one.
- Uses the tighter-RBAC container for `raw-documents/` (master §8). **This is the only pipeline component that writes this layer** (the labeled-PDF importer stores its PDFs through `ingest_pdf`). `raw-documents/` is kept in the **Cool tier once processed, never Archive** (arch §18.2): a corpus rebuild after an OCR pin change needs the PDFs readable without hours of rehydration. The tier is a storage-account setting; no code here sets it.


### 2. `data_pipeline/ocr/mineru_version.py` *(new — arch §8a)*
- `get_mineru_version()` → the exact installed MinerU version string (`"3.4.5"`), from the `mineru` package's metadata (else `mineru.version.__version__`, else `"unknown"`, which is recorded rather than raised). A corpus read by MinerU 1.x (`magic-pdf`) records a 1.x version — another distribution.
- `assert_version_matches(corpus_manifest)` → raises when the running MinerU version **or device** differs from what a corpus manifest pins. Device is included because §8a's rule is about the *output distribution*, and GPU and CPU MinerU can format differently — confirm empirically and relax the device check (`DEVICE_AFFECTS_OUTPUT`) if they prove identical.
- **Serving a model trained on MinerU v{n} output against documents processed by MinerU v{n+1} is distribution shift, and is treated as a regression trigger — not a routine dependency bump.** This helper is called by the dataset builder (IMPL-05), the serving pipeline (IMPL-11), and the testing harness (IMPL-12); a mismatch is a loud failure with the remediation stated (reprocess affected documents, increment the corpus version, retrain).

### 3. `data_pipeline/ocr/run_mineru.py`
Batch-process documents from `raw-documents/` (or a passed list of source_ids):
- Run **MinerU 3.4.5's pipeline backend on GPU** to produce per-page markdown. GPU is the **only** execution mode (arch §14): layout detection, table recognition and the OCR models are GPU-bound, and the CPU path falls back to lighter model variants that are both slower and less accurate. `cpu` is refused (`mineru_version.resolve_device`), and the device is **recorded** in `ocr_meta.json`, never assumed.
- Choose how each document is read: from its **text layer** (`parse_method="txt"`) when every page carries visible text and its figures are typed, through **OCR** (`parse_method="ocr"`) when any page lacks a text layer, when a native PDF draws its figures as vector outlines, or — offline — when MinerU's own classification says OCR (`data_pipeline.ocr.modality.reads_by_ocr`, below).
- Render each page to PNG at the **resolution cap from `configs/base_model.yaml`** (`vision.max_image_long_side_px`) — identical in training data prep and production inference. This consistency is mandatory (arch §11): image token count is a direct function of page resolution and is the single biggest cost/latency lever.
- Write `processed/{tenant_id}/{doc_type}/{source_id}/page_{n}.md` and `page_{n}.png`.
- **Preserve page order explicitly** — Interleaved-MRoPE means multi-page ordering is positionally meaningful (arch §3). Page numbering must be stable and 1-based.
- Write per-doc `ocr_meta.json`: `page_count`, **`mineru_version`**, **`ocr_device`** (`cuda`; `cpu` appears only on corpora predating the GPU-only rule), `gpu_name`, **`preprocessing_date`** (ISO 8601, arch §8a), `resolution_cap_px`, `source_checksum`, `failed_pages` (pages OCR read nothing from), **`table_row_counts` per page** (consumed by the list-completeness signal in IMPL-09), `is_scanned` (any page without a visible text layer), `read_by_ocr` (the document went through OCR mode), and `drawn_check` / `ocr_rules` (the version of the reading rules it was read under).
- Idempotent — skip if the meta exists, is not render-only, and the source checksum, MinerU version, device **and** resolution cap all match, and the reading stands under the current reading rules (`OCR_RULES_VERSION`, see Current implementation); a change to any of them forces reprocessing.
- The MinerU invocation sits behind the `OcrEngine` protocol (`MinerUEngine` in production, a stand-in in tests), so it stays swappable.
- CLI: `--doc-type`, `--source-ids ... | --all-unprocessed`, `--tenant`, `--force-reprocess`, `--shard I/N`. `--all-unprocessed` lists documents with no OCR meta, a render-only meta, or a meta from another MinerU version.

### 4. `data_pipeline/ocr/render_only.py` (helper)
- Renders page images without requiring OCR text, at the same resolution cap — the `image_only` mode path (arch §6) for both corpus build and inference. `render_pdf_pages` is also the renderer `MinerUEngine` uses, so OCR'd and image-only documents are the same pixels.
- Writes the same `ocr_meta.json` skeleton (page count, resolution, version, `render_only: true`) minus OCR content, so `image_only` rows are still traceable.

## Constraints
- `raw-documents/` is write-once and tenant-prefixed; assert before writing.
- Resolution cap MUST come from config, never hardcoded, and be logged and recorded in `ocr_meta.json`.
- MinerU version MUST be recorded on every processed document, and a corpus is read by one MinerU version.
- MinerU reads only the pinned model weights, on the GPU.
- No PII in logs (never print OCR text).
- All Blob I/O via IMPL-02.

## GPU only — not a default, a constraint

MinerU runs on **GPU exclusively**. This is not a performance default with an
opt-out; the CPU path is refused.

The reason is correctness, not speed. MinerU's CPU path selects **lighter model
variants**, so the same PDF yields *different markdown*: different table splits,
different cell boundaries, different header handling. The model learns how MinerU
formats its output (arch §8a), so a corpus built across both devices is built
from two distributions — and serving from whichever the pod happened to have is a
third. A speed/cost trade would be a decision; this is a defect.

Consequences, all enforced in code:

- `resolve_device("cpu")` raises. There is no `--device` flag on the OCR CLI.
- With no CUDA device visible, the stage **fails** rather than falling back. A
  silent fallback would finish the job, write markdown from the wrong
  distribution, and report success — invisible until accuracy is unexplainably
  poor.
- MinerU's own device setting is forced too: `MINERU_DEVICE_MODE` is set to
  `cuda` before MinerU is imported, and any other value is refused
  (`mineru_config.assert_on_cuda`) — a GPU being present does not make MinerU
  use it.
- `ocr_device` is still recorded on every document. It is now always `cuda`, and
  that is provenance rather than a variable.
- `assert_gpu_only` refuses to *extend* a corpus that records `cpu` — such a
  corpus predates this rule, stays readable, and cannot take new documents
  without mixing two markdown formats in one training set.
- CI simulates a GPU pod to exercise these paths, because they now refuse to run
  without one.
- **Fideon SPEC_05's CPU OCR is not followed.** SPEC_05 reads native PDFs on
  the CPU and makes the GPU optional for scans; this project runs OCR on the GPU
  only (decision 2026-10-19, arch §0e, §8a). SPEC_05's drawn-figure rule (§4.1)
  and its timeouts are followed.

## Acceptance checklist
- [ ] Ingesting the same PDF twice results in one stored copy (dedup by checksum).
- [ ] Ingesting a corrected document creates a **new `source_id`**; the original is untouched.
- [ ] Digital vs scanned flag is set correctly on sample inputs.
- [ ] A **searchable scan** (page image with invisible OCR text) is a scan: it is read by OCR and recorded `is_scanned`; a blank page does not make a digital PDF a scan.
- [ ] A native PDF whose figures are drawn as outlines goes through OCR; a typed table drawn from thin border rectangles, or a vector logo beside typed text, does not.
- [ ] MinerU output produces `page_*.md` + `page_*.png` at the configured resolution, 1-based and ordered.
- [ ] `ocr_meta.json` records page count, `mineru_version`, **`ocr_device`**, `preprocessing_date`, resolution cap, OCR failures, per-page table row counts, `is_scanned`, `read_by_ocr` and the reading-rules version.
- [ ] The row count discounts **one header per table**, not one per page: a page with two tables otherwise over-reports by one, and the IMPL-09 cross-check then raises a false row-completeness flag — which active learning treats as an unconditional override to full manual review.
- [ ] A **render-only** `ocr_meta.json` (written by `render_only.py` at the same key) does not satisfy the OCR idempotence check. Otherwise a document rendered first is never OCR'd, no markdown is ever produced, and the CLI reports it as processed.
- [ ] `process_batch` reports genuinely skipped documents as skipped rather than counting them as processed.
- [ ] **OCR runs on GPU only.** There is no `--device` flag and no CPU fallback: `resolve_device` refuses `cpu`, and refuses to run at all when no CUDA device is visible.
- [ ] MinerU refuses to run without its config, on a MinerU 1.x config, without pinned pipeline weights on disk, or with `MINERU_MODEL_SOURCE` other than `local`.
- [ ] GPU throughput is measured per document (`benchmark_gpu`), for pod sizing and cost per document. There is no CPU comparison, because there is no CPU path.
- [ ] Re-running OCR on already-processed docs is a no-op; **changing the MinerU version forces reprocessing**, and `--all-unprocessed` lists documents another MinerU read.
- [ ] A text-layer reading stored under older reading rules is read again when the current rules send it through OCR, and recorded as checked when they do not.
- [ ] `--shard I/N` over N processes covers every document exactly once, whatever order each process lists them in.
- [ ] `assert_version_matches` raises on a seeded version mismatch with a remediation message.
- [ ] `render_only` produces images with no OCR dependency.
- [ ] Running `render_only` over an **already-OCR'd** document re-renders its images but **preserves the OCR metadata** — `table_row_counts` survives and the document is not relabelled `render_only`. Overwriting it silently switched off the IMPL-09 detected-row cross-check while the `page_*.md` files still sat there.
- [ ] `find_unprocessed` applies the same render-only test `process_document` does. Counting a render-only `ocr_meta.json` as done filtered the document out of `--all-unprocessed`, so it was never OCR'd and the CLI reported nothing to do.
- [ ] The `_reprocessed` marker is added to the **returned** metadata only, never written to Blob — persisting it made the skip path read it straight back, so `process_batch` reported every skipped document as processed.
- [ ] The serving-side `run_mineru` raises `MinerUError` on a timeout, any engine failure, or fewer than 10 characters read.
- [ ] `--purge-plan SOURCE_ID` lists the document's raw, processed, label and frozen-eval-set prefixes, the tenant's exports, the corpus versions to rebuild and every run trained on it (with weights and releases, "needs retrain or delete"), and every run continued from one of those, down the chain, with a run whose chain reaches a parent not in the registry, another tenant's run or a loop, and a per-type adapter whose Foundation is missing, listed "check by hand"; `--tenant-purge-plan` lists every prefix in `paths.TENANT_SCOPED` and every run of the tenant. Both list the pod-volume copies (training, golden-eval and harness caches, harness results, adapter cache, staged parent adapters, the serving cold start's adapters) and the `intake/` batches of every folder the document was imported from (or say to check them by hand), name every quantized format's directory, end every prefix in `/`, delete nothing, and list a run or storage call that cannot be checked "check by hand".
- [ ] On a common-model policy line (`label_rules.label_rules_apply`), `check_bundle` and `import_prepared` block a bundle whose form number still carries its edition with `edition_date` empty, and warn on a number that may end in one; a self-contained line (auto, wc, umbrella) is not checked for it; the audit reports the same.
- [ ] `check_bundle` and `import_prepared` refuse every gold without a review record — a real `reviewer_id` and an ISO `review_date` not in the future (decision 2026-10-22) — and a twin whose metadata names no seed it inherited its record from, whatever the gold's notes say; both importers copy `reviewer_id`, `review_date`, `draft_backend` and `review_inherited_from` into `label_metadata.json`.
- [ ] `prepare_bundles` writes each bundle's review record from the delivery (a real document's own, a twin's its seed's, naming the seed) and counts the documents left without one; `fill_synthetic_labels` never fills a twin from a seed with no review record.

---

## Current implementation (2026-10-21)

- **MinerU 3.4.5, pipeline backend**: the `ocr` dependency group installs `mineru[pipeline]>=3.4.5,<3.5`
  (plus `six`, which its OCR text system imports, and the training stack's torch 2.8), locked in
  `requirements-ocr.lock` with the same PyMuPDF and pillow as every other environment, so page images are
  drawn identically. Layout, PP-OCRv6 OCR and table models; no vision-language model. The code imports
  `mineru` and records its version in every `ocr_meta.json`.
- **Pinned weights.** Left to itself MinerU downloads the latest weights on first use, so a pod set up next
  month would read pages differently. `scripts/download_mineru_models.sh` downloads the pipeline backend's
  model paths from `opendatalab/PDF-Extract-Kit-1.0` at a fixed revision (`MINERU_MODELS_REVISION`, default
  `ed6b654c018d742e65a17671e379c5e6ecc87ec9`) into `HF_HOME` on the volume, then runs
  `python -m data_pipeline.ocr.mineru_config --write <dir>`, which points `models-dir.pipeline` in MinerU's
  config at them and removes any `model-source` key (MinerU 3.4's config takes only huggingface or
  modelscope there). Reading the local weights is `MINERU_MODEL_SOURCE=local`: the engine sets it and
  `MINERU_DEVICE_MODE=cuda` before importing MinerU (`mineru_config.apply_run_environment`) and refuses
  (`assert_on_cuda`) when either is set to anything else, when the config is missing, when it is a MinerU
  1.x `magic-pdf` config, or when `models-dir.pipeline` names no folder that exists.
  `python -m data_pipeline.ocr.mineru_config` runs the same check and exits 1 when MinerU is not ready.
- **Its config on the volume**: `MINERU_TOOLS_CONFIG_JSON=/workspace/mineru.json` (an absolute path wins over
  MinerU's home-directory default, which a pod stop wipes). `scripts/pod_bootstrap.sh` writes it into `.env`
  (replacing a MinerU 1.x `magic-pdf.json` value), with `MINERU_DEVICE_MODE=cuda`, `MINERU_MODEL_SOURCE=local`,
  `HF_HOME` on the volume and `HF_HUB_ENABLE_HF_TRANSFER=0`, and runs the weight download itself when the
  check fails (step 5). `setup_pod.sh ocr` installs the locked OCR set and runs the check.
- **Its own environment.** MinerU's dependency set (its own transformers, onnxruntime, layout and OCR
  runtimes) is kept apart from the training stack (ms-swift, vLLM 0.11): on the pod it is installed in `/workspace/venv-ocr` (`scripts/pod_bootstrap.sh`) and OCR runs
  there first: `/workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy --all-unprocessed`.
  The pipeline's preprocessing stage then finds every document done (`find_unprocessed` checks for
  `ocr_meta.json`). Run from the training environment with documents still pending, it raises a
  `PipelineError` naming that command (`_assert_ocr_runnable` looks for the `mineru` package).
- **One MinerU per corpus.** `run_mineru --all-unprocessed` re-reads documents another MinerU version read
  (`find_unprocessed(..., mineru_version=...)`), and the corpus build refuses documents read by more than one
  version (`orchestration.pipeline_dag.assert_one_mineru_version`) — the manifest records one version, taken
  from a single document, so a mix would not show there.
- **One MinerU per GPU.** `--shard I/N` keeps the documents whose `crc32(source_id) % N == I`; N processes,
  each with its own `CUDA_VISIBLE_DEVICES`, split one batch with every document in exactly one shard, however
  many are already done.
- **Delivered batches travel through Blob** (the training folders cannot go through git): bundle folders
  (`prepare_bundles` output, below) are staged by azcopy under `intake/{batch}/` in the **raw container**
  (`paths.intake_batch_dir`; `requires_raw_container` covers `intake/`, so only an ingestion or OCR client can
  read it) — `personal-v1` for personal lines, `cgl-v1` for the CGL documents in the smoke run's example —
  pulled onto the volume by `python -m data_pipeline.ingestion.pull_intake --batch <batch>` (to
  `/workspace/intake/<batch>` or `$FIDEON_INTAKE_DIR`, 16 parallel downloads, `.part` then rename, PDFs already
  present skipped, labels and metadata always re-fetched, `--refresh` for everything), then imported from that
  folder. `import_labeled_pdfs --from-blob <batch>` pulls and imports in one command (and imports nothing if
  any file failed to download). Both run detached on the pod.
- **The import** (`import_labeled_pdfs`): every bundle is checked first (`check_bundle`: one PDF and a
  `golden.json` per folder, the label against its line's schema, a valid split; a missing line or
  `field_provenance` only warns). Two more checks block a bundle:
  - **a form's edition left in its form number** with the form's `edition_date` empty, in a shape the label
    rules move (`label_rules.split_form_edition`, IMPL-04), on a line the label rules hold for — a policy on
    a common-model line (`label_rules.label_rules_apply`, `edition_problems`): the bundle was never prepared,
    so re-run `prepare_bundles`. A number that only *may* end in its edition (`may_hold_edition`,
    "CG 20 10 04") is a warning, to check by hand. A self-contained line (auto, wc, umbrella) keeps its form
    numbers as printed: nothing on it moves an edition, in its golds or in serving's answers;
  - **a gold with no review record** (`export_golden_labels.review_problem`, decision 2026-10-22, IMPL-04):
    every gold needs a real `reviewer_id` (a person's name or id in any script; not blank, not a placeholder
    such as "pending", and not a model, system or generator such as "claude", "system" or "fideon_synth") and an ISO
    `review_date` not in the future in `metadata.json`; a twin inherits its seed's, and
    `review_inherited_from` names the seed. The gold's notes and `draft_backend` decide nothing.

  Then each PDF goes through `ingest_pdf` — a re-import of a corrected batch refreshes the label of a document
  already stored and keeps its PDF — and its label and `label_metadata.json` are written with
  `awaiting_ocr: true`, carrying `reviewer_id`, `review_date`, `draft_backend` and `review_inherited_from` from the bundle's metadata
  (IMPL-04 lists the metadata). A PDF that cannot be read fails its own document; the rest of the batch goes
  on. The prepared importer (`import_prepared`) refuses a gold with no review record, and an edition left in a
  form number on a common-model line, the same way, and copies the same four keys.
- **GPU only**, with no CPU option (see the GPU section above and `common/gpu.py` for the same rule on every
  other model step).
- **Render-only documents** (`ocr_meta.render_only: true`, rendered for labeling without OCR) and documents
  whose meta records **no pages** are skipped by the corpus build with a logged reason; they are never
  built into `ocr_plus_image` rows with blank text.
- On the pod, `run_mineru`, `render_only`, `pull_intake`, `prepare_bundles` and all three importers run
  **detached in tmux** (`orchestration/detach.py`) — a closed laptop does not stop an OCR batch.

**MinerU engine, wired** (`data_pipeline/ocr/run_mineru.py::MinerUEngine`, MinerU 3.x pipeline API):
- refuses anything but CUDA (`common.gpu.require_cuda`), applies and checks MinerU's run environment, and
  renders page images with the shared renderer (`render_only.render_pdf_pages`) at the configured cap, so
  OCR'd and image-only documents are the same pixels;
- decides the reading path with `modality.reads_by_ocr`, plus — offline only — MinerU's own
  `mineru.utils.pdf_classify.classify` returning `ocr`; then calls `doc_analyze_streaming` with
  `parse_method` `ocr` or `txt`, OCR language `ch` (`OCR_LANG`: the PP-OCRv6 small pair, which covers English),
  formula recognition off (policies carry no equations) and tables on. Figures MinerU crops go to a temporary
  folder and are dropped: the page image the model sees is the shared renderer's;
- the result's **content list** (`union_make(..., MakeMode.CONTENT_LIST, ...)`) becomes **one markdown string
  per page** (`pages_from_content_list`): headings by `text_level`, list items one per line, tables as MinerU's
  HTML with captions and footnotes, image and chart captions and footnotes only, code, equations as text — and
  the page's **header at its head, its footer, page number, page footnote and margin text at its foot** (MinerU
  1.x dropped them, so a form number printed in a footer reached no page's text). MinerU's own markdown is
  one blob per document, which cannot be split back into pages;
- **table rows are counted in HTML tables** as well as pipe tables (`count_table_rows`) — MinerU writes HTML;
  HTML rows are parsed, not regex-matched: a row is a header only when it has header cells and no data cells
  (so row-header schedules count), and nested tables do not cut the outer table short;
- a page MinerU returns nothing for is kept, and flagged `ocr_failed` when it was read by OCR or has no text
  layer (an empty text-layer page is a blank or picture-only page, not a failure).

Verified here with a stand-in MinerU on a real PDF (`tests/test_mineru_engine.py`,
`tests/test_mineru_force_ocr_and_timeouts.py`); the real library, its pinned weights and GPU output are verified
on the pod by the Phase 0 spike (`check_mineru_gpu`, `check_mineru_determinism`). Serving callers that supply
`page_texts` must produce them with this engine, or the model reads a formatting it never trained on.

**What counts as a text layer, and as a scan** (`run_mineru.text_layer_pages`): a page is digital when it
draws at least 30 visible characters (`TEXT_LAYER_MIN_CHARS`; text render modes 3 and 7 — invisible and
clip-only — and fully transparent text do not count), or when nothing is on it at all (no text, image or
drawing: an empty page has nothing to OCR). Only visible text counts, so a **searchable scan** — a page image
with the scanner's OCR text laid invisibly under it — is a scan, read by MinerU's OCR rather than the scanner's.
`scanned` is recorded per page, `is_scanned` is true when any page is a scan, and any such page sends the
whole document through OCR mode (text mode left scanned endorsement pages of a typed policy unread and
unflagged). `read_by_ocr` records the reading path separately: a native document read by OCR because its
figures are drawn is `read_by_ocr` but not `is_scanned`.

**Drawn figures** (`modality.data_is_drawn`, Fideon SPEC_05 §4.1): a native PDF whose text layer carries the
headings but not the numbers. A page draws its data when it holds at least 100 glyph-sized filled outlines
(0.5–24 pt, at least 1.5 pt thick, curves or five or more segments — rectangles, cell borders and checkboxes
excluded) and at least two per text span. Such a document keeps its modality (`native_pdf`) and is read with
`force_ocr`.

**Readings stored under older rules** (`OCR_RULES_VERSION` = 2; 1 added drawn figures, 2 visible text only).
A stored text-layer reading made under older rules is re-checked on the next run (`_needs_ocr_now`): read
again through OCR when the PDF is now a scan, or (rules < 1) draws its figures; when it stands, its meta gains
`drawn_check` / `ocr_rules` so the PDF is not fetched to check it again. A reading made by OCR, or one whose
check cannot run, is kept as it is.

**Modality, and the serving-side entry point.** `data_pipeline/ocr/modality.py` holds the two modalities
(`native_pdf`, `scanned_pdf`), `detect_modality`, `data_is_drawn` and `reads_by_ocr`: in serving the caller's
`modality` and `force_ocr` are trusted as given (a page without a text layer read in text mode is logged and
flagged), offline they are detected. `data_pipeline/ocr/mineru_runner.py::run_mineru(pdf_bytes, modality,
force_ocr=False)` is the async entry point for a caller that has a PDF rather than page texts: 30 s budget on
the native path, 90 s on the scanned or forced one (Fideon SPEC_05's timeouts, ceilings rather than a latency
budget: no latency SLA is agreed); a timeout, any engine failure, an unknown modality or fewer
than 10 characters read raises `MinerUError` (the document goes to review). The call runs in a worker thread,
so a timeout stops the wait, not MinerU. This repository's own serving endpoint runs no OCR: its requests carry
page texts.

**After OCR** (`python -m data_pipeline.ocr_check --doc-type policy [--tenant T]`): every labelled value is
compared with its page's OCR text, with the audit's matching, and reported per line, real apart from synthetic,
and fields the fill-in added (`fideon:filled`) apart from the generator's own. It fails when fewer than half of
all values are found on a page they cite, or when added fields are found more than `--max-gap` (0.05) less
often than original ones. Report and CSVs in `/workspace/ocr_check` (they quote label values).

### Deliveries into bundles (`prepare_bundles`)

`python -m data_pipeline.ingestion.prepare_bundles --input "<delivery>" --out <folder>` converts a delivery into
import bundles — `<out>/<document>/{document.pdf, golden.json, metadata.json}` — hard-linking the files
(copying across drives; `--mode link|copy|move`). It reads three shapes, chosen by what the input holds:

1. **fideon_synth, first delivery** (`Train|Val|Test/pdfs`, `Train|Val|Test/gold json`, `manifest.csv`):
   metadata from the manifest row — `lob` (after `data/bundle_lob_overrides.csv` and the merged lines),
   `synthetic`, `template_id` (the source), `split`, `carrier`, `sample`, `render_mode` (the manifest's `mode`
   column, else the gold's `fideon:provenance.mode`), and the review record — a real row's own `reviewer_id`, `review_date` and
   `draft_backend` columns; a twin's its seed's (the row of the same source that is not synthetic), with
   `review_inherited_from` naming the seed; a twin's own columns are never read. Rows outside `--lines a,b` (every line by default; compared by schema name, so `--lines wc` keeps a row labelled `workers_comp`: `common.lob.within_lines`), rows with `ok` false and
   rows whose files are missing are skipped and counted; a document named twice is refused. Every document of
   this delivery is a scan without a text layer, so the value-against-page check runs after OCR.
2. **SPEC_21** (`split_manifest.csv`: split, lob, carrier, seed, twin, digital, scanned, gold): each twin
   becomes two documents — `render_mode` digital and scanned, `sample` the twin's number, `template_id`
   `<line>/<seed>` — both reading one corrected gold; a twin whose two renders are the same file (the scanned
   or drawn seeds) is one scanned document. With `--originals` (default `data/source data`: `pdfs/` and
   `gold json/`; `none` leaves them out) each seed is bundled as a real document,
   `<line>__<carrier>__<seed>__original`, in its twins' family and split; a seed with twins in two splits is
   refused. A row's review columns (`reviewer_id`, `review_date`, `draft_backend`) record its **seed's** review:
   the seed's document carries it and every twin inherits it (`review_inherited_from: <line>/<seed>`); rows of
   one seed stating two records are refused. The delivery's `motorcyle` reads as motorcycle (`delivery_line` also applies the merged lines and
   reads `cgl` as `gl`).
3. **Real documents alone, one line** (`--lob <line>`; `<input>/PDFs/<carrier>/<name>.pdf` with
   `<input>/Gold JSON/<carrier>/<name>.gold.json` — the CGL source data, bundled with
   `--lob gl --out data/cgl_bundles`): `--out` is required; a `--lob` outside `--lines`, when that is given,
   is refused; each pair becomes
   `<line>__<carrier folder>__<name>`, with `carrier` left to the gold (a folder is not always one carrier). Its
   review record comes from `<input>/reviews.csv` (`document,reviewer_id,review_date,draft_backend`; `document`
   the gold's path under `Gold JSON/` without `.gold.json`, or the folder name), read with or without a byte-order
   mark (Excel's "CSV UTF-8"); a file that is not UTF-8, or lacks a column, is refused, and the error names the
   header it found. A document it does not list has none. A family is one insured (`named_insured.primary_name` without case, punctuation or legal-form words, hashed),
   so a declarations page, the policy and its renewals stay on one side. The split is drawn as the corpus build
   draws one (`split_groups.assign_group_splits`, seed 42) and delivered in each document's metadata, the same
   on every re-run; `--frozen` (the tenant's eval set is frozen) draws train and val only. The draw holds one
   carrier out of train and val only when the line names at least 10 carriers
   (`split_groups.MIN_CARRIERS_PER_LINE_FOR_HELD_OUT`, decision 2026-10-19; IMPL-05); a placeholder such as
   "N/A" is no carrier. Carriers are keyed as the corpus build keys them (`normalize_carrier` of the gold's
   `company_name`, else `name`, parsed else raw), so "Travelers Indemnity Company" and "TRAVELERS" are one
   carrier. A test document whose carrier key no train document has is marked
   `held_out_carrier: true` — never one of a placeholder carrier. `<out>/split.csv` lists document, family,
   split and `held_out_carrier`. Never `move`: a re-run removes the folder of a document left out, which would be
   the only copy of a moved PDF.

For shapes 2 and 3 every gold goes through `corrected_gold` (on common-model lines: coverage-code recodes, a
premium written twice, a missing carrier, then the label rules — IMPL-04 — which also move a form's edition
printed inside its form number to the form's empty `edition_date`, so the targets change and the bundles are
prepared again); a gold still outside its line's schema, or a package
policy filed under one common-model line, is left out with the reason, and every change is listed in
`<out>/corrections.csv`. Bundle metadata carries `unprinted_values` (values no page prints). A corrected gold
and every `metadata.json` are written as new files, never through a hard link to the delivery or to a batch
linked to the bundles. A re-run leaves `<out>` holding exactly what it bundles: folders an earlier run made for
documents now left out are removed and `corrections.csv` is rewritten. `--dry-run` reports everything and
writes nothing. The report counts the documents bundled with no review record, which the importer refuses. Documents the audit rejected stay out through `data/bundle_exclusions.csv` (`--exclude`).

`held_out_carrier` reaches the bundle's `metadata.json`, and `import_labeled_pdfs.import_bundle` copies it into
`label_metadata.json`: the corpus build reads it (`delivered_held_out`) and records the document with the split's
held-out ids, and the freeze also reads it directly, recording the document's carrier (the build's key) by line in
`held_out_carriers_by_line`, so a later build warns when that carrier starts training. A set frozen before the
importer carried the flag is marked, and its carriers recorded, after a re-import with
`freeze-eval-set --tenant T --mark-held-out` (IMPL-08, IMPL-13).

The audit (`data_pipeline/audit.py`, default input `data/bundles`, every line by default, `--lines a,b` to
hold documents to those lines) blocks a source document found in two splits and a delivery where
only some documents carry a split, and runs the importer's own `check_bundle`, so a form's edition left in its
number (on a common-model line) and a gold with no review record are blockers before upload as well as at
import. It also blocks a folder whose single-line policies mix lines on the common model with self-contained
lines (`output shape`, `audit.output_shape_findings`): one adapter cannot learn two output shapes, and once
they are imported every policy reaches the corpus and training refuses every cycle of the tenant, so they are
left out here, with `prepare_bundles --lines`. It reports
the delivered Train/Val/Test counts (real and synthetic) with the test count checked against the freeze minimum.

### Completing synthetic labels (first delivery)

The generator's labels held ~29% of the fields in their sources' reviewed gold (90,050 of ~300,000 across 1,560
twins; coverages, forms and endorsements, premiums, vehicles almost entirely missing). Trained on them, the model
would learn that most of a policy is absent. `data_pipeline/ingestion/fill_synthetic_labels.py` completes each
twin's label from its source's reviewed gold without regenerating the PDF: a field the twin has is never changed;
a missing field is added **as is** when the generator leaves such values on the page (coverage names, form numbers,
edition dates, the carrier's own name and address), **shifted** by the recorded `date_shift_days` when a full date
(99.6% of shared dates match that shift exactly), **swapped** when it is a replaced identity whose new value is
known from a field both labels share, and **left out** otherwise; amounts are added only when every amount the two
labels share is equal (89% of shared amounts are; some templates change them). Table rows are matched on unchanged
values, and no row is added while the twin has rows that did not match, so none is listed twice. A merged label
that fails the schema is not written. A twin is filled only from a seed whose own manifest row (the row of its
source that is not synthetic) records a review — `reviewer_id` and `review_date`, as `review_problem` reads
them (decision 2026-10-22, IMPL-04); any other twin is refused and counted. A filled twin's `fideon:filled`
names the seed, its review and the seed's own notes. Filling does not make a twin reviewed: its record is its
seed's, inherited at bundling. `--lines a,b` fills those lines only, by schema name; an unknown line is
refused with exit 2. Result on the first delivery: 90,050 → 295,082 fields; 30,268 deliberately
left out. The originals take their reviewed gold through fideon_synth's `use_reviewed_gold.py`. The audit accepts
the reviewed gold's conventions: month/year edition dates in any layout (`05 11`, `0699`, `(06/99)`) and split
limits (several amounts in one field). A SPEC_21 delivery needs none of this: its golds arrive complete and are
corrected at bundling instead.

## Changes since 2026-10-05

- OCR engine: MinerU 1.x (`magic-pdf`, `PymuDocDataset.classify`, `~/magic-pdf.json` `device-mode`, `mineru_config --cuda`) replaced throughout by MinerU 3.4.5's pipeline backend (`doc_analyze_streaming`, `parse_method` txt/ocr, OCR language `ch`), `mineru.json`, `MINERU_DEVICE_MODE` / `MINERU_MODEL_SOURCE`, and the pinned PDF-Extract-Kit-1.0 weights (`download_mineru_models.sh`, `mineru_config --write`).
- Page text now includes header, footer and page number; reading path decided by `modality.reads_by_ocr` (drawn figures force OCR); added `modality.py` and the serving-side `mineru_runner.run_mineru` (timeouts, `MinerUError`).
- Scanned is no longer "MinerU's classification": a text layer is visible text only, so a searchable scan is a scan; `is_scanned` and `read_by_ocr` recorded separately; `OCR_RULES_VERSION` re-check of older readings.
- Corrected the `ocr_meta.json` field list (no low-confidence flags; added `read_by_ocr`, `ocr_rules`, `gpu_name`, `source_checksum`), the idempotence rule (resolution cap too), and the CLIs (`--tenant`, `--shard I/N`; no `--dest-container` on ingestion; `--purge-plan`).
- Ingest flag: documented the pypdf/PyMuPDF fallback and that the ingest flag is a first look, the OCR meta's `is_scanned` the one evaluation reads; per-tenant source-id allocation.
- One MinerU version per corpus (re-read by `--all-unprocessed`, refused by the corpus build).
- Bundles: "Synthetic deliveries" became "Deliveries into bundles" covering the SPEC_21 delivery (two renders per twin, one corrected gold, seeds via `--originals`) and the originals mode (`--lob gl`, insured families, split delivered in metadata, `held_out_carrier`, `split.csv`, `corrections.csv`); noted that the importer does not yet carry `held_out_carrier` into label metadata.
- Added the post-OCR check (`ocr_check`), the importer's behaviour, the `cgl-v1` batch, and `prepare_bundles` / `pull_intake` running detached.
- The importer now copies `held_out_carrier` into `label_metadata.json` (commit f5668a4); `freeze-eval-set --mark-held-out` marks a set frozen before that.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; the architecture reference is `finetuning-architecture-v2.4.docx` (adds §0e, §7c, §18.2–18.3).
- §1 Ingestion: the CLI flags as the parser has them (`--input` only to ingest, `--doc-type` for ingest and `--purge-plan`); purge plans reach the frozen eval set, the tenant's exports and every run trained on the data (`purge_plan`, `tenant_purge_plan`, `runs_trained_on`, `--tenant-purge-plan`), report only; `raw-documents/` is Cool tier once processed, never Archive.
- GPU only: Fideon SPEC_05's CPU OCR is not followed (decision 2026-10-19).
- Acceptance checklist: purge plans, the form-edition blocker, the reviewed-gold rule.
- Current implementation: `check_bundle` blocks an edition left in a form number and an unreviewed model-drafted gold; both importers copy `reviewer_id`, `review_date`, `draft_backend`; the audit runs the same checks; the SPEC_05 timeouts are ceilings, not a latency budget; the originals draw holds a carrier out only in a line naming 10 or more; the label rules move form editions, which changes targets; a filled twin from an unreviewed seed is still refused.

## Changes of 2026-10-22 (code review)

- §1 Purge plans: prefixes end in `/`; `intake` and `pod_volume` entries; each run's staged copy, eval report and calibrator; lineage; unreadable manifests listed by error type; `--purge-plan` uses an ingestion client.
- Not yet: the import records the delivered folder, not the batch, so the plan matches `intake/` by folder name.
- Every gold needs a review record (decision 2026-10-22); a twin inherits its seed's. `check_bundle` and `import_prepared` refuse a gold without one whatever its notes say, and both importers copy `review_inherited_from` too.
- The form-edition blocker holds on common-model policy lines only (`label_rules_apply`), the lines `prepare_bundles` and serving apply the rule to; `import_prepared` checks it as well.
- `prepare_bundles` writes the review record into every bundle (the manifest's columns, a SPEC_21 row's being its seed's; `reviews.csv` for real documents alone) and counts the documents without one; a placeholder carrier is never an unseen one.
- `fill_synthetic_labels` refuses a twin whose seed has no review record, and records the seed, its review and its notes in `fideon:filled`.
- Acceptance checklist: the review-record, form-edition, bundling and filling items.
- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.
- Purge plans: every folder and batch a document was imported from (`imported_from_history`); "none found by name; check by hand" when the lookup is empty; `exports_action`; the golden-eval, harness and adapter caches on the pod's volume; every quantized format; a failed storage call listed by error type while the plan goes on.
- The import keeps `imported_from_history` (`import_history`): a re-delivered PDF no longer erases the folder it came from first.
- `reviewer_id` is a person in any script, never a model, system or generator; `reviews.csv` is read with or without a byte-order mark and its errors name the header found.
- The CGL draw keys carriers as the build does (`normalize_carrier`); the freeze records a delivered held-out carrier by line.

## Changes of 2026-10-22 (one adapter)

- No training scopes (decision 2026-10-22): `prepare_bundles`, `fill_synthetic_labels` and the audit take `--lines a,b` instead of `--scope`, every line by default (they defaulted to `personal_lines`); the CGL bundling command has no `--scope`.
- Code review of 2026-10-23: `prepare_bundles --lines` compares schema names (`wc` keeps a `workers_comp` row); `fill_synthetic_labels` refuses an unknown `--lines` value with exit 2, not a traceback; the audit blocks a folder whose single-line policies mix the two output shapes (`output shape`).

## Changes of 2026-10-23 (incremental training)

- Purge plans (§1, acceptance): a cycle now continues the promoted adapter (decision 2026-10-23), so a run that continued a listed run (`parent_run_id`) is listed with it, down the chain — "continued <parent>, listed here: needs retrain or delete"; retraining after a deletion is a `--fresh` cycle without the document.
- Code review of the incremental training (§1, acceptance): a run whose chain of parents never reaches a run read as clean (a parent not in the registry, another tenant's run, a loop) is listed "check by hand" with every run continued from it, and so is a per-type adapter whose Foundation is missing; both plans name the staged parent adapters (`<staging>/parent-adapters/*/*/`) and the serving cold start's adapters (`<staging>/serving-adapters/`); each run entry names `parent_run_id` and `continued_from`. Known limit: another tenant's run that continued this tenant's run is not listed.

## Changes of 2026-10-23 (architecture v2.6)

- Header: aligned with `finetuning-architecture-v2.6.docx`, which now records the one policy adapter, one version per cycle, incremental training and no merge or quantization.

## Changes of 2026-10-24 (one served MinerU)

- The OCR of record is the MinerU 4 API served on the pod, for the training corpus and for serving alike (decision 2026-10-23): `configs/shared/ocr.yaml` pins its version (4.0.12) and tier (`basic`), read by `common.config.served_mineru`. `data_pipeline.ocr.served_ocr.ServedMinerUEngine` is the OCR stage's engine for it (the `OcrEngine` protocol): pages rendered by the shared renderer while the server reads the PDF, OCR or text mode decided by the stage's rule (`reads_by_ocr`, now `ocr_mode_for`), each task's reported version and tier checked against the pin. It runs wherever the server is reachable - no MinerU and no GPU there - and records `mineru_version` (the pin), `ocr_device: cuda` (the server's), `mineru_backend: api` and `mineru_tier` in `ocr_meta.json` (`process_document` takes the environment from an engine that has one). `run_mineru --mineru-url` (default `$MINERU_URL`, token `$MINERU_TOKEN`) uses it, and `--all-unprocessed` then also lists documents another MinerU read. The in-process MinerU 3.4.5 engine stays for a corpus already read by it; a corpus is still read by one MinerU (`assert_one_mineru_version`).

## Changes of 2026-10-24 (QA fixes)

- A served reading's MinerU tier is part of the skip check: `process_document` re-reads a document whose stored `mineru_tier` is not the engine's, and `find_unprocessed(..., mineru_tier=)` (default: the pinned tier, for readings that record one) lists it, so `--all-unprocessed` reads it again.
- The served MinerU client refuses a task that leaves out any page it was sent (every run of a multi-run read, not only a single task); checks every run's version and tier against the pin (`version_parts`, `tier_parts`), a missing tier included; retries a poll or result fetch after 1, 3 and 8 s on 429/5xx, an unreachable server, a dropped connection or a non-JSON body (never the upload); fails a PDF at the first failing run and stops the others; fails at once on ending statuses MinerU 4 does not use (cancelled, error, ...); and states the server's reason when it cannot read an answer. `select_pages` keeps form fields (pages are copied with `insert_pdf`, as `split_pdf` does). A failed rendering stops MinerU's reading.
- `audit.py` and `label_verification.py` look a value up by its raw form, not a normalised copy: normalised, a code's pieces were words found apart (`common.grounding.appears_on`). Page references are read by `common.canonical.page_numbers`: whole numbers or digit strings; "p1" names no page instead of failing the build.

## Changes of 2026-10-24 (architecture v2.7)

- Header: aligned with `finetuning-architecture-v2.7.docx`, which records the decisions of 2026-10-23 and 2026-10-24: printed values only, consistent output, fields first, one MinerU, production on Azure.
