# Insurance Extraction Fine-Tuning (Fideon L3)

Fine-tunes **Qwen3-VL-8B-Instruct** to extract structured JSON from insurance
PDFs, with a calibrated confidence score on every field.

This repo is the **L3 VLM layer** of the Fideon pipeline. L0 to L2 handle intake,
carrier lookup, and structural inference; L3 is the fallback for the documents
they cannot handle - scanned input, unknown carrier, structure failure.

Its output schema is **owned by Fideon SPEC_00**, not by this repo. The schemas
in `schemas/` are those canonical models serialised to JSON Schema.

**One adapter, for policy documents, one version per cycle, trained incrementally** (decisions 2026-10-22
and 2026-10-23). Each fine-tuning cycle trains one LoRA over all the policy data so far, new and existing
(`common/adapter.py`). By default it **continues the promoted adapter** on the same original base model -
never on a merged model - and starts from the base only when it must (see *Versions and the registry*).
Nothing is merged: the release is the adapter, served as a LoRA on the base. The cycle takes the next
version number (`v1`, `v2`, ...), and that number names its run (`policy-v3`), its adapter and its release. There are no training scopes and no layout families. The code
for other document types is kept, unchanged, but not trained. A new training set and canonical schema are to
be delivered, and the corpus, the frozen eval set and the adapter are rebuilt from them.

## Documents

| Where | What |
|---|---|
| `Documentation/finetuning-architecture-v2.7.docx` | The design and its rationale - the *why* (v2.7: printed values only, consistent output, fields first, one MinerU and production on Azure, on top of v2.6's one policy adapter and incremental training; the decisions of 2026-10-19 to 2026-10-24, §0e and Appendix A). v2.6, v2.5, v2.4, v2.3, v2.1 and v1 are kept for history only |
| `Documentation/Implementation MDs/IMPL-00` .. `IMPL-15` | Module specs and acceptance criteria - the *how*; IMPL-00 §13 indexes what the code does now |
| `Documentation/Implementation MDs/IMPL_ALL_COMBINED.md` | All specs in one file |

The two are in sync. **If they disagree,
that is a bug in one of them** - fix the disagreement rather than picking a winner.

## Two production modes, one model

1. `ocr_plus_image` - MinerU OCR text plus the page image.
2. `image_only` - the page image alone, no OCR.

The second is why a rule-based label mapper cannot serve this system: on a
scanned page there is no text for a rule to read.

## Printed values only, and the same answer twice

Two decisions of 2026-10-24: nothing invented in the served JSON, and consistent output.

- **Printed values only** (`serving/printed_values.py`). After the merge, every value the model wrote (each
  `raw`, extra fields and table rows included) is looked up on the document's pages, on scans too, with a
  search that forgives what OCR does to a scan (`common.grounding.ground_tolerant`: a digit lookalike such as
  `NIPO115` for `NIP0115`, one wrong letter in a long word, a figure's separators, a "$" read as a 5). In
  `strict` mode, the default (`printed_values.mode` in `configs/inference/vllm_serving.yaml`), a value printed
  nowhere is taken out: its key stays, null, flagged `<path>:not_printed_removed`; a table row goes once
  nothing in it is found. `flag` keeps the value and flags it `<path>:not_printed`. The price, accepted: a
  value MinerU misread beyond what the search forgives is removed too, and comes back null with the flag.
  With no page text (image only) nothing is checked (`printed_values:no_page_text`). With fields first, the
  fields served at `fields_ready` are first held to the pages they cite (a value not found there yet is
  held back, `pending_full_text`), and the whole document is checked when MinerU's full text arrives.
- **How a value is looked for** (QA fixes of 2026-10-24). On the cited pages that have text, then on every
  page with text. A value no such page prints is removed when every page it cites has text; one whose cited
  page has no text (an image MinerU could not read) is kept and flagged `<path>:unverified_unread_page` -
  never silent, never counted as invented. A code matches only as one run of words ("PAP-2025-0042" is not
  found on a page printing "PAP-2026-0042" beside a 2025); a figure only whole, and a money amount only an
  amount-like figure (a year or a code is no premium); two-character values as whole tokens; a `parsed`
  value behind a blank `raw` in its printed forms. Yes/no readings and single characters stay unchecked.
  Rows are never renumbered after a removal, so a served row's id and links are the same at `fields_ready`,
  at `done` and on the standard path; a removed value is named `list[#N]` (its place in the model's answer).
- **The tier** (decision 2026-10-26). The gate's primary measure is `served_field_match`: field match on the
  image_only readings of REAL documents, as served (held to the document's page text), what users receive.
  Production needs the gate passed, at least 150 real policies, `served_field_match` >= 0.92 and its
  one-sided 95% bootstrap lower bound >= 0.90; `tier_confidence` is "confirmed" when the bound reaches 0.92.
  Non-inferiority and the improvement rule read it too; `field_normalized_match` (three modes) is reported
  with its 0.85 floor, and real ocr_plus_image accuracy has a 0.85 floor of its own. A continued run is held,
  line by line on the served measure, to its parent (margin 0.015) and to the run from the base (0.03, a slow
  drift), Holm-corrected across lines, on at least 30 shared real documents (fewer: a warning with the
  smallest drop that could have been seen). Every gate stores the answers it scored; when only the scoring
  code or settings changed, the promoted or parent release's stored answers are scored again (the model is
  not run again), so the comparison and continuation still hold.
- **Gated.** `hallucination_rate` (of the values checked, the share no page prints, counted before removal so
  it measures the model) and `run_to_run_agreement` (the golden eval reads each document's main mode a second
  time, its windows sent in reverse order, and counts the fields that agree) are gating metrics, each held to
  non-inferiority with the promoted release. No ceiling or floor is set yet; the user sets them after the
  first measured run. Checkpoint selection takes, of the finalists close on field match, the one inventing
  the fewest values; a finalist that reads the same document differently twice cannot win.

## Operator commands (IMPL-13)

```bash
# 1  ingest -> OCR -> corpus -> train -> select checkpoint  (staged on the pod)
#    A new cycle takes the next version itself and prints it ("this cycle is v3"). It continues the
#    promoted adapter (e.g. v2) unless --fresh, or unless continuing is unsafe (the log says why).
python -m orchestration.run finetune --input ./intake
python -m orchestration.run finetune --input ./intake --fresh     # from the base model

# 2  calibrate -> GATE -> push the adapter + release bundle, written GATED (never promoted here). Nothing is
#    merged or quantized:
#    calibration and the gate score the base with the adapter applied, as it is served.
python -m orchestration.run package --version v3
#    The gate compares v3 with the tenant's promoted release. A release that improves no primary metric
#    needs --fixes-defect "<ticket>" (a blank or placeholder value is refused). Scored on fewer than 150
#    real frozen policies (twins do not count), it is capped at interim. A promoted release scored on
#    another frozen set, tenant or scoring code is no baseline: the candidate is judged on floors alone.
#    A failed gate ships only with a written override, naming real gates:
#    --override-approver "<person>" --override-reason "<why, 20+ chars>" --override-gate <gate>
#    A release that passes is left gated; nothing is archived. Production tier: served field match (image_only
#    readings of real documents, as served) >= 0.92 with its one-sided 95% lower bound >= 0.90 ("confirmed"
#    when the bound >= 0.92), on >= 150 real policies.

# 2b the shadow check, on the pod's real server path: ~50 frozen documents read by the candidate and by the
#    promoted release (vLLM with both adapters, --max-loras 2), scored as the gate scores them, and timed:
#    a 200-page scan's fields in 2 minutes (run it near the pod, as the worker runs)
python -m orchestration.run shadow --version v3 --model-url https://<vllm> --mineru-url https://<mineru>
# 2c promotion, by a person: refused without a passing shadow report (or --skip-shadow --reason "<why>");
#    an interim-tier release also needs --accept-interim, and one that failed on its time alone
#    --accept-latency "<why>". The one it replaces is archived.
python -m orchestration.run promote --version v3 --by "<person>" [--accept-interim] [--accept-latency "<why>"]

# resume a cycle, or rerun one that failed: name its version
python -m orchestration.run finetune --version v3 --from-stage training

# the registry: every version, its run status and its release status
python -m orchestration.run versions
# a rollback: serve v2 again (v3 is archived; its weights stay in Blob). Run it from v2's code.
python -m orchestration.run promote --version v2 --by "<person>"

# 3  extraction through the serving pipeline, on documents already imported and OCR'd
python -m orchestration.run extract --model base|v1|v2 --tenant T --source-ids policy_0001 policy_0002
python -m orchestration.run extract --model base|v1|v2 --tenant T --split test --corpus v1
#    base model + a LoRA adapter applied per request, NOT merged (a checkpoint or the run's adapter):
python -m orchestration.run extract --model base --adapter /workspace/staging/.../checkpoint-280 --tenant T --split test --corpus v1
#    JSON per document in testing/results/<model>/, metrics and summary.json in testing/metrics/<model>/
#    and a testing-record row per document (round, date, split, carrier, line, pages, field match, latency,
#    input/output tokens) in testing/metrics/<model>/test_record.csv - --round names the round, e.g. Baseline.
#    Copy the rows into the team's workbook (fills its "Testing Record" sheet; never overwrites, never twice):
python -m testing.testing_record --model base --round Baseline --xlsx "Documentation/Testing Records/FIdeon SLM - Model Training & Test Record_v1.xlsx"

# all = 1 + 2. Never includes extraction.
python -m orchestration.run all --input ./intake

# once, from the first real corpus build: the gate's frozen eval set
python -m orchestration.run freeze-eval-set --corpus v1

# a candidate label-noise list: the release reads the TRAIN documents; values it reads that the page prints
# but the gold lacks go to a person's review queue (golds are never changed)
python -m orchestration.run label-noise --version v3 [--limit 200]

# training throughput: time N steps on the staged corpus (no checkpoint, no manifest), one lever at a time
python -m training.train --corpus v3 --out-version v3 --benchmark-steps 30
```

`finetune`/`all` take `--ocr-workers N` (served MinerU only) and `--allow-dirty` (a real training run on a
working tree with uncommitted changes is refused otherwise; the manifest records the tree's state). Stage
times go to MLflow when `MLFLOW_TRACKING_URI` is set; a stage failure, a gate block and a finished run are
posted to `FIDEON_NOTIFY_WEBHOOK_URL` (HTTPS) when set.

## Pilot protocol (IMPL-15)

Three experiments in increasing order of investment, run **before** committing to
full annotation. Each one attributes the next one's failures, so a missing
experiment blocks the summary rather than passing by default.

```bash
# A  zero-shot baseline - the untuned base model, no annotation cost
python -m orchestration.run extract --model base --tenant pilot --source-ids ...   # imported + OCR'd first; scored against their labels

# C  ... then the go/no-go across all three
python -m pilot.pilot_report --write
```

The deferred hyperparameter sweep (IMPL-06) runs **after** this passes and
**before** production-scale training: sweeping against 25-30 documents per type
measures noise.

## Setup

```bash
cp .env.example .env              # fill in Azure + RunPod credentials
pip install -r requirements.txt   # data pipeline, eval, test tooling - CPU only
pytest -q                         # fixture-driven, no GPU or live Azure needed
```

### The H200 pod: one command

Clone the repo onto the network volume and run the bootstrap. It runs in tmux,
so the laptop can close while it installs and downloads, and it is safe to
re-run: after a pod stop it only checks what is already there.

```bash
cd /workspace && git clone https://github.com/Ideas-To-Impacts/fine-tuning.git && cd fine-tuning
bash scripts/pod_bootstrap.sh --pdf /workspace/sample_policy.pdf   # --pdf is optional
```

It checks the GPU, the `/workspace` mount and 150 GB free; fills `.env`
(`RUNPOD_VOLUME_MOUNT=/workspace`, `HF_HOME`, `MINERU_TOOLS_CONFIG_JSON`) without
overwriting a value you set; builds **two environments** on the volume; moves
MinerU's config onto the volume and sets it to cuda; downloads the base model at
the pinned revision into `/workspace/models`; and runs the Phase 0 spike. It ends
with a list of anything still to do (the Blob login, MinerU's weights).

| Environment | For | Why separate |
|---|---|---|
| `/workspace/venv` | training, `finetune`, `package`, serving (torch 2.8, ms-swift, vLLM 0.11, flash-attn) | |
| `/workspace/venv-ocr` | OCR only (MinerU 3.4, pipeline backend) | MinerU's dependency set (its own transformers, onnxruntime, layout and OCR runtimes) is kept apart from the training stack |

Everything is on `/workspace` because the container disk is wiped when the pod
stops. For Blob, prefer a SAS login (`AZURE_STORAGE_ACCOUNT_URL` +
`AZURE_STORAGE_SAS_TOKEN`, used when both are set) over the connection string,
which carries the account key. A container SAS reaches one container only: give the
raw container its own token in `AZURE_RAW_SAS_TOKEN`, or the one token must be an
account SAS, which reaches the raw PDFs from every pod that holds it. Keep each in
**double quotes** in `.env`: it is sourced by bash, which would otherwise cut it at
the first `;` or `&` (the bootstrap checks `export` and indented lines too, and lists
half a SAS login under "Still to do").

**OCR runs first, through the served MinerU 4.0.12 (`configs/shared/ocr.yaml`), best before the pod
starts**: a corpus read by another MinerU version or tier is refused at corpus build, at preflight and at
promotion. Run it from any machine with Blob raw access, against a dedicated MinerU 4.0.12/basic API (not
production's), several documents at once:

```bash
MINERU_URL=https://<dedicated mineru> MINERU_TOKEN=... python -m data_pipeline.ocr.run_mineru \
    --doc-type policy --all-unprocessed --workers 4 [--shard 1/2]
python -m orchestration.preflight
# then on the pod, MINERU_URL unset: stage 2 finds every document read at the pin and skips
python -m orchestration.run finetune --skip-ingest
```

If documents still need OCR, `finetune` stops with that command. `venv-ocr` (MinerU 3.4.5, in-process) can no
longer produce a trainable corpus.

### One role at a time

On a RunPod pod, one command per role installs its dependencies in the order
that works and checks CUDA afterwards (the bootstrap calls it for `train` and
`ocr`):

| Pod | Command | Installs |
|---|---|---|
| OCR (stage 2) | `bash scripts/setup_pod.sh ocr` | `requirements-ocr.txt`: MinerU 3.4 (`mineru[pipeline]`); then its pinned weights, `bash scripts/download_mineru_models.sh` (writes `mineru.json`; OCR refuses to run without it) |
| Training | `bash scripts/setup_pod.sh train` | `requirements-train.txt`: ms-swift, torch 2.8, **and vLLM** (checkpoint selection and calibration generate with it), then flash-attn built against that torch |
| Serving | `bash scripts/setup_pod.sh serve` | `requirements-serve.txt`: vLLM 0.11.0, the same build the training pod calibrates with |
| Quantization (only once FP8 is verified) | `bash scripts/setup_pod.sh quantize` | `requirements-quantize.txt`: llmcompressor, in its own environment (its `datasets`/`transformers` ranges conflict with ms-swift and vLLM) |

Versions are declared once, in `pyproject.toml`; the requirements files only
choose dependency groups. `tests/test_dependencies.py` fails if the code imports
a package no group installs.

`requirements.txt` is the **laptop** set (data pipeline, audit, tests). The pods
install `requirements-{train,ocr,serve,quantize}.txt` **through their locks**,
`requirements-*.lock`: every package, transitive ones included, resolved for
Linux x86-64 / Python 3.11, so a pod set up next month runs the same stack.

| Environment | torch | Key pins |
|---|---|---|
| train (`/workspace/venv`) | 2.8.0 (CUDA 12.8) | vLLM 0.11.0, transformers 4.57.6, ms-swift 3.12.6, flash-attn 2.8.3 (pinned; check its prebuilt wheel on the pod, or it compiles for 30-60 min) |
| ocr (`/workspace/venv-ocr`) | 2.8.0 | mineru 3.4.5 (pipeline backend; PDF-Extract-Kit-1.0 weights at a pinned revision) |
| serve | 2.8.0 | identical to train for every package they share |
| quantize | 2.8.0 | llmcompressor 0.9 |

pillow (11.3) and PyMuPDF (1.24) are the same in every environment: the OCR
environment renders the pages the model trains on. After changing a dependency
in `pyproject.toml`, run `bash scripts/lock_requirements.sh` (needs `uv`) and
commit the locks; `tests/test_dependency_locks.py` fails on a stale lock. The pod's
GPU driver must support CUDA 12.8 (the bootstrap checks).

## Running on the pod: closing the laptop does not stop it

On the pod, every long job runs inside tmux automatically - the pipeline
(`python -m orchestration.run ...`), training, sweeps, evaluation, OCR,
ingestion, pre-annotation, the Phase 0 spike and `setup_pod.sh`. Type the
command as usual; it starts itself in a tmux session and returns at once with
the session's name. The job belongs to the pod, not to your SSH connection:
shutting the laptop, letting it sleep or losing Wi-Fi does not stop it.

```bash
python -m orchestration.run finetune --corpus-version v1
#   -> "This one is running as 'finetune-20261003-141502'."

bash scripts/pod_run.sh list                  # every job on this pod
bash scripts/pod_run.sh status <name>         # running / finished + exit code + last log lines
bash scripts/pod_run.sh attach <name>         # watch live; Ctrl-b then d to leave it running
bash scripts/pod_run.sh tail <name>           # follow the log on the volume (/workspace/logs)
```

How it decides: on the pod (`RUNPOD_POD_ID` set, or RunPod's `/etc/rp_environment`,
or Linux with `/workspace` mounted and a GPU present) a command not already
inside tmux re-launches itself through `scripts/pod_run.sh`; inside tmux, or on a
laptop or CI, it runs in the foreground as before. `tests/test_detach.py` fails
if a new long-running entry point is added without this guard.

Nothing stops a run except `pod_run.sh stop <name>`, which asks you to type the
name to confirm. The session stays open after the run ends so its output can be
read. What a run does not survive is the pod itself stopping or restarting;
then `status` says so, and `--from-stage` resumes from the last completed stage.

## GPU only

Every step that runs a model refuses to start without a CUDA device rather than
fall back to the CPU (`common/gpu.py`): training, the merge, quantization, vLLM
generation (checkpoint selection, calibration, the golden eval, serving), the
Hugging Face backend and MinerU OCR. Models are placed with `device_map="cuda"`,
never `"auto"`, which would quietly offload layers to the CPU when VRAM runs
short. The error names the cause - a CPU-only torch build, or no visible GPU.
What stays on the CPU has no GPU form: JSON, tokenizer counts, downloads.

## The eval set: freeze it once

Train and val come from each corpus build. The promotion gate does NOT score the
build's test split — that is re-drawn on every rebuild — but a frozen golden eval
set at `golden-eval-set/{tenant}/` in Blob, the same documents for every model
version. Each tenant has its own, frozen from its own corpus. Create it once per
tenant, from that tenant's first real corpus build:

```bash
python -m orchestration.run freeze-eval-set --corpus v1 [--tenant T]
```

A set frozen before sets were per tenant sits at `golden-eval-set/` itself and
is refused until its files are moved under `golden-eval-set/{tenant}/` for the
tenant it was frozen from (its `manifest.json` names it).

It refuses fewer than 150 real test documents per type (arch §15.4; synthetic twins do not count) unless `--allow-small`. It copies that corpus's test split (labels, metadata, page images, OCR) into the
frozen set and records `manifest.json`. From then on every corpus build leaves the
frozen documents and their families out, and splits new documents into train
and val only. Freezing again is refused; replacing the set is a deliberate delete
followed by a new freeze, after which older scores are not comparable: the next candidate is gated on the
floors alone.

## Check the data before training

The synthetic delivery (`data/training data`: `Train|Val|Test/pdfs`, `Train|Val|Test/gold json`,
`manifest.csv`, from fideon_synth's `build_dataset.py`) is first turned into one folder per document, then
audited on the laptop (read-only):

```bash
python -m data_pipeline.ingestion.prepare_bundles --input "data/training data" --out data/bundles
python -m data_pipeline.audit                      # default: data/bundles, every line
```

`prepare_bundles` writes `data/bundles/<document>/{document.pdf, golden.json, metadata.json}` (hard links:
seconds, no second copy), with `metadata.json` taken from the manifest row: `lob`, `synthetic`, `template_id`
(the source document) and the delivery's `split`. It skips lines outside `--lines` (every line by default), rows the generator
flagged and rows whose files are missing. `data/bundles/` is git-ignored. Documents the audit rejects are
listed in `data/bundle_exclusions.csv` (`document,reason`, git-ignored); a re-run keeps them out and removes
any folder an earlier run made. A line the generator's folder got wrong is corrected in `data/bundle_lob_overrides.csv`
(`source_prefix,lob,reason`, git-ignored). Classic auto is not a line of its own: it is read as `personal_auto`
everywhere (`common.lob.MERGED_LINES`), so Hagerty's classic-car policies train as personal auto, as their labels
are written. The audit accepts month/year dates (edition dates, kept as written) and
word-valued indicator fields, and compares amounts by magnitude (a return is printed positive, recorded negative).

**Real documents of several lines** (the first commercial training set, `data/source data`:
`pdf/<line>/<scanned|hybrid>/` with `gold/<line>/<scanned|hybrid>/`) are bundled the same way, without `--lob`:

```bash
python -m data_pipeline.ingestion.prepare_bundles --input "data/source data" --out data/bundles_c1
```

`cgl` reads as gl and `bop` as boppr; each bundle records `document_kind` (scanned or hybrid); a package gold is
held to its folder's line and a gold with no `coverages` gets `[]` (both listed in `corrections.csv`). The split is
by insured - documents sharing the insured's name, the policy number or the mailing address are one family,
across lines - and balanced within each line and kind, so a line's few hybrid PDFs both train and are measured
(`<out>/split.csv`). Documents held out of the set are listed in `<input>/exclusions.csv` (`document,reason`),
read on every run beside `--exclude`. Every page, a hybrid's too, is read by MinerU's OCR in training and serving
(`configs/shared/ocr.yaml` `read_every_page_by_ocr`, part of the OCR pin).

**Labels come from the reviewed gold.** The generator's own labels cover ~29% of what a source's reviewed gold
holds, and the model reads an omitted field as "not on the page". Two steps fix that without regenerating a PDF:
the originals take their reviewed gold (fideon_synth `scripts/use_reviewed_gold.py`), and
`python -m data_pipeline.ingestion.fill_synthetic_labels --input "data/training data" --apply` adds each synthetic
label's missing fields from its source's reviewed gold — never changing a field the label has: unchanged values
as they are, full dates shifted by the recorded `date_shift_days`, a replaced identity only when its new value is
known (else left out), amounts only where the template kept them, table rows matched on unchanged values and never
duplicated. Every merged label must pass the schema. After OCR, the audit checks every value against the page.

**Every gold needs a review record.** Frontier models are never used for gold, synthetic or training data
(decision 2026-10-19), and a gold trains, is frozen into the eval set or is scored only when its
`metadata.json` names who reviewed it and when: `reviewer_id` (a person, in any script; not "pending", "N/A"
or a model, system or generator such as "claude", "system" or "fideon_synth") and
`review_date` (an ISO date, not in the future) (decision 2026-10-22). Nothing is guessed from the gold's
notes. A synthetic twin carries its seed's record (`review_inherited_from`, written by `prepare_bundles`), so a
twin of an unreviewed seed is unreviewed too. Real documents delivered without a manifest take theirs from
`<input>/reviews.csv` (`document,reviewer_id,review_date,draft_backend`; Excel's "CSV UTF-8" is fine). The audit, the import, the preflight,
the corpus build, the freeze, the gate and the harness all refuse a gold without one. A set frozen before the
freeze carried records is fixed with `freeze-eval-set --tenant T --mark-reviewed`, which copies a record only
onto a frozen gold unchanged since the freeze; one that changed means freezing the set again. It can be
given with `--mark-held-out`, and both run.

**The delivered split is used as given.** The generator put each source document and its ten synthetic twins
in one of Train/Val/Test, so a test layout is one the model never trained on. The corpus build takes that
split when every document carries one (`split_groups.assign_delivered_splits`): one family per source, a source
in two splits refused, synthetic twins allowed in val and test only then. A split the build draws itself holds
one carrier out of train and val per line of business, only in a line naming 10 or more carriers. On policies, a
placeholder carrier ("N/A", "Unknown", "Unknown Insurance Company", "Not Stated") is no carrier ("Other Mutual" is one), and a carrier seen through a synthetic family is never the one held out. The gate report adds a
**`real_only`** section, the same metrics over the real documents alone, and the composition of the set.

It reports blockers (one PDF + `golden.json` + `metadata.json` per folder, PDF readable, `lob` valid (and in
`--lines` when given), `page_ref` inside the PDF, a review record, no form edition left inside a form number on a common-model line),
checks each labelled value against the text on its page, checks formats, label consistency and
totals, and draws a 5% spot-check sample per line. Fix and re-run until it exits 0. The report and CSVs go to
`data/audit_report/`, which is git-ignored: they quote label values.

**Each kind of value written one way** (decision 2026-10-24). The audit warns, and never blocks, when the
labels write a field more than one way: a typed field (date, money, number, percent, boolean, enum) in more
than one shape; a placeholder (N/A, NA, None, Not Applicable, -, --, TBD, Unknown, Nil) beside nulls in the
same field; a date's or an amount's `raw` that contradicts its `parsed`. A field written two ways teaches the
model to pick one at random. The warnings are in `warnings.csv` and the report's "Label consistency"
section; which way is the convention is yours to choose. Run it on the new training set before the first
cycle.

## Getting the data to the pod: through Blob, not git

The training folder (~20 GB) never goes through git. Stage it in Blob from the
laptop with [azcopy](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10),
then pull and import it on the pod:

```powershell
# laptop, after the audit passes. <raw container> is AZURE_RAW_CONTAINER; the SAS needs
# read, write and list on it. Re-run the same command to resume or to push corrections.
azcopy sync "D:\Fine-Tuning\data\training data" "https://<account>.blob.core.windows.net/<raw container>/intake/personal-v1?<SAS>" --recursive
```

```bash
# pod (runs in tmux; re-run to resume)
python -m data_pipeline.ingestion.pull_intake --batch personal-v1        # -> /workspace/intake/personal-v1
python -m data_pipeline.audit --input /workspace/intake/personal-v1      # optional second check
python -m data_pipeline.ingestion.import_labeled_pdfs --input /workspace/intake/personal-v1 --doc-type policy --validate-only
python -m data_pipeline.ingestion.import_labeled_pdfs --input /workspace/intake/personal-v1 --doc-type policy
# or pull and import in one: import_labeled_pdfs --from-blob personal-v1 --doc-type policy
```

`intake/` sits in the **raw container**, under the raw layer's access rule: it is
the delivered PDFs with their PII, readable only by ingestion. The pull runs 16
downloads in parallel, writes each file as `.part` and renames it when complete,
skips PDFs already on the volume, and always re-fetches labels, so a label fixed
and re-synced after an audit is never stale on the pod. The upload itself runs on
the laptop, which has to stay on for it; azcopy is parallel and resumable, and
everything after it runs on the pod.

## On the pod, before training: the post-OCR check and a smoke run

The scans have no text layer, so labelled values are checked against the page only once MinerU has run:

```bash
python -m data_pipeline.ocr_check --doc-type policy          # report in /workspace/ocr_check, exit 1 on FAIL
```

It compares every value with its page's OCR text (the audit's matching), and reports the fields the fill-in
added (`fideon:filled`) apart from the generator's own, real apart from synthetic, per line, with the worst
documents. It fails when added fields are found on their page more than 5 points less often than original ones
(`--max-gap`): then the fill-in put values on synthetic pages that are not there, and training waits.

Before the real run, a smoke run proves the whole path on a few documents, under its own tenant and version:

```bash
MINERU_URL=<served MinerU> python -m orchestration.smoke_run --batch <batch>   # select -> import -> OCR -> check -> finetune, tenant "smoke", v0
```

Before any `finetune`, the preflight checks everything it depends on in minutes, and builds the corpus in
memory with the same function the dataset build uses (nothing written):

```bash
python -m orchestration.preflight                             # exit 1 on any FAIL
```

GPU free, packages, the pinned base model, config invariants, disk, Azure write/overwrite/delete, every
document OCR'd, the post-OCR check passed, and the corpus: rows per split, documents set aside and why, share
over the sequence caps (WARN above 5%, FAIL above 25%). The smoke run runs it before its own `finetune`.

When `finetune` follows straight after it in one chain, add `--skip-corpus`: `finetune` builds the same corpus
itself and refuses the same >25% set-aside before anything trains, so the build runs once (~20 min saved on
the CGL corpus) instead of twice.

It takes 4 train, 1 val and 1 test source document with all their twins (split intact, lines spread; more with
`--train-sources`, `--val-sources`, `--test-sources`), never freezes or packages, and leaves the real data
untouched. Read the corpus build's rejected-row count and the
post-OCR report before the full `finetune`.

## Extracting a PDF through the pod's proxy URL

To check the extraction routine on real PDFs and time it, run the test server on a pod and post PDFs to it
from the laptop. Expose an HTTP port (8000) in the pod's settings; the server runs MinerU (in `venv-ocr`) and
`serving.pipeline.extract` with the base model (or `--adapter`), one document at a time:

```bash
# pod, in /workspace/venv: starts in tmux, prints the URL and the token (also in /workspace/extract_server/token.txt)
python -m testing.extract_server
```

```powershell
# laptop: each --doc is <pdf>=<line of business>; results and a timing table in testing/results/proxy_check/
python -m testing.extract_client --url https://<pod id>-8000.proxy.runpod.net --token <token> `
    --doc "D:\path\a__scan.pdf=personal_auto" --doc "D:\path\b__scan.pdf=homeowners"
```

A PDF is answered at once with a job id and polled (the proxy closes a request after about 100 s). The
table gives each document's pages, windows, model calls, OCR and model seconds, schema validity, review
flags and `text_coverage`. Every call but `/health` needs the token; a job's PDF and pages are deleted after it
runs unless `--keep-files`.

### With the pod's own vLLM and MinerU

When the pod already serves the model (`vllm serve`, port 8000) and a MinerU 4 API (port 8001), run the test
server on a third port with `--model-url` and `--mineru-url`. It then loads nothing itself and only the PDF
crosses the proxy; page images stay on the pod. Expose the port (8002) in the pod's settings.

```bash
# pod: vLLM started with these, among its own flags (--allowed-local-media-path only for --images file)
#   --gpu-memory-utilization 0.80 --max-model-len 32768 --max-num-batched-tokens 16384 \
#   --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
#   --logprobs-mode raw_logprobs --generation-config vllm --allowed-local-media-path /workspace
#   (--generation-config vllm: the model's generation_config.json must not set the sampling; the routine
#    sends repetition_penalty 1.0, top_k -1 and no penalties itself, and asks for the top 2 logprobs)
FIDEON_REMOTE_TOKEN=<their token> python -m testing.extract_server --port 8002 \
    --model-url http://localhost:8000 --mineru-url http://localhost:8001 --images file
```

On the pod, a value set on the command line, as the token above, beats the one in `.env` for every setting
`.env`/`.env.example` names, also once the command detaches (`bash scripts/pod_run.sh env` prints what a run
would get). `--remote-token` works too, but its value then shows in the run script and in `ps`.

Post PDFs from the laptop with `testing.extract_client --url https://<pod id>-8002.proxy.runpod.net` as above.
To measure, run `testing.latency_check` on the pod against the same servers; it also scores each document
against its gold JSON (`--gold-dir`, or the `Gold_json` folder beside the PDF's) and records every window's
tokens and seconds:

```bash
python -m testing.latency_check --model-url http://localhost:8000 --mineru-url http://localhost:8001 \
    --images file --out /workspace/latency_check --doc /workspace/pdfs/a__scan.pdf=personal_auto
```

Windows are taken as each is answered: a window cut off or stopped for repeating itself is split and resent at
once, not after the slowest window. `--ocr-parts N` sends a PDF to MinerU as N runs of pages, one after another,
so it reads the first while the rest upload (measure it against one task). Only the pinned MinerU tier
(`configs/shared/ocr.yaml`) is accepted; `--tier` names another only with `--any-mineru`, for timing, and both
`latency_check` and the job server refuse it before the command detaches.

### From a laptop, through the proxy

Off the pod, the bytes are the cost: a page image goes with every window that reads it, about four times a page,
and a laptop uploads about 1 MB/s (a 114-page scan sent 871 MB of requests: about 15 of its 19 minutes). Three
things cut them without changing what the model sees:

- **Gray pages go as grayscale PNGs**, always: vLLM makes every image RGB, so the model gets the same pixels, at
  66% of the bytes (all 114 pages of that scan checked). `--color-pages` sends them as rendered.
- **`--images link`** uploads each page once to Blob while MinerU reads the PDF, and sends vLLM a read-only link it
  fetches itself: about 105 MB of pages instead of 871 MB of requests. It needs a storage login
  (`AZURE_STORAGE_ACCOUNT_URL` + `AZURE_STORAGE_SAS_TOKEN`, or `AZURE_STORAGE_CONNECTION_STRING`) and a read-only
  SAS for one container for the links (`PAGE_LINK_SAS_TOKEN`: `sr=c`, `sp=r`; refused if it may write, or if it
  is an account SAS, which could read the raw documents), unless the login is a connection string holding the
  account key. On the laptop nothing reads `.env`: set them in the shell (`$env:NAME = "..."` in PowerShell);
  they are checked before anything runs. The pages are deleted after each document.
- **`--ocr-parts 4`** starts MinerU on the first pages while the rest of the PDF uploads.

```powershell
python -m testing.latency_check --model-url https://<pod id>-8000.proxy.runpod.net `
    --mineru-url https://<pod id>-8001.proxy.runpod.net --images link --ocr-parts 4 --concurrency 32 `
    --doc "Test Data\personal_auto__foremost_insurance_company__foremost_autop__twin_015__scan.pdf=personal_auto"
```

A RunPod serverless vLLM endpoint is a `--model-url` too (`https://api.runpod.ai/v2/<endpoint id>/openai`, not yet
tried): `--token` is then the RunPod API key, `--mineru-token` the pod's, and `--images link` is needed, since
the serverless API caps a request's size.

### Fields first: one H100 for the model and MinerU

The production gate is a 200-page scan in 2 minutes, on ONE H100 serving both the model and MinerU. MinerU alone
reads 200 pages in 4-5 minutes, so it cannot stand before the model. Fields first (`serving/fields_first.py`)
reads a document in two phases: the model reads the page **images** (`image_only`, a mode it trains on) with the
GPU to itself; MinerU then reads **only the pages the fields cite** (a few dozen of 200, ~15-40 s) and the fields
are held to that text before anything is served (`fields_ready`): a value found there is served, one not found
there is held back (`<path>:pending_full_text`), and a declarations field the model left empty while those pages
print its label and a value is flagged `likely_missed`. So no value the PDF was not seen to print is ever served.
Then MinerU reads the remaining pages - a run at a time, only while no document's fields are being read
(`GpuTurns`) - the model's answer is held to the whole document (a held-back value printed on another page comes
back; one printed nowhere is removed, `not_printed_removed`) and the full-text tier joins it (`done`). Should the
cited pages fail to be read, nothing is served before `done`. The job server takes `--fields-first` (it needs
`--mineru-url`); `/result` answers from `fields_ready` on; `--unverified-fields` serves unchecked fields, for
timing tests only.

Before relying on it, compare it with the standard path on the same PDFs, accuracy and time in one table:

```bash
python -m testing.latency_check --model-url <vllm> --mineru-url <mineru> --runs standard,fields-first \
    --ocr-parts 4 --doc <pdf>=<line> ...
# MinerU's faster tier, timing only (the model is trained on the pinned tier's text):
python -m testing.latency_check ... --runs standard --tier flash --any-mineru
```

`latency_check --repeat` reads every document twice, interleaved with the others, on the server path, and
writes `run_to_run.json` (per document and overall agreement).

MinerU's turns are runs of at most 20 pages (`--run-pages`), so a document arriving mid full text waits about 25 s; a run waits at most `--text-max-wait` (600 s) for its turn. `latency_check` scores fields first when the fields come (`Exact match at fields`) and at the end. From the laptop with `--images link`, fields first puts the page upload before the model (1.5-2 min for 114 pages); on Azure it is seconds.

On the pod's vLLM, besides the flags above: `--max-num-seqs` small enough that the running windows fit in memory
(`first_token_s_max` in the window summary jumps when vLLM puts windows back and reads them again). Each window
summary now gives every job's answer lengths (`by_group`: median, p95, longest): set the answer limits from them.
Reaching 2 minutes also needs the next training cycle in the fast format profile (below). Every release is held to
it before promotion: the shadow check times each reading to its fields (see "Gated, then promoted by a person").

### The fast format profile

`configs/shared/format.yaml` holds one switch, `profile: standard | fast`, read by the dataset build, training and
serving alike. **`fast` is the shipped profile since 2026-10-26**: the new training set is built, trained and served
in it. `standard` is the earlier format, unchanged, kept for a corpus or adapter built in it. What `fast` changes:

- **Page images first, the same windows for every job.** One system instruction shared by every section group; the
  window's pages; the group's own request last. Every group of a document reads the same windows, so vLLM's prefix
  cache reads each window's pages once for all its groups (a 114-page scan: 60 requests instead of 109, each page
  prefilled about once instead of 3.9 times).
- **A compact prompt.** The group's fields as `key: kind - meaning` lines in place of the schema JSON (63-91% shorter);
  the decoding grammar still enforces the schema.
- **Smaller page images.** 1,254,400 px, 1,225 tokens a page instead of 2,425.
- **A compact answer.** `{"r": printed, "v": value, "p": [pages]}`, `r` only when the printed text differs from the
  value (about 29% shorter). Serving expands every answer to `raw`/`parsed`/`page_ref` before the merge, so the
  printed-values check, confidence and the output are unchanged.
- **Mostly image-only training.** 70% of the rows `image_only` (`ocr_plus_image` 0.25, `noisy_ocr_image` 0.05;
  back to 0.4 / 0.1 / 0.5 if the pod run reads image-only more than 2-3 points worse), and checkpoint
  selection reads `image_only` as well as `ocr_plus_image`.

The profile is part of the prompt hash and the window plan id (`4+fast.aligned-1.compact-1:...`): a corpus or adapter
of one profile is refused by the other at training, packaging, promote and serving. The figures above are measured on
synthetic documents; the speed on the H100 is measured on the pod.

### Production on Azure

The extraction routine runs on Azure (a job per PDF: upload, queue, worker, JSON in Blob); RunPod runs only two
plain servers, a stock vLLM and a MinerU 4 API, reached over HTTPS. No code of ours runs on RunPod. Three
things hold that together:

- **One MinerU for training and serving.** `configs/shared/ocr.yaml` pins the served MinerU (4.0.12, tier `basic`).
  The training corpus is OCR'd through the same server (`MINERU_URL`, `MINERU_TOKEN`; `run_mineru --mineru-url`,
  or the pipeline's preprocessing stage when `MINERU_URL` is set), and every served document is too; a task read
  by another version or tier is refused. A corpus read in-process by MinerU 3.4.5 is read again first.
- **The adapter on the pod's volume.** `python -m scripts.served_adapter` downloads the promoted release's adapter,
  refuses one vLLM cannot serve or one trained on another MinerU's text, and prints the copy command and the vLLM
  flags (`--enable-lora --max-lora-rank R --max-loras 1 --lora-modules policy-vN=/workspace/adapters/policy-vN`). The routine
  asks for `policy-vN` and checks at start that vLLM serves it on our base.
- **Pages leave Azure only encrypted, and briefly.** Model and MinerU URLs must be HTTPS (plain HTTP only to this
  machine); page links are read-only and expire in an hour, and the pages are deleted after each document. Not
  code, but needed: put the Azure region and the RunPod data centre near each other; use RunPod's Secure Cloud
  under a data processing agreement that covers insurance documents; keep vLLM's request logging off (its
  default; never `--enable-log-requests`).

**The service** is `azure_service/` (install `pip install -e ".[data,service]"`; deploy with
`azure_service/DEPLOY.md`): an HTTPS API app and one worker app per GPU on Azure Container Apps, jobs on an Azure
Storage Queue, job state and results as JSON in Blob.

- `POST /v1/extractions?lob=<line>[&webhook_url=https://...]` with the PDF as the body (or multipart `file`) and
  `Authorization: Bearer <key>` answers `202 {job_id, status_url, result_url}`.
- A PDF that needs a password to open is read with its password, sent as the multipart field `password` or the
  header `X-PDF-Password` (never in the URL), and stored unencrypted; the password is used once and kept nowhere.
  One that only restricts printing or copying needs nothing. Without the password a locked PDF is refused (400):
  nothing guesses or breaks one. To unlock a copy by hand (a training PDF the audit blocked, a document for
  `latency_check`): `python -m common.pdf_access unlock locked.pdf unlocked.pdf` (the password is asked for).
- `GET /v1/extractions/<id>`: `queued`, `running`, `fields_ready`, `done` or `failed`, with timings.
- `GET /v1/extractions/<id>/result`: the checked fields from `fields_ready` (`final: false`), the final JSON at
  `done` (`final: true`); 409 before.
- A webhook, if given, is posted at `fields_ready`, `done` and `failed`, with no document content and the header
  `X-Extraction-Signature: sha256=<HMAC-SHA256 of the body with the tenant's secret>`.

One worker per GPU, because the worker holds the GPU's turns between the model and MinerU. A worker restart loses
no job: the queue delivers it again, and phase 2 resumes from the saved fields. Not yet run against real Azure.

## Versions and the registry

Every fine-tuning cycle trains the one policy adapter and takes the next version: one more than the highest
number used by the adapter's runs (the registry index) or by any tenant's corpora and releases. It is one
sequence for every tenant, because runs and weights are not kept per tenant. The number names the run
(`registry/scope/policy/policy-v3/`), the adapter (`adapters/scope/policy/v3/`) and the release
(`releases/<tenant>/v3/`).

- **Incremental by default.** A cycle continues the tenant's promoted adapter: its weights, trained further
  on new and existing data, on the same base, with a fresh optimizer and learning-rate schedule. The run
  records its parent (`parent_run_id`). It **starts from the base** when you pass `--fresh` (after deleting a
  client's data, or now and then to reset drift), when nothing is promoted, or when continuing is unsafe: the
  parent trained on another base model, has another LoRA rank, alpha or target modules, its adapter files are
  not in Blob, its `adapter_config.json` cannot be read or was saved with other LoRA settings (dropout,
  rsLoRA or bias too), its release has no bundle, or it was packaged with another prompt hash or window plan
  (a schema, template or section-map change - so the first cycle on the new canonical schema starts fresh),
  or its eval report is missing or on another yardstick (the set frozen again, or other scoring code: the
  gate could not compare the two line by line, and would block the cycle only after training it), or the
  chain's training documents cannot be read, or the corpus would validate on documents the chain trained on
  (a continuing corpus pins those documents to train). The reason
  is logged and recorded (`fresh_reason`). A release with an old-style name (`release-YYYY.M.N`) is continued
  too. A continued cycle may train with fewer epochs or a lower rate: `continued_training:` in
  `configs/training/unified.yaml` (null = as a fresh run; set it from a measured run). A key the
  `optimization` block does not have, or an epoch count or rate that is not positive, is refused before any
  work. `--fresh` is for a new cycle: with `--version` of a version that already trained, or with a
  `--from-stage` after training, it is refused before the command starts.
- **What continuing costs.** Old documents train again in every cycle, so over many cycles they are seen more
  often than new ones; a wrong label learned once carries forward until it is fixed in the data; reproducing a
  version needs its chain of parents; and a deleted client's data stays in every later adapter until a
  `--fresh` cycle trains without it (the purge plan lists the whole chain).
- **Gated line by line against the runs before it.** On top of the usual comparison with the promoted
  release, each line a continued run serves is held to the best run on that line since the last run from the
  base (the parent, its parent, and so on): the one with the highest field match on the line's real
  documents that both reports scored, with at least 5 such documents (on a tie, the nearer run). A paired
  bootstrap over those documents blocks the line (`line:<line>`) unless the lower bound of the 95% interval
  of (candidate - best) is above -0.015. A line with fewer than 5 shared real documents, or a new line, gets
  a warning only. A line the parent's release served and this run no longer serves blocks. With no
  comparable parent report (missing, on another frozen set or scoring code, or without each document's
  line) the run is blocked (`line_regression_evidence`): train the next cycle `--fresh`, or keep the
  parent's report on the frozen set. A small line has a wide interval, so it may block until it has more
  documents; a written override can waive it as `line:<line>`.
- **Served as a LoRA on the base.** Nothing is merged or quantized: the release is the adapter
  (`served_as: lora_on_base`), bf16 only. `extract --model v3` loads the base with v3's adapter: for a
  version not yet packaged, the checkpoint its selection chose, else its latest checkpoint on the volume
  (with a warning). An adapter runs only on the base it trained on: serving, `promote` and
  `extract --model v3` refuse one trained on another base than `model_id@revision` in
  `configs/base_model.yaml`.
- **Keeping the adapter in Blob.** `finetune --push-adapters` pushes the selected checkpoint to Blob at the
  end of checkpoint selection, and every push records which checkpoint it pushed. If the staging volume is
  lost, `package --version v3 --from-blob` scores and releases the adapter in Blob, with no upload. It is
  refused unless both adapter files are there and the recorded push is the selected checkpoint.
- **Gated, then promoted by a person** (decision 2026-10-26). `package` writes a release whose gate passed as
  `gated` and archives nothing. `shadow --version vN` reads about 50 frozen documents through the production
  path on the pod (fields first, the candidate and the promoted release as `policy-vN`/`policy-vK`, vLLM
  `--max-loras 2`) and passes when the candidate's served field match is at most 0.02 below the promoted
  one's, its invented-value rate at most 0.005 above, with no failures and no lower schema validity
  (`releases/{tenant}/{vN}/shadow/shadow_report.json`). It also times every reading to its fields and fails
  when the slowest 5% of documents take longer than their budget (2 minutes up to 200 pages, in proportion
  beyond) or when a 200-page scan, projected from the documents it read, would take longer than 2 minutes.
  The time includes sending the pages from where the check runs, so run it near the pod, as the worker runs.
  A release that failed on its time alone is promoted only with `--accept-latency "<why>"`, recorded with
  the person's name. `promote --version vN --by "<person>"` then promotes
  it, refused without a passing shadow report for the release as packaged (or `--skip-shadow --reason`,
  recorded) and, for an interim-tier release, without `--accept-interim`. Every promotion is recorded on the
  bundle (`promotions`: who, when, the shadow report or skip reason, tier and tier confidence). The tenant
  then has one promoted release; the one it replaced is archived. A version already promoted or archived is
  not packaged again.
- **Rollback.** `promote --version v2 --by "<person>"` makes v2, promoted before, the promoted release again
  (no shadow and no `--accept-interim` asked) and archives the
  current one. Nothing is deleted. It is refused when serving would refuse v2 at start-up (another prompt
  hash or window plan, or an adapter trained on another base): run `promote` from the code v2 was packaged
  with, and serve that code. The next cycle then continues v2.
- **Lines.** The adapter trains on, is scored on and serves **single-line policies** of the lines it trained
  on; the release records those lines. A policy of any other line, a package of several lines, or a policy
  with no line goes to the base model against `_fallback.json`, flagged `lob_fallback_used` (a policy with
  no line only when the caller sends `allow_lob_fallback`). A release that trained on no line is refused.
- **One output shape.** Policies of lines on the common model (today the seven personal lines and `gl`)
  cannot train beside policies of lines on self-contained schema files. The audit blocks such a folder and
  the preflight fails such a tenant, before training would refuse it; leave those lines out before import
  with `prepare_bundles --lines ...`.

## Build status

All ten phases are built. `pytest -q` runs the whole suite on CPU with no live
Azure account.

| Phase | Scope | State |
|---|---|---|
| 0 | Infra + dependency spike | **pending (yours)** |
| 1 | Contracts, configs, fixtures | done |
| 2 | Blob I/O + run registry | done |
| 3 | Ingest + MinerU OCR (GPU) | done |
| 4 | Inference core | done |
| 5 | Labeling + corpus builder | done |
| 6 | Training | done |
| 7 | Evaluation + calibration | done |
| 8 | Merge/quantize, serving, testing harness | done |
| 9 | Orchestration + CI | done |
| 10 | Pilot validation protocol (IMPL-15) | done; **runs when the corpus arrives** |

**Phases 1-10 prove the plumbing. They do not prove the model works.** No
accuracy claim is possible until the pilot runs on a real corpus - and a green
test suite is not evidence about extraction quality.

## Rules that are not style preferences

- **The repo never stores weights, corpora, or PDFs.** Everything data- or
  artifact-related moves through `artifact_registry/` to Azure Blob at runtime.
- **Training and inference prompts must render byte-identically.** Divergence
  degrades a fine-tuned model and is invisible in training metrics.
- **The image resolution cap is identical in corpus prep and serving.** A
  mismatch means serving a distribution the model never trained on.
- **Loss is computed only on assistant tokens.** System, image, and OCR tokens
  are masked to `-100`; break it and the model trains on its own prompt while
  the loss curve looks normal.
- **The alias registry is never used at inference.** It is labeling and
  evaluation material. The model does the semantic mapping.
- **The promotion gate has no bypass flag.** The one way past it is a written
  override (`--override-approver`, `--override-reason`, `--override-gate`, all three
  together, naming real gates), recorded as a waiver in the gate decision and the release bundle.
- **Every page is read, chunk by chunk, never selected pages** - in training and
  in serving, by one window planner (decision 2026-10-19). A window whose pages'
  own text would not fit its call is split into windows that do (decision
  2026-10-22), and a corpus built with another window plan neither trains nor packages. The plan is
  version 4 since the QA fixes of 2026-10-24 (no window of shared pages only; a text split shares a page;
  a split declarations window keeps its lead): the corpus must be rebuilt and the adapter retrained, which
  the next training set does anyway. The scoring code changed with them too, so a promoted release's eval
  report is on another yardstick until it is scored again.
- **No frontier model touches the data.** Drafts come from the self-hosted
  Qwen3-VL only, and no gold trains or scores without a review record.
- **GPU only.** Every step that runs a model refuses to start without CUDA;
  GGUF is an optional edge export, not a deployment target.

## What is deliberately not wired

Every one of these is a documented boundary, not an oversight: each raises with
the reason and what unblocks it. They are the whole of what Phase 0 gates.

| Where | Waiting on |
|---|---|
| `postprocessing/quantize.py` (FP8/AWQ export, GGUF export) | not used by the pipeline since 2026-10-23 (the release is a LoRA on the bf16 base); kept for an edge export |
| `artifact_registry/transfer.py` (`pull_base_model` from the Hub) | not needed on the pod: the base is read from `/workspace/models` |
| `orchestration/runpod_controller.py` (`RunPodBackend`, endpoint deploy) | `RUNPOD_API_KEY` and the network volume; until then jobs run on the pod in tmux |
| `inference_core/model_runner.py` (vLLM and HF backends, written; `training/merge.py` is no longer called by the pipeline) | a GPU to run on - they refuse without CUDA |
| `testing/run_extraction.py` (`extract`), `pilot/zero_shot_baseline.py` CLIs | a GPU with vLLM, as above; `extract` reads documents already imported and OCR'd |
| `training/data_collator.py` custom hook | nothing - ms-swift collates and masks; the hook exists only for a genuine override |
| `evaluation/golden_eval.py`, `data_pipeline/labeling/active_learning.py` inference loops | a live model backend on the GPU; the golden eval runs through the serving pipeline |

`LocalBackend`, `EchoBackend` and `InMemoryBackend` are **real implementations**,
not stubs: they run the whole DAG, the whole pipeline and the whole registry in
process, which is why CI covers this much without a pod.

## Open items

- **No de-identification (Option A, arch §8b).** Text-only de-identification
  corrupts the training signal, so none is done; PII is protected by tenancy,
  access control and encryption, and that is stated to the compliance owner
  (IMPL-05 section 1).
- **No latency SLA is agreed.** Latency is recorded per release, not gated.
- **No ceiling on invented values, and no floor on run-to-run agreement, yet.** Both
  are gated against the promoted release; the user sets the thresholds after the first measured run.
- **The every-word `text_sections` tier is built from Fideon's section builder**, copied into
  `common/section_builder/` (served since 2026-10-23, never generated by the model). The copy drops
  every MinerU markdown table, which our wrapper works around; the fix belongs in FideonSLM.
- **Token parity is measured on the pod only.** `tests/test_parity.py` needs a
  GPU, the base weights, ms-swift and vLLM, and has not run there yet.
- **Field glosses need SME review.** They are prompt text; changing one after the
  first corpus build forces a rebuild and retrain.
- **The dependency spike has not run.** ms-swift with Qwen3-VL, flash-attn,
  vLLM multi-LoRA, MinerU on GPU and its determinism. `scripts/pod_bootstrap.sh`
  runs it; the base model is pinned to `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
  (the Hub commit of 2025-10-15).
- **The Azure service has not run on Azure.** `azure_service/` is tested with in-memory stand-ins for the queue,
  Blob and the model; the queue, managed identity, ingress limits, the Docker build and the 2-minute timing are
  checked on the first deployment (`azure_service/DEPLOY.md`).
- **The fast format profile is unmeasured on a GPU.** Its sizes are measured on synthetic documents; its speed and
  accuracy come from the first training cycle run in it.
- **The pilot protocol is unrun.** `python -m pilot.pilot_report` reports
  `INCOMPLETE` until Experiments A, B and C have each written a report, and a
  missing experiment blocks rather than passing.
