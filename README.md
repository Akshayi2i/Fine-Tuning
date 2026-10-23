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
| `Documentation/finetuning-architecture-v2.6.docx` | The design and its rationale - the *why* (v2.6: one policy adapter, one version per cycle, incremental training and no merge, with the decisions of 2026-10-19, 2026-10-22 and 2026-10-23, §0e and Appendix A). v2.5, v2.4, v2.3, v2.1 and v1 are kept for history only |
| `Documentation/Implementation MDs/IMPL-00` .. `IMPL-15` | Module specs and acceptance criteria - the *how*; IMPL-00 §13 indexes what the code does now |
| `Documentation/Implementation MDs/IMPL_ALL_COMBINED.md` | All specs in one file |

The two are in sync. **If they disagree, that is a bug in one of them** - fix the
disagreement rather than picking a winner.

## Two production modes, one model

1. `ocr_plus_image` - MinerU OCR text plus the page image.
2. `image_only` - the page image alone, no OCR.

The second is why a rule-based label mapper cannot serve this system: on a
scanned page there is no text for a rule to read.

## Operator commands (IMPL-13)

```bash
# 1  ingest -> OCR -> corpus -> train -> select checkpoint  (staged on the pod)
#    A new cycle takes the next version itself and prints it ("this cycle is v3"). It continues the
#    promoted adapter (e.g. v2) unless --fresh, or unless continuing is unsafe (the log says why).
python -m orchestration.run finetune --input ./intake
python -m orchestration.run finetune --input ./intake --fresh     # from the base model

# 2  calibrate -> GATE -> push the adapter + release bundle -> promote. Nothing is merged or quantized:
#    calibration and the gate score the base with the adapter applied, as it is served.
python -m orchestration.run package --version v3
#    The gate compares v3 with the tenant's promoted release. A release that improves no primary metric
#    needs --fixes-defect "<ticket>" (a blank or placeholder value is refused). Scored on fewer than 150
#    real frozen policies (twins do not count), it is capped at interim. A promoted release scored on
#    another frozen set, tenant or scoring code is no baseline: the candidate is judged on floors alone.
#    A failed gate ships only with a written override, naming real gates:
#    --override-approver "<person>" --override-reason "<why, 20+ chars>" --override-gate <gate>
#    A release that passes is promoted, and the one it replaces is archived.

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

# all = 1 + 2. Never includes extraction.
python -m orchestration.run all --input ./intake

# once, from the first real corpus build: the gate's frozen eval set
python -m orchestration.run freeze-eval-set --corpus v1
```

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
cd /workspace && git clone https://github.com/Akshayi2i/fine-tuning.git && cd fine-tuning
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

**OCR runs first, in its own environment**, then the pipeline:

```bash
/workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy --all-unprocessed
source /workspace/venv/bin/activate
python -m orchestration.run finetune --skip-ingest
```

`finetune` finds every document already OCR'd and skips that stage. If documents
still need OCR it stops with the command above rather than failing on an import.

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
| train (`/workspace/venv`) | 2.8.0 (CUDA 12.8) | vLLM 0.11.0, transformers 4.57.6, ms-swift 3.12.6, flash-attn 2.x built on the pod |
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
checks each labelled value against the text on its page, checks formats and
totals, and draws a 5% spot-check sample per line. Fix and re-run until it exits 0. The report and CSVs go to
`data/audit_report/`, which is git-ignored: they quote label values.

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
python -m orchestration.smoke_run          # select -> import -> OCR -> check -> finetune, tenant "smoke", v0
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
#   --logprobs-mode raw_logprobs --allowed-local-media-path /workspace
FIDEON_REMOTE_TOKEN=<their token> python -m testing.extract_server --port 8002 \
    --model-url http://localhost:8000 --mineru-url http://localhost:8001 --images file
```

Post PDFs from the laptop with `testing.extract_client --url https://<pod id>-8002.proxy.runpod.net` as above.
To measure, run `testing.latency_check` on the pod against the same servers; it also scores each document
against its gold JSON (`--gold-dir`, or the `Gold_json` folder beside the PDF's) and records every window's
tokens and seconds:

```bash
python -m testing.latency_check --model-url http://localhost:8000 --mineru-url http://localhost:8001 \
    --images file --out /workspace/latency_check --doc /workspace/pdfs/a__scan.pdf=personal_auto
```

Windows are taken as each is answered: a window cut off or stopped for repeating itself is split and resent at
once, not after the slowest window. `--ocr-parts N` sends a PDF to MinerU as N runs of pages at once (faster only
if one task leaves the GPU idle: measure it); `--tier` picks MinerU's tier.

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
  gate could not compare the two line by line, and would block the cycle only after training it). The reason
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
- **Promoted.** `package` promotes a release whose gate passed. The tenant then has one promoted release:
  serving reads it, and the next candidate is gated against it. The release it replaced is archived. A dry
  run never promotes. A version already promoted or archived is not packaged again.
- **Rollback.** `promote --version v2 --by "<person>"` makes v2 the promoted release again and archives the
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
  2026-10-22), and a corpus built with another window plan neither trains nor packages.
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
- **No ceiling on invented values is gated.** No spec states one, and the gate
  path does not measure it yet.
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
- **Where serving-time OCR runs is open.** The endpoint does no OCR: requests carry page texts,
  which must come from MinerU at the corpus pin, on a GPU. MinerU cannot share an environment with
  vLLM 0.11, so it cannot simply be added to the serving worker; training is unaffected.
- **The pilot protocol is unrun.** `python -m pilot.pilot_report` reports
  `INCOMPLETE` until Experiments A, B and C have each written a report, and a
  missing experiment blocks rather than passing.
