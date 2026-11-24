# IMPL-16 — Extraction service on Azure (azure_service/)

> Current specification, aligned with the code of 2026-11-24. Not yet deployed on real Azure: every behaviour below is proven with in-memory stand-ins (tests/test_azure_service.py).

> Read `IMPL-00_MASTER_CONTEXT.md` first. Related: IMPL-11 (the serving pipeline this service calls). How to deploy: `azure_service/DEPLOY.md`. What to do when something fails: `azure_service/RUNBOOK.md`.

## Goal

The service is the production extraction routine. A caller sends one policy PDF and a line of business, gets a job id back at once, and polls for the result. The service runs on Azure Container Apps. RunPod serves only a stock vLLM (the base model with the promoted release's LoRA, asked for as `policy-vN`) and a MinerU 4 API, both over HTTPS; none of our code runs there. One GPU serves both, so each document is read fields first: the checked fields are served early (`fields_ready`), and the full text joins later (`done`).

## Parts

| Part | File | Job |
|---|---|---|
| API | `azure_service/api.py` | FastAPI app (`create_app`). Takes a PDF, checks it, stores it and its job record, puts the job on the queue. Answers status and result polls. Started with `python -m azure_service.api`. |
| Worker | `azure_service/worker.py` | `Worker`: takes jobs from the queue and reads them on one GPU. Two threads share the GPU's turns (`serving.fields_first.GpuTurns`): the fields thread (`poll_once`) and the full-text thread (`text_once`). A third thread renews leases. Started with `python -m azure_service.worker`. |
| Queue | `azure_service/job_queue.py` | `JobQueue` with four operations: `send`, `receive`, `extend`, `delete`. `AzureStorageQueue` in production, `InMemoryQueue` in tests. A message is `{"job_id", "tenant"}`, never the document. |
| Job store | `azure_service/store.py` | `JobStore`: status, result, PDF and phase-1 checkpoint as blobs in the jobs container. Also the logins (`blob_backend`, `jobs_client`, `registry_client`, `job_queue`). |
| Runtime | `azure_service/runtime.py` | The worker's cold start (`load_runtime`): each served tenant's promoted release, the served models, the release calibrators, the pinned MinerU reader. |
| Webhooks | `azure_service/webhooks.py` | `WebhookSender`: signed POSTs for `fields_ready`, `done` and `failed`, with no document content. |
| Settings | `azure_service/settings.py` | `Settings.from_env`: the environment both containers read. `check_api` and `check_worker` refuse a start without what each needs. |
| Job lists | `azure_service/jobs.py` | `python -m azure_service.jobs failed|partial`: the jobs that need a person. |
| Retention | `azure_service/retention.py` | Keeping flagged documents for review. Off by default; on only for tenants named in `REVIEW_RETENTION_TENANTS`. |

Blob layout in the jobs container (`store.py`):

```
jobs/<tenant>/<job id>/status.json             the job's state and timings; no document content
jobs/<tenant>/<job id>/result.json             the served JSON: the fields at fields_ready, the final at done
transient/jobs/<tenant>/<job id>/document.pdf  the PDF, until the job ends (KEEP_FILES keeps it)
transient/jobs/<tenant>/<job id>/phase1.json   the model's answer, for phase 2 after a restart
```

Each write is one whole blob (`BlobClient.write_json`). The result is written before the status that announces it, so a poll that sees `fields_ready` or `done` always finds a result. A job id is 32 lowercase hex characters (`store.JOB_ID`); anything else is "no such job".

## A job, start to end

**States** (`store.STATUSES`): `queued` -> `running` -> `fields_ready` -> `done`, or `failed`. `done` and `failed` are final (`store.FINAL`). A result can be read at `fields_ready` and `done` (`store.READABLE`).

1. **Accepted** (`api.create_app`, `accept`). The API checks the line, unlocks the PDF if needed, counts its pages, checks the webhook. It writes the PDF and a `queued` record (`store.new_record`), then sends the queue message. If the queue cannot be reached, the record is set `failed`, the PDF is deleted and the caller gets 503.
2. **Phase 1, the fields** (`Worker.poll_once`, `_take`, `_read_fields`). The worker takes a message, sets `attempts` to the delivery count, and picks the model for the line (`TenantRuntime.reading`). The record becomes `running` with the model, release, route, calibration and `pins`. `serving.fields_first.read_fields` renders the pages, has the model read the page images (`image_only`), then has MinerU read only the pages the fields cite and holds the fields to that text (`verify_fields`). The checkpoint is written first. Then, if the fields were verified, the fields result is written (`final: false`), the job becomes `fields_ready` and the `fields_ready` webhook goes out. If no cited page had text, nothing is served yet and the job stays `running`.
3. **Phase 2, the full text** (`Worker.text_once`, `_full_text`). MinerU reads the remaining pages in runs of `RUN_PAGES`, only while no document's fields are being read (`GpuTurns.wait_for_text_turn`; a run waits at most `TEXT_MAX_WAIT_S`, then goes anyway). `serving.fields_first.add_full_text` holds the model's answer to the whole text, adds the full-text tier, and scores confidence again with the release calibrators. The final result (`final: true`) is written, the job becomes `done`.
4. **The end** (`Worker._finish`). The queue message is deleted. A `done` job may be retained for review. The PDF and checkpoint are deleted unless `KEEP_FILES`. The webhook for the ending goes out.

The fields thread takes no new job while `MAX_TEXT_BACKLOG` jobs wait for their full text (`poll_once`).

**Leases.** A received message is hidden for `VISIBILITY_TIMEOUT_S` (300 s). The lease keeper renews every held lease every `max(5, VISIBILITY_TIMEOUT_S / 3)` seconds (`Worker._lease_loop`, `renew_leases`). The message is deleted only when the job is `done` or `failed`. Before each status or result write, `Worker._assert_current` checks that no later delivery took the job and that it has not ended; if one did, this delivery writes nothing more.

**Retries.** A phase that raises goes to `Worker._failed`:

- While `attempts < MAX_ATTEMPTS` (3), the job goes back on the queue, visible again after `RETRY_DELAY_S` (60 s). A phase-1 job goes back to `queued`; a job already `fields_ready` stays so. The error and `retry_at` are written on the status.
- On the last attempt, the job is `failed` for good (`Worker._give_up`).
- A job the worker can never read (a tenant it does not serve, or a line no release covers with the base fallback off) raises `runtime.JobRefused` and fails at once, with no retry.
- A message delivered more than `MAX_ATTEMPTS` times is failed on arrival ("given up after N attempts") without being read (`_take`). So a job that kills the worker every time never loops.
- A job whose fields were served but whose full text fails for good ends `failed`, not `done` (decision 2026-10-30: extraction never skips MinerU). It has no final result.

**Fields deadline.** If a job's fields take longer than `FIELDS_DEADLINE_S` (1,800 s; 0 turns it off), `renew_leases` stops renewing its lease (`Worker._stop_waiting`). The reason is written to its status (a `running` job goes back to `queued`), unless the job has ended or a later delivery took it. The queue gives the job out again when the lease runs out; that counts as an attempt. The stuck thread is not stopped; anything it writes later is refused.

**A restart mid-job.** The dead worker stops renewing; the message comes back after `VISIBILITY_TIMEOUT_S`, with one more delivery counted. If the phase-1 checkpoint exists, the next worker resumes phase 2: no model time is spent again and the served fields stay as they were. With no checkpoint, the fields are read again from the start. A second delivery of a job already final is only deleted. On SIGTERM or SIGINT (`Worker.shutdown`) the worker takes no new job and makes its held jobs visible again at once.

Proven by `test_a_pdf_is_served_its_checked_fields_then_its_final_result`, `test_a_poison_job_is_failed_after_its_attempts_and_never_loops`, `test_a_worker_that_dies_after_the_fields_leaves_the_full_text_to_the_next`, `test_a_worker_that_dies_during_the_fields_has_them_read_again`, `test_fields_past_their_deadline_are_given_up_by_this_delivery_and_given_out_again`, `test_a_stopping_worker_hands_its_jobs_back_at_once` and `test_a_job_whose_mineru_text_fails_for_good_is_failed_not_done`.

## The API

| Endpoint | Answer |
|---|---|
| `POST /v1/extractions?lob=<line>[&webhook_url=https://...][&name=a.pdf]` | `202 {"job_id", "status", "status_url", "result_url"}` and a `Location` header. Body: the PDF (`application/pdf`, `application/octet-stream` or no type), or `multipart/form-data` with `file` and the form fields `lob`, `webhook_url`, `name`, `password`. |
| `GET /v1/extractions/{job_id}` | The status record (`store.public_view`): state, attempts, timings, pages, model, release, route, pins, error. Never the webhook address (only `webhook: true/false`) and never the extraction. |
| `GET /v1/extractions/{job_id}/result` | The result at `fields_ready` or `done`; otherwise 409 with the status. |
| `GET /healthz` | `{"status": "ok"}`. No key, no HTTPS rule. |

**Auth.** Every call but `/healthz` carries a key, `Authorization: Bearer <key>` or `X-API-Key: <key>`. `EXTRACTION_API_KEYS` is `{"tenant": "key"}` or `{"tenant": ["key", "next key"]}` (two during a rotation). A key shorter than 32 characters (`settings.MIN_KEY_CHARS`), or one key given to two tenants, stops the start (`parse_api_keys`). Keys are held only as SHA-256 digests and compared in constant time against every configured key (`tenant_of`). The key names the tenant; a tenant sees its own jobs only (another tenant's job is 404). A missing or unknown key is 401.

**HTTPS.** With `REQUIRE_HTTPS` on, the API refuses with 403 any request whose `X-Forwarded-Proto` is not `https` (the `guard` middleware in `create_app`). It is on by default since 2026-11-24; `REQUIRE_HTTPS=false` is for local tests only. TLS ends at the Container Apps ingress. Every answer carries `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`. The access log is off (`api.main`).

**Refused before anything is stored:** a body over `MAX_PDF_MB` or a PDF over `MAX_PAGES` pages (413); another content type (415); a body that is not a PDF, cannot be opened, has no pages, or needs a password that was not sent (400); no `lob`, or a line with no canonical policy schema (400, `serving.vllm_entrypoint.request_lob`); a webhook that is not HTTPS, carries credentials or points into a private network (400); a webhook for a tenant with no webhook secret (400); a password in the query string (400).

**Encrypted PDFs** (`api.unlock`, `common.pdf_access`). A PDF that only restricts printing or copying is taken as it is. One that needs a password is opened with the password sent as the form field `password` or the `X-PDF-Password` header. Either is stored unencrypted; the password is never stored, logged or queued. The record says how it came: `encrypted` is `password`, `restrictions` or null.

**A result** (`Worker._result`) holds `job_id`, `status`, `final`, `full_text`, `model`, `release`, `route`, `calibration` (`release` or `identity`), `lob`, `pages`, `fields_verified`, `timings_s`, `pins`, and `output` (`ExtractionResult.as_dict()`, the output contract of IMPL-11).

**Pins** (review S1). Every result and status record carries a `pins` block: `base_model`, `release`, `prompt_hash`, `window_plan`, `format_profile`, `ocr_version`, `ocr_tier`, `calibration` (from `runtime.served_pins`, built once at start) and `schema_version` (added per job by `runtime.schema_pin`; null with a warning if it cannot be read). A base-model reading pins `release: null` and the base as `model_id@revision`. Proven by `test_every_result_and_status_names_what_it_was_read_with`.

**Webhooks** (`webhooks.py`). Optional, per job (`webhook_url`). Sent on `fields_ready`, `done` and `failed`. The body holds `event` (`extraction.<event>`), `job_id`, `status`, `status_url`, `result_url` (null unless `fields_ready` or `done`), `sent_at` and `delivery_id`; no document content. `X-Extraction-Signature: sha256=<hex>` is the HMAC-SHA256 of the exact body bytes with the tenant's secret (`WEBHOOK_SECRETS`, else `WEBHOOK_SECRET`). Receivers check it with `webhooks.verify`. `X-Extraction-Event` and `X-Extraction-Delivery` repeat the event and delivery id. The address is checked again before each send, its resolved addresses included (`post_https`); redirects are refused. Delivery is best effort: three tries (back-off 0, 2, 10 s) on a background thread. A failed delivery never fails the job; polling stays the source of truth.

## Which model and release serve

At start, `runtime.load_runtime` loads, for each tenant in `SERVED_TENANTS`:

1. The promoted release (`serving.release_router.build_serving_plan`), held to `runtime.check_release`:
   - served as `lora_on_base`, with an adapter folder (the only kind the pod's stock vLLM loads);
   - trained on the pinned MinerU's text (`scripts.served_adapter.assert_ocr_pin` against `common.config.served_mineru`);
   - packaged with this code's prompt hash and window plan, on the configured base (`serving.vllm_entrypoint.assert_prompt_hash`, `assert_window_plan_served`, `assert_base_model_served`).
2. The model: vLLM asked for `policy-<release>` (`scripts.served_adapter.lora_name`) through `inference_core.openai_backend.OpenAIChatBackend`, which refuses a name the server does not serve or an adapter on another base. One backend per served name (`runtime.openai_models`). The first one also checks how the server was started (`runtime.check_server_settings`, `OpenAIChatBackend.server_setting_problems`): the vLLM version, its context length, answer spacing, the no-whitespace rule, raw logprobs and image reading. A mismatch stops the start; a server that cannot be probed is only warned about.
3. The release calibrators and thresholds, from Blob (`runtime.load_calibrators`, `serving.vllm_entrypoint.load_release_calibration`). The adapter itself is not downloaded; it is on the pod's volume. A release with no fitted calibrators is served with raw confidence (`calibration: identity`).

`load_runtime` also refuses retired serving settings (`assert_no_retired_settings`) and builds the pinned MinerU reader (`runtime.served_reader`, `adapters.served_ocr.ServedOcr`), which refuses text from another MinerU version or tier. Any failed check is a `ColdStartError`; the worker exits with code 3 (a settings error exits with 2). Proven by `test_a_release_this_code_cannot_serve_refuses_the_cold_start`.

**Base-model fallback.** With nothing promoted, the worker refuses to start unless `ALLOW_BASE_FALLBACK=true` with `BASE_MODEL_NAME` (`runtime.load_tenant`; `Settings.from_env` refuses the first without the second). With the fallback on, a line the release did not train on is read by the base model, route `base_model_fallback`, release null. With it off (the default), such a job fails at once. DEPLOY.md says to keep it off (decided 2026-10-26).

## Settings

Read by `Settings.from_env` in `azure_service/settings.py`. A flag takes `1/true/yes/on` or `0/false/no/off`; a bad value stops the start.

| Variable | Default | What it does |
|---|---|---|
| `EXTRACTION_API_KEYS` | none | `{"tenant": ["key", ...]}`; the API does not start without it (`check_api`). |
| `REQUIRE_HTTPS` | true | Refuse requests not received over HTTPS. |
| `MAX_PDF_MB` | 150 | Larger bodies are refused (413). |
| `MAX_PAGES` | 1000 | PDFs with more pages are refused (413). |
| `PUBLIC_BASE_URL` | none | The API's address, for the webhook's links. |
| `WEBHOOK_SECRETS` | none | `{"tenant": "secret"}`: each tenant's webhook secret. |
| `WEBHOOK_SECRET` | none | The fallback webhook secret. |
| `ALLOW_PRIVATE_WEBHOOKS` | false | Allow webhooks to loopback and private addresses (tests only). |
| `STORAGE_AUTH` | `environment` | `managed_identity`, or `environment` (a SAS or connection string as `artifact_registry.blob_client.azure_from_environment` reads them). |
| `STORAGE_ACCOUNT` | none | Gives the Blob and Queue URLs; needed with `managed_identity` unless both URLs are set. |
| `BLOB_ACCOUNT_URL`, `QUEUE_ACCOUNT_URL` | from `STORAGE_ACCOUNT` | The account's endpoints, when not the standard ones. |
| `AZURE_CLIENT_ID` | none | A user-assigned identity's client id (none: the system-assigned one). |
| `JOBS_CONTAINER` | `extraction-jobs` | The jobs container. |
| `QUEUE_NAME` | `extraction-jobs` | The queue. |
| `AZURE_BLOB_CONTAINER` | none in `Settings` | The registry's container; when unset, `BlobClient` falls back to `insurance-extraction`. |
| `AZURE_RAW_CONTAINER` | none | The raw container: refused as `REVIEW_CONTAINER`; `azure_service.retention` needs it. |
| `MODEL_URL`, `MINERU_URL` | none | The pod's vLLM and MinerU; the worker needs both, over HTTPS (`check_worker`). |
| `MODEL_TOKEN`, `MINERU_TOKEN` | `FIDEON_REMOTE_TOKEN` | The two servers' tokens. |
| `SERVED_TENANTS` | `default` | Tenants whose promoted release is loaded at start. |
| `BASE_MODEL_NAME` | none | The name vLLM serves the bare base model under. |
| `ALLOW_BASE_FALLBACK` | false | Read lines no release covers with the base model. |
| `MAX_ATTEMPTS` | 3 | Deliveries of one job before it is failed. |
| `VISIBILITY_TIMEOUT_S` | 300 (min 30) | How long a received job stays hidden; renewed while it runs. |
| `RETRY_DELAY_S` | 60 | Wait before a retried job is visible again. |
| `FIELDS_DEADLINE_S` | 1800 | Longest a job's fields may take before the lease is no longer renewed; 0 is off. |
| `MODEL_REQUEST_DEADLINE_S` | 600 (min 30) | Longest one model request may take in all. |
| `MAX_CONCURRENCY` | 96 | Model requests in flight. |
| `RUN_PAGES` | 20 | Pages per MinerU run. |
| `TEXT_MAX_WAIT_S` | 600 | Longest a MinerU run waits for its GPU turn. |
| `MAX_TEXT_BACKLOG` | 4 | Jobs waiting for full text before no new job is taken. |
| `KEEP_FILES` | false | Keep PDFs, checkpoints and page images after a job. |
| `WORK_DIR` | `/tmp/extraction-jobs` | Local folder for page files. |
| `PAGE_IMAGES` | `inline` | `inline` (base64 in each request) or `link` (pages in Blob, read by link). |
| `PAGE_LINK_SAS_TOKEN` | none | Read-only SAS for page links; needed with `link` and a managed identity. |
| `REVIEW_RETENTION_TENANTS` | empty (off) | Tenants whose flagged jobs may be kept for review. |
| `REVIEW_RETENTION_SAMPLE` | 0.05 | Share of unflagged jobs kept anyway. |
| `REVIEW_RETENTION_LOW_CONFIDENCE` | 0.80 | Overall confidence below this keeps a job. |
| `REVIEW_RETENTION_FLAG_KINDS` | `settings.DEFAULT_RETENTION_FLAG_KINDS` (22 kinds) | Review-flag kinds that keep a job. |
| `REVIEW_RETENTION_DAYS` | 90 | Days a retained document is kept. |
| `REVIEW_CONTAINER` | `review-retention` | Must differ from the jobs, registry and raw containers. |

Read elsewhere: `API_PORT` (default 8080, `api.main`); `AZURE_STORAGE_CONNECTION_STRING` and `AZURE_QUEUE_SAS_TOKEN` (the queue login with `STORAGE_AUTH=environment`, `store.job_queue`). `Settings.describe` is what is logged at start; secrets are left out of it and of `repr`.

## Failures

`azure_service/RUNBOOK.md` says, case by case, what the caller sees and what to do. Case 1: the model server stops answering, before or during a job; a window that does not reach the model is sent once more, then the whole job is retried from the start (`serving.pipeline._settle`), never served with pages missing (`test_a_model_server_that_cannot_be_reached_puts_the_job_back`). Case 2: the worker stops, between phases, during the fields, on purpose, or past the fields deadline. Case 3: a PDF that fails on every delivery. Case 4: the model answered but some windows were unreadable even one page at a time; only these are served flagged `window_failed:`, with `schema_valid` false (`test_an_answer_the_model_wrote_badly_is_served_flagged_and_listed`). It also covers a line no release serves, MinerU text failing for good, and a prompt-length check that could not run. Two commands list the jobs that need a person, with no document name or value: `python -m azure_service.jobs failed` (status `failed`) and `python -m azure_service.jobs partial` (a result with `window_failed:` flags).

## The API contract

`azure_service/openapi.json` is generated from the app itself, over in-memory stand-ins, by `python -m scripts.export_openapi`. `--check` exits 1 when the committed file differs from the app. `tests/test_api_contract.py` fails on a stale file and checks that the four routes are listed. The app's own docs pages stay off (`docs_url=None`). The contract lists routes and path parameters only; the POST's query and form fields are read from the raw request, so they are not in it.

## What the image contains

`azure_service/Dockerfile` builds two targets from one base: `api` (`python -m azure_service.api`, port 8080) and `worker` (`python -m azure_service.worker`). Base `python:3.11-slim`, run as uid 10001. Dependencies: `requirements-service.lock`, installed with `pip install --no-deps`. No GPU, torch, vLLM or MinerU.

COPY lines: `pyproject.toml`, `requirements-service.lock`, `common/`, `contracts/` and `adapters/` (since 2026-11-24, review A1), `artifact_registry/`, `registry_utils/`, `data_pipeline/`, `inference_core/`, `calibration/`, `serving/`, `postprocessing/`, `evaluation/`, `orchestration/`, `configs/`, `schemas/`, `prompts/`, `azure_service/`.

`training/`, `testing/`, `pilot/`, `data/` and `tests/` are not in it. The served packages (`common`, `contracts`, `adapters`, `inference_core`, `calibration`, `serving`, `azure_service`) import nothing from the build-time ones beyond the few `tests/test_layering.py` still allows, each named there; until that list is empty the image keeps `data_pipeline/`, `evaluation/` and `orchestration/`. `scripts/` left it on 2026-11-24. `tests/test_production_image_imports.py` runs the served code with every package the image leaves out blocked, so served code that imports one fails on a laptop too (`test_the_dockerfile_leaves_training_and_testing_out`). `azure_service/Dockerfile.dockerignore` and the root `.dockerignore` exclude everything else from the build context (`test_the_root_dockerignore_leaves_out_what_the_service_one_does`). DEPLOY.md says to build from a clean `git archive` export, never from the working tree.

## Open

Only what RUNBOOK.md and DEPLOY.md themselves mark open:

- Never deployed. No RUNBOOK case has been rehearsed on real Azure; rehearse each once on the first deployment (RUNBOOK.md).
- How long results are kept: the 30-day lifecycle rule on `jobs/` is "a retention decision to confirm" (DEPLOY.md section 3).
- Where TLS ends for RunPod proxy URLs: to confirm with RunPod (DEPLOY.md section 7).
- Serving a merged model instead of the Qwen3-VL LoRA fix: "an open decision" (DEPLOY.md section 7).
- The ingress request timeout against the largest PDFs on the slowest client link: to check (DEPLOY.md section 6).
- Retention for review stays off until the legal and contract sign-off in DEPLOY.md section 9.
