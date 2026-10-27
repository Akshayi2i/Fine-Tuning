# Deploying the extraction service on Azure Container Apps

The production extraction routine runs on Azure; RunPod serves only a stock vLLM
(Qwen3-VL-8B with the promoted release's LoRA, `policy-vN`) and a MinerU 4 API,
both reached over HTTPS. **No code of ours runs on RunPod.**

```
client --HTTPS--> extraction-api (Container App, external ingress)
                    |  PDF -> Blob (jobs container, transient/), record -> Blob, job id -> Storage Queue
                    v
                  Storage Queue  --->  extraction-worker (Container App, no ingress, 1 replica per GPU)
                                          | phase 1: page images -> vLLM (policy-vN)      --HTTPS--> RunPod H100
                                          | phase 2: PDF runs   -> MinerU 4 API (pinned)  --HTTPS--> (same pod)
                                          v
                                        Blob: status.json / result.json  (+ signed webhook, no content)
```

## 1. What it is

| Endpoint | Answer |
|---|---|
| `POST /v1/extractions?lob=<line>[&webhook_url=https://...][&name=a.pdf]` - body: the PDF (`application/pdf`), or `multipart/form-data` with `file` (`lob`, `webhook_url`, `name`, `password` as form fields); a PDF that needs a password to open comes with it as the form field `password` or the header `X-PDF-Password` | `202 {"job_id", "status": "queued", "status_url", "result_url"}` |
| `GET /v1/extractions/{job_id}` | the job: `status` (`queued`, `running`, `fields_ready`, `done`, `failed`), `timings_s`, pages, model, release, attempts, error |
| `GET /v1/extractions/{job_id}/result` | `fields_ready`: the checked fields (`final: false`); `done`: the final JSON (`final: true`); otherwise `409` |
| `GET /healthz` | liveness (no key) |

Every call but `/healthz` needs `Authorization: Bearer <key>` (or `X-API-Key`). The
key names the tenant; a tenant sees its own jobs only.

```bash
curl -sS -X POST "https://$API_FQDN/v1/extractions?lob=personal_auto" \
     -H "Authorization: Bearer $KEY" -H "Content-Type: application/pdf" --data-binary @policy.pdf
curl -sS -H "Authorization: Bearer $KEY" "https://$API_FQDN/v1/extractions/$JOB"
curl -sS -H "Authorization: Bearer $KEY" "https://$API_FQDN/v1/extractions/$JOB/result"
# A PDF that needs a password to open: the password beside it, never in the URL
curl -sS -X POST "https://$API_FQDN/v1/extractions" -H "Authorization: Bearer $KEY" \
     -F lob=personal_auto -F file=@locked-policy.pdf -F "password=$PDF_PASSWORD"
```

**Encrypted PDFs** (`common/pdf_access.py`). A PDF that only restricts printing or copying is
taken as it is. One that needs a password to open is read with the password sent beside it
(form field `password` or header `X-PDF-Password`); without it, or with a wrong one, the
request is refused (400) and nothing is stored. A password in the query string is refused:
URLs are logged on the way. Either kind is stored **unencrypted** in the jobs container - the
same handling as every other PDF, deleted with the job (and kept for review only if
`REVIEW_RETENTION_*` is on) - and the job's status says how it came (`encrypted`:
`password`, `restrictions` or null). The password is used once, in the API, and is never
stored, logged, queued or echoed. Nothing tries to guess or break a password.

A webhook (optional, `webhook_url`, HTTPS only) is POSTed on `fields_ready`, `done`
and `failed`. Its body carries the job id, status and links - never document
content - and `X-Extraction-Signature: sha256=<hex>` is HMAC-SHA256 of the raw body
with the tenant's webhook secret. A receiver checks it like this, and refuses a
`sent_at` older than a few minutes:

```python
from azure_service.webhooks import verify     # or: hmac.compare_digest(sign(secret, body), header)
ok = verify(secret, request_body_bytes, request.headers["X-Extraction-Signature"])
```

**Storage layout** (the jobs container, `JOBS_CONTAINER`, default `extraction-jobs`):

```
jobs/<tenant>/<job id>/status.json             state and timings - no document content
jobs/<tenant>/<job id>/result.json             the served JSON (fields, then final)
transient/jobs/<tenant>/<job id>/document.pdf  the PDF, deleted when the job ends (KEEP_FILES keeps it)
transient/jobs/<tenant>/<job id>/phase1.json   the model's answer, for phase 2 after a restart; deleted at the end
transient/page-links/...                       page images, only with PAGE_IMAGES=link; deleted after phase 1
```

Nothing else is kept. Retention for review (section 9) is **off by default**; when
a tenant is opted in after legal sign-off, its flagged documents are copied to a
separate container (`REVIEW_CONTAINER`) before the PDF above is deleted.

**Restarts.** The queue redelivers a job whose worker died (after `VISIBILITY_TIMEOUT_S`).
A job whose phase-1 checkpoint exists (`fields_ready`) **resumes phase 2** - MinerU
reads its pages, no model time is spent again, the served fields stay as they
were. A job without one (the worker died during phase 1) is **read again from the
start**. After `MAX_ATTEMPTS` deliveries a job is given up (poison): `failed`, or
`done` with `full_text: false` when its checked fields were already served.

## 2. Prerequisites

* The existing storage account (the registry's: release index, bundles,
  calibrators). The service adds one container and one queue to it.
* A Key Vault, an Azure Container Registry, a Log Analytics workspace.
* A RunPod pod (see section 7) serving vLLM and MinerU over HTTPS, with tokens.
* `az` with the `containerapp` extension: `az extension add -n containerapp --upgrade`.

```bash
RG=rg-fideon-extraction           LOCATION=eastus2
ACR=fideonacr                     ENV=fideon-extraction-env
STORAGE=fideonstore               REGISTRY_CONTAINER=insurance-extraction   # the registry's container
KV=fideon-extraction-kv           ID_NAME=extraction-identity
TAG=$(git rev-parse --short HEAD)
```

## 3. Storage: container, queue, lifecycle

```bash
az storage container create --account-name $STORAGE --name extraction-jobs --auth-mode login
az storage queue create     --account-name $STORAGE --name extraction-jobs --auth-mode login
```

The jobs container is its own, not the raw-documents container (serving may not read
that one, `artifact_registry.blob_client`) and not the registry's: its RBAC can be
granted to the service alone. A lifecycle rule removes what a crash left behind
and keeps results only as long as agreed (30 days below - a retention decision to
confirm):

```bash
cat > lifecycle.json <<'EOF'
{"rules": [
  {"enabled": true, "name": "transient-1d", "type": "Lifecycle", "definition": {
     "filters": {"blobTypes": ["blockBlob"], "prefixMatch": ["extraction-jobs/transient/"]},
     "actions": {"baseBlob": {"delete": {"daysAfterModificationGreaterThan": 1}}}}},
  {"enabled": true, "name": "results-30d", "type": "Lifecycle", "definition": {
     "filters": {"blobTypes": ["blockBlob"], "prefixMatch": ["extraction-jobs/jobs/"]},
     "actions": {"baseBlob": {"delete": {"daysAfterModificationGreaterThan": 30}}}}}
]}
EOF
az storage account management-policy create --account-name $STORAGE -g $RG --policy @lifecycle.json
```

(`management-policy create` replaces the account's whole policy: merge these rules
into an existing one rather than overwrite it.)

## 4. Identity, secrets, roles

One user-assigned identity for both apps (or one each, for least privilege: the API
needs no registry read and only *Message Sender* on the queue).

```bash
az identity create -g $RG -n $ID_NAME
ID=$(az identity show -g $RG -n $ID_NAME --query id -o tsv)
ID_CLIENT=$(az identity show -g $RG -n $ID_NAME --query clientId -o tsv)
ID_PRINCIPAL=$(az identity show -g $RG -n $ID_NAME --query principalId -o tsv)
SA=$(az storage account show -n $STORAGE -g $RG --query id -o tsv)

# Blob: read/write the jobs container only; read the registry's container.
az role assignment create --assignee-object-id $ID_PRINCIPAL --assignee-principal-type ServicePrincipal \
   --role "Storage Blob Data Contributor" --scope "$SA/blobServices/default/containers/extraction-jobs"
az role assignment create --assignee-object-id $ID_PRINCIPAL --assignee-principal-type ServicePrincipal \
   --role "Storage Blob Data Reader" --scope "$SA/blobServices/default/containers/$REGISTRY_CONTAINER"
# Queue: send (API), receive/update/delete (worker).
az role assignment create --assignee-object-id $ID_PRINCIPAL --assignee-principal-type ServicePrincipal \
   --role "Storage Queue Data Message Sender"    --scope "$SA/queueServices/default/queues/extraction-jobs"
az role assignment create --assignee-object-id $ID_PRINCIPAL --assignee-principal-type ServicePrincipal \
   --role "Storage Queue Data Message Processor" --scope "$SA/queueServices/default/queues/extraction-jobs"
# Key Vault secrets, and image pulls.
az role assignment create --assignee-object-id $ID_PRINCIPAL --assignee-principal-type ServicePrincipal \
   --role "Key Vault Secrets User" --scope $(az keyvault show -n $KV --query id -o tsv)
az role assignment create --assignee-object-id $ID_PRINCIPAL --assignee-principal-type ServicePrincipal \
   --role AcrPull --scope $(az acr show -n $ACR --query id -o tsv)
```

The Processor role covers update (lease renewal) and delete. With
`STORAGE_AUTH=managed_identity` no storage key or SAS exists anywhere. The
alternative (`STORAGE_AUTH=environment`) is a SAS or connection string as
`artifact_registry.blob_client.azure_from_environment` reads them, plus
`AZURE_QUEUE_SAS_TOKEN` (or `AZURE_STORAGE_CONNECTION_STRING`) for the queue.

Secrets in Key Vault (never in an image, a YAML file or a log):

```bash
# {"tenant": ["key", "next key during a rotation"]}; keys of 32+ characters.
az keyvault secret set --vault-name $KV -n extraction-api-keys \
   --value "{\"default\": [\"$(python -c 'import secrets; print(secrets.token_urlsafe(32))')\"]}"
az keyvault secret set --vault-name $KV -n webhook-secret --value "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
az keyvault secret set --vault-name $KV -n model-token  --value "<vLLM --api-key>"
az keyvault secret set --vault-name $KV -n mineru-token --value "<MinerU API token>"
```

Several tenants: `WEBHOOK_SECRETS` = `{"tenant": "secret"}` gives each its own
webhook secret (one shared secret would let one tenant forge another's webhooks);
`WEBHOOK_SECRET` is the fallback. A tenant with neither cannot ask for webhooks.

## 5. Images

Build from a **clean export of a commit** - never from the working tree, which holds
client documents (`data/`, `Test Data/`, `testing/`). `git archive` ships tracked
files only (PDFs are git-ignored); the Dockerfile copies named code directories
only, and `azure_service/Dockerfile.dockerignore` narrows a BuildKit context too.

```bash
rm -rf /tmp/extraction-build && mkdir /tmp/extraction-build
git archive HEAD | tar -x -C /tmp/extraction-build
az acr build -r $ACR -f azure_service/Dockerfile --target api    -t extraction-api:$TAG    /tmp/extraction-build
az acr build -r $ACR -f azure_service/Dockerfile --target worker -t extraction-worker:$TAG /tmp/extraction-build
```

(Or `docker buildx build -f azure_service/Dockerfile --target api ...` and `docker push`.)
Everything the pipeline reads at run time (configs, schemas, prompts) must be
committed: an untracked file is not in the image.

The image installs exactly `requirements-service.lock` (`pip install --no-deps`): every
package pinned for Linux x86-64 and Python 3.11, with the training lock's pillow and
PyMuPDF (the worker renders the pages the model reads), so two builds a month apart are
the same. After changing a dependency in `pyproject.toml`, regenerate the locks from any
machine with `uv` and commit them: `bash scripts/lock_requirements.sh`.
`tests/test_dependency_locks.py` fails when the service lock no longer satisfies
`pyproject.toml` or its pixel stack drifts from training's.

## 6. Container Apps

```bash
az containerapp env create -g $RG -n $ENV -l $LOCATION \
   --logs-workspace-id <workspace id> --logs-workspace-key <workspace key>
KVURI=https://$KV.vault.azure.net/secrets
SECRETS_COMMON=(
  "webhook-secret=keyvaultref:$KVURI/webhook-secret,identityref:$ID"
)
ENV_COMMON=(
  STORAGE_AUTH=managed_identity STORAGE_ACCOUNT=$STORAGE AZURE_CLIENT_ID=$ID_CLIENT
  JOBS_CONTAINER=extraction-jobs QUEUE_NAME=extraction-jobs AZURE_BLOB_CONTAINER=$REGISTRY_CONTAINER
  WEBHOOK_SECRET=secretref:webhook-secret
)
```

**The API** - external HTTPS ingress, the only app with one. `allowInsecure` stays
false (HTTP is redirected to HTTPS), and `REQUIRE_HTTPS=true` refuses any request
the ingress did not receive over HTTPS.

```bash
az containerapp create -g $RG -n extraction-api --environment $ENV \
   --image $ACR.azurecr.io/extraction-api:$TAG --registry-server $ACR.azurecr.io --registry-identity $ID \
   --user-assigned $ID --ingress external --target-port 8080 --transport http \
   --cpu 1 --memory 2Gi --min-replicas 1 --max-replicas 5 \
   --scale-rule-name http --scale-rule-type http --scale-rule-http-concurrency 20 \
   --secrets "${SECRETS_COMMON[@]}" "api-keys=keyvaultref:$KVURI/extraction-api-keys,identityref:$ID" \
   --env-vars "${ENV_COMMON[@]}" EXTRACTION_API_KEYS=secretref:api-keys REQUIRE_HTTPS=true MAX_PDF_MB=150
API_FQDN=$(az containerapp show -g $RG -n extraction-api --query properties.configuration.ingress.fqdn -o tsv)
az containerapp update -g $RG -n extraction-api --set-env-vars PUBLIC_BASE_URL=https://$API_FQDN
```

Restrict who may call it with ingress IP restrictions
(`az containerapp ingress access-restriction set`) or a front door / APIM in front.
An upload is held until the PDF is in Blob, then answered at once; check the
ingress' request timeout against your largest PDFs on your slowest client link.

**The worker** - no ingress, **exactly one replica per GPU**:

```bash
az containerapp create -g $RG -n extraction-worker --environment $ENV \
   --image $ACR.azurecr.io/extraction-worker:$TAG --registry-server $ACR.azurecr.io --registry-identity $ID \
   --user-assigned $ID --cpu 4 --memory 8Gi --min-replicas 1 --max-replicas 1 \
   --secrets "${SECRETS_COMMON[@]}" \
       "model-token=keyvaultref:$KVURI/model-token,identityref:$ID" \
       "mineru-token=keyvaultref:$KVURI/mineru-token,identityref:$ID" \
   --env-vars "${ENV_COMMON[@]}" PUBLIC_BASE_URL=https://$API_FQDN \
       MODEL_URL=https://<pod id>-8000.proxy.runpod.net MODEL_TOKEN=secretref:model-token \
       MINERU_URL=https://<pod id>-8001.proxy.runpod.net MINERU_TOKEN=secretref:mineru-token \
       SERVED_TENANTS=default MAX_ATTEMPTS=3 VISIBILITY_TIMEOUT_S=300 RETRY_DELAY_S=60
```

### Settings

| Variable | App | Default | Meaning |
|---|---|---|---|
| `EXTRACTION_API_KEYS` | api | - | `{"tenant": ["key", ...]}` (secret) |
| `MAX_PDF_MB`, `MAX_PAGES` | api | 150, 1000 | refused above (413) |
| `REQUIRE_HTTPS` | api | false | refuse requests not received over HTTPS (set true) |
| `API_PORT` | api | 8080 | the port the API listens on (the ingress's target port) |
| `PUBLIC_BASE_URL` | both | - | `https://<api fqdn>`, for the webhook's links |
| `WEBHOOK_SECRET`, `WEBHOOK_SECRETS` | both | - | the webhook HMAC secret; per tenant (secrets) |
| `ALLOW_PRIVATE_WEBHOOKS` | both | false | webhooks to private/loopback addresses (never in production) |
| `STORAGE_AUTH` | both | environment | `managed_identity`, or a SAS/connection string from the environment |
| `STORAGE_ACCOUNT`, `AZURE_CLIENT_ID` | both | - | the account (URLs derived), the identity's client id |
| `BLOB_ACCOUNT_URL`, `QUEUE_ACCOUNT_URL` | both | derived from `STORAGE_ACCOUNT` | the account's Blob and Queue endpoints, when not the standard ones (Azurite, a private endpoint) |
| `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_QUEUE_SAS_TOKEN` | both | - | `STORAGE_AUTH=environment`: the connection string, or a SAS for the queue (secrets) |
| `JOBS_CONTAINER`, `QUEUE_NAME` | both | extraction-jobs | where jobs live |
| `AZURE_BLOB_CONTAINER` | worker | insurance-extraction | the registry's container |
| `MODEL_URL`, `MODEL_TOKEN` | worker | - | the pod's vLLM (HTTPS), its `--api-key` |
| `MINERU_URL`, `MINERU_TOKEN` | worker | - | the pod's MinerU 4 API (HTTPS), its token |
| `SERVED_TENANTS` | worker | default | tenants whose promoted release is loaded at start |
| `ALLOW_BASE_FALLBACK`, `BASE_MODEL_NAME` | worker | false, - | read a line no release covers with the base model (vLLM's `--served-model-name`). Keep it off (decided 2026-10-26): the base model never learned the format or the schema, so a line the release does not cover is refused, never answered unreliably |
| `MAX_ATTEMPTS` | worker | 3 | deliveries before a job is given up |
| `VISIBILITY_TIMEOUT_S`, `RETRY_DELAY_S` | worker | 300, 60 | lease length (renewed every third of it); wait before a retry |
| `KEEP_FILES` | worker | false | keep PDFs, checkpoints and page images after a job (debugging only) |
| `PAGE_IMAGES`, `PAGE_LINK_SAS_TOKEN` | worker | inline | `link`: pages put in Blob, vLLM fetches them by read-only link |
| `MAX_CONCURRENCY`, `RUN_PAGES`, `TEXT_MAX_WAIT_S`, `MAX_TEXT_BACKLOG` | worker | 96, 20, 600, 4 | windows in flight; MinerU run size; longest wait for a GPU turn; jobs waiting for their full text before no new job is taken |
| `REVIEW_RETENTION_TENANTS` | worker, retention CLI | empty = **off** | tenants whose flagged, low-confidence and sampled documents are kept for review (section 9). Set only after the sign-off there |
| `REVIEW_RETENTION_SAMPLE` | worker | 0.05 | share of an opted-in tenant's unflagged jobs kept anyway (hash of the job id) |
| `REVIEW_RETENTION_LOW_CONFIDENCE` | worker | 0.80 | an overall confidence below this keeps the job |
| `REVIEW_RETENTION_FLAG_KINDS` | worker | `low_confidence,no_confidence,not_printed,not_printed_removed,parsed_mismatch,role_mismatch,printed_off_cited_page,near_tie,likely_missed,unverified_unread_page,link_to_removed_row,row_count_mismatch,window_failed` | review-flag kinds (suffix after the flag's last `:`) that keep a job; document-level flags such as `uncalibrated` or `schema:invalid` never do alone |
| `REVIEW_RETENTION_DAYS` | worker | 90 | days a retained document is kept before `retention purge` deletes it |
| `REVIEW_CONTAINER` | worker, retention CLI | review-retention | where retained documents go: a container of its own (the jobs, registry or raw one is refused) |

### Why one worker replica per GPU

The worker holds the GPU's turns in process (`serving.fields_first.GpuTurns`): MinerU
starts a run of pages only while no document's fields are being read, so the fields
of a new document wait at most one 20-page run. Two worker replicas on one GPU would
each believe the GPU idle and run MinerU beside the other's fields - the
fields-first gate (a 200-page scan's fields in about 2 minutes) and the GPU's memory
are then no longer under control. So:

* **min = max = 1 replica** per worker app. Do not add a queue-length (KEDA) scale
  rule to it: more replicas against the same pod make it slower, not faster.
* **More throughput = more GPUs**: one RunPod H100 pod per worker app
  (`extraction-worker-a` -> pod A, `-b` -> pod B), each with its own `MODEL_URL` and
  `MINERU_URL`, all on the same queue. The queue spreads the jobs.
* The API is stateless and scales on HTTP concurrency.
* On a scale-in or a new revision the worker gets SIGTERM: it takes no new job and
  makes its jobs visible again at once, for the next replica (a job read again from
  phase 1, or from phase 2 when its checkpoint exists).

### Cold start

The worker refuses to start (the log says why) unless the tenant's promoted
release is a LoRA on the base (`lora_on_base`), trained on the pinned MinerU's text,
packaged with this code's prompt hash and window plan, on the configured base, and
vLLM serves `policy-<release>` on our base - or, with nothing promoted,
`ALLOW_BASE_FALLBACK` names the base. A new release therefore means: promote it,
put its adapter on the pod (`python -m scripts.served_adapter`), restart vLLM with
the printed `--lora-modules`, then restart the worker:
`az containerapp revision restart -g $RG -n extraction-worker --revision <name>`.
Deploy a worker image built from the commit the release was packaged from.

## 7. RunPod

* **Secure Cloud only, with RunPod's Data Processing Agreement signed.** Documents
  leave Azure: page images go to vLLM and PDFs to MinerU on the pod. Our code keeps
  nothing there, but MinerU's task output and vLLM's media cache live on the pod's
  disk and volume - configure them to be removed after use and keep the volume
  encrypted.
* **A data centre near the Azure region** (same geography: a US-East pod for East
  US 2): every window of every page travels Azure -> RunPod; distance is latency
  on each of ~100 requests per document, and keeps the data in one jurisdiction.
* Both servers behind tokens (`vllm serve --api-key`, MinerU's token) and reached
  over HTTPS (the RunPod proxy URL). The worker refuses plain HTTP to anything but
  localhost (`common.transport.assert_encrypted`) and refuses redirects. Confirm
  with RunPod where TLS ends for proxy URLs.
* vLLM serves the base with the release's LoRA (`--enable-lora --max-lora-rank ..
  --lora-modules policy-vN=/workspace/adapters/policy-vN`, printed by
  `scripts/served_adapter.py`); MinerU is the pinned version and tier
  (`configs/shared/ocr.yaml`) - the worker refuses text from any other.

## 8. Operating

* **Logs** carry job ids, tenants, sizes, page counts, timings and error types -
  never document content, file names in errors, webhook addresses or keys. The
  API's access log is off (a query names the document).
* **Watch**: the queue's length and oldest message, `failed` jobs, worker restarts,
  `fields_ready` times against the 2-minute gate.
* **Rotating an API key**: add the new key beside the old in the Key Vault secret,
  create a new revision (or restart) so the env refreshes, move clients, remove the old.

## 9. Retention for review (off by default)

Decision 2026-10-26 (#43): flagged production documents can be kept so a person
can label them for the feedback loop. **The code ships disabled.** With
`REVIEW_RETENTION_TENANTS` empty - the default - nothing is kept for anyone, and
the PDF is deleted when its job ends, as before. The API shows nothing of it to
clients: no parameter, no field in a status or result. A request cannot opt a
tenant in; only this setting can.

### Before setting a tenant: legal and contract sign-off

These are insurance policies: names, addresses, dates of birth, driver's
licence numbers, VINs, claims. Keeping them beyond the job, and using them to
train, needs written approval **before** a tenant is named:

* **The client's contract and DPA** allow us to keep their documents after the
  extraction, for this purpose (quality review, labelling, training a model), for
  `REVIEW_RETENTION_DAYS` days, in this Azure region. Where the DPA needs it, the
  client's consent, and their customers' notice (privacy notices) cover it.
* **Training use**: whether a client's documents may train the model that serves
  other clients (one shared policy adapter), and under which tenant they are
  imported. If not, the export stops at review and nothing is imported.
* **Who may see them** (below), and the labellers' agreements (confidentiality,
  where they work, devices).
* **Deletion**: the retention period, deletion on request (`purge --tenant T --all`
  when consent is withdrawn or the contract ends) and on expiry, and that Blob
  soft delete or versioning, if the account has them, keep deleted blobs for
  their own period.
* **Sub-processors and location**: the review container is in the same storage
  account and region as the jobs; an export to a laptop moves documents out of
  Azure (prefer `intake/<batch>`, below).

Record the sign-off (who, when, which tenants, which period) with the change that
sets `REVIEW_RETENTION_TENANTS`.

### What is kept, and for how long

When a job of a named tenant ends `done`, the worker keeps it - before the PDF is
deleted - if **any** of these holds (`azure_service/retention.py`):

* `flagged`: a served value is flagged (`"flagged": true`, unless only because the
  release had no calibrator for it) or the result has a review flag of a kind in
  `REVIEW_RETENTION_FLAG_KINDS` - a value that needs a person: `low_confidence`,
  `no_confidence`, `not_printed`, `not_printed_removed`, `parsed_mismatch`,
  `role_mismatch`, `printed_off_cited_page`, `near_tie`, `likely_missed`,
  `unverified_unread_page`, `link_to_removed_row`, and data that may be missing:
  `row_count_mismatch`, `window_failed`. A document-level system flag
  (`uncalibrated`, `schema:invalid`, `printed_values:no_page_text`) alone keeps
  nothing, or nearly every document would be kept. The manifest names the kinds
  that kept it (`triggered_kinds`);
* `low_confidence`: the overall confidence is below `REVIEW_RETENTION_LOW_CONFIDENCE` (0.80), or there is none;
* `sample`: otherwise, `REVIEW_RETENTION_SAMPLE` (5%) of jobs, chosen by a hash of the job id.

```
review-retention/retained/<tenant>/<job id>/document.pdf   the PDF as the client sent it
review-retention/retained/<tenant>/<job id>/result.json    the final result as served
review-retention/retained/<tenant>/<job id>/manifest.json  tenant, job id, line, reasons, triggered kinds, flag counts and kinds,
                                                           overall confidence, model and release, pages, bytes,
                                                           retained_at, expires_at, state - no file name, no value
```

Failed jobs and other tenants' jobs are never kept. A copy that fails is logged
(job id and error type only) and never fails the job. Kept for
`REVIEW_RETENTION_DAYS` (90) after `retained_at`, or until it is marked reviewed,
whichever comes first.

**Encryption.** Azure Storage encrypts every blob at rest (AES-256,
Microsoft-managed keys) and the service reaches it over HTTPS only. Where the
contract asks for customer-managed keys, give the container an encryption scope
backed by a Key Vault key:

```bash
az storage account encryption-scope create --account-name $STORAGE -g $RG -n review-cmk \
   --key-source Microsoft.KeyVault --key-uri https://$KV.vault.azure.net/keys/review-retention
# (the account's identity needs get/wrapKey/unwrapKey on the key) - then create the container with it:
az storage container create --account-name $STORAGE --name review-retention --auth-mode login \
   --default-encryption-scope review-cmk --prevent-encryption-scope-override true
```

### Setting it up

```bash
az storage container create --account-name $STORAGE --name review-retention --auth-mode login
# The worker writes it; nobody else by default. (Give the worker its own identity when this
# is on: with one identity for both apps, the API could read retained documents too.)
az role assignment create --assignee-object-id $ID_PRINCIPAL --assignee-principal-type ServicePrincipal \
   --role "Storage Blob Data Contributor" --scope "$SA/blobServices/default/containers/review-retention"
# The people who review and label: a named security group, read only.
az role assignment create --assignee-object-id <reviewers group id> --assignee-principal-type Group \
   --role "Storage Blob Data Reader" --scope "$SA/blobServices/default/containers/review-retention"
# Then, after the sign-off, on the worker:
az containerapp update -g $RG -n extraction-worker \
   --set-env-vars REVIEW_RETENTION_TENANTS=<tenant>[,<tenant>] REVIEW_RETENTION_DAYS=90
```

**Who can read it (RBAC).** Only the worker's identity (Contributor, to write)
and the reviewers' group (Reader) - never the API's identity, the training pod's
SAS or a SAS for the jobs container. The operator who runs `purge`, `export` and
`mark-reviewed` needs Contributor on the container, and for an export to
`intake/<batch>` the same write access to the raw container's `intake/` a laptop
uploads deliveries with. Turn on the storage account's diagnostic logs
(`StorageRead`, `StorageWrite`, `StorageDelete` to Log Analytics) so every read
of a retained document is on record. With `STORAGE_AUTH=environment` the login
must reach this container too (a SAS for the jobs container alone cannot).

At start the worker writes and deletes a probe in the container; if it cannot,
it logs `retention for review is OFF` and reads documents as usual, keeping
nothing.

### Purge, export, review

```bash
# Daily (a Container Apps job on a cron, or by hand): expired, reviewed and incomplete items.
python -m azure_service.retention purge [--tenant T] [--dry-run]
# Consent withdrawn or the contract ended: every item of a tenant, whatever its state.
python -m azure_service.retention purge --tenant T --all
# Copy retained documents into the labelling intake as UNLABELLED documents (oldest first).
python -m azure_service.retention export --tenant T --out intake/review-2026-11 --limit 50
python -m azure_service.retention export --tenant T --out ./review-2026-11       # a local folder
# A person has dealt with one (labelled, imported, or judged not worth a label): purge deletes it next.
python -m azure_service.retention mark-reviewed --tenant T --job <job id>
```

`export` writes one folder per document, the layout the audit and
`import_labeled_pdfs` read: `<tenant>__<job id>/document.pdf`, `metadata.json`
(`lob`, `synthetic: false`, the tenant, the job, why it was kept, no review
record) and `draft.json` (the served answer, in `pre_annotate`'s draft shape:
a starting point, never a label). **It never writes `golden.json`.** A person
labels it (writes `golden.json`, and adds `reviewer_id` and `review_date` to
`metadata.json`); the audit and the import refuse the folder until then (every
gold needs a real person's review record). `--out intake/<batch>` writes to the raw container
(`AZURE_RAW_CONTAINER`) for `pull_intake --batch <batch>`; any other value is a
local folder, which then holds client documents on that machine. Only a tenant
in `REVIEW_RETENTION_TENANTS` is exported, so run it with the worker's setting.
An exported item is marked `exported` and is not exported again; it stays until
it is marked reviewed or expires. The intake copy is outside this purge: it
follows the intake's own rules (`pull_raw_pdfs --purge-plan` finds it once imported).

A scheduled purge, as a Container Apps job with the worker's image and identity:

```bash
az containerapp job create -g $RG -n retention-purge --environment $ENV --trigger-type Schedule \
   --cron-expression "30 3 * * *" --replica-timeout 1800 --replica-retry-limit 1 \
   --image $ACR.azurecr.io/extraction-worker:$TAG --registry-server $ACR.azurecr.io --registry-identity $ID \
   --mi-user-assigned $ID --cpu 0.5 --memory 1Gi \
   --command "/bin/sh" --args "-c" "python -m azure_service.retention purge" \
   --env-vars STORAGE_AUTH=managed_identity STORAGE_ACCOUNT=$STORAGE AZURE_CLIENT_ID=$ID_CLIENT \
       REVIEW_CONTAINER=review-retention
```

**The lifecycle rule is the backstop**: should purge stop running, Azure deletes
retained blobs after 90 days anyway (match it to `REVIEW_RETENTION_DAYS`). Merge
it into the account's policy beside the rules of section 3:

```json
{"enabled": true, "name": "review-retention-90d", "type": "Lifecycle", "definition": {
   "filters": {"blobTypes": ["blockBlob"], "prefixMatch": ["review-retention/retained/"]},
   "actions": {"baseBlob": {"delete": {"daysAfterModificationGreaterThan": 90}}}}}
```

## 10. Running it locally

Tests (no Azure, no network, no GPU):

```bash
pip install -e ".[data,service,dev]"
pytest -q tests/test_azure_service.py tests/test_azure_service_runtime.py tests/test_decision_retention.py
```

Against Azurite and a pod:

```bash
docker run -d -p 10000:10000 -p 10001:10001 mcr.microsoft.com/azure-storage/azurite
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
az storage container create -n extraction-jobs --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
az storage queue create     -n extraction-jobs --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
export EXTRACTION_API_KEYS='{"default": ["a-local-key-of-at-least-32-characters"]}' WEBHOOK_SECRET=local
export MODEL_URL=https://<pod>-8000.proxy.runpod.net MINERU_URL=https://<pod>-8001.proxy.runpod.net
export MODEL_TOKEN=... MINERU_TOKEN=... PUBLIC_BASE_URL=http://localhost:8080 WORK_DIR=./work
# with no promoted release in Azurite's registry container:
export ALLOW_BASE_FALLBACK=true BASE_MODEL_NAME=<vLLM served name of the base>
python -m azure_service.api &       # http://localhost:8080
python -m azure_service.worker
```
