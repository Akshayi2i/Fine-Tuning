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
| `POST /v1/extractions?lob=<line>[&webhook_url=https://...][&name=a.pdf]` - body: the PDF (`application/pdf`), or `multipart/form-data` with `file` (`lob`, `webhook_url`, `name` as form fields) | `202 {"job_id", "status": "queued", "status_url", "result_url"}` |
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
```

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

## 9. Running it locally

Tests (no Azure, no network, no GPU):

```bash
pip install -e ".[data,service,dev]"
pytest -q tests/test_azure_service.py tests/test_azure_service_runtime.py
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
