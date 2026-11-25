# Go-live checklist: where documents travel

Decision 2026-11-25 (review S9): **two clouds**. Documents live on Azure; page images and PDFs
travel to the model servers on RunPod. Some conditions of that the code enforces itself; the rest
it cannot, so a named person confirms each before the first client document is sent, and again
whenever the pod, the region or the RunPod account changes. The choice refused, and what reopens
it, are at the end.

## What the code already enforces

| Condition | Where |
|---|---|
| Model and MinerU are reached over HTTPS only (plain HTTP only to this machine); redirects are refused | `common.transport.assert_encrypted`, `common.transport.open_url` |
| The API answers only over HTTPS (`REQUIRE_HTTPS`, on by default) | `azure_service.api` (`guard`) |
| Page links are read-only and expire in an hour; a document's pages are deleted when it is done | `inference_core.page_links` (`expiry_s=3600`), `azure_service.worker` |
| MinerU is the pinned version and tier; text from any other is refused | `adapters.served_ocr` (`expected_version`) |
| Failure messages, job lists and the telemetry line carry no document content (tested) | `tests/test_azure_service.py`, `tests/test_policy_windows.py` |

## What a person confirms

Tick each, write the evidence (a document, a screenshot, a setting's value), sign at the bottom.

| # | Condition | Why | Evidence | OK |
|---|---|---|---|---|
| 1 | The pod runs on RunPod **Secure Cloud**, under RunPod's **Data Processing Agreement**, signed, covering insurance documents | page images and PDFs leave Azure for the pod | | ☐ |
| 2 | The RunPod data centre is **near the Azure region**, in the same jurisdiction (e.g. a US-East pod for East US 2) | every window of every page travels Azure → RunPod; the data stays in one jurisdiction | | ☐ |
| 3 | **vLLM's request logging is off** (its default: never started with `--enable-log-requests`) | a logged request holds a document's page images | | ☐ |
| 4 | **MinerU's task output and vLLM's media cache are removed after use**, and the pod's volume is encrypted | our code keeps nothing on the pod, but those two live on its disk | | ☐ |
| 5 | **Where TLS ends** for RunPod's proxy URLs is confirmed with RunPod | HTTPS to the proxy does not say how far the encryption goes | | ☐ |
| 6 | **Both servers require a token** (`vllm serve --api-key`, MinerU's token) and the worker is given them (`MODEL_TOKEN`, `MINERU_TOKEN`) | the worker starts without them: the code does not refuse an open server | | ☐ |

Confirmed by: ______________________  Date: ____________  Pod / data centre: ______________

## The choice refused, and what reopens it

**Refused:** serving the model inside Azure (Azure GPU machines instead of RunPod), so that no
document leaves Azure. Kept for now because the pod is where the model, MinerU and their measured
speed already are; it would cost a new serving setup.

**What reopens it:** a client contract or the legal review saying documents must stay in Azure;
or a condition above that cannot be met.
