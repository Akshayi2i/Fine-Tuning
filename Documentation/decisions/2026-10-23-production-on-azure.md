# Production on Azure

- **Date:** 2026-10-23
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

The extraction routine runs on Azure. RunPod serves only a vLLM, with the release’s adapter, and the MinerU 4 API; both are reached over HTTPS (§14.1, §14.4). Open: a stock vllm serve 0.11.0 cannot start with the adapter (proven on the pod 2026-11-11); the fix, inference_core/vllm_site switched on by FIDEON_VLLM_PATCH, would put code of ours on the pod, and that is not decided

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
