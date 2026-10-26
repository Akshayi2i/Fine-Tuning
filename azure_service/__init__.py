"""The production extraction service on Azure: an HTTPS API and one worker per GPU.

Production design (decided 2026-10-24): the extraction routine runs on Azure
Container Apps; RunPod serves only a stock vLLM (Qwen3-VL-8B with the promoted
release's LoRA, asked for as ``policy-vN``) and a MinerU 4 API, both reached over
HTTPS. No code of ours runs on RunPod. One H100 serves both, so a document is
read **fields first** (:mod:`serving.fields_first`): the model reads the page
images and the fields are checked against the pages they cite (``fields_ready``),
then MinerU reads the rest in the GPU's idle turns and the full-text tier joins
(``done``).

* :mod:`azure_service.api` - FastAPI: submit a PDF, poll the job, read the result.
* :mod:`azure_service.worker` - takes jobs from the queue and reads them, two
  threads sharing one GPU's turns (as ``testing.extract_server.ExtractService``).
* :mod:`azure_service.runtime` - the worker's cold start: the promoted release,
  the served model, its calibrators, the pinned MinerU.
* :mod:`azure_service.store` - job state and results as JSON blobs.
* :mod:`azure_service.job_queue` - Azure Storage Queue, and an in-memory one for tests.
* :mod:`azure_service.webhooks` - signed callbacks with no document content.
* :mod:`azure_service.settings` - the environment both containers read.
* :mod:`azure_service.retention` - flagged documents kept for review, per tenant
  and OFF by default (legal sign-off first); its purge, export and mark-reviewed CLI.

Deployment: ``azure_service/Dockerfile`` and ``azure_service/DEPLOY.md``.
"""
