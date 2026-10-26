# The extraction service's two containers, from one Dockerfile:
#
#   docker buildx build -f azure_service/Dockerfile --target api    -t $ACR.azurecr.io/extraction-api:$TAG .
#   docker buildx build -f azure_service/Dockerfile --target worker -t $ACR.azurecr.io/extraction-worker:$TAG .
#
# Build from a CLEAN checkout (`git archive`, azure_service/DEPLOY.md): the working tree
# holds client documents (data/, Test Data/, testing/) that must never reach a build
# context, let alone an image. The COPY lines below name the code they take; nothing
# else is copied, whatever the context holds.
#
# No GPU, no torch, no vLLM, no MinerU: the model and MinerU are served on RunPod and
# reached over HTTPS. The worker renders pages (PyMuPDF) and checks the model's JSON.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# The dependencies, from pyproject.toml alone (versions live there): the core list,
# [data] (Blob, PyMuPDF, pillow) and [service] (FastAPI, uvicorn, the queue, identity).
COPY pyproject.toml ./
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; e = p['optional-dependencies']; print('\n'.join(p['dependencies'] + e['data'] + e['service']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The code the API and the worker import, and the configs, schemas and prompts the
# pipeline reads (common.config resolves them next to the code). Not training/,
# testing/, pilot/, data/ or tests/.
COPY common/ common/
COPY artifact_registry/ artifact_registry/
COPY registry_utils/ registry_utils/
COPY data_pipeline/ data_pipeline/
COPY inference_core/ inference_core/
COPY calibration/ calibration/
COPY serving/ serving/
COPY postprocessing/ postprocessing/
COPY evaluation/ evaluation/
COPY orchestration/ orchestration/
COPY scripts/__init__.py scripts/served_adapter.py scripts/
COPY configs/ configs/
COPY schemas/ schemas/
COPY prompts/ prompts/
COPY azure_service/ azure_service/

RUN useradd --create-home --uid 10001 service \
    && mkdir -p /tmp/extraction-jobs && chown 10001 /tmp/extraction-jobs

# --------------------------------------------------------------------------
# The HTTPS API (TLS ends at the Container Apps ingress)
# --------------------------------------------------------------------------
FROM base AS api
USER 10001
EXPOSE 8080
ENV API_PORT=8080
CMD ["python", "-m", "azure_service.api"]

# --------------------------------------------------------------------------
# The worker: one replica per GPU (DEPLOY.md)
# --------------------------------------------------------------------------
FROM base AS worker
USER 10001
ENV WORK_DIR=/tmp/extraction-jobs
CMD ["python", "-m", "azure_service.worker"]
