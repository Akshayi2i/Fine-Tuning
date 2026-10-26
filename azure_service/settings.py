"""The service's configuration, from the environment (Container Apps env vars and secrets).

Both containers read the same :class:`Settings`; each uses its part. Secrets
(API keys, tokens, the webhook secret) arrive as environment variables that
Container Apps fills from Key Vault references, and are never logged: the
dataclass ``repr`` leaves them out.

API keys are kept as SHA-256 digests only (:func:`parse_api_keys`): a key is
compared as a digest, in constant time, against every configured one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The shortest API key accepted: 32 characters (``secrets.token_urlsafe(32)`` gives 43).
MIN_KEY_CHARS = 32

#: A PDF larger than this is refused (the 114-page scan in the data is 84 MB).
DEFAULT_MAX_PDF_MB = 150

#: The review-flag kinds (a flag's suffix after its last ``:``) that say a value
#: needs a person, and so retain a job for review (``REVIEW_RETENTION_FLAG_KINDS``).
#: Each is a kind serving emits on a value: confidence (serving.pipeline,
#: calibration.apply_calibration), printed values (serving.printed_values,
#: serving.conventions) and value checks (serving.value_flags). Document-level
#: system flags - ``uncalibrated``, ``schema:invalid``, ``printed_values:no_page_text``
#: - are not among them: alone they would keep nearly every document.
DEFAULT_RETENTION_FLAG_KINDS = (
    "low_confidence", "no_confidence", "not_printed", "not_printed_removed", "parsed_mismatch", "role_mismatch",
    "printed_off_cited_page", "near_tie", "likely_missed", "unverified_unread_page", "link_to_removed_row",
    # Data that may be missing: rows the table's own count says were not read, a window the model failed.
    "row_count_mismatch", "window_failed",
)


class SettingsError(ValueError):
    """The environment does not configure the service coherently."""


def _flag(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise SettingsError(f"expected true or false, got {value!r}")


def _number(environ: Mapping[str, str], name: str, default: float, *, minimum: float = 0) -> float:
    raw = (environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SettingsError(f"{name} must be a number, got {raw!r}") from None
    if value < minimum:
        raise SettingsError(f"{name} must be at least {minimum}, got {value}")
    return value


def _fraction(environ: Mapping[str, str], name: str, default: float) -> float:
    value = _number(environ, name, default)
    if value > 1:
        raise SettingsError(f"{name} must be a share between 0 and 1, got {value}")
    return value


def _flag_kinds(raw: str | None) -> tuple[str, ...]:
    """``a,b`` -> ``("a", "b")``; unset: :data:`DEFAULT_RETENTION_FLAG_KINDS`."""
    if raw is None:
        return DEFAULT_RETENTION_FLAG_KINDS
    kinds = tuple(dict.fromkeys(kind.strip().lower() for kind in raw.split(",") if kind.strip()))
    bad = [kind for kind in kinds if not re.match(r"^[a-z][a-z0-9_]{0,63}$", kind)]
    if bad:
        raise SettingsError(f"REVIEW_RETENTION_FLAG_KINDS takes flag kinds (the part after a flag's last ':', "
                            f"e.g. low_confidence), not {bad}")
    return kinds


def _tenant_name(tenant: str) -> str:
    """A tenant id as the registry's paths take it (lowercase, ``-``/``_``)."""
    from artifact_registry.paths import PathError, release_index

    name = str(tenant).strip().lower()
    try:
        release_index(name)                     # validates it as a path segment
    except PathError as exc:
        raise SettingsError(str(exc)) from exc
    return name


def key_digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_api_keys(raw: str | None) -> dict[bytes, str]:
    """``{"tenant": "key"}`` or ``{"tenant": ["key", "next key"]}`` (two during a
    rotation) -> ``{sha256(key): tenant}``. A short key, or one key given to two
    tenants, is refused."""
    if not raw or not raw.strip():
        return {}
    try:
        table = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SettingsError(f"EXTRACTION_API_KEYS is not JSON ({exc.msg}); expected "
                            '{"tenant": ["key", ...]}') from None
    if not isinstance(table, dict):
        raise SettingsError('EXTRACTION_API_KEYS must be a JSON object: {"tenant": ["key", ...]}')
    digests: dict[bytes, str] = {}
    for tenant, keys in table.items():
        name = _tenant_name(tenant)
        for key in [keys] if isinstance(keys, str) else list(keys or []):
            if not isinstance(key, str) or len(key.strip()) < MIN_KEY_CHARS:
                raise SettingsError(f"an API key of tenant {name} is shorter than {MIN_KEY_CHARS} characters")
            digest = key_digest(key.strip())
            if digest in digests and digests[digest] != name:
                raise SettingsError(f"one API key is configured for tenants {digests[digest]} and {name}")
            digests[digest] = name
    return digests


def parse_secrets(raw: str | None, what: str) -> dict[str, str]:
    """``{"tenant": "secret"}`` -> the same, tenants normalised."""
    if not raw or not raw.strip():
        return {}
    try:
        table = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SettingsError(f"{what} is not JSON ({exc.msg})") from None
    if not isinstance(table, dict) or not all(isinstance(v, str) and v for v in table.values()):
        raise SettingsError(f'{what} must be a JSON object of strings: {{"tenant": "secret"}}')
    return {_tenant_name(tenant): secret for tenant, secret in table.items()}


@dataclass(frozen=True)
class Settings:
    """Everything both containers are configured with. Secrets are left out of ``repr``."""

    # -- storage -------------------------------------------------------------
    #: The container job state, results and the transient PDFs live in: one of
    #: the existing storage account's, never the raw-documents container.
    jobs_container: str = "extraction-jobs"
    #: The container the release index, bundles and calibrators are read from.
    registry_container: str | None = None
    queue_name: str = "extraction-jobs"
    #: ``managed_identity`` (Container Apps' identity, azure-identity), or
    #: ``environment``: a SAS or connection string as artifact_registry reads them.
    storage_auth: str = "environment"
    #: ``https://<account>.blob.core.windows.net`` / ``.queue.``: for a managed identity.
    blob_account_url: str | None = None
    queue_account_url: str | None = None
    #: A user-assigned identity's client id (None: the system-assigned one).
    managed_identity_client_id: str | None = None

    # -- API -----------------------------------------------------------------
    api_keys: Mapping[bytes, str] = field(default_factory=dict, repr=False)
    max_pdf_bytes: int = DEFAULT_MAX_PDF_MB * 1024 * 1024
    max_pages: int = 1000
    #: How results are reached from outside, for the webhook's links.
    public_base_url: str | None = None
    #: Webhooks to http://localhost and private addresses: tests and a receiver
    #: inside the same network only.
    allow_private_webhooks: bool = False
    #: Refuse a request the ingress did not receive over HTTPS (X-Forwarded-Proto).
    require_https: bool = False

    # -- webhooks (both: the API refuses a webhook a tenant has no secret for) -
    webhook_secrets: Mapping[str, str] = field(default_factory=dict, repr=False)
    webhook_secret: str | None = field(default=None, repr=False)

    # -- worker ----------------------------------------------------------------
    model_url: str | None = None
    model_token: str | None = field(default=None, repr=False)
    mineru_url: str | None = None
    mineru_token: str | None = field(default=None, repr=False)
    #: The tenants this worker serves: each one's promoted release is loaded at start.
    tenants: tuple[str, ...] = ("default",)
    #: The name vLLM serves the bare base model under (its --served-model-name).
    base_model_name: str | None = None
    #: Read a policy no promoted release covers with the base model (no LoRA).
    allow_base_fallback: bool = False
    #: Deliveries of one job before it is failed as poison.
    max_attempts: int = 3
    #: How long a received job stays invisible to other workers; renewed while it runs.
    visibility_timeout_s: int = 300
    #: How long a job that failed for a reason worth retrying waits before its next delivery.
    retry_delay_s: int = 60
    #: Keep each job's PDF, phase-1 checkpoint and page images after it ends.
    keep_files: bool = False
    work_dir: Path = Path("/tmp/extraction-jobs")
    #: ``inline`` (base64 in each request) or ``link`` (pages in Blob, read-only links).
    images: str = "inline"
    page_link_sas: str | None = field(default=None, repr=False)
    max_concurrency: int = 96
    run_pages: int = 20
    text_max_wait_s: float = 600.0
    #: Jobs whose fields are out and whose full text waits: no new job is taken beyond this.
    max_text_backlog: int = 4

    # -- retention for review (decision 2026-10-26, #43): OFF by default --------
    #: The tenants whose finished jobs may be kept for review (flagged, low
    #: confidence or sampled: :mod:`azure_service.retention`). Empty - the
    #: default - keeps nothing of anyone. Set only after the legal and contract
    #: sign-off DEPLOY.md names; no request can opt a tenant in.
    review_retention_tenants: tuple[str, ...] = ()
    #: The share of an opted-in tenant's unflagged jobs kept anyway, chosen by job id.
    review_retention_sample: float = 0.05
    #: An overall confidence below this keeps the job.
    review_retention_low_confidence: float = 0.80
    #: The review-flag kinds that retain a job (:data:`DEFAULT_RETENTION_FLAG_KINDS`).
    review_retention_flag_kinds: tuple[str, ...] = DEFAULT_RETENTION_FLAG_KINDS
    #: How long a retained document is kept before ``retention purge`` deletes it.
    review_retention_days: int = 90
    #: The container retained documents go to: its own, never the jobs, registry or raw one.
    review_container: str = "review-retention"

    def retains_for(self, tenant: str) -> bool:
        """Whether ``tenant``'s jobs may be retained for review: named in
        ``REVIEW_RETENTION_TENANTS``, and nothing else decides it."""
        return bool(tenant) and tenant in self.review_retention_tenants

    def webhook_secret_for(self, tenant: str) -> str | None:
        return self.webhook_secrets.get(tenant) or self.webhook_secret

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env: Mapping[str, str] = os.environ if environ is None else environ

        def text(name: str, default: str | None = None) -> str | None:
            value = (env.get(name) or "").strip()
            return value or default

        images = text("PAGE_IMAGES", "inline")
        if images not in ("inline", "link"):
            raise SettingsError(f"PAGE_IMAGES must be inline or link, got {images!r}")
        storage_auth = text("STORAGE_AUTH", "environment")
        if storage_auth not in ("environment", "managed_identity"):
            raise SettingsError(f"STORAGE_AUTH must be environment or managed_identity, got {storage_auth!r}")
        account = text("STORAGE_ACCOUNT")
        tenants = tuple(_tenant_name(t) for t in (text("SERVED_TENANTS", "default") or "").split(",") if t.strip())
        settings = cls(
            jobs_container=text("JOBS_CONTAINER", "extraction-jobs") or "extraction-jobs",
            registry_container=text("AZURE_BLOB_CONTAINER"),
            queue_name=text("QUEUE_NAME", "extraction-jobs") or "extraction-jobs",
            storage_auth=storage_auth or "environment",
            blob_account_url=text("BLOB_ACCOUNT_URL", f"https://{account}.blob.core.windows.net" if account else None),
            queue_account_url=text("QUEUE_ACCOUNT_URL",
                                   f"https://{account}.queue.core.windows.net" if account else None),
            managed_identity_client_id=text("AZURE_CLIENT_ID"),
            api_keys=parse_api_keys(env.get("EXTRACTION_API_KEYS")),
            max_pdf_bytes=int(_number(env, "MAX_PDF_MB", DEFAULT_MAX_PDF_MB, minimum=1) * 1024 * 1024),
            max_pages=int(_number(env, "MAX_PAGES", 1000, minimum=1)),
            public_base_url=(text("PUBLIC_BASE_URL") or "").rstrip("/") or None,
            allow_private_webhooks=_flag(env.get("ALLOW_PRIVATE_WEBHOOKS")),
            require_https=_flag(env.get("REQUIRE_HTTPS")),
            webhook_secrets=parse_secrets(env.get("WEBHOOK_SECRETS"), "WEBHOOK_SECRETS"),
            webhook_secret=text("WEBHOOK_SECRET"),
            model_url=text("MODEL_URL"),
            model_token=text("MODEL_TOKEN", text("FIDEON_REMOTE_TOKEN")),
            mineru_url=text("MINERU_URL"),
            mineru_token=text("MINERU_TOKEN", text("FIDEON_REMOTE_TOKEN")),
            tenants=tenants or ("default",),
            base_model_name=text("BASE_MODEL_NAME"),
            allow_base_fallback=_flag(env.get("ALLOW_BASE_FALLBACK")),
            max_attempts=int(_number(env, "MAX_ATTEMPTS", 3, minimum=1)),
            visibility_timeout_s=int(_number(env, "VISIBILITY_TIMEOUT_S", 300, minimum=30)),
            retry_delay_s=int(_number(env, "RETRY_DELAY_S", 60, minimum=0)),
            keep_files=_flag(env.get("KEEP_FILES")),
            work_dir=Path(text("WORK_DIR", "/tmp/extraction-jobs") or "/tmp/extraction-jobs"),
            images=images or "inline",
            page_link_sas=text("PAGE_LINK_SAS_TOKEN"),
            max_concurrency=int(_number(env, "MAX_CONCURRENCY", 96, minimum=1)),
            run_pages=int(_number(env, "RUN_PAGES", 20, minimum=1)),
            text_max_wait_s=_number(env, "TEXT_MAX_WAIT_S", 600.0, minimum=1),
            max_text_backlog=int(_number(env, "MAX_TEXT_BACKLOG", 4, minimum=1)),
            review_retention_tenants=tuple(_tenant_name(t) for t in (text("REVIEW_RETENTION_TENANTS") or "")
                                           .split(",") if t.strip()),
            review_retention_sample=_fraction(env, "REVIEW_RETENTION_SAMPLE", 0.05),
            review_retention_low_confidence=_fraction(env, "REVIEW_RETENTION_LOW_CONFIDENCE", 0.80),
            review_retention_flag_kinds=_flag_kinds(text("REVIEW_RETENTION_FLAG_KINDS")),
            review_retention_days=int(_number(env, "REVIEW_RETENTION_DAYS", 90, minimum=1)),
            review_container=text("REVIEW_CONTAINER", "review-retention") or "review-retention",
        )
        others = {settings.jobs_container, settings.registry_container, text("AZURE_RAW_CONTAINER")} - {None}
        if settings.review_container in others:
            raise SettingsError(f"REVIEW_CONTAINER must be a container of its own, not {settings.review_container!r} "
                                "(the jobs, registry or raw one): retained documents get their own access and expiry")
        if settings.allow_base_fallback and not settings.base_model_name:
            raise SettingsError("ALLOW_BASE_FALLBACK needs BASE_MODEL_NAME: the name vLLM serves the base model as")
        if settings.storage_auth == "managed_identity" and not (settings.blob_account_url
                                                                 and settings.queue_account_url):
            raise SettingsError("STORAGE_AUTH=managed_identity needs STORAGE_ACCOUNT (or BLOB_ACCOUNT_URL and "
                                "QUEUE_ACCOUNT_URL)")
        return settings

    def check_api(self) -> None:
        """What the API container cannot start without."""
        if not self.api_keys:
            raise SettingsError("no API keys: set EXTRACTION_API_KEYS to {\"tenant\": [\"key\", ...]}")

    def check_worker(self) -> None:
        """What the worker container cannot start without: both servers, over TLS."""
        from common.transport import InsecureUrlError, assert_encrypted

        missing = [name for name, value in (("MODEL_URL", self.model_url), ("MINERU_URL", self.mineru_url))
                   if not value]
        if missing:
            raise SettingsError(f"{' and '.join(missing)} not set: the worker reads with the served model and MinerU")
        try:
            assert_encrypted(str(self.model_url), "the model server")
            assert_encrypted(str(self.mineru_url), "the MinerU server")
        except InsecureUrlError as exc:
            raise SettingsError(str(exc)) from exc
        if self.images == "link" and not self.page_link_sas and self.storage_auth == "managed_identity":
            raise SettingsError("PAGE_IMAGES=link with a managed identity needs PAGE_LINK_SAS_TOKEN (a read-only "
                                "SAS for the jobs container): an identity cannot sign a blob's link")

    def describe(self) -> dict[str, Any]:
        """What may be logged at start: no secret, no key."""
        return {
            "jobs_container": self.jobs_container, "queue": self.queue_name, "storage_auth": self.storage_auth,
            "tenants": list(self.tenants), "api_keys": len(self.api_keys),
            "max_pdf_mb": self.max_pdf_bytes // 2**20, "max_attempts": self.max_attempts,
            "visibility_timeout_s": self.visibility_timeout_s, "images": self.images,
            "base_fallback": self.allow_base_fallback, "keep_files": self.keep_files,
            "webhook_secrets": sorted(self.webhook_secrets) + (["*"] if self.webhook_secret else []),
            "review_retention": {"tenants": list(self.review_retention_tenants), "container": self.review_container,
                                 "days": self.review_retention_days} if self.review_retention_tenants else "off",
        }
