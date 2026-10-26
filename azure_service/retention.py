"""Retention for review: keep flagged production documents for the feedback loop.

**OFF BY DEFAULT** (decision 2026-10-26, #43). Nothing is kept for any tenant
until ``REVIEW_RETENTION_TENANTS`` names it, and a tenant is named there only
after the legal and contract sign-off ``DEPLOY.md`` ("Retention for review")
lists. Only that setting decides: no request, header or job field can opt a
tenant in, and the API shows nothing of it.

    python -m azure_service.retention purge [--tenant T] [--all] [--dry-run]
    python -m azure_service.retention export --tenant T --out <folder | intake/<batch>> [--limit N]
    python -m azure_service.retention mark-reviewed --tenant T --job J

**What is kept, and when.** When a job of an opted-in tenant ends ``done``, the
worker (:meth:`azure_service.worker.Worker._finish`) keeps it - before its PDF
is deleted - when any of these holds:

* ``flagged``: a served value is flagged (``"flagged": true`` in its envelope,
  for a reason other than an ``uncalibrated`` release) or the result carries a
  review flag of a kind in ``REVIEW_RETENTION_FLAG_KINDS`` (low_confidence,
  not_printed, role_mismatch, near_tie ...: a value needs a person). A
  document-level system flag (``uncalibrated``, ``schema:invalid``) alone keeps
  nothing; the manifest names the kinds that did (``triggered_kinds``);
* ``low_confidence``: its overall confidence is below
  ``REVIEW_RETENTION_LOW_CONFIDENCE`` (0.80), or it has none;
* ``sample``: neither, but the job falls in the ``REVIEW_RETENTION_SAMPLE``
  share (0.05) - chosen by a hash of the job id, so the same job is always in
  or out.

Layout in ``REVIEW_CONTAINER`` (default ``review-retention``: a container of its
own, never the jobs, registry or raw one):

    retained/<tenant>/<job id>/manifest.json  tenant, job, line, reasons, flag counts, dates, model release, state
    retained/<tenant>/<job id>/document.pdf   the PDF as the client sent it
    retained/<tenant>/<job id>/result.json    the final result as it was served

The manifest is written first (state ``retaining``) and again last
(``retained``), so a folder without a manifest is never one being written, and
a ``retaining`` one older than a day was left by a worker that died mid-copy.
It holds no document content: no file name, no value, flag *kinds* only.

**States.** ``retained`` -> ``exported`` (copied to the labelling intake) ->
``reviewed`` (``mark-reviewed``: a person has dealt with it). ``purge`` deletes
an item past its ``expires_at`` (``REVIEW_RETENTION_DAYS``, 90), a reviewed one,
an incomplete one and files without a manifest; ``purge --tenant T --all``
deletes every item of a tenant (consent withdrawn, contract ended). A Blob
lifecycle rule on ``retained/`` is the backstop (DEPLOY.md).

**Export** copies retained documents into the labelling intake as UNLABELLED
documents, one folder each, as ``data_pipeline.audit`` and
``data_pipeline.ingestion.import_labeled_pdfs`` read them::

    <out>/<tenant>__<job id>/document.pdf    the PDF
    <out>/<tenant>__<job id>/metadata.json   lob, synthetic false, the tenant, the job, why it was kept
    <out>/<tenant>__<job id>/draft.json      the served model's answer (pre_annotate's Draft shape) - never a label

No ``golden.json`` is ever written: a person writes it, and adds ``reviewer_id``
and ``review_date`` to ``metadata.json``; the audit and the import refuse the
folder until then (decision 2026-10-22). ``--out intake/<batch>`` writes to the
raw container's ``intake/<batch>/`` (pulled by
``data_pipeline.ingestion.pull_intake``); any other value is a local folder
(``./intake/<batch>`` for a local one of that name). Only a tenant named in
``REVIEW_RETENTION_TENANTS`` is exported.

**Nothing of a document is logged**: job ids, tenants, reasons and counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_registry.blob_client import BlobClient, BlobError
from azure_service.store import JOB_ID

log = logging.getLogger(__name__)

#: Every retained item lives under this prefix of the review container.
PREFIX = "retained"
MANIFEST, PDF, RESULT = "manifest.json", "document.pdf", "result.json"

RETAINING, RETAINED, EXPORTED, REVIEWED = "retaining", "retained", "exported", "reviewed"
STATES = (RETAINING, RETAINED, EXPORTED, REVIEWED)
REASONS = ("flagged", "low_confidence", "sample")

#: A ``retaining`` manifest older than this was left by a worker that died mid-copy.
STALE_RETAINING_S = 24 * 3600

#: What drafted the exported draft.json: the promoted fine-tuned model, served.
DRAFT_BACKEND = "own_finetuned"
#: The intake folder's ``source_system``.
SOURCE_SYSTEM = "production_review"

#: ``--out intake/<batch>``: the raw container's intake (paths.intake_batch_dir's names).
INTAKE_OUT = re.compile(r"^intake/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/?$")

_FLAG_KIND = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_LAST_INDEX = re.compile(r"\[\d+\]$")
#: serving.pipeline's: the release had no calibrator for the value (every value
#: of such a result is flagged), and a window that could not be read.
UNCALIBRATED, WINDOW_FAILED = "uncalibrated", "window_failed"
_STAMP = "%Y-%m-%dT%H:%M:%SZ"


class RetentionError(RuntimeError):
    """A retention command cannot do what it was asked."""


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime(_STAMP)


def _seconds(stamp: Any) -> float:
    return datetime.strptime(str(stamp), _STAMP).replace(tzinfo=UTC).timestamp()


def folder(tenant: str, job_id: str) -> str:
    """``retained/<tenant>/<job id>``; a tenant or job id that is not one is refused."""
    if not JOB_ID.match(str(job_id or "")):
        raise RetentionError(f"not a job id: {job_id!r}")
    if not tenant or not re.match(r"^[a-z0-9][a-z0-9_-]*$", str(tenant)):
        raise RetentionError(f"not a tenant: {tenant!r}")
    return f"{PREFIX}/{tenant}/{job_id}"


# --------------------------------------------------------------------------
# What is kept
# --------------------------------------------------------------------------


def in_sample(job_id: str, share: float) -> bool:
    """Whether ``job_id`` falls in the sampled ``share``: a hash of the id, so a
    job is always in or always out, whichever worker reads it."""
    if share <= 0:
        return False
    if share >= 1:
        return True
    point = int.from_bytes(hashlib.sha256(job_id.encode("utf-8")).digest()[:8], "big") / 2**64
    return point < share


def flag_kind(flag: str) -> str:
    """A review flag's kind - its suffix after the last ``:`` (``low_confidence``,
    ``near_tie`` ...) - never its path; ``other`` when that is no kind's name."""
    head, _, _rest = str(flag).partition(":")
    if head == WINDOW_FAILED:                 # window_failed:<the window>: the kind comes first
        return WINDOW_FAILED
    kind = str(flag).rpartition(":")[2]
    return kind if _FLAG_KIND.match(kind) else "other"


def _unindexed(path: str) -> str:
    return _LAST_INDEX.sub("", path)


def flag_summary(output: dict[str, Any], retain_kinds: tuple[str, ...] = ()) -> dict[str, Any]:
    """How much of a served output is flagged, without any of its content.

    ``flagged_fields``: its envelopes marked ``"flagged": true``, but not one
    whose only reason is that the release had no calibrator for it (an
    ``uncalibrated`` flag on its path: every value of an uncalibrated result is
    flagged, which says nothing of the value). ``flag_kinds``: its review flags
    counted by kind. ``triggered_kinds``: those of ``retain_kinds`` it carries.
    """
    flags = [str(flag) for flag in output.get("review_flags") or []]
    uncalibrated = {flag.rpartition(":")[0] for flag in flags if flag_kind(flag) == UNCALIBRATED}
    flagged = 0

    def walk(node: Any, path: str) -> None:
        nonlocal flagged
        if isinstance(node, dict):
            if "flagged" in node and ("raw" in node or "parsed" in node):      # a value's envelope
                if node.get("flagged") is True and path not in uncalibrated \
                        and _unindexed(path) not in uncalibrated:
                    flagged += 1
                return
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(output.get("extraction"), "")
    kinds = Counter(flag_kind(flag) for flag in flags)
    return {"flagged_fields": flagged, "review_flags": len(flags), "flag_kinds": dict(sorted(kinds.items())),
            "triggered_kinds": sorted(kind for kind in kinds if kind in retain_kinds)}


def overall_confidence(output: dict[str, Any]) -> float | None:
    value = output.get("overall_confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def reasons_to_retain(result: dict[str, Any], job_id: str, settings: Any) -> tuple[list[str], dict[str, Any]]:
    """Why a finished job's result is kept (:data:`REASONS`; none: it is not),
    and its flag summary. ``flagged`` counts a flagged envelope and a review flag
    of a kind in ``REVIEW_RETENTION_FLAG_KINDS`` - never a document-level system
    flag (``uncalibrated``, ``schema:invalid``) alone."""
    output = result.get("output") or {}
    summary = flag_summary(output, tuple(settings.review_retention_flag_kinds))
    reasons = []
    if summary["flagged_fields"] or summary["triggered_kinds"]:
        reasons.append("flagged")
    confidence = overall_confidence(output)
    if confidence is None or confidence < settings.review_retention_low_confidence:
        reasons.append("low_confidence")
    if not reasons and in_sample(job_id, settings.review_retention_sample):
        reasons.append("sample")
    return reasons, summary


# --------------------------------------------------------------------------
# The review container
# --------------------------------------------------------------------------


@dataclass
class Item:
    """One retained job: its blobs and its manifest (None: none, or not JSON)."""

    tenant: str
    job_id: str
    keys: list[str]
    manifest: dict[str, Any] | None = None
    #: The manifest could not be read (Blob unreachable): left alone.
    error: str | None = None

    @property
    def manifest_key(self) -> str:
        return f"{PREFIX}/{self.tenant}/{self.job_id}/{MANIFEST}"

    @property
    def state(self) -> str | None:
        return (self.manifest or {}).get("state")


@dataclass
class PurgeReport:
    deleted: Counter = field(default_factory=Counter)
    kept: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def describe(self) -> str:
        verb = "would delete" if self.dry_run else "deleted"
        reasons = ", ".join(f"{count} {reason}" for reason, count in sorted(self.deleted.items())) or "nothing"
        lines = [f"purge: {verb} {sum(self.deleted.values())} item(s) ({reasons}); kept {self.kept}"]
        lines += [f"  NOT CHECKED {error}" for error in self.errors]
        return "\n".join(lines)


@dataclass
class ExportReport:
    where: str
    exported: list[str] = field(default_factory=list)
    waiting: int = 0
    failed: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"export: {len(self.exported)} unlabelled document(s) to {self.where}"
                 + (f"; {self.waiting} more retained (raise --limit)" if self.waiting else "")]
        lines += [f"  FAILED {failure}" for failure in self.failed]
        if self.exported:
            lines.append("Each folder is UNLABELLED: a person writes golden.json beside it (draft.json is the model's "
                         "answer, never a label) and adds reviewer_id and review_date to metadata.json; then\n"
                         "  python -m data_pipeline.audit --input <folder>\n"
                         "  python -m data_pipeline.ingestion.import_labeled_pdfs --input <folder> --doc-type policy\n"
                         "(intake/<batch>: pull it first with data_pipeline.ingestion.pull_intake --batch <batch>), "
                         "and mark each one reviewed:\n"
                         "  python -m azure_service.retention mark-reviewed --tenant <tenant> --job <job id>")
        return "\n".join(lines)


class LocalIntake:
    """A local folder of document folders (the audit's and the importer's ``--input``)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.where = str(self.root)

    def write(self, name: str, filename: str, data: bytes) -> None:
        target = self.root / name
        target.mkdir(parents=True, exist_ok=True)
        part = target / f"{filename}.part"           # never a truncated file under its real name
        part.write_bytes(data)
        os.replace(part, target / filename)


class BlobIntake:
    """``intake/<batch>/`` in the raw container, as pull_intake reads it."""

    def __init__(self, client: BlobClient, batch: str) -> None:
        from artifact_registry import paths

        self.prefix = paths.intake_batch_dir(batch)
        self.client = client
        self.where = f"{client.raw_container}/{self.prefix}"

    def write(self, name: str, filename: str, data: bytes) -> None:
        self.client.write_bytes(f"{self.prefix}/{name}/{filename}", data)


def _json_bytes(body: Any) -> bytes:
    return json.dumps(body, indent=2, ensure_ascii=False).encode("utf-8")


def intake_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """The exported folder's ``metadata.json``: what the importer reads, and no
    review record - a person adds one when they have labelled it."""
    return {
        "lob": manifest.get("lob"),
        "synthetic": False,
        "tenant_id": manifest.get("tenant"),
        "source_system": SOURCE_SYSTEM,
        "production_job_id": manifest.get("job_id"),
        "retained_reasons": list(manifest.get("reasons") or []),
        "retained_flags": manifest.get("flags"),
        "model_release": manifest.get("release"),
        "labelled": False,
        "to_label": ("UNLABELLED. Write golden.json beside this file - draft.json is the served model's answer, a "
                     "starting point and never a label - and add reviewer_id and review_date here. The audit and "
                     "the import refuse this folder until then."),
    }


def intake_draft(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The exported ``draft.json``, in ``data_pipeline.labeling.pre_annotate.Draft``'s
    shape: the served extraction, never trusted."""
    output = result.get("output") or {}
    confidence = {path: entry["confidence"] for path, entry in (output.get("fields") or {}).items()
                  if isinstance(entry, dict) and isinstance(entry.get("confidence"), (int, float))}
    return {
        "source_id": None, "doc_type": output.get("doc_type") or "policy", "acord_form": None,
        "draft_backend": DRAFT_BACKEND, "model_version": manifest.get("release") or result.get("model"),
        "review_requirement": "full", "schema_valid": bool(output.get("schema_valid")),
        "created_at": manifest.get("retained_at"), "extraction": output.get("extraction") or {},
        "field_confidence": confidence, "trusted": False, "production_job_id": manifest.get("job_id"),
    }


class ReviewRetention:
    """Retained documents in the review container (``REVIEW_CONTAINER``)."""

    def __init__(self, settings: Any, client: BlobClient, *, clock: Any = time.time) -> None:
        self.settings = settings
        self.client = client
        self.clock = clock

    # -- the worker's side ------------------------------------------------------

    def check(self) -> None:
        """Write and delete a probe: the container exists and this login may use it."""
        probe = f"{PREFIX}/.access-check"
        self.client.write_bytes(probe, b"")
        self.client.delete(probe)

    def keep(self, store: Any, record: dict[str, Any]) -> dict[str, Any] | None:
        """Retain a job that ended ``done``, when its tenant is opted in and its
        result calls for it: the PDF, the result and a manifest, copied from the
        jobs container. Call it before the job's PDF is deleted. The manifest, or
        None when nothing was kept."""
        tenant, job_id = str(record.get("tenant") or ""), str(record.get("job_id") or "")
        if record.get("status") != "done" or not self.settings.retains_for(tenant):
            return None
        base = folder(tenant, job_id)
        if self.client.exists(f"{base}/{MANIFEST}"):
            return None                                  # kept already: a second delivery of the job
        if not store.client.exists(store.pdf_key(tenant, job_id)):
            log.debug("job %s: its PDF is gone; nothing to retain", job_id)
            return None
        result = store.read_result(tenant, job_id)
        reasons, summary = reasons_to_retain(result, job_id, self.settings)
        if not reasons:
            return None
        now = self.clock()
        manifest = {
            "tenant": tenant, "job_id": job_id, "lob": record.get("lob") or result.get("lob"),
            "reasons": reasons, "triggered_kinds": summary["triggered_kinds"], "flags": summary,
            "overall_confidence": overall_confidence(result.get("output") or {}),
            "model": result.get("model"), "release": result.get("release"), "route": result.get("route"),
            "full_text": bool(result.get("full_text")), "pages": record.get("pages"), "bytes": record.get("bytes"),
            "rule": {"low_confidence_below": self.settings.review_retention_low_confidence,
                     "sample": self.settings.review_retention_sample,
                     "flag_kinds": list(self.settings.review_retention_flag_kinds)},
            "retained_at": _iso(now), "expires_at": _iso(now + self.settings.review_retention_days * 86400),
            "state": RETAINING,
        }
        self.client.write_json(f"{base}/{MANIFEST}", manifest)
        self.client.write_bytes(f"{base}/{PDF}", store.read_pdf(tenant, job_id))
        self.client.write_json(f"{base}/{RESULT}", result)
        manifest["state"] = RETAINED
        self.client.write_json(f"{base}/{MANIFEST}", manifest)
        log.info("job %s: retained for review (%s) until %s", job_id, ", ".join(reasons), manifest["expires_at"])
        return manifest

    # -- reading it back ------------------------------------------------------------

    def items(self, tenant: str | None = None) -> list[Item]:
        """Every retained item (of ``tenant``), with its manifest."""
        prefix = f"{PREFIX}/{tenant}/" if tenant else f"{PREFIX}/"
        groups: dict[tuple[str, str], list[str]] = {}
        for key in self.client.list(prefix):
            parts = key.split("/")
            # The listing matches by prefix: retained/acme also lists retained/acme2/.
            if not key.startswith(prefix) or len(parts) != 4 or not JOB_ID.match(parts[2]):
                continue
            groups.setdefault((parts[1], parts[2]), []).append(key)
        found = []
        for (owner, job_id), keys in sorted(groups.items()):
            item = Item(owner, job_id, sorted(keys))
            if item.manifest_key in keys:
                try:
                    body = self.client.read_json(item.manifest_key)
                    item.manifest = body if isinstance(body, dict) else None
                except BlobError as exc:
                    item.error = f"{owner}/{job_id}: its manifest could not be read ({type(exc).__name__})"
                except ValueError:
                    item.manifest = None                    # not JSON: no manifest
            found.append(item)
        return found

    def _expired(self, item: Item, now: float) -> bool:
        return _seconds((item.manifest or {}).get("expires_at")) <= now

    def purge_reason(self, item: Item, now: float) -> str | None:
        """Why ``item`` is deleted now; None when it is kept."""
        manifest = item.manifest
        if manifest is None:
            return "no_manifest"
        try:
            if self._expired(item, now):
                return "expired"
            if item.state == RETAINING and now - _seconds(manifest.get("retained_at")) > STALE_RETAINING_S:
                return "incomplete"
        except (TypeError, ValueError):
            return "unreadable_manifest"
        if item.state == REVIEWED:
            return "reviewed"
        if item.state not in STATES:
            return "unknown_state"
        return None

    def _delete(self, item: Item) -> None:
        # The manifest last: a purge cut short leaves it, and the next purge finishes the item.
        for key in [k for k in item.keys if k != item.manifest_key] + [item.manifest_key]:
            if key in item.keys:
                self.client.delete(key)

    def purge(self, tenant: str | None = None, *, everything: bool = False, dry_run: bool = False) -> PurgeReport:
        """Delete expired, reviewed and incomplete items (of ``tenant``); with
        ``everything``, every item of ``tenant``."""
        if everything and not tenant:
            raise RetentionError("--all deletes every retained item of one tenant: name it with --tenant")
        report = PurgeReport(dry_run=dry_run)
        now = self.clock()
        for item in self.items(tenant):
            if item.error:
                report.errors.append(item.error)
                continue
            reason = "all" if everything else self.purge_reason(item, now)
            if reason is None:
                report.kept += 1
                continue
            if not dry_run:
                try:
                    self._delete(item)
                except BlobError as exc:
                    report.errors.append(f"{item.tenant}/{item.job_id}: not deleted ({type(exc).__name__})")
                    continue
            report.deleted[reason] += 1
            log.info("retained job %s/%s: %s (%s)", item.tenant, item.job_id,
                     "would be deleted" if dry_run else "deleted", reason)
        return report

    def mark_reviewed(self, tenant: str, job_id: str) -> dict[str, Any]:
        """A person has dealt with the item: ``purge`` deletes it next."""
        key = f"{folder(tenant, job_id)}/{MANIFEST}"
        if not self.client.exists(key):
            raise RetentionError(f"no retained item {tenant}/{job_id}")
        manifest = self.client.read_json(key)
        manifest.update(state=REVIEWED, reviewed_at=_iso(self.clock()))
        self.client.write_json(key, manifest)
        return manifest

    def export(self, tenant: str, sink: Any, *, limit: int | None = None) -> ExportReport:
        """Copy ``tenant``'s retained, unexported items to the labelling intake
        ``sink`` as unlabelled documents (oldest first), each marked ``exported``."""
        if not self.settings.retains_for(tenant):
            raise RetentionError(f"tenant {tenant!r} is not in REVIEW_RETENTION_TENANTS: nothing of it is exported. "
                                 "Run export with the settings the worker has.")
        if limit is not None and limit < 1:
            raise RetentionError("--limit must be at least 1")
        now = self.clock()
        ready = []
        for item in self.items(tenant):
            try:
                if item.state == RETAINED and not self._expired(item, now):
                    ready.append(item)
            except (TypeError, ValueError):
                continue                                 # an unreadable manifest: purge removes it
        ready.sort(key=lambda item: str((item.manifest or {}).get("retained_at") or ""))
        report = ExportReport(where=sink.where)
        chosen = ready if limit is None else ready[:limit]
        report.waiting = len(ready) - len(chosen)
        for item in chosen:
            manifest = dict(item.manifest or {})
            base = folder(item.tenant, item.job_id)
            name = f"{item.tenant}__{item.job_id}"
            try:
                result = self.client.read_json(f"{base}/{RESULT}")
                sink.write(name, "document.pdf", self.client.read_bytes(f"{base}/{PDF}"))
                sink.write(name, "draft.json", _json_bytes(intake_draft(result, manifest)))
                sink.write(name, "metadata.json", _json_bytes(intake_metadata(manifest)))
                manifest.update(state=EXPORTED, exported_at=_iso(now), exported_to=f"{sink.where}/{name}")
                self.client.write_json(item.manifest_key, manifest)
            except (BlobError, OSError, ValueError) as exc:
                report.failed.append(f"{item.tenant}/{item.job_id}: {type(exc).__name__}")
                continue
            report.exported.append(name)
            log.info("retained job %s/%s exported unlabelled", item.tenant, item.job_id)
        return report


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def review_client(settings: Any, backend: Any = None) -> BlobClient:
    """The review container (``REVIEW_CONTAINER``)."""
    from azure_service.store import blob_backend

    return BlobClient(backend=backend or blob_backend(settings), container=settings.review_container,
                      context="serving")


def review_retention(settings: Any, backend: Any = None, *, clock: Any = time.time) -> ReviewRetention | None:
    """The worker's retention, or None while ``REVIEW_RETENTION_TENANTS`` is empty (the default)."""
    if not settings.review_retention_tenants:
        return None
    return ReviewRetention(settings, review_client(settings, backend), clock=clock)


def intake_client(settings: Any, backend: Any = None) -> BlobClient:
    """The raw container's ``intake/``, in the ingestion context (the only one that may write it)."""
    raw = (os.environ.get("AZURE_RAW_CONTAINER") or "").strip()
    if not raw:
        raise RetentionError("AZURE_RAW_CONTAINER is not set: intake/ lives in the raw container")
    if backend is None:
        if settings.storage_auth == "managed_identity":
            from azure_service.store import blob_backend

            backend = blob_backend(settings)
        else:
            from artifact_registry.blob_client import azure_from_environment

            backend = azure_from_environment(raw_container=raw)
    return BlobClient(backend=backend, container=raw, raw_container=raw, context="ingestion")


def intake_sink(out: str, settings: Any, client: BlobClient | None = None) -> LocalIntake | BlobIntake:
    """``intake/<batch>``: the raw container's intake; anything else: a local folder."""
    match = INTAKE_OUT.match(str(out))
    if match:
        try:
            return BlobIntake(client or intake_client(settings), match.group(1))
        except ValueError as exc:
            raise RetentionError(str(exc)) from exc
    return LocalIntake(Path(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m azure_service.retention",
                                     description="Documents retained for review (off by default; see DEPLOY.md)")
    commands = parser.add_subparsers(dest="command", required=True)
    purge = commands.add_parser("purge", help="delete expired, reviewed and incomplete items")
    purge.add_argument("--tenant", default=None, help="one tenant's items only")
    purge.add_argument("--all", dest="everything", action="store_true",
                       help="every item of --tenant, whatever its state (consent withdrawn, contract ended)")
    purge.add_argument("--dry-run", action="store_true", help="report what would be deleted; delete nothing")
    export = commands.add_parser("export", help="copy retained documents to the labelling intake, unlabelled")
    export.add_argument("--tenant", required=True)
    export.add_argument("--out", required=True,
                        help="a local folder, or intake/<batch> for the raw container's intake")
    export.add_argument("--limit", type=int, default=None, help="at most this many documents (oldest first)")
    reviewed = commands.add_parser("mark-reviewed", help="mark an item reviewed: the next purge deletes it")
    reviewed.add_argument("--tenant", required=True)
    reviewed.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        # On the pod, an export of many PDFs runs detached in tmux; off it, here.
        from orchestration.detach import detach_module_if_needed

        if detach_module_if_needed("azure_service.retention", argv):
            return 0

    from azure_service.settings import Settings, SettingsError, _tenant_name

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        tenant = _tenant_name(args.tenant) if args.tenant else None
        retention = ReviewRetention(settings, review_client(settings))
        if args.command == "purge":
            report = retention.purge(tenant, everything=args.everything, dry_run=args.dry_run)
            print(report.describe())
            return 1 if report.errors else 0
        if args.command == "mark-reviewed":
            retention.mark_reviewed(str(tenant), args.job)
            print(f"{tenant}/{args.job}: reviewed; the next purge deletes it")
            return 0
        exported = retention.export(str(tenant), intake_sink(args.out, settings), limit=args.limit)
        print(exported.describe())
        return 1 if exported.failed else 0
    except (SettingsError, RetentionError, BlobError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
