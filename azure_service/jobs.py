"""The jobs that need a person: given up, or finished with pages unread (review S5).

    python -m azure_service.jobs failed  [--tenant T]   # given up: no result
    python -m azure_service.jobs partial [--tenant T]   # done, but some pages' answers could not be read

``failed`` is every job whose status is ``failed``: a PDF that failed on every
delivery, a job the worker can never read, a full text that failed for good.
``partial`` is every finished job whose result carries ``window_failed:`` review
flags: the model answered, but its answer for those pages could not be read even
one page at a time, so they are missing from the fields (``azure_service/RUNBOOK.md``,
case 4). A model server that stops answering never ends here: that job is retried
whole (case 1).

Prints one JSON line per job: id, tenant, status, attempts, times and the
error or the unread windows. Never the document's name or any extracted value:
no client data in an operator's terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from typing import Any

from azure_service.store import JobStore

#: The review flag a window leaves on a result when the model answered it but the
#: answer could not be read, even one page at a time (RUNBOOK.md, case 4).
WINDOW_FAILED = "window_failed:"


def failed_jobs(store: JobStore, tenant: str | None = None) -> Iterator[dict[str, Any]]:
    """Every given-up job, as an operator sees it."""
    for record in store.statuses(tenant):
        if record.get("status") == "failed":
            yield _line(record, error=record.get("error"))


def partial_jobs(store: JobStore, tenant: str | None = None) -> Iterator[dict[str, Any]]:
    """Every job with a result whose fields miss windows the model never answered."""
    for record in store.statuses(tenant):
        if record.get("status") not in ("fields_ready", "done"):
            continue
        try:
            result = store.read_result(record["tenant"], record["job_id"])
        except Exception:  # noqa: BLE001 - a record whose result is gone is not this report's
            continue
        flags = (result.get("output") or {}).get("review_flags") or []
        unread = sorted(flag[len(WINDOW_FAILED):] for flag in flags if str(flag).startswith(WINDOW_FAILED))
        if unread:
            yield _line(record, unread_windows=unread)


def _line(record: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"job_id": record.get("job_id"), "tenant": record.get("tenant"), "status": record.get("status"),
            "attempts": record.get("attempts"), "submitted": record.get("submitted"),
            "finished": record.get("finished"), **extra}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m azure_service.jobs",
                                     description="List the jobs that need a person")
    parser.add_argument("which", choices=("failed", "partial"))
    parser.add_argument("--tenant", default=None, help="one tenant's jobs only")
    args = parser.parse_args(argv)

    from artifact_registry.blob_client import BlobError
    from azure_service.settings import Settings, SettingsError, _tenant_name
    from azure_service.store import jobs_client

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        tenant = _tenant_name(args.tenant) if args.tenant else None
        store = JobStore(jobs_client(settings))
        lister = failed_jobs if args.which == "failed" else partial_jobs
        count = 0
        for line in lister(store, tenant):
            print(json.dumps(line, ensure_ascii=False))
            count += 1
        print(f"{count} {args.which} job(s)", file=sys.stderr)
        return 0
    except (SettingsError, BlobError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
