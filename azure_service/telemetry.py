"""Five numbers per finished job, for whoever watches the service (review S4, decision 2026-11-25).

Logs only until the first deployment: :class:`LogTelemetry` writes one line per job. The
numbers go through :class:`Telemetry`, so Application Insights or OpenTelemetry can be
plugged in later without touching the worker. Never a document's content, name or values:
counts and seconds only (the repository's rule, no personal data in logs).

The five: the seconds each stage took, how long the job waited in the queue, how many
windows of the document could not be read, how many values were taken out because no page
prints them, and how many guards could not run (the prompt-length check, review R4).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

#: The review flags these numbers are counted from (serving.pipeline, serving.printed_values).
WINDOW_FAILED = "window_failed:"
NOT_PRINTED_REMOVED = ":not_printed_removed"


@dataclass(frozen=True)
class JobNumbers:
    job_id: str
    tenant: str
    #: ``done`` or ``failed``.
    outcome: str
    stage_seconds: dict[str, float] = field(default_factory=dict)
    queue_seconds: float | None = None
    windows_failed: int = 0
    values_removed: int = 0
    guards_not_run: int = 0


class Telemetry(Protocol):
    def job_finished(self, numbers: JobNumbers) -> None: ...


class LogTelemetry:
    """One log line per job (``job numbers: {...}``): the default until a backend is chosen."""

    def job_finished(self, numbers: JobNumbers) -> None:
        log.info("job numbers: %s", json.dumps(asdict(numbers), sort_keys=True))


class NoTelemetry:
    """Nothing emitted."""

    def job_finished(self, numbers: JobNumbers) -> None:
        return None


def job_numbers(record: Mapping[str, Any], outcome: str, result: Any = None) -> JobNumbers:
    """The five numbers of a finished job, from its status record and, when it was read to
    the end, its result (``serving.pipeline.ExtractionResult``). Seconds and counts only."""
    timings = {str(k): round(float(v), 2) for k, v in (record.get("timings_s") or {}).items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    flags = list(getattr(result, "review_flags", None) or []) if result is not None else []
    stats = list(getattr(result, "window_stats", None) or []) if result is not None else []
    return JobNumbers(
        job_id=str(record.get("job_id")),
        tenant=str(record.get("tenant")),
        outcome=outcome,
        stage_seconds={k: v for k, v in timings.items() if k != "queued"},
        queue_seconds=timings.get("queued"),
        windows_failed=sum(1 for flag in flags if flag.startswith(WINDOW_FAILED)),
        values_removed=sum(1 for flag in flags if flag.endswith(NOT_PRINTED_REMOVED)),
        guards_not_run=sum(1 for stat in stats if isinstance(stat, Mapping) and stat.get("prompt_check_error")),
    )
