"""The release step's routing check (Fideon SPEC_09 handoff item 6).

Before a release is written, every line of business is routed as serving will
route it once the release is promoted: to the release's adapter, or to the base
model with ``_fallback.json``. The table goes into the release bundle, and the
release step fails when a line the release was trained for would not reach it.

The lines are this repository's canonical policy schemas
(:func:`common.lob.known_lines`); the 48 L1 codes of the main repository's
``config/lob_registry.yaml`` replace them once that file is here.
"""

from __future__ import annotations

from typing import Any

from serving.release_router import ServingPlan, UnservedDocType, _as_release


class RoutingCheckError(RuntimeError):
    """Raised when a line cannot be routed, or would not reach the release trained for it."""


def plan_with_candidate(bundle: dict[str, Any], fmt: str = "bf16") -> ServingPlan:
    """The plan once ``bundle`` is promoted: it replaces the release served now.

    Whatever the bundle's own status: the table says where each line would go
    if this release is served, which is the question a gated release raises too.
    """
    return ServingPlan(release=_as_release(bundle, fmt))


def routing_table(plan: ServingPlan, release_id: str, release_lines: list[str] | tuple[str, ...]) -> list[dict]:
    """One row per line: where it goes. Raises :class:`RoutingCheckError`."""
    from common.lob import known_lines

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for line in sorted(known_lines()):
        try:
            routed = plan.route("policy", line)
        except UnservedDocType as exc:
            problems.append(f"{line}: {exc}")
            continue
        release = routed.release
        rows.append({
            "lob": line,
            "release_id": release.release_id if release else None,
            "adapter": release.adapter if release else None,
            "lob_fallback_used": routed.lob_fallback_used,
        })
        if line in release_lines and (release is None or release.release_id != release_id):
            problems.append(f"{line} is a line {release_id} was trained for, but routes to "
                            f"{release.release_id if release else 'the base model'}")
    if problems:
        raise RoutingCheckError("routing check failed: " + "; ".join(problems))
    return rows
