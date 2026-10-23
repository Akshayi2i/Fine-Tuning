"""Which release serves a document (arch v2.1 §12.3; decision 2026-10-22).

**One promoted release per tenant.** Every fine-tuning cycle trains the one
policy adapter (:mod:`common.adapter`) and takes the next version; ``package``
promotes it when its gate passed and archives the release it replaces, and
``promote --version vK`` puts an older one back (a rollback). The release in the
tenant's index marked ``promoted`` is the one served. Should the index ever hold
several, the newest answers and a warning says so.

**A type the release was not trained on is refused, not extracted.** The
alternative is extracting it with whichever model is loaded, against a schema
that model never trained on, and returning confident nonsense — which nothing
downstream reports as anything but a bad extraction.

**A line the release was not trained on goes to the base model.** The release
records the lines of business it trained on (single-line policies only). A
policy of another line - or a package of several lines, read against
``_fallback.json``, a shape the adapter is not taught - is read by the base
model with no LoRA against ``_fallback.json``, flagged ``lob_fallback_used``,
rather than by an adapter that never saw it. A policy with no line, or a line
with no schema, is refused unless the caller asks for that fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES

log = logging.getLogger(__name__)


class UnservedDocType(RuntimeError):
    """Raised when the promoted release does not cover a document."""


class ServingPlanError(RuntimeError):
    """Raised when the promoted release cannot be served."""


@dataclass(frozen=True)
class ServedRelease:
    """The promoted release, as serving needs it."""

    release_id: str
    base_model: str
    merged_model: str
    #: Which types this release answers for. Empty means every active type —
    #: which is what a bundle written before types were recorded means.
    doc_types: tuple[str, ...] = ()
    adapter: str | None = None
    #: Where its LoRA weights are in Blob, for an engine serving the base with
    #: the release as its adapter.
    adapter_dir: str | None = None
    calibrators: str | None = None
    prompt_hash: str = ""
    #: The window plan its corpus was cut with (``ReleaseBundle.window_plan_version``).
    window_plan_version: str | None = None
    ocr_pin: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    #: Lines of business this release serves: the lines it trained on. Empty =
    #: every line (a bundle written before lines were recorded).
    lines: tuple[str, ...] = ()
    #: ``lora_on_base``: the engine holds the base model and applies this
    #: release's adapter per request - never merged (every release since
    #: 2026-10-23). ``merged``: its merged model is the engine (older bundles).
    served_as: str = "merged"

    @property
    def covers(self) -> tuple[str, ...]:
        return self.doc_types or tuple(ACTIVE_DOC_TYPES)

    def covers_lob(self, lob: object) -> bool:
        """Every line when unrestricted (an older bundle that records none);
        otherwise a policy of ONE line the release trained on. Never a package:
        it is read against ``_fallback.json``, a shape the adapter is not taught."""
        if not self.lines:
            return True
        from common.lob import single_line

        return single_line(lob) in set(self.lines)

    def describe(self) -> str:
        lines = f", lines {list(self.lines)}" if self.lines else ""
        return f"{self.release_id} covering {list(self.covers)}{lines}"


@dataclass(frozen=True)
class Routed:
    """Where one document goes: the release, or the base model with no LoRA.

    ``lob_fallback_used`` means the base model reads the policy against the
    canonical ``_fallback.json``: the release did not train on its line, nothing
    is promoted, or the caller chose the fallback for a line that cannot be
    routed (Fideon SPEC_06 §9a, handoff item 4).
    """

    release: ServedRelease | None
    lob_fallback_used: bool = False


def _unroutable(lob: object) -> str | None:
    """Why a policy with this line cannot be routed, or None when it can."""
    from common.lob import known_lines, lob_lines

    lines = lob_lines(lob)
    if not lines:
        return ("a policy with no line of business is not routed: its schema cannot be chosen. "
                "Send its `lob`, or ask for the base-model fallback (allow_lob_fallback) to "
                "read it against _fallback.json.")
    unknown = sorted(lines - known_lines())
    if unknown:
        return (f"line(s) of business {unknown} have no canonical policy schema; ask for the "
                "base-model fallback (allow_lob_fallback) to read the policy against "
                "_fallback.json.")
    return None


@dataclass
class ServingPlan:
    """What the endpoint serves: the promoted release, or nothing yet."""

    release: ServedRelease | None = None

    @property
    def served(self) -> list[ServedRelease]:
        """The release this plan routes to, as a list of none or one."""
        return [self.release] if self.release else []

    @property
    def served_doc_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.release.covers)) if self.release else ()

    @property
    def unserved_doc_types(self) -> tuple[str, ...]:
        return tuple(sorted(set(ACTIVE_DOC_TYPES) - set(self.served_doc_types)))

    def release_for(self, doc_type: str, lob: object = None) -> ServedRelease:
        """The release that answers for this document, or raise.

        A policy also needs its lines covered: a line the release never trained
        on is refused here (``route`` sends it to the base model instead).
        """
        release = self.release
        if release is None or doc_type not in release.covers:
            raise UnservedDocType(
                f"no promoted release covers {doc_type!r}. Served types are "
                f"{list(self.served_doc_types)}. Extracting it anyway would run a model that "
                "never trained on this type against a schema it has never seen, and return a "
                "confident answer nobody can tell apart from a real one."
            )
        if doc_type == "policy" and not release.covers_lob(lob):
            raise UnservedDocType(
                f"{release.release_id} serves policies of {list(release.lines)}, not line of "
                f"business {lob!r}: a policy of another line - or with no line given - would be "
                "read by a model that never trained on it."
            )
        return release

    def route(self, doc_type: str, lob: object = None, *, allow_fallback: bool = False) -> Routed:
        """Where this document goes (Fideon SPEC_06 §9a, handoff item 4).

        A policy of a line the release trained on goes to the release. Any other
        known line, a package of known lines - or every line, while nothing is
        promoted - is read by the base model with no LoRA against
        ``_fallback.json`` (``lob_fallback_used``).
        A policy with no line, or a line with no schema, is never routed
        silently: refused, unless the caller chose the fallback - or an older
        release recording no lines is promoted, which reads any line against the
        fallback, as it was trained to. Other types route as
        :meth:`release_for` does.
        """
        if doc_type != "policy":
            return Routed(self.release_for(doc_type, lob))
        problem = _unroutable(lob)
        if problem:
            release = self.release
            if release is not None and "policy" in release.covers and not release.lines:
                return Routed(release)
            if allow_fallback:
                return Routed(None, lob_fallback_used=True)
            raise UnservedDocType(problem)
        try:
            return Routed(self.release_for(doc_type, lob))
        except UnservedDocType:
            # A known line the release does not cover: read by the base model,
            # flagged, rather than refused or given to an adapter that never saw it.
            return Routed(None, lob_fallback_used=True)

    def describe(self) -> str:
        if self.release is None:
            return "nothing promoted"
        return self.release.describe()


def _as_release(bundle: dict[str, Any], fmt: str) -> ServedRelease:
    from common.run_ids import RunIdError, version_of

    serving_formats = bundle.get("serving_formats") or {}
    adapter = bundle.get("adapter")
    scope = str(bundle.get("scope") or "unified")
    served_as = str(bundle.get("served_as") or "merged")
    try:
        adapter_dir = (paths.scoped_adapter_dir(None if scope == "unified" else scope,
                                                version_of(adapter)) if adapter else None)
    except (RunIdError, ValueError):
        adapter_dir = None
    if served_as == "lora_on_base":
        # The format's artifact IS the adapter: nothing merged to load. A format
        # the release was not packaged in has nothing to serve (no calibrator,
        # no gate run), so it is not served from the bf16 adapter either.
        adapter_dir = serving_formats.get(fmt)
        merged_model = ""
    else:
        merged_model = str(serving_formats.get(fmt) or bundle.get("merged_model", ""))
    return ServedRelease(
        release_id=str(bundle.get("release_id", "")),
        base_model=str(bundle.get("base_model", "")),
        merged_model=merged_model,
        served_as=served_as,
        doc_types=tuple(bundle.get("doc_types") or ()),
        adapter=adapter,
        adapter_dir=adapter_dir,
        calibrators=(bundle.get("calibrators") or {}).get(fmt),
        prompt_hash=str(bundle.get("prompt_hash", "")),
        window_plan_version=bundle.get("window_plan_version"),
        ocr_pin=dict(bundle.get("ocr_pin") or {}),
        created_at=str(bundle.get("created_at", "")),
        lines=tuple(bundle.get("lines") or ()),
    )


def promoted_rows(client: BlobClient, tenant_id: str | None = None) -> list[dict[str, Any]]:
    """The tenant's release-index rows marked promoted, newest first."""
    index_key = paths.release_index(tenant_id)
    if not client.exists(index_key):
        return []
    rows = [row for row in client.read_json(index_key) or [] if row.get("status") == "promoted"]
    return sorted(rows, key=lambda r: (str(r.get("created_at", "")), str(r.get("release_id", ""))),
                  reverse=True)


def build_serving_plan(
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    fmt: str = "bf16",
) -> ServingPlan:
    """Read the release index and pick the release that serves."""
    plan = ServingPlan()
    rows = promoted_rows(client, tenant_id)
    if not rows:
        log.warning("no release is promoted in %s — nothing is served by a release yet",
                    paths.release_index(tenant_id))
        return plan
    for row in rows:
        release_id = str(row.get("release_id", ""))
        bundle_key = paths.release_bundle(release_id, tenant_id)
        if not client.exists(bundle_key):
            log.warning("release %s is indexed as promoted but has no bundle", release_id)
            continue
        release = _as_release(client.read_json(bundle_key), fmt)
        servable = release.adapter_dir if release.served_as == "lora_on_base" else release.merged_model
        if not servable:
            log.warning("release %s serves no %s artifact; skipping", release_id, fmt)
            continue
        plan.release = release
        break
    if len(rows) > 1:
        log.warning(
            "%d releases are promoted (%s); serving %s, the newest that can be served. "
            "`promote --version` leaves one promoted.", len(rows),
            ", ".join(str(r.get("release_id")) for r in rows),
            plan.release.release_id if plan.release else "none of them",
        )
    if plan.unserved_doc_types:
        log.warning(
            "the promoted release does not cover %s — documents of those types will be refused "
            "rather than extracted by a model that never trained on them.",
            list(plan.unserved_doc_types),
        )
    log.info("serving plan: %s", plan.describe())
    return plan
