"""Training scopes — WHAT one run covers (arch v2.1 §4.1, §4.2).

A scope names a training run's coverage: which document types it trains, which
tasks its corpus carries, and which document types a release from it may serve.
Each scope is an **independent LoRA on the bf16 base** with its own run id,
checkpoints, merge, quantization, calibration, gate and release bundle. Scopes
are never stacked — vLLM applies one LoRA per request, the same constraint that
made v1's Foundation-plus-per-type topology unservable.

**Why a named scope rather than a bare ``--doc-types`` list.** The name is the
addressing key: it is the run-id lineage and the artifact path segment. Two
different runs over the same type set (a policy run with windowed tasks and one
without) need different names, and a list cannot give them one. A list also
carries no floors and no task set, so the gate would have nothing scope-specific
to read.

**The unified scope keeps the lineage ``extractor``**, so every artifact already
in Blob stays addressable and every unified path renders byte-identically. That
is the compatibility pin the whole design rests on.

**Not-applicable is derived, never simply trusted.** ``CONDITIONAL_METRICS`` in
:mod:`evaluation.gating` already carries a warning that it is a loophole if
widened casually. A scope may declare a metric not applicable only if the
scope's own *shape* implies it — see :func:`structural_not_applicable`. Without
that rule, "configure a scope" would become "waive a gate".
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from common.constants import ACTIVE_DOC_TYPES
from common.run_ids import UNIFIED_LINEAGE, build_run_id
from common.tasks import CORPUS_TASKS, Task, parse

#: The scope every existing artifact belongs to. Its name is load-bearing:
#: unified renders today's paths and today's run ids exactly.
UNIFIED = "unified"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ScopeError(RuntimeError):
    """Raised on an unknown, malformed or incoherent scope."""


@dataclass(frozen=True)
class Scope:
    """One training run's coverage."""

    name: str

    #: The run-id lineage: ``extractor-v2`` for unified, ``policy-v2`` otherwise.
    lineage: str

    #: The document types this run TRAINS. A subset of ``ACTIVE_DOC_TYPES``.
    doc_types: tuple[str, ...]

    #: The tasks its corpus rows carry. Decides the run's ``max_length`` and
    #: which gating metrics have no producer.
    tasks: tuple[Task, ...]

    #: Which ``configs/training/{name}.yaml`` the run reads.
    training_config: str = "unified"

    #: Types a release from this scope may serve. Defaults to ``doc_types``:
    #: serving what was never trained is the failure this exists to prevent.
    serves: tuple[str, ...] = ()

    #: Per-scope overrides on the §0d promotion floors.
    floors: dict[str, float] = field(default_factory=dict)

    #: Metrics this scope structurally cannot produce. Narrowed at load time to
    #: what :func:`structural_not_applicable` allows.
    not_applicable_metrics: frozenset[str] = frozenset()

    #: The lines of business this scope covers, as canonical schema names
    #: (``homeowners``, ``wc``). Empty means every line. Only a policy-only scope
    #: can narrow by line: the line is a policy's, read from its metadata.
    lines: frozenset[str] = frozenset()

    @property
    def is_unified(self) -> bool:
        return self.name == UNIFIED

    @property
    def path_segment(self) -> str:
        """The artifact path segment.

        ``unified`` renders exactly what every existing path renders; anything
        else lands under ``scope/{name}/``, which is a namespace of its own so a
        policy-SCOPE adapter never collides with a graduated per-type policy
        adapter at ``adapters/policy/``. They are different artifacts.
        """
        return UNIFIED if self.is_unified else f"scope/{self.name}"

    def run_id(self, version: str) -> str:
        """``v2`` -> ``extractor-v2`` for unified, ``policy-v2`` otherwise."""
        return build_run_id(self.lineage, version)

    def covers(self, doc_type: str) -> bool:
        return doc_type in self.serves

    def covers_lob(self, lob: object) -> bool:
        """Whether a document with this line of business belongs to the scope.

        Every scope without ``lines`` covers every line. A line-scoped one covers
        a document only when ALL its lines are in scope: a package policy with
        one personal and one commercial line is not a personal-lines document.
        A document with no recorded line is outside a line-scoped scope — its
        line cannot be shown to be one the model trained on. Nor is a package of
        common-model lines (:func:`common_model_package`): it is read against
        another shape than the one the scope's adapter learns.
        """
        if not self.lines:
            return True
        found = lob_lines(lob)
        return bool(found) and found <= self.lines and not common_model_package(found)

    def describe(self) -> str:
        return (
            f"scope {self.name}: {len(self.doc_types)} doc type(s) "
            f"{list(self.doc_types)}, {len(self.tasks)} task(s), lineage {self.lineage}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Recorded on the run manifest and the release bundle."""
        return {
            "name": self.name,
            "lineage": self.lineage,
            "doc_types": list(self.doc_types),
            "tasks": [str(t) for t in self.tasks],
            "training_config": self.training_config,
            "serves": list(self.serves),
            "not_applicable_metrics": sorted(self.not_applicable_metrics),
            "lines": sorted(self.lines),
        }


# --------------------------------------------------------------------------
# Structural not-applicable
# --------------------------------------------------------------------------


def common_model_package(lines: frozenset[str] | set[str]) -> bool:
    """Whether a document naming these lines is a package of common-model lines.

    A package names several lines, so it is read against the client's
    ``_fallback.json`` (common.schemas.schema_key selects a line's schema only
    when exactly one is named) - the old shape. When its lines are SPEC_21
    common-model lines, their adapter learns another shape entirely, so neither
    training nor serving gives such a package to it: it is read as any other
    package is (an unrestricted release, else the base model).
    """
    if len(lines) < 2:
        return False
    from common.schemas import is_common_model

    return any(is_common_model("policy", None, line) for line in lines)


def lob_lines(lob: object) -> frozenset[str]:
    """A document's line(s) of business as canonical schema names.

    Accepts the enum spelling or the schema name (``workers_comp`` and ``wc`` are
    one line), one line or a list of them. A merged line is read as the line it
    is (``classic_auto`` is ``personal_auto``, common.lob.MERGED_LINES), so a
    corpus row or a request written before the merge is still in scope.
    """
    from common.lob import merge_line
    from common.schemas import LOB_SCHEMA_ALIASES

    if lob is None:
        return frozenset()
    values = [lob] if isinstance(lob, str) else list(lob) if isinstance(lob, (list, tuple)) else []
    out = set()
    for value in values:
        line = str(merge_line(str(value).strip().lower()))
        if line:
            out.add(LOB_SCHEMA_ALIASES.get(line, line))
    return frozenset(out)


def known_lines() -> frozenset[str]:
    """Every line with a canonical policy schema of its own."""
    from common.schemas import schema_selectors

    return frozenset(q for doc_type, _f, q in schema_selectors() if doc_type == "policy" and q)


def structural_not_applicable(scope: Scope) -> frozenset[str]:
    """Metrics this scope's SHAPE means nothing can produce.

    Derived from the scope rather than declared by it, because a declared list is
    a waiver with extra steps. Each entry below answers "which document or task
    would have produced this?" with "none in this scope".

    Lives here rather than in :mod:`evaluation.gating` so that scope config can
    be validated at load time without ``common`` importing ``evaluation``.
    """
    absent: set[str] = set()
    # A task the scope declares but the corpus builds no rows for is not
    # trained, so it cannot be scored either. The classifier metric was a
    # required gate while no classify row existed, which blocked every candidate.
    trained = set(scope.tasks) & CORPUS_TASKS

    if "lossrun" not in scope.doc_types:
        # Reconciliation reads a Loss Run's printed totals. No Loss Run, no
        # totals — that is a statement about the eval set, not a pass.
        absent.add("lossrun_totals_reconciliation_rate")
        absent.add("table_f1")

    if Task.CLASSIFY not in trained:
        # A scope that never classifies cannot be scored on classification. Note
        # that a single-type scope SHOULD still classify: it has to recognise the
        # types it does not serve and refuse them.
        absent.add("doc_type_classifier_accuracy")

    if all(doc_type == "policy" for doc_type in scope.doc_types):
        # Every policy is canonical, and a canonical label has no line of
        # business for the model to detect — its line comes from metadata. A
        # policy-only scope has nothing to score LOB detection on.
        absent.add("lob_detection_accuracy")

    if not _has_list_field(scope.doc_types):
        # No repeating structure in any of this scope's schemas, so row recall
        # and list F1 have nothing to count.
        absent.update({"list_field_recall", "field_f1_list_fields"})

    return frozenset(absent)


def _has_list_field(doc_types: tuple[str, ...]) -> bool:
    """Whether any of these types' schemas declares a table of rows."""
    from common.schemas import resolved_schema, schema_selectors

    for doc_type, acord_form, lob in schema_selectors():
        if doc_type not in doc_types:
            continue
        properties = (resolved_schema(doc_type, acord_form, lob) or {}).get("properties", {})
        for definition in properties.values():
            if not isinstance(definition, dict):
                continue
            items = definition.get("items")
            is_table = (
                definition.get("type") == "array"
                and isinstance(items, dict)
                and (items.get("type") == "object" or "properties" in items)
            )
            if is_table:
                return True
    return False


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _build(name: str, body: dict[str, Any]) -> Scope:
    if not _NAME_RE.match(name):
        raise ScopeError(
            f"invalid scope name {name!r}; expected lowercase alphanumeric with underscores "
            "(it becomes a run-id lineage and an artifact path segment)"
        )

    doc_types = tuple(str(d).strip().lower() for d in body.get("doc_types") or ())
    if not doc_types:
        raise ScopeError(f"scope {name!r} trains no document types")
    unknown = [d for d in doc_types if d not in ACTIVE_DOC_TYPES]
    if unknown:
        raise ScopeError(
            f"scope {name!r} names {unknown}, which are not active document types "
            f"{list(ACTIVE_DOC_TYPES)}. A scope narrows the active set; it cannot widen it."
        )

    try:
        tasks = tuple(parse(t) for t in body.get("tasks") or ())
    except Exception as exc:  # noqa: BLE001 - re-raised as a scope error with context
        raise ScopeError(f"scope {name!r} names an unknown task: {exc}") from exc
    if not tasks:
        raise ScopeError(f"scope {name!r} declares no tasks")

    serves = tuple(str(d).strip().lower() for d in body.get("serves") or doc_types)
    over_served = [d for d in serves if d not in doc_types]
    if over_served:
        raise ScopeError(
            f"scope {name!r} would serve {over_served}, which it does not train. A release "
            "serving a type its model never saw returns confident nonsense."
        )

    lineage = str(body.get("lineage") or name).strip().lower()
    if name == UNIFIED and lineage != UNIFIED_LINEAGE:
        raise ScopeError(
            f"the unified scope must keep the lineage {UNIFIED_LINEAGE!r}: every artifact "
            f"already in Blob is addressed as {UNIFIED_LINEAGE}-v{{n}}, and renaming it would "
            "strand them."
        )
    build_run_id(lineage, "v1")  # validates the lineage against the run-id grammar

    scope = Scope(
        name=name,
        lineage=lineage,
        doc_types=doc_types,
        tasks=tasks,
        training_config=str(body.get("training_config") or "unified"),
        serves=serves,
        floors={str(k): float(v) for k, v in (body.get("floors") or {}).items()},
        not_applicable_metrics=frozenset(
            str(m) for m in body.get("not_applicable_metrics") or ()
        ),
        lines=lob_lines(body.get("lines") or ()),
    )

    if scope.lines:
        if doc_types != ("policy",):
            raise ScopeError(
                f"scope {name!r} narrows by line of business but trains {list(doc_types)}. A line "
                "is a policy's (from its metadata); ACORD forms and Loss Runs carry none to filter on."
            )
        unknown_lines = sorted(scope.lines - known_lines())
        if unknown_lines:
            raise ScopeError(
                f"scope {name!r} names lines {unknown_lines} with no canonical policy schema. "
                f"Known lines: {sorted(known_lines())}"
            )

    over_declared = sorted(scope.not_applicable_metrics - structural_not_applicable(scope))
    if over_declared:
        raise ScopeError(
            f"scope {name!r} declares {over_declared} not applicable, but its own shape says "
            "otherwise — it trains a document type or task that produces them. A metric this "
            "scope CAN produce and does not is unmeasured, which is a block, not a waiver "
            "(arch v2.1 §15.2)."
        )
    return scope


def assert_one_output_shape(scope: Scope, lines_present: Collection[str] | None = None) -> None:
    """Refuse a scope that would train on two output shapes: documents of a
    layout family's lines already on the common model and of its lines still on
    self-contained files (:func:`common.schemas.partly_migrated_families`).

    ``lines_present`` - the lines the corpus holds documents of (the corpus
    manifest's ``policy_line_counts``) - narrows it to the shapes actually there:
    a scope covering general liability trains while the corpus holds no
    general-liability documents, or none of its family's other lines. Without
    it, covering both kinds of line is enough to refuse.
    """
    from common.config import lobs_in_family
    from common.schemas import partly_migrated_families

    if "policy" not in scope.doc_types:
        return
    for family, still_old in partly_migrated_families().items():
        lines = set(lobs_in_family(family))
        covered = lines & set(scope.lines) if scope.lines else lines
        if lines_present is not None:
            covered &= set(lines_present)
        old, new = covered & set(still_old), covered - set(still_old)
        if old and new:
            raise ScopeError(
                f"scope {scope.name!r} would train layout family {family!r} on two output shapes: "
                f"{', '.join(sorted(new))} on the common model and {', '.join(sorted(old))} on "
                "self-contained files. Train it once the whole family has moved, or narrow its lines."
            )


@lru_cache(maxsize=1)
def load_scopes() -> dict[str, Scope]:
    """Every scope in ``configs/scopes.yaml``, validated."""
    from common.config import ConfigError, scopes_config

    raw = scopes_config()
    declared = raw.get("scopes") or {}
    if not isinstance(declared, dict) or not declared:
        raise ScopeError("configs/scopes.yaml declares no scopes")

    scopes = {name: _build(name, body or {}) for name, body in declared.items()}

    if UNIFIED not in scopes:
        raise ScopeError(
            "configs/scopes.yaml has no 'unified' scope. It is what every existing artifact "
            "belongs to, and what the default command trains."
        )

    lineages: dict[str, str] = {}
    for scope in scopes.values():
        clash = lineages.get(scope.lineage)
        if clash:
            raise ScopeError(
                f"scopes {clash!r} and {scope.name!r} share the lineage {scope.lineage!r}, so "
                "their run ids would collide and each would resolve to the other's artifacts"
            )
        lineages[scope.lineage] = scope.name

    default = str(raw.get("default_scope") or UNIFIED)
    if default not in scopes:
        raise ConfigError(f"default_scope {default!r} is not a declared scope")
    return scopes


def get_scope(name: str) -> Scope:
    """One scope by name."""
    scopes = load_scopes()
    try:
        return scopes[name.strip().lower()]
    except KeyError:
        raise ScopeError(
            f"unknown scope {name!r}; declared scopes are {sorted(scopes)}. Add it to "
            "configs/scopes.yaml rather than passing document types ad hoc — the name is the "
            "run-id lineage and the artifact path segment."
        ) from None


def default_scope() -> Scope:
    """The scope a command trains when none is named."""
    from common.config import scopes_config

    return get_scope(str(scopes_config().get("default_scope") or UNIFIED))


def parse_scopes(names: list[str] | tuple[str, ...] | None) -> tuple[Scope, ...]:
    """Resolve ``--scope`` arguments, in order, with duplicates refused."""
    if not names:
        return (default_scope(),)
    seen: dict[str, Scope] = {}
    for name in names:
        scope = get_scope(name)
        if scope.name in seen:
            raise ScopeError(f"scope {scope.name!r} was named twice")
        seen[scope.name] = scope
    return tuple(seen.values())


def narrow(scope: Scope, doc_types: list[str] | tuple[str, ...] | None) -> Scope:
    """A scope restricted to some of its own types, for a one-off run.

    Narrowing only. Widening would produce artifacts named for a scope that were
    built from data it does not describe — and the name is what every path, run
    id and release bundle is addressed by.
    """
    if not doc_types:
        return scope
    wanted = tuple(str(d).strip().lower() for d in doc_types)
    outside = [d for d in wanted if d not in scope.doc_types]
    if outside:
        raise ScopeError(
            f"--doc-types {outside} is outside scope {scope.name!r} ({list(scope.doc_types)}). "
            "A scope can be narrowed for a one-off run but never widened: artifacts named "
            f"{scope.name!r} must describe what that scope covers."
        )
    return Scope(
        name=scope.name,
        lineage=scope.lineage,
        doc_types=wanted,
        tasks=scope.tasks,
        training_config=scope.training_config,
        serves=tuple(d for d in scope.serves if d in wanted),
        floors=dict(scope.floors),
        not_applicable_metrics=scope.not_applicable_metrics,
        # A line-scoped scope stays line-scoped: dropping this turned a narrowed
        # personal_lines into a scope covering every line.
        lines=scope.lines,
    )


def assert_scopes_are_coherent() -> None:
    """Load and validate every scope — called from ``config.validate_all``."""
    from evaluation.gating import GATING_METRICS

    for scope in load_scopes().values():
        unknown = sorted(set(scope.floors) - set(GATING_METRICS))
        if unknown:
            raise ScopeError(
                f"scope {scope.name!r} sets floors for {unknown}, which are not gating metrics "
                f"{sorted(GATING_METRICS)}. A floor on a metric the gate never reads is a "
                "number nobody checks."
            )
        if scope.training_config:
            from common.config import training_config

            training_config(scope.training_config)  # raises if the file is missing
