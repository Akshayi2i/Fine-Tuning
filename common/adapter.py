"""The one adapter every fine-tuning cycle trains (decisions 2026-10-22, 2026-10-23).

Each cycle trains **one LoRA on the bf16 base** over every single-line policy
in the tenant's corpus, new and existing. By default it **continues the
promoted adapter**: the promoted version's weights, trained further, on the same
original base - never on a merged model (``orchestration.pipeline_dag.continuation``).
It starts from the base instead when asked (``--fresh``: after a data deletion,
or to reset drift), when nothing is promoted, or when the promoted adapter no
longer fits: another base model, LoRA rank, alpha or target modules, or another
prompt hash or window plan (a schema or template change).

Nothing is merged: the release IS the adapter, served as a LoRA on the base
(``ReleaseBundle.served_as = "lora_on_base"``). The cycle takes the next version
number (``v1``, ``v2``, ..., :func:`artifact_registry.paths.next_version`), and
that one number names the run (``policy-v3``), the adapter and the release.

What it covers is fixed here, not configured: policy documents, the policy
tasks, the ``unified`` training config. There are no scopes and no layout
families to choose between.

The release records the lines of business the adapter trained on. Serving
sends a policy of any other line - or a package, or a policy with no line - to
the base model against ``_fallback.json``, flagged ``lob_fallback_used``.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from functools import lru_cache

from common.run_ids import build_run_id
from common.tasks import CORPUS_TASKS, Task


class AdapterError(RuntimeError):
    """Raised when a corpus cannot train the adapter as it stands."""


@dataclass(frozen=True)
class Adapter:
    """What the adapter trains on, and how its artifacts are named."""

    #: The run-id lineage (``policy-v3``) and the artifact path segment
    #: (``adapters/scope/policy/v3/``, the layout the policy runs always used).
    name: str = "policy"

    #: The document types it trains and serves.
    doc_types: tuple[str, ...] = ("policy",)

    #: The tasks its corpus rows carry. They decide the run's ``max_length``.
    #: ``classify`` stays: a policy release still has to recognise a document
    #: that is not a policy and refuse it.
    tasks: tuple[Task, ...] = (
        Task.CLASSIFY, Task.EXTRACT,
        Task.POLICY_DECLARATIONS, Task.POLICY_SCHEDULE, Task.POLICY_ENDORSEMENTS,
    )

    #: Which ``configs/training/{name}.yaml`` the run reads.
    training_config: str = "unified"

    def run_id(self, version: str) -> str:
        """``v3`` -> ``policy-v3``."""
        return build_run_id(self.name, version)


ADAPTER = Adapter()


@lru_cache(maxsize=1)
def not_applicable_metrics() -> frozenset[str]:
    """Gate metrics nothing in this adapter's corpus can produce.

    Derived from what the adapter trains rather than declared, because a
    declared list is a waiver with extra steps. Each entry answers "which
    document or task would have produced this?" with "none here".
    """
    absent = {
        # Reconciliation reads a Loss Run's printed totals; the adapter reads
        # policies only.
        "lossrun_totals_reconciliation_rate", "table_f1",
        # Every policy is canonical, and its line comes from metadata: there is
        # no line for the model to detect.
        "lob_detection_accuracy",
    }
    if Task.CLASSIFY not in set(ADAPTER.tasks) & CORPUS_TASKS:
        # Declared, but the corpus builds no classify row, so it is not trained
        # and cannot be scored.
        absent.add("doc_type_classifier_accuracy")
    if not _has_list_field():
        absent.update({"list_field_recall", "field_f1_list_fields"})
    return frozenset(absent)


def _has_list_field() -> bool:
    """Whether any policy schema declares a table of rows."""
    from common.schemas import resolved_schema, schema_selectors

    for doc_type, acord_form, lob in schema_selectors():
        if doc_type not in ADAPTER.doc_types:
            continue
        properties = (resolved_schema(doc_type, acord_form, lob) or {}).get("properties", {})
        for definition in properties.values():
            if not isinstance(definition, dict):
                continue
            items = definition.get("items")
            if (definition.get("type") == "array" and isinstance(items, dict)
                    and (items.get("type") == "object" or "properties" in items)):
                return True
    return False


def assert_one_output_shape(lobs: Collection[object]) -> None:
    """Refuse training documents that would teach the adapter two output shapes.

    A policy read against a schema on the SPEC_21 common model is answered in
    one shape, one read against a self-contained schema file in another. One
    adapter trained on both learns neither well, and nothing downstream would
    say so. Each of ``lobs`` is one trained document's line: a line, a list of
    lines (a package, read against ``_fallback.json``), or None (no line, read
    against ``_fallback.json`` too) - judged by the schema it is read against,
    as :func:`common.schemas.is_common_model` selects it.
    """
    from common.schemas import is_common_model

    def label(lob: object) -> str:
        if lob is None:
            return "no line"
        return "+".join(sorted(lob)) if isinstance(lob, (list, tuple)) else str(lob)

    new: set[str] = set()
    old: set[str] = set()
    for lob in lobs:
        value = list(lob) if isinstance(lob, (list, tuple)) else lob
        (new if is_common_model("policy", None, value) else old).add(label(lob))
    if old and new:
        raise AdapterError(
            f"the training documents include policies read against the common model "
            f"({', '.join(sorted(new))}) and policies read against self-contained schema files "
            f"({', '.join(sorted(old))}): one adapter would learn two output shapes. Move those "
            "lines onto the common model, or leave their documents out of the training set."
        )
