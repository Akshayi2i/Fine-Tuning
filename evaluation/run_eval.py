"""The evaluation driver (IMPL-08 §3, arch §8, §15).

Runs a model version over the **frozen golden eval set** through the IMPL-07
inference core, computes every metric, and writes
``eval-reports/v{n}/{doc_type}/report.json`` plus a top-level summary — broken
down per doc type *and* per modality mode.

Four subsets are scored explicitly on top of the full set: ``image_only``,
``scanned``, ``noisy_ocr`` and ``long_policy``. An aggregate hides exactly the
regressions that matter — image-only accuracy can fall ten points while the
overall number moves two, because image-only is a third of the corpus.

**The eval set is frozen and versioned separately**, one per tenant, human
double-verified, and held constant across corpus versions so model versions
compare like with like. :func:`assert_eval_set_disjoint` enforces the part of
that which code can: **no eval ``source_id`` may appear in any of the tenant's
corpus splits.** Train on your eval set and every number in the registry becomes
a measurement of memorisation, with no symptom anywhere — the loss curve looks
fine and the gate passes.

Per-document **error records** are kept, not just aggregates, because
``vit_gate`` (IMPL-06) needs the perception-vs-reasoning split and failure-mode
analysis needs real material.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES

log = logging.getLogger(__name__)

#: Eval subsets scored separately from the full set (IMPL-08 §3).
EVAL_SUBSETS: tuple[str, ...] = ("image_only", "scanned", "noisy_ocr", "long_policy")

#: Where a reading's metadata carries serving's printed-values record
#: (``ExtractionResult.printed_check``, golden_eval): what hallucination_rate
#: reads for a document served through serving.
PRINTED_CHECK = "printed_check"

#: Where a reading's metadata carries a second answer to the same document, read
#: in another order (golden_eval's repeat pass, validation_generation's): what
#: run_to_run_agreement compares the answer with.
REPEAT_ANSWER = "repeat_extraction"

#: The reading mode whose page text an image-only reading borrows, so its
#: values are checked for invention too (decision 2026-10-26): the same
#: document's clean MinerU text, page by page. Production reads image_only and
#: checks the fields on that text when it arrives (serving.fields_first).
TEXT_DONOR_MODE = "ocr_plus_image"

#: The mode that reads no page text: its readings borrow (:data:`TEXT_DONOR_MODE`).
IMAGE_ONLY_MODE = "image_only"

#: Where a reading's metadata says its printed-values record (:data:`PRINTED_CHECK`)
#: was made on borrowed page text (:data:`BORROWED_TEXT`), not on text it was
#: sent (golden_eval's image_only readings).
PRINTED_TEXT = "printed_text"
BORROWED_TEXT = "borrowed"

#: Where such a reading's metadata lists the pages its values cite that have no
#: text to borrow: what was looked up there is unchecked, never clean.
UNREAD_PAGES = "printed_text_unread_pages"

#: Gate metrics pooled over the whole set by their per-document weights - the
#: values checked, the fields compared - rather than each doc type's rate
#: weighted by its readings: hallucination_rate is every value no page prints
#: over every value checked, wherever they are.
POOLED_METRICS: tuple[str, ...] = ("hallucination_rate", "run_to_run_agreement",
                                   "served_field_match", "real_ocr_plus_image_accuracy")

#: Field match per reading mode on REAL documents (decision 2026-10-26), each
#: the mean over its readings of the share of fields right - the per-reading
#: definition of field_normalized_match - pooled over every doc type's readings
#: (:data:`POOLED_METRICS`). Synthetic twins are left out: their labels came
#: from a generator. ``served_field_match`` is what users receive: production
#: reads the page images (image_only) and golden_eval scores those readings as
#: served, held to the document's borrowed MinerU text. The gate's tier, its
#: improvement rule and non-inferiority read it (``gating.SERVED_FIELD_MATCH``).
REAL_MODE_METRICS: dict[str, str] = {
    "served_field_match": IMAGE_ONLY_MODE,
    "real_ocr_plus_image_accuracy": TEXT_DONOR_MODE,
}


class EvalError(RuntimeError):
    """Raised when an evaluation cannot be trusted or cannot be run."""


class EvalSetLeakage(EvalError):
    """Raised when an eval document also appears in a corpus split.

    Separate from :class:`EvalError` because it invalidates results that already
    exist rather than blocking a run: every metric computed against a leaked eval
    set has to be discarded, not re-run.
    """


# --------------------------------------------------------------------------
# The freeze guarantee
# --------------------------------------------------------------------------


def corpus_source_ids(
    client: BlobClient, corpus_version: str, tenant_id: str | None = None,
    *, include_test: bool = False,
) -> set[str]:
    """Every ``source_id`` a model built from this corpus LEARNED from or was
    SELECTED on — its train and val splits (and the adapter's views of them).

    The test split is excluded unless ``include_test``: no model trains or
    selects on it, and it is exactly what the golden eval set is frozen FROM
    (``freeze-eval-set``). Counting it made the natural sequence — build v1,
    freeze v1's test split, gate the model trained on v1 — fail as "leakage"
    for every frozen document. The group split keeps a frozen document's
    family out of train and val, so this is not a way around the guarantee.
    """
    found: set[str] = set()
    prefix = paths.corpus_dir(corpus_version, tenant_id)
    for key in client.list(prefix):
        if not key.endswith(".jsonl"):
            continue
        parts = key[len(prefix):].strip("/").split("/")
        if not include_test and ("test" in parts[:-1] or parts[-1].startswith("test")):
            continue
        for line in client.read_text(key).splitlines():
            if not line.strip():
                continue
            try:
                found.add(json.loads(line)["source_id"])
            except (ValueError, KeyError):
                continue
    return found


def eval_set_keys(client: BlobClient, tenant_id: str | None = None) -> list[str]:
    """Every key in this tenant's frozen golden eval set, and no other tenant's.

    Listing is by bare prefix, and ``golden-eval-set/acme`` is a prefix of
    ``golden-eval-set/acme-2``: without the boundary one tenant would read
    another's documents as its own.
    """
    root = paths.golden_eval_set_dir(tenant_id) + "/"
    return [key for key in client.list(root) if key.startswith(root)]


def eval_set_source_ids(client: BlobClient, tenant_id: str | None = None) -> set[str]:
    """Every ``source_id`` in this tenant's frozen golden eval set.

    Only this tenant's: source ids are numbered per tenant, so another tenant's
    frozen ``policy_0001`` says nothing about this tenant's ``policy_0001``.
    """
    from evaluation.freeze_eval_set import refuse_unscoped_set

    # A set still at the root would make every tenant's set read as empty here,
    # and the leakage check pass by having nothing to compare.
    refuse_unscoped_set(client)
    root = paths.golden_eval_set_dir(tenant_id) + "/"
    return {
        key[len(root):].split("/")[0]
        for key in eval_set_keys(client, tenant_id)
        if key.endswith("golden.json")
    }


def eval_set_record(
    client: BlobClient, tenant_id: str | None, documents: Iterable[Any]
) -> dict[str, Any]:
    """Which frozen set, of which tenant, an eval report was scored on, under
    which scoring code, and how many of its documents per type (``documents``:
    the documents scored, each with a ``doc_type`` and a ``synthetic`` flag).

    The gate compares two reports, and pairs their per-document scores, only
    when both were scored on one yardstick (:data:`EVAL_SET_IDENTITY`): the same
    set of the same tenant, read and scored by the same code. Source ids are
    numbered per tenant, and a set deleted and frozen again (corrected golds)
    reuses them, so ``policy_0001`` in two reports is not always one document
    against one gold; and a frozen gold is read under the label rules of the
    code that scores it (:func:`scoring_fingerprint`).

    ``real_documents_by_doc_type`` is what the release was scored on, which
    decides whether it was scored on too few documents to be called production:
    real documents only, because a synthetic twin does not count toward the
    minimum (decision 2026-10-22). ``documents_by_doc_type`` counts every
    document scored.
    """
    from evaluation.freeze_eval_set import frozen_manifest

    scored = [(d.doc_type, bool(getattr(d, "synthetic", False))) for d in documents]
    manifest = frozen_manifest(client, tenant_id=tenant_id)
    return {
        "tenant_id": paths._tenant(tenant_id),
        # A set frozen again is a new freeze: its frozen_at differs, whatever
        # corpus it came from.
        "frozen_at": manifest.get("frozen_at"),
        "frozen_from_corpus": manifest.get("frozen_from_corpus"),
        "scoring_fingerprint": scoring_fingerprint(),
        # The mode the gate's run-to-run repeat reads (golden_eval.run_to_run_mode):
        # part of the yardstick, folded into the fingerprint above.
        **yardstick_settings(),
        "documents_by_doc_type": dict(sorted(Counter(t for t, _s in scored).items())),
        "real_documents_by_doc_type": dict(sorted(
            Counter(t for t, synthetic in scored if not synthetic).items())),
    }


#: What :func:`eval_set_record` writes that names the yardstick: the set, and
#: the code that read and scored it. Two reports are scored on one yardstick
#: when all of these agree (:func:`yardstick_difference`).
EVAL_SET_IDENTITY: tuple[str, ...] = (
    "tenant_id", "frozen_at", "frozen_from_corpus", "scoring_fingerprint",
)

#: The code that turns a frozen gold and an extraction into a score, hashed by
#: :func:`scoring_fingerprint`: how the gold is read (the label rules, label
#: verification, and golden_eval, which loads it and builds each mode's input,
#: with the noisy OCR mode's corruption), how values are compared and aligned
#: (evaluation.metrics, and the normalisation, grounding, canonical-value, row
#: and line code they use), and how a report aggregates them (this module).
#:
#: Whole modules, not named functions: a helper a scorer starts calling lives
#: in a module listed here, where a hash of named functions would miss it. A
#: change to the unrelated code these modules also hold moves the fingerprint
#: too, and costs one release judged on its floors alone; pairing two
#: yardsticks as one is the mistake this exists to stop. Left out on purpose:
#: serving and the model, which are the candidate being measured, not the
#: yardstick; and the schemas and configs, which are versioned on their own
#: (``schema_versions``) and move the training targets with them.
SCORING_CODE: tuple[str, ...] = (
    "data_pipeline/ingestion/label_rules.py",
    "data_pipeline/dataset_builder/label_verification.py",
    "data_pipeline/dataset_builder/noisy_ocr_augment.py",
    "evaluation/golden_eval.py",
    "evaluation/run_eval.py",
    "evaluation/metrics/*.py",
    "common/normalize.py",
    "common/grounding.py",
    "common/canonical.py",
    "common/structural_ids.py",
    "common/schema_sections.py",
    "common/aliases.py",
    "common/lob.py",
)


def scoring_fingerprint() -> str:
    """A short hash of the scoring code (:data:`SCORING_CODE`) as it runs now.

    Recorded in every eval report (``eval_set``) and part of
    :data:`EVAL_SET_IDENTITY`. A frozen gold is read under the label rules of
    the running code, not frozen with them: one gold read before and after the
    edition rule was scoped is two answer keys. So two reports of one frozen set
    scored by code with different label rules or scorers are two yardsticks,
    and the gate does not compare them (decision 2026-10-22).

    Hashed from each module's syntax tree with its docstrings left out, not
    from its bytes: a comment, a docstring, a line ending (the pod's checkout is
    LF, a Windows one CRLF) or a re-wrapped line cannot change a score, and a
    fingerprint that moved on them would make every edit a reason to compare
    nothing. Any change to the code itself moves it, including one that turns
    out to change no score.

    The yardstick's configured settings (:func:`yardstick_settings`: the mode the
    run-to-run repeat reads) are folded in: run_to_run_agreement over image_only
    readings is another number than over ocr_plus_image ones, so a report made
    on another repeat mode is neither reused for the gate nor compared with it
    as if comparable - wherever the fingerprint is compared (decision 2026-10-26).
    """
    settings = json.dumps(yardstick_settings(), sort_keys=True)
    return hashlib.sha256(f"{_scoring_fingerprint()}\n{settings}".encode()).hexdigest()[:16]


def yardstick_settings() -> dict[str, Any]:
    """What the configuration sets about how the gate measures, recorded in
    every eval report's ``eval_set`` and folded into :func:`scoring_fingerprint`:
    the mode the golden eval reads a second time for run_to_run_agreement
    (``run_to_run_mode``, ``evaluation.golden_eval.run_to_run_mode``), and the
    shares of the frozen set read in noisy_ocr_image and read twice
    (``golden_noisy_share``, ``run_to_run_share``: below 1.0 another set of
    readings is scored, so another number)."""
    from evaluation.golden_eval import golden_noisy_share, run_to_run_mode, run_to_run_share

    return {"run_to_run_mode": run_to_run_mode(), "golden_noisy_share": golden_noisy_share(),
            "run_to_run_share": run_to_run_share()}


@functools.lru_cache(maxsize=1)
def _scoring_fingerprint() -> str:
    return code_fingerprint(Path(__file__).resolve().parent.parent, SCORING_CODE)


def code_fingerprint(root: Path, patterns: Iterable[str]) -> str:
    """A short hash of what the modules under ``root`` matching ``patterns`` do
    (:func:`code_shape`), in a fixed order."""
    digest = hashlib.sha256()
    for pattern in patterns:
        files = sorted(root.glob(pattern))
        if not files:
            # A module renamed or moved changes the fingerprint; it never just
            # drops out of it (test_the_scoring_fingerprint_covers_real_modules).
            digest.update(f"{pattern}: missing\n".encode())
        for path in files:
            digest.update(f"{path.relative_to(root).as_posix()}\n".encode())
            digest.update(code_shape(path.read_text(encoding="utf-8")).encode())
    return digest.hexdigest()[:16]


def code_shape(source: str) -> str:
    """What a module's code does, as text: its syntax tree without docstrings,
    comments or layout. What :func:`scoring_fingerprint` hashes."""
    return repr(_shape(ast.parse(source)))


_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _shape(node: Any) -> Any:
    if isinstance(node, list):
        return [_shape(item) for item in node]
    if not isinstance(node, ast.AST):
        return node
    fields = []
    for name in node._fields:
        value = getattr(node, name, None)
        if (name == "body" and isinstance(node, _DOCUMENTED) and value
                and isinstance(value[0], ast.Expr) and isinstance(value[0].value, ast.Constant)
                and isinstance(value[0].value.value, str)):
            value = value[1:]
        shaped = _shape(value)
        # Empty and absent fields are left out, so a Python release that adds
        # one (type_params in 3.12) does not move the hash of unchanged code.
        if shaped is None or shaped == []:
            continue
        fields.append((name, shaped))
    return type(node).__name__, fields


def yardstick_difference(
    promoted: dict[str, Any] | None, candidate: dict[str, Any] | None, tenant_id: str | None
) -> str | None:
    """Why the promoted release's and the candidate's eval reports, gating
    ``tenant_id``, were not scored on one yardstick; ``None`` when they were.

    Read from each report's ``eval_set`` (:func:`eval_set_record`). A report
    that does not record every part of :data:`EVAL_SET_IDENTITY` (written
    before reports recorded it) cannot be shown to be on the other's yardstick,
    so it is not.
    """
    tenant = paths._tenant(tenant_id)
    promoted, candidate = promoted or {}, candidate or {}
    for side, record in (("promoted", promoted), ("candidate", candidate)):
        missing = [k for k in EVAL_SET_IDENTITY if not record.get(k)]
        if missing:
            return (f"the {side} report does not record which frozen eval set, tenant and scoring "
                    f"code it was scored on ({', '.join(missing)} missing: written before reports "
                    "recorded them), so it cannot be shown to be on the other's yardstick")
    for side, record in (("promoted release", promoted), ("candidate", candidate)):
        if record["tenant_id"] != tenant:
            return (f"the {side} was scored on tenant {record['tenant_id']!r}'s frozen set, "
                    f"gating tenant {tenant!r}; source ids are numbered per tenant")
    if (promoted["frozen_at"], promoted["frozen_from_corpus"]) != (
            candidate["frozen_at"], candidate["frozen_from_corpus"]):
        return (f"the set was frozen again: the promoted release was scored on the set frozen "
                f"{promoted['frozen_at']} from {promoted['frozen_from_corpus']}, the candidate on "
                f"the set frozen {candidate['frozen_at']} from {candidate['frozen_from_corpus']}, "
                "so one source id may be another document or another gold")
    modes = (promoted.get("run_to_run_mode"), candidate.get("run_to_run_mode"))
    if all(modes) and modes[0] != modes[1]:
        return (f"the two read their run-to-run repeat in different modes (the promoted release "
                f"{modes[0]}, the candidate {modes[1]}), so their run_to_run_agreement are two numbers")
    for share in ("golden_noisy_share", "run_to_run_share"):
        shares = (promoted.get(share), candidate.get(share))
        if all(s is not None for s in shares) and float(shares[0]) != float(shares[1]):
            return (f"the two read different shares of the frozen set ({share}: the promoted release "
                    f"{shares[0]}, the candidate {shares[1]}), so their scores are over other readings")
    if promoted["scoring_fingerprint"] != candidate["scoring_fingerprint"]:
        return (f"the two were scored by different scoring code (label rules, label verification "
                f"or scorers; scoring fingerprint {promoted['scoring_fingerprint']} against "
                f"{candidate['scoring_fingerprint']}), so one frozen gold was not read and scored "
                "the same way for both")
    return None


def assert_eval_set_disjoint(
    client: BlobClient, corpus_version: str, tenant_id: str | None = None
) -> None:
    """Refuse to evaluate against a corpus the eval set overlaps.

    This is the assertion that makes "never train on the eval set" a fact rather
    than an intention. Without it the failure is completely silent: training
    succeeds, the loss curve looks healthy, every metric improves, the gate
    passes, and the numbers describe memorisation.

    The tenant's own frozen set against the tenant's own corpus: both number
    their documents per tenant, so comparing across tenants would report leaks
    that are only two documents sharing an id.
    """
    overlap = (eval_set_source_ids(client, tenant_id)
               & corpus_source_ids(client, corpus_version, tenant_id))
    if overlap:
        raise EvalSetLeakage(
            f"{len(overlap)} document(s) are in BOTH the frozen golden eval set "
            f"({paths.golden_eval_set_dir(tenant_id)}/) and corpus {corpus_version}: "
            f"{sorted(overlap)[:10]}. Every metric measured against this eval "
            "set is a measurement of memorisation, and nothing else in the pipeline would show it "
            "— the loss curve looks healthy and the gate passes. Remove them from the corpus (the "
            "eval set is the thing held constant across versions, so it is the corpus that "
            "changes) and rebuild (arch §8)."
        )
    log.info("eval set is disjoint from corpus %s", corpus_version)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass
class SubsetReport:
    """Metrics for one doc type × one subset."""

    doc_type: str
    subset: str
    documents: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error_records: list[dict[str, Any]] = field(default_factory=list)
    #: The errors totalled by class, by line and for the worst fields (:func:`_error_summary`).
    error_summary: dict[str, Any] = field(default_factory=dict)
    #: ``{metric: {source_id: score}}``: each document's own score, the samples
    #: the gate's paired bootstrap needs (:meth:`EvalReport.per_document_scores`).
    #: Kept out of :meth:`as_dict`; the report carries them once, at the top.
    per_document: dict[str, dict[str, float]] = field(default_factory=dict)
    #: ``{metric: {source_id: items}}``: how many items each document's score is
    #: the mean of - its list fields for ``list_field_recall``, its readings for
    #: the others. The subset's metric is the mean over items, so this is how
    #: much each document counts in it (:meth:`EvalReport.per_document_weights`).
    per_document_items: dict[str, dict[str, int]] = field(default_factory=dict)
    #: ``{source_id: line}``: each REAL document's one line
    #: (``common.lob.single_line``), so a continued run's gate can pair its
    #: per-document scores line by line with the runs before it. Synthetic twins
    #: are left out: they count toward no minimum (decision 2026-10-22). Kept
    #: out of :meth:`as_dict`, like the scores.
    per_document_line: dict[str, str] = field(default_factory=dict)
    #: ``{metric: (sum of score x weight, sum of weight)}`` for
    #: :data:`POOLED_METRICS`, which :meth:`EvalReport.gate_metrics` pools
    #: across doc types. Kept out of :meth:`as_dict`.
    pooled: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: What hallucination_rate is over: the values checked, those no page
    #: prints, and the first of them; None when nothing was checked.
    printed_values: dict[str, Any] | None = None
    #: What run_to_run_agreement is over: the documents and fields compared.
    run_to_run: dict[str, Any] | None = None
    #: The full set's error analysis (:func:`_error_analysis`): per carrier,
    #: per mode, per field, the field confusions. Reported, never gated; kept
    #: out of :meth:`as_dict`, carried once at the top (``error_analysis``).
    analysis: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "subset": self.subset,
            "documents": self.documents,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()},
            **({"printed_values": self.printed_values} if self.printed_values else {}),
            **({"run_to_run": self.run_to_run} if self.run_to_run else {}),
            "error_summary": self.error_summary,
            "error_records": self.error_records,
        }


@dataclass
class EvalReport:
    """One model version's evaluation, per doc type and per subset."""

    model_version: str
    corpus_version: str = ""
    subsets: list[SubsetReport] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    #: Filled in on a later pass: classifier accuracy needs the IMPL-11
    #: classifier, and eval must stay runnable before serving is built.
    classifier_scored: bool = False

    def for_doc_type(self, doc_type: str) -> list[SubsetReport]:
        return [s for s in self.subsets if s.doc_type == doc_type]

    def full_set(self) -> list[SubsetReport]:
        return [s for s in self.subsets if s.subset == "full"]

    def gate_metrics(self) -> dict[str, Any]:
        """The flat metric dict the promotion gate reads.

        Document-weighted across doc types, so a type with three eval documents
        does not carry the same weight as one with thirty.
        """
        totals: dict[str, float] = {}
        weights: dict[str, int] = {}
        for report in self.full_set():
            for name, value in report.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[name] = totals.get(name, 0.0) + float(value) * report.documents
                    weights[name] = weights.get(name, 0) + report.documents

        metrics = {
            name: totals[name] / weights[name]
            for name in totals
            if weights.get(name)
        }

        # Subset accuracies are gating metrics in their own right (arch §15):
        # an aggregate that averages them away is how an image-only regression
        # ships behind a two-point overall move.
        for subset, gate_name in (
            ("image_only", "image_only_accuracy"),
            ("scanned", "scanned_accuracy"),
            ("noisy_ocr", "ocr_arbitration_accuracy"),
            # Reported, never gated (gating.GATING_METRICS leaves it out).
            ("held_out_carrier", "held_out_carrier_match"),
        ):
            scores: list[tuple[float, int]] = [
                (value, s.documents)
                for s in self.subsets
                if s.subset == subset
                and isinstance(value := s.metrics.get("field_normalized_match"), float)
            ]
            total_docs = sum(n for _v, n in scores)
            if total_docs:
                metrics[gate_name] = sum(v * n for v, n in scores) / total_docs

        # Pooled over every doc type's documents (POOLED_METRICS), so the rate
        # is every value no page prints over every value checked.
        for name in POOLED_METRICS:
            parts = [report.pooled[name] for report in self.full_set() if name in report.pooled]
            weight = sum(w for _s, w in parts)
            if weight:
                metrics[name] = sum(s for s, _w in parts) / weight
            else:
                metrics.pop(name, None)

        if not self.classifier_scored:
            # Deliberately absent rather than zero: the gate treats a missing
            # metric as "not passed", which is the right answer for one nobody
            # measured. A zero would read as a measured catastrophe.
            metrics.pop("doc_type_classifier_accuracy", None)
        return metrics

    def per_document_scores(self) -> dict[str, dict[str, float]]:
        """``{gate metric: {source_id: score}}`` over the full set.

        What ``promotion_gate(per_document=...)`` pairs with the promoted
        release's on the shared documents, so non-inferiority and the
        improvement rule are judged by a paired bootstrap instead of two point
        estimates (arch §15.5). The subset accuracies come from their subsets'
        field match, as in :meth:`gate_metrics`.
        """
        scores: dict[str, dict[str, float]] = {}
        for report, name, source_metric in self._per_document_sources():
            scores.setdefault(name, {}).update(report.per_document.get(source_metric, {}))
        return {name: by_source for name, by_source in scores.items() if by_source}

    def per_document_weights(self) -> dict[str, dict[str, float]]:
        """``{gate metric: {source_id: weight}}``, aligned with
        :meth:`per_document_scores`: how much each document counts in the metric
        :meth:`gate_metrics` reports.

        The weighted mean of the per-document scores is that metric, so the
        bootstrap tests the number the gate reports and the floor reads. Without
        it, ``list_field_recall`` - a mean over every table in the set - was
        bootstrapped as a mean over documents: a release whose reported recall
        fell from 0.92 to 0.88 bootstrapped as +0.125 and passed as improved.

        A document's weight is its items (list fields, readings) times its doc
        type's readings over that type's items: each type's metric is a mean
        over its items, and ``gate_metrics`` weights each type's metric by its
        readings (``SubsetReport.documents``). A pooled metric
        (:data:`POOLED_METRICS`) is pooled over the items themselves - the
        values checked, the fields compared - so its weight is its items.
        """
        weights: dict[str, dict[str, float]] = {}
        for report, name, source_metric in self._per_document_sources():
            items = report.per_document_items.get(source_metric, {})
            total = sum(items.values())
            if total and report.documents:
                scale = 1.0 if name in POOLED_METRICS else report.documents / total
                weights.setdefault(name, {}).update(
                    {source: n * scale for source, n in items.items()})
        return {name: by_source for name, by_source in weights.items() if by_source}

    def invention_coverage(self) -> dict[str, int]:
        """What hallucination_rate could and could not check, over the full set
        (:attr:`SubsetReport.printed_values`): the document readings checked on
        page text borrowed from their document's :data:`TEXT_DONOR_MODE`
        reading (image_only), the pages they cite that no reading has text for
        (values there unchecked), and the readings that wrote values with no
        text to check them on at all - unscored, never counted as clean."""
        totals = {"borrowed_text_readings": 0, "cited_pages_without_text": 0, "unscored_readings": 0}
        for report in self.full_set():
            values = report.printed_values or {}
            borrowed, unscored = values.get("borrowed_text") or {}, values.get("unscored") or {}
            totals["borrowed_text_readings"] += int(borrowed.get("readings", 0))
            totals["cited_pages_without_text"] += int(borrowed.get("cited_pages_without_text", 0))
            totals["unscored_readings"] += int(unscored.get("readings", 0))
        return totals

    def per_document_lines(self) -> dict[str, str]:
        """``{source_id: line}`` over the full set: each real document's line
        (:attr:`SubsetReport.per_document_line`)."""
        lines: dict[str, str] = {}
        for report in self.full_set():
            lines.update(report.per_document_line)
        return lines

    def _per_document_sources(self) -> list[tuple[SubsetReport, str, str]]:
        """``(subset report, gate metric, its per-document metric)``: the full
        set's own metrics, and the subset accuracies from their subsets' field
        match, as in :meth:`gate_metrics`."""
        sources = [(report, name, name)
                   for report in self.full_set() for name in report.per_document]
        for subset, gate_name in (
            ("image_only", "image_only_accuracy"),
            ("scanned", "scanned_accuracy"),
            ("noisy_ocr", "ocr_arbitration_accuracy"),
        ):
            sources.extend((report, gate_name, "field_normalized_match")
                           for report in self.subsets if report.subset == subset)
        return sources

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "corpus_version": self.corpus_version,
            "generated_at": self.generated_at,
            "classifier_scored": self.classifier_scored,
            "subsets_scored": sorted({s.subset for s in self.subsets}),
            "by_doc_type": {
                doc_type: [s.as_dict() for s in self.for_doc_type(doc_type)]
                for doc_type in sorted({s.doc_type for s in self.subsets})
            },
            "gate_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in sorted(self.gate_metrics().items())
            },
            "per_document": {
                name: {source: round(score, 6) for source, score in sorted(by_source.items())}
                for name, by_source in sorted(self.per_document_scores().items())
            },
            # Read with per_document: a report without them (written before they
            # were kept) pairs as one document, one weight.
            "per_document_weights": {
                name: {source: round(weight, 6) for source, weight in sorted(by_source.items())}
                for name, by_source in sorted(self.per_document_weights().items())
            },
            # Read with per_document: which line each real document is of, for
            # a continued run's line-by-line gate (decision 2026-10-23).
            "per_document_lines": dict(sorted(self.per_document_lines().items())),
            # Where the errors are, over the full set (reported, never gated):
            # per carrier, per mode, per field, which field's value turned up
            # under which other one, and seen against unseen carriers.
            "error_analysis": {report.doc_type: report.analysis
                               for report in self.full_set() if report.analysis},
            # What the gate's tier will rest on (decision 2026-10-26).
            "tier_evidence": self.tier_evidence(),
        }

    def tier_evidence(self) -> dict[str, Any]:
        """What the gate's tier will rest on (``gating.tier_evidence``):
        served_field_match, its one-sided 95% lower bound bootstrapped over the
        real documents, and the tier and tier_confidence they give IF the gate
        passes and the frozen set is full size.

        The eval report is written before the gate decides, so the tier here is
        named ``tier_if_gate_passes``: the gate decision records the tier.
        """
        from evaluation.gating import SERVED_FIELD_MATCH, tier_evidence

        point = self.gate_metrics().get(SERVED_FIELD_MATCH)
        scores = self.per_document_scores().get(SERVED_FIELD_MATCH, {})
        weights = self.per_document_weights().get(SERVED_FIELD_MATCH, {})
        ids = sorted(scores)
        evidence = tier_evidence(
            # As the gate reads it back from this report: rounded to 4 places.
            {} if point is None else {SERVED_FIELD_MATCH: round(point, 4)},
            ([scores[d] for d in ids], [weights.get(d, 1.0) for d in ids]) if ids else None,
        ).as_dict()
        evidence["tier_if_gate_passes"] = evidence.pop("tier")
        evidence["tier_confidence_if_gate_passes"] = evidence.pop("tier_confidence")
        return evidence


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def subset_of(document: dict[str, Any]) -> list[str]:
    """Which eval subsets a document belongs to. A document can be in several."""
    subsets = ["full"]
    if document.get("modality_mode") == "image_only":
        subsets.append("image_only")
    if document.get("modality_mode") == "noisy_ocr_image":
        subsets.append("noisy_ocr")
    if document.get("is_scanned"):
        subsets.append("scanned")
    if document.get("doc_type") == "policy" and int(document.get("page_count", 1)) > 5:
        subsets.append("long_policy")
    # Reported, not expected in every eval set: documents of the carrier held
    # out of train and val, one per line (Fideon SPEC_09 amendment item 4).
    # Its subset report carries accuracy per line.
    if document.get("held_out_carrier"):
        subsets.append("held_out_carrier")
    return subsets


def _table_f1(
    scored: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> float | None:
    """The claims table F1 over the Loss Runs scored; ``None`` with none."""
    from evaluation.metrics.lossrun_table import lossrun_pairs, table_f1

    pairs = lossrun_pairs(scored)
    return table_f1(pairs).f1 if pairs else None


def _reconciliation_rate(
    scored: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> float | None:
    """Share of VERIFIABLE Loss Runs whose claims reconciled (arch v2.1 §5.5).

    Read from the serving path's own reconciliation report rather than recomputed
    here, so the number the gate reads is the number production produced — a
    second implementation would eventually disagree with the first, and the gate
    would be scoring something serving does not do.

    ``None`` when no document carried one: a subset with no Loss Runs has nothing
    to say about reconciliation, and emitting 0.0 would report a total failure of
    a check that never ran.
    """
    reports = [
        metadata["reconciliation"] for _, _, metadata in scored
        if isinstance(metadata.get("reconciliation"), dict)
    ]
    if not reports:
        return None
    verifiable = [r for r in reports if r.get("status") != "unverifiable"]
    if not verifiable:
        # Every Loss Run printed no totals. Unverifiable is not failure, but it
        # is not evidence either — reported as None so the gate does not read an
        # absence of evidence as a score.
        log.warning(
            "%d Loss Run(s) scored and none printed totals, so row completeness is "
            "unverifiable across the whole subset (arch v2.1 §5.5).", len(reports),
        )
        return None
    return sum(1 for r in verifiable if r.get("status") == "reconciled") / len(verifiable)


def score_subset(
    doc_type: str,
    subset: str,
    scored: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> SubsetReport:
    """Score one doc type × subset from ``(expected, got, metadata)`` triples."""
    from calibration.features import infer_field_type
    from common.lob import merge_line, single_line
    from evaluation.metrics.auto_accept import AutoAcceptTally, score_auto_accept
    from evaluation.metrics.common_model import CommonModelTally, core_for_scoring, without_overflow
    from evaluation.metrics.confusable import aggregate_misattribution, score_misattribution
    from evaluation.metrics.coverage_metrics import (
        SchemaValidityReport,
        expected_calibration_error,
        score_lob,
        score_schema_validity,
    )
    from evaluation.metrics.extraction_faults import (
        _is_empty,
        score_false_nulls,
        score_hallucinations,
        score_page_refs,
    )
    from evaluation.metrics.field_accuracy import score_all_list_fields, score_fields
    from training.vit_gate import classify_error

    report = SubsetReport(doc_type=doc_type, subset=subset, documents=len(scored))
    if not scored:
        return report

    # Common-model lines (SPEC_21): values inside rows count, links, codes and
    # overflow are scored apart, and no id is ever a value (build_report has
    # already put both sides through comparable_view).
    common_model = [_is_common_model_document(doc_type, m) for _e, _g, m in scored]
    # Each common-model document's line, read once: classic auto is personal
    # auto (common.lob.merge_line) - one row of the report, one alias table.
    # Every other document keeps the line its metadata names, as before.
    lines = [merge_line(m.get("lob")) if cm else m.get("lob")
             for (_e, _g, m), cm in zip(scored, common_model, strict=True)]
    model_tally = CommonModelTally()
    # Field accuracy on printed labels training showed vs never showed, where the
    # eval metadata carries the training set's seen labels and the page text.
    by_label: dict[str, list[bool]] = {"seen": [], "unseen": []}

    accuracies, exacts, recalls, f1s, list_precisions = [], [], [], [], []
    # Single-value fields as a retrieval task, counted over every field of the
    # subset: precision = of the values the model wrote, the share that were
    # right; recall = of the values the label holds, the share found. Accuracy
    # alone cannot tell a model that writes little but well from one that
    # writes everything and half of it wrong.
    written = expected_filled = right_and_written = 0
    by_lob: dict[str, list[int]] = {}
    by_field: dict[str, list[int]] = {}
    # Both of these are GATING_METRICS with require_all_measured=True, and
    # neither was ever computed here — so the gate blocked every candidate that
    # ever reached it, permanently, with no override. `score_misattribution` had
    # zero production callers; ECE had none at all. Computing them is the fix:
    # dropping them from the gate would have made it pass by no longer checking
    # the two things this project exists to get right.
    misattributions = []
    # Values delivered without a review flag, and how many were wrong: what
    # the review thresholds promise, measured on documents they never saw.
    auto_accept = AutoAcceptTally()
    confidences: list[float] = []
    correctness: list[bool] = []
    # The same pairs by field type (reported: is a type's confidence honest).
    confidence_by_type: dict[str, tuple[list[float], list[bool]]] = {}
    # Field match without the codes the model assigns rather than reads
    # (coverage_code and the like): per reading, as field match is pooled.
    pooled_without_codes: dict[Any, list[int]] = {}
    # Where the errors are, over the full set only (reported, never gated).
    analysis = _ErrorAnalysis() if subset == "full" else None

    # Field accuracy is pooled per DOCUMENT reading (source x modality), then
    # averaged over documents. Averaged per row, a policy read as sixty windows
    # counted sixty times and a one-page certificate once, so the gate measured
    # long policies and little else. Pooling also means a window with nothing to
    # score — boilerplate pages answered with nothing — adds nothing, rather
    # than a 0.0.
    pooled: dict[Any, list[int]] = {}
    # The readings (keys of ``pooled``) of real documents, for REAL_MODE_METRICS.
    real_readings: set[Any] = set()
    # Each document's list recalls, for its per-document score.
    recalls_by_source: dict[str, list[float]] = {}
    for index, (expected, got, metadata) in enumerate(scored):
        if common_model[index]:
            model_tally.add(expected, got, lines[index])
            accuracy = score_fields(*core_for_scoring(expected, got, lines[index]),
                                    skip_lists=False)
            if metadata.get("seen_labels") is not None and metadata.get("ocr_text"):
                from evaluation.metrics.unseen_labels import common_model_aliases, label_split

                split = label_split(accuracy.results, str(metadata["ocr_text"]),
                                    common_model_aliases(_line_name(lines[index])),
                                    set(metadata["seen_labels"]))
                for bucket, outcomes in split.items():
                    by_label[bucket].extend(outcomes)
        else:
            accuracy = score_fields(expected, got)
        key = (metadata.get("source_id") or f"#{index}", metadata.get("modality_mode"))
        if metadata.get("source_id") and not metadata.get("synthetic"):
            document_line = single_line(metadata.get("lob"))
            if document_line:
                report.per_document_line[str(metadata["source_id"])] = document_line
        if not metadata.get("synthetic"):
            real_readings.add(key)
        tally = pooled.setdefault(key, [0, 0, 0])
        tally[0] += sum(r.correct for r in accuracy.results)
        tally[1] += sum(r.exact for r in accuracy.results)
        tally[2] += accuracy.total
        codes = model_assigned_paths(expected, got)
        read = [r for r in accuracy.results if _field_pattern(r.field_path) not in codes]
        without_codes = pooled_without_codes.setdefault(key, [0, 0])
        without_codes[0] += sum(r.correct for r in read)
        without_codes[1] += len(read)
        if analysis is not None:
            analysis.add(metadata, accuracy.results, _line_name(lines[index]))
        lob_tally = by_lob.setdefault(_line_name(lines[index]), [0, 0])
        lob_tally[0] += sum(r.correct for r in accuracy.results)
        lob_tally[1] += accuracy.total
        for result in accuracy.results:
            has_value, wrote = not _is_empty(result.expected), not _is_empty(result.got)
            written += wrote
            expected_filled += has_value
            right_and_written += bool(result.correct and has_value and wrote)
            # Per field, not per row: coverages[3].limits[0].amount is one field.
            field_tally = by_field.setdefault(re.sub(r"\[\d+\]", "[]", result.field_path), [0, 0])
            field_tally[0] += bool(result.correct)
            field_tally[1] += 1

        auto_accept.add(score_auto_accept(expected, got, common_model=common_model[index]))
        misattributions.append(
            score_misattribution(
                expected, got, doc_type, source_id=metadata.get("source_id", "")
            )
        )

        # ECE pairs each field's stated confidence with whether it was right.
        # Fields carrying no confidence are skipped rather than assumed correct
        # or assumed zero — an unmeasured field is not evidence either way.
        stated = metadata.get("field_confidence") or {}
        for result in accuracy.results:
            score = stated.get(result.field_path)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                confidences.append(float(score))
                correctness.append(bool(result.correct))
                field_type = infer_field_type(result.field_path, result.got, common_model=common_model[index])
                pairs = confidence_by_type.setdefault(field_type, ([], []))
                pairs[0].append(float(score))
                pairs[1].append(bool(result.correct))
                if analysis is not None:
                    analysis.confidence(metadata, field_type, float(score), bool(result.correct))

        for result in accuracy.failures():
            record = {
                "source_id": metadata.get("source_id", ""),
                "field_path": result.field_path,
                "expected": result.expected,
                "got": result.got,
                "error_class": classify_error(result.expected, result.got, all_expected=expected),
                "modality_mode": metadata.get("modality_mode", "ocr_plus_image"),
                "is_scanned": bool(metadata.get("is_scanned")),
                "lob": _line_name(lines[index]),
            }
            if metadata.get("carrier"):
                record["carrier"] = str(metadata["carrier"])
            report.error_records.append(record)
            if analysis is not None:
                analysis.error(metadata, record["error_class"])

        tables = (without_overflow(expected), without_overflow(got)) if common_model[index] else (expected, got)
        for list_report in score_all_list_fields(*tables, common_model=common_model[index]).values():
            recalls.append(list_report.recall)
            if metadata.get("source_id"):
                recalls_by_source.setdefault(metadata["source_id"], []).append(list_report.recall)
            f1s.append(list_report.f1)
            if list_report.got_rows:
                list_precisions.append(list_report.precision)

    accuracies_without_codes = [correct / total for correct, total in pooled_without_codes.values() if total]
    if analysis is not None:
        report.analysis = analysis.as_dict(by_field)

    # Per document: the mean of its readings (one per modality), so that over
    # documents read in the same modes it averages to the subset's own number.
    readings_by_source: dict[str, list[tuple[float, float]]] = {}
    for (source, _mode), (correct, exact, total) in pooled.items():
        if total:
            accuracies.append(correct / total)
            exacts.append(exact / total)
            if not str(source).startswith("#"):
                readings_by_source.setdefault(source, []).append((correct / total, exact / total))
    # Field match per mode on real documents (REAL_MODE_METRICS): each reading
    # counts once, as in field_normalized_match.
    real_by_mode: dict[str, _Pooled] = {name: _Pooled() for name in REAL_MODE_METRICS}
    for (source, mode), (correct, _exact, total) in pooled.items():
        if total and (source, mode) in real_readings:
            for name, wanted in REAL_MODE_METRICS.items():
                if mode == wanted:
                    real_by_mode[name].add(str(source), correct / total, 1.0)

    lob = score_lob([(e, g) for e, g, _m in scored])
    misattribution = aggregate_misattribution(misattributions)
    page_refs = score_page_refs([
        _read_values(expected, got, cm) for (expected, got, _m), cm in zip(scored, common_model, strict=True)
    ])
    report.error_summary = _error_summary(report.error_records)

    # An ACORD document with no recorded form has no selectable schema. Counting
    # it as invalid keeps the run going and does not flatter the result; letting
    # the SchemaError escape would discard every other document's score too.
    validatable, unselectable = [], []
    # A common-model document served with its verdict (golden_eval): serving's
    # own judgement, the audit gate's. The JSON it served has every key filled,
    # so a section no window wrote is there as nulls and checking it again
    # would pass what serving refused.
    served = SchemaValidityReport()
    for (_e, got, meta), cm in zip(scored, common_model, strict=True):
        form = meta.get("acord_form")
        if cm and isinstance(meta.get("schema_valid"), bool):
            served.total += 1
            if meta["schema_valid"]:
                served.valid += 1
            else:
                served.invalid_source_ids.append(meta.get("source_id", ""))
        elif doc_type == "acord" and not form:
            unselectable.append(meta.get("source_id", ""))
        else:
            validatable.append((
                # A common-model document as it was produced (build_report), not
                # the scored copy, which no schema of the client's describes.
                meta.get("source_id", ""), meta.get(_PRODUCED, got), doc_type, form,
                meta.get("lob"), meta.get("sections"),
            ))

    validity = score_schema_validity(validatable)
    validity.valid += served.valid
    validity.total += served.total
    validity.invalid_source_ids.extend(served.invalid_source_ids)
    validity_total = validity.total + len(unselectable)
    validity_rate = validity.valid / validity_total if validity_total else 0.0
    invalid_by_source = Counter(validity.invalid_source_ids) + Counter(unselectable)
    validity_readings = Counter(m.get("source_id", "") for _e, _g, m in scored)

    report.per_document = {
        "field_normalized_match": {
            source: sum(a for a, _x in r) / len(r) for source, r in readings_by_source.items()},
        "field_exact_match": {
            source: sum(x for _a, x in r) / len(r) for source, r in readings_by_source.items()},
        "list_field_recall": {
            source: sum(r) / len(r) for source, r in recalls_by_source.items() if r},
        "schema_validity_rate": {
            source: 1.0 - min(invalid_by_source[source], n) / n
            for source, n in validity_readings.items() if source and n
        },
    }
    # What each score above is the mean of. The metrics below are means over
    # these items, not over documents: list recall over every list field, so a
    # policy with four tables counts four times and one with one table once.
    report.per_document_items = {
        "field_normalized_match": {source: len(r) for source, r in readings_by_source.items()},
        "field_exact_match": {source: len(r) for source, r in readings_by_source.items()},
        "list_field_recall": {source: len(r) for source, r in recalls_by_source.items() if r},
        "schema_validity_rate": {
            source: n for source, n in validity_readings.items() if source and n},
    }
    # Pooled over values and fields, not averaged over readings: a document
    # counts by the values its text could settle, and by the fields its two
    # readings compared (POOLED_METRICS).
    printed = _printed_values(scored, common_model, score_hallucinations)
    agreement = _run_to_run(scored, common_model, lines)
    for name, tally in (("hallucination_rate", printed), ("run_to_run_agreement", agreement),
                        *real_by_mode.items()):
        report.per_document[name] = tally.per_document()
        report.per_document_items[name] = tally.weights()
        if tally.weight:
            report.pooled[name] = (tally.score, tally.weight)
    # Also when nothing was checked but readings went unscored: a reading with
    # no text to check is shown, never read as a clean one (decision 2026-10-26).
    coverage = printed.coverage()
    if printed.weight or coverage:
        report.printed_values = {
            "documents": len(printed.by_document), "checked": int(printed.weight),
            "not_printed": int(printed.score), "instances": printed.instances[:25],
            **coverage,
        }
    if agreement.weight:
        report.run_to_run = {"documents": len(agreement.by_document),
                             "fields_compared": round(agreement.weight, 1)}
    if unselectable:
        log.warning(
            "%d ACORD eval document(s) record no acord_form, so no schema could be selected and "
            "they are counted invalid: %s. That is a defect in the frozen eval set — the form is "
            "what selects the schema (arch §4b).",
            len(unselectable), sorted(unselectable)[:10],
        )

    report.metrics = {
        # None when no row had anything to score — not measured, not zero.
        "field_normalized_match": sum(accuracies) / len(accuracies) if accuracies else None,
        "field_exact_match": sum(exacts) / len(exacts) if exacts else None,
        "list_field_recall": sum(recalls) / len(recalls) if recalls else None,
        # The name the gate (GATING_METRICS) and RunManifest both use. Emitting
        # `list_field_f1` meant the gate never received it: an F1 collapse did
        # not block, and once any promoted manifest carried the real name every
        # later candidate would be blocked forever as "not measured".
        "field_f1_list_fields": sum(f1s) / len(f1s) if f1s else None,
        # Of the table rows the model wrote, the share that are real rows.
        "list_field_precision": (
            sum(list_precisions) / len(list_precisions) if list_precisions else None
        ),
        "field_precision": right_and_written / written if written else None,
        "field_recall": right_and_written / expected_filled if expected_filled else None,
        "field_f1": (
            2 * right_and_written / (written + expected_filled)
            if written + expected_filled else None
        ),
        # Reported, not gated: where accuracy is won and lost.
        "field_accuracy_by_lob": {
            line: round(correct / total, 4) for line, (correct, total) in sorted(by_lob.items())
            if total
        } or None,
        # How many values each line's accuracy is over, so a reader can tell a
        # line's number from noise.
        "field_scored_by_lob": {
            line: total for line, (_correct, total) in sorted(by_lob.items()) if total
        } or None,
        # The fields with at least 3 scorings, lowest accuracy first.
        "weakest_fields": [
            {"field": path, "accuracy": round(correct / total, 4), "scored": total}
            for path, (correct, total) in sorted(
                by_field.items(), key=lambda item: (item[1][0] / item[1][1], -item[1][1])
            )
            if total >= 3
        ][:15] or None,
        "schema_validity_rate": validity_rate,
        # Reported, not gated: do right values cite the pages that print them
        # (every page, as the convention asks)? None where no page was compared.
        "page_ref_exact_rate": page_refs.exact_rate,
        "page_ref_precision": page_refs.precision,
        "page_ref_recall": page_refs.recall,
        # None when no value carried a flag (a flat extraction), or when none
        # was accepted: no error rate among accepted values exists then. The
        # gate reads None as unmeasured (a conditional metric, gating.py).
        "auto_accept_error_rate": auto_accept.rate,
        # Reported, never gated: of the values a flag decided, the share
        # accepted - overall and per field type - and how right the accepted
        # ones were, with the Wilson 95% lower bound (selective accuracy).
        "auto_accept_coverage": auto_accept.coverage,
        "auto_accept_coverage_by_type": auto_accept.coverage_by_type() or None,
        "auto_accept_counts": ({"decided": auto_accept.flags_seen,
                                "accepted": auto_accept.accepted, "wrong": auto_accept.wrong}
                               if auto_accept.flags_seen else None),
        "selective_accuracy": auto_accept.selective_accuracy,
        "selective_accuracy_lower_95": auto_accept.selective_accuracy_lower,
        "selective_accuracy_by_type": auto_accept.selective_by_type() or None,
        # Reported, never gated (the tier reads served_field_match): the
        # same field match without the codes the model assigns rather than
        # reads off the page (coverage_code and every other bare value of an
        # enveloped answer, model_assigned_paths).
        "field_normalized_match_without_codes": (
            sum(accuracies_without_codes) / len(accuracies_without_codes)
            if accuracies_without_codes else None),
        # Absent, not 0.0, when no document in the set carries a line to score.
        "lob_detection_accuracy": lob.overall if lob.scored else None,
        "lob_accuracy_by_value": lob.accuracy_by_value(),
        # A gating metric in its own right (arch §15): returning the certificate
        # holder's name for insured_name is the failure the canonical mapping
        # exists to prevent, and it is invisible in an aggregate field score.
        "confusable_misattribution_rate": misattribution.rate,
        # Deliberately absent — not zero — when no field carried a confidence.
        # Zero is the best possible ECE, so defaulting to it would report
        # perfect calibration for a run where calibration was never measured.
        "ece_confidence": (
            expected_calibration_error(confidences, correctness) if confidences else None
        ),
        # Reported, not gated: the same gap per field type (each its own calibrator).
        "ece_by_field_type": {
            field_type: round(expected_calibration_error(scores, right), 4)
            for field_type, (scores, right) in sorted(confidence_by_type.items()) if scores
        } or None,
        # --- arch v2.1 §15.2: faults an aggregate field score cannot see ------
        #
        # A false null produces NO tokens, so §5 confidence is blind to it —
        # only this metric sees a value that was on the page and came back empty.
        "false_null_rate": score_false_nulls(
            [_read_values(expected, got, cm) for (expected, got, _), cm in zip(scored, common_model, strict=True)],
            source_ids=[m.get("source_id", "") for _, _, m in scored],
        ).rate,
        # Scored against the TEXT OF THE DOCUMENT'S PAGES, not against the
        # golden label: checking against the label alone would call every wrong
        # value a hallucination, including an honest misread of something
        # actually printed — and those have different remedies. Through
        # serving, serving's own printed-values check (_printed_values).
        #
        # Only readings with text: against an image-only row's empty text
        # every value the model read off the image would count as invented. So
        # an image-only reading is checked on its document's page text
        # borrowed from its ocr_plus_image reading (decision 2026-10-26), and
        # one with none to borrow is unscored, shown in printed_values.
        # Every value the model wrote counts, as serving counts it; the extra
        # fields are also reported alone where scored here: free text by
        # design, inside the one rate they can hide the schema's own fields.
        "hallucination_rate": printed.rate,
        "additional_fields_hallucination_rate": _hallucination_rate(
            scored, common_model, score_hallucinations, overflow=True),
        # Two readings of one document in the main mode, the second sent in
        # another order: the share of fields they agree on (run_to_run_agreement).
        "run_to_run_agreement": agreement.rate,
        # Field match on real documents' image_only readings as served (what
        # users receive) and on their ocr_plus_image readings (REAL_MODE_METRICS).
        # Absent where the subset holds no real reading of that mode.
        **{name: tally.rate for name, tally in real_by_mode.items()},
        # Emitted only where reconciliation ran — Loss Runs. gate_metrics()
        # weights by the documents that produced each metric, so a metric that
        # only one doc type can produce still reaches the gate, weighted by that
        # type's documents rather than diluted by the others.
        "lossrun_totals_reconciliation_rate": _reconciliation_rate(scored),
        # Loss Runs only: claims matched by claim number with total incurred
        # within a cent (evaluation.metrics.lossrun_table).
        "table_f1": _table_f1(scored),
        # The weakest common-model line's field match: gated (a conditional
        # metric), so one line cannot hide behind the others' volume.
        "worst_line_field_match": _worst_line(by_lob, lines, common_model),
        **(model_tally.metrics() if any(common_model) else {}),
        "seen_label_field_match": (
            round(sum(by_label["seen"]) / len(by_label["seen"]), 4) if by_label["seen"] else None),
        "unseen_label_field_match": (
            round(sum(by_label["unseen"]) / len(by_label["unseen"]), 4) if by_label["unseen"] else None),
    }
    report.metrics = {k: v for k, v in report.metrics.items() if v is not None}
    return report


def _is_common_model_document(doc_type: str, metadata: dict[str, Any]) -> bool:
    from common.schemas import SchemaError, is_common_model

    if doc_type != "policy":
        return False
    try:
        return is_common_model("policy", None, metadata.get("lob"))
    except SchemaError:
        return False


#: Where :func:`build_report` keeps a common-model document as it was produced,
#: on its own copy of the metadata: the document schema validity is asked of.
_PRODUCED = "_produced_output"


def _error_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The errors totalled: by class (left empty, invented, misread, value of
    another field, other wrong value), by line, and for the fields that fail
    most. Reported, not gated - where the next fix should go."""
    from collections import Counter

    if not records:
        return {}
    by_line: dict[str, Counter] = {}
    by_field: dict[str, Counter] = {}
    for record in records:
        kind = record.get("error_class") or "unknown"
        by_line.setdefault(record.get("lob") or "unknown", Counter())[kind] += 1
        by_field.setdefault(re.sub(r"\[[^\]]*\]", "[]", record.get("field_path", "")), Counter())[kind] += 1
    return {
        "by_class": dict(Counter(r.get("error_class") or "unknown" for r in records).most_common()),
        "by_line": {line: dict(counts.most_common()) for line, counts in sorted(by_line.items())},
        "top_fields": [
            {"field": name, "errors": sum(counts.values()), "by_class": dict(counts.most_common())}
            for name, counts in sorted(by_field.items(), key=lambda item: -sum(item[1].values()))[:15]
        ],
    }


def _field_pattern(path: str) -> str:
    """A field's path with its rows unnumbered: ``coverages[3].limit`` is
    ``coverages[].limit``, one field of every row."""
    return re.sub(r"\[[^\]]*\]", "[]", path)


def model_assigned_paths(expected: Any, got: Any) -> set[str]:
    """The fields (:func:`_field_pattern`) the model assigns rather than reads:
    every bare value - a scalar, or a list of them, outside any envelope - of an
    answer or label written in envelopes. A common-model policy's
    ``coverage_code`` and the other codes and types; its ids and links are not
    scored as values at all. Empty for a flat document (no envelopes), where no
    value is told from another."""
    from evaluation.metrics.extraction_faults import _has_envelope

    if not (_has_envelope(expected) or _has_envelope(got)):
        return set()
    return _bare_paths(expected, "") | _bare_paths(got, "")


def _bare_paths(node: Any, path: str) -> set[str]:
    from common.canonical import is_field_value

    if is_field_value(node):
        return set()
    if isinstance(node, dict):
        out: set[str] = set()
        for key, value in node.items():
            out |= _bare_paths(value, f"{path}.{key}" if path else str(key))
        return out
    if isinstance(node, list):
        if node and not any(isinstance(item, (dict, list)) for item in node):
            return {_field_pattern(path)}
        out = set()
        for item in node:
            out |= _bare_paths(item, f"{path}[]")
        return out
    return {_field_pattern(path)} if node not in (None, "") and path else set()


#: The field confusions kept per report: the most frequent first.
CONFUSIONS_REPORTED = 50


@dataclass
class _ErrorAnalysis:
    """The full set's errors, tallied for :attr:`SubsetReport.analysis`:
    field match per carrier, per reading mode and - where the readings say -
    per carrier seen in training or not; the error classes of each; the
    calibration error per mode and field type; and which label field's value
    the answer wrote under another field (``field_confusion``)."""

    carriers: dict[str, dict[str, Any]] = field(default_factory=dict)
    modes: dict[str, dict[str, Any]] = field(default_factory=dict)
    seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    confusions: Counter = field(default_factory=Counter)
    examples: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    mode_confidence: dict[str, dict[str, tuple[list[float], list[bool]]]] = field(default_factory=dict)

    @staticmethod
    def _keys(metadata: dict[str, Any]) -> tuple[str, str, str | None]:
        seen = metadata.get("carrier_seen_in_train")
        return (str(metadata.get("carrier") or "unknown"), str(metadata.get("modality_mode") or "unknown"),
                ("seen" if seen else "unseen") if isinstance(seen, bool) else None)

    def _rows(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        carrier, mode, seen = self._keys(metadata)
        tables = [(self.carriers, carrier), (self.modes, mode)] + ([(self.seen, seen)] if seen else [])
        return [table.setdefault(key, {"documents": set(), "readings": 0, "scored": 0, "correct": 0,
                                       "errors_by_class": Counter()}) for table, key in tables]

    def add(self, metadata: dict[str, Any], results: Sequence[Any], line: str) -> None:
        for row in self._rows(metadata):
            row["documents"].add(str(metadata.get("source_id") or ""))
            row["readings"] += 1
            row["scored"] += len(results)
            row["correct"] += sum(bool(r.correct) for r in results)
        self._confusions(results, metadata, line)

    def error(self, metadata: dict[str, Any], error_class: str) -> None:
        for row in self._rows(metadata):
            row["errors_by_class"][error_class or "unknown"] += 1

    def confidence(self, metadata: dict[str, Any], field_type: str, score: float, correct: bool) -> None:
        by_type = self.mode_confidence.setdefault(str(metadata.get("modality_mode") or "unknown"), {})
        pairs = by_type.setdefault(field_type, ([], []))
        pairs[0].append(score)
        pairs[1].append(correct)

    def _confusions(self, results: Sequence[Any], metadata: dict[str, Any], line: str) -> None:
        """A label value of field A written under another field B: indexed by
        its normalised form, so each wrong value is looked up, not compared with
        every field; confirmed as field match compares (``values_agree``)."""
        from evaluation.metrics.extraction_faults import _is_empty
        from evaluation.metrics.field_accuracy import values_agree

        gold: dict[str, list[Any]] = {}
        for result in results:
            for key in () if _is_empty(result.expected) else _confusion_keys(result.expected, result.field_path):
                gold.setdefault(key, []).append(result)
        for result in results:
            if result.correct or _is_empty(result.got):
                continue
            found_under = _field_pattern(result.field_path)
            candidates = {id(o): o for key in _confusion_keys(result.got, result.field_path) for o in gold.get(key, [])}
            for other in sorted(candidates.values(), key=lambda o: o.field_path):
                gold_field = _field_pattern(other.field_path)
                if gold_field == found_under or not values_agree(other.expected, result.got, other.field_path):
                    continue
                pair = (gold_field, found_under)
                self.confusions[pair] += 1
                kept = self.examples.setdefault(pair, [])
                if len(kept) < 3:
                    kept.append({"source_id": metadata.get("source_id", ""), "value": str(result.got)[:80],
                                 "mode": metadata.get("modality_mode"), "lob": line})
                break

    def as_dict(self, by_field: dict[str, list[int]]) -> dict[str, Any]:
        from evaluation.metrics.coverage_metrics import expected_calibration_error

        def table(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return {key: {"documents": len(row["documents"]), "readings": row["readings"],
                          "fields_scored": row["scored"], "fields_wrong": row["scored"] - row["correct"],
                          "field_match": round(row["correct"] / row["scored"], 4) if row["scored"] else None,
                          "errors_by_class": dict(row["errors_by_class"].most_common())}
                    for key, row in sorted(rows.items())}

        def ece(pairs: tuple[list[float], list[bool]]) -> float | None:
            return round(expected_calibration_error(*pairs), 4) if pairs[0] else None

        modes = table(self.modes)
        for mode, by_type in self.mode_confidence.items():
            if mode in modes:
                every = ([s for scores, _ in by_type.values() for s in scores],
                         [ok for _, right in by_type.values() for ok in right])
                modes[mode]["ece"] = ece(every)
                modes[mode]["ece_by_field_type"] = {t: ece(pairs) for t, pairs in sorted(by_type.items())}
        out: dict[str, Any] = {
            "by_carrier": table(self.carriers),
            "by_mode": modes,
            "by_field": {name: {"scored": total, "correct": correct, "accuracy": round(correct / total, 4)}
                         for name, (correct, total) in sorted(by_field.items()) if total},
            "field_confusion": [
                {"gold_field": gold, "found_under": under, "count": count, "examples": self.examples[(gold, under)]}
                for (gold, under), count in self.confusions.most_common(CONFUSIONS_REPORTED)
            ],
        }
        if self.seen:
            out["by_carrier_seen_in_train"] = table(self.seen)
        return out


def _confusion_keys(value: Any, path: str) -> set[str]:
    """A value's forms to find it under another field by: normalised for its
    own field, and as its bare letters and digits (two fields of different
    kinds normalise one value differently). None too short or too plain to say
    anything (a flag, a ``1``)."""
    from common.normalize import normalize_value

    if isinstance(value, (bool, dict, list)) or value is None:
        return set()
    keys = {re.sub(r"[^0-9a-z]", "", str(value).casefold())}
    # A value no normaliser reads is looked up bare.
    with contextlib.suppress(Exception):
        keys.add(str(normalize_value(value, field_path=path) or "").strip())
    return {key for key in keys if len(key) >= 4}


def field_deltas(report: dict[str, Any], promoted: dict[str, Any] | None) -> dict[str, Any] | None:
    """Each field's accuracy against the promoted release's eval report, per doc
    type: ``{doc_type: {"against", "same_yardstick", "fields": [...]}}``, the
    fields both scored, the largest drop first. Read from each report's
    ``error_analysis`` ``by_field``; a promoted report written before it was
    kept offers its ``weakest_fields`` (the fields it reported at all). ``None``
    with no promoted report, or nothing to compare. Reported, never gated: on
    another yardstick (``same_yardstick`` false) a move may be the set or the
    scoring, not the model."""
    if not promoted:
        return None
    mine = report.get("error_analysis") or {}
    theirs_all = promoted.get("error_analysis") or {}
    same = yardstick_difference(promoted.get("eval_set"), report.get("eval_set"), None) is None if (
        promoted.get("eval_set") and report.get("eval_set")) else None
    out: dict[str, Any] = {}
    for doc_type, analysis in sorted(mine.items()):
        current = analysis.get("by_field") or {}
        theirs = (theirs_all.get(doc_type) or {}).get("by_field") or _weakest_fields(promoted, doc_type)
        fields = []
        for name in sorted(set(current) & set(theirs)):
            now, then = current[name], theirs[name]
            if now.get("accuracy") is None or then.get("accuracy") is None:
                continue
            fields.append({"field": name, "accuracy": now["accuracy"], "promoted": then["accuracy"],
                           "delta": round(now["accuracy"] - then["accuracy"], 4),
                           "scored": now.get("scored"), "promoted_scored": then.get("scored")})
        if fields:
            out[doc_type] = {"against": promoted.get("model_version"), "same_yardstick": same,
                             "fields": sorted(fields, key=lambda f: (f["delta"], f["field"]))}
    return out or None


def _weakest_fields(report: dict[str, Any], doc_type: str) -> dict[str, dict[str, Any]]:
    for subset in (report.get("by_doc_type") or {}).get(doc_type) or []:
        if subset.get("subset") == "full":
            return {row["field"]: {"accuracy": row.get("accuracy"), "scored": row.get("scored")}
                    for row in subset.get("weakest_fields") or [] if row.get("field")}
    return {}


def _hallucination_rate(scored: list[tuple[Any, Any, dict[str, Any]]], common_model: list[bool],
                        score: Any, *, overflow: bool = False) -> float | None:
    """``score_hallucinations`` over the rows that were sent OCR text
    (:func:`_reading_text`), or borrow it (:func:`_invention_text`); ``None``
    when nothing checkable was emitted on a row with text - not measured,
    rather than a perfect 0."""
    sent = []
    for (expected, got, metadata), cm in zip(scored, common_model, strict=True):
        if _reading_text(metadata) is None and _BORROWED_PAGES not in metadata:
            continue
        read = _read_values(expected, got, cm)
        found = _invention_text(metadata, read[1])
        if found is not None:
            sent.append((*read, found[0]))
    report = score(sent, overflow=overflow) if sent else None
    return report.rate if report is not None and report.opportunities else None


#: Where :func:`build_report` keeps, on a reading's own copy of its metadata,
#: every page of its document the eval holds, prepared once
#: (``common.grounding.tolerant_pages``): a policy's windows each carry their
#: own pages, and a value is looked for on all of the document's.
_DOCUMENT_PAGES = "_document_pages"

#: Where :func:`build_report` keeps the second answer (:data:`REPEAT_ANSWER`)
#: as the first is scored: narrowed to the schema, ids made comparable.
_REPEAT_SCORED = "_repeat_scored"


def _reading_text(metadata: dict[str, Any]) -> Any:
    """The text a reading's values are looked for in: every page of its
    document (:data:`_DOCUMENT_PAGES`), else the reading's own pages
    (``ocr_pages``), else its text whole; ``None`` for a reading sent no text."""
    if not metadata.get("ocr_text"):
        return None
    return metadata.get(_DOCUMENT_PAGES) or metadata.get("ocr_pages") or str(metadata["ocr_text"])


#: Where :func:`build_report` keeps, on an image-only reading's own copy of its
#: metadata, the page text it borrows: every page of its document that the
#: document's :data:`TEXT_DONOR_MODE` readings carry, prepared once.
_BORROWED_PAGES = "_borrowed_pages"


def _borrows_text(metadata: dict[str, Any]) -> bool:
    """Whether a reading is checked on borrowed text: an image-only reading of
    a document (it has a source id), sent no text, and not checked by serving
    already (a reading served by the golden eval borrows there, golden_eval)."""
    return (metadata.get("modality_mode") == IMAGE_ONLY_MODE and bool(metadata.get("source_id"))
            and not metadata.get("ocr_text") and PRINTED_CHECK not in metadata)


def cited_pages(answer: Any) -> set[int]:
    """Every page a value of ``answer`` cites (its envelopes' ``page_ref``)."""
    from evaluation.metrics.extraction_faults import _written

    return {int(page) for _path, _forms, cited in _written(answer, bare_printed=False)
            for page in cited if not isinstance(page, bool) and str(page).isdigit()}


def page_has_text(page: Any) -> bool:
    """Whether a page's text (a string, or prepared, ``common.grounding.TolerantPage``)
    has anything to look a value up in; a page that is absent has none."""
    from common.grounding import TolerantPage

    if page is None:
        return False
    return (page if isinstance(page, TolerantPage) else TolerantPage(str(page))).has_text


def _invention_text(metadata: dict[str, Any], got: Any) -> tuple[Any, list[int] | None] | None:
    """The text a reading's written values are looked up in, and - for borrowed
    text - the pages they cite that it has no text for; ``None`` with no text
    at all, own or borrowed.

    Its own text first (:func:`_reading_text`; the pages then ``None``). An
    image-only reading has none and borrows its document's
    (:data:`_BORROWED_PAGES`): there, a page it cites that no reading carries
    text for is added blank, so a value found on no page with text while citing
    it is unchecked (``common.grounding``: on an unread page), never invented
    because the page was missing - and those pages are returned, to be counted."""
    own = _reading_text(metadata)
    if own is not None:
        return own, None
    borrowed = metadata.get(_BORROWED_PAGES)
    if borrowed is None:
        return None
    unread = sorted(page for page in cited_pages(got) if not page_has_text(borrowed.get(page)))
    missing = {page: "" for page in unread if page not in borrowed}
    return ({**borrowed, **missing} if missing else borrowed), unread


def _wrote_values(got: Any) -> bool:
    """Whether an answer wrote any value a page could print (as
    ``score_hallucinations`` reads it): a reading with none has nothing to check."""
    from evaluation.metrics.extraction_faults import _has_envelope, _is_empty, _written

    return any(not _is_empty(form) for _path, forms, _cited in _written(got, bare_printed=not _has_envelope(got))
               for form in forms)


@dataclass
class _Pooled:
    """A metric pooled over readings, per document: ``[sum of score x weight,
    sum of weight]``, keyed by source id (``#n`` for a reading without one)."""

    by_document: dict[str, list[float]] = field(default_factory=dict)
    instances: list[dict[str, Any]] = field(default_factory=list)

    def add(self, key: str, weighted_score: float, weight: float) -> None:
        tally = self.by_document.setdefault(key, [0.0, 0.0])
        tally[0] += weighted_score
        tally[1] += weight

    @property
    def score(self) -> float:
        return sum(s for s, _w in self.by_document.values())

    @property
    def weight(self) -> float:
        return sum(w for _s, w in self.by_document.values())

    @property
    def rate(self) -> float | None:
        return self.score / self.weight if self.weight else None

    def per_document(self) -> dict[str, float]:
        return {k: s / w for k, (s, w) in self.by_document.items() if w and not k.startswith("#")}

    def weights(self) -> dict[str, float]:
        return {k: w for k, (_s, w) in self.by_document.items() if w and not k.startswith("#")}


@dataclass
class _Printed(_Pooled):
    """hallucination_rate's counts (:class:`_Pooled`), and what they could not
    see, by document reading - ``(source id, mode)``, one per document and mode."""

    #: Readings checked on text borrowed from their document's donor reading.
    borrowed: set[tuple[str, str]] = field(default_factory=set)
    #: ``(source id, page)``: pages those readings cite that have no text.
    unread_pages: set[tuple[str, int]] = field(default_factory=set)
    #: Readings that wrote values with no text, own or borrowed, to check them on.
    unscored: set[tuple[str, str]] = field(default_factory=set)

    def coverage(self) -> dict[str, Any]:
        """For :attr:`SubsetReport.printed_values`: empty when every reading
        was checked on text it was sent."""
        out: dict[str, Any] = {}
        if self.borrowed:
            out["borrowed_text"] = {
                "from_mode": TEXT_DONOR_MODE,
                "readings": len(self.borrowed),
                "documents": len({source for source, _mode in self.borrowed}),
                # Values citing these are unchecked unless another page prints them.
                "cited_pages_without_text": len(self.unread_pages),
                "pages_without_text": [f"{source} p{page}" for source, page in sorted(self.unread_pages)][:25],
            }
        if self.unscored:
            out["unscored"] = {
                "readings": len(self.unscored),
                "by_mode": dict(sorted(Counter(mode for _source, mode in self.unscored).items())),
                "documents": sorted({source for source, _mode in self.unscored})[:25],
            }
        return out


def _printed_values(scored: Sequence[tuple[Any, Any, dict[str, Any]]], common_model: list[bool],
                    score: Any) -> _Printed:
    """hallucination_rate's counts, per document: of the values its readings
    wrote that the text could settle (the weight), those no page of the
    document prints (decision 2026-10-24).

    A reading served through serving (golden_eval) carries serving's own
    record (:data:`PRINTED_CHECK`): ``len(not_printed)`` of ``checked``,
    recorded before strict mode removed any, so it measures the model. A record
    of None - no page text, as image only reads - leaves the reading out. Any
    other reading that had text (a validation row) is scored here by the same
    matcher (``score_hallucinations``, OCR-tolerant over every page). A reading
    with nothing checked counts nowhere.

    An image-only reading is sent no text: it is checked on the page text it
    borrows from its document's ocr_plus_image reading (:func:`_invention_text`;
    through serving, golden_eval borrows it and marks the record
    :data:`BORROWED_TEXT`), as production checks its fields on MinerU's text
    (decision 2026-10-26). A reading that wrote values with no text to check
    them on - none sent, none to borrow, or a record of None - is counted
    unscored (:meth:`_Printed.coverage`), never as a reading with nothing invented.
    """
    tally = _Printed()
    for index, ((expected, got, metadata), cm) in enumerate(zip(scored, common_model, strict=True)):
        source = str(metadata.get("source_id") or f"#{index}")
        reading = (source, str(metadata.get("modality_mode") or ""))
        if PRINTED_CHECK in metadata:
            record = metadata[PRINTED_CHECK]
            if not isinstance(record, dict):
                if _wrote_values(got):
                    tally.unscored.add(reading)
                continue
            if metadata.get(PRINTED_TEXT) == BORROWED_TEXT:
                tally.borrowed.add(reading)
                tally.unread_pages.update((source, int(page)) for page in metadata.get(UNREAD_PAGES) or [])
            entries = [e for e in record.get("not_printed") or [] if isinstance(e, dict)]
            invented, checked = len(entries), int(record.get("checked") or 0)
            found = [{"field": e.get("path"), "emitted": str(e.get("raw"))[:80],
                      "pages": list(e.get("pages") or [])} for e in entries]
        else:
            if _reading_text(metadata) is None and _BORROWED_PAGES not in metadata:
                if _wrote_values(got):
                    tally.unscored.add(reading)
                continue
            read = _read_values(expected, got, cm)
            found_text = _invention_text(metadata, read[1])
            if found_text is None:
                continue
            text, unread = found_text
            if unread is not None:
                tally.borrowed.add(reading)
                tally.unread_pages.update((source, page) for page in unread)
            fault = score([(*read, text)])
            invented, checked, found = fault.hits, fault.opportunities, fault.instances
        if not checked:
            continue
        tally.add(source, invented, checked)
        tally.instances.extend({"source_id": metadata.get("source_id", ""), **entry} for entry in found)
    return tally


def _run_to_run(scored: Sequence[tuple[Any, Any, dict[str, Any]]], common_model: list[bool],
                lines: Sequence[Any]) -> _Pooled:
    """run_to_run_agreement's counts, per document: each reading that carries a
    second answer (:data:`REPEAT_ANSWER`) against it, weighted by the fields
    compared (:func:`run_to_run_agreement`)."""
    tally = _Pooled()
    for index, ((_expected, got, metadata), cm, line) in enumerate(
            zip(scored, common_model, lines, strict=True)):
        if _REPEAT_SCORED not in metadata:
            continue
        agreed = run_to_run_agreement(got, metadata[_REPEAT_SCORED], common_model=cm, line=line)
        if agreed is None:
            continue
        share, compared = agreed
        tally.add(str(metadata.get("source_id") or f"#{index}"), share * compared, compared)
    return tally


def run_to_run_agreement(first: Any, second: Any, *, common_model: bool = False,
                         line: Any = None) -> tuple[float, float] | None:
    """How alike two answers to one document are: ``(share of fields equal,
    fields compared)``, or ``None`` when neither wrote a field.

    Field match's own scorer (``score_fields``, as :func:`score_subset` calls
    it), each answer held as the reference in turn - the second against the
    first, then the first against the second - and the two shares averaged:
    name and table matching are not symmetric. A field null in both is not
    compared: a served answer holds every key of its schema, most of them null,
    and two readings agreeing a field is absent would bury the fields that were
    read. Fields compared is the mean of the two counts.
    """
    from evaluation.metrics.common_model import core_for_scoring
    from evaluation.metrics.extraction_faults import _is_empty
    from evaluation.metrics.field_accuracy import score_fields

    shares: list[float] = []
    counts: list[int] = []
    for reference, other in ((first or {}, second or {}), (second or {}, first or {})):
        accuracy = (score_fields(*core_for_scoring(reference, other, line), skip_lists=False)
                    if common_model else score_fields(reference, other))
        compared = [r for r in accuracy.results if not (_is_empty(r.expected) and _is_empty(r.got))]
        if compared:
            shares.append(sum(r.correct for r in compared) / len(compared))
            counts.append(len(compared))
    if not counts:
        return None
    return sum(shares) / len(shares), sum(counts) / len(counts)


def _read_values(expected: Any, got: Any, common_model: bool) -> tuple[Any, Any]:
    """What was READ: a common-model document without its codes and links,
    which no page prints, so neither can be a false null or a hallucination.

    Its rows are paired by identity first (``aligned_for_scoring``), as field
    match pairs them: these metrics compare by path, and a coverage table in
    another order put a liability row's limits against a collision row with
    none. Paired on the whole documents, then stripped: the codes and links
    that pair the rows are themselves bare values.

    A row the model wrote that pairs with no label row then takes the next
    label row nothing paired, in table order (``fill_unpaired``): its values
    were written, so a gold row it misread is wrong values, which field match
    prices, never false nulls. Only label rows left after that - rows the model
    did not write at all - count their values as nulls.
    """
    from common.canonical import without_bare_values

    if not common_model:
        return expected, got
    from evaluation.metrics.field_accuracy import aligned_for_scoring

    expected, got = aligned_for_scoring(expected, got, fill_unpaired=True)
    return without_bare_values(expected), without_bare_values(got)


def _line_name(line: Any) -> str:
    """A line as a report row names it: several lines joined, none "unknown"."""
    return str(", ".join(line) if isinstance(line, list) else (line or "unknown"))


#: A line's field match is gated only once it has this many values scored: below
#: it the number is noise, and one odd document would block a release.
WORST_LINE_MIN_VALUES = 200


def _worst_line(by_lob: dict[str, list[int]], lines: Sequence[Any], common_model: list[bool]) -> float | None:
    """The lowest field match among the common-model lines in ``by_lob``, each
    named as ``by_lob`` names it (:func:`_line_name`)."""
    names = {_line_name(line) for line, cm in zip(lines, common_model, strict=True) if cm}
    rates = [correct / total for line, (correct, total) in by_lob.items()
             if line in names and total >= WORST_LINE_MIN_VALUES]
    return round(min(rates), 4) if rates else None


def line_document_scores(
    report: dict[str, Any], metric: str = "field_normalized_match",
) -> dict[str, dict[str, tuple[float, float]]] | None:
    """``{line: {source_id: (score, weight)}}`` over a report's full set: each
    real document's ``metric`` and its weight in it (``per_document_weights``;
    1 when the report holds none), grouped by the document's line
    (``per_document_lines``). What a continued run's gate pairs line by line
    (decisions 2026-10-23 and 2026-10-26): ``served_field_match`` where both
    reports hold it, ``field_normalized_match`` otherwise. ``None`` for a report
    written before it recorded documents' lines; empty when it holds no
    per-document ``metric``."""
    lines = report.get("per_document_lines")
    if lines is None:
        return None
    scores = (report.get("per_document") or {}).get(metric) or {}
    weights = (report.get("per_document_weights") or {}).get(metric) or {}
    grouped: dict[str, dict[str, tuple[float, float]]] = {}
    for source, line in lines.items():
        if source in scores:
            grouped.setdefault(str(line), {})[source] = (float(scores[source]),
                                                         float(weights.get(source) or 1.0))
    return grouped


def expected_subsets() -> set[str]:
    """Which eval subsets a run should contain documents for: every one. The
    adapter reads policies, so ``long_policy`` applies, and §6 requires every
    corpus to cover all three modality regimes."""
    return set(EVAL_SUBSETS)


def build_report(
    model_version: str,
    documents: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    *,
    corpus_version: str = "",
    classifier_scored: bool = False,
) -> EvalReport:
    """Score every doc type × subset from one flat list of results.

    Args:
        documents: ``(expected, got, metadata)`` triples. ``metadata`` carries
            ``doc_type``, ``modality_mode``, ``is_scanned``, ``page_count`` and
            ``source_id`` — the fields :func:`subset_of` reads — and, where
            there is one, serving's printed-values record (:data:`PRINTED_CHECK`)
            and a second answer to compare with (:data:`REPEAT_ANSWER`).
    """
    report = EvalReport(
        model_version=model_version,
        corpus_version=corpus_version,
        classifier_scored=classifier_scored,
    )

    from common.lob import merge_line

    pages = _document_pages(documents)
    buckets: dict[tuple[str, str], list[Any]] = {}
    for expected, got, metadata in documents:
        doc_type = metadata.get("doc_type", "unknown")
        common_model = _is_common_model_document(doc_type, metadata)
        if common_model:
            # Validity is asked of what the model or serving produced. The scored
            # copy below is narrowed to the model view and stripped of its ids, so
            # against the client's schema even a perfect answer was invalid.
            # Kept on a copy: the caller's metadata is not ours to change.
            metadata = {**metadata, _PRODUCED: got}
        selectors = (doc_type, metadata.get("acord_form"), metadata.get("lob"))
        line = merge_line(metadata.get("lob")) if common_model else None
        expected = _scored_view(expected, selectors, line, common_model)
        got = _scored_view(got, selectors, line, common_model)
        if REPEAT_ANSWER in metadata:
            # The second answer is compared with the first as the first is
            # scored: the same fields, the same comparable ids.
            metadata = {**metadata, _REPEAT_SCORED: _scored_view(
                metadata[REPEAT_ANSWER] or {}, selectors, line, common_model)}
        document = (metadata.get("source_id"), metadata.get("modality_mode"))
        if document in pages:
            metadata = {**metadata, _DOCUMENT_PAGES: pages[document]}
        elif _borrows_text(metadata) and (metadata.get("source_id"), TEXT_DONOR_MODE) in pages:
            # An image-only reading, sent no text: its values are looked up on
            # its document's pages as the ocr_plus_image readings carry them -
            # the same pinned MinerU text (decision 2026-10-26).
            metadata = {**metadata, _BORROWED_PAGES: pages[(metadata.get("source_id"), TEXT_DONOR_MODE)]}
        for subset in subset_of(metadata):
            buckets.setdefault((doc_type, subset), []).append((expected, got, metadata))

    for (doc_type, subset), rows in sorted(buckets.items()):
        report.subsets.append(score_subset(doc_type, subset, rows))

    missing = expected_subsets() - {s.subset for s in report.subsets}
    if missing:
        # Reported rather than raised: an eval set with no scanned documents is
        # a coverage gap in the eval set, and pretending it scored 100% would be
        # far worse than saying it was not measured.
        log.warning(
            "eval set contains no %s document(s), so those subset metrics are absent. The "
            "promotion gate blocks on an absent metric unless it is conditional (scanned "
            "documents: some lines arrive digital only) and the baseline did not measure it "
            "either (evaluation.gating.CONDITIONAL_METRICS).",
            sorted(missing),
        )
    return report


def _scored_view(doc: Any, selectors: tuple[Any, Any, Any], line: Any, common_model: bool) -> Any:
    """An answer or a label as :func:`build_report` scores it."""
    from common.canonical import schema_label, without_system_fields

    # Not scored: the system supplies them, the model is never asked
    # (common.canonical.SYSTEM_SUPPLIED_FIELDS). Stripped from both sides, so a
    # gold label or frozen eval set that still carries them counts nothing.
    doc = without_system_fields(doc)
    # A gold label written for another line's schema is scored in this line's
    # (configs/label_mappings.yaml), and only on what that schema can hold - as
    # its training target was built (common.canonical.schema_label). Both
    # sides: a key outside the schema is not a field this report measures. The
    # constrained model cannot write one; an unconstrained output that does
    # fails schema_validity_rate, which is where it counts.
    doc = schema_label(doc, *selectors)
    if common_model:
        # Ids are the writer's numbering, never right or wrong: links are
        # compared by what they name (common.structural_ids).
        from common.structural_ids import comparable_view

        doc = comparable_view(doc, line)
    return doc


def _document_pages(documents: Sequence[tuple[Any, Any, dict[str, Any]]]) -> dict[tuple[Any, Any], Any]:
    """Every page's text the readings of each document carry, by ``(source id,
    mode)``, prepared once for the OCR-tolerant search: a policy's windows each
    carry their own pages, and serving looks for a value on all of them. Only
    readings with a source id: two without one are not one document."""
    from common.grounding import tolerant_pages

    texts: dict[tuple[Any, Any], dict[int, str]] = {}
    for _expected, _got, metadata in documents:
        if metadata.get("source_id") and metadata.get("ocr_text") and metadata.get("ocr_pages"):
            texts.setdefault((metadata["source_id"], metadata.get("modality_mode")), {}).update(
                {int(number): text for number, text in metadata["ocr_pages"].items()})
    return {document: tolerant_pages(pages) for document, pages in texts.items()}


def write_report(report: EvalReport, client: BlobClient) -> list[str]:
    """Write per-doc-type reports plus the top-level summary."""
    written = []
    for doc_type in sorted({s.doc_type for s in report.subsets}):
        key = paths.eval_report(report.model_version, doc_type)
        client.write_json(key, {
            "model_version": report.model_version,
            "doc_type": doc_type,
            "subsets": [s.as_dict() for s in report.for_doc_type(doc_type)],
        })
        written.append(key)

    summary_key = paths.eval_report(report.model_version)
    client.write_json(summary_key, report.as_dict())
    written.append(summary_key)
    return written


def render(report: EvalReport) -> str:
    lines = [f"eval {report.model_version} (corpus {report.corpus_version or 'unrecorded'})"]
    for subset in report.subsets:
        accuracy = subset.metrics.get("field_normalized_match")
        lines.append(
            f"  {subset.doc_type:8} {subset.subset:12} n={subset.documents:3} "
            + (f"field={accuracy:.3f}" if isinstance(accuracy, float) else "field=n/a")
        )
    if not report.classifier_scored:
        lines.append("  doc_type_classifier_accuracy: not scored this pass (IMPL-11 classifier)")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Evaluate a model version on the frozen eval set")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus-version", dest="corpus_version", required=True)
    parser.add_argument("--doc-types", nargs="+", default=list(ACTIVE_DOC_TYPES),
                        choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--tenant", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('evaluation.run_eval', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()

    # Runs before anything is evaluated: a leaked eval set makes every number
    # below meaningless, so it is not worth computing them first.
    assert_eval_set_disjoint(client, args.corpus_version, args.tenant)

    # Through serving.pipeline, never a bespoke inference path (IMPL-08): the
    # report measures what production serves.
    from evaluation.golden_eval import dumps, evaluate_version
    from inference_core.model_runner import load_model

    body = evaluate_version(
        client, load_model(args.model, client),
        version=args.model, corpus_version=args.corpus_version, tenant_id=args.tenant,
    )

    print(dumps(body.get("gate_metrics") or {}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
