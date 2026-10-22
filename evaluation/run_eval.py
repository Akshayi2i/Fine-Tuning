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
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES

log = logging.getLogger(__name__)

#: Eval subsets scored separately from the full set (IMPL-08 §3).
EVAL_SUBSETS: tuple[str, ...] = ("image_only", "scanned", "noisy_ocr", "long_policy")


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
    SELECTED on — its train and val splits (and their scope views).

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
    client: BlobClient, tenant_id: str | None, doc_types: Iterable[str]
) -> dict[str, Any]:
    """Which frozen set, of which tenant, an eval report was scored on, and how
    many of its documents per type (``doc_types``: one entry per document scored).

    The gate pairs two reports' per-document scores only when both were scored
    on the same set of the same tenant: source ids are numbered per tenant, and
    a set deleted and frozen again (corrected golds) reuses them, so
    ``policy_0001`` in two reports is not always one document against one gold.
    The counts are what the scope was scored on, which decides whether it was
    scored on too few documents to be called production.
    """
    from collections import Counter

    from evaluation.freeze_eval_set import frozen_manifest

    manifest = frozen_manifest(client, tenant_id=tenant_id)
    return {
        "tenant_id": paths._tenant(tenant_id),
        # A set frozen again is a new freeze: its frozen_at differs, whatever
        # corpus it came from.
        "frozen_at": manifest.get("frozen_at"),
        "frozen_from_corpus": manifest.get("frozen_from_corpus"),
        "documents_by_doc_type": dict(sorted(Counter(doc_types).items())),
    }


#: What :func:`eval_set_record` writes that names the set itself. Two reports
#: are scored on one yardstick when these agree.
EVAL_SET_IDENTITY: tuple[str, ...] = ("tenant_id", "frozen_at", "frozen_from_corpus")


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "subset": self.subset,
            "documents": self.documents,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()},
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
        readings (``SubsetReport.documents``).
        """
        weights: dict[str, dict[str, float]] = {}
        for report, name, source_metric in self._per_document_sources():
            items = report.per_document_items.get(source_metric, {})
            total = sum(items.values())
            if total and report.documents:
                scale = report.documents / total
                weights.setdefault(name, {}).update(
                    {source: n * scale for source, n in items.items()})
        return {name: by_source for name, by_source in weights.items() if by_source}

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
        }


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
    from common.lob import merge_line
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

    # Field accuracy is pooled per DOCUMENT reading (source x modality), then
    # averaged over documents. Averaged per row, a policy read as sixty windows
    # counted sixty times and a one-page certificate once, so the gate measured
    # long policies and little else. Pooling also means a window with nothing to
    # score — boilerplate pages answered with nothing — adds nothing, rather
    # than a 0.0.
    pooled: dict[Any, list[int]] = {}
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
        tally = pooled.setdefault(key, [0, 0, 0])
        tally[0] += sum(r.correct for r in accuracy.results)
        tally[1] += sum(r.exact for r in accuracy.results)
        tally[2] += accuracy.total
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

        for result in accuracy.failures():
            report.error_records.append({
                "source_id": metadata.get("source_id", ""),
                "field_path": result.field_path,
                "expected": result.expected,
                "got": result.got,
                "error_class": classify_error(result.expected, result.got, all_expected=expected),
                "modality_mode": metadata.get("modality_mode", "ocr_plus_image"),
                "is_scanned": bool(metadata.get("is_scanned")),
                "lob": _line_name(lines[index]),
            })

        tables = (without_overflow(expected), without_overflow(got)) if common_model[index] else (expected, got)
        for list_report in score_all_list_fields(*tables, common_model=common_model[index]).values():
            recalls.append(list_report.recall)
            if metadata.get("source_id"):
                recalls_by_source.setdefault(metadata["source_id"], []).append(list_report.recall)
            f1s.append(list_report.f1)
            if list_report.got_rows:
                list_precisions.append(list_report.precision)

    # Per document: the mean of its readings (one per modality), so that over
    # documents read in the same modes it averages to the subset's own number.
    readings_by_source: dict[str, list[tuple[float, float]]] = {}
    for (source, _mode), (correct, exact, total) in pooled.items():
        if total:
            accuracies.append(correct / total)
            exacts.append(exact / total)
            if not str(source).startswith("#"):
                readings_by_source.setdefault(source, []).append((correct / total, exact / total))

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
        # None when no value carried a flag (a flat extraction): not measured.
        "auto_accept_error_rate": auto_accept.rate,
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
        # --- arch v2.1 §15.2: faults an aggregate field score cannot see ------
        #
        # A false null produces NO tokens, so §5 confidence is blind to it —
        # only this metric sees a value that was on the page and came back empty.
        "false_null_rate": score_false_nulls(
            [_read_values(expected, got, cm) for (expected, got, _), cm in zip(scored, common_model, strict=True)],
            source_ids=[m.get("source_id", "") for _, _, m in scored],
        ).rate,
        # Scored against the TEXT OF THE PAGES THAT WERE SENT, not against the
        # golden label: checking against the label alone would call every wrong
        # value a hallucination, including an honest misread of something
        # actually printed — and those have different remedies.
        #
        # Only rows that were SENT text: against an image-only row's empty text
        # every value the model read off the image would count as invented.
        # The extra fields apart: free text by design, they hid the rate of the
        # schema's own fields.
        "hallucination_rate": _hallucination_rate(scored, common_model, score_hallucinations),
        "additional_fields_hallucination_rate": _hallucination_rate(
            scored, common_model, score_hallucinations, overflow=True),
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


def _hallucination_rate(scored: list[tuple[Any, Any, dict[str, Any]]], common_model: list[bool],
                        score: Any, *, overflow: bool = False) -> float | None:
    """``score_hallucinations`` over the rows that were sent OCR text, each
    page's own text where the row kept it (``ocr_pages``) so a value is looked
    for on the pages it cites; ``None`` when nothing checkable was emitted on a
    row that was sent text - not measured, rather than a perfect 0."""
    sent = [
        (*_read_values(expected, got, cm), metadata.get("ocr_pages") or str(metadata["ocr_text"]))
        for (expected, got, metadata), cm in zip(scored, common_model, strict=True)
        if metadata.get("ocr_text")
    ]
    report = score(sent, overflow=overflow) if sent else None
    return report.rate if report is not None and report.opportunities else None


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


def expected_subsets(scope: Any = None) -> set[str]:
    """Which eval subsets this run should contain documents for.

    ``long_policy`` only exists where policies do, so warning about its absence
    on a Loss Run scope reports a gap that cannot be filled. The modality subsets
    are NOT scope-dependent: §6 requires every corpus to cover all three regimes,
    so their absence is a real defect whatever the scope covers.
    """
    if scope is None:
        return set(EVAL_SUBSETS)
    covered = set(EVAL_SUBSETS)
    if "policy" not in getattr(scope, "doc_types", ()):
        covered.discard("long_policy")
    return covered


def build_report(
    model_version: str,
    documents: Sequence[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    *,
    corpus_version: str = "",
    classifier_scored: bool = False,
    scope: Any = None,
) -> EvalReport:
    """Score every doc type × subset from one flat list of results.

    Args:
        documents: ``(expected, got, metadata)`` triples. ``metadata`` carries
            ``doc_type``, ``modality_mode``, ``is_scanned``, ``page_count`` and
            ``source_id`` — the fields :func:`subset_of` reads.
    """
    report = EvalReport(
        model_version=model_version,
        corpus_version=corpus_version,
        classifier_scored=classifier_scored,
    )

    from common.canonical import without_system_fields

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
        # Not scored: the system supplies them, the model is never asked
        # (common.canonical.SYSTEM_SUPPLIED_FIELDS). Stripped from both sides, so a
        # gold label or frozen eval set that still carries them counts nothing.
        expected, got = without_system_fields(expected), without_system_fields(got)
        # A gold label written for another line's schema is scored in this
        # line's (configs/label_mappings.yaml), and only on what that schema can
        # hold - as its training target was built (common.canonical.schema_label).
        from common.canonical import schema_label

        # Both sides: a key outside the schema is not a field this report
        # measures. The constrained model cannot write one; an unconstrained
        # output that does fails schema_validity_rate, which is where it counts.
        selectors = (doc_type, metadata.get("acord_form"), metadata.get("lob"))
        expected = schema_label(expected, *selectors)
        got = schema_label(got, *selectors)
        if common_model:
            # Ids are the writer's numbering, never right or wrong: links are
            # compared by what they name (common.structural_ids).
            from common.lob import merge_line
            from common.structural_ids import comparable_view

            line = merge_line(metadata.get("lob"))
            expected = comparable_view(expected, line)
            got = comparable_view(got, line)
        for subset in subset_of(metadata):
            buckets.setdefault((doc_type, subset), []).append((expected, got, metadata))

    for (doc_type, subset), rows in sorted(buckets.items()):
        report.subsets.append(score_subset(doc_type, subset, rows))

    missing = expected_subsets(scope) - {s.subset for s in report.subsets}
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
    parser.add_argument("--scope", default=None, help="configs/scopes.yaml; defaults to unified")
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
    from common.scopes import default_scope, get_scope
    from evaluation.golden_eval import dumps, evaluate_version
    from inference_core.model_runner import load_model

    scope = get_scope(args.scope) if args.scope else default_scope()
    body = evaluate_version(
        client, load_model(args.model, client),
        version=args.model, corpus_version=args.corpus_version, scope=scope,
        tenant_id=args.tenant,
    )
    print(dumps(body.get("gate_metrics") or {}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
