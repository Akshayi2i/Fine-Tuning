"""The corpus manifest — what a corpus version contains, and what pins it.

Training and evaluation read this to know exactly what they are working with.
Everything recorded here is either a **reproducibility pin** (a change forces a
corpus rebuild and a new training cycle) or a **coverage measurement** (a warning
that names what the corpus is thin on, before a model is trained on it).

The coverage checks warn loudly but do **not** block the build. The remedy for
under-coverage is collecting more documents, which is a data-acquisition decision
— failing the build would not produce those documents, it would just stop work.
The exception is a zero confusable count, which is a corpus *defect* worth
shouting about: the model would learn the field mapping but never the boundary.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from common import aliases as alias_registry
from common.constants import LOB_COVERAGE_TARGET, MODALITY_MIX, MODALITY_MODES
from common.lob import compute_coverage
from common.prompts import PROMPT_TEMPLATE_VERSION
from common.schemas import schema_version
from data_pipeline.dataset_builder.policy_windows import window_plan_id

log = logging.getLogger(__name__)

#: The review rule every gold in a corpus passed (``export_golden_labels.review_problem``).
#: "1": a gold a model drafted needed a review record (decision 2026-10-19).
#: "2": every gold needs one - a person's reviewer_id and a review_date not in
#: the future - and a twin carries its seed's (decision 2026-10-22). Training
#: and packaging refuse a corpus built under an older rule
#: (``orchestration.pipeline_dag._assert_corpus_pins``): its golds were never
#: checked against this one.
REVIEW_RULE_VERSION = "2"

#: A surface label seen on fewer documents than this is under-represented. The
#: model will be weak on it, and per-alias eval will have too little support to
#: say so reliably.
ALIAS_COVERAGE_FLOOR = 3


@dataclass
class CoverageReport:
    """What the corpus is thin on, named rather than merely counted."""

    warnings: list[str] = field(default_factory=list)
    lob_shares: dict[str, float] = field(default_factory=dict)
    alias_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    confusable_example_count: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.warnings

    def log_all(self) -> None:
        for warning in self.warnings:
            log.warning("%s", warning)


def compute_lob_coverage(
    golden_labels: list[dict[str, Any]], lobs: list[Any] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Per-LoB-value share of the labels that carry ``line_of_business``, against
    the ≥20% target (arch §0b).

    Only labels that CARRY the field are counted here (ACORD, Loss Run). A policy's
    line is metadata — a canonical schema name such as ``flood``, ``gl`` or
    ``cyber`` — not a value of the LOB enum the model outputs, so it is counted by
    :func:`count_policy_lines` instead. Validating it against the enum rejected most
    real policy lines and failed the whole corpus build.
    """
    carried = [
        label.get("line_of_business")
        for i, label in enumerate(golden_labels)
        if not (lobs is not None and lobs[i]) and "line_of_business" in label
    ]
    coverage = compute_coverage(carried)
    warnings: list[str] = []
    if message := coverage.warning():
        warnings.append(message)
    return coverage.shares, warnings


def count_policy_lines(lobs: list[Any]) -> dict[str, int]:
    """Documents per policy line (canonical schema names), from metadata.

    A package policy counts once for each of its lines.
    """
    from collections import Counter

    from common.lob import lob_lines

    counts: Counter[str] = Counter()
    for lob in lobs:
        counts.update(lob_lines(lob))
    return dict(sorted(counts.items()))

def compute_alias_coverage(
    provenance_by_source: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Documents per ``(canonical field, observed surface label)`` (arch §0c).

    **Alias coverage matters more than document count.** 95 documents saying
    *Named Insured* and 5 saying *Applicant* produce a model that is shaky on
    *Applicant* however large the corpus is.
    """
    counts: dict[str, dict[str, int]] = defaultdict(Counter)
    for provenance in provenance_by_source.values():
        for canonical_field, surface_label in provenance.items():
            counts[canonical_field][surface_label] += 1

    warnings: list[str] = []
    for canonical_field, labels in sorted(counts.items()):
        thin = {label: n for label, n in labels.items() if n < ALIAS_COVERAGE_FLOOR}
        # `len(labels) > 1` suppressed the warning for a field seen under exactly
        # one surface label — the thinnest coverage possible, and the case where
        # per-alias eval has nothing to compare against at all.
        if thin:
            named = ", ".join(f"{label!r} ({n} doc{'s' if n != 1 else ''})"
                              for label, n in sorted(thin.items(), key=lambda kv: kv[1]))
            warnings.append(
                f"alias coverage thin for {canonical_field}: {named}. The model will be weak on "
                f"these phrasings, and per-alias eval will have too little support to prove it "
                f"either way. Collect more documents using them (arch §0c)."
            )
    return {k: dict(v) for k, v in counts.items()}, warnings


def count_confusable_examples(
    golden_labels_by_source: dict[str, dict[str, Any]],
    doc_type: str,
) -> tuple[int, list[str]]:
    """Documents where a canonical field and one of its confusables co-occur.

    These teach the **boundary**, not just the mapping. Without them the model
    learns "name-ish label → insured_name" and will happily return the
    certificate holder with high confidence (arch §0c).
    """
    count = 0
    for label in golden_labels_by_source.values():
        for canonical_field in list(label):
            if label.get(canonical_field) is None:
                continue
            for confusable_label in alias_registry.confusables_for(doc_type, canonical_field):
                sibling = alias_registry.canonical_for(doc_type, confusable_label)
                if sibling and sibling != canonical_field and label.get(sibling) is not None:
                    count += 1
                    break
            else:
                continue
            break

    warnings: list[str] = []
    if count == 0 and golden_labels_by_source:
        warnings.append(
            f"NO confusable co-occurrence documents for {doc_type}. This is a corpus defect, not "
            "an acceptable state: the model will learn which label maps to which field but never "
            "the boundary between confusable parties, and misattribution fails silently and "
            "confidently (arch §0c)."
        )
    return count, warnings


def _answer_values(row: dict[str, Any]) -> int | None:
    """How many values a row's answer states (each envelope, compact or full),
    or None when its answer is not JSON."""
    import json

    from common.canonical import expand_answer, is_field_value

    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return None
    try:
        answer = expand_answer(json.loads(messages[-1].get("content") or ""))
    except (TypeError, ValueError):
        return None

    def count(node: Any) -> int:
        if is_field_value(node):
            return int(node.get("raw") is not None or node.get("parsed") is not None)
        if isinstance(node, dict):
            return sum(count(value) for value in node.values())
        if isinstance(node, list):
            return sum(count(value) for value in node)
        return 0

    return count(answer)


def target_stats(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per split, line, section group and reading mode: ``rows``, ``empty_rows``
    (answers stating no value - each one teaches "nothing here") and
    ``mean_values`` per row (review item 33). Absence is taught by the empty
    rows, and by every value a window leaves out; this is how much of it a
    corpus teaches, per mode - an image_only row is taught the same target as
    its OCR twin.

    The line is the row's ``lob`` (a package's lines joined with "+"), else
    its doc type; the group its ``sections``, else its ``task``.
    """
    from common.lob import lob_lines

    totals: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for split, rows in rows_by_split.items():
        for row in rows:
            lines = sorted(lob_lines(row.get("lob"))) if row.get("lob") else []
            line = "+".join(lines) or str(row.get("doc_type") or "unknown")
            group = str(row.get("sections") or row.get("task") or "extract")
            values = _answer_values(row) or 0
            entry = totals[(split, line, group, str(row.get("modality_mode") or "unknown"))]
            entry[0] += 1
            entry[1] += int(values == 0)
            entry[2] += values
    out: dict[str, Any] = {}
    for (split, line, group, mode), (count, empty, values) in sorted(totals.items()):
        out.setdefault(split, {}).setdefault(line, {}).setdefault(group, {})[mode] = {
            "rows": count, "empty_rows": empty, "mean_values": round(values / count, 2) if count else 0.0}
    return out


def _target_counts(counts: dict[str, int] | None) -> dict[str, int]:
    """Every :data:`build_jsonl.TARGET_COUNTS` key, 0 where the build counted
    none, then any other key it gave, in name order."""
    from data_pipeline.dataset_builder.build_jsonl import TARGET_COUNTS

    given = {str(key): int(value) for key, value in (counts or {}).items()}
    return dict(sorted({**dict.fromkeys(TARGET_COUNTS, 0), **given}.items()))


def _schema_pins(doc_types: list[str]) -> dict[str, str]:
    """``{"policy": "1.0.0", "acord:25": "1.0.0", ...}`` for the types present."""
    from common.schemas import schema_selectors

    pins: dict[str, str] = {}
    for doc_type, acord_form, lob in schema_selectors():
        if doc_type not in doc_types:
            continue
        qualifier = acord_form or lob
        key = f"{doc_type}:{qualifier}" if qualifier else doc_type
        # Passed in the right slot: handing a line of business to `acord_form`
        # made schema_key ignore it and pin the GENERIC policy version under the
        # per-LOB key, so a schema change to one line would not force a rebuild.
        pins[key] = schema_version(doc_type, acord_form, lob)
    return dict(sorted(pins.items()))


def build_manifest(
    *,
    corpus_version: str,
    tenant_id: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    golden_labels_by_source: dict[str, dict[str, Any]],
    provenance_by_source: dict[str, dict[str, str]],
    split_assignment: dict[str, Any],
    ocr_environment: dict[str, Any],
    doc_types: list[str],
    seed: int = 42,
    git_commit: str = "unknown",
    edge_case_counts: dict[str, int] | None = None,
    lob_by_source: dict[str, Any] | None = None,
    modality_mix_target: dict[str, float] | None = None,
    target_counts: dict[str, int] | None = None,
) -> tuple[dict[str, Any], CoverageReport]:
    """Assemble the manifest and run every coverage check.

    ``target_counts``: what building the targets did
    (``build_jsonl.BuildResult.target_counts``, keys ``build_jsonl.TARGET_COUNTS``):
    rule-added values kept, re-paged and listed unprinted; values with no page
    placed, and the windows and documents that cost or saved; answers read as
    halves. Recorded as given, every key present (0 when none).
    """
    report = CoverageReport()

    # ---- example counts ----
    counts: dict[str, Any] = {}
    for split, rows in sorted(rows_by_split.items()):
        by_type: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            by_type[row["doc_type"]][row["modality_mode"]] += 1
        counts[split] = {dt: dict(modes) for dt, modes in sorted(by_type.items())}

    total_rows = sum(len(rows) for rows in rows_by_split.values())
    # The mix is a TRAINING property (arch v2.1 §6.1): each train document draws
    # one mode per epoch. Val and test rows carry all three modes for every
    # document by design, so counting them pulled the reported mix toward a
    # third each and hid a sampler that drifted off its 50/30/20 target.
    modality_totals = Counter(row["modality_mode"] for row in rows_by_split.get("train", []))
    train_rows = sum(modality_totals.values())

    # ---- coverage ----
    sources = list(golden_labels_by_source)
    lob_shares, lob_warnings = compute_lob_coverage(
        [golden_labels_by_source[s] for s in sources],
        [(lob_by_source or {}).get(s) for s in sources],
    )
    alias_counts, alias_warnings = compute_alias_coverage(provenance_by_source)
    report.warnings.extend(lob_warnings + alias_warnings)
    report.lob_shares = lob_shares
    policy_lines = count_policy_lines([(lob_by_source or {}).get(s) for s in sources])
    report.alias_counts = alias_counts

    confusable_total = 0
    for doc_type in doc_types:
        subset = {
            sid: label for sid, label in golden_labels_by_source.items()
            if any(r["source_id"] == sid and r["doc_type"] == doc_type
                   for rows in rows_by_split.values() for r in rows)
        }
        count, warnings = count_confusable_examples(subset, doc_type)
        confusable_total += count
        report.warnings.extend(warnings)
    report.confusable_example_count = confusable_total

    manifest = {
        "corpus_version": corpus_version,
        "tenant_id": tenant_id,
        "built_at": datetime.now(UTC).isoformat(),
        "doc_types": sorted(doc_types),

        # ---- reproducibility pins: a change to any forces a rebuild + retrain ----
        "mineru_version": ocr_environment.get("mineru_version"),
        "ocr_device": ocr_environment.get("ocr_device"),
        # Every schema the corpus could have used, keyed as `acord:125` where a
        # type has per-form schemas. Pinning only ACORD 25 meant a corpus holding
        # 125s and 140s recorded no version for them — and this pin is what
        # forces a rebuild when a schema changes, so those forms could change
        # underneath a corpus that claimed to be reproducible.
        "schema_versions": _schema_pins(doc_types),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        # How a policy is cut into windows: an adapter is served only the plan
        # its corpus was built with (policy_windows.assert_window_plan).
        "window_plan_version": window_plan_id(),
        # Which review rule every gold passed (REVIEW_RULE_VERSION).
        "review_rule_version": REVIEW_RULE_VERSION,
        "builder_git_commit": git_commit,
        "seed": seed,

        # ---- composition ----
        "example_counts": counts,
        "total_rows": total_rows,
        "modality_mix": {
            mode: round(modality_totals.get(mode, 0) / train_rows, 4) if train_rows else 0.0
            for mode in MODALITY_MODES
        },
        "modality_mix_basis": "train rows",
        # The mix the modes were drawn against: the global default or the
        # training config's own (training.data_mix.assert_corpus_mix reads it).
        "modality_mix_target": dict(modality_mix_target or MODALITY_MIX),
        "source_ids_by_split": {
            split: sorted({row["source_id"] for row in rows})
            for split, rows in sorted(rows_by_split.items())
        },
        "split_assignment": split_assignment,

        # ---- coverage measurements ----
        "lob_coverage": lob_shares,
        # Documents per policy line (canonical schema names, from metadata).
        "policy_line_counts": policy_lines,
        "lob_coverage_target": LOB_COVERAGE_TARGET,
        "alias_coverage": alias_counts,
        "confusable_example_count": confusable_total,
        "edge_case_counts": edge_case_counts or {},
        # What each line, group and mode teaches: rows, empty answers, values
        # per answer (review item 33) - and what building the targets did.
        "target_stats": target_stats(rows_by_split),
        "target_counts": _target_counts(target_counts),

        # ---- de-identification: none, Option A (arch §8b, IMPL-05 §1) ----
        # Recorded honestly rather than omitted. Text-only de-identification
        # would corrupt the training signal, so nothing has been de-identified
        # and the limitation travels with the corpus.
        "deidentified": False,
        "image_redaction": "unresolved",

        "coverage_warnings": report.warnings,
    }

    report.log_all()
    return manifest, report
