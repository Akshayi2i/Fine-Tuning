"""Freeze a corpus's test split into the golden eval set, once per tenant (arch v2.1 §8).

The promotion gate scores the frozen set at ``golden-eval-set/{tenant}/`` — the
same documents for every model version, so two versions' numbers are comparable.
The corpus's own test split cannot be that: it is re-drawn on every rebuild.
Nothing populated the frozen set, and nothing read the test split, so the gate
had no documents and the test documents were held out for nothing.

Each tenant freezes its own set, from its own corpus, and is gated on it. One
shared set would be the first tenant's documents gating every tenant, and since
source ids are numbered per tenant, another tenant's ``policy_0001`` would be
taken as frozen and dropped from its corpus.

This joins the two. The first real corpus build's test split is copied, per
document, into the layout :mod:`evaluation.golden_eval` reads — the golden label,
its metadata, every page image and OCR page — and a ``manifest.json`` records
where it came from. After that:

* every corpus build excludes the frozen documents **and their families**
  (``orchestration.pipeline_dag.stage_dataset_build``), so a renewal of an eval
  document cannot train the model on its own answer key;
* new documents split into train and val only: the test role is the frozen set.

Freezing twice is refused. A set that changes between versions stops being the
yardstick the gate compares them on; replacing it is a deliberate act (delete
the prefix, then freeze again), after which old and new scores are not
comparable.

A set frozen before sets were per tenant sits at the root, ``golden-eval-set/``.
It is refused, never ignored (:func:`refuse_unscoped_set`): ignored, its tenant
would read as unfrozen, draw a new test split and freeze a different set.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.lob import merge_line

log = logging.getLogger(__name__)


class FreezeError(RuntimeError):
    """Raised when the eval set cannot be frozen."""


#: The smallest frozen set per document type worth making permanent — the
#: production target of arch v2.1 §15.4 (≥ 150 per type). At 150 documents a rate
#: near 80% is known to about ±3 points (one standard error: sqrt(0.8·0.2/150));
#: below it the gate cannot tell a regression of a few points from noise, and the
#: set cannot grow once frozen.
MIN_FROZEN_DOCS_PER_TYPE = 150


def is_scanned(ocr_meta: dict[str, Any]) -> bool:
    """Whether a document is a scan: MinerU's own classification when recorded,
    else — for documents OCR'd before it was — whether any page failed OCR."""
    if "is_scanned" in ocr_meta:
        return bool(ocr_meta["is_scanned"])
    return bool(ocr_meta.get("failed_pages"))


def manifest_key(*, tenant_id: str | None = None) -> str:
    return f"{paths.golden_eval_set_dir(tenant_id)}/manifest.json"


def refuse_unscoped_set(client: BlobClient) -> None:
    """Refuse to go on while a set frozen before sets were per tenant is at the root.

    Nothing reads ``golden-eval-set/manifest.json`` any more. Left there, the
    tenant it was frozen from reads as unfrozen: its next corpus build draws a
    test split, and a freeze from that build replaces the yardstick its earlier
    versions were gated on. Moving the files is the operator's call, because only
    the operator can confirm which tenant the set belongs to.
    """
    legacy = paths.legacy_golden_eval_manifest()
    if not client.exists(legacy):
        return
    tenant = str(client.read_json(legacy).get("tenant_id") or "").strip()
    target = (
        f"golden-eval-set/{tenant}/ (its manifest says it was frozen from tenant {tenant!r})"
        if tenant else "golden-eval-set/{tenant}/, for the tenant it was frozen from"
    )
    raise FreezeError(
        f"a golden eval set frozen before eval sets were kept per tenant is still at the root "
        f"({legacy}). Each tenant now reads only its own set under golden-eval-set/{{tenant}}/, "
        "so this one is read by nobody, and its tenant would look unfrozen and could freeze a "
        "different set. Move the root set — manifest.json and each document directory beside "
        f"it — under {target}, then run again."
    )


def is_frozen(client: BlobClient, *, tenant_id: str | None = None) -> bool:
    """Whether this tenant's golden eval set is frozen: its manifest exists.

    The manifest is written last, so it is the commit point. Keying on "any
    golden.json exists" made an interrupted freeze (40 of 180 documents copied)
    permanent — re-freezing was refused and the gate scored 40 documents.
    """
    refuse_unscoped_set(client)
    return client.exists(manifest_key(tenant_id=tenant_id))


def partial_freeze(client: BlobClient, *, tenant_id: str | None = None) -> dict[str, str]:
    """Documents left by an interrupted freeze: ``{source_id: frozen_from_corpus}``."""
    from evaluation.run_eval import eval_set_source_ids

    if is_frozen(client, tenant_id=tenant_id):
        return {}
    root = paths.golden_eval_set_dir(tenant_id)
    found = {}
    for source_id in eval_set_source_ids(client, tenant_id=tenant_id):
        key = f"{root}/{source_id}/metadata.json"
        meta = client.read_json(key) if client.exists(key) else {}
        found[source_id] = str(meta.get("frozen_from_corpus") or "unknown")
    return found


def frozen_manifest(client: BlobClient, *, tenant_id: str | None = None) -> dict[str, Any]:
    key = manifest_key(tenant_id=tenant_id)
    return client.read_json(key) if client.exists(key) else {}


def freeze_eval_set(
    client: BlobClient,
    corpus_version: str,
    *,
    tenant_id: str | None = None,
    git_commit: str = "unknown",
    allow_small: bool = False,
) -> dict[str, Any]:
    """Copy ``corpus_version``'s test split into ``golden-eval-set/{tenant}/``. Returns the manifest.

    The tenant's own corpus, into the tenant's own set: freezing one tenant never
    reads or changes another's.

    Refused when any document type would freeze with fewer than
    :data:`MIN_FROZEN_DOCS_PER_TYPE` documents, unless ``allow_small``: the frozen
    set is the size of the build it came from, permanently, so freezing a pilot
    build fixes a pilot-sized yardstick for every later version.

    Refused, before anything is copied, when any gold has no review record
    (``export_golden_labels.review_problem``, decision 2026-10-22): a gold
    nobody reviewed never scores, and a set holding one would be refused by the
    gate. The record is carried into each frozen document's metadata, where the
    gate checks it.
    """
    from data_pipeline.dataset_builder.split_groups import line_of
    from data_pipeline.labeling.export_golden_labels import load_golden_label, review_problem, review_record

    root = paths.golden_eval_set_dir(tenant_id)
    partial = partial_freeze(client, tenant_id=tenant_id)
    if partial and set(partial.values()) != {corpus_version}:
        raise FreezeError(
            f"an interrupted freeze left {len(partial)} document(s) in {root}/ "
            f"from corpus {sorted(set(partial.values()))}, not {corpus_version}. Delete that prefix "
            "and freeze again, or re-run the freeze from the corpus it started from."
        )
    if partial:
        log.warning(
            "resuming an interrupted freeze from corpus %s: %d document(s) already copied are "
            "copied again", corpus_version, len(partial),
        )
    if is_frozen(client, tenant_id=tenant_id):
        existing = frozen_manifest(client, tenant_id=tenant_id)
        raise FreezeError(
            f"the golden eval set at {root}/ is already frozen (from corpus "
            f"{existing.get('frozen_from_corpus', 'unknown')}, "
            f"{len(existing.get('source_ids', []))} documents). Re-freezing would change the "
            "yardstick every version is compared on. To replace it on purpose, delete "
            f"{root}/ first — and treat older scores as not comparable."
        )

    test_key = paths.corpus_eval_split(corpus_version, "test", tenant_id)
    if not client.exists(test_key):
        raise FreezeError(f"corpus {corpus_version} has no test split at {test_key}")
    documents: dict[str, str] = {}
    for line in client.read_text(test_key).splitlines():
        if line.strip():
            row = json.loads(line)
            documents.setdefault(row["source_id"], row["doc_type"])
    if not documents:
        raise FreezeError(f"the test split of corpus {corpus_version} holds no documents")
    corpus_manifest_key = paths.corpus_manifest(corpus_version, tenant_id)
    corpus_manifest = (
        client.read_json(corpus_manifest_key) if client.exists(corpus_manifest_key) else {}
    )
    # Every type the corpus holds, not only those that drew test documents: a
    # type with NONE in test would otherwise pass the guard and be missing from
    # the frozen set for good, its quality never gated.
    per_type: dict[str, int] = {t: 0 for t in corpus_manifest.get("doc_types") or ()}
    for doc_type in documents.values():
        per_type[doc_type] = per_type.get(doc_type, 0) + 1
    small = {t: n for t, n in sorted(per_type.items()) if n < MIN_FROZEN_DOCS_PER_TYPE}
    if small and not allow_small:
        raise FreezeError(
            f"the test split would freeze only {small} document(s) for these types, under the "
            f"{MIN_FROZEN_DOCS_PER_TYPE} a permanent eval set needs. The set cannot grow once "
            "frozen, so freeze from a build holding most of the labeled documents. To freeze a "
            "smaller set anyway (a pilot), pass --allow-small."
        )

    loaded = {source_id: load_golden_label(source_id, doc_type, client, tenant_id)
              for source_id, doc_type in sorted(documents.items())}
    unreviewed = [(source_id, problem) for source_id, (label, metadata) in loaded.items()
                  if (problem := review_problem(label, metadata))]
    if unreviewed:
        raise FreezeError(
            f"{len(unreviewed)} of the {len(documents)} test document(s) of corpus {corpus_version} have "
            f"no review record, and a gold nobody reviewed is never frozen or scored (decision "
            f"2026-10-22): {unreviewed[0][0]}: {unreviewed[0][1]}. Others: "
            f"{[s for s, _ in unreviewed[1:6]]}. Nothing was frozen. Freeze from a corpus built from "
            "reviewed golds only."
        )

    split = corpus_manifest.get("split_assignment") or {}
    held_out_ids = set(split.get("held_out_source_ids") or [])
    frozen_held_out: list[str] = []

    by_type: dict[str, int] = {}
    by_line: dict[str, int] = {}
    for source_id, doc_type in sorted(documents.items()):
        label, metadata = loaded[source_id]
        ocr_meta = client.read_json(paths.ocr_meta(doc_type, source_id, tenant_id))
        page_count = int(ocr_meta.get("page_count") or 0)
        if page_count < 1:
            raise FreezeError(f"{source_id} records no pages; it cannot be evaluated")

        base = f"{root}/{source_id}"
        for page in range(1, page_count + 1):
            client.write_bytes(
                f"{base}/page_{page}.png",
                client.read_bytes(paths.processed_page(doc_type, source_id, page, "png", tenant_id)),
            )
            md = paths.processed_page(doc_type, source_id, page, "md", tenant_id)
            if client.exists(md):
                client.write_text(f"{base}/page_{page}.md", client.read_text(md))
        client.write_json(f"{base}/metadata.json", {
            "doc_type": doc_type,
            "acord_form": metadata.get("acord_form"),
            # Classic auto is read as personal auto (common.lob.MERGED_LINES).
            "lob": merge_line(metadata.get("lob")),
            "is_scanned": is_scanned(ocr_meta),
            # A delivered split can place synthetic twins in test; the gate then
            # reports the real documents' scores apart (golden_eval "real_only").
            "synthetic": bool(metadata.get("synthetic", False)),
            "frozen_from_corpus": corpus_version,
            # Whether this document's carrier is absent from train and val (Fideon
            # SPEC_09 amendment item 4): held out by the split, or marked so by the
            # delivery. Evaluation reports these apart.
            "carrier": metadata.get("carrier"),
            "held_out_carrier": held_out_carrier(source_id, metadata, held_out_ids),
            # Who reviewed the gold and when, what drafted it, and a twin's seed:
            # the gate scores a frozen gold only with a review record here.
            **review_record(metadata),
        })
        if held_out_carrier(source_id, metadata, held_out_ids):
            frozen_held_out.append(source_id)
        # Written last: a document counts as part of the set once golden.json
        # exists (eval_set_source_ids), so an interrupted freeze leaves no
        # half-copied document that the gate would try to score.
        client.write_json(f"{base}/golden.json", label)
        by_type[doc_type] = by_type.get(doc_type, 0) + 1
        line = line_of(metadata.get("lob"))
        if line:
            by_line[line] = by_line.get(line, 0) + 1

    manifest = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_from_corpus": corpus_version,
        "tenant_id": paths._tenant(tenant_id),
        "git_commit": git_commit,
        "source_ids": sorted(documents),
        "documents_by_doc_type": dict(sorted(by_type.items())),
        # What the gate can say something about, per line. A line absent here is
        # not measured by the gate at all.
        "documents_by_line": dict(sorted(by_line.items())),
        "small_set_accepted": bool(small),
        # The carriers the split held out entirely. They are "unseen" to the model
        # only while no document of theirs trains; the build warns when one does.
        "held_out_carriers": split.get("held_out_carriers", {}),
        "held_out_carriers_by_line": split.get("held_out_carriers_by_line", {}),
        # The frozen documents reported apart as unseen carriers' (the split's own
        # hold-out and the delivery's marks alike).
        "held_out_source_ids": sorted(frozen_held_out),
    }
    client.write_json(manifest_key(tenant_id=tenant_id), manifest)
    log.info(
        "froze %d document(s) from corpus %s into %s: %s",
        len(documents), corpus_version, root, manifest["documents_by_doc_type"],
    )
    return manifest


def held_out_carrier(source_id: str, metadata: dict[str, Any], held_out_ids: set[str]) -> bool:
    """Whether a frozen document is an unseen carrier's: the split held its carrier
    out, or the delivery marked it so (label metadata ``held_out_carrier``)."""
    return source_id in held_out_ids or bool(metadata.get("held_out_carrier", False))


def mark_held_out(client: BlobClient, *, tenant_id: str | None = None) -> list[str]:
    """Mark, in an existing frozen set, the documents the delivery calls unseen
    carriers'. Returns them.

    For a set frozen while the importer still dropped the delivery's
    ``held_out_carrier`` flag: re-import the batch so the label metadata carries
    it, then run this. Only that flag changes - never the documents, labels or
    pages - so the yardstick stays the one every version was scored on; the gate
    can now report these documents apart.
    """
    from data_pipeline.labeling.export_golden_labels import load_golden_label

    root = paths.golden_eval_set_dir(tenant_id)
    manifest = frozen_manifest(client, tenant_id=tenant_id)
    if not manifest:
        raise FreezeError(f"no frozen eval set at {root}/ to mark")
    marked: list[str] = []
    for source_id in manifest.get("source_ids") or []:
        key = f"{root}/{source_id}/metadata.json"
        frozen = client.read_json(key)
        _label, metadata = load_golden_label(source_id, frozen["doc_type"], client, tenant_id)
        if held_out_carrier(source_id, metadata, set()) or frozen.get("held_out_carrier"):
            marked.append(source_id)
            if not frozen.get("held_out_carrier"):
                client.write_json(key, {**frozen, "held_out_carrier": True})
    client.write_json(manifest_key(tenant_id=tenant_id), {**manifest, "held_out_source_ids": sorted(marked)})
    log.info("marked %d frozen document(s) in %s as unseen carriers': %s", len(marked), root, marked)
    return marked


def mark_reviewed(client: BlobClient, *, tenant_id: str | None = None) -> dict[str, str]:
    """Copy each frozen document's review record from its label metadata into the
    frozen set. Returns ``{source_id: problem}`` for those still without one.

    For a set frozen before the freeze carried the record: the gate refuses a
    frozen gold with none (golden_eval.load_golden_set). Only the review keys
    change, and only to a real record - never the documents, labels or pages -
    so the yardstick stays the one every version was scored on. A document
    still listed has no record in its label metadata either: review it and
    import it again, or delete the set and freeze again.
    """
    from data_pipeline.labeling.export_golden_labels import load_golden_label, review_problem, review_record

    root = paths.golden_eval_set_dir(tenant_id)
    manifest = frozen_manifest(client, tenant_id=tenant_id)
    if not manifest:
        raise FreezeError(f"no frozen eval set at {root}/ to mark")
    missing: dict[str, str] = {}
    for source_id in manifest.get("source_ids") or []:
        key = f"{root}/{source_id}/metadata.json"
        frozen = client.read_json(key)
        if review_problem(None, frozen) is None:
            continue
        _label, metadata = load_golden_label(source_id, frozen["doc_type"], client, tenant_id)
        if problem := review_problem(None, metadata):
            missing[source_id] = problem
            continue
        client.write_json(key, {**frozen, **review_record(metadata)})
    log.info("review records in %s: %d frozen document(s) still have none", root, len(missing))
    return missing
