"""Blob and staging-volume layout — expressed once, here.

Every path in the system is built through this module. The layout appearing in
two places is how a training job ends up writing where a serving job will never
look, so nothing else constructs a path by string concatenation.

Two storage surfaces with **different guarantees**, and conflating them is the
bug this module's naming exists to prevent:

Azure Blob
    The artifact of record. Durable, versioned, and where the run registry lives.

RunPod staging volume
    Working storage between ``finetune`` (command 1) and ``package`` (command 2).
    Adapters and the ~16GB merged model sit here so quantization — which also
    runs on RunPod — does not pull them back out of Azure, a 32GB round trip for
    nothing. **No durability guarantee.** It is never called a registry
    (master §12a).

Tenancy (arch §8b): the ``tenant_id`` prefix is *reserved* on every path holding
tenant document data, so no migration is needed when a second broker arrives.
The build is single-tenant — ``tenant_id`` defaults from ``DEFAULT_TENANT_ID``
rather than being a required argument everywhere. The one rule that is live and
enforced is that a corpus file must never mix tenants, because corpus
composition is training data.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Final, Literal

from common.constants import ACTIVE_DOC_TYPES, UNCLASSIFIED

AdapterKind = Literal["foundation", "doc_type"]

#: Blob prefixes carrying tenant data. These get the tenant prefix.
#:
#: ``releases`` is here because **a model trained on a tenant's documents is that
#: tenant's data** (arch v2.1 §8b) — a deletion request that stops at the corpus
#: leaves the tenant's content encoded in weights and in a calibrator fitted on
#: their fields. Everything a release owns lives under that one prefix, rather
#: than being scattered into the v1 ``eval-reports/`` and ``calibration/`` trees:
#: those are still unscoped, and adding a tenant segment to only some paths
#: beneath them would make ``tenant_of`` read a version tag as a tenant id.
#:
#: ``golden-eval-set`` is here because each tenant is gated on its own frozen
#: set. The set is copied from one tenant's test split, so it holds that tenant's
#: documents; and source ids are numbered per tenant, so one shared prefix would
#: let the first tenant to freeze fix the yardstick for every tenant and make
#: another tenant's ``policy_0001`` look frozen. The whole prefix moved at once:
#: a set frozen at its root before this is refused until it is moved under its
#: tenant (``evaluation.freeze_eval_set``), so ``tenant_of`` never reads a source id.
TENANT_SCOPED: Final[frozenset[str]] = frozenset(
    {"raw-documents", "processed", "golden-labels", "corpus", "releases", "exports",
     "golden-eval-set"}
)

#: Blob prefixes holding no tenant data. These are never prefixed — the pinned
#: base model and the cross-tenant run registry.
#:
#: MIGRATION (arch v2.1 §8b): ``adapters``, ``merged-models`` and
#: ``quantized-models`` are still listed here and still unscoped. Under v2.1 they
#: are tenant data and belong above, but their path helpers take no ``tenant_id``
#: and 18 modules call them — that move lands with the serving topology change,
#: not here. Until then a tenant deletion cannot remove trained weights by
#: prefix: the purge plans (``pull_raw_pdfs.purge_plan`` / ``tenant_purge_plan``)
#: name each run trained on the tenant's corpora, its weights and its releases.
SHARED: Final[frozenset[str]] = frozenset(
    {"base-models", "adapters", "merged-models", "quantized-models",
     "registry", "eval-reports", "calibration"}
)

_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$")
_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
#: The adapter's name as a path segment (``common.adapter``; older runs' names
#: too, so the purge plan can still address them) — validated by shape here.
_SCOPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Used where an artifact is not per-document-type (a Foundation-only merge).
UNIFIED = "unified"


class PathError(ValueError):
    """Raised on an invalid tenant, version tag, document type, or format."""


# --------------------------------------------------------------------------
# Validation — cheap, and it turns a silent mis-write into an immediate error
# --------------------------------------------------------------------------

def default_tenant() -> str:
    return os.environ.get("DEFAULT_TENANT_ID") or "default"


def _tenant(tenant_id: str | None) -> str:
    tenant = (tenant_id or default_tenant()).strip().lower()
    if not _TENANT_RE.match(tenant):
        raise PathError(
            f"invalid tenant_id {tenant!r}; expected lowercase alphanumeric with - or _ "
            "(it becomes a path segment, so it cannot contain slashes or spaces)"
        )
    return tenant


def _version(version: str) -> str:
    """Accept ``v1``, ``v2.1``. Reject a bare ``1`` or a floating tag."""
    tag = str(version).strip()
    if not _VERSION_RE.match(tag):
        raise PathError(
            f"invalid version {version!r}; expected a tag like 'v1' or 'v2.1'. "
            "Untagged artifacts cannot be traced back to the run that made them."
        )
    return tag


def _scope(scope: str) -> str:
    """A training scope name, as a path segment."""
    name = str(scope).strip().lower()
    if not _SCOPE_RE.match(name):
        raise PathError(
            f"invalid scope {scope!r}; expected lowercase alphanumeric with underscores "
            "(it becomes a path segment and a run-id lineage)"
        )
    return name


def _scope_segment(scope: str | None) -> str:
    """How a training scope renders inside an artifact path.

    THE COMPATIBILITY RULE, in one place: the unified scope renders exactly what
    every path rendered before scopes existed, so no artifact already in Blob
    moves. Every other scope lands under ``scope/{name}/``.
    """
    name = (scope or UNIFIED).strip().lower()
    return UNIFIED if name == UNIFIED else _join("scope", _scope(name))


def is_valid_scope(scope: str) -> bool:
    """Whether a name could be a scope path segment. Non-raising."""
    return bool(_SCOPE_RE.match(str(scope).strip().lower()))


def _doc_type(doc_type: str, *, allow_unified: bool = False) -> str:
    dt = doc_type.strip().lower()
    if allow_unified and dt == UNIFIED:
        return dt
    # Raw documents may sit under `unclassified/` before their type is known;
    # nothing downstream of ingestion ever asks for that prefix, so it is
    # accepted here rather than blocking the documented holding bucket.
    if dt == UNCLASSIFIED:
        return dt
    if dt not in ACTIVE_DOC_TYPES:
        allowed = list(ACTIVE_DOC_TYPES) + ([UNIFIED] if allow_unified else [])
        raise PathError(f"unknown doc_type {doc_type!r}; expected one of {allowed}")
    return dt


def _join(*parts: str) -> str:
    return str(PurePosixPath(*[p.strip("/") for p in parts if p]))


# --------------------------------------------------------------------------
# Tenant-scoped: the PII-bearing document layers
# --------------------------------------------------------------------------

def raw_document_dir(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """``raw-documents/{tenant}/{doc_type}/{source_id}/``

    The only layer holding unredacted PII. It lives in a separate container with
    tighter RBAC, and only ingestion writes it (arch §18a).
    """
    return _join("raw-documents", _tenant(tenant_id), _doc_type(doc_type), source_id)


def raw_pdf(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """The immutable original. Write-once — a corrected document becomes a NEW
    ``source_id`` so historical training runs stay reproducible against the exact
    bytes they trained on (arch §18a)."""
    return _join(raw_document_dir(doc_type, source_id, tenant_id), "original.pdf")


def raw_metadata(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    return _join(raw_document_dir(doc_type, source_id, tenant_id), "metadata.json")


def processed_dir(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """``processed/{tenant}/{doc_type}/{source_id}/`` — MinerU output + page renders."""
    return _join("processed", _tenant(tenant_id), _doc_type(doc_type), source_id)


def processed_page(doc_type: str, source_id: str, page: int, suffix: str,
                   tenant_id: str | None = None) -> str:
    """A rendered page or its OCR text. Pages are 1-based and ordered — multi-page
    ordering is positionally meaningful to the model (arch §3)."""
    if page < 1:
        raise PathError(f"page numbers are 1-based, got {page}")
    return _join(processed_dir(doc_type, source_id, tenant_id), f"page_{page}.{suffix.lstrip('.')}")


def ocr_meta(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """Records ``mineru_version``, ``ocr_device``, resolution cap, and per-page
    table row counts (the last feeds the row-completeness check in IMPL-09)."""
    return _join(processed_dir(doc_type, source_id, tenant_id), "ocr_meta.json")


def golden_label_dir(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    return _join("golden-labels", _tenant(tenant_id), _doc_type(doc_type), source_id)


def golden_label(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    return _join(golden_label_dir(doc_type, source_id, tenant_id), "golden.json")


def label_metadata(doc_type: str, source_id: str, tenant_id: str | None = None) -> str:
    """Provenance: reviewer, date, draft backend, agreement score, and
    ``field_provenance`` — the observed surface label per canonical field."""
    return _join(golden_label_dir(doc_type, source_id, tenant_id), "label_metadata.json")


def corpus_root(tenant_id: str | None = None) -> str:
    """``corpus/{tenant}`` — every corpus version this tenant has built."""
    return _join("corpus", _tenant(tenant_id))


def corpus_dir(version: str, tenant_id: str | None = None) -> str:
    return _join(corpus_root(tenant_id), _version(version))


def corpus_split(version: str, doc_type: str, split: str, tenant_id: str | None = None) -> str:
    """``corpus/{tenant}/v{n}/{doc_type}/{split}.jsonl``"""
    if split not in ("train", "val", "test"):
        raise PathError(f"unknown split {split!r}; expected train, val or test")
    return _join(corpus_dir(version, tenant_id), _doc_type(doc_type), f"{split}.jsonl")


def corpus_eval_split(version: str, split: str, tenant_id: str | None = None) -> str:
    """``corpus/{tenant}/v{n}/val/val.jsonl`` — one file, not one per doc type.

    Under arch v2.1 §8.1 the corpus is unified: one adapter trains on every
    document type and every task, so splitting the evaluation files by type would
    be organising them by a dimension the training run does not have. The
    per-type ``corpus_split`` remains for the §4.2 graduation path.
    """
    if split not in ("val", "test"):
        raise PathError(
            f"unknown eval split {split!r}; expected 'val' or 'test'. Training reads epoch "
            "files (corpus_epoch_file), not a 'train' split file."
        )
    return _join(corpus_dir(version, tenant_id), split, f"{split}.jsonl")


def corpus_scope_epoch_file(
    version: str, epoch: int, scope: str | None = None, tenant_id: str | None = None
) -> str:
    """One epoch file, filtered to a scope's document types.

    The unified scope returns the UNSCOPED key — the corpus as built — so no copy
    and no migration. A narrower scope reads its own filtered view, written by
    ``training.corpus_view``.

    The corpus itself is built ONCE, for every type, and filtered per scope. A
    per-scope BUILD would draw its own split, groups and modality regimes, so a
    document could be train in one scope and test in another: the two releases
    would be incomparable, and a document the policy model trained on could sit
    in the unified model's test set.
    """
    if not scope or scope == UNIFIED:
        return corpus_epoch_file(version, epoch, tenant_id)
    if not 1 <= int(epoch) <= 4:
        raise PathError(f"epoch {epoch} is outside 1-4")
    return _join(
        corpus_dir(version, tenant_id), "train", "scope", _scope(scope), f"epoch_{int(epoch)}.jsonl"
    )


def corpus_scope_eval_split(
    version: str, split: str, scope: str | None = None, tenant_id: str | None = None
) -> str:
    """``val``/``test`` filtered to a scope's document types."""
    if not scope or scope == UNIFIED:
        return corpus_eval_split(version, split, tenant_id)
    if split not in ("val", "test"):
        raise PathError(f"unknown eval split {split!r}; expected 'val' or 'test'")
    return _join(corpus_dir(version, tenant_id), split, "scope", _scope(scope), f"{split}.jsonl")


def corpus_manifest(version: str, tenant_id: str | None = None) -> str:
    """Pins everything the corpus depends on: MinerU version and device, schema
    and prompt-template versions, LoB and alias coverage, de-identification status."""
    return _join(corpus_dir(version, tenant_id), "manifest.json")


# --------------------------------------------------------------------------
# Shared: models, adapters, registry — no tenant data
# --------------------------------------------------------------------------

def base_model_dir() -> str:
    """Cached from Hugging Face at the pinned revision."""
    return _join("base-models", "qwen3-vl-8b-instruct")


def adapter_dir(kind: AdapterKind, version: str, doc_type: str | None = None) -> str:
    """``adapters/foundation/v{n}/`` or ``adapters/{doc_type}/v{n}/``

    Two lineages only. A per-tenant lineage is **not built** until a broker
    actually requires one (arch §8b).
    """
    tag = _version(version)
    if kind == "foundation":
        if doc_type is not None:
            raise PathError("the Foundation adapter is not per-doc_type — pass doc_type=None")
        return _join("adapters", "foundation", tag)
    if kind == "doc_type":
        if doc_type is None:
            raise PathError("a per-type adapter needs a doc_type")
        return _join("adapters", _doc_type(doc_type), tag)
    raise PathError(f"unknown adapter kind {kind!r}; expected 'foundation' or 'doc_type'")


def scoped_adapter_dir(scope: str | None, version: str) -> str:
    """``adapters/foundation/v{n}/`` for unified, ``adapters/scope/{name}/v{n}/`` otherwise.

    A sibling of :func:`adapter_dir` rather than another ``kind``, because
    ``adapter_dir("doc_type", v, "policy")`` already owns ``adapters/policy/``.
    A policy-SCOPE adapter (trained on the base, covering policies) and a §4.2
    graduated policy adapter (trained on the merged foundation) are different
    artifacts, and one prefix for both is the collision this avoids.
    """
    tag = _version(version)
    if not scope or scope == UNIFIED:
        return adapter_dir("foundation", tag)
    return _join("adapters", "scope", _scope(scope), tag)


def merged_model_dir(
    version: str, doc_type: str | None = None, *, scope: str | None = None
) -> str:
    """``merged-models/{scope|doc_type|unified}/v{n}/`` — bf16, post ``merge_and_unload``.

    ``doc_type`` is the §4.2 graduated lineage and ``scope`` is the training
    scope; they are different axes, so naming both is refused rather than
    silently resolved to one.
    """
    if doc_type and scope and scope != UNIFIED:
        raise PathError(
            f"merged model cannot be both scope {scope!r} and doc_type {doc_type!r} — "
            "a scoped run covers document types, a graduated adapter IS one"
        )
    segment = _doc_type(doc_type, allow_unified=True) if doc_type else _scope_segment(scope)
    return _join("merged-models", segment, _version(version))


#: Formats a release can be served in (arch v2.1 §13a). bf16 is the reference
#: every quantized format's drop is measured against; fp8 is the default serving
#: format from the second cycle; awq_int4 is for VRAM-constrained serving only.
SERVING_FORMATS: Final[frozenset[str]] = frozenset({"bf16", "fp8", "awq_int4"})

#: GGUF export targets llama.cpp, not the vLLM serving runtime — an optional
#: edge export, validated separately, never the serving path.
GGUF_FORMATS: Final[frozenset[str]] = frozenset({"fp16", "bf16", "q8_0", "q6_k", "q5_k_m", "q4_k_m"})


def _format(fmt: str, allowed: frozenset[str] = SERVING_FORMATS) -> str:
    fmt = fmt.strip().lower()
    if fmt not in allowed:
        raise PathError(f"unknown quantization format {fmt!r}; expected one of {sorted(allowed)}")
    return fmt


def quantized_model_dir(
    version: str,
    fmt: str,
    doc_type: str | None = None,
    *,
    runtime: Literal["vllm", "gguf"] | None = None,
    scope: str | None = None,
) -> str:
    """``quantized-models/{scope}/v{n}/{runtime}/{format}/``

    The runtime segment is not decoration. A serving format and a GGUF export are
    loaded by different programs — vLLM and llama.cpp — and putting them under one
    prefix invites deploying the wrong one (arch v2.1 §13a).

    ``runtime`` is inferred when the format names one runtime only. ``bf16`` names
    both, and defaults to ``vllm`` because that is the serving artifact; a GGUF
    edge export must pass ``runtime="gguf"``, or it would be written into the
    directory the serving endpoint loads.
    """
    fmt = _format(fmt, SERVING_FORMATS | GGUF_FORMATS)
    if runtime is None:
        runtime = "vllm" if fmt in SERVING_FORMATS else "gguf"
    allowed = SERVING_FORMATS if runtime == "vllm" else GGUF_FORMATS
    if fmt not in allowed:
        raise PathError(
            f"{fmt!r} is not a {runtime} format; {runtime} formats are {sorted(allowed)}"
        )
    if doc_type and scope and scope != UNIFIED:
        raise PathError(
            f"quantized model cannot be both scope {scope!r} and doc_type {doc_type!r}"
        )
    segment = _doc_type(doc_type, allow_unified=True) if doc_type else _scope_segment(scope)
    return _join("quantized-models", segment, _version(version), runtime, fmt)


def run_manifest(
    run_id: str, run_type: str, doc_type: str | None = None, *, scope: str | None = None
) -> str:
    """``registry/foundation/{run_id}/run_manifest.json`` or
    ``registry/adapters/{doc_type}/{run_id}/run_manifest.json``

    Written even when the weights are only staged, so a reclaimed volume never
    means a training run that happened and left no trace (master §12a).
    """
    # `unified` and `foundation` share the slot deliberately: the v2.1 unified
    # extractor occupies the same position in the lineage the Foundation did, and
    # keeping one prefix means query_registry, the cascade query and the ViT gate
    # keep reading one place across the topology change. The run_type recorded ON
    # the manifest is what says which it actually is.
    if run_type in ("unified", "foundation"):
        return _join("registry", "foundation", run_id, "run_manifest.json")
    if run_type == "per_type_adapter":
        if doc_type is None:
            raise PathError("a per-type adapter manifest needs a doc_type")
        return _join("registry", "adapters", _doc_type(doc_type), run_id, "run_manifest.json")
    if run_type == "scoped":
        # Its own prefix, not the Foundation one: two scoped runs and a unified
        # run can share a version, and filing them together would make
        # "which manifest is v2's" ambiguous.
        if not scope:
            raise PathError("a scoped run manifest needs the scope that produced it")
        return _join("registry", "scope", _scope(scope), run_id, "run_manifest.json")
    raise PathError(
        f"unknown run_type {run_type!r}; expected 'unified', 'foundation', "
        "'per_type_adapter' or 'scoped'"
    )


def model_card(
    run_id: str, run_type: str, doc_type: str | None = None, *, scope: str | None = None
) -> str:
    """``model_card.md`` beside the run's ``run_manifest.json`` (Fideon SPEC_09 amendment item 7)."""
    manifest = run_manifest(run_id, run_type, doc_type, scope=scope)
    return manifest[: -len("run_manifest.json")] + "model_card.md"


def registry_index() -> str:
    """Flat table of every run — status and key metrics at a glance, without
    opening individual manifests."""
    return _join("registry", "registry_index.json")


def calibration_params(version: str, doc_type: str) -> str:
    """Fitted per model version AND doc type; never reused across versions."""
    return _join("calibration", _version(version), f"{_doc_type(doc_type)}.json")


def eval_report(version: str, doc_type: str | None = None, *, scope: str | None = None) -> str:
    base = _join("eval-reports", _version(version), _scope_suffix(scope))
    return _join(base, _doc_type(doc_type), "report.json") if doc_type else _join(base, "summary.json")


def _scope_suffix(scope: str | None) -> str:
    """``""`` for unified, ``scope/{name}`` otherwise.

    Empty for unified so the key is byte-identical to what has always been
    written: ``_join`` drops empty parts.
    """
    name = (scope or UNIFIED).strip().lower()
    return "" if name == UNIFIED else _join("scope", _scope(name))


def gate_decision(version: str, *, scope: str | None = None) -> str:
    """Where the promotion gate's own verdict is written.

    Deliberately NOT ``eval_report(version)``. The gate used to write its thin
    decision dict over ``eval-reports/{v}/summary.json`` — the key
    ``EvalReport.as_dict()`` writes — destroying ``by_doc_type`` and every error
    record. ``vit_gate.evaluate_from_report`` then read zero image-only and zero
    scanned documents and returned ``insufficient_data`` for ever, so the ViT
    escalation decision could never be made on real evidence again.
    """
    return _join("eval-reports", _version(version), _scope_suffix(scope), "gate_decision.json")


#: A release is named by its version (``v3``): one number names the run, the
#: adapter and the release of a cycle. ``release-YYYY.M.N`` is the form releases
#: had before (decision 2026-10-22), still read so an older index loads.
_RELEASE_RE = re.compile(r"^(v\d+(\.\d+)*|release-\d{4}\.\d{1,2}\.\d+)$")


def _release(release_id: str) -> str:
    """A version tag (``v3``). Validated because a release id is what serving
    pulls — a typo there is a deploy that silently serves the wrong weights, or
    nothing at all."""
    if not _RELEASE_RE.match(release_id):
        raise PathError(f"invalid release id {release_id!r}; expected the cycle's version, e.g. v3")
    return release_id


def is_valid_release_id(release_id: str) -> bool:
    return bool(_RELEASE_RE.match(release_id or ""))


def releases_root(tenant_id: str | None = None) -> str:
    """``releases/{tenant}/`` — every release this tenant has."""
    return _join("releases", _tenant(tenant_id))


def version_number(version: str) -> int | None:
    """The whole number of a version tag (``v3`` -> 3, ``v2.1`` -> 2); None for
    anything that is not one."""
    tag = str(version).strip().lower()
    if not _VERSION_RE.match(tag):
        return None
    return int(tag[1:].split(".")[0])


def next_version(used: Iterable[str]) -> str:
    """The next cycle's version: one more than the highest whole number in
    ``used`` (version tags; anything else is ignored), ``v1`` when there is none.

    Only a fresh cycle takes it. A resume names its version (``--version``):
    derived again after the first attempt wrote anything, it would be the next
    number, and one cycle's corpus, run and release would split across two.
    """
    numbers = [n for n in (version_number(v) for v in used) if n is not None]
    return f"v{max(numbers, default=0) + 1}"



def release_bundle(release_id: str, tenant_id: str | None = None) -> str:
    """The bundle that is gated, promoted, served and selected by ``--model``.

    Tenant-scoped, because a model trained on a tenant's documents IS that
    tenant's data (arch v2.1 §8b) — the deletion cascade has to reach it.
    """
    return _join(release_dir(release_id, tenant_id), "bundle.json")


def release_index(tenant_id: str | None = None) -> str:
    """Flat table of every release, for the same reason ``registry_index`` exists."""
    return _join("releases", _tenant(tenant_id), "release_index.json")


def release_dir(release_id: str, tenant_id: str | None = None) -> str:
    """Everything one release owns, under one prefix.

    Deliberately self-contained rather than scattered across the v1
    ``eval-reports/`` and ``calibration/`` trees: a tenant deletion has to remove
    the calibrators fitted on that tenant's fields as well as the bundle, and one
    prefix makes that one delete instead of a checklist (arch v2.1 §8b).
    """
    return _join("releases", _tenant(tenant_id), _release(release_id))


def release_calibrators(release_id: str, fmt: str, tenant_id: str | None = None) -> str:
    """Calibrators are per release AND per serving format.

    Quantization moves the logprob distribution, so a calibrator fitted on bf16
    is wrong for FP8 — it would report confidence for a distribution that format
    does not produce (arch v2.1 §5.3).
    """
    return _join(release_dir(release_id, tenant_id), "calibration", _format(fmt), "calibrators.json")


def release_gate_decision(release_id: str, fmt: str, tenant_id: str | None = None) -> str:
    """One gate run per serving format — a format inherits nothing from bf16's
    result, because quantization degrades exactly what was fine-tuned in."""
    return _join(release_dir(release_id, tenant_id), "gate", _format(fmt), "gate_decision.json")


def corpus_epoch_file(version: str, epoch: int, tenant_id: str | None = None) -> str:
    """One training file per epoch (arch v2.1 §6.1, §8.1).

    Modality mode is sampled per document per epoch, so each epoch is a different
    draw over the same documents. Materializing them makes a run reproducible
    from the corpus alone, rather than depending on a sampler running identically
    at training time. Four are always written; training uses the first N.
    """
    if not 1 <= int(epoch) <= 4:
        raise PathError(
            f"epoch {epoch} is outside 1-4; the corpus always materializes four epoch files "
            "because the §11a epoch sweep tests up to four passes"
        )
    return _join(corpus_dir(version, tenant_id), "train", f"epoch_{int(epoch)}.jsonl")


def checkpoint_selection(
    version: str, tenant_id: str | None = None, *, scope: str | None = None
) -> str:
    """Which checkpoint shipped, and what it beat (arch v2.1 §11.2).

    Recorded because the merged weights do not say. Once the staging volume is
    reclaimed, "which checkpoint is this model" is otherwise unanswerable — and
    the selection margin is what tells a later regression review whether the
    choice was decisive or inside eval noise.
    """
    return _join(
        "eval-reports", _version(version), _scope_suffix(scope), "checkpoint_selection.json"
    )


def golden_eval_set_dir(tenant_id: str | None = None) -> str:
    """``golden-eval-set/{tenant}`` — one tenant's frozen eval set.

    Frozen and versioned separately, held constant across corpus versions so
    model versions stay comparable (arch §8). Never trained on.

    Per tenant, because each tenant's releases are gated on documents like its
    own, and because source ids are allocated per tenant: under one shared
    prefix, the tenant that froze first would fix the yardstick for all of them,
    and another tenant's document with the same id would be taken as frozen.
    """
    return _join("golden-eval-set", _tenant(tenant_id))


def legacy_golden_eval_manifest() -> str:
    """Where a set frozen before eval sets were per tenant kept its manifest.

    Nothing reads a set there. It is named only so it can be refused rather than
    ignored (``evaluation.freeze_eval_set``): ignored, its tenant would look
    unfrozen and could freeze a new set, changing the yardstick its earlier
    versions were compared on.
    """
    return _join("golden-eval-set", "manifest.json")


# --------------------------------------------------------------------------
# RunPod staging volume — working storage, NOT the registry
# --------------------------------------------------------------------------

def staging_root() -> str:
    """Absolute filesystem path to the staging area on the mounted volume.

    Absolute on purpose: ``_join`` strips leading slashes because blob keys have
    none, but this is a real mount point. A relative path here would write to
    whatever the working directory happens to be and silently miss the volume.
    """
    mount = os.environ.get("RUNPOD_VOLUME_MOUNT") or "/runpod-volume"
    return "/" + _join(mount, "staging") if mount.startswith("/") else _join(mount, "staging")


def _under_staging(*parts: str) -> str:
    root = staging_root()
    tail = _join(*parts)
    return f"{root.rstrip('/')}/{tail}"


def staging_adapter_dir(kind: AdapterKind, version: str, doc_type: str | None = None) -> str:
    """Mirrors :func:`adapter_dir` so ``package`` copies rather than translates."""
    return _under_staging(adapter_dir(kind, version, doc_type))


def scoped_staging_adapter_dir(scope: str | None, version: str) -> str:
    """The staging mirror of :func:`scoped_adapter_dir`."""
    return _under_staging(scoped_adapter_dir(scope, version))


def staging_parent_adapter_dir(scope: str, version: str) -> str:
    """Where cycle ``version`` keeps the adapter it continues (its parent's), on
    the staging volume (decision 2026-10-23)."""
    return _under_staging("parent-adapters", _scope(scope), _version(version))


def staging_merged_model_dir(
    version: str, doc_type: str | None = None, *, scope: str | None = None
) -> str:
    return _under_staging(merged_model_dir(version, doc_type, scope=scope))


def staging_quantized_model_dir(
    version: str,
    fmt: str,
    doc_type: str | None = None,
    *,
    runtime: Literal["vllm", "gguf"] | None = None,
    scope: str | None = None,
) -> str:
    return _under_staging(
        quantized_model_dir(version, fmt, doc_type, runtime=runtime, scope=scope)
    )


def staging_train_data_dir(scope: str, version: str) -> str:
    """Local copy of one run's training data: its JSONL files and page images.

    ms-swift reads local files. The corpus lives in Blob, so a run copies what it
    trains on here, onto the volume, before launch (``training.stage_data``).
    Keyed by scope and version, so two scoped runs at one version never share —
    or clear — each other's copy.
    """
    return _under_staging("train-data", scope, version)


def staging_train_images_dir(corpus_version: str, tenant_id: str | None = None) -> str:
    """Page images for one corpus version, shared by every run that trains on it.

    Keyed by corpus version, not by run: a unified run and three scoped runs at
    one version read the same pages, and copying them four times fills the
    volume with duplicates. Not shared ACROSS versions: a re-OCR writes new
    pixels under the same key, and a cache spanning versions would keep serving
    the old ones.
    """
    return _under_staging("train-images", _tenant(tenant_id), corpus_version)


def staging_eval_report(
    version: str, doc_type: str | None = None, *, scope: str | None = None
) -> str:
    return _under_staging(eval_report(version, doc_type, scope=scope))


def staging_run_manifest(run_id: str) -> str:
    """A working copy. The durable one always goes to Blob (master §12a)."""
    return _under_staging("run_manifests", f"{run_id}.json")


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------

def is_tenant_scoped(blob_path: str) -> bool:
    """Whether a path's top-level prefix carries tenant data."""
    return str(blob_path).strip("/").split("/", 1)[0] in TENANT_SCOPED


def tenant_of(blob_path: str) -> str | None:
    """Extract the tenant from a tenant-scoped path, else ``None``.

    Used to assert a corpus file never mixes tenants — the one live tenancy rule.
    """
    parts = str(blob_path).strip("/").split("/")
    return parts[1] if len(parts) >= 2 and parts[0] in TENANT_SCOPED else None


#: Staging for delivered batches, in the raw container: ``intake/{batch}/{document}/``
#: holding the PDF, golden.json and metadata.json exactly as delivered. Uploaded
#: with azcopy; pulled onto the pod by data_pipeline.ingestion.pull_intake.
INTAKE_PREFIX = "intake"
_BATCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def intake_batch_dir(batch: str) -> str:
    """``intake/{batch}`` — one delivered batch, staged before import."""
    if not _BATCH_RE.match(batch or "") or ".." in batch:
        raise ValueError(f"intake batch {batch!r} must be letters, digits, '.', '_' or '-'")
    return _join(INTAKE_PREFIX, batch)


#: Results made for people to download to their own machine - a comparison's
#: PDF, gold and extracted JSONs and workbooks: ``exports/{tenant}/{kind}/{name}``.
#: In the RAW container, under its rules: they hold the same unredacted policy
#: data as the PDFs, and the azcopy access a laptop uploads intake/ with already
#: reaches it, so nothing new has to be granted to download them.
EXPORTS_PREFIX = "exports"


def export_dir(kind: str, name: str, tenant_id: str | None = None) -> str:
    """``exports/{tenant}/{kind}/{name}`` - one downloadable result set."""
    for part in (kind, name):
        if not _BATCH_RE.match(part or "") or ".." in part:
            raise ValueError(f"export name {part!r} must be letters, digits, '.', '_' or '-'")
    return _join(EXPORTS_PREFIX, _tenant(tenant_id), kind, name)


def requires_raw_container(blob_path: str) -> bool:
    """Whether a path belongs in the separately-permissioned raw container.

    ``raw-documents/`` is the only layer with unredacted PII, so it is reachable
    only from ingestion and OCR — never from training or serving (arch §18a).
    """
    # The bare prefix counts. Stripping the trailing slash before testing for
    # one classified "raw-documents" itself as non-raw, so `list("raw-documents")`
    # from a training context bypassed the AccessDeniedError guard entirely.
    normalised = str(blob_path).strip("/")
    # intake/ is the staging area for delivered batches (PDFs and their labels,
    # uploaded with azcopy before import): the same unredacted PII, the same rules.
    return any(normalised == layer or normalised.startswith(layer + "/")
               for layer in ("raw-documents", INTAKE_PREFIX, EXPORTS_PREFIX))
