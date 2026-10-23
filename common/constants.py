"""Project-wide constants — the vocabularies and ratios every stage shares.

Anything here is referenced, never re-declared. Two items in particular exist
here specifically so they have exactly one definition:

* the **doc_type -> Fideon SPEC_00 canonical model mapping** (master §1.2), and
* the **modality mix** (arch §6), which governs corpus composition.
"""

from __future__ import annotations

from typing import Final, NamedTuple

# --------------------------------------------------------------------------
# Document types
# --------------------------------------------------------------------------

#: Active document types. Quote and Endorsement are deferred (master §1).
#:
#: This is the **write** vocabulary: what may be labeled, minted as a new
#: ``source_id``, and served. It is deliberately NOT the set the adapter trains
#: — that is ``common.adapter.ADAPTER.doc_types``, policies. Three sets, three jobs:
#:
#: * ``KNOWN_DOC_TYPES``  — every type this repo has ever named (read path).
#: * ``ACTIVE_DOC_TYPES`` — what may be minted and served today (write path).
#: * ``ADAPTER.doc_types`` — what a fine-tuning cycle trains.
ACTIVE_DOC_TYPES: Final[tuple[str, ...]] = ("acord", "policy", "lossrun")

#: ACORD form numbers with their own schema. One shared `acord` ADAPTER covers
#: all of them (arch §4b Option 1) — schema granularity is not adapter
#: granularity. Per-form adapters wait for 1000+ examples of that form *and*
#: evidence the shared adapter underperforms on it.
ACORD_FORMS: Final[frozenset[str]] = frozenset({"25", "125", "140"})


class CanonicalModel(NamedTuple):
    """How our short ``doc_type`` maps onto the Fideon SPEC_00 canonical model."""

    canonical_key: str
    model_name: str
    active: bool


#: The mapping table from master §1.2. The architecture uses Fideon SPEC_00's canonical
#: names; corpus paths, adapters, and CLIs use the short tags. These are the same
#: things, resolved here and nowhere else.
#:
#: If Fideon SPEC_00 keys its models differently, THIS TABLE is the only place to
#: change — flag it rather than renaming paths.
DOC_TYPE_TO_CANONICAL: Final[dict[str, CanonicalModel]] = {
    "lossrun": CanonicalModel("loss_run", "LossRunDocument", True),
    "policy": CanonicalModel("policy_check", "PolicyCheckDocument", True),
    "acord": CanonicalModel("acord_mapping", "ACORDMappingDocument", True),
    "quote": CanonicalModel("quote_gen", "QuoteGenDocument", False),  # deferred
}

#: Reverse lookup, for reading Fideon-side payloads.
CANONICAL_TO_DOC_TYPE: Final[dict[str, str]] = {
    v.canonical_key: k for k, v in DOC_TYPE_TO_CANONICAL.items()
}

#: Every document type this repo has ever named, active or not — the **read**
#: vocabulary. Historical artifacts must stay readable after a type is retired
#: from ``ACTIVE_DOC_TYPES``: a corpus, a golden label and every ``source_id``
#: that names a retired type were all valid when written, and refusing to parse
#: them would strand the artifacts rather than pause the type.
KNOWN_DOC_TYPES: Final[tuple[str, ...]] = tuple(DOC_TYPE_TO_CANONICAL)


#: The holding bucket for documents whose type is not yet known. Ingestion
#: accepts it so a wrong guess is never baked into the `source_id` — the key
#: every later stage joins on. It is NOT an active type: nothing downstream of
#: ingestion trains, evaluates or serves against it.
UNCLASSIFIED = "unclassified"


def canonical_key(doc_type: str) -> str:
    """``'lossrun'`` -> ``'loss_run'``. Raises on an unknown or deferred type."""
    entry = DOC_TYPE_TO_CANONICAL.get(doc_type.lower())
    if entry is None:
        raise KeyError(f"unknown doc_type {doc_type!r}; known: {sorted(DOC_TYPE_TO_CANONICAL)}")
    if not entry.active:
        raise KeyError(f"doc_type {doc_type!r} is deferred and must not be implemented (master §1.2)")
    return entry.canonical_key


def canonical_model(doc_type: str) -> str:
    """``'lossrun'`` -> ``'LossRunDocument'``."""
    canonical_key(doc_type)  # reuse the active/known checks
    return DOC_TYPE_TO_CANONICAL[doc_type.lower()].model_name


# --------------------------------------------------------------------------
# Modality regimes (arch §6)
# --------------------------------------------------------------------------

#: The three input regimes one source document expands into.
MODALITY_MODES: Final[tuple[str, ...]] = ("ocr_plus_image", "noisy_ocr_image", "image_only")

#: Target share of the Foundation corpus per regime. `noisy_ocr_image` teaches
#: image-over-OCR arbitration; `image_only` satisfies the must-work-without-OCR
#: requirement — it is the regime with no OCR backstop, so the ViT is doing the
#: reading alone.
MODALITY_MIX: Final[dict[str, float]] = {
    "ocr_plus_image": 0.50,
    "noisy_ocr_image": 0.20,
    "image_only": 0.30,
}

#: `noisy_ocr_image` renders the SAME system prompt as `ocr_plus_image` — the
#: noise lives in the data, not the instruction (arch §6). Prompt rendering keys
#: off this rather than special-casing the mode.
PROMPT_MODE_ALIASES: Final[dict[str, str]] = {"noisy_ocr_image": "ocr_plus_image"}

# --------------------------------------------------------------------------
# Corpus splitting (arch §8)
# --------------------------------------------------------------------------


class SplitRatio(NamedTuple):
    train: float
    val: float
    test: float


#: Split ratios scale with per-type volume. At pilot size a strict 70/20/10
#: leaves 2-3 test documents per type, too few to trust any single metric — so
#: pilot numbers are directional, not final.
SPLIT_RATIOS_BY_VOLUME: Final[tuple[tuple[int, SplitRatio], ...]] = (
    (200, SplitRatio(0.70, 0.18, 0.12)),    # pilot (~25-30/type)
    (1000, SplitRatio(0.75, 0.15, 0.10)),   # 200-1000/type
    (10**9, SplitRatio(0.80, 0.10, 0.10)),  # 1000+/type, the target state
)


def _assert_ratios_sum_to_one() -> None:
    """Every band must sum to 1.0.

    The splitter assigns by hash threshold — train below ``train``, val below
    ``train + val``, test above — so a triple summing to 1.05 silently gave test
    10% while the corpus manifest recorded 15%. Checked at import: a ratio table
    is edited by hand, and this is the only thing that would notice.
    """
    for threshold, ratio in SPLIT_RATIOS_BY_VOLUME:
        total = round(sum(ratio), 6)
        if total != 1.0:
            raise ValueError(
                f"split ratio for the <{threshold} band sums to {total}, not 1.0: {ratio}. "
                "The hash-threshold splitter would silently give the test split "
                f"{round(1.0 - ratio.train - ratio.val, 4):.0%} while the manifest recorded "
                f"{ratio.test:.0%}."
            )


_assert_ratios_sum_to_one()


def split_ratio_for(n_documents: int) -> SplitRatio:
    """Pick the split ratio appropriate to how much data actually exists."""
    for threshold, ratio in SPLIT_RATIOS_BY_VOLUME:
        if n_documents < threshold:
            return ratio
    return SPLIT_RATIOS_BY_VOLUME[-1][1]


# --------------------------------------------------------------------------
# Line of Business (arch §0b)
# --------------------------------------------------------------------------

#: Minimum share of training examples per LoB value. Under-coverage is a loud
#: warning, not a build failure — the remedy is collecting documents, which is a
#: data-acquisition decision rather than a build-time one.
LOB_COVERAGE_TARGET: Final[float] = 0.20

# --------------------------------------------------------------------------
# Vision / inference defaults (arch §11)
# --------------------------------------------------------------------------

#: Long-side cap in pixels. Image token count scales with resolution, making this
#: the single biggest cost/latency lever — and it MUST be identical in corpus
#: prep and production inference, or the model sees a different distribution at
#: serving time than it trained on.
DEFAULT_RESOLUTION_CAP_PX: Final[int] = 1792
RESOLUTION_CAP_RANGE_PX: Final[tuple[int, int]] = (1536, 2048)

#: Fields below this calibrated confidence route to human review (arch §5).
#: Tunable — the value is a starting point, not a settled threshold.
DEFAULT_REVIEW_CONFIDENCE_THRESHOLD: Final[float] = 0.70

# --------------------------------------------------------------------------
# Labeling (arch §4c)
# --------------------------------------------------------------------------

#: Below this many labels per doc type, every pre-annotation draft gets 100%
#: human review — no confidence routing, because there is no trustworthy model
#: to route with yet.
DAY_ZERO_MIN_LABELS_PER_TYPE: Final[int] = 25

# --------------------------------------------------------------------------
# LoRA ranks (arch §9, §11)
# --------------------------------------------------------------------------

FOUNDATION_LORA_RANK: Final[int] = 64
FOUNDATION_LORA_ALPHA: Final[int] = 128
PER_TYPE_LORA_RANK: Final[int] = 16
PER_TYPE_LORA_ALPHA: Final[int] = 32
LORA_DROPOUT: Final[float] = 0.05

#: LoRA targets: attention + feed-forward projections in every decoder layer,
#: plus the vision-language projector. The projector is where image evidence
#: fuses with language, making it the locus of OCR-versus-image arbitration
#: (arch §9a). The ViT is excluded by default (arch §3).
LORA_TARGET_MODULES: Final[tuple[str, ...]] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


#: Strings ms-swift's template parses as special tags wherever they appear in
#: message text: media placeholders and grounding markers. Text containing one
#: is rewritten or fails to encode, so the rendered prompt would differ from the
#: one serving sends. A document whose rows contain one is set aside at corpus
#: build (``build_jsonl``) and refused again at staging (``training.stage_data``).
TRAINER_SPECIAL_TAGS: Final[tuple[str, ...]] = (
    "<image>", "<video>", "<audio>", "<bbox>", "<ref-object>",
)


#: The lowest inter-annotator agreement (``export_golden_labels.
#: inter_annotator_agreement``, field level) a double-labelled document may show
#: before training refuses to start. None: no spec states one, so the preflight
#: reports agreement and warns, and it blocks once the product owner sets it.
MIN_INTER_ANNOTATOR_AGREEMENT: float | None = None
