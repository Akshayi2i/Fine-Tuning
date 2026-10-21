"""Message assembly — the single place prompt, images, and OCR come together.

Corpus build (IMPL-05), evaluation (IMPL-08), serving (IMPL-11) and testing
(IMPL-12) all call :func:`build_messages`. There is deliberately no second path,
because four contexts assembling their own inputs is how "test == prod" quietly
stops being true.

What this module guarantees:

* the system prompt is rendered by ``common.prompts`` and nothing else;
* ``image_only`` omits the OCR block *and* uses the image-only prompt, so the
  absence is declared rather than silently implied (arch §6);
* page images stay **ordered**, because multi-page ordering is positionally
  meaningful under Interleaved-MRoPE (arch §3);
* each page's image is **immediately followed by that page's own OCR text**, so
  which text belongs to which image is positional rather than something the
  model has to infer. Concatenating the pages into one block destroyed that
  correspondence, and — because a blank line ends a Markdown table — split every
  table that crossed a page boundary into two, the second half without its
  header;
* a routed subset of pages is the **same structure with fewer pairs**, so a long
  policy is served in a shape the model actually trained on;
* the resolution cap comes from config and matches the cap the corpus was built
  with — asserted, not assumed (arch §11).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config import assert_resolution_parity, resolution_cap_px
from common.constants import MODALITY_MODES
from common.lob import normalize_lob
from common.prompts import (
    PROMPT_TEMPLATE_VERSION,
    prompt_fingerprint,
    render_system_prompt,
)
from common.schemas import schema_version

#: Prefixed to each page's OCR text. Deliberately visible text rather than an
#: HTML comment: models attend to visible tokens far more reliably, and this
#: marker has a second job — on a routed request it tells the model it is
#: looking at a fragment of a longer document, so the fields that are absent are
#: absent because they live on pages it was not shown. Angle brackets are not
#: Markdown syntax, so the marker cannot be mistaken for document content the
#: way a `#` heading or a `---` rule would be.
PAGE_MARKER = "<page {page} of {total}>"


def page_total(messages: list[dict[str, Any]]) -> int | None:
    """The document's page count, read back from a prompt's page markers.

    Every page a prompt carries is marked ``<page i of N>``; ``N`` is the whole
    document, not the window. None when no marker is present.
    """
    import re

    totals = [
        int(match)
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "text"
        for match in re.findall(r"<page \d+ of (\d+)>", part.get("text") or "")
    ]
    return max(totals) if totals else None


def shown_pages(messages: list[dict[str, Any]]) -> list[int]:
    """The page numbers a prompt carries, read back from its ``<page i of N>``
    markers: a window's own pages, in the document's numbering."""
    import re

    return sorted({
        int(match)
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "text"
        for match in re.findall(r"<page (\d+) of \d+>", part.get("text") or "")
    })


class InputBuilderError(RuntimeError):
    """Raised on an unusable modality mode, page list, or image path."""


@dataclass(frozen=True)
class BuiltMessages:
    """A chat-format message list plus the versions it was built against.

    The version fields exist so a caller can check that the prompt it is about to
    send matches the corpus the model was trained on, without diffing strings.
    A mismatch is the silent degradation arch §7 warns about.
    """

    messages: list[dict[str, Any]]
    doc_type: str
    acord_form: str | None
    modality_mode: str
    prompt_template_version: str
    schema_version: str
    prompt_fingerprint: str
    resolution_cap_px: int
    page_count: int
    lob: str | list[str] | None = None
    #: The schema slice this prompt carries (``configs/schema_sections.yaml``),
    #: or ``None`` for the whole schema.
    sections: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def system_prompt(self) -> str:
        return self.messages[0]["content"]

    def assert_matches_corpus(self, corpus_manifest: dict[str, Any]) -> None:
        """Fail if this prompt differs from what the corpus was built with.

        Prompt drift between training and inference degrades a fine-tuned model
        and is invisible in training metrics, because training never sees the
        serving prompt. This is the cheap runtime check for it.

        The schema half reads ``schema_versions`` — the per-selector dict the
        manifest actually writes. It previously read ``schema_version``, singular,
        which no manifest has ever contained: the lookup returned ``None``, the
        guard fell through, and the check reported agreement on every request
        including the ones that disagreed.
        """
        from common.schemas import schema_key

        pinned_prompt = corpus_manifest.get("prompt_template_version")
        pinned_versions = corpus_manifest.get("schema_versions")
        if isinstance(pinned_versions, dict):
            # Keyed exactly as `_schema_pins` writes it: `policy`, `acord:25`,
            # `policy:homeowners`. A row whose selector the corpus never saw has
            # no pin to compare against, which is a coverage question, not drift.
            pinned_schema = pinned_versions.get(
                schema_key(self.doc_type, self.acord_form, self.lob)
            )
        else:
            # A manifest predating the per-selector dict.
            pinned_schema = corpus_manifest.get("schema_version")

        problems = []
        if pinned_prompt and pinned_prompt != self.prompt_template_version:
            problems.append(
                f"prompt template {pinned_prompt} (corpus) vs {self.prompt_template_version} (here)"
            )
        if pinned_schema and pinned_schema != self.schema_version:
            problems.append(f"schema {pinned_schema} (corpus) vs {self.schema_version} (here)")
        if problems:
            raise InputBuilderError(
                "prompt/schema drift between the corpus and this request: "
                + "; ".join(problems)
                + ". Training-time and inference-time prompts must render identically (arch §7) — "
                "a change to either forces a corpus rebuild and a new training cycle."
            )


#: Stands in for a page whose OCR recovered no text, in training AND serving.
#: Serving substituted it and training sent the marker alone, so the model
#: trained on one prompt for a blank page and was served another. An empty
#: string is not an option at serving either: every page is read, a poorly
#: scanned one included, and a request that refused on it would abort exactly
#: the messy documents.
EMPTY_PAGE_TEXT = "(no OCR text recovered for this page — read it from the image)"


def _image_block(image: str | Path | bytes) -> dict[str, Any]:
    """One image content block, in the chat format ms-swift and vLLM expect."""
    if isinstance(image, bytes):
        return {"type": "image", "image": image}
    return {"type": "image", "image": str(image)}


def build_messages(
    doc_type: str,
    image_paths: Sequence[str | Path | bytes],
    ocr_pages: Sequence[str] | None,
    modality_mode: str,
    *,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
    assistant_content: str | None = None,
    check_resolution_parity: bool = True,
    page_numbers: Sequence[int] | None = None,
    total_pages: int | None = None,
) -> BuiltMessages:
    """Assemble the chat-format messages for one document.

    Produces an **interleaved** user turn — image, its text, image, its text —
    so the pairing is positional and needs no inference::

        [img p1][<page 1 of 3> + p1 markdown][img p2][<page 2 of 3> + p2 markdown]…

    Args:
        doc_type: ``acord`` | ``policy`` | ``lossrun``.
        image_paths: page images **in page order**. Order is preserved exactly;
            it carries positional meaning to the model.
        ocr_pages: one markdown string **per image**, same length and order, or
            ``None`` for ``image_only``. A single joined blob is not accepted:
            joining is what destroyed the page correspondence and split every
            table that crossed a page boundary.
        modality_mode: one of :data:`common.constants.MODALITY_MODES`.
        acord_form: required for ACORD — selects the schema (arch §4b).
        lob: a policy's line of business — selects the per-LOB canonical schema.
        sections: the slice of that schema one window asks for — ``decl``,
            ``arrays``, ``lineblk``, ``dtd``. The prompt then describes only what
            the window can answer; ``None`` is the whole schema.
        assistant_content: the golden JSON, when building a *training* row.
            Omitted at inference, where the assistant turn is what gets generated.
        check_resolution_parity: assert the training and serving caps agree.
        page_numbers: the true page numbers of these images. Defaults to
            ``1..n``. A routed request passes the real numbers so the markers
            say ``<page 9 of 20>`` rather than mislabelling page 9 as page 2.
        total_pages: pages in the whole document. Defaults to the number given,
            which is correct only when the whole document is being sent.

    Returns:
        :class:`BuiltMessages` — the messages plus the versions behind them.
    """
    if modality_mode not in MODALITY_MODES:
        raise InputBuilderError(
            f"unknown modality_mode {modality_mode!r}; expected one of {MODALITY_MODES}"
        )
    if not image_paths:
        raise InputBuilderError(
            "at least one page image is required. Every mode is image-bearing — "
            "`image_only` removes the OCR text, not the images."
        )

    if modality_mode == "image_only" and ocr_pages:
        raise InputBuilderError(
            "image_only was requested but OCR text was supplied. Passing OCR into the "
            "image-only pathway would silently train or serve the wrong regime, and the "
            "prompt already declares that no OCR is provided (arch §6)."
        )
    if modality_mode != "image_only" and not ocr_pages:
        raise InputBuilderError(
            f"{modality_mode} requires OCR text. If none is available, use image_only so the "
            "prompt declares the absence explicitly rather than leaving the model to guess."
        )
    if ocr_pages is not None and len(ocr_pages) != len(image_paths):
        raise InputBuilderError(
            f"{len(ocr_pages)} page(s) of OCR text against {len(image_paths)} page image(s). "
            "There must be exactly one text per image: the pairing is what tells the model "
            "which text belongs to which page, and a mismatched list silently shifts every "
            "page's text onto the wrong image."
        )

    numbers = list(page_numbers) if page_numbers is not None else list(range(1, len(image_paths) + 1))
    if len(numbers) != len(image_paths):
        raise InputBuilderError(
            f"{len(numbers)} page number(s) against {len(image_paths)} page image(s)"
        )
    total = total_pages if total_pages is not None else len(image_paths)
    if total < len(image_paths):
        raise InputBuilderError(
            f"total_pages={total} is fewer than the {len(image_paths)} page(s) supplied"
        )

    if check_resolution_parity:
        assert_resolution_parity()
    cap = resolution_cap_px()

    system = render_system_prompt(doc_type, modality_mode, acord_form, lob, sections)

    # Each page's image, then that page's own text. The image comes first
    # because the page is the primary evidence; the text follows immediately so
    # "which text goes with which image" is answered by position rather than by
    # the model matching content across a 40-page document.
    user_content: list[dict[str, Any]] = []
    for index, image in enumerate(image_paths):
        user_content.append(_image_block(image))
        marker = PAGE_MARKER.format(page=numbers[index], total=total)
        if ocr_pages is None:
            # image_only keeps the marker. It is document metadata, not OCR —
            # the mode's guarantee is "no OCR text", not "no text at all" — and
            # without it a routed image_only request cannot tell the model it is
            # holding pages 9 and 14 of 20 rather than a two-page document.
            user_content.append({"type": "text", "text": marker})
        else:
            text = ocr_pages[index] if (ocr_pages[index] or "").strip() else EMPTY_PAGE_TEXT
            user_content.append({"type": "text", "text": marker + "\n\n" + text})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    return BuiltMessages(
        messages=messages,
        doc_type=doc_type,
        acord_form=acord_form,
        modality_mode=modality_mode,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        schema_version=schema_version(doc_type, acord_form, lob, sections),
        prompt_fingerprint=prompt_fingerprint(doc_type, modality_mode, acord_form, lob, sections),
        resolution_cap_px=cap,
        page_count=len(image_paths),
        lob=lob,
        sections=sections,
    )


def build_training_row(
    doc_type: str,
    source_id: str,
    image_paths: Sequence[str | Path],
    ocr_pages: Sequence[str] | None,
    modality_mode: str,
    golden_json: str,
    *,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
    tenant_id: str | None = None,
    split: str | None = None,
    deidentified: bool = False,
    page_numbers: Sequence[int] | None = None,
    total_pages: int | None = None,
) -> dict[str, Any]:
    """One JSONL corpus row (master §9).

    Uses the same :func:`build_messages` the serving path uses — which is what
    makes the training prompt and the inference prompt provably identical rather
    than identical by convention.

    ``page_numbers`` and ``total_pages`` are forwarded, not defaulted. A windowed
    row holds a *subset* of its document's pages, and the markers are how the
    model is told which subset: without them every window claims to be pages
    ``1..n`` of an ``n``-page document, while the template tells the model that
    skipped numbers mean it is seeing selected pages. The model would be trained
    on a page map that contradicts the instruction describing it.
    """
    built = build_messages(
        doc_type, image_paths, ocr_pages, modality_mode,
        acord_form=acord_form, lob=lob, sections=sections, assistant_content=golden_json,
        page_numbers=page_numbers, total_pages=total_pages,
    )
    row = {
        "doc_type": doc_type,
        "acord_form": acord_form,
        # Always a list. The review tool and export_golden_label pass a string,
        # both importers a list; a JSONL column holding both is one pyarrow —
        # and so HF datasets — refuses to load ("changed from string to array").
        "lob": normalize_lob(lob),
        "modality_mode": modality_mode,
        "source_id": source_id,
        "tenant_id": tenant_id,
        "split": split,
        "deidentified": deidentified,
        "messages": built.messages,
    }
    # Only on a windowed row, so a whole-schema row is byte-identical to before.
    if sections:
        row["sections"] = sections
    return row


def page_images_for(
    ocr_meta: dict[str, Any],
    path_builder,
    doc_type: str,
    source_id: str,
    tenant_id: str | None = None,
    pages: Sequence[int] | None = None,
) -> list[str]:
    """Resolve ordered page-image paths from a document's ``ocr_meta.json``.

    ``pages`` selects a subset — the page-routing path for long documents
    (arch §7). The returned list is always sorted by page number, because
    handing the model pages out of order changes what it sees.
    """
    page_count = int(ocr_meta.get("page_count", 0))
    if page_count < 1:
        raise InputBuilderError(f"{source_id} has no processed pages")

    selected = sorted(set(pages)) if pages else list(range(1, page_count + 1))
    for page in selected:
        if not 1 <= page <= page_count:
            raise InputBuilderError(
                f"page {page} out of range for {source_id} (1..{page_count})"
            )
    return [path_builder(doc_type, source_id, p, "png", tenant_id) for p in selected]
