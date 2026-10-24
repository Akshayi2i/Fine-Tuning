"""MinerU OCR and page rendering, GPU-accelerated (arch §14, §8a).

Reads ``raw-documents/``, writes ``processed/``. Never writes back to the raw
layer, and nothing downstream of here touches raw documents again.

Two things are non-negotiable:

**The resolution cap comes from config and is recorded.** Image token count is a
direct function of page resolution — the single biggest cost and latency lever
(arch §11) — and the cap must be *identical* here and at production inference. A
mismatch means serving the model a distribution it never trained on, which shows
up only as degraded accuracy with no error anywhere.

**The MinerU version and device are recorded on every document.** The model
learns how MinerU formats its output, so both are part of the corpus pin
(:mod:`~data_pipeline.ocr.mineru_version`).

The MinerU invocation is deliberately behind a swappable interface: it may run as
a library call or a subprocess, and the Phase 0 spike may replace it outright.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, for_ocr
from common.config import resolution_cap_px
from common.constants import ACTIVE_DOC_TYPES
from data_pipeline.ocr.mineru_version import Device, current_environment, get_mineru_version

log = logging.getLogger(__name__)


class OcrError(RuntimeError):
    """Raised when a document cannot be processed."""


@dataclass
class PageOutput:
    """One page's OCR text and rendered image."""

    page_number: int          # 1-based; ordering is positionally meaningful
    markdown: str
    image_bytes: bytes
    table_row_count: int = 0  # feeds the row-completeness signal (IMPL-09)
    ocr_failed: bool = False
    #: The page has no text layer (a scan). A document with any such page is
    #: OCR'd whole and recorded as ``is_scanned``.
    scanned: bool = False
    #: MinerU read the document in OCR mode (a scan, a mixed document, drawn
    #: figures, or the caller's force_ocr), not from its text layer.
    read_by_ocr: bool = False


class OcrEngine(Protocol):
    """The swappable OCR interface.

    Implemented by :class:`MinerUEngine` in production and by a stub in tests, so
    the whole pipeline is verifiable without MinerU installed.
    """

    def process(self, pdf_bytes: bytes, *, device: Device, max_long_side_px: int,
                modality: str | None = None, force_ocr: bool | None = None) -> list[PageOutput]:
        ...


#: The OCR language MinerU reads scans in. ``ch`` is its PP-OCRv6 small model pair,
#: which covers English (there is no separate English model since PP-OCRv5);
#: pinned rather than left to MinerU's default, since the OCR text is model input.
OCR_LANG = "ch"


class MinerUEngine:
    """Real MinerU 3.x (``mineru``, pipeline backend). Imported lazily so CI needs
    neither MinerU nor CUDA.

    MinerU does the reading; this class does the two things around it that the
    corpus depends on:

    * **one markdown string per page**, built from MinerU's content list
      (:func:`pages_from_content_list`). MinerU's ``get_markdown`` returns one blob
      for the whole document, and a joined blob cannot be split back into pages —
      which is what every training row and serving request is keyed on;
    * **page images from the shared renderer** (:func:`render_only.render_pdf_pages`)
      at the configured cap, so an OCR'd document and an image-only one are the
      same pixels.

    Checked on the pod by the Phase 0 spike (``check_mineru_gpu``): the GPU path,
    the pinned model weights (``mineru.json``) and the formatting of real documents.
    Formula recognition stays off: policies carry no equations.
    """

    def __init__(self, device: Device = "cuda") -> None:
        self.device = device

    def process(self, pdf_bytes: bytes, *, device: Device, max_long_side_px: int,
                modality: str | None = None, force_ocr: bool | None = None) -> list[PageOutput]:
        """``modality`` and ``force_ocr`` come from the caller in serving (Fideon
        SPEC_05 §4) and are trusted as given; the offline tools pass neither, and
        the document is inspected instead (:mod:`data_pipeline.ocr.modality`)."""
        if device != "cuda":
            raise OcrError(f"MinerU runs on the GPU only; device {device!r} is refused")
        from common.gpu import GPUError, require_cuda
        from data_pipeline.ocr.mineru_config import MinerUConfigError, apply_run_environment, assert_on_cuda
        from data_pipeline.ocr.render_only import render_pdf_pages

        try:
            require_cuda("MinerU OCR")
            # The GPU existing is not MinerU using it, and MinerU left alone
            # fetches the latest weights: both are set before it is imported.
            apply_run_environment()
            assert_on_cuda()
        except (GPUError, MinerUConfigError) as exc:
            raise OcrError(str(exc)) from exc
        try:
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
            from mineru.data.data_reader_writer import FileBasedDataWriter
            from mineru.utils.enum_class import MakeMode
            from mineru.utils.pdf_classify import classify
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            # The import's own error first: MinerU installed but one of the
            # libraries it loads failing (OpenCV without the system's libGL) read
            # as "not installed", and the cause was nowhere in the message.
            raise OcrError(
                f"MinerU 3.x could not be imported: {type(exc).__name__}: {exc}. If MinerU is not "
                "installed, install the OCR group on the pod: bash scripts/setup_pod.sh ocr "
                "(mineru[pipeline]), and download its pinned model weights: "
                "bash scripts/download_mineru_models.sh"
            ) from exc

        from data_pipeline.ocr.modality import ModalityError, reads_by_ocr

        try:
            ocr = reads_by_ocr(pdf_bytes, modality=modality, force_ocr=force_ocr)
        except ModalityError as exc:
            raise OcrError(str(exc)) from exc

        images = render_pdf_pages(pdf_bytes, max_long_side_px)
        text_layer = text_layer_pages(pdf_bytes)
        if not ocr and modality is not None and not all(text_layer):
            # Trusted as given (SPEC_05 §4), but said: these pages read as blank.
            log.warning("the caller sent a document as %s, but page(s) %s have no text layer; they "
                        "are read in text mode and flagged as unread", modality,
                        [i + 1 for i, has in enumerate(text_layer) if not has])
        import json
        import tempfile

        # Text mode reads only a text layer. A MIXED document — typed
        # declarations, scanned endorsements — classified as text left its
        # scanned pages unread and unflagged, trained as blank. So any page
        # without a text layer sends the whole document through OCR, and so
        # does a native one whose figures are drawn (reads_by_ocr). Offline,
        # MinerU's own classification is heard too; a caller's modality is not
        # second-guessed.
        scanned = ocr or (modality is None and classify(pdf_bytes) == "ocr")
        read: dict[int, dict[str, Any]] = {}

        def on_doc_ready(doc_index: int, _model_list: Any, middle_json: dict[str, Any], _ocr: bool) -> None:
            read[doc_index] = middle_json

        with tempfile.TemporaryDirectory() as tmp:  # pragma: no cover - needs MinerU and a GPU
            # Figures MinerU crops are written here and dropped: the page image
            # the model sees is the shared renderer's, not MinerU's crops.
            doc_analyze_streaming([pdf_bytes], [FileBasedDataWriter(tmp)], [OCR_LANG], on_doc_ready,
                                  parse_method="ocr" if scanned else "txt",
                                  formula_enable=False, table_enable=True)
            if 0 not in read:
                raise OcrError("MinerU returned no reading of the document")
            content = union_make(read[0]["pdf_info"], MakeMode.CONTENT_LIST, "images")
            if isinstance(content, str):
                content = json.loads(content)

        texts = pages_from_content_list(content, len(images))
        return [
            PageOutput(
                page_number=index + 1,
                markdown=text,
                image_bytes=image,
                table_row_count=count_table_rows(text),
                # Nothing read on a SCANNED page is an OCR failure; on a text-layer
                # page it is a blank or picture-only page, which is not. Either
                # way the page is kept and built with the blank-page placeholder.
                # A page with no text layer read in text mode (the caller said
                # native) yields nothing either, and is flagged the same way.
                ocr_failed=(scanned or not text_layer[index]) and not text.strip(),
                scanned=not text_layer[index],
                read_by_ocr=scanned,
            )
            for index, (text, image) in enumerate(zip(texts, images, strict=True))
        ]


#: Characters of embedded text below which a page counts as having no text layer.
TEXT_LAYER_MIN_CHARS = 30

#: Text render modes that draw nothing on the page: 3 is invisible text - the
#: layer a scanner's OCR lays under a page image to make it searchable - and 7
#: adds the text to a clipping path only.
INVISIBLE_TEXT_MODES = (3, 7)

#: Versions of the rules that decide how a document is read; recorded in its OCR
#: meta so a stored reading made under older rules is checked again
#: (_needs_ocr_now). 1: drawn figures send a native PDF through OCR. 2: only
#: visible text counts as a text layer, so a searchable scan is a scan.
OCR_RULES_VERSION = 2


def visible_text_chars(page: Any) -> int:
    """Characters of text drawn visibly on a page, not counting an invisible
    OCR layer or fully transparent text."""
    return sum(
        len(span.get("chars", ()))
        for span in page.get_texttrace()
        if span.get("type") not in INVISIBLE_TEXT_MODES and span.get("opacity", 1) > 0
    )


def text_layer_pages(pdf_bytes: bytes) -> list[bool]:
    """Per page, whether it is a digital page rather than a scan.

    A page with visible text is digital. Only visible text counts: a scan made
    searchable is a page image with the scanner's OCR text laid invisibly under
    it, and counting that layer called it digital - its own OCR was read
    instead of MinerU's, and it was recorded as not scanned. An EMPTY page - no
    text, no image, no drawing - is digital too: there is nothing on it to OCR.
    Counting it as a scan sent a whole digital policy with one blank page
    through OCR mode and into the scanned eval subset. A page with no visible
    text but an image or drawings on it is a scan (or might be), and is read by
    OCR.
    """
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [
            visible_text_chars(page) >= TEXT_LAYER_MIN_CHARS
            or not (page.get_images() or page.get_drawings())
            for page in doc
        ]


#: Content-list blocks at a page's edges. MinerU 3.x lists them after the page's
#: body; a page's header is put back at its head and the rest at its foot, so the
#: text reads top to bottom. MinerU 1.x dropped them: a form's number, printed in
#: its footer, was in no page's text.
_PAGE_HEAD = ("header",)
_PAGE_FOOT = ("footer", "page_number", "page_footnote", "aside_text")


def pages_from_content_list(blocks: Iterable[dict[str, Any]], page_count: int) -> list[str]:
    """MinerU's content list as one markdown string per page, in reading order.

    MinerU emits blocks each tagged with its 0-based ``page_idx``. Rendered here
    the same way for every document, so training rows and serving requests see
    one formatting:

    * text — a heading gets ``#`` per ``text_level``, body text as is;
    * list — its items, one per line;
    * table — captions, then the table as MinerU gives it (HTML), then footnotes;
    * image, chart — captions and footnotes only (the picture is in the page image);
    * code — captions, its text, footnotes;
    * equation — its text (LaTeX);
    * header — at the head of its page; footer, page number, page footnote and
      margin text — at its foot.
    """
    head: list[list[str]] = [[] for _ in range(page_count)]
    body: list[list[str]] = [[] for _ in range(page_count)]
    foot: list[list[str]] = [[] for _ in range(page_count)]

    def texts(block: dict[str, Any], *keys: str) -> list[str]:
        return [str(item).strip() for key in keys for item in block.get(key) or []]

    for block in blocks:
        index = block.get("page_idx")
        if not isinstance(index, int) or not 0 <= index < page_count:
            continue
        kind = block.get("type")
        parts: list[str] = []
        if kind == "text":
            text = str(block.get("text") or "").strip()
            level = block.get("text_level")
            if text and isinstance(level, int) and level > 0:
                text = "#" * min(level, 6) + " " + text
            parts.append(text)
        elif kind == "list":
            parts.append("\n".join(str(item).strip() for item in block.get("list_items") or []))
        elif kind == "table":
            parts += texts(block, "table_caption")
            parts.append(str(block.get("table_body") or "").strip())
            parts += texts(block, "table_footnote")
        elif kind == "image":
            # image_caption in MinerU 3.x, img_caption in 1.x.
            parts += texts(block, "image_caption", "img_caption")
            parts += texts(block, "image_footnote", "img_footnote")
        elif kind == "chart":
            parts += texts(block, "chart_caption") + texts(block, "chart_footnote")
        elif kind == "code":
            parts += texts(block, "code_caption")
            parts.append(str(block.get("code_body") or "").strip())
            parts += texts(block, "code_footnote")
        else:
            parts.append(str(block.get("text") or "").strip())
        target = head if kind in _PAGE_HEAD else foot if kind in _PAGE_FOOT else body
        target[index] += [part for part in parts if part]
    return ["\n\n".join(head[i] + body[i] + foot[i]) for i in range(page_count)]


def count_table_rows(markdown: str) -> int:
    """Count table body rows — markdown pipe tables and MinerU's HTML tables.

    Consumed by the list-completeness cross-check (IMPL-09): if the model extracts
    six claims from a page MinerU saw eight rows on, that is a recall failure the
    per-field confidence cannot see, because the missing rows generate no tokens.

    MinerU writes tables as HTML. Counting pipe tables alone read every MinerU
    page as having no rows, and the row-completeness signal never fired.
    """
    return _count_pipe_rows(markdown) + _count_html_rows(markdown)


def _count_html_rows(text: str) -> int:
    """Body rows of every HTML table, nested tables included, each counted once.

    A row is a header only when it has header cells and no data cells: a schedule
    whose every row starts with ``<th>Vehicle 1</th>`` and continues in ``<td>``
    is all body rows. A table with no header row treats its first row as the
    header, as a pipe table does. Parsed, not matched by regex: a non-greedy
    ``<table>.*?</table>`` stopped at a nested table's end and cut the outer one short.
    """
    from html.parser import HTMLParser

    class _Rows(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.stack: list[list[tuple[bool, bool]]] = []   # per open table: (has_th, has_td) per row
            self.row: list[list[bool]] = []                   # per open table: current row flags
            self.body = 0

        def handle_starttag(self, tag, attrs):
            if tag == "table":
                self.stack.append([])
                self.row.append([False, False])
            elif self.stack and tag == "tr":
                self.row[-1] = [False, False]
            elif self.stack and tag in ("th", "td"):
                self.row[-1][0 if tag == "th" else 1] = True

        def handle_endtag(self, tag):
            if not self.stack:
                return
            if tag == "tr":
                self.stack[-1].append(tuple(self.row[-1]))
            elif tag == "table":
                rows = self.stack.pop()
                self.row.pop()
                headers = sum(1 for has_th, has_td in rows if has_th and not has_td)
                self.body += max(0, len(rows) - (headers or (1 if rows else 0)))

    parser = _Rows()
    parser.feed(text)
    parser.close()
    return parser.body


def _count_pipe_rows(markdown: str) -> int:
    rows = 0
    tables = 0
    in_table = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            in_table = False          # a non-table line ends the current table
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells if c):   # separator row
            continue
        if not in_table:
            # The first row of a run is that table's header.
            in_table = True
            tables += 1
        rows += 1
    # One header per table, not per page. Discounting a single header for the
    # whole page over-reported a two-table page by one row, which raises a false
    # row-completeness flag — and active learning treats that flag as an
    # unconditional override to full manual review.
    return max(0, rows - tables)


def process_document(
    doc_type: str,
    source_id: str,
    client: BlobClient,
    engine: OcrEngine,
    *,
    tenant_id: str | None = None,
    device: Device | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """OCR and render one document into ``processed/``.

    Idempotent: skips when output already exists **and** the source checksum,
    MinerU version, and device all still match - and, read by a served MinerU,
    its tier. A change to any of them forces reprocessing, because each one
    changes the input distribution the model was trained to arbitrate against.
    """
    # A served MinerU says what it is (the pin it checks every task against);
    # the in-process one is the version installed here.
    env = (engine.environment() if hasattr(engine, "environment")
           else current_environment(device, strict=device == "cuda"))
    # A served MinerU's tier is part of its pin (configs/shared/ocr.yaml): a
    # reading at another tier was made by other models, another distribution.
    tier = (engine.meta() if hasattr(engine, "meta") else {}).get("mineru_tier")
    cap = resolution_cap_px()
    meta_key = paths.ocr_meta(doc_type, source_id, tenant_id)

    if not force and client.exists(meta_key):
        existing = client.read_json(meta_key)
        raw_meta = client.read_json(paths.raw_metadata(doc_type, source_id, tenant_id))
        unchanged = (
            # A render-only meta is written by render_only.py at this same key
            # and satisfies every check below, so a document rendered first was
            # skipped by OCR forever — no markdown was ever produced, and the
            # CLI reported it as processed.
            not existing.get("render_only")
            and existing.get("source_checksum") == raw_meta.get("checksum_sha256")
            and existing.get("mineru_version") == env.mineru_version
            and existing.get("ocr_device") == env.device
            and existing.get("resolution_cap_px") == cap
            and (tier is None or existing.get("mineru_tier") == tier)
        )
        if unchanged:
            redo = _needs_ocr_now(existing, client, doc_type, source_id, tenant_id)
            if not redo:
                if redo is False and _rules_of(existing) < OCR_RULES_VERSION:
                    # Re-checked against the current rules and it stands: recorded,
                    # so the next run does not fetch the PDF to check it again.
                    existing = {**existing, "drawn_check": True, "ocr_rules": OCR_RULES_VERSION}
                    client.write_json(meta_key, existing)
                log.info("skipping %s: already processed and unchanged", source_id)
                return existing

    raw_meta = client.read_json(paths.raw_metadata(doc_type, source_id, tenant_id))
    pdf_bytes = client.read_bytes(paths.raw_pdf(doc_type, source_id, tenant_id))

    try:
        pages = engine.process(pdf_bytes, device=env.device, max_long_side_px=cap)
    except Exception as exc:
        raise OcrError(f"OCR failed for {source_id}: {exc}") from exc
    if not pages:
        raise OcrError(f"OCR produced no pages for {source_id}")

    for page in sorted(pages, key=lambda p: p.page_number):
        client.write_bytes(
            paths.processed_page(doc_type, source_id, page.page_number, "png", tenant_id),
            page.image_bytes,
        )
        client.write_text(
            paths.processed_page(doc_type, source_id, page.page_number, "md", tenant_id),
            page.markdown,
        )

    ocr_meta = {
        "source_id": source_id,
        "doc_type": doc_type,
        "tenant_id": paths._tenant(tenant_id),
        "page_count": len(pages),
        "mineru_version": env.mineru_version,
        "ocr_device": env.device,
        "gpu_name": env.gpu_name,
        "preprocessing_date": datetime.now(UTC).isoformat(),
        "resolution_cap_px": cap,
        "source_checksum": raw_meta.get("checksum_sha256"),
        "table_row_counts": {str(p.page_number): p.table_row_count for p in pages},
        "failed_pages": [p.page_number for p in pages if p.ocr_failed],
        # From MinerU's classification, not from failed pages: a text PDF with one
        # blank page was recorded as a scan, and counted in the scanned gate.
        "is_scanned": any(p.scanned for p in pages),
        # Read in OCR mode, and under which version of the reading rules: an
        # older reading of a document the current rules send through OCR is
        # redone (_needs_ocr_now).
        "read_by_ocr": any(p.read_by_ocr for p in pages),
        "drawn_check": True,
        "ocr_rules": OCR_RULES_VERSION,
        # How it was read: the served API and its tier, when it was (ServedMinerUEngine).
        **(engine.meta() if hasattr(engine, "meta") else {}),
    }
    client.write_json(meta_key, ocr_meta)
    # The marker distinguishes a real OCR pass from the stored metadata returned
    # on a skip, so it is added to the RETURNED copy only. Writing it into blob
    # meant the skip path read it straight back and process_batch filed every
    # skipped document under "processed" — a rerun that did no OCR at all still
    # reported "N processed, 0 skipped".
    ocr_meta = {**ocr_meta, "_reprocessed": True}
    log.info("processed %s: %d pages on %s at %dpx", source_id, len(pages), env.device, cap)
    return ocr_meta


def _rules_of(meta: dict[str, Any]) -> int:
    """The OCR_RULES_VERSION a stored reading was made under."""
    return int(meta.get("ocr_rules", 1 if meta.get("drawn_check") else 0))


def _needs_ocr_now(existing: dict[str, Any], client: BlobClient, doc_type: str, source_id: str,
                   tenant_id: str | None) -> bool | None:
    """Whether a stored reading was made from the text layer under older reading
    rules (OCR_RULES_VERSION) that the current rules send through OCR: a native
    PDF whose figures are drawn (SPEC_05 §4.1), or a searchable scan whose
    invisible OCR layer passed for a text layer. Only those are redone - OCR is
    the expensive part.

    False when the reading stands under the current rules - made under them, or
    re-checked against them just now; None when it is kept unchecked (read by
    OCR, or a check could not run), so a later run asks again."""
    rules = _rules_of(existing)
    if rules >= OCR_RULES_VERSION:
        return False
    if existing.get("read_by_ocr"):
        return None
    from data_pipeline.ocr import modality

    try:
        pdf = client.read_bytes(paths.raw_pdf(doc_type, source_id, tenant_id))
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF is OCR's to report, not the skip check's
        log.warning("%s: could not re-check how it is read (%s); keeping the stored reading", source_id, exc)
        return None
    checked = True
    for reason, check in (("it is a scan (no visible text layer)", lambda: modality.detect_modality(pdf)
                           == modality.SCANNED and not existing.get("is_scanned")),
                          ("its figures are drawn", lambda: rules < 1 and modality.data_is_drawn(pdf))):
        try:
            if check():
                log.info("%s: read from its text layer under older rules, but %s; reading it again "
                         "through OCR", source_id, reason)
                return True
        except Exception as exc:  # noqa: BLE001 - a check that cannot run keeps the stored reading
            log.warning("%s: re-check failed (%s); keeping the stored reading", source_id, exc)
            checked = False
    return False if checked else None


def process_batch(
    source_ids: Iterable[str],
    doc_type: str,
    client: BlobClient,
    engine: OcrEngine,
    *,
    tenant_id: str | None = None,
    device: Device | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Process many documents. One failure never stops the batch."""
    processed: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    for source_id in source_ids:
        try:
            meta = process_document(
                doc_type, source_id, client, engine,
                tenant_id=tenant_id, device=device, force=force,
            )
            # `process_document` returns the stored meta unchanged when it
            # skips, so there is no marker inside it to read — the previous
            # check looked for a `_skipped` key nothing ever wrote, and a run
            # that did no OCR at all reported every document as processed.
            (skipped if meta.get("_reprocessed") is not True else processed).append(source_id)
        except OcrError as exc:
            failed.append((source_id, str(exc)))
            log.warning("OCR failed for %s: %s", source_id, exc)
    return {"processed": processed, "skipped": skipped, "failed": failed}


def find_unprocessed(client: BlobClient, doc_type: str, tenant_id: str | None = None,
                     mineru_version: str | None = None, mineru_tier: str | None = None) -> list[str]:
    """Ingested documents with no ``ocr_meta.json`` yet - and, given the running
    ``mineru_version``, those read by another MinerU: the model learns how MinerU
    formats its output, so a tenant read by two versions is two distributions,
    and ``--all-unprocessed`` reads them again.

    A reading by a served MinerU records its tier, part of the pin: one made at
    another tier than ``mineru_tier`` - by default, given a version, the pinned
    tier (``configs/shared/ocr.yaml``) - is listed too. A reading that records
    no tier (MinerU in-process) is held to its version alone."""
    tenant = paths._tenant(tenant_id)
    ingested = {
        key.split("/")[3]
        for key in client.list(f"raw-documents/{tenant}/{doc_type}/")
        if key.endswith("metadata.json")
    }
    # A render-only ocr_meta.json lives at the same key but records no OCR at
    # all. `process_document` already refuses to skip those; counting them as
    # done here filtered them out of the work list, so `--all-unprocessed`
    # printed "nothing to process" and no page_*.md was ever produced for a
    # document that had only been rendered.
    done = set()
    for key in client.list(f"processed/{tenant}/{doc_type}/"):
        if not key.endswith("ocr_meta.json"):
            continue
        try:
            meta = client.read_json(key)
        except Exception:  # noqa: BLE001 - unreadable metadata is not "done"
            continue
        if meta.get("render_only"):
            continue
        if mineru_version and meta.get("mineru_version") != mineru_version:
            continue
        recorded_tier = meta.get("mineru_tier")
        if (mineru_version or mineru_tier) and recorded_tier is not None:
            if mineru_tier is None:
                from common.config import served_mineru

                mineru_tier = served_mineru()["tier"]
            if recorded_tier != mineru_tier:
                continue
        done.add(key.split("/")[3])
    return sorted(ingested - done)


def benchmark_gpu(
    pdf_bytes: bytes,
    engine_factory: Callable[[Device], OcrEngine],
    *,
    max_long_side_px: int | None = None,
) -> dict[str, Any]:
    """Time one document on GPU, for pod sizing and cost per document.

    There is no CPU comparison to make. MinerU's CPU path uses lighter model
    variants and produces different markdown from the same PDF, so a corpus
    spanning both devices is built from two distributions (arch §8a) — the
    choice was never speed against cost.
    """
    import time

    cap = max_long_side_px or resolution_cap_px()
    started = time.perf_counter()
    pages = engine_factory("cuda").process(pdf_bytes, device="cuda", max_long_side_px=cap)
    seconds = time.perf_counter() - started

    return {
        "resolution_cap_px": cap,
        "device": "cuda",
        "seconds": round(seconds, 2),
        "pages": len(pages),
        "seconds_per_page": round(seconds / len(pages), 2) if pages else None,
    }

def shard_of(source_ids: Iterable[str], shard: str) -> list[str]:
    """Share ``I/N`` of ``source_ids``: every N-th document, from the I-th, in id order.

    N processes with I = 0..N-1 cover every document exactly once - one MinerU
    per GPU instead of one GPU working while the others wait. A document's
    shard depends on its id alone, so the split holds whenever each process
    lists the documents and however many are already done.
    """
    try:
        index, count = (int(part) for part in shard.split("/"))
    except ValueError as exc:
        raise ValueError(f"--shard takes I/N, e.g. 0/4, not {shard!r}") from exc
    if not 0 <= index < count:
        raise ValueError(f"--shard {shard!r}: I must be 0..{count - 1}")
    # By the id itself, not by its place in the list: the list of documents
    # still to do shrinks as they finish, so a shard restarted alone - or one
    # that listed a moment later - took a different slice, and documents fell
    # into no shard or into two.
    import zlib

    return [sid for sid in sorted(source_ids) if zlib.crc32(sid.encode("utf-8")) % count == index]


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Run MinerU OCR + page rendering (GPU by default)")
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-ids", nargs="+")
    group.add_argument("--all-unprocessed", action="store_true")
    # No --device. OCR is GPU-only: MinerU's CPU path uses lighter model variants
    # and produces different markdown from the same PDF, so a corpus spanning
    # both devices is built from two distributions (arch §8a). An option that
    # accepts a single value is a way to be surprised later, not a choice.
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--mineru-url", default=os.environ.get("MINERU_URL"),
                        help="read with the served MinerU at this URL (the pin in configs/shared/ocr.yaml; "
                             "token $MINERU_TOKEN), not the one installed here (default $MINERU_URL)")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="process only this share of the documents (0-based I of N), so N "
                             "processes - one per GPU, each with its own CUDA_VISIBLE_DEVICES - "
                             "split one batch between them")
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('data_pipeline.ocr.run_mineru', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = for_ocr()
    if args.mineru_url:
        from data_pipeline.ocr.served_ocr import MINERU_TOKEN_ENV, ServedMinerUEngine

        served = ServedMinerUEngine(
            args.mineru_url, os.environ.get(MINERU_TOKEN_ENV) or os.environ.get("FIDEON_REMOTE_TOKEN"))
        engine: OcrEngine = served
        version, tier = served.environment().mineru_version, served.tier
    else:
        engine, version, tier = MinerUEngine(device="cuda"), get_mineru_version(), None

    source_ids = (
        find_unprocessed(client, args.doc_type, args.tenant, mineru_version=version, mineru_tier=tier)
        if args.all_unprocessed else args.source_ids
    )
    if args.shard:
        source_ids = shard_of(source_ids, args.shard)
    if not source_ids:
        print("nothing to process")
        return 0

    result = process_batch(
        source_ids, args.doc_type, client, engine,
        tenant_id=args.tenant, force=args.force_reprocess,
    )
    print(f"{len(result['processed'])} processed, {len(result['skipped'])} skipped, "
          f"{len(result['failed'])} failed")
    for source_id, reason in result["failed"]:
        print(f"  FAILED {source_id}: {reason}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
