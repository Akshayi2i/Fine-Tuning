"""MinerU kept loaded in the OCR environment, for a process in the other one.

MinerU lives in ``/workspace/venv-ocr`` and vLLM in ``/workspace/venv``, and the
two cannot share a process (README, "two environments"). The extraction test
server (``testing/extract_server.py``) starts this once, in the OCR environment,
and hands it one PDF at a time on stdin. MinerU loads its models on the first
document and keeps them, so a document's OCR time is its own, not a model load's.

Protocol, one JSON object per line::

    in:   {"pdf": "/path/doc.pdf", "out": "/path/pages", "force_ocr": false}
    out:  OCR-RESULT {"ok": true, "pages": 85, "seconds": 41.2, "modality": "scanned_pdf", ...}
          OCR-RESULT {"ok": false, "error": "..."}

Anything else on stdout (MinerU's own printing) is not a result: the reader
skips every line without the prefix. Pages are written as ``page_001.png`` and
``page_001.md`` into ``out``: the image at the configured resolution cap, and
that page's MinerU markdown - the same page text and pixels the corpus build
and serving read.

Started by the server only (``python -c``), never from a terminal: it has no
``__main__`` of its own.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

#: Marks the one line per request that is a result.
RESULT_PREFIX = "OCR-RESULT "


def ocr_pdf(pdf: Path, out: Path, engine: Any, *, force_ocr: bool = False) -> dict[str, Any]:
    """OCR ``pdf`` into ``out``, one image and one markdown file per page."""
    from common.config import resolution_cap_px

    started = time.perf_counter()
    pages = engine.process(pdf.read_bytes(), device="cuda", max_long_side_px=resolution_cap_px(),
                           force_ocr=force_ocr or None)
    out.mkdir(parents=True, exist_ok=True)
    for page in pages:
        stem = f"page_{page.page_number:03d}"
        (out / f"{stem}.png").write_bytes(page.image_bytes)
        (out / f"{stem}.md").write_text(page.markdown, encoding="utf-8")
    scanned = sum(1 for page in pages if page.scanned)
    return {
        "ok": True,
        "pages": len(pages),
        "seconds": round(time.perf_counter() - started, 2),
        # The modalities the served document records (common.canonical.DOCUMENT_MODALITIES).
        "modality": "scanned_pdf" if scanned else "native_pdf",
        "scanned_pages": scanned,
        "read_by_ocr": any(page.read_by_ocr for page in pages),
        "ocr_failed_pages": [page.page_number for page in pages if page.ocr_failed],
    }


def serve(lines: TextIO = sys.stdin, out: TextIO = sys.stdout, engine: Any = None) -> int:
    """Answer one request per input line until the input closes."""
    for line in lines:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if engine is None:
                from data_pipeline.ocr.run_mineru import MinerUEngine

                engine = MinerUEngine()
            result = ocr_pdf(Path(request["pdf"]), Path(request["out"]), engine,
                             force_ocr=bool(request.get("force_ocr")))
        except Exception as exc:  # noqa: BLE001 - every failure is the caller's to report
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out.write(RESULT_PREFIX + json.dumps(result) + "\n")
        out.flush()
    return 0
