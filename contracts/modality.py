"""How a PDF's pages carry their text (review A1: moved from ``data_pipeline.ocr.modality``,
which re-exports these and decides a document's modality)."""

from __future__ import annotations

#: A born-digital PDF: its pages carry a text layer.
NATIVE = "native_pdf"
#: A scanned PDF: its pages are images only.
SCANNED = "scanned_pdf"
MODALITIES = (NATIVE, SCANNED)
