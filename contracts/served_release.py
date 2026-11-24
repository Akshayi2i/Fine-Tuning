"""How the pod's vLLM names a release's adapter, and the OCR pin a release must share with serving
(review A1: moved from ``scripts.served_adapter``, which re-exports them). The Azure worker's cold start checks a
release with them."""

from __future__ import annotations

from typing import Any


class ServedAdapterError(RuntimeError):
    """The release cannot be served as a LoRA on the pod's vLLM."""

def lora_name(release_id: str) -> str:
    """The name vLLM serves the release's adapter as, and the routine asks for."""
    return f"policy-{release_id}"

def assert_ocr_pin(release_pin: dict[str, Any], pin: dict[str, str]) -> None:
    """Refuse a release whose corpus another MinerU read than the one serving reads
    with: another version, or another tier where the corpus recorded one."""
    trained = release_pin.get("mineru_version")
    if trained != pin["version"]:
        raise ServedAdapterError(
            f"the release was trained on text from MinerU {trained or '(not recorded)'}, but serving reads "
            f"pages with MinerU {pin['version']} (configs/shared/ocr.yaml): its text is formatted otherwise, "
            "input the model never saw. Re-OCR the corpus with the served MinerU and train a new version."
        )
    # The tier, where the corpus recorded one: a reading at another tier was made
    # by other models. An in-process reading records none and is held to its
    # version alone, as find_unprocessed holds it.
    tier = release_pin.get("mineru_tier")
    if tier is not None and tier != pin["tier"]:
        raise ServedAdapterError(
            f"the release was trained on text from MinerU {trained} at tier {tier}, but serving reads pages "
            f"at tier {pin['tier']} (configs/shared/ocr.yaml): another tier reads a page with other models. "
            "Re-OCR the corpus with the served MinerU and train a new version."
        )
    # Whether every page was read by OCR, where the corpus recorded it: a model
    # trained on a hybrid PDF's text layer reads OCR text it never saw, and the
    # other way round.
    every_page = release_pin.get("read_every_page_by_ocr")
    if every_page is not None:
        from common.config import ocr_every_page

        serving = pin.get("read_every_page_by_ocr", ocr_every_page())
        if every_page != serving:
            raise ServedAdapterError(
                f"the release was trained on text read with read_every_page_by_ocr {every_page}, but serving "
                f"reads with {serving} (configs/shared/ocr.yaml): a PDF with a text layer is read another "
                "way. Re-OCR the corpus with the served setting and train a new version."
            )
