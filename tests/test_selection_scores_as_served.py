"""Selection and the training-documents score score an answer as production serves it (2026-11-10 review, finding 2).

The gate reads through serving.pipeline.extract, which holds every common-model answer to the label conventions
(serving.conventions.conform: an edition "0413" becomes "04/13", an edition moves out of a form number) and to the
printed values (strict: a value no page prints is taken out). Selection's sample ranking, its document score and
test 2 scored the raw answer, so an answer production writes correctly failed there.
"""

from __future__ import annotations

import json

from evaluation.run_eval import BORROWED_TEXT, PRINTED_CHECK, PRINTED_TEXT
from evaluation.validation_generation import (
    ValidationGeneration,
    as_served,
    merged_windows,
    score_generations,
    served_documents,
    served_window_answer,
)
from inference_core.input_builder import build_training_row

LOB = "gl"
#: The edition printed run together after the number, as the answer writes it ("0413"); the rule writes "04/13".
PAGE = "FORMS SCHEDULE\nCG 00 01 0413 Commercial General Liability Coverage Form\nPolicy Number GL-1001"


def _env(value, *pages):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


def _row(mode: str, group: str = "forms", *, text: str = PAGE) -> dict:
    texts = None if mode == "image_only" else [text]
    row = build_training_row("policy", "policy_0001", ["processed/p/page_1.png"], texts, mode, json.dumps({}),
                             lob=LOB, sections=group, split="val", page_numbers=[1], total_pages=1)
    row.update(window_pages=[1], window_index=0, window_leads=True, task="policy_schedule")
    return row


GOLD = {"forms_and_endorsements": [{"form_number": _env("CG 00 01", 1), "edition_date": _env("04/13", 1)}]}
#: The model wrote the edition as the page prints it, not as the rule writes it.
ANSWER = {"forms_and_endorsements": [{"form_number": _env("CG 00 01", 1),
                                      "edition_date": {"raw": "0413", "parsed": "0413", "page_ref": [1]}}]}


def _window(mode: str, answer=ANSWER) -> ValidationGeneration:
    return ValidationGeneration(row=_row(mode), golden=GOLD, extraction=answer)


def test_a_window_answer_is_held_to_the_edition_rule_as_production_writes_it():
    served = served_window_answer(_window("ocr_plus_image"))
    assert served["forms_and_endorsements"][0]["edition_date"]["parsed"] == "04/13"
    assert ANSWER["forms_and_endorsements"][0]["edition_date"]["parsed"] == "0413"      # the raw answer is untouched


def test_an_edition_written_as_printed_scores_as_correct_as_the_gate_scores_it():
    metrics = score_generations([_window("ocr_plus_image")])
    assert metrics["field_normalized_match"] == 1.0
    assert metrics["document_field_normalized_match"] == 1.0


def test_a_value_no_page_prints_is_taken_out_before_the_document_is_scored():
    invented = {"forms_and_endorsements": [*ANSWER["forms_and_endorsements"],
                                           {"form_number": _env("ZZ 99 99", 1), "edition_date": _env("01/20", 1)}]}
    [document] = merged_windows([_window("ocr_plus_image", invented)])
    answer, checked = as_served(document, {})
    assert [f["form_number"]["parsed"] for f in answer["forms_and_endorsements"] if f.get("form_number")] == ["CG 00 01"]
    assert PRINTED_CHECK in checked and PRINTED_TEXT not in checked          # checked on its own text


def test_an_image_only_answer_is_checked_on_its_documents_clean_text_borrowed():
    documents = merged_windows([_window("ocr_plus_image"), _window("image_only")])
    donors = {"policy_0001": {1: PAGE}}
    image_only = next(d for d in documents if d.metadata["modality_mode"] == "image_only")
    _answer, checked = as_served(image_only, donors)
    assert checked[PRINTED_TEXT] == BORROWED_TEXT
    assert as_served(image_only, {}) [1] == {}                                 # no text: conventions only, no record


def test_a_second_reading_is_held_to_the_first_readings_text():
    """Both readings of a run-to-run pair lose the same unprinted values, so agreement is not lowered by one side."""
    first = [_window("ocr_plus_image"), _window("image_only")]
    again = [_window("image_only")]
    donors = {"policy_0001": {1: PAGE}}
    [(_gold, _answer, metadata)] = served_documents(again, donors=donors)
    assert metadata[PRINTED_TEXT] == BORROWED_TEXT
    metrics = score_generations(first, repeat=again)
    assert metrics["document_field_normalized_match"] == 1.0
