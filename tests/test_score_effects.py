"""scripts/score_effects.py: what each proven scoring effect costs a run, read off its saved answers (2026-11-10)."""

from __future__ import annotations

import json

from evaluation.validation_generation import ValidationGeneration, score_generations
from inference_core.input_builder import build_training_row
from scripts.score_effects import answer_lengths, load_generations, measure
from scripts.score_on_training_docs import answer_record

LOB = "gl"
#: The page prints the date with a two-digit year and the title with no full stop; its text lacks the policy number
#: (a line OCR missed).
PAGE = ("DECLARATIONS\nPolicy Period 08/18/22 to 08/18/23\n"
        "FORMS SCHEDULE\nCG 00 01 04 13 Commercial General Liability Coverage Form")


def _env(value, *pages, raw=None):
    return {"raw": str(value) if raw is None else raw, "parsed": value, "page_ref": list(pages)}


GOLD = {"policy": {"policy_number": _env("GL-1001", 1), "effective_date": _env("08/18/2022", 1, raw="08/18/22")},
        "forms_and_endorsements": [{"form_number": _env("CG 00 01", 1), "edition_date": _env("04/13", 1),
                                    "title": _env("Commercial General Liability Coverage Form", 1)}]}
#: Right on every value; the date's raw is the cleaned value, not what the page prints, and the title ends in a stop.
ANSWER = {"policy": {"policy_number": _env("GL-1001", 1), "effective_date": _env("08/18/2022", 1)},
          "forms_and_endorsements": [{"form_number": _env("CG 00 01", 1), "edition_date": _env("04/13", 1),
                                      "title": _env("Commercial General Liability Coverage Form.", 1)}]}


def _window(mode="ocr_plus_image", answer=ANSWER) -> ValidationGeneration:
    texts = None if mode == "image_only" else [PAGE]
    row = build_training_row("policy", "policy_0001", ["processed/p/page_1.png"], texts, mode, json.dumps({}),
                             lob=LOB, sections="forms", split="val", page_numbers=[1], total_pages=1)
    row.update(window_pages=[1], window_index=0, window_leads=True, task="policy_schedule")
    return ValidationGeneration(row=row, golden=GOLD, extraction=answer)


def _saved(tmp_path, generations) -> list:
    path = tmp_path / "run.answers.jsonl"
    path.write_text("".join(json.dumps(answer_record(g), default=str) + "\n" for g in generations), encoding="utf-8")
    return load_generations(path)


def test_a_saved_run_scores_again_exactly_as_it_scored(tmp_path):
    """The saved file keeps the page text and the halves, so the documents merge and are checked as they were."""
    half = _window()
    whole = _window()
    whole.cut_off, whole.halves = True, [half]
    original = [whole, _window("image_only")]
    loaded = _saved(tmp_path, original)
    assert loaded[0].ocr_pages == {1: PAGE} and loaded[1].ocr_pages is None
    assert loaded[0].cut_off and len(loaded[0].halves) == 1 and loaded[0].halves[0].ocr_pages == {1: PAGE}
    assert score_generations(loaded) == score_generations(original)


def test_a_file_saved_without_page_text_takes_it_from_the_pages_file(tmp_path):
    record = answer_record(_window())
    del record["row"]["ocr_pages"]
    path = tmp_path / "old.answers.jsonl"
    path.write_text(json.dumps(record, default=str) + "\n", encoding="utf-8")
    [loaded] = load_generations(path, {"policy_0001": {"1": PAGE}})
    assert loaded.ocr_pages == {1: PAGE}
    # Code review 2026-11-10: with no saved text the window read as having none, so its values were not looked for.
    assert loaded.page_text and "08/18/22" in loaded.page_text


def test_the_old_text_rule_is_put_back_only_inside_the_measure():
    from common import normalize
    from scripts.score_effects import punctuation_counted

    assert normalize.values_match("ACCOUNTS RECEIVABLE", "Accounts Receivable.", field_path="coverage.description")
    with punctuation_counted():
        assert not normalize.values_match("ACCOUNTS RECEIVABLE", "Accounts Receivable.",
                                          field_path="coverage.description")
    assert normalize.compare_text("Accounts Receivable.") == "accounts receivable"


def test_each_effect_is_measured_on_its_own(tmp_path):
    report = measure(_saved(tmp_path, [_window()]))
    as_scored, no_removal, punctuation = (report[k]["document_field_normalized_match"]
                                         for k in ("as_scored", "page_check_off", "punctuation_counted"))
    # A1: the policy number, right, is taken out - the page's text lacks it (OCR missed its line); the gold holds
    # it. The date written "08/18/2022" where the page prints "08/18/22" is kept since the 2026-11-10 decision.
    assert report["page_check"]["taken_out"] >= 1 and report["page_check"]["gold_holds"] >= 1
    fields = {e["field"] for e in report["page_check"]["examples"]}
    assert "policy.policy_number" in fields and "policy.effective_date" not in fields
    assert no_removal > as_scored
    # A2: the title with a full stop is right under today's rule, wrong under the rule before 2026-11-10.
    assert punctuation < as_scored


def test_answer_lengths_are_counted_per_task_with_their_writing_time():
    class Tokens:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(range(len(text) // 2))

    long = _window()
    long.unparsed_text = "x" * 8000                       # 4,000 tokens: 80 s at 50 tokens/s
    lengths = answer_lengths([_window(), long], Tokens())["policy_schedule"]
    assert lengths["chars"]["answers"] == 2 and lengths["chars"]["longest"] == 8000
    assert lengths["tokens"]["longest"] == 4000 and lengths["longest_seconds"] == 80 and lengths["over_a_minute"] == 1


def test_the_runs_own_scores_are_read_where_score_on_training_docs_writes_them(tmp_path, capsys):
    """Its file holds them under "trained" (beside selection's): read as "metrics", the line showed dashes (pod,
    2026-11-11)."""
    from scripts.score_effects import main

    answers = tmp_path / "run.answers.jsonl"
    answers.write_text(json.dumps(answer_record(_window()), default=str) + "\n", encoding="utf-8")
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"trained": {"document_field_normalized_match": 0.7871}, "selection": {}}),
                   encoding="utf-8")
    assert main(["--answers", str(answers), "--run", str(run)]) == 0
    own = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("the run's own number"))
    assert "0.7871" in own


def test_window_fields_failing_on_one_side_only_are_listed(tmp_path):
    """The run's own failing window fields beside the re-scored ones (pod 2026-11-11: 0.6951 vs 0.6830)."""
    from evaluation.validation_generation import score_generations
    from scripts.score_effects import compare_windows

    original = [_window()]
    reports: dict = {}
    score_generations(original, reports=reports)
    saved = tmp_path / "run.window_errors.json"
    saved.write_text(json.dumps({"subsets": [s.as_dict() for s in reports["windows"].subsets]}, default=str),
                     encoding="utf-8")
    loaded = _saved(tmp_path, original)
    same = compare_windows(loaded, saved)
    assert same["failing_in_run"] == same["failing_here"] and not same["only_in_run"] and not same["only_here"]
    loaded[0].extraction["policy"]["policy_number"] = _env("GL-9999", 1)        # one more failure here only
    differs = compare_windows(loaded, saved)
    assert [e["field"] for e in differs["only_here"]] == ["policy.policy_number"]
    assert differs["more_failing_here_by_document"] == {"policy_0001": 1}
