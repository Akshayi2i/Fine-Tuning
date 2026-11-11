"""Would production's loop stop cut a good answer short? Read-only: no model, no GPU.

Production reads an answer as it streams and checks it every :data:`REPEAT_CHECK_CHARS` characters with two loop
tests (``inference_core.openai_backend._read_stream``): the end repeating one passage (``repeating_tail``) and one
row name written again and again (``repeating_row``). A stopped answer is read again in halves, and a single page
cannot be - so a test that fires on a real answer costs that page's values. The golds were checked (2026-11-10:
none of 168 trips it); this checks the model's own answers from a run's saved rows file:

* every answer that parsed, written as the model writes it (its compact form, ``json.dumps``' separators), is
  checked at each 1,000-character step and at its end, as the stream would be - **must print 0 stopped**;
* every answer cut off at its token limit (its whole text is kept): the character at which the stop would have
  ended it, and how much of the answer that saves.

    python -m scripts.loop_guard_check --answers /workspace/logs/v3_on_training_docs_halves.answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def records(path: Path) -> Iterator[dict[str, Any]]:
    """Every saved reading, the halves a cut-off window was read again in included."""

    def walk(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield record
        for half in record.get("halves") or []:
            yield from walk(half)

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield from walk(json.loads(line))


def first_stop(text: str, step: int | None = None) -> tuple[int, str] | None:
    """``(characters written, which test)`` where production's stream check would first stop ``text``, or None."""
    from inference_core.openai_backend import REPEAT_CHECK_CHARS, repetition

    step = step or REPEAT_CHECK_CHARS
    points = list(range(step, len(text), step)) + [len(text)]
    for at in points:
        if (found := repetition(text[:at])) is not None:
            return at, f"{found[0]}: {found[1]}"
    return None


def check(path: Path) -> dict[str, Any]:
    from common.canonical import to_compact_answer

    good = stopped_good = 0
    wrongly: list[dict[str, Any]] = []
    cut: list[dict[str, Any]] = []
    for record in records(path):
        row = record.get("row") or {}
        where = {"document": row.get("source_id"), "task": row.get("task"), "mode": row.get("modality_mode"),
                 "pages": row.get("window_pages")}
        # A reading that failed is saved with an empty answer ({}) beside its error (validation_generation
        # _record_failure), so the answer alone does not make it good (2026-11-11: the first pod run counted the 5
        # cut-off readings as good and showed 0 cut off). An answer that ran to the token limit is cut off even
        # when what it wrote still parses: serving refuses it and reads the window again in halves.
        answer = record.get("answer")
        cut_but_parsed = record.get("cut_off") and isinstance(answer, dict) and not record.get("error")
        if record.get("unparsed_text") or cut_but_parsed:
            text = record.get("unparsed_text") or json.dumps(to_compact_answer(answer), ensure_ascii=False)
            stop = first_stop(text)
            cut.append({**where, "length": len(text), "at": stop[0] if stop else None,
                        "test": stop[1] if stop else None})
        elif isinstance(answer, dict) and not record.get("error"):
            good += 1
            text = json.dumps(to_compact_answer(answer), ensure_ascii=False)
            if (stop := first_stop(text)) is not None:
                stopped_good += 1
                wrongly.append({**where, "at": stop[0], "length": len(text), "test": stop[1]})
    return {"good_answers": good, "good_answers_stopped": stopped_good, "wrongly_stopped": wrongly,
            "cut_off_answers": cut}


def render(report: dict[str, Any]) -> str:
    lines = [f"good answers checked: {report['good_answers']}",
             f"real answers wrongly stopped: {report['good_answers_stopped']}"]
    lines += [f"  {w['document']} {w['task']} {w['mode']} pages {w['pages']}: stopped at {w['at']:,} of "
              f"{w['length']:,} characters ({w['test']})" for w in report["wrongly_stopped"]]
    cut = report["cut_off_answers"]
    caught = [c for c in cut if c["at"] is not None]
    lines.append(f"answers cut off at the token limit: {len(cut)}, of them the stop would have caught {len(caught)}")
    lines += [f"  {c['document']} {c['task']} {c['mode']} pages {c['pages']}: "
              + (f"stopped at {c['at']:,} of {c['length']:,} characters ({c['test']}), "
                 f"{1 - c['at'] / c['length']:.0%} not written" if c["at"] is not None
                 else f"not caught ({c['length']:,} characters)")
              for c in cut]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True, help="score_on_training_docs' .answers.jsonl")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = check(Path(args.answers))
    print(json.dumps(report, indent=1) if args.json else render(report))
    return 1 if report["good_answers_stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
