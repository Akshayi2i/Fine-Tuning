"""How many points each known scoring effect costs a run, and how long its answers are. Read-only: no model, no GPU.

The 2026-11-10 review proved, on the real scorer, that some right answers are counted wrong. This measures how much
each costs on a real run, from its saved answers - the score as the run made it, beside the score with one effect
switched off:

* **page check** (A1): strict mode takes out a value no page prints. Until 2026-11-10 only its ``raw`` text was looked
  for, so a right date written "08/18/2022" where the page prints "08/18/22" was taken out; since then its other
  printed forms are looked for too (``common.grounding.printed_forms``). Counted: the values still taken out that the
  gold holds (same field, same value), and the score with nothing taken out.
* **punctuation** (A2): since 2026-11-10 the scorer compares text with punctuation, quote and dash styles left out
  (``common.normalize.compare_text``). The run is also scored under the rule before - "Accounts Receivable." wrong
  against "ACCOUNTS RECEIVABLE" - so both numbers are seen side by side.
* **windows vs documents** (A3): a coverage written as one row or two fails window by window but not on the whole
  document; checkpoint ranking reads the window number. Both are shown.
* **answer length** (B1): characters, and tokens with ``--tokenizer``, per task - the floor of production's
  2-3 minute target: one window's answer is written at about 47-52 tokens/s (measured 2026-10-28, base model).

    python -m scripts.score_effects --answers /workspace/logs/v3_on_training_docs_halves.answers.jsonl \\
        --tokenizer /workspace/models/Qwen3-VL-8B-Instruct

``--answers`` is ``score_on_training_docs``'s rows file. One saved before 2026-11-10 keeps neither the page text nor
the halves a cut-off window was read again in: ``--pages`` ({source_id: {page: text}}) gives the text then, and its
documents are merged without the halves - the "as the run scored" line then differs from the run's own number, and
both are shown. Changes nothing: no score, gold or setting is written.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: The scores shown: the window score, and the whole-document score production is judged by.
SHOWN = ("field_normalized_match", "document_field_normalized_match", "document_hallucination_rate")
#: Tokens a window's answer is written at, per second, on one request (2026-10-28, base model, H200 pod).
TOKENS_PER_SECOND = 50


def load_generations(path: Path, pages: dict[str, dict[Any, str]] | None = None) -> list[Any]:
    """The run's rows as the scorer reads them (``ValidationGeneration``), halves included where they were saved."""
    from evaluation.validation_generation import ValidationGeneration

    def built(record: dict[str, Any]) -> Any:
        row = dict(record.get("row") or {})
        if "ocr_pages" not in row and pages is not None and row.get("modality_mode") != "image_only":
            text = pages.get(str(row.get("source_id"))) or {}
            wanted = {int(p) for p in row.get("window_pages") or []} or {int(p) for p in text}
            row["ocr_pages"] = {int(p): t for p, t in text.items() if int(p) in wanted}
        return ValidationGeneration(row=row, golden=record.get("gold") or {}, extraction=record.get("answer"),
                                    error=record.get("error"), failure_kind=record.get("failure_kind"),
                                    unparsed_text=record.get("unparsed_text"), cut_off=bool(record.get("cut_off")),
                                    halves=[built(half) for half in record.get("halves") or []])

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [built(json.loads(line)) for line in lines if line.strip()]


@contextlib.contextmanager
def page_check_off() -> Iterator[None]:
    """Strict removal switched to flagging: the answer is checked and nothing is taken out."""
    from serving import printed_values

    saved = printed_values.configured_mode
    printed_values.configured_mode = lambda: printed_values.FLAG
    try:
        yield
    finally:
        printed_values.configured_mode = saved


@contextlib.contextmanager
def punctuation_counted() -> Iterator[None]:
    """The text rule before 2026-11-10 (decision A2): punctuation and quote/dash styles count, only case and spacing
    are evened out. The scorer's own comparison (``common.normalize.compare_text``) is the one changed, so the run is
    shown under both rules, side by side."""
    from common import normalize

    saved = normalize.compare_text
    normalize.compare_text = normalize.normalize_text
    try:
        yield
    finally:
        normalize.compare_text = saved


def scores(generations: list[Any]) -> dict[str, Any]:
    from evaluation.validation_generation import score_generations

    metrics = score_generations(generations)
    return {name: metrics.get(name) for name in SHOWN}


def removed_but_in_gold(generations: list[Any]) -> dict[str, Any]:
    """The values strict mode took out of the whole documents, and those of them the gold holds in the same field."""
    from common.normalize import values_match
    from evaluation.run_eval import PRINTED_CHECK
    from evaluation.validation_generation import merged_windows, served_documents, text_donors
    from scripts.show_eval_errors import field_pattern

    def leaves(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
        if isinstance(node, dict):
            if "raw" in node or "parsed" in node:
                yield field_pattern(path), node.get("parsed", node.get("raw"))
                if node.get("raw") is not None:
                    yield field_pattern(path), node.get("raw")
                return
            for key, value in node.items():
                yield from leaves(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from leaves(value, f"{path}[{index}]")

    donors = text_donors(merged_windows(generations))
    removed = held = 0
    examples: list[dict[str, Any]] = []
    for gold, _answer, metadata in served_documents(generations, donors=donors):
        record = metadata.get(PRINTED_CHECK) or {}
        if not record.get("removed"):
            continue
        by_field: dict[str, list[Any]] = defaultdict(list)
        for pattern, value in leaves(gold):
            by_field[pattern].append(value)
        for item in record.get("not_printed") or []:
            removed += 1
            pattern = field_pattern(str(item.get("path", "")))
            raw = item.get("raw")
            if any(values_match(value, raw, field_path=pattern) for value in by_field.get(pattern, [])):
                held += 1
                if len(examples) < 15:
                    examples.append({"document": metadata.get("source_id"), "mode": metadata.get("modality_mode"),
                                     "field": pattern, "raw": raw, "pages": item.get("pages")})
    return {"taken_out": removed, "gold_holds": held, "examples": examples}


def answer_lengths(generations: list[Any], tokenizer: Any = None) -> dict[str, Any]:
    """Per task: how long the answers are (every reading, halves included), and how many would take over a minute."""
    from common.canonical import to_compact_answer

    def every(gens: list[Any]) -> Iterator[Any]:
        for g in gens:
            yield g
            yield from every(g.halves)

    per_task: dict[str, list[int]] = defaultdict(list)
    tokens_per_task: dict[str, list[int]] = defaultdict(list)
    for g in every(generations):
        if g.unparsed_text:
            text = g.unparsed_text
        elif isinstance(g.extraction, dict):
            text = json.dumps(to_compact_answer(g.extraction), ensure_ascii=False)
        else:
            continue
        task = str(g.row.get("task") or "-")
        per_task[task].append(len(text))
        if tokenizer is not None:
            tokens_per_task[task].append(len(tokenizer.encode(text, add_special_tokens=False)))

    def spread(values: list[int]) -> dict[str, int]:
        ordered = sorted(values)
        at = lambda share: ordered[min(len(ordered) - 1, int(share * len(ordered)))]  # noqa: E731
        return {"answers": len(ordered), "median": at(0.5), "p95": at(0.95), "longest": ordered[-1]}

    out: dict[str, Any] = {}
    for task, values in sorted(per_task.items()):
        entry: dict[str, Any] = {"chars": spread(values)}
        if tokens_per_task.get(task):
            counted = tokens_per_task[task]
            entry["tokens"] = spread(counted)
            entry["over_a_minute"] = sum(1 for n in counted if n / TOKENS_PER_SECOND > 60)
            entry["longest_seconds"] = round(max(counted) / TOKENS_PER_SECOND)
        out[task] = entry
    return out


def measure(generations: list[Any], tokenizer: Any = None) -> dict[str, Any]:
    as_scored = scores(generations)
    with page_check_off():
        no_removal = scores(generations)
    with punctuation_counted():
        punctuation = scores(generations)
    return {"as_scored": as_scored, "page_check_off": no_removal, "punctuation_counted": punctuation,
            "page_check": removed_but_in_gold(generations), "answer_lengths": answer_lengths(generations, tokenizer)}


def _cell(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "-"


def render(report: dict[str, Any], stored: dict[str, Any] | None = None) -> str:
    lines = [f"{'score':40} {'window':>8} {'document':>9} {'doc invented':>13}"]
    rows = [("as the run scored (re-scored here)", report["as_scored"])]
    if stored:
        rows.insert(0, ("the run's own number", stored))
    rows += [("page check takes nothing out (A1)", report["page_check_off"]),
             ("punctuation counted, old rule (A2)", report["punctuation_counted"])]
    for name, values in rows:
        lines.append(f"{name:40} {_cell(values.get(SHOWN[0])):>8} {_cell(values.get(SHOWN[1])):>9} "
                     f"{_cell(values.get(SHOWN[2])):>13}")
    check = report["page_check"]
    lines.append(f"\npage check: {check['taken_out']} value(s) taken out of whole documents, "
                 f"{check['gold_holds']} of them held by the gold in the same field")
    lines += [f"  {e['document']} {e['mode']} {e['field']} = {e['raw']!r} (pages {e['pages']})"
              for e in check["examples"]]
    lines.append("\nanswer length per task (every reading, halves included):")
    for task, entry in report["answer_lengths"].items():
        c = entry["chars"]
        line = f"  {task:20} {c['answers']:>4} answers  chars median {c['median']:,} p95 {c['p95']:,} longest {c['longest']:,}"
        if "tokens" in entry:
            t = entry["tokens"]
            line += (f" | tokens median {t['median']:,} p95 {t['p95']:,} longest {t['longest']:,}"
                     f" (~{entry['longest_seconds']} s at {TOKENS_PER_SECOND}/s; {entry['over_a_minute']} over 60 s)")
        lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True, help="score_on_training_docs' .answers.jsonl")
    parser.add_argument("--pages", default=None, help="JSON {source_id: {page: text}}, for a file saved without it")
    parser.add_argument("--run", default=None, help="the run's own .json, to show its stored scores beside")
    parser.add_argument("--tokenizer", default=None, help="the base model folder, to count tokens")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    pages = json.loads(Path(args.pages).read_text(encoding="utf-8")) if args.pages else None
    generations = load_generations(Path(args.answers), pages)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report = measure(generations, tokenizer)
    stored = None
    if args.run:
        run = json.loads(Path(args.run).read_text(encoding="utf-8"))
        # score_on_training_docs writes the run's own scores under "trained" (beside selection's under "selection").
        metrics = run.get("trained") or run.get("metrics") or run
        stored = {name: metrics.get(name) for name in SHOWN}
    print(json.dumps(report, indent=1, default=str) if args.json else render(report, stored))
    return 0


if __name__ == "__main__":
    sys.exit(main())
