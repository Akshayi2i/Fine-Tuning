"""Score a release on documents it trained on: can the model do this task at all?

A release is only ever scored on documents it never saw. A low score there does not say why: too little data and
variety, or labels and training that did not teach the task. This reads a few of its OWN training documents through the
same generation and the same scorer as checkpoint selection, and prints the score beside selection's.

* High on its own documents, low on unseen ones: it learned them and does not carry over. More and more varied data.
* Low on its own documents as well: it did not learn even what it was shown. The labels, the training or the model.

A document's rows are taken from the corpus's first epoch file, so each is read in the mode it trained in that epoch.
Documents are drawn line by line, so a small line is not left out. Needs the GPU and the adapter on the pod.

    python -m scripts.score_on_training_docs --version v3 --adapter <checkpoint folder or Blob prefix> --documents 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

#: The metrics printed beside selection's, the score selection chooses on first.
SHOWN = ("document_field_normalized_match", "field_normalized_match", "document_field_f1", "field_f1",
         "field_f1_list_fields", "document_field_f1_list_fields", "list_field_recall", "document_list_field_recall",
         "hallucination_rate", "document_hallucination_rate", "unusable_output_rate", "generation_failed_rate")


def choose_documents(rows: list[dict[str, Any]], count: int, seed: int = 42) -> list[str]:
    """``count`` of the rows' documents, taken in turn from each line, in an order fixed by ``seed``."""
    by_line: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_line[str(row.get("lob"))].add(str(row.get("source_id")))

    def order(source: str) -> str:
        return hashlib.sha256(f"{seed}:{source}".encode()).hexdigest()

    queues = {line: sorted(sources, key=order) for line, sources in sorted(by_line.items())}
    chosen: list[str] = []
    while len(chosen) < count and any(queues.values()):
        for queue in queues.values():
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return sorted(chosen)


def comparison(own: dict[str, Any], selection: dict[str, Any]) -> str:
    out = [f"{'metric':36} {'trained documents':>18} {'selection (unseen)':>19}"]
    for name in SHOWN:
        if name in own or name in selection:
            cells = [f"{value:.4f}" if isinstance(value, (int, float)) else "-" for value in (own.get(name), selection.get(name))]
            out.append(f"{name:36} {cells[0]:>18} {cells[1]:>19}")
    return "\n".join(out)


#: A label of an ``additional_fields`` entry in a written answer, either format.
_LABEL = re.compile(r'"label":\s*"((?:[^"\\]|\\.)*)"')


def runaway(generations: list[Any]) -> list[str]:
    """One line per answer that could not be read (cut off at the token limit): its window, its length, and how
    many ``additional_fields`` labels it wrote against how many differ - a loop writes the same ones again, a long
    list of real values does not (2026-11-09: such answers ran to 29,000+ characters against golds of 199-16,503)."""
    lines = []
    for generation in generations:
        text = getattr(generation, "unparsed_text", None)
        if not text:
            continue
        labels = _LABEL.findall(text)
        row = generation.row
        lines.append(f"  {row.get('source_id')} {row.get('task')} pages {row.get('window_pages')}: {len(text):,} chars "
                     f"against a gold of {len(json.dumps(generation.golden, ensure_ascii=False)):,}; "
                     f"{len(labels)} additional_fields label(s), {len(set(labels))} different")
    return lines


def errors_path(out: Path) -> Path:
    """The whole-document report beside ``out``: every failing field, as ``show_eval_errors.py --file`` reads it."""
    return out.with_name(out.stem + ".errors.json")


def write_details(out: Path, generations: list[Any], reports: dict[str, Any]) -> list[Path]:
    """Beside ``out``: the failing fields of the whole documents (``.errors.json``, what ships) and of the windows
    (``.window_errors.json``), and every row's answer with its gold (``.answers.jsonl``), so a question about the
    run is answered from these files, not another GPU run (2026-11-09: only the totals were kept)."""
    written: list[Path] = []
    for name, suffix in (("documents", ".errors.json"), ("windows", ".window_errors.json")):
        report = reports.get(name)
        if report is None:
            continue
        path = out.with_name(out.stem + suffix)
        path.write_text(json.dumps({"model_version": report.model_version,
                                    "subsets": [subset.as_dict() for subset in report.subsets]},
                                   indent=1, default=str), encoding="utf-8")
        written.append(path)
    path = out.with_name(out.stem + ".answers.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        for generation in generations:
            row = {k: v for k, v in generation.row.items() if k not in ("messages", "images")}
            handle.write(json.dumps({"row": row, "error": generation.error, "failure_kind": generation.failure_kind,
                                     "answer": generation.extraction, "gold": generation.golden,
                                     "unparsed_text": getattr(generation, "unparsed_text", None)},
                                    ensure_ascii=False, default=str))
            handle.write("\n")
    written.append(path)
    return written


def selected_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """The selected checkpoint's metrics in a selection report, its whole-document scores over its sample's."""
    wanted = report.get("selected")
    metrics: dict[str, Any] = {}
    for candidate in report.get("candidates") or []:
        if candidate.get("checkpoint") == wanted:
            metrics.update(candidate.get("metrics") or {})
    finalists = (report.get("document_choice") or {}).get("finalists") or {}
    metrics.update(finalists.get(wanted) or {})
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="the release, e.g. v3")
    parser.add_argument("--adapter", required=True, help="the adapter: a folder on this machine or a Blob prefix")
    parser.add_argument("--corpus-version", dest="corpus_version", default=None, help="its corpus; default: the one its run trained on")
    parser.add_argument("--documents", type=int, default=8, help="how many training documents to read")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="also write the result to this file")
    args = parser.parse_args(argv)
    if args.documents < 1:
        parser.error("--documents is 1 or more")
    # On the pod, run detached in tmux: this reads pages on the GPU for many minutes.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("scripts.score_on_training_docs", argv):
        return 0

    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from common.adapter import ADAPTER
    from evaluation.validation_generation import generate_validation, read_rows, score_generations
    from inference_core.model_runner import load_model
    from orchestration.label_noise import corpus_of
    from training.stage_data import localize_rows

    client = BlobClient()
    corpus = args.corpus_version or corpus_of(client, args.version)
    rows = [row for row in read_rows(client.read_text(paths.corpus_epoch_file(corpus, 1, args.tenant)))
            if row.get("doc_type") in ADAPTER.doc_types]
    chosen = set(choose_documents(rows, args.documents, args.seed))
    rows = [row for row in rows if str(row.get("source_id")) in chosen]
    if not rows:
        print(f"corpus {corpus} holds no training rows to read", file=sys.stderr)
        return 1
    print(f"reading {len(rows)} row(s) of {len(chosen)} training document(s): {sorted(chosen)}", flush=True)
    rows = localize_rows(rows, client, paths.staging_train_images_dir(corpus, args.tenant))
    # Given to load_model, which refuses a folder holding no adapter: the bare base must not answer in its name.
    model = load_model("base", client, adapter=args.adapter)
    generations = generate_validation(rows, model, adapter=model.default_adapter)
    reports: dict[str, Any] = {}
    own = score_generations(generations, model_version=f"{args.version}-on-its-training-documents",
                            reports=reports)
    key = paths.checkpoint_selection(args.version, scope=ADAPTER.name)
    selection = selected_metrics(client.read_json(key)) if client.exists(key) else {}
    print(comparison(own, selection))
    if lines := runaway(generations):
        print("answers cut off at the token limit (the whole text is in the .answers.jsonl file):")
        print("\n".join(lines))
    if args.out:
        Path(args.out).write_text(json.dumps({"documents": sorted(chosen), "rows": len(rows), "trained": own,
                                              "selection": selection}, indent=1, default=str), encoding="utf-8")
        for path in write_details(Path(args.out), generations, reports):
            print(f"written: {path}")
        print(f"where it went wrong: python scripts/show_eval_errors.py --file {errors_path(Path(args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
