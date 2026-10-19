"""Before ``finetune``: everything it depends on, checked in minutes instead of found hours in.

    python -m orchestration.preflight --scope personal_lines
    python -m orchestration.preflight --scope personal_lines --tenant smoke --ocr-check /workspace/ocr_check_smoke

Every check says PASS, WARN or FAIL, and a FAIL says what to do. Exit 1 on any
FAIL, so a script can refuse to start training on it.

**The environment** - the GPU is free (another process holding memory was the
first thing found on the pod), flash-attn, vLLM and the ``swift`` CLI are
there, the base model is the pinned revision, the config invariants hold.

**Azure** - the token can write, read and delete (a read-only token passes an
upload's first files and fails the run's first overwrite).

**The data** - every labelled document is OCR'd, the post-OCR check passed, the
delivered split is whole, and there are enough documents per type.

**The corpus** - built in memory by the very function the dataset build uses
(``pipeline_dag.plan_corpus``), nothing written: rows per split, documents set
aside and why, rows over their sequence cap. A preflight that passes has built
the corpus the run will write. What it cannot know is memory under real
sequences; the smoke run measures that, and training's own length check reads
every staged row with the real tokenizer before ``swift sft`` starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WORKSPACE = Path(os.environ.get("FIDEON_WORKSPACE") or "/workspace")  # an empty .env value means unset
#: Free GPU memory below this share of the card means something else holds it.
MIN_FREE_GPU_SHARE = 0.9
MIN_FREE_DISK_GB = 100
#: Share of documents set aside by the sequence caps above which the run would
#: train on too little of the corpus to be worth it.
WARN_REJECTED_SHARE, FAIL_REJECTED_SHARE = 0.05, 0.25


@dataclass
class Check:
    name: str
    status: str          # PASS / WARN / FAIL
    detail: str
    fix: str = ""


def _run(name: str, fn, results: list[Check]) -> Any:
    """Run one check; an exception is that check failing, never the preflight."""
    try:
        return fn(results)
    except Exception as exc:  # noqa: BLE001 - reported, the next check still runs
        results.append(Check(name, "FAIL", f"{type(exc).__name__}: {exc}"))
        return None


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def check_gpu(results: list[Check]) -> None:
    import torch

    if not torch.cuda.is_available():
        results.append(Check("gpu", "FAIL", "torch sees no CUDA device",
                             "run from /workspace/venv on the pod; check nvidia-smi"))
        return
    # Every visible card: training uses them all (distributed.gpus: auto), and
    # a leftover vLLM server or OCR shard on GPU 2 passed a check of GPU 0, then
    # took the run down at model load on rank 2.
    count = torch.cuda.device_count()
    busy, lines = [], []
    for index in range(count):
        free, total = torch.cuda.mem_get_info(index)
        lines.append(f"{index}: {free / 2**30:.0f}/{total / 2**30:.0f} GiB free")
        if free < MIN_FREE_GPU_SHARE * total:
            busy.append(index)
    detail = f"{count} x {torch.cuda.get_device_name(0)} ({'; '.join(lines)})"
    if busy:
        results.append(Check("gpu", "FAIL", detail + f" - another process holds GPU {busy}",
                             "nvidia-smi --query-compute-apps=pid,process_name --format=csv; stop it "
                             "(a vLLM server: tmux kill-session -t model)"))
        return
    results.append(Check("gpu", "PASS", detail))
    try:
        from common.config import training_config
        from training.train import _accumulation_steps, training_gpus

        cfg = training_config("unified")
        gpus = training_gpus(cfg.get("distributed", {}))
        _accumulation_steps(cfg["batch"], gpus)      # refuses a batch the GPUs do not divide
        results.append(Check("training gpus", "PASS", f"training will use {gpus} of {count} GPU(s)"))
    except Exception as exc:  # noqa: BLE001 - e.g. a batch size the GPU count does not divide
        results.append(Check("training gpus", "FAIL", f"{type(exc).__name__}: {exc}"[:300],
                             "configs/training/unified.yaml: distributed.gpus / effective_batch_size"))


def check_packages(results: list[Check]) -> None:
    import importlib.metadata as md

    problems, versions = [], {}
    for dist in ("torch", "transformers", "vllm", "flash-attn", "ms-swift", "peft", "deepspeed"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            problems.append(f"{dist} missing")
    for module in ("flash_attn", "vllm", "peft"):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 - an import that fails is the finding
            problems.append(f"import {module}: {type(exc).__name__}: {exc}"[:160])
    if not versions.get("transformers", "").startswith("4.57."):
        problems.append(f"transformers {versions.get('transformers')} (the stack is 4.57)")
    bin_dir = os.path.dirname(sys.executable)
    if shutil.which("swift", path=os.pathsep.join(filter(None, (bin_dir, os.environ.get("PATH"))))) is None:
        problems.append("the swift CLI is not next to this Python")
    detail = ", ".join(f"{k} {v}" for k, v in versions.items())
    results.append(Check("packages", "FAIL" if problems else "PASS",
                         "; ".join(problems) if problems else detail,
                         "bash scripts/pod_bootstrap.sh (rebuilds /workspace/venv)" if problems else ""))


def check_base_model(results: list[Check]) -> None:
    from common.config import base_model_config, base_model_dir

    model = base_model_config()["model"]
    local = base_model_dir()
    if local is None:
        results.append(Check("base model", "FAIL", f"no model under {model.get('local_dir')}",
                             "bash scripts/pod_bootstrap.sh (verifies or downloads it)"))
        return
    marker = local / ".fideon-revision"
    pinned = str(model["revision"])
    if not marker.is_file() or marker.read_text().strip() != pinned:
        results.append(Check("base model", "FAIL", f"{local} is not verified as revision {pinned[:12]}",
                             "bash scripts/pod_bootstrap.sh (verifies the copy against the pin)"))
        return
    results.append(Check("base model", "PASS", f"{local} = {model['model_id']}@{pinned[:12]}"))


def check_config(results: list[Check]) -> None:
    from common.config import validate_all

    validate_all(require_pinned_revision=True)
    results.append(Check("config invariants", "PASS", "resolution caps, budgets, batch arithmetic, pinned revision"))


def check_disk(results: list[Check]) -> None:
    free_gb = shutil.disk_usage(WORKSPACE).free / 1e9
    status = "PASS" if free_gb >= MIN_FREE_DISK_GB else "FAIL"
    results.append(Check("disk", status, f"{free_gb:,.0f} GB free on {WORKSPACE}",
                         "" if status == "PASS" else "grow the network volume"))


# --------------------------------------------------------------------------
# Azure
# --------------------------------------------------------------------------


def check_blob(client, results: list[Check]) -> None:
    key = "_preflight/probe.txt"
    client.write_text(key, "preflight")
    client.write_text(key, "preflight again")               # an overwrite: what a read-only token fails
    assert client.read_text(key) == "preflight again"
    client.delete(key)
    results.append(Check("azure", "PASS", f"write, overwrite, read and delete in {client.container}"))


# --------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------


def check_ocr_done(ctx, results: list[Check]) -> None:
    from artifact_registry.blob_client import for_ocr
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids
    from data_pipeline.ocr.run_mineru import find_unprocessed

    raw = for_ocr()
    for doc_type in ctx.scope.doc_types:
        labelled = list_labeled_source_ids(ctx.client, doc_type, ctx.tenant_id)
        pending = find_unprocessed(raw, doc_type, ctx.tenant_id)
        if not labelled:
            results.append(Check(f"data {doc_type}", "FAIL", "no labelled documents",
                                 "import first: python -m data_pipeline.ingestion.import_labeled_pdfs ..."))
        elif pending:
            results.append(Check(f"data {doc_type}", "FAIL", f"{len(labelled)} labelled, {len(pending)} not OCR'd",
                                 f"/workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type "
                                 f"{doc_type} --all-unprocessed" + (f" --tenant {ctx.tenant_id}" if ctx.tenant_id else "")))
        elif len(labelled) < ctx.min_labels_per_type:
            results.append(Check(f"data {doc_type}", "FAIL",
                                 f"{len(labelled)} labelled, fewer than {ctx.min_labels_per_type}"))
        else:
            results.append(Check(f"data {doc_type}", "PASS", f"{len(labelled)} labelled, all OCR'd"))


def _label_metadata(ctx) -> list[tuple[str, str, dict[str, Any]]]:
    """``(doc_type, source_id, metadata)`` for every labelled document in scope."""
    from artifact_registry import paths
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    found = []
    for doc_type in ctx.scope.doc_types:
        for source_id in list_labeled_source_ids(ctx.client, doc_type, ctx.tenant_id):
            key = paths.label_metadata(doc_type, source_id, ctx.tenant_id)
            metadata = ctx.client.read_json(key) if ctx.client.exists(key) else {}
            found.append((doc_type, source_id, metadata))
    return found


def check_label_agreement(ctx, results: list[Check], labels=None) -> None:
    """How far two annotators agree on the documents both labelled (arch §7).

    Agreement is the noise in the labelling process and a ceiling on what the
    model can score. No spec states a minimum, so this warns and reports it
    (``common.constants.MIN_INTER_ANNOTATOR_AGREEMENT`` makes it a blocker).
    """
    from common.constants import MIN_INTER_ANNOTATOR_AGREEMENT

    labels = _label_metadata(ctx) if labels is None else labels
    scores = {
        source_id: float(m["agreement_score"])
        for _doc_type, source_id, m in labels
        if m.get("double_annotated") and isinstance(m.get("agreement_score"), (int, float))
    }
    if not scores:
        results.append(Check(
            "label agreement", "WARN",
            f"none of {len(labels)} labelled documents is double-labelled, so label noise is "
            "unmeasured", "double-label 10-20% of documents (IMPL-04) and record agreement_score"))
        return
    detail = (f"{len(scores)} double-labelled: mean agreement {sum(scores.values()) / len(scores):.3f}, "
              f"lowest {min(scores.values()):.3f}")
    if MIN_INTER_ANNOTATOR_AGREEMENT is None:
        results.append(Check("label agreement", "WARN", f"{detail}; no minimum set",
                             "set MIN_INTER_ANNOTATOR_AGREEMENT in common/constants.py to block on it"))
        return
    below = sorted(s for s, v in scores.items() if v < MIN_INTER_ANNOTATOR_AGREEMENT)
    if below:
        results.append(Check("label agreement", "FAIL",
                             f"{detail}; {len(below)} below {MIN_INTER_ANNOTATOR_AGREEMENT}: {below[:5]}",
                             "reconcile those labels before training"))
    else:
        results.append(Check("label agreement", "PASS", detail))


def check_reviewed_labels(ctx, results: list[Check], labels=None, read_label=None) -> None:
    """Every model-drafted label has a reviewer recorded (decision 2026-10-19).

    The corpus build refuses the run on the first one it meets; this names all
    of them before anything is built.
    """
    from artifact_registry import paths
    from data_pipeline.labeling.export_golden_labels import has_review_record, review_problem

    labels = _label_metadata(ctx) if labels is None else labels
    if read_label is None:
        def read_label(doc_type, source_id):
            return ctx.client.read_json(paths.golden_label(doc_type, source_id, ctx.tenant_id))
    blocked, unrecorded = [], 0
    for doc_type, source_id, metadata in labels:
        if has_review_record(metadata):
            continue
        if review_problem(read_label(doc_type, source_id), metadata):
            blocked.append(source_id)
        else:
            unrecorded += 1
    if blocked:
        results.append(Check(
            "reviewed labels", "FAIL",
            f"{len(blocked)} of {len(labels)} labels were drafted by a model with no reviewer "
            f"recorded: {blocked[:5]}",
            "have a person review them, record reviewer_id and review_date, and re-import"))
    elif unrecorded:
        results.append(Check("reviewed labels", "WARN",
                             f"{unrecorded} of {len(labels)} labels record no reviewer"))
    else:
        results.append(Check("reviewed labels", "PASS", f"{len(labels)} labels, all reviewed"))


def check_ocr_report(path: Path, results: list[Check]) -> None:
    summary = path / "summary.json"
    if not summary.is_file():
        results.append(Check("post-OCR check", "FAIL", f"no report at {summary}",
                             "python -m data_pipeline.ocr_check --doc-type policy"))
        return
    facts = json.loads(summary.read_text(encoding="utf-8"))
    results.append(Check("post-OCR check", "PASS" if facts.get("passed") else "FAIL",
                         str(facts.get("verdict"))[:300],
                         "" if facts.get("passed") else f"read {path / 'report.md'}; fix the labels first"))


def check_corpus(ctx, results: list[Check]) -> dict[str, Any]:
    from orchestration.pipeline_dag import corpus_set_aside, plan_corpus

    plan = plan_corpus(ctx)
    built = plan.built
    rows = {split: len(built.rows_by_split.get(split, [])) for split in ("train", "val", "test")}
    needed = ("train", "val") if plan.frozen else ("train", "val", "test")
    empty = [s for s in needed if not rows[s]]
    documents = len(plan.documents)
    kept = len({r["source_id"] for r in built.all_rows})
    skipped = Counter(reason.split(":")[0][:80] for _, reason in built.skipped)
    cap = built.cap_report.as_dict()
    set_aside, share = corpus_set_aside(plan.documents, built)
    facts = {"documents": documents, "kept": kept, "rows": rows, "delivered_split": plan.delivered,
             "frozen_eval_set": plan.frozen, "set_aside_reasons": dict(skipped.most_common(10)), "cap_report": cap}
    detail = (f"{documents} documents, {kept} kept; rows train {rows['train']}, val {rows['val']}, "
              f"test {rows['test']}; set aside {set_aside} ({share:.1%})")
    if empty:
        results.append(Check("corpus", "FAIL", f"{detail}; no {'/'.join(empty)} rows",
                             "see set_aside_reasons in the preflight report"))
    elif share > FAIL_REJECTED_SHARE:
        results.append(Check("corpus", "FAIL", detail,
                             "too many documents over the sequence caps: raise max_seq_len/caps in "
                             "configs/shared/sequence.yaml, measured by the smoke run"))
    elif share > WARN_REJECTED_SHARE:
        results.append(Check("corpus", "WARN", detail, "documents over the caps do not train; see cap_report"))
    else:
        results.append(Check("corpus", "PASS", detail))
    return facts


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def run_preflight(args: argparse.Namespace) -> tuple[list[Check], dict[str, Any]]:
    from orchestration.run import build_context

    results: list[Check] = []
    _run("gpu", check_gpu, results)
    _run("packages", check_packages, results)
    _run("base model", check_base_model, results)
    _run("config invariants", check_config, results)
    _run("disk", check_disk, results)
    ctx = build_context(args, dry_run=True)
    ctx.doc_types = list(ctx.scope.doc_types)
    _run("azure", lambda r: check_blob(ctx.client, r), results)
    _run("data", lambda r: check_ocr_done(ctx, r), results)
    labels = _run("labels", lambda r: _label_metadata(ctx), results) or []
    _run("reviewed labels", lambda r: check_reviewed_labels(ctx, r, labels), results)
    _run("label agreement", lambda r: check_label_agreement(ctx, r, labels), results)
    _run("post-OCR check", lambda r: check_ocr_report(args.ocr_check, r), results)
    facts: dict[str, Any] = {}
    if not args.skip_corpus:
        facts = _run("corpus", lambda r: check_corpus(ctx, r), results) or {}
    return results, facts


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over the checks
    parser = argparse.ArgumentParser(description="Check everything finetune depends on, without training")
    parser.add_argument("--scope", dest="scopes", action="append", default=[])
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--corpus-version", dest="corpus_version", default="v1")
    parser.add_argument("--out-version", dest="out_version", default="v1")
    parser.add_argument("--min-labels-per-type", dest="min_labels_per_type", type=int, default=None)
    parser.add_argument("--ocr-check", type=Path, default=WORKSPACE / "ocr_check",
                        help="the post-OCR check's output folder")
    parser.add_argument("--skip-corpus", action="store_true", help="skip the in-memory corpus build")
    parser.add_argument("--out", type=Path, default=WORKSPACE / "logs" / "preflight.json")
    args = parser.parse_args(argv)
    if args.min_labels_per_type is None:
        from common.constants import DAY_ZERO_MIN_LABELS_PER_TYPE

        args.min_labels_per_type = DAY_ZERO_MIN_LABELS_PER_TYPE
    # On the pod, run detached in tmux: the corpus build reads every page from Blob.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("orchestration.preflight", argv):
        return 0

    results, facts = run_preflight(args)
    width = max(len(c.name) for c in results)
    print("\nPREFLIGHT")
    for c in results:
        print(f"  {c.status:4}  {c.name:{width}}  {c.detail}")
        if c.fix and c.status != "PASS":
            print(f"        {'':{width}}  -> {c.fix}")
    failed = [c for c in results if c.status == "FAIL"]
    verdict = "FAIL: fix the items above before finetune" if failed else "PASS: ready for finetune"
    print(f"\n{verdict}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checks": [asdict(c) for c in results], "corpus": facts,
                                    "passed": not failed}, indent=2, default=str), encoding="utf-8")
    print(f"report: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
