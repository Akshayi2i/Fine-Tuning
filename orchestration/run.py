"""The operator command surface (IMPL-13 §1).

Three commands plus one umbrella::

    python -m orchestration.run finetune --input ./intake --out-version v2 --gpu a100-80
    python -m orchestration.run package  --version v2 --formats bf16 fp8
    python -m orchestration.run extract  --model base --input testing/test_data/
    python -m orchestration.run all      --input ./intake --out-version v2

``all`` is ``finetune`` then ``package``. **It never runs ``extract``** —
extraction is run against a chosen model version whenever you want, including
against models trained weeks earlier and against the untuned base, so folding it
into the build would conflate "produce a model" with "use a model".

This module is argument parsing and nothing else. Every decision about stage
order, idempotency and the gate lives in ``pipeline_dag``, so the CLI cannot
develop its own opinion about them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from artifact_registry.blob_client import BlobClient, for_ingestion
from common.constants import ACTIVE_DOC_TYPES, DAY_ZERO_MIN_LABELS_PER_TYPE
from common.scopes import Scope, load_scopes, parse_scopes
from orchestration import pipeline_dag
from orchestration.pipeline_dag import (
    MultiScopeReport,
    RunReport,
    StageContext,
    run_multi_scope,
    run_stages,
    stages_for,
    stages_from,
)
from orchestration.runpod_controller import RunPodController
from orchestration.settings import backoff_seconds, defaults, retry_policy

log = logging.getLogger(__name__)


def build_context(args: argparse.Namespace, *, client: BlobClient | None = None,
                  controller: RunPodController | None = None,
                  scope: Scope | None = None, **over: object) -> StageContext:
    """Turn parsed arguments into the one object the stages share."""
    from registry_utils.write_run_manifest import capture_git_commit

    commit = getattr(args, "commit", None) or capture_git_commit()
    scopes = parse_scopes(getattr(args, "scopes", None))
    scope = scope or scopes[0]
    controller = controller or RunPodController(git_commit=commit)

    kwargs: dict[str, object] = {
        "client": client or BlobClient(),
        # Only ingestion and OCR may reach raw-documents/; every other stage
        # works from processed/ (arch §18a).
        "raw_client": None if client else for_ingestion(),
        "controller": controller,
        "out_version": getattr(args, "out_version", None) or getattr(args, "version", ""),
        "corpus_version": getattr(args, "corpus_version", "") or "",
        "input_dir": Path(args.input) if getattr(args, "input", None) else None,
        "doc_types": list(getattr(args, "doc_types", None) or ACTIVE_DOC_TYPES),
        "tenant_id": getattr(args, "tenant", None),
        "formats": list(getattr(args, "formats", None) or ["bf16"]),
        "release_id": release_id_for(
            getattr(args, "release_id", None), scope, scope_count=len(scopes)
        ),
        "scope": scope,
        "dtype": getattr(args, "dtype", "bf16"),
        "gpu_class": getattr(args, "gpu", None),
        "dry_run": getattr(args, "dry_run", False),
        "skip_ingest": getattr(args, "skip_ingest", False),
        "train_vit": getattr(args, "train_vit", False),
        "min_labels_per_type": getattr(args, "min_labels_per_type", DAY_ZERO_MIN_LABELS_PER_TYPE),
        "push_adapters": getattr(args, "push_adapters", False),
        "skip_quantize": getattr(args, "skip_quantize", False),
        "keep_staging": getattr(args, "keep_staging", False),
        "from_blob": getattr(args, "from_blob", False),
        "fixes_defect": getattr(args, "fixes_defect", None),
        "git_commit": commit,
        "max_attempts": int(retry_policy()["max_attempts"]),
        "retry_backoff_seconds": backoff_seconds(),
    }
    kwargs.update(over)
    return StageContext(**kwargs)  # type: ignore[arg-type]


class ReleaseIdError(ValueError):
    """Raised when --release-id cannot be resolved for the scopes being run."""


def release_id_for(
    given: list[str] | str | None, scope: Scope, *, scope_count: int = 1
) -> str:
    """This scope's release id, from ``--release-id``.

    One bare id when a single scope is named; ``scope=id`` pairs when several
    are. Each scope produces its OWN release — its own calibrators, gate decision
    and bundle, all addressed by the id — so two scopes sharing one id means the
    second bundle overwrites the first and deletes its index row, taking the
    first release's calibrators and gate decision with it.

    That is refused here, before any work, rather than discovered afterwards as a
    release that quietly lost half its record.
    """
    values = [given] if isinstance(given, str) else list(given or [])
    bare = [v.strip() for v in values if "=" not in v and v.strip()]
    pairs = dict(v.split("=", 1) for v in values if "=" in v)

    if scope.name in pairs:
        return pairs[scope.name].strip()
    if bare and scope_count > 1:
        raise ReleaseIdError(
            f"--release-id {bare[0]!r} was given for {scope_count} scopes. Each scope produces "
            "its own release, so they cannot share an id — the second bundle would overwrite "
            "the first. Name one per scope: --release-id "
            f"{scope.name}={bare[0]} --release-id <other-scope>=<its-id>"
        )
    if pairs and not bare:
        # Pairs were given and this scope is not among them: left empty so
        # `assert_release_id` refuses it by name, with the next free id.
        return ""
    return bare[0] if len(bare) == 1 else ""


def _selected_stages(command: str, from_stage: str | None) -> Sequence[pipeline_dag.Stage]:
    if not from_stage:
        return stages_for(command)
    # `--from-stage` on `all` still spans both commands: resuming at `merge`
    # should carry on into packaging, not stop at the end of finetune.
    return stages_from(from_stage, command)


def run_command(command: str, ctx: StageContext, *, from_stage: str | None = None) -> RunReport:
    """Run one build command for ONE scope. ``all`` is the two commands back to back."""
    return run_stages(ctx, _selected_stages(command, from_stage), command=command)


def run_scopes(
    args: argparse.Namespace,
    command: str,
    *,
    client: BlobClient | None = None,
    controller: RunPodController | None = None,
    **over: object,
) -> MultiScopeReport:
    """Run a command across every scope named on the command line.

    Stages 1-4 build the corpus once over the union of the scopes' document
    types; stages 5-11 then run once per scope, each with its own context.
    """
    scopes = parse_scopes(getattr(args, "scopes", None))
    return run_multi_scope(
        lambda scope: build_context(
            args, client=client, controller=controller, scope=scope, **over
        ),
        scopes,
        command=command,
        from_stage=getattr(args, "from_stage", None),
    )


# --------------------------------------------------------------------------
# extract — a wrapper, never a second implementation
# --------------------------------------------------------------------------


def run_extract(args: argparse.Namespace) -> int:
    """Delegate to ``testing/run_extraction.py``, which wraps the serving pipeline.

    Two layers of wrapping with no logic in either is the point: extraction has
    exactly one implementation (``serving/pipeline.py``), so the numbers this
    command produces describe the system that actually runs in production.
    """
    from testing import run_extraction

    argv = ["--model", args.model, "--mode", args.mode, "--doc-type", args.doc_type]
    if args.adapter:
        argv += ["--adapter", args.adapter]
    if args.label:
        argv += ["--label", args.label]
    if args.source_ids:
        argv += ["--source-ids", *args.source_ids]
    if args.split:
        argv += ["--split", args.split]
    if args.corpus:
        argv += ["--corpus", args.corpus]
    if not args.score:
        argv += ["--no-score"]
    if args.quant_format:
        argv += ["--format", args.quant_format]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.tenant:
        argv += ["--tenant", args.tenant]
    return run_extraction.main(argv)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_finetune_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, help="directory of PDFs to ingest")
    parser.add_argument("--doc-types", nargs="+", dest="doc_types", choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--corpus-version", dest="corpus_version", default="")
    parser.add_argument("--out-version", dest="out_version", required=True)
    parser.add_argument("--gpu", default=None, help="GPU class override for the training pod")
    parser.add_argument("--commit", default=None, help="repo commit the pod clones; defaults to HEAD")
    parser.add_argument("--from-stage", dest="from_stage", default=None,
                        help="resume at this stage; earlier stages are not re-run")
    parser.add_argument("--skip-ingest", dest="skip_ingest", action="store_true")
    parser.add_argument("--scope", dest="scopes", action="append", default=[],
                        choices=sorted(load_scopes()),
                        help="what to train (configs/scopes.yaml). Repeatable: each scope is an "
                             "independent adapter. Defaults to the unified scope.")
    parser.add_argument("--train-vit", dest="train_vit", action="store_true",
                        help="LoRA on the vision encoder; never a full fine-tune (arch §3)")
    parser.add_argument("--min-labels-per-type", dest="min_labels_per_type", type=int,
                        default=defaults().get("min_labels_per_type", DAY_ZERO_MIN_LABELS_PER_TYPE))
    parser.add_argument("--push-adapters", dest="push_adapters", action="store_true",
                        help="also push adapters to Blob immediately, leaving only the merged model staged")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="assemble and record everything without launching GPU work")


def _add_package_flags(parser: argparse.ArgumentParser, *, version_required: bool = True) -> None:
    if version_required:
        parser.add_argument("--version", required=True)
    parser.add_argument("--formats", nargs="+", default=list(defaults().get("formats", ["bf16"])))
    parser.add_argument("--release-id", dest="release_id", action="append", default=[],
                        help="release-YYYY.M.N, or scope=release-YYYY.M.N when several scopes "
                             "are named. Required, and the same on a --from-stage resume")
    parser.add_argument("--skip-quantize", dest="skip_quantize", action="store_true")
    parser.add_argument("--keep-staging", dest="keep_staging", action="store_true")
    parser.add_argument("--from-blob", dest="from_blob", action="store_true",
                        help="artifacts were pushed with --push-adapters; read them from Blob")
    # bf16: the adapter trained against a bf16 base and FP8 is quantized from the
    # merge, so an fp16 merge inserts a precision change nothing asked for.
    parser.add_argument("--dtype", default=defaults().get("dtype", "bf16"), choices=["bf16", "fp16"])
    # Satisfies the gate's improvement rule only (arch §15.5): floors and
    # non-inferiority still apply, and the text is recorded in the decision.
    parser.add_argument("--fixes-defect", dest="fixes_defect", default=None,
                        help="the documented defect this release fixes (ticket or sentence); "
                             "stands in for 'a primary metric improved' and nothing else")
    if version_required:
        # Standalone `package` only — under `all` these come from the finetune
        # flags and adding them twice is an argparse conflict.
        #
        # `finetune` decides which models get built; `package` publishes their
        # locations. Without these, a standalone `package` published whatever the
        # defaults said rather than what finetune actually produced, writing Blob
        # prefixes for adapters and merged models that were never built — an
        # empty prefix that later reads as a published model. `--scope` is the
        # sharpest case: packaging the wrong scope publishes another scope's
        # paths under this one's release.
        parser.add_argument("--scope", dest="scopes", action="append", default=[],
                            choices=sorted(load_scopes()),
                            help="must match the finetune run that produced this version")
        parser.add_argument("--doc-types", dest="doc_types", nargs="+",
                            default=list(ACTIVE_DOC_TYPES),
                            help="must match the finetune run that produced this version")
        parser.add_argument("--tenant", default=None)
        parser.add_argument("--corpus-version", dest="corpus_version", default="",
                            help="the corpus the version trained on; read from its run "
                                 "manifest when omitted")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestration.run",
        description="Fine-tuning lifecycle: finetune -> package -> extract. `all` = finetune + package.",
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_finetune_flags(sub.add_parser("finetune", help="stages 1-7: ingest through merge; artifacts staged"))

    package = sub.add_parser("package", help="stages 8-9: quantize and push to Azure Blob")
    _add_package_flags(package)
    parser_all = sub.add_parser("all", help="finetune then package. Never runs extract.")
    _add_finetune_flags(parser_all)
    _add_package_flags(parser_all, version_required=False)

    extract = sub.add_parser("extract", help="extract with a chosen model version")
    extract.add_argument("--model", required=True,
                         help="base | v1 | v2 ... — 'base' is the untuned model with no adapter")
    extract.add_argument("--adapter", default=None,
                         help="LoRA adapter folder applied to the BASE model on every request, not "
                              "merged (use with --model base)")
    extract.add_argument("--label", default=None, help="name to file results under")
    picked = extract.add_mutually_exclusive_group()
    picked.add_argument("--source-ids", dest="source_ids", nargs="+", default=None,
                        help="imported, OCR'd documents to extract")
    picked.add_argument("--split", choices=["train", "val", "test"], default=None)
    extract.add_argument("--corpus", default=None, help="the corpus whose split --split reads")
    extract.add_argument("--doc-type", dest="doc_type", default="policy")
    extract.add_argument("--mode", default="ocr_plus_image", choices=["ocr_plus_image", "image_only"])
    extract.add_argument("--no-score", dest="score", action="store_false",
                         help="extract without comparing to the gold labels")
    extract.add_argument("--format", dest="quant_format", default=None)
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument("--tenant", default=None)

    deploy = sub.add_parser("deploy-endpoint", help="point the serving endpoint at a promoted version")
    deploy.add_argument("--model", required=True)
    deploy.add_argument("--dry-run", dest="dry_run", action="store_true")

    rollback = sub.add_parser("rollback-endpoint", help="restore the previously promoted version")
    rollback.add_argument("--dry-run", dest="dry_run", action="store_true")

    freeze = sub.add_parser(
        "freeze-eval-set",
        help="copy a corpus's test split into its tenant's frozen golden eval set (once per tenant)",
    )
    freeze.add_argument("--corpus", default=None, help="the corpus version whose test split to freeze")
    freeze.add_argument("--tenant", default=None,
                        help="the tenant whose corpus is frozen into golden-eval-set/{tenant}/")
    freeze.add_argument(
        "--allow-small", dest="allow_small", action="store_true",
        help="freeze even when a document type has fewer than 150 test documents (a pilot)",
    )
    freeze.add_argument(
        "--mark-held-out", dest="mark_held_out", action="store_true",
        help="in the tenant's EXISTING frozen set, mark the documents the delivery calls unseen "
             "carriers' (re-import the batch first); nothing else in the set changes",
    )

    return parser


#: Commands that finish in seconds and so run in the foreground even on the pod.
QUICK_COMMANDS = frozenset({"deploy-endpoint", "rollback-endpoint"})


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    from common.logging_setup import setup as setup_logging

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # On the pod, every long command runs detached in tmux, so shutting the laptop
    # or losing Wi-Fi cannot stop it. Flipping the endpoint takes seconds.
    if args.command not in QUICK_COMMANDS:
        from orchestration.detach import detach_module_if_needed

        if detach_module_if_needed("orchestration.run", argv, hint=args.command):
            return 0

    if args.command == "extract":
        return run_extract(args)

    if args.command == "freeze-eval-set":
        from evaluation.freeze_eval_set import freeze_eval_set, mark_held_out
        from registry_utils.write_run_manifest import capture_git_commit

        if args.mark_held_out:
            marked = mark_held_out(BlobClient(), tenant_id=args.tenant)
            print(json.dumps({"held_out_source_ids": marked}, indent=2))
            return 0
        if not args.corpus:
            build_parser().error("freeze-eval-set needs --corpus (or --mark-held-out on a frozen set)")
        manifest = freeze_eval_set(
            BlobClient(), args.corpus, tenant_id=args.tenant, git_commit=capture_git_commit(),
            allow_small=args.allow_small,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    if args.command in ("deploy-endpoint", "rollback-endpoint"):
        controller = RunPodController()
        if args.command == "deploy-endpoint":
            controller.deploy_endpoint(args.model, dry_run=args.dry_run)
        else:
            controller.rollback_endpoint(dry_run=args.dry_run)
        return 0

    if args.command == "all" and not getattr(args, "version", None):
        args.version = args.out_version

    report = run_scopes(args, args.command)

    print(report.render())
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
