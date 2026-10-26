"""The operator command surface (IMPL-13 §1).

One fine-tuning cycle trains the policy adapter (``common.adapter``) and takes
the next version number, which names its run, adapter and release::

    python -m orchestration.run all      --input ./intake          # the next version, e.g. v3
    python -m orchestration.run finetune --input ./intake          # stages 1-6 of a new version
    python -m orchestration.run package  --version v3              # stages 7-9: the release, gated
    python -m orchestration.run shadow   --version v3 --model-url URL --mineru-url URL  # on the server path
    python -m orchestration.run promote  --version v3 --by "Name"  # needs v3's passing shadow report
    python -m orchestration.run promote  --version v2 --by "Name"  # a rollback to v2 (no shadow)
    python -m orchestration.run versions                           # the registry, per version
    python -m orchestration.run extract  --model v2 --split test --corpus v1
    python -m orchestration.run extract  --model base --adapter ./checkpoint-400 --source-ids policy_0001

A fresh ``finetune`` or ``all`` takes the next version itself; ``--version``
names one only to resume a cycle (``--from-stage``) or rerun one that failed.

``package`` never promotes (decision 2026-10-26, #44): the release is left
``gated`` and the run summary prints the next steps. ``shadow`` reads it on the
real server path beside the promoted release (``orchestration.shadow``), and
``promote`` serves it once that passed - or with ``--skip-shadow --reason``,
recorded with the person's name; an interim-tier release also needs
``--accept-interim``.

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
from typing import TYPE_CHECKING

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, BlobError, for_ingestion
from common.adapter import ADAPTER
from common.constants import DAY_ZERO_MIN_LABELS_PER_TYPE
from orchestration import pipeline_dag
from orchestration.pipeline_dag import (
    STAGE_BY_NAME,
    RunReport,
    StageContext,
    run_stages,
    stages_for,
    stages_from,
)
from orchestration.runpod_controller import RunPodController
from orchestration.settings import backoff_seconds, defaults, retry_policy

if TYPE_CHECKING:
    from evaluation.gating import GateOverrideRecord

log = logging.getLogger(__name__)


def build_context(args: argparse.Namespace, *, client: BlobClient | None = None,
                  controller: RunPodController | None = None, **over: object) -> StageContext:
    """Turn parsed arguments into the one object the stages share."""
    from registry_utils.write_run_manifest import capture_git_commit

    commit = getattr(args, "commit", None) or capture_git_commit()
    controller = controller or RunPodController(git_commit=commit)

    kwargs: dict[str, object] = {
        "client": client or BlobClient(),
        # Only ingestion and OCR may reach raw-documents/; every other stage
        # works from processed/ (arch §18a).
        "raw_client": None if client else for_ingestion(),
        "controller": controller,
        "out_version": getattr(args, "version", None) or "",
        "corpus_version": getattr(args, "corpus_version", "") or "",
        "input_dir": Path(args.input) if getattr(args, "input", None) else None,
        "doc_types": list(ADAPTER.doc_types),
        "tenant_id": getattr(args, "tenant", None),
        "fresh": getattr(args, "fresh", False),
        "gpu_class": getattr(args, "gpu", None),
        "dry_run": getattr(args, "dry_run", False),
        "skip_ingest": getattr(args, "skip_ingest", False),
        "ocr_workers": int(getattr(args, "ocr_workers", None) or defaults().get("ocr_workers", 1) or 1),
        "allow_dirty": getattr(args, "allow_dirty", False),
        "train_vit": getattr(args, "train_vit", False),
        "min_labels_per_type": getattr(args, "min_labels_per_type", DAY_ZERO_MIN_LABELS_PER_TYPE),
        "push_adapters": getattr(args, "push_adapters", False),
        "keep_staging": getattr(args, "keep_staging", False),
        "from_blob": getattr(args, "from_blob", False),
        "fixes_defect": fixes_defect_for(getattr(args, "fixes_defect", None)),
        "gate_override": gate_override_from(args),
        "git_commit": commit,
        "max_attempts": int(retry_policy()["max_attempts"]),
        "retry_backoff_seconds": backoff_seconds(),
    }
    kwargs.update(over)
    return StageContext(**kwargs)  # type: ignore[arg-type]


class FixesDefectError(ValueError):
    """Raised when --fixes-defect names no defect."""


def fixes_defect_for(given: str | None) -> str | None:
    """``--fixes-defect``, checked, or ``None``.

    It lifts the improvement rule for this release alone. Checked before any
    work: a blank or placeholder value (``" "``, ``-``) is refused
    (``evaluation.gating.documented_defect``).
    """
    from evaluation.gating import GateError, documented_defect

    if given is None:
        return None
    try:
        return documented_defect(given)
    except GateError as exc:
        raise FixesDefectError(f"--fixes-defect: {exc}") from exc


class GateOverrideError(ValueError):
    """Raised when the --override-* flags do not make a valid written override."""


def gate_override_from(args: argparse.Namespace) -> GateOverrideRecord | None:
    """The written gate override (arch §15.5) the ``--override-*`` flags name, or ``None``.

    The one way past a gate that failed: a named person, a written reason and
    the gates it waives, all three or none, validated before any work exactly
    as the gate validates it (``GateOverrideRecord.validate``, which also refuses
    a name that is not a gate), and recorded in the gate decision and the
    release bundle as waived, never as passed. It lifts only the named gates
    that failed.
    """
    approver = getattr(args, "override_approver", None)
    reason = getattr(args, "override_reason", None)
    gates = [g.strip() for g in getattr(args, "override_gates", None) or [] if g.strip()]
    if approver is None and reason is None and not gates:
        return None
    from evaluation.gating import GateError, GateOverrideRecord

    override = GateOverrideRecord(
        approver=(approver or "").strip(), reason=(reason or "").strip(), waived_gates=gates,
    )
    try:
        override.validate()
    except GateError as exc:
        raise GateOverrideError(
            f"{exc} An override needs --override-approver, --override-reason and at least one "
            "--override-gate."
        ) from exc
    return override


def _selected_stages(command: str, from_stage: str | None) -> Sequence[pipeline_dag.Stage]:
    if not from_stage:
        return stages_for(command)
    # `--from-stage` on `all` still spans both commands: resuming at `checkpoint_eval`
    # should carry on into packaging, not stop at the end of finetune.
    return stages_from(from_stage, command)


def run_command(command: str, ctx: StageContext, *, from_stage: str | None = None) -> RunReport:
    """Run one build command. ``all`` is the two commands back to back."""
    return run_stages(ctx, _selected_stages(command, from_stage), command=command)


class VersionError(ValueError):
    """Raised when a cycle's version cannot be decided."""


def cycle_version(args: argparse.Namespace, client: BlobClient) -> str:
    """The version this command works on.

    ``--version`` when given. A fresh ``finetune`` or ``all`` without one takes
    the next version (``registry_utils.releases.next_cycle_version``); a resume
    (``--from-stage``) must name its version, because the next free number
    after a first attempt that wrote anything is not that attempt's.
    """
    from registry_utils.releases import next_cycle_version

    given = getattr(args, "version", None)
    if given:
        return str(given)
    if getattr(args, "from_stage", None):
        raise VersionError(
            "--from-stage resumes a cycle: name its version with --version (the next free "
            "number would be a new cycle)."
        )
    version = next_cycle_version(client, getattr(args, "tenant", None))
    log.info("this cycle is %s (the next version)", version)
    return version


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
    parser.add_argument("--corpus-version", dest="corpus_version", default="",
                        help="train on a corpus already built; defaults to the cycle's version")
    parser.add_argument("--version", default=None,
                        help="the cycle's version (vN); a fresh cycle takes the next one itself. "
                             "Name it to resume (--from-stage) or rerun a failed cycle")
    parser.add_argument("--gpu", default=None, help="GPU class override for the training pod")
    parser.add_argument("--commit", default=None, help="repo commit the pod clones; defaults to HEAD")
    parser.add_argument("--from-stage", dest="from_stage", default=None,
                        help="resume at this stage; earlier stages are not re-run")
    parser.add_argument("--skip-ingest", dest="skip_ingest", action="store_true")
    parser.add_argument("--ocr-workers", dest="ocr_workers", type=int, default=None,
                        help="documents the OCR stage reads at once against the served MinerU "
                             "($MINERU_URL); default pipeline.yaml defaults.ocr_workers (1). 4 is a good "
                             "start against a dedicated MinerU 4.0.12/basic")
    parser.add_argument("--fresh", action="store_true",
                        help="train from the base model, not the promoted adapter (after a data "
                             "deletion, or to reset drift); a cycle continues the promoted one "
                             "by default")
    parser.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                        help="train on a working tree with uncommitted changes (a real training run refuses "
                             "one otherwise: the manifest's commit would not be the code that ran). Same as "
                             "FIDEON_ALLOW_DIRTY_TREE=1")
    parser.add_argument("--train-vit", dest="train_vit", action="store_true",
                        help="LoRA on the vision encoder; never a full fine-tune (arch §3)")
    parser.add_argument("--min-labels-per-type", dest="min_labels_per_type", type=int,
                        default=defaults().get("min_labels_per_type", DAY_ZERO_MIN_LABELS_PER_TYPE))
    parser.add_argument("--push-adapters", dest="push_adapters", action="store_true",
                        help="also push the adapter to Blob as soon as it is trained")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="assemble and record everything without launching GPU work")


def _add_package_flags(parser: argparse.ArgumentParser, *, version_required: bool = True) -> None:
    if version_required:
        parser.add_argument("--version", required=True)
    parser.add_argument("--keep-staging", dest="keep_staging", action="store_true")
    parser.add_argument("--from-blob", dest="from_blob", action="store_true",
                        help="artifacts were pushed with --push-adapters; read them from Blob")
    # Satisfies the gate's improvement rule only (arch §15.5): floors and
    # non-inferiority still apply, and the text is recorded in the decision.
    parser.add_argument("--fixes-defect", dest="fixes_defect", default=None,
                        help="the documented defect this release fixes (ticket or sentence); "
                             "stands in for 'a primary metric improved' and nothing else")
    # The written override of arch §15.5, never a bypass: all three together,
    # validated before any work, recorded as waived in the decision and bundle.
    parser.add_argument("--override-approver", dest="override_approver", default=None,
                        help="written gate override: the person accountable for it (not a "
                             "system identity); needs --override-reason and --override-gate")
    parser.add_argument("--override-reason", dest="override_reason", default=None,
                        help="written gate override: why this ships despite the gate "
                             "(20 characters or more)")
    parser.add_argument("--override-gate", dest="override_gates", action="append", default=[],
                        help="written gate override: a failed gate to waive, as the gate decision "
                             "names it (repeatable); only gates that failed are lifted")
    if version_required:
        # Standalone `package` only — under `all` these come from the finetune
        # flags and adding them twice is an argparse conflict.
        #
        # `finetune` decides which models get built; `package` publishes their
        # locations, so it reads the version's own run manifest and corpus.
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

    _add_finetune_flags(sub.add_parser(
        "finetune", help="stages 1-6 of a cycle: ingest through checkpoint selection; the adapter "
                         "is staged. Continues the promoted adapter unless --fresh"))

    package = sub.add_parser(
        "package", help="stages 7-9: calibrate, gate, push the adapter and write the release, gated "
                        "(never promoted: shadow, then promote)")
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

    shadow = sub.add_parser(
        "shadow", help="read a gated release on the real server path (fields first, the pod's vLLM and "
                       "MinerU 4) beside the promoted one, score both against the frozen golds and write "
                       "shadow_report.json; promote asks for a passing one")
    shadow.add_argument("--version", required=True, help="the gated release to read, e.g. v3")
    shadow.add_argument("--model-url", dest="model_url", required=True,
                        help="the pod's vLLM (OpenAI API), serving policy-vN and the promoted policy-vK "
                             "together (--max-loras 2)")
    shadow.add_argument("--mineru-url", dest="mineru_url", required=True, help="the served MinerU 4 API")
    shadow.add_argument("--documents", type=int, default=None,
                        help="frozen eval documents to read (default 50): stratified by line, real first")
    shadow.add_argument("--tenant", default=None)

    promote = sub.add_parser(
        "promote", help="make a version the tenant's promoted release; the one promoted now is "
                        "archived. A gated release needs its passing shadow report (or --skip-shadow "
                        "--reason); a rollback to one promoted before does not")
    promote.add_argument("--version", required=True, help="the version to serve, e.g. v2")
    promote.add_argument("--by", dest="promoted_by", required=True,
                         help="the person accountable for the change (not a system identity)")
    promote.add_argument("--skip-shadow", dest="skip_shadow", action="store_true",
                         help="promote a gated release without a passing shadow report; needs --reason, "
                              "recorded in the release with --by")
    promote.add_argument("--reason", dest="skip_reason", default=None,
                         help="with --skip-shadow: why, in 20 characters or more")
    promote.add_argument("--accept-interim", dest="accept_interim", action="store_true",
                         help="promote an interim-tier release (field match below the production bar, or "
                              "an eval set accepted small) knowingly")
    promote.add_argument("--tenant", default=None)

    noise = sub.add_parser(
        "label-noise",
        help="read a release's train documents with its adapter and queue the golds that lack a value "
             "the page prints, for a person to review (GPU; no gold is changed)")
    noise.add_argument("--version", required=True, help="the release whose adapter reads, e.g. v4")
    noise.add_argument("--corpus-version", dest="corpus_version", default=None,
                       help="the corpus whose train documents are read; default the one the version trained on")
    noise.add_argument("--limit", type=int, default=None, help="read at most this many train documents")
    noise.add_argument("--tenant", default=None)

    versions = sub.add_parser("versions", help="the registry: every version, its run and release")
    versions.add_argument("--tenant", default=None)

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
    freeze.add_argument(
        "--mark-reviewed", dest="mark_reviewed", action="store_true",
        help="in the tenant's EXISTING frozen set, copy each document's review record from its label "
             "metadata (the gate refuses a frozen gold without one); nothing else in the set changes",
    )

    return parser


#: Commands that finish in seconds and so run in the foreground even on the pod.
QUICK_COMMANDS = frozenset({"deploy-endpoint", "rollback-endpoint", "promote", "versions"})


def flag_problem(args: argparse.Namespace) -> str | None:
    """Why these flags cannot make a run; ``None`` when they can.

    Checked by :func:`main` before the command detaches on the pod. While only
    :func:`build_context` checked them, after the detach, a refused
    ``--fixes-defect`` or ``--override-*`` surfaced inside tmux with the
    operator's command already exited 0, and off the pod as a traceback.
    ``--fixes-defect`` and the written override are read here exactly as
    ``build_context`` reads them; ``extract``'s flags as
    ``testing.run_extraction`` refuses them.
    """
    if args.command == "freeze-eval-set":
        marks = [flag for flag, given in (("--mark-held-out", args.mark_held_out),
                                          ("--mark-reviewed", args.mark_reviewed)) if given]
        if marks and (args.corpus or args.allow_small):
            # Given together, the freeze flags were dropped without a word.
            return (f"freeze-eval-set {' '.join(marks)} changes the tenant's EXISTING frozen set "
                    "and freezes nothing, so it takes no --corpus or --allow-small")
        if not marks and not args.corpus:
            return "freeze-eval-set needs --corpus (or --mark-held-out / --mark-reviewed on a frozen set)"
        return None
    if args.command == "promote":
        from registry_utils.releases import SKIP_SHADOW_MIN_CHARS, is_named_person

        if not is_named_person(args.promoted_by):
            return "promote --by must name the person accountable for it, not a system identity"
        reason = (args.skip_reason or "").strip()
        if args.skip_shadow and len(reason) < SKIP_SHADOW_MIN_CHARS:
            return (f"promote --skip-shadow needs --reason \"<why>\", {SKIP_SHADOW_MIN_CHARS} characters or "
                    "more: it is recorded in the release with --by")
        if args.skip_reason is not None and not args.skip_shadow:
            return "promote --reason is the written reason for --skip-shadow; give both or neither"
        return None
    if args.command == "shadow":
        return shadow_flag_problem(args)
    if args.command == "extract":
        # testing.run_extraction refuses these too, but only once it runs: inside
        # tmux on the pod, with this command already exited 0. Its words, here.
        if args.split and not args.corpus:
            return "extract --split needs --corpus"
        if args.adapter and args.model != "base":
            return "extract --adapter applies a LoRA to the base model: use --model base with it"
        return None
    if args.command == "label-noise":
        if args.limit is not None and args.limit < 1:
            return "label-noise --limit is 1 or more"
        if paths.version_number(str(args.version)) is None:
            return f"label-noise --version names a release, vN; got {args.version!r}"
        return None
    if getattr(args, "ocr_workers", None) is not None and args.ocr_workers < 1:
        return "--ocr-workers is 1 or more"
    if args.command not in ("finetune", "package", "all"):
        return None
    from_stage = getattr(args, "from_stage", None)
    if from_stage and (problem := from_stage_problem(args.command, from_stage)):
        return problem
    if from_stage and not getattr(args, "version", None):
        return ("--from-stage resumes a cycle: name its version with --version (the next free "
                "number would be a new cycle)")
    if getattr(args, "fresh", False) and from_stage and (
            STAGE_BY_NAME[from_stage].number > STAGE_BY_NAME["training"].number):
        # Training is not in the stages run, so --fresh would change nothing and
        # the adapter already trained would go through the gate as it is.
        return (f"--fresh decides where training starts, and --from-stage {from_stage} resumes "
                f"after training: nothing would train from the base, and {args.version}'s adapter "
                "would go through the gate as it trained. --fresh applies to a new cycle: run "
                "finetune/all without --version")
    try:
        fixes_defect_for(getattr(args, "fixes_defect", None))
        gate_override_from(args)
    except (FixesDefectError, GateOverrideError) as exc:
        return str(exc)
    return None


def shadow_flag_problem(args: argparse.Namespace) -> str | None:
    """Why ``shadow``'s flags cannot make a run; ``None`` when they can. Before
    the detach, from the flags alone: the release itself is checked by the
    run (``orchestration.shadow.run_shadow``), first thing."""
    from common.transport import InsecureUrlError, assert_encrypted

    if paths.version_number(str(args.version)) is None:
        return f"shadow --version names a release, vN; got {args.version!r}"
    if args.documents is not None and args.documents < 1:
        return "shadow --documents is 1 or more"
    try:
        assert_encrypted(args.model_url, "the model server")
        assert_encrypted(args.mineru_url, "the MinerU server")
    except InsecureUrlError as exc:
        return f"shadow: {exc}"
    return None


def run_shadow_command(args: argparse.Namespace, *, client: BlobClient | None = None) -> int:
    """``shadow``: the check, its table, and 0 when it passed."""
    from orchestration.shadow import SHADOW_DOCUMENTS, ShadowError, render, run_shadow

    try:
        report = run_shadow(client or BlobClient(), args.version, model_url=args.model_url,
                            mineru_url=args.mineru_url, documents=args.documents or SHADOW_DOCUMENTS,
                            tenant_id=args.tenant)
    except ShadowError as exc:
        print(f"shadow {args.version}: {exc}", file=sys.stderr)
        return 1
    print(render(report))
    if report["passed"]:
        tenant = f" --tenant {args.tenant}" if args.tenant else ""
        interim = " --accept-interim" if (report.get("candidate") or {}).get("tier") == "interim" else ""
        print(f"  next: python -m orchestration.run promote --version {args.version} --by \"<person>\""
              f"{tenant}{interim}")
    return 0 if report["passed"] else 1


def promotion_evidence(client: BlobClient, version: str, tenant: str | None) -> str | None:
    """What the release's last promotion rests on, as recorded in its bundle."""
    key = paths.release_bundle(version, tenant)
    promotions = (client.read_json(key).get("promotions") or []) if client.exists(key) else []
    if not promotions:
        return None
    last = promotions[-1]
    if last.get("rollback"):
        how = "a rollback to a release promoted before (no shadow check asked)"
    elif last.get("shadow_report"):
        how = f"shadow report {last['shadow_report']}"
    else:
        how = f"shadow check skipped by {last.get('promoted_by')}: {last.get('shadow_skipped_reason')}"
    tier = f"; tier {last['tier']}" if last.get("tier") else ""
    accepted = " (accepted as interim)" if last.get("accepted_interim") else ""
    return f"promoted by {last.get('promoted_by')} at {last.get('promoted_at')} on {how}{tier}{accepted}"


#: Stages removed from the pipeline, and why. Named by ``--from-stage`` they are
#: refused with the reason, before the command detaches.
REMOVED_STAGES: dict[str, str] = {
    "merge": "removed on 2026-10-23: nothing is merged or quantized; the release is its adapter",
    "quantize": "removed on 2026-10-23: nothing is merged or quantized; the release is its adapter",
}


def from_stage_problem(command: str, stage: str) -> str | None:
    """Why ``--from-stage stage`` cannot resume ``command``; ``None`` when it can.

    Checked by :func:`flag_problem`, before the detach: ``stages_from`` refuses
    the same name, but only inside the detached run, with the operator's command
    already exited 0 and the traceback in tmux.
    """
    names = [s.name for s in stages_for(command)]
    if stage in names:
        return None
    if stage in REMOVED_STAGES:
        return f"--from-stage {stage}: the {stage} stage was {REMOVED_STAGES[stage]}. Resume at one of {names}"
    return f"--from-stage {stage!r} is not a stage of {command}; expected one of {names}"


#: Run statuses that mean the run's training finished (``training`` is still
#: running or was interrupted, ``failed`` never finished).
TRAINED_STATUSES = frozenset({"trained", "evaluated", "promoted", "archived"})


def fresh_problem(args: argparse.Namespace, client: BlobClient | None = None) -> str | None:
    """Why ``--fresh`` cannot apply to the version named; ``None`` when it can.

    ``--fresh`` decides where training starts, so it means nothing for a version
    that already trained: training is skipped as complete, and the adapter it
    trained - continued from the promoted one, as a cycle is by default - goes
    through the gate as if it had started from the base. Read from the
    version's run manifest (and, on the pod, the training stage's completion
    marker, as ``pipeline_dag._is_trained`` reads it), by :func:`main` before any
    work and before the command detaches. Kept out of :func:`flag_problem`,
    which reads the flags alone.
    """
    version = getattr(args, "version", None)
    if (args.command not in ("finetune", "all") or not getattr(args, "fresh", False)
            or not version or paths.version_number(str(version)) is None):
        return None
    from registry_utils.query_registry import RegistryQueryError, get
    from training.train import training_completed

    run_id = ADAPTER.run_id(str(version))
    try:
        manifest = get(run_id, client or BlobClient())
    except RegistryQueryError as exc:
        if isinstance(exc.__cause__, BlobError):
            return f"--fresh with --version {version}: cannot read whether it already trained ({exc})"
        manifest = None  # no run of it yet: its first attempt has not trained
    except ValueError as exc:
        return f"--fresh with --version {version}: its run manifest {run_id} cannot be read ({exc})"
    trained = manifest is not None and (
        manifest.status in TRAINED_STATUSES or manifest.artifacts.status == "published")
    if not trained and not training_completed(paths.scoped_staging_adapter_dir(ADAPTER.name, str(version))):
        return None
    if manifest is None:
        how = "its training stage is complete on the staging volume"
    elif manifest.parent_run_id or manifest.continued_from:
        how = f"continued from {manifest.parent_run_id or manifest.continued_from}"
    else:
        how = "from the base" + (f": {manifest.fresh_reason}" if manifest.fresh_reason else "")
    return (f"{version} already trained ({how}); --fresh applies to a new cycle: run finetune/all "
            f"without --version. To resume {version} as it trained, drop --fresh")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    from common.logging_setup import setup as setup_logging

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Before the detach below: a refusal after it is seen by nobody.
    if problem := flag_problem(args):
        parser.error(problem)
    if problem := fresh_problem(args):
        parser.error(problem)

    # On the pod, every long command runs detached in tmux, so shutting the laptop
    # or losing Wi-Fi cannot stop it. Flipping the endpoint takes seconds.
    if args.command not in QUICK_COMMANDS:
        from orchestration.detach import detach_module_if_needed

        if detach_module_if_needed("orchestration.run", argv, hint=args.command):
            return 0

    if args.command == "extract":
        return run_extract(args)

    if args.command == "freeze-eval-set":
        from evaluation.freeze_eval_set import freeze_eval_set, mark_held_out, mark_reviewed
        from registry_utils.write_run_manifest import capture_git_commit

        if args.mark_held_out or args.mark_reviewed:
            # Each one named runs: given together, --mark-reviewed was skipped
            # without a word.
            marked: dict[str, object] = {}
            if args.mark_held_out:
                marked["held_out_source_ids"] = mark_held_out(BlobClient(), tenant_id=args.tenant)
            missing: dict[str, str] = {}
            if args.mark_reviewed:
                missing = mark_reviewed(BlobClient(), tenant_id=args.tenant)
                marked["still_without_review_record"] = missing
            print(json.dumps(marked, indent=2, ensure_ascii=False))
            # Non-zero while any frozen gold still has no record: the gate refuses the set.
            return 1 if missing else 0
        manifest = freeze_eval_set(
            BlobClient(), args.corpus, tenant_id=args.tenant, git_commit=capture_git_commit(),
            allow_small=args.allow_small,
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    if args.command == "promote":
        from registry_utils.releases import ReleaseError, promote

        client = BlobClient()
        try:
            archived = promote(client, args.version, args.tenant, promoted_by=args.promoted_by.strip(),
                               skip_shadow_reason=args.skip_reason if args.skip_shadow else None,
                               accept_interim=args.accept_interim)
        except ReleaseError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"{args.version} is promoted" + (f"; archived {', '.join(archived)}" if archived else ""))
        if evidence := promotion_evidence(client, args.version, args.tenant):
            print(f"  {evidence}")
        return 0

    if args.command == "shadow":
        return run_shadow_command(args)

    if args.command == "label-noise":
        from orchestration.label_noise import LabelNoiseError, main_from_args

        try:
            summary = main_from_args(args)
        except LabelNoiseError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "versions":
        from registry_utils.releases import version_table

        print(json.dumps(version_table(BlobClient(), args.tenant), indent=2, ensure_ascii=False))
        return 0

    if args.command in ("deploy-endpoint", "rollback-endpoint"):
        controller = RunPodController()
        if args.command == "deploy-endpoint":
            controller.deploy_endpoint(args.model, dry_run=args.dry_run)
        else:
            controller.rollback_endpoint(dry_run=args.dry_run)
        return 0

    client = BlobClient()
    try:
        args.version = cycle_version(args, client)
    except VersionError as exc:
        parser.error(str(exc))
    # The raw-documents login too: only ingestion and OCR may reach raw-documents/
    # (arch §18a), and an injected client would otherwise stand in for both.
    ctx = build_context(args, client=client, raw_client=for_ingestion())
    report = run_command(args.command, ctx, from_stage=getattr(args, "from_stage", None))
    # One post saying how the command ended (orchestration.notify; a no-op without a webhook).
    from orchestration.notify import run_complete

    run_complete(report, ctx)


    print(report.render())
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
