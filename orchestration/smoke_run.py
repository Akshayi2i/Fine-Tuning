"""A small end-to-end run on the pod, before the real one: does the whole path work?

    MINERU_URL=https://... python -m orchestration.smoke_run --batch c1   # every step
    python -m orchestration.smoke_run --batch c1 --steps check,finetune

It takes a handful of source documents from the uploaded batch (``--batch``,
required: the default was an old batch, and a run without it smoke-tested other
data) — each with all of its synthetic twins, so the delivered split stays
intact (``--train-sources``, ``--val-sources``, ``--test-sources``: 4, 1 and 1
by default), and only families of at most ``--max-pages`` pages (40 by default:
on the first commercial set the alphabetical pick held two 186-page policies,
about 380 training steps, some 5 hours) — and runs them
through import, OCR, the post-OCR check and ``finetune``, under their own
tenant (``smoke``) and version (``v0``). Nothing it writes touches the real
data: corpus, labels and OCR output are tenant-scoped, and the model versions
start at ``v1``. It never freezes an eval set or packages a release: a smoke
subset is no yardstick. It trains the policy adapter (``common.adapter``), as a
real cycle does.

What it proves: MinerU runs on the GPU, the labels survive import, the corpus
builds with the delivered split, ms-swift trains, a checkpoint is chosen and
merged. What it measures: how many rows exceed the sequence caps (the corpus
build reports rejections) and how long a step takes. Its scores mean nothing.

Steps: ``select`` (pick the documents), ``import``, ``ocr`` (in the OCR
environment), ``check`` (post-OCR values), ``preflight`` (everything finetune
depends on, the corpus built in memory), ``finetune``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

STEPS = ("select", "import", "ocr", "check", "preflight", "finetune")
DEFAULT_TENANT = "smoke"
DEFAULT_VERSION = "v0"
WORKSPACE = Path(os.environ.get("FIDEON_WORKSPACE") or "/workspace")  # an empty .env value means unset
OCR_PYTHON = WORKSPACE / "venv-ocr" / "bin" / "python"


class SmokeError(RuntimeError):
    """Raised when the smoke run cannot be set up."""


def select_sources(bundles: Path, *, train_sources: int = 4, val_sources: int = 1,
                   test_sources: int = 1, max_pages: int | None = None) -> list[Path]:
    """Folders of a few whole source families, spread across lines, split kept.

    A source is taken with every twin it has, so no family is cut in two, and
    train sources are drawn from as many different lines as there are. With
    ``max_pages``, only a family whose PDFs hold at most that many pages in all
    is taken: training time follows pages, not documents.
    """
    families: dict[tuple[str, str], list[Path]] = defaultdict(list)
    line_of: dict[tuple[str, str], str] = {}
    for meta_path in sorted(bundles.glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        split = str(meta.get("split") or "").lower()
        if split not in ("train", "val", "test"):
            raise SmokeError(f"{meta_path.parent.name} has no delivered split; the smoke run needs one")
        key = (split, str(meta.get("template_id") or meta_path.parent.name))
        families[key].append(meta_path.parent)
        line_of[key] = str(meta.get("lob"))
    if max_pages:
        families = {key: folders for key, folders in families.items()
                    if sum(_page_count(folder / "document.pdf") for folder in folders) <= max_pages}
    chosen: list[Path] = []
    for split, count in (("train", train_sources), ("val", val_sources), ("test", test_sources)):
        keys = sorted(k for k in families if k[0] == split)
        if len(keys) < count:
            raise SmokeError(f"the batch has {len(keys)} {split} source(s)"
                             f"{f' of at most {max_pages} pages' if max_pages else ''}; the smoke run "
                             f"needs {count}" + ("; raise --max-pages" if max_pages else ""))
        picked: list[tuple[str, str]] = []
        seen_lines: set[str] = set()
        for key in keys:                                    # one per line first
            if len(picked) < count and line_of[key] not in seen_lines:
                picked.append(key)
                seen_lines.add(line_of[key])
        for key in keys:                                    # then fill up
            if len(picked) < count and key not in picked:
                picked.append(key)
        for key in picked:
            chosen += families[key]
    return chosen


def _page_count(pdf: Path) -> int:
    import pymupdf

    from common.pdf_access import unlocked_pdf

    with pymupdf.open(stream=unlocked_pdf(pdf.read_bytes()), filetype="pdf") as doc:
        return doc.page_count


def assert_served_ocr(steps: list[str], environ: dict[str, str] | None = None) -> None:
    """Refuse an OCR step with no served MinerU named (``MINERU_URL``).

    Without it the step reads with the OCR environment's MinerU, whose reading
    the corpus build and preflight refuse unless it is the pinned version
    (configs/shared/ocr.yaml) - found only after every page was read.
    """
    from orchestration.ocr_server import installed

    environ = os.environ if environ is None else environ
    # With the pod's own server installed (scripts/pod_bootstrap.sh) the OCR step starts it
    # itself, and stops it after, so no address needs to be set.
    if "ocr" in steps and not (environ.get("MINERU_URL") or "").strip() and not installed():
        raise SmokeError("MINERU_URL is not set: the OCR step would read with the OCR environment's "
                         "MinerU, not the pinned served one, and preflight would refuse every reading "
                         "after the whole subset was read. Set MINERU_URL to the served MinerU.")


def assert_fresh(client, tenant: str, version: str, steps: list[str]) -> None:
    """Refuse a smoke run that would land on an earlier one's leftovers.

    Labels, OCR output and the corpus are kept per tenant, and the run registry
    per version. A second smoke run under the same names imported its documents
    beside the first run's, then found the corpus "already built" and the
    training "already complete", re-merged the OLD adapter and reported success
    without training on the new batch at all.

    Checked only when the run starts from the beginning (``select``): resuming a
    failed run is the same run, and is allowed - from ``import`` too, which skips
    the documents it already imported (same checksum). Checked at ``import``, the
    resume this script prints after a failed import was refused whenever any
    document had gone in before the failure.
    """
    from artifact_registry import paths
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    if "select" not in steps:
        return
    held = list_labeled_source_ids(client, "policy", tenant)
    problems = []
    if held:
        problems.append(f"tenant {tenant!r} already holds {len(held)} imported document(s)")
    if client.exists(paths.corpus_manifest(version, tenant)):
        problems.append(f"corpus {version} already exists for tenant {tenant!r}")
    if "finetune" in steps:
        from common.adapter import ADAPTER
        from registry_utils.query_registry import RegistryQueryError
        from registry_utils.query_registry import get as get_manifest

        run_id = ADAPTER.run_id(version)
        try:
            status = get_manifest(run_id, client).status
        except (RegistryQueryError, KeyError, FileNotFoundError):
            status = None
        if status is not None:
            problems.append(f"run {run_id} already exists (status {status})")
    if problems:
        raise SmokeError(
            "; ".join(problems) + ". This smoke run would reuse that state instead of testing the "
            "new batch. Start it under names nothing has used: --tenant smoke2 --version v0.1 "
            "(any unused pair). To carry on a smoke run that stopped part-way, leave out "
            "select: --steps import,ocr,check,preflight,finetune (or from the step that failed)."
        )


def stage_subset(folders: list[Path], out: Path) -> int:
    """Link (or copy) the chosen folders into ``out``, replacing an earlier subset."""
    if out.exists():
        shutil.rmtree(out)
    for folder in folders:
        dest = out / folder.name
        dest.mkdir(parents=True)
        for item in folder.iterdir():
            try:
                os.link(item, dest / item.name)
            except OSError:
                shutil.copy2(item, dest / item.name)
    return len(folders)


def commands(*, batch_dir: Path, subset_dir: Path, tenant: str, version: str,
             check_out: Path, steps: tuple[str, ...] | list[str] = STEPS) -> dict[str, list[str]]:
    """The command each step runs, with the interpreter of its environment.

    When ``finetune`` runs too, preflight skips its in-memory corpus build:
    finetune builds the same corpus and refuses the same >25% set-aside before
    anything trains, so the corpus is built once. A preflight run on its own
    keeps the build - there it is the point.
    """
    py = sys.executable
    # Through the served MinerU when MINERU_URL names one - the OCR pin the corpus
    # build and promotion hold every reading to (configs/shared/ocr.yaml) - from
    # this environment, several documents at once. Otherwise the OCR
    # environment's MinerU, whose reading the corpus build refuses unless it is
    # the pinned version.
    served = (os.environ.get("MINERU_URL") or "").strip()
    ocr = ([py, "-m", "data_pipeline.ocr.run_mineru", "--doc-type", "policy", "--all-unprocessed",
            "--tenant", tenant, "--mineru-url", served, "--workers", "4"] if served else
           [str(OCR_PYTHON), "-m", "data_pipeline.ocr.run_mineru", "--doc-type", "policy",
            "--all-unprocessed", "--tenant", tenant])
    return {
        "import": [py, "-m", "data_pipeline.ingestion.import_labeled_pdfs", "--input", str(subset_dir),
                   "--doc-type", "policy", "--tenant", tenant],
        "ocr": ocr,
        "check": [py, "-m", "data_pipeline.ocr_check", "--doc-type", "policy", "--tenant", tenant,
                  "--out", str(check_out)],
        # --min-labels-per-type 1: the 25-document floor is for a real run. A
        # smoke subset of sources WITHOUT synthetic twins is 6 documents, and
        # both steps refused it; it only ever passed on the twins' numbers.
        "preflight": [py, "-m", "orchestration.preflight", "--tenant", tenant,
                      "--corpus-version", version, "--version", version, "--ocr-check", str(check_out),
                      "--out", str(check_out / "preflight.json"), "--min-labels-per-type", "1",
                      *(["--skip-corpus"] if "finetune" in steps else [])],
        "finetune": [py, "-m", "orchestration.run", "finetune", "--skip-ingest",
                     "--corpus-version", version, "--version", version, "--tenant", tenant,
                     "--min-labels-per-type", "1"],
    }


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """The smoke run's arguments and its steps, refused when they cannot run."""
    parser = argparse.ArgumentParser(description="A small end-to-end run on the pod")
    parser.add_argument("--batch", required=True, help="the uploaded batch under intake/")
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--train-sources", type=int, default=4)
    # One per line first, as for train: a validation and a test source of every
    # line give each line its own checkpoint and calibration documents.
    parser.add_argument("--val-sources", type=int, default=1)
    parser.add_argument("--test-sources", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=40,
                        help="take only families of at most this many pages in all (0: no limit)")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated, from {STEPS}")
    args = parser.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s) {unknown}; choose from {STEPS}")
    if args.tenant in ("", "default"):
        parser.error("the smoke run needs its own tenant, never the real data's")
    return args, steps


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over the functions above
    args, steps = parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("orchestration.smoke_run", argv, hint="smoke"):
        return 0

    from artifact_registry.blob_client import BlobClient

    try:
        assert_served_ocr(steps)
        assert_fresh(BlobClient(), args.tenant, args.version, steps)
    except SmokeError as exc:
        print(f"smoke: {exc}", flush=True)
        return 1

    batch_dir = WORKSPACE / "intake" / args.batch
    subset_dir = WORKSPACE / "intake" / f"{args.tenant}-{args.batch}"
    check_out = WORKSPACE / f"ocr_check_{args.tenant}"
    if "select" in steps:
        if not batch_dir.is_dir():
            from artifact_registry.blob_client import for_ingestion
            from data_pipeline.ingestion.pull_intake import pull_intake

            print(f"pulling {args.batch} to {batch_dir} ...", flush=True)
            report = pull_intake(for_ingestion(), args.batch, batch_dir.parent)
            print(report.describe(), flush=True)
            if report.failed:
                return 1
        try:
            folders = select_sources(batch_dir, train_sources=args.train_sources, val_sources=args.val_sources,
                                     test_sources=args.test_sources, max_pages=args.max_pages or None)
        except SmokeError as exc:
            print(f"smoke: {exc}", flush=True)
            return 1
        print(f"smoke subset: {stage_subset(folders, subset_dir)} document(s) in {subset_dir}", flush=True)
    from orchestration.ocr_server import managed_ocr_server

    def build() -> dict[str, list[str]]:
        return commands(batch_dir=batch_dir, subset_dir=subset_dir, tenant=args.tenant,
                        version=args.version, check_out=check_out, steps=steps)

    for step in build():
        if step not in steps:
            continue
        # The OCR step runs the pod's own server for its length and stops it, so the finetune
        # step has the GPU; a MINERU_URL somebody else serves is used as it is. The command is
        # built once the address is known: it reads through the served MinerU when there is one.
        with managed_ocr_server(step == "ocr"):
            command = build()[step]
            print(f"\n=== smoke: {step} ===\n{' '.join(command)}", flush=True)
            # Already in tmux: the step runs in place.
            code = subprocess.run(command, env={**os.environ, "FIDEON_DETACHED": "1"}).returncode
        if code != 0:
            print(f"smoke: {step} failed (exit {code}); fix it and resume with --steps "
                  f"{','.join(steps[steps.index(step):])}", flush=True)
            return code
    print("\nsmoke run complete. Read the corpus build's rejected-row count and the post-OCR report "
          f"({check_out}/report.md) before the real run.", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
