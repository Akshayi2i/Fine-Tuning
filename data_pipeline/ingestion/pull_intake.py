"""Pull a delivered batch from Blob onto the pod's volume, ready to audit and import.

The training data is too large for git, so it travels through Blob:

    laptop:  azcopy sync "<training data>" "https://<account>.blob.core.windows.net/<raw container>/intake/<batch>?<SAS>" --recursive
    pod:     python -m data_pipeline.ingestion.pull_intake --batch <batch>
             python -m data_pipeline.audit --input /workspace/intake/<batch>
             python -m data_pipeline.ingestion.import_labeled_pdfs --input /workspace/intake/<batch> --doc-type policy

(``import_labeled_pdfs --from-blob <batch>`` does the pull and the import in one.)

``intake/`` lives in the **raw container**: it is the delivered PDFs with their
unredacted PII, so the raw layer's access rule covers it and only an ingestion
client can read it.

**Resumable.** Each file is written to ``<name>.part`` and renamed when complete,
so an interrupted pull never leaves a truncated file under its real name, and a
re-run skips every PDF already on disk. Labels and metadata are small and are
always fetched again, so a label corrected and re-synced after an audit is never
left stale on the pod. A corrected PDF is a new document by design (raw documents
are write-once); re-pull it with ``--refresh``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: Where batches land on the pod: on the volume, which survives a pod stop.
INTAKE_DIR_ENV = "FIDEON_INTAKE_DIR"
DEFAULT_INTAKE_DIR = "/workspace/intake"

#: Parallel downloads. Blob serves each file independently; the pod's network,
#: not one connection, is then the limit.
DEFAULT_WORKERS = 16


class IntakeError(RuntimeError):
    """Raised when a batch cannot be pulled."""


@dataclass
class PullReport:
    local_dir: Path
    downloaded: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"{self.local_dir}: {self.downloaded} downloaded, {self.skipped} already here, "
                 f"{len(self.failed)} failed"]
        lines += [f"  FAILED {key}: {reason}" for key, reason in self.failed[:20]]
        if len(self.failed) > 20:
            lines.append(f"  ... and {len(self.failed) - 20} more")
        return "\n".join(lines)


def default_intake_dir() -> Path:
    return Path(os.environ.get(INTAKE_DIR_ENV) or DEFAULT_INTAKE_DIR)


def _local_path(target: Path, rel: str) -> Path:
    parts = PurePosixPath(rel).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise IntakeError(f"refusing blob name {rel!r}: it would land outside {target}")
    return target.joinpath(*parts)


def pull_intake(
    client: BlobClient,
    batch: str,
    out_dir: Path | None = None,
    *,
    workers: int = DEFAULT_WORKERS,
    refresh: bool = False,
) -> PullReport:
    """Download ``intake/{batch}/`` to ``{out_dir}/{batch}/``."""
    prefix = paths.intake_batch_dir(batch)
    boundary = prefix + "/"
    target = (out_dir or default_intake_dir()) / batch
    keys = [k for k in client.list(prefix) if k.startswith(boundary) and not k.endswith("/")]
    if not keys:
        raise IntakeError(
            f"nothing staged under {client.raw_container}/{boundary}. Upload the batch first, e.g.\n"
            f'  azcopy sync "<training data folder>" '
            f'"https://<account>.blob.core.windows.net/{client.raw_container}/{prefix}?<SAS>" --recursive'
        )
    report = PullReport(local_dir=target)

    def fetch(key: str) -> bool:
        local = _local_path(target, key[len(boundary):])
        if not refresh and local.suffix.lower() == ".pdf" and local.is_file():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + ".part")
        part.write_bytes(client.read_bytes(key))
        os.replace(part, local)
        return True

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, key): key for key in keys}
        for done, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                if future.result():
                    report.downloaded += 1
                else:
                    report.skipped += 1
            except Exception as exc:  # noqa: BLE001 - every failure is reported, none stops the rest
                report.failed.append((key, str(exc)))
            if done % 500 == 0 or done == len(keys):
                log.info("pulled %d/%d files", done, len(keys))
    return report


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from artifact_registry.blob_client import for_ingestion

    parser = argparse.ArgumentParser(description="Pull a staged intake batch from Blob onto the pod")
    parser.add_argument("--batch", required=True, help="the name under intake/ it was uploaded to")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"parent folder (default ${INTAKE_DIR_ENV} or {DEFAULT_INTAKE_DIR})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--refresh", action="store_true", help="download PDFs already on disk again")
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.ingestion.pull_intake", argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    try:
        report = pull_intake(for_ingestion(), args.batch, args.out,
                             workers=args.workers, refresh=args.refresh)
    except (IntakeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(report.describe())
    if report.failed:
        print("Re-run the same command to retry; files already here are kept.")
        return 1
    print(f"\nNext:\n  python -m data_pipeline.audit --input {report.local_dir}\n"
          f"  python -m data_pipeline.ingestion.import_labeled_pdfs --input {report.local_dir} --doc-type policy")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
