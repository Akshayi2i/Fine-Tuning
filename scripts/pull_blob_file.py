"""Download one Blob file to disk in pieces, each with a time limit and retries; a stopped run resumes.

For a big file (an adapter's weights) that a single blocking read would leave stalled at zero bytes
with no message. Each piece is a byte range read under an alarm; the file is written to ``<dest>.part``
and renamed only when its size matches Blob's.

    python scripts/pull_blob_file.py adapters/scope/policy/v3/adapter_model.safetensors /path/to/dest
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

PIECE_BYTES = 8_000_000
LIMIT_SECONDS = 60
TRIES = 6


class PullError(RuntimeError):
    """A piece could not be read in the allowed tries."""


def _stuck(*_args: object) -> None:
    raise TimeoutError(f"no data in {LIMIT_SECONDS} s")


def pull(blob: Any, dest: str | Path, *, piece: int = PIECE_BYTES, limit: int = LIMIT_SECONDS,
         tries: int = TRIES, log=print) -> int:
    """Download ``blob`` (an Azure blob client) to ``dest``; returns its size.

    Resumes from ``<dest>.part`` when one is there. A piece that fails or takes longer than ``limit``
    seconds is read again, up to ``tries`` times; then :class:`PullError`, with the file left to resume.
    """
    dest = Path(dest)
    part = Path(str(dest) + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = int(blob.get_blob_properties().size)
    done = part.stat().st_size if part.exists() else 0
    if done > size:
        part.unlink()
        done = 0
    log(f"blob {size:,} bytes; resuming at {done:,}")
    alarm = hasattr(signal, "SIGALRM")
    if alarm:
        signal.signal(signal.SIGALRM, _stuck)
    started, first = time.time(), done
    with open(part, "ab") as out:
        while done < size:
            length = min(piece, size - done)
            for attempt in range(1, tries + 1):
                try:
                    if alarm:
                        signal.alarm(limit)
                    data = blob.download_blob(offset=done, length=length, max_concurrency=1).readall()
                    if alarm:
                        signal.alarm(0)
                    break
                except Exception as exc:  # noqa: BLE001 - any failure of one piece is retried
                    if alarm:
                        signal.alarm(0)
                    log(f"  piece at {done:,} try {attempt}/{tries} failed: {type(exc).__name__}: {exc}")
                    if attempt == tries:
                        raise PullError(f"gave up at {done:,} of {size:,}; run it again to resume") from exc
            out.write(data)
            out.flush()
            done += len(data)
            rate = (done - first) / 1e6 / max(time.time() - started, 1e-9)
            log(f"{done / 1e6:.0f} of {size / 1e6:.0f} MB, {rate:.1f} MB/s")
    if done != size:
        raise PullError(f"read {done:,} bytes of {size:,}")
    os.replace(part, dest)
    log(f"downloaded {done:,} bytes -> {dest}")
    return size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", help="the Blob key, e.g. adapters/scope/policy/v3/adapter_model.safetensors")
    parser.add_argument("dest", help="the local file to write")
    args = parser.parse_args(argv)

    from artifact_registry.blob_client import BlobClient

    client = BlobClient()
    blob = client._backend._client(client._container_for(args.key), args.key)
    try:
        pull(blob, args.dest)
    except PullError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
