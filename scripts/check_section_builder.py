"""Does our copy of Fideon's section builder still match FideonSLM's?

    python -m scripts.check_section_builder --fideon-slm "D:\\Fideon SLM"
    python -m scripts.check_section_builder                  # FIDEON_SLM_DIR, else only our own copy

``common/section_builder`` holds copies of FideonSLM's ``sections.py`` and
``pdf_text.py``, recorded in ``SOURCE.json`` (commit and SHA-256 of each). This
compares three things: our copy with SOURCE.json (an edit made here), and,
given a FideonSLM clone, that clone's files with SOURCE.json (the team moved
on). Either difference exits 1 and names the file; copy the files again and
update SOURCE.json, never edit them here.

A clone is only as new as its last pull: pull FideonSLM first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "common" / "section_builder"
SOURCE = BUILDER / "SOURCE.json"

#: Where a FideonSLM clone is, when ``--fideon-slm`` is not given.
FIDEON_SLM_ENV = "FIDEON_SLM_DIR"


def digest(path: Path) -> str:
    """SHA-256 of ``path`` with CRLF read as LF: a Windows checkout of the same
    commit is the same file (SOURCE.json ``sha256_of``)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source() -> dict:
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def local_differences() -> list[str]:
    """Each copied file here that no longer matches SOURCE.json."""
    recorded = source()["files"]
    return [
        f"{name}: edited here (SOURCE.json records {expected[:12]}, the copy is {digest(BUILDER / name)[:12]})"
        for name, expected in recorded.items()
        if not (BUILDER / name).exists() or digest(BUILDER / name) != expected
    ]


def upstream_differences(clone: Path) -> list[str]:
    """Each copied file whose FideonSLM version differs from SOURCE.json."""
    record = source()
    folder = clone / record["path"]
    out = []
    for name, expected in record["files"].items():
        theirs = folder / name
        if not theirs.exists():
            out.append(f"{name}: not in {folder}")
        elif digest(theirs) != expected:
            out.append(f"{name}: FideonSLM's copy changed since {record['commit'][:7]} - copy it again")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fideon-slm", type=Path, default=None,
                        help=f"a FideonSLM clone (default: ${FIDEON_SLM_ENV})")
    args = parser.parse_args(argv)

    problems = local_differences()
    clone = args.fideon_slm or (Path(os.environ[FIDEON_SLM_ENV]) if os.environ.get(FIDEON_SLM_ENV) else None)
    if clone is None:
        print(f"no FideonSLM clone given (--fideon-slm or ${FIDEON_SLM_ENV}): checked our copy only")
    else:
        problems += upstream_differences(clone)
    for problem in problems:
        print(f"DIFFERS  {problem}")
    if not problems:
        print(f"section builder matches FideonSLM {source()['commit'][:7]}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
