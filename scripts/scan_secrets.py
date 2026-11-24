"""Fail when a real secret, or a real-looking SSN, is in the committed files.

The CI job this replaces matched the words (``AccountKey=``), so every
placeholder (``AccountKey=...``, ``AccountKey=k`` in tests) failed the build
and the job was red on every push (review R5, 23 November 2026). These
patterns match the secrets' shapes instead:

* an Azure storage account key: ``AccountKey=`` and 86 or more base64
  characters (a real key is 88, ending ``==``);
* a SAS signature: ``sig=`` and 40 or more URL-safe characters (a real one
  is the base64 of an HMAC-SHA256, 44 characters, URL-encoded);
* a private key block: ``-----BEGIN ... PRIVATE KEY``;
* an SSN-shaped string, in the fixtures and schema examples only.

Allowed, by exact file and text: the Azurite development key, which is
public (Microsoft publishes it for the local storage emulator) and is
quoted in ``azure_service/DEPLOY.md`` for local tests. Nothing else is
allowed and no folder is skipped except ``.git``.

    python scripts/scan_secrets.py            # the committed files (git ls-files)
    python scripts/scan_secrets.py PATH...    # these files or folders

Exits 1 and names each file and line when anything is found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "Azure storage account key": re.compile(r"AccountKey=[A-Za-z0-9+/]{86,}={0,2}"),
    "SAS signature": re.compile(r"\bsig=[A-Za-z0-9%+/_-]{40,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
}

#: SSN-shaped strings are checked only where real document data must never be.
SSN_PATTERN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
SSN_FOLDERS = ("tests/fixtures/", "schemas/examples/")

#: The Azurite emulator's well-known account key (public: Microsoft documents it
#: for local development). Allowed only in this file.
AZURITE_KEY = ("Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
               "K1SZFPTOtr/KBHBeksoGMGw==")
ALLOWED: dict[str, tuple[str, ...]] = {
    "azure_service/DEPLOY.md": (f"AccountKey={AZURITE_KEY}",),
}


def _committed_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True)
    return [REPO / name for name in out.stdout.decode("utf-8").split("\0") if name]


def _expand(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from (p for p in sorted(path.rglob("*")) if p.is_file() and ".git" not in p.parts)
        elif path.is_file():
            yield path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def scan_text(name: str, text: str) -> list[tuple[int, str]]:
    """``(line, what)`` for each finding in one file's text."""
    allowed = ALLOWED.get(name, ())
    ssn = name.startswith(SSN_FOLDERS)
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        cleaned = line
        for exact in allowed:
            cleaned = cleaned.replace(exact, "")
        for what, pattern in SECRET_PATTERNS.items():
            if pattern.search(cleaned):
                found.append((number, what))
        if ssn and SSN_PATTERN.search(cleaned):
            found.append((number, "SSN-shaped string in a fixture"))
    return found


def scan(paths: Iterable[Path]) -> list[str]:
    """One message per finding, ``file:line: what``."""
    messages: list[str] = []
    for path in _expand(paths):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:8192]:  # a binary file: no text secret to read
            continue
        name = _relative(path)
        for number, what in scan_text(name, raw.decode("utf-8", errors="replace")):
            messages.append(f"{name}:{number}: {what}")
    return messages


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    paths = [Path(a) for a in args] if args else _committed_files()
    found = scan(paths)
    for message in found:
        print(f"::error::{message}")
    if found:
        print(f"{len(found)} secret-shaped string(s) found")
        return 1
    print("no secrets found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
