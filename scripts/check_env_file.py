"""Find the lines of a ``.env`` that bash would misread when the file is sourced.

The pod scripts load ``.env`` with ``set -a; . ./.env; set +a``, so every line is bash code. An unquoted value
with a space runs its second word as a command, and one with ``;`` or ``&`` is cut there. This reads the file the way bash would and names each such line, with the
fix, before a run starts. Values are never printed, only the key and the line number.

    python scripts/check_env_file.py [path]        # default ./.env ; exit 1 when a line would be misread

Standard library only: it runs before any virtualenv is active.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Characters an unquoted value may hold and bash still reads as itself.
SAFE_UNQUOTED = re.compile(r"^[A-Za-z0-9_@%+=:,./~-]*$")
LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _quoted(value: str) -> bool:
    """Whether ``value`` is one complete quoted string (single, or double without an unescaped ``$`` or backtick)."""
    if len(value) < 2 or value[0] not in "'\"" or value[-1] != value[0]:
        return False
    body = value[1:-1]
    if value[0] == "'":
        return "'" not in body
    return not re.search(r'(?<!\\)(?:"|\$|`)', body)


def problems(text: str) -> list[str]:
    """One message per line bash would misread; empty when the file is safe to source."""
    found: list[str] = []
    for number, raw in enumerate(text.split("\n"), start=1):
        raw = raw.removesuffix("\r")        # a .env saved on Windows: the code strips the CR from each value
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.match(raw)
        if not match:
            found.append(f"line {number}: not KEY=value, so bash would run it as a command")
            continue
        key, value = match.group(1), match.group(2).strip()
        value = re.sub(r"\s+#.*$", "", value) if not value.startswith(("'", '"')) else value
        if not value or _quoted(value) or SAFE_UNQUOTED.match(value):
            continue
        reason = "a space" if " " in value or "\t" in value else "a character bash treats specially"
        found.append(f"line {number}: {key} has {reason} and is not quoted; wrap the whole value in "
                     f"single quotes: {key}='...'")
    return found


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0] if args else ".env")
    if not path.is_file():
        return 0
    found = problems(path.read_text(encoding="utf-8", errors="replace"))
    for message in found:
        print(f"{path}: {message}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
