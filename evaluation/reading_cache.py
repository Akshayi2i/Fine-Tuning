"""A gate that was stopped half way is picked up where it stopped, not read again from the first document.

The gate reads every frozen document in three modes and once more for run-to-run agreement: about three hours on
one GPU. A reading that finished is kept on the pod's volume as soon as it is scored, under a key that names
everything the answer depends on (the release, the adapter checkpoint, the calibrators and thresholds, the
frozen set, the settings, the code commit). A run that starts again with the same key reads the kept readings and
only the missing ones go to the model. Any change to the key reads everything fresh, and a reading that failed is
never kept, so it is tried again.

The files are the run's own scratch (pickles of the objects the scorer holds), read back only by the run that
wrote them on the same volume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FIRST, REPEAT = "first", "repeat"


def cache_key(**parts: Any) -> str:
    """A short name for everything a reading depends on. A part that cannot be read as a stable value (an object
    printed with a memory address) makes the key differ every run, which is a cache that never hits: safe."""
    text = json.dumps(parts, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def default_root() -> Path:
    return Path(os.environ.get("FIDEON_WORKSPACE") or "/workspace") / "gate_cache"


class ReadingCache:
    """Readings kept under ``root / key``; one file per (document, mode, first or repeat)."""

    def __init__(self, root: Path | str, key: str) -> None:
        self.dir = Path(root) / key
        self.key = key
        self.hits = 0
        self.stored = 0

    def _file(self, source_id: str, mode: str, kind: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{source_id}__{mode}__{kind}")
        return self.dir / f"{safe}.pkl"

    def get(self, source_id: str, mode: str, kind: str = FIRST) -> Any | None:
        path = self._file(source_id, mode, kind)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                value = pickle.load(handle)  # noqa: S301 - this run's own scratch, on its own volume
        except Exception as exc:  # noqa: BLE001 - a torn or stale file is a miss, never a failure
            log.warning("gate cache: %s could not be read (%s: %s); reading it again",
                        path.name, type(exc).__name__, exc)
            return None
        self.hits += 1
        return value

    def put(self, source_id: str, mode: str, value: Any, kind: str = FIRST) -> None:
        path = self._file(source_id, mode, kind)
        tmp = path.with_suffix(".tmp")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)             # whole or absent: a stop in the middle leaves no torn file
            self.stored += 1
        except Exception as exc:  # noqa: BLE001 - not kept; the gate still scores what it read
            log.warning("gate cache: %s not kept (%s: %s)", path.name, type(exc).__name__, exc)
            tmp.unlink(missing_ok=True)

    def summary(self) -> str:
        return f"{self.hits} reading(s) taken from the cache, {self.stored} kept for a restart"
