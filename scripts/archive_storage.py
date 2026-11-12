"""Move a tenant's frozen eval set, or some documents' golden labels, out of the way into an archive prefix.

A cycle that replaces the yardstick or leaves documents out has no other way to: a second freeze is refused while
a set is frozen (``evaluation.freeze_eval_set``), the corpus build leaves the frozen set's documents and their
families out of training, and it builds from every golden label the tenant holds
(``export_golden_labels.list_labeled_source_ids``). So, before cycle 2's corpus (user, 2026-11-13: cycle 1's frozen
set gives way to cycle 2's test split, and the 9 cycle-1 originals cycle 2 does not deliver leave its corpus):

* ``--eval-set``: every file under ``golden-eval-set/{tenant}/``;
* ``--labels-from FOLDER ...``: every file under ``golden-labels/{tenant}/{doc_type}/{id}/`` of the documents
  imported from those folders (their label metadata's ``imported_from`` and its history).

Each file is copied to ``<archive>/<its key>``, read back and compared, and only when every copy matches are the
originals deleted (the eval set's manifest first, so nothing gates on a half-removed set). Without ``--apply``
nothing is written: it lists what it would move. Undo by copying the archive's keys back.

    python -m scripts.archive_storage --eval-set --labels-from gl__hybrid__hartford_cgl_renewal ...
    python -m scripts.archive_storage --eval-set --labels-from ... --apply
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ARCHIVE = "archive/2026-11-13-before-cycle2"


@dataclass
class Plan:
    """What would move: each group's keys, and the folders no document was imported from."""

    groups: dict[str, list[str]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    def keys(self) -> list[str]:
        return [key for keys in self.groups.values() for key in keys]


def label_folders(metadata: dict[str, Any]) -> set[str]:
    """Every folder a document was imported from, as its label metadata records them."""
    folders = {str(entry.get("folder")) for entry in metadata.get("imported_from_history") or []
               if isinstance(entry, dict) and entry.get("folder")}
    if metadata.get("imported_from"):
        folders.add(str(metadata["imported_from"]))
    return folders


def plan(client: Any, *, eval_set: bool, folders: list[str], doc_type: str = "policy",
         tenant_id: str | None = None) -> Plan:
    """The keys to move: the frozen eval set's, and the golden labels of documents imported from ``folders``."""
    from artifact_registry import paths
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    def inside(prefix: str) -> list[str]:
        # The listing strips the trailing "/" (blob_client: name_starts_with=prefix.strip("/")), so a bare prefix
        # also matched golden-eval-set/default2/ or policy_00100/: only keys under the folder itself count.
        return sorted(key for key in client.list(prefix) if key.startswith(prefix))

    out = Plan()
    if eval_set:
        out.groups["frozen eval set"] = inside(paths.golden_eval_set_dir(tenant_id) + "/")
    if folders:
        wanted = set(folders)
        found: set[str] = set()
        for source_id in list_labeled_source_ids(client, doc_type, tenant_id):
            key = paths.label_metadata(doc_type, source_id, tenant_id)
            metadata = client.read_json(key) if client.exists(key) else {}
            hit = label_folders(metadata if isinstance(metadata, dict) else {}) & wanted
            if hit:
                found |= hit
                out.groups[f"labels of {source_id} ({', '.join(sorted(hit))})"] = inside(
                    paths.golden_label_dir(doc_type, source_id, tenant_id) + "/")
        out.unmatched = sorted(wanted - found)
    return out


def apply(client: Any, moving: Plan, archive: str) -> int:
    """Copy every key to ``archive``, compare each copy, then delete the originals. The files moved."""
    keys = moving.keys()
    for key in keys:
        target = f"{archive.rstrip('/')}/{key}"
        data = client.read_bytes(key)
        client.write_bytes(target, data)
        if client.read_bytes(target) != data:
            raise RuntimeError(f"the archived copy of {key} does not match it; nothing was deleted")
    # The eval set's manifest is what makes a set frozen: it goes first, so a stop part-way leaves no set that
    # looks frozen with documents missing (a re-run finishes the rest).
    for key in sorted(keys, key=lambda k: not k.endswith("/manifest.json")):
        client.delete(key)
    return len(keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", dest="eval_set", action="store_true", help="the tenant's frozen eval set")
    parser.add_argument("--labels-from", dest="folders", nargs="*", default=[],
                        help="delivered folder names: the golden labels of the documents imported from them")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, help="the prefix the files are copied under")
    parser.add_argument("--doc-type", dest="doc_type", default="policy")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--apply", action="store_true", help="move them; without it, only list them")
    args = parser.parse_args(argv)
    if not args.eval_set and not args.folders:
        parser.error("name what to move: --eval-set and/or --labels-from")
    from artifact_registry.blob_client import BlobClient

    client = BlobClient()
    moving = plan(client, eval_set=args.eval_set, folders=args.folders, doc_type=args.doc_type,
                  tenant_id=args.tenant)
    for name, keys in moving.groups.items():
        print(f"{name}: {len(keys)} file(s)")
    for folder in moving.unmatched:
        print(f"NOT FOUND: no document was imported from {folder}")
    print(f"{len(moving.keys())} file(s) in all, to {args.archive}/")
    if moving.unmatched:
        print("nothing moved: a folder named matched no document (check the name)")
        return 1
    if not args.apply:
        print("dry run: nothing moved. Add --apply to move them.")
        return 0
    moved = apply(client, moving, args.archive)
    print(f"moved {moved} file(s) to {args.archive}/: each copy checked against its original before the original "
          "was deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
