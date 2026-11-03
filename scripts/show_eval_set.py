"""What the frozen eval set holds, one line per document: no model, no GPU, nothing is changed.

For each frozen document: its line, carrier, whether that carrier was held out of training, whether it is a scan or
a synthetic twin, its page count, how many form rows its gold lists and how many of them carry a separate edition
date, and the name its label metadata gives it (to match it with a gold file on disk).

    python scripts/show_eval_set.py            # reads Blob (needs the AZURE_* settings)
    python scripts/show_eval_set.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

#: Label-metadata keys that name where a document came from, shown when present.
NAME_KEYS = ("original_filename", "bundle", "bundle_name", "source_file", "file_name", "delivered_as", "name")


def _raw(value: Any) -> Any:
    return value.get("raw") if isinstance(value, dict) else value


def form_counts(gold: dict[str, Any]) -> tuple[int, int]:
    """``(form rows, form rows with an edition date)`` of a gold, as stored (before any rule reads it)."""
    rows = [r for r in gold.get("forms_and_endorsements") or [] if isinstance(r, dict) and _raw(r.get("form_number"))]
    return len(rows), sum(1 for r in rows if _raw(r.get("edition_date")))


def describe(source_id: str, metadata: dict[str, Any], gold: dict[str, Any], pages: int,
             label_metadata: dict[str, Any] | None) -> dict[str, Any]:
    forms, with_edition = form_counts(gold)
    names = {k: label_metadata[k] for k in NAME_KEYS if label_metadata and label_metadata.get(k)}
    return {
        "source_id": source_id, "lob": metadata.get("lob"), "carrier": metadata.get("carrier"),
        "held_out_carrier": metadata.get("held_out_carrier"), "is_scanned": metadata.get("is_scanned"),
        "synthetic": metadata.get("synthetic"), "pages": pages, "gold_forms": forms,
        "gold_forms_with_edition": with_edition, "frozen_from_corpus": metadata.get("frozen_from_corpus"),
        "names": names,
    }


def render(rows: list[dict[str, Any]]) -> str:
    out = [f"{'document':13} {'line':6} {'held out':8} {'scan':5} {'pages':5} {'forms':5} {'w/ edition':10} carrier / name"]
    for r in rows:
        name = "; ".join(f"{k}={v}" for k, v in r["names"].items())
        out.append(f"{r['source_id']:13} {str(r['lob']):6} {str(r['held_out_carrier']):8} {str(r['is_scanned']):5} "
                   f"{r['pages']:5} {r['gold_forms']:5} {r['gold_forms_with_edition']:10} "
                   f"{r['carrier'] or '?'}" + (f"  [{name}]" if name else ""))
    return "\n".join(out)


def read_set(tenant: str | None) -> list[dict[str, Any]]:
    import re

    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from evaluation.run_eval import eval_set_keys

    client = BlobClient()
    prefix = paths.golden_eval_set_dir(tenant)
    by_source: dict[str, list[str]] = {}
    for key in eval_set_keys(client, tenant):
        rest = key[len(prefix):].strip("/")
        if "/" in rest:
            by_source.setdefault(rest.split("/", 1)[0], []).append(key)
    rows = []
    for source_id, keys in sorted(by_source.items()):
        base = f"{prefix}/{source_id}"
        if f"{base}/golden.json" not in keys:
            continue
        metadata = client.read_json(f"{base}/metadata.json") if f"{base}/metadata.json" in keys else {}
        gold = client.read_json(f"{base}/golden.json")
        pages = sum(1 for k in keys if re.search(r"/page_\d+\.png$", k))
        label_key = paths.label_metadata(metadata.get("doc_type") or "policy", source_id, tenant)
        label_metadata = client.read_json(label_key) if client.exists(label_key) else None
        rows.append(describe(source_id, metadata, gold, pages, label_metadata))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = read_set(args.tenant)
    print(json.dumps(rows, indent=2, ensure_ascii=False, default=str) if args.json else render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
