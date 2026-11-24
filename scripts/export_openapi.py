"""Write the extraction API's contract, ``azure_service/openapi.json`` (review S8).

The API's documentation pages stay off in production (``docs_url=None``); the
contract is published as a committed file instead, generated from the FastAPI
app itself so it cannot drift from the code: ``tests/test_api_contract.py``
fails when the committed file and the app disagree.

    python -m scripts.export_openapi          # rewrite the file
    python -m scripts.export_openapi --check  # exit 1 when it is stale

The app is built over in-memory stand-ins (no Azure, no keys): only its routes,
parameters and answers are read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "azure_service" / "openapi.json"


def openapi_document() -> dict[str, Any]:
    """The app's OpenAPI document, built over in-memory stand-ins."""
    import tempfile

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from azure_service.api import create_app
    from azure_service.job_queue import InMemoryQueue
    from azure_service.settings import Settings, parse_api_keys
    from azure_service.store import JobStore

    settings = Settings(api_keys=parse_api_keys(json.dumps({"default": "contract-key-" + "x" * 32})),
                        webhook_secret="contract-secret-" + "x" * 32, public_base_url="https://api.example.test",
                        work_dir=Path(tempfile.gettempdir()) / "openapi-contract", tenants=("default",))
    store = JobStore(BlobClient(backend=InMemoryBackend(), container="extraction-jobs", context="serving"))
    app = create_app(settings, store, InMemoryQueue())
    return app.openapi()


def render() -> str:
    return json.dumps(openapi_document(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when the committed contract is stale")
    args = parser.parse_args(argv)
    text = render()
    current = CONTRACT.read_text(encoding="utf-8") if CONTRACT.exists() else ""
    if args.check:
        if current != text:
            print(f"{CONTRACT.relative_to(REPO)} differs from the app: run python -m scripts.export_openapi")
            return 1
        print("the API contract is current")
        return 0
    CONTRACT.write_text(text, encoding="utf-8", newline="\n")
    print(CONTRACT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
