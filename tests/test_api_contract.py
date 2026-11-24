"""The committed API contract is the app's (review S8).

``azure_service/openapi.json`` is what callers build against; it is generated
from the FastAPI app (``python -m scripts.export_openapi``). A route, parameter
or answer changed in the code without regenerating the file fails here.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="the [service] extra is not installed")

from scripts.export_openapi import CONTRACT, openapi_document, render  # noqa: E402


def test_the_committed_contract_matches_the_app() -> None:
    assert CONTRACT.exists(), "run python -m scripts.export_openapi"
    assert CONTRACT.read_text(encoding="utf-8") == render(), "stale: run python -m scripts.export_openapi"


def test_the_contract_lists_every_public_route() -> None:
    paths = json.loads(CONTRACT.read_text(encoding="utf-8"))["paths"]
    assert {"/healthz", "/v1/extractions", "/v1/extractions/{job_id}",
            "/v1/extractions/{job_id}/result"} <= set(paths)


def test_a_changed_route_is_seen(monkeypatch) -> None:
    """Seen to fail on purpose: a contract missing a route differs from the app."""
    stale = openapi_document()
    stale["paths"].pop("/healthz")
    assert json.dumps(stale, indent=2, sort_keys=True, ensure_ascii=False) + "\n" != render()
