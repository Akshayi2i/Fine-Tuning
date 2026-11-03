"""A failed read of the registry index must never turn into an empty index written back over it.

``update_registry_index`` used to treat a read error as "no index yet" and write one row, replacing every other
run's row; the queries "what is live" and "what depends on this adapter" read that file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, BlobError, InMemoryBackend
from registry_utils.write_run_manifest import RegistryError, update_registry_index


def _manifest(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(run_id=run_id, index_row=lambda: {
        "run_id": run_id, "created_at": "2026-11-03T00:00:00+00:00", "status": "trained"})


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _runs(client: BlobClient) -> list[str]:
    return [r["run_id"] for r in client.read_json(paths.registry_index())["runs"]]


def test_a_new_index_is_made_when_none_exists(client):
    update_registry_index(_manifest("policy-v1"), client)
    assert _runs(client) == ["policy-v1"]


def test_a_run_is_added_beside_the_others(client):
    update_registry_index(_manifest("policy-v1"), client)
    update_registry_index(_manifest("policy-v2"), client)
    assert _runs(client) == ["policy-v1", "policy-v2"]


def test_a_failed_read_leaves_the_index_alone_and_says_so(client, monkeypatch):
    update_registry_index(_manifest("policy-v1"), client)
    update_registry_index(_manifest("policy-v2"), client)
    before = client.read_json(paths.registry_index())

    real = client.read_json

    def broken(key, *args, **kwargs):
        if key == paths.registry_index():
            raise BlobError("failed reading main/registry_index.json: timed out")
        return real(key, *args, **kwargs)

    monkeypatch.setattr(client, "read_json", broken)
    with pytest.raises(RegistryError, match="Nothing was overwritten"):
        update_registry_index(_manifest("policy-v3"), client)

    monkeypatch.undo()
    assert client.read_json(paths.registry_index()) == before      # both earlier rows are still there
    assert _runs(client) == ["policy-v1", "policy-v2"]


def test_a_real_store_waits_seconds_between_tries_and_says_so(monkeypatch, caplog):
    from artifact_registry import blob_client as bc

    class Failing(bc.InMemoryBackend):
        def read(self, container, key):
            raise bc.BlobError("connection reset")

    waited: list[float] = []
    monkeypatch.setattr(bc.time, "sleep", waited.append)
    real = bc.BlobClient(backend=Failing(), container="main", raw_container="raw", retry_backoff_s=None)
    assert real.retry_backoff_s == 0.1                       # the in-memory store: no network to wait for
    real.retry_backoff_s = bc.AZURE_RETRY_BACKOFF_S
    with caplog.at_level("WARNING"), pytest.raises(bc.BlobError):
        real.read_bytes("a/b.json")
    assert waited == [2.0, 4.0]
    assert any("connection reset" in r.getMessage() for r in caplog.records)


def test_the_azure_services_are_built_with_explicit_timeouts(monkeypatch):
    import sys
    import types

    from artifact_registry import blob_client as bc

    seen: list[dict] = []

    class Service:
        def __init__(self, *a, **kw):
            seen.append(kw)

        @classmethod
        def from_connection_string(cls, conn, **kw):
            seen.append(kw)
            return object.__new__(cls)

    module = types.ModuleType("azure.storage.blob")
    module.BlobServiceClient = Service
    monkeypatch.setitem(sys.modules, "azure.storage.blob", module)
    monkeypatch.setattr(bc, "_pooled_transport", lambda: object())
    bc.AzureBackend(account_url="https://x.blob.core.windows.net", credential="sv=1",
                    raw_container="raw", raw_credential="sv=2")
    bc.AzureBackend(connection_string="DefaultEndpointsProtocol=https;AccountName=a;AccountKey=k")
    assert len(seen) == 3
    assert all(kw["connection_timeout"] == bc.CONNECTION_TIMEOUT_S and kw["read_timeout"] == bc.READ_TIMEOUT_S
               for kw in seen)
