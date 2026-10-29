"""The Blob client keeps a connection pool as large as the pipeline's parallel reads."""

from __future__ import annotations

from artifact_registry.blob_client import POOL_SIZE, _pooled_transport


def test_the_transport_pool_is_larger_than_the_default_of_ten():
    transport = _pooled_transport()
    manager = transport.session.get_adapter("https://account.blob.core.windows.net").poolmanager
    assert POOL_SIZE > 10 and manager.connection_pool_kw["maxsize"] == POOL_SIZE
