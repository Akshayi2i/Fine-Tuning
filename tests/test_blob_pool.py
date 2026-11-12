"""The Blob client keeps a connection pool as large as the pipeline's parallel reads."""

from __future__ import annotations

from artifact_registry.blob_client import POOL_SIZE, _pooled_transport


def test_the_transport_pool_is_larger_than_the_default_of_ten():
    transport = _pooled_transport()
    manager = transport.session.get_adapter("https://account.blob.core.windows.net").poolmanager
    assert POOL_SIZE > 10 and manager.connection_pool_kw["maxsize"] == POOL_SIZE


def test_the_transport_carries_the_time_limits_itself():
    """A supplied transport keeps its own limits: the storage client's connection_timeout / read_timeout
    build only a transport it makes, so ours had azure-core's 300 s each (2026-11-12)."""
    from artifact_registry.blob_client import CONNECTION_TIMEOUT_S, READ_TIMEOUT_S

    config = _pooled_transport().connection_config
    assert (config.timeout, config.read_timeout) == (CONNECTION_TIMEOUT_S, READ_TIMEOUT_S) == (30, 120)


def test_the_service_client_reads_through_that_transport():
    from artifact_registry.blob_client import AzureBackend

    backend = AzureBackend(account_url="https://account.blob.core.windows.net", credential="sv=2024&sig=x")
    transport = backend._service._pipeline._transport
    while not hasattr(transport, "connection_config"):        # the SDK wraps the transport it is given
        transport = transport._transport
    assert (transport.connection_config.timeout, transport.connection_config.read_timeout) == (30, 120)
