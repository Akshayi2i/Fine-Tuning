"""Where a document's pages may be sent: over TLS, unless to this machine.

The extraction routine sends page images and page text - insurance documents,
names, addresses, policy numbers - to a model server and an OCR server that run
elsewhere (RunPod, reached from Azure). Every such address must be HTTPS; plain
HTTP is accepted only to this machine itself (a server on the same pod).
"""

from __future__ import annotations

from urllib.parse import urlsplit

#: Hosts that are this machine: plain HTTP to them never leaves it.
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


class InsecureUrlError(ValueError):
    """A document would leave this machine unencrypted."""


def assert_encrypted(url: str, what: str) -> None:
    """Refuse ``url`` unless it is HTTPS, or HTTP to this machine."""
    parts = urlsplit(url)
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and (parts.hostname or "").lower() in LOOPBACK:
        return
    raise InsecureUrlError(
        f"{what} at {url!r} is not HTTPS: a document's pages and text would leave this machine "
        "unencrypted. Use the server's https:// address (a RunPod proxy URL or API is HTTPS)."
    )
