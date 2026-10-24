"""Where a document's pages may be sent: over TLS, unless to this machine.

The extraction routine sends page images and page text - insurance documents,
names, addresses, policy numbers - to a model server and an OCR server that run
elsewhere (RunPod, reached from Azure). Every such address must be HTTPS; plain
HTTP is accepted only to this machine itself (a server on the same pod).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

#: Hosts that are this machine: plain HTTP to them never leaves it.
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


class InsecureUrlError(ValueError):
    """A document would leave this machine unencrypted."""


class RedirectRefusedError(urllib.error.URLError):
    """A server answered with a redirect, which is not followed (:func:`open_url`)."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """urllib follows a redirect to any address - plain HTTP to another host
    included - and sends the request again with its headers, the bearer token
    among them: :func:`assert_encrypted` checked the configured address only.
    The servers here (vLLM, MinerU) answer their API where they are asked, so
    a redirect is a misconfigured URL or something in between: refused, named."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        fp.close()
        raise RedirectRefusedError(
            f"{req.full_url} redirected ({code}) to {newurl}, which is not followed: the request, its "
            "token and a document's pages would go to an address nobody checked. Use the server's own URL."
        )


_OPENER = urllib.request.build_opener(_RefuseRedirects)


def open_url(request: urllib.request.Request | str, *, timeout: float) -> Any:
    """``urllib.request.urlopen`` that refuses a redirect (:class:`RedirectRefusedError`,
    a ``URLError``) rather than follow it."""
    return _OPENER.open(request, timeout=timeout)


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
