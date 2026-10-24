"""Page images as a served model is sent them: fewer bytes, the same pixels.

A window's request to a served model (``inference_core.openai_backend``) holds
the images of its pages, and a page is read by about four windows. From a
laptop through a RunPod proxy, sending those bytes is most of the model's time
(a 114-page scan: 871 MB at about 1 MB/s). Two things cut them, neither
changing what the model sees:

* **A gray page goes as a grayscale PNG** (:func:`sent_png`). The renderer
  writes RGB, and on a scan red, green and blue are equal everywhere; vLLM
  turns every image into RGB before the model reads it, so an 8-bit grayscale
  PNG of the same page gives the model the same pixels, at about 60% of the
  bytes. A page with any colour goes as rendered.
* **A page goes once, as a link** (:class:`PageLinks`): uploaded to Blob once,
  and each window's request carries a read-only link that the model server
  fetches itself, over the data centre's network. A request is then a few
  kilobytes, which also keeps it under the RunPod serverless API's request cap.
  The blobs are transient (``transient/page-links/``) and deleted once the
  document is read; the links expire on their own, after an hour (a document
  takes minutes): a link's life is how long a copied one would still read.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: Where page images wait for a served model to fetch them.
LINK_PREFIX = "transient/page-links"


def sent_png(image: str | Path, *, gray: bool = True) -> bytes:
    """The bytes a page image is sent as: an RGB PNG whose channels are equal
    everywhere as an 8-bit grayscale PNG, anything else as it is on disk - an
    image with transparency information (a ``tRNS`` colour) included: its
    transparent colour is an RGB triple a grayscale PNG cannot carry, and the
    server would composite it differently."""
    path = Path(image)
    if not gray or path.suffix.lower() != ".png":
        return path.read_bytes()
    stat = path.stat()
    return _gray_png(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _gray_png(path: str, _mtime_ns: int, _size: int) -> bytes:
    """One page, read and re-encoded once however many windows send it (the
    file's time and size in the key, so a page written again is read again)."""
    from PIL import Image, ImageChops

    data = Path(path).read_bytes()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:  # noqa: BLE001 - not an image PIL reads: sent as it is, the server decides
        return data
    with image:
        if image.mode != "RGB" or "transparency" in image.info:
            return data
        red, green, blue = image.split()
        if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
            return data
        out = io.BytesIO()
        red.save(out, format="PNG")              # default compression: 0.26 s a page, `optimize` 5x that for 7%
    return out.getvalue()


class PageLinks:
    """Page images put in Blob once, and their read-only links.

    ``prefix`` is this run's folder under :data:`LINK_PREFIX`; a document's
    pages go under ``<prefix>/<folder>/``. ``read_token`` is a read-only SAS for
    the container (``PAGE_LINK_SAS_TOKEN``); without one each link is signed for
    its blob alone, which needs the account key (``BlobClient.read_link``).
    """

    def __init__(self, client: BlobClient, prefix: str, *, read_token: str | None = None,
                 expiry_s: int = 3600, gray: bool = True, workers: int = 8) -> None:
        self.client = client
        self.prefix = prefix.strip("/")
        self.read_token = read_token
        self.expiry_s = expiry_s
        self.gray = gray
        self.workers = workers
        self.sent_bytes = 0
        self._urls: dict[str, str] = {}
        self._keys: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _id(path: str | Path) -> str:
        return str(Path(path).resolve())

    def check(self) -> None:
        """Raise now, not after a PDF's upload, when no link can be made: the
        token is not read-only, or the login can sign none (nothing is written)."""
        self.client.read_link(f"{self.prefix}/check.png", expiry_s=self.expiry_s, read_token=self.read_token)

    def url(self, path: str | Path) -> str | None:
        """The link to ``path``'s page, or None when it was not published."""
        return self._urls.get(self._id(path))

    def publish(self, paths: list[str], folder: str) -> dict[str, Any]:
        """Upload ``paths`` under ``<prefix>/<folder>/``, a few at once, and make
        their links. Returns the pages, the megabytes and the seconds it took."""
        started = time.perf_counter()

        def one(path: str) -> int:
            data = sent_png(path, gray=self.gray)
            key = f"{self.prefix}/{folder}/{Path(path).name}"
            self.client.write_bytes(key, data)
            link = self.client.read_link(key, expiry_s=self.expiry_s, read_token=self.read_token)
            with self._lock:
                self._urls[self._id(path)] = link
                self._keys[self._id(path)] = key
                self.sent_bytes += len(data)
            return len(data)

        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="page-links") as pool:
            sent = sum(pool.map(one, paths))
        return {"pages": len(paths), "mb": round(sent / 2**20, 1), "seconds": round(time.perf_counter() - started, 2)}

    def discard(self, paths: list[str] | None = None) -> None:
        """Delete the blobs of ``paths`` (every one published, by default) and forget their links."""
        with self._lock:
            ids = list(self._keys) if paths is None else [self._id(p) for p in paths]
            keys = [self._keys.pop(i) for i in ids if i in self._keys]
            for i in ids:
                self._urls.pop(i, None)
        for key in keys:
            try:
                self.client.delete(key)
            except Exception as exc:  # noqa: BLE001 - a page left behind expires; the read still stands
                log.warning("could not delete %s: %s", key, exc)
