"""Put a run's training data where ms-swift can read it, in the shape it reads.

Two gaps between the corpus and the trainer, closed in one place:

**Location.** The corpus — its epoch files, its validation file and every page
image a row points at — lives in Blob, addressed by keys. ``swift sft`` runs on
the pod and reads local files. Handed a Blob key it finds nothing; worse, an
unfamiliar dataset string can be taken for a hub dataset id. So the files a run
trains on are copied onto the staging volume first.

**Shape.** A corpus row stores its user turn as a list — ``{"type": "image"}``,
``{"type": "text"}``, interleaved — beside string system and assistant turns.
That is the shape vLLM serves from, so the corpus keeps it. It is not a shape
the trainer's loader can read: a JSON column that is a string in one place and a
list in another is refused by pyarrow, and so by HF ``datasets``, on the first
row. ms-swift's documented multimodal format is string content with an
``<image>`` placeholder per image and a top-level ``images`` list, in order.

**The conversion must not change what the model sees.** Qwen's chat template
renders a content list by concatenating, with no separator, a vision block for
each image and the text of each text block; ms-swift substitutes the same vision
block for each ``<image>``. So each image becomes ``<image>`` and the parts are
joined with nothing between them. That equivalence is asserted on the real
template by the Phase 0 spike (``check_swift_row_format``), not assumed here —
prompt parity between training and serving is the property the whole corpus
design rests on.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry.blob_client import BlobClient
from common.constants import TRAINER_SPECIAL_TAGS

log = logging.getLogger(__name__)

#: ms-swift's image placeholder, one per entry in the row's ``images`` list.
SWIFT_IMAGE_TAG = "<image>"

#: Concurrent image downloads. Staging runs on the paid pod before launch, and
#: 20,000 pages fetched one at a time is minutes of idle GPU.
DOWNLOAD_WORKERS = 16


class StagingError(RuntimeError):
    """Raised when a run's training data cannot be staged faithfully."""


@dataclass
class StagedData:
    """The local files a run reads, and what staging did."""

    epoch_files: list[str] = field(default_factory=list)
    val_path: str | None = None
    rows: int = 0
    val_rows: int = 0
    images: int = 0
    #: Images actually downloaded; the rest came from the version's shared cache.
    fetched: int = 0
    #: Per staged file, each row's output reservation (its task's
    #: max_output_tokens), in row order — what the pre-launch length check
    #: measures each row's target against. Kept beside the file, not in it:
    #: the trainer reads only messages and images.
    output_caps: dict[str, list[int]] = field(default_factory=dict)
    #: Per staged file, each row's section group (``decl``, ``arrays``,
    #: ``lineblk`` ...; else its task), in row order - what the length check
    #: totals supervised tokens by. Beside the file, like ``output_caps``.
    groups: dict[str, list[str]] = field(default_factory=dict)


def to_swift_row(row: dict[str, Any], local_image: Callable[[str], str]) -> dict[str, Any]:
    """One corpus row in ms-swift's format: string content, ``<image>``, ``images``.

    Only ``messages`` and ``images`` are kept. Everything else on a corpus row is
    bookkeeping the trainer does not read, and every extra column is one more
    whose type has to agree across every row before the file will load at all.
    """
    images: list[str] = []
    messages: list[dict[str, str]] = []
    for message in row.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            _refuse_placeholder(content, row)
            messages.append({"role": message["role"], "content": content})
            continue
        parts: list[str] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "image":
                image = block.get("image")
                if not isinstance(image, str):
                    raise StagingError(
                        f"{row.get('source_id')}: an image block holds {type(image).__name__}, "
                        "not a path. A training row must reference its page by Blob key."
                    )
                parts.append(SWIFT_IMAGE_TAG)
                images.append(local_image(image))
            elif kind == "text":
                _refuse_placeholder(block.get("text", ""), row)
                parts.append(block.get("text", ""))
            else:
                raise StagingError(f"{row.get('source_id')}: unknown content block {kind!r}")
        messages.append({"role": message["role"], "content": "".join(parts)})
    return {"messages": messages, "images": images}


def _refuse_placeholder(text: str, row: dict[str, Any]) -> None:
    """Text containing one of ms-swift's tags would be parsed as one: a literal
    ``<image>`` pairs the next page with the wrong text, ``<bbox>`` or
    ``<video>`` is rewritten or fails to encode. Refused rather than escaped:
    there is no escape the template honours.

    The corpus build sets such documents aside first (``build_jsonl``), so this
    firing means a corpus built before that check. It is the backstop.
    """
    tag = next((t for t in TRAINER_SPECIAL_TAGS if t in text), None)
    if tag:
        raise StagingError(
            f"{row.get('source_id')}: the text contains a literal {tag!r}, which ms-swift "
            "parses as a special tag. Rebuild the corpus; the build now sets such documents aside."
        )


def stage_training_data(
    epoch_files: Sequence[str],
    val_path: str | None,
    client: BlobClient,
    local_root: str | Path,
    *,
    images_root: str | Path | None = None,
    workers: int = DOWNLOAD_WORKERS,
    max_pixels: int | None = None,
    val_filter: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
) -> StagedData:
    """Copy and convert a run's data onto local disk. Returns the local paths.

    ``val_filter`` narrows the validation rows before conversion - the fixed
    validation sample (``evaluation.validation_sample``), which needs the line
    and reading mode the conversion drops.

    **Every row is converted and checked before a single image is fetched.** A
    row carrying a trainer tag, an unknown content block, or a task whose pixel
    budget is not the run's (``max_pixels``) refuses the run at once — not after
    twenty thousand page images have been copied onto a paid pod.

    ``images_root`` is the per-corpus-version cache every run on that version
    shares (``paths.staging_train_images_dir``); without one the images go under
    ``local_root``. An image already in the cache is not fetched again, which is
    only safe because each download goes to its own temporary file and is renamed
    into place: a pod preempted mid-download leaves a ``.part`` file, never a
    truncated image under the real name, and two runs fetching one page at once
    cannot interleave their writes.
    """
    root = Path(local_root)
    cache = Path(images_root) if images_root is not None else root / "images"
    staged = StagedData()

    sources: list[tuple[str, str]] = [
        (f"train/epoch_{index}.jsonl", key) for index, key in enumerate(epoch_files, start=1)
    ]
    if val_path:
        sources.append(("val/val.jsonl", val_path))

    # Convert first. The local path of every image is known before it exists, so
    # the conversion — and every refusal in it — runs before any download.
    converted: list[tuple[str, list[dict[str, Any]], list[int], list[str]]] = []
    keys: set[str] = set()

    def local_image(key: str) -> str:
        keys.add(key)
        return key          # its cache path once every key's is known (image_cache_paths, below)

    for name, key in sources:
        rows = _read_rows(client, key)
        if val_filter is not None and name.startswith("val/"):
            rows = val_filter(rows)
        if max_pixels is not None:
            _refuse_other_budgets(rows, max_pixels)
        converted.append((
            name, [to_swift_row(row, local_image) for row in rows], [_output_cap(r) for r in rows],
            [str(r.get("sections") or r.get("task") or "unknown") for r in rows],
        ))

    # Where each page is cached: by its content, shared by every cycle that reads
    # it, when the reading behind it is known (image_cache_paths).
    local = image_cache_paths(client, sorted(keys), cache, workers=workers,
                              by_content=images_root is not None and in_staging_cache(images_root))
    for _name, rows, *_rest in converted:
        for row in rows:
            row["images"] = [str(local[key].resolve()) for key in row["images"]]
    missing = sorted(key for key in keys if not local[key].exists())
    _download_to(client, [(key, local[key]) for key in missing], workers)

    for name, rows, caps, groups in converted:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        staged.rows += len(rows)
        staged.output_caps[str(target.resolve())] = caps
        staged.groups[str(target.resolve())] = groups
        if name.startswith("val/"):
            staged.val_path, staged.val_rows = str(target.resolve()), len(rows)
        else:
            staged.epoch_files.append(str(target.resolve()))
    staged.images = len(keys)
    staged.fetched = len(missing)

    log.info(
        "staged %d row(s) under %s; %d page image(s), %d fetched, cache %s",
        staged.rows, root, len(keys), len(missing), cache,
    )
    return staged


def _output_cap(row: dict[str, Any]) -> int:
    """The row's task output reservation — how long its target may be."""
    from common.config import sequence_for_task

    task = row.get("task") or "extract"
    return int(sequence_for_task(task, row.get("doc_type"))["max_output_tokens"])


def _refuse_other_budgets(rows: list[dict[str, Any]], max_pixels: int) -> None:
    """The trainer resizes every image in a run to ONE budget. A row whose task
    is budgeted differently — a thumbnail task, at a tenth of the pixels — would
    be trained at the wrong resolution, so it is refused rather than resized."""
    from common.config import vision_for_task

    for row in rows:
        task = row.get("task") or "extract"
        budget = int(vision_for_task(task)["max_pixels"])
        if budget != max_pixels:
            raise StagingError(
                f"{row.get('source_id')}: task {task!r} is budgeted at {budget:,} px but this "
                f"run resizes every page to {max_pixels:,}. The trainer takes one budget per "
                "run; train tasks with different budgets in separate runs."
            )


def _download_all(client: BlobClient, keys: list[str], cache: Path, workers: int) -> None:
    """Fetch ``keys`` concurrently, each to ``cache / key``."""
    _download_to(client, [(key, cache / key) for key in keys], workers)


# --------------------------------------------------------------------------
# The page-image cache, by content
# --------------------------------------------------------------------------

#: The cache shared by every corpus version, beside the per-version folders
#: (``paths.staging_train_images_dir``): a page is filed under its blob key and
#: a tag of the reading that rendered it (:func:`content_tag`).
CONTENT_CACHE_DIR = "by-content"

#: Tags kept per page in the content cache: the newest (what a cycle reads) and
#: the one before (a resumed run, or a comparison, may still read it).
KEEP_CONTENT_TAGS = 2

#: What a document's ``ocr_meta.json`` says about its page images' pixels.
_CONTENT_FIELDS = ("source_checksum", "mineru_version", "mineru_tier", "ocr_device", "resolution_cap_px",
                   "preprocessing_date")


def in_staging_cache(images_root: str | Path) -> bool:
    """Whether ``images_root`` is a per-version folder of the staging image cache
    (``paths.staging_train_images_dir``: ``.../train-images/<tenant>/<version>``),
    beside which the content cache lives - inside the tree the purge plan
    lists (``pull_raw_pdfs``). Any other cache (the testing harness's) stays keyed by blob key."""
    return Path(images_root).parent.parent.name == "train-images"


def content_cache_root(images_root: str | Path) -> Path:
    """The content-keyed cache beside a per-version image cache."""
    return Path(images_root).parent / CONTENT_CACHE_DIR


def content_tag(ocr_meta: dict[str, Any]) -> str | None:
    """A tag for the pixels a document's reading rendered, or None when its
    ``ocr_meta.json`` cannot say.

    From what makes them: the PDF (``source_checksum``), the MinerU that read it,
    its tier and device, the resolution cap, and when it was rendered - a
    re-render after a renderer fix keeps every other field. Two cycles that read
    one reading share its pages; a reading made again is a new tag, so a cache
    spanning versions never serves the old pixels.
    """
    import hashlib

    if not ocr_meta.get("source_checksum") or not ocr_meta.get("preprocessing_date"):
        return None
    identity = json.dumps({k: ocr_meta.get(k) for k in _CONTENT_FIELDS}, sort_keys=True, default=str)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _tagged(key: str, tag: str) -> str:
    """``processed/t/policy/s/page_3.png`` -> ``processed/t/policy/s/page_3.<tag>.png``."""
    folder, _, name = key.rpartition("/")
    stem, dot, suffix = name.rpartition(".")
    name = f"{stem}.{tag}.{suffix}" if dot else f"{name}.{tag}"
    return f"{folder}/{name}" if folder else name


def _readings(client: BlobClient, keys: Sequence[str], workers: int) -> dict[str, dict[str, Any]]:
    """``{document folder: its ocr_meta.json}`` for every page key under
    ``processed/``, read once per document, through a pool. A folder with no
    readable meta is left out."""
    folders = sorted({key.rpartition("/")[0] for key in keys
                      if key.startswith("processed/") and key.count("/") >= 4})
    if not folders:
        return {}

    def read(folder: str) -> dict[str, Any] | None:
        meta = f"{folder}/ocr_meta.json"
        try:
            return client.read_json(meta) if client.exists(meta) else None
        except Exception as exc:  # noqa: BLE001 - an unreadable meta is cached by key, as before
            log.warning("image cache: %s unreadable (%s); its pages are cached per corpus version", meta, exc)
            return None

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(folders)))) as pool:
        metas = list(pool.map(read, folders))
    return {folder: meta for folder, meta in zip(folders, metas, strict=True) if isinstance(meta, dict)}


def image_cache_paths(
    client: BlobClient, keys: Sequence[str], images_root: str | Path, *,
    workers: int = DOWNLOAD_WORKERS, by_content: bool = True,
) -> dict[str, Path]:
    """``{blob key: where the page is cached}``. Downloads nothing.

    By content (:func:`content_tag`), under :data:`CONTENT_CACHE_DIR` beside the
    per-version ``images_root``: every cycle and corpus version that reads a
    page reads one copy, fetched once, until a new reading renders it again.
    Before, the cache was per corpus version, and every cycle fetched every page
    of the corpus again. A key whose reading is unknown - not under
    ``processed/``, or no ``ocr_meta.json`` that says - stays under
    ``images_root / key``, per version, as before.

    Older caches stay readable: a page found in a per-version folder (this
    version's or another's) fetched after its reading was rendered is that
    reading's pixels, and is linked (or copied) into the content cache rather
    than fetched again. One fetched before - a re-OCR since - is left alone.
    """
    root = Path(images_root)
    out = {key: root / key for key in keys}
    if not by_content or not keys:
        return out
    readings = _readings(client, keys, workers)
    content = content_cache_root(root)
    old_roots = _old_caches(root)
    for key in keys:
        meta = readings.get(key.rpartition("/")[0])
        tag = content_tag(meta) if meta else None
        if tag is None:
            continue
        target = content / _tagged(key, tag)
        if not target.exists():
            _adopt(old_roots, key, target, meta.get("preprocessing_date"))
        out[key] = target
    return out


def _old_caches(images_root: Path) -> list[Path]:
    """The per-version caches, this one first, then the most recently used."""
    parent = images_root.parent
    if not parent.is_dir():
        return [images_root]
    others = sorted((d for d in parent.iterdir()
                     if d.is_dir() and d.name != CONTENT_CACHE_DIR and d != images_root),
                    key=lambda d: d.stat().st_mtime, reverse=True)
    return [images_root, *others]


def _adopt(old_roots: Sequence[Path], key: str, target: Path, rendered: Any) -> bool:
    """Place a page an older per-version cache holds at ``target``, when it was
    fetched after its reading rendered it. True when it did."""
    from datetime import datetime

    try:
        since = datetime.fromisoformat(str(rendered)).timestamp()
    except (TypeError, ValueError):
        return False
    for old_root in old_roots:
        old = old_root / key
        try:
            if not old.is_file() or old.stat().st_mtime < since:
                continue
        except OSError:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
        try:
            try:
                os.link(old, partial)
            except OSError:
                import shutil

                shutil.copy2(old, partial)
            os.replace(partial, target)
            return True
        except OSError as exc:
            partial.unlink(missing_ok=True)
            log.debug("image cache: %s not adopted from %s (%s)", key, old_root, exc)
    return False


def _download_to(client: BlobClient, targets: list[tuple[str, Path]], workers: int) -> None:
    """Fetch each ``(key, local path)`` concurrently. The first failure cancels
    what has not started."""
    if not targets:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_download, client, key, target) for key, target in targets]
        try:
            for future in futures:
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _read_rows(client: BlobClient, key: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in client.read_text(key).splitlines() if line.strip()]


def _download(client: BlobClient, key: str, target: Path) -> None:
    """Fetch one image atomically, to a temporary file unique to this download.

    Unique because the cache is shared: two runs on one corpus version can fetch
    the same page at the same moment, and a shared ``.part`` name would let one
    rename the other's half-written file into place. Losing the race is success
    — the page is there, whole, whoever wrote it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        client.download_file(key, partial)
    except Exception as exc:  # noqa: BLE001 - reported with the one question that matters
        partial.unlink(missing_ok=True)
        if not client.exists(key):
            raise StagingError(
                f"page image {key} is referenced by the corpus but is not in Blob. "
                "Training on a row without its page would teach the model to answer "
                "from text it was told came with an image."
            ) from exc
        raise StagingError(f"could not fetch page image {key}: {exc}") from exc
    try:
        os.replace(partial, target)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        if not target.exists():
            raise StagingError(f"could not place page image {key}: {exc}") from exc


def localize_rows(
    rows: list[dict[str, Any]],
    client: BlobClient,
    images_root: str | Path,
    *,
    workers: int = DOWNLOAD_WORKERS,
) -> list[dict[str, Any]]:
    """Corpus rows with every image Blob key replaced by a local cached file.

    For the GENERATION paths — checkpoint selection and calibration — which send
    rows to vLLM in the corpus's own message shape. vLLM opens an image by path;
    handed a Blob key it opens nothing, every row failed, and each failure was
    scored as an empty answer: every checkpoint 0.0, calibration skipped, and no
    error anywhere. Shares the training image cache, so a page is fetched once
    per corpus version whichever path asks first.
    """
    import copy

    keys = sorted({
        block["image"]
        for row in rows
        for message in row.get("messages") or []
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "image" and isinstance(block.get("image"), str)
    })
    local = localize_keys(client, keys, images_root, workers=workers)

    localized = []
    for row in rows:
        row = copy.deepcopy(row)
        for message in row.get("messages") or []:
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if block.get("type") == "image" and isinstance(block.get("image"), str):
                        block["image"] = local[block["image"]]
        localized.append(row)
    return localized


def localize_keys(
    client: BlobClient, keys: Sequence[str], images_root: str | Path,
    *, workers: int = DOWNLOAD_WORKERS,
) -> dict[str, str]:
    """``{blob key: local absolute path}``, fetching what the cache lacks.

    The one download path every consumer shares — training staging, checkpoint
    selection, calibration, the golden eval — so a page is fetched once per
    cache, atomically, whichever asks first.
    """
    local = image_cache_paths(client, list(keys), images_root, workers=workers,
                              by_content=in_staging_cache(images_root))
    _download_to(client, [(key, path) for key, path in local.items() if not path.exists()], workers)
    return {key: str(local[key].resolve()) for key in keys}


#: Corpus versions whose image cache is kept besides the one being staged: the
#: previous version, which a comparison or a resumed run may still read.
KEEP_OTHER_IMAGE_CACHES = 1


def prune_image_caches(images_root: str | Path, *, keep: int = KEEP_OTHER_IMAGE_CACHES) -> list[str]:
    """Delete the page-image caches of older corpus versions. Returns what went.

    The cache is per corpus version (``paths.staging_train_images_dir``), and
    nothing ever removed one: every rebuild left a full copy of the corpus's
    pages on the volume until it filled and a staging download failed mid-run.
    The current version is never touched; of the others the ``keep`` most
    recently used survive.

    The content-keyed cache (:data:`CONTENT_CACHE_DIR`) is shared by every
    version and never removed whole: of each page's tags (a reading each), the
    :data:`KEEP_CONTENT_TAGS` newest stay (:func:`prune_content_cache`).
    """
    import shutil

    current = Path(images_root).resolve()
    parent = current.parent
    if not parent.is_dir():
        return []
    others = sorted(
        (d for d in parent.iterdir()
         if d.is_dir() and d.resolve() != current and d.name != CONTENT_CACHE_DIR),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    removed = []
    for stale in others[keep:]:
        shutil.rmtree(stale, ignore_errors=True)
        removed.append(str(stale))
    if removed:
        log.info("pruned %d old image cache(s): %s", len(removed), removed)
    removed += prune_content_cache(content_cache_root(current))
    return removed


def prune_content_cache(content_root: str | Path, *, keep: int = KEEP_CONTENT_TAGS) -> list[str]:
    """Delete the older readings' copies of each page in the content cache: of
    ``page_N.<tag>.png`` the ``keep`` most recently written stay. Returns what went."""
    root = Path(content_root)
    if not root.is_dir():
        return []
    by_page: dict[tuple[str, str, str], list[Path]] = {}
    for path in root.rglob("*"):
        parts = path.name.split(".")
        if len(parts) < 3 or parts[-1] == "part" or not path.is_file():
            continue
        # page_3.<tag>.png -> (folder, page_3, png)
        by_page.setdefault((str(path.parent), ".".join(parts[:-2]), parts[-1]), []).append(path)
    removed = []
    for copies in by_page.values():
        copies.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in copies[keep:]:
            stale.unlink(missing_ok=True)
            removed.append(str(stale))
    if removed:
        log.info("pruned %d superseded page image(s) from %s", len(removed), root)
    return removed
