"""Review 2026-10-26 #26 and #32: OCR documents at once, and stop redoing a cycle's inputs.

- ``run_mineru.process_batch(workers=N)`` reads N documents at once against a
  served MinerU, with the serial run's results; the OCR stage uses the same pool
  (``StageContext.ocr_workers``).
- ``find_unprocessed`` reads each ``ocr_meta.json`` once per run (``meta_cache``).
- The corpus build reads its page texts through a pool, in order.
- The page-image cache is keyed by content (the reading behind each page), not
  by corpus version: a page is fetched once across cycles, and again only when
  a new reading renders it; the older per-version caches stay readable.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ocr import mineru_version as mv
from data_pipeline.ocr import run_mineru
from data_pipeline.ocr.run_mineru import OcrError, PageOutput


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="ocr")


class _ConcurrentServed:
    """A served MinerU stand-in that takes a moment per document and counts how
    many it was reading at once."""

    tier = "basic"

    def __init__(self, fail: set[bytes] = frozenset(), boom: set[bytes] = frozenset()):
        self.fail, self.boom = fail, boom
        self.lock = threading.Lock()
        self.now = self.peak = 0

    def environment(self) -> mv.OcrEnvironment:
        return mv.OcrEnvironment("4.0.12", "cuda")

    def meta(self) -> dict:
        return {"mineru_backend": "api", "mineru_tier": self.tier, "read_every_page_by_ocr": True}

    def process(self, pdf_bytes: bytes, *, device: str, max_long_side_px: int) -> list[PageOutput]:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            time.sleep(0.05)
            if pdf_bytes in self.boom:
                raise KeyboardInterrupt("not an OCR failure")
            if pdf_bytes in self.fail:
                raise RuntimeError("MinerU refused it")
            text = f"# {pdf_bytes.decode()}"
            return [PageOutput(page_number=1, markdown=text, image_bytes=pdf_bytes, table_row_count=0)]
        finally:
            with self.lock:
                self.now -= 1


def _seed(client: BlobClient, n: int) -> list[str]:
    ids = []
    for i in range(n):
        source_id = f"policy_{i:04d}"
        client.write_bytes(paths.raw_pdf("policy", source_id), f"PDF {source_id}".encode())
        client.write_json(paths.raw_metadata("policy", source_id),
                          {"source_id": source_id, "doc_type": "policy", "checksum_sha256": f"sha-{i}"})
        ids.append(source_id)
    return ids


def _stored(client: BlobClient, ids: list[str]) -> dict:
    """What a batch wrote, without the time it wrote it."""
    out = {}
    for source_id in ids:
        meta = client.read_json(paths.ocr_meta("policy", source_id))
        meta.pop("preprocessing_date")
        out[source_id] = (meta, client.read_text(paths.processed_page("policy", source_id, 1, "md")))
    return out


def test_documents_read_at_once_give_the_serial_runs_results(client):
    ids = _seed(client, 9)
    fail = {f"PDF {ids[4]}".encode()}

    serial_engine = _ConcurrentServed(fail=fail)
    serial = run_mineru.process_batch(ids, "policy", client, serial_engine)
    serial_store = _stored(client, [i for i in ids if i != ids[4]])

    again = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="ocr")
    _seed(again, 9)
    engine = _ConcurrentServed(fail=fail)
    pooled = run_mineru.process_batch(ids, "policy", again, engine, workers=4)

    assert serial_engine.peak == 1 and engine.peak > 1, "nothing was read at once"
    assert pooled["processed"] == serial["processed"] == [i for i in ids if i != ids[4]]
    assert [sid for sid, _ in pooled["failed"]] == [ids[4]]
    assert _stored(again, pooled["processed"]) == serial_store
    # Read again: every document unchanged, in order.
    assert run_mineru.process_batch(ids, "policy", again, engine, workers=4)["skipped"] == pooled["processed"]


def test_an_error_that_is_not_ocrs_stops_the_pooled_batch_as_it_stops_the_serial_one(client):
    ids = _seed(client, 6)
    engine = _ConcurrentServed(boom={f"PDF {ids[1]}".encode()})
    with pytest.raises(KeyboardInterrupt):
        run_mineru.process_batch(ids, "policy", client, engine, workers=3)


def test_an_ocr_failure_is_one_documents_in_the_pool(client):
    ids = _seed(client, 3)
    engine = _ConcurrentServed(fail={f"PDF {i}".encode() for i in ids})
    outcome = run_mineru.process_batch(ids, "policy", client, engine, workers=3)
    assert [sid for sid, _ in outcome["failed"]] == ids
    assert all(isinstance(reason, str) for _sid, reason in outcome["failed"])
    assert issubclass(OcrError, RuntimeError)


def test_find_unprocessed_reads_each_meta_once_per_run(client):
    ids = _seed(client, 4)
    run_mineru.process_batch(ids[:3], "policy", client, _ConcurrentServed())
    reads = []
    original = client.read_json
    client.read_json = lambda key: (reads.append(key), original(key))[1]

    cache: dict = {}
    for _ in range(3):
        assert run_mineru.find_unprocessed(client, "policy", mineru_version="4.0.12", meta_cache=cache) == [ids[3]]
    assert len([k for k in reads if k.endswith("ocr_meta.json")]) == 3     # three documents, once each
    assert run_mineru.find_unprocessed(client, "policy", mineru_version="4.0.12") == [ids[3]]   # no cache: reads


def test_the_ocr_stage_reads_documents_at_once_against_a_served_mineru(monkeypatch):
    from types import SimpleNamespace

    from orchestration import pipeline_dag

    seen = {}

    def batch(ids, doc_type, raw, engine, *, tenant_id=None, workers=1):
        seen["workers"] = workers
        return {"processed": list(ids), "skipped": [], "failed": []}

    monkeypatch.setattr(run_mineru, "process_batch", batch)
    monkeypatch.setattr(run_mineru, "find_unprocessed", lambda raw, dt, tenant, **held: ["a", "b"])
    ctx = SimpleNamespace(raw=object(), tenant_id=None, doc_types=["policy"], ocr_engine=_ConcurrentServed(),
                          ocr_workers=4, ocr_meta_cache={"stale": {}})
    result = pipeline_dag.stage_preprocessing(ctx)
    assert seen["workers"] == 4 and result.data["processed"] == 2
    assert ctx.ocr_meta_cache == {}, "the readings just written must not be read from the cache"


def test_the_ocr_workers_flag_reaches_the_stage():
    from orchestration import run as cli

    args = cli.build_parser().parse_args(["finetune", "--skip-ingest", "--ocr-workers", "4"])
    assert cli.build_context(args, client=object(), controller=object()).ocr_workers == 4
    args = cli.build_parser().parse_args(["finetune", "--skip-ingest"])
    assert cli.build_context(args, client=object(), controller=object()).ocr_workers == 1
    assert "--ocr-workers" in cli.flag_problem(cli.build_parser().parse_args(["finetune", "--ocr-workers", "0"]))


def test_the_corpus_build_reads_page_texts_in_order_through_a_pool():
    from orchestration.pipeline_dag import read_page_texts

    class Slow:
        def __init__(self):
            self.threads = set()

        def read_text(self, key):
            self.threads.add(threading.get_ident())
            time.sleep(0.01 if key.endswith("1") else 0.0)
            return f"text of {key}"

    keys = [[f"d{d}/p{p}" for p in range(1, n + 1)] for d, n in ((1, 3), (2, 0), (3, 5))]
    store = Slow()
    texts = read_page_texts(store, keys, workers=8)
    assert texts == [[f"text of {k}" for k in doc] for doc in keys]
    assert len(store.threads) > 1
    assert read_page_texts(store, keys, workers=1) == texts


# --------------------------------------------------------------------------
# The page-image cache, by content
# --------------------------------------------------------------------------


def _reading(client: BlobClient, source_id: str, *, rendered: datetime, pixels: bytes, pages: int = 2) -> list[str]:
    """A document's reading as the OCR stage leaves it: its pages and its meta."""
    keys = []
    for page in range(1, pages + 1):
        key = paths.processed_page("policy", source_id, page, "png", "t")
        client.write_bytes(key, pixels + bytes([page]))
        keys.append(key)
    client.write_json(paths.ocr_meta("policy", source_id, "t"), {
        "source_checksum": f"sha-{source_id}", "mineru_version": "4.0.12", "mineru_tier": "basic",
        "ocr_device": "cuda", "resolution_cap_px": 1792, "preprocessing_date": rendered.isoformat(),
        "page_count": pages,
    })
    return keys


def _counting(client: BlobClient) -> list[str]:
    fetched: list[str] = []
    original = client.download_file

    def download(key, local):
        fetched.append(key)
        original(key, local)

    client.download_file = download
    return fetched


def _version_root(tmp_path: Path, version: str) -> Path:
    return tmp_path / "staging" / "train-images" / "tenant_t" / version


def test_a_page_is_fetched_once_across_cycles_until_it_is_read_again(tmp_path):
    from training.stage_data import CONTENT_CACHE_DIR, localize_keys

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    keys = _reading(client, "policy_0001", rendered=datetime.now(UTC) - timedelta(days=1), pixels=b"old")
    fetched = _counting(client)

    first = localize_keys(client, keys, _version_root(tmp_path, "v1"))
    second = localize_keys(client, keys, _version_root(tmp_path, "v2"))      # the next cycle's corpus
    assert first == second and len(fetched) == len(keys)
    assert all(CONTENT_CACHE_DIR in path for path in first.values())
    assert [Path(p).read_bytes() for p in second.values()] == [b"old\x01", b"old\x02"]

    # Read again (a re-OCR): the same keys, new pixels - fetched once more, never the old ones.
    _reading(client, "policy_0001", rendered=datetime.now(UTC), pixels=b"new")
    third = localize_keys(client, keys, _version_root(tmp_path, "v3"))
    assert len(fetched) == 2 * len(keys)
    assert [Path(p).read_bytes() for p in third.values()] == [b"new\x01", b"new\x02"]


def test_an_older_per_version_cache_is_read_when_it_holds_this_reading(tmp_path):
    from training.stage_data import localize_keys

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    keys = _reading(client, "policy_0001", rendered=datetime.now(UTC) - timedelta(hours=1), pixels=b"px")
    old = _version_root(tmp_path, "v1")
    for key in keys:                                  # cached by the old per-version scheme, after the render
        (old / key).parent.mkdir(parents=True, exist_ok=True)
        (old / key).write_bytes(client.read_bytes(key))
    fetched = _counting(client)
    local = localize_keys(client, keys, _version_root(tmp_path, "v2"))
    assert fetched == [] and [Path(p).read_bytes() for p in local.values()] == [b"px\x01", b"px\x02"]


def test_an_older_cache_from_before_a_re_ocr_is_not_read(tmp_path):
    from training.stage_data import localize_keys

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    keys = _reading(client, "policy_0001", rendered=datetime.now(UTC) + timedelta(hours=1), pixels=b"new")
    old = _version_root(tmp_path, "v1")
    for key in keys:                                  # fetched before the document was read again
        (old / key).parent.mkdir(parents=True, exist_ok=True)
        (old / key).write_bytes(b"stale")
    fetched = _counting(client)
    local = localize_keys(client, keys, _version_root(tmp_path, "v2"))
    assert len(fetched) == len(keys) and b"stale" not in [Path(p).read_bytes() for p in local.values()]


def test_a_page_whose_reading_is_unknown_is_cached_per_version_as_before(tmp_path):
    from training.stage_data import localize_keys

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    key = "golden-eval-set/tenant_t/policy_0009/page_1.png"
    client.write_bytes(key, b"frozen")
    root = _version_root(tmp_path, "golden-v1")
    assert localize_keys(client, [key], root) == {key: str((root / key).resolve())}
    # Outside the staging tree (the testing harness's cache): by blob key, as before.
    keys = _reading(client, "policy_0001", rendered=datetime.now(UTC), pixels=b"x")
    harness = tmp_path / "page_cache"
    assert localize_keys(client, keys, harness) == {k: str((harness / k).resolve()) for k in keys}


def test_training_staging_shares_the_content_cache(tmp_path):
    import json

    from training.stage_data import localize_keys, stage_training_data

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    keys = _reading(client, "policy_0001", rendered=datetime.now(UTC) - timedelta(minutes=5), pixels=b"p")
    row = {"source_id": "policy_0001", "doc_type": "policy", "task": "extract", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "image", "image": keys[0]}, {"type": "text", "text": "t1"},
                                     {"type": "image", "image": keys[1]}, {"type": "text", "text": "t2"}]},
        {"role": "assistant", "content": "{}"}]}
    client.write_text("corpus/x/epoch_1.jsonl", json.dumps(row) + "\n")
    local = localize_keys(client, keys, _version_root(tmp_path, "v1"))       # an eval read them first
    staged = stage_training_data(["corpus/x/epoch_1.jsonl"], None, client, tmp_path / "run",
                                 images_root=_version_root(tmp_path, "v2"))
    assert staged.fetched == 0 and staged.images == 2
    swift = json.loads(Path(staged.epoch_files[0]).read_text(encoding="utf-8"))
    assert swift["images"] == [local[keys[0]], local[keys[1]]]


def test_pruning_keeps_the_content_cache_and_its_newest_readings(tmp_path):
    from training.stage_data import CONTENT_CACHE_DIR, prune_image_caches

    base = tmp_path / "staging" / "train-images" / "tenant_t"
    for version in ("v1", "v2", "v3"):
        (base / version).mkdir(parents=True)
    page = base / CONTENT_CACHE_DIR / "processed" / "tenant_t" / "policy" / "p1"
    page.mkdir(parents=True)
    for age, tag in enumerate(("aaaa", "bbbb", "cccc")):
        copy = page / f"page_1.{tag}.png"
        copy.write_bytes(tag.encode())
        stamp = time.time() - 100 * (3 - age)
        os.utime(copy, (stamp, stamp))
    (page / "page_1.dddd.png.1.x.part").write_bytes(b"part")

    removed = prune_image_caches(base / "v3", keep=1)
    assert (base / CONTENT_CACHE_DIR).is_dir()
    assert sorted(p.name for p in page.iterdir()) == ["page_1.bbbb.png", "page_1.cccc.png", "page_1.dddd.png.1.x.part"]
    assert str(page / "page_1.aaaa.png") in removed
