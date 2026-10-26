"""training.stage_data — the corpus, where ms-swift can read it, in its format.

The failures this guards were invisible to every other test, because nothing
here ever launches the trainer: the corpus rows could not be loaded by the
reader the trainer uses, and the paths the trainer was given did not exist on
the machine it runs on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.dataset_builder.build_jsonl import (
    SourceDocument,
    build_corpus,
    train_rows_by_epoch,
    write_jsonl,
)
from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits
from training.stage_data import SWIFT_IMAGE_TAG, StagingError, stage_training_data, to_swift_row

FIXTURES = Path(__file__).resolve().parent / "fixtures"
def _png(width: int, height: int) -> bytes:
    """A PNG header with real dimensions: all the length check reads."""
    import struct

    return (
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height) + b"\x00" * 8
    )


#: A US-Letter page rendered inside the extraction budget.
PNG = _png(1275, 1650)


@pytest.fixture(autouse=True)
def _as_if_on_the_pod(monkeypatch):
    """A real-run test runs here, not on a pod: no CUDA, and ms-swift's template
    is not installed. The guard and the masking check each have their own tests."""
    from training import train as T

    monkeypatch.setenv(T.OFF_POD_ENV, "1")
    # A pod's clean checkout, whatever state this one is in (a real run refuses a dirty tree).
    monkeypatch.setattr(T, "worktree_state", lambda: (False, None))
    monkeypatch.setattr(T, "_masking_encoder", lambda swift: _stub_masking_encoder())


def _stub_masking_encoder():
    """Encodes a row as [prompt..., HEADER, answer..., END, NEWLINE] with the
    answer supervised through the newline, as ms-swift's ChatML suffix does."""
    header, end, newline = [900, 901], 999, 198

    def encode(row):
        prompt = [1] * 5
        answer = [7] * 3
        ids = prompt + header + answer + [end, newline]
        labels = [-100] * (len(prompt) + len(header)) + answer + [end, newline]
        return {"input_ids": ids, "labels": labels}

    return encode, header, end, [newline]


def _stub_tokens(texts):
    """A stand-in for the model tokenizer: roughly four characters a token."""
    return [len(t) // 4 + 1 for t in texts]


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _golden(name):
    return json.loads((FIXTURES / f"golden/{name}.golden.json").read_text(encoding="utf-8"))


def _ocr(name):
    return (FIXTURES / f"ocr/{name}_page_1.md").read_text(encoding="utf-8")


def _documents():
    """Both row shapes the corpus holds: a flat Loss Run and a windowed policy,
    with `lob` written the two ways real labels arrive — a string and a list."""
    docs = []
    for i in range(1, 11):
        docs.append(SourceDocument(
            source_id=f"lossrun_{i:04d}", doc_type="lossrun", golden_label=_golden("lossrun_0001"),
            ocr_pages=[_ocr("lossrun_0001")],
            image_paths=[f"processed/default/lossrun/lossrun_{i:04d}/page_1.png"],
            lob=["workers_comp"], tenant_id="default",
        ))
        docs.append(SourceDocument(
            source_id=f"policy_{i:04d}", doc_type="policy", golden_label=_golden("policy_0001"),
            ocr_pages=[_ocr("policy_0001")],
            image_paths=[f"processed/default/policy/policy_{i:04d}/page_1.png"],
            lob="workers_comp", tenant_id="default",
        ))
    return docs


def _seed(client):
    docs = _documents()
    groups: dict[str, list[GroupRecord]] = {}
    for d in docs:
        groups.setdefault(d.doc_type, []).append(
            GroupRecord(group_id=d.source_id, doc_type=d.doc_type, source_ids=[d.source_id])
        )
    built = build_corpus(docs, assign_group_splits(groups, seed=42))
    for epoch, rows in train_rows_by_epoch(built).items():
        client.write_text(paths.corpus_epoch_file("v1", epoch), write_jsonl(rows))
    client.write_text(paths.corpus_eval_split("v1", "val"),
                      write_jsonl(built.rows_by_split["val"]))
    for d in docs:
        for key in d.image_paths:
            client.write_bytes(key, PNG)
    return built


EPOCHS = [paths.corpus_epoch_file("v1", e) for e in (1, 2, 3, 4)]


# --------------------------------------------------------------------------
# The row format
# --------------------------------------------------------------------------

def _row():
    return {"source_id": "x", "messages": [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": [
            {"type": "image", "image": "k/page_1.png"},
            {"type": "text", "text": "<page 1 of 2>\n\none"},
            {"type": "image", "image": "k/page_2.png"},
            {"type": "text", "text": "<page 2 of 2>\n\ntwo"},
        ]},
        {"role": "assistant", "content": "{}"},
    ], "lob": ["workers_comp"], "window_pages": [1, 2]}


def test_a_swift_row_is_string_content_with_one_placeholder_per_image_in_order():
    row = to_swift_row(_row(), lambda key: f"/local/{key}")
    assert set(row) == {"messages", "images"}
    assert all(isinstance(m["content"], str) for m in row["messages"])
    user = row["messages"][1]["content"]
    # Joined with nothing between: that is how the Qwen template concatenates a
    # content list, so the rendered prompt does not change.
    assert user == "<image><page 1 of 2>\n\none<image><page 2 of 2>\n\ntwo"
    assert row["images"] == ["/local/k/page_1.png", "/local/k/page_2.png"]
    assert user.count(SWIFT_IMAGE_TAG) == len(row["images"])


def test_text_containing_the_placeholder_is_refused():
    """It would be read as an image and pair every later page with the wrong text."""
    bad = _row()
    bad["messages"][1]["content"][1]["text"] = "see <image> below"
    with pytest.raises(StagingError, match="special tag"):
        to_swift_row(bad, lambda key: key)


@pytest.mark.parametrize("tag", ["<video>", "<audio>", "<bbox>", "<ref-object>"])
def test_every_trainer_tag_is_refused_not_just_image(tag):
    bad = _row()
    bad["messages"][2]["content"] = f'{{"note": "{tag}"}}'
    with pytest.raises(StagingError, match="special tag"):
        to_swift_row(bad, lambda key: key)


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------

def test_the_corpus_as_written_cannot_be_loaded_by_the_trainers_reader(client, tmp_path):
    """The regression this module exists for, pinned so it cannot quietly return."""
    pyarrow_json = pytest.importorskip("pyarrow.json")
    import pyarrow as pa

    _seed(client)
    local = tmp_path / "raw.jsonl"
    local.write_text(client.read_text(EPOCHS[0]), encoding="utf-8")
    with pytest.raises(pa.ArrowInvalid):
        pyarrow_json.read_json(local)


def test_staged_files_load_with_the_trainers_reader(client, tmp_path):
    pyarrow_json = pytest.importorskip("pyarrow.json")

    _seed(client)
    staged = stage_training_data(EPOCHS, paths.corpus_eval_split("v1", "val"), client, tmp_path)

    for local in [*staged.epoch_files, staged.val_path]:
        table = pyarrow_json.read_json(local)
        assert table.num_rows > 0
        assert set(table.column_names) == {"messages", "images"}


def test_every_image_is_a_local_file_that_exists(client, tmp_path):
    built = _seed(client)
    staged = stage_training_data(EPOCHS, None, client, tmp_path)
    # One page per document; only train documents are in the epoch files.
    assert staged.images == len({r["source_id"] for r in built.rows_by_split["train"]})
    for local in staged.epoch_files:
        for line in Path(local).read_text(encoding="utf-8").splitlines():
            for image in json.loads(line)["images"]:
                assert Path(image).is_absolute() and Path(image).read_bytes() == PNG


def test_a_missing_page_image_is_refused(client, tmp_path):
    built = _seed(client)
    trained = next(r["source_id"] for r in built.rows_by_split["train"] if r["doc_type"] == "policy")
    client.delete(f"processed/default/policy/{trained}/page_1.png")
    with pytest.raises(StagingError, match="not in Blob"):
        stage_training_data(EPOCHS, None, client, tmp_path)


def test_every_row_stores_lob_as_a_list(client):
    """Strings from the review tool, lists from the importers: one JSON column
    holding both is refused by pyarrow."""
    built = _seed(client)
    assert {type(r["lob"]) for r in built.all_rows} == {list}


# --------------------------------------------------------------------------
# train() hands ms-swift the staged files
# --------------------------------------------------------------------------

def test_a_real_run_points_ms_swift_at_local_staged_files(client, tmp_path, monkeypatch):
    from registry_utils.models import DataStats
    from training import train as T

    _seed(client)
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path))
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    monkeypatch.setattr(T, "_token_counter", lambda: _stub_tokens)
    launched = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))

    T.train(corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
            data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0),
            dry_run=False)

    (config,) = launched
    datasets = config.args["dataset"]
    assert datasets and all(Path(p).is_file() for p in datasets), datasets
    assert all(str(tmp_path) in p for p in datasets)
    assert Path(config.args["val_dataset"][0]).is_file()


# --------------------------------------------------------------------------
# Budgets, validation, and what a run records
# --------------------------------------------------------------------------

def test_a_document_over_its_task_budget_is_set_aside_not_truncated():
    """A clipped target trains the model to stop early, so the row is rejected —
    and with it the document, or it would be in some epochs and not others."""
    docs = _documents()
    huge = docs[0]
    huge.ocr_pages = ["claim row " * 60_000]
    groups: dict[str, list[GroupRecord]] = {}
    for d in docs:
        groups.setdefault(d.doc_type, []).append(
            GroupRecord(group_id=d.source_id, doc_type=d.doc_type, source_ids=[d.source_id])
        )
    built = build_corpus(docs, assign_group_splits(groups, seed=42))

    assert huge.source_id not in {r["source_id"] for r in built.all_rows}
    assert any(sid == huge.source_id and "budget" in why for sid, why in built.skipped)
    assert built.cap_report.rejected and built.cap_report.accepted
    train_rows_by_epoch(built)   # every epoch still holds every remaining document


def test_ms_swift_is_told_never_to_cut_a_row():
    from training.train import build_training_config

    swift, _ = build_training_config(
        corpus_paths=["e1", "e2", "e3", "e4"], output_dir="/out", val_paths=["v"]
    )
    assert swift.args["truncation_strategy"] == "delete"


def _train(client, monkeypatch, tmp_path, **kwargs):
    from registry_utils.models import DataStats
    from training import train as T

    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path))
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    monkeypatch.setattr(T, "_token_counter", lambda: _stub_tokens)
    launched = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    _swift, manifest = T.train(
        corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
        data_stats=DataStats(train_examples=999, val_examples=999, test_examples=999),
        **kwargs,
    )
    return manifest, launched


def test_a_run_with_no_validation_rows_stops_before_launch(client, tmp_path, monkeypatch):
    """Checkpoint selection and early stopping read validation; without it the
    run fails after the GPU is paid for, or ships a checkpoint nobody chose."""
    from training.train import TrainingError

    _seed(client)
    client.write_text(paths.corpus_eval_split("v1", "val"), "")
    with pytest.raises(TrainingError, match="no validation rows"):
        _train(client, monkeypatch, tmp_path, dry_run=False)


def test_the_policy_run_records_its_own_counts_not_the_corpus(client, tmp_path, monkeypatch):
    """The corpus holds Loss Runs as well; the run trains the policy rows only,
    and its manifest counts what it trains, not every row the corpus holds."""
    from common.config import training_config

    built = _seed(client)
    client.write_text(paths.corpus_eval_split("v1", "test"), write_jsonl(built.rows_by_split["test"]))
    manifest, _ = _train(client, monkeypatch, tmp_path, dry_run=True)
    epochs = set(range(1, int(training_config("unified")["optimization"]["num_train_epochs"]) + 1))

    def policy_rows(split, epochs=None):
        return sum(
            1 for r in built.rows_by_split[split]
            if r["doc_type"] == "policy" and (epochs is None or r.get("epoch") in epochs)
        )

    stats = manifest.data_stats
    # Only the epochs the run trains, of the policy rows.
    assert stats.train_examples == policy_rows("train", epochs=epochs)
    assert stats.val_examples == policy_rows("val")
    assert stats.test_examples == policy_rows("test")
    assert stats.train_examples < sum(1 for r in built.rows_by_split["train"]
                                      if r.get("epoch") in epochs), "the Loss Runs were counted"


# --------------------------------------------------------------------------
# Re-review: tags at corpus build, document-level budgets, safe staging
# --------------------------------------------------------------------------

def _build(docs):
    groups: dict[str, list[GroupRecord]] = {}
    for d in docs:
        groups.setdefault(d.doc_type, []).append(
            GroupRecord(group_id=d.source_id, doc_type=d.doc_type, source_ids=[d.source_id])
        )
    return build_corpus(docs, assign_group_splits(groups, seed=42))


def test_a_document_whose_text_holds_a_trainer_tag_is_set_aside_at_build():
    """At build, not on the pod: one bad page must not abort every training run
    on the corpus version it was pinned into."""
    docs = _documents()
    docs[0].ocr_pages = ["Claims <bbox> listing"]
    built = _build(docs)
    assert docs[0].source_id not in {r["source_id"] for r in built.all_rows}
    assert any(sid == docs[0].source_id and "<bbox>" in why for sid, why in built.skipped)


def test_the_budget_report_counts_documents_and_names_the_real_limit(monkeypatch):
    """One row of four over budget is the whole document lost, and a target that
    overruns its OUTPUT reservation is not fixed by raising the total cap."""
    from data_pipeline.dataset_builder import cap_check

    real = cap_check.sequence_for_task
    monkeypatch.setattr(
        cap_check, "sequence_for_task",
        lambda task, doc_type=None: {**real(task, doc_type), "max_output_tokens": 5},
    )
    built = _build(_documents())
    report = built.cap_report
    assert report.documents_rejected == 20 and report.documents_accepted == 0
    assert report.accepted == 0, "rows of a set-aside document are not counted as accepted"
    assert all("reserved" in r["reason"] for r in report.rejected)
    assert not any("raise the cap" in r["reason"] for r in report.rejected)


def test_an_interrupted_download_leaves_no_image_a_later_run_would_trust(client, tmp_path):
    _seed(client)

    flaky = BlobClient(backend=client._backend, container="main", raw_container="raw")
    calls = {"n": 0}
    original = flaky.download_file

    def die_once(key, local):
        calls["n"] += 1
        if calls["n"] == 1:
            Path(local).write_bytes(b"\x89PNG half")      # a partial write, then the pod dies
            raise ConnectionError("preempted")
        original(key, local)

    flaky.download_file = die_once
    with pytest.raises(StagingError):
        stage_training_data(EPOCHS, None, flaky, tmp_path / "run", images_root=tmp_path / "cache",
                            workers=1)
    # Images that finished are whole and may be reused; the one that died left
    # nothing under its real name, and no partial file behind.
    cached = list((tmp_path / "cache").rglob("*.png"))
    assert all(p.read_bytes() == PNG for p in cached), "a truncated image survived"
    assert not list((tmp_path / "cache").rglob("*.part"))

    staged = stage_training_data(EPOCHS, None, client, tmp_path / "run",
                                 images_root=tmp_path / "cache")
    for local in staged.epoch_files:
        for line in Path(local).read_text(encoding="utf-8").splitlines():
            assert all(Path(i).read_bytes() == PNG for i in json.loads(line)["images"])


def test_runs_on_one_corpus_version_share_its_images(client, tmp_path):
    _seed(client)
    first = stage_training_data(EPOCHS, None, client, tmp_path / "unified",
                                images_root=tmp_path / "cache")
    second = stage_training_data(EPOCHS, None, client, tmp_path / "lossrun",
                                 images_root=tmp_path / "cache")
    assert first.fetched == first.images > 0
    assert second.fetched == 0, "a second run re-downloaded images the version already had"


def test_a_run_that_will_be_refused_downloads_nothing(client, tmp_path, monkeypatch):
    from training.train import TrainingError

    _seed(client)
    client.write_text(paths.corpus_eval_split("v1", "val"), "")
    with pytest.raises(TrainingError):
        _train(client, monkeypatch, tmp_path, dry_run=False)
    assert not list(tmp_path.rglob("*.png")), "images were staged for a run that was refused"


def test_encoding_is_strict_and_the_pixel_budget_is_explicit():
    from common.config import vision_for_task
    from training.train import build_training_config

    swift, _ = build_training_config(
        corpus_paths=["e1", "e2", "e3", "e4"], output_dir="/out", val_paths=["v"]
    )
    assert swift.args["strict"] is True
    assert swift.env["MAX_PIXELS"] == str(vision_for_task("extract")["max_pixels"])


# --------------------------------------------------------------------------
# Review 3: real lengths before launch, the pixel budget, safe shared staging
# --------------------------------------------------------------------------

def test_image_tokens_follow_the_resize_rule():
    """Qwen's smart_resize: sides rounded to 32, area scaled to the budget."""
    from training.length_check import image_tokens

    # Inside the budget: rounded, not scaled. 1280x1664 / (32*32) = 2080.
    assert image_tokens(1275, 1650, min_pixels=200_704, max_pixels=2_483_712) == 2080
    # Over the budget: scaled down, never past it.
    big = image_tokens(4000, 6000, min_pixels=200_704, max_pixels=2_483_712)
    assert big * 32 * 32 <= 2_483_712
    # Under the floor: scaled up to it.
    assert image_tokens(100, 100, min_pixels=200_704, max_pixels=2_483_712) * 1024 >= 200_704


def test_a_row_over_max_length_by_the_real_count_refuses_the_run(client, tmp_path, monkeypatch):
    """The corpus estimate passed it; the real tokenizer does not. Refused before
    launch, not hours in when strict encoding would stop the run."""
    from training import train as T
    from training.train import TrainingError

    _seed(client)
    monkeypatch.setattr(T, "_token_counter", lambda: (lambda texts: [10**6 for _ in texts]))
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path))
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    launched = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    from registry_utils.models import DataStats

    with pytest.raises(TrainingError, match="exceed max_length"):
        T.train(corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
                data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0),
                dry_run=False)
    assert not launched


def test_the_budget_is_passed_in_both_forms_and_recorded():
    from training.train import build_training_config

    swift, recorded = build_training_config(
        corpus_paths=["e1", "e2", "e3", "e4"], output_dir="/out", val_paths=["v"]
    )
    env = swift.env
    assert int(env["IMAGE_MAX_TOKEN_NUM"]) == int(env["MAX_PIXELS"]) // 1024
    assert int(env["IMAGE_MIN_TOKEN_NUM"]) == int(env["MIN_PIXELS"]) // 1024
    assert recorded.pixel_budget == {k: int(v) for k, v in env.items()}


def test_a_row_budgeted_differently_from_the_run_is_refused_before_any_download(client, tmp_path):
    """A thumbnail task resized to the extraction budget trains at ten times the
    pixels serving sends it."""
    _seed(client)
    rows = [json.loads(line) for line in client.read_text(EPOCHS[0]).splitlines() if line.strip()]
    rows[0]["task"] = "classify"
    client.write_text(EPOCHS[0], "".join(json.dumps(r) + "\n" for r in rows))

    with pytest.raises(StagingError, match="budgeted at"):
        stage_training_data(EPOCHS, None, client, tmp_path / "run",
                            images_root=tmp_path / "cache", max_pixels=2_483_712)
    assert not list((tmp_path / "cache").rglob("*.png")), "images fetched for a refused run"


def test_a_bad_row_refuses_staging_before_any_download(client, tmp_path):
    _seed(client)
    rows = [json.loads(line) for line in client.read_text(EPOCHS[0]).splitlines() if line.strip()]
    rows[-1]["messages"][-1]["content"] = '{"note": "<bbox>"}'
    client.write_text(EPOCHS[0], "".join(json.dumps(r) + "\n" for r in rows))

    with pytest.raises(StagingError, match="special tag"):
        stage_training_data(EPOCHS, None, client, tmp_path / "run", images_root=tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("*.png"))


def test_two_downloads_of_one_page_do_not_share_a_temporary_file(client, tmp_path):
    """The cache is shared by runs on one version; a fixed .part name would let
    one run rename the other's half-written page into place."""
    from training.stage_data import _download

    _seed(client)
    key = "processed/default/lossrun/lossrun_0001/page_1.png"
    seen: list[str] = []
    original = client.download_file

    def record(k, local):
        seen.append(str(local))
        original(k, local)

    client.download_file = record
    target = tmp_path / "cache" / key
    _download(client, key, target)
    _download(client, key, target)            # a second run, same page
    assert len(set(seen)) == 2 and target.read_bytes() == PNG


def test_the_build_counts_every_document_it_set_aside_by_reason():
    docs = _documents()
    docs[0].ocr_pages = ["<image> in the text"]
    docs[1].ocr_pages = ["claim row " * 60_000]
    built = _build(docs)
    assert built.set_aside["trainer_tag"] == 1
    assert built.set_aside["budget"] == 1
    assert built.cap_report.rejection_rate == pytest.approx(1 / 19)


# --------------------------------------------------------------------------
# Phase 1: registry safety, output reservation
# --------------------------------------------------------------------------

def test_a_trained_run_is_never_overwritten(client, tmp_path, monkeypatch):
    """Not by a re-run, and not by a dry run, which used to replace a promoted
    manifest with a fresh "training" one."""
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import write_manifest
    from training.train import TrainingError

    _seed(client)
    manifest, _ = _train(client, monkeypatch, tmp_path, dry_run=True)
    manifest.status = "trained"
    write_manifest(manifest, client)

    with pytest.raises(TrainingError, match="never overwritten"):
        _train(client, monkeypatch, tmp_path, dry_run=True)
    assert get(manifest.run_id, client).status == "trained"


def test_a_target_over_its_output_reservation_is_caught_before_launch(tmp_path):
    """The corpus estimate under-counts digits — exactly the text a target is
    full of — so the pod re-checks each target against its own task cap."""
    from training.length_check import measure

    image = tmp_path / "p.png"
    image.write_bytes(PNG)
    row = {"messages": [{"role": "system", "content": "s"},
                        {"role": "user", "content": "<image>t"},
                        {"role": "assistant", "content": "1234567890" * 50}],
           "images": [str(image)]}
    staged = tmp_path / "e.jsonl"
    staged.write_text(json.dumps(row) + "\n", encoding="utf-8")

    over = measure([staged], max_length=10**6, count_tokens=_stub_tokens,
                   min_pixels=200_704, max_pixels=2_483_712,
                   output_caps={str(staged): [16]})
    assert not over.ok and over.over[0]["output_cap"] == 16
    fine = measure([staged], max_length=10**6, count_tokens=_stub_tokens,
                   min_pixels=200_704, max_pixels=2_483_712,
                   output_caps={str(staged): [10**4]})
    assert fine.ok


def test_a_run_whose_masking_supervises_the_prompt_is_refused(client, tmp_path, monkeypatch):
    """The masking check held in training.data_collator now runs before launch."""
    from registry_utils.models import DataStats
    from training import train as T

    _seed(client)
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path))
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    monkeypatch.setattr(T, "_token_counter", lambda: _stub_tokens)
    header, end = [900, 901], 999

    def leaky(row):
        ids = [1] * 5 + header + [7, 7, 7, end]
        return {"input_ids": ids, "labels": list(ids)}   # the prompt is supervised too

    monkeypatch.setattr(T, "_masking_encoder", lambda swift: (leaky, header, end, []))
    launched = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    with pytest.raises(T.TrainingError, match="masking"):
        T.train(corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
                data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0),
                dry_run=False)
    assert not launched
