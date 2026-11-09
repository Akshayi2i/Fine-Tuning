"""Pictures are not page text, and every labelled document reaches the corpus or the build stops and names it.

Cycle 1 (2026-11-07, on the pod): a blank "Hazardous Materials listing" form OCR'd with 14 pictures inside its
table gave 156,140 characters where the page prints 329; rows went over their token budget, 8 of 84 policies were
set aside whole, under the 25% guard, and training ran without them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.ocr_text import clean_page_text, strip_pictures
from data_pipeline.data_health import assess, same_image_different_text, stop_message
from orchestration.pipeline_dag import PipelineError, load_labeled_documents, read_page_texts
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import make_context, seed_corpus, seed_policy_copy, seeded_policies

PICTURE = '<img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD' + "A" * 5000 + '"/>'
HAZMAT_TABLE = ("<table><tr><td>Identifying Symbol</td><td>Total Amount</td></tr>"
                f"<tr><td>{PICTURE}</td><td></td></tr><tr><td>{PICTURE}</td><td></td></tr></table>")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


# -- pictures are not page text ------------------------------------------------


def test_picture_markup_is_taken_out_and_counted():
    cleaned = strip_pictures("Total Amount " + PICTURE + " | ![](images/abc.jpg) | <IMG SRC='x.png'> done")
    assert cleaned.pictures == 3
    assert "img" not in cleaned.text.lower() and "base64" not in cleaned.text and "images/" not in cleaned.text
    assert cleaned.text.startswith("Total Amount") and cleaned.text.endswith("done")
    assert cleaned.characters_removed == len(PICTURE) + len("![](images/abc.jpg)") + len("<IMG SRC='x.png'>")


def test_text_without_pictures_is_unchanged():
    text = "Policy Number: GL 1234\n<table><tr><td>Limit</td><td>$1,000,000</td></tr></table>\n![not an image"
    assert clean_page_text(text) == text
    assert strip_pictures(text).pictures == 0
    assert clean_page_text(None) == ""


def test_a_picture_inside_a_served_mineru_table_is_not_page_text():
    """Serving reads this function's text directly, so the picture is gone before serving or training sees it."""
    from data_pipeline.ocr.mineru_api import pages_from_structured_content

    content = {"pages": [{"page_idx": 0, "blocks": [
        {"type": "paragraph_title", "content": "V Hazardous Materials listing", "level": 2},
        {"type": "table", "content": HAZMAT_TABLE},
        {"type": "table", "content": PICTURE},
    ]}]}
    [page] = pages_from_structured_content(content, 1)
    assert "Identifying Symbol" in page and "Total Amount" in page and "Hazardous Materials" in page
    assert "base64" not in page and "<img" not in page
    assert len(page) < 300


def test_a_picture_inside_a_mineru3_table_is_not_page_text():
    from data_pipeline.ocr.run_mineru import pages_from_content_list

    [page] = pages_from_content_list([{"type": "table", "page_idx": 0, "table_body": HAZMAT_TABLE,
                                       "table_caption": ["Hazardous Materials"]}], 1)
    assert "Total Amount" in page and "Hazardous Materials" in page
    assert "base64" not in page and "<img" not in page


def test_page_text_stored_with_pictures_is_read_clean(client):
    """Text OCR'd before this fix trains clean without a new OCR run; ``clean=False`` gives it as stored."""
    client.write_text("processed/x/page_1.md", "Total Amount " + PICTURE)
    client.write_text("processed/x/page_2.md", "Named Insured: Acme")
    keys = [["processed/x/page_1.md", "processed/x/page_2.md"]]
    assert read_page_texts(client, keys) == [["Total Amount ", "Named Insured: Acme"]]
    assert read_page_texts(client, keys, clean=False)[0][0].endswith(PICTURE)


def test_the_corpus_loader_cleans_stored_pictures_and_records_where(client, controller):
    seeded = seeded_policies(seed_corpus(client))
    key = paths.processed_page("policy", seeded[0], 1, "md")
    client.write_text(key, client.read_text(key) + "\n" + PICTURE + PICTURE)
    ctx = make_context(client, controller)
    documents = {d.source_id: d for d in load_labeled_documents(ctx)}
    assert "base64" not in documents[seeded[0]].ocr_pages[0]
    assert ctx.ocr_pictures == [{"source_id": seeded[0], "page": 1, "pictures": 2,
                                 "characters_removed": 2 * len(PICTURE)}]


# -- documents that would not reach the corpus ----------------------------------


def _built(kept: list[str], skipped: list[tuple[str, str]]):
    return SimpleNamespace(all_rows=[{"source_id": s} for s in kept], skipped=skipped,
                           cap_report=SimpleNamespace(as_dict=lambda: {"documents_rejected": len(skipped)}))


def _document(source_id: str, pages: list[str]):
    return SimpleNamespace(source_id=source_id, ocr_pages=pages)


def test_each_missing_document_is_named_with_why_and_stops_until_agreed():
    documents = [_document("policy_0001", ["a"]), _document("policy_0002", ["b"]), _document("policy_0003", ["c"])]
    built = _built(["policy_0001"], [("policy_0002", "policy_0002: 3 of 225 row(s) exceed their task budget")])
    health = assess(documents, built, not_loaded=[("policy_0009", "labelled but never OCR'd: run stage 2")])

    assert [s for s, _ in health.missing] == ["policy_0009", "policy_0002", "policy_0003"]
    assert "the build recorded no reason" in dict(health.missing)["policy_0003"]
    message = stop_message(health)
    assert message is not None and "3 labelled document(s) would not reach the corpus" in message
    assert "policy_0002: policy_0002: 3 of 225 row(s) exceed" in message
    assert "--accept-set-aside policy_0009,policy_0002,policy_0003" in message

    assert stop_message(health, {"policy_0009", "policy_0002", "policy_0003"}) is None
    assert "policy_0003" in stop_message(health, {"policy_0009", "policy_0002"})
    assert health.accepted_but_not_missing({"policy_0002", "policy_0077"}) == ["policy_0077"]


def test_a_corpus_that_keeps_everything_does_not_stop():
    documents = [_document("policy_0001", ["a"]), _document("policy_0002", ["b"])]
    health = assess(documents, _built(["policy_0001", "policy_0002"], []))
    assert health.missing == [] and stop_message(health) is None
    assert "every loaded document reaches the corpus" in health.render()


def test_blank_pages_largest_pages_and_pictures_are_reported():
    documents = [_document("policy_0001", ["x" * 10, "", "  \n"]), _document("policy_0002", ["y" * 500, "z"])]
    pictures = [{"source_id": "policy_0001", "page": 1, "pictures": 1, "characters_removed": 40},
                {"source_id": "policy_0002", "page": 1, "pictures": 14, "characters_removed": 155_811}]
    health = assess(documents, _built(["policy_0001", "policy_0002"], []), picture_pages=pictures,
                    excluded_eval=["policy_0007"], twins_left_out=["policy_0100"], largest=2)

    assert health.blank_pages == {"policy_0001": [2, 3]}
    assert health.largest_pages == [{"source_id": "policy_0002", "page": 1, "characters": 500},
                                    {"source_id": "policy_0001", "page": 1, "characters": 10}]
    assert health.picture_pages[0]["source_id"] == "policy_0002"
    assert health.documents_loaded == 4
    summary = health.as_dict()["summary"]
    assert summary["pictures_removed"] == 15 and summary["blank_pages"] == 2
    text = health.render()
    assert "policy_0002 page 1: 14 picture(s), 155,811 characters" in text
    assert "frozen eval set and its families: 1" in text and "twins over the per-seed cap: 1" in text
    assert "same image, different text: not checked" in text


def test_the_same_page_image_read_into_different_text_is_found(client):
    client.write_bytes("p/a1.png", b"form page")
    client.write_bytes("p/b1.png", b"form page")
    client.write_bytes("p/c1.png", b"form page")
    client.write_bytes("p/a2.png", b"other page")
    client.write_bytes("p/b2.png", b"other page")
    documents = [
        SimpleNamespace(source_id="policy_0065", image_paths=["p/a1.png", "p/a2.png"], ocr_pages=["Total Amount", "X"]),
        SimpleNamespace(source_id="policy_0066", image_paths=["p/b1.png", "p/b2.png"], ocr_pages=["Tota1 Amount", "x"]),
        SimpleNamespace(source_id="policy_0067", image_paths=["p/c1.png"], ocr_pages=["Total  Amount"]),
    ]
    groups, unread = same_image_different_text(client, documents, workers=2)
    assert unread == []
    # Page 2 differs only in case: the same text, as near_copy compares it. Page 1 has two texts in three pages.
    assert len(groups) == 1
    assert groups[0]["distinct_texts"] == 2
    assert {(p["source_id"], p["page"]) for p in groups[0]["pages"]} == {
        ("policy_0065", 1), ("policy_0066", 1), ("policy_0067", 1)}


# -- the dataset build ---------------------------------------------------------


def _four_policies(client) -> None:
    seed_corpus(client)
    seed_policy_copy(client, "policy_0003", like="policy_0001")
    seed_policy_copy(client, "policy_0004", like="policy_0002")


def _one_more_set_aside(monkeypatch, source_id: str = "policy_0099"):
    """The real build, plus one document it set aside, as the cap check does."""
    from data_pipeline.dataset_builder import build_jsonl
    from orchestration import pipeline_dag

    real = build_jsonl.build_corpus

    def build(*args, **kwargs):
        built = real(*args, **kwargs)
        # Every split filled from the four policies (two fixtures, two copies), none of them dropped.
        rows = built.all_rows
        ids = sorted({r["source_id"] for r in rows})
        assert len(ids) == 4
        of = {i: [r for r in rows if r["source_id"] == i] for i in ids}
        built.rows_by_split = {"train": of[ids[0]] + of[ids[3]], "val": of[ids[1]], "test": of[ids[2]]}
        built.skipped.append((source_id, f"{source_id}: 3 of 225 row(s) exceed their task budget"))
        return built

    monkeypatch.setattr(pipeline_dag, "build_corpus", build, raising=False)
    monkeypatch.setattr(build_jsonl, "build_corpus", build)


def test_the_build_stops_on_a_set_aside_document_and_writes_nothing_but_the_report(client, controller, monkeypatch):
    from orchestration import pipeline_dag

    _four_policies(client)
    _one_more_set_aside(monkeypatch)
    ctx = make_context(client, controller, dry_run=False)
    with pytest.raises(PipelineError) as stopped:
        pipeline_dag.stage_dataset_build(ctx)
    assert "policy_0099: policy_0099: 3 of 225 row(s) exceed their task budget" in str(stopped.value)
    assert "--accept-set-aside policy_0099" in str(stopped.value)
    # It says what was written: the report, and no corpus file.
    assert "The corpus was not built; only its data health report was written" in str(stopped.value)
    assert "Nothing was written" not in str(stopped.value)
    assert not client.exists(paths.corpus_manifest(ctx.corpus, None))
    report = client.read_json(paths.corpus_data_health(ctx.corpus, None))
    assert report["set_aside"][0]["source_id"] == "policy_0099"
    assert report["summary"]["documents_missing_not_accepted"] == 1


def test_a_document_agreed_to_stay_out_lets_the_build_finish_and_is_recorded(client, controller, monkeypatch):
    from orchestration import pipeline_dag

    _four_policies(client)
    _one_more_set_aside(monkeypatch)
    ctx = make_context(client, controller, dry_run=False, accept_set_aside=frozenset({"policy_0099"}))
    result = pipeline_dag.stage_dataset_build(ctx)
    assert result.status == "completed"
    manifest = client.read_json(paths.corpus_manifest(ctx.corpus, None))
    assert manifest["set_aside_accepted"] == ["policy_0099"]
    assert manifest["data_health"]["documents_missing"] == 1
    assert result.data["data_health"]["documents_missing_not_accepted"] == 0


def test_a_labelled_document_never_ocrd_is_named_by_the_loader(client, controller):
    seeded = seeded_policies(seed_corpus(client))
    client.delete(paths.ocr_meta("policy", seeded[0]))
    ctx = make_context(client, controller)
    load_labeled_documents(ctx)
    assert ctx.documents_not_loaded == [(seeded[0], "labelled but never OCR'd: run stage 2")]
    # A second load (preflight, then the build) does not count it twice.
    load_labeled_documents(ctx)
    assert len(ctx.documents_not_loaded) == 1


def test_accept_set_aside_reads_a_comma_separated_list():
    from orchestration.run import accepted_set_aside

    assert accepted_set_aside(" policy_0001, ,policy_0002 ") == frozenset({"policy_0001", "policy_0002"})
    assert accepted_set_aside(None) == frozenset() and accepted_set_aside("") == frozenset()


# -- gold values no training row teaches -----------------------------------------


def test_values_no_window_teaches_are_counted_once_by_kind():
    from data_pipeline.data_health import untaught_values

    notes = {"policy_0001": [
        "window ocr_plus_image: unread coverages[3].limits[0].amount",
        "window image_only: unread coverages[3].limits[0].amount",          # the same value, another mode
        "window ocr_plus_image: unplaced forms[2].form_number",
        "window image_only: orphaned locations[1]",
        "window image_only: orphaned locations[2]",
        "window image_only: orphan taught nowhere locations[2].address",
        "window ocr_plus_image: away from home coverages[5]",
        "window ocr_plus_image: something else entirely",
    ]}
    assert untaught_values(notes) == {"policy_0001": {"away from home": 1, "orphan taught nowhere": 1,
                                                      "orphaned": 2, "unplaced": 1, "unread": 1}}

    built = _built(["policy_0001"], [])
    built.window_notes = notes
    health = assess([_document("policy_0001", ["a"])], built)
    summary = health.as_dict()["summary"]
    # Lost: the unread and unplaced values, and the one orphan value no other window teaches. Not lost: a row
    # taught on its home pages, and an orphaned piece as such (2026-11-09: every piece was counted as lost).
    assert summary["gold_values_never_taught"] == 3
    assert summary["gold_values_taught_elsewhere"] == 1 and summary["orphaned_row_pieces"] == 2
    assert summary["documents_with_values_never_taught"] == 1
    assert "gold values no training row teaches: 3 in 1 document(s)" in health.render()


# -- planning as if the eval set were not frozen ------------------------------------


def test_the_dataset_build_refuses_to_plan_as_unfrozen(client, controller):
    from orchestration import pipeline_dag

    seed_corpus(client)
    ctx = make_context(client, controller, plan_unfrozen=True)
    with pytest.raises(PipelineError, match="read-only data health report"):
        pipeline_dag.stage_dataset_build(ctx)


def test_planning_as_unfrozen_keeps_a_delivered_test_document_in_test(client, controller, monkeypatch):
    """Cycle 1's policy_0048 was delivered as test and missed the freeze: the report checks it where it belongs."""
    from evaluation import freeze_eval_set
    from orchestration import pipeline_dag

    seed_corpus(client)
    monkeypatch.setattr(freeze_eval_set, "is_frozen", lambda *a, **k: True)
    monkeypatch.setattr(pipeline_dag, "exclude_eval_families", lambda client, docs, tenant_id=None: (docs, []))
    monkeypatch.setattr(freeze_eval_set, "frozen_manifest", lambda *a, **k: {})
    assert pipeline_dag.plan_corpus(make_context(client, controller)).frozen is True
    assert pipeline_dag.plan_corpus(make_context(client, controller, plan_unfrozen=True)).frozen is False


# -- the Python a detached job runs with -------------------------------------------


def test_a_job_started_without_a_venv_runs_with_the_training_venv(monkeypatch, tmp_path):
    import sys

    from orchestration import detach

    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    monkeypatch.setattr(detach, "TRAINING_VENV", tmp_path / "venv")
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)                  # the system Python, no venv
    assert detach.detached_python() == str(venv_python)

    monkeypatch.setattr(sys, "prefix", str(tmp_path / "some-venv"))     # a venv is active: its own Python
    assert detach.detached_python() == sys.executable


def test_without_a_training_venv_the_caller_s_python_is_kept(monkeypatch, tmp_path):
    import sys

    from orchestration import detach

    monkeypatch.setattr(detach, "TRAINING_VENV", tmp_path / "missing")
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    assert detach.detached_python() == sys.executable


# -- near copies are found by page image -------------------------------------------


def test_near_copies_pair_by_page_image_when_their_ocr_text_differs(client):
    """0065 and 0066: 71 of 73 page images byte-identical, OCR text different on 18 of them, so by text they were
    never paired and the consistency test measured nothing (pod, 2026-11-07)."""
    from evaluation.metrics.near_copy import near_copy_pairs, page_image_keys

    a_keys, b_keys, a_text, b_text = [], [], {}, {}
    for page in range(1, 11):
        client.write_bytes(f"g/a/page_{page}.png", f"page {page}".encode())
        client.write_bytes(f"g/b/page_{page}.png", f"page {page}".encode() if page <= 9 else b"other")
        a_keys.append(f"g/a/page_{page}.png")
        b_keys.append(f"g/b/page_{page}.png")
        a_text[page] = f"Commercial General Liability coverage part page {page} " * 2
        # OCR read three identical pages differently.
        b_text[page] = a_text[page] if page > 3 else f"Commerc1al Genera1 Liabi1ity coverage part page {page} " * 2
    documents = [SimpleNamespace(source_id="policy_0065", image_keys=a_keys, page_texts=a_text),
                 SimpleNamespace(source_id="policy_0066", image_keys=b_keys, page_texts=b_text)]

    assert near_copy_pairs(documents) == []                                      # by text: 7 of 10, not paired
    [pair] = near_copy_pairs(documents, page_keys=page_image_keys(client, documents))
    assert (pair.a, pair.b, pair.share) == ("policy_0065", "policy_0066", 0.9)    # by image: 9 of 10
    assert page_image_keys(None, documents) is None
