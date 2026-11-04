"""Test scores reported apart for templates the model trained on and templates it never saw (user, 2026-11-04)."""

from __future__ import annotations

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.freeze_eval_set import (
    FreezeError,
    _train_page_keys,
    manifest_key,
    mark_templates,
    template_seen,
)
from evaluation.run_eval import build_report, subset_of


def _page(word):
    return f"{word} " + " ".join(f"line{i}" for i in range(20))


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _seed_train(client, source_id, pages):
    for number, text in enumerate(pages, 1):
        client.write_text(paths.processed_page("policy", source_id, number, "md"), text)


def _manifest(**over):
    return {"doc_types": ["policy"], "source_ids_by_split": {"train": ["policy_0001"], "test": ["policy_0002"]}, **over}


def test_training_pages_are_read_from_the_training_documents_only(client):
    _seed_train(client, "policy_0001", [_page("a"), _page("b")])
    _seed_train(client, "policy_0002", [_page("c")])
    keys = _train_page_keys(client, _manifest(), None)
    assert len(keys) == 2
    assert template_seen({1: _page("c")}, keys)["template_seen_in_train"] is False


def test_a_manifest_naming_no_training_set_cannot_tell(client):
    assert _train_page_keys(client, {"doc_types": ["policy"]}, None) is None
    assert template_seen({1: _page("a")}, None) == {}


@pytest.mark.parametrize("pages,seen,share", [
    ([_page("a"), _page("b")], True, 1.0),
    ([_page("a"), _page("x")], True, 0.5),
    ([_page("a"), _page("x"), _page("y")], False, 0.3333),
    ([_page("x"), _page("y")], False, 0.0),
])
def test_a_template_is_seen_when_half_its_pages_are_on_training_documents(client, pages, seen, share):
    _seed_train(client, "policy_0001", [_page("a"), _page("b")])
    mark = template_seen(dict(enumerate(pages)), _train_page_keys(client, _manifest(), None))
    assert mark == {"template_seen_in_train": seen, "pages_shared_with_train": share}


def test_a_document_with_no_page_text_cannot_be_told(client):
    _seed_train(client, "policy_0001", [_page("a")])
    assert template_seen({1: ""}, _train_page_keys(client, _manifest(), None)) == {}


def test_an_existing_frozen_set_is_marked_and_nothing_else_changes(client):
    _seed_train(client, "policy_0001", [_page("a"), _page("b")])
    client.write_json(paths.corpus_manifest("v3"), _manifest())
    root = paths.golden_eval_set_dir(None)
    client.write_json(manifest_key(), {"frozen_from_corpus": "v3", "source_ids": ["policy_0002", "policy_0003"]})
    for source_id, text in (("policy_0002", _page("a")), ("policy_0003", _page("z"))):
        client.write_json(f"{root}/{source_id}/metadata.json", {"doc_type": "policy", "reviewer_id": "x"})
        client.write_json(f"{root}/{source_id}/golden.json", {"gold": source_id})
        client.write_text(f"{root}/{source_id}/page_1.md", text)

    marks = mark_templates(client)

    assert marks["policy_0002"]["template_seen_in_train"] is True
    assert marks["policy_0003"]["template_seen_in_train"] is False
    metadata = client.read_json(f"{root}/policy_0002/metadata.json")
    assert metadata["reviewer_id"] == "x" and metadata["template_seen_in_train"] is True
    assert client.read_json(f"{root}/policy_0002/golden.json") == {"gold": "policy_0002"}


def test_marking_with_no_frozen_set_or_no_training_set_is_refused(client):
    with pytest.raises(FreezeError, match="no frozen eval set"):
        mark_templates(client)
    client.write_json(manifest_key(), {"frozen_from_corpus": "v3", "source_ids": []})
    client.write_json(paths.corpus_manifest("v3"), {"doc_types": ["policy"]})
    with pytest.raises(FreezeError, match="names no training documents"):
        mark_templates(client)


def test_the_subsets_follow_the_mark_and_a_document_without_one_is_in_neither():
    assert "template_seen" in subset_of({"template_seen_in_train": True})
    assert "template_unseen" in subset_of({"template_seen_in_train": False})
    unmarked = subset_of({"template_seen_in_train": None})
    assert "template_seen" not in unmarked and "template_unseen" not in unmarked


def test_the_report_carries_both_scores_apart():
    def meta(source, seen):
        return {"source_id": source, "doc_type": "policy", "lob": "gl", "modality_mode": "image_only",
                "template_seen_in_train": seen}

    def answer(number):
        return {"policy": {"policy_number": {"raw": number, "parsed": number, "page_ref": [1]}}}

    right, wrong = answer("POL-123456"), answer("POL-999999")
    report = build_report("v1", [(right, right, meta("a", True)), (right, wrong, meta("b", False))])
    metrics = report.gate_metrics()
    assert metrics["template_seen_match"] == 1.0 and metrics["template_unseen_match"] == 0.0


def test_neither_score_is_a_gate():
    from evaluation.gating import is_gate

    assert not is_gate("template_seen_match") and not is_gate("template_unseen_match")
