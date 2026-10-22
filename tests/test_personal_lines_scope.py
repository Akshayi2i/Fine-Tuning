"""A scope narrowed by line of business: personal lines.

The same one corpus, filtered by line as well as type; a golden eval scored on the
scope's lines only; and a release that answers for its lines and refuses the rest,
rather than letting a model read a policy of a line it never trained on.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.scopes import ScopeError, _build, get_scope, lob_lines
from serving.release_router import UnservedDocType, build_serving_plan
from tests.test_release_router import promote
from training.corpus_view import materialize

PERSONAL = get_scope("personal_lines")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# --------------------------------------------------------------------------
# The scope
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lob,covered", [
    ("homeowners", True),
    # A package of common-model lines is read against _fallback.json, another
    # shape than the personal_lines adapter learns (common.scopes.common_model_package).
    (["homeowners", "personal_auto"], False),
    ("flood", False),                             # flood is not in this adapter
    (["homeowners", "gl"], False),                # personal + commercial is not personal
    ("workers_comp", False),
    ("wc", False),
    (None, False),                                # no line: cannot be shown to be in scope
    ([], False),
])
def test_personal_lines_covers_only_its_lines(lob, covered):
    assert PERSONAL.covers_lob(lob) is covered


def test_scopes_without_lines_cover_every_line():
    assert get_scope("policy").covers_lob("wc")
    assert get_scope("policy").covers_lob(None)


def test_enum_and_schema_spellings_are_one_line():
    assert lob_lines("workers_comp") == lob_lines("wc") == frozenset({"wc"})


def test_a_line_scope_must_be_policy_only():
    with pytest.raises(ScopeError, match="line is a policy"):
        _build("mixed", {"doc_types": ["policy", "acord"], "tasks": ["extract"],
                         "lines": ["homeowners"]})


def test_a_line_with_no_schema_is_refused():
    with pytest.raises(ScopeError, match="no canonical policy schema"):
        _build("pets", {"doc_types": ["policy"], "tasks": ["extract"], "lines": ["pet_insurance"]})


def test_the_line_restriction_is_recorded():
    assert "homeowners" in PERSONAL.as_dict()["lines"]


# --------------------------------------------------------------------------
# One corpus, filtered by line
# --------------------------------------------------------------------------


def _seed(client, rows_by_epoch, val):
    for epoch in (1, 2, 3, 4):
        client.write_text(paths.corpus_epoch_file("v1", epoch),
                          "".join(json.dumps(r) + "\n" for r in rows_by_epoch))
    client.write_text(paths.corpus_eval_split("v1", "val"), "".join(json.dumps(r) + "\n" for r in val))


def test_the_corpus_view_keeps_only_personal_lines(client):
    rows = [
        {"source_id": "h1", "doc_type": "policy", "lob": "homeowners"},
        {"source_id": "a1", "doc_type": "policy", "lob": ["personal_auto"]},
        {"source_id": "g1", "doc_type": "policy", "lob": "gl"},
        {"source_id": "m1", "doc_type": "policy", "lob": ["homeowners", "gl"]},
        {"source_id": "n1", "doc_type": "policy"},
        {"source_id": "f1", "doc_type": "policy", "lob": "flood"},
        {"source_id": "l1", "doc_type": "lossrun"},
    ]
    _seed(client, rows, [{"source_id": "v1", "doc_type": "policy", "lob": "ocean_marine"},
                         {"source_id": "v2", "doc_type": "policy", "lob": "wc"}])
    view = materialize(PERSONAL, "v1", client)
    kept = {json.loads(line)["source_id"]
            for line in client.read_text(view.epoch_files[0]).splitlines() if line.strip()}
    assert kept == {"h1", "a1"}
    assert view.val_rows == 1


# --------------------------------------------------------------------------
# The golden eval: the scope's lines only
# --------------------------------------------------------------------------


def test_the_golden_eval_scores_only_the_scopes_lines(client, monkeypatch):
    from evaluation import golden_eval

    seen = []

    def fake_evaluate(documents, *a, **k):
        seen.extend(d.source_id for d in documents)
        return []

    for source_id, lob in (("h1", "homeowners"), ("g1", "gl"), ("n1", None)):
        base = f"{paths.golden_eval_set_dir()}/{source_id}"
        client.write_json(f"{base}/metadata.json", {"doc_type": "policy", "lob": lob, "reviewer_id": "a.reviewer",
                                                    "review_date": "2026-10-20"})
        client.write_bytes(f"{base}/page_1.png", b"png")
        client.write_json(f"{base}/golden.json", {})
    client.write_json(f"{paths.golden_eval_set_dir()}/manifest.json", {"source_ids": ["h1", "g1", "n1"]})
    monkeypatch.setattr(golden_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr("evaluation.run_eval.assert_eval_set_disjoint", lambda *a, **k: None)
    golden_eval.evaluate_version(client, object(), version="v2", corpus_version="v1", scope=PERSONAL)
    assert seen == ["h1"]


# --------------------------------------------------------------------------
# Serving: a personal-lines release answers for its lines, and refuses the rest
# --------------------------------------------------------------------------


def _promote_personal(client, release_id="release-2026.10.2", **kw):
    promote(client, release_id, scope="personal_lines", doc_types=["policy"], **kw)
    bundle = client.read_json(paths.release_bundle(release_id))
    bundle["lines"] = sorted(PERSONAL.lines)
    client.write_json(paths.release_bundle(release_id), bundle)


def test_a_personal_policy_goes_to_the_personal_release(client):
    promote(client, "release-2026.10.1")                  # unified, every line
    _promote_personal(client)
    plan = build_serving_plan(client)
    assert plan.release_for("policy", "homeowners").release_id == "release-2026.10.2"
    assert plan.release_for("policy", "gl").release_id == "release-2026.10.1"
    assert plan.release_for("policy").release_id == "release-2026.10.1"


def test_alone_a_personal_release_refuses_other_lines_and_unknown_lines(client):
    _promote_personal(client)
    plan = build_serving_plan(client)
    assert "policy" in plan.served_doc_types
    assert plan.release_for("policy", "personal_auto").scope == "personal_lines"
    with pytest.raises(UnservedDocType):
        plan.release_for("policy", ["personal_auto", "motorcycle"])    # a package: not its shape
    with pytest.raises(UnservedDocType, match="cover only"):
        plan.release_for("policy", "wc")
    with pytest.raises(UnservedDocType, match="cover only"):
        plan.release_for("policy", None)


def test_a_per_type_pin_cannot_route_every_policy_to_a_line_release(client):
    from serving.release_router import ServingPlanError

    _promote_personal(client)
    with pytest.raises(ServingPlanError, match="serves only lines"):
        build_serving_plan(client, pins={"policy": "release-2026.10.2"})


def test_the_release_bundle_records_the_scopes_lines():
    from registry_utils.models import ReleaseBundle

    fields = ReleaseBundle.model_fields
    assert "lines" in fields
    assert fields["lines"].default_factory() == []
