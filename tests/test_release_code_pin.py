"""A release names the code it serves with (review S6, 23 November 2026).

A rollback needs the right adapter AND the right code: the code renders the
prompt and cuts the windows the adapter was trained on. The bundle records
the commit it was packaged from; promotion prints the pair; a refusal names
the code that would serve it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from tests.test_registry import _bundle


def test_old_bundles_without_a_commit_still_load() -> None:
    bundle = _bundle()
    assert bundle.code_git_commit is None and bundle.image_digest is None
    assert _bundle(code_git_commit="5c93943").code_git_commit == "5c93943"


def test_a_refusal_names_the_code_that_serves_the_release() -> None:
    from registry_utils.releases import serving_refusal

    body = json.loads(_bundle(code_git_commit="5c93943abc", prompt_hash="sha256:old").model_dump_json())
    refusal = serving_refusal(body)
    assert refusal is not None and "prompt hash" in refusal
    assert "packaged from code commit 5c93943abc: serve it from that code" in refusal

    older = json.loads(_bundle(prompt_hash="sha256:old").model_dump_json())
    assert "records no code commit" in serving_refusal(older)


def test_the_serving_pair_names_adapter_and_code() -> None:
    from registry_utils.releases import serving_pair

    body = json.loads(_bundle(code_git_commit="abc123").model_dump_json())
    pair = serving_pair(body)
    assert pair.startswith("v3 = adapter ") and pair.endswith("+ code commit abc123")
    unrecorded = serving_pair(json.loads(_bundle().model_dump_json()))
    assert "code commit not recorded" in unrecorded


def test_package_records_the_run_commit_and_never_a_guess(monkeypatch) -> None:
    from orchestration import pipeline_dag as dag
    from registry_utils import write_run_manifest as W

    assert dag._packaged_commit(SimpleNamespace(git_commit="feedbeef")) == "feedbeef"
    monkeypatch.setattr(W, "capture_git_commit", lambda short=True: "unknown")
    assert dag._packaged_commit(SimpleNamespace(git_commit=None)) is None
    monkeypatch.setattr(W, "capture_git_commit", lambda short=True: "0123abcd")
    assert dag._packaged_commit(SimpleNamespace(git_commit=None)) == "0123abcd"
