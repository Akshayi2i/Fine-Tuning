"""`shadow --promote-by`: promote in a named person's name, only on a clean pass.

``package`` still never promotes. The one step that needs the served model and OCR is the shadow check, so
the person is named when it is launched; a passing check then promotes with nothing waived. An interim tier,
a missing or failed shadow report, a system identity and a gate passed by written override are all refused
and left for ``promote`` by hand.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import run as cli
from registry_utils.releases import ReleaseError, promote_if_cleared
from tests.test_decision_promotion import _gated, _passing_report, _statuses


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _override(client: BlobClient, version: str) -> None:
    key = paths.release_gate_decision(version, "bf16")
    decision = client.read_json(key)
    client.write_json(key, {**decision, "override": {
        "approver": "A. Reviewer", "reason": "ships although the gate is not passed", "waive_all": True}})


def test_a_clean_pass_is_promoted_in_the_named_persons_name(client):
    _gated(client, "v3", tier="production")
    key = _passing_report(client, "v3", compared_with=None)

    assert promote_if_cleared(client, "v3", promoted_by="B. Operator") == []
    assert _statuses(client) == {"v3": "promoted"}
    [record] = client.read_json(paths.release_bundle("v3"))["promotions"]
    assert record["promoted_by"] == "B. Operator" and record["shadow_report"] == key
    assert record["shadow_skipped_reason"] is None and record["accepted_interim"] is False


def test_without_a_passing_shadow_report_nothing_is_promoted(client):
    _gated(client, "v3", tier="production")
    with pytest.raises(ReleaseError, match="there is no shadow report of v3"):
        promote_if_cleared(client, "v3", promoted_by="B. Operator")
    assert _statuses(client) == {"v3": "gated"}


def test_an_interim_tier_is_left_for_a_person_to_accept(client):
    _gated(client, "v3", tier="interim", tier_confidence=0.4)
    _passing_report(client, "v3", compared_with=None)
    with pytest.raises(ReleaseError, match="--accept-interim"):
        promote_if_cleared(client, "v3", promoted_by="B. Operator")
    assert _statuses(client) == {"v3": "gated"}


def test_a_gate_passed_by_written_override_is_not_promoted_automatically(client):
    _gated(client, "v3", tier="production")
    _passing_report(client, "v3", compared_with=None)
    _override(client, "v3")
    with pytest.raises(ReleaseError, match="written override.*by hand"):
        promote_if_cleared(client, "v3", promoted_by="B. Operator")
    assert _statuses(client) == {"v3": "gated"}


def test_a_system_identity_is_refused(client):
    _gated(client, "v3", tier="production")
    _passing_report(client, "v3", compared_with=None)
    with pytest.raises(ReleaseError, match="person accountable"):
        promote_if_cleared(client, "v3", promoted_by="system")
    assert _statuses(client) == {"v3": "gated"}


# --------------------------------------------------------------------------
# The shadow command
# --------------------------------------------------------------------------


def _shadow_args(**over) -> SimpleNamespace:
    args = dict(version="v3", model_url="https://pod", mineru_url="https://mineru", documents=None,
                tenant=None, promote_by=None)
    args.update(over)
    return SimpleNamespace(**args)


@pytest.fixture
def shadow_passes(monkeypatch):
    import orchestration.shadow as shadow

    monkeypatch.setattr(shadow, "run_shadow", lambda *_a, **_k: {"passed": True, "candidate": {}})
    monkeypatch.setattr(shadow, "render", lambda _report: "")


def test_the_shadow_command_promotes_when_asked_and_cleared(client, shadow_passes, capsys):
    _gated(client, "v3", tier="production")
    _passing_report(client, "v3", compared_with=None)

    assert cli.run_shadow_command(_shadow_args(promote_by="B. Operator"), client=client) == 0
    assert _statuses(client) == {"v3": "promoted"}
    assert "v3 is promoted by B. Operator" in capsys.readouterr().out


def test_the_shadow_command_leaves_promotion_to_a_person_without_the_flag(client, shadow_passes, capsys):
    _gated(client, "v3", tier="production")
    _passing_report(client, "v3", compared_with=None)

    assert cli.run_shadow_command(_shadow_args(), client=client) == 0
    assert _statuses(client) == {"v3": "gated"}
    assert "next: python -m orchestration.run promote --version v3" in capsys.readouterr().out


def test_the_shadow_command_reports_a_refusal_and_exits_non_zero(client, shadow_passes, capsys):
    _gated(client, "v3", tier="interim", tier_confidence=0.4)
    _passing_report(client, "v3", compared_with=None)

    assert cli.run_shadow_command(_shadow_args(promote_by="B. Operator"), client=client) == 1
    assert _statuses(client) == {"v3": "gated"}
    assert "not promoted automatically" in capsys.readouterr().err


def test_a_failed_shadow_check_promotes_nothing_even_when_asked(client, monkeypatch):
    import orchestration.shadow as shadow

    monkeypatch.setattr(shadow, "run_shadow", lambda *_a, **_k: {"passed": False, "candidate": {}})
    monkeypatch.setattr(shadow, "render", lambda _report: "")
    _gated(client, "v3", tier="production")

    assert cli.run_shadow_command(_shadow_args(promote_by="B. Operator"), client=client) == 1
    assert _statuses(client) == {"v3": "gated"}
