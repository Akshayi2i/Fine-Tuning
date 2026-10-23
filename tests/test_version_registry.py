"""One version names a cycle; the CLI takes it, resumes it, and changes which is live.

Decision 2026-10-22: every fine-tuning cycle trains the one policy adapter and
takes the next version number, which names its run (``policy-v3``), its adapter
and its release (since 2026-10-23 it continues the promoted adapter unless told
``--fresh``). A fresh ``finetune`` / ``all``
takes that number itself; a resume names it, because the next free number after
a first attempt that wrote anything is a new cycle's. ``promote`` puts an older
release back (a rollback) and ``versions`` shows the registry.

The release bookkeeping itself (promote, archive, the index order) is
``registry_utils.releases``' and is tested with it; this file holds the command
surface and the cycle's number.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import run as cli
from orchestration.pipeline_dag import RunReport, run_stages, stages_for
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import _calibration_samples, make_context, seed_corpus


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def _args(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def _registry(client: BlobClient, *run_ids: str) -> None:
    """The run registry's index (shared by every tenant), holding these runs."""
    client.write_json(paths.registry_index(), {"runs": [
        {"run_id": run_id, "run_type": "scoped", "scope": run_id.rsplit("-", 1)[0],
         "status": "trained"}
        for run_id in run_ids
    ]})


def _released(client: BlobClient, controller: RunPodController, *versions: str,
              calibrated: bool = True) -> None:
    """Whole dry-run cycles, packaged, then promoted as a real `package` would
    (a dry run itself never promotes). Calibrated, each release is promoted and
    archives the one before it; uncalibrated, it stays gated. Each trains from
    the base (``--fresh``): continuing the one before would hold it to that
    release's per-line scores, which these cycles have none of."""
    from registry_utils.releases import promote_release

    seed_corpus(client)
    for version in versions:
        samples = {"bf16": _calibration_samples()} if calibrated else {}
        ctx = make_context(client, controller, out_version=version, calibration_samples=samples,
                           fresh=True)
        report = run_stages(ctx, stages_for("all"), command="all")
        assert report.ok, report.render()
        if calibrated:
            promote_release(client, version, promoted_by=f"package {version}")


def _statuses(client: BlobClient, tenant_id: str | None = None) -> dict[str, str]:
    return {r["release_id"]: r["status"] for r in client.read_json(paths.release_index(tenant_id))}


# --------------------------------------------------------------------------
# cycle_version: a fresh cycle takes the next number, a resume names its own
# --------------------------------------------------------------------------


def test_a_fresh_cycle_on_an_empty_registry_is_v1(client):
    assert cli.cycle_version(_args("all", "--input", "./intake"), client) == "v1"
    assert cli.cycle_version(_args("finetune", "--input", "./intake"), client) == "v1"


def test_a_fresh_cycle_takes_one_more_than_the_newest_policy_run(client):
    # Runs of another lineage (the unified extractor's) are not this adapter's.
    _registry(client, "policy-v1", "policy-v3", "extractor-v9")
    assert cli.cycle_version(_args("all", "--input", "./intake"), client) == "v4"


def test_every_tenant_s_corpora_and_releases_take_their_numbers_too(client):
    """A cycle that failed before training, or is still building its corpus, left
    one; reusing its number would train on it. The numbers are one sequence for
    every tenant, because the run and its weights are not kept per tenant."""
    _registry(client, "policy-v3")
    client.write_json(paths.corpus_manifest("v5", "acme"), {})
    assert cli.cycle_version(_args("all", "--tenant", "acme"), client) == "v6"
    client.write_json(paths.release_bundle("v7", "acme"), {})
    assert cli.cycle_version(_args("all", "--tenant", "acme"), client) == "v8"
    assert cli.cycle_version(_args("all"), client) == "v8"


def test_a_named_version_wins(client):
    _registry(client, "policy-v3")
    assert cli.cycle_version(_args("all", "--version", "v2"), client) == "v2"
    assert cli.cycle_version(_args("finetune", "--version", "v3", "--from-stage", "calibrate"),
                             client) == "v3"
    assert cli.cycle_version(_args("package", "--version", "v3"), client) == "v3"


def test_a_resume_without_its_version_is_refused(client):
    args = _args("all", "--from-stage", "calibrate")
    problem = cli.flag_problem(args)
    assert problem and "name its version with --version" in problem
    with pytest.raises(cli.VersionError, match="--from-stage"):
        cli.cycle_version(args, client)
    assert cli.flag_problem(_args("all", "--from-stage", "calibrate", "--version", "v3")) is None


def test_main_refuses_a_resume_without_its_version_before_any_work(monkeypatch, capsys):
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: pytest.fail("reached Blob"))
    with pytest.raises(SystemExit) as exited:
        cli.main(["finetune", "--from-stage", "checkpoint_eval"])
    assert exited.value.code == 2 and "--version" in capsys.readouterr().err


def test_main_runs_a_fresh_cycle_under_the_next_version(client, monkeypatch):
    """The version is decided once, by main, and is what every stage, the run
    and the release are named by."""
    import orchestration.detach as detach

    _registry(client, "policy-v3")
    monkeypatch.setattr(detach, "detach_module_if_needed", lambda *a, **k: False)
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: client)
    monkeypatch.setattr(cli, "for_ingestion", lambda *a, **k: client)
    monkeypatch.setattr(cli, "RunPodController", lambda **k: RunPodController(
        backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234"))
    ran: list[tuple] = []

    def run_command(command, ctx, *, from_stage=None):
        ran.append((command, ctx.out_version, ctx.release_id, ctx.corpus, from_stage))
        return RunReport(command=command, version=ctx.out_version)

    monkeypatch.setattr(cli, "run_command", run_command)
    assert cli.main(["all", "--input", "./intake", "--commit", "abc1234"]) == 0
    assert ran == [("all", "v4", "v4", "v4", None)]

    # A resume keeps the number it names, even with later ones taken.
    assert cli.main(["finetune", "--version", "v2", "--from-stage", "checkpoint_eval",
                     "--commit", "abc1234"]) == 0
    assert ran[-1] == ("finetune", "v2", "v2", "v2", "checkpoint_eval")


# --------------------------------------------------------------------------
# promote: a rollback, named by a person
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["system", "CI", " "])
def test_promote_must_name_a_person(name, monkeypatch, capsys):
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: pytest.fail("reached Blob"))
    with pytest.raises(SystemExit) as exited:
        cli.main(["promote", "--version", "v2", "--by", name])
    assert exited.value.code == 2
    assert "must name the person accountable" in capsys.readouterr().err


def test_promote_needs_a_version_and_a_name():
    for argv in (["promote", "--by", "A. Reviewer"], ["promote", "--version", "v2"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(argv)
    assert {"promote", "versions"} <= cli.QUICK_COMMANDS          # never detached on the pod


def test_promote_rolls_back_to_an_older_release(client, controller, monkeypatch, capsys):
    """v2 replaced v1 at package; promoting v1 by name puts it back and
    archives v2. Both run manifests follow, so the registry shows v1 is live."""
    from registry_utils.query_registry import get

    _released(client, controller, "v1", "v2")
    assert _statuses(client) == {"v1": "archived", "v2": "promoted"}
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: client)

    assert cli.main(["promote", "--version", "v1", "--by", "A. Reviewer"]) == 0
    assert "v1 is promoted; archived v2" in capsys.readouterr().out
    assert _statuses(client) == {"v1": "promoted", "v2": "archived"}
    assert client.read_json(paths.release_bundle("v1"))["status"] == "promoted"
    assert client.read_json(paths.release_bundle("v2"))["status"] == "archived"
    rolled_back = get("policy-v1", client)
    assert rolled_back.status == "promoted" and rolled_back.promotion.promoted_by == "A. Reviewer"
    assert get("policy-v2", client).status == "archived"


def test_promote_refuses_a_release_that_cannot_be_served_as_promoted(client, controller,
                                                                     monkeypatch, capsys):
    """An uncalibrated release passed its gate and is gated, not promoted: its
    confidence is not calibrated. Promoting it by hand is refused, as is a
    release that does not exist, and nothing changes."""
    _released(client, controller, "v1", calibrated=False)
    assert _statuses(client) == {"v1": "gated"}
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: client)

    assert cli.main(["promote", "--version", "v1", "--by", "A. Reviewer"]) == 1
    assert "v1 cannot be promoted" in capsys.readouterr().err
    assert cli.main(["promote", "--version", "v9", "--by", "A. Reviewer"]) == 1
    assert "there is no release v9" in capsys.readouterr().err
    assert _statuses(client) == {"v1": "gated"}


# --------------------------------------------------------------------------
# versions: the registry at a glance
# --------------------------------------------------------------------------


def test_versions_prints_the_registry_as_a_json_table(client, controller, monkeypatch, capsys):
    _released(client, controller, "v1", "v2")
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: client)

    assert cli.main(["versions"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert [row["version"] for row in table] == ["v1", "v2"]
    by_version = {row["version"]: row for row in table}
    assert (by_version["v1"]["run_status"], by_version["v1"]["release_status"]) == ("archived", "archived")
    assert (by_version["v2"]["run_status"], by_version["v2"]["release_status"]) == ("promoted", "promoted")

    # Another tenant has released nothing; the shared registry still lists the runs.
    assert cli.main(["versions", "--tenant", "acme"]) == 0
    other = json.loads(capsys.readouterr().out)
    assert [(row["version"], row["release_status"]) for row in other] == [("v1", None), ("v2", None)]
