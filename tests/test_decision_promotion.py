"""Decision 2026-10-26 (#44): package stops at gated; a shadow check on the server path; promote asks for it.

* ``package`` writes a release that passed its gate ``gated``, archives nothing,
  and its run summary prints the next steps: the shadow check, then promote.
* ``shadow`` reads frozen eval documents through the Azure worker's two phases
  (``serving.fields_first``) with the candidate's LoRA and the promoted one's,
  scores both with the gate's scoring and judges them by the pass rule.
* ``promote`` promotes a gated release only on its passing shadow report, or
  with a written reason to skip it; an interim tier needs ``--accept-interim``;
  a rollback needs no shadow; every promotion is recorded on the bundle.

Synthetic only: an in-memory Blob, the echo model as the pod's vLLM and a fake
served MinerU - no network, no GPU.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import run as cli
from orchestration import shadow as S
from registry_utils.releases import (
    ReleaseError,
    promote,
    promote_release,
    shadow_report_key,
    write_index,
)

ROOT = Path(__file__).resolve().parent.parent

PAGES = ["Personal Auto Policy Declarations\nPolicy Number PA-1\nNamed Insured Jane Roe\nCarrier Acme Mutual",
         "PART A - LIABILITY COVERAGE\nWe will pay damages for bodily injury."]


def _env(value: str) -> dict:
    return {"raw": value, "parsed": value, "page_ref": [1]}


GOOD = {"document": {"title": _env("Personal Auto Policy Declarations")}, "carrier": {"name": _env("Acme Mutual")},
        "named_insured": {"primary_name": _env("Jane Roe")}, "policy": {"policy_number": _env("PA-1")}}
#: A policy number no page prints: taken out as served (field match drops) and counted as invented.
WRONG = {**copy.deepcopy(GOOD), "policy": {"policy_number": _env("PA-9")}}
#: No carrier, no named insured: serving's schema verdict is invalid.
PARTIAL = {"policy": {"policy_number": _env("PA-1")}}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw")


@pytest.fixture
def raw(backend) -> BlobClient:
    """The OCR stage's login: it may read raw-documents/."""
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ocr")


def _gated(client: BlobClient, version: str, *, tier: str | None = "production", tier_confidence=None,
           created_at: str = "2026-10-26T00:00:00+00:00", status: str = "gated", passed: bool = True,
           **over) -> dict:
    """A release as `package` writes it since 2026-10-26: a LoRA on the base, gated, its gate passed."""
    from common.config import served_mineru
    from common.prompts import prompt_hash
    from data_pipeline.dataset_builder.policy_windows import window_plan_id
    from registry_utils.models import ReleaseBundle
    from registry_utils.query_registry import configured_base_model

    bundle = ReleaseBundle(**{
        "release_id": version, "status": status, "tier": tier, "tenant_scope": paths.default_tenant(),
        "scope": "policy", "doc_types": ["policy"], "lines": ["personal_auto"], "created_at": created_at,
        "base_model": configured_base_model(), "adapter": f"policy-{version}", "served_as": "lora_on_base",
        "serving_formats": {"bf16": paths.scoped_adapter_dir("policy", version)},
        "calibrators": {"bf16": paths.release_calibrators(version, "bf16")},
        "gate_reports": {"bf16": paths.release_gate_decision(version, "bf16")},
        "prompt_hash": prompt_hash(), "window_plan_version": window_plan_id(),
        "ocr_pin": {"mineru_version": served_mineru()["version"]},
        "vision_config_hash": "sha256:b", "vllm_config_hash": "sha256:c", "lockfile_hash": "sha256:d",
        **over,
    })
    body = json.loads(bundle.model_dump_json())
    client.write_json(paths.release_bundle(version), body)
    index_key = paths.release_index()
    rows = client.read_json(index_key) if client.exists(index_key) else []
    write_index(client, [r for r in rows if r["release_id"] != version] + [bundle.index_row()], None)
    client.write_json(paths.release_gate_decision(version, "bf16"), {
        "passed": passed, "failed_gates": [] if passed else ["field_normalized_match"], "tier": tier,
        **({"tier_confidence": tier_confidence} if tier_confidence is not None else {})})
    return body


def _statuses(client: BlobClient) -> dict[str, str]:
    return {r["release_id"]: r["status"] for r in client.read_json(paths.release_index())}


def _passing_report(client: BlobClient, version: str, *, compared_with: str | None, **over) -> str:
    """A shadow report as `shadow` writes one that passed, of the bundle as packaged now."""
    body = client.read_json(paths.release_bundle(version))
    key = shadow_report_key(version)
    client.write_json(key, {"release_id": version, "bundle_created_at": body["created_at"],
                            "promoted_release_id": compared_with, "passed": True, "reasons": [], **over})
    return key


class FakeReader:
    """The served MinerU as fields first uses it: pages rendered without text, then text alone."""

    def __init__(self, *, fail: dict[str, list[BaseException]] | None = None) -> None:
        self.fail = {name: list(errors) for name, errors in (fail or {}).items()}
        self.read_names: list[str] = []

    def read(self, pdf, pages_dir, *, name, with_text, publish_pages=None):
        assert not with_text and pdf.startswith(b"%PDF")
        self.read_names.append(name)
        if self.fail.get(name):
            raise self.fail[name].pop(0)
        pages_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for number in range(1, len(PAGES) + 1):
            (pages_dir / f"page_{number:03d}.png").write_bytes(b"png")
            images.append(str(pages_dir / f"page_{number:03d}.png"))
        return {"images": images, "texts": {}, "ocr": {}, "links": None, "timings_s": {"render": 0.1, "links": None}}

    def ocr_text(self, pdf, *, name, page_count=None, gate=None, pages=None):
        numbers = pages or list(range(1, len(PAGES) + 1))
        texts = {number: PAGES[number - 1] for number in numbers}
        return {"pages": list(texts.values()), "texts": texts,
                "info": {"version": "4.0.12", "tier": "basic", "ocr_mode": "ocr"}, "seconds": 0.0}


class Models:
    """The pod's vLLM by LoRA name, answered by the echo model; a name it does not serve is refused."""

    def __init__(self, answers: dict[str, dict], *, missing: tuple[str, ...] = ()) -> None:
        self.answers, self.missing, self.asked = answers, missing, []

    def __call__(self, name: str):
        from azure_service.runtime import served_model
        from inference_core.model_runner import EchoBackend, ModelRunnerError

        self.asked.append(name)
        if name in self.missing:
            raise ModelRunnerError(f"https://pod does not serve {name!r} (it serves ['policy-v3'])")
        return served_model(EchoBackend(json.dumps(self.answers[name])), name)


def _documents(raw_writer: BlobClient, count: int = 4, *, synthetic: tuple[int, ...] = ()) -> list:
    from evaluation.golden_eval import GoldenDocument

    docs = []
    for number in range(1, count + 1):
        source_id = f"policy_{number:04d}"
        raw_writer.write_bytes(paths.raw_pdf("policy", source_id), b"%PDF-1.4 synthetic")
        docs.append(GoldenDocument(source_id=source_id, doc_type="policy", golden=copy.deepcopy(GOOD),
                                   image_keys=["p1", "p2"], page_texts={1: PAGES[0], 2: PAGES[1]},
                                   lob="personal_auto", synthetic=number in synthetic))
    return docs


@pytest.fixture
def documents(backend) -> list:
    return _documents(BlobClient(backend=backend, container="main", raw_container="raw", context="ingestion"))


def _shadow(client, raw, documents, tmp_path, models, *, reader=None, version="v3", count=4):
    return S.run_shadow(client, version, model_url="https://pod-8000.proxy.runpod.net",
                        mineru_url="https://pod-8001.proxy.runpod.net", documents=count, raw_client=raw,
                        models=models, reader=reader or FakeReader(), golden=documents, work_dir=tmp_path / "work")


# --------------------------------------------------------------------------
# package: gated, never promoted, and the next steps
# --------------------------------------------------------------------------


def test_package_leaves_the_release_gated_and_prints_shadow_then_promote(client):
    """The package stage's next steps: the shadow command (both LoRAs served,
    --max-loras 2, when one is promoted), then promote - with --accept-interim
    for an interim tier. The run summary prints them."""
    from orchestration.pipeline_dag import RunReport, StageResult, release_next_steps

    _gated(client, "v2")
    promote_release(client, "v2", promoted_by="A. Reviewer")
    bundle = SimpleNamespace(release_id="v3", tier="interim")
    ctx = SimpleNamespace(client=client, tenant_id=None, dry_run=False)
    steps = release_next_steps(ctx, bundle, [])
    assert steps[0].startswith("1. shadow:  python -m orchestration.run shadow --version v3 --model-url")
    assert "policy-v3 and policy-v2" in steps[0] and "--max-loras 2" in steps[0]
    assert steps[1] == '2. promote: python -m orchestration.run promote --version v3 --by "<person>" --accept-interim'

    report = RunReport("package", "v3", results=[StageResult("package", "completed", "release v3 gated",
                                                             {"next_steps": steps})])
    rendered = report.render()
    assert "next steps:" in rendered
    assert rendered.index("orchestration.run shadow") < rendered.index("orchestration.run promote")

    # A release promote would refuse says what to fix; a dry run has nothing to read.
    assert "cannot be promoted yet (bf16 has no calibrator" in release_next_steps(
        ctx, bundle, ["bf16 has no calibrator"])[0]
    assert release_next_steps(SimpleNamespace(client=client, tenant_id=None, dry_run=True), bundle, [])[0] \
        .startswith("dry run:")


def test_package_s_stage_result_carries_the_next_steps(client):
    """End to end through `all` (a dry run): the release is gated, nothing is promoted,
    and the package stage's data holds the steps the summary prints."""
    from orchestration.pipeline_dag import run_stages, stages_for
    from orchestration.runpod_controller import LocalBackend, RunPodController
    from tests.test_orchestration import make_context, seed_corpus

    controller = RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()
    assert _statuses(client) == {"v1": "gated"}
    assert ctx.results["package"].data["next_steps"]
    assert "next steps:" in report.render()


def test_only_the_promote_command_promotes():
    """Nothing but registry_utils.releases calls promote_release: package no
    longer does, so no code path promotes without the shadow check's clearance."""
    callers = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", ".")) or "/." in rel or rel == "registry_utils/releases.py":
            continue
        if "promote_release(" in path.read_text(encoding="utf-8", errors="replace"):
            callers.append(rel)
    assert callers == []


# --------------------------------------------------------------------------
# promote: the shadow report, the reason to skip it, the tier, rollbacks
# --------------------------------------------------------------------------


def test_a_gated_release_without_a_passing_shadow_report_is_not_promoted(client):
    _gated(client, "v3")
    with pytest.raises(ReleaseError, match="there is no shadow report of v3") as refused:
        promote(client, "v3", promoted_by="A. Reviewer")
    assert "orchestration.run shadow --version v3" in str(refused.value)
    assert "--skip-shadow --reason" in str(refused.value)

    _passing_report(client, "v3", compared_with=None, passed=False, reasons=["served field match 0.81 against 0.9"])
    with pytest.raises(ReleaseError, match="the shadow check of v3 failed .served field match"):
        promote(client, "v3", promoted_by="A. Reviewer")
    assert _statuses(client) == {"v3": "gated"}


def test_a_passing_shadow_report_promotes_and_the_promotion_is_recorded(client):
    _gated(client, "v2")
    promote_release(client, "v2", promoted_by="A. Reviewer")
    _gated(client, "v3", tier="production", tier_confidence=0.97)
    key = _passing_report(client, "v3", compared_with="v2")

    assert promote(client, "v3", promoted_by="B. Operator") == ["v2"]
    assert _statuses(client) == {"v2": "archived", "v3": "promoted"}
    [record] = client.read_json(paths.release_bundle("v3"))["promotions"]
    assert record["promoted_by"] == "B. Operator" and record["promoted_at"]
    assert record["shadow_report"] == key and record["shadow_skipped_reason"] is None
    assert (record["tier"], record["tier_confidence"], record["rollback"]) == ("production", 0.97, False)


@pytest.mark.parametrize("problem,report", [
    ("packaged again", {"bundle_created_at": "2026-10-01T00:00:00+00:00"}),
    ("compared v3 with nothing promoted, and v2 is promoted now", {"promoted_release_id": None}),
    ("is 'v4''s, not v3's", {"release_id": "v4"}),
])
def test_a_stale_shadow_report_is_refused(client, problem, report):
    """A report of the bundle as packaged before, or compared with another
    release than the one promoted now, says nothing about this promotion."""
    _gated(client, "v2")
    promote_release(client, "v2", promoted_by="A. Reviewer")
    _gated(client, "v3")
    _passing_report(client, "v3", compared_with="v2", **report)
    with pytest.raises(ReleaseError, match=problem.replace("(", r"\(")):
        promote(client, "v3", promoted_by="A. Reviewer")
    assert _statuses(client)["v2"] == "promoted"


def test_skipping_the_shadow_needs_a_written_reason_recorded_with_the_name(client):
    _gated(client, "v3")
    with pytest.raises(ReleaseError, match="20 characters or more"):
        promote(client, "v3", promoted_by="A. Reviewer", skip_shadow_reason="pod is down")
    reason = "the pod is down and v3 fixes the edition-date defect TICKET-12"
    assert promote(client, "v3", promoted_by="A. Reviewer", skip_shadow_reason=reason) == []
    [record] = client.read_json(paths.release_bundle("v3"))["promotions"]
    assert record["promoted_by"] == "A. Reviewer" and record["shadow_skipped_reason"] == reason
    assert "there is no shadow report" in record["shadow_problem"] and record["shadow_report"] is None


def test_an_interim_release_needs_accept_interim(client):
    _gated(client, "v3", tier="interim", tier_confidence=0.4)
    _passing_report(client, "v3", compared_with=None)
    with pytest.raises(ReleaseError, match="interim-tier release .the gate's tier, tier confidence 0.4.*--accept-interim"):
        promote(client, "v3", promoted_by="A. Reviewer")
    promote(client, "v3", promoted_by="A. Reviewer", accept_interim=True)
    [record] = client.read_json(paths.release_bundle("v3"))["promotions"]
    assert record["accepted_interim"] is True and record["tier"] == "interim" and record["tier_confidence"] == 0.4


def test_a_rollback_to_a_release_promoted_before_needs_no_shadow(client):
    _gated(client, "v2")
    promote_release(client, "v2", promoted_by="A. Reviewer")
    _gated(client, "v3")
    promote(client, "v3", promoted_by="A. Reviewer", skip_shadow_reason="an urgent fix agreed with the client")
    assert _statuses(client) == {"v2": "archived", "v3": "promoted"}

    assert promote(client, "v2", promoted_by="C. Oncall") == ["v3"]
    assert _statuses(client) == {"v2": "promoted", "v3": "archived"}
    first, rollback = client.read_json(paths.release_bundle("v2"))["promotions"]
    assert first["rollback"] is False and rollback["rollback"] is True
    assert rollback["promoted_by"] == "C. Oncall" and rollback["shadow_report"] is None


def test_serving_and_ocr_pin_refusals_stay_in_force_at_promote(client):
    """A passing shadow report, or a written reason, never lets through a release
    serving would refuse at cold start."""
    _gated(client, "v3", prompt_hash="sha256:another")
    _passing_report(client, "v3", compared_with=None)
    with pytest.raises(ReleaseError, match="serving would refuse it at cold start"):
        promote(client, "v3", promoted_by="A. Reviewer", skip_shadow_reason="x" * 30)
    _gated(client, "v4", ocr_pin={"mineru_version": "3.4.5"})
    _passing_report(client, "v4", compared_with=None)
    with pytest.raises(ReleaseError, match="serving would refuse it"):
        promote(client, "v4", promoted_by="A. Reviewer")
    assert _statuses(client) == {"v3": "gated", "v4": "gated"}


def test_promote_command_flags(client, monkeypatch, capsys):
    """--skip-shadow needs a 20-character --reason, --reason needs --skip-shadow:
    refused before anything is read. A promotion prints what it rests on."""
    for argv, words in (
        (["--skip-shadow"], "needs --reason"),
        (["--skip-shadow", "--reason", "too short"], "20 characters or more"),
        (["--reason", "a reason long enough to count here"], "give both or neither"),
    ):
        with pytest.raises(SystemExit) as exited:
            cli.main(["promote", "--version", "v3", "--by", "A. Reviewer", *argv])
        assert exited.value.code == 2 and words in capsys.readouterr().err

    _gated(client, "v3")
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: client)
    assert cli.main(["promote", "--version", "v3", "--by", "A. Reviewer"]) == 1
    assert "there is no shadow report of v3" in capsys.readouterr().err
    key = _passing_report(client, "v3", compared_with=None)
    assert cli.main(["promote", "--version", "v3", "--by", "A. Reviewer"]) == 0
    out = capsys.readouterr().out
    assert "v3 is promoted" in out and f"shadow report {key}" in out


# --------------------------------------------------------------------------
# shadow: which documents
# --------------------------------------------------------------------------


def _stub(source_id, lob, *, synthetic=False, doc_type="policy"):
    return SimpleNamespace(source_id=source_id, lob=lob, synthetic=synthetic, doc_type=doc_type)


def test_documents_are_stratified_by_line_real_first_and_deterministic():
    docs = ([_stub(f"auto_{i}", "personal_auto") for i in range(6)]
            + [_stub(f"home_{i}", "homeowners") for i in range(2)]
            + [_stub("home_twin", "homeowners", synthetic=True), _stub("acord", "personal_auto", doc_type="acord"),
               _stub("pkg", ["personal_auto", "homeowners"]), _stub("wc_1", "wc")])
    serves = lambda line: line != "wc"  # noqa: E731
    has_pdf = lambda source_id: source_id != "auto_3"  # noqa: E731

    chosen, passed = S.select_documents(docs, 6, serves=serves, has_pdf=has_pdf)
    lines = [line for _doc, line in chosen]
    # A line at a time in turn: homeowners has two real documents, then auto fills up.
    assert lines[:4] == ["homeowners", "personal_auto", "homeowners", "personal_auto"]
    assert all(not doc.synthetic for doc, _line in chosen)          # real before any twin
    assert passed["not_a_policy"] == ["acord"] and passed["no_single_line"] == ["pkg"]
    assert passed["line_not_served"] == ["wc_1"]
    again, _ = S.select_documents(list(reversed(docs)), 6, serves=serves, has_pdf=has_pdf)
    assert [d.source_id for d, _ in again] == [d.source_id for d, _ in chosen]

    # Twins only once every real document is taken; a document with no PDF is passed over.
    every, passed = S.select_documents(docs, 20, serves=serves, has_pdf=has_pdf)
    assert every[-1][0].source_id == "home_twin" and "auto_3" not in [d.source_id for d, _ in every]
    assert passed["no_pdf"] == ["auto_3"]


# --------------------------------------------------------------------------
# shadow: the check on the server path
# --------------------------------------------------------------------------


def test_a_candidate_as_good_as_the_promoted_release_passes_and_can_be_promoted(
        client, raw, documents, tmp_path, monkeypatch, capsys):
    _gated(client, "v2")
    promote_release(client, "v2", promoted_by="A. Reviewer")
    _gated(client, "v3")
    models = Models({"policy-v3": GOOD, "policy-v2": GOOD})

    report = _shadow(client, raw, documents, tmp_path, models)
    assert report["passed"], report["reasons"]
    assert models.asked == ["policy-v3", "policy-v2"]           # each adapter checked by name, once
    assert report["promoted_release_id"] == "v2" and report["baseline"]["compared"] is True
    assert report["documents"]["paired"] == 4 and report["documents"]["by_line"] == {"personal_auto": 4}
    paired = report["metrics"]["paired"]
    assert paired["candidate"] == paired["promoted"]
    assert paired["candidate"]["schema_validity_rate"] == 1.0 and paired["candidate"]["hallucination_rate"] == 0.0
    assert report["servers"]["max_loras"] == 2
    row = report["per_document"][0]
    assert row["candidate"]["route"] == "release" and row["candidate"]["invented"] == 0
    assert client.read_json(shadow_report_key("v3"))["passed"] is True

    table = S.render(report)
    assert "served field match" in table and "PASSED" in table and "vs v2 (promoted)" in table

    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: client)
    assert cli.main(["promote", "--version", "v3", "--by", "A. Reviewer"]) == 0
    assert _statuses(client) == {"v2": "archived", "v3": "promoted"}


def test_a_candidate_that_drops_field_match_and_invents_fails(client, raw, documents, tmp_path):
    _gated(client, "v2")
    promote_release(client, "v2", promoted_by="A. Reviewer")
    _gated(client, "v3")

    report = _shadow(client, raw, documents, tmp_path, Models({"policy-v3": WRONG, "policy-v2": GOOD}))
    assert not report["passed"]
    failed = {check["check"] for check in report["checks"] if not check["passed"]}
    assert failed == {"field_normalized_match", "hallucination_rate"}
    field_match = next(c for c in report["checks"] if c["check"] == "field_normalized_match")
    assert field_match["delta"] < -S.MAX_FIELD_MATCH_DROP
    assert report["per_document"][0]["candidate"]["invented"] == 1
    assert "FAIL" in S.render(report)
    with pytest.raises(ReleaseError, match="the shadow check of v3 failed"):
        promote(client, "v3", promoted_by="A. Reviewer")


def test_the_golds_are_the_tenant_s_frozen_set_as_the_gate_reads_them(client, raw, documents, tmp_path):
    """Without injected documents the check reads the frozen eval set
    (golden_eval.load_golden_set) - and refuses a tenant that froze none."""
    from evaluation.freeze_eval_set import manifest_key

    _gated(client, "v3")
    with pytest.raises(S.ShadowError, match="has no frozen eval set"):
        _shadow(client, raw, None, tmp_path, Models({"policy-v3": GOOD}))
    root = paths.golden_eval_set_dir(None)
    for doc in documents[:2]:
        client.write_json(f"{root}/{doc.source_id}/metadata.json", {
            "doc_type": "policy", "lob": "personal_auto", "reviewer_id": "A. Reviewer",
            "review_date": "2026-10-01"})
        for number, text in enumerate(PAGES, 1):
            client.write_bytes(f"{root}/{doc.source_id}/page_{number}.png", b"png")
            client.write_text(f"{root}/{doc.source_id}/page_{number}.md", text)
        client.write_json(f"{root}/{doc.source_id}/golden.json", GOOD)
    client.write_json(manifest_key(), {"source_ids": [d.source_id for d in documents[:2]]})

    report = _shadow(client, raw, None, tmp_path, Models({"policy-v3": GOOD}))
    assert report["passed"], report["reasons"]
    assert sorted(row["source_id"] for row in report["per_document"]) == ["policy_0001", "policy_0002"]


def test_with_nothing_promoted_the_candidate_is_held_to_the_golds_alone(client, raw, documents, tmp_path):
    _gated(client, "v3")
    report = _shadow(client, raw, documents, tmp_path, Models({"policy-v3": GOOD}))
    assert report["passed"] and report["baseline"] is None and report["promoted_release_id"] is None
    assert [c["check"] for c in report["checks"]] == ["infrastructure", "schema_validity"]
    assert "nothing promoted" in S.render(report)

    _gated(client, "v4")
    report = _shadow(client, raw, documents, tmp_path, Models({"policy-v4": PARTIAL}), version="v4")
    assert not report["passed"]
    assert report["reasons"] == [f"schema validity 0.0 on its own, at least {S.MIN_SCHEMA_VALIDITY_ALONE}"]


def test_failures_and_timeouts_fail_the_check_and_a_retried_reading_does_not(client, raw, documents, tmp_path):
    _gated(client, "v3")
    flaky = FakeReader(fail={"policy_0001.pdf": [ConnectionError("MinerU unreachable")]})
    report = _shadow(client, raw, documents, tmp_path, Models({"policy-v3": GOOD}), reader=flaky)
    assert report["passed"]                                              # tried again, and read
    assert report["per_document"][0]["candidate"]["attempts"] == 2

    broken = FakeReader(fail={"policy_0002.pdf": [TimeoutError("read timed out")] * S.READING_ATTEMPTS})
    report = _shadow(client, raw, documents, tmp_path, Models({"policy-v3": GOOD}), reader=broken)
    assert not report["passed"]
    assert report["failures"]["candidate"]["count"] == 1 and report["failures"]["candidate"]["timeouts"] == 1
    assert "policy_0002" in report["failures"]["candidate"]["documents"]
    assert report["checks"][0] == {"check": "infrastructure", "passed": False,
                                   "detail": "1 infrastructure failure(s)", "failures": 1}


def test_an_adapter_the_pod_does_not_serve_fails_the_check(client, raw, documents, tmp_path):
    """vLLM started with one adapter (--max-loras 1, no policy-v2): the
    promoted release cannot be read, which is an infrastructure failure and no
    comparison - never the rule for a tenant with nothing promoted."""
    _gated(client, "v2")
    promote_release(client, "v2", promoted_by="A. Reviewer")
    _gated(client, "v3")
    report = _shadow(client, raw, documents, tmp_path,
                     Models({"policy-v3": GOOD}, missing=("policy-v2",)))
    assert not report["passed"]
    assert "does not serve 'policy-v2'" in report["failures"]["promoted"]["adapter_check"]
    assert {c["check"] for c in report["checks"] if not c["passed"]} == {"infrastructure", "comparison"}

    # The candidate's own adapter missing: nothing is read at all.
    reader = FakeReader()
    report = _shadow(client, raw, documents, tmp_path, Models({}, missing=("policy-v3",)), reader=reader)
    assert not report["passed"] and reader.read_names == []
    assert report["failures"]["candidate"]["adapter_check"]


def test_shadow_refuses_a_release_promote_would_refuse_or_does_not_need(client, raw, documents, tmp_path):
    _gated(client, "v3", passed=False)
    with pytest.raises(S.ShadowError, match="would not be promoted whatever its shadow check says"):
        _shadow(client, raw, documents, tmp_path, Models({}))
    _gated(client, "v4")
    promote_release(client, "v4", promoted_by="A. Reviewer")
    with pytest.raises(S.ShadowError, match="already the promoted release"):
        _shadow(client, raw, documents, tmp_path, Models({}), version="v4")
    _gated(client, "v5")
    promote_release(client, "v5", promoted_by="A. Reviewer")
    with pytest.raises(S.ShadowError, match="a rollback to it needs no shadow check"):
        _shadow(client, raw, documents, tmp_path, Models({}), version="v4")


def test_prompt_check_warnings_are_counted_as_a_parity_signal():
    stats = [{"prompt_check": {"plausible": True}}, {"prompt_check": {"plausible": False}}, {"outcome": "read"}]
    readings = {"a": S.Reading("a", result=SimpleNamespace(window_stats=stats)),
                "b": S.Reading("b", error="ConnectionError: down")}
    assert S.prompt_parity(readings) == {"windows_checked": 2, "implausible": 1, "documents": ["a"]}


def test_a_timeout_is_told_apart_from_another_failure():
    assert S._is_timeout(TimeoutError("x"))
    wrapped = RuntimeError("window failed")
    wrapped.__cause__ = OSError("The read operation timed out")
    assert S._is_timeout(wrapped)
    assert not S._is_timeout(ConnectionError("refused"))


# --------------------------------------------------------------------------
# shadow: the command, its flags and the detach guard
# --------------------------------------------------------------------------


@pytest.fixture
def launched(monkeypatch):
    from orchestration import detach

    calls = []
    monkeypatch.setattr(detach.subprocess, "run", lambda cmd, check: calls.append(cmd))
    for var in ("RUNPOD_POD_ID", "TMUX", detach.DETACHED_ENV, detach.NO_DETACH_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(detach, "RUNPOD_ENV_FILE", Path("/nonexistent/rp_environment"))
    monkeypatch.setattr(detach, "_is_linux", lambda: False)
    return calls


SHADOW_ARGS = ["shadow", "--version", "v3", "--model-url", "https://pod-8000.proxy.runpod.net",
               "--mineru-url", "https://pod-8001.proxy.runpod.net"]


@pytest.mark.parametrize("change,words", [
    (["--model-url", "http://pod.example.com"], "is not HTTPS"),
    (["--documents", "0"], "--documents is 1 or more"),
    (["--version", "latest"], "names a release, vN"),
])
def test_shadow_flags_are_checked_before_the_detach(launched, monkeypatch, capsys, change, words):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    argv = list(SHADOW_ARGS)
    for flag, value in zip(change[::2], change[1::2], strict=True):
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv += [flag, value]
    with pytest.raises(SystemExit) as exited:
        cli.main(argv)
    assert exited.value.code == 2 and words in capsys.readouterr().err
    assert launched == []


def test_shadow_detaches_on_the_pod_and_runs_in_place_off_it(launched, monkeypatch, capsys):
    monkeypatch.setattr(S, "run_shadow", lambda *a, **k: pytest.fail("ran in the foreground"))
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    assert cli.main(SHADOW_ARGS) == 0
    [cmd] = launched
    assert cmd[-len(SHADOW_ARGS):] == SHADOW_ARGS and cmd[3].startswith("shadow-")
    assert "shadow" not in cli.QUICK_COMMANDS

    monkeypatch.delenv("RUNPOD_POD_ID")
    seen = {}

    def fake(client, version, **kwargs):
        seen.update(version=version, **kwargs)
        return {"release_id": version, "passed": False, "report_key": "k", "candidate": {"tier": None},
                "documents": {"selected": 1, "paired": 1, "real": 1, "synthetic": 0, "by_line": {"auto": 1}},
                "baseline": None, "metrics": {"paired": {"candidate": {}, "promoted": None}},
                "checks": [{"check": "infrastructure", "passed": False, "detail": "1 infrastructure failure(s)"}],
                "failures": {"candidate": {"count": 1, "timeouts": 0}, "promoted": {"count": 0, "timeouts": 0}},
                "prompt_check": {"candidate": {"implausible": 0, "windows_checked": 0}, "promoted": None}}

    monkeypatch.setattr(S, "run_shadow", fake)
    monkeypatch.setattr(cli, "BlobClient", lambda *a, **k: object())
    assert cli.main([*SHADOW_ARGS, "--documents", "7", "--tenant", "acme"]) == 1     # failed: exit 1
    assert seen["version"] == "v3" and seen["documents"] == 7 and seen["tenant_id"] == "acme"
    assert "FAILED" in capsys.readouterr().out
