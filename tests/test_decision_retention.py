"""Retention for review (decision 2026-10-26, #43): off by default, per tenant, never via a request.

Synthetic only: the in-memory Blob backend and queue, the echo model and the
fake served MinerU of tests/test_azure_service.py - no Azure, no network, no GPU.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the [service] extra is not installed")
pytest.importorskip("httpx", reason="the [service] extra is not installed")

from artifact_registry.blob_client import (  # noqa: E402
    AccessDeniedError,
    BlobClient,
    BlobError,
    InMemoryBackend,
)
from azure_service import retention as retention_module  # noqa: E402
from azure_service.job_queue import job_message  # noqa: E402
from azure_service.retention import (  # noqa: E402
    EXPORTED,
    RETAINED,
    RETAINING,
    REVIEWED,
    BlobIntake,
    LocalIntake,
    RetentionError,
    ReviewRetention,
    flag_summary,
    in_sample,
    intake_sink,
    reasons_to_retain,
    review_retention,
)
from azure_service.settings import Settings, SettingsError  # noqa: E402
from azure_service.store import JobStore, new_record  # noqa: E402
from azure_service.webhooks import WebhookSender  # noqa: E402
from azure_service.worker import Worker  # noqa: E402
from tests.test_azure_service import OTHER_KEY, PDF, Clock, Service  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REVIEW = "review-retention"
NAME = "jane doe policy.pdf"          # a caller's file name may name a person
DAY = 86400.0


def clean_output(confidence: float | None = 0.95, **changes) -> dict:
    output = {
        "doc_type": "policy", "schema_valid": True, "overall_confidence": confidence, "review_flags": [],
        "extraction": {"policy": {"policy_number": {"raw": "PA-1", "parsed": "PA-1", "page_ref": [1],
                                                    "confidence": {"score": 0.95}, "flagged": False}}},
        "fields": {"policy.policy_number": {"value": "PA-1", "confidence": 0.95}},
    }
    output.update(changes)
    return output


def flagged_output() -> dict:
    output = clean_output()
    output["extraction"]["policy"]["policy_number"]["flagged"] = True
    return output


class Parts:
    """A jobs container and a review container on one in-memory account."""

    def __init__(self, clock: Clock | None = None, **settings) -> None:
        self.clock = clock or Clock()
        self.settings = Settings(**{"review_retention_tenants": ("default",), "review_retention_sample": 0.0,
                                    **settings})
        self.blob = InMemoryBackend()
        self.store = JobStore(BlobClient(backend=self.blob, container="extraction-jobs", context="serving"))
        self.review = BlobClient(backend=self.blob, container=REVIEW, context="serving")
        self.retention = ReviewRetention(self.settings, self.review, clock=self.clock)

    def done_job(self, *, tenant: str = "default", output: dict | None = None, status: str = "done") -> dict:
        job_id = uuid.uuid4().hex
        record = new_record(job_id, tenant, lob="personal_auto", name=NAME, size=len(PDF), pages=3,
                            webhook_url=None)
        record["status"] = status
        self.store.put_pdf(tenant, job_id, PDF)
        self.store.write_status(record)
        self.store.write_result(tenant, job_id, {
            "job_id": job_id, "status": "done", "final": True, "full_text": True, "model": "policy-v3",
            "release": "v3", "route": "release", "lob": "personal_auto",
            "output": output if output is not None else flagged_output()})
        return record

    def kept(self) -> list[str]:
        return sorted(key for (container, key) in self.blob._store if container == REVIEW)


def review_keys(service: Service) -> list[str]:
    return sorted(key for (container, key) in service.blob._store if container == REVIEW)


def jobs_keys(service: Service) -> list[str]:
    return sorted(key for (container, key) in service.blob._store if container == "extraction-jobs")


def run_job(service: Service, worker: Worker, **submit) -> str:
    job_id = service.submit(**submit)["job_id"]
    assert worker.poll_once() and worker.text_once()
    return job_id


def worker_with(service: Service, retention) -> Worker:
    webhooks = WebhookSender(service.settings, post=service.hooks, background=False, backoff_s=(0.0,))
    return Worker(service.store, service.queue, service.runtime, service.settings, webhooks=webhooks,
                  clock=service.clock, retention=retention)


def retention_for(service: Service, backend=None) -> ReviewRetention:
    client = BlobClient(backend=backend or service.blob, container=REVIEW, context="serving")
    return ReviewRetention(service.settings, client, clock=service.clock)


# --------------------------------------------------------------------------
# Off by default
# --------------------------------------------------------------------------


def test_with_the_default_settings_nothing_is_retained(tmp_path):
    defaults = Settings.from_env({})
    assert defaults.review_retention_tenants == () and not defaults.retains_for("default")
    assert defaults.describe()["review_retention"] == "off"
    assert review_retention(defaults) is None and review_retention(Settings()) is None

    # Even a worker handed a retention keeps nothing while no tenant is named -
    # for a job that is flagged, uncalibrated and, by the default rule, low on nothing.
    service = Service(tmp_path)
    worker = worker_with(service, retention_for(service))
    job_id = run_job(service, worker, name=NAME)
    assert service.status(job_id)["status"] == "done"
    assert review_keys(service) == []
    assert jobs_keys(service) == [f"jobs/default/{job_id}/result.json", f"jobs/default/{job_id}/status.json"]

    parts = Parts(review_retention_tenants=())
    assert parts.retention.keep(parts.store, parts.done_job(output=clean_output(None))) is None
    assert parts.kept() == []


# --------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------


def test_an_uncalibrated_result_alone_keeps_nothing(tmp_path):
    """The synthetic release has no calibrators: its result carries only the
    document-level `uncalibrated` and `schema:invalid` flags, and every value is
    flagged for that reason alone. Kept, it would keep nearly every document."""
    service = Service(tmp_path, review_retention_tenants=("default",), review_retention_sample=0.0)
    worker = worker_with(service, retention_for(service))
    job_id = run_job(service, worker)
    output = service.store.read_result("default", job_id)["output"]
    assert {flag.rpartition(":")[2] for flag in output["review_flags"]} == {"uncalibrated", "invalid"}
    assert output["extraction"]["policy"]["policy_number"]["flagged"] is True
    assert review_keys(service) == []


def test_an_opted_in_tenants_flagged_job_is_kept_before_its_pdf_is_deleted(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="azure_service")
    # Below this, the echo model's 0.96 is low: the job is kept for its confidence.
    service = Service(tmp_path, review_retention_tenants=("default",), review_retention_low_confidence=0.99)
    worker = worker_with(service, retention_for(service))
    job_id = run_job(service, worker, name=NAME)

    base = f"retained/default/{job_id}"
    assert review_keys(service) == [f"{base}/document.pdf", f"{base}/manifest.json", f"{base}/result.json"]
    review = BlobClient(backend=service.blob, container=REVIEW, context="serving")
    assert review.read_bytes(f"{base}/document.pdf") == PDF               # copied before it was deleted ...
    assert review.read_json(f"{base}/result.json") == service.store.read_result("default", job_id)
    assert jobs_keys(service) == [f"jobs/default/{job_id}/result.json", f"jobs/default/{job_id}/status.json"]

    manifest = review.read_json(f"{base}/manifest.json")
    assert manifest["tenant"] == "default" and manifest["job_id"] == job_id and manifest["lob"] == "personal_auto"
    assert manifest["reasons"] == ["low_confidence"] and manifest["triggered_kinds"] == []
    assert manifest["state"] == RETAINED and manifest["overall_confidence"] == pytest.approx(0.9608)
    assert manifest["model"] == "policy-v3" and manifest["release"] == "v3"
    assert manifest["flags"]["review_flags"] == 2 and set(manifest["flags"]["flag_kinds"]) == {"invalid",
                                                                                             "uncalibrated"}
    assert manifest["retained_at"] == "1970-01-12T13:46:40Z" and manifest["expires_at"] == "1970-04-12T13:46:40Z"
    # No document content: neither the file name, nor a value, nor a field path.
    text = json.dumps(manifest)
    assert "jane" not in text and "PA-1" not in text and "policy_number" not in text

    worker_logs = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("azure_service"))
    assert f"job {job_id}: retained for review (low_confidence)" in worker_logs
    assert "jane" not in worker_logs and "PA-1" not in worker_logs


def test_another_tenant_is_never_retained_even_if_its_request_asks(tmp_path):
    service = Service(tmp_path, review_retention_tenants=("default",), review_retention_sample=1.0,
                      review_retention_low_confidence=1.0)
    worker = worker_with(service, retention_for(service))
    theirs = run_job(service, worker, key=OTHER_KEY, retain="true", review_retention="1", keep="true",
                     review="true")
    assert all(f"/{theirs}/" not in key for key in review_keys(service)) and review_keys(service) == []
    ours = run_job(service, worker)
    assert len(review_keys(service)) == 3

    # The API shows nothing new: the kept job's status and result have the keys
    # the other's have, and none of them speaks of retention.
    kept_view, other_view = service.status(ours), service.get(f"/v1/extractions/{theirs}", key=OTHER_KEY).json()
    assert sorted(kept_view) == sorted(other_view)
    kept_result = service.get(f"/v1/extractions/{ours}/result").json()
    other_result = service.get(f"/v1/extractions/{theirs}/result", key=OTHER_KEY).json()
    assert sorted(kept_result) == sorted(other_result)
    assert not any("retain" in key or "review_" in key for key in [*kept_view, *kept_result])


def test_a_failed_copy_never_fails_the_job(tmp_path, caplog):
    class NoReviewWrites(InMemoryBackend):
        def write(self, container, key, data):
            if container == REVIEW:
                raise BlobError("forbidden")
            super().write(container, key, data)

    service = Service(tmp_path, review_retention_tenants=("default",), review_retention_low_confidence=0.99)
    backend = NoReviewWrites()
    backend._store = service.blob._store                 # the same account, its review container refused
    worker = worker_with(service, retention_for(service, backend))
    with caplog.at_level(logging.WARNING, logger="azure_service.worker"):
        job_id = run_job(service, worker, name=NAME)
    assert service.status(job_id)["status"] == "done" and service.status(job_id)["error"] is None
    assert jobs_keys(service) == [f"jobs/default/{job_id}/result.json", f"jobs/default/{job_id}/status.json"]
    assert f"job {job_id}: not retained for review (BlobError)" in caplog.text and "jane" not in caplog.text


def test_a_job_whose_worker_died_before_retaining_it_is_kept_on_its_next_delivery(tmp_path):
    service = Service(tmp_path, review_retention_tenants=("default",), review_retention_low_confidence=0.99,
                      keep_files=True)
    job_id = run_job(service, worker_with(service, None))      # ended done; nothing kept; the PDF still there
    assert review_keys(service) == []
    retaining = worker_with(service, retention_for(service))
    service.queue.send(job_message(job_id, "default"))         # the message came back
    assert retaining.poll_once()
    assert len(review_keys(service)) == 3 and len(service.queue) == 0
    first = service.blob._store[(REVIEW, f"retained/default/{job_id}/manifest.json")]
    service.clock.advance(60)
    service.queue.send(job_message(job_id, "default"))
    assert retaining.poll_once()
    assert service.blob._store[(REVIEW, f"retained/default/{job_id}/manifest.json")] == first   # kept once


# --------------------------------------------------------------------------
# What is kept
# --------------------------------------------------------------------------


def test_a_job_is_kept_when_flagged_low_on_confidence_or_sampled():
    rule = Settings(review_retention_tenants=("default",), review_retention_sample=0.0)
    job = uuid.uuid4().hex

    def reasons(output, settings=rule):
        return reasons_to_retain({"output": output}, job, settings)[0]

    assert reasons(clean_output()) == []
    assert reasons(flagged_output()) == ["flagged"]
    assert reasons(clean_output(review_flags=["policy.policy_number:near_tie"])) == ["flagged"]
    assert reasons(clean_output(0.79)) == ["low_confidence"]
    assert reasons(clean_output(0.80)) == []
    assert reasons(clean_output(None)) == ["low_confidence"] and reasons(clean_output(float("nan"))) == [
        "low_confidence"]
    assert reasons({}) == ["low_confidence"]
    everyone = Settings(review_retention_tenants=("default",), review_retention_sample=1.0)
    assert reasons(clean_output(), everyone) == ["sample"]
    assert reasons(flagged_output(), everyone) == ["flagged"]         # the sample is of unflagged jobs

    summary = flag_summary(clean_output(review_flags=["a.b:low_confidence", "a.c:low_confidence", "x:Jane Doe",
                                                      "routing:no_extractable_candidate",
                                                      "window_failed:declarations p1-3"]),
                           ("low_confidence", "near_tie"))
    assert summary == {"flagged_fields": 0, "review_flags": 5, "triggered_kinds": ["low_confidence"],
                       "flag_kinds": {"low_confidence": 2, "no_extractable_candidate": 1, "other": 1,
                                      "window_failed": 1}}


def test_a_document_level_flag_alone_keeps_nothing_and_a_value_flag_does():
    rule = Settings(review_retention_tenants=("default",), review_retention_sample=0.0)
    job = uuid.uuid4().hex

    # Only `uncalibrated` (and the schema's): the value it flagged is not counted either.
    uncalibrated = flagged_output()
    uncalibrated["review_flags"] = ["policy.policy_number:uncalibrated", "schema:invalid",
                                    "printed_values:no_page_text"]
    reasons, summary = reasons_to_retain({"output": uncalibrated}, job, rule)
    assert reasons == [] and summary["flagged_fields"] == 0 and summary["triggered_kinds"] == []
    assert reasons_to_retain({"output": clean_output(review_flags=["policy.policy_number:uncalibrated"])},
                             job, rule)[0] == []

    # A value that needs a person: kept, and the manifest names the kind - never the path.
    mismatch = clean_output(review_flags=["policy.policy_number:uncalibrated", "named_insured[0].name:role_mismatch"])
    reasons, summary = reasons_to_retain({"output": mismatch}, job, rule)
    assert reasons == ["flagged"] and summary["triggered_kinds"] == ["role_mismatch"]
    parts = Parts()
    manifest = parts.retention.keep(parts.store, parts.done_job(output=mismatch))
    assert manifest["reasons"] == ["flagged"] and manifest["triggered_kinds"] == ["role_mismatch"]
    assert "named_insured" not in json.dumps(manifest) and "role_mismatch" in manifest["rule"]["flag_kinds"]

    # A flagged value whose path is not the uncalibrated one still counts.
    other_path = flagged_output()
    other_path["review_flags"] = ["policy.effective_date:uncalibrated"]
    assert reasons_to_retain({"output": other_path}, job, rule)[0] == ["flagged"]
    # The kinds are a setting: without role_mismatch it keeps nothing.
    narrow = Settings(review_retention_tenants=("default",), review_retention_sample=0.0,
                      review_retention_flag_kinds=("low_confidence",))
    assert reasons_to_retain({"output": mismatch}, job, narrow)[0] == []


def test_the_sample_is_a_stable_share_of_job_ids():
    ids = [uuid.UUID(int=n * 7919 + 13).hex for n in range(4000)]
    share = sum(in_sample(job, 0.05) for job in ids) / len(ids)
    assert 0.035 < share < 0.065
    assert [in_sample(job, 0.05) for job in ids[:200]] == [in_sample(job, 0.05) for job in ids[:200]]
    assert not any(in_sample(job, 0.0) for job in ids[:50]) and all(in_sample(job, 1.0) for job in ids[:50])


def test_keep_copies_once_and_only_a_done_job_with_its_pdf():
    parts = Parts()
    record = parts.done_job()
    manifest = parts.retention.keep(parts.store, record)
    assert manifest is not None and manifest["reasons"] == ["flagged"] and manifest["state"] == RETAINED
    assert "jane" not in json.dumps(manifest) and "document" not in manifest
    assert parts.retention.keep(parts.store, record) is None                       # kept already
    assert parts.retention.keep(parts.store, parts.done_job(output=clean_output())) is None   # nothing to review
    assert parts.retention.keep(parts.store, parts.done_job(status="failed")) is None
    gone = parts.done_job()
    parts.store.delete_transient("default", gone["job_id"])
    assert parts.retention.keep(parts.store, gone) is None
    assert parts.retention.keep(parts.store, parts.done_job(tenant="other")) is None
    assert len(parts.kept()) == 3


# --------------------------------------------------------------------------
# Purge, review, export
# --------------------------------------------------------------------------


def test_purge_deletes_expired_reviewed_and_incomplete_items_and_nothing_else():
    parts = Parts(review_retention_tenants=("acme", "acme2"))
    old = parts.retention.keep(parts.store, parts.done_job(tenant="acme"))
    parts.clock.advance(80 * DAY)
    fresh = parts.retention.keep(parts.store, parts.done_job(tenant="acme"))
    reviewed = parts.retention.keep(parts.store, parts.done_job(tenant="acme"))
    neighbour = parts.retention.keep(parts.store, parts.done_job(tenant="acme2"))
    parts.retention.mark_reviewed("acme", reviewed["job_id"])
    stale = uuid.uuid4().hex                                    # a worker died mid-copy, two days ago
    parts.review.write_json(f"retained/acme/{stale}/manifest.json", {
        "tenant": "acme", "job_id": stale, "state": RETAINING, "retained_at": "1970-04-01T00:00:00Z",
        "expires_at": "1970-06-30T00:00:00Z"})
    orphan = uuid.uuid4().hex
    parts.review.write_bytes(f"retained/acme/{orphan}/document.pdf", PDF)
    parts.clock.advance(11 * DAY)                               # `old` is 91 days old now

    with pytest.raises(RetentionError, match="--tenant"):
        parts.retention.purge(everything=True)
    dry = parts.retention.purge("acme", dry_run=True)
    assert dry.deleted == {"expired": 1, "reviewed": 1, "incomplete": 1, "no_manifest": 1} and dry.kept == 1
    before = parts.kept()
    assert parts.retention.purge("acme", dry_run=True) and parts.kept() == before
    report = parts.retention.purge("acme")
    assert "deleted 4 item(s)" in report.describe() and not report.errors
    left = {key.split("/")[2] for key in parts.kept()}
    assert left == {fresh["job_id"], neighbour["job_id"]}       # acme2 is not acme's
    assert old["job_id"] not in left
    assert parts.retention.purge("acme2", everything=True).deleted == {"all": 1}
    assert {key.split("/")[1] for key in parts.kept()} == {"acme"}
    with pytest.raises(RetentionError, match="no retained item"):
        parts.retention.mark_reviewed("acme", uuid.uuid4().hex)


def test_export_writes_unlabelled_intake_folders_and_never_a_gold(tmp_path):
    from data_pipeline.audit import AuditReport, audit_document
    from data_pipeline.ingestion.import_labeled_pdfs import LabeledPdfError, read_bundle
    from data_pipeline.labeling.export_golden_labels import review_problem

    parts = Parts()
    first = parts.retention.keep(parts.store, parts.done_job())
    parts.clock.advance(60)
    second = parts.retention.keep(parts.store, parts.done_job())
    out = tmp_path / "review-2026-11"

    report = parts.retention.export("default", LocalIntake(out), limit=1)
    name = f"default__{first['job_id']}"
    assert report.exported == [name] and report.waiting == 1 and "UNLABELLED" in report.describe()
    folder = out / name
    assert sorted(p.name for p in folder.iterdir()) == ["document.pdf", "draft.json", "metadata.json"]
    assert (folder / "document.pdf").read_bytes() == PDF
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["lob"] == "personal_auto" and metadata["synthetic"] is False and metadata["labelled"] is False
    assert metadata["production_job_id"] == first["job_id"] and metadata["retained_reasons"] == ["flagged"]
    assert not {"reviewer_id", "review_date"} & set(metadata)
    draft = json.loads((folder / "draft.json").read_text(encoding="utf-8"))
    assert draft["trusted"] is False and draft["draft_backend"] == "own_finetuned"
    assert draft["extraction"]["policy"]["policy_number"]["raw"] == "PA-1"
    assert draft["field_confidence"] == {"policy.policy_number": 0.95}

    # The import and the audit refuse it until a person labels it.
    with pytest.raises(LabeledPdfError, match="no golden.json"):
        read_bundle(folder)
    assert review_problem({"policy": {}}, metadata)                    # and a gold dropped in without a record
    audit = AuditReport(root=str(out))
    audit_document(folder, audit, {})
    assert ("structure", "no golden.json") in {(f.check, f.detail) for f in audit.blockers}

    manifest = parts.review.read_json(f"retained/default/{first['job_id']}/manifest.json")
    assert manifest["state"] == EXPORTED and manifest["exported_to"].endswith(name)
    assert parts.retention.export("default", LocalIntake(out)).exported == [f"default__{second['job_id']}"]
    assert parts.retention.export("default", LocalIntake(out)).exported == []      # each once

    with pytest.raises(RetentionError, match="REVIEW_RETENTION_TENANTS"):
        Parts(review_retention_tenants=("acme",)).retention.export("default", LocalIntake(out))
    parts.retention.mark_reviewed("default", first["job_id"])
    assert parts.retention.purge().deleted == {"reviewed": 1}


def test_export_to_the_raw_containers_intake_is_what_pull_intake_reads(tmp_path):
    from data_pipeline.ingestion.pull_intake import pull_intake

    parts = Parts()
    kept = parts.retention.keep(parts.store, parts.done_job())
    raw = BlobClient(backend=parts.blob, container="main", raw_container="raw", context="ingestion")
    sink = intake_sink("intake/review-1", parts.settings, client=raw)
    assert isinstance(sink, BlobIntake) and isinstance(intake_sink("./intake/review-1", parts.settings),
                                                       LocalIntake)
    with pytest.raises(RetentionError):
        intake_sink("intake/a..b", parts.settings, client=raw)
    assert parts.retention.export("default", sink).exported == [f"default__{kept['job_id']}"]
    assert {container for (container, key) in parts.blob._store if key.startswith("intake/")} == {"raw"}

    pulled = pull_intake(raw, "review-1", tmp_path, workers=2)
    assert pulled.downloaded == 3 and not pulled.failed
    folder = tmp_path / "review-1" / f"default__{kept['job_id']}"
    assert sorted(p.name for p in folder.iterdir()) == ["document.pdf", "draft.json", "metadata.json"]

    # The service's own client may not write the raw container's intake.
    with pytest.raises(AccessDeniedError):
        BlobIntake(parts.review, "review-2").write("x", "document.pdf", PDF)


def test_the_retention_cli(monkeypatch, tmp_path, capsys):
    from orchestration import detach

    parts = Parts()
    record = parts.done_job()
    ReviewRetention(parts.settings, parts.review).keep(parts.store, record)     # now: the CLI's clock
    monkeypatch.setattr(retention_module, "review_client", lambda settings, backend=None: parts.review)
    monkeypatch.setattr(detach, "detach_module_if_needed", lambda *a, **k: False)
    for name in ("REVIEW_CONTAINER", "AZURE_RAW_CONTAINER", "JOBS_CONTAINER", "AZURE_BLOB_CONTAINER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STORAGE_AUTH", "environment")
    monkeypatch.setenv("REVIEW_RETENTION_TENANTS", "default")

    main = retention_module.main
    assert main(["purge", "--all"]) == 2 and "--tenant" in capsys.readouterr().err
    assert main(["export", "--tenant", "Default", "--out", str(tmp_path / "out")]) == 0
    assert "1 unlabelled document(s)" in capsys.readouterr().out
    assert main(["mark-reviewed", "--tenant", "default", "--job", record["job_id"]]) == 0
    assert parts.review.read_json(f"retained/default/{record['job_id']}/manifest.json")["state"] == REVIEWED
    assert main(["purge", "--dry-run"]) == 0 and "would delete 1" in capsys.readouterr().out
    assert main(["purge", "--tenant", "default"]) == 0 and parts.kept() == []
    assert main(["mark-reviewed", "--tenant", "default", "--job", "nope"]) == 2


# --------------------------------------------------------------------------
# Settings and documentation
# --------------------------------------------------------------------------


NAMES = ("REVIEW_RETENTION_TENANTS", "REVIEW_RETENTION_SAMPLE", "REVIEW_RETENTION_LOW_CONFIDENCE",
         "REVIEW_RETENTION_FLAG_KINDS", "REVIEW_RETENTION_DAYS", "REVIEW_CONTAINER")


def test_the_default_flag_kinds_are_kinds_serving_emits_and_no_document_level_one():
    from azure_service.settings import DEFAULT_RETENTION_FLAG_KINDS
    from calibration.list_completeness import ROW_MISMATCH_SUFFIX
    from serving import conventions, pipeline, printed_values, value_flags

    emitted = {
        "low_confidence", "no_confidence",                         # serving.pipeline, calibration.apply_calibration
        conventions.NOT_PRINTED_FLAG, conventions.LIKELY_MISSED_FLAG, printed_values.REMOVED_FLAG,
        printed_values.UNREAD_PAGE_FLAG, printed_values.LINK_REMOVED_FLAG, value_flags.OFF_CITED_PAGE_FLAG,
        value_flags.ROLE_MISMATCH_FLAG, value_flags.NEAR_TIE_FLAG, value_flags.PARSED_MISMATCH_FLAG,
        ROW_MISMATCH_SUFFIX.lstrip(":"), pipeline.WINDOW_FAILED_FLAG,  # data that may be missing
    }
    assert set(DEFAULT_RETENTION_FLAG_KINDS) == emitted
    serving_source = "\n".join(path.read_text(encoding="utf-8")
                               for path in (ROOT / "serving").glob("*.py")) + (
        ROOT / "calibration" / "apply_calibration.py").read_text(encoding="utf-8") + (
        ROOT / "calibration" / "list_completeness.py").read_text(encoding="utf-8")
    assert all(f":{kind}" in serving_source or f'"{kind}"' in serving_source for kind in DEFAULT_RETENTION_FLAG_KINDS)
    assert not {"uncalibrated", "invalid", "no_page_text"} & set(DEFAULT_RETENTION_FLAG_KINDS)
    env = Settings.from_env({"REVIEW_RETENTION_FLAG_KINDS": " Role_Mismatch, near_tie,near_tie "})
    assert env.review_retention_flag_kinds == ("role_mismatch", "near_tie")
    assert Settings.from_env({}).review_retention_flag_kinds == DEFAULT_RETENTION_FLAG_KINDS
    with pytest.raises(SettingsError, match="flag kinds"):
        Settings.from_env({"REVIEW_RETENTION_FLAG_KINDS": "policy.number:near_tie"})


def test_the_retention_settings_come_from_the_environment_and_are_documented():
    settings = Settings.from_env({"REVIEW_RETENTION_TENANTS": "Acme, default", "REVIEW_RETENTION_SAMPLE": "0.1",
                                  "REVIEW_RETENTION_LOW_CONFIDENCE": "0.7", "REVIEW_RETENTION_DAYS": "30",
                                  "REVIEW_CONTAINER": "kept-for-review"})
    assert settings.review_retention_tenants == ("acme", "default") and settings.retains_for("acme")
    assert not settings.retains_for("other") and not settings.retains_for("")
    assert (settings.review_retention_sample, settings.review_retention_low_confidence,
            settings.review_retention_days, settings.review_container) == (0.1, 0.7, 30, "kept-for-review")
    assert settings.describe()["review_retention"] == {"tenants": ["acme", "default"],
                                                       "container": "kept-for-review", "days": 30}
    defaults = Settings.from_env({})
    assert (defaults.review_retention_sample, defaults.review_retention_low_confidence,
            defaults.review_retention_days, defaults.review_container) == (0.05, 0.80, 90, "review-retention")
    for env, message in (({"REVIEW_RETENTION_SAMPLE": "1.5"}, "between 0 and 1"),
                         ({"REVIEW_RETENTION_LOW_CONFIDENCE": "-0.1"}, "at least"),
                         ({"REVIEW_RETENTION_DAYS": "0"}, "at least"),
                         ({"REVIEW_CONTAINER": "extraction-jobs"}, "of its own"),
                         ({"REVIEW_CONTAINER": "raw", "AZURE_RAW_CONTAINER": "raw"}, "of its own"),
                         ({"REVIEW_RETENTION_TENANTS": "a/b"}, "tenant")):
        with pytest.raises(SettingsError, match=message):
            Settings.from_env(env)

    template = (ROOT / ".env.example").read_text(encoding="utf-8")
    deploy = (ROOT / "azure_service" / "DEPLOY.md").read_text(encoding="utf-8")
    for name in NAMES:
        assert f"\n{name}=" in template and f"`{name}`" in deploy
    assert "## 9. Retention for review (off by default)" in deploy
    for topic in ("DPA", "consent", "RBAC", "purge", "export", "lifecycle", "customer-managed keys"):
        assert topic.lower() in deploy.lower(), topic


def test_the_start_probe_leaves_nothing_and_is_never_an_item():
    parts = Parts()
    retention = review_retention(parts.settings, parts.blob)
    assert retention is not None and retention.client.container == REVIEW
    retention.check()
    assert parts.kept() == []
    parts.review.write_bytes("retained/.access-check", b"")        # a probe a crash left
    assert parts.retention.items() == [] and parts.retention.purge().deleted == {}
