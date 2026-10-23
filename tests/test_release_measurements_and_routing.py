"""The release step records latency, GPU memory and a routing table, and fails
when routing does (Fideon SPEC_09 handoff item 6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.lob import known_lines
from common.schemas import SNAPSHOT_LINES
from evaluation import release_measurements
from evaluation.release_measurements import BASE_MODEL, latency_by_adapter, measurements, p95
from serving.routing_check import RoutingCheckError, plan_with_candidate, routing_table
from tests.test_registry import _bundle


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# --------------------------------------------------------------------------
# Latency and memory
# --------------------------------------------------------------------------

def test_p95_is_the_nearest_rank():
    assert p95(list(range(1, 101))) == 95 and p95([7.0]) == 7.0 and p95([]) is None


def test_latency_is_kept_per_adapter_one_entry_per_call():
    scored = [
        ({}, {}, {"adapter": "policy-v3", "window_latencies_ms": [100, 200, 300]}),
        ({}, {}, {"adapter": "policy-v3", "window_latencies_ms": [400]}),
        ({}, {}, {"adapter": None, "window_latencies_ms": [50]}),
        ({}, {}, {"adapter": "x", "error": "failed"}),
    ]
    assert latency_by_adapter(scored) == {
        BASE_MODEL: {"p95_ms": 50.0, "calls": 1},
        "policy-v3": {"p95_ms": 400.0, "calls": 4},
    }


def test_measurements_say_what_they_measured(monkeypatch):
    monkeypatch.setattr(release_measurements, "peak_gpu_memory_mb", lambda: 71234.0)
    body = measurements([({}, {}, {"adapter": "a", "window_latencies_ms": [10]})])
    assert body["peak_gpu_memory_mb"] == 71234.0 and "golden eval" in body["measured_with"]


def test_no_gpu_is_no_reading(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(release_measurements.subprocess, "run", missing)
    assert release_measurements.peak_gpu_memory_mb() is None


def test_extract_records_one_latency_per_generation_round(client):
    """A round's windows run together and vLLM gives each the round's time, so one
    entry per round - not one per window, which counted it once per window."""
    from tests.test_lossrun_windows_and_table_f1 import _serve, _Windows

    assert len(_serve(client, _Windows()).window_latencies_ms) == 1          # three windows, one round
    assert len(_serve(client, _Windows(), pages=2).window_latencies_ms) == 1  # one call

    class _FailsFirst(_Windows):
        def generate(self, messages, config, adapter=None):
            result = super().generate(messages, config, adapter)
            if self.windows[-1] == [1, 2, 3]:
                result.finish_reason = "length"                              # cut off: split and retried
            return result

    assert len(_serve(client, _FailsFirst()).window_latencies_ms) == 2       # a second round for the halves


def test_a_call_with_no_adapter_is_labelled_with_the_model_the_eval_loaded():
    scored = [({}, {}, {"adapter": None, "window_latencies_ms": [120.0]})]
    assert latency_by_adapter(scored, served_as="policy-v3") == {
        "policy-v3": {"p95_ms": 120.0, "calls": 1}}


def test_memory_is_the_peak_sampled_while_the_eval_runs(monkeypatch):
    readings = iter([1000.0, 5000.0, 3000.0, 2000.0, 2000.0, 2000.0])
    monkeypatch.setattr(release_measurements, "peak_gpu_memory_mb", lambda: next(readings, 2000.0))
    import time

    with release_measurements.GpuMemorySampler(interval_s=0.01) as sampler:
        time.sleep(0.15)
    assert sampler.peak_mb == 5000.0
    monkeypatch.setattr(release_measurements, "peak_gpu_memory_mb", lambda: None)
    with release_measurements.GpuMemorySampler(interval_s=0.01) as none:
        pass
    assert none.peak_mb is None


def test_the_golden_eval_keeps_the_adapter_and_each_calls_latency(client, monkeypatch):
    from evaluation import golden_eval
    from evaluation.golden_eval import GoldenDocument
    from inference_core.model_runner import load_model
    from tests.test_lossrun_windows_and_table_f1 import GOLD, PAGES, _page_text, _Windows

    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})
    doc = GoldenDocument(source_id="lossrun_0001", doc_type="lossrun", golden=GOLD,
                         image_keys=[f"page_{p}.png" for p in range(1, PAGES + 1)],
                         page_texts={p: _page_text(p) for p in range(1, PAGES + 1)})
    model = load_model("base", client, backend_impl=_Windows())
    (_, _, metadata), = golden_eval.evaluate([doc], model, client, "/tmp", modes=("ocr_plus_image",))
    assert "adapter" in metadata and len(metadata["window_latencies_ms"]) == 1   # one round of windows


# --------------------------------------------------------------------------
# The routing check
# --------------------------------------------------------------------------

def _candidate(release_id="v3", **over):
    """The policy adapter's release, trained on the personal lines."""
    from tests.test_release_router import BASE

    return _bundle(**{"release_id": release_id, "lines": list(SNAPSHOT_LINES),
                      "created_at": "2026-11-01T00:00:00+00:00", "adapter": "policy-v3",
                      "base_model": BASE, **over})


def test_every_line_is_routed_and_the_releases_own_lines_reach_it():
    bundle = _candidate()
    plan = plan_with_candidate(bundle.model_dump(mode="json"))
    rows = routing_table(plan, bundle.release_id, bundle.lines)
    assert [row["lob"] for row in rows] == sorted(known_lines())
    assert set(rows[0]) == {"lob", "release_id", "adapter", "lob_fallback_used"}
    by_line = {row["lob"]: row for row in rows}
    assert by_line["personal_auto"]["release_id"] == bundle.release_id
    assert by_line["personal_auto"]["adapter"] == "policy-v3"
    assert not by_line["personal_auto"]["lob_fallback_used"]
    assert by_line["agriculture_farm"]["lob_fallback_used"] and by_line["agriculture_farm"]["release_id"] is None


def test_the_candidate_is_routed_in_the_format_asked_for():
    bundle = _candidate(serving_formats={"bf16": "merged/bf16", "fp8": "merged/fp8"})
    assert plan_with_candidate(bundle.model_dump(mode="json"), "fp8").release.merged_model == "merged/fp8"


def test_a_release_recording_no_lines_takes_every_line():
    bundle = _candidate(lines=[])
    rows = routing_table(plan_with_candidate(bundle.model_dump(mode="json")), bundle.release_id, [])
    assert {row["release_id"] for row in rows} == {"v3"} and not any(r["lob_fallback_used"] for r in rows)


def test_a_line_the_release_was_trained_for_that_would_not_reach_it_fails_the_check():
    # It records the personal lines but serves no policies, so none reaches it.
    bundle = _candidate(doc_types=["acord"])
    plan = plan_with_candidate(bundle.model_dump(mode="json"))
    with pytest.raises(RoutingCheckError, match="homeowners is a line v3 was trained for, but routes to "
                                                "the base model"):
        routing_table(plan, bundle.release_id, bundle.lines)


def test_the_check_names_the_release_a_line_reaches_when_it_is_another():
    bundle = _candidate()
    plan = plan_with_candidate(bundle.model_dump(mode="json"))
    with pytest.raises(RoutingCheckError, match="homeowners is a line v4 was trained for, but routes to v3"):
        routing_table(plan, "v4", bundle.lines)


def test_the_release_step_writes_the_measurements_and_the_table_or_stops(client):
    from orchestration.pipeline_dag import PipelineError, record_release_measurements

    client.write_json(paths.eval_report("v3", scope="policy"), {"release_measurements": {
        "latency_p95_ms_by_adapter": {"policy-v3": {"p95_ms": 812.5, "calls": 40}},
        "peak_gpu_memory_mb": 70100.0, "measured_with": "the golden eval"}})
    ctx = SimpleNamespace(client=client, out_version="v3", tenant_id=None)
    bundle = _candidate()
    record_release_measurements(ctx, bundle)
    assert bundle.latency_p95_ms_by_adapter == {"policy-v3": 812.5}
    assert bundle.peak_gpu_memory_mb == 70100.0 and bundle.measured_with == "the golden eval"
    assert len(bundle.routing) == len(known_lines())

    with pytest.raises(PipelineError, match="is not released: routing check failed"):
        record_release_measurements(ctx, _candidate(doc_types=["acord"]))


def test_the_candidate_replaces_the_release_promoted_now(client):
    """The table says where each line goes once the candidate is served: the
    release promoted today - even one reading every line - takes none of it."""
    from orchestration.pipeline_dag import record_release_measurements
    from tests.test_release_router import promote

    promote(client, "v2", lines=())                                     # every line
    ctx = SimpleNamespace(client=client, out_version="v3", tenant_id=None)
    bundle = _candidate()
    record_release_measurements(ctx, bundle)
    assert {row["release_id"] for row in bundle.routing} == {"v3", None}
    assert {row["lob"] for row in bundle.routing if row["release_id"] == "v3"} == set(SNAPSHOT_LINES)
