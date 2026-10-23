"""The third review: the smoke-test path, for a batch of original documents on a 4-GPU pod."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.dataset_builder.policy_windows import (
    PolicyWindowPlan,
    TargetReport,
    multi_window_sections,
    window_target,
    with_inferred_pages,
)
from serving.policy_merge import PolicyWindow, merge_policy_windows

ROOT = Path(__file__).resolve().parents[1]


def _env(value, *pages):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# 1 - the smoke run brings its own label floor
def test_the_smoke_run_lowers_the_label_floor_for_its_own_steps(tmp_path):
    from orchestration.smoke_run import commands

    steps = commands(batch_dir=tmp_path, subset_dir=tmp_path, tenant="smoke2", version="v0.1",
                     check_out=tmp_path)
    for name in ("preflight", "finetune"):
        command = steps[name]
        assert command[command.index("--min-labels-per-type") + 1] == "1"


# 6 - a second smoke run does not reuse the first one's state
def test_a_smoke_run_refuses_a_tenant_that_already_holds_documents(client):
    from orchestration.smoke_run import SmokeError, assert_fresh

    assert_fresh(client, "smoke2", "v0.1", ["select", "import", "finetune"])          # nothing there yet
    client.write_json(paths.golden_label("policy", "policy_0001", "smoke2"), {})
    with pytest.raises(SmokeError, match="already holds 1 imported document"):
        assert_fresh(client, "smoke2", "v0.1", ["select", "import", "finetune"])
    # Carrying on a run that stopped part-way is the same run - from the import
    # too, as the smoke run's own message after a failed import says.
    assert_fresh(client, "smoke2", "v0.1", ["check", "preflight", "finetune"])
    assert_fresh(client, "smoke2", "v0.1", ["import", "ocr", "check", "preflight", "finetune"])


# 2 - the launcher carries the caller's GPU selection into the session
def _runner(tmp_path, env):
    fake = tmp_path / "bin"
    fake.mkdir()
    # No session exists yet (has-session fails); starting one succeeds.
    (fake / "tmux").write_text('#!/usr/bin/env bash\n[ "$1" = "has-session" ] && exit 1\nexit 0\n',
                               encoding="utf-8", newline="\n")
    (fake / "tmux").chmod(0o755)
    full = {**env, "PATH": f"{fake}:{env.get('PATH', '')}", "FIDEON_LOG_DIR": str(tmp_path / "logs")}
    done = subprocess.run(["bash", str(ROOT / "scripts" / "pod_run.sh"), "start", "job", "--", "true"],
                          env=full, capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip(f"bash/tmux stub unavailable here: {done.stderr[:120]}")
    return (tmp_path / "logs" / "job.run.sh").read_text(encoding="utf-8")


def test_the_launcher_exports_the_callers_gpu_selection(tmp_path):
    import os

    script = _runner(tmp_path, {**os.environ, "CUDA_VISIBLE_DEVICES": "2"})
    assert "export CUDA_VISIBLE_DEVICES=2" in script


def test_the_launcher_clears_a_gpu_selection_the_caller_does_not_have(tmp_path):
    import os

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    assert "unset CUDA_VISIBLE_DEVICES" in _runner(tmp_path, env)


def test_two_jobs_started_in_the_same_second_get_different_names(monkeypatch):
    from orchestration import detach

    first = detach.run_name("run_mineru")
    monkeypatch.setattr(detach.os, "getpid", lambda: 4242)
    assert detach.run_name("run_mineru") != first


# 3 - a document's shard does not depend on what is already done
def test_a_documents_shard_is_the_same_whatever_else_is_in_the_list():
    from data_pipeline.ocr.run_mineru import shard_of

    ids = [f"policy_{i:04d}" for i in range(1, 41)]
    shards = [shard_of(ids, f"{i}/4") for i in range(4)]
    assert sorted(sid for shard in shards for sid in shard) == ids          # each exactly once
    leftovers = shards[1][3:]                                                 # shard 1 restarted alone
    assert shard_of(leftovers, "1/4") == leftovers
    assert shard_of([s for s in ids if s not in shards[0][:5]], "2/4") == shards[2]


# 5 - the post-OCR check fails on OCR that did not work, added fields or not
def _ocr_report(ok, missing):
    from data_pipeline.ocr_check import OcrCheckReport, Row

    report = OcrCheckReport()
    report.rows = [Row("d", f"p{i}", "v", "1", "ok" if i < ok else "not_found", "", "original", "real", "homeowners")
                   for i in range(ok + missing)]
    return report


def test_originals_whose_values_are_not_on_the_page_fail_the_check():
    from data_pipeline.ocr_check import verdict

    passed, reason = verdict(_ocr_report(ok=10, missing=90))
    assert not passed and "OCR text is empty or misnumbered" in reason
    passed, reason = verdict(_ocr_report(ok=90, missing=10))
    assert passed and "no rule-added fields" in reason


# 4 - a fragment that carries identified rows of its own is taught and joined
VEHICLE = {"policy": {"policy_number": _env("PA-1", 1)}, "auto": {"vehicles": [{
    "vin": _env("1HGCM82633A004352", 6), "make": _env("Honda", 6),
    "coverages": [{"coverage_name": _env("Collision", 7), "premium": _env(281.0, 7)},
                  {"coverage_name": _env("Comprehensive", 7), "premium": _env(120.0, 7)}],
}]}}


def _plan(pages, index):
    return PolicyWindowPlan("lineblk", index, tuple(pages), single=False)


def test_a_coverage_table_on_the_page_after_its_vin_is_taught():
    report = TargetReport()
    second = window_target(VEHICLE, "commercial_auto", _plan([7, 8, 9], 1), report)
    (vehicle,) = second["auto"]["vehicles"]
    assert [c["coverage_name"]["raw"] for c in vehicle["coverages"]] == ["Collision", "Comprehensive"]
    assert "vin" not in vehicle and not report.orphaned


def test_a_fragment_with_nothing_identified_is_still_left_out():
    label = {"auto": {"vehicles": [{"vin": _env("1HGCM82633A004352", 6),
                                    "coverages": [{"coverage_name": _env("Collision", 6),
                                                   "premium": _env(1.0, 6, 7, 8)}]}]}}
    report = TargetReport()
    assert not window_target(label, "commercial_auto", _plan([7, 8, 9], 1), report).get("auto", {}).get("vehicles")
    assert report.orphaned


def test_the_merge_joins_the_fragment_to_the_only_vehicle_and_says_so():
    first = window_target(VEHICLE, "commercial_auto", _plan([4, 5, 6], 0))
    second = window_target(VEHICLE, "commercial_auto", _plan([7, 8, 9], 1))
    merged = merge_policy_windows([PolicyWindow("lineblk", [4, 5, 6], first),
                                   PolicyWindow("lineblk", [7, 8, 9], second)], lob="commercial_auto")
    (vehicle,) = merged.extraction["auto"]["vehicles"]
    assert vehicle["vin"]["raw"] == "1HGCM82633A004352" and len(vehicle["coverages"]) == 2
    assert "auto.vehicles:joined_without_identifier" in merged.review_flags


def test_a_fragment_is_not_joined_when_two_vehicles_could_own_it():
    def vehicle(vin):
        return {"vin": _env(vin, 1)}

    fragment = {"coverages": [{"coverage_name": _env("Collision", 2)}]}
    # The line named: no line is the common-model fallback, merged by its own rules.
    merged = merge_policy_windows([
        PolicyWindow("lineblk", [1], {"auto": {"vehicles": [vehicle("VIN-A"), vehicle("VIN-B")]}}),
        PolicyWindow("lineblk", [2], {"auto": {"vehicles": [fragment]}}),
    ], lob="commercial_auto")
    assert len(merged.extraction["auto"]["vehicles"]) == 3
    assert not merged.joined_unidentified


# 8 - a page is inferred only where the value could not otherwise be placed
def test_a_single_window_group_keeps_a_value_that_records_no_page():
    pages = ["Declarations: insured printed across cells", "p2", "p3", "p4", "p5", "p6",
             "Notice to Jane Rivera"]
    label = {"named_insured": {"primary_name": {"raw": "Jane Rivera", "parsed": "Jane Rivera", "page_ref": []}},
             "policy": {}, "carrier": {}}
    # A self-contained line: a common-model line's view requires every value to
    # cite a page (general liability composes the common model since 1.1.0).
    decl = PolicyWindowPlan("decl", 0, (1, 2, 3), single=True)
    sections = multi_window_sections("property", [decl])
    assert "named_insured" not in sections
    placed = with_inferred_pages(label, pages, sections=sections)
    assert placed is label
    assert window_target(placed, "property", decl)["named_insured"]["primary_name"]["raw"] == "Jane Rivera"


def test_a_multi_window_group_still_gets_its_pages_inferred():
    plans = [PolicyWindowPlan("lineblk", 0, (1, 2), single=False), PolicyWindowPlan("lineblk", 1, (3, 4), single=False)]
    label = {"auto": {"vehicles": [{"vin": {"raw": "1HGCM82633A004352", "parsed": "1HGCM82633A004352",
                                            "page_ref": []}}]}}
    pages = ["cover", "schedule", "vehicle 1HGCM82633A004352 listed", "end"]
    placed = with_inferred_pages(label, pages, sections=multi_window_sections("commercial_auto", plans))
    assert placed["auto"]["vehicles"][0]["vin"]["page_ref"] == [3]


# 9 - the fingerprint moves when a setting that shapes the data moves
def test_the_fingerprint_covers_the_training_settings_but_not_the_gpu_count(monkeypatch):
    from common import config
    from training import train

    base = {"lora": {"rank": 64}, "line_balance": {"max_repeat": 3},
            "batch": {"effective_batch_size": 8, "per_device_train_batch_size": 1,
                      "gradient_accumulation_steps": 8},
            "distributed": {"gpus": "auto"}}
    current = {"cfg": base}
    read: list[str] = []
    monkeypatch.setattr(config, "training_config", lambda name: read.append(name) or current["cfg"])
    first = train._settings_hash()
    assert read == ["unified"]                                     # the adapter's training config
    current["cfg"] = {**base, "distributed": {"gpus": 4},
                      "batch": {**base["batch"], "gradient_accumulation_steps": 2}}
    assert train._settings_hash() == first                         # another pod, the same run
    current["cfg"] = {**base, "line_balance": {"max_repeat": 1}}
    assert train._settings_hash() != first                         # a different training set


# 10 - the flat-folder refusal names what works, and no flag that does not exist
def test_the_flat_folder_message_names_the_folder_layout_not_a_flag(tmp_path, client):
    from orchestration.pipeline_dag import PipelineError, stage_ingestion

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    ctx = SimpleNamespace(skip_ingest=False, input_dir=tmp_path, raw=client, tenant_id=None,
                          doc_types=["acord", "policy", "lossrun"])
    with pytest.raises(PipelineError) as raised:
        stage_ingestion(ctx)
    message = str(raised.value)
    assert f"({tmp_path.name}/policy/...)" in message
    # --scope and --doc-types are gone from the CLI (decision 2026-10-22).
    assert "--scope" not in message and "--doc-types" not in message


def test_the_window_notes_report_orphaned_fragments():
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, _policy_window_rows

    label = {"policy": {"policy_number": _env("PA-1", 1)}, "carrier": {}, "named_insured": {},
             "auto": {"vehicles": [{"vin": _env("1HGCM82633A004352", 1), "make": _env("Honda", 1, 9)}]}}
    document = SourceDocument(source_id="s1", doc_type="policy", golden_label=label,
                              ocr_pages=[f"page {i}" for i in range(1, 13)],
                              image_paths=[f"p{i}.png" for i in range(1, 13)], lob="commercial_auto")
    details: list[str] = []
    _policy_window_rows(document, "train", "image_only", None, details)
    assert any("orphaned" in d for d in details), json.dumps(details)[:300]
