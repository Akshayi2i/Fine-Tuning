"""The review of 4dd059f..4025f6f (2026-11-07): each finding, fixed and held here."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend

PICTURE = '<img src="data:image/jpeg;base64,' + "A" * 3000 + '"/>'
FORM = "Hazardous Materials listing. Identifying Symbol. Total Amount. Location of the materials."


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# 1. the template mark compares pages without pictures on both sides ---------------


def test_a_training_page_stored_with_pictures_matches_the_same_page_frozen_without(client):
    from evaluation.freeze_eval_set import _train_page_keys, template_seen

    client.write_text(f"{paths.processed_dir('policy', 'policy_0001', None)}/page_1.md", f"{FORM}\n{PICTURE}")
    manifest = {"source_ids_by_split": {"train": ["policy_0001"]}, "doc_types": ["policy"]}
    train_keys = _train_page_keys(client, manifest, None)
    # The frozen copy is stored without pictures (freeze_eval_set cleans it before copying).
    assert template_seen({1: FORM}, train_keys)["template_seen_in_train"] is True


# 2. a run stopped by a signal is STOPPED, not FINISHED with exit -1 ----------------


def test_the_status_tool_reports_a_run_stopped_by_a_signal(tmp_path):
    from orchestration.status import read_status, render

    (tmp_path / "finetune-v4.log").write_text("working\n=== finetune-v4 stopped by signal HUP ===\n", encoding="utf-8")
    (tmp_path / "finetune-v4.exit").write_text("signal HUP\n", encoding="utf-8")
    status = read_status("finetune-v4", tmp_path)
    assert (status.state, status.stopped_by, status.exit_code) == ("STOPPED", "HUP", None)
    assert "STOPPED by signal HUP" in render(status) and "--from-stage" in render(status)


def test_pod_run_list_says_stopped_for_a_signal():
    script = (Path(__file__).resolve().parent.parent / "scripts" / "pod_run.sh").read_text(encoding="utf-8")
    listing = script[script.index("cmd_list() {"):]
    assert "STOPPED by %s - resume it with --from-stage" in listing[:listing.index("\n}\n")]


# 3. --accept-set-aside reaches preflight and the smoke run --------------------------


def test_preflight_takes_accept_set_aside_on_its_command_line(monkeypatch, tmp_path, client):
    from orchestration import detach, preflight
    from orchestration.run import build_context

    seen = {}

    def fake_run(args):
        seen["accepted"] = build_context(args, client=client, dry_run=True).accept_set_aside
        return [preflight.Check("corpus", "PASS", "ok")], {}

    monkeypatch.setattr(detach, "should_detach", lambda: False)
    monkeypatch.setattr(preflight, "run_preflight", fake_run)
    code = preflight.main(["--accept-set-aside", "policy_0048,policy_0099", "--out", str(tmp_path / "p.json")])
    assert code == 0 and seen["accepted"] == frozenset({"policy_0048", "policy_0099"})


def test_the_smoke_run_passes_accept_set_aside_to_preflight_and_finetune(tmp_path):
    from orchestration.smoke_run import commands

    built = commands(batch_dir=tmp_path, subset_dir=tmp_path, tenant="smoke", version="v0", check_out=tmp_path,
                     accept_set_aside="policy_0048")
    for step in ("preflight", "finetune"):
        assert built[step][-2:] == ["--accept-set-aside", "policy_0048"]
    plain = commands(batch_dir=tmp_path, subset_dir=tmp_path, tenant="smoke", version="v0", check_out=tmp_path)
    assert "--accept-set-aside" not in plain["finetune"]


# 4. a signal to the runner stops the run at once and never overwrites a real exit ----


def _git_bash() -> str | None:
    found = shutil.which("bash")
    # Windows' System32 bash is WSL, a different machine's paths; the pod and Git Bash are what this runs on.
    return None if not found or "system32" in found.lower() else found


@pytest.mark.skipif(_git_bash() is None, reason="needs bash")
def test_a_signal_to_the_runner_stops_the_run_at_once(tmp_path):
    """Bash runs a trap only between commands: with the job in the foreground, a TERM to the runner waited for the
    whole run, then wrote 'signal TERM' over its real exit (a job that would exit 0 after 12 s: 'signal TERM' at 13 s,
    on this laptop, 2026-11-07)."""
    (tmp_path / "repo" / "scripts").mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    (tmp_path / "logs").mkdir()
    shutil.copy(Path(__file__).resolve().parent.parent / "scripts" / "pod_run.sh", tmp_path / "repo" / "scripts")
    (tmp_path / "bin" / "tmux").write_bytes(b'#!/usr/bin/env bash\ncase "$1" in has-session) exit 1 ;; *) exit 0 ;; esac\n')
    (tmp_path / "drive.sh").write_bytes(b"""#!/usr/bin/env bash
S="$(pwd)"; export PATH="$S/bin:$PATH" FIDEON_LOG_DIR="$S/logs"
chmod +x "$S/bin/tmux"
bash "$S/repo/scripts/pod_run.sh" start sig -- bash -c "sleep 6; touch $S/late" >/dev/null || exit 9
set -m
bash "$S/logs/sig.run.sh" </dev/null >"$S/pane.out" 2>&1 &
sleep 2
kill -s TERM $!
for i in $(seq 1 30); do [ -f "$S/logs/sig.exit" ] && break; sleep 0.2; done
""")
    started = time.monotonic()
    subprocess.run([_git_bash(), "drive.sh"], cwd=tmp_path, check=True, timeout=60)
    took = time.monotonic() - started
    assert (tmp_path / "logs" / "sig.exit").read_text().strip() == "signal TERM"
    assert took < 5.5, f"the runner waited for the job ({took:.1f}s)"
    assert "stopped by signal TERM" in (tmp_path / "logs" / "sig.log").read_text()
    time.sleep(6)
    assert not (tmp_path / "late").exists(), "the job kept running after the runner stopped"


# 5. near copies by image leave out pages with no text to tell them apart ------------


def _documents(client, pages_a, pages_b):
    docs = []
    for name, pages in (("a", pages_a), ("b", pages_b)):
        keys, texts = [], {}
        for number, (image, text) in enumerate(pages, start=1):
            key = f"g/{name}/page_{number}.png"
            client.write_bytes(key, image)
            keys.append(key)
            texts[number] = text
        docs.append(SimpleNamespace(source_id=f"policy_{name}", image_keys=keys, page_texts=texts))
    return docs


def test_blank_pages_do_not_make_two_policies_near_copies(client):
    from evaluation.metrics.near_copy import near_copy_pairs, page_image_keys

    blank = (b"blank", "")
    documents = _documents(client, [(b"a declarations", "A " + FORM), blank, blank],
                           [(b"b declarations", "B " + FORM), blank, blank])
    keys = page_image_keys(client, documents)
    assert all(len(pages) == 1 for pages in keys.values())          # the blank pages are left out
    assert near_copy_pairs(documents, share=0.6, page_keys=keys) == []


# 6. an unreadable page image is named, not fatal -------------------------------------


def test_the_image_check_names_an_unreadable_page_and_carries_on(client):
    from data_pipeline.data_health import same_image_different_text

    client.write_bytes("p/a1.png", b"form page")
    client.write_bytes("p/b1.png", b"form page")
    documents = [
        SimpleNamespace(source_id="policy_0065", image_paths=["p/a1.png", "p/gone.png"], ocr_pages=["Total", "x"]),
        SimpleNamespace(source_id="policy_0066", image_paths=["p/b1.png"], ocr_pages=["Tota1"]),
    ]
    groups, unread = same_image_different_text(client, documents, workers=2)
    assert unread == ["p/gone.png"]
    assert [(p["source_id"], p["page"]) for p in groups[0]["pages"]] == [("policy_0065", 1), ("policy_0066", 1)]


def test_near_copy_pairing_falls_back_to_text_when_a_page_image_is_missing(client):
    from evaluation.metrics.near_copy import page_image_keys

    documents = [SimpleNamespace(source_id="policy_a", image_keys=["g/missing.png"], page_texts={1: FORM})]
    assert page_image_keys(client, documents) is None


# 7. a stopped build says the report was written --------------------------------------
# (asserted in tests/test_data_health.py, test_the_build_stops_on_a_set_aside_document_...)


# 8. no interpreter is chosen, or announced, for a job that is not moved -----------


def test_a_job_that_stays_in_the_foreground_chooses_no_interpreter(monkeypatch):
    from orchestration import detach

    def chosen():
        raise AssertionError("detached_python was called for a job that is not detached")

    monkeypatch.setattr(detach, "should_detach", lambda: False)
    monkeypatch.setattr(detach, "detached_python", chosen)
    assert detach.detach_module_if_needed("data_pipeline.data_health", []) is False
    assert detach.detach_script_if_needed("scripts/x.py", [], "x") is False


# 9-10. one page-image hash, read from disk when the gate already has the page ------


def test_page_images_on_disk_are_hashed_without_asking_storage(tmp_path):
    from evaluation.metrics.near_copy import page_image_digests, page_image_keys

    (tmp_path / "1.png").write_bytes(b"page one")
    local = {"g/a/page_1.png": str(tmp_path / "1.png"), "g/b/page_1.png": str(tmp_path / "1.png")}

    class NoStorage:
        def md5(self, key):
            raise AssertionError(f"storage was asked about {key}")

        read_bytes = md5

    documents = [SimpleNamespace(source_id="a", image_keys=["g/a/page_1.png"], page_texts={1: FORM}),
                 SimpleNamespace(source_id="b", image_keys=["g/b/page_1.png"], page_texts={1: FORM})]
    keys = page_image_keys(NoStorage(), documents, local=local)
    assert keys["a"] == keys["b"] and len(keys["a"]) == 1
    assert page_image_keys(None, documents, local=local) == keys
    assert page_image_digests(None, ["g/a/page_1.png", "not/on/disk.png"], local=local)[1] is None


def test_the_data_health_image_check_uses_the_near_copy_hash():
    from data_pipeline import data_health

    source = Path(data_health.__file__).read_text(encoding="utf-8")
    assert "page_image_digests" in source and "hashlib" not in source


def test_the_gate_hands_its_downloaded_pages_to_the_near_copy_check():
    from evaluation import golden_eval

    triples = golden_eval.GoldenTriples()
    assert triples.local_images == {}
    source = Path(golden_eval.__file__).read_text(encoding="utf-8")
    assert "triples.local_images = dict(local)" in source
    assert 'local=getattr(triples, "local_images", None)' in source
