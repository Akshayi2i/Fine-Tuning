"""Selection and post-training generation cost (review #27, #28).

#27: the finalists' second reading covers only the gate's repeat mode
(image_only), and a pass with fewer candidates than GPUs is sharded across
every GPU - with exactly the calls, and so the scores, it makes unsharded.

#28: calibration fits from the selected finalist's saved answers, generating
only the rows they do not cover, and only when they were made by the same
checkpoint with the same generation settings.

Synthetic rows and a deterministic backend: no GPU, no weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation import checkpoint_eval as C
from inference_core.model_runner import EchoBackend, load_model

#: A self-contained line: its report scores the label as it is.
LINE = "cyber"
#: Each document's page text and the policy number it prints.
DOCUMENTS = {"d1": ("HO-1", "Declarations Policy HO-1 Effective 01/01/2026"),
             "d2": ("HO-2", "Declarations Policy HO-2 Effective 01/01/2026"),
             "d3": ("HO-3", "Declarations Policy HO-3 Effective 01/01/2026")}
MODES = ("ocr_plus_image", "noisy_ocr_image", "image_only")
VAL = "corpus/val.jsonl"


def env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


def answer(number, effective="01/01/2026"):
    return {"policy": {"policy_number": env(number), "effective_date": env(effective)}}


def row(source, mode, *, synthetic=False, half="calibration"):
    marker = "<page 1 of 1>" if mode == "image_only" else f"<page 1 of 1>\n\n{DOCUMENTS[source][1]}"
    return {"source_id": source, "doc_type": "policy", "lob": [LINE], "modality_mode": mode,
            "sections": "decl", "window_pages": [1], "window_index": 0, "synthetic": synthetic,
            "val_half": half,
            "messages": [{"role": "user", "content": [{"type": "image", "image": f"{source}/page_1.png"},
                                                      {"type": "text", "text": marker}]},
                         {"role": "assistant", "content": json.dumps(answer(DOCUMENTS[source][0]))}]}


def rows(modes=MODES):
    return [row(source, mode, half="calibration" if i % 2 else "threshold")
            for i, source in enumerate(DOCUMENTS) for mode in modes]


def reads(source, mode, adapter):
    """What the stub model answers: right, except checkpoint-20 invents d2's
    number when it reads d2 from the image alone."""
    if adapter and adapter.endswith("checkpoint-20") and source == "d2" and mode == "image_only":
        return answer("ZZ-404-9")
    return answer(DOCUMENTS[source][0])


class RowBackend(EchoBackend):
    """Answers each row by its document, its mode and the adapter; records each call."""

    def __init__(self):
        super().__init__()
        self.batches: list[tuple] = []

    def generate(self, messages, config, adapter=None):
        parts = [p for m in messages if isinstance(m.get("content"), list) for p in m["content"]]
        source = next(p["image"] for p in parts if p.get("type") == "image").split("/")[0]
        texts = [p.get("text") or "" for p in parts if p.get("type") == "text"]
        mode = "ocr_plus_image" if any("\n\n" in t for t in texts) else "image_only"
        self.response = json.dumps(reads(source, mode, adapter), separators=(",", ":"))
        return super().generate(messages, config, adapter)

    def generate_batch(self, messages_list, configs, adapter=None):
        # Each call: its adapter and its prompts, in the order sent.
        self.batches.append((adapter, tuple(json.dumps(msgs, sort_keys=True) for msgs in messages_list)))
        return super().generate_batch(messages_list, configs, adapter)


@pytest.fixture
def client():
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")


@pytest.fixture
def small_calls(monkeypatch):
    """Two rows a call, so a pass of a few rows makes several calls to shard."""
    from evaluation import validation_generation

    monkeypatch.setattr(validation_generation, "configured_batch_rows", lambda: 2)


def model_for(client, backend=None):
    return load_model("base", client, backend_impl=backend or RowBackend())


def in_process_workers(monkeypatch, client, model):
    """``_spawn`` runs the worker's own entry point here, on ``model``."""
    import inference_core.model_runner as model_runner
    from evaluation import checkpoint_score_worker as W

    monkeypatch.setenv("FIDEON_NO_DETACH", "1")
    monkeypatch.setattr("orchestration.detach.detach_module_if_needed", lambda *a, **k: False)
    monkeypatch.setattr("artifact_registry.blob_client.BlobClient", lambda: client)
    monkeypatch.setattr(model_runner, "load_model", lambda *a, **k: model)
    started = []

    def spawn(command, env, log_path):
        started.append((command, env))
        log_path.write_text("", encoding="utf-8")
        code = W.main(command[3:])
        return SimpleNamespace(poll=lambda: code, returncode=code)

    monkeypatch.setattr(C, "_spawn", spawn)
    return started


# --------------------------------------------------------------------------
# #27 - the repeat reads the gate's repeat mode only
# --------------------------------------------------------------------------

def test_the_second_reading_covers_only_the_repeat_mode(client, small_calls):
    model = model_for(client)
    scorer = C.generation_scorer(rows(), model, repeat=True, repeat_modes=["image_only"])
    metrics = scorer("/ck/checkpoint-10")
    first, = scorer.passes
    assert first["generated"] == 9 and first["repeat"]["generated"] == 3          # 3 image_only rows, not 9
    assert metrics["run_to_run_agreement"] == 1.0 and metrics["document_run_to_run_agreement"] == 1.0
    assert C.generation_cost(scorer)["rows"] == 12


def test_the_repeat_mode_follows_the_gate_and_falls_back_when_the_pass_does_not_read_it():
    from orchestration.pipeline_dag import _selection_settings

    fast = _selection_settings(SimpleNamespace())
    assert fast["selection_document_modes"] == ["ocr_plus_image", "image_only"]
    assert fast["selection_repeat_modes"] == ["image_only"]


@pytest.mark.format_profile("standard")
def test_a_pass_that_does_not_read_the_repeat_mode_reads_all_its_rows_twice():
    from orchestration.pipeline_dag import _selection_settings

    standard = _selection_settings(SimpleNamespace())
    assert standard["selection_document_modes"] == ["ocr_plus_image"]
    assert standard["selection_repeat_modes"] == []


# --------------------------------------------------------------------------
# #27 - the finalists' pass sharded across every GPU
# --------------------------------------------------------------------------

def test_fewer_candidates_than_gpus_are_sharded_and_an_even_count_is_dealt(tmp_path):
    scorer = C.ParallelScorer(val_path=VAL, images_root=None, sample_rows=0, gpus=["0", "1", "2", "3"],
                              work_dir=tmp_path, rows=rows())
    assert scorer.sharded(["/ck/checkpoint-10", "/ck/checkpoint-20"])                  # 2 on 4
    assert scorer.sharded([f"/ck/checkpoint-{s}" for s in (10, 20, 30, 40, 50)])     # 5 on 4
    assert not scorer.sharded([f"/ck/checkpoint-{s}" for s in (10, 20, 30, 40)])     # 4 on 4: dealt
    assert not C.ParallelScorer(val_path=VAL, images_root=None, sample_rows=0, gpus=["0", "1", "2", "3"],
                                work_dir=tmp_path).sharded(["/ck/checkpoint-10"])    # no rows: dealt


def test_a_sharded_pass_makes_the_same_calls_and_scores_as_the_pass_unsharded(client, small_calls, tmp_path,
                                                                             monkeypatch):
    client.write_text(VAL, "\n".join(json.dumps(r) for r in rows()))
    finalists = ["/ck/checkpoint-10", "/ck/checkpoint-20"]

    alone = RowBackend()
    single = C.vllm_scorer(client=client, val_path=VAL, model=model_for(client, alone), repeat=True,
                           repeat_modes=["image_only"], save_dir=tmp_path / "single")
    expected = {checkpoint: single(checkpoint) for checkpoint in finalists}

    sharded_backend = RowBackend()
    started = in_process_workers(monkeypatch, client, model_for(client, sharded_backend))
    parallel = C.ParallelScorer(val_path=VAL, images_root=None, sample_rows=0, gpus=["0", "1", "2"],
                                work_dir=tmp_path / "work", repeat=True, repeat_modes=["image_only"],
                                finalists=True, save_dir=tmp_path / "sharded", rows=rows(), poll_seconds=0)
    got = parallel.score_many(finalists)

    assert len(started) == 3                                                   # every GPU, one worker each
    assert {env["CUDA_VISIBLE_DEVICES"] for _c, env in started} == {"0", "1", "2"}
    assert all(command.count("--checkpoint") == 2 for command, _env in started)   # each holds both
    assert got == expected                                                     # identical scores
    assert expected["/ck/checkpoint-20"]["gate_mirror_accuracy"] < 1.0         # the invention counted
    # Call for call: the same rows in the same calls, shared out between the GPUs.
    assert sorted(sharded_backend.batches) == sorted(alone.batches)
    # And the answers saved for calibration are the same.
    for checkpoint in finalists:
        a = C.saved_pass_paths(tmp_path / "single", checkpoint)["generations"].read_text().splitlines()[1:]
        b = Path(parallel.saved[checkpoint]["generations"]).read_text().splitlines()[1:]
        assert a == b
    shares = [p for p in parallel.passes if p["checkpoint"] == finalists[0]]
    assert len(shares) == 3 and sum(p["generated"] for p in shares) == 9
    assert sum(p.get("repeat", {}).get("generated", 0) for p in shares) == 3


def test_a_shard_that_died_fails_its_candidates_with_its_log(client, small_calls, tmp_path, monkeypatch):
    client.write_text(VAL, "\n".join(json.dumps(r) for r in rows()))
    in_process_workers(monkeypatch, client, model_for(client))
    real = C._spawn

    def spawn(command, env, log_path):
        if env["CUDA_VISIBLE_DEVICES"] == "1":
            log_path.write_text("Traceback\nCUDA out of memory\n", encoding="utf-8")
            return SimpleNamespace(poll=lambda: 1, returncode=1)
        return real(command, env, log_path)

    monkeypatch.setattr(C, "_spawn", spawn)
    parallel = C.ParallelScorer(val_path=VAL, images_root=None, sample_rows=0, gpus=["0", "1", "2"],
                                work_dir=tmp_path, rows=rows(), poll_seconds=0)
    outcomes = parallel.score_many(["/ck/checkpoint-10"])
    failed = outcomes["/ck/checkpoint-10"]
    assert isinstance(failed, C.CheckpointEvalError) and "CUDA out of memory" in str(failed)


def test_generate_validation_shards_partition_the_calls(client, small_calls):
    from evaluation.validation_generation import generate_validation, reading_key

    model = model_for(client)
    whole = generate_validation(rows(), model)
    parts = [generate_validation(rows(), model, shard=(k, 3)) for k in range(3)]
    keys = [reading_key(g.row) for part in parts for g in part]
    assert sorted(keys) == sorted(reading_key(g.row) for g in whole) and len(keys) == len(set(keys))
    with pytest.raises(ValueError, match="shard"):
        generate_validation(rows(), model, shard=(3, 3))


def test_the_halves_of_a_window_are_two_readings_merged_in_page_order(client):
    """A window whose answer overran its reservation is built as serving's
    halves (review #38): both keep the window's index, ``window_half`` tells
    them apart - in every key that pairs, saves or shards a reading."""
    from evaluation.validation_generation import (
        ValidationGeneration,
        merged_windows,
        reading_key,
        restored_generation,
    )

    def half(name, pages, number):
        r = row("d1", "ocr_plus_image")
        r.update(window_half=name, window_pages=pages)
        r["messages"][-1]["content"] = json.dumps({"policy": {"policy_number": env(number, pages[0])}})
        return r

    halves = [half("1", [2], "HO-1"), half("0", [1], "HO-1")]
    assert reading_key(halves[0]) != reading_key(halves[1])
    assert reading_key(row("d1", "ocr_plus_image")) not in {reading_key(h) for h in halves}
    generations = [ValidationGeneration(row=h, golden=json.loads(h["messages"][-1]["content"]),
                                        extraction=json.loads(h["messages"][-1]["content"])) for h in halves]
    document, = merged_windows(generations)
    assert [g.row["window_half"] for g in document.members] == ["0", "1"]          # page order
    # Shardable and restorable: two readings, each answer back on its own half.
    scorer = C.ParallelScorer(val_path=VAL, images_root=None, sample_rows=0, gpus=["0", "1"],
                              work_dir=Path("."), rows=halves + [row("d2", "ocr_plus_image")])
    assert scorer.sharded(["/ck/checkpoint-10"])
    from evaluation.validation_generation import generation_record

    record = generation_record(generations[0])
    assert restored_generation(halves[0], record).row["window_half"] == "1"


# --------------------------------------------------------------------------
# #28 - calibration fits from the selected finalist's saved answers
# --------------------------------------------------------------------------

def _calibration_ctx(client, loader, selected="/ck/checkpoint-10"):
    from orchestration.pipeline_dag import StageContext

    client.write_text(paths.corpus_scope_eval_split("v1", "val", "policy"),
                      "".join(json.dumps(r) + "\n" for r in rows()))
    client.write_json(paths.checkpoint_selection("v1", scope="policy"), {"selected": selected})
    return StageContext(client=client, controller=None, out_version="v1",  # type: ignore[arg-type]
                        formats=["bf16"], serving_model_loader=loader)


def _save_finalist(client, tmp_path, checkpoint="/ck/checkpoint-10", modes=("ocr_plus_image", "image_only")):
    """The finalists' pass of ``checkpoint`` saved as selection saves it, and the
    selection record pointing at it."""
    scorer = C.generation_scorer(rows(modes), model_for(client), finalists=True, save_dir=tmp_path)
    scorer(checkpoint)
    key = paths.checkpoint_selection("v1", scope="policy")
    client.write_json(key, {**client.read_json(key), "finalist_pass": {
        "saved": True, "checkpoint": checkpoint, "local": scorer.saved[checkpoint]["generations"]}})
    return Path(scorer.saved[checkpoint]["generations"])


def _loader(generated):
    def loader(ctx, fmt):
        backend = RowBackend()
        model = load_model("base", ctx.client, backend_impl=backend)
        original = backend.generate

        def generate(messages, config, adapter=None):
            generated.append(messages)
            return original(messages, config, adapter)

        backend.generate = generate
        return model
    return loader


def _pairs(by_mode):
    """Each mode's halves (calibration.population.samples_by_mode), as comparable rows -
    the span details calibration learns from included."""
    return {(mode, half): sorted((f.field_path, str(f.value), f.r_written, f.top2_margin, ok) for f, ok in pairs)
            for mode, halves in by_mode.items() for half, pairs in halves.items()}


def test_calibration_reuses_the_finalists_answers_and_generates_only_the_other_modes(client, tmp_path):
    from orchestration.pipeline_dag import _collect_calibration_samples

    generated: list = []
    ctx = _calibration_ctx(client, _loader(generated))
    _save_finalist(client, tmp_path)
    samples, how = _collect_calibration_samples(ctx)
    assert how["bf16"]["source"] == "finalist_answers+generated"
    assert (how["bf16"]["reused_rows"], how["bf16"]["generated_rows"]) == (6, 3)
    assert len(generated) == 3                                         # the noisy_ocr_image rows only
    # The samples are those a fresh generation of the whole split gives.
    fresh: list = []
    client.write_json(paths.checkpoint_selection("v1", scope="policy"), {"selected": "/ck/checkpoint-10"})
    again, how_again = _collect_calibration_samples(_calibration_ctx(client, _loader(fresh)))
    assert how_again["bf16"]["source"] == "generated" and len(fresh) == 9
    assert _pairs(samples["bf16"]) == _pairs(again["bf16"])
    # By mode, and the saved spans' details survive: r_written is learned from them.
    assert set(samples["bf16"]) == {"ocr_plus_image", "noisy_ocr_image", "image_only"}
    reused = [f for f, _ok in samples["bf16"]["image_only"]["calibration"] + samples["bf16"]["image_only"]["threshold"]]
    assert reused and all(f.r_written is True for f in reused)


def test_answers_made_by_another_checkpoint_are_not_reused(client, tmp_path):
    from orchestration.pipeline_dag import _collect_calibration_samples

    generated: list = []
    ctx = _calibration_ctx(client, _loader(generated), selected="/ck/checkpoint-20")
    _save_finalist(client, tmp_path, checkpoint="/ck/checkpoint-10")
    _samples, how = _collect_calibration_samples(ctx)
    assert how["bf16"]["source"] == "generated" and len(generated) == 9
    assert "made by /ck/checkpoint-10" in how["bf16"]["why"]


def test_answers_made_with_other_generation_settings_are_not_reused(client, tmp_path):
    from orchestration.pipeline_dag import _collect_calibration_samples

    generated: list = []
    ctx = _calibration_ctx(client, _loader(generated))
    saved = _save_finalist(client, tmp_path)
    lines = saved.read_text().splitlines()
    header = json.loads(lines[0])
    header["header"]["settings"]["batch_rows"] = 7
    saved.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    _samples, how = _collect_calibration_samples(ctx)
    assert how["bf16"]["source"] == "generated" and "other settings" in how["bf16"]["why"]


def test_answers_to_another_split_are_not_reused(client, tmp_path):
    from orchestration.pipeline_dag import _collect_calibration_samples

    generated: list = []
    ctx = _calibration_ctx(client, _loader(generated))
    _save_finalist(client, tmp_path)
    changed = rows()
    changed[0]["messages"][-1]["content"] = json.dumps(answer("HO-1", effective="02/02/2026"))
    client.write_text(paths.corpus_scope_eval_split("v1", "val", "policy"),
                      "".join(json.dumps(r) + "\n" for r in changed))
    _samples, how = _collect_calibration_samples(ctx)
    assert how["bf16"]["source"] == "generated" and "another validation split" in how["bf16"]["why"]


def test_reuse_can_be_turned_off_and_nothing_saved_means_generating(client, tmp_path, monkeypatch):
    from orchestration import pipeline_dag

    generated: list = []
    ctx = _calibration_ctx(client, _loader(generated))
    _samples, how = pipeline_dag._collect_calibration_samples(ctx)
    assert how["bf16"]["source"] == "generated" and "no finalist answers" in how["bf16"]["why"]
    _save_finalist(client, tmp_path)
    settings = pipeline_dag._selection_settings(ctx)
    monkeypatch.setattr(pipeline_dag, "_selection_settings",
                        lambda c: {**settings, "calibration_reuses_selection": False})
    _samples, how = pipeline_dag._collect_calibration_samples(ctx)
    assert how["bf16"]["source"] == "generated" and "off" in how["bf16"]["why"]


def test_the_remaining_rows_are_sharded_across_gpus_on_a_multi_gpu_pod(client, small_calls, tmp_path,
                                                                     monkeypatch):
    """Off the test loader, several GPUs: the rows no saved answer covers are
    generated by the workers, the selected adapter on the base."""
    from orchestration import pipeline_dag

    checkpoint = tmp_path / "run" / "checkpoint-10"        # a checkpoint directory, as selection names it
    checkpoint.mkdir(parents=True)
    ctx = _calibration_ctx(client, None, selected=str(checkpoint))
    _save_finalist(client, tmp_path / "saved", checkpoint=str(checkpoint))
    monkeypatch.setattr(pipeline_dag, "_calibration_gpus", lambda: ["0", "1"])
    monkeypatch.setattr(paths, "staging_root", lambda: str(tmp_path / "staging"))
    import training.stage_data as stage_data

    monkeypatch.setattr(stage_data, "localize_rows", lambda rows_, client_, root: rows_)
    started = in_process_workers(monkeypatch, client, model_for(client))
    client.write_text(paths.corpus_scope_eval_split("v1", "val", "policy"),
                      "".join(json.dumps(r) + "\n" for r in rows()))
    samples, how = pipeline_dag._collect_calibration_samples(ctx)
    assert how["bf16"]["source"] == "finalist_answers+generated" and how["bf16"]["gpus"] == 2
    assert len(started) == 2 and all("--only-keys" in command for command, _env in started)
    assert any(rows for halves in samples["bf16"].values() for rows in halves.values())
