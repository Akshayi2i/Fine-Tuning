"""The inputs checkpoint selection and calibration never received.

``ctx.checkpoints`` and ``ctx.calibration_samples`` were read by their stages and
filled in by nothing, so on every real run checkpoint selection skipped and every
release shipped uncalibrated. These run the producers against a stub backend and
real corpus rows, so what they emit is what the stages read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.checkpoint_eval import discover_checkpoints, generation_scorer, select_best
from evaluation.validation_generation import (
    calibration_samples,
    generate_validation,
    score_generations,
    split_prompt,
)
from inference_core.input_builder import build_training_row
from inference_core.model_runner import EchoBackend, load_model

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "golden" / "policy_0001.golden.json").read_text(encoding="utf-8"))
OCR = (FIXTURES / "ocr" / "policy_0001_page_1.md").read_text(encoding="utf-8")
#: The line GOLDEN is valid for: a self-contained one. With no line it would be
#: read against the fallback, which composes common model 1.1.0: GOLDEN is
#: the old shape there, and a bare value is bookkeeping, never a field to calibrate.
LOB = "property"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _val_rows(n: int = 4) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        row = build_training_row(
            "policy", f"policy_{i:04d}", [f"processed/default/policy/policy_{i:04d}/page_1.png"],
            [OCR], "ocr_plus_image", json.dumps(GOLDEN), lob=LOB, split="val",
        )
        row["val_half"] = "calibration" if i % 2 else "threshold"
        rows.append(row)
    return rows


def _model(client, response: dict | str = GOLDEN):
    text = response if isinstance(response, str) else json.dumps(response, separators=(",", ":"))
    return load_model("base", client, backend_impl=EchoBackend(text))


def test_the_prompt_is_the_row_without_its_golden_answer():
    messages, golden = split_prompt(_val_rows(1)[0])
    assert golden == GOLDEN
    assert messages[-1]["role"] != "assistant"


def test_calibration_samples_come_from_both_halves_and_are_labelled(client):
    halves = calibration_samples(generate_validation(_val_rows(), _model(client)))

    assert halves["calibration"] and halves["threshold"]
    features, correct = halves["calibration"][0]
    assert features.is_usable
    assert all(correct for _, correct in halves["calibration"]), "an exact echo scored wrong"


def test_a_wrong_answer_is_labelled_wrong(client):
    wrong = {**GOLDEN, "policy_number": "NOT-THE-NUMBER"}
    halves = calibration_samples(generate_validation(_val_rows(), _model(client, wrong)))
    labels = {f.field_path: c for f, c in halves["calibration"]}
    assert labels["policy_number"] is False


def test_a_failed_generation_is_scored_not_dropped(client):
    generations = generate_validation(_val_rows(), _model(client, "not json"))
    assert all(g.error for g in generations)
    metrics = score_generations(generations)
    assert metrics["field_normalized_match"] < 0.5


def test_checkpoint_selection_scores_with_the_gate_metric(client):
    """Each checkpoint is passed as the adapter; the score is the gate's own."""
    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    model = load_model("base", client, backend_impl=backend)
    scorer = generation_scorer(_val_rows(), model)

    report = select_best(["/s/checkpoint-10", "/s/checkpoint-20"], scorer)

    assert report.selected == "/s/checkpoint-20"   # tie breaks to the later step
    assert {c["adapter"] for c in backend.calls} == {"/s/checkpoint-10", "/s/checkpoint-20"}
    assert report.scores[0].metrics["field_normalized_match"] > 0.9


def test_checkpoints_are_discovered_under_the_versioned_run_directory(tmp_path):
    run = tmp_path / "v0-20260917-101500"
    for step in (10, 20, 30):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    (run / "checkpoint-30" / "trainer_state.json").write_text(
        json.dumps({"best_model_checkpoint": "/elsewhere/v0-20260917-101500/checkpoint-20"}),
        encoding="utf-8",
    )

    checkpoints, best = discover_checkpoints(str(tmp_path))

    assert [Path(c).name for c in checkpoints] == ["checkpoint-10", "checkpoint-20", "checkpoint-30"]
    assert best is not None and Path(best).name == "checkpoint-20"
    assert discover_checkpoints(str(tmp_path / "missing")) == ([], None)


def test_calibrate_collects_its_own_samples_per_format(client):
    """collect_calibration_samples generates with each staged format, so the
    stage has evidence without an operator hand-building feature vectors."""
    from orchestration.pipeline_dag import StageContext, collect_calibration_samples

    # The policy adapter's own validation view, where the corpus view writes it.
    client.write_text(
        paths.corpus_scope_eval_split("v1", "val", "policy"),
        "".join(json.dumps(r) + "\n" for r in _val_rows()),
    )
    loaded: list[str] = []

    def loader(_ctx, fmt):
        loaded.append(fmt)
        return _model(client)

    ctx = StageContext(
        client=client, controller=None, out_version="v1",  # type: ignore[arg-type]
        formats=["bf16"], serving_model_loader=loader,
    )
    samples = collect_calibration_samples(ctx)

    assert loaded == ["bf16"]
    # By reading mode (calibration.population.samples_by_mode); these rows are ocr_plus_image.
    assert samples["bf16"]["ocr_plus_image"]["calibration"] and samples["bf16"]["ocr_plus_image"]["threshold"]


def test_rows_go_to_the_backend_in_batches_not_one_by_one(client):
    """One row at a time left vLLM idle between answers: hours per checkpoint."""
    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    calls = []
    real = backend.generate_batch
    backend.generate_batch = lambda msgs, configs, adapter=None: (calls.append(len(msgs)),
                                                                   real(msgs, configs, adapter))[1]
    model = load_model("base", client, backend_impl=backend)
    generations = generate_validation(_val_rows(5), model, batch_rows=2)
    assert calls == [2, 2, 1]
    assert all(g.error is None and g.extraction for g in generations)


def test_a_batch_that_fails_together_is_retried_row_by_row(client):
    """One unreadable page refuses the whole vLLM call; only its row may be lost."""
    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    backend.generate_batch = lambda msgs, configs, adapter=None: [RuntimeError("bad page")] * len(msgs)
    real_generate = backend.generate
    rows = _val_rows(3)

    def generate(messages, config, adapter=None):
        if messages == split_prompt(rows[1])[0]:
            raise RuntimeError("bad page")
        return real_generate(messages, config, adapter)

    backend.generate = generate
    generations = generate_validation(rows, load_model("base", client, backend_impl=backend))
    assert [g.error is None for g in generations] == [True, False, True]
    assert "bad page" in generations[1].error and generations[1].extraction == {}


def _window(sections, index, pages, golden, extraction, source="s1"):
    from evaluation.validation_generation import ValidationGeneration

    row = {"doc_type": "policy", "source_id": source, "lob": "homeowners", "sections": sections,
           "window_index": index, "window_pages": pages, "modality_mode": "image_only", "messages": []}
    return ValidationGeneration(row=row, golden=golden, extraction=extraction)


def _env(raw, pages):
    return {"raw": raw, "parsed": raw, "page_ref": pages}


def test_a_policys_windows_are_scored_as_one_document_too():
    from evaluation.validation_generation import merged_documents, score_generations

    decl = {"document": {}, "carrier": {}, "named_insured": {}, "policy": {"policy_number": _env("HO-778812", [1])}}
    arrays = {"coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A",
                             "coverage_name": _env("Dwelling", [2])}]}
    generations = [_window("arrays", 0, [2], arrays, None),            # this window failed
                   _window("decl", 0, [1], decl, decl)]
    documents = merged_documents(generations)
    assert len(documents) == 1
    gold, answer, metadata = documents[0]
    assert gold["policy"]["policy_number"]["raw"] == "HO-778812" and gold["coverages"]
    assert not answer.get("coverages") and metadata["sections"] is None and metadata["page_count"] == 2
    metrics = score_generations(generations)
    assert 0 < metrics["document_field_normalized_match"] < 1


def test_rows_read_whole_have_no_document_metrics():
    from evaluation.validation_generation import ValidationGeneration, score_generations

    row = {"doc_type": "lossrun", "source_id": "l1", "modality_mode": "image_only", "messages": []}
    golden = {"carrier_name": "Northfield Mutual"}
    metrics = score_generations([ValidationGeneration(row=row, golden=golden, extraction=golden)])
    assert not any(name.startswith("document_") for name in metrics)


def test_a_windowed_policy_is_calibrated_as_one_served_document():
    from evaluation.validation_generation import calibration_samples

    decl = {"document": {}, "carrier": {}, "named_insured": {}, "policy": {"policy_number": _env("HO-778812", [1])}}
    arrays = {"coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A",
                             "coverage_name": _env("Dwelling", [2])}],
              "additional_fields": [{"label": "Policy Number", "value": _env("HO-778812", [2]),
                                     "page_ref": [2]}]}                 # repeats a field: folded, as served
    generations = [_window("decl", 0, [1], decl, decl), _window("arrays", 0, [2], arrays, arrays)]
    for g in generations:
        g.row["val_half"] = "calibration"
        g.logprobs_by_path = {"policy.policy_number": [-0.1], "coverages[0].coverage_name": [-0.2],
                              "additional_fields[0].value": [-0.3]}
    samples = calibration_samples(generations)["calibration"]
    paths = sorted(features.field_path for features, _ in samples)
    assert "policy.policy_number" in paths and "coverages[0].coverage_name" in paths
    assert not any(path.startswith("additional_fields") for path in paths)
    assert all(correct for _, correct in samples)


def test_within_a_call_the_longest_answer_goes_first_and_each_result_reaches_its_row(client):
    """A call ends when its longest answer does, so that request starts first.
    The rows keep their order in the output, each with its own answer."""
    rows = _val_rows(3)
    long_golden = {**GOLDEN, "notes": "x" * 400}
    rows[1]["messages"][-1]["content"] = json.dumps(long_golden)
    answers = {r["source_id"]: r["messages"][-1]["content"] for r in rows}
    asked = []

    def generate_batch(msgs, configs, adapter=None):
        out = []
        for messages, config in zip(msgs, configs, strict=True):
            source = next(s for s in answers if s in json.dumps(messages))
            asked.append(source)
            out.append(EchoBackend(answers[source]).generate(messages, config, adapter))
        return out

    backend = EchoBackend("{}")
    backend.generate_batch = generate_batch
    generations = generate_validation(rows, load_model("base", client, backend_impl=backend), batch_rows=3)
    assert asked[0] == rows[1]["source_id"]
    assert [g.row["source_id"] for g in generations] == [r["source_id"] for r in rows]
    # Dates are normalised on the way out (with_output_dates), so the marker tells the answers apart.
    assert generations[1].extraction["notes"] == "x" * 400
    assert "notes" not in generations[0].extraction and "notes" not in generations[2].extraction


def test_the_rows_per_call_come_from_the_serving_config_and_the_pass_reports_its_cost(client, monkeypatch):
    from evaluation import validation_generation as V

    assert V.configured_batch_rows() == 96
    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    calls = []
    real = backend.generate_batch
    backend.generate_batch = lambda msgs, configs, adapter=None: (calls.append(len(msgs)),
                                                                   real(msgs, configs, adapter))[1]
    monkeypatch.setattr(V, "configured_batch_rows", lambda: 2)
    stats: dict = {}
    generate_validation(_val_rows(3), load_model("base", client, backend_impl=backend), stats=stats)
    assert calls == [2, 1]
    assert stats["rows"] == 3 and stats["calls"] == 2 and stats["batch_rows"] == 2
    assert stats["seconds"] >= 0 and stats["slowest_call_seconds"] >= 0


def test_selection_records_what_each_generation_pass_cost(client):
    from evaluation.checkpoint_eval import generation_cost

    backend = EchoBackend(json.dumps(GOLDEN, separators=(",", ":")))
    scorer = generation_scorer(_val_rows(3), load_model("base", client, backend_impl=backend))
    scorer.full = generation_scorer(_val_rows(4), load_model("base", client, backend_impl=backend))
    select_best(["/s/checkpoint-10", "/s/checkpoint-20"], scorer)
    scorer.full("/s/checkpoint-20")
    cost = generation_cost(scorer)
    assert [p["checkpoint"] for p in cost["passes"]] == ["/s/checkpoint-10", "/s/checkpoint-20", "/s/checkpoint-20"]
    assert cost["rows"] == 3 + 3 + 4 and cost["seconds"] >= 0
    assert generation_cost(object()) == {}


def test_a_failed_call_is_retried_in_halves_so_only_the_bad_row_goes_alone(client):
    """One unreadable page refuses the whole call. Halving finds its row in a few
    batched rounds; one row at a time would make every row of the call a call."""
    rows = _val_rows(8)
    bad = rows[5]["source_id"]
    text = json.dumps(GOLDEN, separators=(",", ":"))
    singles = []

    def generate_batch(msgs, configs, adapter=None):
        if any(bad in json.dumps(m) for m in msgs):
            return [RuntimeError("bad page")] * len(msgs)
        return [EchoBackend(text).generate(m, c, adapter) for m, c in zip(msgs, configs, strict=True)]

    def generate(messages, config, adapter=None):
        singles.append(next(r["source_id"] for r in rows if r["source_id"] in json.dumps(messages)))
        if bad in json.dumps(messages):
            raise RuntimeError("bad page")
        return EchoBackend(text).generate(messages, config, adapter)

    backend = EchoBackend(text)
    backend.generate_batch = generate_batch
    backend.generate = generate
    generations = generate_validation(rows, load_model("base", client, backend_impl=backend), batch_rows=8)
    assert [g.error is None for g in generations] == [r["source_id"] != bad for r in rows]
    assert len(singles) <= 2 and bad in singles            # the bad row's pair, not all eight
