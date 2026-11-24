"""``run_extraction --as-served``: the test routine reading as production serves (review R1).

Done when (EXTRACTION_FIX_INSTRUCTIONS, R1): with the echo backend, ``--as-served``
gives the same answer object as the golden eval's reading of the same stored
document. Here the gate's reading is run on its own - ``golden_eval.evaluate``
over the document's Blob keys - and compared with the routine's.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from inference_core.model_runner import EchoBackend, load_model
from tests.test_extract_with_adapter import _adapter, _imported_document

#: One value the pages print (HO-1, on page 1) and one they do not (a date no page holds).
ANSWER = json.dumps({"policy": {
    "policy_number": {"raw": "HO-1", "parsed": "HO-1", "page_ref": [1]},
    "effective_date": {"raw": "07/04/2031", "parsed": "2031-07-04", "page_ref": [2]},
}})


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _gate_reading(client, model, tmp_path):
    """The gate's reading of policy_0001, as evaluate makes it: image_only, from the frozen document."""
    from evaluation.golden_eval import GoldenDocument, evaluate

    tenant = "t1"
    doc = GoldenDocument(
        source_id="policy_0001", doc_type="policy",
        golden=client.read_json(paths.golden_label("policy", "policy_0001", tenant)),
        image_keys=[paths.processed_page("policy", "policy_0001", page, "png", tenant) for page in (1, 2)],
        page_texts={page: f"page {page} HO-1" for page in (1, 2)}, lob="homeowners",
    )
    [(_expected, extraction, metadata)] = evaluate([doc], model, client, tmp_path / "gate_pages",
                                                   modes=("image_only",), run_to_run=False, in_flight=1)
    return extraction, metadata


def test_as_served_gives_the_gates_answer_and_the_fields_first_result(client, tmp_path):
    from testing.run_extraction import run_batch

    _imported_document(client)
    echo = EchoBackend(ANSWER)
    model = load_model("base", client, backend_impl=echo, adapter=str(_adapter(tmp_path)))
    gate_extraction, gate_metadata = _gate_reading(client, model, tmp_path)
    gate_calls = len(echo.calls)

    summary = run_batch(model, client, ["policy_0001"], tenant_id="t1", mode="image_only", as_served=True,
                        images_root=tmp_path / "pages", root=tmp_path / "out")
    assert summary.counts["failed"] == 0 and summary.counts["scored"] == 1
    run = summary.run_id
    out = tmp_path / "out"
    served = json.loads((out / "results" / model.tag / run / "policy_0001.json").read_text(encoding="utf-8"))

    # The same answer object as the gate's reading.
    assert served["extraction"] == gate_extraction
    # Held to the stored page text: the printed value kept, the one no page prints taken out.
    assert served["extraction"]["policy"]["policy_number"]["raw"] == "HO-1"
    # (The old image_only path served it, at 0.96 confidence: checked 2026-11-24.)
    assert served["extraction"]["policy"]["effective_date"]["raw"] is None
    assert gate_metadata["printed_text"] == "borrowed"
    # Image only, as production reads: no page text was sent to the model.
    sent = json.dumps([call["messages"] for call in echo.calls[gate_calls:]], default=str)
    assert "page 1 HO-1" not in sent and "page 2 HO-1" not in sent

    # What a user sees first, beside it.
    first = json.loads((out / "results" / model.tag / run / "policy_0001.fields_first.json")
                       .read_text(encoding="utf-8"))
    assert first["extraction"]["policy"]["policy_number"]["raw"] == "HO-1"
    assert any(flag.endswith(":pending_full_text") for flag in first["review_flags"])   # the date waits
    metrics = json.loads((out / "metrics" / model.tag / run / "policy_0001.metrics.json").read_text(encoding="utf-8"))
    assert metrics["path"] == "as_served" and metrics["calibration"] == "identity"
    assert metrics["fields_first"]["served"] is True and metrics["fields_first"]["values_held_back"] >= 1
    assert "field_normalized_match_rate" in metrics["fields_first"]
    assert summary.metrics["fields_first"]["documents_served_first"] == 1

    # run.json says which path and which calibration made the numbers.
    record = json.loads((out / "results" / model.tag / run / "run.json").read_text(encoding="utf-8"))
    assert record["path"] == "as_served" and record["calibration"] == "identity" and record["mode"] == "image_only"
    assert run.endswith("-image_only-as_served")


def test_the_default_path_is_unchanged_and_says_so(client, tmp_path):
    from testing.run_extraction import run_batch

    _imported_document(client)
    model = load_model("base", client, backend_impl=EchoBackend(ANSWER), adapter=str(_adapter(tmp_path)))
    summary = run_batch(model, client, ["policy_0001"], tenant_id="t1", images_root=tmp_path / "pages",
                        root=tmp_path / "out")
    metrics = json.loads((tmp_path / "out" / "metrics" / model.tag / summary.run_id / "policy_0001.metrics.json")
                         .read_text(encoding="utf-8"))
    assert metrics["path"] == "harness" and metrics["calibration"] == "identity" and "fields_first" not in metrics
    assert not (tmp_path / "out" / "results" / model.tag / summary.run_id / "policy_0001.fields_first.json").exists()


def test_as_served_refuses_any_mode_but_image_only(client, tmp_path):
    from testing.run_extraction import HarnessError, run_batch

    model = load_model("base", client, backend_impl=EchoBackend(ANSWER), adapter=str(_adapter(tmp_path)))
    with pytest.raises(HarnessError, match="image_only"):
        run_batch(model, client, ["policy_0001"], mode="ocr_plus_image", as_served=True, root=tmp_path / "out")


@pytest.mark.parametrize("argv,message", [
    (["--model", "v4", "--as-served", "--mode", "ocr_plus_image", "--source-ids", "p"], "leave out --mode"),
    (["--model", "base", "--as-served", "--source-ids", "p"], "name it with --release"),
    (["--model", "v4", "--release", "v4", "--source-ids", "p"], "only with --as-served"),
])
def test_the_command_refuses_what_as_served_cannot_mean(argv, message, capsys):
    from testing.run_extraction import main

    with pytest.raises(SystemExit) as exited:
        main(argv)
    assert exited.value.code == 2 and message in capsys.readouterr().err


def test_a_release_is_read_with_its_calibrators_or_says_it_has_none(client):
    from testing.run_extraction import HarnessError, release_calibration

    with pytest.raises(HarnessError, match="no bundle"):
        release_calibration(client, "v9")
    client.write_json(paths.release_bundle("v4", None), {
        "release_id": "v4", "served_as": "lora_on_base", "serving_formats": {"bf16": "adapters/v4"},
        "calibrators": {}})
    served = release_calibration(client, "v4")
    assert served["calibrators"] is None
    assert served["calibration_source"] == "release v4 has no calibrators: identity, as serving falls back"


def test_the_default_image_only_path_is_not_what_is_served(client, tmp_path):
    """Seen to fail on purpose (review R1): the default path read image_only
    serves the value no page prints, so its answer is not the gate's."""
    from testing.run_extraction import run_batch

    _imported_document(client)
    model = load_model("base", client, backend_impl=EchoBackend(ANSWER), adapter=str(_adapter(tmp_path)))
    gate_extraction, _ = _gate_reading(client, model, tmp_path)
    summary = run_batch(model, client, ["policy_0001"], tenant_id="t1", mode="image_only",
                        images_root=tmp_path / "pages", root=tmp_path / "out")
    raw = json.loads((tmp_path / "out" / "results" / model.tag / summary.run_id / "policy_0001.json")
                     .read_text(encoding="utf-8"))
    assert raw["extraction"] != gate_extraction
    assert raw["extraction"]["policy"]["effective_date"]["raw"] == "07/04/2031"


def test_release_check_is_extract_as_served_by_its_own_name(monkeypatch):
    from orchestration import run as cli
    from testing import run_extraction

    for var in ("RUNPOD_POD_ID", "TMUX"):
        monkeypatch.delenv(var, raising=False)
    sent = []
    monkeypatch.setattr(run_extraction, "main", lambda argv: sent.append(argv) or 0)
    assert cli.main(["release-check", "--version", "v4", "--corpus", "c3", "--limit", "5"]) == 0
    assert cli.main(["extract", "--model", "v4", "--as-served", "--source-ids", "p1"]) == 0
    check, extract = sent
    assert check[:4] == ["--model", "v4", "--doc-type", "policy"]
    assert "--as-served" in check and check[check.index("--release") + 1] == "v4"
    assert check[check.index("--split") + 1] == "test" and check[check.index("--corpus") + 1] == "c3"
    assert "--mode" not in check and "--as-served" in extract and "--mode" not in extract
    for refused in (["extract", "--model", "v4", "--as-served", "--mode", "ocr_plus_image", "--source-ids", "p1"],
                    ["release-check", "--version", "latest", "--corpus", "c3"]):
        with pytest.raises(SystemExit) as exited:
            cli.main(refused)
        assert exited.value.code == 2
    assert len(sent) == 2
