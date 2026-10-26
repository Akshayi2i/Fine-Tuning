"""Review 2026-10-26 #35: label-noise detection.

``python -m orchestration.run label-noise --version vN`` reads a release's train
documents with its adapter (the golden eval's generation path) and queues, for
a person, the golds that lack a value the model read and the page prints
(``common.grounding.ground_tolerant`` on the clean text), through
``active_learning.build_queue``. No gold is changed. The model here is a stub:
``evaluate`` is injected, so no GPU.
"""

from __future__ import annotations

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import label_noise

PAGE = ("DECLARATIONS\nPolicy Number: PAP-2026-0042\nNamed Insured: Riverside Bakery LLC\n"
        "Total Premium $2,019.00\nVehicle 2 VIN 1HGCM82633A004352")


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _envelope(raw, parsed=None):
    return {"raw": raw, "parsed": parsed if parsed is not None else raw}


def test_a_printed_value_the_gold_lacks_is_found():
    gold = {"policy": {"policy_number": "PAP-2026-0042", "total_premium": None},
            "vehicles": [{"vin": None}]}
    got = {"policy": {"policy_number": _envelope("PAP-2026-0042"),
                      "total_premium": _envelope("$2,019.00", 2019.0),
                      "carrier": _envelope("Acme Mutual")},              # printed nowhere: not this question
           "named_insured": {"name": "Riverside Bakery LLC"},              # the gold has no such key
           "vehicles": [{"vin": "1HGCM82633A004352"}, {"vin": None}]}
    findings = label_noise.missing_from_gold(gold, got, {1: PAGE})
    assert [(f.path, f.value, f.pages) for f in findings] == [
        ("named_insured.name", "Riverside Bakery LLC", (1,)),
        ("policy.total_premium", "$2,019.00", (1,)),
        ("vehicles[0].vin", "1HGCM82633A004352", (1,)),
    ]


def test_values_the_gold_has_or_no_page_prints_are_not_flagged():
    gold = {"policy": {"policy_number": "PAP-2026-0042"}}
    got = {"policy": {"policy_number": "PAP-2026-0099", "carrier": "Acme Mutual", "blank": "  "}}
    assert label_noise.missing_from_gold(gold, got, {1: PAGE}) == []


def _seed(client: BlobClient, n: int = 3) -> list[str]:
    ids = []
    for i in range(n):
        source_id = f"policy_{i:04d}"
        client.write_json(paths.golden_label("policy", source_id), {"policy": {"policy_number": "PAP-2026-0042"}})
        client.write_json(paths.label_metadata("policy", source_id),
                          {"lob": "personal_auto", "reviewer_id": "a.reviewer", "review_date": "2026-10-20",
                           **({"synthetic": True} if i == n - 1 else {})})
        client.write_json(paths.ocr_meta("policy", source_id), {"page_count": 1})
        client.write_text(paths.processed_page("policy", source_id, 1, "md"), PAGE)
        client.write_bytes(paths.processed_page("policy", source_id, 1, "png"), b"\x89PNG")
        ids.append(source_id)
    client.write_json(paths.corpus_manifest("v4"),
                      {"source_ids_by_split": {"train": ids, "val": ["policy_9999"], "test": []}})
    return ids


def test_the_train_documents_are_read_and_the_flagged_ones_queued_for_a_person(client):
    ids = _seed(client)
    stored = {s: client.read_json(paths.golden_label("policy", s)) for s in ids}
    seen = {}

    def evaluate(documents, model, client_, images_root, *, modes, run_to_run):
        seen.update(ids=[d.source_id for d in documents], modes=modes, run_to_run=run_to_run,
                    images_root=images_root)
        out = []
        for doc in documents:
            meta = {"source_id": doc.source_id, "doc_type": doc.doc_type, "modality_mode": modes[0]}
            if doc.source_id == ids[1]:
                out.append((doc.golden, {}, {**meta, "error": "RuntimeError: engine died"}))
                continue
            got = {"policy": {"policy_number": "PAP-2026-0042", "total_premium": "$2,019.00"}}
            out.append((doc.golden, got, meta))
        return out

    summary = label_noise.run_label_noise(client, object(), version="v4", corpus_version="v4", evaluate=evaluate)

    # Train only, the synthetic twin left out, read in ocr_plus_image once.
    assert seen["ids"] == ids[:2] and seen["modes"] == ("ocr_plus_image",) and seen["run_to_run"] is False
    assert seen["images_root"] == paths.staging_train_images_dir("v4", None)
    assert (summary["documents"], summary["failed"], summary["flagged"], summary["findings"]) == (2, 1, 1, 1)

    queue = client.read_json(summary["queue"])
    assert summary["queue"] == f"golden-labels/{paths._tenant(None)}/_review_queue/label-noise-v4.json"
    [item] = queue["queue"]
    assert item["source_id"] == ids[0] and item["routing"] == "field_review"
    assert item["low_confidence_fields"][0]["field"] == "policy.total_premium"
    assert item["review_flags"] == ["label_noise:missing_from_gold:policy.total_premium"]
    details = client.read_json(summary["details"])
    assert details["findings"][ids[0]] == [{"path": "policy.total_premium", "value": "$2,019.00", "pages": [1]}]
    assert details["documents_failed"][0]["source_id"] == ids[1]
    # No gold is changed.
    assert {s: client.read_json(paths.golden_label("policy", s)) for s in ids} == stored


def test_limit_reads_at_most_that_many(client):
    _seed(client, 5)
    documents = label_noise.train_documents(client, "v4", limit=2)
    assert [d.source_id for d in documents] == ["policy_0000", "policy_0001"]
    assert documents[0].page_texts == {1: PAGE}
    assert documents[0].image_keys == [paths.processed_page("policy", "policy_0000", 1, "png")]


def test_a_corpus_without_a_manifest_is_refused(client):
    with pytest.raises(label_noise.LabelNoiseError, match="no manifest"):
        label_noise.train_documents(client, "v404")


def test_the_command_loads_the_release_and_releases_it(client, monkeypatch):
    from types import SimpleNamespace

    _seed(client)
    released = []
    monkeypatch.setattr("inference_core.model_runner.release_model", lambda model: released.append(model))
    monkeypatch.setattr(label_noise, "run_label_noise",
                        lambda c, model, **kw: {"model": model, **kw})
    args = SimpleNamespace(version="v4", corpus_version=None, tenant=None, limit=3)
    out = label_noise.main_from_args(args, client=client, load=lambda tag, c: f"loaded {tag}")
    assert out == {"model": "loaded v4", "version": "v4", "corpus_version": None, "tenant_id": None, "limit": 3}
    assert released == ["loaded v4"]


def test_the_cli_takes_the_command_and_refuses_bad_flags():
    from orchestration import run as cli

    args = cli.build_parser().parse_args(["label-noise", "--version", "v4", "--limit", "50"])
    assert (args.version, args.limit, args.corpus_version) == ("v4", 50, None)
    assert cli.flag_problem(args) is None
    assert "--limit" in cli.flag_problem(cli.build_parser().parse_args(["label-noise", "--version", "v4",
                                                                        "--limit", "0"]))
    assert "vN" in cli.flag_problem(cli.build_parser().parse_args(["label-noise", "--version", "latest"]))
    assert "label-noise" not in cli.QUICK_COMMANDS, "a GPU job: detached on the pod"


# --------------------------------------------------------------------------
# active_learning's CLI, wired: the unlabelled backlog through the release
# --------------------------------------------------------------------------


def test_the_backlog_is_the_unlabelled_documents_read_with_the_releases_calibrators(client, monkeypatch):
    from types import SimpleNamespace

    import serving.pipeline
    import testing.run_extraction
    from data_pipeline.labeling.active_learning import build_queue

    labelled = _seed(client, 2)
    for source_id in ("policy_0100", "policy_0101", "policy_0102"):
        client.write_json(paths.ocr_meta("policy", source_id), {"page_count": 1})
    monkeypatch.setattr(label_noise, "release_calibration", lambda c, version, tenant: ("CAL", "THR"))
    monkeypatch.setattr(testing.run_extraction, "document_request",
                        lambda c, sid, **kw: (SimpleNamespace(source_id=sid, known_acord_form=None), None))
    calls = []

    def extract(request, model, classifier, calibration, *, calibrators, thresholds, strict_schema):
        calls.append((request.source_id, calibrators, thresholds))
        if request.source_id == "policy_0101":
            raise RuntimeError("engine died")
        return SimpleNamespace(source_id=request.source_id, doc_type="policy", overall_confidence=0.5,
                               fields={"policy.policy_number": {"confidence": 0.4}}, review_flags=[])

    monkeypatch.setattr(serving.pipeline, "extract", extract)
    assert label_noise.unlabelled_documents(client, "policy") == ["policy_0100", "policy_0101", "policy_0102"]
    results = label_noise.backlog_results(client, object(), version="v4", doc_types=["policy"])
    assert [c[0] for c in calls] == ["policy_0100", "policy_0101", "policy_0102"]
    assert all(c[1:] == ("CAL", "THR") for c in calls) and not set(labelled) & {c[0] for c in calls}
    assert [r.source_id for r in results] == ["policy_0100", "policy_0102"]
    assert build_queue(results, model_version="v4").by_routing == {"field_review": 2}
    assert len(label_noise.backlog_results(client, object(), version="v4", doc_types=["policy"], limit=1)) == 1


def test_a_version_with_no_release_routes_without_calibrators(client):
    assert label_noise.release_calibration(client, "v404") == (None, None)
