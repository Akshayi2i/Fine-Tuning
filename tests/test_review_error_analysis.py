"""Error analysis in the golden and validation reports (review 2026-10-26):
per carrier and per mode, which field's value turned up under another field,
seen against unseen carriers, and each field against the promoted release.
Reported, never gated."""

from __future__ import annotations

import pytest

import common.config
from evaluation.run_eval import build_report, field_deltas


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(common.config, "shared_format_config", lambda: {"profile": "fast"})


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


GOLD = {"policy": {"policy_number": _env("HO-778812"), "account_id": _env("A-55120")}}


def _meta(source, mode="ocr_plus_image", **extra):
    return {"source_id": source, "doc_type": "policy", "lob": ["property"], "modality_mode": mode, **extra}


def _analysis(triples):
    return build_report("v", triples).as_dict()["error_analysis"]["policy"]


def test_errors_are_tabled_per_carrier_and_per_mode():
    swapped = {"policy": {"policy_number": _env("HO-778812"), "account_id": _env("A-99999")}}
    triples = [
        (GOLD, GOLD, _meta("p1", carrier="Acme Mutual")),
        (GOLD, swapped, _meta("p1", "image_only", carrier="Acme Mutual")),
        (GOLD, GOLD, _meta("p2", carrier="Zenith")),
    ]
    analysis = _analysis(triples)
    acme, zenith = analysis["by_carrier"]["Acme Mutual"], analysis["by_carrier"]["Zenith"]
    assert (acme["documents"], acme["readings"], acme["fields_scored"], acme["fields_wrong"]) == (1, 2, 4, 1)
    assert acme["field_match"] == 0.75 and sum(acme["errors_by_class"].values()) == 1
    assert zenith["field_match"] == 1.0 and zenith["errors_by_class"] == {}
    assert analysis["by_mode"]["image_only"]["field_match"] == 0.5
    assert analysis["by_mode"]["ocr_plus_image"]["fields_wrong"] == 0
    assert analysis["by_field"]["policy.account_id"] == {"scored": 3, "correct": 2, "accuracy": 0.6667}
    assert "by_carrier_seen_in_train" not in analysis           # no reading says


def test_a_label_value_written_under_another_field_is_a_confusion():
    confused = {"policy": {"policy_number": _env("A-55120"), "account_id": _env("A-55120")}}
    analysis = _analysis([(GOLD, confused, _meta("p1"))])
    [row] = analysis["field_confusion"]
    assert (row["gold_field"], row["found_under"], row["count"]) == ("policy.account_id", "policy.policy_number", 1)
    assert row["examples"][0]["source_id"] == "p1"


def test_seen_and_unseen_carriers_are_split_when_the_readings_say():
    triples = [(GOLD, GOLD, _meta("p1", carrier_seen_in_train=True)),
               (GOLD, {"policy": {}}, _meta("p2", carrier_seen_in_train=False))]
    split = _analysis(triples)["by_carrier_seen_in_train"]
    assert split["seen"]["field_match"] == 1.0 and split["unseen"]["field_match"] == 0.0


def test_each_field_is_compared_with_the_promoted_release():
    swapped = {"policy": {"policy_number": _env("HO-778812"), "account_id": _env("A-99999")}}
    promoted = build_report("v1", [(GOLD, GOLD, _meta("p1"))]).as_dict()
    candidate = build_report("v2", [(GOLD, swapped, _meta("p1"))]).as_dict()
    deltas = field_deltas(candidate, promoted)["policy"]
    assert deltas["against"] == "v1"
    assert deltas["fields"][0] == {"field": "policy.account_id", "accuracy": 0.0, "promoted": 1.0, "delta": -1.0,
                                   "scored": 1, "promoted_scored": 1}
    assert field_deltas(candidate, None) is None
    # A promoted report written before by_field was kept offers its weakest fields.
    old = {"model_version": "v0", "by_doc_type": {"policy": [{"subset": "full", "weakest_fields": [
        {"field": "policy.account_id", "accuracy": 0.5, "scored": 4}]}]}}
    assert field_deltas(candidate, old)["policy"]["fields"][0]["delta"] == -0.5


def test_the_golden_document_carries_the_frozen_carrier(monkeypatch):
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation import golden_eval

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    base = f"{paths.golden_eval_set_dir(None)}/policy_0001"
    client.write_json(f"{base}/golden.json", GOLD)
    client.write_json(f"{base}/metadata.json", {"doc_type": "policy", "lob": "property", "carrier": "acme mutual",
                                                "carrier_seen_in_train": False})
    client.write_bytes(f"{base}/page_1.png", b"png")
    client.write_text(f"{base}/page_1.md", "Policy HO-778812 Account A-55120")
    monkeypatch.setattr("data_pipeline.labeling.export_golden_labels.review_problem", lambda label, meta: None)
    [doc] = golden_eval.load_golden_set(client)
    assert (doc.carrier, doc.carrier_seen_in_train) == ("acme mutual", False)
