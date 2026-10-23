"""The base-versus-checkpoint comparison in scripts/diagnose_generation.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.validation_generation import ValidationGeneration

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_generation.py"


def _script():
    spec = importlib.util.spec_from_file_location("diagnose_generation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


GOLDEN = {"policy": {"policy_number": _env("HO-1"), "effective_date": _env("01/01/2026")},
          "named_insured": {"primary_name": _env("Jane Rivera")}}


def _row(source_id, mode="ocr_plus_image"):
    content = [{"type": "image", "image": "x"}]
    if mode != "image_only":
        content.append({"type": "text", "text": "<page 1 of 1>\nPolicy HO-1 Jane Rivera 01/01/2026"})
    return {"source_id": source_id, "doc_type": "policy", "lob": "homeowners",
            "modality_mode": mode, "messages": [{"role": "user", "content": content}]}


def test_base_and_checkpoint_are_scored_on_the_same_rows(tmp_path, monkeypatch):
    """Whether fine-tuning beat the model it started from: the base runs with no
    adapter, the checkpoint with its own, on identical rows, scored for real."""
    import evaluation.validation_generation as V
    import inference_core.model_runner as M

    adapters = []
    base_answer = {"policy": {"policy_number": _env("HO-9"), "agent": _env("Invented Agency")}}

    def generate(rows, model, adapter=None):
        adapters.append(adapter)
        return [ValidationGeneration(row=r, golden=GOLDEN,
                                     extraction=GOLDEN if adapter else base_answer) for r in rows]

    monkeypatch.setattr(V, "generate_validation", generate)
    monkeypatch.setattr(M, "load_model", lambda tag, client: object())
    monkeypatch.setattr(M, "release_model", lambda model: None)

    out = tmp_path / "report.json"
    rows = [_row("p1"), _row("p2", "image_only")]
    rc = _script()._against_base(rows, None, SimpleNamespace(checkpoint="/ckpt", out=str(out)))
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and adapters == [None, "/ckpt"]
    tuned, base = report["checkpoint"]["metrics"], report["base"]["metrics"]
    assert tuned["field_normalized_match"] == 1.0 and base["field_normalized_match"] == 0.0
    assert tuned["field_precision"] == 1.0 and tuned["field_recall"] == 1.0
    assert base["field_recall"] == 0.0
    # The invented agent is not in the text sent; checked on the OCR row only.
    assert base["hallucination_rate"] > 0 and tuned["hallucination_rate"] == 0.0
    assert report["checkpoint"]["details"]["field_accuracy_by_lob"] == {"homeowners": 1.0}


def test_a_saved_comparison_can_be_rescored_without_the_gpu(tmp_path, monkeypatch):
    """Scoring questions (list recall read 0.0 under the old row matcher) must
    not cost another pass over the validation set on the pod."""
    import evaluation.validation_generation as V
    import inference_core.model_runner as M

    golden = {"forms_and_endorsements": [
        {"form_number": {"raw": "HO 00 03", "parsed": "HO 00 03", "page_ref": [1]},
         "form_title": {"raw": "Homeowners 3", "parsed": "Homeowners 3", "page_ref": [1]}}]}
    got = {"forms_and_endorsements": [
        {"form_number": {"raw": "HO-0003", "parsed": "HO-0003", "page_ref": [2]},
         "form_title": {"raw": "Homeowners", "parsed": "Homeowners", "page_ref": [2]}}]}
    row = {"source_id": "p1", "doc_type": "policy", "lob": "homeowners", "sections": "arrays",
           "messages": [{"role": "user", "content": [{"type": "image", "image": "x"}]}]}

    monkeypatch.setattr(V, "generate_validation", lambda rows, model, adapter=None: [
        ValidationGeneration(row=r, golden=golden, extraction=got) for r in rows])
    monkeypatch.setattr(M, "load_model", lambda tag, client: object())
    monkeypatch.setattr(M, "release_model", lambda model: None)

    script = _script()
    out = tmp_path / "report.json"
    script._against_base([row], None, SimpleNamespace(checkpoint="/ckpt", out=str(out)))
    saved = json.loads(out.read_text(encoding="utf-8"))["checkpoint"]["generations"][0]
    assert saved["row"]["messages"][0]["content"] == [{"type": "image"}]   # images as a count

    rescored = script.rescore(str(out))
    assert rescored["checkpoint"]["metrics"]["list_field_recall"] == 1.0


def test_table_rows_can_be_compared_from_a_saved_report(tmp_path):
    script = _script()
    env = lambda v: {"raw": v, "parsed": v, "page_ref": [1]}  # noqa: E731
    report = {"checkpoint": {"generations": [{
        "row": {"source_id": "p1", "lob": "homeowners", "messages": []},
        "golden": {"forms_and_endorsements": [{"form_number": env("HO 00 03")}]},
        "extraction": {"forms_and_endorsements": [{"form_number": env("CA 00 01")}]},
    }]}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    [line] = script.list_details(str(path))
    assert "key=['form_number']" in line and "HO 00 03" in line and "CA 00 01" in line


def test_every_wrong_field_is_counted_by_type(tmp_path):
    script = _script()
    env = lambda v: {"raw": v, "parsed": v, "page_ref": [1]}  # noqa: E731
    golden = {"policy": {"policy_number": env("HO-1"), "effective_date": env("01/01/2026"),
                         "expiration_date": env("01/01/2027"), "insured": env("Jane Rivera"),
                         "premium": env("1200")},
              "forms_and_endorsements": [{"form_number": env("HO 00 03")},
                                         {"form_number": env("HO 04 90")}]}
    got = {"policy": {"policy_number": env("HO-1"),                 # correct
                      "effective_date": env("01/01/2027"),          # value from another field
                      "insured": env("Jane Riveia"),                # misread
                      "premium": env("9999"),                       # wrong value
                      "agent": env("Smith")},                       # invented
           # expiration_date missing -> left empty
           "forms_and_endorsements": [{"form_number": env("HO 00 03")}, {"form_title": env("x")}]}
    report = {"checkpoint": {"generations": [
        {"row": {"source_id": "p1"}, "golden": golden, "extraction": got, "error": None},
        {"row": {"source_id": "p2"}, "golden": golden, "extraction": {}, "error": "JSONDecodeError: x"},
    ]}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = script.error_breakdown(str(path))["checkpoint"]
    assert result["errors"] == {"left empty": 1, "wrong value": 1, "value from another field": 1,
                                "misread (near miss)": 1, "invented (not in the label)": 1}
    stats = result["stats"]
    assert stats["unusable answers"] == 1 and stats["fields correct"] == 1
    assert stats["table rows expected"] == 2 and stats["table rows found"] == 1
    assert stats["table rows written with no/unknown ID"] == 1


def test_one_document_is_compared_field_by_field_across_its_windows(tmp_path):
    script = _script()
    null = {"raw": None, "parsed": None, "page_ref": []}
    gold_decl = {"policy": {"policy_number": _env("HO-1"), "effective_date": _env("01/01/2026"),
                            "premium": _env("1200")}}
    gold_arrays = {"forms_and_endorsements": [{"form_number": _env("HO 00 03")}]}
    row = lambda sections: {"source_id": "p1", "lob": "homeowners", "sections": sections,  # noqa: E731
                            "modality_mode": "ocr_plus_image", "messages": []}
    base = [{"row": row("decl"), "golden": gold_decl, "error": None,
             "extraction": {"policy": {"policy_number": _env("HO-9"),          # wrong value
                                       "effective_date": null,                # written as null
                                       "agent": _env("Invented")}}},          # invented
            {"row": row("arrays"), "golden": gold_arrays, "error": None, "extraction": {}}]
    tuned = [{"row": row("decl"), "golden": gold_decl, "error": None,
              "extraction": {"policy": {"policy_number": _env("HO-1"),
                                        "effective_date": _env("01/01/2026")}}},
             {"row": row("arrays"), "golden": gold_arrays, "error": None,
              "extraction": {"forms_and_endorsements": [{"form_number": _env("HO-0003")}]}}]
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"base": {"generations": base}, "checkpoint": {"generations": tuned}}),
                    encoding="utf-8")

    result = script.document_comparison(str(path), "p1")
    by_field = {r["field"]: r for r in result["fields"]}
    assert by_field["policy.policy_number"]["base outcome"] == "wrong value"
    assert by_field["policy.effective_date"]["base outcome"] == "written as null"
    assert by_field["policy.premium"]["base outcome"] == "missing (not written)"
    assert by_field["policy.agent"]["base outcome"] == "invented (not in gold)"
    assert by_field["policy.policy_number"]["checkpoint outcome"] == "correct"
    tuned_summary = result["summary"]["checkpoint"]
    assert tuned_summary["outcomes"]["correct"] >= 2 and tuned_summary["outcomes"]["missing (not written)"] == 1
    assert tuned_summary["table_rows"]["forms_and_endorsements"] == "1 of 1 found, 1 written"
    assert result["summary"]["base"]["accuracy"] == 0.0

    out = tmp_path / "compare"
    rc = script.main(["--document", "p1", "--report", str(path), "--out", str(out)])
    assert rc == 0
    assert "policy.premium" in (out / "fields.csv").read_text(encoding="utf-8-sig")
    assert json.loads((out / "gold.json").read_text(encoding="utf-8"))["policy"]["premium"]["raw"] == "1200"
    tuned_json = json.loads((out / "fine_tuned_output.json").read_text(encoding="utf-8"))
    assert tuned_json["forms_and_endorsements"][0]["form_number"]["raw"] == "HO-0003"   # windows merged
    assert "policy" in json.loads((out / "base_output.json").read_text(encoding="utf-8"))
    assert "accuracy (single fields)" in (out / "summary.txt").read_text(encoding="utf-8")


def test_the_pdf_is_found_from_the_bundle_the_document_was_imported_from(tmp_path):
    script = _script()
    bundle = tmp_path / "smoke-personal-v1" / "rv__allstate__original"
    bundle.mkdir(parents=True)
    (bundle / "document.pdf").write_bytes(b"%PDF")
    found = script._find_pdf({"imported_from": "rv__allstate__original"}, roots=(str(tmp_path),))
    assert found == bundle / "document.pdf"
    assert script._find_pdf({}, roots=(str(tmp_path),)) is None


def test_the_test_split_gets_its_own_report():
    script = _script()
    assert script._report_path("val").endswith("base_vs_checkpoint.json")
    assert script._report_path("test").endswith("base_vs_checkpoint_test.json")


def test_the_test_split_falls_back_to_the_corpus_wide_file_for_the_adapter():
    """Training writes the adapter's own train and val files only; test is
    corpus-wide, narrowed to the adapter's document types (policies)."""
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    script = _script()
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    rows = [{"source_id": "p1", "doc_type": "policy"}, {"source_id": "l1", "doc_type": "lossrun"}]
    client.write_text(paths.corpus_eval_split("v0", "test", "smoke"),
                      "\n".join(json.dumps(r) for r in rows))
    args = SimpleNamespace(corpus="v0", split="test", tenant="smoke")
    assert [r["source_id"] for r in script._eval_rows(client, args)] == ["p1"]

    # The adapter's own copy, when training wrote one, is read as it is.
    client.write_text(paths.corpus_scope_eval_split("v0", "test", "policy", "smoke"),
                      json.dumps({"source_id": "p2", "doc_type": "policy"}))
    assert [r["source_id"] for r in script._eval_rows(client, args)] == ["p2"]


def test_one_documents_rows_are_built_from_blob_in_every_reading_mode():
    """For a document outside val and test (a training one included)."""
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    script = _script()
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    label = {"policy": {"policy_number": _env("HO-1")},
             "named_insured": {"primary_name": _env("Jane Rivera")}}
    client.write_json(paths.golden_label("policy", "policy_0001", "smoke"), label)
    client.write_json(paths.label_metadata("policy", "policy_0001", "smoke"),
                      {"lob": "homeowners", "split": "train", "synthetic": False})
    client.write_json(paths.ocr_meta("policy", "policy_0001", "smoke"), {"page_count": 2})
    for page in (1, 2):
        client.write_text(paths.processed_page("policy", "policy_0001", page, "md", "smoke"),
                          "DECLARATIONS Policy HO-1 Jane Rivera")
    rows = script._document_rows(client, "policy_0001", "smoke")
    assert rows and {r["source_id"] for r in rows} == {"policy_0001"}
    assert {r["modality_mode"] for r in rows} == {"ocr_plus_image", "noisy_ocr_image", "image_only"}
    assert script._document_report_path("policy_0001").endswith("base_vs_checkpoint_policy_0001.json")


def test_the_label_audit_says_which_side_the_page_supports(tmp_path):
    script = _script()
    gold = {"policy": {"policy_number": _env("HO-1"), "premium": _env("9999")},
            "homeowners": {"deductibles": {}}}
    answer = {"policy": {"policy_number": _env("HO-1")},
              "homeowners": {"deductibles": {"theft_deductible": _env("$500"),     # printed, gold lacks
                                             "named_storm_deductible": _env("2%")}}}  # not printed
    text = "<page 1 of 1>\nPolicy HO-1  Theft deductible $500  Total premium 1,200"
    row = {"source_id": "p1", "lob": "homeowners", "modality_mode": "ocr_plus_image",
           "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"checkpoint": {"generations": [
        {"row": row, "golden": gold, "extraction": answer, "error": None}]}}), encoding="utf-8")

    findings = {f["field"]: f["finding"] for f in script.label_audit(str(path))}
    assert findings["homeowners.deductibles.theft_deductible"].startswith("gold likely missing")
    assert findings["homeowners.deductibles.named_storm_deductible"].startswith("model likely invented")
    assert findings["policy.premium"].startswith("gold value not in the page text")
    assert "policy.policy_number" not in findings                       # agreed and printed

    out = tmp_path / "audit.csv"
    assert script.main(["--audit-labels", str(path), "--out", str(out)]) == 0
    assert "theft_deductible" in out.read_text(encoding="utf-8-sig")
