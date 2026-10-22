"""Importing PDFs that already carry their golden label (IMPL-03 §1, IMPL-04).

The delivery shape in use: one directory per document holding the source PDF and
its canonical JSON. MinerU is **not** run here — it runs on the GPU pod as
pipeline stage 2 — so an import is cheap, runs anywhere, and leaves a document in
the state a freshly-ingested one is in, plus its label.

What these cover: the validation report is something you can act on in one pass,
and an invalid label never reaches the label store whatever the report says.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ingestion.import_labeled_pdfs import (
    LabeledPdfError,
    check_bundle,
    import_batch,
    read_bundle,
)

#: A canonical policy label — every policy is written in the client's canonical
#: schema. Its line travels in metadata (`lob`), since that schema has no
#: top-level home for one.
GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "fixtures/golden/policy_0001.golden.json")
    .read_text(encoding="utf-8")
)

#: The line a bundle here is filed under unless a test says otherwise: one whose
#: schema is still self-contained, the shape GOLDEN is written in. A bundle of no
#: line is checked against the fallback, which composes the common model since
#: common model 1.1.0, and GOLDEN is not that shape.
LINE = "property"

#: A review record: every gold needs one to import (decision 2026-10-22).
REVIEWED = {"reviewer_id": "a.reviewer", "review_date": "2026-10-20"}


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture(autouse=True)
def _stub_pdf_inspection(monkeypatch):
    """pypdf is an optional extra. What these tests are about is the label and
    the layout, not PDF parsing — the same stub the ingestion tests use."""
    from data_pipeline.ingestion import pull_raw_pdfs

    monkeypatch.setattr(
        pull_raw_pdfs, "inspect_pdf",
        lambda path: {"page_count": 2, "is_scanned": False,
                      "text_chars_per_sampled_page": 900.0},
    )


@pytest.fixture
def raw_client(client: BlobClient) -> BlobClient:
    """The one context allowed near the originals (arch §18a)."""
    return BlobClient(
        backend=client._backend, container="main", raw_container="raw", context="ingestion"
    )


def bundle(tmp_path, name="acme-wc-2026", *, golden=None, metadata=None, pdf=b"%PDF-1.7 one",
           pdfs=1):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(pdfs):
        suffix = "" if index == 0 else f"-{index}"
        (directory / f"document{suffix}.pdf").write_bytes(pdf)
    (directory / "golden.json").write_text(
        json.dumps(golden if golden is not None else GOLDEN), encoding="utf-8"
    )
    metadata = {"lob": [LINE], **REVIEWED} if metadata is None else metadata
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return directory


def batch(tmp_path, names=("a", "b"), **kw):
    root = tmp_path / "batch"
    root.mkdir(exist_ok=True)
    for index, name in enumerate(names):
        bundle(root, name, pdf=f"%PDF-1.7 {index}".encode(), **kw)
    return root


# --------------------------------------------------------------------------
# Reading a bundle
# --------------------------------------------------------------------------


def test_a_directory_is_one_document(tmp_path):
    """One golden label describes one document, so two PDFs would leave the label
    describing whichever was read first."""
    with pytest.raises(LabeledPdfError, match="holds 2 PDFs"):
        read_bundle(bundle(tmp_path, pdfs=2))


def test_a_bundle_with_no_pdf_or_no_label_is_refused(tmp_path):
    no_pdf = bundle(tmp_path, "no_pdf")
    (no_pdf / "document.pdf").unlink()
    with pytest.raises(LabeledPdfError, match="no PDF"):
        read_bundle(no_pdf)

    no_label = bundle(tmp_path, "no_label")
    (no_label / "golden.json").unlink()
    with pytest.raises(LabeledPdfError, match="no golden.json"):
        read_bundle(no_label)


# --------------------------------------------------------------------------
# Validation reports rather than stonewalls
# --------------------------------------------------------------------------


def test_every_problem_with_a_label_is_reported_at_once(tmp_path):
    """So a batch is corrected in one pass rather than one document per run."""
    check = check_bundle(
        read_bundle(bundle(tmp_path, golden={"insured_name": 42, "policy_number": []})),
        "policy",
    )

    assert not check.ok
    assert len(check.errors) >= 2


def test_an_unreadable_pdf_fails_its_document_and_the_rest_are_imported(tmp_path, client, raw_client,
                                                                       monkeypatch):
    """One PDF no reader could open ended an import part-way."""
    from data_pipeline.ingestion import pull_raw_pdfs

    def inspect(path):
        if path.read_bytes() == b"%PDF-1.7 0":
            raise pull_raw_pdfs.IngestionError(f"could not read {path.name}: Unexpected end of stream.")
        return {"page_count": 2, "is_scanned": False, "text_chars_per_sampled_page": 900.0}

    monkeypatch.setattr(pull_raw_pdfs, "inspect_pdf", inspect)
    report = import_batch(batch(tmp_path, ("a", "b")), "policy", client, raw_client)
    assert list(report.imported) == ["b"]
    assert report.failed and any("Unexpected end of stream" in e for c in report.checks for e in c.errors)


def test_validate_only_writes_nothing(tmp_path, client, raw_client):
    """The mode to use while labels are still being corrected."""
    report = import_batch(
        batch(tmp_path), "policy", client, raw_client, validate_only=True
    )

    assert report.ok and not report.imported
    assert not client.list("golden-labels/")
    assert not list(client._backend.list("raw", "raw-documents/"))


def test_a_missing_provenance_warns_without_blocking(tmp_path):
    """It costs something real — per-alias evaluation cannot score this
    document — but it does not make the label wrong."""
    check = check_bundle(read_bundle(bundle(tmp_path)), "policy")

    assert check.ok, "a missing provenance map does not make the label wrong"
    assert any("field_provenance" in w for w in check.warnings)


def test_a_complete_bundle_reports_nothing(tmp_path):
    complete = bundle(tmp_path, metadata={
        "field_provenance": {"named_insured.primary_name": 1},
        "policy_number": "WC-8842317-01",
        "lob": ["workers_comp"],
        "reviewer_id": "a.reviewer", "review_date": "2026-10-20",
    })
    check = check_bundle(read_bundle(complete), "policy")

    assert check.ok and not check.warnings


@pytest.mark.parametrize("note", [None, "Seed gold drafted by Claude Code", "not drafted by AI"])
def test_a_gold_with_no_review_record_is_refused_whatever_its_notes_say(tmp_path, note):
    """Every gold needs a review record to train or score (decision 2026-10-22):
    its notes are no evidence either way."""
    gold = {**GOLDEN, "fideon:provenance": {"provenance": note}} if note else GOLDEN
    check = check_bundle(read_bundle(bundle(tmp_path, golden=gold, metadata={"lob": [LINE]})), "policy")
    assert not check.ok and any("no review record" in e for e in check.errors)

    pending = bundle(tmp_path, "pending", golden=gold,
                     metadata={"lob": [LINE], "reviewer_id": "pending", "review_date": "TBD"})
    errors = check_bundle(read_bundle(pending), "policy").errors
    assert any("'pending' names nobody" in e and "'TBD' is not an ISO date" in e for e in errors)

    reviewed = bundle(tmp_path, "reviewed", golden=gold, metadata={"lob": [LINE], **REVIEWED})
    assert check_bundle(read_bundle(reviewed), "policy").ok


def test_a_twin_imports_only_with_its_seeds_review(tmp_path):
    twin = {"lob": [LINE], "synthetic": True, "sample": 1, **REVIEWED}
    check = check_bundle(read_bundle(bundle(tmp_path, "own", metadata=twin)), "policy")
    assert any("a twin is reviewed only when its seed was" in e for e in check.errors)
    inherited = bundle(tmp_path, "inherited", metadata={**twin, "review_inherited_from": "property/seed_1"})
    assert check_bundle(read_bundle(inherited), "policy").ok


def test_the_review_record_reaches_the_label_metadata(tmp_path, client, raw_client):
    root = tmp_path / "batch"
    root.mkdir()
    bundle(root, "doc", metadata={"lob": [LINE], "reviewer_id": "a.reviewer",
                                  "review_date": "2026-10-20", "draft_backend": "base_qwen3vl"})
    bundle(root, "twin", pdf=b"%PDF-1.7 twin", metadata={
        "lob": [LINE], "synthetic": True, **REVIEWED, "review_inherited_from": "property/seed_1"})
    report = import_batch(root, "policy", client, raw_client)
    assert report.ok, report.describe()
    stored = client.read_json(paths.label_metadata("policy", report.imported["doc"]))
    assert (stored["reviewer_id"], stored["review_date"], stored["draft_backend"]) == (
        "a.reviewer", "2026-10-20", "base_qwen3vl")
    # A twin keeps the seed it inherited its record from: every later reader checks it.
    twin = client.read_json(paths.label_metadata("policy", report.imported["twin"]))
    assert twin["review_inherited_from"] == "property/seed_1"


# --------------------------------------------------------------------------
# What an import leaves behind
# --------------------------------------------------------------------------


def test_an_import_stores_the_pdf_and_its_label_but_runs_no_ocr(tmp_path, client, raw_client):
    """MinerU runs on the GPU pod as stage 2. Rendering locally would mean a
    second MinerU whose version has to match the pod's, and a mismatch is
    distribution shift the model sees and nothing reports."""
    report = import_batch(batch(tmp_path, ("only",)), "policy", client, raw_client)

    assert report.ok, report.describe()
    source_id = report.imported["only"]

    assert raw_client.exists(paths.raw_pdf("policy", source_id))
    assert client.read_json(paths.golden_label("policy", source_id)) == GOLDEN
    assert client.read_json(paths.label_metadata("policy", source_id))["awaiting_ocr"] is True
    # OCR output is the next stage's job, not this one's.
    assert not client.exists(paths.ocr_meta("policy", source_id))
    assert not client.exists(paths.processed_page("policy", source_id, 1, "png"))


def test_an_invalid_label_is_skipped_while_the_rest_import(tmp_path, client, raw_client):
    """A hundred-document batch should not be lost to one bad label — and the bad
    one must not reach the label store either."""
    root = batch(tmp_path, ("good",))
    bundle(root, "bad", golden={"insured_name": 42}, pdf=b"%PDF-1.7 bad")

    report = import_batch(root, "policy", client, raw_client)

    assert set(report.imported) == {"good"}
    assert [c.name for c in report.failed] == ["bad"]
    labels = [k for k in client.list("golden-labels/") if k.endswith("golden.json")]
    assert len(labels) == 1, "the invalid label reached the store"
    assert "ERROR" in report.describe()


def test_re_importing_a_corrected_batch_relabels_rather_than_duplicating(
    tmp_path, client, raw_client
):
    """Correcting a label and re-running must not store a second copy of the same
    PDF under a new source_id — the checksum is what makes that safe."""
    root = batch(tmp_path, ("only",))
    first = import_batch(root, "policy", client, raw_client).imported["only"]

    corrected = json.loads(json.dumps(GOLDEN))
    corrected["named_insured"]["primary_name"]["raw"] = "Rivera Fabrication LLC (corrected)"
    (root / "only" / "golden.json").write_text(json.dumps(corrected), encoding="utf-8")
    second_report = import_batch(root, "policy", client, raw_client)

    assert second_report.imported["only"] == first
    assert second_report.duplicates == {"only": first}
    assert client.read_json(paths.golden_label("policy", first)) == corrected


def test_imported_documents_are_what_the_labeling_stage_counts(tmp_path, client, raw_client):
    """The import lands in the existing layout, so nothing downstream needs to
    know a document arrived this way."""
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    import_batch(batch(tmp_path, ("a", "b")), "policy", client, raw_client)

    assert list_labeled_source_ids(client, "policy") == ["policy_0001", "policy_0002"]


def test_the_deliverys_unseen_carrier_mark_reaches_the_label_metadata(tmp_path, client, raw_client):
    """prepare_bundles marks a test document whose carrier no training document
    has; dropped here, every delivered document read as a seen carrier's and the
    gate could not report the unseen ones apart."""
    root = tmp_path / "batch"
    root.mkdir()
    bundle(root, "unseen", pdf=b"%PDF-1.7 u", metadata={"lob": [LINE], "held_out_carrier": True, **REVIEWED})
    bundle(root, "seen", pdf=b"%PDF-1.7 s", metadata={"lob": [LINE], **REVIEWED})
    report = import_batch(root, "policy", client, raw_client)
    assert report.ok, report.describe()

    def label_metadata(name):
        return client.read_json(paths.label_metadata("policy", report.imported[name]))

    assert label_metadata("unseen")["held_out_carrier"] is True
    assert label_metadata("seen")["held_out_carrier"] is False


#: A common-model line's gold: the label rules - the form-edition rule among
#: them - hold on these lines only (label_rules.label_rules_apply).
COMMON_GOLD = json.loads((Path(__file__).resolve().parent.parent / "configs/canonical schema/common schema/"
                          "examples/homeowners_minimal.json").read_text(encoding="utf-8"))


def _with_form(gold, number):
    return {**gold, "forms_and_endorsements": [
        {"form_number": {"raw": number, "parsed": number, "page_ref": [1],
                         "confidence": {"score": 1.0, "source": "vlm"}, "flagged": False}}]}


def test_an_edition_left_inside_a_form_number_blocks_and_a_doubtful_one_warns(tmp_path):
    """The label rules move it at preparation; one still there means the bundle
    was never prepared, and the model would learn two conventions."""
    common = {"lob": ["homeowners"], **REVIEWED}
    check = check_bundle(read_bundle(bundle(tmp_path, golden=_with_form(COMMON_GOLD, "CG 00 01 04 13"),
                                            metadata=common)), "policy")
    assert any("still carry their edition" in e for e in check.errors)

    check = check_bundle(read_bundle(bundle(tmp_path, "doubtful", golden=_with_form(COMMON_GOLD, "CG 20 10 04"),
                                            metadata=common)), "policy")
    assert not any("edition" in e for e in check.errors)
    assert any("may end in their edition" in w for w in check.warnings)


@pytest.mark.parametrize("line", ["property", "auto", "wc", "umbrella"])
def test_a_self_contained_line_keeps_its_editions_in_the_form_number(tmp_path, line):
    """Nothing applies the label rules on a self-contained line - not
    prepare_bundles, not serving - so "re-run prepare_bundles" could never fix
    it: the edition is not asked of it."""
    gold = _with_form(GOLDEN, "CG 00 01 04 13")
    check = check_bundle(read_bundle(bundle(tmp_path, golden=gold, metadata={"lob": [line], **REVIEWED})),
                         "policy")
    assert not any("edition" in text for text in check.errors + check.warnings)
