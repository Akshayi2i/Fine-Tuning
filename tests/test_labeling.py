"""IMPL-04 — golden-label admission, the day-zero rule, and alias derivation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.labeling import derive_aliases as DA
from data_pipeline.labeling.export_golden_labels import (
    LabelValidationError,
    export_golden_label,
    inter_annotator_agreement,
    list_labeled_source_ids,
    review_requirement,
    validate_golden_label,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _policy_label(**over):
    """A canonical policy label: the client's tree, every value in an envelope."""
    label = json.loads((FIXTURES / "golden/policy_0001.golden.json").read_text(encoding="utf-8"))
    label.update(over)
    return label


def _envelope(raw, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed,
            "confidence": {"score": 1.0, "source": "audit"}, "page_ref": [1], "flagged": False}


def _common_model_label(line: str | None = None) -> dict:
    """The canonical policy label in the common model's shape (SPEC_21): dates
    parsed MM/DD/YYYY, the carrier named, the address and the premium under the
    common model's names. The fallback - a policy of no single line - composes
    the common model since common model 1.1.0, so this is its shape; a line on
    the common model adds the blocks its overlay requires: the document, the
    one part of a single-line policy, and its coverages."""
    label = {
        "carrier": {"name": _envelope("Granite Mutual Insurance Company")},
        "producer": {"agency_name": _envelope("Hanover Risk Partners")},
        "policy": {
            "policy_number": _envelope("WC-8842317-01"),
            "effective_date": _envelope("04/01/2026"),
            "expiration_date": _envelope("04/01/2027"),
        },
        "named_insured": {
            "primary_name": _envelope("Rivera Fabrication LLC"),
            "mailing_address": {"street": _envelope("1420 Foundry Road"), "city": _envelope("Toledo"),
                                "state": _envelope("OH"), "postal_code": _envelope("43604")},
        },
        "premium": {"total": _envelope("$47,250.00", 47250.0)},
    }
    if line is None:
        return label
    return {"document": {"doc_type": "policy_check"}, **label,
            "lob_parts": [{"part_id": "part_1", "lob": line}], "coverages": []}


def _with_insured(name: str) -> dict:
    """The canonical policy label with its named insured replaced."""
    label = _policy_label()
    label["named_insured"] = {
        **label["named_insured"],
        "primary_name": {**label["named_insured"]["primary_name"], "raw": name, "parsed": name},
    }
    return label


def _flat_label(**over):
    """A flat label — a Loss Run. The line_of_business rules are properties of
    the flat schemas, which carry the field at the top level; a canonical policy
    takes its line from metadata instead (see the canonical tests below)."""
    label = json.loads((FIXTURES / "golden/lossrun_0001.golden.json").read_text(encoding="utf-8"))
    label.update(over)
    return label


def _acord_label():
    return json.loads((FIXTURES / "golden/acord_0001.golden.json").read_text(encoding="utf-8"))


def _fixture_documents():
    docs = []
    for golden in sorted((FIXTURES / "golden").glob("*.golden.json")):
        source_id = golden.name.replace(".golden.json", "")
        ocr = list((FIXTURES / "ocr").glob(f"{source_id}_page_*.md"))
        if ocr:
            docs.append((
                source_id,
                json.loads(golden.read_text(encoding="utf-8")),
                ocr[0].read_text(encoding="utf-8"),
            ))
    return docs


# --------------------------------------------------------------------------
# Label admission
# --------------------------------------------------------------------------

def test_valid_label_passes():
    validate_golden_label(_policy_label(), "policy", lob="workers_comp")
    validate_golden_label(_flat_label(), "lossrun")


def test_label_without_line_of_business_is_rejected():
    """Required in every flat label, even when empty (arch §0b)."""
    label = _flat_label()
    del label["line_of_business"]
    with pytest.raises(LabelValidationError, match="line_of_business is missing"):
        validate_golden_label(label, "lossrun")


def test_out_of_enum_line_of_business_is_rejected():
    with pytest.raises(LabelValidationError, match="invalid line_of_business"):
        validate_golden_label(_flat_label(line_of_business=["marine_cargo"]), "lossrun")


def test_an_empty_line_of_business_is_accepted():
    """An empty list means the document does not determine a line — a correct
    answer, and distinct from omitting the field. Under v1 this was `null`;
    v2.1 §0b makes it a list, so `[]` carries that meaning."""
    validate_golden_label(_flat_label(line_of_business=[]), "lossrun")


def test_several_lines_are_accepted_on_one_document():
    """The whole reason the field became a list: a package policy or certificate
    routinely covers several lines, and the v1 scalar forced the annotator to
    pick one and discard the rest."""
    validate_golden_label(
        _flat_label(line_of_business=["general_liability", "property"]), "lossrun"
    )


def test_a_duplicated_line_is_rejected():
    validate = validate_golden_label
    with pytest.raises(LabelValidationError):
        validate(_flat_label(line_of_business=["property", "property"]), "lossrun")


# --------------------------------------------------------------------------
# Canonical policy labels
# --------------------------------------------------------------------------

def test_a_canonical_policy_label_needs_no_top_level_line_of_business():
    """The client's schema has no top-level line_of_business, so a canonical
    label cannot carry one. Its line travels in the metadata instead."""
    label = _policy_label()
    assert "line_of_business" not in label
    validate_golden_label(label, "policy", lob="workers_comp")
    # An unknown line: the fallback schema, on the common model since 1.1.0.
    fallback = _common_model_label()
    assert "line_of_business" not in fallback
    validate_golden_label(fallback, "policy")


def test_a_canonical_policy_label_with_a_line_that_has_no_schema_is_rejected():
    with pytest.raises(LabelValidationError, match="names no canonical policy schema"):
        validate_golden_label(_policy_label(), "policy", lob="marine_cargo")


@pytest.mark.parametrize("lob,label", [
    ("flood", _policy_label()),
    ("gl", _common_model_label("gl")),                  # a common-model overlay since 1.1.0
    ("cyber", _policy_label()),
    ("workers_comp", _policy_label()),
    (["homeowners", "personal_auto"], _common_model_label()),  # several lines: the fallback
], ids=["flood", "gl", "cyber", "workers_comp", "homeowners+personal_auto"])
def test_a_policy_line_is_any_line_with_a_canonical_schema(lob, label):
    """A policy's line names its schema. The LOB enum (13 values) rejected flood,
    cyber and the schema spellings (gl, wc), so real labels failed export. Each
    label is in its schema's shape: gl and the fallback compose the common model."""
    validate_golden_label(label, "policy", lob=lob)


def test_a_flat_label_under_a_canonical_policy_is_rejected():
    """A pre-canonical policy label is the wrong shape for every policy now."""
    flat = {"insured_name": "Rivera Fabrication LLC", "line_of_business": ["workers_comp"]}
    with pytest.raises(LabelValidationError, match="required property"):
        validate_golden_label(flat, "policy", lob="workers_comp")


def test_a_canonical_provenance_names_a_nested_path():
    validate_golden_label(
        _policy_label(), "policy", lob="workers_comp",
        field_provenance={"named_insured.primary_name": "Applicant"},
    )


def test_a_canonical_provenance_naming_a_confusable_is_rejected():
    """The alias registry is keyed by canonical field name, so a nested path is
    checked by its leaf — and the confusable check still bites."""
    with pytest.raises(LabelValidationError, match="registered CONFUSABLE"):
        validate_golden_label(
            _policy_label(), "policy", lob="workers_comp",
            field_provenance={"named_insured.primary_name": "Certificate Holder"},
        )


def test_acord_label_without_a_form_is_rejected():
    """The form selects the schema, which is why classification is two-level."""
    label = json.loads((FIXTURES / "golden/acord_0001.golden.json").read_text(encoding="utf-8"))
    with pytest.raises(LabelValidationError, match="acord_form"):
        validate_golden_label(label, "acord")


def test_provenance_naming_a_confusable_is_rejected():
    """The highest-value annotation check: this mistake teaches the model to
    conflate distinct parties, and it would train perfectly happily."""
    with pytest.raises(LabelValidationError, match="registered CONFUSABLE"):
        validate_golden_label(
            _acord_label(), "acord", acord_form="25",
            field_provenance={"insured_name": "Certificate Holder"},
        )


def test_provenance_with_a_real_alias_is_accepted():
    validate_golden_label(
        _acord_label(), "acord", acord_form="25", field_provenance={"insured_name": "INSURED"}
    )


def test_provenance_naming_an_absent_field_is_rejected():
    with pytest.raises(LabelValidationError, match="not in the label"):
        validate_golden_label(
            _policy_label(), "policy", field_provenance={"nonexistent_field": "Whatever"}
        )


# --------------------------------------------------------------------------
# Day-zero rule (arch §4c)
# --------------------------------------------------------------------------

def test_day_zero_requires_full_review_until_the_threshold(client):
    assert review_requirement(client, "policy") == "full"
    assert review_requirement(client, "policy", minimum=0) == "confidence_routed"


def test_accepting_a_draft_unreviewed_is_refused_on_day_zero(client):
    """Drafts at this stage come from the untuned base model; trusting one puts
    its errors straight into the training target."""
    with pytest.raises(LabelValidationError, match="every draft still requires full human"):
        export_golden_label(
            _policy_label(), "policy_0001", "policy", client,
            reviewer_id="alice", accepted_without_review=True,
        )


def test_a_draft_accepted_without_review_is_refused_after_day_zero_too(client, monkeypatch):
    """No review record, so it could never train or score (decision 2026-10-22):
    written, it would only stop the next corpus build."""
    from data_pipeline.labeling import export_golden_labels as E

    monkeypatch.setattr(E, "review_requirement", lambda *a, **k: "confidence_routed")
    with pytest.raises(LabelValidationError, match="no review record"):
        export_golden_label(_policy_label(), "policy_0001", "policy", client,
                            reviewer_id="alice", accepted_without_review=True, lob="property")
    assert not client.exists(paths.golden_label("policy", "policy_0001"))


@pytest.mark.parametrize("reviewer", ["", " ", "pending", "N/A"])
def test_an_export_naming_no_reviewer_writes_nothing(client, reviewer):
    with pytest.raises(LabelValidationError, match="reviewer_id"):
        export_golden_label(_policy_label(), "policy_0001", "policy", client, reviewer_id=reviewer,
                            lob="property")
    assert not client.exists(paths.golden_label("policy", "policy_0001"))


def test_export_writes_label_and_provenance(client):
    export_golden_label(
        _policy_label(), "policy_0001", "policy", client,
        reviewer_id="alice", draft_backend="base_qwen3vl",
        field_provenance={"named_insured.primary_name": "Applicant"},
        lob="workers_comp",
    )
    assert client.exists(paths.golden_label("policy", "policy_0001"))

    meta = client.read_json(paths.label_metadata("policy", "policy_0001"))
    assert meta["reviewer_id"] == "alice"
    assert meta["draft_backend"] == "base_qwen3vl"
    assert meta["field_provenance"]["named_insured.primary_name"] == "Applicant"
    assert meta["review_requirement"] == "full"
    # Where the dataset build reads the line from to select the schema.
    assert meta["lob"] == "workers_comp"


def test_a_relabel_keeps_what_the_delivery_recorded(client):
    """Rewritten from the review alone, a re-reviewed gold lost its delivered
    split and held-out mark, and the purge plan the folders to find its copies."""
    from data_pipeline.labeling.export_golden_labels import DELIVERY_KEYS

    delivered = {
        "imported_from": "HO_0002", "split": "test", "held_out_carrier": True, "carrier": "Dryden Mutual",
        "twin_index": None, "synthetic": False, "template_id": "t1", "render_mode": "scan",
        "document_kind": "original",
        "imported_from_history": [{"folder": "HO_0001", "batch": "personal-v1"},
                                  {"folder": "HO_0002", "batch": None}],
    }
    assert set(delivered) == set(DELIVERY_KEYS)
    key = paths.label_metadata("policy", "policy_0001")
    client.write_json(key, {**delivered, "reviewer_id": "bob", "review_date": "2026-10-01",
                            "unprinted_values": ["policy.policy_term_months"]})

    export_golden_label(_policy_label(), "policy_0001", "policy", client, reviewer_id="alice",
                        lob="property")
    meta = client.read_json(key)
    assert {k: meta[k] for k in DELIVERY_KEYS} == delivered
    assert meta["reviewer_id"] == "alice" and meta["review_date"] != "2026-10-01"
    assert "unprinted_values" not in meta                              # read from the old label


def test_labeled_source_ids_are_listed(client):
    for source_id in ("policy_0001", "policy_0002"):
        export_golden_label(_policy_label(), source_id, "policy", client, reviewer_id="alice",
                            lob="property")
    assert list_labeled_source_ids(client, "policy") == ["policy_0001", "policy_0002"]


def test_inter_annotator_agreement_measures_labeling_noise():
    """Sets a realistic ceiling on model scores — some residual error at plateau
    is human disagreement, not model failure (arch §7)."""
    a = _policy_label()
    b = _with_insured("Someone Else Entirely")
    agreement, disagreements = inter_annotator_agreement(a, b)
    assert agreement < 1.0
    # Field by field, not object by object: one wrong name is one disagreement,
    # not the whole `named_insured` block.
    assert disagreements == ["named_insured.primary_name"]

    perfect, none = inter_annotator_agreement(a, _policy_label())
    assert perfect == 1.0 and not none


def test_agreement_ignores_the_annotators_own_confidence():
    """The envelope's confidence is not an annotation. Two annotators who agree
    on every value agree, whatever scores their tools attached."""
    a = _policy_label()
    b = _policy_label()
    b["policy"]["policy_number"] = {**b["policy"]["policy_number"],
                                    "confidence": {"score": 0.4, "source": "vlm"}}
    assert inter_annotator_agreement(a, b) == (1.0, [])


def test_agreement_uses_normalized_comparison():
    """`Acme Mfg LLC` and `ACME MANUFACTURING LLC` are the same answer."""
    a = _with_insured("Acme Mfg LLC")
    b = _with_insured("ACME MANUFACTURING LLC")
    agreement, disagreements = inter_annotator_agreement(a, b)
    assert "named_insured.primary_name" not in disagreements
    assert agreement == 1.0


# --------------------------------------------------------------------------
# Alias derivation (IMPL-04 §3) — the headline capability
# --------------------------------------------------------------------------

def test_one_canonical_field_derives_its_several_surface_labels():
    """The whole point: the golden JSON gives the value, the OCR gives the text,
    and the label is whatever introduces that value on the page."""
    report = DA.derive_aliases(_fixture_documents())
    # The canonical policies' named insured, found under two different labels.
    policy = {label.casefold() for label in report.aliases["named_insured.primary_name"]}
    assert {"applicant", "named insured"} <= policy
    # The flat types keep their own field name, and their own labels.
    assert "insured" in {label.casefold() for label in report.aliases["insured_name"]}


def test_a_canonical_label_is_aligned_by_its_printed_value():
    """The envelope's keys are not fields: no `…raw` or `…confidence` path may
    come out of derivation, or the registry fills with nonsense aliases."""
    report = DA.derive_aliases(_fixture_documents())
    for field in report.aliases:
        assert not field.endswith((".raw", ".parsed", ".page_ref", ".flagged")), field
        assert ".confidence" not in field, field


def test_normalized_matching_anchors_dates_and_currency():
    """The golden JSON is canonical (2026-03-31) while the page is not
    (03/31/2026). Without the normalized pass nearly every date would be
    unresolved."""
    report = DA.derive_aliases(_fixture_documents())
    assert "Valued As Of" in report.aliases["valuation_date"]
    assert any("Premium" in label for label in report.aliases["premium.total_policy_premium"])


def test_table_cells_resolve_to_their_column_header():
    """In a multi-column data row the cell to the LEFT is a different field's
    value, not this field's label."""
    report = DA.derive_aliases(_fixture_documents())
    assert "Claim Number" in report.aliases["claims[].claim_number"]
    assert "Total Incurred" in report.aliases["claims[].total_incurred"]


def test_list_rows_keep_their_leaf_identity():
    """Collapsing to the array name would make every cell in a Loss Run table an
    alias for `claims`, which is noise rather than evidence."""
    report = DA.derive_aliases(_fixture_documents())
    assert "claims" not in report.aliases
    assert any(field.startswith("claims[].") for field in report.aliases)


def test_confusables_are_derived_from_co_occurrence():
    """Evidence beats guessing: these are the distractors the real documents
    actually contain."""
    report = DA.derive_aliases(_fixture_documents())
    confusable = {label.casefold() for label in report.confusables["insured_name"]}
    assert "certificate holder" in confusable
    assert "producer" in confusable or "PRODUCER".casefold() in confusable


def test_provenance_is_backfilled_without_reannotation():
    report = DA.derive_aliases(_fixture_documents())
    assert report.provenance["policy_0001"]["named_insured.primary_name"] == "Applicant"
    assert report.provenance["policy_0002"]["named_insured.primary_name"] == "Named Insured"


def test_values_absent_from_their_document_are_reported():
    """The free QA pass. line_of_business is inferred, never printed on a page —
    so it *should* appear here, and that is the correct behaviour."""
    report = DA.derive_aliases(_fixture_documents())
    unresolved_fields = {u["field"] for u in report.unresolved}
    assert "line_of_business" in unresolved_fields


def test_low_entropy_values_are_skipped_not_guessed():
    """A value of "CA" or "3" matches everywhere and produces noise."""
    report = DA.derive_aliases(_fixture_documents())
    assert report.skipped_low_entropy
    assert all(len(s["value"]) < DA.MIN_VALUE_CHARS for s in report.skipped_low_entropy)


def test_markdown_emphasis_is_not_part_of_the_label():
    assert DA.clean_label("**Named Insured:**") == "Named Insured"
    assert DA.clean_label("| Policy No. |") == "Policy No"
    assert DA.clean_label("Applicant .........") == "Applicant"


def test_proposal_carries_evidence_and_never_overwrites(tmp_path):
    """High-quality candidate generation, not ground truth. A human confirms."""
    report = DA.derive_aliases(_fixture_documents())
    out = tmp_path / "policy.aliases.proposed.json"
    DA.write_proposal(report, str(out))

    proposed = json.loads(out.read_text(encoding="utf-8"))
    assert "DERIVED" in proposed["$comment"]
    assert "NEVER rendered into a prompt" in proposed["$comment"]

    entry = proposed["insured_name"]["aliases"][0]
    assert {"label", "documents", "confidence", "pattern", "examples"} <= set(entry)
    assert "$unresolved" in proposed        # the QA pass travels with the proposal


def test_a_broken_document_does_not_stop_the_batch():
    docs = _fixture_documents() + [("broken_0001", {"insured_name": object()}, "text")]
    report = DA.derive_aliases(docs)
    assert report.aliases["insured_name"], "good documents must still be processed"
