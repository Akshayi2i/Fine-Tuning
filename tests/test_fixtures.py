"""Fixture contract — the fixtures must exercise what later phases will test against.

Fixtures are pulled forward from IMPL-14 to Phase 1 because there is no labeled
corpus during the build: they are the *only* way to verify anything until real
data arrives. That makes their coverage a contract, not an incidental detail — a
fixture set that quietly stops covering confusables would let Phase 5 and Phase 7
pass while testing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common import aliases, ids, schemas
from common.canonical import field_paths
from common.lob import compute_coverage, validate_lob

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_DIR = FIXTURES / "golden"
OCR_DIR = FIXTURES / "ocr"


def _golden_files() -> list[Path]:
    return sorted(GOLDEN_DIR.glob("*.golden.json"))


def _meta_for(golden: Path) -> dict:
    meta = golden.with_name(golden.name.replace(".golden.json", ".label_metadata.json"))
    return json.loads(meta.read_text(encoding="utf-8"))


def _load(golden: Path) -> tuple[dict, dict, str, str | None]:
    label = json.loads(golden.read_text(encoding="utf-8"))
    meta = _meta_for(golden)
    return label, meta, meta["doc_type"], meta.get("acord_form")


def _lines_of(golden: Path) -> list[str]:
    """The document's lines: on a flat label its own field, on a canonical policy
    the metadata's `lob` — the client's schema has no top-level home for it."""
    label, meta, doc_type, form = _load(golden)
    if schemas.is_canonical(doc_type, form, meta.get("lob")):
        lob = meta.get("lob")
        return [lob] if isinstance(lob, str) else list(lob or [])
    return list(label["line_of_business"])


ALL_GOLDEN = _golden_files()


def test_fixtures_exist():
    assert ALL_GOLDEN, "no fixtures — every later phase has nothing to verify against"


@pytest.mark.parametrize("golden", ALL_GOLDEN, ids=lambda p: p.stem)
def test_golden_label_validates_against_its_schema(golden):
    """IMPL-04 rejects a label that does not validate; fixtures must pass that bar."""
    label, meta, doc_type, acord_form = _load(golden)
    schemas.validate(label, doc_type, acord_form, meta.get("lob"))


@pytest.mark.parametrize("golden", ALL_GOLDEN, ids=lambda p: p.stem)
def test_golden_label_carries_line_of_business(golden):
    """Every document records its line, even when empty (arch §0b): a flat label
    in its own field, a canonical policy in its metadata."""
    label, meta, doc_type, form = _load(golden)
    if schemas.is_canonical(doc_type, form, meta.get("lob")):
        # As the importer checks it (export_golden_labels.validate_label): a line
        # that names a canonical schema, not a value of the flat labels' LOB enum.
        from common.lob import known_lines, lob_lines

        assert "lob" in meta, f"{golden.stem}: a canonical policy records its line in metadata"
        assert lob_lines(meta["lob"]) <= known_lines(), f"{golden.stem}: {meta['lob']!r} names no schema"
    else:
        assert "line_of_business" in label
        validate_lob(_lines_of(golden))


@pytest.mark.parametrize("golden", ALL_GOLDEN, ids=lambda p: p.stem)
def test_source_id_is_well_formed_and_matches_filename(golden):
    _label, meta, doc_type, _form = _load(golden)
    parsed = ids.parse_source_id(meta["source_id"])
    assert parsed.doc_type == doc_type
    assert golden.name.startswith(meta["source_id"])


@pytest.mark.parametrize("golden", ALL_GOLDEN, ids=lambda p: p.stem)
def test_field_provenance_never_names_a_confusable(golden):
    """The highest-value annotation check in the system (IMPL-04).

    A label claiming ``insured_name`` was found under "Certificate Holder" is the
    exact mistake that teaches the model to conflate distinct parties — and it
    would train perfectly happily.
    """
    _label, meta, doc_type, _form = _load(golden)
    for field, surface_label in (meta.get("field_provenance") or {}).items():
        # By leaf: the registry is keyed by canonical field name, and a
        # canonical provenance names a path (`named_insured.primary_name`).
        leaf = field.rsplit(".", 1)[-1]
        assert not aliases.is_confusable(doc_type, leaf, surface_label), (
            f"{golden.stem}: {field!r} is recorded as found under {surface_label!r}, "
            f"which is a registered CONFUSABLE for that field"
        )


@pytest.mark.parametrize("golden", ALL_GOLDEN, ids=lambda p: p.stem)
def test_every_provenance_field_exists_in_the_label(golden):
    label, meta, *_ = _load(golden)
    present = field_paths(label)
    for field in (meta.get("field_provenance") or {}):
        assert field in present, f"{golden.stem}: provenance names {field!r}, absent from the label"


# --------------------------------------------------------------------------
# Coverage the later phases depend on
# --------------------------------------------------------------------------

def test_fixtures_cover_all_active_doc_types():
    covered = {_meta_for(g)["doc_type"] for g in ALL_GOLDEN}
    assert covered == {"policy", "lossrun", "acord"}


def test_same_canonical_field_appears_under_two_surface_labels():
    """Without this, ``alias_accuracy`` has nothing to slice (IMPL-08).

    One label per field would let the model memorise label strings and still
    score perfectly, which is precisely the outcome the canonical-mapping design
    claims to avoid.
    """
    by_field: dict[tuple[str, str], set[str]] = {}
    for golden in ALL_GOLDEN:
        _label, meta, doc_type, _form = _load(golden)
        for field, surface in (meta.get("field_provenance") or {}).items():
            by_field.setdefault((doc_type, field), set()).add(surface)

    multi = {k: v for k, v in by_field.items() if len(v) > 1}
    assert multi, "no canonical field appears under two different surface labels"
    assert ("policy", "named_insured.primary_name") in multi, (
        "the named insured is the worked example throughout the specs — it must carry alias "
        "variety (on a canonical policy it is named_insured.primary_name)"
    )


def test_a_confusable_co_occurrence_fixture_exists():
    """Required corpus edge case (IMPL-05).

    Documents where a field and its confusables appear together are what teach
    the *boundary*. Without one, the corpus teaches the mapping and the model
    happily returns the certificate holder as the insured.
    """
    found = []
    for golden in ALL_GOLDEN:
        label, _meta, doc_type, _form = _load(golden)
        for field in ("insured_name",):
            if label.get(field) is None:
                continue
            for confusable_label in aliases.confusables_for(doc_type, field):
                canonical = aliases.canonical_for(doc_type, confusable_label)
                if canonical and canonical != field and label.get(canonical) is not None:
                    found.append((golden.stem, field, canonical))
    assert found, "no fixture places a canonical field beside one of its confusables"


def test_fixtures_include_a_legitimately_absent_field():
    """Null-handling is a required edge case — absent must mean null, not invented."""
    assert any(
        any(v is None for v in json.loads(g.read_text(encoding="utf-8")).values())
        for g in ALL_GOLDEN
    ), "no fixture exercises a legitimately absent field"


def test_fixtures_include_a_variable_length_list_field():
    lossrun = [g for g in ALL_GOLDEN if _meta_for(g)["doc_type"] == "lossrun"]
    assert lossrun, "no lossrun fixture — the list-field case is untested"
    label = json.loads(lossrun[0].read_text(encoding="utf-8"))
    assert len(label["claims"]) >= 2
    assert label["total_claims_reported"] == len(label["claims"]), (
        "the document-stated count must match the rows, or the row-completeness "
        "cross-check has no correct baseline to test against"
    )


def test_fixtures_cover_more_than_one_lob_value():
    # The LOB enum is the flat labels' own field (ACORD, Loss Run); a canonical
    # policy outputs no line_of_business - its line is the schema it is read with.
    values = [_lines_of(g) for g in ALL_GOLDEN
              if not schemas.is_canonical(*_load(g)[2:], _meta_for(g).get("lob"))]
    coverage = compute_coverage(values)
    assert len([v for v, c in coverage.counts.items() if c]) >= 2, (
        "a single LoB value across all fixtures makes per-value LoB accuracy untestable"
    )


@pytest.mark.parametrize("golden", ALL_GOLDEN, ids=lambda p: p.stem)
def test_ocr_fixture_exists_for_each_golden(golden):
    """``derive_aliases`` anchors golden values in OCR text — it needs both halves."""
    source_id = _meta_for(golden)["source_id"]
    assert list(OCR_DIR.glob(f"{source_id}_page_*.md")), f"no OCR fixture for {source_id}"
