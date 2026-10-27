"""The two field-review scripts read a common-model line (SPEC_21) through its bundle.

An overlay is a list of ``$ref`` pointers into the common model, so a script that
reads the overlay file as it lies on disk sees every block as one bare field and
none of the line's own fields. Both scripts feed a person's review - the alias
registry and the field-type proposal - and neither says when a field is missing,
so these tests pin which fields each one reaches. Old-style lines are read as
they always were.
"""

from __future__ import annotations

import json
import shutil

import pytest

from common import schemas as S
from scripts import derive_aliases_from_canonical as aliases
from scripts import propose_field_types as proposer

LINES = (
    "homeowners", "personal_auto", "dwelling_fire", "ocean_marine",
    "motorcycle", "recreational_vehicle", "personal_umbrella",
)


def _rows(lob):
    return {path: (kind, found) for path, kind, found in
            aliases.walk_bundle(S.load_schema("policy", None, lob))}


def _collected(monkeypatch, tmp_path, *stems):
    """``collect_schemas`` over just these canonical files."""
    for stem in stems:
        shutil.copy(aliases.CANONICAL / f"{stem}.json", tmp_path / f"{stem}.json")
    monkeypatch.setattr(aliases, "CANONICAL", tmp_path)
    return aliases.collect_schemas()


# --------------------------------------------------------------------------
# derive_aliases_from_canonical
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lob", LINES)
def test_an_overlay_is_versioned_as_the_bundle_it_loads_as(monkeypatch, tmp_path, lob):
    """An overlay has no ``fideon:source``; read raw, every line said "unknown"."""
    _, versions = _collected(monkeypatch, tmp_path, lob)
    assert versions[lob] == S.load_schema("policy", None, lob)[S.BUNDLE_VERSION_KEY]


def test_an_overlay_reaches_the_lines_fields_not_just_its_block_names():
    rows = _rows("personal_auto")
    # Blocks are sections and tables, as in an old-style file, never leaves.
    assert rows["carrier"][0] == "section" and rows["coverages"][0] == "table"
    # A plain value and a typed one are each one field, not raw/parsed/page_ref.
    assert rows["vehicles[].vin"][0] == "leaf"
    assert rows["policy.effective_date"][0] == "leaf"
    assert rows["coverages[].limits[].amount"][0] == "leaf"
    assert not [p for p in rows if p.endswith((".raw", ".parsed"))]
    # Provenance is not a field.
    assert not [p for p in rows if "fideon:" in p]


def test_both_places_spec21_keeps_aliases_are_read():
    """The common model's own aliases, and the overlay's, keyed by field path."""
    bundle = S.load_schema("policy", None, "personal_auto")
    kind, found = _rows("personal_auto")["carrier.name"]
    assert "Insurer" in found and "Carrier" in found  # common model
    assert set(bundle["fideon:aliases"]["carrier.name"]) <= set(found)  # overlay
    # A block's aliases sit on its definition.
    assert "Schedule of Coverages" in _rows("personal_auto")["coverages"][1]


def test_an_overlay_alias_for_a_field_the_line_lacks_is_refused():
    """Dropped quietly, the alias would vanish from the registry with no trace."""
    bundle = {
        "fideon:lob": "test_line",
        "fideon:aliases": {"carrier.nmae": ["Insurer"]},
        "properties": {"carrier": {"$ref": "#/$defs/carrier"}},
        "$defs": {
            "carrier": {"type": "object", "properties": {"name": {"$ref": "#/$defs/FieldValue"}}},
            "FieldValue": {"type": "object", "properties": {
                "raw": {}, "parsed": {}, "page_ref": {}}},
        },
    }
    with pytest.raises(ValueError, match="carrier.nmae"):
        aliases.walk_bundle(bundle)


def test_a_definition_that_contains_itself_is_recorded_once_not_followed_forever():
    bundle = {
        "properties": {"party": {"$ref": "#/$defs/Party"}},
        "$defs": {"Party": {"type": "object", "properties": {
            "name": {"type": "string"}, "parent": {"$ref": "#/$defs/Party"}}}},
    }
    paths = [path for path, _, _ in aliases.walk_bundle(bundle)]
    assert paths == ["party", "party.name", "party.parent"]


@pytest.mark.parametrize("stem", ["equipment_breakdown", "classic_auto"])
def test_old_style_files_are_read_as_they_lie_on_disk(monkeypatch, tmp_path, stem):
    """``classic_auto`` is personal auto to the loader, but its own file is
    self-contained: bundling it as personal auto's overlay would version it and
    list its fields as the common model's."""
    fields, versions = _collected(monkeypatch, tmp_path, stem)
    raw = json.loads((tmp_path / f"{stem}.json").read_text(encoding="utf-8"))
    assert versions[stem] == raw["fideon:source"]["version"]
    line_block = raw["fideon:source"].get("line_specific_block")
    expected = {aliases.unify(path, line_block) for path, _, _ in aliases.walk(raw)}
    got = {path for entry in fields.values() for path in entry["paths"]}
    assert got == {p for p in expected if aliases.field_name(p)}


def test_the_fallback_is_read_through_the_bundle_it_loads_as(monkeypatch, tmp_path):
    """The fallback is no line's file, but since common model 1.1.0 it is an
    overlay like the lines': read as it lies on disk it is versioned "unknown"
    and gives the registry its block names as fields, none of its own."""
    fields, versions = _collected(monkeypatch, tmp_path, "_fallback")
    bundle = S.load_schema("policy", None, None)
    assert bundle.get(S.COMMON_MODEL_VERSION_KEY)
    assert versions["_fallback"] == bundle[S.BUNDLE_VERSION_KEY]
    expected = {aliases.unify(path, None) for path, _, _ in aliases.walk_bundle(bundle)}
    got = {path for entry in fields.values() for path in entry["paths"]}
    assert got == {p for p in expected if aliases.field_name(p)}


# --------------------------------------------------------------------------
# propose_field_types
# --------------------------------------------------------------------------

def test_every_common_model_field_has_a_row_to_review():
    """A row written as variants (``Limit``, ``Deductible``) and a typed value
    (``MoneyValue``, ``DateValue``) were both skipped: exactly the money and date
    fields whose type matters most."""
    fields = proposer.schema_fields()
    for path in ("coverages.limits.amount", "coverages.limits.description",
                 "deductibles.peril", "policy.effective_date", "premium.total"):
        assert path in fields, path
    assert not [p for p in fields if p.endswith((".raw", ".parsed"))]


def _walked_as_before(node, defs, path, out, depth=0):
    """The walk an old-style line has always had: a ``FieldValue`` is a field,
    nothing else is followed but references, items and properties."""
    if depth > 25 or not isinstance(node, dict):
        return
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name == "FieldValue":
            out.add(path)
            return
        node = defs.get(name, {})
    if node.get("type") == "array":
        _walked_as_before(node.get("items", {}), defs, path, out, depth + 1)
        return
    for key, value in (node.get("properties") or {}).items():
        _walked_as_before(value, defs, f"{path}.{key}" if path else key, out, depth + 1)


def test_an_old_style_lines_fields_are_unchanged():
    old_style = tuple(sorted(
        path.stem for path in aliases.CANONICAL.glob("*.json")
        if not S.is_common_model("policy", None, path.stem)
    ))
    assert "equipment_breakdown" in old_style and "homeowners" not in old_style
    expected: set[str] = set()
    for lob in old_style:
        schema = S.resolved_schema("policy", None, lob, None)
        _walked_as_before(schema, schema.get("$defs", {}), "", expected)
    assert proposer.schema_fields(old_style) == expected
