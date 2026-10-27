"""Labels written for one line's schema, used under another's (common.label_mapping).

No mapping is configured today, so the engine is tested against one given here:
a line whose labels keep their equipment under another line's block.
"""

from __future__ import annotations

import pytest

from common import label_mapping
from common.label_mapping import map_label


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


#: The line-block machinery is the self-contained files' (the common-model lines
#: have no line block - commercial auto among them now), so a self-contained line
#: is the target: agriculture/farm, whose labels here keep their equipment and
#: livestock under a foreign "ranch" block.
MAPPING = {"agriculture_farm": {"from": "ranch", "to": "farm", "sections": {
    "machinery": {"to": "scheduled_farm_equipment", "rename": {"body_type": "body_style_unknown_to_schema"}},
    "herds": {"to": "livestock"},
}}}

LABEL = {
    "policy": {"policy_number": _env("FO-1")},
    "ranch": {"machinery": [{"serial_number": _env("1HD1KEF18KB668131"), "year": _env(2019),
                             "engine_displacement": _env("1868")}],
              "herds": [{"livestock_type": _env("Dairy cattle")}]},
}


@pytest.fixture(autouse=True)
def mapping(monkeypatch):
    monkeypatch.setattr(label_mapping, "_mappings", lambda: MAPPING)


def test_the_source_block_moves_into_the_line_block():
    mapped = map_label(LABEL, ["agriculture_farm"])
    assert mapped["farm"]["scheduled_farm_equipment"][0]["serial_number"]["raw"] == "1HD1KEF18KB668131"
    assert mapped["farm"]["livestock"][0]["livestock_type"]["raw"] == "Dairy cattle"
    assert "ranch" not in mapped and mapped["policy"] == LABEL["policy"]


def test_fields_the_target_schema_lacks_are_dropped_not_guessed():
    equipment = map_label(LABEL, ["agriculture_farm"])["farm"]["scheduled_farm_equipment"][0]
    assert "engine_displacement" not in equipment and "year" in equipment


def test_the_stored_label_is_never_changed():
    map_label(LABEL, ["agriculture_farm"])
    assert "ranch" in LABEL and "farm" not in LABEL


def test_other_lines_and_labels_already_in_the_target_format_are_untouched():
    assert map_label(LABEL, ["gl"]) is LABEL
    native = {"farm": {"scheduled_farm_equipment": [{"serial_number": _env("X")}]}, "ranch": {"machinery": []}}
    assert map_label(native, ["agriculture_farm"]) is native


def test_no_mapping_is_configured():
    """Classic auto was the one mapped line; it is personal auto now."""
    from pathlib import Path

    import yaml

    assert not yaml.safe_load(Path(label_mapping.CONFIG).read_text(encoding="utf-8"))
