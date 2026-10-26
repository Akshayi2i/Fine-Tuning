"""The lines that stay on self-contained schema files render exactly as frozen.

The common-model work (SPEC_21) changes code every policy runs through - the
schema registry, the model view, the templates, the section map, the target
builder. None of it may change what a self-contained line, ACORD form or Loss
Run renders: those prompts are what their adapters train and serve on.

Frozen in the standard format profile: the fast one writes a policy window
another way by design (configs/shared/format.yaml).
"""

import json

import pytest

from testing.render_prompts import FINGERPRINTS_PATH, old_style_fingerprints

pytestmark = pytest.mark.format_profile("standard")


def test_the_old_style_selectors_render_exactly_as_frozen():
    frozen = json.loads(FINGERPRINTS_PATH.read_text(encoding="utf-8"))
    current = old_style_fingerprints()

    assert set(current) == set(frozen), (
        f"selectors added {sorted(set(current) - set(frozen))}, "
        f"removed {sorted(set(frozen) - set(current))}"
    )
    changed = {
        key: sorted(part for part in frozen[key] if frozen[key][part] != current[key].get(part))
        for key in frozen
        if current[key] != frozen[key]
    }
    assert not changed, (
        f"these renderings changed: {changed}. If the change is deliberate, re-freeze with "
        "`python -m testing.render_prompts --write-fingerprints` and say why in the commit."
    )


def test_the_frozen_set_leaves_out_only_the_common_model_lines():
    from common.schemas import is_common_model, schema_selectors

    frozen = json.loads(FINGERPRINTS_PATH.read_text(encoding="utf-8"))
    whole = {key.split("#")[0] for key in frozen}
    for doc_type, form, lob in schema_selectors():
        from testing.render_prompts import stem

        if doc_type == "policy" and is_common_model(doc_type, form, lob):
            assert stem(doc_type, form, lob) not in whole
        else:
            assert stem(doc_type, form, lob) in whole
