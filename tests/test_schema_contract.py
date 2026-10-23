"""IMPL-01 acceptance criteria for the schema and prompt contracts.

These guard failures that are otherwise silent — a schema that stops requiring
``line_of_business``, a field whose gloss goes missing, an alias string leaking
into a prompt. None of them would surface as an exception at training time; they
would surface as a worse model, weeks later, with no obvious cause.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from common import aliases, prompts, schemas
from common.config import sequence_for_task
from common.constants import ACORD_FORMS, ACTIVE_DOC_TYPES, MODALITY_MODES, canonical_key, canonical_model
from common.lob import lob_values

ROOT = Path(__file__).resolve().parent.parent

#: The flat schemas: one object, scalar leaves, every key emitted with `null` for
#: absence, and `line_of_business` a top-level required list.
EXAMPLE_STEMS = {
    ("lossrun", None): "lossrun",
    ("acord", "25"): "acord25",
    ("acord", "125"): "acord125",
    ("acord", "140"): "acord140",
}
FLAT_KEYS = list(EXAMPLE_STEMS)

#: The client's canonical schemas: a nested tree whose every leaf is a
#: `FieldValue` envelope, emitted sparsely. `policy` is one of these now — the
#: bare key resolves to their `_fallback.json`, not the old flat policy schema.
CANONICAL_KEYS = [("policy", None)]

ALL_KEYS = FLAT_KEYS + CANONICAL_KEYS


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_schema_loads_and_no_ref_points_outside_the_document(doc_type, acord_form):
    """A ref the model cannot follow must be inlined. A ref it can, need not be.

    This asserted that NO ``$ref`` survived, which was right while every schema
    ``$ref``-ed another file: ``common_fields.json#/$defs/insured_name`` means
    nothing to a model that was never shown that file.

    The canonical schemas ``$ref`` only ``#/$defs/FieldValue``, in their own
    document, and ``$defs`` travels with the schema into the prompt — so the
    model can already see what it points at. Inlining those would restate one
    480-character envelope once per leaf: 532 times in homeowners, turning a 42k
    schema into 281k characters of prompt.
    """
    resolved = schemas.resolved_schema(doc_type, acord_form)
    assert resolved["type"] == "object"
    assert json.dumps(resolved), "resolved schema must be JSON-serialisable for prompt injection"

    for ref in re.findall(r'"\$ref"\s*:\s*"([^"]+)"', json.dumps(resolved)):
        assert ref.startswith("#/"), f"{ref!r} points outside the document the model is shown"
        pointer = resolved
        for step in ref[2:].split("/"):
            assert isinstance(pointer, dict) and step in pointer, f"{ref!r} does not resolve"
            pointer = pointer[step]


@pytest.mark.parametrize("doc_type,acord_form", FLAT_KEYS)
def test_every_field_has_a_description(doc_type, acord_form):
    """Descriptions are prompt text (arch §0c), not documentation.

    A field without one gives the model no semantic anchor, which is exactly what
    the canonical-mapping design depends on.
    """
    schemas.assert_all_fields_described(doc_type, acord_form)


@pytest.mark.parametrize("doc_type,acord_form", FLAT_KEYS)
def test_line_of_business_is_a_required_list_of_enum_values(doc_type, acord_form):
    """The VLM is the fallback LoB detector when L1/L2 miss (arch §0b), and it is
    a LIST under v2.1: a certificate or package policy routinely covers several
    lines, and the v1 scalar forced the annotator to pick one and discard the
    rest — which taught the model to do the same."""
    resolved = schemas.resolved_schema(doc_type, acord_form)
    lob = resolved["properties"]["line_of_business"]
    assert "line_of_business" in resolved["required"]
    assert lob["type"] == "array", "a document can cover several lines"
    assert set(lob["items"]["enum"]) == set(lob_values())
    assert lob.get("uniqueItems") is True, "the same line twice is a labelling error"


@pytest.mark.parametrize("doc_type,acord_form", FLAT_KEYS)
def test_an_empty_lob_list_is_valid_and_means_undetermined(doc_type, acord_form):
    """Undetermined is a correct answer, not a missing one. Under v1 that was
    `null`; under a list it is `[]`, and it must stay representable or the model
    learns to guess a line rather than decline."""
    path = ROOT / "schemas" / "examples" / f"{EXAMPLE_STEMS[(doc_type, acord_form)]}.example.json"
    inst = json.loads(path.read_text(encoding="utf-8"))
    inst["line_of_business"] = []
    schemas.validate(inst, doc_type, acord_form)


def test_lines_outside_the_enum_go_to_their_own_field():
    """Inland marine, cyber, crime and EPLI are real lines this enum does not
    support. Recorded rather than dropped, because the count of what lands here
    is the evidence for whether the enum should grow — and it is a list, because
    a document can name several (v2.1 correction)."""
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business_other"] = ["inland_marine", "cyber"]
    schemas.validate(inst, "lossrun")


@pytest.mark.parametrize("doc_type,acord_form", FLAT_KEYS)
def test_schema_validates_its_example(doc_type, acord_form):
    path = ROOT / "schemas" / "examples" / f"{EXAMPLE_STEMS[(doc_type, acord_form)]}.example.json"
    schemas.validate(json.loads(path.read_text(encoding="utf-8")), doc_type, acord_form)


def test_schema_rejects_out_of_enum_lob():
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business"] = ["marine_cargo"]
    assert not schemas.is_valid(inst, "lossrun")


def test_schema_rejects_the_v1_scalar_shape():
    """A corpus row still carrying the v1 scalar must fail loudly here rather
    than reaching training, where it would be one silently malformed target."""
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business"] = "workers_comp"
    assert not schemas.is_valid(inst, "lossrun")


def test_schema_rejects_a_duplicated_line():
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    inst["line_of_business"] = ["property", "property"]
    assert not schemas.is_valid(inst, "lossrun")


def test_schema_rejects_unexpected_field():
    """A surface label arriving as a KEY, rather than as a value under one.

    Read from the lossrun example rather than the policy one: `policy` now
    resolves to the client's canonical schema, so a flat policy example is
    invalid before the extra key is added and the test would pass without
    testing anything.
    """
    inst = json.loads((ROOT / "schemas/examples/lossrun.example.json").read_text(encoding="utf-8"))
    assert schemas.is_valid(inst, "lossrun"), "the example must be valid before we break it"
    inst["surface_label_leaked_in"] = "Applicant"
    assert not schemas.is_valid(inst, "lossrun")


def test_shared_fields_do_not_drift_between_schemas():
    """The whole reason ``common_fields.json`` exists.

    Defining ``insured_name`` per-schema lets two definitions of the same
    canonical field diverge, and nothing catches it — each schema still validates
    on its own. This is the test that would.

    Scoped to the flat schemas, because they are the ones that share that file.
    The canonical schemas are the client's, pre-merged by their own registry from
    their own ``_common``; they do not ``$ref`` ours and have no field named
    ``insured_name``. Holding them to a common file they never referenced would
    fail on a difference that is not drift.
    """
    per_schema = {
        key: dict(schemas.iter_described_fields(schemas.resolved_schema(*key)))
        for key in FLAT_KEYS
    }
    shared = set.intersection(*(set(d) for d in per_schema.values()))
    assert {"insured_name", "line_of_business"} <= shared

    for field in shared:
        descriptions = {json.dumps(d[field], sort_keys=True) for d in per_schema.values()}
        assert len(descriptions) == 1, f"{field!r} has drifted between schemas: {descriptions}"


def test_acord_requires_a_form_to_select_a_schema():
    """Two-level classification (arch §4b) — the form picks the schema."""
    with pytest.raises(schemas.SchemaError, match="acord_form"):
        schemas.load_schema("acord")
    with pytest.raises(schemas.SchemaError, match="unknown ACORD form"):
        schemas.load_schema("acord", "999")


# --------------------------------------------------------------------------
# Prompts — the parity-critical path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_noisy_ocr_renders_identically_to_ocr_plus_image(doc_type, acord_form):
    """The noise lives in the data, not the instruction (arch §6).

    A distinct prompt for noisy OCR would teach the model that bad OCR is
    announced — and at inference nothing announces it.
    """
    a = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    b = prompts.render_system_prompt(doc_type, "noisy_ocr_image", acord_form)
    assert a == b


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_image_only_prompt_declares_the_absence(doc_type, acord_form):
    """Explicit declaration, not a silently missing block (arch §6).

    Asserts the property rather than one sentence: the image-only prompt must
    say that no OCR is given, **and** must not carry the OCR-precedence
    instructions. A prompt that merely dropped the OCR block would leave the
    model with no statement about what it is reading from.
    """
    rendered = prompts.render_system_prompt(doc_type, "image_only", acord_form)
    ocr_mode = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)

    assert "no OCR text" in rendered
    assert "primary source" not in rendered, "image-only carries OCR-precedence rules"
    assert "primary source" in ocr_mode
    assert rendered != ocr_mode


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_no_alias_string_reaches_the_prompt(doc_type, acord_form):
    """The alias registry never goes in (master §1.4).

    An alias list in the prompt gives the model a lexical prior that makes
    confusable errors worse, and would make every newly observed alias a retrain
    trigger. Covers the canonical schemas too, which is where the risk is real:
    they carry ``fideon:aliases`` inline, and only the strip in
    ``resolved_schema`` keeps those out of the prompt.
    """
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    registry_doc_type = "acord" if doc_type == "acord" else doc_type
    # The prompt embeds the schema, so anything the SCHEMA says reaches the model
    # by construction — a field description mentioning "Hired Auto" is not the
    # alias registry leaking in. A phrase in neither means someone pasted it.
    embedded = schemas.schema_text(doc_type, acord_form)
    for field, entry in aliases.load_registry(registry_doc_type).items():
        for alias in entry.aliases:
            # A one-word alias may legitimately collide with ordinary prose
            # ("Insured", "Company"); multi-word aliases are unambiguous.
            if " " in alias and alias not in embedded:
                assert alias not in rendered, f"alias {alias!r} leaked into the prompt for {field}"


@pytest.mark.parametrize("doc_type,acord_form", FLAT_KEYS)
def test_field_descriptions_reach_the_model(doc_type, acord_form):
    """The gloss is how the model maps an unseen surface label onto a canonical
    key (arch §0c) — it is the thing that stands in for the alias list the prompt
    is forbidden to carry."""
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    resolved = schemas.resolved_schema(doc_type, acord_form)
    some_description = resolved["properties"]["line_of_business"]["description"]
    assert some_description[:40] in rendered, "field descriptions must reach the model"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The client's canonical schemas carry 5 descriptions across 532 fields, so the prompt's "
        "own instruction — 'Each field's description in the schema is its definition' — is true "
        "of almost none of them. Closed by the descriptions sidecar "
        "(schemas/descriptions/{lob}.json, merged at render time, their files never edited). "
        "strict=True so this FAILS once the sidecar lands, forcing the marker off."
    ),
)
@pytest.mark.parametrize("lob", schemas.SNAPSHOT_LINES)
def test_every_canonical_field_has_a_description(lob):
    schemas.assert_all_fields_described("policy", None, lob)


@pytest.mark.parametrize("modality_mode", MODALITY_MODES)
def test_prompt_rendering_is_deterministic(modality_mode):
    """Prompt parity is meaningless if rendering is not reproducible."""
    a = prompts.render_system_prompt("policy", modality_mode)
    b = prompts.render_system_prompt("policy", modality_mode)
    assert a == b
    assert prompts.prompt_fingerprint("policy", modality_mode) == prompts.prompt_fingerprint("policy", modality_mode)


def test_unknown_modality_mode_raises():
    with pytest.raises(prompts.PromptError, match="unknown modality_mode"):
        prompts.render_system_prompt("policy", "ocr_only")


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

def test_doc_type_to_canonical_mapping():
    """master §1.2 — the two vocabularies resolve in exactly one place."""
    assert canonical_key("lossrun") == "loss_run"
    assert canonical_model("lossrun") == "LossRunDocument"
    assert canonical_key("policy") == "policy_check"
    assert canonical_key("acord") == "acord_mapping"


def test_deferred_doc_type_is_refused():
    """Quote is designed for but explicitly not implemented (master §1)."""
    with pytest.raises(KeyError, match="deferred"):
        canonical_key("quote")


def test_active_types_and_forms_are_what_the_specs_say():
    assert set(ACTIVE_DOC_TYPES) == {"acord", "policy", "lossrun"}
    assert {"25", "125", "140"} == ACORD_FORMS


# --------------------------------------------------------------------------
# Prompt structure — invariants, not prose
# --------------------------------------------------------------------------

PROMPT_SECTIONS = ("# Output", "# Resolving fields", "# Distinct roles",
                   "# Value formats", "# Absence", "# Repeating tables")


@pytest.mark.parametrize("modality_mode", ["ocr_plus_image", "image_only"])
def test_every_rule_section_survives_in_both_modes(modality_mode):
    """Only the Evidence block is modality-specific. A rule that exists in one
    mode and not the other trains two different behaviours for one task."""
    rendered = prompts.render_system_prompt("policy", modality_mode)
    for section in PROMPT_SECTIONS:
        assert section in rendered, f"{section} missing from the {modality_mode} prompt"


def test_the_two_modes_differ_only_in_the_evidence_block():
    """Guards the `{% if %}` branch: editing one arm and not the other is how
    the modes silently diverge on a rule that has nothing to do with OCR."""
    ocr = prompts.render_system_prompt("policy", "ocr_plus_image")
    image = prompts.render_system_prompt("policy", "image_only")

    assert ocr.split("# Output", 1)[1] == image.split("# Output", 1)[1]
    assert ocr.split("# Evidence", 1)[0] == image.split("# Evidence", 1)[0]


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_schema_is_the_last_thing_in_the_prompt(doc_type, acord_form):
    """Instructions before data. The schema is the largest block by far, and
    rules placed after it are read as commentary on it."""
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    assert rendered.index("Schema:") > rendered.index("# Repeating tables")


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_prompt_leaves_room_for_the_document(doc_type, acord_form):
    """The prompt is a fixed cost on every row; the document is what varies.

    This used to be a flat ``< 4000`` — a number with no derivation, which said
    nothing about whether a prompt actually fit. Now it is arithmetic against the
    task's own budget: whatever the prompt spends, the pages and their OCR have
    to fit in what is left, or the row is rejected at corpus-build time.

    ``PAGES`` is the routed-page count the policy budget is written for, and OCR
    is charged at a deliberately thin ~700 tokens a page. Thin because this is a
    floor, not a forecast: a prompt that fails here cannot carry a short
    document, let alone a real one.
    """
    from data_pipeline.dataset_builder.cap_check import (
        CHARS_PER_TOKEN,
        TEMPLATE_OVERHEAD_TOKENS,
        estimate_visual_tokens,
    )

    PAGES, OCR_PER_PAGE = 4, 700
    budget = sequence_for_task("extract", doc_type)
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    prompt_tokens = len(rendered) / CHARS_PER_TOKEN

    spent = (
        prompt_tokens
        + estimate_visual_tokens(PAGES, "extract")
        + PAGES * OCR_PER_PAGE
        + TEMPLATE_OVERHEAD_TOKENS
        + budget["max_output_tokens"]
    )
    assert spent < budget["max_seq_len"], (
        f"{doc_type}{'/' + acord_form if acord_form else ''} spends ~{spent:.0f} of "
        f"{budget['max_seq_len']} tokens on a {PAGES}-page document, of which the prompt alone is "
        f"~{prompt_tokens:.0f}. The prompt is paid on every row — slice the schema per window "
        "rather than raising the cap."
    )


@pytest.mark.parametrize("lob", schemas.SNAPSHOT_LINES)
def test_each_canonical_line_leaves_room_for_the_document(lob):
    """The same arithmetic per line of business, which is where it actually bites.

    A canonical schema is an order of magnitude larger than the flat one it
    replaced, and it is the whole schema that goes into the prompt. Nothing else
    covers the per-LOB prompts, so without this the first evidence that a line
    does not fit would be a corpus build rejecting every one of its documents.
    """
    from data_pipeline.dataset_builder.cap_check import (
        CHARS_PER_TOKEN,
        TEMPLATE_OVERHEAD_TOKENS,
        estimate_visual_tokens,
    )

    PAGES, OCR_PER_PAGE = 4, 700
    budget = sequence_for_task("extract", "policy")
    prompt_tokens = (
        len(prompts.render_system_prompt("policy", "ocr_plus_image", lob=lob)) / CHARS_PER_TOKEN
    )
    spent = (
        prompt_tokens
        + estimate_visual_tokens(PAGES, "extract")
        + PAGES * OCR_PER_PAGE
        + TEMPLATE_OVERHEAD_TOKENS
        + budget["max_output_tokens"]
    )
    assert spent < budget["max_seq_len"], (
        f"a {PAGES}-page {lob} policy spends ~{spent:.0f} of {budget['max_seq_len']} tokens, "
        f"of which the prompt alone is ~{prompt_tokens:.0f}. Slice the schema per window."
    )


# --------------------------------------------------------------------------
# Per-document-type rules (IMPL-12 §2)
# --------------------------------------------------------------------------

DOC_TYPE_BLOCK = "# This document type"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_every_type_carries_its_own_rules_block(doc_type, acord_form):
    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    assert DOC_TYPE_BLOCK in rendered


def test_every_active_doc_type_has_a_rules_file():
    """A missing partial raises TemplateNotFound at render time rather than
    rendering an empty block — but only if someone renders that type. This
    fails at the moment a type is added instead."""
    from common.constants import ACTIVE_DOC_TYPES

    for doc_type in ACTIVE_DOC_TYPES:
        assert (prompts.PROMPT_DIR / "doc_types" / f"{doc_type}.jinja").exists(), (
            f"{doc_type} has no per-type rules block"
        )


def test_the_type_blocks_are_actually_different():
    """Three identical blocks would be three copies of the shared rules with
    extra tokens and no per-type guidance."""
    blocks = {
        doc_type: prompts.render_system_prompt(doc_type, "ocr_plus_image",
                                               "25" if doc_type == "acord" else None)
                        .split(DOC_TYPE_BLOCK, 1)[1].split("Schema:", 1)[0]
        for doc_type in ("policy", "lossrun", "acord")
    }
    assert len(set(blocks.values())) == 3


def test_each_acord_form_gets_its_own_paragraph():
    """One adapter, but a distinct schema per form (arch §4b) — so the prose
    describing the form must move with the schema, not with the type."""
    rendered = {
        form: prompts.render_system_prompt("acord", "ocr_plus_image", form)
        for form in ("25", "125", "140")
    }
    bodies = {f: t.split(DOC_TYPE_BLOCK, 1)[1].split("Schema:", 1)[0] for f, t in rendered.items()}
    assert len(set(bodies.values())) == 3


def test_the_form_paragraph_travels_with_the_form_schema():
    """The same `acord_form` selects both, so a prompt cannot describe one form
    while carrying another's schema."""
    for form in ("25", "125", "140"):
        rendered = prompts.render_system_prompt("acord", "ocr_plus_image", form)
        schema = rendered.split("Schema:", 1)[1]
        assert f"acord{form}" in schema or f"acord_{form}" in schema or form in schema[:400]


@pytest.mark.parametrize("modality_mode", ["ocr_plus_image", "image_only"])
def test_the_type_block_is_identical_across_modalities(modality_mode):
    """Type structure does not change with the evidence available."""
    reference = prompts.render_system_prompt("lossrun", "ocr_plus_image")
    rendered = prompts.render_system_prompt("lossrun", modality_mode)
    assert (
        rendered.split(DOC_TYPE_BLOCK, 1)[1] == reference.split(DOC_TYPE_BLOCK, 1)[1]
    )


def test_the_stated_count_rule_survives_in_the_lossrun_block():
    """IMPL-09's row-completeness check compares the model's row count against a
    count the *document* states. A model that computes that count from its own
    rows makes the cross-check compare a number to itself, and the one signal
    that can see a dropped row goes silent."""
    rendered = prompts.render_system_prompt("lossrun", "ocr_plus_image")
    block = rendered.split(DOC_TYPE_BLOCK, 1)[1]
    assert "never computed from the rows you produced" in block


TYPE_BLOCK_SECTIONS = ("## What you are given", "## Where the values are",
                       "## How to work through it", "## Rules for this document type",
                       "## Output for this document type")


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_each_type_block_carries_the_full_set_of_sections(doc_type, acord_form):
    """Input shape, value locations, an ordered procedure, the binding rules and
    the output shape — every type answers all five, or the types are not
    comparable to each other."""
    block = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form) \
                   .split(DOC_TYPE_BLOCK, 1)[1]
    for section in TYPE_BLOCK_SECTIONS:
        assert section in block, f"{doc_type} {acord_form or ''} is missing {section}"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_output_shape_is_derived_from_the_schema(doc_type, acord_form):
    """A hand-written key list in a prompt file drifts the first time a field is
    added, and the drift is silent: the model is shown one shape and validated
    against another."""
    from common.schemas import resolved_schema

    rendered = prompts.render_system_prompt(doc_type, "ocr_plus_image", acord_form)
    shape = prompts.output_shape_for_prompt(doc_type, acord_form)

    assert shape in rendered
    for key in resolved_schema(doc_type, acord_form).get("properties", {}):
        assert key in shape, f"{key} is in the schema but missing from the shown output shape"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_output_shape_names_every_row_key_of_every_array(doc_type, acord_form):
    from common.schemas import resolved_schema

    shape = prompts.output_shape_for_prompt(doc_type, acord_form)
    for name, node in resolved_schema(doc_type, acord_form).get("properties", {}).items():
        for row_key in (node.get("items") or {}).get("properties", {}):
            assert row_key in shape, f"{name}[].{row_key} is missing from the shown output shape"


@pytest.mark.parametrize("doc_type,acord_form", ALL_KEYS)
def test_the_output_shape_is_an_outline_not_a_second_schema(doc_type, acord_form):
    """Shape only — no types, no descriptions, no JSON syntax. The schema
    follows immediately after and is the authority on both; a second copy here
    would double the token cost to say the same thing twice.

    Asserted structurally rather than by keyword, because JSON Schema keywords
    collide with real field names — `description` is a genuine key on a claim row.
    """
    shape = prompts.output_shape_for_prompt(doc_type, acord_form)
    schema = prompts.schema_json_for_prompt(doc_type, acord_form)

    assert '"' not in shape, "the shape is a bare key outline, not JSON"
    assert ":" not in shape.replace(": [", ""), "only array keys carry a colon"
    assert len(shape) < len(schema) / 3, "the outline has grown into a second schema"


# --------------------------------------------------------------------------
# Schema selection by line of business (arch v2.1 §0b)
# --------------------------------------------------------------------------


def test_a_policy_with_no_line_of_its_own_falls_back():
    """A line with no canonical file of its own resolves to the bare `policy`
    key — which is itself the client's `_fallback.json`, not the old flat schema.
    Their own note on that file says it is "used when the line is unknown or no
    line-specific schema exists", so the fallback is their rule, not ours."""
    from common.schemas import schema_key

    assert schema_key("policy") == "policy"
    assert schema_key("policy", None, "a_line_with_no_file") == "policy"
    assert schema_key("policy", None, ["a_line_with_no_file"]) == "policy"


def test_an_lob_selects_a_schema_only_when_one_is_registered():
    """The per-LOB schemas have landed, so this tests them rather than a stand-in.

    Previously this monkeypatched a fake entry into ``_SCHEMA_FILES`` to simulate
    the state after they arrived. They have arrived, and a test that still asserts
    against a fake would keep passing if the real registration broke.
    """
    from common import schemas

    assert schemas.schema_key("policy", None, "homeowners") == "policy:homeowners"
    assert schemas.schema_key("policy", None, ["ocean_marine"]) == "policy:ocean_marine"
    # Our LOB enum and the client's filenames disagree for a few lines, so the
    # enum value is translated rather than looked up directly. Without that a
    # Workers' Comp policy finds no `workers_comp.json` and silently falls back.
    assert schemas.schema_key("policy", None, "workers_comp") == "policy:wc"
    # The client's file for commercial auto is auto.json (SPEC_21 delivery).
    assert schemas.schema_key("policy", None, "commercial_auto") == "policy:auto"


def test_a_package_policy_uses_the_generic_schema_rather_than_one_of_its_lines():
    """A policy covering GL, Property and Auto is one document with a section per
    line. Picking one line's schema would validate the whole document against a
    third of itself."""
    from common import schemas

    assert schemas.schema_key("policy", None, ["homeowners", "personal_auto"]) == "policy"


# --------------------------------------------------------------------------
# The client's canonical schemas (configs/canonical schema/policy_check)
# --------------------------------------------------------------------------

#: The flat schemas — the ones the canonical files did NOT replace. Hashed so
#: that a change to the canonical branch of `resolved_schema` cannot move them
#: unnoticed: every prompt fingerprint and corpus schema pin is downstream of
#: this text, and a silent shift would invalidate a corpus with nothing to say so.
#:
#: `policy` is deliberately absent. It no longer names the flat
#: `policy_doc.schema.json`; a policy's output contract is now the client's
#: canonical JSON whatever its line, so the generic key resolves to their
#: `_fallback.json`. Pinning it here would pin the shape we just replaced.
_FLAT_SCHEMA_TEXT = {
    ("lossrun", None): "bfa81452535699ba295d9c99bf9d5d52e9cfe7fd14e66403a31d50ca0ab7d7d4",
    ("acord", "25"): "a6581f7783d69d6867806df89080cc705b388c27cbe5517fbb559bb25c3b4ade",
    ("acord", "125"): "62b47a3628cc84a96addf4aa0093414e327c3486bf85b09e4bd818a6266a124a",
    ("acord", "140"): "ba327b5db7e8b2b02cc82ef19f29c891bda4dbc131f608eb51df43c16895b5ce",
}


def test_the_flat_schemas_are_not_moved_by_the_canonical_branch():
    """`resolved_schema` grew a branch for the canonical files. If that branch
    touched the other path by so much as a key order, every prompt fingerprint
    and every corpus schema pin would move and no other test would say why."""
    import hashlib

    from common import schemas

    for (doc_type, form), want in _FLAT_SCHEMA_TEXT.items():
        got = hashlib.sha256(schemas.schema_text(doc_type, form).encode()).hexdigest()
        assert got == want, f"{doc_type}/{form} schema text changed"


def test_every_snapshot_line_has_a_registered_canonical_schema():
    """The lines whose prompts are pinned (common.schemas.SNAPSHOT_LINES) each
    have a canonical schema of their own.

    A pinned line without one would render the generic policy prompt, and the
    snapshot would review a prompt no document of that line is ever given.
    """
    from common.lob import known_lines
    from common.schemas import SNAPSHOT_LINES, schema_key

    # Classic auto is personal auto (common.lob.MERGED_LINES).
    assert len(SNAPSHOT_LINES) == len(set(SNAPSHOT_LINES)) == 7
    for lob in SNAPSHOT_LINES:
        assert schema_key("policy", None, lob) == f"policy:{lob}", (
            f"{lob} is a snapshot line but has no canonical schema, "
            "so it would silently fall back to the generic policy schema"
        )
    assert set(SNAPSHOT_LINES) <= known_lines()


def test_a_canonical_schema_is_not_ref_inlined():
    """Every leaf $refs #/$defs/FieldValue — 532 of them in homeowners.

    Inlining them restates the same 480-character envelope once per field, which
    turns a 42k-character schema into 281k characters of prompt: ~70k tokens
    saying one thing over and over, and far past any sequence budget.
    """
    from common.schemas import resolved_schema, schema_text

    schema = resolved_schema("policy", None, ["homeowners"])
    assert "$defs" in schema and "FieldValue" in schema["$defs"]
    assert "$ref" in json.dumps(schema), "refs were inlined"
    assert len(schema_text("policy", None, ["homeowners"])) < 60_000


def test_no_fideon_key_survives_into_the_schema_the_model_sees():
    """fideon:aliases records the printed labels carriers use for each field.

    That is corpus-analysis knowledge and it is barred from the prompt (master
    §1.4): a model handed the lookup table never learns the semantics, and the
    first label not in the table misses with nothing to signal it.

    The registry's own guard is an AST import check on `common.aliases`, which
    this walks straight past — these aliases arrive inside the client's schema
    file, not through that module.
    """
    from common.schemas import SNAPSHOT_LINES, load_schema, schema_text

    for lob in SNAPSHOT_LINES:
        assert "fideon:" not in schema_text("policy", None, lob), f"{lob} leaks fideon: keys"
        # …while the raw schema, which is what validation judges against, keeps them.
        assert "fideon:" in json.dumps(load_schema("policy", None, lob))


def test_a_canonical_schema_version_comes_from_fideon_source():
    """The self-contained files carry no top-level `version`; theirs is under
    `fideon:source`. A common-model overlay's is the pair: its own and the common
    model's, written onto the bundle. General liability and the fallback are
    overlays since common model 1.1.0.

    `schema_version` is called for every registered schema when a corpus manifest
    is written, so reading the wrong path fails the whole build.
    """
    from common.schemas import schema_version

    assert schema_version("policy", None, ["property"]) == "1.5.0"
    assert schema_version("policy", None, ["management_liability"]) == "3.1.0"
    assert schema_version("policy", None, ["homeowners"]) == "1.0.0+common.1.1.0"
    assert schema_version("policy", None, ["gl"]) == "3.0.0+common.1.1.0"
    assert schema_version("policy") == "1.1.0+common.1.1.0"
    # A line with no file of its own reads the fallback's.
    assert schema_version("policy", None, ["a_line_with_no_file"]) == schema_version("policy")


def test_a_canonical_schema_validates_the_fieldvalue_envelope():
    """Validation judges a label against the client's file byte for byte —
    `fideon:` keys and all — because that file is the contract, not our
    rendering of it."""
    from common.schemas import is_valid, required_fields

    fv = {
        "raw": "HO-1234", "parsed": "HO-1234",
        "confidence": {"score": 0.97, "source": "vlm"},
        "page_ref": [1], "flagged": False,
    }
    whole = {
        "carrier": {"company_name": fv},
        "named_insured": {"primary_name": fv},
        "policy": {"policy_number": fv},
    }
    assert is_valid(whole, "policy", None, ["property"])
    assert required_fields("policy", None, ["property"]) == [
        "carrier", "named_insured", "policy",
    ]

    # Assert on the REASON, not just the verdict. The generic policy schema
    # rejects this instance too, for its own unrelated reasons, so a bare
    # `not is_valid` would pass even if the canonical schema were never reached.
    from common.schemas import iter_validation_errors

    bare = {**whole, "carrier": {"company_name": {"raw": "X"}}}
    errors = list(iter_validation_errors(bare, "policy", None, ["property"]))
    assert any("'confidence' is a required property" in e for e in errors), (
        f"validation did not enforce the FieldValue envelope; got: {errors[:3]}"
    )


def test_unqualified_selectors_come_first():
    """Callers that scan until they find what they need should read the five small
    generic schemas before the eight large canonical ones."""
    from common.schemas import schema_selectors

    selectors = schema_selectors()
    qualified = [i for i, s in enumerate(selectors) if s[1] or s[2]]
    unqualified = [i for i, s in enumerate(selectors) if not (s[1] or s[2])]
    assert max(unqualified) < min(qualified)


def test_the_lob_reaches_the_rendered_prompt():
    """`lob` used to die in common.prompts: the schemas were selectable for
    validation and invisible to the model, so a homeowners document trained
    against the generic policy schema and nothing said so."""
    from common.prompts import render_system_prompt

    home = render_system_prompt("policy", "ocr_plus_image", lob="homeowners")
    auto = render_system_prompt("policy", "ocr_plus_image", lob="personal_auto")
    generic = render_system_prompt("policy", "ocr_plus_image")

    assert home != auto != generic
    # Each line's own common-model blocks: homeowners schedules items, auto vehicles.
    assert "scheduled_items" in home and '"vehicles"' not in home
    assert '"vehicles"' in auto and "scheduled_items" not in auto
    # A package policy names more than one line, so it still gets the generic one.
    assert render_system_prompt(
        "policy", "ocr_plus_image", lob=["homeowners", "personal_auto"]
    ) == generic


def test_the_outline_and_the_schema_describe_the_same_line():
    """They render into the same prompt a few lines apart. Given different lines
    they would disagree about the field set, and the model is told the schema is
    the authority — so it would be shown an outline it must not follow."""
    from common.prompts import output_shape_for_prompt, schema_json_for_prompt

    for lob in ("homeowners", "ocean_marine"):
        outline = output_shape_for_prompt("policy", None, lob)
        schema = schema_json_for_prompt("policy", None, lob)
        names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", outline))
        assert names, f"{lob}: outline named nothing"
        for key in sorted(names):
            assert f'"{key}"' in schema, f"{lob}: outline names {key!r}, schema does not"


def test_the_fingerprint_distinguishes_lines_of_business():
    """A fingerprint blind to `lob` reports two different prompts as the same
    one, which is exactly the drift it exists to catch."""
    from common.prompts import prompt_fingerprint

    prints = {
        lob: prompt_fingerprint("policy", "ocr_plus_image", None, lob)
        for lob in (None, "homeowners", "personal_auto", "ocean_marine")
    }
    assert len(set(prints.values())) == len(prints), f"fingerprints collided: {prints}"


def test_a_windowed_row_carries_its_real_page_numbers():
    """build_messages has accepted page_numbers/total_pages all along and
    build_training_row never passed them, so every windowed row would have
    claimed to be pages 1..n of an n-page document — while the template tells the
    model that skipped numbers mean it is seeing selected pages."""
    from inference_core.input_builder import build_training_row

    row = build_training_row(
        "policy", "policy_0001", ["p9.png", "p10.png"], ["nine", "ten"],
        "ocr_plus_image", "{}", page_numbers=[9, 10], total_pages=20,
    )
    rendered = json.dumps(row["messages"])
    assert "page 9 of 20" in rendered and "page 10 of 20" in rendered
    assert "page 1 of 2" not in rendered


def test_the_corpus_drift_check_reads_the_key_the_manifest_writes():
    """It read `schema_version`; the manifest writes `schema_versions`, a dict
    keyed by selector. The lookup returned None on every real manifest, the guard
    fell through, and the check reported agreement including when it disagreed."""
    from inference_core.input_builder import InputBuilderError, build_messages

    built = build_messages("policy", ["p1.png"], ["text"], "ocr_plus_image", lob="homeowners")
    assert built.schema_version == "1.0.0+common.1.1.0"

    built.assert_matches_corpus({"schema_versions": {"policy:homeowners": "1.0.0+common.1.1.0"}})
    with pytest.raises(InputBuilderError, match="drift"):
        built.assert_matches_corpus({"schema_versions": {"policy:homeowners": "0.0.1"}})
    # A pin for a different line says nothing about this row.
    built.assert_matches_corpus({"schema_versions": {"policy:personal_auto": "0.0.1"}})


def test_a_list_valued_lob_does_not_break_the_schema_cache():
    """line_of_business is a list, and the loaders are cached — caching on the
    arguments would raise 'unhashable type: list' on every multi-LOB call."""
    from common.schemas import load_schema, resolved_schema, validator_for

    assert load_schema("policy", None, ["workers_comp", "property"])["type"] == "object"
    assert resolved_schema("policy", None, ["workers_comp"])["properties"]
    # Two argument sets that resolve to one key share one parsed copy — the
    # point of caching on the key string rather than on the signature.
    assert validator_for("policy", None, ["a_line_with_no_file"]) is validator_for("policy")
    assert validator_for("policy", None, ["workers_comp"]) is validator_for("policy", None, "wc")
