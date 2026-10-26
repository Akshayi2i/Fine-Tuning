"""Prompt rendering — the single renderer used by every stage.

Corpus build (IMPL-05), evaluation (IMPL-08), serving (IMPL-11) and testing
(IMPL-12) all render prompts through :func:`render_system_prompt`. There is
deliberately no second path.

Why that matters: the system prompt is the conditioning input on every training
row *and* every production request. If the two ever diverge — by a character —
the fine-tuned model degrades at serving time, and **nothing in the training
metrics shows it**, because training never sees the serving prompt (arch §7).
``test_prompt_parity`` asserts the equality this module exists to provide.

The rendered prompt carries each field's ``description``: a semantic gloss with
exclusions, which is how the model maps an unseen surface label onto a canonical
key (arch §0c). It carries **no alias list** — see the template header for why.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from common.constants import MODALITY_MODES, PROMPT_MODE_ALIASES
from common.normalize import OUTPUT_DATE_LABEL
from common.schemas import (
    is_canonical,
    is_common_model,
    required_fields,
    resolved_schema,
    schema_version,
)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

#: Bumped whenever the template's rendered output changes in any way.
#: Recorded in the corpus manifest; a change forces a corpus rebuild and a new
#: training cycle, exactly like a schema change (arch §7).
PROMPT_TEMPLATE_VERSION = "7.2.0"

_SYSTEM_TEMPLATE = "system_prompt_template.jinja"
_CLASSIFIER_TEMPLATE = "doc_type_classifier_prompt.jinja"

#: The fast format profile's templates (configs/shared/format.yaml): the system
#: message every group of a policy shares, and the group's own block, sent last.
FAST_TEMPLATE_DIR = PROMPT_DIR / "fast"
_FAST_SYSTEM_TEMPLATE = "fast/system.jinja"
_FAST_GROUP_TEMPLATE = "fast/group.jinja"

#: Bumped whenever the fast layout - its templates' output, the field list, or
#: the rule that aligns a document's windows across groups - changes. Part of
#: the fast profile's identity (:func:`format_identity`); the standard profile
#: does not read it.
FAST_LAYOUT_VERSION = "1"

#: How the document type is named to the model in prose.
_DOC_TYPE_LABELS: dict[str, str] = {
    "lossrun": "Loss Run",
    "policy": "Policy",
    "acord": "ACORD",
}


class PromptError(RuntimeError):
    """Raised on an unknown modality mode or a template that fails to render."""


@lru_cache(maxsize=1)
def _env() -> Environment:
    # StrictUndefined: a typo'd variable must explode at render time, not
    # silently produce an empty string in a prompt we then train on.
    return Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=False,
    )


def effective_prompt_mode(modality_mode: str) -> str:
    """Map a modality mode onto the mode the *prompt* uses.

    ``noisy_ocr_image`` renders identically to ``ocr_plus_image``: the noise
    lives in the data, not the instruction (arch §6). Teaching the model that a
    special prompt accompanies bad OCR would defeat the point — at inference you
    do not know the OCR is bad.
    """
    if modality_mode not in MODALITY_MODES:
        raise PromptError(f"unknown modality_mode {modality_mode!r}; expected one of {MODALITY_MODES}")
    return PROMPT_MODE_ALIASES.get(modality_mode, modality_mode)


def doc_type_label(doc_type: str, acord_form: str | None = None) -> str:
    """Human-facing name for the document type, e.g. ``ACORD 25``."""
    label = _DOC_TYPE_LABELS.get(doc_type.lower())
    if label is None:
        raise PromptError(f"no label for doc_type {doc_type!r}")
    if doc_type.lower() == "acord" and acord_form:
        return f"{label} {acord_form}"
    return label


def schema_json_for_prompt(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """The schema as it is injected into the prompt.

    ``$ref``-resolved where the refs point at other files the model was never
    shown, left alone where they point inside the schema itself — the decision
    lives in :func:`common.schemas.resolved_schema`, which owns it for validation
    and rendering alike.

    Serialised with sorted keys and a fixed separator so rendering is
    deterministic; an unstable key order would break prompt parity for no reason.
    """
    schema = resolved_schema(doc_type, acord_form, lob, sections)
    stripped = _strip_authoring_keys(schema)
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _strip_authoring_keys(node: Any) -> Any:
    """Drop keys meant for humans maintaining the schema, not for the model.

    ``$comment`` and ``$schema``/``$id`` are authoring metadata. Leaving them in
    spends tokens on every training row and every request to say nothing the
    model can act on.
    """
    if isinstance(node, list):
        return [_strip_authoring_keys(v) for v in node]
    if isinstance(node, dict):
        return {
            k: _strip_authoring_keys(v)
            for k, v in node.items()
            if k not in ("$comment", "$schema", "$id", "version")
        }
    return node


def _is_array(node: dict[str, Any]) -> bool:
    declared = node.get("type")
    return declared == "array" or (isinstance(declared, list) and "array" in declared)


def output_shape_for_prompt(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """A compact skeleton of the JSON this document type must return.

    **Derived from the schema, never hand-written.** A hand-kept copy of the key
    set in a prompt file drifts the first time a field is added, and the drift is
    silent: the model is shown one shape and validated against another. Rendering
    it from :func:`resolved_schema` means the two cannot disagree.

    Takes ``lob`` for the same reason :func:`schema_json_for_prompt` does, and
    must be called with the same value: one renders the outline and the other the
    detail, into the same prompt. Given different lines they would describe
    different field sets a few lines apart.

    Deliberately shape-only — no descriptions, no types. The schema follows
    immediately after and is the authority on both; this exists so the model sees
    the outline before the detail.
    """
    from common.schemas import is_common_model, resolve_local, row_keys

    schema = resolved_schema(doc_type, acord_form, lob, sections)
    properties: dict[str, Any] = schema.get("properties", {})
    if doc_type.lower() == "policy" and is_common_model(doc_type, acord_form, lob):
        # Every common-model block is a $ref to a definition, so the outline is
        # read through them; read as written, every block would look scalar.
        defs = schema.get("$defs") or {}
        properties = {name: resolve_local(node, defs) for name, node in properties.items()}

        def row(node: dict[str, Any]) -> list[str]:
            return row_keys(node.get("items") or {}, defs)
    else:
        def row(node: dict[str, Any]) -> list[str]:
            return list((node.get("items") or {}).get("properties", {}))

    scalars = [name for name, node in properties.items() if not _is_array(node)]
    lines = [f'  {", ".join(scalars)}'] if scalars else []

    for name, node in properties.items():
        if not _is_array(node):
            continue
        keys = row(node)
        lines.append(f'  {name}: [ {{ {", ".join(keys)} }}, ... ]' if keys else f"  {name}: [ ... ]")
    return "\n".join(lines)


def render_system_prompt(
    doc_type: str,
    modality_mode: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """Render the system prompt. **The** entrypoint — never bypass it.

    Args:
        doc_type: ``acord`` | ``policy`` | ``lossrun``.
        modality_mode: one of :data:`common.constants.MODALITY_MODES`.
        acord_form: required when ``doc_type == "acord"``, selects the schema.
        lob: a policy's line of business. Selects the per-LOB canonical schema
            when exactly one line is named and a schema is registered for it,
            per :func:`common.schemas.schema_key`. A homeowners policy and a
            personal auto policy share a header and little else, so showing the
            model the generic schema for both would train it on a field set that
            matches neither document.

    Returns:
        The system message, identical for the same inputs in every context.
        Under the fast format profile a policy window's system message is the
        instruction every group of the document shares; its group's own block
        is :func:`render_group_block` (:func:`fast_layout`).
    """
    mode = effective_prompt_mode(modality_mode)
    if fast_layout(doc_type, acord_form, lob, sections):
        return _env().get_template(_FAST_SYSTEM_TEMPLATE).render(
            doc_type_label=doc_type_label(doc_type, acord_form),
            modality_mode=mode,
            common_model=is_common_model(doc_type, acord_form, lob),
            date_format=OUTPUT_DATE_LABEL,
        ).strip()
    template = _env().get_template(_SYSTEM_TEMPLATE)
    return template.render(
        # `doc_type` selects the per-type rules block, `acord_form` the form
        # paragraph inside it, and `lob` the canonical schema — the same values
        # that select the schema, so a prompt can never describe one form or
        # line while carrying another's schema.
        doc_type=doc_type.lower(),
        acord_form=acord_form,
        # The client's canonical FieldValue shape (sparse envelopes) or the flat
        # one (every key, null for absence). Decided by the schema registry, so
        # the prompt can never describe one shape while carrying the other's
        # schema.
        canonical=is_canonical(doc_type, acord_form, lob),
        # A SPEC_21 line on the common model: its values, ids and codes, and its
        # coverage table, are described in their own words. The self-contained
        # lines render exactly as before.
        common_model=doc_type.lower() == "policy" and is_common_model(doc_type, acord_form, lob),
        required_keys=_prose_list(required_fields(doc_type, acord_form, lob, sections)),
        read_elsewhere=_read_elsewhere(doc_type, acord_form, lob, sections),
        output_shape=output_shape_for_prompt(doc_type, acord_form, lob, sections),
        doc_type_label=doc_type_label(doc_type, acord_form),
        modality_mode=mode,
        # Rendered from the constant the post-process formats with, not written
        # into the template. A literal here is a second source of truth for the
        # output date format, and the failure is silent in both directions: the
        # model is asked for one format and its answer rewritten into another,
        # which reads as the model getting dates wrong.
        date_format=OUTPUT_DATE_LABEL,
        schema_json=schema_json_for_prompt(doc_type, acord_form, lob, sections),
    ).strip()


# --------------------------------------------------------------------------
# The fast format profile (configs/shared/format.yaml)
# --------------------------------------------------------------------------


def fast_layout(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> bool:
    """Whether a request is written in the fast layout: the fast profile is on
    and it is a canonical policy window (a section group's slice). Every other
    request - ACORD, Loss Run, a whole-schema call - is written as in standard.

    The fast layout and the compact answer go together: a request written in
    it is answered compact (``common.canonical.to_compact_answer``)."""
    from common.config import fast_profile

    return (bool(sections) and doc_type.lower() == "policy" and fast_profile()
            and is_canonical(doc_type, acord_form, lob))


def render_group_block(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str | None:
    """A fast-layout window's own instruction, sent after its pages: the group's
    top-level keys and its fields, compactly (:func:`compact_fields_for_prompt`).
    None for a request in the standard layout."""
    if not fast_layout(doc_type, acord_form, lob, sections):
        return None
    from common.schema_sections import _rule

    listing = compact_fields_for_prompt(doc_type, acord_form, lob, sections)
    top = list(resolved_schema(doc_type, acord_form, lob, sections).get("properties") or {})
    return _env().get_template(_FAST_GROUP_TEMPLATE).render(
        title=str(_rule(str(sections)).get("title") or sections),
        top_keys=_prose_list(top),
        required_keys=_prose_list(required_fields(doc_type, acord_form, lob, sections)),
        read_elsewhere=_read_elsewhere(doc_type, acord_form, lob, sections),
        fields=listing["fields"],
        kinds=listing["kinds"],
        codes=listing["codes"],
    ).strip()


#: Between a fast-layout window's last page and its group block. The trainer
#: and the chat template both join a turn's text parts with nothing between.
GROUP_BLOCK_SEPARATOR = "\n\n"


def prompt_text(
    doc_type: str,
    modality_mode: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """Every instruction one request carries: its system message, and in the
    fast layout its group block. What a window is sized against; in the
    standard layout exactly :func:`render_system_prompt`."""
    system = render_system_prompt(doc_type, modality_mode, acord_form, lob, sections)
    block = render_group_block(doc_type, acord_form, lob, sections)
    return system if block is None else system + GROUP_BLOCK_SEPARATOR + block


#: A value kind by its model-view definition; any other is named after its
#: definition (``DateValue`` -> ``date``).
_VALUE_KIND_NAMES = {"FieldValue": "text", "BoolValue": "yesno", "PostalCodeValue": "zip"}

#: A meaning longer than this is cut to its first sentence.
SHORT_MEANING_CHARS = 240


def compact_fields_for_prompt(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> dict[str, str]:
    """A schema slice as the fast layout lists it: ``{"fields", "kinds", "codes"}``.

    ``fields``: one line per field, ``key: kind - meaning``, under the object
    that holds it; each object definition is listed once, however many fields
    hold one (an address under the carrier, the producer and the insured).
    ``kinds``: each value kind used - what ``v`` holds, and its meaning.
    ``codes``: each code list used, with each code's meaning.

    Derived from :func:`common.schemas.resolved_schema`, the model view the
    decoder is held to, so the list and the grammar cannot disagree.
    """
    return _CompactListing(resolved_schema(doc_type, acord_form, lob, sections)).render()


def _short(text: Any) -> str:
    """A meaning as the fast field list shows it: whole up to
    :data:`SHORT_MEANING_CHARS`, else its first sentence."""
    import re

    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    if len(text) <= SHORT_MEANING_CHARS:
        return text
    for match in re.finditer(r"\.\s+(?=[A-Z`'\"(])", text):
        head = text[:match.start()]
        if not re.search(r"(?:\be\.g|\bi\.e|\betc|\bvs|\bNo)$", head):
            return head + "."
    return text[:SHORT_MEANING_CHARS - 3].rsplit(" ", 1)[0] + "..."


class _CompactListing:
    """One schema slice walked into the fast layout's field list."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.defs: dict[str, Any] = schema.get("$defs") or {}
        self.kinds: dict[str, str] = {}
        self.codes: dict[str, str] = {}
        self.types: dict[str, dict[str, Any]] = {}
        self.queue: list[str] = []

    def deref(self, node: Any) -> tuple[Any, str | None]:
        name = None
        while isinstance(node, dict) and isinstance(node.get("$ref"), str):
            ref = node["$ref"]
            if not ref.startswith("#/$defs/"):
                return {}, name
            name = ref.rsplit("/", 1)[-1]
            node = self.defs.get(name)
        return node, name

    @staticmethod
    def _is_value(node: Any) -> bool:
        props = node.get("properties") if isinstance(node, dict) else None
        return isinstance(props, dict) and {"raw", "parsed", "page_ref"} <= set(props)

    @staticmethod
    def _types(node: dict[str, Any]) -> list[str]:
        declared = node.get("type")
        types = [declared] if isinstance(declared, str) else list(declared or ())
        return [t for t in types if t != "null"]

    def _register(self, name: str, node: dict[str, Any]) -> str:
        if name not in self.types:
            self.types[name] = node
            self.queue.append(name)
        return name

    def kind(self, node: Any) -> str:
        """The kind a field's line names, registering what it refers to."""
        node, name = self.deref(node)
        if not isinstance(node, dict):
            return "any"
        if self._is_value(node):
            return self.value_kind(node, name)
        branches = node.get("anyOf") or node.get("oneOf")
        if branches and "properties" not in node and "items" not in node:
            resolved = [self.deref(branch)[0] for branch in branches]
            if all(isinstance(b, dict) and "const" in b for b in resolved):
                label = name or "Code"
                self.codes.setdefault(label, f"{label}: " + "; ".join(
                    f"{b['const']} = {_short(b['description'])}" if b.get("description") else str(b["const"])
                    for b in resolved))
                return label
            if name and all(isinstance(b, dict) and "properties" in b for b in resolved):
                return self._register(name, self._union(resolved, node))
            return " or ".join(self.kind(branch) for branch in branches)
        if "enum" in node:
            values = [str(v) for v in node["enum"] if v is not None]
            if name and len(values) > 3:
                self.codes.setdefault(name, f"{name}: " + " | ".join(values))
                return name
            return "one of " + "|".join(values)
        if "const" in node:
            return json.dumps(node["const"])
        types = self._types(node)
        if "array" in types or "items" in node:
            return f"[{self.kind(node.get('items') or {})}]"
        if "object" in types or "properties" in node:
            return self._register(name, node) if name else "object"
        if types == ["string"] and node.get("pattern"):
            return "id"
        return "/".join(types) or "any"

    def value_kind(self, node: dict[str, Any], name: str | None) -> str:
        label = _VALUE_KIND_NAMES.get(name or "") or (
            name[:-len("Value")].lower() if name and name.endswith("Value") else (name or "value").lower())
        if label not in self.kinds:
            parsed = node["properties"]["parsed"]
            parsed = self.deref(parsed)[0] if isinstance(parsed, dict) else {}
            branches = [self.deref(b)[0] for b in (parsed.get("anyOf") or parsed.get("oneOf") or [parsed])
                        ] if isinstance(parsed, dict) else []
            if isinstance(parsed, dict) and "enum" in parsed:
                holds = "one of " + "|".join(str(v) for v in parsed["enum"] if v is not None)
            else:
                types = [t for b in branches if isinstance(b, dict) for t in self._types(b)]
                holds = "/".join(dict.fromkeys(types)) or "any"
            meaning = _short(node.get("description"))
            self.kinds[label] = f"{label}: {holds}" + (f" - {meaning}" if meaning else "")
        return label

    @staticmethod
    def _union(variants: list[dict[str, Any]], node: dict[str, Any]) -> dict[str, Any]:
        """Variant objects as one: every key any variant declares, in order,
        each with the first meaning given for it."""
        props: dict[str, Any] = {}
        for variant in variants:
            for key, sub in (variant.get("properties") or {}).items():
                if key not in props or (not props[key].get("description") and sub.get("description")):
                    props[key] = sub
        return {"type": "object", "properties": props, "description": node.get("description")}

    def lines(self, properties: dict[str, Any], indent: str) -> list[str]:
        out: list[str] = []
        for key, sub in properties.items():
            node, name = self.deref(sub)
            kind = self.kind(sub)
            own = sub.get("description") if isinstance(sub, dict) else None
            if not own and isinstance(node, dict) and kind not in self.types and not self._is_value(node):
                own = node.get("description")
            meaning = _short(own)
            out.append(f"{indent}{key}: {kind}" + (f" - {meaning}" if meaning else ""))
            if kind in ("object", "[object]"):
                inner = node if kind == "object" else self.deref(node.get("items") or {})[0]
                out += self.lines((inner or {}).get("properties") or {}, indent + "  ")
        return out

    def render(self) -> dict[str, str]:
        out: list[str] = []
        for key, sub in (self.schema.get("properties") or {}).items():
            node, name = self.deref(sub)
            if isinstance(node, dict) and "properties" in node and not self._is_value(node):
                # A top-level object is listed under its own key.
                if name:
                    self.types.setdefault(name, node)
                meaning = _short(node.get("description") or (sub.get("description") if isinstance(sub, dict) else None))
                out.append(key + (f" - {meaning}" if meaning else ""))
                out += self.lines(node.get("properties") or {}, "  ")
            else:
                out += self.lines({key: sub}, "")
        while self.queue:
            name = self.queue.pop(0)
            node = self.types[name]
            meaning = _short(node.get("description"))
            out.append(name + (f" - {meaning}" if meaning else ""))
            out += self.lines(node.get("properties") or {}, "  ")
        return {"fields": "\n".join(out), "kinds": "\n".join(self.kinds.values()),
                "codes": "\n".join(self.codes.values())}


def format_identity() -> dict[str, Any] | None:
    """What makes the active format profile a different input and answer from
    the standard one, or None under the standard profile: the profile, the fast
    layout's version, the answer format and the pixel budget. Folded into
    :func:`prompt_hash` and the window plan id
    (``policy_windows.window_plan_id``), so an adapter trained in one profile is
    refused in the other, and a corpus built in one neither trains nor
    packages in the other."""
    from common.config import fast_profile, format_profile, pixel_budget

    if not fast_profile():
        return None
    from common.canonical import COMPACT_ANSWER_FORMAT

    return {
        "profile": format_profile(),
        "layout": FAST_LAYOUT_VERSION,
        "answer_format": COMPACT_ANSWER_FORMAT,
        "pixel_budget": list(pixel_budget()),
    }


def _read_elsewhere(
    doc_type: str, acord_form: str | None, lob: str | list[str] | None, sections: str | None,
) -> str:
    """The top-level keys other windows read, for a common-model window whose
    slice holds the additional fields - else "".

    The window that decides what has no field is shown only its own slice: a
    policy number printed on its pages has no field THERE, and by the rule for
    values with no field it went under the additional fields - in every window
    of every page that prints one, against labels that hold it in its field.
    Derived from the schema, like the outline, so it names no key the line lacks.
    """
    if sections is None or doc_type.lower() != "policy" or not is_common_model(doc_type, acord_form, lob):
        return ""
    here = resolved_schema(doc_type, acord_form, lob, sections).get("properties") or {}
    if "additional_fields" not in here:
        return ""
    whole = resolved_schema(doc_type, acord_form, lob).get("properties") or {}
    return _prose_list([name for name in whole if name not in here])


def _prose_list(names: list[str]) -> str:
    """``carrier``, ``named_insured`` and ``policy`` — for a sentence, not a list."""
    quoted = [f"`{n}`" for n in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def lob_meanings() -> dict[str, str]:
    """Each registered line of business and what it is (configs/lob_meanings.yaml),
    in the file's order. Role-based: no line is described by a printed label."""
    from common.config import CONFIG_DIR, load_yaml
    from common.lob import known_lines

    meanings = load_yaml(CONFIG_DIR / "lob_meanings.yaml").get("lines") or {}
    registered = known_lines()
    return {str(line): str(text) for line, text in meanings.items() if line in registered}


def render_classifier_prompt(*, lines: Iterable[str] | None = None, line_only: bool = False) -> str:
    """Render the zero-shot document-type classifier prompt (arch §4a, §4b).

    A policy's line of business is asked for too, from ``lines`` (every
    registered line by default), each with its meaning: a policy that reaches L3
    with no line from L1/L2 still needs one to choose its schema.
    ``line_only`` asks for the line alone, among ``lines``.
    """
    meanings = lob_meanings()
    chosen = list(meanings) if lines is None else [line for line in lines if line in meanings]
    return _env().get_template(_CLASSIFIER_TEMPLATE).render(
        lines=[(line, meanings[line]) for line in chosen], line_only=line_only,
    ).strip()


def prompt_fingerprint(
    doc_type: str,
    modality_mode: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """A short hash of the rendered prompt.

    Cheap way for a caller to assert the prompt it is about to send matches the
    one the corpus was built with, without diffing the whole string. Takes the
    same selectors as the render it hashes — a fingerprint blind to ``lob`` would
    report two different prompts as the same one, which is precisely the drift it
    exists to catch.
    """
    import hashlib

    # The whole instruction: in the fast layout the group block too.
    rendered = prompt_text(doc_type, modality_mode, acord_form, lob, sections)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def prompt_versions(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> dict[str, str]:
    """Versions to record in the corpus manifest (arch §7).

    A change to either forces a corpus rebuild and a new training cycle.
    """
    return {
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "schema_version": schema_version(doc_type, acord_form, lob),
    }


def prompt_input_files() -> list[Path]:
    """Every file that shapes what the model is shown, in a fixed order.

    The templates (the per-doc-type ones in ``prompts/doc_types/`` included),
    the schemas the prompt embeds — ours and the client's canonical files — and
    the section map that slices a policy into windows. The release hash used to
    cover the top-level templates alone, so an edit to ``policy.jinja``, a
    canonical schema or a section group changed every served prompt and left
    the hash — the one thing meant to notice — exactly as it was.

    The line overlays compose the common model, and their coverage-code files
    give each code the meaning the prompt shows, so both are prompt input too.
    So is the classifier's: each line's meaning (``lob_meanings.yaml``) is
    written into its prompt.

    The fast profile's templates (``prompts/fast/``) and its switch
    (``configs/shared/format.yaml``) are prompt input only under that profile:
    they shape no standard prompt, so the standard profile's files - and its
    hash - are exactly what they were before the profile existed.
    """
    from common.config import CONFIG_DIR, SHARED_FORMAT_CONFIG, fast_profile
    from common.model_view import MODEL_VIEW_CONFIG
    from common.schemas import CANONICAL_DIR, COMMON_MODEL, SCHEMA_DIR

    fast = fast_profile()
    templates = [path for path in sorted(PROMPT_DIR.rglob("*.jinja"))
                 if fast or FAST_TEMPLATE_DIR not in path.parents]
    return (
        templates
        + sorted(SCHEMA_DIR.glob("*.json"))
        + sorted(CANONICAL_DIR.glob("*.json"))
        + [COMMON_MODEL]
        + sorted(CANONICAL_DIR.glob("*.coverage_codes.yaml"))
        + [MODEL_VIEW_CONFIG]
        + [CONFIG_DIR / "schema_sections.yaml"]
        + [CONFIG_DIR / "lob_meanings.yaml"]
        + ([SHARED_FORMAT_CONFIG] if fast else [])
    )


def prompt_hash() -> str:
    """SHA-256 over :func:`prompt_input_files`, each keyed by its repo path,
    and - under a format profile other than standard - its identity
    (:func:`format_identity`)."""
    import hashlib

    root = PROMPT_DIR.parent
    digest = hashlib.sha256()
    for path in prompt_input_files():
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    identity = format_identity()
    if identity is not None:
        digest.update(b"format_profile\0")
        digest.update(json.dumps(identity, sort_keys=True).encode())
    return digest.hexdigest()
