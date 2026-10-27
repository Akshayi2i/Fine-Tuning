"""Schema registry — load, resolve, and validate against the canonical schemas.

The schemas in ``schemas/`` are the Fideon SPEC_00 canonical models serialised to
JSON Schema (master §1.1). They are the training target *and* the inference
output contract, so this module is the single place they get loaded.

Two things here are load-bearing rather than incidental:

``$ref`` resolution
    ``common_fields.json`` and ``lob.enum.json`` hold definitions shared across
    document types. Defining a field per-schema instead would let two
    definitions of the same canonical field drift apart, and no test would catch
    it because each schema still validates on its own (IMPL-01).

Field descriptions
    Every field carries a ``description`` — a semantic gloss with exclusions.
    These are **prompt text**: they are injected into the system prompt at
    corpus-build and inference time, so they are part of what the model trains
    on (arch §0c). :func:`assert_all_fields_described` enforces their presence.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from common.constants import ACORD_FORMS, ACTIVE_DOC_TYPES

log = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

#: The client's canonical models. These are **theirs**: read-only here, never
#: written, never edited. Everything this repo adds to them — field descriptions,
#: section maps — lives in a separate file.
CANONICAL_ROOT = Path(__file__).resolve().parent.parent / "configs" / "canonical schema"

#: One file per line of business.
CANONICAL_DIR = CANONICAL_ROOT / "LOB Schema"

#: The common model the SPEC_21 line overlays compose (SPEC_21 §4.2): one
#: structure for every line, which an overlay ``$ref``s block by block. The
#: overlays name it ``../common/common_model.json``, the main repository's
#: layout; here it sits in ``common schema``. It is therefore found by FILE NAME
#: (:func:`_bundle`), so the client's files are never edited to match this repo.
COMMON_MODEL = CANONICAL_ROOT / "common schema" / "common_model.json"

#: Carried by an overlay and by the common model; equal values mean the overlay
#: was written against the common model that is loaded.
COMMON_MODEL_VERSION_KEY = "fideon:common_model_version"

#: Written onto a bundle: the overlay's version and the common model's together,
#: so a change to either moves the version a corpus manifest records.
BUNDLE_VERSION_KEY = "fideon:bundle_version"

#: The canonical file a policy uses when no single line selects one: the line is
#: unknown, the policy covers several lines, or the line has no file of its own.
#: The client's own note on it says exactly that ("used when the line is unknown
#: or no line-specific schema exists"), so this is their rule, not ours.
CANONICAL_FALLBACK = "_fallback"

#: Files in ``CANONICAL_DIR`` that are not a line's schema.
#:
#: * ``_common`` is merged into every line file by the client's registry and
#:   never loaded for extraction on its own (Fideon SPEC_00 §5.2).
#: * ``classic_auto`` is personal auto (``common.lob.MERGED_LINES``): one line,
#:   one schema. Registered, it would be a line of its own that no policy can
#:   select, counted wherever every line is listed.
_CANONICAL_NOT_REGISTERED = ("_common", "classic_auto")

#: LOB enum values whose canonical file is named differently. The enum is ours
#: (``schemas/lob.enum.json``); the file names are the client's. Without this a
#: Workers' Comp policy would find no ``workers_comp.json`` and fall back.
LOB_SCHEMA_ALIASES: dict[str, str] = {
    "workers_comp": "wc",
    "general_liability": "gl",
    # The L1 code of commercial general liability, and its SPEC_21 routing key.
    "cgl": "gl",
    "commercial_auto": "auto",
    # One canonical schema with health since 10 October 2026 (CR-HEALTH-01): the
    # owner's lob_schema_map.yaml files New York DBL/PFL policies under health.
    "statutory_disability": "health",
}

#: Shared definition files, loaded into the registry so ``$ref`` can reach them.
_SHARED = ("common_fields.json", "lob.enum.json")

#: doc_type (+ acord_form) -> schema filename, for the schemas in ``schemas/``.
#: Kept as the legacy table: :func:`all_schema_keys` still reads it, and its
#: two-tuple shape cannot express a line of business.
_SCHEMA_FILES: dict[str, str] = {
    "lossrun": "lossrun.schema.json",
    "policy": "policy_doc.schema.json",
    "acord:25": "acord25.schema.json",
    "acord:125": "acord125.schema.json",
    "acord:140": "acord140.schema.json",
}

#: Separates a schema key from the name of the slice of it a window asks for:
#: ``policy:homeowners#arrays``. Safe as a separator because ``doc_type`` is
#: constrained to ``ACTIVE_DOC_TYPES``, an LOB is normalised to the client's file
#: stem, and an ACORD form never reaches that branch — so it cannot occur in a
#: base key by accident.
SLICE_SEPARATOR = "#"

#: The lines whose rendered prompts are pinned as reference snapshots
#: (``testing/prompts``). Every canonical file is registered; this only names the
#: lines whose prompts are kept under review.
SNAPSHOT_LINES: tuple[str, ...] = (
    "homeowners", "personal_auto", "dwelling_fire", "ocean_marine",
    "motorcycle", "recreational_vehicle", "personal_umbrella",
)


class SchemaError(RuntimeError):
    """Raised when a schema is missing, unresolvable, or structurally invalid."""


@dataclass(frozen=True)
class SchemaSource:
    """Where one registry key's schema comes from, and how to read it.

    Two families of schema live behind one interface, and they differ in three
    ways that each cause a silent failure if assumed away:

    ``inline_refs``
        The schemas in ``schemas/`` ``$ref`` *other files* (``common_fields.json``),
        which the model cannot follow, so they are inlined into the prompt. The
        canonical schemas ``$ref`` only their own ``#/$defs/FieldValue`` — once
        per leaf, 532 times in homeowners. Inlining that turns a 41,955-character
        file into 281,355 characters of identical repeated envelope, which is
        ~70k tokens of prompt saying one thing over and over.

    ``version_at``
        ``schemas/`` declares a top-level ``version``. The canonical files carry
        theirs at ``fideon:source.version``. :func:`schema_version` is called for
        every registered schema when a corpus manifest is written, so a wrong
        path here fails every build rather than one call.

    ``strip_prefixes``
        Key prefixes dropped before the schema reaches the model. The canonical
        files carry ``fideon:aliases`` — the printed labels carriers use for each
        field. Those must never reach a prompt (master §1.4): a model handed a
        lookup table never learns the semantics, and the first unseen label
        produces a miss with no signal that anything went wrong.

    ``common_model``
        A SPEC_21 line overlay: a thin file whose blocks ``$ref`` the common
        model in another file. Loaded as a bundle (:func:`_bundle`), so every
        reader still gets one self-contained schema. The self-contained line
        files carry no common-model version and are read as they are.
    """

    path: Path
    inline_refs: bool
    version_at: tuple[str, ...]
    strip_prefixes: tuple[str, ...]
    common_model: bool = False


@lru_cache(maxsize=1)
def _sources() -> dict[str, SchemaSource]:
    """Every registry key mapped to its source. Built lazily.

    Lazily because it reads every schema file, which no import should pay for.

    The ``schemas/`` entries come first, and :func:`schema_selectors` preserves
    that order, so a caller that scans until it finds what it needs scans the
    small generic schemas before the large canonical ones.
    """
    sources: dict[str, SchemaSource] = {
        key: SchemaSource(
            path=SCHEMA_DIR / filename,
            inline_refs=True,
            version_at=("version",),
            strip_prefixes=(),
        )
        for key, filename in _SCHEMA_FILES.items()
    }

    # Refused rather than skipped. This directory was renamed once and the
    # registry skipped every missing file, so each policy quietly rendered the
    # generic schema and nothing reported it.
    fallback = CANONICAL_DIR / f"{CANONICAL_FALLBACK}.json"
    if not fallback.exists():
        raise SchemaError(
            f"the canonical policy schemas are missing: expected {fallback}. Every policy is "
            "extracted into the client's canonical JSON, so without them there is no policy "
            "output contract at all."
        )

    # A policy with no single line of business is extracted into the client's
    # canonical fallback, not the generic flat schema: a policy's output contract
    # is the canonical JSON whatever its line. The key stays `policy`, so corpus
    # manifests and schema pins keep addressing it by the same name.
    sources["policy"] = _canonical_source(fallback)
    for path in sorted(CANONICAL_DIR.glob("*.json")):
        if path.stem in (CANONICAL_FALLBACK, *_CANONICAL_NOT_REGISTERED):
            continue
        sources[f"policy:{path.stem}"] = _canonical_source(path)
    return sources


def _canonical_source(path: Path) -> SchemaSource:
    common_model = COMMON_MODEL_VERSION_KEY in _load_json(path)
    return SchemaSource(
        path=path,
        inline_refs=False,
        version_at=(BUNDLE_VERSION_KEY,) if common_model else ("fideon:source", "version"),
        strip_prefixes=("fideon:",),
        common_model=common_model,
    )


def is_canonical(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> bool:
    """Whether this selection's output is the client's canonical ``FieldValue`` JSON.

    True for every policy. The two output shapes differ in how absence is written
    (omitted, not ``null``) and in what a leaf is (an envelope, not a value), so
    the prompt, the training target and the serving post-process all branch on
    this one answer rather than each deciding for itself.
    """
    return not _sources()[base_key(schema_key(doc_type, acord_form, lob))].inline_refs


def is_common_model(
    doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None
) -> bool:
    """Whether this selection's schema is a SPEC_21 overlay on the common model.

    The one switch every schema-2 difference keys off — the model view, the
    prompt text, the section map, the target builder and the merge — so a line
    is read one way everywhere or not at all.
    """
    return _sources()[base_key(schema_key(doc_type, acord_form, lob))].common_model


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SchemaError(f"schema file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _common_model() -> dict[str, Any]:
    return _load_json(COMMON_MODEL)


def composes_common(wanted: Any, loaded: Any) -> bool:
    """Whether an overlay pinned at common model ``wanted`` composes ``loaded``.

    The same version, or an earlier minor of the same major: the common model's
    change control (``CONTRIBUTING_common.md``, ``tools/check_common_compat.py``)
    makes a minor release additions only - nothing renamed, removed, retyped or
    made required - so a line pinned at 1.1.0 composes 1.4.0 unchanged, as the
    schema owner's own registry composes it. A later version than the one
    loaded, another major, or an unreadable version is refused. The bundle
    records the version it was composed with (``BUNDLE_VERSION_KEY``).
    """
    def parts(version: Any) -> tuple[int, ...] | None:
        try:
            return tuple(int(part) for part in str(version).split("."))
        except ValueError:
            return None

    if wanted == loaded:
        return True
    want, have = parts(wanted), parts(loaded)
    return want is not None and have is not None and want[0] == have[0] and want <= have


def _bundle(overlay: dict[str, Any], path: Path) -> dict[str, Any]:
    """A SPEC_21 overlay as one self-contained schema.

    Each ``$ref`` into the common model is rewritten to ``#/$defs/<name>`` and the
    common definitions it reaches are copied in, so the validator, the section
    slicer, ``with_all_keys`` and the rest read a bundle exactly as they read a
    self-contained line file. Done once, here, rather than taught to each reader:
    a reader left out would see a block as a bare pointer and silently skip it.

    Refused, never guessed: a reference into any other file, a pointer outside
    ``$defs``, an overlay written against a common-model version this one does
    not compose (:func:`composes_common`), or a coverage-code list the line's code
    file does not name. Each would ship a schema that differs from the one the
    overlay's author reviewed.

    Also attached, for the model view to read before ``fideon:`` keys are
    stripped: the version of the pair, and each coverage code's meaning.
    """
    import copy

    common = _common_model()
    wanted, loaded = overlay.get(COMMON_MODEL_VERSION_KEY), common.get(COMMON_MODEL_VERSION_KEY)
    if not composes_common(wanted, loaded):
        raise SchemaError(
            f"{path.name} composes common model {wanted}, but {COMMON_MODEL} is {loaded}"
        )
    definitions = common.get("$defs") or {}
    needed: set[str] = set()

    def localise(node: Any) -> Any:
        if isinstance(node, list):
            return [localise(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {k: localise(v) for k, v in node.items() if k != "$ref"}
        if "$ref" in node:
            target, _, pointer = str(node["$ref"]).partition("#")
            if target:
                if Path(target).name != COMMON_MODEL.name or not pointer.startswith("/$defs/"):
                    raise SchemaError(
                        f"{path.name}: $ref {node['$ref']!r} points outside the common "
                        "model's definitions"
                    )
                needed.add(pointer.split("/")[2])
            out["$ref"] = f"#{pointer}"
        return out

    bundled = localise(overlay)
    pending = sorted(needed)
    while pending:
        name = pending.pop()
        if name not in definitions:
            raise SchemaError(f"{path.name}: the common model defines no {name!r}")
        for reached in _local_ref_names(definitions[name]):
            if reached not in needed:
                needed.add(reached)
                pending.append(reached)

    own = bundled.get("$defs") or {}
    if clash := needed & set(own):
        raise SchemaError(f"{path.name} redefines common-model definitions {sorted(clash)}")
    bundled["$defs"] = {**{n: copy.deepcopy(definitions[n]) for n in sorted(needed)}, **own}
    bundled[BUNDLE_VERSION_KEY] = f"{overlay.get('fideon:schema_version')}+common.{loaded}"
    bundled["fideon:coverage_code_names"] = _coverage_code_names(overlay, path, common)
    return bundled


def _local_ref_names(node: Any) -> set[str]:
    """The ``$defs`` names a node's own ``#/$defs/...`` references reach directly."""
    names: set[str] = set()
    if isinstance(node, list):
        for item in node:
            names |= _local_ref_names(item)
    elif isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            names.add(ref.split("/")[2])
        for value in node.values():
            names |= _local_ref_names(value)
    return names


def _coverage_code_names(
    overlay: dict[str, Any], path: Path, common: dict[str, Any]
) -> dict[str, str]:
    """Each of the line's coverage codes with its meaning, in the overlay's order.

    The meaning is what the model chooses a code by, so a code without one, or a
    code file that lists a different set than the overlay, is refused.
    """
    from common.config import load_yaml

    codes = list(overlay.get("fideon:coverage_codes") or [])
    code_file = overlay.get("fideon:coverage_codes_file")
    if not codes and not code_file:
        return {}
    if not code_file:
        raise SchemaError(f"{path.name} lists coverage codes but names no coverage-code file")
    listed = {
        entry["code"]: entry.get("name")
        for entry in (load_yaml(path.parent / code_file).get("codes") or [])
    }
    if set(listed) != set(codes):
        raise SchemaError(
            f"{code_file} and {path.name} list different coverage codes: "
            f"only in the file {sorted(set(listed) - set(codes))}, "
            f"only in the schema {sorted(set(codes) - set(listed))}"
        )
    shared = common.get("fideon:shared_coverage_codes") or {}
    names = {code: listed[code] or (shared.get(code) or {}).get("name") for code in codes}
    if unnamed := sorted(code for code, name in names.items() if not name):
        raise SchemaError(f"{code_file}: coverage code(s) with no name: {unnamed}")
    return names


@lru_cache(maxsize=1)
def _registry() -> Registry:
    """A ``referencing`` registry holding every schema, keyed by filename.

    Refs are written relative (``common_fields.json#/$defs/insured_name``), so
    registering under the bare filename is what makes them resolve.
    """
    registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        contents = _load_json(path)
        resource = Resource.from_contents(contents, default_specification=DRAFT202012)
        registry = resource @ registry  # register under its own $id
        registry = registry.with_resource(path.name, resource)  # and under filename
    return registry


def schema_key(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """Build the registry key for a document type, form and line of business.

    ACORD is two-level (arch §4b): one shared adapter, but a distinct schema per
    form, because field sets genuinely differ between 25, 125 and 140.

    **Policies are two-level too, by line of business.** A Workers' Comp policy
    and a Commercial Auto policy share a header and almost nothing else, so each
    LOB may register its own canonical schema (``policy:workers_comp``).

    Two rules, both here rather than at the call sites that would otherwise each
    decide for themselves:

    * **Fall back when no per-LOB schema is registered.** For a policy the
      fallback is the client's canonical ``_fallback.json``, so the output is
      canonical JSON either way. Adding a line is dropping its file into
      ``CANONICAL_DIR``, with no call-site change.
    * **An LOB selects a schema only when exactly one is named.** A package
      policy covering GL, Property and Auto is one document with a section per
      line, so it uses the fallback — picking one of its lines would validate the
      whole document against a third of itself.
    """
    doc_type = doc_type.lower()
    if doc_type == "acord":
        if acord_form is None:
            raise SchemaError(
                "acord requires an acord_form (25 | 125 | 140) to select a schema — "
                "the classifier is two-level for exactly this reason (arch §4b)"
            )
        form = str(acord_form).strip()
        if form not in ACORD_FORMS:
            raise SchemaError(f"unknown ACORD form {form!r}; known forms: {sorted(ACORD_FORMS)}")
        return f"acord:{form}"
    if doc_type not in ACTIVE_DOC_TYPES:
        raise SchemaError(f"unknown doc_type {doc_type!r}; active types: {ACTIVE_DOC_TYPES}")

    lines = [lob] if isinstance(lob, str) else list(lob or [])
    base = doc_type
    if len(lines) == 1:
        from common.lob import merge_line

        line = str(merge_line(str(lines[0]).strip().lower()))
        scoped = f"{doc_type}:{LOB_SCHEMA_ALIASES.get(line, line)}"
        if scoped in _sources():
            base = scoped
    return f"{base}{SLICE_SEPARATOR}{sections}" if sections else base


def base_key(key: str) -> str:
    """``policy:homeowners#arrays`` -> ``policy:homeowners``.

    A slice is a view of a schema, not a schema. Everything that addresses the
    FILE — where it lives, which version it declares, whether it is canonical —
    resolves on the base, because two slices of homeowners are two views of one
    file at one version.
    """
    return key.split(SLICE_SEPARATOR, 1)[0]


def line_of_key(key: str) -> str | None:
    """``policy:homeowners#arrays`` -> ``homeowners``; ``None`` for a key that
    names no line - the fallback (``policy``), which composes the common model
    too since common model 1.1.0."""
    base = base_key(key)
    return base.split(":", 1)[1] if ":" in base else None


def slice_of(key: str) -> str | None:
    """The slice name in a key, or ``None`` for a whole schema."""
    _, separator, name = key.partition(SLICE_SEPARATOR)
    return name if separator else None


def load_schema(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> dict[str, Any]:
    """Return the raw (unresolved) schema for a document type."""
    return _schema_for_key(schema_key(doc_type, acord_form, lob, sections))


def validator_for(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> Draft202012Validator:
    """A ref-resolving validator for this document type."""
    return _validator_for_key(schema_key(doc_type, acord_form, lob, sections))


# Cached on the resolved KEY rather than on the arguments: `lob` arrives as a
# list (line_of_business is a list under §0b) and a list is unhashable, so
# caching the public signature would raise on every multi-LOB call. The key is
# also the right cache granularity — two argument sets that resolve to one schema
# should share one parsed copy.
@cache
def _schema_for_key(key: str) -> dict[str, Any]:
    base = base_key(key)
    try:
        source = _sources()[base]
    except KeyError:
        raise SchemaError(f"no schema registered for key {base!r}") from None
    schema = _load_json(source.path)
    if source.common_model:
        schema = _bundle(schema, source.path)

    name = slice_of(key)
    if not name:
        return schema
    sliced = _slice_sections(schema, name, key)
    if not source.common_model:
        return sliced
    # A bundle carries the definitions of every block; a slice keeps only those
    # its own sections reach, or each window's prompt would describe the whole
    # common model. The self-contained files keep their $defs whole, as before.
    lob = line_of_key(key)
    return _prune_definitions(_without_cross_group_references(sliced, lob, name))


def cross_group_references(lob: str | list[str] | None, group: str) -> frozenset[str]:
    """The reference fields a common-model line's ``group`` leaves out: those
    whose tables are all read by another group.

    A window could only guess at the id of a row it was not asked for: a
    premium item's vehicle in the declarations window, a form's unit in the
    forms window. The field is left out of that window's schema slice, so the
    model is neither shown it nor able to write it, and out of the window's
    training target (``policy_windows.window_target``), where each value left
    out counts as a dangling reference. One set for both: a field the slice
    drops and the target keeps is a target the slice cannot hold.

    Empty for a self-contained line, which has no references.
    """
    from common.schema_sections import references, sections_for

    present = set(sections_for(group, lob))
    return frozenset(
        field for field, tables in references(lob).items() if not present & set(tables)
    )


def _without_cross_group_references(
    sliced: dict[str, Any], lob: str, group: str
) -> dict[str, Any]:
    """A slice without the reference fields whose tables another group reads
    (:func:`cross_group_references`), wherever a definition declares one."""
    import copy

    dropped = cross_group_references(lob, group)
    if not dropped:
        return sliced
    definitions = copy.deepcopy(sliced.get("$defs") or {})
    for definition in definitions.values():
        _drop_properties(definition, set(dropped))
    return {**sliced, "$defs": definitions}


def _drop_properties(node: Any, names: set[str]) -> None:
    """Remove ``names`` from every object schema within ``node``, in place."""
    if isinstance(node, list):
        for item in node:
            _drop_properties(item, names)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        for name in names & set(properties):
            del properties[name]
        if isinstance(node.get("required"), list):
            node["required"] = [r for r in node["required"] if r not in names]
    for key, value in node.items():
        if key != "properties":
            _drop_properties(value, names)
        else:
            for sub in value.values():
                _drop_properties(sub, names)


def _prune_definitions(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with only the ``$defs`` its properties reach, transitively."""
    definitions = schema.get("$defs") or {}
    reached = _local_ref_names(schema.get("properties") or {})
    pending = sorted(reached)
    while pending:
        for name in _local_ref_names(definitions.get(pending.pop(), {})):
            if name not in reached:
                reached.add(name)
                pending.append(name)
    return {**schema, "$defs": {k: v for k, v in definitions.items() if k in reached}}


def _slice_sections(schema: dict[str, Any], group: str, key: str) -> dict[str, Any]:
    """One group's view of a schema: its sections, and nothing else's.

    ``required`` is narrowed to the sections that survive, never left whole. A
    slice that kept the full ``required`` would have structured decoding force
    an ``arrays`` window to emit ``carrier``, ``named_insured`` and ``policy`` as
    ``{}`` — and then two windows would both claim to own ``carrier``, giving the
    merge a conflict that should not exist. On the training side the same list
    reaches :func:`common.canonical.to_model_target`, so the wrong one there
    teaches every schedule target to emit an empty ``carrier``.

    ``$defs`` travels with every slice: each leaf still ``$ref``s the envelope,
    and a slice whose refs do not resolve is not a schema.
    """
    from common.schema_sections import SectionMapError, sections_for

    lob = line_of_key(key)
    try:
        wanted = set(sections_for(group, lob))
    except SectionMapError as exc:
        raise SchemaError(f"cannot slice {key!r}: {exc}") from exc

    properties = {k: v for k, v in (schema.get("properties") or {}).items() if k in wanted}
    sliced = {k: v for k, v in schema.items() if k not in ("properties", "required")}
    sliced["properties"] = properties
    required = [name for name in (schema.get("required") or []) if name in properties]
    if required:
        sliced["required"] = required
    return sliced


@cache
def _validator_for_key(key: str) -> Draft202012Validator:
    """A validator built against the RAW schema, ``fideon:`` keys and all.

    Deliberately not the stripped, prompt-facing form: validation is the
    client's contract and must judge a label against the file they supplied,
    byte for byte. Stripping is a rendering concern, applied one layer up in
    :func:`resolved_schema`.

    ``Draft202012Validator`` is used for every schema regardless of the ``$schema``
    each declares. The canonical files say draft-07, under which a key sitting
    beside a ``$ref`` is *ignored* while :func:`resolved_schema` deliberately
    merges it — so honouring the declared dialect here would make validation and
    prompt rendering disagree about the same file, silently.
    """
    return Draft202012Validator(_schema_for_key(key), registry=_registry())


def validate(
    instance: Any, doc_type: str, acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> None:
    """Validate an extraction or golden label. Raises ``ValidationError``.

    Used by IMPL-04 before a golden label is admitted to the corpus, and by
    IMPL-11 before an inference response is returned — mirroring the Fideon
    Fideon SPEC_07 Stage 3 audit gate (arch §0a).
    """
    validator_for(doc_type, acord_form, lob).validate(instance)


def is_valid(
    instance: Any, doc_type: str, acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> bool:
    """Non-raising form of :func:`validate`."""
    return validator_for(doc_type, acord_form, lob).is_valid(instance)


def iter_validation_errors(
    instance: Any, doc_type: str, acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> Iterator[str]:
    """Human-readable validation errors, best-match first."""
    for err in sorted(validator_for(doc_type, acord_form, lob).iter_errors(instance), key=str):
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        yield f"{where}: {err.message}"


def iter_validation_errors_by_keyword(
    instance: Any, doc_type: str, acord_form: str | None = None,
    lob: str | list[str] | None = None, *, keywords: tuple[str, ...],
) -> Iterator[str]:
    """:func:`iter_validation_errors`, only those a rule in ``keywords`` raised.

    Selected by the rule's keyword (``minProperties``), never by the wording of
    its message, which belongs to the jsonschema release installed. Only the
    errors validation reports at the top: a rule inside an ``anyOf`` or a
    ``oneOf`` branch fails its alternation, not itself.
    """
    wanted = set(keywords)
    for err in sorted(validator_for(doc_type, acord_form, lob).iter_errors(instance), key=str):
        if err.validator in wanted:
            where = "/".join(str(p) for p in err.absolute_path) or "<root>"
            yield f"{where}: {err.message}"


def old_shape_hint(
    label: Any, doc_type: str, acord_form: str | None = None, lob: str | list[str] | None = None,
) -> str | None:
    """Why a label fails a common-model line's schema, when the reason is its shape.

    A gold written for the self-contained schema (a line block, ``terrorism``,
    ``carrier.company_name``) fails a common-model line's schema on dozens of
    paths. The real cause is one sentence; this is it, or ``None``.
    """
    if doc_type.lower() != "policy" or not isinstance(label, dict):
        return None
    if not is_common_model(doc_type, acord_form, lob):
        return None
    declared = set(load_schema(doc_type, acord_form, lob).get("properties") or {})
    foreign = sorted(k for k in label if k not in declared and not str(k).startswith("fideon:"))
    carrier = label.get("carrier")
    if isinstance(carrier, dict) and "company_name" in carrier:
        foreign.append("carrier.company_name")
    if not foreign:
        return None
    return (
        f"this gold is in the self-contained (pre-SPEC_21) shape ({', '.join(foreign[:4])}), but "
        "this line's schema is a SPEC_21 overlay on the common model: it needs a SPEC_21 gold"
    )


def required_fields(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> list[str]:
    """Top-level required field names for this document type.

    For a common-model line, the model view's: the keys the model must always
    write. The client's file also requires lists the pipeline can supply, which
    would otherwise reach every window's target as an empty list.
    """
    if doc_type.lower() == "policy" and is_common_model(doc_type, acord_form, lob):
        return list(resolved_schema(doc_type, acord_form, lob, sections).get("required", []))
    return list(load_schema(doc_type, acord_form, lob, sections).get("required", []))


def schema_version(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """The schema's declared version, recorded in the corpus manifest (arch §7).

    A change here forces a corpus rebuild and a new training cycle.

    Where the version sits differs by source: ``schemas/`` declares a top-level
    ``version``, the canonical files carry theirs under ``fideon:source``. Read
    through :attr:`SchemaSource.version_at` rather than guessing, because this is
    called for every registered schema when a corpus manifest is written — so a
    schema whose version cannot be found fails the whole build, not one lookup.
    """
    key = schema_key(doc_type, acord_form, lob, sections)
    node: Any = _schema_for_key(key)
    for step in _sources()[base_key(key)].version_at:
        node = node.get(step) if isinstance(node, dict) else None
    if not node:
        where = ".".join(_sources()[base_key(key)].version_at)
        raise SchemaError(f"schema {key!r} declares no version at {where!r}")
    return str(node)


#: A window's pages are offered as an ``enum`` up to this many; a longer,
#: scattered list is held to its first and last page instead, which keeps the
#: grammar small.
MAX_PAGE_ENUM = 32


def with_page_bounds(
    schema: dict[str, Any], page_count: int | None, pages: list[int] | None = None,
) -> dict[str, Any]:
    """The decoding schema with every ``page_ref`` held to the pages the request shows.

    ``page_ref`` is a list of integers with no bound, and a model that starts a
    run of consecutive pages - labels carry lists like ``[.., 98, 99, 100]`` -
    can count on past the last page: a smoke-run answer reached ``1528`` and hit
    max_new_tokens, scored as a wrong answer after 8,192 tokens. Bounded to
    ``1..page_count``, with at most ``page_count`` entries, it must close the list.

    ``pages`` - the page numbers the request carries - holds it to those: a
    window that shows pages 5-8 cites only pages 5-8, as every training target
    does (``policy_windows`` narrows ``page_ref`` to the window), and cannot
    write "page 1" for a value it read on page 4. Consecutive pages bound the
    range; scattered ones are listed (up to :data:`MAX_PAGE_ENUM`).

    For the decoding constraint only. The schema rendered into the prompt stays
    the one training showed. A copy; the cached schema is not touched. Unknown
    or non-positive ``page_count`` and no ``pages`` return ``schema`` unchanged.
    """
    import copy

    shown = sorted({int(p) for p in pages or () if int(p) >= 1})
    if not shown and (not page_count or page_count < 1):
        return schema
    if shown:
        consecutive = shown[-1] - shown[0] + 1 == len(shown)
        bound: dict[str, Any] = ({"enum": shown} if not consecutive and len(shown) <= MAX_PAGE_ENUM
                                 else {"minimum": shown[0], "maximum": shown[-1]})
        most = len(shown)
    else:
        bound, most = {"minimum": 1, "maximum": int(page_count)}, int(page_count)
    bounded = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = (node.get("properties") or {}).get("page_ref")
            if isinstance(ref, dict) and ref.get("type") == "array":
                items = ref.get("items") if isinstance(ref.get("items"), dict) else {}
                items = {k: v for k, v in items.items() if k not in ("minimum", "maximum", "enum")}
                ref["items"] = {**items, **bound}
                ref["maxItems"] = most
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(bounded)
    return bounded


def resolved_schema(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> dict[str, Any]:
    """The schema as the model sees it: refs inlined where that helps, never where it hurts.

    For the schemas in ``schemas/`` this inlines every ``$ref``, because they
    point at *other files* (``common_fields.json#/$defs/insured_name``) and the
    model cannot follow a pointer to a file it was never shown.

    For the canonical schemas it does not, and must not. Their only ``$ref`` is
    ``#/$defs/FieldValue``, repeated once per leaf — 532 times in homeowners.
    ``$defs`` travels *with* the schema, so the model can already see what the
    ref means; inlining it would restate the same 480-character envelope 532
    times, turning a 41,955-character schema into 281,355 characters of prompt.

    Either way the result is stripped of keys the model cannot act on, including
    the ``fideon:aliases`` blocks the canonical files carry (master §1.4).
    """
    key = schema_key(doc_type, acord_form, lob, sections)
    source = _sources()[base_key(key)]
    if source.common_model:
        return _model_view_for_key(key)
    if not source.inline_refs:
        return _model_facing_canonical(
            _strip_prefixed(_without_fsm_excluded(_schema_for_key(key)), source.strip_prefixes)
        )

    registry = _registry()
    resolver = registry.resolver()

    def _resolve(node: Any, depth: int = 0) -> Any:
        if depth > 25:
            raise SchemaError("$ref nesting too deep — probable reference cycle")
        if isinstance(node, list):
            return [_resolve(item, depth + 1) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = resolver.lookup(node["$ref"])
            merged = _resolve(target.contents, depth + 1)
            # Keep any sibling keys that sat alongside the $ref.
            extra = {k: _resolve(v, depth + 1) for k, v in node.items() if k != "$ref"}
            return {**merged, **extra} if extra else merged
        return {k: _resolve(v, depth + 1) for k, v in node.items()}

    return _strip_prefixed(_resolve(_schema_for_key(key)), source.strip_prefixes)


@cache
def _model_view_for_key(key: str) -> dict[str, Any]:
    """A common-model line's schema as the model is shown and held to it
    (:mod:`common.model_view`). Cached per key: every window of every request
    asks for it. Shared - callers that change it must copy it first, as
    :func:`with_page_bounds` does."""
    from common.model_view import model_view

    return model_view(_schema_for_key(key))


#: What the model writes for one canonical leaf. The client's ``FieldValue`` also
#: carries ``confidence`` and ``flagged``; those are the PIPELINE's to fill, from
#: the calibrated logprobs and the review thresholds, and never the model's — a
#: model asked for its own confidence produces a number that looks calibrated and
#: is not (arch §5). :mod:`common.canonical` adds them after generation.
MODEL_FIELD_VALUE: dict[str, Any] = {
    "type": "object",
    "required": ["raw", "parsed", "page_ref"],
    "additionalProperties": False,
    "properties": {
        "raw": {"type": ["string", "null"]},
        "parsed": {"type": ["string", "number", "null"]},
        "page_ref": {"type": "array", "items": {"type": "integer"}},
    },
}


def _model_facing_canonical(schema: dict[str, Any]) -> dict[str, Any]:
    """The canonical schema with its ``FieldValue`` narrowed to what the model writes.

    Rendered into the prompt and handed to structured decoding, so the model is
    both told and constrained to write ``raw``/``parsed``/``page_ref`` and nothing
    else. Validation still judges the enveloped result against the client's full
    file (:func:`validator_for`).
    """
    defs = dict(schema.get("$defs") or {})
    if "FieldValue" in defs:
        defs["FieldValue"] = MODEL_FIELD_VALUE
    return {**schema, "$defs": defs}


#: The client's mark on a property the extraction model never fills: today only
#: ``text_sections``, which "is populated by the text extraction path, not the VLM
#: FSM". Shown in a prompt, it asks the model for the document's full wording -
#: a target no training row teaches (policy_windows.EXCLUDED_FROM_TARGETS).
FSM_EXCLUDE_KEY = "fideon:fsm_exclude"


def _without_fsm_excluded(node: Any) -> Any:
    """``node`` without the properties the client marks ``fideon:fsm_exclude``."""
    if isinstance(node, list):
        return [_without_fsm_excluded(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {k: _without_fsm_excluded(v) for k, v in node.items()}
    properties = node.get("properties")
    if isinstance(properties, dict):
        dropped = {
            name for name, sub in properties.items()
            if isinstance(sub, dict) and sub.get(FSM_EXCLUDE_KEY)
        }
        if dropped:
            out["properties"] = {k: v for k, v in out["properties"].items() if k not in dropped}
            if isinstance(node.get("required"), list):
                out["required"] = [r for r in node["required"] if r not in dropped]
    return out


def _strip_prefixed(node: Any, prefixes: tuple[str, ...]) -> Any:
    """Drop every key beginning with one of ``prefixes``, recursively.

    ``fideon:aliases`` is the reason this exists. The canonical schemas record,
    per field, the printed labels carriers actually use — 110 such blocks in
    homeowners alone. That is corpus-analysis knowledge and it is barred from the
    prompt: a model given the lookup table never has to learn the semantics, and
    the first label not in the table produces a miss with nothing to signal it
    (master §1.4).

    The registry's own guard is an import check on ``common.aliases``, which this
    would walk straight past — the aliases arrive inside the client's schema
    file, not through that module. Hence a value-level strip as well.
    """
    if not prefixes:
        return node
    if isinstance(node, list):
        return [_strip_prefixed(item, prefixes) for item in node]
    if isinstance(node, dict):
        return {
            k: _strip_prefixed(v, prefixes)
            for k, v in node.items()
            if not any(k.startswith(p) for p in prefixes)
        }
    return node


def resolve_local(node: Any, defs: dict[str, Any]) -> Any:
    """``node`` with its ``#/$defs/...`` reference followed, keeping the
    description written beside the reference (it describes this use)."""
    seen: set[str] = set()
    out = node
    while isinstance(out, dict) and isinstance(out.get("$ref"), str) and out["$ref"].startswith("#/$defs/"):
        name = out["$ref"].split("/")[2]
        if name in seen or name not in defs:
            break
        seen.add(name)
        here = {k: v for k, v in out.items() if k != "$ref"}
        out = {**defs[name], **here}
    return out


def row_keys(node: Any, defs: dict[str, Any]) -> list[str]:
    """The fields an object (or each of its ``anyOf`` variants) can hold, in order."""
    node = resolve_local(node, defs)
    if not isinstance(node, dict):
        return []
    keys = list(node.get("properties") or {})
    for variant in node.get("anyOf") or []:
        keys += [k for k in row_keys(variant, defs) if k not in keys]
    return keys


def _is_value_node(node: Any) -> bool:
    return isinstance(node, dict) and {"raw", "parsed", "page_ref"} <= set(node.get("properties") or {})


def iter_described_fields(
    schema: dict[str, Any], prefix: str = "", defs: dict[str, Any] | None = None
) -> Iterator[tuple[str, str | None]]:
    """Walk a resolved schema yielding ``(field_path, description)`` for each field.

    Recurses into object properties and array items so nested fields —
    ``claims[].total_incurred``, ``coverage_schedule[].limit`` — are covered too.

    With ``defs`` (a common-model schema, whose blocks are all references), the
    walk follows ``#/$defs/...`` references, reads a value type as one field
    rather than its ``raw``/``parsed``/``page_ref``, and reads a row with
    variants as the union of their fields.
    """
    if defs is not None:
        yield from _iter_described_through_refs(schema, prefix, defs, frozenset())
        return
    for name, node in (schema.get("properties") or {}).items():
        if not isinstance(node, dict):
            continue
        path = f"{prefix}{name}"
        yield path, node.get("description")
        if node.get("properties"):
            yield from iter_described_fields(node, prefix=f"{path}.")
        items = node.get("items")
        if isinstance(items, dict) and items.get("properties"):
            yield from iter_described_fields(items, prefix=f"{path}[].")


def _iter_described_through_refs(
    schema: dict[str, Any], prefix: str, defs: dict[str, Any], path_defs: frozenset[str]
) -> Iterator[tuple[str, str | None]]:
    properties: dict[str, Any] = {}
    for variant in [schema, *(schema.get("anyOf") or [])]:
        for name, node in (variant.get("properties") or {}).items():
            properties.setdefault(name, node)
    for name, node in properties.items():
        if not isinstance(node, dict):
            continue
        path = f"{prefix}{name}"
        ref = node.get("$ref", "")
        target = resolve_local(node, defs)
        yield path, target.get("description") if isinstance(target, dict) else None
        if not isinstance(target, dict) or _is_value_node(target):
            continue
        reached = path_defs | ({ref.split("/")[-1]} if ref else set())
        if ref and ref.split("/")[-1] in path_defs:
            continue  # a definition that contains itself
        if target.get("properties") or target.get("anyOf"):
            yield from _iter_described_through_refs(target, f"{path}.", defs, reached)
        items = target.get("items")
        if isinstance(items, dict):
            item = resolve_local(items, defs)
            if isinstance(item, dict) and not _is_value_node(item) and (
                    item.get("properties") or item.get("anyOf")):
                item_ref = items.get("$ref", "")
                yield from _iter_described_through_refs(
                    item, f"{path}[].", defs, reached | ({item_ref.split("/")[-1]} if item_ref else set()))


def assert_all_fields_described(doc_type: str, acord_form: str | None = None) -> None:
    """Fail loudly if any field lacks a ``description``.

    Descriptions are prompt text, so a missing one silently removes the model's
    only semantic anchor for that field (arch §0c). This is a IMPL-01 acceptance
    criterion, enforced here so it is checkable rather than aspirational.
    """
    missing = [
        path
        for path, desc in iter_described_fields(resolved_schema(doc_type, acord_form))
        if not (desc or "").strip()
    ]
    if missing:
        label = f"{doc_type}" + (f"/{acord_form}" if acord_form else "")
        raise SchemaError(
            f"{label}: {len(missing)} field(s) have no description, so the model gets no "
            f"semantic anchor for them (arch §0c): {', '.join(sorted(missing))}"
        )


def schema_selectors() -> list[tuple[str, str | None, str | None]]:
    """Every registered schema as ``(doc_type, acord_form, lob)``.

    The second level means different things per type — an ACORD form number, a
    policy's line of business — so a caller that reads it positionally pins the
    wrong thing. ``all_schema_keys`` keeps its two-tuple shape for the callers
    that only ever see ACORD.

    Unqualified selectors come first, so a caller that scans until it finds what
    it needs reads the small generic schemas before the large canonical ones.
    """
    out: list[tuple[str, str | None, str | None]] = []
    for key in _sources():
        if ":" not in key:
            out.append((key, None, None))
            continue
        doc_type, qualifier = key.split(":", 1)
        if doc_type == "acord":
            out.append((doc_type, qualifier, None))
        else:
            out.append((doc_type, None, qualifier))
    # Stable, and unqualified first — `sorted` is guaranteed stable, so within
    # each group the declaration order from `_sources` survives.
    return sorted(out, key=lambda sel: sel[1] is not None or sel[2] is not None)


def schema_text(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> str:
    """The schema as the prompt embeds it — titles, descriptions, enum values.

    Anything here reaches the model by construction: the prompt shows it the
    schema it must fill. So a phrase the schema itself uses cannot have been
    "leaked" by the alias registry, which is what the no-alias-leak guards exist
    to catch. A phrase in neither ("Pol. No.", "Underwritten By") appearing in a
    prompt means someone pasted the registry in.
    """
    return json.dumps(resolved_schema(doc_type, acord_form, lob, sections), ensure_ascii=False)


def all_schema_keys() -> list[tuple[str, str | None]]:
    """Every (doc_type, acord_form) pair with a schema — for tests and sweeps."""
    out: list[tuple[str, str | None]] = []
    for key in _SCHEMA_FILES:
        if ":" in key:
            doc_type, form = key.split(":", 1)
            out.append((doc_type, form))
        else:
            out.append((key, None))
    return out
