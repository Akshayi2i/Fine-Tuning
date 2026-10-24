"""The client's canonical ``FieldValue`` JSON, and the two shapes around it.

Every policy is extracted into the client's canonical schema
(``configs/canonical schema/LOB Schema``), where each leaf is an envelope::

    {"raw": "04/01/2026", "parsed": "04/01/2026",
     "confidence": {"score": 0.97, "source": "vlm"}, "page_ref": [1], "flagged": false}

Three shapes of one document meet here, and this module is the only place that
converts between them:

**Canonical** — the client's contract, what the endpoint returns and what a
golden label is written in. Validated against the client's file byte for byte.

**Model form** — what the model writes: the same tree, each leaf narrowed to
``raw``/``parsed``/``page_ref``, and **sparse** — a field the document does not
state is omitted rather than written as an envelope of nulls. Sparse is not a
style choice: homeowners declares 532 leaves, and an envelope for each is some
twenty thousand output tokens before the first value. ``confidence`` and
``flagged`` are never the model's to write; the pipeline fills them from the
calibrated logprobs and the review thresholds.

**Values view** — each envelope collapsed to its ``parsed`` value (``raw`` when
nothing was parsed). Calibration features, row completeness and field scoring
all read values, and reading them off this view means none of them had to learn
what an envelope is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from common.normalize import format_output_date, infer_field_kind


def page_numbers(refs: Any) -> list[int]:
    """The pages a ``page_ref`` names, in its order: whole numbers, written as
    numbers or as digits ("12"). Anything else ("p1", a range, a bool) names no
    page. A label or an answer citing one is not a page any code here can
    place: the value is then one with no recorded page, never a crash."""
    out: list[int] = []
    for ref in refs if isinstance(refs, list) else ():
        if isinstance(ref, bool):
            continue
        if isinstance(ref, int):
            out.append(ref)
        elif isinstance(ref, str) and ref.strip().isdigit():
            out.append(int(ref.strip()))
    return out


#: The keys of the client's ``FieldValue`` envelope.
FIELD_VALUE_KEYS: frozenset[str] = frozenset({"raw", "parsed", "confidence", "page_ref", "flagged"})

#: The subset the model writes (``common.schemas.MODEL_FIELD_VALUE``).
MODEL_VALUE_KEYS: tuple[str, ...] = ("raw", "parsed", "page_ref")

#: Written into ``confidence.source`` for every value this model produced. One of
#: the four the client's enum permits.
CONFIDENCE_SOURCE = "vlm"


class CanonicalLabelError(ValueError):
    """Raised when a label that should be canonical is not."""


def is_field_value(node: Any) -> bool:
    """Whether ``node`` is one leaf's envelope, in either the full or model form."""
    return (
        isinstance(node, dict)
        and "raw" in node
        and "parsed" in node
        and set(node) <= FIELD_VALUE_KEYS
    )


def _is_date(path: str) -> bool:
    return infer_field_kind(path) == "date"


def _output_value(path: str, value: Any) -> Any:
    """``parsed`` as it leaves the pipeline: a date in ``MM/DD/YYYY``.

    A date that cannot be read is kept as written, never replaced with ``None``:
    losing a value because the formatter could not parse it would turn a
    formatting question into a false null.
    """
    if value is None or not _is_date(path):
        return value
    return format_output_date(value) or value


def _join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


# --------------------------------------------------------------------------
# Canonical -> model form (training targets)
# --------------------------------------------------------------------------


#: Fields the system knows from the file itself, never from reading a page: the
#: PDF's file name and its page count. Every delivered label carries both, so the
#: model was trained to write a file name no page shows - teaching it to emit
#: values it cannot see - and scored as wrong on both in every document. They are
#: kept out of training targets and out of scoring, and serving fills them in
#: (with_system_fields).
#:
#: The common-model lines add three more the pipeline knows and no page prints:
#: the document's type and modality, and whether the policy is a package (it is
#: when it has more than one part). No self-contained schema has them.
SYSTEM_SUPPLIED_FIELDS: tuple[tuple[str, str], ...] = (
    ("document", "source_file_name"),
    ("document", "page_count"),
    ("document", "doc_type"),
    ("document", "modality"),
    ("policy", "is_package"),
)


def without_system_fields(document: Any) -> Any:
    """``document`` without :data:`SYSTEM_SUPPLIED_FIELDS`; a copy where any is present."""
    if not isinstance(document, dict):
        return document
    trimmed = document
    for section, name in SYSTEM_SUPPLIED_FIELDS:
        part = trimmed.get(section)
        if isinstance(part, dict) and name in part:
            if trimmed is document:
                trimmed = dict(document)
            trimmed[section] = {k: v for k, v in part.items() if k != name}
    return trimmed


def with_system_fields(
    output: dict[str, Any], *, page_count: int | None, source_file_name: str | None = None,
    lob: str | list[str] | None = None, modality: str | None = None,
) -> dict[str, Any]:
    """Fill :data:`SYSTEM_SUPPLIED_FIELDS` into a canonical (enveloped) output.

    From the request, at full confidence with source ``deterministic`` - what the
    schema's confidence source means for a value no model produced. A value the
    request does not know is left out, as the schema allows.

    A common-model line (``lob``) declares these as plain values, not envelopes,
    and gets the rest of what the pipeline supplies with them: the document's
    type and modality, a part when the model read none, whether the policy is a
    package, and the lists the schema requires.
    """
    from common.schemas import is_common_model

    # No line is the fallback, on the common model too since common model 1.1.0.
    if is_common_model("policy", None, lob):
        return _with_common_model_system_fields(
            output, page_count=page_count, source_file_name=source_file_name, lob=lob,
            modality=modality)
    known = {"source_file_name": source_file_name, "page_count": page_count}
    filled = dict(output)
    for section, name in SYSTEM_SUPPLIED_FIELDS:
        value = known.get(name)
        if value in (None, ""):
            continue
        part = dict(filled.get(section) or {})
        part[name] = {
            "raw": str(value), "parsed": value,
            "confidence": {"score": 1.0, "source": "deterministic"},
            "page_ref": [], "flagged": False,
        }
        filled[section] = part
    return filled


#: The modalities a common-model document records (SPEC_01 step 0), as the OCR
#: stage names them (data_pipeline.ocr.modality).
DOCUMENT_MODALITIES = ("native_pdf", "scanned_pdf")


def _with_common_model_system_fields(
    output: dict[str, Any], *, page_count: int | None, source_file_name: str | None,
    lob: str | list[str] | None, modality: str | None,
) -> dict[str, Any]:
    from common.schemas import line_of_key, load_schema, schema_key

    filled = dict(output)
    document = dict(filled.get("document") or {})
    document["doc_type"] = "policy_check"
    if modality in DOCUMENT_MODALITIES:
        document["modality"] = modality
    if page_count:
        document["page_count"] = int(page_count)
    if source_file_name:
        document["source_file_name"] = source_file_name
    filled["document"] = document
    line = line_of_key(schema_key("policy", None, lob))
    if not filled.get("lob_parts") and line is not None:
        # A single-line policy has one part, and the model is never asked for it
        # outside the declarations window. The schema requires at least one. A
        # policy of no known line (the fallback) has no part to state.
        filled["lob_parts"] = [{"part_id": "part_1", "lob": line}]
    filled["policy"] = {**(filled.get("policy") or {}), "is_package": len(filled.get("lob_parts") or []) > 1}
    for name in load_schema("policy", None, lob).get("required") or []:
        if name not in filled:
            filled[name] = [] if name in ("lob_parts", "coverages") else {}
    return filled


def without_bare_values(node: Any) -> Any:
    """``node`` with only its envelopes: no common-model id, reference, code or type.

    Calibration scores what a model READ, from the tokens it read it with. A bare
    value is the model's bookkeeping - which vehicle a coverage applies to, its
    code - with no printed span to measure, and scored it would be flagged on
    every field of every document for having no confidence.
    """
    if is_field_value(node):
        return node
    if isinstance(node, dict):
        return {k: without_bare_values(v) for k, v in node.items() if _holds_readings(v)}
    if isinstance(node, list):
        return [without_bare_values(v) for v in node if isinstance(v, (dict, list))]
    return node


def _holds_readings(value: Any) -> bool:
    """An object, or a list with an object in it: not a bare value, nor a list of them."""
    if isinstance(value, dict):
        return True
    return isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value)


def to_model_target(
    label: dict[str, Any], *, required: Iterable[str] = (), schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The training target for a canonical golden label.

    Narrows every envelope to ``raw``/``parsed``/``page_ref``, writes dates in
    ``MM/DD/YYYY``, and drops what the document does not state — the exact shape
    the prompt asks the model to produce, so the target and the instruction agree.

    ``required`` are the schema's top-level required objects. They are always
    emitted, as ``{}`` when empty, because the canonical schema requires them and
    a target that omitted them would teach an output that fails validation.

    ``schema`` - the model view the target is written against - is given for a
    common-model line. Its ids, references, codes and types are bare strings the
    schema declares, and are kept; only a ``DateValue`` is reformatted, so a form
    edition printed ``05 11`` is never read as a date because of its name.

    Raises :class:`CanonicalLabelError` for a label with a bare value where an
    envelope belongs — a flat, pre-canonical label. Training on one under a
    canonical prompt would teach the model the wrong shape, silently.
    """
    if not isinstance(label, dict):
        raise CanonicalLabelError(f"a canonical label is a JSON object, not {type(label).__name__}")
    # Never taught: the system supplies them (SYSTEM_SUPPLIED_FIELDS).
    stripped = without_system_fields(label)
    target = (_slim(stripped, "") if schema is None else _slim_against(stripped, schema)) or {}
    for key in required:
        target.setdefault(key, {})
    return target


def _slim(node: Any, path: str) -> Any:
    if is_field_value(node):
        if node.get("raw") is None and node.get("parsed") is None:
            return None
        return {
            "raw": node.get("raw"),
            "parsed": _output_value(path, node.get("parsed")),
            "page_ref": page_numbers(node.get("page_ref")),
        }
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            slim = _slim(value, _join(path, key))
            if slim not in (None, {}, []):
                out[key] = slim
        return out
    if isinstance(node, list):
        rows = [_slim(item, f"{path}[{index}]") for index, item in enumerate(node)]
        return [row for row in rows if row not in (None, {}, [])]
    if node is None:
        return None
    raise CanonicalLabelError(
        f"{path or '<root>'} holds a bare value ({node!r}) where the canonical schema puts a "
        "FieldValue envelope. This is a flat, pre-canonical label; convert it to the canonical "
        "schema before it enters the corpus."
    )


def _is_value_schema(node: Any) -> bool:
    return isinstance(node, dict) and {"raw", "parsed", "page_ref"} <= set(node.get("properties") or {})


def _schema_branch(
    sub: Any, value: Any, defs: dict[str, Any], seen: str | None = None
) -> tuple[dict[str, Any], str | None]:
    """The schema ``value`` is written against, with the name of the definition
    it came from: ``$ref`` followed, and of ``anyOf`` variants the one whose
    properties hold every key ``value`` has (else the one holding most)."""
    name = seen
    while isinstance(sub, dict) and isinstance(sub.get("$ref"), str):
        ref = sub["$ref"]
        if not ref.startswith("#/$defs/"):
            return {}, name
        name = ref.rsplit("/", 1)[-1]
        sub = defs.get(name)
    if not isinstance(sub, dict):
        return {}, name
    branches = sub.get("anyOf") or sub.get("oneOf")
    if branches and "properties" not in sub and "items" not in sub:
        keys = set(value) if isinstance(value, dict) else set()
        best: tuple[dict[str, Any], str | None] = ({}, name)
        best_score = -1
        for branch in branches:
            resolved, _ = _schema_branch(branch, value, defs, name)
            props = set(resolved.get("properties") or {})
            if isinstance(value, dict) and keys <= props:
                return resolved, name
            score = len(keys & props) if isinstance(value, dict) else (
                1 if isinstance(value, list) and "items" in resolved else 0)
            if score > best_score:
                best, best_score = (resolved, name), score
        return best
    return sub, name


def _slim_against(node: Any, schema: dict[str, Any]) -> Any:
    """:func:`_slim`, read against a common-model view."""
    defs = schema.get("$defs") or {}

    def walk(value: Any, sub: Any, path: str) -> Any:
        resolved, name = _schema_branch(sub, value, defs)
        if is_field_value(value):
            if value.get("raw") is None and value.get("parsed") is None:
                return None
            parsed = value.get("parsed")
            if name == "DateValue" and parsed is not None:
                parsed = format_output_date(parsed) or parsed
            return {"raw": value.get("raw"), "parsed": parsed,
                    "page_ref": page_numbers(value.get("page_ref"))}
        if isinstance(value, dict):
            props = resolved.get("properties") or {}
            out = {}
            for key, item in value.items():
                slim = walk(item, props.get(key), _join(path, key))
                if slim not in (None, {}, []):
                    out[key] = slim
            return out
        if isinstance(value, list):
            rows = [walk(item, resolved.get("items"), f"{path}[{i}]") for i, item in enumerate(value)]
            return [row for row in rows if row not in (None, {}, [])]
        if value is None:
            return None
        if resolved and not _is_value_schema(resolved):
            return value  # an id, a reference, a code or a type: the schema's own bare value
        raise CanonicalLabelError(
            f"{path or '<root>'} holds a bare value ({value!r}) where the schema puts a "
            "value envelope."
        )

    return walk(node, schema, "")


def in_schema_order(node: Any, schema: dict[str, Any]) -> Any:
    """``node`` with every object's keys in the order its schema declares them.

    Structured decoding (xgrammar) builds each object's grammar from its
    properties IN ORDER: a key may be skipped, never written after one declared
    later. Labels keep whatever order they were written in - some template
    families put ``form_title`` before ``form_number`` - and the targets kept it,
    so the model learned to open a form with its title, after which the grammar
    no longer allowed the number. On the smoke run every form came back without
    one. Training on schema order makes what the model learned writable.

    Keys the schema does not declare keep their place after the declared ones.
    Values are untouched; only key order changes.
    """
    defs = schema.get("$defs") or {}

    def walk(value: Any, sub: Any) -> Any:
        # A row with variants (a limit, a deductible) is ordered by the variant
        # it is written against: every variant lists its fields in one order.
        sub, _ = _schema_branch(sub, value, defs)
        if isinstance(value, dict):
            props = sub.get("properties") or {}
            ordered = {key: walk(value[key], props[key]) for key in props if key in value}
            for key, item in value.items():
                if key not in ordered:
                    ordered[key] = walk(item, None)
            return ordered
        if isinstance(value, list):
            return [walk(item, sub.get("items")) for item in value]
        return value

    return walk(node, schema)


def within_schema(node: Any, schema: dict[str, Any]) -> tuple[Any, list[str]]:
    """``node`` without the keys its schema does not declare, and their paths.

    Labels carry structure the canonical schema has no place for: annotation
    blocks (``fideon:*``, ``text_sections``, ``additional_fields``) and nested
    shapes of their own (``dwelling_fire.dwellings[].coverages[]`` where the
    schema has ``dwelling`` and ``property_coverages``) - 18% of the values
    inside schema sections, half of dwelling fire's. Decoding is held to the
    schema, so such a key can never be written: kept in a training target it
    teaches a key the grammar refuses, and kept in a scored label it is a miss
    no model can fix.

    Only where the schema DECLARES an object's properties is anything dropped;
    an object or array whose schema says nothing is kept whole.
    """
    defs = schema.get("$defs") or {}
    dropped: list[str] = []

    def resolve(sub: Any, value: Any) -> dict[str, Any]:
        while isinstance(sub, dict):
            if "$ref" in sub:
                ref = sub["$ref"]
                sub = defs.get(ref.rsplit("/", 1)[-1]) if ref.startswith("#/$defs/") else None
                continue
            branches = sub.get("anyOf") or sub.get("oneOf")
            if branches and "properties" not in sub and "items" not in sub:
                want = "properties" if isinstance(value, dict) else "items"
                resolved = [resolve(b, value) for b in branches]
                sub = next((b for b in resolved if want in b), {})
            break
        return sub if isinstance(sub, dict) else {}

    def walk(value: Any, sub: Any, path: str) -> Any:
        if is_field_value(value):
            return value
        sub = resolve(sub, value)
        if isinstance(value, dict):
            props = sub.get("properties")
            if not props:
                return value
            kept = {}
            for key, item in value.items():
                if key in props:
                    kept[key] = walk(item, props[key], _join(path, key))
                else:
                    dropped.append(_join(path, key))
            return kept
        if isinstance(value, list) and sub.get("items") is not None:
            return [walk(item, sub["items"], f"{path}[{i}]") for i, item in enumerate(value)]
        return value

    return walk(node, schema, ""), dropped


#: The confidence source of a value the pipeline wrote because the schema has
#: the key and nothing was extracted for it - the schema's "structural".
SKELETON_SOURCE = "structural"


def empty_field_value() -> dict[str, Any]:
    """An envelope for a key the model gave no value: null, score 0, not flagged.

    Not flagged: most of a schema's fields are simply not on a given policy, and
    flagging every one would bury the fields that do need a person.
    """
    return {"raw": None, "parsed": None,
            "confidence": {"score": 0.0, "source": SKELETON_SOURCE},
            "page_ref": [], "flagged": False}


def with_all_keys(output: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """``output`` with every key its canonical schema declares, in schema order.

    The model writes only what it found - asking it to write 1,500 keys per
    window would multiply output length and latency for nulls - so the served
    JSON had a different key set for every document and for every model. This
    fills the rest after the fact: a missing value becomes
    :func:`empty_field_value`, a missing object its own full set of keys, a
    missing table ``[]``. A table's rows each get every key of a row; the NUMBER
    of rows is what the model found, and is not padded. Values present are
    never changed. So that the result stays valid, a key is left absent where
    filling it would break the schema: a bare (non-envelope) field the schema
    does not allow to be null, a missing table the schema requires rows in
    (``minItems``), and a key that requires others (``dependentRequired``)
    the fill cannot supply.
    """
    defs = schema.get("$defs") or {}

    def resolve(sub: Any, value: Any) -> dict[str, Any]:
        for _ in range(20):
            if not isinstance(sub, dict):
                return {}
            if "$ref" in sub:
                ref = sub["$ref"]
                sub = defs.get(ref.rsplit("/", 1)[-1]) if ref.startswith("#/$defs/") else None
                continue
            branches = sub.get("anyOf") or sub.get("oneOf")
            if branches and "properties" not in sub and "items" not in sub:
                want = "items" if isinstance(value, list) else "properties"
                sub = next((b for b in (resolve(b, value) for b in branches) if want in b), {})
                continue
            return sub
        return {}

    def is_envelope_schema(sub: dict[str, Any]) -> bool:
        return {"raw", "parsed", "page_ref"} <= set(sub.get("properties") or {})

    def allows_null(sub: dict[str, Any]) -> bool:
        kind = sub.get("type")
        return kind == "null" or (isinstance(kind, list) and "null" in kind)

    missing = object()

    def walk(value: Any, sub: Any, depth: int) -> Any:
        sub = resolve(sub, value)
        if depth > 30 or not sub:
            return None if value is missing else value
        if is_envelope_schema(sub):
            return empty_field_value() if value is missing or value is None else value
        props = sub.get("properties")
        if props:
            if value is missing or value is None:
                value = {}
            if not isinstance(value, dict):
                return value
            out: dict[str, Any] = {}
            for key, child in props.items():
                filled = walk(value.get(key, missing), child, depth + 1)
                if filled is not missing:
                    out[key] = filled
            out.update({k: v for k, v in value.items() if k not in out})
            # A key that requires others (a limit's `percentage` requires its
            # `basis_coverage_code`, a bare code the fill cannot invent) is
            # taken out again when the fill added it and those others are not
            # all there. A key the model wrote stays, whatever it lacks.
            for key, needs in _dependent_required(sub).items():
                if key in out and key not in value and not all(n in out for n in needs):
                    del out[key]
            return out
        if sub.get("type") == "array" or "items" in sub:
            if value is missing and (sub.get("minItems") or 0) > 0:
                # [] would break the minimum (a form's page_range is a pair):
                # a table that must have rows is left absent, not emptied.
                return missing
            if value is missing or value is None:
                return []
            if not isinstance(value, list):
                return value
            return [walk(item, sub.get("items"), depth + 1) for item in value]
        if value is missing:
            return None if allows_null(sub) else missing
        return value

    filled = walk(output, schema, 0)
    return filled if isinstance(filled, dict) else output


def _dependent_required(sub: dict[str, Any]) -> dict[str, list[str]]:
    """An object schema's ``dependentRequired``, with the list form of the older
    ``dependencies`` keyword folded in: key -> the keys it requires."""
    out: dict[str, list[str]] = {}
    for keyword in ("dependencies", "dependentRequired"):
        for key, needs in (sub.get(keyword) or {}).items():
            if isinstance(needs, list):
                out.setdefault(key, []).extend(n for n in needs if isinstance(n, str))
    return out


def schema_label(
    label: Any,
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> Any:
    """A golden label as training and scoring use it: moved into its line's
    block (configs/label_mappings.yaml), then narrowed to what the line's
    canonical schema can hold. Flat document types are returned as they are.

    On a common-model line, a part whose line is read as this one
    (``common.lob.merge_line``: classic auto is personal auto) is written as
    this line, which is what the model view holds a part to and what training
    teaches. Done here, where training (``window_target``) and scoring
    (``build_report``, both sides) both start, so the two cannot drift: a
    classic-auto gold scored the answer training taught as a wrong part.
    """
    from common.label_mapping import map_label
    from common.schemas import SchemaError, is_canonical, is_common_model, resolved_schema

    label = map_label(label, lob)
    try:
        canonical = isinstance(label, dict) and is_canonical(doc_type, acord_form, lob)
    except SchemaError:
        # No selectable schema (an ACORD document with no form): nothing to
        # narrow to. Scoring counts such a document invalid on its own terms.
        return label
    if not canonical:
        return label
    if is_common_model(doc_type, acord_form, lob):
        label = _parts_as_line(label, doc_type, acord_form, lob)
    return within_schema(label, resolved_schema(doc_type, acord_form, lob))[0]


def _parts_as_line(
    label: dict[str, Any], doc_type: str, acord_form: str | None, lob: str | list[str] | None,
) -> dict[str, Any]:
    """``label`` with each ``lob_parts[].lob`` that ``merge_line`` reads as the
    schema's line written as that line; a copy when one is rewritten. A part
    of another line is left as it is (training refuses such a label)."""
    from common.lob import merge_line
    from common.schemas import line_of_key, schema_key

    parts = label.get("lob_parts")
    if not isinstance(parts, list):
        return label
    line = line_of_key(schema_key(doc_type, acord_form, lob))
    if line is None:
        return label                 # the fallback: no line to own a part

    def own(part: Any) -> Any:
        name = part.get("lob") if isinstance(part, dict) else None
        if not isinstance(name, str) or name == line or merge_line(name.strip().lower()) != line:
            return part
        return {**part, "lob": line}

    rewritten = [own(part) for part in parts]
    if all(new is old for new, old in zip(rewritten, parts, strict=True)):
        return label
    return {**label, "lob_parts": rewritten}


def training_target(
    label: dict[str, Any],
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
) -> dict[str, Any]:
    """What the model is trained to write for one golden label.

    Chosen by the same selectors that choose the prompt's schema, so the target
    and the prompt describe one shape: the sparse model form for a canonical
    policy, the label itself for a flat type — dates in ``MM/DD/YYYY`` either way.
    Golden labels keep whatever date format they were written in; only the target
    is converted.
    """
    from common.schemas import is_canonical, required_fields, resolved_schema

    # Moved into the line's block, narrowed to what the schema can hold.
    label = schema_label(label, doc_type, acord_form, lob)
    # In the order the decoding grammar writes keys (in_schema_order).
    schema = resolved_schema(doc_type, acord_form, lob)
    if is_canonical(doc_type, acord_form, lob):
        target = to_model_target(label, required=required_fields(doc_type, acord_form, lob))
        return in_schema_order(target, schema)
    return in_schema_order(with_output_dates(label), schema)


# --------------------------------------------------------------------------
# Model form -> canonical (serving)
# --------------------------------------------------------------------------


def with_output_dates(node: Any, path: str = "") -> Any:
    """Every date in ``MM/DD/YYYY``, whichever shape the document is in.

    In a canonical document dates live in an envelope's ``parsed`` (``raw`` stays
    as printed); in a flat one — ACORD, Loss Run — they are the values
    themselves. Applied to training targets and to every generation, so the
    format is a guarantee rather than something the prompt merely asks for.
    """
    if is_field_value(node):
        return {**node, "parsed": _output_value(path, node.get("parsed"))}
    if isinstance(node, dict):
        return {k: with_output_dates(v, _join(path, k)) for k, v in node.items()}
    if isinstance(node, list):
        return [with_output_dates(v, f"{path}[{i}]") for i, v in enumerate(node)]
    return _output_value(path, node)


def envelope(
    generated: dict[str, Any],
    confidence: Mapping[str, tuple[float, bool]],
) -> dict[str, Any]:
    """The model's output as the client's canonical JSON.

    ``confidence`` maps a values-view path (``policy.effective_date``,
    ``locations[0].address.city``) to ``(calibrated score, needs review)``. A
    leaf with no entry gets score 0 and is flagged: a field with no measured
    confidence is one a person has to look at, never one that ships as certain.

    A list of envelopes is one field in the values view (it is scored as a set),
    so each of its elements carries that field's score.
    """
    return _envelope(generated, "", confidence)


def _envelope(node: Any, path: str, confidence: Mapping[str, tuple[float, bool]]) -> Any:
    if is_field_value(node):
        score, flagged = confidence.get(path, (0.0, True))
        return {
            "raw": node.get("raw"),
            "parsed": node.get("parsed"),
            "confidence": {"score": round(float(score), 4), "source": CONFIDENCE_SOURCE},
            # Once each, in order: the client's schema holds page_ref to unique
            # items, and a value a merge joined across windows can cite one twice.
            "page_ref": sorted(set(page_numbers(node.get("page_ref")))),
            "flagged": bool(flagged),
        }
    if isinstance(node, dict):
        return {k: _envelope(v, _join(path, k), confidence) for k, v in node.items()}
    if isinstance(node, list):
        if node and all(is_field_value(item) for item in node):
            return [_envelope(item, path, confidence) for item in node]
        return [_envelope(item, f"{path}[{i}]", confidence) for i, item in enumerate(node)]
    return node


# --------------------------------------------------------------------------
# Values view (calibration, completeness, scoring)
# --------------------------------------------------------------------------


def values_view(node: Any) -> Any:
    """Every envelope collapsed to its value: ``parsed``, else ``raw``.

    A no-op on a document with no envelopes, so a caller can apply it without
    first asking which kind of document it holds.
    """
    if is_field_value(node):
        parsed = node.get("parsed")
        return parsed if parsed is not None else node.get("raw")
    if isinstance(node, dict):
        return {k: values_view(v) for k, v in node.items()}
    if isinstance(node, list):
        return [values_view(v) for v in node]
    return node


def printed_view(node: Any) -> Any:
    """Every envelope collapsed to what the PAGE shows: ``raw``, else ``parsed``.

    For aligning a label against OCR text, where the printed form is the one
    that can be found — ``April 1, 2026``, not ``04/01/2026``. A no-op on a flat
    document.
    """
    if is_field_value(node):
        raw = node.get("raw")
        return raw if raw is not None else node.get("parsed")
    if isinstance(node, dict):
        return {k: printed_view(v) for k, v in node.items()}
    if isinstance(node, list):
        return [printed_view(v) for v in node]
    return node


def has_envelopes(node: Any) -> bool:
    """Whether a document is in the canonical shape at all."""
    if is_field_value(node):
        return True
    if isinstance(node, dict):
        return any(has_envelopes(v) for v in node.values())
    if isinstance(node, list):
        return any(has_envelopes(v) for v in node)
    return False


def leaf_values(node: Any, path: str = "") -> dict[str, Any]:
    """``path -> value`` for every field, envelopes collapsed to their value.

    Rows are indexed (``locations[0].address.city``); a list of scalars is one
    field, kept whole.
    """
    if is_field_value(node):
        return {path: values_view(node)}
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            out.update(leaf_values(value, _join(path, key)))
        return out
    if isinstance(node, list):
        if all(is_field_value(v) or not isinstance(v, dict | list) for v in node):
            return {path: values_view(node)}
        out = {}
        for index, value in enumerate(node):
            out.update(leaf_values(value, f"{path}[{index}]"))
        return out
    return {path: node}


def field_paths(node: Any, path: str = "") -> set[str]:
    """Every field path in a document: ``named_insured.primary_name``, and for a
    flat document its plain keys. An envelope is one field; its keys are not."""
    if is_field_value(node) or not isinstance(node, dict | list):
        return {path} if path else set()
    if isinstance(node, list):
        return {p for i, v in enumerate(node) for p in field_paths(v, f"{path}[{i}]")}
    out: set[str] = set()
    for key, value in node.items():
        child = _join(path, key)
        out.add(child)
        out |= field_paths(value, child)
    return out


def collapse_spans(spans: Mapping[str, Any]) -> dict[str, Any]:
    """Re-key token spans from envelope paths to values-view paths.

    The span mapper addresses every scalar it finds, so a canonical leaf arrives
    as ``policy.effective_date.raw``, ``….parsed`` and ``….page_ref[0]``. The
    field's span is ``raw``'s - the reading of the page, written first. Doubt
    about a character shows there; ``parsed`` is the same value reformatted,
    written after it and largely decided by it, so its tokens are near-certain
    whether the reading was right or not. With no ``raw``, ``parsed``'s stands
    in. ``page_ref`` tokens say where a value was found, not what it is, and are
    dropped.

    Spans with no envelope suffix are returned untouched, so a flat document
    passes through unchanged.
    """
    if not any(path.endswith(".parsed") for path in spans):
        return dict(spans)

    out: dict[str, Any] = {}
    for path, span in spans.items():
        if not path.endswith(".parsed"):
            continue
        field = path[: -len(".parsed")]
        raw = spans.get(f"{field}.raw")
        if raw is not None and getattr(raw, "value", None) not in (None, ""):
            span = raw
        out[field] = span
    return out
