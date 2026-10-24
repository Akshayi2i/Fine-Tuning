"""False nulls and hallucinations (arch v2.1 §15.2).

Metrics that share a shape — each counts a specific way an
extraction can be wrong that **field accuracy cannot see** — and that is why they
are separate metrics rather than folded into one number.

* **False null.** The value is on the page and the model emitted ``null``. An
  aggregate field score counts this as one miss among many, but it is the failure
  mode that quietly halves a form's usefulness while looking like ordinary
  imperfection. It also produces no low-confidence signal: a null has no tokens
  to be uncertain about, so §5 confidence is blind to it and only this metric
  sees it.

* **Hallucination.** A value was emitted that appears nowhere in the document. A
  fluent, plausible, invented policy number scores as one wrong field — but a
  wrong value is worse than an absent one, because an absent one gets reviewed
  and a confident wrong one gets used.

Page-selection recall is gone: every page is read, chunk by chunk, never
selected pages (decision 2026-10-19), so there is no selector to score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evaluation.metrics.field_accuracy import values_agree


@dataclass
class FaultReport:
    """One fault type's rate, with the instances that produced it."""

    name: str
    hits: int = 0
    opportunities: int = 0
    instances: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rate(self) -> float | None:
        """Hits per opportunity; ``None`` when there was no opportunity.

        Not 0.0: zero is the BEST false-null or invention rate, so an eval in
        which nothing could be counted (no gold value, nothing checkable
        written) read as a perfect score, and the gate passed it as measured.
        Absent is "not measured", as elsewhere in the eval report (ECE, LoB
        accuracy): ``build_report`` drops a None metric, and the gate blocks on
        a gating metric it was not given.
        """
        return self.hits / self.opportunities if self.opportunities else None

    def as_dict(self) -> dict[str, Any]:
        rate = self.rate
        return {
            "metric": self.name,
            "rate": round(rate, 4) if rate is not None else None,
            "hits": self.hits,
            "opportunities": self.opportunities,
            # Capped: the rate is the metric, the instances are for diagnosis,
            # and a report carrying every instance of a bad run is unreadable.
            "instances": self.instances[:25],
        }


def _is_empty(value: Any) -> bool:
    if isinstance(value, (list, dict)):
        return not value
    if isinstance(value, str):
        return not value.strip()
    return value is None


def score_false_nulls(
    documents: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    source_ids: Sequence[str] | None = None,
) -> FaultReport:
    """Fields the golden label has a value for, which the model emitted as null.

    The denominator is **opportunities** — fields that actually had a value —
    not all fields. Dividing by every field would let a schema with many
    legitimately-empty fields dilute the rate toward zero.
    """
    from evaluation.metrics.field_accuracy import flatten_scalars

    report = FaultReport("false_null_rate")
    for index, (expected, got) in enumerate(documents):
        source_id = source_ids[index] if source_ids and index < len(source_ids) else str(index)
        expected_flat = flatten_scalars(expected)
        got_flat = flatten_scalars(got)

        for path, truth in expected_flat.items():
            if _is_empty(truth):
                continue
            report.opportunities += 1
            if _is_empty(got_flat.get(path)):
                report.hits += 1
                report.instances.append({
                    "source_id": source_id, "field": path, "expected": str(truth)[:80],
                })
    return report


def score_hallucinations(
    documents: Sequence[tuple[dict[str, Any], dict[str, Any], str | Mapping[int, Any]]],
    *,
    overflow: bool = False,
) -> FaultReport:
    """Values the model wrote that no page of the document prints.

    Each tuple is ``(expected, got, text)``: the text of the document's pages,
    as one string or as page number -> that page's text (or the page prepared
    once, ``common.grounding.TolerantPage``). Checking against the golden label
    alone would call every wrong value a hallucination, including an honest
    misread of something printed on the page, and those have different
    remedies: a misread wants better perception, an invention wants better
    grounding.

    Looked for as serving's printed-values check looks (``serving.printed_values``,
    decision 2026-10-24), so checkpoint selection reads the rate the gate reads:
    every value the model wrote (:func:`_written`: each envelope's ``raw``, extra
    fields and list members included), by :func:`common.grounding.ground_tolerant`
    over every page - the value AS PRINTED ("$2,100,000", not the reformatted
    2100000.0), forgiving the misreads OCR makes. Printed only on a page it does
    not cite is a wrong page, not an invention. A value the text cannot settle -
    too short to look for ("1", "NY"), a reading (a bool), or one citing a page
    with no text - is left out of the rate rather than counted as printed.

    Measured on the model's answer: every value it wrote, right or wrong by the
    label. A right value written in a form no page prints is one serving holds
    back too, so the label does not excuse it. ``expected`` is not read.

    ``overflow`` scores the extra fields (``additional_fields``) alone, reported
    beside the rate: overflow is free text by design, and inside the one rate
    it can hide how often the schema's own fields are invented.

    A field is only checked when the model emitted something. Correctly-absent
    values cannot be hallucinated, and checking them would make the denominator
    meaningless.
    """
    from common import grounding

    report = FaultReport("additional_fields_hallucination_rate" if overflow else "hallucination_rate")
    for _expected, got, text in documents:
        by_page = isinstance(text, Mapping)
        # Prepared once per document: a policy has hundreds of values to look up.
        pages = grounding.tolerant_pages({int(n): t for n, t in text.items()} if by_page else {1: text or ""})

        for path, forms, cited in _written(got, bare_printed=not _has_envelope(got)):
            forms = tuple(form for form in forms if not _is_empty(form))
            if not forms or (overflow and not path.startswith("additional_fields")):
                continue
            where = _first_printed(grounding, forms, cited if by_page else None, pages)
            if where.status == grounding.UNCHECKED:
                continue
            report.opportunities += 1
            if where.status == grounding.NOT_PRINTED:
                report.hits += 1
                report.instances.append({
                    "field": path, "emitted": str(forms[0])[:80],
                    "reason": "printed on no page of the document",
                })
    return report


def _first_printed(grounding: Any, forms: tuple[Any, ...], cited: Any, pages: Any) -> Any:
    """The verdict on a value printed in any of ``forms``: the first form a page
    prints, else the first form's verdict - as serving's ``_grounded`` decides."""
    verdicts = []
    for form in forms:
        found = grounding.ground_tolerant(form, cited, pages)
        if found.printed:
            return found
        verdicts.append(found)
    return verdicts[0]


@dataclass
class PageRefReport:
    """Whether right values cite the right pages."""

    #: Right values whose label records pages.
    compared: int = 0
    #: Of those, the ones citing exactly the label's pages.
    exact: int = 0
    #: Pages the model cited, and how many of them the label cites too.
    cited: int = 0
    cited_right: int = 0
    #: Pages the label cites.
    gold: int = 0

    @property
    def exact_rate(self) -> float | None:
        return self.exact / self.compared if self.compared else None

    @property
    def precision(self) -> float | None:
        return self.cited_right / self.cited if self.cited else None

    @property
    def recall(self) -> float | None:
        return self.cited_right / self.gold if self.gold else None


def score_page_refs(documents: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> PageRefReport:
    """Whether a correct value's ``page_ref`` lists the pages the label lists.

    A value printed several times appears once, citing every page that prints it
    (the agreed convention). A right value citing a wrong page sends a reviewer
    to the wrong page; one missing a page hides where else it is printed. Field
    accuracy compares values only and sees neither.

    Only right values (field match's own check) whose label records pages: a
    wrong value's pages say nothing about citing.
    """
    report = PageRefReport()
    for expected, got in documents:
        gold = {path: (value, cited) for path, value, _printed, cited in _emitted(expected)}
        for path, value, _printed, cited in _emitted(got):
            truth = gold.get(path)
            if truth is None or not truth[1] or _is_empty(value) or not values_agree(truth[0], value, path):
                continue
            want, have = {int(p) for p in truth[1]}, {int(p) for p in cited}
            report.compared += 1
            report.exact += want == have
            report.cited += len(have)
            report.cited_right += len(have & want)
            report.gold += len(want)
    return report


def _emitted(node: Any, prefix: str = "", *, bare_printed: bool = True) -> Any:
    """Each emitted value as ``(path, value, printed, cited pages)``.

    Paths and values as :func:`evaluation.metrics.field_accuracy.flatten_scalars`
    gives them, so the gold lines up; ``printed`` is what the page shows: an
    envelope's ``raw`` (else its ``parsed``), or a bare value itself when
    ``bare_printed``. In a canonical answer a bare value is a code, an id, a
    link or a section hint, which no page prints, so there it is ``None`` - as is
    a list of values, a set of codes rather than a printed phrase.
    """
    from common.canonical import is_field_value, values_view

    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if is_field_value(value):
            raw = value.get("raw")
            yield path, values_view(value), raw if raw not in (None, "") else value.get("parsed"), \
                value.get("page_ref") or []
        elif isinstance(value, dict):
            yield from _emitted(value, f"{path}.", bare_printed=bare_printed)
        elif isinstance(value, list):
            if value and all(is_field_value(item) for item in value):
                yield path, values_view(value), None, []
            elif any(isinstance(item, dict) for item in value):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        yield from _emitted(item, f"{path}[{index}].", bare_printed=bare_printed)
                    else:
                        yield f"{path}[{index}]", item, item if bare_printed else None, []
            else:
                yield path, value, None, []
        else:
            yield path, value, value if bare_printed else None, []


def _written(node: Any, prefix: str = "", *, bare_printed: bool = True) -> Any:
    """Each value written, in the forms a page prints it, as ``(path, forms,
    cited pages)``: every envelope's :func:`common.grounding.printed_forms` -
    its ``raw``, or with ``raw`` blank the forms its ``parsed`` prints in, as
    serving's printed-values check looks them up (an invented ``parsed`` behind
    a blank ``raw`` is still invented, and served) - each of a list of
    envelopes, the extra fields' too; and in a flat answer (``bare_printed``)
    each bare value itself. A canonical answer's bare values are codes, ids,
    links and section hints, which no page prints; a list of bare values is a
    set of codes, not a printed phrase."""
    from common.canonical import is_field_value
    from common.grounding import printed_forms

    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if is_field_value(value):
            yield path, printed_forms(value), value.get("page_ref") or []
        elif isinstance(value, dict):
            yield from _written(value, f"{path}.", bare_printed=bare_printed)
        elif isinstance(value, list):
            table = any(isinstance(item, dict) for item in value)
            for index, item in enumerate(value):
                if is_field_value(item):
                    yield f"{path}[{index}]", printed_forms(item), item.get("page_ref") or []
                elif isinstance(item, dict):
                    yield from _written(item, f"{path}[{index}].", bare_printed=bare_printed)
                elif table and bare_printed:
                    yield f"{path}[{index}]", (item,), []
        elif bare_printed:
            yield path, (value,), []


def _has_envelope(node: Any) -> bool:
    from common.canonical import is_field_value

    if is_field_value(node):
        return True
    if isinstance(node, dict):
        return any(_has_envelope(value) for value in node.values())
    if isinstance(node, list):
        return any(_has_envelope(item) for item in node)
    return False
