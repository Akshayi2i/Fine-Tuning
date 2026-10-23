"""Line of Business — enum, validation, and corpus coverage (arch §0b).

L1 (carrier registry) and L2 (structural inference) both attempt LoB detection
before a document reaches L3. When L3 is invoked, **the VLM is expected to detect
and output ``line_of_business`` itself**, so it is a field the model is trained
to produce — not metadata attached afterwards.

**It is a LIST** (arch v2.1 §0b). A certificate or a package policy routinely
covers several lines, and the v1 single-valued field forced the annotator to pick
one and discard the rest — which taught the model to do the same, and made
"correct" unachievable on exactly the documents that matter most. An empty list
means the document determines no line; it is not the same as a line outside the
enum, which goes in ``line_of_business_other`` (also a list, because a document
can name several).

Two rules this module exists to enforce:

* ``line_of_business`` is present in **every** golden label, for every document
  type, even when empty.
* Coverage target of **≥20% of training documents per LoB value**. Because a
  document can carry several lines the shares are per-document and do not sum to
  1. Under-coverage is a loud warning rather than a build failure: the remedy is
  collecting documents, which is a data-acquisition decision, not a build-time one.

Accuracy is reported **per LoB value** and is a gating metric (IMPL-08). It is
never averaged into overall field accuracy, because a rare class would then hide
inside a healthy-looking aggregate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from common.constants import LOB_COVERAGE_TARGET

LOB_ENUM_PATH = Path(__file__).resolve().parent.parent / "schemas" / "lob.enum.json"


class LobError(ValueError):
    """Raised on a value outside the LOB enum."""


#: Lines read as another line. Classic auto IS personal auto: one line of
#: business, one canonical schema (``personal_auto.json``, block ``auto``). The
#: Hagerty labels were already written that way. Applied wherever a line is
#: read - labels, bundle and label metadata, schema selection - so a stored
#: ``classic_auto`` never selects a schema, a split or a report row of
#: its own.
#:
#: ``watercraft`` is the L1 code for a personal boat policy (ACORD BOAT), and
#: the boat line is ``ocean_marine`` (lob_schema_map.yaml, decision D-B): a
#: policy sent as ``watercraft`` reads the ocean_marine schema and adapter
#: rather than being refused or read against the fallback.
MERGED_LINES: dict[str, str] = {"classic_auto": "personal_auto", "watercraft": "ocean_marine"}


def merge_line(lob: object) -> object:
    """``lob`` with every merged line replaced by the line it is read as, in the
    shape it came (a string, a list - de-duplicated, order kept - or None)."""
    if isinstance(lob, str):
        return MERGED_LINES.get(lob.strip().lower(), lob)
    if isinstance(lob, (list, tuple)):
        out: list[object] = []
        for value in lob:
            merged = merge_line(value)
            if merged not in out:
                out.append(merged)
        return out
    return lob


@lru_cache(maxsize=1)
def lob_values() -> tuple[str, ...]:
    """The supported LOB values, read from the one schema that defines them."""
    with LOB_ENUM_PATH.open(encoding="utf-8") as fh:
        return tuple(json.load(fh)["$defs"]["lob_value"]["enum"])


def is_valid_lob(value: str) -> bool:
    """Whether a single value is one of the supported lines."""
    return value in lob_values()


def normalize_lob(value: object) -> list[str]:
    """Coerce a label's ``line_of_business`` to the list form.

    Accepts the v1 scalar (and ``None``) so a corpus or golden label written
    before v2.1 still reads, rather than failing at a layer that cannot explain
    itself. Everything downstream sees a list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [merge_line(value)] if value else []
    if isinstance(value, (list, tuple)):
        return merge_line([v for v in value if v])
    raise LobError(
        f"line_of_business must be a list of values (arch v2.1 §0b), got {type(value).__name__}"
    )


def validate_lob(value: object) -> list[str]:
    """Return the normalized list if every entry is valid, else raise.

    An empty list is valid: it means the document does not determine a line. That
    is a correct label, not a missing one — and distinct from a line outside the
    enum, which belongs in ``line_of_business_other``.
    """
    values = normalize_lob(value)
    invalid = [v for v in values if not is_valid_lob(v)]
    if invalid:
        raise LobError(
            f"invalid line_of_business {invalid}; expected values from {lob_values()}. "
            "An empty list means the document does not determine a line — not 'unknown to "
            "the annotator'. A line outside the enum goes in line_of_business_other."
        )
    if len(set(values)) != len(values):
        raise LobError(f"line_of_business contains duplicates: {values}")
    return values


class LobCoverage(NamedTuple):
    """Per-value LoB distribution across a corpus, against the target.

    Multi-label: a document covering property and umbrella counts toward both, so
    ``shares`` are per-document and do NOT sum to 1. That is the honest reading —
    "what share of documents show the model this line" is the question coverage
    is asked to answer.
    """

    counts: dict[str, int]
    total: int
    shares: dict[str, float]
    under_target: dict[str, float]

    #: Documents whose LoB list is empty — the document determines no line. A
    #: correct label, not a missing one, which is why these are excluded from the
    #: share denominator rather than counted as a sixth class.
    undetermined_count: int

    #: Values present in the enum with zero documents. A different problem from
    #: under-coverage: there is nothing to be weak at, and no per-value accuracy
    #: to measure at all.
    @property
    def unmeasured_values(self) -> tuple[str, ...]:
        return tuple(v for v, c in sorted(self.counts.items()) if c == 0)

    @property
    def is_balanced(self) -> bool:
        return not self.under_target

    def warning(self) -> str | None:
        """A message naming the under-represented values, or ``None``."""
        if self.is_balanced:
            return None
        named = ", ".join(
            f"{value} ({share:.1%})"
            for value, share in sorted(self.under_target.items(), key=lambda kv: kv[1])
        )
        return (
            f"LoB coverage below the {LOB_COVERAGE_TARGET:.0%} target for: {named}. "
            f"The model will be weak on these values. The remedy is collecting documents, "
            f"not changing the build — see arch §0b."
        )


def compute_coverage(
    values: Iterable[object],
    target: float = LOB_COVERAGE_TARGET,
) -> LobCoverage:
    """Count LoB values across a corpus and flag under-representation.

    Each item is one document's ``line_of_business`` — a list under v2.1, or the
    v1 scalar, which ``normalize_lob`` accepts so an older corpus still reads.

    A document with an empty list is counted and reported but excluded from the
    share denominator: a legitimately-undetermined LoB is a correct label, not a
    missing one, so letting it dilute the shares would make coverage look worse
    than it is.
    """
    counter: Counter[str] = Counter()
    undetermined = 0
    documents_with_a_line = 0
    for value in values:
        lines = validate_lob(value)
        if not lines:
            undetermined += 1
            continue
        documents_with_a_line += 1
        for line in lines:
            counter[line] += 1

    counts = {v: counter.get(v, 0) for v in lob_values()}
    # Denominator is DOCUMENTS, not label-instances. With multi-label values a
    # label-instance denominator would shrink every share as soon as documents
    # started carrying two lines — coverage would appear to fall while the corpus
    # was strictly improving.
    shares = {
        v: (c / documents_with_a_line if documents_with_a_line else 0.0)
        for v, c in counts.items()
    }
    # The floor is capped at a share that is actually reachable. Under v1's
    # single-valued field there were five values and a flat 0.20 target WAS the
    # uniform share, so every corpus that was not perfectly uniform put at least
    # one value under it and the warning fired forever carrying no information.
    #
    # The architecture's intent is "no LoB value is starved", so the effective
    # floor is three quarters of the uniform share — strictly below it, and it
    # scales automatically if a value is added to the enum. It never exceeds the
    # configured target, so tightening LOB_COVERAGE_TARGET still tightens this.
    #
    # Multi-label makes the target easier to hit, not harder: a document covering
    # three lines feeds three counters. That is correct — the model really does
    # see that line in that document.
    uniform_share = 1.0 / len(shares) if shares else 0.0
    floor = min(target, uniform_share * 0.75)
    under = {v: s for v, s in shares.items() if 0.0 < s < floor}
    return LobCoverage(counts, documents_with_a_line, shares, under, undetermined)


# --------------------------------------------------------------------------
# A document's lines, as canonical schema names
# --------------------------------------------------------------------------


def lob_lines(lob: object) -> frozenset[str]:
    """A document's line(s) of business as canonical schema names.

    Accepts the enum spelling or the schema name (``workers_comp`` and ``wc`` are
    one line), one line or a list of them. A merged line is read as the line it
    is (``classic_auto`` is ``personal_auto``, :data:`MERGED_LINES`), so a corpus
    row or a request written before the merge still reads.
    """
    from common.schemas import LOB_SCHEMA_ALIASES

    if lob is None:
        return frozenset()
    values = [lob] if isinstance(lob, str) else list(lob) if isinstance(lob, (list, tuple)) else []
    out = set()
    for value in values:
        line = str(merge_line(str(value).strip().lower()))
        if line:
            out.add(LOB_SCHEMA_ALIASES.get(line, line))
    return frozenset(out)


def single_line(lob: object) -> str | None:
    """The one canonical line a policy is of, or None for a package (several
    lines) or a policy with no line.

    The adapter trains on, is scored on and serves single-line policies only: a
    package and a policy with no line are read against ``_fallback.json``, a
    shape the adapter is not taught, so the base model reads them.
    """
    found = lob_lines(lob)
    return next(iter(found)) if len(found) == 1 else None


def known_lines() -> frozenset[str]:
    """Every line with a canonical policy schema of its own."""
    from common.schemas import schema_selectors

    return frozenset(q for doc_type, _f, q in schema_selectors() if doc_type == "policy" and q)


def common_model_package(lines: frozenset[str] | set[str]) -> bool:
    """Whether a document naming these lines is a package of common-model lines.

    A package names several lines, so it is read against the client's
    ``_fallback.json`` (common.schemas.schema_key selects a line's schema only
    when exactly one is named). When its lines are SPEC_21 common-model lines,
    the adapter learned another shape for them, so neither training
    (``training.corpus_view``) nor serving (``ServedRelease.covers_lob``) gives
    such a package to it: the base model reads it.
    """
    if len(lines) < 2:
        return False
    from common.schemas import is_common_model

    return any(is_common_model("policy", None, line) for line in lines)


def parse_lines(value: str | None) -> frozenset[str] | None:
    """A ``--lines`` argument (``homeowners,personal_auto``) as canonical line
    names; ``None`` (every line) when it is empty. Raises :class:`LobError` on a
    line with no canonical policy schema, which would otherwise filter out
    every document without a word."""
    if not value or not value.strip():
        return None
    lines = lob_lines([part for part in value.split(",") if part.strip()])
    unknown = sorted(lines - known_lines())
    if unknown:
        raise LobError(f"--lines names {unknown}, which have no canonical policy schema; "
                       f"known lines: {sorted(known_lines())}")
    return lines


def within_lines(lob: object, lines: frozenset[str] | None) -> bool:
    """Whether a document of this line passes a ``--lines`` filter.

    No filter passes everything. A filter passes a document only when ALL its
    lines are in it - a package with one line outside is outside - and never a
    package of common-model lines, which is read against another shape
    (:func:`common_model_package`). A document with no recorded line is outside
    any filter: its line cannot be shown to be one asked for.
    """
    if not lines:
        return True
    found = lob_lines(lob)
    return bool(found) and found <= lines and not common_model_package(found)
