"""The six tasks the unified model is trained on (arch v2.1 §7b).

v1 had one task: extract a whole document in one call. That does not survive
contact with a 40-page policy or a dense Loss Run, so v2.1 decomposes the work
and trains **one adapter on all of it** — the task is declared in the prompt, so
the model is never guessing which output shape is wanted.

Two consequences run through the rest of the codebase:

* **Vision and sequence budgets are per task, not global.** A classification
  thumbnail and a six-page policy extraction have nothing in common in either
  dimension, and one global cap has to be wrong for one of them. See
  ``configs/shared/vision.yaml`` and ``configs/shared/sequence.yaml``.
* **A document expands into several training examples, of several tasks.** They
  all share the document's split (arch §8.2), or the same page appears on both
  sides of it.
"""

from __future__ import annotations

from enum import StrEnum


class Task(StrEnum):
    """What a single model call is being asked to do."""

    #: Page 1 at thumbnail resolution -> doc_type, acord_form, acord_edition.
    #: Selects the prompt, the schema and (after graduation) the adapter, so a
    #: wrong answer wastes everything downstream of it (arch §4a).
    CLASSIFY = "classify"

    #: The whole document in ONE call (an ACORD form, a short Loss Run). Pages
    #: are never selected: every page is read (decision 2026-10-19).
    EXTRACT = "extract"

    #: Loss Run pages 1-2 -> carrier, insured, policy periods, valuation date.
    LOSSRUN_HEADER = "lossrun_header"

    #: An adaptive 1-3 page window -> claim rows, plus any total/subtotal rows on
    #: those pages tagged with their row_type. Window size falls as row density
    #: rises, because a dense page overflows the output budget, not the input one.
    LOSSRUN_ROWS = "lossrun_rows"

    #: The last 2 pages -> grand totals and per-period totals. Separate from
    #: LOSSRUN_ROWS because printed totals sit at the end of the report, not on
    #: the first pages where v2.0 looked for them (v2.1 correction).
    LOSSRUN_TOTALS = "lossrun_totals"

    #: The declarations pages -> the policy-level fields: named insured, policy
    #: number, period, carrier, producer, limits, premium. One dense block that
    #: usually carries every policy-level value at once, which is why it is worth
    #: its own pass rather than being read out of a 60-page whole.
    POLICY_DECLARATIONS = "policy_declarations"

    #: A window of schedule pages -> the rows on them: vehicles, locations,
    #: coverages. Output-bound like LOSSRUN_ROWS: a 20-page vehicle schedule
    #: cannot be returned in one call, and a routed 6-page read would silently
    #: drop fourteen pages of it.
    POLICY_SCHEDULE = "policy_schedule"

    #: The endorsement pages -> what each endorsement changed, its number and its
    #: effective date. Endorsements are scattered through a policy rather than
    #: gathered, so they are found by page tagging rather than by position.
    POLICY_ENDORSEMENTS = "policy_endorsements"


#: Tasks that read page images at full extraction resolution. The two thumbnail
#: tasks are excluded: they recognise layout, and paying extraction resolution
#: for that is the single most wasteful thing this pipeline could do.
FULL_RESOLUTION_TASKS: frozenset[Task] = frozenset({
    Task.EXTRACT, Task.LOSSRUN_HEADER, Task.LOSSRUN_ROWS, Task.LOSSRUN_TOTALS,
    Task.POLICY_DECLARATIONS, Task.POLICY_SCHEDULE, Task.POLICY_ENDORSEMENTS,
})

#: The Loss Run decomposition, in the order the serving flow runs it.
LOSSRUN_TASKS: tuple[Task, ...] = (
    Task.LOSSRUN_HEADER, Task.LOSSRUN_ROWS, Task.LOSSRUN_TOTALS,
)

#: The Policy decomposition, in the order the serving flow runs it. Declarations
#: first because everything else is attributed to the policy it identifies.
POLICY_TASKS: tuple[Task, ...] = (
    Task.POLICY_DECLARATIONS, Task.POLICY_SCHEDULE, Task.POLICY_ENDORSEMENTS,
)


def parse(value: str | Task) -> Task:
    """Coerce a string to a ``Task``, naming the valid set when it is not one."""
    if isinstance(value, Task):
        return value
    try:
        return Task(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown task {value!r}; expected one of {[t.value for t in Task]}"
        ) from exc


#: The tasks the corpus build actually emits training rows for: whole-document
#: ``extract`` for the flat types, and the three section-window tasks for
#: canonical policies. A scope may DECLARE more (classify), but a
#: task with no rows is not trained, and a metric scoring an untrained task is
#: not a measurement of this model. ``tests/test_policy_windows`` pins this to
#: what the build produces, so the day classify rows are built this changes too.
CORPUS_TASKS: frozenset[Task] = frozenset({
    Task.EXTRACT, Task.POLICY_DECLARATIONS, Task.POLICY_SCHEDULE, Task.POLICY_ENDORSEMENTS,
})
