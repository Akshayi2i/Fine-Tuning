"""Which canonical sections one extraction call asks for (``configs/schema_sections.yaml``).

A canonical policy schema does not fit in a prompt alongside the pages it
describes. This module cuts it into named slices — ``decl``, ``arrays``,
``lineblk``, ``dtd`` — so a window's prompt describes only what that window can
answer, and so a window has room for seven or eight pages instead of four.

**A slice is a pure function of (lob, group).** No gold, no page list, no
document. That is the property serving depends on: it has no label to consult,
so anything derived from one cannot be reproduced at inference time, and a
training row whose shape cannot be reproduced is a row that teaches a shape the
model will never be asked for again.

The group a section belongs to is declared. The *pages* a group reads are
computed — every page of the document and, for the declarations group's first
window, where the declarations were actually found.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

#: The slice that reads the pages the declarations are printed on.
DECLARATIONS_GROUP = "decl"

#: Written in the YAML where a group takes every section no other group claims.
REMAINDER = "remainder"


class SectionMapError(RuntimeError):
    """Raised when the section map and a canonical schema disagree."""


@lru_cache(maxsize=1)
def _map() -> dict[str, Any]:
    from common.config import CONFIG_DIR, load_yaml

    return load_yaml(CONFIG_DIR / "schema_sections.yaml")


@lru_cache(maxsize=1)
def _common_model_profile() -> dict[str, Any]:
    """The map a common-model line is read with (the YAML's ``common_model``).

    Its groups carry only their sections; title, task and page rule are the
    ones the same group declares for every line, so a window's task, its pages
    and its place in the merge order mean the same thing on both kinds of line.
    """
    block = _map().get("common_model") or {}
    groups: dict[str, Any] = {}
    for name, sections in (block.get("groups") or {}).items():
        if name not in _map()["groups"]:
            raise SectionMapError(
                f"the common_model profile names group {name!r}, which the map does not declare"
            )
        groups[name] = {**_map()["groups"][name], "sections": sections}
    order = [name for name in _map()["groups"] if name in groups]
    return {**block, "groups": {name: groups[name] for name in order}}


def _profile(lob: str | list[str] | None) -> dict[str, Any]:
    """The section map ``lob`` is read with: the common-model profile for a
    line whose schema composes the common model, the map above otherwise. No
    line is the fallback, which composes the common model since common model
    1.1.0."""
    from common.schemas import is_common_model

    return _common_model_profile() if is_common_model("policy", None, lob) else _map()


def group_names(lob: str | list[str] | None = None) -> tuple[str, ...]:
    """Every declared group, in declaration order.

    Order is the order windows are built in, so it is also the order a merge
    sees them. ``decl`` leads: the declarations value wins a conflict, and a
    merge that reads it first can apply that rule by construction. A
    common-model line's groups are a subset, in the same order.
    """
    return tuple(_profile(lob)["groups"])


def task_for(group: str) -> str:
    return str(_rule(group)["task"])


def _rule(group: str) -> dict[str, Any]:
    """A group's title, task and page rule. They are declared once, for every
    line - a common-model profile carries them unchanged - so they are read
    from the map whatever line the group is asked on (``dtd`` has no place in
    the common-model profile, and the fallback is read with it)."""
    try:
        return _map()["groups"][group]
    except KeyError:
        raise SectionMapError(
            f"no section group {group!r}; declared groups: {list(_map()['groups'])}"
        ) from None


def _declared(group: str, lob: str | list[str] | None = None) -> dict[str, Any]:
    try:
        return _profile(lob)["groups"][group]
    except KeyError:
        raise SectionMapError(
            f"no section group {group!r}; declared groups: {list(group_names(lob))}"
        ) from None


def _excluded(section: str, profile: dict[str, Any]) -> bool:
    """Sections no group asks for: annotations and the full-text tier."""
    return section.startswith("fideon:") or section in (profile.get("exclude") or ())


def sections_for(group: str, lob: str | list[str] | None = None) -> tuple[str, ...]:
    """The top-level sections this group asks for, in schema order.

    Schema order rather than the order written in the YAML, so the sliced schema
    reads the way the client's file does and two slices of one schema cannot
    disagree about how it is laid out.

    A group declared ``remainder`` takes every section no other group claims.
    That is what lets a line this repo has never seen — a new section named
    ``covered_classes`` — still have it asked for, rather than silently dropped
    because nobody added it to a list.
    """
    from common.schemas import load_schema

    profile = _profile(lob)
    properties = [
        name for name in load_schema("policy", None, lob).get("properties") or {}
        if not _excluded(name, profile)
    ]
    declared = _declared(group, lob)["sections"]

    if declared == REMAINDER:
        claimed = {
            name
            for other in group_names(lob)
            if _declared(other, lob)["sections"] != REMAINDER
            for name in _declared(other, lob)["sections"]
        }
        return tuple(name for name in properties if name not in claimed)

    wanted = set(declared)
    return tuple(name for name in properties if name in wanted)


def groups_for(lob: str | list[str] | None = None) -> tuple[str, ...]:
    """The groups that have anything to ask for on this line.

    ``dtd`` is carried by three lines out of thirty-three, and ``lineblk`` is
    empty on the fallback schema. A group with no sections would render a prompt
    with an empty schema and ask the model for ``{}`` — a wasted call whose
    output is indistinguishable from a page holding nothing.
    """
    return tuple(name for name in group_names(lob) if sections_for(name, lob))


#: The page rules a group may declare (configs/schema_sections.yaml).
PAGE_RULES = ("every", "declarations")


def reads_declarations(group: str) -> bool:
    """Whether ``group``'s first window is the declarations pages, read together."""
    return _page_rule(group) == "declarations"


def _page_rule(group: str) -> str:
    rule = str(_rule(group)["pages"])
    if rule not in PAGE_RULES:
        raise SectionMapError(f"group {group!r} declares an unknown page rule {rule!r}")
    return rule


def pages_for(group: str, pages: list[int]) -> list[int]:
    """The pages this group reads: every one of them (decision 2026-10-19).

    Every page, chunk by chunk, never selected pages - for every group. A page
    no group read was a page whose values were silently missing.
    """
    _page_rule(group)
    return sorted(set(pages))


def leading_pages(group: str, pages: list[int], *, declarations_page: int | None = None) -> list[int]:
    """The pages a declarations group reads together in its first window.

    The leading pages plus, when it falls outside them, the page the
    declarations were actually found on. That last clause is the difference
    between working and not on scanned intake: a policy that opens with a fax
    cover sheet or a billing notice has its declarations on page 4 or 5, and
    reading them apart from the leading pages would split one block in two.
    Empty for every other group.
    """
    if _page_rule(group) != "declarations":
        return []
    leading = int(_map()["pages"]["declarations_leading"])
    found = {p for p in pages if p <= leading}
    if declarations_page and declarations_page in pages:
        found.add(declarations_page)
    return sorted(found) or sorted(pages)[:1]


def array_key(section: str, lob: str | list[str] | None = None) -> tuple[str, ...]:
    """The fields that identify one row of ``section``, for de-duplication.

    An array is asked over several page windows, so a table spanning a window
    boundary comes back twice — legitimately. Comparison is on normalised
    values — case, spacing, punctuation — so ``" TOLEDO "`` and ``"Toledo"``
    match. A prefix is not stripped: ``"LOC 1"`` and ``"1"`` stay distinct,
    because stripping would also merge ``"LOC 1"`` with ``"BLDG 1"``.
    """
    return tuple(_profile(lob).get("array_keys", {}).get(section, ()))


def structural_ids(lob: str | list[str] | None = None) -> dict[str, dict[str, str]]:
    """Table -> ``{field, prefix}`` of the structural id its rows carry, for a
    common-model line; empty for a self-contained one."""
    return dict(_profile(lob).get("ids") or {})


def references(lob: str | list[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Reference field -> the tables whose ids it can hold (common-model lines)."""
    return {k: tuple(v) for k, v in (_profile(lob).get("references") or {}).items()}


def unit_keys(lob: str | list[str] | None = None) -> dict[str, tuple[Any, ...]]:
    """Unit table -> the keys that identify a row, first stated one wins."""
    return {k: tuple(v) for k, v in (_profile(lob).get("unit_keys") or {}).items()}


def run_overlap(lob: str | list[str] | None = None) -> int:
    """How many pages two windows of one split page run share (``run_overlap_pages``):
    a row printed across their boundary is then whole in one of them. 0 for a
    line whose map sets none."""
    return int(_profile(lob).get("run_overlap_pages") or 0)


def assert_sections_cover_the_schema(lob: str | list[str] | None = None) -> None:
    """Every top-level section is asked for by exactly one group.

    The check that keeps this file honest when the client ships a new schema
    version. A section in no group is a section no window asks for — the field
    returns null and the extraction still validates, which is the failure this
    whole design exists to avoid. A section in two groups is two windows both
    claiming to own it, and a merge with a conflict that should not exist.

    A group naming a section the schema does not have is also refused: it means
    the client renamed something and this map still describes the old shape.
    """
    from common.schemas import _common_model, is_common_model, load_schema

    profile = _profile(lob)
    properties = {
        name for name in load_schema("policy", None, lob).get("properties") or {}
        if not _excluded(name, profile)
    }
    # A common-model line uses some of the shared blocks; a group may name the
    # others. A name that is no block at all is still refused, as a typo would be.
    blocks: set[str] = set()
    if is_common_model("policy", None, lob):
        common = _common_model()
        blocks = set(common.get("fideon:envelope_blocks") or []) | set(
            common.get("fideon:building_blocks") or [])
    seen: dict[str, str] = {}
    for group in group_names(lob):
        declared = _declared(group, lob)["sections"]
        if declared != REMAINDER:
            missing = set(declared) - properties - blocks
            if missing and group != "dtd":
                raise SectionMapError(
                    f"group {group!r} names {sorted(missing)}, which {lob or 'the fallback'} does "
                    "not have. The client's schema was renamed and this map still describes the "
                    "old shape."
                )
        for section in sections_for(group, lob):
            if section in seen:
                raise SectionMapError(
                    f"{section!r} is claimed by both {seen[section]!r} and {group!r}, so two "
                    "windows would own it and the merge would have a conflict that should not "
                    "exist."
                )
            seen[section] = group

    unclaimed = properties - set(seen)
    if unclaimed:
        raise SectionMapError(
            f"no group asks for {sorted(unclaimed)} on {lob or 'the fallback'}, so no window would "
            "request those fields. They would come back null and the extraction would still "
            "validate."
        )
