"""The two name grammars: ``source_id`` and ``run_id``.

Both are join keys — a source_id ties a prediction back to its page, a run_id
ties an artifact back to the run that made it — so both fail in the same way:
silently, by resolving to something that matches nothing, long after the write.

These cover the two rules that make the policy adapter's runs possible at all:

* reading a name is not minting one (a paused document type stays readable), and
* one parser, so the id a trainer writes is the id the gate can reconstruct.
"""

from __future__ import annotations

import pytest

from common.constants import ACTIVE_DOC_TYPES, KNOWN_DOC_TYPES
from common.ids import SourceIdError, build_source_id, parse_source_id
from common.run_ids import (
    UNIFIED_LINEAGE,
    RunIdError,
    build_run_id,
    is_valid_run_id,
    lineage_of,
    parse_run_id,
    version_of,
)

# --------------------------------------------------------------------------
# source_id — read vocabulary vs write vocabulary
# --------------------------------------------------------------------------


def test_every_active_type_is_a_known_type():
    """ACTIVE narrows KNOWN; it never adds to it. A type that could be minted but
    not parsed would produce ids nothing downstream could read."""
    assert set(ACTIVE_DOC_TYPES) <= set(KNOWN_DOC_TYPES)


def test_a_known_but_inactive_type_parses_and_cannot_be_minted():
    """`quote` is deferred — the same position a paused type lands in. Its ids
    must stay READABLE, or pausing a type would strand every path, corpus row and
    golden label naming it, rather than just stopping new ones."""
    assert "quote" in KNOWN_DOC_TYPES and "quote" not in ACTIVE_DOC_TYPES

    parsed = parse_source_id("quote_0007")
    assert parsed.doc_type == "quote" and parsed.index == 7

    with pytest.raises(SourceIdError, match="unknown doc_type"):
        build_source_id("quote", 7)


def test_a_genuinely_unknown_type_is_still_refused():
    with pytest.raises(SourceIdError, match="unknown doc_type"):
        parse_source_id("invoice_0001")


def test_unclassified_round_trips():
    """The holding bucket for a document whose type is not yet known — it must
    survive both directions, or ingestion cannot park anything."""
    assert parse_source_id(build_source_id("unclassified", 3)).doc_type == "unclassified"


# --------------------------------------------------------------------------
# run_id — one grammar
# --------------------------------------------------------------------------


def test_the_unified_lineage_still_mints_the_id_it_always_did():
    """The compatibility pin. `extractor-v1` is in Blob; if this changes, every
    artifact already written becomes unreachable by the name that addresses it."""
    assert build_run_id(UNIFIED_LINEAGE, "v1") == "extractor-v1"


@pytest.mark.parametrize(
    ("run_id", "lineage", "version"),
    [
        ("extractor-v2", "extractor", "v2"),
        ("extractor-v2.1", "extractor", "v2.1"),
        ("foundation-v10", "foundation", "v10"),
        ("policy-v2", "policy", "v2"),
        ("lossrun-adapter-v3", "lossrun-adapter", "v3"),
    ],
)
def test_a_run_id_splits_at_the_version_not_at_the_last_dash(run_id, lineage, version):
    """`lossrun-adapter-v3` is the case rsplit("-", 1) gets right only by
    accident: the lineage itself contains a dash."""
    assert parse_run_id(run_id) == (lineage, version, run_id)
    assert lineage_of(run_id) == lineage
    assert version_of(run_id) == version


@pytest.mark.parametrize("bad", ["extractor", "v2", "extractor-2", "extractor-vX", "-v2", ""])
def test_a_malformed_run_id_is_refused_rather_than_guessed(bad):
    assert not is_valid_run_id(bad)
    with pytest.raises(RunIdError):
        parse_run_id(bad)


def test_minting_validates_on_the_way_out():
    """An id that does not parse is one the registry cannot look up again, and
    the failure would otherwise surface at the gate — after the weights."""
    with pytest.raises(RunIdError):
        build_run_id("extractor", "2")


def test_the_run_ids_the_adapter_and_older_runs_mint_are_valid():
    """The policy adapter mints `policy-v{n}` (common.adapter); older runs minted
    their own lineages. All satisfy one grammar, including the checkpoint-path
    guard that refuses a run id."""
    from common.adapter import ADAPTER

    for lineage in ("extractor", ADAPTER.name, "lossrun", "policy_only"):
        run_id = build_run_id(lineage, "v2")
        assert parse_run_id(run_id).lineage == lineage
    assert ADAPTER.run_id("v3") == build_run_id("policy", "v3") == "policy-v3"
    assert version_of(ADAPTER.run_id("v3")) == "v3"


def test_the_checkpoint_guard_refuses_a_scoped_run_id_too():
    """`--continue-from` takes a DIRECTORY. The regex this replaced listed only
    the lineages v1 minted, so a scoped id would have sailed through and trained
    from base while the manifest recorded a lineage that never happened."""
    from training.train import TrainingError, assert_checkpoint_path

    for run_id in ("extractor-v2", "policy-v2", "lossrun-adapter-v3"):
        with pytest.raises(TrainingError, match="checkpoint path"):
            assert_checkpoint_path(run_id)

    assert_checkpoint_path("/runpod-volume/staging/adapters/foundation/v3")
