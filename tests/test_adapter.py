"""The one adapter every fine-tuning cycle trains (common.adapter; decision 2026-10-22).

What it covers is fixed, not configured, so three properties have to hold:

* **one version names everything** - ``v3`` is the run ``policy-v3``, the
  adapter at ``adapters/scope/policy/v3`` (the layout the policy runs always
  used) and the release;
* **nothing excuses it from a metric it can produce** - not-applicable is
  derived from what it trains, never declared, or "not applicable" becomes a
  waiver with extra steps;
* **its corpus teaches one output shape** - common-model lines and
  self-contained schema files are not trained into one adapter.

And the lines of business it is filtered by (``--lines``, common.lob).
"""

from __future__ import annotations

import pytest

from artifact_registry import paths
from common.adapter import ADAPTER, AdapterError, assert_one_output_shape, not_applicable_metrics
from common.constants import ACTIVE_DOC_TYPES
from common.lob import LobError, known_lines, lob_lines, parse_lines, within_lines
from common.tasks import Task


@pytest.fixture
def uncached(monkeypatch):
    """``not_applicable_metrics`` is cached: recompute it for this test, and
    again for the next one once this test's patches are undone."""
    not_applicable_metrics.cache_clear()
    yield
    not_applicable_metrics.cache_clear()


# --------------------------------------------------------------------------
# One version names the run and its artifacts
# --------------------------------------------------------------------------


def test_one_version_names_the_run_and_its_artifacts():
    from common.run_ids import parse_run_id

    assert ADAPTER.name == "policy"
    assert ADAPTER.run_id("v3") == "policy-v3"
    parsed = parse_run_id(ADAPTER.run_id("v3"))
    assert (parsed.lineage, parsed.version) == ("policy", "v3")
    assert paths.scoped_adapter_dir(ADAPTER.name, "v3") == "adapters/scope/policy/v3"
    assert paths.merged_model_dir("v3", scope=ADAPTER.name) == "merged-models/scope/policy/v3"
    assert "/scope/policy/" in paths.run_manifest(ADAPTER.run_id("v3"), "scoped", scope=ADAPTER.name)


def test_the_adapter_never_lands_on_a_graduated_per_type_prefix():
    """`adapters/policy/` belongs to a §4.2 per-type adapter trained on the
    merged foundation; this one is trained on the base."""
    assert paths.scoped_adapter_dir(ADAPTER.name, "v2") != paths.adapter_dir("doc_type", "v2", "policy")


def test_it_trains_policies_with_the_unified_training_config():
    from common.config import CONFIG_DIR

    assert ADAPTER.doc_types == ("policy",)
    assert set(ADAPTER.doc_types) <= set(ACTIVE_DOC_TYPES)
    assert (CONFIG_DIR / "training" / f"{ADAPTER.training_config}.yaml").is_file()


def test_it_still_classifies_so_the_floor_keeps_meaning():
    """A policy release still has to recognise an ACORD and refuse it. If
    `classify` were dropped, classifier accuracy would become not-applicable and
    a real guarantee would quietly disappear."""
    assert Task.CLASSIFY in ADAPTER.tasks
    assert Task.LOSSRUN_ROWS not in ADAPTER.tasks and Task.LOSSRUN_HEADER not in ADAPTER.tasks


# --------------------------------------------------------------------------
# Not-applicable is derived, not declared
# --------------------------------------------------------------------------


def test_a_policy_adapter_has_no_loss_run_to_reconcile(uncached):
    """No Loss Run in the corpus means no printed totals to reconcile. That is a
    statement about what it trains, not a pass."""
    assert {"lossrun_totals_reconciliation_rate", "table_f1"} <= not_applicable_metrics()


def test_line_detection_is_not_applicable_while_every_line_comes_from_metadata(uncached):
    assert "lob_detection_accuracy" in not_applicable_metrics()


def test_the_classifier_metric_applies_exactly_when_classify_rows_are_trained(monkeypatch, uncached):
    """Declared is not trained. While the corpus builds no classify rows the
    classifier is never trained, so its accuracy is not a measurement of this
    model. The moment the corpus emits classify rows, the floor applies again."""
    from common import adapter as A
    from common.tasks import CORPUS_TASKS

    assert Task.CLASSIFY not in CORPUS_TASKS
    assert "doc_type_classifier_accuracy" in not_applicable_metrics()

    monkeypatch.setattr(A, "CORPUS_TASKS", CORPUS_TASKS | {Task.CLASSIFY})
    not_applicable_metrics.cache_clear()
    assert "doc_type_classifier_accuracy" not in not_applicable_metrics()


def test_list_metrics_apply_while_a_policy_schema_holds_a_table(monkeypatch, uncached):
    from common import adapter as A

    assert A._has_list_field()
    assert not {"list_field_recall", "field_f1_list_fields"} & not_applicable_metrics()

    monkeypatch.setattr(A, "_has_list_field", lambda: False)
    not_applicable_metrics.cache_clear()
    assert {"list_field_recall", "field_f1_list_fields"} <= not_applicable_metrics()


def test_every_not_applicable_metric_is_a_gate_metric(uncached):
    """A misspelt name would excuse nothing while saying it did."""
    from evaluation.gating import GATING_METRICS

    assert not_applicable_metrics() <= set(GATING_METRICS)


def test_nothing_the_adapter_can_produce_is_excused(uncached):
    from evaluation.gating import PRIMARY_METRICS

    for name in (*PRIMARY_METRICS, "field_exact_match", "schema_validity_rate", "ece_confidence",
                 "image_only_accuracy", "ocr_arbitration_accuracy"):
        assert name not in not_applicable_metrics(), name


def test_the_gate_reads_the_derived_set(uncached):
    from evaluation.gating import CONDITIONAL_METRICS, not_applicable_for

    assert not_applicable_for() == CONDITIONAL_METRICS | not_applicable_metrics()


# --------------------------------------------------------------------------
# One output shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lobs", [
    ["homeowners", "personal_auto", "gl"],          # lines on the common model
    # No line, and a package: both read against _fallback.json, the common model.
    ["homeowners", None, ["wc", "auto"]],
    ["wc", "auto", "flood"],                        # self-contained schema files
    [],
])
def test_training_documents_on_one_output_shape_are_accepted(lobs):
    assert_one_output_shape(lobs)


def test_documents_read_against_the_common_model_and_self_contained_schemas_are_refused():
    """One entry per trained document, judged by the schema it is read against."""
    with pytest.raises(AdapterError, match="two output shapes") as refused:
        assert_one_output_shape(["homeowners", "homeowners", "wc", None, ["homeowners", "gl"]])
    message = str(refused.value)
    assert "common model (gl+homeowners, homeowners, no line)" in message
    assert "self-contained schema files (wc)" in message


def test_a_package_of_self_contained_lines_is_read_in_the_common_models_shape():
    """A package is read against _fallback.json, so it does not share a shape
    with a policy of one of its lines on its own schema file."""
    with pytest.raises(AdapterError, match="two output shapes"):
        assert_one_output_shape(["wc", ["wc", "auto"]])


# --------------------------------------------------------------------------
# Lines of business (common.lob)
# --------------------------------------------------------------------------

#: The delivered SPEC_21 map: the 48 L1 codes, each with its schema.
LOB_SCHEMA_MAP_PARTS = ("canonical schema", "common schema", "lob_schema_map.yaml")

#: L1 codes the map names a schema for that was never delivered: this
#: repository cannot read them yet. A new one here is a gap to close.
NO_SCHEMA_DELIVERED = {"health", "life", "earthquake", "specialty"}


def test_enum_and_schema_spellings_are_one_line():
    assert lob_lines("workers_comp") == lob_lines("wc") == frozenset({"wc"})
    assert lob_lines("classic_auto") == frozenset({"personal_auto"})
    assert lob_lines(["homeowners", " Homeowners "]) == frozenset({"homeowners"})
    assert lob_lines(None) == lob_lines([]) == frozenset()


def test_the_known_lines_are_the_canonical_policy_schemas_on_disk():
    from common.schemas import _CANONICAL_NOT_REGISTERED, CANONICAL_DIR

    # classic_auto.json is on disk but read as personal auto, so it is no line.
    on_disk = {p.stem for p in CANONICAL_DIR.glob("*.json")
               if not p.stem.startswith("_") and p.stem not in _CANONICAL_NOT_REGISTERED}
    assert known_lines() == on_disk
    assert "classic_auto" not in known_lines()


def test_every_l1_code_the_map_names_a_schema_for_is_a_line_here_or_a_known_gap():
    import yaml

    from common.config import CONFIG_DIR
    from common.lob import merge_line
    from common.schemas import LOB_SCHEMA_ALIASES

    rows = yaml.safe_load(CONFIG_DIR.joinpath(*LOB_SCHEMA_MAP_PARTS).read_text(encoding="utf-8"))["lobs"]
    assert len(rows) == 48
    named = {row["code"]: row["schema_lob"] for row in rows if row.get("schema_lob") not in (None, "TBD")}
    ours = {code: LOB_SCHEMA_ALIASES.get(merge_line(code), merge_line(code)) for code in named}
    assert {code for code, line in ours.items() if line not in known_lines()} == NO_SCHEMA_DELIVERED
    # Where both name it, they name the same schema.
    assert {code: line for code, line in ours.items()
            if line in known_lines() and line != named[code]} == {}


def test_a_lines_argument_reads_as_canonical_lines():
    assert parse_lines("homeowners, personal_auto") == frozenset({"homeowners", "personal_auto"})
    assert parse_lines("workers_comp,classic_auto") == frozenset({"wc", "personal_auto"})
    assert parse_lines(None) is None and parse_lines("") is None and parse_lines("   ") is None


def test_a_line_with_no_schema_is_refused_rather_than_filtering_out_everything():
    with pytest.raises(LobError, match="no canonical policy schema"):
        parse_lines("homeowners,pet_insurance")


@pytest.mark.parametrize("lob,inside", [
    ("homeowners", True),
    (["personal_auto"], True),
    ("classic_auto", True),                         # read as personal auto
    ("gl", False),
    (["homeowners", "gl"], False),                  # one line outside
    # A package of common-model lines is read against _fallback.json: another shape.
    (["homeowners", "personal_auto"], False),
    (None, False),                                  # no line: cannot be shown to be one asked for
    ([], False),
])
def test_a_lines_filter_passes_a_document_only_when_all_its_lines_are_in_it(lob, inside):
    assert within_lines(lob, frozenset({"homeowners", "personal_auto"})) is inside


def test_a_package_of_self_contained_lines_passes_a_filter_naming_them():
    assert within_lines(["wc", "auto"], frozenset({"wc", "auto"}))


def test_no_filter_passes_everything():
    for lob in ("gl", None, ["homeowners", "personal_auto"]):
        assert within_lines(lob, None)
