"""Scope-addressed artifact paths (arch v2.1 §4.1, §18).

The policy adapter (common.adapter) is filed under its name, ``scope/policy``;
older runs were filed as ``unified``. Two properties, and the first one is the
whole reason the design is shaped this way:

1. **The unified layout renders byte-identically to what it always rendered.**
   ``adapters/foundation/v1``, ``merged-models/unified/v1`` and
   ``eval-reports/v1/gate_decision.json`` exist in Blob today. If any of those
   strings moves, the artifacts are still there and nothing can address them —
   a silent break, discovered when a deploy pulls an empty prefix.

2. **Two names at one version never collide.** Same version tag, different
   name, different key, for every artifact a run produces.

The expected strings below are written out as literals rather than derived, so
the test fails when a path changes rather than changing with it.
"""

from __future__ import annotations

import pytest

from artifact_registry import paths

#: Every helper a scoped run touches, with what the UNIFIED scope must render.
#: This is the compatibility pin: these are the keys already in Blob.
UNIFIED_RENDERS: list[tuple[str, str]] = [
    (paths.scoped_adapter_dir(None, "v1"), "adapters/foundation/v1"),
    (paths.scoped_adapter_dir("unified", "v1"), "adapters/foundation/v1"),
    (paths.adapter_dir("foundation", "v1"), "adapters/foundation/v1"),
    (paths.merged_model_dir("v1"), "merged-models/unified/v1"),
    (paths.merged_model_dir("v1", scope="unified"), "merged-models/unified/v1"),
    (paths.quantized_model_dir("v1", "fp8"), "quantized-models/unified/v1/vllm/fp8"),
    (paths.quantized_model_dir("v1", "bf16", scope="unified"),
     "quantized-models/unified/v1/vllm/bf16"),
    (paths.eval_report("v1"), "eval-reports/v1/summary.json"),
    (paths.eval_report("v1", "policy"), "eval-reports/v1/policy/report.json"),
    (paths.gate_decision("v1"), "eval-reports/v1/gate_decision.json"),
    (paths.gate_decision("v1", scope="unified"), "eval-reports/v1/gate_decision.json"),
    (paths.checkpoint_selection("v1"), "eval-reports/v1/checkpoint_selection.json"),
    (paths.run_manifest("extractor-v1", "unified"),
     "registry/foundation/extractor-v1/run_manifest.json"),
]


@pytest.mark.parametrize(("rendered", "expected"), UNIFIED_RENDERS)
def test_the_unified_scope_renders_exactly_what_it_always_rendered(rendered, expected):
    assert rendered == expected


def test_staging_mirrors_blob_for_a_scoped_run():
    """`package` copies rather than translates, so a scoped staging path has to
    end with its own blob key — otherwise the push reads one place and writes
    another."""
    assert paths.staging_merged_model_dir("v2", scope="policy").endswith(
        paths.merged_model_dir("v2", scope="policy")
    )
    assert paths.staging_quantized_model_dir("v2", "fp8", scope="policy").endswith(
        paths.quantized_model_dir("v2", "fp8", scope="policy")
    )
    assert paths.staging_merged_model_dir("v2", scope="policy").startswith("/")


# --------------------------------------------------------------------------
# Two names, one version
# --------------------------------------------------------------------------


SCOPED_HELPERS = [
    ("adapter", lambda s: paths.scoped_adapter_dir(s, "v2")),
    ("merged", lambda s: paths.merged_model_dir("v2", scope=s)),
    ("quantized", lambda s: paths.quantized_model_dir("v2", "fp8", scope=s)),
    ("staged merge", lambda s: paths.staging_merged_model_dir("v2", scope=s)),
    ("gate", lambda s: paths.gate_decision("v2", scope=s)),
    ("selection", lambda s: paths.checkpoint_selection("v2", scope=s)),
    ("eval report", lambda s: paths.eval_report("v2", scope=s)),
]


@pytest.mark.parametrize(("label", "render"), SCOPED_HELPERS)
def test_two_scopes_at_one_version_never_share_a_key(label, render):
    """An older unified run and the policy adapter can both be "v2". Sharing a
    key would mean the second one silently overwrites the first — its weights,
    or its gate verdict."""
    keys = {render("unified"), render("policy"), render("lossrun")}
    assert len(keys) == 3, f"{label} collides across scopes: {sorted(keys)}"


def test_a_scoped_adapter_never_lands_on_the_graduated_prefix():
    """`adapters/policy/` belongs to a §4.2 graduated per-type adapter, trained on
    the merged foundation. A policy-SCOPE adapter is trained on the base. Two
    different artifacts, so two different prefixes."""
    assert paths.scoped_adapter_dir("policy", "v2") != paths.adapter_dir("doc_type", "v2", "policy")
    assert paths.scoped_adapter_dir("policy", "v2") == "adapters/scope/policy/v2"


def test_scope_and_doc_type_are_different_axes():
    """A scoped run COVERS document types; a graduated adapter IS one. Naming both
    is refused rather than silently resolved to whichever the code checks first."""
    with pytest.raises(paths.PathError, match="cannot be both"):
        paths.merged_model_dir("v2", "policy", scope="policy")
    with pytest.raises(paths.PathError, match="cannot be both"):
        paths.quantized_model_dir("v2", "fp8", "policy", scope="policy")


def test_a_malformed_scope_is_refused_as_a_path_segment():
    with pytest.raises(paths.PathError, match="invalid scope"):
        paths.merged_model_dir("v2", scope="policy adapter")
    with pytest.raises(paths.PathError, match="invalid scope"):
        paths.scoped_adapter_dir("policy/../unified", "v2")


def test_scoped_artifacts_keep_their_tenancy_classification():
    """Tenancy is classified on the first path segment, so a new `scope/`
    sub-prefix inherits it — model weights stay shared, not tenant-scoped."""
    assert not paths.is_tenant_scoped(paths.scoped_adapter_dir("policy", "v2"))
    assert not paths.is_tenant_scoped(paths.merged_model_dir("v2", scope="policy"))
