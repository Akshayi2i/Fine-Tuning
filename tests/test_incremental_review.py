"""One checkpoint from selection to release (code review of 2026-10-23).

Calibration and the gate score the base with one adapter checkpoint applied, and
package publishes one. Each run of ``package`` selects again, so:

* an eval report or a set of calibrators records the checkpoint it was made
  from, and is never reused for another one selected since;
* ``package --from-blob`` scores and releases the adapter in Blob, and only
  once the push has recorded it is that checkpoint;
* every push, by ``finetune --push-adapters`` or by package, records which
  checkpoint it pushed.
"""

from __future__ import annotations

import pytest

from artifact_registry import paths
from orchestration.pipeline_dag import (
    PipelineError,
    _assert_in_blob,
    _push_weights,
    adapter_checkpoint,
    eval_report_metrics,
    release_calibration,
    serving_adapter,
)
from tests.test_incremental_training import (  # noqa: F401 - fixtures
    _report,
    client,
    controller,
    released,
    write_adapter,
)
from tests.test_orchestration import make_context

SELECTED = "/runpod-volume/staging/adapters/policy/v1/v0-1/checkpoint-150"
EARLIER = "/runpod-volume/staging/adapters/policy/v1/v0-1/checkpoint-100"


def _select(store, checkpoint: str, **more) -> None:
    store.write_json(paths.checkpoint_selection("v1", scope="policy"),
                      {"selected": checkpoint, **more})


def test_the_checkpoint_is_the_selected_one_else_the_staged_adapter(released, controller):  # noqa: F811
    ctx = make_context(released, controller, out_version="v1")
    assert adapter_checkpoint(ctx) == paths.scoped_staging_adapter_dir("policy", "v1")
    _select(released, SELECTED)
    assert adapter_checkpoint(ctx) == SELECTED == serving_adapter(ctx)
    blob = make_context(released, controller, out_version="v1", from_blob=True)
    assert serving_adapter(blob) == paths.scoped_adapter_dir("policy", "v1")
    assert adapter_checkpoint(blob) == SELECTED


def test_an_eval_report_of_another_checkpoint_is_not_reused(released, controller):  # noqa: F811
    _select(released, SELECTED)
    key = paths.eval_report("v1", scope="policy")
    released.write_json(key, {**_report({"auto": [0.9] * 6}), "adapter_checkpoint": EARLIER})
    ctx = make_context(released, controller, out_version="v1")
    with pytest.raises(PipelineError, match=f"it scored checkpoint {EARLIER}, and the checkpoint "
                                            f"selected now is {SELECTED}"):
        eval_report_metrics(ctx)

    released.write_json(key, {**_report({"auto": [0.9] * 6}), "adapter_checkpoint": SELECTED})
    assert eval_report_metrics(ctx)["field_normalized_match"] == 0.93


def test_calibrators_fitted_on_another_checkpoint_are_refused(released, controller):  # noqa: F811
    _select(released, SELECTED)
    released.write_json(paths.release_calibrators("v1", "bf16"),
                        {"calibrators": {}, "thresholds": {}, "adapter_checkpoint": EARLIER})
    with pytest.raises(PipelineError, match="fitted on checkpoint .*checkpoint-100.*Run package "
                                            "again from calibrate"):
        release_calibration(make_context(released, controller, out_version="v1"), "bf16")


def _blob_ctx(store, ctl):
    return make_context(store, ctl, out_version="v1", from_blob=True, dry_run=False)


def test_from_blob_needs_the_adapter_in_blob(client, controller):  # noqa: F811
    with pytest.raises(PipelineError, match="holds no adapter_config.json, adapter_model.safetensors"):
        _assert_in_blob(_blob_ctx(client, controller))


def test_from_blob_needs_the_push_to_be_the_selected_checkpoint(released, controller):  # noqa: F811
    _select(released, SELECTED)
    with pytest.raises(PipelineError, match="records no push"):
        _assert_in_blob(_blob_ctx(released, controller))
    _select(released, SELECTED, pushed=EARLIER)
    with pytest.raises(PipelineError, match=f"records {EARLIER}"):
        _assert_in_blob(_blob_ctx(released, controller))

    _select(released, SELECTED, pushed=SELECTED)
    _assert_in_blob(_blob_ctx(released, controller))
    # Nothing to upload: the adapter in Blob is the release.
    assert _push_weights(_blob_ctx(released, controller)) == {
        "adapter:policy": paths.scoped_adapter_dir("policy", "v1")}


def test_a_dry_run_from_blob_checks_nothing(client, controller):  # noqa: F811
    _assert_in_blob(make_context(client, controller, out_version="v1", from_blob=True))


def test_package_records_the_checkpoint_it_pushed(released, controller, tmp_path):  # noqa: F811
    """So a release left gated, its staging cleared, can be packaged again
    with --from-blob."""
    checkpoint = tmp_path / "checkpoint-150"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text('{"r": 64}', encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    _select(released, str(checkpoint))
    ctx = make_context(released, controller, out_version="v1", dry_run=False)

    assert _push_weights(ctx) == {"adapter:policy": paths.scoped_adapter_dir("policy", "v1")}
    selection = released.read_json(paths.checkpoint_selection("v1", scope="policy"))
    assert selection == {"selected": str(checkpoint), "pushed": str(checkpoint)}
    _assert_in_blob(_blob_ctx(released, controller))
