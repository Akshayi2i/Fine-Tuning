"""The unified extractor LoRA (arch v2.1 §4.1, §9, §10).

A **thin wrapper**: it assembles an ms-swift configuration and launches it. The
training loop is ms-swift's own ``Seq2SeqTrainer``, built on the Hugging Face
Transformers ``Trainer`` — this file deliberately contains no optimizer step, no
backward pass and no collator (arch v2.1 §10).

*(TRL is installed as an ms-swift dependency for its RLHF trainers. It is not the
SFT loop used here — the v1 documentation said it was, and that was wrong.)*

**One run, not four.** v1 trained a Foundation LoRA and then a per-type LoRA for
each document type, stacked on top. That topology is unservable: vLLM applies one
LoRA per request, so the two could never both be active. It also did not fit the
data — at 25-30 documents per type a rank-16 adapter memorises its own training
set. So: one adapter, every document type, every task, conditioned by the prompt.

What this file owns is the policy the trainer cannot enforce for itself:

* **ViT and mergers frozen.** ``freeze_vit`` and ``freeze_aligner``, and the
  merger is not a LoRA target. Arbitration is learned in the decoder, where image
  tokens and OCR text tokens attend to each other; the mergers never see OCR text.
* **One max_length across a mixed-task corpus**, taken from the largest task cap,
  because ms-swift takes a single value and the corpus interleaves a 4k classify
  example with a 32k policy extraction.
* **A manifest written before launch**, so a pod that dies at step 40 leaves a
  record rather than nothing.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.adapter import ADAPTER
from common.config import (
    CONTINUED_DATA_KEYS,
    base_model_config,
    base_model_dir,
    base_model_source,
    continued_training_problem,
    sequence_for_task,
    training_config,
    validate_all,
)
from common.run_ids import is_valid_run_id
from registry_utils.models import (
    Artifacts,
    DataStats,
    Dependencies,
    RunManifest,
    TrainingConfig,
)
from registry_utils.write_run_manifest import capture_git_commit, write_manifest
from training.base_precision import manifest_descriptor, swift_quantization_args, technique
from training.corpus_view import materialize

log = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when a run cannot be configured or launched safely."""


@dataclass
class SwiftConfig:
    """The ms-swift invocation. Rendered, inspectable, and recorded.

    Kept as data rather than built inline so a run's exact configuration can be
    logged and diffed — which is what makes "why did this regress" answerable.
    """

    args: dict[str, Any]
    #: Environment for the launch. ms-swift's Qwen-VL processor reads its resize
    #: budget from here, not from a CLI flag.
    env: dict[str, str] = field(default_factory=dict)

    #: Rendered as ``--flag true`` / ``--flag false`` rather than as a bare
    #: presence flag. ms-swift parses these with HfArgumentParser, which accepts
    #: an explicit value — and dropping a False was silently disabling every
    #: option whose correct value is False: ``freeze_vit=False`` (so the flag
    #: never reached the trainer while the manifest recorded that it had),
    #: ``bnb_4bit_use_double_quant``, ``load_best_model_at_end``.
    BOOL_FLAGS = ("true", "false")

    def to_cli(self) -> list[str]:
        argv = ["swift", "sft"]
        for key, value in sorted(self.args.items()):
            if value is None:
                continue
            flag = f"--{key}"
            if isinstance(value, bool):
                argv.extend([flag, "true" if value else "false"])
            elif isinstance(value, (list, tuple)):
                # One argv element per item. Joining them into a single
                # space-separated string gave HfArgumentParser's nargs="+" a
                # one-element list containing every value as one token, so six
                # corpus paths became one nonexistent filename and the LoRA
                # target modules matched nothing at all.
                argv.append(flag)
                argv.extend(str(v) for v in value)
            else:
                argv.extend([flag, str(value)])
        return argv


#: ms-swift 3 names for the attention implementations base_model.yaml uses.
_ATTN_IMPL = {"flash_attention_2": "flash_attn"}

#: Evaluations a run should get at least, so early checkpoints are saved and
#: checkpoint selection has several to choose between.
MIN_EVALUATIONS = 5

#: ``batch`` keys that change a run's speed and never what it learns. Each is
#: passed to ms-swift only when set; null leaves ms-swift's own default, which
#: is what every run before them trained with. Left out of the run fingerprint.
THROUGHPUT_KEYS = ("vit_gradient_checkpointing", "dataloader_num_workers", "per_device_eval_batch_size")


def throughput_args(batch: dict[str, Any]) -> dict[str, Any]:
    """The set throughput levers (:data:`THROUGHPUT_KEYS`), as ms-swift arguments.

    Raises TrainingError on a value of the wrong kind, before anything is staged.
    """
    out: dict[str, Any] = {}
    for key in THROUGHPUT_KEYS:
        value = batch.get(key)
        if value is None:
            continue
        if key == "vit_gradient_checkpointing":
            ok = isinstance(value, bool)
        else:
            low = 0 if key == "dataloader_num_workers" else 1
            ok = isinstance(value, int) and not isinstance(value, bool) and value >= low
        if not ok:
            want = "true or false" if key == "vit_gradient_checkpointing" else (
                "a whole number >= " + ("0" if key == "dataloader_num_workers" else "1"))
            raise TrainingError(f"batch.{key} is {value!r}; it must be {want}, or null for ms-swift's default")
        out[key] = value
    return out


def _eval_interval(
    train_rows: int | None, *, configured: int, effective_batch: int, evaluations: int | None = None,
) -> int:
    """Steps between evaluations.

    With ``evaluations`` (the validation cut), the run is divided into that many
    checks, however long it is: a fixed interval made the number of checks grow
    with the corpus, ~50 on the delivered one. Without it, the configured value,
    shrunk so a run gets at least :data:`MIN_EVALUATIONS`. Unknown row counts keep
    the configured value.
    """
    if not train_rows:
        return configured
    steps = max(1, -(-train_rows // max(1, effective_batch)))
    if evaluations:
        return max(1, -(-steps // max(1, evaluations)))
    return max(1, min(configured, steps // MIN_EVALUATIONS or 1))


def training_gpus(distributed: dict[str, Any]) -> int:
    """GPUs to train on: ``distributed.gpus`` - "auto" (every visible GPU) or a number.

    "auto" off a GPU machine is 1, so configuration built anywhere else matches
    a one-GPU run.
    """
    configured = distributed.get("gpus", 1)
    if str(configured).lower() != "auto":
        return max(1, int(configured))
    try:
        import torch

        return max(1, torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - no torch / no CUDA: one GPU's worth of config
        return 1


def _accumulation_steps(batch: dict[str, Any], gpus: int) -> int:
    """Gradient accumulation that keeps the effective batch the configured size.

    Refused when it cannot: an effective batch the GPUs do not divide would
    silently train with a different one, and a different number of steps.
    """
    per_step = int(batch["per_device_train_batch_size"]) * gpus
    effective = int(batch.get("effective_batch_size")
                    or int(batch["per_device_train_batch_size"]) * int(batch["gradient_accumulation_steps"]))
    if effective % per_step:
        raise TrainingError(
            f"an effective batch of {effective} rows cannot be split over {gpus} GPU(s) at "
            f"{batch['per_device_train_batch_size']} row(s) each; set distributed.gpus or "
            "batch.effective_batch_size so the one divides the other"
        )
    return max(1, effective // per_step)


def loss_sample_rows(evaluation: dict[str, Any]) -> int:
    """Rows of the in-training loss checks: ``loss_sample_rows``, else the
    selection sample (``validation_sample_rows``); 0 reads the whole split.

    The loss only draws the progress curve - no early stopping is passed to
    ms-swift and the best-loss weights are never loaded - while checkpoint
    selection generates on its own, larger sample. So the curve can be read off
    fewer rows: 160 instead of 400 took each of the CGL run's ten checks from
    ~7.4 min to ~3 min.
    """
    selection = int(evaluation.get("validation_sample_rows") or 0)
    return int(evaluation.get("loss_sample_rows") or selection)


def _checkpoint_cadence(interval: int, evaluation: dict[str, Any], *, gpus: int = 1) -> dict[str, int]:
    """Save a resume point at least every ``checkpoint_every_steps``, evaluate on
    every n-th of them, and keep enough that the last evaluation checkpoints
    checkpoint selection compares are never rotated away.

    Without a separate cadence a checkpoint was saved only at an evaluation, and
    a crash cost everything since the last one - hours, once evaluations are few.
    """
    # checkpoint_every_steps is sized for one GPU (~1 h); n GPUs run n x as many
    # steps in that hour.
    every = max(1, int(evaluation.get("checkpoint_every_steps") or interval) * max(1, gpus))
    # The fewest saves per evaluation that keep them at least `every` apart, and
    # the save interval that divides the evaluation interval evenly: evaluations
    # stay where the cut put them, saves come a little MORE often than `every`.
    per_eval = max(1, -(-interval // every))
    save_steps = max(1, -(-interval // per_eval))
    return {
        "save_steps": save_steps,
        "eval_steps": save_steps * per_eval,
        # The configured limit counts evaluation checkpoints; resume points in
        # between would otherwise push the earlier candidates out.
        "save_total_limit": int(evaluation["save_total_limit"]) * per_eval + 1,
    }


def resumed_cadence(checkpoint: str | Path, evaluation: dict[str, Any]) -> dict[str, int] | None:
    """The save and evaluation cadence of the run ``checkpoint`` belongs to,
    for resuming it; None when its ``trainer_state.json`` records none.

    On resume the HF Trainer restores ``save_steps`` and ``eval_steps`` from the
    checkpoint's trainer state - it overrides the arguments, with a warning - so
    a run resumed on another GPU count keeps saving and evaluating on the
    cadence it started with. ``save_total_limit`` is the one it takes from the
    arguments, and :func:`_checkpoint_cadence` sizes it from the NEW GPU count's
    saves per evaluation: resumed on more GPUs it shrank (17 -> 5 for 1 -> 4
    GPUs) while saves came as often as before, and rotation deleted the earlier
    evaluation checkpoints that checkpoint selection compares. From the
    checkpoint, all three agree with what the trainer does.
    """
    import json

    try:
        state = json.loads((Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
        save_steps, eval_steps = int(state.get("save_steps") or 0), int(state.get("eval_steps") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if save_steps <= 0 or eval_steps <= 0:
        return None
    per_eval = max(1, -(-eval_steps // save_steps))
    return {
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "save_total_limit": int(evaluation["save_total_limit"]) * per_eval + 1,
    }


def planned_steps(train_rows: int | None, effective_batch: int) -> int | None:
    """Optimizer steps a run of ``train_rows`` rows takes; None when unknown."""
    if not train_rows:
        return None
    return max(1, -(-int(train_rows) // max(1, int(effective_batch))))


def keep_two_thirds_limit(total_steps: int, save_steps: int) -> int:
    """The smallest ``save_total_limit`` that keeps the checkpoint nearest
    two-thirds of a ``total_steps`` run that saves every ``save_steps``.

    The trainer rotates the OLDEST checkpoints away, so the two-thirds one
    survives when the limit covers every save from it to the end - ms-swift
    saves the last step as well - plus one slot for the best-loss checkpoint
    rotation also keeps, and one for a step count a save off this estimate.
    Checkpoint selection scores that checkpoint as a candidate (evaluation.
    checkpoint_eval), so rotation must never be what decides it is gone.
    """
    total, every = max(1, int(total_steps)), max(1, int(save_steps))
    saves = list(range(every, total + 1, every))
    if not saves or saves[-1] != total:
        saves.append(total)
    target = 2 * total / 3
    nearest = min(saves, key=lambda step: (abs(step - target), step))
    return sum(1 for step in saves if step >= nearest) + 2


def _keep_two_thirds(args: dict[str, Any], total_steps: int | None) -> None:
    """Raise ``args["save_total_limit"]`` to :func:`keep_two_thirds_limit`, never lower it."""
    if not total_steps or not args.get("save_steps") or args.get("save_total_limit") is None:
        return
    needed = keep_two_thirds_limit(total_steps, int(args["save_steps"]))
    if needed > int(args["save_total_limit"]):
        log.info("save_total_limit %s -> %d, so the checkpoint nearest two-thirds of the run's %d "
                 "steps is kept", args["save_total_limit"], needed, total_steps)
        args["save_total_limit"] = needed


def two_thirds_checkpoint(output_dir: str | Path) -> dict[str, Any] | None:
    """The finished run's checkpoint nearest two-thirds of its final step.

    ``{"final_step", "two_thirds_target", "two_thirds_step", "two_thirds_checkpoint"}``,
    or None when the run left no checkpoint. Read from the checkpoint
    directories of the most recent run under ``output_dir`` (ms-swift writes
    ``output_dir/v<n>-<time>/checkpoint-<step>``), as checkpoint selection reads them.
    """
    import re

    root = Path(output_dir)
    found = [(int(m.group(1)), path) for path in root.rglob("checkpoint-*")
             if path.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", path.name))] if root.is_dir() else []
    if not found:
        return None
    latest = max({path.parent for _, path in found}, key=lambda d: d.stat().st_mtime)
    steps = sorted((step, path) for step, path in found if path.parent == latest)
    final = steps[-1][0]
    target = 2 * final / 3
    step, path = min(steps, key=lambda item: (abs(item[0] - target), item[0]))
    return {"final_step": final, "two_thirds_target": round(target, 1),
            "two_thirds_step": step, "two_thirds_checkpoint": str(path)}


#: Written beside the adapter checkpoints when ms-swift returns successfully.
#: The pipeline's "trained" means this file, never the directory: ms-swift
#: creates the directory when training STARTS, so a crashed run looked finished
#: and a rerun went straight on to choose among its partial checkpoints.
TRAINING_COMPLETE = "training_complete.json"


def training_completed(output_dir: str | Path) -> bool:
    return (Path(output_dir) / TRAINING_COMPLETE).is_file()


def mark_training_complete(output_dir: str | Path, run_id: str,
                           summary: dict[str, Any] | None = None) -> None:
    """Write :data:`TRAINING_COMPLETE`, with ``summary`` (:func:`training_summary`) in it."""
    import json
    from datetime import UTC, datetime

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / TRAINING_COMPLETE).write_text(json.dumps({
        **(summary or {}), "run_id": run_id, "completed_at": datetime.now(UTC).isoformat(),
    }), encoding="utf-8")


def read_training_summary(output_dir: str | Path) -> dict[str, Any]:
    """What :data:`TRAINING_COMPLETE` records (``two_thirds_step`` ...); {} without one."""
    import json

    try:
        data = json.loads((Path(output_dir) / TRAINING_COMPLETE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def training_summary(output_dir: str | Path, *, lora_dropout: float | None = None) -> dict[str, Any]:
    """What a finished run left on disk, for :data:`TRAINING_COMPLETE`.

    The checkpoint nearest two-thirds of the final step (:func:`two_thirds_checkpoint`),
    with a warning when rotation took the one that was nearer; and the dropout
    the last checkpoint's ``adapter_config.json`` was saved with, against the
    one the manifest records.
    """
    import json

    summary: dict[str, Any] = dict(two_thirds_checkpoint(output_dir) or {})
    if not summary:
        log.warning("%s holds no checkpoint, so no two-thirds checkpoint could be recorded", output_dir)
        return summary
    final = Path(summary["two_thirds_checkpoint"]).parent / f"checkpoint-{summary['final_step']}"
    try:
        state = json.loads((final / "trainer_state.json").read_text(encoding="utf-8"))
        save_steps = int(state.get("save_steps") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        save_steps = 0
    if save_steps and abs(summary["two_thirds_step"] - summary["two_thirds_target"]) > save_steps:
        log.warning("the checkpoint nearest two-thirds of the run (step ~%s) is gone: the nearest left "
                    "is %s. save_total_limit rotated it away.", summary["two_thirds_target"],
                    summary["two_thirds_step"])
    try:
        saved = json.loads((final / "adapter_config.json").read_text(encoding="utf-8"))
        dropout = saved.get("lora_dropout") if isinstance(saved, dict) else None
    except (OSError, ValueError):
        dropout = None
    if dropout is not None:
        summary["adapter_lora_dropout"] = dropout
        if lora_dropout is not None and _number(dropout) != _number(lora_dropout):
            log.warning("the adapter was saved with lora_dropout %s, the manifest records %s",
                        dropout, lora_dropout)
    return summary


RUN_FINGERPRINT = "run_fingerprint.json"

#: What a resumed run must share with the run that wrote the checkpoint. Not
#: the per-device batch or the accumulation steps: those follow the GPU count,
#: and resuming on a different pod is the point of resuming.
_FINGERPRINT_SETTINGS = ("lora_rank", "lora_alpha", "learning_rate", "lr_scheduler", "epochs",
                         "effective_batch_size", "target_modules")


def run_fingerprint(corpus_version: str, corpus_manifest: Any, recorded: Any,
                    parent_run_id: str | None = None, continue_from: str | None = None) -> dict:
    """The corpus, the settings and the starting weights a run trains with -
    what a checkpoint is OF. A run continued from v3 and one from the base are
    not one run, so neither resumes the other's checkpoints.

    The starting weights are named by :func:`starting_weights`, not by the
    parent alone: a run continued by hand (``--continue-from``) has no parent
    run, and recorded only that, it looked like a run from the base - so
    ``--fresh`` resumed its checkpoint, the weights of an adapter it had meant
    to leave behind, and recorded no parent.
    """
    import hashlib
    import json

    corpus = json.dumps(corpus_manifest, sort_keys=True, default=str)
    return {
        "corpus_version": corpus_version,
        "corpus_manifest_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "adapter": ADAPTER.name,
        "parent_run_id": parent_run_id,
        "starting_weights": starting_weights(continue_from),
        **{name: getattr(recorded, name, None) for name in _FINGERPRINT_SETTINGS},
        "settings_sha256": _settings_hash(),
    }


def starting_weights(continue_from: str | None) -> str | None:
    """The weights a run starts from: None for the base; for an adapter, the
    sha256 of its ``adapter_model.safetensors``, else (not there yet) its
    normalised directory.

    The weights' hash rather than the directory: the pipeline stages a cycle's
    parent at one path whichever release it is, and a by-hand run can point
    anywhere, so the path alone cannot tell two adapters apart.
    """
    if not continue_from:
        return None
    directory = Path(os.path.normpath(continue_from.strip()))
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file():
        return directory.as_posix()
    import hashlib

    digest = hashlib.sha256()
    with weights.open("rb") as fh:
        # In chunks: a rank-64 adapter is a few hundred MB.
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _settings_hash() -> str | None:
    """Every setting that shapes what is trained on, as one hash.

    The training file (line balance, dropout, validation sample ...), the pixel
    budget and the sequence caps. Lowering line_balance.max_repeat and resuming
    left fewer rows than the checkpoint's step count: ms-swift ran zero steps,
    exited 0, and the old checkpoints were selected as a finished run. Left
    out: ``distributed`` and the per-device batch, which follow the GPU count -
    resuming on another pod is the point of resuming.
    """
    import hashlib
    import json

    name = ADAPTER.training_config
    try:
        import yaml

        from common.config import (
            BASE_MODEL_CONFIG,
            SHARED_FORMAT_CONFIG,
            SHARED_SEQUENCE_CONFIG,
            SHARED_VISION_CONFIG,
            training_config,
        )

        cfg = json.loads(json.dumps(training_config(name), default=str))
        cfg.pop("distributed", None)
        batch = dict(cfg.get("batch") or {})
        # The throughput levers too: they change how fast, never what is learned.
        for key in ("per_device_train_batch_size", "per_device_eval_batch_size",
                    "gradient_accumulation_steps", *THROUGHPUT_KEYS):
            batch.pop(key, None)
        cfg["batch"] = batch
        # The format profile (what a row looks like) and the base model (which
        # weights, which precision) shape the run as much as the training file.
        shared = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in
                  (SHARED_VISION_CONFIG, SHARED_SEQUENCE_CONFIG, SHARED_FORMAT_CONFIG)]
        base = yaml.safe_load(BASE_MODEL_CONFIG.read_text(encoding="utf-8")) or {}
        # Where a pod keeps the weights, not which weights: resuming elsewhere is the point.
        (base.get("model") or {}).pop("local_dir", None)
    except Exception as exc:  # noqa: BLE001 - a fingerprint must not stop a run from starting
        log.warning("the run fingerprint could not read the training settings: %s", exc)
        return None
    body = json.dumps([cfg, shared, base], sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def record_fingerprint(output_dir: str | Path, fingerprint: dict) -> None:
    import json

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / RUN_FINGERPRINT).write_text(
        json.dumps(fingerprint, indent=2, default=str), encoding="utf-8")


def assert_resumable(output_dir: str | Path, fingerprint: dict) -> None:
    """Refuse to resume a checkpoint another corpus or configuration wrote.

    Re-running a failed out-version after rebuilding the corpus would restore
    the OLD run's weights, optimizer and step and train a few more steps on
    data it never saw the start of, while the manifest recorded the new corpus.
    A run started before fingerprints were written has none: it resumes, said.
    One whose fingerprint lacks a key this one has is refused: it cannot be
    shown to be this run (written before ``starting_weights``, a run from the
    base and one continued by hand look the same).
    """
    import json

    path = Path(output_dir) / RUN_FINGERPRINT
    if not path.is_file():
        log.warning("%s has no %s (a run started before it was recorded); resuming without "
                    "checking that the corpus and settings are the same", output_dir, RUN_FINGERPRINT)
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(fingerprint, default=str))
    unrecorded = sorted(k for k in current if k not in stored)
    if unrecorded:
        start = (" - above all, whether they started from the base or from an adapter, and which"
                 if "starting_weights" in unrecorded else "")
        raise TrainingError(
            f"{output_dir} holds checkpoints whose {RUN_FINGERPRINT} was written before "
            f"{unrecorded} were recorded, so they cannot be shown to be of this run{start}. "
            "Resuming could continue weights this run does not start from. Train under a new "
            "out-version, or remove that directory to start this one again."
        )
    changed = sorted(k for k in {*stored, *current} if stored.get(k) != current.get(k))
    if changed:
        start = ("; a run from the base and one continued from an adapter, or from another "
                 "adapter, are not one run" if {"parent_run_id", "starting_weights"} & {*changed} else "")
        raise TrainingError(
            f"{output_dir} holds checkpoints of a different run: {changed} changed since they "
            f"were written{start}. Resuming would continue the old weights and optimizer on a "
            "corpus, configuration or start they were not trained with. Train under a new "
            "out-version, or remove that directory to start this one again."
        )


def resume_point(output_dir: str | Path) -> Path | None:
    """The latest complete checkpoint of an unfinished run, or None.

    Complete = it holds the trainer state, the adapter and the optimizer state:
    a run killed while saving leaves a checkpoint without them, and resuming
    from it would fail or silently restart the optimizer.
    """
    import re

    root = Path(output_dir)
    if not root.is_dir() or training_completed(root):
        return None
    candidates = []
    for checkpoint in root.rglob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
        if not (match and checkpoint.is_dir()):
            continue
        names = {f.name for f in checkpoint.iterdir()}
        has_adapter = any(n.startswith("adapter_model") for n in names)
        has_optimizer = bool(names & {"optimizer.pt", "optimizer.bin"}) or any(
            n.startswith("global_step") for n in names)
        if "trainer_state.json" in names and has_adapter and has_optimizer:
            candidates.append((int(match.group(1)), checkpoint))
    return max(candidates)[1] if candidates else None


def corpus_max_length() -> int:
    """The single ``max_length`` for a mixed-task corpus.

    ms-swift takes one value and the corpus interleaves every task, so this is
    the **largest** per-task cap (arch v2.1 §7a). Taking anything smaller would
    truncate the longest task rather than reject it, and a clipped assistant span
    trains the model to stop early.

    It is the cap that decides whether a run fits on one 80GB card, which is why
    ``use_logits_to_keep`` and padding-free batching are not optional here.

    **Narrowed to what the run actually carries:** the adapter's own tasks, and
    only the per-doc-type overrides for the types it trains. The maximum over
    every task and type could give a run a cap none of its rows needs, and that
    cap is what decides the card.
    """
    from common.config import shared_sequence_config

    declared = shared_sequence_config().get("tasks", {})
    tasks = tuple(ADAPTER.tasks)
    doc_types = set(ADAPTER.doc_types)

    caps: list[int] = []
    for task in tasks:
        name = str(task)
        caps.append(int(sequence_for_task(name)["max_seq_len"]))
        # Per-doc-type overrides are separate caps, not variations on one.
        # Iterating only the bare tasks gave 24576 while the policy extraction
        # override is 32768, so every routed policy would have been TRUNCATED —
        # which is the exact failure the §7a caps exist to prevent.
        for doc_type in (declared.get(name, {}).get("by_doc_type") or {}):
            if doc_type not in doc_types:
                continue
            caps.append(int(sequence_for_task(name, doc_type)["max_seq_len"]))
    return max(caps)


def build_training_config(
    *,
    corpus_paths: list[str],
    output_dir: str,
    val_paths: list[str] | None = None,
    train_vit: bool = False,
    deepspeed: str | None = None,
    resume_from: str | None = None,
    config_name: str | None = None,
    train_rows: int | None = None,
    lora_dropout: float | None = None,
    lora_dropout_choice: dict[str, Any] | None = None,
) -> tuple[SwiftConfig, TrainingConfig]:
    """Assemble the ms-swift arguments and the manifest's record of them.

    Both come from the same source so the manifest describes what actually ran,
    rather than what a YAML file happened to say afterwards.

    ``corpus_paths`` are the materialized epoch files, in order. The run reads the
    first ``num_train_epochs`` of them **once each** — see the epoch comment
    below for why ms-swift is told one epoch.

    ``lora_dropout`` is the value :func:`resolve_lora_dropout` chose (and
    ``lora_dropout_choice`` why); without it, the config's number, or for
    ``auto`` with no corpus counted, the pilot-scale value.
    """
    base = base_model_config()
    cfg = training_config(config_name or ADAPTER.training_config)
    lora, batch = cfg["lora"], cfg["batch"]
    if lora_dropout is None:
        lora_dropout, default_choice = configured_dropout(lora)
        lora_dropout_choice = lora_dropout_choice or default_choice
    # A run continuing an adapter trains with the config's continued_training
    # values where it sets them (fewer passes, a lower rate), else as a fresh run.
    opt = optimization_for(cfg, continuing=resume_from is not None)
    evaluation, memory = cfg["evaluation"], cfg.get("memory", {})

    epochs = int(opt["num_train_epochs"])
    if len(corpus_paths) < epochs:
        raise TrainingError(
            f"the config asks for {epochs} epochs but only {len(corpus_paths)} epoch file(s) were "
            "given. Each epoch is its own file with its own modality draw (arch v2.1 §6.1); "
            "reusing a file would show every document in the same regime twice."
        )

    target_modules = list(lora["target_modules"])
    if lora.get("include_vision_projector", False):
        # Off by default under v2.1 §9a. Reachable only through the vision
        # ablation, and a vision-module LoRA is merged into the base rather than
        # hot-swapped — tower/connector LoRA in vLLM is experimental.
        target_modules.append("merger")

    max_length = corpus_max_length()
    gpus = training_gpus(cfg.get("distributed", {}))
    accumulation = _accumulation_steps(batch, gpus)
    interval = _eval_interval(
        train_rows,
        configured=int(evaluation["eval_steps"]),
        evaluations=int(evaluation.get("evaluations_per_run") or 0) or None,
        effective_batch=int(batch["per_device_train_batch_size"]) * accumulation * gpus,
    )

    # ms-swift 3 argument names, one version, throughout (pinned in pyproject).
    # The command used to mix 2.x names (model_id_or_path, lora_target_modules,
    # quantization_bit) with 3.x-only ones (strict, padding_free), which no single
    # ms-swift accepts — its parser refuses an unknown flag, so the run died at
    # argument parsing. The Phase 0 spike parses this exact command with
    # ms-swift's own parser (check_ms_swift) so a wrong name fails there.
    args: dict[str, Any] = {
        # No model_type: ms-swift 3 infers it from the checkpoint, and the 2.x
        # name "qwen3-vl-8b-instruct" is not one it registers.
        # The pod's local copy (configs/base_model.yaml `local_dir`). A revision
        # means nothing to a directory, so it is passed only for a Hub id.
        "model": base_model_source(),
        "model_revision": None if base_model_dir() else base["model"]["revision"],
        # From the Hugging Face hub, not ModelScope (ms-swift's default): the
        # pinned revision is a Hugging Face commit, and the tokenizer the
        # pre-launch length check loads, serving and the manifest all assume it.
        "use_hf": True,
        # The first N epoch files, read once each. Every file holds every train
        # document, in that epoch's modality draw, so the concatenation has the
        # CONTENT of an N-epoch run: each document N times, in N regimes. Not its
        # ORDER — the Trainer shuffles the concatenated rows, so a document's
        # passes can land next to each other, and an
        # early stop can come after some documents have been seen more often
        # than others. Over a run that averages out; within one it does not.
        "dataset": list(corpus_paths[:epochs]),
        "output_dir": output_dir,
        "train_type": "lora",
        "lora_rank": lora["rank"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "use_rslora": bool(lora.get("use_rslora", False)),
        # bf16 frozen base by default; 4-bit NF4 only when the config asks for it
        # (arch §9). One helper shared with any future trainer so they cannot
        # disagree about how the base is held.
        **swift_quantization_args(base),
        "attn_impl": _ATTN_IMPL.get(
            base["attention"]["attn_implementation"], base["attention"]["attn_implementation"]
        ),
        "learning_rate": opt["learning_rate"],
        "lr_scheduler_type": opt["lr_scheduler_type"],
        # Steps, not a ratio: at pilot volume a 0.03 ratio over a handful of
        # steps rounds to zero warmup, and the first optimizer step then lands at
        # full learning rate on a freshly-initialised adapter.
        "warmup_steps": opt["warmup_steps"],
        # One pass over the concatenated epoch files, never N. Telling ms-swift
        # N here as well loops the N files N times — nine passes for a "3 epoch"
        # run, the inflation per-epoch sampling exists to remove. The logical
        # epoch count is recorded on the manifest below.
        "num_train_epochs": 1,
        "optim": opt["optim"],
        "weight_decay": opt["weight_decay"],
        "max_grad_norm": opt["max_grad_norm"],
        "per_device_train_batch_size": batch["per_device_train_batch_size"],
        # Across every GPU, rows per optimizer step = per-device batch x this x
        # GPUs = the configured effective batch (training_gpus, _accumulation_steps).
        "gradient_accumulation_steps": accumulation,
        "gradient_checkpointing": batch["gradient_checkpointing"],
        # vit_gradient_checkpointing, dataloader_num_workers,
        # per_device_eval_batch_size: only those the config sets.
        **throughput_args(batch),
        "dataset_num_proc": max(1, min(int(batch.get("dataset_num_proc", 1)), os.cpu_count() or 1)),
        "bf16": batch["bf16"],
        "torch_dtype": "bfloat16" if batch["bf16"] else "float32",
        "max_length": max_length,
        # Never "left" or "right": "left" removes the system prompt, "right" the
        # end of the target, and a clipped target trains the model to stop early.
        # "delete" makes an over-length row fail to encode instead.
        "truncation_strategy": "delete",
        # ...and strict makes that failure a HARD ERROR. ms-swift tokenizes
        # multimodal rows lazily, and by default a row that fails to encode is
        # replaced with a random other row — one document lost, another trained
        # twice, nothing recorded. A stopped run beats a silently wrong one.
        # Neither should ever fire: every row is measured against max_length on
        # the pod before launch (training.length_check), so an over-length row
        # refuses the run before training starts, not hours into it.
        "strict": True,
        # The three settings that make the largest task cap affordable (§9.3).
        # Without use_logits_to_keep the LM head produces a 151k-vocabulary
        # distribution at every position of a 32k sequence, which dominates
        # activation memory on its own.
        "use_logits_to_keep": bool(memory.get("use_logits_to_keep", True)),
        "padding_free": bool(memory.get("padding_free", True)),
        # The Hugging Face TrainingArguments name. "length_grouped_sampling" is
        # not an argument of either ms-swift or HF, and was refused at parsing.
        # Off by default: with one row per device and padding_free it saves no
        # padding, and it made ms-swift encode every row before step 1.
        "group_by_length": bool(memory.get("length_grouped_sampling", False)),
        # Frozen, both of them (arch v2.1 §9.4). `train_vit` is the §3 escalation
        # and remains LoRA-on-ViT, never a full fine-tune.
        "freeze_vit": not train_vit,
        "freeze_aligner": True,
        # The validation file is the ONLY eval set. Without this, ms-swift
        # carves its own eval split out of the training data as well — the
        # leakage keeping --dataset and --val_dataset apart exists to prevent.
        "split_dataset_ratio": 0.0,
        "eval_strategy": evaluation["eval_strategy"],
        "save_strategy": evaluation.get("save_strategy", "steps"),
        # Sized to the run, not fixed at 50: at pilot volume a whole run is ~56
        # optimizer steps, so a fixed 50 gave one evaluation and one checkpoint,
        # and checkpoint selection had nothing to choose between.
        **_checkpoint_cadence(interval, evaluation, gpus=gpus),
        # Standard HF TrainingArguments. There is deliberately no early-stopping
        # patience flag: "early_stopping_patience" is not an argument ms-swift 3
        # accepts, and what ships is chosen by generated field F1 over every
        # saved checkpoint (checkpoint_eval), not by where loss stopped falling.
        "load_best_model_at_end": bool(evaluation["load_best_model_at_end"]),
        "metric_for_best_model": evaluation["metric_for_best_model"],
        "greater_is_better": bool(evaluation.get("greater_is_better", False)),
        "logging_steps": cfg["logging"]["logging_steps"],
        # Never a tracker nobody configured: "[]" in the YAML used to fall back
        # to the HF default, which enables whatever tracker is installed.
        "report_to": "none",
        "seed": cfg["seed"],
    }
    # Rotation must leave the checkpoint nearest two-thirds of the run for
    # checkpoint selection, whatever save_total_limit says.
    _keep_two_thirds(args, planned_steps(
        train_rows, int(batch["per_device_train_batch_size"]) * accumulation * gpus))

    distributed = cfg.get("distributed", {})
    if deepspeed:
        args["deepspeed"] = f"configs/deepspeed/{deepspeed}.json"
    if int(distributed.get("sequence_parallel_size", 1)) > 1:
        # Reached before any ZeRO config when a task cap does not fit (§9.3).
        args["sequence_parallel_size"] = distributed["sequence_parallel_size"]
    if val_paths:
        args["val_dataset"] = val_paths
    if resume_from:
        # A NEW run that starts from the adapter's weights. resume_from_checkpoint
        # restores the old run's optimizer, schedule and global_step: continuing
        # v2 on a new corpus would resume mid-cosine, or run zero steps when the
        # old step count already exceeds the new run's, and exit "trained" with
        # the old weights.
        args["adapters"] = [resume_from]

    recorded = TrainingConfig(
        # Derived from the config that actually ran, never defaulted. These three
        # carried QLoRA/NF4/paged-8bit defaults on the model, so a bf16 run that
        # did not pass them recorded a technique it never used — in the one
        # record the whole reproducibility story rests on.
        technique=technique(base),
        base_quantization=manifest_descriptor(base),
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora_dropout,
        lora_dropout_choice=lora_dropout_choice,
        bias=lora["bias"],
        learning_rate=opt["learning_rate"],
        lr_scheduler=opt["lr_scheduler_type"],
        warmup_ratio=0.0,
        epochs=opt["num_train_epochs"],
        optimizer=opt["optim"],
        adam_beta1=opt["adam_beta1"],
        adam_beta2=opt["adam_beta2"],
        adam_epsilon=opt["adam_epsilon"],
        weight_decay=opt["weight_decay"],
        max_grad_norm=opt["max_grad_norm"],
        per_device_batch_size=batch["per_device_train_batch_size"],
        gradient_accumulation_steps=accumulation,
        effective_batch_size=batch["effective_batch_size"],
        gradient_checkpointing=batch["gradient_checkpointing"],
        mixed_precision="bf16" if batch["bf16"] else "fp32",
        target_modules=target_modules,
        vit_trainable=train_vit,
        vit_method="lora" if train_vit else "frozen",
        resolution_cap_px=base["vision"]["max_image_long_side_px"],
        max_seq_len=max_length,
        seed=cfg["seed"],
    )
    env = _pixel_budget()
    recorded.pixel_budget = {key: int(value) for key, value in env.items()}
    if gpus > 1:
        # ms-swift starts one training process per GPU (torchrun) when this is set.
        env["NPROC_PER_NODE"] = str(gpus)
        # Every trainable parameter (the LoRA layers) takes part in every step, so
        # DDP need not search the graph for unused ones - and with reentrant
        # gradient checkpointing that search fails.
        args["ddp_find_unused_parameters"] = False
    return SwiftConfig(args, env=env), recorded


def _pixel_budget() -> dict[str, str]:
    """The image resize budget ms-swift's Qwen-VL processor reads from the env.

    The same per-task ``max_pixels`` the corpus was sized against (cap_check) and
    serving uses. Left to the processor's default, training would resize pages to
    a budget nobody chose — today the rendered pages happen to sit inside it, so
    nothing differs, but the day a render or a vision budget changes, training
    and serving would see different pixels with nothing to say so.
    """
    from common.config import ConfigError, pixel_budget
    from data_pipeline.dataset_builder.cap_check import PIXELS_PER_VISUAL_TOKEN

    # One budget for the whole run, because the trainer takes one — and the
    # same function serving's engine takes its budget from. Every row the corpus
    # builds today is full resolution, so the tasks agree; the day a corpus
    # carries rows of a task with a different budget (the thumbnail tasks),
    # training.stage_data refuses such rows rather than train them wrongly.
    try:
        min_pixels, max_pixels = pixel_budget(ADAPTER.tasks)
    except ConfigError as exc:
        raise TrainingError(f"the adapter's tasks: {exc}") from exc
    per_token = PIXELS_PER_VISUAL_TOKEN
    # Both forms. Qwen2/2.5-VL read MAX_PIXELS/MIN_PIXELS; ms-swift's Qwen3-VL
    # template reads a token budget. Whichever this ms-swift reads, it reads the
    # SAME budget — and the spike (swift_image_budget) measures which one took.
    return {
        "MAX_PIXELS": str(max_pixels),
        "MIN_PIXELS": str(min_pixels),
        "IMAGE_MAX_TOKEN_NUM": str(max_pixels // per_token),
        "IMAGE_MIN_TOKEN_NUM": str(min_pixels // per_token),
    }


def optimization_for(cfg: dict[str, Any], *, continuing: bool) -> dict[str, Any]:
    """The ``optimization`` block a run trains with.

    A run continuing an adapter (decision 2026-10-23) takes the training
    config's ``continued_training`` values where they are set - its documents
    already trained in the cycles before, so fewer passes or a lower rate may do
    - and the ``optimization`` values everywhere else. Null keeps the fresh
    run's value: set them from a measured continued run, not by guess.

    Raises ValueError on a block that would not do what it says: a key the
    ``optimization`` block does not have (``epochs:``, silently unused), or an
    epoch count or learning rate that is not positive.
    """
    problem = continued_training_problem(cfg)
    if problem:
        raise ValueError(problem)
    opt = dict(cfg["optimization"])
    if continuing:
        for key, value in (cfg.get("continued_training") or {}).items():
            # Rehearsal's keys choose the data (training.corpus_view), not the optimizer.
            if value is not None and key not in CONTINUED_DATA_KEYS:
                opt[key] = value
    return opt


def build_manifest(
    *,
    run_id: str,
    corpus_version: str,
    corpus_manifest: dict[str, Any],
    training_cfg: TrainingConfig,
    data_stats: DataStats,
    staging_path: str,
    continued_from: str | None = None,
    parent_run_id: str | None = None,
    fresh_reason: str | None = None,
    tenant_id: str | None = None,
) -> RunManifest:
    """Build the run manifest. Written to Blob even while weights are staged."""
    base = base_model_config()["model"]
    dirty, diff_sha256 = worktree_state()
    if dirty:
        log.warning(
            "the working tree has uncommitted changes, so code_git_commit does not fully "
            "describe what ran — this run is not exactly reproducible (diff sha256 %s).", diff_sha256
        )

    return RunManifest(
        run_id=run_id,
        # Filed where the policy runs always were (registry/scope/policy/), and
        # recording what the run covers.
        run_type="scoped",
        scope=ADAPTER.name,
        doc_types=list(ADAPTER.doc_types),
        # Whose corpus trained it. The field existed and nothing set it, so a
        # manifest could not say which tenant's documents its weights came from.
        tenant_id=paths._tenant(tenant_id),
        continued_from=continued_from,
        parent_run_id=parent_run_id,
        fresh_reason=fresh_reason,
        dependencies=Dependencies(
            base_model=f"{base['model_id']}@{base['revision']}",
            corpus_version=corpus_version,
            code_git_commit=capture_git_commit(),
            code_dirty=dirty,
            code_diff_sha256=diff_sha256,
            mineru_version=corpus_manifest.get("mineru_version"),
            ocr_device=corpus_manifest.get("ocr_device"),
            schema_version=str(corpus_manifest.get("schema_versions", {})),
            prompt_template_version=corpus_manifest.get("prompt_template_version"),
        ),
        training_config=training_cfg,
        data_stats=data_stats,
        artifacts=Artifacts(status="staged", staging_path=staging_path),
        status="training",
    )


#: The repository this code runs from: the tree whose state a run records.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Set to 1 to train on a working tree with uncommitted changes: what
#: ``--allow-dirty`` does for a run started through the pipeline.
ALLOW_DIRTY_ENV = "FIDEON_ALLOW_DIRTY_TREE"


def worktree_state() -> tuple[bool | None, str | None]:
    """``(dirty, sha256 of the changes)`` of the code's git working tree.

    The changes are ``git diff HEAD`` (staged or not) and the names of the
    untracked files git does not ignore - by name only, not their contents.
    ``(False, None)`` on a clean tree, ``(None, None)`` where git cannot be
    read (a tarball on a pod): unknown, which the manifest records as such.
    """
    import hashlib
    import subprocess

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, timeout=30,
                              check=True).stdout

    try:
        status = git("status", "--porcelain")
        if not status.strip():
            return False, None
        digest = hashlib.sha256(git("diff", "HEAD", "--binary"))
        digest.update(b"\0untracked\0" + git("ls-files", "--others", "--exclude-standard"))
    except Exception:  # noqa: BLE001 - no git, not a repository, a timeout: unknown
        return None, None
    return True, digest.hexdigest()


def assert_clean_tree(allow_dirty: bool = False) -> None:
    """Refuse a real run on a working tree with uncommitted changes.

    ``code_git_commit`` names a commit; with changes on top of it, the run's
    code is not that commit and cannot be checked out again. ``allow_dirty``
    (``--allow-dirty``, or :data:`ALLOW_DIRTY_ENV`) trains anyway, said, and
    the manifest records ``code_dirty`` and the diff's hash.
    """
    dirty, digest = worktree_state()
    if not dirty:
        return
    if allow_dirty or os.environ.get(ALLOW_DIRTY_ENV) == "1":
        log.warning("training on a working tree with uncommitted changes (diff sha256 %s), as asked; "
                    "the manifest records it", digest)
        return
    raise TrainingError(
        f"the working tree at {REPO_ROOT} has uncommitted changes, so the run's code would not be "
        "the commit its manifest names, and the run could not be reproduced. Commit them, or pass "
        f"--allow-dirty ({ALLOW_DIRTY_ENV}=1 for a pipeline run) to train anyway: the manifest "
        "then records code_dirty and the hash of the changes."
    )


#: Set to train on a GPU host that is not a RunPod pod (a workstation with its
#: own disk). The check below exists for the case nobody meant: a laptop run.
OFF_POD_ENV = "FIDEON_ALLOW_OFF_POD"


def assert_on_pod() -> None:
    """Refuse a real launch anywhere but a GPU pod with the staging volume.

    ``finetune`` defaults to a real run, and off the pod ms-swift would start a
    CPU "training" run that takes days, or download 16 GB of base weights to a
    laptop, while the staging paths it writes to — ``/runpod-volume/...`` — do
    not exist, so nothing it produced could be found by the stages after it.
    """
    import os

    if os.environ.get(OFF_POD_ENV) == "1":
        return
    # The mount the staging paths resolve under, however it is configured.
    mount = os.environ.get("RUNPOD_VOLUME_MOUNT") or "/runpod-volume"

    from common.gpu import GPUError, require_cuda

    problems = []
    try:
        require_cuda("training")
    except GPUError as exc:
        problems.append(str(exc))
    if not Path(mount).is_dir():
        problems.append(f"the staging volume is not mounted at {mount}")
    # Configured but absent means ms-swift would fall back to the Hub and pull
    # 16 GB onto the pod — at whatever revision is current, not the pinned one.
    local_dir = os.environ.get("FIDEON_BASE_MODEL_DIR") or base_model_config()["model"].get(
        "local_dir"
    )
    if local_dir and base_model_dir() is None:
        problems.append(f"no base model (a config.json) was found under {local_dir}")
    if problems:
        raise TrainingError(
            "refusing to launch training here: " + "; ".join(problems) + ". Run it on the "
            f"RunPod pod, use --dry-run to check the plan, or set {OFF_POD_ENV}=1 on a GPU "
            "host whose staging paths exist."
        )


def train(
    *,
    corpus_version: str,
    out_version: str,
    client: BlobClient,
    corpus_manifest: dict[str, Any],
    data_stats: DataStats,
    train_vit: bool = False,
    deepspeed: str | None = None,
    continue_from: str | None = None,
    dry_run: bool = False,
    tenant_id: str | None = None,
    parent_run_id: str | None = None,
    fresh_reason: str | None = None,
    allow_dirty: bool = False,
) -> tuple[SwiftConfig, RunManifest]:
    """Configure and launch one training run.

    Args:
        continue_from: **a filesystem adapter directory**, not a registry
            run-id: the weights a NEW run starts from (ms-swift ``adapters``,
            with a fresh optimizer and schedule) - the parent's adapter when a
            cycle continues (decision 2026-10-23). A run-id like
            ``extractor-v3`` once produced a run that silently trained from base
            while its manifest recorded a lineage that never happened;
            ``assert_checkpoint_path`` refuses that shape. An adapter saved with
            other LoRA settings than the config's is refused too
            (``assert_adapter_matches_config``).
        parent_run_id: the run whose adapter ``continue_from`` is, recorded on
            the manifest and in the run fingerprint.
        fresh_reason: why a cycle started from the base, recorded on the manifest.
        dry_run: assemble and record everything without launching.
        tenant_id: whose corpus to read. Omitting it reads the default tenant's
            files, which for any other tenant do not exist — or worse, do.
        allow_dirty: train on a working tree with uncommitted changes, which a
            real run otherwise refuses (:func:`assert_clean_tree`).
    """
    validate_all(require_pinned_revision=not dry_run)
    if not dry_run:
        assert_on_pod()
        assert_clean_tree(allow_dirty)
    if train_vit:
        # Refused, not attempted. The LoRA targets are the decoder's projection
        # names; Qwen3-VL's vision blocks use different ones, and ms-swift only
        # applies freeze_vit=false when expanding "all-linear". So --train-vit
        # trained the same decoder-only adapter while the manifest recorded
        # vit_trainable=True. The vision module names are what the spike's
        # merger_module_names check measures; this unblocks once they are known.
        raise TrainingError(
            "--train-vit is not supported yet: the vision-tower LoRA target names are unconfirmed, "
            "so the run would train the decoder only while recording a vision LoRA. Confirm the "
            "module names with the Phase 0 spike (merger_module_names) first."
        )

    # Train and val are kept apart. Passing both to `--dataset` made ms-swift
    # treat the validation split as training data and then carve its own eval
    # split out of the union, so the selected checkpoint was chosen on documents
    # the model had memorised — and the promotion gate read that number.
    from training.data_mix import DataMixError, assert_corpus_mix

    try:
        assert_corpus_mix(corpus_manifest)
    except DataMixError as exc:
        raise TrainingError(str(exc)) from exc
    # The run reads a VIEW of the corpus (training.corpus_view): its policy rows,
    # with line balance and render rotation applied - and, for a continued run,
    # rehearsal; the parent (or promoted) run's validation errors pick the hard examples.
    view = materialize(corpus_version, client, tenant_id=tenant_id,
                       parent_run_id=parent_run_id if continue_from else None, out_version=out_version)
    from common.adapter import AdapterError, assert_one_output_shape
    from training.corpus_view import trained_lobs

    try:
        # On the documents the view really trains, before anything is staged.
        assert_one_output_shape(trained_lobs(view))
    except AdapterError as exc:
        raise TrainingError(str(exc)) from exc
    corpus_paths = list(view.epoch_files)
    val_paths = [view.val_path]
    staging = paths.scoped_staging_adapter_dir(ADAPTER.name, out_version)

    # The rows the run actually trains on: the first N epoch files, not all four
    # that are materialized. Counting all four overstated the data by a third,
    # and the same count sizes the evaluation interval.
    epochs = int(optimization_for(training_config(ADAPTER.training_config),
                                  continuing=continue_from is not None)["num_train_epochs"])
    train_rows = sum(view.rows_by_epoch.get(e, 0) for e in range(1, epochs + 1))
    # The caller counts the whole corpus. The run trains on its view, and its
    # manifest has to say so.
    update: dict[str, Any] = {
        "train_examples": train_rows, "val_examples": view.val_rows, "test_examples": view.test_rows,
        # The shares configured and reached, per line, for the manifest and the
        # model card.
        "data_mix": {"settings": view.mix_settings, "lines": view.data_mix,
                     "rested_documents": view.rested_documents,
                     "render_rotation": view.render_rotation,
                     # Empty when off: which documents were new and which old
                     # ones each epoch rehearsed; twins per epoch; hard examples.
                     "rehearsal": view.rehearsal, "twin_rotation": view.twin_rotation,
                     "hard_examples": view.hard_examples, "line_balance": view.line_balance},
        "examples_by_line": view.examples_by_line,
    }
    data_stats = data_stats.model_copy(update=update)

    # A version already trained or promoted is never overwritten — not by a
    # re-run, and not by a dry run, which used to replace a promoted manifest
    # with a fresh "training" one and leave that row in the index for good.
    _refuse_existing_run(ADAPTER.run_id(out_version), client)

    # Every check that costs nothing runs before staging, which downloads every
    # page image the run reads. A run that is going to be refused should be
    # refused before that, not after thousands of images reach a paid pod.
    if continue_from:
        # Existence is checked on a real run only: the directory is on the pod's
        # volume, which a dry run on an operator machine cannot see.
        assert_checkpoint_path(continue_from, must_exist=not dry_run)
        if not dry_run:
            assert_adapter_matches_config(continue_from)
    # Counted only for a real run, where it decides anything; a dry run launches
    # nothing and would read the unified validation file for no reason.
    if not dry_run and not view.val_rows:
        # The config asks for evaluation, early stopping and
        # load_best_model_at_end, and checkpoint selection generates on
        # validation: with no rows the run fails after the GPU is paid for, or
        # trains with nothing to choose its checkpoint by.
        raise TrainingError(
            f"corpus {corpus_version} has no validation rows for the adapter. "
            "Checkpoint selection and early stopping both read validation, so the run "
            "cannot choose what to ship. Rebuild the corpus with enough documents of "
            f"{list(ADAPTER.doc_types)} to fill a validation split."
        )

    # lora.dropout: a number as it is; auto from this corpus at a run from the
    # base, the parent adapter's own at a continued one.
    dropout, dropout_choice = resolve_lora_dropout(
        training_config(ADAPTER.training_config)["lora"],
        train_lines=view.examples_by_line.get("train"), continue_from=continue_from,
        parent_run_id=parent_run_id, client=client, dry_run=dry_run)
    if dropout_choice["source"] != "config":
        log.info("lora_dropout %s (%s: %s)", dropout, dropout_choice["source"], dropout_choice.get("reason"))

    # Built against the Blob keys first: this is where the epoch count and every
    # other configuration error surfaces, still before any download.
    swift, recorded = build_training_config(
        corpus_paths=corpus_paths,
        val_paths=val_paths,
        output_dir=staging,
        train_vit=train_vit,
        deepspeed=deepspeed,
        resume_from=continue_from,
        train_rows=train_rows,
        lora_dropout=dropout,
        lora_dropout_choice=dropout_choice,
    )

    # Before the manifest is rewritten and every page image is staged: a resume
    # that will be refused should be refused in seconds, having changed nothing.
    # A continued run that stopped part-way resumes its OWN checkpoint, which
    # already started from the parent's weights; the fingerprint, which names the
    # weights it starts from, refuses a checkpoint of another start. A dry run
    # neither resumes nor records one, so it hashes no adapter.
    resumed = None
    if not dry_run:
        fingerprint = run_fingerprint(corpus_version, corpus_manifest, recorded, parent_run_id,
                                      continue_from)
        resumed = resume_point(staging)
        if resumed is not None:
            assert_resumable(staging, fingerprint)

    if not dry_run:
        # ms-swift reads LOCAL files in ITS row format. The view is Blob keys in
        # the corpus's own shape — handed over as-is, the trainer finds nothing,
        # and a row that did load would fail on its mixed-type content column.
        # A dry run launches nothing, so it records the Blob keys instead.
        from training.stage_data import prune_image_caches, stage_training_data

        epochs = len(swift.args["dataset"])
        prune_image_caches(paths.staging_train_images_dir(corpus_version, tenant_id))
        from evaluation.validation_sample import validation_sample

        sample_rows = loss_sample_rows(training_config(ADAPTER.training_config)["evaluation"])
        staged = stage_training_data(
            view.epoch_files[:epochs], view.val_path, client,
            paths.staging_train_data_dir(ADAPTER.name, out_version),
            images_root=paths.staging_train_images_dir(corpus_version, tenant_id),
            max_pixels=int(swift.env["MAX_PIXELS"]),
            # The in-training checks read a fixed stratified validation sample,
            # chosen on the corpus rows before staging strips the line and mode off them.
            val_filter=(lambda rows: validation_sample(rows, sample_rows)) if sample_rows else None,
        )
        swift.args["dataset"] = staged.epoch_files
        swift.args["val_dataset"] = [staged.val_path]
        lengths = _assert_rows_fit(swift, [*staged.epoch_files, staged.val_path], staged.output_caps,
                                   groups=staged.groups, stats_files=staged.epoch_files)
        # The rows' lengths and the answer tokens per section group, on the manifest.
        data_stats = data_stats.model_copy(update={"token_stats": lengths.summary()})
        _assert_masking(staged.epoch_files, swift)

    # Only the epochs this run uses. Four files are always materialized because
    # the §11a sweep tests up to four passes (§6.1), and a sweep that regenerates
    # its own data is not comparing what it thinks it is.
    manifest = build_manifest(
        run_id=ADAPTER.run_id(out_version),
        corpus_version=corpus_version,
        corpus_manifest=corpus_manifest,
        training_cfg=recorded,
        data_stats=data_stats,
        staging_path=staging,
        continued_from=continue_from,
        parent_run_id=parent_run_id,
        fresh_reason=fresh_reason,
        tenant_id=tenant_id,
    )
    write_manifest(manifest, client)

    if dry_run:
        log.info("dry run — configuration recorded, nothing launched")
        return swift, manifest

    if resumed is None:
        record_fingerprint(staging, fingerprint)
    else:
        # The SAME run, carried on: ms-swift restores the weights, the optimizer,
        # the learning-rate schedule and the step, and skips the batches already
        # trained on, into the run directory it was writing. Not --continue-from,
        # which starts a NEW run from the weights with the schedule at zero.
        swift.args["resume_from_checkpoint"] = str(resumed)
        swift.args["output_dir"] = str(resumed.parent)
        swift.args["add_version"] = False
        # The checkpoint holds the weights; loading the parent's adapter as well
        # would start the resumed run from two places.
        swift.args.pop("adapters", None)
        # The cadence the run started with, which the trainer restores anyway:
        # sized for this pod's GPU count, save_total_limit rotated away the
        # earlier evaluation checkpoints selection compares.
        cadence = resumed_cadence(resumed, training_config(ADAPTER.training_config)["evaluation"])
        if cadence is not None:
            changed = {k: (swift.args.get(k), v) for k, v in cadence.items() if swift.args.get(k) != v}
            if changed:
                log.info("resuming on the run's own checkpoint cadence (this pod's would be "
                         "otherwise): %s", changed)
            swift.args.update(cadence)
        # ...and still never rotating away the two-thirds checkpoint.
        _keep_two_thirds(swift.args, planned_steps(train_rows, int(recorded.effective_batch_size)))
        log.info("resuming the unfinished run from %s", resumed)


    manifest = launch_and_record(swift, manifest, client)
    return swift, manifest


def _token_counter():
    """The model's tokenizer as a counter. A seam, so tests need no model."""
    from training.length_check import tokenizer_counter

    model = base_model_config()["model"]
    if base_model_dir() is not None:
        return tokenizer_counter(base_model_source())
    return tokenizer_counter(model["model_id"], model.get("revision"))


def _masking_encoder(swift: SwiftConfig):  # pragma: no cover - needs ms-swift and the tokenizer
    """ms-swift's template encode for this model, with the budget env applied.

    Returns ``(encode, assistant_header_ids, end_token_id, end_suffix_ids)``: the
    suffix is what ms-swift's template writes after the end token and supervises
    (the ChatML ``<|im_end|>\n`` newline). A seam, so tests need neither
    ms-swift nor a model.
    """
    import os

    from swift.llm import get_model_tokenizer, get_template

    # The pixel budget only: NPROC_PER_NODE is for the trainer's launcher, and
    # left in this process it would reach every later stage run here.
    os.environ.update({k: v for k, v in swift.env.items() if k != "NPROC_PER_NODE"})
    _model, processor = get_model_tokenizer(
        swift.args["model"], load_model=False, revision=swift.args.get("model_revision"),
    )
    template = get_template(processor.model_meta.template, processor)
    template.set_mode("train")
    # ms-swift's import raises the root log handlers; without this every later
    # stage run in this process - checkpoint selection included - logs nothing.
    from common.logging_setup import reassert

    reassert()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    suffix = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)[1:]
    return template.encode, header, end, suffix


def _assert_masking(files: list[str], swift: SwiftConfig) -> None:
    """Refuse the run if the trainer would supervise anything but the answer.

    ``training.data_collator`` held this check and nothing called it. A masking
    error trains the model to reproduce its prompt while the loss curve looks
    normal; encoded here with ms-swift's own template, it is caught before launch.
    """
    from training.data_collator import MaskingError, verify_staged_rows

    encode, header, end, suffix = _masking_encoder(swift)
    try:
        report = verify_staged_rows(files, encode=encode, assistant_header_ids=header,
                                    end_token_id=end, end_suffix_ids=suffix)
    except MaskingError as exc:
        raise TrainingError(f"label masking check failed before launch: {exc}") from exc
    log.info("masking check: %s", report.summary())


def _assert_rows_fit(
    swift: SwiftConfig, files: list[str], output_caps: dict[str, list[int]] | None = None,
    *, groups: dict[str, list[str]] | None = None, stats_files: list[str] | None = None,
):
    """Refuse the run before launch if any staged row is over ``max_length``.

    Under strict encoding such a row stops the run when the trainer reaches it,
    possibly hours in. Measured here instead, with the real tokenizer and the
    real resize rule, it stops the run before any GPU time is spent on it.

    Returns the :class:`~training.length_check.LengthReport`: the rows of
    ``stats_files`` (the epoch files) as a length histogram and supervised
    tokens per section group (``groups``, as staging records them).
    """
    from training.length_check import measure

    report = measure(
        files,
        max_length=int(swift.args["max_length"]),
        count_tokens=_token_counter(),
        min_pixels=int(swift.env["MIN_PIXELS"]),
        max_pixels=int(swift.env["MAX_PIXELS"]),
        output_caps=output_caps,
        groups=groups,
        stats_files=stats_files,
    )
    log.info("length check: %d row(s), longest %d tokens", report.rows, report.longest)
    summary = report.summary()
    log.info("supervised tokens per group: %s", {
        group: f"{counts['tokens']} ({counts['share']:.0%})"
        for group, counts in summary["supervised_by_group"].items()})
    for warning in report.warnings():
        log.warning("length check: %s", warning)
    if not report.ok:
        worst = sorted(report.over, key=lambda o: -int(o["tokens"]))[:5]
        raise TrainingError(
            f"{len(report.over)} staged row(s) exceed max_length "
            f"{swift.args['max_length']} or their task's output reservation, by the real "
            f"tokenizer and resize rule, though the corpus estimate passed them: {worst}. The corpus budget estimate is wrong for "
            "these rows — fix it (cap_check) and rebuild, rather than letting the trainer "
            "reach them."
        )
    return report


#: Statuses a run can be overwritten from: one that never finished.
_REPLACEABLE = ("training", "failed")


def _refuse_existing_run(run_id: str, client: BlobClient) -> None:
    """Refuse to overwrite a run that trained, was evaluated or promoted."""
    from registry_utils.query_registry import RegistryQueryError
    from registry_utils.query_registry import get as get_manifest

    try:
        existing = get_manifest(run_id, client)
    except (RegistryQueryError, KeyError, FileNotFoundError):
        return
    if existing.status not in _REPLACEABLE:
        raise TrainingError(
            f"{run_id} already exists with status {existing.status!r}. A trained, evaluated or "
            "promoted run is never overwritten — train the next version instead."
        )


def assert_checkpoint_path(continue_from: str, *, must_exist: bool = False) -> None:
    """Refuse a registry run-id where a checkpoint directory is required.

    ms-swift's ``resume_from_checkpoint`` reads a directory. Handed a run-id it
    finds nothing, trains from base, and the manifest records a lineage that
    never happened — a run that claims to continue v2 while being v1 again.

    The shape comes from :mod:`common.run_ids`, so ``policy-v2`` is refused for
    the same reason ``extractor-v2`` is. The three-alternative regex this
    replaced listed the lineages v1 happened to mint, and would have waved every
    other id straight through.
    """
    stripped = continue_from.strip().rstrip("/\\")
    # A bare run-id, with or without a trailing slash or a "./" prefix, is still
    # a run-id: "extractor-v3/" slipped through the old check.
    bare = stripped[2:] if stripped.startswith(("./", ".\\")) else stripped
    if is_valid_run_id(bare) and not any(sep in bare for sep in "/\\"):
        raise TrainingError(
            f"--continue-from got {continue_from!r}, which is a registry run-id, not a "
            "checkpoint path. ms-swift resumes from a DIRECTORY; given a run-id it silently "
            "trains from base while the manifest records a lineage that never happened. "
            "Pass the staged checkpoint directory instead."
        )
    if must_exist:
        from pathlib import Path

        directory = Path(stripped)
        if not (directory / "adapter_config.json").is_file():
            raise TrainingError(
                f"--continue-from {continue_from!r} is not an adapter directory (no "
                "adapter_config.json there). Pass the checkpoint directory itself, e.g. "
                ".../checkpoint-150."
            )


def _number(value: Any) -> float | None:
    return None if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)


#: ``lora.dropout: auto`` (:func:`resolve_lora_dropout`).
AUTO_DROPOUT = "auto"
#: Real (not synthetic) training documents every line needs for the lower rate.
AUTO_DROPOUT_MIN_REAL_DOCUMENTS = 200
AUTO_DROPOUT_LARGE_CORPUS = 0.05
AUTO_DROPOUT_SMALL_CORPUS = 0.10


def is_auto_dropout(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO_DROPOUT


def configured_dropout(lora: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The config's dropout with no corpus to count: its number as it is, or
    for ``auto`` the pilot-scale rate. Raises TrainingError on anything else."""
    value = lora.get("dropout")
    if is_auto_dropout(value):
        return AUTO_DROPOUT_SMALL_CORPUS, {
            "configured": AUTO_DROPOUT, "chosen": AUTO_DROPOUT_SMALL_CORPUS, "source": "default",
            "reason": "no corpus was counted; auto falls back to the pilot-scale rate"}
    number = _number(value)
    if number is None or not 0.0 <= number < 1.0:
        raise TrainingError(f"lora.dropout is {value!r}; it must be a number in [0, 1) or {AUTO_DROPOUT!r}")
    return value, {"configured": value, "chosen": value, "source": "config"}


def auto_dropout(train_lines: dict[str, dict[str, int]] | None) -> tuple[float, str]:
    """The rate ``auto`` chooses for a run from the base, and why: 0.05 when
    every line the run trains has at least 200 real training documents, else
    0.10 (the config's own rule). ``train_lines`` is the view's
    ``examples_by_line["train"]``: line -> {"documents", "real", "synthetic", "rows"}.
    """
    floor = AUTO_DROPOUT_MIN_REAL_DOCUMENTS
    real = {line: int((counts or {}).get("real") or 0) for line, counts in (train_lines or {}).items()}
    if not real:
        return AUTO_DROPOUT_SMALL_CORPUS, "no trained line was counted"
    short = {line: n for line, n in sorted(real.items()) if n < floor}
    if short:
        listed = ", ".join(f"{line} {n}" for line, n in sorted(short.items(), key=lambda kv: kv[1])[:5])
        return AUTO_DROPOUT_SMALL_CORPUS, (
            f"{len(short)} of {len(real)} trained line(s) have fewer than {floor} real training "
            f"documents ({listed}{', ...' if len(short) > 5 else ''})")
    return AUTO_DROPOUT_LARGE_CORPUS, (
        f"every trained line has at least {floor} real training documents (fewest: {min(real.values())})")


def parent_dropout(continue_from: str, parent_run_id: str | None = None,
                   client: BlobClient | None = None) -> tuple[float | None, str]:
    """The dropout the adapter a run continues was saved with, and where it was read.

    Its ``adapter_config.json`` first - ms-swift trains a continued adapter
    with that value whatever the command says - else its run's manifest
    (a dry run, where the pod's volume is not visible).
    """
    import json

    path = Path(continue_from.strip().rstrip("/\\")) / "adapter_config.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = None
    value = _number(saved.get("lora_dropout")) if isinstance(saved, dict) else None
    if value is not None:
        return value, f"saved in {path}"
    if parent_run_id and client is not None:
        from registry_utils.query_registry import RegistryQueryError
        from registry_utils.query_registry import get as get_manifest

        try:
            return float(get_manifest(parent_run_id, client).training_config.lora_dropout), (
                f"recorded on {parent_run_id}'s run manifest")
        except (RegistryQueryError, KeyError, FileNotFoundError, ValueError):
            pass
    return None, f"neither {path} nor the parent run's manifest gives it"


def resolve_lora_dropout(
    lora: dict[str, Any],
    *,
    train_lines: dict[str, dict[str, int]] | None,
    continue_from: str | None = None,
    parent_run_id: str | None = None,
    client: BlobClient | None = None,
    dry_run: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """The LoRA dropout a run trains with, and the record of why.

    A number in the config is used as it is. ``auto``: a run continuing an
    adapter keeps the adapter's own value (ms-swift would anyway, and so a
    cycle never stops continuing over it); a run from the base chooses from
    its corpus (:func:`auto_dropout`). Recorded on the manifest
    (``training_config.lora_dropout`` and ``lora_dropout_choice``).
    """
    if not is_auto_dropout(lora.get("dropout")):
        return configured_dropout(lora)
    if continue_from:
        inherited, where = parent_dropout(continue_from, parent_run_id, client)
        if inherited is not None:
            return inherited, {"configured": AUTO_DROPOUT, "chosen": inherited, "source": "parent",
                               "reason": f"continues an adapter saved at {inherited} ({where})"}
        if not dry_run:
            raise TrainingError(
                f"lora.dropout is auto and the run continues {continue_from!r}, whose dropout cannot "
                f"be read ({where}). ms-swift would train with the adapter's own value, which the "
                "manifest could not record."
            )
    value, reason = auto_dropout(train_lines)
    if continue_from:
        reason = f"the parent's value is not readable from here (a dry run); {reason}"
    return value, {"configured": AUTO_DROPOUT, "chosen": value, "source": "corpus", "reason": reason}


def adapter_config_mismatch(adapter_config: dict[str, Any], lora: dict[str, Any]) -> str | None:
    """The first LoRA setting an adapter was saved with that the config does not
    give it, or None when they agree.

    A continued run trains with the ADAPTER's settings: ms-swift (3.12) loads
    ``adapters[0]`` with ``from_pretrained(..., is_trainable=True)``, which takes
    its rank, alpha, dropout, rsLoRA, bias and targets from the adapter's own
    ``adapter_config.json`` and ignores the ``lora_*`` arguments. Once the
    config moved dropout from 0.10 to 0.05, every continued cycle would train
    at 0.10 and record 0.05.

    ``adapter_config`` is the parsed ``adapter_config.json`` (PEFT's names);
    ``lora`` is the training config's ``lora`` block.

    ``dropout: auto`` asks for the adapter's own dropout on a continued run
    (:func:`resolve_lora_dropout`), so any value it was saved with agrees -
    as long as it was saved with one.
    """
    import math

    auto = is_auto_dropout(lora.get("dropout"))
    for key, wanted in (("r", lora["rank"]), ("lora_alpha", lora["alpha"]),
                        ("lora_dropout", lora["dropout"])):
        if adapter_config.get(key) is None:
            return f"{key} is missing from the adapter, {wanted} in the config"
        have, want = _number(adapter_config[key]), _number(wanted)
        if auto and key == "lora_dropout":
            if have is None:
                return f"{key} is {adapter_config[key]!r} in the adapter, not a number"
            continue
        if have is None or want is None or not math.isclose(have, want, rel_tol=1e-9, abs_tol=1e-9):
            return f"{key} is {adapter_config[key]} in the adapter, {wanted} in the config"
    # Missing means PEFT's default, which is what from_pretrained would train with.
    for key, wanted, default in (("use_rslora", bool(lora.get("use_rslora", False)), False),
                                 ("bias", lora.get("bias", "none"), "none")):
        have = adapter_config.get(key)
        have = default if have is None else have
        # Strictly: an adapter's 1 is not the config's True.
        if have != wanted or type(have) is not type(wanted):
            return f"{key} is {have} in the adapter, {wanted} in the config"
    modules = set(lora["target_modules"]) | ({"merger"} if lora.get("include_vision_projector") else set())
    saved = adapter_config.get("target_modules")
    if isinstance(saved, str):
        # The regex ms-swift builds for a multimodal model: it must name every
        # module the config targets.
        absent = sorted(m for m in modules if m not in saved)
        if absent:
            return f"target_modules in the adapter is the pattern {saved!r}, which does not name {absent}"
    elif isinstance(saved, (list, tuple)):
        if set(map(str, saved)) != modules:
            return (f"target_modules are {sorted(map(str, saved))} in the adapter, "
                    f"{sorted(modules)} in the config")
    else:
        return f"target_modules is missing from the adapter, {sorted(modules)} in the config"
    return None


def assert_adapter_matches_config(continue_from: str) -> None:
    """Refuse to continue an adapter saved with other LoRA settings than the
    config's (:func:`adapter_config_mismatch`): ms-swift would train with the
    adapter's, while the manifest recorded the config's."""
    import json

    path = Path(continue_from.strip().rstrip("/\\")) / "adapter_config.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingError(f"--continue-from {continue_from!r}: {path} cannot be read ({exc})") from exc
    name = ADAPTER.training_config
    reason = (adapter_config_mismatch(saved, training_config(name)["lora"]) if isinstance(saved, dict)
              else f"{path.name} is not a JSON object")
    if reason:
        raise TrainingError(
            f"--continue-from {continue_from!r} does not have the LoRA settings of "
            f"configs/training/{name}.yaml: {reason}. ms-swift trains a continued run with the "
            "adapter's own settings (its adapter_config.json) and ignores the config's, while the "
            "manifest would record the config's. Start this cycle from the base instead (finetune "
            "--fresh; by hand, leave out --continue-from), or set the config's lora block back to "
            "the adapter's values."
        )


def count_examples(bucket: Any) -> int:
    """Sum a ``{doc_type: {mode: count}}`` bucket from the corpus manifest.

    The manifest nests counts by type and modality. ``int()`` on that dict is a
    TypeError, which is how the CLI used to fail before training started.
    """
    if not isinstance(bucket, dict):
        return 0
    return sum(
        count
        for modes in bucket.values()
        if isinstance(modes, dict)
        for count in modes.values()
    )


def launch_and_record(
    config: SwiftConfig,
    manifest: RunManifest,
    client: BlobClient,
) -> RunManifest:
    """Run training, then record what actually happened.

    The manifest is written before this is called — the run_id has to be reserved
    and the configuration captured even for a run that dies. But "written" is not
    "trained": the status stays ``training`` until ms-swift returns, and becomes
    ``failed`` if it does not, so the registry never claims weights that a
    crashed run never wrote.
    """
    try:
        launch(config)
    except Exception as exc:
        manifest.status = "failed"
        write_manifest(manifest, client)
        log.error("training failed (%s); manifest %s marked failed", exc, manifest.run_id)
        raise
    manifest.status = "trained"
    write_manifest(manifest, client)
    if manifest.artifacts and manifest.artifacts.staging_path:
        # The checkpoint nearest two-thirds of the run (checkpoint selection's
        # candidate) and the dropout the adapter was saved with, beside the marker.
        recorded = getattr(manifest, "training_config", None)
        summary = training_summary(manifest.artifacts.staging_path,
                                   lora_dropout=getattr(recorded, "lora_dropout", None))
        mark_training_complete(manifest.artifacts.staging_path, manifest.run_id, summary)
    return manifest


def launch(config: SwiftConfig) -> None:
    """Launch ms-swift. The training loop lives there, not here (arch v2.1 §10)."""
    import os
    import shutil
    import subprocess
    import sys

    # The environment this Python runs in comes first: ms-swift is installed next
    # to it, and PATH alone can point elsewhere (an unactivated venv, a tmux
    # session whose PATH came from the tmux server).
    search_path = os.pathsep.join(filter(None, (os.path.dirname(sys.executable), os.environ.get("PATH"))))
    swift = shutil.which("swift", path=search_path)
    if swift is None:
        raise TrainingError(
            "the `swift` CLI is not on PATH. ms-swift is the Layer-3 entrypoint (arch v2.1 §10); "
            'install the [train] extra on the pod: pip install -e ".[train]"\n'
            "If ms-swift turns out to lack a Qwen3-VL capability this needs, the documented "
            "fallback is a framework migration to TRL SFTTrainer — including re-implementing "
            "the template, collator and masking against the §10.2 parity tests. A contingency, "
            "not a layer swap."
        )
    argv = [swift, *config.to_cli()[1:]]
    log.info("launching: %s (env %s)", " ".join(argv), config.env)
    # swift starts its own workers (torchrun, python) by name: the same PATH.
    env = {**trainer_environment(os.environ), **config.env, "PATH": search_path}
    subprocess.run(argv, check=True, env=env)


#: Environment names the trainer never needs and must never hold. ms-swift runs
#: third-party code (the model's remote code, report integrations) and writes its
#: environment into run logs; the Blob connection string rode along into both.
_SECRET_MARKERS = (
    "AZURE", "CONNECTION_STRING", "SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY",
    "SAS", "RUNPOD", "CREDENTIAL",
)
#: Secrets the trainer does need: pulling the pinned base from the Hub.
_TRAINER_SECRETS = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"})


def trainer_environment(environ: Any) -> dict[str, str]:
    """``environ`` without the credentials the training subprocess has no use for."""
    return {
        key: value for key, value in environ.items()
        if key in _TRAINER_SECRETS or not any(m in key.upper() for m in _SECRET_MARKERS)
    }


# --------------------------------------------------------------------------
# Benchmark mode (--benchmark-steps): time a few steps, change one lever, repeat
# --------------------------------------------------------------------------

#: Written into :func:`benchmark_dir` by :func:`benchmark`.
BENCHMARK_FILE = "benchmark.json"

#: Training arguments a timing run has no use for: no evaluation, no checkpoint.
_BENCHMARK_DROPPED = ("val_dataset", "save_steps", "eval_steps", "save_total_limit")


def benchmark_dir(out_version: str) -> Path:
    """Where a benchmark for ``out_version`` writes: beside the runs' staging,
    never inside a run's own directory, so no checkpoint search ever sees it."""
    return Path(paths.staging_root()) / "benchmarks" / ADAPTER.name / out_version


def benchmark_args(steps: int) -> dict[str, Any]:
    """What turns the training command into a timing run of ``steps`` optimizer steps."""
    return {"max_steps": int(steps), "save_strategy": "no", "eval_strategy": "no",
            "logging_steps": 1, "load_best_model_at_end": False}


def read_step_log(path: str | Path) -> list[dict[str, Any]]:
    """The entries of ms-swift's ``logging.jsonl``; unreadable lines are skipped."""
    import json

    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def step_timings(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Seconds per optimizer step from ms-swift's per-step log (``logging_steps: 1``).

    ms-swift logs ``global_step/max_steps`` ("7/30") and ``train_speed(iter/s)``,
    the steps so far over the seconds since training began - so a step's
    elapsed time is its number over that speed, and a step's own time the
    difference from the step before. The first step carries the trainer's
    start-up and is reported apart (``first_step``). ``memory(GiB)`` gives
    the peak; ``train_runtime`` (the trainer's last log) the whole run.
    """
    elapsed: dict[int, float] = {}
    peak: float | None = None
    runtime: float | None = None
    for entry in entries:
        memory = _number(entry.get("memory(GiB)"))
        if memory is not None:
            peak = memory if peak is None else max(peak, memory)
        if _number(entry.get("train_runtime")) is not None:
            runtime = float(entry["train_runtime"])
        speed = _number(entry.get("train_speed(iter/s)"))
        try:
            step = int(str(entry.get("global_step/max_steps", "")).split("/")[0])
        except ValueError:
            continue
        if step > 0 and speed:
            elapsed[step] = step / speed
    ordered = sorted(elapsed.items())
    per_step = [(t2 - t1) / (s2 - s1) for (s1, t1), (s2, t2) in zip(ordered, ordered[1:], strict=False)]
    return {"first_step": ordered[0][1] / ordered[0][0] if ordered else None, "per_step": per_step,
            "logged_steps": len(ordered), "peak_memory_gib": peak, "train_runtime": runtime}


def _percentile(values: list[float], percent: int) -> float | None:
    """Nearest-rank percentile (``percent`` a whole number); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = -(-percent * len(ordered) // 100)          # ceil, in integers
    return ordered[min(len(ordered), max(1, rank)) - 1]


def suggest_next_experiment(batch: dict[str, Any], *, peak_memory_gib: float | None,
                            gpu_memory_gib: float | None) -> str:
    """The next lever of the measurement plan (configs/training/unified.yaml,
    ``batch``), one at a time: vision-tower checkpointing off, then two rows per
    device if memory leaves room, then dataloader workers."""
    if batch.get("vit_gradient_checkpointing") is not False:
        return ("set batch.vit_gradient_checkpointing: false (the vision tower is frozen, so "
                "recomputing its activations buys little memory) and benchmark again")
    headroom = (peak_memory_gib is None or gpu_memory_gib is None
                or peak_memory_gib <= 0.55 * gpu_memory_gib)
    if int(batch.get("per_device_train_batch_size") or 1) == 1 and headroom:
        return ("set batch.per_device_train_batch_size: 2 (gradient accumulation halves on its own, "
                f"the effective batch stays {batch.get('effective_batch_size')}) and benchmark again; "
                "keep it only if peak memory stays under ~90% of the card")
    workers = batch.get("dataloader_num_workers")
    if not workers:
        return "set batch.dataloader_num_workers: 4 and benchmark again; then 8"
    if int(workers) < 8:
        return "set batch.dataloader_num_workers: 8 and benchmark again"
    return ("every lever of the measurement plan is set: compare the benchmark.json files and keep "
            "each lever that lowered the median step time by more than the p90-median spread")


def _gpu_memory_gib() -> float | None:
    """One visible GPU's memory, where torch sees one."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:  # noqa: BLE001 - no torch, no GPU: not known
        pass
    return None


def benchmark_result(
    swift: SwiftConfig, timings: dict[str, Any], lengths: Any, *, batch: dict[str, Any],
    steps: int, wall_seconds: float | None = None, gpu_memory_gib: float | None = None,
) -> dict[str, Any]:
    """What :data:`BENCHMARK_FILE` holds: step time (median, p90), tokens/s,
    peak memory, the row-length histogram, supervised tokens per group, the
    levers it ran with, and the next lever to try.

    Tokens per step are estimated: the effective batch's rows at the staged
    rows' mean length (the rows are drawn at random, so the mean is fair).
    """
    import statistics

    per_step = list(timings.get("per_step") or [])
    median = statistics.median(per_step) if per_step else None
    if median is None and timings.get("train_runtime") and steps:
        median = float(timings["train_runtime"]) / steps          # no per-step log: the mean
    summary = lengths.summary() if lengths is not None else {}
    rows_per_step = (int(swift.args["per_device_train_batch_size"])
                     * int(swift.args["gradient_accumulation_steps"])
                     * max(1, int(swift.env.get("NPROC_PER_NODE") or 1)))
    tokens_per_step = rows_per_step * int(summary.get("mean_tokens") or 0) or None
    peak = timings.get("peak_memory_gib")
    dirty, diff = worktree_state()
    return {
        "steps": steps,
        "step_seconds": {"median": median, "p90": _percentile(per_step, 90),
                         "mean": statistics.fmean(per_step) if per_step else None,
                         "first_step": timings.get("first_step"), "timed_steps": len(per_step)},
        "rows_per_step": rows_per_step,
        "tokens_per_step_estimated": tokens_per_step,
        "tokens_per_second_estimated": round(tokens_per_step / median, 1) if tokens_per_step and median else None,
        "peak_memory_gib": peak,
        "gpu_memory_gib": gpu_memory_gib,
        "wall_seconds": wall_seconds,
        "train_runtime": timings.get("train_runtime"),
        "levers": {key: swift.args.get(key) for key in (
            "per_device_train_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
            *THROUGHPUT_KEYS, "padding_free", "max_length")},
        "gpus": int(swift.env.get("NPROC_PER_NODE") or 1),
        "length_histogram": summary.get("length_histogram", {}),
        "histogram_bucket_tokens": summary.get("histogram_bucket_tokens"),
        "supervised_by_group": summary.get("supervised_by_group", {}),
        "supervised_tokens": summary.get("supervised_tokens"),
        "code": {"commit": capture_git_commit(), "dirty": dirty, "diff_sha256": diff},
        "next_experiment": suggest_next_experiment(batch, peak_memory_gib=peak, gpu_memory_gib=gpu_memory_gib),
    }


def benchmark(
    *,
    corpus_version: str,
    out_version: str,
    client: BlobClient,
    steps: int,
    continue_from: str | None = None,
    parent_run_id: str | None = None,
    dry_run: bool = False,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Time ``steps`` optimizer steps of the run ``out_version`` would be, on
    its staged corpus, and write :data:`BENCHMARK_FILE` under :func:`benchmark_dir`.

    The training command the run would launch, with ``--max_steps``, no
    evaluation and no checkpoint (:func:`benchmark_args`): no manifest is
    written, no version reserved, nothing selected. A dirty working tree is
    allowed - measuring one lever at a time means editing the config between
    benchmarks - and recorded in the file. A dry run returns the command.
    """
    import json
    import time

    if steps < 1:
        raise TrainingError(f"--benchmark-steps must be at least 1, got {steps}")
    validate_all(require_pinned_revision=not dry_run)
    if not dry_run:
        assert_on_pod()
    view = materialize(corpus_version, client, tenant_id=tenant_id,
                       parent_run_id=parent_run_id if continue_from else None, out_version=out_version)
    cfg = training_config(ADAPTER.training_config)
    epochs = int(optimization_for(cfg, continuing=continue_from is not None)["num_train_epochs"])
    if continue_from:
        assert_checkpoint_path(continue_from, must_exist=not dry_run)
        if not dry_run:
            assert_adapter_matches_config(continue_from)
    dropout, choice = resolve_lora_dropout(
        cfg["lora"], train_lines=view.examples_by_line.get("train"), continue_from=continue_from,
        parent_run_id=parent_run_id, client=client, dry_run=dry_run)
    root = benchmark_dir(out_version)
    swift, _recorded = build_training_config(
        corpus_paths=list(view.epoch_files), output_dir=str(root / "run"), resume_from=continue_from,
        train_rows=sum(view.rows_by_epoch.get(e, 0) for e in range(1, epochs + 1)),
        lora_dropout=dropout, lora_dropout_choice=choice)
    for key in _BENCHMARK_DROPPED:
        swift.args.pop(key, None)
    swift.args.update(benchmark_args(steps))
    if dry_run:
        log.info("dry run - the benchmark would launch: %s", " ".join(swift.to_cli()))
        return {"dry_run": True, "argv": swift.to_cli(), "steps": steps}

    from training.stage_data import prune_image_caches, stage_training_data

    prune_image_caches(paths.staging_train_images_dir(corpus_version, tenant_id))
    staged = stage_training_data(
        view.epoch_files[:len(swift.args["dataset"])], None, client, root / "data",
        images_root=paths.staging_train_images_dir(corpus_version, tenant_id),
        max_pixels=int(swift.env["MAX_PIXELS"]),
    )
    swift.args["dataset"] = staged.epoch_files
    lengths = _assert_rows_fit(swift, staged.epoch_files, staged.output_caps, groups=staged.groups)
    _assert_masking(staged.epoch_files, swift)

    started = time.monotonic()
    launch(swift)
    wall = round(time.monotonic() - started, 1)
    logs = sorted((root / "run").rglob("logging.jsonl"), key=lambda p: p.stat().st_mtime)
    timings = step_timings(read_step_log(logs[-1])) if logs else step_timings([])
    if not logs:
        log.warning("ms-swift wrote no logging.jsonl under %s: no step times to read", root / "run")
    result = benchmark_result(swift, timings, lengths, batch=dict(cfg["batch"]), steps=steps,
                              wall_seconds=wall, gpu_memory_gib=_gpu_memory_gib())
    result.update({"corpus_version": corpus_version, "out_version": out_version,
                   "log": str(logs[-1]) if logs else None})
    root.mkdir(parents=True, exist_ok=True)
    (root / BENCHMARK_FILE).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    seconds = result["step_seconds"]
    print(f"\nbenchmark ({steps} steps): median {seconds['median']} s/step, p90 {seconds['p90']}, "
          f"~{result['tokens_per_second_estimated']} tokens/s, peak {result['peak_memory_gib']} GiB"
          f"\n  written: {root / BENCHMARK_FILE}\n  next experiment: {result['next_experiment']}\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Train the policy adapter LoRA")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-version", required=True)
    parser.add_argument("--deepspeed", default=None, choices=[None, "zero2", "zero3"])
    parser.add_argument("--train-vit", action="store_true", help="the §3 escalation; LoRA-on-ViT")
    parser.add_argument("--continue-from", default=None, help="a checkpoint DIRECTORY, not a run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tenant", default=None, help="whose corpus to read; defaults from env")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="train on a working tree with uncommitted changes (recorded on the manifest)")
    parser.add_argument("--benchmark-steps", type=int, default=None, metavar="N",
                        help="time N optimizer steps on the staged corpus instead of training: no "
                             "checkpoint, no evaluation, no manifest; writes benchmark.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('training.train', argv,
                               hint="train-benchmark" if args.benchmark_steps else None):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()
    if args.benchmark_steps is not None:
        benchmark(corpus_version=args.corpus, out_version=args.out_version, client=client,
                  steps=args.benchmark_steps, continue_from=args.continue_from,
                  dry_run=args.dry_run, tenant_id=args.tenant)
        return 0
    corpus_manifest = client.read_json(paths.corpus_manifest(args.corpus, args.tenant))
    counts = corpus_manifest.get("example_counts", {})
    train(
        corpus_version=args.corpus,
        out_version=args.out_version,
        client=client,
        corpus_manifest=corpus_manifest,
        data_stats=DataStats(
            train_examples=count_examples(counts.get("train")),
            val_examples=count_examples(counts.get("val")),
            test_examples=count_examples(counts.get("test")),
        ),
        train_vit=args.train_vit,
        deepspeed=args.deepspeed,
        continue_from=args.continue_from,
        dry_run=args.dry_run,
        tenant_id=args.tenant,
        allow_dirty=args.allow_dirty,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
