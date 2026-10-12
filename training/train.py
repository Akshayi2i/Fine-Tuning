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
from common.config import (
    base_model_config,
    base_model_dir,
    base_model_source,
    sequence_for_task,
    training_config,
    validate_all,
)
from common.run_ids import is_valid_run_id
from common.scopes import Scope, default_scope, get_scope
from common.tasks import Task
from registry_utils.models import (
    Artifacts,
    DataStats,
    Dependencies,
    RunManifest,
    TrainingConfig,
)
from registry_utils.write_run_manifest import capture_git_commit, is_dirty_worktree, write_manifest
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


#: Written beside the adapter checkpoints when ms-swift returns successfully.
#: The pipeline's "trained" means this file, never the directory: ms-swift
#: creates the directory when training STARTS, so a crashed run looked finished
#: and a rerun went straight on to choose among its partial checkpoints.
TRAINING_COMPLETE = "training_complete.json"


def training_completed(output_dir: str | Path) -> bool:
    return (Path(output_dir) / TRAINING_COMPLETE).is_file()


def mark_training_complete(output_dir: str | Path, run_id: str) -> None:
    import json
    from datetime import UTC, datetime

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / TRAINING_COMPLETE).write_text(json.dumps({
        "run_id": run_id, "completed_at": datetime.now(UTC).isoformat(),
    }), encoding="utf-8")


RUN_FINGERPRINT = "run_fingerprint.json"

#: What a resumed run must share with the run that wrote the checkpoint. Not
#: the per-device batch or the accumulation steps: those follow the GPU count,
#: and resuming on a different pod is the point of resuming.
_FINGERPRINT_SETTINGS = ("lora_rank", "lora_alpha", "learning_rate", "lr_scheduler", "epochs",
                         "effective_batch_size", "target_modules")


def run_fingerprint(corpus_version: str, corpus_manifest: Any, recorded: Any, scope: Any) -> dict:
    """The corpus and the settings a run trains with - what a checkpoint is OF."""
    import hashlib
    import json

    corpus = json.dumps(corpus_manifest, sort_keys=True, default=str)
    return {
        "corpus_version": corpus_version,
        "corpus_manifest_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "scope": getattr(scope, "name", None),
        **{name: getattr(recorded, name, None) for name in _FINGERPRINT_SETTINGS},
        "settings_sha256": _settings_hash(scope),
    }


def _settings_hash(scope: Any) -> str | None:
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

    name = getattr(scope, "training_config", None)
    if not name:
        return None
    try:
        import yaml

        from common.config import SHARED_SEQUENCE_CONFIG, SHARED_VISION_CONFIG, training_config

        cfg = json.loads(json.dumps(training_config(name), default=str))
        cfg.pop("distributed", None)
        batch = dict(cfg.get("batch") or {})
        for key in ("per_device_train_batch_size", "per_device_eval_batch_size",
                    "gradient_accumulation_steps"):
            batch.pop(key, None)
        cfg["batch"] = batch
        shared = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in
                  (SHARED_VISION_CONFIG, SHARED_SEQUENCE_CONFIG)]
    except Exception as exc:  # noqa: BLE001 - a fingerprint must not stop a run from starting
        log.warning("the run fingerprint could not read the training settings: %s", exc)
        return None
    body = json.dumps([cfg, shared], sort_keys=True, default=str)
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
    """
    import json

    path = Path(output_dir) / RUN_FINGERPRINT
    if not path.is_file():
        log.warning("%s has no %s (a run started before it was recorded); resuming without "
                    "checking that the corpus and settings are the same", output_dir, RUN_FINGERPRINT)
        return
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(fingerprint, default=str))
    changed = sorted(k for k in {*stored, *current} if stored.get(k) != current.get(k))
    if changed:
        raise TrainingError(
            f"{output_dir} holds checkpoints of a different run: {changed} changed since they "
            "were written. Resuming would continue the old weights and optimizer on a corpus "
            "or configuration they were not trained with. Train under a new out-version, or "
            "remove that directory to start this one again."
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


def corpus_max_length(scope: Scope | None = None) -> int:
    """The single ``max_length`` for a mixed-task corpus.

    ms-swift takes one value and the corpus interleaves every task, so this is
    the **largest** per-task cap (arch v2.1 §7a). Taking anything smaller would
    truncate the longest task rather than reject it, and a clipped assistant span
    trains the model to stop early.

    It is the cap that decides whether a run fits on one 80GB card, which is why
    ``use_logits_to_keep`` and padding-free batching are not optional here.

    **Scoped to what the run actually carries.** Taking the maximum over every
    task and every document type gives 32768 — the policy extraction override —
    to a lossrun-only run whose largest cap is 20480, and that cap is what
    decides the card. A scope narrows both dimensions: its own tasks, and only
    the per-doc-type overrides for types it trains.
    """
    from common.config import shared_sequence_config

    declared = shared_sequence_config().get("tasks", {})
    tasks = tuple(scope.tasks) if scope else tuple(Task)
    doc_types = set(scope.doc_types) if scope else None

    caps: list[int] = []
    for task in tasks:
        name = str(task)
        caps.append(int(sequence_for_task(name)["max_seq_len"]))
        # Per-doc-type overrides are separate caps, not variations on one.
        # Iterating only the bare tasks gave 24576 while the policy extraction
        # override is 32768, so every routed policy would have been TRUNCATED —
        # which is the exact failure the §7a caps exist to prevent.
        for doc_type in (declared.get(name, {}).get("by_doc_type") or {}):
            if doc_types is not None and doc_type not in doc_types:
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
    scope: Scope | None = None,
    train_rows: int | None = None,
) -> tuple[SwiftConfig, TrainingConfig]:
    """Assemble the ms-swift arguments and the manifest's record of them.

    Both come from the same source so the manifest describes what actually ran,
    rather than what a YAML file happened to say afterwards.

    ``corpus_paths`` are the materialized epoch files, in order. The run reads the
    first ``num_train_epochs`` of them **once each** — see the epoch comment
    below for why ms-swift is told one epoch.
    """
    base = base_model_config()
    scope = scope or default_scope()
    cfg = training_config(config_name or scope.training_config)
    lora, opt, batch = cfg["lora"], cfg["optimization"], cfg["batch"]
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

    max_length = corpus_max_length(scope)
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
        "lora_dropout": lora["dropout"],
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
        lora_dropout=lora["dropout"],
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
    env = _pixel_budget(scope)
    recorded.pixel_budget = {key: int(value) for key, value in env.items()}
    if gpus > 1:
        # ms-swift starts one training process per GPU (torchrun) when this is set.
        env["NPROC_PER_NODE"] = str(gpus)
        # Every trainable parameter (the LoRA layers) takes part in every step, so
        # DDP need not search the graph for unused ones - and with reentrant
        # gradient checkpointing that search fails.
        args["ddp_find_unused_parameters"] = False
    return SwiftConfig(args, env=env), recorded


def _pixel_budget(scope: Scope) -> dict[str, str]:
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
        min_pixels, max_pixels = pixel_budget(scope.tasks)
    except ConfigError as exc:
        raise TrainingError(f"scope {scope.name!r}: {exc}") from exc
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


def build_manifest(
    *,
    run_id: str,
    corpus_version: str,
    corpus_manifest: dict[str, Any],
    training_cfg: TrainingConfig,
    data_stats: DataStats,
    staging_path: str,
    continued_from: str | None = None,
    scope: Scope | None = None,
    tenant_id: str | None = None,
) -> RunManifest:
    """Build the run manifest. Written to Blob even while weights are staged."""
    base = base_model_config()["model"]
    if is_dirty_worktree():
        log.warning(
            "the working tree has uncommitted changes, so code_git_commit does not fully "
            "describe what ran — this run is not exactly reproducible."
        )

    scope = scope or default_scope()
    return RunManifest(
        run_id=run_id,
        # A unified-scope run keeps writing "unified", so extractor-v1 and
        # extractor-v2 remain the same kind of thing in the registry. Anything
        # narrower is "scoped" and says what it covers.
        run_type="unified" if scope.is_unified else "scoped",
        scope=None if scope.is_unified else scope.name,
        doc_types=[] if scope.is_unified else list(scope.doc_types),
        # Whose corpus trained it. The field existed and nothing set it, so a
        # manifest could not say which tenant's documents its weights came from.
        tenant_id=paths._tenant(tenant_id),
        continued_from=continued_from,
        dependencies=Dependencies(
            base_model=f"{base['model_id']}@{base['revision']}",
            corpus_version=corpus_version,
            code_git_commit=capture_git_commit(),
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
    scope: Scope | None = None,
) -> tuple[SwiftConfig, RunManifest]:
    """Configure and launch one training run.

    Args:
        continue_from: **a filesystem checkpoint path**, not a registry run-id.
            It goes straight to ms-swift's ``resume_from_checkpoint``, which reads
            a directory — passing a run-id like ``extractor-v3`` produced a run
            that silently trained from base while its manifest recorded a lineage
            that never happened. ``assert_checkpoint_path`` refuses the run-id
            shape rather than letting it through.
        dry_run: assemble and record everything without launching.
        tenant_id: whose corpus to read. Omitting it reads the default tenant's
            files, which for any other tenant do not exist — or worse, do.
        scope: what this run covers (``common.scopes``). Defaults to the unified
            scope, which trains every document type and writes exactly the run
            id, staging path and manifest shape it always did.
    """
    validate_all(require_pinned_revision=not dry_run)
    if not dry_run:
        assert_on_pod()
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
    scope = scope or default_scope()
    from training.data_mix import DataMixError, assert_corpus_mix

    try:
        assert_corpus_mix(scope, corpus_manifest)
    except DataMixError as exc:
        raise TrainingError(str(exc)) from exc
    # The corpus is built once for every type; a narrower scope reads a filtered
    # VIEW of it rather than a corpus of its own, so the split and the group
    # assignment are shared and the two runs stay comparable.
    view = materialize(scope, corpus_version, client, tenant_id=tenant_id)
    corpus_paths = list(view.epoch_files)
    val_paths = [view.val_path]
    staging = paths.scoped_staging_adapter_dir(scope.name, out_version)

    # The rows the run actually trains on: the first N epoch files, not all four
    # that are materialized. Counting all four overstated the data by a third,
    # and the same count sizes the evaluation interval.
    epochs = int(training_config(scope.training_config)["optimization"]["num_train_epochs"])
    train_rows = (
        sum(view.rows_by_epoch.get(e, 0) for e in range(1, epochs + 1)) if not scope.is_unified
        else sum(_count_rows(client, key) for key in view.epoch_files[:epochs])
    )
    update: dict[str, Any] = {"train_examples": train_rows}
    if not scope.is_unified:
        # The caller counts the whole corpus. A scoped run trains on its view,
        # and its manifest has to say so: a lossrun run recording every policy
        # row as its training data describes a run that did not happen.
        update |= {"val_examples": view.val_rows, "test_examples": view.test_rows,
                   # The shares configured and reached, per line, for the
                   # manifest and the model card.
                   "data_mix": {"settings": view.mix_settings, "lines": view.data_mix,
                                "rested_documents": view.rested_documents,
                                "render_rotation": view.render_rotation},
                   "examples_by_line": view.examples_by_line}
    data_stats = data_stats.model_copy(update=update)

    # A version already trained or promoted is never overwritten — not by a
    # re-run, and not by a dry run, which used to replace a promoted manifest
    # with a fresh "training" one and leave that row in the index for good.
    _refuse_existing_run(scope.run_id(out_version), client)

    # Every check that costs nothing runs before staging, which downloads every
    # page image the run reads. A run that is going to be refused should be
    # refused before that, not after thousands of images reach a paid pod.
    if continue_from:
        # Existence is checked on a real run only: the directory is on the pod's
        # volume, which a dry run on an operator machine cannot see.
        assert_checkpoint_path(continue_from, must_exist=not dry_run)
    # Counted only for a real run, where it decides anything; a dry run launches
    # nothing and would read the unified validation file for no reason.
    if not dry_run and not (
        view.val_rows if not scope.is_unified else _count_rows(client, view.val_path)
    ):
        # The config asks for evaluation, early stopping and
        # load_best_model_at_end, and checkpoint selection generates on
        # validation: with no rows the run fails after the GPU is paid for, or
        # trains with nothing to choose its checkpoint by.
        raise TrainingError(
            f"scope {scope.name!r} has no validation rows in corpus {corpus_version}. "
            "Checkpoint selection and early stopping both read validation, so the run "
            "cannot choose what to ship. Rebuild the corpus with enough documents of "
            f"{list(scope.doc_types)} to fill a validation split."
        )

    # Built against the Blob keys first: this is where the epoch count and every
    # other configuration error surfaces, still before any download.
    swift, recorded = build_training_config(
        corpus_paths=corpus_paths,
        val_paths=val_paths,
        output_dir=staging,
        train_vit=train_vit,
        deepspeed=deepspeed,
        resume_from=continue_from,
        scope=scope,
        train_rows=train_rows,
    )

    # Before the manifest is rewritten and every page image is staged: a resume
    # that will be refused should be refused in seconds, having changed nothing.
    fingerprint = run_fingerprint(corpus_version, corpus_manifest, recorded, scope)
    resumed = None if (continue_from or dry_run) else resume_point(staging)
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

        sample_rows = loss_sample_rows(training_config(scope.training_config)["evaluation"])
        staged = stage_training_data(
            view.epoch_files[:epochs], view.val_path, client,
            paths.staging_train_data_dir(scope.name, out_version),
            images_root=paths.staging_train_images_dir(corpus_version, tenant_id),
            max_pixels=int(swift.env["MAX_PIXELS"]),
            # The in-training checks read a fixed stratified validation sample,
            # chosen on the corpus rows before staging strips the line and mode off them.
            val_filter=(lambda rows: validation_sample(rows, sample_rows)) if sample_rows else None,
        )
        swift.args["dataset"] = staged.epoch_files
        swift.args["val_dataset"] = [staged.val_path]
        _assert_rows_fit(swift, [*staged.epoch_files, staged.val_path], staged.output_caps)
        _assert_masking(staged.epoch_files, swift)

    # Only the epochs this run uses. Four files are always materialized because
    # the §11a sweep tests up to four passes (§6.1), and a sweep that regenerates
    # its own data is not comparing what it thinks it is.
    manifest = build_manifest(
        run_id=scope.run_id(out_version),
        corpus_version=corpus_version,
        corpus_manifest=corpus_manifest,
        training_cfg=recorded,
        data_stats=data_stats,
        staging_path=staging,
        continued_from=continue_from,
        scope=scope,
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
    swift: SwiftConfig, files: list[str], output_caps: dict[str, list[int]] | None = None
) -> None:
    """Refuse the run before launch if any staged row is over ``max_length``.

    Under strict encoding such a row stops the run when the trainer reaches it,
    possibly hours in. Measured here instead, with the real tokenizer and the
    real resize rule, it stops the run before any GPU time is spent on it.
    """
    from training.length_check import measure

    report = measure(
        files,
        max_length=int(swift.args["max_length"]),
        count_tokens=_token_counter(),
        min_pixels=int(swift.env["MIN_PIXELS"]),
        max_pixels=int(swift.env["MAX_PIXELS"]),
        output_caps=output_caps,
    )
    log.info("length check: %d row(s), longest %d tokens", report.rows, report.longest)
    if not report.ok:
        worst = sorted(report.over, key=lambda o: -int(o["tokens"]))[:5]
        raise TrainingError(
            f"{len(report.over)} staged row(s) exceed max_length "
            f"{swift.args['max_length']} or their task's output reservation, by the real "
            f"tokenizer and resize rule, though the corpus estimate passed them: {worst}. The corpus budget estimate is wrong for "
            "these rows — fix it (cap_check) and rebuild, rather than letting the trainer "
            "reach them."
        )


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
            "promoted run is never overwritten — choose a new --out-version."
        )


def _count_rows(client: BlobClient, key: str) -> int:
    """Rows in a Blob JSONL file, without staging it. Zero when it is absent."""
    if not client.exists(key):
        return 0
    return sum(1 for line in client.read_text(key).splitlines() if line.strip())


def assert_checkpoint_path(continue_from: str, *, must_exist: bool = False) -> None:
    """Refuse a registry run-id where a checkpoint directory is required.

    ms-swift's ``resume_from_checkpoint`` reads a directory. Handed a run-id it
    finds nothing, trains from base, and the manifest records a lineage that
    never happened — a run that claims to continue v2 while being v1 again.

    The shape comes from :mod:`common.run_ids`, so a scoped id (``policy-v2``) is
    refused for the same reason ``extractor-v2`` is. The three-alternative regex
    this replaced listed the lineages v1 happened to mint, and would have waved
    every scoped id straight through.
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
        mark_training_complete(manifest.artifacts.staging_path, manifest.run_id)
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


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Train the unified extractor LoRA")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out-version", required=True)
    parser.add_argument("--deepspeed", default=None, choices=[None, "zero2", "zero3"])
    parser.add_argument("--train-vit", action="store_true", help="the §3 escalation; LoRA-on-ViT")
    parser.add_argument("--continue-from", default=None, help="a checkpoint DIRECTORY, not a run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tenant", default=None, help="whose corpus to read; defaults from env")
    parser.add_argument("--scope", default=None,
                        help="what this run covers (configs/scopes.yaml); defaults to unified")
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('training.train', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()
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
        scope=get_scope(args.scope) if args.scope else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
