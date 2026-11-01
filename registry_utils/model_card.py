"""A model card per run, beside its manifest in the registry (Fideon SPEC_09 amendment item 7).

Markdown for a person: what the run trained on, line by line and split by
split; the real and synthetic documents and the shares they were sampled to;
the input-mode mix; the carriers held out; the hyperparameters; and the gate's
verdict. Rendered from the run manifest alone and rewritten with it
(:func:`registry_utils.write_run_manifest.write_manifest`), so the card never
says something the manifest does not.
"""

from __future__ import annotations

from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from registry_utils.models import RunManifest

SPLITS = ("train", "val", "test")


def _value(value: Any) -> str:
    if value is None:
        return "not set"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value)) or "none"
    return str(value)


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_value(c) if not isinstance(c, str) else c for c in row) + " |"
            for row in rows]
    return out


def _share(configured: Any, realised: Any) -> str:
    return f"{_value(configured)} -> {_value(realised) if realised is not None else 'unknown'}"


def _examples(stats: Any) -> list[str]:
    by_split = stats.examples_by_line or {}
    out = ["### Examples per line and split", ""]
    if not by_split:
        return out + [f"Per line not recorded (the unified scope reads the corpus whole). Rows: "
                      f"train {stats.train_examples}, val {stats.val_examples}, "
                      f"test {stats.test_examples}.", ""]
    lines = sorted({line for split in by_split.values() for line in split})
    rows = []
    for line in lines:
        cells: list[Any] = [line]
        for split in SPLITS:
            entry = by_split.get(split, {}).get(line)
            cells.append(f"{entry['documents']} ({entry['real']} real, {entry['synthetic']} synthetic)"
                         if entry else "0")
        rows.append(cells)
    out += _table(["Line", "Train documents", "Val documents", "Test documents"], rows)
    return out + ["", f"Train documents are one epoch's, before line balance repeats any. Rows over "
                      f"the epochs the run uses: train {stats.train_examples}, val "
                      f"{stats.val_examples}, test {stats.test_examples}.", ""]


def _mix(stats: Any) -> list[str]:
    mix = stats.data_mix or {}
    settings = mix.get("settings") or {}
    out = ["### Real and synthetic documents, synthetic_fraction and scanned_share", ""]
    if not mix:
        return out + ["Not applied: the unified scope trains on the corpus as built.", ""]
    out += [f"Scope defaults: synthetic_fraction {_value(settings.get('synthetic_fraction'))}, "
            f"scanned_share {_value(settings.get('scanned_share'))}. Per line:", ""]
    configured = settings.get("lines") or {}
    rows = []
    for line, entry in sorted((mix.get("lines") or {}).items()):
        rows.append([
            line, entry.get("real"),
            f"{entry.get('synthetic_kept')} of {entry.get('synthetic_available')}",
            _share(entry.get("synthetic_fraction"), entry.get("realised_synthetic_fraction")),
            _share(entry.get("scanned_share"), entry.get("realised_scanned_share")),
            entry.get("repeats", 1), entry.get("note") or "",
        ])
    out += _table(["Line", "Real", "Synthetic kept", "synthetic_fraction (configured -> reached)",
                   "scanned_share (configured -> reached)", "Repeats per epoch", "Note"], rows)
    unused = sorted(set(configured) - set(mix.get("lines") or {}))
    if unused:
        out += ["", "Configured for lines with no train documents in this run: " + ", ".join(
            f"{line} (synthetic_fraction {_value(configured[line].get('synthetic_fraction'))}, "
            f"scanned_share {_value(configured[line].get('scanned_share'))})" for line in unused)]
    rested = mix.get("rested_documents") or 0
    out += ["", f"Train documents the shares left out: {rested}. Every real document trains.", ""]
    rotation = mix.get("render_rotation") or {}
    if not rotation.get("enabled"):
        return out + ["Render rotation: off - both renders of a twin train in every epoch.", ""]
    out += ["Render rotation: on - each twin trains in one render per epoch, the other the next "
            "epoch. Per line:", ""]
    out += _table(["Line", "Train documents", "Documents per epoch", "Twins rotated", "Both renders kept"],
                  [[line, r.get("documents"), r.get("documents_per_epoch"), r.get("twins_rotated"),
                    "yes (under the balance floor)" if r.get("both_renders") else "no"]
                   for line, r in sorted((rotation.get("lines") or {}).items())])
    return out + [""]


def _modes(stats: Any) -> list[str]:
    target = stats.modality_mix_target or {}
    realised = stats.modality_mix or {}
    out = ["### Input-mode mix", ""]
    modes = sorted(set(target) | set(realised))
    if not modes:
        return out + ["Not recorded.", ""]
    out += _table(["Mode", "Target", "Train rows"],
                  [[m, target.get(m), realised.get(m)] for m in modes])
    return out + [""]


def _hold_out(stats: Any) -> list[str]:
    policy = stats.split_policy or {}
    out = ["### Carrier hold-out", ""]
    held = policy.get("held_out_carriers_by_line") or {}
    if not policy:
        return out + ["Not recorded.", ""]
    if policy.get("frozen_eval_set"):
        out += ["The eval set is frozen: its documents are the test set, and the carriers below are "
                "the ones held out when it was frozen.", ""]
    rows = [[doc_type, line, carrier] for doc_type, by_line in sorted(held.items())
            for line, carrier in sorted(by_line.items())]
    whole = {k: v for k, v in (policy.get("held_out_carriers") or {}).items() if v}
    out += (_table(["Doc type", "Line", "Carrier held out of train and val in this line"], rows) if rows
            else [] if whole else ["No carrier held out."])
    if whole:
        out += ["", "Held out of every line: " + "; ".join(
            f"{k}: {', '.join(v)}" for k, v in sorted(whole.items())) + "."]
    single = {k: v for k, v in (policy.get("single_carrier_lines") or {}).items() if v}
    kept = {k: v for k, v in (policy.get("lines_not_held_out") or {}).items() if v}
    unnamed = {k: v for k, v in (policy.get("lines_without_carrier") or {}).items() if v}
    out += ["", "Lines with one carrier, none held out: "
            + ("; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(single.items())) or "none") + ".",
            "Lines whose only train or validation carrier could not be held out: "
            + ("; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sorted(kept.items())) or "none") + ".",
            "Lines with no carrier named: "
            + ("; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(unnamed.items())) or "none") + ".",
            f"Families moved into test by the hold-out: "
            f"{sum((policy.get('moved_to_test') or {}).values())}.",
            f"Twin cap per seed and render mode: {_value(policy.get('twin_cap'))}; twins it left out "
            f"of train: {policy.get('twins_dropped') or 0}.", ""]
    return out


def _hyperparameters(manifest: RunManifest) -> list[str]:
    config = manifest.training_config.model_dump(mode="json")
    return ["## Hyperparameters", "",
            *_table(["Setting", "Value"], [[k, v] for k, v in config.items() if v is not None]), ""]


def _gate(manifest: RunManifest) -> list[str]:
    promotion = manifest.promotion
    out = ["## Gate", ""]
    if promotion.beat_previous_on_all_gates is None and not promotion.failed_gates:
        return out + ["Not gated yet.", ""]
    waived = bool(promotion.failed_gates) and all(g.endswith("(WAIVED)") for g in promotion.failed_gates)
    verdict = ("passed" if promotion.beat_previous_on_all_gates
               else "gate not passed, promoted by override" if waived else "blocked")
    if promotion.tier and verdict != "blocked":
        verdict += f", {promotion.tier} release"
    out += [f"Verdict: {verdict}.",
            f"Failed gates: {', '.join(promotion.failed_gates) or 'none'}.",
            f"Gated against: {promotion.gated_against or 'no previous release'}.", ""]
    scores = [[name, value] for name, value in manifest.eval_metrics.model_dump(mode="json").items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if scores:
        out += _table(["Metric", "Score"], scores) + [""]
    return out


def render_model_card(manifest: RunManifest) -> str:
    stats = manifest.data_stats
    deps = manifest.dependencies
    lines = [
        f"# Model card: {manifest.run_id}", "",
        *_table(["", ""], [
            ["Scope", manifest.scope or manifest.run_type],
            ["Document types", manifest.doc_types or manifest.doc_type],
            ["Status", manifest.status],
            ["Created", manifest.created_at.isoformat()],
            ["Corpus", deps.corpus_version],
            ["Base model", deps.base_model],
            ["Code commit", deps.code_git_commit],
        ]), "",
        "## Training data", "",
        *_examples(stats), *_mix(stats), *_modes(stats), *_hold_out(stats),
        *_hyperparameters(manifest),
        *_gate(manifest),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_model_card(manifest: RunManifest, client: BlobClient) -> str:
    key = paths.model_card(manifest.run_id, manifest.run_type, manifest.doc_type, scope=manifest.scope)
    client.write_text(key, render_model_card(manifest))
    return key
