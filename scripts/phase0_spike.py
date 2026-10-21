"""Phase 0 dependency spike — run this on a throwaway RunPod pod.

Answers, in one command, every question the build is currently guessing at. Each
one de-risks a decision that is expensive to reverse later:

======================================  ===================================================
check                                   what its answer changes
======================================  ===================================================
CUDA and GPU class                      whether anything below can run at all
ms-swift + Qwen3-VL multimodal LoRA     the Layer-3 entrypoint choice (arch §10). If it
                                        fails, the documented fallback is TRL SFTTrainer
flash-attn                              install pain is the usual blocker; sdpa is the
                                        fallback, at a speed cost
interleaved image/text content          the corpus row format (IMPL-07). If a trainer or
                                        server rejects it, page pairing needs the
                                        single-block fallback with inline markers
vLLM multi-LoRA hot-swap                IMPL-11's serving design. If it fails, serve merged
                                        per-type models instead
MinerU on CUDA                          the whole data pipeline; GPU is the default mode
llama.cpp mmproj for Qwen3-VL           whether GGUF export can produce a model that can
                                        see. Optional edge export only under arch v2.1 §13a
merger / aligner module names           the freeze flags and any vision LoRA target (§9a). A
                                        wrong name does not raise — it silently trains or
                                        freezes nothing
visual tokens per page                  every sequence cap in §7a. Measured through the real
                                        image processor, not derived from an assumed patch
                                        geometry
peak VRAM per task cap                  the §14 pod-class table and whether the 32k cap fits
                                        on one 80GB card (§9.3)
sequence parallelism + memory flags     the §9.3 escape hatch, and use_logits_to_keep /
                                        padding-free / freeze_aligner from §11.1
vLLM structured outputs                 whether schema validity is guaranteed at decode
                                        time (§13)
raw logprobs under constraint           whether confidence features see the model's own
                                        distribution or the post-mask one (§5.1). If masked,
                                        every calibrator is fitted on an artefact
FP8 export (llm-compressor)             the v2.1 default serving format (§13a)
MinerU determinism                      whether the OCR pin actually makes a corpus
                                        reproducible (§8a)
======================================  ===================================================

**Every check is independent and none aborts the run.** A spike that stops at the
first failure tells you one thing; this tells you all of them in one pod-hour.

Items map to arch v2.1 §16.0, which is explicit that nothing is annotated at
scale until they are confirmed on the pinned versions.

    pip install -e ".[data,train,serve]"        # expect flash-attn to be the hard part
    python scripts/phase0_spike.py --pdf sample.pdf --out spike_report.json

The report is written as JSON and printed as prose. Paste the prose back.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HUB_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _model_source() -> str:
    """The pod's local base (configs/base_model.yaml `local_dir`), else the Hub id.

    The spike checks the weights training and serving will load, so it loads
    them from the same place.
    """
    try:
        from common.config import base_model_source

        return base_model_source()
    except Exception:  # noqa: BLE001 - the spike must start even without the repo on path
        return HUB_MODEL_ID


MODEL_ID = _model_source()


@dataclass
class Result:
    """One check. ``ok=None`` means it could not be determined, which is not a pass."""

    name: str
    ok: bool | None = None
    detail: str = ""
    decides: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "UNKNOWN" if self.ok is None else ("PASS" if self.ok else "FAIL")


def run(name: str, decides: str, fn) -> Result:
    """Run one check, converting any exception into a reported failure."""
    result = Result(name=name, decides=decides)
    started = time.perf_counter()
    try:
        fn(result)
    except Exception as exc:  # noqa: BLE001 - a check that raises is a check that failed
        result.ok = False
        result.detail = f"{type(exc).__name__}: {exc}"
        result.data["traceback"] = traceback.format_exc(limit=4)
    result.data["seconds"] = round(time.perf_counter() - started, 1)
    print(f"  [{result.status:7}] {name}: {result.detail[:150]}")
    return result


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _struct_fields(cls) -> set[str]:
    """A class's field names, whatever it is built with: vLLM 0.11's SamplingParams
    is a msgspec Struct (``__struct_fields__``), not a dataclass, so looking only at
    ``__dataclass_fields__`` found no fields at all."""
    import contextlib
    import inspect

    fields = set(getattr(cls, "__struct_fields__", ()))
    fields |= set(getattr(cls, "__dataclass_fields__", {}))
    with contextlib.suppress(TypeError, ValueError):
        fields |= set(inspect.signature(cls).parameters)
    return fields


def check_gpu(r: Result) -> None:
    import torch

    r.data["torch"] = torch.__version__
    if not torch.cuda.is_available():
        r.ok = False
        r.detail = "no CUDA device visible — nothing below can be trusted"
        return
    r.ok = True
    r.data["gpu"] = torch.cuda.get_device_name(0)
    r.data["count"] = torch.cuda.device_count()
    r.data["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    r.detail = f"{r.data['gpu']} x{r.data['count']}, {r.data['vram_gb']}GB, torch {torch.__version__}"


def check_flash_attn(r: Result) -> None:
    try:
        import flash_attn

        r.ok = True
        r.detail = f"flash-attn {getattr(flash_attn, '__version__', 'installed')}"
    except ImportError as exc:
        # Not fatal: sdpa works. Recorded because it is the usual install failure
        # and its absence costs throughput, which changes pod-hour estimates.
        r.ok = False
        r.detail = f"not importable ({exc}); fall back to attn_implementation='sdpa' and note the cost"


def check_ms_swift(r: Result) -> None:
    """Does ms-swift accept the EXACT command training renders?

    Parsed with ms-swift's own argument parser, not grepped from ``--help``. The
    grep checked six flags by substring and could not see a renamed argument, a
    2.x name in a 3.x install, or a value outside a flag's choices — the ways a
    real run dies at argument parsing, on the pod, after staging. ``parse_known_args``
    only parses: no model is loaded and no dataset path is opened.
    """
    import shutil

    if shutil.which("swift") is None:
        r.ok = False
        r.detail = "the `swift` CLI is not on PATH; install the [train] extra"
        return

    from transformers import HfArgumentParser

    try:
        from swift.llm import TrainArguments
    except ImportError as exc:
        r.ok = False
        r.detail = f"not ms-swift 3 (no swift.llm.TrainArguments: {exc}); pyproject pins >=3.9"
        return

    from training.train import build_training_config

    swift, _ = build_training_config(
        corpus_paths=["e1.jsonl", "e2.jsonl", "e3.jsonl", "e4.jsonl"],
        output_dir="/tmp/spike-out", val_paths=["val.jsonl"], train_rows=400,
    )
    argv = swift.to_cli()[2:]                       # drop "swift sft"
    try:
        _namespace, unknown = HfArgumentParser(TrainArguments).parse_known_args(argv)
    except SystemExit as exc:                        # argparse: a value outside its choices
        r.ok = False
        r.detail = f"ms-swift refused the rendered command (argparse exit {exc.code}); see argv"
        r.data["argv"] = argv
        return
    r.data["argv"] = argv
    r.data["unknown"] = unknown
    r.ok = not unknown
    r.detail = (
        "ms-swift's parser accepts every argument and value training renders"
        if r.ok else f"ms-swift does not know {unknown} — rename them in training/train.py"
    )


def check_model_loads(r: Result) -> None:
    """Config-only, not weights. Confirms the model id resolves and the class exists."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    r.ok = True
    r.data["architectures"] = getattr(config, "architectures", None)
    r.data["model_type"] = getattr(config, "model_type", None)
    r.detail = f"{r.data['model_type']} / {r.data['architectures']}"


def check_interleaved_content(r: Result) -> None:
    """Does the processor accept image, text, image, text in one user turn?

    This is the corpus row format. If a trainer or server flattens or rejects it,
    page pairing falls back to one text block with inline `<page N of M>` markers
    — which fixes the split-table problem but not the image/text correspondence.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    messages = [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": [
            {"type": "image", "image": "page_1.png"},
            {"type": "text", "text": "<page 1 of 2>\n\nfirst page"},
            {"type": "image", "image": "page_2.png"},
            {"type": "text", "text": "<page 2 of 2>\n\nsecond page"},
        ]},
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    r.data["rendered_chars"] = len(rendered)
    r.data["image_placeholders"] = rendered.count("<|image_pad|>") or rendered.count("<image>")
    # Both pages' text must survive, and in order.
    first, second = rendered.find("first page"), rendered.find("second page")
    r.ok = first >= 0 and second > first
    r.detail = (
        f"interleaved content preserved in order ({r.data['image_placeholders']} image slots)"
        if r.ok else "the template dropped or reordered interleaved blocks — see the fallback above"
    )
    r.data["sample"] = rendered[:400]


def check_swift_row_format(r: Result) -> None:
    """Does ms-swift render a staged row exactly as serving renders the corpus row?

    ``training.stage_data`` converts each corpus row — a content LIST of image and
    text blocks — into ms-swift's format: string content with ``<image>``
    placeholders, joined with no separator, plus an ``images`` list. That is only
    safe if the two render to the same tokens. A difference here is prompt drift
    between training and serving, which degrades a fine-tuned model and shows up
    in no training metric.

    Uses the ms-swift 3 template API (``get_model_tokenizer`` / ``get_template``).
    Compares TOKEN IDS, not decoded text: the prompt's ids up to the assistant
    header, and each image's visual-token count, against the model's processor
    tokenizing the same messages and pages (``training.parity``). Two renders
    that decode alike can still differ by a special token.
    """
    import tempfile

    from PIL import Image
    from swift.llm import get_model_tokenizer, get_template
    from transformers import AutoProcessor

    from training.stage_data import to_swift_row

    tmp = Path(tempfile.mkdtemp())
    for name in ("page_1.png", "page_2.png"):
        Image.new("RGB", (448, 448), "white").save(tmp / name)
    original = {"source_id": "spike", "messages": [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": [
            {"type": "image", "image": "page_1.png"},
            {"type": "text", "text": "<page 1 of 2>\n\nfirst page"},
            {"type": "image", "image": "page_2.png"},
            {"type": "text", "text": "<page 2 of 2>\n\nsecond page"},
        ]},
        {"role": "assistant", "content": '{"policy": {}}'},
    ]}
    row = to_swift_row(original, lambda key: str(tmp / key))

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    _model, swift_processor = get_model_tokenizer(MODEL_ID, load_model=False)
    template = get_template(swift_processor.model_meta.template, swift_processor)
    # Training mode, as `swift sft` sets it: the default (inference) mode stops at
    # the assistant header and leaves out the response the row trains on.
    if hasattr(template, "set_mode"):
        template.set_mode("train")
    else:
        template.mode = "train"
    encoded = template.encode({"messages": row["messages"], "images": row["images"]})

    from training.parity import ASSISTANT_HEADER, IMAGE_PAD, flat_ids, image_pad_runs, prompt_ids

    tokenizer = swift_processor.tokenizer
    header = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
    pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    trained = prompt_ids(encoded["input_ids"], header)
    # The same messages as serving sends them: the pages at their local paths,
    # no answer, and the generation header the model answers after.
    serving_messages = [
        {**m, "content": [
            {**b, "image": str(tmp / b["image"])} if b.get("type") == "image" else b
            for b in m["content"]
        ]} if isinstance(m["content"], list) else m
        for m in original["messages"] if m["role"] != "assistant"
    ]
    try:
        served = flat_ids(processor.apply_chat_template(
            serving_messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        )["input_ids"])
    except Exception as exc:  # noqa: BLE001 - this transformers cannot tokenize pages in a template
        r.ok = None
        r.detail = f"the processor could not tokenize the pages in the chat template: {exc}"
        return

    r.data["images"] = len(row["images"])
    r.data["image_tokens_trained"] = image_pad_runs(trained, pad)
    r.data["image_tokens_served"] = image_pad_runs(served, pad)
    r.ok = trained == served
    r.detail = (
        "a staged row tokenizes identically to the serving prompt" if r.ok else
        "ms-swift tokenizes the staged row DIFFERENTLY from serving — prompt drift; see sample"
    )
    if not r.ok:
        first = next((i for i, (a, b) in enumerate(zip(trained, served, strict=False)) if a != b),
                     min(len(trained), len(served)))
        r.data["first_difference_at"] = first
        r.data["trained"] = tokenizer.decode(trained[max(0, first - 20):first + 20])
        r.data["served"] = tokenizer.decode(served[max(0, first - 20):first + 20])


def check_swift_image_budget(r: Result) -> None:
    """Does ms-swift resize pages to the budget training passes it?

    training.train passes the budget in BOTH forms — MAX_PIXELS/MIN_PIXELS for
    Qwen2/2.5-VL, IMAGE_MAX_TOKEN_NUM/IMAGE_MIN_TOKEN_NUM for Qwen3-VL — because
    which one this ms-swift reads for this model is not known until here. If the
    processor honours neither, it resizes to its own default, and the model trains
    on pages a different size from the ones it is served.

    Measured in a FRESH process with the budget in its environment from the
    start, exactly as ``swift sft`` is launched. ms-swift reads these settings
    once, when its template is first built, and keeps them: measured in this
    process, after another check has built a template without them, it reports
    its own default (16384 tokens) whatever the budget says.
    """
    import os
    import textwrap

    from common.scopes import default_scope
    from training.length_check import image_tokens
    from training.train import _pixel_budget

    env = _pixel_budget(default_scope())
    code = textwrap.dedent(f'''
        import tempfile
        from pathlib import Path
        from PIL import Image
        from swift.llm import get_model_tokenizer, get_template
        page = Path(tempfile.mkdtemp()) / 'page.png'
        Image.new('RGB', (1700, 2200), 'white').save(page)
        _m, processor = get_model_tokenizer({MODEL_ID!r}, load_model=False)
        template = get_template(processor.model_meta.template, processor)
        encoded = template.encode({{'messages': [{{'role': 'user', 'content': '<image>page'}},
            {{'role': 'assistant', 'content': '{{}}'}}], 'images': [str(page)]}})
        pad = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        print('MEASURED', sum(1 for t in encoded['input_ids'] if t == pad))
    ''')
    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env},
                         capture_output=True, text=True, timeout=900)
    line = next((row for row in out.stdout.splitlines() if row.startswith("MEASURED ")), None)
    if line is None:
        raise RuntimeError(f"the measuring process failed (exit {out.returncode}): {out.stderr[-600:]}")
    measured = int(line.split()[1])
    predicted = image_tokens(
        1700, 2200, min_pixels=int(env["MIN_PIXELS"]), max_pixels=int(env["MAX_PIXELS"])
    )

    r.data.update({"env": env, "measured_image_tokens": measured, "predicted": predicted})
    r.ok = measured == predicted
    r.detail = (
        f"a letter page is {measured} image tokens, as the length check predicts"
        if r.ok else
        f"ms-swift produced {measured} image tokens where {predicted} were predicted: the "
        "pixel budget is not being honoured (or the resize factor differs) — training would "
        "see a different resolution from serving, and the pre-launch length check is off"
    )


def check_vllm_multi_lora(r: Result) -> None:
    """Does vLLM support multi-LoRA for this architecture?

    Checked by capability, not by serving: loading an 8B model twice is not worth
    a pod-hour when the question is whether the feature exists for this model class.
    """
    import vllm
    from vllm.lora.request import LoRARequest  # noqa: F401

    r.data["vllm"] = vllm.__version__
    from vllm import EngineArgs

    fields = set(getattr(EngineArgs, "__dataclass_fields__", {}))
    r.data["enable_lora"] = "enable_lora" in fields
    r.data["max_loras"] = "max_loras" in fields
    r.ok = r.data["enable_lora"] and r.data["max_loras"]
    r.detail = (
        f"vLLM {vllm.__version__} exposes enable_lora/max_loras — confirm hot-swap on a real "
        "adapter before relying on it" if r.ok
        else f"vLLM {vllm.__version__} lacks multi-LoRA args; serve merged per-type models (IMPL-11)"
    )


def _mineru(pdf: Path) -> tuple[str, float]:
    """Run MinerU once on GPU. Returns (markdown, seconds).

    No device parameter: OCR is GPU-only, because MinerU's CPU path uses lighter
    model variants and produces different markdown from the same PDF (arch §8a).
    """
    from data_pipeline.ocr.run_mineru import MinerUEngine

    started = time.perf_counter()
    pages = MinerUEngine().process(pdf.read_bytes(), device="cuda", max_long_side_px=1792)
    return "\n\n".join(p.markdown for p in pages), time.perf_counter() - started


def check_mineru_gpu(pdf: Path):
    def _check(r: Result) -> None:
        markdown, seconds = _mineru(pdf)
        r.ok = bool(markdown.strip())
        r.data["seconds"] = round(seconds, 1)
        r.data["chars"] = len(markdown)
        r.detail = f"{len(markdown)} chars in {seconds:.1f}s on cuda"
    return _check


def check_llama_cpp_mmproj(r: Result) -> None:
    """Can llama.cpp export a vision projector for this architecture?

    Without a working mmproj a GGUF loads and cannot see, failing silently on
    every image-only document. Checked by inspecting the converter's supported
    architectures rather than by running a 16GB conversion.
    """
    import shutil

    # Bounded search. Globbing from the filesystem root walks every mounted
    # volume and dies on the first unreadable directory — which is what it did.
    converter = shutil.which("convert_hf_to_gguf.py")
    if converter is None:
        for root in (Path.cwd(), Path.home(), Path("/workspace"), Path("/opt"), Path("/usr/local")):
            try:
                found = next(root.glob("*/convert_hf_to_gguf.py"), None) or next(
                    root.glob("*/*/convert_hf_to_gguf.py"), None
                )
            except (OSError, PermissionError):
                continue
            if found:
                converter = str(found)
                break
    if converter is None:
        r.ok = None
        r.detail = (
            "llama.cpp not found (searched PATH, cwd, home, /workspace, /opt, /usr/local) — "
            "clone it to answer this, or leave GGUF deferred"
        )
        return

    text = Path(converter).read_text(encoding="utf-8", errors="replace").lower()
    r.data["mentions_qwen3vl"] = "qwen3vl" in text or "qwen3_vl" in text
    r.data["mentions_mmproj"] = "mmproj" in text
    r.ok = r.data["mentions_qwen3vl"] and r.data["mentions_mmproj"]
    r.detail = (
        "the converter references Qwen3-VL and mmproj — attempt a real export next"
        if r.ok else
        "no Qwen3-VL + mmproj support found; GGUF stays deferred and vLLM on the merged "
        "model remains the serving path (arch §13a)"
    )


def check_merger_module_names(r: Result) -> None:
    """The exact module names the freeze flags and any vision LoRA must reference.

    arch v2.1 §9a specifies ``freeze_vit`` and ``freeze_aligner`` and names a patch
    merger plus DeepStack mergers. A wrong name does not raise — PEFT attaches LoRA
    to nothing and training proceeds silently without the target, and a freeze flag
    that matches nothing silently trains what it was meant to hold still.

    Read from the safetensors index rather than by instantiating the model: the
    parameter names are what matters and the index is a single small file.
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    local_index = Path(MODEL_ID) / "model.safetensors.index.json"
    index = local_index if local_index.is_file() else hf_hub_download(
        HUB_MODEL_ID, "model.safetensors.index.json"
    )
    keys = list(_json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"])
    r.data["parameter_count"] = len(keys)

    def prefixes(needle: str) -> list[str]:
        found = set()
        for k in keys:
            if needle in k:
                parts = k.split(".")
                hit = next(i for i, p in enumerate(parts) if needle in p)
                found.add(".".join(parts[: hit + 1]))
        return sorted(found)

    r.data["merger_modules"] = prefixes("merger")
    r.data["aligner_modules"] = prefixes("aligner")
    r.data["visual_roots"] = sorted({k.split(".")[0] for k in keys if k.startswith(("visual", "vision"))})
    r.data["decoder_targets_present"] = sorted(
        t for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        if any(f".{t}." in k for k in keys)
    )

    connector = r.data["merger_modules"] + r.data["aligner_modules"]
    r.ok = bool(connector) and len(r.data["decoder_targets_present"]) == 7
    r.detail = (
        f"connector modules {connector}; visual roots {r.data['visual_roots']}; "
        f"all 7 decoder targets present"
        if r.ok else
        f"connector={connector or 'NONE FOUND'} decoder_targets="
        f"{r.data['decoder_targets_present']} — do not ship a freeze flag or LoRA target "
        "naming a module that is not in this list"
    )


def check_visual_token_geometry(r: Result) -> None:
    """How many visual tokens a page actually costs, measured not assumed.

    Every sequence cap in arch §7a is derived from this number, and the v1 caps
    were set from an assumed patch geometry. Measured through the real image
    processor, so patch size, spatial merge and any smart-resize rounding are all
    included rather than modelled.
    """
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", processor)
    merge = int(getattr(image_processor, "merge_size", 2))
    patch = int(getattr(image_processor, "patch_size", 14))
    r.data["patch_size"] = patch
    r.data["merge_size"] = merge
    r.data["px_per_token"] = patch * merge

    # US Letter at three candidate pixel budgets, portrait.
    measured = {}
    for label, max_pixels in (("0.26M_thumbnail", 262_144), ("1.64M", 1_638_400),
                              ("2.48M", 2_483_712), ("3.24M", 3_240_000)):
        image = Image.new("RGB", (1384, 1792), "white")
        out = image_processor(images=image, max_pixels=max_pixels, return_tensors="pt")
        grid = out["image_grid_thw"][0].tolist()          # [t, h, w] in patches
        tokens = (grid[0] * grid[1] * grid[2]) // (merge * merge)
        measured[label] = {"grid_thw": grid, "tokens_per_page": tokens}
    r.data["per_page"] = measured

    extract = measured["2.48M"]["tokens_per_page"]
    r.data["three_page_lossrun_tokens"] = extract * 3
    r.ok = extract > 0
    r.detail = (
        f"{patch}px patches x{merge} merge = {patch * merge}px per token; "
        f"at 2.48M max_pixels a US-Letter page costs {extract:,} visual tokens "
        f"({extract * 3:,} for three). Set the §7a caps from this table."
    )


def check_peak_vram_per_cap(r: Result) -> None:
    """Peak allocation over one forward/backward at each task cap (arch §9.3).

    The v1 pod-class table was arithmetic. This replaces it with a measurement, and
    it is the number that decides whether a cap fits on one 80GB card.
    """
    import torch

    if not torch.cuda.is_available():
        r.ok = None
        r.detail = "no CUDA device — cannot measure"
        return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText

    # The whole model - vision tower included - with the LoRA training attaches,
    # not the text tower alone: Qwen3-VL's text config is not a causal-LM
    # config, and the base alone would count gradients LoRA never keeps.
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    with torch.device("cuda"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    measured = {}
    for cap in (4_096, 12_288, 20_480, 24_576, 32_768):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            ids = torch.randint(0, 1000, (1, cap), device="cuda")
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            measured[cap] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
            model.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError:
            measured[cap] = "OOM"
        except Exception as exc:  # noqa: BLE001
            measured[cap] = f"error: {type(exc).__name__}"
    r.data["peak_gb_by_cap"] = measured
    r.data["note"] = (
        "the whole model with rank-64 LoRA and gradient checkpointing, text tokens only, and "
        "full-vocabulary logits (training's use_logits_to_keep saves most of those) - an upper "
        "bound for a row of that length. The smoke run measures the real footprint."
    )
    r.ok = any(isinstance(v, float) for v in measured.values())
    r.detail = f"peak GB by sequence cap: {measured}"


def check_sequence_parallel(r: Result) -> None:
    """ms-swift sequence parallelism — the escape hatch before any ZeRO-3 config."""
    import dataclasses

    # ms-swift's own argument dataclass - what its parser accepts - not a grep of
    # `swift sft --help`, which does not print every inherited argument.
    try:
        from swift.llm import TrainArguments
    except ImportError as exc:
        r.ok = False
        r.detail = f"not ms-swift 3 (no swift.llm.TrainArguments: {exc})"
        return
    known = {f.name for f in dataclasses.fields(TrainArguments)}
    flags = {name: name in known for name in
             ("sequence_parallel_size", "use_logits_to_keep", "padding_free", "freeze_aligner")}
    r.data["flags"] = flags
    missing = [k for k, v in flags.items() if not v]
    r.ok = not missing
    r.detail = (
        "ms-swift exposes sequence parallelism, use_logits_to_keep, padding-free and freeze_aligner"
        if r.ok else f"ms-swift does not expose {missing} — arch §9.3 and §11.1 depend on these"
    )


def check_structured_outputs(r: Result) -> None:
    """vLLM structured decoding against a Fideon SPEC_00 schema (arch §13)."""
    import vllm

    r.data["vllm"] = vllm.__version__
    try:
        from vllm.sampling_params import GuidedDecodingParams  # noqa: F401
        r.data["guided_decoding_params"] = True
    except ImportError:
        r.data["guided_decoding_params"] = False
    from vllm import SamplingParams

    fields = _struct_fields(SamplingParams)
    r.data["sampling_fields"] = sorted(f for f in fields if "guided" in f or "structur" in f)
    r.ok = r.data["guided_decoding_params"] or bool(r.data["sampling_fields"])
    r.detail = (
        f"structured outputs available ({r.data['sampling_fields'] or 'GuidedDecodingParams'}) — "
        "measure schema compile time and throughput on the real Fideon SPEC_00 schemas next"
        if r.ok else
        "no structured-output surface found; schema validity cannot be guaranteed at decode time"
    )


def check_raw_logprobs(r: Result) -> None:
    """``logprobs_mode: raw_logprobs`` must survive structured decoding (arch §5.1).

    Constrained decoding masks invalid tokens. If the returned logprobs are the
    post-mask distribution, every confidence feature is computed on a distribution
    the model did not produce, and the calibrator is fitted on an artefact.
    """
    from vllm import SamplingParams

    fields = _struct_fields(SamplingParams)
    r.data["has_logprobs"] = "logprobs" in fields
    try:
        from vllm.config import ModelConfig

        model_fields = set(getattr(ModelConfig, "__dataclass_fields__", {}))
        r.data["logprobs_mode_supported"] = "logprobs_mode" in model_fields
    except Exception:  # noqa: BLE001
        r.data["logprobs_mode_supported"] = False
    r.ok = r.data["has_logprobs"]
    r.detail = (
        f"logprobs available; logprobs_mode field {'present' if r.data['logprobs_mode_supported'] else 'ABSENT'} "
        "— confirm on a live server that values are pre-mask before fitting any calibrator"
    )


def check_fp8_export(r: Result) -> None:
    """llm-compressor FP8 with the vision tower and lm_head excluded (arch §13a)."""
    try:
        import llmcompressor

        r.data["llmcompressor"] = getattr(llmcompressor, "__version__", "installed")
    except ImportError:
        # Deferred, not failed: llmcompressor has its own environment
        # (requirements-quantize), and cycle 1 serves bf16. Run this check there.
        r.ok = None
        r.detail = ("deferred: llmcompressor lives in its own environment (requirements-quantize.txt) "
                    "and cycle 1 serves bf16 - run the spike there before the first FP8 export")
        return
    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: F401

        r.data["quantization_modifier"] = True
    except ImportError:
        r.data["quantization_modifier"] = False
    r.ok = r.data["quantization_modifier"]
    r.detail = (
        "llm-compressor exposes QuantizationModifier — run a real FP8 export with "
        "ignore=['visual.*','lm_head'] and load it in vLLM next"
        if r.ok else "llm-compressor is installed but QuantizationModifier is missing"
    )


def check_mineru_determinism(pdf: Path):
    """Same PDF twice on the same GPU must give byte-identical markdown (arch §8a).

    The model learns how MinerU formats its output, so nondeterminism is training
    noise that no seed controls and no manifest records.
    """

    def _check(r: Result) -> None:
        first, first_seconds = _mineru(pdf)
        second, _ = _mineru(pdf)
        r.data["chars"] = len(first)
        r.data["seconds_first_run"] = round(first_seconds, 1)
        r.data["identical"] = first == second
        r.ok = r.data["identical"]
        r.detail = (
            f"two runs produced identical markdown ({len(first):,} chars)"
            if r.ok else
            "TWO RUNS DIFFERED — the OCR pin in the corpus manifest does not make the corpus "
            "reproducible; find the nondeterminism before building a corpus"
        )

    return _check


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def render(results: list[Result]) -> str:
    lines = [
        "PHASE 0 DEPENDENCY SPIKE",
        f"  {datetime.now(UTC).isoformat()}  ·  python {platform.python_version()}  ·  {platform.platform()}",
        "",
    ]
    for r in results:
        lines.append(f"[{r.status:7}] {r.name}")
        lines.append(f"          {r.detail}")
        if r.decides:
            lines.append(f"          decides: {r.decides}")
    failed = [r for r in results if r.ok is False]
    unknown = [r for r in results if r.ok is None]

    lines += ["", f"{len(results) - len(failed) - len(unknown)} passed, "
                  f"{len(failed)} failed, {len(unknown)} undetermined."]
    blocking = [r for r in failed if r.name == "cuda_available"]
    recoverable = [r for r in failed if r not in blocking]
    if blocking:
        lines += ["", "BLOCKING — nothing below this can be trusted:"]
        lines += [f"  - {r.name}: {r.detail}" for r in blocking]
    if recoverable:
        lines += ["", "Each of these has a documented fallback — none stops the project:"]
        lines += [f"  - {r.name}: {r.decides}" for r in recoverable]
    if unknown:
        lines.append("")
        lines.append("Undetermined is not a pass. These still need an answer:")
        lines += [f"  - {r.name}" for r in unknown]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 dependency spike")
    parser.add_argument("--pdf", type=Path, help="a real insurance PDF for the MinerU checks")
    parser.add_argument("--out", type=Path, default=Path("spike_report.json"))
    parser.add_argument("--skip-mineru", action="store_true")
    parser.add_argument("--only-mineru", action="store_true",
                        help="run only the MinerU checks: for the OCR environment, which has "
                             "MinerU but not vLLM or ms-swift")
    args = parser.parse_args(argv)
    if args.only_mineru and (args.skip_mineru or not args.pdf):
        parser.error("--only-mineru needs --pdf and cannot be combined with --skip-mineru")
    # On the pod, run detached in tmux: a closed laptop must not stop the spike.
    try:
        from orchestration.detach import detach_script_if_needed
    except ImportError:   # run from outside the repo: nothing to detach through
        detach_script_if_needed = None
    if detach_script_if_needed and detach_script_if_needed(__file__, argv, "spike"):
        return 0

    print("Phase 0 spike — every check is independent; none aborts the run.\n")
    results = [] if args.only_mineru else [
        run("cuda_available", "whether anything below can run", check_gpu),
        run("flash_attn", "throughput; sdpa is the fallback", check_flash_attn),
        run("ms_swift_arguments", "the Layer-3 entrypoint; TRL SFTTrainer is the fallback (arch §10)",
            check_ms_swift),
        run("model_config_resolves", "the pinned model id and revision", check_model_loads),
        run("interleaved_image_text", "the corpus row format; single-block + markers is the fallback",
            check_interleaved_content),
        run("swift_row_format", "that a staged training row renders exactly as serving renders it",
            check_swift_row_format),
        run("swift_image_budget", "that training resizes pages to the budget serving uses, and "
            "that the pre-launch length check counts image tokens right",
            check_swift_image_budget),
        run("vllm_multi_lora", "IMPL-11 serving; merged per-type models are the fallback",
            check_vllm_multi_lora),
        run("llama_cpp_mmproj", "whether GGUF export can produce a model that can see",
            check_llama_cpp_mmproj),
        # --- arch v2.1 §16.0 additions ---------------------------------------
        run("merger_module_names", "the freeze flags and any vision LoRA target (§9a)",
            check_merger_module_names),
        run("visual_token_geometry", "every sequence cap in §7a", check_visual_token_geometry),
        run("peak_vram_per_cap", "the pod-class table in §14 and whether 32k fits on one card (§9.3)",
            check_peak_vram_per_cap),
        run("sequence_parallel_and_memory_flags", "the §9.3 escape hatch and the §11.1 memory settings",
            check_sequence_parallel),
        run("vllm_structured_outputs", "whether schema validity is guaranteed at decode time (§13)",
            check_structured_outputs),
        run("raw_logprobs_under_constraint", "whether confidence features are computed on the "
            "model's own distribution (§5.1)", check_raw_logprobs),
        run("fp8_export", "the v2.1 default serving format (§13a)", check_fp8_export),
    ]

    if args.pdf and not args.skip_mineru:
        results.append(run("mineru_cuda", "the whole data pipeline", check_mineru_gpu(args.pdf)))
        results.append(run("mineru_determinism", "whether the OCR pin makes the corpus reproducible (§8a)",
                           check_mineru_determinism(args.pdf)))
        # No device-parity check: OCR is GPU-only, so there is no second device
        # whose output could differ. What was a measurement is now a constraint.
    else:
        print("  [SKIPPED] MinerU checks — pass --pdf to answer the ocr_device pin question")

    report = render(results)
    print("\n" + report)

    args.out.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": [asdict(r) | {"status": r.status} for r in results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # Non-zero only when a check genuinely failed. An undetermined check is
    # reported loudly but does not fail the run, because the usual cause is a
    # tool that is simply not installed on this pod yet.
    return 1 if any(r.ok is False for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
