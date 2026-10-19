"""Training and serving must agree (arch v2.1 §10.2).

The three checks that make "train what you serve" a fact rather than an
intention. Every other test in this suite verifies one side of a boundary; these
run **both sides** and compare.

They matter because the failure they catch is silent. If ms-swift's chat template
and vLLM's processor tokenize the same example differently — one extra BOS token,
a different image-token count, a system prompt joined with a different separator
— nothing raises. The model simply performs worse in production than it did in
evaluation, and the gap looks like ordinary generalisation loss.

**Two of these need a GPU and the pinned stack**, so they are marked ``gpu`` and
skipped in CI. That is a real limitation, not a formality: until they run on a
pod, train/serve parity is asserted by construction rather than measured. The
third — prompt parity — is pure and runs everywhere, because the prompt renderer
is the one piece both sides share in this repo today.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: A staged ms-swift epoch file on the pod (``/workspace/.../epoch_1.jsonl``):
#: when set, the token parity test reads its first rows - real pages, real
#: prompts - instead of the fixed examples it builds.
PARITY_CORPUS_ENV = "FIDEON_PARITY_CORPUS"

#: Ten fixed examples, per §10.2. Fixed rather than sampled so a failure is
#: reproducible and a fix is verifiable against the same inputs.
PARITY_EXAMPLE_COUNT = 10

#: Field values must match on at least this many. Not all ten, because bf16
#: kernels differ slightly between HF and vLLM and one document occasionally
#: lands on the other side of a rounding boundary.
MIN_MATCHING_EXAMPLES = 9


# --------------------------------------------------------------------------
# Prompt parity — pure, runs everywhere
# --------------------------------------------------------------------------

def test_the_serving_prompt_is_the_corpus_prompt():
    """One renderer, one source of truth (arch v2.1 §7.3).

    v1 had three prompt texts and two prompt homes. A corpus built against one
    and served against another trains a model to expect instructions it never
    receives — and the symptom is a quieter model, not an error.
    """
    from common.prompts import render_system_prompt

    for doc_type, acord_form in (("policy", None), ("lossrun", None), ("acord", "25")):
        for mode in ("ocr_plus_image", "image_only"):
            rendered = render_system_prompt(doc_type, mode, acord_form)
            assert rendered.strip(), f"{doc_type}/{mode} rendered an empty prompt"
            # The schema is injected, not described: the model is given the
            # exact structure it must produce.
            assert "{" in rendered and "}" in rendered


def test_the_reference_prompts_match_the_renderer():
    """The snapshots under testing/prompts are generated, never hand-edited.

    A hand-edited snapshot is a prompt that exists in the repo, is read by a
    human reviewing behaviour, and is not the one the model ever saw.
    """
    from testing.render_prompts import drifted

    assert not drifted(), (
        "reference prompts no longer match common.prompts. Regenerate with "
        "`python -m testing.render_prompts --write` — never hand-edit them."
    )


def test_noisy_ocr_is_not_announced_in_the_prompt():
    """The noise lives in the data, not the instruction (arch §6).

    A distinct prompt for noisy OCR would teach the model that bad OCR is
    announced — and at inference nothing announces it.
    """
    from common.prompts import render_system_prompt

    assert (
        render_system_prompt("policy", "ocr_plus_image")
        == render_system_prompt("policy", "noisy_ocr_image")
    )


def test_the_prompt_version_is_recorded_so_drift_is_attributable():
    """A corpus manifest records the prompt version it was built with (§8.1). A
    release records the version it serves with (§12.3). Comparing them is what
    turns "the model got worse" into "the prompt changed on the 14th"."""
    from common.prompts import prompt_versions

    for doc_type, acord_form in (("policy", None), ("acord", "25")):
        versions = prompt_versions(doc_type, acord_form)
        assert versions["prompt_template_version"], f"{doc_type} records no prompt version"
        # Schema and prompt travel together: a schema change with an unchanged
        # prompt version is a corpus nobody can attribute a regression to.
        assert versions.get("schema_version")


# --------------------------------------------------------------------------
# Token parity — needs the pinned stack
# --------------------------------------------------------------------------

def _pinned_stack_or_skip() -> Path:
    """The base model's directory, or a skip naming what is missing.

    Runs only on the pod: a CUDA GPU, the pinned base weights on disk, and the
    training environment's ms-swift and vLLM.
    """
    import importlib.util

    try:
        import torch
    except ImportError:
        pytest.skip("needs torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from common.config import base_model_dir

    model_dir = base_model_dir()
    if model_dir is None:
        pytest.skip("needs the pinned base model on disk (configs/base_model.yaml model.local_dir)")
    missing = [name for name in ("swift", "vllm") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"needs {missing}: run it in the training environment")
    return model_dir


#: Page sizes the fixed examples are drawn at: a letter page at the 1792px cap
#: both ways, and smaller renders.
_PAGE_SIZES = ((1386, 1792), (1792, 1386), (1275, 1650), (1224, 1584), (850, 1100))


def _parity_rows(tmp_path: Path) -> list[dict]:
    """PARITY_EXAMPLE_COUNT staged (ms-swift) rows, fixed so a failure reproduces.

    The first rows of ``$FIDEON_PARITY_CORPUS`` when set; otherwise examples
    built with the shared prompt renderer - one to three pages, both modes.
    """
    corpus = os.environ.get(PARITY_CORPUS_ENV)
    if corpus:
        rows: list[dict] = []
        with open(corpus, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) == PARITY_EXAMPLE_COUNT:
                    break
        return rows

    from PIL import Image

    from inference_core.input_builder import build_messages
    from training.stage_data import to_swift_row

    rows = []
    for i in range(PARITY_EXAMPLE_COUNT):
        paths = []
        for page in range(1 + i % 3):
            width, height = _PAGE_SIZES[(i + page) % len(_PAGE_SIZES)]
            path = tmp_path / f"example_{i}_page_{page + 1}.png"
            Image.new("RGB", (width, height), "white").save(path)
            paths.append(str(path))
        mode = "image_only" if i % 4 == 3 else "ocr_plus_image"
        text = (f"POLICY NUMBER GL-{1000 + i}\n| Coverage | Limit |\n|---|---|\n"
                "| Each Occurrence | $1,000,000 |")
        built = build_messages(
            "policy", paths, None if mode == "image_only" else [text] * len(paths), mode,
            lob="gl", assistant_content=json.dumps({"policy": {"policy_number": f"GL-{1000 + i}"}}),
        )
        rows.append(to_swift_row({"source_id": f"parity_{i}", "messages": built.messages},
                                 lambda key: key))
    return rows


@pytest.mark.gpu
def test_template_token_parity(tmp_path, record_property):  # pragma: no cover - needs the pod
    """ms-swift's template and vLLM's processor must produce IDENTICAL prompt
    token IDs and identical image-token counts for the same example.

    This is the check that makes train/serve parity a measurement. A single
    extra special token shifts every position the model learned; a different
    image-token count means the picture it sees at serving time is not the one it
    trained on. Neither raises.

    It also measures the pixels one visual token covers, from the model's own
    processor, and checks every page's token count against the resize rule the
    sequence caps are built on (``training.length_check.image_tokens``).
    """
    model_dir = _pinned_stack_or_skip()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from common.config import pixel_budget, serving_config
    from data_pipeline.dataset_builder.cap_check import PIXELS_PER_VISUAL_TOKEN
    from inference_core.model_runner import to_vllm_messages
    from training.length_check import RESIZE_FACTOR, image_tokens, png_size
    from training.parity import (
        ASSISTANT_HEADER,
        IMAGE_PAD,
        from_swift_row,
        image_pad_runs,
        prompt_ids,
    )
    from training.train import _masking_encoder, build_training_config

    rows = _parity_rows(tmp_path)
    assert len(rows) == PARITY_EXAMPLE_COUNT

    processor = AutoProcessor.from_pretrained(str(model_dir))
    image_processor = processor.image_processor
    measured = (int(image_processor.patch_size) * int(image_processor.merge_size)) ** 2
    record_property("pixels_per_visual_token", measured)
    assert measured == PIXELS_PER_VISUAL_TOKEN == RESIZE_FACTOR ** 2, (
        f"the processor covers {measured} pixels per visual token; cap_check assumes "
        f"{PIXELS_PER_VISUAL_TOKEN}, and every sequence cap derives from it"
    )
    tokenizer = processor.tokenizer
    header = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
    pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD)

    # The trainer's own encoder: ms-swift's template, with the pixel budget env.
    swift, _recorded = build_training_config(
        corpus_paths=[str(tmp_path / "parity.jsonl")] * 16, output_dir=str(tmp_path / "out"))
    encode, _header, _end, _suffix = _masking_encoder(swift)

    # The engine settings serving builds with that decide the prompt's tokens:
    # the weights and the processor's pixel budget (inference_core.model_runner).
    min_pixels, max_pixels = pixel_budget()
    engine = LLM(
        model=str(model_dir),
        mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
        max_model_len=int(serving_config()["sequence"]["max_seq_len"]),
        gpu_memory_utilization=0.85, seed=0,
    )
    once = SamplingParams(max_tokens=1, temperature=0.0)

    for index, row in enumerate(rows):
        trained = prompt_ids(encode(row)["input_ids"], header)
        prompt = [m for m in from_swift_row(row) if m["role"] != "assistant"]
        served = [int(t) for t in engine.chat(to_vllm_messages(prompt), once)[0].prompt_token_ids]
        assert trained == served, f"row {index}: training and serving tokenize the prompt differently"
        expected = [
            image_tokens(*png_size(image), min_pixels=min_pixels, max_pixels=max_pixels)
            for image in row["images"]
        ]
        assert image_pad_runs(trained, pad) == image_pad_runs(served, pad) == expected, (
            f"row {index}: image-token counts differ (trained, served, resize rule)"
        )


@pytest.mark.gpu
def test_serving_output_parity():  # pragma: no cover - needs a GPU
    """HF and vLLM must extract the same VALUES from the same ten documents.

    Parsed and normalized, not compared as text: bf16 kernels differ slightly
    between the two runtimes, so byte equality would fail on a correct pair. What
    must hold is that the extracted field values agree on at least nine of ten,
    and that no single example differs in more than one field — a broad, shallow
    disagreement is rounding, while a deep one on a single document is a bug.
    """
    pytest.skip(
        f"Phase 8 GPU milestone. Generate over {PARITY_EXAMPLE_COUNT} fixed examples "
        "through the HF backend and through vLLM on the merged model, normalize both "
        f"with common.normalize, and assert agreement on >= {MIN_MATCHING_EXAMPLES} "
        "with no example differing in more than one field (arch v2.1 §10.2)."
    )


# --------------------------------------------------------------------------
# The contract these tests are about, checked without a GPU
# --------------------------------------------------------------------------

def test_training_and_serving_read_the_same_vision_settings():
    """The settings that decide what the model sees come from ONE file.

    Two files holding a resolution cap is how a model gets trained at one
    resolution and served at another — and nothing downstream reports that as
    anything but degraded accuracy (arch v2.1 §7a).
    """
    from common.config import SHARED_VISION_CONFIG, vision_for_task
    from common.tasks import Task

    assert SHARED_VISION_CONFIG.exists()
    for task in Task:
        budget = vision_for_task(str(task))
        assert budget["max_pixels"] > 0
        assert budget["min_pixels"] < budget["max_pixels"], (
            f"{task}: min_pixels must be below max_pixels, or every page is upscaled"
        )


def test_training_and_serving_read_the_same_sequence_caps():
    from common.config import SHARED_SEQUENCE_CONFIG, sequence_for_task
    from common.tasks import Task

    assert SHARED_SEQUENCE_CONFIG.exists()
    for task in Task:
        budget = sequence_for_task(str(task))
        assert budget["max_output_tokens"] < budget["max_seq_len"]


def test_the_trainer_max_length_covers_every_task_the_corpus_holds():
    """ms-swift takes ONE max_length and the corpus interleaves every task, so
    anything below the largest cap truncates the longest one rather than
    rejecting it — and a clipped assistant span trains the model to stop early."""
    from common.config import seq_cap_for_task, shared_sequence_config
    from common.tasks import Task
    from training.train import corpus_max_length

    largest = corpus_max_length()
    declared = shared_sequence_config().get("tasks", {})
    for task in Task:
        name = str(task)
        assert seq_cap_for_task(name) <= largest
        for doc_type in (declared.get(name, {}).get("by_doc_type") or {}):
            assert seq_cap_for_task(name, doc_type) <= largest, (
                f"{name}/{doc_type} exceeds the trainer's max_length"
            )


def test_the_serving_config_and_the_shared_caps_agree():
    """A serving endpoint configured below a task's cap truncates exactly the
    documents the cap was raised for."""
    from common.config import serving_config
    from training.train import corpus_max_length

    serving_cap = int(serving_config()["sequence"]["max_seq_len"])
    # The trainer's own number, including per-doc-type overrides. Recomputing it
    # here would be a second definition that could disagree with the first.
    largest = corpus_max_length()
    assert serving_cap >= largest, (
        f"serving is capped at {serving_cap} while the largest task cap is {largest}; "
        "a routed policy would be truncated at inference only"
    )


def test_the_gpu_parity_tests_are_registered_rather_than_forgotten():
    """A skipped test that nobody remembers is an unchecked assumption with
    extra steps. This asserts they exist and are marked, so `pytest -m gpu` on a
    pod runs them rather than finding nothing."""
    import inspect

    source = inspect.getsource(__import__(__name__, fromlist=["_"]))
    for name in ("test_template_token_parity", "test_serving_output_parity"):
        assert f"def {name}" in source
        assert source.index("@pytest.mark.gpu") < source.index(f"def {name}")


def test_the_parity_fixtures_exist_for_the_gpu_run():
    """Ten fixed examples, so a parity failure is reproducible and a fix is
    verifiable against the same inputs."""
    golden = sorted((ROOT / "tests" / "fixtures" / "golden").glob("*.golden.json"))
    assert golden, "no golden fixtures to build parity examples from"
    for path in golden:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # The line of business sits in one of two places depending on the shape.
        # A flat label carries it as a top-level key. A canonical one has no such
        # key — the client's schema records the printed line as
        # `document.line_of_business_as_stated`, which is what the page said, not
        # what the document IS — so the label's own line lives beside it in the
        # metadata sidecar. Either is a line; neither is a fixture that cannot be
        # routed to a schema.
        sidecar = path.with_name(path.name.replace(".golden.json", ".label_metadata.json"))
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        assert "line_of_business" in payload or metadata.get("lob"), (
            f"{path.name} names no line of business, so no schema can be selected for it"
        )


# --------------------------------------------------------------------------
# What the GPU test compares, checked without a GPU
# --------------------------------------------------------------------------


def test_the_prompt_ends_at_the_last_assistant_header():
    from training.parity import ParityError, prompt_ids

    assert prompt_ids([1, 7, 8, 2, 7, 8, 5, 6], [7, 8]) == [1, 7, 8, 2, 7, 8]
    with pytest.raises(ParityError):
        prompt_ids([1, 2, 3], [7, 8])


def test_image_token_counts_are_the_runs_of_the_pad_token():
    from training.parity import image_pad_runs

    assert image_pad_runs([1, 9, 9, 9, 2, 9, 9, 3], 9) == [3, 2]
    assert image_pad_runs([9, 9], 9) == [2]
    assert image_pad_runs([1, 2], 9) == []


def test_a_staged_row_reads_back_as_the_pages_serving_would_send():
    from training.parity import ParityError, flat_ids, from_swift_row
    from training.stage_data import to_swift_row

    original = {"messages": [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": [
            {"type": "image", "image": "p1.png"}, {"type": "text", "text": "<page 1 of 2>"},
            {"type": "image", "image": "p2.png"}, {"type": "text", "text": "<page 2 of 2>"},
        ]},
        {"role": "assistant", "content": "{}"},
    ]}
    staged = to_swift_row(original, lambda key: f"/local/{key}")
    back = from_swift_row(staged)
    assert back[1]["content"][0] == {"type": "image", "image": "/local/p1.png"}
    assert to_swift_row({"messages": back}, lambda key: key) == staged
    with pytest.raises(ParityError):
        from_swift_row({**staged, "images": staged["images"][:1]})
    assert flat_ids([[4, 5]]) == [4, 5]


def test_the_parity_examples_are_built_from_the_shared_renderer(tmp_path):
    """The fixed rows the GPU test encodes, built here without a GPU."""
    rows = _parity_rows(tmp_path)
    assert len(rows) == PARITY_EXAMPLE_COUNT
    for row in rows:
        assert row["images"] and row["messages"][-1]["role"] == "assistant"
