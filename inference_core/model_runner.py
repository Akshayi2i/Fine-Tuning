"""Loading a model version and generating from it — the shared primitive.

Evaluation (IMPL-08), calibration (IMPL-09), serving (IMPL-11) and testing
(IMPL-12) all run the model through here. Building it once, below all four, is
what breaks the dependency cycle they would otherwise form, and what makes
"test == prod" a property of the code rather than a discipline.

Scope is deliberately narrow: **resolve a version, generate, return text and
logprobs.** No confidence calibration, no document-type classification, no
adapter routing — those are higher-level concerns that import this module, never
the other way round.

Backends are swappable (vLLM for serving, HF for local eval, GGUF for the
portable path) behind one interface, so the same calls work everywhere. Logprobs
are mandatory-capable: a backend that cannot return them cannot support the
confidence signal the whole pipeline depends on (arch §5).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry.blob_client import BlobClient
from inference_core.runner_config import Backend, RunnerConfig, load_runner_config
from registry_utils.query_registry import ResolvedModel, resolve_model_version

log = logging.getLogger(__name__)


class ModelRunnerError(RuntimeError):
    """Raised when a model cannot be loaded or generation fails."""


@dataclass
class Generation:
    """One generation, with everything confidence needs."""

    text: str
    tokens: list[str] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    generation_fingerprint: str = ""
    latency_ms: float | None = None
    #: The prompt's length in tokens, page images included, when the backend says.
    prompt_tokens: int | None = None
    #: Milliseconds until the first answer token arrived - time queued plus time
    #: reading the prompt - when the backend streams; the rest is writing.
    first_token_ms: float | None = None

    @property
    def has_logprobs(self) -> bool:
        return bool(self.tokens) and len(self.tokens) == len(self.token_logprobs)

    def assert_logprobs(self) -> None:
        """Fail loudly when logprobs are missing.

        Silently returning an extraction with no confidence would hand the caller
        something that looks like a clean result and cannot be risk-routed.
        """
        if not self.has_logprobs:
            raise ModelRunnerError(
                f"generation returned no usable logprobs ({len(self.tokens)} tokens, "
                f"{len(self.token_logprobs)} logprobs). Per-field confidence is derived from "
                "them (arch §5); without them nothing downstream can route low-confidence "
                "fields to review."
            )

    def truncated(self) -> bool:
        """Whether the answer was cut off: at the token limit, or stopped by the
        backend because it had started repeating itself (:data:`STOPPED_REPEATING`).

        Worth surfacing: on a long Loss Run, truncation silently drops claim rows
        — a recall failure that per-field confidence cannot see, because the
        missing rows produced no tokens.
        """
        return self.finish_reason in ("length", "max_tokens", STOPPED_REPEATING)


#: The finish reason of an answer a backend stopped because its end had become
#: one passage over and over (``inference_core.openai_backend.repeating_tail``).
#: Left to run, such an answer writes to the token limit - minutes for one
#: window - and is then refused as cut off anyway.
STOPPED_REPEATING = "repetition"


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

def assert_tokens_reconstruct(text: str, tokens: list[str], where: str) -> None:
    """Assert the token list rebuilds the generated text exactly.

    **This is the invariant the whole confidence signal rests on.**
    :func:`inference_core.span_map.map_field_spans` locates each field by
    character offset and then attributes the tokens covering that range. If the
    tokens do not concatenate back to the same string — one dropped special
    token, one decode that stripped whitespace — every offset after the
    divergence is wrong, and confidence is attributed to the wrong field.

    Nothing downstream can detect that. The numbers stay in range, the schema
    still validates, and a field the model was unsure about reports the
    confidence of its neighbour. So it is checked here, once, at the only place
    that knows both halves.
    """
    rebuilt = "".join(tokens)
    if rebuilt == text:
        return

    # Locate the first divergence: "they differ" is not actionable on a 4,000
    # character generation.
    index = next(
        (i for i, (a, b) in enumerate(zip(rebuilt, text, strict=False)) if a != b),
        min(len(rebuilt), len(text)),
    )
    raise ModelRunnerError(
        f"{where}: the returned tokens do not reconstruct the generated text "
        f"({len(rebuilt)} chars from {len(tokens)} tokens vs {len(text)} chars of text; "
        f"first difference at {index}: {rebuilt[max(0, index - 30):index + 30]!r} vs "
        f"{text[max(0, index - 30):index + 30]!r}). Per-field confidence is attributed by "
        "character offset, so a mismatch here silently gives each field its neighbour's "
        "confidence — with no error anywhere downstream. Decode without skipping special "
        "tokens, and return the per-token strings the decoder actually produced."
    )


class ModelBackend(ABC):
    """What every backend must provide."""

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], config: RunnerConfig,
                 adapter: str | None = None) -> Generation: ...

    @abstractmethod
    def supports_logprobs(self) -> bool: ...

    def close(self) -> None:
        """Release whatever the backend holds (GPU memory). Default: nothing."""
        return None

    def supports_submit(self) -> bool:
        """Whether :meth:`submit` sends one request on its own, to be answered
        while others run. A backend that can only run a batch to its end (the
        in-process engine) cannot: a caller then waits for the whole batch."""
        return False

    def submit(self, messages: list[dict[str, Any]], config: RunnerConfig,
               adapter: str | None = None) -> Future:
        """One request, sent now: a future of its :class:`Generation`, or of the
        exception that failed it (returned, never raised)."""
        raise NotImplementedError(f"{type(self).__name__} runs batches only")

    def generate_batch(
        self,
        messages_list: list[list[dict[str, Any]]],
        configs: list[RunnerConfig],
        adapter: str | None = None,
    ) -> list[Generation | Exception]:
        """Several independent requests. One result per request, in order.

        A failure is returned in its slot rather than raised, so one bad request
        does not lose the others. The default runs them one at a time; a backend
        that can batch (vLLM) overrides it and runs them together.
        """
        import time

        out: list[Generation | Exception] = []
        for messages, config in zip(messages_list, configs, strict=True):
            started = time.perf_counter()
            try:
                result = self.generate(messages, config, adapter=adapter)
            except Exception as exc:  # noqa: BLE001 - reported per request, not raised
                out.append(exc)
                continue
            if result.latency_ms is None:
                # One at a time, so each request's own time is how long it took.
                result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            out.append(result)
        return out


class EchoBackend(ModelBackend):
    """Deterministic stub for tests — no GPU, no weights, no network.

    Returns a canned response with synthetic tokens that reconstruct the text
    exactly, so :mod:`inference_core.span_map` can be exercised end to end.
    """

    def __init__(self, response: str = '{"line_of_business":[]}', chunk: int = 4) -> None:
        self.response = response
        self.chunk = chunk
        self.calls: list[dict[str, Any]] = []

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        self.calls.append({"messages": messages, "adapter": adapter,
                           "temperature": config.temperature,
                           "json_schema": config.json_schema})
        tokens = [self.response[i:i + self.chunk] for i in range(0, len(self.response), self.chunk)]
        return Generation(
            text=self.response,
            tokens=tokens,
            token_logprobs=[-0.01 * (i % 5) for i in range(len(tokens))],
            generation_fingerprint=config.fingerprint(),
        )


#: Structured decoding without free whitespace. With it allowed, the grammar
#: accepts any run of spaces and newlines between tokens, and a model that
#: drifts there writes newlines until max_new_tokens - one validation answer
#: ran 30,000 lines. xgrammar without it writes json.dumps' own separators
#: (", " and ": "), which is how the training answers are written. Only the
#: xgrammar and guidance backends take the setting, so xgrammar is named; every
#: scope's schema is inside what it supports (tests/test_vllm_patches.py).
STRUCTURED_OUTPUTS_ENGINE = {"backend": "xgrammar", "disable_any_whitespace": True}


def to_vllm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat messages with ms-swift image parts rewritten for ``LLM.chat``.

    Corpus rows and serving requests carry ``{"type": "image", "image": x}`` -
    ms-swift's shape, which training reads. vLLM's chat parser knows no ``image``
    part and refused every row ("Unknown part type: image"), so checkpoint
    selection scored nothing. A path or bytes becomes an opened ``image_pil``
    (no local-media allow-list needed); a URL becomes ``image_url``.
    """
    import io

    def part(block: Any) -> Any:
        if not (isinstance(block, dict) and block.get("type") == "image"):
            return block
        image = block.get("image")
        if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": image}}
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                opened.load()
                return {"type": "image_pil", "image_pil": opened.copy()}
        if isinstance(image, bytes):
            with Image.open(io.BytesIO(image)) as opened:
                opened.load()
                return {"type": "image_pil", "image_pil": opened.copy()}
        if isinstance(image, Image.Image):
            return {"type": "image_pil", "image_pil": image}
        raise ModelRunnerError(f"image part holds {type(image).__name__}, not a path, bytes or image")

    return [
        {**message, "content": [part(b) for b in message["content"]]}
        if isinstance(message.get("content"), list) else message
        for message in messages
    ]


def repeatable_engine_settings(config: RunnerConfig) -> dict[str, Any]:
    """The engine arguments, and the environment, for ``config.repeatable``.

    Batch-invariant kernels (``VLLM_BATCH_INVARIANT``) make a request's numbers
    independent of what shares its batch, where the installed vLLM has them; the
    prefix cache is turned off either way, so no request reuses another's
    computed prompt. Set before vLLM is imported: it reads the variable then.
    """
    if not config.repeatable:
        return {}
    import importlib.util
    import os

    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    try:
        has_kernels = importlib.util.find_spec("vllm.model_executor.layers.batch_invariant") is not None
    except ModuleNotFoundError:
        has_kernels = False
    if not has_kernels:
        log.warning(
            "repeatable generation asked for, and this vLLM has no batch-invariant kernels: "
            "only the prefix cache is turned off, so answers may still differ between runs"
        )
    return {"enable_prefix_caching": False}


class VLLMBackend(ModelBackend):
    """vLLM — the serving path. Multi-LoRA hot-swap, OpenAI-compatible."""

    def __init__(self, resolved: ResolvedModel, config: RunnerConfig) -> None:
        self.resolved = resolved
        self.config = config
        self._engine = None

    def supports_logprobs(self) -> bool:
        return True

    def close(self) -> None:
        """Drop the engine and hand its GPU memory back.

        One pipeline process loads several engines in turn — the base for
        checkpoint selection, the merged bf16 for calibration and the golden
        eval, then each quantized format. An engine takes
        ``gpu_memory_utilization`` of the card and nothing freed it, so the
        second load failed for want of memory it could see was allocated.
        """
        if self._engine is None:
            return
        self._engine = None
        import gc

        try:  # pragma: no cover - needs vLLM and a GPU
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001 - best effort; the GC below still runs
            pass
        gc.collect()
        try:  # pragma: no cover - needs torch with CUDA
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load(self, config: RunnerConfig):
        """Build the engine once and keep it. A cold start per request would
        reload 16GB of weights for every document."""
        if self._engine is not None:
            return self._engine

        from inference_core.vllm_patches import patch_qwen3_vl_lora

        # Before vLLM is imported (inference_core.vllm_patches).
        patch_qwen3_vl_lora()
        repeatable = repeatable_engine_settings(config)
        from vllm import LLM

        from common.gpu import require_cuda

        require_cuda("the vLLM engine")

        # Checked where the engine is built, which is the only place the mode can
        # still be changed. The comment below used to claim this and nothing did.
        config.assert_logprobs_are_the_models_own()
        model_path = self.resolved.get("merged_model") or self.resolved.get("base_model")
        if not model_path:
            raise ModelRunnerError(
                f"{self.resolved.get('tag')} resolves to no servable weights "
                f"({dict(self.resolved)}). vLLM serves a merged model or the base; a bare adapter "
                "stack has to be merged first (IMPL-10)."
            )

        from common.config import pixel_budget

        # The pixel budget training resized pages to (common.config.pixel_budget),
        # passed to the processor explicitly. Left to its default the engine
        # resized to a budget nobody chose — identical today by coincidence, and
        # silently different the day either side's budget moved.
        min_pixels, max_pixels = pixel_budget()
        self._engine = LLM(
            model=model_path,
            mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
            enable_lora=config.enable_lora,
            max_loras=config.max_loras,
            max_lora_rank=config.max_lora_rank,
            max_model_len=config.max_seq_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            # Engine-level, not per request: without it the confidence features
            # read the post-mask distribution under structured decoding (§5.1).
            logprobs_mode=config.logprobs_mode,
            structured_outputs_config=STRUCTURED_OUTPUTS_ENGINE,
            **repeatable,
        )
        log.info("vLLM engine up on %s (lora=%s, repeatable=%s)", model_path, config.enable_lora,
                 config.repeatable)
        return self._engine

    @staticmethod
    def _imports():
        try:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise ModelRunnerError(
                'vLLM is not installed. Install the [serve] extra on the pod: pip install -e ".[serve]"'
            ) from exc
        return SamplingParams, LoRARequest

    @staticmethod
    def _sampling_params(config: RunnerConfig):
        SamplingParams, _ = VLLMBackend._imports()
        sampling: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_new_tokens,
            # Requested per token, not per sequence: confidence is per field, and
            # a sequence-level score cannot be attributed to one.
            "logprobs": config.top_logprobs if config.logprobs else None,
            "seed": config.seed,
        }

        # Structured decoding against the target schema (arch v2.1 §13).
        # Guarantees structural validity and stops an enum field taking an
        # invalid value — the model cannot emit a doc type or LoB outside the
        # enum, because those tokens are masked at decode time.
        #
        # THE CAVEAT THAT MATTERS: masking changes the distribution the sampler
        # sees, so `logprobs_mode: raw_logprobs` must be set on the engine or
        # every confidence feature is computed on the post-mask distribution
        # rather than the model's own (§5.1). That is an engine-level setting,
        # asserted at load, not a per-request one — a calibrator fitted on masked
        # logprobs describes a distribution the model never produced.
        if getattr(config, "json_schema", None):
            try:
                from vllm.sampling_params import GuidedDecodingParams

                sampling["guided_decoding"] = GuidedDecodingParams(json=config.json_schema)
            except ImportError:  # pragma: no cover - older vLLM
                log.warning(
                    "this vLLM build exposes no GuidedDecodingParams, so schema validity is "
                    "NOT guaranteed at decode time and the IMPL-07 audit gate is the only "
                    "thing catching an invalid extraction (arch v2.1 §13)."
                )
        return SamplingParams(**{k: v for k, v in sampling.items() if v is not None})

    @staticmethod
    def _lora(adapter: str | None):
        _, LoRARequest = VLLMBackend._imports()
        # Hot-swap rather than reload. The adapter is a per-request argument
        # precisely so one engine serves every document type (arch §4).
        return LoRARequest(adapter, abs(hash(adapter)) % (10 ** 8), adapter) if adapter else None

    @staticmethod
    def _to_generation(output: Any, config: RunnerConfig, latency_ms: float) -> Generation:
        if not output or not output.outputs:
            raise ModelRunnerError("vLLM returned no completion for this request")
        completion = output.outputs[0]

        tokens, logprobs = [], []
        token_ids = list(getattr(completion, "token_ids", None) or [])
        for index, step in enumerate(completion.logprobs or []):
            # Each step maps token_id -> Logprob. The sampled token is looked up
            # by its id, not as the rank-1 entry: under raw_logprobs with
            # structured decoding the sampled token is the best VALID one, which
            # need not be rank 1 in the unmasked distribution — picking rank 1
            # recorded a token that was never emitted and broke reconstruction.
            chosen = step.get(token_ids[index]) if index < len(token_ids) else None
            if chosen is None:
                chosen = next((lp for lp in step.values() if getattr(lp, "rank", 1) == 1), None)
            if chosen is None:
                continue
            tokens.append(chosen.decoded_token)
            logprobs.append(float(chosen.logprob))

        text = completion.text
        if tokens and "".join(tokens) != text and "".join(tokens[:-1]) == text:
            # The end-of-turn token (<|im_end|>) that stopped generation: vLLM
            # returns it among the tokens and logprobs but not in the text. It
            # carries no field, so it is dropped - only it, and only when the rest
            # reconstructs the text exactly. Left in, every completed answer
            # failed the check below and was scored as empty.
            tokens, logprobs = tokens[:-1], logprobs[:-1]
        if tokens:
            assert_tokens_reconstruct(text, tokens, "vLLM")

        prompt_ids = getattr(output, "prompt_token_ids", None)
        return Generation(
            text=text,
            tokens=tokens,
            token_logprobs=logprobs,
            finish_reason=completion.finish_reason or "stop",
            generation_fingerprint=config.fingerprint(),
            latency_ms=round(latency_ms, 1),
            prompt_tokens=len(prompt_ids) if prompt_ids is not None else None,
        )

    def generate(self, messages, config, adapter=None) -> Generation:
        import time

        engine = self._load(config)
        params = self._sampling_params(config)
        started = time.perf_counter()
        outputs = engine.chat(to_vllm_messages(messages), params,
                              lora_request=self._lora(adapter))
        latency_ms = (time.perf_counter() - started) * 1000
        return self._to_generation(outputs[0] if outputs else None, config, latency_ms)

    def generate_batch(self, messages_list, configs, adapter=None):
        """All requests in ONE ``engine.chat`` call.

        vLLM schedules them together (continuous batching), so a policy's windows
        run concurrently rather than one after another. Each request keeps its
        own sampling parameters — every window is constrained to a different
        schema slice. The shared wall time is reported on each result, because
        that is how long each one actually waited.
        """
        import time

        if not messages_list:
            return []
        engine = self._load(configs[0])
        params = [self._sampling_params(config) for config in configs]
        started = time.perf_counter()
        try:
            outputs = engine.chat([to_vllm_messages(m) for m in messages_list], params,
                                  lora_request=self._lora(adapter))
        except Exception as exc:  # noqa: BLE001 - the whole batch failed; say so per slot
            return [exc for _ in messages_list]
        latency_ms = (time.perf_counter() - started) * 1000

        results: list[Generation | Exception] = []
        for index, config in enumerate(configs):
            try:
                output = outputs[index] if index < len(outputs) else None
                results.append(self._to_generation(output, config, latency_ms))
            except Exception as exc:  # noqa: BLE001 - reported per request
                results.append(exc)
        return results


class HFBackend(ModelBackend):
    """Transformers — local evaluation, and the fallback when vLLM cannot serve."""

    def __init__(self, resolved: ResolvedModel, config: RunnerConfig) -> None:
        self.resolved = resolved
        self.config = config
        self._model = None
        self._processor = None

    def supports_logprobs(self) -> bool:
        return True   # generate(output_scores=True, return_dict_in_generate=True)

    def _load(self, config: RunnerConfig):
        """Load model and processor once, applying the adapter stack if present."""
        if self._model is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        from common.gpu import require_cuda

        require_cuda("the Hugging Face backend")
        base = self.resolved.get("merged_model") or self.resolved.get("base_model")
        if not base:
            raise ModelRunnerError(
                f"{self.resolved.get('tag')} resolves to no weights ({dict(self.resolved)})"
            )

        model = AutoModelForVision2Seq.from_pretrained(
            # "cuda", not "auto": auto quietly offloads layers to the CPU when
            # VRAM runs short, and every generation then crawls.
            base, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )

        # Foundation first, then the per-type LoRA on top — the order they were
        # trained in. Reversing it produces a different model from the one that
        # was evaluated (arch §4).
        for adapter_path in (self.resolved.get("foundation_adapter"),
                             self.resolved.get("type_adapter")):
            if not adapter_path:
                continue
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            log.info("applied adapter %s", adapter_path)

        model.eval()
        self._model = model
        self._processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        return self._model, self._processor

    def loaded_adapters(self) -> list[str]:
        """The adapters bound at load time: the only ones a request may name."""
        return [str(path) for path in (self.resolved.get("foundation_adapter"),
                                       self.resolved.get("type_adapter")) if path]

    def generate(self, messages, config, adapter=None) -> Generation:
        import contextlib
        import time

        # Checked before the dependency import: asking this backend to hot-swap
        # an adapter is a contract violation whether or not torch is installed,
        # and reporting it as a missing dependency sends the reader to the wrong
        # problem entirely.
        if adapter and adapter not in self.loaded_adapters():
            # HF applies adapters at load time, not per request. Silently ignoring
            # a mismatch would serve one document type's weights under another's
            # routing decision.
            raise ModelRunnerError(
                f"this backend was loaded with adapter(s) {self.loaded_adapters()} and "
                f"was asked to generate with {adapter!r}. Transformers binds adapters at load "
                "time; per-request hot-swap is the vLLM path (IMPL-11). Load a runner per adapter, "
                "or serve through vLLM."
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise ModelRunnerError(
                'transformers/torch are not installed. Install the [train] extra: pip install -e ".[train]"'
            ) from exc

        model, processor = self._load(config)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        # No adapter on the base + adapter model is the base alone (NO_ADAPTER: the
        # line classifier's read): the adapter bound at load is switched off for it.
        base_only = (adapter is None and self.resolved.get("kind") == "base_plus_adapter"
                     and hasattr(model, "disable_adapter"))
        started = time.perf_counter()
        with torch.no_grad(), (model.disable_adapter() if base_only else contextlib.nullcontext()):
            out = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=not config.is_greedy,
                temperature=config.temperature if not config.is_greedy else None,
                top_p=config.top_p if not config.is_greedy else None,
                output_scores=config.logprobs,
                return_dict_in_generate=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000

        generated_ids = out.sequences[0][prompt_len:]
        # Special tokens are NOT skipped: the token strings have to concatenate
        # back to the decoded text, and skipping them breaks that by exactly the
        # characters they occupy.
        text = processor.decode(generated_ids, skip_special_tokens=True)

        tokens: list[str] = []
        logprobs: list[float] = []
        if config.logprobs and getattr(out, "scores", None):
            for step, token_id in zip(out.scores, generated_ids, strict=False):
                token = processor.decode([token_id], skip_special_tokens=True)
                if not token:
                    continue      # a special token contributes no text
                tokens.append(token)
                logprobs.append(
                    float(torch.log_softmax(step[0].float(), dim=-1)[token_id])
                )

        if tokens:
            assert_tokens_reconstruct(text, tokens, "Transformers")

        finished = generated_ids.shape[-1] >= config.max_new_tokens
        return Generation(
            text=text,
            tokens=tokens,
            token_logprobs=logprobs,
            finish_reason="length" if finished else "stop",
            generation_fingerprint=config.fingerprint(),
            latency_ms=round(latency_ms, 1),
        )


class GGUFBackend(ModelBackend):
    """llama.cpp — the portable/edge path.

    Qwen3-VL is multimodal, so a GGUF export needs the ``mmproj`` projector file
    alongside the quantized LLM. Whether llama.cpp supports it for this model is
    unverified (arch §13a), which is why GGUF is not the primary serving route.
    """

    def __init__(self, resolved: ResolvedModel, config: RunnerConfig) -> None:
        self.resolved = resolved
        self.config = config

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        raise NotImplementedError(
            "GGUF serving is the portable path, not the first-cycle one. Confirm llama.cpp "
            "supports Qwen3-VL with an mmproj file before wiring it (arch §13a)."
        )


#: Constructors, not classes: every real backend takes ``(resolved, config)``
#: while the ABC itself takes nothing, so the callable type is the accurate one.
_BACKENDS: dict[Backend, Callable[[ResolvedModel, RunnerConfig], ModelBackend]] = {
    "vllm": VLLMBackend,
    "hf": HFBackend,
    "gguf": GGUFBackend,
}


# --------------------------------------------------------------------------
# Loading and generating
# --------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """A resolved model version plus the backend that runs it."""

    tag: str
    resolved: ResolvedModel
    backend: ModelBackend
    config: RunnerConfig
    #: A LoRA adapter applied to every request that names none of its own: the
    #: base model plus a trained adapter, NOT merged into it (``extract
    #: --adapter``). A request's own adapter (a per-type LoRA) still wins.
    default_adapter: str | None = None

    @property
    def is_base(self) -> bool:
        """Whether this is the untuned base with no adapter.

        The shared path for the pilot's zero-shot baseline (IMPL-15), day-zero
        pre-annotation (IMPL-04), and ``extract --model base`` (IMPL-13).
        """
        return self.resolved.get("kind") == "base"


def release_model(model: Any) -> None:
    """Free a loaded model's backend. A no-op for anything without one."""
    backend = getattr(model, "backend", None)
    if backend is not None and hasattr(backend, "close"):
        backend.close()


def load_model(
    tag: str,
    client: BlobClient,
    *,
    doc_type: str | None = None,
    backend: Backend | None = None,
    quant_format: str | None = None,
    config: RunnerConfig | None = None,
    backend_impl: ModelBackend | None = None,
    adapter: str | None = None,
    label: str | None = None,
) -> LoadedModel:
    """Resolve a version tag and prepare it for generation.

    ``tag`` accepts ``base`` as a first-class value — the untuned model with no
    adapter — as well as ``v1``, ``v2``, and so on.

    Args:
        backend_impl: inject a backend directly, for tests. Production callers
            leave this unset and get the configured backend.
        adapter: a LoRA adapter directory (a training checkpoint, or the run's
            adapter folder) - local, or a Blob prefix copied down once - applied
            to the BASE model on every request, without merging. Only with
            ``tag="base"``. The engine then loads the base weights with LoRA on.
        label: the version name results are filed under; defaults to the base
            plus the adapter's folder names.
    """
    config = config or load_runner_config(backend)
    if adapter is not None and tag != "base":
        raise ModelRunnerError(
            f"--adapter applies a LoRA to the BASE model, but --model is {tag!r}. Use --model base "
            "with --adapter for base + adapter, or --model vN alone for that version (the base with its adapter)."
        )
    resolved = resolve_model_version(tag, client, doc_type=doc_type, quant_format=quant_format)
    default_adapter = None
    if adapter is None and resolved.get("kind") == "base_plus_adapter" and resolved.get("foundation_adapter"):
        # A version of the policy adapter is served as a LoRA on the base, never
        # merged (decision 2026-10-23): the base with its adapter on every request.
        default_adapter = prepare_adapter(str(resolved["foundation_adapter"]), client, config)
    if adapter is not None:
        default_adapter = prepare_adapter(adapter, client, config)
        tag = label or adapter_label(default_adapter)
    if default_adapter is not None:
        # The local copy, not the Blob prefix: vLLM is handed it per request, and
        # the HF backend binds it at load time (HFBackend._load).
        resolved = type(resolved)(resolved)
        resolved["kind"] = "base_plus_adapter"
        resolved["foundation_adapter"] = default_adapter

    if backend_impl is not None:
        impl = backend_impl
    else:
        backend_cls = _BACKENDS.get(config.backend)
        if backend_cls is None:
            raise ModelRunnerError(
                f"unknown backend {config.backend!r}; expected one of {sorted(_BACKENDS)}"
            )
        impl = backend_cls(resolved, config)

    if config.logprobs and not impl.supports_logprobs():
        raise ModelRunnerError(
            f"backend {config.backend!r} cannot return logprobs, but per-field confidence "
            "depends on them (arch §5). Use a backend that can, or accept that this run "
            "produces no confidence signal — and say so explicitly rather than by omission."
        )

    log.info(
        "loaded %s (kind=%s, backend=%s, staged=%s%s)",
        tag, resolved.get("kind"), config.backend, resolved.get("from_staging"),
        f", adapter {default_adapter} applied to every request, not merged" if default_adapter else "",
    )
    return LoadedModel(tag=tag, resolved=resolved, backend=impl, config=config,
                       default_adapter=default_adapter)


def without_adapter(model: LoadedModel) -> LoadedModel:
    """The same engine answering with no adapter: the base model.

    For comparing base and base + adapter on the same documents with one model
    load. The engine was built with LoRA on, and a request with no LoRA request
    runs the base weights alone.
    """
    import dataclasses

    resolved = type(model.resolved)(model.resolved)
    resolved["kind"] = "base"
    resolved["foundation_adapter"] = None
    return dataclasses.replace(model, tag="base", resolved=resolved, default_adapter=None)


#: Where an adapter named by a Blob prefix is copied, once per prefix.
ADAPTER_CACHE = Path("/workspace/adapters") if Path("/workspace").is_dir() else Path.home() / ".cache" / "fideon-adapters"


def prepare_adapter(adapter: str, client: BlobClient, config: RunnerConfig) -> str:
    """A local LoRA adapter directory the engine can load, checked.

    ``adapter`` is a local directory or a Blob prefix (copied to
    :data:`ADAPTER_CACHE`). Refused when it holds no ``adapter_config.json``,
    when its rank is above the engine's ``max_lora_rank``, or when the engine
    is configured without LoRA - each would otherwise fail at engine start or,
    worse, run the bare base and report it as the adapter.
    """
    import re

    local = Path(adapter)
    if not local.is_dir():
        target = ADAPTER_CACHE / re.sub(r"[^A-Za-z0-9._-]+", "_", adapter.strip("/"))
        if not (target / "adapter_config.json").is_file():
            copied = client.download_dir(adapter, target)
            log.info("copied adapter %s (%d file(s)) to %s", adapter, copied, target)
        local = target
    config_file = local / "adapter_config.json"
    if not config_file.is_file():
        raise ModelRunnerError(
            f"{adapter} holds no adapter_config.json, so it is not a LoRA adapter. Point --adapter at "
            "a checkpoint folder (.../checkpoint-N) or the run's adapter folder."
        )
    rank = int(json.loads(config_file.read_text(encoding="utf-8")).get("r") or 0)
    if rank > config.max_lora_rank:
        raise ModelRunnerError(
            f"{adapter} is a rank-{rank} adapter, above the engine's max_lora_rank "
            f"{config.max_lora_rank} (configs/inference/vllm_serving.yaml)."
        )
    if not config.enable_lora:
        raise ModelRunnerError(
            "the engine is configured with enable_lora: false, so the adapter would be ignored "
            "and the bare base model would answer. Set enable_lora: true to extract with --adapter."
        )
    return str(local.resolve())


def adapter_label(adapter: str) -> str:
    """``base+<run>-<checkpoint>``: where results with this adapter are filed."""
    import re

    parts = [part for part in Path(adapter).parts[-2:] if part not in ("", "/", "\\")]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(parts)).strip("-") or "adapter"
    return f"base+{name}"


def generate(
    model: LoadedModel,
    messages: list[dict[str, Any]],
    *,
    adapter: str | None = None,
    want_logprobs: bool = True,
    json_schema: dict[str, Any] | None = None,
    max_new_tokens: int | None = None,
) -> Generation:
    """Generate from prepared messages.

    Args:
        adapter: per-type LoRA to apply for this request. ``None`` means
            Foundation-only — the classifier's low-confidence fallback path
            (arch §4a), not an error.
        json_schema: the resolved target schema to constrain this request to.
            Per request because it is per document type.
        max_new_tokens: this request's answer limit (common.config.answer_cap),
            never above the configured one.
    """
    import time

    config = _request_config(model.config, json_schema, max_new_tokens)
    adapter = _adapter_for(model, adapter)
    started = time.perf_counter()
    try:
        result = model.backend.generate(messages, config, adapter=adapter)
    except NotImplementedError:
        raise
    except Exception as exc:
        raise ModelRunnerError(f"generation failed for {model.tag}: {exc}") from exc

    result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    if not result.generation_fingerprint:
        result.generation_fingerprint = model.config.fingerprint()

    if want_logprobs:
        result.assert_logprobs()
    if result.truncated():
        log.warning(
            "generation for %s was cut off (%s) — output may be incomplete. On a long "
            "Loss Run this silently drops claim rows, which per-field confidence cannot detect.",
            model.tag, result.finish_reason,
        )
    return result


#: Pass as ``adapter`` to read with the engine's weights alone: no LoRA, and
#: not the model's default adapter either (``None`` means that default).
NO_ADAPTER = ""


def _adapter_for(model: LoadedModel, adapter: str | None) -> str | None:
    """The LoRA a request is run with: its own, the model's default when it names
    none, or nothing at all for :data:`NO_ADAPTER`."""
    if adapter is None:
        return model.default_adapter
    return adapter or None


def _request_config(
    config: RunnerConfig, json_schema: dict[str, Any] | None, max_new_tokens: int | None,
) -> RunnerConfig:
    """The config for one request: its schema, and its answer limit when it has one."""
    import dataclasses

    changes: dict[str, Any] = {}
    if json_schema:
        changes["json_schema"] = json_schema
    if max_new_tokens:
        changes["max_new_tokens"] = min(int(max_new_tokens), int(config.max_new_tokens))
    return dataclasses.replace(config, **changes) if changes else config


def generate_batch(
    model: LoadedModel,
    requests: list[tuple[list[dict[str, Any]], dict[str, Any] | None]],
    *,
    adapter: str | None = None,
    want_logprobs: bool = True,
) -> list[Generation | ModelRunnerError]:
    """Several independent generations, run together where the backend can.

    ``requests`` is ``[(messages, json_schema), ...]`` — each constrained to its
    own schema, as a policy's windows are — or ``(messages, json_schema,
    max_new_tokens)`` with a per-request answer limit (common.config.answer_cap).
    One result per request, in order; a failed request is a
    :class:`ModelRunnerError` in its slot, never raised, so one window cannot
    lose the others. The checks :func:`generate` applies — logprobs present —
    apply to each result.
    """
    configs = [_request_config(model.config, request[1], request[2] if len(request) > 2 else None)
               for request in requests]
    raw = model.backend.generate_batch(
        [request[0] for request in requests], configs, adapter=_adapter_for(model, adapter)
    )
    return [_checked(model, result, want_logprobs) for result in raw]


def can_submit(model: LoadedModel) -> bool:
    """Whether ``model``'s backend answers requests one by one as they finish
    (:func:`submit_generation`), rather than only a whole batch at a time."""
    return model.backend.supports_submit()


def submit_generation(
    model: LoadedModel,
    request: tuple[Any, ...],
    *,
    adapter: str | None = None,
    want_logprobs: bool = True,
) -> Future:
    """One request of :func:`generate_batch`'s form, sent now.

    A future of its :class:`Generation` or of the :class:`ModelRunnerError`
    that failed it, checked as :func:`generate_batch` checks each result. Only
    for a backend that :func:`can_submit`.
    """
    import time

    config = _request_config(model.config, request[1], request[2] if len(request) > 2 else None)
    started = time.perf_counter()
    sent = model.backend.submit(request[0], config, adapter=_adapter_for(model, adapter))
    checked: Future = Future()

    def finish(done: Future) -> None:
        try:
            result = done.result()
        except Exception as exc:  # noqa: BLE001 - a failed request is its own result
            result = exc
        if isinstance(result, Generation) and result.latency_ms is None:
            result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        checked.set_result(_checked(model, result, want_logprobs))

    sent.add_done_callback(finish)
    return checked


def _checked(model: LoadedModel, result: Any, want_logprobs: bool) -> Generation | ModelRunnerError:
    """One backend result as the pipeline takes it: an error as a
    :class:`ModelRunnerError`, a generation fingerprinted, with its logprobs."""
    if isinstance(result, Exception):
        return (result if isinstance(result, ModelRunnerError)
                else ModelRunnerError(f"generation failed for {model.tag}: {result}"))
    if not result.generation_fingerprint:
        result.generation_fingerprint = model.config.fingerprint()
    if want_logprobs and not result.has_logprobs:
        try:
            result.assert_logprobs()
        except ModelRunnerError as exc:
            return exc
    return result
