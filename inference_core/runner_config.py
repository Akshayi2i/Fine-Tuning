"""Generation and backend settings, read once and shared.

Evaluation, serving, and testing must generate **identically** or their numbers
do not describe the same system. Generation parameters are as capable of
breaking that as the prompt is, so they come from one config
(``configs/inference/vllm_serving.yaml``) rather than from each caller's
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from common.config import ConfigError, generation_config, resolution_cap_px, serving_config

Backend = Literal["vllm", "hf", "gguf"]


@dataclass(frozen=True)
class RunnerConfig:
    """Everything that determines what the model produces for a given input."""

    backend: Backend = "vllm"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 4096
    seed: int = 42
    logprobs: bool = True
    top_logprobs: int = 1
    resolution_cap_px: int = 1792
    max_seq_len: int = 8192
    gpu_memory_utilization: float = 0.90
    enable_lora: bool = True

    #: vLLM applies ONE LoRA per request (arch v2.1 §4.1), so this caps how many
    #: are kept resident, not how many are active on a call. The v1 topology
    #: needed two active at once and could never have been served.
    max_loras: int = 4
    max_lora_rank: int = 64

    #: GiB of vLLM's processed-image cache; ``None`` leaves vLLM's default (4), 0 turns it off
    #: (configs/inference/vllm_serving.yaml says why it is off).
    mm_processor_cache_gb: float | None = None

    #: The Fideon SPEC_00 target schema for structured decoding (arch v2.1 §13).
    #: ``None`` leaves generation unconstrained, which is what the evaluation job
    #: uses for its training-health signal: whether the model learned the format
    #: on its own is a different question from whether the format is enforced.
    json_schema: dict[str, Any] | None = None

    #: MUST be ``raw_logprobs`` when structured decoding is on. Constrained
    #: decoding masks invalid tokens, so the post-mask distribution is not the
    #: model's own — a calibrator fitted on it describes a distribution the model
    #: never produced (§5.1). Asserted at load rather than hoped for.
    logprobs_mode: str = "raw_logprobs"

    #: Whether serving constrains generation to the target schema. The schema
    #: itself is per document type, so it is attached per request
    #: (``json_schema``); this flag is what the engine is loaded for.
    structured_outputs: bool = False

    #: Generate the same answer for the same input on every run: batch-invariant
    #: kernels where the installed vLLM has them, and no prefix caching. Greedy
    #: decoding alone is not enough - which requests share a batch changes the
    #: floating-point reductions, and a near-tie then flips (the smoke run read
    #: one document as 15 coverages and as 40). Off by default: without the
    #: prefix cache every window pays its whole system prompt again. For a
    #: run-to-run check; measure its cost before turning it on for evaluation.
    repeatable: bool = False

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    def assert_logprobs_are_the_models_own(self) -> None:
        """Refuse to serve masked logprobs as if they were the model's own.

        Every confidence feature in §5.1 reads this distribution, and every
        review threshold in §5.4 is defined against the result. Fitting a
        calibrator on post-mask logprobs produces a number that looks like a
        probability, is not one, and cannot be told apart from one downstream.
        """
        constrained = self.json_schema is not None or self.structured_outputs
        if constrained and self.logprobs_mode != "raw_logprobs":
            raise ValueError(
                f"structured decoding is on with logprobs_mode={self.logprobs_mode!r}. "
                "Constrained decoding masks invalid tokens, so these are not the model's own "
                "probabilities — set logprobs_mode: raw_logprobs in "
                "configs/inference/vllm_serving.yaml (arch v2.1 §5.1)."
            )

    def as_generation_kwargs(self) -> dict[str, Any]:
        """Backend-agnostic generation arguments."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "logprobs": self.logprobs,
        }

    def fingerprint(self) -> str:
        """Short hash of the settings that affect output.

        Recorded alongside a result so two runs that disagree can be checked for
        having actually generated the same way, rather than that being assumed.
        """
        import hashlib

        from common.config import resolution_cap_px

        # Constraint and logprob mode change what is generated and what the
        # confidence features read, so two runs differing in either did not
        # generate the same way.
        payload = (
            f"{self.backend}|{self.temperature}|{self.top_p}|{self.max_new_tokens}"
            f"|{self.seed}|{resolution_cap_px()}|{self.structured_outputs}|{self.logprobs_mode}"
        )
        if self.repeatable:
            payload += "|repeatable"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_runner_config(backend: Backend | None = None) -> RunnerConfig:
    """Build the config from ``configs/inference/vllm_serving.yaml``.

    Rejects a config with logprobs disabled: per-field confidence is derived from
    token logprobs (arch §5), so a runner that cannot return them produces
    extractions no one can route for review — which is the point of the
    confidence work.
    """
    serving = serving_config()
    generation = generation_config()          # raises when logprobs are off
    serve = serving.get("serving", {})
    sequence = serving.get("sequence", {})

    config = RunnerConfig(
        backend=backend or serve.get("backend", "vllm"),
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
        max_new_tokens=int(generation.get("max_new_tokens", 4096)),
        seed=int(generation.get("seed", 42)),
        logprobs=bool(generation.get("logprobs", True)),
        top_logprobs=int(generation.get("top_logprobs", 1)),
        resolution_cap_px=resolution_cap_px(),
        max_seq_len=int(sequence.get("max_seq_len", 8192)),
        gpu_memory_utilization=float(serve.get("gpu_memory_utilization", 0.90)),
        enable_lora=bool(serve.get("enable_lora", True)),
        max_loras=int(serve.get("max_loras", 4)),
        max_lora_rank=int(serve.get("max_lora_rank", 64)),
        mm_processor_cache_gb=(float(serve["mm_processor_cache_gb"])
                               if serve.get("mm_processor_cache_gb") is not None else None),
        # Read, not defaulted: both were in the YAML and reached nothing, so the
        # engine loaded with vLLM's own logprobs mode and generation ran
        # unconstrained while the config said otherwise.
        logprobs_mode=str(generation.get("logprobs_mode", "raw_logprobs")),
        structured_outputs=bool(generation.get("structured_outputs", False)),
        repeatable=bool(generation.get("repeatable", False)),
    )
    config.assert_logprobs_are_the_models_own()

    if not config.is_greedy:
        # Not fatal, but it costs reproducibility: the same document would
        # extract differently on two runs, and the calibration fitted on one
        # would not describe the other.
        import warnings

        warnings.warn(
            f"temperature is {config.temperature}, not 0.0. Extraction is not a creative task; "
            "sampling makes the same document extract differently across runs and invalidates "
            "the confidence calibration fitted on it.",
            stacklevel=2,
        )
    return config


def assert_lora_rank_supported(config: RunnerConfig, foundation_rank: int) -> None:
    """The serving stack must admit the Foundation adapter's rank.

    Catches the mismatch at config time rather than as a confusing load failure
    on a cold start, when it is much harder to attribute.
    """
    if foundation_rank > config.max_lora_rank:
        raise ConfigError(
            f"the Foundation adapter is rank {foundation_rank} but serving allows at most "
            f"{config.max_lora_rank}. Raise max_lora_rank in configs/inference/vllm_serving.yaml."
        )
