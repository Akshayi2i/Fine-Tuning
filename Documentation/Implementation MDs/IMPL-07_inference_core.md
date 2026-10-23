# SPEC 07 — Inference Core (Shared Model-Runner Primitive)

> Aligned with the code of 2026-10-22.

> Read `IMPL-00_MASTER_CONTEXT.md` first. Dependencies: IMPL-01, IMPL-02. **This module exists to break the cycle between evaluation, calibration, serving, and testing** — they all need one shared way to run a model version over a document. Build it here, once; everything downstream imports it.
>
> **Naming hazard (master §0.1):** in this folder `IMPL-07` is the inference core. The architecture's "**Fideon SPEC_07 Stage 3**" is the external production **audit gate** — a different document. They are unrelated.
>
> **Architecture refs:** `finetuning-architecture-v2.5.docx` §5 (logprobs → field spans), §6 (modality modes), §7 (prompt template identity, multi-page inputs; §7b windows over every page), §11 (resolution cap).

## Why this module exists

Evaluation (IMPL-08), calibration (IMPL-09), serving (IMPL-11), and testing (IMPL-12) must all run the model **identically** — same prompt assembly, same generation config, same logprob capture — or their numbers won't agree and "test == prod" breaks. Divergence between training-time and inference-time prompts is the single most common cause of post-fine-tuning degradation, and it is invisible in training metrics because it only manifests at serving time (arch §7). That shared primitive lives here, with **no dependency on calibration, classification, or routing** — those are higher-level concerns layered on later.

## Goal

A clean, low-level inference engine: given a resolved model version + a prepared document input + a doc_type, produce raw generated JSON text and per-token logprobs. No confidence calibration, no classification, no routing here — just correct, reproducible generation.

## Deliverables

### 1. `inference_core/model_runner.py`
- `load_model(tag, client, *, doc_type=None, backend=None, quant_format=None, config=None, backend_impl=None, adapter=None, label=None)` → `LoadedModel` — resolves the tag (`base` is first-class, then `v1`, `v2`, …) to concrete artifacts via IMPL-02 `resolve_model_version` and builds the configured backend. The vLLM backend serves the base — with an adapter applied per request — or an older run's **merged** model. A version of the policy adapter (`v3`, run `policy-v3`) resolves to kind `base_plus_adapter`, and `load_model` applies its adapter as the default adapter: nothing is merged (decision 2026-10-23), and a non-bf16 `quant_format` for it is refused. Backend-swappable: vLLM for serving and every pipeline generation pass, HF for local eval, GGUF for the offline/edge path only (a stub that raises until llama.cpp support for Qwen3-VL with an `mmproj` file is confirmed).
  - **Base + adapter, not merged.** `adapter=` (accepted only with `tag="base"`) names a LoRA directory — a training checkpoint or a run's adapter folder, local or a Blob prefix copied once to `ADAPTER_CACHE`. `prepare_adapter` refuses one with no `adapter_config.json`, a rank above `max_lora_rank`, or an engine configured with `enable_lora: false` (the bare base would otherwise answer under the adapter's name). It becomes `LoadedModel.default_adapter`: applied **per request** to every request that names no adapter of its own, never folded into the weights. Results are filed under `label`, or `adapter_label()` (`base+<run>-<checkpoint>`). `without_adapter(model)` answers from the same engine with the base alone, and `NO_ADAPTER` (`""`) does the same for one request.
- `generate(model, messages, *, adapter=None, want_logprobs=True, json_schema=None, max_new_tokens=None)` → `Generation` (`text`, `tokens`, `token_logprobs`, `finish_reason`, `generation_fingerprint`, `latency_ms`).
  - **Logprobs are mandatory-capable** — vLLM supports them natively and HF via `generate(output_scores=True)`. Calibration (IMPL-09) is built entirely on them, so a backend that cannot return them is not a valid backend for anything but smoke testing. `load_model` refuses a backend that cannot return them; `Generation.assert_logprobs()` refuses an answer that came back without them.
  - `json_schema` constrains this one request (per request, because it is per document type and per window); `max_new_tokens` is this request's answer cap, never above the configured `max_new_tokens`. A generation that stops at the limit is logged (`Generation.truncated()`).
- `generate_batch(model, requests, *, adapter=None, want_logprobs=True)` — `requests` is `[(messages, json_schema)]` or `[(messages, json_schema, max_new_tokens)]`. The vLLM backend runs them in **one `engine.chat` call** (continuous batching), each with its own sampling parameters. One result per request, in order: a failed request is a `ModelRunnerError` in its slot, never raised, and a call that fails as a whole puts its exception in every slot. Backends that cannot batch run the requests one at a time.
- Deterministic given seed + greedy/temperature settings; every result carries `RunnerConfig.fingerprint()` (backend, temperature, top_p, `max_new_tokens`, seed, resolution cap, `structured_outputs`, `logprobs_mode`, and `repeatable` when on) so an output can be reproduced.
- **Structured outputs.** The vLLM engine is built with `structured_outputs_config=STRUCTURED_OUTPUTS_ENGINE` — `{"backend": "xgrammar", "disable_any_whitespace": True}`: no free whitespace between JSON tokens, so a drifting model cannot write newlines until the limit, and the separators are `json.dumps`' own (`", "`, `": "`), as the training answers are written. Each request carries its schema as `guided_decoding=GuidedDecodingParams(json=schema)`. `logprobs_mode` (`raw_logprobs`) is an engine argument, asserted where the engine is built (`RunnerConfig.assert_logprobs_are_the_models_own`), so confidence reads the model's own distribution, not the post-mask one. The sampled token is looked up by its token id (under masking it need not be rank 1), the stopping `<|im_end|>` token is dropped, and `assert_tokens_reconstruct` refuses tokens that do not concatenate back to the text — the invariant span mapping rests on.
- **Adapters.** vLLM applies **one LoRA per request** (`LoRARequest`), hot-swapped without reloading the base; `max_loras` caps how many stay resident, not how many are active. Loading with no adapter must also work — it is the classifier's low-confidence fallback path (IMPL-11) and the base-model reading. The HF backend binds adapters at load (Foundation, then the per-type adapter, in training order) and refuses a request naming a different adapter. It runs `load_model("vN")` too: it binds the version's local adapter copy at load, and a request with no adapter (`NO_ADAPTER`, the line classifier's read) runs the base alone, through PEFT's `disable_adapter`.
- **Repeatable mode** (`generation.repeatable`, off by default): sets `VLLM_BATCH_INVARIANT=1` before vLLM is imported (batch-invariant kernels where the installed vLLM has them, a warning where it has none) and turns prefix caching off, so a request's answer does not depend on what shares its batch.
- `to_vllm_messages` rewrites the ms-swift `{"type": "image"}` parts corpus rows and requests carry into `image_pil` / `image_url` parts `LLM.chat` accepts.
- `inference_core/vllm_patches.patch_qwen3_vl_lora` runs before vLLM is imported: on vLLM 0.11.0 it points Qwen3-VL's LoRA exclusion at the engine's own module names, so LoRA stays off the vision tower (otherwise the first request with an image fails in `_lora_shrink`), and it keeps the V1 engine core in this process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the patch reaches it.

### 2. `inference_core/input_builder.py`
- `build_messages(doc_type, image_paths, ocr_pages, modality_mode, *, acord_form=None, lob=None, sections=None, assistant_content=None, check_resolution_parity=True, page_numbers=None, total_pages=None)` → `BuiltMessages`: the exact chat-format `messages` (master §9), rendered through **`common.prompts`**. `lob` selects a policy's per-line schema, `sections` the slice of it one window asks for (`configs/schema_sections.yaml`), and `assistant_content` the golden JSON when `build_training_row` builds a corpus row. This is the single place prompt + image + OCR get assembled, so dataset build, evaluation, serving, and testing are guaranteed identical.
  - **`ocr_pages` is one markdown string per image, not one joined blob.** A mismatched length is refused: a short or long list shifts every page's text onto the wrong image, and every value after the shift is then attributed to the wrong page.
  - The user turn is **interleaved** — `[img_1][<page 1 of M> + md_1][img_2][<page 2 of M> + md_2]…` — so which text belongs to which image is positional rather than something the model infers. Joining destroyed that, and (because a blank line ends a Markdown table) split every table crossing a page boundary in two.
  - `page_numbers` / `total_pages` let one window of a long document label its pages truthfully (`<page 9 of 20>`), in the document's own numbering. Every page of a document is read (decision 2026-10-19): a request is either the whole document or one window of it — a policy's section-group windows (`policy_windows.plan_windows`, the same planner the corpus build uses) or a long Loss Run's row windows — never a selected subset. A window's numbering (pages 6–12 of 40, or a declarations window of pages 1–5 plus page 9) is what tells the model it holds a fragment, so fields living on pages other windows read are absent rather than missing. `build_training_row` forwards both, so a corpus row of a window carries the same markers.
  - `image_only` carries the markers and no markdown. The guarantee is *no OCR text*, not *no text at all*.
- Handles `image_only` (omit the OCR block, use the image-only prompt with its explicit "no OCR provided" declaration) vs `ocr_plus_image` (arch §6). `image_only` with OCR text, or another mode without it, is refused.
- **Multi-page inputs:** accepts an ordered list of page images and emits them as ordered `image` blocks, each followed by its own page's text. **Page order is positionally meaningful** under Interleaved-MRoPE (arch §3) — ordering must be explicit and stable, never set-like.
- `shown_pages(messages)` and `page_total(messages)` read a prompt's page numbers and the document's page count back from its `<page i of N>` markers — what validation generation and serving bound a window's `page_ref` with (§4).
- Applies the **resolution cap from config** to any image passed in — identical to the cap used at corpus build (arch §11). Assert the cap matches `configs/base_model.yaml` (`common.config.assert_resolution_parity`); a silent mismatch between training prep and inference is a correctness bug, not a tuning knob. The vLLM engine additionally receives the **pixel budget** as `mm_processor_kwargs` from `common.config.pixel_budget()`, the function training takes its budget from.
- A page whose OCR text is blank gets `EMPTY_PAGE_TEXT` — the same placeholder in training rows and serving requests.
- Exposes `prompt_template_version`, `schema_version` and `prompt_fingerprint` on the built message set so callers can verify they match the corpus the model was trained on; `BuiltMessages.assert_matches_corpus(manifest)` compares them with the corpus manifest's per-selector `schema_versions`.

### 3. `inference_core/span_map.py`
- `map_field_spans(generated_text, tokens, token_logprobs, *, strict=False)` → for each JSON field path — **including nested fields and individual list rows** (e.g. `claims[0].amount`) — the token span and its logprobs.
- This is the raw material calibration (IMPL-09) turns into confidence. Pure and testable; contains **no calibration logic itself**.
- Must handle: values spanning multiple tokens, `null` values, numeric vs string values, and rows inside arrays. A field whose span cannot be mapped is reported explicitly (`FieldSpan.mapped`, `unmapped_fields`) rather than silently dropped — an unmapped field would otherwise get no confidence and look like a clean extraction.
- **A string's span is its own characters, not its quotes** (an empty string keeps them): a quote shares its token with the JSON around it (`": "`, `",`), whose near-certain logprob says the structure was predictable, not that the value was read.
- Canonical envelopes are re-keyed by `common.canonical.collapse_spans`: `policy.effective_date.raw` / `.parsed` / `.page_ref[i]` become `policy.effective_date`, carrying **`raw`'s span** — the reading of the page, written first — with `parsed`'s standing in only when nothing was read; `page_ref` tokens are dropped.
- `list_field_paths` and `row_count` give the row-completeness signal (IMPL-09) the rows that were generated.

### 4. `inference_core/runner_config.py`
- Central generation + backend config (`RunnerConfig`: backend, temperature, top_p, `max_new_tokens`, seed, logprobs / `top_logprobs`, `logprobs_mode`, `structured_outputs`, `repeatable`, `max_seq_len`, GPU memory share, LoRA settings), loaded from `configs/inference/vllm_serving.yaml` by `load_runner_config()`. One source of truth so eval, serving, and testing cannot drift on generation parameters either. Logprobs off is refused; a non-greedy temperature warns.
- Current values: greedy (`temperature: 0.0`), `max_new_tokens: 12288` (at least every task's reserved answer), `max_seq_len: 32768`, `structured_outputs: true`, `logprobs_mode: raw_logprobs`, `repeatable: false`, `enable_lora: true`, `max_loras: 4`, `max_lora_rank: 64`. The same file's `generation.validation_batch_rows: 96` is read by evaluation's whole-split generation (IMPL-08), not by `RunnerConfig`.
- **Answer caps per task.** Callers pass each request's cap from `common.config.answer_cap(task, doc_type)` — the task's reserved `max_output_tokens` in `configs/shared/sequence.yaml`, which training targets are refused above, so a longer generation is a loop, not an answer: `policy_schedule` 12,288, `lossrun_rows` 8,192, `policy_declarations` 4,096, policy `extract` 3,072, `policy_endorsements` 3,072, the default 2,048.
- **Page bounds on the decoding schema.** Callers constrain with `common.schemas.with_page_bounds(schema, page_count, pages=shown_pages(messages))`: every `page_ref` is held to the pages the request shows — a range when they are consecutive, an `enum` when scattered (up to `MAX_PAGE_ENUM` = 32) — with at most that many entries, so a window cites only its own pages, as every training target does, and cannot count past the last page. For decoding only; the schema rendered into the prompt stays the one training showed.

## Constraints
- **No imports from** calibration, serving, evaluation, or testing (they import this, not vice-versa).
- Backend-agnostic interface so the same calls work in every context.
- Resolution cap and prompt rendering come from config/`common.prompts` — never inline.
- No PII in logs (never log generated field values or OCR text).

## Acceptance checklist
- [ ] `load_model("v2")` loads the base with v2's adapter applied as the default adapter, not merged (an older run's merged weights where that is what it has); `load_model("base", adapter=...)` serves base + a LoRA applied per request, not merged; base-only loading also works. The HF backend runs `load_model("vN")` with the adapter bound at load, and answers a request with no adapter from the base alone.
- [ ] `build_messages(...)` yields **byte-identical** structure for eval, serving, and testing given the same inputs.
- [ ] Each page image is immediately followed by that page's own text, and every text block carries its `<page N of M>` marker.
- [ ] A **window is a subsequence of the full document** — the same blocks, fewer pairs, the document's own page numbers — so a long policy is served in the shape it trained on. Comparing only *system* prompts does not catch this; the user turn must be compared too.
- [ ] No request carries a selected subset of pages: every page of a document is in the one whole-document request or in a window (`policy_windows.plan_windows` serves the windows the corpus was built with).
- [ ] A mismatch between the page-image count and the page-text count is refused.
- [ ] Two pages' markdown is never concatenated, so a table crossing a page boundary is not split.
- [ ] `build_messages(...)` rendering matches what IMPL-05 wrote into the corpus for the same doc_type + modality_mode (round-trip test against a corpus row).
- [ ] Multi-page input produces ordered image blocks; reversing the input order changes the output message order.
- [ ] Resolution cap mismatch between `base_model.yaml` and `vllm_serving.yaml` raises.
- [ ] `generate(...)` returns text + aligned token logprobs; `generate_batch(...)` returns one result per request in order, with a failure in its own slot.
- [ ] `map_field_spans(...)` correctly maps a scalar field, a `null` field, and a `claims[i].amount` list-row field on a sample generation; an unmappable field is reported, not dropped.
- [ ] Module imports without pulling in calibration/serving/eval/testing.

---

## Current implementation (2026-10-21)

- **Every page is read, never selected pages** (decision 2026-10-19). There is no page routing, page threshold
  or `long_doc_types` any more: `serving.page_router.plan_pages` returns every page and only detects the
  declarations page (`RoutingPlan` has no `routed`), and the routed single-call branch of serving is gone. A
  request reaching `build_messages` is the whole document (ACORD, a Loss Run whose rows fit one answer, any
  other type) or one window: a canonical policy's windows over every page, planned by `policy_windows.plan_windows` from
  `pages_to_read` exactly as the corpus build plans them, the declarations group's first window being pages
  1–5 plus the detected declarations page; or a long Loss Run's row windows, a shape no training row has yet
  (Loss Runs train as whole documents, IMPL-05). Each window is built with its own `page_numbers` and the
  document's `total_pages`.
- **vLLM backend**: refuses to build an engine without CUDA (`common.gpu.require_cuda`); passes
  `mm_processor_kwargs={"min_pixels", "max_pixels"}` from `pixel_budget()`; `close()` releases the engine
  (tears down vLLM's parallel state, collects, empties the CUDA cache) and `release_model(model)` calls it.
  The pipeline releases each engine before the next stage loads one — calibration per format and the golden
  eval each load a model on one card; checkpoint selection loads the base once per process, with one worker
  process and engine per GPU when training used several (IMPL-08).
- **HF backend**: `device_map="cuda"`, never `"auto"` (which silently offloads layers to the CPU when VRAM
  runs short); refuses without CUDA. For a version of the policy adapter it binds the local adapter copy at
  load (`HFBackend._load`), and turns it off with PEFT's `disable_adapter` for a request with no adapter.
- **Base weights** resolve to the pod's local copy (`common.config.base_model_source()`); the registry used
  to hand vLLM the identity string `model_id@revision`, which no loader accepts.
- **Base + adapter callers**: checkpoint selection applies each candidate checkpoint as the request's LoRA on
  one base engine (IMPL-08); calibration and the gate load the base with the selected adapter
  (`load_model("base", adapter=...)`), as it is served (decision 2026-10-23); `testing/compare_models.py` loads `load_model("base", adapter=...)` and reads
  the base through `without_adapter`.
- **Dates**: generations are post-processed to `MM/DD/YYYY` (`common.canonical.with_output_dates`); a field is
  a date by a whole word of its name (`date`, `dates`, `dated`, `dob`), list elements included.
- **Pinned**: vLLM `==0.11.0` on both the training and the serving pod — calibration is fitted on the logprobs
  one build produces, so serving runs the same build. The engine already takes `structured_outputs_config`;
  the per-request constraint is still 0.11's `GuidedDecodingParams`, which has to move to vLLM's
  structured-outputs request parameters before the pin is raised, and the Qwen3-VL LoRA patch is only needed
  (and only applied) on 0.11.0.

## Changes since 2026-10-05

- `load_model` signature corrected; added base + adapter loading (`adapter=`, `prepare_adapter`, `default_adapter`, `adapter_label`, `without_adapter`, `NO_ADAPTER`); "Foundation + per-type LoRA stacking" replaced by one LoRA per request (vLLM) and load-time binding (HF).
- `generate` signature and `Generation` fields corrected; added per-request `json_schema` / `max_new_tokens` and `generate_batch` (one `engine.chat` call per batch, failures per slot).
- Added structured outputs (xgrammar, `disable_any_whitespace`), `raw_logprobs` asserted at engine build, token-id lookup, end-of-turn token drop and token reconstruction check.
- Added repeatable mode, `to_vllm_messages`, and the vLLM 0.11.0 Qwen3-VL LoRA patch (`vllm_patches`).
- `build_messages` signature updated (`lob`, `sections`, `assistant_content`); removed the stale "concatenated OCR text" wording; added `shown_pages` / `page_total` and `assert_matches_corpus`.
- Span mapping: string spans exclude quotes; `collapse_spans` takes `raw`'s span.
- `runner_config`: listed the real fields and current values (`max_new_tokens` 12,288, `max_seq_len` 32,768); added per-task answer caps (`answer_cap`, policy schedule 12,288) and decoding page bounds (`with_page_bounds`).
- Current implementation: checkpoint selection's per-GPU engines, base + adapter callers, and the pin note updated.

## Changes since 2026-10-13

- Header: aligned with the code of 2026-10-21; architecture reference `finetuning-architecture-v2.4.docx` (§7b named).
- §2 `build_messages`: `page_numbers` / `total_pages` label one window of a long document, not a routed request; every page is read, a request is the whole document or one window, and `build_training_row` forwards the same markers.
- Acceptance checklist: "a routed subset" replaced by "a window is a subsequence of the full document"; new item: no request carries a selected subset of pages.
- Current implementation: every page read — no page routing, page threshold, `long_doc_types` or `RoutingPlan.routed`; the shapes a request can take (whole document, a canonical policy's windows, a long Loss Run's windows).

## Changes of 2026-10-22 (code review)

- Header: aligned with `finetuning-architecture-v2.5.docx` and the code of 2026-10-22.

## Changes of 2026-10-23 (incremental training)

- Nothing is merged (decision 2026-10-23): `load_model("vN")` resolves to kind `base_plus_adapter` and applies the version's adapter as the default adapter, refusing a non-bf16 format (§1, acceptance); calibration and the gate are base + adapter callers too.
- Code review of the incremental training: the HF backend runs `--model vN`, with the version's local adapter bound at load and the base alone for a request with no adapter (`disable_adapter`) (§1, acceptance, *Current implementation*).
