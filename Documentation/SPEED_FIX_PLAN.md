# Speed-fix plan — cycle-2 reading speed

Status: **in progress** (started 2026-11-12). Goal: cut the model's reading time
before the cycle-2 real run, without changing any output. Two measured causes.

### Progress
- **Bottleneck 1 (concurrent page prep): DONE 2026-11-12.** `read_pages_concurrently`
  in `inference_core/model_runner.py` builds a batch's requests on a thread pool;
  `generate_batch` uses it. 49 generation/runner/consistency tests pass; outputs
  unchanged (same objects, same order). Real speedup to be measured on the pod via
  the call's "reading pages" log line.
- **Bottleneck 2: reviewed, guards sound (2026-11-12).** Chosen approach: tune the
  existing guards (no truncation). Findings:
  - Repetition stop (`openai_backend.repetition`: `repeating_tail` 3000 chars ×6,
    `repeating_row`/`body` same key ×40 in last 12000, checked every 1000 chars)
    requires GENUINE repetition — safe, won't truncate a valid long answer.
  - Selection now uses the SAME repetition stop as production (`watched_chat`
    imports it) and the SAME config (`vllm_serving.yaml`), so it stops run-away
    answers like production and has the 1800 s process backstop — the two things
    missing when cycle-1 hung. Both already fixed.
  - **Only safe lever left = the `call_timeout_seconds` hang backstop (now 1800 s).**
    Lowering it would catch a hang faster, but a value below the real per-batch
    time on 4xH200 would kill valid work. **Set it from measured pod batch time**
    (so: after the smoke run). No threshold changes locally (would change outputs).

## Bottleneck 2 — a run-away / cap-length row holds or kills a call

- **Where:** `inference_core/model_runner.py` `generate_batch` wraps the engine
  call in `stall_guard(call_timeout_seconds, …)`, but `stall_guard`
  (`inference_core/stall_guard.py`) **ends the whole process** on timeout. A row
  that generates to its token cap holds the batch; a stuck call kills the
  process — this hung cycle-1 checkpoint selection.
- **Have already:** vLLM `abort_request` for a single generation
  (`model_runner.py` ~L435, `watched_chat`/`_WrittenSoFar`); longest-answer-first
  ordering in `evaluation/validation_generation.py:327-332`.
- **Fix:** a **graceful per-request cutoff** — abort only the run-away slot and
  return a timed-out result for it, instead of holding the batch or killing the
  process. Keep `stall_guard` as a longer last-resort backstop.
- **Verify:** force a capped/looping row; the batch returns on time, that slot
  marked timed-out, the others unaffected. Correctness via consistency/parity tests.

## Bottleneck 1 — serial page reading during request prep

- **Where:** `inference_core/model_runner.py:732`
  `requests = [to_vllm_messages(m) for m in messages_list]` builds each request's
  pages one at a time. Already measured by the log at `model_runner.py:741`
  ("X s reading pages, Y s in the engine"). Second prep site: the per-row loop in
  `evaluation/validation_generation.py:285-300`.
- **Fix:** build `to_vllm_messages` for the batch **concurrently** (thread pool;
  image decode/IO releases the GIL), output order unchanged.
- **Verify:** the "reading pages" log gives before/after; outputs byte-identical
  under the consistency/parity tests.

## Scope & rules
- Serving/inference code: **speed only, outputs unchanged.** Guarded by the
  existing consistency/parity tests after every change.
- **Final proof needs the pod** (real per-call timings on 4×H200). Locally:
  correctness tests + the relative prep speedup from the log.

## Order
1. Bottleneck 2 (cutoff) — also removes the cycle-1 hang risk.
2. Bottleneck 1 (concurrent prep).
3. Full test suite after each.
