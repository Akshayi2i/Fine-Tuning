# Pod session checklist — before cycle 3

What to do on the next pod before running cycle 3. Written 2026-11-27.

Nearly all the code is already built and pushed. This session is mostly about
**proving it works on real hardware** and making **one split decision**. Times
are in IST; the pod clock is UTC, so convert.

**Every session starts the same way:**

```bash
cd /workspace/fine-tuning && git pull
/workspace/venv/bin/python --version   # a fresh container may not have python on PATH
```

---

## 0. Before any long work — check the GPUs are healthy

A card that is hot but running slow (cycle 2: GPU 0 at 88 °C / 615 MHz) will drag
a whole run and cost money. Check first; if one is bad, use another machine or
leave that card out.

```bash
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,clocks.sm --format=csv
```

- [ ] All GPUs: reasonable temperature, full power, full clocks, no thermal slowdown.

---

## 1. The split decision (correctness — do NOT skip)

**The problem:** in cycle 2, three insureds had documents on both sides of the
train/test line (U.S. Underwriters boppr, ShelterPoint, Main Street America
autob). That makes some "unseen" test docs not really unseen, and makes the
gate score look better than it is.

**What the code now does:** at the split step it will **refuse** any split that
puts one insured (same name, policy number, or mailing address) in two splits
(`assert_no_insured_across_splits`). You fix it by moving those documents so each
insured sits in one split (cycle‑3 rehearsal needed 5 such moves).

**Two things to watch:**
- [ ] Re‑splitting **changes the frozen test set** → this is a major call.
      **Confirm with Akshay before re‑freezing.**
- [ ] Preparation must run **on the pod**, not the laptop — Windows blocks the
      PyMuPDF library there, so the text checks silently do nothing.

---

## 2. Prove the built speed/parallel work on the pod

All of this is code‑complete and pushed but **never run on real hardware.** Verify
before trusting it in a `$21/h` cycle.

- [ ] **Eval‑set freeze in parallel with training** (`5248e28`). The orchestration
      (`scripts/full_cycle.sh`) passes local tests only. Do a dry check of the
      sequence — corpus build → background freeze → train → wait‑for‑freeze →
      package — before relying on it in a real run.
- [ ] **Training reads the pod's own corpus copy** instead of re‑downloading
      (`487da19`) — confirm training start‑up no longer re‑downloads the corpus.
- [ ] **Gate speedups + window batching** (`55b56f3`) — confirm the gate reads
      several documents per call (cycle 2 read one doc at a time, ~17 page
      groups/min/GPU) and starts up without the ~1 h idle re‑download.
- [ ] **Shared work queue** (`0c635b0`) — confirm a free GPU takes the next batch
      in checkpoint selection, calibration, and the gate (no GPU idle for hours
      behind a slow one).
- [ ] **Status screen** (`d08afe8`) — `scripts/cycle_status.sh` reads the worker
      logs and shows a "still waiting" line; confirm it no longer falsely says
      "may be stuck".

---

## 3. Measurements (code built, numbers not yet taken)

### 3a. Served accuracy check — the go/no‑go for the 95% target (B2)

Two GPUs side by side, ~30–60 min, ~$6–12 (one GPU = run them one after the other).

```bash
bash scripts/pod_run.sh start servednow -- env CUDA_VISIBLE_DEVICES=0 \
    /workspace/venv/bin/python -m scripts.check_served_fixes \
    --adapter adapters/scope/policy/v4 --out /workspace/logs/served_check_v4.json

bash scripts/pod_run.sh start servedsharp -- env CUDA_VISIBLE_DEVICES=1 \
    /workspace/venv/bin/python -m scripts.check_served_fixes \
    --adapter adapters/scope/policy/v4 --max-pixels 2483200 \
    --out /workspace/logs/served_check_v4_sharp.json
```

Read from the results:
- [ ] Run 1 (1.25 MP): v4 served accuracy on the 9 real test docs vs official
      0.7778 and offline estimate 0.8089.
      **≥0.90 = mostly data; 0.83–0.90 = data + model; <0.83 = design.**
- [ ] Merchants dates read correctly (07/11/25 → 07/11/2025, not 2555).
- [ ] Run 2 (2.48 MP): fewer misreads (WW534→WWS34, WCAI→WCA1)? how much slower?

### 3b. OCR — 2 servers per GPU

Placement code is built (`i % gpu_count`); the expected ~1.5× speed‑up is
**unmeasured.** Measure 4 vs 8 servers on the same documents before the cycle‑3
OCR run (cycle 2: 4 servers gave ~4.7 pages/s with GPU and CPU mostly idle).

- [ ] Measure pages/s with `MINERU_SERVERS=4` vs `MINERU_SERVERS=8`; keep the
      faster one only if it actually helps.

---

## 4. Still not built (optimization — time/cost, not correctness)

- [ ] **Training upload→download round trip.** Per the record, the training stage
      uploads the epoch files (~9.6 GB) to Blob and then downloads them straight
      back (~1 h+ before the first step). Item 8 fixed the read side; confirm on
      the pod whether the full round trip is gone, and if not, stage locally /
      skip the round trip. Not blocking — it's wasted time, not a wrong answer.

---

## Reminders that keep costing money if forgotten

- [ ] Always add `--keep-staging` to `package` — otherwise it deletes the pod's
      adapter copy (cost 3 checkpoints on 2026-11-02).
- [ ] Read a command's re‑run behaviour in the code **before** advising a
      stop/restart — the pod is ~$21/h.
- [ ] MinerU models may need re‑downloading on a new container —
      `bash scripts/mineru_server.sh check --pdf <small.pdf>` (the plain check
      misses it).

---

## Order to work in

1. GPU health (§0).
2. Served check §3a — this tells you whether cycle 3 should change data, model,
   or both. Do it early.
3. The parallel/speed verifications §2 as part of a small dry run.
4. OCR measurement §3b before the real OCR pass.
5. At split time: the split decision §1 — **stop and confirm with Akshay** before
   re‑freezing.
