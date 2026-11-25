# No merge, no quantization

- **Date:** 2026-10-23
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

Nothing is merged or quantized: the release is its adapter, served as a LoRA on the bf16 base, and calibration and the gate score it that way, as it is served (§12.3, §13a, §14.4)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
