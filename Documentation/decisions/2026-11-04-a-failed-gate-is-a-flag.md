# A failed gate is a flag

- **Date:** 2026-11-04
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

A gate failed on the model’s quality no longer stops package or promotion: it is recorded as “Gate failed due to …” on the gate decision, the release and its promotion record. A gate that could not decide still stops; shadow --promote-by promotes a clean pass only (§15.5)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
