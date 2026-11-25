# Forms and batching

- **Date:** 2026-10-23
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

On the common-model lines the forms list is read by a window group of its own (forms). Serving sends up to 96 windows per vLLM call, the longest expected answers first (§7b)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
