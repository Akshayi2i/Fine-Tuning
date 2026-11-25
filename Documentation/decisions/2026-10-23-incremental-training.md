# Incremental training

- **Date:** 2026-10-23
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

Each cycle continues the tenant’s promoted adapter on new and existing data, always on the original base model, never on a merged one. It starts from the base with --fresh, when nothing is promoted, or when the parent cannot be continued (§4.2, §12.1)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
