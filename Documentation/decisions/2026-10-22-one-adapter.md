# One adapter

- **Date:** 2026-10-22
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

One policy adapter, with no training scopes and no layout families. Each fine-tuning cycle takes the next version number (v1, v2 …, one sequence for every tenant), which names its run (policy-v3), its adapter and its release. Fideon SPEC_09’s policy adapter groups are not followed (§4.1, §12)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
