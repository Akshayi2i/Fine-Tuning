# Fast format profile

- **Date:** 2026-10-26
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

configs/shared/format.yaml profile: fast: page images first, the same windows for every section group, a compact field list and compact answers, 1,254,400 px (1,225 visual tokens) a page. Corpus, adapter and serving must share one profile; each refuses another (§6, §7)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
