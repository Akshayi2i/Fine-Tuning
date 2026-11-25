# Type errors frozen

- **Date:** 2026-11-24
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

The type errors older than CI’s check are frozen per file (scripts/check_type_errors.py, mypy 2.3.1): a new one fails, the count only goes down (271 when frozen; 268 after extract was split, 2026-11-24)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
