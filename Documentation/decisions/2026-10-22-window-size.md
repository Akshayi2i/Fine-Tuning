# Window size

- **Date:** 2026-10-22
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

Each policy window is sized by its pages’ own OCR text, not by page count alone: a window that would not fit its call is split, the same way in training and serving. Every page is still read (§7b)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
