# What is gated, and who promotes

- **Date:** 2026-10-26
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

served_field_match (real documents read image_only, as served) is the primary metric, floored at 0.85 beside field_normalized_match; real_ocr_plus_image_accuracy floored at 0.85; the production tier rests on the served metric’s lower bound. package writes the release gated and never promotes; a person promotes it after a shadow check on the server path (§0d, §12, §15.5)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
