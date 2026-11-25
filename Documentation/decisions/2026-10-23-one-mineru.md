# One MinerU

- **Date:** 2026-10-23
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

The MinerU 4 API (4.0.12, tier basic) reads the training corpus and every served document alike; each task’s version and tier are checked, and a corpus is read by one MinerU pin (§8a). On the training pod one server runs per GPU, all at the same pin, each document sent whole to the server with the fewest in progress (data_pipeline/ocr/served_ocr.PooledServedEngine, 2026-11-14)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
