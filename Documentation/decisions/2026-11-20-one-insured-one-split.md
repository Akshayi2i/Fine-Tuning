# One insured, one split

- **Date:** 2026-11-20
- **Decided by:** the user (recorded in architecture v2.9 §0e)
- **Where it is written:** Documentation/Architecture/finetuning-architecture-v2.9.docx §0e and the sections it names

## The decision

A delivered split that puts one insured’s real documents (same name, policy number or mailing address) in two splits is refused; with a frozen test set the same insured’s other documents and their copies stay out of training (pipeline_dag.assert_no_insured_across_splits, exclude_eval_families). Cycle 2’s split had one such family (§8.2)

## Options refused

Not recorded when it was decided.

## What would reopen it

Not recorded when it was decided.
