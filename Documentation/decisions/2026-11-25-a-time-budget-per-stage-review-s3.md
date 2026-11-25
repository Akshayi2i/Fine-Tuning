# A time budget per stage (review S3)

- **Date:** 2026-11-25
- **Decided by:** the user, on the options written up for the review
- **Where it is written:** this page; the README's 2-minute target

## The decision

Measure first: the speed fixes of 2026-11-20 run on a pod, then budgets per page-count band are set from those numbers and latency_check reports each stage against them. No gating on time until the user says so.

## Options refused

One target only (a 200-page scan's fields in 120 s); budgets set now without a measurement (the last measured: 400 s for 224 pages, v3, 2026-11-13).

## What would reopen it

The pod measurement, which sets the bands.
