# Workers per GPU (review S2)

- **Date:** 2026-11-25
- **Decided by:** the user, on the options written up for the review
- **Where it is written:** azure_service/DEPLOY.md (the worker)

## The decision

One worker per GPU, written into the deployment guide as a limit.

## Options refused

Separate GPUs for the model and for OCR (no turn-taking, a second GPU's cost); a queue in front of each server so the servers own the order (new code at the server side).

## What would reopen it

A measured load the one worker per GPU cannot carry.
