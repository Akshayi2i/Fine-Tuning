# Where monitoring numbers go (review S4)

- **Date:** 2026-11-25
- **Decided by:** the user, on the options written up for the review
- **Where it is written:** azure_service/telemetry.py

## The decision

Logs only until the first deployment: five numbers per finished job (seconds per stage, queue age, windows not read, values removed as not printed, guards that could not run), one log line each, through an interface a backend can be plugged into. Never a document's content, name or values.

## Options refused

Azure Monitor / Application Insights now; OpenTelemetry now.

## What would reopen it

The first deployment, or the team's choice of a backend.
