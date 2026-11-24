# Extraction service: what happens when something fails

For the people who run the service. Each case says what the caller sees, what to
check, and what to do. Every case here is proven by a test with in-memory
stand-ins (named under each case). None has been rehearsed on real Azure yet: the
service has not been deployed. Rehearse each case once on the first deployment.

Settings named below are in `azure_service/settings.py` and are set by environment
variables (`MAX_ATTEMPTS`, `RETRY_DELAY_S`, `FIELDS_DEADLINE_S`).

## Two commands

```
python -m azure_service.jobs failed  [--tenant T]   # jobs that were given up: no result
python -m azure_service.jobs partial [--tenant T]   # jobs served with pages whose answer could not be read
```

Both print one line per job: id, tenant, status, attempts, times, and the error or
the unread windows. Neither prints a document's name or any extracted value.

## Case 1: the model server stops answering

Down before the job starts, or stopping partway through it: either way the job is
**never served with pages missing**. A part of the document that does not reach the
model (connection refused, the answer cut off, a 5xx from the server or its proxy)
is sent once more; failing again, the whole job goes back on the queue and is read
again from the start after `RETRY_DELAY_S` (60 s). Its status shows `queued`, the
attempt count, and the error ("... never reached the model ..."). After
`MAX_ATTEMPTS` (3) deliveries it is `failed`.

- What to check: `python -m azure_service.jobs failed` once the server is back.
- What to do: submit the listed documents again.

Test: `test_a_model_server_that_cannot_be_reached_puts_the_job_back` (both cases).

## Case 2: the worker stops

**Between the fields and the full text.** The fields were already served
(`fields_ready`). The job's lease runs out and the queue hands it to another
worker. That worker resumes at the full text, so the model does not read the
document again. A late write from the stopped worker cannot overwrite the newer
state. Test: `test_a_worker_that_dies_after_the_fields_leaves_the_full_text_to_the_next`.

**While reading the fields.** The next delivery reads the fields again from the
start. Test: `test_a_worker_that_dies_during_the_fields_has_them_read_again`.

**Shut down on purpose (scaling in, a new release).** The worker hands its jobs back
at once, so the next replica takes them without waiting for the lease to run out.
Test: `test_a_stopping_worker_hands_its_jobs_back_at_once`.

**Fields take too long.** After `FIELDS_DEADLINE_S` (1,800 s) the worker stops
renewing the job's lease. The queue hands the job out again, which counts as an
attempt, and the reason is written to its status.
Test: `test_fields_past_their_deadline_are_given_up_by_this_delivery_and_given_out_again`.

## Case 3: a PDF that fails on every delivery

After `MAX_ATTEMPTS` deliveries the job is `failed` with "given up after 3 attempts".
It is never read again and is removed from the queue. Only its status record
remains. A webhook, if one was given, gets `extraction.failed` with no result link.
`/result` answers 409.

- What to check: `python -m azure_service.jobs failed`.
- What to do: read the error. If the PDF is damaged, ask the caller for a new copy.
  If the error names our code, raise it as a bug with the job id. Never re-submit it
  unchanged: it will fail the same way.

Test: `test_a_poison_job_is_failed_after_its_attempts_and_never_loops`.

## Case 4: the model answered, but some pages could not be read

The model was reached, but its answer for part of the document was not readable
JSON, even one page at a time (a page it could not make sense of). Those pages are
dropped. The job **finishes as `done` with no error**; the gap shows in the result's
`output.review_flags` as `window_failed:<kind>:p<page>`, and `output.schema_valid`
is `false`.

- What the caller must do: treat any `window_failed:` flag as an incomplete result.
- What to check: `python -m azure_service.jobs partial`.
- What to do: send the listed pages to a person; trying again reads them the same
  way (the model's reading is fixed, not random).

Test: `test_an_answer_the_model_wrote_badly_is_served_flagged_and_listed`.

## Other endings that need a person

- **A line no release serves**, with the base-model fallback off: failed at once
  (one attempt), with the reason. Test: `test_a_job_the_worker_can_never_read_fails_at_once`.
- **MinerU's full text fails for good:** the fields were served at `fields_ready`,
  then the job is `failed`. There is no final result.
  Test: `test_a_job_whose_mineru_text_fails_for_good_is_failed_not_done`.
- **The prompt-length check could not run:** each window's stats carry
  `prompt_check_error`, and the worker logs one warning per reason. The job is not
  affected, but a server reading pages at the wrong size would go unnoticed until it
  is fixed. Test: `tests/test_production_image_imports.py`.

## Which release and code read a result

Every result and status record has a `pins` block: the base model and revision,
the release, the prompt hash, the window plan, the format profile, the MinerU
version and tier, the calibration, and the line's schema version. To roll back, the
promote command prints the adapter **and** the code commit the release needs
(`python -m orchestration.run promote`): serve the old release from that commit.
