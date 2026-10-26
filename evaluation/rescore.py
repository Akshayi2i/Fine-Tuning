"""Re-score a release's stored golden-eval answers under the scoring code running now.

Decision 2026-10-26 (#15). The gate compares a candidate with the promoted
release, and holds a continued cycle to its parent line by line, only on one
yardstick (``run_eval.yardstick_difference``): the same frozen set of the same
tenant, scored by the same code under the same settings. Most changes to the
scoring code move it. The candidate was then judged on its floors alone (no
non-inferiority, no improvement rule), and a cycle that could have continued
its parent started from the base - a full retrain - because the gate could not
compare it with its parent.

So the golden eval stores every reading's served answer (:func:`store_answers`,
at ``paths.eval_answers_key``): the extraction as scored, and beside it the
metadata ``run_eval.build_report`` reads - mode, source id, serving's
printed-values record, the text the values were held to, the repeat answer,
the field confidences, the schema verdict, synthetic, carrier and the rest -
with the frozen set's identity and a digest of each frozen gold. When only the
scoring code or the settings moved, :func:`rescore_report` reloads those
answers and the frozen golds - through ``golden_eval.load_golden_set``, as the
gate reads them, ``verified_label`` included - and runs the CURRENT
``build_report`` over them. The old model is never run again: the decision of
2026-10-22 not to run inference for an old release stands.

Limits:

* Re-scoring measures the OLD answers under the NEW scoring. It does not show
  what the old model would serve under new SERVING code: a new printed-values
  check, merge or post-process is not applied to old answers, which are scored
  as they were served then. The golden eval's own serving-side steps (the
  borrowed printed-values check of an image-only reading, and its confidence
  computed again on the page text) are part of what was stored, and are not
  run again either.
* A release scored before answers were stored has none. The first gate after
  this change still judges the candidate on its floors alone when the promoted
  release is on another yardstick, and a cycle whose parent was scored before
  it still starts from the base. From the release after, the stored answers
  are there.
* Refused, never approximated (``(None, why)``): another frozen set (the set
  frozen again), another tenant, answers missing, unreadable or incomplete - a
  reading that failed to extract, a frozen gold changed in place, or a reading
  the current settings ask for (a larger share, another repeat mode, a new
  mode) that was never made. Those keep the floors-only behaviour. A reading
  made then that the current settings no longer ask for (a smaller share) is
  left out, as the eval would not read it now.

Layout of the stored answers (gzip JSON lines): the first line a header -
``kind``, ``format``, ``version``, ``tenant_id``, the report's ``eval_set``
(frozen set, scoring fingerprint and settings it was scored under),
``lines_scored``, the settings it was read under, the infrastructure failures,
the reading count, and ``golds`` (``{source_id: digest of its frozen
golden.json}``) - then one line per reading, ``{"answer", "metadata"}``, in
the order the eval scored them.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import logging
import weakref
import zlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: The stored answers' layout. A file of another format is not read.
ANSWERS_FORMAT = 1
_KIND = "golden_eval_answers"

#: The settings the golden eval read under, as its report records them.
_SETTINGS: tuple[str, ...] = ("run_to_run_check", "run_to_run_mode", "run_to_run_share", "golden_noisy_share")

#: What the eval report records of the readings themselves, not of how they
#: were scored: carried onto the re-scored report as it was.
_CARRIED: tuple[str, ...] = (
    "infrastructure_failures", "lines_scored", "calibrated", "adapter_checkpoint",
    "release_measurements", "windowing_ceiling", "noisy_held_to",
)

#: Re-scored reports by client, so one gate re-scores a release once however
#: many of its checks ask (the baseline, the pairing, the line rule).
_MEMO: weakref.WeakKeyDictionary[BlobClient, dict[str, tuple[dict[str, Any] | None, str]]] = (
    weakref.WeakKeyDictionary())


class _Refused(Exception):
    """Why a release's stored answers cannot be re-scored."""


# --------------------------------------------------------------------------
# Storing
# --------------------------------------------------------------------------


def store_answers(
    client: BlobClient,
    report: dict[str, Any],
    triples: Sequence[tuple[Any, Any, dict[str, Any]]],
    documents: Sequence[Any],
    *,
    version: str,
    tenant_id: str | None,
) -> dict[str, Any]:
    """Write every reading's served answer - exactly what ``build_report``
    scored - to ``paths.eval_answers_key``; return what the eval report records
    of it (``answers_stored``): the key, the readings and documents, the bytes
    written (gzip) and before compression, and the digest of what was written.

    ``report`` is the eval report's body as evaluate_version built it (its
    ``eval_set``, ``lines_scored``, settings and failures are recorded with the
    answers); ``documents`` the frozen documents scored, whose golds are
    recorded by digest. Never fails the eval: answers that cannot be stored are
    recorded as not stored (``{"key": None, "error"}``), and a later gate on
    other scoring code judges this release's successor on its floors alone.
    """
    from common.adapter import ADAPTER

    try:
        key = paths.eval_answers_key(ADAPTER.run_id(version), tenant_id)
        failures = report.get("infrastructure_failures")
        header = {
            "kind": _KIND,
            "format": ANSWERS_FORMAT,
            "version": version,
            "tenant_id": paths._tenant(tenant_id),
            "corpus_version": report.get("corpus_version") or "",
            "eval_set": report.get("eval_set"),
            "lines_scored": report.get("lines_scored"),
            "settings": {name: report.get(name) for name in _SETTINGS},
            "infrastructure_failures": int(failures.get("count") or 0) if isinstance(failures, dict) else 0,
            "readings": len(triples),
            "golds": {doc.source_id: _gold_digest(client, tenant_id, doc.source_id) for doc in documents},
            "written_at": datetime.now(UTC).isoformat(),
        }
        lines = [_dumps(header)] + [_dumps({"answer": got, "metadata": metadata})
                                    for _expected, got, metadata in triples]
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        data = gzip.compress(raw, compresslevel=6, mtime=0)
        client.write_bytes(key, data)
    except Exception as exc:  # noqa: BLE001 - storing is for a later gate; it never fails this one
        log.warning("golden eval of %s: the answers could not be stored, so a later gate on other "
                    "scoring code cannot re-score them: %s", version, exc)
        return {"key": None, "error": f"{type(exc).__name__}: {exc}"}
    log.info("golden eval of %s: %d answers stored at %s (%d bytes, %d before gzip)",
             version, len(triples), key, len(data), len(raw))
    return {"key": key, "format": ANSWERS_FORMAT, "readings": len(triples),
            "documents": len(header["golds"]), "bytes": len(data), "raw_bytes": len(raw),
            "sha256": hashlib.sha256(data).hexdigest()}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_jsonable)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if callable(getattr(value, "as_dict", None)):
        return value.as_dict()
    return str(value)


def _gold_digest(client: BlobClient, tenant_id: str | None, source_id: str) -> str | None:
    """A digest of a frozen gold as frozen (its golden.json, read as JSON), or
    None when there is none: which gold the answers were scored against."""
    key = f"{paths.golden_eval_set_dir(tenant_id)}/{source_id}/golden.json"
    if not client.exists(key):
        return None
    canonical = json.dumps(client.read_json(key), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_answers(client: BlobClient, key: str, sha256: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(header, readings)`` of the answers stored at ``key``; refused
    (:class:`_Refused`) when they are missing, unreadable, not the ones the
    report recorded (``sha256``), of another format, or fewer than the header
    says."""
    if not client.exists(key):
        raise _Refused(f"its stored answers are missing ({key})")
    data = client.read_bytes(key)
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise _Refused(f"the answers at {key} are not the ones its report recorded (another digest)")
    try:
        rows = [json.loads(line) for line in gzip.decompress(data).decode("utf-8").splitlines() if line.strip()]
    except (OSError, EOFError, zlib.error, ValueError, UnicodeDecodeError) as exc:
        raise _Refused(f"its stored answers at {key} cannot be read ({type(exc).__name__}: {exc})") from exc
    if not rows or not isinstance(rows[0], dict) or rows[0].get("kind") != _KIND:
        raise _Refused(f"{key} holds no stored golden-eval answers")
    header, readings = rows[0], rows[1:]
    if header.get("format") != ANSWERS_FORMAT:
        raise _Refused(f"the answers at {key} are in format {header.get('format')!r}; this code reads "
                       f"{ANSWERS_FORMAT}")
    if len(readings) != header.get("readings") or not all(
            isinstance(r, dict) and "answer" in r and isinstance(r.get("metadata"), dict) for r in readings):
        raise _Refused(f"its stored answers at {key} are incomplete: {len(readings)} reading(s) of the "
                       f"{header.get('readings')} written")
    return header, readings


# --------------------------------------------------------------------------
# Re-scoring
# --------------------------------------------------------------------------


def rescore_report(
    client: BlobClient, version: str, tenant_id: str | None = None, *,
    eval_set: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """``version``'s eval report, scored again from its stored answers by the
    scoring code running now: ``(report, what was done)``, or ``(None, why
    not)``. The old model is not run.

    The report is what evaluate_version would write for these readings today:
    the current ``build_report`` over the stored answers and the frozen golds
    read as the gate reads them, under the current scoring fingerprint and
    settings (``eval_set``), with what the readings did carried over
    (infrastructure failures, lines scored, calibrated, adapter checkpoint,
    measurements), tagged ``rescored_from: {fingerprint, eval_set, at}`` - the
    yardstick it was scored on before, and when it was re-scored.

    ``eval_set``: the yardstick it is wanted on (a candidate's report's
    ``eval_set``); default the frozen set and the scoring code as they stand
    now. Refused when the re-scored report is still not on it.

    Refused - ``(None, why)``, and the caller keeps its floors-only behaviour -
    when the frozen set differs (frozen again), the tenant differs, or the
    stored answers are missing or incomplete (module docstring).
    """
    from common.adapter import ADAPTER
    from evaluation.run_eval import eval_set_record, scoring_fingerprint

    report_key = paths.eval_report(version, scope=ADAPTER.name)
    if not client.exists(report_key):
        return None, f"{version} has no eval report at {report_key}"
    stored = client.read_json(report_key)
    target = eval_set if eval_set is not None else eval_set_record(client, tenant_id, [])
    memo_key = json.dumps([version, paths._tenant(tenant_id), scoring_fingerprint(), target,
                           stored.get("answers_stored"), stored.get("generated_at")],
                          sort_keys=True, default=str)
    try:
        memo = _MEMO.setdefault(client, {})
    except TypeError:  # a client that cannot be weakly referenced is not remembered
        memo = {}
    if memo_key not in memo:
        try:
            memo[memo_key] = _rescore(client, version, tenant_id, report_key, stored, target)
        except _Refused as exc:
            memo[memo_key] = (None, f"{version}'s stored answers were not re-scored: {exc}")
        except Exception as exc:  # noqa: BLE001 - the caller falls back to what it did before re-scoring
            log.warning("re-scoring %s failed: %s", version, exc, exc_info=True)
            memo[memo_key] = (None, f"{version}'s stored answers were not re-scored: re-scoring failed "
                                    f"({type(exc).__name__}: {exc})")
    report, why = memo[memo_key]
    log.info("%s", why)
    return (copy.deepcopy(report) if report is not None else None), why


def rescored_or_none(
    client: BlobClient, version: str, tenant_id: str | None = None,
    eval_set: dict[str, Any] | None = None, *, notes: list[str] | None = None,
) -> dict[str, Any] | None:
    """``version``'s eval report on the yardstick of ``eval_set`` (a candidate's
    report's ``eval_set``; default: the frozen set and the scoring code as they
    stand now), or None when it cannot be had.

    Its own report when it is already on that yardstick; its stored answers
    re-scored (:func:`rescore_report`) when it is not; None when it has no
    report, or its answers cannot be re-scored. What was done, or why not, is
    added to ``notes``. The helper the gate's line rule calls for each run it
    holds a line to (the parent, the last run from the base), as the gate's
    baseline and the continuation do.
    """
    from common.adapter import ADAPTER
    from evaluation.run_eval import eval_set_record, yardstick_difference

    notes = notes if notes is not None else []
    key = paths.eval_report(version, scope=ADAPTER.name)
    if not client.exists(key):
        notes.append(f"{version} has no eval report at {key}")
        return None
    target = eval_set if eval_set is not None else eval_set_record(client, tenant_id, [])
    stored = client.read_json(key)
    different = yardstick_difference(stored.get("eval_set"), target, tenant_id)
    if different is None:
        return stored
    report, why = rescore_report(client, version, tenant_id, eval_set=target)
    notes.append(f"{version}'s eval report was on another yardstick ({different}). {why}")
    return report


def _rescore(
    client: BlobClient, version: str, tenant_id: str | None, report_key: str,
    stored: dict[str, Any], target: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    from common.adapter import ADAPTER
    from common.lob import single_line
    from evaluation import golden_eval
    from evaluation.run_eval import (
        TEXT_DONOR_MODE,
        build_report,
        eval_set_record,
        scoring_fingerprint,
        yardstick_difference,
    )

    tenant = paths._tenant(tenant_id)
    record = stored.get("answers_stored")
    if not isinstance(record, dict) or not record.get("key"):
        error = record.get("error") if isinstance(record, dict) else None
        raise _Refused(f"its eval report at {report_key} stores no answers ("
                       + (f"they could not be stored: {error})" if error else
                          "scored before the golden eval stored them, decision 2026-10-26)"))
    scored_on = stored.get("eval_set") or {}
    if scored_on.get("tenant_id") != tenant:
        raise _Refused(f"it was scored on tenant {scored_on.get('tenant_id')!r}'s frozen set, and this is "
                       f"tenant {tenant!r}'s: source ids are numbered per tenant")
    answers_key = paths.eval_answers_key(ADAPTER.run_id(version), tenant_id)
    if record["key"] != answers_key:
        raise _Refused(f"its report names answers at {record['key']}, not this tenant's ({answers_key})")
    header, rows = load_answers(client, answers_key, record.get("sha256"))
    if header.get("tenant_id") != tenant or header.get("version") != version:
        raise _Refused(f"the answers at {answers_key} are {header.get('version')!r} of tenant "
                       f"{header.get('tenant_id')!r}, not {version!r} of {tenant!r}")
    failed = int(header.get("infrastructure_failures") or 0)
    if failed or stored.get("gate_metrics_partial"):
        raise _Refused(f"its stored answers are incomplete: {failed or 'some'} reading(s) failed to extract "
                       "twice and were never scored")

    # The same frozen set: a set frozen again may hold another document or
    # another gold under one source id.
    now = eval_set_record(client, tenant_id, [])
    then = header.get("eval_set") or {}
    frozen_then = (then.get("frozen_at"), then.get("frozen_from_corpus"))
    frozen_now = (now.get("frozen_at"), now.get("frozen_from_corpus"))
    if not all(frozen_then) or frozen_then != frozen_now:
        raise _Refused(f"its answers were read on the set frozen {frozen_then[0]} from {frozen_then[1]}, and "
                       f"the set now is frozen {frozen_now[0]} from {frozen_now[1]}: the set was frozen "
                       "again, so one source id may be another document or another gold")

    # The golds, read as the gate reads them (verified_label under the running
    # label rules), narrowed to the lines the release was scored on.
    try:
        documents = golden_eval.load_golden_set(client, ADAPTER.doc_types, tenant_id=tenant_id)
    except golden_eval.GoldenEvalError as exc:
        raise _Refused(f"the frozen golds cannot be read as the gate reads them: {exc}") from exc
    lines = header.get("lines_scored")
    if lines is not None:
        documents = [d for d in documents if single_line(d.lob) in set(lines)]
    if not documents:
        raise _Refused(f"the frozen set holds no document of the lines it was scored on ({lines})")
    golds = header.get("golds") or {}
    for doc in documents:
        if doc.source_id not in golds:
            raise _Refused(f"its stored answers are incomplete: none for {doc.source_id}, which the gate "
                           "scores now")
        if golds[doc.source_id] != _gold_digest(client, tenant_id, doc.source_id):
            raise _Refused(f"{doc.source_id}'s frozen gold is not the one its answers were scored against "
                           "(changed in place since)")

    triples, settings = _readings(documents, rows)
    corpus_version = str(header.get("corpus_version") or stored.get("corpus_version") or "")
    report = build_report(version, triples, corpus_version=corpus_version)
    body = report.as_dict()
    for name in _CARRIED:
        if name in stored:
            body[name] = copy.deepcopy(stored[name])
    body.update(settings)
    body["documents_failed"] = 0
    body["repeats_failed"] = 0
    body["invention_coverage"] = {"borrowed_from": TEXT_DONOR_MODE, **report.invention_coverage()}
    body["eval_set"] = eval_set_record(client, tenant_id, documents)
    body.update(golden_eval.real_only_section(version, triples, corpus_version=corpus_version))
    # Against the release promoted when it was scored: not recomputed.
    body["field_delta_vs_promoted"] = None
    body["answers_stored"] = copy.deepcopy(record)
    body["rescored_from"] = {
        "fingerprint": scored_on.get("scoring_fingerprint"),
        "eval_set": copy.deepcopy(scored_on),
        "at": datetime.now(UTC).isoformat(),
        "answers": answers_key,
        "readings": len(triples),
        # Read then, and not asked for by the settings now (a smaller share).
        "readings_left_out": len(rows) - len(triples),
    }
    if why := yardstick_difference(body["eval_set"], target, tenant_id):
        raise _Refused(f"re-scored, it is still not on the yardstick asked for: {why}")
    done = (f"{version}'s {len(triples)} stored readings (scored under fingerprint "
            f"{scored_on.get('scoring_fingerprint')}) were re-scored under the scoring code running now "
            f"({scoring_fingerprint()}); the old model was not run again")
    return body, done


def _readings(documents: Sequence[Any], rows: Sequence[dict[str, Any]]
              ) -> tuple[list[tuple[Any, Any, dict[str, Any]]], dict[str, Any]]:
    """The ``(expected, got, metadata)`` triples the golden eval would score
    under the settings now, from the stored readings, in its order (document,
    then mode); and those settings. Refused when one it asks for was never made."""
    from evaluation import golden_eval
    from evaluation.run_eval import REPEAT_ANSWER

    repeat, repeat_mode = golden_eval.run_to_run_check(), golden_eval.run_to_run_mode()
    noisy_share, repeat_share = golden_eval.golden_noisy_share(), golden_eval.run_to_run_share()
    stored = {(str(row["metadata"].get("source_id")), str(row["metadata"].get("modality_mode"))): row
              for row in rows}
    triples: list[tuple[Any, Any, dict[str, Any]]] = []
    for doc in documents:
        for mode in golden_eval.GOLDEN_MODES:
            if mode == "noisy_ocr_image" and not golden_eval.in_share(doc.source_id, noisy_share, "noisy"):
                continue
            row = stored.get((doc.source_id, mode))
            if row is None:
                raise _Refused(f"its stored answers are incomplete: no {mode} reading of {doc.source_id}, "
                               "which the golden eval reads now (another share or mode than then)")
            metadata = dict(row["metadata"])
            if repeat and mode == repeat_mode and golden_eval.in_share(doc.source_id, repeat_share, "repeat"):
                if REPEAT_ANSWER not in metadata:
                    raise _Refused(f"its stored answers are incomplete: {doc.source_id} was not read twice in "
                                   f"{mode}, which run_to_run_agreement compares now (another repeat mode or "
                                   "share than then)")
            else:
                metadata.pop(REPEAT_ANSWER, None)
            triples.append((doc.golden, row["answer"], metadata))
    settings = {"run_to_run_check": repeat, "run_to_run_mode": repeat_mode,
                "run_to_run_share": repeat_share, "golden_noisy_share": noisy_share}
    return triples, settings
