"""A gate stopped half way is picked up where it stopped (evaluation.reading_cache)."""

from __future__ import annotations

import pytest

from evaluation import golden_eval
from evaluation.reading_cache import FIRST, REPEAT, ReadingCache, cache_key
from tests.test_invention_and_agreement import _fake_extract, _golden_docs


class Stop(BaseException):
    """The pod going away: not an Exception, so nothing in the gate catches and retries it."""


def _run(cache, **kwargs):
    return golden_eval.evaluate(_golden_docs(), object(), object(), "/tmp", run_to_run=True,
                                repeat_mode="image_only", cache=cache, **kwargs)


def _stop_after(monkeypatch, calls, count):
    import serving.pipeline as pipeline

    real = pipeline.extract

    def extract(request, *args, **kwargs):
        if len(calls) >= count:
            raise Stop()
        return real(request, *args, **kwargs)

    monkeypatch.setattr(pipeline, "extract", extract)


def test_a_second_run_reads_nothing_and_scores_the_same(monkeypatch, tmp_path):
    calls: list = []
    _fake_extract(monkeypatch, calls)
    cache = ReadingCache(tmp_path, "k")
    first = _run(cache)
    done = len(calls)
    assert done == 8                                        # 2 documents x 3 modes, and 2 repeats

    again = _run(ReadingCache(tmp_path, "k"))
    assert len(calls) == done                               # the model was not asked again
    assert [m for _e, _g, m in again] == [m for _e, _g, m in first]
    assert [g for _e, g, _m in again] == [g for _e, g, _m in first]


def test_a_stop_in_the_first_pass_costs_only_the_readings_not_yet_made(monkeypatch, tmp_path):
    calls: list = []
    _fake_extract(monkeypatch, calls)
    _stop_after(monkeypatch, calls, 4)                      # p1's three modes and p2's first, then the stop
    with pytest.raises(Stop):
        _run(ReadingCache(tmp_path, "k"))
    assert len(calls) == 4

    calls.clear()
    _fake_extract(monkeypatch, calls)
    triples = _run(ReadingCache(tmp_path, "k"))
    assert [(s, m, o) for s, m, o in calls] == [
        ("p2", "noisy_ocr_image", "planned"), ("p2", "image_only", "planned"),
        ("p1", "image_only", "reversed"), ("p2", "image_only", "reversed"),
    ]
    assert len(triples) == 6


def test_a_stop_in_the_repeat_pass_reads_only_the_repeats_not_yet_made(monkeypatch, tmp_path):
    calls: list = []
    _fake_extract(monkeypatch, calls)
    _stop_after(monkeypatch, calls, 7)                      # all six readings, one repeat, then the stop
    with pytest.raises(Stop):
        _run(ReadingCache(tmp_path, "k"))

    calls.clear()
    _fake_extract(monkeypatch, calls)
    triples = _run(ReadingCache(tmp_path, "k"))
    assert calls == [("p2", "image_only", "reversed")]
    from evaluation.run_eval import REPEAT_ANSWER

    assert sum(1 for _e, _g, m in triples if REPEAT_ANSWER in m) == 2      # both repeats are in the scores


def test_a_reading_that_failed_is_not_kept_so_it_is_tried_again(monkeypatch, tmp_path):
    calls: list = []
    _fake_extract(monkeypatch, calls, fail_repeat=True)
    cache = ReadingCache(tmp_path, "k")
    first = _run(cache)
    assert len(first.infrastructure_failures) == 2
    assert cache.get("p1", "image_only", REPEAT) is None    # the failed repeat was not kept
    assert cache.get("p1", "image_only", FIRST) is not None

    calls.clear()
    _fake_extract(monkeypatch, calls)
    second = _run(ReadingCache(tmp_path, "k"))
    assert not second.infrastructure_failures
    assert calls == [("p1", "image_only", "reversed"), ("p2", "image_only", "reversed")]


def test_another_key_reads_everything_fresh(monkeypatch, tmp_path):
    calls: list = []
    _fake_extract(monkeypatch, calls)
    _run(ReadingCache(tmp_path, cache_key(checkpoint="checkpoint-857")))
    calls.clear()
    _run(ReadingCache(tmp_path, cache_key(checkpoint="checkpoint-688")))
    assert len(calls) == 8


def test_a_torn_file_is_a_miss_not_a_failure(tmp_path, caplog):
    cache = ReadingCache(tmp_path, "k")
    cache.put("p1", "image_only", {"a": 1})
    cache._file("p1", "image_only", FIRST).write_bytes(b"not a pickle")
    with caplog.at_level("WARNING"):
        assert cache.get("p1", "image_only") is None
    assert any("could not be read" in r.getMessage() for r in caplog.records)


def test_the_key_changes_with_anything_a_reading_depends_on():
    base = dict(version="v3", checkpoint="checkpoint-857", commit="abc", settings="(True,)")
    assert cache_key(**base) == cache_key(**base)
    for name, value in (("version", "v4"), ("checkpoint", "checkpoint-688"), ("commit", "def"),
                        ("settings", "(False,)")):
        assert cache_key(**{**base, name: value}) != cache_key(**base)


def test_the_cache_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("FIDEON_GATE_CACHE", "0")
    assert golden_eval.reading_cache_for(None, None, "v3", "v3", _golden_docs(), object(), "c", None, None,
                                         settings=()) is None
