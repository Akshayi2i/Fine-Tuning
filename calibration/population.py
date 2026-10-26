"""Which validation readings the calibrators and thresholds are fitted on.

**Standard profile: every mode, pooled** - as before: one logistic regression
per field type over the calibration half's ocr_plus_image, noisy_ocr_image and
image_only readings together, thresholds on the threshold half's.

**Fast profile: image_only, the mode production reads** (fields first: the
page images alone, the fields then checked on MinerU's text). A calibrator
fitted mostly on readings that were sent page text describes those readings;
production serves none of them. So under the fast profile the calibrators and
thresholds are fitted on image_only readings alone, and - as production checks
them once the text arrives (``calibration.post_text``) - their text agreement
is computed on their document's page text, borrowed from its ocr_plus_image
readings (the same pinned MinerU text, as the golden gate borrows it). A
document with no such reading keeps text agreement absent, and is counted.

Either way, how well the fitted calibrators hold is reported per mode and per
field type (:func:`ece_by_mode`): the expected calibration error on the
threshold half of each mode's readings, fitted on or not.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: The mode production reads.
PRODUCTION_MODE = "image_only"

#: The mode whose page text an image_only reading borrows (as the golden gate's
#: ``run_eval.TEXT_DONOR_MODE``).
TEXT_DONOR_MODE = "ocr_plus_image"

#: The halves of validation (arch v2.1 §8.2).
HALVES = ("calibration", "threshold")


def fitting_modes(profile: str | None = None) -> tuple[str, ...] | None:
    """The reading modes calibration is fitted on under ``profile`` (default:
    the configured one, ``common.config.format_profile``): image_only under
    fast, ``None`` - every mode, pooled - under standard."""
    from common.config import FAST_PROFILE, format_profile

    return (PRODUCTION_MODE,) if (profile or format_profile()) == FAST_PROFILE else None


def samples_by_mode(
    generations: Sequence[Any], *, borrow_text: bool | None = None, profile: str | None = None,
) -> dict[str, dict[str, list[tuple[Any, bool]]]]:
    """``{mode: {half: [(features, correct), ...]}}``: the validation readings'
    labelled features (``evaluation.validation_generation.calibration_samples``,
    unchanged) for each reading mode, each feature vector tagged with its mode.

    ``borrow_text`` (default: on under the fast profile, off under standard, as
    before): an image_only reading's text agreement is computed on its
    document's page text, gathered from the document's ocr_plus_image readings
    - what production checks those fields on once the text arrives."""
    from calibration.features import refresh_text_features
    from evaluation.validation_generation import calibration_samples

    borrow = fitting_modes(profile) is not None if borrow_text is None else borrow_text
    by_mode: dict[str, list[Any]] = {}
    for generation in generations:
        by_mode.setdefault(str(generation.row.get("modality_mode") or TEXT_DONOR_MODE), []).append(generation)
    donor = _document_texts(by_mode.get(TEXT_DONOR_MODE, [])) if borrow else {}

    out: dict[str, dict[str, list[tuple[Any, bool]]]] = {}
    for mode, members in sorted(by_mode.items()):
        halves: dict[str, list[tuple[Any, bool]]] = {half: [] for half in HALVES}
        if mode == PRODUCTION_MODE and borrow:
            # One document at a time, so each one's features are grounded on its
            # own text: calibration_samples returns them pooled.
            for source, readings in sorted(_by_source(members).items()):
                texts = donor.get(source)
                for half, rows in calibration_samples(readings).items():
                    for features, correct in rows:
                        if texts:
                            refresh_text_features(features, texts)
                        halves.setdefault(half, []).append((features, correct))
            missing = sorted(set(_by_source(members)) - set(donor))
            if missing:
                log.warning("calibration: %d image_only document(s) have no %s reading to borrow page "
                            "text from, so their text agreement stays absent: %s",
                            len(missing), TEXT_DONOR_MODE, missing[:5])
        else:
            for half, rows in calibration_samples(members).items():
                halves.setdefault(half, []).extend(rows)
        for rows in halves.values():
            for features, _correct in rows:
                features.modality_mode = mode
        out[mode] = halves
    return out


def production_samples(
    by_mode: Mapping[str, Mapping[str, Sequence[tuple[Any, bool]]]], profile: str | None = None,
) -> dict[str, list[tuple[Any, bool]]]:
    """``{half: [(features, correct), ...]}`` - what is fitted on under
    ``profile``: image_only alone under fast, every mode pooled under standard."""
    modes = fitting_modes(profile)
    chosen = sorted(by_mode) if modes is None else [m for m in modes if m in by_mode]
    return {half: [row for mode in chosen for row in by_mode[mode].get(half, [])] for half in HALVES}


@dataclass
class FittedCalibration:
    """A format's calibrators and thresholds, and what they were fitted on."""

    calibrators: Any
    thresholds: Any
    report: dict[str, Any] = field(default_factory=dict)


def fit_release_calibration(
    by_mode: Mapping[str, Mapping[str, Sequence[tuple[Any, bool]]]],
    *,
    release_id: str,
    serving_format: str,
    profile: str | None = None,
    targets: dict[str, float] | None = None,
    min_instances: int | None = None,
) -> FittedCalibration:
    """Calibrators on the calibration half and thresholds on the threshold half
    (arch v2.1 §8.2) of the readings ``profile`` fits on
    (:func:`production_samples`), with a report: the profile, the modes fitted
    on, the rows of each half, the auto-accept error on the threshold half, and
    the calibration error per mode and per field type (:func:`ece_by_mode`)."""
    from calibration.feature_calibrator import MIN_INSTANCES, fit_calibrators
    from calibration.thresholds import auto_accept_error_rate, fit_thresholds
    from common.config import format_profile

    profile = profile or format_profile()
    samples = production_samples(by_mode, profile)
    calibrators = fit_calibrators(samples["calibration"], release_id=release_id, serving_format=serving_format,
                                  min_instances=MIN_INSTANCES if min_instances is None else min_instances)
    scored_by_type = scored_by_field_type(calibrators, samples["threshold"])
    thresholds = fit_thresholds(scored_by_type, release_id=release_id, serving_format=serving_format,
                                targets=targets)
    modes = fitting_modes(profile)
    report = {
        "profile": profile,
        "fitted_on_modes": list(modes) if modes is not None else sorted(by_mode),
        "rows": {half: len(rows) for half, rows in samples.items()},
        "rows_by_mode": {mode: {half: len(rows) for half, rows in halves.items()}
                         for mode, halves in sorted(by_mode.items())},
        "auto_accept_error_rate": round(auto_accept_error_rate(scored_by_type, thresholds), 4),
        "ece_by_mode": ece_by_mode(calibrators, by_mode),
    }
    return FittedCalibration(calibrators, thresholds, report)


def scored_by_field_type(calibrators: Any, rows: Iterable[tuple[Any, bool]]) -> dict[str, list[tuple[float, bool]]]:
    """``(calibrated confidence, correct)`` per field type, for the fields a
    calibrator scores."""
    out: dict[str, list[tuple[float, bool]]] = {}
    for features, correct in rows:
        confidence = calibrators.predict(features)
        if confidence is not None:
            out.setdefault(features.field_type, []).append((confidence, bool(correct)))
    return out


def ece_by_mode(
    calibrators: Any, by_mode: Mapping[str, Mapping[str, Sequence[tuple[Any, bool]]]], *, half: str = "threshold",
) -> dict[str, dict[str, Any]]:
    """``{mode: {"all": ece, "fields": n, "by_field_type": {type: ece}}}`` on
    ``half`` of each mode's readings: whether the confidence means what it says
    in each mode production or the gate reads. ``all`` is None where the
    calibrators scored nothing."""
    from evaluation.metrics.coverage_metrics import expected_calibration_error

    def ece(pairs: list[tuple[float, bool]]) -> float | None:
        return round(expected_calibration_error([c for c, _ in pairs], [ok for _, ok in pairs]), 4) if pairs else None

    out: dict[str, dict[str, Any]] = {}
    for mode, halves in sorted(by_mode.items()):
        scored = scored_by_field_type(calibrators, halves.get(half, []))
        every = [pair for pairs in scored.values() for pair in pairs]
        out[mode] = {"all": ece(every), "fields": len(every),
                     "by_field_type": {t: ece(pairs) for t, pairs in sorted(scored.items())}}
    return out


def _by_source(generations: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for index, generation in enumerate(generations):
        grouped.setdefault(str(generation.row.get("source_id") or f"#{index}"), []).append(generation)
    return grouped


def _document_texts(generations: Iterable[Any]) -> dict[str, dict[int, str]]:
    """Each document's page text by page, from every window of its readings."""
    texts: dict[str, dict[int, str]] = {}
    for generation in generations:
        source = generation.row.get("source_id")
        pages = getattr(generation, "ocr_pages", None)
        if source and pages:
            texts.setdefault(str(source), {}).update({int(p): str(t or "") for p, t in pages.items()})
    return {source: pages for source, pages in texts.items() if any(t.strip() for t in pages.values())}
