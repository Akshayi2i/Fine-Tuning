"""Turn a synthetic-data delivery into import bundles: one folder per document.

The generator (fideon_synth ``scripts/build_dataset.py``) delivers

    <input>/Train|Val|Test/pdfs/<name>.pdf
    <input>/Train|Val|Test/gold json/<name>.json
    <input>/manifest.csv      split, lob, carrier, source, sample, kind, pdf, gold, ok, ...

and the importer and the audit read one folder per document:

    <out>/<name>/document.pdf, golden.json, metadata.json

``metadata.json`` is written from the manifest row, nothing guessed:

    lob          the line, which selects the canonical schema
    synthetic    true for a generated twin, false for the source itself
    template_id  the source document it was made from - the family the delivery
                 was split by, so a source and its twins stay on one side
    split        the delivery's own train/val/test, which the corpus build then
                 uses as given (split_groups.assign_delivered_splits)
    reviewer_id, review_date, draft_backend
                 the review record: the row's own columns for a real document;
                 for a twin, its seed's (a twin is reviewed only when its seed
                 was), with ``review_inherited_from`` naming the seed. A seed with
                 no record leaves its twins none: the importer refuses every gold
                 without one (export_golden_labels.review_problem, decision
                 2026-10-22). A twin's own review columns are never read.

Rows outside the scope's lines (flood, for personal_lines), rows the generator
flagged (``ok`` false) and rows whose files are missing are skipped and counted.

    python -m data_pipeline.ingestion.prepare_bundles --input "data/training data" --out data/bundles

Files are hard-linked by default: instant, and no second copy of 16 GB on the
same drive. Where a link is impossible (another drive) they are copied.

Documents the audit rejected are listed in an exclusion file (``--exclude``,
default ``data/bundle_exclusions.csv``: ``document,reason``), so a re-run keeps
them out - and removes their folders if an earlier run made them - instead of
the fix living only in a folder someone deleted by hand.

**A SPEC_21 delivery** (``split_manifest.csv``: split, lob, carrier, seed, twin,
digital, scanned, gold) delivers each twin twice - a digital PDF and a scanned
rendering of it - with one gold. A row's review columns (``reviewer_id``,
``review_date``, ``draft_backend``) record its SEED's review, which the seed's
document and every twin of it carry; every row of one seed must state the same
record. Each render becomes a document of its own
(``render_mode`` digital or scanned, ``sample`` the twin's number, the seed as
the family), both reading one gold - one document when the two are the same
file, as a scanned seed's twins are. The gold the bundles hold is corrected
where the delivered one cannot be trained on as it is (:func:`corrected_gold`),
each change listed in ``<out>/corrections.csv``; the delivery is never changed
(a corrected gold is written as a new file, never through a hard link to the
delivered one). A gold still outside its line's schema after that is left out,
with the reason. ``--dry-run`` reports all of it and writes nothing.

A re-run leaves ``<out>`` holding exactly what it bundles of the delivery: a
folder an earlier run made for a document this run leaves out (its gold, its
files, its line, ``--originals none``) is removed, and ``corrections.csv`` is
rewritten. Folders of other deliveries are not touched.

**The seeds themselves** - the real documents the twins were made from - are
delivered apart (``--originals``, default ``data/source data``: ``pdfs/`` and
``gold json/``, ``<line>/<carrier>/<seed>.pdf`` and ``<seed>.gold.json``). Each
becomes a real document (``synthetic`` false) in its twins' family and split,
read from the twins' manifest, so a seed and its twins are never on two sides;
its gold gets the same corrections.

**Real documents alone** - no twins, no manifest, no split - of one line
(``--lob``), laid out as ``<input>/PDFs/<carrier>/<name>.pdf`` with
``<input>/Gold JSON/<carrier>/<name>.gold.json`` (the CGL source data):

    python -m data_pipeline.ingestion.prepare_bundles --input "data/CGL source data" --lob gl \\
        --scope casualty_fleet --out data/cgl_bundles

Each pair becomes ``<line>__<carrier>__<name>``, its gold corrected as above. Its
review record comes from ``<input>/reviews.csv`` (``document,reviewer_id,
review_date,draft_backend``; ``document`` the gold's path under ``Gold JSON/``
without ``.gold.json``, or the folder name): none there, none in the bundle. A
family is one insured: a declarations page, the policy and its renewals stay on
one side. The split is drawn as the corpus build draws one
(:func:`~data_pipeline.dataset_builder.split_groups.assign_group_splits`) and
delivered in the metadata, so it is the same on every re-run and a smoke batch
can be drawn from it; ``<out>/split.csv`` lists it for review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

DEFAULT_OUT = Path("data") / "bundles"
#: Where a SPEC_21 delivery's seed documents are: pdfs/ and gold json/.
DEFAULT_ORIGINALS = Path("data") / "source data"
DEFAULT_EXCLUSIONS = Path("data") / "bundle_exclusions.csv"
#: ``source_prefix,lob,reason``: a line the generator's folder got wrong, e.g. a
#: carrier's classic-car policies filed under personal_auto.
DEFAULT_LOB_OVERRIDES = Path("data") / "bundle_lob_overrides.csv"
REQUIRED_COLUMNS = ("split", "lob", "source", "kind", "pdf", "gold", "ok")

#: A SPEC_21 delivery's manifest: one row per twin, both renders and the gold.
TWIN_MANIFEST = "split_manifest.csv"
#: Real documents with no twins (:func:`prepare_original_bundles`).
ORIGINAL_PDFS = "PDFs"
ORIGINAL_GOLDS = "Gold JSON"
TWIN_COLUMNS = ("split", "lob", "carrier", "seed", "twin", "digital", "scanned", "gold")

#: A manifest's review columns: who reviewed a gold, when, and what drafted it.
REVIEW_COLUMNS = ("reviewer_id", "review_date", "draft_backend")
#: Real documents alone: their review records, beside PDFs/ and Gold JSON/.
REVIEWS_FILE = "reviews.csv"

#: Line names a delivery misspells, in its folders and its manifest's lob column.
DELIVERY_SPELLINGS = {"motorcyle": "motorcycle"}

#: Coverage codes in no list of their line, and the listed code each one is
#: (the overlays' changelogs; see the file).
RECODES = Path(__file__).resolve().parents[2] / "configs" / "coverage_code_recodes.yaml"


class BundleError(RuntimeError):
    """Raised when a delivery cannot be read at all."""


@dataclass
class PrepareReport:
    out: Path
    written: Counter = field(default_factory=Counter)          # (split, kind) -> documents
    skipped: Counter = field(default_factory=Counter)          # reason -> rows
    skipped_lines: Counter = field(default_factory=Counter)    # lob -> rows outside the scope
    relined: Counter = field(default_factory=Counter)          # "old -> new" -> rows given another line
    linked: int = 0
    copied: int = 0
    #: Folders an earlier run made for documents this run leaves out.
    removed: int = 0
    #: Documents bundled with no review record: the importer refuses them.
    unreviewed: int = 0
    #: Gold corrections by kind, and one row per change: (document, kind, detail).
    corrections: Counter = field(default_factory=Counter)
    correction_rows: list = field(default_factory=list)
    dry_run: bool = False

    @property
    def documents(self) -> int:
        return sum(self.written.values())

    def describe(self) -> str:
        lines = [f"{self.documents} document folder(s) in {self.out}"]
        for split in ("train", "val", "test"):
            parts = {kind: n for (s, kind), n in self.written.items() if s == split}
            if parts:
                lines.append(f"  {split:5} " + ", ".join(f"{n} {kind}" for kind, n in sorted(parts.items())))
        for reason, n in sorted(self.skipped.items()):
            lines.append(f"  skipped {n}: {reason}")
        if self.skipped_lines:
            lines.append("  outside the scope: " + ", ".join(f"{k} {v}" for k, v in sorted(self.skipped_lines.items())))
        if self.relined:
            lines.append("  line overridden: " + ", ".join(f"{k} ({v})" for k, v in sorted(self.relined.items())))
        for kind, n in sorted(self.corrections.items()):
            lines.append(f"  gold corrected ({kind}): {n}")
        lines.append(f"  files linked {self.linked}, copied {self.copied}")
        if self.unreviewed:
            lines.append(f"  no review record: {self.unreviewed} document(s) - the importer refuses them "
                         "until reviewer_id and review_date are recorded (a twin's: its seed's)")
        if self.removed:
            verb = "would remove" if self.dry_run else "removed"
            lines.append(f"  {verb} {self.removed} folder(s) an earlier run made for documents now left out")
        if self.dry_run:
            lines.append("  dry run: nothing was written")
        return "\n".join(lines)


def _place(source: Path, dest: Path, mode: str, report: PrepareReport) -> None:
    """Put ``source`` at ``dest``: a hard link where possible, else a copy (or a move)."""
    if dest.exists():
        if dest.stat().st_size == source.stat().st_size:
            return
        dest.unlink()
    if mode == "move":
        shutil.move(str(source), str(dest))
        report.copied += 1
        return
    if mode == "link":
        try:
            os.link(source, dest)
            report.linked += 1
            return
        except OSError:
            pass                                    # another drive, or no link support
    shutil.copy2(source, dest)
    report.copied += 1


def _write_new(path: Path, text: str) -> None:
    """Write ``text`` as a NEW file at ``path``, never through the file there. An
    earlier run may have hard-linked the delivered gold there, and a batch drawn
    from the bundles (the smoke batch) is hard links to their files: writing in
    place overwrote the delivery's own gold, or changed a batch's metadata under
    its old gold."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_gold(path: Path, gold: dict) -> None:
    """Write a corrected gold as a new file at ``path`` (:func:`_write_new`)."""
    _write_new(path, json.dumps(gold, indent=2, ensure_ascii=False))


def prepare_bundles(
    delivery: Path,
    out: Path = DEFAULT_OUT,
    *,
    lines: frozenset[str] | None = None,
    mode: str = "link",
    exclusions: dict[str, str] | None = None,
    lob_overrides: list[tuple[str, str]] | None = None,
    dry_run: bool = False,
) -> PrepareReport:
    """Write ``out/<name>/{document.pdf, golden.json, metadata.json}`` for each usable row
    (with ``dry_run``, count them and write nothing)."""
    if (delivery / TWIN_MANIFEST).is_file():
        return prepare_twin_bundles(delivery, out, lines=lines, mode=mode, exclusions=exclusions,
                                    dry_run=dry_run)
    manifest = delivery / "manifest.csv"
    if not manifest.is_file():
        raise BundleError(f"no manifest.csv in {delivery}")
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing_columns = [c for c in REQUIRED_COLUMNS if rows and c not in rows[0]]
    if not rows or missing_columns:
        raise BundleError(f"{manifest} is empty or lacks columns {missing_columns}")
    if mode not in ("link", "copy", "move"):
        raise BundleError(f"mode {mode!r} is not link, copy or move")

    report = PrepareReport(out=out, dry_run=dry_run)
    seen: dict[str, str] = {}
    # Each seed's review record, from its own row, for its twins to inherit.
    reviews = seed_review_records(rows)
    for row in rows:
        lob = row["lob"].strip()
        for prefix, new_lob in lob_overrides or ():
            if row["source"].strip().startswith(prefix):
                report.relined[f"{lob} -> {new_lob}"] += 1
                lob = new_lob
                break
        from common.lob import merge_line

        # Classic auto is read as personal auto (common.lob.MERGED_LINES).
        lob = str(merge_line(lob))
        if lines is not None and lob not in lines:
            report.skipped_lines[lob] += 1
            continue
        if row["ok"].strip().lower() != "true":
            report.skipped["flagged by the generator (ok is false)"] += 1
            continue
        pdf, gold = delivery / row["pdf"], delivery / row["gold"]
        if not pdf.is_file() or not gold.is_file():
            report.skipped["files listed in the manifest are missing"] += 1
            continue
        name = Path(row["pdf"]).stem
        if exclusions and name in exclusions:
            report.skipped[f"excluded: {exclusions[name]}"] += 1
            if (out / name).is_dir():
                report.removed += 1
                if not dry_run:
                    shutil.rmtree(out / name)      # made by an earlier run: take it out
            continue
        if name in seen:
            raise BundleError(f"two manifest rows name the document {name!r} ({seen[name]}, {row['pdf']})")
        seen[name] = row["pdf"]
        split = row["split"].strip().split("/")[-1].lower()
        if split not in ("train", "val", "test"):
            raise BundleError(f"{name}: split {row['split']!r} is not Train, Val or Test")
        kind = row["kind"].strip().lower()
        template_id = template_of(row)

        metadata = {
            "lob": lob,
            "synthetic": kind == "synthetic",
            "template_id": template_id,
            "split": split,
            "carrier": row.get("carrier") or None,
            "source_system": "fideon_synth",
            "sample": int(row["sample"]) if (row.get("sample") or "").strip().isdigit() else None,
            # The twin's render mode (SPEC_21 §7.8: native, scanned_from_digital,
            # ...): the per-seed twin cap counts each mode apart.
            "render_mode": render_mode(row, gold),
            # A real document's review record is its own row's; a twin's, its seed's.
            **(inherited_review(reviews.get(template_id), template_id) if kind == "synthetic"
               else row_review(row)),
        }
        report.written[(split, "synthetic" if metadata["synthetic"] else "real")] += 1
        report.unreviewed += _unreviewed(metadata)
        if dry_run:
            continue
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        _place(pdf, folder / "document.pdf", mode, report)
        _place(gold, folder / "golden.json", mode, report)
        _write_new(folder / "metadata.json", json.dumps(metadata, indent=2))
    return report


def delivery_line(lob: str) -> str:
    """A delivery's line name as this repository reads it: its misspellings put
    right, classic auto read as personal auto (common.lob.MERGED_LINES), and an
    enum or L1 spelling read as the schema's name (``cgl`` is ``gl``)."""
    from common.lob import merge_line
    from common.schemas import LOB_SCHEMA_ALIASES

    name = lob.strip().lower()
    line = str(merge_line(DELIVERY_SPELLINGS.get(name, name)))
    return LOB_SCHEMA_ALIASES.get(line, line)


def prepare_twin_bundles(
    delivery: Path,
    out: Path = DEFAULT_OUT,
    *,
    lines: frozenset[str] | None = None,
    mode: str = "link",
    exclusions: dict[str, str] | None = None,
    dry_run: bool = False,
    originals: Path | None = None,
) -> PrepareReport:
    """Bundles from a SPEC_21 delivery: two documents per twin, one corrected gold,
    and with ``originals`` each seed as a real document of its twins' family."""
    import re

    manifest = delivery / TWIN_MANIFEST
    with manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing_columns = [c for c in TWIN_COLUMNS if rows and c not in rows[0]]
    if not rows or missing_columns:
        raise BundleError(f"{manifest} is empty or lacks columns {missing_columns}")
    if mode not in ("link", "copy", "move"):
        raise BundleError(f"mode {mode!r} is not link, copy or move")

    recodes = read_recodes()
    report = PrepareReport(out=out, dry_run=dry_run)
    seen: dict[str, str] = {}
    # Each seed's review record, from its twins' rows (twin_seed_reviews).
    reviews = twin_seed_reviews(rows)
    for row in rows:
        lob = delivery_line(row["lob"])
        if lines is not None and lob not in lines:
            report.skipped_lines[lob] += 2
            continue
        gold_path = delivery / row["gold"]
        renders = (("digital", delivery / row["digital"]), ("scanned", delivery / row["scanned"]))
        if not gold_path.is_file() or not all(pdf.is_file() for _, pdf in renders):
            report.skipped["files listed in the manifest are missing"] += 2
            continue
        split = row["split"].strip().lower()
        if split not in ("train", "val", "test"):
            raise BundleError(f"{row['twin']}: split {row['split']!r} is not train, val or test")

        gold, notes, problem = corrected_gold(
            json.loads(gold_path.read_text(encoding="utf-8")), lob,
            carrier=row["carrier"], text_pdf=delivery / row["digital"], recodes=recodes)
        if problem:
            report.skipped[f"gold outside its line's schema after correction: {problem}"] += 2
            report.correction_rows.append((row["twin"], "left out", problem))
            continue
        if _same_file(renders[0][1], renders[1][1]):
            # A scanned seed's twin is a scan in both folders, byte for byte: one
            # document, a scan, not the same PDF twice under two render modes.
            notes.append(("renders identical", f"{renders[0][1].name} is the scan; one document"))
            renders = renders[1:]
        changed = _gold_changed(notes)
        for kind, detail in notes:
            report.corrections[kind] += 1
            report.correction_rows.append((row["twin"], kind, detail))

        number = re.search(r"(\d+)$", row["twin"].strip())
        for index, (render, pdf) in enumerate(renders):
            name = pdf.stem
            if exclusions and name in exclusions:
                report.skipped[f"excluded: {exclusions[name]}"] += 1
                continue
            if name in seen:
                raise BundleError(f"two manifest rows name the document {name!r} ({seen[name]}, {pdf})")
            seen[name] = str(pdf)
            metadata = {
                "lob": lob,
                "synthetic": True,
                # The seed: every twin of it, in both renders, is one family.
                "template_id": f"{lob}/{row['seed'].strip()}",
                "split": split,
                "carrier": row["carrier"].strip() or None,
                "source_system": "fideon_synth",
                "sample": int(number.group(1)) if number else None,
                "render_mode": render,
                # Values (fields or extra fields) no page prints: the corpus build
                # leaves out the windows holding them (label_rules.PRINTED_NOWHERE).
                "unprinted_values": _unprinted(notes),
                # The seed's review record: a twin is reviewed only when its seed was.
                **inherited_review(reviews.get(row["seed"].strip()), f"{lob}/{row['seed'].strip()}"),
            }
            report.written[(split, "synthetic")] += 1
            report.unreviewed += _unreviewed(metadata)
            if dry_run:
                continue
            folder = out / name
            folder.mkdir(parents=True, exist_ok=True)
            _place(pdf, folder / "document.pdf", mode, report)
            if changed:
                _write_gold(folder / "golden.json", gold)
            else:
                # The two renders share one delivered gold: moved with the last,
                # copied for the one before (moving it first left nothing to place).
                last = index == len(renders) - 1
                _place(gold_path, folder / "golden.json", "copy" if mode == "move" and not last else mode,
                       report)
            _write_new(folder / "metadata.json", json.dumps(metadata, indent=2))

    if originals is not None:
        _bundle_originals(originals, rows, out, lines=lines, mode=mode, exclusions=exclusions,
                          dry_run=dry_run, recodes=recodes, report=report, seen=seen, reviews=reviews)

    # What an earlier run bundled of this delivery and this one leaves out.
    for name in sorted(_delivery_names(rows) - set(seen)):
        if (out / name).is_dir():
            report.removed += 1
            if not dry_run:
                shutil.rmtree(out / name)

    if not dry_run:
        # Rewritten every run, if only with its header: an earlier run's list
        # would describe golds this run did not write.
        out.mkdir(parents=True, exist_ok=True)
        with (out / "corrections.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["twin", "correction", "detail"])
            writer.writerows(report.correction_rows)
    return report


def _gold_changed(notes: list[tuple[str, str]]) -> bool:
    """Whether the corrections changed the gold itself: a note that the renders
    are one file, or that a value is printed nowhere, leaves it as delivered."""
    from data_pipeline.ingestion.label_rules import PRINTED_NOWHERE

    return any(kind not in ("renders identical", PRINTED_NOWHERE) for kind, _ in notes)


def _unprinted(notes: list[tuple[str, str]]) -> list[str]:
    from data_pipeline.ingestion.label_rules import PRINTED_NOWHERE

    return [detail for kind, detail in notes if kind == PRINTED_NOWHERE]


def _original_name(lob: str, carrier: str, seed: str) -> str:
    """The folder of a seed's own document."""
    return f"{lob}__{carrier.strip()}__{seed.strip()}__original"


def _delivery_names(rows: list[dict[str, str]]) -> set[str]:
    """Every folder a SPEC_21 delivery's bundles can have: both renders of each
    twin, and each seed's original."""
    names: set[str] = set()
    for row in rows:
        names.update((Path(row["digital"]).stem, Path(row["scanned"]).stem,
                      _original_name(delivery_line(row["lob"]), row["carrier"], row["seed"])))
    return names


def _same_file(a: Path, b: Path) -> bool:
    """Whether two files hold the same bytes (size first, then a hash)."""
    import hashlib

    if a.stat().st_size != b.stat().st_size:
        return False
    return hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def _bundle_originals(
    originals: Path, rows: list[dict[str, str]], out: Path, *, lines: frozenset[str] | None, mode: str,
    exclusions: dict[str, str] | None, dry_run: bool, recodes: dict[str, dict[str, str]],
    report: PrepareReport, seen: dict[str, str], reviews: dict[str, dict[str, str | None]] | None = None,
) -> None:
    """Each seed of the twins' manifest as a real document, in its twins' family and split,
    with the review record its twins' rows state (``reviews``, :func:`twin_seed_reviews`)."""
    pdfs = {p.stem: p for p in (originals / "pdfs").rglob("*.pdf")}
    golds = {p.name[: -len(".gold.json")]: p for p in (originals / "gold json").rglob("*.gold.json")}
    seeds: dict[str, dict[str, str]] = {}
    for row in rows:
        seed = row["seed"].strip()
        known = seeds.setdefault(seed, row)
        if known["split"].strip().lower() != row["split"].strip().lower():
            raise BundleError(f"seed {seed!r} has twins in {known['split']!r} and {row['split']!r}")
    for seed, row in sorted(seeds.items()):
        lob = delivery_line(row["lob"])
        if lines is not None and lob not in lines:
            report.skipped_lines[lob] += 1
            continue
        pdf, gold_path = pdfs.get(seed), golds.get(seed)
        if pdf is None or gold_path is None:
            report.skipped["seed document or its gold not in the originals"] += 1
            continue
        name = _original_name(lob, row["carrier"], seed)
        gold, notes, problem = corrected_gold(json.loads(gold_path.read_text(encoding="utf-8")), lob,
                                              carrier=row["carrier"], text_pdf=pdf, recodes=recodes)
        if problem:
            report.skipped[f"gold outside its line's schema after correction: {problem}"] += 1
            report.correction_rows.append((name, "left out", problem))
            continue
        if exclusions and name in exclusions:
            report.skipped[f"excluded: {exclusions[name]}"] += 1
            continue
        if name in seen:
            raise BundleError(f"two documents are named {name!r}")
        seen[name] = str(pdf)
        for kind, detail in notes:
            report.corrections[kind] += 1
            report.correction_rows.append((name, kind, detail))
        split = row["split"].strip().lower()
        metadata = {
            "lob": lob,
            "synthetic": False,
            "template_id": f"{lob}/{seed}",
            "split": split,
            "carrier": row["carrier"].strip() or None,
            "source_system": "spec21_seed",
            "sample": None,
            "render_mode": None,
            "unprinted_values": _unprinted(notes),
            **((reviews or {}).get(seed) or row_review({})),
        }
        report.written[(split, "real")] += 1
        report.unreviewed += _unreviewed(metadata)
        if dry_run:
            continue
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        _place(pdf, folder / "document.pdf", mode, report)
        if _gold_changed(notes):
            _write_gold(folder / "golden.json", gold)
        else:
            _place(gold_path, folder / "golden.json", mode, report)
        _write_new(folder / "metadata.json", json.dumps(metadata, indent=2))


def _slug(text: str) -> str:
    """``Johnson & Johnson`` -> ``johnson_johnson``: a folder name with no spaces
    or symbols, for a blob path and a source id's provenance."""
    import re

    return "_".join(re.findall(r"[a-z0-9]+", text.lower()))


#: Words an insured's name is printed with or without: ``Rivera Fabrication, LLC``
#: and ``RIVERA FABRICATION`` are one insured.
_NAME_NOISE = frozenset({"the", "and", "llc", "l", "c", "inc", "incorporated", "corp", "corporation", "co",
                         "company", "ltd", "limited", "lp", "llp", "pllc", "pc", "dba"})


def _insured_key(name: str) -> str:
    """An insured's name with case, punctuation and legal-form words left out."""
    return "_".join(word for word in _slug(name).split("_") if word not in _NAME_NOISE)


def _insured_family(gold: dict, line: str, fallback: str) -> str:
    """One insured's documents, as one family: ``<line>/<hash of the name>``.

    The declarations, the policy and its renewals print the same insured, and a
    model scored on one of them after training on another is scored on a
    document it has as good as seen. Hashed so a family id does not spell the
    name out.
    """
    import hashlib

    name = (gold.get("named_insured") or {}).get("primary_name")
    name = name.get("raw") if isinstance(name, dict) else name
    key = _insured_key(str(name)) if name else ""
    return f"{line}/{hashlib.sha256((key or fallback).encode()).hexdigest()[:10]}"


def _printed_carrier_key(gold: dict) -> str | None:
    """The carrier the gold names, case and punctuation aside - a folder is not
    always one carrier (a broker's folder holds several)."""
    name = (gold.get("carrier") or {}).get("name")
    name = name.get("raw") if isinstance(name, dict) else name
    return (_slug(str(name)) or None) if name else None


def prepare_original_bundles(
    source: Path,
    out: Path = DEFAULT_OUT,
    *,
    lob: str,
    lines: frozenset[str] | None = None,
    mode: str = "link",
    exclusions: dict[str, str] | None = None,
    dry_run: bool = False,
    seed: int = 42,
    frozen: bool = False,
) -> PrepareReport:
    """Bundles from real documents of one line with no twins and no split.

    ``<source>/PDFs/<carrier>/<name>.pdf`` with ``<source>/Gold JSON/<carrier>/
    <name>.gold.json``. Each gold is corrected as a seed's is (:func:`corrected_gold`);
    one outside the line's schema, or a package policy, is left out with the
    reason. The split is drawn by insured family and written into each
    document's metadata, so the corpus build uses it as delivered.

    A test document whose carrier no training document has is marked
    ``held_out_carrier``: the gate reports those documents apart, as the model's
    reading of carriers it never saw. ``frozen`` once the tenant's eval set is
    frozen: the frozen set is the test set and takes no new document, so every
    document is drawn into train or val.

    Never ``move``: a re-run leaves out a document whose PDF is gone, and removes
    its folder - the only copy of a moved PDF.
    """
    from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits

    line = delivery_line(lob)
    if lines is not None and line not in lines:
        raise BundleError(f"{line} is outside the scope's lines {sorted(lines)}; pass that line's scope")
    if mode not in ("link", "copy"):
        raise BundleError(f"mode {mode!r}: real documents are bundled by link or copy, never moved - "
                          "a re-run would remove the folder holding the only copy of a moved PDF")
    pdf_root, gold_root = source / ORIGINAL_PDFS, source / ORIGINAL_GOLDS
    pdfs = {p.relative_to(pdf_root).with_suffix(""): p for p in pdf_root.rglob("*.pdf")}
    golds = {p.relative_to(gold_root).with_name(p.name[: -len(".gold.json")]): p
             for p in gold_root.rglob("*.gold.json")}
    if not pdfs:
        raise BundleError(f"no PDFs under {pdf_root}")

    recodes = read_recodes()
    report = PrepareReport(out=out, dry_run=dry_run)
    reviews = read_reviews(source / REVIEWS_FILE)
    names = {key: f"{line}__{_slug(str(key.parent))}__{_slug(key.name)}" for key in pdfs.keys() | golds.keys()}
    clashes = Counter(names.values())
    if duplicate := sorted(n for n, count in clashes.items() if count > 1):
        raise BundleError(f"two documents would share the folder name(s) {duplicate}")

    kept: dict[str, tuple[Path, dict, list[tuple[str, str]], Path]] = {}
    for key in sorted(pdfs.keys() | golds.keys()):
        name = names[key]
        if key not in golds or key not in pdfs:
            report.skipped["a PDF without its gold" if key in pdfs else "a gold without its PDF"] += 1
            continue
        if exclusions and name in exclusions:
            report.skipped[f"excluded: {exclusions[name]}"] += 1
            continue
        # No carrier from the folder: a folder is not always one carrier (a
        # broker's holds several), and a gold missing its carrier would be
        # filled with the broker's name.
        gold, notes, problem = corrected_gold(
            json.loads(golds[key].read_text(encoding="utf-8")), line,
            carrier="", text_pdf=pdfs[key], recodes=recodes)
        if problem:
            report.skipped[f"gold left out: {problem.split(':', 1)[0]}"] += 1
            report.correction_rows.append((name, "left out", problem))
            continue
        kept[name] = (pdfs[key], gold, notes, golds[key])

    # One family per insured, split as the corpus build splits a tenant with no
    # delivered split - so the documents are where the build would have put them.
    family = {name: _insured_family(gold, line, name) for name, (_, gold, _, _) in kept.items()}
    members: dict[str, list[str]] = {}
    for name, group in sorted(family.items()):
        members.setdefault(group, []).append(name)
    records = [GroupRecord(group_id=group, doc_type="policy", source_ids=docs, line=line,
                           carrier=_printed_carrier_key(kept[docs[0]][1]))
               for group, docs in sorted(members.items())]
    assignment = assign_group_splits({"policy": records}, seed=seed, with_test=not frozen,
                                     hold_out_carriers=not frozen).assignment
    # Carriers the model trains on; a test document of any other is its reading
    # of a carrier it never saw.
    # A placeholder carrier ("N/A", "Unknown", none at all) is no carrier, so
    # never an unseen one (split_groups.is_placeholder_carrier).
    from data_pipeline.dataset_builder.split_groups import is_placeholder_carrier

    trained = {_printed_carrier_key(gold) for name, (_, gold, _, _) in kept.items()
               if assignment[family[name]] == "train"}
    unseen = {name for name, (_, gold, _, _) in kept.items()
              if assignment[family[name]] == "test" and _printed_carrier_key(gold) not in trained
              and not is_placeholder_carrier(_printed_carrier_key(gold))}
    key_of = {name: key for key, name in names.items()}

    for name, (pdf, gold, notes, gold_path) in sorted(kept.items()):
        split = assignment[family[name]]
        for kind, detail in notes:
            report.corrections[kind] += 1
            report.correction_rows.append((name, kind, detail))
        metadata = {
            "lob": line,
            "synthetic": False,
            "template_id": family[name],
            "split": split,
            # Left to the gold's own carrier: a folder is not always one carrier.
            "carrier": None,
            "source_system": "original",
            "sample": None,
            "render_mode": None,
            "unprinted_values": _unprinted(notes),
            "held_out_carrier": name in unseen,
            # From reviews.csv, by the gold's path under Gold JSON/ or the folder name.
            **(reviews.get(key_of[name].as_posix()) or reviews.get(name) or row_review({})),
        }
        report.written[(split, "real")] += 1
        report.unreviewed += _unreviewed(metadata)
        if dry_run:
            continue
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        _place(pdf, folder / "document.pdf", mode, report)
        if _gold_changed(notes):
            _write_gold(folder / "golden.json", gold)
        else:
            _place(gold_path, folder / "golden.json", mode, report)
        _write_new(folder / "metadata.json", json.dumps(metadata, indent=2))

    # What an earlier run bundled of this source and this one leaves out.
    for name in sorted(set(names.values()) - set(kept)):
        if (out / name).is_dir():
            report.removed += 1
            if not dry_run:
                shutil.rmtree(out / name)

    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "corrections.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["document", "correction", "detail"])
            writer.writerows(report.correction_rows)
        with (out / "split.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["document", "family", "split", "held_out_carrier"])
            writer.writerows((name, family[name], assignment[family[name]], name in unseen)
                             for name in sorted(kept))
    return report


def template_of(row: dict[str, str]) -> str:
    """A manifest.csv row's family: the source document it was made from, without its suffix."""
    return str(PurePosixPath(row["source"].strip()).with_suffix(""))


def row_review(row: dict[str, str]) -> dict[str, str | None]:
    """The review record a row states (:data:`REVIEW_COLUMNS`), a blank cell as none."""
    return {key: (row.get(key) or "").strip() or None for key in REVIEW_COLUMNS}


def inherited_review(seed_record: dict[str, str | None] | None, seed: str) -> dict[str, str | None]:
    """A twin's review record: its seed's, naming the seed it came from
    (``review_inherited_from``) - or none at all when the seed has no real one,
    so the importer refuses the twin with its seed (decision 2026-10-22)."""
    from data_pipeline.labeling.export_golden_labels import has_review_record

    if seed_record and has_review_record(seed_record):
        return {**seed_record, "review_inherited_from": seed}
    return {**row_review({}), "draft_backend": (seed_record or {}).get("draft_backend"),
            "review_inherited_from": None}


def seed_review_records(rows: list[dict[str, str]]) -> dict[str, dict[str, str | None]]:
    """``{family: review record}`` from a manifest.csv delivery: each seed's own
    row (``kind`` not synthetic), the record its twins inherit."""
    records: dict[str, dict[str, str | None]] = {}
    for row in rows:
        if (row.get("kind") or "").strip().lower() != "synthetic":
            records.setdefault(template_of(row), row_review(row))
    return records


def twin_seed_reviews(rows: list[dict[str, str]]) -> dict[str, dict[str, str | None]]:
    """``{seed: review record}`` from a SPEC_21 manifest. A row is a twin and
    names its seed, and its review columns record the seed's review - a twin is
    reviewed only through its seed - so every row of one seed states one record."""
    records: dict[str, dict[str, str | None]] = {}
    for row in rows:
        seed, record = row["seed"].strip(), row_review(row)
        known = records.setdefault(seed, record)
        if known != record:
            raise BundleError(f"seed {seed!r}: its twins' rows state two review records ({known} and {record})")
    return records


def read_reviews(path: Path) -> dict[str, dict[str, str | None]]:
    """``{document: review record}`` from a ``document,reviewer_id,review_date,draft_backend``
    CSV (real documents alone, :data:`REVIEWS_FILE`); empty when there is none."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and not {"document", "reviewer_id", "review_date"} <= set(rows[0]):
        raise BundleError(f"{path} needs the columns document,reviewer_id,review_date[,draft_backend]")
    return {r["document"].strip().replace("\\", "/"): row_review(r) for r in rows if (r["document"] or "").strip()}


def _unreviewed(metadata: dict) -> int:
    """1 when a bundle's metadata holds no real review record: the importer refuses it."""
    from data_pipeline.labeling.export_golden_labels import has_review_record

    return int(not has_review_record(metadata))


def read_recodes(path: Path = RECODES) -> dict[str, dict[str, str]]:
    """``{line: {code in no list: listed code}}``.

    Two sources: the codes a line's own coverage-code file says a listed code
    replaces (``replaces_codes`` in ``<line>.coverage_codes.yaml`` - gl 3.0.0
    moved GL_ADDITIONAL_INSURED to the shared X_ADDITIONAL_INSURED), and this
    repository's table at ``path``, which wins where both name a code.
    """
    import yaml

    from common.schemas import CANONICAL_DIR

    out: dict[str, dict[str, str]] = {}
    for codes_file in sorted(CANONICAL_DIR.glob("*.coverage_codes.yaml")):
        data = yaml.safe_load(codes_file.read_text(encoding="utf-8")) or {}
        line = str(data.get("lob") or codes_file.name.split(".", 1)[0])
        for entry in data.get("codes") or []:
            for old in (entry.get("replaces_codes") or []) if isinstance(entry, dict) else []:
                out.setdefault(line, {})[str(old)] = str(entry["code"])
    if path.is_file():
        for line, codes in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items():
            out.setdefault(str(line), {}).update({str(k): str(v) for k, v in (codes or {}).items()})
    return out


def _amount(value: object) -> float | None:
    parsed = value.get("parsed") if isinstance(value, dict) else value
    try:
        return round(float(parsed), 2) if parsed is not None else None
    except (TypeError, ValueError):
        return None


def corrected_gold(
    gold: dict, lob: str, *, carrier: str, text_pdf: Path, recodes: dict[str, dict[str, str]],
) -> tuple[dict, list[tuple[str, str]], str | None]:
    """``(gold, [(kind, detail)], problem)``: the gold a bundle holds.

    For a SPEC_21 common-model line, three corrections, each only where the
    delivered gold cannot be trained on as it is, then the label rules
    (``data_pipeline.ingestion.label_rules``; a value printed nowhere is listed
    as a :data:`~data_pipeline.ingestion.label_rules.PRINTED_NOWHERE` note):

    * **coverage code** in no list of the line, re-coded to the listed code the
      line's code file says replaces it, or the overlays' changelogs give for it
      (:func:`read_recodes`):
      the decoder only writes listed codes, so a target with another one is an
      answer the served model can never produce;
    * **premium written twice**: a ``premium.items`` entry for a coverage whose
      own ``premium`` holds the same amount - or, for a code on several rows
      (one per vehicle), their sum - is dropped, so a coverage's premium has one
      home (``coverages[].premium``), the schema review's proposal. An item whose
      coverage row has no premium, or that names no coverage row, is the only
      place that premium is written, and stays;
    * **carrier missing**, where the delivery's carrier is printed in the
      document: ``carrier.name`` is filled with it as printed, citing the pages
      that print it (All State's policies print only the brand, "Allstate").

    ``problem`` is the first schema error left after them - or, on a
    common-model line, a package policy (two parts, or a part of another line)
    - and such a gold is not bundled.
    """
    import copy

    from common.schemas import is_common_model, iter_validation_errors, load_schema
    from data_pipeline.ingestion.label_rules import label_rules_apply

    notes: list[tuple[str, str]] = []
    # Where the label rules hold (label_rules_apply): the importers, the corpus
    # build and the gate read a gold under them on exactly these lines.
    if label_rules_apply("policy", lob):
        gold = copy.deepcopy(gold)
        listed = set(load_schema("policy", None, lob).get("fideon:coverage_codes") or [])
        table = recodes.get(lob, {})
        premium = gold.get("premium") if isinstance(gold.get("premium"), dict) else {}
        for where, rows in (("coverages", gold.get("coverages") or []),
                            ("premium.items", premium.get("items") or [])):
            for index, row in enumerate(rows):
                code = row.get("coverage_code") if isinstance(row, dict) else None
                if isinstance(code, str) and code not in listed and code in table:
                    row["coverage_code"] = table[code]
                    notes.append(("coverage code", f"{where}[{index}] {code} -> {table[code]}"))

        # Each coverage row's premium, and per code the sum of its rows' premiums:
        # an item holding either is the same money written a second time.
        by_code: dict[object, list[float]] = {}
        for c in gold.get("coverages") or []:
            if isinstance(c, dict) and _amount(c.get("premium")) is not None:
                by_code.setdefault(c.get("coverage_code"), []).append(_amount(c.get("premium")))
        own = {(code, amount) for code, amounts in by_code.items() for amount in amounts}
        own |= {(code, round(sum(amounts), 2)) for code, amounts in by_code.items() if len(amounts) > 1}
        if premium.get("items"):
            kept = []
            for item in premium["items"]:
                if (isinstance(item, dict) and item.get("unit_type") == "coverage"
                        and (item.get("coverage_code"), _amount(item.get("amount"))) in own):
                    notes.append(("premium written twice",
                                  f"dropped premium.items {item.get('coverage_code')} {_amount(item.get('amount'))}"))
                    continue
                kept.append(item)
            premium["items"] = kept

        if not gold.get("carrier") and carrier.strip():
            printed = printed_carrier(text_pdf, carrier)
            if printed:
                raw, pages = printed
                gold["carrier"] = {"name": {"raw": raw, "parsed": raw,
                                            "confidence": {"score": 1.0, "source": "deterministic"},
                                            "page_ref": pages, "flagged": False}}
                notes.append(("carrier missing", f"carrier.name = {raw!r} as printed on page(s) {pages}"))

        # The agreed conventions (data_pipeline.ingestion.label_rules): each value
        # once, citing every page that prints it; extra fields only for values
        # with no field of their own; numbers only where printed as numbers.
        from data_pipeline.ingestion.label_rules import apply_label_rules, page_texts, scanned_pages

        notes.extend(apply_label_rules(gold, page_texts(text_pdf), scanned=scanned_pages(text_pdf)))

    errors = list(iter_validation_errors(gold, "policy", None, lob))
    if not errors and is_common_model("policy", None, lob):
        # A package policy (a general liability part and an inland marine part)
        # lists every line and is read against the fallback: filed under one
        # common-model line it is a target the corpus build refuses, and one
        # refused target stops the whole build.
        from common.canonical import CanonicalLabelError
        from data_pipeline.dataset_builder.policy_windows import with_own_part

        try:
            with_own_part(gold, lob)
        except CanonicalLabelError as exc:
            errors = [f"package policy: {str(exc).split(', but', 1)[0]}"]
    return gold, notes, (errors[0][:200] if errors else None)


def printed_carrier(pdf: Path, carrier: str) -> tuple[str, list[int]] | None:
    """The delivery's carrier as the document prints it, and the pages it is on.

    ``carrier`` is the manifest's name for it (``all_state``); it matches
    "Allstate" or "All State" in the visible text. None when no page prints it.
    """
    import re

    import pymupdf

    tokens = [t for t in re.split(r"[\s_]+", carrier.strip()) if t]
    if not tokens:
        return None
    pattern = re.compile(r"\b" + r"[\s-]*".join(map(re.escape, tokens)) + r"\b", re.IGNORECASE)
    raw, pages = None, []
    with pymupdf.open(pdf) as doc:
        for number, page in enumerate(doc, start=1):
            found = pattern.search(page.get_text())
            if found:
                raw = raw or found.group(0)
                pages.append(number)
    return (raw, pages) if raw else None


def render_mode(row: dict[str, str], gold: Path) -> str | None:
    """A twin's render mode: the manifest's ``mode`` column (SPEC_21 manifest.csv),
    else its gold's ``fideon:provenance.mode``; None when neither says."""
    value = (row.get("mode") or row.get("render_mode") or "").strip()
    if value:
        return value.lower()
    try:
        provenance = json.loads(gold.read_text(encoding="utf-8")).get("fideon:provenance") or {}
    except (OSError, ValueError, AttributeError):
        return None
    mode = provenance.get("mode") if isinstance(provenance, dict) else None
    return str(mode).strip().lower() if isinstance(mode, str) and mode.strip() else None


def read_exclusions(path: Path) -> dict[str, str]:
    """``{document: reason}`` from a ``document,reason`` CSV; empty when there is none."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and not {"document", "reason"} <= set(rows[0]):
        raise BundleError(f"{path} needs the columns document,reason")
    return {r["document"].strip(): (r["reason"] or "").strip() for r in rows if r["document"].strip()}


def read_lob_overrides(path: Path) -> list[tuple[str, str]]:
    """``[(source_prefix, lob)]`` from a ``source_prefix,lob,reason`` CSV; empty when there is none."""
    from common.scopes import known_lines

    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and not {"source_prefix", "lob"} <= set(rows[0]):
        raise BundleError(f"{path} needs the columns source_prefix,lob,reason")
    out = []
    from common.lob import merge_line

    for r in rows:
        # Classic auto is read as personal auto (common.lob.MERGED_LINES): an
        # override file written before the merge still names it.
        lob = str(merge_line(r["lob"].strip()))
        if lob not in known_lines():
            raise BundleError(f"{path}: {lob!r} is not a line with a canonical schema")
        out.append((r["source_prefix"].strip(), lob))
    return out


def main(argv: list[str] | None = None) -> int:
    from common.scopes import get_scope

    parser = argparse.ArgumentParser(description="Turn a synthetic-data delivery into import bundles")
    parser.add_argument("--input", required=True, type=Path, help="the delivery: Train/Val/Test + manifest.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where the bundles go (default {DEFAULT_OUT}; required for real documents alone)")
    parser.add_argument("--scope", default="personal_lines",
                        help="keep only this scope's lines ('none' keeps every line)")
    parser.add_argument("--mode", choices=("link", "copy", "move"), default="link",
                        help="link (default: instant, no second copy on the same drive), copy, or move")
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUSIONS,
                        help="document,reason CSV of documents to leave out")
    parser.add_argument("--lob-overrides", type=Path, default=DEFAULT_LOB_OVERRIDES,
                        help="source_prefix,lob,reason CSV of lines to correct")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the bundles (and a SPEC_21 delivery's gold corrections), write nothing")
    parser.add_argument("--originals", type=Path, default=DEFAULT_ORIGINALS,
                        help="SPEC_21 delivery: the seed documents (pdfs/, gold json/); 'none' leaves them out")
    parser.add_argument("--lob", help="real documents alone (PDFs/ + Gold JSON/): the line every one is of")
    parser.add_argument("--frozen", action="store_true",
                        help="real documents alone: the tenant's eval set is frozen - draw train and val only")
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.ingestion.prepare_bundles", argv):
        return 0

    lines = None if args.scope == "none" else get_scope(args.scope).lines or None
    try:
        if (args.input / ORIGINAL_PDFS).is_dir() and (args.input / ORIGINAL_GOLDS).is_dir():
            if not args.lob:
                print(f"{args.input} holds real documents alone ({ORIGINAL_PDFS}/, {ORIGINAL_GOLDS}/): "
                      "pass --lob, the line they are all of", file=sys.stderr)
                return 1
            if args.out is None:
                # The default is the personal-lines bundles: mixed in, these would
                # be uploaded and trained with them, and its corrections.csv lost.
                print(f"pass --out for real documents alone (e.g. data/cgl_bundles): the default, "
                      f"{DEFAULT_OUT}, holds another delivery's bundles", file=sys.stderr)
                return 1
            report = prepare_original_bundles(args.input, args.out, lob=args.lob, lines=lines,
                                              mode=args.mode, exclusions=read_exclusions(args.exclude),
                                              dry_run=args.dry_run, frozen=args.frozen)
        elif (args.input / TWIN_MANIFEST).is_file():
            args.out = args.out or DEFAULT_OUT
            originals = None if str(args.originals).lower() == "none" else args.originals
            if originals is not None and not (originals / "pdfs").is_dir():
                print(f"no seed documents at {originals} (pdfs/); pass --originals none to bundle the "
                      "twins alone", file=sys.stderr)
                return 1
            report = prepare_twin_bundles(args.input, args.out, lines=lines, mode=args.mode,
                                          exclusions=read_exclusions(args.exclude), dry_run=args.dry_run,
                                          originals=originals)
        else:
            args.out = args.out or DEFAULT_OUT
            report = prepare_bundles(args.input, args.out, lines=lines, mode=args.mode,
                                     exclusions=read_exclusions(args.exclude),
                                     lob_overrides=read_lob_overrides(args.lob_overrides), dry_run=args.dry_run)
    except BundleError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(report.describe())
    scope = "" if args.scope == "personal_lines" else f" --scope {args.scope}"
    print(f"\nNext: python -m data_pipeline.audit --input \"{args.out}\"{scope}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
