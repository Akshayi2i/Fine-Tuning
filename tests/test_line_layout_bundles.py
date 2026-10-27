"""Real documents of several lines, laid out by line and by kind - the first
commercial training set (27 October 2026) - turned into bundles
(data_pipeline.ingestion.prepare_bundles: line_layout, own_line_gold,
prepare_original_bundles without --lob)."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from data_pipeline.ingestion import prepare_bundles as P
from tests.test_cgl_originals import _gl_gold

pymupdf = pytest.importorskip("pymupdf")


def _gold(line: str, insured: str, *, parts: tuple[str, ...] | None = None, coverages: bool = True) -> dict:
    """A one-coverage gold of ``line`` (its first listed code), shaped as _gl_gold's."""
    from common.schemas import load_schema

    code = load_schema("policy", None, line)["fideon:coverage_codes"][0]
    gold = _gl_gold(insured=insured, parts=parts or (line,), code=code)
    if not coverages:
        del gold["coverages"]
    return gold


def _source(tmp_path: Path, docs: dict[tuple[str, str, str], dict | None]) -> Path:
    """``docs``: (line folder, kind, name) -> gold, or None for a PDF with no gold."""
    root = tmp_path / "source data"
    for (folder, kind, name), gold in docs.items():
        pdf = root / "pdf" / folder / kind / f"{name}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        doc.new_page(width=612, height=792).insert_text((72, 72), "Policy Declarations")
        doc.save(pdf)
        if gold is not None:
            path = root / "gold" / folder / kind / f"{name}.gold.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(gold), encoding="utf-8")
    return root


def test_a_folder_line_is_read_as_its_schema_and_bop_as_boppr(tmp_path):
    root = _source(tmp_path, {("cgl", "hybrid", "a"): _gold("gl", "A CO"),
                              ("bop", "scanned", "b"): _gold("boppr", "B CO"),
                              ("wc", "scanned", "c"): _gold("wc", "C CO")})
    pdfs, golds, line_of, kind_of = P.line_layout(root)
    assert set(pdfs) == set(golds)
    assert {key.as_posix(): (line_of[key], kind_of[key]) for key in pdfs} == {
        "cgl/hybrid/a": ("gl", "hybrid"), "bop/scanned/b": ("boppr", "scanned"), "wc/scanned/c": ("wc", "scanned")}
    # Lines outside --lines are left out.
    assert {line_of[key] for key in P.line_layout(root, frozenset({"wc"}))[0]} == {"wc"}


@pytest.mark.parametrize("bad, words", [
    (("cgl", "native", "a"), "a kind is one of"),
    (("basket_weaving", "scanned", "a"), "no canonical policy schema"),
])
def test_an_unknown_line_or_kind_folder_is_refused(tmp_path, bad, words):
    root = _source(tmp_path, {bad: _gold("gl", "A CO")})
    with pytest.raises(P.BundleError, match=words):
        P.line_layout(root)


def test_a_package_keeps_the_part_of_its_folder_s_line_and_a_missing_table_is_empty():
    gold = _gold("bopgl", "A CO", parts=("property", "bopgl", "umbrella"))
    held, notes = P.own_line_gold(gold, "bopgl")
    assert [part["lob"] for part in held["lob_parts"]] == ["bopgl"]
    assert held["lob_parts"][0]["part_id"] == "part_2"            # the part of that line, where there is one
    assert notes == [("package parts", "lob_parts property+bopgl+umbrella -> bopgl; "
                                       "2 value(s) of the others kept as extra fields")]
    assert len(gold["lob_parts"]) == 3                               # the delivered gold is never changed
    # The other parts' printed titles stay, as extra fields: dropped, a window
    # showing them was taught to leave them out.
    assert [(entry["label"], entry["value"], entry["section_hint"]) for entry in held["additional_fields"]] == [
        ("Coverage Part", gold["lob_parts"][0]["title"], "lob_parts"),
        ("Coverage Part", gold["lob_parts"][2]["title"], "lob_parts")]
    premium = copy.deepcopy(gold)
    premium["lob_parts"][0]["premium"] = {**gold["lob_parts"][0]["title"], "raw": "$1,250", "parsed": 1250}
    labels = [entry["label"] for entry in P.own_line_gold(premium, "bopgl")[0]["additional_fields"]]
    assert f"{gold['lob_parts'][0]['title']['raw']} Premium" in labels

    # A part named for another line only (bop read as boppr): written as the line.
    held, notes = P.own_line_gold(_gold("boppr", "B CO", parts=("bopgl",)), "boppr")
    assert held["lob_parts"][0]["lob"] == "boppr" and notes[0][0] == "package parts"

    held, notes = P.own_line_gold(_gold("wc", "C CO", coverages=False), "wc")
    assert held["coverages"] == [] and notes == [("coverages missing",
                                                  "coverages = [] (the document prints no coverage table)")]
    plain = _gold("gl", "D CO")
    assert P.own_line_gold(plain, "gl") == (plain, [])


def test_every_line_is_bundled_in_one_draw_with_its_kind_and_review(tmp_path):
    docs = {("cgl", "hybrid", f"gl_{n}"): _gold("gl", f"GL INSURED {n}") for n in range(8)}
    docs[("bopgl", "hybrid", "pkg")] = _gold("bopgl", "PACKAGE CO", parts=("property", "bopgl"))
    docs[("wc", "scanned", "endo")] = _gold("wc", "WC CO", coverages=False)
    docs[("wc", "scanned", "orphan")] = None
    root = _source(tmp_path, docs)
    (root / "reviews.csv").write_text(
        "document,reviewer_id,review_date,draft_backend\n"
        "*,A. Reviewer,2026-10-27,data team\n"
        "cgl/hybrid/gl_0,B. Second,2026-10-26,data team\n", encoding="utf-8")
    out = tmp_path / "bundles"

    report = P.prepare_original_bundles(root, out)
    assert report.skipped["a PDF without its gold"] == 1 and report.unreviewed == 0
    assert report.corrections["package parts"] == 1 and report.corrections["coverages missing"] == 1
    folders = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert "bopgl__hybrid__pkg" in folders and "wc__scanned__endo" in folders and len(folders) == 10

    meta = json.loads((out / "bopgl__hybrid__pkg" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["lob"] == "bopgl" and meta["document_kind"] == "hybrid" and meta["synthetic"] is False
    assert (meta["reviewer_id"], meta["review_date"]) == ("A. Reviewer", "2026-10-27")     # the * row
    first = json.loads((out / "gl__hybrid__gl_0" / "metadata.json").read_text(encoding="utf-8"))
    assert first["reviewer_id"] == "B. Second"                                              # its own row wins
    gold = json.loads((out / "bopgl__hybrid__pkg" / "golden.json").read_text(encoding="utf-8"))
    assert [part["lob"] for part in gold["lob_parts"]] == ["bopgl"]

    with (out / "split.csv").open(encoding="utf-8") as fh:
        split = {row["document"]: row["split"] for row in csv.DictReader(fh)}
    assert set(split) == set(folders) and {"train"} <= set(split.values())
    with (out / "corrections.csv").open(encoding="utf-8") as fh:
        kinds = {row["correction"] for row in csv.DictReader(fh)}
    assert {"package parts", "coverages missing"} <= kinds


def test_every_line_and_kind_of_three_or_more_trains_and_is_measured(tmp_path):
    """The first commercial set's boppr: ten scans and three hybrids. Hashed by
    line alone, all three hybrids drew val or test and none trained."""
    docs = {("bop", "scanned", f"s_{n}"): _gold("boppr", f"SCAN INSURED {n}") for n in range(10)}
    docs.update({("bop", "hybrid", f"h_{n}"): _gold("boppr", f"HYBRID INSURED {n}") for n in range(3)})
    docs.update({("wc", "hybrid", f"w_{n}"): _gold("wc", f"WC INSURED {n}") for n in range(3)})
    docs.update({("wc", "scanned", f"x_{n}"): _gold("wc", f"WC SCAN {n}") for n in range(2)})
    out = tmp_path / "bundles"
    P.prepare_original_bundles(_source(tmp_path, docs), out)

    with (out / "split.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cells: dict[str, list[str]] = {}
    for row in rows:
        cells.setdefault("/".join(row["document"].split("__")[:2]), []).append(row["split"])
    assert sorted(cells["boppr/hybrid"]) == ["test", "train", "val"]
    assert sorted(cells["wc/hybrid"]) == ["test", "train", "val"]
    assert cells["wc/scanned"] == ["train", "train"]                     # fewer than three: trains
    scans = cells["boppr/scanned"]
    assert scans.count("train") == 7 and scans.count("val") == 2 and scans.count("test") == 1

    # Once the eval set is frozen, nothing new is drawn into test.
    frozen = tmp_path / "frozen"
    P.prepare_original_bundles(_source(tmp_path, docs), frozen, frozen=True)
    with (frozen / "split.csv").open(encoding="utf-8") as fh:
        assert "test" not in {row["split"] for row in csv.DictReader(fh)}


def test_one_insured_by_policy_number_or_address_is_one_family_across_lines(tmp_path):
    """The first commercial set had 13 pairs sharing a policy number or address
    on both sides of the split: a renewal printing the insured's name another way."""
    renewal = _gold("gl", "NORTHWIND HOLDINGS")
    renewal["policy"]["policy_number"].update(raw="CGL 4471-0093", parsed="CGL 4471-0093")
    first = _gold("gl", "NORTHWIND HLDGS GROUP")
    first["policy"]["policy_number"].update(raw="CGL4471 0093", parsed="CGL4471 0093")
    moved = _gold("wc", "BLUE HERON CAFE")
    moved["named_insured"]["mailing_address"]["street"].update(raw="12 Elm St", parsed="12 Elm St")
    same_place = _gold("boppr", "HERON CAFE OF TROY")
    same_place["named_insured"]["mailing_address"]["street"].update(raw="12 ELM ST.", parsed="12 ELM ST.")
    docs = {("cgl", "hybrid", "renewal"): renewal, ("cgl", "scanned", "first"): first,
            ("wc", "scanned", "moved"): moved, ("bop", "scanned", "same_place"): same_place}
    docs.update({("cgl", "hybrid", f"other_{n}"): _gold("gl", f"OTHER INSURED {n}") for n in range(6)})
    out = tmp_path / "bundles"
    P.prepare_original_bundles(_source(tmp_path, docs), out)

    def family(name: str) -> str:
        return json.loads((out / name / "metadata.json").read_text(encoding="utf-8"))["template_id"]

    assert family("gl__hybrid__renewal") == family("gl__scanned__first")          # one policy number
    assert family("wc__scanned__moved") == family("boppr__scanned__same_place")    # one address, two lines
    assert len({family(f"gl__hybrid__other_{n}") for n in range(6)}) == 6
    with (out / "split.csv").open(encoding="utf-8") as fh:
        split_of = {row["document"]: row["split"] for row in csv.DictReader(fh)}
    assert split_of["gl__hybrid__renewal"] == split_of["gl__scanned__first"]


def test_the_importer_records_a_canonical_gold_s_policy_number():
    assert P.gold_policy_number(_gold("gl", "A CO")) == _gold("gl", "A CO")["policy"]["policy_number"]["raw"]
    assert P.gold_policy_number({"policy_number": "P-1"}) == "P-1"                # a flat label's
    assert P.gold_policy_number({"policy": {}}) is None


def test_a_line_s_held_out_carrier_still_goes_to_test():
    from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_cell_splits

    records = [GroupRecord(group_id=f"gl/{n:02}", doc_type="policy", source_ids=[f"d{n}"], line="gl",
                           carrier=f"carrier {n % 12}") for n in range(36)]
    result = assign_cell_splits(records, {r.group_id: "gl/hybrid" for r in records})
    held = result.held_out_carriers_by_line["policy"]["gl"]
    assert {result.assignment[r.group_id] for r in records if r.carrier == held} == {"test"}
    assert {"train", "val", "test"} == set(result.assignment.values())


def test_the_command_bundles_the_layout_without_lob_and_asks_for_out(tmp_path, capsys):
    root = _source(tmp_path, {("cgl", "hybrid", "a"): copy.deepcopy(_gold("gl", "A CO"))})
    assert P.main(["--input", str(root), "--dry-run"]) == 1
    assert "pass --out for real documents of several lines" in capsys.readouterr().err
    assert P.main(["--input", str(root), "--out", str(tmp_path / "b"), "--dry-run"]) == 0
    assert "1 document folder(s)" in capsys.readouterr().out


def test_the_source_s_own_exclusions_hold_on_a_re_run_without_exclude(tmp_path):
    """The 17-page boppr held for recheck went into train on a re-run that left
    out --exclude."""
    root = _source(tmp_path, {("cgl", "hybrid", f"g_{n}"): _gold("gl", f"INSURED {n}") for n in range(4)})
    (root / "exclusions.csv").write_text("document,reason\ngl__hybrid__g_0,held for recheck\n", encoding="utf-8")
    out = tmp_path / "bundles"
    report = P.prepare_original_bundles(root, out)
    assert not (out / "gl__hybrid__g_0").exists()
    assert report.skipped["excluded: held for recheck"] == 1
