"""Real documents of one line with no twins and no split - the CGL source data -
turned into bundles (data_pipeline.ingestion.prepare_bundles.prepare_original_bundles)."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from data_pipeline.ingestion.prepare_bundles import (
    BundleError,
    corrected_gold,
    prepare_original_bundles,
    read_recodes,
)

pymupdf = pytest.importorskip("pymupdf")

EXAMPLE = Path("configs/canonical schema/common schema/examples/homeowners_minimal.json")


def _gl_gold(insured: str = "TAYLOR SAMPLE", parts: tuple[str, ...] = ("gl",), code: str = "GL_BI_PD",
             carrier: str | None = None) -> dict:
    """The homeowners example's shared blocks, as a one-coverage general liability gold."""
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    gold = {key: copy.deepcopy(example[key]) for key in ("document", "carrier", "named_insured", "policy")}
    gold["named_insured"]["primary_name"].update(raw=insured, parsed=insured)
    if carrier is not None:
        gold["carrier"]["name"].update(raw=carrier, parsed=carrier)
    gold["lob_parts"] = [{**copy.deepcopy(example["lob_parts"][0]), "part_id": f"part_{n}", "lob": lob}
                         for n, lob in enumerate(parts, start=1)]
    coverage = copy.deepcopy(example["coverages"][0])
    for key in ("applies_to", "included"):
        coverage.pop(key, None)
    gold["coverages"] = [{**coverage, "coverage_code": code}]
    return gold


def _source(tmp_path: Path, docs: dict[tuple[str, str], dict | None]) -> Path:
    """``docs``: (carrier folder, name) -> gold, or None for a PDF with no gold."""
    root = tmp_path / "CGL source data"
    for (carrier, name), gold in docs.items():
        pdf = root / "PDFs" / carrier / f"{name}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        doc.new_page(width=612, height=792).insert_text((72, 72), "Commercial General Liability Declarations")
        doc.save(pdf)
        if gold is not None:
            path = root / "Gold JSON" / carrier / f"{name}.gold.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(gold), encoding="utf-8")
    return root


def test_a_renamed_coverage_code_is_recoded_as_the_line_s_code_file_says():
    """gl 3.0.0 moved three codes to the shared X_ codes and says so in its code
    file (replaces_codes): a gold drafted before the move is re-coded, not
    refused at the corpus build."""
    recodes = read_recodes()
    assert recodes["gl"]["GL_ADDITIONAL_INSURED"] == "X_ADDITIONAL_INSURED"
    assert recodes["gl"]["GL_WAIVER_OF_SUBROGATION"] == "X_WAIVER_OF_SUBROGATION"

    fixed, notes, problem = corrected_gold(_gl_gold(code="GL_ADDITIONAL_INSURED"), "gl", carrier="",
                                           text_pdf=Path("none.pdf"), recodes=recodes)
    assert problem is None and fixed["coverages"][0]["coverage_code"] == "X_ADDITIONAL_INSURED"
    assert ("coverage code", "coverages[0] GL_ADDITIONAL_INSURED -> X_ADDITIONAL_INSURED") in notes


def test_the_repository_s_recode_table_wins_over_the_code_file(tmp_path):
    table = tmp_path / "recodes.yaml"
    table.write_text("gl:\n  GL_ADDITIONAL_INSURED: X_PRIMARY_NONCONTRIBUTORY\n", encoding="utf-8")
    assert read_recodes(table)["gl"]["GL_ADDITIONAL_INSURED"] == "X_PRIMARY_NONCONTRIBUTORY"


def test_a_package_policy_filed_under_one_common_model_line_is_left_out():
    """A general liability part and an inland marine part: the corpus build refuses
    such a target and stops on it, so it never reaches a bundle."""
    _, _, problem = corrected_gold(_gl_gold(parts=("gl", "inland_marine")), "gl", carrier="",
                                   text_pdf=Path("none.pdf"), recodes={})
    assert problem is not None and problem.startswith("package policy: lob_parts lists 2 part(s)")
    _, _, problem = corrected_gold(_gl_gold(parts=("gl", "gl")), "gl", carrier="",
                                   text_pdf=Path("none.pdf"), recodes={})
    assert problem is not None and problem.startswith("package policy")


def test_each_pair_becomes_a_real_document_with_its_split_delivered(tmp_path):
    root = _source(tmp_path, {
        ("Johnson & Johnson", "dec_01"): _gl_gold("RIVERA FABRICATION LLC"),
        ("Johnson & Johnson", "Renewal_01"): _gl_gold("Rivera Fabrication, LLC"),
        ("Utica First", "dec_01"): _gl_gold("ANOTHER INSURED"),
    })
    out = tmp_path / "bundles"

    report = prepare_original_bundles(root, out, lob="gl")

    assert report.documents == 3
    folder = out / "gl__johnson_johnson__renewal_01"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert (folder / "document.pdf").is_file() and (folder / "golden.json").is_file()
    assert meta["lob"] == "gl" and meta["synthetic"] is False and meta["source_system"] == "original"
    assert meta["split"] in ("train", "val", "test")
    # A folder is not always one carrier (a broker's holds several): the gold's own.
    assert meta["carrier"] is None
    # One insured, however it is spelt: one family, so one side of the split.
    twin = json.loads((out / "gl__johnson_johnson__dec_01" / "metadata.json").read_text(encoding="utf-8"))
    assert twin["template_id"] == meta["template_id"] and twin["split"] == meta["split"]
    other = json.loads((out / "gl__utica_first__dec_01" / "metadata.json").read_text(encoding="utf-8"))
    assert other["template_id"] != meta["template_id"]
    # The family names no one.
    assert "rivera" not in meta["template_id"].lower()

    with (out / "split.csv").open(encoding="utf-8", newline="") as fh:
        listed = {row["document"]: row["split"] for row in csv.DictReader(fh)}
    assert listed["gl__johnson_johnson__renewal_01"] == meta["split"] and len(listed) == 3


def test_the_same_source_splits_the_same_way_every_time(tmp_path):
    docs = {(f"Carrier {n % 7}", f"policy_{n:02d}"): _gl_gold(f"INSURED {n}") for n in range(30)}
    root = _source(tmp_path, docs)
    first = prepare_original_bundles(root, tmp_path / "a", lob="gl")
    second = prepare_original_bundles(root, tmp_path / "b", lob="gl")

    assert first.written == second.written
    assert {s for s, _ in first.written} == {"train", "val", "test"}
    assert (tmp_path / "a" / "split.csv").read_text() == (tmp_path / "b" / "split.csv").read_text()


def test_what_cannot_be_trained_is_left_out_and_said_so(tmp_path):
    root = _source(tmp_path, {
        ("Philadelphia", "renewal_01"): _gl_gold(parts=("gl", "inland_marine")),
        ("Philadelphia", "renewal_02"): _gl_gold(),
        ("Philadelphia", "orphan"): None,
    })
    out = tmp_path / "bundles"
    (out / "gl__philadelphia__renewal_01").mkdir(parents=True)           # an earlier run's folder

    report = prepare_original_bundles(root, out, lob="gl")

    assert report.documents == 1
    assert report.skipped["gold left out: package policy"] == 1
    assert report.skipped["a PDF without its gold"] == 1
    assert not (out / "gl__philadelphia__renewal_01").exists() and report.removed == 1
    corrections = (out / "corrections.csv").read_text(encoding="utf-8")
    assert 'gl__philadelphia__renewal_01,left out,"package policy' in corrections


def test_two_documents_that_would_share_a_folder_are_refused(tmp_path):
    root = _source(tmp_path, {("Great American", "dec_01"): _gl_gold(),
                              ("Great-American", "dec_01"): _gl_gold()})
    with pytest.raises(BundleError, match="share the folder name"):
        prepare_original_bundles(root, tmp_path / "bundles", lob="gl")


def test_a_line_outside_lines_is_refused(tmp_path):
    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold()})

    with pytest.raises(BundleError, match=r"outside --lines \['homeowners', 'personal_auto'\]"):
        prepare_original_bundles(root, tmp_path / "bundles", lob="gl",
                                 lines=frozenset({"homeowners", "personal_auto"}))


def test_the_command_line_needs_the_line_of_real_documents_alone(tmp_path, capsys):
    from data_pipeline.ingestion.prepare_bundles import main

    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold()})
    assert main(["--input", str(root), "--out", str(tmp_path / "bundles")]) == 1
    assert "pass --lob" in capsys.readouterr().err
    assert main(["--input", str(root), "--out", str(tmp_path / "bundles"), "--lob", "gl"]) == 0
    assert (tmp_path / "bundles" / "gl__utica_first__dec_01" / "metadata.json").is_file()


def test_the_command_line_keeps_the_lines_it_is_given(tmp_path, capsys):
    """--lines is optional (every line by default); given, the real documents'
    line must be one of them, and a line with no schema is refused outright."""
    from data_pipeline.ingestion.prepare_bundles import main

    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold()})
    out = tmp_path / "bundles"
    assert main(["--input", str(root), "--out", str(out), "--lob", "gl", "--lines", "homeowners"]) == 1
    assert "outside --lines" in capsys.readouterr().err
    assert main(["--input", str(root), "--out", str(out), "--lob", "gl", "--lines", "no_such_line"]) == 1
    assert "no canonical policy schema" in capsys.readouterr().err
    assert not out.exists() or not any(out.iterdir())
    assert main(["--input", str(root), "--out", str(out), "--lob", "gl", "--lines", "general_liability"]) == 0
    assert "--lines gl" in capsys.readouterr().out                 # the audit it points to


def test_the_command_line_needs_an_out_folder_for_real_documents(tmp_path, capsys):
    """The default --out is the personal-lines bundles: mixed in, these would be
    uploaded and trained with them."""
    from data_pipeline.ingestion.prepare_bundles import main

    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold()})
    assert main(["--input", str(root), "--lob", "gl"]) == 1
    assert "pass --out" in capsys.readouterr().err


def test_the_cgl_code_bundles_as_general_liability(tmp_path):
    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold(code="GL_ADDITIONAL_INSURED")})

    out = tmp_path / "bundles"
    report = prepare_original_bundles(root, out, lob="CGL", lines=frozenset({"gl"}))

    assert report.documents == 1
    meta = json.loads((out / "gl__utica_first__dec_01" / "metadata.json").read_text(encoding="utf-8"))
    gold = json.loads((out / "gl__utica_first__dec_01" / "golden.json").read_text(encoding="utf-8"))
    assert meta["lob"] == "gl" and meta["template_id"].startswith("gl/")
    assert gold["coverages"][0]["coverage_code"] == "X_ADDITIONAL_INSURED"          # gl's recodes applied


def test_a_part_named_by_another_spelling_of_the_line_is_the_line():
    for spelling in ("cgl", "general_liability", "GL"):
        _, _, problem = corrected_gold(_gl_gold(parts=(spelling,)), "gl", carrier="",
                                       text_pdf=Path("none.pdf"), recodes={})
        assert problem is None, spelling


def test_real_documents_are_never_moved(tmp_path):
    """A re-run leaves out a document whose PDF is gone and removes its folder:
    after a move, that folder held the only copy."""
    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold()})
    with pytest.raises(BundleError, match="never moved"):
        prepare_original_bundles(root, tmp_path / "bundles", lob="gl", mode="move")
    assert (root / "PDFs" / "Utica First" / "dec_01.pdf").is_file()


def test_once_the_eval_set_is_frozen_documents_are_drawn_into_train_and_val_only(tmp_path):
    """The frozen set is the test set: a new document delivered as test stops the
    corpus build, so none is."""
    docs = {(f"Carrier {n % 7}", f"policy_{n:02d}"): _gl_gold(f"INSURED {n}", carrier=f"Carrier {n % 7}")
            for n in range(30)}
    root = _source(tmp_path, docs)
    report = prepare_original_bundles(root, tmp_path / "bundles", lob="gl", frozen=True)
    assert {split for split, _ in report.written} == {"train", "val"}


def test_a_gold_with_no_carrier_is_never_given_its_folder_s_name(tmp_path):
    """A folder is not always one carrier - a broker's holds several - so a gold
    missing its carrier is left out, not filled with the folder's name."""
    gold = _gl_gold()
    del gold["carrier"]
    root = _source(tmp_path, {("Johnson & Johnson", "dec_01"): gold})
    report = prepare_original_bundles(root, tmp_path / "bundles", lob="gl")
    assert report.documents == 0 and report.skipped
    assert not report.corrections.get("carrier missing")


def test_one_insured_spelt_with_and_without_its_legal_form_is_one_family(tmp_path):
    root = _source(tmp_path, {
        ("Utica First", "dec_01"): _gl_gold("RIVERA FABRICATION LLC"),
        ("Utica First", "renewal_01"): _gl_gold("The Rivera Fabrication Co."),
        ("Utica First", "renewal_02"): _gl_gold("Rivera Fabrication & Welding, Inc."),
    })
    out = tmp_path / "bundles"
    prepare_original_bundles(root, out, lob="gl")
    family = {name: json.loads((out / name / "metadata.json").read_text(encoding="utf-8"))["template_id"]
              for name in ("gl__utica_first__dec_01", "gl__utica_first__renewal_01", "gl__utica_first__renewal_02")}
    assert family["gl__utica_first__dec_01"] == family["gl__utica_first__renewal_01"]
    assert family["gl__utica_first__renewal_02"] != family["gl__utica_first__dec_01"]


def test_a_test_document_of_a_carrier_training_never_sees_is_marked_held_out(tmp_path):
    """The gate reports those documents apart: the model reading a carrier it never saw."""
    docs = {(f"Carrier {n % 10}", f"policy_{n:02d}"): _gl_gold(f"INSURED {n}", carrier=f"Carrier {n % 10}")
            for n in range(40)}
    root = _source(tmp_path, docs)
    out = tmp_path / "bundles"
    prepare_original_bundles(root, out, lob="gl")

    metas = {folder.name: json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
             for folder in out.iterdir() if folder.is_dir()}
    carrier = {name: f"carrier {int(name.rsplit('_', 1)[1]) % 10}" for name in metas}
    trained = {carrier[name] for name, meta in metas.items() if meta["split"] == "train"}
    for name, meta in metas.items():
        expected = meta["split"] == "test" and carrier[name] not in trained
        assert meta["held_out_carrier"] is expected, name
    # The gl line names ten carriers, so the draw holds one of them out.
    assert any(meta["held_out_carrier"] for meta in metas.values())
    with (out / "split.csv").open(encoding="utf-8", newline="") as fh:
        assert {row["document"]: row["held_out_carrier"] == "True" for row in csv.DictReader(fh)} == {
            name: meta["held_out_carrier"] for name, meta in metas.items()}


def test_a_placeholder_carrier_is_never_an_unseen_one(tmp_path):
    """"N/A" names no carrier, so no carrier the model never saw (decision 2026-10-22)."""
    docs = {(f"Carrier {n % 10}", f"policy_{n:02d}"): _gl_gold(f"INSURED {n}", carrier=f"Carrier {n % 10}")
            for n in range(40)}
    docs[("Broker", "unknown_0")] = _gl_gold("OTHER 0", carrier="N/A")       # drawn into test
    out = tmp_path / "bundles"
    prepare_original_bundles(_source(tmp_path, docs), out, lob="gl")
    meta = json.loads((out / "gl__broker__unknown_0" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["split"] == "test" and meta["held_out_carrier"] is False


def test_real_documents_carry_the_review_record_their_reviews_file_gives(tmp_path):
    from data_pipeline.labeling.export_golden_labels import has_review_record

    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold("A"), ("Utica First", "dec_02"): _gl_gold("B"),
                              ("Utica First", "dec_03"): _gl_gold("C")})
    (root / "reviews.csv").write_text(
        "document,reviewer_id,review_date,draft_backend\n"
        "Utica First/dec_01,a.reviewer,2026-10-20,label_studio\n"          # by the gold's path
        "gl__utica_first__dec_02,b.reviewer,2026-10-21,\n",                # or by the folder name
        encoding="utf-8")
    out = tmp_path / "bundles"
    report = prepare_original_bundles(root, out, lob="gl")
    metas = {n: json.loads((out / f"gl__utica_first__{n}" / "metadata.json").read_text(encoding="utf-8"))
             for n in ("dec_01", "dec_02", "dec_03")}
    assert (metas["dec_01"]["reviewer_id"], metas["dec_01"]["draft_backend"]) == ("a.reviewer", "label_studio")
    assert metas["dec_02"]["review_date"] == "2026-10-21" and has_review_record(metas["dec_02"])
    assert metas["dec_03"]["reviewer_id"] is None and report.unreviewed == 1


def test_one_carrier_printed_two_ways_is_one_carrier_to_the_draw(tmp_path):
    """Nine carriers, one printed "Travelers Indemnity Company" and "TRAVELERS".
    Keyed by the printed name they were ten: one was held out against the
    ten-carrier rule, and a carrier that trained was marked unseen."""
    from common.normalize import normalize_carrier

    docs = {("Broker", f"doc{n:03d}"): _gl_gold(f"INSURED {n}", carrier=f"Carrier{n // 4} Insurance Company")
            for n in range(32)}
    docs |= {("Broker", f"doc{n:03d}"): _gl_gold(f"INSURED {n}", carrier="Travelers Indemnity Company")
             for n in range(32, 36)}
    docs[("Broker", "doc036")] = _gl_gold("INSURED 36", carrier="TRAVELERS")
    printed = {f"gl__broker__{name}": gold["carrier"]["name"]["raw"] for (_, name), gold in docs.items()}
    root = _source(tmp_path, docs)
    for seed in (42, 43, 44, 45):
        out = tmp_path / f"bundles_{seed}"
        prepare_original_bundles(root, out, lob="gl", seed=seed)
        with (out / "split.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        trained = {normalize_carrier(printed[r["document"]]) for r in rows if r["split"] == "train"}
        for row in rows:
            unseen = row["split"] == "test" and normalize_carrier(printed[row["document"]]) not in trained
            assert (row["held_out_carrier"] == "True") is unseen, (seed, row)
        # Nine carriers hold none out: here every one of them trains.
        assert trained == {normalize_carrier(name) for name in printed.values()}, seed
        assert len(trained) == 9 and not [r for r in rows if r["held_out_carrier"] == "True"], seed


def test_a_reviews_file_excel_saved_as_csv_utf8_is_read(tmp_path):
    """Excel's "CSV UTF-8" starts with a byte-order mark: read as plain UTF-8,
    the first column was "\ufeffdocument" and the file was refused."""
    root = _source(tmp_path, {("Utica First", "dec_01"): _gl_gold("A")})
    (root / "reviews.csv").write_text(
        "document,reviewer_id,review_date,draft_backend\r\nUtica First/dec_01,李明,2026-10-20,\r\n",
        encoding="utf-8-sig")
    out = tmp_path / "bundles"
    report = prepare_original_bundles(root, out, lob="gl")
    meta = json.loads((out / "gl__utica_first__dec_01" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["reviewer_id"] == "李明" and report.unreviewed == 0


def test_a_reviews_file_without_its_columns_says_what_its_header_has(tmp_path):
    from data_pipeline.ingestion.prepare_bundles import read_reviews

    path = tmp_path / "reviews.csv"
    path.write_text("doc,reviewer\nUtica First/dec_01,a.reviewer\n", encoding="utf-8-sig")
    with pytest.raises(BundleError, match=r"needs the columns document,reviewer_id,review_date.*\['doc', 'reviewer'\]"):
        read_reviews(path)
    path.write_bytes("document,reviewer_id,review_date\nUtica First/dec_01,José,2026-10-20\n".encode("cp1252"))
    with pytest.raises(BundleError, match="not UTF-8 text.*CSV UTF-8"):
        read_reviews(path)


def test_the_corpus_build_records_the_delivered_held_out_documents():
    """So the frozen set marks them and the gate reports them apart."""
    from types import SimpleNamespace

    from orchestration.pipeline_dag import delivered_held_out_ids

    def doc(source_id, held, group):
        return SimpleNamespace(source_id=source_id, delivered_held_out=held, group_id=group)

    documents = [doc("policy_0001", True, "g1"), doc("policy_0002", True, "g2"),
                 doc("policy_0003", False, "g3"), doc("policy_0004", True, "g4")]
    assignment = SimpleNamespace(assignment={"g1": "test", "g2": "val", "g3": "test", "g4": "test"},
                                 held_out_source_ids=["policy_0004"])
    assert delivered_held_out_ids(documents, assignment) == ["policy_0001"]


# --------------------------------------------------------------------------
# The general liability line
# --------------------------------------------------------------------------


def test_every_spelling_of_general_liability_is_one_line():
    from common.lob import lob_lines, within_lines

    assert lob_lines("cgl") == lob_lines("general_liability") == lob_lines("gl") == frozenset({"gl"})
    assert within_lines("cgl", frozenset({"gl"}))
    assert not within_lines("auto", frozenset({"gl"}))


def test_the_cgl_code_reads_the_general_liability_schema():
    from common.schemas import schema_key

    assert schema_key("policy", None, "cgl") == schema_key("policy", None, "gl") == "policy:gl"


def test_the_smoke_run_of_the_cgl_batch_trains_the_one_adapter(tmp_path):
    """No scope to pick any more: a CGL smoke run trains the policy adapter on
    whatever its batch holds, as a real cycle does."""
    from orchestration.smoke_run import commands, parse_args

    args, _ = parse_args(["--batch", "cgl-v1", "--tenant", "smoke-cgl"])
    steps = commands(batch_dir=tmp_path, subset_dir=tmp_path, tenant=args.tenant, version="v0",
                     check_out=tmp_path)
    for step in ("preflight", "finetune"):
        assert "--scope" not in steps[step]
        assert steps[step][steps[step].index("--tenant") + 1] == "smoke-cgl"
    with pytest.raises(SystemExit):
        parse_args(["--scope", "casualty_fleet", "--tenant", "smoke-cgl"])
