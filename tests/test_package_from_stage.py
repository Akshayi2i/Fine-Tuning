"""`package --from-stage` resumes a failed package run at the stage that failed."""

from __future__ import annotations

from orchestration.run import build_parser, flag_problem


def _parse(*argv):
    return build_parser().parse_args(list(argv))


def test_package_accepts_from_stage():
    args = _parse("package", "--version", "v3", "--from-stage", "evaluation_gate")
    assert args.from_stage == "evaluation_gate" and args.version == "v3"


def test_package_without_it_starts_at_the_first_stage():
    assert _parse("package", "--version", "v3").from_stage is None


def test_a_stage_package_does_not_have_is_refused_with_the_ones_it_has():
    from orchestration.run import from_stage_problem

    problem = from_stage_problem("package", "training")
    assert problem and "calibrate" in problem and "evaluation_gate" in problem


def test_all_and_finetune_still_take_it():
    assert _parse("finetune", "--version", "v3", "--from-stage", "training").from_stage == "training"
    assert _parse("all", "--version", "v3", "--from-stage", "package").from_stage == "package"


def test_resuming_needs_the_version_named():
    args = _parse("package", "--version", "v3", "--from-stage", "evaluation_gate")
    assert flag_problem(args) is None
    args.version = None
    assert "name its version" in flag_problem(args)
