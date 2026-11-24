"""Runtime code never imports build-time code (review A1, 2026-11-24).

The production image (``azure_service/Dockerfile``) runs ``common``, ``contracts``,
``inference_core``, ``calibration``, ``serving`` and ``azure_service``. An import from
``data_pipeline``, ``training``, ``evaluation``, ``orchestration``, ``testing``,
``scripts`` or ``pilot`` there ties the served path to build-time code: R4's guard
died of one. Shared pieces move to ``contracts/`` (which imports only the standard
library and ``common``), re-exported at their old paths.

:data:`ALLOWED` lists the imports that are still there, each to be removed. The list
may only shrink: :func:`test_the_allow_list_holds_only_imports_that_still_exist`
fails on an entry that no longer matches, so a fixed import must leave the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RUNTIME = ("common", "contracts", "inference_core", "calibration", "serving", "azure_service")
BUILD_TIME = ("data_pipeline", "training", "evaluation", "orchestration", "testing", "scripts", "pilot")

#: (importing file, imported module) still allowed - today's violations, to be removed.
ALLOWED: set[tuple[str, str]] = {
    # The served MinerU client (an adapter, not a contract): the worker reads pages with it.
    ("azure_service/runtime.py", "data_pipeline.ocr.served_ocr"),
    # The retention CLI's tmux guard, run by hand; not the worker's path.
    ("azure_service/retention.py", "orchestration.detach"),
    # Fitting the calibrators reads validation generations: build time, in a runtime package.
    ("calibration/population.py", "evaluation.validation_generation"),
    # A stalled engine sends the operator a message.
    ("inference_core/stall_guard.py", "orchestration.notify"),
    # The OCR environment probe (installed MinerU, CUDA): an adapter, at the old endpoint's cold start.
    ("serving/vllm_entrypoint.py", "data_pipeline.ocr.mineru_version"),
}

#: ``contracts`` may lean on the standard library and ``common`` only.
CONTRACTS_MAY_IMPORT = ("common", "contracts")


def upward_imports() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for package in RUNTIME:
        for path in sorted((REPO / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    if module.split(".")[0] in BUILD_TIME:
                        found.add((path.relative_to(REPO).as_posix(), module))
    return found


def contracts_imports() -> set[tuple[str, str]]:
    """Every non-standard-library import inside ``contracts`` that is not ``common``/``contracts``."""
    import sys

    found: set[tuple[str, str]] = set()
    for path in sorted((REPO / "contracts").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top not in CONTRACTS_MAY_IMPORT and top not in sys.stdlib_module_names and top != "__future__":
                    found.add((path.relative_to(REPO).as_posix(), module))
    return found


def test_runtime_code_imports_no_new_build_time_code() -> None:
    new = sorted(upward_imports() - ALLOWED)
    assert not new, f"runtime code imports build-time code (move the piece to contracts/): {new}"


def test_the_allow_list_holds_only_imports_that_still_exist() -> None:
    gone = sorted(ALLOWED - upward_imports())
    assert not gone, f"fixed - remove from ALLOWED: {gone}"


def test_contracts_leans_on_common_and_the_standard_library_only() -> None:
    assert contracts_imports() == set()


def test_a_new_upward_import_is_seen(tmp_path, monkeypatch) -> None:
    """Seen to fail on purpose: a runtime module importing training."""
    import tests.test_layering as module

    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "x.py").write_text("from training.train import main\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "RUNTIME", ("serving",))
    assert module.upward_imports() == {("serving/x.py", "training.train")}
