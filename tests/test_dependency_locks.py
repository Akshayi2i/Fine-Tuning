"""Each pod environment is installed from a lock, and the locks agree where they must.

pyproject.toml declares ranges. Unpinned, the serving pod resolved transformers 5
while training ran 4.57, and the OCR environment resolved the newest torch, built
for a CUDA the pod's driver may not run. The locks (scripts/lock_requirements.sh)
pin every package for Linux x86-64 / Python 3.11; setup_pod.sh installs through them.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
ROLES = ("train", "ocr", "serve", "quantize")


def _pins(role: str) -> dict[str, str]:
    text = (ROOT / f"requirements-{role}.lock").read_text(encoding="utf-8")
    return {canonicalize_name(n): v for n, v in re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", text, re.M)}


def _groups(role: str) -> list[str]:
    line = next(row for row in (ROOT / f"requirements-{role}.txt").read_text(encoding="utf-8").splitlines()
                if row.startswith("-e"))
    return re.search(r"\[([^\]]+)\]", line).group(1).split(",")


@pytest.mark.parametrize("role", ROLES)
def test_every_pod_role_has_a_lock_with_no_local_paths(role):
    text = (ROOT / f"requirements-{role}.lock").read_text(encoding="utf-8")
    assert "-e " not in text and "file:" not in text and ":\\" not in text
    assert len(_pins(role)) > 50


@pytest.mark.parametrize("role", ROLES)
def test_each_lock_satisfies_what_pyproject_declares(role):
    """Fails when pyproject changes and the locks were not regenerated."""
    pins = _pins(role)
    specs = list(PYPROJECT["project"]["dependencies"])
    for group in _groups(role):
        specs += PYPROJECT["project"]["optional-dependencies"][group]
    stale = []
    for spec in specs:
        req = Requirement(spec)
        version = pins.get(canonicalize_name(req.name))
        if version is None or not req.specifier.contains(version, prereleases=True):
            stale.append(f"{req}: locked {version}")
    assert not stale, f"requirements-{role}.lock is stale (bash scripts/lock_requirements.sh): {stale}"


def test_serving_runs_exactly_the_versions_training_calibrated_with():
    train, serve = _pins("train"), _pins("serve")
    assert set(serve) <= set(train)
    assert {k: (train[k], serve[k]) for k in serve if train[k] != serve[k]} == {}


@pytest.mark.parametrize("package", ["torch", "pillow", "pymupdf"])
def test_every_environment_shares_the_gpu_and_pixel_stack(package):
    """One torch build for the pod's CUDA; one pillow and one PyMuPDF, because
    the OCR environment renders the pages the training environment trains on."""
    versions = {role: _pins(role).get(package) for role in ROLES}
    assert len(set(versions.values())) == 1, versions


def test_the_stack_is_the_one_the_code_is_written_for():
    train, ocr = _pins("train"), _pins("ocr")
    assert train["torch"].startswith("2.8.")
    assert train["vllm"] == "0.11.0"
    assert train["transformers"].startswith("4.57.")
    assert train["ms-swift"].startswith("3.")
    assert ocr["mineru"].startswith("3.4.")
    # Imported by MinerU 3.4.5's OCR text system but not declared by it: without it
    # the pipeline backend fails to import (a fresh pod, 2026-10-09).
    assert "six" in ocr


def test_setup_installs_through_the_lock_and_caps_flash_attn():
    text = (ROOT / "scripts" / "setup_pod.sh").read_text(encoding="utf-8")
    assert 'python -m pip install -r "$requirements" -c "$lock"' in text
    # One exact version (review 2026-10-26 #46), checked after the install.
    assert 'FLASH_ATTN_VERSION="2.8.3"' in text
    assert '"flash-attn==${FLASH_ATTN_VERSION}"' in text
    assert "flash-attn>=" not in text
    assert "libgl1" in text


def test_the_release_records_the_training_lock():
    text = (ROOT / "orchestration" / "pipeline_dag.py").read_text(encoding="utf-8")
    assert 'lock = root / "requirements-train.lock"' in text
