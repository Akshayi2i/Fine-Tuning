"""Every third-party import is installed by some dependency group.

A module nobody declared installs as nothing on a pod, and the failure is an
ImportError at the stage that needs it — MinerU was exactly that: imported by
the OCR stage, installed by no group.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

#: Import name -> the distribution that provides it, where they differ.
DISTRIBUTION = {
    "PIL": "pillow", "fitz": "pymupdf", "yaml": "pyyaml",
    "swift": "ms-swift", "llmcompressor": "llmcompressor", "azure": "azure-storage-blob",
    "flash_attn": "flash-attn", "sklearn": "scikit-learn", "jinja2": "jinja2",
    "dotenv": "python-dotenv", "qwen_vl_utils": "qwen-vl-utils",
}
#: Installed by another declared package, never imported on its own terms.
TRANSITIVE = {"huggingface_hub": "transformers", "pymupdf": "pymupdf"}
#: Installed outside pip's extras, by scripts/setup_pod.sh.
SETUP_SCRIPT = {"flash-attn"}
#: Our own modules imported by their bare name where they are copied beside the importer: the vLLM site
#: hook falls back to ``import vllm_patches`` on a serving pod without the repository.
COPIED_BESIDE = {"vllm_patches"}


def _declared() -> set[str]:
    specs = list(PYPROJECT["project"]["dependencies"])
    for group in PYPROJECT["project"]["optional-dependencies"].values():
        specs += group
    return {re.split(r"[\s<>=!;\[~]", s, maxsplit=1)[0].lower() for s in specs}


def _imports() -> set[str]:
    local = {p.name for p in ROOT.iterdir() if p.is_dir()} | {p.stem for p in ROOT.glob("*.py")} | COPIED_BESIDE
    found: set[str] = set()
    for path in ROOT.rglob("*.py"):
        if any(part.startswith(".") or part in ("venv", "build") for part in path.parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top not in local and top != "__future__":
                    found.add(top)
    return found


def test_every_third_party_import_is_declared():
    declared = _declared() | SETUP_SCRIPT
    missing = sorted(
        name for name in _imports()
        if DISTRIBUTION.get(name, name).lower() not in declared
        and TRANSITIVE.get(name, "").lower() not in declared
    )
    assert not missing, f"imported but installed by no dependency group: {missing}"


def test_every_requirements_file_names_real_groups():
    groups = set(PYPROJECT["project"]["optional-dependencies"])
    files = sorted(ROOT.glob("requirements*.txt"))
    assert {f.name for f in files} >= {
        "requirements.txt", "requirements-ocr.txt", "requirements-train.txt",
        "requirements-serve.txt",
    }
    for path in files:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")]
        assert lines, f"{path.name} installs nothing"
        for line in lines:
            match = re.fullmatch(r"-e \.\[([a-z,]+)\]", line)
            assert match, f"{path.name}: {line!r} — versions belong in pyproject.toml"
            unknown = set(match.group(1).split(",")) - groups
            assert not unknown, f"{path.name} names unknown groups {unknown}"


def test_the_training_pod_has_vllm_for_checkpoint_selection():
    """Checkpoint selection and calibration generate with vLLM in the finetune
    process, so the training pod needs the serve group as well as train."""
    text = (ROOT / "requirements-train.txt").read_text(encoding="utf-8")
    groups = re.search(r"-e \.\[([a-z,]+)\]", text).group(1).split(",")
    assert {"train", "serve", "data"} <= set(groups)
