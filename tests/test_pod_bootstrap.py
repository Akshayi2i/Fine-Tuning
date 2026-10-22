"""The one-command pod bootstrap, and the two environments it builds.

MinerU 1.x cannot share an environment with vLLM 0.11 / torch 2.8, so OCR has its
own. The pipeline in the training environment then refuses clearly, rather than
with an ImportError, when documents still need OCR.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "pod_bootstrap.sh"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")
BASH = shutil.which("bash")


# --------------------------------------------------------------------------
# The script
# --------------------------------------------------------------------------


def test_everything_it_builds_lives_on_the_volume():
    """The container disk is wiped on stop; the volume is not."""
    assert 'VENV="$WORKSPACE/venv"' in SCRIPT
    assert 'VENV_OCR="$WORKSPACE/venv-ocr"' in SCRIPT
    assert 'MODELS="$WORKSPACE/models"' in SCRIPT
    assert 'ensure_env HF_HOME "$WORKSPACE/.cache/huggingface"' in SCRIPT
    assert 'MINERU_CONFIG="$WORKSPACE/mineru.json"' in SCRIPT


def test_mineru_runs_on_the_gpu_on_its_pinned_weights_and_a_1x_config_path_is_replaced():
    assert 'ensure_env MINERU_DEVICE_MODE cuda' in SCRIPT
    assert 'ensure_env MINERU_MODEL_SOURCE local' in SCRIPT
    # A pod set up for MinerU 1.x has magic-pdf.json in .env: replaced like a template value.
    assert 'ensure_env MINERU_TOOLS_CONFIG_JSON "$MINERU_CONFIG" "$WORKSPACE/magic-pdf.json"' in SCRIPT


def test_ocr_and_training_get_separate_environments():
    assert 'make_venv "$VENV" train requirements-train.txt' in SCRIPT
    assert 'make_venv "$VENV_OCR" ocr requirements-ocr.txt' in SCRIPT


def test_it_detaches_into_tmux_on_the_pod():
    assert "pod_run.sh start" in SCRIPT and "RUNPOD_POD_ID" in SCRIPT


def test_it_checks_the_gpu_the_mount_and_the_space_first():
    first_install = SCRIPT.index("make_venv \"$VENV\" train")
    for check in ("nvidia-smi", "mountpoint -q", "MIN_FREE_GB"):
        assert SCRIPT.index(check) < first_install, check
    assert 'MIN_FREE_GB="${FIDEON_MIN_FREE_GB:-150}"' in SCRIPT


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_the_gpu_check_survives_a_pod_with_several_gpus(tmp_path):
    """nvidia-smi writes one line per GPU. Cut short with `| head -n1`, it died
    of SIGPIPE on a 4x H200 pod and pipefail ended the bootstrap at step 1."""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "nvidia-smi").write_text(
        '#!/usr/bin/env bash\nfor i in 0 1 2 3; do echo "NVIDIA H200, 143771 MiB"; sleep 0.05; done\n',
        encoding="utf-8", newline="\n")
    (fake / "nvidia-smi").chmod(0o755)
    start = SCRIPT.index('gpus="$(nvidia-smi')
    end = SCRIPT.index("\n", SCRIPT.index('echo "GPU:', start))
    check = "set -euo pipefail\n" + SCRIPT[start:end] + "\necho survived\n"
    done = subprocess.run([BASH, "-c", check], capture_output=True, text=True,
                          env={"PATH": f"{fake.as_posix()}:/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["GPU: NVIDIA H200, 143771 MiB x4", "survived"]


def test_the_base_model_comes_at_the_pinned_revision():
    assert "snapshot_download(repo_id=model_id, revision=revision" in SCRIPT
    assert '"PIN_ME"' in SCRIPT   # refuses to download a floating revision


def test_the_spike_runs_in_both_environments():
    assert re.search(r'"\$VENV/bin/python" scripts/phase0_spike.py --skip-mineru', SCRIPT)
    assert re.search(r'"\$VENV_OCR/bin/python" scripts/phase0_spike.py --only-mineru', SCRIPT)


def test_the_script_has_unix_line_endings():
    assert b"\r\n" not in SCRIPT_PATH.read_bytes()


# --------------------------------------------------------------------------
# .env: fill the pod's keys, never overwrite the operator's
# --------------------------------------------------------------------------


def _ensure_env_function() -> str:
    start = SCRIPT.index("ensure_env() {")
    return SCRIPT[start:SCRIPT.index("\n}\n", start) + 3]


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_ensure_env_fills_blanks_and_template_defaults_but_keeps_chosen_values(tmp_path):
    (tmp_path / ".env").write_text(
        "RUNPOD_VOLUME_MOUNT=/runpod-volume\nHF_HOME=\nMINERU_TOOLS_CONFIG_JSON=/mine.json\n",
        encoding="utf-8", newline="\n",
    )
    script = _ensure_env_function() + (
        "ensure_env RUNPOD_VOLUME_MOUNT /workspace /runpod-volume\n"
        "ensure_env HF_HOME /workspace/.cache/huggingface\n"
        "ensure_env MINERU_TOOLS_CONFIG_JSON /workspace/magic-pdf.json\n"
        "ensure_env NEW_KEY value\n"
    )
    subprocess.run([BASH, "-c", script], cwd=tmp_path, check=True, capture_output=True)
    env = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert env == [
        "RUNPOD_VOLUME_MOUNT=/workspace",               # the template's default: replaced
        "HF_HOME=/workspace/.cache/huggingface",        # blank: filled
        "MINERU_TOOLS_CONFIG_JSON=/mine.json",          # chosen: kept
        "NEW_KEY=value",                                # missing: appended
    ]


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_the_templates_connection_string_survives_being_sourced(tmp_path):
    """pod_run.sh sources .env with bash. Unquoted, the value ends at its first ';'."""
    line = next(row for row in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
                if row.startswith("AZURE_STORAGE_CONNECTION_STRING="))
    (tmp_path / "env").write_text(line + "\n", encoding="utf-8", newline="\n")
    out = subprocess.run([BASH, "-c", 'set -a; . ./env; printf %s "$AZURE_STORAGE_CONNECTION_STRING"'],
                         cwd=tmp_path, check=True, capture_output=True, text=True).stdout
    assert out.endswith("AccountKey=..."), out


def test_an_unquoted_connection_string_is_refused():
    assert "contains ';' but is not quoted" in SCRIPT


def test_an_unquoted_sas_token_is_refused_and_either_login_will_do():
    """Sourced by bash, an unquoted token's '&' sends the rest to the background."""
    assert "contains '&' but is not quoted" in SCRIPT
    assert "for sas_key in AZURE_STORAGE_SAS_TOKEN AZURE_RAW_SAS_TOKEN; do" in SCRIPT
    assert "AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_SAS_TOKEN, or AZURE_STORAGE_CONNECTION_STRING" in SCRIPT


def _login_check(tmp_path: Path, env_text: str) -> subprocess.CompletedProcess:
    """Run the script's .env checks, from its quoting checks to its login check,
    against ``env_text`` as the pod's .env."""
    start = SCRIPT.index("env_value() {")
    end = SCRIPT.index("\n# --- 3 and 4.", start)
    (tmp_path / ".env").write_text(env_text, encoding="utf-8", newline="\n")
    check = (
        "set -euo pipefail\n"
        'die() { echo "pod_bootstrap: $*" >&2; exit 1; }\n'
        'PENDING=(); REPO="$(pwd)"\n'
        + SCRIPT[start:end] + "\n"
        '[ ${#PENDING[@]} -eq 0 ] || printf "PENDING %s\\n" "${PENDING[@]}"\n'
        'printf "TOKEN=%s\\n" "${AZURE_STORAGE_SAS_TOKEN:-}"\n'
    )
    # Only PATH: a login in the test runner's own environment must not leak in.
    return subprocess.run([BASH, "-c", check], cwd=tmp_path, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"})


_URL = "AZURE_STORAGE_ACCOUNT_URL=https://acct.blob.core.windows.net\n"
_CONN = 'AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=fake"\n'


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("line, key", [
    ("export AZURE_STORAGE_SAS_TOKEN=sv=1&sig=fake\n", "AZURE_STORAGE_SAS_TOKEN"),
    ("   AZURE_STORAGE_SAS_TOKEN=sv=1&sig=fake\n", "AZURE_STORAGE_SAS_TOKEN"),
    ("AZURE_RAW_SAS_TOKEN=sv=1&sig=fake\n", "AZURE_RAW_SAS_TOKEN"),
    ("  export AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=a;AccountKey=fake\n",
     "AZURE_STORAGE_CONNECTION_STRING"),
])
def test_an_unquoted_login_is_refused_however_the_line_is_written(tmp_path, line, key):
    """bash sources `export KEY=...` and an indented line as readily as a bare
    one. Matched only from the first column, those slipped past the check and
    the token was cut at its '&' - unset after sourcing."""
    done = _login_check(tmp_path, _URL + line)
    assert done.returncode == 1, done.stdout
    assert f"{key} in .env contains" in done.stderr and "but is not quoted" in done.stderr


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_a_quoted_exported_token_is_read_whole(tmp_path):
    done = _login_check(tmp_path, _URL + 'export AZURE_STORAGE_SAS_TOKEN="sv=1&sig=fake"\n')
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["TOKEN=sv=1&sig=fake"]


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("env_text", [
    _URL + "AZURE_STORAGE_SAS_TOKEN=\n" + _CONN,
    'AZURE_STORAGE_SAS_TOKEN="sv=1&sig=fake"\n' + _CONN,
    'AZURE_RAW_SAS_TOKEN="sv=1&sig=fake"\n' + _CONN,
])
def test_half_a_sas_login_is_reported_even_beside_a_connection_string(tmp_path, env_text):
    """Python refuses half a SAS login on every Blob call rather than falling
    back to the connection string, so the bootstrap must not call it done."""
    done = _login_check(tmp_path, env_text)
    assert done.returncode == 0, done.stderr
    assert any(line.startswith("PENDING ") and "half a SAS login" in line
               for line in done.stdout.splitlines()), done.stdout


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_a_raw_token_needs_a_separate_raw_container(tmp_path):
    login = _URL + 'AZURE_STORAGE_SAS_TOKEN="sv=1&sig=fake"\nAZURE_RAW_SAS_TOKEN="sv=1&sig=raw-fake"\n'
    done = _login_check(tmp_path, login)
    assert "AZURE_RAW_SAS_TOKEN is set" in done.stdout and "no separate raw container" in done.stdout

    done = _login_check(tmp_path, login + "AZURE_RAW_CONTAINER=insurance-extraction-raw\n")
    assert done.returncode == 0, done.stderr
    assert "PENDING" not in done.stdout


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_one_whole_login_is_enough_and_none_is_reported(tmp_path):
    assert "PENDING" not in _login_check(tmp_path, _CONN).stdout
    assert "PENDING" not in _login_check(tmp_path, _URL + 'AZURE_STORAGE_SAS_TOKEN="sv=1&sig=fake"\n').stdout
    missing = _login_check(tmp_path, "AZURE_STORAGE_ACCOUNT_URL=\n").stdout
    assert "PENDING set AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_SAS_TOKEN, or" in missing


# --------------------------------------------------------------------------
# The pipeline in the training environment, without MinerU
# --------------------------------------------------------------------------


def _ctx(pending):
    from types import SimpleNamespace

    return SimpleNamespace(raw=object(), tenant_id=None, doc_types=["policy"],
                           ocr_engine=None, pending=pending)


def test_pending_ocr_without_mineru_names_the_ocr_environment(monkeypatch):
    from data_pipeline.ocr import run_mineru
    from orchestration import pipeline_dag

    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    monkeypatch.setattr(run_mineru, "find_unprocessed", lambda raw, dt, tenant: ["d1", "d2"])
    with pytest.raises(pipeline_dag.PipelineError) as err:
        pipeline_dag.stage_preprocessing(_ctx(2))
    message = str(err.value)
    assert "policy: 2" in message
    assert f"{pipeline_dag.OCR_VENV}/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy" in message


def test_nothing_pending_needs_no_mineru(monkeypatch):
    from data_pipeline.ocr import run_mineru
    from orchestration import pipeline_dag

    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    monkeypatch.setattr(run_mineru, "find_unprocessed", lambda raw, dt, tenant: [])
    monkeypatch.setattr(run_mineru, "MinerUEngine", lambda: object())
    result = pipeline_dag.stage_preprocessing(_ctx(0))
    assert result.status == "completed"


# --------------------------------------------------------------------------
# The spike's OCR-environment mode
# --------------------------------------------------------------------------


def test_only_mineru_needs_a_pdf():
    from scripts import phase0_spike

    with pytest.raises(SystemExit):
        phase0_spike.main(["--only-mineru"])
