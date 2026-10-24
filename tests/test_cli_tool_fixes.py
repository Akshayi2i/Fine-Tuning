"""The operator's command-line tools: the extraction test server and its client,
the timing runner, the pipeline CLI and the pod launcher - each refusal made
before a command detaches, every result written, no request dropped.

All in process: the echo model, fake readers, a stub tmux. No GPU, no network."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from inference_core.model_runner import EchoBackend, load_model
from testing import extract_client, extract_server, latency_check
from testing.extract_server import ExtractService, RequestError, handler_for

ROOT = Path(__file__).resolve().parent.parent
PDF = b"%PDF-1.7 a stand-in; the fake readers never read it"
PAGES = ["Personal Auto Policy Declarations\nPolicy Number PA-1",
         "PART A - LIABILITY COVERAGE\nWe will pay damages for bodily injury.",
         "PART B - MEDICAL PAYMENTS\nWe will pay reasonable expenses."]
ANSWER = json.dumps({"policy": {"policy_number": {"raw": "PA-1", "parsed": "PA-1", "page_ref": [1]}}})
TOKEN = "secret-token"


@pytest.fixture
def model():
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    return load_model("base", client, backend_impl=EchoBackend(ANSWER))


@pytest.fixture
def no_pod(monkeypatch):
    """Off the pod whatever this machine is: nothing detaches."""
    import orchestration.detach as detach

    monkeypatch.setattr(detach, "should_detach", lambda: False)


@pytest.fixture
def pod(monkeypatch):
    """On the pod, with the detach stubbed: the commands it would start are recorded."""
    import orchestration.detach as detach

    started: list[list[str]] = []
    monkeypatch.setattr(detach, "should_detach", lambda: True)
    monkeypatch.setattr(detach, "detach", lambda command, hint: started.append(list(command)) or f"{hint}-test")
    return started


def fake_ocr(pages=PAGES):
    def ocr(pdf: Path, out: Path, *, force_ocr: bool = False):
        out.mkdir(parents=True, exist_ok=True)
        for number, text in enumerate(pages, start=1):
            (out / f"page_{number:03d}.png").write_bytes(b"png")
            (out / f"page_{number:03d}.md").write_text(text, encoding="utf-8")
        return {"ok": True, "pages": len(pages), "seconds": 0.1, "modality": "scanned_pdf",
                "scanned_pages": len(pages), "ocr_failed_pages": []}
    return ocr


class FieldsFirstReader:
    """The served MinerU's reader for fields first. ``fail_cited``: the cited
    pages cannot be read; ``fail_full``: the full text fails after its turn."""

    def __init__(self, *, fail_cited=False, fail_full=False):
        self.fail_cited, self.fail_full = fail_cited, fail_full

    def read(self, pdf, pages_dir, *, name, with_text=True, publish_pages=None):
        pages_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for number in range(1, len(PAGES) + 1):
            (pages_dir / f"page_{number:03d}.png").write_bytes(b"png")
            paths.append(str(pages_dir / f"page_{number:03d}.png"))
        texts = {i + 1: text for i, text in enumerate(PAGES)} if with_text else {}
        return {"images": paths, "texts": texts, "ocr": {}, "links": None,
                "timings_s": {"render": 0.1, "links": None, "ocr": 0.1}}

    def ocr_text(self, pdf, *, name, page_count=None, gate=None, pages=None):
        if gate is not None:
            gate()
            if self.fail_full:
                raise RuntimeError("MinerU went away during the full text")
        elif self.fail_cited:
            raise RuntimeError("MinerU unreachable")
        numbers = pages or list(range(1, len(PAGES) + 1))
        texts = {number: PAGES[number - 1] for number in numbers}
        return {"pages": list(texts.values()), "texts": texts, "info": {"version": "4.0.12", "tier": "basic"},
                "seconds": 0.2}


@pytest.fixture
def serve():
    servers = []

    def start(service, token=TOKEN):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service, token))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"

    yield start
    for httpd in servers:
        httpd.shutdown()


def _pdf(folder: Path, name: str, data: bytes = PDF) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(data)
    return folder / name


def _client(base, out, *docs, extra=()):
    argv = ["--url", base, "--token", TOKEN, "--poll", "0.02", "--out", str(out), *extra]
    for pdf, lob in docs:
        argv += ["--doc", f"{pdf}={lob}"]
    return extract_client.main(argv)


# --------------------------------------------------------------------------
# The client: every result written, whatever happened (M6, L8, L11)
# --------------------------------------------------------------------------


def test_a_fields_first_job_whose_full_text_failed_is_reported_not_a_crash(tmp_path, model, serve, no_pod):
    """Its timings have no total: the table raised KeyError and nothing was written."""
    _httpd, base = serve(ExtractService(model, FieldsFirstReader(fail_full=True), tmp_path / "work",
                                        default_mode="fields_first"))
    out = tmp_path / "out"
    assert _client(base, out, (_pdf(tmp_path, "scan.pdf"), "personal_auto")) == 1
    report = (out / "summary.md").read_text(encoding="utf-8")
    assert "| fields served, full text failed: RuntimeError: MinerU went away" in report
    [view] = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert view["status"] == "done" and "total" not in view["summary"]["timings_s"]
    assert (out / "scan.json").is_file()                            # the fields were served: saved


def test_a_failed_job_shows_failed_even_with_a_summary(tmp_path, model, serve, no_pod):
    _httpd, base = serve(ExtractService(model, FieldsFirstReader(fail_cited=True, fail_full=True),
                                        tmp_path / "work", default_mode="fields_first"))
    out = tmp_path / "out"
    assert _client(base, out, (_pdf(tmp_path, "scan.pdf"), "personal_auto")) == 1
    [view] = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert view["status"] == "failed" and view["summary"]
    assert "| FAILED: full text: RuntimeError" in (out / "summary.md").read_text(encoding="utf-8")
    assert not (out / "scan.json").exists()


def test_the_table_leaves_a_timing_the_server_did_not_report_blank():
    view = {"name": "a.pdf", "lob": "personal_auto", "status": "done", "fields_s": 4.2,
            "summary": {"document": "a.pdf", "lob": "personal_auto", "pages": 3, "windows_planned": 3,
                        "model_calls": 1, "timings_s": {"model": 2.0}, "schema_valid": True,
                        "review_flags": 0, "text_coverage": None}}
    row = extract_client.table([view]).splitlines()[-1]
    assert row == "| a.pdf | personal_auto | done | 3 | 3 | 1 |  | 2.0 | 4.2 |  | True | 0 |  |"


def test_documents_of_one_name_and_a_summary_pdf_each_keep_their_file(tmp_path, model, serve, no_pod):
    _httpd, base = serve(ExtractService(model, fake_ocr(), tmp_path / "work"))
    out = tmp_path / "out"
    docs = [(_pdf(tmp_path / "auto", "policy.pdf"), "personal_auto"),
            (_pdf(tmp_path / "home", "policy.pdf"), "homeowners"),
            (_pdf(tmp_path, "summary.pdf"), "personal_auto")]
    assert _client(base, out, *docs) == 0
    assert sorted(p.name for p in out.iterdir()) == ["policy-2.json", "policy.json", "summary-2.json",
                                                     "summary.json", "summary.md"]
    assert json.loads((out / "policy-2.json").read_text(encoding="utf-8"))["lob"] == "homeowners"
    views = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [v["output"] for v in views] == ["policy.json", "policy-2.json", "summary-2.json"]
    assert extract_server.unique_names(["a", "A", "summary", "a-2"]) == ["a", "A-2", "summary-2", "a-2-2"]


def test_a_line_the_server_would_refuse_is_refused_before_anything_is_sent(tmp_path, model, serve, no_pod,
                                                                           capsys):
    service = ExtractService(model, fake_ocr(), tmp_path / "work", start=False)
    _httpd, base = serve(service)
    with pytest.raises(SystemExit):
        _client(base, tmp_path / "out", (_pdf(tmp_path, "a.pdf"), "personal_auto"),
                (_pdf(tmp_path, "b.pdf"), "homeowner"))
    assert "no canonical policy schema for line 'homeowner'" in capsys.readouterr().err
    assert not service.jobs


def test_a_later_refusal_still_collects_the_jobs_queued_before_it(tmp_path, model, serve, no_pod, monkeypatch):
    service = ExtractService(model, fake_ocr(), tmp_path / "work")
    real = service.submit
    calls = []

    def submit(pdf, **kwargs):
        calls.append(kwargs["name"])
        if len(calls) == 2:
            raise RequestError("fields_first needs the served MinerU")
        return real(pdf, **kwargs)

    monkeypatch.setattr(service, "submit", submit)
    _httpd, base = serve(service)
    out = tmp_path / "out"
    docs = [(_pdf(tmp_path, n), "personal_auto") for n in ("a.pdf", "b.pdf", "c.pdf")]
    assert _client(base, out, *docs) == 1
    assert calls == ["a.pdf", "b.pdf"]                              # nothing after the refusal is sent
    views = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [v["status"] for v in views] == ["done", "not_sent", "not_sent"]
    assert "400" in views[1]["error"] and (out / "a.json").is_file()
    assert "| NOT SENT: 400 from /jobs" in (out / "summary.md").read_text(encoding="utf-8")


def test_an_unreachable_server_is_reported_cleanly(tmp_path, no_pod, capsys):
    with socket.socket() as probe:                                 # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    out = tmp_path / "out"
    assert _client(f"http://127.0.0.1:{port}", out, (_pdf(tmp_path, "a.pdf"), "personal_auto")) == 1
    assert "could not reach" in capsys.readouterr().err
    [view] = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert view["status"] == "not_sent" and "could not reach" in view["error"]


def test_a_pdf_over_the_servers_cap_is_refused_before_any_upload(tmp_path, model, serve, no_pod, monkeypatch):
    monkeypatch.setattr(extract_server, "MAX_PDF_BYTES", 1024)
    service = ExtractService(model, fake_ocr(), tmp_path / "work", start=False)
    _httpd, base = serve(service)
    out = tmp_path / "out"
    docs = [(_pdf(tmp_path, "small.pdf"), "personal_auto"),
            (_pdf(tmp_path, "big.pdf", b"%PDF" + b"0" * 4096), "personal_auto")]
    assert _client(base, out, *docs) == 1
    assert not service.jobs                                         # not even the small one
    views = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert all("big.pdf" in v["error"] for v in views)


def test_the_client_follows_no_redirect_and_sends_only_over_tls(tmp_path, no_pod, capsys):
    asked: list[str] = []

    class Redirecting(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            asked.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Redirecting)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(extract_client.ClientError, match="redirected"):
            extract_client.call(f"http://127.0.0.1:{httpd.server_address[1]}/health", TOKEN)
    finally:
        httpd.shutdown()
    assert asked == ["/health"]
    with pytest.raises(SystemExit):
        _client("http://10.1.2.3:8000", tmp_path / "out", (_pdf(tmp_path, "a.pdf"), "personal_auto"))
    assert "is not HTTPS" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The server: requests answered, never dropped (L9, L10)
# --------------------------------------------------------------------------


def _raw(port: int, request: bytes) -> str:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.sendall(request)
        answer = conn.recv(65536).decode("latin-1")
    return answer.split("\r\n", 1)[0] if answer else "<no answer>"


def test_odd_headers_are_answered_not_dropped(tmp_path, model, serve):
    service = ExtractService(model, fake_ocr(), tmp_path, start=False)
    httpd, _base = serve(service)
    port = httpd.server_address[1]
    assert " 401 " in _raw(port, b"GET /jobs HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer caf\xe9\r\n"
                                 b"Connection: close\r\n\r\n")
    assert " 401 " in _raw(port, b"POST /jobs?lob=personal_auto HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer \xff"
                                 b"\r\nContent-Length: 4\r\nConnection: close\r\n\r\n%PDF")
    for length in (b"abc", b"-5", b"\xb2"):
        answer = _raw(port, b"POST /jobs?lob=personal_auto HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer "
                            + TOKEN.encode() + b"\r\nContent-Length: " + length + b"\r\nConnection: close\r\n\r\n%PDF")
        assert " 400 " in answer, (length, answer)
    assert not service.jobs


def test_the_job_table_is_only_changed_or_listed_under_its_lock(tmp_path, model, serve):
    """/health and /jobs iterated the table while a POST inserted into it:
    'dictionary changed size during iteration', the request dropped."""
    service = ExtractService(model, fake_ocr(), tmp_path, start=False)
    lock = service.jobs_lock

    class Guarded(dict):
        def __setitem__(self, key, value):
            assert lock.locked(), "a job inserted without the lock"
            super().__setitem__(key, value)

        def values(self):
            assert lock.locked(), "the jobs listed without the lock"
            return super().values()

    service.jobs = Guarded()
    _httpd, base = serve(service)
    pdf = _pdf(tmp_path, "a.pdf")
    posted = extract_client.submit(base, TOKEN, pdf, "personal_auto", None)
    assert [view["job"] for view in extract_client.call(f"{base}/jobs", TOKEN)] == [posted["job"]]
    health = extract_client.call(f"{base}/health", "-")
    assert health["jobs"]["queued"] == 1 and health["max_pdf_bytes"] == extract_server.MAX_PDF_BYTES
    assert extract_client.call(f"{base}/jobs/{posted['job']}", TOKEN)["status"] == "queued"


# --------------------------------------------------------------------------
# Refused before the detach (M5, M13, L7, --concurrency 0)
# --------------------------------------------------------------------------


class _Started(Exception):
    pass


def test_the_server_refuses_a_tier_other_than_the_pinned_one(tmp_path, pod, monkeypatch, capsys):
    base = ["--work-dir", str(tmp_path), "--model-url", "https://m", "--mineru-url", "https://o"]
    with pytest.raises(SystemExit):
        extract_server.main([*base, "--tier", "flash"])
    assert "--tier flash is not the pinned tier (basic)" in capsys.readouterr().err
    assert pod == []
    for argv in ([*base, "--concurrency", "0"], ["--work-dir", str(tmp_path), "--model-url", "http://10.1.2.3:8000"],
                 ["--work-dir", str(tmp_path), "--mineru-url", "http://10.1.2.3:8001"]):
        with pytest.raises(SystemExit):
            extract_server.main(argv)
    assert pod == []
    # --any-mineru lets it through, to the detach.
    assert extract_server.main([*base, "--tier", "flash", "--any-mineru"]) == 0 and len(pod) == 1


def test_the_timing_runner_refuses_a_bad_setup_before_it_detaches(tmp_path, pod, monkeypatch, capsys):
    for var in ("AZURE_STORAGE_ACCOUNT_URL", "AZURE_STORAGE_SAS_TOKEN", "AZURE_RAW_SAS_TOKEN",
                "AZURE_STORAGE_CONNECTION_STRING", latency_check.LINK_TOKEN_ENV):
        monkeypatch.delenv(var, raising=False)
    # The run's environment as pod_run.sh would give it: here, this process's own.
    monkeypatch.setattr(latency_check, "run_environment", lambda detaching: dict(os.environ))
    pdf = _pdf(tmp_path, "a.pdf")
    base = ["--model-url", "https://m", "--mineru-url", "https://o", "--doc", f"{pdf}=personal_auto"]
    for extra, said in ((["--images", "link"], "no storage login"),
                        (["--model-url", "http://10.1.2.3:8000"], "the model server at 'http://10.1.2.3:8000' is not"),
                        (["--mineru-url", "http://10.1.2.3:8001"], "the MinerU server at"),
                        (["--concurrency", "0"], "must be 1 or more"),
                        (["--runs", "standard,fields-first,standard"], "names standard more than once")):
        with pytest.raises(SystemExit):
            latency_check.main([*base, *extra])
        assert said in capsys.readouterr().err, extra
    assert pod == []

    # A link token that may write is refused; a login holding the account key signs its own links.
    monkeypatch.setenv(latency_check.LINK_TOKEN_ENV, "sv=2024&sr=c&sp=rw&sig=x")
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING",
                       "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=a2V5;EndpointSuffix=core.windows.net")
    with pytest.raises(SystemExit):
        latency_check.main([*base, "--images", "link", "--link-token", "sv=2024&sr=c&sp=rw&sig=x"])
    assert "sp=rw" in capsys.readouterr().err and pod == []
    monkeypatch.delenv(latency_check.LINK_TOKEN_ENV)
    assert latency_check.main([*base, "--images", "link"]) == 0 and len(pod) == 1


def test_the_runs_environment_is_the_callers_off_the_pod(monkeypatch):
    monkeypatch.setenv("FIDEON_REMOTE_TOKEN", "mine")
    assert latency_check.run_environment(False)["FIDEON_REMOTE_TOKEN"] == "mine"


def test_extract_flags_are_refused_before_the_pipeline_cli_detaches(pod, capsys):
    from orchestration import run as cli

    for argv, said in ((["extract", "--model", "v2", "--adapter", "x", "--source-ids", "a"], "use --model base"),
                       (["extract", "--model", "base", "--split", "test"], "--split needs --corpus")):
        with pytest.raises(SystemExit) as exited:
            cli.main(argv)
        assert exited.value.code == 2 and said in capsys.readouterr().err
    assert pod == []
    assert cli.main(["extract", "--model", "base", "--split", "test", "--corpus", "v1"]) == 0 and len(pod) == 1


def test_every_command_in_the_cli_docstring_parses():
    from orchestration import run as cli

    lines = [line.split("  #")[0].strip() for line in cli.__doc__.splitlines()
             if line.strip().startswith("python -m orchestration.run")]
    assert any(" extract " in line for line in lines)
    for line in lines:
        cli.build_parser().parse_args(shlex.split(line)[3:])


# --------------------------------------------------------------------------
# The timing runner: lines named as serving names them, one file per row (M14, L8)
# --------------------------------------------------------------------------


def test_a_doc_line_is_checked_and_named_as_serving_names_it():
    import argparse

    assert latency_check.parse_doc(r"D:\data\a=b\x.pdf=Classic_Auto") == (Path(r"D:\data\a=b\x.pdf"), "personal_auto")
    with pytest.raises(argparse.ArgumentTypeError, match="no canonical policy schema for line 'marine_cargo'"):
        latency_check.parse_doc("x.pdf=marine_cargo")


def test_the_timing_runner_writes_each_row_to_its_own_file(tmp_path, monkeypatch, no_pod):
    import data_pipeline.ocr.served_ocr as served_ocr
    import inference_core.openai_backend as openai_backend

    class Backend(EchoBackend):
        def __init__(self, base_url, token=None, model=None, **kwargs):
            super().__init__(ANSWER)
            self.model, self.images, self.sent_bytes = model or "served", kwargs.get("images", "inline"), 0

        def close(self):
            pass

    monkeypatch.setattr(openai_backend, "OpenAIChatBackend", Backend)
    monkeypatch.setattr(served_ocr, "ServedOcr", lambda *a, **k: FieldsFirstReader())
    monkeypatch.setattr(latency_check, "score", lambda result, gold, lob: {"field_exact_match_rate": 1.0})
    out = tmp_path / "out"
    docs = [_pdf(tmp_path / "auto", "policy.pdf"), _pdf(tmp_path / "home", "policy.pdf"), _pdf(tmp_path, "summary.pdf")]
    argv = ["--model-url", "https://m", "--mineru-url", "https://o", "--out", str(out), "--runs", "standard,fields-first",
            "--doc", f"{docs[0]}=Classic_Auto", "--doc", f"{docs[1]}=homeowners", "--doc", f"{docs[2]}=personal_auto"]
    assert latency_check.main(argv) == 0
    rows = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [row["output"] for row in rows] == ["policy.json", "policy.fields-first.json", "policy-2.json",
                                               "policy.fields-first-2.json", "summary-2.json",
                                               "summary.fields-first.json"]
    assert all((out / row["output"]).is_file() for row in rows)
    assert [row["lob"] for row in rows[:4]] == ["personal_auto", "personal_auto", "homeowners", "homeowners"]


# --------------------------------------------------------------------------
# The truncation message names why the answer was cut
# --------------------------------------------------------------------------


@pytest.mark.parametrize("finish", ["length", "repetition", "stream_cut"])
def test_a_cut_off_answer_says_how_it_was_cut(finish):
    from inference_core.model_runner import Generation
    from serving.pipeline import PipelineError, _parse_generation

    with pytest.raises(PipelineError, match=f"cut off \\(finish {finish}"):
        _parse_generation(Generation(text="{}", finish_reason=finish), SimpleNamespace(schema_doc_type="policy"),
                          refuse_truncated=True)


# --------------------------------------------------------------------------
# The pod launcher keeps the caller's own settings over .env (M15)
# --------------------------------------------------------------------------


@pytest.fixture
def launcher(tmp_path):
    """pod_run.sh in a repo of its own, with a .env and a tmux that runs the session at once."""
    if shutil.which("bash") is None:
        pytest.skip("no bash here")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "pod_run.sh", repo / "scripts" / "pod_run.sh")
    (repo / ".env").write_text("FIDEON_REMOTE_TOKEN=\nEXTRACT_SERVER_TOKEN=from-dotenv\n", encoding="utf-8",
                               newline="\n")
    (repo / ".env.example").write_text("FIDEON_REMOTE_TOKEN=\nPAGE_LINK_SAS_TOKEN=\n", encoding="utf-8", newline="\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "tmux").write_text('#!/usr/bin/env bash\n[ "$1" = "has-session" ] && exit 1\n'
                               'if [ "$1" = "new-session" ]; then eval "${@: -1}" </dev/null >/dev/null 2>&1; fi\n'
                               "exit 0\n", encoding="utf-8", newline="\n")
    (fake / "tmux").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("FIDEON_REMOTE_TOKEN", "EXTRACT_SERVER_TOKEN",
                                                            "PAGE_LINK_SAS_TOKEN")}
    # Forward slashes: Git Bash on Windows takes C:/... as a path, C:\... not everywhere.
    env.update({"PATH": f"{fake}{os.pathsep}{env.get('PATH', '')}", "FIDEON_LOG_DIR": (tmp_path / "logs").as_posix()})

    def run(*args, **extra):
        return subprocess.run(["bash", (repo / "scripts" / "pod_run.sh").as_posix(), *args], env={**env, **extra},
                              capture_output=True, timeout=60)
    return run, tmp_path / "logs"


def test_an_inline_token_reaches_the_detached_run_over_an_empty_env_line(launcher):
    run, logs = launcher
    shown = 'echo "remote=[$FIDEON_REMOTE_TOKEN] server=[$EXTRACT_SERVER_TOKEN] link=[${PAGE_LINK_SAS_TOKEN-unset}]"'
    done = run("start", "job", "--", "bash", "-c", shown, FIDEON_REMOTE_TOKEN="inline token $1")
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    assert (logs / "job.log").read_text(encoding="utf-8").strip() == \
        "remote=[inline token $1] server=[from-dotenv] link=[unset]"
    # The token went through a file of its own, deleted once read: not in the script, not left behind.
    assert "inline token" not in (logs / "job.run.sh").read_text(encoding="utf-8")
    assert not (logs / "job.env").exists()


def test_the_launcher_shows_the_environment_a_run_would_get(launcher):
    run, _logs = launcher
    done = run("env", FIDEON_REMOTE_TOKEN="mine")
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    env = dict(item.split("=", 1) for item in done.stdout.decode().split("\0") if "=" in item)
    assert env["FIDEON_REMOTE_TOKEN"] == "mine" and env["EXTRACT_SERVER_TOKEN"] == "from-dotenv"
    assert env["FIDEON_DETACHED"] == "1"


def test_the_timing_runner_checks_against_the_environment_the_launcher_gives(launcher, tmp_path, monkeypatch):
    """Before the detach, the storage login is looked for where the run will have
    it: the repo's .env, under the caller's own values."""
    import orchestration.detach as detach

    monkeypatch.setattr(detach, "POD_RUN", tmp_path / "repo" / "scripts" / "pod_run.sh")
    monkeypatch.delenv("EXTRACT_SERVER_TOKEN", raising=False)
    monkeypatch.setenv("FIDEON_REMOTE_TOKEN", "mine")
    env = latency_check.run_environment(True)
    assert env["FIDEON_REMOTE_TOKEN"] == "mine" and env["EXTRACT_SERVER_TOKEN"] == "from-dotenv"


class _BlankCitedPages(FieldsFirstReader):
    """The cited page is read but has no text: nothing could be checked."""

    def ocr_text(self, pdf, *, name, page_count=None, gate=None, pages=None):
        read = super().ocr_text(pdf, name=name, page_count=page_count, gate=gate, pages=pages)
        if gate is None:
            read["texts"] = {number: "" for number in read["texts"]}
            read["pages"] = ["" for _ in read["pages"]]
        return read


def test_fields_nothing_could_check_wait_for_the_full_text(tmp_path, model):
    """read_fields serves nothing (served None) when the cited pages have no text:
    the job is never fields_ready, and its fields come with the full text."""
    service = ExtractService(model, _BlankCitedPages(), tmp_path, start=False, default_mode="fields_first")
    job = service.submit(PDF, name="scan.pdf", lob="personal_auto")
    service.run(job)
    assert job.status == "running" and job.fields_s is None and not service.result_path(job).exists()
    assert job.summary["fields_served"] is False and job.summary["fields_verified"] is False
    service.add_text(job)
    final = json.loads(service.result_path(job).read_text(encoding="utf-8"))
    assert job.status == "done" and job.error is None and final["extraction"]["text_sections"]
