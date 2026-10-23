"""The extraction test server (testing/extract_server.py), its OCR worker and its
laptop client - with a fake MinerU and the echo model, so no GPU is needed."""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from data_pipeline.ocr.ocr_worker import RESULT_PREFIX, serve
from inference_core.model_runner import EchoBackend, load_model
from testing import extract_client
from testing.extract_server import ExtractService, RequestError, _flag_kinds, handler_for

PDF = b"%PDF-1.7 a stand-in; the fake OCR never reads it"
PAGES = ["Personal Auto Policy Declarations\nPolicy Number PA-1\n| Vehicle | VIN |\n|---|---|\n| 1 | 1HGCM |",
         "PART A - LIABILITY COVERAGE\nWe will pay damages for bodily injury.",
         "PART B - MEDICAL PAYMENTS\nWe will pay reasonable expenses."]
ANSWER = json.dumps({"policy": {"policy_number": {"raw": "PA-1", "parsed": "PA-1", "page_ref": [1]}}})


def fake_ocr(pages=PAGES, *, ok=True):
    """MinerU as the worker answers: one image and one markdown file per page."""
    calls = []

    def ocr(pdf: Path, out: Path, *, force_ocr: bool = False):
        calls.append(pdf)
        if not ok:
            return {"ok": False, "error": "RuntimeError: no GPU"}
        out.mkdir(parents=True, exist_ok=True)
        for number, text in enumerate(pages, start=1):
            (out / f"page_{number:03d}.png").write_bytes(b"png")
            (out / f"page_{number:03d}.md").write_text(text, encoding="utf-8")
        return {"ok": True, "pages": len(pages), "seconds": 0.1, "modality": "scanned_pdf",
                "scanned_pages": len(pages), "ocr_failed_pages": []}

    ocr.calls = calls
    return ocr


@pytest.fixture
def model():
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    return load_model("base", client, backend_impl=EchoBackend(ANSWER))


# --------------------------------------------------------------------------
# The OCR worker
# --------------------------------------------------------------------------


@dataclass
class _Page:
    page_number: int
    markdown: str
    image_bytes: bytes = b"png"
    scanned: bool = True
    read_by_ocr: bool = True
    ocr_failed: bool = False


class _Engine:
    def process(self, pdf_bytes, *, device, max_long_side_px, force_ocr=None):
        assert device == "cuda" and max_long_side_px > 0
        if pdf_bytes == b"broken":
            raise ValueError("cannot open")
        return [_Page(1, "# Declarations"), _Page(2, "Conditions")]


def test_the_worker_answers_each_line_with_one_marked_result(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.pdf").write_bytes(b"broken")
    lines = io.StringIO("\n".join(json.dumps({"pdf": str(tmp_path / n), "out": str(tmp_path / f"{n}.pages")})
                                  for n in ("a.pdf", "b.pdf")) + "\n")
    out = io.StringIO()
    serve(lines, out, engine=_Engine())
    results = [json.loads(line[len(RESULT_PREFIX):]) for line in out.getvalue().splitlines()]
    assert results[0]["ok"] and results[0]["pages"] == 2 and results[0]["modality"] == "scanned_pdf"
    assert (tmp_path / "a.pdf.pages" / "page_002.md").read_text(encoding="utf-8") == "Conditions"
    assert results[1] == {"ok": False, "error": "ValueError: cannot open"}


# --------------------------------------------------------------------------
# A job
# --------------------------------------------------------------------------


def test_a_job_is_read_by_the_serving_pipeline_and_timed(tmp_path, model):
    service = ExtractService(model, fake_ocr(), tmp_path, start=False)
    job = service.submit(PDF, name="scan.pdf", lob="personal_auto")
    service.run(job)
    assert job.status == "done", job.error
    summary = job.summary
    assert summary["pages"] == 3 and summary["model"] == "base" and summary["lob"] == "personal_auto"
    assert summary["windows_planned"] >= 3 and summary["model_calls"] >= 1
    assert set(summary["timings_s"]) == {"ocr", "model", "generation", "calls", "total", "queued"}
    assert summary["text_coverage"] == 1.0 and summary["extracted"]["text_sections"] >= 1
    stored = json.loads(service.result_path(job).read_text(encoding="utf-8"))
    assert stored["extraction"]["policy"]["policy_number"]["raw"] == "PA-1"
    # The document and its pages are gone; the result stays.
    assert not (tmp_path / job.id / "document.pdf").exists() and not (tmp_path / job.id / "pages").exists()


def test_an_image_only_job_carries_no_page_text(tmp_path, model):
    service = ExtractService(model, fake_ocr(), tmp_path, start=False, keep_files=True)
    job = service.submit(PDF, name="scan.pdf", lob="personal_auto", mode="image_only")
    service.run(job)
    assert job.status == "done", job.error
    assert job.summary["text_coverage"] is None
    assert job.summary["review_flag_kinds"].get("text_sections:no_page_text") == 1
    assert (tmp_path / job.id / "document.pdf").exists()


def test_a_failed_ocr_fails_its_job_not_the_server(tmp_path, model):
    service = ExtractService(model, fake_ocr(ok=False), tmp_path, start=False)
    job = service.submit(PDF, name="scan.pdf", lob="homeowners")
    service.run(job)
    assert job.status == "failed" and "no GPU" in job.error and job.view()["status"] == "failed"


@pytest.mark.parametrize("body, lob, mode, why", [
    (b"not a pdf", "personal_auto", "ocr_plus_image", "not a PDF"),
    (PDF, "", "ocr_plus_image", "lob is required"),
    (PDF, "underwater_basket_weaving", "ocr_plus_image", "no canonical policy schema"),
    (PDF, "personal_auto", "noisy_ocr_image", "mode must be"),
])
def test_a_request_the_pipeline_cannot_read_is_refused_up_front(tmp_path, model, body, lob, mode, why):
    service = ExtractService(model, fake_ocr(), tmp_path, start=False)
    with pytest.raises(RequestError, match=why):
        service.submit(body, name="x.pdf", lob=lob, mode=mode)
    assert not service.jobs


def test_review_flags_are_counted_by_kind():
    flags = ["policy.policy_number:low_confidence", "coverages[0].premium:low_confidence",
             "window_failed:arrays:p7", "window_failed:decl:p2", "text_sections:no_page_text", "lob:uncertain"]
    assert _flag_kinds(flags) == {"lob:uncertain": 1, "low_confidence": 2,
                                  "text_sections:no_page_text": 1, "window_failed": 2}


# --------------------------------------------------------------------------
# Over HTTP, with the laptop client
# --------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path, model):
    service = ExtractService(model, fake_ocr(), tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service, "secret-token"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", service
    httpd.shutdown()


def test_a_pdf_posted_is_answered_at_once_and_its_result_polled(server, tmp_path):
    base, _service = server
    pdf = tmp_path / "plymouth__scan.pdf"
    pdf.write_bytes(PDF)
    assert extract_client.call(f"{base}/health", "anything")["status"] == "ready"
    posted = extract_client.submit(base, "secret-token", pdf, "personal_auto", "ocr_plus_image")
    assert posted["status"] in ("queued", "running", "done") and posted["poll"] == f"/jobs/{posted['job']}"
    views = extract_client.wait(base, "secret-token", [posted["job"]], poll_s=0.05)
    assert views[posted["job"]]["status"] == "done"
    result = extract_client.call(f"{base}/jobs/{posted['job']}/result", "secret-token")
    assert result["document"] == "plymouth__scan.pdf" and "extraction" in result


def test_every_call_but_health_needs_the_token(server, tmp_path):
    base, _service = server
    with pytest.raises(extract_client.ClientError, match="401"):
        extract_client.call(f"{base}/jobs", "wrong")
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(PDF)
    with pytest.raises(extract_client.ClientError, match="401"):
        extract_client.submit(base, "wrong", pdf, "personal_auto", "ocr_plus_image")
    with pytest.raises(extract_client.ClientError, match="400.*lob is required"):
        extract_client.submit(base, "secret-token", pdf, "", "ocr_plus_image")


def test_the_client_reads_a_windows_path_and_reports_a_failed_job():
    path, lob = extract_client.parse_doc(r"D:\Fine-Tuning\data\a=b\x__scan.pdf=personal_auto")
    assert str(path).endswith("x__scan.pdf") and lob == "personal_auto"
    with pytest.raises(Exception, match="<pdf>=<line of business>"):
        extract_client.parse_doc("no-line.pdf")
    report = extract_client.table([{"name": "x.pdf", "lob": "homeowners", "status": "failed",
                                    "error": "OCR failed: no GPU"}])
    assert "FAILED: OCR failed: no GPU" in report
