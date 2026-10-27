"""Encrypted PDFs (common.pdf_access): a restricted one read as it is, a locked
one read with the password sent beside it - and the password kept nowhere.

* ``unlocked_pdf`` returns a plain PDF untouched, and an encrypted one re-saved
  without its encryption: the same pages, the same pixels.
* The Azure API takes the password as a form field or a header, never in the
  URL; stores the PDF unencrypted; records how it came; never echoes or logs
  the password.
* The renderer, the MinerU client, the testing job server, latency_check and
  the training intake tell a locked PDF apart from a broken one, by name.
"""

from __future__ import annotations

import logging

import pymupdf
import pytest

from common import pdf_access
from common.pdf_access import (
    NEEDS_PASSWORD,
    UNLOCK_HINT,
    WRONG_PASSWORD,
    PdfLocked,
    encryption_kind,
    unlocked_pdf,
)

PASSWORD = "Open-Sesame-4417"


def _pdf(*, user_pw: str | None = None, owner_pw: str | None = None, pages: int = 2) -> bytes:
    """A small policy-like PDF: plain, restricted (owner password only) or locked (a password to open)."""
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Policy Number PA-{number}", fontsize=12)
        page.draw_rect(pymupdf.Rect(100, 100, 300, 200), color=(1, 0, 0), fill=(0, 0, 1))
    if user_pw is None and owner_pw is None:
        return document.tobytes()
    return document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=user_pw or "",
                            owner_pw=owner_pw or "owner-only-secret", permissions=int(pymupdf.PDF_PERM_ACCESSIBILITY))


PLAIN = _pdf()
RESTRICTED = _pdf(owner_pw="owner-only-secret")
LOCKED = _pdf(user_pw=PASSWORD)


def _plain_text(pdf: bytes) -> list[str]:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        assert not doc.needs_pass and not (doc.metadata or {}).get("encryption")
        return [page.get_text().strip() for page in doc]


# --------------------------------------------------------------------------
# unlocked_pdf
# --------------------------------------------------------------------------


def test_a_plain_pdf_comes_back_as_the_very_bytes_it_was():
    assert unlocked_pdf(PLAIN) is PLAIN
    assert unlocked_pdf(PLAIN, "a password nobody needed") is PLAIN
    assert encryption_kind(PLAIN) is None


def test_a_restricted_pdf_is_re_saved_unencrypted_with_the_same_pixels():
    from data_pipeline.ocr.render_only import render_pdf_pages

    assert encryption_kind(RESTRICTED) == "restrictions"
    unlocked = unlocked_pdf(RESTRICTED)
    assert _plain_text(unlocked) == ["Policy Number PA-1", "Policy Number PA-2"]
    assert render_pdf_pages(unlocked, 800, budget=None) == render_pdf_pages(PLAIN, 800, budget=None)


def test_a_locked_pdf_opens_with_its_password_only():
    from data_pipeline.ocr.render_only import render_pdf_pages

    assert encryption_kind(LOCKED) == "password" and pdf_access.is_locked(LOCKED)
    with pytest.raises(PdfLocked, match=NEEDS_PASSWORD):
        unlocked_pdf(LOCKED)
    for wrong in ("open-sesame-4417", "x" * (pdf_access.MAX_PASSWORD_CHARS + 1)):
        with pytest.raises(PdfLocked) as refused:
            unlocked_pdf(LOCKED, wrong)
        assert str(refused.value) == WRONG_PASSWORD and wrong not in str(refused.value)
    unlocked = unlocked_pdf(LOCKED, PASSWORD)
    assert _plain_text(unlocked) == ["Policy Number PA-1", "Policy Number PA-2"]
    assert render_pdf_pages(unlocked, 800, budget=None) == render_pdf_pages(PLAIN, 800, budget=None)


def test_the_unlock_command_asks_for_the_password_and_writes_a_copy(tmp_path, monkeypatch, capsys):
    source = tmp_path / "locked.pdf"
    source.write_bytes(LOCKED)
    asked = []
    monkeypatch.setattr(pdf_access.getpass, "getpass", lambda prompt: asked.append(prompt) or PASSWORD)
    assert pdf_access.main(["unlock", str(source), str(tmp_path / "unlocked.pdf")]) == 0
    assert asked == ["password for locked.pdf: "] and source.read_bytes() == LOCKED
    assert _plain_text((tmp_path / "unlocked.pdf").read_bytes())[0] == "Policy Number PA-1"
    assert "unlocked with its password" in capsys.readouterr().out

    monkeypatch.setattr(pdf_access.getpass, "getpass", lambda prompt: "wrong")
    assert pdf_access.main(["unlock", str(source), str(tmp_path / "again.pdf")]) == 1
    assert WRONG_PASSWORD in capsys.readouterr().err and not (tmp_path / "again.pdf").exists()

    plain = tmp_path / "plain.pdf"
    plain.write_bytes(PLAIN)
    assert pdf_access.main(["unlock", str(plain), str(tmp_path / "copy.pdf")]) == 0
    assert "not encrypted: nothing to unlock" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pdf_access.main(["unlock", str(source), str(source)])


# --------------------------------------------------------------------------
# The readers: renderer, MinerU client, testing tools, intake
# --------------------------------------------------------------------------


def test_the_renderer_names_a_locked_pdf_and_renders_a_restricted_one():
    from data_pipeline.ocr.render_only import render_pdf_pages

    with pytest.raises(PdfLocked, match=f"{NEEDS_PASSWORD}; {UNLOCK_HINT.split(':')[0]}"):
        render_pdf_pages(LOCKED, 800, budget=None)
    assert len(render_pdf_pages(RESTRICTED, 800, budget=None)) == 2


def test_mineru_is_sent_a_restricted_pdf_unencrypted_and_never_a_locked_one(monkeypatch):
    from data_pipeline.ocr import mineru_api

    sent = []

    def task(pdf, *_args, **_kwargs):
        sent.append(pdf)
        return ["page"] * 2, {"version": "4.0.12"}

    monkeypatch.setattr(mineru_api, "_ocr_task", task)
    mineru_api.ocr_via_api(RESTRICTED, "https://mineru.example.test")
    assert _plain_text(sent[0]) == ["Policy Number PA-1", "Policy Number PA-2"]
    mineru_api.ocr_via_api(PLAIN, "https://mineru.example.test")
    assert sent[1] is PLAIN
    with pytest.raises(PdfLocked, match="scan.pdf: the PDF needs a password"):
        mineru_api.ocr_via_api(LOCKED, "https://mineru.example.test", filename="scan.pdf")
    assert len(sent) == 2


def test_the_testing_job_server_takes_the_password_as_a_header_and_stores_the_pdf_unencrypted(tmp_path):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import EchoBackend, load_model
    from testing.extract_server import ExtractService, RequestError
    from tests.test_extract_server import ANSWER, fake_ocr

    model = load_model("base", BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw"),
                       backend_impl=EchoBackend(ANSWER))
    service = ExtractService(model, fake_ocr(), tmp_path, start=False)
    with pytest.raises(RequestError, match="needs a password to open.*X-PDF-Password header"):
        service.submit(LOCKED, name="locked.pdf", lob="personal_auto")
    with pytest.raises(RequestError, match=WRONG_PASSWORD):
        service.submit(LOCKED, name="locked.pdf", lob="personal_auto", password="wrong")
    assert not service.jobs
    job = service.submit(LOCKED, name="locked.pdf", lob="personal_auto", password=PASSWORD)
    assert _plain_text((tmp_path / job.id / "document.pdf").read_bytes())[0] == "Policy Number PA-1"


def test_latency_check_refuses_a_locked_pdf_by_name_before_anything_runs(tmp_path, capsys):
    from testing import latency_check

    locked = tmp_path / "locked scan.pdf"
    locked.write_bytes(LOCKED)
    (tmp_path / "restricted.pdf").write_bytes(RESTRICTED)
    assert latency_check._locked(locked) and not latency_check._locked(tmp_path / "restricted.pdf")
    with pytest.raises(SystemExit) as exited:
        latency_check.main(["--model-url", "https://pod-8000.proxy.runpod.net", "--mineru-url",
                            "https://pod-8001.proxy.runpod.net", "--doc", f"{locked}=personal_auto"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "need a password to open" in err and "locked scan.pdf" in err and "common.pdf_access unlock" in err


def test_the_training_audit_names_a_locked_pdf_and_how_to_unlock_it(tmp_path):
    from data_pipeline.audit import _read_pdf

    (tmp_path / "locked.pdf").write_bytes(LOCKED)
    (tmp_path / "restricted.pdf").write_bytes(RESTRICTED)
    texts, problem = _read_pdf(tmp_path / "locked.pdf")
    assert texts == [] and "password-protected" in problem and "common.pdf_access unlock" in problem
    texts, problem = _read_pdf(tmp_path / "restricted.pdf")
    assert problem is None and "Policy Number PA-1" in texts[0]


# --------------------------------------------------------------------------
# The Azure API
# --------------------------------------------------------------------------


@pytest.fixture
def service(tmp_path):
    from tests.test_azure_service import Service

    return Service(tmp_path)


def _post(service, body: bytes, *, form: dict | None = None, headers: dict | None = None, **params):
    from tests.test_azure_service import KEY

    auth = {"Authorization": f"Bearer {KEY}"}
    if form is not None:
        return service.client.post("/v1/extractions", files={"file": ("policy.pdf", body, "application/pdf")},
                                   data={"lob": "personal_auto", **form}, params=params, headers={**auth, **(headers or {})})
    return service.client.post("/v1/extractions", params={"lob": "personal_auto", **params}, content=body,
                               headers={**auth, "Content-Type": "application/pdf", **(headers or {})})


def _stored_pdf(service, job_id: str) -> bytes:
    return service.store.read_pdf("default", job_id)


def test_a_locked_pdf_is_refused_without_its_password_saying_how_to_send_it(service):
    answer = _post(service, LOCKED)
    assert answer.status_code == 400
    assert answer.json()["error"] == (f"{NEEDS_PASSWORD}: send its password as the multipart field `password` "
                                      "or the X-PDF-Password header")
    answer = _post(service, LOCKED, form={"password": "Wrong-4417"})
    assert answer.status_code == 400 and answer.json()["error"] == WRONG_PASSWORD
    assert "Wrong-4417" not in answer.text
    assert service.stored() == [] and len(service.queue) == 0


@pytest.mark.parametrize("how", ["form", "header"])
def test_a_locked_pdf_sent_with_its_password_is_stored_unencrypted_and_the_password_nowhere(service, caplog, how):
    caplog.set_level(logging.DEBUG)
    sent = {"form": {"password": PASSWORD}} if how == "form" else {"headers": {"X-PDF-Password": PASSWORD}}
    answer = _post(service, LOCKED, **sent)
    assert answer.status_code == 202, answer.text
    job_id = answer.json()["job_id"]
    assert _plain_text(_stored_pdf(service, job_id))[0] == "Policy Number PA-1"
    view = service.status(job_id)
    assert view["encrypted"] == "password" and view["pages"] == 2
    # Not in the record, the queue, the stored blobs or the log.
    stored = b"".join(data for data in service.blob._store.values())
    assert PASSWORD.encode() not in stored
    assert PASSWORD not in repr(service.store.read_status("default", job_id))
    assert PASSWORD not in repr(service.queue.__dict__)
    assert PASSWORD not in caplog.text and "it came with password" in caplog.text


def test_a_restricted_pdf_is_taken_without_a_password_and_stored_unencrypted(service):
    answer = _post(service, RESTRICTED)
    assert answer.status_code == 202, answer.text
    job_id = answer.json()["job_id"]
    assert _plain_text(_stored_pdf(service, job_id))[1] == "Policy Number PA-2"
    assert service.status(job_id)["encrypted"] == "restrictions"
    plain = _post(service, PLAIN).json()["job_id"]
    assert service.status(plain)["encrypted"] is None and _stored_pdf(service, plain) == PLAIN


def test_a_password_in_the_url_is_refused_before_anything_is_read(service):
    answer = _post(service, LOCKED, password=PASSWORD)
    assert answer.status_code == 400 and "never put the PDF's password in the URL" in answer.json()["error"]
    assert PASSWORD not in answer.text
    assert service.stored() == [] and len(service.queue) == 0
