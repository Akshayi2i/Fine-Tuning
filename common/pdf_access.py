"""Encrypted PDFs: opened once - with their password when they need one - and passed on unencrypted.

Two kinds of encrypted PDF reach the pipeline:

* **Restricted.** It opens without a password and only limits what a reader
  may do (print, copy, edit); carriers often issue policies so. PyMuPDF renders
  and reads it as it is, but not every reader downstream need: MinerU is sent
  the PDF whole when it reads it in one task.
* **Locked.** It needs a password to open: its pages are encrypted (AES or
  RC4), and nothing reads them without the password - which only its sender
  has. It is never guessed or broken: a locked PDF without its password is
  refused, saying so.

:func:`unlocked_pdf` takes either and returns the same document unencrypted:
the pages, their content and images unchanged (the pixels rendered from it are
the never-encrypted original's), the encryption and its restrictions gone, so
every reader downstream reads one plain PDF. A PDF that is not encrypted comes
back as the very bytes it was.

The password is used here, once, to open the document. It is never stored,
logged or put into an error; callers keep it out of URLs (the Azure API takes
it as a form field or a header, never a query parameter).

To unlock a copy by hand - a training PDF the audit blocked, a document for
``testing.latency_check`` - the password asked for at the prompt, never typed
on the command line:

    python -m common.pdf_access unlock locked.pdf unlocked.pdf
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

#: The longest password taken. A PDF's own limit is 127 bytes (AES-256, UTF-8);
#: anything far longer is not a password for one.
MAX_PASSWORD_CHARS = 256

#: What a locked PDF sent without its password is told.
NEEDS_PASSWORD = "the PDF needs a password to open, and none was given"
#: What a locked PDF sent with a wrong password is told: never the password itself.
WRONG_PASSWORD = "the password given does not open the PDF"
#: Where a reader with no way to take a password (the renderer, the MinerU
#: client, a testing tool reading from disk) sends the person.
UNLOCK_HINT = "unlock a copy first: python -m common.pdf_access unlock <locked.pdf> <unlocked.pdf>"


class PdfLocked(ValueError):
    """The PDF needs a password to open, and none - or a wrong one - was given."""


def is_locked(pdf: bytes) -> bool:
    """Whether ``pdf`` needs a password to open."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return bool(doc.needs_pass)


def unlocked_pdf(pdf: bytes, password: str | None = None) -> bytes:
    """``pdf`` as every reader downstream opens it.

    Not encrypted: the same bytes (``password`` is then not used). Restricted:
    re-saved without its encryption. Locked: opened with ``password`` and
    re-saved without it; :class:`PdfLocked` with no password, a wrong one or
    one too long to be a PDF's. A PDF PyMuPDF cannot open at all raises
    PyMuPDF's own error, as before.
    """
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        if doc.needs_pass:
            if not password:
                raise PdfLocked(NEEDS_PASSWORD)
            if len(password) > MAX_PASSWORD_CHARS or not doc.authenticate(password):
                raise PdfLocked(WRONG_PASSWORD)
        elif not (doc.metadata or {}).get("encryption"):
            return pdf
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_NONE)


def encryption_kind(pdf: bytes) -> str | None:
    """``"password"`` for a locked PDF, ``"restrictions"`` for a restricted one, else None."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        if doc.needs_pass:
            return "password"
        return "restrictions" if (doc.metadata or {}).get("encryption") else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m common.pdf_access",
                                     description="Unlock a copy of an encrypted PDF (the password is asked for).")
    sub = parser.add_subparsers(dest="command", required=True)
    unlock = sub.add_parser("unlock", help="write an unencrypted copy of a locked or restricted PDF")
    unlock.add_argument("source", type=Path)
    unlock.add_argument("target", type=Path)
    args = parser.parse_args(argv)

    if args.target.resolve() == args.source.resolve():
        parser.error("write the unlocked copy to another file: the original stays as it was")
    pdf = args.source.read_bytes()
    try:
        kind = encryption_kind(pdf)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF's reasons, told as they are
        print(f"{args.source}: cannot be opened ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    if kind is None:
        print(f"{args.source} is not encrypted: nothing to unlock")
        return 0
    password = getpass.getpass(f"password for {args.source.name}: ") if kind == "password" else None
    try:
        args.target.write_bytes(unlocked_pdf(pdf, password))
    except PdfLocked as exc:
        print(f"{args.source}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.target} ({'unlocked with its password' if kind == 'password' else 'restrictions removed'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
