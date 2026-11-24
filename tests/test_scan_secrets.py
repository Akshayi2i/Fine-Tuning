"""The secret scan matches secrets, not the words around them (review R5)."""

from __future__ import annotations

import base64
from pathlib import Path

from scripts.scan_secrets import AZURITE_KEY, main, scan, scan_text


def _fake_key() -> str:
    return base64.b64encode(b"z" * 64).decode()  # 88 characters, like a real key


def test_placeholders_are_not_secrets() -> None:
    for line in ('AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=..."',
                 "AccountName=acct;AccountKey=k",
                 "credential-shaped strings (`AccountKey=`, SAS signatures)"):
        assert scan_text("notes.md", line) == []


def test_a_real_shaped_storage_key_is_caught() -> None:
    found = scan_text("settings.py", f"conn = 'AccountName=a;AccountKey={_fake_key()}'")
    assert found == [(1, "Azure storage account key")]


def test_the_azurite_key_is_allowed_only_in_deploy_md() -> None:
    line = f'export X="AccountName=devstoreaccount1;AccountKey={AZURITE_KEY};"'
    assert scan_text("azure_service/DEPLOY.md", line) == []
    assert scan_text("azure_service/settings.py", line) == [(1, "Azure storage account key")]


def test_a_sas_signature_and_a_private_key_are_caught() -> None:
    sas = "https://a.blob.core.windows.net/c/b?sv=2024-01-01&se=2026&sp=r&sig=" + "Ab1%2Bcd" * 7
    assert scan_text("x.txt", sas) == [(1, "SAS signature")]
    # Joined here, never written whole: this file is scanned too once committed.
    assert scan_text("x.pem", "-----BEGIN RSA " + "PRIVATE KEY-----") == [(1, "private key block")]


def test_ssn_shapes_count_only_in_fixtures() -> None:
    assert scan_text("tests/fixtures/policy.json", '"ssn": "123-45-6789"') == [(1, "SSN-shaped string in a fixture")]
    assert scan_text("docs/notes.md", "call 123-45-6789") == []


def test_the_scan_fails_on_a_planted_key_and_passes_without_it(tmp_path: Path) -> None:
    clean = tmp_path / "clean.txt"
    clean.write_text("AccountKey=...\n", encoding="utf-8")
    assert main([str(clean)]) == 0
    planted = tmp_path / "planted.txt"
    planted.write_text(f"AccountKey={_fake_key()}\n", encoding="utf-8")
    assert main([str(tmp_path)]) == 1
    assert len(scan([tmp_path])) == 1


def test_the_committed_tree_is_clean() -> None:
    assert main([]) == 0
