"""scripts/pull_blob_file.py: a big download in pieces that retries a failed piece and resumes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.pull_blob_file import PullError, pull

PAYLOAD = bytes(range(256)) * 40          # 10,240 bytes


class FakeBlob:
    def __init__(self, data: bytes, *, fail_first: int = 0, always_fail_at: int | None = None) -> None:
        self.data, self.fail_first, self.always_fail_at, self.reads = data, fail_first, always_fail_at, []

    def get_blob_properties(self):
        return SimpleNamespace(size=len(self.data))

    def download_blob(self, offset=0, length=None, max_concurrency=1):
        self.reads.append(offset)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise TimeoutError("no data")
        if self.always_fail_at == offset:
            raise TimeoutError("no data")
        return SimpleNamespace(readall=lambda: self.data[offset:offset + length])


def test_a_file_is_written_whole_from_pieces(tmp_path):
    blob = FakeBlob(PAYLOAD)
    dest = tmp_path / "w.safetensors"
    assert pull(blob, dest, piece=4000, log=lambda _m: None) == len(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert blob.reads == [0, 4000, 8000]
    assert not (tmp_path / "w.safetensors.part").exists()


def test_a_failed_piece_is_read_again(tmp_path):
    blob = FakeBlob(PAYLOAD, fail_first=2)
    dest = tmp_path / "w.safetensors"
    pull(blob, dest, piece=4000, tries=5, log=lambda _m: None)
    assert dest.read_bytes() == PAYLOAD


def test_giving_up_leaves_the_part_file_to_resume(tmp_path):
    dest = tmp_path / "w.safetensors"
    with pytest.raises(PullError, match="run it again to resume"):
        pull(FakeBlob(PAYLOAD, always_fail_at=4000), dest, piece=4000, tries=2, log=lambda _m: None)
    assert not dest.exists()
    assert (tmp_path / "w.safetensors.part").stat().st_size == 4000

    resumed = FakeBlob(PAYLOAD)
    pull(resumed, dest, piece=4000, log=lambda _m: None)
    assert resumed.reads == [4000, 8000]            # picked up at the byte it stopped at
    assert dest.read_bytes() == PAYLOAD


def test_a_part_file_longer_than_the_blob_starts_again(tmp_path):
    dest = tmp_path / "w.safetensors"
    (tmp_path / "w.safetensors.part").write_bytes(b"x" * (len(PAYLOAD) + 5))
    pull(FakeBlob(PAYLOAD), dest, piece=4000, log=lambda _m: None)
    assert dest.read_bytes() == PAYLOAD
