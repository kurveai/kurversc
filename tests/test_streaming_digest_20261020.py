import hashlib
from pathlib import Path

from scripts.run_gap_followup_queue import digest


def test_streaming_digest_is_identical_without_reading_the_whole_file(tmp_path, monkeypatch):
    data = b"0123456789"*240_001
    path = tmp_path / "wide-frame.bin"
    path.write_bytes(data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Whole-file temporary allocation is forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert digest(path) == hashlib.sha256(data).hexdigest()
    (tmp_path / "empty").touch()
    assert digest(tmp_path / "empty") == hashlib.sha256(b"").hexdigest()
