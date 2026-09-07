"""Idempotent write tracking.

Every ingest step records the checksum of its input and the fingerprint of its
output. Re-running a step whose input has not changed is a no-op, which is what
makes the pipeline cheap to iterate on and honest about provenance.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(*parts: str) -> str:
    """Fingerprint for non-file inputs (API query params, generator seeds)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class Manifest:
    """A JSON sidecar recording what was built, from what, and when."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def is_current(self, key: str, input_sha: str, output: Path) -> bool:
        entry = self._data.get(key)
        if entry is None:
            return False
        if entry.get("input_sha") != input_sha:
            return False
        return output.exists()

    def record(self, key: str, input_sha: str, output: Path, rows: int, **extra: Any) -> None:
        self._data[key] = {
            "input_sha": input_sha,
            "output": str(output),
            "rows": rows,
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def keys(self) -> list[str]:
        return sorted(self._data)
