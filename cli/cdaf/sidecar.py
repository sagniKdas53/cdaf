"""Parse, serialize, and validate CDAF sidecar files (spec v1.0).

Zero-dependency by design: agents and CI can validate sidecars without
installing the Gemini SDK.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SPEC_VERSION = "1.0"
SIDECAR_EXTENSION = ".cdaf"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

REQUIRED_KEYS = ("video", "sha256", "bytes", "generator", "created")

_HEADER_OPEN = re.compile(r"^---\s*CDAF/(\d+)\.(\d+)\s*$")
_HEADER_KV = re.compile(r"^([a-z0-9][a-z0-9_-]*)\s*:\s*(.*)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT = re.compile(
    r"^\[(\d{1,2}:)?\d{1,2}:\d{2}(\.\d+)?\s*[-–]\s*(\d{1,2}:)?\d{1,2}:\d{2}(\.\d+)?\]\s+\S"
)


class SidecarError(ValueError):
    """Raised when a sidecar file is malformed."""


@dataclass
class Sidecar:
    """An in-memory CDAF sidecar: spec version, header keys, markdown body."""

    version: str = SPEC_VERSION
    header: dict[str, str] = field(default_factory=dict)
    body: str = ""

    @property
    def video(self) -> str:
        return self.header.get("video", "")

    @property
    def sha256(self) -> str:
        return self.header.get("sha256", "")

    @property
    def bytes(self) -> int | None:
        raw = self.header.get("bytes")
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None


def parse(text: str) -> Sidecar:
    """Parse sidecar text into a Sidecar. Raises SidecarError on malformed input."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        raise SidecarError("empty file")

    m = _HEADER_OPEN.match(lines[0])
    if not m:
        raise SidecarError(
            f"first line must be '--- CDAF/<major>.<minor>', got: {lines[0]!r}"
        )
    major, minor = m.group(1), m.group(2)
    if major != SPEC_VERSION.split(".")[0]:
        raise SidecarError(f"unsupported CDAF major version: {major}.{minor}")

    header: dict[str, str] = {}
    body_start = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = i + 1
            break
        if not line.strip():
            continue
        kv = _HEADER_KV.match(line)
        if not kv:
            raise SidecarError(f"malformed header line {i + 1}: {line!r}")
        header[kv.group(1)] = kv.group(2).strip()

    if body_start is None:
        raise SidecarError("header block never closed with '---'")

    missing = [k for k in REQUIRED_KEYS if not header.get(k)]
    if missing:
        raise SidecarError(f"missing required header keys: {', '.join(missing)}")
    if not _SHA256.match(header["sha256"]):
        raise SidecarError("sha256 must be 64 lowercase hex characters")
    if not header["bytes"].isdigit():
        raise SidecarError("bytes must be a non-negative integer")

    body = "\n".join(lines[body_start:]).strip("\n")
    if "## Segments" not in body:
        raise SidecarError("body is missing the required '## Segments' section")

    return Sidecar(version=f"{major}.{minor}", header=header, body=body)


def dumps(sc: Sidecar) -> str:
    """Serialize a Sidecar to canonical text."""
    ordered = ["video", "sha256", "bytes", "duration", "resolution", "fps",
               "generator", "mode", "cost", "prompt_tokens", "output_tokens",
               "created", "detail", "lang"]
    keys = [k for k in ordered if k in sc.header]
    keys += [k for k in sc.header if k not in ordered]
    lines = [f"--- CDAF/{sc.version}"]
    lines += [f"{k}: {sc.header[k]}" for k in keys]
    lines.append("---")
    return "\n".join(lines) + "\n\n" + sc.body.strip("\n") + "\n"


def load(path: str | Path) -> Sidecar:
    return parse(Path(path).read_text(encoding="utf-8"))


def save(sc: Sidecar, path: str | Path) -> None:
    Path(path).write_text(dumps(sc), encoding="utf-8", newline="\n")


def sidecar_path_for(video: str | Path) -> Path:
    return Path(video).with_suffix(SIDECAR_EXTENSION)


def video_path_for(sidecar: str | Path, video_name: str | None = None) -> Path | None:
    """Locate the paired video: by header name first, then same-basename search."""
    sidecar = Path(sidecar)
    if video_name:
        candidate = sidecar.parent / video_name
        if candidate.is_file():
            return candidate
    for ext in sorted(VIDEO_EXTENSIONS):
        candidate = sidecar.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Chunked SHA-256 of a file's bytes, lowercase hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def check_freshness(video: str | Path, sc: Sidecar, *, fast: bool = False) -> str:
    """Return 'fresh' or 'stale' for a (video, sidecar) pair.

    fast=True only compares file size — it can prove staleness but only
    presume freshness; use full hashing when correctness matters.
    """
    video = Path(video)
    size = video.stat().st_size
    if sc.bytes is not None and size != sc.bytes:
        return "stale"
    if fast:
        return "fresh"
    return "fresh" if hash_file(video) == sc.sha256 else "stale"


def segment_lines(sc: Sidecar) -> list[str]:
    """Extract well-formed segment lines from the body's Segments section."""
    in_segments = False
    out: list[str] = []
    for line in sc.body.split("\n"):
        if line.startswith("## "):
            in_segments = line.strip() == "## Segments"
            continue
        if in_segments and _SEGMENT.match(line.strip()):
            out.append(line.strip())
    return out
