"""Build and query a searchable index over a library's .cdaf sidecars.

`cdaf index ./footage` walks a tree, reads every sidecar it finds, and writes a
single `cdaf-index.json`. `cdaf search "golden hour" ./footage` then answers
from that file, so an agent looking for a shot never re-reads the footage --
which is the whole point of keeping sidecars in the first place.

The index is one JSON file and the search is a scan over it -- stdlib only, no
database, nothing to keep in sync. A library big enough to need an inverted
index is a good reason to add one later; it is not a good reason to ship two
query paths now.

Index format:

    {
      "version": 1,
      "generated": "2026-08-28T00:00:00Z",
      "root": "footage",
      "entries": [
        {
          "video": "sunset-drone.mp4",   # name recorded in the sidecar header
          "video_path": "b-roll/sunset-drone.mp4",   # relative to root, or null
          "sidecar": "b-roll/sunset-drone.cdaf",     # relative to root
          "state": "fresh" | "stale" | "orphan" | "invalid",
          "duration": "31s",
          "summary": "...",
          "tags": ["drone", "golden hour"],
          "segments": ["[00:00.0-00:05.0] ..."],
          "transcript": "...",
          "on_screen_text": "...",
          "header": {...},
          "body": "full markdown"
        }
      ]
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .sidecar import SidecarError, check_freshness, load, segment_lines, video_path_for

INDEX_VERSION = 1
INDEX_FILENAME = "cdaf-index.json"

_SEARCHABLE_FIELDS = ("video", "summary", "tags", "segments", "transcript", "on_screen_text")


def _sections(body: str) -> dict[str, list[str]]:
    """Split a sidecar body into `## Heading` -> lines."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out.setdefault(current, [])
        elif current is not None:
            out[current].append(line)
    return out


def _joined(sections: dict[str, list[str]], name: str) -> str:
    return "\n".join(line for line in sections.get(name, []) if line.strip()).strip()


def _tags(sections: dict[str, list[str]]) -> list[str]:
    """Comma-separated tags, de-duplicated case-insensitively, order preserved."""
    tags: list[str] = []
    seen: set[str] = set()
    for line in sections.get("Tags", []):
        for raw in line.split(","):
            tag = raw.strip()
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                tags.append(tag)
    return tags


def _iter_sidecars(root: Path) -> list[Path]:
    """Every .cdaf file under `root`, in a stable order."""
    if root.is_file():
        return [root] if root.suffix.lower() == ".cdaf" else []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(".cdaf"):
                found.append(Path(dirpath) / name)
    return found


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _entry(sidecar: Path, root: Path, *, verify: bool) -> dict:
    """One index record for one sidecar. Never raises: bad sidecars are recorded."""
    rel_sidecar = _relative(sidecar, root)
    try:
        sc = load(sidecar)
    except (SidecarError, OSError) as exc:
        return {
            "video": sidecar.stem,
            "video_path": None,
            "sidecar": rel_sidecar,
            "state": "invalid",
            "error": str(exc),
            "duration": None,
            "summary": "",
            "tags": [],
            "segments": [],
            "transcript": "",
            "on_screen_text": "",
            "header": {},
            "body": "",
        }

    video = video_path_for(sidecar, sc.video)
    if video is None or not video.is_file():
        state = "orphan"
    else:
        try:
            state = check_freshness(video, sc, fast=not verify)
        except OSError:
            state = "orphan"

    sections = _sections(sc.body)
    return {
        "video": sc.video or sidecar.stem,
        "video_path": _relative(video, root) if video and video.is_file() else None,
        "sidecar": rel_sidecar,
        "state": state,
        "duration": sc.header.get("duration"),
        "summary": _joined(sections, "Summary"),
        "tags": _tags(sections),
        "segments": segment_lines(sc),
        "transcript": _joined(sections, "Transcript"),
        "on_screen_text": _joined(sections, "On-screen Text"),
        "header": dict(sc.header),
        "body": sc.body,
    }


def build_index(path: str | Path, *, verify: bool = False) -> dict:
    """Read every sidecar under `path` and return the index document.

    `verify=False` (the default) size-checks each video rather than hashing it,
    which keeps a large library fast; it can prove staleness but only presume
    freshness. Pass `verify=True` to hash.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"library path not found: {root}")
    base = root if root.is_dir() else root.parent
    entries = [_entry(sc, base, verify=verify) for sc in _iter_sidecars(root)]
    return {
        "version": INDEX_VERSION,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": base.as_posix(),
        "entries": entries,
    }


def write_index(index: dict, path: str | Path) -> Path:
    """Write the index JSON. Returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_index(path: str | Path) -> dict:
    """Read an index written by `write_index`."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("version") != INDEX_VERSION:
        raise ValueError(f"unrecognized index format in {path}")
    return doc


def _searchable_text(entry: dict) -> str:
    parts: list[str] = []
    for field in _SEARCHABLE_FIELDS:
        value = entry.get(field) or ""
        parts.append("\n".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def default_index_path(library: str | Path) -> Path:
    lib = Path(library)
    base = lib if lib.is_dir() else lib.parent
    return base / INDEX_FILENAME


def search(query: str, index: dict, limit: int = 20) -> list[dict]:
    """Rank index entries against `query`.

    Every whitespace-separated term must appear somewhere in an entry (AND), and
    entries are ordered by total term occurrences, then by video name for a
    stable result. Matching is case-insensitive substring matching, which keeps
    partial words like "sunse" useful.
    """
    terms = [t.lower() for t in query.split() if t.strip()]
    if not terms:
        return []
    hits: list[tuple[int, str, dict]] = []
    for entry in index.get("entries", []):
        haystack = _searchable_text(entry).lower()
        counts = [haystack.count(term) for term in terms]
        if all(counts):
            hits.append((-sum(counts), entry.get("video", ""), entry))
    hits.sort(key=lambda h: (h[0], h[1]))
    return [entry for _, _, entry in hits[:limit]]
