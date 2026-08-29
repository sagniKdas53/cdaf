"""Optional video metadata via ffprobe. Everything here degrades gracefully:
if ffprobe is absent or fails, we simply omit the optional header keys."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def probe(video: str | Path) -> dict[str, str]:
    """Return optional CDAF header keys (duration, resolution, fps) or {}."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        raw = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(video),
            ],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        data = json.loads(raw)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return {}

    out: dict[str, str] = {}
    try:
        seconds = float(data["format"]["duration"])
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        out["duration"] = f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    except (KeyError, ValueError, TypeError):
        pass

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video_stream:
        w, h = video_stream.get("width"), video_stream.get("height")
        if w and h:
            out["resolution"] = f"{w}x{h}"
        rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        if rate and "/" in rate:
            num, den = rate.split("/")
            try:
                fps = float(num) / float(den)
                out["fps"] = f"{fps:.3f}".rstrip("0").rstrip(".")
            except (ValueError, ZeroDivisionError):
                pass
    return out


def probe_duration_seconds(video: str | Path) -> float | None:
    """Return video duration in seconds as float, or None if unavailable."""
    if not shutil.which("ffprobe"):
        return None
    try:
        raw = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", str(video),
            ],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        data = json.loads(raw)
        return float(data["format"]["duration"])
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError, TypeError, OSError):
        return None
