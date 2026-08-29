"""CDAF — Cached Descriptive Asset Files for video.

Core library: parse, serialize, and validate .cdaf sidecars (zero dependencies).
Generation via Gemini lives in cdaf.generate and requires the `google-genai` extra.
"""

__version__ = "1.0.0"

from .sidecar import (  # noqa: F401
    SPEC_VERSION,
    VIDEO_EXTENSIONS,
    Sidecar,
    SidecarError,
    check_freshness,
    dumps,
    hash_file,
    load,
    parse,
    save,
    segment_lines,
    sidecar_path_for,
    video_path_for,
)
