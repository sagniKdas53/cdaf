"""Generate a CDAF sidecar body from a video.

Supports:
- Gemini via Google GenAI SDK (requires `google-genai` and GEMINI_API_KEY)
- OpenRouter via Chat Completions API (requires OPENROUTER_API_KEY)
- Local OpenAI-compatible endpoints with ffmpeg scene detection (see local.py)
- Chunking & parallelism for long/large videos
- Model selection via CLI (with presets and aliases)
- Cost calculation and tracking per video
- Intelligent classification & domain guidance for screencasts, demos, meetings, and presentations
"""

from __future__ import annotations

import base64
import concurrent.futures
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .probe import probe, probe_duration_seconds
from .sidecar import SPEC_VERSION, Sidecar, hash_file

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_MODEL = os.environ.get("CDAF_MODEL", "gemini-2.5-flash")
DEFAULT_PROVIDER = os.environ.get("CDAF_PROVIDER", "auto")
PROVIDERS = ("auto", "gemini", "openrouter", "local")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MAX_INLINE_UPLOAD_MB = 45.0

MODEL_ALIASES: dict[str, str] = {
    # Gemini models
    "flash": "gemini-2.5-flash",
    "flash-2.5": "gemini-2.5-flash",
    "flash-3.7": "gemini-3.7-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-3.7": "gemini-3.7-flash",
    "pro": "gemini-2.5-pro",
    "gemini-pro": "gemini-2.5-pro",
    # OpenRouter models
    "or-flash": "google/gemini-2.5-flash",
    "or-flash-3.7": "google/gemini-3.7-flash",
    "or-pro": "google/gemini-2.5-pro",
    "qwen": "qwen/qwen2.5-vl-72b-instruct",
    "qwen-vl": "qwen/qwen2.5-vl-72b-instruct",
    "qwen3": "qwen/qwen3.8-flash",
    "pixtral": "mistralai/pixtral-large-2411",
    "pixtral-12b": "mistralai/pixtral-12b",
    "llama": "meta-llama/llama-3.2-90b-vision-instruct",
    "llama-11b": "meta-llama/llama-3.2-11b-vision-instruct",
    "gpt4o": "openai/gpt-4o",
    "gpt-4o": "openai/gpt-4o",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}

# Approximate pricing in USD per 1M tokens: (prompt_price_per_1M, output_price_per_1M).
# Provider prices change; the `cost:` header is an estimate, not a billing record.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.7-flash": (0.75, 3.75),
    "google/gemini-3.7-flash": (0.375, 1.875),
    "gemini-2.0-flash": (0.10, 0.40),
    "google/gemini-2.0-flash": (0.10, 0.40),
    "qwen/qwen3.8-flash": (0.15, 0.47),
    "qwen/qwen3.5-flash-02-23": (0.065, 0.26),
    "gemini-2.5-pro": (1.25, 10.00),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "qwen/qwen2.5-vl-72b-instruct": (0.40, 0.40),
    "mistralai/pixtral-large-2411": (2.00, 6.00),
    "mistralai/pixtral-12b": (0.15, 0.15),
    "meta-llama/llama-3.2-90b-vision-instruct": (0.90, 0.90),
    "meta-llama/llama-3.2-11b-vision-instruct": (0.18, 0.18),
}

_DETAIL_GUIDANCE = {
    "brief": (
        "Keep segments coarse (5-15 seconds each) and descriptions to one short "
        "sentence. Skip the Transcript and On-screen Text sections unless speech or "
        "text is central to the clip."
    ),
    "standard": (
        "Use natural shot boundaries for segments (typically 2-8 seconds each). "
        "Describe each segment in 1-2 sentences."
    ),
    "rich": (
        "Segment at every cut or distinct beat. Describe each segment in 2-4 "
        "sentences covering subjects, actions, setting, camera framing and movement, "
        "lighting, color, and mood. Note anything an editor would cut on."
    ),
}

_MODE_GUIDANCE = {
    "auto": (
        "\nIntelligently detect the video domain and adapt your output accordingly:\n"
        "- If a software screencast / demo: Identify active applications (e.g. browser, VS Code, terminal), "
        "UI buttons clicked, URLs, and code/commands in Segments.\n"
        "- If an online meeting / conference call: Identify participants, active speaker, screen-sharing windows, "
        "and meeting discussion points.\n"
        "- If a slide presentation: Note slide titles, bullet points, and speaker transitions."
    ),
    "screencast": (
        "\nThis video is a software screencast / desktop demonstration. You MUST:\n"
        "1. Include an '## Environment & Tools' section directly after '## Summary' listing visible OS, "
        "applications (e.g. VS Code, Firefox, Terminal, Figma), web services/URLs, and developer tools.\n"
        "2. In '## Segments', prefix each segment with the active application/window: "
        "'[MM:SS.d-MM:SS.d] (App/Tool Name) Description of UI interactions, code edits, or commands.'\n"
        "3. In '## On-screen Text', capture exact button labels, terminal commands, URLs, and code snippets."
    ),
    "meeting": (
        "\nThis video is an online meeting / conference call. You MUST:\n"
        "1. Include a '## Meeting Details' section after '## Summary' noting platform (Zoom, Google Meet, "
        "Teams, Discord), visible attendees, active speaker layout, and screen-sharing status.\n"
        "2. In '## Segments', track speaker turns and screen-sharing transitions: "
        "'[MM:SS.d-MM:SS.d] (Speaker / Screen Share) Discussion on topic...'\n"
        "3. In '## Transcript', attribute spoken dialogue to specific participant names or roles.\n"
        "4. Include an '## Action Items & Decisions' section before '## Tags' summarizing agreed next steps."
    ),
    "demo": (
        "\nThis video is a product demonstration. You MUST:\n"
        "1. Include an '## Environment & Tools' section after '## Summary' listing the product, platform, and tools.\n"
        "2. In '## Segments', highlight feature walkthroughs and user flows: "
        "'[MM:SS.d-MM:SS.d] (Feature Name) Detailed interaction flow...'\n"
        "3. In '## On-screen Text', record key value propositions, UI labels, and data displayed."
    ),
    "presentation": (
        "\nThis video is a slide presentation / lecture. You MUST:\n"
        "1. Include a '## Presentation Overview' section with slide deck title and presenter name.\n"
        "2. In '## Segments', reference slide numbers/titles: "
        "'[MM:SS.d-MM:SS.d] [Slide N: Title] Presenter discussion...'\n"
        "3. In '## On-screen Text', record slide headings and key bullet points."
    ),
    "general": "",
}

_PROMPT_TEMPLATE = """You are producing the body of a CDAF (Cached Descriptive Asset File): a \
timestamped description of a video that AI agents will read INSTEAD of watching the \
video. Your output must let an agent make editing and analysis decisions without ever \
seeing the footage. Describe only what is objectively visible or audible; never \
speculate or embellish.

Output GitHub-flavored markdown with sections in logical order (no preamble, no code fences):

## Summary
One short paragraph: what this clip is, its overall arc, and what it would be useful for.

## Segments
Chronological, contiguous coverage from 00:00.0 to the end of the video. One line per \
segment, formatted exactly as:
[MM:SS.d-MM:SS.d] Description.
Use HH:MM:SS.d timestamps only if the video exceeds one hour. {detail_guidance}

## Transcript
Spoken words with timestamps: [MM:SS.d] Speaker: words. Label speakers (Man, Woman, \
Narrator, Speaker 1, or participant names) consistently. If there is no speech, output exactly: (no speech)

## On-screen Text
Visible text (titles, captions, signs, UI, commands, code) with timestamps: [MM:SS.d] "text". If there \
is none, output exactly: (none)

## Tags
One comma-separated line of retrieval keywords: subjects, actions, software tools, setting, mood, \
camera work, lighting, genre.{mode_guidance}
"""


class GenerationError(RuntimeError):
    pass


def load_dotenv_if_present() -> None:
    """Load `.env` from the current working directory only. Existing env vars win.

    Called by the CLI before generation; never at import time, and never from
    the user's home directory or the installed package tree.
    """
    candidate = Path.cwd() / ".env"
    if not candidate.is_file():
        return
    try:
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            if k and k not in os.environ:
                os.environ[k] = v
    except OSError:
        pass


def calculate_cost(model: str, prompt_tokens: int | None, output_tokens: int | None) -> float | None:
    """Estimate total USD cost for a model run given prompt and output token counts."""
    if prompt_tokens is None and output_tokens is None:
        return None
    pt = prompt_tokens or 0
    ot = output_tokens or 0

    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        for k, v in MODEL_PRICING.items():
            if k in model:
                pricing = v
                break
    if pricing is None:
        return None  # unknown model: omit the cost header rather than invent a price

    in_rate, out_rate = pricing
    return (pt * in_rate + ot * out_rate) / 1_000_000.0


def format_cost(cost: float | None) -> str | None:
    """Format cost float into a dollar string like '$0.0018'."""
    if cost is None:
        return None
    if cost < 0.0001:
        return f"${cost:.6f}"
    return f"${cost:.4f}"


def resolve_provider_and_model(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str]:
    """Resolve the effective provider ('gemini' | 'openrouter' | 'local') and model ID."""
    p = (provider or os.environ.get("CDAF_PROVIDER", "auto")).lower().strip()
    raw_m = model or os.environ.get("CDAF_MODEL")

    if raw_m:
        raw_m = raw_m.strip()
        m = MODEL_ALIASES.get(raw_m.lower(), raw_m)
    else:
        m = None

    if p not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}': provider must be one of {PROVIDERS}")

    if p == "auto":
        if m and ("/" in m or m.startswith("openrouter:")):
            p = "openrouter"
        elif api_key and api_key.startswith("sk-or-"):
            p = "openrouter"
        elif os.environ.get("OPENROUTER_API_KEY") and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            p = "openrouter"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            p = "gemini"
        elif os.environ.get("OPENROUTER_API_KEY"):
            p = "openrouter"
        else:
            p = "gemini"

    if not m:
        if p == "local":
            from .local import DEFAULT_LOCAL_MODEL
            m = DEFAULT_LOCAL_MODEL
        else:
            m = DEFAULT_OPENROUTER_MODEL if p == "openrouter" else DEFAULT_GEMINI_MODEL

    return p, m


def parse_timestamp(ts: str) -> float:
    """Parse MM:SS.d or HH:MM:SS.d into seconds as float."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def format_timestamp(seconds: float, use_hours: bool = False) -> str:
    """Format seconds into MM:SS.d or HH:MM:SS.d."""
    # Round to tenths first so 59.96 rolls over to 01:00.0 instead of 00:60.0.
    total_tenths = round(max(0.0, seconds) * 10)
    h, rem = divmod(total_tenths, 36000)
    m, tenths = divmod(rem, 600)
    s = tenths / 10.0
    if use_hours or h > 0:
        return f"{h:02d}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


def offset_body_timestamps(body: str, offset: float, use_hours: bool = False) -> str:
    """Offset all segment, transcript, and on-screen text timestamps by offset seconds."""
    if offset == 0.0 and not use_hours:
        return body

    seg_pattern = re.compile(r"\[(?P<start>[\d:.]+)\s*[-–]\s*(?P<end>[\d:.]+)\]")
    single_ts_pattern = re.compile(r"\[(?P<ts>[\d:.]+)\]")

    def offset_range(m: re.Match) -> str:
        s_sec = parse_timestamp(m.group("start")) + offset
        e_sec = parse_timestamp(m.group("end")) + offset
        return f"[{format_timestamp(s_sec, use_hours)}-{format_timestamp(e_sec, use_hours)}]"

    def offset_single(m: re.Match) -> str:
        ts_sec = parse_timestamp(m.group("ts")) + offset
        return f"[{format_timestamp(ts_sec, use_hours)}]"

    lines = body.splitlines()
    out = []
    in_segments = False
    in_transcript = False
    in_text = False
    for line in lines:
        if line.startswith("## "):
            in_segments = line.strip() == "## Segments"
            in_transcript = line.strip() == "## Transcript"
            in_text = line.strip() == "## On-screen Text"
            out.append(line)
            continue
        if in_segments:
            line = seg_pattern.sub(offset_range, line)
        elif in_transcript or in_text:
            line = single_ts_pattern.sub(offset_single, line)
        out.append(line)
    return "\n".join(out)


def parse_sections(body: str) -> dict[str, list[str]]:
    """Parse body markdown into dictionary of section lines keyed by heading."""
    sections: dict[str, list[str]] = {}
    cur_sec = None
    for line in body.splitlines():
        if line.startswith("## "):
            cur_sec = line[3:].strip()
            sections[cur_sec] = []
        elif cur_sec is not None:
            sections[cur_sec].append(line)
    return sections


def merge_chunk_bodies(bodies_with_spans: list[tuple[str, float, float]]) -> str:
    """Merge multiple chunk bodies into a single coherent CDAF body, preserving all sections."""
    if not bodies_with_spans:
        return ""
    if len(bodies_with_spans) == 1:
        return bodies_with_spans[0][0]

    all_sections: dict[str, list[str]] = {}
    ordered_section_names: list[str] = []

    for body, _, _ in bodies_with_spans:
        sec_map = parse_sections(body)
        for sec_name, lines in sec_map.items():
            if sec_name not in all_sections:
                all_sections[sec_name] = []
                ordered_section_names.append(sec_name)
            all_sections[sec_name].extend(lines)

    # Standard sections ordering if present
    standard_order = [
        "Summary",
        "Environment & Tools",
        "Meeting Details",
        "Presentation Overview",
        "Segments",
        "Transcript",
        "On-screen Text",
        "Action Items & Decisions",
        "Tags",
    ]
    seen = set()
    final_order = []
    for s in standard_order:
        if s in ordered_section_names:
            final_order.append(s)
            seen.add(s)
    for s in ordered_section_names:
        if s not in seen:
            final_order.append(s)
            seen.add(s)

    out = []
    for sec in final_order:
        out.append(f"## {sec}")
        lines = all_sections[sec]
        if sec == "Summary":
            sum_lines = [l.strip() for l in lines if l.strip()]
            out.append("\n\n".join(sum_lines) if sum_lines else "Video overview across chronological segments.")
        elif sec == "Segments":
            seg_lines = [l.strip() for l in lines if l.strip().startswith("[")]
            out.extend(seg_lines if seg_lines else ["[00:00.0-00:00.0] Complete video duration."])
        elif sec == "Transcript":
            speech_lines = [l.strip() for l in lines if l.strip() and l.strip() != "(no speech)"]
            out.extend(speech_lines if speech_lines else ["(no speech)"])
        elif sec == "On-screen Text":
            text_lines = [l.strip() for l in lines if l.strip() and l.strip() != "(none)"]
            out.extend(text_lines if text_lines else ["(none)"])
        elif sec == "Tags":
            all_tags = []
            for line in lines:
                for t in line.split(","):
                    t_clean = t.strip()
                    if t_clean and t_clean.lower() not in [x.lower() for x in all_tags]:
                        all_tags.append(t_clean)
            out.append(", ".join(all_tags) if all_tags else "video")
        else:
            clean_lines = [l.strip() for l in lines if l.strip()]
            dedup = []
            for l in clean_lines:
                if l not in dedup:
                    dedup.append(l)
            out.extend(dedup if dedup else ["(none)"])
        out.append("")

    return "\n".join(out)


def _video_data_url(video_path: Path) -> str:
    """Encode video bytes to a base64 data URL with appropriate MIME type."""
    ext = video_path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }
    mime = mime_map.get(ext, "video/mp4")
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _gemini_client(api_key: str | None):
    try:
        from google import genai  # noqa: PLC0415
    except ImportError as e:
        raise GenerationError(
            "google-genai is not installed. Run: pip install cdaf[generate]"
        ) from e
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GenerationError(
            "No Gemini API key found. Set GEMINI_API_KEY (get one free at https://aistudio.google.com/apikey) "
            "or use OpenRouter by setting OPENROUTER_API_KEY."
        )
    return genai.Client(api_key=key)


def _describe_gemini(
    video: Path,
    *,
    model: str,
    prompt: str,
    api_key: str | None = None,
    usage_out: dict | None = None,
) -> str:
    client = _gemini_client(api_key)
    uploaded = client.files.upload(file=str(video))
    try:
        deadline = time.monotonic() + 600
        while uploaded.state and uploaded.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                raise GenerationError("timed out waiting for Gemini to process the upload")
            time.sleep(3)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state and uploaded.state.name == "FAILED":
            raise GenerationError(f"Gemini could not process {video.name} (upload FAILED)")

        started = time.monotonic()
        response = client.models.generate_content(model=model, contents=[uploaded, prompt])

        body = (response.text or "").strip()
        if usage_out is not None:
            usage = getattr(response, "usage_metadata", None)
            usage_out.update({
                "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                "seconds": round(time.monotonic() - started, 2),
            })
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
    return body


def _optimize_video_for_upload(
    video_path: str | Path,
    max_size_mb: float = 6.0,
    fps: int = 5,
    crf: int = 34,
) -> tuple[Path, bool]:
    """If video size exceeds max_size_mb and ffmpeg is installed, create a lightweight temp MP4."""
    video_path = Path(video_path)
    size_mb = video_path.stat().st_size / (1024 * 1024)

    if size_mb <= max_size_mb or not shutil.which("ffmpeg"):
        return video_path, False

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-vf", f"scale=640:-2,fps={fps}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "32k",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            return Path(tmp_path), True
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return video_path, False


def _extract_video_frames(
    video_path: Path,
    max_frames: int = 24,
) -> list[str]:
    """Extract evenly spaced jpeg frames as base64 data URLs from video."""
    dur = probe_duration_seconds(video_path) or 10.0
    fps_rate = max(0.05, min(1.0, max_frames / dur))

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-vf", f"scale=480:-2,fps={fps_rate:.3f}",
            "-q:v", "4",
            f"{tmpdir}/f_%04d.jpg",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=120)
            frames = sorted(Path(tmpdir).glob("*.jpg"))
            if frames:
                if len(frames) > max_frames:
                    step = len(frames) / max_frames
                    frames = [frames[int(i * step)] for i in range(max_frames)]

                data_urls = []
                for f in frames:
                    b64 = base64.b64encode(f.read_bytes()).decode("ascii")
                    data_urls.append(f"data:image/jpeg;base64,{b64}")
                return data_urls
        except Exception:
            pass

    return []


def _describe_openrouter(
    video: Path,
    *,
    model: str,
    prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    usage_out: dict | None = None,
    timeout: float = 600.0,
) -> str:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise GenerationError(
            "No OpenRouter API key found. Set OPENROUTER_API_KEY or pass --api-key."
        )

    base = (base_url or os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
    url = f"{base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/UditAkhourii/cdaf",
        "X-Title": "CDAF",
    }

    use_frames = not (model.startswith("google/gemini-") or model.startswith("gemini-"))

    def execute_request(payload_content: list[dict]) -> tuple[str, dict]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": payload_content}],
        }
        started = time.monotonic()
        try:
            import requests
        except ImportError:
            requests = None

        if requests is not None:
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except requests.RequestException as e:
                raise GenerationError(f"OpenRouter request failed: {e}") from e
            try:
                data = resp.json()
            except Exception:
                data = {}
            if resp.status_code != 200:
                err_msg = data.get("error", {}).get("message") if isinstance(data, dict) else None
                raise GenerationError(
                    f"OpenRouter API error (HTTP {resp.status_code}): {err_msg or resp.text}"
                )
        else:
            import json
            import urllib.error
            import urllib.request

            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read().decode("utf-8")
                    err_json = json.loads(err_body)
                    err_msg = err_json.get("error", {}).get("message", err_body)
                except Exception:
                    err_msg = str(e)
                raise GenerationError(f"OpenRouter API error (HTTP {e.code}): {err_msg}") from e
            except urllib.error.URLError as e:
                raise GenerationError(f"OpenRouter request failed: {e}") from e

        if "error" in data:
            err_msg = data.get("error", {}).get("message") if isinstance(data["error"], dict) else str(data["error"])
            raise GenerationError(f"OpenRouter API error: {err_msg}")

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise GenerationError(f"Unexpected OpenRouter response format: {data}")

        message = choices[0].get("message", {})
        body = (message.get("content") or "").strip()
        if not body and "text" in choices[0]:
            body = choices[0]["text"].strip()

        usage_dict = {}
        usage = data.get("usage", {})
        usage_dict["prompt_tokens"] = usage.get("prompt_tokens")
        usage_dict["output_tokens"] = usage.get("completion_tokens")
        usage_dict["seconds"] = round(time.monotonic() - started, 2)
        return body, usage_dict

    if use_frames:
        frame_urls = _extract_video_frames(video)
        if not frame_urls:
            raise GenerationError(
                f"could not extract video frames for {video.name} "
                "(image-vision models need ffmpeg installed for frame sampling)"
            )
        content: list[dict] = [{"type": "text", "text": prompt + "\nNote: Attached are chronological image frames sampled across the video."}]
        for furl in frame_urls:
            content.append({"type": "image_url", "image_url": {"url": furl}})
        body, u = execute_request(content)
        if usage_out is not None:
            usage_out.update(u)
        return body

    upload_video, is_temp = _optimize_video_for_upload(video)
    try:
        upload_mb = upload_video.stat().st_size / (1024 * 1024)
        if upload_mb > MAX_INLINE_UPLOAD_MB:
            raise GenerationError(
                f"{video.name} is {upload_mb:.0f} MB after optimization, above the "
                f"{MAX_INLINE_UPLOAD_MB} MB inline-upload limit for OpenRouter. "
                "Install ffmpeg so the video can be compressed, or use --provider gemini."
            )
        data_url = _video_data_url(upload_video)
        content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": data_url}},
        ]
        try:
            body, u = execute_request(content)
            if usage_out is not None:
                usage_out.update(u)
            return body
        except GenerationError as ge:
            if "not support" in str(ge).lower() or "HTTP 404" in str(ge) or "HTTP 400" in str(ge):
                frame_urls = _extract_video_frames(upload_video)
                if frame_urls:
                    content_frames: list[dict] = [{"type": "text", "text": prompt + "\nNote: Attached are chronological image frames sampled across the video."}]
                    for furl in frame_urls:
                        content_frames.append({"type": "image_url", "image_url": {"url": furl}})
                    body, u = execute_request(content_frames)
                    if usage_out is not None:
                        usage_out.update(u)
                    return body
            raise
    finally:
        if is_temp and upload_video.is_file():
            try:
                upload_video.unlink()
            except OSError:
                pass


def _describe_single_clip(
    video: Path,
    *,
    provider: str,
    model: str,
    prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    usage_out: dict | None = None,
) -> str:
    """Internal helper to execute one describe call on a video file."""
    if provider == "gemini":
        body = _describe_gemini(
            video, model=model, prompt=prompt, api_key=api_key, usage_out=usage_out
        )
    elif provider == "openrouter":
        body = _describe_openrouter(
            video,
            model=model,
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            usage_out=usage_out,
        )
    else:
        raise ValueError(f"unknown provider: {provider!r}")

    body = re.sub(r"<think>[\s\S]*?</think>", "", body).strip()
    body = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", body).strip()
    if "## Segments" not in body:
        raise GenerationError(
            f"model output for {video.name} lacked a '## Segments' section; not saving"
        )
    return body


def describe_video(
    video: str | Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    detail: str = "standard",
    mode: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
    usage_out: dict | None = None,
    chunk_duration: float | None = None,
    parallel: int = 1,
) -> str:
    """Describe a video using Gemini or OpenRouter and return the CDAF body markdown."""
    if detail not in _DETAIL_GUIDANCE:
        raise ValueError(f"detail must be one of {sorted(_DETAIL_GUIDANCE)}")

    prov, resolved_model = resolve_provider_and_model(provider, model, api_key)
    mode_text = _MODE_GUIDANCE.get(mode, _MODE_GUIDANCE["auto"])
    prompt = _PROMPT_TEMPLATE.format(
        detail_guidance=_DETAIL_GUIDANCE[detail],
        mode_guidance=mode_text,
    )
    video = Path(video)

    duration_sec = probe_duration_seconds(video)

    # Check if chunking is requested and applicable
    if (
        chunk_duration
        and chunk_duration > 0
        and duration_sec
        and duration_sec > chunk_duration
        and shutil.which("ffmpeg")
    ):
        spans: list[tuple[float, float]] = []
        cur = 0.0
        while cur < duration_sec:
            nxt = min(cur + chunk_duration, duration_sec)
            spans.append((cur, nxt))
            cur = nxt
        # Fold a sliver of a final chunk (probe rounding noise) into the previous one,
        # so ffmpeg never gets asked for a near-zero-length clip.
        if len(spans) > 1 and (spans[-1][1] - spans[-1][0]) < 1.0:
            last = spans.pop()
            spans[-1] = (spans[-1][0], last[1])

        workers = max(1, min(parallel or 4, len(spans)))
        use_hours = duration_sec > 3600

        def process_chunk(idx: int, span: tuple[float, float]) -> tuple[int, str, float, float, dict]:
            start_s, end_s = span
            fd, tmp_chunk = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            # Gemini reads full-quality video natively, so keep the original
            # resolution and frame rate there; only the frame-sampled OpenRouter
            # path gets the aggressive 640p/5fps shrink.
            if prov == "gemini":
                quality_args = ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                                "-c:a", "aac", "-b:a", "128k"]
            else:
                quality_args = ["-vf", "scale=640:-2,fps=5",
                                "-c:v", "libx264", "-crf", "34", "-preset", "ultrafast",
                                "-c:a", "aac", "-b:a", "32k"]
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", str(start_s),
                "-i", str(video),
                "-t", str(end_s - start_s),
                *quality_args,
                tmp_chunk,
            ]
            chunk_usage: dict = {}
            try:
                subprocess.run(cmd, check=True, timeout=600)
                chunk_body = _describe_single_clip(
                    Path(tmp_chunk),
                    provider=prov,
                    model=resolved_model,
                    prompt=prompt,
                    api_key=api_key,
                    base_url=base_url,
                    usage_out=chunk_usage,
                )
                chunk_body = offset_body_timestamps(chunk_body, start_s, use_hours=use_hours)
                return idx, chunk_body, start_s, end_s, chunk_usage
            finally:
                if os.path.exists(tmp_chunk):
                    try:
                        os.remove(tmp_chunk)
                    except OSError:
                        pass

        results: list[tuple[int, str, float, float, dict]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_chunk, i, span) for i, span in enumerate(spans)]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        # Sort results by chunk index
        results.sort(key=lambda x: x[0])
        bodies_with_spans = [(r[1], r[2], r[3]) for r in results]
        body = merge_chunk_bodies(bodies_with_spans)

        if usage_out is not None:
            total_prompt = sum(r[4].get("prompt_tokens") or 0 for r in results)
            total_output = sum(r[4].get("output_tokens") or 0 for r in results)
            max_seconds = max((r[4].get("seconds") or 0.0) for r in results) if results else 0.0
            usage_out.update({
                "prompt_tokens": total_prompt or None,
                "output_tokens": total_output or None,
                "seconds": round(max_seconds, 2),
            })
        return body

    # Single-clip execution
    return _describe_single_clip(
        video,
        provider=prov,
        model=resolved_model,
        prompt=prompt,
        api_key=api_key,
        base_url=base_url,
        usage_out=usage_out,
    )


def generate_sidecar(
    video: str | Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    detail: str = "standard",
    mode: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
    scene_threshold: float | None = None,
    usage_out: dict | None = None,
    chunk_duration: float | None = None,
    parallel: int = 1,
) -> Sidecar:
    """Full pipeline: hash + probe + describe → a ready-to-save Sidecar with cost tracking."""
    video = Path(video)
    prov, resolved_model = resolve_provider_and_model(provider, model, api_key)

    header = {
        "video": video.name,
        "sha256": hash_file(video),
        "bytes": str(video.stat().st_size),
        **probe(video),
        "generator": resolved_model,
    }

    if prov == "local":
        from . import local

        threshold = local.DEFAULT_SCENE_THRESHOLD if scene_threshold is None else scene_threshold
        body = local.describe_video_local(
            video,
            model=resolved_model,
            detail=detail,
            base_url=base_url or local.DEFAULT_BASE_URL,
            scene_threshold=threshold,
            usage_out=usage_out,
        )
        header.update(
            local.local_header_extras(
                continuity=True,
                transcribed="(no speech)" not in body,
                threshold=threshold,
            )
        )
    else:
        internal_usage: dict = {}
        body = describe_video(
            video,
            provider=prov,
            model=resolved_model,
            detail=detail,
            mode=mode,
            api_key=api_key,
            base_url=base_url,
            usage_out=internal_usage,
            chunk_duration=chunk_duration,
            parallel=parallel,
        )

        if usage_out is not None:
            usage_out.update(internal_usage)

        cost_val = calculate_cost(
            resolved_model,
            internal_usage.get("prompt_tokens"),
            internal_usage.get("output_tokens"),
        )
        cost_str = format_cost(cost_val)

        if mode and mode != "auto":
            header["mode"] = mode

        if cost_str:
            header["cost"] = cost_str
        if internal_usage.get("prompt_tokens") is not None:
            header["prompt_tokens"] = str(internal_usage["prompt_tokens"])
        if internal_usage.get("output_tokens") is not None:
            header["output_tokens"] = str(internal_usage["output_tokens"])

    header["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header["detail"] = detail
    header["lang"] = "en"

    return Sidecar(version=SPEC_VERSION, header=header, body=body)
