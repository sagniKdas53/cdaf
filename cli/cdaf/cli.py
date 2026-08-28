"""cdaf command-line interface.

Zero third-party dependencies for validate / read / status.
Generation needs google-genai, requests, or local endpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .sidecar import (
    SidecarError,
    check_freshness,
    load,
    save,
    sidecar_path_for,
    video_path_for,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _iter_videos(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() == ".cdaf":
                try:
                    sc = load(path)
                    v = video_path_for(path, sc.video)
                    if v:
                        out.append(v)
                    else:
                        print(f"warning: no video found for sidecar {path}", file=sys.stderr)
                except SidecarError as e:
                    print(f"warning: skipping invalid sidecar {path}: {e}", file=sys.stderr)
            elif path.suffix.lower() in VIDEO_EXTS:
                out.append(path)
            else:
                print(f"warning: skipping {path} (not a video file or directory)", file=sys.stderr)
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in VIDEO_EXTS:
                        out.append(fp)
        else:
            print(f"warning: skipping {path} (not a video file or directory)", file=sys.stderr)
    seen: set[Path] = set()
    uniq: list[Path] = []
    for v in sorted(out):
        resolved = v.resolve()
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(v)
    return uniq


def _sidecar_state(video: Path) -> str:
    sidecar = sidecar_path_for(video)
    if not sidecar.is_file():
        return "missing"
    try:
        sc = load(sidecar)
    except SidecarError:
        return "invalid"
    return check_freshness(video, sc)


def cmd_models(args: argparse.Namespace) -> int:
    """List available model aliases, pricing, and recommendations."""
    from . import pricing as pricing_cache
    from .generate import MODEL_ALIASES, MODEL_PRICING

    if getattr(args, "refresh", False):
        if pricing_cache.requests is None:
            print(
                "error: --refresh needs the requests package; "
                'install it with pip install "cdaf[openrouter]"',
                file=sys.stderr,
            )
            return 1
        targets = sorted(set(MODEL_ALIASES.values()))
        print(f"Syncing pricing for {len(targets)} models from OpenRouter...")
        results = pricing_cache.refresh(targets, force=True)
        priced = sum(1 for v in results.values() if v is not None)
        print(f"  {priced}/{len(targets)} priced, cached at {pricing_cache.cache_path()}\n")

    print("CDAF Supported Models & Aliases\n")
    print(f"{'Alias / Short Name':<20} | {'Full Model Identifier':<42} | {'Input $/1M':<10} | {'Output $/1M':<10}")
    print("-" * 90)
    synced = pricing_cache.load_prices()  # one cache read for the whole table
    synced_rows = 0
    for alias, full in sorted(MODEL_ALIASES.items()):
        is_synced = full.lower() in synced
        synced_rows += is_synced
        pricing = synced[full.lower()] if is_synced else pricing_cache.match_price(full, MODEL_PRICING)
        mark = "*" if is_synced else ""
        in_p = f"${pricing[0]:.2f}{mark}" if pricing else "N/A"
        out_p = f"${pricing[1]:.2f}{mark}" if pricing else "N/A"
        print(f"{alias:<20} | {full:<42} | {in_p:<10} | {out_p:<10}")
    if synced_rows:
        print("\n* synced from OpenRouter; unmarked rows come from the built-in table.")

    print("\nRecommended for General Video:")
    print("  --model flash-3.7       (Google Gemini 3.7 Flash - latest, fast, native video)")
    print("  --model flash           (Google Gemini 2.5 Flash - native video, cost efficient)")
    print("  --model pro             (Google Gemini 2.5 Pro - highest fidelity)")
    print("  --model qwen            (Qwen 2.5 VL 72B - open vision model via OpenRouter frame sampling)")
    print("\nPrices are approximate; check your provider's pricing page before relying on cost estimates.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .generate import GenerationError, generate_sidecar, load_dotenv_if_present

    if getattr(args, "list_models", False):
        return cmd_models(args)

    load_dotenv_if_present()

    if not args.paths:
        print("error: no video files or paths specified", file=sys.stderr)
        return 1

    videos = _iter_videos(args.paths)
    if not videos:
        print("no video files found", file=sys.stderr)
        return 1

    failures = 0
    for video in videos:
        sidecar = sidecar_path_for(video)
        if not args.force and _sidecar_state(video) == "fresh":
            print(f"  fresh   {video}  (skipped; use --force to regenerate)")
            continue
        print(f"  generating  {video} ...", flush=True)
        try:
            sc = generate_sidecar(
                video,
                provider=args.provider,
                model=args.model,
                detail=args.detail,
                mode=getattr(args, "mode", "auto"),
                api_key=args.api_key,
                base_url=getattr(args, "base_url", None),
                scene_threshold=getattr(args, "scene_threshold", None),
                chunk_duration=getattr(args, "chunk_duration", None),
                parallel=getattr(args, "parallel", 1),
            )
            save(sc, sidecar)
            cost_info = f" (cost: {sc.header['cost']})" if "cost" in sc.header else ""
            print(f"  wrote   {sidecar}{cost_info}")
        except (GenerationError, OSError) as e:
            failures += 1
            print(f"  FAILED  {video}: {e}", file=sys.stderr)
    return 1 if failures else 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    sidecar_file = path if path.suffix.lower() == ".cdaf" else sidecar_path_for(path)
    if not sidecar_file.is_file():
        print(f"MISSING  no sidecar at {sidecar_file}")
        return 2
    try:
        sc = load(sidecar_file)
    except SidecarError as e:
        print(f"INVALID  {sidecar_file}: {e}")
        return 3
    video = video_path_for(sidecar_file, sc.video)
    if not video:
        print(f"ORPHAN   {sidecar_file}: paired video '{sc.video}' not found")
        return 4
    state = check_freshness(video, sc, fast=args.fast)
    print(f"{state.upper():<8} {sidecar_file}  <->  {video.name}")
    return 0 if state == "fresh" else 5


def cmd_read(args: argparse.Namespace) -> int:
    path = Path(args.path)
    sidecar_file = path if path.suffix.lower() == ".cdaf" else sidecar_path_for(path)
    if not sidecar_file.is_file():
        print(f"error: no sidecar at {sidecar_file}", file=sys.stderr)
        return 2
    try:
        sc = load(sidecar_file)
    except SidecarError as e:
        print(f"error: invalid sidecar: {e}", file=sys.stderr)
        return 3
    if not args.no_verify:
        video = video_path_for(sidecar_file, sc.video)
        if video and check_freshness(video, sc) == "stale":
            print(
                f"error: sidecar is STALE — {video.name} changed since it was written. "
                "Regenerate with: cdaf generate --force",
                file=sys.stderr,
            )
            return 5
    print(sc.body)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    videos = _iter_videos([args.path])
    if not videos:
        print("no video files found", file=sys.stderr)
        return 1
    counts = {"fresh": 0, "stale": 0, "missing": 0, "invalid": 0}
    for video in videos:
        state = _sidecar_state(video)
        counts[state] += 1
        print(f"  {state.upper():<8} {video}")
    total = len(videos)
    print(
        f"\n{total} video(s): {counts['fresh']} fresh, {counts['stale']} stale, "
        f"{counts['missing']} missing, {counts['invalid']} invalid"
    )
    return 0 if counts["fresh"] == total else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cdaf",
        description="CDAF: Cached Descriptive Asset Files. Generate, validate, and read sidecars.",
    )
    parser.add_argument("--version", action="version", version=f"cdaf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create/refresh sidecars for videos")
    g.add_argument("paths", nargs="*", help="video files, sidecars, or directories (recursive)")
    g.add_argument(
        "--provider",
        choices=["auto", "gemini", "openrouter", "local"],
        default="auto",
        help="AI provider (default: auto; gemini, openrouter, or local OpenAI-compatible endpoint)",
    )
    g.add_argument(
        "--local",
        dest="provider",
        action="store_const",
        const="local",
        help="shorthand for --provider local (uses local OpenAI-compatible endpoint and ffmpeg scene detection)",
    )
    g.add_argument(
        "--model",
        default=None,
        help="model id or alias (e.g. flash-3.7, flash, pro, qwen, pixtral, llama, gpt4o; see `cdaf models`)",
    )
    g.add_argument(
        "--mode",
        choices=["auto", "screencast", "meeting", "demo", "presentation", "general"],
        default="auto",
        help="intelligent domain mode (screencast: track OS/apps/tools; meeting: track speakers/agenda/actions; demo: product flow; presentation: slides; default: auto)",
    )
    g.add_argument("--detail", choices=["brief", "standard", "rich"], default="standard")
    g.add_argument("--force", action="store_true", help="regenerate even if sidecar is fresh")
    g.add_argument("--api-key", default=None, help="API key (Gemini or OpenRouter)")
    g.add_argument("--base-url", default=None, help="custom API base URL (for OpenRouter or local endpoint)")
    g.add_argument(
        "--scene-threshold",
        type=float,
        default=None,
        metavar="F",
        help="local: ffmpeg scene-detect sensitivity, 0-1 (default 0.15)",
    )
    g.add_argument(
        "--chunk-duration",
        "--chunk-size",
        type=float,
        default=None,
        help="split long videos into chunks of N seconds and process in parallel (e.g. --chunk-duration 180)",
    )
    g.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=4,
        help="maximum number of concurrent workers for parallel chunk processing (default: 4)",
    )
    g.add_argument(
        "--list-models",
        action="store_true",
        help="list recommended model aliases, endpoints, and pricing",
    )
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("models", help="list recommended models, aliases, and pricing")
    m.add_argument(
        "--refresh",
        action="store_true",
        help="sync pricing from OpenRouter into a local 7-day cache (the only pricing network call)",
    )
    m.set_defaults(func=cmd_models)

    v = sub.add_parser("validate", help="check one sidecar is well-formed and fresh")
    v.add_argument("path", help="a video file or a .cdaf file")
    v.add_argument("--fast", action="store_true", help="size check only, skip hashing")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("read", help="print a sidecar's body (verifies freshness first)")
    r.add_argument("path", help="a video file or a .cdaf file")
    r.add_argument("--no-verify", action="store_true", help="print without hashing the video")
    r.set_defaults(func=cmd_read)

    s = sub.add_parser("status", help="fresh/stale/missing report for a directory tree")
    s.add_argument("path", nargs="?", default=".", help="directory to scan (default: .)")
    s.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
