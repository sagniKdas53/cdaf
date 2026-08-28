# CDAF — Cached Descriptive Asset Files

[![tests](https://github.com/UditAkhourii/cdaf/actions/workflows/tests.yml/badge.svg)](https://github.com/UditAkhourii/cdaf/actions/workflows/tests.yml)
[![npm](https://img.shields.io/npm/v/cdaf-skill.svg)](https://www.npmjs.com/package/cdaf-skill)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Spec: v1.0](https://img.shields.io/badge/spec-v1.0-blue.svg)](SPEC.md)

**Stop making AI agents watch the same video twice.**

CDAF is an open sidecar format for video: a plain-text, timestamped description that
lives next to the video file with the same basename. Generate it once, and every AI
agent that touches that footage afterward reads a few hundred text tokens instead of
running a full video-understanding pass.

### Teach your agent the format — one command

```bash
npx cdaf-skill
```

Works on Windows, macOS, and Linux. That installs the [agent skill](#-agent-skill), so
your coding agent checks for a `.cdaf` sidecar and verifies it against the video's hash
*before* it ever spends tokens watching. Then describe your footage once:

```bash
pip install "cdaf[generate] @ git+https://github.com/UditAkhourii/cdaf.git#subdirectory=cli"
cdaf generate ./footage
```

```
footage/
├── sunset-drone.mp4     ← plays everywhere, untouched
└── sunset-drone.cdaf    ← what agents read instead of watching
```

![CDAF turns repeated per-task video analysis into one per-asset description pass followed by verified text reads](figures/concept.svg)

**Measured** (reproducible benchmark, `gemini-2.5-flash`, 20 questions): answering
from the sidecar matched direct video analysis on accuracy — **20/20 vs 19/20** — at
**10.1× fewer prompt tokens per question** (303 vs 3,066) and ~35% lower latency. The
ratio grows linearly with clip length (~50× for 60-second footage). In production use
at scale, this pattern cut video-workflow AI costs to roughly **1/25th**.

---

## Components

| Component | Where | What it does |
|---|---|---|
| **[Spec](#-spec)** | [SPEC.md](SPEC.md) | The normative format definition (v1.0) |
| **[Engine](#-engine-core-library)** | [cli/cdaf/](cli/cdaf/) | Python library: parse, validate, hash, generate |
| **[CLI](#-cli)** | [cli/](cli/) | `cdaf` command: generate / validate / read / status |
| **Local provider** | [cli/cdaf/local.py](cli/cdaf/local.py) | `--local`: generate with a local model, no API key |
| **[Agent Skill](#-agent-skill)** | [skills/](skills/claude-code/cdaf/SKILL.md) | Teaches agents the sidecar-first protocol · `npx cdaf-skill` |
| **[Benchmarks](#-benchmarks)** | [benchmarks/](benchmarks/) | Reproducible eval: sidecar vs direct video |
| **[Paper](#-paper)** | [paper/PREPRINT.md](paper/PREPRINT.md) | arXiv preprint draft built on the benchmark |

---

## 📄 Spec

A CDAF asset is `clip.mp4` + `clip.cdaf`: a UTF-8 sidecar with a minimal `key: value`
header and a markdown body. The header carries the **SHA-256 and byte size of the
exact video described** — edit the video and every conforming tool detects the sidecar
as stale and refuses to use it. The body is optimized for the actual reader (an LLM):

```
--- CDAF/1.0
video: sunset-drone.mp4
sha256: 4a7d1ed4…
bytes: 48211394
duration: 00:00:31.500
generator: gemini-2.5-flash
created: 2026-08-26T14:03:12Z
---

## Summary
A 31-second aerial drone clip of a coastal highway at golden hour…

## Segments
[00:00.0-00:05.2] Wide aerial establishing shot: a two-lane coastal highway…
[00:05.2-00:12.8] The drone pushes in and descends toward the highway…

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
drone, aerial, coastal highway, golden hour, sunset, b-roll, …
```

Full normative rules (freshness semantics, versioning, section grammar):
**[SPEC.md](SPEC.md)**. A complete example:
[examples/sunset-drone.cdaf](examples/sunset-drone.cdaf).

## ⚙️ Engine (core library)

The `cdaf` Python package is split by dependency weight:

- **Zero-dependency core** ([sidecar.py](cli/cdaf/sidecar.py)) — parse, serialize,
  chunked SHA-256, freshness checking, segment extraction. Agents and CI can validate
  sidecars with nothing but the standard library.
- **Generator** ([generate.py](cli/cdaf/generate.py)) — Gemini & OpenRouter multi-provider
  support, bring-your-own-key, `brief`/`standard`/`rich` detail profiles, token-usage capture.
  Supports any video-capable model on OpenRouter (`google/gemini-3.7-flash`, `qwen/qwen3.5-flash-02-23`,
  `qwen/qwen2.5-vl-72b-instruct`, etc.) or direct Gemini API.
- **Probe** ([probe.py](cli/cdaf/probe.py)) — optional ffprobe metadata
  (duration/resolution/fps), degrades gracefully when ffprobe is absent.

```python
from cdaf import load, check_freshness, sidecar_path_for

sc = load(sidecar_path_for("footage/sunset-drone.mp4"))
if check_freshness("footage/sunset-drone.mp4", sc) == "fresh":
    context_for_llm = sc.body
```

The format is model-agnostic: use OpenRouter to access dozens of video-capable models,
direct Gemini, or custom endpoints behind the same `Sidecar` type.

## 💻 CLI

Requires Python ≥ 3.10. Bring your own Gemini API key or OpenRouter API key —
your footage goes to your key, not to us.

```bash
pip install "cdaf[generate] @ git+https://github.com/UditAkhourii/cdaf.git#subdirectory=cli"

# Option A: With Gemini

export GEMINI_API_KEY=your-key          # PowerShell: $env:GEMINI_API_KEY="your-key"
cdaf generate ./footage                 # describe every video, skip fresh sidecars

# Option B: With OpenRouter (use any model that supports video!)
export OPENROUTER_API_KEY=your-key      # PowerShell: $env:OPENROUTER_API_KEY="your-key"
cdaf generate ./footage --model or-flash-3.7
cdaf generate ./footage --model qwen
cdaf generate ./footage --model pixtral

# Intelligent Domain Modes: Screencasts, Meetings, Demos & Presentations
cdaf generate ./screencasts --mode screencast       # tracks active OS, IDEs, terminals & browser tabs
cdaf generate ./recorded-calls --mode meeting       # tracks participants, active speakers & action items
cdaf generate ./product-demos --mode demo           # tracks feature walkthroughs and UI flows

# Advanced Generation: Chunking & Parallelism for Large Videos
cdaf generate ./long-footage --chunk-duration 180 --parallel 4

# Model catalog & pricing table
cdaf models
cdaf models --refresh                   # sync prices from OpenRouter into a 7-day local cache

# Verification & Inspection (zero external dependencies)
cdaf status ./footage                   # FRESH / STALE / MISSING report
cdaf read ./footage/sunset-drone.mp4    # print the description (verifies hash first)
cdaf validate ./footage/clip.mp4        # exit 0 iff sidecar is well-formed and fresh
```

Working from a clone instead? `pip install ./cli[generate]`.

Flags:
- `--provider auto|gemini|openrouter`: AI provider backend
- `--model <id-or-alias>`: Model selector (`flash-3.7`, `flash`, `pro`, `qwen`, `pixtral`, `llama`, `gpt4o`, etc.)
- `--mode auto|screencast|meeting|demo|presentation|general`: Domain-aware prompt tuning (tracks active OS/apps/tools for screencasts, attendees/actions for meetings)
- `--chunk-duration <seconds>`: Automatically split long video files into temporal slices and describe in parallel
- `--parallel <N>` / `-j <N>`: Concurrency worker count for chunked processing (default: 4)
- `--detail brief|standard|rich`: Output granularity
- `--force`: Regenerate sidecar even if fresh
- `--api-key <key>`, `--base-url <url>`: Custom API credentials and OpenAI-compatible endpoints
- `cdaf models`: Display all model aliases, full identifiers, and per-token pricing

Cost Tracking:
Generated sidecars include estimated cost and token metrics in the header (`cost: $0.0018`, `prompt_tokens: 12045`, `output_tokens: 450`). Costs are estimates from a built-in pricing table that can drift from provider pricing. `cdaf models --refresh` syncs live rates from OpenRouter's public API into `~/.cache/cdaf/pricing.json` (7-day TTL) and later runs prefer those. That refresh is the only pricing code that touches the network — generating a sidecar never does, whichever provider you use. Unknown models omit the cost header rather than guess a rate. `validate`/`read`/`status` need **no dependencies and no API key**. `cdaf read` refuses to print a stale sidecar.

## 🤖 Agent Skill

The skill turns the format into behavior: **before analyzing any video, check for the
sidecar; verify freshness (size check for exploration, full hash for consequential
decisions); read it instead of watching; grep `.cdaf` files to search whole libraries;
regenerate when stale.**

Install it with `npx cdaf-skill` (see [above](#teach-your-agent-the-format--one-command)),
or pick a scope:

| Command | Installs to |
|---|---|
| `npx cdaf-skill` | `~/.claude/skills/cdaf` (all your projects) |
| `npx cdaf-skill --project` | `./.claude/skills/cdaf` (this project only) |
| `npx cdaf-skill --dir <path>` | anywhere you want |
| `npx cdaf-skill --print` | stdout — for pasting into other agent frameworks |

Re-running is safe: an unchanged skill is left alone, and a modified one is backed up
to `SKILL.md.bak` before updating. Prefer to copy it by hand? The file is
[skills/claude-code/cdaf/SKILL.md](skills/claude-code/cdaf/SKILL.md).

Any other agent framework — the whole contract is one system-prompt paragraph:

> Before analyzing a video file, look for a `.cdaf` file with the same basename. If
> its `bytes`/`sha256` header matches the video file, read it instead of processing
> the video.

### Why agentic video editors care

Programmatic and agent-native editors (Remotion, HyperFrames, prompt-to-edit tools)
already do everything in text — compositions are code, cut decisions are data. Video
understanding is the one step that forces them out of the text domain, priced per
exposure (~263 tokens per second of footage on Gemini-class models). With CDAF:

- **Selection becomes retrieval** — "sunset coastal shots, no people" is a grep over
  `.cdaf` files, not a multimodal sweep of the library.
- **Cut lists come from segment timestamps** — `[00:05.2-00:12.8]` lines map straight
  to Remotion `<Sequence>`s or HyperFrames clips.
- **Captions and audio sync for free** — the Transcript section aligns speech to the
  timeline.
- **The library appreciates** — 40 candidate clips ≈ 473k tokens to watch, ~24k to
  read — every session after the first.

## 📊 Benchmarks

Fully reproducible: [bench.py](benchmarks/bench.py) synthesizes test videos with
ffmpeg from scripted recipes (known colors, hard cuts, timed word overlays), so ground
truth is exact and grading is objective — no LLM judge, no dataset license.

```bash
python benchmarks/bench.py make     # synthesize testset (needs ffmpeg)
python benchmarks/bench.py run     # both conditions via Gemini (needs GEMINI_API_KEY)
python benchmarks/bench.py report  # writes benchmarks/RESULTS.md
```

![Accuracy, prompt-token cost, and latency for 20 questions in each condition](figures/benchmark.svg)

| Condition | Accuracy | Mean prompt tokens/question | Mean latency |
|---|---|---|---|
| Direct video | 19/20 (95%) | 3,066 | 3.46 s |
| **CDAF sidecar** | **20/20 (100%)** | **303** | **2.24 s** |

Each sidecar-answered question saves 2,763 prompt tokens, so generating a sidecar
(3,601 tokens) **breaks even after ≈1.3 questions per video** — everything after that
is the ~10× saving. Per-clip and per-question detail:
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## 📝 Paper

**[https://zenodo.org/records/22110594](https://zenodo.org/records/22110594)** — *Cached Descriptive Asset Files (CDAF):
A Sidecar Format for Token-Efficient Video Understanding in Agentic Pipelines* —
arXiv draft (cs.MM / cs.AI): format rationale, benchmark methodology and results,
production case study, integration analysis for agentic editors, limitations.

## Roadmap

- PyPI release of the `cdaf` CLI (`pip install cdaf`)
- MCP server exposing `cdaf_read` / `cdaf_status` / `cdaf_generate` to any MCP client
- Hosted free converter (no local install; bring-your-own-key or free quota)
- Additional generator backends (Claude, GPT, local VLMs)
- Node/TypeScript port of the engine
- Natural-footage benchmark extension with human-verified QA
- Signed sidecars (`signature` header) for cross-organization trust

## License

MIT — see [LICENSE](LICENSE).
