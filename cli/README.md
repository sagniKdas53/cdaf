# cdaf (CLI)

Generate and validate CDAF sidecars — timestamped descriptive text files that let AI
agents reuse one video-understanding pass instead of re-analyzing footage.

Install from the repository:

```bash
pip install "cdaf[generate] @ git+https://github.com/UditAkhourii/cdaf.git#subdirectory=cli"
```

Drop the `[generate]` extra for `validate` / `read` / `status` only -- those need no
dependencies and no key.

Three generation providers:

```bash
# Option 1: Gemini
export GEMINI_API_KEY=your-key
cdaf generate ./footage

# Option 2: OpenRouter (Gemini, Qwen, Pixtral, Llama, GPT-4o)
export OPENROUTER_API_KEY=your-key
cdaf generate ./footage --model or-flash
cdaf generate ./footage --model qwen

# Option 3: Local OpenAI-compatible endpoint (Ollama / vLLM)
cdaf generate ./clip.mp4 --local
```

`--local` needs `ffmpeg` and a served multimodal model instead of a key, and never
sends the footage anywhere. Point it with `--base-url` / `--model` (or `CDAF_BASE_URL`
/ `CDAF_LOCAL_MODEL`); set `CDAF_PROVIDER=local` to make it the default.

Commands: `generate`, `validate`, `read`, `status`, `models`. Full docs and the format
specification live in the repository root: see `README.md` and `SPEC.md`.
