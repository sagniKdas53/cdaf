"""Optional pricing autosync for the cost headers CDAF writes into sidecars.

`generate.MODEL_PRICING` is a small hand-maintained table, so it drifts every
time a provider changes its rates. This module adds an opt-in cache that
`cdaf models --refresh` fills from OpenRouter's public endpoints API.

Two rules keep it unsurprising:

- Generating a sidecar never makes a pricing network call. `lookup()` reads the
  on-disk cache and nothing else; `refresh()` is the only function that talks to
  OpenRouter, and only `cdaf models --refresh` calls it. Running `cdaf generate`
  against Gemini or a local endpoint stays entirely off openrouter.ai.
- An unknown model resolves to None so the caller omits the cost header, rather
  than guessing a rate.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:  # requests is an optional extra; the cache works fine without it
    import requests
except ImportError:  # pragma: no cover - exercised via the openrouter extra
    requests = None  # type: ignore[assignment]

CACHE_VERSION = 1
CACHE_TTL_SECONDS = 7 * 24 * 3600
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


def cache_path() -> Path:
    """Where the synced prices live. Override with CDAF_PRICING_CACHE (tests use this)."""
    override = os.environ.get("CDAF_PRICING_CACHE")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "cdaf" / "pricing.json"


def match_price(
    model: str, table: dict[str, tuple[float, float]]
) -> tuple[float, float] | None:
    """Resolve a model id against a price table.

    An exact match wins. Otherwise the *longest* table key that is a substring of
    the model id wins, so "openai/gpt-4o-mini-2024-07-18" gets the mini rate
    instead of whichever of "openai/gpt-4o" and "openai/gpt-4o-mini" the dict
    happens to list first.

    The match is deliberately one-directional. Also accepting a key because it
    *contains* the model id resolves the plain "openai/gpt-4o" to the
    "openai/gpt-4o-mini" row -- a 16x understatement of the real rate.
    """
    if not model:
        return None
    if is_free_model(model):
        return (0.0, 0.0)
    needle = model.lower()
    exact = table.get(model) or table.get(needle)
    if exact is not None:
        return exact
    best_key: str | None = None
    for key in table:
        if key.lower() in needle and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return table[best_key] if best_key is not None else None


def is_free_model(model: str) -> bool:
    """True for OpenRouter's zero-cost variants.

    Only the ":free" suffix and a trailing "-free" count. Matching "free"
    anywhere in the id prices unrelated models such as
    "mistralai/mistral-freeform" at $0.
    """
    low = model.lower()
    return low.endswith(":free") or low.endswith("-free") or low.endswith("/free") or low == "free"


def _load_cache(path: Path | None = None) -> dict:
    path = path or cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": CACHE_VERSION, "models": {}}
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "models": {}}
    models = data.get("models")
    if not isinstance(models, dict):
        return {"version": CACHE_VERSION, "models": {}}
    return {"version": CACHE_VERSION, "models": models}


def _save_cache(cache: dict, path: Path | None = None) -> None:
    path = path or cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass  # a read-only cache dir must not break generation


def _fresh_entry(entry: object, now: float) -> dict | None:
    if not isinstance(entry, dict):
        return None
    fetched_at = entry.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return None
    if now - fetched_at >= CACHE_TTL_SECONDS:
        return None
    return entry


def lookup(model: str, *, now: float | None = None) -> tuple[float, float] | None:
    """Synced per-million (prompt, completion) rates for `model`, or None.

    Reads the cache only. A miss -- no cache, a stale entry, or a model a
    previous refresh could not price -- returns None so the caller falls back to
    the static table.
    """
    now = time.time() if now is None else now
    models = _load_cache().get("models", {})
    entry = _fresh_entry(models.get(model.lower()), now)
    return _rates(entry) if entry is not None else None


def _rates(entry: dict) -> tuple[float, float] | None:
    prompt = entry.get("prompt_per_million")
    completion = entry.get("completion_per_million")
    if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
        return None  # a recorded miss, or a malformed entry
    return (float(prompt), float(completion))


def load_prices(*, now: float | None = None) -> dict[str, tuple[float, float]]:
    """Every fresh synced price, keyed by lowercased model id.

    For callers that price many models at once -- one cache read instead of one
    per model.
    """
    now = time.time() if now is None else now
    out: dict[str, tuple[float, float]] = {}
    for key, entry in _load_cache().get("models", {}).items():
        fresh = _fresh_entry(entry, now)
        rates = _rates(fresh) if fresh else None
        if rates is not None:
            out[key] = rates
    return out


def fetch_price(model: str, timeout: float = 10.0) -> tuple[float, float] | None:
    """Average per-million (prompt, completion) rates across a model's endpoints.

    Returns None when the model is unknown, unpriced, or the request fails.
    """
    if is_free_model(model):
        return (0.0, 0.0)
    if requests is None:
        return None
    try:
        resp = requests.get(ENDPOINTS_URL.format(model=model), timeout=timeout)
        if resp.status_code != 200:
            return None
        endpoints = (resp.json().get("data") or {}).get("endpoints") or []
    except Exception:
        return None

    prompts: list[float] = []
    completions: list[float] = []
    for endpoint in endpoints:
        pricing = endpoint.get("pricing") or {}
        try:  # OpenRouter reports per-token rates, as strings
            prompts.append(float(pricing.get("prompt") or 0))
            completions.append(float(pricing.get("completion") or 0))
        except (TypeError, ValueError):
            continue
    if not prompts:
        return None
    per_million = 1_000_000.0
    return (
        sum(prompts) / len(prompts) * per_million,
        sum(completions) / len(completions) * per_million,
    )


def refresh(models: list[str], *, force: bool = False) -> dict[str, tuple[float, float] | None]:
    """Sync prices for `models` into the cache. This is the only network path.

    Fresh entries are reused unless `force`. Models that cannot be priced are
    recorded as misses so a repeated refresh does not re-request them for a week.
    """
    now = time.time()
    cache = _load_cache()
    stored = cache.setdefault("models", {})
    results: dict[str, tuple[float, float] | None] = {}

    for model in models:
        key = model.lower()
        cached = None if force else _fresh_entry(stored.get(key), now)
        if cached is not None:
            results[model] = _rates(cached)
            continue
        price = fetch_price(model)
        if price is None:
            stored[key] = {"fetched_at": now, "source": "unavailable"}
            results[model] = None
        else:
            stored[key] = {
                "prompt_per_million": price[0],
                "completion_per_million": price[1],
                "fetched_at": now,
                "source": "free" if is_free_model(model) else "openrouter",
            }
            results[model] = price

    _save_cache(cache)
    return results
