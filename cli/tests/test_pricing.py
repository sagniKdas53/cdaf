"""Tests for the optional pricing autosync cache."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cdaf import pricing
from cdaf.generate import MODEL_PRICING, calculate_cost, resolve_pricing


class _TempCache:
    """Point the cache at a scratch file for the duration of a test."""

    def __enter__(self) -> Path:
        self._dir = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("CDAF_PRICING_CACHE")
        self.path = Path(self._dir.name) / "pricing.json"
        os.environ["CDAF_PRICING_CACHE"] = str(self.path)
        return self.path

    def __exit__(self, *exc) -> None:
        if self._prev is None:
            os.environ.pop("CDAF_PRICING_CACHE", None)
        else:
            os.environ["CDAF_PRICING_CACHE"] = self._prev
        self._dir.cleanup()


def _write_cache(path: Path, models: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": pricing.CACHE_VERSION, "models": models}))


class TestMatchPrice(unittest.TestCase):
    def test_exact_match_wins(self):
        table = {"openai/gpt-4o-mini": (0.15, 0.60), "openai/gpt-4o": (2.50, 10.00)}
        self.assertEqual(pricing.match_price("openai/gpt-4o", table), (2.50, 10.00))

    def test_never_matches_a_longer_key_against_a_shorter_model(self):
        """Regression: a bidirectional match priced gpt-4o at the mini rate (16x low)."""
        table = {"openai/gpt-4o-mini": (0.15, 0.60)}
        self.assertIsNone(pricing.match_price("openai/gpt-4o", table))

    def test_longest_substring_wins_regardless_of_dict_order(self):
        table = {"openai/gpt-4o": (2.50, 10.00), "openai/gpt-4o-mini": (0.15, 0.60)}
        self.assertEqual(
            pricing.match_price("openai/gpt-4o-mini-2024-07-18", table), (0.15, 0.60)
        )

    def test_unknown_model_is_none(self):
        self.assertIsNone(pricing.match_price("who/knows", MODEL_PRICING))

    def test_empty_model_is_none(self):
        self.assertIsNone(pricing.match_price("", MODEL_PRICING))

    def test_shipped_aliases_price_correctly(self):
        self.assertEqual(resolve_pricing("openai/gpt-4o"), (2.50, 10.00))
        self.assertEqual(resolve_pricing("openai/gpt-4o-mini"), (0.15, 0.60))


class TestFreeModelDetection(unittest.TestCase):
    def test_free_suffixes(self):
        self.assertTrue(pricing.is_free_model("qwen/qwen3.8-flash:free"))
        self.assertTrue(pricing.is_free_model("google/gemini-2.5-flash:free"))
        self.assertTrue(pricing.is_free_model("openrouter/free"))

    def test_non_openrouter_free_suffixes_are_not_free(self):
        # A trailing "-free" or "/free" is a real (priced) model: a user-named
        # self-host like "some/model-free", or a vendor id ending "/free" that
        # isn't OpenRouter's aggregator. They must NOT resolve to $0.
        self.assertFalse(pricing.is_free_model("some/model-free"))
        self.assertFalse(pricing.is_free_model("vendor/freestyle-vl"))
        self.assertFalse(pricing.is_free_model("mistralai/mistral-freeform"))
        # Empty / weird input
        self.assertFalse(pricing.is_free_model(""))
        # A user could name their own local model "free" — we still don't
        # treat that as free, because we only know OpenRouter's two shapes.
        self.assertFalse(pricing.is_free_model("free"))

    def test_free_models_resolve_to_zero(self):
        self.assertEqual(pricing.match_price("google/gemini-2.5-flash:free", MODEL_PRICING), (0.0, 0.0))
        self.assertEqual(pricing.match_price("openrouter/free", MODEL_PRICING), (0.0, 0.0))
        self.assertEqual(resolve_pricing("google/gemini-2.5-flash:free"), (0.0, 0.0))
        self.assertEqual(resolve_pricing("qwen/qwen3.8-flash:free"), (0.0, 0.0))
        self.assertEqual(resolve_pricing("openrouter/free"), (0.0, 0.0))

    def test_free_substring_is_not_free(self):
        """Regression: 'free' anywhere in the id priced unrelated models at $0."""
        self.assertFalse(pricing.is_free_model("mistralai/mistral-freeform"))
        self.assertFalse(pricing.is_free_model("vendor/freestyle-vl"))


class TestLookup(unittest.TestCase):
    def test_missing_cache_returns_none(self):
        with _TempCache():
            self.assertIsNone(pricing.lookup("openai/gpt-4o"))

    def test_fresh_entry_is_returned(self):
        with _TempCache() as path:
            _write_cache(path, {"openai/gpt-4o": {
                "prompt_per_million": 1.0,
                "completion_per_million": 2.0,
                "fetched_at": time.time(),
                "source": "openrouter",
            }})
            self.assertEqual(pricing.lookup("openai/gpt-4o"), (1.0, 2.0))

    def test_stale_entry_is_ignored(self):
        with _TempCache() as path:
            _write_cache(path, {"openai/gpt-4o": {
                "prompt_per_million": 1.0,
                "completion_per_million": 2.0,
                "fetched_at": time.time() - pricing.CACHE_TTL_SECONDS - 1,
                "source": "openrouter",
            }})
            self.assertIsNone(pricing.lookup("openai/gpt-4o"))

    def test_recorded_miss_returns_none(self):
        with _TempCache() as path:
            _write_cache(path, {"who/knows": {
                "fetched_at": time.time(), "source": "unavailable",
            }})
            self.assertIsNone(pricing.lookup("who/knows"))

    def test_corrupt_cache_is_ignored(self):
        with _TempCache() as path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not json")
            self.assertIsNone(pricing.lookup("openai/gpt-4o"))

    def test_synced_price_overrides_the_static_table(self):
        with _TempCache() as path:
            _write_cache(path, {"gemini-2.5-flash": {
                "prompt_per_million": 9.0,
                "completion_per_million": 9.0,
                "fetched_at": time.time(),
                "source": "openrouter",
            }})
            self.assertEqual(resolve_pricing("gemini-2.5-flash"), (9.0, 9.0))


class TestNoNetworkDuringGeneration(unittest.TestCase):
    def test_calculate_cost_never_fetches(self):
        """Cost accounting must not reach openrouter.ai, whatever the provider."""
        with _TempCache():
            with patch.object(pricing, "fetch_price", side_effect=AssertionError("network!")):
                self.assertIsNotNone(calculate_cost("gemini-2.5-flash", 1000, 1000))
                self.assertIsNone(calculate_cost("who/knows", 1000, 1000))

    def test_resolve_pricing_survives_a_broken_cache_module(self):
        with patch.object(pricing, "lookup", side_effect=RuntimeError("boom")):
            self.assertEqual(resolve_pricing("gemini-2.5-flash"), (0.30, 2.50))


class TestRefresh(unittest.TestCase):
    def _response(self, prompt: str, completion: str) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": {"endpoints": [{"pricing": {"prompt": prompt, "completion": completion}}]}
        }
        return resp

    @patch("requests.get")
    def test_refresh_writes_per_million_rates(self, mock_get):
        mock_get.return_value = self._response("0.0000025", "0.00001")
        with _TempCache() as path:
            results = pricing.refresh(["openai/gpt-4o"], force=True)
            self.assertEqual(results["openai/gpt-4o"], (2.50, 10.00))
            self.assertEqual(pricing.lookup("openai/gpt-4o"), (2.50, 10.00))
            self.assertTrue(path.is_file())

    @patch("requests.get")
    def test_refresh_averages_endpoints(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": {"endpoints": [
            {"pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"pricing": {"prompt": "0.000003", "completion": "0.000004"}},
        ]}}
        mock_get.return_value = resp
        with _TempCache():
            self.assertEqual(pricing.refresh(["a/b"], force=True)["a/b"], (2.0, 3.0))

    @patch("requests.get")
    def test_failed_fetch_is_recorded_as_a_miss(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp
        with _TempCache() as path:
            self.assertIsNone(pricing.refresh(["who/knows"], force=True)["who/knows"])
            cached = json.loads(path.read_text())["models"]["who/knows"]
            self.assertEqual(cached["source"], "unavailable")

    @patch("requests.get")
    def test_second_refresh_reuses_a_fresh_entry(self, mock_get):
        mock_get.return_value = self._response("0.000001", "0.000002")
        with _TempCache():
            pricing.refresh(["a/b"], force=True)
            self.assertEqual(mock_get.call_count, 1)
            pricing.refresh(["a/b"])  # not forced: served from cache
            self.assertEqual(mock_get.call_count, 1)

    @patch("requests.get")
    def test_recorded_miss_is_not_refetched_for_a_week(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp
        with _TempCache():
            pricing.refresh(["who/knows"], force=True)
            pricing.refresh(["who/knows"])
            self.assertEqual(mock_get.call_count, 1)

    @patch("requests.get")
    def test_free_variant_skips_the_network(self, mock_get):
        with _TempCache():
            self.assertEqual(pricing.refresh(["a/b:free"], force=True)["a/b:free"], (0.0, 0.0))
            mock_get.assert_not_called()

    @patch("requests.get", side_effect=OSError("no route to host"))
    def test_network_error_is_swallowed(self, mock_get):
        with _TempCache():
            self.assertIsNone(pricing.refresh(["a/b"], force=True)["a/b"])


class TestLoadPrices(unittest.TestCase):
    def test_returns_only_fresh_entries(self):
        now = time.time()
        with _TempCache() as path:
            _write_cache(path, {
                "fresh/model": {
                    "prompt_per_million": 1.0, "completion_per_million": 2.0,
                    "fetched_at": now, "source": "openrouter",
                },
                "stale/model": {
                    "prompt_per_million": 3.0, "completion_per_million": 4.0,
                    "fetched_at": now - pricing.CACHE_TTL_SECONDS - 1, "source": "openrouter",
                },
                "missed/model": {"fetched_at": now, "source": "unavailable"},
            })
            self.assertEqual(pricing.load_prices(), {"fresh/model": (1.0, 2.0)})

    def test_empty_without_a_cache(self):
        with _TempCache():
            self.assertEqual(pricing.load_prices(), {})


class TestModelsCommand(unittest.TestCase):
    """`cdaf models` must not misprice a shipped alias."""

    def _table(self) -> dict[str, tuple[str, str]]:
        import io
        from contextlib import redirect_stdout

        from cdaf.cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["models"]), 0)
        rows = {}
        for line in buf.getvalue().splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 4 and parts[3].startswith("$"):
                rows[parts[0]] = (parts[2], parts[3])
        return rows

    def test_gpt4o_is_not_priced_at_the_mini_rate(self):
        with _TempCache():
            rows = self._table()
            self.assertEqual(rows["gpt4o"], ("$2.50", "$10.00"))
            self.assertEqual(rows["gpt4o-mini"], ("$0.15", "$0.60"))

    def test_synced_prices_are_used_and_marked(self):
        with _TempCache() as path:
            _write_cache(path, {"openai/gpt-4o": {
                "prompt_per_million": 7.0, "completion_per_million": 8.0,
                "fetched_at": time.time(), "source": "openrouter",
            }})
            self.assertEqual(self._table()["gpt4o"], ("$7.00*", "$8.00*"))

    def test_listing_models_makes_no_network_call(self):
        with _TempCache():
            with patch.object(pricing, "fetch_price", side_effect=AssertionError("network!")):
                self._table()


if __name__ == "__main__":
    unittest.main()
