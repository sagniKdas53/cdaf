"""Tests for video description generation with OpenRouter, Gemini, chunking, and parallelism."""

import base64
import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


@contextlib.contextmanager
def temp_video(suffix=".mp4", data=b"dummy content"):
    """A real closed file on disk (NamedTemporaryFile stays locked on Windows)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"clip{suffix}"
        p.write_bytes(data)
        yield p

from cdaf.generate import (
    GenerationError,
    _video_data_url,
    calculate_cost,
    describe_video,
    format_cost,
    format_timestamp,
    generate_sidecar,
    merge_chunk_bodies,
    offset_body_timestamps,
    parse_timestamp,
    resolve_provider_and_model,
)
from cdaf.sidecar import Sidecar

SAMPLE_BODY = """## Summary
A brief test video.

## Segments
[00:00.0-00:05.0] Opening test scene.
[00:05.0-00:10.0] Closing scene.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
test, demo
"""


class TestProviderAndModelResolution(unittest.TestCase):
    def test_explicit_provider(self):
        p, m = resolve_provider_and_model(provider="openrouter", model="custom/model")
        self.assertEqual(p, "openrouter")
        self.assertEqual(m, "custom/model")

        p, m = resolve_provider_and_model(provider="gemini", model="gemini-1.5-pro")
        self.assertEqual(p, "gemini")
        self.assertEqual(m, "gemini-1.5-pro")

    def test_explicit_provider_default_models(self):
        p, m = resolve_provider_and_model(provider="openrouter", model=None)
        self.assertEqual(p, "openrouter")
        self.assertEqual(m, "google/gemini-2.5-flash")

        p, m = resolve_provider_and_model(provider="gemini", model=None)
        self.assertEqual(p, "gemini")
        self.assertEqual(m, "gemini-2.5-flash")

    def test_model_aliases(self):
        p, m = resolve_provider_and_model(model="flash")
        self.assertEqual(m, "gemini-2.5-flash")

        p, m = resolve_provider_and_model(model="flash-3.7")
        self.assertEqual(m, "gemini-3.7-flash")

        p, m = resolve_provider_and_model(model="or-flash-3.7")
        self.assertEqual(m, "google/gemini-3.7-flash")
        self.assertEqual(p, "openrouter")

        p, m = resolve_provider_and_model(model="qwen")
        self.assertEqual(m, "qwen/qwen2.5-vl-72b-instruct")
        self.assertEqual(p, "openrouter")

        p, m = resolve_provider_and_model(model="pixtral")
        self.assertEqual(m, "mistralai/pixtral-large-2411")
        self.assertEqual(p, "openrouter")

    def test_auto_detect_from_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            p, m = resolve_provider_and_model(provider="auto", api_key="sk-or-mock-key")
            self.assertEqual(p, "openrouter")

    def test_auto_detect_from_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test", "GEMINI_API_KEY": ""}, clear=True):
            p, m = resolve_provider_and_model(provider="auto", model=None)
            self.assertEqual(p, "openrouter")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-gemini-key", "OPENROUTER_API_KEY": ""}, clear=True):
            p, m = resolve_provider_and_model(provider="auto", model=None)
            self.assertEqual(p, "gemini")

    def test_invalid_provider_raises(self):
        with self.assertRaises(ValueError):
            resolve_provider_and_model(provider="invalid_provider")


class TestCostCalculation(unittest.TestCase):
    def test_calculate_cost(self):
        # gemini-2.5-flash: $0.30/1M in, $2.50/1M out
        cost = calculate_cost("gemini-2.5-flash", prompt_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(cost, 2.80, places=4)

        # 10,000 in, 1,000 out -> (10,000 * 0.30 + 1,000 * 2.50) / 1,000,000 = 0.0055
        cost_small = calculate_cost("google/gemini-2.5-flash", prompt_tokens=10_000, output_tokens=1_000)
        self.assertAlmostEqual(cost_small, 0.0055, places=6)

    def test_unknown_model_has_no_cost(self):
        self.assertIsNone(calculate_cost("some/unknown-model", prompt_tokens=1000, output_tokens=100))

    def test_free_model_cost_is_zero(self):
        cost = calculate_cost("google/gemini-2.5-flash:free", prompt_tokens=10000, output_tokens=1000)
        self.assertEqual(cost, 0.0)
        cost_qwen = calculate_cost("qwen/qwen3.8-flash:free", prompt_tokens=5000, output_tokens=500)
        self.assertEqual(cost_qwen, 0.0)

    def test_format_cost(self):
        self.assertEqual(format_cost(0.0), "$0.00")
        self.assertEqual(format_cost(0), "$0.00")
        self.assertEqual(format_cost(0.0021), "$0.0021")
        self.assertEqual(format_cost(0.001), "$0.0010")
        self.assertEqual(format_cost(0.000045), "$0.000045")
        self.assertIsNone(format_cost(None))


class TestTimestampManipulation(unittest.TestCase):
    def test_parse_and_format_timestamp(self):
        self.assertEqual(parse_timestamp("01:23.4"), 83.4)
        self.assertEqual(parse_timestamp("01:02:03.5"), 3723.5)
        self.assertEqual(format_timestamp(83.4), "01:23.4")
        self.assertEqual(format_timestamp(3723.5), "01:02:03.5")

    def test_offset_body_timestamps(self):
        body = """## Summary
Test.

## Segments
[00:00.0-00:15.0] Intro scene.
[00:15.0-01:00.0] Middle scene.

## Transcript
[00:05.0] Speaker: Hello.

## On-screen Text
[00:01.0] "Title"
"""
        offset_body = offset_body_timestamps(body, 120.0)
        self.assertIn("[02:00.0-02:15.0] Intro scene.", offset_body)
        self.assertIn("[02:15.0-03:00.0] Middle scene.", offset_body)
        self.assertIn("[02:05.0] Speaker: Hello.", offset_body)
        self.assertIn('[02:01.0] "Title"', offset_body)

    def test_format_timestamp_rolls_over_minute_boundary(self):
        self.assertEqual(format_timestamp(59.96), "01:00.0")
        self.assertEqual(format_timestamp(3599.98), "01:00:00.0")
        self.assertEqual(format_timestamp(83.4, use_hours=True), "00:01:23.4")

    def test_offset_multiple_ranges_on_one_line(self):
        body = "## Segments\n[00:10.0-00:20.0] A, echoing [00:30.0-00:40.0] B."
        offset_body = offset_body_timestamps(body, 100.0)
        self.assertIn("[01:50.0-02:00.0] A, echoing [02:10.0-02:20.0] B.", offset_body)

    def test_offset_multiple_stamps_on_one_transcript_line(self):
        body = "## Transcript\n[00:05.0] Ann: hi. [00:07.0] Bob: hello."
        offset_body = offset_body_timestamps(body, 60.0)
        self.assertIn("[01:05.0] Ann: hi. [01:07.0] Bob: hello.", offset_body)


class TestChunkMerging(unittest.TestCase):
    def test_merge_chunk_bodies(self):
        c1 = """## Summary
Part 1 summary.

## Segments
[00:00.0-01:00.0] First scene.

## Transcript
(no speech)

## On-screen Text
[00:10.0] "Intro"

## Tags
action, intro
"""
        c2 = """## Summary
Part 2 summary.

## Segments
[01:00.0-02:00.0] Second scene.

## Transcript
[01:15.0] Bob: Done!

## On-screen Text
(none)

## Tags
action, climax
"""
        merged = merge_chunk_bodies([(c1, 0, 60), (c2, 60, 120)])
        self.assertIn("## Summary", merged)
        self.assertIn("Part 1 summary.", merged)
        self.assertIn("Part 2 summary.", merged)
        self.assertIn("[00:00.0-01:00.0] First scene.", merged)
        self.assertIn("[01:00.0-02:00.0] Second scene.", merged)
        self.assertIn("[01:15.0] Bob: Done!", merged)
        self.assertIn('[00:10.0] "Intro"', merged)
        self.assertIn("action, intro, climax", merged)

    def test_merge_inserts_gap_for_chunk_boundary(self):
        """When chunk A's last segment ends at 1:05 and chunk B starts at 4:00,
        the merged body must include a gap-fill segment naming 1:05-4:00. This
        is the SPEC §2.2 contiguity requirement; without the fill, a consumer
        querying 'what's at 2:00' has no answer.
        """
        c1 = """## Segments
[00:00.0-01:05.0] First part.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
tag1
"""
        c2 = """## Segments
[04:00.0-08:05.0] Second part, after a 3-minute boundary gap.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
tag2
"""
        merged = merge_chunk_bodies([(c1, 0, 240), (c2, 240, 480)])
        # The gap-fill segment should appear, with a recognizable marker.
        self.assertIn("chunk boundary gap", merged)
        # And it should sit between the two real segments in chronological order.
        gap_idx = merged.index("chunk boundary gap")
        first_idx = merged.index("First part.")
        second_idx = merged.index("Second part")
        self.assertLess(first_idx, gap_idx)
        self.assertLess(gap_idx, second_idx)

    def test_merge_handles_internal_gap_within_a_chunk(self):
        """Gap detection must work even when a single chunk's model output has
        its own internal gaps (e.g. the model skipped a beat). The fill is based
        on segment-timestamp contiguity, not on chunk boundaries alone.
        """
        c1 = """## Segments
[00:00.0-00:30.0] Early.
[01:00.0-02:00.0] Then a jump to 1:00.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
x
"""
        merged = merge_chunk_bodies([(c1, 0, 120)])
        # 30s-60s gap should be filled even with a single chunk
        self.assertIn("chunk boundary gap", merged)
        # Order preserved
        self.assertLess(merged.index("Early."), merged.index("chunk boundary gap"))
        self.assertLess(merged.index("chunk boundary gap"), merged.index("Then a jump"))

    def test_merge_no_fill_when_segments_abut(self):
        """Back-to-back chunks whose segments abut (chunk N ends at exactly the
        time chunk N+1's first segment starts) must NOT get a phantom gap fill.
        This is the common case for well-aligned chunked output.
        """
        c1 = """## Segments
[00:00.0-04:00.0] First four minutes.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
x
"""
        c2 = """## Segments
[04:00.0-08:00.0] Next four minutes.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
y
"""
        merged = merge_chunk_bodies([(c1, 0, 240), (c2, 240, 480)])
        self.assertNotIn("chunk boundary gap", merged)

    def test_merge_sorts_out_of_order_segments(self):
        """If a chunk produces segments that aren't strictly increasing (e.g.
        a model that re-orders a flashback), the merge should still produce a
        chronologically sorted Segments section, with any intra-gap filled.
        """
        c1 = """## Segments
[04:00.0-05:00.0] B segment (out of order).
[00:00.0-01:00.0] A segment.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
x
"""
        merged = merge_chunk_bodies([(c1, 0, 300)])
        # A should appear before B in the merged output
        a_idx = merged.index("A segment.")
        b_idx = merged.index("B segment")
        self.assertLess(a_idx, b_idx)


class TestVideoDataUrl(unittest.TestCase):
    def test_video_data_url(self):
        with temp_video(data=b"fake video bytes") as p:
            url = _video_data_url(p)
            self.assertTrue(url.startswith("data:video/mp4;base64,"))
            payload_b64 = url.split(",", 1)[1]
            self.assertEqual(base64.b64decode(payload_b64), b"fake video bytes")

    def test_mkv_mime(self):
        with temp_video(suffix=".mkv", data=b"fake mkv bytes") as p:
            url = _video_data_url(p)
            self.assertTrue(url.startswith("data:video/x-matroska;base64,"))


class TestOpenRouterGeneration(unittest.TestCase):
    @patch("requests.post")
    def test_openrouter_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": SAMPLE_BODY,
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
        }
        mock_post.return_value = mock_resp

        with temp_video() as p:
            usage = {}
            body = describe_video(
                str(p),
                provider="openrouter",
                model="google/gemini-2.5-flash",
                api_key="sk-or-test",
                usage_out=usage,
            )
            self.assertIn("## Summary", body)
            self.assertIn("## Segments", body)
            self.assertEqual(usage.get("prompt_tokens"), 120)
            self.assertEqual(usage.get("output_tokens"), 80)

    @patch("requests.post")
    def test_generate_sidecar_with_cost_and_tokens(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": SAMPLE_BODY}}],
            "usage": {"prompt_tokens": 10000, "completion_tokens": 500},
        }
        mock_post.return_value = mock_resp

        with temp_video() as p:
            sc = generate_sidecar(
                str(p),
                provider="openrouter",
                model="google/gemini-2.5-flash",
                api_key="sk-or-test",
            )
            self.assertIsInstance(sc, Sidecar)
            self.assertEqual(sc.header.get("generator"), "google/gemini-2.5-flash")
            self.assertEqual(sc.header.get("video"), p.name)
            self.assertEqual(sc.header.get("prompt_tokens"), "10000")
            self.assertEqual(sc.header.get("output_tokens"), "500")
            self.assertIn("cost", sc.header)
            self.assertTrue(sc.header["cost"].startswith("$"))

    @patch("requests.post")
    def test_mode_screencast_prompt_passed(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": SAMPLE_BODY}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_post.return_value = mock_resp

        with temp_video() as p:
            sc = generate_sidecar(
                str(p),
                provider="openrouter",
                model="google/gemini-2.5-flash",
                api_key="sk-or-test",
                mode="screencast",
            )
            self.assertEqual(sc.header.get("mode"), "screencast")
            call_payload = mock_post.call_args[1]["json"]
            prompt_sent = call_payload["messages"][0]["content"][0]["text"]
            self.assertIn("Environment & Tools", prompt_sent)

    @patch("requests.post")
    def test_mode_meeting_prompt_passed(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": SAMPLE_BODY}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_post.return_value = mock_resp

        with temp_video() as p:
            sc = generate_sidecar(
                str(p),
                provider="openrouter",
                model="google/gemini-2.5-flash",
                api_key="sk-or-test",
                mode="meeting",
            )
            self.assertEqual(sc.header.get("mode"), "meeting")
            call_payload = mock_post.call_args[1]["json"]
            prompt_sent = call_payload["messages"][0]["content"][0]["text"]
            self.assertIn("Action Items & Decisions", prompt_sent)


if __name__ == "__main__":
    unittest.main()
