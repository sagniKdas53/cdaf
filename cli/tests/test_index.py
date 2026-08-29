"""Tests for building and searching a library index."""

import json
from pathlib import Path

import pytest

from cdaf import Sidecar, hash_file, save
from cdaf.index import (
    INDEX_VERSION,
    build_index,
    default_index_path,
    load_index,
    search,
    write_index,
)

BODY = """## Summary
An aerial drone clip of a coastal highway at golden hour.

## Segments
[00:00.0-00:05.0] Wide establishing shot over the cliffs.
[00:05.0-00:10.0] Slow push-in toward the traffic below.

## Transcript
(no speech)

## On-screen Text
(none)

## Tags
drone, aerial, golden hour, Drone, sunset
"""

OTHER_BODY = """## Summary
A screencast of a terminal session.

## Segments
[00:00.0-00:04.0] The prompt appears and a command is typed.

## Transcript
[00:01.0] Speaker: let's run the build.

## On-screen Text
[00:02.0] "npm run build"

## Tags
screencast, terminal
"""


def _make_pair(directory: Path, stem: str, body: str) -> Path:
    """A fake video plus a fresh sidecar for it. Returns the video path."""
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / f"{stem}.mp4"
    video.write_bytes(f"fake-bytes-for-{stem}".encode() * 50)
    sc = Sidecar(
        header={
            "video": video.name,
            "sha256": hash_file(video),
            "bytes": str(video.stat().st_size),
            "duration": "00:00:31.500",
            "generator": "gemini-2.5-flash",
            "created": "2026-08-28T00:00:00Z",
        },
        body=body,
    )
    save(sc, directory / f"{stem}.cdaf")
    return video


@pytest.fixture
def library(tmp_path: Path) -> Path:
    _make_pair(tmp_path / "b-roll", "sunset-drone", BODY)
    _make_pair(tmp_path / "demos", "terminal-demo", OTHER_BODY)
    return tmp_path


class TestBuildIndex:
    def test_finds_sidecars_recursively(self, library: Path):
        index = build_index(library)
        assert index["version"] == INDEX_VERSION
        assert {e["video"] for e in index["entries"]} == {
            "sunset-drone.mp4",
            "terminal-demo.mp4",
        }

    def test_paths_are_relative_to_the_library_root(self, library: Path):
        entry = next(e for e in build_index(library)["entries"] if "sunset" in e["video"])
        assert entry["sidecar"] == "b-roll/sunset-drone.cdaf"
        assert entry["video_path"] == "b-roll/sunset-drone.mp4"

    def test_entries_are_ordered_deterministically(self, library: Path):
        first = [e["sidecar"] for e in build_index(library)["entries"]]
        second = [e["sidecar"] for e in build_index(library)["entries"]]
        assert first == second == sorted(first)

    def test_extracts_sections(self, library: Path):
        entry = next(e for e in build_index(library)["entries"] if "terminal" in e["video"])
        assert entry["summary"] == "A screencast of a terminal session."
        assert entry["segments"] == ["[00:00.0-00:04.0] The prompt appears and a command is typed."]
        assert "let's run the build" in entry["transcript"]
        assert '"npm run build"' in entry["on_screen_text"]

    def test_tags_deduplicate_case_insensitively_preserving_order(self, library: Path):
        entry = next(e for e in build_index(library)["entries"] if "sunset" in e["video"])
        assert entry["tags"] == ["drone", "aerial", "golden hour", "sunset"]

    def test_fresh_video_is_reported_fresh(self, library: Path):
        entry = next(e for e in build_index(library, verify=True)["entries"] if "sunset" in e["video"])
        assert entry["state"] == "fresh"

    def test_modified_video_is_reported_stale(self, library: Path):
        (library / "b-roll" / "sunset-drone.mp4").write_bytes(b"different length entirely")
        entry = next(e for e in build_index(library)["entries"] if "sunset" in e["video"])
        assert entry["state"] == "stale"

    def test_missing_video_is_reported_orphan(self, library: Path):
        (library / "b-roll" / "sunset-drone.mp4").unlink()
        entry = next(e for e in build_index(library)["entries"] if "sunset" in e["video"])
        assert entry["state"] == "orphan"
        assert entry["video_path"] is None

    def test_malformed_sidecar_is_recorded_not_raised(self, library: Path):
        (library / "broken.cdaf").write_text("this is not a sidecar")
        entry = next(e for e in build_index(library)["entries"] if e["sidecar"] == "broken.cdaf")
        assert entry["state"] == "invalid"
        assert entry["error"]

    def test_single_sidecar_path(self, library: Path):
        index = build_index(library / "b-roll" / "sunset-drone.cdaf")
        assert len(index["entries"]) == 1

    def test_empty_directory_yields_no_entries(self, tmp_path: Path):
        assert build_index(tmp_path)["entries"] == []

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            build_index(tmp_path / "nope")


class TestIndexRoundTrip:
    def test_write_then_load(self, library: Path, tmp_path: Path):
        out = write_index(build_index(library), tmp_path / "out" / "cdaf-index.json")
        assert out.is_file()
        assert len(load_index(out)["entries"]) == 2

    def test_index_is_valid_utf8_json(self, library: Path, tmp_path: Path):
        out = write_index(build_index(library), tmp_path / "i.json")
        json.loads(out.read_text(encoding="utf-8"))

    def test_unknown_version_is_rejected(self, tmp_path: Path):
        path = tmp_path / "i.json"
        path.write_text(json.dumps({"version": 99, "entries": []}))
        with pytest.raises(ValueError):
            load_index(path)

    def test_default_index_path(self, library: Path):
        assert default_index_path(library) == library / "cdaf-index.json"
        assert default_index_path(library / "b-roll" / "sunset-drone.cdaf") == (
            library / "b-roll" / "cdaf-index.json"
        )


class TestSearch:
    @pytest.fixture
    def index(self, library: Path) -> dict:
        return build_index(library)

    def test_matches_summary_text(self, index: dict):
        assert [e["video"] for e in search("coastal highway", index)] == ["sunset-drone.mp4"]

    def test_matches_tags(self, index: dict):
        assert [e["video"] for e in search("screencast", index)] == ["terminal-demo.mp4"]

    def test_matches_transcript_and_on_screen_text(self, index: dict):
        assert len(search("npm", index)) == 1
        assert len(search("build", index)) == 1

    def test_is_case_insensitive(self, index: dict):
        assert len(search("GOLDEN HOUR", index)) == 1

    def test_matches_partial_words(self, index: dict):
        assert len(search("termin", index)) == 1

    def test_all_terms_must_match(self, index: dict):
        assert search("drone terminal", index) == []
        assert len(search("drone aerial", index)) == 1

    def test_no_match_returns_empty(self, index: dict):
        assert search("zebra", index) == []

    def test_empty_query_returns_empty(self, index: dict):
        assert search("", index) == []
        assert search("   ", index) == []

    def test_limit_is_respected(self, index: dict):
        assert len(search("00", index, limit=1)) == 1

    def test_ranks_more_occurrences_first(self, library: Path):
        _make_pair(library / "extra", "drone-heavy", BODY.replace("## Tags", "## Tags\ndrone, drone"))
        results = search("drone", build_index(library))
        assert results[0]["video"] == "drone-heavy.mp4"

    def test_results_are_stable_across_runs(self, library: Path):
        index = build_index(library)
        assert [e["video"] for e in search("00", index)] == [
            e["video"] for e in search("00", index)
        ]
