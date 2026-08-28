"""Unit tests for the local provider. No network, no ffmpeg, no model."""

import pytest

from cdaf import local
from cdaf.generate import PROVIDERS, generate_sidecar


# ------------------------------------------------------------------- shots

def test_build_shots_splits_on_cuts():
    assert local.build_shots(12.0, [3.0, 6.0, 9.0]) == [
        (0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0)
    ]


def test_build_shots_no_cuts_is_one_shot():
    assert local.build_shots(4.0, []) == [(0.0, 4.0)]


def test_build_shots_absorbs_slivers():
    # A cut 0.1s before the end must not produce a 0.1s segment.
    shots = local.build_shots(10.0, [5.0, 9.95])
    assert shots == [(0.0, 5.0), (5.0, 10.0)]


def test_build_shots_is_contiguous_and_covers_duration():
    shots = local.build_shots(20.0, [1.0, 4.5, 11.0, 19.0])
    assert shots[0][0] == 0.0
    assert shots[-1][1] == 20.0
    for (_, end), (start, _) in zip(shots, shots[1:]):
        assert end == start


def test_coarsen_merges_toward_target():
    fine = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 12.0)]
    coarse = local.coarsen(fine, target=8.0)
    assert coarse[0][0] == 0.0
    assert coarse[-1][1] == 12.0
    assert len(coarse) < len(fine)


def test_sample_times_stay_inside_the_shot():
    for n in (1, 2, 3):
        for t in local._sample_times(10.0, 14.0, n):
            assert 10.0 < t < 14.0


# ------------------------------------------------------------- json recovery

def test_parse_json_tolerates_code_fences():
    assert local._parse_json('```json\n{"description": "ok"}\n```')["description"] == "ok"


def test_parse_json_tolerates_trailing_prose():
    assert local._parse_json('{"description": "ok"} Hope that helps!')["description"] == "ok"


def test_parse_json_raises_without_an_object():
    with pytest.raises(ValueError):
        local._parse_json("no json here")


def test_salvage_recovers_a_truncated_description():
    """A description cut off by max_tokens is still usable; only the brace is missing."""
    truncated = '{"description": "A hand opens a drawer of tools. In the second frame it is'
    assert local._salvage(truncated)["description"] == "A hand opens a drawer of tools."


def test_salvage_returns_empty_when_there_is_nothing_to_recover():
    assert local._salvage('{"setting": "kitchen"}') == {}


# ------------------------------------------------------- transcript guard

@pytest.mark.parametrize("reply", [
    "I am a large language model, trained by Google.",
    "There is no speech in this audio.",
    "I cannot transcribe this file.",
    "Sorry, the audio appears to be silent.",
    "Unfortunately, the audio contains no discernible speech.",
    "It seems the track is silent.",
    "Please provide an audio file containing speech.",
    "",
    "   ",
])
def test_non_transcript_replies_are_recognised(reply):
    """Handed a silent track, a model answers as a chatbot rather than declining.

    Such a reply must never reach the sidecar looking like narration.
    """
    assert local._is_not_a_transcript(reply)


@pytest.mark.parametrize("line", [
    "Funny how the things you put off always seem bigger than they are.",
    # Real speech that the earlier substring-anywhere guard discarded outright.
    "Sorry, I'm late - traffic was awful.",
    "I cannot believe how well this turned out.",
    "There is no better way to start the morning.",
    "I can't wait to show you what we built.",
    "This bread is trained by decades of tradition.",
    "The audio guide said to turn left at the chapel.",
    "It seems simple, but there is no shortcut to a good loaf.",
])
def test_real_narration_is_not_mistaken_for_a_refusal(line):
    """A speaker can say \"sorry\" or \"I cannot\" mid-sentence.

    Only a reply that *opens* with a refusal phrase is treated as one, so ordinary
    narration is not silently dropped.
    """
    assert not local._is_not_a_transcript(line)


def test_leading_punctuation_does_not_defeat_the_anchor():
    """The opening check survives quotes, dashes and stray whitespace."""
    assert local._is_not_a_transcript('"Sorry, the audio is silent."')
    assert local._is_not_a_transcript("  - Unfortunately, there is no speech here.")
    assert local._is_not_a_transcript("“It seems this recording is empty.")


def test_a_generic_apology_is_left_alone():
    """Without a reference to the recording, a refusal-shaped line is just speech.

    The measured-silence short-circuit in _transcribe is what protects a silent
    track; this guard only has to catch replies that are plainly about the audio.
    """
    assert not local._is_not_a_transcript("Sorry, I cannot help with that.")


# ------------------------------------------------------------ header extras

def test_header_extras_record_provenance():
    extras = local.local_header_extras(continuity=True, transcribed=True, threshold=0.15)
    assert extras["x-shot-source"] == "ffmpeg-scene-detect@0.15"
    assert extras["x-transcript-timing"] == "measured-rms"


def test_header_extras_mark_an_untranscribed_clip():
    extras = local.local_header_extras(continuity=False, transcribed=False)
    assert extras["x-transcript-timing"] == "none"
    assert extras["x-continuity-pass"] == "no"


def test_header_extras_use_only_reserved_x_keys():
    """SPEC.md reserves the `x-` prefix for producer extensions."""
    assert all(k.startswith("x-") for k in
               local.local_header_extras(continuity=True, transcribed=True))


# ---------------------------------------------------------------- provider

def test_known_providers():
    assert PROVIDERS == ("auto", "gemini", "openrouter", "local")


def test_unknown_provider_is_rejected(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    with pytest.raises(ValueError, match="provider must be one of"):
        generate_sidecar(video, provider="nope")


def test_detail_must_be_known(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    with pytest.raises(ValueError, match="detail must be one of"):
        local.describe_video_local(video, detail="exhaustive")
