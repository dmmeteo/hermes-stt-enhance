"""Audio helpers: ffmpeg preprocessing, duration probing, chunking, merging."""

from __future__ import annotations

import subprocess
import sys
import wave

import pytest

from conftest import requires_ffmpeg


# ---------------------------------------------------------------------------
# atempo filter chains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "speed,expected",
    [
        (1.0, "atempo=1.000000"),
        (1.25, "atempo=1.250000"),
        (2.0, "atempo=2.000000"),
        # Above 2.0x ffmpeg needs a chain: 2.0 * 1.25 == 2.5.
        (2.5, "atempo=2.000000,atempo=1.250000"),
        # Below 0.5x likewise: 0.5 * 0.8 == 0.4.
        (0.4, "atempo=0.500000,atempo=0.800000"),
    ],
)
def test_atempo_filter_chains_out_of_range_factors(audio_mod, speed, expected):
    assert audio_mod.atempo_filter(speed) == expected


def test_atempo_factors_multiply_back_to_the_requested_speed(audio_mod):
    product = 1.0
    for part in audio_mod.atempo_filter(2.8).split(","):
        product *= float(part.split("=")[1])
    assert product == pytest.approx(2.8, rel=1e-4)


@pytest.mark.parametrize("speed", [0, -1.5])
def test_atempo_filter_rejects_non_positive_speed(audio_mod, speed):
    with pytest.raises(audio_mod.AudioError):
        audio_mod.atempo_filter(speed)


# ---------------------------------------------------------------------------
# WAV inspection
# ---------------------------------------------------------------------------


def test_wave_info_and_pcm16_mono_detection(audio_mod, make_wav, tmp_path):
    path = make_wav(seconds=0.5)
    assert audio_mod.wave_info(str(path)) == (1, 2, 16000)
    assert audio_mod.is_pcm16_mono(str(path)) is True
    # A different target rate is not a match.
    assert audio_mod.is_pcm16_mono(str(path), 44100) is False

    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00\x00\x00" * 100)
    assert audio_mod.is_pcm16_mono(str(stereo)) is False


def test_wave_info_returns_none_for_non_wav(audio_mod, tmp_path):
    blob = tmp_path / "audio.ogg"
    blob.write_bytes(b"not a wav file")
    assert audio_mod.wave_info(str(blob)) is None
    assert audio_mod.is_pcm16_mono(str(blob)) is False


# ---------------------------------------------------------------------------
# Duration probing
# ---------------------------------------------------------------------------


def test_probe_duration_reads_the_wave_header(audio_mod, make_wav):
    path = make_wav(seconds=2.5)
    assert audio_mod.probe_duration_seconds(str(path)) == pytest.approx(2.5)


@requires_ffmpeg
def test_probe_duration_falls_back_to_ffprobe(audio_mod, make_wav, tmp_path):
    """Non-WAV input has no readable header, so ffprobe answers instead."""
    source = make_wav(seconds=1.5)
    encoded = tmp_path / "audio.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), str(encoded)],
        check=True, capture_output=True,
    )

    assert audio_mod.wave_info(str(encoded)) is None
    assert audio_mod.probe_duration_seconds(str(encoded)) == pytest.approx(1.5, abs=0.2)


def test_probe_duration_returns_none_without_ffprobe(audio_mod, monkeypatch, tmp_path):
    blob = tmp_path / "audio.m4a"
    blob.write_bytes(b"not audio")
    monkeypatch.setattr(audio_mod, "find_binary", lambda name: None)

    assert audio_mod.probe_duration_seconds(str(blob)) is None


def test_probe_duration_returns_none_when_ffprobe_fails(audio_mod, monkeypatch, tmp_path):
    blob = tmp_path / "audio.m4a"
    blob.write_bytes(b"not audio")
    monkeypatch.setattr(audio_mod, "find_binary", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        audio_mod.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "ffprobe")),
    )

    assert audio_mod.probe_duration_seconds(str(blob)) is None


def test_find_binary_prefers_hermes_lookup(audio_mod, monkeypatch):
    monkeypatch.setattr(
        sys.modules["tools.transcription_tools"], "_find_binary",
        lambda name: f"/opt/homebrew/bin/{name}",
    )
    assert audio_mod.find_binary("ffmpeg") == "/opt/homebrew/bin/ffmpeg"


def test_find_binary_falls_back_to_path(audio_mod):
    # The conftest stub returns None, mirroring a machine without Homebrew.
    assert audio_mod.find_binary("definitely-not-a-real-binary") is None


# ---------------------------------------------------------------------------
# WAV preparation
# ---------------------------------------------------------------------------


def test_prepare_wav_passes_through_matching_input(audio_mod, make_wav, tmp_path):
    """The common case (16 kHz mono, no speed change) must not call ffmpeg."""
    path = make_wav(seconds=1.0)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    assert audio_mod.prepare_wav(str(path), str(work_dir)) == str(path)
    assert list(work_dir.iterdir()) == []


@requires_ffmpeg
def test_prepare_wav_resamples_and_downmixes(audio_mod, tmp_path):
    source = tmp_path / "stereo-44k.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x10\x00\x20\x00" * 44100)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    prepared = audio_mod.prepare_wav(str(source), str(work_dir))

    assert prepared != str(source)
    assert audio_mod.wave_info(prepared) == (1, 2, 16000)
    assert audio_mod.probe_duration_seconds(prepared) == pytest.approx(1.0, abs=0.05)


@requires_ffmpeg
def test_prepare_wav_speeds_audio_up(audio_mod, make_wav, tmp_path):
    source = make_wav(seconds=4.0)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    prepared = audio_mod.prepare_wav(str(source), str(work_dir), speed=2.0)

    assert prepared != str(source)
    assert audio_mod.probe_duration_seconds(prepared) == pytest.approx(2.0, abs=0.2)


def test_prepare_wav_requires_ffmpeg(audio_mod, make_wav, tmp_path, monkeypatch):
    monkeypatch.setattr(audio_mod, "find_binary", lambda name: None)

    with pytest.raises(audio_mod.AudioError, match="ffmpeg is required"):
        audio_mod.prepare_wav(str(make_wav()), str(tmp_path), speed=1.25)


def test_prepare_wav_reports_ffmpeg_failure(audio_mod, make_wav, tmp_path, monkeypatch):
    monkeypatch.setattr(audio_mod, "find_binary", lambda name: "/usr/bin/ffmpeg")

    def _fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="Invalid data found")

    monkeypatch.setattr(audio_mod.subprocess, "run", _fail)

    with pytest.raises(audio_mod.AudioError, match="Invalid data found"):
        audio_mod.prepare_wav(str(make_wav()), str(tmp_path), speed=1.25)


def test_prepare_wav_reports_ffmpeg_timeout(audio_mod, make_wav, tmp_path, monkeypatch):
    monkeypatch.setattr(audio_mod, "find_binary", lambda name: "/usr/bin/ffmpeg")

    def _timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 1))

    monkeypatch.setattr(audio_mod.subprocess, "run", _timeout)

    with pytest.raises(audio_mod.AudioError, match="timed out"):
        audio_mod.prepare_wav(str(make_wav()), str(tmp_path), speed=1.25)


def test_prepare_wav_reports_empty_output(audio_mod, make_wav, tmp_path, monkeypatch):
    monkeypatch.setattr(audio_mod, "find_binary", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio_mod.subprocess, "run", lambda command, **kwargs: None)

    with pytest.raises(audio_mod.AudioError, match="produced no audio"):
        audio_mod.prepare_wav(str(make_wav()), str(tmp_path), speed=1.25)


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------


def test_plan_chunks_returns_one_window_for_short_audio(audio_mod):
    assert audio_mod.plan_chunks(30.0, 45.0, 2.0) == [(0.0, 30.0)]


def test_plan_chunks_returns_nothing_for_empty_audio(audio_mod):
    assert audio_mod.plan_chunks(0.0, 45.0, 2.0) == []
    assert audio_mod.plan_chunks(-5.0, 45.0, 2.0) == []


def test_plan_chunks_windows_overlap_and_cover_the_audio(audio_mod):
    chunks = audio_mod.plan_chunks(120.0, 45.0, 2.0)

    assert chunks[0] == (0.0, 45.0)
    # Each window starts one step (chunk - overlap) after the previous one.
    starts = [start for start, _ in chunks]
    assert starts == [0.0, 43.0, 86.0]
    # Full coverage, and no window runs past the end of the audio.
    assert chunks[-1][0] + chunks[-1][1] == pytest.approx(120.0)
    for index in range(1, len(chunks)):
        prev_end = chunks[index - 1][0] + chunks[index - 1][1]
        assert starts[index] < prev_end, "windows must overlap, not leave a gap"


def test_plan_chunks_folds_a_trailing_crumb_into_the_previous_window(audio_mod):
    """A sub-overlap tail carries no new speech, so it must not be its own chunk."""
    chunks = audio_mod.plan_chunks(39.75, 20.0, 0.5)

    assert len(chunks) == 2
    assert chunks[-1] == (19.5, pytest.approx(20.25))
    assert chunks[-1][0] + chunks[-1][1] == pytest.approx(39.75)


def test_plan_chunks_progresses_when_overlap_equals_the_window(audio_mod):
    """A degenerate overlap must still advance (step is floored at half a window)."""
    chunks = audio_mod.plan_chunks(100.0, 20.0, 20.0)

    assert [start for start, _ in chunks] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    assert chunks[-1][0] + chunks[-1][1] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Reading sample windows
# ---------------------------------------------------------------------------


def test_read_wave_window_reads_the_whole_file(audio_mod, make_wav):
    samples, rate = audio_mod.read_wave_window(str(make_wav(seconds=1.0, amplitude=16384)))

    assert rate == 16000
    assert len(samples) == 16000
    assert float(samples[0]) == pytest.approx(0.5, abs=1e-4)


def test_read_wave_window_slices_a_window(audio_mod, make_wav):
    path = make_wav(seconds=3.0)

    samples, rate = audio_mod.read_wave_window(str(path), 1.0, 0.5)

    assert rate == 16000
    assert len(samples) == 8000


def test_read_wave_window_clamps_out_of_range_requests(audio_mod, make_wav):
    path = make_wav(seconds=1.0)

    tail, _ = audio_mod.read_wave_window(str(path), 0.75, 10.0)
    assert len(tail) == 4000

    past_end, _ = audio_mod.read_wave_window(str(path), 5.0, 1.0)
    assert len(past_end) == 0

    negative_start, _ = audio_mod.read_wave_window(str(path), -3.0, 0.25)
    assert len(negative_start) == 4000


def test_read_wave_window_rejects_non_mono_16bit(audio_mod, tmp_path):
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00\x00\x00" * 100)

    with pytest.raises(audio_mod.AudioError, match="mono 16-bit"):
        audio_mod.read_wave_window(str(stereo))


def test_read_wave_window_rejects_unreadable_files(audio_mod, tmp_path):
    with pytest.raises(audio_mod.AudioError, match="Cannot read WAV file"):
        audio_mod.read_wave_window(str(tmp_path / "missing.wav"))


def test_read_wave_window_without_numpy(audio_mod, make_wav, monkeypatch):
    """The stdlib fallback must produce the same floats as the numpy path."""
    path = make_wav(seconds=0.1, amplitude=16384)
    expected, _ = audio_mod.read_wave_window(str(path))

    monkeypatch.setitem(sys.modules, "numpy", None)
    samples, rate = audio_mod.read_wave_window(str(path))

    assert rate == 16000
    assert isinstance(samples, list)
    assert len(samples) == len(expected)
    assert samples[0] == pytest.approx(float(expected[0]))


# ---------------------------------------------------------------------------
# Transcript merging
# ---------------------------------------------------------------------------


def test_merge_transcripts_drops_overlapping_text(audio_mod):
    merged = audio_mod.merge_transcripts(
        ["deploy the staging cluster and then", "and then restart the gateway"]
    )

    assert merged == "deploy the staging cluster and then restart the gateway"


def test_merge_transcripts_ignores_punctuation_and_casing_in_the_overlap(audio_mod):
    merged = audio_mod.merge_transcripts(
        ["we shipped the fix, then rolled back", "Then rolled back the schema"]
    )

    assert merged == "we shipped the fix, then rolled back the schema"


def test_merge_transcripts_drops_a_single_word_overlap(audio_mod):
    """A short overlap often shares just one word — it is still decoded twice."""
    merged = audio_mod.merge_transcripts(["please restart the", "the gateway now"])

    assert merged == "please restart the gateway now"


def test_merge_transcripts_keeps_repetitions_away_from_the_seam(audio_mod):
    """Only the chunk boundary is deduplicated; the speaker's repeats survive."""
    parts = ["we we need to go go now", "then then we left"]

    assert audio_mod.merge_transcripts(parts) == "we we need to go go now then then we left"
    # A single chunk is never rewritten at all.
    assert audio_mod.merge_transcripts(["the the gateway is down"]) == "the the gateway is down"


def test_merge_transcripts_respects_the_overlap_window(audio_mod):
    parts = ["one two three four five six", "three four five six seven"]

    # A window too small to see the repeated run keeps both copies…
    assert audio_mod.merge_transcripts(parts, max_overlap_words=2) == (
        "one two three four five six three four five six seven"
    )
    # …while a window that covers it collapses the duplicate.
    assert audio_mod.merge_transcripts(parts, max_overlap_words=4) == (
        "one two three four five six seven"
    )


def test_merge_transcripts_skips_empty_chunks(audio_mod):
    assert audio_mod.merge_transcripts(["", "  ", "hello there", "", None]) == "hello there"
    assert audio_mod.merge_transcripts([]) == ""
    assert audio_mod.merge_transcripts(["", ""]) == ""


def test_merge_transcripts_does_not_merge_on_punctuation_only_tokens(audio_mod):
    """Tokens that normalize to nothing must never count as an overlap match."""
    merged = audio_mod.merge_transcripts(["hello - -", "- - world"])

    assert merged == "hello - - - - world"
