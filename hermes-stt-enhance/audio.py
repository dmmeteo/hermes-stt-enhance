"""Audio helpers: ffmpeg/ffprobe preprocessing, chunk planning, transcript merge.

Only the parts the STT backends actually need: speed-adjusted 16 kHz mono WAV
preparation, duration probing, overlapping chunk windows for long audio, and a
conservative merge of the per-chunk transcripts.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
FFMPEG_TIMEOUT_SECONDS = 600
FFPROBE_TIMEOUT_SECONDS = 30

# ffmpeg's atempo filter is only well behaved within this factor range, so
# larger changes are expressed as a chain of factors.
_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0

_WORD_SPLIT = re.compile(r"\s+")
_WORD_STRIP = re.compile(r"^\W+|\W+$", re.UNICODE)


class AudioError(RuntimeError):
    """Raised when audio cannot be probed, converted or read."""


def find_binary(name: str) -> Optional[str]:
    """Locate a binary, preferring Hermes' own lookup (Homebrew prefixes etc.)."""
    try:
        from tools.transcription_tools import _find_binary

        found = _find_binary(name)
        if found:
            return found
    except Exception:
        pass
    return shutil.which(name)


def atempo_filter(speed: float) -> str:
    """Return an ``atempo`` filter chain for ``speed``."""
    if speed <= 0:
        raise AudioError(f"Invalid audio_speed: {speed}")
    factors: List[float] = []
    remaining = float(speed)
    while remaining > _ATEMPO_MAX:
        factors.append(_ATEMPO_MAX)
        remaining /= _ATEMPO_MAX
    while remaining < _ATEMPO_MIN:
        factors.append(_ATEMPO_MIN)
        remaining /= _ATEMPO_MIN
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def wave_info(path: str) -> Optional[Tuple[int, int, int]]:
    """Return ``(channels, sample_width, sample_rate)`` for a RIFF WAV file."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
    except Exception:
        return None


def is_pcm16_mono(path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> bool:
    info = wave_info(path)
    return info == (1, 2, sample_rate)


def probe_duration_seconds(path: str) -> Optional[float]:
    """Duration in seconds via the wave header, falling back to ffprobe."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate > 0:
                return handle.getnframes() / float(rate)
    except Exception:
        pass

    ffprobe = find_binary("ffprobe")
    if not ffprobe:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=FFPROBE_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL, check=True,
        )
        duration = float((completed.stdout or "").strip())
    except Exception as exc:
        logger.debug("ffprobe could not read the duration of %s: %s", path, exc)
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def prepare_wav(
    file_path: str,
    work_dir: str,
    *,
    speed: float = 1.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> str:
    """Return a 16-bit mono WAV at ``sample_rate``, time-stretched by ``speed``.

    Returns ``file_path`` unchanged when it already satisfies the requirement
    and no speed change was requested, so the common case stays ffmpeg-free.
    """
    if abs(speed - 1.0) < 1e-6 and is_pcm16_mono(file_path, sample_rate):
        return file_path

    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        raise AudioError(
            "ffmpeg is required to prepare audio for this backend "
            "(16 kHz mono WAV / audio_speed) but was not found on PATH."
        )

    target = os.path.join(work_dir, f"{Path(file_path).stem}.prepared.wav")
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(file_path)]
    if abs(speed - 1.0) >= 1e-6:
        command += ["-filter:a", atempo_filter(speed)]
    command += ["-vn", "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", target]

    try:
        subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=FFMPEG_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffmpeg timed out preparing {Path(file_path).name}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise AudioError(f"ffmpeg failed preparing {Path(file_path).name}: {details}") from exc

    if not os.path.exists(target) or os.path.getsize(target) == 0:
        raise AudioError(f"ffmpeg produced no audio for {Path(file_path).name}")
    return target


def plan_chunks(
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> List[Tuple[float, float]]:
    """Return ``(start, length)`` windows covering ``duration_seconds``.

    Windows advance by ``chunk_seconds - overlap_seconds`` so word boundaries
    are never cut in a way that loses text; the last window always reaches the
    end of the audio.
    """
    if duration_seconds <= 0:
        return []
    step = max(chunk_seconds - overlap_seconds, chunk_seconds / 2.0)
    chunks: List[Tuple[float, float]] = []
    start = 0.0
    while True:
        remaining = duration_seconds - start
        if remaining <= chunk_seconds:
            # Fold a sub-overlap crumb into the previous window instead of
            # decoding a fragment that carries no new speech.
            if chunks and remaining < max(1.0, overlap_seconds):
                prev_start, _ = chunks[-1]
                chunks[-1] = (prev_start, duration_seconds - prev_start)
            else:
                chunks.append((start, remaining))
            return chunks
        chunks.append((start, chunk_seconds))
        start += step


def read_wave_window(
    path: str,
    start_seconds: float = 0.0,
    length_seconds: Optional[float] = None,
) -> Tuple[Sequence[float], int]:
    """Read a mono 16-bit WAV window as float samples in ``[-1, 1)``.

    Uses numpy when available (sherpa-onnx ships it) and falls back to the
    stdlib so callers without numpy still work.
    """
    try:
        handle = wave.open(str(path), "rb")
    except Exception as exc:
        raise AudioError(f"Cannot read WAV file {path}: {exc}") from exc

    with handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        if channels != 1 or width != 2:
            raise AudioError(
                f"Expected mono 16-bit PCM WAV, got channels={channels} sample_width={width}"
            )
        total = handle.getnframes()
        start_frame = min(max(0, int(start_seconds * rate)), total)
        frames = total - start_frame if length_seconds is None else int(length_seconds * rate)
        frames = max(0, min(frames, total - start_frame))
        handle.setpos(start_frame)
        raw = handle.readframes(frames)

    try:
        import numpy as np

        samples: Sequence[float] = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except ImportError:  # pragma: no cover - numpy ships with sherpa-onnx
        import array

        pcm = array.array("h")
        pcm.frombytes(raw)
        samples = [value / 32768.0 for value in pcm]
    return samples, rate


def _normalized_words(words: Sequence[str]) -> List[str]:
    """Comparison keys, one per input token so indexes stay aligned."""
    return [_WORD_STRIP.sub("", word).lower() for word in words]


def merge_transcripts(parts: Sequence[str], max_overlap_words: int = 12) -> str:
    """Join chunk transcripts, dropping text duplicated by the chunk overlap.

    Consecutive windows are cut from overlapping audio, so words that end one
    chunk and open the next are the same speech decoded twice. The longest such
    repeated run — down to a single word, which is the common case for a short
    overlap — is dropped once, comparing words case- and punctuation-insensitively.
    Only the seam is examined: repetitions anywhere else inside a chunk are the
    speaker's own and are always kept.
    """
    merged: List[str] = []
    for part in parts:
        words = _WORD_SPLIT.split((part or "").strip())
        words = [word for word in words if word]
        if not words:
            continue
        if not merged:
            merged = words
            continue
        window = min(max_overlap_words, len(merged), len(words))
        tail = _normalized_words(merged[-window:])
        head = _normalized_words(words[:window])
        drop = 0
        for size in range(window, 0, -1):
            # ``any`` keeps a run of tokens that normalize to nothing (stray
            # punctuation) from matching everything.
            if tail[-size:] == head[:size] and any(tail[-size:]):
                drop = size
                break
        merged.extend(words[drop:])
    return " ".join(merged).strip()
