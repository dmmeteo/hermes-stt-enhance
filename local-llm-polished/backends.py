"""ASR backends for the ``local_llm_polished`` provider.

Two backends share one interface:

* ``faster_whisper`` — delegates to Hermes' own local faster-whisper path so the
  default configuration behaves exactly like the built-in ``local`` provider.
* ``parakeet`` — sherpa-onnx offline transducer (NVIDIA Parakeet TDT 0.6B v3
  INT8). Decodes whole waveforms in memory, so it opts into chunking and a
  process-wide concurrency limit.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from . import audio
from .config import (
    BACKEND_FASTER_WHISPER,
    BACKEND_PARAKEET,
    ConfigError,
    ParakeetConfig,
    Settings,
)

logger = logging.getLogger(__name__)

# A queued voice message should not wait forever behind a wedged transcription.
CONCURRENCY_WAIT_SECONDS = 600.0

_recognizer_cache: Dict[Tuple[Any, ...], Any] = {}
_recognizer_lock = threading.Lock()

_semaphores: Dict[int, threading.BoundedSemaphore] = {}
_semaphores_lock = threading.Lock()


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a transcript."""


@contextmanager
def concurrency_guard(max_concurrency: int):
    """Limit concurrent decodes in this process (Parakeet saturates every core)."""
    if max_concurrency <= 0:
        yield
        return
    with _semaphores_lock:
        semaphore = _semaphores.get(max_concurrency)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max_concurrency)
            _semaphores[max_concurrency] = semaphore
    if not semaphore.acquire(timeout=CONCURRENCY_WAIT_SECONDS):
        raise BackendError(
            f"Timed out after {CONCURRENCY_WAIT_SECONDS:.0f}s waiting for a free "
            f"transcription slot (max_concurrency={max_concurrency})."
        )
    try:
        yield
    finally:
        semaphore.release()


class SttBackend:
    """Common backend surface used by the provider pipeline."""

    name = ""
    # True when the backend must be fed a 16 kHz mono PCM WAV.
    requires_wav = False
    # True when the backend decodes an entire waveform at once and therefore
    # benefits from the chunking guard.
    supports_chunking = False
    max_concurrency = 0

    def transcribe_file(self, file_path: str, language: Optional[str]) -> str:
        raise NotImplementedError

    def transcribe_samples(self, samples: Sequence[float], sample_rate: int) -> str:
        raise NotImplementedError


class FasterWhisperBackend(SttBackend):
    """Hermes' built-in local faster-whisper path."""

    name = BACKEND_FASTER_WHISPER
    # faster-whisper does its own windowed decoding and streams segments, so it
    # neither needs pre-converted WAV input nor external chunking.
    requires_wav = False
    supports_chunking = False

    def __init__(self, model: str) -> None:
        self.model = model

    def transcribe_file(self, file_path: str, language: Optional[str]) -> str:
        try:
            from tools.transcription_tools import _transcribe_local

            result = _transcribe_local(file_path, self.model)
        except Exception as exc:
            raise BackendError(f"Local transcription failed: {exc}") from exc

        if not isinstance(result, dict):
            raise BackendError("Local transcription returned an invalid result.")
        if not result.get("success"):
            raise BackendError(str(result.get("error") or "Local transcription failed."))
        return str(result.get("transcript") or "")


class ParakeetBackend(SttBackend):
    """sherpa-onnx offline transducer backend."""

    name = BACKEND_PARAKEET
    requires_wav = True
    supports_chunking = True

    def __init__(self, cfg: ParakeetConfig) -> None:
        self.cfg = cfg
        self.max_concurrency = cfg.max_concurrency

    def _recognizer(self) -> Any:
        try:
            files = self.cfg.resolve_files()
        except ConfigError as exc:
            raise BackendError(str(exc)) from exc

        key = self.cfg.cache_key(files)
        with _recognizer_lock:
            cached = _recognizer_cache.get(key)
            if cached is not None:
                return cached

            try:
                import sherpa_onnx
            except ImportError as exc:
                raise BackendError(
                    "The parakeet backend needs sherpa-onnx. Install it with "
                    "'pip install sherpa-onnx numpy'."
                ) from exc

            logger.info(
                "Loading Parakeet recognizer from %s (threads=%d, %s)",
                Path(files["encoder"]).parent, self.cfg.num_threads, self.cfg.model_type,
            )
            try:
                recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=files["encoder"],
                    decoder=files["decoder"],
                    joiner=files["joiner"],
                    tokens=files["tokens"],
                    num_threads=self.cfg.num_threads,
                    sample_rate=self.cfg.sample_rate,
                    feature_dim=self.cfg.feature_dim,
                    decoding_method=self.cfg.decoding_method,
                    provider=self.cfg.onnx_provider,
                    model_type=self.cfg.model_type,
                )
            except Exception as exc:
                raise BackendError(f"Failed to load the Parakeet model: {exc}") from exc

            _recognizer_cache[key] = recognizer
            return recognizer

    def transcribe_samples(self, samples: Sequence[float], sample_rate: int) -> str:
        if len(samples) == 0:
            return ""
        recognizer = self._recognizer()
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            return str(stream.result.text or "").strip()
        except Exception as exc:
            raise BackendError(f"Parakeet decoding failed: {exc}") from exc

    def transcribe_file(self, file_path: str, language: Optional[str]) -> str:
        samples, sample_rate = audio.read_wave_window(file_path)
        return self.transcribe_samples(samples, sample_rate)


def build_backend(settings: Settings) -> SttBackend:
    if settings.backend == BACKEND_PARAKEET:
        return ParakeetBackend(settings.parakeet)
    return FasterWhisperBackend(settings.model)


def reset_caches() -> None:
    """Drop cached recognizers and semaphores (used by tests)."""
    with _recognizer_lock:
        _recognizer_cache.clear()
    with _semaphores_lock:
        _semaphores.clear()
