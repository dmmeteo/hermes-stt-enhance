"""Local multi-backend STT provider for Hermes.

Registers ``local_llm_polished`` as a speech-to-text provider. Audio is
transcribed locally — by Hermes' own faster-whisper path (default) or by a
sherpa-onnx Parakeet V3 offline transducer — and the raw transcript then
optionally goes through a bounded LLM post-processing step before it is
returned to Hermes.

    audio → [ffmpeg speed/16 kHz mono] → local ASR (+chunking) → LLM cleanup → Hermes
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

from . import audio, post_processing
from .backends import (
    BackendError,
    SttBackend,
    build_backend,
    concurrency_guard,
)
from .config import (
    BACKEND_PARAKEET,
    PROVIDER_NAME,
    ConfigError,
    Settings,
    load_settings,
    load_stt_config,
)

logger = logging.getLogger(__name__)

# Roughly three words per second of speech, plus slack, bounded to keep the
# overlap comparison cheap and conservative.
_OVERLAP_WORDS_PER_SECOND = 3
_MIN_OVERLAP_WORDS = 2
_MAX_OVERLAP_WORDS = 24


class LocalLlmPolishedProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Local LLM Post-Processed STT"

    def list_models(self) -> List[Dict[str, Any]]:
        # Model ids apply to the faster-whisper backend; the parakeet backend is
        # selected via ``backend: parakeet`` and configured by model_path.
        return [
            {"id": "tiny", "display": "faster-whisper tiny"},
            {"id": "base", "display": "faster-whisper base"},
            {"id": "small", "display": "faster-whisper small"},
            {"id": "medium", "display": "faster-whisper medium"},
            {"id": "large-v3", "display": "faster-whisper large-v3"},
            {"id": "large-v3-turbo", "display": "faster-whisper large-v3-turbo"},
        ]

    def default_model(self) -> Optional[str]:
        return "base"

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local+LLM",
            "tag": "Local faster-whisper or Parakeet V3 STT plus optional LLM post-processing",
            "env_vars": [],
        }

    def is_available(self) -> bool:
        """Never raises — reports whether the configured backend can run."""
        try:
            settings = load_settings(load_stt_config())
            if settings.backend != BACKEND_PARAKEET:
                return True
            import importlib.util

            if importlib.util.find_spec("sherpa_onnx") is None:
                return False
            settings.parakeet.resolve_files()
            return True
        except Exception as exc:
            logger.debug("%s availability check failed: %s", PROVIDER_NAME, exc)
            return False

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def _target_sample_rate(self, settings: Settings) -> int:
        if settings.backend == BACKEND_PARAKEET:
            return settings.parakeet.sample_rate
        return audio.TARGET_SAMPLE_RATE

    def _transcribe_chunked(
        self,
        backend: SttBackend,
        wav_path: str,
        duration_seconds: float,
        settings: Settings,
    ) -> tuple[str, int]:
        chunking = settings.chunking
        chunks = audio.plan_chunks(duration_seconds, chunking.chunk_seconds, chunking.overlap_seconds)
        logger.info(
            "%s: decoding %.1fs of audio in %d chunks of %.0fs (overlap %.1fs)",
            PROVIDER_NAME, duration_seconds, len(chunks), chunking.chunk_seconds, chunking.overlap_seconds,
        )
        texts: List[str] = []
        for index, (start, length) in enumerate(chunks, start=1):
            samples, sample_rate = audio.read_wave_window(wav_path, start, length)
            texts.append(backend.transcribe_samples(samples, sample_rate))
            logger.debug("%s: chunk %d/%d decoded", PROVIDER_NAME, index, len(chunks))
        max_overlap_words = min(
            _MAX_OVERLAP_WORDS,
            max(_MIN_OVERLAP_WORDS, int(chunking.overlap_seconds * _OVERLAP_WORDS_PER_SECOND) + 2),
        )
        return audio.merge_transcripts(texts, max_overlap_words), len(chunks)

    def _run_asr(self, file_path: str, settings: Settings) -> tuple[str, int]:
        """Return ``(transcript, chunk_count)``; raises on failure."""
        backend = build_backend(settings)
        needs_wav = backend.requires_wav or abs(settings.audio_speed - 1.0) >= 1e-6

        with tempfile.TemporaryDirectory(prefix="hermes-stt-") as work_dir:
            wav_path = file_path
            if needs_wav:
                wav_path = audio.prepare_wav(
                    file_path,
                    work_dir,
                    speed=settings.audio_speed,
                    sample_rate=self._target_sample_rate(settings),
                )

            with concurrency_guard(backend.max_concurrency):
                if backend.supports_chunking and settings.chunking.enabled:
                    duration = audio.probe_duration_seconds(wav_path)
                    if duration is None:
                        logger.warning(
                            "%s: could not determine the duration of %s — decoding in one pass.",
                            PROVIDER_NAME, Path(file_path).name,
                        )
                    elif duration > settings.chunking.threshold_seconds:
                        return self._transcribe_chunked(backend, wav_path, duration, settings)
                return backend.transcribe_file(wav_path, settings.language), 1

    # ------------------------------------------------------------------
    # Provider entry point
    # ------------------------------------------------------------------

    def _error(self, message: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": False,
            "transcript": "",
            "provider": PROVIDER_NAME,
            "error": message,
        }
        if settings is not None:
            result["backend"] = settings.backend
        return result

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        try:
            settings = load_settings(load_stt_config(), model=model, language=language)
        except Exception as exc:
            logger.error("%s: invalid configuration: %s", PROVIDER_NAME, exc)
            return self._error(f"Invalid {PROVIDER_NAME} configuration: {exc}")

        try:
            raw, chunk_count = self._run_asr(file_path, settings)
        except (BackendError, audio.AudioError, ConfigError) as exc:
            logger.error("%s: %s backend failed: %s", PROVIDER_NAME, settings.backend, exc)
            return self._error(f"{settings.backend} transcription failed: {exc}", settings)
        except Exception as exc:  # defensive: the ABC forbids raising
            logger.error("%s: unexpected failure: %s", PROVIDER_NAME, exc, exc_info=True)
            return self._error(f"{settings.backend} transcription failed: {exc}", settings)

        transcript, post_error = post_processing.apply(raw, settings.post_processing)
        result: Dict[str, Any] = {
            "success": True,
            "transcript": transcript,
            "provider": PROVIDER_NAME,
            "backend": settings.backend,
            "audio_speed": settings.audio_speed,
            "chunks": chunk_count,
            "post_processing_applied": bool(
                transcript != raw and post_error is None and post_processing.is_enabled(settings.post_processing)
            ),
        }
        if post_error:
            result["post_processing_error"] = post_error
        if settings.language:
            result["language"] = settings.language
        return result


def register(ctx):
    ctx.register_transcription_provider(LocalLlmPolishedProvider())
