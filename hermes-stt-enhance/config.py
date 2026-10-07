"""Configuration handling for the ``stt_enhance`` STT provider.

All user-facing config lives under ``stt.stt_enhance`` in Hermes
``config.yaml``. Every key is optional: an empty block keeps the historical
behaviour (faster-whisper, ``stt.local`` fallbacks, LLM post-processing on).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PROVIDER_NAME = "stt_enhance"

BACKEND_FASTER_WHISPER = "faster_whisper"
BACKEND_PARAKEET = "parakeet"
DEFAULT_BACKEND = BACKEND_FASTER_WHISPER

# Tolerant spellings so a typo in config.yaml does not silently pick the
# wrong engine. Unknown values fall back to the default backend with a warning.
_BACKEND_ALIASES = {
    "faster_whisper": BACKEND_FASTER_WHISPER,
    "fasterwhisper": BACKEND_FASTER_WHISPER,
    "whisper": BACKEND_FASTER_WHISPER,
    "local": BACKEND_FASTER_WHISPER,
    "parakeet": BACKEND_PARAKEET,
    "parakeet_v3": BACKEND_PARAKEET,
    "parakeetv3": BACKEND_PARAKEET,
    "sherpa_onnx": BACKEND_PARAKEET,
    "sherpaonnx": BACKEND_PARAKEET,
}

# faster-whisper stays at 1.0 for byte-identical behaviour with the built-in
# ``local`` provider. Parakeet defaults to the benchmarked 1.25x recommendation.
DEFAULT_AUDIO_SPEED = {
    BACKEND_FASTER_WHISPER: 1.0,
    BACKEND_PARAKEET: 1.25,
}
MIN_AUDIO_SPEED = 0.5
MAX_AUDIO_SPEED = 3.0

PARAKEET_MODEL_PATH_ENV = "HERMES_PARAKEET_MODEL_PATH"

# Checked in order when neither config nor env var names a model directory.
PARAKEET_MODEL_DIR_CANDIDATES = (
    "~/.hermes/models/parakeet-v3-int8",
    "~/models/parakeet-v3-int8",
    "~/parakeet-v3-int8",
)

# int8 names first (the benchmarked export), then the float variants.
PARAKEET_MODEL_FILES: Dict[str, Tuple[str, ...]] = {
    "encoder": ("encoder.int8.onnx", "encoder.onnx"),
    "decoder": ("decoder.int8.onnx", "decoder.onnx"),
    "joiner": ("joiner.int8.onnx", "joiner.onnx"),
    "tokens": ("tokens.txt",),
}

DEFAULT_PARAKEET_THREADS = 6
DEFAULT_PARAKEET_SAMPLE_RATE = 16000
DEFAULT_PARAKEET_FEATURE_DIM = 80
DEFAULT_PARAKEET_MAX_CONCURRENCY = 1

MIN_CHUNK_SECONDS = 5.0


class ConfigError(ValueError):
    """Raised for config that cannot produce a working backend."""


def _cfg_get(config: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _as_float(value: Any, default: float, *, label: str = "value") -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("%s: ignoring non-numeric %s %r (using %s)", PROVIDER_NAME, label, value, default)
        return default


def _as_int(value: Any, default: int, *, label: str = "value") -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("%s: ignoring non-integer %s %r (using %s)", PROVIDER_NAME, label, value, default)
        return default


def expand_path(value: Any) -> Optional[str]:
    """Expand ``~`` and ``$VARS`` so config never needs an absolute user path."""
    if not value:
        return None
    return os.path.expanduser(os.path.expandvars(str(value).strip()))


def load_stt_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        return {}
    return _as_dict(cfg.get("stt")) if isinstance(cfg, dict) else {}


def normalize_backend(value: Any) -> str:
    if value is None:
        return DEFAULT_BACKEND
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return DEFAULT_BACKEND
    backend = _BACKEND_ALIASES.get(key)
    if backend:
        return backend
    logger.warning(
        "%s: unknown backend %r — using %r. Valid values: %s, %s.",
        PROVIDER_NAME, value, DEFAULT_BACKEND, BACKEND_FASTER_WHISPER, BACKEND_PARAKEET,
    )
    return DEFAULT_BACKEND


@dataclass(frozen=True)
class ChunkingConfig:
    """Long-audio guard rails.

    Parakeet holds the whole utterance in ONNX inference state, so a single
    long file can exhaust RAM (a 343s file was OOM-killed during benchmarking).
    Anything longer than ``threshold_seconds`` is decoded in overlapping
    windows instead.
    """

    enabled: bool = True
    threshold_seconds: float = 120.0
    chunk_seconds: float = 45.0
    overlap_seconds: float = 2.0

    @classmethod
    def from_dict(cls, raw: Any) -> "ChunkingConfig":
        cfg = _as_dict(raw)
        defaults = cls()
        chunk = _as_float(cfg.get("chunk_seconds"), defaults.chunk_seconds, label="chunking.chunk_seconds")
        chunk = max(MIN_CHUNK_SECONDS, chunk)
        overlap = _as_float(cfg.get("overlap_seconds"), defaults.overlap_seconds, label="chunking.overlap_seconds")
        # An overlap close to the window size would decode the same audio
        # repeatedly and could stall progress entirely.
        overlap = min(max(0.0, overlap), chunk / 4.0)
        threshold = _as_float(cfg.get("threshold_seconds"), defaults.threshold_seconds, label="chunking.threshold_seconds")
        threshold = max(chunk, threshold)
        return cls(
            enabled=_as_bool(cfg.get("enabled"), defaults.enabled),
            threshold_seconds=threshold,
            chunk_seconds=chunk,
            overlap_seconds=overlap,
        )


@dataclass(frozen=True)
class ParakeetConfig:
    """sherpa-onnx offline transducer settings."""

    model_path: Optional[str] = None
    file_overrides: Dict[str, str] = field(default_factory=dict)
    num_threads: int = DEFAULT_PARAKEET_THREADS
    model_type: str = "nemo_transducer"
    decoding_method: str = "greedy_search"
    onnx_provider: str = "cpu"
    sample_rate: int = DEFAULT_PARAKEET_SAMPLE_RATE
    feature_dim: int = DEFAULT_PARAKEET_FEATURE_DIM
    max_concurrency: int = DEFAULT_PARAKEET_MAX_CONCURRENCY
    # sherpa-onnx is installed lazily, so a Hermes venv rebuild wipes it.
    # Reinstalling it on first use (through Hermes' own gated installer) is
    # the default; set false to keep the environment strictly frozen.
    auto_install_deps: bool = True

    @classmethod
    def from_dict(cls, raw: Any) -> "ParakeetConfig":
        cfg = _as_dict(raw)
        defaults = cls()
        overrides = {
            key: path
            for key in PARAKEET_MODEL_FILES
            if (path := expand_path(cfg.get(key)))
        }
        return cls(
            model_path=expand_path(cfg.get("model_path") or os.getenv(PARAKEET_MODEL_PATH_ENV)),
            file_overrides=overrides,
            num_threads=max(1, _as_int(cfg.get("num_threads"), defaults.num_threads, label="parakeet.num_threads")),
            model_type=str(cfg.get("model_type") or defaults.model_type).strip(),
            decoding_method=str(cfg.get("decoding_method") or defaults.decoding_method).strip(),
            onnx_provider=str(cfg.get("provider") or defaults.onnx_provider).strip(),
            sample_rate=max(8000, _as_int(cfg.get("sample_rate"), defaults.sample_rate, label="parakeet.sample_rate")),
            feature_dim=max(1, _as_int(cfg.get("feature_dim"), defaults.feature_dim, label="parakeet.feature_dim")),
            max_concurrency=max(
                1,
                _as_int(cfg.get("max_concurrency"), defaults.max_concurrency, label="parakeet.max_concurrency"),
            ),
            auto_install_deps=_as_bool(cfg.get("auto_install_deps"), defaults.auto_install_deps),
        )

    def resolve_files(self) -> Dict[str, str]:
        """Return absolute encoder/decoder/joiner/tokens paths.

        Raises:
            ConfigError: when no model directory is configured or a required
                file is missing — the message names the config key to set.
        """
        resolved: Dict[str, str] = {}
        missing: list[str] = []

        model_dir = self.model_path or _first_existing_dir(PARAKEET_MODEL_DIR_CANDIDATES)
        if not model_dir and set(self.file_overrides) != set(PARAKEET_MODEL_FILES):
            raise ConfigError(
                "Parakeet model directory is not configured. Set "
                f"stt.{PROVIDER_NAME}.parakeet.model_path (or ${PARAKEET_MODEL_PATH_ENV}) to a "
                "sherpa-onnx Parakeet export containing "
                + ", ".join(names[0] for names in PARAKEET_MODEL_FILES.values())
                + "."
            )

        base = Path(model_dir) if model_dir else None
        if base is not None and not base.is_dir():
            raise ConfigError(f"Parakeet model directory not found: {base}")

        for key, candidates in PARAKEET_MODEL_FILES.items():
            override = self.file_overrides.get(key)
            if override:
                if Path(override).is_file():
                    resolved[key] = override
                else:
                    missing.append(override)
                continue
            found = next((str(base / name) for name in candidates if base and (base / name).is_file()), None)
            if found:
                resolved[key] = found
            else:
                missing.append(f"{base / candidates[0]}" if base else candidates[0])

        if missing:
            raise ConfigError(
                "Parakeet model files missing: " + ", ".join(missing)
                + f". Check stt.{PROVIDER_NAME}.parakeet.model_path."
            )
        return resolved

    def cache_key(self, files: Dict[str, str]) -> Tuple[Any, ...]:
        return (
            tuple(sorted(files.items())),
            self.num_threads,
            self.model_type,
            self.decoding_method,
            self.onnx_provider,
            self.sample_rate,
            self.feature_dim,
        )


def _first_existing_dir(candidates: Tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        path = expand_path(candidate)
        if path and Path(path).is_dir():
            return path
    return None


@dataclass(frozen=True)
class Settings:
    """Fully resolved provider settings for one transcription call."""

    backend: str
    model: str
    language: Optional[str]
    audio_speed: float
    chunking: ChunkingConfig
    parakeet: ParakeetConfig
    post_processing: Dict[str, Any]


def _normalize_model(configured: Any) -> str:
    try:
        from tools.transcription_tools import _normalize_local_model

        return _normalize_local_model(configured)
    except Exception:
        return str(configured or "base")


def _normalize_language(configured: Any) -> Optional[str]:
    try:  # Present in newer Hermes builds; harmless when absent.
        from tools.transcription_tools import _normalize_stt_language

        return _normalize_stt_language(configured)
    except Exception:
        return str(configured).strip() if configured else None


def load_settings(
    stt_config: Dict[str, Any],
    *,
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> Settings:
    """Merge call arguments, provider config and ``stt.local`` fallbacks."""
    provider_cfg = _as_dict(stt_config.get(PROVIDER_NAME))
    backend = normalize_backend(provider_cfg.get("backend"))
    parakeet_cfg = _as_dict(provider_cfg.get(BACKEND_PARAKEET))

    # Backend-scoped audio_speed wins over the shared key, which in turn wins
    # over the per-backend default.
    speed_raw = parakeet_cfg.get("audio_speed") if backend == BACKEND_PARAKEET else None
    if speed_raw is None:
        speed_raw = provider_cfg.get("audio_speed")
    speed = _as_float(speed_raw, DEFAULT_AUDIO_SPEED[backend], label="audio_speed")
    if not MIN_AUDIO_SPEED <= speed <= MAX_AUDIO_SPEED:
        logger.warning(
            "%s: audio_speed %s out of range [%s, %s] — clamping.",
            PROVIDER_NAME, speed, MIN_AUDIO_SPEED, MAX_AUDIO_SPEED,
        )
        speed = min(max(speed, MIN_AUDIO_SPEED), MAX_AUDIO_SPEED)

    return Settings(
        backend=backend,
        model=_normalize_model(
            model or provider_cfg.get("model") or _cfg_get(stt_config, "local", "model") or "base"
        ),
        language=_normalize_language(
            language or provider_cfg.get("language") or _cfg_get(stt_config, "local", "language")
        ),
        audio_speed=speed,
        chunking=ChunkingConfig.from_dict(provider_cfg.get("chunking")),
        parakeet=ParakeetConfig.from_dict(parakeet_cfg),
        post_processing=_as_dict(provider_cfg.get("post_processing")),
    )
