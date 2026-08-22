"""Config resolution: backend selection, legacy configs, defaults, aliases."""

from __future__ import annotations

import pytest


def test_defaults_match_the_builtin_local_provider(config_mod):
    settings = config_mod.load_settings({})

    assert settings.backend == config_mod.BACKEND_FASTER_WHISPER
    assert settings.model == "base"
    assert settings.language is None
    assert settings.audio_speed == 1.0
    assert settings.post_processing == {}
    assert settings.post_processing_key == ""


def test_legacy_faster_whisper_config_still_resolves(config_mod):
    """The 0.1.0 config shape — no backend key, ``polish`` stage."""
    stt = {
        "local": {"model": "small", "language": "uk"},
        "local_llm_polished": {
            "model": "base",
            "language": "en",
            "polish": {"enabled": True, "provider": "default", "model": "gpt-5.5"},
        },
    }
    settings = config_mod.load_settings(stt)

    assert settings.backend == config_mod.BACKEND_FASTER_WHISPER
    assert settings.model == "base"
    assert settings.language == "en"
    assert settings.audio_speed == 1.0
    assert settings.post_processing_key == "polish"
    assert settings.post_processing["model"] == "gpt-5.5"


def test_stt_local_is_used_as_fallback(config_mod):
    settings = config_mod.load_settings({"local": {"model": "medium", "language": "de"}})

    assert settings.model == "medium"
    assert settings.language == "de"


def test_explicit_call_arguments_win(config_mod):
    stt = {"local_llm_polished": {"model": "base", "language": "en"}}
    settings = config_mod.load_settings(stt, model="large-v3", language="uk")

    assert settings.model == "large-v3"
    assert settings.language == "uk"


def test_cloud_model_names_fall_back_to_a_local_size(config_mod):
    settings = config_mod.load_settings({"local_llm_polished": {"model": "whisper-1"}})

    assert settings.model == "base"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("faster_whisper", "faster_whisper"),
        ("faster-whisper", "faster_whisper"),
        ("Whisper", "faster_whisper"),
        ("parakeet", "parakeet"),
        ("Parakeet-V3", "parakeet"),
        ("sherpa onnx", "parakeet"),
        (None, "faster_whisper"),
        ("", "faster_whisper"),
    ],
)
def test_backend_aliases(config_mod, value, expected):
    assert config_mod.normalize_backend(value) == expected


def test_unknown_backend_falls_back_with_a_warning(config_mod, caplog):
    with caplog.at_level("WARNING"):
        settings = config_mod.load_settings({"local_llm_polished": {"backend": "vosk"}})

    assert settings.backend == config_mod.BACKEND_FASTER_WHISPER
    assert "unknown backend" in caplog.text


def test_parakeet_backend_defaults(config_mod):
    settings = config_mod.load_settings({"local_llm_polished": {"backend": "parakeet"}})

    assert settings.backend == config_mod.BACKEND_PARAKEET
    # 1.25x was the best speed/accuracy point in the benchmark.
    assert settings.audio_speed == 1.25
    assert settings.parakeet.num_threads == 6
    assert settings.parakeet.model_type == "nemo_transducer"
    assert settings.parakeet.decoding_method == "greedy_search"
    assert settings.parakeet.onnx_provider == "cpu"
    assert settings.parakeet.sample_rate == 16000
    assert settings.parakeet.feature_dim == 80
    assert settings.parakeet.max_concurrency == 1
    # Optional deps heal themselves after a Hermes venv rebuild unless the
    # user opts out.
    assert settings.parakeet.auto_install_deps is True


def test_parakeet_auto_install_deps_can_be_disabled(config_mod):
    settings = config_mod.load_settings(
        {
            "local_llm_polished": {
                "backend": "parakeet",
                "parakeet": {"auto_install_deps": False},
            }
        }
    )

    assert settings.parakeet.auto_install_deps is False


def test_audio_speed_overrides_and_precedence(config_mod):
    shared = config_mod.load_settings(
        {"local_llm_polished": {"backend": "parakeet", "audio_speed": 1.0}}
    )
    assert shared.audio_speed == 1.0

    scoped = config_mod.load_settings(
        {
            "local_llm_polished": {
                "backend": "parakeet",
                "audio_speed": 1.0,
                "parakeet": {"audio_speed": 1.5},
            }
        }
    )
    assert scoped.audio_speed == 1.5

    whisper = config_mod.load_settings(
        {"local_llm_polished": {"audio_speed": 1.25, "parakeet": {"audio_speed": 1.5}}}
    )
    assert whisper.audio_speed == 1.25


def test_audio_speed_is_clamped_and_validated(config_mod, caplog):
    with caplog.at_level("WARNING"):
        fast = config_mod.load_settings({"local_llm_polished": {"audio_speed": 9}})
    assert fast.audio_speed == config_mod.MAX_AUDIO_SPEED
    assert "audio_speed" in caplog.text

    bad = config_mod.load_settings({"local_llm_polished": {"audio_speed": "quick"}})
    assert bad.audio_speed == 1.0


def test_parakeet_thread_and_concurrency_overrides(config_mod):
    settings = config_mod.load_settings(
        {
            "local_llm_polished": {
                "backend": "parakeet",
                "parakeet": {"num_threads": 2, "max_concurrency": 3, "provider": "cuda"},
            }
        }
    )

    assert settings.parakeet.num_threads == 2
    assert settings.parakeet.max_concurrency == 3
    assert settings.parakeet.onnx_provider == "cuda"


def test_chunking_defaults(config_mod):
    chunking = config_mod.load_settings({}).chunking

    assert chunking.enabled is True
    assert chunking.threshold_seconds == 120.0
    assert chunking.chunk_seconds == 45.0
    assert chunking.overlap_seconds == 2.0


def test_chunking_values_are_sanitized(config_mod):
    chunking = config_mod.ChunkingConfig.from_dict(
        {"chunk_seconds": 20, "overlap_seconds": 30, "threshold_seconds": 5}
    )

    # Overlap can never swallow the window, and the threshold cannot be lower
    # than one window (that would chunk audio that fits in a single pass).
    assert chunking.overlap_seconds == 5.0
    assert chunking.threshold_seconds == 20.0

    tiny = config_mod.ChunkingConfig.from_dict({"chunk_seconds": 0.5})
    assert tiny.chunk_seconds == config_mod.MIN_CHUNK_SECONDS


def test_chunking_can_be_disabled(config_mod):
    assert config_mod.ChunkingConfig.from_dict({"enabled": "off"}).enabled is False


@pytest.mark.parametrize(
    "blocks,expected_key",
    [
        ({"post_processing": {"model": "a"}}, "post_processing"),
        ({"polish": {"model": "b"}}, "polish"),
        ({"repair": {"model": "c"}}, "repair"),
        ({"post_processing": {"model": "a"}, "polish": {"model": "b"}}, "post_processing"),
        ({"polish": {"model": "b"}, "repair": {"model": "c"}}, "polish"),
        ({}, ""),
    ],
)
def test_post_processing_alias_precedence(config_mod, blocks, expected_key):
    settings = config_mod.load_settings({"local_llm_polished": dict(blocks)})

    assert settings.post_processing_key == expected_key
    if expected_key:
        assert settings.post_processing == blocks[expected_key]


# ---------------------------------------------------------------------------
# Parakeet model path resolution
# ---------------------------------------------------------------------------


def test_model_path_expands_user_and_env(config_mod, parakeet_model_dir, monkeypatch):
    monkeypatch.setenv("HOME", str(parakeet_model_dir.parent))
    monkeypatch.setenv("MODELS", str(parakeet_model_dir.parent))

    for raw in ("~/parakeet-v3-int8", "$MODELS/parakeet-v3-int8"):
        settings = config_mod.load_settings(
            {"local_llm_polished": {"backend": "parakeet", "parakeet": {"model_path": raw}}}
        )
        files = settings.parakeet.resolve_files()
        assert files["encoder"] == str(parakeet_model_dir / "encoder.int8.onnx")
        assert files["tokens"] == str(parakeet_model_dir / "tokens.txt")


def test_model_path_from_environment_variable(config_mod, parakeet_model_dir, monkeypatch):
    monkeypatch.setenv(config_mod.PARAKEET_MODEL_PATH_ENV, str(parakeet_model_dir))

    settings = config_mod.load_settings({"local_llm_polished": {"backend": "parakeet"}})

    assert settings.parakeet.resolve_files()["joiner"].endswith("joiner.int8.onnx")


def test_default_model_dir_candidate_is_used(config_mod, tmp_path, monkeypatch):
    monkeypatch.delenv(config_mod.PARAKEET_MODEL_PATH_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / ".hermes" / "models" / "parakeet-v3-int8"
    model_dir.mkdir(parents=True)
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (model_dir / name).write_bytes(b"stub")

    settings = config_mod.load_settings({"local_llm_polished": {"backend": "parakeet"}})

    assert settings.parakeet.resolve_files()["encoder"] == str(model_dir / "encoder.int8.onnx")


def test_float_model_filenames_are_accepted(config_mod, tmp_path):
    model_dir = tmp_path / "parakeet-v3"
    model_dir.mkdir()
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (model_dir / name).write_bytes(b"stub")

    cfg = config_mod.ParakeetConfig.from_dict({"model_path": str(model_dir)})

    assert cfg.resolve_files()["encoder"] == str(model_dir / "encoder.onnx")


def test_explicit_file_overrides(config_mod, parakeet_model_dir):
    cfg = config_mod.ParakeetConfig.from_dict(
        {
            "encoder": str(parakeet_model_dir / "encoder.int8.onnx"),
            "decoder": str(parakeet_model_dir / "decoder.int8.onnx"),
            "joiner": str(parakeet_model_dir / "joiner.int8.onnx"),
            "tokens": str(parakeet_model_dir / "tokens.txt"),
        }
    )

    assert cfg.model_path is None
    assert cfg.resolve_files()["decoder"].endswith("decoder.int8.onnx")


def test_missing_model_path_names_the_config_key(config_mod, tmp_path, monkeypatch):
    monkeypatch.delenv(config_mod.PARAKEET_MODEL_PATH_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no candidate directory exists

    cfg = config_mod.ParakeetConfig.from_dict({})
    with pytest.raises(config_mod.ConfigError) as excinfo:
        cfg.resolve_files()

    assert "stt.local_llm_polished.parakeet.model_path" in str(excinfo.value)
    assert "encoder.int8.onnx" in str(excinfo.value)


def test_missing_model_directory_is_reported(config_mod, tmp_path):
    cfg = config_mod.ParakeetConfig.from_dict({"model_path": str(tmp_path / "nope")})

    with pytest.raises(config_mod.ConfigError, match="model directory not found"):
        cfg.resolve_files()


def test_missing_model_files_are_listed(config_mod, tmp_path):
    model_dir = tmp_path / "partial"
    model_dir.mkdir()
    (model_dir / "encoder.int8.onnx").write_bytes(b"stub")

    cfg = config_mod.ParakeetConfig.from_dict({"model_path": str(model_dir)})
    with pytest.raises(config_mod.ConfigError) as excinfo:
        cfg.resolve_files()

    message = str(excinfo.value)
    assert "decoder.int8.onnx" in message and "tokens.txt" in message
    assert "encoder.int8.onnx" not in message
