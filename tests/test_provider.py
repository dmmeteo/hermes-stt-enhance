"""Provider surface: Hermes contract, envelopes, and the full transcribe pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from conftest import requires_ffmpeg

PROVIDER_NAME = "stt_enhance"


@pytest.fixture
def provider(plugin):
    return plugin.SttEnhanceProvider()


class RecordingBackend:
    """Stand-in backend that records how the pipeline calls it."""

    name = "recording"

    def __init__(self, text="recorded text", *, requires_wav=False, supports_chunking=False,
                 max_concurrency=0, error=None):
        self.text = text
        self.requires_wav = requires_wav
        self.supports_chunking = supports_chunking
        self.max_concurrency = max_concurrency
        self.error = error
        self.files = []
        self.windows = []

    def transcribe_file(self, file_path, language):
        self.files.append((file_path, language))
        if self.error is not None:
            raise self.error
        return self.text

    def transcribe_samples(self, samples, sample_rate):
        self.windows.append((len(samples), sample_rate))
        if self.error is not None:
            raise self.error
        return self.text


@pytest.fixture
def use_backend(plugin, monkeypatch):
    """Force the pipeline to use a given backend instance."""

    def _set(backend):
        monkeypatch.setattr(plugin, "build_backend", lambda settings: backend)
        return backend

    return _set


def _parakeet_config(model_dir, **overrides):
    parakeet = {"model_path": str(model_dir)}
    parakeet.update(overrides.pop("parakeet", {}))
    return {
        "stt": {
            "stt_enhance": {
                "backend": "parakeet",
                # The fixture WAVs are already 16 kHz mono, so 1.0 keeps ffmpeg
                # out of the picture for tests that are not about resampling.
                "audio_speed": 1.0,
                "parakeet": parakeet,
                **overrides,
            }
        }
    }


# ---------------------------------------------------------------------------
# Hermes provider contract
# ---------------------------------------------------------------------------


def test_provider_implements_the_hermes_abc(provider):
    from agent.transcription_provider import TranscriptionProvider

    assert isinstance(provider, TranscriptionProvider)


def test_provider_name_does_not_shadow_a_builtin(provider):
    # Hermes rejects plugin providers that collide with a built-in name.
    builtin = {"local", "local_command", "groq", "openai", "mistral", "xai"}

    assert provider.name == PROVIDER_NAME
    assert provider.name not in builtin
    assert provider.name == provider.name.strip().lower()


def test_display_name_and_setup_schema(provider):
    schema = provider.get_setup_schema()

    assert provider.display_name == "STT Enhance"
    assert schema["name"] == provider.display_name
    assert schema["badge"] == "local+LLM"
    assert schema["env_vars"] == []
    # The picker subtitle should name both engines.
    assert "Parakeet" in schema["tag"]
    assert "whisper" in schema["tag"]


def test_model_catalog_and_default(provider):
    ids = [entry["id"] for entry in provider.list_models()]

    assert ids == ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
    assert all(entry["display"] for entry in provider.list_models())
    # ``base`` matches the default the config layer falls back to.
    assert provider.default_model() == "base"


def test_register_hands_a_provider_instance_to_hermes(plugin):
    registered = []

    class Ctx:
        def register_transcription_provider(self, instance):
            registered.append(instance)

    plugin.register(Ctx())

    assert len(registered) == 1
    assert isinstance(registered[0], plugin.SttEnhanceProvider)
    assert registered[0].name == PROVIDER_NAME


# ---------------------------------------------------------------------------
# Availability probing (must never raise)
# ---------------------------------------------------------------------------


def test_faster_whisper_backend_is_always_available(provider, hermes_config):
    hermes_config({})

    assert provider.is_available() is True


def test_parakeet_is_available_with_sherpa_and_a_model(
    provider, hermes_config, parakeet_model_dir, fake_sherpa
):
    fake_sherpa()
    hermes_config(_parakeet_config(parakeet_model_dir))

    assert provider.is_available() is True


def test_parakeet_is_unavailable_when_the_runtime_cannot_be_provided(
    provider, hermes_config, parakeet_model_dir, no_sherpa, install_gate
):
    # Runtime installs turned off and no plugin runtime on disk.
    install_gate(allow=False)
    hermes_config(_parakeet_config(parakeet_model_dir))

    assert provider.is_available() is False


def test_parakeet_stays_available_when_the_runtime_can_be_provisioned(
    provider, hermes_config, parakeet_model_dir, no_sherpa, install_gate,
    runtime_deps_mod, monkeypatch
):
    """The production regression, at the provider surface.

    Hermes short-circuits with "STT plugin is not available" *before* calling
    ``transcribe``. Reporting False for a dependency first use would provision
    is what turned a venv rebuild into a dead voice pipeline.
    """
    install_gate()
    installs = []
    monkeypatch.setattr(
        runtime_deps_mod, "install_runtime",
        lambda lock=None: installs.append(lock) or Path("/unused"),
    )
    hermes_config(_parakeet_config(parakeet_model_dir))

    assert provider.is_available() is True
    assert installs == [], "availability probing must not install"


def test_availability_defers_to_the_runtime_probe(
    provider, hermes_config, parakeet_model_dir, runtime_deps_mod, monkeypatch
):
    hermes_config(_parakeet_config(parakeet_model_dir))
    monkeypatch.setattr(runtime_deps_mod, "can_provide", lambda **kwargs: False)

    assert provider.is_available() is False

    monkeypatch.setattr(runtime_deps_mod, "can_provide", lambda **kwargs: True)

    assert provider.is_available() is True


def test_parakeet_is_unavailable_without_model_files(
    provider, hermes_config, tmp_path, fake_sherpa
):
    fake_sherpa()
    hermes_config(_parakeet_config(tmp_path / "missing"))

    assert provider.is_available() is False


def test_is_available_never_raises(provider, plugin, monkeypatch):
    monkeypatch.setattr(
        plugin, "load_stt_config", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert provider.is_available() is False


# ---------------------------------------------------------------------------
# faster-whisper pipeline
# ---------------------------------------------------------------------------


def test_default_pipeline_transcribes_and_post_processes(
    provider, hermes_config, local_whisper, call_llm
):
    hermes_config({"stt": {"stt_enhance": {"model": "base", "language": "en"}}})
    whisper_calls = local_whisper({"success": True, "transcript": "deploy the stage in cluster"})
    llm_calls = call_llm(reply="Deploy the staging cluster.")

    result = provider.transcribe("/tmp/voice.ogg")

    assert result == {
        "success": True,
        "transcript": "Deploy the staging cluster.",
        "provider": PROVIDER_NAME,
        "backend": "faster_whisper",
        "audio_speed": 1.0,
        "chunks": 1,
        "post_processing_applied": True,
        "language": "en",
    }
    # The original file is handed straight to Hermes — no ffmpeg round-trip.
    assert whisper_calls == [{"file_path": "/tmp/voice.ogg", "model": "base"}]
    assert len(llm_calls) == 1


def test_config_model_and_language_reach_the_backend(
    provider, hermes_config, local_whisper, call_llm
):
    hermes_config({"stt": {"local": {"model": "small", "language": "uk"}}})
    whisper_calls = local_whisper({"success": True, "transcript": "привіт"})
    call_llm(reply="Привіт.")

    result = provider.transcribe("/tmp/voice.ogg")

    assert whisper_calls[0]["model"] == "small"
    assert result["language"] == "uk"


def test_call_arguments_override_the_config(provider, hermes_config, local_whisper, call_llm):
    hermes_config({"stt": {"stt_enhance": {"model": "base", "language": "en"}}})
    whisper_calls = local_whisper({"success": True, "transcript": "hello"})
    call_llm(reply="Hello.")

    result = provider.transcribe("/tmp/voice.ogg", model="medium", language="de")

    assert whisper_calls[0]["model"] == "medium"
    assert result["language"] == "de"


def test_post_processing_can_be_disabled(provider, hermes_config, local_whisper, call_llm):
    hermes_config(
        {"stt": {"stt_enhance": {"post_processing": {"enabled": False}}}}
    )
    local_whisper({"success": True, "transcript": "raw transcript"})
    llm_calls = call_llm()

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["transcript"] == "raw transcript"
    assert result["post_processing_applied"] is False
    assert "post_processing_error" not in result
    assert llm_calls == []


def test_post_processing_failure_keeps_the_raw_transcript(
    provider, hermes_config, local_whisper, call_llm
):
    hermes_config({})
    local_whisper({"success": True, "transcript": "raw transcript"})
    call_llm(error=RuntimeError("upstream 503"))

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is True
    assert result["transcript"] == "raw transcript"
    assert result["post_processing_applied"] is False
    assert result["post_processing_error"] == "upstream 503"


def test_unchanged_text_is_not_reported_as_post_processed(
    provider, hermes_config, local_whisper, call_llm
):
    hermes_config({})
    local_whisper({"success": True, "transcript": "already clean"})
    call_llm(reply="already clean")

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["transcript"] == "already clean"
    assert result["post_processing_applied"] is False


def test_backend_failure_returns_an_error_envelope(
    provider, hermes_config, local_whisper, call_llm
):
    hermes_config({})
    local_whisper({"success": False, "transcript": "", "error": "faster-whisper not installed"})
    llm_calls = call_llm()

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is False
    assert result["transcript"] == ""
    assert result["provider"] == PROVIDER_NAME
    assert result["backend"] == "faster_whisper"
    assert "faster-whisper not installed" in result["error"]
    # Nothing to clean up, so the LLM is never called.
    assert llm_calls == []


def test_invalid_configuration_returns_an_error_envelope(provider, plugin, monkeypatch):
    monkeypatch.setattr(
        plugin, "load_settings",
        lambda *a, **k: (_ for _ in ()).throw(plugin.ConfigError("bad chunking block")),
    )

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is False
    assert result["provider"] == PROVIDER_NAME
    assert "bad chunking block" in result["error"]


def test_unexpected_errors_are_converted_to_an_envelope(
    provider, hermes_config, use_backend, caplog
):
    """The ABC forbids raising, so even a bug must come back as an envelope."""
    hermes_config({})
    use_backend(RecordingBackend(error=ZeroDivisionError("division by zero")))

    with caplog.at_level("ERROR"):
        result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is False
    assert "division by zero" in result["error"]
    assert "unexpected failure" in caplog.text


def test_audio_errors_are_converted_to_an_envelope(
    provider, hermes_config, monkeypatch, plugin, make_wav
):
    hermes_config({"stt": {"stt_enhance": {"audio_speed": 1.5}}})
    monkeypatch.setattr(plugin.audio, "find_binary", lambda name: None)

    result = provider.transcribe(str(make_wav()))

    assert result["success"] is False
    assert "ffmpeg is required" in result["error"]
    assert result["backend"] == "faster_whisper"


# ---------------------------------------------------------------------------
# Audio preparation and temp-file hygiene
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_audio_speed_prepares_a_temporary_wav_for_faster_whisper(
    provider, hermes_config, use_backend, call_llm, make_wav, audio_mod
):
    hermes_config({"stt": {"stt_enhance": {"audio_speed": 2.0}}})
    backend = use_backend(RecordingBackend("sped up"))
    call_llm(reply="Sped up.")
    source = make_wav(seconds=4.0)

    result = provider.transcribe(str(source))

    (prepared, language), = backend.files
    assert prepared != str(source)
    assert result["audio_speed"] == 2.0
    # The temporary working directory is removed once transcribe() returns.
    assert not os.path.exists(prepared)


def test_matching_input_is_passed_through_untouched(
    provider, hermes_config, use_backend, call_llm, make_wav
):
    hermes_config(
        {"stt": {"stt_enhance": {"audio_speed": 1.0, "post_processing": {"enabled": False}}}}
    )
    backend = use_backend(RecordingBackend("as-is", requires_wav=True))
    call_llm()
    source = make_wav(seconds=1.0)

    provider.transcribe(str(source))

    assert backend.files == [(str(source), None)]


# ---------------------------------------------------------------------------
# Parakeet pipeline
# ---------------------------------------------------------------------------


def test_parakeet_single_pass(
    provider, hermes_config, parakeet_model_dir, fake_sherpa, call_llm, make_wav
):
    recorder = fake_sherpa(lambda stream: "parakeet transcript")
    hermes_config(_parakeet_config(parakeet_model_dir, post_processing={"enabled": False}))

    result = provider.transcribe(str(make_wav(seconds=2.0)))

    assert result["success"] is True
    assert result["transcript"] == "parakeet transcript"
    assert result["backend"] == "parakeet"
    assert result["chunks"] == 1
    assert result["audio_speed"] == 1.0
    assert len(recorder["decoded"]) == 1


def test_parakeet_heals_a_rebuilt_venv_on_the_next_voice_message(
    provider, hermes_config, parakeet_model_dir, no_sherpa, runtime_deps_mod,
    fake_sherpa, install_gate, make_wav, monkeypatch
):
    """End-to-end version of the outage: deps gone, one voice message, working STT."""
    install_gate()
    provisioned = []

    def _ensure(*, auto_install, lock=None):
        provisioned.append(auto_install)
        fake_sherpa(lambda stream: "parakeet transcript")
        return "installed"

    monkeypatch.setattr(runtime_deps_mod, "ensure", _ensure)
    hermes_config(_parakeet_config(parakeet_model_dir, post_processing={"enabled": False}))

    assert provider.is_available() is True
    result = provider.transcribe(str(make_wav(seconds=2.0)))

    assert result["success"] is True
    assert result["transcript"] == "parakeet transcript"
    assert provisioned == [True]


def test_parakeet_missing_deps_return_an_actionable_error_envelope(
    provider, hermes_config, parakeet_model_dir, no_sherpa, install_gate,
    runtime_deps_mod, make_wav
):
    install_gate(allow=False)
    hermes_config(_parakeet_config(parakeet_model_dir))

    result = provider.transcribe(str(make_wav(seconds=2.0)))

    assert result["success"] is False
    assert result["backend"] == "parakeet"
    # The envelope reaches the user as the STT failure message, so it has to
    # carry the fix rather than a generic "dependencies are not configured".
    assert runtime_deps_mod.manual_install_command() in result["error"]


def test_parakeet_chunks_long_audio_and_merges_the_result(
    provider, hermes_config, parakeet_model_dir, fake_sherpa, call_llm, make_wav
):
    texts = iter(
        [
            "first chunk of speech and then",
            "and then the second chunk and finally",
            "and finally the third chunk",
        ]
    )
    recorder = fake_sherpa(lambda stream: next(texts))
    hermes_config(
        _parakeet_config(
            parakeet_model_dir,
            chunking={"threshold_seconds": 5, "chunk_seconds": 5, "overlap_seconds": 1},
            post_processing={"enabled": False},
        )
    )

    result = provider.transcribe(str(make_wav(seconds=12.0)))

    assert result["chunks"] == 3
    assert len(recorder["decoded"]) == 3
    # Windows advance by chunk - overlap = 4s and the last one reaches the end.
    assert [samples for samples, _ in
            [(len(stream.accepted[0][1] * [0]), 0) for stream in recorder["recognizers"][0].streams]
            ] == [80000, 80000, 64000]
    assert result["transcript"] == (
        "first chunk of speech and then the second chunk and finally the third chunk"
    )


def test_parakeet_chunking_can_be_disabled(
    provider, hermes_config, parakeet_model_dir, fake_sherpa, make_wav
):
    recorder = fake_sherpa(lambda stream: "one long pass")
    hermes_config(
        _parakeet_config(
            parakeet_model_dir,
            chunking={"enabled": False, "threshold_seconds": 5, "chunk_seconds": 5},
            post_processing={"enabled": False},
        )
    )

    result = provider.transcribe(str(make_wav(seconds=12.0)))

    assert result["chunks"] == 1
    assert result["transcript"] == "one long pass"
    assert len(recorder["decoded"]) == 1


def test_parakeet_falls_back_to_one_pass_when_the_duration_is_unknown(
    provider, hermes_config, parakeet_model_dir, fake_sherpa, make_wav, plugin, monkeypatch, caplog
):
    fake_sherpa(lambda stream: "single pass")
    monkeypatch.setattr(plugin.audio, "probe_duration_seconds", lambda path: None)
    hermes_config(
        _parakeet_config(
            parakeet_model_dir,
            chunking={"threshold_seconds": 5, "chunk_seconds": 5},
            post_processing={"enabled": False},
        )
    )

    with caplog.at_level("WARNING"):
        result = provider.transcribe(str(make_wav(seconds=12.0)))

    assert result["chunks"] == 1
    assert "could not determine the duration" in caplog.text


def test_parakeet_short_audio_is_not_chunked(
    provider, hermes_config, parakeet_model_dir, fake_sherpa, make_wav
):
    fake_sherpa(lambda stream: "short")
    hermes_config(
        _parakeet_config(
            parakeet_model_dir,
            chunking={"threshold_seconds": 10, "chunk_seconds": 5},
            post_processing={"enabled": False},
        )
    )

    result = provider.transcribe(str(make_wav(seconds=6.0)))

    assert result["chunks"] == 1


@requires_ffmpeg
def test_parakeet_speeds_up_and_resamples_before_decoding(
    provider, hermes_config, parakeet_model_dir, fake_sherpa, make_wav
):
    recorder = fake_sherpa(lambda stream: "fast")
    config = _parakeet_config(parakeet_model_dir, post_processing={"enabled": False})
    # The benchmarked default: 1.25x speed-up before decoding.
    del config["stt"]["stt_enhance"]["audio_speed"]
    hermes_config(config)

    result = provider.transcribe(str(make_wav(seconds=5.0)))

    assert result["audio_speed"] == 1.25
    (stream,) = recorder["recognizers"][0].streams
    sample_rate, sample_count = stream.accepted[0]
    assert sample_rate == 16000
    # 5s at 1.25x is ~4s of audio at 16 kHz.
    assert sample_count == pytest.approx(64000, abs=2000)


def test_parakeet_reports_a_missing_model_as_an_error_envelope(
    provider, hermes_config, tmp_path, fake_sherpa, make_wav
):
    fake_sherpa()
    hermes_config(_parakeet_config(tmp_path / "missing"))

    result = provider.transcribe(str(make_wav()))

    assert result["success"] is False
    assert result["backend"] == "parakeet"
    assert "model directory not found" in result["error"]


def test_parakeet_post_processing_runs_on_the_merged_transcript(
    provider, hermes_config, parakeet_model_dir, fake_sherpa, call_llm, make_wav
):
    texts = iter(["hello there and", "and welcome"])
    fake_sherpa(lambda stream: next(texts))
    llm_calls = call_llm(reply="Hello there and welcome.")
    hermes_config(
        _parakeet_config(
            parakeet_model_dir,
            chunking={"threshold_seconds": 5, "chunk_seconds": 5, "overlap_seconds": 1},
        )
    )

    result = provider.transcribe(str(make_wav(seconds=8.0)))

    assert result["chunks"] == 2
    assert result["transcript"] == "Hello there and welcome."
    assert "hello there and welcome" in llm_calls[0]["messages"][1]["content"]


def test_unloadable_skill_returns_the_raw_transcript_with_a_visible_error(
    provider, hermes_config, local_whisper, call_llm, skill_roots
):
    hermes_config({"stt": {"stt_enhance": {"post_processing": {"skill": "not-installed"}}}})
    local_whisper({"success": True, "transcript": "raw transcript"})
    calls = call_llm()

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is True
    assert result["transcript"] == "raw transcript"
    assert result["post_processing_applied"] is False
    assert "not-installed" in result["post_processing_error"]
    assert calls == []


def test_inaccessible_skill_path_keeps_the_asr_transcript(
    provider, hermes_config, local_whisper, call_llm, stat_denied
):
    stat_denied("/denied")
    hermes_config({"stt": {"stt_enhance": {"post_processing": {"skill": "/denied/x/SKILL.md"}}}})
    local_whisper({"success": True, "transcript": "raw transcript"})
    calls = call_llm()

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is True
    assert result["transcript"] == "raw transcript"
    assert result["post_processing_applied"] is False
    assert "Permission denied" in result["post_processing_error"]
    assert calls == []


def test_inaccessible_skill_root_keeps_the_asr_transcript(
    provider, hermes_config, local_whisper, call_llm, skill_roots, stat_denied
):
    (profile, _external), _ = skill_roots
    stat_denied(profile, methods=("is_dir",))
    hermes_config({"stt": {"stt_enhance": {"post_processing": {"skill": "cleanup"}}}})
    local_whisper({"success": True, "transcript": "raw transcript"})
    call_llm()

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is True
    assert result["transcript"] == "raw transcript"
    assert "Permission denied" in result["post_processing_error"]


def test_unmigrated_config_makes_no_llm_call(provider, hermes_config, local_whisper, call_llm):
    hermes_config({"stt": {"stt_enhance": {}, "local_llm_polished": {"post_processing": {"enabled": False}}}})
    local_whisper({"success": True, "transcript": "raw"})
    llm_calls = call_llm(reply="Cleaned.")

    result = provider.transcribe("/tmp/voice.ogg")

    assert result["success"] is False
    assert "migrate_config.py" in result["error"]
    assert llm_calls == []
    assert provider.is_available() is False
