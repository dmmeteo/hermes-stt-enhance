"""ASR backends: faster-whisper delegation, Parakeet decoding, concurrency."""

from __future__ import annotations

import sys
import threading

import pytest


def _parakeet_settings(config_mod, model_dir, **parakeet):
    return config_mod.load_settings(
        {
            "local_llm_polished": {
                "backend": "parakeet",
                "parakeet": {"model_path": str(model_dir), **parakeet},
            }
        }
    )


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_build_backend_selects_faster_whisper_by_default(backends_mod, config_mod):
    backend = backends_mod.build_backend(config_mod.load_settings({}))

    assert isinstance(backend, backends_mod.FasterWhisperBackend)
    assert backend.name == config_mod.BACKEND_FASTER_WHISPER
    assert backend.model == "base"
    # faster-whisper handles its own windowing and input formats.
    assert backend.requires_wav is False
    assert backend.supports_chunking is False
    assert backend.max_concurrency == 0


def test_build_backend_selects_parakeet(backends_mod, config_mod, parakeet_model_dir):
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    assert isinstance(backend, backends_mod.ParakeetBackend)
    assert backend.name == config_mod.BACKEND_PARAKEET
    assert backend.requires_wav is True
    assert backend.supports_chunking is True
    assert backend.max_concurrency == 1


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------


def test_faster_whisper_delegates_to_hermes(backends_mod, local_whisper):
    calls = local_whisper({"success": True, "transcript": "  hello world  "})

    backend = backends_mod.FasterWhisperBackend("small")
    assert backend.transcribe_file("/tmp/audio.ogg", "en") == "  hello world  "
    assert calls == [{"file_path": "/tmp/audio.ogg", "model": "small"}]


def test_faster_whisper_propagates_the_hermes_error_message(backends_mod, local_whisper):
    local_whisper({"success": False, "transcript": "", "error": "faster-whisper not installed"})

    with pytest.raises(backends_mod.BackendError, match="faster-whisper not installed"):
        backends_mod.FasterWhisperBackend("base").transcribe_file("/tmp/a.ogg", None)


def test_faster_whisper_reports_a_missing_error_message(backends_mod, local_whisper):
    local_whisper({"success": False, "transcript": ""})

    with pytest.raises(backends_mod.BackendError, match="Local transcription failed"):
        backends_mod.FasterWhisperBackend("base").transcribe_file("/tmp/a.ogg", None)


def test_faster_whisper_rejects_a_non_dict_result(backends_mod, local_whisper):
    local_whisper("just a string")

    with pytest.raises(backends_mod.BackendError, match="invalid result"):
        backends_mod.FasterWhisperBackend("base").transcribe_file("/tmp/a.ogg", None)


def test_faster_whisper_wraps_exceptions(backends_mod, local_whisper):
    local_whisper(lambda *_: (_ for _ in ()).throw(RuntimeError("model download failed")))

    with pytest.raises(backends_mod.BackendError, match="model download failed"):
        backends_mod.FasterWhisperBackend("base").transcribe_file("/tmp/a.ogg", None)


def test_faster_whisper_tolerates_a_missing_transcript_key(backends_mod, local_whisper):
    local_whisper({"success": True})

    assert backends_mod.FasterWhisperBackend("base").transcribe_file("/tmp/a.ogg", None) == ""


# ---------------------------------------------------------------------------
# Parakeet backend
# ---------------------------------------------------------------------------


def test_parakeet_builds_the_recognizer_from_the_resolved_files(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    recorder = fake_sherpa(lambda stream: "parakeet text")
    settings = _parakeet_settings(config_mod, parakeet_model_dir, num_threads=3)

    backend = backends_mod.build_backend(settings)
    assert backend.transcribe_samples([0.0] * 16000, 16000) == "parakeet text"

    (kwargs,) = recorder["from_transducer"]
    assert kwargs["encoder"] == str(parakeet_model_dir / "encoder.int8.onnx")
    assert kwargs["decoder"] == str(parakeet_model_dir / "decoder.int8.onnx")
    assert kwargs["joiner"] == str(parakeet_model_dir / "joiner.int8.onnx")
    assert kwargs["tokens"] == str(parakeet_model_dir / "tokens.txt")
    assert kwargs["num_threads"] == 3
    assert kwargs["model_type"] == "nemo_transducer"
    assert kwargs["decoding_method"] == "greedy_search"
    assert kwargs["provider"] == "cpu"
    assert kwargs["sample_rate"] == 16000
    assert kwargs["feature_dim"] == 80


def test_parakeet_strips_the_decoded_text(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    fake_sherpa(lambda stream: "  padded text \n")
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    assert backend.transcribe_samples([0.1] * 100, 16000) == "padded text"


def test_parakeet_passes_the_waveform_and_sample_rate_to_sherpa(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    recorder = fake_sherpa()
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    backend.transcribe_samples([0.0] * 4321, 16000)

    (recognizer,) = recorder["recognizers"]
    (stream,) = recognizer.streams
    assert stream.accepted == [(16000, 4321)]
    assert recorder["decoded"] == [stream]


def test_parakeet_transcribes_a_wav_file(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa, make_wav
):
    recorder = fake_sherpa(lambda stream: "from file")
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    assert backend.transcribe_file(str(make_wav(seconds=0.5)), None) == "from file"

    (recognizer,) = recorder["recognizers"]
    assert recognizer.streams[0].accepted == [(16000, 8000)]


def test_parakeet_skips_empty_audio_without_loading_the_model(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    recorder = fake_sherpa()
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    assert backend.transcribe_samples([], 16000) == ""
    assert recorder["from_transducer"] == []


def test_parakeet_caches_the_recognizer_across_instances(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    """Loading the ONNX model is expensive — identical config must reuse it."""
    recorder = fake_sherpa()
    settings = _parakeet_settings(config_mod, parakeet_model_dir)

    backends_mod.build_backend(settings).transcribe_samples([0.0] * 10, 16000)
    backends_mod.build_backend(settings).transcribe_samples([0.0] * 10, 16000)

    assert len(recorder["from_transducer"]) == 1


def test_parakeet_reloads_when_the_configuration_changes(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    recorder = fake_sherpa()

    for threads in (2, 4, 4):
        settings = _parakeet_settings(config_mod, parakeet_model_dir, num_threads=threads)
        backends_mod.build_backend(settings).transcribe_samples([0.0] * 10, 16000)

    assert [kwargs["num_threads"] for kwargs in recorder["from_transducer"]] == [2, 4]


def test_reset_caches_drops_the_loaded_recognizer(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    recorder = fake_sherpa()
    settings = _parakeet_settings(config_mod, parakeet_model_dir)

    backends_mod.build_backend(settings).transcribe_samples([0.0] * 10, 16000)
    backends_mod.reset_caches()
    backends_mod.build_backend(settings).transcribe_samples([0.0] * 10, 16000)

    assert len(recorder["from_transducer"]) == 2


def test_parakeet_without_sherpa_onnx_explains_the_install(
    backends_mod, config_mod, parakeet_model_dir, monkeypatch
):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    with pytest.raises(backends_mod.BackendError, match="pip install"):
        backend.transcribe_samples([0.0] * 10, 16000)


def test_parakeet_provisions_its_runtime_before_loading_the_model(
    backends_mod, runtime_deps_mod, config_mod, parakeet_model_dir, no_sherpa,
    fake_sherpa, monkeypatch
):
    """A rebuilt Hermes venv heals itself on the next voice message."""
    calls = []

    def _ensure(*, auto_install, lock=None):
        calls.append(auto_install)
        fake_sherpa(lambda stream: "healed")  # what a provisioned runtime gives us
        return "installed"

    monkeypatch.setattr(runtime_deps_mod, "ensure", _ensure)
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    assert backend.transcribe_samples([0.0] * 10, 16000) == "healed"
    assert calls == [True]


def test_parakeet_dependency_failure_names_the_provisioning_command(
    backends_mod, runtime_deps_mod, config_mod, parakeet_model_dir, no_sherpa,
    install_gate
):
    install_gate(allow=False)
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    with pytest.raises(backends_mod.BackendError) as exc:
        backend.transcribe_samples([0.0] * 10, 16000)

    assert runtime_deps_mod.manual_install_command() in str(exc.value)
    assert "--target" in str(exc.value)


def test_parakeet_passes_the_auto_install_setting_through(
    backends_mod, runtime_deps_mod, config_mod, parakeet_model_dir, no_sherpa,
    monkeypatch
):
    calls = []

    def _ensure(*, auto_install, lock=None):
        calls.append(auto_install)
        raise runtime_deps_mod.DependencyError("auto_install_deps is false")

    monkeypatch.setattr(runtime_deps_mod, "ensure", _ensure)
    settings = config_mod.load_settings({
        "local_llm_polished": {
            "backend": "parakeet",
            "parakeet": {
                "model_path": str(parakeet_model_dir),
                "auto_install_deps": False,
            },
        }
    })
    backend = backends_mod.build_backend(settings)

    with pytest.raises(backends_mod.BackendError, match="auto_install_deps"):
        backend.transcribe_samples([0.0] * 10, 16000)
    assert calls == [False]


def test_parakeet_reports_a_model_load_failure(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    fake_sherpa(load_error=RuntimeError("corrupt encoder.int8.onnx"))
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    with pytest.raises(backends_mod.BackendError, match="Failed to load the Parakeet model"):
        backend.transcribe_samples([0.0] * 10, 16000)


def test_parakeet_config_errors_surface_as_backend_errors(
    backends_mod, config_mod, tmp_path, monkeypatch
):
    monkeypatch.delenv(config_mod.PARAKEET_MODEL_PATH_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    backend = backends_mod.build_backend(
        config_mod.load_settings({"local_llm_polished": {"backend": "parakeet"}})
    )

    with pytest.raises(backends_mod.BackendError, match="parakeet.model_path"):
        backend.transcribe_samples([0.0] * 10, 16000)


def test_parakeet_reports_a_decoding_failure(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    def _explode(stream):
        raise RuntimeError("onnxruntime out of memory")

    fake_sherpa(_explode)
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    with pytest.raises(backends_mod.BackendError, match="Parakeet decoding failed"):
        backend.transcribe_samples([0.0] * 10, 16000)


def test_parakeet_tolerates_an_empty_result(
    backends_mod, config_mod, parakeet_model_dir, fake_sherpa
):
    fake_sherpa(lambda stream: None)
    backend = backends_mod.build_backend(_parakeet_settings(config_mod, parakeet_model_dir))

    assert backend.transcribe_samples([0.0] * 10, 16000) == ""


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------


def test_base_backend_methods_are_abstract(backends_mod):
    backend = backends_mod.SttBackend()

    with pytest.raises(NotImplementedError):
        backend.transcribe_file("/tmp/a.wav", None)
    with pytest.raises(NotImplementedError):
        backend.transcribe_samples([0.0], 16000)


# ---------------------------------------------------------------------------
# Concurrency guard
# ---------------------------------------------------------------------------


def test_concurrency_guard_is_a_noop_when_unlimited(backends_mod):
    with backends_mod.concurrency_guard(0):
        with backends_mod.concurrency_guard(0):
            pass  # nesting would deadlock a real semaphore of size 1


def test_concurrency_guard_serializes_decodes(backends_mod):
    """max_concurrency=1 must never let two decodes run at the same time."""
    inside = 0
    peak = 0
    lock = threading.Lock()
    start = threading.Event()

    def _work():
        nonlocal inside, peak
        start.wait(5)
        with backends_mod.concurrency_guard(1):
            with lock:
                inside += 1
                peak = max(peak, inside)
            # Long enough for a competing thread to reach the guard.
            threading.Event().wait(0.05)
            with lock:
                inside -= 1

    threads = [threading.Thread(target=_work) for _ in range(4)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(10)

    assert peak == 1


def test_concurrency_guard_times_out_instead_of_hanging(backends_mod, monkeypatch):
    monkeypatch.setattr(backends_mod, "CONCURRENCY_WAIT_SECONDS", 0.05)

    with backends_mod.concurrency_guard(1):
        with pytest.raises(backends_mod.BackendError, match="Timed out"):
            with backends_mod.concurrency_guard(1):
                pass


def test_concurrency_guard_releases_the_slot_on_failure(backends_mod):
    with pytest.raises(ValueError):
        with backends_mod.concurrency_guard(1):
            raise ValueError("decode blew up")

    # The slot is free again, so the next caller does not block.
    with backends_mod.concurrency_guard(1):
        pass
