"""Test harness for the plugin.

The plugin directory is hyphenated (``hermes-stt-enhance/``) and imports Hermes
modules, so tests load it exactly the way ``hermes_cli.plugins`` does — as
``hermes_plugins.hermes_stt_enhance`` with ``submodule_search_locations`` — on
top of minimal stand-ins for the Hermes APIs it touches. That keeps the suite
runnable without a Hermes checkout.
"""

from __future__ import annotations

import abc
import importlib.machinery
import importlib.util
import os
import shutil
import sys
import types
import wave
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "hermes-stt-enhance"
MODULE_NAME = "hermes_plugins.hermes_stt_enhance"

HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


# ---------------------------------------------------------------------------
# Hermes API stand-ins
# ---------------------------------------------------------------------------


class _StubTranscriptionProvider(abc.ABC):
    """Mirrors ``agent.transcription_provider.TranscriptionProvider``."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def display_name(self) -> str:
        return self.name.title()

    def is_available(self) -> bool:
        return True

    def list_models(self):
        return []

    def default_model(self):
        models = self.list_models()
        return models[0].get("id") if models else None

    def get_setup_schema(self):
        return {"name": self.display_name, "badge": "", "tag": "", "env_vars": []}

    @abc.abstractmethod
    def transcribe(self, file_path, *, model=None, language=None, **extra): ...


def _normalize_local_model(model_name):
    cloud_only = {"whisper-1", "whisper-large-v3"}
    if not model_name or model_name in cloud_only:
        return "base"
    return model_name


def _make_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    # A real spec keeps ``importlib.util.find_spec`` working against these
    # stand-ins, which is how the provider probes for optional dependencies.
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_hermes_stubs() -> None:
    agent = _make_module("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    tools = _make_module("tools")
    tools.__path__ = []  # type: ignore[attr-defined]
    hermes_cli = _make_module("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]

    def _unstubbed_transcribe_local(file_path, model_name):
        raise AssertionError("tools.transcription_tools._transcribe_local was not stubbed")

    def _unstubbed_call_llm(**kwargs):
        raise AssertionError("agent.auxiliary_client.call_llm was not stubbed")

    modules = {
        "agent": agent,
        "agent.transcription_provider": _make_module(
            "agent.transcription_provider", TranscriptionProvider=_StubTranscriptionProvider
        ),
        "agent.auxiliary_client": _make_module("agent.auxiliary_client", call_llm=_unstubbed_call_llm),
        "tools": tools,
        "tools.transcription_tools": _make_module(
            "tools.transcription_tools",
            _normalize_local_model=_normalize_local_model,
            _transcribe_local=_unstubbed_transcribe_local,
            # Hermes' binary lookup finds nothing here, so the plugin falls back
            # to shutil.which — the behaviour on a machine without Homebrew.
            _find_binary=lambda name: None,
        ),
        "hermes_cli": hermes_cli,
        "hermes_cli.config": _make_module("hermes_cli.config", load_config=lambda: {}),
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


def _load_plugin() -> types.ModuleType:
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    if "hermes_plugins" not in sys.modules:
        namespace = _make_module("hermes_plugins")
        namespace.__path__ = []  # type: ignore[attr-defined]
        sys.modules["hermes_plugins"] = namespace
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.__package__ = MODULE_NAME
    module.__path__ = [str(PLUGIN_DIR)]  # type: ignore[attr-defined]
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_install_hermes_stubs()
_PLUGIN = _load_plugin()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plugin():
    return _PLUGIN


@pytest.fixture
def config_mod():
    return sys.modules[f"{MODULE_NAME}.config"]


@pytest.fixture
def audio_mod():
    return sys.modules[f"{MODULE_NAME}.audio"]


@pytest.fixture
def backends_mod():
    return sys.modules[f"{MODULE_NAME}.backends"]


@pytest.fixture
def post_processing_mod():
    return sys.modules[f"{MODULE_NAME}.post_processing"]


@pytest.fixture
def runtime_deps_mod():
    return sys.modules[f"{MODULE_NAME}.runtime_deps"]


@pytest.fixture(autouse=True)
def _reset_backend_caches():
    backends = sys.modules[f"{MODULE_NAME}.backends"]
    backends.reset_caches()
    yield
    backends.reset_caches()


@pytest.fixture
def hermes_config(monkeypatch):
    """Set the dict returned by ``hermes_cli.config.load_config``."""

    def _set(config):
        monkeypatch.setattr(sys.modules["hermes_cli.config"], "load_config", lambda: config)
        return config

    return _set


@pytest.fixture
def local_whisper(monkeypatch):
    """Stub Hermes' faster-whisper helper; returns the recorded call list."""
    calls = []

    def _set(result):
        def _transcribe_local(file_path, model_name):
            calls.append({"file_path": file_path, "model": model_name})
            return result(file_path, model_name) if callable(result) else result

        monkeypatch.setattr(
            sys.modules["tools.transcription_tools"], "_transcribe_local", _transcribe_local
        )
        return calls

    return _set


@pytest.fixture
def call_llm(monkeypatch):
    """Stub ``agent.auxiliary_client.call_llm``; returns the recorded call list."""
    calls = []

    def _set(reply="cleaned text", error=None):
        def _call_llm(**kwargs):
            calls.append(kwargs)
            if error is not None:
                raise error
            message = types.SimpleNamespace(content=reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        monkeypatch.setattr(sys.modules["agent.auxiliary_client"], "call_llm", _call_llm)
        return calls

    return _set


class FakeStream:
    def __init__(self, transcriber):
        self._transcriber = transcriber
        self.accepted = []

    def accept_waveform(self, sample_rate, samples):
        self.accepted.append((sample_rate, len(samples)))

    @property
    def result(self):
        return types.SimpleNamespace(text=self._transcriber(self))


class FakeRecognizer:
    def __init__(self, recorder, transcriber):
        self._recorder = recorder
        self._transcriber = transcriber
        self.streams = []

    def create_stream(self):
        stream = FakeStream(self._transcriber)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        self._recorder["decoded"].append(stream)


@pytest.fixture
def fake_sherpa(monkeypatch):
    """Install a fake ``sherpa_onnx`` module and return the call recorder."""
    recorder = {"from_transducer": [], "decoded": [], "recognizers": []}

    def _set(transcriber=None, load_error=None):
        texts = transcriber or (lambda stream: "fake transcript")

        def from_transducer(**kwargs):
            recorder["from_transducer"].append(kwargs)
            if load_error is not None:
                raise load_error
            recognizer = FakeRecognizer(recorder, texts)
            recorder["recognizers"].append(recognizer)
            return recognizer

        module = _make_module(
            "sherpa_onnx",
            OfflineRecognizer=types.SimpleNamespace(from_transducer=from_transducer),
        )
        monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
        return recorder

    return _set


@pytest.fixture
def no_sherpa(monkeypatch):
    """Make ``sherpa_onnx`` unimportable, the way a rebuilt venv leaves it."""
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    return None


# ---------------------------------------------------------------------------
# Plugin-owned dependency runtime
# ---------------------------------------------------------------------------


@pytest.fixture
def install_gate(monkeypatch):
    """Stand in for Hermes' ``security.allow_lazy_installs`` gate."""

    def _set(allow=True):
        module = _make_module("tools.lazy_deps", _allow_lazy_installs=lambda: allow)
        monkeypatch.setitem(sys.modules, "tools.lazy_deps", module)
        # ``from tools import lazy_deps`` may otherwise return a package
        # attribute cached by a real Hermes import before this fixture ran.
        monkeypatch.setattr(sys.modules["tools"], "lazy_deps", module, raising=False)
        return module

    return _set


@pytest.fixture
def retired_internals(monkeypatch):
    """Poisoned stand-ins for Hermes internals the plugin must not touch.

    On current Hermes ``hermes_cli.managed_uv.resolve_uv`` is a retired updater
    shim that raises ``SystemExit``, and ``tools.lazy_deps._allow_lazy_installs``
    is gone. The stand-ins record every call; the uv one exits like the real
    shim, and the gate one answers "allowed" so consulting it cannot pass for
    honouring the policy. Neither runs an updater or installs anything.
    """
    calls = {"resolve_uv": 0, "_allow_lazy_installs": 0}

    def resolve_uv(*args, **kwargs):
        calls["resolve_uv"] += 1
        raise SystemExit("retired updater shim called")

    def _allow_lazy_installs():
        calls["_allow_lazy_installs"] += 1
        return True

    managed_uv = _make_module("hermes_cli.managed_uv", resolve_uv=resolve_uv)
    lazy_deps = _make_module("tools.lazy_deps", _allow_lazy_installs=_allow_lazy_installs)
    monkeypatch.setitem(sys.modules, "hermes_cli.managed_uv", managed_uv)
    monkeypatch.setitem(sys.modules, "tools.lazy_deps", lazy_deps)
    monkeypatch.setattr(sys.modules["hermes_cli"], "managed_uv", managed_uv, raising=False)
    monkeypatch.setattr(sys.modules["tools"], "lazy_deps", lazy_deps, raising=False)
    return calls


@pytest.fixture
def readonly_config(monkeypatch):
    """Set what ``hermes_cli.config.load_config_readonly`` returns, or raises."""

    def _set(config=None, *, error=None):
        def load_config_readonly():
            if error is not None:
                raise error
            return config

        monkeypatch.setattr(
            sys.modules["hermes_cli.config"], "load_config_readonly", load_config_readonly,
            raising=False,
        )
        return config

    return _set


@pytest.fixture(autouse=True)
def _isolate_runtime_path(monkeypatch, tmp_path_factory):
    """Keep runtime activation from leaking between tests.

    Each test gets its own runtime root, and any ``sys.path`` entry or module
    a test's runtime activated is removed afterwards — otherwise a package
    installed by one test would satisfy the next one's probe.
    """
    root = tmp_path_factory.mktemp("plugin-runtimes")
    monkeypatch.setenv("HERMES_STT_ENHANCE_RUNTIME_ROOT", str(root))
    before_path = list(sys.path)
    before_modules = set(sys.modules)
    yield root
    sys.path[:] = before_path
    for name in set(sys.modules) - before_modules:
        # Faux dependency modules imported from a test runtime.
        if name.startswith("faux"):
            sys.modules.pop(name, None)
    import importlib

    importlib.invalidate_caches()


@pytest.fixture
def faux_lock(runtime_deps_mod, monkeypatch):
    """Swap the real pins for faux packages the fake installer can materialize.

    Using names that genuinely do not exist keeps the tests honest: activation
    and imports are real, not simulated with ``sys.modules`` sentinels.
    """

    def _set(*, asr_version="1.0.0", numeric_version="2.0.0"):
        lock = (
            runtime_deps_mod.Requirement("faux-asr", asr_version, "faux_asr"),
            runtime_deps_mod.Requirement("faux-numeric", numeric_version, "faux_numeric"),
        )
        monkeypatch.setattr(runtime_deps_mod, "DEPENDENCY_LOCK", lock)
        monkeypatch.setattr(runtime_deps_mod, "REQUIRED_IMPORTS", ("faux_asr",))
        return lock

    return _set


@pytest.fixture
def fake_installer(runtime_deps_mod, monkeypatch):
    """Replace the pip/uv subprocess with one that materializes real files.

    The staged directory ends up holding importable modules and ``dist-info``
    metadata, so version verification and ``sys.path`` activation exercise the
    same code paths a real install would.
    """
    recorder = {"calls": []}

    def _set(*, ok=True, output="", versions=None, before=None):
        def _run_installer(stage, specs, timeout):
            recorder["calls"].append(
                {"stage": Path(stage), "specs": tuple(specs), "timeout": timeout}
            )
            if before is not None:
                before(Path(stage), tuple(specs))
            if not ok:
                return False, output or "installer failed"
            for spec in specs:
                dist, _, version = spec.partition("==")
                version = (versions or {}).get(dist, version)
                module = dist.replace("-", "_")
                stage_path = Path(stage)
                stage_path.mkdir(parents=True, exist_ok=True)
                (stage_path / f"{module}.py").write_text(
                    f'__version__ = "{version}"\n', encoding="utf-8"
                )
                dist_info = stage_path / f"{module}-{version}.dist-info"
                dist_info.mkdir(parents=True, exist_ok=True)
                (dist_info / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n",
                    encoding="utf-8",
                )
            return True, output or "installed"

        monkeypatch.setattr(runtime_deps_mod, "_run_installer", _run_installer)
        return recorder

    return _set


@pytest.fixture
def restart_process(runtime_deps_mod):
    """Simulate the next Hermes process: nothing activated, nothing imported.

    What survives is exactly what is on disk — which is the whole point of a
    runtime that lives outside the venv.
    """

    def _restart():
        import importlib

        root = str(runtime_deps_mod.runtime_root())
        sys.path[:] = [entry for entry in sys.path if not str(entry).startswith(root)]
        for name in list(sys.modules):
            if name.startswith("faux"):
                del sys.modules[name]
        importlib.invalidate_caches()

    return _restart


@pytest.fixture
def parakeet_model_dir(tmp_path):
    """A directory shaped like the benchmarked int8 sherpa-onnx export."""
    model_dir = tmp_path / "parakeet-v3-int8"
    model_dir.mkdir()
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (model_dir / name).write_bytes(b"stub")
    return model_dir


@pytest.fixture
def make_wav(tmp_path):
    """Write a mono 16-bit PCM WAV of the requested duration."""

    def _make(name="audio.wav", seconds=1.0, sample_rate=16000, amplitude=1000):
        path = tmp_path / name
        frames = int(seconds * sample_rate)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            # A cheap non-silent pattern keeps the file compressible-free and
            # makes sample counts observable in fake backends.
            handle.writeframes((amplitude).to_bytes(2, "little", signed=True) * frames)
        return path

    return _make


def _iter_skill_index_files(skills_dir, filename):
    """Hermes 0.15 ``agent.skill_utils.iter_skill_index_files``: sorted, no pruning of support dirs."""
    matches = [Path(root) / filename for root, _dirs, files in os.walk(skills_dir) if filename in files]
    yield from sorted(matches, key=lambda p: str(p.relative_to(skills_dir)))


def _parse_frontmatter(content):
    import re

    if not content.startswith("---"):
        return {}, content
    end = re.search(r"\n---\s*\n", content[3:])
    if not end:
        return {}, content
    meta = {}
    for line in content[3:end.start() + 3].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, content[end.end() + 3:]


@pytest.fixture
def skill_roots(monkeypatch, tmp_path):
    """Install a stand-in ``agent.skill_utils`` whose roots are the given dirs.

    Returns ``(roots, calls)``; ``calls`` counts root lookups so tests can
    assert that a path was never resolved.
    """
    roots = [tmp_path / "profile-skills", tmp_path / "external-skills"]
    for root in roots:
        root.mkdir()
    calls = []

    def _get_all_skills_dirs():
        calls.append("get_all_skills_dirs")
        return list(roots)

    module = _make_module(
        "agent.skill_utils",
        get_all_skills_dirs=_get_all_skills_dirs,
        iter_skill_index_files=_iter_skill_index_files,
        parse_frontmatter=_parse_frontmatter,
    )
    monkeypatch.setitem(sys.modules, "agent.skill_utils", module)
    return roots, calls


def write_skill(root, relative, body="Fix product names.", *, name=None, raw=None):
    """Create ``root/relative/SKILL.md``; returns its path."""
    skill_dir = Path(root) / relative
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    if raw is not None:
        path.write_bytes(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    else:
        path.write_text(
            f"---\nname: {name or skill_dir.name}\ndescription: test skill\n---\n\n{body}\n",
            encoding="utf-8",
        )
    return path


@pytest.fixture
def stat_denied(monkeypatch):
    """Make ``Path.is_dir``/``is_file`` raise ``PermissionError`` under a prefix.

    Python 3.11 (Hermes' runtime) propagates EACCES from these calls; newer
    interpreters swallow it, so the deterministic version is patched in.
    """

    def _deny(prefix, *, methods=("is_dir", "is_file")):
        prefix = str(prefix)
        for method in methods:
            original = getattr(Path, method)

            def _patched(self, *args, _original=original, **kwargs):
                if str(self).startswith(prefix):
                    raise PermissionError(13, "Permission denied", str(self))
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(Path, method, _patched)

    return _deny


@pytest.fixture
def untraversable_dir(tmp_path):
    """A real directory with mode 000; skips when permissions are not enforced."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    locked = tmp_path / "locked"
    locked.mkdir()
    write_skill(locked, "inside")
    locked.chmod(0)
    yield locked
    locked.chmod(0o700)
