"""The plugin-owned dependency runtime.

The regression behind this module: a Hermes update can recreate the agent venv
from ``pyproject.toml``, which wipes anything installed lazily into it, and
Hermes' repair pass cannot heal what a rebuild removed (it only refreshes
features whose anchor package is still present). So the parakeet backend keeps
its dependencies in a content-addressed directory *outside* the venv.

These tests cover the lifecycle, not just the happy already-installed state:
fresh install, the next process, a venv rebuild, an interpreter upgrade, a lock
upgrade, rollback when an upgrade cannot reach the index, integrity, pruning,
and multi-profile sharing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Identity: where a runtime lives and what its name means
# ---------------------------------------------------------------------------


def test_runtime_dir_is_addressed_by_interpreter_and_lock(runtime_deps_mod, faux_lock):
    lock = faux_lock()
    target = runtime_deps_mod.runtime_dir(lock)

    assert target.parent == runtime_deps_mod.runtime_root()
    assert target.name.startswith(runtime_deps_mod.environment_tag() + "-")
    assert target.name.endswith(runtime_deps_mod.lock_digest(lock))


def test_a_changed_pin_addresses_a_different_directory(runtime_deps_mod, faux_lock):
    first = runtime_deps_mod.runtime_dir(faux_lock(asr_version="1.0.0"))
    second = runtime_deps_mod.runtime_dir(faux_lock(asr_version="1.1.0"))

    # Upgrades install into a clean directory: ``pip install --target
    # --upgrade`` leaves two dist-info dirs for one package (pypa/pip#13763),
    # which makes importlib.metadata ambiguous.
    assert first != second


def test_profiles_share_one_runtime_root(runtime_deps_mod, monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_STT_ENHANCE_RUNTIME_ROOT", raising=False)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "prime"))
    prime = runtime_deps_mod.runtime_root()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "shopy"))
    shopy = runtime_deps_mod.runtime_root()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    default = runtime_deps_mod.runtime_root()

    # Four gateways on one machine download the wheels once, not four times.
    assert prime == shopy == default == tmp_path / "plugin-runtimes" / "hermes-stt-enhance"


def test_the_installer_is_pinned_to_the_running_interpreter(
    runtime_deps_mod, faux_lock, monkeypatch, tmp_path
):
    """uv resolves wheels for the interpreter it picks, not the one we mean.

    Left to itself it discovers a venv from the working directory — which is
    how a runtime tagged ``cpython-311`` could end up holding wheels built by
    some other interpreter that happened to be nearby.
    """
    lock = faux_lock()
    fake_managed_uv = type(sys)("hermes_cli.managed_uv")
    fake_managed_uv.resolve_uv = lambda: "/usr/bin/uv"
    monkeypatch.setitem(sys.modules, "hermes_cli.managed_uv", fake_managed_uv)
    monkeypatch.setattr(runtime_deps_mod.shutil, "which", lambda name: "/usr/bin/uv")

    command = runtime_deps_mod._installer_command(tmp_path, runtime_deps_mod.lock_specs(lock))

    assert command[0] == "/usr/bin/uv"
    assert "--python" in command
    assert command[command.index("--python") + 1] == sys.executable
    assert command[command.index("--target") + 1] == str(tmp_path)


def test_the_installer_falls_back_to_pip_in_the_same_interpreter(
    runtime_deps_mod, faux_lock, monkeypatch, tmp_path
):
    lock = faux_lock()
    monkeypatch.setattr(runtime_deps_mod.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.managed_uv", None)

    command = runtime_deps_mod._installer_command(tmp_path, runtime_deps_mod.lock_specs(lock))

    assert command[:4] == [sys.executable, "-m", "pip", "install"]


def test_manual_command_targets_the_plugin_runtime_not_the_hermes_venv(
    runtime_deps_mod, faux_lock
):
    lock = faux_lock()
    command = runtime_deps_mod.manual_install_command(lock)

    assert command.startswith(f"{sys.executable} -m pip install --target ")
    assert str(runtime_deps_mod.runtime_dir(lock)) in command
    assert "'faux-asr==1.0.0'" in command


# ---------------------------------------------------------------------------
# Fresh install
# ---------------------------------------------------------------------------


def test_fresh_install_provisions_activates_and_imports(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()

    assert runtime_deps_mod.ensure(auto_install=True, lock=lock) == "installed"

    target = runtime_deps_mod.runtime_dir(lock)
    assert runtime_deps_mod.verify_runtime(target, lock) is None
    assert runtime_deps_mod.missing_imports() == ()
    import faux_asr

    assert faux_asr.__version__ == "1.0.0"
    assert Path(faux_asr.__file__).parent == target
    # Installed into a staging dir, not straight into the final location.
    assert recorder["calls"][0]["stage"] != target
    assert recorder["calls"][0]["specs"] == ("faux-asr==1.0.0", "faux-numeric==2.0.0")


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits")
def test_runtime_root_is_private_even_when_it_already_existed(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    root = runtime_deps_mod.runtime_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o755)
    os.chmod(root, 0o755)
    install_gate()
    fake_installer()

    runtime_deps_mod.ensure(auto_install=True, lock=lock)

    assert root.stat().st_mode & 0o777 == 0o700


def test_the_runtime_records_what_it_installed(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    install_gate()
    fake_installer()

    runtime_deps_mod.ensure(auto_install=True, lock=lock)
    state = json.loads(
        (runtime_deps_mod.runtime_dir(lock) / "runtime.json").read_text(encoding="utf-8")
    )

    assert state["lock"] == ["faux-asr==1.0.0", "faux-numeric==2.0.0"]
    assert state["environment"] == runtime_deps_mod.environment_tag()
    assert state["versions"] == {"faux-asr": "1.0.0", "faux-numeric": "2.0.0"}
    assert state["plugin"] == "hermes-stt-enhance"


def test_the_runtime_never_shadows_hermes_core(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    install_gate()
    fake_installer()

    runtime_deps_mod.ensure(auto_install=True, lock=lock)

    # Appended, never prepended: core site-packages wins every name collision,
    # so a bad plugin dependency cannot break Hermes itself.
    assert sys.path[-1] == str(runtime_deps_mod.runtime_dir(lock))


def test_dependencies_already_provided_by_hermes_are_left_alone(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, monkeypatch
):
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()
    monkeypatch.setattr(runtime_deps_mod, "missing_imports", lambda: ())

    assert runtime_deps_mod.ensure(auto_install=True, lock=lock) == "hermes"
    assert recorder["calls"] == []


# ---------------------------------------------------------------------------
# The next process, and the venv rebuild that started all this
# ---------------------------------------------------------------------------


def test_the_next_process_activates_without_installing(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=lock)

    restart_process()

    assert runtime_deps_mod.ensure(auto_install=True, lock=lock) == "runtime"
    assert len(recorder["calls"]) == 1
    import faux_asr  # noqa: F401  — importable again purely from disk


def test_a_hermes_venv_rebuild_costs_nothing(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    """The whole point of the model.

    A rebuilt venv is a fresh interpreter with none of the lazily-installed
    packages. The plugin runtime lives outside it, so the first voice message
    after an update activates and transcribes — no install, no network.
    """
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=lock)

    restart_process()  # new process, wiped venv, same disk
    fake_installer(ok=False, output="network is unreachable")  # offline, too

    assert runtime_deps_mod.ensure(auto_install=True, lock=lock) == "runtime"
    assert len(recorder["calls"]) == 1


def test_availability_activates_an_existing_runtime_without_installing(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=lock)
    restart_process()

    assert runtime_deps_mod.can_provide(auto_install=True, lock=lock) is True
    assert len(recorder["calls"]) == 1
    assert runtime_deps_mod.missing_imports() == ()


# ---------------------------------------------------------------------------
# Upgrade, interpreter change, rollback
# ---------------------------------------------------------------------------


def test_an_interpreter_upgrade_installs_a_fresh_runtime(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process,
    monkeypatch
):
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=lock)
    old = runtime_deps_mod.runtime_dir(lock)
    restart_process()

    # A Hermes update that moves Python 3.11 → 3.12 must not import wheels
    # compiled for the old ABI.
    monkeypatch.setattr(runtime_deps_mod, "environment_tag", lambda: "cpython-312-linux-x86_64")

    assert runtime_deps_mod.ensure(auto_install=True, lock=lock) == "installed"
    assert runtime_deps_mod.runtime_dir(lock) != old
    assert old.is_dir(), "the runtime for the old interpreter is still there"
    assert len(recorder["calls"]) == 2


def test_a_lock_upgrade_installs_beside_the_old_runtime(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    old_lock = faux_lock(asr_version="1.0.0")
    install_gate()
    fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=old_lock)
    old_dir = runtime_deps_mod.runtime_dir(old_lock)
    restart_process()

    new_lock = faux_lock(asr_version="1.1.0")
    assert runtime_deps_mod.ensure(auto_install=True, lock=new_lock) == "installed"

    import faux_asr

    assert faux_asr.__version__ == "1.1.0"
    assert old_dir.is_dir(), "the superseded runtime is kept for rollback"


def test_an_offline_upgrade_rolls_back_to_the_working_runtime(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process, caplog
):
    """An upgrade that cannot reach the index must never take STT away."""
    old_lock = faux_lock(asr_version="1.0.0")
    install_gate()
    fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=old_lock)
    restart_process()

    new_lock = faux_lock(asr_version="1.1.0")
    fake_installer(ok=False, output="Temporary failure in name resolution")

    with caplog.at_level(logging.WARNING):
        assert runtime_deps_mod.ensure(auto_install=True, lock=new_lock) == "fallback-runtime"

    import faux_asr

    assert faux_asr.__version__ == "1.0.0"
    assert runtime_deps_mod.missing_imports() == ()
    assert any("falling back" in record.getMessage() for record in caplog.records)


def test_a_failed_fresh_install_says_what_to_run(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    install_gate()
    fake_installer(ok=False, output="Temporary failure in name resolution")

    with pytest.raises(runtime_deps_mod.DependencyError) as exc:
        runtime_deps_mod.ensure(auto_install=True, lock=lock)

    message = str(exc.value)
    assert "offline" in message  # the failure is classified, not just dumped
    assert runtime_deps_mod.manual_install_command(lock) in message
    assert not runtime_deps_mod.runtime_dir(lock).exists(), "no half-built runtime"


def test_pruning_keeps_one_runtime_for_rollback(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    install_gate()
    fake_installer()
    dirs = []
    for version in ("1.0.0", "1.1.0", "1.2.0", "1.3.0"):
        lock = faux_lock(asr_version=version)
        runtime_deps_mod.ensure(auto_install=True, lock=lock)
        dirs.append(runtime_deps_mod.runtime_dir(lock))
        restart_process()

    alive = [path for path in dirs if path.is_dir()]

    # Current plus one predecessor; the older two are reclaimed.
    assert alive == dirs[-2:]


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_an_interrupted_install_is_not_trusted(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    target = runtime_deps_mod.runtime_dir(lock)
    target.mkdir(parents=True)
    (target / "faux_asr.py").write_text("__version__ = 'partial'\n", encoding="utf-8")

    assert "incomplete install" in (runtime_deps_mod.verify_runtime(target, lock) or "")

    install_gate()
    recorder = fake_installer()
    assert runtime_deps_mod.ensure(auto_install=True, lock=lock) == "installed"
    assert len(recorder["calls"]) == 1

    import faux_asr

    assert faux_asr.__version__ == "1.0.0"


def test_a_tampered_or_drifted_runtime_is_reinstalled(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=lock)
    target = runtime_deps_mod.runtime_dir(lock)
    restart_process()

    # Something replaced the pinned package with a different version.
    for dist_info in target.glob("faux_asr-*.dist-info"):
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: faux-asr\nVersion: 9.9.9\n", encoding="utf-8"
        )

    assert "expected 1.0.0" in (runtime_deps_mod.verify_runtime(target, lock) or "")
    assert runtime_deps_mod.ensure(auto_install=True, lock=lock) == "installed"
    assert len(recorder["calls"]) == 2


def test_an_install_that_resolves_the_wrong_version_is_rejected(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    install_gate()
    fake_installer(versions={"faux-asr": "0.9.0"})

    with pytest.raises(runtime_deps_mod.DependencyError) as exc:
        runtime_deps_mod.ensure(auto_install=True, lock=lock)

    assert "pins 1.0.0" in str(exc.value)
    assert not runtime_deps_mod.runtime_dir(lock).exists()


def test_a_runtime_built_for_another_interpreter_is_ignored(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, monkeypatch
):
    lock = faux_lock()
    target = runtime_deps_mod.runtime_dir(lock)
    target.mkdir(parents=True)
    (target / "runtime.json").write_text(
        json.dumps({
            "schema": runtime_deps_mod.LOCK_SCHEMA,
            "lock": list(runtime_deps_mod.lock_specs(lock)),
            "environment": "cpython-39-darwin-arm64",
            "installed_at": 1.0,
        }),
        encoding="utf-8",
    )

    assert "built for" in (runtime_deps_mod.verify_runtime(target, lock) or "")


# ---------------------------------------------------------------------------
# Gating and error UX
# ---------------------------------------------------------------------------


def test_hermes_install_gate_is_honoured(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    install_gate(allow=False)
    recorder = fake_installer()

    with pytest.raises(runtime_deps_mod.DependencyError) as exc:
        runtime_deps_mod.ensure(auto_install=True, lock=lock)

    assert "allow_lazy_installs" in str(exc.value)
    assert runtime_deps_mod.manual_install_command(lock) in str(exc.value)
    assert recorder["calls"] == []
    assert runtime_deps_mod.can_provide(auto_install=True, lock=lock) is False


def test_auto_install_deps_false_freezes_the_environment(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()

    with pytest.raises(runtime_deps_mod.DependencyError) as exc:
        runtime_deps_mod.ensure(auto_install=False, lock=lock)

    assert "auto_install_deps" in str(exc.value)
    assert recorder["calls"] == []
    assert runtime_deps_mod.can_provide(auto_install=False, lock=lock) is False


def test_a_frozen_environment_still_uses_a_runtime_it_already_has(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    lock = faux_lock()
    install_gate()
    fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=lock)
    restart_process()

    # Freezing installs must not disable a runtime that is already provisioned.
    assert runtime_deps_mod.ensure(auto_install=False, lock=lock) == "runtime"
    assert runtime_deps_mod.can_provide(auto_install=False, lock=lock) is True


def test_can_provide_is_true_for_a_gap_first_use_would_fill(
    runtime_deps_mod, faux_lock, fake_installer, install_gate
):
    """The production regression.

    Hermes returns "STT plugin is not available" *instead of* calling the
    provider, so reporting False for a recoverable gap is what turned a venv
    rebuild into a dead voice pipeline.
    """
    lock = faux_lock()
    install_gate()
    recorder = fake_installer()

    assert runtime_deps_mod.can_provide(auto_install=True, lock=lock) is True
    assert recorder["calls"] == [], "probing must never install"


def test_missing_dependencies_are_never_a_silent_false(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, caplog
):
    lock = faux_lock()
    install_gate(allow=False)
    fake_installer()

    with caplog.at_level(logging.WARNING):
        runtime_deps_mod.can_provide(auto_install=True, lock=lock)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(runtime_deps_mod.manual_install_command(lock) in w for w in warnings)


# ---------------------------------------------------------------------------
# status(): what the repair script and logs report
# ---------------------------------------------------------------------------


def test_status_reports_a_missing_runtime(runtime_deps_mod, faux_lock, install_gate):
    lock = faux_lock()
    install_gate()

    state = runtime_deps_mod.status(lock)

    assert state.satisfied is False
    assert state.source == "none"
    assert state.detail == "not installed"


def test_status_reports_the_installed_runtime(
    runtime_deps_mod, faux_lock, fake_installer, install_gate, restart_process
):
    lock = faux_lock()
    install_gate()
    fake_installer()
    runtime_deps_mod.ensure(auto_install=True, lock=lock)
    restart_process()

    state = runtime_deps_mod.status(lock)

    assert state.satisfied is True
    assert state.source == "runtime"
    assert state.path == runtime_deps_mod.runtime_dir(lock)
