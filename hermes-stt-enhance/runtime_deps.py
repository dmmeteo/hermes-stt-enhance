"""Plugin-owned runtime for the parakeet backend's optional dependencies.

Why this exists
---------------

``sherpa-onnx`` and ``numpy`` are not Hermes core dependencies. Hermes installs
packages like these lazily *into its own venv*, and a Hermes update can
recreate that venv from ``pyproject.toml`` — which wipes everything installed
lazily. Hermes' repair pass (``lazy_deps.refresh_active_features``) only heals
features whose anchor package is still present, which is never true after a
rebuild, so the packages stay gone and STT dies with a generic "not available".

So this plugin does not keep its dependencies in Hermes' venv at all. It owns a
runtime directory *outside* the venv, which a venv rebuild cannot touch:

    ~/.hermes/plugin-runtimes/hermes-stt-enhance/<env-tag>-<lock-digest>/

The directory is **content-addressed**: its name carries the interpreter ABI,
the platform, and a digest of the exact pinned dependency lock. That single
property gives most of the lifecycle for free.

* **Update-safe** — a Hermes venv rebuild leaves the directory untouched, so
  the next voice message installs nothing.
* **Upgrade** — bumping :data:`DEPENDENCY_LOCK` changes the digest, so the new
  set installs into a *new, empty* directory. ``pip install --target`` cannot
  upgrade in place without leaving two ``dist-info`` dirs for one package
  (pypa/pip#13763), which makes ``importlib.metadata`` ambiguous; a clean
  directory per lock sidesteps that entirely.
* **Rollback** — the previous directory is still there. If an install fails
  (offline, PyPI outage, a bad pin), a previous runtime that still verifies is
  activated instead, so an upgrade attempt can never take working STT away.
* **Interpreter changes** — a Python upgrade changes the ABI tag, so compiled
  wheels built for the old interpreter are never imported by the new one.
* **Integrity** — installs stage into a temp directory and land atomically;
  the completion marker (``runtime.json``) is written last and records the
  exact lock and the versions actually installed. A directory whose recorded
  versions do not match the lock is not used.
* **Multi-profile** — the root is resolved above ``profiles/<name>/``, so every
  profile on this machine shares one install instead of four.

The directory is appended to ``sys.path``, never prepended, so anything Hermes
core ships always wins the name. The worst a bad install here can do is fail to
import and report the backend unavailable — it cannot shadow or break core.
This mirrors Hermes' own durable lazy-install target
(``tools.lazy_deps._activate_target_on_syspath``).

Runtime installs stay gated by ``security.allow_lazy_installs`` and by this
plugin's ``stt.stt_enhance.parakeet.auto_install_deps``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """One exactly-pinned dependency."""

    dist: str
    version: str
    import_name: str

    @property
    def spec(self) -> str:
        return f"{self.dist}=={self.version}"


# Bump ``LOCK_SCHEMA`` when the *shape* of the runtime changes (not the pins);
# it is part of the digest, so a schema change re-installs rather than reusing
# a directory laid out by an older plugin version.
LOCK_SCHEMA = 1

# Exact pins, matching the versions Hermes itself pins for sherpa-onnx/numpy
# (``tools/lazy_deps.py``: ``wake.sherpa``). Exact rather than ranged so the
# runtime directory is reproducible and its digest is meaningful.
DEPENDENCY_LOCK: Tuple[Requirement, ...] = (
    Requirement("sherpa-onnx", "1.13.4", "sherpa_onnx"),
    Requirement("numpy", "2.4.3", "numpy"),
)

# The import the backend genuinely cannot run without. numpy has a stdlib
# fallback in audio.py, so it is installed but not part of the go/no-go probe.
REQUIRED_IMPORTS: Tuple[str, ...] = ("sherpa_onnx",)

PLUGIN_ID = "hermes-stt-enhance"
RUNTIME_STATE_FILE = "runtime.json"
RUNTIME_ROOT_ENV = "HERMES_STT_ENHANCE_RUNTIME_ROOT"

# A voice message is already waiting on this, so it is bounded well below the
# 300 s Hermes uses for background lazy installs.
INSTALL_TIMEOUT_SECONDS = 240
# Concurrent gateways (one per profile) would otherwise each download the same
# wheels after an update.
INSTALL_LOCK_TIMEOUT_SECONDS = 300
# How many superseded runtimes to keep for the same interpreter, for rollback.
KEEP_PREVIOUS_RUNTIMES = 1


class DependencyError(RuntimeError):
    """The parakeet backend's dependencies are missing and cannot be provided."""


# ---------------------------------------------------------------------------
# Identity: where the runtime lives
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    """Hermes' home for this process, preferring Hermes' own resolution."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        raw = os.environ.get("HERMES_HOME", "").strip()
        return Path(raw) if raw else Path.home() / ".hermes"


def runtime_root() -> Path:
    """Root holding this plugin's runtimes, shared by every profile.

    Named profiles live in ``<hermes home>/profiles/<name>``; resolving above
    that means four gateways share one install instead of downloading the same
    wheels four times.
    """
    override = os.environ.get(RUNTIME_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    home = _hermes_home()
    if home.parent.name == "profiles":
        home = home.parent.parent
    return home / "plugin-runtimes" / PLUGIN_ID


def environment_tag() -> str:
    """Interpreter + platform identity of packages that may hold C extensions."""
    abi = sysconfig.get_config_var("SOABI") or ""
    if abi:
        # e.g. "cpython-311-x86_64-linux-gnu" → "cpython-311"
        parts = abi.split("-")
        abi = "-".join(parts[:2]) if len(parts) >= 2 else abi
    else:  # pragma: no cover - interpreters without SOABI
        abi = f"{platform.python_implementation().lower()}-{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine() or "unknown"
    return f"{abi}-{sys.platform}-{machine}".replace(" ", "_")


def lock_specs(lock: Optional[Sequence[Requirement]] = None) -> Tuple[str, ...]:
    return tuple(req.spec for req in (lock if lock is not None else DEPENDENCY_LOCK))


def lock_digest(lock: Optional[Sequence[Requirement]] = None) -> str:
    payload = f"schema={LOCK_SCHEMA}\n" + "\n".join(sorted(lock_specs(lock)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def runtime_dir(lock: Optional[Sequence[Requirement]] = None) -> Path:
    """The directory this exact lock installs into on this exact interpreter."""
    return runtime_root() / f"{environment_tag()}-{lock_digest(lock)}"


def manual_install_command(lock: Optional[Sequence[Requirement]] = None) -> str:
    """The exact command that provisions the runtime by hand.

    Uses ``sys.executable`` (the Hermes venv python, which is usually not the
    first Python on PATH) and the plugin's own ``--target``, so it never
    touches Hermes' venv. Add ``--no-index --find-links <dir>`` for air-gapped
    installs.
    """
    specs = " ".join(f"'{spec}'" for spec in lock_specs(lock))
    return (
        f"{sys.executable} -m pip install --target '{runtime_dir(lock)}' {specs}"
    )


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def module_available(name: str) -> bool:
    """Is ``name`` importable right now? Never raises."""
    module = sys.modules.get(name, False)
    if module is not False:
        # ``None`` is the "blocked import" sentinel Hermes and tests both use.
        return module is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
    except Exception:
        return False


def missing_imports() -> Tuple[str, ...]:
    return tuple(name for name in REQUIRED_IMPORTS if not module_available(name))


def _installed_versions(target: Path) -> Dict[str, str]:
    """Map ``dist name → version`` for distributions inside ``target``."""
    versions: Dict[str, str] = {}
    try:
        from importlib.metadata import distributions

        for dist in distributions(path=[str(target)]):
            try:
                name = (dist.metadata["Name"] or "").strip()
            except Exception:
                name = ""
            if name and dist.version:
                versions[name.lower().replace("_", "-")] = dist.version
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not read distributions from %s: %s", target, exc)
    return versions


def read_runtime_state(target: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(target / RUNTIME_STATE_FILE, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def verify_runtime(
    target: Path, lock: Optional[Sequence[Requirement]] = None
) -> Optional[str]:
    """Return None when ``target`` is a complete runtime, else why it is not.

    The state file is written last and atomically, so an interrupted install
    leaves a directory that fails here rather than one that half-works.
    """
    if not target.is_dir():
        return "not installed"
    state = read_runtime_state(target)
    if state is None:
        return "incomplete install (no runtime.json)"
    if state.get("schema") != LOCK_SCHEMA:
        return f"built by a different runtime schema ({state.get('schema')!r})"
    if state.get("environment") != environment_tag():
        return (
            f"built for {state.get('environment')!r}, running {environment_tag()!r}"
        )
    wanted = list(lock if lock is not None else DEPENDENCY_LOCK)
    if sorted(state.get("lock") or []) != sorted(lock_specs(wanted)):
        return "recorded lock does not match this plugin's lock"

    # Integrity: the versions actually on disk must still be the pinned ones.
    on_disk = _installed_versions(target)
    for req in wanted:
        key = req.dist.lower().replace("_", "-")
        found = on_disk.get(key)
        if found is None:
            return f"{req.dist} is missing from the runtime"
        if found != req.version:
            return f"{req.dist} is {found}, expected {req.version}"
    return None


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def activate(target: Path) -> None:
    """Append ``target`` to ``sys.path`` so its packages import.

    Appended to the END, never prepended, so Hermes' own site-packages wins
    every collision — the same guarantee Hermes gives its durable lazy target.
    ``site.addsitedir`` is used so ``.pth`` files inside the runtime are
    honoured, then the ordering is re-enforced. Idempotent.
    """
    target_str = str(target)
    before = list(sys.path)
    if target_str not in before:
        site.addsitedir(target_str)
    new_entries = [entry for entry in sys.path if entry not in before]
    if new_entries:
        sys.path[:] = [e for e in sys.path if e not in new_entries] + new_entries
    importlib.invalidate_caches()


def activate_existing(lock: Optional[Sequence[Requirement]] = None) -> Optional[Path]:
    """Activate an already-installed runtime if there is a usable one.

    Never installs. Prefers the runtime for the current lock, then falls back
    to a superseded-but-verifiable one for this interpreter, which is what
    keeps a failed upgrade from taking STT away.
    """
    current = runtime_dir(lock)
    if verify_runtime(current, lock) is None:
        activate(current)
        return current
    previous = previous_runtimes(lock)
    for candidate, candidate_lock in previous:
        if _candidate_usable(candidate, candidate_lock):
            activate(candidate)
            return candidate
    return None


def _candidate_usable(target: Path, recorded_lock: Sequence[str]) -> bool:
    """Is a superseded runtime still safe to fall back to?

    Its own recorded lock must verify (right interpreter, versions intact) and
    it must actually contain the import the backend needs.
    """
    state = read_runtime_state(target)
    if state is None or state.get("environment") != environment_tag():
        return False
    on_disk = _installed_versions(target)
    for spec in recorded_lock:
        dist, _, version = spec.partition("==")
        key = dist.strip().lower().replace("_", "-")
        if not version or on_disk.get(key) != version:
            return False
    return any((target / name).exists() or (target / f"{name}.py").exists()
               for name in REQUIRED_IMPORTS)


def previous_runtimes(
    lock: Optional[Sequence[Requirement]] = None,
) -> List[Tuple[Path, Tuple[str, ...]]]:
    """Superseded runtimes for this interpreter, newest first."""
    root = runtime_root()
    current = runtime_dir(lock)
    prefix = f"{environment_tag()}-"
    found: List[Tuple[float, Path, Tuple[str, ...]]] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir() or child == current or not child.name.startswith(prefix):
            continue
        state = read_runtime_state(child)
        if state is None:
            continue
        recorded = tuple(str(spec) for spec in (state.get("lock") or []))
        found.append((float(state.get("installed_at") or 0.0), child, recorded))
    found.sort(key=lambda item: item[0], reverse=True)
    return [(path, recorded) for _, path, recorded in found]


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------


def installs_allowed() -> bool:
    """Best-effort read of Hermes' runtime-install gate.

    There is no public query for it, so an unreadable gate is treated as open;
    this only decides whether the plugin calls the backend recoverable.
    """
    try:
        from tools import lazy_deps
    except Exception:
        return True  # not running under Hermes — nothing gates us
    probe = getattr(lazy_deps, "_allow_lazy_installs", None)
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:
        return True


def _installer_env() -> Dict[str, str]:
    try:
        from tools.environments.local import hermes_subprocess_env

        env = dict(hermes_subprocess_env(inherit_credentials=False))
    except Exception:
        env = dict(os.environ)
    # --target installs must not pick up the user site dir or a stray
    # PYTHONPATH; the runtime has to be self-contained to stay reproducible.
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _installer_command(stage: Path, specs: Sequence[str]) -> List[str]:
    """Prefer uv (fast, no pip needed in the venv), fall back to pip."""
    uv_bin = None
    try:
        from hermes_cli.managed_uv import resolve_uv

        uv_bin = resolve_uv()
    except Exception:
        uv_bin = None
    uv_bin = uv_bin or shutil.which("uv")
    if uv_bin:
        # ``--python`` is not optional: uv otherwise discovers an interpreter
        # from the working directory, and it resolves wheels for whichever one
        # it picked. The runtime directory is named after *this* interpreter's
        # ABI, so it must be filled by this interpreter.
        return [
            str(uv_bin), "pip", "install",
            "--python", sys.executable,
            "--target", str(stage),
            *specs,
        ]
    return [
        sys.executable, "-m", "pip", "install",
        "--target", str(stage), "--no-warn-script-location", *specs,
    ]


def _run_installer(stage: Path, specs: Sequence[str], timeout: int) -> Tuple[bool, str]:
    """Run the install into ``stage``. Returns ``(ok, output)``. Never raises."""
    command = _installer_command(stage, specs)
    logger.info("Installing %s into %s", " ".join(specs), stage)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_installer_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"install timed out after {timeout}s"
    except OSError as exc:
        return False, f"could not run {command[0]}: {exc}"
    if result.returncode != 0:
        return False, ((result.stderr or result.stdout or "").strip())[-1500:]
    return True, (result.stdout or "").strip()[-500:]


def _looks_offline(output: str) -> bool:
    lowered = (output or "").lower()
    return any(
        marker in lowered
        for marker in (
            "temporary failure in name resolution", "network is unreachable",
            "failed to establish a new connection", "connection refused",
            "no matching distribution", "could not resolve host",
            "connection timed out", "proxy", "ssl",
        )
    )


class _InstallLock:
    """Cross-process lock so gateways don't all download the same wheels.

    Best effort: if locking is unavailable the install still runs, and the
    atomic directory swap keeps concurrent installs correct — they just
    duplicate work.
    """

    def __init__(self, path: Path, timeout: float) -> None:
        self._path = path
        self._timeout = timeout
        self._handle = None

    def __enter__(self) -> "_InstallLock":
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self._path, "a+")
        except OSError:
            self._handle = None
            return self
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            return self
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Timed out waiting for the plugin runtime install lock "
                        "at %s; proceeding anyway", self._path,
                    )
                    return self
                time.sleep(0.5)

    def __exit__(self, *exc_info) -> None:
        if self._handle is None:
            return
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._handle.close()
        except OSError:
            pass


def install_runtime(lock: Optional[Sequence[Requirement]] = None) -> Path:
    """Install the locked dependencies into their runtime directory.

    Stages into a temporary directory beside the target and lands it with a
    single rename, so a crashed or offline install never leaves a half-built
    runtime that a later run would trust.

    Raises:
        DependencyError: with actionable detail, on any failure.
    """
    requirements = list(lock if lock is not None else DEPENDENCY_LOCK)
    target = runtime_dir(requirements)
    root = target.parent
    specs = lock_specs(requirements)

    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # ``mkdir(mode=...)`` is filtered by umask and does not tighten a
        # pre-existing directory. This runtime holds executable third-party
        # code, so make the ownership boundary explicit on every install.
        if os.name == "posix":
            os.chmod(root, 0o700)
    except OSError as exc:
        raise DependencyError(
            f"Cannot create the plugin runtime directory {root}: {exc}. "
            f"Set {RUNTIME_ROOT_ENV} to a writable path, or install manually: "
            f"{manual_install_command(requirements)}"
        ) from exc

    with _InstallLock(root / ".install.lock", INSTALL_LOCK_TIMEOUT_SECONDS):
        # Another process may have finished while we waited for the lock.
        if verify_runtime(target, requirements) is None:
            return target
        if target.exists():
            # An earlier attempt left something unusable — replace it wholesale
            # rather than installing over it (pypa/pip#13763).
            shutil.rmtree(target, ignore_errors=True)

        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(root)))
        try:
            ok, output = _run_installer(stage, specs, INSTALL_TIMEOUT_SECONDS)
            if not ok:
                hint = (
                    "the machine looks offline or the index is unreachable"
                    if _looks_offline(output)
                    else "the installer failed"
                )
                raise DependencyError(
                    f"Installing the parakeet runtime failed — {hint}: {output}. "
                    f"Install it manually with: {manual_install_command(requirements)}"
                )

            installed = _installed_versions(stage)
            for req in requirements:
                key = req.dist.lower().replace("_", "-")
                found = installed.get(key)
                if found != req.version:
                    raise DependencyError(
                        f"Installed {req.dist} {found or 'nothing'} but the lock "
                        f"pins {req.version}; refusing to use this runtime. "
                        f"Install manually with: {manual_install_command(requirements)}"
                    )

            state = {
                "schema": LOCK_SCHEMA,
                "plugin": PLUGIN_ID,
                "lock": list(specs),
                "environment": environment_tag(),
                "python": platform.python_version(),
                "executable": sys.executable,
                "versions": {req.dist: req.version for req in requirements},
                "installed_at": time.time(),
            }
            # Written last: its presence is what makes the runtime trusted.
            state_path = stage / RUNTIME_STATE_FILE
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())

            try:
                os.rename(str(stage), str(target))
            except OSError as exc:
                if verify_runtime(target, requirements) is None:
                    return target  # lost a benign race; the winner is valid
                raise DependencyError(
                    f"Could not put the parakeet runtime in place at {target}: "
                    f"{exc}. Install manually with: "
                    f"{manual_install_command(requirements)}"
                ) from exc
            stage = None  # type: ignore[assignment]  # renamed, nothing to clean
        finally:
            if stage is not None and Path(stage).exists():
                shutil.rmtree(stage, ignore_errors=True)

    logger.info("Parakeet runtime installed: %s", target)
    prune_runtimes(requirements)
    return target


def prune_runtimes(
    lock: Optional[Sequence[Requirement]] = None, *, keep: int = KEEP_PREVIOUS_RUNTIMES
) -> List[Path]:
    """Delete superseded runtimes beyond ``keep``, newest kept. Never raises."""
    removed: List[Path] = []
    try:
        stale = previous_runtimes(lock)[keep:]
    except Exception:  # pragma: no cover - defensive
        return removed
    for path, _ in stale:
        try:
            shutil.rmtree(path)
            removed.append(path)
            logger.info("Removed superseded parakeet runtime %s", path)
        except OSError as exc:
            logger.debug("Could not remove %s: %s", path, exc)
    return removed


# ---------------------------------------------------------------------------
# The two entry points the backend uses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeStatus:
    """Where the parakeet dependencies are coming from (or why they are not)."""

    satisfied: bool
    source: str  # "hermes" | "runtime" | "fallback-runtime" | "none"
    path: Optional[Path]
    detail: str

    def describe(self) -> str:
        where = f" ({self.path})" if self.path else ""
        return f"{self.source}{where}: {self.detail}"


def status(lock: Optional[Sequence[Requirement]] = None) -> RuntimeStatus:
    """Non-installing description of the current dependency situation."""
    if not missing_imports():
        return RuntimeStatus(True, "hermes", None, "already importable")

    current = runtime_dir(lock)
    reason = verify_runtime(current, lock)
    if reason is None:
        return RuntimeStatus(True, "runtime", current, "installed and verified")

    for candidate, recorded in previous_runtimes(lock):
        if _candidate_usable(candidate, recorded):
            return RuntimeStatus(
                True, "fallback-runtime", candidate,
                "a superseded runtime is available for rollback",
            )
    return RuntimeStatus(False, "none", current, reason)


def can_provide(*, auto_install: bool, lock: Optional[Sequence[Requirement]] = None) -> bool:
    """Can the backend run — now, or after a first-use install?

    This is the probe behind ``is_available()``. It activates an existing
    runtime (cheap, idempotent) but never installs, because Hermes calls
    availability from pickers and setup screens.

    It reports True for a dependency that first use would install: Hermes
    refuses to call a provider that reports False, so answering "no" to a
    recoverable gap is what turns a venv rebuild into a dead pipeline.
    """
    if not missing_imports():
        return True

    activated = activate_existing(lock)
    if activated is not None and not missing_imports():
        logger.info("parakeet backend: using plugin runtime %s", activated)
        return True

    command = manual_install_command(lock)
    if not auto_install:
        logger.warning(
            "parakeet backend: %s is unavailable and "
            "stt.stt_enhance.parakeet.auto_install_deps is false. "
            "Provision it with: %s",
            ", ".join(missing_imports()), command,
        )
        return False
    if not installs_allowed():
        logger.warning(
            "parakeet backend: %s is unavailable and Hermes runtime installs "
            "are turned off (security.allow_lazy_installs). Provision it "
            "with: %s",
            ", ".join(missing_imports()), command,
        )
        return False

    logger.warning(
        "parakeet backend: %s is unavailable — installing the plugin runtime "
        "on the next transcription. To do it now, run: %s",
        ", ".join(missing_imports()), command,
    )
    return True


def ensure(*, auto_install: bool, lock: Optional[Sequence[Requirement]] = None) -> str:
    """Make the parakeet dependencies importable. Returns what happened.

    One of ``"hermes"`` (core already provides them), ``"runtime"`` (an
    existing plugin runtime was activated), ``"fallback-runtime"`` (an install
    failed and a superseded runtime was used instead), or ``"installed"``.

    Raises:
        DependencyError: with the exact provisioning command, when the
            dependencies cannot be made importable.
    """
    if not missing_imports():
        return "hermes"

    current = runtime_dir(lock)
    if verify_runtime(current, lock) is None:
        activate(current)
        if not missing_imports():
            logger.info("parakeet backend: activated plugin runtime %s", current)
            return "runtime"

    command = manual_install_command(lock)
    if not auto_install:
        fallback = _use_fallback(lock)
        if fallback is not None:
            return "fallback-runtime"
        raise DependencyError(
            "The parakeet backend's dependencies are not installed and "
            "stt.stt_enhance.parakeet.auto_install_deps is false. "
            f"Provision them with: {command}"
        )
    if not installs_allowed():
        fallback = _use_fallback(lock)
        if fallback is not None:
            return "fallback-runtime"
        raise DependencyError(
            "The parakeet backend's dependencies are not installed and Hermes "
            "runtime installs are turned off (security.allow_lazy_installs). "
            f"Provision them with: {command}"
        )

    logger.warning(
        "parakeet backend: provisioning the plugin runtime (%s) — a Hermes "
        "venv rebuild or an updated dependency lock needs one install",
        ", ".join(lock_specs(lock)),
    )
    try:
        target = install_runtime(lock)
    except DependencyError as exc:
        # An upgrade that cannot reach the index must not take working STT
        # away: fall back to whatever verified runtime is still on disk.
        fallback = _use_fallback(lock, because=str(exc))
        if fallback is not None:
            return "fallback-runtime"
        raise
    except Exception as exc:  # defensive: install_runtime owns its failures
        fallback = _use_fallback(lock, because=str(exc))
        if fallback is not None:
            return "fallback-runtime"
        raise DependencyError(
            f"Installing the parakeet runtime failed: {exc}. "
            f"Provision it with: {command}"
        ) from exc

    activate(target)
    still_missing = missing_imports()
    if still_missing:
        raise DependencyError(
            f"The parakeet runtime installed into {target} but "
            f"{', '.join(still_missing)} is still not importable. "
            f"Provision it with: {command}"
        )
    return "installed"


def _use_fallback(
    lock: Optional[Sequence[Requirement]] = None, *, because: str = ""
) -> Optional[Path]:
    """Activate a superseded runtime, if one still verifies. Never raises."""
    for candidate, recorded in previous_runtimes(lock):
        if not _candidate_usable(candidate, recorded):
            continue
        activate(candidate)
        if missing_imports():
            continue
        logger.warning(
            "parakeet backend: falling back to the previous plugin runtime %s"
            "%s. Transcription keeps working; the pinned upgrade has not been "
            "applied.",
            candidate, f" ({because})" if because else "",
        )
        return candidate
    return None
