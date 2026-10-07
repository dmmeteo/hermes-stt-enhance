#!/usr/bin/env python3
"""Inspect, provision, or prune this plugin's dependency runtime.

Run it with the interpreter Hermes runs, so the runtime is built for the ABI
that will import it:

    ~/.hermes/hermes-agent/venv/bin/python scripts/plugin_runtime.py --status
    ~/.hermes/hermes-agent/venv/bin/python scripts/plugin_runtime.py --install

Nothing here is required in normal use — the backend provisions itself on first
use. It exists for the cases where doing it by hand is the point: pre-seeding
an air-gapped or image-baked install, checking what an update did, or
reclaiming disk after several upgrades.

``--json`` prints a machine-readable report for health checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "hermes-stt-enhance"


def _load_runtime_deps():
    """Load ``runtime_deps`` standalone (the plugin dir is not importable)."""
    spec = importlib.util.spec_from_file_location(
        "stt_enhance_runtime_deps", PLUGIN_DIR / "runtime_deps.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"Could not load {PLUGIN_DIR / 'runtime_deps.py'}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses resolves string annotations through
    # ``sys.modules[cls.__module__]``, which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolved_origins(rd) -> dict:
    """Where each dependency actually imports from, after activation."""
    import importlib.util

    origins = {}
    for req in rd.DEPENDENCY_LOCK:
        try:
            spec = importlib.util.find_spec(req.import_name)
        except Exception:
            spec = None
        origins[req.import_name] = getattr(spec, "origin", None) if spec else None
    return origins


def _report(rd) -> dict:
    state = rd.status()
    current = rd.runtime_dir()
    return {
        "plugin": rd.PLUGIN_ID,
        "python": sys.executable,
        "environment": rd.environment_tag(),
        "lock": list(rd.lock_specs()),
        "lock_digest": rd.lock_digest(),
        "runtime_root": str(rd.runtime_root()),
        "runtime_dir": str(current),
        "runtime_state": rd.verify_runtime(current) or "ok",
        "satisfied": state.satisfied,
        "source": state.source,
        "detail": state.detail,
        "installs_allowed": rd.installs_allowed(),
        "previous_runtimes": [str(path) for path, _ in rd.previous_runtimes()],
        "manual_install_command": rd.manual_install_command(),
    }


def _print_report(report: dict) -> None:
    print(f"plugin            : {report['plugin']}")
    print(f"python            : {report['python']}")
    print(f"environment       : {report['environment']}")
    print(f"lock              : {' '.join(report['lock'])}")
    print(f"runtime dir       : {report['runtime_dir']}")
    print(f"runtime state     : {report['runtime_state']}")
    print(f"resolves from     : {report['source']} ({report['detail']})")
    print(f"installs allowed  : {report['installs_allowed']}")
    for path in report["previous_runtimes"]:
        print(f"previous runtime  : {path}")
    if not report["satisfied"]:
        print(f"\nprovision with    : {report['manual_install_command']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true",
                        help="report where the dependencies resolve from (default)")
    action.add_argument("--install", action="store_true",
                        help="provision the runtime for this interpreter now")
    action.add_argument("--prune", action="store_true",
                        help="delete superseded runtimes for this interpreter")
    action.add_argument("--verify", action="store_true",
                        help="activate an existing runtime (never installs) and "
                             "report where each dependency resolves from")
    parser.add_argument("--keep", type=int, default=1, metavar="N",
                        help="runtimes to keep for rollback when pruning (default 1)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.json else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    rd = _load_runtime_deps()

    if args.install:
        try:
            target = rd.install_runtime()
        except rd.DependencyError as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            else:
                print(f"install failed: {exc}", file=sys.stderr)
            return 1
        if not args.json:
            print(f"runtime ready: {target}")
    elif args.prune:
        removed = rd.prune_runtimes(keep=args.keep)
        if not args.json:
            for path in removed:
                print(f"removed {path}")
            if not removed:
                print("nothing to prune")

    report = _report(rd)
    if args.verify:
        # auto_install=False: whatever this reports, no install could have run.
        try:
            report["resolution"] = rd.ensure(auto_install=False)
        except rd.DependencyError as exc:
            report["resolution"] = "unavailable"
            report["error"] = str(exc)
        report["origins"] = _resolved_origins(rd)
        report["satisfied"] = report["resolution"] != "unavailable"

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
        for name, origin in (report.get("origins") or {}).items():
            print(f"imports {name:<14}: {origin or 'NOT IMPORTABLE'}")
        if report.get("error"):
            print(f"\n{report['error']}")
    return 0 if report["satisfied"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
