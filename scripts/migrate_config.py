#!/usr/bin/env python3
"""Rename a profile's config.yaml from ``local-llm-polished`` to ``hermes-stt-enhance``.

0.5.0 renamed the plugin, its provider and its config block. This rewrites only
the keys that carry the old identity and leaves every value and every other key
alone:

    stt.provider: local_llm_polished      -> stt_enhance
    stt.local_llm_polished                -> stt.stt_enhance
      .polish / .repair (prototype names) -> .post_processing
    auxiliary.stt_polish                  -> auxiliary.stt_enhance
    plugins.enabled / plugins.disabled    local-llm-polished -> hermes-stt-enhance
    plugins.entries.<old key>             -> plugins.entries.hermes-stt-enhance

It refuses rather than guesses when an old and a new key both exist with
different values. Running it twice is a no-op; ``--rollback`` applies the
inverse identity mapping (prototype stage names are not restored — 0.4.0 reads
``post_processing`` too).

    python scripts/migrate_config.py ~/.hermes/config.yaml --check
    python scripts/migrate_config.py ~/.hermes/config.yaml           # writes, keeps a backup
    python scripts/migrate_config.py ~/.hermes/config.yaml --rollback

Run it with Hermes' interpreter: the file is written through Hermes' own
comment-preserving ``atomic_config_write``.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

OLD_PROVIDER, NEW_PROVIDER = "local_llm_polished", "stt_enhance"
OLD_TASK, NEW_TASK = "stt_polish", "stt_enhance"
OLD_PLUGIN, NEW_PLUGIN = "local-llm-polished", "hermes-stt-enhance"
# A git install of the old repo registered the plugin as ``<repo>/<subdir>``.
OLD_PLUGIN_KEYS = (OLD_PLUGIN, f"hermes-{OLD_PLUGIN}/{OLD_PLUGIN}")
PROTOTYPE_STAGE_KEYS = ("polish", "repair")


class MigrationConflict(ValueError):
    """An old and a new key both exist and disagree."""


def _move_key(parent: Dict[str, Any], old: str, new: str, where: str, changes: List[str]) -> None:
    if old not in parent:
        return
    value = parent.pop(old)
    if new in parent and parent[new] != value:
        raise MigrationConflict(f"{where}.{old} and {where}.{new} both exist with different values")
    parent.setdefault(new, value)
    changes.append(f"{where}.{old} -> {where}.{new}")


def _rename_list_items(items: Any, old_keys: Tuple[str, ...], new: str, where: str, changes: List[str]) -> Any:
    if not isinstance(items, list) or not any(item in old_keys for item in items):
        return items
    renamed: List[Any] = []
    for item in items:
        item = new if item in old_keys else item
        if item not in renamed:
            renamed.append(item)
    changes.append(f"{where}: {'/'.join(k for k in old_keys if k in items)} -> {new}")
    return renamed


def _merge_plugin_entries(entries: Dict[str, Any], old_keys: Tuple[str, ...], new: str, changes: List[str]) -> None:
    present = [key for key in old_keys if key in entries]
    if not present:
        return
    values = [entries.pop(key) for key in present]
    if new in entries:
        values.append(entries[new])
    if any(value != values[0] for value in values):
        raise MigrationConflict(
            f"plugins.entries {', '.join(present + ([new] if new in entries else []))} differ; merge them by hand"
        )
    entries[new] = values[0]
    changes.append(f"plugins.entries.{'/'.join(present)} -> plugins.entries.{new}")


def _migrate_stage_names(block: Dict[str, Any], changes: List[str]) -> None:
    present = [key for key in PROTOTYPE_STAGE_KEYS if isinstance(block.get(key), dict)]
    if not present:
        return
    if "post_processing" in block or len(present) > 1:
        raise MigrationConflict(
            f"stt.{NEW_PROVIDER} has {', '.join(present)} next to another post-processing block; keep one by hand"
        )
    block["post_processing"] = block.pop(present[0])
    changes.append(f"stt.{NEW_PROVIDER}.{present[0]} -> stt.{NEW_PROVIDER}.post_processing")


def migrate(config: Dict[str, Any], *, rollback: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """Return ``(migrated_copy, changes)``; ``changes`` is empty when nothing applies."""
    data = copy.deepcopy(config)
    changes: List[str] = []
    src_provider, dst_provider = (NEW_PROVIDER, OLD_PROVIDER) if rollback else (OLD_PROVIDER, NEW_PROVIDER)
    src_task, dst_task = (NEW_TASK, OLD_TASK) if rollback else (OLD_TASK, NEW_TASK)
    src_plugins, dst_plugin = ((NEW_PLUGIN,), OLD_PLUGIN) if rollback else (OLD_PLUGIN_KEYS, NEW_PLUGIN)

    stt = data.get("stt")
    if isinstance(stt, dict):
        if stt.get("provider") == src_provider:
            stt["provider"] = dst_provider
            changes.append(f"stt.provider: {src_provider} -> {dst_provider}")
        _move_key(stt, src_provider, dst_provider, "stt", changes)
        if not rollback and isinstance(stt.get(NEW_PROVIDER), dict):
            _migrate_stage_names(stt[NEW_PROVIDER], changes)

    auxiliary = data.get("auxiliary")
    if isinstance(auxiliary, dict):
        _move_key(auxiliary, src_task, dst_task, "auxiliary", changes)

    plugins = data.get("plugins")
    if isinstance(plugins, dict):
        for key in ("enabled", "disabled"):
            if key in plugins:
                plugins[key] = _rename_list_items(plugins[key], src_plugins, dst_plugin, f"plugins.{key}", changes)
        if isinstance(plugins.get("entries"), dict):
            _merge_plugin_entries(plugins["entries"], src_plugins, dst_plugin, changes)

    return data, changes


def _backup(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.pre-stt-enhance-{time.strftime('%Y%m%dT%H%M%S')}")
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)
    return backup


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="path to a profile's config.yaml")
    parser.add_argument("--check", action="store_true", help="report what would change; exit 1 if anything would")
    parser.add_argument("--rollback", action="store_true", help="apply the inverse mapping")
    args = parser.parse_args(argv)

    from hermes_cli.config import atomic_config_write, read_user_config_raw

    path = args.config.expanduser().resolve()
    migrated, changes = migrate(read_user_config_raw(path), rollback=args.rollback)
    for change in changes:
        print(f"{path}: {change}")
    if not changes:
        print(f"{path}: nothing to migrate")
        return 0
    if args.check:
        return 1

    mode = stat.S_IMODE(path.stat().st_mode)
    print(f"{path}: backup at {_backup(path)}")
    atomic_config_write(path, migrated)
    os.chmod(path, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
