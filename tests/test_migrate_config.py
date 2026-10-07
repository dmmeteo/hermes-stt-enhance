"""scripts/migrate_config.py: the 0.4.0 -> 0.5.0 identity rename of a profile config."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_config.py"
_spec = importlib.util.spec_from_file_location("migrate_config", _SCRIPT)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

POST_PROCESSING = {
    "enabled": True,
    "provider": "openrouter",
    "model": "vendor/some-model",
    "reasoning_effort": "low",
    "timeout": 45,
    "prompt": "Keep the speaker's words.",
}


def _legacy_config():
    """The shape the live profiles had on 0.4.0, including per-engine legacy fields."""
    return {
        "model": {"default": "main-model"},
        "stt": {
            "enabled": True,
            "local": {"model": "base", "language": "uk"},
            "provider": "local_llm_polished",
            "local_llm_polished": {
                "model": "base",
                "language": "uk",
                "audio_speed": 1.0,
                "backend": "parakeet",
                "parakeet": {"model_path": "/models/parakeet", "num_threads": 6, "audio_speed": 1.0},
                "chunking": {"enabled": True, "threshold_seconds": 40, "chunk_seconds": 40},
                "post_processing": dict(POST_PROCESSING),
            },
        },
        "auxiliary": {"vision": {"provider": "auto"}},
        "plugins": {
            "enabled": ["other-plugin", "local-llm-polished", "hindsight"],
            "disabled": ["stt-repair"],
            "entries": {
                "local-llm-polished": {"allow_tool_override": False},
                "hermes-local-llm-polished/local-llm-polished": {"allow_tool_override": False},
                "other-plugin": {"x": 1},
            },
        },
    }


def test_renames_only_the_identity_keys():
    old = _legacy_config()

    new, changes = mc.migrate(old)

    assert new["stt"]["provider"] == "stt_enhance"
    assert "local_llm_polished" not in new["stt"]
    assert new["stt"]["stt_enhance"] == old["stt"]["local_llm_polished"]
    assert new["plugins"]["enabled"] == ["other-plugin", "hermes-stt-enhance", "hindsight"]
    assert new["plugins"]["disabled"] == ["stt-repair"]
    assert new["plugins"]["entries"] == {
        "other-plugin": {"x": 1},
        "hermes-stt-enhance": {"allow_tool_override": False},
    }
    for unrelated in ("model", "auxiliary"):
        assert new[unrelated] == old[unrelated]
    assert {k: v for k, v in new["stt"].items() if k not in ("provider", "stt_enhance")} == {
        k: v for k, v in old["stt"].items() if k not in ("provider", "local_llm_polished")
    }
    assert len(changes) == 4
    assert old == _legacy_config(), "input must not be mutated"


def test_is_idempotent():
    once, _ = mc.migrate(_legacy_config())

    twice, changes = mc.migrate(once)

    assert changes == []
    assert twice == once


def test_rollback_is_the_inverse_for_a_single_entry_config():
    old = _legacy_config()
    del old["plugins"]["entries"]["hermes-local-llm-polished/local-llm-polished"]

    restored, _ = mc.migrate(mc.migrate(old)[0], rollback=True)

    assert restored["stt"] == old["stt"]
    assert restored["plugins"] == old["plugins"]
    assert mc.migrate(restored, rollback=True)[1] == []


def test_auxiliary_task_route_moves_with_its_values():
    old = _legacy_config()
    route = {"provider": "custom", "base_url": "http://127.0.0.1:1/v1", "model": "m", "max_concurrency": 2}
    old["auxiliary"]["stt_polish"] = copy.deepcopy(route)

    new, _ = mc.migrate(old)

    assert new["auxiliary"] == {"vision": {"provider": "auto"}, "stt_enhance": route}


@pytest.mark.parametrize("stage", ["polish", "repair"])
def test_prototype_stage_name_becomes_post_processing(stage):
    old = _legacy_config()
    block = old["stt"]["local_llm_polished"]
    block[stage] = block.pop("post_processing")

    new, _ = mc.migrate(old)

    assert new["stt"]["stt_enhance"]["post_processing"] == POST_PROCESSING
    assert stage not in new["stt"]["stt_enhance"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["stt"].__setitem__("stt_enhance", {"backend": "faster_whisper"}),
        lambda c: c["auxiliary"].update(stt_polish={"model": "a"}, stt_enhance={"model": "b"}),
        lambda c: c["plugins"]["entries"]["hermes-local-llm-polished/local-llm-polished"].update(x=2),
        lambda c: c["stt"]["local_llm_polished"].__setitem__("polish", {"model": "x"}),
    ],
    ids=["provider-block", "aux-task", "plugin-entries", "two-stage-blocks"],
)
def test_refuses_conflicting_old_and_new_keys(mutate):
    config = _legacy_config()
    mutate(config)

    with pytest.raises(mc.MigrationConflict):
        mc.migrate(config)


def test_an_unrelated_provider_is_left_alone():
    config = {"stt": {"provider": "local", "local": {"model": "base"}}, "plugins": {"enabled": ["x"]}}

    assert mc.migrate(config) == (config, [])


def test_migrated_block_resolves_to_the_same_settings(config_mod):
    old = _legacy_config()
    new, _ = mc.migrate(old)
    renamed_old_stt = {**old["stt"], "stt_enhance": old["stt"]["local_llm_polished"]}

    assert config_mod.load_settings(new["stt"]) == config_mod.load_settings(renamed_old_stt)
    assert config_mod.load_settings(new["stt"]).post_processing == POST_PROCESSING
