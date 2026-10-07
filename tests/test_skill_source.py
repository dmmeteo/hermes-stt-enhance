"""Skill reference resolution: names via Hermes skill roots, explicit paths, bounds."""

from __future__ import annotations

import os
import sys

import pytest

from conftest import write_skill


@pytest.fixture
def skill_source():
    return sys.modules["hermes_plugins.local_llm_polished.skill_source"]


def test_name_resolves_in_the_profile_root_without_frontmatter(skill_source, skill_roots):
    (profile, _external), _ = skill_roots
    write_skill(profile, "transcript-cleanup", body="Fix Kubernetes spelling.\n\nKeep slang.")

    assert skill_source.load_instructions("transcript-cleanup") == "Fix Kubernetes spelling.\n\nKeep slang."


def test_category_path_and_declared_name_both_resolve(skill_source, skill_roots):
    (profile, _external), _ = skill_roots
    write_skill(profile, "voice/stt-terms", body="Terms.", name="team-glossary")

    assert skill_source.load_instructions("voice/stt-terms") == "Terms."
    assert skill_source.load_instructions("stt-terms") == "Terms."
    assert skill_source.load_instructions("team-glossary") == "Terms."


def test_profile_root_wins_over_external_dirs(skill_source, skill_roots):
    (profile, external), _ = skill_roots
    write_skill(external, "cleanup", body="External.")
    write_skill(profile, "cleanup", body="Profile.")

    assert skill_source.load_instructions("cleanup") == "Profile."


def test_external_dir_is_used_when_the_profile_lacks_the_skill(skill_source, skill_roots):
    (_profile, external), _ = skill_roots
    write_skill(external, "cleanup", body="External.")

    assert skill_source.load_instructions("cleanup") == "External."


def test_two_different_skills_with_one_name_in_a_root_are_refused(skill_source, skill_roots):
    (profile, _external), _ = skill_roots
    write_skill(profile, "a/cleanup", body="One.")
    write_skill(profile, "b/cleanup", body="Two.")

    with pytest.raises(skill_source.SkillError, match="ambiguous"):
        skill_source.load_instructions("cleanup")
    assert skill_source.load_instructions("a/cleanup") == "One."


def test_a_skill_md_inside_another_skills_support_files_is_ignored(skill_source, skill_roots):
    (profile, _external), _ = skill_roots
    write_skill(profile, "outer", body="Outer.")
    write_skill(profile, "outer/references/inner", body="Nested.")

    with pytest.raises(skill_source.SkillError, match="not installed"):
        skill_source.load_instructions("inner")


def test_missing_skill_names_the_problem(skill_source, skill_roots):
    with pytest.raises(skill_source.SkillError, match="'nope' is not installed"):
        skill_source.load_instructions("nope")


@pytest.mark.parametrize(
    "name", ["a/../b", ".hidden", "a b", "café", "a//b", "a/b/c/d/e", "x" * 101, "skill:ns"]
)
def test_invalid_names_never_touch_skill_roots(skill_source, skill_roots, name):
    _, calls = skill_roots

    with pytest.raises(skill_source.SkillError, match="not a valid skill name"):
        skill_source.load_instructions(name)
    assert calls == []


@pytest.mark.parametrize("ref", ["https://example.com/SKILL.md", "file:///etc/passwd", "git+ssh://host/repo"])
def test_urls_are_refused_and_not_fetched(skill_source, skill_roots, ref):
    _, calls = skill_roots

    with pytest.raises(skill_source.SkillError, match="is a URL"):
        skill_source.load_instructions(ref)
    assert calls == []


@pytest.mark.parametrize("ref", ["", "   ", "x" * 300])
def test_empty_or_overlong_references_are_refused(skill_source, skill_roots, ref):
    with pytest.raises(skill_source.SkillError):
        skill_source.load_instructions(ref)


def test_names_need_hermes_discovery_helpers(skill_source, monkeypatch):
    monkeypatch.delitem(sys.modules, "agent.skill_utils", raising=False)

    with pytest.raises(skill_source.SkillError, match="no skill discovery helpers"):
        skill_source.load_instructions("cleanup")


def test_explicit_absolute_path_skips_name_resolution(skill_source, skill_roots, tmp_path):
    _, calls = skill_roots
    path = write_skill(tmp_path / "elsewhere", "mine", body="Mine.")

    assert skill_source.load_instructions(str(path)) == "Mine."
    assert skill_source.load_instructions(str(path.parent)) == "Mine."
    assert calls == []


def test_explicit_home_relative_path_is_expanded(skill_source, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_skill(tmp_path, "skills/mine", body="Mine.")

    assert skill_source.load_instructions("~/skills/mine/SKILL.md") == "Mine."


@pytest.mark.parametrize("ref", ["./SKILL.md", "../x/SKILL.md", "notes.md"])
def test_relative_paths_are_refused(skill_source, ref):
    with pytest.raises(skill_source.SkillError, match="must be absolute"):
        skill_source.load_instructions(ref)


def test_missing_explicit_path_is_an_error(skill_source, tmp_path):
    with pytest.raises(skill_source.SkillError, match="not found"):
        skill_source.load_instructions(str(tmp_path / "absent" / "SKILL.md"))


@pytest.mark.parametrize(
    "raw,match",
    [
        ("Just text, no frontmatter.", "no YAML frontmatter"),
        ("---\nname: x\nNo closing fence.", "unterminated"),
        ("---\nname: x\n---\n   \n", "no instructions"),
        (b"---\nname: x\n---\n\xff\xfe bad", "not valid UTF-8"),
    ],
)
def test_invalid_skill_files_are_errors(skill_source, tmp_path, raw, match):
    path = write_skill(tmp_path, "bad", raw=raw)

    with pytest.raises(skill_source.SkillError, match=match):
        skill_source.load_instructions(str(path))


def test_read_size_is_bounded(skill_source, tmp_path):
    path = write_skill(tmp_path, "big", body="x" * skill_source.MAX_SKILL_BYTES)

    with pytest.raises(skill_source.SkillError, match="larger than 64 KiB"):
        skill_source.load_instructions(str(path))


def test_bom_and_crlf_skill_files_are_read(skill_source, tmp_path):
    path = write_skill(tmp_path, "win", raw="\ufeff---\r\nname: win\r\n---\r\nFix names.\r\n")

    assert skill_source.load_instructions(str(path)) == "Fix names."


# ---------------------------------------------------------------------------
# Filesystem errors become SkillError (raw-transcript fallback), never escape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", ["", "/SKILL.md"])
def test_stat_permission_error_on_an_explicit_path_is_a_skill_error(skill_source, stat_denied, tmp_path, suffix):
    stat_denied(tmp_path / "denied")

    with pytest.raises(skill_source.SkillError, match="Permission denied"):
        skill_source.load_instructions(f"{tmp_path}/denied/x{suffix}")


@pytest.mark.parametrize("suffix", ["", "/SKILL.md", "/inside", "/inside/SKILL.md"])
def test_a_real_untraversable_path_is_a_skill_error(skill_source, untraversable_dir, suffix):
    with pytest.raises(skill_source.SkillError):
        skill_source.load_instructions(f"{untraversable_dir}{suffix}")


def test_an_inaccessible_skill_root_is_a_skill_error(skill_source, skill_roots, stat_denied):
    (profile, _external), _ = skill_roots
    stat_denied(profile, methods=("is_dir",))

    with pytest.raises(skill_source.SkillError, match="Permission denied"):
        skill_source.load_instructions("cleanup")


def test_a_failing_skill_index_walk_is_a_skill_error(skill_source, skill_roots, monkeypatch):
    def _broken_walk(root, filename):
        raise PermissionError(13, "Permission denied", str(root))

    monkeypatch.setattr(sys.modules["agent.skill_utils"], "iter_skill_index_files", _broken_walk)

    with pytest.raises(skill_source.SkillError, match="Permission denied"):
        skill_source.load_instructions("cleanup")


def test_a_stat_error_while_matching_candidates_is_a_skill_error(skill_source, skill_roots, stat_denied):
    (profile, _external), _ = skill_roots
    write_skill(profile, "voice/cleanup")
    stat_denied(profile / "voice", methods=("is_file",))

    with pytest.raises(skill_source.SkillError, match="Permission denied"):
        skill_source.load_instructions("cleanup")


def test_an_unreadable_unrelated_skill_does_not_block_frontmatter_matching(skill_source, skill_roots):
    (profile, _external), _ = skill_roots
    write_skill(profile, "glossary", body="Glossary.", name="team-terms")
    other = write_skill(profile, "other")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    other.chmod(0)
    try:
        assert skill_source.load_instructions("team-terms") == "Glossary."
    finally:
        other.chmod(0o600)
