"""Load post-processing instructions from an installed Hermes skill.

``post_processing.skill`` names a skill (resolved in the active profile's
skill roots through Hermes' own discovery helpers) or points at an explicit
local ``SKILL.md``. Only the instruction body is read: frontmatter is dropped,
nothing is fetched, executed or registered, and supporting files are ignored.

Every failure raises :class:`SkillError` with a message meant for the user;
the caller keeps the raw transcript rather than falling back to other
instructions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SKILL_FILENAME = "SKILL.md"
MAX_SKILL_BYTES = 64 * 1024
MAX_REFERENCE_LENGTH = 256

# A name is one or more path segments (``my-skill`` or ``category/my-skill``).
# Each segment starts alphanumeric, so ``.``, ``..`` and hidden dirs never match.
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}(/[A-Za-z0-9][A-Za-z0-9._-]{0,99}){0,3}")
_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_PATH_PREFIXES = ("/", "~", "./", "../", "$")


class SkillError(Exception):
    """The configured skill cannot supply instructions."""


def is_path_reference(reference: str) -> bool:
    """Explicit path syntax; anything else is a skill name."""
    return reference.startswith(_PATH_PREFIXES) or reference.lower().endswith(".md")


def load_instructions(reference: object) -> str:
    """Return the instruction body of the referenced skill."""
    ref = str(reference if reference is not None else "").strip()
    if not ref:
        raise SkillError("post_processing.skill is empty")
    if len(ref) > MAX_REFERENCE_LENGTH:
        raise SkillError(f"post_processing.skill is longer than {MAX_REFERENCE_LENGTH} characters")
    if _URL_PATTERN.match(ref):
        raise SkillError(f"post_processing.skill {ref!r} is a URL; install the skill locally and reference it by name")
    # Resolution stats and walks user-chosen paths and skill roots. Python 3.11
    # raises EACCES/ENAMETOOLONG from Path.is_dir/is_file, and a bad path can
    # raise ValueError; both mean "this skill cannot be loaded".
    try:
        path = _resolve_path(ref) if is_path_reference(ref) else _resolve_name(ref)
        return _read_body(path, ref)
    except (OSError, ValueError) as exc:
        raise SkillError(f"cannot access skill {ref!r}: {exc}") from exc


def _resolve_path(ref: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(ref)))
    if not path.is_absolute():
        raise SkillError(f"skill path {ref!r} must be absolute or start with ~")
    if path.is_dir():
        path = path / SKILL_FILENAME
    if not path.is_file():
        raise SkillError(f"skill file not found: {path}")
    return path


def _resolve_name(name: str) -> Path:
    if not _NAME_PATTERN.fullmatch(name):
        raise SkillError(
            f"post_processing.skill {name!r} is not a valid skill name "
            "(letters, digits, '.', '_', '-', optionally 'category/name'); use ~/ or / for a file path"
        )
    try:
        from agent.skill_utils import get_all_skills_dirs, iter_skill_index_files, parse_frontmatter
    except ImportError as exc:
        raise SkillError(f"this Hermes build has no skill discovery helpers ({exc}); use a SKILL.md path") from exc

    # Roots come in Hermes precedence order (profile skills first); the first
    # root holding a match wins, and two different matches in one root are
    # refused rather than guessed.
    for root in get_all_skills_dirs():
        root = Path(root)
        if not root.is_dir():
            continue
        matches = [p for p in iter_skill_index_files(root, SKILL_FILENAME) if _matches(p, root, name, parse_frontmatter)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            listed = ", ".join(str(p.parent.relative_to(root)) for p in matches)
            raise SkillError(f"skill name {name!r} is ambiguous in {root} ({listed}); reference it as category/name")
    raise SkillError(f"skill {name!r} is not installed in this profile's skill directories")


def _matches(skill_md: Path, root: Path, name: str, parse_frontmatter) -> bool:
    skill_dir = skill_md.parent
    try:
        relative = skill_dir.relative_to(root)
    except ValueError:
        return False
    # A SKILL.md inside another skill's support files is not a skill.
    if any((root / Path(*relative.parts[:i]) / SKILL_FILENAME).is_file() for i in range(1, len(relative.parts))):
        return False
    if name in (relative.as_posix(), skill_dir.name):
        return True
    try:
        frontmatter, _ = parse_frontmatter(_read_text(skill_md))
    except SkillError:
        return False
    return str(frontmatter.get("name") or "").strip() == name


def _read_text(path: Path) -> str:
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_SKILL_BYTES + 1)
    except OSError as exc:
        raise SkillError(f"cannot read skill file {path}: {exc.strerror or exc}") from exc
    if len(data) > MAX_SKILL_BYTES:
        raise SkillError(f"skill file {path} is larger than {MAX_SKILL_BYTES // 1024} KiB")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SkillError(f"skill file {path} is not valid UTF-8") from exc


def _read_body(path: Path, ref: str) -> str:
    text = _read_text(path)
    body = _strip_frontmatter(text, path)
    if not body:
        raise SkillError(f"skill {ref!r} ({path}) has no instructions after its frontmatter")
    return body


def _strip_frontmatter(text: str, path: Path) -> str:
    if not text.startswith("---"):
        raise SkillError(f"skill file {path} has no YAML frontmatter; it is not a SKILL.md")
    closing = re.search(r"\n---[ \t\r]*(\n|$)", text[3:])
    if closing is None:
        raise SkillError(f"skill file {path} has an unterminated frontmatter block")
    return text[3 + closing.end():].strip()
