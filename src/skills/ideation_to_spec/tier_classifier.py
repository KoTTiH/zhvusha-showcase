"""Deterministic tier classification for spec drafts.

The function is path-and-keyword based on purpose: an LLM may suggest a
draft tier in its YAML output, but the final tier is decided here so the
same evidence always produces the same tier across model versions,
prompts, and seeds.

The Tier 3 path list mirrors ``scripts/check_tier3_protection.sh`` and
:mod:`src.skills.spec_command.parser`. Any change must be reflected in
all three places.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.skills.base import CoreCapability

# Mirror of scripts/check_tier3_protection.sh + parser.py.
_TIER3_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "src/skills/base.py",
        "src/skills/__init__.py",
        "src/skills/registry.py",
        "src/personality/decision.py",
        ".importlinter",
        "AGENTS.md",
        "CLAUDE.md",
        "scripts/check_tier3_protection.sh",
    }
)
_TIER3_PREFIXES: tuple[str, ...] = ("src/safety/",)
_TIER3_SUFFIXES: tuple[str, ...] = ("/protocols.py",)

_PERSONALITY_ANCHOR_PATH = "src/skills/chat_response/prompts.py"
_PERSONALITY_ANCHOR_CONTRACT_MARKERS: tuple[str, ...] = (
    "personality_anchor",
    "identity contract",
    "identity anchor",
    "personality contract",
    "personality anchor",
    "shared identity",
    "shared personality",
    "общий anchor",
    "общем anchor",
    "общий prompt anchor",
    "общем prompt anchor",
    "контракт личности",
    "contract личности",
    "якорь личности",
    "инвариант личности",
    "ядро личности",
    "личность намертво",
    "всеобъемлющ",
)

# Existing capability modules (top-level paths under each become Tier 2).
# ``src/skills/`` is special-cased through ``_EXISTING_SKILL_DIRS``.
_TIER2_CAPABILITY_PREFIXES: tuple[str, ...] = (
    "src/memory/",
    "src/llm/",
    "src/personality/",
    "src/daemon/",
    "src/knowledge/",
    "src/bot/",
    "src/core/",
    "src/collectors/",
    "src/mcp_server/",
    "src/monitoring/",
)

# Existing skill subpackages — modification is Tier 2; new ones are Tier 1.
# Keep this synced with src/skills/__init__.py registrations.
_EXISTING_SKILL_DIRS: frozenset[str] = frozenset(
    {
        "channel_writer",
        "chat_response",
        "delegate",
        "kwork_monitor",
        "spec_command",
        "workspace_session",
    }
)

# Refactor keywords — only escalate if at least one path is not a fresh
# new-skill subdir.
_REFACTOR_KEYWORDS: tuple[str, ...] = (
    "изменить",
    "поправить",
    "переписать",
    "fix",
    "refactor",
    "rewrite",
    "modify",
)


def _is_tier3_path(path: str) -> bool:
    return (
        path in _TIER3_EXACT_PATHS
        or any(path.startswith(prefix) for prefix in _TIER3_PREFIXES)
        or any(path.endswith(suffix) for suffix in _TIER3_SUFFIXES)
    )


def _is_existing_capability_path(path: str) -> bool:
    """True if the path belongs to an already-existing capability module."""
    if path.startswith("src/skills/"):
        # src/skills/<name>/...
        parts = path.split("/", 3)
        return len(parts) >= 3 and parts[2] in _EXISTING_SKILL_DIRS
    return any(path.startswith(prefix) for prefix in _TIER2_CAPABILITY_PREFIXES)


def _is_personality_anchor_contract_change(
    whitelist_paths: list[str], *, goal: str, rationale: str
) -> bool:
    """True for shared Жвуша identity-contract edits in chat_response prompts.

    ``src/skills/chat_response/prompts.py`` still contains ordinary Tier 2
    implementation details, so the path alone is too broad. The protected case
    is changing ``PERSONALITY_ANCHOR`` or an all-mode identity/personality
    contract, which behaves like Personality core even though the file lives in
    a skill package.
    """
    if _PERSONALITY_ANCHOR_PATH not in whitelist_paths:
        return False

    evidence = f"{goal}\n{rationale}".casefold()
    return any(marker in evidence for marker in _PERSONALITY_ANCHOR_CONTRACT_MARKERS)


def _is_new_skill_path(path: str) -> bool:
    """True for ``src/skills/<new-name>/...`` (or non-skill paths, which we
    don't escalate on refactor keyword alone)."""
    if not path.startswith("src/skills/"):
        return True
    parts = path.split("/", 3)
    if len(parts) < 3:
        return True
    return parts[2] not in _EXISTING_SKILL_DIRS


def _has_refactor_keyword(goal: str) -> bool:
    lower = goal.lower()
    return any(kw in lower for kw in _REFACTOR_KEYWORDS)


def classify_spec_tier(
    *,
    whitelist_paths: list[str],
    goal: str,
    modifies_capabilities: Iterable[CoreCapability],
    rationale: str = "",
) -> int:
    """Return the tier (1, 2, or 3) for a draft spec.

    Rules in order:

    1. Any modifies_capabilities entry → Tier 3 (recursive
       self-improvement gate).
    2. Any path matches the Tier 3 list → Tier 3.
    3. Shared ``PERSONALITY_ANCHOR`` / identity-contract changes in
       ``chat_response`` → Tier 3.
    4. Any path inside an existing capability module → Tier 2.
    5. ``goal`` contains a refactor keyword AND at least one path is not a
       brand-new skill subdir → Tier 2.
    6. Otherwise → Tier 1.

    Empty ``whitelist_paths`` is accepted (returns Tier 1 by default) so
    the classifier composes cleanly with upstream draft-validation: the
    spec-level invariant that whitelist must be non-empty is enforced
    separately by :class:`SpecModel`.
    """
    if any(True for _ in modifies_capabilities):
        return 3

    for path in whitelist_paths:
        if _is_tier3_path(path):
            return 3

    if _is_personality_anchor_contract_change(
        whitelist_paths, goal=goal, rationale=rationale
    ):
        return 3

    for path in whitelist_paths:
        if _is_existing_capability_path(path):
            return 2

    if _has_refactor_keyword(goal):
        all_new = all(_is_new_skill_path(path) for path in whitelist_paths)
        if not all_new:
            return 2

    return 1
