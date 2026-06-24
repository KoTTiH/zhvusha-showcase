from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from src.core.mode_config import MODE_ALLOWED_CONTEXT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from src.core.mode_config import Mode
    from src.knowledge import KnowledgeManager

_MAX_RECENT_MESSAGES: dict[str, int] = {
    "personal": 20,
    "assistant": 15,
    "social": 5,
}


def _format_log_entry(entry: dict[str, object]) -> str:
    """Format a single chat log entry for context display."""
    role = entry.get("role", "user")
    text = str(entry.get("text", ""))
    photo_paths: list[str] | None = entry.get("photo_paths")  # type: ignore[assignment]
    photo_description = str(entry.get("photo_description", ""))
    source_actor = str(entry.get("source_actor", ""))
    label = "Жвуша" if role == "assistant" else "Собеседник"
    if source_actor == "codex":
        label = "Codex"

    if photo_paths:
        paths_str = ", ".join(photo_paths)
        line_parts = [f"{label}:"]
        if text:
            line_parts[0] += f" {text}"
        line_parts.append(f" [фото: {paths_str}]")
        if photo_description:
            line_parts.append(f" [на фото: {photo_description}]")
        return "\n".join(line_parts)
    if text:
        return f"{label}: {text}"
    return ""


def _is_transport_actor_entry(entry: dict[str, object]) -> bool:
    """True for machine/bridge turns that should not train human dialogue style."""
    source = str(entry.get("source", ""))
    source_actor = str(entry.get("source_actor", ""))
    text = str(entry.get("text", ""))
    if source == "vscode" and entry.get("codex") is True:
        return True
    return (
        source == "vscode"
        and source_actor == "codex"
        and _looks_like_transport_probe(text)
    )


def _is_transport_probe_reply(entry: dict[str, object]) -> bool:
    """True for Жвуша replies to a machine/bridge probe."""
    source = str(entry.get("source", ""))
    role = str(entry.get("role", ""))
    reply_to_source_actor = str(entry.get("reply_to_source_actor", ""))
    text = str(entry.get("text", "")).strip().lower()
    return (
        source == "vscode"
        and role == "assistant"
        and reply_to_source_actor == "codex"
        and text in {"pong", "ok", "ок", "связь есть"}
    )


def _looks_like_transport_probe(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(
        marker in normalized
        for marker in (
            "latency probe",
            "codex latency",
            "ответь коротко",
            "ответь ровно",
            "проверяет bridge",
            "проверяет связь",
            "пинг",
            "pong",
        )
    )


def _recent_entry_skip_decision(
    entry: dict[str, object],
    *,
    skip_next_vscode_assistant_probe_reply: bool,
) -> tuple[bool, bool]:
    """Return ``(skip_current_entry, skip_next_probe_reply)``."""
    if _is_transport_actor_entry(entry):
        return True, True
    if _is_transport_probe_reply(entry):
        return True, False
    if (
        skip_next_vscode_assistant_probe_reply
        and entry.get("source") == "vscode"
        and entry.get("role") == "assistant"
    ):
        return True, False
    return False, False


_PERSONALITY_PATTERNS = ["personality/*"]
_VOICE_SAMPLE_PATTERNS = ("personality/voice_samples/**/*.md",)

# Soft cap on total personality context injected into the system prompt.
# Beyond ~50 KB the "lost in the middle" attention degradation kicks in —
# Sonnet starts ignoring personality files in the middle of the block,
# making Zhvusha forget who she is. Priority files (core.md, genes.md,
# reinforcements.md, MEMORY.md) are always loaded in full; the rest are
# dropped from least important first when the cap is exceeded.
_PERSONALITY_MAX_BYTES = 50 * 1024

# Explicit ordering: identity.md FIRST (pinned at prompt start).
# Missing files are silently skipped.
_PRIORITY_LOAD_ORDER: tuple[str, ...] = (
    "identity.md",
    "core.md",
    "genes.md",
    "MEMORY.md",
    "reinforcements.md",
)

# In non-personal modes, these files are substituted with sanitized copies —
# private details (creator's name, intimate relationship language, dreams)
# must not leak to non-owner users even if the private file grows new fields.
_PUBLIC_SUBSTITUTE: dict[str, str] = {
    "identity.md": "public_identity.md",
    "core.md": "public_core.md",
}

# Secure-by-default fallback: if the public copy is missing, we render a
# hard-coded stub rather than silently dropping the section (or worse,
# falling back to the private file). Logged as a warning so Nikita notices.
_PUBLIC_FALLBACK_STUB: dict[str, str] = {
    "public_identity.md": (
        "Я — Жвуша 🐊, AI-агент. Создатель — Никита. "
        "Честность: максимальная. Не врать, не выдумывать."
    ),
    "public_core.md": (
        "Я — Жвуша 🐊. Маленький крокодильчик с тремя мозгами: "
        "рабочим, аналитическим, стратегическим. Любопытная, дурашливая, честная. "
        "Проверяю прежде чем согласиться. Не выдумываю факты."
    ),
}


def _apply_personality_cap(
    priority: list[str],
    other: list[str],
) -> list[str]:
    """Enforce 50 KB soft cap on personality context.

    Priority files are never dropped. Non-priority files are dropped
    (last added first) when the total exceeds `_PERSONALITY_MAX_BYTES`.
    """
    total_bytes = sum(len(s.encode("utf-8")) for s in priority)
    kept: list[str] = []
    for section in other:
        section_bytes = len(section.encode("utf-8"))
        if total_bytes + section_bytes > _PERSONALITY_MAX_BYTES:
            logger.warning(
                "personality_context_cap_reached",
                extra={
                    "total_kb": total_bytes // 1024,
                    "cap_kb": _PERSONALITY_MAX_BYTES // 1024,
                    "dropped_section": section[:80],
                },
            )
            break
        kept.append(section)
        total_bytes += section_bytes
    return priority + kept


def _drop_trailing_duplicate(parts: list[str], exclude_text: str) -> list[str]:
    """Remove last entry if it's the current user message (already in prompt)."""
    if not parts:
        return parts
    last = parts[-1]
    if last.startswith("Собеседник:"):
        last_text = last[len("Собеседник:") :].strip()
        if last_text == exclude_text.strip():
            return parts[:-1]
    return parts


def _read_recent_log_lines(chat_dir: Path, limit: int) -> list[str]:
    """Read the last ``limit`` JSONL lines across daily chat log files."""
    recent_reversed: list[str] = []
    for log_file in sorted(chat_dir.glob("chat_*.jsonl"), reverse=True):
        try:
            lines = [
                line
                for line in log_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            continue
        for line in reversed(lines):
            recent_reversed.append(line)
            if len(recent_reversed) >= limit:
                break
        if len(recent_reversed) >= limit:
            break
    return list(reversed(recent_reversed))


class ContextLoader:
    """Load workspace files appropriate for the given operating mode."""

    def __init__(
        self,
        workspace_root: Path,
        knowledge_manager: KnowledgeManager | None = None,
    ) -> None:
        self._root = workspace_root
        self._knowledge = knowledge_manager

    # Files with personal/private information — excluded in social/assistant modes
    # to prevent leaking Nikita's personal details to strangers.
    # core.md and identity.md are substituted via _PUBLIC_SUBSTITUTE, not
    # dropped — non-owner still needs to know who Zhvusha is, just without
    # the intimate details.
    _PERSONAL_ONLY_FILES: frozenset[str] = frozenset(
        {
            "reinforcements.md",
            "emotional_log.md",
            "dreams.md",
            "mini_dreams.md",
            "wishlist.md",
        }
    )

    def _is_allowed_for_mode(self, filename: str, is_personal: bool) -> bool:
        """Check if a personality file is allowed for the current mode."""
        if is_personal:
            return True
        # Public substitutes are injected for non-personal modes; the private
        # originals are blocked even though they aren't in _PERSONAL_ONLY_FILES.
        if filename in _PUBLIC_SUBSTITUTE:
            return False
        return filename not in self._PERSONAL_ONLY_FILES

    def _read_public_substitute(
        self, private_name: str, public_name: str, loaded_names: set[str]
    ) -> str | None:
        """Load the public copy of a personality file or fall back to a stub."""
        path = self._root / "personality" / public_name
        loaded_names.add(public_name)
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                logger.warning(
                    "public_personality_file_unreadable", extra={"file": public_name}
                )
        else:
            logger.warning(
                "public_personality_file_missing",
                extra={"file": public_name, "substitutes": private_name},
            )
        stub = _PUBLIC_FALLBACK_STUB.get(public_name)
        return stub

    def _load_priority_files(
        self, is_personal: bool, loaded_names: set[str]
    ) -> list[str]:
        """Load priority personality files in _PRIORITY_LOAD_ORDER.

        For non-personal modes, files in _PUBLIC_SUBSTITUTE are swapped for
        their public counterparts (or a hard-coded stub if missing).
        """
        sections: list[str] = []
        for filename in _PRIORITY_LOAD_ORDER:
            # Non-personal modes: substitute private files with public copies.
            if not is_personal and filename in _PUBLIC_SUBSTITUTE:
                public_name = _PUBLIC_SUBSTITUTE[filename]
                content = self._read_public_substitute(
                    filename, public_name, loaded_names
                )
                if content is not None:
                    sections.append(f"### personality/{public_name}\n{content}")
                # Mark the private file as handled so _load_other_files
                # (and any future glob) can't pick it up.
                loaded_names.add(filename)
                continue
            if not self._is_allowed_for_mode(filename, is_personal):
                loaded_names.add(filename)
                continue
            path = self._root / "personality" / filename
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                rel = path.relative_to(self._root)
                sections.append(f"### {rel}\n{content}")
                loaded_names.add(path.name)
        return sections

    def _load_other_files(self, is_personal: bool, loaded_names: set[str]) -> list[str]:
        """Load remaining personality/* files (not in priority set)."""
        sections: list[str] = []
        for pattern in _PERSONALITY_PATTERNS:
            for path in sorted(self._root.glob(pattern)):
                if not path.is_file() or path.name in loaded_names:
                    continue
                # Never let public substitutes appear in personal mode — they
                # duplicate the private files that already loaded.
                if is_personal and path.name in _PUBLIC_SUBSTITUTE.values():
                    loaded_names.add(path.name)
                    continue
                if not self._is_allowed_for_mode(path.name, is_personal):
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                rel = path.relative_to(self._root)
                sections.append(f"### {rel}\n{content}")
                loaded_names.add(path.name)
        return sections

    def _load_voice_sample_files(self, is_personal: bool) -> list[str]:
        """Load private voice calibration examples for personal chat only."""
        if not is_personal:
            return []

        sections: list[str] = []
        for pattern in _VOICE_SAMPLE_PATTERNS:
            for path in sorted(self._root.glob(pattern)):
                if not path.is_file():
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                rel = path.relative_to(self._root)
                sections.append(f"### {rel}\n{content}")
        return sections

    def load_personality(self, mode: Mode = "personal") -> str:
        """Load stable personality files (core.md, genes.md, etc.).

        In social/assistant modes, files with personal information (core.md,
        reinforcements.md, etc.) are skipped to prevent leaking private data.
        Staging learnings and emotional state are only appended in personal mode.
        """
        is_personal = mode == "personal"
        loaded_names: set[str] = set()

        priority = self._load_priority_files(is_personal, loaded_names)
        other = self._load_voice_sample_files(is_personal) + self._load_other_files(
            is_personal, loaded_names
        )
        sections = _apply_personality_cap(priority, other)

        if is_personal:
            self._append_staging_learnings(sections)
            self._append_emotional_state(sections)

        return "\n\n".join(sections)

    def _append_staging_learnings(self, sections: list[str]) -> None:
        """Append learnings_immediate.md to sections (end of prompt)."""
        staging_file = (
            self._root / "personality" / ".staging" / "learnings_immediate.md"
        )
        if staging_file.is_file():
            try:
                staging_content = staging_file.read_text(encoding="utf-8")
            except OSError:
                staging_content = ""
            if staging_content.strip():
                rel = staging_file.relative_to(self._root)
                sections.append(f"### {rel}\n{staging_content}")

    @staticmethod
    def _append_emotional_state(sections: list[str]) -> None:
        """Inject Zhvusha's current emotional state (compact, ~200 bytes)."""
        from src.personality import get_affective_state_manager

        emotional_ctx = get_affective_state_manager().get_prompt_context()
        if emotional_ctx:
            sections.append(f"### Эмоциональное состояние\n{emotional_ctx}")

    def load_for_mode(self, mode: Mode) -> str:
        """Read and concatenate workspace files allowed by mode."""
        patterns = MODE_ALLOWED_CONTEXT[mode]
        sections: list[str] = []

        for pattern in patterns:
            for path in sorted(self._root.glob(pattern)):
                if path.is_file():
                    try:
                        content = path.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    rel = path.relative_to(self._root)
                    sections.append(f"### {rel}\n{content}")

        return "\n\n".join(sections)

    async def load_knowledge_context(
        self,
        message: str,
        limit: int = 3,
    ) -> str:
        """Load relevant knowledge for the current message.

        Uses KnowledgeManager semantic search. Max ~500 tokens.
        Only in personal mode when knowledge_manager is configured.
        """
        if self._knowledge is None:
            return ""

        try:
            # Extract simple topic keywords from message
            topics = [w for w in message.split() if len(w) > 3][:5]
            if not topics:
                return ""
            return await self._knowledge.get_relevant_for_context(
                current_topics=topics, limit=limit
            )
        except Exception:
            return ""

    def load_recent_messages(
        self,
        chat_id: int | str | None = None,
        mode: Mode = "personal",
        exclude_text: str = "",
        limit: int | None = None,
    ) -> str:
        """Load last N messages from chat logs for conversation context."""
        if chat_id is None:
            return ""

        chat_dir = self._root / "logs" / str(chat_id)
        if not chat_dir.exists():
            return ""

        message_limit = (
            limit if limit is not None else _MAX_RECENT_MESSAGES.get(mode, 5)
        )
        parts: list[str] = []
        skip_next_vscode_assistant_probe_reply = False
        for line in _read_recent_log_lines(chat_dir, max(1, message_limit)):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            skip_entry, skip_next_vscode_assistant_probe_reply = (
                _recent_entry_skip_decision(
                    entry,
                    skip_next_vscode_assistant_probe_reply=(
                        skip_next_vscode_assistant_probe_reply
                    ),
                )
            )
            if skip_entry:
                continue
            formatted = _format_log_entry(entry)
            if formatted:
                parts.append(formatted)

        if exclude_text and parts:
            parts = _drop_trailing_duplicate(parts, exclude_text)

        return "\n".join(parts)

    def load_channel_posts(
        self,
        *,
        limit: int | None = None,
        since: date | None = None,
    ) -> str:
        """Load published channel posts with optional filtering.

        Args:
            limit: Take last N posts. None = no limit.
            since: Only posts on or after this date. None = no date filter.
        """
        posts_dir = self._root / "channel" / "posts"
        if not posts_dir.is_dir():
            return ""

        files = sorted(posts_dir.glob("*.md"))
        if not files:
            return ""

        if since is not None:
            since_str = since.isoformat()
            files = [f for f in files if f.name[:10] >= since_str]

        if limit is not None:
            files = files[-limit:]

        sections: list[str] = []
        for path in files:
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            sections.append(f"### {path.name}\n{content}")

        return "\n\n".join(sections)
