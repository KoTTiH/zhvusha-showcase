"""Per-user chat-mode session state for ``chat_self_coding`` (Phase 40).

State is small (stage + active spec slug + last few messages) and lives
in Redis with a 24-hour TTL — long enough to span a normal session,
short enough that an abandoned mode doesn't leak into next week. The
value type is a frozen dataclass with explicit copy-on-write mutators
(``with_stage`` / ``with_active_spec`` / ``append_message``); the store
is the only thing that talks to Redis.

Design choices:

* ``is_open`` separates "the working room exists" from "chat-mode is
  currently intercepting normal messages". ``выход`` can close the room
  without deleting context; the next ``/код`` or ``/code`` reopens it.
* Recent messages are kept as an immutable tuple of at most
  ``MAX_RECENT_MESSAGES`` strings, used by the intent classifier as
  short-term context. Older entries roll off the front.
* Serialization is plain JSON — schema is small and stable, no need to
  pull in a heavier codec.
* Redis ``get`` is sometimes wired with ``decode_responses=False`` (the
  bot bootstraps it that way), so ``load`` decodes ``bytes`` if needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from time import time
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from src.skills.chat_self_coding.intent_classifier import Stage

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_RECENT_MESSAGES: int = 10
"""Cap on the rolling tail of messages kept for LLM context."""

DEFAULT_TTL_SECONDS: int = 86_400  # 24 hours
"""Redis TTL for a chat-mode session — covers normal continuity gaps."""

DRAFTING_STALE_SECONDS: float = 900.0
"""Max age for an in-flight plan draft before it is treated as abandoned."""


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------


class TaskPhase(StrEnum):
    """Durable high-level state machine for one /код task."""

    DISCUSSION = "discussion"
    SPEC = "spec"
    APPROVAL = "approval"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    COMMIT = "commit"
    REVIEW = "review"
    REPAIR = "repair"
    DONE = "done"


@dataclass(frozen=True)
class ChatSelfCodingState:
    """Immutable per-user chat-mode state."""

    user_id: int
    stage: Stage
    code_task_id: str = field(default_factory=lambda: f"code-task-{uuid4().hex}")
    task_phase: TaskPhase = TaskPhase.DISCUSSION
    active_spec_slug: str | None = None
    is_open: bool = True
    drafting_started_at_epoch: float | None = None
    recent_messages: tuple[str, ...] = field(default_factory=tuple)
    recovery_kind: str | None = None
    recovery_text: str | None = None
    recovery_error: str | None = None
    recovery_needs_user_decision: bool = True
    recovery_question: str | None = None
    active_goal: str | None = None
    compact_summary: str | None = None
    readonly_codex_session_id: str | None = None
    editor_codex_session_id: str | None = None
    failed_worktree_path: str | None = None
    failed_worktree_label: str | None = None
    failed_worktree_base_branch: str | None = None
    failed_worktree_base_sha: str | None = None

    def with_stage(self, new_stage: Stage) -> ChatSelfCodingState:
        if new_stage is Stage.DRAFTING:
            return replace(self, stage=new_stage)
        return replace(self, stage=new_stage, drafting_started_at_epoch=None)

    def with_drafting_started(self, started_at_epoch: float) -> ChatSelfCodingState:
        return replace(
            self,
            stage=Stage.DRAFTING,
            drafting_started_at_epoch=float(started_at_epoch),
        )

    def is_stale_drafting(
        self,
        now_epoch: float,
        *,
        stale_after_seconds: float = DRAFTING_STALE_SECONDS,
    ) -> bool:
        if self.stage is not Stage.DRAFTING:
            return False
        if self.drafting_started_at_epoch is None:
            return True
        return now_epoch - self.drafting_started_at_epoch >= stale_after_seconds

    def with_task_phase(self, new_phase: TaskPhase) -> ChatSelfCodingState:
        return replace(self, task_phase=new_phase)

    def with_active_spec(self, slug: str | None) -> ChatSelfCodingState:
        return replace(self, active_spec_slug=slug)

    def with_new_code_task(self) -> ChatSelfCodingState:
        return replace(
            self,
            code_task_id=f"code-task-{uuid4().hex}",
            task_phase=TaskPhase.DISCUSSION,
            readonly_codex_session_id=None,
            editor_codex_session_id=None,
            failed_worktree_path=None,
            failed_worktree_label=None,
            failed_worktree_base_branch=None,
            failed_worktree_base_sha=None,
            drafting_started_at_epoch=None,
        )

    def with_open(self, is_open: bool) -> ChatSelfCodingState:
        return replace(self, is_open=is_open)

    def with_recovery(
        self,
        *,
        kind: str,
        text: str,
        error: str,
        needs_user_decision: bool = True,
        question: str | None = None,
    ) -> ChatSelfCodingState:
        return replace(
            self,
            recovery_kind=kind,
            recovery_text=text,
            recovery_error=error,
            recovery_needs_user_decision=needs_user_decision,
            recovery_question=question,
        )

    def clear_recovery(self) -> ChatSelfCodingState:
        return replace(
            self,
            recovery_kind=None,
            recovery_text=None,
            recovery_error=None,
            recovery_needs_user_decision=True,
            recovery_question=None,
        )

    def with_active_goal(self, goal: str | None) -> ChatSelfCodingState:
        clean = goal.strip() if isinstance(goal, str) else ""
        return replace(self, active_goal=clean or None)

    def with_compact_summary(self, summary: str | None) -> ChatSelfCodingState:
        clean = summary.strip() if isinstance(summary, str) else ""
        return replace(self, compact_summary=clean or None)

    def with_readonly_codex_session(
        self, session_id: str | None
    ) -> ChatSelfCodingState:
        clean = session_id.strip() if isinstance(session_id, str) else ""
        return replace(self, readonly_codex_session_id=clean or None)

    def with_editor_resume(
        self,
        *,
        session_id: str | None,
        worktree_path: str | None,
        worktree_label: str | None,
        base_branch: str | None,
        base_sha: str | None,
    ) -> ChatSelfCodingState:
        return replace(
            self,
            editor_codex_session_id=_clean_optional(session_id),
            failed_worktree_path=_clean_optional(worktree_path),
            failed_worktree_label=_clean_optional(worktree_label),
            failed_worktree_base_branch=_clean_optional(base_branch),
            failed_worktree_base_sha=_clean_optional(base_sha),
        )

    def clear_editor_resume(self) -> ChatSelfCodingState:
        return replace(
            self,
            editor_codex_session_id=None,
            failed_worktree_path=None,
            failed_worktree_label=None,
            failed_worktree_base_branch=None,
            failed_worktree_base_sha=None,
        )

    def append_message(self, message: str) -> ChatSelfCodingState:
        new_tail = (*self.recent_messages, message)[-MAX_RECENT_MESSAGES:]
        return replace(self, recent_messages=new_tail)


# ---------------------------------------------------------------------------
# Store contract + Redis implementation
# ---------------------------------------------------------------------------


class StateStore(Protocol):
    """Async per-user state store for the chat-self-coding mode."""

    async def load(self, user_id: int) -> ChatSelfCodingState | None: ...
    async def save(self, state: ChatSelfCodingState) -> None: ...
    async def clear(self, user_id: int) -> None: ...


_KEY_PREFIX = "chat_self_coding:state:"


def _key_for(user_id: int) -> str:
    return f"{_KEY_PREFIX}{user_id}"


def _clean_optional(value: str | None) -> str | None:
    clean = value.strip() if isinstance(value, str) else ""
    return clean or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _serialize(state: ChatSelfCodingState) -> str:
    return json.dumps(
        {
            "user_id": state.user_id,
            "stage": state.stage.value,
            "code_task_id": state.code_task_id,
            "task_phase": state.task_phase.value,
            "active_spec_slug": state.active_spec_slug,
            "is_open": state.is_open,
            "drafting_started_at_epoch": state.drafting_started_at_epoch,
            "recent_messages": list(state.recent_messages),
            "recovery_kind": state.recovery_kind,
            "recovery_text": state.recovery_text,
            "recovery_error": state.recovery_error,
            "recovery_needs_user_decision": state.recovery_needs_user_decision,
            "recovery_question": state.recovery_question,
            "active_goal": state.active_goal,
            "compact_summary": state.compact_summary,
            "readonly_codex_session_id": state.readonly_codex_session_id,
            "editor_codex_session_id": state.editor_codex_session_id,
            "failed_worktree_path": state.failed_worktree_path,
            "failed_worktree_label": state.failed_worktree_label,
            "failed_worktree_base_branch": state.failed_worktree_base_branch,
            "failed_worktree_base_sha": state.failed_worktree_base_sha,
        }
    )


def _deserialize(raw: str) -> ChatSelfCodingState:
    data = json.loads(raw)
    return ChatSelfCodingState(
        user_id=int(data["user_id"]),
        stage=Stage(data["stage"]),
        code_task_id=str(data.get("code_task_id") or f"code-task-{uuid4().hex}"),
        task_phase=TaskPhase(data.get("task_phase", TaskPhase.DISCUSSION.value)),
        active_spec_slug=data.get("active_spec_slug"),
        is_open=bool(data.get("is_open", True)),
        drafting_started_at_epoch=_optional_float(
            data.get("drafting_started_at_epoch")
        ),
        recent_messages=tuple(data.get("recent_messages", [])),
        recovery_kind=data.get("recovery_kind"),
        recovery_text=data.get("recovery_text"),
        recovery_error=data.get("recovery_error"),
        recovery_needs_user_decision=bool(
            data.get("recovery_needs_user_decision", True)
        ),
        recovery_question=data.get("recovery_question"),
        active_goal=data.get("active_goal"),
        compact_summary=data.get("compact_summary"),
        readonly_codex_session_id=data.get("readonly_codex_session_id"),
        editor_codex_session_id=data.get("editor_codex_session_id"),
        failed_worktree_path=data.get("failed_worktree_path"),
        failed_worktree_label=data.get("failed_worktree_label"),
        failed_worktree_base_branch=data.get("failed_worktree_base_branch"),
        failed_worktree_base_sha=data.get("failed_worktree_base_sha"),
    )


class RedisStateStore:
    """Redis-backed ``StateStore`` with TTL.

    Accepts any object that implements the narrow ``get`` / ``set`` /
    ``delete`` subset of ``redis.asyncio.Redis`` we use. Keeps the
    dependency surface small enough that tests substitute an in-process
    fake without pulling in fakeredis.
    """

    def __init__(
        self,
        *,
        redis: Any,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        time_fn: Callable[[], float] = time,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._time = time_fn

    async def load(self, user_id: int) -> ChatSelfCodingState | None:
        raw = await self._redis.get(_key_for(user_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        state = _deserialize(raw)
        if state.is_stale_drafting(self._time()):
            await self.clear(user_id)
            return None
        return state

    async def save(self, state: ChatSelfCodingState) -> None:
        await self._redis.set(
            _key_for(state.user_id),
            _serialize(state),
            ex=self._ttl,
        )

    async def clear(self, user_id: int) -> None:
        await self._redis.delete(_key_for(user_id))
