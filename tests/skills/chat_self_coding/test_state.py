"""Tests for ``chat_self_coding.state`` (Phase 40).

Per-user chat-mode state is held in Redis with a 24-hour TTL — long
enough to span a normal user session, short enough that abandoned modes
don't leak. The state is a small frozen dataclass: stage, active spec
slug, recent message tail (for LLM context). Mutators return new
instances; the store is the only thing that talks to Redis.

A tiny in-process ``FakeRedis`` covers the three ops we use (``get`` /
``set`` with ``ex`` TTL / ``delete``); see test_caps_enforcer.py for the
same idiom.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest


class FakeRedis:
    """Minimal in-memory stand-in for ``redis.asyncio.Redis`` k/v ops."""

    def __init__(self) -> None:
        self._kv: dict[str, str] = {}
        self._ttl: dict[str, int] = {}
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.calls.append(("set", (key, value, ex)))
        self._kv[key] = value
        if ex is not None:
            self._ttl[key] = ex

    async def get(self, key: str) -> str | None:
        self.calls.append(("get", (key,)))
        return self._kv.get(key)

    async def delete(self, key: str) -> int:
        self.calls.append(("delete", (key,)))
        if key in self._kv:
            del self._kv[key]
            self._ttl.pop(key, None)
            return 1
        return 0


class FakeRedisBytes(FakeRedis):
    """Variant returning ``bytes`` from get — matches default redis.asyncio."""

    async def get(self, key: str) -> bytes | None:  # type: ignore[override]
        self.calls.append(("get", (key,)))
        v = self._kv.get(key)
        return v.encode("utf-8") if v is not None else None


# ---------------------------------------------------------------------------
# State value type
# ---------------------------------------------------------------------------


class TestState:
    def test_state_is_frozen(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        with pytest.raises(FrozenInstanceError):
            state.user_id = 2  # type: ignore[misc]

    def test_recent_messages_default_is_empty_tuple(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        assert state.recent_messages == ()
        assert isinstance(state.recent_messages, tuple)

    def test_new_state_has_durable_code_task_id(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState, TaskPhase

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)

        assert state.code_task_id.startswith("code-task-")
        assert state.task_phase is TaskPhase.DISCUSSION

    def test_with_stage_returns_new_instance(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        moved = state.with_stage(Stage.RUNNING)
        assert moved is not state
        assert moved.stage == Stage.RUNNING
        assert state.stage == Stage.IDLE  # original unchanged

    def test_drafting_timestamp_marks_and_clears_inflight_plan(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)

        drafting = state.with_drafting_started(100.0)
        moved = drafting.with_stage(Stage.IDLE)

        assert drafting.stage is Stage.DRAFTING
        assert drafting.drafting_started_at_epoch == 100.0
        assert moved.drafting_started_at_epoch is None

    def test_stale_drafting_detects_missing_or_expired_timestamp(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            DRAFTING_STALE_SECONDS,
            ChatSelfCodingState,
        )

        legacy = ChatSelfCodingState(user_id=1, stage=Stage.DRAFTING)
        fresh = legacy.with_drafting_started(1_000.0)
        stale = legacy.with_drafting_started(1_000.0)

        assert legacy.is_stale_drafting(1_000.0)
        assert not fresh.is_stale_drafting(1_000.0 + DRAFTING_STALE_SECONDS - 1)
        assert stale.is_stale_drafting(1_000.0 + DRAFTING_STALE_SECONDS)

    def test_with_task_phase_returns_new_instance(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState, TaskPhase

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        moved = state.with_task_phase(TaskPhase.IMPLEMENTATION)

        assert moved is not state
        assert moved.task_phase is TaskPhase.IMPLEMENTATION
        assert state.task_phase is TaskPhase.DISCUSSION

    def test_with_active_spec_returns_new_instance(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        bound = state.with_active_spec("foo")
        assert bound is not state
        assert bound.active_spec_slug == "foo"
        assert state.active_spec_slug is None

    def test_new_code_task_returns_new_id(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(
            user_id=1,
            stage=Stage.IDLE,
            code_task_id="code-task-old",
        )

        next_state = state.with_new_code_task()

        assert next_state is not state
        assert next_state.code_task_id.startswith("code-task-")
        assert next_state.code_task_id != "code-task-old"
        assert state.code_task_id == "code-task-old"

    def test_clear_active_spec_returns_state_without_slug(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.DONE, active_spec_slug="x")
        cleared = state.with_active_spec(None)
        assert cleared.active_spec_slug is None

    def test_with_open_returns_new_instance(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        closed = state.with_open(False)
        assert closed is not state
        assert closed.is_open is False
        assert state.is_open is True

    def test_recovery_mutators_return_new_instances(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        blocked = state.with_recovery(
            kind="create_spec",
            text="исходный запрос",
            error="parse failed",
        )
        cleared = blocked.clear_recovery()

        assert blocked is not state
        assert blocked.recovery_kind == "create_spec"
        assert blocked.recovery_text == "исходный запрос"
        assert blocked.recovery_error == "parse failed"
        assert cleared is not blocked
        assert cleared.recovery_kind is None
        assert cleared.recovery_text is None
        assert cleared.recovery_error is None

    def test_goal_compact_and_codex_session_mutators_return_new_instances(
        self,
    ) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)

        updated = (
            state.with_active_goal("  довести /код до Codex-level  ")
            .with_compact_summary("  обсудили persistent session  ")
            .with_readonly_codex_session(" 019e1cf5-a63c-7ca1-a44e-44e555239799 ")
            .with_editor_resume(
                session_id=" codex-editor-thread-1 ",
                worktree_path=" /repo-worktrees/failed ",
                worktree_label=" isolated:spec:1:1 ",
                base_branch=" main ",
                base_sha=" abc123 ",
            )
        )

        assert updated is not state
        assert updated.active_goal == "довести /код до Codex-level"
        assert updated.compact_summary == "обсудили persistent session"
        assert (
            updated.readonly_codex_session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"
        )
        assert updated.editor_codex_session_id == "codex-editor-thread-1"
        assert updated.failed_worktree_path == "/repo-worktrees/failed"
        assert updated.failed_worktree_label == "isolated:spec:1:1"
        assert updated.failed_worktree_base_branch == "main"
        assert updated.failed_worktree_base_sha == "abc123"
        assert state.active_goal is None
        assert state.compact_summary is None
        assert state.readonly_codex_session_id is None
        assert state.editor_codex_session_id is None

    def test_append_message_appends(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import ChatSelfCodingState

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        s1 = state.append_message("first")
        s2 = s1.append_message("second")
        assert s2.recent_messages == ("first", "second")

    def test_append_message_truncates_to_max(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            MAX_RECENT_MESSAGES,
            ChatSelfCodingState,
        )

        state = ChatSelfCodingState(user_id=1, stage=Stage.IDLE)
        for i in range(MAX_RECENT_MESSAGES + 5):
            state = state.append_message(f"msg-{i}")
        assert len(state.recent_messages) == MAX_RECENT_MESSAGES
        # oldest dropped — last one preserved
        assert state.recent_messages[-1] == f"msg-{MAX_RECENT_MESSAGES + 4}"


# ---------------------------------------------------------------------------
# Redis store
# ---------------------------------------------------------------------------


class TestRedisStore:
    async def test_save_then_load_returns_equal_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        store = RedisStateStore(redis=redis)
        state = ChatSelfCodingState(
            user_id=42,
            stage=Stage.PENDING_APPROVAL,
            active_spec_slug="my-spec",
            recent_messages=("a", "b"),
            recovery_kind="create_spec",
            recovery_text="исходный запрос",
            recovery_error="parse failed",
        )
        await store.save(state)
        loaded = await store.load(42)
        assert loaded == state

    async def test_load_legacy_state_without_code_task_id_backfills_one(
        self,
    ) -> None:
        from src.skills.chat_self_coding.state import RedisStateStore, TaskPhase

        redis = FakeRedis()
        redis._kv["chat_self_coding:state:42"] = (
            '{"user_id": 42, "stage": "idle", "active_spec_slug": null, '
            '"is_open": true, "recent_messages": []}'
        )
        store = RedisStateStore(redis=redis)

        loaded = await store.load(42)

        assert loaded is not None
        assert loaded.code_task_id.startswith("code-task-")
        assert loaded.task_phase is TaskPhase.DISCUSSION

    async def test_load_returns_none_when_missing(self) -> None:
        from src.skills.chat_self_coding.state import RedisStateStore

        store = RedisStateStore(redis=FakeRedis())
        assert await store.load(999) is None

    async def test_load_clears_legacy_drafting_state_without_timestamp(self) -> None:
        from src.skills.chat_self_coding.state import RedisStateStore

        redis = FakeRedis()
        redis._kv["chat_self_coding:state:42"] = (
            '{"user_id": 42, "stage": "drafting", "active_spec_slug": null, '
            '"is_open": true, "recent_messages": []}'
        )
        store = RedisStateStore(redis=redis)

        assert await store.load(42) is None
        assert "chat_self_coding:state:42" not in redis._kv

    async def test_load_clears_expired_drafting_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            DRAFTING_STALE_SECONDS,
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        store = RedisStateStore(
            redis=redis,
            time_fn=lambda: 1_000.0 + DRAFTING_STALE_SECONDS,
        )
        await store.save(
            ChatSelfCodingState(user_id=42, stage=Stage.IDLE).with_drafting_started(
                1_000.0
            )
        )

        assert await store.load(42) is None
        assert "chat_self_coding:state:42" not in redis._kv

    async def test_load_keeps_fresh_drafting_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            DRAFTING_STALE_SECONDS,
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        state = ChatSelfCodingState(
            user_id=42,
            stage=Stage.IDLE,
        ).with_drafting_started(1_000.0)
        store = RedisStateStore(
            redis=redis,
            time_fn=lambda: 1_000.0 + DRAFTING_STALE_SECONDS - 1,
        )
        await store.save(state)

        assert await store.load(42) == state

    async def test_clear_removes_state(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        store = RedisStateStore(redis=redis)
        await store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        await store.clear(1)
        assert await store.load(1) is None

    async def test_save_applies_ttl(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            DEFAULT_TTL_SECONDS,
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        store = RedisStateStore(redis=redis)
        await store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        # The set call must include ex= matching default TTL.
        set_calls = [c for c in redis.calls if c[0] == "set"]
        assert len(set_calls) == 1
        _, args = set_calls[0]
        assert args[2] == DEFAULT_TTL_SECONDS

    async def test_custom_ttl_overrides_default(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        store = RedisStateStore(redis=redis, ttl_seconds=60)
        await store.save(ChatSelfCodingState(user_id=1, stage=Stage.IDLE))
        set_call = next(c for c in redis.calls if c[0] == "set")
        assert set_call[1][2] == 60

    async def test_load_handles_bytes_response(self) -> None:
        """redis.asyncio.Redis returns bytes by default — must decode."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedisBytes()
        store = RedisStateStore(redis=redis)
        state = ChatSelfCodingState(user_id=7, stage=Stage.RUNNING)
        await store.save(state)
        loaded = await store.load(7)
        assert loaded == state

    async def test_key_format_is_namespaced_per_user(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        store = RedisStateStore(redis=redis)
        await store.save(ChatSelfCodingState(user_id=42, stage=Stage.IDLE))
        set_call = next(c for c in redis.calls if c[0] == "set")
        key = set_call[1][0]
        assert "chat_self_coding" in key
        assert "42" in key

    async def test_two_users_dont_collide(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            ChatSelfCodingState,
            RedisStateStore,
        )

        redis = FakeRedis()
        store = RedisStateStore(redis=redis)
        await store.save(
            ChatSelfCodingState(user_id=1, stage=Stage.IDLE, active_spec_slug="alice")
        )
        await store.save(
            ChatSelfCodingState(user_id=2, stage=Stage.RUNNING, active_spec_slug="bob")
        )
        a = await store.load(1)
        b = await store.load(2)
        assert a is not None and a.active_spec_slug == "alice"
        assert b is not None and b.active_spec_slug == "bob"

    async def test_save_serializes_all_fields(self) -> None:
        """Round-trip preserves stage, slug, full recent_messages tuple."""
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import (
            ChatSelfCodingState,
            RedisStateStore,
            TaskPhase,
        )

        redis = FakeRedis()
        store = RedisStateStore(redis=redis)
        state = ChatSelfCodingState(
            user_id=99,
            stage=Stage.DONE,
            active_spec_slug="finished-spec",
            task_phase=TaskPhase.DONE,
            is_open=False,
            recent_messages=("hi", "yo", "ok"),
            active_goal="finish goal",
            compact_summary="compact context",
            readonly_codex_session_id="019e1cf5-a63c-7ca1-a44e-44e555239799",
        )
        await store.save(state)
        loaded = await store.load(99)
        assert loaded == state
        assert loaded is not None
        assert loaded.task_phase is TaskPhase.DONE
        assert loaded.is_open is False
        assert loaded.recent_messages == ("hi", "yo", "ok")
        assert loaded.active_goal == "finish goal"
        assert loaded.compact_summary == "compact context"
        assert (
            loaded.readonly_codex_session_id == "019e1cf5-a63c-7ca1-a44e-44e555239799"
        )

    async def test_load_legacy_state_defaults_to_open(self) -> None:
        from src.skills.chat_self_coding.intent_classifier import Stage
        from src.skills.chat_self_coding.state import RedisStateStore

        redis = FakeRedis()
        redis._kv["chat_self_coding:state:5"] = (
            '{"user_id": 5, "stage": "idle", "active_spec_slug": null, '
            '"recent_messages": []}'
        )
        store = RedisStateStore(redis=redis)

        loaded = await store.load(5)

        assert loaded is not None
        assert loaded.stage == Stage.IDLE
        assert loaded.is_open is True
