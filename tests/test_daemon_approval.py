"""Tests for daemon approval flow: PendingAction store, middleware, main loop."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.llm.protocols import LLMResponse, LLMUsage


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="haiku", usage=LLMUsage())


# ---------------------------------------------------------------------------
# ApprovalStore tests
# ---------------------------------------------------------------------------


class TestApprovalStoreCreate:
    async def test_creates_and_returns_id(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session

        # Simulate flush setting the id
        def _set_id(entry: Any) -> None:
            entry.id = 42

        session.add = MagicMock(side_effect=_set_id)

        store = ApprovalStore(mock_session_maker)
        action_id = await store.create(
            signal_id="sig-1",
            tool_name="send_telegram",
            tool_params={"text": "hello"},
            decision_type="act_notify",
            reasoning="test reason",
            safety_reason="needs approval",
        )
        assert action_id == 42
        session.commit.assert_awaited_once()


class TestApprovalStoreSetStatus:
    async def test_approve_pending_action(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ActionStatus, ApprovalStore

        session = mock_session_maker._mock_session

        # Simulate rowcount = 1 (found and updated)
        result_mock = MagicMock()
        result_mock.rowcount = 1
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        updated = await store.set_status(42, ActionStatus.APPROVED)
        assert updated is True
        session.commit.assert_awaited_once()

    async def test_set_status_not_found(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ActionStatus, ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 0
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        updated = await store.set_status(999, ActionStatus.APPROVED)
        assert updated is False

    async def test_set_status_idempotent(self, mock_session_maker: MagicMock) -> None:
        """Calling set_status on already-resolved action returns False."""
        from src.daemon.pending_action import ActionStatus, ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 0  # WHERE status='pending' won't match
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        updated = await store.set_status(42, ActionStatus.REJECTED)
        assert updated is False


class TestApprovalStoreGetApproved:
    async def test_returns_approved_actions(
        self, mock_session_maker: MagicMock
    ) -> None:
        from src.daemon.pending_action import ApprovalStore, PendingActionDTO

        session = mock_session_maker._mock_session
        fake_orm = MagicMock()
        fake_dto = PendingActionDTO(
            id=1,
            signal_id="s",
            tool_name="send_telegram",
            tool_params=None,
            decision_type="act",
            reasoning=None,
            safety_reason=None,
            status="approved",
            telegram_message_id=None,
        )
        fake_orm.to_dto.return_value = fake_dto

        scalars_mock = MagicMock()
        scalars_mock.all.return_value = [fake_orm]
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        actions = await store.get_approved(limit=5)
        assert len(actions) == 1
        assert actions[0].status == "approved"
        assert isinstance(actions[0], PendingActionDTO)


class TestApprovalStoreMarkExecuted:
    async def test_marks_as_executed(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 1
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        await store.mark_executed(42, success=True)
        session.commit.assert_awaited_once()

    async def test_marks_as_failed(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 1
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        await store.mark_executed(42, success=False)
        session.commit.assert_awaited_once()


class TestApprovalStoreMarkExecuting:
    async def test_claims_approved_action(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 1
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        claimed = await store.mark_executing(42)
        assert claimed is True
        session.commit.assert_awaited_once()

    async def test_returns_false_if_not_approved(
        self, mock_session_maker: MagicMock
    ) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 0
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        claimed = await store.mark_executing(42)
        assert claimed is False


class TestApprovalStoreRecoverStuck:
    async def test_recovers_stuck_actions(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 2
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        count = await store.recover_stuck(timeout_minutes=10)
        assert count == 2
        session.commit.assert_awaited_once()

    async def test_returns_zero_when_nothing_stuck(
        self, mock_session_maker: MagicMock
    ) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        result_mock = MagicMock()
        result_mock.rowcount = 0
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        count = await store.recover_stuck(timeout_minutes=5)
        assert count == 0


class TestApprovalStoreGetByTelegramMessageId:
    async def test_finds_action_by_message_id(
        self, mock_session_maker: MagicMock
    ) -> None:
        from src.daemon.pending_action import ApprovalStore, PendingActionDTO

        session = mock_session_maker._mock_session
        fake_orm = MagicMock()
        fake_dto = PendingActionDTO(
            id=7,
            signal_id="s",
            tool_name="t",
            tool_params=None,
            decision_type="act",
            reasoning=None,
            safety_reason=None,
            status="pending",
            telegram_message_id=12345,
        )
        fake_orm.to_dto.return_value = fake_dto

        scalars_mock = MagicMock()
        scalars_mock.first.return_value = fake_orm
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        action = await store.get_by_telegram_message_id(12345)
        assert action is not None
        assert action.id == 7
        assert isinstance(action, PendingActionDTO)

    async def test_returns_none_when_not_found(
        self, mock_session_maker: MagicMock
    ) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        scalars_mock = MagicMock()
        scalars_mock.first.return_value = None
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        action = await store.get_by_telegram_message_id(99999)
        assert action is None


# ---------------------------------------------------------------------------
# DaemonApprovalMiddleware tests
# ---------------------------------------------------------------------------


_ADMIN_ID = 12345


def _make_message(
    text: str = "да",
    reply_message_id: int | None = None,
    reply_text: str | None = None,
    chat_id: int = 100,
    from_user_id: int = _ADMIN_ID,
) -> MagicMock:
    """Create a mock aiogram Message with optional reply_to_message."""
    msg = AsyncMock()
    msg.text = text
    msg.chat = MagicMock()
    msg.chat.id = chat_id
    msg.answer = AsyncMock()
    msg.from_user = MagicMock()
    msg.from_user.id = from_user_id

    if reply_message_id is not None:
        reply_msg = MagicMock()
        reply_msg.message_id = reply_message_id
        reply_msg.text = reply_text
        msg.reply_to_message = reply_msg
    else:
        msg.reply_to_message = None

    return msg


def _mock_llm(intent: str = "approve") -> AsyncMock:
    """Create a mock LLMRouter that returns the given intent."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=_llm_resp(intent))
    return llm


class TestDaemonApprovalMiddleware:
    async def test_passes_through_non_reply(self) -> None:
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware

        store = AsyncMock()
        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm())
        handler = AsyncMock(return_value="handled")
        msg = _make_message(reply_message_id=None)

        result = await mw(handler, msg, {})
        assert result == "handled"
        handler.assert_awaited_once()

    async def test_passes_through_reply_to_non_approval(self) -> None:
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=None)
        store.get_by_id = AsyncMock(return_value=None)
        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm())
        handler = AsyncMock(return_value="handled")
        msg = _make_message(text="привет", reply_message_id=555)

        result = await mw(handler, msg, {})
        assert result == "handled"
        handler.assert_awaited_once()

    async def test_approves_when_llm_says_approve(self) -> None:
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(
            id=10, status=ActionStatus.PENDING, tool_name="send_telegram"
        )

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=True)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm("approve"))
        handler = AsyncMock()
        msg = _make_message(text="да конечно, давай", reply_message_id=777)

        await mw(handler, msg, {})

        store.set_status.assert_awaited_once_with(10, ActionStatus.APPROVED)
        msg.answer.assert_awaited_once()
        assert "одобрен" in msg.answer.call_args[0][0].lower()
        handler.assert_not_awaited()

    async def test_rejects_when_llm_says_reject(self) -> None:
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(
            id=11, status=ActionStatus.PENDING, tool_name="knowledge_store"
        )

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=True)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm("reject"))
        handler = AsyncMock()
        msg = _make_message(text="не надо, спасибо", reply_message_id=888)

        await mw(handler, msg, {})

        store.set_status.assert_awaited_once_with(11, ActionStatus.REJECTED)
        msg.answer.assert_awaited_once()
        assert "отклонен" in msg.answer.call_args[0][0].lower()
        handler.assert_not_awaited()

    async def test_unclear_response_asks_again(self) -> None:
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm("unclear"))
        handler = AsyncMock()
        msg = _make_message(text="хм не знаю", reply_message_id=1)

        await mw(handler, msg, {})

        store.set_status.assert_not_awaited()
        msg.answer.assert_awaited_once()
        handler.assert_not_awaited()

    async def test_empty_text_asks_again(self) -> None:
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)

        llm = _mock_llm()
        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, llm)
        msg = _make_message(text="", reply_message_id=1)
        await mw(AsyncMock(), msg, {})

        # LLM should NOT be called for empty text
        llm.generate.assert_not_awaited()
        msg.answer.assert_awaited_once()

    async def test_already_resolved_action(self) -> None:
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.APPROVED, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm())
        handler = AsyncMock()
        msg = _make_message(text="да", reply_message_id=1)

        await mw(handler, msg, {})

        store.set_status.assert_not_awaited()
        msg.answer.assert_awaited_once()
        assert "уже" in msg.answer.call_args[0][0].lower()
        handler.assert_not_awaited()

    async def test_race_approve_returns_already_handled(self) -> None:
        """If set_status returns False (race), user sees 'уже обработано'."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=False)  # lost race

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm("approve"))
        msg = _make_message(text="да", reply_message_id=1)
        await mw(AsyncMock(), msg, {})

        store.set_status.assert_awaited_once_with(1, ActionStatus.APPROVED)
        msg.answer.assert_awaited_once()
        assert "уже обработано" in msg.answer.call_args[0][0].lower()

    async def test_race_reject_returns_already_handled(self) -> None:
        """If set_status returns False on reject (race), user sees 'уже обработано'."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=False)  # lost race

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm("reject"))
        msg = _make_message(text="нет", reply_message_id=1)
        await mw(AsyncMock(), msg, {})

        store.set_status.assert_awaited_once_with(1, ActionStatus.REJECTED)
        msg.answer.assert_awaited_once()
        assert "уже обработано" in msg.answer.call_args[0][0].lower()

    async def test_non_admin_passes_through(self) -> None:
        """Non-admin reply to approval message is passed to handler."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm())
        handler = AsyncMock(return_value="handled")
        msg = _make_message(text="да", reply_message_id=1, from_user_id=99999)

        result = await mw(handler, msg, {})

        assert result == "handled"
        store.set_status.assert_not_awaited()
        handler.assert_awaited_once()


class TestClassifyIntent:
    """Unit tests for classify_intent function."""

    async def test_approve_response(self) -> None:
        from src.bot.middleware.daemon_approval import classify_intent

        llm = _mock_llm("approve")
        assert await classify_intent(llm, "да конечно") == "approve"

    async def test_approved_normalized(self) -> None:
        from src.bot.middleware.daemon_approval import classify_intent

        llm = _mock_llm("Approved")
        assert await classify_intent(llm, "ок") == "approve"

    async def test_reject_response(self) -> None:
        from src.bot.middleware.daemon_approval import classify_intent

        llm = _mock_llm("reject")
        assert await classify_intent(llm, "не надо, спасибо") == "reject"

    async def test_rejected_normalized(self) -> None:
        from src.bot.middleware.daemon_approval import classify_intent

        llm = _mock_llm("Rejected")
        assert await classify_intent(llm, "нет") == "reject"

    async def test_unclear_response(self) -> None:
        from src.bot.middleware.daemon_approval import classify_intent

        llm = _mock_llm("unclear")
        assert await classify_intent(llm, "может быть") == "unclear"

    async def test_unexpected_response_returns_unclear(self) -> None:
        from src.bot.middleware.daemon_approval import classify_intent

        llm = _mock_llm("something weird")
        assert await classify_intent(llm, "test") == "unclear"

    async def test_uses_worker_tier(self) -> None:
        from src.bot.middleware.daemon_approval import classify_intent

        llm = _mock_llm("approve")
        await classify_intent(llm, "да")

        request = llm.generate.call_args.args[0]
        assert request.tier == "worker"
        assert request.temperature == 0.0
        assert request.caller == "daemon_approval"


# ---------------------------------------------------------------------------
# Daemon main loop tests
# ---------------------------------------------------------------------------


def _make_signal(signal_id: str = "sig-1") -> MagicMock:
    """Create a mock Signal."""
    sig = MagicMock()
    sig.id = signal_id
    sig.source = "test"
    sig.signal_type = "test_signal"
    sig.priority = "normal"
    sig.payload = {}
    sig.requires_response = False
    sig.stream_entry_id = b"1-0"
    return sig


class TestDaemonMainLoopTryExcept:
    async def test_exception_doesnt_crash_loop(self) -> None:
        """If _process_signal raises, the loop continues and ACKs the signal."""
        from src.daemon.main import ZhvushaDaemon

        stream = AsyncMock()
        signal = _make_signal()
        # Return signal on first call, then always empty
        stream.read_priority = AsyncMock(side_effect=[[signal], []])
        stream.ack = AsyncMock()
        stream.ensure_groups = AsyncMock()

        approval_store = AsyncMock()
        approval_store.get_approved = AsyncMock(return_value=[])

        daemon = ZhvushaDaemon(
            signal_stream=stream,
            decision_engine=AsyncMock(),
            safety_guard=MagicMock(),
            tool_registry=AsyncMock(),
            audit_log=AsyncMock(),
            approval_store=approval_store,
        )

        # Make _process_signal raise
        daemon._process_signal = AsyncMock(side_effect=RuntimeError("boom"))

        # Stop after first tick completes (via side_effect on ticker)
        tick_count = 0

        async def _tick() -> bool:
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 2:
                daemon._running = False
            return False

        daemon._ticker.wait_for_next_tick = _tick
        daemon._running = True

        await daemon._main_loop()

        # Signal was ACK'd despite the error
        stream.ack.assert_awaited_once_with(signal)

    async def test_ack_failure_after_exception_doesnt_crash(self) -> None:
        """If ACK also fails after _process_signal error, loop still continues."""
        from src.daemon.main import ZhvushaDaemon

        stream = AsyncMock()
        signal = _make_signal()
        stream.read_priority = AsyncMock(side_effect=[[signal], []])
        stream.ack = AsyncMock(side_effect=RuntimeError("ack failed"))
        stream.ensure_groups = AsyncMock()

        ticker = AsyncMock()
        call_count = 0

        async def _fake_tick() -> bool:
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                daemon._running = False
            return False

        ticker.wait_for_next_tick = _fake_tick
        ticker.wake = MagicMock()

        approval_store = AsyncMock()
        approval_store.get_approved = AsyncMock(return_value=[])

        daemon = ZhvushaDaemon(
            signal_stream=stream,
            decision_engine=AsyncMock(),
            safety_guard=MagicMock(),
            tool_registry=AsyncMock(),
            audit_log=AsyncMock(),
            approval_store=approval_store,
        )
        daemon._ticker = ticker
        daemon._process_signal = AsyncMock(side_effect=RuntimeError("boom"))

        # Should not raise
        await daemon._main_loop()


class TestDaemonApprovalPersistence:
    async def test_approval_creates_pending_action(self) -> None:
        """When verdict.needs_approval, daemon stores pending action."""
        from src.daemon.decision import ActionSpec, DaemonDecision, DaemonDecisionType
        from src.daemon.main import ZhvushaDaemon
        from src.daemon.safety import SafetyVerdict
        from src.daemon.tools.base import ToolResult

        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_NOTIFY,
            reasoning="test",
            action=ActionSpec(tool="send_telegram", params={"text": "hi"}),
            requires_approval=True,
        )
        verdict = SafetyVerdict(
            needs_approval=True,
            reason="requires confirmation",
        )

        decision_engine = AsyncMock()
        decision_engine.decide = AsyncMock(return_value=decision)

        safety = MagicMock()
        safety.check = MagicMock(return_value=verdict)

        notify_tool = AsyncMock()
        notify_tool.execute = AsyncMock(
            return_value=ToolResult(
                success=True, message="sent", data={"message_id": 999}
            )
        )

        tool_registry = AsyncMock()
        tool_registry.get = MagicMock(return_value=notify_tool)

        approval_store = AsyncMock()
        approval_store.create = AsyncMock(return_value=42)

        audit = AsyncMock()
        stream = AsyncMock()

        daemon = ZhvushaDaemon(
            signal_stream=stream,
            decision_engine=decision_engine,
            safety_guard=safety,
            tool_registry=tool_registry,
            audit_log=audit,
            approval_store=approval_store,
            admin_chat_id=12345,
        )

        signal = _make_signal()
        await daemon._process_signal(signal)

        approval_store.create.assert_awaited_once()
        create_kwargs = approval_store.create.call_args.kwargs
        assert create_kwargs["tool_name"] == "send_telegram"
        assert create_kwargs["signal_id"] == "sig-1"

        # Telegram message_id stored
        approval_store.set_telegram_message_id.assert_awaited_once_with(42, 999)

        # Audit recorded
        audit.record.assert_awaited_once()
        audit_kwargs = audit.record.call_args.kwargs
        assert audit_kwargs["result"] == "pending_approval"

        # Signal ACK'd
        stream.ack.assert_awaited_once_with(signal)

    async def test_notification_failure_doesnt_crash(self) -> None:
        """If sending notification fails, action still stored in DB."""
        from src.daemon.decision import ActionSpec, DaemonDecision, DaemonDecisionType
        from src.daemon.main import ZhvushaDaemon
        from src.daemon.safety import SafetyVerdict
        from src.daemon.tools.base import ToolResult

        decision = DaemonDecision(
            decision=DaemonDecisionType.ACT_NOTIFY,
            reasoning="test",
            action=ActionSpec(tool="send_telegram", params={"text": "hi"}),
            requires_approval=True,
        )
        verdict = SafetyVerdict(needs_approval=True, reason="test")

        decision_engine = AsyncMock()
        decision_engine.decide = AsyncMock(return_value=decision)

        safety = MagicMock()
        safety.check = MagicMock(return_value=verdict)

        notify_tool = AsyncMock()
        notify_tool.execute = AsyncMock(
            return_value=ToolResult(success=False, message="Send failed")
        )

        tool_registry = AsyncMock()
        tool_registry.get = MagicMock(return_value=notify_tool)

        approval_store = AsyncMock()
        approval_store.create = AsyncMock(return_value=42)

        daemon = ZhvushaDaemon(
            signal_stream=AsyncMock(),
            decision_engine=decision_engine,
            safety_guard=safety,
            tool_registry=tool_registry,
            audit_log=AsyncMock(),
            approval_store=approval_store,
            admin_chat_id=12345,
        )

        await daemon._process_signal(_make_signal())

        # Action still created in DB
        approval_store.create.assert_awaited_once()
        # message_id NOT stored (send failed)
        approval_store.set_telegram_message_id.assert_not_awaited()


class TestDaemonExecuteApproved:
    async def test_executes_approved_actions(self) -> None:
        """Daemon picks up approved actions and executes them."""
        from src.daemon.main import ZhvushaDaemon
        from src.daemon.tools.base import ToolResult

        fake_action = SimpleNamespace(
            id=10,
            signal_id="sig-1",
            tool_name="knowledge_store",
            tool_params={"path": "test.txt", "content": "hi"},
            decision_type="act_silent",
        )

        approval_store = AsyncMock()
        approval_store.get_approved = AsyncMock(return_value=[fake_action])
        approval_store.mark_executing = AsyncMock(return_value=True)
        approval_store.mark_executed = AsyncMock()

        tool_registry = AsyncMock()
        tool_registry.execute = AsyncMock(
            return_value=ToolResult(success=True, message="ok")
        )

        audit = AsyncMock()

        daemon = ZhvushaDaemon(
            signal_stream=AsyncMock(),
            decision_engine=AsyncMock(),
            safety_guard=MagicMock(),
            tool_registry=tool_registry,
            audit_log=audit,
            approval_store=approval_store,
        )

        await daemon._execute_approved_actions()

        approval_store.mark_executing.assert_awaited_once_with(10)
        tool_registry.execute.assert_awaited_once_with(
            "knowledge_store", {"path": "test.txt", "content": "hi"}
        )
        approval_store.mark_executed.assert_awaited_once_with(10, success=True)
        audit.record.assert_awaited_once()

    async def test_skips_if_claim_fails(self) -> None:
        """If mark_executing returns False, action is skipped (already claimed)."""
        from src.daemon.main import ZhvushaDaemon

        fake_action = SimpleNamespace(
            id=10,
            signal_id="sig-1",
            tool_name="knowledge_store",
            tool_params={},
            decision_type="act_silent",
        )

        approval_store = AsyncMock()
        approval_store.get_approved = AsyncMock(return_value=[fake_action])
        approval_store.mark_executing = AsyncMock(return_value=False)

        tool_registry = AsyncMock()

        daemon = ZhvushaDaemon(
            signal_stream=AsyncMock(),
            decision_engine=AsyncMock(),
            safety_guard=MagicMock(),
            tool_registry=tool_registry,
            audit_log=AsyncMock(),
            approval_store=approval_store,
        )

        await daemon._execute_approved_actions()

        tool_registry.execute.assert_not_awaited()
        approval_store.mark_executed.assert_not_awaited()

    async def test_marks_failed_on_tool_error(self) -> None:
        """Failed tool execution marks action as failed."""
        from src.daemon.main import ZhvushaDaemon
        from src.daemon.tools.base import ToolResult

        fake_action = SimpleNamespace(
            id=5,
            signal_id="sig-2",
            tool_name="send_telegram",
            tool_params={},
            decision_type="act_notify",
        )

        approval_store = AsyncMock()
        approval_store.get_approved = AsyncMock(return_value=[fake_action])
        approval_store.mark_executing = AsyncMock(return_value=True)
        approval_store.mark_executed = AsyncMock()

        tool_registry = AsyncMock()
        tool_registry.execute = AsyncMock(
            return_value=ToolResult(success=False, message="error")
        )

        daemon = ZhvushaDaemon(
            signal_stream=AsyncMock(),
            decision_engine=AsyncMock(),
            safety_guard=MagicMock(),
            tool_registry=tool_registry,
            audit_log=AsyncMock(),
            approval_store=approval_store,
        )

        await daemon._execute_approved_actions()

        approval_store.mark_executed.assert_awaited_once_with(5, success=False)


# ---------------------------------------------------------------------------
# SendTelegramTool message_id return
# ---------------------------------------------------------------------------


class TestSendTelegramMessageId:
    async def test_returns_message_id(self) -> None:
        from src.daemon.tools.send_telegram import SendTelegramTool

        bot = AsyncMock()
        sent_msg = MagicMock()
        sent_msg.message_id = 42
        bot.send_message = AsyncMock(return_value=sent_msg)

        tool = SendTelegramTool(bot, admin_chat_id=100)
        result = await tool.execute({"text": "test"})

        assert result.success is True
        assert result.data is not None
        assert result.data["message_id"] == 42

    async def test_backward_compatible_without_reply_markup(self) -> None:
        from src.daemon.tools.send_telegram import SendTelegramTool

        bot = AsyncMock()
        sent_msg = MagicMock()
        sent_msg.message_id = 1
        bot.send_message = AsyncMock(return_value=sent_msg)

        tool = SendTelegramTool(bot, admin_chat_id=100)
        result = await tool.execute({"text": "hello"})

        assert result.success is True
        bot.send_message.assert_awaited_once_with(chat_id=100, text="hello")


# ---------------------------------------------------------------------------
# Fallback action_id lookup (race condition fix)
# ---------------------------------------------------------------------------


class TestFallbackActionIdLookup:
    async def test_fallback_finds_action_by_id_in_text(self) -> None:
        """When telegram_message_id not yet stored, parse action_id from text."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(
            id=42, status=ActionStatus.PENDING, tool_name="send_telegram"
        )

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=None)
        store.get_by_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=True)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm("approve"))
        handler = AsyncMock()
        msg = _make_message(
            text="да",
            reply_message_id=999,
            reply_text="🔐 Подтверждение #42:\nДействие: send_telegram",
        )

        await mw(handler, msg, {})

        store.get_by_id.assert_awaited_once_with(42)
        store.set_status.assert_awaited_once_with(42, ActionStatus.APPROVED)
        handler.assert_not_awaited()

    async def test_fallback_no_action_id_in_text_passes_through(self) -> None:
        """No #N in reply text → pass through to handler."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=None)
        store.get_by_id = AsyncMock(return_value=None)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm())
        handler = AsyncMock(return_value="handled")
        msg = _make_message(
            text="привет",
            reply_message_id=999,
            reply_text="Просто сообщение без номера",
        )

        result = await mw(handler, msg, {})
        assert result == "handled"
        store.get_by_id.assert_not_awaited()


# ---------------------------------------------------------------------------
# Wake daemon on approve (Redis Pub/Sub)
# ---------------------------------------------------------------------------


class TestWakeDaemonOnApprove:
    async def test_wake_published_on_approve(self) -> None:
        """After approve, middleware publishes wake signal via Redis."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus
        from src.daemon.stream import WAKE_CHANNEL

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=True)

        redis = AsyncMock()
        mw = DaemonApprovalMiddleware(
            store, _ADMIN_ID, _mock_llm("approve"), redis=redis
        )
        msg = _make_message(text="да", reply_message_id=1)
        await mw(AsyncMock(), msg, {})

        redis.publish.assert_awaited_once_with(WAKE_CHANNEL, "wake")

    async def test_no_wake_on_reject(self) -> None:
        """Reject does NOT wake the daemon."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=True)

        redis = AsyncMock()
        mw = DaemonApprovalMiddleware(
            store, _ADMIN_ID, _mock_llm("reject"), redis=redis
        )
        msg = _make_message(text="нет", reply_message_id=1)
        await mw(AsyncMock(), msg, {})

        redis.publish.assert_not_awaited()

    async def test_wake_failure_doesnt_crash(self) -> None:
        """If Redis publish fails, middleware still completes."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=True)

        redis = AsyncMock()
        redis.publish = AsyncMock(side_effect=ConnectionError("Redis down"))
        mw = DaemonApprovalMiddleware(
            store, _ADMIN_ID, _mock_llm("approve"), redis=redis
        )
        msg = _make_message(text="да", reply_message_id=1)

        # Should not raise
        await mw(AsyncMock(), msg, {})

        msg.answer.assert_awaited_once()
        assert "одобрен" in msg.answer.call_args[0][0].lower()

    async def test_no_redis_doesnt_crash(self) -> None:
        """Without redis parameter, wake is silently skipped."""
        from src.bot.middleware.daemon_approval import DaemonApprovalMiddleware
        from src.daemon.pending_action import ActionStatus

        fake_action = SimpleNamespace(id=1, status=ActionStatus.PENDING, tool_name="t")

        store = AsyncMock()
        store.get_by_telegram_message_id = AsyncMock(return_value=fake_action)
        store.set_status = AsyncMock(return_value=True)

        mw = DaemonApprovalMiddleware(store, _ADMIN_ID, _mock_llm("approve"))
        msg = _make_message(text="да", reply_message_id=1)

        # Should not raise (no redis passed)
        await mw(AsyncMock(), msg, {})
        msg.answer.assert_awaited_once()


# ---------------------------------------------------------------------------
# ApprovalStore.get_by_id
# ---------------------------------------------------------------------------


class TestApprovalStoreGetById:
    async def test_finds_action_by_id(self, mock_session_maker: MagicMock) -> None:
        from src.daemon.pending_action import ApprovalStore, PendingActionDTO

        session = mock_session_maker._mock_session
        fake_orm = MagicMock()
        fake_dto = PendingActionDTO(
            id=42,
            signal_id="s",
            tool_name="t",
            tool_params=None,
            decision_type="act",
            reasoning=None,
            safety_reason=None,
            status="pending",
            telegram_message_id=None,
        )
        fake_orm.to_dto.return_value = fake_dto

        scalars_mock = MagicMock()
        scalars_mock.first.return_value = fake_orm
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        action = await store.get_by_id(42)
        assert action is not None
        assert action.id == 42

    async def test_returns_none_for_missing_id(
        self, mock_session_maker: MagicMock
    ) -> None:
        from src.daemon.pending_action import ApprovalStore

        session = mock_session_maker._mock_session
        scalars_mock = MagicMock()
        scalars_mock.first.return_value = None
        result_mock = MagicMock()
        result_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=result_mock)

        store = ApprovalStore(mock_session_maker)
        action = await store.get_by_id(999)
        assert action is None


# ---------------------------------------------------------------------------
# recover_stuck throttle
# ---------------------------------------------------------------------------


class TestRecoverStuckThrottle:
    async def test_recover_stuck_throttled(self) -> None:
        """recover_stuck only called once per RECOVER_INTERVAL."""
        import time

        from src.daemon.main import ZhvushaDaemon

        approval_store = AsyncMock()
        approval_store.get_approved = AsyncMock(return_value=[])
        approval_store.recover_stuck = AsyncMock(return_value=0)

        daemon = ZhvushaDaemon(
            signal_stream=AsyncMock(),
            decision_engine=AsyncMock(),
            safety_guard=MagicMock(),
            tool_registry=AsyncMock(),
            audit_log=AsyncMock(),
            approval_store=approval_store,
        )

        # First call should trigger recover_stuck
        await daemon._execute_approved_actions()
        assert approval_store.recover_stuck.await_count == 1

        # Second call immediately after should NOT trigger
        await daemon._execute_approved_actions()
        assert approval_store.recover_stuck.await_count == 1

        # Simulate time passage past the interval
        daemon._last_recover = time.monotonic() - daemon._RECOVER_INTERVAL - 1
        await daemon._execute_approved_actions()
        assert approval_store.recover_stuck.await_count == 2
