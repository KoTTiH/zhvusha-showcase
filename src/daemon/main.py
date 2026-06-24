"""Zhvusha Sense-Think-Act Daemon — main loop.

Continuously processes signals from Redis Streams,
makes decisions via LLM, and executes actions.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import structlog

from src.core.process_guard import FileProcessOwnershipGuard
from src.daemon.audit import AuditLog
from src.daemon.decision import DaemonDecisionEngine, DaemonDecisionType
from src.daemon.pending_action import ApprovalStore
from src.daemon.safety import SafetyGuard, SafetyGuardConfig
from src.daemon.sleep_agent import SleepTimeAgent
from src.daemon.stream import SignalStream
from src.daemon.ticker import AdaptiveTicker
from src.daemon.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from typing import Any

    from aiogram import Bot

    from src.agent_runtime.runtime import AgentRuntime
    from src.core.config import Settings
    from src.daemon.signals import Signal
    from src.life_runtime import LifeEvent, LifeTick
    from src.llm.router import LLMRouter

logger = structlog.get_logger()

_PROCESS_OWNERSHIP_HEARTBEAT_SECONDS = 30.0


class LifeRuntimeRunnerProtocol(Protocol):
    """Narrow hook used by the daemon without owning LifeRuntime internals."""

    def run_once(self, event: LifeEvent) -> LifeTick: ...


@dataclass
class _DaemonProcessOwnership:
    guard: FileProcessOwnershipGuard
    owner_id: str
    heartbeat_task: asyncio.Task[None] | None = None


def _daemon_process_owner_id() -> str:
    return f"daemon-process:{os.getpid()}:{int(time.time())}"


def _acquire_standalone_daemon_ownership(
    *,
    workspace_root: Path,
    owner_id: str | None = None,
    pid: int | None = None,
    guard: FileProcessOwnershipGuard | None = None,
) -> _DaemonProcessOwnership:
    from src.core.process_guard import render_process_ownership_report

    process_guard = guard or FileProcessOwnershipGuard(
        workspace_root / "runtime" / "process-owners.json"
    )
    resolved_owner = owner_id or _daemon_process_owner_id()
    status = process_guard.acquire(
        service="daemon",
        owner_id=resolved_owner,
        pid=pid,
    )
    if not status.acquired:
        raise RuntimeError(render_process_ownership_report((status,)))
    return _DaemonProcessOwnership(guard=process_guard, owner_id=resolved_owner)


def _start_standalone_daemon_ownership_heartbeat(
    ownership: _DaemonProcessOwnership,
    *,
    interval_seconds: float = _PROCESS_OWNERSHIP_HEARTBEAT_SECONDS,
) -> None:
    if ownership.heartbeat_task is not None and not ownership.heartbeat_task.done():
        return
    ownership.heartbeat_task = asyncio.create_task(
        _run_standalone_daemon_ownership_heartbeat(
            ownership,
            interval_seconds=interval_seconds,
        )
    )


async def _run_standalone_daemon_ownership_heartbeat(
    ownership: _DaemonProcessOwnership,
    *,
    interval_seconds: float,
) -> None:
    while True:
        await asyncio.sleep(max(interval_seconds, 0.1))
        status = ownership.guard.heartbeat("daemon", owner_id=ownership.owner_id)
        if not status.acquired:
            logger.warning(
                "daemon_process_ownership_heartbeat_failed",
                reason=status.reason,
            )


async def _release_standalone_daemon_ownership(
    ownership: _DaemonProcessOwnership,
) -> None:
    if ownership.heartbeat_task is not None:
        ownership.heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ownership.heartbeat_task
        ownership.heartbeat_task = None
    status = ownership.guard.release("daemon", owner_id=ownership.owner_id)
    if status.reason not in {"released", "not_owned"}:
        logger.warning(
            "daemon_process_ownership_release_failed",
            reason=status.reason,
        )


class ZhvushaDaemon:
    """Main daemon orchestrator."""

    def __init__(
        self,
        signal_stream: SignalStream,
        decision_engine: DaemonDecisionEngine,
        safety_guard: SafetyGuard,
        tool_registry: ToolRegistry,
        audit_log: AuditLog,
        approval_store: ApprovalStore,
        sleep_agent: SleepTimeAgent | None = None,
        life_runtime_runner: LifeRuntimeRunnerProtocol | None = None,
        life_runtime_enabled: bool = False,
        *,
        admin_chat_id: int = 0,
    ) -> None:
        self._stream = signal_stream
        self._decision = decision_engine
        self._safety = safety_guard
        self._tools = tool_registry
        self._audit = audit_log
        self._approval_store = approval_store
        self._sleep_agent = sleep_agent
        self._life_runtime_runner = life_runtime_runner
        self._life_runtime_enabled = life_runtime_enabled
        self._admin_chat_id = admin_chat_id
        self._ticker = AdaptiveTicker()
        self._consumer_name = platform.node() or "daemon-0"
        self._running = False
        self._wake_task: asyncio.Task[None] | None = None
        self._last_recover: float = 0.0
        self._RECOVER_INTERVAL: float = 300.0  # 5 minutes

    def tool_names(self) -> tuple[str, ...]:
        """Return daemon tools registered for this process."""
        return tuple(sorted(self._tools.list_tools()))

    async def start(self) -> None:
        """Initialize and enter the main loop."""
        await self._stream.ensure_groups()
        self._running = True
        self._wake_task = asyncio.create_task(
            self._stream.start_wake_listener(self._ticker.wake)
        )
        logger.info("daemon_started", consumer=self._consumer_name)

        try:
            await self._main_loop()
        finally:
            if self._wake_task is not None:
                self._wake_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._wake_task
            self._running = False
            logger.info("daemon_stopped")

    async def stop(self) -> None:
        """Signal the daemon to stop."""
        self._running = False
        self._ticker.wake()

    async def _main_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            woken = await self._ticker.wait_for_next_tick()

            signals = await self._stream.read_priority(self._consumer_name, count=10)

            for signal in signals:
                try:
                    await self._process_signal(signal)
                except Exception:
                    logger.exception(
                        "signal_processing_failed",
                        signal_id=signal.id,
                        source=signal.source,
                    )
                    try:
                        await self._stream.ack(signal)
                    except Exception:
                        logger.exception(
                            "ack_after_failure_failed",
                            signal_id=signal.id,
                        )

            # Execute any approved pending actions
            try:
                await self._execute_approved_actions()
            except Exception:
                logger.warning("execute_approved_failed", exc_info=True)

            await self._run_idle_maintenance(
                woken=woken,
                signal_count=len(signals),
            )

    async def _run_idle_maintenance(self, *, woken: bool, signal_count: int) -> None:
        """Run optional idle consumers without changing signal processing."""

        if signal_count > 0 or woken:
            return
        if self._life_runtime_enabled and self._life_runtime_runner is not None:
            try:
                from src.life_runtime import (
                    LifeEvent,
                    LifeEventKind,
                    LifeEventSource,
                    LifePriority,
                )

                now = datetime.now(UTC)
                tick = self._life_runtime_runner.run_once(
                    LifeEvent(
                        id=f"silence:{now.isoformat()}",
                        kind=LifeEventKind.SILENCE_TICK,
                        source=LifeEventSource.DAEMON,
                        priority=LifePriority.BACKGROUND,
                        payload={"reason": "daemon idle timeout"},
                        observed_at=now,
                    )
                )
                logger.info(
                    "life_runtime_tick",
                    tick_id=str(getattr(tick, "id", "")),
                )
            except Exception:
                logger.warning("life_runtime_tick_failed", exc_info=True)
        if self._sleep_agent is not None:
            try:
                count = await self._sleep_agent.run_maintenance_cycle()
                if count > 0:
                    logger.info("sleep_agent_proposals", count=count)
            except Exception:
                logger.warning("sleep_agent_failed", exc_info=True)

    async def _process_signal(self, signal: Signal) -> None:
        """Process a single signal through the decision pipeline."""
        logger.info(
            "processing_signal",
            signal_id=signal.id,
            source=signal.source,
            type=signal.signal_type,
        )

        # Get decision from LLM
        decision = await self._decision.decide(signal)

        # Safety check
        verdict = self._safety.check(decision)

        if verdict.blocked:
            await self._audit.record(
                signal_id=signal.id,
                decision=decision.decision.value,
                reasoning=decision.reasoning,
                tool_name=decision.action.tool if decision.action else None,
                result="blocked",
                result_details={"reason": verdict.reason},
            )
            logger.warning("signal_blocked", signal_id=signal.id, reason=verdict.reason)
            await self._stream.ack(signal)
            return

        if verdict.needs_approval:
            if decision.action is None:
                # No action to approve — nothing to persist or notify about
                await self._audit.record(
                    signal_id=signal.id,
                    decision=decision.decision.value,
                    reasoning=decision.reasoning,
                    result="skipped",
                    result_details={"reason": "needs_approval but no action"},
                )
                await self._stream.ack(signal)
                return

            tool_name = decision.action.tool
            tool_params = decision.action.params

            # 1. Persist the pending action
            action_id = await self._approval_store.create(
                signal_id=signal.id,
                tool_name=tool_name,
                tool_params=tool_params,
                decision_type=decision.decision.value,
                reasoning=decision.reasoning,
                safety_reason=verdict.reason,
            )

            # 2. Send notification (reply to this message = approve/reject)
            notification_sent = False
            if self._admin_chat_id:
                notify_tool = self._tools.get("send_telegram")
                if notify_tool is not None:
                    params_preview = str(tool_params)[:200]
                    result = await notify_tool.execute(
                        {
                            "text": (
                                f"🔐 Подтверждение #{action_id}:\n"
                                f"Действие: {tool_name}\n"
                                f"Параметры: {params_preview}\n"
                                f"Причина: {decision.reasoning}\n\n"
                                "Ответь свободно: разрешить, отменить, "
                                "отложить или что изменить."
                            ),
                            "chat_id": self._admin_chat_id,
                        }
                    )
                    if result.success and result.data and "message_id" in result.data:
                        await self._approval_store.set_telegram_message_id(
                            action_id, result.data["message_id"]
                        )
                        notification_sent = True

            if not notification_sent:
                logger.warning(
                    "approval_notification_not_sent",
                    action_id=action_id,
                    tool_name=tool_name,
                )

            await self._audit.record(
                signal_id=signal.id,
                decision=decision.decision.value,
                reasoning=decision.reasoning,
                tool_name=tool_name,
                result="pending_approval",
                result_details={
                    "reason": verdict.reason,
                    "action_id": action_id,
                },
            )
            await self._stream.ack(signal)
            return

        # Execute action
        if decision.decision == DaemonDecisionType.IGNORE:
            await self._audit.record(
                signal_id=signal.id,
                decision="ignore",
                reasoning=decision.reasoning,
                result="ignored",
            )
        elif decision.action is not None:
            tool_result = await self._tools.execute(
                decision.action.tool, decision.action.params
            )
            await self._audit.record(
                signal_id=signal.id,
                decision=decision.decision.value,
                reasoning=decision.reasoning,
                tool_name=decision.action.tool,
                tool_params=decision.action.params,
                result="success" if tool_result.success else "failure",
                result_details={"message": tool_result.message},
            )

        await self._stream.ack(signal)

    async def _execute_approved_actions(self) -> None:
        """Poll for approved pending actions and execute them."""
        now = time.monotonic()
        if now - self._last_recover >= self._RECOVER_INTERVAL:
            self._last_recover = now
            try:
                await self._approval_store.recover_stuck(timeout_minutes=10)
            except Exception:
                logger.warning("recover_stuck_failed", exc_info=True)

        try:
            approved = await self._approval_store.get_approved(limit=5)
        except Exception:
            logger.warning("approved_actions_fetch_failed", exc_info=True)
            return

        for action in approved:
            try:
                # Atomically claim (approved → executing) before running
                claimed = await self._approval_store.mark_executing(action.id)
                if not claimed:
                    continue

                tool_result = await self._tools.execute(
                    action.tool_name, action.tool_params or {}
                )
                await self._approval_store.mark_executed(
                    action.id, success=tool_result.success
                )
                await self._audit.record(
                    signal_id=action.signal_id,
                    decision=action.decision_type,
                    reasoning=f"Approved action #{action.id}",
                    tool_name=action.tool_name,
                    tool_params=action.tool_params,
                    result="success" if tool_result.success else "failure",
                    result_details={
                        "message": tool_result.message,
                        "action_id": action.id,
                    },
                )
                logger.info(
                    "approved_action_executed",
                    action_id=action.id,
                    tool=action.tool_name,
                    success=tool_result.success,
                )
            except Exception:
                logger.exception("approved_action_failed", action_id=action.id)
                try:
                    await self._approval_store.mark_executed(action.id, success=False)
                except Exception:
                    logger.exception("mark_executed_failed", action_id=action.id)


def build_daemon(
    *,
    redis: Any,
    llm_router: LLMRouter,
    bot: Bot,
    settings: Settings,
    session_maker: Any,
    usage_tracker: Any = None,
    agent_runtime: AgentRuntime | None = None,
) -> ZhvushaDaemon:
    """Build daemon from pre-initialized resources (for embedding in bot process)."""
    from src.daemon.tools.knowledge_store_tool import KnowledgeStoreTool
    from src.daemon.tools.send_telegram import SendTelegramTool
    from src.daemon.tools.workspace_read import (
        WorkspaceListTool,
        WorkspaceReadTool,
        WorkspaceSearchTool,
    )
    from src.knowledge import KnowledgeStore

    stream = SignalStream(redis)
    tool_registry = ToolRegistry()
    decision_engine = DaemonDecisionEngine(
        llm_router, tool_registry, tier=settings.daemon_decision_tier
    )

    safety_config = SafetyGuardConfig(
        max_llm_cost_per_day_usd=settings.daemon_max_llm_cost_per_day_usd,
        max_llm_calls_per_hour=settings.daemon_max_llm_calls_per_hour,
    )
    safety_guard = SafetyGuard(
        config=safety_config,
        usage_tracker=usage_tracker,
        tool_registry=tool_registry,
    )

    audit_log = AuditLog(session_maker)
    approval_store = ApprovalStore(session_maker)

    knowledge_store = KnowledgeStore(session_maker)
    sleep_agent = SleepTimeAgent(
        knowledge_store, llm_router, usage_tracker=usage_tracker
    )
    life_runtime_runner = None
    if settings.life_runtime_enabled:
        from src.life_runtime import FileLifeRuntimeStore, LifeTickRunner

        life_runtime_runner = LifeTickRunner(
            store=FileLifeRuntimeStore(
                Path(settings.life_runtime_state_path).expanduser()
            )
        )

    tool_registry.register(KnowledgeStoreTool(knowledge_store))
    tool_registry.register(SendTelegramTool(bot, settings.admin_user_id))
    # Workspace: read-only for daemon (write access only via MCP/chat)
    ws_root = Path(settings.workspace_path).expanduser()
    tool_registry.register(WorkspaceListTool(ws_root))
    tool_registry.register(WorkspaceReadTool(ws_root))
    tool_registry.register(WorkspaceSearchTool(ws_root))
    if settings.daemon_agent_runtime_enabled and agent_runtime is not None:
        from src.agent_runtime.profiles import BUILTIN_INVOCATION_PROFILES
        from src.daemon.agent_runtime_requester import (
            DaemonAgentRuntimeJobRequester,
        )
        from src.daemon.tools.agent_runtime import AgentRuntimeEnqueueTool

        tool_registry.register(
            AgentRuntimeEnqueueTool(
                requester=DaemonAgentRuntimeJobRequester.from_profiles(
                    BUILTIN_INVOCATION_PROFILES,
                    enabled=True,
                ),
                runtime=agent_runtime,
            )
        )
    elif settings.daemon_agent_runtime_enabled:
        logger.warning("daemon_agent_runtime_enabled_without_runtime")

    return ZhvushaDaemon(
        signal_stream=stream,
        decision_engine=decision_engine,
        safety_guard=safety_guard,
        tool_registry=tool_registry,
        audit_log=audit_log,
        approval_store=approval_store,
        sleep_agent=sleep_agent,
        life_runtime_runner=life_runtime_runner,
        life_runtime_enabled=settings.life_runtime_enabled,
        admin_chat_id=settings.admin_user_id,
    )


async def run_daemon() -> None:
    """Entry point: build components and start the daemon (standalone mode)."""
    import redis.asyncio as aioredis
    from aiogram import Bot

    from src.core.config import get_settings
    from src.llm.router import get_router
    from src.memory.database import get_engine, get_session_maker
    from src.monitoring.usage_tracker import UsageTracker as StandaloneTracker

    settings = get_settings()
    ws_root = Path(settings.workspace_path).expanduser()
    process_ownership = _acquire_standalone_daemon_ownership(workspace_root=ws_root)
    _start_standalone_daemon_ownership_heartbeat(process_ownership)
    logger.info(
        "daemon_process_ownership_acquired",
        owner_id=process_ownership.owner_id,
    )

    engine: Any | None = None
    redis: Any | None = None
    bot: Bot | None = None

    try:
        engine = get_engine(settings.database_url)
        session_maker = get_session_maker(engine)
        redis = aioredis.from_url(settings.redis_url)
        llm_router = get_router()
        bot = Bot(token=settings.bot_token)

        standalone_tracker = StandaloneTracker(ws_root / "monitoring")

        daemon = build_daemon(
            redis=redis,
            llm_router=llm_router,
            bot=bot,
            settings=settings,
            session_maker=session_maker,
            usage_tracker=standalone_tracker,
        )
        await daemon.start()
    finally:
        await _release_standalone_daemon_ownership(process_ownership)
        logger.info("daemon_process_ownership_released")
        if bot is not None:
            await bot.session.close()
        if redis is not None:
            await redis.close()
        if engine is not None:
            await engine.dispose()
