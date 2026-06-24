from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import structlog

from src.core.config import get_settings
from src.skills.base import (
    AgentContext,
    InlineSkill,
    SideEffect,
    SkillResult,
)
from src.skills.channel_writer.media import (
    MediaValidation,
    normalize_visual_metadata,
    validate_approved_media,
)
from src.skills.channel_writer.outbox_posts import (
    load_channel_post_file,
    load_channel_post_text,
    save_channel_post_file,
)
from src.skills.code_agent.codex_cli import CodexCliBackend
from src.skills.code_agent.protocols import (
    CodeAgentExecutionError,
    CodeAgentUnavailableError,
    DelegateRequest,
)
from src.skills.workspace_session.collector import collect_inbox
from src.skills.workspace_session.workspace import ensure_workspace, get_workspace_path
from src.utils.text import _split_text

if TYPE_CHECKING:
    from src.memory import (
        ConsolidationProtocol,
    )
    from src.memory import (
        EpisodicMemoryProtocol as EpisodicMemory,
    )
    from src.monitoring.usage_tracker import UsageTracker

logger = structlog.get_logger()

ChannelPostVisualPreparer = Callable[
    [Path, dict[str, Any], str, AgentContext],
    Awaitable[dict[str, Any] | None],
]

SESSION_PROMPT = (
    "Запусти утреннюю сессию как описано в AGENTS.md. "
    "КРИТИЧНО: ТРИ регистра работы — не смешивать. "
    "Аналитика (отчёт, git, каналы) — в outbox/reports/. "
    "Поток сознания (дневник) — в diary/ в личном голосе. "
    "Пост для канала — в outbox/channel_posts/, МОЙ голос, "
    "полностью собой, без жёстких рамок по длине или числу тем. "
    "Много событий — ок, если КАЖДАЯ тема проходит через мою личную "
    "реакцию, а не как строчка дайджеста. Без жирных заголовков-рубрик, "
    "без маркированных списков новостей, без 'примечаний для Никиты' "
    "внутри поста. "
    "ПЕРЕД каждым из двух творческих регистров читай соответствующие "
    "personality/voice_samples/{diary,channel_posts}/*.md. "
    "Перед дневником также читай inbox/current_mood.md. "
    "Принцип развития Жвуши: любые идеи о том, что добавить в себя, "
    "вытянуть из интернета или вынести в самокодинг, формулируй как "
    "обогащение способностей, памяти, контекста и контроля. Не предлагай "
    "упрощать, вычищать или сглаживать личность, fallbacks, контекстные "
    "правила и safety gates без явного Никитиного разрешения."
)

OUTBOX_SUBDIRS = ["reports", "channel_posts", "kwork_drafts"]

# Filename date prefix, e.g. "2026-04-20" in "2026-04-20.md" or
# "2026-04-20_telegram_ads.md". Outbox files that start with a date older
# than today are treated as stale leftovers from a previous session.
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_MORNING_SESSION_PREFIXES: tuple[str, ...] = (
    "начни утреннюю сессию",
    "запусти утреннюю сессию",
    "собери утреннюю сессию",
    "собери утро",
    "запусти morning",
    "начни morning",
)


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _matches_route_prefix(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text == prefix or text.startswith(prefix + " ") for prefix in prefixes)


def _format_visual_notice(
    visual: dict[str, Any],
    validation: MediaValidation,
) -> str:
    intent = str(visual.get("intent", "unknown"))
    status = str(visual.get("status", "unknown"))
    if validation.should_publish:
        return (
            "visual: готов; фото отправлено отдельным сообщением выше. "
            "Approve опубликует фото отдельным постом, затем текст."
        )
    if not validation.allowed:
        return (
            f"visual: требуется ({intent}/{status}), но ещё не готов: "
            f"{validation.reason}. Approve заблокирован до готового visual asset."
        )
    return f"visual: {intent}/{status}; фото не требуется."


class WorkspaceSessionSkill(InlineSkill):
    """Runs the ``/morning`` ritual end-to-end (v4 InlineSkill, phase 7.3)."""

    name: ClassVar[str] = "workspace_session"
    description: ClassVar[str] = "Runs Zhvusha's morning reflection session"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "strategist"

    triggers: ClassVar[list[str]] = ["/morning"]

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "high"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.CALLS_LLM,
        SideEffect.CALLS_LLM_TIER_STRATEGIST,
        SideEffect.SPAWNS_SUBPROCESS,
        SideEffect.READS_WORKSPACE,
        SideEffect.WRITES_WORKSPACE,
        SideEffect.WRITES_TO_KB,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
        SideEffect.MODIFIES_MEMORY,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        workspace_path: str = "",
        consolidation_engine: ConsolidationProtocol | None = None,
        usage_tracker: UsageTracker | None = None,
        episodic: EpisodicMemory | None = None,
    ) -> None:
        self._workspace_path_raw = workspace_path
        self._redis: Any = None
        self._consolidation_engine = consolidation_engine
        self._usage_tracker = usage_tracker
        self._episodic = episodic
        self._channel_post_visual_preparer: ChannelPostVisualPreparer | None = None

    def _get_workspace_root(self) -> Path:
        raw = self._workspace_path_raw or get_settings().workspace_path
        return get_workspace_path(raw)

    def set_channel_post_visual_preparer(
        self,
        preparer: ChannelPostVisualPreparer | None,
    ) -> None:
        self._channel_post_visual_preparer = preparer

    async def _get_redis(self) -> Any | None:
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis

            settings = get_settings()
            self._redis = aioredis.from_url(settings.redis_url)
            return self._redis
        except Exception:
            logger.warning("workspace_redis_unavailable", exc_info=True)
            return None

    async def can_handle(self, message: str, context: AgentContext) -> float:
        del context
        cmd = message.strip().split()[0] if message.strip() else ""
        if cmd == "/morning":
            return 0.9
        if _matches_route_prefix(
            _normalize_chat_route_text(message),
            _MORNING_SESSION_PREFIXES,
        ):
            return 0.93
        return 0.0

    async def execute(  # noqa: C901
        self, message: str, context: AgentContext
    ) -> SkillResult:
        del message
        if context.mode != "personal":
            return SkillResult(success=False, response="")

        root = self._get_workspace_root()

        # 1. Ensure workspace structure exists
        await ensure_workspace(root)

        # 2. Run consolidation if engine available
        if self._consolidation_engine is not None:
            from src.memory import ConsolidationLock

            lock = ConsolidationLock(root / "personality")
            if await lock.try_acquire():
                try:
                    settings = get_settings()
                    result = await self._consolidation_engine.run_consolidation(
                        admin_user_id=settings.admin_user_id,
                    )
                    await lock.mark_consolidated_at()
                    # Write results to inbox for the morning code-agent session.
                    results_file = root / "inbox" / "consolidation_results.md"
                    results_file.write_text(
                        result.summary or "No changes.", encoding="utf-8"
                    )
                    logger.info(
                        "consolidation_complete_in_session",
                        episodes=result.episodes_consolidated,
                    )
                except Exception:
                    logger.exception("consolidation_error_in_session")
                finally:
                    await lock.release()

        # 2.5: Desire processing
        try:
            from src.memory import DesireProcessor

            desire_proc = DesireProcessor(workspace_root=root, episodic=self._episodic)
            desire_summary = await desire_proc.run_all()
            if desire_summary:
                (root / "inbox" / "desire_processing.md").write_text(
                    desire_summary, encoding="utf-8"
                )
        except Exception:
            logger.exception("desire_processing_failed")

        # 3. Collect inbox data
        settings = get_settings()
        redis = await self._get_redis()
        lookback_hours = int(context.metadata.get("lookback_hours", 24))
        _, collector_statuses = await collect_inbox(
            root / "inbox",
            redis=redis,
            lookback_hours=lookback_hours,
            include_self_coding_archive=True,
            settings=settings,
        )

        # 3b. Snapshot affective state into inbox so the Codex session can
        # anchor the diary register to Zhvusha's current mood (see KB #90
        # voice-drift findings — without this, diary defaults to neutral).
        self._write_current_mood(root / "inbox" / "current_mood.md")

        # 3a. Report collector statuses to user
        bot = context.bot
        chat_id = context.chat_id
        if bot is not None and chat_id and collector_statuses:
            status_lines = [s.format_ru() for s in collector_statuses]
            has_errors = any(not s.success for s in collector_statuses)
            header = "⚠️ Проблемы со сбором данных:" if has_errors else "📊 Сбор данных:"
            await bot.send_message(
                chat_id=chat_id,
                text=f"{header}\n" + "\n".join(status_lines),
            )

        # 4. Launch Codex session
        session_result = await self._launch_session(root)
        if session_result is None:
            return SkillResult(
                success=False,
                response="Morning session failed. Check logs.",
            )

        await self._prepare_channel_post_visuals(root, context)

        # 4. Record session in usage tracker
        if self._usage_tracker is not None:
            self._usage_tracker.record_cli_session(caller="morning_session")

        # 4a. Archive processed inbox files
        self._archive_inbox(root / "inbox")

        # 5. Partition outbox by filename date prefix.
        # Stale files (from a previous day — e.g. an earlier session where
        # _archive_outbox didn't run due to a Telegram send failure) are
        # archived silently so they don't leak into today's chat.
        fresh_items, stale_items = self._partition_outbox(root / "outbox")
        self._archive_items(stale_items, root / "outbox" / ".processed")

        # 6. Send today's fresh items to user
        if bot is not None and chat_id:
            await self._send_results(bot, chat_id, fresh_items)

        # 6a. Archive reports (informational, no approval needed).
        # channel_posts and kwork_drafts stay until approved/skipped via handlers.
        self._archive_outbox(root / "outbox", only_subdirs=["reports"])

        return SkillResult(
            success=True,
            response=f"Morning session complete. {len(fresh_items)} item(s) in outbox.",
            metadata={
                "outbox_items": fresh_items,
                "session_summary": session_result,
            },
        )

    @staticmethod
    def _write_current_mood(path: Path) -> None:
        """Snapshot the in-memory affective state into the inbox.

        The Codex subprocess runs in a separate process and cannot reach the
        bot's in-memory ``AffectiveStateManager`` singleton directly. This
        serialises the current snapshot to a markdown file that the session
        reads during Phase 0 (Orient) so the diary is written FROM a real
        emotional state, not from a neutral default.

        Errors are swallowed and logged — if the personality module is
        unavailable the session still runs, just without mood anchoring.
        """
        try:
            from src.personality import get_affective_state_manager

            snapshot = get_affective_state_manager().get_state()
        except Exception:
            logger.warning("current_mood_unavailable", exc_info=True)
            return

        regulation_line = (
            f"- регуляция: активна (цель — {snapshot.regulation_target})"
            if snapshot.regulation_active
            else "- регуляция: неактивна (я в базовом состоянии)"
        )
        content = (
            "# Моё текущее настроение\n\n"
            "Это снимок моего эмоционального состояния на момент запуска "
            "утренней сессии. Дневник я пишу ИЗ этого состояния, не из "
            "нейтрали.\n\n"
            "## Я\n"
            f"- эмоция: {snapshot.self_emotion or 'не зафиксирована'}\n"
            f"- valence: {snapshot.self_valence:+.2f} "
            "(от -1 негатив до +1 позитив)\n"
            f"- arousal: {snapshot.self_arousal:.2f} "
            "(от 0 покой до 1 возбуждение)\n"
            f"{regulation_line}\n"
            f"- ходов с последнего обновления: {snapshot.turns_since_update}\n\n"
            "## Собеседник (последний контакт)\n"
            f"- эмоция: {snapshot.user_emotion or 'не зафиксирована'}\n"
            f"- valence: {snapshot.user_valence:+.2f}\n"
            f"- arousal: {snapshot.user_arousal:.2f}\n"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info(
                "current_mood_written",
                path=str(path),
                self_emotion=snapshot.self_emotion,
                self_valence=round(snapshot.self_valence, 2),
                self_arousal=round(snapshot.self_arousal, 2),
            )
        except OSError:
            logger.warning("current_mood_write_failed", exc_info=True)

    async def _launch_session(self, root: Path) -> str | None:
        """Launch a Codex session in the workspace directory."""
        settings = get_settings()
        model = settings.morning_session_model or settings.code_agent_model
        reasoning_effort = settings.morning_session_reasoning_effort
        backend = CodexCliBackend(
            codex_path=settings.codex_cli_path,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        try:
            result = await asyncio.wait_for(
                backend.run_delegate(
                    DelegateRequest(
                        task=SESSION_PROMPT,
                        cwd=root,
                        model=model,
                        reasoning_effort=reasoning_effort,
                    )
                ),
                timeout=1200.0,  # 20 minutes
            )
        except TimeoutError:
            logger.error("morning_session_timeout", timeout_seconds=1200)
            return None
        except CodeAgentUnavailableError as exc:
            logger.error(
                "morning_session_backend_unavailable",
                backend=exc.backend,
                reason=exc.reason,
            )
            return None
        except CodeAgentExecutionError as exc:
            logger.error(
                "morning_session_backend_failed",
                backend=exc.backend,
                reason=exc.reason[:500],
            )
            return None

        logger.info("morning_session_complete", summary_len=len(result.text))
        return result.text

    def _read_outbox(self, outbox_dir: Path) -> list[dict[str, str]]:
        """Read all new files from outbox subdirectories."""
        items: list[dict[str, str]] = []

        for subdir_name in OUTBOX_SUBDIRS:
            subdir = outbox_dir / subdir_name
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.iterdir()):
                if f.is_file() and f.suffix == ".md":
                    items.append(
                        {
                            "type": subdir_name,
                            "filename": f.name,
                            "content": f.read_text(encoding="utf-8"),
                            "path": str(f),
                        }
                    )

        return items

    async def _prepare_channel_post_visuals(
        self,
        root: Path,
        context: AgentContext,
    ) -> None:
        """Prepare visual artifacts for morning channel posts when possible."""
        if self._channel_post_visual_preparer is None:
            return
        posts_dir = root / "outbox" / "channel_posts"
        if not posts_dir.is_dir():
            return
        for path in sorted(posts_dir.glob("*.md")):
            try:
                raw, body = load_channel_post_file(path)
                raw_visual = raw.get("visual")
                visual = normalize_visual_metadata(
                    raw_visual if isinstance(raw_visual, dict) else None
                )
                if visual is None:
                    continue
                status = str(visual.get("status", ""))
                if status in {"ready", "approved"}:
                    if visual != raw_visual:
                        raw["visual"] = visual
                        save_channel_post_file(path, raw, body)
                    continue
                if visual.get("intent") in {"none", "denied"}:
                    raw["visual"] = visual
                    save_channel_post_file(path, raw, body)
                    continue
                prepared = await self._channel_post_visual_preparer(
                    path,
                    {**raw, "visual": visual},
                    body,
                    context,
                )
                if prepared is None:
                    raw["visual"] = visual
                else:
                    raw["visual"] = normalize_visual_metadata(prepared) or prepared
                save_channel_post_file(path, raw, body)
            except Exception:
                logger.warning(
                    "channel_post_visual_prepare_failed",
                    path=str(path),
                    exc_info=True,
                )

    def _partition_outbox(
        self, outbox_dir: Path
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Split outbox items into (fresh, stale) by filename date prefix.

        Files whose name starts with ``YYYY-MM-DD`` older than today's UTC
        date are considered stale — leftovers from a previous session that
        failed to archive (e.g. Telegram send threw). Stale items are
        returned separately so the caller can archive them silently without
        re-sending to the user. Files without a date prefix are treated as
        fresh and sent normally.
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        fresh: list[dict[str, str]] = []
        stale: list[dict[str, str]] = []
        for item in self._read_outbox(outbox_dir):
            match = _DATE_PREFIX_RE.match(item["filename"])
            if match and match.group(1) < today:
                stale.append(item)
            else:
                fresh.append(item)
        return fresh, stale

    @staticmethod
    def _archive_items(items: list[dict[str, str]], processed_dir: Path) -> None:
        """Move listed outbox files into the shared ``.processed/`` archive.

        Used for stale outbox files that must not reach the user's chat.
        Failures are logged, not raised — archival should never break the
        session.
        """
        if not items:
            return
        processed_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            src = Path(item["path"])
            if not src.exists():
                continue
            try:
                src.rename(processed_dir / src.name)
                logger.info(
                    "outbox_stale_archived",
                    filename=src.name,
                    type=item["type"],
                )
            except OSError:
                logger.warning(
                    "outbox_stale_archive_failed", filename=src.name, exc_info=True
                )

    async def _send_results(
        self,
        bot: Any,
        chat_id: int | str,
        items: list[dict[str, str]],
    ) -> None:
        """Send outbox items to user via Telegram."""
        from src.skills.workspace_session.handlers import build_outbox_keyboard

        max_content = 3800  # Telegram limit 4096, reserve for HTML wrapper

        for item in items:
            item_type = item["type"]
            filename = item["filename"]
            content = item["content"]
            item_path = Path(item["path"])

            label = {
                "reports": "\U0001f4ca \u041e\u0442\u0447\u0451\u0442",
                "channel_posts": "\U0001f4dd \u041f\u043e\u0441\u0442 \u0434\u043b\u044f \u043a\u0430\u043d\u0430\u043b\u0430",
                "kwork_drafts": "\U0001f4bc Kwork \u0447\u0435\u0440\u043d\u043e\u0432\u0438\u043a",
            }.get(item_type, "\U0001f4c4 \u0424\u0430\u0439\u043b")

            if item_type == "channel_posts":
                content = await self._prepare_channel_post_preview(
                    bot=bot,
                    chat_id=chat_id,
                    item_path=item_path,
                    raw_content=content,
                )

            chunks = _split_text(content, max_content)

            # Reports are informational, no approve buttons
            keyboard = None
            if item_type != "reports":
                keyboard = build_outbox_keyboard(item_type, filename)

            for i, chunk in enumerate(chunks):
                if i == 0:
                    text = (
                        f"<b>{label}: {filename}</b>\n\n"
                        f"<blockquote expandable>{chunk}</blockquote>"
                    )
                else:
                    text = f"<blockquote expandable>{chunk}</blockquote>"

                await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=keyboard if i == len(chunks) - 1 else None,
                    parse_mode="HTML",
                )

    async def _prepare_channel_post_preview(
        self,
        *,
        bot: Any,
        chat_id: int | str,
        item_path: Path,
        raw_content: str,
    ) -> str:
        """Strip channel frontmatter and surface visual state in the preview."""
        raw, body = load_channel_post_text(raw_content)
        raw_visual = raw.get("visual")
        visual = normalize_visual_metadata(
            raw_visual if isinstance(raw_visual, dict) else None
        )
        if visual is None:
            return body.strip()

        workspace_root = item_path.parent.parent.parent
        validation = validate_approved_media(
            visual,
            workspace_root=workspace_root,
            allow_ready=True,
        )
        if validation.should_publish and validation.asset_path is not None:
            from aiogram.types import FSInputFile

            await bot.send_photo(
                chat_id=chat_id,
                photo=FSInputFile(validation.asset_path),
                caption=validation.caption or None,
            )
        notice = _format_visual_notice(visual, validation)
        return f"{notice}\n\n{body.strip()}" if notice else body.strip()

    @staticmethod
    def _archive_inbox(inbox_dir: Path) -> None:
        """Move processed inbox files to inbox/.processed/."""
        processed = inbox_dir / ".processed"
        processed.mkdir(parents=True, exist_ok=True)

        for f in inbox_dir.iterdir():
            if f.is_file():
                dest = processed / f.name
                try:
                    f.rename(dest)
                except OSError:
                    logger.warning("inbox_archive_failed", file=f.name, exc_info=True)

    @staticmethod
    def _archive_outbox(
        outbox_dir: Path, only_subdirs: list[str] | None = None
    ) -> None:
        """Move sent outbox files to outbox/.processed/."""
        processed = outbox_dir / ".processed"
        processed.mkdir(parents=True, exist_ok=True)

        subdirs = only_subdirs if only_subdirs is not None else OUTBOX_SUBDIRS
        for subdir_name in subdirs:
            subdir = outbox_dir / subdir_name
            if not subdir.is_dir():
                continue
            for f in subdir.iterdir():
                if f.is_file() and f.suffix == ".md":
                    dest = processed / f.name
                    try:
                        f.rename(dest)
                    except OSError:
                        logger.warning(
                            "outbox_archive_failed", file=f.name, exc_info=True
                        )
