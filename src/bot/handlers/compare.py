"""/compare command — admin-only A/B testing of two LLMs in chat.

Sends the user's prompt to two destinations in parallel and replies with
two messages, one per side, each labeled with the model that produced it.

The "main" side runs through ``LLMRouter.generate`` with the tier picked
by ``COMPARE_MAIN_TIER`` (``worker``/``analyst``/``strategist``); the
"shadow" side dispatches via ``LLMRouter.generate_oneoff`` to an explicit
``(COMPARE_PROVIDER, COMPARE_MODEL)`` pair from settings.

Both sides share:
  * the same personal-mode system prompt (IDENTITY_BLOCK +
    PERSONALITY_ANCHOR + load_personality + GROUNDING + PERSONAL_SYSTEM)
    — so the comparison is "Zhvusha on model A vs Zhvusha on model B",
    not raw LLMs;
  * the same user prompt that wraps recent conversation history in
    ``<CONVERSATION_HISTORY>`` so short reactive replies (a one-word
    follow-up to the previous turn) don't lose context.

Both calls go through the same usage tracker.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Literal

import structlog
from aiogram import Router
from aiogram.filters import Command

from src.bot.middleware.chat_logger import log_bot_response
from src.llm.protocols import LLMError, LLMRequest, LLMResponse
from src.skills.chat_response.context_loader import ContextLoader
from src.skills.chat_response.prompts import (
    ASSISTANT_SYSTEM,
    EXECUTION_PROTOCOL,
    GROUNDING_SECTION,
    IDENTITY_BLOCK,
    IDENTITY_RULES_NON_PERSONAL,
    PERSONAL_SYSTEM,
    PERSONALITY_ANCHOR,
    PUBLIC_CONTACT_SECTION,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.types import Message

    from src.core.config import Settings
    from src.llm.router import LLMRouter

logger = structlog.get_logger()

router = Router(name="compare")

_router: LLMRouter | None = None
_settings: Settings | None = None
_workspace_root: Path | None = None


def set_compare_deps(
    *, router: LLMRouter, settings: Settings, workspace_root: Path
) -> None:
    """Inject dependencies from main.py on startup."""
    global _router, _settings, _workspace_root
    _router = router
    _settings = settings
    _workspace_root = workspace_root


def _build_personal_system_prompt(settings: Settings, workspace_root: Path) -> str:
    """Personal-mode system prompt for /compare, mirroring chat_response's
    path for Nikita: identity (is_creator=true) + personality + grounding
    + PERSONAL_SYSTEM body. Used by /compare.
    """
    admin_id = settings.admin_user_id
    identity_block = IDENTITY_BLOCK.format(
        creator_user_id=admin_id,
        current_user_id=admin_id,
        is_creator="true",
    )
    loader = ContextLoader(workspace_root)
    personality_context = loader.load_personality(mode="personal") + GROUNDING_SECTION
    body = PERSONAL_SYSTEM.format(
        personality_context=personality_context,
        public_info=settings.public_info_about_nikita,
    )
    return identity_block + "\n" + PERSONALITY_ANCHOR + "\n" + body


# A stable, definitely-not-admin user id used to simulate a stranger when
# /compare_assistant is invoked. Picked far from realistic Telegram ids so
# people-context lookups don't accidentally land on a real profile.
_FAKE_NONADMIN_USER_ID = 999_000_001


def _build_assistant_system_prompt(settings: Settings, workspace_root: Path) -> str:
    """Assistant-mode system prompt for /compare_assistant: simulates how
    Zhvusha replies to a stranger. is_creator=false, identity rules for
    non-personal speakers prepended, ASSISTANT_SYSTEM body, public contact
    section appended when PUBLIC_CONTACT_NIKITA is set, EXECUTION_PROTOCOL
    block included (gates against tool-use bragging / promises).

    Personality is loaded with mode="assistant" so private files (core.md,
    reinforcements.md, etc.) are filtered out — same gates the real
    assistant path applies for non-owner traffic.
    """
    admin_id = settings.admin_user_id
    identity_block = IDENTITY_BLOCK.format(
        creator_user_id=admin_id,
        current_user_id=_FAKE_NONADMIN_USER_ID,
        is_creator="false",
    )
    loader = ContextLoader(workspace_root)
    personality_context = (
        loader.load_personality(mode="assistant")
        + GROUNDING_SECTION
        + EXECUTION_PROTOCOL
    )
    contact = getattr(settings, "public_contact_nikita", "") or ""
    public_contact_section = (
        PUBLIC_CONTACT_SECTION.format(public_contact=contact) if contact else ""
    )
    body = ASSISTANT_SYSTEM.format(
        personality_context=personality_context,
        public_info=settings.public_info_about_nikita,
        public_contact_section=public_contact_section,
    )
    return (
        identity_block
        + "\n"
        + IDENTITY_RULES_NON_PERSONAL
        + "\n"
        + PERSONALITY_ANCHOR
        + "\n"
        + body
    )


def _build_user_prompt(
    text: str, *, workspace_root: Path, chat_id: int | str | None
) -> str:
    """Wrap the current message in the same XML provenance envelope chat_response
    uses, with recent conversation history prepended when available. Without
    history a short reactive reply (a one-word follow-up to the prior turn)
    is unmoored — both worker and shadow lose the thread. Mirrors
    ``ChatResponseSkill._build_user_prompt`` for personal mode.
    """
    loader = ContextLoader(workspace_root)
    history = loader.load_recent_messages(
        chat_id=chat_id, mode="personal", exclude_text=text
    )
    parts: list[str] = []
    if history:
        parts.append(f"<CONVERSATION_HISTORY>\n{history}\n</CONVERSATION_HISTORY>")
    parts.append(f"<CURRENT_MESSAGE>\n{text}\n</CURRENT_MESSAGE>")
    return "\n\n".join(parts)


def _format_side(label: str, model: str, result: LLMResponse | Exception) -> str:
    """Render one side of the comparison: label + model + body or error."""
    head = f"<b>{label}</b> · <code>{model}</code>"
    if isinstance(result, Exception):
        return f"{head}\n\n⚠️ ошибка: {result}"
    return f"{head}\n\n{result.text}"


async def _run_compare(
    message: Message, mode: Literal["personal", "assistant"]
) -> None:
    """Shared dispatch for /compare (personal) and /compare_assistant.

    Difference is the system prompt: personal mode mirrors how Zhvusha
    talks to Nikita; assistant mode simulates a stranger talking to her
    (non-creator identity, gates active, ASSISTANT_SYSTEM template).
    """
    if _settings is None or _router is None or _workspace_root is None:
        await message.answer("Compare не инициализирован.")
        return

    if message.from_user is None or message.from_user.id != _settings.admin_user_id:
        await message.answer("Эта команда доступна только владельцу.")
        return

    if not _settings.compare_provider or not _settings.compare_model:
        await message.answer(
            "Compare выключен. Задай COMPARE_PROVIDER и COMPARE_MODEL в .env "
            "(см. .env.example)."
        )
        return

    cmd = "/compare_assistant" if mode == "assistant" else "/compare"
    text = (message.text or "").removeprefix(cmd).strip()
    if not text:
        await message.answer(
            f"Использование: {cmd} <твой запрос>\n"
            f"Сравню ответы {_settings.compare_main_tier}-модели и shadow-модели "
            f"(<code>{_settings.compare_provider}/{_settings.compare_model}</code>)."
            + (
                "\nРежим: assistant — Жвуша как для постороннего, не для тебя."
                if mode == "assistant"
                else ""
            )
        )
        return

    main_tier = _settings.compare_main_tier
    main_provider = getattr(_settings, f"{main_tier}_provider", "")
    main_model = getattr(_settings, f"{main_tier}_model", "")
    mode_tag = "[assistant] " if mode == "assistant" else ""
    main_label = f"🤖 {mode_tag}{main_tier} · {main_provider}"
    shadow_label = f"🔬 {mode_tag}shadow · {_settings.compare_provider}"
    shadow_model = _settings.compare_model

    if mode == "assistant":
        system_prompt = _build_assistant_system_prompt(_settings, _workspace_root)
        # Strangers don't get conversation history shipped to them; use a
        # single-turn user prompt to mirror the cold-start assistant path.
        user_prompt = f"<CURRENT_MESSAGE>\n{text}\n</CURRENT_MESSAGE>"
    else:
        system_prompt = _build_personal_system_prompt(_settings, _workspace_root)
        chat_id = message.chat.id if message.chat is not None else None
        user_prompt = _build_user_prompt(
            text, workspace_root=_workspace_root, chat_id=chat_id
        )

    started = time.monotonic()
    main_task = _router.generate(
        LLMRequest(
            prompt=user_prompt,
            system=system_prompt,
            tier=main_tier,
            caller=f"compare/main/{mode}",
        )
    )
    shadow_task = _router.generate_oneoff(
        provider=_settings.compare_provider,
        model=_settings.compare_model,
        prompt=user_prompt,
        system=system_prompt,
        caller=f"compare/shadow/{mode}",
    )

    main_result, shadow_result = await asyncio.gather(
        main_task, shadow_task, return_exceptions=True
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if isinstance(main_result, Exception) and not isinstance(main_result, LLMError):
        # Anything outside our LLMError envelope is a programming error,
        # not a provider failure — surface it to logs but still reply.
        logger.error(
            "compare_main_unexpected_error",
            exc_info=main_result,
        )

    main_text = _format_side(main_label, main_model, main_result)  # type: ignore[arg-type]
    shadow_text = _format_side(
        shadow_label,
        shadow_model,
        shadow_result,  # type: ignore[arg-type]
    )

    await message.answer(main_text, parse_mode="HTML")
    await message.answer(shadow_text, parse_mode="HTML")

    # Log both replies to chat_log so morning consolidation actually sees
    # the A/B exchanges. Without this, ChatLoggerMiddleware records only
    # the user side and the morning session reads the chat log as a stream of /compare
    # prompts with no answers — meaningless for diary/report.
    log_dir = _workspace_root / "logs"
    chat_id = message.chat.id if message.chat is not None else 0
    log_bot_response(log_dir=log_dir, text=main_text, chat_id=chat_id, mode=mode)
    log_bot_response(log_dir=log_dir, text=shadow_text, chat_id=chat_id, mode=mode)

    logger.info(
        "compare_done",
        mode=mode,
        prompt_len=len(text),
        elapsed_ms=elapsed_ms,
        main_ok=not isinstance(main_result, Exception),
        shadow_ok=not isinstance(shadow_result, Exception),
    )


@router.message(Command("compare"))
async def handle_compare(message: Message) -> None:
    await _run_compare(message, mode="personal")


@router.message(Command("compare_assistant"))
async def handle_compare_assistant(message: Message) -> None:
    await _run_compare(message, mode="assistant")
