"""Photo handler — download, describe via Gemini, respond as Zhvusha.

Policy: Gemini is only Zhvusha's eyes. The text LLM is her voice.
All user-facing text is generated with full personality context.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from aiogram import F, Router

from src.bot.middleware.chat_logger import log_bot_response, log_photo_message
from src.core.config import get_settings
from src.dialogue import DialogueStateUpdater, FileDialogueStateStore
from src.llm.protocols import LLMRequest, LLMVisionRequest
from src.llm.router import get_router
from src.memory import get_people_manager
from src.skills.chat_response.context_loader import ContextLoader
from src.skills.chat_response.prompts import GROUNDING_SECTION, PERSONALITY_ANCHOR
from src.skills.workspace_session.workspace import get_workspace_path

if TYPE_CHECKING:
    from pathlib import Path

    from aiogram.types import Message

    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()

router = Router(name="photo")

_ws_root: Path | None = None
_episodic: EpisodicMemory | None = None


def set_photo_deps(
    *,
    ws_root: Path,
    episodic: EpisodicMemory | None = None,
) -> None:
    """Inject dependencies from main.py on startup."""
    global _ws_root, _episodic
    _ws_root = ws_root
    _episodic = episodic


_PHOTO_SYSTEM = """\
{personality_anchor}

{personality_context}

{grounding}

{people_context}

## Контекст: фото
Собеседник прислал фото. Ниже описание того, что ты видишь (от твоих глаз).
Ответь как Жвуша — реагируй на фото, комментируй, задавай вопросы.
Пиши как в мессенджере — коротко, естественно.
"""


@router.message(F.photo)
async def handle_photo(
    message: Message,
    mode: str = "personal",
    album: list[Any] | None = None,
) -> None:
    """Handle incoming photos: vision model describes, text LLM responds."""
    settings = get_settings()
    ws_root = _ws_root or get_workspace_path(settings.workspace_path)
    messages: list[Message] = album or [message]

    user_id = message.from_user.id if message.from_user else 0
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    # 1. Download + save to workspace/media/
    media_dir = ws_root / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    image_bytes_list: list[bytes] = []
    saved_paths: list[str] = []

    for i, msg in enumerate(messages):
        if not msg.photo:
            continue
        photo = msg.photo[-1]  # largest size
        if msg.bot is None:
            continue
        file = await msg.bot.download(photo.file_id)
        if file is None:
            continue
        img_bytes: bytes = file.read()
        filename = f"{today}_{msg.message_id}_{i}.jpg"
        (media_dir / filename).write_bytes(img_bytes)
        image_bytes_list.append(img_bytes)
        saved_paths.append(f"media/{filename}")

    if not image_bytes_list:
        return

    # 2. Gemini describes (vision — only eyes)
    llm = get_router()
    caption = message.caption or ""
    vision_prompt = "Опиши что ты видишь на изображении(ях). Детально."
    if caption:
        vision_prompt = (
            f'Контекст от пользователя: "{caption}". '
            "Опиши что ты видишь на изображении(ях). Детально."
        )

    vision_response = await llm.describe_images(
        LLMVisionRequest(
            images=image_bytes_list,
            prompt=vision_prompt,
            caller="photo_vision",
        )
    )
    description = vision_response.text

    # 3. Build system prompt with personality context.
    # Use load_personality so non-owner gets public_core.md / public_identity.md
    # via the same substitution logic as chat_response.
    loader = ContextLoader(ws_root)
    personality_context = loader.load_personality(mode=mode)  # type: ignore[arg-type]

    people = get_people_manager()
    people_context = people.get_profile_for_context(user_id, mode)  # type: ignore[arg-type]

    system_prompt = _PHOTO_SYSTEM.format(
        personality_anchor=PERSONALITY_ANCHOR,
        personality_context=personality_context,
        grounding=GROUNDING_SECTION,
        people_context=people_context,
    )

    # 4. Text LLM generates response as Zhvusha (analyst — voice)
    user_prompt = f"Прислали {len(image_bytes_list)} фото.\n"
    if caption:
        user_prompt += f"Подпись: {caption}\n"
    user_prompt += f"Я вижу:\n{description}"

    llm_response = await llm.generate(
        LLMRequest(
            prompt=user_prompt,
            system=system_prompt,
            tier="analyst",
            caller="photo",
        )
    )
    response = llm_response.text

    # 5. Record episode
    if _episodic is not None:
        photo_desc = description[:300]
        episode_content = (
            f"{caption} [фото: {len(saved_paths)} изображений "
            f"— {', '.join(saved_paths)}]\n"
            f"Gemini описал: {photo_desc}"
        )
        await _episodic.record(
            content=episode_content,
            user_id=user_id,
            chat_type=mode,
            role="user",
            source="chat",
            metadata={"photo_paths": saved_paths},
        )

    # 6. Log to chat history
    chat_id = message.chat.id
    log_photo_message(
        log_dir=ws_root / "logs",
        user_id=user_id,
        username=(message.from_user.username or "") if message.from_user else "",
        caption=caption,
        chat_id=chat_id,
        mode=mode,
        photo_paths=saved_paths,
        photo_description=description,
        message_id=message.message_id,
        chat_type=str(getattr(message.chat, "type", "")),
        reply_to_message_id=(
            message.reply_to_message.message_id
            if message.reply_to_message is not None
            else None
        ),
    )
    log_bot_response(
        log_dir=ws_root / "logs",
        text=response,
        chat_id=chat_id,
        mode=mode,
    )
    DialogueStateUpdater(FileDialogueStateStore(ws_root)).record_observation(
        chat_id=chat_id,
        mode=mode,
        kind="photo_observation",
        summary=_photo_observation_summary(
            count=len(image_bytes_list),
            caption=caption,
        ),
        source="photo_vision",
    )

    # 7. Reply
    await message.answer(response)

    logger.info(
        "photo_response",
        mode=mode,
        user_id=user_id,
        num_photos=len(image_bytes_list),
        response_len=len(response),
    )


def _photo_observation_summary(*, count: int, caption: str) -> str:
    summary = f"Пользователь прислал фото: {count}."
    cleaned_caption = caption.strip()
    if cleaned_caption:
        summary += f" Подпись: {cleaned_caption[:240]}"
    return summary
