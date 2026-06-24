import re
from html import escape

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from src.skills.kwork_monitor.models import ProjectCard

MAX_DESCRIPTION_LENGTH = 500


def _clean_html(text: str) -> str:
    """Strip HTML tags from Kwork content and escape special characters."""
    clean = re.sub(r"<br\s*/?>", "\n", text)
    clean = re.sub(r"<[^>]+>", "", clean)
    return escape(clean)


def format_project_card(card: ProjectCard) -> str:
    """Format a ProjectCard as Telegram HTML message."""
    title = _clean_html(card.title)
    description = _clean_html(card.description)
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[:MAX_DESCRIPTION_LENGTH] + "..."

    price_str = f"{card.price} руб." if card.price is not None else "не указан"
    offers_str = str(card.offers) if card.offers is not None else "?"
    keywords_str = ", ".join(card.matched_keywords)

    return (
        f"<b>{title}</b>\n\n"
        f"<blockquote expandable>{description}</blockquote>\n"
        f"💰 Бюджет: {price_str}\n"
        f"👥 Откликов: {offers_str}\n"
        f"🔑 Совпадения: {keywords_str}\n"
        f'🔗 <a href="{card.url}">Открыть на Kwork</a>'
    )


def build_project_keyboard(project_id: int) -> InlineKeyboardMarkup:
    """Build inline keyboard for a new project notification."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔍 Проверить",
                    callback_data=f"kwork:check:{project_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Пропустить",
                    callback_data=f"kwork:skip:{project_id}",
                ),
            ]
        ]
    )


def build_evaluate_keyboard(project_id: int) -> InlineKeyboardMarkup:
    """Build inline keyboard after evaluation (verdict shown)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✍️ Написать отклик",
                    callback_data=f"kwork:draft:{project_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Пропустить",
                    callback_data=f"kwork:skip:{project_id}",
                ),
            ]
        ]
    )


def build_draft_keyboard(project_id: int) -> InlineKeyboardMarkup:
    """Build inline keyboard for a draft response."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Готово",
                    callback_data=f"kwork:approve:{project_id}",
                ),
                InlineKeyboardButton(
                    text="✏️ Редактировать",
                    callback_data=f"kwork:edit:{project_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data=f"kwork:cancel:{project_id}",
                ),
            ]
        ]
    )
