from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery
from src.llm.protocols import LLMResponse, LLMUsage
from src.skills.kwork_monitor.handlers import (
    _MAX_PROJECTS,
    _drafts,
    _projects,
    handle_approve,
    handle_cancel,
    handle_check,
    handle_draft,
    handle_edit,
    handle_skip,
    register_project,
)
from src.skills.kwork_monitor.models import DraftState, ProjectCard


def _stale_query_error() -> TelegramBadRequest:
    return TelegramBadRequest(
        method=AnswerCallbackQuery(callback_query_id="x"),
        message="query is too old and response timeout expired or query ID is invalid",
    )


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="sonnet", usage=LLMUsage())


def _card(id: int = 42, **kwargs) -> ProjectCard:
    defaults = {
        "id": id,
        "title": "Python Telegram бот",
        "description": "Нужен бот для автоматизации",
        "price": 5000,
        "offers": 3,
        "username": "client1",
        "url": f"https://kwork.ru/projects/{id}",
        "matched_keywords": ["python", "telegram"],
    }
    return ProjectCard(**{**defaults, **kwargs})


def _callback(data: str) -> MagicMock:
    cb = MagicMock()
    cb.data = data
    cb.answer = AsyncMock()
    cb.message = MagicMock()
    cb.message.message_id = 1
    cb.message.edit_text = AsyncMock()
    cb.message.edit_reply_markup = AsyncMock()
    cb.message.answer = AsyncMock()
    cb.message.delete = AsyncMock()
    return cb


@pytest.fixture(autouse=True)
def _clean_state():
    """Clear handler state between tests."""
    _projects.clear()
    _drafts.clear()
    yield
    _projects.clear()
    _drafts.clear()


def test_register_project():
    card = _card()
    register_project(card)
    assert _projects[42] == card


def test_register_project_evicts_oldest():
    for i in range(_MAX_PROJECTS):
        register_project(_card(id=i))
    assert len(_projects) == _MAX_PROJECTS

    # Adding one more should evict the oldest (id=0)
    register_project(_card(id=9999))
    assert len(_projects) == _MAX_PROJECTS
    assert 0 not in _projects
    assert 9999 in _projects


async def test_handle_check_evaluates_project():
    card = _card()
    register_project(card)

    cb = _callback("kwork:check:42")

    mock_router = MagicMock()
    mock_router.generate = AsyncMock(
        return_value=_llm_resp(
            "**Вердикт:** ОТКЛИКАЕМСЯ\n**Причина:** Подходит по стеку"
        ),
    )

    with patch(
        "src.skills.kwork_monitor.handlers.get_router", return_value=mock_router
    ):
        await handle_check(cb)

    cb.answer.assert_awaited_with("Оцениваю проект...")
    cb.message.edit_text.assert_awaited_once()
    call_kwargs = cb.message.edit_text.call_args
    assert "Оценка:" in call_kwargs[0][0]


async def test_handle_check_project_not_found():
    cb = _callback("kwork:check:999")
    await handle_check(cb)
    cb.answer.assert_awaited_with("Проект не найден", show_alert=True)


async def test_handle_draft_generates_response():
    card = _card()
    register_project(card)

    cb = _callback("kwork:draft:42")

    mock_router = MagicMock()
    mock_router.generate = AsyncMock(return_value=_llm_resp("Готов взяться за проект"))

    with patch(
        "src.skills.kwork_monitor.handlers.get_router", return_value=mock_router
    ):
        await handle_draft(cb)

    cb.answer.assert_awaited_with("Генерирую отклик...")
    cb.message.edit_text.assert_awaited_once()
    assert 42 in _drafts
    assert _drafts[42].draft_text == "Готов взяться за проект"


async def test_handle_draft_project_not_found():
    cb = _callback("kwork:draft:999")
    await handle_draft(cb)
    cb.answer.assert_awaited_with("Проект не найден", show_alert=True)


async def test_handle_skip_deletes_message():
    cb = _callback("kwork:skip:42")
    await handle_skip(cb)
    cb.answer.assert_awaited_with("Пропущено")
    cb.message.delete.assert_awaited_once()


async def test_handle_approve_sends_draft():
    _drafts[42] = DraftState(
        project_id=42,
        project_title="Python бот",
        draft_text="Готов взяться",
    )
    cb = _callback("kwork:approve:42")

    with patch("src.skills.kwork_monitor.handlers._schedule_delete") as mock_schedule:
        await handle_approve(cb)

    cb.answer.assert_awaited_with("Готово!")
    cb.message.edit_text.assert_awaited_once()
    call_text = cb.message.edit_text.call_args[0][0]
    assert "Готов взяться" in call_text
    assert "удалится через 5 мин" in call_text
    assert 42 not in _drafts  # cleaned up
    mock_schedule.assert_called_once()


async def test_handle_approve_draft_not_found():
    cb = _callback("kwork:approve:999")
    await handle_approve(cb)
    cb.answer.assert_awaited_with("Черновик не найден", show_alert=True)


async def test_handle_edit_shows_instruction():
    cb = _callback("kwork:edit:42")
    await handle_edit(cb)
    cb.answer.assert_awaited_once()
    call_kwargs = cb.answer.call_args
    assert call_kwargs.kwargs.get("show_alert") is True


async def test_handle_cancel_deletes_message():
    _drafts[42] = DraftState(
        project_id=42,
        project_title="Python бот",
        draft_text="draft",
    )
    cb = _callback("kwork:cancel:42")
    await handle_cancel(cb)

    cb.answer.assert_awaited_with("Отменено")
    cb.message.delete.assert_awaited_once()
    assert 42 not in _drafts


async def test_handle_skip_survives_stale_callback():
    """Stale callback ack must not abort the skip — message still gets deleted."""
    cb = _callback("kwork:skip:42")
    cb.answer.side_effect = _stale_query_error()

    await handle_skip(cb)

    cb.message.delete.assert_awaited_once()


async def test_handle_approve_survives_stale_callback():
    """Stale callback ack must not abort approve — draft message still edited."""
    _drafts[42] = DraftState(
        project_id=42,
        project_title="Python бот",
        draft_text="Готов взяться",
    )
    cb = _callback("kwork:approve:42")
    cb.answer.side_effect = _stale_query_error()

    with patch("src.skills.kwork_monitor.handlers._schedule_delete"):
        await handle_approve(cb)

    cb.message.edit_text.assert_awaited_once()
    assert 42 not in _drafts


async def test_handle_cancel_survives_stale_callback():
    """Stale callback ack must not abort cancel — message still gets deleted."""
    _drafts[42] = DraftState(
        project_id=42,
        project_title="Python бот",
        draft_text="draft",
    )
    cb = _callback("kwork:cancel:42")
    cb.answer.side_effect = _stale_query_error()

    await handle_cancel(cb)

    cb.message.delete.assert_awaited_once()
    assert 42 not in _drafts
