from src.skills.kwork_monitor.formatting import (
    MAX_DESCRIPTION_LENGTH,
    build_draft_keyboard,
    build_evaluate_keyboard,
    build_project_keyboard,
    format_project_card,
)
from src.skills.kwork_monitor.models import ProjectCard


def _card(**kwargs) -> ProjectCard:
    defaults = {
        "id": 1,
        "title": "Тестовый проект",
        "description": "Нужен Python разработчик",
        "price": 5000,
        "offers": 3,
        "username": "client1",
        "url": "https://kwork.ru/projects/1",
        "matched_keywords": ["python"],
    }
    return ProjectCard(**{**defaults, **kwargs})


def test_format_card_contains_title():
    html = format_project_card(_card())
    assert "<b>Тестовый проект</b>" in html


def test_format_card_contains_price():
    html = format_project_card(_card(price=5000))
    assert "5000 руб." in html


def test_format_card_none_price():
    html = format_project_card(_card(price=None))
    assert "не указан" in html


def test_format_card_contains_offers():
    html = format_project_card(_card(offers=3))
    assert "3" in html


def test_format_card_none_offers():
    html = format_project_card(_card(offers=None))
    assert "?" in html


def test_format_card_contains_keywords():
    html = format_project_card(_card(matched_keywords=["python", "telegram"]))
    assert "python, telegram" in html


def test_format_card_contains_url():
    html = format_project_card(_card(url="https://kwork.ru/projects/42"))
    assert "https://kwork.ru/projects/42" in html


def test_format_card_truncates_long_description():
    long_desc = "x" * 1000
    html = format_project_card(_card(description=long_desc))
    assert "..." in html
    # The description in the output should be truncated
    assert "x" * (MAX_DESCRIPTION_LENGTH + 1) not in html


def test_project_keyboard_has_two_buttons():
    kb = build_project_keyboard(42)
    buttons = kb.inline_keyboard[0]
    assert len(buttons) == 2


def test_project_keyboard_callback_data():
    kb = build_project_keyboard(42)
    buttons = kb.inline_keyboard[0]
    assert buttons[0].callback_data == "kwork:check:42"
    assert buttons[1].callback_data == "kwork:skip:42"


def test_evaluate_keyboard_has_two_buttons():
    kb = build_evaluate_keyboard(42)
    buttons = kb.inline_keyboard[0]
    assert len(buttons) == 2


def test_evaluate_keyboard_callback_data():
    kb = build_evaluate_keyboard(42)
    buttons = kb.inline_keyboard[0]
    assert buttons[0].callback_data == "kwork:draft:42"
    assert buttons[1].callback_data == "kwork:skip:42"


def test_draft_keyboard_has_three_buttons():
    kb = build_draft_keyboard(42)
    buttons = kb.inline_keyboard[0]
    assert len(buttons) == 3


def test_draft_keyboard_callback_data():
    kb = build_draft_keyboard(42)
    buttons = kb.inline_keyboard[0]
    assert buttons[0].callback_data == "kwork:approve:42"
    assert buttons[1].callback_data == "kwork:edit:42"
    assert buttons[2].callback_data == "kwork:cancel:42"
