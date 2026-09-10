"""Tests for standalone domain detection."""

from src.memory.domain import detect_domain


def test_domain_from_source_kwork():
    assert detect_domain("любой текст", source="kwork") == "kwork"


def test_domain_from_source_channel():
    assert detect_domain("любой текст", source="channel") == "content"
    assert detect_domain("любой текст", source="morning_session") == "content"


def test_domain_from_mode_assistant():
    assert detect_domain("привет, мне нужен бот", mode="assistant") == "outreach"


def test_domain_from_keywords():
    assert detect_domain("новый заказ на кворке за 5000") == "kwork"
    assert detect_domain("написать пост для канала") == "content"
    assert detect_domain("сделать лендинг для клиента") == "outreach"


def test_domain_default_chat():
    assert detect_domain("привет, как дела?") == "chat"
    assert detect_domain("что думаешь об этом?") == "chat"
