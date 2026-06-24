"""Tests for observation masking."""

from __future__ import annotations

from src.utils.observation_mask import (
    mask_file_contents,
    mask_memory_facts,
    mask_observations,
)


def test_text_without_tags_unchanged() -> None:
    text = "привет, как дела? Всё хорошо."
    assert mask_observations(text) == text


def test_masks_file_content_block() -> None:
    text = (
        "Вот что я нашёл:\n"
        '<FILE_CONTENT source="diary/2026-04-02.md" read_at="2026-04-02T12:00:00">\n'
        "Сегодня был хороший день. Много работал.\n"
        "</FILE_CONTENT>\n"
        "Надеюсь тебе понравилось!"
    )
    result = mask_file_contents(text)

    assert "[Прочитан файл: diary/2026-04-02.md]" in result
    assert "Сегодня был хороший день" not in result
    assert "Надеюсь тебе понравилось!" in result


def test_masks_two_file_content_blocks() -> None:
    text = (
        '<FILE_CONTENT source="diary/day1.md" read_at="t1">\nday1\n</FILE_CONTENT>\n'
        "between\n"
        '<FILE_CONTENT source="diary/day2.md" read_at="t2">\nday2\n</FILE_CONTENT>'
    )
    result = mask_file_contents(text)

    assert "[Прочитан файл: diary/day1.md]" in result
    assert "[Прочитан файл: diary/day2.md]" in result
    assert "between" in result
    assert "day1\n" not in result
    assert "day2\n" not in result


def test_masks_memory_facts_block() -> None:
    text = (
        "Контекст:\n"
        "<MEMORY_FACTS>\n"
        "Вчера обсуждали aiogram\n"
        "Никита работает над ботом\n"
        "</MEMORY_FACTS>\n"
        "Ответ: да, помню."
    )
    result = mask_memory_facts(text)

    assert "[Были извлечены факты из памяти]" in result
    assert "обсуждали aiogram" not in result
    assert "Ответ: да, помню." in result


def test_masks_mixed_content() -> None:
    text = (
        "Привет!\n"
        '<FILE_CONTENT source="notes.md" read_at="t">\nsecret\n</FILE_CONTENT>\n'
        "середина\n"
        "<MEMORY_FACTS>\nfact1\n</MEMORY_FACTS>\n"
        "конец"
    )
    result = mask_observations(text)

    assert "Привет!" in result
    assert "[Прочитан файл: notes.md]" in result
    assert "[Были извлечены факты из памяти]" in result
    assert "secret" not in result
    assert "fact1" not in result
    assert "середина" in result
    assert "конец" in result
