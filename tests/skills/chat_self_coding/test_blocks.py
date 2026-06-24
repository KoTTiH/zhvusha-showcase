"""Tests for ``chat_self_coding.blocks`` (Phase 40).

The chat-mode skill emits four block messages during a typical cycle —
📋 План → 🔧 Подготовка → ✏️ Реализация → ✅ Готово — plus an explicit
❌ block when something fails. Each block is a short Telegram message
with a bold emoji header and a one-or-two-sentence body. Formatting is
HTML-mode (``<b>``); the surrounding skill is responsible for sending
with ``parse_mode="HTML"``.

Bodies must speak in architectural-orchestrator language: «расширила
систему пресетов», not «added budget_seconds field». Tier is rendered in
human terms (простая / непростая) so the user sees the orchestration
weight, not just a numeric label.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

# ---------------------------------------------------------------------------
# Frozen-ness
# ---------------------------------------------------------------------------


class TestStructure:
    def test_plan_block_is_frozen(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock

        b = PlanBlock(
            architectural_summary="x",
            affected_files=("a.py",),
            tier=1,
            slug="x",
        )
        with pytest.raises(FrozenInstanceError):
            b.tier = 2  # type: ignore[misc]

    def test_done_block_is_frozen(self) -> None:
        from src.skills.chat_self_coding.blocks import CheckResult, DoneBlock

        b = DoneBlock(
            architectural_description="ok",
            files=("a.py",),
            checks=(CheckResult(name="тесты", passed=True),),
        )
        with pytest.raises(FrozenInstanceError):
            b.files = ()  # type: ignore[misc]

    def test_error_block_is_frozen(self) -> None:
        from src.skills.chat_self_coding.blocks import ErrorBlock

        b = ErrorBlock(architectural_reason="r", next_step="n")
        with pytest.raises(FrozenInstanceError):
            b.next_step = "x"  # type: ignore[misc]

    def test_progress_block_is_frozen(self) -> None:
        from src.skills.chat_self_coding.blocks import ProgressBlock

        b = ProgressBlock(percent=25, detail="Разбираю контекст.")
        with pytest.raises(FrozenInstanceError):
            b.percent = 50  # type: ignore[misc]

    def test_code_progress_block_is_frozen(self) -> None:
        from src.skills.chat_self_coding.blocks import CodeProgressBlock

        b = CodeProgressBlock(percent=40, detail="Запускаю Codex Editor.")
        with pytest.raises(FrozenInstanceError):
            b.detail = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 📋 План
# ---------------------------------------------------------------------------


class TestPlanBlock:
    def test_format_includes_emoji_and_bold_header(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="Расширю систему пресетов.",
            affected_files=("src/research/presets.py",),
            tier=1,
            slug="x",
        )
        text = format_plan(block)
        assert "📋" in text
        assert "<b>" in text and "</b>" in text
        assert "План" in text

    def test_format_includes_architectural_summary(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="Расширю систему research-пресетов.",
            affected_files=("a.py",),
            tier=1,
            slug="x",
        )
        text = format_plan(block)
        assert "Расширю систему research-пресетов." in text

    def test_format_summarises_scope_without_listing_files(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="...",
            affected_files=(
                "src/research/presets.py",
                "tests/research/test_presets.py",
            ),
            tier=1,
            slug="x",
        )
        text = format_plan(block)
        assert "src/research/presets.py" not in text
        assert "tests/research/test_presets.py" not in text
        assert "2 рабочих поверхностей" in text

    def test_format_includes_contract_facts(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="...",
            affected_files=("a.py",),
            tier=1,
            slug="x",
            verification="tests/test_x.py::test_keeps_contract",
            deliverables=("style-check перед публикацией",),
            safety_notes=("не публиковать приватные скрины",),
            preserve_items=("старые text-only drafts работают",),
            preserve_count=3,
            risk_count=2,
            allowed_simplifications=("не трогать runtime env",),
        )
        text = format_plan(block)
        assert "Что появится" in text
        assert "style-check перед публикацией" in text
        assert "Контроль и риски" in text
        assert "не публиковать приватные скрины" in text
        assert "Что сохраню" in text
        assert "старые text-only drafts работают" in text
        assert "tests/test_x.py::test_keeps_contract" in text
        assert "2 рисков" in text
        assert "3 условий сохранить" in text
        assert "не трогать runtime env" in text

    def test_format_describes_tier_in_human_terms(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        for tier, marker in (
            (1, "простая"),
            (2, "сложная"),
            (3, "без твоего явного разрешения"),
        ):
            block = PlanBlock(
                architectural_summary="...",
                affected_files=("a.py",),
                tier=tier,
                slug="x",
            )
            text = format_plan(block)
            assert marker in text.lower(), f"tier {tier}: {text!r}"

    def test_tier3_format_is_short_and_defers_details(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="Подключу личный Telegram MCP к runtime.",
            affected_files=("src/a.py", "src/b.py", "tests/test_a.py"),
            tier=3,
            slug="telegram-mcp-runtime",
            verification="tests/test_runtime.py::test_gate",
            deliverables=("полный список deliverables не должен уйти в чат",),
            safety_notes=("подробный risk note не должен уйти в чат",),
            preserve_items=("подробный preserve item не должен уйти в чат",),
            preserve_count=8,
            risk_count=5,
            allowed_simplifications=("детальная оговорка",),
        )

        text = format_plan(block)

        assert "Подключу личный Telegram MCP" in text
        assert "/spec show telegram-mcp-runtime" in text
        assert "полный список deliverables" not in text
        assert "подробный risk note" not in text
        assert "подробный preserve item" not in text
        assert "Граница:" not in text
        assert "Упрощения:" not in text

    def test_format_includes_tier_number(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="...",
            affected_files=("a.py",),
            tier=2,
            slug="x",
        )
        text = format_plan(block)
        assert "Tier 2" in text or "tier 2" in text.lower()

    def test_format_ends_with_discussion_and_run_instruction(self) -> None:
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="...",
            affected_files=("a.py",),
            tier=1,
            slug="x",
        )
        text = format_plan(block)
        assert "обсудить" in text.lower()
        assert "делай" in text.lower()
        assert "Одобряешь?" not in text

    def test_format_escapes_html_in_user_supplied_text(self) -> None:
        """Architectural summary may contain ``<`` or ``&`` — must be escaped."""
        from src.skills.chat_self_coding.blocks import PlanBlock, format_plan

        block = PlanBlock(
            architectural_summary="Заменю <foo> на &bar;.",
            affected_files=("a.py",),
            tier=1,
            slug="x",
        )
        text = format_plan(block)
        assert "&lt;foo&gt;" in text
        assert "&amp;bar;" in text


# ---------------------------------------------------------------------------
# 🔧 Подготовка / ✏️ Реализация (static)
# ---------------------------------------------------------------------------


class TestStaticBlocks:
    def test_preparation_has_emoji_and_short_body(self) -> None:
        from src.skills.chat_self_coding.blocks import format_preparation

        text = format_preparation()
        assert "🔧" in text
        assert "<b>" in text and "Подготовка" in text
        # Short body — no more than three lines counting the header.
        non_empty = [line for line in text.splitlines() if line.strip()]
        assert len(non_empty) <= 3

    def test_preparation_does_not_show_fake_numeric_percent(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            CodeProgressBlock,
            format_preparation,
        )

        text = format_preparation(
            CodeProgressBlock(
                percent=15,
                detail="Создала временную рабочую копию.",
                stage="подготовка",
            )
        )

        assert "15%" not in text
        assert "Подтверждённый этап: подготовка рабочей копии." in text

    def test_implementation_has_emoji_and_short_body(self) -> None:
        from src.skills.chat_self_coding.blocks import format_implementation

        text = format_implementation()
        assert "✏️" in text
        assert "<b>" in text and "Реализация" in text
        non_empty = [line for line in text.splitlines() if line.strip()]
        assert len(non_empty) <= 3

    def test_implementation_can_include_real_progress_facts(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            CodeProgressBlock,
            format_implementation,
        )

        text = format_implementation(
            CodeProgressBlock(
                percent=70,
                detail="Code-agent вернул результат.",
                facts=("backend: codex_cli", "commit: abc123"),
            )
        )
        assert "Подтверждённый этап: правки завершены" in text
        assert "70%" not in text
        assert "backend: codex_cli" in text
        assert "commit: abc123" in text
        assert "Сейчас: Code-agent вернул результат." in text

    def test_implementation_includes_real_stage_and_elapsed_time(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            CodeProgressBlock,
            format_implementation,
        )

        text = format_implementation(
            CodeProgressBlock(
                percent=40,
                detail="Codex Editor запущен во временной рабочей копии.",
                stage="agent работает",
                elapsed_seconds=12,
                facts=("worktree: isolated:visual",),
            )
        )

        assert "Шаг: agent работает." in text
        assert "Прошло: 12 сек." in text
        assert "рабочая копия: isolated:visual" in text
        assert "Контекст:" in text


# ---------------------------------------------------------------------------
# 🎯 Progress
# ---------------------------------------------------------------------------


class TestProgressBlock:
    def test_progress_uses_confirmed_stage_without_fake_percent(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            ProgressBlock,
            format_architect_progress,
        )

        text = format_architect_progress(
            ProgressBlock(percent=35, detail="Разбираю контекст.")
        )

        assert "Жвуша" in text
        assert "Подтверждённый этап:" in text
        assert "Architect собирает spec" in text
        assert "Codex Editor" not in text
        assert "[#######-------------]" not in text
        assert "35%" not in text

    def test_progress_escapes_detail(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            ProgressBlock,
            format_architect_progress,
        )

        text = format_architect_progress(
            ProgressBlock(percent=150, detail="<проверяю>")
        )

        assert "&lt;проверяю&gt;" in text

    def test_progress_includes_stage_and_elapsed_time(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            ProgressBlock,
            format_architect_progress,
        )

        text = format_architect_progress(
            ProgressBlock(
                percent=35,
                detail="Жду Architect.",
                stage="Architect работает",
                elapsed_seconds=45,
            )
        )

        assert "Этап: Architect работает" in text
        assert "Жду: 45 сек." in text


# ---------------------------------------------------------------------------
# ✅ Готово
# ---------------------------------------------------------------------------


def _all_passed_done() -> object:
    from src.skills.chat_self_coding.blocks import CheckResult, DoneBlock

    return DoneBlock(
        architectural_description="Расширила систему пресетов.",
        files=("src/research/presets.py", "tests/research/test_presets.py"),
        checks=(
            CheckResult(name="тесты", passed=True),
            CheckResult(name="стиль", passed=True),
            CheckResult(name="типы", passed=True),
        ),
        branch="zhvusha/research-presets",
        commit_sha="abcdef1234567890",
        backend="codex_cli",
        test_count_delta=2,
    )


class TestDoneBlock:
    def test_all_checks_passed_uses_check_emoji(self) -> None:
        from src.skills.chat_self_coding.blocks import format_done

        text = format_done(_all_passed_done())  # type: ignore[arg-type]
        assert "✅" in text
        assert "Готово" in text

    def test_failed_check_uses_warning_emoji(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            CheckResult,
            DoneBlock,
            format_done,
        )

        block = DoneBlock(
            architectural_description="x",
            files=("a.py",),
            checks=(
                CheckResult(name="тесты", passed=False),
                CheckResult(name="стиль", passed=True),
            ),
        )
        text = format_done(block)
        assert "⚠️" in text or "❌" in text
        assert "✅" not in text.split("\n")[0]  # not in header

    def test_format_includes_architectural_description(self) -> None:
        from src.skills.chat_self_coding.blocks import format_done

        text = format_done(_all_passed_done())  # type: ignore[arg-type]
        assert "Расширила систему пресетов." in text

    def test_format_summarises_runtime_result_without_listing_files(self) -> None:
        from src.skills.chat_self_coding.blocks import format_done

        text = format_done(_all_passed_done())  # type: ignore[arg-type]
        assert "src/research/presets.py" not in text
        assert "tests/research/test_presets.py" not in text
        assert "zhvusha/research-presets" in text
        assert "abcdef123456" in text
        assert "codex_cli" in text
        assert "+2" in text

    def test_format_summarises_checks(self) -> None:
        from src.skills.chat_self_coding.blocks import format_done

        text = format_done(_all_passed_done())  # type: ignore[arg-type]
        # «тесты прошли» / «стиль ок» / «типы ок» style.
        for label in ("тесты", "стиль", "типы"):
            assert label in text

    def test_all_passed_includes_merge_question(self) -> None:
        """Bottom of block must invite a merge — Phase 40 plan example
        ends with «Слить в основную ветку? Скажи "слей" или "посмотрю
        позже".» so we look for both the question mark and the verb,
        not strictly the trailing character."""
        from src.skills.chat_self_coding.blocks import format_done

        text = format_done(_all_passed_done())  # type: ignore[arg-type]
        assert "?" in text
        assert "слить" in text.lower() or "слей" in text.lower()

    def test_all_passed_on_base_branch_says_already_applied(self) -> None:
        from src.skills.chat_self_coding.blocks import (
            CheckResult,
            DoneBlock,
            format_done,
        )

        block = DoneBlock(
            architectural_description="Расширила систему пресетов.",
            files=("src/research/presets.py",),
            checks=(CheckResult(name="тесты", passed=True),),
            branch="main",
            commit_sha="abcdef1234567890",
            backend="codex_cli",
        )

        text = format_done(block)

        assert "уже применено" in text.lower()
        assert "слей" not in text.lower()


# ---------------------------------------------------------------------------
# Error block (❌)
# ---------------------------------------------------------------------------


class TestErrorBlock:
    def test_format_includes_error_emoji(self) -> None:
        from src.skills.chat_self_coding.blocks import ErrorBlock, format_error

        b = ErrorBlock(
            architectural_reason="Architect не смог сформулировать спеку.",
            next_step="Переформулируй задачу или дай больше контекста.",
        )
        text = format_error(b)
        assert "❌" in text

    def test_format_includes_reason_and_next_step(self) -> None:
        from src.skills.chat_self_coding.blocks import ErrorBlock, format_error

        b = ErrorBlock(
            architectural_reason="Тесты не прошли — Editor не смог найти решение.",
            next_step="Можешь посмотреть spec и попробовать ещё раз.",
        )
        text = format_error(b)
        assert "Editor не смог найти решение" in text
        assert "Можешь посмотреть spec" in text

    def test_format_escapes_html(self) -> None:
        from src.skills.chat_self_coding.blocks import ErrorBlock, format_error

        b = ErrorBlock(
            architectural_reason="<not html>",
            next_step="& try again",
        )
        text = format_error(b)
        assert "&lt;not html&gt;" in text
        assert "&amp; try again" in text
