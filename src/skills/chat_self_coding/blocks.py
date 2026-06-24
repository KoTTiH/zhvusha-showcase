"""Block message formatting for ``chat_self_coding`` (Phase 40).

The chat-mode skill emits a small set of named blocks during a self-coding
cycle. Plan and final report are rendered as HTML blocks; the Editor
progress blocks are intended to update one Telegram status message for the
current spec, so the chat does not fill with repeated progress bars. Long
Architect waits use one editable progress message rather than a stream of
chatter like «написала 12 строк» or «запустила pytest».

Output is always HTML-mode ready (``<b>`` headers, escaped user-supplied
text). The skill that owns these messages is responsible for sending
them with ``parse_mode="HTML"``. Pure functions, no side effects.

Translation contract: technical jargon (RED/GREEN, coverage, whitelist,
contract test, refactor kind, payload) must NOT appear in formatted
output — that's the orchestration-language requirement from Phase 40.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanBlock:
    """📋 — Architect-drafted plan, awaiting approval."""

    architectural_summary: str
    affected_files: tuple[str, ...]
    tier: int
    slug: str
    verification: str = ""
    deliverables: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    preserve_items: tuple[str, ...] = ()
    preserve_count: int = 0
    risk_count: int = 0
    allowed_simplifications: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckResult:
    """A single post-cycle check (тесты / стиль / типы)."""

    name: str
    passed: bool


@dataclass(frozen=True)
class DoneBlock:
    """✅ / ⚠️ — Editor cycle finished with an applied or mergeable result."""

    architectural_description: str
    files: tuple[str, ...]
    checks: tuple[CheckResult, ...]
    branch: str = ""
    commit_sha: str = ""
    backend: str = ""
    test_count_delta: int = 0
    allowed_simplifications: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodeProgressBlock:
    """🔧 / ✏️ — Real Editor pipeline stage status."""

    percent: int
    detail: str
    facts: tuple[str, ...] = ()
    stage: str = ""
    elapsed_seconds: int | None = None


@dataclass(frozen=True)
class ErrorBlock:
    """❌ — Cycle failed at some stage; user needs to redirect."""

    architectural_reason: str
    next_step: str


@dataclass(frozen=True)
class ProgressBlock:
    """🎯 — Editable Architect progress status."""

    percent: int
    detail: str
    stage: str = ""
    elapsed_seconds: int | None = None


# ---------------------------------------------------------------------------
# Tier description (architectural-orchestrator language)
# ---------------------------------------------------------------------------

_TIER_DESCRIPTION: dict[int, str] = {
    1: "Это простая задача (Tier 1).",
    2: "Это более сложная задача (Tier 2) — затрагивает существующие модули.",
    3: (
        "Это архитектурная задача (Tier 3): без твоего явного разрешения "
        "я её не запускаю."
    ),
}


def _describe_tier(tier: int) -> str:
    return _TIER_DESCRIPTION.get(tier, f"Это задача Tier {tier}.")


# ---------------------------------------------------------------------------
# 📋 План
# ---------------------------------------------------------------------------


def format_plan(block: PlanBlock) -> str:
    summary = escape(block.architectural_summary)
    tier_line = escape(_describe_tier(block.tier))
    verification = escape(block.verification or "проверка будет взята из spec")
    if block.tier >= 3:
        slug = escape(block.slug)
        return (
            f"<b>📋 Tier 3</b>\n\n"
            f"{summary}\n\n"
            f"{tier_line}\n"
            "Коротко: это меняет архитектурный/safety слой, поэтому сначала "
            "решаем с тобой в чате.\n\n"
            f"Проверка: {verification}.\n"
            f"Детали покажу только по запросу: <code>/spec show {slug}</code>.\n\n"
            "Ответь обычным текстом: запускать, править или что обсудить перед "
            "решением."
        )
    simplifications = (
        "нет"
        if not block.allowed_simplifications
        else "; ".join(escape(item) for item in block.allowed_simplifications)
    )
    scope = len(block.affected_files)
    deliverables = _format_bullets("Что появится", block.deliverables)
    safety = _format_bullets("Контроль и риски", block.safety_notes)
    preserve = _format_bullets("Что сохраню", block.preserve_items)
    return (
        f"<b>📋 План</b>\n\n"
        f"{summary}\n\n"
        f"{deliverables}"
        f"{safety}"
        f"{preserve}"
        f"Проверка: {verification}.\n"
        f"Граница: {scope} рабочих поверхностей, {block.risk_count} рисков, "
        f"{block.preserve_count} условий сохранить.\n"
        f"Упрощения: {simplifications}.\n"
        f"{tier_line}\n\n"
        f"Можем обсудить правки к плану. Чтобы начать реализацию, скажи «делай»."
    )


def _format_bullets(title: str, items: tuple[str, ...]) -> str:
    cleaned = tuple(item.strip() for item in items if item.strip())
    if not cleaned:
        return ""
    bullets = "\n".join(f"- {escape(item)}" for item in cleaned[:7])
    return f"{escape(title)}:\n{bullets}\n\n"


# ---------------------------------------------------------------------------
# 🔧 Подготовка / ✏️ Реализация — static
# ---------------------------------------------------------------------------


def _format_facts(facts: tuple[str, ...]) -> str:
    if not facts:
        return ""
    escaped = "; ".join(escape(_humanize_fact(fact)) for fact in facts)
    return f"\nКонтекст: {escaped}."


def _humanize_fact(fact: str) -> str:
    replacements = {
        "archive context:": "архивный контекст:",
        "baseline tests:": "тестов до старта:",
        "base:": "база:",
        "isolation:": "изоляция:",
        "whitelist paths:": "разрешённых путей:",
        "worktree:": "рабочая копия:",
    }
    for prefix, replacement in replacements.items():
        if fact.startswith(prefix):
            return f"{replacement}{fact.removeprefix(prefix)}"
    return fact


def _implementation_checkpoint(percent: int) -> str:
    if percent >= 85:
        return "commit gate пройден, идёт финальная проверка"
    if percent >= 70:
        return "правки завершены, идут проверки"
    if percent >= 40:
        return "Codex Editor запущен"
    if percent >= 15:
        return "подготовка рабочей копии"
    return "ожидаю подтверждённый этап"


def _architect_checkpoint(percent: int) -> str:
    if percent >= 100:
        return "план собран"
    if percent >= 20:
        return "Architect собирает spec и контекст"
    if percent > 0:
        return "задача принята в работу"
    return "ожидаю подтверждённый этап"


def _format_code_progress(
    header: str, block: CodeProgressBlock, *, show_numeric_progress: bool = True
) -> str:
    percent = max(0, min(100, block.percent))
    detail = escape(block.detail)
    facts = _format_facts(block.facts)
    stage = f"\nШаг: {escape(block.stage)}." if block.stage else ""
    elapsed = (
        f"\nПрошло: {block.elapsed_seconds} сек."
        if block.elapsed_seconds is not None
        else ""
    )
    marker = (
        f"{_progress_bar(percent)} {percent}%"
        if show_numeric_progress
        else f"Подтверждённый этап: {_implementation_checkpoint(percent)}."
    )
    return f"{header}\n\n{marker}\n\nСейчас: {detail}{stage}{elapsed}{facts}"


def format_preparation(block: CodeProgressBlock | None = None) -> str:
    if block is None:
        block = CodeProgressBlock(
            percent=15,
            detail="Готовлю временную рабочую копию.",
        )
    return _format_code_progress(
        "<b>🔧 Подготовка</b>",
        block,
        show_numeric_progress=False,
    )


def format_implementation(block: CodeProgressBlock | None = None) -> str:
    if block is None:
        block = CodeProgressBlock(percent=45, detail="Пишу код.")
    return _format_code_progress(
        "<b>✏️ Реализация</b>",
        block,
        show_numeric_progress=False,
    )


# ---------------------------------------------------------------------------
# 🎯 Architect progress
# ---------------------------------------------------------------------------


def _progress_bar(percent: int) -> str:
    clamped = max(0, min(100, percent))
    width = 20
    filled = width if clamped == 100 else clamped * width // 100
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_architect_progress(block: ProgressBlock) -> str:
    percent = max(0, min(100, block.percent))
    detail = escape(block.detail)
    stage = f"\nЭтап: {escape(block.stage)}" if block.stage else ""
    elapsed = (
        f"\nЖду: {block.elapsed_seconds} сек."
        if block.elapsed_seconds is not None
        else ""
    )
    return (
        f"<b>🎯 Жвуша собирает план</b>\n\n"
        f"Подтверждённый этап: {_architect_checkpoint(percent)}.\n"
        f"{detail}{stage}{elapsed}"
    )


# ---------------------------------------------------------------------------
# ✅ / ⚠️ Готово
# ---------------------------------------------------------------------------


def _format_check(check: CheckResult) -> str:
    label = escape(check.name)
    verdict = "ок" if check.passed else "не прошли"
    return f"{label} {verdict}"


def format_done(block: DoneBlock) -> str:
    all_passed = all(c.passed for c in block.checks)
    header = "<b>✅ Готово</b>" if all_passed else "<b>⚠️ Готово, но с проблемами</b>"
    description = escape(block.architectural_description)
    checks = ", ".join(_format_check(c) for c in block.checks)
    commit = escape(block.commit_sha[:12]) if block.commit_sha else "нет commit"
    branch = escape(block.branch) if block.branch else "ветка не записана"
    backend = escape(block.backend) if block.backend else "backend не записан"
    changed_surface_count = len(block.files)
    test_delta = (
        "не изменилось"
        if block.test_count_delta == 0
        else f"{block.test_count_delta:+d}"
    )
    simplifications = (
        "нет"
        if not block.allowed_simplifications
        else "; ".join(escape(item) for item in block.allowed_simplifications)
    )
    if not all_passed:
        closing = "Что-то пошло не так. Что дальше?"
    elif block.branch.startswith("zhvusha/"):
        closing = 'Слить в основную ветку? Скажи "слей" или "посмотрю позже".'
    else:
        closing = "Уже применено в рабочую ветку. Можем идти дальше."
    return (
        f"{header}\n\n"
        f"Сделала: {description}\n\n"
        f"Проверила: {checks}.\n"
        f"Изменения: {changed_surface_count} рабочих поверхностей; "
        f"тестовый объём — {test_delta}; упрощения — {simplifications}.\n"
        f"Технически: {branch} @ {commit}; backend {backend}.\n\n"
        f"{closing}"
    )


# ---------------------------------------------------------------------------
# Error block (❌)
# ---------------------------------------------------------------------------


def format_error(block: ErrorBlock) -> str:
    reason = escape(block.architectural_reason)
    step = escape(block.next_step)
    return f"<b>❌ Не получилось</b>\n\n{reason}\n\n{step}"
