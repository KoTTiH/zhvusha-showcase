"""Codex CLI worker adapter for Agent Runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus
from src.skills.code_agent.protocols import CodeAgentResult, ExplorerRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.agent_runtime.models import AgentJob, ContextPack


class ExplorerBackend(Protocol):
    """Narrow subset of the code-agent backend used for read-only jobs."""

    async def run_explorer(self, request: ExplorerRequest) -> CodeAgentResult: ...


class CodexWorkerBackend:
    """AgentWorkerBackend wrapper around the existing Codex explorer path."""

    name = "codex_cli"

    def __init__(
        self,
        *,
        code_backend: ExplorerBackend,
        cwd: Path,
        model: str = "",
        reasoning_effort: str = "",
    ) -> None:
        self._code_backend = code_backend
        self._cwd = cwd
        self._model = model
        self._reasoning_effort = reasoning_effort

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        """Run a read-only Codex exploration and wrap output as a capsule."""
        return await self._run(
            job=job,
            context_pack=context_pack,
            progress_callback=None,
        )

    async def run_with_progress(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
        progress_callback: Callable[[str], Awaitable[None]],
    ) -> ContextCapsule:
        """Run Codex and stream explicit TG_STATUS lines into runtime events."""
        return await self._run(
            job=job,
            context_pack=context_pack,
            progress_callback=progress_callback,
        )

    async def _run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
        progress_callback: Callable[[str], Awaitable[None]] | None,
    ) -> ContextCapsule:
        request = ExplorerRequest(
            system_prompt=_system_prompt_for_job(job),
            user_prompt=_build_user_prompt(job=job, context_pack=context_pack),
            cwd=self._cwd,
            progress_callback=progress_callback,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            session_id=_active_state_value(
                context_pack.active_code_state, "codex_session_id"
            ),
            persist_session=_active_state_bool(
                context_pack.active_code_state, "codex_persist_session"
            ),
        )
        result = await self._code_backend.run_explorer(request)
        return _capsule_from_codex_text(result.text, session_id=result.session_id)

    async def cancel(self, job_id: str) -> bool:
        """No live process handle is kept in this adapter yet."""
        del job_id
        return False


def _build_user_prompt(*, job: AgentJob, context_pack: ContextPack) -> str:
    return "\n".join(
        [
            f"Agent Runtime job: {job.id}",
            f"kind: {job.kind}",
            f"profile: {job.profile.id}",
            "allowed_capabilities: "
            + ", ".join(job.profile.allowed_capabilities or ("(none)",)),
            "denied_capabilities: "
            + ", ".join(job.profile.denied_capabilities or ("(none)",)),
            "",
            "# User request",
            context_pack.user_request,
            "",
            "# Chat context",
            "\n".join(context_pack.chat_context) or "(none)",
            "",
            "# Attachments",
            "\n".join(context_pack.attachments) or "(none)",
            "",
            "# Follow-ups",
            "\n".join(job.followups) or "(none)",
            "",
            "# Relevant files",
            "\n".join(context_pack.relevant_files) or "(none)",
            "",
            "# Runtime artifacts",
            "\n".join(job.artifacts) or "(none)",
            "",
            "# Constraints",
            "\n".join(context_pack.constraints) or "(none)",
            "",
            _return_format_instruction(job),
        ]
    )


def _capsule_from_codex_text(text: str, *, session_id: str = "") -> ContextCapsule:
    stripped = text.strip()
    summary = _extract_summary(stripped)
    findings: tuple[Finding, ...] = ()
    finding_lines = _extract_labeled_lines(stripped, "FINDING")
    if finding_lines:
        findings = tuple(
            Finding(
                claim=line,
                status=FindingStatus.CONFIRMED,
                confidence=0.7,
            )
            for line in finding_lines
        )
    artifacts = list(_extract_labeled_lines(stripped, "ARTIFACT"))
    if session_id:
        artifacts.append(f"codex_session_id:{session_id}")
    return ContextCapsule(
        summary=summary,
        processed_context=stripped,
        findings=findings,
        sources=tuple(_extract_labeled_lines(stripped, "SOURCE")),
        artifacts=tuple(artifacts),
        memory_candidates=tuple(_extract_labeled_lines(stripped, "MEMORY")),
        next_actions=tuple(_extract_labeled_lines(stripped, "NEXT")),
        markdown_report=stripped,
    )


def _active_state_value(active_code_state: str, key: str) -> str:
    for line in active_code_state.splitlines():
        raw_key, sep, value = line.partition(":")
        if sep and raw_key.strip() == key:
            return value.strip()
    return ""


def _active_state_bool(active_code_state: str, key: str) -> bool:
    value = _active_state_value(active_code_state, key).lower()
    return value in {"1", "true", "yes", "on"}


def _extract_summary(text: str) -> str:
    for line in text.splitlines():
        normalized = line.strip()
        if normalized.startswith("SUMMARY:"):
            return normalized.removeprefix("SUMMARY:").strip()
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    return first or "Codex worker вернул пустой ответ."


def _extract_labeled_lines(text: str, label: str) -> list[str]:
    prefix = f"{label}:"
    return [
        line.removeprefix(prefix).strip()
        for line in text.splitlines()
        if line.strip().startswith(prefix)
    ]


def _system_prompt_for_job(job: AgentJob) -> str:
    if job.profile.id == "self_coding.readonly_discussion":
        return _SELF_CODING_DISCUSSION_SYSTEM_PROMPT
    return _CODEX_WORKER_SYSTEM_PROMPT


def _return_format_instruction(job: AgentJob) -> str:
    if job.profile.id == "self_coding.readonly_discussion":
        return (
            "Return a normal Telegram chat answer as Жвуша. Do not use labeled "
            "capsule lines like SUMMARY:, FINDING:, SOURCE:, ARTIFACT:, MEMORY: "
            "or NEXT:. Keep evidence and uncertainty in prose."
        )
    return (
        "Return an evidence-backed answer. Include labeled capsule lines when "
        "applicable: SUMMARY:, FINDING:, SOURCE:, ARTIFACT:, MEMORY:, NEXT:."
    )


_CODEX_WORKER_SYSTEM_PROMPT = """\
Ты Codex Worker внутри Agent Runtime ZHVUSHA.

Работай строго в рамках выданного capability profile. Если capability не выдана,
не делай вид, что действие выполнено. Для read-only задач не изменяй файлы, не
делай commit, не трогай env, не запускай restart и не публикуй наружу.

Итог должен быть пригоден для Context Capsule: короткий summary, проверенный
контекст, findings с evidence, unknowns и next actions. Для машинного чтения
используй строки с префиксами SUMMARY:, FINDING:, SOURCE:, ARTIFACT:, MEMORY:,
NEXT: там, где это уместно.
"""


_SELF_CODING_DISCUSSION_SYSTEM_PROMPT = """\
Ты Жвуша в режиме /код, но сейчас это read-only discussion, не реализация.

Работай строго в рамках выданного capability profile: можно читать код, workspace,
вложения и выполнять read-only проверки. Нельзя менять файлы, env, делать commit,
restart, publish или browser submit.

Отвечай Никите как собеседница и инженер: по-русски, конкретно, в женском роде
Жвуши ("посмотрела", "проверила", "собрала"). Не зеркаль мужской род из текста
Никиты и не переходи в голос generic Codex.

Не возвращай machine-capsule формат SUMMARY/FINDING/SOURCE/ARTIFACT/MEMORY/NEXT.
Если изучала код или файлы, дай нормальный человеческий вывод: что поняла, какие
варианты есть, где риск, что можно оформить в plan. Источники можно упоминать
внутри фраз, но без протокольных заголовков.
"""
