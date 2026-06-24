"""IdeationToSpecSkill — DelegatedSkill that turns intent into a spec draft.

Phase 12. Architect half of the Architect/Editor split (Aider-style + Agent0
co-evolution pattern, see plan ``typed-growing-micali.md``).

The skill is **deliberately mockable**: the actual Codex backend call
is the ``sdk_runner`` callable passed at construction time. Production wiring
plugs in a real backend adapter; tests pass an ``AsyncMock``
returning a canned YAML string. Same shape, no LLM in CI.

Side effects (writes_filesystem, network_io_external, …) are declared even
though tests stub them out — the manifest reflects production behaviour.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

import structlog
import yaml

from src.skills.base import (
    AgentContext,
    DelegatedSkill,
    ExecutionPlan,
    SideEffect,
    SkillResult,
)
from src.skills.chat_self_coding.events import (
    BlockEvent,
    BlockEventType,
    NoopBlockPublisher,
)
from src.skills.ideation_to_spec.archive_context import ArchiveContextProvider
from src.skills.ideation_to_spec.prompts import (
    IDEATION_SYSTEM_PROMPT,
    build_user_prompt,
)
from src.skills.ideation_to_spec.self_critique import (
    SelfCritiqueRunner,
    self_critique_draft,
)
from src.skills.ideation_to_spec.spec_writer import (
    build_spec_from_draft,
    extract_clarification_request,
    extract_yaml_block,
    merge_research_citations,
    write_spec_to_disk,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.research.protocols import ResearchResult
    from src.skills.chat_self_coding.events import BlockPublisher
    from src.skills.spec_command.parser import PreviousAttempt

logger = structlog.get_logger()

_TRIGGERS: tuple[str, ...] = ("/spec_create", "/architect")
_NATURAL_SPEC_CREATE_PREFIXES: tuple[str, ...] = (
    "создай spec",
    "сделай spec",
    "сформируй spec",
    "подготовь spec",
)
_DEFAULT_RESEARCH_PRESET = "api_integration"
_DEFAULT_RESEARCH_BUDGET_SECONDS = 45.0
_CHAT_CONTEXT_HEADER = "Контекст предварительного обсуждения в режиме /код:"
_LEGACY_CHAT_CONTEXT_HEADER = (
    "Контекст предварительного обсуждения в режиме самокодинга:"
)
_CURRENT_COMMAND_HEADER = "Текущая команда Никиты:"
_MAX_CHAT_CONTEXT_LINES = 30


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _strip_natural_tail(original: str, prefix: str) -> str:
    pattern = r"^\s*" + r"\s+".join(re.escape(part) for part in prefix.split())
    pattern += r"\s*[:\-—]?\s*"
    return re.sub(pattern, "", original, count=1, flags=re.I).strip(" \t\n\r:-—")


def _extract_natural_spec_request(message: str) -> str | None:
    text = message.strip()
    normalized = _normalize_chat_route_text(text)
    for prefix in _NATURAL_SPEC_CREATE_PREFIXES:
        if normalized.startswith(prefix + " ") or normalized.startswith(prefix + ":"):
            request = _strip_natural_tail(text, prefix)
            return request or None
    return None


class _ResearchProtocol(Protocol):
    """Subset of ``ResearchService`` we depend on (kept narrow for tests)."""

    async def research(
        self, *, query: str, preset: str, budget_seconds: float
    ) -> ResearchResult: ...


class _SDKRunner(Protocol):
    """Callable wrapping the Codex Architect backend.

    The production runner receives the shared system/user prompts and yields
    the assistant's text reply. Tests substitute an ``AsyncMock`` that returns
    canned YAML.
    """

    async def __call__(self, *, system_prompt: str, user_prompt: str) -> str: ...


class _ArchiveContextProtocol(Protocol):
    async def previous_attempts(
        self, query: str, *, top_k: int = 3
    ) -> list[PreviousAttempt]: ...


class IdeationToSpecSkill(DelegatedSkill):
    """Architect: free-text request → validated tasks/<slug>.yaml on disk."""

    name: ClassVar[str] = "ideation_to_spec"
    description: ClassVar[str] = (
        "Architect: translate free-text request into tasks/*.yaml spec via Codex"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "strategist"

    triggers: ClassVar[list[str]] = list(_TRIGGERS)

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "medium"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.DELEGATES_TO_CODE_AGENT,
        SideEffect.CALLS_LLM,
        SideEffect.CALLS_LLM_TIER_STRATEGIST,
        SideEffect.READS_FILESYSTEM,
        SideEffect.READS_FROM_KB,
        SideEffect.WRITES_FILESYSTEM,
        SideEffect.SPAWNS_SUBPROCESS,
        SideEffect.NETWORK_IO_EXTERNAL,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    executor: ClassVar[str] = "codex_cli"
    max_duration_seconds: ClassVar[float] = 240.0

    def __init__(
        self,
        *,
        tasks_dir: Path,
        admin_user_id: int,
        research_service: _ResearchProtocol,
        sdk_runner: _SDKRunner,
        clock: Callable[[], datetime] | None = None,
        block_publisher: BlockPublisher | None = None,
        self_critique_runner: SelfCritiqueRunner | None = None,
        archive_context_provider: _ArchiveContextProtocol | None = None,
    ) -> None:
        self._tasks_dir = tasks_dir
        self._admin_user_id = admin_user_id
        self._research = research_service
        self._sdk_runner = sdk_runner
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        # Phase 40 — emit a PLAN block event when the spec lands so the
        # chat-mode skill can render it as a Telegram message. Defaults to
        # NoopBlockPublisher so callers that don't wire chat mode keep
        # the previous slash-only behaviour.
        self._block_publisher = block_publisher or NoopBlockPublisher()
        self._self_critique_runner = self_critique_runner
        self._archive_context = archive_context_provider or ArchiveContextProvider(None)

    def set_research_service(self, research_service: _ResearchProtocol) -> None:
        """Replace research service after runtime capability graph startup."""

        self._research = research_service

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        text = message.strip()
        if any(text.startswith(trigger) for trigger in _TRIGGERS):
            return 1.0
        if _extract_natural_spec_request(message):
            return 0.93
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        request = self._extract_request(message)
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="delegated",
            human_summary=f"Architect → spec for: {request[:120]}",
            estimated_tokens=20000,
            estimated_cost_usd=Decimal("0.30"),
            estimated_duration_seconds=self.max_duration_seconds,
            files_to_modify=[self._tasks_dir],
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=1,
            delegated_to=self.executor,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        # ``context`` is needed for the per-user block event channel
        # (Phase 40); already gated in can_handle for everything else.
        request = self._extract_request(message)
        if not request:
            return SkillResult(
                success=False,
                response="Опиши задачу после `/spec_create <text>` или `/architect <text>`.",
            )

        research = await self._research.research(
            query=request,
            preset=_DEFAULT_RESEARCH_PRESET,
            budget_seconds=_DEFAULT_RESEARCH_BUDGET_SECONDS,
        )
        previous_attempts = await self._load_previous_attempts(request)

        existing_slugs = self._existing_slugs()
        user_prompt = build_user_prompt(
            request=request,
            research_findings=list(research.citations),
            today=self._clock().date().isoformat(),
            existing_slugs=existing_slugs,
            previous_attempts=previous_attempts,
        )

        draft, early_result = await self._run_architect_for_draft(
            request=request,
            user_prompt=user_prompt,
        )
        if early_result is not None:
            return early_result
        assert draft is not None

        critique = await self_critique_draft(
            draft,
            tasks_dir=self._tasks_dir,
            runner=self._self_critique_runner,
        )
        if critique.blocking:
            return SkillResult(
                success=False,
                response=(
                    "Spec validation failed: "
                    f"Architect self-critique blocked spec: {critique.summary}"
                ),
                metadata={"self_critique": critique.summary},
            )

        try:
            spec = build_spec_from_draft(draft)
            spec = merge_research_citations(spec, list(research.citations))
            chat_context = _extract_chat_context_from_request(request)
            merged_previous_attempts = _merge_previous_attempts(
                list(spec.previous_attempts),
                previous_attempts,
            )
            if chat_context or merged_previous_attempts != list(spec.previous_attempts):
                spec = type(spec).model_validate(
                    {
                        **spec.model_dump(mode="python"),
                        "chat_context": chat_context,
                        "previous_attempts": [
                            attempt.model_dump(mode="python")
                            for attempt in merged_previous_attempts
                        ],
                    }
                )
        except Exception as exc:
            logger.warning(
                "ideation_validation_failed", error=str(exc), slug=draft.get("slug")
            )
            return SkillResult(
                success=False,
                response=f"Spec validation failed:\n{exc}",
            )

        path = write_spec_to_disk(
            tasks_dir=self._tasks_dir, spec=spec, now=self._clock()
        )
        logger.info(
            "ideation_spec_written",
            slug=spec.slug,
            tier=spec.tier,
            path=str(path),
        )
        # Phase 40 — surface the plan to chat mode. The bot listener
        # passes the technical summary through the translator before
        # rendering it as a 📋 План block.
        await self._block_publisher.publish(
            BlockEvent(
                user_id=context.user_id,
                event_type=BlockEventType.PLAN,
                slug=spec.slug,
                task_id=_runtime_code_task_id(context),
                payload={
                    "summary": spec.goal,
                    "files": list(spec.whitelist_paths),
                    "tier": spec.tier,
                    "verification": (
                        f"{spec.failing_test.file}::{spec.failing_test.name}"
                    ),
                    "deliverables": _plan_deliverables_from_spec(spec),
                    "safety_notes": _plan_safety_notes_from_spec(spec),
                    "preserve_items": spec.preserve_behavior[:3],
                    "preserve_count": len(spec.preserve_behavior),
                    "risk_count": len(spec.blast_radius),
                    "allowed_simplifications": list(spec.allowed_simplifications),
                },
            )
        )
        return SkillResult(
            success=True,
            response=(
                f"Spec черновик `{spec.slug}` готов (tier {spec.tier}). "
                f"Файл: `{path.name}`. "
                f"Используй `/spec show {spec.slug}` для просмотра, "
                f"`/spec approve {spec.slug}` чтобы запустить (Phase 13 — implement_spec)."
            ),
            # Phase 40 — chat-mode skill reads metadata to bind the new
            # slug to the session state. Slash-command callers ignore it.
            metadata={"slug": spec.slug, "tier": spec.tier},
        )

    async def _run_architect_for_draft(
        self,
        *,
        request: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any] | None, SkillResult | None]:
        try:
            sdk_output = await self._run_sdk_with_timeout(user_prompt=user_prompt)
        except TimeoutError:
            logger.warning(
                "ideation_sdk_timeout",
                request=request[:100],
                timeout_seconds=self.max_duration_seconds,
            )
            return None, SkillResult(
                success=False,
                response=(
                    "Architect backend не ответил за "
                    f"{_format_seconds(self.max_duration_seconds)}."
                ),
                metadata={"timeout_seconds": self.max_duration_seconds},
            )
        except Exception:
            logger.exception("ideation_sdk_failed", request=request[:100])
            return None, SkillResult(
                success=False,
                response="Architect backend не ответил. Логи в structlog.",
            )

        draft, failure = self._parse_or_clarify_architect_output(
            sdk_output,
            request=request,
        )
        if draft is not None or failure is not None:
            return draft, failure

        retry_prompt = _build_yaml_retry_prompt(
            user_prompt=user_prompt,
            sdk_output=sdk_output,
            error="initial parse failed",
        )
        try:
            retry_output = await self._run_sdk_with_timeout(user_prompt=retry_prompt)
        except TimeoutError:
            logger.warning(
                "ideation_sdk_retry_timeout",
                request=request[:100],
                timeout_seconds=self.max_duration_seconds,
            )
            return None, SkillResult(
                success=False,
                response=(
                    "Architect backend не ответил за "
                    f"{_format_seconds(self.max_duration_seconds)} "
                    "при повторной YAML-попытке."
                ),
                metadata={
                    "timeout_seconds": self.max_duration_seconds,
                    "timeout_stage": "yaml_retry",
                },
            )
        except Exception:
            logger.exception("ideation_sdk_retry_failed", request=request[:100])
            return None, SkillResult(
                success=False,
                response="Architect backend не ответил при повторной YAML-попытке.",
            )
        return self._parse_or_clarify_architect_output(
            retry_output,
            request=request,
            is_retry=True,
        )

    async def _run_sdk_with_timeout(self, *, user_prompt: str) -> str:
        return await asyncio.wait_for(
            self._sdk_runner(
                system_prompt=IDEATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            ),
            timeout=self.max_duration_seconds,
        )

    def _parse_or_clarify_architect_output(
        self,
        sdk_output: str,
        *,
        request: str,
        is_retry: bool = False,
    ) -> tuple[dict[str, Any] | None, SkillResult | None]:
        clarification = extract_clarification_request(sdk_output)
        if clarification is not None:
            logger.info(
                "ideation_clarification_needed_after_retry"
                if is_retry
                else "ideation_clarification_needed",
                request=request[:100],
                question=clarification[:200],
            )
            return None, SkillResult(
                success=False,
                response=clarification,
                metadata={"needs_clarification": True},
            )
        try:
            return _parse_sdk_yaml_draft(sdk_output), None
        except (yaml.YAMLError, ValueError) as exc:
            event = (
                "ideation_yaml_parse_retry_failed"
                if is_retry
                else "ideation_yaml_parse_failed"
            )
            logger.warning(event, error=str(exc))
            if is_retry:
                return None, SkillResult(
                    success=False,
                    response=f"Architect вернул не YAML — parse failed: {exc}",
                )
            return None, None

    @staticmethod
    def _extract_request(message: str) -> str:
        text = message.strip()
        for trigger in _TRIGGERS:
            if text.startswith(trigger):
                return text[len(trigger) :].strip()
        natural = _extract_natural_spec_request(text)
        if natural is not None:
            return natural
        return text

    def _existing_slugs(self) -> list[str]:
        if not self._tasks_dir.exists():
            return []
        slugs: list[str] = []
        for path in self._tasks_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue
            if isinstance(data, dict) and isinstance(data.get("slug"), str):
                slugs.append(data["slug"])
        return slugs

    async def _load_previous_attempts(self, request: str) -> list[PreviousAttempt]:
        try:
            return await self._archive_context.previous_attempts(request, top_k=3)
        except Exception:
            logger.warning("ideation_archive_lookup_failed", exc_info=True)
            return []


def _format_seconds(seconds: float) -> str:
    value = float(seconds)
    if value.is_integer():
        return f"{int(value)} секунд"
    return f"{value:.2f} секунд"


def _extract_chat_context_from_request(request: str) -> list[str]:
    """Recover /код discussion context from the forwarded request.

    Chat mode passes recent dialogue as a plain-text preface to keep the legacy
    /spec_create surface unchanged. This helper makes that context durable in
    the YAML spec instead of trusting the Architect to restate it in rationale.
    """
    header = next(
        (
            candidate
            for candidate in (_CHAT_CONTEXT_HEADER, _LEGACY_CHAT_CONTEXT_HEADER)
            if candidate in request
        ),
        "",
    )
    if not header:
        return []
    _before, _sep, after_header = request.partition(header)
    discussion, _cmd_sep, _after_command = after_header.partition(
        _CURRENT_COMMAND_HEADER
    )
    lines = [line.strip() for line in discussion.splitlines() if line.strip()]
    return lines[-_MAX_CHAT_CONTEXT_LINES:]


def _runtime_code_task_id(context: AgentContext) -> str:
    raw = context.metadata.get("chat_self_coding_code_task_id", "")
    return str(raw).strip() if raw else ""


def _merge_previous_attempts(
    existing: list[PreviousAttempt], retrieved: list[PreviousAttempt]
) -> list[PreviousAttempt]:
    merged: list[PreviousAttempt] = []
    seen: set[str] = set()
    for attempt in [*existing, *retrieved]:
        if attempt.archive_slug in seen:
            continue
        seen.add(attempt.archive_slug)
        merged.append(attempt)
    return merged


def _parse_sdk_yaml_draft(sdk_output: str) -> dict[str, Any]:
    yaml_block = extract_yaml_block(sdk_output)
    draft = yaml.safe_load(yaml_block)
    if not isinstance(draft, dict):
        raise ValueError("YAML did not parse to a mapping")
    return draft


def _build_yaml_retry_prompt(
    *,
    user_prompt: str,
    sdk_output: str,
    error: str,
) -> str:
    return (
        f"{user_prompt}\n\n"
        "Previous Architect output failed YAML parsing.\n"
        f"Parse error:\n{error}\n\n"
        "Broken output excerpt:\n"
        f"{sdk_output[:4000]}\n\n"
        "Repair the draft. Return exactly one fenced ```yaml ... ``` block, "
        "with no prose before or after. Quote scalar values that contain ':'."
    )


def _plan_deliverables_from_spec(spec: Any) -> list[str]:
    items: list[str] = []
    if spec.failing_test.spec:
        items.append(str(spec.failing_test.spec))

    path_text = "\n".join(spec.whitelist_paths).lower()
    category_rules: tuple[tuple[str, str], ...] = (
        ("post_drafts", "черновики постов получат отдельный draft/style/visual слой"),
        ("visual", "появится планирование visual intent и metadata для визуалов"),
        (
            "channel_writer",
            "публикация канала сохранит старый text-only путь и новый media path",
        ),
        ("archive", "архив будет хранить результат публикации и связанные metadata"),
        (
            "agent_runtime",
            "agent runtime будет выдавать artifacts/capabilities для этого сценария",
        ),
        (
            "llm",
            "LLM-контур получит нужный контракт для генерации/подготовки визуала",
        ),
        ("telegram", "Telegram UX будет показывать preview/approval перед side effect"),
        (
            "chat_self_coding",
            "/код будет корректно вести обсуждение, plan и реализацию",
        ),
        (
            "monitoring",
            "закреплённый dashboard будет показывать актуальный runtime статус",
        ),
    )
    for marker, description in category_rules:
        if marker in path_text and description not in items:
            items.append(description)

    if spec.live_env_activation:
        keys = ", ".join(spec.live_env_keys) or "разрешённые runtime ключи"
        items.append(f"live .env activation только для явно разрешённых ключей: {keys}")

    return items[:7]


def _plan_safety_notes_from_spec(spec: Any) -> list[str]:
    notes = list(spec.blast_radius[:3])
    if spec.tier == 3:
        notes.insert(
            0, "Tier 3: после реализации нужен усиленный architecture/safety review"
        )
    if spec.allowed_simplifications:
        notes.append(
            "есть явно разрешённые упрощения: "
            + "; ".join(spec.allowed_simplifications[:2])
        )
    else:
        notes.append("упрощения не разрешены: существующее поведение нужно сохранить")
    return notes[:5]
