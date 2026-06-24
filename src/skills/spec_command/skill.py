"""SpecCommandSkill — InlineSkill for ``/spec list/show/approve/reject``.

Phase 11 of the self-improving architecture (see ``AGENTS.md`` for
the broader self-coding context). The skill is a thin
filesystem-and-formatter layer over ``tasks/*.yaml`` plus a two-tier
parser for natural-language approvals (keyword fast-match → LLM intent
classifier fallback).

What this skill does NOT do (yet):

* It does not invoke ``ImplementSpecSkill`` on approve — that lands in
  Phase 13. ``/spec approve`` only flips ``status: pending_approval →
  approved`` and writes ``approved_at`` / ``approved_by``. The downstream
  Editor will pick up approved specs once it exists.
* It does not call any LLM by default. If an ``intent_classifier``
  callable is injected, ambiguous free-text falls through to it; otherwise
  unrecognized text returns an empty response (so the chat fall-through
  reaches ``chat_response``).
"""

from __future__ import annotations

import re
import shlex
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, Protocol

import structlog

from src.skills.base import (
    AgentContext,
    InlineSkill,
    SideEffect,
    SkillResult,
)
from src.skills.spec_command.git_io import commit_yaml_mutation
from src.skills.spec_command.parser import SpecModel, SpecStatus
from src.skills.spec_command.store import (
    find_spec_path,
    list_spec_files,
    load_spec,
    load_spec_raw,
    save_spec_raw,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.llm.protocols import LLMGatewayProtocol

logger = structlog.get_logger()

# Active = neither terminal nor rejected; default filter for /spec list.
_ACTIVE_STATUSES: frozenset[SpecStatus] = frozenset(
    {SpecStatus.PENDING_APPROVAL, SpecStatus.APPROVED, SpecStatus.IN_PROGRESS}
)

_APPROVE_KEYWORDS: tuple[str, ...] = (
    "одобряю",
    "одобрено",
    "согласен",
    "согласна",
    "давай",
    "запускай",
    "погнали",
    "go",
    "ok",
    "approve",
    "approved",
)
_REJECT_KEYWORDS: tuple[str, ...] = (
    "отклоняю",
    "отклонено",
    "не сейчас",
    "позже",
    "отмена",
    "не надо",
    "reject",
    "rejected",
    "no",
    "nope",
)
_SPEC_LIST_PREFIXES: tuple[str, ...] = (
    "покажи specs",
    "покажи spec",
    "покажи спеки",
    "список specs",
    "список spec",
    "список спеков",
)
_SPEC_SHOW_PREFIXES: tuple[str, ...] = (
    "покажи spec",
    "покажи спек",
    "открой spec",
    "открой спек",
)
_SPEC_APPROVE_PREFIXES: tuple[str, ...] = (
    "одобри spec",
    "одобри спек",
    "approve spec",
)
_SPEC_REJECT_PREFIXES: tuple[str, ...] = (
    "отклони spec",
    "отклони спек",
    "reject spec",
)
_BUNDLE_SCOPE_FLAG = "--scope"
_BUNDLE_REASON_FLAG = "--reason"
_BUNDLE_SEPARATE_APPROVAL_ACTIONS: tuple[str, ...] = (
    "/spec_run implementation",
    "commit",
    "restart",
    ".env/secrets edits",
    "publish/network side effects",
    "private-source read",
    "external account mutation",
)
_CODEX_OPERATOR_ACTORS = {"codex", "codex_operator", "operator"}
_CODEX_GOAL_LOOP_OPERATOR_KINDS = {
    "goal_loop_handoff",
    "goal_loop_proof_replay",
}


class _ApproveBundleArgs(NamedTuple):
    slugs: list[str]
    scope: str
    reason: str


class _ApproveBundleOption(NamedTuple):
    key: Literal["scope", "reason"]
    value: str
    next_index: int


def _is_codex_goal_loop_operator_message(
    message: str,
    context: AgentContext,
) -> bool:
    """Operator handoffs are technical inputs, not Nikita's spec approval."""
    source_actor = str(context.metadata.get("source_actor", "") or "").casefold()
    if source_actor not in _CODEX_OPERATOR_ACTORS:
        return False
    message_kind = str(context.metadata.get("operator_message_kind", "") or "")
    if message_kind in _CODEX_GOAL_LOOP_OPERATOR_KINDS:
        return True
    normalized = " ".join(message.casefold().split())
    return (
        (
            "codex/operator handoff" in normalized
            or "codex/operator proof replay" in normalized
        )
        and "sender=codex" in normalized
        and "не никита" in normalized
    )


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _strip_natural_tail(original: str, prefix: str) -> str:
    pattern = r"^\s*" + r"\s+".join(re.escape(part) for part in prefix.split())
    pattern += r"\s*[:\-—]?\s*"
    return re.sub(pattern, "", original, count=1, flags=re.I).strip(" \t\n\r:-—")


def _command_from_slug_prefixes(
    *,
    original: str,
    normalized: str,
    prefixes: tuple[str, ...],
    subcommand: str,
) -> str | None:
    for prefix in prefixes:
        if normalized.startswith(prefix + " ") or normalized.startswith(prefix + ":"):
            slug = _strip_natural_tail(original, prefix).split(maxsplit=1)[0]
            if slug:
                return f"/spec {subcommand} {slug}"
    return None


def _command_from_args_prefixes(
    *,
    original: str,
    normalized: str,
    prefixes: tuple[str, ...],
    subcommand: str,
) -> str | None:
    for prefix in prefixes:
        if normalized.startswith(prefix + " ") or normalized.startswith(prefix + ":"):
            args = _strip_natural_tail(original, prefix)
            if args:
                return f"/spec {subcommand} {args}"
    return None


def _natural_spec_decision(
    message: str,
) -> tuple[Literal["approve", "reject"], str] | None:
    text = message.strip()
    normalized = _normalize_chat_route_text(text)
    decision_prefixes: tuple[
        tuple[Literal["approve", "reject"], tuple[str, ...]],
        ...,
    ] = (
        ("approve", _SPEC_APPROVE_PREFIXES),
        ("reject", _SPEC_REJECT_PREFIXES),
    )
    for action, prefixes in decision_prefixes:
        for prefix in prefixes:
            if normalized.startswith(prefix + " ") or normalized.startswith(
                prefix + ":"
            ):
                slug = _strip_natural_tail(text, prefix).split(maxsplit=1)[0]
                if slug:
                    return action, slug
    return None


def _normalize_message_to_command(message: str) -> str | None:
    text = message.strip()
    lower = text.lower()
    if lower == "/spec" or lower.startswith("/spec "):
        return text
    normalized = _normalize_chat_route_text(text)
    for prefixes, subcommand in (
        (_SPEC_SHOW_PREFIXES, "show"),
        (_SPEC_APPROVE_PREFIXES, "approve"),
    ):
        command = _command_from_slug_prefixes(
            original=text,
            normalized=normalized,
            prefixes=prefixes,
            subcommand=subcommand,
        )
        if command is not None:
            return command
    command = _command_from_args_prefixes(
        original=text,
        normalized=normalized,
        prefixes=_SPEC_REJECT_PREFIXES,
        subcommand="reject",
    )
    if command is not None:
        return command
    if normalized in _SPEC_LIST_PREFIXES:
        return "/spec list"
    return None


class ApprovalIntent(StrEnum):
    """Result of the natural-language intent classifier."""

    APPROVE = "approve"
    REJECT = "reject"
    CLARIFY = "clarify"
    IRRELEVANT = "irrelevant"


class IntentClassifier(Protocol):
    """Async callable: free-text + slug → ApprovalIntent."""

    async def __call__(self, text: str, slug: str) -> ApprovalIntent: ...


class LLMSpecApprovalClassifier:
    """LLM-backed approval classifier for natural-language spec decisions."""

    def __init__(self, *, llm_router: LLMGatewayProtocol) -> None:
        self._llm = llm_router

    async def __call__(self, text: str, slug: str) -> ApprovalIntent:
        from src.llm.protocols import LLMRequest

        response = await self._llm.generate(
            LLMRequest(
                prompt=(
                    f"Активный spec: {slug}\n"
                    f"Ответ Никиты:\n{text}\n\n"
                    "Классифицируй решение по spec."
                ),
                system=(
                    "Ты классифицируешь решение Никиты по pending self-coding "
                    "spec. Не опирайся на фиксированный список слов: оцени смысл "
                    "ответа в контексте обсуждения. approve только если Никита "
                    "явно разрешает запуск/реализацию именно этого spec. reject "
                    "если он отклоняет, переносит или запрещает запуск. clarify "
                    "если он задаёт вопрос, хочет обсудить, просит детали или "
                    "решение неоднозначно. irrelevant если сообщение не про "
                    "approval. Ответь одним словом: approve, reject, clarify, "
                    "irrelevant."
                ),
                tier="worker",
                temperature=0.0,
                caller="spec_command_approval",
            )
        )
        token = response.text.strip().lower()
        if token in {item.value for item in ApprovalIntent}:
            return ApprovalIntent(token)
        return ApprovalIntent.IRRELEVANT


class SpecCommandSkill(InlineSkill):
    """Inline skill exposing ``/spec list/show/approve/reject`` to the admin."""

    name: ClassVar[str] = "spec_command"
    description: ClassVar[str] = (
        "Manage tasks/*.yaml specs via /spec list/show/approve/reject"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"

    triggers: ClassVar[list[str]] = ["/spec"]

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_FILESYSTEM,
        SideEffect.CALLS_LLM,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        tasks_dir: Path,
        admin_user_id: int,
        intent_classifier: IntentClassifier | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self._tasks_dir = tasks_dir
        self._admin_user_id = admin_user_id
        self._intent_classifier = intent_classifier
        self._intent_cache: dict[tuple[str, str], ApprovalIntent] = {}
        # Phase 19 — when supplied, /spec approve and /spec reject will
        # auto-commit the yaml mutation so the next /spec_run starts on
        # a clean worktree. ``None`` keeps pre-Phase-19 behaviour for
        # callers (and tests) that don't run inside a git repo.
        self._repo_root = repo_root

    # ----------------------------------------------------------------- routing

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        if _is_codex_goal_loop_operator_message(message, context):
            return 0.0
        text = message.strip().lower()
        # Match only ``/spec`` or ``/spec <subcommand>`` — never ``/spec_run``
        # / ``/spec_create`` which belong to ImplementSpec / IdeationToSpec.
        if text == "/spec" or text.startswith("/spec "):
            return 1.0
        natural_confidence = await self._natural_command_confidence(message)
        if natural_confidence > 0.0:
            return natural_confidence
        pending_spec = self._latest_pending_spec()
        if pending_spec is None:
            return 0.0
        if pending_spec.tier >= 3:
            return await self._tier3_freetext_confidence(message, pending_spec)
        if any(kw in text for kw in _APPROVE_KEYWORDS):
            return 0.85
        if any(kw in text for kw in _REJECT_KEYWORDS):
            return 0.85
        # Free-text fallback only if a classifier is wired up; otherwise
        # we let chat_response handle the message.
        if self._intent_classifier is not None:
            return 0.6
        return 0.0

    async def _natural_command_confidence(self, message: str) -> float:
        natural_decision = _natural_spec_decision(message)
        if natural_decision is not None:
            _action, slug = natural_decision
            if self._spec_tier(slug) >= 3:
                return await self._tier3_explicit_natural_confidence(message, slug)
            return 0.93
        if _normalize_message_to_command(message) is not None:
            return 0.93
        return 0.0

    async def _tier3_freetext_confidence(
        self, message: str, pending_spec: SpecModel
    ) -> float:
        if self._intent_classifier is None:
            return 0.0
        text = message.strip()
        intent = await self._intent_classifier(text, pending_spec.slug)
        if intent in {ApprovalIntent.APPROVE, ApprovalIntent.REJECT}:
            self._intent_cache[(text, pending_spec.slug)] = intent
            return 0.85
        return 0.0

    async def _tier3_explicit_natural_confidence(
        self,
        message: str,
        slug: str,
    ) -> float:
        if self._intent_classifier is None:
            return 0.93
        text = message.strip()
        intent = await self._intent_classifier(text, slug)
        if intent in {
            ApprovalIntent.APPROVE,
            ApprovalIntent.REJECT,
            ApprovalIntent.CLARIFY,
        }:
            self._intent_cache[(text, slug)] = intent
            return 0.93
        return 0.0

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context  # already gated in can_handle
        raw_text = message.strip()
        natural_decision = _natural_spec_decision(raw_text)
        if natural_decision is not None:
            _action, slug = natural_decision
            if self._spec_tier(slug) >= 3:
                return await self._handle_tier3_explicit_natural_decision(
                    raw_text,
                    slug,
                )
        text = (_normalize_message_to_command(raw_text) or raw_text).strip()
        lower = text.lower()

        if lower.startswith("/spec list"):
            return self._cmd_list()
        if lower.startswith("/spec show"):
            return self._cmd_show(self._parse_arg(text, "/spec show"))
        if lower.startswith("/spec approve-bundle"):
            return self._cmd_approve_bundle(
                text.removeprefix("/spec approve-bundle").strip()
            )
        if lower.startswith("/spec approve"):
            return self._cmd_approve(self._parse_arg(text, "/spec approve"))
        if lower.startswith("/spec reject"):
            return self._cmd_reject(text.removeprefix("/spec reject").strip())
        # Match ``/spec`` or ``/spec <unknown>`` only — never ``/spec_run``
        # / ``/spec_create`` (those belong to sibling skills).
        if lower == "/spec" or lower.startswith("/spec "):
            return SkillResult(success=True, response=_HELP_TEXT)

        return await self._handle_freetext(text, lower)

    async def _handle_tier3_explicit_natural_decision(
        self,
        text: str,
        slug: str,
    ) -> SkillResult:
        if self._intent_classifier is None:
            return SkillResult(
                success=True,
                response=(
                    f"Tier 3 spec `{slug}` требует AI-классификации свободного "
                    "ответа. Используй точную slash-команду, если это "
                    f"административное действие: `/spec approve {slug}`."
                ),
            )
        cached_intent = self._intent_cache.pop((text, slug), None)
        intent = (
            cached_intent
            if cached_intent is not None
            else await self._intent_classifier(text, slug)
        )
        return self._dispatch_intent(intent, slug, text)

    async def _handle_freetext(self, text: str, lower: str) -> SkillResult:
        """Approval / rejection from natural language; classifier fallback."""
        pending_spec = self._latest_pending_spec()
        if pending_spec is None:
            return SkillResult(success=False, response="")
        pending = pending_spec.slug

        if pending_spec.tier >= 3:
            if self._intent_classifier is None:
                return SkillResult(
                    success=True,
                    response=(
                        f"Tier 3 spec `{pending}` требует обсуждения и "
                        "AI-классификации свободного ответа. Коротко напиши, "
                        "разрешаешь запуск или что обсудить; детали покажу по "
                        f"`/spec show {pending}`."
                    ),
                )
            cached_intent = self._intent_cache.pop((text, pending), None)
            if cached_intent is not None:
                return self._dispatch_intent(cached_intent, pending, text)
            return self._dispatch_intent(
                await self._intent_classifier(text, pending), pending, text
            )

        if any(kw in lower for kw in _APPROVE_KEYWORDS):
            return self._cmd_approve(pending)
        if any(kw in lower for kw in _REJECT_KEYWORDS):
            return self._cmd_reject(f"{pending} {text}")

        if self._intent_classifier is None:
            return SkillResult(success=False, response="")

        return self._dispatch_intent(
            await self._intent_classifier(text, pending), pending, text
        )

    def _dispatch_intent(
        self, intent: ApprovalIntent, pending: str, text: str
    ) -> SkillResult:
        if intent == ApprovalIntent.APPROVE:
            return self._cmd_approve(pending)
        if intent == ApprovalIntent.REJECT:
            return self._cmd_reject(f"{pending} {text}")
        if intent == ApprovalIntent.CLARIFY:
            if self._spec_tier(pending) >= 3:
                return SkillResult(
                    success=True,
                    response=(
                        f"Коротко: `{pending}` — Tier 3, без твоего разрешения "
                        "не запускаю. Ответь обычным текстом: запускать, "
                        "править или что обсудить. Детали покажу по "
                        f"`/spec show {pending}`."
                    ),
                )
            return SkillResult(
                success=True,
                response=(
                    f"Не уверена — одобрить или отклонить spec `{pending}`? "
                    f"Используй `/spec approve {pending}` или "
                    f"`/spec reject {pending} <reason>`. "
                    f"`/spec show {pending}` чтобы посмотреть."
                ),
            )
        return SkillResult(success=False, response="")

    # -------------------------------------------------------------- subcommands

    def _cmd_list(self) -> SkillResult:
        files = list_spec_files(self._tasks_dir)
        active: dict[SpecStatus, list[SpecModel]] = {}
        for path in files:
            try:
                spec = load_spec(path)
            except (ValueError, KeyError) as exc:
                logger.warning("spec_load_failed", path=str(path), error=str(exc))
                continue
            if spec.status not in _ACTIVE_STATUSES:
                continue
            active.setdefault(spec.status, []).append(spec)

        if not active:
            return SkillResult(
                success=True,
                response="Нет активных specs в tasks/. Используй `/spec show <slug>` для done/rejected.",
            )

        order = (
            SpecStatus.PENDING_APPROVAL,
            SpecStatus.APPROVED,
            SpecStatus.IN_PROGRESS,
        )
        sections: list[str] = []
        for status in order:
            specs = active.get(status, [])
            if not specs:
                continue
            sections.append(f"**{status.value}** ({len(specs)})")
            for spec in specs:
                sections.append(f"  • `{spec.slug}` (tier {spec.tier}) — {spec.title}")
        return SkillResult(success=True, response="\n".join(sections))

    def _cmd_show(self, slug: str) -> SkillResult:
        if not slug:
            return SkillResult(
                success=True,
                response="Укажи slug: `/spec show <slug>`. Список — `/spec list`.",
            )
        path = find_spec_path(self._tasks_dir, slug)
        if path is None:
            return SkillResult(
                success=True,
                response=f"Spec `{slug}` не найден в tasks/.",
            )
        spec = load_spec(path)
        return SkillResult(success=True, response=_format_spec(spec, path))

    def _cmd_approve(self, slug: str) -> SkillResult:
        if not slug:
            return SkillResult(
                success=True,
                response="Укажи slug: `/spec approve <slug>`.",
            )
        path = find_spec_path(self._tasks_dir, slug)
        if path is None:
            return SkillResult(
                success=True, response=f"Spec `{slug}` не найден в tasks/."
            )
        raw = load_spec_raw(path)
        # Validate before mutating — reject malformed specs early.
        SpecModel.model_validate(raw)

        current = raw.get("status")
        if current == SpecStatus.APPROVED.value:
            return SkillResult(
                success=True,
                response=f"Spec `{slug}` уже approved. `/spec show {slug}` для деталей.",
            )
        if current in {
            SpecStatus.DONE.value,
            SpecStatus.REJECTED.value,
        }:
            return SkillResult(
                success=True,
                response=f"Spec `{slug}` в терминальном статусе `{current}` — изменить нельзя.",
            )

        was_failed = current == SpecStatus.FAILED.value
        raw["status"] = SpecStatus.APPROVED.value
        raw["approved_at"] = _now_iso()
        raw["approved_by"] = "nikita"
        if was_failed:
            # The failed attempt remains in ``failed_attempts``. Runtime fields
            # from the abandoned attempt must not look like the next run's
            # active branch/commit.
            raw["branch"] = None
            raw["commit_sha"] = None
        # Validate the merged dict to ensure invariants still hold.
        SpecModel.model_validate(raw)
        save_spec_raw(path, raw)
        if self._repo_root is not None:
            commit_yaml_mutation(
                spec_path=path,
                repo_root=self._repo_root,
                subject=(
                    f"chore(self_coding): retry {slug}"
                    if was_failed
                    else f"chore(self_coding): approve {slug}"
                ),
            )
        logger.info(
            "spec_approved",
            slug=slug,
            tier=raw.get("tier"),
            path=str(path),
            retry=was_failed,
        )
        if was_failed:
            return SkillResult(
                success=True,
                response=(
                    f"Spec `{slug}` возвращён в approved для повторной попытки. "
                    f"Tier {raw.get('tier')}."
                ),
            )
        return SkillResult(
            success=True,
            response=(
                f"Spec `{slug}` approved. Tier {raw.get('tier')}. "
                f"После Phase 13 — Editor подхватит и запустит `implement_spec`."
            ),
        )

    def _cmd_approve_bundle(self, args: str) -> SkillResult:
        parsed = self._parse_approve_bundle_args(args)
        if isinstance(parsed, str):
            return SkillResult(success=True, response=parsed)

        if len(set(parsed.slugs)) != len(parsed.slugs):
            return SkillResult(
                success=True,
                response="В approve-bundle каждый slug должен быть указан один раз.",
            )

        pending_specs: list[tuple[str, Path, dict[str, Any]]] = []
        for slug in parsed.slugs:
            path = find_spec_path(self._tasks_dir, slug)
            if path is None:
                return SkillResult(
                    success=True, response=f"Spec `{slug}` не найден в tasks/."
                )
            raw = load_spec_raw(path)
            # Validate before deciding whether this yaml is safe to mutate.
            SpecModel.model_validate(raw)
            current = raw.get("status")
            if current != SpecStatus.PENDING_APPROVAL.value:
                return SkillResult(
                    success=True,
                    response=(
                        f"Spec `{slug}` должен быть pending_approval для "
                        f"approve-bundle, сейчас `{current}`."
                    ),
                )
            pending_specs.append((slug, path, raw))

        approved_at = _now_iso()
        audit_line = _format_approve_bundle_audit(
            slugs=parsed.slugs,
            scope=parsed.scope,
            reason=parsed.reason,
        )
        for slug, path, raw in pending_specs:
            raw["status"] = SpecStatus.APPROVED.value
            raw["approved_at"] = approved_at
            raw["approved_by"] = "nikita"
            raw["chat_context"] = [*raw.get("chat_context", []), audit_line]
            # Validate the merged dict before each yaml write.
            SpecModel.model_validate(raw)
            save_spec_raw(path, raw)
            logger.info(
                "spec_bundle_approved",
                slug=slug,
                tier=raw.get("tier"),
                path=str(path),
                bundle_scope=parsed.scope,
            )

        return SkillResult(
            success=True,
            response=(
                "Approve-bundle записан: "
                f"{', '.join(f'`{slug}`' for slug in parsed.slugs)} approved. "
                f"Scope: {parsed.scope}. Reason: {parsed.reason}. "
                "Это только approval metadata; отдельное разрешение всё ещё "
                "требуется для "
                f"{', '.join(_BUNDLE_SEPARATE_APPROVAL_ACTIONS)}."
            ),
        )

    def _cmd_reject(self, args: str) -> SkillResult:
        # args may be "<slug>" or "<slug> <reason ...>".
        parts = args.split(maxsplit=1)
        if not parts:
            return SkillResult(
                success=True,
                response="Укажи slug: `/spec reject <slug> [reason]`.",
            )
        slug = parts[0]
        reason = parts[1].strip() if len(parts) > 1 else None

        path = find_spec_path(self._tasks_dir, slug)
        if path is None:
            return SkillResult(
                success=True, response=f"Spec `{slug}` не найден в tasks/."
            )
        raw = load_spec_raw(path)
        SpecModel.model_validate(raw)

        current = raw.get("status")
        if current in {
            SpecStatus.DONE.value,
            SpecStatus.FAILED.value,
            SpecStatus.REJECTED.value,
        }:
            return SkillResult(
                success=True,
                response=f"Spec `{slug}` в терминальном статусе `{current}` — изменить нельзя.",
            )

        raw["status"] = SpecStatus.REJECTED.value
        raw["rejected_reason"] = reason
        SpecModel.model_validate(raw)
        save_spec_raw(path, raw)
        if self._repo_root is not None:
            commit_yaml_mutation(
                spec_path=path,
                repo_root=self._repo_root,
                subject=f"chore(self_coding): reject {slug}",
            )
        logger.info("spec_rejected", slug=slug, reason=reason or "")
        return SkillResult(
            success=True,
            response=f"Spec `{slug}` rejected. Reason: {reason or '—'}",
        )

    # ------------------------------------------------------------------ helpers

    def _has_active_pending_spec(self) -> bool:
        return self._latest_pending_slug() is not None

    def _latest_pending_slug(self) -> str | None:
        spec = self._latest_pending_spec()
        return spec.slug if spec is not None else None

    def _latest_pending_spec(self) -> SpecModel | None:
        for path in reversed(list_spec_files(self._tasks_dir)):
            try:
                spec = load_spec(path)
            except (ValueError, KeyError):
                continue
            if spec.status == SpecStatus.PENDING_APPROVAL:
                return spec
        return None

    def _spec_tier(self, slug: str) -> int:
        path = find_spec_path(self._tasks_dir, slug)
        if path is None:
            return 0
        try:
            return load_spec(path).tier
        except (ValueError, KeyError):
            return 0

    @staticmethod
    def _parse_arg(text: str, prefix: str) -> str:
        return (
            text[len(prefix) :].strip().split(maxsplit=1)[0]
            if text[len(prefix) :].strip()
            else ""
        )

    @staticmethod
    def _parse_approve_bundle_args(args: str) -> _ApproveBundleArgs | str:
        if not args:
            return (
                "Укажи минимум два slug и scope: "
                '`/spec approve-bundle <slug1> <slug2> --scope "..." '
                '--reason "..."`.'
            )
        try:
            tokens = shlex.split(args)
        except ValueError as exc:
            return f"Не смогла разобрать approve-bundle: {exc}."

        slugs: list[str] = []
        values = {"scope": "", "reason": ""}
        i = 0
        while i < len(tokens):
            item = tokens[i]
            option = _parse_approve_bundle_option(tokens, i, item)
            if isinstance(option, str):
                return option
            if option is not None:
                values[option.key] = option.value
                i = option.next_index
                continue
            if item.startswith("--"):
                return f"Неизвестный параметр approve-bundle: `{item}`."
            slugs.append(item)
            i += 1

        return _build_approve_bundle_args(
            slugs=slugs,
            scope=values["scope"],
            reason=values["reason"],
        )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _format_spec(spec: SpecModel, path: Path) -> str:
    pillars = (
        f"`{path.name}`\n\n"
        f"**{spec.title}**\n\n"
        f"slug: `{spec.slug}` · tier: {spec.tier} · status: `{spec.status.value}`\n"
        f"created_by: {spec.created_by} · created_at: {spec.created_at.isoformat()}\n"
    )
    if spec.approved_by:
        pillars += f"approved_by: {spec.approved_by} · approved_at: {(spec.approved_at.isoformat() if spec.approved_at else '?')}\n"
    if spec.autonomous_approval_reason:
        pillars += f"autonomous_approval_reason: {spec.autonomous_approval_reason}\n"
    if spec.rejected_reason:
        pillars += f"rejected_reason: {spec.rejected_reason}\n"

    body = (
        f"\n**Goal**\n{spec.goal}\n\n"
        f"**Failing test**\n  {spec.failing_test.file}::{spec.failing_test.name}\n  {spec.failing_test.spec}\n\n"
        f"**Whitelist** ({len(spec.whitelist_paths)})\n"
        + "\n".join(f"  • `{p}`" for p in spec.whitelist_paths)
        + "\n\n"
        "**Preserve behavior**\n"
        + (
            "\n".join(f"  • {p}" for p in spec.preserve_behavior)
            or "  • (legacy spec did not declare this)"
        )
        + "\n\n"
        "**Allowed simplifications**\n"
        + ("\n".join(f"  • {s}" for s in spec.allowed_simplifications) or "  • (none)")
        + "\n\n"
        "**Chat context**\n"
        + ("\n".join(f"  • {line}" for line in spec.chat_context) or "  • (none)")
        + "\n\n"
        "**Previous attempts**\n"
        + (
            "\n".join(
                f"  • `{attempt.archive_slug}` · {attempt.status} · tier "
                f"{attempt.tier}: {attempt.insight}"
                for attempt in spec.previous_attempts
            )
            or "  • (none)"
        )
        + "\n\n"
        "**Blast radius**\n" + "\n".join(f"  • {b}" for b in spec.blast_radius) + "\n\n"
        "**Rollback**\n" + "\n".join(f"  • {r}" for r in spec.rollback_path)
    )
    if spec.research_findings:
        body += "\n\n**Research findings**\n" + "\n".join(
            f"  • {f.source}: {f.relevance}" for f in spec.research_findings
        )
    return pillars + body


def _parse_approve_bundle_option(
    items: list[str],
    index: int,
    item: str,
) -> _ApproveBundleOption | str | None:
    if item == _BUNDLE_SCOPE_FLAG:
        return _bundle_option_from_next(
            items=items,
            index=index,
            key="scope",
            missing_message="После `--scope` нужен текст scope.",
        )
    if item.startswith(f"{_BUNDLE_SCOPE_FLAG}="):
        return _bundle_option_from_inline(item, "scope", _BUNDLE_SCOPE_FLAG, index)
    if item == _BUNDLE_REASON_FLAG:
        return _bundle_option_from_next(
            items=items,
            index=index,
            key="reason",
            missing_message="После `--reason` нужен текст reason.",
        )
    if item.startswith(f"{_BUNDLE_REASON_FLAG}="):
        return _bundle_option_from_inline(item, "reason", _BUNDLE_REASON_FLAG, index)
    return None


def _bundle_option_from_next(
    *,
    items: list[str],
    index: int,
    key: Literal["scope", "reason"],
    missing_message: str,
) -> _ApproveBundleOption | str:
    if index + 1 >= len(items):
        return missing_message
    value = items[index + 1].strip()
    if not value or value.startswith("--"):
        return missing_message
    return _ApproveBundleOption(
        key=key,
        value=value,
        next_index=index + 2,
    )


def _bundle_option_from_inline(
    item: str,
    key: Literal["scope", "reason"],
    flag: str,
    index: int,
) -> _ApproveBundleOption:
    return _ApproveBundleOption(
        key=key,
        value=item.removeprefix(f"{flag}=").strip(),
        next_index=index + 1,
    )


def _build_approve_bundle_args(
    *,
    slugs: list[str],
    scope: str,
    reason: str,
) -> _ApproveBundleArgs | str:
    if len(slugs) < 2:
        return "Approve-bundle требует минимум два spec slug."
    if not scope:
        return "Укажи непустой `--scope` для audit trail."
    if not reason:
        return "Укажи непустой `--reason` для audit trail."
    return _ApproveBundleArgs(slugs=slugs, scope=scope, reason=reason)


def _format_approve_bundle_audit(
    *,
    slugs: list[str],
    scope: str,
    reason: str,
) -> str:
    return (
        "approve-bundle: approved_by=nikita; approved=approval metadata only; "
        f"specs={', '.join(slugs)}; scope={scope}; reason={reason}; "
        "separate_approval_required="
        f"{', '.join(_BUNDLE_SEPARATE_APPROVAL_ACTIONS)}."
    )


_HELP_TEXT = (
    "**Spec commands**\n"
    "  `/spec list` — активные specs (pending / approved / in_progress)\n"
    "  `/spec show <slug>` — полный spec\n"
    "  `/spec approve <slug>` — approve spec для запуска (после Phase 13)\n"
    '  `/spec approve-bundle <slug1> <slug2> --scope "..." --reason "..."` — approve packet без запуска\n'
    "  `/spec reject <slug> [reason]` — отклонить spec\n"
    "\n"
    "Свободный текст «одобряю», «давай», «отклоняю» — тоже работает, "
    "если есть pending spec."
)
