"""ProposalCommandSkill — manages human-approved Tier 3 proposals."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, Literal

import structlog

from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SkillResult,
)
from src.skills.proposal_command.models import ProposalModel, ProposalStatus
from src.skills.proposal_command.store import (
    find_proposal_path,
    list_proposal_files,
    load_proposal,
    load_proposal_raw,
    save_proposal_raw,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger()

_TRIGGER = "/proposal"
_ACTIVE_STATUSES: frozenset[ProposalStatus] = frozenset(
    {
        ProposalStatus.PENDING_APPROVAL,
        ProposalStatus.APPROVED,
        ProposalStatus.DEFERRED,
    }
)
_PROPOSAL_LIST_PREFIXES: tuple[str, ...] = (
    "покажи proposals",
    "покажи proposal",
    "покажи пропозалы",
    "список proposals",
    "список proposal",
    "список пропозалов",
)
_PROPOSAL_SHOW_PREFIXES: tuple[str, ...] = (
    "покажи proposal",
    "открой proposal",
    "покажи пропозал",
    "открой пропозал",
)
_PROPOSAL_APPROVE_PREFIXES: tuple[str, ...] = (
    "одобри proposal",
    "approve proposal",
    "одобри пропозал",
)
_PROPOSAL_DEFER_PREFIXES: tuple[str, ...] = (
    "отложи proposal",
    "defer proposal",
    "отложи пропозал",
)
_PROPOSAL_REJECT_PREFIXES: tuple[str, ...] = (
    "отклони proposal",
    "reject proposal",
    "отклони пропозал",
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
                return f"{_TRIGGER} {subcommand} {slug}"
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
                return f"{_TRIGGER} {subcommand} {args}"
    return None


def _natural_message_is_mutation(message: str) -> bool:
    if message.strip().lower().startswith(_TRIGGER):
        return False
    command = _normalize_message_to_command(message)
    if command is None:
        return False
    lower = command.lower()
    return any(
        lower.startswith(f"{_TRIGGER} {subcommand}")
        for subcommand in ("approve", "defer", "reject")
    )


def _normalize_message_to_command(message: str) -> str | None:
    text = message.strip()
    lower = text.lower()
    if lower == _TRIGGER or lower.startswith(_TRIGGER + " "):
        return text
    normalized = _normalize_chat_route_text(text)
    for prefixes, subcommand in (
        (_PROPOSAL_SHOW_PREFIXES, "show"),
        (_PROPOSAL_APPROVE_PREFIXES, "approve"),
    ):
        command = _command_from_slug_prefixes(
            original=text,
            normalized=normalized,
            prefixes=prefixes,
            subcommand=subcommand,
        )
        if command is not None:
            return command
    for prefixes, subcommand in (
        (_PROPOSAL_DEFER_PREFIXES, "defer"),
        (_PROPOSAL_REJECT_PREFIXES, "reject"),
    ):
        command = _command_from_args_prefixes(
            original=text,
            normalized=normalized,
            prefixes=prefixes,
            subcommand=subcommand,
        )
        if command is not None:
            return command
    if normalized in _PROPOSAL_LIST_PREFIXES:
        return f"{_TRIGGER} list"
    return None


class ProposalCommandSkill(InlineSkill):
    """Inline skill exposing ``/proposal`` lifecycle commands."""

    name: ClassVar[str] = "proposal_command"
    description: ClassVar[str] = "Manage proposals/*.md Tier 3 proposals via /proposal"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"

    triggers: ClassVar[list[str]] = [_TRIGGER]

    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = "auto"

    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_FILESYSTEM,
        SideEffect.SENDS_TELEGRAM_MESSAGE,
    ]

    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(self, *, proposals_dir: Path, admin_user_id: int) -> None:
        self._proposals_dir = proposals_dir
        self._admin_user_id = admin_user_id

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        command = _normalize_message_to_command(message)
        if command is None:
            return 0.0
        return 1.0 if message.strip().lower().startswith(_TRIGGER) else 0.93

    def requires_approval_for_message(
        self,
        message: str,
        context: AgentContext,
    ) -> bool:
        return context.mode == "personal" and _natural_message_is_mutation(message)

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context

        command = _normalize_message_to_command(message) or message.strip()
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary=(
                f"Изменить статус proposal: {command.removeprefix(_TRIGGER).strip()}"
            ),
            estimated_tokens=0,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=1.0,
            files_to_read=[self._proposals_dir],
            files_to_modify=[self._proposals_dir],
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=0,
            metadata={"internal_action": command},
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        del context
        text = (_normalize_message_to_command(message) or message).strip()
        lower = text.lower()

        if lower.startswith("/proposal list"):
            return self._cmd_list()
        if lower.startswith("/proposal show"):
            return self._cmd_show(_parse_arg(text, "/proposal show"))
        if lower.startswith("/proposal approve"):
            return self._cmd_approve(_parse_arg(text, "/proposal approve"))
        if lower.startswith("/proposal defer"):
            return self._cmd_defer(text.removeprefix("/proposal defer").strip())
        if lower.startswith("/proposal reject"):
            return self._cmd_reject(text.removeprefix("/proposal reject").strip())
        if lower == _TRIGGER or lower.startswith(_TRIGGER + " "):
            return SkillResult(success=True, response=_HELP_TEXT)
        return SkillResult(success=False, response="")

    def _cmd_list(self) -> SkillResult:
        proposals: dict[ProposalStatus, list[ProposalModel]] = {}
        for path in list_proposal_files(self._proposals_dir):
            try:
                proposal = load_proposal(path)
            except (ValueError, KeyError) as exc:
                logger.warning("proposal_load_failed", path=str(path), error=str(exc))
                continue
            if proposal.status not in _ACTIVE_STATUSES:
                continue
            proposals.setdefault(proposal.status, []).append(proposal)

        if not proposals:
            return SkillResult(
                success=True,
                response="Нет активных proposals в proposals/.",
            )

        lines: list[str] = []
        for status in (
            ProposalStatus.PENDING_APPROVAL,
            ProposalStatus.APPROVED,
            ProposalStatus.DEFERRED,
        ):
            items = proposals.get(status, [])
            if not items:
                continue
            lines.append(f"**{status.value}** ({len(items)})")
            for item in items:
                lines.append(f"  • `{item.slug}` — {item.title}")
        return SkillResult(success=True, response="\n".join(lines))

    def _cmd_show(self, slug: str) -> SkillResult:
        if not slug:
            return SkillResult(
                success=True,
                response="Укажи slug: `/proposal show <slug>`. Список — `/proposal list`.",
            )
        path = find_proposal_path(self._proposals_dir, slug)
        if path is None:
            return SkillResult(
                success=True,
                response=f"Proposal `{slug}` не найден в proposals/.",
            )
        proposal = load_proposal(path)
        return SkillResult(success=True, response=_format_proposal(proposal, path))

    def _cmd_approve(self, slug: str) -> SkillResult:
        if not slug:
            return SkillResult(
                success=True,
                response="Укажи slug: `/proposal approve <slug>`.",
            )
        return self._mutate(
            slug=slug,
            target=ProposalStatus.APPROVED,
            reason=None,
        )

    def _cmd_defer(self, args: str) -> SkillResult:
        slug, reason = _split_slug_reason(args)
        if not slug:
            return SkillResult(
                success=True,
                response="Укажи slug: `/proposal defer <slug> [reason]`.",
            )
        return self._mutate(slug=slug, target=ProposalStatus.DEFERRED, reason=reason)

    def _cmd_reject(self, args: str) -> SkillResult:
        slug, reason = _split_slug_reason(args)
        if not slug:
            return SkillResult(
                success=True,
                response="Укажи slug: `/proposal reject <slug> [reason]`.",
            )
        return self._mutate(slug=slug, target=ProposalStatus.REJECTED, reason=reason)

    def _mutate(
        self,
        *,
        slug: str,
        target: ProposalStatus,
        reason: str | None,
    ) -> SkillResult:
        path = find_proposal_path(self._proposals_dir, slug)
        if path is None:
            return SkillResult(
                success=True,
                response=f"Proposal `{slug}` не найден в proposals/.",
            )
        raw, body = load_proposal_raw(path)
        proposal = ProposalModel.model_validate(raw)
        if proposal.status == ProposalStatus.DONE:
            return SkillResult(
                success=True,
                response=f"Proposal `{slug}` уже done — менять нельзя.",
            )
        if proposal.status == ProposalStatus.REJECTED:
            return SkillResult(
                success=True,
                response=f"Proposal `{slug}` уже rejected — менять нельзя.",
            )
        if proposal.status == target:
            return SkillResult(
                success=True,
                response=f"Proposal `{slug}` уже `{target.value}`.",
            )

        raw["status"] = target.value
        if target == ProposalStatus.APPROVED:
            raw["approved_at"] = _now_iso()
            raw["approved_by"] = "nikita"
            raw.pop("deferred_reason", None)
            raw.pop("rejected_reason", None)
        elif target == ProposalStatus.DEFERRED:
            raw["deferred_reason"] = reason
        elif target == ProposalStatus.REJECTED:
            raw["rejected_reason"] = reason

        ProposalModel.model_validate(raw)
        save_proposal_raw(path, raw, body)
        logger.info(
            "proposal_status_changed",
            slug=slug,
            status=target.value,
            path=str(path),
        )
        return SkillResult(
            success=True,
            response=_status_response(slug=slug, status=target, reason=reason),
            metadata={"slug": slug, "status": target.value, "path": str(path)},
        )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_arg(text: str, prefix: str) -> str:
    tail = text[len(prefix) :].strip()
    return tail.split(maxsplit=1)[0] if tail else ""


def _split_slug_reason(args: str) -> tuple[str, str | None]:
    parts = args.split(maxsplit=1)
    if not parts:
        return "", None
    reason = parts[1].strip() if len(parts) > 1 else None
    return parts[0], reason or None


def _format_proposal(proposal: ProposalModel, path: Path) -> str:
    files = "\n".join(f"  • `{item}`" for item in proposal.files_likely_touched)
    sources = "\n".join(
        f"  • {source.url} ({source.trust_tier}): {source.claim}"
        for source in proposal.source_provenance
    )
    body = (
        f"`{path.name}`\n\n"
        f"**{proposal.title}**\n\n"
        f"slug: `{proposal.slug}` · tier: {proposal.tier} · "
        f"status: `{proposal.status.value}`\n"
        f"created_by: {proposal.created_by} · "
        f"created_at: {proposal.created_at.isoformat()}\n"
    )
    if proposal.approved_by:
        approved_at = proposal.approved_at.isoformat() if proposal.approved_at else "?"
        body += f"approved_by: {proposal.approved_by} · approved_at: {approved_at}\n"
    if proposal.deferred_reason:
        body += f"deferred_reason: {proposal.deferred_reason}\n"
    if proposal.rejected_reason:
        body += f"rejected_reason: {proposal.rejected_reason}\n"
    return (
        body
        + f"\n**Суть**\n{proposal.summary}\n\n"
        + f"**Изменение**\n{proposal.proposed_change}\n\n"
        + f"**Почему**\n{proposal.rationale}\n\n"
        + f"**Вероятные файлы**\n{files or '  • уточнить после approve'}\n\n"
        + f"**Риск**\n{proposal.risk}\n\n"
        + f"**Источники**\n{sources}"
    )


def _status_response(*, slug: str, status: ProposalStatus, reason: str | None) -> str:
    if status == ProposalStatus.APPROVED:
        return (
            f"Proposal `{slug}` approved. Автокодинг не запускаю: "
            "Tier 3 сначала раскладывается на отдельные approved specs."
        )
    if status == ProposalStatus.DEFERRED:
        return f"Proposal `{slug}` deferred. Reason: {reason or '—'}"
    if status == ProposalStatus.REJECTED:
        return f"Proposal `{slug}` rejected. Reason: {reason or '—'}"
    return f"Proposal `{slug}` -> `{status.value}`."


_HELP_TEXT = (
    "**Proposal commands**\n"
    "  `/proposal list` — активные Tier 3 proposals\n"
    "  `/proposal show <slug>` — полный proposal\n"
    "  `/proposal approve <slug>` — одобрить направление без запуска кода\n"
    "  `/proposal defer <slug> [reason]` — отложить\n"
    "  `/proposal reject <slug> [reason]` — отклонить\n"
)
