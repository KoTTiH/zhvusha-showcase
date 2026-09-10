"""Inline skill that creates and inspects channel post drafts."""

from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SkillResult,
)
from src.skills.post_drafts.models import PostDraft, PostTopic, build_post_draft
from src.skills.post_drafts.store import (
    find_draft_path,
    list_draft_files,
    load_post_draft,
    save_draft_raw,
    write_post_draft,
)
from src.skills.post_drafts.visual_assets import approve_visual_asset

if TYPE_CHECKING:
    from pathlib import Path

_TRIGGER = "/post_drafts"
_LIST_PREFIXES: tuple[str, ...] = (
    "покажи черновики постов",
    "покажи черновики",
    "список черновиков постов",
    "список черновиков",
)
_SHOW_PREFIXES: tuple[str, ...] = (
    "покажи черновик поста",
    "покажи черновик",
    "открой черновик поста",
    "открой черновик",
)
_GENERATE_PREFIXES: tuple[str, ...] = (
    "создай черновики постов",
    "собери черновики постов",
    "подготовь черновики постов",
)


def _normalize_chat_route_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("ё", "е").split())


def _strip_natural_tail(original: str, prefix: str) -> str:
    pattern = r"^\s*" + r"\s+".join(re.escape(part) for part in prefix.split())
    pattern += r"\s*[:\-—]?\s*"
    return re.sub(pattern, "", original, count=1, flags=re.I).strip(" \t\n\r:-—")


def _normalize_message_to_command(message: str) -> str | None:
    text = message.strip()
    lower = text.lower()
    if lower == _TRIGGER or lower.startswith(_TRIGGER + " "):
        return text
    normalized = _normalize_chat_route_text(text)
    for prefix in _LIST_PREFIXES:
        if normalized == prefix:
            return f"{_TRIGGER} list"
    for prefix in _SHOW_PREFIXES:
        if normalized.startswith(prefix + " ") or normalized.startswith(prefix + ":"):
            slug = _strip_natural_tail(text, prefix).split(maxsplit=1)[0]
            if slug:
                return f"{_TRIGGER} show {slug}"
    for prefix in _GENERATE_PREFIXES:
        if (
            normalized == prefix
            or normalized.startswith(prefix + " ")
            or normalized.startswith(prefix + ":")
        ):
            tail = _strip_natural_tail(text, prefix)
            return f"{_TRIGGER} generate {tail}".strip()
    return None


class PostTopicProvider(Protocol):
    async def list_post_topics(
        self, *, limit: int = 10, min_money_alignment: float = 0.5
    ) -> list[PostTopic]: ...


class VisualAssetPreparer(Protocol):
    async def __call__(
        self,
        draft: PostDraft,
        context: AgentContext,
    ) -> dict[str, Any] | None: ...


class PostDraftsSkill(InlineSkill):
    """Generate channel drafts from topic backlog without publishing them."""

    name: ClassVar[str] = "post_drafts"
    description: ClassVar[str] = "Creates channel post drafts from topic backlog"
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [_TRIGGER]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "medium"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_WORKSPACE,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        workspace_root: Path,
        topic_provider: PostTopicProvider,
        visual_preparer: VisualAssetPreparer | None = None,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._workspace_root = workspace_root
        self._topics = topic_provider
        self._visual_preparer = visual_preparer

    def set_visual_preparer(self, preparer: VisualAssetPreparer | None) -> None:
        self._visual_preparer = preparer

    async def can_handle(self, message: str, context: AgentContext) -> float:
        if context.user_id != self._admin_user_id or context.mode != "personal":
            return 0.0
        command = _normalize_message_to_command(message)
        if command is None:
            return 0.0
        return 1.0 if command.strip().lower().startswith(_TRIGGER) else 0.93

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        del context
        command = _normalize_message_to_command(message) or message.strip()
        return ExecutionPlan(
            skill_name=self.name,
            skill_type="inline",
            human_summary=f"Черновики постов: {command.removeprefix(_TRIGGER).strip() or 'generate'}",
            estimated_tokens=0,
            estimated_cost_usd=Decimal("0"),
            estimated_duration_seconds=3.0,
            files_to_read=[self._workspace_root],
            files_to_modify=[self._workspace_root],
            side_effects_invoked=list(self.side_effects),
            llm_calls_planned=0,
            metadata={"internal_action": command},
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        text = (_normalize_message_to_command(message) or message).strip()
        lower = text.lower()
        if lower == _TRIGGER or lower.startswith(_TRIGGER + " generate"):
            return await self._cmd_generate(text, context)
        if lower.startswith(_TRIGGER + " list"):
            return self._cmd_list()
        if lower.startswith(_TRIGGER + " show"):
            return self._cmd_show(_parse_arg(text, _TRIGGER + " show"))
        if lower.startswith(_TRIGGER + " approve_visual"):
            return self._cmd_approve_visual(text)
        return SkillResult(success=True, response=_HELP_TEXT)

    async def _cmd_generate(self, text: str, context: AgentContext) -> SkillResult:
        limit = _parse_limit(text, default=3)
        topics = await self._topics.list_post_topics(limit=limit)
        if not topics:
            return SkillResult(
                success=True,
                response="Нет backlog-тем с сильным money/channel alignment.",
            )
        drafts: list[PostDraft] = []
        for topic in topics:
            draft = build_post_draft(topic)
            prepared_visual = await self._prepare_visual(draft, context)
            if prepared_visual is not None:
                draft = replace(draft, visual=prepared_visual)
            drafts.append(draft)
        paths = [write_post_draft(self._workspace_root, draft) for draft in drafts]
        lines = [
            "Черновики постов готовы:",
            *[
                f"  • `{path.name}` — `/post_drafts show {path.stem.split('-', 3)[-1]}`"
                for path in paths
            ],
            "",
            "Visual/style preview уже записан во frontmatter. "
            "Если visual required, публикация дождётся approved asset.",
            "",
            "Публикация только вручную: `/post_draft publish <slug>`.",
        ]
        return SkillResult(
            success=True,
            response="\n".join(lines),
            metadata={"count": len(paths), "paths": [str(path) for path in paths]},
        )

    async def _prepare_visual(
        self,
        draft: PostDraft,
        context: AgentContext,
    ) -> dict[str, Any] | None:
        if self._visual_preparer is None or not draft.visual:
            return None
        if draft.visual.get("intent") in {"none", "denied"}:
            return None
        return await self._visual_preparer(draft, context)

    def _cmd_list(self) -> SkillResult:
        files = list_draft_files(self._workspace_root)
        if not files:
            return SkillResult(success=True, response="Черновиков постов нет.")
        lines = ["**Черновики постов**"]
        for path in files[-20:]:
            try:
                raw, _body = load_post_draft(path)
            except ValueError:
                continue
            lines.append(
                f"  • `{raw.get('slug', path.stem)}` · "
                f"{raw.get('status', 'draft')} · "
                f"visual={_visual_status(raw)} — {raw.get('title', path.name)}"
            )
        return SkillResult(success=True, response="\n".join(lines))

    def _cmd_show(self, slug: str) -> SkillResult:
        if not slug:
            return SkillResult(
                success=True,
                response="Укажи slug: `/post_drafts show <slug>`.",
            )
        path = find_draft_path(self._workspace_root, slug)
        if path is None:
            return SkillResult(
                success=True,
                response=f"Черновик `{slug}` не найден.",
            )
        raw, body = load_post_draft(path)
        raw_visual = raw.get("visual")
        raw_style = raw.get("style")
        visual = raw_visual if isinstance(raw_visual, dict) else {}
        style = raw_style if isinstance(raw_style, dict) else {}
        return SkillResult(
            success=True,
            response=(
                f"`{path.name}`\n"
                f"status: `{raw.get('status', 'draft')}` · "
                f"source: `{raw.get('source_cluster', 'unknown')}` · "
                f"visual: `{visual.get('intent', 'legacy-none')}/"
                f"{visual.get('status', 'none')}` · "
                f"style: `{style.get('status', 'legacy')}`\n\n"
                f"{body.strip()}"
            ),
            metadata={"path": str(path), "slug": str(raw.get("slug", slug))},
        )

    def _cmd_approve_visual(self, text: str) -> SkillResult:
        parts = text.split(maxsplit=3)
        if len(parts) < 4:
            return SkillResult(
                success=False,
                response=(
                    "Используй: `/post_drafts approve_visual <slug> <asset_path>`."
                ),
            )
        slug = parts[2].strip()
        asset_path = parts[3].strip()
        path = find_draft_path(self._workspace_root, slug)
        if path is None:
            return SkillResult(success=False, response=f"Черновик `{slug}` не найден.")
        raw, body = load_post_draft(path)
        visual = raw.get("visual")
        if not isinstance(visual, dict):
            return SkillResult(
                success=False,
                response=f"У черновика `{slug}` нет visual metadata.",
            )
        try:
            raw["visual"] = approve_visual_asset(
                visual,
                workspace_root=self._workspace_root,
                asset_path=asset_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            return SkillResult(success=False, response=str(exc))
        save_draft_raw(path, raw, body)
        return SkillResult(
            success=True,
            response=f"Visual asset approved для `{slug}`.",
            metadata={"path": str(path), "asset_path": asset_path},
        )


def _parse_arg(text: str, prefix: str) -> str:
    tail = text[len(prefix) :].strip()
    return tail.split(maxsplit=1)[0] if tail else ""


def _parse_limit(text: str, *, default: int) -> int:
    parts = text.split()
    for part in reversed(parts):
        try:
            parsed = int(part)
        except ValueError:
            continue
        return max(1, min(parsed, 10))
    return default


def _visual_status(raw: dict[str, object]) -> str:
    visual = raw.get("visual")
    if not isinstance(visual, dict):
        return "legacy-none"
    return f"{visual.get('intent', 'none')}/{visual.get('status', 'none')}"


_HELP_TEXT = (
    "**Post draft commands**\n"
    "  `/post_drafts generate [limit]` — сделать черновики из backlog\n"
    "  `/post_drafts list` — показать черновики\n"
    "  `/post_drafts show <slug>` — открыть черновик\n"
    "  `/post_drafts approve_visual <slug> <asset_path>` — утвердить visual artifact\n"
    "  `/post_draft publish <slug>` — явно опубликовать через channel_writer\n"
)
