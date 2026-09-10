"""Social judgement policy for permitted social surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agency.models import (
    SocialJudgementAction,
    SocialJudgementDecision,
    SocialJudgementInput,
    SocialPermissionGrant,
    SocialPermissionScope,
)

if TYPE_CHECKING:
    from datetime import datetime


class SocialJudgement:
    """Decide whether Жвуша should speak, wait or stay silent."""

    def evaluate(
        self,
        item: SocialJudgementInput,
        *,
        grant: SocialPermissionGrant | None,
        now: datetime | None = None,
    ) -> SocialJudgementDecision:
        if grant is None or not grant.is_active(now=now):
            return SocialJudgementDecision(
                action=SocialJudgementAction.ASK_NIKITA,
                reason="Нет активного permission grant для social action.",
                can_send=False,
            )
        if item.privacy_risk or item.conflict_or_private:
            return SocialJudgementDecision(
                action=SocialJudgementAction.ASK_NIKITA,
                reason="Есть риск раскрыть приватное или влезть в конфликт.",
                can_send=False,
                grant_id=grant.id,
            )
        if item.recent_messages_sent >= grant.max_messages_per_window:
            return SocialJudgementDecision(
                action=SocialJudgementAction.WAIT,
                reason="Rate limit для этого social grant уже исчерпан.",
                can_send=False,
                grant_id=grant.id,
            )
        if item.repeats_obvious or not item.tone_ok:
            return SocialJudgementDecision(
                action=SocialJudgementAction.DRAFT,
                reason="Лучше подготовить черновик, а не отправлять сразу.",
                can_send=False,
                grant_id=grant.id,
            )
        scopes = set(grant.scopes)
        if SocialPermissionScope.REPLY_IF_ADDRESSED in scopes:
            if not item.addressed_to_zhvusha:
                return SocialJudgementDecision(
                    action=SocialJudgementAction.READ_ONLY,
                    reason="Разумнее молчать: к Жвуше не обращались.",
                    can_send=False,
                    grant_id=grant.id,
                )
            if item.has_value_to_add:
                return SocialJudgementDecision(
                    action=SocialJudgementAction.REPLY,
                    reason="К Жвуше обратились, и есть что добавить.",
                    can_send=True,
                    grant_id=grant.id,
                )
            return SocialJudgementDecision(
                action=SocialJudgementAction.IGNORE,
                reason="К Жвуше обратились, но полезного ответа нет.",
                can_send=False,
                grant_id=grant.id,
            )
        if SocialPermissionScope.FREE_CONVERSATION in scopes or (
            SocialPermissionScope.INITIATE_OCCASIONALLY in scopes
            and item.has_value_to_add
        ):
            return SocialJudgementDecision(
                action=SocialJudgementAction.SEND,
                reason="Grant разрешает инициативное сообщение, и оно уместно.",
                can_send=True,
                grant_id=grant.id,
            )
        return SocialJudgementDecision(
            action=SocialJudgementAction.READ_ONLY,
            reason="Grant не разрешает сообщение, остаёмся в чтении.",
            can_send=False,
            grant_id=grant.id,
        )
