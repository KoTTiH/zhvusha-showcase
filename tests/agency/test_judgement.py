from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.agency.judgement import SocialJudgement
from src.agency.models import (
    SocialJudgementAction,
    SocialJudgementInput,
    SocialPermissionGrant,
    SocialPermissionScope,
    SocialTargetType,
)


def _grant() -> SocialPermissionGrant:
    return SocialPermissionGrant(
        target_id="@devchat",
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        expires_at=datetime(2026, 5, 13, tzinfo=UTC) + timedelta(hours=1),
        max_messages_per_window=2,
    )


def test_judgement_keeps_right_to_stay_silent_when_not_addressed() -> None:
    decision = SocialJudgement().evaluate(
        SocialJudgementInput(
            target_id="@devchat",
            addressed_to_zhvusha=False,
            has_value_to_add=False,
            recent_messages_sent=0,
        ),
        grant=_grant(),
        now=datetime(2026, 5, 13, tzinfo=UTC),
    )

    assert decision.action is SocialJudgementAction.READ_ONLY
    assert decision.can_send is False
    assert "молч" in decision.reason.lower() or "silent" in decision.reason.lower()


def test_judgement_replies_when_addressed_and_useful_inside_rate_limit() -> None:
    decision = SocialJudgement().evaluate(
        SocialJudgementInput(
            target_id="@devchat",
            addressed_to_zhvusha=True,
            has_value_to_add=True,
            recent_messages_sent=1,
        ),
        grant=_grant(),
        now=datetime(2026, 5, 13, tzinfo=UTC),
    )

    assert decision.action is SocialJudgementAction.REPLY
    assert decision.can_send is True


def test_judgement_waits_when_rate_limit_is_hit() -> None:
    decision = SocialJudgement().evaluate(
        SocialJudgementInput(
            target_id="@devchat",
            addressed_to_zhvusha=True,
            has_value_to_add=True,
            recent_messages_sent=2,
        ),
        grant=_grant(),
        now=datetime(2026, 5, 13, tzinfo=UTC),
    )

    assert decision.action is SocialJudgementAction.WAIT
    assert decision.can_send is False


def test_judgement_asks_nikita_on_privacy_risk() -> None:
    decision = SocialJudgement().evaluate(
        SocialJudgementInput(
            target_id="@devchat",
            addressed_to_zhvusha=True,
            has_value_to_add=True,
            privacy_risk=True,
        ),
        grant=_grant(),
        now=datetime(2026, 5, 13, tzinfo=UTC),
    )

    assert decision.action is SocialJudgementAction.ASK_NIKITA
    assert decision.can_send is False
