"""Pre-send gate for autonomous social actions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from src.agency.models import (
    SocialJudgementInput,
    SocialPermissionGrant,
    SocialPermissionScope,
    SocialTargetType,
)
from src.agency.store import FileSocialPermissionStore

if TYPE_CHECKING:
    from pathlib import Path


def _grant(*, target_id: str = "@devchat") -> SocialPermissionGrant:
    return SocialPermissionGrant(
        id="grant-devchat",
        target_id=target_id,
        target_type=SocialTargetType.GROUP,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        allowed_topics=("runtime",),
        forbidden_topics=("личная жизнь Никиты",),
        expires_at=datetime(2026, 5, 14, 13, tzinfo=UTC),
        max_messages_per_window=2,
    )


def test_social_send_gate_blocks_without_grant(tmp_path: Path) -> None:
    from src.agency.social_gate import SocialSendGate, SocialSendRequest

    gate = SocialSendGate(
        store=FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    )

    result = gate.evaluate(
        SocialSendRequest(
            target_id="@devchat",
            message="Добавлю контекст по runtime.",
            judgement=SocialJudgementInput(
                target_id="@devchat",
                addressed_to_zhvusha=True,
                has_value_to_add=True,
            ),
        ),
        now=datetime(2026, 5, 14, 12, tzinfo=UTC),
    )

    assert result.allowed is False
    assert result.reason == "missing_active_grant"
    assert result.audit_event.event_type == "social_send_blocked"


def test_social_send_gate_allows_inside_grant_and_judgement(
    tmp_path: Path,
) -> None:
    from src.agency.social_gate import SocialSendGate, SocialSendRequest

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    store.add(_grant())
    gate = SocialSendGate(store=store)

    result = gate.evaluate(
        SocialSendRequest(
            target_id="@devchat",
            message="Добавлю контекст по runtime.",
            topic="runtime",
            judgement=SocialJudgementInput(
                target_id="@devchat",
                addressed_to_zhvusha=True,
                has_value_to_add=True,
                recent_messages_sent=1,
            ),
        ),
        now=datetime(2026, 5, 14, 12, tzinfo=UTC),
    )

    assert result.allowed is True
    assert result.grant_id == "grant-devchat"
    assert result.judgement is not None
    assert result.judgement.can_send is True
    assert result.audit_event.grant_id == "grant-devchat"


def test_social_send_gate_blocks_forbidden_topic_and_rate_limit(
    tmp_path: Path,
) -> None:
    from src.agency.social_gate import SocialSendGate, SocialSendRequest

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    store.add(_grant())
    gate = SocialSendGate(store=store)

    forbidden = gate.evaluate(
        SocialSendRequest(
            target_id="@devchat",
            message="Скажу про личную жизнь Никиты.",
            topic="личная жизнь Никиты",
            judgement=SocialJudgementInput(
                target_id="@devchat",
                addressed_to_zhvusha=True,
                has_value_to_add=True,
            ),
        ),
        now=datetime(2026, 5, 14, 12, tzinfo=UTC),
    )
    rate_limited = gate.evaluate(
        SocialSendRequest(
            target_id="@devchat",
            message="Добавлю контекст по runtime.",
            topic="runtime",
            judgement=SocialJudgementInput(
                target_id="@devchat",
                addressed_to_zhvusha=True,
                has_value_to_add=True,
                recent_messages_sent=2,
            ),
        ),
        now=datetime(2026, 5, 14, 12, tzinfo=UTC),
    )

    assert forbidden.allowed is False
    assert forbidden.reason == "topic_forbidden_by_grant"
    assert rate_limited.allowed is False
    assert rate_limited.reason == "social_judgement_wait"


def test_social_send_gate_uses_recorded_send_count_for_rate_limit(
    tmp_path: Path,
) -> None:
    from src.agency.social_gate import SocialSendGate, SocialSendRequest

    now = datetime(2026, 5, 14, 12, tzinfo=UTC)
    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    grant = _grant()
    store.add(grant)
    store.record_sent(
        grant_id=grant.id,
        target_id="@devchat",
        sent_at=now - timedelta(minutes=20),
    )
    store.record_sent(
        grant_id=grant.id,
        target_id="@devchat",
        sent_at=now - timedelta(minutes=5),
    )
    gate = SocialSendGate(store=store)

    result = gate.evaluate(
        SocialSendRequest(
            target_id="@devchat",
            message="Добавлю контекст по runtime.",
            topic="runtime",
            judgement=SocialJudgementInput(
                target_id="@devchat",
                addressed_to_zhvusha=True,
                has_value_to_add=True,
                recent_messages_sent=0,
            ),
        ),
        now=now,
    )

    assert result.allowed is False
    assert result.reason == "social_judgement_wait"
    assert result.grant_id == grant.id
    assert result.judgement is not None
    assert result.judgement.reason == "Rate limit для этого social grant уже исчерпан."


def test_social_send_gate_blocks_emergency_stop_even_with_grant(
    tmp_path: Path,
) -> None:
    from src.agency.social_gate import SocialSendGate, SocialSendRequest

    store = FileSocialPermissionStore(tmp_path / "agency-permissions.jsonl")
    store.add(_grant())
    store.set_emergency_stop(True, reason="manual stop")
    gate = SocialSendGate(store=store)

    result = gate.evaluate(
        SocialSendRequest(
            target_id="@devchat",
            message="Добавлю контекст по runtime.",
            topic="runtime",
            judgement=SocialJudgementInput(
                target_id="@devchat",
                addressed_to_zhvusha=True,
                has_value_to_add=True,
            ),
        ),
        now=datetime(2026, 5, 14, 12, tzinfo=UTC),
    )

    assert result.allowed is False
    assert result.reason == "emergency_stop"


def test_social_permission_scope_contains_full_discussed_vocabulary() -> None:
    assert SocialPermissionScope.INITIATE_ONCE.value == "initiate_once"
    assert SocialPermissionScope.INITIATE_OCCASIONALLY.value == (
        "initiate_occasionally"
    )
    assert SocialPermissionScope.FREE_CONVERSATION.value == "free_conversation"
    assert SocialPermissionScope.JOIN_AND_OBSERVE.value == "join_and_observe"
    assert SocialPermissionScope.TOPIC_LIMITED_POSTING.value == (
        "topic_limited_posting"
    )
