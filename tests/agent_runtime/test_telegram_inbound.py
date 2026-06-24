"""Personal Telegram inbound read-only observation contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from src.agency.models import (
    AgencyAuditEvent,
    SocialJudgementAction,
    SocialJudgementDecision,
    SocialPermissionGrant,
    SocialPermissionScope,
    SocialTargetType,
)


def _grant() -> SocialPermissionGrant:
    return SocialPermissionGrant(
        id="grant-devchat",
        target_id="@devchat",
        target_type=SocialTargetType.CHAT,
        scopes=(SocialPermissionScope.REPLY_IF_ADDRESSED,),
        expires_at=datetime(2026, 5, 14, tzinfo=UTC) + timedelta(hours=1),
    )


def test_personal_telegram_inbound_capsule_is_read_only_even_with_grant() -> None:
    from src.agent_runtime.telegram_inbound import (
        PersonalTelegramInboundEvent,
        build_personal_telegram_inbound_capsule,
    )

    event = PersonalTelegramInboundEvent(
        event_id="tg-personal:42",
        chat_id="@devchat",
        sender_id="100",
        sender_name="Тоша",
        text="Жвуша, ты тут?",
        received_at=datetime(2026, 5, 14, tzinfo=UTC),
    )
    judgement = SocialJudgementDecision(
        action=SocialJudgementAction.REPLY,
        can_send=True,
        reason="addressed and useful",
        grant_id="grant-devchat",
    )

    capsule = build_personal_telegram_inbound_capsule(
        event,
        grant=_grant(),
        judgement=judgement,
    )

    assert capsule.summary == "Personal Telegram inbound event observed read-only."
    assert "Жвуша, ты тут?" in capsule.processed_context
    assert "can_auto_reply:false" in capsule.artifacts
    assert "grant_id:grant-devchat" in capsule.artifacts
    assert any("не отвечать автоматически" in item for item in capsule.next_actions)
    assert "send_message" not in "\n".join(capsule.next_actions)


def test_personal_telegram_inbound_without_grant_asks_orchestrator() -> None:
    from src.agent_runtime.telegram_inbound import (
        PersonalTelegramInboundEvent,
        build_personal_telegram_inbound_capsule,
    )

    event = PersonalTelegramInboundEvent(
        event_id="tg-personal:43",
        chat_id="@unknown",
        sender_id="101",
        sender_name="Новый человек",
        text="привет",
        received_at=datetime(2026, 5, 14, tzinfo=UTC),
    )

    capsule = build_personal_telegram_inbound_capsule(event)

    assert "grant_id:missing" in capsule.artifacts
    assert "can_auto_reply:false" in capsule.artifacts
    assert any("grant" in item.lower() for item in capsule.next_actions)


def test_personal_telegram_inbound_carries_social_gate_decision_without_reply() -> None:
    from src.agency.social_gate import SocialSendGateResult
    from src.agent_runtime.telegram_inbound import (
        PersonalTelegramInboundEvent,
        build_personal_telegram_inbound_capsule,
    )

    event = PersonalTelegramInboundEvent(
        event_id="tg-personal:44",
        chat_id="@devchat",
        sender_id="100",
        sender_name="Тоша",
        text="Жвуша, ответишь?",
        received_at=datetime(2026, 5, 14, tzinfo=UTC),
    )
    gate_result = SocialSendGateResult(
        allowed=True,
        reason="allowed_by_grant_and_judgement",
        target_id="@devchat",
        grant_id="grant-devchat",
        audit_event=AgencyAuditEvent(
            event_type="social_send_allowed",
            reason="addressed and useful",
            target_id="@devchat",
            grant_id="grant-devchat",
        ),
    )

    capsule = build_personal_telegram_inbound_capsule(
        event,
        grant=_grant(),
        send_gate_result=gate_result,
    )

    assert "can_auto_reply:false" in capsule.artifacts
    assert "social_gate_allowed:true" in capsule.artifacts
    assert "social_gate_reason:allowed_by_grant_and_judgement" in capsule.artifacts
    assert any("draft/approval" in item for item in capsule.next_actions)


def test_personal_telegram_inbound_chat_render_keeps_message_body_private() -> None:
    from src.agent_runtime.telegram_inbound import (
        PersonalTelegramInboundEvent,
        build_personal_telegram_inbound_capsule,
        render_personal_telegram_inbound_capsule_for_chat,
    )

    event = PersonalTelegramInboundEvent(
        event_id="tg-personal:45",
        chat_id="@devchat",
        sender_id="100",
        sender_name="Тоша",
        text="секретный личный текст: token=12345",
        received_at=datetime(2026, 5, 14, tzinfo=UTC),
    )

    capsule = build_personal_telegram_inbound_capsule(event, grant=_grant())
    rendered = render_personal_telegram_inbound_capsule_for_chat(capsule)

    assert "секретный личный текст" in capsule.processed_context
    assert "token=12345" in capsule.processed_context
    assert "Personal Telegram inbound event observed read-only." in rendered
    assert "can_auto_reply:false" in rendered
    assert "не отвечать автоматически" in rendered
    assert "tg-personal:45" in rendered
    assert "секретный личный текст" not in rendered
    assert "token=12345" not in rendered


def test_personal_telegram_inbound_store_replays_pending_and_dead_letters(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundEvent,
    )

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    first = PersonalTelegramInboundEvent(event_id="tg:1", chat_id="@dev")
    second = PersonalTelegramInboundEvent(event_id="tg:2", chat_id="@dev")

    store.record_pending(first)
    store.record_pending(second)
    store.mark_processed("tg:1")

    assert [event.event_id for event in store.list_pending()] == ["tg:2"]

    store.mark_dead_letter("tg:2", reason="orchestrator_failed")

    assert store.list_pending() == ()
    latest = store.latest_records()
    assert latest["tg:1"].status == "processed"
    assert latest["tg:2"].status == "dead_letter"
    assert latest["tg:2"].reason == "orchestrator_failed"


def test_personal_telegram_inbound_ingestor_is_disabled_by_default(tmp_path) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundEvent,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    ingestor = PersonalTelegramInboundReadOnlyIngestor(store=store)

    result = ingestor.ingest(
        PersonalTelegramInboundEvent(
            event_id="tg:disabled",
            chat_id="@dev",
            text="не записывать пока flag off",
        )
    )

    assert result.accepted is False
    assert result.reason == "inbound_listener_disabled"
    assert result.capsule is None
    assert store.list_pending() == ()


def test_personal_telegram_inbound_ingestor_records_readonly_pending_event(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundEvent,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    ingestor = PersonalTelegramInboundReadOnlyIngestor(store=store, enabled=True)

    result = ingestor.ingest(
        PersonalTelegramInboundEvent(
            event_id="tg:enabled",
            chat_id="@dev",
            text="Жвуша, пинг",
        )
    )

    assert result.accepted is True
    assert result.capsule is not None
    assert "can_auto_reply:false" in result.capsule.artifacts
    assert [event.event_id for event in store.list_pending()] == ["tg:enabled"]


def test_personal_telegram_inbound_ingestor_can_mark_live_reply_contract(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundEvent,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    ingestor = PersonalTelegramInboundReadOnlyIngestor(
        store=store,
        enabled=True,
        can_auto_reply=True,
    )

    result = ingestor.ingest(
        PersonalTelegramInboundEvent(
            event_id="tg:reply",
            chat_id="@dev",
            chat_type="private",
            text="ответишь?",
        )
    )

    assert result.accepted is True
    assert result.capsule is not None
    assert result.capsule.summary == (
        "Personal Telegram inbound event captured for live response."
    )
    assert "can_auto_reply:true" in result.capsule.artifacts


def test_personal_telegram_inbound_listener_converts_incoming_update(tmp_path) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundListener,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
        ),
        account_label="personal",
    )
    update = SimpleNamespace(
        chat_id=-100,
        sender_id=42,
        sender=SimpleNamespace(first_name="Никита", username="kot"),
        message=SimpleNamespace(
            id=777,
            text="Жвуша, ты тут?",
            date=datetime(2026, 5, 14, 11, tzinfo=UTC),
            out=False,
        ),
    )

    result = listener.handle_update(update)

    assert result.accepted is True
    assert result.capsule is not None
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].event_id == "tg-personal:-100:777"
    assert pending[0].chat_type == "group"
    assert pending[0].addressed is True
    assert pending[0].sender_name == "Никита @kot"
    assert pending[0].text == "Жвуша, ты тут?"


def test_personal_telegram_inbound_listener_ignores_outgoing_update(tmp_path) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundListener,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
        ),
        account_label="personal",
    )
    update = SimpleNamespace(
        chat_id=-100,
        sender_id=42,
        message=SimpleNamespace(id=778, text="мой исходящий текст", out=True),
    )

    result = listener.handle_update(update)

    assert result.accepted is False
    assert result.reason == "outgoing_event_ignored"
    assert store.list_pending() == ()


@pytest.mark.asyncio
async def test_personal_telegram_inbound_listener_replies_and_marks_processed(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundListener,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    class FakeSender:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str | None]] = []

        async def send_message(
            self,
            chat_id: str,
            text: str,
            *,
            parse_mode: str | None = None,
        ) -> None:
            self.messages.append((chat_id, text, parse_mode))

    async def responder(event, capsule, sender) -> str:
        assert event.event_id == "tg-personal:123:779"
        assert "can_auto_reply:true" in capsule.artifacts
        assert sender is fake_sender
        return "ответ через Жвушу"

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    fake_sender = FakeSender()
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
            can_auto_reply=True,
        ),
        account_label="personal",
        responder=responder,
    )
    update = SimpleNamespace(
        chat_id=123,
        sender_id=42,
        message=SimpleNamespace(
            id=779,
            text="Жвуша, ответь",
            date=datetime(2026, 5, 14, 11, tzinfo=UTC),
            out=False,
        ),
    )

    result = await listener.handle_update_and_reply(update, fake_sender)

    assert result.accepted is True
    assert result.reason == "processed_by_responder"
    assert fake_sender.messages == [("123", "ответ через Жвушу", None)]
    assert store.list_pending() == ()
    assert store.latest_records()["tg-personal:123:779"].status == "processed"


@pytest.mark.asyncio
async def test_personal_telegram_inbound_listener_skips_channel_auto_reply(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundListener,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    class FakeSender:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str | None]] = []

        async def send_message(
            self,
            chat_id: str,
            text: str,
            *,
            parse_mode: str | None = None,
        ) -> None:
            self.messages.append((chat_id, text, parse_mode))

    responder_calls = 0

    async def responder(event, capsule, sender) -> str:
        nonlocal responder_calls
        del event, capsule, sender
        responder_calls += 1
        return "не должен отправляться"

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    fake_sender = FakeSender()
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
            can_auto_reply=True,
        ),
        account_label="personal",
        responder=responder,
    )
    update = SimpleNamespace(
        chat_id=-1001057593719,
        sender_id=-1001057593719,
        is_channel=True,
        is_group=False,
        sender=SimpleNamespace(username="Futuris"),
        message=SimpleNamespace(
            id=4338,
            text="OpenAI объявили, что Codex теперь доступен в мобильном приложении",
            date=datetime(2026, 5, 14, 20, 13, 17, tzinfo=UTC),
            out=False,
        ),
    )

    result = await listener.handle_update_and_reply(update, fake_sender)

    assert result.accepted is True
    assert result.reason == "auto_reply_not_allowed"
    assert result.event is not None
    assert result.event.chat_type == "channel"
    assert result.event.addressed is False
    assert result.capsule is not None
    assert "chat_type:channel" in result.capsule.artifacts
    assert "addressed:false" in result.capsule.artifacts
    assert "can_auto_reply:false" in result.capsule.artifacts
    assert responder_calls == 0
    assert fake_sender.messages == []
    latest = store.latest_records()["tg-personal:-1001057593719:4338"]
    assert latest.status == "processed"
    assert latest.reason == "auto_reply_not_allowed"


@pytest.mark.asyncio
async def test_personal_telegram_inbound_listener_skips_unaddressed_group(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundListener,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    class FakeSender:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str | None]] = []

        async def send_message(
            self,
            chat_id: str,
            text: str,
            *,
            parse_mode: str | None = None,
        ) -> None:
            self.messages.append((chat_id, text, parse_mode))

    responder_calls = 0

    async def responder(event, capsule, sender) -> str:
        nonlocal responder_calls
        del event, capsule, sender
        responder_calls += 1
        return "не должен отправляться"

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    fake_sender = FakeSender()
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
            can_auto_reply=True,
        ),
        account_label="personal",
        responder=responder,
    )
    update = SimpleNamespace(
        chat_id=-100,
        sender_id=42,
        is_group=True,
        message=SimpleNamespace(
            id=781,
            text="Красота! Используем!",
            date=datetime(2026, 5, 14, 20, 16, 11, tzinfo=UTC),
            out=False,
        ),
    )

    result = await listener.handle_update_and_reply(update, fake_sender)

    assert result.accepted is True
    assert result.reason == "auto_reply_not_allowed"
    assert result.event is not None
    assert result.event.chat_type == "group"
    assert result.event.addressed is False
    assert responder_calls == 0
    assert fake_sender.messages == []
    latest = store.latest_records()["tg-personal:-100:781"]
    assert latest.status == "processed"
    assert latest.reason == "auto_reply_not_allowed"


@pytest.mark.asyncio
async def test_personal_telegram_inbound_listener_replies_to_addressed_group(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundListener,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    class FakeSender:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str | None]] = []

        async def send_message(
            self,
            chat_id: str,
            text: str,
            *,
            parse_mode: str | None = None,
        ) -> None:
            self.messages.append((chat_id, text, parse_mode))

    async def responder(event, capsule, sender) -> str:
        assert event.event_id == "tg-personal:-100:782"
        assert event.chat_type == "group"
        assert event.addressed is True
        assert "can_auto_reply:true" in capsule.artifacts
        assert sender is fake_sender
        return "адресный ответ"

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    fake_sender = FakeSender()
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
            can_auto_reply=True,
        ),
        account_label="personal",
        responder=responder,
    )
    update = SimpleNamespace(
        chat_id=-100,
        sender_id=42,
        is_group=True,
        message=SimpleNamespace(
            id=782,
            text="Жвуша, что думаешь?",
            date=datetime(2026, 5, 14, 20, 17, tzinfo=UTC),
            out=False,
        ),
    )

    result = await listener.handle_update_and_reply(update, fake_sender)

    assert result.accepted is True
    assert result.reason == "processed_by_responder"
    assert fake_sender.messages == [("-100", "адресный ответ", None)]
    assert store.latest_records()["tg-personal:-100:782"].status == "processed"


@pytest.mark.asyncio
async def test_personal_telegram_inbound_listener_dead_letters_responder_failure(
    tmp_path,
) -> None:
    from src.agent_runtime.telegram_inbound import (
        FilePersonalTelegramInboundEventStore,
        PersonalTelegramInboundListener,
        PersonalTelegramInboundReadOnlyIngestor,
    )

    class FakeSender:
        async def send_message(
            self,
            chat_id: str,
            text: str,
            *,
            parse_mode: str | None = None,
        ) -> None:
            del chat_id, text, parse_mode

    async def responder(event, capsule, sender) -> str:
        del event, capsule, sender
        raise RuntimeError("boom")

    store = FilePersonalTelegramInboundEventStore(tmp_path / "inbound.jsonl")
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
            can_auto_reply=True,
        ),
        account_label="personal",
        responder=responder,
    )
    update = SimpleNamespace(
        chat_id=123,
        sender_id=42,
        message=SimpleNamespace(id=780, text="сломайся", out=False),
    )

    result = await listener.handle_update_and_reply(update, FakeSender())

    assert result.accepted is False
    assert result.reason == "responder_failed"
    latest = store.latest_records()["tg-personal:123:780"]
    assert latest.status == "dead_letter"
    assert latest.reason == "responder_failed"
