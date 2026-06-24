"""Personal Telegram inbound event observations and live response bridge."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.agent_runtime.models import ContextCapsule, Finding, FindingStatus

if TYPE_CHECKING:
    from src.agency.models import SocialJudgementDecision, SocialPermissionGrant
    from src.agency.social_gate import SocialSendGateResult

logger = structlog.get_logger()


_CHAT_SAFE_ARTIFACT_PREFIXES = (
    "personal_telegram_event_id:",
    "account:",
    "chat_type:",
    "addressed:",
    "grant_id:",
    "judgement_action:",
    "social_gate_allowed:",
    "social_gate_reason:",
    "can_auto_reply:",
)


PersonalTelegramInboundStatus = Literal["pending", "processed", "dead_letter"]
PersonalTelegramChatType = Literal["private", "group", "channel", "unknown"]


class PersonalTelegramReplySender(Protocol):
    """Minimal sender contract for replying through the personal account."""

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> Any: ...


class PersonalTelegramInboundResponder(Protocol):
    """Жвуша-facing responder for accepted personal Telegram events."""

    async def __call__(
        self,
        event: PersonalTelegramInboundEvent,
        capsule: ContextCapsule,
        sender: PersonalTelegramReplySender,
    ) -> str | None: ...


class PersonalTelegramInboundEvent(BaseModel):
    """Incoming personal-account event before any reply decision."""

    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    sender_id: str = ""
    sender_name: str = ""
    text: str = ""
    account_label: str = "personal"
    chat_type: PersonalTelegramChatType = "unknown"
    addressed: bool = False
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator(
        "event_id",
        "chat_id",
        "sender_id",
        "sender_name",
        "text",
        "account_label",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("chat_type", mode="before")
    @classmethod
    def _normalize_chat_type(cls, value: object) -> PersonalTelegramChatType:
        normalized = str(value or "unknown").strip().lower()
        if normalized == "private":
            return "private"
        if normalized == "group":
            return "group"
        if normalized == "channel":
            return "channel"
        return "unknown"


class PersonalTelegramInboundRecord(BaseModel):
    """Append-only record for inbound replay/dead-letter handling."""

    event: PersonalTelegramInboundEvent
    status: PersonalTelegramInboundStatus = "pending"
    reason: str = ""
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PersonalTelegramInboundIngestResult(BaseModel):
    """Result of read-only inbound event ingestion."""

    accepted: bool
    reason: str
    event: PersonalTelegramInboundEvent | None = None
    capsule: ContextCapsule | None = None


class FilePersonalTelegramInboundEventStore:
    """Append-only JSONL store for personal Telegram inbound observations."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record_pending(self, event: PersonalTelegramInboundEvent) -> None:
        """Append a pending inbound event for later orchestrator replay."""

        self._append(PersonalTelegramInboundRecord(event=event, status="pending"))

    def mark_processed(self, event_id: str, *, reason: str = "") -> None:
        """Mark an inbound event as handed to Жвуша's orchestrator."""

        record = self.latest_records().get(event_id)
        if record is not None:
            self._append(
                record.model_copy(update={"status": "processed", "reason": reason})
            )

    def mark_dead_letter(self, event_id: str, *, reason: str) -> None:
        """Mark an inbound event as failed without dropping audit evidence."""

        record = self.latest_records().get(event_id)
        if record is not None:
            self._append(
                record.model_copy(update={"status": "dead_letter", "reason": reason})
            )

    def list_pending(self) -> tuple[PersonalTelegramInboundEvent, ...]:
        """Return events whose latest status is still pending."""

        return tuple(
            record.event
            for record in self.latest_records().values()
            if record.status == "pending"
        )

    def latest_records(self) -> dict[str, PersonalTelegramInboundRecord]:
        """Return the latest record per event id."""

        latest: dict[str, PersonalTelegramInboundRecord] = {}
        if not self._path.exists():
            return latest
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = PersonalTelegramInboundRecord.model_validate_json(line)
            latest[record.event.event_id] = record
        return latest

    def _append(self, record: PersonalTelegramInboundRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")


class PersonalTelegramInboundReadOnlyIngestor:
    """Record inbound events as observations before any responder side effect."""

    def __init__(
        self,
        *,
        store: FilePersonalTelegramInboundEventStore,
        enabled: bool = False,
        can_auto_reply: bool = False,
    ) -> None:
        self._store = store
        self._enabled = enabled
        self._can_auto_reply = can_auto_reply

    def ingest(
        self,
        event: PersonalTelegramInboundEvent,
    ) -> PersonalTelegramInboundIngestResult:
        """Record one event and return its read-only capsule when enabled."""

        if not self._enabled:
            return PersonalTelegramInboundIngestResult(
                accepted=False,
                reason="inbound_listener_disabled",
            )
        self._store.record_pending(event)
        can_auto_reply = self._can_auto_reply and _event_allows_auto_reply(event)
        return PersonalTelegramInboundIngestResult(
            accepted=True,
            reason="recorded_inbound_observation",
            event=event,
            capsule=build_personal_telegram_inbound_capsule(
                event,
                can_auto_reply=can_auto_reply,
            ),
        )

    def mark_processed(self, event_id: str, *, reason: str = "") -> None:
        self._store.mark_processed(event_id, reason=reason)

    def mark_dead_letter(self, event_id: str, *, reason: str) -> None:
        self._store.mark_dead_letter(event_id, reason=reason)


class PersonalTelegramInboundListener:
    """Convert personal Telegram updates into recorded observations."""

    def __init__(
        self,
        *,
        ingestor: PersonalTelegramInboundReadOnlyIngestor,
        account_label: str = "personal",
        responder: PersonalTelegramInboundResponder | None = None,
    ) -> None:
        self._ingestor = ingestor
        self._account_label = account_label
        self._responder = responder

    def handle_update(self, update: object) -> PersonalTelegramInboundIngestResult:
        """Record one incoming update without sending a reply."""

        event = _event_from_telethon_update(update, account_label=self._account_label)
        if event is None:
            return PersonalTelegramInboundIngestResult(
                accepted=False,
                reason="outgoing_event_ignored",
            )
        return self._ingestor.ingest(event)

    async def handle_update_and_reply(
        self,
        update: object,
        sender: PersonalTelegramReplySender,
    ) -> PersonalTelegramInboundIngestResult:
        """Record one incoming update and route it to Жвуша when configured."""

        result = self.handle_update(update)
        if (
            not result.accepted
            or result.event is None
            or result.capsule is None
            or self._responder is None
        ):
            return result

        if not _capsule_allows_auto_reply(result.capsule):
            self._ingestor.mark_processed(
                result.event.event_id,
                reason="auto_reply_not_allowed",
            )
            logger.info(
                "personal_telegram_inbound_auto_reply_skipped",
                event_id=result.event.event_id,
                chat_type=result.event.chat_type,
                addressed=result.event.addressed,
            )
            return result.model_copy(update={"reason": "auto_reply_not_allowed"})

        try:
            reply = await self._responder(result.event, result.capsule, sender)
            if reply:
                await sender.send_message(result.event.chat_id, reply)
            self._ingestor.mark_processed(result.event.event_id)
            return result.model_copy(update={"reason": "processed_by_responder"})
        except Exception:
            self._ingestor.mark_dead_letter(
                result.event.event_id,
                reason="responder_failed",
            )
            logger.warning(
                "personal_telegram_inbound_responder_failed",
                event_id=result.event.event_id,
                exc_info=True,
            )
            return result.model_copy(
                update={"accepted": False, "reason": "responder_failed"}
            )


class TelethonPersonalTelegramInboundListener:
    """Live Telethon wrapper for personal-account incoming messages."""

    def __init__(
        self,
        *,
        api_id: int,
        api_hash: str,
        session_path: str | Path,
        listener: PersonalTelegramInboundListener,
    ) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session_path = Path(session_path).expanduser()
        self._listener = listener
        self._client: Any | None = None

    async def start(self) -> None:
        """Connect Telethon and subscribe to incoming NewMessage events."""

        from telethon import TelegramClient, events

        self._client = TelegramClient(
            str(self._session_path),
            self._api_id,
            self._api_hash,
        )
        await self._client.connect()
        authorized = await self._client.is_user_authorized()
        if not authorized:
            await self._client.disconnect()
            self._client = None
            raise RuntimeError("personal Telegram session is not authorized")
        self._client.add_event_handler(
            self._handle_event,
            events.NewMessage(incoming=True),
        )
        logger.info("personal_telegram_inbound_listener_started")

    async def run_until_disconnected(self) -> None:
        """Block until Telethon disconnects after start()."""

        if self._client is None:
            raise RuntimeError("listener is not started")
        await self._client.disconnected

    async def stop(self) -> None:
        """Disconnect the live Telethon listener."""

        if self._client is None:
            return
        await self._client.disconnect()
        self._client = None
        logger.info("personal_telegram_inbound_listener_stopped")

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> Any:
        """Send a reply through the connected personal Telegram account."""

        if self._client is None:
            raise RuntimeError("listener is not started")
        return await self._client.send_message(
            _parse_telegram_chat_id(chat_id),
            text,
            parse_mode=parse_mode,
        )

    async def _handle_event(self, update: object) -> None:
        result = await self._listener.handle_update_and_reply(update, self)
        if not result.accepted:
            logger.info("personal_telegram_inbound_event_ignored", reason=result.reason)


def build_personal_telegram_inbound_listener_from_settings(
    settings: Any,
    *,
    responder: PersonalTelegramInboundResponder | None = None,
) -> TelethonPersonalTelegramInboundListener | None:
    """Build the live personal Telegram listener when explicitly enabled."""

    if not bool(getattr(settings, "personal_telegram_inbound_enabled", False)):
        return None
    api_id = int(getattr(settings, "telegram_api_id", 0))
    api_hash = str(getattr(settings, "telegram_api_hash", "")).strip()
    if not api_id or not api_hash:
        raise ValueError("telegram_api_id and telegram_api_hash are required")
    store = FilePersonalTelegramInboundEventStore(
        Path(
            str(
                getattr(
                    settings,
                    "personal_telegram_inbound_store_path",
                    "~/zhvusha-workspace/telegram/inbound_events.jsonl",
                )
            )
        ).expanduser()
    )
    listener = PersonalTelegramInboundListener(
        ingestor=PersonalTelegramInboundReadOnlyIngestor(
            store=store,
            enabled=True,
            can_auto_reply=responder is not None,
        ),
        account_label=str(getattr(settings, "telegram_mcp_account_label", "personal")),
        responder=responder,
    )
    return TelethonPersonalTelegramInboundListener(
        api_id=api_id,
        api_hash=api_hash,
        session_path=str(
            getattr(settings, "telethon_session_path", "~/.zhvusha_telethon.session")
        ),
        listener=listener,
    )


def build_personal_telegram_inbound_capsule(
    event: PersonalTelegramInboundEvent,
    *,
    grant: SocialPermissionGrant | None = None,
    judgement: SocialJudgementDecision | None = None,
    send_gate_result: SocialSendGateResult | None = None,
    can_auto_reply: bool = False,
) -> ContextCapsule:
    """Convert an inbound event to a read-only ContextCapsule for Жвуша."""
    grant_id = grant.id if grant is not None else "missing"
    judgement_action = judgement.action.value if judgement is not None else "not_run"
    gate_allowed = (
        "not_run" if send_gate_result is None else str(send_gate_result.allowed).lower()
    )
    gate_reason = "not_run" if send_gate_result is None else send_gate_result.reason
    processed_context = _render_event_context(event)
    artifacts = (
        f"personal_telegram_event_id:{event.event_id}",
        f"account:{event.account_label}",
        f"chat_type:{event.chat_type}",
        f"addressed:{str(event.addressed).lower()}",
        f"grant_id:{grant_id}",
        f"judgement_action:{judgement_action}",
        f"social_gate_allowed:{gate_allowed}",
        f"social_gate_reason:{gate_reason}",
        f"can_auto_reply:{str(can_auto_reply).lower()}",
    )
    next_actions = _next_actions(
        grant=grant,
        judgement=judgement,
        send_gate_result=send_gate_result,
        can_auto_reply=can_auto_reply,
    )
    summary = (
        "Personal Telegram inbound event captured for live response."
        if can_auto_reply
        else "Personal Telegram inbound event observed read-only."
    )
    finding_claim = (
        "Incoming personal Telegram event was captured and handed to "
        "Жвуша's live response pipeline."
        if can_auto_reply
        else (
            "Incoming personal Telegram event was captured as a "
            "read-only observation; no reply was executed."
        )
    )
    return ContextCapsule(
        summary=summary,
        processed_context=processed_context,
        findings=(
            Finding(
                claim=finding_claim,
                status=FindingStatus.CONFIRMED,
                confidence=0.95,
                evidence=(event.event_id, event.chat_id),
            ),
        ),
        sources=(event.event_id,),
        artifacts=artifacts,
        next_actions=next_actions,
        markdown_report=f"{processed_context}\n\n" + "\n".join(next_actions),
    )


def render_personal_telegram_inbound_capsule_for_chat(
    capsule: ContextCapsule,
) -> str:
    """Render inbound status without exposing the private message body."""
    lines = [capsule.summary.strip()]
    safe_artifacts = tuple(
        artifact
        for artifact in capsule.artifacts
        if artifact.startswith(_CHAT_SAFE_ARTIFACT_PREFIXES)
    )

    if capsule.findings:
        lines.extend(("", "Что зафиксировано:"))
        for finding in capsule.findings:
            confidence = round(finding.confidence * 100)
            lines.append(f"- {finding.claim} [{finding.status.value}, {confidence}%]")
            if finding.evidence:
                lines.append(f"  evidence: {', '.join(finding.evidence)}")

    if safe_artifacts:
        lines.extend(("", "Контракт:"))
        lines.extend(f"- {artifact}" for artifact in safe_artifacts)

    if capsule.sources:
        lines.extend(("", "Источники:"))
        lines.extend(f"- {source}" for source in capsule.sources)

    if capsule.next_actions:
        lines.extend(("", "Дальше:"))
        lines.extend(f"- {action}" for action in capsule.next_actions)

    return "\n".join(line for line in lines if line or lines)


def _render_event_context(event: PersonalTelegramInboundEvent) -> str:
    return "\n".join(
        [
            "# Personal Telegram inbound event",
            f"event_id: {event.event_id}",
            f"account: {event.account_label}",
            f"chat_id: {event.chat_id}",
            f"chat_type: {event.chat_type}",
            f"addressed: {str(event.addressed).lower()}",
            f"sender_id: {event.sender_id or '(unknown)'}",
            f"sender_name: {event.sender_name or '(unknown)'}",
            f"received_at: {event.received_at.astimezone(UTC).isoformat()}",
            "",
            event.text,
        ]
    ).strip()


def _event_from_telethon_update(
    update: object,
    *,
    account_label: str,
) -> PersonalTelegramInboundEvent | None:
    message = getattr(update, "message", update)
    if bool(getattr(update, "out", False)) or bool(getattr(message, "out", False)):
        return None
    chat_id = _string_from_attrs(update, message, names=("chat_id", "peer_id"))
    sender_id = _string_from_attrs(update, message, names=("sender_id", "from_id"))
    message_id = _string_from_attrs(message, update, names=("id", "message_id"))
    text = _message_text(update=update, message=message)
    chat_type = _chat_type_from_update(
        update=update,
        message=message,
        chat_id=chat_id,
        sender_id=sender_id,
    )
    addressed = chat_type == "private" or _event_is_addressed_to_zhvusha(
        update=update,
        message=message,
        text=text,
    )
    received_at = getattr(message, "date", None)
    if not isinstance(received_at, datetime):
        received_at = datetime.now(UTC)
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=UTC)
    if not message_id:
        message_id = uuid4().hex
    if not chat_id:
        chat_id = "unknown"
    return PersonalTelegramInboundEvent(
        event_id=f"tg-{account_label}:{chat_id}:{message_id}",
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name=_sender_name(update),
        text=text,
        account_label=account_label,
        chat_type=chat_type,
        addressed=addressed,
        received_at=received_at.astimezone(UTC),
    )


def _string_from_attrs(
    primary: object,
    secondary: object,
    *,
    names: tuple[str, ...],
) -> str:
    for source in (primary, secondary):
        for name in names:
            value = getattr(source, name, "")
            if value:
                return _stringify_telethon_value(value)
    return ""


def _message_text(*, update: object, message: object) -> str:
    for source, names in (
        (update, ("raw_text", "text")),
        (message, ("raw_text", "text", "message")),
    ):
        for name in names:
            value = getattr(source, name, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _stringify_telethon_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    for attr in ("channel_id", "chat_id", "user_id"):
        nested = getattr(value, attr, "")
        if nested:
            return str(nested).strip()
    return str(value).strip()


def _parse_telegram_chat_id(chat_id: str) -> int | str:
    stripped = str(chat_id).strip()
    try:
        return int(stripped)
    except ValueError:
        return stripped


def _event_allows_auto_reply(event: PersonalTelegramInboundEvent) -> bool:
    if event.chat_type == "private":
        return True
    if event.chat_type == "group":
        return event.addressed
    return False


def _capsule_allows_auto_reply(capsule: ContextCapsule) -> bool:
    return "can_auto_reply:true" in capsule.artifacts


def _chat_type_from_update(
    *,
    update: object,
    message: object,
    chat_id: str,
    sender_id: str,
) -> PersonalTelegramChatType:
    if _truthy_member(update, "is_private") or _truthy_member(message, "is_private"):
        return "private"
    if _truthy_member(update, "is_group") or _truthy_member(message, "is_group"):
        return "group"
    if _truthy_member(update, "is_channel") or _truthy_member(message, "is_channel"):
        return "channel"

    parsed_chat_id = _parse_int(chat_id)
    parsed_sender_id = _parse_int(sender_id)
    if parsed_chat_id is None:
        return "unknown"
    if parsed_chat_id < 0 and parsed_sender_id == parsed_chat_id:
        return "channel"
    if parsed_chat_id < 0:
        return "group"
    if parsed_chat_id > 0:
        return "private"
    return "unknown"


def _event_is_addressed_to_zhvusha(
    *,
    update: object,
    message: object,
    text: str,
) -> bool:
    if _truthy_member(update, "mentioned") or _truthy_member(message, "mentioned"):
        return True

    normalized = text.casefold()
    return "жвуш" in normalized or "zhvusha" in normalized or "@zhvusha" in normalized


def _truthy_member(source: object, name: str) -> bool:
    value = getattr(source, name, False)
    if callable(value):
        try:
            value = value()
        except TypeError:
            return False
    return bool(value)


def _parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _sender_name(update: object) -> str:
    sender = getattr(update, "sender", None)
    if sender is None:
        sender = getattr(getattr(update, "message", update), "sender", None)
    if sender is None:
        return ""
    parts = [
        str(getattr(sender, "first_name", "") or "").strip(),
        str(getattr(sender, "last_name", "") or "").strip(),
    ]
    name = " ".join(part for part in parts if part)
    username = str(getattr(sender, "username", "") or "").strip()
    if username:
        username_part = username if username.startswith("@") else f"@{username}"
        return f"{name} {username_part}".strip()
    return name


def _next_actions(
    *,
    grant: SocialPermissionGrant | None,
    judgement: SocialJudgementDecision | None,
    send_gate_result: SocialSendGateResult | None,
    can_auto_reply: bool,
) -> tuple[str, ...]:
    if can_auto_reply:
        return (
            "передать событие в единый chat/skill pipeline Жвуши.",
            "отправить итоговый ответ через personal Telegram sender и отметить событие processed.",
        )
    if grant is None:
        return (
            "не отвечать автоматически; передать observation Жвуше и спросить Никиту, нужен ли social grant.",
            "оставить событие read-only, если grant не нужен.",
        )
    if send_gate_result is not None:
        if send_gate_result.allowed:
            return (
                "не отвечать автоматически; social gate разрешил только draft/approval path через Жвушу.",
                "создать черновик ответа и провести обычные approval/rate/privacy gates перед отправкой.",
            )
        return (
            "не отвечать автоматически; social gate заблокировал ответ: "
            f"{send_gate_result.reason}.",
        )
    if judgement is None:
        return (
            "не отвечать автоматически; сначала выполнить SocialJudgement, rate и privacy checks.",
        )
    if judgement.can_send:
        return (
            "не отвечать автоматически; подготовить draft/approval path через Жвушу и social gates.",
        )
    return (
        f"не отвечать автоматически; SocialJudgement выбрал {judgement.action.value}: {judgement.reason}",
    )
