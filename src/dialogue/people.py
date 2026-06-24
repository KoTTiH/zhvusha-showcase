"""People/alias evidence for dialogue memory."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

_USERNAME_RE = r"@[A-Za-z0-9_]+|-?\d{5,}"
_TARGET_FIRST_RE = re.compile(
    rf"(?P<target>{_USERNAME_RE})\s*(?:это|=|[-—])\s*(?P<alias>[^\n,;:.!?@]{{1,48}})",
    re.IGNORECASE,
)

PeopleAliasLookupStatus = Literal[
    "not_found",
    "needs_confirmation",
    "insufficient_confidence",
    "stale",
    "rejected",
]
_ALIAS_FIRST_RE = re.compile(
    rf"(?P<alias>[A-Za-zА-Яа-яЁё][^\n,;:.!?@]{{0,48}}?)\s*(?:это|=|[-—])\s*(?P<target>{_USERNAME_RE})",
    re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class PeopleAliasCandidate(BaseModel):
    """Evidence that a human alias may map to an executable Telegram id."""

    model_config = ConfigDict(extra="ignore")

    alias: str
    executable_chat_id: str
    source_text: str = ""
    source_message_id: str = ""
    scope: str = "chat"
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    observed_at: str = Field(default_factory=_now_iso)

    @field_validator(
        "alias",
        "executable_chat_id",
        "source_text",
        "source_message_id",
        "scope",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("executable_chat_id")
    @classmethod
    def _validate_executable_chat_id(cls, value: str) -> str:
        if not _looks_like_executable_chat_id(value):
            raise ValueError(
                "executable_chat_id must be explicit @username or numeric id"
            )
        return value


class PeopleAliasLookupResult(BaseModel):
    """Safe lookup result. It never authorizes immediate send execution."""

    alias: str
    status: PeopleAliasLookupStatus
    suggested_recipient: str = ""
    can_execute: bool = False
    missing_fields: tuple[str, ...] = ("chat_id",)
    reason: str = ""
    rejected_recipient: str = ""
    candidates: tuple[PeopleAliasCandidate, ...] = ()


class FilePeopleAliasStore:
    """File-backed per-chat alias candidate store."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

    def append(
        self, chat_id: int | str | None, candidate: PeopleAliasCandidate
    ) -> None:
        path = self._path(chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(candidate.model_dump_json() + "\n")

    def list_candidates(
        self, chat_id: int | str | None
    ) -> tuple[PeopleAliasCandidate, ...]:
        path = self._path(chat_id)
        if not path.exists():
            return ()
        candidates: list[PeopleAliasCandidate] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                candidates.append(PeopleAliasCandidate.model_validate_json(line))
            except (ValueError, json.JSONDecodeError):
                continue
        return tuple(candidates)

    def lookup(
        self,
        chat_id: int | str | None,
        alias: str,
        *,
        min_confidence: float = 0.7,
        max_age_days: int | None = 180,
    ) -> PeopleAliasLookupResult:
        alias_text = _clean_alias(alias)
        alias_key = _alias_key(alias_text)
        matches = tuple(
            candidate
            for candidate in reversed(self.list_candidates(chat_id))
            if _alias_key(candidate.alias) == alias_key
        )
        if not matches:
            return PeopleAliasLookupResult(alias=alias_text, status="not_found")
        blocked_status: PeopleAliasLookupStatus = "not_found"
        blocked_reason = ""
        for candidate in matches:
            if candidate.confidence < min_confidence:
                blocked_status = "insufficient_confidence"
                blocked_reason = "candidate confidence is below lookup threshold"
                continue
            if _candidate_is_stale(candidate, max_age_days=max_age_days):
                blocked_status = "stale"
                blocked_reason = "candidate evidence is older than lookup window"
                continue
            return PeopleAliasLookupResult(
                alias=alias_text,
                status="needs_confirmation",
                suggested_recipient=candidate.executable_chat_id,
                can_execute=False,
                missing_fields=("chat_id",),
                candidates=matches,
            )
        return PeopleAliasLookupResult(
            alias=alias_text,
            status=blocked_status,
            can_execute=False,
            missing_fields=("chat_id",),
            reason=blocked_reason,
            candidates=matches,
        )

    def _path(self, chat_id: int | str | None) -> Path:
        return (
            self._workspace_root
            / "logs"
            / _normalize_chat_id(chat_id)
            / "people_alias_candidates.jsonl"
        )


def extract_people_alias_candidates(
    text: str,
    *,
    source_message_id: str = "",
    scope: str = "chat",
) -> tuple[PeopleAliasCandidate, ...]:
    """Extract explicit alias facts from a message without guessing."""
    source_text = str(text or "").strip()
    if not source_text:
        return ()
    candidates: list[PeopleAliasCandidate] = []
    for pattern in (_TARGET_FIRST_RE, _ALIAS_FIRST_RE):
        for match in pattern.finditer(source_text):
            alias = _clean_alias(match.group("alias"))
            target = match.group("target").strip()
            if not alias or alias.startswith("@"):
                continue
            candidates.append(
                PeopleAliasCandidate(
                    alias=alias,
                    executable_chat_id=target,
                    source_text=source_text,
                    source_message_id=source_message_id,
                    scope=scope,
                    confidence=0.85,
                )
            )
    return tuple(_dedupe_candidates(candidates))


def render_people_alias_lookup_status(result: PeopleAliasLookupResult) -> str:
    """Render safe lookup diagnostics without raw source message text."""
    lines = ["People alias lookup:"]
    lines.append(f"- alias: {result.alias or 'missing'}")
    lines.append(f"- status: {result.status}")
    lines.append(f"- can_execute: {'yes' if result.can_execute else 'no'}")
    if result.suggested_recipient:
        lines.append(f"- suggested_recipient: {result.suggested_recipient}")
    if result.rejected_recipient:
        lines.append(f"- rejected_recipient: {result.rejected_recipient}")
    if result.reason:
        lines.append(f"- reason: {result.reason}")
    lines.append(f"- candidates: {len(result.candidates)}")
    for candidate in result.candidates[:3]:
        lines.append(
            "- candidate: "
            f"alias={candidate.alias}, "
            f"target={candidate.executable_chat_id}, "
            f"confidence={candidate.confidence:.2f}, "
            f"scope={candidate.scope}, "
            f"observed_at={candidate.observed_at}"
        )
    return "\n".join(lines)


def _dedupe_candidates(
    candidates: list[PeopleAliasCandidate],
) -> list[PeopleAliasCandidate]:
    seen: set[tuple[str, str]] = set()
    result: list[PeopleAliasCandidate] = []
    for candidate in candidates:
        key = (_alias_key(candidate.alias), candidate.executable_chat_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _clean_alias(value: str) -> str:
    return str(value or "").strip(" \t\n\r\"'«»`.,;:!?")


def _alias_key(value: str) -> str:
    key = "".join(char.lower() for char in _clean_alias(value) if char.isalnum())
    key = key.replace("ё", "е")
    if key.endswith("е") and len(key) > 2:
        return key[:-1] + "а"
    return key


def _normalize_chat_id(chat_id: int | str | None) -> str:
    return str(chat_id or "unknown").strip() or "unknown"


def _candidate_is_stale(
    candidate: PeopleAliasCandidate,
    *,
    max_age_days: int | None,
) -> bool:
    if max_age_days is None:
        return False
    observed_at = _parse_datetime(candidate.observed_at)
    if observed_at is None:
        return True
    age_seconds = (datetime.now(tz=UTC) - observed_at).total_seconds()
    return age_seconds > max(max_age_days, 0) * 24 * 60 * 60


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _looks_like_executable_chat_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.lstrip("-").isdigit():
        return True
    if not text.startswith("@"):
        return False
    username = text[1:]
    return bool(username) and all(
        char.isascii() and (char.isalnum() or char == "_") for char in username
    )
