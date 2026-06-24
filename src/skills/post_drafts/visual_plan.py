"""Visual intent planning for channel post drafts."""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
_INTERNAL_RE = re.compile(
    r"\b(жвуш|self-coding|самокод|agent runtime|memory|архитектур|codex|workspace)\b",
    re.IGNORECASE,
)
_INTIMATE_RE = re.compile(
    r"\b(интим|личн|дневник|без визуала|уязвим|страшно|стыд|сон|чувств)\b",
    re.IGNORECASE,
)
_UNSAFE_RE = re.compile(
    r"(```|\.env\b|token\b|secret\b|password\b|api[_-]?key\b|"
    r"stack trace|traceback|/home/|/tmp/|localhost|127\.0\.0\.1|"
    r"private dashboard|dashboard|admin panel|лог[и]?|logs?\b|terminal|терминал)",
    re.IGNORECASE,
)
_PRIVATE_HOSTS = {"localhost", "metadata.google.internal"}


def plan_visual_for_draft(
    *,
    title: str,
    source_cluster: str,
    text: str,
) -> dict[str, Any]:
    """Classify visual intent and fail closed on private/sensitive material."""
    combined = "\n".join((title, source_cluster, text))
    safety_notes = _safety_notes(combined)
    if safety_notes:
        return _denied_plan("; ".join(safety_notes))

    url = _extract_first_url(combined)
    if url:
        if not is_public_source_url(url):
            return _denied_plan("source_url is not a public http(s) source")
        return {
            "intent": "source_screenshot",
            "required": True,
            "status": "planned",
            "source_url": url,
            "caption": "Публичный источник к посту",
            "safety_notes": [],
        }

    if _INTIMATE_RE.search(combined):
        return {
            "intent": "none",
            "required": False,
            "status": "none",
            "reason": "visual would weaken the author's voice",
            "safety_notes": [],
        }

    if _INTERNAL_RE.search(combined):
        prompt = (
            "Сделай честную визуальную карточку или простую схему для поста "
            "канала Жвуши. Это не скриншот, не код, не терминал и не приватный "
            f"интерфейс. Тема: {title.strip()}. Смысл: {_compact(text)}"
        )
        return {
            "intent": "generated",
            "required": True,
            "status": "planned",
            "prompt": prompt,
            "caption": "Визуальная схема к мысли Жвуши",
            "safety_notes": [],
        }

    return {
        "intent": "none",
        "required": False,
        "status": "none",
        "reason": "no clear explanatory value",
        "safety_notes": [],
    }


def is_public_source_url(url: str) -> bool:
    """Return False for local, private, credentialed or non-http URLs."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    host = parsed.hostname.strip().lower().rstrip(".")
    if host in _PRIVATE_HOSTS or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _extract_first_url(text: str) -> str:
    match = _URL_RE.search(text)
    return match.group(0).rstrip(".,;:!?)]}") if match else ""


def _safety_notes(text: str) -> list[str]:
    notes: list[str] = []
    if _UNSAFE_RE.search(text):
        notes.append("sensitive_or_private_material")
    return notes


def _denied_plan(reason: str) -> dict[str, Any]:
    return {
        "intent": "denied",
        "required": False,
        "status": "denied",
        "denial_reason": reason,
        "safety_notes": [reason],
    }


def _compact(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:500]
