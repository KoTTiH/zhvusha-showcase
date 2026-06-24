"""Bot-level routing helpers for messages arriving during active agent jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from src.agent_runtime.models import AgentJob, AgentJobStatus
from src.agent_runtime.rendering import render_job_status

if TYPE_CHECKING:
    from src.agent_runtime.runtime import AgentRuntime


class AgentMessageIntent(StrEnum):
    """Classification for a chat message while an agent job is active."""

    NO_ACTIVE_JOB = "no_active_job"
    STATUS_QUERY = "status_query"
    FOLLOWUP = "followup"
    NEW_TASK = "new_task"
    CANCEL = "cancel"


@dataclass(frozen=True)
class AgentRoutingDecision:
    """Result of routing one incoming message against active jobs."""

    intent: AgentMessageIntent
    job_id: str = ""
    reply: str = ""


async def route_message_for_active_job(
    runtime: AgentRuntime,
    *,
    chat_id: int,
    text: str,
) -> AgentRoutingDecision:
    """Route a message to status/follow-up/new-task behavior."""
    job = await latest_active_job_for_chat(runtime, chat_id)
    if job is None:
        return AgentRoutingDecision(intent=AgentMessageIntent.NO_ACTIVE_JOB)

    if _is_cancel_request(text):
        canceled = await runtime.cancel(job.id, reason="Никита отменил agent job")
        return AgentRoutingDecision(
            intent=AgentMessageIntent.CANCEL,
            job_id=job.id,
            reply=render_job_status(canceled, tuple(runtime.events_for(job.id))),
        )

    if _is_status_query(text):
        return AgentRoutingDecision(
            intent=AgentMessageIntent.STATUS_QUERY,
            job_id=job.id,
            reply=render_job_status(job, tuple(runtime.events_for(job.id))),
        )

    if job.status is AgentJobStatus.AWAITING_INPUT or _looks_like_followup(text):
        await runtime.attach_followup(job.id, text.strip())
        return AgentRoutingDecision(
            intent=AgentMessageIntent.FOLLOWUP,
            job_id=job.id,
            reply="Поняла, добавила это к текущей agent-задаче.",
        )

    return AgentRoutingDecision(
        intent=AgentMessageIntent.NEW_TASK,
        job_id=job.id,
        reply=(
            "Вижу новую тему. Сейчас у меня уже идёт agent-задача; "
            "не смешиваю их в один контекст."
        ),
    )


async def latest_active_job_for_chat(
    runtime: AgentRuntime,
    chat_id: int,
) -> AgentJob | None:
    """Return the newest non-terminal job for one chat, if any."""
    jobs = await runtime.store.list_by_status(
        (
            AgentJobStatus.AWAITING_INPUT,
            AgentJobStatus.QUEUED,
            AgentJobStatus.RUNNING,
            AgentJobStatus.WAITING_USER,
        )
    )
    matching = [job for job in jobs if job.chat_id == chat_id]
    if not matching:
        return None
    return max(matching, key=lambda job: job.updated_at)


def _is_status_query(text: str) -> bool:
    lower = text.strip().lower()
    return bool(
        re.search(
            r"(что\s+там|как\s+ид[её]т|завис|думаешь|почему\s+молч|"
            r"готовишь|статус|ответ\s+будет)",
            lower,
        )
    )


def _is_cancel_request(text: str) -> bool:
    lower = text.strip().lower()
    return bool(
        re.search(
            r"(отмени|отменить|останови|остановить|прерви|прервать|cancel|stop)"
            r".{0,40}(agent|агент|задач|job|джоб|ответ|анализ|кодинг|код)",
            lower,
        )
    )


def _looks_like_followup(text: str) -> bool:
    lower = text.strip().lower()
    if lower.startswith(
        (
            "и еще",
            "и ещё",
            "еще",
            "ещё",
            "добавь",
            "дополни",
            "уточнение",
            "вот",
            "держи",
            "кстати",
        )
    ):
        return True
    return bool(
        re.search(
            r"(прикреп|вложен|файл|скрин|фото|лог|это тоже|это к прошл)",
            lower,
        )
    )
