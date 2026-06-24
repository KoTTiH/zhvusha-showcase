"""Browser workflow draft worker backed by Tool Gateway."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from src.agent_runtime.models import (
    ContextCapsule,
    Finding,
    FindingStatus,
)

if TYPE_CHECKING:
    from src.agent_runtime.models import AgentJob, ContextPack
    from src.agent_runtime.tools import ToolGateway


class BrowserWorkflowDraftWorkerBackend:
    """Read a form page and prepare a draft artifact without submitting it."""

    name = "browser_workflow"

    def __init__(self, *, tool_gateway: ToolGateway, max_source_chars: int = 12_000):
        self._tool_gateway = tool_gateway
        self._max_source_chars = max_source_chars

    async def run(
        self,
        *,
        job: AgentJob,
        context_pack: ContextPack,
    ) -> ContextCapsule:
        payload = _payload_from_context(context_pack)
        read_url = _string(payload, "read_url") or _string(payload, "form_url")
        action_url = _string(payload, "action_url") or _string(payload, "url")
        if not read_url and action_url:
            read_url = action_url
        if not read_url:
            return _failed_capsule("Нужен read_url или action_url для формы.")

        source_text = str(
            await self._tool_gateway.execute(
                job.profile,
                "browser_read_url",
                {"url": read_url},
            )
        )
        form = _FirstFormParser.from_html(source_text, base_url=read_url)
        if not action_url:
            action_url = form.action_url or read_url
        method = (_string(payload, "method") or form.method or "POST").upper()
        fields = payload.get("fields")
        if fields is None:
            fields = dict.fromkeys(form.field_names, "")

        draft_artifact = str(
            await self._tool_gateway.execute(
                job.profile,
                "browser_draft_form",
                {
                    "url": action_url,
                    "method": method,
                    "fields": fields,
                    "purpose": _string(payload, "purpose"),
                    "notes": _string(payload, "notes"),
                },
            )
        )

        field_names = tuple(str(key) for key in dict(fields))
        processed_context = _processed_context(
            read_url=read_url,
            action_url=action_url,
            method=method,
            parsed_fields=form.field_names,
            requested_fields=field_names,
            source_text=source_text,
        )
        return ContextCapsule(
            summary="Подготовлен browser form draft без отправки формы.",
            processed_context=processed_context,
            findings=(
                Finding(
                    claim=f"Форма прочитана через browser_read_url: {read_url}",
                    status=FindingStatus.CONFIRMED,
                    confidence=0.9,
                    evidence=(read_url,),
                ),
                Finding(
                    claim=f"Draft artifact создан через browser_draft_form: {draft_artifact}",
                    status=FindingStatus.CONFIRMED,
                    confidence=0.95,
                    evidence=(draft_artifact,),
                ),
                Finding(
                    claim="Submit capability не использовалась; draft сохраняет submit boundary.",
                    status=FindingStatus.CONFIRMED,
                    confidence=0.95,
                    evidence=("browser_submit denied by profile", draft_artifact),
                ),
            ),
            sources=(read_url,),
            artifacts=(draft_artifact,),
            next_actions=("Передать draft observation Жвуше для ответа.",),
            markdown_report=processed_context,
        )

    async def cancel(self, job_id: str) -> bool:
        """No long-lived process is held by this worker."""
        del job_id
        return False


class _FirstFormParser(HTMLParser):
    def __init__(self, *, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url
        self._inside_first_form = False
        self._seen_form = False
        self.action_url = ""
        self.method = ""
        self.field_names: tuple[str, ...] = ()
        self._fields: list[str] = []

    @classmethod
    def from_html(cls, html: str, *, base_url: str) -> _FirstFormParser:
        parser = cls(base_url=base_url)
        parser.feed(html)
        parser.field_names = tuple(dict.fromkeys(parser._fields))
        return parser

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "form" and not self._seen_form:
            self._seen_form = True
            self._inside_first_form = True
            action = attrs_map.get("action", "").strip()
            self.action_url = urljoin(self._base_url, action) if action else ""
            self.method = attrs_map.get("method", "").strip().upper()
            return
        if not self._inside_first_form:
            return
        if tag.lower() in {"input", "textarea", "select"}:
            name = attrs_map.get("name", "").strip()
            if name:
                self._fields.append(name)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "form" and self._inside_first_form:
            self._inside_first_form = False


def _payload_from_context(context_pack: ContextPack) -> dict[str, Any]:
    raw = context_pack.metadata.get("browser_workflow_payload", "")
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("browser workflow payload must be a JSON object")
    return data


def _string(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key, "")).strip()


def _processed_context(
    *,
    read_url: str,
    action_url: str,
    method: str,
    parsed_fields: tuple[str, ...],
    requested_fields: tuple[str, ...],
    source_text: str,
) -> str:
    source_excerpt = source_text.strip()[:12_000]
    if len(source_text.strip()) > 12_000:
        source_excerpt += "\n...[truncated]"
    return "\n".join(
        (
            "# Browser workflow draft",
            f"Read URL: {read_url}",
            f"Action URL: {action_url}",
            f"Method: {method}",
            "Parsed fields: "
            + (", ".join(parsed_fields) if parsed_fields else "unknown"),
            "Draft fields: "
            + (", ".join(requested_fields) if requested_fields else "none"),
            "Submit boundary: browser_submit was not used and remains denied.",
            "",
            "## Source excerpt",
            source_excerpt,
        )
    )


def _failed_capsule(summary: str) -> ContextCapsule:
    return ContextCapsule(
        summary=summary,
        findings=(
            Finding(
                claim=summary,
                status=FindingStatus.UNCONFIRMED,
                confidence=1.0,
            ),
        ),
        next_actions=("Указать публичный URL формы и значения draft fields.",),
        markdown_report=summary,
    )
