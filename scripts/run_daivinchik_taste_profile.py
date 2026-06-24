#!/usr/bin/env python3
"""Run the read/media-only Daivinchik taste profile Agent Runtime pass."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import cast

from src.agent_runtime.events import FileAgentEventStream
from src.agent_runtime.models import ContextPack
from src.agent_runtime.profiles import (
    DAIVINCHIK_AUTOLIKE_MVP,
    DAIVINCHIK_TASTE_PROFILE_READONLY,
)
from src.agent_runtime.runtime import AgentRuntime
from src.agent_runtime.storage import FileAgentJobStore
from src.agent_runtime.workers.daivinchik_profile import (
    DaivinchikTasteProfileWorkerBackend,
    TerminalCodexProfileMessageClassifier,
    TerminalCodexVisionDescriber,
)
from src.agent_runtime.workers.telegram_mcp import (
    MCPStdioTelegramClient,
    build_telegram_mcp_tool_gateway,
)
from src.core.config import ReasoningEffort, get_settings
from src.skills.workspace_session.workspace import get_workspace_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build ~/zhvusha-workspace/social/daivinchik/taste_profile.md "
            "from Daivinchik Telegram history without send/modify/admin actions."
        )
    )
    parser.add_argument("chat_id", help="Exact Daivinchik chat_id or username.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Optional history limit. In autolike-decision mode defaults to a "
            "small current-window read when omitted."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("profile", "autolike_decision", "autolike_live"),
        default="profile",
        help=(
            "profile rebuilds the one-time taste report; autolike_decision scores "
            "only the current visible card; autolike_live also presses like/skip."
        ),
    )
    parser.add_argument(
        "--notify-chat-id",
        default="",
        help="Telegram chat_id/username for stop notifications in autolike_live mode.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=1,
        help="Maximum like/skip actions for this bounded live run.",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("collect", "stop", "ignore"),
        default="collect",
        help=(
            "collect records non-profile Daivinchik messages; stop aborts before "
            "media processing so live scrolling can ask Никита."
        ),
    )
    parser.add_argument(
        "--vision-model",
        default="",
        help="Codex model for terminal image recognition; default uses analyst_model.",
    )
    parser.add_argument(
        "--vision-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="medium",
        help="Codex reasoning effort for terminal image recognition.",
    )
    return parser


async def _run(
    chat_id: str,
    *,
    limit: int,
    mode: str,
    notify_chat_id: str,
    max_actions: int,
    attention_mode: str,
    vision_model: str,
    vision_effort: str,
) -> int:
    settings = get_settings()
    if not settings.telegram_mcp_enabled:
        sys.stderr.write("TELEGRAM_MCP_ENABLED=true is required.\n")
        return 2
    if not (
        settings.telegram_mcp_session_string_personal
        or settings.telegram_mcp_session_name_personal
    ):
        sys.stderr.write("Personal Telegram MCP session is not configured.\n")
        return 2

    workspace_root = get_workspace_path(settings.workspace_path)
    async with MCPStdioTelegramClient() as client:
        gateway = build_telegram_mcp_tool_gateway(
            client=client,
            account_label=settings.telegram_mcp_account_label,
        )
        worker = DaivinchikTasteProfileWorkerBackend(
            tool_gateway=gateway,
            workspace_root=workspace_root,
            llm=None,
            vision_describer=TerminalCodexVisionDescriber(
                codex_path=settings.codex_cli_path,
                model=vision_model.strip() or settings.analyst_model,
                reasoning_effort=cast("ReasoningEffort", vision_effort),
            ),
            profile_classifier=TerminalCodexProfileMessageClassifier(
                codex_path=settings.codex_cli_path,
            ),
        )
        profile = (
            DAIVINCHIK_AUTOLIKE_MVP
            if mode == "autolike_live"
            else DAIVINCHIK_TASTE_PROFILE_READONLY
        )
        runtime = AgentRuntime(
            store=FileAgentJobStore(workspace_root / "agent_runtime"),
            events=FileAgentEventStream(workspace_root / "agent_runtime"),
            workers={DAIVINCHIK_TASTE_PROFILE_READONLY.worker: worker},
        )
        request: dict[str, str | int] = {"chat_id": chat_id}
        if mode in {"autolike_decision", "autolike_live"} and limit <= 0:
            limit = 20
        if limit > 0:
            request["limit"] = limit
        request["attention_mode"] = attention_mode
        request["mode"] = mode
        if notify_chat_id:
            request["notify_chat_id"] = notify_chat_id
        if mode == "autolike_live":
            request["max_actions"] = max(max_actions, 1)
        job = await runtime.create_job(
            owner_user_id=settings.admin_user_id,
            chat_id=settings.admin_user_id,
            source_message_id=f"daivinchik-profile-{int(time.time())}",
            fingerprint=f"daivinchik-profile:{chat_id}:{limit}:{time.time_ns()}",
            kind="daivinchik_taste_profile",
            profile=profile,
            context_pack=ContextPack(
                user_request=json.dumps(request, ensure_ascii=False),
                constraints=(
                    "read/media-only Telegram MCP pass",
                    "do not send messages",
                    "do not press inline buttons",
                    "except autolike_live mode with dedicated Daivinchik button capability",
                    "delete temp media after report",
                    "autolike_decision mode scores only the current card",
                ),
                metadata={
                    "agent_tool_approval_id": "manual-daivinchik-cli-run",
                    "agent_tool_approval_capabilities": (
                        "telegram_mcp_daivinchik_button,telegram_mcp_daivinchik_notify"
                    ),
                }
                if mode == "autolike_live"
                else {},
            ),
        )
        completed = await runtime.start(job.id)
    if completed.result is None:
        sys.stderr.write((completed.error or "profile run failed") + "\n")
        return 1
    sys.stdout.write(completed.result.markdown_report + "\n")
    for artifact in completed.result.artifacts:
        sys.stdout.write(f"artifact: {artifact}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return asyncio.run(
        _run(
            args.chat_id,
            limit=max(args.limit, 0),
            mode=args.mode,
            notify_chat_id=args.notify_chat_id,
            max_actions=max(args.max_actions, 1),
            attention_mode=args.attention_mode,
            vision_model=args.vision_model,
            vision_effort=args.vision_effort,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
