"""Claude CLI adapter — spawns the ``claude`` binary via async subprocess.

Private adapter. External modules import via ``LLMRouter`` and
``LLMGatewayProtocol`` from ``src.llm.protocols``.
"""

from __future__ import annotations

import asyncio
import json

import structlog

from src.core.config import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.protocols import LLMError, LLMRequest, LLMResponse, LLMUsage
from src.utils.subprocess_env import clean_env_for_claude_cli

logger = structlog.get_logger()


class ClaudeCLIAdapter(BaseLLMAdapter):
    """Legacy non-self-coding adapter that calls Claude CLI as a subprocess.

    Uses ``asyncio.create_subprocess_exec`` (not shell) so the prompt is
    passed as a direct argument with no shell interpolation.
    ``ANTHROPIC_API_KEY`` is stripped from the environment so the CLI uses
    its OAuth subscription when this fallback is explicitly configured.
    """

    name: str = "claude_cli"
    default_model: str = "sonnet"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        settings = get_settings()
        model = request.model or self.default_model

        cmd = [
            settings.claude_cli_path,
            "-p",
            request.prompt,
            "--model",
            model,
            "--output-format",
            "json",
        ]
        if request.system:
            cmd.extend(["--system-prompt", request.system])

        logger.info(
            "llm_request",
            adapter=self.name,
            model=model,
            prompt_len=len(request.prompt),
            has_system=bool(request.system),
            caller=request.caller,
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=clean_env_for_claude_cli(),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err_msg = ""
            try:
                data = json.loads(stdout)
                err_msg = str(data.get("result", ""))
            except (json.JSONDecodeError, ValueError):
                err_msg = stderr.decode(errors="replace").strip()
            logger.error(
                "claude_cli_error",
                returncode=proc.returncode,
                error=err_msg[:500],
            )
            raise LLMError(f"Claude CLI exited with code {proc.returncode}: {err_msg}")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse Claude CLI JSON output: {e}") from e

        if data.get("is_error"):
            raise LLMError(
                f"Claude CLI returned error: {data.get('result', 'unknown')}"
            )

        text: str = data.get("result", "")

        logger.info(
            "llm_response",
            adapter=self.name,
            model=model,
            response_len=len(text),
        )

        # Claude CLI does not expose token counts in its JSON output. Coarse
        # usage accounting for CLI calls happens via
        # ``UsageTracker.record_cli_call()``; we return zeroed LLMUsage here.
        return LLMResponse(text=text, model=model, usage=LLMUsage())
