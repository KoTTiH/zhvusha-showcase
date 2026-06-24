"""Anthropic API adapter for worker/analyst tiers (real-time chat).

Private adapter. External modules import via ``LLMRouter`` and
``LLMGatewayProtocol`` from ``src.llm.protocols``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.core.config import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.protocols import (
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMToolRequest,
    LLMToolResponse,
    LLMUsage,
)

if TYPE_CHECKING:
    import anthropic

logger = structlog.get_logger()


def _extract_usage(response: Any) -> LLMUsage:
    """Build an ``LLMUsage`` from the raw Anthropic API response usage block."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    )


class AnthropicAPIAdapter(BaseLLMAdapter):
    """Adapter using the official Anthropic Python SDK (async).

    Explicit non-self-coding LLM adapter. Faster than a CLI fallback because
    it avoids subprocess overhead. System prompts are sent with
    ``cache_control={"type": "ephemeral"}`` for prompt caching.
    """

    name: str = "anthropic_api"
    # Kept on the class to satisfy ``BaseLLMAdapter`` — the router resolves
    # the real api_id via ``src.llm.providers`` before each call.
    default_model: str = ""

    def __init__(self) -> None:
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            import anthropic

            settings = get_settings()
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        if not model:
            raise LLMError("AnthropicAPIAdapter received empty model")

        logger.info(
            "llm_request",
            adapter=self.name,
            model=model,
            prompt_len=len(request.prompt),
            has_system=bool(request.system),
            caller=request.caller,
        )

        try:
            client = self._get_client()
            messages: list[dict[str, str]] = [
                {"role": "user", "content": request.prompt}
            ]
            kwargs: dict[str, object] = {
                "model": model,
                "max_tokens": 4096,
                "messages": messages,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.system:
                system_blocks: list[dict[str, object]] = [
                    {
                        "type": "text",
                        "text": request.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                kwargs["system"] = system_blocks
            response = await client.messages.create(**kwargs)  # type: ignore[call-overload]
        except Exception as exc:
            logger.error("anthropic_api_error", error=str(exc)[:500])
            raise LLMError(f"Anthropic API error: {exc}") from exc

        if not response.content:
            raise LLMError("Empty response from Anthropic API")
        block = response.content[0]
        if not hasattr(block, "text"):
            raise LLMError(f"Unexpected response block type: {type(block).__name__}")
        text: str = block.text

        usage = _extract_usage(response)

        logger.info(
            "llm_response",
            adapter=self.name,
            model=model,
            response_len=len(text),
            cache_read=usage.cache_read_tokens,
            cache_write=usage.cache_write_tokens,
        )

        return LLMResponse(text=text, model=model, usage=usage)

    async def generate_with_tools(self, request: LLMToolRequest) -> LLMToolResponse:
        """Generate with tool_use support via the Anthropic Messages API.

        Returns an ``LLMToolResponse`` whose ``content_blocks`` may include
        text blocks and tool_use blocks. The caller manages the agentic loop.
        """
        model = request.model or self.default_model
        if not model:
            raise LLMError("AnthropicAPIAdapter received empty model")

        logger.info(
            "llm_tool_request",
            adapter=self.name,
            model=model,
            num_messages=len(request.messages),
            num_tools=len(request.tools),
            has_system=bool(request.system),
            caller=request.caller,
        )

        api_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in request.tools
        ]

        try:
            client = self._get_client()
            kwargs: dict[str, object] = {
                "model": model,
                "max_tokens": 4096,
                "messages": request.messages,
                "tools": api_tools,
            }
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.system:
                system_blocks: list[dict[str, object]] = [
                    {
                        "type": "text",
                        "text": request.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                kwargs["system"] = system_blocks
            response = await client.messages.create(**kwargs)  # type: ignore[call-overload]
        except Exception as exc:
            logger.error("anthropic_api_tool_error", error=str(exc)[:500])
            raise LLMError(f"Anthropic API tool_use error: {exc}") from exc

        usage = _extract_usage(response)

        logger.info(
            "llm_tool_response",
            adapter=self.name,
            model=model,
            stop_reason=response.stop_reason,
            num_blocks=len(response.content),
            cache_read=usage.cache_read_tokens,
            cache_write=usage.cache_write_tokens,
        )

        return LLMToolResponse(
            content_blocks=list(response.content),
            stop_reason=response.stop_reason,
            model=model,
            usage=usage,
        )
