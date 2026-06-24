"""LLM Gateway capability module (v4).

Public contract: :mod:`src.llm.protocols`.
Concrete implementation: :mod:`src.llm.router` (``LLMRouter``).

External modules should import **types** from ``src.llm.protocols`` and the
factory (``get_router`` / ``create_router``) from ``src.llm`` directly.
Concrete adapters (``src.llm.anthropic_api`` / ``src.llm.claude_cli`` /
``src.llm.codex_cli`` / ``src.llm.gemini`` / ``src.llm.base``) are private — importing them from
outside ``src/llm/`` is blocked by the ``llm_gateway_isolation`` rule in
``.importlinter``.
"""

from src.llm.base import BaseLLMAdapter
from src.llm.claude_cli import ClaudeCLIAdapter
from src.llm.codex_cli import CodexCLIAdapter
from src.llm.gemini import GeminiAdapter
from src.llm.protocols import (
    DEFAULT_VISION_PROMPT,
    AuthenticationError,
    BudgetExceededError,
    LLMError,
    LLMGatewayProtocol,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolRequest,
    LLMToolResponse,
    LLMUsage,
    LLMVisionRequest,
    ProviderUnavailableError,
    RateLimitError,
    Tier,
    ToolDefinition,
)
from src.llm.router import LLMRouter, create_router, get_router

__all__ = [
    "DEFAULT_VISION_PROMPT",
    "AuthenticationError",
    "BaseLLMAdapter",
    "BudgetExceededError",
    "ClaudeCLIAdapter",
    "CodexCLIAdapter",
    "GeminiAdapter",
    "LLMError",
    "LLMGatewayProtocol",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMRouter",
    "LLMToolRequest",
    "LLMToolResponse",
    "LLMUsage",
    "LLMVisionRequest",
    "ProviderUnavailableError",
    "RateLimitError",
    "Tier",
    "ToolDefinition",
    "create_router",
    "get_router",
]
