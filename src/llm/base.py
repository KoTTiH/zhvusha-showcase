"""Base adapter interface for the LLM Gateway.

Adapters are **private** implementation details of the LLM Gateway capability
module. External modules MUST NOT import this file — instead, depend on
``src.llm.protocols.LLMGatewayProtocol`` and use ``LLMRouter`` (or a mock)
as the implementation. The ``llm_gateway_isolation`` rule in ``.importlinter``
enforces this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.protocols import (
        LLMRequest,
        LLMResponse,
        LLMToolRequest,
        LLMToolResponse,
    )


class BaseLLMAdapter(ABC):
    """Base class for all LLM provider adapters.

    Subclasses wire a concrete provider (Codex CLI, Anthropic SDK,
    Google GenAI, OpenAI-compatible routers, or legacy Claude CLI) to the
    common ``LLMRequest`` / ``LLMResponse`` shape.
    ``LLMRouter`` is the sole public consumer of this class.
    """

    name: str
    default_model: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Dispatch a single-shot request to the provider and return a response.

        The router resolves ``request.tier`` to a concrete model before calling
        the adapter, so adapters can treat ``request.model`` as authoritative
        (falling back to ``default_model`` only if explicitly set to ``None``).
        """

    async def generate_with_tools(
        self,
        request: LLMToolRequest,
    ) -> LLMToolResponse:
        """Tool-use generation. Default implementation raises.

        Subclasses that support ``tool_use`` override this method. Others
        inherit the ``NotImplementedError`` — the router will propagate it
        to callers.
        """
        raise NotImplementedError(f"{self.name} does not support tool_use")
