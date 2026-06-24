"""Registries for extensible Agent Runtime profiles and capabilities."""

from __future__ import annotations

from src.agent_runtime.models import (
    AgentDefinition,
    CapabilityDefinition,
    InvocationProfile,
)


class AgentRegistry:
    """Registry of reusable agent definitions."""

    def __init__(self, *, agents: tuple[AgentDefinition, ...] = ()) -> None:
        self._agents: dict[str, AgentDefinition] = {}
        for agent in agents:
            self.register(agent)

    def register(self, agent: AgentDefinition) -> None:
        """Register or replace one agent definition by id."""
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> AgentDefinition:
        """Return one registered agent definition."""
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"unknown agent profile: {agent_id}") from exc

    def all(self) -> tuple[AgentDefinition, ...]:
        """Return all registered agents in deterministic order."""
        return tuple(self._agents[key] for key in sorted(self._agents))

    def make_invocation_profile(
        self,
        agent_id: str,
        *,
        profile_id: str = "",
        worker: str = "",
        allowed_capabilities: tuple[str, ...] | None = None,
        denied_capabilities: tuple[str, ...] = (),
        metadata: dict[str, str] | None = None,
    ) -> InvocationProfile:
        """Create a concrete capability profile from an AgentDefinition."""
        agent = self.get(agent_id)
        allowed = (
            agent.allowed_capabilities
            if allowed_capabilities is None
            else allowed_capabilities
        )
        extra = set(allowed) - set(agent.allowed_capabilities)
        if extra:
            names = ", ".join(sorted(extra))
            raise ValueError(f"capabilities not allowed for {agent_id}: {names}")
        return InvocationProfile(
            id=profile_id or f"{agent.id}.v{agent.version}",
            worker=worker or agent.default_worker,
            allowed_capabilities=allowed,
            denied_capabilities=denied_capabilities,
            metadata=metadata or {},
        )


class CapabilityRegistry:
    """Registry of known runtime capabilities."""

    def __init__(
        self,
        *,
        capabilities: tuple[CapabilityDefinition, ...] = (),
    ) -> None:
        self._capabilities: dict[str, CapabilityDefinition] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: CapabilityDefinition) -> None:
        """Register or replace one capability definition."""
        self._capabilities[capability.id] = capability

    def get(self, capability_id: str) -> CapabilityDefinition:
        """Return one capability definition."""
        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {capability_id}") from exc

    def validate_profile(self, profile: InvocationProfile) -> None:
        """Ensure an InvocationProfile mentions only known capabilities."""
        known = set(self._capabilities)
        requested = set(profile.allowed_capabilities) | set(profile.denied_capabilities)
        unknown = requested - known
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown capability in {profile.id}: {names}")
