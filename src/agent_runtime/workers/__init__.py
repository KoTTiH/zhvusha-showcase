"""Agent Runtime worker backends."""

from src.agent_runtime.computer_use import ComputerUseWorkerBackend
from src.agent_runtime.workers.agency import AgencyWorkerBackend
from src.agent_runtime.workers.browser_workflow import BrowserWorkflowDraftWorkerBackend
from src.agent_runtime.workers.channel_visual import ChannelVisualWorkerBackend
from src.agent_runtime.workers.codex import CodexWorkerBackend
from src.agent_runtime.workers.daivinchik_profile import (
    DaivinchikTasteProfileWorkerBackend,
)
from src.agent_runtime.workers.external_skill import (
    ExternalSkillAgentWorker,
    ExternalSkillInvocationAdapter,
)
from src.agent_runtime.workers.self_improvement import (
    AutonomousSelfCodingWorkerBackend,
)
from src.agent_runtime.workers.source_compare import SourceCompareWorkerBackend
from src.agent_runtime.workers.telegram_mcp import TelegramMCPWorkerBackend
from src.agent_runtime.workers.web import WebResearchWorkerBackend

__all__ = [
    "AgencyWorkerBackend",
    "AutonomousSelfCodingWorkerBackend",
    "BrowserWorkflowDraftWorkerBackend",
    "ChannelVisualWorkerBackend",
    "CodexWorkerBackend",
    "ComputerUseWorkerBackend",
    "DaivinchikTasteProfileWorkerBackend",
    "ExternalSkillAgentWorker",
    "ExternalSkillInvocationAdapter",
    "SourceCompareWorkerBackend",
    "TelegramMCPWorkerBackend",
    "WebResearchWorkerBackend",
]
