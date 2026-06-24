"""User-facing external skill acquisition through the central skill gate."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field

from src.llm.protocols import LLMRequest
from src.skills.base import (
    AgentContext,
    ExecutionPlan,
    InlineSkill,
    SideEffect,
    SimulatedResult,
    SkillResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.agent_runtime.capability_graph import CapabilityGraph
    from src.llm.protocols import LLMGatewayProtocol

from src.skills.external_skill_loader.acquisition import (
    AgentskillsCatalogSourceProvider,
    CapabilityGapDetector,
    CapabilityGapReport,
    ExternalSkillAcquisitionImporter,
    ExternalSkillAcquisitionProposal,
    ExternalSkillCandidate,
    ExternalSkillCandidateSearchReport,
    ExternalSkillCandidateSearchService,
    LocalFolderExternalSkillSourceProvider,
)
from src.skills.external_skill_loader.loader import (
    FileExternalSkillQuarantineStore,
    FilePersonalSkillRegistry,
)
from src.skills.external_skill_loader.native_conversion import (
    NativeSkillConversionSpecGenerator,
)

SEARCH_PREFIX = "/external_skill_search"
IMPORT_PREFIX = "/external_skill_import"
APPROVE_READONLY_PREFIX = "/external_skill_approve_readonly"
APPROVE_EXECUTION_PREFIX = "/external_skill_approve_execution"
MARK_NATIVE_PREFIX = "/external_skill_mark_native"
DEFAULT_AGENTSKILLS_CATALOG_URL = "https://agentskills.io/catalog.json"
MAX_AGENTSKILLS_ARCHIVE_BYTES = 10_000_000
AgentskillsCatalogFetcher = Callable[[str], str]
AgentskillsArchiveFetcher = Callable[[str], bytes]
GAP_INTENT_MIN_CONFIDENCE = 0.72
GAP_INTENT_TIMEOUT_SECONDS = 15.0
_GAP_ACTION_MARKERS: tuple[str, ...] = (
    "проверь",
    "провер",
    "найди",
    "диагност",
    "почини",
    "исправ",
    "разбери",
    "аудит",
    "ошиб",
    "проблем",
    "debug",
    "diagnos",
    "troubleshoot",
    "fix",
)
_GENERIC_CAPABILITY_TOKENS: frozenset[str] = frozenset(
    {
        "read",
        "run",
        "write",
        "send",
        "source",
        "sources",
        "command",
        "commands",
        "debug",
        "readonly",
        "workspace",
        "external",
        "skill",
        "tool",
        "tools",
    }
)


class ExternalSkillGapIntent(BaseModel):
    """Classifier result for normal-chat capability gap acquisition."""

    should_acquire_skill: bool = False
    required_capabilities: tuple[str, ...] = ()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ExternalSkillGapClassifier(Protocol):
    """Classify normal chat into a possible external skill gap."""

    async def classify(self, message: str) -> ExternalSkillGapIntent: ...


class LLMExternalSkillGapClassifier:
    """Worker-tier classifier for normal-chat external skill acquisition gaps."""

    def __init__(self, *, llm_router: LLMGatewayProtocol) -> None:
        self._llm_router = llm_router

    async def classify(self, message: str) -> ExternalSkillGapIntent:
        response = await self._llm_router.generate(
            LLMRequest(
                prompt=(f"Сообщение Никиты:\n{message.strip()}\n\nВерни только JSON."),
                system=_GAP_INTENT_SYSTEM,
                tier="worker",
                reasoning_effort="low",
                temperature=0.0,
                caller="external_skill_gap_intent",
            )
        )
        return _parse_gap_intent_json(response.text)


@dataclass(frozen=True)
class _AcquisitionCommand:
    action: Literal[
        "search",
        "import",
        "approve_readonly",
        "approve_execution",
        "mark_native",
    ]
    requested_capabilities: tuple[str, ...] = ()
    user_request: str = ""
    candidate_id: str = ""
    skill_id: str = ""
    allowed_sources: tuple[str, ...] = ("local_folder",)


@dataclass(frozen=True)
class _SearchState:
    proposal: ExternalSkillAcquisitionProposal
    report: ExternalSkillCandidateSearchReport


class ExternalSkillAcquisitionSkill(InlineSkill):
    """Search and import external skill candidates with scoped approvals."""

    name: ClassVar[str] = "external_skill_acquisition"
    description: ClassVar[str] = (
        "Search and import external skill candidates through scoped approval"
    )
    llm_tier: ClassVar[Literal["worker", "analyst", "strategist"]] = "worker"
    triggers: ClassVar[list[str]] = [
        SEARCH_PREFIX,
        IMPORT_PREFIX,
        APPROVE_READONLY_PREFIX,
        APPROVE_EXECUTION_PREFIX,
        MARK_NATIVE_PREFIX,
    ]
    cost_estimate: ClassVar[Literal["low", "medium", "high"]] = "low"
    approval_policy: ClassVar[Literal["auto", "required", "mode_dependent"]] = (
        "required"
    )
    side_effects: ClassVar[list[SideEffect]] = [
        SideEffect.READS_FILESYSTEM,
        SideEffect.WRITES_FILESYSTEM,
        SideEffect.NETWORK_IO_EXTERNAL,
    ]
    mode_tags: ClassVar[list[Literal["personal", "assistant", "social"]]] = ["personal"]

    def __init__(
        self,
        *,
        admin_user_id: int,
        catalog_root: Path,
        quarantine_root: Path,
        registry_root: Path,
        agentskills_catalog_url: str = DEFAULT_AGENTSKILLS_CATALOG_URL,
        fetch_agentskills_catalog: AgentskillsCatalogFetcher | None = None,
        fetch_agentskills_archive: AgentskillsArchiveFetcher | None = None,
        gap_classifier: ExternalSkillGapClassifier | None = None,
        gap_detector: CapabilityGapDetector | None = None,
        native_conversion_minimum_uses: int = 3,
        gap_intent_timeout_seconds: float = GAP_INTENT_TIMEOUT_SECONDS,
    ) -> None:
        self._admin_user_id = admin_user_id
        self._catalog_root = catalog_root.expanduser().resolve()
        self._quarantine_root = quarantine_root.expanduser().resolve()
        self._registry_root = registry_root.expanduser().resolve()
        self._fetch_agentskills_archive = (
            fetch_agentskills_archive or _fetch_agentskills_archive
        )
        self._gap_classifier = gap_classifier
        self._gap_detector = gap_detector or CapabilityGapDetector()
        self._capability_graph: CapabilityGraph | None = None
        self._native_conversion_minimum_uses = native_conversion_minimum_uses
        self._gap_intent_timeout_seconds = max(0.1, gap_intent_timeout_seconds)
        providers: list[
            LocalFolderExternalSkillSourceProvider | AgentskillsCatalogSourceProvider
        ] = [
            LocalFolderExternalSkillSourceProvider(root=self._catalog_root),
        ]
        if agentskills_catalog_url:
            providers.append(
                AgentskillsCatalogSourceProvider(
                    catalog_url=agentskills_catalog_url,
                    fetch_catalog=fetch_agentskills_catalog
                    or _fetch_agentskills_catalog,
                )
            )
        self._search_service = ExternalSkillCandidateSearchService(
            providers=tuple(providers)
        )
        self._prepared_commands: dict[
            tuple[int, int | None, str, str], _AcquisitionCommand
        ] = {}
        self._search_state: dict[tuple[int, int | None, str], _SearchState] = {}
        self._gap_reports: dict[
            tuple[int, int | None, str, str], CapabilityGapReport
        ] = {}

    def set_capability_graph(self, graph: CapabilityGraph) -> None:
        """Inject the current truth graph used for normal-chat gap detection."""
        self._capability_graph = graph

    async def can_handle(self, message: str, context: AgentContext) -> float:
        """Match explicit acquisition commands in Никита's personal chat."""
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return 0.0
        if context.metadata.get("prefer_chat_response_only") is True:
            return 0.0
        text = message.strip()
        if text.startswith(
            (
                SEARCH_PREFIX,
                IMPORT_PREFIX,
                APPROVE_READONLY_PREFIX,
                APPROVE_EXECUTION_PREFIX,
                MARK_NATIVE_PREFIX,
            )
        ):
            return 0.95
        if text.startswith("/"):
            return 0.0
        if _looks_like_codex_goal_handoff(text, context):
            return 0.0
        if _explicitly_disallows_external_skill_acquisition(text):
            return 0.0
        report = await self._detect_normal_chat_gap(text, context)
        if report is not None and report.proposal is not None:
            return 0.92
        return 0.0

    async def prepare(self, message: str, context: AgentContext) -> ExecutionPlan:
        """Build a search/import approval plan without side effects."""
        try:
            command = _parse_command(message)
        except ValueError as exc:
            report = self._gap_reports.get(_message_key(context, message))
            if report is None or report.proposal is None:
                return _missing_input_plan(str(exc))
            command = _command_from_gap_report(report)
        self._prepared_commands[_message_key(context, message)] = command
        import_candidate_source = ""
        if command.action == "import":
            state = self._search_state.get(_context_key(context))
            candidate = (
                _find_candidate(state.report.candidates, command.candidate_id)
                if state is not None
                else None
            )
            import_candidate_source = candidate.source_type if candidate else ""
        return _plan_from_command(
            command,
            import_candidate_source=import_candidate_source,
        )

    async def dry_run(self, plan: ExecutionPlan) -> SimulatedResult:
        """Fail closed before approval when local prerequisites are missing."""
        blockers: list[str] = []
        allowed_sources = tuple(
            str(source)
            for source in plan.metadata.get("external_skill_sources", ())
            if str(source)
        )
        if (
            plan.metadata.get("external_skill_acquisition_action") == "search"
            and "local_folder" in allowed_sources
            and set(allowed_sources) <= {"local_folder"}
            and not self._catalog_root.exists()
        ):
            blockers.append("external skill local catalog does not exist")
        if plan.metadata.get("external_skill_acquisition_action") == "import":
            candidate_id = str(plan.metadata.get("external_skill_candidate_id") or "")
            if not candidate_id:
                blockers.append("external skill candidate id is missing")
        if plan.metadata.get("external_skill_acquisition_action") in {
            "approve_readonly",
            "approve_execution",
            "mark_native",
        }:
            skill_id = str(plan.metadata.get("external_skill_id") or "")
            if not skill_id:
                blockers.append("external skill id is missing")
            elif not (self._registry_root / f"{skill_id}.json").is_file():
                blockers.append("external skill registry record not found")
        return SimulatedResult(
            would_succeed=not blockers,
            would_produce=plan.human_summary,
            dependencies_available=not blockers,
            estimated_actual_cost=plan.estimated_cost_usd,
            blockers=blockers,
        )

    async def execute(self, message: str, context: AgentContext) -> SkillResult:
        """Run search/import after SkillInvocationService approval."""
        if context.mode != "personal" or context.user_id != self._admin_user_id:
            return SkillResult(success=False, response="")
        command = self._prepared_commands.pop(_message_key(context, message), None)
        if command is None:
            try:
                command = _parse_command(message)
            except ValueError as exc:
                return SkillResult(success=False, response=str(exc))
        if command.action == "search":
            return self._execute_search(command, context)
        if command.action == "import":
            return self._execute_import(command, context)
        if command.action == "approve_readonly":
            return self._execute_approve_readonly(command, context)
        if command.action == "approve_execution":
            return self._execute_approve_execution(command, context)
        return self._execute_mark_native(command, context)

    async def _detect_normal_chat_gap(
        self,
        message: str,
        context: AgentContext,
    ) -> CapabilityGapReport | None:
        if (
            self._gap_classifier is None
            or self._capability_graph is None
            or context.mode != "personal"
            or context.user_id != self._admin_user_id
        ):
            return None
        key = _message_key(context, message)
        cached = self._gap_reports.get(key)
        if cached is not None:
            return cached
        try:
            intent = await asyncio.wait_for(
                self._gap_classifier.classify(message),
                timeout=self._gap_intent_timeout_seconds,
            )
        except Exception:
            return None
        if (
            not intent.should_acquire_skill
            or intent.confidence < GAP_INTENT_MIN_CONFIDENCE
            or not intent.required_capabilities
        ):
            hinted_capabilities = _capability_hints_from_graph(
                message=message,
                capability_graph=self._capability_graph,
            )
            if not hinted_capabilities:
                return None
            intent = ExternalSkillGapIntent(
                should_acquire_skill=True,
                required_capabilities=hinted_capabilities,
                confidence=GAP_INTENT_MIN_CONFIDENCE,
                reason="capability graph hint",
            )
        report = self._gap_detector.detect(
            user_request=message,
            required_capabilities=intent.required_capabilities,
            capability_graph=self._capability_graph,
        )
        if report.proposal is None:
            hinted_capabilities = tuple(
                capability
                for capability in _capability_hints_from_graph(
                    message=message,
                    capability_graph=self._capability_graph,
                )
                if capability not in set(intent.required_capabilities)
            )
            if not hinted_capabilities:
                return None
            report = self._gap_detector.detect(
                user_request=message,
                required_capabilities=(
                    *intent.required_capabilities,
                    *hinted_capabilities,
                ),
                capability_graph=self._capability_graph,
            )
            if report.proposal is None:
                return None
        self._gap_reports[key] = report
        return report

    def _execute_search(
        self,
        command: _AcquisitionCommand,
        context: AgentContext,
    ) -> SkillResult:
        approval_id = str(context.metadata.get("skill_approval_id") or "")
        proposal = ExternalSkillAcquisitionProposal(
            user_request=command.user_request,
            requested_capabilities=command.requested_capabilities,
            allowed_sources=command.allowed_sources,
        ).approve_for_search(
            approval_id=approval_id,
            approved_by_user_id=context.user_id,
            grants_network_fetch="agentskills.io" in set(command.allowed_sources),
            grants_import_to_quarantine="local_folder" in set(command.allowed_sources),
        )
        report = self._search_service.search(proposal)
        self._search_state[_context_key(context)] = _SearchState(
            proposal=proposal,
            report=report,
        )
        return SkillResult(
            success=True,
            response=report.render_for_chat(),
            metadata={
                "skill_name": self.name,
                "external_skill_sources": command.allowed_sources,
                "external_skill_candidates": [
                    candidate.model_dump(mode="json") for candidate in report.candidates
                ],
                "dialogue_state_patch": {
                    "selected_skill": self.name,
                    "last_result": "external_skill_candidates_found",
                    "source": "external_skill_acquisition.search",
                },
            },
        )

    def _execute_import(
        self,
        command: _AcquisitionCommand,
        context: AgentContext,
    ) -> SkillResult:
        state = self._search_state.get(_context_key(context))
        if state is None:
            return SkillResult(
                success=False,
                response="Сначала нужен approved search: `/external_skill_search ...`.",
            )
        candidate = _find_candidate(state.report.candidates, command.candidate_id)
        if candidate is None:
            return SkillResult(
                success=False,
                response=f"Candidate `{command.candidate_id}` не найден в последнем search.",
            )
        importer = ExternalSkillAcquisitionImporter(
            quarantine_store=FileExternalSkillQuarantineStore(self._quarantine_root),
            registry=FilePersonalSkillRegistry(self._registry_root),
        )
        import_proposal = _proposal_for_import(
            state.proposal,
            candidate=candidate,
            approval_id=str(context.metadata.get("skill_approval_id") or ""),
            approved_by_user_id=context.user_id,
        )
        try:
            result = self._search_service.import_candidate(
                import_proposal,
                candidate=candidate,
                importer=importer,
                fetch_remote_archive=self._fetch_agentskills_archive,
            )
        except (PermissionError, ValueError) as exc:
            return SkillResult(
                success=False,
                response=f"Не импортирую candidate: {exc}",
                metadata={
                    "skill_name": self.name,
                    "external_skill_candidate_id": candidate.candidate_id,
                    "external_skill_candidate_source": candidate.source_type,
                },
            )
        return SkillResult(
            success=True,
            response=result.render_audit_for_chat(),
            metadata={
                "skill_name": self.name,
                "external_skill_id": result.registry_record.skill_id,
                "external_skill_status": result.registry_record.status.value,
                "dialogue_state_patch": {
                    "selected_skill": self.name,
                    "last_result": "external_skill_imported_to_quarantine",
                    "source": "external_skill_acquisition.import",
                },
            },
        )

    def _execute_approve_readonly(
        self,
        command: _AcquisitionCommand,
        context: AgentContext,
    ) -> SkillResult:
        approval_id = str(context.metadata.get("skill_approval_id") or "")
        registry = FilePersonalSkillRegistry(self._registry_root)
        try:
            record = registry.approve_readonly(
                command.skill_id,
                approval_id=approval_id,
                approved_by_user_id=context.user_id,
            )
        except (OSError, ValueError) as exc:
            return SkillResult(
                success=False,
                response=f"Не включаю read-only external skill: {exc}",
                metadata={
                    "skill_name": self.name,
                    "external_skill_id": command.skill_id,
                },
            )
        return SkillResult(
            success=True,
            response=(
                f"External skill `{record.skill_id}` approved for read-only "
                "procedural use. Execution всё ещё требует отдельный approval."
            ),
            metadata={
                "skill_name": self.name,
                "external_skill_id": record.skill_id,
                "external_skill_status": record.status.value,
                "dialogue_state_patch": {
                    "selected_skill": self.name,
                    "last_result": "external_skill_approved_readonly",
                    "source": "external_skill_acquisition.approve_readonly",
                },
            },
        )

    def _execute_approve_execution(
        self,
        command: _AcquisitionCommand,
        context: AgentContext,
    ) -> SkillResult:
        approval_id = str(context.metadata.get("skill_approval_id") or "")
        registry = FilePersonalSkillRegistry(self._registry_root)
        try:
            record = registry.approve_execution(
                command.skill_id,
                approval_id=approval_id,
                approved_by_user_id=context.user_id,
                approved_capabilities=command.requested_capabilities,
            )
        except (OSError, ValueError) as exc:
            return SkillResult(
                success=False,
                response=f"Не включаю execution external skill: {exc}",
                metadata={
                    "skill_name": self.name,
                    "external_skill_id": command.skill_id,
                },
            )
        capabilities = ", ".join(record.approved_capabilities) or "нет"
        return SkillResult(
            success=True,
            response=(
                f"External skill `{record.skill_id}` execution approved for: "
                f"{capabilities}. ToolGateway всё равно требует отдельный "
                "approval на каждый side-effect invocation."
            ),
            metadata={
                "skill_name": self.name,
                "external_skill_id": record.skill_id,
                "external_skill_status": record.status.value,
                "external_skill_approved_capabilities": record.approved_capabilities,
                "dialogue_state_patch": {
                    "selected_skill": self.name,
                    "last_result": "external_skill_approved_execution",
                    "source": "external_skill_acquisition.approve_execution",
                },
            },
        )

    def _execute_mark_native(
        self,
        command: _AcquisitionCommand,
        context: AgentContext,
    ) -> SkillResult:
        approval_id = str(context.metadata.get("skill_approval_id") or "")
        registry = FilePersonalSkillRegistry(self._registry_root)
        try:
            record = registry.mark_native_conversion_candidate(
                command.skill_id,
                approval_id=approval_id,
                approved_by_user_id=context.user_id,
                minimum_successful_uses=self._native_conversion_minimum_uses,
            )
        except (OSError, ValueError) as exc:
            return SkillResult(
                success=False,
                response=f"Не отмечаю external skill для native conversion: {exc}",
                metadata={
                    "skill_name": self.name,
                    "external_skill_id": command.skill_id,
                },
            )
        title = record.native_conversion_spec_title or (
            f"Convert external skill {record.name} to native ZHVUSHA skill"
        )
        spec_draft = NativeSkillConversionSpecGenerator().generate(record)
        return SkillResult(
            success=True,
            response=(
                f"External skill `{record.skill_id}` marked as candidate for "
                f"native Жвуша skill spec: {title}. Generated draft: "
                f"{spec_draft.filename}. Код не генерирую: дальше нужен "
                "отдельный spec-first self-coding flow."
            ),
            metadata={
                "skill_name": self.name,
                "external_skill_id": record.skill_id,
                "external_skill_status": record.status.value,
                "native_conversion_spec_title": title,
                "native_conversion_spec_filename": spec_draft.filename,
                "native_conversion_spec_yaml": spec_draft.yaml_text,
                "native_conversion_migration_note": spec_draft.migration_note_markdown,
                "dialogue_state_patch": {
                    "selected_skill": self.name,
                    "last_result": "external_skill_marked_native_conversion",
                    "source": "external_skill_acquisition.mark_native",
                },
            },
        )


def _parse_command(message: str) -> _AcquisitionCommand:
    text = message.strip()
    if text.startswith(SEARCH_PREFIX):
        return _parse_search_command(text)
    if text.startswith(IMPORT_PREFIX):
        return _parse_skill_id_command(
            text,
            prefix=IMPORT_PREFIX,
            action="import",
            format_hint="Формат: /external_skill_import <candidate_id>",
            target_field="candidate_id",
        )
    if text.startswith(APPROVE_READONLY_PREFIX):
        return _parse_skill_id_command(
            text,
            prefix=APPROVE_READONLY_PREFIX,
            action="approve_readonly",
            format_hint="Формат: /external_skill_approve_readonly <skill_id>",
        )
    if text.startswith(APPROVE_EXECUTION_PREFIX):
        return _parse_execution_approval_command(text)
    if text.startswith(MARK_NATIVE_PREFIX):
        return _parse_skill_id_command(
            text,
            prefix=MARK_NATIVE_PREFIX,
            action="mark_native",
            format_hint="Формат: /external_skill_mark_native <skill_id>",
        )
    raise ValueError("Неизвестная external skill acquisition команда.")


def _parse_search_command(text: str) -> _AcquisitionCommand:
    rest = text.removeprefix(SEARCH_PREFIX).strip()
    parts = [part.strip() for part in rest.split("|", maxsplit=2)]
    raw_capabilities = parts[0] if parts else ""
    user_request = parts[1] if len(parts) >= 2 else ""
    source_directive = parts[2] if len(parts) >= 3 else ""
    capabilities = _parse_capabilities(raw_capabilities)
    if not capabilities or not user_request:
        raise ValueError(
            "Формат: /external_skill_search <capability[,capability]> | "
            "<запрос> [| local_folder|agentskills.io|all]"
        )
    return _AcquisitionCommand(
        action="search",
        requested_capabilities=capabilities,
        user_request=user_request,
        allowed_sources=_parse_allowed_sources(source_directive),
    )


def _parse_skill_id_command(
    text: str,
    *,
    prefix: str,
    action: Literal["import", "approve_readonly", "mark_native"],
    format_hint: str,
    target_field: Literal["skill_id", "candidate_id"] = "skill_id",
) -> _AcquisitionCommand:
    identifier = text.removeprefix(prefix).strip()
    if not identifier:
        raise ValueError(format_hint)
    if target_field == "candidate_id":
        return _AcquisitionCommand(action=action, candidate_id=identifier)
    return _AcquisitionCommand(action=action, skill_id=identifier)


def _parse_execution_approval_command(text: str) -> _AcquisitionCommand:
    rest = text.removeprefix(APPROVE_EXECUTION_PREFIX).strip()
    skill_id, raw_capabilities = _split_once(rest, "|")
    capabilities = _parse_capabilities(raw_capabilities)
    if not skill_id or not capabilities:
        raise ValueError(
            "Формат: /external_skill_approve_execution "
            "<skill_id> | <capability[,capability]>"
        )
    return _AcquisitionCommand(
        action="approve_execution",
        skill_id=skill_id,
        requested_capabilities=capabilities,
    )


def _command_from_gap_report(report: CapabilityGapReport) -> _AcquisitionCommand:
    proposal = report.proposal
    if proposal is None:
        raise ValueError("Capability gap report has no acquisition proposal.")
    return _AcquisitionCommand(
        action="search",
        requested_capabilities=proposal.requested_capabilities,
        user_request=proposal.user_request,
        allowed_sources=proposal.allowed_sources,
    )


def _plan_from_command(
    command: _AcquisitionCommand,
    *,
    import_candidate_source: str = "",
) -> ExecutionPlan:
    if command.action == "search":
        summary = (
            "Искать external skill candidates для capabilities: "
            f"{', '.join(command.requested_capabilities)} "
            f"в источниках: {', '.join(command.allowed_sources)}"
        )
        side_effects = _search_side_effects(command.allowed_sources)
        metadata: dict[str, object] = {
            "external_skill_acquisition_action": "search",
            "requested_capabilities": command.requested_capabilities,
            "external_skill_sources": command.allowed_sources,
        }
    elif command.action == "import":
        summary = f"Импортировать external skill candidate `{command.candidate_id}` в quarantine"
        side_effects = [SideEffect.WRITES_FILESYSTEM]
        if import_candidate_source == "agentskills.io":
            side_effects.append(SideEffect.NETWORK_IO_EXTERNAL)
        metadata = {
            "external_skill_acquisition_action": "import",
            "external_skill_candidate_id": command.candidate_id,
            "external_skill_candidate_source": import_candidate_source,
        }
    elif command.action == "approve_readonly":
        summary = f"Разрешить read-only external skill `{command.skill_id}`"
        side_effects = [SideEffect.WRITES_FILESYSTEM]
        metadata = {
            "external_skill_acquisition_action": "approve_readonly",
            "external_skill_id": command.skill_id,
        }
    elif command.action == "approve_execution":
        summary = (
            f"Разрешить execution external skill `{command.skill_id}` "
            f"для capabilities: {', '.join(command.requested_capabilities)}"
        )
        side_effects = [SideEffect.WRITES_FILESYSTEM]
        metadata = {
            "external_skill_acquisition_action": "approve_execution",
            "external_skill_id": command.skill_id,
            "approved_capabilities": command.requested_capabilities,
        }
    else:
        summary = (
            f"Отметить external skill `{command.skill_id}` как candidate "
            "для spec-first native conversion"
        )
        side_effects = [SideEffect.WRITES_FILESYSTEM]
        metadata = {
            "external_skill_acquisition_action": "mark_native",
            "external_skill_id": command.skill_id,
        }
    return ExecutionPlan(
        skill_name=ExternalSkillAcquisitionSkill.name,
        skill_type="inline",
        human_summary=summary,
        estimated_tokens=200,
        estimated_cost_usd=Decimal("0"),
        estimated_duration_seconds=1.0,
        side_effects_invoked=side_effects,
        metadata=metadata,
    )


def _missing_input_plan(reason: str) -> ExecutionPlan:
    return ExecutionPlan(
        skill_name=ExternalSkillAcquisitionSkill.name,
        skill_type="inline",
        human_summary=reason,
        estimated_tokens=100,
        estimated_cost_usd=Decimal("0"),
        estimated_duration_seconds=0.0,
        metadata={
            "requires_user_input": True,
            "missing_fields": ("external_skill_acquisition_command",),
        },
    )


def _parse_capabilities(raw: str) -> tuple[str, ...]:
    return tuple(
        item
        for item in (token.strip() for token in raw.replace(",", " ").split())
        if item
    )


def _parse_allowed_sources(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if not text:
        return ("local_folder",)
    if text.startswith("source="):
        text = text.removeprefix("source=").strip()
    sources: list[str] = []
    for token in text.replace(",", " ").split():
        normalized = token.strip().casefold().replace("-", "_")
        if normalized in {"all", "*"}:
            sources.extend(("local_folder", "agentskills.io"))
        elif normalized in {"local", "local_folder", "folder"}:
            sources.append("local_folder")
        elif normalized in {"agentskills", "agentskills.io", "remote"}:
            sources.append("agentskills.io")
        else:
            raise ValueError("source должен быть local_folder, agentskills.io или all")
    return tuple(dict.fromkeys(sources))


def _search_side_effects(allowed_sources: tuple[str, ...]) -> list[SideEffect]:
    side_effects: list[SideEffect] = []
    if "local_folder" in set(allowed_sources):
        side_effects.append(SideEffect.READS_FILESYSTEM)
    if "agentskills.io" in set(allowed_sources):
        side_effects.append(SideEffect.NETWORK_IO_EXTERNAL)
    return side_effects


def _find_candidate(
    candidates: tuple[ExternalSkillCandidate, ...],
    candidate_id: str,
) -> ExternalSkillCandidate | None:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def _proposal_for_import(
    proposal: ExternalSkillAcquisitionProposal,
    *,
    candidate: ExternalSkillCandidate,
    approval_id: str,
    approved_by_user_id: int,
) -> ExternalSkillAcquisitionProposal:
    return proposal.model_copy(
        update={
            "acquisition_approval_id": approval_id,
            "approved_by_user_id": approved_by_user_id,
            "grants_network_fetch": candidate.source_type == "agentskills.io",
            "grants_import": True,
            "grants_readonly_use": False,
            "grants_execution": False,
        }
    )


def _parse_gap_intent_json(text: str) -> ExternalSkillGapIntent:
    try:
        data = json.loads(_extract_json_object(text))
    except (json.JSONDecodeError, ValueError, TypeError):
        return ExternalSkillGapIntent()
    if not isinstance(data, dict):
        return ExternalSkillGapIntent()
    capabilities = data.get("required_capabilities", ())
    if isinstance(capabilities, str):
        raw_capabilities: tuple[object, ...] = tuple(capabilities.split(","))
    elif isinstance(capabilities, list | tuple):
        raw_capabilities = tuple(capabilities)
    else:
        raw_capabilities = ()
    clean_capabilities = tuple(
        capability
        for capability in (
            str(raw_capability).strip().casefold().replace("-", "_")
            for raw_capability in raw_capabilities
        )
        if capability
    )
    return ExternalSkillGapIntent(
        should_acquire_skill=bool(data.get("should_acquire_skill", False)),
        required_capabilities=clean_capabilities,
        confidence=_coerce_confidence(data.get("confidence", 0.0)),
        reason=str(data.get("reason") or "").strip(),
    )


def _capability_hints_from_graph(
    *,
    message: str,
    capability_graph: CapabilityGraph,
) -> tuple[str, ...]:
    """Infer missing capability ids from CapabilityGraph labels as a fallback."""
    lower = message.casefold()
    if not any(marker in lower for marker in _GAP_ACTION_MARKERS):
        return ()
    message_tokens = set(re.findall(r"[a-zа-яё0-9]+", lower))
    if not message_tokens:
        return ()

    hints: list[str] = []
    unavailable_statuses = {
        "configured_only",
        "degraded",
        "disabled",
        "orphaned",
        "quarantined",
        "needs_review",
        "blocked",
    }
    for node in capability_graph.capabilities:
        if (
            node.kind.value != "agent_capability"
            or node.status.value not in unavailable_statuses
            or not node.capability_id
        ):
            continue
        capability_tokens = {
            token
            for token in re.split(r"[^a-zа-яё0-9]+", node.capability_id.casefold())
            if len(token) >= 4 and token not in _GENERIC_CAPABILITY_TOKENS
        }
        label_tokens = {
            token
            for token in re.split(r"[^a-zа-яё0-9]+", node.label.casefold())
            if len(token) >= 4 and token not in _GENERIC_CAPABILITY_TOKENS
        }
        if message_tokens & (capability_tokens | label_tokens):
            hints.append(node.capability_id)

    return tuple(dict.fromkeys(hints))


_EXTERNAL_SKILL_TERMS = (
    r"external[_\s-]*skills?",
    r"skills?",
    r"скилл\w*",
    r"навык\w*",
)
_EXTERNAL_SKILL_TERM_PATTERN = r"(?:{})".format("|".join(_EXTERNAL_SKILL_TERMS))
_EXTERNAL_SKILL_OPT_OUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b(?:не|не\s+надо|no|do\s+not|don't)\s+"
        rf"(?:запускай|используй|ищи|импортируй|подключай|вызывай|run|use|search|import|call)"
        rf"[\s\S]{{0,120}}\b{_EXTERNAL_SKILL_TERM_PATTERN}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:без|without|no)\s+{_EXTERNAL_SKILL_TERM_PATTERN}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:не|no|do\s+not|don't)\s+"
        r"(?:external_skill_search|external_skill_import|external_skill_acquisition)\b",
        re.IGNORECASE,
    ),
)
_CODEX_GOAL_HANDOFF_MARKERS = (
    "codex/operator handoff",
    "local goal-supervisor",
    "active goal",
    "recent runner context",
    "agent runtime job evidence",
    "runner_action",
)


def _looks_like_codex_goal_handoff(message: str, context: AgentContext) -> bool:
    """Keep structured Codex goal packets in Жвуша's cognitive loop."""
    if str(context.metadata.get("source_actor", "") or "").casefold() != "codex":
        return False
    lowered = message.casefold()
    return sum(marker in lowered for marker in _CODEX_GOAL_HANDOFF_MARKERS) >= 2


def _explicitly_disallows_external_skill_acquisition(message: str) -> bool:
    """Honor explicit normal-chat opt-outs before running acquisition routing."""
    return any(pattern.search(message) for pattern in _EXTERNAL_SKILL_OPT_OUT_PATTERNS)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return stripped[start : end + 1]


def _coerce_confidence(value: object) -> float:
    if not isinstance(value, int | float | str) or isinstance(value, bool):
        return 0.0
    try:
        confidence = float(value)
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, confidence))


def _split_once(value: str, delimiter: str) -> tuple[str, str]:
    if delimiter not in value:
        return value.strip(), ""
    left, right = value.split(delimiter, maxsplit=1)
    return left.strip(), right.strip()


def _context_key(context: AgentContext) -> tuple[int, int | None, str]:
    return (context.user_id, context.chat_id, context.mode)


def _message_key(
    context: AgentContext,
    message: str,
) -> tuple[int, int | None, str, str]:
    return (context.user_id, context.chat_id, context.mode, message)


def _fetch_agentskills_catalog(url: str) -> str:
    import httpx

    response = httpx.get(
        url,
        headers={"User-Agent": "ZHVUSHA external skill acquisition"},
        timeout=20.0,
        follow_redirects=False,
    )
    response.raise_for_status()
    payload = response.text
    if len(payload.encode("utf-8")) > 1_000_000:
        raise ValueError("agentskills.io catalog is too large")
    return payload


def _fetch_agentskills_archive(url: str) -> bytes:
    import httpx

    chunks: list[bytes] = []
    total = 0
    with (
        httpx.Client(timeout=30.0, follow_redirects=False) as client,
        client.stream(
            "GET",
            url,
            headers={"User-Agent": "ZHVUSHA external skill acquisition"},
        ) as response,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > MAX_AGENTSKILLS_ARCHIVE_BYTES:
                raise ValueError("agentskills.io skill archive is too large")
            chunks.append(chunk)
    return b"".join(chunks)


_GAP_INTENT_SYSTEM = """\
Ты классифицируешь личные сообщения Никиты для Жвуши.

Определи, просит ли сообщение задачу, для которой Жвуше может не хватать
capability и стоит предложить безопасный поиск external Hermes/agentskills.io
compatible skill. Не предлагай acquisition для обычной беседы, вопросов,
социальных сообщений, простых объяснений или задач, которые не требуют нового
workflow/tool skill.

Верни только JSON:
{
  "should_acquire_skill": true|false,
  "required_capabilities": ["capability_id"],
  "confidence": 0.0-1.0,
  "reason": "short"
}

Используй короткие snake_case capability ids. Примеры:
- kubernetes_debug
- browser_read
- browser_screenshot
- browser_download
- web_search_sources
- run_readonly_commands
- read_workspace
- telegram_mcp_read
- telegram_mcp_send

Если не уверен, верни should_acquire_skill=false.
"""
