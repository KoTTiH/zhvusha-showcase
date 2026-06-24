"""External skill acquisition skill contracts."""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

import pytest
import yaml
from src.skills.base import AgentContext
from src.skills.invocation import (
    ApprovalVerdict,
    InMemorySkillApprovalStore,
    SkillInvocationService,
)


def _ctx() -> AgentContext:
    return AgentContext(user_id=12345, chat_id=12345, mode="personal", message_id=7)


def _service(*verdicts: ApprovalVerdict) -> SkillInvocationService:
    queued_verdicts = list(verdicts or ("yes",))

    async def classify_approval(text: str) -> ApprovalVerdict:
        del text
        if queued_verdicts:
            return queued_verdicts.pop(0)
        return "yes"

    return SkillInvocationService(
        approval_store=InMemorySkillApprovalStore(),
        approval_classifier=classify_approval,
        is_skill_allowed=lambda _name, _mode: True,
    )


class _GapClassifier:
    def __init__(self, *, should_acquire: bool, capabilities: tuple[str, ...]) -> None:
        self.calls: list[str] = []
        self._should_acquire = should_acquire
        self._capabilities = capabilities

    async def classify(self, message: str):  # type: ignore[no-untyped-def]
        from src.skills.external_skill_acquisition.skill import ExternalSkillGapIntent

        self.calls.append(message)
        return ExternalSkillGapIntent(
            should_acquire_skill=self._should_acquire,
            required_capabilities=self._capabilities,
            confidence=0.9,
            reason="test",
        )


class _HangingGapClassifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def classify(self, message: str):  # type: ignore[no-untyped-def]
        self.calls.append(message)
        await asyncio.sleep(60.0)
        from src.skills.external_skill_acquisition.skill import ExternalSkillGapIntent

        return ExternalSkillGapIntent()


def _capability_graph_for_gap():
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )

    return CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_capability.browser_read",
                label="browser_read",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.AVAILABLE,
                capability_id="browser_read",
            ),
            CapabilityNode(
                id="agent_capability.kubernetes_debug",
                label="kubernetes_debug",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.DISABLED,
                reason="no native Kubernetes debug skill",
                capability_id="kubernetes_debug",
            ),
        )
    )


def _remote_skill_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "remote-kube/SKILL.md",
            "\n".join(
                [
                    "---",
                    "name: remote-kube",
                    "description: Remote Kubernetes ingress helper",
                    "tools: [browser_read]",
                    "---",
                    "# Remote Kubernetes ingress debug",
                    "Inspect manifests and collect evidence.",
                ]
            ),
        )
    return buffer.getvalue()


def test_manifest_matches_class() -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )
    from src.skills.manifest import (
        load_manifest_for_skill_class,
        validate_manifest_matches_class,
    )

    manifest = load_manifest_for_skill_class(ExternalSkillAcquisitionSkill)
    validate_manifest_matches_class(manifest, ExternalSkillAcquisitionSkill)


@pytest.mark.asyncio
async def test_search_command_requires_approval_before_candidate_lookup(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    _write_external_skill_folder(tmp_path / "catalog" / "kube-debug")
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
    )
    service = _service("yes")

    pending = await service.dispatch(
        "/external_skill_search browser_read | Проверь Kubernetes ingress",
        _ctx(),
        [skill],
    )
    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert "Нужно решение" in pending.result.response
    assert (
        "network_io_external"
        not in pending.result.metadata["pending_decision"]["proposal"]["side_effects"]
    )
    assert not (tmp_path / "registry").exists()

    completed = await service.dispatch("да", _ctx(), [skill])

    assert completed.result is not None
    assert completed.result.success is True
    assert "Candidate 1: kube-debug" in completed.result.response
    assert not tuple((tmp_path / "registry").glob("*.json"))


@pytest.mark.asyncio
async def test_agentskills_search_requires_network_approval_and_keeps_inactive(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    fetched_urls: list[str] = []

    def fetch_catalog(url: str) -> str:
        fetched_urls.append(url)
        return (
            '{"skills":[{"name":"remote-kube","description":"Kubernetes ingress",'
            '"url":"https://agentskills.io/skills/remote-kube.zip",'
            '"capabilities":["kubernetes_debug"],"tools":["browser_read"]}]}'
        )

    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "missing-catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="https://agentskills.io/catalog.json",
        fetch_agentskills_catalog=fetch_catalog,
    )
    service = _service("yes")

    pending = await service.dispatch(
        "/external_skill_search browser_read,kubernetes_debug | "
        "Проверь Kubernetes ingress | agentskills.io",
        _ctx(),
        [skill],
    )

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert (
        "network_io_external"
        in pending.result.metadata["pending_decision"]["proposal"]["side_effects"]
    )
    assert fetched_urls == []

    completed = await service.dispatch("да", _ctx(), [skill])

    assert completed.result is not None
    assert completed.result.success is True
    assert fetched_urls == ["https://agentskills.io/catalog.json"]
    assert "Candidate 1: remote-kube" in completed.result.response
    assert (
        completed.result.metadata["external_skill_candidates"][0]["source_type"]
        == "agentskills.io"
    )
    assert not (tmp_path / "registry").exists()
    assert not (tmp_path / "quarantine").exists()


@pytest.mark.asyncio
async def test_normal_chat_capability_gap_requests_search_approval(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    (tmp_path / "catalog").mkdir()
    fetched_urls: list[str] = []

    def fetch_catalog(url: str) -> str:
        fetched_urls.append(url)
        return (
            '{"skills":[{"name":"remote-kube","description":"Kubernetes ingress",'
            '"url":"https://agentskills.io/skills/remote-kube.zip",'
            '"capabilities":["kubernetes_debug"],"tools":["browser_read"]}]}'
        )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("kubernetes_debug", "browser_read"),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="https://agentskills.io/catalog.json",
        fetch_agentskills_catalog=fetch_catalog,
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())
    service = _service("yes")
    ctx = _ctx()

    pending = await service.dispatch(
        "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
        ctx,
        [skill],
    )

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert pending.result.metadata["requested_capabilities"] == ("kubernetes_debug",)
    assert (
        "network_io_external"
        in pending.result.metadata["pending_decision"]["proposal"]["side_effects"]
    )
    assert fetched_urls == []

    completed = await service.dispatch("да", ctx, [skill])

    assert completed.result is not None
    assert completed.result.success is True
    assert fetched_urls == ["https://agentskills.io/catalog.json"]
    assert "Candidate 1: remote-kube" in completed.result.response
    assert not (tmp_path / "registry").exists()
    assert classifier.calls == [
        "Жвуша, проверь мой Kubernetes ingress и найди проблему."
    ]


@pytest.mark.asyncio
async def test_normal_chat_gap_can_request_approval_when_local_catalog_is_missing(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    fetched_urls: list[str] = []

    def fetch_catalog(url: str) -> str:
        fetched_urls.append(url)
        return '{"skills":[]}'

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("kubernetes_debug",),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "missing-catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="https://agentskills.io/catalog.json",
        fetch_agentskills_catalog=fetch_catalog,
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    pending = await _service("yes").dispatch(
        "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
        _ctx(),
        [skill],
    )

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert pending.result.metadata["external_skill_sources"] == (
        "local_folder",
        "agentskills.io",
    )
    assert fetched_urls == []


@pytest.mark.asyncio
async def test_direct_address_technical_gap_selects_acquisition_not_codebase(
    tmp_path: Path,
) -> None:
    from unittest.mock import AsyncMock

    from src.skills.codebase_explorer.skill import CodebaseExplorerSkill
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("kubernetes_debug",),
    )
    acquisition = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="https://agentskills.io/catalog.json",
        gap_classifier=classifier,
    )
    acquisition.set_capability_graph(_capability_graph_for_gap())
    explorer_runner = AsyncMock(return_value="codebase should not run")
    codebase = CodebaseExplorerSkill(
        admin_user_id=12345,
        workspace_root=tmp_path,
        explorer_runner=explorer_runner,
    )

    pending = await _service("yes").dispatch(
        "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
        _ctx(),
        [codebase, acquisition],
    )

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert pending.result.metadata["skill_name"] == "external_skill_acquisition"
    explorer_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_hint_fallback_detects_missed_technical_gap(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=False,
        capabilities=(),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="https://agentskills.io/catalog.json",
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    pending = await _service("yes").dispatch(
        "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
        _ctx(),
        [skill],
    )

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert pending.result.metadata["requested_capabilities"] == ("kubernetes_debug",)
    assert classifier.calls == [
        "Жвуша, проверь мой Kubernetes ingress и найди проблему."
    ]


@pytest.mark.asyncio
async def test_graph_hint_fallback_supplements_available_llm_capability(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("browser_read",),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="https://agentskills.io/catalog.json",
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    pending = await _service("yes").dispatch(
        "Жвуша, проверь мой Kubernetes ingress и найди проблему.",
        _ctx(),
        [skill],
    )

    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert pending.result.metadata["requested_capabilities"] == ("kubernetes_debug",)


@pytest.mark.asyncio
async def test_normal_chat_gap_does_not_intercept_available_capability(
    tmp_path: Path,
) -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("browser_read",),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="",
        gap_classifier=classifier,
    )
    skill.set_capability_graph(
        CapabilityGraph(
            capabilities=(
                CapabilityNode(
                    id="agent_capability.browser_read",
                    label="browser_read",
                    kind=CapabilityKind.AGENT_CAPABILITY,
                    status=CapabilityStatus.AVAILABLE,
                    capability_id="browser_read",
                ),
            )
        )
    )

    outcome = await _service().dispatch(
        "Жвуша, открой сайт и прочитай страницу.",
        _ctx(),
        [skill],
    )

    assert outcome.handled is False
    assert classifier.calls == ["Жвуша, открой сайт и прочитай страницу."]


@pytest.mark.asyncio
async def test_codex_author_marker_does_not_skip_external_skill_gap_classifier(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("kubernetes_debug",),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    score = await skill.can_handle(
        "нужен kubernetes debug",
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            message_id=7,
            metadata={"source_actor": "codex"},
        ),
    )

    assert score == 0.92
    assert classifier.calls == ["нужен kubernetes debug"]


@pytest.mark.asyncio
async def test_codex_goal_handoff_does_not_trigger_external_skill_gap_classifier(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("kubernetes_debug",),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    score = await skill.can_handle(
        (
            "Codex/operator handoff, sender=codex, не Никита.\n\n"
            "Я запускаю active goal как локальный goal-supervisor.\n\n"
            "Recent Runner Context:\n"
            "Agent Runtime Job Evidence:\n"
            "- job_id=job-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
            "status=failed error=agent job timed out\n\n"
            "capabilities: kubernetes_debug, life_stage_memory_candidate"
        ),
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            message_id=7,
            metadata={"source_actor": "codex"},
        ),
    )

    assert score == 0.0
    assert classifier.calls == []


@pytest.mark.asyncio
async def test_explicit_no_external_skill_boundary_skips_gap_classifier(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("agency_stage_memory", "self_approve_low_risk_specs"),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    score = await skill.can_handle(
        (
            "Проверь по уже доступным project read-only sources. "
            "Не запускай external skills, не ищи и не импортируй skills."
        ),
        AgentContext(
            user_id=12345,
            chat_id=-7331,
            mode="personal",
            message_id=7,
            metadata={"source_actor": "codex"},
        ),
    )

    assert score == 0.0
    assert classifier.calls == []


@pytest.mark.asyncio
async def test_fast_chat_preselection_does_not_run_external_skill_gap_classifier(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _GapClassifier(
        should_acquire=True,
        capabilities=("kubernetes_debug",),
    )
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        gap_classifier=classifier,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    score = await skill.can_handle(
        "как дела у тебя?",
        AgentContext(
            user_id=12345,
            chat_id=12345,
            mode="personal",
            message_id=7,
            metadata={"prefer_chat_response_only": True},
        ),
    )

    assert score == 0.0
    assert classifier.calls == []


@pytest.mark.asyncio
async def test_external_skill_gap_classifier_timeout_falls_back_to_chat(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    classifier = _HangingGapClassifier()
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        gap_classifier=classifier,
        gap_intent_timeout_seconds=0.01,
    )
    skill.set_capability_graph(_capability_graph_for_gap())

    score = await asyncio.wait_for(
        skill.can_handle("как дела у тебя?", _ctx()),
        timeout=0.5,
    )

    assert score == 0.0
    assert classifier.calls == ["как дела у тебя?"]


@pytest.mark.asyncio
async def test_import_command_requires_approval_and_keeps_skill_inactive(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    _write_external_skill_folder(tmp_path / "catalog" / "kube-debug")
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
    )
    service = _service("yes", "yes")
    ctx = _ctx()

    await service.dispatch(
        "/external_skill_search browser_read | Проверь Kubernetes ingress",
        ctx,
        [skill],
    )
    await service.dispatch("да", ctx, [skill])
    pending_import = await service.dispatch(
        "/external_skill_import local_folder:kube-debug",
        ctx,
        [skill],
    )

    assert pending_import.result is not None
    assert pending_import.result.metadata["approval_pending"] is True

    imported = await service.dispatch("да", ctx, [skill])

    assert imported.result is not None
    assert imported.result.success is True
    assert "External skill импортирован в quarantine" in imported.result.response
    assert "не активирован" in imported.result.response
    assert len(tuple((tmp_path / "registry").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_readonly_approval_command_requires_approval_and_activates_readonly(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
    )

    _write_external_skill_folder(tmp_path / "catalog" / "kube-debug")
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
    )
    service = _service("yes", "yes", "yes")
    ctx = _ctx()

    await service.dispatch(
        "/external_skill_search browser_read | Проверь Kubernetes ingress",
        ctx,
        [skill],
    )
    await service.dispatch("да", ctx, [skill])
    await service.dispatch(
        "/external_skill_import local_folder:kube-debug", ctx, [skill]
    )
    imported = await service.dispatch("да", ctx, [skill])
    assert imported.result is not None
    skill_id = imported.result.metadata["external_skill_id"]

    pending = await service.dispatch(
        f"/external_skill_approve_readonly {skill_id}",
        ctx,
        [skill],
    )
    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert (
        "writes_filesystem"
        in pending.result.metadata["pending_decision"]["proposal"]["side_effects"]
    )

    approved = await service.dispatch("да", ctx, [skill])

    assert approved.result is not None
    assert approved.result.success is True
    assert "read-only" in approved.result.response
    record = FilePersonalSkillRegistry(tmp_path / "registry").get(str(skill_id))
    assert record.status is ExternalSkillStatus.APPROVED_READONLY
    assert record.readonly_approval_id
    assert len(FilePersonalSkillRegistry(tmp_path / "registry").active_records()) == 1


@pytest.mark.asyncio
async def test_execution_approval_command_requires_readonly_and_scopes_capabilities(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
    )

    _write_external_skill_folder(tmp_path / "catalog" / "kube-debug")
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
    )
    service = _service("yes", "yes", "yes", "yes")
    ctx = _ctx()

    await service.dispatch(
        "/external_skill_search browser_read | Проверь Kubernetes ingress",
        ctx,
        [skill],
    )
    await service.dispatch("да", ctx, [skill])
    await service.dispatch(
        "/external_skill_import local_folder:kube-debug", ctx, [skill]
    )
    imported = await service.dispatch("да", ctx, [skill])
    assert imported.result is not None
    skill_id = imported.result.metadata["external_skill_id"]
    await service.dispatch(f"/external_skill_approve_readonly {skill_id}", ctx, [skill])
    await service.dispatch("да", ctx, [skill])

    pending = await service.dispatch(
        f"/external_skill_approve_execution {skill_id} | browser_read",
        ctx,
        [skill],
    )
    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True

    approved = await service.dispatch("да", ctx, [skill])

    assert approved.result is not None
    assert approved.result.success is True
    assert "ToolGateway" in approved.result.response
    record = FilePersonalSkillRegistry(tmp_path / "registry").get(str(skill_id))
    assert record.status is ExternalSkillStatus.EXECUTION_APPROVED
    assert record.approved_capabilities == ("browser_read",)
    assert record.execution_approval_id


@pytest.mark.asyncio
async def test_native_conversion_command_requires_approval_and_marks_candidate(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillStatus,
        FilePersonalSkillRegistry,
    )
    from src.skills.spec_command.parser import SpecModel

    _write_external_skill_folder(tmp_path / "catalog" / "kube-debug")
    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        native_conversion_minimum_uses=3,
    )
    service = _service("yes", "yes", "yes", "yes")
    ctx = _ctx()

    await service.dispatch(
        "/external_skill_search browser_read | Проверь Kubernetes ingress",
        ctx,
        [skill],
    )
    await service.dispatch("да", ctx, [skill])
    await service.dispatch(
        "/external_skill_import local_folder:kube-debug", ctx, [skill]
    )
    imported = await service.dispatch("да", ctx, [skill])
    assert imported.result is not None
    skill_id = str(imported.result.metadata["external_skill_id"])
    await service.dispatch(f"/external_skill_approve_readonly {skill_id}", ctx, [skill])
    await service.dispatch("да", ctx, [skill])

    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    registry.record_successful_use(skill_id)
    registry.record_successful_use(skill_id)
    registry.record_successful_use(skill_id)

    pending = await service.dispatch(
        f"/external_skill_mark_native {skill_id}",
        ctx,
        [skill],
    )
    assert pending.result is not None
    assert pending.result.metadata["approval_pending"] is True
    assert (
        "writes_filesystem"
        in pending.result.metadata["pending_decision"]["proposal"]["side_effects"]
    )

    converted = await service.dispatch("да", ctx, [skill])

    assert converted.result is not None
    assert converted.result.success is True
    assert "native Жвуша skill spec" in converted.result.response
    assert "draft:" in converted.result.response
    record = registry.get(skill_id)
    assert record.status is ExternalSkillStatus.NATIVE_CONVERSION_CANDIDATE
    assert record.native_conversion_approval_id
    assert record.native_conversion_reason
    spec_yaml = str(converted.result.metadata["native_conversion_spec_yaml"])
    spec = SpecModel.model_validate(yaml.safe_load(spec_yaml))
    assert spec.slug == "convert-external-skill-kube-debug"
    assert str(converted.result.metadata["native_conversion_spec_filename"]).endswith(
        "-convert-external-skill-kube-debug.yaml"
    )
    assert "approval-" not in spec_yaml
    assert "approval-" not in str(
        converted.result.metadata["native_conversion_migration_note"]
    )


@pytest.mark.asyncio
async def test_remote_candidate_import_downloads_to_quarantine_after_approval(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_acquisition.skill import (
        ExternalSkillAcquisitionSkill,
    )

    fetched_archives: list[str] = []

    def fetch_catalog(_url: str) -> str:
        return (
            '{"skills":[{"name":"remote-kube","description":"Kubernetes ingress",'
            '"url":"https://agentskills.io/skills/remote-kube.zip",'
            '"capabilities":["kubernetes_debug"],"tools":["browser_read"]}]}'
        )

    def fetch_archive(url: str) -> bytes:
        fetched_archives.append(url)
        return _remote_skill_zip_bytes()

    skill = ExternalSkillAcquisitionSkill(
        admin_user_id=12345,
        catalog_root=tmp_path / "missing-catalog",
        quarantine_root=tmp_path / "quarantine",
        registry_root=tmp_path / "registry",
        agentskills_catalog_url="https://agentskills.io/catalog.json",
        fetch_agentskills_catalog=fetch_catalog,
        fetch_agentskills_archive=fetch_archive,
    )
    service = _service("yes", "yes")
    ctx = _ctx()

    await service.dispatch(
        "/external_skill_search browser_read,kubernetes_debug | "
        "Проверь Kubernetes ingress | agentskills.io",
        ctx,
        [skill],
    )
    search_result = await service.dispatch("да", ctx, [skill])
    assert search_result.result is not None
    candidate_id = search_result.result.metadata["external_skill_candidates"][0][
        "candidate_id"
    ]
    pending_import = await service.dispatch(
        f"/external_skill_import {candidate_id}",
        ctx,
        [skill],
    )
    assert pending_import.result is not None
    assert pending_import.result.metadata["approval_pending"] is True
    assert (
        "network_io_external"
        in pending_import.result.metadata["pending_decision"]["proposal"][
            "side_effects"
        ]
    )
    assert (
        "writes_filesystem"
        in pending_import.result.metadata["pending_decision"]["proposal"][
            "side_effects"
        ]
    )
    assert fetched_archives == []

    imported = await service.dispatch("да", ctx, [skill])

    assert imported.result is not None
    assert imported.result.success is True
    assert "External skill импортирован в quarantine" in imported.result.response
    assert "не активирован" in imported.result.response
    assert fetched_archives == ["https://agentskills.io/skills/remote-kube.zip"]
    assert len(tuple((tmp_path / "registry").glob("*.json"))) == 1
    assert any((tmp_path / "quarantine").iterdir())


def _write_external_skill_folder(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: kube-debug",
                "description: Investigate Kubernetes ingress",
                "tools: [browser_read]",
                "---",
                "# Kubernetes ingress debug",
                "Inspect manifests and collect evidence before proposing fixes.",
            ]
        ),
        encoding="utf-8",
    )
    return root
