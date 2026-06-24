"""Capability gap and external skill acquisition proposal contracts."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path


def test_gap_detector_proposes_acquisition_without_expanding_tool_surface() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.skills.external_skill_loader.acquisition import (
        CapabilityGapDetector,
        ExternalSkillAcquisitionProposalStatus,
    )

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_capability.web_research.readonly.browser_read",
                label="browser_read",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.AVAILABLE,
                capability_id="browser_read",
            ),
            CapabilityNode(
                id="agent_capability.kubernetes.debug",
                label="kubernetes_debug",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.DISABLED,
                reason="no native Kubernetes debug skill installed",
                capability_id="kubernetes_debug",
            ),
        )
    )

    report = CapabilityGapDetector().detect(
        user_request="Проверь Kubernetes ingress и найди проблему.",
        required_capabilities=("kubernetes_debug", "browser_read"),
        capability_graph=graph,
        allowed_sources=("local_folder", "agentskills.io"),
    )

    assert report.has_gap is True
    assert report.available_capabilities == ("browser_read",)
    assert report.missing_capabilities == ("kubernetes_debug",)
    assert report.proposal is not None
    assert report.proposal.status is ExternalSkillAcquisitionProposalStatus.PROPOSED
    assert report.proposal.requested_capabilities == ("kubernetes_debug",)
    assert report.proposal.allowed_sources == ("local_folder", "agentskills.io")
    assert report.proposal.grants_execution is False
    assert report.proposal.grants_network_fetch is False
    assert "не скачиваю" in report.proposal.render_for_chat()
    assert "не выполняю" in report.proposal.render_for_chat()


def test_gap_detector_uses_existing_degraded_surface_before_skill_acquisition() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.skills.external_skill_loader.acquisition import CapabilityGapDetector

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_capability.web_research.readonly.browser_read",
                label="browser_read",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.DEGRADED,
                reason="browser provider missing",
                capability_id="browser_read",
            ),
        )
    )

    report = CapabilityGapDetector().detect(
        user_request="Открой сайт и найди X.",
        required_capabilities=("browser_read",),
        capability_graph=graph,
    )

    assert report.has_gap is True
    assert report.missing_capabilities == ()
    assert report.degraded_capabilities == ("browser_read",)
    assert report.proposal is None
    assert report.next_actions == (
        "Восстановить degraded capability browser_read: browser provider missing",
    )


def test_gap_detector_does_not_propose_when_existing_capability_is_available() -> None:
    from src.agent_runtime.capability_graph import (
        CapabilityGraph,
        CapabilityKind,
        CapabilityNode,
        CapabilityStatus,
    )
    from src.skills.external_skill_loader.acquisition import CapabilityGapDetector

    graph = CapabilityGraph(
        capabilities=(
            CapabilityNode(
                id="agent_capability.web_research.readonly.browser_read",
                label="browser_read",
                kind=CapabilityKind.AGENT_CAPABILITY,
                status=CapabilityStatus.AVAILABLE,
                capability_id="browser_read",
            ),
        )
    )

    report = CapabilityGapDetector().detect(
        user_request="Прочитай ссылку.",
        required_capabilities=("browser_read",),
        capability_graph=graph,
    )

    assert report.has_gap is False
    assert report.proposal is None
    assert report.next_actions == (
        "Use existing capabilities; do not acquire a skill.",
    )


def test_acquisition_importer_refuses_import_before_acquisition_approval(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionImporter,
        ExternalSkillAcquisitionProposal,
    )
    from src.skills.external_skill_loader.loader import (
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
    )

    source_root = _write_external_skill_folder(tmp_path / "candidate")
    importer = ExternalSkillAcquisitionImporter(
        quarantine_store=FileExternalSkillQuarantineStore(tmp_path / "quarantine"),
        registry=FilePersonalSkillRegistry(tmp_path / "registry"),
    )
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Нужен Kubernetes debug skill.",
        requested_capabilities=("kubernetes_debug",),
        allowed_sources=("local_folder",),
    )

    try:
        importer.import_local_folder(proposal, source_root=source_root)
    except PermissionError as exc:
        assert "acquisition approval" in str(exc)
    else:
        raise AssertionError("local import must require acquisition approval")

    assert tuple((tmp_path / "registry").glob("*.json")) == ()


def test_acquisition_importer_quarantines_audits_and_registers_candidate(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionImporter,
        ExternalSkillAcquisitionProposal,
        ExternalSkillAcquisitionProposalStatus,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillStatus,
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
    )

    source_root = _write_external_skill_folder(tmp_path / "candidate")
    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Нужен Kubernetes debug skill.",
        requested_capabilities=("kubernetes_debug",),
        allowed_sources=("local_folder",),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_import_to_quarantine=True,
    )
    importer = ExternalSkillAcquisitionImporter(
        quarantine_store=FileExternalSkillQuarantineStore(tmp_path / "quarantine"),
        registry=registry,
    )

    result = importer.import_local_folder(proposal, source_root=source_root)
    rendered = result.render_audit_for_chat()

    assert result.proposal.status is (
        ExternalSkillAcquisitionProposalStatus.IMPORTED_TO_QUARANTINE
    )
    assert result.quarantined.source.acquisition_approval_id == "approval-acquire"
    assert result.registry_record.status is ExternalSkillStatus.NEEDS_REVIEW
    assert result.audit_report.read_only_allowed is True
    assert result.audit_report.execution_allowed is False
    assert registry.active_records() == ()
    assert "quarantine" in rendered
    assert "read-only approval" in rendered
    assert "execution approval" in rendered
    assert "не активирован" in rendered


def test_candidate_search_refuses_before_acquisition_approval(tmp_path: Path) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidateSearchService,
        LocalFolderExternalSkillSourceProvider,
    )

    catalog_root = tmp_path / "catalog"
    _write_external_skill_folder(catalog_root / "kube-debug")
    service = ExternalSkillCandidateSearchService(
        providers=(LocalFolderExternalSkillSourceProvider(root=catalog_root),)
    )
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Нужен Kubernetes debug skill.",
        requested_capabilities=("browser_read",),
        allowed_sources=("local_folder",),
    )

    try:
        service.search(proposal)
    except PermissionError as exc:
        assert "search approval" in str(exc)
    else:
        raise AssertionError("candidate search must require acquisition approval")


def test_candidate_search_lists_local_folder_candidates_after_approval(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidateSearchService,
        LocalFolderExternalSkillSourceProvider,
    )

    catalog_root = tmp_path / "catalog"
    _write_external_skill_folder(catalog_root / "kube-debug")
    (catalog_root / "not-a-skill").mkdir(parents=True)
    service = ExternalSkillCandidateSearchService(
        providers=(LocalFolderExternalSkillSourceProvider(root=catalog_root),)
    )
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("browser_read",),
        allowed_sources=("local_folder",),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_import_to_quarantine=True,
    )

    report = service.search(proposal)
    rendered = report.render_for_chat()

    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.name == "kube-debug"
    assert candidate.source_type == "local_folder"
    assert candidate.matched_capabilities == ("browser_read",)
    assert candidate.locator.endswith("catalog/kube-debug")
    assert "не активирован" in rendered
    assert "Candidate 1" in rendered


def test_candidate_search_keeps_local_candidates_when_remote_source_fails(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidateSearchService,
        LocalFolderExternalSkillSourceProvider,
    )

    class FailingRemoteProvider:
        source_type = "agentskills.io"

        def search(self, proposal: ExternalSkillAcquisitionProposal):
            del proposal
            raise RuntimeError("catalog 404")

    catalog_root = tmp_path / "catalog"
    _write_external_skill_folder(catalog_root / "kube-debug")
    service = ExternalSkillCandidateSearchService(
        providers=(
            LocalFolderExternalSkillSourceProvider(root=catalog_root),
            FailingRemoteProvider(),
        )
    )
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("browser_read",),
        allowed_sources=("local_folder", "agentskills.io"),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_network_fetch=True,
        grants_import_to_quarantine=True,
    )

    report = service.search(proposal)

    assert len(report.candidates) == 1
    assert report.candidates[0].source_type == "local_folder"
    assert report.source_errors == ("agentskills.io: RuntimeError: catalog 404",)
    assert "agentskills.io" in report.render_for_chat()


def test_import_selected_candidate_uses_same_approved_proposal_and_keeps_inactive(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionImporter,
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidateSearchService,
        LocalFolderExternalSkillSourceProvider,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillStatus,
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
    )

    catalog_root = tmp_path / "catalog"
    _write_external_skill_folder(catalog_root / "kube-debug")
    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("browser_read",),
        allowed_sources=("local_folder",),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_import_to_quarantine=True,
    )
    service = ExternalSkillCandidateSearchService(
        providers=(LocalFolderExternalSkillSourceProvider(root=catalog_root),)
    )
    candidate = service.search(proposal).candidates[0]
    importer = ExternalSkillAcquisitionImporter(
        quarantine_store=FileExternalSkillQuarantineStore(tmp_path / "quarantine"),
        registry=registry,
    )

    result = service.import_candidate(
        proposal,
        candidate=candidate,
        importer=importer,
    )

    assert result.quarantined.source.acquisition_approval_id == "approval-acquire"
    assert result.registry_record.status is ExternalSkillStatus.NEEDS_REVIEW
    assert registry.active_records() == ()
    assert (tmp_path / "quarantine").exists()


def test_import_remote_agentskills_candidate_requires_network_and_import_grants(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionImporter,
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidate,
        ExternalSkillCandidateSearchService,
    )
    from src.skills.external_skill_loader.loader import (
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
    )

    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("browser_read",),
        allowed_sources=("agentskills.io",),
    ).approve_for_search(
        approval_id="approval-import",
        approved_by_user_id=1291112109,
        grants_network_fetch=True,
        grants_import_to_quarantine=False,
    )
    candidate = ExternalSkillCandidate(
        candidate_id="agentskills.io:remote",
        name="remote-kube",
        source_type="agentskills.io",
        locator="https://agentskills.io/skills/remote-kube.zip",
        requested_capabilities=("browser_read",),
        matched_capabilities=("browser_read",),
    )
    service = ExternalSkillCandidateSearchService(providers=())
    importer = ExternalSkillAcquisitionImporter(
        quarantine_store=FileExternalSkillQuarantineStore(tmp_path / "quarantine"),
        registry=FilePersonalSkillRegistry(tmp_path / "registry"),
    )

    try:
        service.import_candidate(
            proposal,
            candidate=candidate,
            importer=importer,
            fetch_remote_archive=lambda _url: _remote_skill_zip_bytes(),
        )
    except PermissionError as exc:
        assert "quarantine import" in str(exc)
    else:
        raise AssertionError("remote import must require import grant")

    assert tuple((tmp_path / "registry").glob("*.json")) == ()


def test_import_remote_agentskills_candidate_downloads_to_quarantine_and_registers(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionImporter,
        ExternalSkillAcquisitionProposal,
        ExternalSkillAcquisitionProposalStatus,
        ExternalSkillCandidate,
        ExternalSkillCandidateSearchService,
    )
    from src.skills.external_skill_loader.loader import (
        ExternalSkillStatus,
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
    )

    fetched_urls: list[str] = []

    def fetch_archive(url: str) -> bytes:
        fetched_urls.append(url)
        return _remote_skill_zip_bytes()

    registry = FilePersonalSkillRegistry(tmp_path / "registry")
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("browser_read",),
        allowed_sources=("agentskills.io",),
    ).approve_for_search(
        approval_id="approval-import",
        approved_by_user_id=1291112109,
        grants_network_fetch=True,
        grants_import_to_quarantine=True,
    )
    candidate = ExternalSkillCandidate(
        candidate_id="agentskills.io:remote",
        name="remote-kube",
        source_type="agentskills.io",
        locator="https://agentskills.io/skills/remote-kube.zip",
        requested_capabilities=("browser_read",),
        matched_capabilities=("browser_read",),
    )
    service = ExternalSkillCandidateSearchService(providers=())
    importer = ExternalSkillAcquisitionImporter(
        quarantine_store=FileExternalSkillQuarantineStore(tmp_path / "quarantine"),
        registry=registry,
    )

    result = service.import_candidate(
        proposal,
        candidate=candidate,
        importer=importer,
        fetch_remote_archive=fetch_archive,
    )

    assert fetched_urls == ["https://agentskills.io/skills/remote-kube.zip"]
    assert result.proposal.status is (
        ExternalSkillAcquisitionProposalStatus.IMPORTED_TO_QUARANTINE
    )
    assert result.quarantined.source.source_type == "agentskills.io"
    assert result.quarantined.source.locator == (
        "https://agentskills.io/skills/remote-kube.zip"
    )
    assert result.quarantined.source.acquisition_approval_id == "approval-import"
    assert Path(result.quarantined.quarantine_path).is_dir()
    assert result.registry_record.status is ExternalSkillStatus.NEEDS_REVIEW
    assert result.audit_report.read_only_allowed is True
    assert registry.active_records() == ()


def test_remote_agentskills_import_rejects_unsafe_archive_paths(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        ExternalSkillAcquisitionImporter,
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidate,
        ExternalSkillCandidateSearchService,
    )
    from src.skills.external_skill_loader.loader import (
        FileExternalSkillQuarantineStore,
        FilePersonalSkillRegistry,
    )

    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("browser_read",),
        allowed_sources=("agentskills.io",),
    ).approve_for_search(
        approval_id="approval-import",
        approved_by_user_id=1291112109,
        grants_network_fetch=True,
        grants_import_to_quarantine=True,
    )
    candidate = ExternalSkillCandidate(
        candidate_id="agentskills.io:remote",
        name="remote-kube",
        source_type="agentskills.io",
        locator="https://agentskills.io/skills/remote-kube.zip",
    )
    service = ExternalSkillCandidateSearchService(providers=())
    importer = ExternalSkillAcquisitionImporter(
        quarantine_store=FileExternalSkillQuarantineStore(tmp_path / "quarantine"),
        registry=FilePersonalSkillRegistry(tmp_path / "registry"),
    )

    try:
        service.import_candidate(
            proposal,
            candidate=candidate,
            importer=importer,
            fetch_remote_archive=lambda _url: _zip_bytes({"../escape.txt": "x"}),
        )
    except ValueError as exc:
        assert "unsafe archive path" in str(exc)
    else:
        raise AssertionError("remote archive path traversal must be rejected")

    assert tuple((tmp_path / "registry").glob("*.json")) == ()


def test_agentskills_catalog_search_refuses_without_network_fetch_grant() -> None:
    from src.skills.external_skill_loader.acquisition import (
        AgentskillsCatalogSourceProvider,
        ExternalSkillAcquisitionProposal,
    )

    fetched_urls: list[str] = []

    def fetch_catalog(url: str) -> str:
        fetched_urls.append(url)
        return "[]"

    proposal = ExternalSkillAcquisitionProposal(
        user_request="Нужен Kubernetes debug skill.",
        requested_capabilities=("kubernetes_debug",),
        allowed_sources=("agentskills.io",),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_network_fetch=False,
    )
    provider = AgentskillsCatalogSourceProvider(
        catalog_url="https://agentskills.io/catalog.json",
        fetch_catalog=fetch_catalog,
    )

    try:
        provider.search(proposal)
    except PermissionError as exc:
        assert "network fetch approval" in str(exc)
    else:
        raise AssertionError("agentskills.io search must require network grant")

    assert fetched_urls == []


def test_agentskills_catalog_search_lists_remote_candidates_without_side_effects(
    tmp_path: Path,
) -> None:
    from src.skills.external_skill_loader.acquisition import (
        AgentskillsCatalogSourceProvider,
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidateSearchService,
    )

    fetched_urls: list[str] = []
    catalog_payload = json.dumps(
        {
            "skills": [
                {
                    "name": "kube-debug",
                    "description": "Inspect Kubernetes ingress with browser evidence",
                    "url": "https://agentskills.io/skills/kube-debug.zip",
                    "capabilities": ["kubernetes_debug"],
                    "tools": ["browser_read"],
                },
                {
                    "name": "unrelated",
                    "description": "Design greeting cards",
                    "url": "https://agentskills.io/skills/cards.zip",
                    "capabilities": ["image_generation"],
                    "tools": [],
                },
            ]
        }
    )

    def fetch_catalog(url: str) -> str:
        fetched_urls.append(url)
        return catalog_payload

    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("kubernetes_debug", "browser_read"),
        allowed_sources=("agentskills.io",),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_network_fetch=True,
        grants_import_to_quarantine=False,
    )
    service = ExternalSkillCandidateSearchService(
        providers=(
            AgentskillsCatalogSourceProvider(
                catalog_url="https://agentskills.io/catalog.json",
                fetch_catalog=fetch_catalog,
            ),
        )
    )

    report = service.search(proposal)

    assert fetched_urls == ["https://agentskills.io/catalog.json"]
    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.name == "kube-debug"
    assert candidate.source_type == "agentskills.io"
    assert candidate.locator == "https://agentskills.io/skills/kube-debug.zip"
    assert candidate.requested_capabilities == (
        "browser_read",
        "kubernetes_debug",
    )
    assert candidate.matched_capabilities == (
        "browser_read",
        "kubernetes_debug",
    )
    assert candidate.score == 1.0
    assert tuple((tmp_path / "registry").glob("*.json")) == ()
    assert not (tmp_path / "quarantine").exists()


def test_agentskills_catalog_search_ignores_unsafe_or_invalid_remote_entries() -> None:
    from src.skills.external_skill_loader.acquisition import (
        AgentskillsCatalogSourceProvider,
        ExternalSkillAcquisitionProposal,
    )

    catalog_payload = json.dumps(
        [
            {
                "name": "localhost",
                "description": "Unsafe locator",
                "url": "http://127.0.0.1/skill.zip",
                "capabilities": ["browser_read"],
            },
            {
                "name": "file",
                "description": "Unsafe scheme",
                "url": "file:///tmp/skill.zip",
                "capabilities": ["browser_read"],
            },
            {
                "description": "Missing name",
                "url": "https://agentskills.io/skills/missing-name.zip",
                "capabilities": ["browser_read"],
            },
        ]
    )
    proposal = ExternalSkillAcquisitionProposal(
        user_request="Открой сайт.",
        requested_capabilities=("browser_read",),
        allowed_sources=("agentskills.io",),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_network_fetch=True,
    )
    provider = AgentskillsCatalogSourceProvider(
        catalog_url="https://agentskills.io/catalog.json",
        fetch_catalog=lambda _url: catalog_payload,
    )

    assert provider.search(proposal) == ()


def test_fetch_agentskills_archive_streams_chunks(monkeypatch) -> None:
    import src.skills.external_skill_acquisition.skill as skill_module

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self):
            yield b"abc"
            yield b"def"

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            headers: dict[str, str],
        ) -> FakeResponse:
            assert method == "GET"
            assert url == "https://agentskills.io/skills/demo.zip"
            assert headers["User-Agent"].startswith("ZHVUSHA")
            return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        type("FakeHttpx", (), {"Client": FakeClient}),
    )

    payload = skill_module._fetch_agentskills_archive(
        "https://agentskills.io/skills/demo.zip"
    )

    assert payload == b"abcdef"


def test_candidate_search_does_not_call_disallowed_agentskills_provider() -> None:
    from src.skills.external_skill_loader.acquisition import (
        AgentskillsCatalogSourceProvider,
        ExternalSkillAcquisitionProposal,
        ExternalSkillCandidateSearchService,
    )

    fetched_urls: list[str] = []

    def fetch_catalog(url: str) -> str:
        fetched_urls.append(url)
        return "[]"

    proposal = ExternalSkillAcquisitionProposal(
        user_request="Проверь Kubernetes ingress.",
        requested_capabilities=("kubernetes_debug",),
        allowed_sources=("local_folder",),
    ).approve_for_search(
        approval_id="approval-acquire",
        approved_by_user_id=1291112109,
        grants_network_fetch=True,
    )
    service = ExternalSkillCandidateSearchService(
        providers=(
            AgentskillsCatalogSourceProvider(
                catalog_url="https://agentskills.io/catalog.json",
                fetch_catalog=fetch_catalog,
            ),
        )
    )

    report = service.search(proposal)

    assert report.candidates == ()
    assert fetched_urls == []


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


def _remote_skill_zip_bytes() -> bytes:
    return _zip_bytes(
        {
            "remote-kube/SKILL.md": "\n".join(
                [
                    "---",
                    "name: remote-kube",
                    "description: Remote Kubernetes ingress helper",
                    "tools: [browser_read]",
                    "---",
                    "# Remote Kubernetes ingress debug",
                    "Inspect manifests and collect evidence.",
                ]
            )
        }
    )


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
