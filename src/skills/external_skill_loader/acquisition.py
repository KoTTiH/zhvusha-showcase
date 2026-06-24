"""Capability gap detection and scoped external skill acquisition proposals."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from enum import StrEnum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.agent_runtime.builtin_tools import is_public_http_url
from src.agent_runtime.capability_graph import (
    CapabilityGraph,
    CapabilityNode,
    CapabilityStatus,
)
from src.skills.external_skill_loader.loader import (
    CapabilityMapper,
    ExternalSkillAuditReport,
    ExternalSkillSource,
    FileExternalSkillQuarantineStore,
    FilePersonalSkillRegistry,
    PersonalSkillRegistryRecord,
    QuarantinedExternalSkill,
    audit_external_skill_package,
    parse_external_skill_folder,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    class ExternalSkillSourceProvider(Protocol):
        """Candidate source provider contract."""

        source_type: str

        def search(
            self,
            proposal: ExternalSkillAcquisitionProposal,
        ) -> tuple[ExternalSkillCandidate, ...]: ...


CatalogFetcher = Callable[[str], str]
ArchiveFetcher = Callable[[str], bytes]
MAX_REMOTE_ARCHIVE_BYTES = 10_000_000
MAX_REMOTE_UNCOMPRESSED_BYTES = 25_000_000
MAX_REMOTE_ARCHIVE_FILES = 500


class ExternalSkillAcquisitionProposalStatus(StrEnum):
    """Lifecycle state for a search/import proposal before any network action."""

    PROPOSED = "proposed"
    APPROVED_FOR_SEARCH = "approved_for_search"
    REJECTED = "rejected"
    IMPORTED_TO_QUARANTINE = "imported_to_quarantine"


class ExternalSkillAcquisitionProposal(BaseModel):
    """Permission ask for searching/importing an external skill candidate."""

    user_request: str
    requested_capabilities: tuple[str, ...]
    allowed_sources: tuple[str, ...] = ("local_folder", "agentskills.io")
    status: ExternalSkillAcquisitionProposalStatus = (
        ExternalSkillAcquisitionProposalStatus.PROPOSED
    )
    grants_network_fetch: bool = False
    grants_import: bool = False
    grants_readonly_use: bool = False
    grants_execution: bool = False
    audit_required: bool = True
    quarantine_required: bool = True
    acquisition_approval_id: str = ""
    approved_by_user_id: int | None = None
    approval_question: str = "Разрешаешь искать candidate skill для этого gap?"

    def approve_for_search(
        self,
        *,
        approval_id: str,
        approved_by_user_id: int,
        grants_network_fetch: bool = False,
        grants_import_to_quarantine: bool = True,
    ) -> ExternalSkillAcquisitionProposal:
        """Return a scoped acquisition approval without granting skill use."""
        return self.model_copy(
            update={
                "status": ExternalSkillAcquisitionProposalStatus.APPROVED_FOR_SEARCH,
                "acquisition_approval_id": approval_id,
                "approved_by_user_id": approved_by_user_id,
                "grants_network_fetch": grants_network_fetch,
                "grants_import": grants_import_to_quarantine,
                "grants_readonly_use": False,
                "grants_execution": False,
            }
        )

    def render_for_chat(self) -> str:
        """Render secret-free proposal text for Жвуша's chat response."""
        capabilities = ", ".join(self.requested_capabilities) or "неизвестно"
        sources = ", ".join(self.allowed_sources)
        return (
            "Вижу capability gap.\n\n"
            f"Запрос: {self.user_request}\n"
            f"Не хватает: {capabilities}\n"
            f"Разрешённые источники для поиска после approval: {sources}\n\n"
            "До твоего разрешения я не скачиваю, не импортирую и не обновляю "
            "external skill. Это разрешение не даёт execution: я не выполняю "
            "scripts, shell/browser/file/network/Telegram/env действия и не "
            "расширяю active tool surface. Любой найденный skill сначала пойдёт "
            "в quarantine и static audit.\n\n"
            f"{self.approval_question}"
        )


class ExternalSkillImportResult(BaseModel):
    """Result of one approved acquisition import into quarantine."""

    proposal: ExternalSkillAcquisitionProposal
    quarantined: QuarantinedExternalSkill
    audit_report: ExternalSkillAuditReport
    registry_record: PersonalSkillRegistryRecord

    def render_audit_for_chat(self) -> str:
        """Render the post-import audit summary without activating the skill."""
        capabilities = ", ".join(self.audit_report.requested_capabilities) or "нет"
        risk = self.audit_report.risk_level
        findings = "\n".join(
            f"- {finding.severity}: {finding.code}"
            for finding in self.audit_report.findings
        )
        if not findings:
            findings = "- нет findings"
        return (
            "External skill импортирован в quarantine и не активирован.\n\n"
            f"Skill: {self.quarantined.name}\n"
            f"Risk: {risk}\n"
            f"Requested capabilities: {capabilities}\n"
            f"Registry status: {self.registry_record.status.value}\n\n"
            "Для использования нужен отдельный read-only approval. Для любых "
            "tool calls нужен отдельный execution approval, InvocationProfile "
            "и ToolGateway grant.\n\n"
            f"Findings:\n{findings}"
        )


class ExternalSkillCandidate(BaseModel):
    """Search result for a potential external skill before quarantine import."""

    candidate_id: str
    name: str
    description: str = ""
    source_type: str
    locator: str
    requested_capabilities: tuple[str, ...] = ()
    matched_capabilities: tuple[str, ...] = ()
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()


class ExternalSkillCandidateSearchReport(BaseModel):
    """Approved candidate search output with no active skill changes."""

    proposal: ExternalSkillAcquisitionProposal
    candidates: tuple[ExternalSkillCandidate, ...] = ()
    source_errors: tuple[str, ...] = ()

    def render_for_chat(self) -> str:
        """Render candidate choices without implying activation."""
        if not self.candidates:
            message = (
                "Поиск external skill candidates завершён: подходящих кандидатов "
                "не найдено. Ничего не импортировано и не активировано."
            )
            if self.source_errors:
                message += "\n\nНедоступные источники:\n" + "\n".join(
                    f"- {error}" for error in self.source_errors
                )
            return message
        lines = [
            "Нашла external skill candidates. Ничего не импортировано и не активировано.",
            "",
        ]
        for index, candidate in enumerate(self.candidates, start=1):
            matched = ", ".join(candidate.matched_capabilities) or "нет"
            lines.extend(
                (
                    f"Candidate {index}: {candidate.name}",
                    f"- source: {candidate.source_type}",
                    f"- matched capabilities: {matched}",
                    f"- score: {candidate.score:.2f}",
                )
            )
            if candidate.description:
                lines.append(f"- description: {candidate.description}")
        lines.append("")
        lines.append(
            "Для продолжения нужен выбор candidate и отдельный import в quarantine; "
            "read-only/execution approval это не выдаёт."
        )
        if self.source_errors:
            lines.extend(("", "Недоступные источники:"))
            lines.extend(f"- {error}" for error in self.source_errors)
        return "\n".join(lines)


class LocalFolderExternalSkillSourceProvider:
    """Search local approved skill catalog folders after acquisition approval."""

    source_type = "local_folder"

    def __init__(
        self,
        *,
        root: Path,
        max_candidates: int = 20,
        mapper: CapabilityMapper | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.max_candidates = max_candidates
        self.mapper = mapper or CapabilityMapper()

    def search(
        self,
        proposal: ExternalSkillAcquisitionProposal,
    ) -> tuple[ExternalSkillCandidate, ...]:
        """Return local folder candidates without importing or activating them."""
        if not self.root.exists():
            return ()
        candidates: list[ExternalSkillCandidate] = []
        for child in sorted(self.root.iterdir(), key=lambda item: item.name):
            if len(candidates) >= self.max_candidates:
                break
            if not child.is_dir() or child.is_symlink():
                continue
            if not (child / "SKILL.md").is_file():
                continue
            candidate = self._candidate_from_folder(child, proposal)
            if candidate is not None:
                candidates.append(candidate)
        return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))

    def _candidate_from_folder(
        self,
        folder: Path,
        proposal: ExternalSkillAcquisitionProposal,
    ) -> ExternalSkillCandidate | None:
        try:
            package = parse_external_skill_folder(
                folder,
                source=ExternalSkillSource(
                    source_type="local_folder",
                    locator=str(folder),
                    acquisition_approval_id=proposal.acquisition_approval_id,
                    approved_by_user_id=proposal.approved_by_user_id,
                ),
            )
        except (OSError, ValueError):
            return None
        requested_capabilities = self.mapper.map_package(package)
        matched = tuple(
            capability
            for capability in requested_capabilities
            if capability in set(proposal.requested_capabilities)
        )
        query_score = _text_match_score(
            proposal.user_request,
            " ".join((package.name, package.description, package.skill_markdown)),
        )
        if not matched and query_score <= 0.0:
            return None
        capability_score = 0.7 if matched else 0.0
        score = min(1.0, capability_score + query_score)
        return ExternalSkillCandidate(
            candidate_id=f"local_folder:{package.skill_id}",
            name=package.name,
            description=package.description,
            source_type="local_folder",
            locator=str(folder),
            requested_capabilities=requested_capabilities,
            matched_capabilities=matched,
            score=score,
            evidence=(
                f"root={self.root.name}",
                f"folder={folder.name}",
            ),
        )


class AgentskillsCatalogSourceProvider:
    """Search an approved agentskills.io catalog without downloading packages."""

    source_type = "agentskills.io"

    def __init__(
        self,
        *,
        catalog_url: str,
        fetch_catalog: CatalogFetcher,
        max_candidates: int = 20,
        mapper: CapabilityMapper | None = None,
    ) -> None:
        normalized_url = catalog_url.strip()
        if not is_public_http_url(normalized_url):
            raise ValueError("agentskills.io catalog url must be public http(s)")
        self.catalog_url = normalized_url
        self._fetch_catalog = fetch_catalog
        self.max_candidates = max_candidates
        self.mapper = mapper or CapabilityMapper()

    def search(
        self,
        proposal: ExternalSkillAcquisitionProposal,
    ) -> tuple[ExternalSkillCandidate, ...]:
        """Fetch metadata only after explicit network-fetch approval."""
        _ensure_network_search_allowed(proposal)
        raw_catalog = self._fetch_catalog(self.catalog_url)
        try:
            catalog = json.loads(raw_catalog)
        except json.JSONDecodeError:
            return ()
        candidates: list[ExternalSkillCandidate] = []
        for entry in _catalog_entries(catalog):
            if len(candidates) >= self.max_candidates:
                break
            candidate = self._candidate_from_entry(entry, proposal)
            if candidate is not None:
                candidates.append(candidate)
        return tuple(sorted(candidates, key=lambda item: item.score, reverse=True))

    def _candidate_from_entry(
        self,
        entry: Mapping[str, object],
        proposal: ExternalSkillAcquisitionProposal,
    ) -> ExternalSkillCandidate | None:
        name = str(entry.get("name") or "").strip()
        description = str(entry.get("description") or "").strip()
        locator = str(
            entry.get("url") or entry.get("download_url") or entry.get("locator") or ""
        ).strip()
        if not name or not is_public_http_url(locator):
            return None
        declared_capabilities = _string_tuple(
            entry.get("capabilities") or entry.get("capability")
        )
        declared_tools = _string_tuple(
            entry.get("tools") or entry.get("required_tools") or entry.get("tool")
        )
        requested_capabilities = _catalog_requested_capabilities(
            mapper=self.mapper,
            declared_capabilities=declared_capabilities,
            declared_tools=declared_tools,
        )
        matched = tuple(
            capability
            for capability in requested_capabilities
            if capability in set(proposal.requested_capabilities)
        )
        query_score = _text_match_score(
            proposal.user_request,
            " ".join(
                (
                    name,
                    description,
                    " ".join(declared_capabilities),
                    " ".join(declared_tools),
                )
            ),
        )
        if not matched and query_score <= 0.0:
            return None
        capability_score = 0.7 if matched else 0.0
        score = min(1.0, capability_score + query_score)
        return ExternalSkillCandidate(
            candidate_id=f"agentskills.io:{_short_hash(locator)}",
            name=name,
            description=description,
            source_type=self.source_type,
            locator=locator,
            requested_capabilities=requested_capabilities,
            matched_capabilities=matched,
            score=score,
            evidence=(
                "source=agentskills.io",
                f"catalog={self.catalog_url}",
            ),
        )


class ExternalSkillCandidateSearchService:
    """Search approved external skill sources without importing candidates."""

    def __init__(
        self,
        *,
        providers: Sequence[ExternalSkillSourceProvider],
    ) -> None:
        self._providers = tuple(providers)

    def search(
        self,
        proposal: ExternalSkillAcquisitionProposal,
    ) -> ExternalSkillCandidateSearchReport:
        """Search only after acquisition approval and source allow-list checks."""
        _ensure_search_allowed(proposal)
        candidates: list[ExternalSkillCandidate] = []
        source_errors: list[str] = []
        allowed = set(proposal.allowed_sources)
        for provider in self._providers:
            if provider.source_type not in allowed:
                continue
            try:
                candidates.extend(provider.search(proposal))
            except PermissionError:
                raise
            except Exception as exc:
                source_errors.append(_provider_error_message(provider.source_type, exc))
        return ExternalSkillCandidateSearchReport(
            proposal=proposal,
            candidates=tuple(
                sorted(candidates, key=lambda item: item.score, reverse=True)
            ),
            source_errors=tuple(source_errors),
        )

    def import_candidate(
        self,
        proposal: ExternalSkillAcquisitionProposal,
        *,
        candidate: ExternalSkillCandidate,
        importer: ExternalSkillAcquisitionImporter,
        fetch_remote_archive: ArchiveFetcher | None = None,
    ) -> ExternalSkillImportResult:
        """Import one selected candidate into quarantine via the importer."""
        _ensure_import_allowed(proposal, source_type=candidate.source_type)
        if candidate.source_type == "local_folder":
            return importer.import_local_folder(
                proposal,
                source_root=Path(candidate.locator),
            )
        if candidate.source_type == "agentskills.io":
            if not proposal.grants_network_fetch:
                raise PermissionError(
                    "remote candidate import requires network fetch approval"
                )
            if fetch_remote_archive is None:
                raise PermissionError("remote candidate import requires downloader")
            if not is_public_http_url(candidate.locator):
                raise ValueError("remote candidate locator must be public http(s)")
            archive_bytes = fetch_remote_archive(candidate.locator)
            return importer.import_remote_archive(
                proposal,
                candidate=candidate,
                archive_bytes=archive_bytes,
            )
        raise PermissionError(
            f"candidate source cannot be imported: {candidate.source_type}"
        )


class ExternalSkillAcquisitionImporter:
    """Approved local-folder or remote-archive import into quarantine."""

    def __init__(
        self,
        *,
        quarantine_store: FileExternalSkillQuarantineStore,
        registry: FilePersonalSkillRegistry,
    ) -> None:
        self._quarantine_store = quarantine_store
        self._registry = registry

    def import_remote_archive(
        self,
        proposal: ExternalSkillAcquisitionProposal,
        *,
        candidate: ExternalSkillCandidate,
        archive_bytes: bytes,
    ) -> ExternalSkillImportResult:
        """Safely extract an approved remote archive, then quarantine/audit it."""
        _ensure_import_allowed(proposal, source_type=candidate.source_type)
        if candidate.source_type != "agentskills.io":
            raise PermissionError(
                f"remote archive import does not support source: {candidate.source_type}"
            )
        if not proposal.grants_network_fetch:
            raise PermissionError(
                "remote candidate import requires network fetch approval"
            )
        if not is_public_http_url(candidate.locator):
            raise ValueError("remote candidate locator must be public http(s)")
        with tempfile.TemporaryDirectory(
            prefix="remote-external-skill-",
            dir=self._quarantine_store.root,
        ) as temp_dir:
            staging_root = Path(temp_dir) / "extracted"
            staging_root.mkdir(parents=True)
            source_root = _extract_remote_skill_archive(archive_bytes, staging_root)
            source = ExternalSkillSource(
                source_type="agentskills.io",
                locator=candidate.locator,
                acquisition_approval_id=proposal.acquisition_approval_id,
                approved_by_user_id=proposal.approved_by_user_id,
            )
            return self._import_prepared_folder(
                proposal,
                source_root=source_root,
                source=source,
            )

    def import_local_folder(
        self,
        proposal: ExternalSkillAcquisitionProposal,
        *,
        source_root: Path,
    ) -> ExternalSkillImportResult:
        """Import a local folder only after scoped acquisition approval."""
        _ensure_import_allowed(proposal, source_type="local_folder")
        source = ExternalSkillSource(
            source_type="local_folder",
            locator=str(source_root.expanduser().resolve()),
            acquisition_approval_id=proposal.acquisition_approval_id,
            approved_by_user_id=proposal.approved_by_user_id,
        )
        return self._import_prepared_folder(
            proposal,
            source_root=source_root,
            source=source,
        )

    def _import_prepared_folder(
        self,
        proposal: ExternalSkillAcquisitionProposal,
        *,
        source_root: Path,
        source: ExternalSkillSource,
    ) -> ExternalSkillImportResult:
        """Quarantine, audit and register a prepared external skill folder."""
        quarantined = self._quarantine_store.import_folder(source_root, source=source)
        audit_report = audit_external_skill_package(quarantined.package)
        record = self._registry.register_quarantined(
            quarantined,
            audit_report=audit_report,
        )
        imported_proposal = proposal.model_copy(
            update={
                "status": ExternalSkillAcquisitionProposalStatus.IMPORTED_TO_QUARANTINE
            }
        )
        return ExternalSkillImportResult(
            proposal=imported_proposal,
            quarantined=quarantined,
            audit_report=audit_report,
            registry_record=record,
        )


class CapabilityGapReport(BaseModel):
    """Structured result of capability truth lookup for one user request."""

    user_request: str
    required_capabilities: tuple[str, ...]
    available_capabilities: tuple[str, ...] = ()
    degraded_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    disabled_capabilities: tuple[str, ...] = ()
    has_gap: bool = False
    proposal: ExternalSkillAcquisitionProposal | None = None
    next_actions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class CapabilityGapDetector(BaseModel):
    """Detect whether a request needs external skill acquisition."""

    default_allowed_sources: tuple[str, ...] = ("local_folder", "agentskills.io")
    acquisition_candidate_statuses: tuple[CapabilityStatus, ...] = Field(
        default=(
            CapabilityStatus.DISABLED,
            CapabilityStatus.CONFIGURED_ONLY,
            CapabilityStatus.ORPHANED,
            CapabilityStatus.BLOCKED,
        )
    )

    def detect(
        self,
        *,
        user_request: str,
        required_capabilities: tuple[str, ...],
        capability_graph: CapabilityGraph,
        allowed_sources: tuple[str, ...] | None = None,
    ) -> CapabilityGapReport:
        """Return a fail-closed gap report without doing search/import."""
        available: list[str] = []
        degraded: list[str] = []
        disabled: list[str] = []
        missing: list[str] = []
        evidence: list[str] = []
        next_actions: list[str] = []

        for capability in required_capabilities:
            node = _find_capability_node(capability_graph, capability)
            if node is None:
                missing.append(capability)
                evidence.append(f"{capability}: missing from CapabilityGraph")
                continue
            evidence.append(f"{node.id}: {node.status.value}")
            if node.status is CapabilityStatus.AVAILABLE:
                available.append(capability)
            elif node.status is CapabilityStatus.DEGRADED:
                degraded.append(capability)
                reason = f": {node.reason}" if node.reason else ""
                next_actions.append(
                    f"Восстановить degraded capability {capability}{reason}"
                )
            elif node.status in self.acquisition_candidate_statuses:
                disabled.append(capability)
                missing.append(capability)
            else:
                degraded.append(capability)
                reason = f": {node.reason}" if node.reason else ""
                next_actions.append(
                    f"Проверить capability {capability} со статусом "
                    f"{node.status.value}{reason}"
                )

        proposal: ExternalSkillAcquisitionProposal | None = None
        if missing:
            proposal = ExternalSkillAcquisitionProposal(
                user_request=user_request,
                requested_capabilities=tuple(dict.fromkeys(missing)),
                allowed_sources=allowed_sources or self.default_allowed_sources,
            )
            next_actions.append(
                "Спросить Никиту о scoped approval на поиск/import candidate skill."
            )
        elif not degraded:
            next_actions.append("Use existing capabilities; do not acquire a skill.")

        return CapabilityGapReport(
            user_request=user_request,
            required_capabilities=required_capabilities,
            available_capabilities=tuple(dict.fromkeys(available)),
            degraded_capabilities=tuple(dict.fromkeys(degraded)),
            missing_capabilities=tuple(dict.fromkeys(missing)),
            disabled_capabilities=tuple(dict.fromkeys(disabled)),
            has_gap=bool(missing or degraded),
            proposal=proposal,
            next_actions=tuple(dict.fromkeys(next_actions)),
            evidence=tuple(evidence),
        )


def _find_capability_node(
    graph: CapabilityGraph,
    capability: str,
) -> CapabilityNode | None:
    for node in graph.capabilities:
        if node.capability_id == capability or node.label == capability:
            return node
    return None


def _ensure_import_allowed(
    proposal: ExternalSkillAcquisitionProposal,
    *,
    source_type: str,
) -> None:
    if (
        proposal.status
        is not ExternalSkillAcquisitionProposalStatus.APPROVED_FOR_SEARCH
    ):
        raise PermissionError("external skill import requires acquisition approval")
    if not proposal.acquisition_approval_id:
        raise PermissionError("external skill import requires acquisition approval id")
    if not proposal.grants_import:
        raise PermissionError("acquisition approval does not grant quarantine import")
    if source_type not in set(proposal.allowed_sources):
        raise PermissionError(f"source type is not allowed: {source_type}")


def _ensure_search_allowed(proposal: ExternalSkillAcquisitionProposal) -> None:
    if (
        proposal.status
        is not ExternalSkillAcquisitionProposalStatus.APPROVED_FOR_SEARCH
    ):
        raise PermissionError("external skill search requires search approval")
    if not proposal.acquisition_approval_id:
        raise PermissionError("external skill search requires acquisition approval id")
    if not proposal.allowed_sources:
        raise PermissionError("external skill search requires allowed sources")


def _ensure_network_search_allowed(
    proposal: ExternalSkillAcquisitionProposal,
) -> None:
    _ensure_search_allowed(proposal)
    if not proposal.grants_network_fetch:
        raise PermissionError(
            "agentskills.io candidate search requires network fetch approval"
        )


def _extract_remote_skill_archive(archive_bytes: bytes, destination: Path) -> Path:
    """Extract a remote zip archive without trusting paths or permissions."""
    if len(archive_bytes) > MAX_REMOTE_ARCHIVE_BYTES:
        raise ValueError("remote skill archive is too large")
    try:
        archive = zipfile.ZipFile(BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("remote skill archive must be a zip file") from exc
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > MAX_REMOTE_ARCHIVE_FILES:
            raise ValueError("remote skill archive has too many files")
        total_size = 0
        for info in infos:
            total_size += info.file_size
            if total_size > MAX_REMOTE_UNCOMPRESSED_BYTES:
                raise ValueError("remote skill archive expands too large")
            relative_path = _safe_archive_relative_path(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"unsafe archive path: {info.filename}")
            target = (destination / Path(*relative_path.parts)).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive path: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return _find_extracted_skill_root(destination)


def _safe_archive_relative_path(filename: str) -> PurePosixPath:
    relative_path = PurePosixPath(filename)
    if (
        not filename
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"unsafe archive path: {filename}")
    return relative_path


def _find_extracted_skill_root(destination: Path) -> Path:
    if (destination / "SKILL.md").is_file():
        return destination
    candidates = [
        child
        for child in destination.iterdir()
        if child.is_dir() and not child.is_symlink() and (child / "SKILL.md").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "remote skill archive must contain SKILL.md at root or one top-level folder"
    )


def _catalog_entries(catalog: object) -> tuple[Mapping[str, object], ...]:
    raw_entries: object
    if isinstance(catalog, Mapping):
        raw_entries = (
            catalog.get("skills")
            or catalog.get("items")
            or catalog.get("candidates")
            or ()
        )
    else:
        raw_entries = catalog
    if not isinstance(raw_entries, list):
        return ()
    return tuple(entry for entry in raw_entries if isinstance(entry, Mapping))


def _catalog_requested_capabilities(
    *,
    mapper: CapabilityMapper,
    declared_capabilities: tuple[str, ...],
    declared_tools: tuple[str, ...],
) -> tuple[str, ...]:
    capabilities: set[str] = set()
    for declared in (*declared_capabilities, *declared_tools):
        normalized = _normalize_catalog_token(declared)
        if not normalized:
            continue
        mapped = mapper.tool_capability_map.get(normalized)
        if mapped:
            capabilities.update(mapped)
        else:
            capabilities.add(normalized)
    return tuple(sorted(capabilities))


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: tuple[object, ...] = tuple(value.split(","))
    elif isinstance(value, list | tuple | set):
        raw_values = tuple(value)
    else:
        return ()
    return tuple(
        item for item in (str(raw_value).strip() for raw_value in raw_values) if item
    )


def _normalize_catalog_token(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _provider_error_message(source_type: str, exc: Exception) -> str:
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return f"{source_type}: {exc.__class__.__name__}: {message}"


def _text_match_score(query: str, text: str) -> float:
    tokens = {
        token
        for token in _tokenize(query)
        if len(token) >= 4 and token not in _LOW_SIGNAL_TOKENS
    }
    if not tokens:
        return 0.0
    haystack = set(_tokenize(text))
    matches = tokens & haystack
    if not matches:
        return 0.0
    return min(0.3, len(matches) / max(len(tokens), 1) * 0.3)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token.strip(".,:;!?()[]{}<>`'\"").casefold()
        for token in text.split()
        if token.strip(".,:;!?()[]{}<>`'\"")
    )


_LOW_SIGNAL_TOKENS = {
    "skill",
    "external",
    "candidate",
    "навык",
    "нужен",
    "проверь",
}
