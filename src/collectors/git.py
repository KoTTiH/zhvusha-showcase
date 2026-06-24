"""Git changes collector — tracks commits in project_path between morning sessions."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from types import SimpleNamespace

    from src.memory import EpisodicMemoryProtocol as EpisodicMemory

logger = structlog.get_logger()

# ASCII separators for git log --format parsing
_RECORD_SENTINEL = "\x1eCOMMIT"
_FIELD_SEP = "\x1f"
_GIT_LOG_FORMAT = "%x1eCOMMIT%x1f%H%x1f%ai%x1f%an%x1f%ae%x1f%s"

_DEFAULT_MAX_COMMITS = 100
_SUBPROCESS_TIMEOUT_SECONDS = 30.0
_ZHVUSHA_AUTHOR_EMAIL = "zhvusha@local"
_ZHVUSHA_AUTHOR_PREFIX = "zhvusha"


@dataclass
class GitCommit:
    sha: str
    short_sha: str
    date: datetime
    author: str
    email: str
    subject: str
    files_changed: int
    insertions: int
    deletions: int


@dataclass
class GitCollectionResult:
    commits: list[GitCommit] = field(default_factory=list)
    branch: str = ""
    head_sha: str = ""
    head_short: str = ""
    range_start_sha: str = ""
    range_mode: str = "since"
    truncated: bool = False
    files_touched: dict[str, int] = field(default_factory=dict)
    error: str = ""


class GitChangesCollector:
    """Collects git commits from project_path between morning sessions.

    State persistence: stores last_head_sha in {workspace}/state/git_collector.json.
    On next run, uses last_head_sha..HEAD range if SHA still exists (handles
    rebases via cat-file check), else falls back to --since=<lookback>.
    """

    def __init__(self, config: SimpleNamespace) -> None:
        self._project_path = Path(
            getattr(config, "project_path", "~/Projects/ZHVUSHA")
        ).expanduser()
        self._workspace = Path(
            getattr(config, "workspace_path", "~/zhvusha-workspace")
        ).expanduser()
        self._admin_user_id = getattr(config, "admin_user_id", 0)
        self._max_commits = int(
            getattr(config, "git_max_commits", _DEFAULT_MAX_COMMITS)
        )
        self._state_path = self._workspace / "state" / "git_collector.json"

    # --- Public API ---

    async def collect(
        self,
        since: datetime | None = None,
        *,
        force_since: bool = False,
    ) -> GitCollectionResult:
        """Collect git commits since last state or an explicit time window."""
        result = GitCollectionResult()

        if not await self._preflight(result):
            return result

        if since is None:
            since = datetime.now(tz=UTC) - timedelta(hours=24)

        if force_since:
            result.range_mode = "lookback"
            use_sha_range = False
        else:
            use_sha_range = await self._resolve_range(result)
        if result.error or result.range_mode == "empty_repo":
            return result
        # Empty SHA range (last_sha == HEAD) — keep result empty, state still refreshed later
        if (
            result.range_mode == "sha_range"
            and result.range_start_sha
            and result.range_start_sha == result.head_sha
        ):
            return result

        range_start = result.range_start_sha if use_sha_range else None
        log_since = None if use_sha_range else since

        try:
            commits, truncated = await self._run_git_log(
                range_start=range_start, since=log_since
            )
        except Exception as exc:
            logger.warning("git_collector_log_failed", exc_info=True)
            result.error = str(exc)[:200]
            return result

        result.commits = commits
        result.truncated = truncated
        if commits:
            result.files_touched = await self._collect_files_touched(
                range_start=range_start, since=log_since
            )
        return result

    async def _preflight(self, result: GitCollectionResult) -> bool:
        """Validate project_path and branch/HEAD; populate result. Returns True on success."""
        if not self._project_path.exists():
            result.error = f"Путь проекта не найден: {self._project_path}"
            return False
        if not await self._is_git_repo():
            result.error = "Не git-репозиторий"
            return False

        result.branch = await self._current_branch()
        head = await self._rev_parse("HEAD")
        if not head:
            result.range_mode = "empty_repo"
            return False
        result.head_sha = head
        result.head_short = head[:7]
        return True

    async def _resolve_range(self, result: GitCollectionResult) -> bool:
        """Decide SHA-range vs time-based range. Returns True iff using SHA range."""
        state = self._load_state()
        if state and "last_head_sha" in state:
            last_sha = str(state["last_head_sha"])
            if await self._sha_exists(last_sha):
                result.range_start_sha = last_sha
                result.range_mode = "sha_range"
                return True
            logger.warning(
                "git_collector_stale_sha",
                last_sha=last_sha,
                hint="rebase/force-push detected, falling back to --since",
            )
        result.range_mode = "first_run" if not state else "since"
        return False

    async def collect_and_save(
        self,
        episodic: EpisodicMemory | None = None,
        since: datetime | None = None,
        *,
        force_since: bool = False,
    ) -> str:
        """Full pipeline: collect → markdown → inbox → episodes → state."""
        result = await self.collect(since=since, force_since=force_since)

        if result.error:
            self._write_error_inbox(result)
            return f"Git: ошибка — {result.error}"

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        markdown = self._format_markdown(result, today)

        inbox_dir = self._workspace / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        inbox_path = inbox_dir / f"git_{today}.md"
        inbox_path.write_text(markdown, encoding="utf-8")
        logger.info(
            "git_inbox_written",
            path=str(inbox_path),
            commits=len(result.commits),
        )

        if result.range_mode == "empty_repo":
            return "Репозиторий пустой"

        if episodic is not None and result.commits:
            for commit in result.commits:
                importance = self._importance_for_commit(commit)
                actor = _actor_for_commit(commit)
                await episodic.record(
                    content=(
                        f"{actor.label} commit {commit.short_sha}: {commit.subject}"
                    ),
                    user_id=self._admin_user_id,
                    chat_type="personal",
                    role="user",
                    source="git",
                    importance=importance,
                    person_name=actor.person_name,
                    significance="inner_circle",
                    domain="chat",
                    metadata={
                        "sha": commit.sha,
                        "branch": result.branch,
                        "actor": actor.kind,
                        "author": commit.author,
                        "email": commit.email,
                        "self_coding_commit": str(actor.is_self_coding).lower(),
                        "files_changed": str(commit.files_changed),
                        "insertions": str(commit.insertions),
                        "deletions": str(commit.deletions),
                    },
                )

        self._save_state(
            head_sha=result.head_sha,
            head_short=result.head_short,
            branch=result.branch,
            commit_count=len(result.commits),
        )

        if not result.commits:
            return "Нет новых коммитов"
        suffix = " (обрезано)" if result.truncated else ""
        return f"{len(result.commits)} коммитов{suffix} в {result.branch}"

    # --- Git subprocess wrappers ---

    async def _run(
        self, *args: str, timeout: float = _SUBPROCESS_TIMEOUT_SECONDS
    ) -> tuple[int, str, str]:
        """Run `git <args...>` in project_path. Returns (rc, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(self._project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
        )

    async def _is_git_repo(self) -> bool:
        try:
            rc, out, _ = await self._run("rev-parse", "--is-inside-work-tree")
        except Exception:
            return False
        return rc == 0 and out.strip() == "true"

    async def _current_branch(self) -> str:
        rc, out, _ = await self._run("rev-parse", "--abbrev-ref", "HEAD")
        if rc != 0:
            return "unknown"
        branch = out.strip()
        if branch == "HEAD":
            _, sha_out, _ = await self._run("rev-parse", "--short", "HEAD")
            return f"detached@{sha_out.strip()}"
        return branch

    async def _rev_parse(self, ref: str) -> str:
        rc, out, _ = await self._run("rev-parse", ref)
        return out.strip() if rc == 0 else ""

    async def _sha_exists(self, sha: str) -> bool:
        if not sha or len(sha) < 4:
            return False
        rc, _, _ = await self._run("cat-file", "-e", f"{sha}^{{commit}}")
        return rc == 0

    async def _run_git_log(
        self,
        range_start: str | None,
        since: datetime | None,
    ) -> tuple[list[GitCommit], bool]:
        """Execute `git log` with configured format and range, parse output."""
        args = [
            "log",
            f"--format={_GIT_LOG_FORMAT}",
            "--shortstat",
            "--no-color",
            "--no-merges",
            f"-n{self._max_commits + 1}",
        ]
        if range_start:
            args.append(f"{range_start}..HEAD")
        elif since is not None:
            args.append(f"--since={since.isoformat()}")

        rc, stdout, stderr = await self._run(*args)
        if rc != 0:
            raise RuntimeError(f"git log failed: {stderr.strip()[:200]}")

        commits = self._parse_git_log(stdout)
        truncated = len(commits) > self._max_commits
        if truncated:
            commits = commits[: self._max_commits]
        return commits, truncated

    async def _collect_files_touched(
        self,
        range_start: str | None,
        since: datetime | None,
    ) -> dict[str, int]:
        """Second git log pass with --name-only to aggregate file paths → count."""
        args = [
            "log",
            "--format=%x1eCOMMIT",
            "--name-only",
            "--no-color",
            "--no-merges",
            f"-n{self._max_commits}",
        ]
        if range_start:
            args.append(f"{range_start}..HEAD")
        elif since is not None:
            args.append(f"--since={since.isoformat()}")
        rc, stdout, _ = await self._run(*args)
        if rc != 0:
            return {}
        counts: dict[str, int] = {}
        for chunk in stdout.split("\x1eCOMMIT"):
            for raw_line in chunk.strip().splitlines():
                line = raw_line.strip()
                if line:
                    counts[line] = counts.get(line, 0) + 1
        return counts

    # --- Parsing ---

    @staticmethod
    def _parse_git_log(raw: str) -> list[GitCommit]:
        """Parse git log output with leading \\x1eCOMMIT sentinel format."""
        commits: list[GitCommit] = []
        chunks = raw.split(_RECORD_SENTINEL)
        for chunk in chunks[1:]:
            stripped_chunk = chunk.lstrip()
            if not stripped_chunk:
                continue
            first_line, _, rest = stripped_chunk.partition("\n")
            # Format after split: ["", sha, date, author, email, subject]
            fields = first_line.split(_FIELD_SEP)
            parts = [p for p in fields if p]
            if len(parts) < 5:
                continue
            sha, ai_date, author, email, subject = parts[:5]
            try:
                date = datetime.fromisoformat(ai_date.strip())
            except ValueError:
                date = datetime.now(tz=UTC)

            files_changed = insertions = deletions = 0
            for raw_line in rest.splitlines():
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue
                if "file" in stripped_line and "changed" in stripped_line:
                    files_changed, insertions, deletions = (
                        GitChangesCollector._parse_shortstat(stripped_line)
                    )
                    break

            commits.append(
                GitCommit(
                    sha=sha,
                    short_sha=sha[:7],
                    date=date,
                    author=author,
                    email=email,
                    subject=subject,
                    files_changed=files_changed,
                    insertions=insertions,
                    deletions=deletions,
                )
            )
        return commits

    @staticmethod
    def _parse_shortstat(line: str) -> tuple[int, int, int]:
        """Parse '5 files changed, 66 insertions(+), 92 deletions(-)'."""
        files = insertions = deletions = 0
        for part in line.split(","):
            part_stripped = part.strip()
            tokens = part_stripped.split()
            if not tokens:
                continue
            try:
                n = int(tokens[0])
            except ValueError:
                continue
            if "file" in part_stripped:
                files = n
            elif "insertion" in part_stripped:
                insertions = n
            elif "deletion" in part_stripped:
                deletions = n
        return files, insertions, deletions

    # --- State persistence ---

    def _load_state(self) -> dict[str, object]:
        if not self._state_path.exists():
            return {}
        try:
            loaded = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "git_state_read_failed",
                path=str(self._state_path),
                exc_info=True,
            )
            return {}
        if isinstance(loaded, dict):
            return loaded
        return {}

    def _save_state(
        self, head_sha: str, head_short: str, branch: str, commit_count: int
    ) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_head_sha": head_sha,
            "last_head_short": head_short,
            "last_branch": branch,
            "last_commit_count": commit_count,
            "last_run_at": datetime.now(tz=UTC).isoformat(),
        }
        self._state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("git_state_saved", head=head_short, branch=branch)

    # --- Formatting ---

    @staticmethod
    def _importance_for_commit(commit: GitCommit) -> float:
        """Importance grows with commit footprint, capped at 0.7."""
        base = 0.4
        size_bonus = min(0.3, commit.files_changed * 0.05)
        return round(base + size_bonus, 2)

    def _format_markdown(self, result: GitCollectionResult, today: str) -> str:
        lines = [f"# Git изменения — {today}", ""]

        if result.range_mode == "empty_repo":
            lines.append("_Репозиторий пустой._")
            lines.append("")
            return "\n".join(lines)

        range_desc = self._describe_range(result)
        lines.append(
            f"**Репозиторий**: {self._project_path.name} (ветка: `{result.branch}`)"
        )
        lines.append(f"**Период**: {range_desc}")
        lines.append("")

        if not result.commits:
            lines.append("_Нет новых коммитов за период._")
            lines.append("")
            return "\n".join(lines)

        if result.truncated:
            lines.append(
                f"> Показаны первые {len(result.commits)} коммитов "
                f"(лимит git_max_commits)."
            )
            lines.append("")

        lines.append("## Коммиты")
        lines.append("")
        for c in result.commits:
            time_str = c.date.strftime("%Y-%m-%d %H:%M")
            actor = _actor_for_commit(c)
            lines.append(f"### `{c.short_sha}` — {c.subject}")
            lines.append(
                f"*{time_str}* · {actor.label} · "
                f"{c.author} <{c.email}> · {c.files_changed} files, "
                f"+{c.insertions} −{c.deletions}"
            )
            lines.append("")

        total_ins = sum(c.insertions for c in result.commits)
        total_del = sum(c.deletions for c in result.commits)
        zhvusha_commits = sum(
            1 for commit in result.commits if _actor_for_commit(commit).is_self_coding
        )
        human_commits = len(result.commits) - zhvusha_commits
        lines.append("## Итого")
        lines.append(f"- {len(result.commits)} коммитов")
        lines.append(f"- Никита: {human_commits} коммит.")
        lines.append(f"- Жвуша self-coding: {zhvusha_commits} коммит.")
        lines.append(
            f"- {len(result.files_touched)} уникальных файлов "
            f"(+{total_ins} / −{total_del})"
        )

        if result.files_touched:
            top = sorted(result.files_touched.items(), key=lambda kv: -kv[1])[:10]
            lines.append("- Top файлы:")
            for path, count in top:
                lines.append(f"  - `{path}` ({count} коммит.)")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _describe_range(result: GitCollectionResult) -> str:
        if result.range_mode == "sha_range" and result.range_start_sha:
            return f"`{result.range_start_sha[:7]}..{result.head_short}`"
        if result.range_mode == "first_run":
            return f"первый запуск → до `{result.head_short}`"
        if result.range_mode == "lookback":
            return f"явный lookback по времени → до `{result.head_short}`"
        return f"fallback по времени → до `{result.head_short}`"

    def _write_error_inbox(self, result: GitCollectionResult) -> None:
        try:
            today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
            inbox_dir = self._workspace / "inbox"
            inbox_dir.mkdir(parents=True, exist_ok=True)
            (inbox_dir / f"git_{today}.md").write_text(
                f"# Git изменения — {today}\n\n"
                f"_Коллектор упал с ошибкой:_ `{result.error}`\n",
                encoding="utf-8",
            )
        except OSError:
            logger.warning("git_error_inbox_write_failed", exc_info=True)


@dataclass(frozen=True)
class _CommitActor:
    kind: str
    label: str
    person_name: str
    is_self_coding: bool


def _actor_for_commit(commit: GitCommit) -> _CommitActor:
    author = commit.author.lower()
    email = commit.email.lower()
    if email == _ZHVUSHA_AUTHOR_EMAIL or author.startswith(_ZHVUSHA_AUTHOR_PREFIX):
        return _CommitActor(
            kind="zhvusha_self_coding",
            label="Жвуша self-coding",
            person_name="Жвуша",
            is_self_coding=True,
        )
    return _CommitActor(
        kind="nikita",
        label="Никита",
        person_name="Никита",
        is_self_coding=False,
    )
