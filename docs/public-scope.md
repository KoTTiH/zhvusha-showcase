# Public Snapshot Scope

This repository is a portfolio snapshot, not a production export.

Included:

- source code under `src/`;
- tests under `tests/`;
- database migrations under `alembic/`;
- Docker Compose for local PostgreSQL and Redis;
- Python project metadata and lockfile;
- sanitized `.env.example`;
- smoke CI configuration;
- public architecture notes.

Excluded:

- Git history from the private repository;
- `.env` and any real credentials;
- Telegram session files and personal account state;
- database dumps;
- runtime artifacts, logs, screenshots and generated reports;
- local agent/codex/Claude runtime settings;
- private workspaces and personal memory files;
- large binary installers and manual check artifacts.

Reviewers should treat this as a code and architecture sample. It is not
configured to reproduce the private live environment.
