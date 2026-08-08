from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.agentic.models import ResearchRun, RunStatus, utc_now


class AgenticRunRepository:
    """Independent v2 persistence; it does not alter the v1 database schema."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agentic_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS agentic_runs_status_updated_idx
                ON agentic_runs(status, updated_at)
                """
            )

    def save(self, run: ResearchRun) -> None:
        self.initialize()
        payload = run.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agentic_runs(
                    run_id, status, user_query, payload_json,
                    created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    user_query = excluded.user_query,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run.run_id,
                    run.status.value,
                    run.user_query,
                    payload,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                ),
            )

    def get(self, run_id: str) -> ResearchRun | None:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM agentic_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return ResearchRun.model_validate_json(row["payload_json"])

    def request_cancel(self, run_id: str) -> ResearchRun | None:
        self.initialize()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_json FROM agentic_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            run = ResearchRun.model_validate_json(row["payload_json"])
            if run.status == RunStatus.COMPLETED:
                return run
            run.cancel_requested = True
            run.status = RunStatus.CANCELLED
            run.stop_reason = "cancel_requested"
            run.updated_at = utc_now()
            connection.execute(
                """
                UPDATE agentic_runs
                SET status = ?, payload_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    run.status.value,
                    run.model_dump_json(),
                    run.updated_at.isoformat(),
                    run_id,
                ),
            )
        return run

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
