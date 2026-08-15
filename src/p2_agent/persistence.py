from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from p2_agent.schemas import WorkflowState


class SQLiteStateStore:
    """Small durable store for task snapshots and node trace.

    This is intentionally application-level persistence for the current stage.
    LangGraph's production checkpointer will be introduced after the contract is
    stable; this store lets us demonstrate restart recovery without a database
    server.
    """

    def __init__(self, path: str | Path = "data/p2_state.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_snapshots (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def save(self, state: WorkflowState) -> None:
        payload = json.dumps(state.public_dict(), ensure_ascii=False, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO task_snapshots(task_id, status, state_json)
                VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    state_json=excluded.state_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (str(state.task_id), state.status, payload),
            )
            connection.commit()

    def get(self, task_id: UUID | str) -> WorkflowState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM task_snapshots WHERE task_id = ?",
                (str(task_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            return WorkflowState.model_validate_json(row["state_json"])
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"corrupted snapshot for task_id={task_id}") from exc

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id, status, updated_at FROM task_snapshots ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, task_id: UUID | str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM task_snapshots WHERE task_id = ?", (str(task_id),))
            connection.commit()
