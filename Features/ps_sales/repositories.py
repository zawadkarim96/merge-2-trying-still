"""Database abstractions using SQLite."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .config import AppConfig


@dataclass
class Database:
    config: AppConfig
    db_path: Path

    @classmethod
    def from_config(cls, config: AppConfig) -> "Database":
        if not config.db_url.startswith("sqlite://"):
            raise ValueError("Only SQLite URLs are supported in the bundled runtime.")
        path_str = config.db_url.split("sqlite:///")[-1]
        db_path = Path(path_str).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(config=config, db_path=db_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def begin(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def raw_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def fetch_df(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)


class UserRepository:
    """Encapsulate CRUD logic for user accounts."""

    def __init__(self, db: Database):
        self._db = db

    def fetch_by_username(self, username: str) -> Optional[dict]:
        with self._db.connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, username, role, pass_hash, display_name, designation, phone
                FROM users
                WHERE username=?
                """,
                (username,),
            ).fetchone()
            return dict(row) if row else None

    def update_password_hash(self, user_id: int, new_hash: str) -> None:
        with self._db.begin() as conn:
            conn.execute(
                "UPDATE users SET pass_hash=? WHERE user_id=?",
                (new_hash, user_id),
            )

    def create_login_event(self, username: str, success: bool) -> None:
        with self._db.begin() as conn:
            conn.execute(
                """
                INSERT INTO login_events(username, success, occurred_at)
                VALUES (?, ?, datetime('now'))
                """,
                (username, int(success)),
            )

    def count_recent_failures(self, username: str, minutes: int) -> int:
        with self._db.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS failures
                FROM login_events
                WHERE username=? AND success=0
                  AND occurred_at >= datetime('now', ?)
                """,
                (username, f"-{minutes} minutes"),
            ).fetchone()
            return row[0] if row else 0

    def latest_failure_time(self, username: str) -> Optional[str]:
        with self._db.connect() as conn:
            row = conn.execute(
                """
                SELECT occurred_at
                FROM login_events
                WHERE username=? AND success=0
                ORDER BY occurred_at DESC
                LIMIT 1
                """,
                (username,),
            ).fetchone()
            return row[0] if row else None

    def purge_login_history(self, minutes: int) -> None:
        with self._db.begin() as conn:
            conn.execute(
                "DELETE FROM login_events WHERE occurred_at < datetime('now', ?)",
                (f"-{minutes} minutes",),
            )

    def clear_login_events(self, username: str) -> None:
        with self._db.begin() as conn:
            conn.execute(
                "DELETE FROM login_events WHERE username=?",
                (username,),
            )
