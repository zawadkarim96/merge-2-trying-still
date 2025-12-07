"""Shared utilities for the PS Business Suites by ZAD Streamlit app.

This module consolidates configuration, database helpers and security
services that were previously kept in a separate repository. Keeping
these utilities alongside ``sales_app.py`` ensures the sales and CRM
apps can coexist in a single codebase without circular imports or
missing dependencies.
"""
from __future__ import annotations

import base64
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Iterable, Optional, Sequence
import secrets
import hashlib
import sys
import tempfile

DEFAULT_ITERATIONS = 260_000


@dataclass(slots=True)
class AppConfig:
    """Lightweight configuration container for the sales app."""

    data_dir: Path
    db_url: str
    upload_retention: Optional[int]
    virus_scan_command: Optional[str]
    allowed_mime_types: Sequence[str]
    login_max_attempts: int
    login_lockout_minutes: int
    pre_due_warning_days: int
    db_host: str = "localhost"
    db_name: str = "crmpsengltd_crm_db"
    db_user: str = "crmpsengltd_crm_user"
    db_password: str = "r9tJYARukZ!yB6zC"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _default_data_dir() -> Path:
    """Return a writable data directory for the sales app.

    Deployments can override the location via ``PS_SALES_DATA_DIR``. Otherwise,
    choose an OS-appropriate application data folder that lives **outside** the
    repository so deployments do not pollute the cloned source. If that
    location is unavailable we fall back to a temporary directory.
    """

    override = os.getenv("PS_SALES_DATA_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform.startswith("win"):
        base_dir = Path(os.getenv("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    else:
        base_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    preferred = base_dir / "ps-business-suites" / "sales"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ps-sales"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def load_config() -> AppConfig:
    data_dir = _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    db_url = os.getenv("PS_SALES_DB_URL") or f"sqlite:///{data_dir / 'ps_sales.db'}"

    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "crmpsengltd_crm_db")
    db_user = os.getenv("DB_USER", "crmpsengltd_crm_user")
    db_password = os.getenv("DB_PASSWORD", "r9tJYARukZ!yB6zC")

    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, ""))
        except ValueError:
            return default

    upload_retention = os.getenv("PS_SALES_UPLOAD_RETENTION_DAYS")
    try:
        retention = int(upload_retention) if upload_retention else None
    except ValueError:
        retention = None

    allowed_mime_types = os.getenv("PS_SALES_ALLOWED_MIME", "application/pdf")
    mime_tuple: Sequence[str] = tuple(
        m.strip() for m in allowed_mime_types.split(",") if m.strip()
    )

    return AppConfig(
        data_dir=data_dir,
        db_url=db_url,
        upload_retention=retention,
        virus_scan_command=os.getenv("PS_SALES_VIRUS_SCAN"),
        allowed_mime_types=mime_tuple or ("application/pdf",),
        login_max_attempts=_int_env("PS_SALES_LOGIN_MAX_ATTEMPTS", 5),
        login_lockout_minutes=_int_env("PS_SALES_LOGIN_LOCKOUT_MINUTES", 15),
        pre_due_warning_days=_int_env("PS_SALES_PRE_DUE_WARNING_DAYS", 3),
        db_host=db_host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
    )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


class Database:
    """Simple SQLite database wrapper used by the sales app."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: AppConfig) -> "Database":
        prefix = "sqlite:///"
        if config.db_url.startswith(prefix):
            db_path = Path(config.db_url[len(prefix) :])
        else:
            db_path = Path(config.db_url)
        return cls(db_path)

    def raw_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def begin(self) -> Iterable[sqlite3.Connection]:
        conn = self.raw_connection()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# User repository and security services
# ---------------------------------------------------------------------------


class PasswordService:
    def __init__(self, *, iterations: int = DEFAULT_ITERATIONS):
        self.iterations = iterations

    @classmethod
    def default(cls) -> "PasswordService":
        return cls()

    def hash(self, password: str) -> str:
        salt = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self.iterations
        )
        return "pbkdf2_sha256$%d$%s$%s" % (
            self.iterations,
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )

    def verify(self, password: str, stored_hash: str) -> bool:
        if not stored_hash:
            return False
        if stored_hash.startswith("pbkdf2_sha256$"):
            try:
                _, iter_str, salt_b64, hash_b64 = stored_hash.split("$", 3)
                iterations = int(iter_str)
                salt = base64.b64decode(salt_b64)
                expected = base64.b64decode(hash_b64)
            except (ValueError, base64.binascii.Error):
                return False
            derived = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, iterations
            )
            return secrets.compare_digest(derived, expected)
        # Legacy SHA-256 fallback
        legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return secrets.compare_digest(legacy, stored_hash)

    def needs_update(self, stored_hash: str) -> bool:
        if not stored_hash.startswith("pbkdf2_sha256$"):
            return True
        try:
            _, iter_str, *_ = stored_hash.split("$", 3)
            return int(iter_str) < self.iterations
        except ValueError:
            return True


class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    def fetch_by_username(self, username: str):
        with self.db.begin() as conn:
            row = conn.execute(
                "SELECT user_id, username, pass_hash, role, display_name, designation, phone "
                "FROM users WHERE username=?",
                (username,),
            ).fetchone()
            return dict(row) if row else None

    def update_password_hash(self, user_id: int, password_hash: str) -> None:
        with self.db.begin() as conn:
            conn.execute(
                "UPDATE users SET pass_hash=? WHERE user_id=?", (password_hash, user_id)
            )


class AccountLockoutService:
    """Track authentication attempts to prevent brute-force logins."""

    def __init__(self, config: AppConfig, repository: UserRepository):
        self.config = config
        self.repository = repository
        self.db = repository.db

    def record_attempt(self, username: str, success: bool) -> None:
        with self.db.begin() as conn:
            if success:
                conn.execute("DELETE FROM login_events WHERE username=?", (username,))
                return
            conn.execute(
                "INSERT INTO login_events (username, success, occurred_at)"
                " VALUES (?, ?, datetime('now'))",
                (username, 1 if success else 0),
            )

    def is_locked(self, username: str) -> Optional[datetime]:
        cutoff = datetime.utcnow() - timedelta(minutes=self.config.login_lockout_minutes)
        with self.db.begin() as conn:
            rows = conn.execute(
                "SELECT occurred_at FROM login_events "
                "WHERE username=? AND success=0 ORDER BY occurred_at DESC"
                " LIMIT ?",
                (username, self.config.login_max_attempts),
            ).fetchall()
        if len(rows) < self.config.login_max_attempts:
            return None
        try:
            newest = datetime.strptime(rows[0]["occurred_at"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
        if newest < cutoff:
            return None
        return newest + timedelta(minutes=self.config.login_lockout_minutes)

    def lockout_message(self, username: str) -> str:
        unlock = self.is_locked(username)
        if not unlock:
            return "Too many login attempts. Please try again later."
        return f"Too many login attempts. Try again after {unlock:%H:%M}."


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


class UploadManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.base_dir = config.data_dir / "uploads"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, name: str) -> str:
        cleaned = name.replace("\\", "/").split("/")[-1]
        return cleaned or "upload"

    def _target_dir(self, subdir: str) -> Path:
        target = (self.base_dir / subdir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save(self, uploaded_file, subdir: str) -> str:
        target_dir = self._target_dir(subdir)
        filename = self._safe_name(getattr(uploaded_file, "name", "upload"))
        target_path = target_dir / filename
        data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        with open(target_path, "wb") as f:
            f.write(data)
        return str(target_path.relative_to(self.config.data_dir))

    def metadata(self, relative_path: str):
        target = (self.config.data_dir / relative_path).resolve()
        if not target.exists():
            return None
        stat = target.stat()
        return {
            "size": stat.st_size,
            "uploaded": datetime.fromtimestamp(stat.st_mtime),
        }

    def enforce_retention(self) -> None:
        if not self.config.upload_retention:
            return
        cutoff = datetime.now() - timedelta(days=self.config.upload_retention)
        for path in self.base_dir.rglob("*"):
            if path.is_file() and datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------


class NotificationScheduler:
    """Generate reminder notifications based on database state."""

    def __init__(self, database: Database, config: AppConfig):
        self.database = database
        self.config = config

    def _ensure_row_factory(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row

    def create_notification(self, user_id: int, message: str, due_date: date) -> Optional[int]:
        due_iso = due_date.isoformat()
        with self.database.begin() as conn:
            self._ensure_row_factory(conn)
            existing = conn.execute(
                "SELECT notification_id FROM notifications WHERE user_id=? AND message=? AND due_date=?",
                (user_id, message, due_iso),
            ).fetchone()
            if existing:
                return existing[0]
            cur = conn.execute(
                "INSERT INTO notifications(user_id, message, due_date, read) VALUES (?, ?, ?, 0)",
                (user_id, message, due_iso),
            )
            return cur.lastrowid

    def notify_follow_up(self, quotation_id: int) -> None:
        with self.database.raw_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT q.follow_up_date, q.salesperson_id, c.name AS company
                FROM quotations q
                LEFT JOIN companies c ON c.company_id = q.company_id
                WHERE q.quotation_id=?
                """,
                (quotation_id,),
            ).fetchone()
        if not row:
            return
        follow_up = row["follow_up_date"]
        if not follow_up:
            return
        try:
            due = datetime.strptime(str(follow_up), "%Y-%m-%d").date()
        except ValueError:
            return
        label = row["company"] or "quotation"
        message = f"Follow-up reminder for quotation #{quotation_id} ({label})"
        salesperson_id = row["salesperson_id"]
        if salesperson_id:
            self.create_notification(int(salesperson_id), message, due)

    def generate_system_notifications(self) -> None:
        warning_window = date.today() + timedelta(days=self.config.pre_due_warning_days)
        with self.database.raw_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT q.quotation_id, q.follow_up_date, q.salesperson_id, c.name AS company
                FROM quotations q
                LEFT JOIN companies c ON c.company_id = q.company_id
                WHERE q.follow_up_date IS NOT NULL AND q.status IN ('pending','inform_later')
                """
            ).fetchall()
        for row in rows:
            try:
                due = datetime.strptime(str(row["follow_up_date"]), "%Y-%m-%d").date()
            except ValueError:
                continue
            if due > warning_window:
                continue
            label = row["company"] or "quotation"
            message = f"Upcoming follow-up for quotation #{row['quotation_id']} ({label})"
            salesperson_id = row["salesperson_id"]
            if salesperson_id:
                self.create_notification(int(salesperson_id), message, due)


__all__ = [
    "AccountLockoutService",
    "AppConfig",
    "Database",
    "NotificationScheduler",
    "PasswordService",
    "UploadManager",
    "UserRepository",
    "load_config",
]
