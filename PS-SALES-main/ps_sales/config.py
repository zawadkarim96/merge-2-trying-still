"""Application configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    """Hold runtime configuration options for the application."""

    data_dir: Path
    db_url: str
    upload_retention: Optional[timedelta]
    virus_scan_command: Optional[str]
    allowed_mime_types: tuple[str, ...]
    login_max_attempts: int
    login_lockout_minutes: int
    pre_due_warning_days: int

    @property
    def db_is_sqlite(self) -> bool:
        return self.db_url.startswith("sqlite:")


DEFAULT_ALLOWED_MIME_TYPES = (
    "application/pdf",
    "image/png",
    "image/jpeg",
)


def load_config() -> AppConfig:
    """Load settings from environment variables with sane defaults."""

    data_dir = Path(
        os.environ.get("PS_SALES_DATA_DIR", _default_data_dir())
    ).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("quotations", "work_orders", "delivery_orders", "receipts"):
        (data_dir / "uploads" / subdir).mkdir(parents=True, exist_ok=True)

    db_url = os.environ.get("PS_SALES_DB_URL")
    if not db_url:
        db_path = data_dir / "ps_sales.db"
        db_url = f"sqlite:///{db_path}" if os.name != "nt" else f"sqlite:///{db_path.as_posix()}"

    retention_days = os.environ.get("PS_SALES_UPLOAD_RETENTION_DAYS")
    upload_retention = (
        timedelta(days=int(retention_days)) if retention_days else None
    )

    virus_scan_command = os.environ.get("PS_SALES_VIRUS_SCAN_CMD")

    mime_types = os.environ.get("PS_SALES_ALLOWED_MIME_TYPES")
    if mime_types:
        allowed_mime_types = tuple(mt.strip() for mt in mime_types.split(",") if mt.strip())
    else:
        allowed_mime_types = DEFAULT_ALLOWED_MIME_TYPES

    max_attempts = int(os.environ.get("PS_SALES_LOGIN_MAX_ATTEMPTS", "5"))
    lockout_minutes = int(os.environ.get("PS_SALES_LOGIN_LOCKOUT_MINUTES", "15"))
    warning_days = int(os.environ.get("PS_SALES_PRE_DUE_WARNING_DAYS", "3"))

    return AppConfig(
        data_dir=data_dir,
        db_url=db_url,
        upload_retention=upload_retention,
        virus_scan_command=virus_scan_command,
        allowed_mime_types=allowed_mime_types,
        login_max_attempts=max_attempts,
        login_lockout_minutes=lockout_minutes,
        pre_due_warning_days=warning_days,
    )


def _default_data_dir() -> Path:
    if os.environ.get("PS_SALES_DATA_DIR"):
        return Path(os.environ["PS_SALES_DATA_DIR"]).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA", Path.home()))
    else:
        if os.environ.get("RENDER"):
            root = Path.cwd()
        else:
            root = Path.home()
    return root / ".ps_sales"
