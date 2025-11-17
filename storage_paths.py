"""Shared helpers for resolving the PS Service Software storage directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_STORAGE_SUBDIR = "ps-business-suites"


def get_storage_dir() -> Path:
    """Return the default writable directory for application data."""

    if sys.platform.startswith("win"):
        base_dir = Path(os.getenv("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support"
    else:
        base_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    storage_dir = base_dir / APP_STORAGE_SUBDIR
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir
