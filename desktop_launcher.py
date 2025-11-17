#!/usr/bin/env python3
"""Desktop-friendly launcher for the PS Business Suites Streamlit application.

This module powers the "double-click" experience for staff members. It is used
both when the project is executed directly from source (``python
desktop_launcher.py``) and when a PyInstaller bundle produced by
``build_executable.py`` is launched. The launcher performs a few key tasks:

* Ensure all mutable files (database, uploads, import templates) live in a
  writable per-user directory rather than alongside the executable.
* Start the Streamlit runtime in-process without opening an external browser
  window.
* Host the application inside a lightweight native window via ``pywebview`` so
  the login page appears like a traditional desktop dialog.

Users who prefer the browser experience can continue to rely on
``streamlit run main.py``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

from storage_paths import get_storage_dir

from streamlit.web import bootstrap

try:  # ``pywebview`` provides the native desktop window.
    import webview
except ImportError as exc:  # pragma: no cover - dependency should be present
    raise RuntimeError(
        "pywebview is required for the desktop launcher. "
        "Please reinstall the application dependencies."
    ) from exc


APP_SCRIPT_NAME = os.getenv("PS_APP_SCRIPT", "main.py")
IMPORT_TEMPLATE_NAME = "import_template.xlsx"
HOST_ADDRESS = "127.0.0.1"
SERVER_STARTUP_TIMEOUT = 30.0  # seconds
WINDOW_TITLE = "PS Business Suites"


def resource_path(relative_name: str) -> Path:
    """Return the path to a bundled resource for both source and frozen runs."""

    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS, relative_name)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / relative_name


def ensure_template_file(storage_dir: Path) -> None:
    """Copy the Excel import template into the storage directory if needed."""

    template_source = resource_path(IMPORT_TEMPLATE_NAME)
    template_target = storage_dir / IMPORT_TEMPLATE_NAME

    if template_source.exists() and not template_target.exists():
        shutil.copy2(template_source, template_target)


def main() -> None:
    storage_dir = get_storage_dir()
    ensure_template_file(storage_dir)

    os.environ.setdefault("APP_STORAGE_DIR", str(storage_dir))
    # Streamlit tries to launch the system browser unless this environment
    # variable is set. We want everything contained inside the desktop window.
    os.environ.setdefault("BROWSER", "none")

    app_script = resource_path(APP_SCRIPT_NAME)

    port = _reserve_port()
    flag_options = {
        "server.headless": True,
        "server.address": HOST_ADDRESS,
        "server.port": port,
    }

    streamlit_thread = threading.Thread(
        target=bootstrap.run,
        args=(str(app_script), "", [], flag_options),
        name="streamlit-runtime",
        daemon=True,
    )
    streamlit_thread.start()

    if not _wait_for_server(port, timeout=SERVER_STARTUP_TIMEOUT):
        raise RuntimeError(
            "Timed out waiting for the Streamlit server to start. Please try again."
        )

    app_url = f"http://{HOST_ADDRESS}:{port}"

    try:
        webview.create_window(WINDOW_TITLE, app_url, width=1100, height=760)
        webview.start(debug=False)
    except Exception as exc:  # pragma: no cover - GUI availability depends on platform
        raise RuntimeError(
            "Unable to initialize the desktop window. "
            "Please ensure your system supports pywebview."
        ) from exc


def _reserve_port() -> int:
    """Ask the OS for a free TCP port and return it."""

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((HOST_ADDRESS, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _wait_for_server(port: int, *, timeout: float) -> bool:
    """Poll until the Streamlit server is accepting connections."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.settimeout(0.5)
                sock.connect((HOST_ADDRESS, port))
            except OSError:
                time.sleep(0.2)
            else:
                return True
    return False


if __name__ == "__main__":
    main()

