#!/usr/bin/env python3
"""Convenience launcher for the PS Service Software desktop app.

Running this script will create (or reuse) a local virtual environment in
```.venv``` next to the repository, install the dependencies declared in
``requirements.txt``, and finally launch the Streamlit app inside the
pywebview-powered desktop shell. Subsequent runs reuse the cached
environment unless the requirements file changes. It is intended to provide a
one-click/one-command way to get to the login page without touching pip
manually or dealing with a browser window. On Windows the launcher prefers
``pythonw.exe`` so the login dialog opens without a console window.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS = ROOT_DIR / "requirements.txt"
SETUP_STAMP = VENV_DIR / ".ps-requirements"
DESKTOP_LAUNCHER = ROOT_DIR / "desktop_launcher.py"


class LauncherError(RuntimeError):
    """Raised when a critical step in the launch process fails."""


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    """Execute a subprocess command, raising LauncherError on failure."""

    try:
        subprocess.check_call(command, cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise LauncherError(f"Command failed with exit code {exc.returncode}: {command}") from exc


def ensure_virtual_environment() -> Path:
    """Create the project's dedicated virtual environment if missing."""

    if not VENV_DIR.exists():
        print("Creating Python virtual environment in .venv ...")
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)])

    if os.name == "nt":
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = VENV_DIR / "bin" / "python"

    if not python_path.exists():  # pragma: no cover - should never happen
        raise LauncherError("The virtual environment was created but the Python binary is missing.")

    return python_path


def _requirements_fingerprint() -> str:
    """Return a hash that represents the current dependency lock."""

    contents = REQUIREMENTS.read_bytes()
    return hashlib.sha256(contents).hexdigest()


def install_dependencies(venv_python: Path) -> None:
    """Install or update required dependencies inside the virtual environment."""

    fingerprint = _requirements_fingerprint()
    if SETUP_STAMP.exists() and SETUP_STAMP.read_text().strip() == fingerprint:
        print("Dependencies already installed. Skipping setup.")
        return

    print("Preparing virtual environment dependencies ...")
    pip_base = [str(venv_python), "-m", "pip", "--disable-pip-version-check"]
    print("Ensuring pip is up to date ...")
    run_command(pip_base + ["install", "--upgrade", "pip"])

    print("Installing project requirements ...")
    run_command(pip_base + ["install", "-r", str(REQUIREMENTS)])

    SETUP_STAMP.write_text(fingerprint)


def _select_launch_interpreter(venv_python: Path) -> Path:
    """Return the interpreter that should host the desktop app."""

    if os.name != "nt":
        return venv_python

    pythonw = venv_python.with_name("pythonw.exe")
    if pythonw.exists():
        return pythonw

    return venv_python


def launch_desktop_app(venv_python: Path) -> None:
    """Launch the desktop experience using the virtual environment's Python."""

    print("Starting the PS Service Software desktop app ...")
    interpreter = _select_launch_interpreter(venv_python)
    command = [str(interpreter), str(DESKTOP_LAUNCHER)]
    run_command(command, cwd=ROOT_DIR)


def main() -> None:
    try:
        venv_python = ensure_virtual_environment()
        install_dependencies(venv_python)
        launch_desktop_app(venv_python)
    except LauncherError as exc:
        print(f"\nERROR: {exc}\n")
        print("Please ensure Python 3.9+ is installed and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
