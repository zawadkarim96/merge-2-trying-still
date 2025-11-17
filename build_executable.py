#!/usr/bin/env python3
"""Helper script for packaging PS Service Software into a standalone executable.

Running this script will:

1. Create (or reuse) a dedicated virtual environment in ``.build-venv``.
2. Install the application's runtime dependencies along with PyInstaller.
3. Produce a platform-specific executable in ``dist/PS Service Software`` that can be
   distributed to staff members.

The resulting bundle includes the Streamlit app, the Excel import template,
and the ``desktop_launcher`` entry point. All user data is stored outside of
the read-only bundle so the executable can be double-clicked like a typical
desktop application.
"""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
BUILD_VENV = ROOT_DIR / ".build-venv"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
ENTRY_POINT = ROOT_DIR / "desktop_launcher.py"
APP_DEST_DIR = "."
APP_DISPLAY_NAME = "PS Service Software"
ICON_BINARY = ROOT_DIR / "assets" / "ps_service_software.ico"
ICON_BASE64 = ROOT_DIR / "assets" / "ps_service_software.ico.b64"
BUILD_STAMP = BUILD_VENV / ".ps-build-requirements"


class BuildError(RuntimeError):
    """Raised when the executable build process fails."""


def run_command(command: list[str]) -> None:
    try:
        subprocess.check_call(command, cwd=str(ROOT_DIR))
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        raise BuildError(f"Command failed with exit code {exc.returncode}: {command}") from exc


def ensure_build_environment() -> Path:
    """Create the isolated build virtual environment if it does not exist."""

    if not BUILD_VENV.exists():
        print("Creating build virtual environment in .build-venv ...")
        run_command([sys.executable, "-m", "venv", str(BUILD_VENV)])

    if os.name == "nt":
        python_path = BUILD_VENV / "Scripts" / "python.exe"
    else:
        python_path = BUILD_VENV / "bin" / "python"

    if not python_path.exists():  # pragma: no cover - defensive
        raise BuildError("Virtual environment created but Python binary is missing.")

    return python_path


def _requirements_fingerprint() -> str:
    """Return a hash used to determine if dependencies need to be reinstalled."""

    contents = REQUIREMENTS_FILE.read_bytes()
    return hashlib.sha256(contents).hexdigest()


def install_dependencies(python_path: Path) -> None:
    """Install runtime dependencies and PyInstaller into the build venv."""

    fingerprint = _requirements_fingerprint()
    if BUILD_STAMP.exists() and BUILD_STAMP.read_text().strip() == fingerprint:
        print("Build dependencies already prepared; skipping reinstall.")
        try:
            run_command([str(python_path), "-m", "PyInstaller", "--version"])
            return
        except BuildError:
            print("PyInstaller missing; reinstalling ...")

    pip = [str(python_path), "-m", "pip", "--disable-pip-version-check"]
    print("Upgrading pip inside the build environment ...")
    run_command(pip + ["install", "--upgrade", "pip"])

    print("Installing project requirements ...")
    run_command(pip + ["install", "-r", str(REQUIREMENTS_FILE)])

    print("Installing PyInstaller ...")
    run_command(pip + ["install", "pyinstaller"])

    BUILD_STAMP.write_text(fingerprint)


def _ensure_icon_file() -> Path:
    """Return a filesystem path to the icon, decoding from base64 if required."""

    if ICON_BINARY.exists():
        return ICON_BINARY

    if not ICON_BASE64.exists():  # pragma: no cover - configuration error
        raise BuildError(
            "Icon asset is missing. Expected assets/ps_service_software.ico or .ico.b64."
        )

    try:
        icon_bytes = base64.b64decode(ICON_BASE64.read_text())
    except Exception as exc:  # pragma: no cover - invalid base64 should never happen
        raise BuildError("Unable to decode icon asset from base64 data.") from exc

    icon_target = ROOT_DIR / "assets" / ".ps_service_software.ico"
    icon_target.write_bytes(icon_bytes)
    return icon_target


def build_executable(python_path: Path) -> None:
    """Invoke PyInstaller with the required options to bundle the app."""

    data_separator = ";" if os.name == "nt" else ":"
    add_data_args = [
        f"{ROOT_DIR / 'app.py'}{data_separator}{APP_DEST_DIR}",
        f"{ROOT_DIR / 'import_template.xlsx'}{data_separator}{APP_DEST_DIR}",
    ]

    icon_file = _ensure_icon_file()

    command = [
        str(python_path),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--noconsole",
        "--name",
        APP_DISPLAY_NAME,
        "--add-data",
        add_data_args[0],
        "--add-data",
        add_data_args[1],
        "--icon",
        str(icon_file),
        str(ENTRY_POINT),
    ]

    print("Running PyInstaller ...")
    run_command(command)


def main() -> None:
    try:
        python_path = ensure_build_environment()
        install_dependencies(python_path)
        build_executable(python_path)
    except BuildError as exc:
        print(f"\nERROR: {exc}\n")
        print("Unable to build executable. Please review the output above for details.")
        sys.exit(1)

    print(
        "\nBuild complete! The packaged application is available in dist/"
        f"{APP_DISPLAY_NAME}/"
    )


if __name__ == "__main__":
    main()
