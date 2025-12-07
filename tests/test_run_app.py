import sys
from pathlib import Path

import pytest

import run_app


def test_ensure_pywebview_skips_when_present(monkeypatch):
    calls: list[list[str]] = []

    def fake_run_command(command: list[str], *, cwd: Path | None = None) -> None:
        calls.append(command)

    monkeypatch.setattr(run_app, "run_command", fake_run_command)

    run_app.ensure_pywebview_available(Path(sys.executable))

    assert len(calls) == 1
    assert "find_spec('webview')" in calls[0][-1]


def test_ensure_pywebview_installs_when_missing(monkeypatch):
    commands: list[list[str]] = []

    def fake_run_command(command: list[str], *, cwd: Path | None = None) -> None:
        commands.append(command)
        if "find_spec('webview')" in command[-1]:
            raise run_app.LauncherError("missing")

    monkeypatch.setattr(run_app, "run_command", fake_run_command)

    run_app.ensure_pywebview_available(Path(sys.executable))

    assert len(commands) == 2
    assert "find_spec('webview')" in commands[0][-1]
    assert "pip" in commands[1]
    assert "pywebview>=4.4" in commands[1]


def test_ensure_pywebview_failure_is_clear(monkeypatch):
    def fake_run_command(command: list[str], *, cwd: Path | None = None) -> None:
        if "find_spec('webview')" in command[-1]:
            raise run_app.LauncherError("missing")
        raise run_app.LauncherError("pip failed")

    monkeypatch.setattr(run_app, "run_command", fake_run_command)

    with pytest.raises(run_app.LauncherError, match="Failed to install pywebview"):
        run_app.ensure_pywebview_available(Path(sys.executable))
