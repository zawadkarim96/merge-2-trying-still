import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
spec = importlib.util.spec_from_file_location("desktop_launcher", ROOT_DIR / "desktop_launcher.py")
desktop_launcher = importlib.util.module_from_spec(spec)
assert spec and spec.loader  # satisfy mypy/linters
spec.loader.exec_module(desktop_launcher)


def test_import_pywebview_installs_when_missing(monkeypatch):
    install_calls = []

    def fake_check_call(cmd, *_, **__):
        install_calls.append(cmd)

    monkeypatch.setattr(desktop_launcher.subprocess, "check_call", fake_check_call)

    original_import = builtins.__import__
    import_attempt = {"count": 0}

    def fake_import(name, *args, **kwargs):
        if name == "webview":
            import_attempt["count"] += 1
            if import_attempt["count"] == 1:
                raise ImportError("missing module")
            return SimpleNamespace(create_window=lambda *a, **k: None, start=lambda **k: None)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    webview = desktop_launcher._import_pywebview()

    assert import_attempt["count"] == 2  # tried once, installed, then retried
    assert len(install_calls) == 1
    assert "pywebview>=4.4" in install_calls[0]
    assert hasattr(webview, "create_window")


def test_import_pywebview_reports_install_failure(monkeypatch):
    def fake_check_call(*_, **__):
        raise subprocess.CalledProcessError(1, ["pip"])

    monkeypatch.setattr(desktop_launcher.subprocess, "check_call", fake_check_call)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "webview":
            raise ImportError("missing module")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        desktop_launcher._import_pywebview()

    assert "pywebview installation failed" in str(excinfo.value)
