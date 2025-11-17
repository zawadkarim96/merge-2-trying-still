import importlib.util
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def app_module():
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "app.py"
    spec = importlib.util.spec_from_file_location("app_for_tests", app_path)
    module = importlib.util.module_from_spec(spec)
    module._streamlit_runtime_active = lambda: False
    module._bootstrap_streamlit_app = lambda: None
    loader = spec.loader
    if loader is None:
        raise RuntimeError("Unable to load app module for tests")
    loader.exec_module(module)
    return module


@pytest.fixture()
def db_conn(app_module, monkeypatch):
    monkeypatch.setenv("ADMIN_USER", "test_admin")
    monkeypatch.setenv("ADMIN_PASS", "secret123")
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    app_module.init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()
