import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ps_sales


def test_default_data_dir_uses_platform_storage(monkeypatch, tmp_path):
    """Ensure the data dir lands in the platform storage tree, not the repo."""

    monkeypatch.delenv("PS_SALES_DATA_DIR", raising=False)
    base_dir = tmp_path / "xdg_home"
    monkeypatch.setenv("XDG_DATA_HOME", str(base_dir))

    data_dir = ps_sales._default_data_dir()

    expected = base_dir / "ps-business-suites" / "sales"
    assert data_dir == expected
    assert data_dir.exists()


def test_default_data_dir_respects_override(monkeypatch, tmp_path):
    override = tmp_path / "custom_sales_home"
    monkeypatch.setenv("PS_SALES_DATA_DIR", str(override))

    data_dir = ps_sales._default_data_dir()

    assert data_dir == override
