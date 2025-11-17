from datetime import timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ps_sales import AppConfig, PasswordService, AccountLockoutService, Database, UserRepository


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path,
        db_url=f"sqlite:///{tmp_path / 'test.db'}",
        upload_retention=None,
        virus_scan_command=None,
        allowed_mime_types=("application/pdf",),
        login_max_attempts=3,
        login_lockout_minutes=1,
        pre_due_warning_days=2,
    )


def test_password_service_handles_legacy_hash(tmp_path: Path) -> None:
    service = PasswordService.default()
    password = "Secret123!"
    modern_hash = service.hash(password)
    assert service.verify(password, modern_hash)
    assert not service.needs_update(modern_hash)

    legacy_hash = __import__("hashlib").sha256(password.encode("utf-8")).hexdigest()
    assert service.verify(password, legacy_hash)
    assert service.needs_update(legacy_hash)


def test_account_lockout_triggers_after_failures(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    db = Database.from_config(config)
    with db.begin() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS login_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                success INTEGER NOT NULL,
                occurred_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
    repo = UserRepository(db)
    lockout = AccountLockoutService(config, repo)

    for _ in range(config.login_max_attempts):
        lockout.record_attempt("jane", False)
    assert lockout.is_locked("jane") is not None

    # Successful attempt clears failure history.
    lockout.record_attempt("jane", True)
    assert lockout.is_locked("jane") is None
