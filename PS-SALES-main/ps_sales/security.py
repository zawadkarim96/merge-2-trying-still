"""Security helpers for password hashing and account lockout."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import os
from typing import Optional

from .config import AppConfig
from .repositories import UserRepository


@dataclass
class PasswordService:
    """Manage password hashing and upgrades using PBKDF2."""

    iterations: int = 120_000
    algorithm: str = "sha256"

    @classmethod
    def default(cls) -> "PasswordService":
        return cls()

    def hash(self, password: str) -> str:
        salt = os.urandom(16)
        derived = hashlib.pbkdf2_hmac(
            self.algorithm, password.encode("utf-8"), salt, self.iterations
        )
        encoded_salt = base64.b64encode(salt).decode("ascii")
        encoded_hash = base64.b64encode(derived).decode("ascii")
        return f"pbkdf2${self.algorithm}${self.iterations}${encoded_salt}${encoded_hash}"

    def verify(self, password: str, stored_hash: str) -> bool:
        if stored_hash.startswith("pbkdf2$"):
            try:
                _, algorithm, iteration_str, salt_b64, hash_b64 = stored_hash.split("$")
                iterations = int(iteration_str)
                salt = base64.b64decode(salt_b64)
                expected = base64.b64decode(hash_b64)
            except (ValueError, TypeError):
                return False
            derived = hashlib.pbkdf2_hmac(
                algorithm, password.encode("utf-8"), salt, iterations
            )
            return hmac.compare_digest(derived, expected)

        if stored_hash and len(stored_hash) == 64 and all(c in "0123456789abcdef" for c in stored_hash):
            legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
            return hmac.compare_digest(legacy, stored_hash)
        return False

    def needs_update(self, stored_hash: str) -> bool:
        return not stored_hash.startswith("pbkdf2$")


@dataclass
class AccountLockoutService:
    """Handle login throttling based on repeated failed attempts."""

    config: AppConfig
    users: UserRepository

    def is_locked(self, username: str) -> Optional[str]:
        failures = self.users.count_recent_failures(
            username, self.config.login_lockout_minutes
        )
        if failures >= self.config.login_max_attempts:
            last_failure = self.users.latest_failure_time(username)
            return last_failure
        return None

    def record_attempt(self, username: str, success: bool) -> None:
        self.users.create_login_event(username, success)
        self.users.purge_login_history(self.config.login_lockout_minutes)
        if success:
            self.users.clear_login_events(username)
        
    def lockout_message(self, username: str) -> str:
        last_failure = self.users.latest_failure_time(username)
        if not last_failure:
            return "Too many failed attempts. Please try again later."
        try:
            timestamp = datetime.fromisoformat(last_failure)
            unlock_time = timestamp + timedelta(minutes=self.config.login_lockout_minutes)
            return f"Account locked due to repeated failures. Try again after {unlock_time:%Y-%m-%d %H:%M}."
        except ValueError:
            return "Account locked due to repeated failures."
