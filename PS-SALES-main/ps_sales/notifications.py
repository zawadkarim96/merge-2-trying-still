"""Notification orchestration helpers."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

from .config import AppConfig
from .repositories import Database


@dataclass
class NotificationScheduler:
    """Generate proactive reminders for follow-ups and overdue tasks."""

    db: Database
    config: AppConfig

    def create_notification(self, user_id: int, message: str, due_date: date) -> None:
        try:
            with self.db.begin() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM notifications WHERE user_id=? AND message=? AND due_date=?",
                    (user_id, message, due_date.isoformat()),
                ).fetchone()
                if exists:
                    return
                conn.execute(
                    """
                    INSERT INTO notifications(user_id, message, due_date)
                    VALUES (?, ?, ?)
                    """,
                    (user_id, message, due_date.isoformat()),
                )
        except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
            if self._is_missing_table_error(exc):
                return
            raise

    def notify_follow_up(self, quotation_id: int) -> None:
        quotation = self._safe_fetchone(
            "SELECT quotation_id, salesperson_id, follow_up_date FROM quotations WHERE quotation_id=?",
            (quotation_id,),
        )
        if not quotation or not quotation["follow_up_date"]:
            return
        follow_date = datetime.fromisoformat(quotation["follow_up_date"]).date()
        admins = [
            row["user_id"]
            for row in self._safe_fetchall("SELECT user_id FROM users WHERE role='admin'")
        ]
        message = f"Follow up quotation #{quotation_id}"
        self.create_notification(quotation["salesperson_id"], message, follow_date)
        for admin in admins:
            self.create_notification(admin, message, follow_date)

    def generate_system_notifications(self) -> None:
        admins = [
            row["user_id"]
            for row in self._safe_fetchall("SELECT user_id FROM users WHERE role='admin'")
        ]
        work_order_rows = self._safe_fetchall(
            """
            SELECT q.quotation_id, q.salesperson_id, q.quote_date
            FROM quotations q
            LEFT JOIN work_orders w ON q.quotation_id = w.quotation_id
            WHERE q.status='accepted' AND w.work_order_id IS NULL
            """
        )
        today = date.today()
        grace_settings = self._load_grace_periods()
        warn_days = self.config.pre_due_warning_days

        for row in work_order_rows:
            quote_date = datetime.fromisoformat(row["quote_date"]).date()
            due_date = quote_date + timedelta(days=grace_settings["work_order"])
            message = f"Quotation #{row['quotation_id']} is missing a work order"
            self._broadcast_with_warning(message, due_date, row["salesperson_id"], admins, warn_days)

        delivery_rows = self._safe_fetchall(
            """
            SELECT w.work_order_id, w.upload_date, w.quotation_id, q.salesperson_id
            FROM work_orders w
            LEFT JOIN delivery_orders d ON w.work_order_id = d.work_order_id
            JOIN quotations q ON q.quotation_id = w.quotation_id
            WHERE d.do_id IS NULL
            """
        )
        for row in delivery_rows:
            upload_date = datetime.fromisoformat(row["upload_date"]).date()
            due_date = upload_date + timedelta(days=grace_settings["delivery_order"])
            message = f"Work order #{row['work_order_id']} has no delivery order"
            self._broadcast_with_warning(message, due_date, row["salesperson_id"], admins, warn_days)

        payment_rows = self._safe_fetchall(
            """
            SELECT d.do_id, d.upload_date, q.salesperson_id
            FROM delivery_orders d
            JOIN work_orders w ON w.work_order_id = d.work_order_id
            JOIN quotations q ON q.quotation_id = w.quotation_id
            WHERE d.payment_received = 0
            """
        )
        for row in payment_rows:
            due_date = datetime.fromisoformat(row["upload_date"]).date() + timedelta(
                days=grace_settings["payment_due"]
            )
            message = f"Payment overdue for delivery order #{row['do_id']}"
            self._broadcast_with_warning(message, due_date, row["salesperson_id"], admins, warn_days)

        stale_quotes = self._safe_fetchall(
            """
            SELECT quotation_id, quote_date, salesperson_id
            FROM quotations
            WHERE status='pending'
            """
        )
        for row in stale_quotes:
            quote_date = datetime.fromisoformat(row["quote_date"]).date()
            due_date = quote_date + timedelta(days=grace_settings["quotation_pending"])
            message = f"Quotation #{row['quotation_id']} pending follow-up"
            self._broadcast_with_warning(message, due_date, row["salesperson_id"], admins, warn_days)

    def _broadcast_with_warning(
        self, message: str, due_date: date, salesperson_id: int, admins: Iterable[int], warn_days: int
    ) -> None:
        today = date.today()
        recipients = [salesperson_id, *admins]
        if due_date <= today:
            for rid in recipients:
                self.create_notification(rid, message, due_date)
        elif due_date - timedelta(days=warn_days) <= today:
            warn_message = f"Upcoming: {message}"
            for rid in recipients:
                self.create_notification(rid, warn_message, due_date)

    def _load_grace_periods(self) -> dict[str, int]:
        rows = self._safe_fetchall("SELECT key, value FROM settings")
        lookup = {row["key"]: int(row["value"]) for row in rows}
        return {
            "work_order": lookup.get("work_order_grace_days", 7),
            "delivery_order": lookup.get("delivery_order_grace_days", 7),
            "payment_due": lookup.get("payment_due_days", 14),
            "quotation_pending": lookup.get("quotation_pending_days", 10),
        }

    def _safe_fetchall(self, query: str, params: Sequence | None = None) -> list[sqlite3.Row]:
        try:
            with self.db.connect() as conn:
                cursor = conn.execute(query, tuple(params or ()))
                return cursor.fetchall()
        except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
            if self._is_missing_table_error(exc):
                return []
            raise

    def _safe_fetchone(self, query: str, params: Sequence | None = None) -> sqlite3.Row | None:
        try:
            with self.db.connect() as conn:
                cursor = conn.execute(query, tuple(params or ()))
                return cursor.fetchone()
        except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
            if self._is_missing_table_error(exc):
                return None
            raise

    @staticmethod
    def _is_missing_table_error(exc: sqlite3.OperationalError) -> bool:
        return "no such table" in str(exc).lower()
