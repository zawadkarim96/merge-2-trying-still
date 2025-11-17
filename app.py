import contextlib
import io
import json
import math
import os
import re
import sqlite3
import hashlib
import zipfile
from calendar import monthrange
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv
from textwrap import dedent
import pandas as pd


import streamlit as st
from collections import OrderedDict

try:
    from storage_paths import get_storage_dir
except ModuleNotFoundError:  # pragma: no cover - defensive for bundled test imports
    import importlib.util

    _storage_module_path = Path(__file__).resolve().parent / "storage_paths.py"
    spec = importlib.util.spec_from_file_location("storage_paths", _storage_module_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:  # pragma: no cover - should not happen
        raise
    loader.exec_module(module)
    get_storage_dir = module.get_storage_dir

# ---------- Config ----------
load_dotenv()
DEFAULT_BASE_DIR = get_storage_dir()
BASE_DIR = Path(os.getenv("APP_STORAGE_DIR", DEFAULT_BASE_DIR))
BASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "ps_crm.db"))
DATE_FMT = "%d-%m-%Y"
CURRENCY_SYMBOL = os.getenv("APP_CURRENCY_SYMBOL", "₹")

UPLOADS_DIR = BASE_DIR / "uploads"
DELIVERY_ORDER_DIR = UPLOADS_DIR / "delivery_orders"
SERVICE_DOCS_DIR = UPLOADS_DIR / "service_documents"
MAINTENANCE_DOCS_DIR = UPLOADS_DIR / "maintenance_documents"
CUSTOMER_DOCS_DIR = UPLOADS_DIR / "customer_documents"
SERVICE_BILL_DIR = UPLOADS_DIR / "service_bills"
REPORT_DOCS_DIR = UPLOADS_DIR / "report_documents"

DEFAULT_QUOTATION_VALID_DAYS = 30

REQUIRED_CUSTOMER_FIELDS = {
    "name": "Name",
    "phone": "Phone",
    "address": "Address",
}

SERVICE_STATUS_OPTIONS = ["In progress", "Completed", "Haven't started"]
DEFAULT_SERVICE_STATUS = SERVICE_STATUS_OPTIONS[0]
GENERATOR_CONDITION_OPTIONS = ["Mint", "Good", "Bad"]

REPORT_PERIOD_OPTIONS = OrderedDict(
    [
        ("daily", "Daily"),
        ("weekly", "Weekly"),
        ("monthly", "Monthly"),
    ]
)

REPORT_GRID_FIELDS = OrderedDict(
    [
        ("customer_name", {"label": "Customer Name", "type": "text"}),
        (
            "reported_complaints",
            {"label": "Reported Complaints", "type": "text"},
        ),
        ("product_details", {"label": "Product Details", "type": "text"}),
        ("details_remarks", {"label": "Details Remarks", "type": "text"}),
        ("status", {"label": "Status", "type": "text"}),
        ("quotation_tk", {"label": "Quotation Tk", "type": "number"}),
        ("bill_tk", {"label": "Bill TK", "type": "number"}),
        ("work_done_date", {"label": "Work Done Date", "type": "date"}),
        ("donation_cost", {"label": "Donation Cost", "type": "number"}),
    ]
)

REPORT_GRID_DISPLAY_COLUMNS = [
    config["label"] for config in REPORT_GRID_FIELDS.values()
]


def _default_report_grid_row() -> dict[str, object]:
    row: dict[str, object] = {}
    for key, config in REPORT_GRID_FIELDS.items():
        if config["type"] == "number":
            row[key] = None
        else:
            row[key] = ""
    return row


def _coerce_grid_number(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_grid_rows(rows: Iterable[dict]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    if not rows:
        return normalized
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        entry: dict[str, object] = {}
        for key, config in REPORT_GRID_FIELDS.items():
            value = raw.get(key)
            if config["type"] == "text":
                entry[key] = clean_text(value)
            elif config["type"] == "number":
                entry[key] = _coerce_grid_number(value)
            elif config["type"] == "date":
                entry[key] = to_iso_date(value)
            else:
                entry[key] = value
        if any(val not in (None, "") for val in entry.values()):
            normalized.append(entry)
    return normalized


def parse_report_grid_payload(value: Optional[str]) -> list[dict[str, object]]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []
    if isinstance(parsed, list):
        return _normalize_grid_rows(parsed)
    return []


def prepare_report_grid_payload(rows: Iterable[dict]) -> Optional[str]:
    normalized = _normalize_grid_rows(rows)
    if not normalized:
        return None
    return json.dumps(normalized, ensure_ascii=False)


def format_report_grid_rows_for_display(
    rows: Iterable[dict], *, empty_ok: bool = False
) -> pd.DataFrame:
    normalized = _normalize_grid_rows(rows)
    if not normalized and not empty_ok:
        return pd.DataFrame(columns=REPORT_GRID_DISPLAY_COLUMNS)
    if not normalized:
        normalized = []
    formatted: list[dict[str, object]] = []
    for entry in normalized:
        display_row: dict[str, object] = {}
        for key, config in REPORT_GRID_FIELDS.items():
            label = config["label"]
            value = entry.get(key)
            if config["type"] == "text":
                display_row[label] = clean_text(value) or ""
            elif config["type"] == "number":
                display_row[label] = value if value is not None else None
            elif config["type"] == "date":
                iso = to_iso_date(value)
                if iso:
                    try:
                        parsed = datetime.strptime(iso, "%Y-%m-%d")
                        display_row[label] = parsed.strftime(DATE_FMT)
                    except ValueError:
                        display_row[label] = iso
                else:
                    display_row[label] = ""
            else:
                display_row[label] = value
        formatted.append(display_row)
    if not formatted:
        return pd.DataFrame(columns=REPORT_GRID_DISPLAY_COLUMNS)
    df = pd.DataFrame(formatted)
    return df.reindex(columns=REPORT_GRID_DISPLAY_COLUMNS)


def _grid_rows_for_editor(rows: Iterable[dict]) -> list[dict[str, object]]:
    """Coerce stored report rows into a format suitable for the data editor."""

    normalized = _normalize_grid_rows(rows)
    source_rows: list[dict[str, object]]
    if normalized:
        source_rows = normalized
    else:
        source_rows = [dict(entry) for entry in rows or []]  # type: ignore[arg-type]
    if not source_rows:
        return []

    editor_rows: list[dict[str, object]] = []
    for entry in source_rows:
        editor_entry: dict[str, object] = {}
        for key, config in REPORT_GRID_FIELDS.items():
            value = entry.get(key)
            if config["type"] == "text":
                editor_entry[key] = clean_text(value) or ""
            elif config["type"] == "number":
                editor_entry[key] = _coerce_grid_number(value)
            elif config["type"] == "date":
                iso = to_iso_date(value)
                if iso:
                    try:
                        editor_entry[key] = datetime.strptime(
                            iso, "%Y-%m-%d"
                        ).date()
                    except ValueError:
                        editor_entry[key] = None
                else:
                    editor_entry[key] = None
            else:
                editor_entry[key] = value
        editor_rows.append(editor_entry)
    return editor_rows


def _grid_rows_from_editor(df: Optional[pd.DataFrame]) -> list[dict[str, object]]:
    """Normalize rows captured from the Streamlit data editor widget."""

    if df is None or not isinstance(df, pd.DataFrame):
        return []
    try:
        records = df.to_dict("records")
    except Exception:
        return []
    return _normalize_grid_rows(records)


def _summarize_grid_column(rows: Iterable[dict[str, object]], key: str) -> Optional[str]:
    """Combine a grid column into a legacy text summary for backwards compatibility."""

    values: list[str] = []
    seen: set[str] = set()
    for row in rows or []:
        text = clean_text(row.get(key))
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        values.append(text)
    if not values:
        return None
    return "\n".join(values)

NOTIFICATION_BUFFER_KEY = "runtime_notifications"
MAX_RUNTIME_NOTIFICATIONS = 40
ACTIVITY_FEED_LIMIT = 25

NOTIFICATION_EVENT_LABELS = {
    "customer_created": "Customer added",
    "customer_updated": "Customer updated",
    "customer_deleted": "Customer removed",
    "service_created": "Service created",
    "service_updated": "Service updated",
    "maintenance_created": "Maintenance created",
    "maintenance_updated": "Maintenance updated",
    "warranty_updated": "Warranty updated",
    "report_submitted": "Report submitted",
    "report_updated": "Report updated",
}


def customer_complete_clause(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return " AND ".join(
        [
            f"TRIM(COALESCE({prefix}name, '')) <> ''",
            f"TRIM(COALESCE({prefix}phone, '')) <> ''",
            f"TRIM(COALESCE({prefix}address, '')) <> ''",
        ]
    )


def customer_incomplete_clause(alias: str = "") -> str:
    return f"NOT ({customer_complete_clause(alias)})"

# ---------- Schema ----------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    pass_hash TEXT,
    role TEXT DEFAULT 'staff',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    company_name TEXT,
    phone TEXT,
    email TEXT,
    address TEXT,
    delivery_address TEXT,
    remarks TEXT,
    purchase_date TEXT,
    product_info TEXT,
    delivery_order_code TEXT,
    sales_person TEXT,
    amount_spent REAL,
    created_by INTEGER,
    attachment_path TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    dup_flag INTEGER DEFAULT 0,
    FOREIGN KEY(created_by) REFERENCES users(user_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    model TEXT,
    serial TEXT,
    dup_flag INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    order_date TEXT,
    delivery_date TEXT,
    notes TEXT,
    dup_flag INTEGER DEFAULT 0,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER DEFAULT 1,
    FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS warranties (
    warranty_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product_id INTEGER,
    serial TEXT,
    issue_date TEXT,
    expiry_date TEXT,
    status TEXT DEFAULT 'active',
    remarks TEXT,
    dup_flag INTEGER DEFAULT 0,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS delivery_orders (
    do_number TEXT PRIMARY KEY,
    customer_id INTEGER,
    order_id INTEGER,
    description TEXT,
    sales_person TEXT,
    remarks TEXT,
    file_path TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS services (
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    do_number TEXT,
    customer_id INTEGER,
    service_date TEXT,
    service_start_date TEXT,
    service_end_date TEXT,
    description TEXT,
    status TEXT DEFAULT 'In progress',
    remarks TEXT,
    service_product_info TEXT,
    condition_status TEXT,
    condition_remarks TEXT,
    bill_amount REAL,
    bill_document_path TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(do_number) REFERENCES delivery_orders(do_number) ON DELETE SET NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS maintenance_records (
    maintenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    do_number TEXT,
    customer_id INTEGER,
    maintenance_date TEXT,
    maintenance_start_date TEXT,
    maintenance_end_date TEXT,
    description TEXT,
    status TEXT DEFAULT 'In progress',
    remarks TEXT,
    maintenance_product_info TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(do_number) REFERENCES delivery_orders(do_number) ON DELETE SET NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS service_documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_id INTEGER,
    file_path TEXT,
    original_name TEXT,
    uploaded_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(service_id) REFERENCES services(service_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS maintenance_documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    maintenance_id INTEGER,
    file_path TEXT,
    original_name TEXT,
    uploaded_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(maintenance_id) REFERENCES maintenance_records(maintenance_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS customer_notes (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    note TEXT,
    remind_on TEXT,
    is_done INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_customer_notes_customer ON customer_notes(customer_id);
CREATE INDEX IF NOT EXISTS idx_customer_notes_remind ON customer_notes(remind_on, is_done);
CREATE TABLE IF NOT EXISTS import_history (
    import_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product_id INTEGER,
    order_id INTEGER,
    order_item_id INTEGER,
    warranty_id INTEGER,
    do_number TEXT,
    import_tag TEXT,
    imported_at TEXT DEFAULT (datetime('now')),
    original_date TEXT,
    customer_name TEXT,
    address TEXT,
    phone TEXT,
    product_label TEXT,
    notes TEXT,
    amount_spent REAL,
    imported_by INTEGER,
    deleted_at TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE SET NULL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE SET NULL,
    FOREIGN KEY(order_item_id) REFERENCES order_items(order_item_id) ON DELETE SET NULL,
    FOREIGN KEY(warranty_id) REFERENCES warranties(warranty_id) ON DELETE SET NULL,
    FOREIGN KEY(imported_by) REFERENCES users(user_id) ON DELETE SET NULL
);
CREATE TABLE IF NOT EXISTS work_reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    period_type TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    tasks TEXT,
    remarks TEXT,
    research TEXT,
    grid_payload TEXT,
    attachment_path TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_work_reports_user_period ON work_reports(user_id, period_type, period_start);
CREATE INDEX IF NOT EXISTS idx_work_reports_period ON work_reports(period_type, period_start, period_end);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_work_reports_user_period ON work_reports(user_id, period_type, period_start);
CREATE TABLE IF NOT EXISTS needs (
    need_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    product TEXT,
    unit TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS activity_log (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    event_type TEXT,
    entity_type TEXT,
    entity_id INTEGER,
    description TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_activity_log_created ON activity_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_log_entity ON activity_log(entity_type, entity_id);
"""

# ---------- Helpers ----------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_schema(conn):
    ensure_upload_dirs()
    conn.executescript(SCHEMA_SQL)
    ensure_schema_upgrades(conn)
    conn.commit()
    # bootstrap admin if empty
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        admin_user = os.getenv("ADMIN_USER", "admin")
        admin_pass = os.getenv("ADMIN_PASS", "admin123")
        h = hashlib.sha256(admin_pass.encode("utf-8")).hexdigest()
        conn.execute("INSERT INTO users (username, pass_hash, role) VALUES (?, ?, 'admin')", (admin_user, h))
        conn.commit()


def ensure_schema_upgrades(conn):
    def has_column(table: str, column: str) -> bool:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return any(str(row[1]) == column for row in cur.fetchall())

    def add_column(table: str, column: str, definition: str) -> None:
        if not has_column(table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    add_column("customers", "company_name", "TEXT")
    add_column("customers", "delivery_address", "TEXT")
    add_column("customers", "remarks", "TEXT")
    add_column("customers", "purchase_date", "TEXT")
    add_column("customers", "product_info", "TEXT")
    add_column("customers", "delivery_order_code", "TEXT")
    add_column("customers", "attachment_path", "TEXT")
    add_column("customers", "sales_person", "TEXT")
    add_column("customers", "amount_spent", "REAL")
    add_column("customers", "created_by", "INTEGER")
    add_column("services", "status", "TEXT DEFAULT 'In progress'")
    add_column("services", "service_start_date", "TEXT")
    add_column("services", "service_end_date", "TEXT")
    add_column("services", "service_product_info", "TEXT")
    add_column("services", "condition_status", "TEXT")
    add_column("services", "condition_remarks", "TEXT")
    add_column("services", "bill_amount", "REAL")
    add_column("services", "bill_document_path", "TEXT")
    add_column("maintenance_records", "status", "TEXT DEFAULT 'In progress'")
    add_column("maintenance_records", "maintenance_start_date", "TEXT")
    add_column("maintenance_records", "maintenance_end_date", "TEXT")
    add_column("maintenance_records", "maintenance_product_info", "TEXT")
    add_column("warranties", "remarks", "TEXT")
    add_column("delivery_orders", "remarks", "TEXT")
    add_column("import_history", "amount_spent", "REAL")
    add_column("import_history", "imported_by", "INTEGER")
    add_column("work_reports", "grid_payload", "TEXT")
    add_column("work_reports", "attachment_path", "TEXT")

    cur = conn.execute(
        """
        SELECT report_id, user_id, LOWER(COALESCE(period_type, '')), COALESCE(period_start, '')
        FROM work_reports
        ORDER BY user_id, LOWER(COALESCE(period_type, '')), COALESCE(period_start, ''), report_id DESC
        """
    )
    seen_keys: set[tuple[int, str, str]] = set()
    duplicates: list[int] = []
    for report_id, user_id, period_type, period_start in cur.fetchall():
        key = (int(user_id), period_type or "", period_start or "")
        if key in seen_keys:
            duplicates.append(int(report_id))
        else:
            seen_keys.add(key)
    for report_id in duplicates:
        conn.execute("DELETE FROM work_reports WHERE report_id=?", (int(report_id),))

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            note TEXT,
            remind_on TEXT,
            is_done INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_customer_notes_customer ON customer_notes(customer_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_customer_notes_remind ON customer_notes(remind_on, is_done)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_customers_created_by ON customers(created_by)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_import_history_imported_by ON import_history(imported_by)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS work_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            period_type TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            tasks TEXT,
            remarks TEXT,
            research TEXT,
            attachment_path TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_work_reports_user_period ON work_reports(user_id, period_type, period_start)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_work_reports_period ON work_reports(period_type, period_start, period_end)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uniq_work_reports_user_period ON work_reports(user_id, period_type, period_start)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_log (
            activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event_type TEXT,
            entity_type TEXT,
            entity_id INTEGER,
            description TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_log_created ON activity_log(created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_log_entity ON activity_log(entity_type, entity_id)"
    )


def get_current_user() -> dict:
    return st.session_state.get("user") or {}


def current_user_id() -> Optional[int]:
    user = get_current_user()
    try:
        return int(user.get("user_id")) if user.get("user_id") is not None else None
    except (TypeError, ValueError):
        return None


def current_user_is_admin() -> bool:
    return get_current_user().get("role") == "admin"


def customer_scope_filter(alias: str = "") -> tuple[str, tuple[object, ...]]:
    user = get_current_user()
    if not user or user.get("role") == "admin":
        return "", ()
    user_id = current_user_id()
    if user_id is None:
        return "1=0", ()
    prefix = f"{alias}." if alias else ""
    return f"{prefix}created_by = ?", (user_id,)


def accessible_customer_ids(conn) -> Optional[set[int]]:
    if current_user_is_admin():
        return None
    user_id = current_user_id()
    if user_id is None:
        return set()
    df = df_query(conn, "SELECT customer_id FROM customers WHERE created_by=?", (user_id,))
    if df.empty:
        return set()
    ids: set[int] = set()
    for value in df["customer_id"].dropna().tolist():
        try:
            ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def df_query(conn, q, params=()):
    return pd.read_sql_query(q, conn, params=params)

def fmt_dates(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(DATE_FMT)
    return df


def add_months(base: date, months: int) -> date:
    """Return ``base`` shifted by ``months`` while clamping the day to the target month."""

    if not isinstance(base, date):
        raise TypeError("base must be a date instance")
    try:
        months = int(months)
    except (TypeError, ValueError):
        raise TypeError("months must be an integer") from None

    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    day = min(base.day, monthrange(year, month)[1])
    return date(year, month, day)


def month_bucket_counts(
    conn,
    table: str,
    date_column: str,
    *,
    where: Optional[str] = None,
    params: Optional[Iterable[object]] = None,
) -> tuple[int, int]:
    """Return the current and previous month counts for ``table.date_column``."""

    params = tuple(params or ())
    criteria = [f"{date_column} IS NOT NULL"]
    if where:
        criteria.append(f"({where})")
    where_clause = " AND ".join(criteria)
    query = dedent(
        f"""
        SELECT
            SUM(CASE WHEN strftime('%Y-%m', {date_column}) = strftime('%Y-%m', 'now') THEN 1 ELSE 0 END) AS current_month,
            SUM(CASE WHEN strftime('%Y-%m', {date_column}) = strftime('%Y-%m', date('now', '-1 month')) THEN 1 ELSE 0 END) AS previous_month
        FROM {table}
        WHERE {where_clause}
        """
    )
    cur = conn.execute(query, params)
    row = cur.fetchone()
    if not row:
        return 0, 0
    current, previous = row
    current_count = int(current or 0)
    previous_count = int(previous or 0)
    return current_count, previous_count


def format_metric_delta(current: int, previous: int) -> str:
    """Format a delta label comparing the current value to the previous month."""

    diff = int(current) - int(previous)
    if diff == 0:
        return "On par with last month"
    if previous == 0:
        return f"+{current} (new this month)"
    pct = (diff / previous) * 100
    return f"{diff:+d} ({pct:+.1f}%) vs last month"


def upcoming_warranty_projection(conn, months_ahead: int = 6) -> pd.DataFrame:
    """Return a month-by-month projection of expiring active warranties."""

    try:
        months = int(months_ahead)
    except (TypeError, ValueError):
        months = 6
    months = max(1, min(months, 24))

    today = date.today()
    start_month = date(today.year, today.month, 1)
    last_bucket = add_months(start_month, months - 1)
    last_day = monthrange(last_bucket.year, last_bucket.month)[1]
    range_end = date(last_bucket.year, last_bucket.month, last_day)

    scope_clause, scope_params = customer_scope_filter("c")
    projection = df_query(
        conn,
        dedent(
            """
            SELECT strftime('%Y-%m', w.expiry_date) AS month_bucket,
                   COUNT(*) AS total
            FROM warranties w
            LEFT JOIN customers c ON c.customer_id = w.customer_id
            WHERE w.status='active'
              AND w.expiry_date IS NOT NULL
              AND date(w.expiry_date) BETWEEN date(?) AND date(?)
              {scope_filter}
            GROUP BY month_bucket
            ORDER BY month_bucket
            """
        ).format(scope_filter=f" AND {scope_clause}" if scope_clause else ""),
        params=(start_month.isoformat(), range_end.isoformat(), *scope_params),
    )

    records: list[dict[str, object]] = []
    current_bucket = start_month
    while len(records) < months:
        bucket_key = current_bucket.strftime("%Y-%m")
        label = current_bucket.strftime("%b %Y")
        matching = projection[projection["month_bucket"] == bucket_key]
        if matching.empty:
            count = 0
        else:
            count = int(matching.iloc[0]["total"] or 0)
        records.append({"Month": label, "Expiring warranties": count})
        current_bucket = add_months(current_bucket, 1)

    return pd.DataFrame(records)


def upcoming_warranty_breakdown(
    conn, days_ahead: int = 60, group_by: str = "sales_person"
) -> pd.DataFrame:
    """Summarise upcoming expiries grouped by a chosen dimension."""

    try:
        days = int(days_ahead)
    except (TypeError, ValueError):
        days = 60
    days = max(1, min(days, 365))

    grouping_options = {
        "sales_person": (
            "COALESCE(NULLIF(TRIM(c.sales_person), ''), 'Unassigned')",
            "Sales person",
        ),
        "customer": (
            "COALESCE(NULLIF(TRIM(c.name), ''), '(Unknown customer)')",
            "Customer",
        ),
        "product": (
            "COALESCE(NULLIF(TRIM(COALESCE(p.name, '') || CASE WHEN p.model IS NULL OR TRIM(p.model) = '' THEN '' ELSE ' ' || p.model END), ''), '(Unspecified product)')",
            "Product",
        ),
    }

    normalized_group = (group_by or "sales_person").lower()
    group_expr, column_label = grouping_options.get(
        normalized_group, grouping_options["sales_person"]
    )

    today = date.today()
    range_end = today + timedelta(days=days)

    scope_clause, scope_params = customer_scope_filter("c")
    scope_filter = f" AND {scope_clause}" if scope_clause else ""
    breakdown = df_query(
        conn,
        dedent(
            f"""
            SELECT {group_expr} AS bucket,
                   COUNT(*) AS total
            FROM warranties w
            LEFT JOIN customers c ON c.customer_id = w.customer_id
            LEFT JOIN products p ON p.product_id = w.product_id
            WHERE w.status='active'
              AND w.expiry_date IS NOT NULL
              AND date(w.expiry_date) BETWEEN date(?) AND date(?)
              {scope_filter}
            GROUP BY bucket
            ORDER BY total DESC, bucket ASC
            """
        ),
        params=(today.isoformat(), range_end.isoformat(), *scope_params),
    )

    if breakdown.empty:
        return pd.DataFrame(columns=[column_label, "Expiring warranties"])

    renamed = breakdown.rename(
        columns={"bucket": column_label, "total": "Expiring warranties"}
    )
    renamed["Expiring warranties"] = renamed["Expiring warranties"].astype(int)
    return renamed


def clean_text(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = str(value).strip()
    return value or None


def _parse_sqlite_timestamp(value: Optional[str]) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_time_ago(value: Optional[str]) -> str:
    timestamp = _parse_sqlite_timestamp(value)
    if not timestamp:
        return clean_text(value) or ""
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)
    seconds = max(int((datetime.utcnow() - timestamp).total_seconds()), 0)
    if seconds < 5:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks}w ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"


def _notification_store() -> list[dict[str, object]]:
    buffer = st.session_state.get(NOTIFICATION_BUFFER_KEY)
    if not isinstance(buffer, list):
        buffer = []
    st.session_state[NOTIFICATION_BUFFER_KEY] = buffer
    return buffer


def get_runtime_notifications() -> list[dict[str, object]]:
    return list(_notification_store())


def push_runtime_notification(
    title: Optional[str],
    message: Optional[str],
    *,
    severity: str = "info",
    details: Optional[Iterable[str]] = None,
) -> None:
    if not title and not message:
        return
    entry = {
        "title": clean_text(title) or "Notification",
        "message": clean_text(message) or "",
        "severity": (clean_text(severity) or "info").lower(),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "details": [
            clean_text(item) for item in (details or []) if clean_text(item)
        ],
    }
    buffer = _notification_store()
    buffer.append(entry)
    if len(buffer) > MAX_RUNTIME_NOTIFICATIONS:
        del buffer[0 : len(buffer) - MAX_RUNTIME_NOTIFICATIONS]
    st.session_state[NOTIFICATION_BUFFER_KEY] = buffer


def log_activity(
    conn,
    *,
    event_type: Optional[str],
    description: Optional[str],
    entity_type: Optional[str] = None,
    entity_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> None:
    event_key = clean_text(event_type)
    description_text = clean_text(description)
    if not event_key and not description_text:
        return
    actor_id = user_id if user_id is not None else current_user_id()
    try:
        conn.execute(
            """
            INSERT INTO activity_log (user_id, event_type, entity_type, entity_id, description, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                actor_id,
                event_key,
                clean_text(entity_type),
                entity_id,
                description_text or description or "",
            ),
        )
        conn.commit()
    except sqlite3.Error:
        with contextlib.suppress(Exception):
            conn.rollback()


def fetch_activity_feed(conn, limit: int = ACTIVITY_FEED_LIMIT) -> list[dict[str, object]]:
    try:
        resolved_limit = int(limit)
    except (TypeError, ValueError):
        resolved_limit = ACTIVITY_FEED_LIMIT
    resolved_limit = max(1, min(resolved_limit, 100))
    df = df_query(
        conn,
        dedent(
            """
            SELECT a.activity_id,
                   a.event_type,
                   a.entity_type,
                   a.entity_id,
                   a.description,
                   a.created_at,
                   u.username
            FROM activity_log a
            LEFT JOIN users u ON u.user_id = a.user_id
            ORDER BY datetime(a.created_at) DESC, a.activity_id DESC
            LIMIT ?
            """
        ),
        (resolved_limit,),
    )
    if df.empty:
        return []
    feed: list[dict[str, object]] = []
    for record in df.to_dict("records"):
        event_type = clean_text(record.get("event_type")) or "activity"
        label = NOTIFICATION_EVENT_LABELS.get(
            event_type, event_type.replace("_", " ").title()
        )
        feed.append(
            {
                "title": label,
                "message": clean_text(record.get("description")) or "",
                "timestamp": clean_text(record.get("created_at")) or "",
                "actor": clean_text(record.get("username")) or "Team member",
                "severity": "info",
                "event_type": event_type,
            }
        )
    return feed


def to_iso_date(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    if isinstance(value, datetime):
        return value.date().strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    try:
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        if len(parsed) == 0:
            return None
        parsed = parsed[0]
    return pd.Timestamp(parsed).normalize().strftime("%Y-%m-%d")


def format_money(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    symbol = CURRENCY_SYMBOL.strip()
    if symbol:
        return f"{symbol} {amount:,.2f}"
    return f"{amount:,.2f}"


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def parse_amount(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        amount = float(value)
        return round(amount, 2) if amount != 0 else amount
    text = clean_text(value)
    if not text:
        return None
    normalized = re.sub(r"[^0-9.\-]", "", text)
    if normalized in {"", ".", "-", "-."}:
        return None
    try:
        amount = float(normalized)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount):
        return None
    if math.isclose(amount, 0.0):
        return 0.0
    return round(amount, 2)


def format_period_label(period_type: str) -> str:
    if not period_type:
        return "Unknown"
    key = str(period_type).strip().lower()
    return REPORT_PERIOD_OPTIONS.get(key, key.title())


def format_period_range(start: Optional[str], end: Optional[str]) -> str:
    def _label(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.DatetimeIndex):
            if len(parsed) == 0:
                return None
            parsed = parsed[0]
        return pd.Timestamp(parsed).strftime(DATE_FMT)

    start_label = _label(start)
    end_label = _label(end)
    if start_label and end_label:
        if start_label == end_label:
            return start_label
        return f"{start_label} → {end_label}"
    return start_label or end_label or "—"


def _clamp_percentage(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 100.0:
        return 100.0
    return value


def _value_or_default(value: object, default: object) -> object:
    if value is None:
        return default
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return default
    except Exception:
        pass
    if isinstance(value, str) and value.strip() == "":
        return default
    return value


def normalize_product_entries(
    entries: Iterable[dict[str, object]]
) -> tuple[list[dict[str, object]], list[str]]:
    cleaned: list[dict[str, object]] = []
    labels: list[str] = []
    for entry in entries:
        name_clean = clean_text(entry.get("name")) if isinstance(entry, dict) else None
        model_clean = clean_text(entry.get("model")) if isinstance(entry, dict) else None
        serial_clean = clean_text(entry.get("serial")) if isinstance(entry, dict) else None
        quantity_raw = entry.get("quantity") if isinstance(entry, dict) else None
        qty_val = _coerce_float(quantity_raw, 1.0)
        try:
            qty_val_int = int(round(qty_val))
        except Exception:
            qty_val_int = 1
        qty_val = max(qty_val_int, 1)
        if not any([name_clean, model_clean, serial_clean]):
            continue
        cleaned.append(
            {
                "name": name_clean,
                "model": model_clean,
                "serial": serial_clean,
                "quantity": qty_val,
            }
        )
        label_parts = [val for val in [name_clean, model_clean] if val]
        label = " - ".join(label_parts)
        if qty_val > 1:
            label = f"{label} ×{qty_val}" if label else f"×{qty_val}"
        if serial_clean:
            label = f"{label} (Serial: {serial_clean})" if label else f"Serial: {serial_clean}"
        if label:
            labels.append(label)
    return cleaned, labels


def normalize_quotation_items(
    entries: Iterable[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    cleaned: list[dict[str, object]] = []
    totals = {
        "gross_total": 0.0,
        "discount_total": 0.0,
        "taxable_total": 0.0,
        "cgst_total": 0.0,
        "sgst_total": 0.0,
        "igst_total": 0.0,
        "grand_total": 0.0,
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        description = clean_text(entry.get("description"))
        if not description:
            continue

        hsn = clean_text(entry.get("hsn"))
        unit = clean_text(entry.get("unit"))
        quantity = max(_coerce_float(entry.get("quantity"), 0.0), 0.0)
        rate = max(_coerce_float(entry.get("rate"), 0.0), 0.0)
        discount_pct = _clamp_percentage(_coerce_float(entry.get("discount"), 0.0))
        cgst_pct = _clamp_percentage(_coerce_float(entry.get("cgst"), 0.0))
        sgst_pct = _clamp_percentage(_coerce_float(entry.get("sgst"), 0.0))
        igst_pct = _clamp_percentage(_coerce_float(entry.get("igst"), 0.0))

        gross_amount = quantity * rate
        discount_amount = gross_amount * (discount_pct / 100.0)
        taxable_value = max(gross_amount - discount_amount, 0.0)
        cgst_amount = taxable_value * (cgst_pct / 100.0)
        sgst_amount = taxable_value * (sgst_pct / 100.0)
        igst_amount = taxable_value * (igst_pct / 100.0)
        line_total = taxable_value + cgst_amount + sgst_amount + igst_amount

        item = {
            "Item": len(cleaned) + 1,
            "Description": description,
            "HSN/SAC": hsn,
            "Unit": unit,
            "Quantity": quantity,
            "Rate": rate,
            "Gross amount": gross_amount,
            "Discount (%)": discount_pct,
            "Discount amount": discount_amount,
            "Taxable value": taxable_value,
            "CGST (%)": cgst_pct,
            "CGST amount": cgst_amount,
            "SGST (%)": sgst_pct,
            "SGST amount": sgst_amount,
            "IGST (%)": igst_pct,
            "IGST amount": igst_amount,
            "Line total": line_total,
        }
        cleaned.append(item)

        totals["gross_total"] += gross_amount
        totals["discount_total"] += discount_amount
        totals["taxable_total"] += taxable_value
        totals["cgst_total"] += cgst_amount
        totals["sgst_total"] += sgst_amount
        totals["igst_total"] += igst_amount
        totals["grand_total"] += line_total

    return cleaned, totals


def format_period_span(
    start: Optional[str], end: Optional[str], *, joiner: str = " → "
) -> Optional[str]:
    start_clean = clean_text(start)
    end_clean = clean_text(end)
    if not start_clean and not end_clean:
        return None
    if start_clean and end_clean:
        if start_clean == end_clean:
            return start_clean
        return f"{start_clean}{joiner}{end_clean}"
    return start_clean or end_clean


def get_status_choice(prefix: str, fallback: str = DEFAULT_SERVICE_STATUS) -> str:
    choice = st.session_state.get(f"{prefix}_status_choice", fallback)
    if isinstance(choice, str) and choice in SERVICE_STATUS_OPTIONS:
        return choice
    return fallback


def ensure_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().date()
    except Exception:
        pass
    try:
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    except Exception:
        parsed = None
    if parsed is None or pd.isna(parsed):
        return None
    if isinstance(parsed, pd.DatetimeIndex) and len(parsed) > 0:
        parsed = parsed[0]
    if isinstance(parsed, datetime):
        return parsed.date()
    try:
        return parsed.to_pydatetime().date()
    except Exception:
        return None


def determine_period_dates(
    status_choice: str, raw_value
) -> tuple[Optional[date], Optional[date], Optional[date]]:
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    if status_choice == "Completed":
        raw_items: list[Optional[date]]
        if isinstance(raw_value, (list, tuple)):
            raw_items = [ensure_date(v) for v in raw_value]
        else:
            raw_items = [ensure_date(raw_value)]
        clean_items = [item for item in raw_items if item is not None]
        if clean_items:
            start_date = clean_items[0]
            end_date = clean_items[-1]
            if end_date is None:
                end_date = start_date
            if start_date and end_date and end_date < start_date:
                start_date, end_date = end_date, start_date
    else:
        if isinstance(raw_value, (list, tuple)):
            raw_value = raw_value[0] if raw_value else None
        start_date = ensure_date(raw_value)
        end_date = None
    primary_date = start_date or end_date
    return primary_date, start_date, end_date


def determine_period_strings(
    status_choice: str, raw_value
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    primary_date, start_date, end_date = determine_period_dates(status_choice, raw_value)

    def _to_str(value: Optional[date]) -> Optional[str]:
        return value.strftime("%Y-%m-%d") if value else None

    return _to_str(primary_date), _to_str(start_date), _to_str(end_date)


def is_pending_status(status: Optional[str]) -> bool:
    text = clean_text(status)
    if not text:
        return True
    normalized = text.lower()
    return normalized not in {"completed", "in progress"}


def status_input_widget(prefix: str, default_status: Optional[str] = None) -> str:
    lookup = {opt.lower(): opt for opt in SERVICE_STATUS_OPTIONS}
    default_choice = DEFAULT_SERVICE_STATUS
    custom_default = "Haven't started"
    default_clean = clean_text(default_status)
    if default_clean:
        normalized = default_clean.lower()
        if normalized in lookup and lookup[normalized] != "Haven't started":
            default_choice = lookup[normalized]
        elif normalized == "haven't started":
            default_choice = lookup[normalized]
            custom_default = lookup[normalized]
        else:
            default_choice = "Haven't started"
            custom_default = default_clean

    choice = st.selectbox(
        "Status",
        SERVICE_STATUS_OPTIONS,
        index=SERVICE_STATUS_OPTIONS.index(default_choice),
        key=f"{prefix}_status_choice",
    )
    if choice == "Haven't started":
        custom_value = st.text_input(
            "Custom status label",
            value=custom_default or "Haven't started",
            key=f"{prefix}_status_custom",
            help="Customize the saved status when a record hasn't started yet.",
        )
        return clean_text(custom_value) or "Haven't started"
    return choice


def link_delivery_order_to_customer(
    conn: sqlite3.Connection, do_number: Optional[str], customer_id: Optional[int]
) -> None:
    do_serial = clean_text(do_number)
    if not do_serial:
        return
    cur = conn.cursor()
    row = cur.execute(
        "SELECT customer_id FROM delivery_orders WHERE do_number = ?",
        (do_serial,),
    ).fetchone()
    if row is None:
        if customer_id is not None:
            cur.execute(
                "UPDATE customers SET delivery_order_code = ? WHERE customer_id = ?",
                (do_serial, int(customer_id)),
            )
        return
    previous_customer = int(row[0]) if row and row[0] is not None else None
    if customer_id is not None:
        cur.execute(
            "UPDATE delivery_orders SET customer_id = ? WHERE do_number = ?",
            (int(customer_id), do_serial),
        )
        cur.execute(
            "UPDATE customers SET delivery_order_code = ? WHERE customer_id = ?",
            (do_serial, int(customer_id)),
        )
    else:
        cur.execute(
            "UPDATE delivery_orders SET customer_id = NULL WHERE do_number = ?",
            (do_serial,),
        )
    if previous_customer and previous_customer != (int(customer_id) if customer_id is not None else None):
        cur.execute(
            "UPDATE customers SET delivery_order_code = NULL WHERE customer_id = ? AND delivery_order_code = ?",
            (previous_customer, do_serial),
        )


def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def _default_new_customer_products() -> list[dict[str, object]]:
    return [{"name": "", "model": "", "serial": "", "quantity": 1}]


def _reset_new_customer_form_state() -> None:
    default_products = _default_new_customer_products()
    st.session_state["new_customer_products_rows"] = default_products
    for key in [
        "new_customer_name",
        "new_customer_company",
        "new_customer_phone",
        "new_customer_address",
        "new_customer_delivery_address",
        "new_customer_purchase_date",
        "new_customer_do_code",
        "new_customer_sales_person",
        "new_customer_remarks",
        "new_customer_amount_spent",
        "new_customer_pdf",
        "new_customer_do_pdf",
        "new_customer_products_table",
    ]:
        st.session_state.pop(key, None)


def _default_quotation_items() -> list[dict[str, object]]:
    return [
        {
            "description": "",
            "hsn": "",
            "unit": "",
            "quantity": 1.0,
            "rate": 0.0,
            "discount": 0.0,
            "cgst": 0.0,
            "sgst": 0.0,
            "igst": 0.0,
        }
    ]


def _reset_quotation_form_state() -> None:
    default_items = _default_quotation_items()
    st.session_state["quotation_item_rows"] = default_items
    st.session_state["quotation_items_table"] = pd.DataFrame(default_items)
    for key in [
        "quotation_reference",
        "quotation_date",
        "quotation_prepared_by",
        "quotation_valid_days",
        "quotation_round_total",
        "quotation_company_name",
        "quotation_company_details",
        "quotation_customer_name",
        "quotation_customer_contact",
        "quotation_customer_address",
        "quotation_project_name",
        "quotation_subject",
        "quotation_scope_notes",
        "quotation_terms",
        "quotation_default_discount",
        "quotation_default_cgst",
        "quotation_default_sgst",
        "quotation_default_igst",
    ]:
        st.session_state.pop(key, None)
    st.session_state.pop("quotation_result", None)


def _streamlit_runtime_active() -> bool:
    """Return True when running inside a Streamlit runtime."""

    runtime = None
    try:
        from streamlit import runtime as st_runtime

        runtime = st_runtime
    except Exception:
        runtime = None

    if runtime is not None:
        try:
            if runtime.exists():
                return True
        except Exception:
            pass

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False

    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


def ensure_upload_dirs():
    for path in (
        UPLOADS_DIR,
        DELIVERY_ORDER_DIR,
        SERVICE_DOCS_DIR,
        MAINTENANCE_DOCS_DIR,
        CUSTOMER_DOCS_DIR,
        SERVICE_BILL_DIR,
        REPORT_DOCS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, target_dir: Path, filename: Optional[str] = None) -> Optional[Path]:
    if uploaded_file is None:
        return None
    ensure_upload_dirs()
    safe_name = filename or uploaded_file.name
    safe_name = "".join(ch for ch in safe_name if ch.isalnum() or ch in (".", "_", "-"))
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    dest = target_dir / safe_name
    counter = 1
    while dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.read())
    return dest


def store_uploaded_pdf(uploaded_file, target_dir: Path, filename: Optional[str] = None) -> Optional[str]:
    """Persist an uploaded PDF and return its path relative to ``BASE_DIR``.

    Streamlit's ``UploadedFile`` objects expose a ``read`` method and ``name``
    attribute. This helper mirrors ``save_uploaded_file`` but normalises the
    resulting path so callers can safely stash it in the database without
    worrying about absolute paths or platform differences.
    """

    saved_path = save_uploaded_file(uploaded_file, target_dir, filename=filename)
    if not saved_path:
        return None
    try:
        return str(saved_path.relative_to(BASE_DIR))
    except ValueError:
        return str(saved_path)


def store_report_attachment(uploaded_file, *, identifier: Optional[str] = None) -> Optional[str]:
    """Persist a supporting document for a work report."""

    if uploaded_file is None:
        return None

    ensure_upload_dirs()
    raw_name = uploaded_file.name or "attachment"
    allowed_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
    suffix = Path(raw_name).suffix.lower()
    if suffix not in allowed_exts:
        suffix = ".pdf"
    stem = Path(raw_name).stem
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in ("_", "-")) or "attachment"
    safe_stem = safe_stem.strip("_") or "attachment"
    if identifier:
        ident = "".join(ch for ch in identifier if ch.isalnum() or ch in ("_", "-"))
        if ident:
            safe_stem = f"{ident}_{safe_stem}"
    dest = REPORT_DOCS_DIR / f"{safe_stem}{suffix}"
    counter = 1
    while dest.exists():
        dest = REPORT_DOCS_DIR / f"{safe_stem}_{counter}{suffix}"
        counter += 1
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.read())
    try:
        return str(dest.relative_to(BASE_DIR))
    except ValueError:
        return str(dest)


def resolve_upload_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


_ATTACHMENT_UNCHANGED = object()


def normalize_report_window(period_type: str, start_value, end_value) -> tuple[str, date, date]:
    """Return a canonical report period and date window."""

    key = (period_type or "").strip().lower()
    if key not in REPORT_PERIOD_OPTIONS:
        key = "daily"

    def _coerce(value) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        iso = to_iso_date(value)
        if iso:
            try:
                return datetime.strptime(iso, "%Y-%m-%d").date()
            except ValueError:
                pass
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            parsed = None
        if parsed is None or pd.isna(parsed):
            return None
        if isinstance(parsed, pd.DatetimeIndex):
            if len(parsed) == 0:
                return None
            parsed = parsed[0]
        if hasattr(parsed, "to_pydatetime"):
            parsed = parsed.to_pydatetime()
        if isinstance(parsed, datetime):
            return parsed.date()
        if isinstance(parsed, date):
            return parsed
        return None

    start_date = _coerce(start_value)
    end_date = _coerce(end_value)

    if key == "daily":
        anchor = start_date or end_date
        if anchor is None:
            raise ValueError("Select a date for the daily report.")
        start_date = end_date = anchor
    elif key == "weekly":
        anchor = start_date or end_date
        if anchor is None:
            raise ValueError("Select a week for the report.")
        start_date = anchor - timedelta(days=anchor.weekday())
        end_date = start_date + timedelta(days=6)
    else:
        anchor = start_date or end_date
        if anchor is None:
            raise ValueError("Select a month for the report.")
        start_date = anchor.replace(day=1)
        last_day = monthrange(start_date.year, start_date.month)[1]
        end_date = date(start_date.year, start_date.month, last_day)

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    return key, start_date, end_date


def _sanitize_path_component(value: Optional[str]) -> str:
    if not value:
        return "item"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.() ")
    cleaned = "".join(ch if ch in allowed else "_" for ch in str(value))
    cleaned = cleaned.strip()
    return cleaned or "item"


def build_customer_groups(conn, only_complete: bool = True):
    criteria = []
    params: list[object] = []
    if only_complete:
        criteria.append(customer_complete_clause())
    scope_clause, scope_params = customer_scope_filter()
    if scope_clause:
        criteria.append(scope_clause)
        params.extend(scope_params)
    where_clause = f"WHERE {' AND '.join(criteria)}" if criteria else ""
    df = df_query(
        conn,
        f"SELECT customer_id, TRIM(name) AS name FROM customers {where_clause}",
        tuple(params),
    )
    if df.empty:
        return [], {}
    df["name"] = df["name"].fillna("")
    df["norm_name"] = df["name"].astype(str).str.strip()
    df.sort_values(by=["norm_name", "customer_id"], inplace=True)
    groups = []
    label_by_id = {}
    for norm_name, group in df.groupby("norm_name", sort=False):
        ids = group["customer_id"].astype(int).tolist()
        primary = ids[0]
        raw_name = clean_text(group.iloc[0].get("name"))
        count = len(ids)
        base_label = raw_name or f"Customer #{primary}"
        if raw_name and count > 1:
            display_label = f"{base_label} ({count} records)"
        else:
            display_label = base_label
        groups.append(
            {
                "norm_name": norm_name,
                "primary_id": primary,
                "ids": ids,
                "raw_name": raw_name,
                "label": display_label,
                "count": count,
            }
        )
        for cid in ids:
            label_by_id[int(cid)] = display_label
    groups.sort(key=lambda g: (g["norm_name"] or "").lower())
    return groups, label_by_id


def fetch_customer_choices(conn):
    groups, label_by_id = build_customer_groups(conn, only_complete=True)
    options = [None]
    labels = {None: "-- Select customer --"}
    group_map = {}
    for group in groups:
        primary = group["primary_id"]
        options.append(primary)
        labels[primary] = group["label"]
        group_map[primary] = group["ids"]
    return options, labels, group_map, label_by_id


def attach_documents(
    conn,
    table: str,
    fk_column: str,
    record_id: int,
    files,
    target_dir: Path,
    prefix: str,
):
    if not files:
        return 0
    saved = 0
    for idx, uploaded in enumerate(files, start=1):
        if uploaded is None:
            continue
        original_name = uploaded.name or f"{prefix}_{idx}.pdf"
        safe_original = Path(original_name).name
        filename = f"{prefix}_{idx}_{safe_original}"
        stored_path = store_uploaded_pdf(uploaded, target_dir, filename=filename)
        if not stored_path:
            continue
        conn.execute(
            f"INSERT INTO {table} ({fk_column}, file_path, original_name) VALUES (?, ?, ?)",
            (int(record_id), stored_path, safe_original),
        )
        saved += 1
    return saved


def bundle_documents_zip(documents):
    if not documents:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc in documents:
            path = doc.get("path")
            archive_name = doc.get("archive_name")
            if not path or not archive_name:
                continue
            if not path.exists():
                continue
            zf.write(path, archive_name)
    buffer.seek(0)
    return buffer


def dedupe_join(values: Iterable[Optional[str]]) -> str:
    seen = []
    for value in values:
        if value is None:
            continue
        val = str(value).strip()
        if not val:
            continue
        if val not in seen:
            seen.append(val)
    return ", ".join(seen)


def merge_customer_records(conn, customer_ids) -> bool:
    ids = []
    for cid in customer_ids:
        cid_int = int_or_none(cid)
        if cid_int is not None and cid_int not in ids:
            ids.append(cid_int)
    if len(ids) < 2:
        return False

    placeholders = ",".join(["?"] * len(ids))
    query = dedent(
        f"""
        SELECT customer_id, name, company_name, phone, email, address, delivery_address, remarks, purchase_date, product_info, delivery_order_code, sales_person, created_at
        FROM customers
        WHERE customer_id IN ({placeholders})
        """
    )
    df = df_query(conn, query, params=tuple(ids))
    if df.empty:
        return False

    df["created_at_dt"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    df.sort_values(by=["created_at_dt", "customer_id"], inplace=True, na_position="last")
    base_row = df.iloc[0]
    base_id = int(base_row.get("customer_id"))
    other_ids = []
    for row in df.get("customer_id", pd.Series(dtype=object)).tolist():
        rid = int_or_none(row)
        if rid is not None and rid != base_id and rid not in other_ids:
            other_ids.append(rid)
    if not other_ids:
        return False

    name_values = [clean_text(v) for v in df.get("name", pd.Series(dtype=object)).tolist()]
    name_values = [v for v in name_values if v]
    company_values = [clean_text(v) for v in df.get("company_name", pd.Series(dtype=object)).tolist()]
    company_values = [v for v in company_values if v]
    address_values = [clean_text(v) for v in df.get("address", pd.Series(dtype=object)).tolist()]
    address_values = [v for v in address_values if v]
    delivery_values = [clean_text(v) for v in df.get("delivery_address", pd.Series(dtype=object)).tolist()]
    delivery_values = [v for v in delivery_values if v]
    remarks_values = [clean_text(v) for v in df.get("remarks", pd.Series(dtype=object)).tolist()]
    remarks_values = [v for v in remarks_values if v]
    email_values = [clean_text(v) for v in df.get("email", pd.Series(dtype=object)).tolist()]
    email_values = [v for v in email_values if v]
    phone_values = [clean_text(v) for v in df.get("phone", pd.Series(dtype=object)).tolist()]
    phone_values = [v for v in phone_values if v]
    phones_to_recalc: set[str] = set(phone_values)

    base_name = clean_text(base_row.get("name")) or (name_values[0] if name_values else None)
    base_company = clean_text(base_row.get("company_name")) or (company_values[0] if company_values else None)
    base_address = clean_text(base_row.get("address")) or (address_values[0] if address_values else None)
    base_delivery_address = clean_text(base_row.get("delivery_address")) or (delivery_values[0] if delivery_values else None)
    combined_remarks = dedupe_join(remarks_values)
    base_email = clean_text(base_row.get("email")) or (email_values[0] if email_values else None)
    base_phone = clean_text(base_row.get("phone")) or (phone_values[0] if phone_values else None)

    do_codes = []
    product_lines = []
    fallback_products = []
    purchase_dates = []
    sales_people = []

    for record in df.to_dict("records"):
        date_raw = clean_text(record.get("purchase_date"))
        product_raw = clean_text(record.get("product_info"))
        do_raw = clean_text(record.get("delivery_order_code"))
        sales_raw = clean_text(record.get("sales_person"))
        if do_raw:
            do_codes.append(do_raw)
        if product_raw:
            fallback_products.append(product_raw)
        dt = parse_date_value(record.get("purchase_date"))
        if dt is not None:
            purchase_dates.append(dt)
            date_label = dt.strftime(DATE_FMT)
        else:
            date_label = date_raw
        if date_label and product_raw:
            product_lines.append(f"{date_label} – {product_raw}")
        elif product_raw:
            product_lines.append(product_raw)
        elif date_label:
            product_lines.append(date_label)
        if sales_raw:
            sales_people.append(sales_raw)

    earliest_purchase = min(purchase_dates).strftime("%Y-%m-%d") if purchase_dates else None
    combined_products = dedupe_join(product_lines or fallback_products)
    combined_do_codes = dedupe_join(do_codes)
    combined_sales = dedupe_join(sales_people)

    conn.execute(
        """
        UPDATE customers
        SET name=?, company_name=?, phone=?, email=?, address=?, delivery_address=?, remarks=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, dup_flag=0
        WHERE customer_id=?
        """,
        (
            base_name,
            base_company,
            base_phone,
            base_email,
            base_address,
            base_delivery_address,
            clean_text(combined_remarks),
            earliest_purchase,
            clean_text(combined_products),
            clean_text(combined_do_codes),
            clean_text(combined_sales),
            base_id,
        ),
    )

    related_tables = (
        "orders",
        "warranties",
        "delivery_orders",
        "services",
        "maintenance_records",
        "needs",
    )
    for cid in other_ids:
        for table in related_tables:
            conn.execute(f"UPDATE {table} SET customer_id=? WHERE customer_id=?", (base_id, cid))
        conn.execute("UPDATE import_history SET customer_id=? WHERE customer_id=?", (base_id, cid))
        conn.execute("DELETE FROM customers WHERE customer_id=?", (cid,))

    if base_phone:
        phones_to_recalc.add(base_phone)
    if phones_to_recalc:
        for phone in phones_to_recalc:
            recalc_customer_duplicate_flag(conn, phone)
    conn.commit()
    return True


def delete_customer_record(conn, customer_id: int) -> None:
    """Delete a customer and related records, recalculating duplicate flags."""

    try:
        cid = int(customer_id)
    except (TypeError, ValueError):
        return

    cur = conn.execute(
        "SELECT name, phone, delivery_order_code, attachment_path FROM customers WHERE customer_id=?",
        (cid,),
    )
    row = cur.fetchone()
    if not row:
        return

    name_val = clean_text(row[0])
    phone_val = clean_text(row[1])
    do_code = clean_text(row[2])
    attachment_path = row[3]

    conn.execute("DELETE FROM customers WHERE customer_id=?", (cid,))
    if do_code:
        conn.execute(
            "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
            (do_code, cid),
        )
    conn.execute(
        "UPDATE import_history SET deleted_at = datetime('now') WHERE customer_id=? AND deleted_at IS NULL",
        (cid,),
    )
    conn.commit()

    if phone_val:
        recalc_customer_duplicate_flag(conn, phone_val)
        conn.commit()

    if attachment_path:
        path = resolve_upload_path(attachment_path)
        if path and path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    summary_bits: list[str] = []
    if name_val:
        summary_bits.append(name_val)
    if phone_val:
        summary_bits.append(f"phone {phone_val}")
    description = "; ".join(summary_bits) or f"ID #{cid}"
    log_activity(
        conn,
        event_type="customer_deleted",
        description=f"Deleted customer {description}",
        entity_type="customer",
        entity_id=cid,
    )


def collapse_warranty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    work = df.copy()
    work["description"] = work.apply(
        lambda row: dedupe_join(
            [
                clean_text(row.get("product")),
                clean_text(row.get("model")),
                clean_text(row.get("serial")),
            ]
        ),
        axis=1,
    )
    if "remarks" in work.columns:
        work["remarks_clean"] = work["remarks"].apply(clean_text)
    else:
        work["remarks_clean"] = None
    issue_dt = pd.to_datetime(work.get("issue_date"), errors="coerce")
    expiry_dt = pd.to_datetime(work.get("expiry_date"), errors="coerce")
    work["issue_fmt"] = issue_dt.dt.strftime(DATE_FMT)
    work.loc[issue_dt.isna(), "issue_fmt"] = None
    work["expiry_fmt"] = expiry_dt.dt.strftime(DATE_FMT)
    work.loc[expiry_dt.isna(), "expiry_fmt"] = None
    work["expiry_dt"] = expiry_dt

    grouped = (
        work.groupby("customer", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "description": dedupe_join(g["description"].tolist()),
                    "issue_date": dedupe_join(g["issue_fmt"].tolist()),
                    "expiry_date": dedupe_join(g["expiry_fmt"].tolist()),
                    "remarks": dedupe_join(g["remarks_clean"].tolist()),
                    "_sort": g["expiry_dt"].min(),
                }
            )
        )
        .reset_index()
    )
    grouped = grouped.sort_values("_sort", na_position="last").drop(columns=["_sort"])
    grouped.rename(
        columns={
            "customer": "Customer",
            "description": "Description",
            "issue_date": "Issue date",
            "expiry_date": "Expiry date",
            "remarks": "Remarks",
        },
        inplace=True,
    )
    if "Customer" in grouped.columns:
        grouped["Customer"] = grouped["Customer"].fillna("(unknown)")
    return grouped


def _build_customers_export(conn) -> pd.DataFrame:
    scope_clause, scope_params = customer_scope_filter("c")
    where_sql = f"WHERE {scope_clause}" if scope_clause else ""
    query = dedent(
        f"""
        SELECT c.customer_id,
               c.name,
               c.phone,
               c.email,
               c.address,
               c.amount_spent,
               c.purchase_date,
               c.product_info,
               c.delivery_order_code,
               c.sales_person,
               c.created_at,
               COALESCE(u.username, '(unknown)') AS uploaded_by
        FROM customers c
        LEFT JOIN users u ON u.user_id = c.created_by
        {where_sql}
        ORDER BY datetime(c.created_at) DESC, c.customer_id DESC
        """
    )
    df = df_query(conn, query, scope_params if scope_clause else ())
    df = fmt_dates(df, ["purchase_date", "created_at"])
    return df.rename(
        columns={
            "customer_id": "Customer ID",
            "name": "Customer",
            "phone": "Phone",
            "email": "Email",
            "address": "Address",
            "amount_spent": "Amount spent",
            "purchase_date": "Purchase date",
            "product_info": "Product info",
            "delivery_order_code": "Delivery order",
            "sales_person": "Sales person",
            "created_at": "Created at",
            "uploaded_by": "Uploaded by",
        }
    )


def _build_delivery_orders_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT d.do_number,
               COALESCE(c.name, '(unknown)') AS customer,
               d.description,
               d.sales_person,
               d.remarks,
               d.created_at
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        ORDER BY datetime(d.created_at) DESC, d.do_number DESC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["created_at"])
    return df.rename(
        columns={
            "do_number": "DO number",
            "customer": "Customer",
            "description": "Description",
            "sales_person": "Sales person",
            "remarks": "Remarks",
            "created_at": "Created at",
        }
    )


def _build_warranties_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT w.warranty_id,
               COALESCE(c.name, '(unknown)') AS customer,
               COALESCE(p.name, '') AS product,
               p.model,
               w.serial,
               w.issue_date,
               w.expiry_date,
               w.status,
               w.remarks
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        ORDER BY date(w.expiry_date) ASC, w.warranty_id ASC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["issue_date", "expiry_date"])
    if "status" in df.columns:
        df["status"] = df["status"].fillna("Active").apply(lambda x: str(x).title())
    return df.rename(
        columns={
            "warranty_id": "Warranty ID",
            "customer": "Customer",
            "product": "Product",
            "model": "Model",
            "serial": "Serial",
            "issue_date": "Issue date",
            "expiry_date": "Expiry date",
            "status": "Status",
            "remarks": "Remarks",
        }
    )


def _build_services_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT s.service_id,
               s.do_number,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               s.service_date,
               s.service_start_date,
               s.service_end_date,
               s.service_product_info,
               s.description,
               s.status,
               s.remarks,
               s.condition_status,
               s.condition_remarks,
               s.bill_amount,
               s.bill_document_path,
               s.updated_at
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = s.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        ORDER BY datetime(s.service_date) DESC, s.service_id DESC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["service_date", "service_start_date", "service_end_date", "updated_at"])
    if "status" in df.columns:
        df["status"] = df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
    return df.rename(
        columns={
            "service_id": "Service ID",
            "do_number": "DO number",
            "customer": "Customer",
            "service_date": "Service date",
            "service_start_date": "Service start date",
            "service_end_date": "Service end date",
            "service_product_info": "Products sold",
            "description": "Description",
            "status": "Status",
            "remarks": "Remarks",
            "condition_status": "Condition",
            "condition_remarks": "Condition notes",
            "bill_amount": "Bill amount",
            "bill_document_path": "Bill document",
            "updated_at": "Updated at",
        }
    )


def _build_maintenance_export(conn) -> pd.DataFrame:
    query = dedent(
        """
        SELECT m.maintenance_id,
               m.do_number,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               m.maintenance_date,
               m.maintenance_start_date,
               m.maintenance_end_date,
               m.maintenance_product_info,
               m.description,
               m.status,
               m.remarks,
               m.updated_at
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = m.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        ORDER BY datetime(m.maintenance_date) DESC, m.maintenance_id DESC
        """
    )
    df = df_query(conn, query)
    df = fmt_dates(df, ["maintenance_date", "maintenance_start_date", "maintenance_end_date", "updated_at"])
    if "status" in df.columns:
        df["status"] = df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
    return df.rename(
        columns={
            "maintenance_id": "Maintenance ID",
            "do_number": "DO number",
            "customer": "Customer",
            "maintenance_date": "Maintenance date",
            "maintenance_start_date": "Maintenance start date",
            "maintenance_end_date": "Maintenance end date",
            "maintenance_product_info": "Products sold",
            "description": "Description",
            "status": "Status",
            "remarks": "Remarks",
            "updated_at": "Updated at",
        }
    )


def _build_master_sheet(sheets: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = [
        {
            "Sheet": "Export generated at",
            "Details": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]
    for sheet_name, df in sheets:
        count = len(df.index) if df is not None else 0
        label = "record" if count == 1 else "records"
        rows.append({"Sheet": sheet_name, "Details": f"{count} {label}"})
    return pd.DataFrame(rows, columns=["Sheet", "Details"])


def export_database_to_excel(conn) -> bytes:
    sheet_builders = [
        ("Customers", _build_customers_export),
        ("Delivery orders", _build_delivery_orders_export),
        ("Warranties", _build_warranties_export),
        ("Services", _build_services_export),
        ("Maintenance", _build_maintenance_export),
    ]

    sheet_data: list[tuple[str, pd.DataFrame]] = []
    for name, builder in sheet_builders:
        df = builder(conn)
        sheet_data.append((name, df))

    master_df = _build_master_sheet(sheet_data)
    ordered_sheets = [("Master", master_df)] + sheet_data

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in ordered_sheets[:6]:
            safe_name = sheet_name[:31] if sheet_name else "Sheet"
            if not safe_name:
                safe_name = "Sheet"
            if df is None or df.empty:
                df_to_write = pd.DataFrame()
            else:
                df_to_write = df
            df_to_write.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def fetch_warranty_window(conn, start_days: int, end_days: int) -> pd.DataFrame:
    scope_clause, scope_params = customer_scope_filter("c")
    filters = [
        "w.status='active'",
        "date(w.expiry_date) BETWEEN date('now', ?) AND date('now', ?)",
    ]
    params: list[object] = []
    if scope_clause:
        filters.append(scope_clause)
        params.extend(scope_params)
    where_clause = " AND ".join(filters)
    query = dedent(
        f"""
        SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date, w.remarks
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE {where_clause}
        ORDER BY date(w.expiry_date) ASC
        """
    )
    start = f"+{start_days} day"
    end = f"+{end_days} day"
    return df_query(conn, query, (start, end, *params))


def format_warranty_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    work = df.copy()
    expiry_raw = pd.to_datetime(work.get("expiry_date"), errors="coerce")
    today = pd.Timestamp.now().normalize()
    status_labels = []
    work["Description"] = work.apply(
        lambda row: dedupe_join(
            [
                clean_text(row.get("product")),
                clean_text(row.get("model")),
                clean_text(row.get("serial")),
            ]
        ),
        axis=1,
    )
    for idx in work.index:
        exp = expiry_raw.loc[idx] if expiry_raw is not None and idx in expiry_raw.index else pd.NaT
        if pd.notna(exp) and exp.normalize() < today:
            status_labels.append("Expired")
        else:
            base_status = clean_text(work.loc[idx, "status"]) if "status" in work.columns else None
            status_labels.append((base_status or "Active").title())
    work["Status"] = status_labels
    for col in ("product", "model", "serial"):
        if col in work.columns:
            work.drop(columns=[col], inplace=True)
    if "status" in work.columns:
        work.drop(columns=["status"], inplace=True)
    rename_map = {
        "customer": "Customer",
        "issue_date": "Issue date",
        "expiry_date": "Expiry date",
        "remarks": "Remarks",
    }
    work.rename(columns={k: v for k, v in rename_map.items() if k in work.columns}, inplace=True)
    for col in ("dup_flag", "id", "duplicate"):
        if col in work.columns:
            work.drop(columns=[col], inplace=True)
    return work


def _pdf_escape_text(value: str) -> str:
    replacements = [("\\", "\\\\"), ("(", "\\("), (")", "\\)")]
    escaped = value
    for old, new in replacements:
        escaped = escaped.replace(old, new)
    return escaped


def _build_simple_pdf_document(lines: list[str]) -> bytes:
    if not lines:
        lines = [""]
    commands = ["BT", "/F1 12 Tf", "72 770 Td"]
    for idx, line in enumerate(lines):
        escaped = _pdf_escape_text(line)
        if idx == 0:
            commands.append(f"({escaped}) Tj")
        else:
            commands.append("0 -14 Td")
            commands.append(f"({escaped}) Tj")
    commands.append("ET")
    stream_bytes = "\n".join(commands).encode("latin-1", "replace")

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = []

    def write_obj(obj_id: int, body: bytes) -> None:
        offsets.append(buffer.tell())
        buffer.write(f"{obj_id} 0 obj\n".encode("latin-1"))
        buffer.write(body)
        if not body.endswith(b"\n"):
            buffer.write(b"\n")
        buffer.write(b"endobj\n")

    write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>\n")
    write_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n")
    write_obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n",
    )
    stream_obj = b"<< /Length %d >>\nstream\n" % len(stream_bytes) + stream_bytes + b"\nendstream\n"
    write_obj(4, stream_obj)
    write_obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n")

    xref_offset = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets) + 1}\n".encode("latin-1"))
    buffer.write(b"0000000000 65535 f \n")
    for off in offsets:
        buffer.write(f"{off:010d} 00000 n \n".encode("latin-1"))
    buffer.write(b"trailer\n")
    buffer.write(f"<< /Size {len(offsets) + 1} /Root 1 0 R >>\n".encode("latin-1"))
    buffer.write(b"startxref\n")
    buffer.write(f"{xref_offset}\n".encode("latin-1"))
    buffer.write(b"%%EOF\n")
    return buffer.getvalue()


def generate_customer_summary_pdf(customer_name: str, info: dict, warranties: Optional[pd.DataFrame], services: pd.DataFrame, maintenance: pd.DataFrame) -> bytes:
    lines: list[str] = [f"Customer Summary – {customer_name}", ""]
    lines.extend(
        [
            f"Phone: {clean_text(info.get('phone')) or '-'}",
            f"Address: {clean_text(info.get('address')) or '-'}",
            f"Purchase: {clean_text(info.get('purchase_dates')) or '-'}",
            f"Product: {clean_text(info.get('products')) or '-'}",
            f"Delivery order: {clean_text(info.get('do_codes')) or '-'}",
            "",
        ]
    )

    def extend_section(title: str, rows: list[str]) -> None:
        lines.append(title)
        if not rows:
            lines.append("  (no records)")
        else:
            for row in rows:
                lines.append(f"  • {row}")
        lines.append("")

    warranty_rows: list[str] = []
    if warranties is not None and isinstance(warranties, pd.DataFrame) and not warranties.empty:
        for _, row in warranties.iterrows():
            warranty_rows.append(
                " | ".join(
                    [
                        f"Description: {clean_text(row.get('Description')) or '-'}",
                        f"Issue: {clean_text(row.get('Issue date')) or '-'}",
                        f"Expiry: {clean_text(row.get('Expiry date')) or '-'}",
                        f"Status: {clean_text(row.get('Status')) or '-'}",
                    ]
                )
            )

    service_rows: list[str] = []
    if isinstance(services, pd.DataFrame) and not services.empty:
        for _, row in services.iterrows():
            service_rows.append(
                " | ".join(
                    [
                        f"DO: {clean_text(row.get('do_number')) or '-'}",
                        f"Date: {clean_text(row.get('service_date')) or '-'}",
                        f"Desc: {clean_text(row.get('description')) or '-'}",
                        f"Remarks: {clean_text(row.get('remarks')) or '-'}",
                    ]
                )
            )

    maintenance_rows: list[str] = []
    if isinstance(maintenance, pd.DataFrame) and not maintenance.empty:
        for _, row in maintenance.iterrows():
            maintenance_rows.append(
                " | ".join(
                    [
                        f"DO: {clean_text(row.get('do_number')) or '-'}",
                        f"Date: {clean_text(row.get('maintenance_date')) or '-'}",
                        f"Desc: {clean_text(row.get('description')) or '-'}",
                        f"Remarks: {clean_text(row.get('remarks')) or '-'}",
                    ]
                )
            )

    extend_section("Warranties", warranty_rows)
    extend_section("Service history", service_rows)
    extend_section("Maintenance history", maintenance_rows)

    return _build_simple_pdf_document(lines)


def _streamlit_flag_options_from_env() -> dict[str, object]:
    """Derive Streamlit bootstrap flag options from environment variables."""

    flag_options: dict[str, object] = {}

    port_env = os.getenv("PORT")
    if port_env:
        try:
            port = int(port_env)
        except (TypeError, ValueError):
            port = None
        if port and port > 0:
            flag_options["server.port"] = port

    address_env = os.getenv("HOST") or os.getenv("BIND_ADDRESS")
    flag_options["server.address"] = address_env or "0.0.0.0"

    headless_env = os.getenv("STREAMLIT_SERVER_HEADLESS")
    if headless_env is None:
        flag_options["server.headless"] = True
    else:
        flag_options["server.headless"] = headless_env.strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    return flag_options


def _bootstrap_streamlit_app() -> None:
    """Launch the Streamlit app when executed via ``python app.py``."""

    try:
        from streamlit.web import bootstrap
    except Exception:
        return

    try:
        bootstrap.run(
            os.path.abspath(__file__),
            False,
            [],
            _streamlit_flag_options_from_env(),
        )
    except Exception:
        pass


def recalc_customer_duplicate_flag(conn, phone):
    if not phone or str(phone).strip() == "":
        return
    cur = conn.execute(
        "SELECT customer_id, purchase_date FROM customers WHERE phone = ?",
        (str(phone).strip(),),
    )
    rows = cur.fetchall()
    if not rows:
        return

    grouped: dict[Optional[str], list[int]] = {}
    for cid, purchase_date in rows:
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        key = clean_text(purchase_date) or None
        grouped.setdefault(key, []).append(cid_int)

    updates: list[tuple[int, int]] = []
    for cid_list in grouped.values():
        dup_flag = 1 if len(cid_list) > 1 else 0
        updates.extend((dup_flag, cid) for cid in cid_list)

    if updates:
        conn.executemany(
            "UPDATE customers SET dup_flag=? WHERE customer_id=?",
            updates,
        )


def init_ui():
    st.set_page_config(page_title="PS Mini CRM", page_icon="🧰", layout="wide")
    st.title("PS Engineering – Mini CRM")
    st.caption("Customers • Warranties • Needs • Summaries")
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #f5f9ff;
            border-radius: 0.8rem;
            padding: 0.85rem;
            border: 1px solid rgba(49, 51, 63, 0.08);
        }
        div[data-testid="stPopover"] > button {
            border: none !important;
            background: transparent !important;
            font-size: 1.25rem;
            padding: 0.15rem 0.35rem !important;
            color: #1d3b64 !important;
        }
        .ps-notification-popover {
            display: flex;
            justify-content: flex-end;
        }
        .ps-notification-popover button:hover {
            background: rgba(29, 59, 100, 0.08) !important;
        }
        .ps-notification-section-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #1d3b64;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "user" not in st.session_state:
        st.session_state.user = None

# ---------- Auth ----------
def login_box(conn):
    st.sidebar.markdown("### Login")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as {st.session_state.user['username']} ({st.session_state.user['role']})")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.page = "Dashboard"
            _safe_rerun()
        return True
    with st.sidebar.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
    if ok:
        row = df_query(conn, "SELECT user_id, username, pass_hash, role FROM users WHERE username = ?", (u,))
        if not row.empty and hashlib.sha256(p.encode("utf-8")).hexdigest() == row.iloc[0]["pass_hash"]:
            st.session_state.user = {"user_id": int(row.iloc[0]["user_id"]), "username": row.iloc[0]["username"], "role": row.iloc[0]["role"]}
            st.session_state.page = "Dashboard"
            st.session_state.just_logged_in = True
            _safe_rerun()
        else:
            st.sidebar.error("Invalid credentials")
    st.stop()

def ensure_auth(role=None):
    if role and st.session_state.user and st.session_state.user["role"] != role:
        st.warning("You do not have permission to access this page.")
        st.stop()

# ---------- Pages ----------
def dashboard(conn):
    st.subheader("📊 Dashboard")
    st.markdown(
        "<div style='text-align: right; font-size: 0.6rem; color: #888;'>by ZAD</div>",
        unsafe_allow_html=True,
    )
    header_cols = st.columns((0.85, 0.15))
    with header_cols[1]:
        render_notification_bell(conn)
    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"
    allowed_customers = accessible_customer_ids(conn)
    scope_clause, scope_params = customer_scope_filter("c")

    if "show_today_expired" not in st.session_state:
        st.session_state.show_today_expired = False

    if is_admin:
        col1, col2, col3, col4 = st.columns(4)
        complete_count = int(
            df_query(conn, f"SELECT COUNT(*) c FROM customers WHERE {customer_complete_clause()}").iloc[0]["c"]
        )
        scrap_count = int(
            df_query(conn, f"SELECT COUNT(*) c FROM customers WHERE {customer_incomplete_clause()}").iloc[0]["c"]
        )
        with col1:
            st.metric("Customers", complete_count)
        with col2:
            st.metric("Scraps", scrap_count)
        with col3:
            st.metric(
                "Active Warranties",
                int(
                    df_query(
                        conn,
                        "SELECT COUNT(*) c FROM warranties WHERE status='active' AND date(expiry_date) >= date('now')",
                    ).iloc[0]["c"]
                ),
            )
        with col4:
            expired_count = int(
                df_query(
                    conn,
                    "SELECT COUNT(*) c FROM warranties WHERE status='active' AND date(expiry_date) < date('now')",
                ).iloc[0]["c"]
            )
            st.metric("Expired", expired_count)

        st.markdown("#### Daily report coverage")
        report_date_value = st.date_input(
            "Review date",
            value=date.today(),
            key="dashboard_daily_report_date",
            help="Identify who submitted a daily report on the selected date.",
        )
        report_iso = to_iso_date(report_date_value) or date.today().isoformat()
        staff_df = df_query(
            conn,
            dedent(
                """
                SELECT user_id, username
                FROM users
                WHERE LOWER(COALESCE(role, 'staff')) <> 'admin'
                ORDER BY LOWER(username)
                """
            ),
        )

        if staff_df.empty:
            st.info("No staff accounts available for coverage tracking yet.")
        else:
            staff_df["user_id"] = staff_df["user_id"].apply(lambda val: int(float(val)))
            staff_df["username"] = staff_df.apply(
                lambda row: clean_text(row.get("username")) or f"User #{int(row['user_id'])}",
                axis=1,
            )
            submitted_df = df_query(
                conn,
                dedent(
                    """
                    SELECT DISTINCT user_id
                    FROM work_reports
                    WHERE period_type='daily' AND date(period_start)=date(?)
                    """
                ),
                (report_iso,),
            )
            submitted_ids: set[int] = set()
            if not submitted_df.empty:
                submitted_ids = {
                    int(float(uid))
                    for uid in submitted_df["user_id"].dropna().tolist()
                }

            staff_df["Submitted"] = staff_df["user_id"].apply(lambda uid: uid in submitted_ids)
            total_staff = int(staff_df.shape[0])
            submitted_total = int(staff_df["Submitted"].sum())
            missing_total = total_staff - submitted_total

            coverage_cols = st.columns(3)
            coverage_cols[0].metric("Team members", total_staff)
            coverage_cols[1].metric("Reports filed", submitted_total)
            coverage_cols[2].metric("Missing reports", missing_total)

            missing_df = staff_df[~staff_df["Submitted"]]
            if missing_total:
                st.warning("Daily reports pending for the following team members:")
                st.markdown("\n".join(f"- {name}" for name in missing_df["username"]))
            else:
                st.success("All tracked team members have filed their daily report.")

            status_table = staff_df.rename(
                columns={"username": "Team member", "Submitted": "Daily report"}
            )
            status_table["Daily report"] = status_table["Daily report"].map(
                {True: "Submitted", False: "Missing"}
            )
            st.dataframe(
                status_table[["Team member", "Daily report"]],
                use_container_width=True,
            )
            st.caption(
                f"Coverage for {format_period_range(report_iso, report_iso)} • Admins are excluded from this list."
            )
            staff_options = [int(uid) for uid in staff_df["user_id"].tolist()]
            staff_labels = {
                int(row["user_id"]): row["username"] for _, row in staff_df.iterrows()
            }
            if staff_options:
                st.markdown("##### Review daily submissions")
                submitted_options = [
                    int(uid)
                    for uid in staff_df.loc[staff_df["Submitted"], "user_id"].tolist()
                ]
                default_staff_id = (
                    submitted_options[0] if submitted_options else staff_options[0]
                )
                try:
                    default_index = staff_options.index(default_staff_id)
                except ValueError:
                    default_index = 0
                selected_staff_id = int(
                    st.selectbox(
                        "Team member report",
                        staff_options,
                        index=default_index,
                        format_func=lambda uid: staff_labels.get(
                            int(uid), f"User #{int(uid)}"
                        ),
                        key="dashboard_daily_report_user",
                    )
                )
                selected_staff_name = staff_labels.get(
                    selected_staff_id, f"User #{selected_staff_id}"
                )
                report_detail = df_query(
                    conn,
                    dedent(
                        """
                        SELECT report_id,
                               tasks,
                               remarks,
                               research,
                               attachment_path,
                               period_start,
                               period_end,
                               created_at,
                               updated_at
                        FROM work_reports
                        WHERE user_id=?
                          AND period_type='daily'
                          AND date(period_start) = date(?)
                        ORDER BY datetime(updated_at) DESC, report_id DESC
                        LIMIT 1
                        """
                    ),
                    (selected_staff_id, report_iso),
                )
                has_marked_submitted = bool(
                    staff_df.loc[staff_df["user_id"] == selected_staff_id, "Submitted"].any()
                )
                if report_detail.empty:
                    if has_marked_submitted:
                        st.warning(
                            f"No daily report could be located for {selected_staff_name}.",
                            icon="⚠️",
                        )
                    else:
                        st.info(
                            f"{selected_staff_name} has not submitted a daily report for "
                            f"{format_period_range(report_iso, report_iso)}.",
                        )
                else:
                    record = report_detail.iloc[0].to_dict()
                    st.markdown(
                        f"**Period:** {format_period_range(record.get('period_start'), record.get('period_end'))}"
                    )
                    st.markdown("**Tasks completed**")
                    st.write(clean_text(record.get("tasks")) or "—")
                    st.markdown("**Remarks / blockers**")
                    st.write(clean_text(record.get("remarks")) or "—")
                    st.markdown("**Research / learnings**")
                    st.write(clean_text(record.get("research")) or "—")
                    submitted_label = (
                        format_time_ago(record.get("created_at"))
                        or format_period_range(record.get("created_at"), record.get("created_at"))
                    )
                    updated_label = (
                        format_time_ago(record.get("updated_at"))
                        or format_period_range(record.get("updated_at"), record.get("updated_at"))
                    )
                    st.caption(
                        f"Submitted {submitted_label} • Last updated {updated_label}"
                    )
                    attachment_value = clean_text(record.get("attachment_path"))
                    if attachment_value:
                        attachment_path = resolve_upload_path(attachment_value)
                        attachment_bytes = None
                        if attachment_path and attachment_path.exists():
                            try:
                                attachment_bytes = attachment_path.read_bytes()
                            except OSError:
                                attachment_bytes = None
                        if attachment_bytes:
                            st.download_button(
                                "Download attachment",
                                data=attachment_bytes,
                                file_name=(
                                    attachment_path.name if attachment_path else "attachment"
                                ),
                                key=f"dashboard_report_attachment_{record.get('report_id')}",
                            )
                        else:
                            st.warning(
                                "The attached file could not be found on disk.",
                                icon="⚠️",
                            )
                    else:
                        st.caption("No attachment uploaded for this report.")
    else:
        st.info("Staff view: focus on upcoming activities below. Metrics are available to admins only.")

    month_expired_current, month_expired_previous = month_bucket_counts(
        conn,
        "warranties",
        "expiry_date",
        where="status='active' AND date(expiry_date) < date('now')",
    )
    month_expired = month_expired_current
    expired_delta = format_metric_delta(month_expired_current, month_expired_previous)

    service_month_current, service_month_previous = month_bucket_counts(
        conn,
        "services",
        "service_date",
    )
    service_month = service_month_current
    service_delta = format_metric_delta(service_month_current, service_month_previous)

    maintenance_month_current, maintenance_month_previous = month_bucket_counts(
        conn,
        "maintenance_records",
        "maintenance_date",
    )
    maintenance_month = maintenance_month_current
    maintenance_delta = format_metric_delta(
        maintenance_month_current, maintenance_month_previous
    )
    today_expired_df = df_query(
        conn,
        """
        SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE w.status='active' AND date(w.expiry_date) = date('now')
        ORDER BY date(w.expiry_date) ASC
        """,
    )
    today_expired_count = len(today_expired_df.index)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Expired this month", month_expired, delta=expired_delta)
    with col6:
        st.metric("Services this month", service_month, delta=service_delta)
    with col7:
        st.metric(
            "Maintenance this month",
            maintenance_month,
            delta=maintenance_delta,
        )
    with col8:
        st.metric("Expired today", today_expired_count)
        toggle_label = "Show list" if not st.session_state.get("show_today_expired") else "Hide list"
        if st.button(toggle_label, key="toggle_expired_today"):
            st.session_state.show_today_expired = not st.session_state.get("show_today_expired")
            show_today_expired = st.session_state.show_today_expired
        else:
            show_today_expired = st.session_state.get("show_today_expired")

    if not today_expired_df.empty:
        notice = collapse_warranty_rows(today_expired_df)
        lines = []
        for _, row in notice.iterrows():
            customer = row.get("Customer") or "(unknown)"
            description = row.get("Description") or ""
            if description:
                lines.append(f"- {customer}: {description}")
            else:
                lines.append(f"- {customer}")
        st.warning("⚠️ Warranties expiring today:\n" + "\n".join(lines))

    show_today_expired = st.session_state.get("show_today_expired")
    if show_today_expired:
        if today_expired_df.empty:
            st.info("No warranties expire today.")
        else:
            today_detail = fmt_dates(today_expired_df, ["issue_date", "expiry_date"])
            today_table = format_warranty_table(today_detail)
            st.markdown("#### Warranties expiring today")
            st.dataframe(today_table, use_container_width=True)

    if is_admin:
        if "show_deleted_panel" not in st.session_state:
            st.session_state.show_deleted_panel = False

        excel_bytes = export_database_to_excel(conn)
        admin_action_cols = st.columns([0.78, 0.22])
        with admin_action_cols[0]:
            st.download_button(
                "⬇️ Download full database (Excel)",
                excel_bytes,
                file_name="ps_crm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        toggle_label = (
            "🗑️ Deleted data"
            if not st.session_state.get("show_deleted_panel")
            else "Hide deleted data"
        )
        with admin_action_cols[1]:
            if st.button(
                toggle_label,
                key="toggle_deleted_panel",
                help="Admins can review deleted import records here.",
            ):
                st.session_state.show_deleted_panel = not st.session_state.get(
                    "show_deleted_panel", False
                )

        if st.session_state.get("show_deleted_panel"):
            deleted_df = df_query(
                conn,
                """
                SELECT import_id, imported_at, customer_name, phone, product_label, original_date, do_number, deleted_at
                FROM import_history
                WHERE deleted_at IS NOT NULL
                ORDER BY datetime(deleted_at) DESC
                """,
            )

            if deleted_df.empty:
                st.info("No deleted import entries found.")
            else:
                formatted_deleted = fmt_dates(
                    deleted_df,
                    ["imported_at", "original_date", "deleted_at"],
                )
                deleted_bytes = io.BytesIO()
                with pd.ExcelWriter(deleted_bytes, engine="openpyxl") as writer:
                    formatted_deleted.to_excel(
                        writer, index=False, sheet_name="deleted_imports"
                    )
                deleted_bytes.seek(0)

                st.markdown("#### Deleted import history")
                st.caption(
                    "Only administrators can access this view. Download the Excel file for a full audit trail."
                )
                st.download_button(
                    "Download deleted imports",
                    deleted_bytes.getvalue(),
                    file_name="deleted_imports.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="deleted_imports_dl",
                )
                preview_cols = [
                    "import_id",
                    "imported_at",
                    "customer_name",
                    "phone",
                    "product_label",
                    "do_number",
                    "original_date",
                    "deleted_at",
                ]
                st.dataframe(
                    formatted_deleted[preview_cols],
                    use_container_width=True,
                )

    st.markdown("---")
    st.subheader("🔎 Quick snapshots")
    tab1, tab2, tab3 = st.tabs([
        "Upcoming expiries (next 60 days)",
        "Recent services",
        "Recent maintenance",
    ])

    with tab1:
        range_col, projection_col = st.columns((2, 1))
        with range_col:
            days_window = st.slider(
                "Upcoming window (days)",
                min_value=7,
                max_value=180,
                value=60,
                step=1,
                help="Adjust how far ahead to look for upcoming warranty expiries.",
            )
        with projection_col:
            months_projection = st.slider(
                "Projection window (months)",
                min_value=1,
                max_value=12,
                value=6,
                help="Preview the workload trend for active warranties.",
            )

        upcoming = fetch_warranty_window(conn, 0, int(days_window))
        upcoming = format_warranty_table(upcoming)
        upcoming_count = int(len(upcoming.index)) if upcoming is not None else 0

        metric_col1, metric_col2 = st.columns((1, 1))
        with metric_col1:
            st.metric("Upcoming expiries", upcoming_count)
        with metric_col2:
            st.metric("Days in view", int(days_window))

        st.caption(
            f"Active warranties scheduled to expire in the next {int(days_window)} days."
        )

        if upcoming is None or upcoming.empty:
            st.info("No active warranties are due within the selected window.")
        else:
            show_all = False
            if len(upcoming.index) > 10:
                show_all = st.checkbox(
                    "Show all upcoming expiries", key="show_all_upcoming"
                )
            upcoming_display = upcoming if show_all else upcoming.head(10)
            st.dataframe(upcoming_display, use_container_width=True)

            csv_bytes = upcoming.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download upcoming expiries (CSV)",
                csv_bytes,
                file_name="upcoming_warranties.csv",
                mime="text/csv",
                key="download_upcoming_csv",
            )

            breakdown_labels = {
                "sales_person": "Sales person",
                "customer": "Customer",
                "product": "Product",
            }
            selected_breakdown = st.selectbox(
                "Group upcoming expiries by",
                options=list(breakdown_labels.keys()),
                format_func=lambda key: breakdown_labels[key],
                help="Identify who or what is most affected in the chosen window.",
                key="upcoming_breakdown_selector",
            )

            breakdown_df = upcoming_warranty_breakdown(
                conn, int(days_window), selected_breakdown
            )
            label_column = breakdown_df.columns[0] if not breakdown_df.empty else breakdown_labels[selected_breakdown]
            st.caption(
                "Use this breakdown to prioritise outreach for the busiest owners or products."
            )
            if breakdown_df.empty:
                st.info(
                    f"No grouping data available for the selected window ({label_column})."
                )
            else:
                st.dataframe(breakdown_df, use_container_width=True)
                top_focus = breakdown_df.iloc[0]
                st.success(
                    f"Highest upcoming load: {top_focus[label_column]} ({int(top_focus['Expiring warranties'])} warranties)."
                )

        projection_df = upcoming_warranty_projection(conn, int(months_projection))
        st.caption("Projected monthly warranty expiries")
        if projection_df.empty:
            st.info("No active warranties are scheduled to expire in the selected projection window.")
        else:
            st.bar_chart(projection_df.set_index("Month"))
            peak_row = projection_df.loc[projection_df["Expiring warranties"].idxmax()]
            peak_value = int(peak_row["Expiring warranties"])
            if peak_value > 0:
                st.success(
                    f"Peak month: {peak_row['Month']} with {peak_value} scheduled expiries."
                )
            else:
                st.info("All selected months currently show zero scheduled expiries.")

    with tab2:
        recent_services = df_query(
            conn,
            """
            SELECT s.do_number,
                   s.customer_id,
                   d.customer_id AS do_customer_id,
                   s.service_date,
                   COALESCE(c.name, cdo.name, '(unknown)') AS customer,
                   s.description
            FROM services s
            LEFT JOIN customers c ON c.customer_id = s.customer_id
            LEFT JOIN delivery_orders d ON d.do_number = s.do_number
            LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
            ORDER BY datetime(s.service_date) DESC, s.service_id DESC
            LIMIT 10
            """,
        )
        if allowed_customers is not None:
            recent_services = recent_services[
                recent_services.apply(
                    lambda row: any(
                        cid in allowed_customers
                        for cid in [
                            int(row.get("customer_id"))
                            if pd.notna(row.get("customer_id"))
                            else None,
                            int(row.get("do_customer_id"))
                            if pd.notna(row.get("do_customer_id"))
                            else None,
                        ]
                        if cid is not None
                    ),
                    axis=1,
                )
            ]
        recent_services = fmt_dates(recent_services, ["service_date"])
        st.dataframe(
            recent_services.rename(
                columns={
                    "do_number": "DO Serial",
                    "service_date": "Service date",
                    "customer": "Customer",
                    "description": "Description",
                }
            ).drop(columns=["customer_id", "do_customer_id"], errors="ignore"),
            use_container_width=True,
        )

    with tab3:
        recent_maintenance = df_query(
            conn,
            """
            SELECT m.do_number,
                   m.customer_id,
                   d.customer_id AS do_customer_id,
                   m.maintenance_date,
                   COALESCE(c.name, cdo.name, '(unknown)') AS customer,
                   m.description
            FROM maintenance_records m
            LEFT JOIN customers c ON c.customer_id = m.customer_id
            LEFT JOIN delivery_orders d ON d.do_number = m.do_number
            LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
            ORDER BY datetime(m.maintenance_date) DESC, m.maintenance_id DESC
            LIMIT 10
            """,
        )
        if allowed_customers is not None:
            recent_maintenance = recent_maintenance[
                recent_maintenance.apply(
                    lambda row: any(
                        cid in allowed_customers
                        for cid in [
                            int(row.get("customer_id"))
                            if pd.notna(row.get("customer_id"))
                            else None,
                            int(row.get("do_customer_id"))
                            if pd.notna(row.get("do_customer_id"))
                            else None,
                        ]
                        if cid is not None
                    ),
                    axis=1,
                )
            ]
        recent_maintenance = fmt_dates(recent_maintenance, ["maintenance_date"])
        st.dataframe(
            recent_maintenance.rename(
                columns={
                    "do_number": "DO Serial",
                    "maintenance_date": "Maintenance date",
                    "customer": "Customer",
                    "description": "Description",
                }
            ).drop(columns=["customer_id", "do_customer_id"], errors="ignore"),
            use_container_width=True,
        )


def show_expiry_notifications(conn):
    if not st.session_state.get("just_logged_in"):
        return

    scope_clause, scope_params = customer_scope_filter("c")
    allowed_customers = accessible_customer_ids(conn)
    scheduled_services = df_query(
        conn,
        """
        SELECT s.service_id,
               s.customer_id,
               d.customer_id AS do_customer_id,
               s.do_number,
               COALESCE(s.service_start_date, s.service_date) AS start_date,
               s.status,
               s.description,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = s.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        WHERE COALESCE(s.service_start_date, s.service_date) IS NOT NULL
          AND date(COALESCE(s.service_start_date, s.service_date)) = date('now')
        ORDER BY datetime(COALESCE(s.service_start_date, s.service_date)) ASC, s.service_id ASC
        """,
    )
    scheduled_maintenance = df_query(
        conn,
        """
        SELECT m.maintenance_id,
               m.customer_id,
               d.customer_id AS do_customer_id,
               m.do_number,
               COALESCE(m.maintenance_start_date, m.maintenance_date) AS start_date,
               m.status,
               m.description,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = m.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        WHERE COALESCE(m.maintenance_start_date, m.maintenance_date) IS NOT NULL
          AND date(COALESCE(m.maintenance_start_date, m.maintenance_date)) = date('now')
        ORDER BY datetime(COALESCE(m.maintenance_start_date, m.maintenance_date)) ASC, m.maintenance_id ASC
        """,
    )

    if not scheduled_services.empty:
        scheduled_services = scheduled_services[
            scheduled_services["status"].apply(is_pending_status)
        ]
    if not scheduled_maintenance.empty:
        scheduled_maintenance = scheduled_maintenance[
            scheduled_maintenance["status"].apply(is_pending_status)
        ]

    scheduled_services = fmt_dates(scheduled_services, ["start_date"])
    scheduled_maintenance = fmt_dates(scheduled_maintenance, ["start_date"])
    if allowed_customers is not None:
        def _belongs(row):
            candidates = []
            for key in ("customer_id", "do_customer_id"):
                value = row.get(key)
                if pd.notna(value):
                    try:
                        candidates.append(int(value))
                    except (TypeError, ValueError):
                        continue
            return any(cid in allowed_customers for cid in candidates)

        scheduled_services = scheduled_services[scheduled_services.apply(_belongs, axis=1)]
        scheduled_maintenance = scheduled_maintenance[scheduled_maintenance.apply(_belongs, axis=1)]
    scheduled_services.drop(columns=["customer_id", "do_customer_id"], inplace=True, errors="ignore")
    scheduled_maintenance.drop(columns=["customer_id", "do_customer_id"], inplace=True, errors="ignore")

    upcoming_sections: list[pd.DataFrame] = []
    upcoming_messages: list[str] = []

    due_notes = df_query(
        conn,
        """
        SELECT n.note_id,
               n.customer_id,
               n.note,
               n.remind_on,
               c.name AS customer
        FROM customer_notes n
        JOIN customers c ON c.customer_id = n.customer_id
        WHERE n.is_done = 0
          AND n.remind_on IS NOT NULL
          AND date(n.remind_on) <= date('now')
        ORDER BY date(n.remind_on) ASC, datetime(n.created_at) ASC
        """,
    )
    due_notes = fmt_dates(due_notes, ["remind_on"])
    if allowed_customers is not None:
        due_notes = due_notes[due_notes["customer_id"].apply(lambda value: pd.notna(value) and int(value) in allowed_customers)]
    due_notes.drop(columns=["customer_id"], inplace=True, errors="ignore")
    notes_display = pd.DataFrame()
    if not due_notes.empty:
        notes_display = due_notes.rename(
            columns={
                "customer": "Customer",
                "note": "Remark",
                "remind_on": "Due date",
            }
        )[
            ["Customer", "Remark", "Due date"]
        ]
        for record in due_notes.to_dict("records"):
            customer_ref = clean_text(record.get("customer")) or "(unknown)"
            note_text = clean_text(record.get("note")) or "Follow-up due"
            due_label = clean_text(record.get("remind_on")) or datetime.now().strftime(DATE_FMT)
            upcoming_messages.append(
                f"Reminder: follow up with {customer_ref} – {note_text} (due {due_label})."
            )

    if not scheduled_services.empty:
        service_records = scheduled_services.to_dict("records")
        for record in service_records:
            do_ref = clean_text(record.get("do_number"))
            if not do_ref:
                try:
                    service_identifier = int(record.get("service_id"))
                    do_ref = f"Service #{service_identifier}"
                except Exception:
                    do_ref = "Service"
            customer_ref = clean_text(record.get("customer")) or "(unknown)"
            start_label = clean_text(record.get("start_date")) or datetime.now().strftime(DATE_FMT)
            upcoming_messages.append(
                f"Service {do_ref} for {customer_ref} starts today ({start_label})."
            )
        service_display = scheduled_services.copy()
        service_display = service_display.rename(
            columns={
                "do_number": "DO Serial",
                "start_date": "Start date",
                "status": "Status",
                "description": "Description",
                "customer": "Customer",
            }
        )
        service_display.insert(0, "Type", "Service")
        service_display = service_display.drop(columns=["service_id"], errors="ignore")
        upcoming_sections.append(
            service_display[[
                "Type",
                "DO Serial",
                "Customer",
                "Start date",
                "Status",
                "Description",
            ]]
        )

    if not scheduled_maintenance.empty:
        maintenance_records = scheduled_maintenance.to_dict("records")
        for record in maintenance_records:
            do_ref = clean_text(record.get("do_number"))
            if not do_ref:
                try:
                    maintenance_identifier = int(record.get("maintenance_id"))
                    do_ref = f"Maintenance #{maintenance_identifier}"
                except Exception:
                    do_ref = "Maintenance"
            customer_ref = clean_text(record.get("customer")) or "(unknown)"
            start_label = clean_text(record.get("start_date")) or datetime.now().strftime(DATE_FMT)
            upcoming_messages.append(
                f"Maintenance {do_ref} for {customer_ref} starts today ({start_label})."
            )
        maintenance_display = scheduled_maintenance.copy()
        maintenance_display = maintenance_display.rename(
            columns={
                "do_number": "DO Serial",
                "start_date": "Start date",
                "status": "Status",
                "description": "Description",
                "customer": "Customer",
            }
        )
        maintenance_display.insert(0, "Type", "Maintenance")
        maintenance_display = maintenance_display.drop(columns=["maintenance_id"], errors="ignore")
        upcoming_sections.append(
            maintenance_display[[
                "Type",
                "DO Serial",
                "Customer",
                "Start date",
                "Status",
                "Description",
            ]]
        )

    upcoming_df = (
        pd.concat(upcoming_sections, ignore_index=True)
        if upcoming_sections
        else pd.DataFrame()
    )

    scope_filter_clause = f" AND {scope_clause}" if scope_clause else ""
    total_expired_query = dedent(
        f"""
        SELECT COUNT(*) c
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        WHERE date(w.expiry_date) < date('now'){scope_filter_clause}
        """
    )
    total_expired = int(df_query(conn, total_expired_query, scope_params).iloc[0]["c"])
    month_expired = 0
    formatted = pd.DataFrame()
    if total_expired > 0:
        month_expired_query = dedent(
            f"""
            SELECT COUNT(*) c
            FROM warranties w
            LEFT JOIN customers c ON c.customer_id = w.customer_id
            WHERE date(w.expiry_date) < date('now')
              AND strftime('%Y-%m', w.expiry_date) = strftime('%Y-%m', 'now'){scope_filter_clause}
            """
        )
        month_expired = int(df_query(conn, month_expired_query, scope_params).iloc[0]["c"])
        expired_recent_query = dedent(
            f"""
            SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date
            FROM warranties w
            LEFT JOIN customers c ON c.customer_id = w.customer_id
            LEFT JOIN products p ON p.product_id = w.product_id
            WHERE date(w.expiry_date) < date('now'){scope_filter_clause}
            ORDER BY date(w.expiry_date) DESC
            LIMIT 12
            """
        )
        expired_recent = df_query(conn, expired_recent_query, scope_params)
        formatted = format_warranty_table(expired_recent)

    show_upcoming = not upcoming_df.empty
    show_expired = total_expired > 0
    show_notes = not notes_display.empty

    if not show_upcoming and not show_expired and not show_notes:
        st.session_state.just_logged_in = False
        return

    if show_upcoming:
        upcoming_preview = upcoming_df.head(5)
        upcoming_details = []
        for record in upcoming_preview.to_dict("records"):
            type_label = clean_text(record.get("Type")) or ""
            customer_label = clean_text(record.get("Customer")) or "(unknown)"
            start_label = clean_text(record.get("Start date")) or ""
            description_label = clean_text(record.get("Description")) or clean_text(
                record.get("Status")
            ) or ""
            detail_parts = [part for part in [type_label, customer_label, start_label] if part]
            detail_line = " • ".join(detail_parts)
            if description_label:
                detail_line = (
                    f"{detail_line} – {description_label}" if detail_line else description_label
                )
            upcoming_details.append(detail_line)
        push_runtime_notification(
            "Today's schedule",
            f"{len(upcoming_df.index)} task(s) scheduled for today.",
            severity="info",
            details=upcoming_details,
        )

    if show_notes:
        notes_preview = notes_display.head(5)
        notes_details = []
        for record in notes_preview.to_dict("records"):
            customer_label = clean_text(record.get("Customer")) or "(unknown)"
            due_label = clean_text(record.get("Due date")) or ""
            remark_label = clean_text(record.get("Remark")) or ""
            detail_line = " • ".join(
                part for part in [customer_label, due_label, remark_label] if part
            )
            notes_details.append(detail_line)
        push_runtime_notification(
            "Follow-up reminders",
            f"{len(notes_display.index)} customer reminder(s) due.",
            severity="warning",
            details=notes_details,
        )

    if show_expired:
        expiry_preview = formatted.head(5) if isinstance(formatted, pd.DataFrame) else pd.DataFrame()
        expiry_details = []
        if isinstance(expiry_preview, pd.DataFrame) and not expiry_preview.empty:
            for record in expiry_preview.to_dict("records"):
                customer_label = clean_text(record.get("Customer")) or "(unknown)"
                product_label = clean_text(record.get("Product")) or clean_text(
                    record.get("Model")
                ) or ""
                expiry_label = clean_text(record.get("Expiry date")) or ""
                detail_line = " • ".join(
                    part for part in [customer_label, product_label, expiry_label] if part
                )
                expiry_details.append(detail_line)
        push_runtime_notification(
            "Expired warranties",
            f"{total_expired} warranty record(s) need attention ({month_expired} this month).",
            severity="warning",
            details=expiry_details,
        )

    for message in upcoming_messages:
        try:
            st.toast(message)
        except Exception:
            break
    try:
        if show_expired:
            st.toast(f"{total_expired} warranties require attention.")
    except Exception:
        pass

    st.session_state.just_logged_in = False


def _render_notification_entry(entry: dict[str, object], *, include_actor: bool = False) -> None:
    severity = str(entry.get("severity") or "info").lower()
    icon = {
        "warning": "🟠",
        "error": "🔴",
        "success": "🟢",
    }.get(severity, "🔵")
    title = clean_text(entry.get("title")) or "Notification"
    message = clean_text(entry.get("message")) or ""
    st.markdown(f"{icon} **{title}**")
    if message:
        st.write(message)
    details = entry.get("details") or []
    for detail in list(details)[:5]:
        st.caption(f"• {detail}")
    footer_bits: list[str] = []
    if include_actor:
        actor = clean_text(entry.get("actor"))
        if actor:
            footer_bits.append(actor)
    time_label = format_time_ago(entry.get("timestamp"))
    if time_label:
        footer_bits.append(time_label)
    if footer_bits:
        st.caption(" · ".join(footer_bits))


def _render_notification_section(
    entries: list[dict[str, object]],
    *,
    include_actor: bool = False,
    heading: Optional[str] = None,
) -> None:
    if not entries:
        return
    if heading:
        st.markdown(
            f"<div class='ps-notification-section-title'>{heading}</div>",
            unsafe_allow_html=True,
        )
    first = True
    for entry in entries:
        if not first:
            st.divider()
        _render_notification_entry(entry, include_actor=include_actor)
        first = False


def _render_notification_body(
    alerts: list[dict[str, object]],
    activity: list[dict[str, object]],
) -> None:
    if not alerts and not activity:
        st.caption("No notifications yet. Updates will appear here as your team works.")
        return
    _render_notification_section(alerts, heading="Alerts")
    if alerts and activity:
        st.divider()
    _render_notification_section(activity, include_actor=True, heading="Recent activity")


def render_notification_bell(conn) -> None:
    if not current_user_is_admin():
        return
    alerts = list(reversed(get_runtime_notifications()))
    activity = fetch_activity_feed(conn, limit=ACTIVITY_FEED_LIMIT)
    total = len(alerts) + len(activity)
    label = "🔔" if total == 0 else f"🔔 {total}"
    container = st.container()
    with container:
        st.markdown("<div class='ps-notification-popover'>", unsafe_allow_html=True)
        popover = getattr(st, "popover", None)
        if callable(popover):
            with popover(label, help="View alerts and staff activity", use_container_width=True):
                _render_notification_body(alerts, activity)
        else:
            with st.expander(f"{label} Notifications", expanded=False):
                _render_notification_body(alerts, activity)
        st.markdown("</div>", unsafe_allow_html=True)


def customers_page(conn):
    st.subheader("👥 Customers")
    feedback = st.session_state.pop("new_customer_feedback", None)
    if feedback:
        level, message = feedback
        if level == "success":
            st.success(message)
        elif level == "info":
            st.info(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.write(message)

    with st.expander("Add new customer"):
        products_state = st.session_state.get(
            "new_customer_products_rows",
            _default_new_customer_products(),
        )
        st.session_state.setdefault(
            "new_customer_products_rows", products_state
        )
        with st.form("new_customer"):
            name = st.text_input("Customer name *", key="new_customer_name")
            company = st.text_input(
                "Company name",
                key="new_customer_company",
                help="Optional organisation or business associated with this customer.",
            )
            phone = st.text_input("Phone", key="new_customer_phone")
            address = st.text_area(
                "Billing address",
                key="new_customer_address",
                help="Primary mailing or billing address for this customer.",
            )
            delivery_address = st.text_area(
                "Delivery address",
                key="new_customer_delivery_address",
                help="Where goods should be delivered. Leave blank if same as billing.",
            )
            purchase_date = st.date_input(
                "Purchase/Issue date",
                value=datetime.now().date(),
                key="new_customer_purchase_date",
            )
            remarks = st.text_area(
                "Remarks",
                key="new_customer_remarks",
                help="Internal notes or special instructions for this customer.",
            )
            amount_spent_input = st.number_input(
                "Amount spent",
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="new_customer_amount_spent",
                help="Record how much the customer has spent so far.",
            )
            st.markdown("#### Products / services purchased")
            st.caption(
                "Use the **Add row** option below to record each product or service purchased."
            )
            products_df = pd.DataFrame(products_state)
            required_columns = ["name", "model", "serial", "quantity"]
            for column in required_columns:
                if column not in products_df.columns:
                    default_value = 1 if column == "quantity" else ""
                    products_df[column] = default_value
            products_df = products_df[required_columns]
            edited_products = st.data_editor(
                products_df,
                key="new_customer_products_table",
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "name": st.column_config.TextColumn(
                        "Product / service",
                        help="Name or brief description of the item purchased.",
                    ),
                    "model": st.column_config.TextColumn(
                        "Model",
                        help="Add model or variant details to help identify the product.",
                    ),
                    "serial": st.column_config.TextColumn(
                        "Serial / ID",
                        help="Serial number or unique identifier (optional).",
                    ),
                    "quantity": st.column_config.NumberColumn(
                        "Quantity",
                        min_value=1,
                        step=1,
                        format="%d",
                    ),
                },
            )
            product_entries = edited_products.to_dict("records")
            st.session_state["new_customer_products_rows"] = product_entries
            with st.expander("Attachments & advanced details", expanded=True):
                do_code = st.text_input(
                    "Delivery order (DO) code",
                    key="new_customer_do_code",
                    help="Link the customer to an existing delivery order if available.",
                )
                sales_person_input = st.text_input(
                    "Sales person",
                    key="new_customer_sales_person",
                    help="Record who handled this sale for quick reference later.",
                )
                customer_pdf = st.file_uploader(
                    "Attach customer PDF",
                    type=["pdf"],
                    key="new_customer_pdf",
                    help="Upload signed agreements, invoices or other supporting paperwork.",
                )
                do_pdf = st.file_uploader(
                    "Attach Delivery Order (PDF)",
                    type=["pdf"],
                    key="new_customer_do_pdf",
                    help="Upload the delivery order so it is linked to this record.",
                )
            action_cols = st.columns((1, 1))
            submitted = action_cols[0].form_submit_button(
                "Save new customer", type="primary"
            )
            reset_form = action_cols[1].form_submit_button("Reset form")
            if reset_form:
                _reset_new_customer_form_state()
                st.session_state["new_customer_feedback"] = (
                    "info",
                    "Customer form cleared. You can start again with a blank form.",
                )
                _safe_rerun()
                return
            if submitted:
                errors: list[str] = []
                if not name.strip():
                    errors.append("Customer name is required before saving.")
                do_serial = clean_text(do_code)
                if do_pdf is not None and not do_serial:
                    errors.append(
                        "Enter a delivery order code before attaching a delivery order PDF."
                    )
                if errors:
                    for msg in errors:
                        st.error(msg)
                    return
                cur = conn.cursor()
                name_val = clean_text(name)
                company_val = clean_text(company)
                phone_val = clean_text(phone)
                address_val = clean_text(address)
                delivery_address_val = clean_text(delivery_address)
                remarks_val = clean_text(remarks)
                cleaned_products, product_labels = normalize_product_entries(product_entries)
                product_label = "\n".join(product_labels) if product_labels else None
                purchase_str = purchase_date.strftime("%Y-%m-%d") if purchase_date else None
                amount_value = parse_amount(amount_spent_input)
                if amount_value == 0.0 and (amount_spent_input is None or amount_spent_input == 0.0):
                    amount_value = None
                created_by = current_user_id()
                cur.execute(
                    "INSERT INTO customers (name, company_name, phone, address, delivery_address, remarks, purchase_date, product_info, delivery_order_code, sales_person, amount_spent, created_by, dup_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (
                        name_val,
                        company_val,
                        phone_val,
                        address_val,
                        delivery_address_val,
                        remarks_val,
                        purchase_str,
                        product_label,
                        do_serial,
                        clean_text(sales_person_input),
                        amount_value,
                        created_by,
                    ),
                )
                cid = cur.lastrowid
                conn.commit()
                if cleaned_products:
                    for prod in cleaned_products:
                        if not prod.get("name"):
                            continue
                        cur.execute(
                            "SELECT product_id FROM products WHERE name=? AND IFNULL(model,'')=IFNULL(?, '') LIMIT 1",
                            (prod.get("name"), prod.get("model")),
                        )
                        row = cur.fetchone()
                        if row:
                            pid = row[0]
                        else:
                            cur.execute(
                                "INSERT INTO products (name, model, serial) VALUES (?, ?, ?)",
                                (
                                    prod.get("name"),
                                    prod.get("model"),
                                    prod.get("serial"),
                                ),
                            )
                            pid = cur.lastrowid
                        issue = purchase_date.strftime("%Y-%m-%d") if purchase_date else None
                        expiry = (
                            (purchase_date + timedelta(days=365)).strftime("%Y-%m-%d")
                            if purchase_date
                            else None
                        )
                        cur.execute(
                            "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status, remarks) VALUES (?, ?, ?, ?, ?, 'active', ?)",
                            (cid, pid, prod.get("serial"), issue, expiry, remarks_val),
                        )
                    conn.commit()
                if do_serial:
                    stored_path = None
                    if do_pdf is not None:
                        safe_name = _sanitize_path_component(do_serial)
                        stored_path = store_uploaded_pdf(
                            do_pdf, DELIVERY_ORDER_DIR, filename=f"{safe_name}.pdf"
                        )
                    cur = conn.cursor()
                    existing = cur.execute(
                        "SELECT customer_id, file_path FROM delivery_orders WHERE do_number = ?",
                        (do_serial,),
                    ).fetchone()
                    product_summary = (
                        cleaned_products[0].get("name") if cleaned_products else product_label
                    )
                    sales_clean = clean_text(sales_person_input)
                    if existing:
                        existing_customer = existing[0]
                        existing_path = existing[1]
                        if existing_customer and int(existing_customer) != int(cid):
                            st.warning(
                                "Delivery order code already linked to another customer. Upload skipped."
                            )
                            if stored_path and stored_path != existing_path:
                                new_path = resolve_upload_path(stored_path)
                                if new_path and new_path.exists():
                                    try:
                                        new_path.unlink()
                                    except Exception:
                                        pass
                        else:
                            final_path = stored_path or existing_path
                            if stored_path and existing_path and stored_path != existing_path:
                                old_path = resolve_upload_path(existing_path)
                                if old_path and old_path.exists():
                                    try:
                                        old_path.unlink()
                                    except Exception:
                                        pass
                            conn.execute(
                                "UPDATE delivery_orders SET customer_id=?, description=?, sales_person=?, file_path=? WHERE do_number=?",
                                (
                                    cid,
                                    product_summary,
                                    sales_clean,
                                    final_path,
                                    do_serial,
                                ),
                            )
                            conn.commit()
                    else:
                        conn.execute(
                            "INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, remarks, file_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                do_serial,
                                cid,
                                None,
                                product_summary,
                                sales_clean,
                                remarks_val,
                                stored_path,
                            ),
                        )
                        conn.commit()
                if customer_pdf is not None:
                    stored_path = store_uploaded_pdf(
                        customer_pdf, CUSTOMER_DOCS_DIR, filename=f"customer_{cid}.pdf"
                    )
                    if stored_path:
                        conn.execute(
                            "UPDATE customers SET attachment_path=? WHERE customer_id=?",
                            (stored_path, cid),
                        )
                        conn.commit()
                if phone_val:
                    recalc_customer_duplicate_flag(conn, phone_val)
                    conn.commit()
                display_name = name_val or f"Customer #{int(cid)}"
                product_count = len(
                    [prod for prod in cleaned_products if prod.get("name")]
                )
                details = (
                    f"{display_name} with {product_count} product(s)"
                    if product_count
                    else display_name
                )
                log_activity(
                    conn,
                    event_type="customer_created",
                    description=f"Added {details}",
                    entity_type="customer",
                    entity_id=int(cid),
                )
                _reset_new_customer_form_state()
                st.session_state["new_customer_feedback"] = (
                    "success",
                    f"Customer {name_val or 'record'} saved successfully.",
                )
                _safe_rerun()
                return
    sort_dir = st.radio("Sort by created date", ["Newest first", "Oldest first"], horizontal=True)
    order = "DESC" if sort_dir == "Newest first" else "ASC"
    q = st.text_input("Search (name/phone/address/product/DO)")
    scope_clause, scope_params = customer_scope_filter("c")
    search_clause = dedent(
        """
        (? = ''
         OR c.name LIKE '%'||?||'%'
         OR c.company_name LIKE '%'||?||'%'
         OR c.phone LIKE '%'||?||'%'
         OR c.email LIKE '%'||?||'%'
         OR c.address LIKE '%'||?||'%'
         OR c.delivery_address LIKE '%'||?||'%'
         OR c.remarks LIKE '%'||?||'%'
         OR c.product_info LIKE '%'||?||'%'
         OR c.delivery_order_code LIKE '%'||?||'%'
         OR c.sales_person LIKE '%'||?||'%')
        """
    ).strip()
    where_parts = [search_clause]
    params: list[object] = [q, q, q, q, q, q, q, q, q, q, q]
    if scope_clause:
        where_parts.append(scope_clause)
        params.extend(scope_params)
    where_sql = " AND ".join(where_parts)
    df_raw = df_query(
        conn,
        f"""
        SELECT
            c.customer_id AS id,
            c.name,
            c.company_name,
            c.phone,
            c.email,
            c.address,
            c.delivery_address,
            c.remarks,
            c.purchase_date,
            c.product_info,
            c.delivery_order_code,
            c.sales_person,
            c.attachment_path,
            c.created_at,
            c.dup_flag,
            c.created_by,
            COALESCE(u.username, '(unknown)') AS uploaded_by
        FROM customers c
        LEFT JOIN users u ON u.user_id = c.created_by
        WHERE {where_sql}
        ORDER BY datetime(c.created_at) {order}
    """,
        tuple(params),
    )
    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"
    st.markdown("### Quick edit or delete")
    if df_raw.empty:
        st.info("No customers found for the current filters.")
    else:
        original_map: dict[int, dict] = {}
        for record in df_raw.to_dict("records"):
            cid = int_or_none(record.get("id"))
            if cid is not None:
                original_map[cid] = record
        editor_df = df_raw.copy()
        editor_df["purchase_date"] = pd.to_datetime(editor_df["purchase_date"], errors="coerce")
        editor_df["created_at"] = pd.to_datetime(editor_df["created_at"], errors="coerce")
        if "dup_flag" in editor_df.columns:
            editor_df["duplicate"] = editor_df["dup_flag"].apply(lambda x: "🔁 duplicate phone" if int_or_none(x) == 1 else "")
        else:
            editor_df["duplicate"] = ""
        editor_df["Action"] = "Keep"
        column_order = [
            col
            for col in [
                "id",
                "name",
                "company_name",
                "phone",
                "address",
                "delivery_address",
                "remarks",
                "purchase_date",
                "product_info",
                "delivery_order_code",
                "sales_person",
                "duplicate",
                "created_at",
                "uploaded_by",
                "Action",
            ]
            if col in editor_df.columns
        ]
        editor_df = editor_df[column_order]
        editor_state = st.data_editor(
            editor_df,
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "id": st.column_config.Column("ID", disabled=True),
                "name": st.column_config.TextColumn("Name"),
                "company_name": st.column_config.TextColumn(
                    "Company",
                    help="Optional organisation linked to the customer.",
                ),
                "phone": st.column_config.TextColumn("Phone"),
                "address": st.column_config.TextColumn("Billing address"),
                "delivery_address": st.column_config.TextColumn("Delivery address"),
                "remarks": st.column_config.TextColumn("Remarks"),
                "purchase_date": st.column_config.DateColumn("Purchase date", format="DD-MM-YYYY", required=False),
                "product_info": st.column_config.TextColumn("Product"),
                "delivery_order_code": st.column_config.TextColumn("DO code"),
                "sales_person": st.column_config.TextColumn("Sales person"),
                "duplicate": st.column_config.Column("Duplicate", disabled=True),
                "created_at": st.column_config.DatetimeColumn("Created", format="DD-MM-YYYY HH:mm", disabled=True),
                "uploaded_by": st.column_config.Column("Uploaded by", disabled=True),
                "Action": st.column_config.SelectboxColumn("Action", options=["Keep", "Delete"], required=True),
            },
        )
        if not is_admin:
            st.caption("Set Action to “Delete” requires admin access; non-admin changes will be ignored.")
        if is_admin and not editor_df.empty:
            delete_labels: dict[int, str] = {}
            for record in editor_df.to_dict("records"):
                cid = int_or_none(record.get("id"))
                if cid is None:
                    continue
                name_val = clean_text(record.get("name")) or "(no name)"
                phone_val = clean_text(record.get("phone")) or "-"
                delete_labels[cid] = f"#{cid} – {name_val} | {phone_val}"
            delete_choices = sorted(
                delete_labels.keys(), key=lambda cid: delete_labels[cid].lower()
            )
            with st.form("bulk_customer_delete"):
                selected_delete_ids = st.multiselect(
                    "Select customers to delete",
                    delete_choices,
                    format_func=lambda cid: delete_labels.get(
                        int(cid), f"Customer #{cid}"
                    ),
                    help="Removes the selected customers and their related records.",
                )
                bulk_delete_submit = st.form_submit_button(
                    "Delete selected customers",
                    disabled=not selected_delete_ids,
                    type="secondary",
                )
            if bulk_delete_submit and selected_delete_ids:
                deleted_count = 0
                for cid in selected_delete_ids:
                    try:
                        delete_customer_record(conn, int(cid))
                        deleted_count += 1
                    except Exception as err:
                        st.error(f"Unable to delete customer #{cid}: {err}")
                if deleted_count:
                    st.warning(f"Deleted {deleted_count} customer(s).")
                    _safe_rerun()
        if st.button("Apply table updates", type="primary"):
            editor_result = editor_state if isinstance(editor_state, pd.DataFrame) else pd.DataFrame(editor_state)
            if editor_result.empty:
                st.info("No rows to update.")
            else:
                phones_to_recalc: set[str] = set()
                updates = deletes = 0
                errors: list[str] = []
                made_updates = False
                activity_events: list[tuple[str, int, str]] = []
                for row in editor_result.to_dict("records"):
                    cid = int_or_none(row.get("id"))
                    if cid is None or cid not in original_map:
                        continue
                    action = str(row.get("Action") or "Keep").strip().lower()
                    if action == "delete":
                        if is_admin:
                            delete_customer_record(conn, cid)
                            deletes += 1
                        else:
                            errors.append(f"Only admins can delete customers (ID #{cid}).")
                        continue
                    new_name = clean_text(row.get("name"))
                    new_company = clean_text(row.get("company_name"))
                    new_phone = clean_text(row.get("phone"))
                    new_address = clean_text(row.get("address"))
                    new_delivery_address = clean_text(row.get("delivery_address"))
                    new_remarks = clean_text(row.get("remarks"))
                    purchase_str, _ = date_strings_from_input(row.get("purchase_date"))
                    product_label = clean_text(row.get("product_info"))
                    new_do = clean_text(row.get("delivery_order_code"))
                    new_sales_person = clean_text(row.get("sales_person"))
                    original_row = original_map[cid]
                    old_name = clean_text(original_row.get("name"))
                    old_company = clean_text(original_row.get("company_name"))
                    old_phone = clean_text(original_row.get("phone"))
                    old_address = clean_text(original_row.get("address"))
                    old_delivery_address = clean_text(original_row.get("delivery_address"))
                    old_remarks = clean_text(original_row.get("remarks"))
                    old_purchase = clean_text(original_row.get("purchase_date"))
                    old_product = clean_text(original_row.get("product_info"))
                    old_do = clean_text(original_row.get("delivery_order_code"))
                    old_sales_person = clean_text(original_row.get("sales_person"))
                    changes: list[str] = []
                    if (
                        new_name == old_name
                        and new_company == old_company
                        and new_phone == old_phone
                        and new_address == old_address
                        and new_delivery_address == old_delivery_address
                        and new_remarks == old_remarks
                        and purchase_str == old_purchase
                        and product_label == old_product
                        and new_do == old_do
                        and new_sales_person == old_sales_person
                    ):
                        continue
                    conn.execute(
                        "UPDATE customers SET name=?, company_name=?, phone=?, address=?, delivery_address=?, remarks=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, dup_flag=0 WHERE customer_id=?",
                        (
                            new_name,
                            new_company,
                            new_phone,
                            new_address,
                            new_delivery_address,
                            new_remarks,
                            purchase_str,
                            product_label,
                            new_do,
                            new_sales_person,
                            cid,
                        ),
                    )
                    if new_name != old_name:
                        changes.append("name")
                    if new_company != old_company:
                        changes.append("company")
                    if new_phone != old_phone:
                        changes.append("phone")
                    if new_address != old_address:
                        changes.append("billing address")
                    if new_delivery_address != old_delivery_address:
                        changes.append("delivery address")
                    if new_remarks != old_remarks:
                        changes.append("remarks")
                    if purchase_str != old_purchase:
                        changes.append("purchase date")
                    if product_label != old_product:
                        changes.append("products")
                    if new_do != old_do:
                        changes.append("DO code")
                    if new_sales_person != old_sales_person:
                        changes.append("sales person")
                    if new_do:
                        conn.execute(
                            """
                            INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, remarks, file_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(do_number) DO UPDATE SET
                                customer_id=excluded.customer_id,
                                description=excluded.description,
                                sales_person=excluded.sales_person,
                                remarks=excluded.remarks
                            """,
                            (
                                new_do,
                                cid,
                                None,
                                product_label,
                                new_sales_person,
                                new_remarks,
                                None,
                            ),
                        )
                    if old_do and old_do != new_do:
                        conn.execute(
                            "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                            (old_do, cid),
                        )
                    conn.execute(
                        "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                        (
                            new_name,
                            new_phone,
                            new_address,
                            product_label,
                            new_do,
                            purchase_str,
                            cid,
                        ),
                    )
                    if old_phone and old_phone != new_phone:
                        phones_to_recalc.add(old_phone)
                    if new_phone:
                        phones_to_recalc.add(new_phone)
                    updates += 1
                    made_updates = True
                    if changes:
                        display_name = new_name or old_name or f"Customer #{cid}"
                        summary = ", ".join(changes)
                        activity_events.append(
                            (
                                "customer_updated",
                                cid,
                                f"Updated {display_name} ({summary})",
                            )
                        )
                if made_updates:
                    conn.commit()
                if phones_to_recalc:
                    for phone_value in phones_to_recalc:
                        recalc_customer_duplicate_flag(conn, phone_value)
                    conn.commit()
                for event_type, entity_id, description in activity_events:
                    log_activity(
                        conn,
                        event_type=event_type,
                        description=description,
                        entity_type="customer",
                        entity_id=int(entity_id),
                    )
                if errors:
                    for err in errors:
                        st.error(err)
                if updates or deletes:
                    st.success(f"Updated {updates} row(s) and deleted {deletes} row(s).")
                    if not errors:
                        _safe_rerun()
                elif not errors:
                    st.info("No changes detected.")
    st.markdown("### Detailed editor & attachments")
    with st.expander("Open detailed editor", expanded=False):
        df_form = fmt_dates(df_raw.copy(), ["created_at", "purchase_date"])
        if df_form.empty:
            st.info("No customers to edit yet.")
        else:
            records_fmt = df_form.to_dict("records")
            raw_map = {int(row["id"]): row for row in df_raw.to_dict("records") if int_or_none(row.get("id")) is not None}
            option_ids = [int(row["id"]) for row in records_fmt]
            labels = {}
            for row in records_fmt:
                cid = int(row["id"])
                label_name = clean_text(row.get("name")) or "(no name)"
                label_phone = clean_text(row.get("phone")) or "-"
                labels[cid] = f"{label_name} – {label_phone}"
            selected_customer_id = st.selectbox(
                "Select customer",
                option_ids,
                format_func=lambda cid: labels.get(int(cid), str(cid)),
            )
            selected_raw = raw_map[int(selected_customer_id)]
            selected_fmt = next(r for r in records_fmt if int(r["id"]) == int(selected_customer_id))
            attachment_path = selected_raw.get("attachment_path")
            resolved_attachment = resolve_upload_path(attachment_path)
            if resolved_attachment and resolved_attachment.exists():
                st.download_button(
                    "Download current PDF",
                    data=resolved_attachment.read_bytes(),
                    file_name=resolved_attachment.name,
                    key=f"cust_pdf_dl_{selected_customer_id}",
                )
            else:
                st.caption("No customer PDF attached yet.")
            is_admin = user.get("role") == "admin"
            uploader_name = clean_text(selected_raw.get("uploaded_by"))
            if is_admin:
                st.caption(f"Uploaded by: {uploader_name or '(unknown)'}")
            with st.form(f"edit_customer_{selected_customer_id}"):
                name_edit = st.text_input("Name", value=clean_text(selected_raw.get("name")) or "")
                company_edit = st.text_input(
                    "Company",
                    value=clean_text(selected_raw.get("company_name")) or "",
                )
                phone_edit = st.text_input("Phone", value=clean_text(selected_raw.get("phone")) or "")
                email_edit = st.text_input("Email", value=clean_text(selected_raw.get("email")) or "")
                address_edit = st.text_area(
                    "Billing address",
                    value=clean_text(selected_raw.get("address")) or "",
                )
                delivery_address_edit = st.text_area(
                    "Delivery address",
                    value=clean_text(selected_raw.get("delivery_address")) or "",
                )
                remarks_edit = st.text_area(
                    "Remarks",
                    value=clean_text(selected_raw.get("remarks")) or "",
                )
                purchase_edit = st.text_input(
                    "Purchase date (DD-MM-YYYY)", value=clean_text(selected_fmt.get("purchase_date")) or ""
                )
                product_edit = st.text_input("Product", value=clean_text(selected_raw.get("product_info")) or "")
                do_edit = st.text_input(
                    "Delivery order code", value=clean_text(selected_raw.get("delivery_order_code")) or ""
                )
                sales_person_edit = st.text_input(
                    "Sales person", value=clean_text(selected_raw.get("sales_person")) or ""
                )
                new_pdf = st.file_uploader(
                    "Attach/replace customer PDF", type=["pdf"], key=f"edit_customer_pdf_{selected_customer_id}"
                )
                col1, col2 = st.columns(2)
                save_customer = col1.form_submit_button("Save changes", type="primary")
                delete_customer = col2.form_submit_button("Delete customer", disabled=not is_admin)
            if save_customer:
                old_phone = clean_text(selected_raw.get("phone"))
                new_name = clean_text(name_edit)
                new_company = clean_text(company_edit)
                new_phone = clean_text(phone_edit)
                new_email = clean_text(email_edit)
                new_address = clean_text(address_edit)
                new_delivery_address = clean_text(delivery_address_edit)
                new_remarks = clean_text(remarks_edit)
                purchase_str, _ = date_strings_from_input(purchase_edit)
                product_label = clean_text(product_edit)
                new_do = clean_text(do_edit)
                old_do = clean_text(selected_raw.get("delivery_order_code"))
                new_sales_person = clean_text(sales_person_edit)
                new_attachment_path = attachment_path
                if new_pdf is not None:
                    stored_path = store_uploaded_pdf(
                        new_pdf,
                        CUSTOMER_DOCS_DIR,
                        filename=f"customer_{selected_customer_id}.pdf",
                    )
                    if stored_path:
                        new_attachment_path = stored_path
                        if attachment_path:
                            old_path = resolve_upload_path(attachment_path)
                            if old_path and old_path.exists():
                                new_path = resolve_upload_path(stored_path)
                                if not new_path or new_path != old_path:
                                    try:
                                        old_path.unlink()
                                    except Exception:
                                        pass
                conn.execute(
                    "UPDATE customers SET name=?, company_name=?, phone=?, email=?, address=?, delivery_address=?, remarks=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, attachment_path=?, dup_flag=0 WHERE customer_id=?",
                    (
                        new_name,
                        new_company,
                        new_phone,
                        new_email,
                        new_address,
                        new_delivery_address,
                        new_remarks,
                        purchase_str,
                        product_label,
                        new_do,
                        new_sales_person,
                        new_attachment_path,
                        int(selected_customer_id),
                    ),
                )
                if new_do:
                    conn.execute(
                        """
                        INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, remarks, file_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(do_number) DO UPDATE SET
                            customer_id=excluded.customer_id,
                            description=excluded.description,
                            sales_person=excluded.sales_person,
                            remarks=excluded.remarks
                        """,
                        (
                            new_do,
                            int(selected_customer_id),
                            None,
                            product_label,
                            new_sales_person,
                            new_remarks,
                            None,
                        ),
                    )
                if old_do and old_do != new_do:
                    conn.execute(
                        "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                        (old_do, int(selected_customer_id)),
                    )
                conn.execute(
                    "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                    (
                        new_name,
                        new_phone,
                        new_address,
                        product_label,
                        new_do,
                        purchase_str,
                        int(selected_customer_id),
                    ),
                )
                conn.commit()
                if old_phone and old_phone != new_phone:
                    recalc_customer_duplicate_flag(conn, old_phone)
                if new_phone:
                    recalc_customer_duplicate_flag(conn, new_phone)
                conn.commit()
                st.success("Customer updated.")
                _safe_rerun()
            if delete_customer:
                if is_admin:
                    delete_customer_record(conn, int(selected_customer_id))
                    st.warning("Customer deleted.")
                    _safe_rerun()
                else:
                    st.error("Only admins can delete customers.")
            st.markdown("#### Follow-ups & reminders")
            with st.form(f"customer_note_add_{selected_customer_id}"):
                new_note_text = st.text_area(
                    "Add a remark", placeholder="e.g. Call back next week with pricing update"
                )
                enable_follow_up = st.checkbox(
                    "Schedule follow-up reminder",
                    value=False,
                    key=f"customer_note_followup_{selected_customer_id}",
                )
                default_date = datetime.now().date()
                reminder_date = st.date_input(
                    "Reminder date",
                    value=default_date,
                    key=f"customer_note_reminder_{selected_customer_id}",
                    disabled=not enable_follow_up,
                )
                add_note = st.form_submit_button("Add remark", type="primary")
            if add_note:
                note_value = clean_text(new_note_text)
                if not note_value:
                    st.error("Remark text is required.")
                else:
                    reminder_value = to_iso_date(reminder_date) if enable_follow_up else None
                    conn.execute(
                        "INSERT INTO customer_notes (customer_id, note, remind_on) VALUES (?, ?, ?)",
                        (int(selected_customer_id), note_value, reminder_value),
                    )
                    conn.commit()
                    st.success("Remark added.")
                    _safe_rerun()

            notes_df = df_query(
                conn,
                """
                SELECT note_id, note, remind_on, is_done, created_at, updated_at
                FROM customer_notes
                WHERE customer_id = ?
                ORDER BY datetime(COALESCE(remind_on, created_at)) DESC, note_id DESC
                """,
                (int(selected_customer_id),),
            )
            if notes_df.empty:
                st.caption("No saved remarks yet.")
            else:
                notes_original = {
                    int(row["note_id"]): row for row in notes_df.to_dict("records") if int_or_none(row.get("note_id")) is not None
                }
                editor_df = notes_df.copy()
                editor_df["remind_on"] = pd.to_datetime(editor_df["remind_on"], errors="coerce")
                editor_df["created_at"] = pd.to_datetime(editor_df["created_at"], errors="coerce")
                editor_df["updated_at"] = pd.to_datetime(editor_df["updated_at"], errors="coerce")
                editor_df["Done"] = editor_df.get("is_done", 0).fillna(0).astype(int).apply(lambda v: bool(v))
                editor_df["Action"] = "Keep"
                column_order = [
                    col
                    for col in [
                        "note_id",
                        "note",
                        "remind_on",
                        "Done",
                        "created_at",
                        "updated_at",
                        "Action",
                    ]
                    if col in editor_df.columns
                ]
                editor_view = editor_df[column_order]
                note_editor_state = st.data_editor(
                    editor_view,
                    hide_index=True,
                    num_rows="fixed",
                    use_container_width=True,
                    column_config={
                        "note_id": st.column_config.Column("ID", disabled=True),
                        "note": st.column_config.TextColumn("Remark"),
                        "remind_on": st.column_config.DateColumn(
                            "Reminder date", format="DD-MM-YYYY", required=False
                        ),
                        "Done": st.column_config.CheckboxColumn("Completed"),
                        "created_at": st.column_config.DatetimeColumn(
                            "Created", format="DD-MM-YYYY HH:mm", disabled=True
                        ),
                        "updated_at": st.column_config.DatetimeColumn(
                            "Updated", format="DD-MM-YYYY HH:mm", disabled=True
                        ),
                        "Action": st.column_config.SelectboxColumn(
                            "Action", options=["Keep", "Delete"], required=True
                        ),
                    },
                    key=f"customer_notes_editor_{selected_customer_id}",
                )
                if st.button(
                    "Apply note updates",
                    key=f"apply_customer_notes_{selected_customer_id}",
                ):
                    note_result = (
                        note_editor_state
                        if isinstance(note_editor_state, pd.DataFrame)
                        else pd.DataFrame(note_editor_state)
                    )
                    if note_result.empty:
                        st.info("No notes to update.")
                    else:
                        changes = False
                        errors: list[str] = []
                        for row in note_result.to_dict("records"):
                            note_id = int_or_none(row.get("note_id"))
                            if note_id is None or note_id not in notes_original:
                                continue
                            action = str(row.get("Action") or "Keep").strip().lower()
                            if action == "delete":
                                conn.execute(
                                    "DELETE FROM customer_notes WHERE note_id = ? AND customer_id = ?",
                                    (note_id, int(selected_customer_id)),
                                )
                                changes = True
                                continue
                            new_note_text = clean_text(row.get("note"))
                            if not new_note_text:
                                errors.append(f"Remark #{note_id} cannot be empty.")
                                continue
                            reminder_iso = to_iso_date(row.get("remind_on"))
                            completed_flag = bool(row.get("Done"))
                            original = notes_original[note_id]
                            original_note = clean_text(original.get("note"))
                            original_reminder = to_iso_date(original.get("remind_on"))
                            original_done = bool(int_or_none(original.get("is_done")) or 0)
                            if (
                                new_note_text == original_note
                                and reminder_iso == original_reminder
                                and completed_flag == original_done
                            ):
                                continue
                            conn.execute(
                                """
                                UPDATE customer_notes
                                SET note = ?, remind_on = ?, is_done = ?, updated_at = datetime('now')
                                WHERE note_id = ? AND customer_id = ?
                                """,
                                (
                                    new_note_text,
                                    reminder_iso,
                                    1 if completed_flag else 0,
                                    note_id,
                                    int(selected_customer_id),
                                ),
                            )
                            changes = True
                        if errors:
                            for err in errors:
                                st.error(err)
                        if changes and not errors:
                            conn.commit()
                            st.success("Notes updated.")
                            _safe_rerun()
                        elif not changes and not errors:
                            st.info("No changes detected.")
                        elif changes:
                            conn.commit()
                            st.warning("Some changes were saved, but please review the errors above.")
    st.markdown("**Recently Added Customers**")
    recent_where = f"WHERE {scope_clause}" if scope_clause else ""
    recent_params = scope_params if scope_clause else ()
    recent_df = df_query(
        conn,
        f"""
        SELECT
            c.customer_id AS id,
            c.name,
            c.company_name,
            c.phone,
            c.email,
            c.address,
            c.delivery_address,
            c.remarks,
            c.purchase_date,
            c.product_info,
            c.delivery_order_code,
            c.sales_person,
            c.amount_spent,
            c.created_at,
            COALESCE(u.username, '(unknown)') AS uploaded_by
        FROM customers c
        LEFT JOIN users u ON u.user_id = c.created_by
        {recent_where}
        ORDER BY datetime(c.created_at) DESC LIMIT 200
    """,
        recent_params,
    )
    recent_df = fmt_dates(recent_df, ["created_at", "purchase_date"])
    recent_df = recent_df.rename(
        columns={
            "sales_person": "Sales person",
            "amount_spent": "Amount spent",
            "uploaded_by": "Uploaded by",
        }
    )
    st.dataframe(recent_df.drop(columns=["id"], errors="ignore"))
def warranties_page(conn):
    st.subheader("🛡️ Warranties")
    sort_dir = st.radio("Sort by expiry date", ["Soonest first", "Latest first"], horizontal=True)
    order = "ASC" if sort_dir == "Soonest first" else "DESC"
    q = st.text_input("Search (customer/product/model/serial)")

    base = dedent(
        """
        SELECT w.warranty_id as id, c.name as customer, p.name as product, p.model, w.serial,
               w.issue_date, w.expiry_date, w.status, w.remarks, w.dup_flag
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE {filters}
        ORDER BY date(w.expiry_date) {order}
        """
    )

    search_filter = "(? = '' OR c.name LIKE '%'||?||'%' OR p.name LIKE '%'||?||'%' OR p.model LIKE '%'||?||'%' OR w.serial LIKE '%'||?||'%')"
    status_filter = "(w.status IS NULL OR w.status <> 'deleted')"
    scope_clause, scope_params = customer_scope_filter("c")

    def build_filters(date_condition: str) -> tuple[str, tuple[object, ...]]:
        clauses = [search_filter, status_filter, date_condition]
        params = [q, q, q, q, q]
        if scope_clause:
            clauses.append(scope_clause)
            params.extend(scope_params)
        return " AND ".join(clauses), tuple(params)

    active_filters, active_params = build_filters("date(w.expiry_date) >= date('now')")
    active_query = base.format(filters=active_filters, order=order)
    active = df_query(conn, active_query, active_params)
    active = fmt_dates(active, ["issue_date","expiry_date"])
    if "dup_flag" in active.columns:
        active = active.assign(Duplicate=active["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
        active.drop(columns=["dup_flag"], inplace=True)
    active = format_warranty_table(active)
    st.markdown("**Active Warranties**")
    st.dataframe(active, use_container_width=True)

    expired_filters, expired_params = build_filters("date(w.expiry_date) < date('now')")
    expired_query = base.format(filters=expired_filters, order="DESC")
    expired = df_query(conn, expired_query, expired_params)
    expired = fmt_dates(expired, ["issue_date","expiry_date"])
    if "dup_flag" in expired.columns:
        expired = expired.assign(Duplicate=expired["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
        expired.drop(columns=["dup_flag"], inplace=True)
    expired = format_warranty_table(expired)
    st.markdown("**Expired Warranties**")
    st.dataframe(expired, use_container_width=True)

    st.markdown("---")
    st.subheader("🔔 Upcoming Expiries")
    col1, col2 = st.columns(2)
    soon3 = collapse_warranty_rows(fetch_warranty_window(conn, 0, 3))
    soon60 = collapse_warranty_rows(fetch_warranty_window(conn, 0, 60))
    with col1:
        st.caption("Next **3** days")
        st.dataframe(soon3, use_container_width=True)
    with col2:
        st.caption("Next **60** days")
        st.dataframe(soon60, use_container_width=True)


def _render_service_section(conn, *, show_heading: bool = True):
    if show_heading:
        st.subheader("🛠️ Service Records")
    _, customer_label_map = build_customer_groups(conn, only_complete=False)
    customer_options, customer_labels, _, label_by_id = fetch_customer_choices(conn)
    do_df = df_query(
        conn,
        """
        SELECT d.do_number, d.customer_id, COALESCE(c.name, '(unknown)') AS customer_name, d.description, d.remarks
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        ORDER BY datetime(d.created_at) DESC
        """,
    )
    allowed_customers = accessible_customer_ids(conn)
    if allowed_customers is not None:
        do_df = do_df[do_df["customer_id"].apply(lambda value: int(value) in allowed_customers if pd.notna(value) else False)]
    do_options = [None]
    do_labels = {None: "No delivery order (manual entry)"}
    do_customer_map = {}
    do_customer_name_map = {}
    for _, row in do_df.iterrows():
        do_num = clean_text(row.get("do_number"))
        if not do_num:
            continue
        cust_id = int(row["customer_id"]) if not pd.isna(row.get("customer_id")) else None
        summary = clean_text(row.get("description"))
        cust_name = customer_label_map.get(cust_id) if cust_id else clean_text(row.get("customer_name"))
        label_parts = [do_num]
        if cust_name:
            label_parts.append(f"({cust_name})")
        if summary:
            snippet = summary[:40]
            if len(summary) > 40:
                snippet += "…"
            label_parts.append(f"– {snippet}")
        label = " ".join(part for part in label_parts if part)
        do_options.append(do_num)
        do_labels[do_num] = label
        do_customer_map[do_num] = cust_id
        do_customer_name_map[do_num] = cust_name or "(not linked)"

    with st.form("service_form"):
        selected_do = st.selectbox(
            "Delivery order",
            options=do_options,
            format_func=lambda do: do_labels.get(do, str(do)),
        )
        default_customer = do_customer_map.get(selected_do)
        state_key = "service_customer_link"
        last_do_key = "service_customer_last_do"
        linked_customer = default_customer
        if default_customer is not None:
            st.session_state[last_do_key] = selected_do
            st.session_state[state_key] = default_customer
            customer_label = (
                customer_labels.get(default_customer)
                or customer_label_map.get(default_customer)
                or label_by_id.get(default_customer)
                or do_customer_name_map.get(selected_do)
                or f"Customer #{default_customer}"
            )
            st.text_input("Customer", value=customer_label, disabled=True)
        else:
            choices = list(customer_options)
            if st.session_state.get(last_do_key) != selected_do:
                st.session_state[last_do_key] = selected_do
                st.session_state[state_key] = None
            linked_customer = st.selectbox(
                "Customer *",
                options=choices,
                format_func=lambda cid: customer_labels.get(cid, "-- Select customer --"),
                key=state_key,
            )
        status_value = status_input_widget("service_new", DEFAULT_SERVICE_STATUS)
        status_choice = get_status_choice("service_new")
        today = datetime.now().date()
        if status_choice == "Completed":
            service_period_value = st.date_input(
                "Service period",
                value=(today, today),
                help="Select the start and end dates for the completed service.",
                key="service_new_period_completed",
            )
        elif status_choice == "In progress":
            service_period_value = st.date_input(
                "Service start date",
                value=today,
                help="Choose when this service work began.",
                key="service_new_period_start",
            )
        else:
            service_period_value = st.date_input(
                "Planned start date",
                value=today,
                help="Select when this service is scheduled to begin.",
                key="service_new_period_planned",
            )
        description = st.text_area("Service description")
        remarks = st.text_area("Remarks / updates")
        cond_cols = st.columns(2)
        with cond_cols[0]:
            condition_option = st.selectbox(
                "Generator condition after work",
                ["Not recorded"] + GENERATOR_CONDITION_OPTIONS,
                index=0,
                key="service_new_condition",
                help="Capture the condition of the generator once the work is completed.",
            )
        with cond_cols[1]:
            bill_amount_input = st.number_input(
                "Bill amount",
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key="service_new_bill_amount",
                help="Track how much the customer was billed for this service.",
            )
        condition_notes = st.text_area(
            "Condition remarks",
            key="service_new_condition_notes",
            help="Add any notes about the generator condition once the job is done.",
        )
        service_product_count = st.number_input(
            "Products sold during service",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="service_additional_product_count",
            help="Capture any new items sold while this service was in progress.",
        )
        service_product_entries: list[dict[str, object]] = []
        for idx in range(int(service_product_count)):
            cols = st.columns((2, 2, 2, 1))
            with cols[0]:
                product_name = st.text_input(
                    f"Product {idx + 1} details",
                    key=f"service_product_name_{idx}",
                )
            with cols[1]:
                product_model = st.text_input(
                    f"Model {idx + 1}",
                    key=f"service_product_model_{idx}",
                )
            with cols[2]:
                product_serial = st.text_input(
                    f"Serial {idx + 1}",
                    key=f"service_product_serial_{idx}",
                )
            with cols[3]:
                product_quantity = st.number_input(
                    f"Qty {idx + 1}",
                    min_value=1,
                    max_value=999,
                    value=1,
                    step=1,
                    key=f"service_product_quantity_{idx}",
                )
            service_product_entries.append(
                {
                    "name": product_name,
                    "model": product_model,
                    "serial": product_serial,
                    "quantity": int(product_quantity),
                }
        )
        service_files = st.file_uploader(
            "Attach service documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="service_new_docs",
        )
        bill_file = st.file_uploader(
            "Upload bill / invoice (PDF)",
            type=["pdf"],
            key="service_new_bill_file",
            help="Store the supporting invoice for this service.",
        )
        submit = st.form_submit_button("Log service", type="primary")

    if submit:
        selected_customer = (
            linked_customer if linked_customer is not None else do_customer_map.get(selected_do)
        )
        selected_customer = int(selected_customer) if selected_customer is not None else None
        cur = conn.cursor()
        (
            service_date_str,
            service_start_str,
            service_end_str,
        ) = determine_period_strings(status_choice, service_period_value)
        valid_entry = True
        if selected_customer is None:
            st.error("Select a customer to log this service entry.")
            valid_entry = False
        if status_choice == "Completed" and (
            not service_start_str or not service_end_str
        ):
            st.error("Start and end dates are required for completed services.")
            valid_entry = False
        if status_choice != "Completed" and not service_start_str:
            st.error("Select a start date for this service entry.")
            valid_entry = False
        if valid_entry:
            _cleaned_service_products, service_product_labels = normalize_product_entries(
                service_product_entries
            )
            service_product_label = (
                "\n".join(service_product_labels) if service_product_labels else None
            )
            condition_value = (
                condition_option if condition_option in GENERATOR_CONDITION_OPTIONS else None
            )
            condition_notes_value = clean_text(condition_notes)
            bill_amount_value = None
            try:
                if bill_amount_input is not None and float(bill_amount_input) > 0:
                    bill_amount_value = round(float(bill_amount_input), 2)
            except Exception:
                bill_amount_value = None
            cur.execute(
                """
                INSERT INTO services (
                    do_number,
                    customer_id,
                    service_date,
                    service_start_date,
                    service_end_date,
                    description,
                    status,
                    remarks,
                    service_product_info,
                    condition_status,
                    condition_remarks,
                    bill_amount,
                    bill_document_path,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selected_do,
                    selected_customer,
                    service_date_str,
                    service_start_str,
                    service_end_str,
                    clean_text(description),
                    status_value,
                    clean_text(remarks),
                    service_product_label,
                    condition_value,
                    condition_notes_value,
                    bill_amount_value,
                    None,
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            service_id = cur.lastrowid
            if selected_do and selected_customer is not None:
                link_delivery_order_to_customer(conn, selected_do, selected_customer)
                saved_docs = attach_documents(
                    conn,
                    "service_documents",
                    "service_id",
                    service_id,
                    service_files,
                    SERVICE_DOCS_DIR,
                    f"service_{service_id}",
                )
                bill_saved = False
                if bill_file is not None:
                    stored_path = store_uploaded_pdf(
                        bill_file,
                        SERVICE_BILL_DIR,
                        filename=f"service_{service_id}_bill.pdf",
                    )
                    if stored_path:
                        conn.execute(
                            "UPDATE services SET bill_document_path = ? WHERE service_id = ?",
                            (stored_path, int(service_id)),
                        )
                        bill_saved = True
                conn.commit()
                service_label = do_labels.get(selected_do) if selected_do else None
                if not service_label:
                    service_label = f"Service #{service_id}"
                customer_name = None
                if selected_customer is not None:
                    customer_name = (
                        label_by_id.get(int(selected_customer))
                        or customer_label_map.get(int(selected_customer))
                    )
                summary_parts = [service_label]
                if customer_name:
                    summary_parts.append(customer_name)
                status_label = clean_text(status_value) or DEFAULT_SERVICE_STATUS
                summary_parts.append(f"status {status_label}")
                log_activity(
                    conn,
                    event_type="service_created",
                    description=" – ".join(summary_parts),
                    entity_type="service",
                    entity_id=int(service_id),
                )
                message = "Service record saved."
                if saved_docs:
                    message = f"{message} Attached {saved_docs} document(s)."
                if bill_amount_value is not None:
                    message = f"{message} Recorded bill amount {format_money(bill_amount_value)}."
                if bill_saved:
                    message = f"{message} Invoice uploaded."
                st.success(message)
                _safe_rerun()

    service_df = df_query(
        conn,
        """
        SELECT s.service_id,
               s.customer_id,
               d.customer_id AS do_customer_id,
               s.do_number,
               s.service_date,
               s.service_start_date,
               s.service_end_date,
               s.service_product_info,
               s.description,
               s.status,
               s.remarks,
               s.condition_status,
               s.condition_remarks,
               s.bill_amount,
               s.bill_document_path,
               s.updated_at,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(sd.document_id) AS doc_count
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = s.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN service_documents sd ON sd.service_id = s.service_id
        GROUP BY s.service_id
        ORDER BY datetime(COALESCE(s.service_start_date, s.service_date)) DESC, s.service_id DESC
        """,
    )
    if allowed_customers is not None:
        def _service_row_allowed(row):
            service_cust = row.get("customer_id")
            do_cust = row.get("do_customer_id")
            candidates = []
            if pd.notna(service_cust):
                candidates.append(int(service_cust))
            if pd.notna(do_cust):
                candidates.append(int(do_cust))
            return any(cid in allowed_customers for cid in candidates)

        service_df = service_df[service_df.apply(_service_row_allowed, axis=1)]
    if not service_df.empty:
        service_df = fmt_dates(service_df, ["service_date", "service_start_date", "service_end_date"])
        service_df["service_period"] = service_df.apply(
            lambda row: format_period_span(
                row.get("service_start_date"), row.get("service_end_date")
            ),
            axis=1,
        )
        service_df.drop(columns=["customer_id", "do_customer_id"], inplace=True, errors="ignore")
        service_df["Last update"] = pd.to_datetime(service_df.get("updated_at"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M")
        service_df.loc[service_df["Last update"].isna(), "Last update"] = None
        if "status" in service_df.columns:
            service_df["status"] = service_df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
        if "condition_status" in service_df.columns:
            service_df["condition_status"] = service_df["condition_status"].apply(
                lambda x: clean_text(x) or "Not recorded"
            )
        if "bill_amount" in service_df.columns:
            service_df["bill_amount_display"] = service_df["bill_amount"].apply(format_money)
        if "bill_document_path" in service_df.columns:
            service_df["bill_document_display"] = service_df["bill_document_path"].apply(
                lambda x: "📄" if clean_text(x) else ""
            )
        display = service_df.rename(
            columns={
                "do_number": "DO Serial",
                "service_date": "Service date",
                "service_start_date": "Service start date",
                "service_end_date": "Service end date",
                "service_period": "Service period",
                "service_product_info": "Products sold",
                "description": "Description",
                "status": "Status",
                "remarks": "Remarks",
                "condition_status": "Condition",
                "condition_remarks": "Condition notes",
                "bill_amount_display": "Bill amount",
                "bill_document_display": "Bill document",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        display = display.drop(columns=["bill_document_path"], errors="ignore")
        st.markdown("### Service history")
        st.dataframe(
            display.drop(columns=["updated_at", "service_id"], errors="ignore"),
            use_container_width=True,
        )

        records = service_df.to_dict("records")
        st.markdown("#### Update status & remarks")
        options = [int(r["service_id"]) for r in records]
        def service_label(record):
            do_ref = clean_text(record.get("do_number")) or "(no DO)"
            date_ref = clean_text(record.get("service_period")) or clean_text(
                record.get("service_date")
            )
            customer_ref = clean_text(record.get("customer"))
            parts = [do_ref]
            if date_ref:
                parts.append(f"· {date_ref}")
            if customer_ref:
                parts.append(f"· {customer_ref}")
            return " ".join(parts)

        labels = {int(r["service_id"]): service_label(r) for r in records}
        selected_service_id = st.selectbox(
            "Select service entry",
            options,
            format_func=lambda rid: labels.get(rid, str(rid)),
        )
        selected_record = next(r for r in records if int(r["service_id"]) == int(selected_service_id))
        new_status = status_input_widget(
            f"service_edit_{selected_service_id}", selected_record.get("status")
        )
        edit_status_choice = get_status_choice(f"service_edit_{selected_service_id}")
        existing_start = ensure_date(selected_record.get("service_start_date")) or ensure_date(
            selected_record.get("service_date")
        )
        existing_end = ensure_date(selected_record.get("service_end_date")) or existing_start
        today = datetime.now().date()
        default_start = existing_start or today
        default_end = existing_end or default_start
        if edit_status_choice == "Completed":
            edit_period_value = st.date_input(
                "Service period",
                value=(default_start, default_end),
                key=f"service_edit_{selected_service_id}_period_completed",
                help="Update the start and end dates for this service.",
            )
        elif edit_status_choice == "In progress":
            edit_period_value = st.date_input(
                "Service start date",
                value=default_start,
                key=f"service_edit_{selected_service_id}_period_start",
                help="Adjust when this service began.",
            )
        else:
            edit_period_value = st.date_input(
                "Planned start date",
                value=default_start,
                key=f"service_edit_{selected_service_id}_period_planned",
                help="Adjust when this service is scheduled to begin.",
            )
        new_remarks = st.text_area(
            "Remarks",
            value=clean_text(selected_record.get("remarks")) or "",
            key=f"service_edit_{selected_service_id}",
        )
        condition_cols = st.columns(2)
        existing_condition = clean_text(selected_record.get("condition_status"))
        condition_options = ["Not recorded"] + GENERATOR_CONDITION_OPTIONS
        default_condition = (
            existing_condition if existing_condition in GENERATOR_CONDITION_OPTIONS else "Not recorded"
        )
        with condition_cols[0]:
            condition_choice_edit = st.selectbox(
                "Generator condition",
                condition_options,
                index=condition_options.index(default_condition),
                key=f"service_edit_condition_{selected_service_id}",
            )
        existing_bill_amount = selected_record.get("bill_amount")
        try:
            bill_amount_default = float(existing_bill_amount) if existing_bill_amount is not None else 0.0
        except (TypeError, ValueError):
            bill_amount_default = 0.0
        with condition_cols[1]:
            bill_amount_edit = st.number_input(
                "Bill amount",
                value=float(bill_amount_default),
                min_value=0.0,
                step=100.0,
                format="%.2f",
                key=f"service_edit_bill_amount_{selected_service_id}",
            )
        condition_notes_edit = st.text_area(
            "Condition remarks",
            value=clean_text(selected_record.get("condition_remarks")) or "",
            key=f"service_edit_condition_notes_{selected_service_id}",
        )
        existing_bill_path = clean_text(selected_record.get("bill_document_path"))
        resolved_bill_path = resolve_upload_path(existing_bill_path)
        bill_col1, bill_col2 = st.columns([1, 1])
        with bill_col1:
            bill_file_edit = st.file_uploader(
                "Replace bill / invoice (PDF)",
                type=["pdf"],
                key=f"service_edit_bill_upload_{selected_service_id}",
            )
        with bill_col2:
            clear_bill = st.checkbox(
                "Remove existing bill",
                value=False,
                key=f"service_edit_bill_clear_{selected_service_id}",
            )
        if resolved_bill_path and resolved_bill_path.exists():
            st.download_button(
                "Download current bill",
                data=resolved_bill_path.read_bytes(),
                file_name=resolved_bill_path.name,
                key=f"service_bill_download_{selected_service_id}",
            )
        elif existing_bill_path:
            st.caption("Bill file not found. Upload a fresh copy to replace it.")
        if st.button("Save updates", key="save_service_updates"):
            (
                service_date_str,
                service_start_str,
                service_end_str,
            ) = determine_period_strings(edit_status_choice, edit_period_value)
            valid_update = True
            if edit_status_choice == "Completed" and (
                not service_start_str or not service_end_str
            ):
                st.error("Provide both start and end dates for completed services.")
                valid_update = False
            if edit_status_choice != "Completed" and not service_start_str:
                st.error("Select a start date for this service entry.")
                valid_update = False
            if valid_update:
                condition_update_value = (
                    condition_choice_edit
                    if condition_choice_edit in GENERATOR_CONDITION_OPTIONS
                    else None
                )
                condition_notes_update = clean_text(condition_notes_edit)
                bill_amount_update = None
                try:
                    if bill_amount_edit is not None and float(bill_amount_edit) > 0:
                        bill_amount_update = round(float(bill_amount_edit), 2)
                except Exception:
                    bill_amount_update = None
                current_bill_path = clean_text(selected_record.get("bill_document_path"))
                bill_path_value = current_bill_path
                replaced_bill = False
                removed_bill = False
                if bill_file_edit is not None:
                    stored_path = store_uploaded_pdf(
                        bill_file_edit,
                        SERVICE_BILL_DIR,
                        filename=f"service_{selected_service_id}_bill.pdf",
                    )
                    if stored_path:
                        bill_path_value = stored_path
                        replaced_bill = True
                        if current_bill_path:
                            old_path = resolve_upload_path(current_bill_path)
                            if old_path and old_path.exists():
                                new_path = resolve_upload_path(stored_path)
                                if not new_path or new_path != old_path:
                                    try:
                                        old_path.unlink()
                                    except Exception:
                                        pass
                elif clear_bill and current_bill_path:
                    old_path = resolve_upload_path(current_bill_path)
                    if old_path and old_path.exists():
                        try:
                            old_path.unlink()
                        except Exception:
                            pass
                    bill_path_value = None
                    removed_bill = True
                conn.execute(
                    """
                    UPDATE services
                    SET status = ?,
                        remarks = ?,
                        service_date = ?,
                        service_start_date = ?,
                        service_end_date = ?,
                        condition_status = ?,
                        condition_remarks = ?,
                        bill_amount = ?,
                        bill_document_path = ?,
                        updated_at = datetime('now')
                    WHERE service_id = ?
                    """,
                    (
                        new_status,
                        clean_text(new_remarks),
                        service_date_str,
                        service_start_str,
                        service_end_str,
                        condition_update_value,
                        condition_notes_update,
                        bill_amount_update,
                        bill_path_value,
                        int(selected_service_id),
                    ),
                )
                conn.commit()
                label_text = labels.get(int(selected_service_id), f"Service #{int(selected_service_id)}")
                status_label = clean_text(new_status) or DEFAULT_SERVICE_STATUS
                message_summary = label_text
                if status_label:
                    message_summary = f"{label_text} → {status_label}"
                log_activity(
                    conn,
                    event_type="service_updated",
                    description=message_summary,
                    entity_type="service",
                    entity_id=int(selected_service_id),
                )
                message_bits = ["Service record updated."]
                if bill_amount_update is not None:
                    message_bits.append(f"Bill amount {format_money(bill_amount_update)}")
                if replaced_bill:
                    message_bits.append("Invoice replaced")
                elif removed_bill:
                    message_bits.append("Invoice removed")
                st.success(". ".join(message_bits))
                _safe_rerun()

        attachments_df = df_query(
            conn,
            """
            SELECT document_id, file_path, original_name, uploaded_at
            FROM service_documents
            WHERE service_id = ?
            ORDER BY datetime(uploaded_at) DESC, document_id DESC
            """,
            (int(selected_service_id),),
        )
        st.markdown("**Attached documents**")
        if attachments_df.empty:
            st.caption("No documents attached yet.")
        else:
            for _, doc_row in attachments_df.iterrows():
                path = resolve_upload_path(doc_row.get("file_path"))
                display_name = clean_text(doc_row.get("original_name"))
                if path and path.exists():
                    label = display_name or path.name
                    st.download_button(
                        f"Download {label}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"service_doc_dl_{int(doc_row['document_id'])}",
                    )
                else:
                    label = display_name or "Document"
                    st.caption(f"⚠️ Missing file: {label}")

        with st.form(f"service_doc_upload_{selected_service_id}"):
            more_docs = st.file_uploader(
                "Add more service documents (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"service_doc_files_{selected_service_id}",
            )
            upload_docs = st.form_submit_button("Upload documents")
        if upload_docs:
            if more_docs:
                saved = attach_documents(
                    conn,
                    "service_documents",
                    "service_id",
                    int(selected_service_id),
                    more_docs,
                    SERVICE_DOCS_DIR,
                    f"service_{selected_service_id}",
                )
                conn.commit()
                st.success(f"Uploaded {saved} document(s).")
                _safe_rerun()
            else:
                st.info("Select at least one PDF to upload.")
    else:
        st.info("No service records yet. Log one using the form above.")


def _build_quotation_workbook(
    *,
    metadata: dict[str, Optional[str]],
    items: list[dict[str, object]],
    totals: list[tuple[str, float]],
) -> bytes:
    buffer = io.BytesIO()
    summary_rows = [(key, metadata.get(key) or "-") for key in metadata]
    summary_df = pd.DataFrame(summary_rows, columns=["Field", "Value"])
    items_df = pd.DataFrame(items)
    totals_df = pd.DataFrame(totals, columns=["Label", "Amount"])

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Quotation", index=False)
        start_row = len(summary_df) + 2
        if not items_df.empty:
            items_df.to_excel(writer, sheet_name="Quotation", index=False, startrow=start_row)
            totals_start = start_row + len(items_df) + 2
        else:
            totals_start = start_row
        if not totals_df.empty:
            totals_df.to_excel(writer, sheet_name="Quotation", index=False, startrow=totals_start)

    buffer.seek(0)
    return buffer.read()


def _render_quotation_section():
    st.subheader("🧾 Create quotation")

    default_date = datetime.now().date()
    result_key = "quotation_result"
    feedback = st.session_state.pop("quotation_feedback", None)
    if feedback:
        level, message = feedback
        if level == "success":
            st.success(message)
        elif level == "info":
            st.info(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.write(message)

    st.session_state.setdefault("quotation_item_rows", _default_quotation_items())

    with st.form("quotation_form"):
        col_left, col_right = st.columns(2)
        with col_left:
            reference_value = st.text_input(
                "Quotation reference", key="quotation_reference"
            )
            quotation_date = st.date_input(
                "Quotation date",
                value=default_date,
                key="quotation_date",
            )
            prepared_by = st.text_input("Prepared by", key="quotation_prepared_by")
        with col_right:
            valid_days = st.number_input(
                "Valid for (days)",
                min_value=0,
                value=DEFAULT_QUOTATION_VALID_DAYS,
                step=1,
                format="%d",
                key="quotation_valid_days",
            )
            round_total = st.checkbox(
                "Round grand total to nearest rupee",
                value=False,
                help="If enabled the final amount will be rounded to the nearest rupee.",
                key="quotation_round_total",
            )

        company_name = st.text_input("Company / branch", key="quotation_company_name")
        company_details = st.text_area(
            "Company details / address",
            help="Include GST number, phone number or any header text that should appear with the quotation.",
            key="quotation_company_details",
        )

        customer_cols = st.columns(2)
        with customer_cols[0]:
            customer_name = st.text_input(
                "Customer / organisation", key="quotation_customer_name"
            )
        with customer_cols[1]:
            customer_contact = st.text_input(
                "Customer contact (phone / email)",
                key="quotation_customer_contact",
            )
        customer_address = st.text_area(
            "Customer address / details", key="quotation_customer_address"
        )

        project_cols = st.columns(2)
        with project_cols[0]:
            project_name = st.text_input("Project / site", key="quotation_project_name")
        with project_cols[1]:
            subject_line = st.text_input(
                "Subject / scope title", key="quotation_subject"
            )

        notes_value = st.text_area(
            "Scope / notes",
            help="Outline the deliverables or inclusions for this quotation.",
            key="quotation_scope_notes",
        )
        terms_value = st.text_area(
            "Terms & conditions",
            help="Payment terms, validity clauses or other conditions that should be shared with the client.",
            key="quotation_terms",
        )

        defaults_cols = st.columns(4)
        with defaults_cols[0]:
            default_discount = st.number_input(
                "Default discount (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                format="%.2f",
                key="quotation_default_discount",
            )
        with defaults_cols[1]:
            default_cgst = st.number_input(
                "Default CGST (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                format="%.2f",
                key="quotation_default_cgst",
            )
        with defaults_cols[2]:
            default_sgst = st.number_input(
                "Default SGST (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                format="%.2f",
                key="quotation_default_sgst",
            )
        with defaults_cols[3]:
            default_igst = st.number_input(
                "Default IGST (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                format="%.2f",
                key="quotation_default_igst",
            )

        st.markdown("#### Line items")
        st.caption(
            "Add each item or service below. Use the built-in controls to add, duplicate or reorder rows."
        )

        items_state = st.session_state.get(
            "quotation_item_rows", _default_quotation_items()
        )
        items_df = pd.DataFrame(items_state)
        required_item_columns = [
            "description",
            "hsn",
            "unit",
            "quantity",
            "rate",
            "discount",
            "cgst",
            "sgst",
            "igst",
        ]
        for column in required_item_columns:
            if column not in items_df.columns:
                if column in {"description", "hsn", "unit"}:
                    items_df[column] = ""
                elif column == "quantity":
                    items_df[column] = 1.0
                else:
                    items_df[column] = 0.0
        items_df = items_df[required_item_columns]
        edited_items = st.data_editor(
            items_df,
            key="quotation_items_table",
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "description": st.column_config.TextColumn(
                    "Description",
                    help="Item or service being quoted.",
                ),
                "hsn": st.column_config.TextColumn(
                    "HSN / SAC",
                    help="Optional tax code (HSN/SAC).",
                ),
                "unit": st.column_config.TextColumn(
                    "Unit",
                    help="Unit of measure (Nos, Lot, etc.).",
                ),
                "quantity": st.column_config.NumberColumn(
                    "Quantity",
                    min_value=0.0,
                    step=0.5,
                    format="%.2f",
                ),
                "rate": st.column_config.NumberColumn(
                    "Rate",
                    min_value=0.0,
                    step=10.0,
                    format="%.2f",
                ),
                "discount": st.column_config.NumberColumn(
                    "Discount (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    format="%.2f",
                ),
                "cgst": st.column_config.NumberColumn(
                    "CGST (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    format="%.2f",
                ),
                "sgst": st.column_config.NumberColumn(
                    "SGST (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    format="%.2f",
                ),
                "igst": st.column_config.NumberColumn(
                    "IGST (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    format="%.2f",
                ),
            },
        )
        item_records = edited_items.to_dict("records")
        st.session_state["quotation_item_rows"] = item_records
        st.session_state["quotation_items_table"] = edited_items

        action_cols = st.columns((1, 1))
        submit = action_cols[0].form_submit_button("Create quotation", type="primary")
        reset_form = action_cols[1].form_submit_button("Reset form")
        if reset_form:
            _reset_quotation_form_state()
            st.session_state["quotation_feedback"] = (
                "info",
                "Quotation form cleared. Start with fresh details.",
            )
            _safe_rerun()
            return

    if submit:
        prepared_items: list[dict[str, object]] = []
        for record in item_records:
            prepared_items.append(
                {
                    "description": record.get("description"),
                    "hsn": record.get("hsn"),
                    "unit": record.get("unit"),
                    "quantity": _value_or_default(record.get("quantity"), 1.0),
                    "rate": _value_or_default(record.get("rate"), 0.0),
                    "discount": _value_or_default(
                        record.get("discount"), default_discount
                    ),
                    "cgst": _value_or_default(record.get("cgst"), default_cgst),
                    "sgst": _value_or_default(record.get("sgst"), default_sgst),
                    "igst": _value_or_default(record.get("igst"), default_igst),
                }
            )

        items_clean, totals_data = normalize_quotation_items(prepared_items)

        if not items_clean:
            st.error("Add at least one item with a description to create a quotation.")
            st.session_state.pop(result_key, None)
            return

        valid_days_value = int(valid_days)
        valid_until = None
        if valid_days_value > 0:
            valid_until = quotation_date + timedelta(days=valid_days_value)

        metadata = OrderedDict(
            {
                "Quotation reference": clean_text(reference_value) or "-",
                "Quotation date": quotation_date.strftime(DATE_FMT),
                "Valid until": valid_until.strftime(DATE_FMT) if valid_until else "-",
                "Prepared by": clean_text(prepared_by) or "-",
                "Company / branch": clean_text(company_name) or "-",
                "Company details": clean_text(company_details) or "-",
                "Customer / organisation": clean_text(customer_name) or "-",
                "Customer contact": clean_text(customer_contact) or "-",
                "Customer address": clean_text(customer_address) or "-",
                "Project / site": clean_text(project_name) or "-",
                "Subject / scope": clean_text(subject_line) or "-",
            }
        )
        notes_clean = clean_text(notes_value)
        if notes_clean:
            metadata["Scope / notes"] = notes_clean
        terms_clean = clean_text(terms_value)
        if terms_clean:
            metadata["Terms & conditions"] = terms_clean

        grand_total_before_round = totals_data["grand_total"]
        rounded_grand_total = grand_total_before_round
        round_off_value = 0.0
        if round_total:
            rounded_grand_total = float(round(grand_total_before_round))
            round_off_value = rounded_grand_total - grand_total_before_round

        totals_rows: list[tuple[str, float]] = []
        totals_rows.append(("Gross amount", totals_data["gross_total"]))
        if totals_data["discount_total"]:
            totals_rows.append(("Discount total", totals_data["discount_total"]))
        totals_rows.append(("Taxable value", totals_data["taxable_total"]))
        if totals_data["cgst_total"]:
            totals_rows.append(("CGST total", totals_data["cgst_total"]))
        if totals_data["sgst_total"]:
            totals_rows.append(("SGST total", totals_data["sgst_total"]))
        if totals_data["igst_total"]:
            totals_rows.append(("IGST total", totals_data["igst_total"]))
        if round_total:
            totals_rows.append(("Grand total (before round)", grand_total_before_round))
        if round_off_value:
            totals_rows.append(("Round off", round_off_value))
        totals_rows.append(("Grand total", rounded_grand_total))

        workbook_items = [item.copy() for item in items_clean]

        workbook_bytes = _build_quotation_workbook(
            metadata=metadata,
            items=workbook_items,
            totals=totals_rows,
        )

        display_df = pd.DataFrame(workbook_items)

        def _format_quantity_display(value: object) -> str:
            amount = _coerce_float(value, 0.0)
            if math.isclose(amount, round(amount)):
                return f"{int(round(amount))}"
            return f"{amount:,.2f}"

        money_columns = [
            "Rate",
            "Gross amount",
            "Discount amount",
            "Taxable value",
            "CGST amount",
            "SGST amount",
            "IGST amount",
            "Line total",
        ]
        for col in money_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: format_money(value) or f"{_coerce_float(value, 0.0):,.2f}"
                )
        if "Quantity" in display_df.columns:
            display_df["Quantity"] = display_df["Quantity"].apply(_format_quantity_display)
        percentage_columns = ["Discount (%)", "CGST (%)", "SGST (%)", "IGST (%)"]
        for col in percentage_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda value: f"{_coerce_float(value, 0.0):.2f}%")
        display_df = display_df.fillna("")

        base_filename = clean_text(reference_value) or f"quotation_{quotation_date.strftime('%Y%m%d')}"
        safe_name = _sanitize_path_component(base_filename)
        if not safe_name:
            safe_name = f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = f"{safe_name}.xlsx"

        st.session_state[result_key] = {
            "display": display_df,
            "metadata_items": list(metadata.items()),
            "totals_rows": totals_rows,
            "gross_total": totals_data["gross_total"],
            "taxable_total": totals_data["taxable_total"],
            "grand_total": rounded_grand_total,
            "grand_total_before_round": grand_total_before_round,
            "round_off": round_off_value,
            "metadata": metadata,
            "excel_bytes": workbook_bytes,
            "filename": filename,
        }

    result = st.session_state.get(result_key)
    if result:
        st.success("Quotation ready. Review the details below or download the Excel file.")
        metadata_df = pd.DataFrame(result["metadata_items"], columns=["Field", "Value"])
        st.table(metadata_df)

        st.dataframe(result["display"], use_container_width=True)

        totals_rows = result.get("totals_rows", [])
        if totals_rows:
            totals_df = pd.DataFrame(totals_rows, columns=["Label", "Amount"])
            totals_df["Amount"] = totals_df["Amount"].apply(
                lambda value: format_money(value) or f"{_coerce_float(value, 0.0):,.2f}"
            )
            st.table(totals_df)

        grand_total_label = format_money(result["grand_total"]) or f"{result['grand_total']:,.2f}"
        st.markdown(f"**Grand total:** {grand_total_label}")
        round_off_value = result.get("round_off", 0.0)
        if round_off_value:
            round_off_label = format_money(round_off_value) or f"{round_off_value:,.2f}"
            original_total = result.get("grand_total_before_round", result["grand_total"])
            original_label = format_money(original_total) or f"{original_total:,.2f}"
            st.caption(
                f"Rounded from {original_label} (round off {round_off_label})."
            )

        st.download_button(
            "Download quotation",
            data=result["excel_bytes"],
            file_name=result["filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="quotation_download",
        )


def _render_maintenance_section(conn, *, show_heading: bool = True):
    if show_heading:
        st.subheader("🔧 Maintenance Records")
    _, customer_label_map = build_customer_groups(conn, only_complete=False)
    customer_options, customer_labels, _, label_by_id = fetch_customer_choices(conn)
    do_df = df_query(
        conn,
        """
        SELECT d.do_number, d.customer_id, COALESCE(c.name, '(unknown)') AS customer_name, d.description, d.remarks
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        ORDER BY datetime(d.created_at) DESC
        """,
    )
    allowed_customers = accessible_customer_ids(conn)
    if allowed_customers is not None:
        do_df = do_df[
            do_df["customer_id"].apply(
                lambda value: int(value) in allowed_customers if pd.notna(value) else False
            )
        ]
    do_options = [None]
    do_labels = {None: "No delivery order (manual entry)"}
    do_customer_map = {}
    do_customer_name_map = {}
    for _, row in do_df.iterrows():
        do_num = clean_text(row.get("do_number"))
        if not do_num:
            continue
        cust_id = int(row["customer_id"]) if not pd.isna(row.get("customer_id")) else None
        summary = clean_text(row.get("description"))
        cust_name = customer_label_map.get(cust_id) if cust_id else clean_text(row.get("customer_name"))
        label_parts = [do_num]
        if cust_name:
            label_parts.append(f"({cust_name})")
        if summary:
            snippet = summary[:40]
            if len(summary) > 40:
                snippet += "…"
            label_parts.append(f"– {snippet}")
        label = " ".join(part for part in label_parts if part)
        do_options.append(do_num)
        do_labels[do_num] = label
        do_customer_map[do_num] = cust_id
        do_customer_name_map[do_num] = cust_name or "(not linked)"

    with st.form("maintenance_form"):
        selected_do = st.selectbox(
            "Delivery order",
            options=do_options,
            format_func=lambda do: do_labels.get(do, str(do)),
        )
        default_customer = do_customer_map.get(selected_do)
        state_key = "maintenance_customer_link"
        last_do_key = "maintenance_customer_last_do"
        linked_customer = default_customer
        if default_customer is not None:
            st.session_state[last_do_key] = selected_do
            st.session_state[state_key] = default_customer
            customer_label = (
                customer_labels.get(default_customer)
                or customer_label_map.get(default_customer)
                or label_by_id.get(default_customer)
                or do_customer_name_map.get(selected_do)
                or f"Customer #{default_customer}"
            )
            st.text_input("Customer", value=customer_label, disabled=True)
        else:
            choices = list(customer_options)
            if st.session_state.get(last_do_key) != selected_do:
                st.session_state[last_do_key] = selected_do
                st.session_state[state_key] = None
            linked_customer = st.selectbox(
                "Customer *",
                options=choices,
                format_func=lambda cid: customer_labels.get(cid, "-- Select customer --"),
                key=state_key,
            )
        status_value = status_input_widget("maintenance_new", DEFAULT_SERVICE_STATUS)
        maintenance_status_choice = get_status_choice("maintenance_new")
        today = datetime.now().date()
        if maintenance_status_choice == "Completed":
            maintenance_period_value = st.date_input(
                "Maintenance period",
                value=(today, today),
                help="Select the start and end dates for the maintenance work.",
                key="maintenance_new_period_completed",
            )
        elif maintenance_status_choice == "In progress":
            maintenance_period_value = st.date_input(
                "Maintenance start date",
                value=today,
                help="Choose when this maintenance began.",
                key="maintenance_new_period_start",
            )
        else:
            maintenance_period_value = st.date_input(
                "Planned start date",
                value=today,
                help="Select when this maintenance is scheduled to begin.",
                key="maintenance_new_period_planned",
            )
        description = st.text_area("Maintenance description")
        remarks = st.text_area("Remarks / updates")
        maintenance_product_count = st.number_input(
            "Products sold during maintenance",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="maintenance_additional_product_count",
            help="Track any new items purchased while maintenance was carried out.",
        )
        maintenance_product_entries: list[dict[str, object]] = []
        for idx in range(int(maintenance_product_count)):
            cols = st.columns((2, 2, 2, 1))
            with cols[0]:
                product_name = st.text_input(
                    f"Product {idx + 1} details",
                    key=f"maintenance_product_name_{idx}",
                )
            with cols[1]:
                product_model = st.text_input(
                    f"Model {idx + 1}",
                    key=f"maintenance_product_model_{idx}",
                )
            with cols[2]:
                product_serial = st.text_input(
                    f"Serial {idx + 1}",
                    key=f"maintenance_product_serial_{idx}",
                )
            with cols[3]:
                product_quantity = st.number_input(
                    f"Qty {idx + 1}",
                    min_value=1,
                    max_value=999,
                    value=1,
                    step=1,
                    key=f"maintenance_product_quantity_{idx}",
                )
            maintenance_product_entries.append(
                {
                    "name": product_name,
                    "model": product_model,
                    "serial": product_serial,
                    "quantity": int(product_quantity),
                }
            )
        maintenance_files = st.file_uploader(
            "Attach maintenance documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="maintenance_new_docs",
        )
        submit = st.form_submit_button("Log maintenance", type="primary")

    if submit:
        selected_customer = (
            linked_customer if linked_customer is not None else do_customer_map.get(selected_do)
        )
        selected_customer = int(selected_customer) if selected_customer is not None else None
        cur = conn.cursor()
        (
            maintenance_date_str,
            maintenance_start_str,
            maintenance_end_str,
        ) = determine_period_strings(
            maintenance_status_choice, maintenance_period_value
        )
        valid_entry = True
        if selected_customer is None:
            st.error("Select a customer to log this maintenance entry.")
            valid_entry = False
        if maintenance_status_choice == "Completed" and (
            not maintenance_start_str or not maintenance_end_str
        ):
            st.error("Start and end dates are required for completed maintenance work.")
            valid_entry = False
        if maintenance_status_choice != "Completed" and not maintenance_start_str:
            st.error("Select a start date for this maintenance entry.")
            valid_entry = False
        if valid_entry:
            _cleaned_maintenance_products, maintenance_product_labels = normalize_product_entries(
                maintenance_product_entries
            )
            maintenance_product_label = (
                "\n".join(maintenance_product_labels)
                if maintenance_product_labels
                else None
            )
            cur.execute(
                """
                INSERT INTO maintenance_records (
                    do_number,
                    customer_id,
                    maintenance_date,
                    maintenance_start_date,
                    maintenance_end_date,
                    description,
                    status,
                    remarks,
                    maintenance_product_info,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selected_do,
                    selected_customer,
                    maintenance_date_str,
                    maintenance_start_str,
                    maintenance_end_str,
                    clean_text(description),
                    status_value,
                    clean_text(remarks),
                    maintenance_product_label,
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            maintenance_id = cur.lastrowid
            if selected_do and selected_customer is not None:
                link_delivery_order_to_customer(conn, selected_do, selected_customer)
            saved_docs = attach_documents(
                conn,
                "maintenance_documents",
                "maintenance_id",
                maintenance_id,
                maintenance_files,
                MAINTENANCE_DOCS_DIR,
                f"maintenance_{maintenance_id}",
            )
            conn.commit()
            maintenance_label = do_labels.get(selected_do) if selected_do else None
            if not maintenance_label:
                maintenance_label = f"Maintenance #{maintenance_id}"
                customer_name = None
                if selected_customer is not None:
                    customer_name = (
                        label_by_id.get(int(selected_customer))
                        or customer_label_map.get(int(selected_customer))
                    )
                summary_parts = [maintenance_label]
                if customer_name:
                    summary_parts.append(customer_name)
                status_label = clean_text(status_value) or DEFAULT_SERVICE_STATUS
                summary_parts.append(f"status {status_label}")
                log_activity(
                    conn,
                    event_type="maintenance_created",
                    description=" – ".join(summary_parts),
                    entity_type="maintenance",
                    entity_id=int(maintenance_id),
                )
                message = "Maintenance record saved."
                if saved_docs:
                    message = f"{message} Attached {saved_docs} document(s)."
                st.success(message)
                _safe_rerun()

    maintenance_df = df_query(
        conn,
        """
        SELECT m.maintenance_id,
               m.customer_id,
               d.customer_id AS do_customer_id,
               m.do_number,
               m.maintenance_date,
               m.maintenance_start_date,
               m.maintenance_end_date,
               m.maintenance_product_info,
               m.description,
               m.status,
               m.remarks,
               m.updated_at,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(md.document_id) AS doc_count
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = m.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN maintenance_documents md ON md.maintenance_id = m.maintenance_id
        GROUP BY m.maintenance_id
        ORDER BY datetime(COALESCE(m.maintenance_start_date, m.maintenance_date)) DESC, m.maintenance_id DESC
        """,
    )
    if allowed_customers is not None:
        def _maintenance_row_allowed(row):
            maint_cust = row.get("customer_id")
            do_cust = row.get("do_customer_id")
            candidates = []
            if pd.notna(maint_cust):
                candidates.append(int(maint_cust))
            if pd.notna(do_cust):
                candidates.append(int(do_cust))
            return any(cid in allowed_customers for cid in candidates)

        maintenance_df = maintenance_df[maintenance_df.apply(_maintenance_row_allowed, axis=1)]
    if not maintenance_df.empty:
        maintenance_df = fmt_dates(
            maintenance_df,
            ["maintenance_date", "maintenance_start_date", "maintenance_end_date"],
        )
        maintenance_df.drop(columns=["customer_id", "do_customer_id"], inplace=True, errors="ignore")
        maintenance_df["maintenance_period"] = maintenance_df.apply(
            lambda row: format_period_span(
                row.get("maintenance_start_date"), row.get("maintenance_end_date")
            ),
            axis=1,
        )
        maintenance_df["Last update"] = pd.to_datetime(maintenance_df.get("updated_at"), errors="coerce").dt.strftime("%d-%m-%Y %H:%M")
        maintenance_df.loc[maintenance_df["Last update"].isna(), "Last update"] = None
        if "status" in maintenance_df.columns:
            maintenance_df["status"] = maintenance_df["status"].apply(lambda x: clean_text(x) or DEFAULT_SERVICE_STATUS)
        display = maintenance_df.rename(
            columns={
                "do_number": "DO Serial",
                "maintenance_date": "Maintenance date",
                "maintenance_start_date": "Maintenance start date",
                "maintenance_end_date": "Maintenance end date",
                "maintenance_period": "Maintenance period",
                "maintenance_product_info": "Products sold",
                "description": "Description",
                "status": "Status",
                "remarks": "Remarks",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        st.markdown("### Maintenance history")
        st.dataframe(
            display.drop(columns=["updated_at", "maintenance_id"], errors="ignore"),
            use_container_width=True,
        )

        records = maintenance_df.to_dict("records")
        st.markdown("#### Update status & remarks")
        options = [int(r["maintenance_id"]) for r in records]
        def maintenance_label(record):
            do_ref = clean_text(record.get("do_number")) or "(no DO)"
            date_ref = clean_text(record.get("maintenance_period")) or clean_text(
                record.get("maintenance_date")
            )
            customer_ref = clean_text(record.get("customer"))
            parts = [do_ref]
            if date_ref:
                parts.append(f"· {date_ref}")
            if customer_ref:
                parts.append(f"· {customer_ref}")
            return " ".join(parts)

        labels = {int(r["maintenance_id"]): maintenance_label(r) for r in records}
        selected_maintenance_id = st.selectbox(
            "Select maintenance entry",
            options,
            format_func=lambda rid: labels.get(rid, str(rid)),
        )
        selected_record = next(r for r in records if int(r["maintenance_id"]) == int(selected_maintenance_id))
        new_status = status_input_widget(
            f"maintenance_edit_{selected_maintenance_id}",
            selected_record.get("status"),
        )
        maintenance_edit_choice = get_status_choice(
            f"maintenance_edit_{selected_maintenance_id}"
        )
        existing_start = ensure_date(selected_record.get("maintenance_start_date")) or ensure_date(
            selected_record.get("maintenance_date")
        )
        existing_end = ensure_date(selected_record.get("maintenance_end_date")) or existing_start
        today = datetime.now().date()
        default_start = existing_start or today
        default_end = existing_end or default_start
        if maintenance_edit_choice == "Completed":
            maintenance_period_edit = st.date_input(
                "Maintenance period",
                value=(default_start, default_end),
                key=f"maintenance_edit_{selected_maintenance_id}_period_completed",
                help="Update the start and end dates for this maintenance record.",
            )
        elif maintenance_edit_choice == "In progress":
            maintenance_period_edit = st.date_input(
                "Maintenance start date",
                value=default_start,
                key=f"maintenance_edit_{selected_maintenance_id}_period_start",
                help="Adjust when this maintenance began.",
            )
        else:
            maintenance_period_edit = st.date_input(
                "Planned start date",
                value=default_start,
                key=f"maintenance_edit_{selected_maintenance_id}_period_planned",
                help="Adjust when this maintenance is scheduled to begin.",
            )
        new_remarks = st.text_area(
            "Remarks",
            value=clean_text(selected_record.get("remarks")) or "",
            key=f"maintenance_edit_{selected_maintenance_id}",
        )
        if st.button("Save maintenance updates", key="save_maintenance_updates"):
            (
                maintenance_date_str,
                maintenance_start_str,
                maintenance_end_str,
            ) = determine_period_strings(
                maintenance_edit_choice, maintenance_period_edit
            )
            valid_update = True
            if maintenance_edit_choice == "Completed" and (
                not maintenance_start_str or not maintenance_end_str
            ):
                st.error(
                    "Provide both start and end dates for completed maintenance records."
                )
                valid_update = False
            if maintenance_edit_choice != "Completed" and not maintenance_start_str:
                st.error("Select a start date for this maintenance entry.")
                valid_update = False
            if valid_update:
                conn.execute(
                    """
                    UPDATE maintenance_records
                    SET status = ?,
                        remarks = ?,
                        maintenance_date = ?,
                        maintenance_start_date = ?,
                        maintenance_end_date = ?,
                        updated_at = datetime('now')
                    WHERE maintenance_id = ?
                    """,
                    (
                        new_status,
                        clean_text(new_remarks),
                        maintenance_date_str,
                        maintenance_start_str,
                        maintenance_end_str,
                        int(selected_maintenance_id),
                    ),
                )
                conn.commit()
                label_text = labels.get(
                    int(selected_maintenance_id),
                    f"Maintenance #{int(selected_maintenance_id)}",
                )
                status_label = clean_text(new_status) or DEFAULT_SERVICE_STATUS
                summary = f"{label_text} → {status_label}" if status_label else label_text
                log_activity(
                    conn,
                    event_type="maintenance_updated",
                    description=summary,
                    entity_type="maintenance",
                    entity_id=int(selected_maintenance_id),
                )
                st.success("Maintenance record updated.")
                _safe_rerun()

        attachments_df = df_query(
            conn,
            """
            SELECT document_id, file_path, original_name, uploaded_at
            FROM maintenance_documents
            WHERE maintenance_id = ?
            ORDER BY datetime(uploaded_at) DESC, document_id DESC
            """,
            (int(selected_maintenance_id),),
        )
        st.markdown("**Attached documents**")
        if attachments_df.empty:
            st.caption("No documents attached yet.")
        else:
            for _, doc_row in attachments_df.iterrows():
                path = resolve_upload_path(doc_row.get("file_path"))
                display_name = clean_text(doc_row.get("original_name"))
                if path and path.exists():
                    label = display_name or path.name
                    st.download_button(
                        f"Download {label}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"maintenance_doc_dl_{int(doc_row['document_id'])}",
                    )
                else:
                    label = display_name or "Document"
                    st.caption(f"⚠️ Missing file: {label}")

        with st.form(f"maintenance_doc_upload_{selected_maintenance_id}"):
            more_docs = st.file_uploader(
                "Add more maintenance documents (PDF)",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"maintenance_doc_files_{selected_maintenance_id}",
            )
            upload_docs = st.form_submit_button("Upload documents")
        if upload_docs:
            if more_docs:
                saved = attach_documents(
                    conn,
                    "maintenance_documents",
                    "maintenance_id",
                    int(selected_maintenance_id),
                    more_docs,
                    MAINTENANCE_DOCS_DIR,
                    f"maintenance_{selected_maintenance_id}",
                )
                conn.commit()
                st.success(f"Uploaded {saved} document(s).")
                _safe_rerun()
            else:
                st.info("Select at least one PDF to upload.")
    else:
        st.info("No maintenance records yet. Log one using the form above.")


def service_maintenance_page(conn):
    st.subheader("🛠️ Maintenance and Service")
    st.markdown("### Service records")
    _render_service_section(conn, show_heading=False)
    st.markdown("---")
    st.info("Quotation creation is temporarily unavailable.")


def customer_summary_page(conn):
    st.subheader("📒 Customer Summary")
    blank_label = "(blank)"
    complete_clause = customer_complete_clause()
    scope_clause, scope_params = customer_scope_filter()
    where_parts = [complete_clause]
    params: list[object] = []
    if scope_clause:
        where_parts.append(scope_clause)
        params.extend(scope_params)
    where_sql = " AND ".join(where_parts)
    customers = df_query(
        conn,
        f"""
        SELECT TRIM(name) AS name, GROUP_CONCAT(customer_id) AS ids, COUNT(*) AS cnt
        FROM customers
        WHERE {where_sql}
        GROUP BY TRIM(name)
        ORDER BY TRIM(name) ASC
        """,
        tuple(params),
    )
    if customers.empty:
        st.info(
            "No complete customers available for your account. Check the Scraps page for records that need details."
        )
        return

    names = customers["name"].tolist()
    name_map = {
        row["name"]: f"{row['name']} ({int(row['cnt'])} records)" if int(row["cnt"]) > 1 else row["name"]
        for _, row in customers.iterrows()
    }
    sel_name = st.selectbox("Select customer", names, format_func=lambda n: name_map.get(n, n))
    row = customers[customers["name"] == sel_name].iloc[0]
    ids = [int(i) for i in str(row["ids"]).split(",") if i]
    cnt = int(row["cnt"])

    placeholder_block = ','.join('?' * len(ids))
    info = df_query(
        conn,
        f"""
        SELECT
            MAX(name) AS name,
            GROUP_CONCAT(DISTINCT phone) AS phone,
            GROUP_CONCAT(DISTINCT address) AS address,
            GROUP_CONCAT(DISTINCT purchase_date) AS purchase_dates,
            GROUP_CONCAT(DISTINCT product_info) AS products,
            GROUP_CONCAT(DISTINCT delivery_order_code) AS do_codes
        FROM customers
        WHERE customer_id IN ({placeholder_block})
        """,
        ids,
    ).iloc[0].to_dict()

    st.write("**Name:**", info.get("name") or blank_label)
    st.write("**Phone:**", info.get("phone"))
    st.write("**Address:**", info.get("address"))
    st.write("**Purchase:**", info.get("purchase_dates"))
    st.write("**Product:**", info.get("products"))
    st.write("**Delivery order:**", info.get("do_codes"))
    if cnt > 1:
        st.caption(f"Merged from {cnt} duplicates")

    st.markdown("---")
    placeholders = ",".join("?" * len(ids))

    warr = df_query(
        conn,
        f"""
        SELECT w.warranty_id as id, c.name as customer, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date, w.status, w.remarks, w.dup_flag
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        WHERE w.customer_id IN ({placeholders})
        ORDER BY date(w.expiry_date) DESC
        """,
        ids,
    )
    warr = fmt_dates(warr, ["issue_date", "expiry_date"])
    if "dup_flag" in warr.columns:
        warr = warr.assign(duplicate=warr["dup_flag"].apply(lambda x: "🔁 duplicate" if int(x) == 1 else ""))
    warr_display = format_warranty_table(warr)

    service_df = df_query(
        conn,
        f"""
        SELECT s.service_id,
               s.do_number,
               s.service_date,
               s.service_start_date,
               s.service_end_date,
               s.service_product_info,
               s.description,
               s.remarks,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(sd.document_id) AS doc_count
        FROM services s
        LEFT JOIN customers c ON c.customer_id = s.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = s.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN service_documents sd ON sd.service_id = s.service_id
        WHERE COALESCE(s.customer_id, d.customer_id) IN ({placeholders})
        GROUP BY s.service_id
        ORDER BY datetime(COALESCE(s.service_start_date, s.service_date)) DESC, s.service_id DESC
        """,
        ids,
    )
    service_df = fmt_dates(service_df, ["service_date", "service_start_date", "service_end_date"])
    if not service_df.empty:
        service_df["service_period"] = service_df.apply(
            lambda row: format_period_span(
                row.get("service_start_date"), row.get("service_end_date")
            ),
            axis=1,
        )

    maintenance_df = df_query(
        conn,
        f"""
        SELECT m.maintenance_id,
               m.do_number,
               m.maintenance_date,
               m.maintenance_start_date,
               m.maintenance_end_date,
               m.maintenance_product_info,
               m.description,
               m.remarks,
               COALESCE(c.name, cdo.name, '(unknown)') AS customer,
               COUNT(md.document_id) AS doc_count
        FROM maintenance_records m
        LEFT JOIN customers c ON c.customer_id = m.customer_id
        LEFT JOIN delivery_orders d ON d.do_number = m.do_number
        LEFT JOIN customers cdo ON cdo.customer_id = d.customer_id
        LEFT JOIN maintenance_documents md ON md.maintenance_id = m.maintenance_id
        WHERE COALESCE(m.customer_id, d.customer_id) IN ({placeholders})
        GROUP BY m.maintenance_id
        ORDER BY datetime(COALESCE(m.maintenance_start_date, m.maintenance_date)) DESC, m.maintenance_id DESC
        """,
        ids,
    )
    maintenance_df = fmt_dates(
        maintenance_df,
        ["maintenance_date", "maintenance_start_date", "maintenance_end_date"],
    )
    if not maintenance_df.empty:
        maintenance_df["maintenance_period"] = maintenance_df.apply(
            lambda row: format_period_span(
                row.get("maintenance_start_date"), row.get("maintenance_end_date")
            ),
            axis=1,
        )

    do_df = df_query(
        conn,
        f"""
        SELECT d.do_number,
               COALESCE(c.name, '(unknown)') AS customer,
               d.description,
               d.sales_person,
               d.remarks,
               d.created_at,
               d.file_path
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        WHERE d.customer_id IN ({placeholders})
        ORDER BY datetime(d.created_at) DESC
        """,
        ids,
    )
    if not do_df.empty:
        do_df = fmt_dates(do_df, ["created_at"])
        do_df["do_number"] = do_df["do_number"].apply(clean_text)
        do_df["Document"] = do_df["file_path"].apply(lambda fp: "📎" if clean_text(fp) else "")

    do_numbers = set()
    if not do_df.empty and "do_number" in do_df.columns:
        do_numbers.update(val for val in do_df["do_number"].tolist() if val)
    if not service_df.empty and "do_number" in service_df.columns:
        do_numbers.update(clean_text(val) for val in service_df["do_number"].tolist() if clean_text(val))
    if not maintenance_df.empty and "do_number" in maintenance_df.columns:
        do_numbers.update(clean_text(val) for val in maintenance_df["do_number"].tolist() if clean_text(val))
    do_numbers = {val for val in do_numbers if val}

    present_dos = set()
    if not do_df.empty and "do_number" in do_df.columns:
        present_dos.update(val for val in do_df["do_number"].tolist() if val)
    missing_dos = sorted(do for do in do_numbers if do not in present_dos)
    if missing_dos:
        extra_df = df_query(
            conn,
            f"""
            SELECT d.do_number,
                   COALESCE(c.name, '(unknown)') AS customer,
                   d.description,
                   d.sales_person,
                   d.remarks,
                   d.created_at,
                   d.file_path
            FROM delivery_orders d
            LEFT JOIN customers c ON c.customer_id = d.customer_id
            WHERE d.do_number IN ({','.join('?' * len(missing_dos))})
            """,
            missing_dos,
        )
        if not extra_df.empty:
            extra_df = fmt_dates(extra_df, ["created_at"])
            extra_df["do_number"] = extra_df["do_number"].apply(clean_text)
            extra_df["Document"] = extra_df["file_path"].apply(lambda fp: "📎" if clean_text(fp) else "")
            do_df = pd.concat([do_df, extra_df], ignore_index=True) if not do_df.empty else extra_df
            present_dos.update(val for val in extra_df["do_number"].tolist() if val)
    orphan_dos = sorted(do for do in do_numbers if do not in present_dos)

    st.markdown("**Delivery orders**")
    if (do_df is None or do_df.empty) and not orphan_dos:
        st.info("No delivery orders found for this customer.")
    else:
        if do_df is not None and not do_df.empty:
            st.dataframe(
                do_df.rename(
                    columns={
                        "do_number": "DO Serial",
                        "customer": "Customer",
                        "description": "Description",
                        "sales_person": "Sales person",
                        "remarks": "Remarks",
                        "created_at": "Created",
                        "Document": "Document",
                    }
                ).drop(columns=["file_path"], errors="ignore"),
                use_container_width=True,
            )
        if orphan_dos:
            st.caption("Referenced DO codes without a recorded delivery order: " + ", ".join(orphan_dos))

    st.markdown("**Warranties**")
    if warr_display is None or warr_display.empty:
        st.info("No warranties recorded for this customer.")
    else:
        st.dataframe(warr_display)

    st.markdown("**Service records**")
    if service_df.empty:
        st.info("No service records found for this customer.")
    else:
        service_display = service_df.rename(
            columns={
                "do_number": "DO Serial",
                "service_date": "Service date",
                "service_start_date": "Service start date",
                "service_end_date": "Service end date",
                "service_period": "Service period",
                "service_product_info": "Products sold",
                "description": "Description",
                "remarks": "Remarks",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        st.dataframe(
            service_display.drop(columns=["service_id"], errors="ignore"),
            use_container_width=True,
        )

    st.markdown("**Maintenance records**")
    if maintenance_df.empty:
        st.info("No maintenance records found for this customer.")
    else:
        maintenance_display = maintenance_df.rename(
            columns={
                "do_number": "DO Serial",
                "maintenance_date": "Maintenance date",
                "maintenance_start_date": "Maintenance start date",
                "maintenance_end_date": "Maintenance end date",
                "maintenance_period": "Maintenance period",
                "maintenance_product_info": "Products sold",
                "description": "Description",
                "remarks": "Remarks",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        st.dataframe(
            maintenance_display.drop(columns=["maintenance_id"], errors="ignore"),
            use_container_width=True,
        )

    documents = []
    if do_df is not None and not do_df.empty:
        for _, row in do_df.iterrows():
            path = resolve_upload_path(row.get("file_path"))
            if not path or not path.exists():
                continue
            do_ref = clean_text(row.get("do_number")) or "delivery_order"
            display_name = path.name
            archive_name = "/".join(
                [
                    _sanitize_path_component("delivery_orders"),
                    f"{_sanitize_path_component(do_ref)}_{_sanitize_path_component(display_name)}",
                ]
            )
            documents.append(
                {
                    "source": "Delivery order",
                    "reference": do_ref,
                    "display": display_name,
                    "path": path,
                    "archive_name": archive_name,
                    "key": f"do_{do_ref}",
                }
            )

    service_docs = pd.DataFrame()
    if "service_id" in service_df.columns and not service_df.empty:
        service_ids = [int(val) for val in service_df["service_id"].dropna().astype(int).tolist()]
        if service_ids:
            service_docs = df_query(
                conn,
                f"""
                SELECT document_id, service_id, file_path, original_name, uploaded_at
                FROM service_documents
                WHERE service_id IN ({','.join('?' * len(service_ids))})
                ORDER BY datetime(uploaded_at) DESC, document_id DESC
                """,
                service_ids,
            )
    service_lookup = {}
    if "service_id" in service_df.columns and not service_df.empty:
        for _, row in service_df.iterrows():
            if pd.isna(row.get("service_id")):
                continue
            service_lookup[int(row["service_id"])] = row
    if not service_docs.empty:
        for _, doc_row in service_docs.iterrows():
            path = resolve_upload_path(doc_row.get("file_path"))
            if not path or not path.exists():
                continue
            service_id = int(doc_row.get("service_id"))
            record = service_lookup.get(service_id, {})
            reference = clean_text(record.get("do_number")) or f"Service #{service_id}"
            display_name = clean_text(doc_row.get("original_name")) or path.name
            uploaded = pd.to_datetime(doc_row.get("uploaded_at"), errors="coerce")
            uploaded_fmt = uploaded.strftime("%d-%m-%Y %H:%M") if pd.notna(uploaded) else None
            archive_name = "/".join(
                [
                    _sanitize_path_component("service"),
                    f"{_sanitize_path_component(reference)}_{_sanitize_path_component(display_name)}",
                ]
            )
            documents.append(
                {
                    "source": "Service",
                    "reference": reference,
                    "display": display_name,
                    "uploaded": uploaded_fmt,
                    "path": path,
                    "archive_name": archive_name,
                    "key": f"service_{service_id}_{int(doc_row['document_id'])}",
                }
            )

    maintenance_docs = pd.DataFrame()
    if "maintenance_id" in maintenance_df.columns and not maintenance_df.empty:
        maintenance_ids = [int(val) for val in maintenance_df["maintenance_id"].dropna().astype(int).tolist()]
        if maintenance_ids:
            maintenance_docs = df_query(
                conn,
                f"""
                SELECT document_id, maintenance_id, file_path, original_name, uploaded_at
                FROM maintenance_documents
                WHERE maintenance_id IN ({','.join('?' * len(maintenance_ids))})
                ORDER BY datetime(uploaded_at) DESC, document_id DESC
                """,
                maintenance_ids,
            )
    maintenance_lookup = {}
    if "maintenance_id" in maintenance_df.columns and not maintenance_df.empty:
        for _, row in maintenance_df.iterrows():
            if pd.isna(row.get("maintenance_id")):
                continue
            maintenance_lookup[int(row["maintenance_id"])] = row
    if not maintenance_docs.empty:
        for _, doc_row in maintenance_docs.iterrows():
            path = resolve_upload_path(doc_row.get("file_path"))
            if not path or not path.exists():
                continue
            maintenance_id = int(doc_row.get("maintenance_id"))
            record = maintenance_lookup.get(maintenance_id, {})
            reference = clean_text(record.get("do_number")) or f"Maintenance #{maintenance_id}"
            display_name = clean_text(doc_row.get("original_name")) or path.name
            uploaded = pd.to_datetime(doc_row.get("uploaded_at"), errors="coerce")
            uploaded_fmt = uploaded.strftime("%d-%m-%Y %H:%M") if pd.notna(uploaded) else None
            archive_name = "/".join(
                [
                    _sanitize_path_component("maintenance"),
                    f"{_sanitize_path_component(reference)}_{_sanitize_path_component(display_name)}",
                ]
            )
            documents.append(
                {
                    "source": "Maintenance",
                    "reference": reference,
                    "display": display_name,
                    "uploaded": uploaded_fmt,
                    "path": path,
                    "archive_name": archive_name,
                    "key": f"maintenance_{maintenance_id}_{int(doc_row['document_id'])}",
                }
            )

    documents.sort(key=lambda d: (d["source"], d.get("reference") or "", d.get("display") or ""))

    st.markdown("**Documents**")
    if not documents:
        st.info("No documents attached for this customer.")
    else:
        for idx, doc in enumerate(documents, start=1):
            path = doc.get("path")
            if not path or not path.exists():
                continue
            label = f"{doc['source']}: {doc['reference']} – {doc['display']}"
            if doc.get("uploaded"):
                label = f"{label} (uploaded {doc['uploaded']})"
            st.download_button(
                f"Download {label}",
                data=path.read_bytes(),
                file_name=path.name,
                key=f"cust_doc_{doc['key']}_{idx}",
            )
        zip_buffer = bundle_documents_zip(documents)
        if zip_buffer is not None:
            archive_title = _sanitize_path_component(info.get("name") or blank_label)
            st.download_button(
                "⬇️ Download all documents (.zip)",
                data=zip_buffer.getvalue(),
                file_name=f"{archive_title}_documents.zip",
                mime="application/zip",
                key="cust_docs_zip",
            )

    pdf_bytes = generate_customer_summary_pdf(
        info.get("name") or blank_label,
        info,
        warr_display,
        service_df,
        maintenance_df,
    )
    st.download_button(
        "⬇️ Download summary (PDF)",
        data=pdf_bytes,
        file_name=f"customer_summary_{clean_text(info.get('name')) or 'customer'}.pdf",
        mime="application/pdf",
    )


def scraps_page(conn):
    st.subheader("🗂️ Scraps (Incomplete Records)")
    st.caption(
        "Rows listed here are missing key details (name, phone, or address). They stay hidden from summaries until completed."
    )
    scope_clause, scope_params = customer_scope_filter()
    where_parts = [customer_incomplete_clause()]
    params: list[object] = []
    if scope_clause:
        where_parts.append(scope_clause)
        params.extend(scope_params)
    where_sql = " AND ".join(where_parts)
    scraps = df_query(
        conn,
        f"""
        SELECT customer_id as id, name, phone, email, address, remarks, purchase_date, product_info, delivery_order_code, created_at
        FROM customers
        WHERE {where_sql}
        ORDER BY datetime(created_at) DESC
        """,
        tuple(params),
    )
    scraps = fmt_dates(scraps, ["created_at", "purchase_date"])
    if scraps.empty:
        st.success("No scraps! All customer rows have the required details.")
        return

    def missing_fields(row):
        missing = []
        for col, label in REQUIRED_CUSTOMER_FIELDS.items():
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                missing.append(label)
        return ", ".join(missing)

    scraps = scraps.assign(missing=scraps.apply(missing_fields, axis=1))
    display_cols = ["name", "phone", "email", "address", "remarks", "purchase_date", "product_info", "delivery_order_code", "missing", "created_at"]
    st.dataframe(scraps[display_cols])

    st.markdown("### Update scrap record")
    records = scraps.to_dict("records")
    option_keys = [int(r["id"]) for r in records]
    option_labels = {}
    for r in records:
        rid = int(r["id"])
        name_label = clean_text(r.get("name")) or "(no name)"
        missing_label = clean_text(r.get("missing")) or "—"
        details = missing_label or "complete"
        created = clean_text(r.get("created_at"))
        created_fmt = f" – added {created}" if created else ""
        option_labels[rid] = f"{name_label or '(no name)'} (missing: {details}){created_fmt}"
    selected_id = st.selectbox(
        "Choose a record to fix",
        option_keys,
        format_func=lambda k: option_labels[k],
    )
    selected = next(r for r in records if int(r["id"]) == selected_id)

    def existing_value(key):
        return clean_text(selected.get(key)) or ""

    with st.form("scrap_update_form"):
        name = st.text_input("Name", existing_value("name"))
        phone = st.text_input("Phone", existing_value("phone"))
        email = st.text_input("Email", existing_value("email"))
        address = st.text_area("Address", existing_value("address"))
        purchase = st.text_input("Purchase date (DD-MM-YYYY)", existing_value("purchase_date"))
        product = st.text_input("Product", existing_value("product_info"))
        do_code = st.text_input("Delivery order code", existing_value("delivery_order_code"))
        remarks_text = st.text_area("Remarks", existing_value("remarks"))
        col1, col2 = st.columns(2)
        save = col1.form_submit_button("Save changes", type="primary")
        delete = col2.form_submit_button("Delete scrap")

    if save:
        new_name = clean_text(name)
        new_phone = clean_text(phone)
        new_email = clean_text(email)
        new_address = clean_text(address)
        new_remarks = clean_text(remarks_text)
        purchase_str, _ = date_strings_from_input(purchase)
        new_product = clean_text(product)
        new_do = clean_text(do_code)
        old_phone = clean_text(selected.get("phone"))
        conn.execute(
            "UPDATE customers SET name=?, phone=?, email=?, address=?, remarks=?, purchase_date=?, product_info=?, delivery_order_code=?, dup_flag=0 WHERE customer_id=?",
            (
                new_name,
                new_phone,
                new_email,
                new_address,
                new_remarks,
                purchase_str,
                new_product,
                new_do,
                int(selected_id),
            ),
        )
        old_do = clean_text(selected.get("delivery_order_code"))
        if new_do:
            conn.execute(
                """
                INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, remarks, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(do_number) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    description=excluded.description,
                    remarks=excluded.remarks
                """,
                (
                    new_do,
                    int(selected_id),
                    None,
                    new_product,
                    None,
                    new_remarks,
                    None,
                ),
            )
        if old_do and old_do != new_do:
            conn.execute(
                "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                (old_do, int(selected_id)),
            )
        if old_phone and old_phone != new_phone:
            recalc_customer_duplicate_flag(conn, old_phone)
        if new_phone:
            recalc_customer_duplicate_flag(conn, new_phone)
        conn.commit()
        conn.execute(
            "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
            (
                new_name,
                new_phone,
                new_address,
                new_product,
                new_do,
                purchase_str,
                int(selected_id),
            ),
        )
        conn.commit()
        if new_name and new_phone and new_address:
            st.success("Details saved. This record is now complete and will appear in other pages.")
        else:
            st.info("Details saved, but the record is still incomplete and will remain in Scraps until all required fields are filled.")
        _safe_rerun()

    if delete:
        conn.execute("DELETE FROM customers WHERE customer_id=?", (int(selected_id),))
        conn.commit()
        st.warning("Scrap record deleted.")
        _safe_rerun()

# ---------- Import helpers ----------
def refine_multiline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _as_list(value: object) -> list[object]:
        if isinstance(value, str):
            items = [s.strip() for s in value.splitlines() if s.strip()]
            return items or [None]
        if pd.isna(value):
            return [None]
        return [value]

    listified = {col: df[col].apply(_as_list) for col in df.columns}

    normalized_rows: list[dict[str, object]] = []
    for _, row in pd.DataFrame(listified).iterrows():
        lengths = [len(values) for values in row]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            normalized_rows.append({col: None for col in df.columns})
            continue
        expanded = {
            col: values + [None] * (max_len - len(values))
            for col, values in row.items()
        }
        for idx in range(max_len):
            normalized_rows.append({col: expanded[col][idx] for col in df.columns})

    return pd.DataFrame(normalized_rows, columns=df.columns)


_TRAILING_ZERO_NUMBER = re.compile(r"^-?\d+\.0+$")


def _normalize_sort_value(value: object) -> str:
    text = clean_text(value)
    if text is None:
        return ""
    if _TRAILING_ZERO_NUMBER.match(text):
        return text.split(".", 1)[0]
    return text


def _sort_dataframe_safe(df: pd.DataFrame, sort_columns: Iterable[str]) -> pd.DataFrame:
    columns = [col for col in sort_columns if col in df.columns]
    if not columns:
        return df

    def _sort_key(series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        name = str(series.name or "").lower()
        if "date" in name:
            converted = pd.to_datetime(series, errors="coerce", dayfirst=True)
            if not converted.isna().all():
                return converted
        return series.map(_normalize_sort_value)

    return df.sort_values(by=columns, key=_sort_key, na_position="last")


def normalize_headers(cols):
    norm = []
    for c in cols:
        s = str(c).strip().lower().replace(" ", "_")
        norm.append(s)
    return norm

HEADER_MAP = {
    "date": {"date", "delivery_date", "issue_date", "order_date", "dt", "d_o", "d", "sale_date"},
    "customer_name": {"customer_name", "customer", "company", "company_name", "client", "party", "name"},
    "address": {"address", "addr", "street", "location"},
    "phone": {"phone", "mobile", "contact", "contact_no", "phone_no", "phone_number", "cell", "whatsapp"},
    "email": {"email", "mail", "e_mail", "email_address", "contact_email"},
    "product": {"product", "item", "generator", "model", "description"},
    "do_code": {"do_code", "delivery_order", "delivery_order_code", "delivery_order_no", "do", "d_o_code", "do_number"},
    "remarks": {"remarks", "remark", "notes", "note", "comments", "comment"},
    "amount_spent": {"amount", "amount_spent", "value", "price", "invoice_amount", "total", "total_amount", "amt"},
}

def map_headers_guess(cols):
    cols_norm = normalize_headers(cols)
    mapping = {k: None for k in HEADER_MAP.keys()}
    for i, cn in enumerate(cols_norm):
        for target, aliases in HEADER_MAP.items():
            if cn in aliases and mapping[target] is None:
                mapping[target] = i
                break
    default_order = ["date", "customer_name", "address", "phone", "product", "do_code"]
    if cols_norm[: len(default_order)] == default_order:
        mapping = {field: idx for idx, field in enumerate(default_order)}
    return mapping


def split_product_label(label: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if label is None:
        return None, None
    text = clean_text(label)
    if not text:
        return None, None
    if "-" in text:
        left, right = text.split("-", 1)
        return clean_text(left), clean_text(right)
    return text, None


def parse_date_value(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    dt = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    if isinstance(dt, pd.DatetimeIndex):
        dt = dt[0]
    return dt.normalize()


def date_strings_from_input(value) -> tuple[Optional[str], Optional[str]]:
    dt = parse_date_value(value)
    if dt is None:
        return None, None
    expiry = dt + pd.Timedelta(days=365)
    return dt.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d")


def int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def coerce_excel_date(series):
    s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if s.isna().mean() > 0.5:
        try:
            num = pd.to_numeric(series, errors="coerce")
            if num.notna().sum() > 0 and (num.dropna().median() > 20000):
                s = pd.to_datetime(num, unit="d", origin="1899-12-30", errors="coerce")
        except Exception:
            pass
    return s

def import_page(conn):
    st.subheader("⬆️ Import from Excel/CSV (append)")
    st.caption("We’ll auto-detect columns; you can override mapping. Dates accept DD-MM-YYYY or Excel serials.")
    f = st.file_uploader("Upload .xlsx or .csv", type=["xlsx","csv"])
    if f is None:
        st.markdown("---")
        manage_import_history(conn)
        return
    # Streamlit reruns the script whenever widgets change state. This means the
    # uploaded file object is reused across runs and its pointer sits at the end
    # after the first read. Attempting to read again (e.g. after a selectbox
    # change) would therefore raise an "Excel file format cannot be determined"
    # error or return empty data, effectively restarting the app view. Reset the
    # pointer before every read so interactive mapping works reliably.
    f.seek(0)
    if f.name.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        df = pd.read_excel(f)
    st.write("Preview:", df.head())

    guess = map_headers_guess(list(df.columns))
    cols = list(df.columns)
    opts = ["(blank)"] + cols
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    col6, col7 = st.columns(2)
    col8, _ = st.columns(2)
    sel_date = col1.selectbox(
        "Date", options=opts, index=(guess["date"] + 1) if guess.get("date") is not None else 0
    )
    sel_name = col2.selectbox(
        "Customer name", options=opts, index=(guess["customer_name"] + 1) if guess.get("customer_name") is not None else 0
    )
    sel_addr = col3.selectbox(
        "Address", options=opts, index=(guess["address"] + 1) if guess.get("address") is not None else 0
    )
    sel_phone = col4.selectbox(
        "Phone", options=opts, index=(guess["phone"] + 1) if guess.get("phone") is not None else 0
    )
    sel_prod = col5.selectbox(
        "Product", options=opts, index=(guess["product"] + 1) if guess.get("product") is not None else 0
    )
    sel_do = col6.selectbox(
        "Delivery order code", options=opts, index=(guess["do_code"] + 1) if guess.get("do_code") is not None else 0
    )
    sel_remarks = col7.selectbox(
        "Remarks", options=opts, index=(guess.get("remarks", None) + 1) if guess.get("remarks") is not None else 0
    )
    sel_amount = col8.selectbox(
        "Amount spent", options=opts, index=(guess.get("amount_spent", None) + 1) if guess.get("amount_spent") is not None else 0
    )

    def pick(col_name):
        return df[col_name] if col_name != "(blank)" else pd.Series([None] * len(df))

    df_norm = pd.DataFrame(
        {
            "date": pick(sel_date),
            "customer_name": pick(sel_name),
            "address": pick(sel_addr),
            "phone": pick(sel_phone),
            "product": pick(sel_prod),
            "do_code": pick(sel_do),
            "remarks": pick(sel_remarks),
            "amount_spent": pick(sel_amount),
        }
    )
    skip_blanks = st.checkbox("Skip blank rows", value=True)
    df_norm = refine_multiline(df_norm)
    df_norm["date"] = coerce_excel_date(df_norm["date"])
    df_norm = df_norm.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    if skip_blanks:
        df_norm = df_norm.dropna(how="all")
    df_norm = df_norm.drop_duplicates()
    df_norm = _sort_dataframe_safe(df_norm, ["date", "customer_name", "phone", "do_code"]).reset_index(drop=True)
    st.markdown("#### Review & edit rows before importing")
    preview = df_norm.copy()
    preview["Action"] = "Import"
    editor = st.data_editor(
        preview,
        key="import_editor",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD-MM-YYYY", required=False),
            "Action": st.column_config.SelectboxColumn("Action", options=["Import", "Skip"], required=True),
            "remarks": st.column_config.TextColumn("Remarks", required=False),
            "amount_spent": st.column_config.NumberColumn(
                "Amount spent", min_value=0.0, step=0.01, format="%.2f", required=False
            ),
        },
    )

    if st.button("Append into database"):
        editor = editor if isinstance(editor, pd.DataFrame) else pd.DataFrame(editor)
        ready = editor[editor["Action"].fillna("Import").str.lower() == "import"].copy()
        ready.drop(columns=["Action"], inplace=True, errors="ignore")
        seeded, d_c, d_p = _import_clean6(conn, ready, tag="Manual import (mapped)")
        if seeded == 0:
            st.warning("No rows added (rows empty/invalid). Check mapping or file.")
        else:
            st.success(f"Imported {seeded} rows. Duplicates flagged — customers: {d_c}, products: {d_p}.")

    st.markdown("---")
    manage_import_history(conn)

def manual_merge_section(conn, customers_df: pd.DataFrame) -> None:
    if customers_df is None or customers_df.empty:
        return

    if "id" not in customers_df.columns:
        return

    work_df = customers_df.copy()
    work_df["id"] = work_df["id"].apply(int_or_none)
    work_df = work_df[work_df["id"].notna()]
    if work_df.empty:
        return

    work_df["id"] = work_df["id"].astype(int)

    def build_label(row):
        name_val = clean_text(row.get("name")) or "(no name)"
        phone_val = clean_text(row.get("phone")) or "(no phone)"
        address_val = clean_text(row.get("address")) or "-"
        product_val = clean_text(row.get("product_info")) or "-"
        do_val = clean_text(row.get("delivery_order_code")) or "-"
        date_dt = parse_date_value(row.get("purchase_date"))
        if date_dt is not None:
            date_label = date_dt.strftime(DATE_FMT)
        else:
            date_label = clean_text(row.get("purchase_date")) or "-"
        return f"#{row['id']} – {name_val} | Phone: {phone_val} | Date: {date_label} | Product: {product_val} | DO: {do_val}"

    work_df["_label"] = work_df.apply(build_label, axis=1)
    work_df["_search_blob"] = work_df.apply(
        lambda row: " ".join(
            filter(
                None,
                [
                    clean_text(row.get("name")),
                    clean_text(row.get("phone")),
                    clean_text(row.get("address")),
                    clean_text(row.get("product_info")),
                    clean_text(row.get("delivery_order_code")),
                ],
            )
        ),
        axis=1,
    )
    work_df["_search_blob"] = work_df["_search_blob"].fillna("").str.lower()

    label_map = {row["id"]: row["_label"] for row in work_df.to_dict("records")}

    st.divider()
    st.markdown("#### Manual customer merge")
    st.caption(
        "Select multiple customer records that refer to the same person even if the phone number or purchase date differs. "
        "The earliest record will be kept and enriched with the combined details."
    )

    filter_value = st.text_input(
        "Filter customers by name, phone, address, product, or DO (optional)",
        key="manual_merge_filter",
    ).strip()

    filtered_df = work_df
    if filter_value:
        escaped = re.escape(filter_value.lower())
        mask = filtered_df["_search_blob"].str.contains(escaped, regex=True, na=False)
        filtered_df = filtered_df[mask]

    options = filtered_df["id"].tolist()
    if not options:
        st.info("No customer records match the current filter.")
        return

    with st.form("manual_merge_form"):
        selected_ids = st.multiselect(
            "Select customer records to merge",
            options=options,
            format_func=lambda cid: label_map.get(cid, f"#{cid}"),
        )

        preview_df = work_df[work_df["id"].isin(selected_ids)]
        if not preview_df.empty:
            preview_df = preview_df.copy()
            preview_df["purchase_date"] = pd.to_datetime(preview_df["purchase_date"], errors="coerce")
            preview_df["purchase_date"] = preview_df["purchase_date"].dt.strftime(DATE_FMT)
            preview_df["purchase_date"] = preview_df["purchase_date"].fillna("-")
            preview_cols = [
                col
                for col in [
                    "id",
                    "name",
                    "phone",
                    "address",
                    "purchase_date",
                    "product_info",
                    "delivery_order_code",
                    "created_at",
                ]
                if col in preview_df.columns
            ]
            st.dataframe(
                preview_df[preview_cols]
                .rename(
                    columns={
                        "id": "ID",
                        "name": "Name",
                        "phone": "Phone",
                        "address": "Address",
                        "purchase_date": "Purchase date",
                        "product_info": "Product",
                        "delivery_order_code": "DO code",
                        "created_at": "Created",
                    }
                )
                .sort_values("ID"),
                use_container_width=True,
                hide_index=True,
            )

        submitted = st.form_submit_button("Merge selected customers", type="primary")

    if submitted:
        if len(selected_ids) < 2:
            st.warning("Select at least two customers to merge.")
            return
        if merge_customer_records(conn, selected_ids):
            st.success(f"Merged {len(selected_ids)} customer records.")
            _safe_rerun()
        else:
            st.error("Could not merge the selected customers. Please try again.")


def duplicates_page(conn):
    st.subheader("⚠️ Possible Duplicates")
    scope_clause, scope_params = customer_scope_filter("c")
    where_sql = f"WHERE {scope_clause}" if scope_clause else ""
    cust_raw = df_query(
        conn,
        f"""
        SELECT
            c.customer_id as id,
            c.name,
            c.phone,
            c.email,
            c.address,
            c.purchase_date,
            c.product_info,
            c.delivery_order_code,
            c.dup_flag,
            c.created_at
        FROM customers c
        {where_sql}
        ORDER BY datetime(c.created_at) DESC
        """,
        scope_params if scope_clause else (),
    )
    warr = df_query(
        conn,
        "SELECT w.warranty_id as id, c.name as customer, p.name as product, p.model, w.serial, w.issue_date, w.expiry_date, w.remarks, w.dup_flag FROM warranties w LEFT JOIN customers c ON c.customer_id = w.customer_id LEFT JOIN products p ON p.product_id = w.product_id ORDER BY date(w.issue_date) DESC",
    )
    duplicate_customers = pd.DataFrame()
    if not cust_raw.empty:
        duplicate_customers = cust_raw[cust_raw["dup_flag"] == 1].copy()
    if duplicate_customers.empty:
        st.success("No customer duplicates detected at the moment.")
    else:
        editor_df = duplicate_customers.copy()
        editor_df["__group_key"] = [
            " | ".join(
                [
                    clean_text(row.get("phone")) or "(no phone)",
                    (
                        parse_date_value(row.get("purchase_date")).strftime(DATE_FMT)
                        if parse_date_value(row.get("purchase_date")) is not None
                        else "-"
                    ),
                    clean_text(row.get("product_info")) or "-",
                ]
            )
            for _, row in editor_df.iterrows()
        ]
        preview_df = editor_df.assign(
            duplicate="🔁 duplicate phone",
            purchase_date_fmt=pd.to_datetime(editor_df["purchase_date"], errors="coerce").dt.strftime(DATE_FMT),
            created_at_fmt=pd.to_datetime(editor_df["created_at"], errors="coerce").dt.strftime("%d-%m-%Y %H:%M"),
        )
        preview_cols = [
            col
            for col in [
                "__group_key",
                "name",
                "phone",
                "email",
                "address",
                "purchase_date_fmt",
                "product_info",
                "delivery_order_code",
                "duplicate",
                "created_at_fmt",
            ]
            if col in preview_df.columns
        ]
        if preview_cols:
            display_df = (
                preview_df[preview_cols]
                .rename(
                    columns={
                        "__group_key": "Duplicate set",
                        "purchase_date_fmt": "Purchase date",
                        "product_info": "Product",
                        "delivery_order_code": "DO code",
                        "created_at_fmt": "Created",
                    }
                )
                .sort_values(by=["Duplicate set", "Created"], na_position="last")
            )
            display_df["Purchase date"] = display_df["Purchase date"].fillna("-")
            display_df["Created"] = display_df["Created"].fillna("-")
            st.markdown("#### Duplicate rows")
            st.caption(
                "Each duplicate set groups rows sharing the same phone, purchase date, and product so you can double-check real multi-unit sales."
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        group_counts = editor_df.groupby("__group_key").size().to_dict()
        selection_options = [(None, "All duplicate rows")] + [
            (label, f"{label} ({group_counts.get(label, 0)} row(s))") for label in sorted(editor_df["__group_key"].unique())
        ]
        selected_group, _ = st.selectbox(
            "Focus on a duplicate set (optional)",
            options=selection_options,
            index=0,
            format_func=lambda opt: opt[1],
        )
        if selected_group:
            editor_df = editor_df[editor_df["__group_key"] == selected_group]
        if editor_df.empty:
            st.info("No rows match the selected duplicate set.")
        else:
            editor_df["duplicate"] = "🔁 duplicate phone"
            editor_df["purchase_date"] = pd.to_datetime(editor_df["purchase_date"], errors="coerce")
            editor_df["created_at"] = pd.to_datetime(editor_df["created_at"], errors="coerce")
            editor_df["Action"] = "Keep"
            column_order = [
                col
                for col in [
                    "id",
                    "name",
                    "phone",
                    "email",
                    "address",
                    "purchase_date",
                    "product_info",
                    "delivery_order_code",
                    "duplicate",
                    "created_at",
                    "Action",
                ]
                if col in editor_df.columns
            ]
            editor_df = editor_df[column_order]
            st.markdown("#### Edit duplicate entries")
            editor_state = st.data_editor(
                editor_df,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "id": st.column_config.Column("ID", disabled=True),
                    "name": st.column_config.TextColumn("Name"),
                    "phone": st.column_config.TextColumn("Phone"),
                    "email": st.column_config.TextColumn("Email"),
                    "address": st.column_config.TextColumn("Address"),
                    "purchase_date": st.column_config.DateColumn("Purchase date", format="DD-MM-YYYY", required=False),
                    "product_info": st.column_config.TextColumn("Product"),
                    "delivery_order_code": st.column_config.TextColumn("DO code"),
                    "duplicate": st.column_config.Column("Duplicate", disabled=True),
                    "created_at": st.column_config.DatetimeColumn("Created", format="DD-MM-YYYY HH:mm", disabled=True),
                    "Action": st.column_config.SelectboxColumn("Action", options=["Keep", "Delete"], required=True),
                },
            )
            user = st.session_state.user or {}
            is_admin = user.get("role") == "admin"
            if not is_admin:
                st.caption("Deleting rows requires admin privileges; non-admin delete actions will be ignored.")
            raw_map = {int(row["id"]): row for row in duplicate_customers.to_dict("records") if int_or_none(row.get("id")) is not None}
            if st.button("Apply duplicate table updates", type="primary"):
                editor_result = editor_state if isinstance(editor_state, pd.DataFrame) else pd.DataFrame(editor_state)
                if editor_result.empty:
                    st.info("No rows to update.")
                else:
                    phones_to_recalc: set[str] = set()
                    updates = deletes = 0
                    errors: list[str] = []
                    made_updates = False
                    for row in editor_result.to_dict("records"):
                        cid = int_or_none(row.get("id"))
                        if cid is None or cid not in raw_map:
                            continue
                        action = str(row.get("Action") or "Keep").strip().lower()
                        if action == "delete":
                            if is_admin:
                                delete_customer_record(conn, cid)
                                deletes += 1
                            else:
                                errors.append(f"Only admins can delete customers (ID #{cid}).")
                            continue
                        new_name = clean_text(row.get("name"))
                        new_phone = clean_text(row.get("phone"))
                        new_email = clean_text(row.get("email"))
                        new_address = clean_text(row.get("address"))
                        purchase_str, _ = date_strings_from_input(row.get("purchase_date"))
                        product_label = clean_text(row.get("product_info"))
                        new_do = clean_text(row.get("delivery_order_code"))
                        original_row = raw_map[cid]
                        old_name = clean_text(original_row.get("name"))
                        old_phone = clean_text(original_row.get("phone"))
                        old_email = clean_text(original_row.get("email"))
                        old_address = clean_text(original_row.get("address"))
                        old_purchase = clean_text(original_row.get("purchase_date"))
                        old_product = clean_text(original_row.get("product_info"))
                        old_do = clean_text(original_row.get("delivery_order_code"))
                        if (
                            new_name == old_name
                            and new_phone == old_phone
                            and new_email == old_email
                            and new_address == old_address
                            and purchase_str == old_purchase
                            and product_label == old_product
                            and new_do == old_do
                        ):
                            continue
                        conn.execute(
                            "UPDATE customers SET name=?, phone=?, email=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, dup_flag=0 WHERE customer_id=?",
                            (
                                new_name,
                                new_phone,
                                new_email,
                                new_address,
                                purchase_str,
                                product_label,
                                new_do,
                                cid,
                            ),
                        )
                        if new_do:
                            conn.execute(
                                """
                                INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, remarks, file_path)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(do_number) DO UPDATE SET
                                    customer_id=excluded.customer_id,
                                    description=excluded.description,
                                    remarks=excluded.remarks
                                """,
                                (
                                    new_do,
                                    cid,
                                    None,
                                    product_label,
                                    None,
                                    None,
                                    None,
                                ),
                            )
                        if old_do and old_do != new_do:
                            conn.execute(
                                "DELETE FROM delivery_orders WHERE do_number=? AND (customer_id IS NULL OR customer_id=?)",
                                (old_do, cid),
                            )
                        conn.execute(
                            "UPDATE import_history SET customer_name=?, phone=?, address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                            (
                                new_name,
                                new_phone,
                                new_address,
                                product_label,
                                new_do,
                                purchase_str,
                                cid,
                            ),
                        )
                        if old_phone and old_phone != new_phone:
                            phones_to_recalc.add(old_phone)
                        if new_phone:
                            phones_to_recalc.add(new_phone)
                        updates += 1
                        made_updates = True
                    if made_updates:
                        conn.commit()
                    if phones_to_recalc:
                        for phone_value in phones_to_recalc:
                            recalc_customer_duplicate_flag(conn, phone_value)
                        conn.commit()
                    if errors:
                        for err in errors:
                            st.error(err)
                    if updates or deletes:
                        st.success(f"Updated {updates} row(s) and deleted {deletes} row(s).")
                        if not errors:
                            _safe_rerun()
                    elif not errors:
                        st.info("No changes detected.")
    manual_merge_section(conn, cust_raw)

    if not warr.empty:
        warr = fmt_dates(warr, ["issue_date", "expiry_date"])
        warr = warr.assign(duplicate=warr["dup_flag"].apply(lambda x: "🔁 duplicate serial" if int(x)==1 else ""))
        st.markdown("**Warranties (duplicate serial)**")
        st.dataframe(
            warr[warr["dup_flag"] == 1].drop(columns=["id", "dup_flag"], errors="ignore"),
            use_container_width=True,
        )
def users_admin_page(conn):
    ensure_auth(role="admin")
    st.subheader("👤 Users (Admin)")
    users = df_query(conn, "SELECT user_id as id, username, role, created_at FROM users ORDER BY datetime(created_at) DESC")
    users = users.assign(created_at=pd.to_datetime(users["created_at"], errors="coerce").dt.strftime(DATE_FMT))
    st.dataframe(users.drop(columns=["id"], errors="ignore"))

    with st.expander("Add user"):
        with st.form("add_user"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["staff", "admin"])
            ok = st.form_submit_button("Create")
            if ok and u.strip() and p.strip():
                h = hashlib.sha256(p.encode("utf-8")).hexdigest()
                try:
                    conn.execute("INSERT INTO users (username, pass_hash, role) VALUES (?, ?, ?)", (u.strip(), h, role))
                    conn.commit()
                    st.success("User added")
                except sqlite3.IntegrityError:
                    st.error("Username already exists")

    with st.expander("Reset password / delete"):
        uid = st.number_input("User ID", min_value=1, step=1)
        newp = st.text_input("New password", type="password")
        col1, col2 = st.columns(2)
        if col1.button("Set new password"):
            h = hashlib.sha256(newp.encode("utf-8")).hexdigest()
            conn.execute("UPDATE users SET pass_hash=? WHERE user_id=?", (h, int(uid)))
            conn.commit()
            st.success("Password updated")
        if col2.button("Delete user"):
            conn.execute("DELETE FROM users WHERE user_id=?", (int(uid),))
            conn.commit()
            st.warning("User deleted")

# ---------- Import engine ----------
def _import_clean6(conn, df, tag="Import"):
    """Import cleaned dataframe into database.

    The function is resilient to messy input: it normalizes and sorts the
    dataframe internally so callers can pass raw data without pre-processing.
    """
    # ensure dataframe is normalized even if caller didn't pre-clean
    df = df.copy()
    df = refine_multiline(df)
    if "date" in df.columns:
        df["date"] = coerce_excel_date(df["date"])
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    sort_cols = [col for col in ["date", "customer_name", "phone", "do_code"] if col in df.columns]
    if not sort_cols:
        sort_cols = df.columns.tolist()
    df = df.dropna(how="all").drop_duplicates()
    df = _sort_dataframe_safe(df, sort_cols).reset_index(drop=True)

    cur = conn.cursor()
    seeded = 0
    d_c = d_p = 0
    phones_to_recalc: set[str] = set()
    created_by = current_user_id()
    for _, r in df.iterrows():
        d = r.get("date", pd.NaT)
        cust = clean_text(r.get("customer_name"))
        addr = clean_text(r.get("address"))
        phone = clean_text(r.get("phone"))
        email = clean_text(r.get("email"))
        product_label = clean_text(r.get("product"))
        do_serial = clean_text(r.get("do_code"))
        remarks_val = clean_text(r.get("remarks"))
        amount_value = parse_amount(r.get("amount_spent"))
        if cust is None and phone is None and product_label is None:
            continue
        purchase_dt = parse_date_value(d)
        purchase_str = purchase_dt.strftime("%Y-%m-%d") if isinstance(purchase_dt, pd.Timestamp) else None
        # dup checks
        def exists_phone(phone_value, purchase_value, do_value, product_value):
            normalized_phone = clean_text(phone_value)
            if not normalized_phone:
                return False
            clauses = ["phone = ?"]
            params: list[object] = [normalized_phone]
            if purchase_value:
                clauses.append("IFNULL(purchase_date, '') = ?")
                params.append(purchase_value)
            else:
                clauses.append("(purchase_date IS NULL OR purchase_date = '')")
            if do_value:
                clauses.append("LOWER(IFNULL(delivery_order_code, '')) = LOWER(?)")
                params.append(do_value)
            elif product_value:
                clauses.append("LOWER(IFNULL(product_info, '')) = LOWER(?)")
                params.append(product_value)
            query = f"SELECT 1 FROM customers WHERE {' AND '.join(clauses)} LIMIT 1"
            cur.execute(query, tuple(params))
            return cur.fetchone() is not None

        dupc = 1 if exists_phone(phone, purchase_str, do_serial, product_label) else 0
        cur.execute(
            "INSERT INTO customers (name, phone, email, address, remarks, amount_spent, created_by, dup_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cust, phone, email, addr, remarks_val, amount_value, created_by, dupc),
        )
        cid = cur.lastrowid
        if dupc:
            d_c += 1
        if phone:
            normalized_phone = clean_text(phone)
            if normalized_phone:
                phones_to_recalc.add(normalized_phone)

        name, model = split_product_label(product_label)

        def exists_prod(name, model):
            if not name:
                return False
            cur.execute(
                "SELECT 1 FROM products WHERE name = ? AND IFNULL(model,'') = IFNULL(?, '') LIMIT 1",
                (name, model),
            )
            return cur.fetchone() is not None

        dupp = 1 if exists_prod(name, model) else 0
        cur.execute(
            "INSERT INTO products (name, model, dup_flag) VALUES (?, ?, ?)",
            (name, model, dupp),
        )
        pid = cur.lastrowid
        if dupp:
            d_p += 1

        # we still record orders (hidden) to keep a timeline if needed
        base_dt = purchase_dt or pd.Timestamp.now().normalize()
        order_date = base_dt
        delivery_date = base_dt
        cur.execute(
            "INSERT INTO orders (customer_id, order_date, delivery_date, notes) VALUES (?, ?, ?, ?)",
            (
                cid,
                order_date.strftime("%Y-%m-%d") if order_date is not None else None,
                delivery_date.strftime("%Y-%m-%d") if delivery_date is not None else None,
                f"Imported ({tag})",
            ),
        )
        oid = cur.lastrowid
        cur.execute(
            "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?, ?, ?)",
            (oid, pid, 1),
        )
        order_item_id = cur.lastrowid

        base = base_dt
        expiry = base + pd.Timedelta(days=365)
        cur.execute(
            "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status, remarks, dup_flag) VALUES (?, ?, ?, ?, ?, 'active', ?, 0)",
            (
                cid,
                pid,
                None,
                base.strftime("%Y-%m-%d"),
                expiry.strftime("%Y-%m-%d"),
                remarks_val,
            ),
        )
        warranty_id = cur.lastrowid

        if do_serial:
            description = product_label
            cur.execute(
                "INSERT OR IGNORE INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, remarks, file_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    do_serial,
                    cid,
                    oid,
                    description,
                    None,
                    remarks_val,
                    None,
                ),
            )
        purchase_date = purchase_str or (base.strftime("%Y-%m-%d") if isinstance(base, pd.Timestamp) else None)
        cur.execute(
            "UPDATE customers SET purchase_date=?, product_info=?, delivery_order_code=?, remarks=?, amount_spent=? WHERE customer_id=?",
            (
                purchase_date,
                product_label,
                do_serial,
                remarks_val,
                amount_value,
                cid,
            ),
        )
        cur.execute(
            "INSERT INTO import_history (customer_id, product_id, order_id, order_item_id, warranty_id, do_number, import_tag, original_date, customer_name, address, phone, product_label, notes, amount_spent, imported_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cid,
                pid,
                oid,
                order_item_id,
                warranty_id,
                do_serial,
                tag,
                purchase_date,
                cust,
                addr,
                phone,
                product_label,
                remarks_val,
                amount_value,
                created_by,
            ),
        )
        seeded += 1
    conn.commit()
    for p in phones_to_recalc:
        recalc_customer_duplicate_flag(conn, p)
    conn.commit()
    return seeded, d_c, d_p


def update_import_entry(conn, record: dict, updates: dict) -> None:
    cur = conn.cursor()
    import_id = int_or_none(record.get("import_id"))
    if import_id is None:
        return

    customer_id = int_or_none(record.get("customer_id"))
    product_id = int_or_none(record.get("product_id"))
    order_id = int_or_none(record.get("order_id"))
    order_item_id = int_or_none(record.get("order_item_id"))
    warranty_id = int_or_none(record.get("warranty_id"))

    old_phone = clean_text(record.get("live_phone")) or clean_text(record.get("phone"))
    old_do = clean_text(record.get("do_number"))

    new_name = clean_text(updates.get("customer_name"))
    new_phone = clean_text(updates.get("phone"))
    new_address = clean_text(updates.get("address"))
    purchase_date_str, expiry_str = date_strings_from_input(updates.get("purchase_date"))
    product_label = clean_text(updates.get("product_label"))
    new_do = clean_text(updates.get("do_number"))
    new_remarks = clean_text(updates.get("remarks"))
    new_amount = parse_amount(updates.get("amount_spent"))
    product_name, product_model = split_product_label(product_label)

    if customer_id is not None:
        cur.execute(
            "UPDATE customers SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, remarks=?, amount_spent=?, dup_flag=0 WHERE customer_id=?",
            (
                new_name,
                new_phone,
                new_address,
                purchase_date_str,
                product_label,
                new_do,
                new_remarks,
                new_amount,
                customer_id,
            ),
        )

    if order_id is not None:
        cur.execute(
            "UPDATE orders SET order_date=?, delivery_date=? WHERE order_id=?",
            (purchase_date_str, purchase_date_str, order_id),
        )

    if order_item_id is not None:
            cur.execute(
                "UPDATE order_items SET quantity=? WHERE order_item_id=?",
                (1, order_item_id),
            )

    if product_id is not None:
            cur.execute(
                "UPDATE products SET name=?, model=? WHERE product_id=?",
                (product_name, product_model, product_id),
            )

    if warranty_id is not None:
        cur.execute(
            "UPDATE warranties SET issue_date=?, expiry_date=?, status='active', remarks=? WHERE warranty_id=?",
            (purchase_date_str, expiry_str, new_remarks, warranty_id),
        )

    if new_do:
        cur.execute(
            """
            INSERT INTO delivery_orders (do_number, customer_id, order_id, description, sales_person, remarks, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(do_number) DO UPDATE SET
                customer_id=excluded.customer_id,
                order_id=excluded.order_id,
                description=excluded.description,
                remarks=excluded.remarks
            """,
            (
                new_do,
                customer_id,
                order_id,
                product_label,
                None,
                new_remarks,
                None,
            ),
        )
    if old_do and old_do != new_do:
        params = [old_do]
        query = "DELETE FROM delivery_orders WHERE do_number=?"
        if order_id is not None:
            query += " AND (order_id IS NULL OR order_id=?)"
            params.append(order_id)
        cur.execute(query, tuple(params))

        cur.execute(
            "UPDATE import_history SET original_date=?, customer_name=?, address=?, phone=?, product_label=?, do_number=?, notes=?, amount_spent=? WHERE import_id=?",
            (
                purchase_date_str,
                new_name,
                new_address,
                new_phone,
                product_label,
                new_do,
                new_remarks,
                new_amount,
                import_id,
            ),
        )
    conn.commit()

    if old_phone and old_phone != new_phone:
        recalc_customer_duplicate_flag(conn, old_phone)
    if new_phone:
        recalc_customer_duplicate_flag(conn, new_phone)
    conn.commit()


def delete_import_entry(conn, record: dict) -> None:
    cur = conn.cursor()
    import_id = int_or_none(record.get("import_id"))
    if import_id is None:
        return

    customer_id = int_or_none(record.get("customer_id"))
    product_id = int_or_none(record.get("product_id"))
    order_id = int_or_none(record.get("order_id"))
    order_item_id = int_or_none(record.get("order_item_id"))
    warranty_id = int_or_none(record.get("warranty_id"))
    do_number = clean_text(record.get("do_number"))
    attachment_path = record.get("live_attachment_path")

    old_phone = clean_text(record.get("live_phone")) or clean_text(record.get("phone"))

    if do_number:
        params = [do_number]
        query = "DELETE FROM delivery_orders WHERE do_number=?"
        if order_id is not None:
            query += " AND (order_id IS NULL OR order_id=?)"
            params.append(order_id)
        cur.execute(query, tuple(params))

    if warranty_id is not None:
        cur.execute("DELETE FROM warranties WHERE warranty_id=?", (warranty_id,))
    if order_item_id is not None:
        cur.execute("DELETE FROM order_items WHERE order_item_id=?", (order_item_id,))
    if order_id is not None:
        cur.execute("DELETE FROM orders WHERE order_id=?", (order_id,))
    if product_id is not None:
        cur.execute("DELETE FROM products WHERE product_id=?", (product_id,))
    if customer_id is not None:
        cur.execute("DELETE FROM customers WHERE customer_id=?", (customer_id,))

    cur.execute("UPDATE import_history SET deleted_at = datetime('now') WHERE import_id=?", (import_id,))
    conn.commit()

    if attachment_path:
        path = resolve_upload_path(attachment_path)
        if path and path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    if old_phone:
        recalc_customer_duplicate_flag(conn, old_phone)
        conn.commit()


def _normalize_report_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    cleaned = clean_text(value)
    return cleaned


def upsert_work_report(
    conn,
    *,
    report_id: Optional[int],
    user_id: int,
    period_type: str,
    period_start,
    period_end,
    tasks: Optional[str],
    remarks: Optional[str],
    research: Optional[str],
    grid_rows: Optional[Iterable[dict]] = None,
    attachment_path=_ATTACHMENT_UNCHANGED,
    current_attachment: Optional[str] = None,
) -> int:
    if user_id is None:
        raise ValueError("User is required to save a report.")

    key, normalized_start, normalized_end = normalize_report_window(
        period_type, period_start, period_end
    )
    start_iso = normalized_start.isoformat()
    end_iso = normalized_end.isoformat()

    tasks_val = _normalize_report_text(tasks)
    remarks_val = _normalize_report_text(remarks)
    research_val = _normalize_report_text(research)
    grid_payload_val = prepare_report_grid_payload(grid_rows or [])

    cur = conn.cursor()
    effective_id = report_id
    cur.execute(
        """
        SELECT report_id, attachment_path
        FROM work_reports
        WHERE user_id=? AND period_type=? AND period_start=?
        LIMIT 1
        """,
        (user_id, key, start_iso),
    )
    row = cur.fetchone()
    if row:
        existing_id = int(row[0])
        if effective_id is None:
            effective_id = existing_id
            if current_attachment is None:
                current_attachment = row[1]
        elif existing_id != effective_id:
            raise ValueError(
                "Another report already exists for this period. Select it from the dropdown to edit."
            )

    if effective_id is not None and current_attachment is None:
        cur.execute(
            "SELECT attachment_path FROM work_reports WHERE report_id=?",
            (effective_id,),
        )
        match = cur.fetchone()
        if match:
            current_attachment = match[0]

    if attachment_path is _ATTACHMENT_UNCHANGED:
        attachment_value = current_attachment
    else:
        attachment_value = attachment_path

    created_new = False
    if effective_id is None:
        try:
            cur.execute(
                """
                INSERT INTO work_reports (user_id, period_type, period_start, period_end, tasks, remarks, research, grid_payload, attachment_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    key,
                    start_iso,
                    end_iso,
                    tasks_val,
                    remarks_val,
                    research_val,
                    grid_payload_val,
                    attachment_value,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Another report already exists for this period. Select it from the dropdown to edit."
            ) from exc
        effective_id = int(cur.lastrowid)
        created_new = True
    else:
        try:
            cur.execute(
                """
                UPDATE work_reports
                SET period_type=?, period_start=?, period_end=?, tasks=?, remarks=?, research=?, grid_payload=?, attachment_path=?, updated_at=datetime('now')
                WHERE report_id=?
                """,
                (
                    key,
                    start_iso,
                    end_iso,
                    tasks_val,
                    remarks_val,
                    research_val,
                    grid_payload_val,
                    attachment_value,
                    effective_id,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Another report already exists for this period. Select it from the dropdown to edit."
            ) from exc

    conn.commit()
    cadence_label = REPORT_PERIOD_OPTIONS.get(key, key.title())
    period_label = format_period_range(start_iso, end_iso)
    owner_label = None
    try:
        owner_row = conn.execute(
            "SELECT username FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if owner_row:
            owner_label = clean_text(owner_row[0])
    except sqlite3.Error:
        owner_label = None
    actor = owner_label or f"User #{user_id}"
    event_type = "report_submitted" if created_new else "report_updated"
    verb = "submitted" if created_new else "updated"
    description = f"{actor} {verb} {cadence_label.lower()} report ({period_label})"
    log_activity(
        conn,
        event_type=event_type,
        description=description,
        entity_type="report",
        entity_id=int(effective_id),
        user_id=user_id,
    )
    return effective_id


def manage_import_history(conn):
    st.subheader("🗃️ Manage imported rows")
    user = get_current_user()
    is_admin = user.get("role") == "admin"
    where_parts = ["ih.deleted_at IS NULL"]
    params: list[object] = []
    if not is_admin:
        user_id = current_user_id()
        if user_id is None:
            where_parts.append("1=0")
        else:
            where_parts.append(
                "(ih.imported_by = ? OR (ih.imported_by IS NULL AND c.created_by = ?))"
            )
            params.extend([user_id, user_id])
    where_clause = " AND ".join(where_parts)
    hist = df_query(
        conn,
        f"""
        SELECT ih.*, c.name AS live_customer_name, c.address AS live_address, c.phone AS live_phone,
               c.purchase_date AS live_purchase_date, c.product_info AS live_product_info,
               c.delivery_order_code AS live_do_code, c.attachment_path AS live_attachment_path
        FROM import_history ih
        LEFT JOIN customers c ON c.customer_id = ih.customer_id
        WHERE {where_clause}
        ORDER BY ih.import_id DESC
        LIMIT 200
        """,
        tuple(params),
    )
    if hist.empty:
        st.info("No imported rows yet. Upload a file to get started.")
        return

    display_cols = [
        "import_id",
        "import_tag",
        "imported_at",
        "customer_name",
        "phone",
        "product_label",
        "do_number",
        "amount_spent",
    ]
    display = hist[display_cols].copy()
    display = fmt_dates(display, ["imported_at"])
    display.rename(
        columns={
            "import_id": "ID",
            "import_tag": "Tag",
            "imported_at": "Imported",
            "customer_name": "Customer",
            "phone": "Phone",
            "product_label": "Product",
            "do_number": "DO code",
            "amount_spent": "Amount spent",
        },
        inplace=True,
    )
    st.dataframe(display, use_container_width=True)

    ids = hist["import_id"].astype(int).tolist()
    label_map = {}
    for _, row in hist.iterrows():
        name = clean_text(row.get("customer_name")) or clean_text(row.get("live_customer_name")) or "(no name)"
        tag = clean_text(row.get("import_tag")) or "import"
        label_map[int(row["import_id"])] = f"#{int(row['import_id'])} – {name} ({tag})"

    selected_id = st.selectbox(
        "Select an import entry",
        ids,
        format_func=lambda x: label_map.get(int(x), str(x)),
    )
    selected = hist[hist["import_id"] == selected_id].iloc[0].to_dict()
    current_name = clean_text(selected.get("live_customer_name")) or clean_text(selected.get("customer_name")) or ""
    current_phone = clean_text(selected.get("live_phone")) or clean_text(selected.get("phone")) or ""
    current_address = clean_text(selected.get("live_address")) or clean_text(selected.get("address")) or ""
    current_product = clean_text(selected.get("live_product_info")) or clean_text(selected.get("product_label")) or ""
    current_do = clean_text(selected.get("live_do_code")) or clean_text(selected.get("do_number")) or ""
    purchase_seed = selected.get("live_purchase_date") or selected.get("original_date")
    purchase_str = clean_text(purchase_seed) or ""
    amount_seed = selected.get("amount_spent")
    amount_value = parse_amount(amount_seed)
    amount_display = ""
    if amount_value is not None:
        amount_display = format_money(amount_value) or f"{amount_value:,.2f}"

    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"

    with st.form(f"manage_import_{selected_id}"):
        name_input = st.text_input("Customer name", value=current_name)
        phone_input = st.text_input("Phone", value=current_phone)
        address_input = st.text_area("Address", value=current_address)
        purchase_input = st.text_input("Purchase date (DD-MM-YYYY)", value=purchase_str)
        product_input = st.text_input("Product", value=current_product)
        do_input = st.text_input("Delivery order code", value=current_do)
        remarks_input = st.text_area(
            "Remarks",
            value=clean_text(selected.get("notes")) or "",
            help="Optional remarks stored with this import entry.",
        )
        amount_input = st.text_input(
            "Amount spent",
            value=amount_display,
            help="Track how much was spent for this imported row.",
        )
        col1, col2 = st.columns(2)
        save_btn = col1.form_submit_button("Save changes", type="primary")
        delete_btn = col2.form_submit_button("Delete import", disabled=not is_admin)

    if save_btn:
        update_import_entry(
            conn,
            selected,
            {
                "customer_name": name_input,
                "phone": phone_input,
                "address": address_input,
                "purchase_date": purchase_input,
                "product_label": product_input,
                "do_number": do_input,
                "remarks": remarks_input,
                "amount_spent": amount_input,
            },
        )
        conn.execute(
            "UPDATE import_history SET notes=?, amount_spent=? WHERE import_id=?",
            (
                clean_text(remarks_input),
                parse_amount(amount_input),
                int(selected_id),
            ),
        )
        conn.commit()
        st.success("Import entry updated.")
        _safe_rerun()

    if delete_btn and is_admin:
        delete_import_entry(conn, selected)
        st.warning("Import entry deleted.")
        _safe_rerun()
    elif delete_btn and not is_admin:
        st.error("Only admins can delete import rows.")

# ---------- Reports ----------
def reports_page(conn):
    st.subheader("📈 Work reports")
    user = get_current_user()
    if not user:
        st.info("Log in to capture and review team reports.")
        return

    viewer_id = current_user_id()
    if viewer_id is None:
        st.warning("Unable to determine your account. Please log in again.")
        return

    is_admin = user.get("role") == "admin"
    today = datetime.now().date()
    current_week_start = today - timedelta(days=today.weekday())
    current_week_end = current_week_start + timedelta(days=6)
    current_month_start = date(today.year, today.month, 1)
    current_month_end = date(
        today.year, today.month, monthrange(today.year, today.month)[1]
    )
    st.caption(
        "Staff can see only their own entries. Admins can review every team member's submissions."
    )

    directory = df_query(
        conn,
        "SELECT user_id, username, role FROM users ORDER BY LOWER(username)",
    )
    user_labels: dict[int, str] = {}
    if not directory.empty:
        for _, row in directory.iterrows():
            try:
                uid = int(row["user_id"])
            except Exception:
                continue
            username = clean_text(row.get("username")) or f"User #{uid}"
            role_label = clean_text(row.get("role"))
            if role_label == "admin":
                username = f"{username} (admin)"
            user_labels[uid] = username
    if viewer_id not in user_labels:
        user_labels[viewer_id] = clean_text(user.get("username")) or f"User #{viewer_id}"

    sorted_users = sorted(user_labels.items(), key=lambda item: item[1].lower())
    user_ids = [uid for uid, _ in sorted_users]
    label_map = {uid: label for uid, label in sorted_users}
    if not user_ids:
        user_ids = [viewer_id]
        label_map[viewer_id] = clean_text(user.get("username")) or f"User #{viewer_id}"

    if is_admin:
        default_index = user_ids.index(viewer_id) if viewer_id in user_ids else 0
        report_owner_id = st.selectbox(
            "Report owner",
            user_ids,
            index=default_index,
            format_func=lambda uid: label_map.get(uid, f"User #{uid}"),
            key="report_owner_select",
        )
    else:
        report_owner_id = viewer_id
        st.info(
            f"Recording progress for **{label_map.get(viewer_id, 'you')}**.",
            icon="📝",
        )
        st.caption(
            "Daily entries are limited to today. Weekly reports unlock on Saturdays, and monthly reports cover only the current month."
        )

    def _date_or(value, fallback: date) -> date:
        if value is None:
            return fallback
        parsed_iso = to_iso_date(value)
        if parsed_iso:
            try:
                return datetime.strptime(parsed_iso, "%Y-%m-%d").date()
            except ValueError:
                pass
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            return fallback
        if pd.isna(parsed):
            return fallback
        if isinstance(parsed, pd.DatetimeIndex):
            if len(parsed) == 0:
                return fallback
            parsed = parsed[0]
        return pd.Timestamp(parsed).date()

    def _staff_report_window_allows_edit(
        row,
        *,
        today: date,
        week_start: date,
        week_end: date,
        month_start: date,
        month_end: date,
    ) -> bool:
        period_key = clean_text(row.get("period_type")) or ""
        period_key = period_key.lower()
        row_start = _date_or(row.get("period_start"), today)
        row_end = _date_or(row.get("period_end"), row_start)
        if period_key == "daily":
            return row_start == today == row_end
        if period_key == "weekly":
            return (
                row_start == week_start
                and row_end == week_end
                and today.weekday() == 5
            )
        if period_key == "monthly":
            return row_start == month_start and row_end == month_end
        return False


    owner_reports = df_query(
        conn,
        dedent(
            """
            SELECT report_id, period_type, period_start, period_end, tasks, remarks, research, grid_payload, attachment_path, created_at, updated_at
            FROM work_reports
            WHERE user_id=?
            ORDER BY date(period_start) DESC, report_id DESC
            LIMIT 50
            """
        ),
        (report_owner_id,),
    )
    record_labels: dict[int, str] = {}
    selectable_reports = owner_reports.copy()
    if not owner_reports.empty:
        selectable_reports["report_id"] = selectable_reports["report_id"].apply(
            lambda val: int(float(val))
        )
        for _, row in selectable_reports.iterrows():
            rid = int(row["report_id"])
            record_labels[rid] = (
                f"{format_period_label(row.get('period_type'))} – "
                f"{format_period_range(row.get('period_start'), row.get('period_end'))}"
            )
        if not is_admin:
            selectable_reports["__staff_can_edit__"] = selectable_reports.apply(
                lambda row: _staff_report_window_allows_edit(
                    row,
                    today=today,
                    week_start=current_week_start,
                    week_end=current_week_end,
                    month_start=current_month_start,
                    month_end=current_month_end,
                ),
                axis=1,
            )
            selectable_reports = selectable_reports[
                selectable_reports["__staff_can_edit__"] == True  # noqa: E712
            ].copy()
            selectable_reports.drop(
                columns=["__staff_can_edit__"],
                inplace=True,
                errors="ignore",
            )

    selectable_ids: list[int] = []
    if not selectable_reports.empty and "report_id" in selectable_reports.columns:
        selectable_ids = (
            selectable_reports["report_id"].astype(int).tolist()
        )

    report_choices = [None] + selectable_ids

    def _format_report_choice(value):
        if value is None:
            return "➕ Create new report"
        try:
            return record_labels.get(int(value), f"Report #{int(value)}")
        except Exception:
            return "Report"

    if "report_edit_select_pending" in st.session_state:
        pending_selection = st.session_state.pop("report_edit_select_pending")
        if pending_selection is not None:
            st.session_state["report_edit_select"] = pending_selection

    selected_report_id = st.selectbox(
        "Load an existing report",
        report_choices,
        format_func=_format_report_choice,
        key="report_edit_select",
    )

    editing_record: Optional[dict] = None
    if selected_report_id is not None and not selectable_reports.empty:
        match = selectable_reports[
            selectable_reports["report_id"] == int(selected_report_id)
        ]
        if not match.empty:
            editing_record = match.iloc[0].to_dict()
    default_period_key = "daily"
    if editing_record:
        seed_period = clean_text(editing_record.get("period_type"))
        if seed_period:
            seed_period = seed_period.lower()
            if seed_period in REPORT_PERIOD_OPTIONS:
                default_period_key = seed_period
    period_keys = list(REPORT_PERIOD_OPTIONS.keys())
    if not is_admin:
        allowed_periods = ["daily"]
        if today.weekday() == 5 or default_period_key == "weekly":
            allowed_periods.append("weekly")
        allowed_periods.append("monthly")
        period_keys = [key for key in period_keys if key in allowed_periods]
    if not period_keys:
        period_keys = ["daily"]
    if default_period_key not in period_keys:
        default_period_key = period_keys[0]
    period_index = (
        period_keys.index(default_period_key)
        if default_period_key in period_keys
        else 0
    )

    default_start = _date_or(editing_record.get("period_start") if editing_record else None, today)
    default_end = _date_or(editing_record.get("period_end") if editing_record else None, default_start)
    legacy_tasks = clean_text(editing_record.get("tasks")) if editing_record else None
    legacy_remarks = clean_text(editing_record.get("remarks")) if editing_record else None
    legacy_research = clean_text(editing_record.get("research")) if editing_record else None

    start_date = default_start
    end_date = default_end

    existing_attachment_value: Optional[str] = (
        editing_record.get("attachment_path") if editing_record else None
    )

    grid_seed_rows = (
        parse_report_grid_payload(editing_record.get("grid_payload"))
        if editing_record
        else []
    )
    if not grid_seed_rows:
        fallback_row = _default_report_grid_row()
        if legacy_tasks:
            fallback_row["reported_complaints"] = legacy_tasks
        if legacy_remarks:
            fallback_row["details_remarks"] = legacy_remarks
        if legacy_research:
            fallback_row["product_details"] = legacy_research
        if any(val not in (None, "") for val in fallback_row.values()):
            grid_seed_rows = [fallback_row]
    existing_attachment_path = (
        resolve_upload_path(existing_attachment_value)
        if existing_attachment_value
        else None
    )
    existing_attachment_bytes: Optional[bytes] = None
    existing_attachment_name: Optional[str] = None
    if existing_attachment_path and existing_attachment_path.exists():
        existing_attachment_name = existing_attachment_path.name
        try:
            existing_attachment_bytes = existing_attachment_path.read_bytes()
        except OSError:
            existing_attachment_bytes = None

    with st.form("work_report_form"):
        period_choice = st.selectbox(
            "Report cadence",
            period_keys,
            index=period_index,
            format_func=lambda key: REPORT_PERIOD_OPTIONS.get(key, key.title()),
            key="report_period_type",
        )
        if period_choice == "daily":
            day_kwargs: dict[str, object] = {}
            if not is_admin:
                day_kwargs["min_value"] = today
                day_kwargs["max_value"] = today
            day_value = st.date_input(
                "Report date",
                value=default_start,
                key="report_period_daily",
                **day_kwargs,
            )
            start_date = day_value
            end_date = day_value
        elif period_choice == "weekly":
            base_start = default_start if editing_record else today - timedelta(days=today.weekday())
            base_end = default_end if editing_record else base_start + timedelta(days=6)
            week_kwargs: dict[str, object] = {}
            if not is_admin:
                week_kwargs["min_value"] = current_week_start
                week_kwargs["max_value"] = current_week_end
            week_value = st.date_input(
                "Week range",
                value=(base_start, base_end),
                key="report_period_weekly",
                **week_kwargs,
            )
            if isinstance(week_value, (list, tuple)) and len(week_value) == 2:
                start_date, end_date = week_value
            else:
                start_date = week_value
                end_date = week_value + timedelta(days=6)
            st.caption(
                f"Selected window: {format_period_range(to_iso_date(start_date), to_iso_date(end_date))}"
            )
        else:
            base_month = default_start if editing_record else today
            try:
                month_seed = base_month.replace(day=1)
            except Exception:
                month_seed = date(today.year, today.month, 1)
            month_kwargs: dict[str, object] = {}
            if not is_admin:
                month_kwargs["min_value"] = current_month_start
                month_kwargs["max_value"] = current_month_end
            month_value = st.date_input(
                "Month",
                value=month_seed,
                key="report_period_monthly",
                **month_kwargs,
            )
            if isinstance(month_value, (list, tuple)) and month_value:
                month_seed = month_value[0]
            else:
                month_seed = month_value
            if not isinstance(month_seed, date):
                month_seed = month_seed.to_pydatetime().date() if hasattr(month_seed, "to_pydatetime") else month_seed
            if not isinstance(month_seed, date):
                month_seed = date(today.year, today.month, 1)
            month_start = month_seed.replace(day=1)
            last_day = monthrange(month_start.year, month_start.month)[1]
            month_end = date(month_start.year, month_start.month, last_day)
            start_date, end_date = month_start, month_end
            st.caption(
                f"Selected window: {format_period_range(to_iso_date(start_date), to_iso_date(end_date))}"
            )

        st.caption(
            "Log service progress in a spreadsheet-style grid. Add rows for each customer or job completed."
        )
        seed_for_editor = grid_seed_rows or [_default_report_grid_row()]
        editor_seed = _grid_rows_for_editor(seed_for_editor)
        if not editor_seed:
            editor_seed = _grid_rows_for_editor([_default_report_grid_row()])
        grid_df_seed = pd.DataFrame(editor_seed, columns=REPORT_GRID_FIELDS.keys())
        column_config = {
            "customer_name": st.column_config.TextColumn(
                REPORT_GRID_FIELDS["customer_name"]["label"],
                help="Who received the service work.",
            ),
            "reported_complaints": st.column_config.TextColumn(
                REPORT_GRID_FIELDS["reported_complaints"]["label"],
                help="Issues raised by the customer.",
            ),
            "product_details": st.column_config.TextColumn(
                REPORT_GRID_FIELDS["product_details"]["label"],
                help="Model, serial, or generator description.",
            ),
            "details_remarks": st.column_config.TextColumn(
                REPORT_GRID_FIELDS["details_remarks"]["label"],
                help="Technician notes or actions taken.",
            ),
            "status": st.column_config.TextColumn(
                REPORT_GRID_FIELDS["status"]["label"],
                help="Current job status (e.g. Completed, Pending).",
            ),
            "quotation_tk": st.column_config.NumberColumn(
                REPORT_GRID_FIELDS["quotation_tk"]["label"],
                help="Quoted amount in Taka.",
                format="%.2f",
                step=100.0,
            ),
            "bill_tk": st.column_config.NumberColumn(
                REPORT_GRID_FIELDS["bill_tk"]["label"],
                help="Billed amount in Taka.",
                format="%.2f",
                step=100.0,
            ),
            "work_done_date": st.column_config.DateColumn(
                REPORT_GRID_FIELDS["work_done_date"]["label"],
                help="When the work was completed.",
                format="DD-MM-YYYY",
            ),
            "donation_cost": st.column_config.NumberColumn(
                REPORT_GRID_FIELDS["donation_cost"]["label"],
                help="Any donation or complimentary cost.",
                format="%.2f",
                step=100.0,
            ),
        }
        report_grid_df = st.data_editor(
            grid_df_seed,
            column_config=column_config,
            column_order=list(REPORT_GRID_FIELDS.keys()),
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            key="report_grid_editor",
        )
        remove_attachment = False
        attachment_upload = None
        if existing_attachment_value:
            st.caption("Current attachment")
            if existing_attachment_bytes and existing_attachment_name:
                st.download_button(
                    "Download current attachment",
                    data=existing_attachment_bytes,
                    file_name=existing_attachment_name,
                    key="report_attachment_download",
                )
            else:
                st.warning(
                    "The saved attachment could not be located on disk.",
                    icon="⚠️",
                )
            remove_attachment = st.checkbox(
                "Remove current attachment",
                key="report_remove_attachment",
            )
        attachment_upload = st.file_uploader(
            "Attach supporting document (PDF or image)",
            type=["pdf", "png", "jpg", "jpeg", "webp", "gif"],
            key="report_attachment_uploader",
            help="Optional proof of work, photos, or documentation.",
        )
        submitted = st.form_submit_button("Save report", type="primary")

    if submitted:
        cleanup_path: Optional[str] = None
        grid_rows_to_store = _grid_rows_from_editor(report_grid_df)
        tasks_summary = _summarize_grid_column(
            grid_rows_to_store, "reported_complaints"
        )
        remarks_summary = _summarize_grid_column(
            grid_rows_to_store, "details_remarks"
        )
        research_summary = _summarize_grid_column(
            grid_rows_to_store, "product_details"
        )
        try:
            normalized_key, normalized_start, normalized_end = normalize_report_window(
                period_choice, start_date, end_date
            )
        except ValueError as err:
            st.error(str(err))
        else:
            attachment_to_store = _ATTACHMENT_UNCHANGED
            attachment_save_failed = False
            save_allowed = True

            if not is_admin:
                validation_error: Optional[str] = None
                if normalized_key == "daily":
                    if normalized_start != today or normalized_end != today:
                        validation_error = "Daily reports can only be submitted for today."
                elif normalized_key == "weekly":
                    if today.weekday() != 5:
                        validation_error = "Weekly reports can only be submitted on Saturdays."
                    elif not (
                        normalized_start == current_week_start
                        and normalized_end == current_week_end
                    ):
                        validation_error = (
                            "Weekly reports must cover the current week (Monday to Sunday)."
                        )
                elif normalized_key == "monthly":
                    if not (
                        normalized_start == current_month_start
                        and normalized_end == current_month_end
                    ):
                        validation_error = "Monthly reports must cover the current month."
                if validation_error:
                    st.error(validation_error)
                    save_allowed = False

            if save_allowed and attachment_upload is not None:
                identifier = (
                    f"user{report_owner_id}_{normalized_key}_{normalized_start.isoformat()}"
                )
                stored_path = store_report_attachment(
                    attachment_upload,
                    identifier=identifier,
                )
                if stored_path:
                    attachment_to_store = stored_path
                    if existing_attachment_value:
                        cleanup_path = existing_attachment_value
                else:
                    st.error("Attachment could not be saved. Please try again.")
                    attachment_save_failed = True
            elif save_allowed and remove_attachment and existing_attachment_value:
                attachment_to_store = None
                cleanup_path = existing_attachment_value

            if save_allowed and not attachment_save_failed:
                if not grid_rows_to_store:
                    st.error("Add at least one row to the report grid before saving.")
                else:
                    try:
                        saved_id = upsert_work_report(
                            conn,
                            report_id=int(selected_report_id) if selected_report_id is not None else None,
                            user_id=int(report_owner_id),
                            period_type=normalized_key,
                            period_start=normalized_start,
                            period_end=normalized_end,
                            tasks=tasks_summary,
                            remarks=remarks_summary,
                            research=research_summary,
                            grid_rows=grid_rows_to_store,
                            attachment_path=attachment_to_store,
                            current_attachment=existing_attachment_value,
                        )
                    except ValueError as err:
                        st.error(str(err))
                    else:
                        st.success("Report saved successfully.")
                        if cleanup_path:
                            old_path = resolve_upload_path(cleanup_path)
                            if old_path and old_path.exists():
                                with contextlib.suppress(OSError):
                                    old_path.unlink()
                        st.session_state["report_edit_select_pending"] = saved_id
                        st.session_state.pop("report_attachment_uploader", None)
                        _safe_rerun()

    st.markdown("---")
    st.markdown("#### Report history")

    def _display_text(value: Optional[object]) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).strip()

    history_user = viewer_id
    if is_admin:
        history_options: list[Optional[int]] = [None] + user_ids

        def _history_label(uid: Optional[int]) -> str:
            if uid is None:
                return "All team members"
            return label_map.get(uid, f"User #{uid}")

        default_history_index = (
            history_options.index(report_owner_id)
            if report_owner_id in history_options
            else 0
        )
        history_user = st.selectbox(
            "Team member",
            history_options,
            index=default_history_index,
            format_func=_history_label,
            key="report_history_user",
        )

    period_keys = list(REPORT_PERIOD_OPTIONS.keys())
    history_periods = st.multiselect(
        "Cadence",
        period_keys,
        default=period_keys,
        format_func=lambda key: REPORT_PERIOD_OPTIONS.get(key, key.title()),
        key="report_history_periods",
    )

    default_history_start = today - timedelta(days=30)
    history_range = st.date_input(
        "Period range",
        value=(default_history_start, today),
        key="report_history_range",
    )
    range_start = range_end = None
    if isinstance(history_range, (list, tuple)) and len(history_range) == 2:
        range_start, range_end = history_range
    elif history_range:
        range_start = history_range
        range_end = history_range

    search_term = st.text_input(
        "Search notes",
        key="report_history_search",
        placeholder="Keyword in tasks, remarks, or research",
    )

    filters: list[str] = []
    params: list[object] = []
    if not is_admin or history_user is not None:
        target = history_user if history_user is not None else viewer_id
        filters.append("wr.user_id = ?")
        params.append(int(target))
    if history_periods and len(history_periods) != len(period_keys):
        placeholders = ",".join("?" for _ in history_periods)
        filters.append(f"wr.period_type IN ({placeholders})")
        params.extend(history_periods)
    if range_start:
        filters.append("date(wr.period_start) >= date(?)")
        params.append(to_iso_date(range_start))
    if range_end:
        filters.append("date(wr.period_end) <= date(?)")
        params.append(to_iso_date(range_end))
    if search_term:
        keyword = search_term.strip()
        if keyword:
            filters.append(
                "(wr.tasks LIKE '%'||?||'%' OR wr.remarks LIKE '%'||?||'%' OR wr.research LIKE '%'||?||'%')"
            )
            params.extend([keyword, keyword, keyword])

    where_clause = " AND ".join(filters) if filters else "1=1"
    history_df = df_query(
        conn,
        dedent(
            f"""
            SELECT wr.report_id, wr.user_id, wr.period_type, wr.period_start, wr.period_end,
                   wr.tasks, wr.remarks, wr.research, wr.grid_payload, wr.attachment_path, wr.created_at, wr.updated_at,
                   u.username
            FROM work_reports wr
            JOIN users u ON u.user_id = wr.user_id
            WHERE {where_clause}
            ORDER BY date(wr.period_start) DESC, wr.report_id DESC
            """
        ),
        tuple(params),
    )

    if history_df.empty:
        st.info("No reports found for the selected filters.")
        return

    history_df["report_id"] = history_df["report_id"].apply(lambda val: int(float(val)))
    history_df["username"] = history_df.apply(
        lambda row: clean_text(row.get("username")) or f"User #{int(row['user_id'])}",
        axis=1,
    )

    history_df["grid_rows"] = history_df.apply(
        lambda row: parse_report_grid_payload(row.get("grid_payload")),
        axis=1,
    )

    def _legacy_rows(row: pd.Series) -> list[dict[str, object]]:
        fallback = _default_report_grid_row()
        legacy_flag = False
        if clean_text(row.get("tasks")):
            fallback["reported_complaints"] = clean_text(row.get("tasks"))
            legacy_flag = True
        if clean_text(row.get("remarks")):
            fallback["details_remarks"] = clean_text(row.get("remarks"))
            legacy_flag = True
        if clean_text(row.get("research")):
            fallback["product_details"] = clean_text(row.get("research"))
            legacy_flag = True
        return [fallback] if legacy_flag else []

    history_df["grid_rows"] = history_df.apply(
        lambda row: row.get("grid_rows") or _legacy_rows(row),
        axis=1,
    )

    entry_records: list[dict[str, object]] = []
    download_records: list[dict[str, object]] = []
    for _, record in history_df.iterrows():
        owner = record.get("username") or f"User #{int(record.get('user_id'))}"
        cadence_label = format_period_label(record.get("period_type"))
        period_label = format_period_range(
            record.get("period_start"), record.get("period_end")
        )
        grid_rows = record.get("grid_rows") or []
        display_df = format_report_grid_rows_for_display(
            grid_rows, empty_ok=True
        )
        if display_df.empty:
            continue
        for entry in display_df.to_dict("records"):
            entry_records.append(entry)
            download_entry = {
                "Team member": owner,
                "Cadence": cadence_label,
                "Period": period_label,
            }
            download_entry.update(entry)
            download_records.append(download_entry)

    entry_table = pd.DataFrame(entry_records)
    if not entry_table.empty:
        for key, config in REPORT_GRID_FIELDS.items():
            label = config["label"]
            if label not in entry_table.columns:
                entry_table[label] = pd.NA
            if config["type"] == "number":
                entry_table[label] = pd.to_numeric(
                    entry_table[label], errors="coerce"
                )
            else:
                entry_table[label] = entry_table[label].fillna("")
        entry_table = entry_table.reindex(columns=REPORT_GRID_DISPLAY_COLUMNS)
        st.dataframe(entry_table, use_container_width=True)
    else:
        st.info(
            "No structured report entries are available for the selected filters."
        )

    download_df = pd.DataFrame(download_records)
    if not download_df.empty:
        desired_columns = [
            "Team member",
            "Cadence",
            "Period",
            *REPORT_GRID_DISPLAY_COLUMNS,
        ]
        download_df = download_df.reindex(columns=desired_columns, fill_value="")
    elif not entry_table.empty:
        download_df = entry_table.reindex(
            columns=REPORT_GRID_DISPLAY_COLUMNS, fill_value=""
        )
    if not download_df.empty:
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered reports",
            data=csv_data,
            file_name="work_reports.csv",
            mime="text/csv",
            key="reports_download",
        )

    cadence_summary = (
        history_df.assign(label=history_df["period_type"].apply(format_period_label))
        .groupby("label")["report_id"]
        .count()
        .sort_index()
    )
    if not cadence_summary.empty:
        st.markdown("##### Cadence summary")
        cols = st.columns(len(cadence_summary))
        for col, (label, count) in zip(cols, cadence_summary.items()):
            col.metric(label, int(count))

    if is_admin and history_user is None:
        coverage = (
            history_df.assign(
                label=history_df["period_type"].apply(format_period_label),
                owner=history_df["username"],
            )
            .pivot_table(
                index="owner",
                columns="label",
                values="report_id",
                aggfunc="count",
                fill_value=0,
            )
            .astype(int)
        )
        coverage.columns.name = None
        coverage = coverage.reset_index().rename(columns={"owner": "Team member"})
        st.markdown("##### Reports by team member")
        st.dataframe(coverage, use_container_width=True)

    detail_limit = min(len(history_df), 20)
    if detail_limit:
        st.markdown("##### Quick read")
        for _, row in history_df.head(detail_limit).iterrows():
            header = (
                f"{row['username']} – {format_period_label(row['period_type'])} "
                f"({format_period_range(row.get('period_start'), row.get('period_end'))})"
            )
            with st.expander(header, expanded=False):
                grid_df = format_report_grid_rows_for_display(
                    row.get("grid_rows"), empty_ok=True
                )
                if not grid_df.empty:
                    st.dataframe(grid_df, use_container_width=True)
                else:
                    st.write("No structured entries recorded for this report.")
                    legacy_blocks = [
                        ("Tasks completed", _display_text(row.get("tasks"))),
                        ("Remarks / blockers", _display_text(row.get("remarks"))),
                        ("Research / learnings", _display_text(row.get("research"))),
                    ]
                    for title, text in legacy_blocks:
                        if text:
                            st.markdown(f"**{title}**")
                            st.write(text)
                created_label = format_period_range(
                    row.get("created_at"), row.get("created_at")
                )
                updated_label = format_period_range(
                    row.get("updated_at"), row.get("updated_at")
                )
                st.caption(f"Logged on {created_label} • Last updated {updated_label}")

# ---------- Main ----------
def main():
    init_ui()
    conn = get_conn()
    init_schema(conn)
    login_box(conn)

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    user = st.session_state.user or {}
    role = user.get("role")
    with st.sidebar:
        if role == "admin":
            pages = [
                "Dashboard",
                "Customers",
                "Customer Summary",
                "Scraps",
                "Warranties",
                "Import",
                "Reports",
                "Duplicates",
                "Users (Admin)",
                "Maintenance and Service",
            ]
        else:
            pages = [
                "Dashboard",
                "Customers",
                "Customer Summary",
                "Warranties",
                "Import",
                "Reports",
                "Maintenance and Service",
            ]
        if st.session_state.page not in pages:
            st.session_state.page = pages[0]
        current_index = pages.index(st.session_state.page)
        page = st.radio("Navigate", pages, index=current_index, key="nav_page")
        st.session_state.page = page

    show_expiry_notifications(conn)

    if page == "Dashboard":
        dashboard(conn)
    elif page == "Customers":
        customers_page(conn)
    elif page == "Customer Summary":
        customer_summary_page(conn)
    elif page == "Scraps":
        scraps_page(conn)
    elif page == "Warranties":
        warranties_page(conn)
    elif page == "Import":
        import_page(conn)
    elif page == "Reports":
        reports_page(conn)
    elif page == "Duplicates":
        duplicates_page(conn)
    elif page == "Users (Admin)":
        users_admin_page(conn)
    elif page == "Maintenance and Service":
        service_maintenance_page(conn)

if _streamlit_runtime_active():
    main()
elif __name__ == "__main__":
    _bootstrap_streamlit_app()
