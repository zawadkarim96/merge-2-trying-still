import contextlib
import base64
import html
import http.server
import io
import json
import math
import threading
import time
from reportlab.lib.utils import ImageReader
import os
import re
import sqlite3
import hashlib
import uuid
import zipfile
from calendar import monthrange
from datetime import datetime, timedelta, date
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
import urllib.parse

from dotenv import load_dotenv
from textwrap import dedent
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from pypdf import PdfReader

from openpyxl import load_workbook


import streamlit as st
from streamlit.components.v1 import html as st_components_html
from collections import OrderedDict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
PROJECT_ROOT = Path(__file__).resolve().parent
DATE_FMT = "%d-%m-%Y"
CURRENCY_SYMBOL = os.getenv("APP_CURRENCY_SYMBOL", "৳")

UPLOADS_DIR = BASE_DIR / "uploads"
DELIVERY_ORDER_DIR = UPLOADS_DIR / "delivery_orders"
SERVICE_DOCS_DIR = UPLOADS_DIR / "service_documents"
MAINTENANCE_DOCS_DIR = UPLOADS_DIR / "maintenance_documents"
CUSTOMER_DOCS_DIR = UPLOADS_DIR / "customer_documents"
SERVICE_BILL_DIR = UPLOADS_DIR / "service_bills"
REPORT_DOCS_DIR = UPLOADS_DIR / "report_documents"
QUOTATION_RECEIPT_DIR = UPLOADS_DIR / "quotation_receipts"
QUOTATION_DOCS_DIR = UPLOADS_DIR / "quotation_documents"
DELIVERY_RECEIPT_DIR = UPLOADS_DIR / "delivery_receipts"
QUOTATION_EDITOR_PORT = int(os.getenv("QUOTATION_EDITOR_PORT", "8502"))

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

_quotation_editor_server: Optional[http.server.ThreadingHTTPServer] = None
_quotation_editor_thread: Optional[threading.Thread] = None


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


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower())


def _suggest_report_column_mapping(columns: Iterable[str]) -> dict[str, str]:
    """Suggest a mapping from target fields to uploaded columns."""

    header_map: dict[str, str] = {}
    for key, config in REPORT_GRID_FIELDS.items():
        header_map[_normalize_header(key)] = key
        header_map[_normalize_header(config["label"])] = key

    suggestions: dict[str, str] = {}
    for col in columns:
        normalized = _normalize_header(str(col))
        target = header_map.get(normalized)
        if target and target not in suggestions:
            suggestions[target] = col
    return suggestions


def _load_report_grid_dataframe(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Convert uploaded file content into a DataFrame for report imports."""

    name = filename.lower()
    if not file_bytes:
        return None
    buffer = io.BytesIO(file_bytes)
    try:
        if name.endswith(".csv"):
            return pd.read_csv(buffer)
        return pd.read_excel(buffer)
    except Exception:
        return None


def _import_report_grid_from_dataframe(
    raw_df: Optional[pd.DataFrame], column_mapping: Optional[dict[str, str]] = None
) -> list[dict[str, object]]:
    """Parse a DataFrame into report grid rows with optional custom mapping."""

    imported_rows: list[dict[str, object]] = []
    if raw_df is None or raw_df.empty:
        return imported_rows

    header_map: dict[str, str] = {}
    for key, config in REPORT_GRID_FIELDS.items():
        header_map[_normalize_header(key)] = key
        header_map[_normalize_header(config["label"])] = key

    resolved_mapping: dict[str, str] = {}
    if column_mapping:
        for source, target in column_mapping.items():
            if source in raw_df.columns and target in REPORT_GRID_FIELDS:
                resolved_mapping[source] = target

    if not resolved_mapping:
        for col in raw_df.columns:
            normalized = _normalize_header(str(col))
            target = header_map.get(normalized)
            if target:
                resolved_mapping[col] = target

    if not resolved_mapping:
        return imported_rows

    for _, row in raw_df.iterrows():
        entry = _default_report_grid_row()
        for source, target in resolved_mapping.items():
            value = row.get(source)
            config = REPORT_GRID_FIELDS.get(target, {})
            if config.get("type") == "number":
                entry[target] = _coerce_grid_number(value)
            elif config.get("type") == "date":
                entry[target] = to_iso_date(value)
            else:
                entry[target] = clean_text(value)
        if any(val not in (None, "") for val in entry.values()):
            imported_rows.append(entry)

    return imported_rows


def _import_report_grid_from_file(
    uploaded_file, column_mapping: Optional[dict[str, str]] = None
) -> list[dict[str, object]]:
    """Parse an uploaded spreadsheet into report grid rows.

    Supports CSV and Excel formats. Column headers are matched against
    ``REPORT_GRID_FIELDS`` labels and keys (case-insensitive). A custom
    ``column_mapping`` can be supplied to map uploaded columns to fields.
    """

    if uploaded_file is None:
        return []

    file_bytes = uploaded_file.getvalue()
    dataframe = _load_report_grid_dataframe(file_bytes, uploaded_file.name)
    return _import_report_grid_from_dataframe(dataframe, column_mapping)


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
    "quotation_created": "Quotation created",
    "quotation_updated": "Quotation updated",
    "warranty_updated": "Warranty updated",
    "report_submitted": "Report submitted",
    "report_updated": "Report updated",
    "delivery_order_created": "Delivery order saved",
    "delivery_order_updated": "Delivery order updated",
    "work_done_created": "Work done saved",
    "work_done_updated": "Work done updated",
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
    phone TEXT,
    email TEXT,
    title TEXT,
    role TEXT DEFAULT 'staff',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    company_name TEXT,
    phone TEXT,
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
    record_type TEXT DEFAULT 'delivery_order',
    status TEXT DEFAULT 'pending',
    payment_receipt_path TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
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
    delivery_address TEXT,
    phone TEXT,
    product_label TEXT,
    notes TEXT,
    amount_spent REAL,
    quantity INTEGER DEFAULT 1,
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
CREATE TABLE IF NOT EXISTS quotations (
    quotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference TEXT,
    quote_date TEXT,
    customer_name TEXT,
    customer_company TEXT,
    customer_address TEXT,
    customer_district TEXT,
    customer_contact TEXT,
    attention_name TEXT,
    attention_title TEXT,
    subject TEXT,
    salutation TEXT,
    introduction TEXT,
    closing TEXT,
    quote_type TEXT,
    total_amount REAL,
    discount_pct REAL,
    status TEXT DEFAULT 'pending',
    payment_receipt_path TEXT,
    follow_up_status TEXT,
    follow_up_notes TEXT,
    follow_up_date TEXT,
    reminder_label TEXT,
    letter_template TEXT,
    salesperson_name TEXT,
    salesperson_title TEXT,
    salesperson_contact TEXT,
    salesperson_email TEXT,
    document_path TEXT,
    remarks_internal TEXT,
    created_by INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(created_by) REFERENCES users(user_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_quotations_status ON quotations(status);
CREATE INDEX IF NOT EXISTS idx_quotations_owner ON quotations(created_by);
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
    add_column("users", "phone", "TEXT")
    add_column("users", "email", "TEXT")
    add_column("users", "title", "TEXT")
    add_column("services", "status", "TEXT DEFAULT 'In progress'")
    add_column("services", "service_start_date", "TEXT")
    add_column("services", "service_end_date", "TEXT")
    add_column("services", "service_product_info", "TEXT")
    add_column("services", "condition_status", "TEXT")
    add_column("services", "condition_remarks", "TEXT")
    add_column("services", "bill_amount", "REAL")
    add_column("services", "bill_document_path", "TEXT")
    add_column("services", "created_by", "INTEGER")
    add_column("maintenance_records", "status", "TEXT DEFAULT 'In progress'")
    add_column("services", "payment_status", "TEXT DEFAULT 'pending'")
    add_column("services", "payment_receipt_path", "TEXT")
    add_column("maintenance_records", "payment_status", "TEXT DEFAULT 'pending'")
    add_column("maintenance_records", "payment_receipt_path", "TEXT")
    add_column("quotations", "payment_receipt_path", "TEXT")
    add_column("maintenance_records", "maintenance_start_date", "TEXT")
    add_column("maintenance_records", "maintenance_end_date", "TEXT")
    add_column("maintenance_records", "maintenance_product_info", "TEXT")
    add_column("maintenance_records", "created_by", "INTEGER")
    add_column("quotations", "document_path", "TEXT")
    add_column("warranties", "remarks", "TEXT")
    add_column("delivery_orders", "remarks", "TEXT")
    add_column("delivery_orders", "items_payload", "TEXT")
    add_column("delivery_orders", "total_amount", "REAL")
    add_column("delivery_orders", "created_by", "INTEGER")
    add_column("delivery_orders", "record_type", "TEXT DEFAULT 'delivery_order'")
    add_column("delivery_orders", "status", "TEXT DEFAULT 'pending'")
    add_column("delivery_orders", "payment_receipt_path", "TEXT")
    add_column("delivery_orders", "updated_at", "TEXT DEFAULT (datetime('now'))")
    add_column("import_history", "amount_spent", "REAL")
    add_column("import_history", "imported_by", "INTEGER")
    add_column("import_history", "delivery_address", "TEXT")
    add_column("import_history", "quantity", "INTEGER DEFAULT 1")
    add_column("work_reports", "grid_payload", "TEXT")
    add_column("work_reports", "attachment_path", "TEXT")
    add_column("service_documents", "uploaded_by", "INTEGER")
    add_column("maintenance_documents", "uploaded_by", "INTEGER")
    add_column("quotations", "salesperson_email", "TEXT")

    # Remove stored email data for privacy; the app no longer collects it.
    if has_column("customers", "email"):
        conn.execute("UPDATE customers SET email=NULL WHERE email IS NOT NULL")

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
        CREATE TABLE IF NOT EXISTS dashboard_remarks (
            remark_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            note TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dashboard_remarks_created_at ON dashboard_remarks(created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dashboard_remarks_user ON dashboard_remarks(user_id)"
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


def filter_delivery_orders_for_view(
    do_df: pd.DataFrame,
    allowed_customers: Optional[set[int]],
    *,
    record_types: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Limit delivery order rows to records the current user can access.

    Admins (``allowed_customers`` is ``None``) see all rows. Staff can always
    see records linked to customers they own **or** records they personally
    created, even when the customer link is missing.
    """

    if do_df is None or do_df.empty or allowed_customers is None:
        return do_df

    viewer_id = current_user_id()

    def _allowed(row: pd.Series) -> bool:
        if record_types is not None:
            row_type = clean_text(row.get("record_type")) or "delivery_order"
            if row_type not in record_types:
                return False
        cust_id = row.get("customer_id")
        creator_id = row.get("created_by")

        try:
            if pd.notna(cust_id) and int(cust_id) in allowed_customers:
                return True
        except Exception:
            pass

        if viewer_id is not None:
            try:
                if pd.notna(creator_id) and int(creator_id) == int(viewer_id):
                    return True
            except Exception:
                pass

        return False

    return do_df[do_df.apply(_allowed, axis=1)]


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


def _extract_text_from_quotation_upload(upload) -> tuple[str, list[str]]:
    """Return extracted text and warnings from an uploaded quotation file."""

    warnings: list[str] = []
    text_content = ""
    suffix = Path(upload.name).suffix.lower()
    file_bytes = upload.getvalue()

    def _ocr_image(image: Image.Image) -> str:
        try:
            import pytesseract
        except Exception:  # pragma: no cover - optional dependency
            warnings.append(
                "Install pytesseract and Tesseract OCR to read text from images."
            )
            return ""

        try:
            grayscale = ImageOps.grayscale(image)
            boosted = ImageOps.autocontrast(grayscale)
            boosted = ImageEnhance.Contrast(boosted).enhance(1.8)
            boosted = ImageEnhance.Sharpness(boosted).enhance(1.2)
            primary_text = pytesseract.image_to_string(boosted)

            if len(primary_text.strip()) >= 12:
                return primary_text

            inverted = ImageOps.invert(grayscale)
            inverted = ImageEnhance.Contrast(inverted).enhance(1.6)
            alt_text = pytesseract.image_to_string(inverted)
            return alt_text if len(alt_text.strip()) > len(primary_text.strip()) else primary_text
        except Exception as exc:  # pragma: no cover - defensive against OCR failures
            warnings.append(f"OCR failed: {exc}")
            return ""

    if suffix == ".pdf":
        pages: list[str] = []
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                try:
                    text_block = page.extract_text() or ""
                except Exception:
                    text_block = ""
                image_ocr: list[str] = []
                for image_file in getattr(page, "images", []):
                    try:
                        pil_image = Image.open(io.BytesIO(image_file.data))
                        image_ocr.append(_ocr_image(pil_image))
                    except Exception:
                        continue
                if image_ocr:
                    text_block = "\n".join([text_block, *image_ocr]).strip()
                pages.append(text_block)
            text_content = "\n".join(pages)
            if not text_content.strip():
                warnings.append("No readable text found in the uploaded PDF.")
        except Exception as exc:  # pragma: no cover - defensive against damaged uploads
            warnings.append(f"Could not read PDF: {exc}")
        if len(text_content.strip()) < 80:
            try:
                from pdf2image import convert_from_bytes

                last_page = max(1, min(2, len(pages) or 2))
                raster_pages = convert_from_bytes(
                    file_bytes, dpi=300, first_page=1, last_page=last_page
                )
                extra_ocr = [_ocr_image(image) for image in raster_pages]
                combined = "\n".join([text_content, *extra_ocr]).strip()
                if combined:
                    text_content = combined
            except Exception as exc:  # pragma: no cover - optional dependency
                warnings.append(
                    "Add pdf2image with poppler to strengthen OCR for scanned PDFs"
                )
                if str(exc).strip():
                    warnings.append(f"OCR raster fallback failed: {exc}")
    else:
        try:
            image = Image.open(io.BytesIO(file_bytes))
        except Exception as exc:  # pragma: no cover - defensive against damaged uploads
            warnings.append(f"Could not open the uploaded image: {exc}")
            return "", warnings

        text_content = _ocr_image(image)

    return text_content, warnings


def _parse_date_from_text(value: str) -> Optional[date]:
    cleaned = clean_text(value)
    if not cleaned:
        return None
    cleaned = cleaned.replace(".", "-").replace("/", "-")
    candidates = re.findall(r"\d{1,2}-\d{1,2}-\d{2,4}", cleaned)
    if not candidates:
        return None
    for candidate in candidates:
        for fmt in ["%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d"]:
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    return None


def _parse_line_items_from_text(lines: list[str]) -> list[dict[str, object]]:
    """Extract probable line items from OCR'd inquiry text."""

    items: list[dict[str, object]] = []
    qty_pattern = re.compile(r"(?P<qty>\d+(?:\.\d+)?)\s*(pcs|nos|units|unit|qty|set|sets|pairs)?", re.IGNORECASE)
    rate_pattern = re.compile(r"(?:@|x|\b)\s*(?P<rate>\d{2,}(?:[\d,.]*\d)?)")
    model_pattern = re.compile(r"model[:\s]*([A-Za-z0-9-_.]+)", re.IGNORECASE)

    for raw_line in lines:
        line = raw_line.strip()
        if len(line.split()) < 2:
            continue

        qty_match = qty_pattern.search(line)
        model_match = model_pattern.search(line)
        rate_match = rate_pattern.search(line)

        qty_value = float(qty_match.group("qty")) if qty_match else 1.0
        rate_value = 0.0
        if rate_match:
            try:
                rate_value = float(rate_match.group("rate").replace(",", ""))
            except ValueError:
                rate_value = 0.0

        model_value = clean_text(model_match.group(1)) if model_match else None

        # Remove obvious numeric tokens when building description for clarity.
        cleaned_line = re.sub(r"\s{2,}", " ", line)
        description = cleaned_line
        if qty_match:
            description = description.replace(qty_match.group(0), "").strip(", -")
        if rate_match:
            description = description.replace(rate_match.group(0), "").strip(", -")
        if model_value:
            description = description.replace(model_value, "").strip(", -")
        description = description.strip() or cleaned_line

        if not description:
            continue

        items.append(
            {
                "description": description,
                "model": model_value or "",
                "quantity": qty_value,
                "rate": rate_value,
                "discount": 0.0,
            }
        )

    return items


def _extract_quotation_metadata(text: str) -> dict[str, object]:
    """Detect useful fields and probable items from uploaded quotation text."""

    if not text:
        return {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = "\n".join(lines)
    updates: dict[str, object] = {}

    def _match(patterns: Iterable[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _header_value(prefix: str) -> Optional[str]:
        for line in lines:
            if line.lower().startswith(prefix.lower()):
                candidate = line.split(":", 1)[-1].strip()
                return candidate or None
        return None

    def _collect_after(keyword: str, max_lines: int = 6) -> list[str]:
        for idx, line in enumerate(lines):
            if line.lower().startswith(keyword.lower()):
                collected: list[str] = []
                for entry in lines[idx + 1 : idx + 1 + max_lines]:
                    if not entry.strip():
                        break
                    collected.append(entry)
                return collected
        return []

    def _parse_amount(value: str) -> Optional[float]:
        match = re.search(r"([1-9]\d{1,2}(?:,\d{3})*(?:\.\d+)?)", value)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None

    def _extract_price_schedule() -> list[dict[str, object]]:
        """Capture the generator line item from the uploaded quotation."""

        description_lines: list[str] = []
        price_hint: Optional[float] = None
        quantity_hint: float = 1.0

        for idx, line in enumerate(lines):
            if re.search(r"brand\s+new\s+diesel\s+generating\s+set", line, re.IGNORECASE):
                # Collect nearby descriptive lines until we hit an obvious footer.
                block: list[str] = [line]
                for entry in lines[idx + 1 : min(len(lines), idx + 12)]:
                    if re.search(r"total\s+amount", entry, re.IGNORECASE):
                        break
                    block.append(entry)
                description_lines = [entry for entry in block if entry.strip()]
                joined_block = " ".join(description_lines)
                price_hint = _parse_amount(joined_block)
                qty_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:set|pcs|piece|unit)s?", joined_block, re.IGNORECASE)
                if qty_match:
                    try:
                        quantity_hint = float(qty_match.group(1))
                    except ValueError:
                        quantity_hint = 1.0
                break

        if not description_lines:
            return []

        cleaned_description = "; ".join(description_lines)
        items: list[dict[str, object]] = [
            {
                "description": cleaned_description,
                "model": "",
                "quantity": quantity_hint,
                "rate": price_hint or 0.0,
                "discount": 0.0,
            }
        ]
        return items

    reference = _header_value("ref") or _match(
        [
            r"quotation\s*(?:no\.|number|#)[:#\s]*([\w/-]+)",
            r"quote\s*(?:no\.|number|#)[:#\s]*([\w/-]+)",
            r"reference[:#\s]*([\w/-]+)",
            r"ref[:#\s]*([\w/-]+)",
            r"(?:inquiry|enquiry)\s*(?:no\.|number|#)[:#\s]*([\w/-]+)",
        ]
    )
    if reference:
        updates["quotation_reference"] = reference

    subject_line = _match([r"subject[:\s]*([^\n]+)", r"scope[:\s]*([^\n]+)"])
    if subject_line:
        updates["quotation_subject"] = subject_line

    attention = _header_value("attention") or _match(
        [
            r"attn\.?[:\s]*([^\n]+)",
            r"attention[:\s]*([^\n]+)",
            r"dear\s+([^\n,]+)",
            r"contact[:\s]*([^\n,]+)",
        ]
    )
    if attention:
        updates["quotation_attention_name"] = attention
        updates.setdefault("quotation_customer_contact_name", attention)

    company = _header_value("to") or _match(
        [
            r"company[:\s]*([^\n]+)",
            r"to[:\s]*([^\n]+)",
            r"for[:\s]*([^\n]+)",
            r"customer[:\s]*([^\n]+)",
        ]
    )
    if company:
        updates["quotation_company_name"] = company

    phone = _match([r"(?:phone|tel|mobile)[:\s]*([+\d][\d\s\-()]+)", r"(?:cell|contact)[:\s]*([+\d][\d\s\-()]+)"])
    if phone:
        updates["quotation_customer_contact"] = phone

    email = _match([r"email[:\s]*([^\n\s]+@[^\n\s]+)"])
    if email:
        updates["quotation_customer_email"] = email
        existing = updates.get("quotation_customer_contact")
        updates["quotation_customer_contact"] = (
            f"{existing}, {email}" if existing else email
        )

    address_block = _collect_after("to")
    address = _match([r"address[:\s]*([^\n]+(?:\n[^\n]+){0,3})", r"(?:office|site)\s*address[:\s]*([^\n]+)"])
    if address_block:
        updates["quotation_customer_address"] = "\n".join(address_block)
    elif address:
        updates["quotation_customer_address"] = address.replace("\n", " ")

    detected_date = _header_value("date") or _match(
        [
            r"date[:\s]*([\d./\-]+)",
            r"valid\s*(?:until|till|up to)[:\s]*([\d./\-]+)",
        ]
    )
    parsed_date = _parse_date_from_text(detected_date) if detected_date else None
    if parsed_date:
        updates["quotation_date"] = parsed_date

    total_matches = re.findall(
        r"(?:grand\s*total|total\s*amount|sub\s*total|total)[^\d]{0,10}([\d,.]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    parsed_totals: list[float] = []
    for raw in total_matches:
        cleaned = raw.replace(",", "")
        try:
            parsed_totals.append(float(cleaned))
        except ValueError:
            continue
    if parsed_totals:
        best_total = max(parsed_totals)
        updates["quotation_detected_total"] = best_total

    detected_items = _extract_price_schedule() or _parse_line_items_from_text(lines)
    if detected_items:
        updates["_detected_items"] = detected_items

    return updates


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


def _build_staff_alerts(conn, *, user_id: Optional[int]) -> list[dict[str, object]]:
    alerts: list[dict[str, object]] = []
    if user_id is None:
        return alerts

    today_iso = date.today().isoformat()
    follow_ups = df_query(
        conn,
        dedent(
            """
            SELECT reference, follow_up_date, reminder_label
            FROM quotations
            WHERE created_by = ?
              AND follow_up_date IS NOT NULL
              AND LOWER(IFNULL(status, 'pending')) <> 'paid'
            ORDER BY date(follow_up_date) ASC
            LIMIT 12
            """
        ),
        (user_id,),
    )
    if not follow_ups.empty:
        for _, row in follow_ups.iterrows():
            follow_date_val = clean_text(row.get("follow_up_date"))
            follow_dt = pd.to_datetime(follow_date_val, errors="coerce")
            severity = "info"
            if pd.notna(follow_dt) and follow_dt.date() <= date.today():
                severity = "warning"
            follow_label = format_period_range(follow_date_val, follow_date_val) or (follow_date_val or "(date pending)")
            alerts.append(
                {
                    "title": clean_text(row.get("reference")) or "Quotation follow-up",
                    "message": clean_text(row.get("reminder_label"))
                    or f"Follow-up scheduled for {follow_label}",
                    "severity": severity,
                }
            )

    pending_report = df_query(
        conn,
        dedent(
            """
            SELECT 1 FROM work_reports
            WHERE user_id=?
              AND period_type='daily'
              AND date(period_start)=date(?)
            LIMIT 1
            """
        ),
        (user_id, today_iso),
    )
    if pending_report.empty:
        alerts.append(
            {
                "title": "Daily report due",
                "message": f"Submit today's update ({format_period_range(today_iso, today_iso)}).",
                "severity": "warning",
            }
        )

    return alerts


def _fetch_entity_activity(
    conn,
    entity_types: Iterable[str],
    *,
    user_filter: Optional[int] = None,
    limit: int = 30,
) -> pd.DataFrame:
    types = [clean_text(t) for t in entity_types if clean_text(t)]
    if not types:
        return pd.DataFrame()

    filters = [f"a.entity_type IN ({','.join('?' for _ in types)})"]
    params: list[object] = list(types)
    try:
        resolved_limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        resolved_limit = 30

    if user_filter is not None:
        filters.append("a.user_id = ?")
        params.append(int(user_filter))

    params.append(resolved_limit)

    where_clause = " AND ".join(filters)

    return df_query(
        conn,
        dedent(
            f"""
            SELECT a.activity_id,
                   a.entity_type,
                   a.event_type,
                   a.description,
                   a.created_at,
                   a.user_id,
                   COALESCE(u.username, 'Team member') AS actor
            FROM activity_log a
            LEFT JOIN users u ON u.user_id = a.user_id
            WHERE {where_clause}
            ORDER BY datetime(a.created_at) DESC, a.activity_id DESC
            LIMIT ?
            """
        ),
        tuple(params),
    )


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
    actor_label = clean_text(get_current_user().get("username")) if get_current_user() else None
    label = NOTIFICATION_EVENT_LABELS.get(
        event_key, (event_key or "Activity").replace("_", " ").title()
    )
    should_notify = event_key not in {"login", "logout"}
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
    else:
        if should_notify:
            message = description_text or description or label or "Activity logged"
            details: list[str] = []
            if entity_type and entity_id:
                details.append(
                    f"{clean_text(entity_type).title()} #{entity_id}" if clean_text(entity_type) else f"Record #{entity_id}"
                )
            push_runtime_notification(
                label or "Activity logged",
                f"{actor_label or 'Team member'}: {message}",
                severity="info",
                details=details,
            )


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


def format_amount_in_words(value: object) -> Optional[str]:
    amount = parse_amount(value)
    if amount is None:
        return None

    taka = int(amount)
    paisa = int(round((amount - taka) * 100))

    below_twenty = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens_words = [
        "",
        "ten",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]
    scales = ["", "thousand", "million", "billion", "trillion"]

    def _three_digit_words(number: int) -> list[str]:
        words: list[str] = []
        hundreds, remainder = divmod(number, 100)
        if hundreds:
            words.extend([below_twenty[hundreds], "hundred"])
            if remainder:
                words.append("and")
        if remainder:
            if remainder < 20:
                words.append(below_twenty[remainder])
            else:
                tens, ones = divmod(remainder, 10)
                words.append(tens_words[tens])
                if ones:
                    words.append(below_twenty[ones])
        return words

    def _number_to_words(number: int) -> str:
        if number == 0:
            return "zero"
        words: list[str] = []
        scale_index = 0
        while number > 0 and scale_index < len(scales):
            number, remainder = divmod(number, 1000)
            if remainder:
                chunk_words = _three_digit_words(remainder)
                if scales[scale_index]:
                    chunk_words.append(scales[scale_index])
                words = chunk_words + words
            scale_index += 1
        return " ".join(words)

    parts = [f"{_number_to_words(taka)} taka"]
    if paisa:
        parts.append(f"{_number_to_words(paisa)} paisa")

    return (" and ".join(parts)).capitalize()


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, str):
        value = value.strip().replace(",", "")
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


def parse_quantity(value, *, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if not value:
            return default
    try:
        qty = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(qty) or qty <= 0:
        return default
    return max(1, int(round(qty)))


def _format_editor_date(value: object) -> Optional[str]:
    iso_date = to_iso_date(value)
    if not iso_date:
        return None
    try:
        return datetime.fromisoformat(iso_date).strftime(DATE_FMT)
    except Exception:
        return iso_date


def _fetch_quotation_for_editor(conn, quotation_id: int) -> Optional[dict[str, object]]:
    cursor = conn.execute(
        dedent(
            """
            SELECT quotation_id, reference, quote_date, customer_name, customer_company,
                   customer_address, customer_district, customer_contact, attention_name,
                   attention_title, subject, salutation, introduction, closing, total_amount,
                   discount_pct, document_path, letter_template, salesperson_name,
                   salesperson_title, salesperson_contact, quote_type
              FROM quotations
             WHERE quotation_id=?
            """
        ),
        (quotation_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    columns = [
        "quotation_id",
        "reference",
        "quote_date",
        "customer_name",
        "customer_company",
        "customer_address",
        "customer_district",
        "customer_contact",
        "attention_name",
        "attention_title",
        "subject",
        "salutation",
        "introduction",
        "closing",
        "total_amount",
        "discount_pct",
        "document_path",
        "letter_template",
        "salesperson_name",
        "salesperson_title",
        "salesperson_contact",
        "quote_type",
    ]
    return {col: row[idx] for idx, col in enumerate(columns)}


def _apply_editor_payload(conn, quotation_id: int, payload: Mapping[str, object]) -> bool:
    updates = {
        "reference": clean_text(payload.get("reference")),
        "quote_date": to_iso_date(payload.get("date")),
        "customer_company": clean_text(payload.get("customer_company")),
        "customer_name": clean_text(payload.get("customer_contact"))
        or clean_text(payload.get("attention")),
        "customer_address": clean_text(payload.get("address")),
        "attention_name": clean_text(payload.get("attention")),
        "subject": clean_text(payload.get("subject")),
        "salutation": clean_text(payload.get("salutation")),
        "introduction": clean_text(payload.get("introduction"))
        if payload.get("introduction")
        else None,
        "closing": clean_text(payload.get("closing")) if payload.get("closing") else None,
    }
    set_parts: list[str] = []
    values: list[object] = []
    for column, value in updates.items():
        set_parts.append(f"{column}=?")
        values.append(value)

    if not set_parts:
        return False

    values.append(quotation_id)
    cursor = conn.execute(
        f"UPDATE quotations SET {', '.join(set_parts)}, updated_at=datetime('now') WHERE quotation_id=?",
        tuple(values),
    )
    conn.commit()
    return cursor.rowcount > 0


def _build_editor_metadata(
    record: Mapping[str, object], payload: Mapping[str, object]
) -> OrderedDict:
    merged = dict(record)
    merged.update(
        {
            "quote_date": to_iso_date(payload.get("date")) or merged.get("quote_date"),
            "reference": clean_text(payload.get("reference")) or merged.get("reference"),
            "customer_company": clean_text(payload.get("customer_company"))
            or merged.get("customer_company"),
            "customer_address": payload.get("address") or merged.get("customer_address"),
            "customer_name": clean_text(payload.get("customer_contact"))
            or clean_text(payload.get("attention"))
            or merged.get("customer_name"),
            "attention_name": clean_text(payload.get("attention")) or merged.get("attention_name"),
            "subject": clean_text(payload.get("subject")) or merged.get("subject"),
            "salutation": clean_text(payload.get("salutation")) or merged.get("salutation"),
            "introduction": payload.get("introduction") or merged.get("introduction"),
            "closing": payload.get("closing") or merged.get("closing"),
        }
    )

    metadata = OrderedDict()
    metadata["Reference number"] = merged.get("reference")
    metadata["Date"] = _format_editor_date(merged.get("quote_date"))
    metadata["Customer contact name"] = merged.get("customer_name")
    metadata["Customer company"] = merged.get("customer_company")
    metadata["Customer address"] = merged.get("customer_address")
    metadata["Customer district"] = merged.get("customer_district")
    metadata["Customer contact"] = merged.get("customer_contact")
    metadata["Attention name"] = merged.get("attention_name")
    metadata["Attention title"] = merged.get("attention_title")
    metadata["Subject"] = merged.get("subject")
    metadata["Salutation"] = merged.get("salutation") or "Dear Sir,"
    metadata["Introduction"] = merged.get("introduction")
    metadata["Quote type"] = merged.get("quote_type") or "Quotation"
    metadata["Closing / thanks"] = merged.get("closing")
    metadata["Salesperson name"] = merged.get("salesperson_name")
    metadata["Salesperson title"] = merged.get("salesperson_title")
    metadata["Salesperson contact"] = merged.get("salesperson_contact")
    return metadata


def _generate_editor_pdf(
    quotation_id: int, payload: Mapping[str, object]
) -> Optional[bytes]:
    conn = get_conn()
    try:
        record = _fetch_quotation_for_editor(conn, quotation_id)
    finally:
        conn.close()

    if not record:
        return None

    metadata = _build_editor_metadata(record, payload)
    grand_total = _coerce_float(
        payload.get("total_amount") or record.get("total_amount"), 0.0
    )
    totals = {
        "grand_total": grand_total,
        "discount_total": 0.0,
        "gross_total": grand_total,
    }
    grand_total_label = format_money(grand_total) or f"{grand_total:,.2f}"
    grand_total_words = format_amount_in_words(grand_total)
    template_choice = clean_text(record.get("letter_template")) or "PS letterhead"

    return _build_quotation_pdf(
        metadata=metadata,
        items=[],
        totals=totals,
        grand_total_label=grand_total_label,
        template_choice=template_choice,
        grand_total_words=grand_total_words,
    )


class _QuotationEditorRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):  # type: ignore[override]
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args):  # pragma: no cover - reduce noise
        return

    def _parse_quotation_id(self, path_parts: list[str]) -> Optional[int]:
        if len(path_parts) < 3:
            return None
        try:
            quotation_id = int(path_parts[2])
        except (TypeError, ValueError):
            return None
        return quotation_id if quotation_id > 0 else None

    def _send_json(self, status: int, payload: Mapping[str, object]):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[dict[str, object]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        data = self.rfile.read(length) if length > 0 else b""
        if not data:
            return None
        try:
            parsed = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802 - following BaseHTTPRequestHandler signature
        parsed = urllib.parse.urlparse(self.path)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "api" and path_parts[1] == "quotation":
            quotation_id = self._parse_quotation_id(path_parts)
            if quotation_id is None:
                self._send_json(400, {"error": "Invalid quotation ID"})
                return

            conn = get_conn()
            try:
                record = _fetch_quotation_for_editor(conn, quotation_id)
            finally:
                conn.close()

            if not record:
                self._send_json(404, {"error": "Quotation not found"})
                return

            payload = {
                "quotation_id": quotation_id,
                "date": to_iso_date(record.get("quote_date")),
                "reference": clean_text(record.get("reference")) or "",
                "customer_company": clean_text(record.get("customer_company")) or "",
                "attention": clean_text(record.get("attention_name")) or "",
                "subject": clean_text(record.get("subject")) or "",
                "address": record.get("customer_address") or "",
                "salutation": clean_text(record.get("salutation")) or "Dear Sir,",
                "introduction": record.get("introduction") or "",
                "closing": record.get("closing") or "",
                "customer_contact": clean_text(record.get("customer_name")) or "",
            }
            self._send_json(200, payload)
            return

        return super().do_GET()

    def do_POST(self):  # noqa: N802 - following BaseHTTPRequestHandler signature
        parsed = urllib.parse.urlparse(self.path)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 3 and path_parts[0] == "api" and path_parts[1] == "quotation":
            quotation_id = self._parse_quotation_id(path_parts)
            if quotation_id is None:
                self._send_json(400, {"error": "Invalid quotation ID"})
                return

            payload = self._read_json() or {}
            if len(path_parts) >= 4 and path_parts[3] == "save":
                conn = get_conn()
                try:
                    updated = _apply_editor_payload(conn, quotation_id, payload)
                finally:
                    conn.close()

                if not updated:
                    self._send_json(404, {"error": "Quotation not found or unchanged"})
                    return
                self._send_json(200, {"status": "ok", "quotation_id": quotation_id})
                return

            if len(path_parts) >= 4 and path_parts[3] == "pdf":
                conn = get_conn()
                try:
                    _apply_editor_payload(conn, quotation_id, payload)
                finally:
                    conn.close()

                pdf_bytes = _generate_editor_pdf(quotation_id, payload)
                if not pdf_bytes:
                    self._send_json(404, {"error": "Quotation not found"})
                    return

                filename = f"quotation_{quotation_id}.pdf"
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header(
                    "Content-Disposition", f"attachment; filename={filename}"
                )
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(pdf_bytes)))
                self.end_headers()
                self.wfile.write(pdf_bytes)
                return

        self._send_json(404, {"error": "Unsupported endpoint"})


def _ensure_quotation_editor_server():
    global _quotation_editor_server, _quotation_editor_thread
    if _quotation_editor_server is not None:
        return

    try:
        handler_cls = partial(
            _QuotationEditorRequestHandler, directory=str(PROJECT_ROOT.resolve())
        )
        server = http.server.ThreadingHTTPServer(
            ("0.0.0.0", QUOTATION_EDITOR_PORT), handler_cls
        )
    except OSError:
        return

    _quotation_editor_server = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _quotation_editor_thread = thread


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
        price_raw = None
        if isinstance(entry, dict):
            price_raw = entry.get("unit_price") if "unit_price" in entry else entry.get("price")
        qty_val = _coerce_float(quantity_raw, 1.0)
        try:
            qty_val_int = int(round(qty_val))
        except Exception:
            qty_val_int = 1
        qty_val = max(qty_val_int, 1)
        unit_price = max(_coerce_float(price_raw, 0.0), 0.0)
        line_total = unit_price * qty_val if unit_price else None
        if not any([name_clean, model_clean, serial_clean]):
            continue
        cleaned.append(
            {
                "name": name_clean,
                "model": model_clean,
                "serial": serial_clean,
                "quantity": qty_val,
                "unit_price": unit_price if unit_price else None,
                "total": line_total if line_total else None,
            }
        )
        label_parts = [val for val in [name_clean, model_clean] if val]
        label = " - ".join(label_parts)
        if qty_val > 1:
            label = f"{label} ×{qty_val}" if label else f"×{qty_val}"
        if unit_price:
            price_block = f"Tk {unit_price:,.2f}"
            if line_total:
                price_block = f"{price_block} (Total Tk {line_total:,.2f})"
            label = f"{label} @ {price_block}" if label else price_block
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
        "grand_total": 0.0,
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        description = clean_text(entry.get("description") or entry.get("Description"))
        if not description:
            continue

        kva = clean_text(entry.get("kva"))
        model = clean_text(entry.get("model"))
        specs = clean_text(entry.get("specs")) or model
        note = clean_text(entry.get("note"))
        hsn = clean_text(entry.get("hsn"))
        unit = clean_text(entry.get("unit"))

        quantity = max(_coerce_float(entry.get("quantity"), 0.0), 0.0)
        rate = max(_coerce_float(entry.get("rate"), 0.0), 0.0)
        discount_pct = _clamp_percentage(_coerce_float(entry.get("discount"), 0.0))

        gross_amount = quantity * rate
        discount_amount = gross_amount * (discount_pct / 100.0)
        override_total = _coerce_float(entry.get("total_price"), None)
        line_total = (
            override_total
            if override_total is not None and override_total >= 0
            else max(gross_amount - discount_amount, 0.0)
        )

        description_label = dedupe_join([description, model], " – ") or description

        item = {
            "Sl No.": len(cleaned) + 1,
            "Description": description,
            "Description of Generator": description_label,
            "Quantity": quantity,
            "Qty.": quantity,
            "Unit": unit,
            "HSN/SAC": hsn,
            "KVA": kva,
            "Specs": specs,
            "Notes": note,
            "Rate": rate,
            "Unit Price, Tk.": rate,
            "Gross amount": gross_amount,
            "Discount (%)": discount_pct,
            "Discount amount": discount_amount,
            "Line total": line_total,
            "Total Price, Tk.": line_total,
        }

        # Remove optional empty fields to keep downstream tables tidy
        for optional_key in ["Unit", "HSN/SAC", "KVA", "Specs", "Notes"]:
            if not item.get(optional_key):
                item.pop(optional_key, None)

        cleaned.append(item)

        totals["gross_total"] += gross_amount
        totals["discount_total"] += discount_amount
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
    return [
        {
            "name": "",
            "model": "",
            "serial": "",
            "quantity": 1,
            "unit_price": 0.0,
        }
    ]


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
        "new_customer_create_delivery_order",
        "new_customer_create_work_done",
        "new_customer_work_done_number",
        "new_customer_work_done_notes",
        "new_customer_work_done_pdf",
        "new_customer_create_service",
        "new_customer_service_date",
        "new_customer_service_description",
        "new_customer_create_maintenance",
        "new_customer_maintenance_date",
        "new_customer_maintenance_description",
        "new_customer_products_table",
    ]:
        st.session_state.pop(key, None)


def _default_quotation_items() -> list[dict[str, object]]:
    return [
        {
            "description": "",
            "quantity": 1.0,
            "rate": 0.0,
            "total_price": 0.0,
        }
    ]


def _default_delivery_items() -> list[dict[str, object]]:
    return [
        {
            "description": "",
            "quantity": 1.0,
            "unit_price": 0.0,
            "discount": 0.0,
        }
    ]


def _products_to_delivery_items(products: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for entry in products or []:
        if not isinstance(entry, dict):
            continue
        name = clean_text(entry.get("name"))
        model = clean_text(entry.get("model"))
        description = " ".join(part for part in [name, model] if part).strip()
        if not description:
            continue
        items.append(
            {
                "description": description,
                "quantity": _coerce_float(entry.get("quantity"), 1.0),
                "unit_price": _coerce_float(entry.get("unit_price"), 0.0),
                "discount": 0.0,
            }
        )
    return items


def normalize_delivery_items(rows: Iterable[dict[str, object]]) -> tuple[list[dict[str, object]], float]:
    normalized: list[dict[str, object]] = []
    total_amount = 0.0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        description = clean_text(row.get("description"))
        if not description:
            continue
        quantity = _coerce_float(row.get("quantity"), 1.0)
        if quantity < 0:
            quantity = 0.0
        unit_price = _coerce_float(row.get("unit_price"), 0.0)
        if unit_price < 0:
            unit_price = 0.0
        discount = _coerce_float(row.get("discount"), 0.0)
        if discount < 0:
            discount = 0.0
        if discount > 100:
            discount = 100.0
        line_total = quantity * unit_price * (1 - (discount / 100))
        total_amount += line_total
        normalized.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "line_total": line_total,
            }
        )
    return normalized, total_amount


def parse_delivery_items_payload(value: Optional[str]) -> list[dict[str, object]]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []
    rows: list[dict[str, object]] = []
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "description": clean_text(entry.get("description")) or "",
                    "quantity": _coerce_float(entry.get("quantity"), 1.0),
                    "unit_price": _coerce_float(entry.get("unit_price"), 0.0),
                    "discount": _coerce_float(entry.get("discount"), 0.0),
                    "line_total": _coerce_float(entry.get("line_total"), 0.0),
                }
            )
    return rows


def _reset_quotation_form_state() -> None:
    default_items = _default_quotation_items()
    st.session_state["quotation_item_rows"] = default_items
    st.session_state["quotation_preview_item_rows"] = []
    st.session_state["quotation_preview_items_dirty"] = True
    st.session_state.pop("quotation_form_initialized", None)
    for key in [
        "quotation_reference",
        "quotation_date",
        "quotation_prepared_by",
        "quotation_valid_days",
        "quotation_company_name",
        "quotation_company_details",
        "quotation_customer_name",
        "quotation_customer_contact",
        "quotation_customer_address",
        "quotation_project_name",
        "quotation_subject",
        "quotation_scope_notes",
        "quotation_terms",
        "quotation_status",
        "quotation_follow_up_status",
        "quotation_follow_up_notes",
        "quotation_follow_up_date",
        "quotation_follow_up_date_toggle",
        "quotation_follow_up_choice",
        "quotation_salesperson_title",
        "quotation_salesperson_contact",
        "quotation_attention_name",
        "quotation_attention_title",
        "quotation_salutation",
        "quotation_introduction",
        "quotation_closing",
        "quotation_quote_type",
        "quotation_customer_district",
        "quotation_customer_district_select",
        "quotation_letter_template",
        "quotation_admin_notes",
        "quotation_reminder_label",
        "quotation_customer_contact_name",
        "quotation_receipt_upload",
        "quotation_document_path",
        "quotation_payment_receipt_path",
        "quotation_manual_total",
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
        QUOTATION_RECEIPT_DIR,
        QUOTATION_DOCS_DIR,
        DELIVERY_RECEIPT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(
    uploaded_file,
    target_dir: Path,
    filename: Optional[str] = None,
    *,
    allowed_extensions: Optional[set[str]] = None,
    default_extension: str = ".pdf",
) -> Optional[Path]:
    if uploaded_file is None:
        return None
    ensure_upload_dirs()
    raw_name = filename or uploaded_file.name or "upload"
    raw_name = "".join(ch for ch in raw_name if ch.isalnum() or ch in (".", "_", "-"))
    stem = Path(raw_name).stem or "upload"
    ext = Path(raw_name).suffix.lower()

    if allowed_extensions:
        normalized_allowed = {val.lower() for val in allowed_extensions}
        if ext not in normalized_allowed:
            ext = default_extension if default_extension.startswith(".") else f".{default_extension}"
        safe_name = f"{stem}{ext}"
    else:
        if ext != ".pdf":
            ext = default_extension if default_extension.startswith(".") else f".{default_extension}"
        safe_name = f"{stem}{ext}"
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


def store_payment_receipt(
    uploaded_file,
    *,
    identifier: Optional[str] = None,
    target_dir: Path = QUOTATION_RECEIPT_DIR,
) -> Optional[str]:
    """Persist an uploaded receipt (PDF or image) for paid records."""

    if uploaded_file is None:
        return None
    filename = identifier or uploaded_file.name or "receipt"
    saved_path = save_uploaded_file(
        uploaded_file,
        target_dir,
        filename=filename,
        allowed_extensions={".pdf", ".png", ".jpg", ".jpeg", ".webp"},
        default_extension=".pdf",
    )
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
    allowed_exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".xlsx", ".xls"}
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
    try:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    except Exception:
        cols = set()
    include_uploader = "uploaded_by" in cols
    uploader_id = current_user_id()
    for idx, uploaded in enumerate(files, start=1):
        if uploaded is None:
            continue
        original_name = uploaded.name or f"{prefix}_{idx}.pdf"
        safe_original = Path(original_name).name
        filename = f"{prefix}_{idx}_{safe_original}"
        stored_path = store_uploaded_pdf(uploaded, target_dir, filename=filename)
        if not stored_path:
            continue
        if include_uploader:
            conn.execute(
                f"INSERT INTO {table} ({fk_column}, file_path, original_name, uploaded_by) VALUES (?, ?, ?, ?)",
                (int(record_id), stored_path, safe_original, uploader_id),
            )
        else:
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


def dedupe_join(values: Iterable[Optional[str]], joiner: str = ", ") -> str:
    seen = []
    for value in values:
        if value is None:
            continue
        val = str(value).strip()
        if not val:
            continue
        if val not in seen:
            seen.append(val)
    return joiner.join(seen)


def join_with_counts(values: Iterable[Optional[str]]) -> str:
    """Join values with commas, annotating duplicates with counts."""

    counts: dict[str, int] = {}
    order: list[str] = []
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip()
        if not cleaned:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1
        if cleaned not in order:
            order.append(cleaned)

    parts: list[str] = []
    for item in order:
        count = counts.get(item, 0)
        if count > 1:
            parts.append(f"{item} (x{count})")
        else:
            parts.append(item)
    return ", ".join(parts)


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
        SELECT customer_id, name, company_name, phone, address, delivery_address, remarks, purchase_date, product_info, delivery_order_code, sales_person, created_at
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
    phone_values = [clean_text(v) for v in df.get("phone", pd.Series(dtype=object)).tolist()]
    phone_values = [v for v in phone_values if v]
    phones_to_recalc: set[str] = set(phone_values)

    base_name = clean_text(base_row.get("name")) or (name_values[0] if name_values else None)
    base_company = clean_text(base_row.get("company_name")) or (company_values[0] if company_values else None)
    base_address = clean_text(base_row.get("address")) or (address_values[0] if address_values else None)
    base_delivery_address = clean_text(base_row.get("delivery_address")) or (delivery_values[0] if delivery_values else None)
    combined_remarks = dedupe_join(remarks_values)
    base_phone = clean_text(base_row.get("phone")) or (phone_values[0] if phone_values else None)

    do_codes = []
    product_lines = []
    fallback_products = []
    purchase_dates = []
    purchase_labels = []
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
        if date_label:
            purchase_labels.append(date_label)
        if date_label and product_raw:
            product_lines.append(f"{date_label} – {product_raw}")
        elif product_raw:
            product_lines.append(product_raw)
        elif date_label:
            product_lines.append(date_label)
        if sales_raw:
            sales_people.append(sales_raw)

    earliest_purchase = min(purchase_dates).strftime("%Y-%m-%d") if purchase_dates else None
    combined_products = join_with_counts(product_lines or fallback_products)
    combined_do_codes = join_with_counts(do_codes)
    combined_sales = join_with_counts(sales_people)
    combined_purchase_labels = join_with_counts(purchase_labels)

    conn.execute(
        """
        UPDATE customers
        SET name=?, company_name=?, phone=?, address=?, delivery_address=?, remarks=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, dup_flag=0
        WHERE customer_id=?
        """,
        (
            base_name,
            base_company,
            base_phone,
            base_address,
            base_delivery_address,
            clean_text(combined_remarks),
            earliest_purchase,
            clean_text(combined_products or combined_purchase_labels),
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


def auto_merge_matching_customers(conn) -> bool:
    """Automatically merge customers sharing the same name and address."""

    df = df_query(
        conn,
        dedent(
            """
            SELECT customer_id, name, company_name, phone, address, delivery_address,
                   purchase_date, product_info, delivery_order_code, sales_person, remarks, created_at
            FROM customers
            WHERE TRIM(COALESCE(name, '')) <> '' AND TRIM(COALESCE(address, '')) <> ''
            """
        ),
    )
    if df.empty:
        return False

    def _normalize(value: object) -> str:
        cleaned = clean_text(value) or ""
        return " ".join(cleaned.lower().split())

    df["_name_norm"] = df.get("name", pd.Series(dtype=object)).apply(_normalize)
    df["_address_norm"] = df.get("address", pd.Series(dtype=object)).apply(_normalize)

    merged_any = False
    grouped = df.groupby(["_name_norm", "_address_norm"], dropna=False)
    for _, group in grouped:
        ids = [int(cid) for cid in group.get("customer_id", []) if int_or_none(cid) is not None]
        if len(ids) < 2:
            continue
        if merge_customer_records(conn, ids):
            merged_any = True

    return merged_any


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
    if "staff" in work.columns:
        work["staff_clean"] = work["staff"].apply(clean_text)
    elif "sales_person" in work.columns:
        work["staff_clean"] = work["sales_person"].apply(clean_text)
    else:
        work["staff_clean"] = None
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
                    "staff": dedupe_join(g.get("staff_clean", pd.Series(dtype=object)).tolist()),
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
            "staff": "Staff",
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
        WHERE COALESCE(d.record_type, 'delivery_order') = 'delivery_order'
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


def _build_quotations_export(conn) -> pd.DataFrame:
    scope_clause, scope_params = _quotation_scope_filter()
    query = dedent(
        f"""
        SELECT quotation_id,
               reference,
               quote_date,
               customer_company,
               customer_contact,
               customer_address,
               customer_district,
               attention_name,
               subject,
               total_amount,
               discount_pct,
               status,
               follow_up_status,
               follow_up_notes,
               follow_up_date,
               reminder_label,
               payment_receipt_path,
               salesperson_name,
               salesperson_title,
               salesperson_contact,
               remarks_internal,
               created_at,
               updated_at
        FROM quotations
        {scope_clause}
        ORDER BY datetime(quote_date) DESC, quotation_id DESC
        """
    )
    df = df_query(conn, query, scope_params)
    df = fmt_dates(df, ["quote_date", "follow_up_date", "created_at", "updated_at"])
    return df.rename(
        columns={
            "quotation_id": "Quotation ID",
            "reference": "Reference",
            "quote_date": "Quote date",
            "customer_company": "Customer",
            "customer_contact": "Contact",
            "customer_address": "Address",
            "customer_district": "District",
            "attention_name": "Attention",
            "subject": "Subject",
            "total_amount": "Total amount",
            "discount_pct": "Discount (%)",
            "status": "Status",
            "follow_up_status": "Follow-up status",
            "follow_up_notes": "Follow-up notes",
            "follow_up_date": "Follow-up date",
            "reminder_label": "Reminder",
            "payment_receipt_path": "Receipt path",
            "salesperson_name": "Salesperson",
            "salesperson_title": "Salesperson title",
            "salesperson_contact": "Salesperson contact",
            "remarks_internal": "Internal remarks",
            "created_at": "Created at",
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
        for sheet_name, df in ordered_sheets:
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


def export_full_archive(
    conn: Optional[sqlite3.Connection] = None, excel_bytes: Optional[bytes] = None
) -> bytes:
    """Bundle all user data, uploads, and database exports into one archive.

    The archive now includes:
    - The live SQLite database file.
    - A full SQL dump for recovery.
    - The Excel export of every table.
    - Every file under the application storage directory (uploads, receipts, etc.).
    """

    buffer = io.BytesIO()
    close_conn = False
    active_conn = conn
    if active_conn is None:
        active_conn = sqlite3.connect(DB_PATH)
        close_conn = True

    try:
        dump_buffer = io.StringIO()
        for line in active_conn.iterdump():
            dump_buffer.write(f"{line}\n")
        dump_bytes = dump_buffer.getvalue().encode("utf-8")

        if excel_bytes is None:
            excel_bytes = export_database_to_excel(active_conn)

        db_path = Path(DB_PATH)

        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            if db_path.exists():
                zf.write(db_path, arcname=f"database/{db_path.name}")

            if dump_bytes:
                zf.writestr("exports/ps_crm.sql", dump_bytes)

            if excel_bytes:
                zf.writestr(
                    "exports/ps_crm.xlsx",
                    excel_bytes,
                )

            if BASE_DIR.exists():
                for path in BASE_DIR.rglob("*"):
                    if path.is_file():
                        if db_path.exists() and path.resolve() == db_path.resolve():
                            continue
                        zf.write(path, arcname=str(Path("storage") / path.relative_to(BASE_DIR)))
    finally:
        if close_conn:
            active_conn.close()

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
        SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date, w.remarks,
               COALESCE(c.sales_person, u.username) AS staff
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        LEFT JOIN users u ON u.user_id = c.created_by
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
        "staff": "Staff",
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
    st.set_page_config(page_title="PS Business Suites", page_icon="🧰", layout="wide")
    st.title("PS Engineering – Business Suites")
    st.caption("Customers • Warranties • Needs • Summaries")
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border-radius: 0.8rem;
            padding: 0.85rem;
            border: 1px solid rgba(250, 250, 250, 0.12);
        }
        @media (prefers-color-scheme: light) {
            [data-testid="stMetric"] {
                background: #f7f9fc;
                border: 1px solid rgba(49, 51, 63, 0.08);
            }
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
def login_box(conn, *, render_id=None):
    logout_button_key = f"sidebar_logout_main_{render_id or st.session_state.get('_render_id', 0)}"
    st.sidebar.markdown("### Login")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as {st.session_state.user['username']} ({st.session_state.user['role']})")
        if st.sidebar.button("Logout", key=logout_button_key):
            st.session_state.user = None
            st.session_state.page = "Dashboard"
            _safe_rerun()
        return True
    with st.sidebar.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")
    if ok:
        row = df_query(
            conn,
            "SELECT user_id, username, pass_hash, role, phone, title, email FROM users WHERE username = ?",
            (u,),
        )
        if not row.empty and hashlib.sha256(p.encode("utf-8")).hexdigest() == row.iloc[0]["pass_hash"]:
            st.session_state.user = {
                "user_id": int(row.iloc[0]["user_id"]),
                "username": row.iloc[0]["username"],
                "role": row.iloc[0]["role"],
                "phone": clean_text(row.iloc[0].get("phone")),
                "title": clean_text(row.iloc[0].get("title")),
                "email": clean_text(row.iloc[0].get("email")),
            }
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


def _render_admin_record_history(conn):
    user = get_current_user()
    if clean_text(user.get("role")) != "admin":
        return

    st.markdown("#### Team record history")
    st.caption(
        "Review recent submissions across reports, delivery orders, work done, and maintenance/service."
    )

    users_df = df_query(
        conn,
        "SELECT user_id, username FROM users ORDER BY LOWER(COALESCE(username, 'user'))",
    )
    member_options: list[Optional[int]] = [None]
    member_labels: dict[Optional[int], str] = {None: "All team members"}
    if not users_df.empty:
        for _, row in users_df.iterrows():
            try:
                uid = int(row.get("user_id"))
            except Exception:
                continue
            label = clean_text(row.get("username")) or f"User #{uid}"
            member_options.append(uid)
            member_labels[uid] = label

    selected_user = st.selectbox(
        "Filter by team member",
        member_options,
        format_func=lambda uid: member_labels.get(uid, "All team members"),
        key="dashboard_record_history_user",
    )

    tabs = st.tabs(
        ["Reports", "Delivery orders", "Work done", "Maintenance & Service"]
    )

    tab_map: list[tuple[Any, tuple[str, ...], str]] = [
        (tabs[0], ("report",), "No reports recorded yet."),
        (tabs[1], ("delivery_order",), "No delivery orders recorded yet."),
        (tabs[2], ("work_done",), "No work done entries recorded yet."),
        (
            tabs[3],
            ("service", "maintenance"),
            "No maintenance or service updates recorded yet.",
        ),
    ]

    for tab, entities, empty_message in tab_map:
        with tab:
            history_df = _fetch_entity_activity(
                conn, entities, user_filter=selected_user, limit=50
            )
            if history_df.empty:
                st.info(empty_message)
                continue

            history_df["When"] = history_df["created_at"].apply(
                lambda value: format_time_ago(value)
                or format_period_range(value, value)
            )
            history_df["Record type"] = history_df["entity_type"].apply(
                lambda val: (clean_text(val) or "Record").replace("_", " ").title()
            )
            display_df = history_df.rename(
                columns={"actor": "Team member", "description": "Details"}
            )
            st.dataframe(
                display_df[
                    ["Team member", "Record type", "Details", "When"]
                ],
                use_container_width=True,
                hide_index=True,
            )


def _render_admin_kpi_panel(conn) -> None:
    user = get_current_user()
    if clean_text(user.get("role")) != "admin":
        return

    staff_df = df_query(
        conn,
        dedent(
            """
            SELECT user_id, COALESCE(username, 'User #' || user_id) AS username
            FROM users
            WHERE LOWER(COALESCE(role, 'staff')) <> 'admin'
            ORDER BY LOWER(username)
            """
        ),
    )

    if staff_df.empty:
        st.caption("No staff accounts available for KPI tracking yet.")
        return

    today = date.today()
    start_month = today.replace(day=1)
    days_elapsed = max((today - start_month).days + 1, 1)

    monthly_df = df_query(
        conn,
        dedent(
            """
            SELECT user_id,
                   COUNT(*) AS monthly_reports,
                   MAX(date(period_start)) AS last_report_date
            FROM work_reports
            WHERE period_type='daily'
              AND strftime('%Y-%m', period_start) = strftime('%Y-%m', 'now')
            GROUP BY user_id
            """
        ),
    ).rename(columns={"last_report_date": "last_report_month"})

    lifetime_df = df_query(
        conn,
        dedent(
            """
            SELECT user_id,
                   COUNT(*) AS total_reports,
                   MIN(date(period_start)) AS first_report_date,
                   MAX(date(period_start)) AS last_report_date
            FROM work_reports
            WHERE period_type='daily'
            GROUP BY user_id
            """
        ),
    ).rename(columns={"last_report_date": "last_report_all"})

    def _parse_date(value: object) -> Optional[date]:
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            return None

        if pd.isna(parsed):
            return None

        try:
            return parsed.date()
        except AttributeError:
            return None

    staff_df["user_id"] = staff_df["user_id"].apply(lambda val: int(float(val)))
    staff_df = staff_df.rename(columns={"username": "Team member"})

    merged = (
        staff_df.merge(monthly_df, on="user_id", how="left")
        .merge(lifetime_df, on="user_id", how="left")
    )

    merged[["monthly_reports", "total_reports"]] = merged[
        ["monthly_reports", "total_reports"]
    ].fillna(0)

    kpi_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        monthly_reports = int(_coerce_float(row.get("monthly_reports"), 0.0))
        total_reports = int(_coerce_float(row.get("total_reports"), 0.0))
        last_seen = _parse_date(row.get("last_report_month")) or _parse_date(
            row.get("last_report_all")
        )
        first_seen = _parse_date(row.get("first_report_date"))

        days_since_last = (today - last_seen).days if last_seen else days_elapsed + 7
        months_active = (
            (today.year - first_seen.year) * 12 + today.month - first_seen.month + 1
            if first_seen
            else 1
        )
        months_active = max(months_active, 1)

        monthly_completion = (monthly_reports / days_elapsed) * 100
        recency_penalty = min(days_since_last * 3.0, 60.0)
        momentum_boost = min((total_reports / months_active) * 2.5, 25.0)
        monthly_score = max(
            0.0,
            min(100.0, monthly_completion - recency_penalty + 20.0 + momentum_boost),
        )

        lifetime_velocity = (total_reports / months_active) / 20.0
        lifetime_score = max(
            0.0,
            min(
                100.0,
                (lifetime_velocity * 100.0) - min(days_since_last * 1.5, 40.0) + 20.0,
            ),
        )

        kpi_rows.append(
            {
                "Team member": clean_text(row.get("Team member"))
                or f"User #{int(row.get('user_id'))}",
                "Monthly KPI": f"{monthly_score:,.0f}/100",
                "Lifetime KPI": f"{lifetime_score:,.0f}/100",
            }
        )

    kpi_table = pd.DataFrame(kpi_rows)
    if kpi_table.empty:
        st.caption("KPI scores will appear after the team logs their first daily reports.")
        return

    kpi_table = kpi_table[["Team member", "Monthly KPI", "Lifetime KPI"]]

    st.markdown("##### 🛰️ Admin KPI snapshot")
    st.caption("Simplified lifetime and monthly scores for quick admin review.")
    st.dataframe(
        kpi_table,
        use_container_width=True,
        hide_index=True,
    )


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
    quick_links = st.columns(3)
    if quick_links[0].button("Work done", key="dashboard_work_done_link"):
        st.session_state.page = "Work done"
        st.session_state["nav_page"] = "Work done"
        _safe_rerun()
    if quick_links[1].button("Delivery orders", key="dashboard_delivery_orders_link"):
        st.session_state.page = "Delivery Orders"
        st.session_state["nav_page"] = "Delivery Orders"
        _safe_rerun()
    if quick_links[2].button("Quotation", key="dashboard_create_quotation_link"):
        st.session_state.page = "Quotation"
        st.session_state["nav_page"] = "Quotation"
        _safe_rerun()
    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"
    allowed_customers = accessible_customer_ids(conn)
    scope_clause, scope_params = customer_scope_filter("c")

    announcement = df_query(
        conn,
        dedent(
            """
            SELECT dr.remark_id, dr.note, dr.created_at, COALESCE(u.username, 'User') AS author
            FROM dashboard_remarks dr
            LEFT JOIN users u ON u.user_id = dr.user_id
            ORDER BY datetime(dr.created_at) DESC
            LIMIT 1
            """
        ),
    )
    current_announcement = announcement.iloc[0] if not announcement.empty else None

    st.markdown("#### Admin announcement for all staff")
    if current_announcement is None:
        st.caption("No announcement is currently set.")
    else:
        st.info(
            f"**{current_announcement['note']}**\n\n"
            f"Posted by {current_announcement['author']} on "
            f"{fmt_dates(announcement, ['created_at']).iloc[0]['created_at']}",
            icon="📢",
        )

    if is_admin:
        if st.session_state.pop("dashboard_remark_reset", False):
            st.session_state["dashboard_remark_text"] = ""
        st.session_state.setdefault(
            "dashboard_remark_text",
            clean_text(current_announcement["note"]) if current_announcement is not None else "",
        )

        st.markdown("##### Update or clear the announcement")
        with st.form("dashboard_remark_form"):
            remark_text = st.text_area(
                "Message visible to all staff",
                help="Admins can share reminders or updates. The latest message stays pinned until replaced or deleted.",
                key="dashboard_remark_text",
            )
            submit_remark = st.form_submit_button("Save announcement", type="primary")

        if submit_remark:
            cleaned_note = clean_text(remark_text)
            if not cleaned_note:
                st.warning("Please enter a message before saving.")
            else:
                conn.execute(
                    "INSERT INTO dashboard_remarks (user_id, note) VALUES (?, ?)",
                    (current_user_id(), cleaned_note),
                )
                conn.commit()
                st.success("Announcement saved for all staff.")
                st.session_state["dashboard_remark_reset"] = True
                _safe_rerun()

        if current_announcement is not None:
            if st.button("Delete current announcement", type="secondary"):
                conn.execute(
                    "DELETE FROM dashboard_remarks WHERE remark_id = ?",
                    (int(current_announcement["remark_id"]),),
                )
                conn.commit()
                st.success("Announcement removed.")
                st.session_state["dashboard_remark_reset"] = True
                _safe_rerun()

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
                    submitted_label = (
                        format_time_ago(record.get("created_at"))
                        or format_period_range(record.get("created_at"), record.get("created_at"))
                    )
                    updated_label = (
                        format_time_ago(record.get("updated_at"))
                        or format_period_range(record.get("updated_at"), record.get("updated_at"))
                    )
                    st.markdown("**Tasks completed**")
                    st.write(clean_text(record.get("tasks")) or "—")
                    st.markdown("**Remarks / blockers**")
                    st.write(clean_text(record.get("remarks")) or "—")
                    st.markdown("**Research / learnings**")
                    st.write(clean_text(record.get("research")) or "—")
                    summary_lines = [
                        f"Daily submission for {selected_staff_name}",
                        f"Date: {format_period_range(record.get('period_start'), record.get('period_end'))}",
                        "",
                        "Tasks completed:",
                        clean_text(record.get("tasks")) or "—",
                        "",
                        "Remarks / blockers:",
                        clean_text(record.get("remarks")) or "—",
                        "",
                        "Research / learnings:",
                        clean_text(record.get("research")) or "—",
                        "",
                        f"Submitted: {submitted_label}",
                        f"Last updated: {updated_label}",
                    ]
                    summary_payload = "\n".join(summary_lines)
                    download_name = _sanitize_path_component(
                        f"daily_submission_{selected_staff_name}_{report_iso}"
                    )
                    st.download_button(
                        "Download daily submission",
                        data=summary_payload.encode("utf-8"),
                        file_name=f"{download_name or 'daily_submission'}.txt",
                        mime="text/plain",
                        key=f"dashboard_daily_submission_{record.get('report_id')}",
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

    report_scope = ""
    report_params: tuple[object, ...] = ()
    viewer_id = current_user_id()
    if not is_admin:
        if viewer_id is not None:
            report_scope = "WHERE wr.user_id = ?"
            report_params = (viewer_id,)
        else:
            report_scope = "WHERE 1=0"

    recent_reports = df_query(
        conn,
        dedent(
            f"""
            SELECT wr.report_id,
                   wr.period_type,
                   wr.period_start,
                   wr.period_end,
                   wr.created_at,
                   COALESCE(u.username, 'User #' || wr.user_id) AS owner
            FROM work_reports wr
            LEFT JOIN users u ON u.user_id = wr.user_id
            {report_scope}
            ORDER BY datetime(wr.created_at) DESC, wr.report_id DESC
            LIMIT 6
            """
        ),
        report_params,
    )

    if not recent_reports.empty:
        st.markdown("#### Recent report submissions")
        recent_reports["Period"] = recent_reports.apply(
            lambda row: format_period_range(row.get("period_start"), row.get("period_end")),
            axis=1,
        )
        recent_reports["Cadence"] = recent_reports["period_type"].apply(
            lambda val: REPORT_PERIOD_OPTIONS.get(clean_text(val) or "", str(val).title())
        )
        recent_reports["When"] = recent_reports["created_at"].apply(
            lambda value: format_time_ago(value) or format_period_range(value, value)
        )
        display_cols = ["Team member", "Cadence", "Period", "When"]
        recent_display = recent_reports.rename(columns={"owner": "Team member"})
        st.dataframe(
            recent_display[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    uploads_scope = report_scope.replace("WHERE", "AND", 1) if report_scope else ""
    uploads_params = tuple(report_params)
    uploads_df = df_query(
        conn,
        dedent(
            f"""
            SELECT wr.report_id,
                   wr.attachment_path,
                   wr.period_start,
                   wr.period_end,
                   COALESCE(u.username, 'User #' || wr.user_id) AS owner,
                   wr.created_at
            FROM work_reports wr
            LEFT JOIN users u ON u.user_id = wr.user_id
            WHERE wr.attachment_path IS NOT NULL AND wr.attachment_path != ''
            {uploads_scope}
            ORDER BY datetime(wr.created_at) DESC
            LIMIT 5
            """
        ),
        uploads_params,
    )

    if not uploads_df.empty:
        st.markdown("#### Latest report uploads")
        uploads_df["Period"] = uploads_df.apply(
            lambda row: format_period_range(row.get("period_start"), row.get("period_end")),
            axis=1,
        )
        uploads_df["When"] = uploads_df["created_at"].apply(
            lambda value: format_time_ago(value) or format_period_range(value, value)
        )
        for _, row in uploads_df.iterrows():
            attachment_value = clean_text(row.get("attachment_path"))
            if not attachment_value:
                continue
            path = resolve_upload_path(attachment_value)
            if not path or not path.exists():
                continue
            try:
                payload = path.read_bytes()
            except OSError:
                continue
            label = f"{row.get('owner')} • {row.get('Period')}"
            st.download_button(
                label,
                data=payload,
                file_name=path.name,
                key=f"recent_attachment_{row.get('report_id')}",
            )

    if is_admin:
        _render_admin_record_history(conn)

        staff_quotes = df_query(
            conn,
            dedent(
                """
                SELECT q.quotation_id,
                       q.reference,
                       q.customer_company,
                       q.total_amount,
                       q.status,
                       q.quote_date,
                       q.document_path,
                       q.salesperson_name,
                       u.username
                FROM quotations q
                LEFT JOIN users u ON u.user_id = q.created_by
                ORDER BY datetime(q.quote_date) DESC, q.quotation_id DESC
                LIMIT 8
                """
            ),
        )
        if not staff_quotes.empty:
            staff_quotes = fmt_dates(staff_quotes, ["quote_date"])
            st.markdown("#### Team quotations (downloadable)")
            with st.container():
                st.markdown(
                    "<div style='max-height: 320px; overflow-y: auto;'>",
                    unsafe_allow_html=True,
                )
                for _, row in staff_quotes.iterrows():
                    cols = st.columns((1.5, 1, 1, 1))
                    cols[0].markdown(
                        f"**{clean_text(row.get('reference')) or 'Quotation'}**\n"
                        f"{clean_text(row.get('customer_company')) or '(customer)'}"
                    )
                    cols[1].write(
                        clean_text(row.get("salesperson_name"))
                        or clean_text(row.get("username"))
                        or "—"
                    )
                    total_value = format_money(row.get("total_amount")) or f"{_coerce_float(row.get('total_amount'), 0.0):,.2f}"
                    cols[2].write(
                        f"{clean_text(row.get('status')).title() if row.get('status') else 'Pending'}\n"
                        f"{row.get('quote_date')}\n{total_value}"
                    )

                    download_key = f"dash_quote_{int(row.get('quotation_id'))}"
                    doc_path = clean_text(row.get("document_path"))
                    file_path = BASE_DIR / doc_path if doc_path else None

                    # Prefer a PDF version for dashboard downloads; fall back to any
                    # matching PDF file for the quotation ID if the stored path is
                    # missing or points to a non-PDF asset. If only an Excel workbook
                    # exists, regenerate the PDF from the workbook contents and persist
                    # it for future downloads.
                    pdf_candidate = None
                    regenerated_bytes: Optional[bytes] = None
                    if file_path and file_path.suffix.lower() != ".pdf":
                        alt_pdf = file_path.with_suffix(".pdf")
                        if alt_pdf.exists():
                            pdf_candidate = alt_pdf
                        elif file_path.exists():
                            regenerated_bytes = _regenerate_quotation_pdf_from_workbook(
                                file_path
                            )
                    if pdf_candidate is None and regenerated_bytes is None:
                        try:
                            quote_id = int(row.get("quotation_id"))
                        except Exception:
                            quote_id = None
                        if quote_id:
                            for alt_path in QUOTATION_RECEIPT_DIR.glob(
                                f"quotation_{quote_id}_*.pdf"
                            ):
                                pdf_candidate = alt_path
                                break
                    if pdf_candidate:
                        file_path = pdf_candidate

                    download_payload: Optional[bytes] = None
                    download_name: Optional[str] = None

                    if regenerated_bytes:
                        try:
                            quote_id = int(row.get("quotation_id"))
                        except Exception:
                            quote_id = None
                        persisted_path = None
                        if quote_id:
                            persisted_path = _persist_quotation_pdf(
                                quote_id,
                                regenerated_bytes,
                                clean_text(row.get("reference")),
                            )
                            if persisted_path:
                                try:
                                    conn.execute(
                                        "UPDATE quotations SET document_path=? WHERE quotation_id=?",
                                        (persisted_path, quote_id),
                                    )
                                    conn.commit()
                                    file_path = BASE_DIR / persisted_path
                                except sqlite3.Error:
                                    pass
                        download_payload = regenerated_bytes
                        download_name = (
                            file_path.name
                            if file_path and file_path.exists()
                            else f"{Path(doc_path).stem}.pdf" if doc_path else "quotation.pdf"
                        )

                    if download_payload is None and file_path and file_path.exists():
                        try:
                            download_payload = file_path.read_bytes()
                        except OSError:
                            download_payload = None
                        download_name = file_path.name

                    if download_payload:
                        cols[3].download_button(
                            "Download PDF" if download_name.endswith(".pdf") else "Download",
                            download_payload,
                            file_name=download_name,
                            key=download_key,
                            mime="application/pdf"
                            if download_name.endswith(".pdf")
                            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    else:
                        cols[3].caption("File missing")
                st.markdown("</div>", unsafe_allow_html=True)

        uploaded_quotes = df_query(
            conn,
            dedent(
                """
                SELECT q.quotation_id,
                       q.reference,
                       q.customer_company,
                       q.customer_name,
                       q.customer_contact,
                       q.total_amount,
                       q.status,
                       q.quote_date,
                       q.document_path,
                       q.follow_up_status,
                       q.follow_up_date,
                       q.follow_up_notes,
                       q.salesperson_name,
                       q.updated_at,
                       COALESCE(u.username, 'User #' || q.created_by) AS owner
                FROM quotations q
                LEFT JOIN users u ON u.user_id = q.created_by
                WHERE q.document_path IS NOT NULL AND q.document_path != ''
                ORDER BY datetime(q.updated_at) DESC, datetime(q.quote_date) DESC, q.quotation_id DESC
                LIMIT 12
                """
            ),
        )

        if not uploaded_quotes.empty:
            uploaded_quotes = fmt_dates(
                uploaded_quotes, ["quote_date", "follow_up_date", "updated_at"]
            )
            uploaded_quotes["quotation_id"] = uploaded_quotes["quotation_id"].apply(
                lambda value: int(_coerce_float(value, 0))
            )
            st.markdown("#### Uploaded quotation details")
            option_labels = {
                int(row["quotation_id"]): (
                    f"{clean_text(row.get('reference')) or 'Quotation'} • "
                    f"{clean_text(row.get('customer_company')) or clean_text(row.get('customer_name')) or 'Customer'}"
                )
                for _, row in uploaded_quotes.iterrows()
            }
            quote_options = list(option_labels.keys())
            selected_quote_id = st.selectbox(
                "Select a quotation to inspect",
                quote_options,
                format_func=lambda qid: option_labels.get(qid, f"Quotation #{qid}"),
                key="dashboard_uploaded_quote_selector",
            )

            quote_match = uploaded_quotes[
                uploaded_quotes["quotation_id"] == selected_quote_id
            ].iloc[0]
            left, right = st.columns((2, 1))
            customer_lines = [
                f"**Customer:** {clean_text(quote_match.get('customer_company')) or clean_text(quote_match.get('customer_name')) or '—'}",
                f"**Contact:** {clean_text(quote_match.get('customer_contact')) or '—'}",
            ]
            if quote_match.get("follow_up_status"):
                follow_label = clean_text(quote_match.get("follow_up_status"))
                follow_date = clean_text(quote_match.get("follow_up_date")) or "—"
                customer_lines.append(
                    f"**Follow-up:** {follow_label.title()} • {follow_date}"
                )
            if quote_match.get("follow_up_notes"):
                customer_lines.append(
                    f"**Admin remarks:** {clean_text(quote_match.get('follow_up_notes'))}"
                )
            left.markdown("\n".join(customer_lines))

            status_label = clean_text(quote_match.get("status")) or "Pending"
            amount_label = format_money(quote_match.get("total_amount")) or f"{_coerce_float(quote_match.get('total_amount'), 0.0):,.2f}"
            right.metric("Total (BDT)", amount_label, help=f"Status: {status_label.title()}")
            right.caption(
                f"Sales: {clean_text(quote_match.get('salesperson_name')) or clean_text(quote_match.get('owner')) or '—'}"
            )
            right.caption(
                f"Last updated {format_time_ago(quote_match.get('updated_at')) or quote_match.get('updated_at') or '—'}"
            )

            doc_path = clean_text(quote_match.get("document_path"))
            resolved_path = resolve_upload_path(doc_path)
            if resolved_path and resolved_path.exists():
                try:
                    payload = resolved_path.read_bytes()
                except OSError:
                    payload = None
                if payload:
                    st.download_button(
                        "Download uploaded quotation",
                        payload,
                        file_name=resolved_path.name,
                        key=f"dashboard_quote_download_{selected_quote_id}",
                    )
                else:
                    st.warning("The uploaded quotation file could not be read.", icon="⚠️")
            else:
                st.caption("Uploaded quotation file not found on disk.")

    quote_scope, quote_params = _quotation_scope_filter()
    quote_clause = quote_scope.replace("WHERE", "WHERE", 1)
    quotes_df = df_query(
        conn,
        dedent(
            f"""
            SELECT quotation_id, reference, quote_date, total_amount, status
            FROM quotations
            {quote_clause}
            ORDER BY datetime(quote_date) DESC, quotation_id DESC
            LIMIT 5
            """
        ),
        quote_params,
    )
    if not quotes_df.empty:
        st.markdown("#### Quotation insights")
        total_quotes = int(len(quotes_df.index))
        paid_quotes = int((quotes_df["status"].str.lower() == "paid").sum())
        conversion = (paid_quotes / total_quotes) * 100 if total_quotes else 0.0
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Quotations created", total_quotes)
        metrics_cols[1].metric("Paid / converted", paid_quotes)
        metrics_cols[2].metric("Conversion rate", f"{conversion:.1f}%")

        quotes_df = fmt_dates(quotes_df, ["quote_date"])
        quotes_df["total_amount"] = quotes_df["total_amount"].apply(
            lambda value: format_money(value) or f"{_coerce_float(value, 0.0):,.2f}"
        )
        st.dataframe(
            quotes_df.rename(
                columns={
                    "reference": "Reference",
                    "quote_date": "Date",
                    "total_amount": "Total (BDT)",
                    "status": "Status",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

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
        SELECT c.name AS customer, p.name AS product, p.model, w.serial, w.issue_date, w.expiry_date,
               COALESCE(c.sales_person, u.username) AS staff
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        LEFT JOIN users u ON u.user_id = c.created_by
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
            staff_label = row.get("Staff") or ""
            if description:
                staff_note = f" (by {staff_label})" if staff_label else ""
                lines.append(f"- {customer}: {description}{staff_note}")
            else:
                staff_note = f" (by {staff_label})" if staff_label else ""
                lines.append(f"- {customer}{staff_note}")
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
        archive_bytes = export_full_archive(conn, excel_bytes)
        download_cols = st.columns([0.5, 0.5])
        with download_cols[0]:
            st.download_button(
                "⬇️ Download full database (Excel)",
                excel_bytes,
                file_name="ps_crm.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with download_cols[1]:
            st.download_button(
                "⬇️ Download full archive (.rar)",
                archive_bytes,
                file_name="ps_crm_full.rar",
                mime="application/x-rar-compressed",
                help="Bundles the database, uploads, and receipts into one portable file.",
            )

        toggle_label = (
            "🗑️ Deleted data"
            if not st.session_state.get("show_deleted_panel")
            else "Hide deleted data"
        )
        toggle_cols = st.columns([0.78, 0.22])
        with toggle_cols[1]:
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

    if is_admin:
        st.markdown("---")
        _render_admin_kpi_panel(conn)


def show_expiry_notifications(conn):
    is_admin = current_user_is_admin()

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
    user = get_current_user()
    if not user:
        return

    is_admin = user.get("role") == "admin"
    alerts = list(reversed(get_runtime_notifications()))
    user_id = current_user_id()
    alerts.extend(_build_staff_alerts(conn, user_id=user_id))
    activity = fetch_activity_feed(conn, limit=ACTIVITY_FEED_LIMIT) if is_admin else []

    total = len(alerts) + len(activity)
    label = "🔔" if total == 0 else f"🔔 {total}"
    container = st.container()
    with container:
        for alert in alerts:
            if alert.get("severity") in {"warning", "error", "danger"}:
                toast_msg = alert.get("message") or alert.get("title")
                if toast_msg:
                    st.toast(toast_msg, icon="⚠️")
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
        user_seed = get_current_user()
        salesperson_seed = ""
        if user_seed:
            bits = [clean_text(user_seed.get("username")), clean_text(user_seed.get("phone"))]
            salesperson_seed = " ".join(bit for bit in bits if bit)
        st.session_state.setdefault("new_customer_sales_person", salesperson_seed)
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
            required_columns = ["name", "model", "serial", "quantity", "unit_price", "total"]
            for column in required_columns:
                if column not in products_df.columns:
                    default_value = 0.0 if column in ["unit_price", "total"] else ""
                    if column == "quantity":
                        default_value = 1
                    products_df[column] = default_value
            products_df = products_df[required_columns]
            products_df["total"] = products_df.apply(
                lambda row: max(
                    _coerce_float(row.get("quantity"), 1.0)
                    * _coerce_float(row.get("unit_price"), 0.0),
                    0.0,
                ),
                axis=1,
            )
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
                    "unit_price": st.column_config.NumberColumn(
                        "Unit price",
                        min_value=0.0,
                        step=100.0,
                        format="%.2f",
                    ),
                    "total": st.column_config.NumberColumn(
                        "Line total",
                        format="%.2f",
                        help="Quantity × unit price (read only)",
                        disabled=True,
                    ),
                },
            )
            editor_df = edited_products if isinstance(edited_products, pd.DataFrame) else pd.DataFrame(edited_products)
            editor_df["total"] = editor_df.apply(
                lambda row: max(
                    _coerce_float(row.get("quantity"), 1.0)
                    * _coerce_float(row.get("unit_price"), 0.0),
                    0.0,
                ),
                axis=1,
            )
            product_entries = editor_df.to_dict("records")
            st.session_state["new_customer_products_rows"] = product_entries
            with st.expander("Attachments & advanced details", expanded=True):
                do_code = st.text_input(
                    "Delivery order (DO) code",
                    key="new_customer_do_code",
                    help="Link the customer to an existing delivery order if available.",
                )
                work_done_number = st.text_input(
                    "Work done number (optional)",
                    key="new_customer_work_done_number",
                    help="Reference code used when creating a work done record.",
                )
                service_reference = st.text_input(
                    "Service reference (optional)",
                    key="new_customer_service_reference",
                    help="Code or tag to use when creating the linked service record.",
                )
                maintenance_reference = st.text_input(
                    "Maintenance reference (optional)",
                    key="new_customer_maintenance_reference",
                    help="Code or tag for the maintenance record created with this customer.",
                )
                sales_person_input = st.text_input(
                    "Sales person",
                    value=st.session_state.get("new_customer_sales_person", salesperson_seed),
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
                st.markdown("---")
                st.caption(
                    "Create delivery, work done, service or maintenance entries alongside this customer."
                )

                default_related: list[str] = []
                if st.session_state.get("new_customer_create_delivery_order") or do_code:
                    default_related.append("Delivery order")
                if st.session_state.get("new_customer_create_work_done"):
                    default_related.append("Work done")
                if st.session_state.get("new_customer_create_service"):
                    default_related.append("Service")
                if st.session_state.get("new_customer_create_maintenance"):
                    default_related.append("Maintenance")

                selected_related = st.multiselect(
                    "Related records to create",
                    options=["Delivery order", "Work done", "Service", "Maintenance"],
                    default=default_related,
                    help=(
                        "Select the record types you want to save with this customer. Only the "
                        "relevant inputs will be shown so the section matches the dedicated pages."
                    ),
                )

                create_delivery_order = "Delivery order" in selected_related
                create_work_done = "Work done" in selected_related
                create_service = "Service" in selected_related
                create_maintenance = "Maintenance" in selected_related
                st.session_state["new_customer_create_delivery_order"] = create_delivery_order
                st.session_state["new_customer_create_work_done"] = create_work_done
                st.session_state["new_customer_create_service"] = create_service
                st.session_state["new_customer_create_maintenance"] = create_maintenance

                if create_delivery_order:
                    st.subheader("Delivery order details")
                    st.caption(
                        "Uses the delivery order code and product list exactly as on the Delivery orders page."
                    )
                    do_status = st.selectbox(
                        "Delivery order status",
                        options=["pending", "paid"],
                        key="new_customer_do_status",
                        help="Mark as paid to require an accompanying receipt.",
                    )
                    st.file_uploader(
                        "Delivery order receipt (required if paid)",
                        type=["pdf", "png", "jpg", "jpeg", "webp"],
                        key="new_customer_do_receipt",
                        help="Upload proof of payment when marking the DO as paid.",
                    )

                if create_work_done:
                    st.subheader("Work done details")
                    st.caption(
                        "Matches the work done form so you can fill in the reference, remarks and PDF attachment."
                    )
                    work_done_cols = st.columns((1, 2, 2))
                    with work_done_cols[0]:
                        st.caption(
                            f"Work done number: {clean_text(work_done_number) or '— set above —'}"
                        )
                    work_done_status = work_done_cols[1].selectbox(
                        "Work done status",
                        options=["pending", "paid"],
                        key="new_customer_work_done_status",
                        help="Mark as paid to require an accompanying receipt.",
                    )
                    work_done_pdf = work_done_cols[2].file_uploader(
                        "Attach work done PDF",
                        type=["pdf"],
                        key="new_customer_work_done_pdf",
                    )
                    work_done_receipt = st.file_uploader(
                        "Payment receipt (required for paid work done)",
                        type=["pdf", "png", "jpg", "jpeg", "webp"],
                        key="new_customer_work_done_receipt",
                        help="Upload proof of payment if this work done is already paid.",
                    )
                    work_done_notes = st.text_area(
                        "Work done description / remarks",
                        key="new_customer_work_done_notes",
                    )
                else:
                    work_done_status = st.session_state.get("new_customer_work_done_status", "pending")
                    work_done_pdf = st.session_state.get("new_customer_work_done_pdf")
                    work_done_receipt = st.session_state.get("new_customer_work_done_receipt")
                    work_done_notes = st.session_state.get("new_customer_work_done_notes")

                if create_service or create_maintenance:
                    st.subheader("After-sales records")

                if create_service:
                    service_cols = st.columns((1, 1, 1))
                    service_date_input = service_cols[0].date_input(
                        "Service date",
                        value=purchase_date,
                        key="new_customer_service_date",
                    )
                    service_description = service_cols[1].text_area(
                        "Service description",
                        key="new_customer_service_description",
                        help="Mirror of the service page description field.",
                    )
                    service_status = service_cols[2].selectbox(
                        "Service payment status",
                        options=["pending", "paid"],
                        key="new_customer_service_payment_status",
                        help="Track whether the service has been paid.",
                    )
                    service_receipt = st.file_uploader(
                        "Service receipt (required if paid)",
                        type=["pdf", "png", "jpg", "jpeg", "webp"],
                        key="new_customer_service_receipt",
                        help="Upload payment receipt when marking the service as paid.",
                    )
                else:
                    service_date_input = st.session_state.get("new_customer_service_date")
                    service_description = st.session_state.get("new_customer_service_description")
                    service_status = st.session_state.get("new_customer_service_payment_status", "pending")
                    service_receipt = st.session_state.get("new_customer_service_receipt")

                if create_maintenance:
                    maintenance_cols = st.columns((1, 1, 1))
                    maintenance_date_input = maintenance_cols[0].date_input(
                        "Maintenance date",
                        value=purchase_date,
                        key="new_customer_maintenance_date",
                    )
                    maintenance_description = maintenance_cols[1].text_area(
                        "Maintenance description",
                        key="new_customer_maintenance_description",
                        help="Same fields as the maintenance page for easy entry.",
                    )
                    maintenance_status = maintenance_cols[2].selectbox(
                        "Maintenance payment status",
                        options=["pending", "paid"],
                        key="new_customer_maintenance_payment_status",
                        help="Track whether maintenance has been paid.",
                    )
                    maintenance_receipt = st.file_uploader(
                        "Maintenance receipt (required if paid)",
                        type=["pdf", "png", "jpg", "jpeg", "webp"],
                        key="new_customer_maintenance_receipt",
                        help="Upload payment receipt when marking maintenance as paid.",
                    )
                else:
                    maintenance_date_input = st.session_state.get("new_customer_maintenance_date")
                    maintenance_description = st.session_state.get("new_customer_maintenance_description")
                    maintenance_status = st.session_state.get(
                        "new_customer_maintenance_payment_status", "pending"
                    )
                    maintenance_receipt = st.session_state.get("new_customer_maintenance_receipt")
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
                if create_delivery_order and not do_serial:
                    errors.append(
                        "Enter a delivery order code to create a linked delivery order record."
                    )
                work_done_serial = clean_text(work_done_number)
                if create_work_done and not work_done_serial:
                    errors.append("Work done number is required when creating a record.")
                if create_service and not clean_text(service_description):
                    errors.append("Add a short service description to create the service record.")
                if create_maintenance and not clean_text(maintenance_description):
                    errors.append(
                        "Add a short maintenance description to create the maintenance record."
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
                sales_person_value = clean_text(sales_person_input) or salesperson_seed
                cleaned_products, product_labels = normalize_product_entries(product_entries)
                product_label = "\n".join(product_labels) if product_labels else None
                product_items = _products_to_delivery_items(cleaned_products)
                delivery_items_payload = None
                delivery_total = 0.0
                do_status_value = (
                    clean_text(st.session_state.get("new_customer_do_status")) or "pending"
                )
                if do_status_value not in {"pending", "paid"}:
                    do_status_value = "pending"
                if product_items:
                    normalized_items, delivery_total = normalize_delivery_items(product_items)
                    if normalized_items:
                        delivery_items_payload = json.dumps(
                            normalized_items, ensure_ascii=False
                        )
                    else:
                        delivery_total = 0.0
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
                        sales_person_value,
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
                if do_serial and (create_delivery_order or do_pdf is not None):
                    stored_path = None
                    if do_pdf is not None:
                        safe_name = _sanitize_path_component(do_serial)
                        stored_path = store_uploaded_pdf(
                            do_pdf, DELIVERY_ORDER_DIR, filename=f"{safe_name}.pdf"
                        )
                    do_receipt_path = None
                    cur = conn.cursor()
                    existing = cur.execute(
                        """
                        SELECT customer_id, file_path, items_payload, total_amount, payment_receipt_path
                        FROM delivery_orders
                        WHERE do_number = ? AND COALESCE(record_type, 'delivery_order') = 'delivery_order'
                        """,
                        (do_serial,),
                    ).fetchone()
                    existing_receipt = clean_text(existing[4]) if existing else None
                    if do_status_value == "paid":
                        do_receipt = st.session_state.get("new_customer_do_receipt")
                        if do_receipt:
                            do_receipt_path = store_payment_receipt(
                                do_receipt,
                                identifier=f"{_sanitize_path_component(do_serial) or 'do'}_receipt",
                                target_dir=DELIVERY_RECEIPT_DIR,
                            )
                        if not do_receipt_path and existing_receipt:
                            do_receipt_path = existing_receipt
                        if not do_receipt_path:
                            st.error("Upload a payment receipt before marking the DO as paid.")
                            return
                    product_summary = (
                        cleaned_products[0].get("name") if cleaned_products else product_label
                    )
                    sales_clean = clean_text(sales_person_input)
                    if existing:
                        existing_customer, existing_path, existing_items, existing_total, existing_receipt = existing
                        if do_status_value == "paid" and not do_receipt_path:
                            do_receipt_path = clean_text(existing_receipt)
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
                                """
                                UPDATE delivery_orders
                                   SET customer_id=?,
                                       description=?,
                                       sales_person=?,
                                       remarks=?,
                                       file_path=?,
                                       items_payload=COALESCE(?, items_payload),
                                       total_amount=COALESCE(?, total_amount),
                                       status=?,
                                       payment_receipt_path=COALESCE(?, payment_receipt_path),
                                       record_type='delivery_order'
                                 WHERE do_number=? AND COALESCE(record_type, 'delivery_order') = 'delivery_order'
                                """,
                                (
                                    cid,
                                    product_summary,
                                    sales_clean,
                                    remarks_val,
                                    final_path,
                                    delivery_items_payload,
                                    delivery_total if delivery_items_payload else existing_total,
                                    do_status_value,
                                    do_receipt_path,
                                    do_serial,
                                ),
                            )
                            conn.commit()
                    else:
                        conn.execute(
                            """
                            INSERT INTO delivery_orders (
                                do_number,
                                customer_id,
                                order_id,
                                description,
                                sales_person,
                                remarks,
                                file_path,
                                items_payload,
                                total_amount,
                                status,
                                payment_receipt_path,
                                record_type
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'delivery_order')
                            """,
                            (
                                do_serial,
                                cid,
                                None,
                                product_summary,
                                sales_clean,
                                remarks_val,
                                stored_path,
                                delivery_items_payload,
                                delivery_total if delivery_items_payload else None,
                                do_status_value,
                                do_receipt_path,
                            ),
                        )
                        conn.commit()

                if create_work_done and work_done_serial:
                    if not product_items:
                        st.warning(
                            "Add at least one product row before creating a work done record."
                        )
                    else:
                        work_done_items, work_done_total = normalize_delivery_items(
                            product_items
                        )
                        if not work_done_items:
                            st.warning(
                                "Work done could not be saved because no valid products were provided."
                            )
                        else:
                            work_done_payload = json.dumps(
                                work_done_items, ensure_ascii=False
                            )
                            work_done_path = None
                            if work_done_pdf is not None:
                                safe_name = _sanitize_path_component(work_done_serial)
                                work_done_path = store_uploaded_pdf(
                                    work_done_pdf,
                                    DELIVERY_ORDER_DIR,
                                    filename=f"work_done_{safe_name}.pdf",
                                )
                            work_done_saved = False
                            work_done_status_value = clean_text(work_done_status) or "pending"
                            if work_done_status_value not in {"pending", "paid"}:
                                work_done_status_value = "pending"
                            work_done_receipt_path = None
                            if work_done_status_value == "paid":
                                work_done_receipt_path = store_payment_receipt(
                                    work_done_receipt,
                                    identifier=f"{_sanitize_path_component(work_done_serial) or 'work_done'}_receipt",
                                    target_dir=DELIVERY_RECEIPT_DIR,
                                )
                            existing_work_done = df_query(
                                conn,
                                "SELECT record_type, file_path, payment_receipt_path FROM delivery_orders WHERE do_number = ?",
                                (work_done_serial,),
                            )
                            existing_work_done_receipt = None
                            if not existing_work_done.empty:
                                existing_work_done_receipt = clean_text(
                                    existing_work_done.iloc[0].get("payment_receipt_path")
                                )
                            if work_done_status_value == "paid" and not (
                                work_done_receipt_path or existing_work_done_receipt
                            ):
                                st.error(
                                    "Upload a payment receipt before marking the work done as paid."
                                )
                                return
                            work_done_description = clean_text(work_done_notes) or product_label
                            if existing_work_done.empty:
                                conn.execute(
                                    """
                                    INSERT INTO delivery_orders (
                                        do_number,
                                        customer_id,
                                        description,
                                        sales_person,
                                        remarks,
                                        file_path,
                                        items_payload,
                                        total_amount,
                                        status,
                                        payment_receipt_path,
                                        record_type
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'work_done')
                                    """,
                                    (
                                        work_done_serial,
                                        cid,
                                        work_done_description,
                                        sales_person_value,
                                        clean_text(work_done_notes),
                                        work_done_path,
                                        work_done_payload,
                                        work_done_total,
                                        work_done_status_value,
                                        work_done_receipt_path,
                                    ),
                                )
                                work_done_saved = True
                            else:
                                existing_type = clean_text(
                                    existing_work_done.iloc[0].get("record_type")
                                ) or "delivery_order"
                                existing_work_done_receipt = clean_text(
                                    existing_work_done.iloc[0].get("payment_receipt_path")
                                )
                                if work_done_status_value == "paid" and not work_done_receipt_path:
                                    work_done_receipt_path = existing_work_done_receipt
                                if existing_type != "work_done":
                                    st.error(
                                        "A delivery order already uses this number. Choose a different work done number."
                                    )
                                    if work_done_path:
                                        new_path = resolve_upload_path(work_done_path)
                                        if new_path and new_path.exists():
                                            try:
                                                new_path.unlink()
                                            except Exception:
                                                pass
                                else:
                                    existing_path = clean_text(
                                        existing_work_done.iloc[0].get("file_path")
                                    )
                                    final_path = work_done_path or existing_path
                                    if work_done_path and existing_path and work_done_path != existing_path:
                                        old_path = resolve_upload_path(existing_path)
                                        if old_path and old_path.exists():
                                            try:
                                                old_path.unlink()
                                            except Exception:
                                                pass
                                    conn.execute(
                                        """
                                        UPDATE delivery_orders
                                           SET customer_id=?,
                                               description=?,
                                               sales_person=?,
                                               remarks=?,
                                               file_path=?,
                                               items_payload=?,
                                               total_amount=?,
                                               status=?,
                                               payment_receipt_path=COALESCE(?, payment_receipt_path),
                                               record_type='work_done'
                                         WHERE do_number=? AND COALESCE(record_type, 'delivery_order') = 'work_done'
                                        """,
                                        (
                                            cid,
                                            work_done_description,
                                            sales_person_value,
                                            clean_text(work_done_notes),
                                            final_path,
                                            work_done_payload,
                                            work_done_total,
                                            work_done_status_value,
                                            work_done_receipt_path,
                                            work_done_serial,
                                        ),
                                    )
                                    work_done_saved = True
                            if work_done_saved:
                                formatted_work_done = format_money(work_done_total)
                                if not formatted_work_done and work_done_total is not None:
                                    try:
                                        formatted_work_done = f"{float(work_done_total):,.2f}"
                                    except (TypeError, ValueError):
                                        formatted_work_done = ""
                                log_activity(
                                    conn,
                                    event_type="work_done_created",
                                    description=(
                                        f"Work done {work_done_serial} saved for {name_val or 'customer'}"
                                        f" ({formatted_work_done})"
                                    ),
                                    entity_type="work_done",
                                    entity_id=None,
                                )
                                conn.commit()

                if create_service:
                    service_date_str = to_iso_date(service_date_input) or purchase_str
                    if not service_date_str:
                        service_date_str = datetime.utcnow().strftime("%Y-%m-%d")
                    service_reference = clean_text(service_reference) or do_serial
                    service_status_value = clean_text(service_status) or "pending"
                    if service_status_value not in {"pending", "paid"}:
                        service_status_value = "pending"
                    service_receipt_path = None
                    if service_status_value == "paid":
                        service_receipt_path = store_payment_receipt(
                            service_receipt,
                            identifier=f"{_sanitize_path_component(service_reference) or 'service'}_receipt",
                            target_dir=SERVICE_BILL_DIR,
                        )
                        if not service_receipt_path:
                            st.error(
                                "Upload a payment receipt before marking the service as paid."
                            )
                            return
                    conn.execute(
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
                            payment_status,
                            payment_receipt_path,
                            updated_at,
                            created_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
                        """,
                        (
                            service_reference,
                            cid,
                            service_date_str,
                            service_date_str,
                            service_date_str,
                            clean_text(service_description),
                            DEFAULT_SERVICE_STATUS,
                            remarks_val,
                            product_label,
                            service_status_value,
                            service_receipt_path,
                            created_by,
                        ),
                    )
                    log_activity(
                        conn,
                        event_type="service_created",
                        description=f"Service logged for {name_val or 'customer'}",
                        entity_type="service",
                        entity_id=None,
                    )
                    conn.commit()

                if create_maintenance:
                    maintenance_date_str = to_iso_date(maintenance_date_input) or purchase_str
                    if not maintenance_date_str:
                        maintenance_date_str = datetime.utcnow().strftime("%Y-%m-%d")
                    maintenance_reference = clean_text(maintenance_reference) or do_serial
                    maintenance_status_value = clean_text(maintenance_status) or "pending"
                    if maintenance_status_value not in {"pending", "paid"}:
                        maintenance_status_value = "pending"
                    maintenance_receipt_path = None
                    if maintenance_status_value == "paid":
                        maintenance_receipt_path = store_payment_receipt(
                            maintenance_receipt,
                            identifier=f"{_sanitize_path_component(maintenance_reference) or 'maintenance'}_receipt",
                            target_dir=MAINTENANCE_DOCS_DIR,
                        )
                        if not maintenance_receipt_path:
                            st.error(
                                "Upload a payment receipt before marking the maintenance as paid."
                            )
                            return
                    conn.execute(
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
                            payment_status,
                            payment_receipt_path,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """,
                        (
                            maintenance_reference,
                            cid,
                            maintenance_date_str,
                            maintenance_date_str,
                            maintenance_date_str,
                            clean_text(maintenance_description),
                            DEFAULT_SERVICE_STATUS,
                            remarks_val,
                            product_label,
                            maintenance_status_value,
                            maintenance_receipt_path,
                        ),
                    )
                    log_activity(
                        conn,
                        event_type="maintenance_created",
                        description=f"Maintenance logged for {name_val or 'customer'}",
                        entity_type="maintenance",
                        entity_id=None,
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
         OR c.address LIKE '%'||?||'%'
         OR c.delivery_address LIKE '%'||?||'%'
         OR c.remarks LIKE '%'||?||'%'
         OR c.product_info LIKE '%'||?||'%'
         OR c.delivery_order_code LIKE '%'||?||'%'
         OR c.sales_person LIKE '%'||?||'%')
        """
    ).strip()
    where_parts = [search_clause]
    params: list[object] = [q, q, q, q, q, q, q, q, q, q]
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
            st.session_state.setdefault("bulk_delete_ids", [])
            with st.form("bulk_customer_delete"):
                selected_delete_ids = st.multiselect(
                    "Select customers to delete",
                    delete_choices,
                    key="bulk_delete_ids",
                    format_func=lambda cid: delete_labels.get(
                        int(cid), f"Customer #{cid}"
                    ),
                    help="Removes the selected customers and their related records.",
                )
                bulk_delete_submit = st.form_submit_button(
                    "Delete selected customers",
                    disabled=(not is_admin) or (not selected_delete_ids),
                    type="secondary",
                )
            if bulk_delete_submit and selected_delete_ids:
                if not is_admin:
                    st.error("Only admins can delete customer records.")
                    return
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
                        "UPDATE import_history SET customer_name=?, phone=?, address=?, delivery_address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                        (
                            new_name,
                            new_phone,
                            new_address,
                            new_delivery_address,
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
                    "UPDATE customers SET name=?, company_name=?, phone=?, address=?, delivery_address=?, remarks=?, purchase_date=?, product_info=?, delivery_order_code=?, sales_person=?, attachment_path=?, dup_flag=0 WHERE customer_id=?",
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
                    "UPDATE import_history SET customer_name=?, phone=?, address=?, delivery_address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
                    (
                        new_name,
                        new_phone,
                        new_address,
                        new_delivery_address,
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
               w.issue_date, w.expiry_date, w.status, w.remarks, w.dup_flag,
               COALESCE(c.sales_person, u.username) AS staff
        FROM warranties w
        LEFT JOIN customers c ON c.customer_id = w.customer_id
        LEFT JOIN products p ON p.product_id = w.product_id
        LEFT JOIN users u ON u.user_id = c.created_by
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
    viewer_id = current_user_id()
    do_df = df_query(
        conn,
        """
        SELECT d.do_number, d.customer_id, d.created_by, COALESCE(c.name, '(unknown)') AS customer_name, d.description, d.remarks, d.record_type
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        WHERE COALESCE(d.record_type, 'delivery_order') = 'delivery_order'
        ORDER BY datetime(d.created_at) DESC
        """,
    )
    allowed_customers = accessible_customer_ids(conn)
    do_df = filter_delivery_orders_for_view(
        do_df, allowed_customers, record_types={"delivery_order"}
    )
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
        st.markdown("**Products sold during service**")
        service_product_rows = st.session_state.get(
            "service_product_rows",
            [
                {
                    "name": "",
                    "model": "",
                    "serial": "",
                    "quantity": 1,
                }
            ],
        )
        service_product_editor = st.data_editor(
            pd.DataFrame(service_product_rows),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Product", help="Name / description"),
                "model": st.column_config.TextColumn("Model"),
                "serial": st.column_config.TextColumn("Serial"),
                "quantity": st.column_config.NumberColumn(
                    "Qty",
                    min_value=1,
                    step=1,
                    format="%d",
                ),
            },
            key="service_product_table",
        )
        service_product_entries = (
            service_product_editor.to_dict("records")
            if isinstance(service_product_editor, pd.DataFrame)
            else []
        )
        st.session_state["service_product_rows"] = service_product_entries
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
                    updated_at,
                    created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    current_user_id(),
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
                s.payment_receipt_path,
               s.updated_at,
               s.created_by,
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
            creator_id = row.get("created_by")
            candidates = []
            if pd.notna(service_cust):
                candidates.append(int(service_cust))
            if pd.notna(do_cust):
                candidates.append(int(do_cust))
            try:
                if viewer_id is not None and pd.notna(creator_id) and int(creator_id) == int(viewer_id):
                    return True
            except Exception:
                pass
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
        service_df.drop(
            columns=["customer_id", "do_customer_id", "created_by"],
            inplace=True,
            errors="ignore",
        )
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
        if "payment_receipt_path" in service_df.columns:
            service_df["payment_receipt_display"] = service_df["payment_receipt_path"].apply(
                lambda x: "📎" if clean_text(x) else ""
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
                "payment_receipt_display": "Receipt",
                "customer": "Customer",
                "doc_count": "Documents",
            }
        )
        display = display.drop(
            columns=["bill_document_path", "payment_receipt_path"], errors="ignore"
        )
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
        receipt_col1, receipt_col2 = st.columns([1, 1])
        existing_receipt_path = clean_text(selected_record.get("payment_receipt_path"))
        resolved_receipt = resolve_upload_path(existing_receipt_path)
        with receipt_col1:
            receipt_upload = st.file_uploader(
                "Upload payment receipt (PDF or image)",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key=f"service_edit_receipt_upload_{selected_service_id}",
            )
        with receipt_col2:
            clear_receipt = st.checkbox(
                "Remove receipt",
                value=False,
                key=f"service_edit_receipt_clear_{selected_service_id}",
            )
        if resolved_receipt and resolved_receipt.exists():
            st.download_button(
                "Download receipt",
                data=resolved_receipt.read_bytes(),
                file_name=resolved_receipt.name,
                key=f"service_receipt_download_{selected_service_id}",
            )
        elif existing_receipt_path:
            st.caption("Receipt file not found. Upload a new copy to replace it.")
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
                receipt_path_value = clean_text(selected_record.get("payment_receipt_path"))
                replaced_receipt = False
                cleared_receipt = False
                if receipt_upload is not None:
                    receipt_path_value = store_payment_receipt(
                        receipt_upload,
                        identifier=f"service_{selected_service_id}_receipt",
                        target_dir=SERVICE_BILL_DIR,
                    )
                    replaced_receipt = bool(receipt_path_value)
                elif clear_receipt and receipt_path_value:
                    old_receipt = resolve_upload_path(receipt_path_value)
                    if old_receipt and old_receipt.exists():
                        try:
                            old_receipt.unlink()
                        except Exception:
                            pass
                    receipt_path_value = None
                    cleared_receipt = True
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
                        payment_receipt_path = COALESCE(?, payment_receipt_path),
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
                        receipt_path_value,
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
                if replaced_receipt:
                    message_bits.append("Receipt uploaded")
                elif cleared_receipt:
                    message_bits.append("Receipt removed")
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


def _regenerate_quotation_pdf_from_workbook(file_path: Path) -> Optional[bytes]:
    """Recreate a quotation PDF from an archived Excel workbook.

    Older records may only have the Excel workbook stored; this helper rebuilds
    the PDF so dashboard downloads remain consistent.
    """

    if not file_path.exists():
        return None

    try:
        workbook = load_workbook(filename=file_path, data_only=True)
    except Exception:
        return None

    if "Quotation" not in workbook.sheetnames:
        return None

    ws = workbook["Quotation"]
    rows = list(ws.iter_rows(values_only=True))

    def _is_blank(row: tuple[object, ...]) -> bool:
        return all(cell is None or str(cell).strip() == "" for cell in row)

    metadata: dict[str, Optional[str]] = {}
    items: list[dict[str, object]] = []
    totals_rows: list[tuple[str, object]] = []

    section = "summary"
    header: list[str] = []
    totals_header: list[str] = []

    for row in rows:
        if _is_blank(row):
            if section == "summary":
                section = "items_header"
            elif section == "items":
                section = "totals_header"
            continue

        if section == "summary":
            key = clean_text(row[0]) if len(row) else None
            value = clean_text(row[1]) if len(row) > 1 else None
            if key:
                metadata[key] = value
            continue

        if section == "items_header":
            header = [clean_text(cell) or "" for cell in row if cell is not None]
            section = "items"
            continue

        if section == "items":
            if not header:
                continue
            item_values = list(row)
            if all(cell is None for cell in item_values):
                section = "totals_header"
                continue
            item: dict[str, object] = {}
            for idx, col_name in enumerate(header):
                if not col_name:
                    continue
                value = item_values[idx] if idx < len(item_values) else None
                if isinstance(value, str):
                    value = clean_text(value)
                item[col_name] = value
            if item:
                items.append(item)
            continue

        if section == "totals_header":
            totals_header = [clean_text(cell) or "" for cell in row if cell is not None]
            section = "totals"
            continue

        if section == "totals":
            if not totals_header:
                continue
            if all(cell is None for cell in row):
                break
            label = clean_text(row[0]) if len(row) else None
            value = row[1] if len(row) > 1 else None
            if label:
                totals_rows.append((label, value))

    def _total_from_label(label: str) -> Optional[float]:
        for key, value in totals_rows:
            if clean_text(key) == label:
                return _coerce_float(value, 0.0)
        return None

    for item in items:
        for tax_label in [
            "CGST (%)",
            "CGST amount",
            "SGST (%)",
            "SGST amount",
            "IGST (%)",
            "IGST amount",
        ]:
            item.pop(tax_label, None)

    totals = {
        "gross_total": _total_from_label("gross amount")
        or _total_from_label("manual total override")
        or sum(
            _coerce_float(
                item.get("Total Price, Tk.") or item.get("Gross amount"), 0.0
            )
            for item in items
        ),
        "discount_total": _total_from_label("discount total")
        or sum(_coerce_float(item.get("Discount amount"), 0.0) for item in items),
        "grand_total": _total_from_label("grand total")
        or _total_from_label("manual total override")
        or sum(
            _coerce_float(
                item.get("Total Price, Tk.") or item.get("Line total"), 0.0
            )
            for item in items
        ),
    }

    grand_total_label = format_money(totals["grand_total"]) or f"{totals['grand_total']:,.2f}"
    grand_total_words = format_amount_in_words(totals["grand_total"]) or grand_total_label

    try:
        return _build_quotation_pdf(
            metadata=metadata,
            items=items,
            totals=totals,
            grand_total_label=grand_total_label,
            grand_total_words=grand_total_words,
        )
    except Exception:
        return None


def _resolve_letterhead_path(template_choice: Optional[str] = None) -> Optional[Path]:
    template_choice = template_choice or "Default letterhead"
    base_dir = Path(__file__).resolve().parent
    sandbox_letterhead = Path("/mnt/data/eed2a8fe-ec62-4729-9ed4-aedb65953acf.png")
    default_candidates = [
        sandbox_letterhead,
        base_dir / "letterhead.png",
        base_dir / "letterhead",
        base_dir / "ps_letterhead.png",
    ]

    preferred: list[Path] = []
    if template_choice == "PS letterhead":
        preferred = [sandbox_letterhead, base_dir / "ps_letterhead.png"]
    elif template_choice == "Default letterhead":
        preferred = [sandbox_letterhead, base_dir / "letterhead.png", base_dir / "letterhead"]

    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in preferred + default_candidates:
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)

    for path in candidates:
        if path.exists():
            return path
    return None




def _build_quotation_pdf(
    *,
    metadata: dict[str, Optional[str]],
    items: list[dict[str, object]],
    totals: dict[str, float],
    grand_total_label: str,
    template_choice: Optional[str] = None,
    grand_total_words: Optional[str] = None,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Muted",
            parent=styles["Normal"],
            textColor=colors.HexColor("#475569"),
            fontSize=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
        )
    )

    reference = metadata.get("Reference number") or metadata.get("Quotation reference") or "—"
    date_value = metadata.get("Date") or "—"
    customer_name = metadata.get("Customer company") or metadata.get("Customer / organisation") or "—"
    customer_contact = metadata.get("Customer contact name") or ""
    customer_contact_details = metadata.get("Customer contact") or ""
    customer_address = metadata.get("Customer address") or ""
    customer_district = metadata.get("Customer district") or ""
    attention_name = metadata.get("Attention name") or "—"
    prepared_by = metadata.get("Salesperson name") or "—"
    prepared_title = metadata.get("Salesperson title") or ""
    prepared_contact = metadata.get("Salesperson contact") or ""
    prepared_email = metadata.get("Salesperson email") or ""
    terms = metadata.get("Notes / terms") or ""

    story: list[object] = []
    letterhead_path = _resolve_letterhead_path(template_choice)
    letterhead_img_path = str(letterhead_path) if letterhead_path and letterhead_path.exists() else None

    def _draw_letterhead_background(canvas, _doc):
        if not letterhead_img_path:
            return
        canvas.saveState()
        try:
            reader = ImageReader(letterhead_img_path)
            img_width, img_height = reader.getSize()
            page_width, page_height = _doc.pagesize
            target_width = page_width
            target_height = target_width * (img_height / img_width)
            canvas.drawImage(
                letterhead_img_path,
                0,
                page_height - target_height,
                width=target_width,
                height=target_height,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass
        finally:
            canvas.restoreState()

    story.append(Spacer(1, 18))

    header_table = Table(
        [
            ["", Paragraph(f"<b>Date:</b> {date_value}", styles["BodySmall"])],
            ["", Paragraph(f"<b>Ref:</b> {reference}", styles["BodySmall"])],
        ],
        colWidths=[doc.width * 0.58, doc.width * 0.42],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 10))

    address_lines = [customer_name, customer_contact, customer_contact_details, customer_address, customer_district]
    address_text = "<br/>".join(filter(None, address_lines)) or "—"
    story.append(Paragraph("<b>Customer</b>", styles["BodySmall"]))
    story.append(Paragraph(address_text, styles["BodySmall"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"<b>Attention:</b> {attention_name}",
            styles["BodySmall"],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "<b>PRICE SCHEDULE</b>",
            styles["BodySmall"],
        )
    )
    story.append(Spacer(1, 4))

    table_data = [
        [
            "Sl No.",
            "Description of Generator",
            "Qty.",
            "Unit Price, Tk.",
            "Total Price, Tk.",
        ]
    ]

    for idx, item in enumerate(items, start=1):
        description_parts = []
        description_parts.append(
            html.escape(
                clean_text(item.get("Description of Generator"))
                or clean_text(item.get("Description"))
                or clean_text(item.get("description"))
                or "Item"
            )
        )
        specs_text = clean_text(item.get("Specs")) or ""
        note_text = (
            clean_text(item.get("Notes"))
            or clean_text(item.get("Note"))
            or clean_text(item.get("note"))
            or ""
        )
        if specs_text:
            description_parts.append(
                f"<font color='#475569'>{html.escape(specs_text)}</font>"
            )
        if note_text:
            description_parts.append(
                f"<font color='#94a3b8' size='9'>{html.escape(note_text)}</font>"
            )
        description = "<br/>".join(filter(None, description_parts))
        qty_value = _coerce_float(
            item.get("Qty.") or item.get("Quantity") or item.get("quantity"), 0.0
        )
        rate_value = (
            item.get("Unit Price, Tk.")
            if "Unit Price, Tk." in item
            else item.get("Rate")
            if "Rate" in item
            else item.get("unit_price")
        )
        line_total_value = (
            item.get("Total Price, Tk.")
            if "Total Price, Tk." in item
            else item.get("Line total")
            if "Line total" in item
            else item.get("line_total")
        )
        table_data.append(
            [
                str(idx),
                Paragraph(description, styles["BodySmall"]),
                Paragraph(f"{qty_value:,.0f}", styles["BodySmall"]),
                Paragraph(format_money(rate_value) or f"{_coerce_float(rate_value, 0.0):,.2f}", styles["BodySmall"]),
                Paragraph(
                    format_money(line_total_value) or f"{_coerce_float(line_total_value, 0.0):,.2f}",
                    styles["BodySmall"],
                ),
            ]
        )

    table_data.append(
        [
            "",
            "",
            "",
            Paragraph("<b>Total Amount (Tk.)</b>", styles["BodySmall"]),
            Paragraph(f"<b>{grand_total_label}</b>", styles["BodySmall"]),
        ]
    )

    col_widths = [doc.width * 0.08, doc.width * 0.48, doc.width * 0.1, doc.width * 0.17, doc.width * 0.17]
    pricing_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    pricing_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -2), 0.5, colors.grey),
                ("BOX", (0, 0), (-1, -2), 0.5, colors.grey),
                ("SPAN", (0, -1), (2, -1)),
                ("BACKGROUND", (0, -1), (-1, -1), colors.whitesmoke),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("GRID", (0, -1), (-1, -1), 0.5, colors.grey),
                ("BOX", (0, -1), (-1, -1), 0.5, colors.grey),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(pricing_table)
    story.append(Spacer(1, 10))

    discount_value = _coerce_float(totals.get("discount_total") if totals else None, 0.0)
    if not grand_total_words:
        amount_in_words = format_amount_in_words(totals.get("grand_total"))
        grand_total_words = amount_in_words or grand_total_label

    if discount_value:
        discount_label = format_money(discount_value) or f"{discount_value:,.2f}"
        story.append(
            Paragraph(
                f"Discount applied: {discount_label} (reflected in totals)",
                styles["Muted"],
            )
        )
        story.append(Spacer(1, 6))

    story.append(Paragraph(f"In Words: {grand_total_words}", styles["BodySmall"]))
    if terms:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Terms & Conditions</b>", styles["BodySmall"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(html.escape(terms).replace("\n", "<br/>"), styles["BodySmall"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph(prepared_by, styles["BodySmall"]))
    if prepared_title:
        story.append(Paragraph(prepared_title, styles["BodySmall"]))
    if prepared_contact:
        story.append(Paragraph(prepared_contact, styles["BodySmall"]))
    if prepared_email:
        story.append(Paragraph(prepared_email, styles["BodySmall"]))

    doc.build(
        story,
        onFirstPage=_draw_letterhead_background,
        onLaterPages=_draw_letterhead_background,
    )

    return buffer.getvalue()


def _load_letterhead_data_uri(template_choice: Optional[str] = None) -> Optional[str]:
    """Load the configured letterhead into a data URI for the live preview."""

    template_path = _resolve_letterhead_path(template_choice)
    if not template_path:
        st.warning(
            "Letterhead template missing. Upload ps_letterhead.png to see the preview.",
            icon="⚠️",
        )
        return None

    try:
        mime = "image/png" if template_path.suffix.lower() == ".png" else "application/pdf"
        encoded = base64.b64encode(template_path.read_bytes()).decode("utf-8")
        suffix = "#page=1" if mime == "application/pdf" else ""
        return f"data:{mime};base64,{encoded}{suffix}"
    except OSError:
        st.warning(
            "Could not read the letterhead file. Please re-upload it and try again.",
            icon="⚠️",
        )
        return None


def _render_letterhead_preview(
    metadata: dict[str, Optional[str]],
    grand_total: str,
    template_choice: Optional[str] = None,
    items: Optional[list[dict[str, object]]] = None,
    totals: Optional[dict[str, float]] = None,
) -> None:
    metadata = metadata or {}

    resolved_render_id = st.session_state.get("_render_id", 0)
    overlay_id = f"letterhead-preview-{resolved_render_id}"

    template_path = _resolve_letterhead_path(template_choice)
    if not template_path:
        st.warning(
            "Letterhead template missing. Upload ps_letterhead.png to see the preview.",
            icon="⚠️",
        )
        return

    data_uri = _load_letterhead_data_uri(template_choice)
    if not data_uri:
        return

    def _format_currency(value: object) -> str:
        return format_money(value) or f"{_coerce_float(value, 0.0):,.2f}"

    items = items or []
    totals = totals or {}

    grand_total_value = _coerce_float(totals.get("grand_total"), 0.0)
    if grand_total_value <= 0:
        parsed_grand_total = parse_amount(totals.get("grand_total"))
        if parsed_grand_total is not None:
            grand_total_value = parsed_grand_total
    if grand_total_value <= 0:
        parsed_from_label = parse_amount(grand_total)
        if parsed_from_label is not None:
            grand_total_value = parsed_from_label

    if grand_total_value <= 0 and items:
        item_total = 0.0
        for item in items:
            line_total = parse_amount(
                item.get("Total Price, Tk.")
                or item.get("Line total")
                or item.get("line_total")
                or item.get("Amount")
            )
            if line_total is not None:
                item_total += line_total
        if item_total > 0:
            grand_total_value = item_total

    grand_total_label = _format_currency(grand_total_value) if grand_total_value > 0 else "—"
    grand_total_words = (
        format_amount_in_words(grand_total_value)
        if grand_total_value > 0
        else "Add item pricing to calculate totals"
    )

    customer = html.escape(
        str(
            metadata.get("Customer company")
            or metadata.get("Customer / organisation")
            or metadata.get("Hospital / customer")
            or metadata.get("Customer")
            or ""
        )
    )
    raw_contact = (
        metadata.get("Customer contact name")
        or metadata.get("Customer contact")
        or ""
    )
    contact = html.escape(str(raw_contact))
    reference = html.escape(
        str(
            metadata.get("Reference number")
            or metadata.get("Quotation reference")
            or ""
        )
    )
    address = html.escape(str(metadata.get("Customer address") or ""))
    district = html.escape(str(metadata.get("Customer district") or ""))
    date_value = html.escape(str(metadata.get("Date") or ""))
    prepared_by = html.escape(str(metadata.get("Salesperson name") or ""))
    prepared_title = html.escape(str(metadata.get("Salesperson title") or ""))
    prepared_contact = html.escape(str(metadata.get("Salesperson contact") or ""))
    attention_name = html.escape(str(metadata.get("Attention name") or ""))

    line_items = items[:8]
    discount_value = _coerce_float(totals.get("discount_total") if totals else None, 0.0)
    discount_label = (
        format_money(discount_value) or f"{discount_value:,.2f}"
        if discount_value
        else ""
    )
    rows_markup = []
    for idx, item in enumerate(line_items, start=1):
        title = html.escape(
            clean_text(item.get("Description of Generator"))
            or clean_text(item.get("Description"))
            or "Item"
        )
        specs = html.escape(clean_text(item.get("Specs")) or "")
        note = html.escape(clean_text(item.get("Notes")) or clean_text(item.get("note")) or "")
        qty = _coerce_float(item.get("Qty.") or item.get("Quantity"), 0.0)
        qty_display = f"{int(qty)}" if math.isclose(qty, round(qty)) else f"{qty:,.2f}"
        rate_display = _format_currency(
            item.get("Unit Price, Tk.")
            or item.get("Rate")
            or item.get("unit_price")
        )
        total_display = _format_currency(
            item.get("Total Price, Tk.")
            or item.get("Line total")
            or item.get("line_total")
        )
        detail_lines = [title]
        if specs:
            detail_lines.append(f"<span style='color:#475569;'>{specs}</span>")
        if note:
            detail_lines.append(
                f"<span style='color:#94a3b8; font-size:12px;'>{note}</span>"
            )
        detail_block = "<br/>".join(detail_lines)
        rows_markup.append(
            f"<tr>"
            f"<td style='padding:6px; text-align:center; border:1px solid #cbd5e1;'>{idx}</td>"
            f"<td style='padding:6px; border:1px solid #cbd5e1;'>{detail_block}</td>"
            f"<td style='padding:6px; text-align:center; border:1px solid #cbd5e1;'>{qty_display}</td>"
            f"<td style='padding:6px; text-align:right; border:1px solid #cbd5e1;'>{rate_display}</td>"
            f"<td style='padding:6px; text-align:right; border:1px solid #cbd5e1;'>{total_display}</td>"
            f"</tr>"
        )

    total_row = (
        f"<tr style='background:#f8fafc;'>"
        f"<td colspan='3' style='padding:6px; border:1px solid #cbd5e1;'></td>"
        f"<td style='padding:6px; text-align:right; border:1px solid #cbd5e1; font-weight:700;'>Total Amount (Tk.)</td>"
        f"<td style='padding:6px; text-align:right; border:1px solid #cbd5e1; font-weight:700;'>{grand_total_label}</td>"
        f"</tr>"
    )

    address_block = "<br/>".join(filter(None, [customer, contact, address, district]))

    letterhead_style = (
        f"position: relative; width: 940px; min-height: 1100px; border: 1px solid #e5e7eb; border-radius: 12px;"
        f" overflow: hidden; box-shadow: 0 18px 48px rgba(15, 23, 42, 0.14); background: #f8fafc;"
        f" background-image: url('{template_path.as_posix()}'), url('{data_uri}');"
        " background-size: contain; background-repeat: no-repeat; background-position: top center;"
    )

    preview_html = dedent(
        f"""
        <style>
          .letterhead-wrapper[data-overlay-id='{overlay_id}'] {{
            {letterhead_style}
          }}
          .letterhead-wrapper[data-overlay-id='{overlay_id}'] .letterhead-content {{
            position: relative;
            padding: 130px 72px 90px 72px;
            color: #0f172a;
            font-family: 'Arial', sans-serif;
          }}
        </style>
        <div style="margin-top: 1rem; display: flex; justify-content: center;">
          <div class="letterhead-wrapper" data-overlay-id="{overlay_id}">
            <div class="letterhead-content">
              <div style="text-align: right; font-size: 13px; line-height: 1.5;">
                <div><strong>Date:</strong> {date_value or '—'}</div>
                <div><strong>Ref:</strong> {reference or '—'}</div>
              </div>
              <div style="margin-top: 12px; font-size: 13px; line-height: 1.6;">
                <div><strong>Customer</strong></div>
                <div>{address_block or '—'}</div>
              </div>
              <div style="margin-top: 8px; font-size: 13px;"><strong>Attention:</strong> {attention_name or '—'}</div>
              <div style="margin-top: 14px; font-size: 13px; font-weight: 700;">PRICE SCHEDULE</div>
              <table style="width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 12.5px;">
                <thead>
                  <tr style="background: #e2e8f0;">
                    <th style="padding: 6px; border: 1px solid #cbd5e1; text-align: center;">Sl No.</th>
                    <th style="padding: 6px; border: 1px solid #cbd5e1; text-align: left;">Description of Generator</th>
                    <th style="padding: 6px; border: 1px solid #cbd5e1; text-align: center;">Qty.</th>
                    <th style="padding: 6px; border: 1px solid #cbd5e1; text-align: right;">Unit Price, Tk.</th>
                    <th style="padding: 6px; border: 1px solid #cbd5e1; text-align: right;">Total Price, Tk.</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(rows_markup) or '<tr><td colspan="5" style="padding:8px; text-align:center; border:1px solid #cbd5e1; color:#64748b;">Add items to see the price schedule.</td></tr>'}
                  {total_row}
                </tbody>
              </table>
              {f"<div style='margin-top: 8px; font-size: 12px; color:#475569;'>Discount applied: {discount_label} (reflected above)</div>" if discount_label else ''}
              <div style="margin-top: 10px; font-size: 13px;">In Words: {grand_total_words}</div>
              <div style="margin-top: 40px; font-size: 13px; line-height: 1.4;">
                <div>{prepared_by or ''}</div>
                <div>{prepared_title or ''}</div>
                <div>{prepared_contact or ''}</div>
              </div>
            </div>
          </div>
        </div>
        """
    )

    st.markdown(preview_html, unsafe_allow_html=True)
    return preview_html

def _quotation_scope_filter() -> tuple[str, tuple[object, ...]]:
    if current_user_is_admin():
        return "", ()
    uid = current_user_id()
    if uid is None:
        return "WHERE 1=0", ()
    return "WHERE created_by = ?", (uid,)


def _save_quotation_record(conn, payload: dict) -> Optional[int]:
    columns = [
        "reference",
        "quote_date",
        "customer_name",
        "customer_company",
        "customer_address",
        "customer_district",
        "customer_contact",
        "attention_name",
        "attention_title",
        "subject",
        "salutation",
        "introduction",
        "closing",
        "quote_type",
        "total_amount",
        "discount_pct",
        "status",
        "payment_receipt_path",
        "follow_up_status",
        "follow_up_notes",
        "follow_up_date",
        "reminder_label",
        "letter_template",
        "salesperson_name",
        "salesperson_title",
        "salesperson_contact",
        "salesperson_email",
        "document_path",
        "remarks_internal",
        "created_by",
    ]
    placeholders = ",".join("?" for _ in columns)
    values = [payload.get(col) for col in columns]
    try:
        cur = conn.execute(
            f"INSERT INTO quotations ({','.join(columns)}) VALUES ({placeholders})",
            tuple(values),
        )
        conn.execute(
            "UPDATE quotations SET updated_at=datetime('now') WHERE quotation_id=?",
            (cur.lastrowid,),
        )
        conn.commit()
    except sqlite3.Error:
        return None
    return int(cur.lastrowid)


def _upsert_customer_from_manual_quotation(
    conn,
    *,
    name: Optional[str],
    company: Optional[str],
    phone: Optional[str],
    address: Optional[str],
    district: Optional[str],
    reference: Optional[str] = None,
    created_by: Optional[int] = None,
) -> Optional[int]:
    """Insert or backfill a customer captured from a manual quotation entry."""

    customer_name = clean_text(name)
    company_name = clean_text(company)
    phone_number = clean_text(phone)
    street_address = clean_text(address)
    district_label = clean_text(district)
    reference_label = clean_text(reference)

    if not any([customer_name, company_name, phone_number, street_address, district_label]):
        return None

    cursor = conn.cursor()

    def _fetch_existing(query: str, params: tuple[object, ...]):
        return cursor.execute(query, params).fetchone()

    existing = None
    if phone_number:
        existing = _fetch_existing(
            """
            SELECT customer_id, name, company_name, phone, address, delivery_address, remarks
            FROM customers
            WHERE TRIM(IFNULL(phone, '')) = ?
            ORDER BY customer_id DESC
            LIMIT 1
            """,
            (phone_number,),
        )
    if existing is None and company_name:
        existing = _fetch_existing(
            """
            SELECT customer_id, name, company_name, phone, address, delivery_address, remarks
            FROM customers
            WHERE LOWER(IFNULL(company_name, '')) = LOWER(?)
            ORDER BY customer_id DESC
            LIMIT 1
            """,
            (company_name,),
        )
    if existing is None and customer_name:
        existing = _fetch_existing(
            """
            SELECT customer_id, name, company_name, phone, address, delivery_address, remarks
            FROM customers
            WHERE LOWER(IFNULL(name, '')) = LOWER(?)
            ORDER BY customer_id DESC
            LIMIT 1
            """,
            (customer_name,),
        )

    if existing:
        (
            customer_id,
            existing_name,
            existing_company,
            existing_phone,
            existing_address,
            existing_delivery,
            existing_remarks,
        ) = existing

        updates: dict[str, object] = {}
        if not existing_name and customer_name:
            updates["name"] = customer_name
        if not existing_company and company_name:
            updates["company_name"] = company_name
        if not existing_phone and phone_number:
            updates["phone"] = phone_number
        if not existing_address and street_address:
            updates["address"] = street_address
        if not existing_delivery and street_address:
            updates["delivery_address"] = street_address
        if not existing_remarks and district_label:
            updates["remarks"] = f"District: {district_label}"

        if updates:
            set_clause = ", ".join(f"{col}=?" for col in updates)
            cursor.execute(
                f"UPDATE customers SET {set_clause} WHERE customer_id=?",
                (*updates.values(), customer_id),
            )
            conn.commit()
        return customer_id

    remark_parts = []
    if district_label:
        remark_parts.append(f"District: {district_label}")
    if reference_label:
        remark_parts.append(f"Quotation ref: {reference_label}")
    remarks_value = " | ".join(remark_parts) if remark_parts else None

    cursor.execute(
        """
        INSERT INTO customers (name, company_name, phone, address, delivery_address, remarks, created_by, dup_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            customer_name or company_name,
            company_name,
            phone_number,
            street_address,
            street_address,
            remarks_value,
            created_by,
        ),
    )
    conn.commit()
    return cursor.lastrowid

def _persist_quotation_pdf(
    record_id: int, pdf_bytes: bytes, reference: Optional[str]
) -> Optional[str]:
    if not pdf_bytes or record_id is None:
        return None
    ensure_upload_dirs()
    safe_ref = re.sub(r"[^a-zA-Z0-9_-]+", "-", clean_text(reference) or "").strip("-")
    safe_ref = safe_ref or "quotation"
    dest = QUOTATION_RECEIPT_DIR / f"quotation_{record_id}_{safe_ref}.pdf"
    counter = 1
    while dest.exists():
        dest = QUOTATION_RECEIPT_DIR / f"quotation_{record_id}_{safe_ref}_{counter}.pdf"
        counter += 1
    try:
        with open(dest, "wb") as fh:
            fh.write(pdf_bytes)
        return str(dest.relative_to(BASE_DIR))
    except (OSError, ValueError):
        return None


def _update_quotation_records(conn, updates: Iterable[dict[str, object]]) -> dict[str, list[int]]:
    updated: list[int] = []
    locked: list[int] = []
    for entry in updates:
        try:
            quotation_id = int(entry.get("quotation_id"))
        except Exception:
            continue
        cur = conn.execute(
            """
            SELECT status, follow_up_status, follow_up_notes, follow_up_date, reminder_label,
                   payment_receipt_path, reference
            FROM quotations
            WHERE quotation_id=?
            """,
            (quotation_id,),
        )
        row = cur.fetchone()
        if not row:
            continue
        current_status = clean_text(row[0]) or "pending"
        if current_status in {"paid", "rejected"}:
            locked.append(quotation_id)
            continue
        status_value = clean_text(entry.get("status")) or current_status
        follow_up_status = clean_text(entry.get("follow_up_status")) or clean_text(row[1])
        follow_up_notes = clean_text(entry.get("follow_up_notes")) or clean_text(row[2])
        follow_up_date = None
        if status_value != "paid":
            follow_up_date = to_iso_date(entry.get("follow_up_date") or row[3])
        reminder_label = (
            "Payment marked as received; follow-up reminders disabled."
            if status_value == "paid"
            else clean_text(entry.get("reminder_label"))
            or clean_text(row[4])
        )
        receipt_path = clean_text(entry.get("payment_receipt_path")) or clean_text(row[5])
        if status_value == "paid" and not receipt_path:
            locked.append(quotation_id)
            continue
        conn.execute(
            """
            UPDATE quotations
            SET status=?,
                follow_up_status=?,
                follow_up_notes=?,
                follow_up_date=?,
                reminder_label=?,
                payment_receipt_path=COALESCE(?, payment_receipt_path),
                updated_at=datetime('now')
            WHERE quotation_id=?
            """,
            (
                status_value,
                follow_up_status,
                follow_up_notes,
                follow_up_date,
                reminder_label,
                receipt_path,
                quotation_id,
            ),
        )
        updated.append(quotation_id)
        reference_label = clean_text(row[6]) or f"Quotation #{quotation_id}"
        if status_value != current_status:
            receipt_note = " with receipt" if receipt_path and status_value == "paid" else ""
            log_activity(
                conn,
                event_type="quotation_updated",
                description=f"{reference_label} marked as {status_value}{receipt_note}",
                entity_type="quotation",
                entity_id=quotation_id,
            )
    conn.commit()
    return {"updated": updated, "locked": locked}


def _render_quotation_section(conn, *, render_id: Optional[int] = None):
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

    user = get_current_user()
    salesperson_profile = {
        "name": clean_text(user.get("username")) or "",
        "title": clean_text(user.get("title")) or "",
        "phone": clean_text(user.get("phone")) or "",
        "email": clean_text(user.get("email")) or "",
    }

    uid = current_user_id()
    if uid is not None:
        try:
            profile_df = df_query(
                conn,
                "SELECT username, phone, email, title FROM users WHERE user_id=?",
                (uid,),
            )
            if not profile_df.empty:
                row = profile_df.iloc[0]
                salesperson_profile = {
                    "name": clean_text(row.get("username")) or salesperson_profile["name"],
                    "title": clean_text(row.get("title")) or salesperson_profile["title"],
                    "phone": clean_text(row.get("phone")) or salesperson_profile["phone"],
                    "email": clean_text(row.get("email")) or salesperson_profile["email"],
                }
        except Exception:
            pass

    salesperson_seed = salesperson_profile["name"]
    customer_df = df_query(
        conn,
        """
        SELECT customer_id, name, company_name, address, delivery_address, phone, COALESCE(delivery_address, address) AS district
        FROM customers
        ORDER BY LOWER(COALESCE(name, company_name, phone, 'customer'))
        LIMIT 200
        """,
    )
    autofill_options = [None]
    autofill_labels = {None: "Manual entry"}
    autofill_records: dict[int, dict[str, object]] = {}
    if not customer_df.empty:
        for _, row in customer_df.iterrows():
            try:
                cid = int(row.get("customer_id"))
            except Exception:
                continue
            label_parts = [clean_text(row.get("name")) or clean_text(row.get("company_name"))]
            phone_val = clean_text(row.get("phone"))
            if phone_val:
                label_parts.append(phone_val)
            autofill_options.append(cid)
            autofill_labels[cid] = " • ".join(part for part in label_parts if part)
            autofill_records[cid] = row.to_dict()

    status_choices = ["pending", "paid", "rejected"]
    follow_up_presets = {
        "In 3 days": 3,
        "In 1 week": 7,
        "In 2 weeks": 14,
        "Custom date": None,
    }

    st.session_state["quotation_autofill_customer"] = None
    st.markdown("### Upload to auto-fill")
    st.caption(
        "Upload a quotation (PDF, DOCX, or TXT) to detect customer info, contact details, and line items automatically."
    )
    prefill_upload = st.file_uploader(
        "Quotation file",
        type=["pdf", "doc", "docx", "txt"],
        key="quotation_prefill_upload",
    )

    if prefill_upload:
        text, warnings = _extract_text_from_quotation_upload(prefill_upload)
        for warning in warnings:
            st.warning(warning)
        saved_prefill = save_uploaded_file(
            prefill_upload,
            QUOTATION_DOCS_DIR,
            filename=prefill_upload.name or "quotation_upload",
            allowed_extensions={".pdf", ".doc", ".docx", ".txt"},
            default_extension=".pdf",
        )
        if saved_prefill:
            try:
                st.session_state["quotation_document_path"] = str(
                    saved_prefill.relative_to(BASE_DIR)
                )
            except ValueError:
                st.session_state["quotation_document_path"] = str(saved_prefill)
        updates = _extract_quotation_metadata(text)
        detected_items = updates.pop("_detected_items", None)
        if updates:
            st.session_state.update(updates)
        if detected_items:
            st.session_state["quotation_item_rows"] = detected_items
        if updates or detected_items:
            st.success("Quotation fields auto-filled from the uploaded file.")

    with st.form("quotation_form"):
        st.markdown("### Quotation details")
        basic_cols = st.columns((1, 1))
        with basic_cols[0]:
            quotation_date = st.date_input(
                "Quotation date",
                value=st.session_state.get("quotation_date") or default_date,
                key="quotation_date",
            )
            customer_company = st.text_input(
                "Customer name",
                value=st.session_state.get("quotation_company_name", ""),
                key="quotation_company_name",
            )
            customer_contact_name = st.text_input(
                "Attention / contact person",
                value=st.session_state.get("quotation_customer_contact_name", ""),
                key="quotation_customer_contact_name",
            )
        with basic_cols[1]:
            reference_value = st.text_input(
                "Reference / Quotation #",
                value=st.session_state.get("quotation_reference", ""),
                key="quotation_reference",
            )
            subject_line = st.text_input(
                "Product or project subject",
                value=st.session_state.get("quotation_subject", ""),
                key="quotation_subject",
            )
            customer_contact = st.text_input(
                "Customer contact details (phone / email)",
                value=st.session_state.get("quotation_customer_contact", ""),
                key="quotation_customer_contact",
            )
            attention_name = st.text_input(
                "Attention line (optional)",
                value=st.session_state.get("quotation_attention_name", "")
                or st.session_state.get("quotation_customer_contact_name", ""),
                key="quotation_attention_name",
            )

        customer_address = st.text_area(
            "Delivery location / address",
            value=st.session_state.get("quotation_customer_address", ""),
            key="quotation_customer_address",
        )

        st.markdown("### Product / service details")
        item_rows = st.session_state.get("quotation_item_rows") or _default_quotation_items()
        items_df = pd.DataFrame(item_rows)
        for column in ["description", "quantity", "rate", "total_price"]:
            if column not in items_df.columns:
                default_value = 0.0 if column != "description" else ""
                items_df[column] = default_value
        items_df = items_df[["description", "quantity", "rate", "total_price"]]
        edited_df = st.data_editor(
            items_df,
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            key="quotation_items_editor",
            column_config={
                "description": st.column_config.TextColumn(
                    "Description of Generator", required=True
                ),
                "quantity": st.column_config.NumberColumn(
                    "Qty.", min_value=0.0, step=1.0, format="%d"
                ),
                "rate": st.column_config.NumberColumn(
                    "Unit Price, Tk.", min_value=0.0, step=100.0, format="%.2f"
                ),
                "total_price": st.column_config.NumberColumn(
                    "Total Price, Tk.",
                    min_value=0.0,
                    step=100.0,
                    format="%.2f",
                    help="Override the line total when it differs from quantity × unit price.",
                ),
            },
        )
        if isinstance(edited_df, pd.DataFrame):
            st.session_state["quotation_item_rows"] = (
                edited_df.fillna("").to_dict("records")
            )

        manual_total_value = st.number_input(
            "Manual total amount (Tk.)",
            min_value=0.0,
            step=1000.0,
            format="%.2f",
            value=_coerce_float(st.session_state.get("quotation_manual_total"), 0.0),
            key="quotation_manual_total",
            help="Optional override if the quoted amount differs from the calculated line totals.",
        )

        terms_notes = st.text_area(
            "Special notes / terms & conditions",
            value=st.session_state.get("quotation_terms", ""),
            key="quotation_terms",
        )

        st.markdown("#### Admin follow-up")
        follow_cols = st.columns((1, 1))
        follow_statuses = ["Pending", "Hot", "Possible", "Closed"]
        saved_follow_status = clean_text(st.session_state.get("quotation_follow_up_status"))
        follow_status_default = saved_follow_status.title() if saved_follow_status else "Pending"
        follow_status_default = (
            follow_status_default if follow_status_default in follow_statuses else "Pending"
        )
        with follow_cols[0]:
            follow_up_status = st.selectbox(
                "Follow-up status",
                follow_statuses,
                index=follow_statuses.index(follow_status_default),
                key="quotation_follow_up_status",
                help="Visible to admins for tracking next steps.",
            )
            status_value = st.selectbox(
                "Quotation status",
                status_choices,
                index=status_choices.index(clean_text(st.session_state.get("quotation_status")) or "pending")
                if clean_text(st.session_state.get("quotation_status")) in status_choices
                else 0,
                key="quotation_status",
            )
        with follow_cols[1]:
            follow_up_choice = st.selectbox(
                "Reminder preset",
                list(follow_up_presets.keys()),
                index=1,
                key="quotation_follow_up_choice",
            )
            enable_follow_date = st.checkbox(
                "Set follow-up date",
                value=bool(st.session_state.get("quotation_follow_up_date")),
                key="quotation_follow_up_date_toggle",
            )
            follow_up_date_value = None
            if enable_follow_date:
                follow_up_date_value = st.date_input(
                    "Next follow-up date",
                    value=st.session_state.get("quotation_follow_up_date")
                    or datetime.now().date(),
                    key="quotation_follow_up_date",
                )
        receipt_upload = None
        existing_receipt_path = st.session_state.get("quotation_payment_receipt_path")
        if status_value == "paid":
            receipt_upload = st.file_uploader(
                "Upload payment receipt (required for paid status)",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key="quotation_receipt_upload",
            )
            if receipt_upload:
                safe_ref = _sanitize_path_component(reference_value) or f"quotation_{quotation_date.strftime('%Y%m%d')}"
                saved_receipt = store_payment_receipt(
                    receipt_upload, identifier=f"{safe_ref}_receipt"
                )
                if saved_receipt:
                    st.session_state["quotation_payment_receipt_path"] = saved_receipt
            elif existing_receipt_path:
                st.caption("Using the previously uploaded receipt for this quotation.")
            else:
                st.caption("Attach the receipt before saving a paid quotation.")
        follow_up_notes = st.text_area(
            "Follow-up remarks for admins",
            value=st.session_state.get("quotation_follow_up_notes", ""),
            key="quotation_follow_up_notes",
            help="Internal notes that help admins continue the conversation.",
        )

        form_actions = st.columns((1, 1))
        submit = form_actions[0].form_submit_button("Save quotation", type="primary")
        reset = form_actions[1].form_submit_button("Reset form")

    template_choice = "Default letterhead"
    quote_type = st.session_state.get("quotation_quote_type", "Standard")
    default_discount = _coerce_float(
        st.session_state.get("quotation_discount_default"), 0.0
    )
    customer_district = st.session_state.get("quotation_customer_district", "")
    attention_title = st.session_state.get("quotation_attention_title", "")
    admin_notes = terms_notes
    follow_up_choice = st.session_state.get("quotation_follow_up_choice")
    status_value = clean_text(st.session_state.get("quotation_status")) or "pending"
    salesperson_title = salesperson_profile.get("title", "")
    salesperson_contact = salesperson_profile.get("phone", "")
    salesperson_email = salesperson_profile.get("email", "")
    prepared_by = (
        st.session_state.get("quotation_prepared_by") or salesperson_profile.get("name", "")
    )
    salutation = st.session_state.get("quotation_salutation", "")
    intro_text = st.session_state.get("quotation_introduction", "")
    closing_text = st.session_state.get("quotation_closing", "")
    attention_name = st.session_state.get("quotation_attention_name", attention_name)
    if reset:
        _reset_quotation_form_state()
        st.session_state["quotation_feedback"] = (
            "info",
            "Quotation form reset to defaults.",
        )
        _safe_rerun()

    if submit:
        prepared_items = [dict(item) for item in st.session_state.get("quotation_item_rows", [])]
        for item in prepared_items:
            if item.get("discount") in (None, ""):
                item["discount"] = default_discount

        items_clean, totals_data = normalize_quotation_items(prepared_items)

        reminder_days = follow_up_presets.get(follow_up_choice)
        follow_up_date = follow_up_date_value
        if reminder_days is not None:
            follow_up_date = quotation_date + timedelta(days=reminder_days)
        follow_up_iso = to_iso_date(follow_up_date) if follow_up_date else None
        follow_up_label = format_period_range(follow_up_iso, follow_up_iso) if follow_up_iso else ""
        reminder_label = None
        if follow_up_label and reminder_days is not None:
            reminder_label = f"Reminder scheduled in {reminder_days} days on {follow_up_label}."
        receipt_path = st.session_state.get("quotation_payment_receipt_path")
        if status_value == "paid" and not receipt_path:
            st.error("Upload a payment receipt before saving a paid quotation.")
            return
        manual_total_override = max(
            _coerce_float(manual_total_value, 0.0),
            0.0,
        )
        grand_total_value = (
            manual_total_override if manual_total_override > 0 else totals_data["grand_total"]
        )

        customer_contact_combined = dedupe_join(
            [st.session_state.get("quotation_customer_contact"), attention_name], " / "
        )

        metadata = OrderedDict()
        metadata["Reference number"] = reference_value
        metadata["Date"] = quotation_date.strftime(DATE_FMT)
        metadata["Customer contact name"] = customer_contact_name
        metadata["Customer company"] = customer_company
        metadata["Customer address"] = customer_address
        metadata["Customer contact"] = customer_contact_combined
        metadata["Attention name"] = attention_name
        metadata["Notes / terms"] = terms_notes
        metadata["Salesperson name"] = prepared_by
        metadata["Salesperson title"] = salesperson_title
        metadata["Salesperson contact"] = salesperson_contact
        metadata["Salesperson email"] = salesperson_email
        metadata["Total amount (BDT)"] = grand_total_value
        if manual_total_override > 0:
            metadata["Manual total override (BDT)"] = manual_total_override
        metadata["Status"] = status_value.title()

        totals_rows = [("Gross amount", totals_data["gross_total"])]
        if totals_data["discount_total"]:
            totals_rows.append(("Discount total", totals_data["discount_total"]))
        if manual_total_override > 0 and not math.isclose(
            manual_total_override, totals_data["grand_total"]
        ):
            totals_rows.append(("Manual total override", manual_total_override))
        totals_rows.append(("Grand total", grand_total_value))

        workbook_items = [item.copy() for item in items_clean]
        workbook_bytes = _build_quotation_workbook(
            metadata=metadata,
            items=workbook_items,
            totals=totals_rows,
        )

        display_df = pd.DataFrame(workbook_items)

        column_order = [
            col
            for col in [
                "Sl No.",
                "Description of Generator",
                "Qty.",
                "Unit Price, Tk.",
                "Total Price, Tk.",
                "Notes",
            ]
            if col in display_df.columns
        ]
        if column_order:
            display_df = display_df[column_order]

        def _format_quantity_display(value: object) -> str:
            amount = _coerce_float(value, 0.0)
            if math.isclose(amount, round(amount)):
                return f"{int(round(amount))}"
            return f"{amount:,.2f}"

        money_columns = ["Unit Price, Tk.", "Total Price, Tk."]
        for col in money_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda value: format_money(value) or f"{_coerce_float(value, 0.0):,.2f}"
                )
        for qty_col in ["Qty.", "Quantity"]:
            if qty_col in display_df.columns:
                display_df[qty_col] = display_df[qty_col].apply(_format_quantity_display)
        display_df = display_df.fillna("")

        base_filename = clean_text(reference_value) or f"quotation_{quotation_date.strftime('%Y%m%d')}"
        safe_name = _sanitize_path_component(base_filename)
        if not safe_name:
            safe_name = f"quotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename = f"{safe_name}.xlsx"

        payload = {
            "reference": reference_value,
            "quote_date": quotation_date.isoformat(),
            "customer_name": customer_contact_name,
            "customer_company": customer_company,
            "customer_address": customer_address,
            "customer_district": customer_district,
            "customer_contact": customer_contact_combined,
            "attention_name": attention_name,
            "attention_title": attention_title,
            "subject": subject_line,
            "salutation": salutation,
            "introduction": intro_text,
            "closing": closing_text,
            "quote_type": quote_type,
            "total_amount": grand_total_value,
            "discount_pct": default_discount,
            "status": status_value,
            "payment_receipt_path": receipt_path,
            "follow_up_status": follow_up_status,
            "follow_up_notes": follow_up_notes,
            "follow_up_date": follow_up_iso,
            "reminder_label": reminder_label,
            "letter_template": template_choice,
            "salesperson_name": prepared_by,
            "salesperson_title": salesperson_title,
            "salesperson_contact": salesperson_contact,
            "salesperson_email": salesperson_email,
            "document_path": st.session_state.get("quotation_document_path"),
            "remarks_internal": terms_notes,
            "created_by": current_user_id(),
        }
        record_id = _save_quotation_record(conn, payload)

        st.session_state[result_key] = {
            "display": display_df,
            "metadata_items": list(metadata.items()),
            "totals_rows": totals_rows,
            "grand_total": grand_total_value,
            "metadata": metadata,
            "excel_bytes": workbook_bytes,
            "filename": filename,
            "record_id": record_id,
            "reminder_label": reminder_label,
            "letter_template": template_choice,
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


def _render_quotation_management(conn):
    st.markdown("### Quotation tracker")
    scope_clause, scope_params = _quotation_scope_filter()
    quotes_df = df_query(
        conn,
        dedent(
            f"""
            SELECT q.quotation_id, q.reference, q.quote_date, q.customer_company, q.customer_contact,
                   q.total_amount, q.status, q.follow_up_status, q.follow_up_notes, q.follow_up_date,
                   q.reminder_label, q.payment_receipt_path, q.created_by, COALESCE(u.username, '(user)') AS created_by_name
            FROM quotations q
            LEFT JOIN users u ON u.user_id = q.created_by
            {scope_clause}
            ORDER BY datetime(q.quote_date) DESC, q.quotation_id DESC
            LIMIT 50
            """
        ),
        scope_params,
    )

    if quotes_df.empty:
        st.info("No quotations recorded yet. Create a quotation above to start tracking.")
        return

    def _as_editable_date(value: object) -> Optional[date]:
        if value in (None, "", "nan", "NaT"):
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
        except Exception:
            return None
        if isinstance(parsed, pd.DatetimeIndex):
            parsed = parsed[0] if len(parsed) else None
        if parsed is None or pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            parsed = parsed.to_pydatetime()
        if isinstance(parsed, datetime):
            return parsed.date()
        return None

    quotes_df = quotes_df.copy()
    quotes_df["status"] = quotes_df["status"].apply(lambda val: clean_text(val) or "pending")
    quotes_df["follow_up_date"] = quotes_df.get("follow_up_date", pd.Series(dtype=object)).apply(
        _as_editable_date
    )
    quotes_df = fmt_dates(quotes_df, ["quote_date"])
    quotes_df["total_amount"] = quotes_df["total_amount"].apply(
        lambda value: format_money(value) or f"{_coerce_float(value, 0.0):,.2f}"
    )

    closed_statuses = {"paid", "rejected"}
    locked_mask = quotes_df["status"].str.lower().isin(closed_statuses)
    editable_df = quotes_df[~locked_mask].copy()
    locked_df = quotes_df[locked_mask].copy()

    tracker_state_key = "quotation_tracker_rows"

    def _normalize_tracker_rows(rows: Iterable[dict[str, object]]):
        normalized: list[dict[str, object]] = []
        for row in rows:
            normalized_row: dict[str, object] = {}
            for col in editable_df.columns:
                value = row.get(col)
                if col == "follow_up_date":
                    value = _as_editable_date(value)
                normalized_row[col] = value
            normalized.append(normalized_row)
        return normalized

    current_records = _normalize_tracker_rows(editable_df.to_dict("records"))
    tracker_state: list[dict[str, object]] = st.session_state.get(tracker_state_key, [])
    current_ids = {row.get("quotation_id") for row in current_records}
    state_ids = {row.get("quotation_id") for row in tracker_state}
    if (
        not tracker_state
        or current_ids != state_ids
        or len(tracker_state) != len(current_records)
    ):
        tracker_state = current_records
        st.session_state[tracker_state_key] = tracker_state

    tracker_df = pd.DataFrame(tracker_state)
    tracker_df = tracker_df[[col for col in editable_df.columns if col in tracker_df.columns]]
    tracker_df = tracker_df.reindex(columns=editable_df.columns)

    edit_config = {
        "status": st.column_config.TextColumn("Status", disabled=True),
        "follow_up_status": st.column_config.TextColumn(
            "Follow-up status",
            help="Example: Possible, Hot, Cold, Closed",
        ),
        "follow_up_notes": st.column_config.TextColumn("Follow-up notes"),
        "follow_up_date": st.column_config.DateColumn(
            "Follow-up date", format="DD-MM-YYYY"
        ),
        "reminder_label": st.column_config.TextColumn("Reminder"),
        "reference": st.column_config.TextColumn("Reference"),
        "customer_company": st.column_config.TextColumn("Customer"),
        "customer_contact": st.column_config.TextColumn("Contact"),
        "total_amount": st.column_config.TextColumn("Total amount (BDT)"),
        "quote_date": st.column_config.TextColumn("Quote date"),
        "payment_receipt_path": st.column_config.TextColumn(
            "Receipt proof",
            help="Relative path to the uploaded payment receipt, if any.",
            disabled=True,
        ),
        "created_by_name": st.column_config.TextColumn("Created by", disabled=True),
    }

    st.caption(
        "Edit follow-up status, notes, and dates directly in the table, then press **Save quotation updates** to persist your changes."
    )

    edited_records: list[dict[str, object]] = []
    if not editable_df.empty:
        edited = st.data_editor(
            tracker_df,
            hide_index=True,
            use_container_width=True,
            column_config=edit_config,
            key="quotation_tracker_editor",
        )
        if isinstance(edited, pd.DataFrame):
            edited_records = _normalize_tracker_rows(edited.to_dict("records"))
            st.session_state[tracker_state_key] = edited_records
    else:
        st.info("No pending quotations to update.")

    save_disabled = editable_df.empty or not edited_records
    if st.button(
        "Save quotation updates",
        key="quotation_tracker_save",
        disabled=save_disabled,
    ) and edited_records:
        result = _update_quotation_records(conn, edited_records)
        updated_count = len(result.get("updated", []))
        locked_count = len(result.get("locked", []))
        if updated_count:
            st.success(f"Updated {updated_count} quotation(s).")
            st.toast("Quotation tracker updated", icon="✅")
        if locked_count:
            st.info("Some quotations are locked because they are marked as paid or rejected.")

    st.markdown("#### Status updates with receipt")
    status_candidates = editable_df.to_dict("records") if not editable_df.empty else []
    if not status_candidates:
        st.caption("All tracked quotations are already marked as paid or rejected.")
    else:
        status_labels: dict[int, str] = {}
        for record in status_candidates:
            try:
                qid = int(record.get("quotation_id"))
            except Exception:
                continue
            reference = clean_text(record.get("reference")) or f"Quotation #{qid}"
            customer = clean_text(record.get("customer_company")) or clean_text(record.get("customer_contact"))
            status_labels[qid] = " • ".join(part for part in [reference, customer] if part)
        selectable_ids = list(status_labels.keys())
        selected_id = st.selectbox(
            "Pick a quotation to update",
            selectable_ids,
            format_func=lambda val: status_labels.get(val, f"Quotation #{val}"),
            key="quotation_status_update_id",
        )
        selected_record = next((row for row in status_candidates if int(row.get("quotation_id")) == int(selected_id)), {})
        current_status = clean_text(selected_record.get("status")) or "pending"
        new_status = st.selectbox(
            "New status",
            ["pending", "paid", "rejected"],
            index=["pending", "paid", "rejected"].index(current_status)
            if current_status in {"pending", "paid", "rejected"}
            else 0,
            key="quotation_status_update_choice",
        )
        receipt_upload = None
        existing_receipt = clean_text(selected_record.get("payment_receipt_path"))
        if new_status == "paid":
            receipt_upload = st.file_uploader(
                "Upload payment receipt (required when marking as paid)",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key=f"quotation_status_receipt_{selected_id}",
            )
            if not (receipt_upload or existing_receipt):
                st.caption("Attach a receipt to lock this quotation as paid.")

        if st.button("Apply status update", key="quotation_status_update_button"):
            if current_status in closed_statuses:
                st.warning("This quotation is locked and cannot be changed.")
                return
            if new_status == current_status:
                st.info("No changes to apply.")
                return
            if new_status == "paid" and not (receipt_upload or existing_receipt):
                st.error("Upload a receipt before marking the quotation as paid.")
                return
            safe_ref = _sanitize_path_component(clean_text(selected_record.get("reference")) or f"quotation_{selected_id}")
            receipt_path = existing_receipt
            if receipt_upload:
                receipt_path = store_payment_receipt(
                    receipt_upload,
                    identifier=f"{safe_ref}_receipt",
                )
            update_result = _update_quotation_records(
                conn,
                [
                    {
                        "quotation_id": selected_id,
                        "status": new_status,
                        "payment_receipt_path": receipt_path,
                    }
                ],
            )
            if update_result.get("updated"):
                st.success("Quotation status updated and locked.")
                _safe_rerun()
            elif update_result.get("locked"):
                st.info("This quotation was already locked and was not changed.")

    if not locked_df.empty:
        st.markdown("#### Paid / rejected quotations (locked)")
        locked_df = locked_df.rename(
            columns={
                "payment_receipt_path": "receipt_path",
                "created_by_name": "Owner",
            }
        )
        locked_df["Receipt"] = locked_df["receipt_path"].apply(
            lambda path: "📎" if clean_text(path) else ""
        )
        st.dataframe(
            locked_df.drop(columns=["receipt_path"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Update saved quotation details")
    detail_labels: dict[int, str] = {}
    for _, row in quotes_df.iterrows():
        try:
            detail_id = int(row.get("quotation_id"))
        except Exception:
            continue
        reference = clean_text(row.get("reference")) or f"Quotation #{detail_id}"
        customer = clean_text(row.get("customer_company")) or clean_text(
            row.get("customer_contact")
        )
        detail_labels[detail_id] = " • ".join(
            part for part in [reference, customer] if part
        )

    detail_choices = list(detail_labels.keys())
    if not detail_choices:
        st.info("No quotations available to edit.")
        return

    selected_detail_id = st.selectbox(
        "Select a quotation",
        detail_choices,
        format_func=lambda val: detail_labels.get(val, f"Quotation #{val}"),
        key="quotation_detail_select",
    )

    selected_row = quotes_df[
        quotes_df["quotation_id"] == selected_detail_id
    ].iloc[0]
    selected_status = clean_text(selected_row.get("status")) or "pending"
    existing_receipt = clean_text(selected_row.get("payment_receipt_path"))
    follow_up_status_value = clean_text(selected_row.get("follow_up_status")) or ""
    follow_up_notes_value = clean_text(selected_row.get("follow_up_notes")) or ""
    follow_up_date_seed = selected_row.get("follow_up_date")
    if isinstance(follow_up_date_seed, pd.Timestamp):
        follow_up_date_seed = follow_up_date_seed.date()

    col_left, col_right = st.columns(2)
    with col_left:
        follow_up_status_input = st.text_input(
            "Follow-up status",
            value=follow_up_status_value,
            key="quotation_detail_follow_up_status",
        )
        clear_follow_up_date = st.checkbox(
            "No follow-up date",
            value=follow_up_date_seed is None,
            key="quotation_detail_clear_date",
        )
        follow_up_date_input: Optional[date] = None
        if not clear_follow_up_date:
            follow_up_date_input = st.date_input(
                "Follow-up date",
                value=follow_up_date_seed or date.today(),
                key="quotation_detail_follow_up_date",
                format="DD-MM-YYYY",
            )
    with col_right:
        follow_up_notes_input = st.text_area(
            "Follow-up notes",
            value=follow_up_notes_value,
            key="quotation_detail_follow_up_notes",
        )
        receipt_upload = None
        if selected_status == "paid":
            receipt_upload = st.file_uploader(
                "Attach receipt for this paid quotation",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key=f"quotation_detail_receipt_{selected_detail_id}",
            )
            if not (receipt_upload or existing_receipt):
                st.caption("Upload a receipt to keep this paid quotation locked.")

    if st.button("Update quotation details", type="primary", key="quotation_detail_save"):
        receipt_path = existing_receipt
        if selected_status == "paid":
            if receipt_upload:
                safe_ref = _sanitize_path_component(
                    clean_text(selected_row.get("reference"))
                    or f"quotation_{selected_detail_id}"
                )
                receipt_path = store_payment_receipt(
                    receipt_upload, identifier=f"{safe_ref}_receipt"
                )
            if not receipt_path:
                st.error("Upload a receipt before saving a paid quotation.")
                return

        follow_up_iso = to_iso_date(follow_up_date_input) if follow_up_date_input else None
        reminder_label = (
            format_period_range(follow_up_iso, follow_up_iso)
            if follow_up_iso
            else clean_text(selected_row.get("reminder_label"))
        )

        conn.execute(
            """
            UPDATE quotations
               SET follow_up_status=?,
                   follow_up_notes=?,
                   follow_up_date=?,
                   reminder_label=?,
                   payment_receipt_path=COALESCE(?, payment_receipt_path),
                   updated_at=datetime('now')
             WHERE quotation_id=?
            """,
            (
                clean_text(follow_up_status_input) or None,
                clean_text(follow_up_notes_input) or None,
                follow_up_iso,
                reminder_label,
                receipt_path,
                selected_detail_id,
            ),
        )
        conn.commit()
        st.success("Quotation details updated.")
        _safe_rerun()


def advanced_search_page(conn):
    st.subheader("🔎 Advanced Search")
    if not current_user_is_admin():
        st.warning("Advanced filters are available to admins only.")
        return

    search_text = st.text_input(
        "Keyword search",
        key="advanced_search_keyword",
        help="Search across quotations, customers, delivery orders, services, and maintenance logs.",
    )
    date_window = st.date_input(
        "Date window",
        value=(date.today() - timedelta(days=30), date.today()),
        help="Filter results by creation or activity date.",
        key="advanced_search_dates",
    )
    min_amount = st.number_input(
        "Minimum amount (for quotations)",
        min_value=0.0,
        step=100.0,
        format="%.2f",
        key="advanced_search_min_amount",
    )
    record_types = [
        "Quotations",
        "Services",
        "Maintenance",
        "Delivery orders",
        "Customers",
    ]
    selected_types = st.multiselect(
        "Record types",
        record_types,
        default=record_types,
        key="advanced_search_types",
    )

    staff_df = df_query(conn, "SELECT user_id, username FROM users ORDER BY LOWER(username)")
    staff_map = {
        int(row["user_id"]): clean_text(row.get("username")) or f"User #{int(row['user_id'])}"
        for _, row in staff_df.iterrows()
    }
    staff_choices = list(staff_map.keys())
    staff_filter = st.multiselect(
        "Staff filter",
        staff_choices,
        format_func=lambda uid: staff_map.get(uid, f"User #{uid}"),
        key="advanced_search_staff",
    )

    start_iso = end_iso = None
    if isinstance(date_window, (list, tuple)) and len(date_window) == 2:
        start_iso = to_iso_date(date_window[0])
        end_iso = to_iso_date(date_window[1])

    results: list[dict[str, object]] = []

    def _apply_date_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if df.empty or column not in df.columns or not start_iso or not end_iso:
            return df
        df[column] = pd.to_datetime(df[column], errors="coerce")
        start_dt = pd.to_datetime(start_iso)
        end_dt = pd.to_datetime(end_iso) + pd.Timedelta(days=1)
        return df[(df[column] >= start_dt) & (df[column] < end_dt)]

    def _append_results(df: pd.DataFrame, type_label: str, date_col: str, build_details):
        if df.empty:
            return
        filtered = _apply_date_filter(df, date_col)
        for row in filtered.to_dict("records"):
            staff_raw = row.get("created_by")
            staff_id = (
                int(staff_raw)
                if staff_raw is not None and not pd.isna(staff_raw)
                else None
            )
            if staff_filter and staff_id not in staff_filter:
                continue
            details = build_details(row)
            search_blob = " ".join(str(val) for val in details.values() if val)
            results.append(
                {
                    "Type": type_label,
                    "Title": details.get("title"),
                    "Details": details.get("details"),
                    "Date": details.get("date"),
                    "Staff": staff_map.get(staff_id),
                    "Amount": details.get("amount"),
                    "Status": details.get("status"),
                    "Attachment": details.get("attachment"),
                    "_search": search_blob.lower(),
                }
            )

    if "Quotations" in selected_types:
        quotes_df = df_query(
            conn,
            dedent(
                """
                SELECT q.quotation_id,
                       q.reference,
                       q.customer_company,
                       q.quote_date,
                       q.total_amount,
                       q.status,
                       q.document_path,
                       q.created_by,
                       q.salesperson_name,
                       u.username
                FROM quotations q
                LEFT JOIN users u ON u.user_id = q.created_by
                ORDER BY datetime(q.quote_date) DESC, q.quotation_id DESC
                LIMIT 200
                """
            ),
        )
        if not quotes_df.empty:
            if min_amount > 0:
                quotes_df = quotes_df[quotes_df["total_amount"].apply(lambda v: _coerce_float(v, 0.0) >= min_amount)]
            _append_results(
                quotes_df,
                "Quotation",
                "quote_date",
                lambda row: {
                    "title": clean_text(row.get("reference")) or clean_text(row.get("customer_company")),
                    "details": f"{clean_text(row.get('customer_company')) or '(customer)'} • {clean_text(row.get('salesperson_name')) or 'Sales'}",
                    "date": row.get("quote_date"),
                    "amount": _coerce_float(row.get("total_amount"), 0.0),
                    "status": clean_text(row.get("status")),
                    "attachment": clean_text(row.get("document_path")),
                },
            )

    if "Services" in selected_types:
        service_df = df_query(
            conn,
            """
            SELECT service_id, description, service_start_date, service_end_date, service_product_info, status
            FROM services
            ORDER BY datetime(COALESCE(service_start_date, service_end_date)) DESC, service_id DESC
            LIMIT 200
            """,
        )
        _append_results(
            service_df,
            "Service",
            "service_start_date",
            lambda row: {
                "title": clean_text(row.get("description")) or f"Service #{row.get('service_id')}",
                "details": clean_text(row.get("service_product_info")),
                "date": row.get("service_start_date") or row.get("service_end_date"),
                "status": clean_text(row.get("status")),
                "amount": None,
                "attachment": None,
            },
        )

    if "Maintenance" in selected_types:
        maintenance_df = df_query(
            conn,
            """
            SELECT maintenance_id, description, maintenance_start_date, maintenance_end_date, maintenance_product_info, status
            FROM maintenance_records
            ORDER BY datetime(COALESCE(maintenance_start_date, maintenance_end_date)) DESC, maintenance_id DESC
            LIMIT 200
            """,
        )
        _append_results(
            maintenance_df,
            "Maintenance",
            "maintenance_start_date",
            lambda row: {
                "title": clean_text(row.get("description")) or f"Maintenance #{row.get('maintenance_id')}",
                "details": clean_text(row.get("maintenance_product_info")),
                "date": row.get("maintenance_start_date") or row.get("maintenance_end_date"),
                "status": clean_text(row.get("status")),
                "amount": None,
                "attachment": None,
            },
        )

    if "Delivery orders" in selected_types:
        do_df = df_query(
            conn,
            """
            SELECT do_number, description, sales_person, remarks, created_at, created_by, total_amount
            FROM delivery_orders
            WHERE COALESCE(record_type, 'delivery_order') = 'delivery_order'
            ORDER BY datetime(created_at) DESC
            LIMIT 200
            """,
        )
        _append_results(
            do_df,
            "Delivery order",
            "created_at",
            lambda row: {
                "title": clean_text(row.get("do_number")),
                "details": clean_text(row.get("description")) or clean_text(row.get("remarks")),
                "date": row.get("created_at"),
                "status": clean_text(row.get("sales_person")),
                "amount": format_money(row.get("total_amount")) or row.get("total_amount"),
                "attachment": None,
            },
        )

    if "Customers" in selected_types:
        customer_df = df_query(
            conn,
            """
            SELECT name, company_name, phone, address, created_at, created_by
            FROM customers
            ORDER BY datetime(created_at) DESC
            LIMIT 200
            """,
        )
        _append_results(
            customer_df,
            "Customer",
            "created_at",
            lambda row: {
                "title": clean_text(row.get("name")) or clean_text(row.get("company_name")),
                "details": clean_text(row.get("phone")) or clean_text(row.get("address")),
                "date": row.get("created_at"),
                "status": None,
                "amount": None,
                "attachment": None,
            },
        )

    if not results:
        st.info("No records found for the selected filters.")
        return

    results_df = pd.DataFrame(results)
    if search_text:
        needle = re.escape(search_text.lower())
        results_df = results_df[results_df["_search"].str.contains(needle, regex=True, na=False)]

    display_cols = ["Type", "Title", "Details", "Date", "Staff", "Amount", "Status", "Attachment"]
    results_df = results_df.fillna("")
    st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

    csv_bytes = results_df.drop(columns=["_search"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results",
        data=csv_bytes,
        file_name="advanced_search.csv",
        mime="text/csv",
        key="advanced_search_download",
    )


def _render_maintenance_section(conn, *, show_heading: bool = True):
    if show_heading:
        st.subheader("🔧 Maintenance Records")
    _, customer_label_map = build_customer_groups(conn, only_complete=False)
    customer_options, customer_labels, _, label_by_id = fetch_customer_choices(conn)
    viewer_id = current_user_id()
    do_df = df_query(
        conn,
        """
        SELECT d.do_number, d.customer_id, d.created_by, COALESCE(c.name, '(unknown)') AS customer_name, d.description, d.remarks, d.record_type
        FROM delivery_orders d
        LEFT JOIN customers c ON c.customer_id = d.customer_id
        WHERE COALESCE(d.record_type, 'delivery_order') = 'delivery_order'
        ORDER BY datetime(d.created_at) DESC
        """,
    )
    allowed_customers = accessible_customer_ids(conn)
    do_df = filter_delivery_orders_for_view(
        do_df, allowed_customers, record_types={"delivery_order"}
    )
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
        st.markdown("**Products sold during maintenance**")
        maintenance_rows = st.session_state.get(
            "maintenance_product_rows",
            [
                {
                    "name": "",
                    "model": "",
                    "serial": "",
                    "quantity": 1,
                }
            ],
        )
        maintenance_editor = st.data_editor(
            pd.DataFrame(maintenance_rows),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Product", help="Name / description"),
                "model": st.column_config.TextColumn("Model"),
                "serial": st.column_config.TextColumn("Serial"),
                "quantity": st.column_config.NumberColumn(
                    "Qty",
                    min_value=1,
                    step=1,
                    format="%d",
                ),
            },
            key="maintenance_product_table",
        )
        maintenance_product_entries = (
            maintenance_editor.to_dict("records")
            if isinstance(maintenance_editor, pd.DataFrame)
            else []
        )
        st.session_state["maintenance_product_rows"] = maintenance_product_entries
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
                    updated_at,
                    created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    current_user_id(),
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
               m.payment_receipt_path,
               m.updated_at,
               m.created_by,
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
            creator_id = row.get("created_by")
            candidates = []
            if pd.notna(maint_cust):
                candidates.append(int(maint_cust))
            if pd.notna(do_cust):
                candidates.append(int(do_cust))
            try:
                if viewer_id is not None and pd.notna(creator_id) and int(creator_id) == int(viewer_id):
                    return True
            except Exception:
                pass
            return any(cid in allowed_customers for cid in candidates)

        maintenance_df = maintenance_df[maintenance_df.apply(_maintenance_row_allowed, axis=1)]
    if not maintenance_df.empty:
        maintenance_df = fmt_dates(
            maintenance_df,
            ["maintenance_date", "maintenance_start_date", "maintenance_end_date"],
        )
        maintenance_df.drop(
            columns=["customer_id", "do_customer_id", "created_by"],
            inplace=True,
            errors="ignore",
        )
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
        if "payment_receipt_path" in maintenance_df.columns:
            maintenance_df["payment_receipt_display"] = maintenance_df[
                "payment_receipt_path"
            ].apply(lambda x: "📎" if clean_text(x) else "")
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
                "payment_receipt_display": "Receipt",
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
        existing_receipt_path = clean_text(selected_record.get("payment_receipt_path"))
        resolved_receipt = resolve_upload_path(existing_receipt_path)
        receipt_cols = st.columns([1, 1])
        with receipt_cols[0]:
            maintenance_receipt_upload = st.file_uploader(
                "Upload payment receipt (PDF or image)",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key=f"maintenance_edit_receipt_{selected_maintenance_id}",
            )
        with receipt_cols[1]:
            clear_maintenance_receipt = st.checkbox(
                "Remove receipt",
                value=False,
                key=f"maintenance_clear_receipt_{selected_maintenance_id}",
            )
        if resolved_receipt and resolved_receipt.exists():
            st.download_button(
                "Download receipt",
                data=resolved_receipt.read_bytes(),
                file_name=resolved_receipt.name,
                key=f"maintenance_receipt_download_{selected_maintenance_id}",
            )
        elif existing_receipt_path:
            st.caption("Receipt file not found. Upload a new copy to replace it.")
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
                receipt_path_value = existing_receipt_path
                replaced_receipt = False
                cleared_receipt = False
                if maintenance_receipt_upload is not None:
                    receipt_path_value = store_payment_receipt(
                        maintenance_receipt_upload,
                        identifier=f"maintenance_{selected_maintenance_id}_receipt",
                        target_dir=MAINTENANCE_DOCS_DIR,
                    )
                    replaced_receipt = bool(receipt_path_value)
                elif clear_maintenance_receipt and receipt_path_value:
                    old_receipt = resolve_upload_path(receipt_path_value)
                    if old_receipt and old_receipt.exists():
                        try:
                            old_receipt.unlink()
                        except Exception:
                            pass
                    receipt_path_value = None
                    cleared_receipt = True
                conn.execute(
                    """
                    UPDATE maintenance_records
                    SET status = ?,
                        remarks = ?,
                        maintenance_date = ?,
                        maintenance_start_date = ?,
                        maintenance_end_date = ?,
                        payment_receipt_path = COALESCE(?, payment_receipt_path),
                        updated_at = datetime('now')
                    WHERE maintenance_id = ?
                    """,
                    (
                        new_status,
                        clean_text(new_remarks),
                        maintenance_date_str,
                        maintenance_start_str,
                        maintenance_end_str,
                        receipt_path_value,
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
                message_bits = ["Maintenance record updated."]
                if replaced_receipt:
                    message_bits.append("Receipt uploaded")
                elif cleared_receipt:
                    message_bits.append("Receipt removed")
                st.success(" ".join(message_bits))
                _safe_rerun()
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


def delivery_orders_page(
    conn,
    *,
    show_heading: bool = True,
    record_type_label: str = "Delivery order",
    record_type_key: str = "delivery_order",
):
    if show_heading:
        st.subheader("🚚 Delivery orders")

    record_label_input = clean_text(record_type_label)
    record_label = record_label_input or "Delivery order"
    record_label_lower = record_label.lower()
    st.session_state.setdefault("delivery_order_items_rows", _default_delivery_items())
    st.session_state.setdefault("delivery_order_number", "")
    st.session_state.setdefault("delivery_order_customer", None)
    st.session_state.setdefault("delivery_order_description", "")
    st.session_state.setdefault("delivery_order_sales_person", "")
    st.session_state.setdefault("delivery_order_remarks", "")
    st.session_state.setdefault("delivery_order_status", "pending")
    autofill_customer_key = f"{record_type_key}_autofill_customer"
    st.session_state.setdefault(autofill_customer_key, None)

    customer_options, customer_labels, _, _ = fetch_customer_choices(conn)
    scope_clause, scope_params = customer_scope_filter("c")
    where_sql = f"WHERE {scope_clause}" if scope_clause else ""
    do_rows = df_query(
        conn,
        f"SELECT c.customer_id, c.delivery_order_code FROM customers c {where_sql}",
        tuple(scope_params),
    )
    customer_do_map: dict[int, str] = {}
    if not do_rows.empty:
        for _, row in do_rows.iterrows():
            customer_id = int_or_none(row.get("customer_id"))
            code = clean_text(row.get("delivery_order_code"))
            if customer_id is None or not code:
                continue
            customer_do_map[customer_id] = code
    existing_dos = df_query(
        conn,
        dedent(
            """
            SELECT d.do_number,
                   d.customer_id,
                   COALESCE(c.name, c.company_name, '(unknown)') AS customer,
                   d.description,
                   d.sales_person,
                   d.remarks,
                   d.file_path,
                   d.items_payload,
                   d.total_amount,
                   d.created_by,
                   d.created_at,
                   d.status,
                   d.payment_receipt_path,
                   d.updated_at,
                   d.record_type,
                   u.username
            FROM delivery_orders d
            LEFT JOIN customers c ON c.customer_id = d.customer_id
            LEFT JOIN users u ON u.user_id = d.created_by
            WHERE COALESCE(d.record_type, 'delivery_order') = ?
            ORDER BY datetime(d.created_at) DESC
            LIMIT 200
            """
        ),
        (record_type_key,),
    )
    load_labels: dict[Optional[str], str] = {None: f"-- New {record_label_lower} --"}
    load_choices = [None]
    if not existing_dos.empty:
        for _, row in existing_dos.iterrows():
            do_num = clean_text(row.get("do_number"))
            if not do_num:
                continue
            customer_name = clean_text(row.get("customer")) or "(customer)"
            load_choices.append(do_num)
            load_labels[do_num] = f"{do_num} • {customer_name}"

    closed_statuses = {"paid", "rejected"}
    current_receipt_path: Optional[str] = None

    st.markdown(f"### Create or update a {record_label_lower}")
    selected_existing = st.selectbox(
        f"Load existing {record_label_lower}",
        load_choices,
        format_func=lambda val: load_labels.get(val, "-- New delivery order --"),
        key="do_form_loader",
    )

    if selected_existing:
        match = existing_dos[existing_dos["do_number"] == selected_existing]
        if not match.empty:
            row = match.iloc[0]
            st.session_state["delivery_order_number"] = clean_text(row.get("do_number")) or ""
            cust_id = row.get("customer_id")
            st.session_state["delivery_order_customer"] = int(cust_id) if pd.notna(cust_id) else None
            st.session_state["delivery_order_description"] = clean_text(row.get("description")) or ""
            st.session_state["delivery_order_sales_person"] = clean_text(row.get("sales_person")) or ""
            st.session_state["delivery_order_remarks"] = clean_text(row.get("remarks")) or ""
            st.session_state["delivery_order_status"] = clean_text(row.get("status")) or "pending"
            current_receipt_path = clean_text(row.get("payment_receipt_path"))
            loaded_items = parse_delivery_items_payload(row.get("items_payload"))
            st.session_state["delivery_order_items_rows"] = loaded_items or _default_delivery_items()
        st.session_state[autofill_customer_key] = None
    else:
        selected_customer_state = int_or_none(st.session_state.get("delivery_order_customer"))
        last_autofill_customer = int_or_none(st.session_state.get(autofill_customer_key))
        current_number = clean_text(st.session_state.get("delivery_order_number"))
        suggested_code = None
        if selected_customer_state:
            suggested_code = customer_do_map.get(selected_customer_state)
            if not suggested_code:
                candidate = existing_dos[existing_dos["customer_id"] == selected_customer_state]
                if not candidate.empty:
                    suggested_code = clean_text(candidate.iloc[0].get("do_number"))
        should_autofill = (
            selected_customer_state
            and (selected_customer_state != last_autofill_customer or not current_number)
        )
        if should_autofill and suggested_code:
            st.session_state["delivery_order_number"] = suggested_code
            current_number = suggested_code
            st.session_state[autofill_customer_key] = selected_customer_state
        elif selected_customer_state is None:
            st.session_state[autofill_customer_key] = None

    with st.form("delivery_order_form"):
        do_number = st.text_input(
            f"{record_label} number *",
            value=st.session_state.get("delivery_order_number", ""),
            key="delivery_order_number",
        )
        selected_customer = st.selectbox(
            "Customer",
            customer_options,
            format_func=lambda cid: customer_labels.get(cid, "-- Select customer --"),
            key="delivery_order_customer",
        )
        description = st.text_area(
            "Description / items",
            value=st.session_state.get("delivery_order_description", ""),
            key="delivery_order_description",
        )
        sales_person = st.text_input(
            "Sales person / owner",
            value=st.session_state.get("delivery_order_sales_person", ""),
            key="delivery_order_sales_person",
        )
        remarks = st.text_area(
            "Remarks",
            value=st.session_state.get("delivery_order_remarks", ""),
            key="delivery_order_remarks",
        )
        status_value = st.selectbox(
            "Status",
            ["pending", "paid", "rejected"],
            index=["pending", "paid", "rejected"].index(
                clean_text(st.session_state.get("delivery_order_status"))
                or "pending"
            ),
            help="Paid or rejected delivery orders are locked against further edits.",
            disabled=(clean_text(st.session_state.get("delivery_order_status")) or "").lower()
            in closed_statuses,
            key="delivery_order_status",
        )
        receipt_upload = None
        if status_value == "paid":
            receipt_upload = st.file_uploader(
                f"{record_label} payment receipt (PDF or image)",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                key=f"{record_label.lower().replace(' ', '_')}_receipt_upload",
                help="Upload proof of payment before locking the record.",
            )
            if current_receipt_path:
                st.caption("A receipt is already attached; uploading a new file will replace it.")
        st.markdown("**Products / items**")
        items_df_seed = pd.DataFrame(
            st.session_state.get("delivery_order_items_rows", _default_delivery_items())
        )
        if "line_total" in items_df_seed.columns:
            items_df_seed = items_df_seed.drop(columns=["line_total"], errors="ignore")
        items_editor = st.data_editor(
            items_df_seed,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="delivery_order_items_editor",
            column_config={
                "description": st.column_config.TextColumn(
                    "Description", help="What is being delivered / sold"
                ),
                "quantity": st.column_config.NumberColumn(
                    "Qty", min_value=0.0, step=1.0, format="%.2f"
                ),
                "unit_price": st.column_config.NumberColumn(
                    "Unit price", min_value=0.0, step=50.0, format="%.2f"
                ),
                "discount": st.column_config.NumberColumn(
                    "Discount (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f"
                ),
            },
        )
        st.session_state["delivery_order_items_rows"] = (
            items_editor.to_dict("records") if isinstance(items_editor, pd.DataFrame) else []
        )
        items_clean, estimated_total = normalize_delivery_items(
            st.session_state.get("delivery_order_items_rows", [])
        )
        st.caption(
            f"Estimated total: {format_money(estimated_total) or f'{estimated_total:,.2f}'}"
        )
        do_file = st.file_uploader(
            f"Attach {record_label.lower()} (PDF)",
            type=["pdf"],
            help="Optional supporting document stored alongside the record.",
        )
        submit = st.form_submit_button(f"Save {record_label_lower}", type="primary")

    if submit:
        cleaned_number = clean_text(do_number)
        if not cleaned_number:
            st.error(f"{record_label} number is required.")
        else:
            cur = conn.cursor()
            conflicting_type = df_query(
                conn,
                "SELECT COALESCE(record_type, 'delivery_order') AS record_type FROM delivery_orders WHERE do_number = ? AND COALESCE(record_type, 'delivery_order') <> ?",
                (cleaned_number, record_type_key),
            )
            if not conflicting_type.empty:
                conflict_label = clean_text(conflicting_type.iloc[0].get("record_type")) or "delivery order"
                st.error(
                    f"This number is already used for a {conflict_label.replace('_', ' ')}. Choose a different {record_label_lower} number."
                )
                return
            existing = df_query(
                conn,
                "SELECT file_path, items_payload, total_amount, created_by, status, payment_receipt_path FROM delivery_orders WHERE do_number = ? AND COALESCE(record_type, 'delivery_order') = ?",
                (cleaned_number, record_type_key),
            )
            stored_path = None
            existing_status = "pending"
            existing_receipt = None
            if not existing.empty:
                stored_path = clean_text(existing.iloc[0].get("file_path"))
                existing_status = clean_text(existing.iloc[0].get("status")) or "pending"
                existing_receipt = clean_text(existing.iloc[0].get("payment_receipt_path"))
            locked_record = existing_status.lower() in closed_statuses
            receipt_only_update = (
                locked_record
                and status_value == existing_status
                and existing_status.lower() == "paid"
                and receipt_upload is not None
            )
            if locked_record and not receipt_only_update:
                st.warning(
                    f"{record_label} {cleaned_number} is locked because it is marked as {existing_status}."
                )
                return
            if do_file is not None:
                stored_path = store_uploaded_pdf(
                    do_file,
                    DELIVERY_ORDER_DIR,
                    filename=f"do_{_sanitize_path_component(cleaned_number)}.pdf",
                )
            receipt_path = existing_receipt
            if status_value == "paid":
                if receipt_upload:
                    receipt_identifier = _sanitize_path_component(cleaned_number) or "do_receipt"
                    receipt_path = store_payment_receipt(
                        receipt_upload,
                        identifier=f"{receipt_identifier}_receipt",
                        target_dir=DELIVERY_RECEIPT_DIR,
                    )
                if not receipt_path:
                    st.error("Upload a receipt before marking this record as paid.")
                    return
            if receipt_only_update:
                conn.execute(
                    """
                    UPDATE delivery_orders
                       SET payment_receipt_path=COALESCE(?, payment_receipt_path),
                           updated_at=datetime('now')
                     WHERE do_number=? AND COALESCE(record_type, 'delivery_order') = ?
                    """,
                    (
                        receipt_path,
                        cleaned_number,
                        record_type_key,
                    ),
                )
                conn.commit()
                st.success("Receipt added to locked record.")
                return
            items_clean, total_amount_value = normalize_delivery_items(
                st.session_state.get("delivery_order_items_rows", [])
            )
            if not items_clean:
                st.error("Add at least one product line with pricing to save the delivery order.")
                return
            items_payload = json.dumps(items_clean, ensure_ascii=False)
            creator_id = current_user_id()
            if existing.empty:
                cur.execute(
                    """
                    INSERT INTO delivery_orders (do_number, customer_id, description, sales_person, remarks, file_path, items_payload, total_amount, created_by, status, payment_receipt_path, updated_at, record_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
                    """,
                    (
                        cleaned_number,
                        int(selected_customer) if selected_customer else None,
                        clean_text(description),
                        clean_text(sales_person),
                        clean_text(remarks),
                        stored_path,
                        items_payload,
                        total_amount_value,
                        creator_id,
                        clean_text(status_value) or "pending",
                        receipt_path,
                        record_type_key,
                    ),
                )
            else:
                existing_creator = existing.iloc[0].get("created_by")
                cur.execute(
                    """
                    UPDATE delivery_orders
                       SET customer_id=?, description=?, sales_person=?, remarks=?, file_path=?, items_payload=?, total_amount=?, status=?, payment_receipt_path=COALESCE(?, payment_receipt_path), updated_at=datetime('now'), created_by=COALESCE(created_by, ?), record_type=?
                     WHERE do_number=? AND COALESCE(record_type, 'delivery_order') = ?
                    """,
                    (
                        int(selected_customer) if selected_customer else None,
                        clean_text(description),
                        clean_text(sales_person),
                        clean_text(remarks),
                        stored_path,
                        items_payload,
                        total_amount_value,
                        clean_text(status_value) or existing_status,
                        receipt_path,
                        creator_id,
                        record_type_key,
                        cleaned_number,
                        record_type_key,
                    ),
                )
            if selected_customer:
                link_delivery_order_to_customer(conn, cleaned_number, int(selected_customer))
            conn.commit()
            action_label = "updated" if not existing.empty else "created"
            customer_display = customer_labels.get(
                int(selected_customer) if selected_customer else None, "(customer)"
            )
            total_display = format_money(total_amount_value) or f"{_coerce_float(total_amount_value, 0.0):,.2f}"
            event_prefix = record_label.lower().replace(" ", "_") or "delivery_order"
            event_type = f"{event_prefix}_{'updated' if not existing.empty else 'created'}"
            entity_type = clean_text(record_type_key) or (
                "work_done" if event_prefix == "work_done" else "delivery_order"
            )
            status_label = clean_text(status_value) or existing_status
            status_note = f" [{status_label.title()}]" if status_label else ""

            log_activity(
                conn,
                event_type=event_type,
                description=(
                    f"{record_label} {cleaned_number} {action_label} for {customer_display}"
                    f" ({total_display}){status_note}"
                ),
                entity_type=entity_type,
                entity_id=None,
            )

            st.success(f"{record_label} {cleaned_number} saved successfully.")

    number_label = f"{record_label} number"
    st.markdown(f"### {record_label} search")
    filter_cols = st.columns((1.2, 1, 1, 1))
    with filter_cols[0]:
        query_text = st.text_input(
            f"Search by {record_label.lower()} number, description or remarks",
            key="do_filter_text",
        )
    with filter_cols[1]:
        customer_filter = st.selectbox(
            "Filter by customer",
            options=[None] + [opt for opt in customer_options if opt is not None],
            format_func=lambda cid: customer_labels.get(cid, "(any)"),
            key="do_filter_customer",
        )
    with filter_cols[2]:
        sales_filter = st.text_input("Sales person filter", key="do_filter_sales")
    with filter_cols[3]:
        use_date_filter = st.checkbox("Filter by created date", key="do_filter_date_toggle")
    date_range = None
    if use_date_filter:
        date_range = st.date_input(
            "Created between",
            value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
            key="do_filter_date_range",
        )

    do_df = df_query(
        conn,
        """
        SELECT d.do_number,
               d.customer_id,
               COALESCE(c.name, '(unknown)') AS customer,
               d.description,
               d.sales_person,
               d.remarks,
               d.created_at,
               d.file_path,
               d.total_amount,
               d.items_payload,
               d.created_by,
               d.status,
               d.payment_receipt_path,
               d.updated_at,
               COALESCE(u.username, '(user)') AS created_by_name
          FROM delivery_orders d
          LEFT JOIN customers c ON c.customer_id = d.customer_id
          LEFT JOIN users u ON u.user_id = d.created_by
         WHERE COALESCE(d.record_type, 'delivery_order') = ?
         ORDER BY datetime(d.created_at) DESC
        """,
        (record_type_key,),
    )
    allowed_customers = accessible_customer_ids(conn)
    do_df = filter_delivery_orders_for_view(do_df, allowed_customers)
    if not do_df.empty:
        do_df = fmt_dates(do_df, ["created_at", "updated_at"])
        if query_text:
            needle = query_text.lower()
            do_df = do_df[
                do_df.apply(
                    lambda row: any(
                        needle in str(row.get(col, "")).lower()
                        for col in ["do_number", "description", "remarks"]
                    ),
                    axis=1,
                )
            ]
        if customer_filter:
            do_df = do_df[do_df["customer_id"] == int(customer_filter)]
        if sales_filter:
            sales_needle = sales_filter.lower()
            do_df = do_df[
                do_df["sales_person"].apply(
                    lambda value: sales_needle in str(value).lower() if pd.notna(value) else False
                )
            ]
        if use_date_filter and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
            start_iso = to_iso_date(start_date)
            end_iso = to_iso_date(end_date)
            if start_iso and end_iso:
                do_df = do_df[
                    do_df["created_at"].apply(
                        lambda value: start_iso <= to_iso_date(value) <= end_iso if to_iso_date(value) else False
                    )
                ]
        do_df["Document"] = do_df["file_path"].apply(lambda fp: "📎" if clean_text(fp) else "")
        do_df["Receipt"] = do_df["payment_receipt_path"].apply(lambda fp: "📎" if clean_text(fp) else "")
        do_df["status"] = do_df["status"].apply(lambda s: (clean_text(s) or "pending").title())
        if "total_amount" in do_df.columns:
            def _format_total_value(value: object) -> str:
                if value is None:
                    return ""
                try:
                    if pd.isna(value):
                        return ""
                except Exception:
                    pass
                return format_money(value) or f"{_coerce_float(value, 0.0):,.2f}"

            do_df["total_amount"] = do_df["total_amount"].apply(_format_total_value)

        st.dataframe(
            do_df.rename(
                columns={
                    "do_number": number_label,
                    "customer": "Customer",
                    "description": "Description",
                    "sales_person": "Sales person",
                    "remarks": "Remarks",
                    "created_at": "Created",
                    "updated_at": "Updated",
                    "Document": "Attachment",
                    "Receipt": "Receipt",
                    "total_amount": "Total",
                    "created_by_name": "Created by",
                    "status": "Status",
                }
            ).drop(
                columns=[
                    "customer_id",
                    "file_path",
                    "items_payload",
                    "created_by",
                    "payment_receipt_path",
                ],
                errors="ignore",
            ),
            use_container_width=True,
        )

    downloads = {}
    if not do_df.empty:
        downloads = {
            clean_text(row["do_number"]): clean_text(row["file_path"])
            for _, row in do_df.iterrows()
            if clean_text(row.get("file_path"))
        }
    if downloads:
        st.markdown(f"#### Download {record_label.lower()}")
        selected_download = st.selectbox(
            f"Pick a {record_label_lower}", list(downloads.keys()), key="do_download_select"
        )
        path_value = downloads.get(selected_download)
        file_path = resolve_upload_path(path_value)
        if file_path and file_path.exists():
            st.download_button(
                f"Download {selected_download}",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                key="do_download_button",
            )
        else:
            st.info("The selected delivery order file could not be found.")
    elif st.session_state.get("do_filter_text") or query_text:
        st.caption(
            f"No matching {record_label_lower} records found for the applied filters."
        )


def quotation_page(conn, *, render_id: Optional[int] = None):
    st.subheader("🧾 Quotation")
    _render_quotation_section(conn, render_id=render_id)
    st.markdown("---")
    _render_quotation_management(conn)


def work_done_page(conn):
    st.subheader("✅ Work done")
    st.caption("Create, update, and download work completion slips just like delivery orders.")
    delivery_orders_page(
        conn,
        show_heading=False,
        record_type_label="Work done",
        record_type_key="work_done",
    )


def service_maintenance_page(conn):
    st.subheader("🛠️ Maintenance and Service")
    tabs = st.tabs(["Service", "Maintenance"])
    with tabs[0]:
        st.markdown("### Service records")
        _render_service_section(conn, show_heading=False)
    with tabs[1]:
        st.markdown("### Maintenance records")
        _render_maintenance_section(conn, show_heading=False)
    st.markdown("---")
    st.info("Create new quotations from the dedicated 'Quotation' page in the sidebar.")


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
        SELECT customer_id as id, name, phone, address, remarks, purchase_date, product_info, delivery_order_code, created_at
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
    display_cols = ["name", "phone", "address", "remarks", "purchase_date", "product_info", "delivery_order_code", "missing", "created_at"]
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
        new_address = clean_text(address)
        new_remarks = clean_text(remarks_text)
        purchase_str, _ = date_strings_from_input(purchase)
        new_product = clean_text(product)
        new_do = clean_text(do_code)
        old_phone = clean_text(selected.get("phone"))
        conn.execute(
            "UPDATE customers SET name=?, phone=?, address=?, remarks=?, purchase_date=?, product_info=?, delivery_order_code=?, dup_flag=0 WHERE customer_id=?",
            (
                new_name,
                new_phone,
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
            "UPDATE import_history SET customer_name=?, phone=?, address=?, delivery_address=?, product_label=?, do_number=?, original_date=? WHERE customer_id=? AND deleted_at IS NULL",
            (
                new_name,
                new_phone,
                new_address,
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
    "delivery_address": {"delivery_address", "delivery_addr", "shipping_address", "ship_to", "delivery"},
    "phone": {"phone", "mobile", "contact", "contact_no", "phone_no", "phone_number", "cell", "whatsapp"},
    "product": {"product", "item", "generator", "model", "description"},
    "do_code": {"do_code", "delivery_order", "delivery_order_code", "delivery_order_no", "do", "d_o_code", "do_number"},
    "quantity": {"quantity", "qty", "count", "units", "pcs", "pieces"},
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
    default_order = [
        "date",
        "customer_name",
        "address",
        "delivery_address",
        "phone",
        "product",
        "do_code",
        "quantity",
    ]
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
    col8, col9 = st.columns(2)
    col10, _ = st.columns(2)
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
        "Delivery address", options=opts, index=(guess["delivery_address"] + 1) if guess.get("delivery_address") is not None else 0
    )
    sel_do_code = col7.selectbox(
        "Delivery order code (optional)", options=opts, index=(guess["do_code"] + 1) if guess.get("do_code") is not None else 0
    )
    sel_remarks = col8.selectbox(
        "Remarks", options=opts, index=(guess.get("remarks", None) + 1) if guess.get("remarks") is not None else 0
    )
    sel_amount = col9.selectbox(
        "Amount spent", options=opts, index=(guess.get("amount_spent", None) + 1) if guess.get("amount_spent") is not None else 0
    )
    sel_quantity = col10.selectbox("Quantity", options=opts, index=(guess.get("quantity", None) + 1) if guess.get("quantity") is not None else 0)

    def pick(col_name):
        return df[col_name] if col_name != "(blank)" else pd.Series([None] * len(df))

    df_norm = pd.DataFrame(
        {
            "date": pick(sel_date),
            "customer_name": pick(sel_name),
            "address": pick(sel_addr),
            "delivery_address": pick(sel_do),
            "phone": pick(sel_phone),
            "product": pick(sel_prod),
            "do_code": pick(sel_do_code),
            "remarks": pick(sel_remarks),
            "amount_spent": pick(sel_amount),
            "quantity": pick(sel_quantity),
        }
    )
    df_norm["quantity"] = df_norm["quantity"].apply(parse_quantity)
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
            "quantity": st.column_config.NumberColumn(
                "Quantity", min_value=1, step=1, format="%d", required=False
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
    def _compose_search_blob(row) -> str:
        fields = [
            row.get("name"),
            row.get("company_name"),
            row.get("phone"),
            row.get("address"),
            row.get("delivery_address"),
            row.get("product_info"),
            row.get("delivery_order_code"),
            row.get("remarks"),
            row.get("sales_person"),
            row.get("purchase_date"),
        ]
        cleaned: list[str] = []
        for value in fields:
            text = clean_text(value)
            if not text:
                continue
            cleaned.append(" ".join(text.split()))
        return " ".join(cleaned).lower()

    work_df["_search_blob"] = work_df.apply(_compose_search_blob, axis=1)
    work_df["_search_blob"] = work_df["_search_blob"].fillna("")

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
    filter_normalized = " ".join(filter_value.lower().split())

    filtered_df = work_df
    if filter_normalized:
        mask = filtered_df["_search_blob"].str.contains(
            filter_normalized, regex=False, na=False
        )
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
    if auto_merge_matching_customers(conn):
        st.info(
            "Automatically merged customers sharing the same name and address.",
            icon="✅",
        )
        _safe_rerun()
        return
    scope_clause, scope_params = customer_scope_filter("c")
    where_sql = f"WHERE {scope_clause}" if scope_clause else ""
    cust_raw = df_query(
        conn,
        f"""
        SELECT
            c.customer_id as id,
            c.name,
            c.phone,
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
            combined_preview = (
                preview_df.groupby(["name", "phone"], dropna=False)
                .apply(
                    lambda g: pd.Series(
                        {
                            "Address": dedupe_join(
                                [
                                    clean_text(val)
                                    for val in g.get("address", pd.Series(dtype=object)).tolist()
                                    if clean_text(val)
                                ]
                            ),
                            "Products": dedupe_join(
                                [
                                    clean_text(val)
                                    for val in g.get("product_info", pd.Series(dtype=object)).tolist()
                                    if clean_text(val)
                                ]
                            ),
                            "Purchase dates": dedupe_join(
                                [
                                    val
                                    for val in g.get("purchase_date_fmt", pd.Series(dtype=object)).tolist()
                                    if val
                                ]
                            ),
                            "DO codes": dedupe_join(
                                [
                                    clean_text(val)
                                    for val in g.get("delivery_order_code", pd.Series(dtype=object)).tolist()
                                    if clean_text(val)
                                ]
                            ),
                            "Created": dedupe_join(
                                [
                                    val
                                    for val in g.get("created_at_fmt", pd.Series(dtype=object)).tolist()
                                    if val
                                ]
                            ),
                        }
                    )
                )
                .reset_index()
            )
            if not combined_preview.empty:
                combined_preview = combined_preview.rename(
                    columns={"name": "Name", "phone": "Phone"}
                )
                st.caption("Condensed view (grouped by customer and phone)")
                st.dataframe(
                    combined_preview,
                    use_container_width=True,
                    hide_index=True,
                )
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
                        new_address = clean_text(row.get("address"))
                        purchase_str, _ = date_strings_from_input(row.get("purchase_date"))
                        product_label = clean_text(row.get("product_info"))
                        new_do = clean_text(row.get("delivery_order_code"))
                        original_row = raw_map[cid]
                        old_name = clean_text(original_row.get("name"))
                        old_phone = clean_text(original_row.get("phone"))
                        old_address = clean_text(original_row.get("address"))
                        old_purchase = clean_text(original_row.get("purchase_date"))
                        old_product = clean_text(original_row.get("product_info"))
                        old_do = clean_text(original_row.get("delivery_order_code"))
                        if (
                            new_name == old_name
                            and new_phone == old_phone
                            and new_address == old_address
                            and purchase_str == old_purchase
                            and product_label == old_product
                            and new_do == old_do
                        ):
                            continue
                        conn.execute(
                            "UPDATE customers SET name=?, phone=?, address=?, purchase_date=?, product_info=?, delivery_order_code=?, dup_flag=0 WHERE customer_id=?",
                            (
                                new_name,
                                new_phone,
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
    users = df_query(
        conn,
        """
        SELECT user_id as id, username, phone, email, title, role, created_at
        FROM users
        ORDER BY datetime(created_at) DESC
        """,
    )
    users = users.assign(created_at=pd.to_datetime(users["created_at"], errors="coerce").dt.strftime(DATE_FMT))
    st.dataframe(users.drop(columns=["id"], errors="ignore"))

    with st.expander("Add user"):
        with st.form("add_user"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            phone = st.text_input("Phone number (required)")
            email = st.text_input("Email (optional)")
            title = st.text_input("Title / role", help="Shown on quotations by default")
            role = st.selectbox("Role", ["staff", "admin"])
            ok = st.form_submit_button("Create")
            if ok and u.strip() and p.strip():
                if not phone.strip():
                    st.error("Phone number is required for staff accounts.")
                    return
                h = hashlib.sha256(p.encode("utf-8")).hexdigest()
                try:
                    conn.execute(
                        "INSERT INTO users (username, pass_hash, phone, email, title, role) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            u.strip(),
                            h,
                            clean_text(phone),
                            clean_text(email),
                            clean_text(title),
                            role,
                        ),
                    )
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
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
        delivery_addr = clean_text(r.get("delivery_address"))
        phone = clean_text(r.get("phone"))
        product_label = clean_text(r.get("product"))
        do_serial = clean_text(r.get("do_code"))
        remarks_val = clean_text(r.get("remarks"))
        amount_value = parse_amount(r.get("amount_spent"))
        quantity_value = parse_quantity(r.get("quantity"), default=1)
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
            "INSERT INTO customers (name, phone, address, delivery_address, remarks, amount_spent, created_by, dup_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cust,
                phone,
                addr,
                delivery_addr,
                remarks_val,
                amount_value,
                created_by,
                dupc,
            ),
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
            (oid, pid, quantity_value),
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
            "UPDATE customers SET purchase_date=?, product_info=?, delivery_order_code=?, remarks=?, amount_spent=?, delivery_address=? WHERE customer_id=?",
            (
                purchase_date,
                product_label,
                do_serial,
                remarks_val,
                amount_value,
                delivery_addr,
                cid,
            ),
        )
        cur.execute(
            "INSERT INTO import_history (customer_id, product_id, order_id, order_item_id, warranty_id, do_number, import_tag, original_date, customer_name, address, phone, product_label, notes, amount_spent, imported_by, delivery_address, quantity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                delivery_addr,
                quantity_value,
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
    new_delivery_address = clean_text(updates.get("delivery_address"))
    purchase_date_str, expiry_str = date_strings_from_input(updates.get("purchase_date"))
    product_label = clean_text(updates.get("product_label"))
    new_do = clean_text(updates.get("do_number"))
    new_remarks = clean_text(updates.get("remarks"))
    new_amount = parse_amount(updates.get("amount_spent"))
    quantity_value = parse_quantity(updates.get("quantity"), default=1)
    product_name, product_model = split_product_label(product_label)

    if customer_id is not None:
        cur.execute(
            "UPDATE customers SET name=?, phone=?, address=?, delivery_address=?, purchase_date=?, product_info=?, delivery_order_code=?, remarks=?, amount_spent=?, dup_flag=0 WHERE customer_id=?",
            (
                new_name,
                new_phone,
                new_address,
                new_delivery_address,
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
            (quantity_value, order_item_id),
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
        "UPDATE import_history SET original_date=?, customer_name=?, address=?, delivery_address=?, phone=?, product_label=?, do_number=?, notes=?, amount_spent=?, quantity=? WHERE import_id=?",
        (
            purchase_date_str,
            new_name,
            new_address,
            new_delivery_address,
            new_phone,
            product_label,
            new_do,
            new_remarks,
            new_amount,
            quantity_value,
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
               c.delivery_order_code AS live_do_code, c.delivery_address AS live_delivery_address,
               c.attachment_path AS live_attachment_path
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
        "delivery_address",
        "product_label",
        "do_number",
        "quantity",
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
            "delivery_address": "Delivery address",
            "product_label": "Product",
            "do_number": "DO code",
            "quantity": "Quantity",
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
    current_delivery_address = (
        clean_text(selected.get("live_delivery_address"))
        or clean_text(selected.get("delivery_address"))
        or ""
    )
    current_product = clean_text(selected.get("live_product_info")) or clean_text(selected.get("product_label")) or ""
    current_do = clean_text(selected.get("live_do_code")) or clean_text(selected.get("do_number")) or ""
    purchase_seed = selected.get("live_purchase_date") or selected.get("original_date")
    purchase_str = clean_text(purchase_seed) or ""
    amount_seed = selected.get("amount_spent")
    amount_value = parse_amount(amount_seed)
    amount_display = ""
    if amount_value is not None:
        amount_display = format_money(amount_value) or f"{amount_value:,.2f}"
    current_quantity = parse_quantity(selected.get("quantity"), default=1)

    user = st.session_state.user or {}
    is_admin = user.get("role") == "admin"

    with st.form(f"manage_import_{selected_id}"):
        name_input = st.text_input("Customer name", value=current_name)
        phone_input = st.text_input("Phone", value=current_phone)
        address_input = st.text_area("Address", value=current_address)
        delivery_address_input = st.text_area(
            "Delivery address", value=current_delivery_address
        )
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
        quantity_input = st.number_input(
            "Quantity",
            min_value=1,
            value=current_quantity,
            step=1,
            help="How many units were recorded on this import line.",
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
                "delivery_address": delivery_address_input,
                "purchase_date": purchase_input,
                "product_label": product_input,
                "do_number": do_input,
                "remarks": remarks_input,
                "amount_spent": amount_input,
                "quantity": quantity_input,
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
        selectable_ids = [
            int(val) for val in selectable_reports["report_id"].tolist() if not pd.isna(val)
        ]

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
            try:
                st.session_state["report_edit_select"] = int(pending_selection)
            except Exception:
                st.session_state["report_edit_select"] = None

    current_selection = st.session_state.get("report_edit_select")
    try:
        if current_selection is not None and int(current_selection) not in report_choices:
            st.session_state["report_edit_select"] = None
    except Exception:
        st.session_state["report_edit_select"] = None

    selected_report_id = st.selectbox(
        "Load an existing report",
        report_choices,
        format_func=_format_report_choice,
        key="report_edit_select",
    )

    # Preserve spreadsheet-style edits while working on the same report, but
    # reset when switching to another record.
    if st.session_state.get("report_grid_current_id") != selected_report_id:
        st.session_state["report_grid_current_id"] = selected_report_id
        st.session_state.pop("report_grid_editor_state", None)

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

    st.markdown("##### Import report data")
    import_payload = st.session_state.get("report_grid_import_payload")
    import_file = st.file_uploader(
        "Upload report grid (Excel or CSV)",
        type=["xlsx", "xls", "csv"],
        help="Populate the grid below by importing a spreadsheet with columns matching the report table.",
        key="report_grid_importer",
    )
    uploaded_df: Optional[pd.DataFrame] = None
    if import_file is not None:
        import_payload = {
            "name": import_file.name,
            "data": import_file.getvalue(),
        }
        st.session_state["report_grid_import_payload"] = import_payload
        st.session_state.pop("report_grid_mapping_choices", None)

    if import_payload:
        uploaded_df = _load_report_grid_dataframe(
            import_payload.get("data", b""), import_payload.get("name", "")
        )
        if uploaded_df is not None:
            suggestions = _suggest_report_column_mapping(uploaded_df.columns)
            if suggestions and import_file is not None:
                auto_mapping = {source: target for target, source in suggestions.items()}
                imported_rows = _import_report_grid_from_dataframe(
                    uploaded_df, auto_mapping
                )
                if imported_rows:
                    st.session_state["report_grid_import_rows"] = imported_rows
                    st.session_state["report_grid_mapping_choices"] = suggestions
                    st.success(
                        f"Loaded {len(imported_rows)} row(s) from the uploaded file. They are ready in the grid below."
                    )
                else:
                    st.warning(
                        "We could not read any matching columns from that file. Adjust the mapping below to continue.",
                        icon="⚠️",
                    )
            mapping_seed = st.session_state.get("report_grid_mapping_choices", {})
            map_options = ["(Do not import)"] + list(uploaded_df.columns)
            with st.form("report_grid_import_mapper"):
                st.caption(
                    "Align columns from the uploaded file to the report grid fields. Skipped columns will be ignored."
                )
                selected_mapping: dict[str, str] = {}
                for key, config in REPORT_GRID_FIELDS.items():
                    default_choice = mapping_seed.get(key) or suggestions.get(key)
                    if default_choice not in map_options:
                        default_choice = "(Do not import)"
                    choice = st.selectbox(
                        config["label"],
                        options=map_options,
                        index=map_options.index(default_choice)
                        if default_choice in map_options
                        else 0,
                        key=f"report_map_{key}",
                        help=f"Select the column that represents '{config['label']}'.",
                    )
                    if choice != "(Do not import)":
                        selected_mapping[choice] = key
                load_clicked = st.form_submit_button("Load mapped rows into grid")

            if load_clicked:
                st.session_state["report_grid_mapping_choices"] = {
                    key: st.session_state.get(f"report_map_{key}")
                    for key in REPORT_GRID_FIELDS.keys()
                }
                imported_rows = _import_report_grid_from_dataframe(
                    uploaded_df, selected_mapping
                )
                if imported_rows:
                    st.session_state["report_grid_import_rows"] = imported_rows
                    st.success(
                        f"Loaded {len(imported_rows)} row(s) using the selected mapping."
                    )
                else:
                    st.warning(
                        "No rows were imported with that mapping. Please review the selections and try again.",
                        icon="⚠️",
                    )
        else:
            st.warning(
                "We could not read any matching columns from that file. Ensure the headers match the report grid labels.",
                icon="⚠️",
            )

    grid_seed_rows = (
        parse_report_grid_payload(editing_record.get("grid_payload"))
        if editing_record
        else []
    )
    if "report_grid_import_rows" in st.session_state:
        grid_seed_rows = st.session_state.pop("report_grid_import_rows") or grid_seed_rows
    elif st.session_state.get("report_grid_editor_state"):
        grid_seed_rows = st.session_state["report_grid_editor_state"] or grid_seed_rows
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
        st.session_state["report_grid_editor_state"] = _grid_rows_from_editor(report_grid_df)
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
            "Attach supporting document (PDF, image, or Excel)",
            type=["pdf", "png", "jpg", "jpeg", "webp", "gif", "xlsx", "xls"],
            key="report_attachment_uploader",
            help="Optional proof of work, photos, or documentation. Excel uploads are supported for imported reports.",
        )
        submitted = st.form_submit_button("Save report", type="primary")

    if submitted:
        cleanup_path: Optional[str] = None
        grid_rows_to_store = _grid_rows_from_editor(report_grid_df)
        st.session_state.pop("report_grid_editor_state", None)
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
    st.session_state["_render_id"] = st.session_state.get("_render_id", 0) + 1
    render_id = st.session_state["_render_id"]
    init_ui()
    _ensure_quotation_editor_server()
    conn = get_conn()
    init_schema(conn)
    login_box(conn, render_id=render_id)

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    user = st.session_state.user or {}
    role = user.get("role")
    with st.sidebar:
        st.markdown("### Navigation")
        if role == "admin":
            pages = [
                "Dashboard",
                "Customers",
                "Work done",
                "Delivery Orders",
                "Quotation",
                "Customer Summary",
                "Scraps",
                "Warranties",
                "Import",
                "Advanced Search",
                "Reports",
                "Duplicates",
                "Users (Admin)",
                "Maintenance and Service",
            ]
        else:
            pages = [
                "Dashboard",
                "Customers",
                "Work done",
                "Delivery Orders",
                "Quotation",
                "Customer Summary",
                "Warranties",
                "Import",
                "Reports",
                "Maintenance and Service",
            ]

        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = st.session_state.get("page", pages[0])
        if st.session_state.get("nav_page") not in pages:
            st.session_state["nav_page"] = pages[0]

        current_page = st.session_state.get("nav_page", pages[0])
        if "nav_selection" not in st.session_state:
            st.session_state["nav_selection"] = current_page
        if st.session_state.get("nav_selection") not in pages:
            st.session_state["nav_selection"] = current_page
        page_choice = st.radio(
            "Navigate",
            pages,
            index=pages.index(current_page),
            key="nav_selection",
        )
        st.session_state["nav_page"] = page_choice
        page = page_choice
        st.session_state.page = page

    show_expiry_notifications(conn)

    if page == "Dashboard":
        dashboard(conn)
    elif page == "Work done":
        work_done_page(conn)
    elif page == "Delivery Orders":
        delivery_orders_page(conn)
    elif page == "Quotation":
        quotation_page(conn, render_id=render_id)
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
    elif page == "Advanced Search":
        advanced_search_page(conn)
    elif page == "Reports":
        reports_page(conn)
    elif page == "Duplicates":
        duplicates_page(conn)
    elif page == "Users (Admin)":
        users_admin_page(conn)
    elif page == "Maintenance and Service":
        service_maintenance_page(conn)

if __name__ == "__main__":
    if _streamlit_runtime_active():
        main()
    else:
        _bootstrap_streamlit_app()
