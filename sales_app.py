"""Streamlit-based sales management application."""

from __future__ import annotations

import base64
import io
import hashlib
import os
import re
import sqlite3
import textwrap
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import html

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ps_sales import (
    AccountLockoutService,
    NotificationScheduler,
    PasswordService,
    UploadManager,
    UserRepository,
    Database,
    load_config,
)


# ---------------------------------------------------------------------------
# Global application services
# ---------------------------------------------------------------------------


CONFIG = load_config()
DATABASE = Database.from_config(CONFIG)
USER_REPOSITORY = UserRepository(DATABASE)
PASSWORD_SERVICE = PasswordService.default()
LOCKOUT_SERVICE = AccountLockoutService(CONFIG, USER_REPOSITORY)
UPLOAD_MANAGER = UploadManager(CONFIG)


DEFAULT_QUOTATION_STATUSES: Tuple[str, ...] = (
    "pending",
    "accepted",
    "declined",
    "inform_later",
)

LETTER_FOLLOW_UP_STATUSES: Tuple[str, ...] = (
    "paid",
    "possible",
    "rejected",
)

LETTER_FOLLOW_UP_LABELS: Dict[str, str] = {
    "paid": "Paid",
    "possible": "Possible",
    "rejected": "Rejected",
}

CUSTOM_FOLLOW_UP_CHOICE = "Custom date"
FOLLOW_UP_SUGGESTIONS: Dict[str, Optional[int]] = {
    "In 3 days": 3,
    "In 1 week": 7,
    "In 2 weeks": 14,
    CUSTOM_FOLLOW_UP_CHOICE: None,
}
DEFAULT_FOLLOW_UP_CHOICE = "In 3 days"


class SafeFormatDict(dict):
    """Dictionary returning the placeholder for missing keys during format."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return "{" + key + "}"


LETTER_TEMPLATE_LIBRARY: Tuple[Dict[str, Any], ...] = (
    {
        "key": "standard_supply",
        "label": "Standard supply proposal",
        "description": (
            "Balanced template for quoting equipment supply with delivery, warranty "
            "and implementation services."
        ),
        "suggested_follow_up": "In 3 days",
        "fields": {
            "subject_line": "Proposal for {customer_company}",
            "salutation": "Dear {customer_salutation},",
            "body_intro": textwrap.dedent(
                """\
                Thank you for inviting PS Business Suites by ZAD to submit our quotation for {customer_company}.\n"
                "We appreciate the opportunity to support your team and have summarised the deliverables below for your review."
                """
            ).strip(),
            "product_details": textwrap.dedent(
                """\
                • Supply of listed products with manufacturer warranty\n"
                "• Delivery to the designated site including handling and logistics\n"
                "• Installation and commissioning by certified engineers\n"
                "• User orientation with documentation handover\n"
                "• Complimentary post-installation health check within 30 days"
                """
            ).strip(),
            "closing_text": textwrap.dedent(
                """\
                We are ready to mobilise within three working days of receiving your confirmation.\n"
                "Should you need any clarification, please reach out directly so we can refine the scope together."
                """
            ).strip(),
            "quotation_remark": (
                "Pricing is valid for 14 days from {quote_date_long} and includes standard delivery within Dhaka."
            ),
            "follow_up_status": "possible",
            "follow_up_note": (
                "Schedule a call with {customer_contact_name} to confirm technical details and target delivery date."
            ),
        },
    },
    {
        "key": "maintenance_services",
        "label": "Maintenance & support plan",
        "description": (
            "Ideal for annual maintenance contracts that combine preventative visits, spare parts and SLA-backed support."
        ),
        "suggested_follow_up": "In 1 week",
        "fields": {
            "subject_line": "Comprehensive maintenance proposal for {customer_company}",
            "salutation": "Dear {customer_salutation},",
            "body_intro": textwrap.dedent(
                """\
                Following our recent discussion, we are pleased to outline a preventative maintenance and support\n"
                "programme tailored to keep {customer_company}'s operations running smoothly throughout the year."
                """
            ).strip(),
            "product_details": textwrap.dedent(
                """\
                • Quarterly preventative maintenance visits with detailed reporting\n"
                "• Priority response hotline backed by a 4-hour acknowledgement SLA\n"
                "• Replacement of critical wear-and-tear parts from our local inventory\n"
                "• Remote monitoring health-checks and configuration backups\n"
                "• Optional operator refresher training during each visit"
                """
            ).strip(),
            "closing_text": textwrap.dedent(
                """\
                The plan can commence as early as next week. We are happy to fine-tune the visit schedule\n"
                "or response commitments so they align with your internal processes."
                """
            ).strip(),
            "quotation_remark": (
                "Contract tenure: 12 months from signing. Renewal reminders will be sent 60 days before expiry."
            ),
            "follow_up_status": "possible",
            "follow_up_note": (
                "Share the proposed maintenance calendar and confirm service windows with {customer_contact_name}."
            ),
        },
    },
    {
        "key": "project_solution",
        "label": "Project implementation package",
        "description": (
            "Use when presenting turnkey project solutions that include design, deployment and handover milestones."
        ),
        "suggested_follow_up": "In 2 weeks",
        "fields": {
            "subject_line": "Turnkey implementation for {customer_company}",
            "salutation": "Dear {customer_salutation},",
            "body_intro": textwrap.dedent(
                """\
                As discussed, the PS Business Suites by ZAD project team has prepared a turnkey implementation package\n"
                "covering the full project lifecycle for {customer_company}. The proposal highlights the milestones below."
                """
            ).strip(),
            "product_details": textwrap.dedent(
                """\
                • Detailed design workshop and documentation handover\n"
                "• On-site deployment managed by a dedicated project coordinator\n"
                "• Integrated testing with customer sign-off checkpoints\n"
                "• User enablement sessions and go-live support\n"
                "• Post-implementation review with optimisation recommendations"
                """
            ).strip(),
            "closing_text": textwrap.dedent(
                """\
                We recommend a joint planning session to confirm dependencies and agree on milestone owners.\n"
                "Once approved, our team can initiate the project kickoff within ten working days."
                """
            ).strip(),
            "quotation_remark": (
                "Quoted milestone dates assume a notice-to-proceed no later than {quote_date_long}."
            ),
            "follow_up_status": "possible",
            "follow_up_note": (
                "Arrange a milestone review meeting with {customer_contact_name} to discuss resource allocation."
            ),
        },
    },
)

LETTER_TEMPLATE_INDEX: Dict[str, Dict[str, Any]] = {
    template["key"]: template for template in LETTER_TEMPLATE_LIBRARY
}
LETTER_TEMPLATE_LABEL_MAP: Dict[str, str] = {
    template["key"]: template["label"] for template in LETTER_TEMPLATE_LIBRARY
}
LETTER_TEMPLATE_KEY_BY_LABEL: Dict[str, str] = {
    template["label"]: template["key"] for template in LETTER_TEMPLATE_LIBRARY
}

NOTIFICATION_SCHEDULER = NotificationScheduler(DATABASE, CONFIG)
UPLOAD_MANAGER.enforce_retention()


def load_letterhead_base64() -> Optional[str]:
    """Return the base64-encoded contents of the quotation letterhead."""

    letterhead_path = Path(__file__).with_name("ps_letterhead.png")
    if not letterhead_path.exists():
        return None
    return base64.b64encode(letterhead_path.read_bytes()).decode("utf-8")


LETTERHEAD_BASE64 = load_letterhead_base64()


def default_salesperson_display(user: Dict) -> str:
    """Return a human readable version of the salesperson username."""

    display_name = user.get("display_name")
    if display_name:
        return str(display_name)
    username = user.get("username", "")
    if not username:
        return ""
    transformed = username.replace("_", " ")
    return transformed.title()


LETTER_FORM_FIELDS: Tuple[str, ...] = (
    "reference_no",
    "quote_date",
    "customer_name",
    "customer_company",
    "customer_address",
    "customer_contact",
    "district_id",
    "attention_name",
    "attention_title",
    "subject_line",
    "salutation",
    "body_intro",
    "product_details",
    "tracked_products",
    "amount",
    "quote_type",
    "closing_text",
    "salesperson_name",
    "salesperson_title",
    "salesperson_contact",
    "quotation_remark",
    "follow_up_status",
    "follow_up_date",
    "follow_up_note",
    "payment_status",
    "payment_receipt",
    "pdf_path",
)


LETTER_REQUIRED_LABELS: Dict[str, str] = {
    "reference_no": "Reference number",
    "customer_name": "Customer contact name",
    "customer_company": "Customer company",
    "customer_address": "Customer address",
    "district_id": "Customer district",
    "quote_type": "Quote type",
    "subject_line": "Subject",
    "salesperson_name": "Salesperson name",
}


LETTER_RECOMMENDED_LABELS: Dict[str, str] = {
    "body_intro": "Introduction",
    "product_details": "Product details",
    "quotation_remark": "Quotation remarks",
    "follow_up_note": "Follow-up plan",
    "tracked_products": "Tracked products",
}


LETTER_TEMPLATE_PREVIEW_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("subject_line", "Subject"),
    ("body_intro", "Introduction"),
    ("product_details", "Deliverables"),
    ("closing_text", "Closing"),
    ("quotation_remark", "Remarks"),
    ("follow_up_note", "Follow-up note"),
)


def letter_form_key(field: str) -> str:
    return f"quotation_letter_{field}"


def letter_form_aux_key(field: str) -> str:
    return f"quotation_letter_aux_{field}"


def default_letter_values(user: Dict) -> Dict[str, Any]:
    """Baseline values for a new quotation letter."""

    default_district = ensure_default_district()
    return {
        "reference_no": "",
        "quote_date": date.today(),
        "customer_name": "",
        "customer_company": "",
        "customer_address": "",
        "customer_contact": "",
        "district_id": default_district,
        "attention_name": "",
        "attention_title": "",
        "subject_line": "",
        "salutation": "Dear Sir,",
        "body_intro": (
            "We thank you for your inquiry and are pleased to submit our best "
            "proposal as per the below details."
        ),
        "product_details": "",
        "tracked_products": "",
        "amount": 0.0,
        "quote_type": "retail",
        "closing_text": "With Thanks & Kind Regards",
        "salesperson_name": default_salesperson_display(user),
        "salesperson_title": user.get("designation") or "",
        "salesperson_contact": user.get("phone") or "",
        "quotation_remark": "",
        "follow_up_status": "possible",
        "follow_up_date": date.today() + timedelta(days=3),
        "follow_up_note": "",
        "payment_status": "pending",
        "payment_receipt": None,
        "pdf_path": None,
    }


def _letter_field_completed(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (date, datetime)):
        return True
    if isinstance(value, (int, float)):
        return True
    return bool(value)


def summarise_letter_completion(state: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """Return completion ratio along with missing required and optional labels."""

    missing_required = [
        label
        for field, label in LETTER_REQUIRED_LABELS.items()
        if not _letter_field_completed(state.get(field))
    ]
    required_total = len(LETTER_REQUIRED_LABELS) or 1
    completion_ratio = (required_total - len(missing_required)) / required_total
    missing_optional = [
        label
        for field, label in LETTER_RECOMMENDED_LABELS.items()
        if not _letter_field_completed(state.get(field))
    ]
    return completion_ratio, missing_required, missing_optional


def fetch_company_autofill(company_id: int) -> Optional[sqlite3.Row]:
    """Return company details suitable for autofilling quotation letters."""

    with get_conn() as conn:
        return conn.execute(
            textwrap.dedent(
                """
                SELECT c.company_id,
                       c.name,
                       c.contact_person,
                       c.phone,
                       c.address,
                       c.delivery_address,
                       c.district_id,
                       c.type AS company_type,
                       d.name AS district,
                       GROUP_CONCAT(cat.name, ', ') AS products
                FROM companies c
                LEFT JOIN districts d ON d.district_id = c.district_id
                LEFT JOIN company_categories cc ON cc.company_id = c.company_id
                LEFT JOIN categories cat ON cat.category_id = cc.category_id
                WHERE c.company_id=?
                GROUP BY c.company_id
                """
            ),
            (company_id,),
        ).fetchone()


def apply_company_autofill(company_row: sqlite3.Row) -> None:
    """Populate key letter fields based on a stored company profile."""

    if not company_row:
        return

    company = dict(company_row)

    def assign(field: str, value: Optional[str]) -> None:
        if value is None:
            return
        st.session_state[letter_form_key(field)] = value

    assign("customer_company", company.get("name"))
    contact_person = company.get("contact_person")
    if contact_person:
        assign("customer_name", contact_person)
        assign("attention_name", contact_person)
    phone = company.get("phone")
    if phone:
        assign("customer_contact", phone)
    district = company.get("district")
    address_parts = [company.get("address"), district]
    address = "\n".join(part for part in address_parts if part)
    if address:
        assign("customer_address", address)
    district_id = company.get("district_id")
    if district_id is not None:
        try:
            st.session_state[letter_form_key("district_id")] = int(district_id)
        except (TypeError, ValueError):
            pass
    company_type = str(company.get("company_type") or "").lower()
    if company_type in {"retail", "wholesale"}:
        assign("quote_type", company_type)
    product_list = company.get("products")
    if product_list:
        assign("tracked_products", product_list)
    delivery_address = company.get("delivery_address")
    if delivery_address:
        current_remark = st.session_state.get(letter_form_key("quotation_remark")) or ""
        if not current_remark.strip():
            assign(
                "quotation_remark",
                f"Delivery to: {delivery_address}",
            )
    if contact_person and phone:
        follow_note_key = letter_form_key("follow_up_note")
        current_note = st.session_state.get(follow_note_key) or ""
        if not current_note.strip():
            st.session_state[follow_note_key] = (
                f"Call {contact_person} at {phone} to confirm order progress."
            )

def ensure_letter_form_state(user: Dict, existing: Optional[sqlite3.Row]) -> None:
    """Populate Streamlit session state for the quotation letter form."""

    target_id = existing["letter_id"] if existing else None
    active_id = st.session_state.get("letter_form_active_id")

    values = default_letter_values(user)
    if existing:
        record = dict(existing)
        for field in LETTER_FORM_FIELDS:
            if field not in record:
                continue
            if field in {"quote_date", "follow_up_date"} and record[field]:
                try:
                    values[field] = date.fromisoformat(record[field])
                except ValueError:
                    values[field] = date.today()
            else:
                values[field] = record[field]
    if not values.get("district_id"):
        values["district_id"] = ensure_default_district()
    quote_type_value = str(values.get("quote_type") or "retail").lower()
    if quote_type_value not in {"retail", "wholesale"}:
        quote_type_value = "retail"
    values["quote_type"] = quote_type_value
    tracked_products_value = values.get("tracked_products")
    if tracked_products_value is None:
        values["tracked_products"] = ""
    if user.get("role") == "staff":
        values["salesperson_name"] = default_salesperson_display(user)
        values["salesperson_title"] = user.get("designation") or ""
        values["salesperson_contact"] = user.get("phone") or ""
    if "letter_form_active_id" not in st.session_state or active_id != target_id:
        for field, value in values.items():
            st.session_state[letter_form_key(field)] = value
    else:
        for field, value in values.items():
            st.session_state.setdefault(letter_form_key(field), value)
    st.session_state["letter_form_active_id"] = target_id


def get_letter_form_state() -> Dict[str, Any]:
    """Return the current quotation letter data from session state."""

    state: Dict[str, Any] = {}
    for field in LETTER_FORM_FIELDS:
        state[field] = st.session_state.get(letter_form_key(field))
    return state


def _coerce_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _clean_text(value: Any, fallback: str = "") -> str:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or fallback
    if value is None:
        return fallback
    return str(value)


def build_letter_template_context(
    user: Dict, state: Optional[Dict[str, Any]] = None
) -> SafeFormatDict:
    """Construct placeholder values used when applying templates."""

    state = state or get_letter_form_state()
    quote_date_value = _coerce_date(state.get("quote_date")) or date.today()
    follow_up_value = _coerce_date(state.get("follow_up_date"))
    customer_name = _clean_text(state.get("customer_name"), "Sir/Madam")
    salesperson_name = _clean_text(
        state.get("salesperson_name"), default_salesperson_display(user)
    )
    context = SafeFormatDict(
        customer_name=customer_name,
        customer_salutation=customer_name,
        customer_company=_clean_text(state.get("customer_company"), "your organisation"),
        customer_contact_name=_clean_text(state.get("customer_name"), "your contact"),
        customer_contact_number=_clean_text(state.get("customer_contact"), "their direct line"),
        salesperson_name=salesperson_name,
        salesperson_first_name=salesperson_name.split(" ")[0]
        if salesperson_name
        else "",
        salesperson_title=_clean_text(
            state.get("salesperson_title"), user.get("designation") or "Sales Consultant"
        ),
        salesperson_contact=_clean_text(state.get("salesperson_contact"), user.get("phone") or ""),
        attention_name=_clean_text(state.get("attention_name"), ""),
        attention_title=_clean_text(state.get("attention_title"), ""),
        quote_date_long=quote_date_value.strftime("%d %B %Y"),
        current_year=str(date.today().year),
    )
    if follow_up_value:
        context["follow_up_date_long"] = follow_up_value.strftime("%d %B %Y")
    else:
        context["follow_up_date_long"] = ""
    return context


def set_follow_up_choice(choice: str) -> None:
    """Persist the quick follow-up selection and align the reminder date."""

    st.session_state[letter_form_aux_key("follow_up_choice")] = choice
    st.session_state["_letter_follow_up_last_selection"] = choice
    days = FOLLOW_UP_SUGGESTIONS.get(choice)
    if days is not None:
        st.session_state[letter_form_key("follow_up_date")] = date.today() + timedelta(days=days)


def determine_follow_up_choice(current_value: Optional[date]) -> str:
    """Return the matching quick-select label for the current follow-up date."""

    if not current_value:
        return CUSTOM_FOLLOW_UP_CHOICE
    today_value = date.today()
    for label, days in FOLLOW_UP_SUGGESTIONS.items():
        if days is None:
            continue
        if current_value == today_value + timedelta(days=days):
            return label
    return CUSTOM_FOLLOW_UP_CHOICE


def ensure_letter_aux_state() -> None:
    """Guarantee helper UI state such as quick follow-up selection exists."""

    choice_key = letter_form_aux_key("follow_up_choice")
    if choice_key in st.session_state:
        return
    current_follow_up = st.session_state.get(letter_form_key("follow_up_date"))
    current_date = _coerce_date(current_follow_up)
    choice = determine_follow_up_choice(current_date)
    set_follow_up_choice(choice)


def reset_letter_form_state(user: Dict) -> None:
    """Restore default form values for a fresh quotation letter."""

    defaults = default_letter_values(user)
    for field, value in defaults.items():
        st.session_state[letter_form_key(field)] = value
    set_follow_up_choice(DEFAULT_FOLLOW_UP_CHOICE)


def apply_letter_template(template_key: str, user: Dict) -> None:
    """Apply one of the predefined templates to the current letter."""

    template = LETTER_TEMPLATE_INDEX.get(template_key)
    if not template:
        return
    state = get_letter_form_state()
    context = build_letter_template_context(user, state)
    for field, value in template.get("fields", {}).items():
        if isinstance(value, str):
            st.session_state[letter_form_key(field)] = value.format_map(context)
        else:
            st.session_state[letter_form_key(field)] = value
    suggested_follow_up = template.get("suggested_follow_up")
    if suggested_follow_up and suggested_follow_up in FOLLOW_UP_SUGGESTIONS:
        set_follow_up_choice(suggested_follow_up)


def prepare_letter_payload(
    user: Dict, existing: Optional[sqlite3.Row], selected_id: Optional[int]
) -> Dict[str, Any]:
    """Translate session state into a payload for database persistence."""

    data = get_letter_form_state()
    if user.get("role") == "staff":
        data["salesperson_name"] = default_salesperson_display(user)
        data["salesperson_title"] = user.get("designation") or ""
        data["salesperson_contact"] = user.get("phone") or ""

    quote_date = data.get("quote_date")
    if isinstance(quote_date, date):
        data["quote_date"] = quote_date.isoformat()
    elif isinstance(quote_date, datetime):
        data["quote_date"] = quote_date.date().isoformat()
    elif isinstance(quote_date, str) and quote_date:
        try:
            data["quote_date"] = date.fromisoformat(quote_date).isoformat()
        except ValueError:
            data["quote_date"] = date.today().isoformat()
    else:
        data["quote_date"] = date.today().isoformat()

    amount = data.get("amount")
    if isinstance(amount, str):
        amount = amount.strip()
        if amount:
            try:
                data["amount"] = float(amount)
            except ValueError:
                data["amount"] = None
        else:
            data["amount"] = None
    elif isinstance(amount, (int, float)):
        data["amount"] = float(amount)
    else:
        data["amount"] = None

    follow_up_status = data.get("follow_up_status") or "possible"
    if follow_up_status not in LETTER_FOLLOW_UP_STATUSES:
        follow_up_status = "possible"
    data["follow_up_status"] = follow_up_status

    follow_up_date = data.get("follow_up_date")
    if follow_up_status == "possible" and follow_up_date:
        if isinstance(follow_up_date, datetime):
            data["follow_up_date"] = follow_up_date.date().isoformat()
        elif isinstance(follow_up_date, date):
            data["follow_up_date"] = follow_up_date.isoformat()
        elif isinstance(follow_up_date, str):
            try:
                data["follow_up_date"] = date.fromisoformat(follow_up_date).isoformat()
            except ValueError:
                data["follow_up_date"] = None
        else:
            data["follow_up_date"] = None
    else:
        data["follow_up_date"] = None

    payment_status_map = {
        "paid": "paid",
        "possible": "pending",
        "rejected": "declined",
    }
    data["payment_status"] = payment_status_map.get(follow_up_status, "pending")

    quote_type = str(data.get("quote_type") or "retail").lower()
    if quote_type not in {"retail", "wholesale"}:
        quote_type = "retail"
    data["quote_type"] = quote_type

    district_value = data.get("district_id")
    district_id: Optional[int]
    if isinstance(district_value, (int, float)):
        district_id = int(district_value)
    elif isinstance(district_value, str):
        stripped = district_value.strip()
        try:
            district_id = int(stripped)
        except ValueError:
            district_id = None
    else:
        district_id = None
    if not district_id:
        district_id = ensure_default_district()
    data["district_id"] = district_id

    tracked_products_raw = data.get("tracked_products") or ""
    tracked_names = parse_product_names(tracked_products_raw)
    data["tracked_products"] = ", ".join(tracked_names)

    data["salesperson_id"] = (
        existing["salesperson_id"] if existing else user["user_id"]
    )
    data["letter_id"] = selected_id

    return data


def render_letter_preview(state: Dict[str, Any]) -> None:
    """Render a visual preview of the quotation letter on company letterhead."""

    if LETTERHEAD_BASE64:
        style = f"""
        <style>
        .letter-preview-wrapper {{
            max-width: 850px;
            margin: 0 auto;
        }}
        .letter-preview {{
            background-image: url('data:image/png;base64,{LETTERHEAD_BASE64}');
            background-size: cover;
            background-repeat: no-repeat;
            min-height: 1100px;
            padding: 190px 80px 140px 80px;
            box-sizing: border-box;
            font-family: "Times New Roman", serif;
            color: #1f1f1f;
            line-height: 1.6;
            font-size: 16px;
        }}
        .letter-preview .meta {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 24px;
            font-weight: 600;
        }}
        .letter-preview .section {{
            margin-bottom: 18px;
        }}
        .letter-preview .subject {{
            font-weight: 700;
            text-decoration: underline;
            margin-bottom: 16px;
        }}
        .letter-preview .signature {{
            margin-top: 40px;
            font-weight: 600;
        }}
        .letter-preview .remarks-box {{
            margin-top: 32px;
            padding: 12px 16px;
            border: 1px solid #d0d0d0;
            background-color: rgba(255, 255, 255, 0.85);
        }}
        </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    else:
        st.warning("Letterhead image missing. Upload ps_letterhead.png to enable preview.")

    def esc(value: Any) -> str:
        return html.escape(value or "")

    def format_multiline(value: Any) -> str:
        if not value:
            return ""
        return "<br>".join(esc(line) for line in str(value).splitlines())

    def format_date(value: Any) -> str:
        if isinstance(value, date):
            return value.strftime("%d-%m-%Y")
        if isinstance(value, datetime):
            return value.strftime("%d-%m-%Y")
        if isinstance(value, str) and value:
            try:
                parsed = date.fromisoformat(value)
                return parsed.strftime("%d-%m-%Y")
            except ValueError:
                return esc(value)
        return date.today().strftime("%d-%m-%Y")

    date_str = format_date(state.get("quote_date"))
    reference = esc(state.get("reference_no"))
    salutation = esc(state.get("salutation") or "Dear Sir,")
    subject_line = esc(state.get("subject_line") or "Quotation")
    amount_value = state.get("amount")
    if isinstance(amount_value, (int, float)):
        amount_display = f"Amount: BDT {amount_value:,.2f}"
    else:
        amount_display = ""

    address_lines = [
        state.get("customer_name"),
        state.get("customer_company"),
        state.get("customer_address"),
    ]
    if state.get("customer_contact"):
        address_lines.append(f"Contact: {state['customer_contact']}")
    address_html = "<br>".join(esc(line) for line in address_lines if line)

    attention_parts = []
    if state.get("attention_name"):
        name_line = f"Attention: {esc(state['attention_name'])}"
        attention_parts.append(f"<strong>{name_line}</strong>")
    if state.get("attention_title"):
        attention_parts.append(f"<span>{esc(state['attention_title'])}</span>")
    attention_html = "<br>".join(attention_parts)

    product_html = format_multiline(state.get("product_details")) or ""
    body_html = format_multiline(state.get("body_intro"))
    closing_html = format_multiline(state.get("closing_text")) or "With Thanks &amp; Kind Regards"

    signature_lines = [state.get("salesperson_name"), state.get("salesperson_title")]
    if state.get("salesperson_contact"):
        signature_lines.append(state["salesperson_contact"])
    signature_html = "<br>".join(esc(line) for line in signature_lines if line)

    remark_html = format_multiline(state.get("quotation_remark"))
    follow_up_status = LETTER_FOLLOW_UP_LABELS.get(
        state.get("follow_up_status"), ""
    )
    follow_up_note = format_multiline(state.get("follow_up_note"))

    content = [
        "<div class=\"letter-preview-wrapper\">",
        "<div class=\"letter-preview\">",
        f"<div class=\"meta\"><span>Date: {date_str}</span><span>Ref: {reference}</span></div>",
        f"<div class=\"section\"><strong>To</strong><br>{address_html}</div>",
    ]
    if attention_html:
        content.append(f"<div class=\"section\">{attention_html}</div>")
    content.append(
        f"<div class=\"section subject\"><strong>Subject:</strong> {subject_line}</div>"
    )
    content.append(f"<div class=\"section\">{salutation}</div>")
    if body_html:
        content.append(f"<div class=\"section\">{body_html}</div>")
    if product_html:
        content.append(f"<div class=\"section\">{product_html}</div>")
    if amount_display:
        content.append(f"<div class=\"section\"><strong>{amount_display}</strong></div>")
    if closing_html:
        content.append(f"<div class=\"section\">{closing_html}</div>")
    if signature_html:
        content.append(f"<div class=\"signature\">{signature_html}</div>")
    remarks_section: List[str] = []
    if remark_html:
        remarks_section.append(
            f"<div><strong>Quotation remarks:</strong><br>{remark_html}</div>"
        )
    follow_up_parts: List[str] = []
    if follow_up_status:
        follow_up_parts.append(
            f"<div><strong>Salesperson follow-up:</strong> {esc(follow_up_status)}</div>"
        )
    if follow_up_note:
        follow_up_parts.append(f"<div>{follow_up_note}</div>")
    if remarks_section or follow_up_parts:
        merged = "".join(remarks_section + follow_up_parts)
        content.append(f"<div class=\"remarks-box\">{merged}</div>")
    content.append("</div></div>")

    st.markdown("".join(content), unsafe_allow_html=True)


def save_uploaded_file(uploaded_file, subdir: str) -> Optional[str]:
    return UPLOAD_MANAGER.save(uploaded_file, subdir)


def generate_letter_pdf(state: Dict[str, Any]) -> bytes:
    """Build a PDF document for the supplied quotation letter state."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=35 * mm,
        bottomMargin=25 * mm,
    )
    styles = getSampleStyleSheet()
    normal = styles["BodyText"].clone("LetterBody")
    normal.leading = 16
    normal.spaceAfter = 12
    bold_style = styles["Heading4"].clone("LetterSubject")
    bold_style.fontSize = 12
    bold_style.spaceAfter = 16
    meta_style = styles["Normal"].clone("Meta")
    meta_style.spaceAfter = 12

    def format_date(value: Any) -> str:
        if isinstance(value, datetime):
            return value.date().strftime("%d-%m-%Y")
        if isinstance(value, date):
            return value.strftime("%d-%m-%Y")
        if isinstance(value, str) and value:
            try:
                return date.fromisoformat(value).strftime("%d-%m-%Y")
            except ValueError:
                return value
        return date.today().strftime("%d-%m-%Y")

    def format_block(value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, str):
            lines = [line.strip() for line in value.splitlines() if line.strip()]
        else:
            lines = [str(value)]
        if not lines:
            return ""
        return "<br/>".join(html.escape(line) for line in lines)

    background_image: Optional[ImageReader] = None
    letterhead_path = Path(__file__).with_name("ps_letterhead.png")
    if letterhead_path.exists():
        try:
            background_image = ImageReader(str(letterhead_path))
        except Exception:
            background_image = None

    def draw_letterhead(canvas_obj, document) -> None:
        if not background_image:
            return
        canvas_obj.saveState()
        page_width, page_height = document.pagesize
        try:
            img_width, img_height = background_image.getSize()
        except Exception:
            img_width = img_height = 0
        if img_width and img_height:
            scale = max(
                page_width / float(img_width), page_height / float(img_height)
            )
            draw_width = float(img_width) * scale
            draw_height = float(img_height) * scale
        else:
            draw_width = page_width
            draw_height = page_height
        x_position = (page_width - draw_width) / 2.0
        y_position = (page_height - draw_height) / 2.0
        canvas_obj.drawImage(
            background_image,
            x_position,
            y_position,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=False,
            mask="auto",
        )
        canvas_obj.restoreState()

    elements: List[Any] = []

    date_str = format_date(state.get("quote_date"))
    reference = html.escape(str(state.get("reference_no") or ""))
    meta_text = f"Date: {date_str}    &nbsp;&nbsp; Reference: {reference}"
    elements.append(Paragraph(meta_text, meta_style))

    address_lines = [
        state.get("customer_name"),
        state.get("customer_company"),
        state.get("customer_address"),
    ]
    if state.get("customer_contact"):
        address_lines.append(f"Contact: {state['customer_contact']}")
    address_block = format_block("\n".join(filter(None, address_lines)))
    if address_block:
        elements.append(Paragraph(f"<strong>To</strong><br/>{address_block}", normal))

    attention_lines = []
    if state.get("attention_name"):
        attention_lines.append(f"Attention: {state['attention_name']}")
    if state.get("attention_title"):
        attention_lines.append(state["attention_title"])
    attention_block = format_block("\n".join(attention_lines))
    if attention_block:
        elements.append(Paragraph(attention_block, normal))

    salutation = html.escape(state.get("salutation") or "Dear Sir,")
    elements.append(Paragraph(salutation, normal))

    subject_line = html.escape(state.get("subject_line") or "Quotation")
    elements.append(Paragraph(subject_line, bold_style))

    intro = format_block(state.get("body_intro"))
    if intro:
        elements.append(Paragraph(intro, normal))

    product_details = format_block(state.get("product_details"))
    if product_details:
        elements.append(Paragraph(product_details, normal))

    amount = state.get("amount")
    if isinstance(amount, (int, float)):
        elements.append(Paragraph(f"<strong>Amount:</strong> BDT {amount:,.2f}", normal))

    closing = format_block(state.get("closing_text")) or "With Thanks &amp; Kind Regards"
    elements.append(Paragraph(closing, normal))

    signature_parts = [
        state.get("salesperson_name"),
        state.get("salesperson_title"),
        state.get("salesperson_contact"),
    ]
    signature_block = format_block("\n".join(filter(None, signature_parts)))
    if signature_block:
        elements.append(Paragraph(signature_block, normal))

    remarks_block = format_block(state.get("quotation_remark"))
    follow_up_status_key = state.get("follow_up_status")
    follow_up_status = LETTER_FOLLOW_UP_LABELS.get(follow_up_status_key)
    follow_up_note = format_block(state.get("follow_up_note"))
    follow_up_date = format_date(state.get("follow_up_date")) if state.get("follow_up_date") else ""
    payment_status = html.escape(state.get("payment_status") or "")

    # Remarks, follow-up plans and payment flags are presented in the on-screen
    # preview but deliberately excluded from the exported PDF so that customers
    # receive a clean quotation without internal annotations.

    doc.build(elements, onFirstPage=draw_letterhead, onLaterPages=draw_letterhead)
    buffer.seek(0)
    return buffer.getvalue()


def persist_letter_pdf(letter_id: int, state: Dict[str, Any]) -> str:
    """Generate and store a PDF for the given quotation letter."""

    pdf_bytes = generate_letter_pdf(state)
    target_dir = CONFIG.data_dir / "uploads" / "quotation_letters"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"quotation_letter_{letter_id}.pdf"
    path = target_dir / filename
    with open(path, "wb") as handle:
        handle.write(pdf_bytes)
    return str(path.relative_to(CONFIG.data_dir))


def ensure_default_district() -> int:
    """Return the district ID used for auto-created quotation companies."""

    with get_conn() as conn:
        row = conn.execute(
            "SELECT district_id FROM districts WHERE lower(name)=lower(?)",
            ("Unknown",),
        ).fetchone()
    if row:
        return int(row["district_id"])
    with get_cursor() as cur:
        cur.execute("INSERT INTO districts(name) VALUES (?)", ("Unknown",))
        return int(cur.lastrowid)


def ensure_default_category() -> int:
    """Return a generic category identifier for auto-created quotations."""

    with get_conn() as conn:
        row = conn.execute(
            "SELECT category_id FROM categories WHERE lower(name)=lower(?)",
            ("general",),
        ).fetchone()
    if row:
        return int(row["category_id"])
    with get_cursor() as cur:
        cur.execute("INSERT INTO categories(name) VALUES (?)", ("General",))
        return int(cur.lastrowid)


def ensure_company_for_letter(data: Dict[str, Any]) -> int:
    """Create or find a company record matching the quotation letter details."""

    company_name = (data.get("customer_company") or "").strip() or "Unknown Company"
    quote_type = str(data.get("quote_type") or "retail").lower()
    if quote_type not in {"retail", "wholesale"}:
        quote_type = "retail"
    district_value = data.get("district_id")
    if isinstance(district_value, (int, float)):
        requested_district = int(district_value)
    elif isinstance(district_value, str):
        try:
            requested_district = int(district_value.strip())
        except ValueError:
            requested_district = None
    else:
        requested_district = None
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT company_id, district_id FROM companies WHERE lower(name)=lower(?)",
            (company_name.lower(),),
        ).fetchone()
    district_id = requested_district or ensure_default_district()
    if existing:
        if not existing["district_id"] or (
            requested_district and existing["district_id"] != requested_district
        ):
            with get_cursor() as cur:
                cur.execute(
                    "UPDATE companies SET district_id=?, type=? WHERE company_id=?",
                    (district_id, quote_type, existing["company_id"]),
                )
        elif quote_type:
            with get_cursor() as cur:
                cur.execute(
                    "UPDATE companies SET type=? WHERE company_id=?",
                    (quote_type, existing["company_id"]),
                )
        return int(existing["company_id"])

    contact_person = (data.get("customer_name") or "").strip() or None
    phone = (data.get("customer_contact") or "").strip() or None
    address = (data.get("customer_address") or "").strip() or None
    with get_cursor() as cur:
        cur.execute(
            textwrap.dedent(
                """
                INSERT INTO companies(name, contact_person, phone, address, delivery_address, district_id, type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
            ),
            (company_name, contact_person, phone, address, address, district_id, quote_type),
        )
        company_id = int(cur.lastrowid)
    category_id = ensure_default_category()
    link_company_product(company_id, category_id)
    return company_id


def update_letter_pdf_path(letter_id: int, pdf_path: str) -> None:
    with get_cursor() as cur:
        cur.execute(
            "UPDATE quotation_letters SET pdf_path=? WHERE letter_id=?",
            (pdf_path, letter_id),
        )


def sync_letter_tracking(letter_id: int, data: Dict[str, Any]) -> Tuple[int, Optional[str]]:
    """Ensure a corresponding quotation record exists for downstream workflows."""

    company_id = ensure_company_for_letter(data)
    with get_conn() as conn:
        company_row = conn.execute(
            "SELECT district_id FROM companies WHERE company_id=?",
            (company_id,),
        ).fetchone()
        existing = conn.execute(
            "SELECT quotation_id, payment_status FROM quotations WHERE letter_id=?",
            (letter_id,),
        ).fetchone()
    district_value = data.get("district_id")
    if isinstance(district_value, (int, float)):
        district_candidate = int(district_value)
    elif isinstance(district_value, str):
        try:
            district_candidate = int(district_value.strip())
        except ValueError:
            district_candidate = None
    else:
        district_candidate = None
    if district_candidate and district_candidate > 0:
        district_id = district_candidate
    elif company_row and company_row["district_id"]:
        district_id = int(company_row["district_id"])
    else:
        district_id = ensure_default_district()
    status_map = {
        "paid": "accepted",
        "possible": "pending",
        "rejected": "declined",
    }
    status = status_map.get(data.get("follow_up_status"), "pending")
    follow_up_date = data.get("follow_up_date")
    salesperson_id = data.get("salesperson_id")
    quote_type = str(data.get("quote_type") or "retail").lower()
    if quote_type not in {"retail", "wholesale"}:
        quote_type = "retail"

    tracked_names = parse_product_names(data.get("tracked_products"))
    if not tracked_names:
        raw_details = (data.get("product_details") or "").replace("•", " ")
        tracked_names = parse_product_names(raw_details)

    product_ids: List[int] = []
    for name in tracked_names:
        try:
            identifier = ensure_product(name)
        except ValueError:
            continue
        product_ids.append(int(identifier))

    if product_ids:
        category_id = product_ids[0]
    else:
        category_id = ensure_default_category()

    quantity_value = max(len(product_ids), 1)
    quotation_data = (
        salesperson_id,
        company_id,
        district_id,
        category_id,
        data.get("quote_date"),
        status,
        follow_up_date,
        quote_type,
        None,
        data.get("pdf_path"),
        data.get("quotation_remark"),
        data.get("payment_status"),
        data.get("payment_receipt"),
        quantity_value,
        letter_id,
    )
    with get_cursor() as cur:
        if existing:
            cur.execute(
                textwrap.dedent(
                    """
                    UPDATE quotations
                    SET salesperson_id=?,
                        company_id=?,
                        district_id=?,
                        category_id=?,
                        quote_date=?,
                        status=?,
                        follow_up_date=?,
                        quote_type=?,
                        kva=?,
                        pdf_path=?,
                        notes=?,
                        payment_status=?,
                        payment_receipt=?,
                        quantity=?
                    WHERE letter_id=?
                    """
                ),
                quotation_data,
            )
            quotation_id = int(existing["quotation_id"])
            previous_status = existing["payment_status"]
        else:
            cur.execute(
                textwrap.dedent(
                    """
                    INSERT INTO quotations(
                        salesperson_id,
                        company_id,
                        district_id,
                        category_id,
                        quote_date,
                        status,
                        follow_up_date,
                        quote_type,
                        kva,
                        pdf_path,
                        notes,
                        payment_status,
                        payment_receipt,
                        quantity,
                        letter_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                quotation_data,
            )
            quotation_id = int(cur.lastrowid)
            previous_status = None
    set_quotation_products(quotation_id, [(pid, 1) for pid in product_ids])
    for pid in product_ids:
        link_company_product(company_id, pid)
    return quotation_id, previous_status


st.set_page_config(page_title="PS Business Suites by ZAD", page_icon="📊", layout="wide")


def rerun() -> None:
    """Trigger a Streamlit rerun across supported versions."""

    if hasattr(st, "rerun"):
        st.rerun()
        return
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        return
    raise RuntimeError("Streamlit rerun function not available")


def resolve_upload_path(relative_path: str) -> Path:
    """Resolve a relative upload path to an absolute location inside DATA_DIR."""

    target = (CONFIG.data_dir / relative_path).resolve()
    if CONFIG.data_dir not in target.parents and target != CONFIG.data_dir:
        raise ValueError("Invalid upload path outside data directory")
    return target


# ---------------------------------------------------------------------------
# Database helpers and initialisation
# ---------------------------------------------------------------------------


def get_conn() -> sqlite3.Connection:
    return DATABASE.raw_connection()


@contextmanager
def get_cursor() -> Iterable[sqlite3.Cursor]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    finally:
        cur.close()
        conn.close()


BANGLADESH_DISTRICTS = [
    "Bagerhat",
    "Bandarban",
    "Barguna",
    "Barishal",
    "Bhola",
    "Bogura",
    "Brahmanbaria",
    "Chandpur",
    "Chattogram",
    "Chuadanga",
    "Cox's Bazar",
    "Cumilla",
    "Dhaka",
    "Dinajpur",
    "Faridpur",
    "Feni",
    "Gaibandha",
    "Gazipur",
    "Gopalganj",
    "Habiganj",
    "Jamalpur",
    "Jashore",
    "Jhalokathi",
    "Jhenaidah",
    "Joypurhat",
    "Khagrachhari",
    "Khulna",
    "Kishoreganj",
    "Kurigram",
    "Kushtia",
    "Lakshmipur",
    "Lalmonirhat",
    "Madaripur",
    "Magura",
    "Manikganj",
    "Meherpur",
    "Moulvibazar",
    "Munshiganj",
    "Mymensingh",
    "Naogaon",
    "Narail",
    "Narayanganj",
    "Narsingdi",
    "Natore",
    "Nawabganj",
    "Netrakona",
    "Nilphamari",
    "Noakhali",
    "Pabna",
    "Panchagarh",
    "Patuakhali",
    "Pirojpur",
    "Rajbari",
    "Rajshahi",
    "Rangamati",
    "Rangpur",
    "Satkhira",
    "Shariatpur",
    "Sherpur",
    "Sirajganj",
    "Sunamganj",
    "Sylhet",
    "Tangail",
    "Thakurgaon",
]


def hash_password(password: str) -> str:
    return PASSWORD_SERVICE.hash(password)


PRODUCT_LIST_SUBQUERY = textwrap.dedent(
    """
    SELECT qp.quotation_id,
           GROUP_CONCAT(cat.name || ' (x' || qp.quantity || ')', ', ') AS names
    FROM (
        SELECT quotation_id, line_no, category_id, quantity
        FROM quotation_products
        ORDER BY quotation_id, line_no
    ) AS qp
    JOIN categories cat ON cat.category_id = qp.category_id
    GROUP BY qp.quotation_id
    """
)


def init_db() -> None:
    with get_cursor() as cur:
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                pass_hash     TEXT NOT NULL,
                display_name  TEXT,
                designation   TEXT,
                phone         TEXT,
                role          TEXT NOT NULL CHECK(role IN ('admin', 'staff')),
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS districts (
                district_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS companies (
                company_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT NOT NULL,
                contact_person TEXT,
                phone          TEXT,
                address        TEXT,
                delivery_address TEXT,
                district_id    INTEGER NOT NULL,
                type           TEXT NOT NULL CHECK(type IN ('retail', 'wholesale')),
                created_at     TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(district_id) REFERENCES districts(district_id)
            );

            CREATE TABLE IF NOT EXISTS company_categories (
                company_id  INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                PRIMARY KEY (company_id, category_id),
                FOREIGN KEY(company_id) REFERENCES companies(company_id) ON DELETE CASCADE,
                FOREIGN KEY(category_id) REFERENCES categories(category_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS quotations (
                quotation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                salesperson_id INTEGER NOT NULL,
                company_id     INTEGER NOT NULL,
                district_id    INTEGER NOT NULL,
                category_id    INTEGER NOT NULL,
                quote_date     TEXT NOT NULL,
                status         TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'declined', 'inform_later')) DEFAULT 'pending',
                follow_up_date TEXT,
                quote_type     TEXT NOT NULL CHECK(quote_type IN ('retail', 'wholesale')),
                kva            REAL,
                pdf_path       TEXT,
                notes          TEXT,
                payment_status TEXT NOT NULL CHECK(payment_status IN ('pending', 'paid', 'declined')) DEFAULT 'pending',
                payment_receipt TEXT,
                quantity       INTEGER NOT NULL DEFAULT 1,
                letter_id      INTEGER UNIQUE,
                created_at     TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(salesperson_id) REFERENCES users(user_id),
                FOREIGN KEY(company_id) REFERENCES companies(company_id),
                FOREIGN KEY(district_id) REFERENCES districts(district_id),
                FOREIGN KEY(category_id) REFERENCES categories(category_id),
                FOREIGN KEY(letter_id) REFERENCES quotation_letters(letter_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS quotation_products (
                item_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                quotation_id  INTEGER NOT NULL,
                line_no       INTEGER NOT NULL,
                category_id   INTEGER NOT NULL,
                quantity      INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(quotation_id) REFERENCES quotations(quotation_id) ON DELETE CASCADE,
                FOREIGN KEY(category_id) REFERENCES categories(category_id) ON DELETE CASCADE,
                UNIQUE(quotation_id, line_no)
            );

            CREATE TABLE IF NOT EXISTS work_orders (
                work_order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                quotation_id  INTEGER NOT NULL,
                upload_date   TEXT NOT NULL,
                pdf_path      TEXT,
                notes         TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(quotation_id) REFERENCES quotations(quotation_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS delivery_orders (
                do_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                work_order_id    INTEGER NOT NULL,
                do_number        TEXT NOT NULL,
                upload_date      TEXT NOT NULL,
                pdf_path         TEXT,
                price            REAL NOT NULL DEFAULT 0,
                payment_received INTEGER NOT NULL DEFAULT 0,
                payment_date     TEXT,
                notes            TEXT,
                receipt_path     TEXT,
                created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(work_order_id) REFERENCES work_orders(work_order_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS quotation_letters (
                letter_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                salesperson_id       INTEGER NOT NULL,
                reference_no         TEXT NOT NULL,
                quote_date           TEXT NOT NULL,
                customer_name        TEXT NOT NULL,
                customer_company     TEXT NOT NULL,
                customer_address     TEXT NOT NULL,
                customer_contact     TEXT,
                district_id          INTEGER,
                attention_name       TEXT,
                attention_title      TEXT,
                subject_line         TEXT NOT NULL,
                salutation           TEXT,
                body_intro           TEXT,
                product_details      TEXT,
                tracked_products     TEXT,
                amount               REAL,
                quote_type           TEXT NOT NULL CHECK(quote_type IN ('retail', 'wholesale')) DEFAULT 'retail',
                closing_text         TEXT,
                salesperson_name     TEXT NOT NULL,
                salesperson_title    TEXT,
                salesperson_contact  TEXT,
                quotation_remark     TEXT,
                follow_up_status     TEXT NOT NULL CHECK(follow_up_status IN ('paid', 'possible', 'rejected')),
                follow_up_note       TEXT,
                follow_up_date       TEXT,
                payment_status       TEXT NOT NULL CHECK(payment_status IN ('pending', 'paid', 'declined')) DEFAULT 'pending',
                payment_receipt      TEXT,
                pdf_path             TEXT,
                created_at           TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(salesperson_id) REFERENCES users(user_id),
                FOREIGN KEY(district_id) REFERENCES districts(district_id)
            );

            CREATE TABLE IF NOT EXISTS notifications (
                notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL,
                message         TEXT NOT NULL,
                due_date        TEXT NOT NULL,
                read            INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS login_events (
                event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT NOT NULL,
                success     INTEGER NOT NULL,
                occurred_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        cur.execute("PRAGMA table_info(users)")
        user_columns = {row[1] for row in cur.fetchall()}
        if "display_name" not in user_columns:
            cur.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
        if "designation" not in user_columns:
            cur.execute("ALTER TABLE users ADD COLUMN designation TEXT")
        if "phone" not in user_columns:
            cur.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        cur.execute(
            """
            UPDATE users
               SET display_name = COALESCE(NULLIF(TRIM(display_name), ''), username)
             WHERE display_name IS NULL OR TRIM(display_name) = ''
            """
        )
        cur.execute("PRAGMA table_info(companies)")
        company_columns = {row[1] for row in cur.fetchall()}
        if "delivery_address" not in company_columns:
            cur.execute("ALTER TABLE companies ADD COLUMN delivery_address TEXT")
        cur.execute("PRAGMA table_info(quotations)")
        columns = {row[1] for row in cur.fetchall()}
        if "kva" not in columns:
            cur.execute("ALTER TABLE quotations ADD COLUMN kva REAL")
        if "quantity" not in columns:
            cur.execute(
                "ALTER TABLE quotations ADD COLUMN quantity INTEGER NOT NULL DEFAULT 1"
            )
        if "payment_status" not in columns:
            cur.execute(
                "ALTER TABLE quotations ADD COLUMN payment_status TEXT NOT NULL DEFAULT 'pending'"
            )
        if "payment_receipt" not in columns:
            cur.execute("ALTER TABLE quotations ADD COLUMN payment_receipt TEXT")
        if "letter_id" not in columns:
            cur.execute("ALTER TABLE quotations ADD COLUMN letter_id INTEGER UNIQUE")

        cur.execute("PRAGMA table_info(quotation_products)")
        qp_columns = {row[1] for row in cur.fetchall()}
        if "quantity" not in qp_columns:
            cur.execute(
                "ALTER TABLE quotation_products ADD COLUMN quantity INTEGER NOT NULL DEFAULT 1"
            )

        cur.execute("PRAGMA table_info(quotation_letters)")
        letter_columns = {row[1] for row in cur.fetchall()}
        if "follow_up_date" not in letter_columns:
            cur.execute("ALTER TABLE quotation_letters ADD COLUMN follow_up_date TEXT")
        if "payment_status" not in letter_columns:
            cur.execute(
                "ALTER TABLE quotation_letters ADD COLUMN payment_status TEXT NOT NULL DEFAULT 'pending'"
            )
        if "payment_receipt" not in letter_columns:
            cur.execute("ALTER TABLE quotation_letters ADD COLUMN payment_receipt TEXT")
        if "pdf_path" not in letter_columns:
            cur.execute("ALTER TABLE quotation_letters ADD COLUMN pdf_path TEXT")
        if "district_id" not in letter_columns:
            cur.execute("ALTER TABLE quotation_letters ADD COLUMN district_id INTEGER")
        if "tracked_products" not in letter_columns:
            cur.execute("ALTER TABLE quotation_letters ADD COLUMN tracked_products TEXT")
        if "quote_type" not in letter_columns:
            cur.execute(
                "ALTER TABLE quotation_letters ADD COLUMN quote_type TEXT NOT NULL DEFAULT 'retail'"
            )

        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='quotation_letters'"
        )
        letter_table_definition = cur.fetchone()
        if (
            letter_table_definition
            and letter_table_definition[0]
            and "follow_up_status IN ('inform_later'" in letter_table_definition[0]
        ):
            cur.executescript(
                textwrap.dedent(
                    """
                    ALTER TABLE quotation_letters RENAME TO quotation_letters_old;

                    CREATE TABLE quotation_letters (
                        letter_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        salesperson_id       INTEGER NOT NULL,
                        reference_no         TEXT NOT NULL,
                        quote_date           TEXT NOT NULL,
                        customer_name        TEXT NOT NULL,
                        customer_company     TEXT NOT NULL,
                        customer_address     TEXT NOT NULL,
                        customer_contact     TEXT,
                        district_id          INTEGER,
                        attention_name       TEXT,
                        attention_title      TEXT,
                        subject_line         TEXT NOT NULL,
                        salutation           TEXT,
                        body_intro           TEXT,
                        product_details      TEXT,
                        tracked_products     TEXT,
                        amount               REAL,
                        quote_type           TEXT NOT NULL CHECK(quote_type IN ('retail', 'wholesale')) DEFAULT 'retail',
                        closing_text         TEXT,
                        salesperson_name     TEXT NOT NULL,
                        salesperson_title    TEXT,
                        salesperson_contact  TEXT,
                        quotation_remark     TEXT,
                        follow_up_status     TEXT NOT NULL CHECK(follow_up_status IN ('paid', 'possible', 'rejected')),
                        follow_up_note       TEXT,
                        follow_up_date       TEXT,
                        payment_status       TEXT NOT NULL CHECK(payment_status IN ('pending', 'paid', 'declined')) DEFAULT 'pending',
                        payment_receipt      TEXT,
                        pdf_path             TEXT,
                        created_at           TEXT NOT NULL DEFAULT (datetime('now')),
                        FOREIGN KEY(salesperson_id) REFERENCES users(user_id),
                        FOREIGN KEY(district_id) REFERENCES districts(district_id)
                    );

                    INSERT INTO quotation_letters (
                        letter_id,
                        salesperson_id,
                        reference_no,
                        quote_date,
                        customer_name,
                        customer_company,
                        customer_address,
                        customer_contact,
                        district_id,
                        attention_name,
                        attention_title,
                        subject_line,
                        salutation,
                        body_intro,
                        product_details,
                        tracked_products,
                        amount,
                        quote_type,
                        closing_text,
                        salesperson_name,
                        salesperson_title,
                        salesperson_contact,
                        quotation_remark,
                        follow_up_status,
                        follow_up_note,
                        follow_up_date,
                        payment_status,
                        payment_receipt,
                        pdf_path,
                        created_at
                    )
                    SELECT
                        letter_id,
                        salesperson_id,
                        reference_no,
                        quote_date,
                        customer_name,
                        customer_company,
                        customer_address,
                        customer_contact,
                        NULL,
                        attention_name,
                        attention_title,
                        subject_line,
                        salutation,
                        body_intro,
                        product_details,
                        NULL,
                        amount,
                        'retail',
                        closing_text,
                        salesperson_name,
                        salesperson_title,
                        salesperson_contact,
                        quotation_remark,
                        CASE follow_up_status WHEN 'inform_later' THEN 'possible' ELSE follow_up_status END,
                        follow_up_note,
                        follow_up_date,
                        payment_status,
                        payment_receipt,
                        pdf_path,
                        created_at
                    FROM quotation_letters_old;

                    DROP TABLE quotation_letters_old;
                    """
                )
            )

        cur.execute("PRAGMA table_info(delivery_orders)")
        delivery_columns = {row[1] for row in cur.fetchall()}
        if "source_type" not in delivery_columns:
            cur.executescript(
                textwrap.dedent(
                    """
                    ALTER TABLE delivery_orders RENAME TO delivery_orders_old;

                    CREATE TABLE delivery_orders (
                        do_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_type      TEXT NOT NULL CHECK(source_type IN ('work_order', 'quotation', 'third_party')),
                        salesperson_id   INTEGER NOT NULL,
                        work_order_id    INTEGER,
                        quotation_id     INTEGER,
                        third_party_name TEXT,
                        do_number        TEXT NOT NULL,
                        upload_date      TEXT NOT NULL,
                        pdf_path         TEXT,
                        price            REAL NOT NULL DEFAULT 0,
                        payment_received INTEGER NOT NULL DEFAULT 0,
                        payment_date     TEXT,
                        notes            TEXT,
                        receipt_path     TEXT,
                        created_at       TEXT NOT NULL DEFAULT (datetime('now')),
                        FOREIGN KEY(work_order_id) REFERENCES work_orders(work_order_id) ON DELETE SET NULL,
                        FOREIGN KEY(quotation_id) REFERENCES quotations(quotation_id) ON DELETE SET NULL,
                        FOREIGN KEY(salesperson_id) REFERENCES users(user_id) ON DELETE CASCADE
                    );

                    INSERT INTO delivery_orders(
                        do_id, source_type, salesperson_id, work_order_id, quotation_id,
                        third_party_name, do_number, upload_date, pdf_path, price,
                        payment_received, payment_date, notes, receipt_path, created_at
                    )
                    SELECT
                        old.do_id,
                        'work_order',
                        q.salesperson_id,
                        old.work_order_id,
                        w.quotation_id,
                        NULL,
                        old.do_number,
                        old.upload_date,
                        old.pdf_path,
                        old.price,
                        old.payment_received,
                        old.payment_date,
                        old.notes,
                        NULL,
                        old.created_at
                    FROM delivery_orders_old old
                    JOIN work_orders w ON w.work_order_id = old.work_order_id
                    JOIN quotations q ON q.quotation_id = w.quotation_id;

                    DROP TABLE delivery_orders_old;
                    """
                )
            )
            cur.execute("PRAGMA table_info(delivery_orders)")
            delivery_columns = {row[1] for row in cur.fetchall()}
        if "receipt_path" not in delivery_columns:
            cur.execute("ALTER TABLE delivery_orders ADD COLUMN receipt_path TEXT")

    with get_cursor() as cur:
        cur.executemany(
            "INSERT OR IGNORE INTO districts(name) VALUES (?)",
            [(d,) for d in BANGLADESH_DISTRICTS],
        )

    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            default_users = [
                ("admin", "admin", "admin", "Admin", None, None),
                ("salesperson", "admin", "staff", "Salesperson", None, None),
            ]
            cur.executemany(
                textwrap.dedent(
                    """
                    INSERT INTO users(username, pass_hash, role, display_name, designation, phone)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """
                ),
                [
                    (
                        username,
                        hash_password(password),
                        role,
                        display_name,
                        designation,
                        phone,
                    )
                    for username, password, role, display_name, designation, phone in default_users
                ],
            )

    default_settings = {
        "work_order_grace_days": "7",
        "delivery_order_grace_days": "7",
        "payment_due_days": "14",
        "quotation_pending_days": "10",
    }
    with get_cursor() as cur:
        for key, value in default_settings.items():
            cur.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, value)
            )

# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------


def fetchall_df(query: str, params: Tuple = ()) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def list_quotation_letters(user: Dict) -> pd.DataFrame:
    """Return quotation letters visible to the current user."""

    base_query = textwrap.dedent(
        """
        SELECT l.*, COALESCE(u.display_name, u.username) AS salesperson
        FROM quotation_letters l
        JOIN users u ON u.user_id = l.salesperson_id
        {where}
        ORDER BY l.quote_date DESC, l.letter_id DESC
        """
    )
    if user["role"] == "admin":
        return fetchall_df(base_query.format(where=""))
    return fetchall_df(
        base_query.format(where="WHERE l.salesperson_id = ?"),
        (user["user_id"],),
    )


def get_quotation_letter(letter_id: Optional[int]) -> Optional[sqlite3.Row]:
    if letter_id is None:
        return None
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM quotation_letters WHERE letter_id = ?", (letter_id,)
        )
        row = cur.fetchone()
    return row


def upsert_quotation_letter(data: Dict) -> int:
    """Create or update a quotation letter entry."""

    if data.get("letter_id"):
        with get_cursor() as cur:
            cur.execute(
                textwrap.dedent(
                    """
                    UPDATE quotation_letters
                    SET reference_no = ?,
                        quote_date = ?,
                        customer_name = ?,
                        customer_company = ?,
                        customer_address = ?,
                        customer_contact = ?,
                        district_id = ?,
                        attention_name = ?,
                        attention_title = ?,
                        subject_line = ?,
                        salutation = ?,
                        body_intro = ?,
                        product_details = ?,
                        tracked_products = ?,
                        amount = ?,
                        quote_type = ?,
                        closing_text = ?,
                        salesperson_name = ?,
                        salesperson_title = ?,
                        salesperson_contact = ?,
                        quotation_remark = ?,
                        follow_up_status = ?,
                        follow_up_note = ?,
                        follow_up_date = ?,
                        payment_status = ?,
                        payment_receipt = ?,
                        pdf_path = ?
                    WHERE letter_id = ?
                    """
                ),
                (
                    data["reference_no"],
                    data["quote_date"],
                    data["customer_name"],
                    data["customer_company"],
                    data["customer_address"],
                    data.get("customer_contact"),
                    data.get("district_id"),
                    data.get("attention_name"),
                    data.get("attention_title"),
                    data["subject_line"],
                    data.get("salutation"),
                    data.get("body_intro"),
                    data.get("product_details"),
                    data.get("tracked_products"),
                    data.get("amount"),
                    data.get("quote_type"),
                    data.get("closing_text"),
                    data["salesperson_name"],
                    data.get("salesperson_title"),
                    data.get("salesperson_contact"),
                    data.get("quotation_remark"),
                    data["follow_up_status"],
                    data.get("follow_up_note"),
                    data.get("follow_up_date"),
                    data.get("payment_status"),
                    data.get("payment_receipt"),
                    data.get("pdf_path"),
                    data["letter_id"],
                ),
            )
        return int(data["letter_id"])

    with get_cursor() as cur:
        cur.execute(
            textwrap.dedent(
                """
                INSERT INTO quotation_letters (
                    salesperson_id,
                    reference_no,
                    quote_date,
                    customer_name,
                    customer_company,
                    customer_address,
                    customer_contact,
                    district_id,
                    attention_name,
                    attention_title,
                    subject_line,
                    salutation,
                    body_intro,
                    product_details,
                    tracked_products,
                    amount,
                    quote_type,
                    closing_text,
                    salesperson_name,
                    salesperson_title,
                    salesperson_contact,
                    quotation_remark,
                    follow_up_status,
                    follow_up_note,
                    follow_up_date,
                    payment_status,
                    payment_receipt,
                    pdf_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            ),
            (
                data["salesperson_id"],
                data["reference_no"],
                    data["quote_date"],
                    data["customer_name"],
                    data["customer_company"],
                    data["customer_address"],
                    data.get("customer_contact"),
                    data.get("district_id"),
                    data.get("attention_name"),
                    data.get("attention_title"),
                    data["subject_line"],
                    data.get("salutation"),
                    data.get("body_intro"),
                    data.get("product_details"),
                    data.get("tracked_products"),
                    data.get("amount"),
                    data.get("quote_type"),
                    data.get("closing_text"),
                    data["salesperson_name"],
                    data.get("salesperson_title"),
                    data.get("salesperson_contact"),
                    data.get("quotation_remark"),
                data["follow_up_status"],
                data.get("follow_up_note"),
                data.get("follow_up_date"),
                data.get("payment_status"),
                data.get("payment_receipt"),
                data.get("pdf_path"),
            ),
        )
        letter_id = cur.lastrowid
    return int(letter_id)


def get_users(role: Optional[str] = None) -> List[sqlite3.Row]:
    with get_conn() as conn:
        if role:
            cur = conn.execute(
                "SELECT * FROM users WHERE role = ? ORDER BY username", (role,)
            )
        else:
            cur = conn.execute("SELECT * FROM users ORDER BY username")
        return cur.fetchall()


def export_data_frames() -> Dict[str, pd.DataFrame]:
    """Collect application data for export into spreadsheets."""

    frames: Dict[str, pd.DataFrame] = {}
    frames["Users"] = fetchall_df(
        textwrap.dedent(
            """
            SELECT user_id, username, role, display_name, designation, phone, created_at
            FROM users
            ORDER BY username
            """
        )
    )
    frames["Quotation letters"] = fetchall_df(
        textwrap.dedent(
            """
            SELECT l.letter_id,
                   l.reference_no,
                   l.quote_date,
                   l.customer_name,
                   l.customer_company,
                   l.customer_address,
                   l.customer_contact,
                   d.name AS district,
                   l.attention_name,
                   l.attention_title,
                   l.subject_line,
                   l.quote_type,
                   l.amount,
                   l.tracked_products,
                   l.follow_up_status,
                   l.follow_up_date,
                   l.payment_status,
                   l.payment_receipt,
                   l.pdf_path,
                   COALESCE(u.display_name, u.username) AS salesperson
            FROM quotation_letters l
            JOIN users u ON u.user_id = l.salesperson_id
            LEFT JOIN districts d ON d.district_id = l.district_id
            ORDER BY l.quote_date DESC, l.letter_id DESC
            """
        )
    )
    frames["Companies"] = list_companies()
    frames["Quotations"] = fetchall_df(
        textwrap.dedent(
            """
            SELECT q.quotation_id, q.quote_date, q.status, q.payment_status, q.quantity,
                   q.quote_type, q.payment_receipt, q.notes,
                   c.name AS company, d.name AS district,
                   COALESCE(prod.names, cat.name || CASE WHEN q.quantity > 1 THEN ' (x' || q.quantity || ')' ELSE '' END) AS product,
                   COALESCE(u.display_name, u.username) AS salesperson
            FROM quotations q
            JOIN companies c ON c.company_id = q.company_id
            JOIN districts d ON d.district_id = q.district_id
            JOIN categories cat ON cat.category_id = q.category_id
            JOIN users u ON u.user_id = q.salesperson_id
            LEFT JOIN ({subquery}) prod ON prod.quotation_id = q.quotation_id
            ORDER BY q.quote_date DESC
            """
        ).format(subquery=PRODUCT_LIST_SUBQUERY)
    )
    frames["Work orders"] = fetchall_df(
        """
        SELECT w.work_order_id, w.upload_date, w.pdf_path, w.notes,
               q.quotation_id, c.name AS company
        FROM work_orders w
        JOIN quotations q ON q.quotation_id = w.quotation_id
        JOIN companies c ON c.company_id = q.company_id
        ORDER BY w.upload_date DESC
        """
    )
    frames["Delivery orders"] = fetchall_df(
        """
        SELECT d.do_id, d.source_type, d.do_number, d.upload_date, d.price,
               d.payment_received, d.payment_date, d.receipt_path, d.notes,
               COALESCE(c.name, d.third_party_name, '—') AS company,
               q.quotation_id
        FROM delivery_orders d
        LEFT JOIN quotations q ON q.quotation_id = COALESCE(d.quotation_id, (
            SELECT quotation_id FROM work_orders WHERE work_order_id = d.work_order_id
        ))
        LEFT JOIN companies c ON c.company_id = q.company_id
        ORDER BY d.upload_date DESC
        """
    )
    frames["Notifications"] = fetchall_df(
        "SELECT notification_id, user_id, message, due_date, read, created_at FROM notifications"
    )
    return frames


def build_excel_export() -> bytes:
    """Create an Excel workbook containing the key application datasets."""

    frames = export_data_frames()
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for sheet_name, frame in frames.items():
                safe_name = sheet_name[:31]
                frame.to_excel(writer, sheet_name=safe_name, index=False)
    except ModuleNotFoundError:
        st.warning("Install 'xlsxwriter' to enable Excel exports.")
        return b""
    buffer.seek(0)
    return buffer.getvalue()


def build_full_archive(excel_bytes: Optional[bytes] = None) -> bytes:
    """Create a compressed archive containing exports and uploaded documents."""

    archive_buffer = io.BytesIO()
    if excel_bytes is None:
        excel_bytes = build_excel_export()
    uploads_root = CONFIG.data_dir / "uploads"

    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if excel_bytes:
            zf.writestr("reports/ps_sales_summary.xlsx", excel_bytes)
        if uploads_root.exists():
            for path in uploads_root.rglob("*"):
                if path.is_file():
                    arcname = Path("uploads") / path.relative_to(uploads_root)
                    zf.write(path, arcname.as_posix())

    archive_buffer.seek(0)
    return archive_buffer.getvalue()


def get_settings() -> Dict[str, int]:
    with get_conn() as conn:
        cur = conn.execute("SELECT key, value FROM settings")
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def update_setting(key: str, value: int) -> None:
    with get_cursor() as cur:
        cur.execute("REPLACE INTO settings(key, value) VALUES (?, ?)", (key, str(value)))


def create_notification(user_id: int, message: str, due_date: date) -> None:
    NOTIFICATION_SCHEDULER.create_notification(user_id, message, due_date)


def mark_notification_read(notification_id: int) -> None:
    with get_cursor() as cur:
        cur.execute(
            "UPDATE notifications SET read=1 WHERE notification_id=?",
            (notification_id,),
        )


def get_user_notifications(user_id: int, include_read: bool = False) -> pd.DataFrame:
    query = "SELECT * FROM notifications WHERE user_id=?"
    params: List = [user_id]
    if not include_read:
        query += " AND read=0"
    query += " ORDER BY due_date"
    return fetchall_df(query, tuple(params))


# ---------------------------------------------------------------------------
# Notification generation logic
# ---------------------------------------------------------------------------


def schedule_follow_up_notifications(quotation_id: int) -> None:
    NOTIFICATION_SCHEDULER.notify_follow_up(quotation_id)


def generate_system_notifications() -> None:
    NOTIFICATION_SCHEDULER.generate_system_notifications()


def notify_admin_activity(message: str, actor: Dict, due_date: Optional[date] = None) -> None:
    """Send an activity notification to all admins for staff actions."""

    if actor.get("role") != "staff":
        return
    admins = get_users("admin")
    if not admins:
        return
    actor_label = default_salesperson_display(actor) or actor.get("username") or "Sales staff"
    final_message = f"{actor_label}: {message}"
    due = due_date or date.today()
    for admin in admins:
        create_notification(admin["user_id"], final_message, due)


def notify_payment_recorded(quotation_id: int, actor: Dict) -> None:
    """Send a notification to all admins when a quotation payment is recorded."""

    notify_admin_activity(
        f"Payment received for quotation #{quotation_id}", actor, due_date=date.today()
    )


def notify_new_quotation(letter_id: int, data: Dict[str, Any], actor: Dict) -> None:
    """Alert all admins when a new quotation letter is created."""

    customer = data.get("customer_company") or data.get("customer_name") or "Customer"
    message = f"New quotation letter #{letter_id} for {customer}"
    notify_admin_activity(message, actor, due_date=date.today())


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------


def authenticate(username: str, password: str) -> Tuple[Optional[Dict], Optional[str]]:
    username = username.strip()
    if not username:
        return None, "Username is required."

    locked = LOCKOUT_SERVICE.is_locked(username)
    if locked:
        return None, LOCKOUT_SERVICE.lockout_message(username)

    user = USER_REPOSITORY.fetch_by_username(username)
    if not user:
        LOCKOUT_SERVICE.record_attempt(username, False)
        return None, "Invalid credentials."

    stored_hash = user.get("pass_hash", "")
    if PASSWORD_SERVICE.verify(password, stored_hash):
        LOCKOUT_SERVICE.record_attempt(username, True)
        if PASSWORD_SERVICE.needs_update(stored_hash):
            USER_REPOSITORY.update_password_hash(
                user["user_id"], PASSWORD_SERVICE.hash(password)
            )
        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "role": user["role"],
            "display_name": user.get("display_name"),
            "designation": user.get("designation"),
            "phone": user.get("phone"),
        }, None

    LOCKOUT_SERVICE.record_attempt(username, False)
    return None, "Invalid credentials."


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def login_screen() -> None:
    st.title("PS Business Suites by ZAD")
    st.caption("Quotation and sales workflow management")
    with st.form("login_form"):
        cols = st.columns(2)
        with cols[0]:
            username = st.text_input("Username", help="Use your assigned account name.")
        with cols[1]:
            password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
    if submitted:
        user, error = authenticate(username, password)
        if user:
            st.session_state["user"] = user
            st.success(f"Welcome back, {user['username']}")
            rerun()
        else:
            st.error(error or "Invalid credentials")


def sidebar(user: Dict) -> str:
    pages_common = {
        "Dashboard": "dashboard",
        "Work Orders": "work_orders",
        "Delivery Orders": "delivery_orders",
        "Notifications": "notifications",
    }
    pages_admin = {
        "Companies": "companies",
        "Advanced Filters": "admin_filters",
        "Users": "users",
        "Settings": "settings",
    }
    if user["role"] == "admin":
        pages = {**pages_common, **pages_admin}
    else:
        pages = pages_common

    st.sidebar.title("Navigation")
    st.session_state.setdefault("active_page", "dashboard")

    labels = list(pages.keys())
    st.session_state.setdefault("navigation_choice", labels[0])

    def _set_page_from_radio() -> None:
        choice = st.session_state.get("navigation_choice", labels[0])
        st.session_state["active_page"] = pages[choice]

    if st.sidebar.button("Create quotation", use_container_width=True):
        st.session_state["active_page"] = "quotation_letters"

    current_page = st.session_state.get("active_page", "dashboard")
    if current_page != "quotation_letters":
        for label, slug in pages.items():
            if slug == current_page:
                st.session_state["navigation_choice"] = label
                break

    st.sidebar.radio(
        "Go to",
        labels,
        key="navigation_choice",
        on_change=_set_page_from_radio,
    )
    st.sidebar.write("---")
    st.sidebar.write(f"Logged in as **{user['username']}** ({user['role']})")
    if st.sidebar.button("Logout"):
        st.session_state.pop("user", None)
        rerun()
    return st.session_state.get("active_page", "dashboard")


def show_pdf_link(relative_path: Optional[str], label: str) -> None:
    if not relative_path:
        st.write("No file uploaded")
        return
    try:
        path = resolve_upload_path(relative_path)
    except ValueError:
        st.error("Stored path is invalid")
        return
    if not path.exists():
        st.warning("File missing on disk")
        return
    meta = UPLOAD_MANAGER.metadata(relative_path)
    cols = st.columns([3, 1])
    with open(path, "rb") as f:
        cols[0].download_button(label=label, data=f.read(), file_name=path.name)
    if meta:
        size_kb = meta["size"] / 1024
        cols[1].metric(
            "File info",
            f"{size_kb:.1f} KB",
            delta=f"Uploaded {meta['uploaded']:%Y-%m-%d %H:%M}",
        )


# ---------------------------------------------------------------------------
# CRUD helper functions for companies, products, districts
# ---------------------------------------------------------------------------


def list_companies() -> pd.DataFrame:
    query = textwrap.dedent(
        """
        SELECT c.company_id, c.name, c.contact_person, c.phone,
               c.address, c.delivery_address, d.name AS district,
               GROUP_CONCAT(cat.name, ', ') AS products
        FROM companies c
        JOIN districts d ON d.district_id = c.district_id
        LEFT JOIN company_categories cc ON cc.company_id = c.company_id
        LEFT JOIN categories cat ON cat.category_id = cc.category_id
        GROUP BY c.company_id
        ORDER BY c.name
        """
    )
    return fetchall_df(query)


def get_company_products(company_id: int) -> List[int]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT category_id FROM company_categories WHERE company_id=?",
            (company_id,),
        ).fetchall()
        return [row["category_id"] for row in rows]


def ensure_product(name: str) -> int:
    """Return the product ID for the given name, creating it if necessary."""

    product_name = name.strip()
    if not product_name:
        raise ValueError("Product name is required")

    with get_conn() as conn:
        row = conn.execute(
            "SELECT category_id FROM categories WHERE lower(name)=lower(?)",
            (product_name.lower(),),
        ).fetchone()
    if row:
        return row["category_id"]

    with get_cursor() as cur:
        cur.execute("INSERT INTO categories(name) VALUES (?)", (product_name,))
        return cur.lastrowid


def parse_product_names(raw: str) -> List[str]:
    """Split a free-form product string into distinct product names."""

    if not raw:
        return []

    seen: set[str] = set()
    names: List[str] = []
    for part in re.split(r"[\n,]+", raw):
        name = part.strip()
        key = name.lower()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    return names


def link_company_product(company_id: int, product_id: int) -> None:
    """Associate a company with a product if not already linked."""

    with get_cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO company_categories(company_id, category_id) VALUES (?, ?)",
            (company_id, product_id),
        )


def get_quotation_products(quotation_id: Optional[int]) -> List[Tuple[int, int]]:
    """Return ordered product rows (category_id, quantity) for a quotation."""

    if not quotation_id:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            textwrap.dedent(
                """
                SELECT category_id, quantity
                FROM quotation_products
                WHERE quotation_id=?
                ORDER BY line_no
                """
            ),
            (quotation_id,),
        ).fetchall()
    return [(row["category_id"], row["quantity"]) for row in rows]


def set_quotation_products(quotation_id: int, items: Sequence[Tuple[int, int]]) -> None:
    """Persist ordered product rows for a quotation."""

    with get_cursor() as cur:
        cur.execute("DELETE FROM quotation_products WHERE quotation_id=?", (quotation_id,))
        cur.executemany(
            "INSERT INTO quotation_products(quotation_id, line_no, category_id, quantity) VALUES (?, ?, ?, ?)",
            (
                (quotation_id, index, category_id, quantity)
                for index, (category_id, quantity) in enumerate(items, start=1)
            ),
        )


def upsert_company(data: Dict, product_ids: List[int]) -> None:
    company_type = data.get("type")
    if data.get("company_id") and company_type is None:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT type FROM companies WHERE company_id=?",
                (data["company_id"],),
            ).fetchone()
        company_type = row["type"] if row else "retail"
    if company_type is None:
        company_type = "retail"
    with get_cursor() as cur:
        if data.get("company_id"):
            cur.execute(
                textwrap.dedent(
                    """
                    UPDATE companies
                    SET name=?, contact_person=?, phone=?, address=?, delivery_address=?, district_id=?, type=?
                    WHERE company_id=?
                    """
                ),
                (
                    data["name"],
                    data.get("contact_person"),
                    data.get("phone"),
                    data.get("address"),
                    data.get("delivery_address"),
                    data["district_id"],
                    company_type,
                    data["company_id"],
                ),
            )
            cur.execute(
                "DELETE FROM company_categories WHERE company_id=?",
                (data["company_id"],),
            )
            company_id = data["company_id"]
        else:
            cur.execute(
                textwrap.dedent(
                    """
                    INSERT INTO companies(
                        name,
                        contact_person,
                        phone,
                        address,
                        delivery_address,
                        district_id,
                        type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                (
                    data["name"],
                    data.get("contact_person"),
                    data.get("phone"),
                    data.get("address"),
                    data.get("delivery_address"),
                    data["district_id"],
                    company_type,
                ),
            )
            company_id = cur.lastrowid
        for product_id in product_ids:
            cur.execute(
                "INSERT OR IGNORE INTO company_categories(company_id, category_id) VALUES (?, ?)",
                (company_id, product_id),
            )


def delete_company(company_id: int) -> None:
    with get_cursor() as cur:
        cur.execute("DELETE FROM companies WHERE company_id=?", (company_id,))


def upsert_product(product_id: Optional[int], name: str) -> None:
    with get_cursor() as cur:
        if product_id:
            cur.execute(
                "UPDATE categories SET name=? WHERE category_id=?",
                (name, product_id),
            )
        else:
            cur.execute("INSERT INTO categories(name) VALUES (?)", (name,))


def delete_product(product_id: int) -> None:
    with get_cursor() as cur:
        cur.execute("DELETE FROM categories WHERE category_id=?", (product_id,))


def upsert_district(district_id: Optional[int], name: str) -> None:
    with get_cursor() as cur:
        if district_id:
            cur.execute(
                "UPDATE districts SET name=? WHERE district_id=?",
                (name, district_id),
            )
        else:
            cur.execute("INSERT INTO districts(name) VALUES (?)", (name,))

# ---------------------------------------------------------------------------
# Quotation, work order and delivery order helpers
# ---------------------------------------------------------------------------


def list_quotations(user: Dict) -> pd.DataFrame:
    params: List = []
    condition = ""
    if user["role"] == "staff":
        condition = "WHERE q.salesperson_id = ?"
        params.append(user["user_id"])
    query = textwrap.dedent(
        f"""
        SELECT q.quotation_id, q.quote_date, q.status, q.follow_up_date, q.quote_type, q.quantity,
               q.payment_status,
               c.name AS company, d.name AS district,
               COALESCE(prod.names, cat.name || CASE WHEN q.quantity > 1 THEN ' (x' || q.quantity || ')' ELSE '' END) AS product,
               COALESCE(u.display_name, u.username) AS salesperson
        FROM quotations q
        JOIN companies c ON c.company_id = q.company_id
        JOIN districts d ON d.district_id = q.district_id
        JOIN categories cat ON cat.category_id = q.category_id
        JOIN users u ON u.user_id = q.salesperson_id
        LEFT JOIN ({PRODUCT_LIST_SUBQUERY}) prod ON prod.quotation_id = q.quotation_id
        {condition}
        ORDER BY q.quote_date DESC
        """
    )
    return fetchall_df(query, tuple(params))


def get_quotation(quotation_id: Optional[int]) -> Optional[sqlite3.Row]:
    if quotation_id is None:
        return None
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM quotations WHERE quotation_id=?", (quotation_id,)
        ).fetchone()


def upsert_quotation(data: Dict) -> int:
    with get_cursor() as cur:
        if data.get("quotation_id"):
            cur.execute(
                textwrap.dedent(
                    """
                    UPDATE quotations
                    SET company_id=?, district_id=?, category_id=?, quote_date=?, status=?,
                        follow_up_date=?, kva=?, pdf_path=?, notes=?, quote_type=?, quantity=?,
                        payment_status=?, payment_receipt=?
                    WHERE quotation_id=?
                    """
                ),
                (
                    data["company_id"],
                    data["district_id"],
                    data["category_id"],
                    data["quote_date"],
                    data["status"],
                    data.get("follow_up_date"),
                    data.get("kva"),
                    data.get("pdf_path"),
                    data.get("notes"),
                    data["quote_type"],
                    data.get("quantity", 1),
                    data.get("payment_status", "pending"),
                    data.get("payment_receipt"),
                    data["quotation_id"],
                ),
            )
            quotation_id = data["quotation_id"]
        else:
            cur.execute(
                textwrap.dedent(
                    """
                    INSERT INTO quotations(salesperson_id, company_id, district_id, category_id,
                        quote_date, status, follow_up_date, kva, pdf_path, notes, quote_type, quantity,
                        payment_status, payment_receipt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                (
                    data["salesperson_id"],
                    data["company_id"],
                    data["district_id"],
                    data["category_id"],
                    data["quote_date"],
                    data["status"],
                    data.get("follow_up_date"),
                    data.get("kva"),
                    data.get("pdf_path"),
                    data.get("notes"),
                    data["quote_type"],
                    data.get("quantity", 1),
                    data.get("payment_status", "pending"),
                    data.get("payment_receipt"),
                ),
            )
            quotation_id = cur.lastrowid
        return quotation_id


def set_quotation_payment_status(
    quotation_id: int, status: str, receipt_path: Optional[str]
) -> None:
    with get_cursor() as cur:
        cur.execute(
            "UPDATE quotations SET payment_status=?, payment_receipt=? WHERE quotation_id=?",
            (status, receipt_path, quotation_id),
        )


def decline_quotations(quotation_ids: Sequence[int]) -> None:
    """Bulk mark quotations as declined."""

    ids = [int(qid) for qid in quotation_ids if qid is not None]
    if not ids:
        return
    with get_cursor() as cur:
        cur.executemany(
            "UPDATE quotations SET status='declined', follow_up_date=NULL WHERE quotation_id=?",
            ((qid,) for qid in ids),
        )


def upsert_work_order(data: Dict) -> int:
    with get_cursor() as cur:
        if data.get("work_order_id"):
            cur.execute(
                """
                UPDATE work_orders SET quotation_id=?, upload_date=?, pdf_path=?, notes=?
                WHERE work_order_id=?
                """,
                (
                    data["quotation_id"],
                    data["upload_date"],
                    data.get("pdf_path"),
                    data.get("notes"),
                    data["work_order_id"],
                ),
            )
            return data["work_order_id"]
        cur.execute(
            """
            INSERT INTO work_orders(quotation_id, upload_date, pdf_path, notes)
            VALUES (?, ?, ?, ?)
            """,
            (
                data["quotation_id"],
                data["upload_date"],
                data.get("pdf_path"),
                data.get("notes"),
            ),
        )
        return cur.lastrowid


def upsert_delivery_order(data: Dict) -> int:
    with get_cursor() as cur:
        if data.get("do_id"):
            cur.execute(
                textwrap.dedent(
                    """
                    UPDATE delivery_orders
                    SET source_type=?, salesperson_id=?, work_order_id=?, quotation_id=?, third_party_name=?,
                        do_number=?, upload_date=?, pdf_path=?, price=?,
                        payment_received=?, payment_date=?, notes=?, receipt_path=?
                    WHERE do_id=?
                    """
                ),
                (
                    data["source_type"],
                    data["salesperson_id"],
                    data.get("work_order_id"),
                    data.get("quotation_id"),
                    data.get("third_party_name"),
                    data["do_number"],
                    data["upload_date"],
                    data.get("pdf_path"),
                    data.get("price", 0.0),
                    1 if data.get("payment_received") else 0,
                    data.get("payment_date"),
                    data.get("notes"),
                    data.get("receipt_path"),
                    data["do_id"],
                ),
            )
            return data["do_id"]
        cur.execute(
            textwrap.dedent(
                """
                INSERT INTO delivery_orders(source_type, salesperson_id, work_order_id, quotation_id, third_party_name,
                    do_number, upload_date, pdf_path, price, payment_received, payment_date, notes, receipt_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            ),
            (
                data["source_type"],
                data["salesperson_id"],
                data.get("work_order_id"),
                data.get("quotation_id"),
                data.get("third_party_name"),
                data["do_number"],
                data["upload_date"],
                data.get("pdf_path"),
                data.get("price", 0.0),
                1 if data.get("payment_received") else 0,
                data.get("payment_date"),
                data.get("notes"),
                data.get("receipt_path"),
            ),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------


def quotation_metrics(user: Dict) -> Dict[str, int]:
    where = ""
    params: List = []
    if user["role"] == "staff":
        where = "WHERE salesperson_id=?"
        params.append(user["user_id"])
    with get_conn() as conn:
        counts: Dict[str, int] = {}
        for status in ["pending", "accepted", "declined", "inform_later"]:
            if where:
                query = f"SELECT COUNT(*) FROM quotations {where} AND status=?"
                row = conn.execute(query, tuple(params + [status])).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM quotations WHERE status=?", (status,)
                ).fetchone()
            counts[status] = row[0]
        if where:
            row = conn.execute(
                f"SELECT COUNT(*) FROM quotations {where}", tuple(params)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM quotations").fetchone()
        counts["total"] = row[0]
    return counts


def quotation_trends(user: Dict, period: str = "M") -> pd.DataFrame:
    where = ""
    params: List = []
    if user["role"] == "staff":
        where = "WHERE salesperson_id=?"
        params.append(user["user_id"])
    query = textwrap.dedent(f"SELECT quote_date, status FROM quotations {where}")
    df = fetchall_df(query, tuple(params))
    if df.empty:
        return df
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["period"] = df["quote_date"].dt.to_period(period).dt.to_timestamp()
    trend = df.groupby(["period", "status"]).size().unstack(fill_value=0)
    return trend


def revenue_summary(user: Dict) -> pd.DataFrame:
    where = ""
    params: List = []
    if user["role"] == "staff":
        where = "WHERE COALESCE(q.salesperson_id, d.salesperson_id)=?"
        params.append(user["user_id"])
    query = textwrap.dedent(
        f"""
        SELECT
            d.do_id,
            d.price,
            d.payment_received,
            d.payment_date,
            d.upload_date,
            d.source_type,
            d.third_party_name,
            COALESCE(q.salesperson_id, d.salesperson_id) AS salesperson_id,
            COALESCE(u.display_name, u.username) AS salesperson,
            COALESCE(comp.name, d.third_party_name, 'Unassigned') AS company,
            dist.name AS district
        FROM delivery_orders d
        LEFT JOIN work_orders w ON w.work_order_id = d.work_order_id
        LEFT JOIN quotations q ON q.quotation_id = COALESCE(d.quotation_id, w.quotation_id)
        LEFT JOIN companies comp ON comp.company_id = q.company_id
        LEFT JOIN districts dist ON dist.district_id = q.district_id
        LEFT JOIN users u ON u.user_id = COALESCE(q.salesperson_id, d.salesperson_id)
        {where}
        """
    )
    df = fetchall_df(query, tuple(params))
    if df.empty:
        return df
    df["company"] = df["company"].fillna("Unassigned")
    return df


def follow_up_overview(user: Dict) -> pd.DataFrame:
    where = ""
    params: List = []
    if user["role"] == "staff":
        where = "WHERE q.salesperson_id=?"
        params.append(user["user_id"])
    query = textwrap.dedent(
        f"""
        SELECT q.quotation_id, q.status, q.follow_up_date, q.quote_date,
               c.name AS company, COALESCE(u.display_name, u.username) AS salesperson
        FROM quotations q
        JOIN companies c ON c.company_id = q.company_id
        JOIN users u ON u.user_id = q.salesperson_id
        {where}
        """
    )
    df = fetchall_df(query, tuple(params))
    if df.empty:
        return df
    settings = get_settings()
    pending_days = settings.get("quotation_pending_days", 10)
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["follow_up_date"] = pd.to_datetime(df["follow_up_date"], errors="coerce")
    df["due_date"] = df.apply(
        lambda row: row["follow_up_date"]
        if pd.notna(row["follow_up_date"])
        else row["quote_date"] + pd.Timedelta(days=pending_days),
        axis=1,
    )
    df["overdue"] = df["due_date"].dt.date < date.today()
    warning_window = date.today() + timedelta(days=CONFIG.pre_due_warning_days)
    df["upcoming"] = df["due_date"].dt.date <= warning_window
    return df


def outstanding_payments(user: Dict) -> pd.DataFrame:
    revenue = revenue_summary(user)
    if revenue.empty:
        return revenue
    outstanding = revenue[revenue["payment_received"] == 0]
    if outstanding.empty:
        return outstanding
    outstanding = outstanding.copy()
    outstanding["district"] = outstanding["district"].fillna("—")
    outstanding["salesperson"] = outstanding["salesperson"].fillna("Unassigned")
    return (
        outstanding.groupby(["salesperson", "district"])["price"]
        .sum()
        .reset_index()
    )


def quotation_status_breakdown() -> Dict[str, Dict[str, int]]:
    df = fetchall_df("SELECT quote_date, status FROM quotations")
    if df.empty:
        return {}

    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce").dt.date
    today = date.today()
    week_start = today - timedelta(days=6)
    month_start = today.replace(day=1)

    periods = {
        "today": df["quote_date"] == today,
        "week": df["quote_date"].ge(week_start),
        "month": df["quote_date"].ge(month_start),
    }

    statuses = ["accepted", "declined", "pending", "inform_later"]
    breakdown: Dict[str, Dict[str, int]] = {}
    for label, mask in periods.items():
        subset = df[mask]
        counts = {status: int((subset["status"] == status).sum()) for status in statuses}
        counts["total"] = int(subset.shape[0])
        breakdown[label] = counts
    return breakdown


def admin_salesperson_overview() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return monitoring datasets for the admin dashboard."""

    df = load_admin_dataset()
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    dataset = df.copy()
    dataset["quote_date"] = pd.to_datetime(dataset["quote_date"], errors="coerce")
    dataset["follow_up_date"] = pd.to_datetime(dataset["follow_up_date"], errors="coerce")
    dataset["price"] = pd.to_numeric(dataset["price"], errors="coerce").fillna(0.0)
    dataset["active_quote"] = dataset["status"].isin(["pending", "inform_later"])
    dataset["has_work_order"] = dataset["work_order_id"].notna()
    dataset["has_delivery_order"] = dataset["delivery_order_id"].notna()
    payment_flags = dataset["payment_received"].fillna(0).astype(int)
    dataset["outstanding_value"] = dataset["price"].where(
        (dataset["has_delivery_order"]) & (payment_flags == 0), 0.0
    )

    today = pd.Timestamp.today().normalize()
    recent_threshold = today - pd.Timedelta(days=6)
    dataset["new_quote"] = dataset["quote_date"].ge(recent_threshold)

    quote_level = (
        dataset.groupby(["quotation_id", "salesperson"], dropna=False)
        .agg(
            company=("company", "first"),
            status=("status", "first"),
            quote_date=("quote_date", "first"),
            follow_up_date=("follow_up_date", "first"),
            active_quote=("active_quote", "max"),
            new_quote=("new_quote", "max"),
            has_work_order=("has_work_order", "max"),
            has_delivery=("has_delivery_order", "max"),
            payments_received=("payment_received", "max"),
        )
        .reset_index()
    )
    quote_level["has_work_order"] = quote_level["has_work_order"].astype(int)
    quote_level["has_delivery"] = quote_level["has_delivery"].astype(int)
    quote_level["payments_received"] = (
        quote_level["payments_received"].fillna(0).astype(int)
    )

    summary = quote_level.groupby("salesperson").agg(
        total_quotes=("quotation_id", "count"),
        companies_engaged=("company", "nunique"),
        new_this_week=("new_quote", "sum"),
        active_quotes=("active_quote", "sum"),
        with_work_order=("has_work_order", "sum"),
        delivery_orders=("has_delivery", "sum"),
        payments_received=("payments_received", "sum"),
    )

    status_counts = (
        quote_level.groupby(["salesperson", "status"]).size().unstack(fill_value=0)
    )
    for status_name in ["pending", "accepted", "declined", "inform_later"]:
        if status_name in status_counts.columns:
            summary[status_name] = status_counts[status_name]
        else:
            summary[status_name] = 0

    outstanding_totals = (
        dataset.groupby("salesperson")["outstanding_value"].sum().rename("outstanding_value")
    )
    summary = summary.join(outstanding_totals, how="left").fillna({"outstanding_value": 0.0})
    summary = summary.reset_index().rename(columns={
        "salesperson": "Salesperson",
        "total_quotes": "Total quotations",
        "companies_engaged": "Companies engaged",
        "new_this_week": "New (7d)",
        "active_quotes": "Active pursuits",
        "with_work_order": "Work orders",
        "delivery_orders": "Delivery orders",
        "pending": "Pending",
        "accepted": "Accepted",
        "declined": "Declined",
        "inform_later": "Inform later",
        "outstanding_value": "Outstanding value",
        "payments_received": "Paid quotations",
    })

    summary["Outstanding value"] = summary["Outstanding value"].round(2)

    active_details = quote_level[quote_level["active_quote"] > 0].copy()
    active_details["quote_date"] = active_details["quote_date"].dt.date
    active_details["follow_up_date"] = active_details["follow_up_date"].dt.date
    active_details = active_details.rename(
        columns={
            "salesperson": "Salesperson",
            "quotation_id": "Quotation #",
            "quote_date": "Quoted on",
            "follow_up_date": "Next follow-up",
            "status": "Status",
            "company": "Company",
        }
    )

    latest_quotes = quote_level.sort_values(
        "quote_date", ascending=False, na_position="last"
    ).head(20)
    latest_quotes["quote_date"] = latest_quotes["quote_date"].dt.date
    latest_quotes["follow_up_date"] = latest_quotes["follow_up_date"].dt.date
    latest_quotes = latest_quotes.rename(
        columns={
            "salesperson": "Salesperson",
            "quotation_id": "Quotation #",
            "quote_date": "Quoted on",
            "follow_up_date": "Next follow-up",
            "status": "Status",
            "company": "Company",
        }
    )

    return summary, active_details, latest_quotes

# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


def render_dashboard(user: Dict) -> None:
    generate_system_notifications()
    notifications_df = get_user_notifications(user["user_id"], include_read=False)
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.header("Revenue & Follow-up Dashboard")
        st.markdown(
            "<div style='text-align:right;font-size:0.75rem;opacity:0.7;'>by ZAD</div>",
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        if st.button("Create quotation", type="primary", use_container_width=True):
            st.session_state["active_page"] = "quotation_letters"
            rerun()
    action_cols = st.columns(4)
    if action_cols[0].button("Create quotation", use_container_width=True):
        st.session_state["active_page"] = "quotation_letters"
        rerun()
    if action_cols[1].button("Work orders", use_container_width=True):
        st.session_state["active_page"] = "work_orders"
        rerun()
    if action_cols[2].button("Delivery orders", use_container_width=True):
        st.session_state["active_page"] = "delivery_orders"
        rerun()
    advanced_label = "Advanced filters" if user["role"] == "admin" else "Advanced filters (admin only)"
    if action_cols[3].button(advanced_label, use_container_width=True, disabled=user["role"] != "admin"):
        st.session_state["active_page"] = "admin_filters"
        rerun()
    if user["role"] == "admin" and not notifications_df.empty:
        seen_ids = set(st.session_state.get("_seen_notification_ids", []))
        new_rows = notifications_df[
            (~notifications_df["notification_id"].isin(seen_ids))
            & (notifications_df["message"].str.startswith("New quotation #"))
        ]
        for row in new_rows.itertuples():
            st.toast(row.message)
            seen_ids.add(row.notification_id)
        st.session_state["_seen_notification_ids"] = list(seen_ids)
    if user["role"] == "admin":
        export_cols = st.columns(2)
        excel_data = build_excel_export()
        archive_data = build_full_archive(excel_data)
        with export_cols[0]:
            st.download_button(
                "Download Excel summary",
                data=excel_data,
                file_name="ps_sales_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=not excel_data,
                help=None if excel_data else "Install 'xlsxwriter' to enable Excel exports.",
            )
        with export_cols[1]:
            st.download_button(
                "Download full archive (.rar)",
                data=archive_data,
                file_name="ps_sales_full_export.rar",
                mime="application/octet-stream",
            )
    counts = quotation_metrics(user)
    followups = follow_up_overview(user)
    overdue = followups[followups["overdue"]] if not followups.empty else pd.DataFrame()
    upcoming = followups[(~followups["overdue"]) & (followups["upcoming"])] if not followups.empty else pd.DataFrame()
    outstanding_df = outstanding_payments(user)
    outstanding_total = float(outstanding_df["price"].sum()) if not outstanding_df.empty else 0.0
    conversion_rate = (
        (counts.get("accepted", 0) / counts.get("total", 1)) * 100 if counts.get("total", 0) else 0.0
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Quotations", counts.get("total", 0))
    metric_cols[1].metric(
        "Conversion rate",
        f"{conversion_rate:.1f}%",
        delta=f"{counts.get('accepted', 0)} accepted",
    )
    metric_cols[2].metric("Overdue follow-ups", len(overdue))
    metric_cols[3].metric("Outstanding payments", f"${outstanding_total:,.2f}")

    if counts.get("total", 0) == 0:
        st.info(
            "No quotations recorded yet. Use the Create quotation button above to get started."
        )

    letters_df = list_quotation_letters(user)
    st.subheader("Quotation letter follow-ups")
    if letters_df.empty:
        st.info("No quotation letters recorded yet.")
    else:
        letters_summary = letters_df.copy()
        letters_summary["quote_date"] = pd.to_datetime(
            letters_summary["quote_date"], errors="coerce"
        ).dt.date
        letters_summary["follow_up_date"] = pd.to_datetime(
            letters_summary.get("follow_up_date"), errors="coerce"
        ).dt.date
        letters_summary["receipt_flag"] = letters_summary.get("payment_receipt").notna()
        status_counts = (
            letters_summary["follow_up_status"]
            .value_counts()
            .reindex(LETTER_FOLLOW_UP_STATUSES, fill_value=0)
        )
        metric_titles = {
            "paid": "Paid (Accepted)",
            "possible": "Possible",
            "rejected": "Rejected",
        }
        letter_metrics = st.columns(len(LETTER_FOLLOW_UP_STATUSES))
        for idx, status in enumerate(LETTER_FOLLOW_UP_STATUSES):
            letter_metrics[idx].metric(
                metric_titles.get(status, LETTER_FOLLOW_UP_LABELS.get(status, status.title())),
                int(status_counts.get(status, 0)),
            )

        tab_labels = [metric_titles.get(status, LETTER_FOLLOW_UP_LABELS.get(status, status.title())) for status in LETTER_FOLLOW_UP_STATUSES]
        status_tabs = st.tabs(tab_labels)
        display_columns = [
            ("letter_id", "Letter #"),
            ("reference_no", "Reference"),
            ("quote_date", "Date"),
            ("customer_company", "Company"),
            ("follow_up_date", "Follow-up date"),
            ("receipt_flag", "Receipt"),
            ("quotation_remark", "Remarks"),
        ]
        if user["role"] == "admin" and "salesperson" in letters_summary.columns:
            display_columns.append(("salesperson", "Salesperson"))

        for tab, status in zip(status_tabs, LETTER_FOLLOW_UP_STATUSES):
            with tab:
                subset = letters_summary[
                    letters_summary["follow_up_status"] == status
                ].copy()
                if subset.empty:
                    st.info(f"No quotations marked as {metric_titles.get(status, status)} yet.")
                else:
                    subset["receipt_flag"] = subset["receipt_flag"].map({True: "Yes", False: "No"})
                    subset["follow_up_date"] = subset["follow_up_date"].apply(
                        lambda value: value if pd.notna(value) else "—"
                    )
                    subset["quotation_remark"] = subset["quotation_remark"].fillna("—")
                    column_keys = [col for col, _ in display_columns if col in subset.columns]
                    display = subset[column_keys].rename(
                        columns={source: label for source, label in display_columns}
                    )
                    st.dataframe(display, use_container_width=True)

    admin_collection_stats: Dict[str, float] = {}
    admin_quote_breakdown: Dict[str, Dict[str, int]] = {}
    if user["role"] == "admin":
        revenue_snapshot = revenue_summary(user)
        if not revenue_snapshot.empty:
            revenue_snapshot["payment_date"] = pd.to_datetime(
                revenue_snapshot["payment_date"], errors="coerce"
            )
            revenue_snapshot["upload_date"] = pd.to_datetime(
                revenue_snapshot["upload_date"], errors="coerce"
            )
            today = pd.Timestamp.today().normalize()
            week_start = today - pd.Timedelta(days=6)
            month_start = today.replace(day=1)

            paid = revenue_snapshot[revenue_snapshot["payment_received"] == 1]
            outstanding = revenue_snapshot[revenue_snapshot["payment_received"] == 0]

            admin_collection_stats = {
                "collected_today": float(
                    paid[paid["payment_date"].dt.date == today.date()]["price"].sum()
                ),
                "collected_week": float(
                    paid[paid["payment_date"] >= week_start]["price"].sum()
                ),
                "collected_month": float(
                    paid[paid["payment_date"] >= month_start]["price"].sum()
                ),
                "pending_today": float(
                    outstanding[
                        outstanding["upload_date"].dt.date == today.date()
                    ]["price"].sum()
                ),
                "pending_week": float(
                    outstanding[outstanding["upload_date"] >= week_start]["price"].sum()
                ),
                "pending_month": float(
                    outstanding[outstanding["upload_date"] >= month_start]["price"].sum()
                ),
            }
        admin_quote_breakdown = quotation_status_breakdown()

    if user["role"] == "admin":
        if admin_collection_stats:
            st.subheader("Collections snapshot")
            collected_cols = st.columns(3)
            collected_cols[0].metric(
                "Collected today",
                f"${admin_collection_stats['collected_today']:,.2f}",
            )
            collected_cols[1].metric(
                "Collected (7d)",
                f"${admin_collection_stats['collected_week']:,.2f}",
            )
            collected_cols[2].metric(
                "Collected (month)",
                f"${admin_collection_stats['collected_month']:,.2f}",
            )

            pending_cols = st.columns(3)
            pending_cols[0].metric(
                "Pending today",
                f"${admin_collection_stats['pending_today']:,.2f}",
            )
            pending_cols[1].metric(
                "Pending (7d)",
                f"${admin_collection_stats['pending_week']:,.2f}",
            )
            pending_cols[2].metric(
                "Pending (month)",
                f"${admin_collection_stats['pending_month']:,.2f}",
            )

        if admin_quote_breakdown:
            st.subheader("Quotation outcomes")
            quote_df = pd.DataFrame(admin_quote_breakdown).T
            column_order = ["total", "accepted", "declined", "pending", "inform_later"]
            for column in column_order:
                if column not in quote_df:
                    quote_df[column] = 0
            quote_df = quote_df[column_order]
            quote_df.index = quote_df.index.str.capitalize()
            st.dataframe(quote_df, use_container_width=True)

    if user["role"] == "admin":
        team_tab, trend_tab, follow_tab, revenue_tab, notifications_tab = st.tabs(
            ["Team monitor", "Trends", "Follow-ups", "Revenue", "Notifications"]
        )
    else:
        trend_tab, follow_tab, revenue_tab, notifications_tab = st.tabs(
            ["Trends", "Follow-ups", "Revenue", "Notifications"]
        )
        team_tab = None

    if team_tab is not None:
        with team_tab:
            summary, active_details, latest_quotes = admin_salesperson_overview()
            if summary.empty:
                st.info("No quotations recorded yet.")
            else:
                summary_display = summary.copy()
                count_columns = [
                    "Total quotations",
                    "Companies engaged",
                    "New (7d)",
                    "Active pursuits",
                    "Work orders",
                    "Delivery orders",
                    "Paid quotations",
                    "Pending",
                    "Accepted",
                    "Declined",
                    "Inform later",
                ]
                for column in count_columns:
                    if column in summary_display:
                        summary_display[column] = summary_display[column].fillna(0).astype(int)

                team_metrics = st.columns(4)
                team_metrics[0].metric("Salespeople", summary_display.shape[0])
                team_metrics[1].metric(
                    "Active pursuits",
                    int(summary_display["Active pursuits"].sum()),
                )
                team_metrics[2].metric(
                    "New quotations (7d)",
                    int(summary_display["New (7d)"].sum()),
                )
                team_metrics[3].metric(
                    "Outstanding value",
                    f"${summary_display['Outstanding value'].sum():,.2f}",
                )

                st.subheader("Salesperson overview")
                st.dataframe(summary_display, use_container_width=True)

                st.subheader("Active quotations to chase")
                if active_details.empty:
                    st.info("No active quotations at the moment.")
                else:
                    active_display = active_details[[
                        "Salesperson",
                        "Quotation #",
                        "Company",
                        "Status",
                        "Quoted on",
                        "Next follow-up",
                    ]].copy()
                    st.dataframe(
                        active_display.sort_values(
                            ["Next follow-up", "Quoted on"], ascending=[True, False]
                        ),
                        use_container_width=True,
                    )

                st.subheader("Latest quotations")
                if latest_quotes.empty:
                    st.info("No quotations found.")
                else:
                    latest_display = latest_quotes[[
                        "Salesperson",
                        "Quotation #",
                        "Company",
                        "Status",
                        "Quoted on",
                        "Next follow-up",
                    ]].copy()
                    st.dataframe(latest_display, use_container_width=True)

    with trend_tab:
        trend = quotation_trends(user)
        if not trend.empty:
            st.write("Monthly quotation volume by status")
            st.line_chart(trend)
            st.dataframe(trend.tail(12), use_container_width=True)
        else:
            st.info("No quotations available for trend analysis yet.")

    with follow_tab:
        if followups.empty:
            st.info("No quotations require follow-up at the moment.")
        else:
            col1, col2 = st.columns(2)
            col1.metric("Due soon", len(upcoming))
            col2.metric("Overdue", len(overdue))
            if not overdue.empty:
                st.subheader("Overdue follow-ups")
                st.dataframe(
                    overdue[["quotation_id", "company", "salesperson", "due_date", "status"]]
                    .rename(columns={"due_date": "Due", "salesperson": "Owner"})
                , use_container_width=True)
            if not upcoming.empty:
                st.subheader("Upcoming follow-ups")
                st.dataframe(
                    upcoming[["quotation_id", "company", "salesperson", "due_date", "status"]]
                    .rename(columns={"due_date": "Due", "salesperson": "Owner"})
                , use_container_width=True)

    with revenue_tab:
        revenue = revenue_summary(user)
        if revenue.empty:
            st.info("No delivery orders recorded yet.")
        else:
            revenue["status"] = revenue["payment_received"].map({1: "Received", 0: "Outstanding"})
            st.bar_chart(revenue.groupby("status")["price"].sum())
            if not outstanding_df.empty:
                st.subheader("Outstanding payments by owner & district")
                st.dataframe(
                    outstanding_df.rename(
                        columns={"salesperson": "Salesperson", "price": "Amount", "district": "District"}
                    )
                , use_container_width=True)
            st.subheader("Delivery order revenue")
            st.dataframe(
                revenue[["company", "district", "price", "status", "salesperson"]]
                .rename(columns={"salesperson": "Salesperson"})
            , use_container_width=True)

    with notifications_tab:
        if notifications_df.empty:
            st.write("No unread notifications")
        else:
            st.dataframe(
                notifications_df[["message", "due_date", "created_at"]]
                .rename(columns={"message": "Message", "due_date": "Due"})
            , use_container_width=True)
 

def render_quotation_letter_page(user: Dict) -> None:
    """Streamlit page for composing quotation letters."""

    st.header("Create quotation")
    st.caption("Compose, save and track quotations from a single workspace.")

    flash = st.session_state.pop("_letter_template_flash", None)
    if flash:
        level, message = flash
        if level == "success":
            st.success(message)
        elif level == "info":
            st.info(message)
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.write(message)

    letters_df = list_quotation_letters(user)
    options: List[Tuple[str, Optional[int]]] = [("New quotation", None)]
    if not letters_df.empty:
        for row in letters_df.itertuples():
            company = getattr(row, "customer_company", "—") or "—"
            quote_date = getattr(row, "quote_date", "") or ""
            label = f"#{int(row.letter_id)} – {company}"
            if quote_date:
                label = f"{label} ({quote_date})"
            salesperson = getattr(row, "salesperson", None)
            if salesperson and user["role"] == "admin":
                label = f"{label} – {salesperson}"
            options.append((label, int(row.letter_id)))

    option_labels = [label for label, _ in options]
    default_index = 0
    active_letter_id = st.session_state.get("letter_form_active_id")
    if active_letter_id is not None:
        for idx, (_, value) in enumerate(options):
            if value == active_letter_id:
                default_index = idx
                break

    selection_label = st.selectbox(
        "Select quotation letter",
        option_labels,
        index=default_index,
        key="quotation_letter_selector",
    )
    selection_map = dict(options)
    selected_id = selection_map.get(selection_label)
    existing = get_quotation_letter(selected_id)
    ensure_letter_form_state(user, existing)
    ensure_letter_aux_state()

    existing_receipt_path = existing["payment_receipt"] if existing else None
    existing_pdf_path = existing["pdf_path"] if existing else None

    form_col, preview_col = st.columns([1, 1.1], gap="large")

    receipt_upload = None
    with form_col:
        st.subheader("Letter details")
        with st.expander("Productivity assists", expanded=False):
            st.markdown(
                "Select a smart template or restore the default layout to speed up drafting."
            )
            template_labels = [template["label"] for template in LETTER_TEMPLATE_LIBRARY]
            selected_template_label = st.selectbox(
                "Quotation template",
                template_labels,
                key=letter_form_aux_key("template_selector"),
            )
            template_key = LETTER_TEMPLATE_KEY_BY_LABEL[selected_template_label]
            template_details = LETTER_TEMPLATE_INDEX[template_key]
            st.caption(template_details.get("description", ""))
            context = build_letter_template_context(user)
            suggested_follow_up = template_details.get("suggested_follow_up")
            if suggested_follow_up and suggested_follow_up in FOLLOW_UP_SUGGESTIONS:
                st.caption(
                    f"Suggested follow-up reminder: {suggested_follow_up}."
                )
            with st.expander("Preview with current details", expanded=False):
                fields = template_details.get("fields", {})
                for field_name, field_label in LETTER_TEMPLATE_PREVIEW_FIELDS:
                    value = fields.get(field_name)
                    if isinstance(value, str):
                        preview_text = value.format_map(context).strip()
                        if preview_text:
                            st.markdown(f"**{field_label}**\n\n{preview_text}")
            helper_cols = st.columns(2)
            with helper_cols[0]:
                if st.button(
                    "Apply template",
                    key="apply_letter_template_button",
                    use_container_width=True,
                ):
                    apply_letter_template(template_key, user)
                    st.session_state["_letter_template_flash"] = (
                        "success",
                        f"Applied template: {template_details['label']}. Review the details before saving.",
                    )
                    rerun()
            with helper_cols[1]:
                if st.button(
                    "Reset to defaults",
                    key="reset_letter_template_button",
                    use_container_width=True,
                ):
                    reset_letter_form_state(user)
                    st.session_state["_letter_template_flash"] = (
                        "info",
                        "Restored the default quotation letter values.",
                    )
                    rerun()
            st.caption(
                "Templates pre-fill the subject, body, remarks and follow-up notes. You can continue editing all fields."
            )
        with st.expander("Smart autofill from companies", expanded=False):
            companies_df = list_companies()
            if companies_df.empty:
                st.info("Add company records to enable instant autofill.")
            else:
                search_value = st.text_input(
                    "Search saved companies",
                    key=letter_form_aux_key("company_search"),
                    placeholder="Type company, contact or phone",
                ).strip()
                filtered_companies = companies_df
                if search_value:
                    filtered_companies = companies_df[
                        companies_df["name"].str.contains(search_value, case=False, na=False)
                        | companies_df["contact_person"].fillna("").str.contains(
                            search_value, case=False, na=False
                        )
                        | companies_df["phone"].fillna("").str.contains(
                            search_value, case=False, na=False
                        )
                    ]
                if filtered_companies.empty:
                    st.warning("No companies match the search yet.")
                else:
                    options = [-1] + filtered_companies["company_id"].astype(int).tolist()
                    display_map: Dict[int, str] = {-1: "Select company"}
                    for _, row in filtered_companies.iterrows():
                        company_id = int(row["company_id"])
                        contact = str(row.get("contact_person") or "").strip()
                        phone = str(row.get("phone") or "").strip()
                        district = str(row.get("district") or "").strip()
                        meta_parts = [part for part in [contact, phone, district] if part]
                        label = row["name"]
                        if meta_parts:
                            label = f"{label} – {' • '.join(meta_parts)}"
                        display_map[company_id] = label
                    selected_company_id = st.selectbox(
                        "Use stored company profile",
                        options,
                        key=letter_form_aux_key("company_selector"),
                        format_func=lambda cid: display_map.get(cid, display_map[-1]),
                    )
                    if selected_company_id != -1:
                        company_id_int = int(selected_company_id)
                        selected_details = companies_df[
                            companies_df["company_id"] == company_id_int
                        ]
                        if not selected_details.empty:
                            detail_row = selected_details.iloc[0]
                            detail_lines: List[str] = []
                            contact_person = detail_row.get("contact_person")
                            if pd.notna(contact_person) and str(contact_person).strip():
                                detail_lines.append(f"**Contact:** {contact_person}")
                            phone_value = detail_row.get("phone")
                            if pd.notna(phone_value) and str(phone_value).strip():
                                detail_lines.append(f"**Phone:** {phone_value}")
                            address_value = detail_row.get("address")
                            if pd.notna(address_value) and str(address_value).strip():
                                detail_lines.append(f"**Address:** {address_value}")
                            delivery_value = detail_row.get("delivery_address")
                            if pd.notna(delivery_value) and str(delivery_value).strip():
                                detail_lines.append(f"**Delivery:** {delivery_value}")
                            district_value = detail_row.get("district")
                            if pd.notna(district_value) and str(district_value).strip():
                                detail_lines.append(f"**District:** {district_value}")
                            products_value = detail_row.get("products")
                            if pd.notna(products_value) and str(products_value).strip():
                                detail_lines.append(f"**Products:** {products_value}")
                            if detail_lines:
                                st.markdown("\n\n".join(detail_lines))
                            if st.button(
                                "Apply company details",
                                key=f"apply_company_{company_id_int}",
                                use_container_width=True,
                            ):
                                company_profile = fetch_company_autofill(company_id_int)
                                if company_profile:
                                    apply_company_autofill(company_profile)
                                    company_name = dict(company_profile).get(
                                        "name", "Selected company"
                                    )
                                    st.session_state["_letter_template_flash"] = (
                                        "success",
                                        f"Loaded {company_name} details into the letter.",
                                    )
                                else:
                                    st.session_state["_letter_template_flash"] = (
                                        "error",
                                        "Unable to load the selected company details.",
                                    )
                                rerun()
                        st.caption(
                            "Autofill updates the recipient, address and follow-up hints when blank."
                        )
        st.text_input(
            "Reference number",
            key=letter_form_key("reference_no"),
            help="Official reference number for this quotation.",
        )
        st.date_input(
            "Date",
            key=letter_form_key("quote_date"),
            help="Quotation issue date.",
        )
        st.text_input(
            "Customer contact name",
            key=letter_form_key("customer_name"),
            help="Primary recipient name (e.g. Managing Director).",
        )
        st.text_input(
            "Customer company",
            key=letter_form_key("customer_company"),
            help="Company or organisation name.",
        )
        st.text_area(
            "Customer address",
            key=letter_form_key("customer_address"),
            help="Mailing address shown beneath the recipient.",
        )
        with get_conn() as conn:
            district_rows = conn.execute(
                "SELECT district_id, name FROM districts ORDER BY name"
            ).fetchall()
        if not district_rows:
            ensure_default_district()
            with get_conn() as conn:
                district_rows = conn.execute(
                    "SELECT district_id, name FROM districts ORDER BY name"
                ).fetchall()
        district_labels = {row["district_id"]: row["name"] for row in district_rows}
        district_options = [row["district_id"] for row in district_rows]
        st.selectbox(
            "Customer district",
            options=district_options,
            format_func=lambda value: district_labels.get(value, "Unknown"),
            key=letter_form_key("district_id"),
            help="Select the delivery district for this quotation.",
        )
        st.text_input(
            "Customer contact number",
            key=letter_form_key("customer_contact"),
        )
        st.text_input(
            "Attention name",
            key=letter_form_key("attention_name"),
            help="Optional attention line within the letter body.",
        )
        st.text_input(
            "Attention title",
            key=letter_form_key("attention_title"),
        )
        st.text_input(
            "Subject",
            key=letter_form_key("subject_line"),
            help="Subject line printed in bold.",
        )
        st.text_input(
            "Salutation",
            key=letter_form_key("salutation"),
        )
        st.text_area(
            "Introduction / cover paragraph",
            key=letter_form_key("body_intro"),
            help="Opening paragraph placed after the salutation.",
        )
        st.text_area(
            "Product details",
            key=letter_form_key("product_details"),
            help="Technical specification, pricing breakdown or bullet points.",
        )
        st.text_input(
            "Tracked products",
            key=letter_form_key("tracked_products"),
            help="Comma separated list used to sync with reporting filters (e.g. Generator, ATS Panel).",
        )
        st.number_input(
            "Total amount (BDT)",
            min_value=0.0,
            step=1000.0,
            key=letter_form_key("amount"),
        )
        st.selectbox(
            "Quote type",
            options=["retail", "wholesale"],
            format_func=lambda value: value.title(),
            key=letter_form_key("quote_type"),
            help="Specify whether this is a retail or wholesale quotation.",
        )
        st.text_area(
            "Closing / thanks",
            key=letter_form_key("closing_text"),
            help="Closing paragraph before the signature block.",
        )
        st.text_input(
            "Salesperson name",
            key=letter_form_key("salesperson_name"),
            disabled=user["role"] != "admin",
        )
        st.text_input(
            "Salesperson title",
            key=letter_form_key("salesperson_title"),
            disabled=user["role"] != "admin",
        )
        st.text_input(
            "Salesperson contact",
            key=letter_form_key("salesperson_contact"),
            help="Phone to show beneath the signature.",
            disabled=user["role"] != "admin",
        )
        st.text_area(
            "Quotation remarks for admin",
            key=letter_form_key("quotation_remark"),
            help="Notes about this quotation visible to admins.",
        )
        st.selectbox(
            "Salesperson follow-up status",
            options=list(LETTER_FOLLOW_UP_STATUSES),
            key=letter_form_key("follow_up_status"),
            format_func=lambda value: LETTER_FOLLOW_UP_LABELS.get(value, value.title()),
            help="Flag whether the quotation has been paid, is still possible or was rejected.",
        )
        st.text_area(
            "Salesperson follow-up notes",
            key=letter_form_key("follow_up_note"),
            help="Explain the follow-up status for admin review.",
        )

        follow_status_value = st.session_state.get(
            letter_form_key("follow_up_status"), "possible"
        )
        if follow_status_value == "possible":
            quick_choice = st.radio(
                "Suggested follow-up timing",
                list(FOLLOW_UP_SUGGESTIONS.keys()),
                key=letter_form_aux_key("follow_up_choice"),
                horizontal=True,
            )
            previous_choice = st.session_state.get("_letter_follow_up_last_selection")
            if quick_choice != previous_choice:
                set_follow_up_choice(quick_choice)
            st.date_input(
                "Follow-up date",
                key=letter_form_key("follow_up_date"),
                min_value=date.today(),
                help="Choose when to receive a reminder while the opportunity is possible.",
            )
            scheduled_date = _coerce_date(
                st.session_state.get(letter_form_key("follow_up_date"))
            )
            if scheduled_date:
                delta_days = (scheduled_date - date.today()).days
                if delta_days >= 0:
                    st.caption(
                        f"Reminder scheduled in {delta_days} day{'s' if delta_days != 1 else ''} on {scheduled_date:%d %b %Y}."
                    )
                else:
                    st.caption(
                        f"Reminder date is {abs(delta_days)} day{'s' if delta_days != -1 else ''} in the past ({scheduled_date:%d %b %Y})."
                    )
            else:
                st.caption("Select a follow-up date to enable reminder scheduling.")
        else:
            st.session_state[letter_form_key("follow_up_date")] = None
            set_follow_up_choice(CUSTOM_FOLLOW_UP_CHOICE)

        payment_options = ["pending", "paid", "declined"]
        derived_payment_status = {
            "paid": "paid",
            "possible": "pending",
            "rejected": "declined",
        }.get(follow_status_value, "pending")
        st.session_state[letter_form_key("payment_status")] = derived_payment_status
        st.selectbox(
            "Payment status",
            options=payment_options,
            index=payment_options.index(derived_payment_status),
            key=letter_form_key("payment_status"),
            format_func=lambda value: value.replace("_", " ").title(),
            help="Payment status is derived from the follow-up decision.",
            disabled=True,
        )

        if follow_status_value == "paid":
            receipt_upload = st.file_uploader(
                "Payment receipt",
                type=["pdf", "png", "jpg", "jpeg"],
                key=f"letter_receipt_{selected_id or 'new'}",
                help="Upload proof of payment when the quotation is marked as paid.",
            )
            if existing_receipt_path and not receipt_upload:
                st.caption("A receipt is already stored for this quotation.")
                show_pdf_link(existing_receipt_path, "Download receipt")
        elif existing_receipt_path:
            st.caption("A payment receipt is already stored for this quotation.")
            show_pdf_link(existing_receipt_path, "Download receipt")

        if st.button("Save quotation", use_container_width=True):
            payload = prepare_letter_payload(user, existing, selected_id)
            payload["pdf_path"] = existing_pdf_path
            missing = [
                label
                for field, label in LETTER_REQUIRED_LABELS.items()
                if not _letter_field_completed(payload.get(field))
            ]
            if missing:
                st.error(
                    "Please complete the required fields: "
                    + ", ".join(missing)
                )
            elif (
                payload["follow_up_status"] == "possible"
                and not payload.get("follow_up_date")
            ):
                st.error("Select a follow-up date when marking the opportunity as possible.")
            else:
                receipt_path = existing_receipt_path
                if payload["follow_up_status"] == "paid":
                    if receipt_upload:
                        receipt_path = save_uploaded_file(
                            receipt_upload, "quotation_receipts"
                        )
                        if receipt_upload and not receipt_path:
                            return
                    elif not existing_receipt_path:
                        st.error(
                            "Upload a payment receipt when marking the quotation as paid."
                        )
                        return
                payload["payment_receipt"] = receipt_path
                st.session_state[letter_form_key("payment_receipt")] = receipt_path
                created_new = selected_id is None
                letter_id = upsert_quotation_letter(payload)
                pdf_path = persist_letter_pdf(letter_id, payload)
                if pdf_path and pdf_path != payload.get("pdf_path"):
                    update_letter_pdf_path(letter_id, pdf_path)
                    payload["pdf_path"] = pdf_path
                st.session_state[letter_form_key("pdf_path")] = payload.get("pdf_path")
                quotation_id, previous_payment_status = sync_letter_tracking(
                    letter_id, payload
                )
                if created_new:
                    notify_new_quotation(letter_id, payload, user)
                else:
                    customer = (
                        payload.get("customer_company")
                        or payload.get("customer_name")
                        or "customer"
                    )
                    customer_label = str(customer).strip() or "customer"
                    notify_admin_activity(
                        f"Updated quotation letter #{letter_id} for {customer_label}",
                        user,
                    )
                if (
                    payload["follow_up_status"] == "possible"
                    and payload.get("follow_up_date")
                ):
                    schedule_follow_up_notifications(quotation_id)
                if (
                    payload["payment_status"] == "paid"
                    and previous_payment_status != "paid"
                ):
                    notify_payment_recorded(quotation_id, user)
                st.success(f"Quotation #{letter_id} saved successfully.")
                st.session_state["letter_form_active_id"] = letter_id
                rerun()

    with preview_col:
        st.subheader("Preview")
        current_state = get_letter_form_state()
        completion_ratio, missing_required, missing_optional = summarise_letter_completion(
            current_state
        )
        st.metric("Form readiness", f"{completion_ratio * 100:.0f}% complete")
        st.progress(completion_ratio)
        if missing_required:
            st.warning(
                "Complete the required fields: " + ", ".join(missing_required)
            )
        elif missing_optional:
            st.info(
                "Consider enriching these sections: " + ", ".join(missing_optional)
            )
        follow_up_status = current_state.get("follow_up_status")
        follow_up_date = _coerce_date(current_state.get("follow_up_date"))
        if follow_up_status == "possible" and not follow_up_date:
            st.warning("Set a follow-up date to schedule reminders.")
        elif follow_up_status != "possible" and follow_up_date:
            st.info(
                "The follow-up date will be cleared when the status is not marked as possible."
            )
        render_letter_preview(current_state)
        preview_pdf = generate_letter_pdf(current_state)
        reference = str(current_state.get("reference_no") or "quotation_preview")
        safe_reference = re.sub(r"[^A-Za-z0-9]+", "_", reference).strip("_") or "quotation_preview"
        if user["role"] == "admin" or existing_pdf_path:
            st.download_button(
                "Download current preview (PDF)",
                data=preview_pdf,
                file_name=f"{safe_reference}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Save the quotation to enable PDF downloads.")
        if existing_pdf_path:
            show_pdf_link(existing_pdf_path, "Download saved quotation")

    st.divider()
    st.subheader("Saved quotation letters")
    if letters_df.empty:
        st.info("No quotation letters recorded yet.")
    else:
        display_df = letters_df.copy()
        display_df["quote_date"] = pd.to_datetime(
            display_df["quote_date"], errors="coerce"
        ).dt.date
        display_df["follow_up_date"] = pd.to_datetime(
            display_df.get("follow_up_date"), errors="coerce"
        ).dt.date
        display_df["follow_up_status"] = display_df["follow_up_status"].map(
            LETTER_FOLLOW_UP_LABELS
        ).fillna(display_df["follow_up_status"])
        display_df["payment_status"] = (
            display_df.get("payment_status", "pending")
            .fillna("pending")
            .astype(str)
            .str.replace("_", " ")
            .str.title()
        )
        columns = [
            "letter_id",
            "reference_no",
            "quote_date",
            "customer_company",
            "follow_up_status",
            "follow_up_date",
            "payment_status",
            "quotation_remark",
        ]
        if user["role"] == "admin" and "salesperson" in display_df.columns:
            columns.append("salesperson")
        display_df = display_df[columns]
        display_df = display_df.rename(
            columns={
                "letter_id": "Letter #",
                "reference_no": "Reference",
                "quote_date": "Date",
                "customer_company": "Company",
                "follow_up_status": "Follow-up",
                "follow_up_date": "Follow-up date",
                "payment_status": "Payment",
                "quotation_remark": "Remarks",
                "salesperson": "Salesperson",
            }
        )
        st.dataframe(display_df, use_container_width=True)

        detail_options: List[Tuple[str, Optional[int]]] = [
            ("Select quotation", None)
        ]
        for row in letters_df.itertuples():
            label = f"#{int(row.letter_id)} – {row.customer_company}"
            if getattr(row, "quote_date", None):
                label = f"{label} ({row.quote_date})"
            detail_options.append((label, int(row.letter_id)))
        detail_label = st.selectbox(
            "View saved quotation details",
            [label for label, _ in detail_options],
            key="quotation_letter_detail_selector",
        )
        detail_id = dict(detail_options)[detail_label]
        if detail_id:
            detail_row = letters_df.loc[letters_df["letter_id"] == detail_id].iloc[0]
            info_cols = st.columns(2)
            info_cols[0].write(f"**Reference:** {detail_row['reference_no']}")
            info_cols[0].write(f"**Date:** {detail_row['quote_date']}")
            info_cols[0].write(
                f"**Customer:** {detail_row['customer_name']} ({detail_row['customer_company']})"
            )
            info_cols[0].write(f"**Contact:** {detail_row.get('customer_contact') or '—'}")
            info_cols[1].write(
                "**Payment status:** "
                f"{str(detail_row.get('payment_status') or 'pending').replace('_', ' ').title()}"
            )
            follow_up_date = detail_row.get("follow_up_date")
            if follow_up_date:
                info_cols[1].write(f"**Follow-up date:** {follow_up_date}")
            info_cols[1].write(
                "**Follow-up status:** "
                f"{LETTER_FOLLOW_UP_LABELS.get(detail_row.get('follow_up_status'), '—')}"
            )
            st.markdown("**Address**")
            st.write(detail_row["customer_address"])
            if detail_row.get("product_details"):
                st.markdown("**Product details**")
                st.write(detail_row["product_details"])
            if detail_row.get("quotation_remark"):
                st.markdown("**Remarks**")
                st.write(detail_row["quotation_remark"])
            amount_value = detail_row.get("amount")
            if amount_value is not None:
                st.markdown("**Amount**")
                st.write(f"BDT {float(amount_value):,.2f}")
            if detail_row.get("pdf_path"):
                show_pdf_link(detail_row["pdf_path"], "Download quotation PDF")
            if detail_row.get("payment_receipt"):
                show_pdf_link(detail_row["payment_receipt"], "Download payment receipt")


def render_quotations(user: Dict) -> None:
    st.header("Quotations")

    with get_conn() as conn:
        companies = conn.execute(
            "SELECT * FROM companies ORDER BY name"
        ).fetchall()
        districts = conn.execute(
            "SELECT district_id, name FROM districts ORDER BY name"
        ).fetchall()
        products = conn.execute(
            "SELECT category_id, name FROM categories ORDER BY name"
        ).fetchall()

    st.subheader("Create or edit quotation")
    existing_df = list_quotations(user)
    options = [("New quotation", None)] + [
        (f"#{row['quotation_id']} – {row['company']} ({row['status']})", row["quotation_id"])
        for _, row in existing_df.iterrows()
    ]
    selection = st.selectbox("Select quotation", [label for label, _ in options])
    selected_id = dict(options)[selection]
    existing = get_quotation(selected_id)

    district_map = {row["name"]: row["district_id"] for row in districts}
    district_by_id = {row["district_id"]: row["name"] for row in districts}
    products_by_id = {row["category_id"]: row["name"] for row in products}

    company_id = None
    district_id: Optional[int] = None
    contact_person = ""
    phone = ""
    address = ""
    delivery_address = ""
    company_name = ""
    company_district = districts[0]["name"] if districts else ""
    company_options: List[Tuple[str, Optional[int]]] = [
        ("-- Select --", None),
        *[(row["name"], row["company_id"]) for row in companies],
        ("Add new company…", "__new__"),
    ]
    default_index = 0
    if existing:
        default_index = next(
            (
                i
                for i, option in enumerate(company_options)
                if option[1] == existing["company_id"]
            ),
            0,
        )
    elif st.session_state.get("quotation_form_company_id"):
        saved_company_id = st.session_state.get("quotation_form_company_id")
        default_index = next(
            (
                i
                for i, option in enumerate(company_options)
                if option[1] == saved_company_id
            ),
            0,
        )
    elif not companies:
        default_index = len(company_options) - 1
    selected_company = st.selectbox(
        "Company",
        options=company_options,
        index=default_index,
        format_func=lambda option: option[0],
    )

    if selected_company[1] not in (None, "__new__"):
        st.session_state["quotation_form_company_id"] = int(selected_company[1])
    elif selected_company[1] == "__new__":
        st.session_state.pop("quotation_form_company_id", None)

    new_company_selected = selected_company[1] == "__new__"
    if new_company_selected:
        with st.expander("New company details", expanded=True):
            info_cols = st.columns(2)
            with info_cols[0]:
                company_name = st.text_input("Company name")
                contact_person = st.text_input("Contact person")
                phone = st.text_input("Phone")
            with info_cols[1]:
                company_district = (
                    st.selectbox(
                        "District",
                        options=[row["name"] for row in districts],
                    )
                    if districts
                    else ""
                )
            address = st.text_area("Company address")
            delivery_address = st.text_area("Delivery address")
    elif selected_company[1]:
        company_id = int(selected_company[1])

    selected_company_row: Optional[sqlite3.Row] = None
    if existing and existing.get("district_id"):
        district_id = existing["district_id"]
    if selected_company[1] and selected_company[1] != "__new__":
        company_id = int(selected_company[1])
        selected_company_row = next(
            (row for row in companies if row["company_id"] == company_id),
            None,
        )
        if selected_company_row:
            district_id = selected_company_row["district_id"]
    district_name = district_by_id.get(district_id or -1)
    if selected_company_row and district_name:
        st.caption(f"District: {district_name}")

    selected_key = selected_id or "new"
    existing_product_rows = get_quotation_products(selected_id)
    if not existing_product_rows and existing:
        fallback_quantity = int(existing["quantity"]) if existing["quantity"] else 1
        existing_product_rows = [(existing["category_id"], fallback_quantity)]
    existing_names = [products_by_id.get(pid, "") for pid, _ in existing_product_rows]
    existing_quantities = [qty for _, qty in existing_product_rows]
    line_count_default = max(len(existing_product_rows), 1)
    line_count_key = f"line_count_{selected_key}"
    status_key = f"quotation_status_{selected_key}"
    follow_up_key = f"quotation_follow_up_{selected_key}"
    active_key = st.session_state.get("quotation_form_active_key")
    default_status = existing["status"] if existing else "pending"
    default_follow_up = (
        datetime.fromisoformat(existing["follow_up_date"]).date()
        if existing and existing["follow_up_date"]
        else date.today() + timedelta(days=3)
    )
    if active_key != selected_key:
        st.session_state["quotation_form_active_key"] = selected_key
        st.session_state[line_count_key] = line_count_default
        st.session_state[status_key] = default_status
        if default_status == "inform_later":
            st.session_state[follow_up_key] = default_follow_up
        else:
            st.session_state.pop(follow_up_key, None)
    st.session_state.setdefault(line_count_key, line_count_default)
    st.session_state.setdefault(status_key, default_status)
    st.number_input(
        "Quantity",
        min_value=1,
        value=st.session_state[line_count_key],
        step=1,
        help="Increase to add more product rows below.",
        key=line_count_key,
    )
    status_follow_cols = st.columns(2)
    follow_up_date_value: Optional[date] = None
    with status_follow_cols[0]:
        status = st.selectbox(
            "Status",
            ["pending", "accepted", "declined", "inform_later"],
            key=status_key,
        )
    with status_follow_cols[1]:
        if status == "inform_later":
            if follow_up_key not in st.session_state:
                st.session_state[follow_up_key] = default_follow_up
            follow_up_date_value = st.date_input(
                "Follow-up date (required)",
                key=follow_up_key,
            )
        else:
            st.session_state.pop(follow_up_key, None)
    if status != "inform_later":
        follow_up_date_value = None

    with st.form("quotation_form"):
        line_item_count = int(st.session_state.get(line_count_key, line_count_default))
        product_entries: List[Tuple[str, int]] = []
        for idx in range(line_item_count):
            row_cols = st.columns((3, 1))
            name_default = existing_names[idx] if idx < len(existing_names) else (existing_names[0] if existing_names and idx == 0 else "")
            qty_default = (
                int(existing_quantities[idx])
                if idx < len(existing_quantities)
                else (int(existing_quantities[0]) if existing_quantities and idx == 0 else 1)
            )
            with row_cols[0]:
                name_value = st.text_input(
                    f"Product #{idx + 1}",
                    value=name_default,
                    help="Enter the product for this quotation.",
                    placeholder="50 kVA generator" if idx == 0 else "Additional product",
                    key=f"product_{selected_key}_{idx}",
                )
            with row_cols[1]:
                quantity_value = st.number_input(
                    f"Quantity #{idx + 1}",
                    min_value=1,
                    value=int(qty_default) if qty_default else 1,
                    step=1,
                    key=f"product_qty_{selected_key}_{idx}",
                )
            product_entries.append((name_value, int(quantity_value)))

        existing_receipt_path = existing["payment_receipt"] if existing else None
        receipt_upload = None
        detail_cols = st.columns(2)
        with detail_cols[0]:
            quote_type = st.selectbox(
                "Quote type",
                ["retail", "wholesale"],
                index=["retail", "wholesale"].index(existing["quote_type"]) if existing else 0,
            )
            quote_date = st.date_input(
                "Quote date",
                value=datetime.fromisoformat(existing["quote_date"]).date() if existing else date.today(),
            )
        with detail_cols[1]:
            payment_status_options = ["pending", "paid", "declined"]
            current_payment_status = (
                existing["payment_status"]
                if existing and existing["payment_status"] in payment_status_options
                else "pending"
            )
            payment_status = st.selectbox(
                "Payment status",
                options=payment_status_options,
                index=payment_status_options.index(current_payment_status),
                help="Track whether this quotation has been paid.",
            )
            receipt_key = f"receipt_{selected_id or 'new'}"
            if payment_status == "paid":
                receipt_upload = st.file_uploader(
                    "Payment receipt",
                    type=["pdf", "png", "jpg", "jpeg"],
                    key=receipt_key,
                    help="Upload the payment proof for this quotation.",
                )
                if existing_receipt_path and not receipt_upload:
                    st.caption("A receipt is already stored for this quotation.")
            elif existing_receipt_path:
                st.caption("Stored receipt available for download below.")

        notes = st.text_area("Notes", value=existing["notes"] if existing else "")
        pdf_upload = st.file_uploader("Quotation PDF", type=["pdf"])
        if existing_receipt_path and (payment_status != "paid" or not receipt_upload):
            st.markdown("**Stored receipt**")
            show_pdf_link(existing_receipt_path, "Download receipt")
        submitted = st.form_submit_button("Save quotation", use_container_width=True)

        if submitted:
            created_new = existing is None
            payment_receipt_path = existing_receipt_path
            product_items: List[Tuple[int, int]] = []
            for idx, (name_value, qty_value) in enumerate(product_entries, start=1):
                product_name_value = name_value.strip()
                if not product_name_value:
                    st.error(f"Enter a product name for row {idx}.")
                    return
                try:
                    product_identifier = ensure_product(product_name_value)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                product_items.append((int(product_identifier), int(qty_value)))
            if not product_items:
                st.error("Please enter at least one product")
                return
            unique_product_ids = list(dict.fromkeys(product_id for product_id, _ in product_items))
            total_quantity = sum(qty for _, qty in product_items)
            if total_quantity <= 0:
                st.error("Quantity must be at least 1")
                return

            if new_company_selected:
                required = [
                    company_name.strip(),
                    phone.strip(),
                    address.strip(),
                    delivery_address.strip(),
                    company_district,
                ]
                if not all(required):
                    st.error("Please complete new company details")
                    return
                if company_district not in district_map:
                    st.error("Select a district for the new company")
                    return
                upsert_company(
                    {
                        "name": company_name.strip(),
                        "contact_person": contact_person.strip() or None,
                        "phone": phone.strip(),
                        "address": address.strip(),
                        "delivery_address": delivery_address.strip(),
                        "district_id": district_map[company_district],
                    },
                    unique_product_ids,
                )
                with get_conn() as conn:
                    row = conn.execute(
                        "SELECT company_id FROM companies WHERE name=? ORDER BY company_id DESC LIMIT 1",
                        (company_name.strip(),),
                    ).fetchone()
                company_id = row[0] if row else None
                if company_district in district_map:
                    district_id = district_map[company_district]
                if company_id:
                    st.session_state["quotation_form_company_id"] = int(company_id)
            if not company_id:
                st.error("Please select a company")
                return
            if not district_id:
                st.error("Select a district")
                return
            follow_up_date = follow_up_date_value
            if status == "inform_later" and not follow_up_date:
                st.error("Select a follow-up date for inform later status")
                return
            for product_id in unique_product_ids:
                link_company_product(company_id, product_id)
            if payment_status == "paid":
                if receipt_upload:
                    saved_receipt = save_uploaded_file(receipt_upload, "receipts")
                    if not saved_receipt:
                        st.error("Receipt upload failed")
                        return
                    payment_receipt_path = saved_receipt
                elif not payment_receipt_path:
                    st.error("Upload a receipt before marking as paid")
                    return
            elif receipt_upload:
                saved_receipt = save_uploaded_file(receipt_upload, "receipts")
                if saved_receipt:
                    payment_receipt_path = saved_receipt
            pdf_path = (
                save_uploaded_file(pdf_upload, "quotations")
                if pdf_upload
                else (existing["pdf_path"] if existing else None)
            )
            data = {
                "quotation_id": existing["quotation_id"] if existing else None,
                "salesperson_id": existing["salesperson_id"] if existing else user["user_id"],
                "company_id": company_id,
                "district_id": district_id,
                "category_id": product_items[0][0],
                "quote_date": quote_date.isoformat(),
                "status": status,
                "follow_up_date": follow_up_date.isoformat() if follow_up_date else None,
                "pdf_path": pdf_path,
                "notes": notes,
                "quote_type": quote_type,
                "kva": None,
                "quantity": int(total_quantity),
                "payment_status": payment_status,
                "payment_receipt": payment_receipt_path,
            }
            quotation_id = upsert_quotation(data)
            set_quotation_products(quotation_id, product_items)
            with get_conn() as conn:
                detail = conn.execute(
                    textwrap.dedent(
                        """
                        SELECT c.name AS company
                        FROM quotations q
                        JOIN companies c ON c.company_id = q.company_id
                        WHERE q.quotation_id=?
                        """
                    ),
                    (quotation_id,),
                ).fetchone()
            company_label = (
                detail["company"].strip() if detail and detail["company"] else "quotation"
            )
            verb = "Created" if created_new else "Updated"
            notify_admin_activity(
                f"{verb} quotation #{quotation_id} for {company_label}",
                user,
            )
            if status == "inform_later" and follow_up_date:
                schedule_follow_up_notifications(quotation_id)
            if payment_status == "paid" and (
                not existing or existing.get("payment_status") != "paid"
            ):
                notify_payment_recorded(quotation_id, user)
            st.success("Quotation saved")
            rerun()

    st.subheader("Quotation list")
    df = existing_df.copy()
    if df.empty:
        st.info("No quotations recorded yet.")
        return

    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
    min_quote_date = df["quote_date"].dropna().min()
    max_quote_date = df["quote_date"].dropna().max()

    statuses_from_data = {
        str(status).strip()
        for status in df["status"].dropna()
        if str(status).strip()
    }
    available_statuses: List[str] = list(DEFAULT_QUOTATION_STATUSES)
    for status in sorted(statuses_from_data):
        if status not in available_statuses:
            available_statuses.append(status)

    filters = st.expander("Filters", expanded=False)
    with filters:
        status_filter = st.multiselect("Status", available_statuses)
        start_date = st.date_input(
            "Start date",
            value=None,
            key="quotation_filter_start_date",
            min_value=min_quote_date.date() if pd.notna(min_quote_date) else None,
            max_value=max_quote_date.date() if pd.notna(max_quote_date) else None,
        )
        end_date = st.date_input(
            "End date",
            value=None,
            key="quotation_filter_end_date",
            min_value=min_quote_date.date() if pd.notna(min_quote_date) else None,
            max_value=max_quote_date.date() if pd.notna(max_quote_date) else None,
        )
        if start_date and end_date and start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

    filtered_df = df.copy()
    if status_filter:
        filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
    if start_date:
        filtered_df = filtered_df[
            filtered_df["quote_date"].dt.date >= start_date
        ]
    if end_date:
        filtered_df = filtered_df[filtered_df["quote_date"].dt.date <= end_date]

    if filtered_df.empty:
        st.info("No quotations match the selected filters.")
        return

    display_df = filtered_df.copy()
    display_df.sort_values(["quote_date", "quotation_id"], ascending=[False, True], inplace=True)
    display_df.reset_index(drop=True, inplace=True)
    display_df["quote_date"] = display_df["quote_date"].dt.strftime("%Y-%m-%d").fillna("")

    editable_column = "Mark as declined"
    display_df[editable_column] = (
        display_df["status"].astype(str).str.strip().str.lower() == "declined"
    )

    disabled_columns = [
        column for column in display_df.columns if column != editable_column
    ]

    decline_state_key = "_quotation_decline_table_state"
    table_signature = hashlib.md5(
        display_df.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    decline_state = st.session_state.get(decline_state_key)
    if not decline_state or decline_state.get("signature") != table_signature:
        st.session_state[decline_state_key] = {
            "signature": table_signature,
            "data": display_df.copy(),
        }
        decline_state = st.session_state[decline_state_key]

    working_df = decline_state.get("data", display_df).copy()

    decline_form_key = "quotation_decline_editor"
    save_declines = False
    edited_df = working_df
    with st.form(decline_form_key):
        edited_df = st.data_editor(
            working_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                editable_column: st.column_config.CheckboxColumn(
                    "Decline",
                    help="Select to mark the quotation as declined.",
                )
            },
            disabled=disabled_columns,
            key="quotation_table_editor",
        )
        st.caption(
            "Checked rows will be recorded as declined when you save the table updates."
        )
        save_declines = st.form_submit_button(
            "Save table updates", use_container_width=True
        )

    st.session_state[decline_state_key]["data"] = edited_df.copy()

    if save_declines:
        decline_targets = (
            edited_df.loc[
                (edited_df[editable_column]) & (edited_df["status"] != "declined"),
                "quotation_id",
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        if decline_targets:
            decline_quotations(decline_targets)
            decline_labels = ", ".join(f"#{qid}" for qid in decline_targets[:5])
            if len(decline_targets) > 5:
                decline_labels += ", …"
            notify_admin_activity(
                f"Declined quotation(s) {decline_labels}",
                user,
            )
            st.success(f"Declined {len(decline_targets)} quotation(s).")
            rerun()
        else:
            st.info("No changes to save from the quotations table.")


def render_work_orders(user: Dict) -> None:
    st.header("Work orders")
    with get_conn() as conn:
        query = textwrap.dedent(
            """
            SELECT q.quotation_id, q.status, c.name AS company
            FROM quotations q
            JOIN companies c ON c.company_id = q.company_id
            {condition}
            ORDER BY q.quote_date DESC
            """
        ).format(condition="WHERE q.salesperson_id=?" if user["role"] == "staff" else "")
        params: Tuple = ((user["user_id"],) if user["role"] == "staff" else ())
        available_quotes = conn.execute(query, params).fetchall()

        wo_query = textwrap.dedent(
            """
            SELECT w.work_order_id, w.upload_date, w.pdf_path, w.notes,
                   q.quotation_id, c.name AS company
            FROM work_orders w
            JOIN quotations q ON q.quotation_id = w.quotation_id
            JOIN companies c ON c.company_id = q.company_id
            {condition}
            ORDER BY w.upload_date DESC
            """
        ).format(condition="WHERE q.salesperson_id=?" if user["role"] == "staff" else "")
        work_orders = conn.execute(
            wo_query, (user["user_id"],) if user["role"] == "staff" else ()
        ).fetchall()

    options = [("New work order", None)] + [
        (f"#{row['work_order_id']} – quotation #{row['quotation_id']}", row["work_order_id"])
        for row in work_orders
    ]
    selection = st.selectbox("Select work order", [label for label, _ in options])
    work_order_id = dict(options)[selection]
    existing = next((row for row in work_orders if row["work_order_id"] == work_order_id), None)

    with st.form("work_order_form"):
        quotation_options = [
            ("Select quotation", None),
            *[
                (
                    f"#{row['quotation_id']} – {row['company']} ({row['status']})",
                    row["quotation_id"],
                )
                for row in available_quotes
            ],
        ]
        if existing and existing["quotation_id"]:
            present_ids = {value for _, value in quotation_options if value is not None}
            if existing["quotation_id"] not in present_ids:
                quotation_options.append(
                    (
                        f"#{existing['quotation_id']} – {existing['company']}",
                        existing["quotation_id"],
                    )
                )
        option_labels = [label for label, _ in quotation_options]
        default_index = 0
        if existing and existing["quotation_id"]:
            default_index = next(
                (
                    i
                    for i, (_, value) in enumerate(quotation_options)
                    if value == existing["quotation_id"]
                ),
                0,
            )
        quotation_label = st.selectbox(
            "Quotation",
            options=option_labels,
            index=default_index,
        )
        selected_quotation_id = dict(quotation_options)[quotation_label]
        if not any(value is not None for _, value in quotation_options):
            st.info("No quotations available yet. Create a quotation first.")
        upload_date = st.date_input(
            "Upload date",
            value=(
                datetime.fromisoformat(existing["upload_date"]).date()
                if existing
                else date.today()
            ),
        )
        notes = st.text_area("Notes", value=existing["notes"] if existing else "")
        pdf_upload = st.file_uploader("Work order PDF", type=["pdf"], key="wo_pdf")
        submitted = st.form_submit_button("Save work order")

        if submitted:
            if not selected_quotation_id:
                st.error("Select a quotation to save the work order")
                return
            pdf_path = (
                save_uploaded_file(pdf_upload, "work_orders")
                if pdf_upload
                else (existing["pdf_path"] if existing else None)
            )
            data = {
                "work_order_id": existing["work_order_id"] if existing else None,
                "quotation_id": int(selected_quotation_id),
                "upload_date": upload_date.isoformat(),
                "pdf_path": pdf_path,
                "notes": notes,
            }
            work_order_id = upsert_work_order(data)
            with get_conn() as conn:
                detail = conn.execute(
                    textwrap.dedent(
                        """
                        SELECT q.quotation_id, c.name AS company
                        FROM work_orders w
                        JOIN quotations q ON q.quotation_id = w.quotation_id
                        JOIN companies c ON c.company_id = q.company_id
                        WHERE w.work_order_id=?
                        """
                    ),
                    (work_order_id,),
                ).fetchone()
            company_label = (
                detail["company"].strip() if detail and detail["company"] else "quotation"
            )
            quotation_ref = detail["quotation_id"] if detail else selected_quotation_id
            notify_admin_activity(
                f"Saved work order #{work_order_id} for quotation #{quotation_ref} ({company_label})",
                user,
            )
            st.success("Work order saved")
            rerun()

    st.subheader("Existing work orders")
    if work_orders:
        for row in work_orders:
            with st.expander(
                f"Work order #{row['work_order_id']} – quotation #{row['quotation_id']}"
            ):
                st.write(f"Company: {row['company']}")
                st.write(f"Upload date: {row['upload_date']}")
                st.write(f"Notes: {row['notes'] or '—'}")
                show_pdf_link(row["pdf_path"], "Download work order")
    else:
        st.info("No work orders yet")


def render_delivery_orders(user: Dict) -> None:
    st.header("Delivery orders")
    with get_conn() as conn:
        work_order_query = textwrap.dedent(
            """
            SELECT w.work_order_id, w.upload_date, q.quotation_id, q.status, c.name AS company
            FROM work_orders w
            JOIN quotations q ON q.quotation_id = w.quotation_id
            JOIN companies c ON c.company_id = q.company_id
            {condition}
            ORDER BY w.upload_date DESC
            """
        ).format(condition="WHERE q.salesperson_id=?" if user["role"] == "staff" else "")
        params = (user["user_id"],) if user["role"] == "staff" else ()
        work_orders = conn.execute(work_order_query, params).fetchall()

        quotation_query = textwrap.dedent(
            """
            SELECT q.quotation_id, q.status, q.quote_date, c.name AS company
            FROM quotations q
            JOIN companies c ON c.company_id = q.company_id
            {condition}
            ORDER BY q.quote_date DESC
            """
        ).format(condition="WHERE q.salesperson_id=?" if user["role"] == "staff" else "")
        quotations = conn.execute(quotation_query, params).fetchall()

        delivery_query = textwrap.dedent(
            """
            SELECT d.*, COALESCE(c.name, d.third_party_name, '—') AS company_name,
                   q.quotation_id AS linked_quotation_id
            FROM delivery_orders d
            LEFT JOIN work_orders w ON w.work_order_id = d.work_order_id
            LEFT JOIN quotations q ON q.quotation_id = COALESCE(d.quotation_id, w.quotation_id)
            LEFT JOIN companies c ON c.company_id = q.company_id
            {condition}
            ORDER BY d.upload_date DESC
            """
        ).format(condition="WHERE d.salesperson_id=?" if user["role"] == "staff" else "")
        delivery_orders = conn.execute(
            delivery_query, (user["user_id"],) if user["role"] == "staff" else ()
        ).fetchall()

    source_labels = {
        "work_order": "Workdone",
        "quotation": "Quotation",
        "third_party": "Third-party",
    }
    options = [("New delivery order", None)] + [
        (
            f"#{row['do_id']} – {source_labels.get(row['source_type'], row['source_type'])}",
            row["do_id"],
        )
        for row in delivery_orders
    ]
    selection = st.selectbox("Select delivery order", [label for label, _ in options])
    do_id = dict(options)[selection]
    existing = next((row for row in delivery_orders if row["do_id"] == do_id), None)

    do_key = do_id or "new"
    source_key = f"delivery_source_{do_key}"
    active_delivery_key = st.session_state.get("delivery_form_active_key")
    default_source = existing["source_type"] if existing else "work_order"
    if active_delivery_key != do_key:
        st.session_state["delivery_form_active_key"] = do_key
        st.session_state[source_key] = default_source
    st.session_state.setdefault(source_key, default_source)
    source_type = st.radio(
        "Delivery order source",
        options=list(source_labels.keys()),
        format_func=lambda value: source_labels[value],
        key=source_key,
        horizontal=True,
    )
    existing_receipt_path = existing["receipt_path"] if existing else None
    with st.form("delivery_order_form"):

        selected_work_order: Optional[int] = None
        selected_quotation: Optional[int] = None
        existing_work_order_id = existing["work_order_id"] if existing else None
        existing_quotation_id = existing["quotation_id"] if existing else None
        existing_linked_quotation_id = existing["linked_quotation_id"] if existing else None
        third_party_name = (existing["third_party_name"] or "") if existing else ""

        if source_type == "work_order":
            work_order_options = [
                ("Select work order", None),
                *[
                    (
                        f"#{row['work_order_id']} – {row['company']} ({row['status']})",
                        row["work_order_id"],
                    )
                    for row in work_orders
                ],
            ]
            if existing_work_order_id:
                present_ids = {value for _, value in work_order_options if value is not None}
                if existing_work_order_id not in present_ids:
                    fallback_label = (
                        f"#{existing_work_order_id} – {existing['company_name']}"
                        if existing and existing.get("company_name")
                        else f"#{existing_work_order_id}"
                    )
                    work_order_options.append((fallback_label, existing_work_order_id))
            work_order_labels = [label for label, _ in work_order_options]
            default_index = 0
            if existing_work_order_id:
                default_index = next(
                    (
                        i
                        for i, (_, value) in enumerate(work_order_options)
                        if value == existing_work_order_id
                    ),
                    0,
                )
            work_order_label = st.selectbox(
                "Work order",
                options=work_order_labels,
                index=default_index,
            )
            selected_work_order = dict(work_order_options)[work_order_label]
            if not any(value is not None for _, value in work_order_options):
                st.info("No work orders available yet.")
        elif source_type == "quotation":
            quotation_options = [
                ("Select quotation", None),
                *[
                    (
                        f"#{row['quotation_id']} – {row['company']} ({row['status']})",
                        row["quotation_id"],
                    )
                    for row in quotations
                ],
            ]
            existing_target = existing_quotation_id or existing_linked_quotation_id
            if existing_target:
                present_ids = {value for _, value in quotation_options if value is not None}
                if existing_target not in present_ids:
                    fallback_label = (
                        f"#{existing_target} – {existing['company_name']}"
                        if existing and existing.get("company_name")
                        else f"#{existing_target}"
                    )
                    quotation_options.append((fallback_label, existing_target))
            quotation_labels = [label for label, _ in quotation_options]
            default_index = 0
            if existing_target:
                default_index = next(
                    (
                        i
                        for i, (_, value) in enumerate(quotation_options)
                        if value == existing_target
                    ),
                    0,
                )
            quotation_label = st.selectbox(
                "Quotation",
                options=quotation_labels,
                index=default_index,
            )
            selected_quotation = dict(quotation_options)[quotation_label]
            if not any(value is not None for _, value in quotation_options):
                st.info("No quotations available yet.")
        else:
            third_party_name = st.text_input(
                "Third-party name",
                value=third_party_name,
                help="Enter the company or customer name for external work.",
            )
        do_number = st.text_input(
            "Delivery order number", value=existing["do_number"] if existing else ""
        )
        upload_date = st.date_input(
            "Upload date",
            value=(
                datetime.fromisoformat(existing["upload_date"]).date()
                if existing
                else date.today()
            ),
        )
        price = st.number_input(
            "Price", min_value=0.0, value=float(existing["price"]) if existing else 0.0
        )
        payment_received = st.checkbox(
            "Payment received", value=bool(existing["payment_received"]) if existing else False
        )
        payment_date = None
        if payment_received:
            payment_date = st.date_input(
                "Payment date",
                value=(
                    datetime.fromisoformat(existing["payment_date"]).date()
                    if existing and existing["payment_date"]
                    else date.today()
                ),
            )
        receipt_upload = None
        if payment_received:
            receipt_upload = st.file_uploader(
                "Payment receipt",
                type=["pdf", "png", "jpg", "jpeg"],
                key=f"do_receipt_{do_id or 'new'}",
                help="Upload proof of payment.",
            )
            if existing_receipt_path and not receipt_upload:
                st.caption("A receipt is already stored for this delivery order.")
        elif existing_receipt_path:
            st.caption("Stored receipt available for download below.")
        notes = st.text_area("Notes", value=existing["notes"] if existing else "")
        pdf_upload = st.file_uploader("Delivery order PDF", type=["pdf"], key="do_pdf")
        if existing_receipt_path and (not payment_received or not receipt_upload):
            st.markdown("**Stored receipt**")
            show_pdf_link(existing_receipt_path, "Download receipt")
        submitted = st.form_submit_button("Save delivery order")

    if submitted:
        if source_type == "work_order" and not selected_work_order:
            st.error("Select a work order to link this delivery order to.")
            return
        if source_type == "quotation" and not selected_quotation:
            st.error("Select a quotation to link this delivery order to.")
            return
        if source_type == "third_party" and not third_party_name.strip():
            st.error("Enter the third-party name.")
            return
        linked_quotation = None
        if source_type == "work_order" and selected_work_order:
            linked_quotation = next(
                (
                    row["quotation_id"]
                    for row in work_orders
                    if row["work_order_id"] == selected_work_order
                ),
                None,
            )
        elif source_type == "quotation":
            linked_quotation = selected_quotation
        receipt_path = existing_receipt_path
        if payment_received:
            if receipt_upload:
                saved_receipt = save_uploaded_file(receipt_upload, "receipts")
                if not saved_receipt:
                    st.error("Receipt upload failed")
                    return
                receipt_path = saved_receipt
            elif not receipt_path:
                st.error("Upload a receipt before marking this delivery order as paid.")
                return
        elif receipt_upload:
            saved_receipt = save_uploaded_file(receipt_upload, "receipts")
            if saved_receipt:
                receipt_path = saved_receipt
        pdf_path = (
            save_uploaded_file(pdf_upload, "delivery_orders")
            if pdf_upload
            else (existing["pdf_path"] if existing else None)
        )
        data = {
            "do_id": existing["do_id"] if existing else None,
            "source_type": source_type,
            "salesperson_id": existing["salesperson_id"] if existing else user["user_id"],
            "work_order_id": selected_work_order,
            "quotation_id": linked_quotation,
            "third_party_name": third_party_name.strip() if source_type == "third_party" else None,
            "do_number": do_number,
            "upload_date": upload_date.isoformat(),
            "pdf_path": pdf_path,
            "price": price,
            "payment_received": payment_received,
            "payment_date": payment_date.isoformat() if payment_date else None,
            "notes": notes,
            "receipt_path": receipt_path,
        }
        existing_paid = bool(existing["payment_received"]) if existing else False
        delivery_id = upsert_delivery_order(data)
        if linked_quotation:
            if payment_received:
                set_quotation_payment_status(linked_quotation, "paid", receipt_path)
            else:
                set_quotation_payment_status(linked_quotation, "pending", receipt_path)
        if payment_received and not existing_paid and linked_quotation:
            notify_payment_recorded(linked_quotation, user)
        company_label: Optional[str] = None
        context_parts: List[str] = []
        if linked_quotation:
            context_parts.append(f"quotation #{linked_quotation}")
            with get_conn() as conn:
                company_row = conn.execute(
                    textwrap.dedent(
                        """
                        SELECT c.name AS company
                        FROM quotations q
                        JOIN companies c ON c.company_id = q.company_id
                        WHERE q.quotation_id=?
                        """
                    ),
                    (linked_quotation,),
                ).fetchone()
            if company_row and company_row["company"]:
                company_label = str(company_row["company"]).strip() or None
        if source_type == "work_order" and selected_work_order:
            context_parts.append(f"work order #{selected_work_order}")
            work_row = next(
                (row for row in work_orders if row["work_order_id"] == selected_work_order),
                None,
            )
            if work_row and work_row["company"]:
                company_label = str(work_row["company"]).strip() or company_label
        if source_type == "third_party":
            context_parts.append("third-party")
            company_label = third_party_name.strip() or company_label or "third-party"
        company_fragment = f" for {company_label}" if company_label else ""
        context_fragment = f" ({', '.join(context_parts)})" if context_parts else ""
        notify_admin_activity(
            f"Saved delivery order #{delivery_id}{company_fragment}{context_fragment}",
            user,
        )
        st.success("Delivery order saved")
        rerun()

    st.subheader("Existing delivery orders")
    if delivery_orders:
        for row in delivery_orders:
            title_parts = [f"Delivery order #{row['do_id']}"]
            if row["source_type"] == "work_order" and row["work_order_id"]:
                title_parts.append(f"work order #{row['work_order_id']}")
            elif row["source_type"] == "quotation" and row["linked_quotation_id"]:
                title_parts.append(f"quotation #{row['linked_quotation_id']}")
            elif row["source_type"] == "third_party":
                title_parts.append("third-party")
            with st.expander(" – ".join(title_parts)):
                st.write(f"Source: {source_labels[row['source_type']]}")
                st.write(f"Company: {row['company_name']}")
                if row["linked_quotation_id"]:
                    st.write(f"Related quotation: #{row['linked_quotation_id']}")
                st.write(f"Upload date: {row['upload_date']}")
                st.write(f"Price: {row['price']}")
                st.write(
                    "Payment received: "
                    + ("Yes" if row["payment_received"] else "No")
                )
                st.write(
                    f"Payment date: {row['payment_date'] or '—'}"
                )
                st.write(f"Notes: {row['notes'] or '—'}")
                show_pdf_link(row["pdf_path"], "Download delivery order")
                if row["receipt_path"]:
                    show_pdf_link(row["receipt_path"], "Download receipt")
    else:
        st.info("No delivery orders yet")


def load_admin_dataset() -> pd.DataFrame:
    query = textwrap.dedent(
        """
        SELECT
            q.quotation_id,
            q.quote_date,
            q.status,
            q.follow_up_date,
            q.quote_type,
            q.quantity,
            q.notes,
            c.name AS company,
            d.name AS district,
            COALESCE(prod.names, cat.name || CASE WHEN q.quantity > 1 THEN ' (x' || q.quantity || ')' ELSE '' END) AS product,
            COALESCE(u.display_name, u.username) AS salesperson,
            w.work_order_id,
            w.upload_date AS work_order_date,
            do_tbl.do_id AS delivery_order_id,
            do_tbl.do_number,
            do_tbl.upload_date AS delivery_order_date,
            do_tbl.price,
            do_tbl.payment_received,
            do_tbl.payment_date,
            do_tbl.source_type,
            do_tbl.third_party_name
        FROM quotations q
        JOIN companies c ON c.company_id = q.company_id
        JOIN districts d ON d.district_id = q.district_id
        JOIN categories cat ON cat.category_id = q.category_id
        JOIN users u ON u.user_id = q.salesperson_id
        LEFT JOIN ({subquery}) prod ON prod.quotation_id = q.quotation_id
        LEFT JOIN work_orders w ON w.quotation_id = q.quotation_id
        LEFT JOIN delivery_orders do_tbl ON (
            (do_tbl.work_order_id = w.work_order_id AND do_tbl.source_type = 'work_order')
            OR (do_tbl.quotation_id = q.quotation_id AND do_tbl.source_type = 'quotation')
        )
        ORDER BY q.quote_date DESC
        """
    ).format(subquery=PRODUCT_LIST_SUBQUERY)
    return fetchall_df(query)


def render_admin_filters() -> None:
    st.header("Advanced filters")
    st.caption("Explore all sales records with multi-dimensional filters.")
    df = load_admin_dataset()
    if df.empty:
        st.info("No data available yet.")
        return

    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df["follow_up_date"] = pd.to_datetime(df["follow_up_date"], errors="coerce")
    df["work_order_date"] = pd.to_datetime(df["work_order_date"], errors="coerce")
    df["delivery_order_date"] = pd.to_datetime(df["delivery_order_date"], errors="coerce")
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    with st.expander("Filters", expanded=True):
        first_row = st.columns(3)
        status_filter = first_row[0].multiselect(
            "Status",
            sorted(df["status"].dropna().unique().tolist()),
        )
        quote_type_filter = first_row[1].multiselect(
            "Quote type",
            sorted(df["quote_type"].dropna().unique().tolist()),
        )
        product_filter = first_row[2].multiselect(
            "Product",
            sorted(df["product"].dropna().unique().tolist()),
        )

        second_row = st.columns(3)
        district_filter = second_row[0].multiselect(
            "District",
            sorted(df["district"].dropna().unique().tolist()),
        )
        salesperson_filter = second_row[1].multiselect(
            "Salesperson",
            sorted(df["salesperson"].dropna().unique().tolist()),
        )
        company_filter = second_row[2].multiselect(
            "Company",
            sorted(df["company"].dropna().unique().tolist()),
        )

        third_row = st.columns(3)
        payment_state = third_row[0].selectbox(
            "Payment status",
            ["All", "Received", "Outstanding"],
        )
        work_order_state = third_row[1].selectbox(
            "Work order",
            ["All", "With work order", "Without work order"],
        )
        source_filter = third_row[2].multiselect(
            "Delivery order source",
            sorted(df["source_type"].dropna().unique().tolist()),
        )

        fourth_row = st.columns(3)
        quantity_range = None
        include_missing_quantity = True
        quantity_values = df["quantity"].dropna()
        if not quantity_values.empty:
            q_min, q_max = int(quantity_values.min()), int(quantity_values.max())
            if q_min < q_max:
                quantity_range = fourth_row[0].slider(
                    "Quantity range",
                    min_value=q_min,
                    max_value=q_max,
                    value=(q_min, q_max),
                )
            else:
                quantity_range = (q_min, q_max)
                fourth_row[0].info(
                    f"All records currently have a quantity of {q_min}."
                )
            include_missing_quantity = fourth_row[0].checkbox(
                "Include rows without quantity",
                value=True,
            )
        else:
            fourth_row[0].info("No quantity data available yet.")
        quote_start = fourth_row[1].date_input("Quote start date", value=None)
        quote_end = fourth_row[2].date_input("Quote end date", value=None)

        follow_row = st.columns(2)
        follow_start = follow_row[0].date_input("Follow-up start", value=None)
        follow_end = follow_row[1].date_input("Follow-up end", value=None)

        search_text = st.text_input(
            "Search text",
            help="Filter by company name, notes, delivery order number or third-party name.",
        )

    filtered = df.copy()

    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]
    if quote_type_filter:
        filtered = filtered[filtered["quote_type"].isin(quote_type_filter)]
    if district_filter:
        filtered = filtered[filtered["district"].isin(district_filter)]
    if product_filter:
        filtered = filtered[filtered["product"].isin(product_filter)]
    if salesperson_filter:
        filtered = filtered[filtered["salesperson"].isin(salesperson_filter)]
    if company_filter:
        filtered = filtered[filtered["company"].isin(company_filter)]
    if source_filter:
        filtered = filtered[filtered["source_type"].isin(source_filter)]
    if payment_state != "All":
        expected = 1 if payment_state == "Received" else 0
        filtered = filtered[filtered["payment_received"].fillna(0) == expected]
    if work_order_state != "All":
        if work_order_state == "With work order":
            filtered = filtered[filtered["work_order_id"].notna()]
        else:
            filtered = filtered[filtered["work_order_id"].isna()]
    if quantity_range:
        lower, upper = quantity_range
        mask = filtered["quantity"].between(lower, upper, inclusive="both")
        if include_missing_quantity:
            mask = mask | filtered["quantity"].isna()
        filtered = filtered[mask]
    if quote_start:
        filtered = filtered[filtered["quote_date"] >= pd.Timestamp(quote_start)]
    if quote_end:
        filtered = filtered[filtered["quote_date"] <= pd.Timestamp(quote_end)]
    if follow_start:
        filtered = filtered[
            filtered["follow_up_date"].notna()
            & (filtered["follow_up_date"] >= pd.Timestamp(follow_start))
        ]
    if follow_end:
        filtered = filtered[
            filtered["follow_up_date"].notna()
            & (filtered["follow_up_date"] <= pd.Timestamp(follow_end))
        ]
    if search_text:
        filtered = filtered[
            filtered["company"].str.contains(search_text, case=False, na=False)
            | filtered["notes"].fillna("").str.contains(search_text, case=False, na=False)
            | filtered["do_number"].fillna("").str.contains(search_text, case=False, na=False)
            | filtered["third_party_name"].fillna("").str.contains(search_text, case=False, na=False)
        ]

    summary_cols = st.columns(3)
    summary_cols[0].metric("Records", len(filtered))
    summary_cols[1].metric(
        "Total value",
        f"{filtered['price'].fillna(0).sum():,.2f}",
    )
    summary_cols[2].metric(
        "Outstanding",
        f"{filtered.loc[filtered['payment_received'] == 0, 'price'].fillna(0).sum():,.2f}",
    )

    display_df = filtered.copy()
    if not display_df.empty:
        try:
            display_df["quantity"] = display_df["quantity"].astype("Int64")
        except Exception:
            pass
        display_df["delivery_company"] = display_df.apply(
            lambda row: row["third_party_name"]
            if row.get("source_type") == "third_party"
            else row["company"],
            axis=1,
        )
    for column in [
        "quote_date",
        "follow_up_date",
        "work_order_date",
        "delivery_order_date",
        "payment_date",
    ]:
        display_df[column] = display_df[column].dt.date

    st.dataframe(display_df, use_container_width=True)

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_data, file_name="advanced_filter.csv")

    st.divider()
    st.subheader("Manage districts")
    with get_conn() as conn:
        districts = conn.execute(
            "SELECT district_id, name FROM districts ORDER BY name"
        ).fetchall()
    st.table(pd.DataFrame(districts))

    district_options = ["New district"] + [
        f"#{row['district_id']} – {row['name']}" for row in districts
    ]
    selection = st.selectbox("Select district", district_options)
    if selection == "New district":
        district_id = None
        current_name = ""
    else:
        district_id = int(selection.split(" – ")[0].lstrip("#"))
        current_name = next(row["name"] for row in districts if row["district_id"] == district_id)

    with st.form("district_form"):
        name = st.text_input("District name", value=current_name)
        submitted = st.form_submit_button("Save district")
    if submitted and name:
        upsert_district(district_id, name)
        st.success("District saved")
        rerun()


def render_companies() -> None:
    st.header("Companies")
    df = list_companies()
    st.dataframe(df, use_container_width=True)

    with get_conn() as conn:
        districts = conn.execute(
            "SELECT district_id, name FROM districts ORDER BY name"
        ).fetchall()
        products = conn.execute(
            "SELECT category_id, name FROM categories ORDER BY name"
        ).fetchall()
    products_by_id = {row["category_id"]: row["name"] for row in products}

    options = ["New company"] + [f"#{row['company_id']} – {row['name']}" for _, row in df.iterrows()]
    selection = st.selectbox("Select company", options)
    selected_company = None
    if selection != "New company":
        company_id = int(selection.split(" – ")[0].lstrip("#"))
        with get_conn() as conn:
            selected_company = conn.execute(
                "SELECT * FROM companies WHERE company_id=?", (company_id,)
            ).fetchone()
            selected_product_ids = get_company_products(company_id)
    else:
        selected_product_ids = []

    with st.form("company_form"):
        name = st.text_input(
            "Company name",
            value=selected_company["name"] if selected_company else "",
        )
        contact_person = st.text_input(
            "Contact person",
            value=(selected_company["contact_person"] or "") if selected_company else "",
        )
        phone = st.text_input(
            "Phone",
            value=(selected_company["phone"] or "") if selected_company else "",
        )
        address = st.text_area(
            "Company address",
            value=(selected_company["address"] or "") if selected_company else "",
        )
        delivery_address = st.text_area(
            "Delivery address",
            value=(selected_company["delivery_address"] or "") if selected_company else "",
        )
        district_choice = st.selectbox(
            "District",
            options=[(row["name"], row["district_id"]) for row in districts],
            format_func=lambda x: x[0],
            index=next(
                (
                    i
                    for i, row in enumerate(districts)
                    if selected_company and row["district_id"] == selected_company["district_id"]
                ),
                0,
            ),
        )
        district_id = district_choice[1]
        products_default = ", ".join(
            products_by_id[pid] for pid in selected_product_ids if pid in products_by_id
        )
        product_text = st.text_input(
            "Products",
            value=products_default,
            help="Describe the products this company sells (comma separated if needed).",
            placeholder="50 kVA generator",
        )
        submitted = st.form_submit_button("Save company")

    if submitted:
        data = {
            "company_id": selected_company["company_id"] if selected_company else None,
            "name": name.strip(),
            "contact_person": contact_person.strip() or None,
            "phone": phone.strip(),
            "address": address.strip(),
            "delivery_address": delivery_address.strip(),
            "district_id": district_id,
            "type": selected_company["type"] if selected_company else "retail",
        }
        product_names = parse_product_names(product_text)
        if not data["name"]:
            st.error("Company name is required")
        elif not data["phone"] or not data["address"] or not data["delivery_address"]:
            st.error("Phone, company address and delivery address are required")
        elif not product_names:
            st.error("Provide the products this company sells")
        else:
            product_ids = []
            for name_value in product_names:
                try:
                    product_identifier = ensure_product(name_value)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                product_ids.append(product_identifier)
            upsert_company(data, product_ids)
            st.success("Company saved")
            rerun()

    if selected_company and st.button("Delete company", type="secondary"):
        delete_company(selected_company["company_id"])
        st.success("Company deleted")
        rerun()


def render_users() -> None:
    st.header("Users")
    with get_conn() as conn:
        users = conn.execute(
            textwrap.dedent(
                """
                SELECT user_id, username, role, display_name, designation, phone, created_at
                FROM users
                ORDER BY username
                """
            )
        ).fetchall()
    st.table(pd.DataFrame(users))

    options = ["New user"] + [f"#{row['user_id']} – {row['username']}" for row in users]
    selection = st.selectbox("Select user", options)
    if selection == "New user":
        user_id = None
        username_default = ""
        role_default = "staff"
        display_name_default = ""
        designation_default = ""
        phone_default = ""
    else:
        user_id = int(selection.split(" – ")[0].lstrip("#"))
        selected_user = next(row for row in users if row["user_id"] == user_id)
        username_default = selected_user["username"]
        role_default = selected_user["role"]
        user_keys = selected_user.keys()
        display_name_default = (selected_user["display_name"] or "") if "display_name" in user_keys else ""
        designation_default = (selected_user["designation"] or "") if "designation" in user_keys else ""
        phone_default = (selected_user["phone"] or "") if "phone" in user_keys else ""

    with st.form("user_form"):
        username = st.text_input("Username", value=username_default)
        role = st.selectbox("Role", ["admin", "staff"], index=["admin", "staff"].index(role_default))
        display_name = st.text_input(
            "Full name",
            value=display_name_default,
            help="Shown as the salesperson name on quotations.",
        )
        designation = st.text_input(
            "Attention title",
            value=designation_default,
            help="Appears beneath the salesperson name in the quotation signature block.",
        )
        phone = st.text_input(
            "Phone number",
            value=phone_default,
            help="Included in the salesperson signature so customers can reach them.",
        )
        password = st.text_input(
            "Password", type="password", help="Leave blank to keep current password"
        )
        submitted = st.form_submit_button("Save user")
    if submitted and username:
        display_name_value = display_name.strip() if display_name else ""
        designation_value = designation.strip() if designation else ""
        phone_value = phone.strip() if phone else ""
        if role == "staff":
            if not display_name_value:
                st.error("Full name is required for sales staff.")
                return
            if not designation_value:
                st.error("Attention title is required for sales staff.")
                return
            if not phone_value:
                st.error("Phone number is required for sales staff.")
                return
        with get_cursor() as cur:
            if user_id:
                cur.execute(
                    textwrap.dedent(
                        """
                        UPDATE users
                           SET username=?,
                               role=?,
                               display_name=?,
                               designation=?,
                               phone=?
                         WHERE user_id=?
                        """
                    ),
                    (
                        username,
                        role,
                        display_name_value or username,
                        designation_value or None,
                        phone_value or None,
                        user_id,
                    ),
                )
                if password:
                    cur.execute(
                        "UPDATE users SET pass_hash=? WHERE user_id=?",
                        (hash_password(password), user_id),
                    )
            else:
                cur.execute(
                    textwrap.dedent(
                        """
                        INSERT INTO users(username, pass_hash, role, display_name, designation, phone)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """
                    ),
                    (
                        username,
                        hash_password(password or "changeme123"),
                        role,
                        display_name_value or username,
                        designation_value or None,
                        phone_value or None,
                    ),
                )
        st.success("User saved")
        rerun()

    if selection != "New user" and st.button("Reset password", type="secondary"):
        new_password = "Password123"
        with get_cursor() as cur:
            cur.execute(
                "UPDATE users SET pass_hash=? WHERE user_id=?",
                (hash_password(new_password), user_id),
            )
        st.info(f"Password reset to {new_password}")


def render_notifications(user: Dict) -> None:
    st.header("Notifications")
    if user["role"] == "admin":
        df = fetchall_df(
            "SELECT n.*, COALESCE(u.display_name, u.username) AS salesperson"
            " FROM notifications n"
            " JOIN users u ON u.user_id = n.user_id"
            " ORDER BY n.due_date"
        )
    else:
        df = fetchall_df(
            "SELECT n.*, COALESCE(u.display_name, u.username) AS salesperson"
            " FROM notifications n"
            " JOIN users u ON u.user_id = n.user_id"
            " WHERE n.user_id=?"
            " ORDER BY n.due_date",
            (user["user_id"],),
        )
    if df.empty:
        st.info("No notifications")
        return
    st.dataframe(df, use_container_width=True)
    unread = df[df["read"] == 0]
    if not unread.empty:
        to_mark = st.multiselect(
            "Mark as read", unread["notification_id"].tolist(), format_func=lambda x: f"#{x}"
        )
        if st.button("Update notifications") and to_mark:
            for notif_id in to_mark:
                mark_notification_read(int(notif_id))
            st.success("Notifications updated")
            rerun()


def render_settings() -> None:
    st.header("Settings")
    settings = get_settings()
    with st.form("settings_form"):
        work_order_grace = st.number_input(
            "Days before work-order reminder", min_value=1, value=settings.get("work_order_grace_days", 7)
        )
        delivery_order_grace = st.number_input(
            "Days before delivery-order reminder", min_value=1, value=settings.get("delivery_order_grace_days", 7)
        )
        payment_due_days = st.number_input(
            "Payment due days", min_value=1, value=settings.get("payment_due_days", 14)
        )
        submitted = st.form_submit_button("Save settings")
    if submitted:
        update_setting("work_order_grace_days", int(work_order_grace))
        update_setting("delivery_order_grace_days", int(delivery_order_grace))
        update_setting("payment_due_days", int(payment_due_days))
        st.success("Settings updated")


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="PS Business Suites by ZAD", layout="wide")
    if "user" not in st.session_state:
        login_screen()
        return

    user = st.session_state["user"]
    page = sidebar(user)

    if page == "dashboard":
        render_dashboard(user)
    elif page == "quotation_letters":
        render_quotation_letter_page(user)
    elif page == "quotations":
        render_quotations(user)
    elif page == "work_orders":
        render_work_orders(user)
    elif page == "delivery_orders":
        render_delivery_orders(user)
    elif page == "companies":
        render_companies()
    elif page == "admin_filters":
        render_admin_filters()
    elif page == "users":
        render_users()
    elif page == "notifications":
        render_notifications(user)
    elif page == "settings":
        render_settings()


if __name__ == "__main__":
    init_db()
    main()
