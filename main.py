"""Unified Streamlit entry point for both applications.

Set the ``PS_APP`` environment variable to choose which experience to
start when deploying (``crm`` for the Business Suites CRM, ``sales`` for
PS Sales Manager). When unset, the CRM experience is launched for backwards
compatibility.
"""
from __future__ import annotations

import os
import streamlit as st


def _target() -> str:
    value = os.getenv("PS_APP", "crm").strip().lower()
    if value in {"sales", "sales_app", "ps_sales"}:
        return "sales"
    return "crm"


def main() -> None:
    target = _target()
    if target == "sales":
        import sales_app

        sales_app.init_db()
        sales_app.main()
    else:
        import app as crm_app

        crm_app.main()


if __name__ == "__main__":
    main()
