from __future__ import annotations

import pytest


def test_normalize_quotation_items_calculates_totals(app_module):
    items, totals = app_module.normalize_quotation_items(
        [
            {
                "description": "Generator maintenance",
                "hsn": "8407",
                "unit": "nos",
                "quantity": 2,
                "rate": 25000,
                "discount": 10,
            }
        ]
    )

    assert len(items) == 1
    item = items[0]
    assert pytest.approx(item["Gross amount"], rel=1e-6) == 50000
    assert pytest.approx(item["Discount amount"], rel=1e-6) == 5000
    line_total = 45000
    assert pytest.approx(item["Line total"], rel=1e-6) == line_total
    assert "CGST amount" not in item
    assert "SGST amount" not in item
    assert "IGST amount" not in item

    assert pytest.approx(totals["gross_total"], rel=1e-6) == 50000
    assert pytest.approx(totals["discount_total"], rel=1e-6) == 5000
    assert pytest.approx(totals["grand_total"], rel=1e-6) == line_total


def test_normalize_quotation_items_skips_blank_descriptions(app_module):
    items, totals = app_module.normalize_quotation_items(
        [
            {"description": " ", "quantity": 1, "rate": 100},
            {"description": "Valid", "quantity": 1, "rate": 100},
        ]
    )

    assert len(items) == 1
    assert items[0]["Description"] == "Valid"
    assert pytest.approx(totals["grand_total"], rel=1e-6) == 100


def test_format_amount_in_words_returns_phrase(app_module):
    assert app_module.format_amount_in_words(1250) == "One thousand two hundred and fifty taka"
    assert (
        app_module.format_amount_in_words(1250.75)
        == "One thousand two hundred and fifty taka and seventy five paisa"
    )
    assert app_module.format_amount_in_words(None) is None
