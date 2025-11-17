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
                "cgst": 9,
                "sgst": 9,
                "igst": 0,
            }
        ]
    )

    assert len(items) == 1
    item = items[0]
    assert pytest.approx(item["Gross amount"], rel=1e-6) == 50000
    assert pytest.approx(item["Discount amount"], rel=1e-6) == 5000
    taxable_value = 45000
    assert pytest.approx(item["Taxable value"], rel=1e-6) == taxable_value
    expected_tax = taxable_value * 0.09
    assert pytest.approx(item["CGST amount"], rel=1e-6) == expected_tax
    assert pytest.approx(item["SGST amount"], rel=1e-6) == expected_tax
    assert pytest.approx(item["IGST amount"], rel=1e-6) == 0
    line_total = taxable_value + expected_tax * 2
    assert pytest.approx(item["Line total"], rel=1e-6) == line_total

    assert pytest.approx(totals["gross_total"], rel=1e-6) == 50000
    assert pytest.approx(totals["discount_total"], rel=1e-6) == 5000
    assert pytest.approx(totals["taxable_total"], rel=1e-6) == taxable_value
    assert pytest.approx(totals["cgst_total"], rel=1e-6) == expected_tax
    assert pytest.approx(totals["sgst_total"], rel=1e-6) == expected_tax
    assert pytest.approx(totals["igst_total"], rel=1e-6) == 0
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
