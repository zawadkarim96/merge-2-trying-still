from datetime import date


def test_month_bucket_counts_tracks_current_and_previous_month(db_conn, app_module):
    today = date.today()
    this_month = date(today.year, today.month, 15)
    last_month = app_module.add_months(this_month, -1)

    cur = db_conn.cursor()
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-C1", this_month.isoformat(), this_month.isoformat(), "active"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-C2", this_month.isoformat(), this_month.isoformat(), "active"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-P1", last_month.isoformat(), last_month.isoformat(), "active"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-INACTIVE", last_month.isoformat(), last_month.isoformat(), "inactive"),
    )
    db_conn.commit()

    current, previous = app_module.month_bucket_counts(
        db_conn,
        "warranties",
        "expiry_date",
        where="status=?",
        params=("active",),
    )

    assert current == 2
    assert previous == 1


def test_format_metric_delta_provides_context(db_conn, app_module):
    assert app_module.format_metric_delta(0, 0) == "On par with last month"
    assert app_module.format_metric_delta(4, 0) == "+4 (new this month)"
    assert app_module.format_metric_delta(5, 2) == "+3 (+150.0%) vs last month"


def test_upcoming_warranty_projection_returns_contiguous_months(db_conn, app_module):
    today = date.today()
    base_month = date(today.year, today.month, 15)
    next_month = app_module.add_months(base_month, 1)
    later_month = app_module.add_months(base_month, 2)

    cur = db_conn.cursor()
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-NOW", base_month.isoformat(), base_month.isoformat(), "active"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-NEXT", next_month.isoformat(), next_month.isoformat(), "active"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-LATER", later_month.isoformat(), later_month.isoformat(), "active"),
    )
    # Outside of the requested window; should be excluded.
    far_future = app_module.add_months(base_month, 6)
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (None, None, "SER-FAR", far_future.isoformat(), far_future.isoformat(), "active"),
    )
    db_conn.commit()

    projection = app_module.upcoming_warranty_projection(db_conn, months_ahead=3)

    assert list(projection.columns) == ["Month", "Expiring warranties"]
    assert len(projection) == 3

    expected_labels = [
        date(today.year, today.month, 1).strftime("%b %Y"),
        app_module.add_months(date(today.year, today.month, 1), 1).strftime("%b %Y"),
        app_module.add_months(date(today.year, today.month, 1), 2).strftime("%b %Y"),
    ]
    assert list(projection["Month"]) == expected_labels
    assert list(projection["Expiring warranties"]) == [1, 1, 1]

    single_month = app_module.upcoming_warranty_projection(db_conn, months_ahead=0)
    assert len(single_month) == 1


def test_upcoming_warranty_breakdown_groups_by_dimension(db_conn, app_module):
    today = date.today()
    soon = today + app_module.timedelta(days=10)
    later = today + app_module.timedelta(days=20)

    cur = db_conn.cursor()
    cur.execute(
        "INSERT INTO customers (name, phone, address, sales_person) VALUES (?, ?, ?, ?)",
        ("Acme", "111", "Main", "Riya"),
    )
    c1 = cur.lastrowid
    cur.execute(
        "INSERT INTO customers (name, phone, address, sales_person) VALUES (?, ?, ?, ?)",
        ("Beta", "222", "Second", "Ayan"),
    )
    c2 = cur.lastrowid
    cur.execute(
        "INSERT INTO products (name, model, serial) VALUES (?, ?, ?)",
        ("GenX", "1000", "GEN-1"),
    )
    p1 = cur.lastrowid
    cur.execute(
        "INSERT INTO products (name, model, serial) VALUES (?, ?, ?)",
        ("GenX", "2000", "GEN-2"),
    )
    p2 = cur.lastrowid

    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (c1, p1, "SER-A", today.isoformat(), soon.isoformat(), "active"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (c1, p2, "SER-B", today.isoformat(), later.isoformat(), "active"),
    )
    cur.execute(
        "INSERT INTO warranties (customer_id, product_id, serial, issue_date, expiry_date, status) VALUES (?, ?, ?, ?, ?, ?)",
        (c2, p2, "SER-C", today.isoformat(), later.isoformat(), "active"),
    )
    db_conn.commit()

    sales_breakdown = app_module.upcoming_warranty_breakdown(db_conn, days_ahead=30, group_by="sales_person")
    assert list(sales_breakdown.columns) == ["Sales person", "Expiring warranties"]
    assert sales_breakdown.iloc[0]["Sales person"] == "Riya"
    assert int(sales_breakdown.iloc[0]["Expiring warranties"]) == 2

    product_breakdown = app_module.upcoming_warranty_breakdown(db_conn, days_ahead=30, group_by="product")
    assert product_breakdown.iloc[0]["Product"].startswith("GenX")
    assert product_breakdown.iloc[0]["Expiring warranties"] == 2

    fallback_breakdown = app_module.upcoming_warranty_breakdown(db_conn, days_ahead=30, group_by="unknown")
    assert list(fallback_breakdown.columns)[0] == "Sales person"
