import io
import pandas as pd


def test_suggest_report_column_mapping_matches_labels(app_module):
    columns = ["Customer Name", "Reported Complaints", "Extra", "Bill TK"]

    suggestions = app_module._suggest_report_column_mapping(columns)

    assert suggestions["customer_name"] == "Customer Name"
    assert suggestions["reported_complaints"] == "Reported Complaints"
    assert suggestions["bill_tk"] == "Bill TK"
    assert "extra" not in suggestions.values()


def test_import_report_grid_from_dataframe_auto_mapping(app_module):
    df = pd.DataFrame(
        {
            "Customer Name": ["Acme"],
            "Bill TK": ["105.50"],
            "Work Done Date": ["01-02-2024"],
        }
    )

    rows = app_module._import_report_grid_from_dataframe(df)

    assert len(rows) == 1
    row = rows[0]
    assert row["customer_name"] == "Acme"
    assert row["bill_tk"] == 105.50
    assert row["work_done_date"] == "2024-02-01"


def test_import_report_grid_respects_custom_mapping(app_module):
    df = pd.DataFrame(
        {
            "Person": ["Nora"],
            "Amount": [500],
        }
    )
    mapping = {"Person": "customer_name", "Amount": "bill_tk"}

    rows = app_module._import_report_grid_from_dataframe(df, mapping)

    assert len(rows) == 1
    row = rows[0]
    assert row["customer_name"] == "Nora"
    assert row["bill_tk"] == 500
    # unmapped fields should fall back to defaults
    assert row["work_done_date"] == ""


def test_load_report_grid_dataframe_reads_csv_and_excel(app_module):
    df = pd.DataFrame({"Customer Name": ["Acme"], "Bill TK": [10]})

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_bytes = excel_buffer.getvalue()

    csv_df = app_module._load_report_grid_dataframe(csv_bytes, "sample.csv")
    excel_df = app_module._load_report_grid_dataframe(excel_bytes, "sample.xlsx")

    assert list(csv_df.columns) == ["Customer Name", "Bill TK"]
    assert list(excel_df.columns) == ["Customer Name", "Bill TK"]
