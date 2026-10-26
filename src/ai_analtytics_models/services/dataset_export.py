
import csv
import io
from typing import Any

import pandas as pd


def _sanitize_cell(value: Any) -> Any:
    """Reduce spreadsheet formula-injection risk for string cells."""
    if isinstance(value, str) and value.startswith(
        ("=", "+", "-", "@", "\t", "\r", "\n")
    ):
        return "'" + value

    return value


def export_dataframe_to_csv(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Export a DataFrame to CSV with basic formula-injection protection."""
    if len(dataframe.columns) == 0:
        raise ValueError("The dataset contains no columns to export.")

    safe_dataframe = dataframe.copy()

    for column in safe_dataframe.columns:
        safe_dataframe[column] = safe_dataframe[column].map(_sanitize_cell)

    output = io.StringIO(newline="")
    safe_dataframe.to_csv(
        output,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    return {
        "file_content": output.getvalue(),
        "row_count": len(safe_dataframe),
        "column_count": len(safe_dataframe.columns),
        "media_type": "text/csv; charset=utf-8",
        "extension": "csv",
    }


def export_dataframe_to_excel(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Export a DataFrame to an Excel workbook."""
    if len(dataframe.columns) == 0:
        raise ValueError("The dataset contains no columns to export.")

    safe_dataframe = dataframe.copy()

    for column in safe_dataframe.columns:
        safe_dataframe[column] = safe_dataframe[column].map(_sanitize_cell)

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        safe_dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Data",
        )

    return {
        "file_content": output.getvalue(),
        "row_count": len(safe_dataframe),
        "column_count": len(safe_dataframe.columns),
        "media_type": (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        "extension": "xlsx",
    }
