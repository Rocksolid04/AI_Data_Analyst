
from typing import Any

import pandas as pd


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build a basic statistical profile of a DataFrame."""
    if len(dataframe.columns) == 0:
        raise ValueError("The dataset contains no columns to profile.")

    columns = []

    for column in dataframe.columns:
        series = dataframe[column]
        profile: dict[str, Any] = {
            "name": str(column),
            "data_type": str(series.dtype),
            "row_count": len(series),
            "null_count": int(series.isna().sum()),
            "null_percentage": (
                round(float(series.isna().mean() * 100), 2)
                if len(series)
                else 0.0
            ),
            "distinct_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            profile.update({
                "min": _json_safe(numeric.min()),
                "max": _json_safe(numeric.max()),
                "mean": _json_safe(numeric.mean()),
                "median": _json_safe(numeric.median()),
            })

        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update({
                "min": _json_safe(series.min()),
                "max": _json_safe(series.max()),
            })

        columns.append(profile)

    return {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "columns": columns,
    }


def _json_safe(value: Any) -> Any:
    """Convert Pandas and NumPy values into JSON-compatible values."""
    if value is None or pd.isna(value):
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)
