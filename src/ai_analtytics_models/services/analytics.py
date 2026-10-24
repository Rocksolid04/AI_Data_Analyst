
from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path, file_type: str) -> pd.DataFrame:
    """Load a supported dataset file into a Pandas DataFrame."""
    normalized_type = file_type.lower().lstrip(".")

    if normalized_type == "csv":
        return pd.read_csv(file_path)

    if normalized_type in {"xlsx", "xls"}:
        return pd.read_excel(file_path)

    raise ValueError(
        f"Unsupported dataset type '{file_type}'. "
        "Supported types are CSV and Excel."
    )
