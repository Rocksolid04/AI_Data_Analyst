
import re
from pathlib import Path

import pandas as pd

from ai_analtytics_models.services.analytics import load_dataset


def load_dataset(file_path: Path, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(file_path)

    if file_type == "xlsx":
        return pd.read_excel(file_path)

    raise ValueError("Unsupported dataset type.")


def answer_question(
    dataframe: pd.DataFrame,
    question: str,
) -> dict:
    question_lower = question.lower().strip()

    if not question_lower:
        return {"answer": "Please provide a question."}

    columns = list(dataframe.columns)

    # 1. Total rows
    if re.search(r"\b(how many|count|number of)\s+(rows|records)\b", question_lower):
        return {
            "question": question,
            "answer": f"The dataset contains {len(dataframe)} rows.",
            "value": len(dataframe),
        }

    # 2. Total columns
    if re.search(r"\b(how many|count|number of)\s+columns\b", question_lower):
        return {
            "question": question,
            "answer": f"The dataset contains {len(columns)} columns.",
            "value": len(columns),
        }

    # 3. Missing values
    if any(word in question_lower for word in ["missing", "null", "empty values"]):
        missing = dataframe.isna().sum().to_dict()
        return {
            "question": question,
            "answer": "Missing-value counts by column.",
            "result": {str(k): int(v) for k, v in missing.items()},
        }

    # 4. Most common values in a column
    if any(word in question_lower for word in ["most common", "value counts", "distribution"]):
        column = next(
            (c for c in columns if str(c).lower() in question_lower),
            None,
        )

        if column is None:
            return {
                "question": question,
                "answer": "I couldn't identify a column.",
                "available_columns": [str(c) for c in columns],
            }

        counts = dataframe[column].value_counts(dropna=False).head(10)
        return {
            "question": question,
            "answer": f"Most common values in '{column}'.",
            "result": {
                str(k): int(v)
                for k, v in counts.items()
            },
        }

    # 5. Numeric aggregations
    aggregation_patterns = {
        "average": ["average", "mean", "avg"],
        "sum": ["sum", "total"],
        "maximum": ["maximum", "max", "highest", "largest"],
        "minimum": ["minimum", "min", "lowest", "smallest"],
        "median": ["median"],
        "standard_deviation": ["standard deviation", "std deviation", "stddev"],
    }

    for aggregation, keywords in aggregation_patterns.items():
        if not any(keyword in question_lower for keyword in keywords):
            continue

        column = next(
            (c for c in columns if str(c).lower() in question_lower),
            None,
        )

        if column is None:
            return {
                "question": question,
                "answer": "I couldn't identify a column in your question.",
                "available_columns": [str(c) for c in columns],
            }

        series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            return {
                "question": question,
                "answer": f"Column '{column}' has no numeric values to analyze.",
            }

        operations = {
            "average": series.mean,
            "sum": series.sum,
            "maximum": series.max,
            "minimum": series.min,
            "median": series.median,
            "standard_deviation": series.std,
        }

        value = float(operations[aggregation]())

        return {
            "question": question,
            "answer": f"The {aggregation.replace('_', ' ')} of '{column}' is {value:.4f}.",
            "column": str(column),
            "operation": aggregation,
            "value": value,
            "numeric_values_analyzed": int(len(series)),
        }

    return {
        "question": question,
        "answer": (
            "I can't answer that question yet. Try asking for row count, "
            "missing values, most common values, average, sum, maximum, "
            "minimum, median, or standard deviation."
        ),
        "available_columns": [str(c) for c in columns],
    }
