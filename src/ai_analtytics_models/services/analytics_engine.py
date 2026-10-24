
from __future__ import annotations

from typing import Any

import math
import pandas as pd
import numpy as np


AGGREGATIONS = {"sum", "mean", "min", "max", "median", "count", "std"}

FILTER_OPERATORS = {
    "eq", "ne", "gt", "gte", "lt", "lte",
    "contains", "in", "is_null", "not_null", "between",
}


def _json_value(value: Any) -> Any:
    """Convert Pandas and NumPy scalar values to JSON-safe Python values."""
    if value is None:
        return None

    # Handle missing values without assuming every input is a scalar.
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
        if getattr(missing, "ndim", 0) == 0 and bool(missing):
            return None
    except (TypeError, ValueError):
        pass

    # Convert NumPy scalar types to native Python types.
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    if value is None or isinstance(value, (str, bool, int)):
       return value

    if isinstance(value, float):
       return value if math.isfinite(value) else None

    # Convert dates and timestamps to ISO 8601 strings.
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (ValueError, TypeError):
            pass

    # Safe fallback for other scalar-like objects.
    return str(value)



def _validate_column(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise ValueError(
            f"Unknown column '{column}'. "
            f"Available columns: {[str(c) for c in dataframe.columns]}"
        )



def _apply_filters(
    dataframe: pd.DataFrame,
    filters: list[dict] | None = None,
    combine_with: str = "and",
) -> pd.DataFrame:
    """Apply validated filter conditions and return a filtered copy."""
    import numpy as np

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The input must be a Pandas DataFrame.")

    if combine_with not in {"and", "or"}:
        raise ValueError("combine_with must be 'and' or 'or'.")

    if filters is None:
        filters = []

    if not isinstance(filters, list):
        raise ValueError("filters must be a list of filter conditions.")

    if len(filters) > 10:
        raise ValueError("A maximum of 10 filter conditions is allowed.")

    if not filters:
        return dataframe.copy()

    masks = []

    for index, condition in enumerate(filters):
        if not isinstance(condition, dict):
            raise ValueError(
                f"Filter {index + 1} must be an object."
            )

        column = condition.get("column")
        operator = condition.get("operator")

        if not isinstance(column, str) or not column:
            raise ValueError(
                f"Filter {index + 1} requires a valid column name."
            )

        _validate_column(dataframe, column)

        if operator not in FILTER_OPERATORS:
            raise ValueError(
                f"Unsupported filter operator '{operator}'. "
                f"Choose from {sorted(FILTER_OPERATORS)}."
            )

        series = dataframe[column]

        # Null checks don't require a value.
        if operator == "is_null":
            mask = series.isna()

        elif operator == "not_null":
            mask = series.notna()

        else:
            if "value" not in condition:
                raise ValueError(
                    f"Filter '{operator}' on '{column}' requires a value."
                )

            value = condition["value"]

            if operator == "contains":
                if not isinstance(value, str):
                    raise ValueError(
                        "The 'contains' filter requires a string value."
                    )

                mask = series.astype("string").str.contains(
                    value,
                    case=False,
                    regex=False,
                    na=False,
                )

            elif operator == "in":
                if (
                    not isinstance(value, list)
                    or not value
                    or len(value) > 100
                ):
                    raise ValueError(
                        "The 'in' filter requires a non-empty list "
                        "of at most 100 values."
                    )

                mask = series.isin(value).fillna(False)

            elif operator == "between":
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(
                        "The 'between' filter requires "
                        "a two-item list: [minimum, maximum]."
                    )

                bounds = pd.to_numeric(
                    pd.Series(value), errors="coerce"
                )

                if bounds.isna().any() or not np.isfinite(
                    bounds.to_numpy(dtype=float)
                ).all():
                    raise ValueError(
                        "The 'between' bounds must be finite numbers."
                    )

                lower, upper = bounds.tolist()

                if lower > upper:
                    raise ValueError(
                        "The lower 'between' bound cannot exceed the upper bound."
                    )

                numeric_series = pd.to_numeric(
                    series, errors="coerce"
                )
                mask = numeric_series.between(
                    lower, upper, inclusive="both"
                )

            elif operator in {"gt", "gte", "lt", "lte"}:
                if isinstance(value, bool):
                    raise ValueError(
                        f"The '{operator}' filter requires a numeric value."
                    )

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"The '{operator}' filter requires a numeric value."
                    ) from None

                if not np.isfinite(numeric_value):
                    raise ValueError(
                        f"The '{operator}' filter requires a finite number."
                    )

                numeric_series = pd.to_numeric(
                    series, errors="coerce"
                )

                if operator == "gt":
                    mask = numeric_series > numeric_value
                elif operator == "gte":
                    mask = numeric_series >= numeric_value
                elif operator == "lt":
                    mask = numeric_series < numeric_value
                else:
                    mask = numeric_series <= numeric_value

            elif operator in {"eq", "ne"}:
                # Compare numerically when the column is numeric or
                # the supplied value is a number; otherwise compare
                # values directly.
                numeric_column = pd.api.types.is_numeric_dtype(
                    series.dtype
                )
                numeric_value = (
                    isinstance(value, (int, float, np.number))
                    and not isinstance(value, (bool, np.bool_))
                )

                if numeric_column or numeric_value:
                    try:
                        comparison_value = float(value)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"The '{operator}' filter value is incompatible "
                            f"with column '{column}'."
                        ) from None

                    if not np.isfinite(comparison_value):
                        raise ValueError(
                            f"The '{operator}' filter requires a finite number."
                        )

                    comparable = pd.to_numeric(
                        series, errors="coerce"
                    )

                    mask = (
                        comparable.eq(comparison_value)
                        if operator == "eq"
                        else comparable.ne(comparison_value)
                    )
                else:
                    mask = (
                        series.eq(value)
                        if operator == "eq"
                        else series.ne(value)
                    )

                # Missing values should not accidentally match "ne".
                mask = mask & series.notna()
                # Numeric comparisons must also exclude values that failed conversion.
                if numeric_column or numeric_value:
                    mask = mask & comparable.notna()

        masks.append(mask.fillna(False).astype(bool))

    combined_mask = masks[0]

    for mask in masks[1:]:
        if combine_with == "and":
            combined_mask = combined_mask & mask
        else:
            combined_mask = combined_mask | mask

    return dataframe.loc[combined_mask].copy()



def _aggregate(
    series: pd.Series,
    aggregation: str,
) -> Any:
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation '{aggregation}'. "
            f"Allowed aggregations: {sorted(AGGREGATIONS)}"
        )

    if aggregation == "count":
        return int(series.count())

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    ) 

    numeric_series = numeric_series[np.isfinite(numeric_series)]

    if numeric_series.empty:
        return None

    result = getattr(numeric_series, aggregation)()

    return _json_value(result)


def run_analysis(
    dataframe: pd.DataFrame,
    operation: str,
    column: str | None = None,
    columns: list[str] | None = None,
    aggregation: str = "mean",
    filters: list[dict] | None = None,
    combine_with: str = "and",
    group_by: list[str] | None = None,
    method: str = "pearson",
    top_n: int = 10,
) -> dict:
    """Execute a validated, deterministic dataset analysis."""

    supported_operations = {
        "summary",
        "missing_values",
        "preview",
        "aggregate",
        "group_by",
        "distribution",
        "correlation",
    }

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The input must be a Pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if operation not in supported_operations:
        raise ValueError(
            f"Unsupported operation '{operation}'. "
            f"Choose from {sorted(supported_operations)}."
        )

    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or not 1 <= top_n <= 100
    ):
        raise ValueError("top_n must be an integer between 1 and 100.")

    filtered = _apply_filters(
        dataframe=dataframe,
        filters=filters,
        combine_with=combine_with,
    )

    if filtered.empty:
        raise ValueError("No rows remain after applying the filters.")

    # SUMMARY
    if operation == "summary":
        columns_summary = {}

        for name in filtered.columns:
            series = filtered[name]

            columns_summary[str(name)] = {
                "dtype": str(series.dtype),
                "missing_values": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
            }

        return {
            "operation": operation,
            "row_count": int(len(filtered)),
            "column_count": int(len(filtered.columns)),
            "duplicate_rows": int(filtered.duplicated().sum()),
            "columns": columns_summary,
        }

    # MISSING VALUES
    if operation == "missing_values":
        missing = filtered.isna().sum()

        results = [
            {
                "column": str(name),
                "missing_count": int(count),
                "missing_percent": round(
                    float(count / len(filtered) * 100), 2
                ),
            }
            for name, count in missing.items()
        ]

        results.sort(
            key=lambda item: item["missing_count"],
            reverse=True,
        )

        return {
            "operation": operation,
            "row_count": int(len(filtered)),
            "columns": results[:top_n],
            "truncated": len(results) > top_n,
        }

    # PREVIEW
    if operation == "preview":
        preview = filtered.head(top_n).copy()
        records = [
            {
                str(key): _json_value(value)
                for key, value in record.items()
            }
            for record in preview.to_dict(orient="records")
        ]

        return {
            "operation": operation,
            "columns": [str(name) for name in filtered.columns],
            "rows": records,
            "returned_rows": len(records),
            "total_rows": int(len(filtered)),
            "truncated": len(filtered) > top_n,
        }

    # AGGREGATE
    if operation == "aggregate":
        if column is None:
            raise ValueError("Aggregate requires a column.")

        _validate_column(filtered, column)

        value = _aggregate(filtered[column], aggregation)

        return {
            "operation": operation,
            "column": column,
            "aggregation": aggregation,
            "value": _json_value(value),
            "rows_analyzed": (
                int(filtered[column].notna().sum())
                if aggregation == "count"
                else int(
                    np.isfinite(
                        pd.to_numeric(
                            filtered[column],
                            errors="coerce",
                        )
                    ).sum()
                )
            ),
        }

    # GROUP BY
    if operation == "group_by":
        if not group_by:
            raise ValueError("Group by requires at least one grouping column.")

        if len(group_by) > 5:
            raise ValueError("Grouping is limited to five columns.")

        if len(group_by) != len(set(group_by)):
            raise ValueError("Grouping columns must not contain duplicates.")

        for name in group_by:
            _validate_column(filtered, name)

        if column is not None:
            _validate_column(filtered, column)

        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation. Choose from {sorted(AGGREGATIONS)}."
            )

        if aggregation == "count" and column is None:
            grouped = filtered.groupby(
                group_by, dropna=False, observed=True
            ).size()
        elif column is None:
            raise ValueError(
                "A value column is required unless aggregation is 'count'."
            )
        else:
            working = filtered.copy()

            if aggregation != "count":
                working[column] = pd.to_numeric(
                    working[column], errors="coerce"
                )

            grouped = working.groupby(
                group_by, dropna=False, observed=True
            )[column].agg(aggregation)

        grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna()

        if grouped.empty:
            raise ValueError("No valid grouped results could be calculated.")

        # Sort by the aggregate value for useful top results.
        grouped = grouped.sort_values(ascending=False)
        total_groups = len(grouped)
        grouped = grouped.head(top_n)

        records = []

        for key, value in grouped.items():
            key_values = key if isinstance(key, tuple) else (key,)

            record = {
                str(name): _json_value(key_value)
                for name, key_value in zip(group_by, key_values)
            }
            record["value"] = _json_value(value)
            records.append(record)

        return {
            "operation": operation,
            "group_by": group_by,
            "column": column,
            "aggregation": aggregation,
            "groups": records,
            "total_groups": total_groups,
            "truncated": total_groups > top_n,
        }

    # DISTRIBUTION
    if operation == "distribution":
        if column is None:
            raise ValueError("Distribution requires a column.")

        _validate_column(filtered, column)

        counts = filtered[column].value_counts(
            dropna=False,
            sort=True,
        )

        total_categories = len(counts)
        counts = counts.head(top_n)

        results = [
            {
                "value": _json_value(value),
                "count": int(count),
                "percent": round(
                    float(count / len(filtered) * 100), 2
                ),
            }
            for value, count in counts.items()
        ]

        return {
            "operation": operation,
            "column": column,
            "categories": results,
            "total_categories": total_categories,
            "truncated": total_categories > top_n,
        }

    # CORRELATION
    if operation == "correlation":
        if method not in {"pearson", "spearman", "kendall"}:
            raise ValueError(
                "method must be pearson, spearman, or kendall."
            )

        if columns is not None:
            if not isinstance(columns, list) or len(columns) < 2:
                raise ValueError(
                    "Select at least two columns for correlation."
                )

            if len(columns) != len(set(columns)):
                raise ValueError(
                    "Correlation columns must not contain duplicates."
                )

            for name in columns:
                _validate_column(filtered, name)

            numeric = filtered[columns].apply(
                pd.to_numeric, errors="coerce"
            )
        else:
            numeric = filtered.select_dtypes(include="number").copy()

        # Remove columns with no usable numeric data or no variation.
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.dropna(axis=1, how="all")
        numeric = numeric.loc[
            :, numeric.nunique(dropna=True) > 1
        ]

        if numeric.shape[1] < 2:
            raise ValueError(
                "At least two varying numeric columns are required "
                "for correlation."
            )

        if numeric.shape[1] > 30:
            raise ValueError(
                "Correlation is limited to 30 columns. "
                "Select a smaller set of columns."
            )

        matrix = numeric.corr(method=method)
        pairs = []

        names = list(matrix.columns)

        for index, first in enumerate(names):
            for second in names[index + 1:]:
                coefficient = matrix.loc[first, second]

                if pd.notna(coefficient) and np.isfinite(coefficient):
                    pairs.append(
                        {
                            "column_1": str(first),
                            "column_2": str(second),
                            "correlation": round(
                                float(coefficient), 4
                            ),
                        }
                    )

        pairs.sort(
            key=lambda item: abs(item["correlation"]),
            reverse=True,
        )

        return {
            "operation": operation,
            "method": method,
            "columns_analyzed": [str(name) for name in names],
            "relationships": pairs[:top_n],
            "total_relationships": len(pairs),
            "truncated": len(pairs) > top_n,
            "interpretation_note": (
                "Correlation measures association, not causation."
            ),
        }

    raise ValueError(f"Operation '{operation}' was not handled.")


def run_time_series_analysis(
    dataframe: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "sum",
    frequency: str = "monthly",
    filters: list[dict] | None = None,
    combine_with: str = "and",
    detect_anomalies: bool = True,
    anomaly_threshold: float = 3.5,
    top_n: int = 100,
) -> dict:
    """Analyze trends, growth, and anomalies over time."""
    import numpy as np

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The input must be a Pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if not isinstance(date_column, str) or not date_column:
        raise ValueError("date_column must be a valid column name.")

    if not isinstance(value_column, str) or not value_column:
        raise ValueError("value_column must be a valid column name.")

    _validate_column(dataframe, date_column)
    _validate_column(dataframe, value_column)

    if date_column == value_column:
        raise ValueError("Date and value columns must be different.")

    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation. Choose from {sorted(AGGREGATIONS)}."
        )

    frequencies = {
        "daily": "D",
        "weekly": "W-SUN",
        "monthly": "MS",
        "quarterly": "QS",
        "yearly": "YS",
    }

    if frequency not in frequencies:
        raise ValueError(
            f"Unsupported frequency. Choose from {sorted(frequencies)}."
        )

    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or not 1 <= top_n <= 500
    ):
        raise ValueError("top_n must be an integer between 1 and 500.")

    if not isinstance(detect_anomalies, bool):
        raise ValueError("detect_anomalies must be a boolean.")

    if (
        isinstance(anomaly_threshold, bool)
        or not isinstance(anomaly_threshold, (int, float, np.number))
        or not np.isfinite(anomaly_threshold)
        or anomaly_threshold <= 0
    ):
        raise ValueError(
            "anomaly_threshold must be a finite positive number."
        )

    filtered = _apply_filters(
        dataframe=dataframe,
        filters=filters,
        combine_with=combine_with,
    )

    if filtered.empty:
        raise ValueError("No rows remain after applying the filters.")

    # Parse dates robustly. Invalid dates are excluded and counted.
    parsed_dates = pd.to_datetime(
        filtered[date_column],
        errors="coerce",
        utc=True,
    )

    valid_dates = parsed_dates.notna()
    invalid_date_rows = int((~valid_dates).sum())

    working = filtered.loc[valid_dates, [value_column]].copy()
    working["_timestamp"] = parsed_dates.loc[valid_dates]

    if working.empty:
        raise ValueError(
            f"Column '{date_column}' contains no valid dates."
        )

    working["_value"] = pd.to_numeric(
        working[value_column], errors="coerce"
    )

    finite_values = np.isfinite(working["_value"])
    invalid_value_rows = int((~finite_values).sum())

    working = working.loc[finite_values].copy()

    if working.empty:
        raise ValueError(
            f"Column '{value_column}' contains no usable finite numeric values."
        )

    working = working.set_index("_timestamp").sort_index()
    rule = frequencies[frequency]

    resampler = working["_value"].resample(rule)

    if aggregation == "count":
        series = resampler.count()
    elif aggregation == "sum":
        series = resampler.sum(min_count=1)
    else:
        series = resampler.agg(aggregation)

    # Remove infinite aggregate results without filling missing periods.
    series = series.replace([np.inf, -np.inf], np.nan)

    if series.notna().sum() == 0:
        raise ValueError(
            "No valid time-series values could be calculated."
        )

    # Growth compares against the previous available period.
    previous = series.shift(1)
    growth = (series - previous) / previous.abs() * 100

    growth = growth.where(
        previous.notna()
        & series.notna()
        & previous.ne(0)
    )


    # Robust anomaly detection using median absolute deviation.
    anomaly_flags = pd.Series(False, index=series.index)
    anomaly_scores = pd.Series(np.nan, index=series.index)

    valid_series = series.dropna()

    if detect_anomalies and len(valid_series) >= 3:
        median = float(valid_series.median())
        deviations = (valid_series - median).abs()
        mad = float(deviations.median())

        if mad > 0:
            # Modified z-score is robust to extreme observations.
            scores = 0.67448975 * (valid_series - median) / mad
        else:
            # MAD can be zero when most values are identical.
            # Fall back to standard deviation-based scores.
            standard_deviation = float(valid_series.std(ddof=0))

            if (
                np.isfinite(standard_deviation)
                and standard_deviation > 0
            ):
                scores = (
                    valid_series - median
                ) / standard_deviation
            else:
                scores = pd.Series(
                    np.nan,
                    index=valid_series.index,
                )

        finite_scores = scores[np.isfinite(scores)]
        anomaly_scores.loc[finite_scores.index] = finite_scores
        anomaly_flags.loc[finite_scores.index] = (
            finite_scores.abs() > anomaly_threshold
        )


    # Return chronological periods, including missing periods as nulls.
    periods = []

    for timestamp, value in series.items():
        value_is_valid = pd.notna(value)
        growth_value = growth.loc[timestamp]
        score = anomaly_scores.loc[timestamp]

        periods.append(
            {
                "period": timestamp.isoformat(),
                "value": (
                    round(float(value), 6)
                    if value_is_valid
                    else None
                ),
                "growth_percent": (
                    round(float(growth_value), 2)
                    if pd.notna(growth_value)
                    and np.isfinite(growth_value)
                    else None
                ),
                "is_anomaly": bool(anomaly_flags.loc[timestamp]),
                "anomaly_score": (
                    round(float(score), 4)
                    if pd.notna(score) and np.isfinite(score)
                    else None
                ),
            }
        )

    anomalies = [
        period for period in periods if period["is_anomaly"]
    ]

    return {
        "operation": "time_series",
        "date_column": date_column,
        "value_column": value_column,
        "aggregation": aggregation,
        "frequency": frequency,
        "periods": periods[:top_n],
        "total_periods": len(periods),
        "truncated": len(periods) > top_n,
        "anomalies": anomalies[:top_n],
        "total_anomalies": len(anomalies),
        "invalid_date_rows": invalid_date_rows,
        "invalid_value_rows": invalid_value_rows,
        "anomaly_detection_enabled": detect_anomalies,
        "anomaly_threshold": float(anomaly_threshold),
        "interpretation_note": (
            "Detected anomalies are statistical outliers, not necessarily "
            "errors or business problems. Growth is undefined when the "
            "previous period is missing or zero."
        ),
    }

def create_chart_spec(
    dataframe: pd.DataFrame,
    chart_type: str,
    x_column: str | None = None,
    y_column: str | None = None,
    aggregation: str = "sum",
    filters: list[dict] | None = None,
    combine_with: str = "and",
    bins: int = 10,
    top_n: int = 30,
) -> dict:
    """Create a validated, frontend-ready chart specification."""
    import numpy as np

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("The input must be a Pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot create a chart from an empty dataset.")

    supported_types = {"bar", "line", "scatter", "histogram"}
    if chart_type not in supported_types:
        raise ValueError(
            f"Unsupported chart type. Choose from {sorted(supported_types)}."
        )

    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or not 1 <= top_n <= 100
    ):
        raise ValueError("top_n must be an integer between 1 and 100.")

    if (
        not isinstance(bins, int)
        or isinstance(bins, bool)
        or not 5 <= bins <= 50
    ):
        raise ValueError("bins must be an integer between 5 and 50.")

    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation. Choose from {sorted(AGGREGATIONS)}."
        )

    filtered = _apply_filters(
        dataframe=dataframe,
        filters=filters,
        combine_with=combine_with,
    )

    if filtered.empty:
        raise ValueError("No rows remain after applying the filters.")

    # HISTOGRAM
    if chart_type == "histogram":
        if x_column is None:
            raise ValueError("A histogram requires x_column.")

        _validate_column(filtered, x_column)

        numeric_values = pd.to_numeric(
            filtered[x_column],
            errors="coerce",
        )
        numeric_values = numeric_values[np.isfinite(numeric_values)]

        if numeric_values.empty:
            raise ValueError(
                f"Column '{x_column}' has no usable finite numeric values."
            )

        counts, edges = pd.cut(
            numeric_values,
            bins=bins,
            include_lowest=True,
            retbins=True,
        )
        frequencies = counts.value_counts(sort=False)

        points = [
            {
                "x": f"{edges[i]:.6g}–{edges[i + 1]:.6g}",
                "bin_start": round(float(edges[i]), 6),
                "bin_end": round(float(edges[i + 1]), 6),
                "y": int(frequencies.iloc[i]),
            }
            for i in range(len(edges) - 1)
        ]

        return {
            "chart_type": "histogram",
            "title": f"Distribution of {x_column}",
            "x_axis": x_column,
            "x_axis_type": "numeric",
            "y_axis": "Count",
            "points": points,
            "data_points": len(points),
            "source_rows": int(len(numeric_values)),
            "truncated": False,
            "missing_periods_filled": False,
        }

    # BAR, LINE AND SCATTER REQUIRE AN X-AXIS COLUMN.
    if x_column is None:
        raise ValueError(f"A {chart_type} chart requires x_column.")

    _validate_column(filtered, x_column)

    if chart_type in {"line", "scatter"} and y_column is None:
        raise ValueError(f"A {chart_type} chart requires y_column.")

    if y_column is not None:
        _validate_column(filtered, y_column)

    # SCATTER
    if chart_type == "scatter":
        if x_column == y_column:
            raise ValueError("Scatter plot columns must be different.")

        x_values = pd.to_numeric(
            filtered[x_column],
            errors="coerce",
        )
        y_values = pd.to_numeric(
            filtered[y_column],
            errors="coerce",
        )

        valid = (
            x_values.notna()
            & y_values.notna()
            & np.isfinite(x_values)
            & np.isfinite(y_values)
        )

        valid_x = x_values[valid]
        valid_y = y_values[valid]
        total_points = len(valid_x)

        if total_points == 0:
            raise ValueError(
                "The selected columns have no valid finite numeric pairs."
            )

        points = [
            {"x": float(x), "y": float(y)}
            for x, y in zip(
                valid_x.iloc[:top_n],
                valid_y.iloc[:top_n],
            )
        ]

        return {
            "chart_type": "scatter",
            "title": f"{y_column} vs {x_column}",
            "x_axis": x_column,
            "x_axis_type": "numeric",
            "y_axis": y_column,
            "points": points,
            "data_points": len(points),
            "source_rows": total_points,
            "truncated": total_points > top_n,
            "missing_periods_filled": False,
        }

    # BAR AND LINE
    selected_columns = [x_column]

    if y_column is not None:
        selected_columns.append(y_column)

    working = filtered[selected_columns].copy()

    if y_column is None:
        grouped = working.groupby(
            x_column,
            dropna=False,
            observed=True,
            sort=False,
        ).size()
        metric_label = "Count"

    elif aggregation == "count":
        grouped = working.groupby(
            x_column,
            dropna=False,
            observed=True,
            sort=False,
        )[y_column].count()
        metric_label = f"Count of {y_column}"

    else:
        working[y_column] = pd.to_numeric(
            working[y_column],
            errors="coerce",
        )

        grouped = working.groupby(
            x_column,
            dropna=False,
            observed=True,
            sort=False,
        )[y_column].agg(aggregation)

        metric_label = f"{aggregation.title()} of {y_column}"

    # Remove infinite or missing aggregate values.
    # Do not fabricate values for missing groups.
    grouped = grouped.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if grouped.empty:
        raise ValueError("No valid chart data could be generated.")

    total_groups = len(grouped)
    x_axis_type = "category"

    if chart_type == "bar":
        # Bar charts prioritize the highest aggregate values.
        grouped = grouped.sort_values(
            ascending=False,
        ).head(top_n)

    else:
        # LINE CHART: identify numeric or date-based x-axes.
        if pd.api.types.is_numeric_dtype(grouped.index.dtype):
            x_axis_type = "numeric"
            grouped = grouped.sort_index()

            # Retain the highest/latest numeric x-values.
            if total_groups > top_n:
                grouped = grouped.tail(top_n)

        else:
            # Only parse labels as dates when they look date-like.
            labels_look_like_dates = all(
                isinstance(value, (pd.Timestamp, np.datetime64))
                or (
                    isinstance(value, str)
                    and any(
                        separator in value
                        for separator in ("-", "/", ":")
                    )
                )
                for value in grouped.index
            )

            parsed_dates = pd.to_datetime(
                grouped.index,
                errors="coerce",
                utc=True,
            )

            if (
                labels_look_like_dates
                and len(parsed_dates) > 0
                and parsed_dates.notna().all()
            ):
                x_axis_type = "date"
                grouped.index = parsed_dates
                grouped = grouped.sort_index()

                # Keep the newest periods in chronological order.
                if total_groups > top_n:
                    grouped = grouped.tail(top_n)

            else:
                # Preserve first-seen order for categorical labels.
                grouped = grouped.head(top_n)

    points = [
        {
            "x": _json_value(x),
            "y": _json_value(y),
        }
        for x, y in grouped.items()
    ]

    return {
        "chart_type": chart_type,
        "title": (
            f"{metric_label} by {x_column}"
            if chart_type == "bar"
            else f"{metric_label} over {x_column}"
        ),
        "x_axis": x_column,
        "x_axis_type": x_axis_type,
        "y_axis": metric_label,
        "aggregation": aggregation if y_column else "count",
        "points": points,
        "data_points": len(points),
        "source_groups": total_groups,
        "truncated": total_groups > top_n,
        "missing_periods_filled": False,
    }