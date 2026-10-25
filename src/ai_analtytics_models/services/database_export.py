
import csv
import io
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine

from ai_analtytics_models.services.database_query import (
    execute_read_only_query,
)


def _csv_value(value: Any) -> Any:
    """Convert database values into CSV-compatible, spreadsheet-safe values."""
    if value is None:
        return ""

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, (Decimal, UUID)):
        value = str(value)

    if isinstance(value, str):
        # Reduce CSV formula-injection risk in spreadsheet applications.
        if value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
            return "'" + value

    return value


async def export_query_to_csv(
    engine: AsyncEngine,
    sql_query: str,
) -> dict[str, Any]:
    """Execute a validated read-only SQL query and export its results as CSV."""
    result = await execute_read_only_query(
        engine=engine,
        query=sql_query,
    )

    output = io.StringIO(newline="")
    writer = csv.writer(output)

    writer.writerow(result["columns"])

    for row in result["rows"]:
        writer.writerow(
            [_csv_value(row.get(column)) for column in result["columns"]]
        )

    return {
        "csv_content": output.getvalue(),
        "row_count": result["row_count"],
        "column_count": len(result["columns"]),
        "truncated": result["truncated"],
        "max_rows": result["max_rows"],
    }
