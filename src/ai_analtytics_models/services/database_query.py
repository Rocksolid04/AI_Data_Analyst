
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

MAX_ROWS = 500
QUERY_TIMEOUT_MS = 5_000

# Reject multiple SQL statements and common write/administrative operations.
FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|UPSERT|MERGE|"
    r"CREATE|ALTER|DROP|TRUNCATE|"
    r"GRANT|REVOKE|"
    r"COPY|CALL|DO|"
    r"VACUUM|ANALYZE|"
    r"REFRESH|"
    r"PG_SLEEP"
    r")\b",
    re.IGNORECASE,
)


def validate_read_only_query(query: str) -> str:
    """Validate a single, read-only SELECT or WITH query."""
    if not query or not query.strip():
        raise ValueError("SQL query cannot be empty.")

    cleaned = query.strip()

    # Allow at most one optional trailing semicolon.
    without_trailing_semicolon = cleaned[:-1].rstrip() if cleaned.endswith(";") else cleaned

    if ";" in without_trailing_semicolon:
        raise ValueError("Only one SQL statement is allowed.")

    if FORBIDDEN_KEYWORDS.search(without_trailing_semicolon):
        raise ValueError("The query contains a forbidden SQL operation.")

    if not re.match(r"^(SELECT|WITH)\b", without_trailing_semicolon, re.IGNORECASE):
        raise ValueError("Only SELECT queries and read-only CTEs are allowed.")

    return without_trailing_semicolon



async def execute_read_only_query(
    engine: AsyncEngine,
    query: str,
) -> dict[str, Any]:
    """Execute one validated read-only query and return bounded results."""
    safe_query = validate_read_only_query(query)

    async with engine.connect() as connection:
        async with connection.begin():
            await connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            await connection.exec_driver_sql(
                f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'"
            )

            result = await connection.execute(text(safe_query))
            columns = list(result.keys())
            rows = result.mappings().fetchmany(MAX_ROWS + 1)

            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]

            return {
                "columns": columns,
                "rows": [dict(row) for row in rows],
                "row_count": len(rows),
                "truncated": truncated,
                "max_rows": MAX_ROWS,
                "summary": {
                    "returned_rows": len(rows),
                    "returned_columns": len(columns),
                    "truncated": truncated,
                },
            }
