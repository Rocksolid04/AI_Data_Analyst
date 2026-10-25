
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Only inspect the current database's ordinary and partitioned tables.
_TABLES_QUERY = text(
    """
    SELECT
        table_schema,
        table_name,
        table_type
    FROM information_schema.tables
    WHERE table_schema NOT IN (
        'information_schema',
        'pg_catalog'
    )
      AND table_type IN ('BASE TABLE', 'VIEW')
    ORDER BY table_schema, table_name
    """
)

_COLUMNS_QUERY = text(
    """
    SELECT
        column_name,
        data_type,
        is_nullable,
        ordinal_position
    FROM information_schema.columns
    WHERE table_schema = :schema_name
      AND table_name = :table_name
    ORDER BY ordinal_position
    """
)


from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


_TABLES_QUERY = text("""
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
  AND table_type IN ('BASE TABLE', 'VIEW')
ORDER BY table_schema, table_name
""")

_COLUMNS_QUERY = text("""
SELECT column_name, data_type, is_nullable, ordinal_position
FROM information_schema.columns
WHERE table_schema = :schema_name
  AND table_name = :table_name
ORDER BY ordinal_position
""")


async def inspect_database(engine: AsyncEngine) -> dict[str, Any]:
    """Return table and column metadata for a PostgreSQL database."""
    tables = []

    async with engine.connect() as connection:
        result = await connection.execute(_TABLES_QUERY)
        table_rows = result.mappings().all()

        for row in table_rows:
            schema_name = row["table_schema"]
            table_name = row["table_name"]

            column_result = await connection.execute(
                _COLUMNS_QUERY,
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                },
            )

            columns = [
                {
                    "name": column["column_name"],
                    "data_type": column["data_type"],
                    "nullable": column["is_nullable"] == "YES",
                    "ordinal_position": column["ordinal_position"],
                }
                for column in column_result.mappings().all()
            ]

            tables.append({
                "schema": schema_name,
                "name": table_name,
                "type": row["table_type"],
                "columns": columns,
            })

    return {
        "table_count": len(tables),
        "tables": tables,
    }


