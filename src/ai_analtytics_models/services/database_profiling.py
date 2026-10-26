
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from ai_analtytics_models.services.database_inspector import (
    inspect_database,
)
from ai_analtytics_models.services.database_query import (
    execute_read_only_query,
)



async def profile_database(
    engine: AsyncEngine,
    table_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return table inventory and optionally profile selected tables.

    Use schema-qualified names such as public.orders when possible.
    Detailed profiling is limited to five tables per call.
    """
    schema = await inspect_database(engine)

    table_inventory = []
    available_tables = []

    for table in schema["tables"]:
        if table["type"] not in {"BASE TABLE", "VIEW"}:
            continue

        table_inventory.append(
            {
                "schema": table["schema"],
                "table": table["name"],
                "type": table["type"],
                "column_count": len(table["columns"]),
                "columns": table["columns"],
            }
        )
        available_tables.append(table)

    profiles = []

    if table_names:
        requested_names = list(dict.fromkeys(
            name.strip()
            for name in table_names
            if isinstance(name, str) and name.strip()
        ))

        if len(requested_names) > 5:
            raise ValueError(
                "Profile at most five tables per request."
            )

        for requested_name in requested_names:
            if "." in requested_name:
                schema_name, table_name = requested_name.split(".", 1)

                matches = [
                    table
                    for table in available_tables
                    if table["schema"] == schema_name
                    and table["name"] == table_name
                ]
            else:
                matches = [
                    table
                    for table in available_tables
                    if table["name"] == requested_name
                ]

                if len(matches) > 1:
                    raise ValueError(
                        f"Table name '{requested_name}' is ambiguous. "
                        "Use a schema-qualified name."
                    )

            if not matches:
                raise ValueError(
                    f"Table '{requested_name}' was not found. "
                    "Use a name from the database schema inventory."
                )

            table = matches[0]

            profile = await profile_database_table(
                engine=engine,
                schema_name=table["schema"],
                table_name=table["name"],
            )
            profiles.append(profile)

    return {
        "table_count": schema["table_count"],
        "tables": table_inventory,
        "profiles": profiles,
        "profiled_table_count": len(profiles),
        "profile_limit": 5,
    }


async def preview_database_table(
    engine: AsyncEngine,
    schema_name: str,
    table_name: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Preview a table using validated, read-only SQL."""
    if not 1 <= limit <= 100:
        raise ValueError("Preview limit must be between 1 and 100.")

    schema = await inspect_database(engine)

    exists = any(
        table["schema"] == schema_name
        and table["name"] == table_name
        and table["type"] in {"BASE TABLE", "VIEW"}
        for table in schema["tables"]
    )

    if not exists:
        raise ValueError("The requested table or view was not found.")

    # Quote identifiers safely by doubling embedded double quotes.
    quoted_schema = '"' + schema_name.replace('"', '""') + '"'
    quoted_table = '"' + table_name.replace('"', '""') + '"'

    query = (
        f"SELECT * FROM {quoted_schema}.{quoted_table} "
        f"LIMIT {limit}"
    )

    result = await execute_read_only_query(
        engine=engine,
        query=query,
    )

    return {
        "schema": schema_name,
        "table": table_name,
        **result,
    }
    

NUMERIC_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "numeric",
    "decimal",
    "real",
    "double precision",
}


def _quote_identifier(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier."""
    return '"' + identifier.replace('"', '""') + '"'

async def profile_database_table(
    engine: AsyncEngine,
    schema_name: str,
    table_name: str,
) -> dict[str, Any]:
    """Return row count, column statistics, and duplicate-row count."""
    schema = await inspect_database(engine)

    table = next(
        (
            item
            for item in schema["tables"]
            if item["schema"] == schema_name
            and item["name"] == table_name
            and item["type"] in {"BASE TABLE", "VIEW"}
        ),
        None,
    )

    if table is None:
        raise ValueError("The requested table or view was not found.")

    all_columns = table["columns"]

    if not all_columns:
        raise ValueError("The requested table has no inspectable columns.")

    # Keep per-column profiling bounded for wide tables.
    columns = all_columns[:50]

    quoted_table = (
        f"{_quote_identifier(schema_name)}."
        f"{_quote_identifier(table_name)}"
    )

    expressions = ["COUNT(*) AS total_rows"]
    column_aliases = {}

    for index, column in enumerate(columns):
        name = column["name"]
        quoted_column = _quote_identifier(name)

        null_alias = f"nulls_{index}"
        distinct_alias = f"distinct_{index}"

        expressions.append(
            f"(COUNT(*) - COUNT({quoted_column})) AS {null_alias}"
        )
        expressions.append(
            f"COUNT(DISTINCT {quoted_column}) AS {distinct_alias}"
        )

        column_info = {
            "name": name,
            "data_type": column["data_type"],
            "nullable": column["nullable"],
            "null_alias": null_alias,
            "distinct_alias": distinct_alias,
        }

        if column["data_type"].lower() in NUMERIC_TYPES:
            min_alias = f"min_{index}"
            max_alias = f"max_{index}"
            avg_alias = f"avg_{index}"

            expressions.extend(
                [
                    f"MIN({quoted_column}) AS {min_alias}",
                    f"MAX({quoted_column}) AS {max_alias}",
                    f"AVG({quoted_column}) AS {avg_alias}",
                ]
            )

            column_info.update(
                {
                    "min_alias": min_alias,
                    "max_alias": max_alias,
                    "avg_alias": avg_alias,
                }
            )

        column_aliases[name] = column_info

    query = f"SELECT {', '.join(expressions)} FROM {quoted_table}"

    result = await execute_read_only_query(
        engine=engine,
        query=query,
    )

    aggregate = result["rows"][0] if result.get("rows") else {}

    column_profiles = []

    for column in columns:
        info = column_aliases[column["name"]]

        profile = {
            "name": info["name"],
            "data_type": info["data_type"],
            "nullable": info["nullable"],
            "null_count": aggregate.get(info["null_alias"]),
            "distinct_count": aggregate.get(info["distinct_alias"]),
        }

        if "min_alias" in info:
            profile["min"] = aggregate.get(info["min_alias"])
            profile["max"] = aggregate.get(info["max_alias"])
            profile["average"] = aggregate.get(info["avg_alias"])

        column_profiles.append(profile)

    # Count extra copies of identical rows across ALL columns.
    # If a row appears 3 times, it contributes 2 duplicate rows.
    duplicate_row_count = None
    duplicate_row_error = None

    try:
        duplicate_query = f"""
            SELECT
                COUNT(*) - COUNT(DISTINCT to_jsonb(t))
                    AS duplicate_row_count
            FROM {quoted_table} AS t
        """

        duplicate_result = await execute_read_only_query(
            engine=engine,
            query=duplicate_query,
        )

        duplicate_rows = duplicate_result.get("rows", [])

        if duplicate_rows:
            duplicate_row_count = int(
                duplicate_rows[0]["duplicate_row_count"]
            )
        else:
            duplicate_row_error = (
                "The database returned no duplicate-count result."
            )

    except Exception:
        logger.exception(
            "Failed to calculate duplicate rows for %s.%s",
            schema_name,
            table_name,
        )
        duplicate_row_error = (
            "Duplicate-row counting failed for this table."
        )

    return {
        "schema": schema_name,
        "table": table_name,
        "row_count": aggregate.get("total_rows", 0),
        "column_count": len(all_columns),
        "profiled_column_count": len(columns),
        "columns_truncated": len(all_columns) > len(columns),
        "columns": column_profiles,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_error": duplicate_row_error,
    }

