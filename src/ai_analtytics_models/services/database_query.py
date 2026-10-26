
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

MAX_ROWS = 500
MAX_QUERY_LENGTH = 50_000
QUERY_TIMEOUT_MS = 5_000

# Reject SQL statement types that can modify data, change database state,
# administer the server, or otherwise exceed the read-only query boundary.
FORBIDDEN_EXPRESSION_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Alter",
    "Drop",
    "TruncateTable",
    "Command",
    "Copy",
    "Grant",
    "Revoke",
    "Set",
    "SetItem",
    "Use",
    "Transaction",
    "Commit",
    "Rollback",
    "Execute",
    "Declare",
    "Pragma",
    "Attach",
    "Detach",
    "Into",
    "Lock",
    "Analyze",
    "Cache",
    "Uncache",
    "LoadData",
    "Export",
    "Import",
)

FORBIDDEN_FUNCTIONS = {
    # Delay and configuration changes.
    "PG_SLEEP",
    "PG_SLEEP_FOR",
    "PG_SLEEP_UNTIL",
    "SET_CONFIG",

    # Sequence mutations.
    "NEXTVAL",
    "SETVAL",
    "PG_SEQUENCE_LAST_VALUE",

    # Server and operating-system access.
    "PG_READ_FILE",
    "PG_READ_BINARY_FILE",
    "PG_WRITE_FILE",
    "PG_WRITE_BINARY_FILE",
    "PG_LS_DIR",
    "PG_STAT_FILE",
    "LO_IMPORT",
    "LO_EXPORT",

    # Remote execution and connection functions.
    "DBLINK_EXEC",
    "DBLINK_SEND_QUERY",
    "DBLINK_GET_RESULT",

    # Server administration.
    "PG_CANCEL_BACKEND",
    "PG_TERMINATE_BACKEND",
    "PG_RELOAD_CONF",
    "PG_ROTATE_LOGFILE",
    "PG_LOG_BACKEND_MEMORY_CONTEXTS",
    "PG_NOTIFY",

    # Advisory locks.
    "PG_ADVISORY_LOCK",
    "PG_ADVISORY_LOCK_SHARED",
    "PG_TRY_ADVISORY_LOCK",
    "PG_TRY_ADVISORY_LOCK_SHARED",
    "PG_ADVISORY_XACT_LOCK",
    "PG_ADVISORY_XACT_LOCK_SHARED",
}


def _forbidden_expression_types() -> tuple[type, ...]:
    """Resolve forbidden SQLGlot expression types available in this version."""
    expression_types = []

    for name in FORBIDDEN_EXPRESSION_NAMES:
        expression_type = getattr(exp, name, None)

        if isinstance(expression_type, type):
            expression_types.append(expression_type)

    return tuple(expression_types)


def _validate_functions(expression: exp.Expression) -> None:
    """Reject known side-effecting or dangerous SQL functions."""
    for node in expression.walk():
        if not isinstance(node, exp.Func):
            continue

        try:
            function_name = node.sql_name().upper()
        except (AttributeError, TypeError):
            function_name = ""

        # Anonymous functions preserve their name in `this`.
        if isinstance(node, exp.Anonymous):
            function_name = node.name.upper()

        if function_name in FORBIDDEN_FUNCTIONS:
            raise ValueError(
                f"The SQL query uses a forbidden function: {function_name}."
            )


def validate_read_only_query(query: str) -> str:
    """Validate a single read-only SELECT, CTE, or set-operation query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("SQL query cannot be empty.")

    cleaned = query.strip()

    if len(cleaned) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"SQL query exceeds the {MAX_QUERY_LENGTH}-character limit."
        )

    try:
        expressions = sqlglot.parse(cleaned, read="postgres")
    except ParseError as exc:
        raise ValueError("The SQL query could not be parsed safely.") from exc

    # A single optional trailing semicolon is allowed. SQLGlot handles
    # semicolons inside string literals and comments as SQL syntax.
    expressions = [expression for expression in expressions if expression is not None]

    if len(expressions) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    expression = expressions[0]

    # SELECT, WITH ... SELECT, and read-only set operations are accepted.
    # SQLGlot represents a WITH query using its underlying query expression.
    allowed_roots = (exp.Select, exp.Union, exp.Intersect, exp.Except)

    if not isinstance(expression, allowed_roots):
        raise ValueError(
            "Only SELECT queries, read-only CTEs, and read-only set "
            "operations are allowed."
        )

    forbidden_types = _forbidden_expression_types()

    for node in expression.walk():
        if forbidden_types and isinstance(node, forbidden_types):
            raise ValueError(
                "The SQL query contains a forbidden operation."
            )

        # Explicitly reject SELECT ... INTO, which can create a table.
        if isinstance(node, exp.Select) and node.args.get("into") is not None:
            raise ValueError("SELECT INTO is not allowed.")

    _validate_functions(expression)

    # Return the original query, excluding an optional trailing semicolon.
    # Do not re-render the AST because rendering could alter SQL semantics.
    return cleaned[:-1].rstrip() if cleaned.endswith(";") else cleaned


async def execute_read_only_query(
    engine: AsyncEngine,
    query: str,
) -> dict[str, Any]:
    """Execute one validated query with transaction and result limits."""
    safe_query = validate_read_only_query(query)

    async with engine.connect() as connection:
        async with connection.begin():
            # These settings are applied before the user query executes.
            await connection.exec_driver_sql(
                "SET TRANSACTION READ ONLY"
            )
            await connection.exec_driver_sql(
                f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'"
            )

            result = await connection.execute(text(safe_query))

            columns = list(result.keys())
            rows = result.mappings().fetchmany(MAX_ROWS + 1)

            truncated = len(rows) > MAX_ROWS
            rows = rows[:MAX_ROWS]

            returned_rows = [dict(row) for row in rows]

            return {
                "columns": columns,
                "rows": returned_rows,
                "row_count": len(returned_rows),
                "truncated": truncated,
                "max_rows": MAX_ROWS,
                "summary": {
                    "returned_rows": len(returned_rows),
                    "returned_columns": len(columns),
                    "truncated": truncated,
                },
            }
