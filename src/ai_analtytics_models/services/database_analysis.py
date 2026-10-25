
from typing import Any

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_analtytics_models.models.database_connection import DatabaseConnection
from ai_analtytics_models.services.credentials_encryption import (
    decrypt_password,
)
from ai_analtytics_models.services.database_inspector import (
    inspect_database,
)
from ai_analtytics_models.services.database_query import (
    execute_read_only_query,
)


CONNECTION_TIMEOUT_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 5


def create_external_engine(connection: DatabaseConnection):
    """Create an engine for a user's saved PostgreSQL connection."""
    password = decrypt_password(connection.encrypted_password)

    url = URL.create(
        drivername="postgresql+asyncpg",
        username=connection.username,
        password=password,
        host=connection.host,
        port=connection.port,
        database=connection.database_name,
    )

    return create_async_engine(
        url,
        poolclass=NullPool,
        connect_args={
            "timeout": CONNECTION_TIMEOUT_SECONDS,
            "command_timeout": QUERY_TIMEOUT_SECONDS,
            "server_settings": {
                "statement_timeout": str(QUERY_TIMEOUT_SECONDS * 1000),
            },
        },
    )


async def inspect_connected_database(
    connection: DatabaseConnection,
) -> dict[str, Any]:
    """Inspect tables and columns in a saved PostgreSQL connection."""
    engine = create_external_engine(connection)

    try:
        return await inspect_database(engine)
    finally:
        await engine.dispose()


async def query_connected_database(
    connection: DatabaseConnection,
    query: str,
) -> dict[str, Any]:
    """Run a query against a saved PostgreSQL connection."""
    engine = create_external_engine(connection)

    try:
        return await execute_read_only_query(
            engine=engine,
            query=query,
        )
    finally:
        await engine.dispose()
