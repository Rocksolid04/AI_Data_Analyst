
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi.responses import Response

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine

from ai_analtytics_models.database import get_db
from ai_analtytics_models.dependencies import get_current_user
from ai_analtytics_models.models.database_connection import DatabaseConnection
from ai_analtytics_models.models.user import User

from ai_analtytics_models.schemas.database_connection import (
    DatabaseConnectionCreate,
    DatabaseConnectionResponse,
    DatabaseConnectionTestResponse,
    DatabaseQueryRequest,
    DatabaseSQLExportRequest,
)
from ai_analtytics_models.schemas.database_analysis import (
    DatabaseAnalysisResponse,
)

from ai_analtytics_models.services.database_export import (
    export_query_to_csv,
)

from ai_analtytics_models.services.database_agent import (
    answer_database_question,
)

from ai_analtytics_models.services.credentials_encryption import (
    encrypt_password,
)
from ai_analtytics_models.services.database_analysis import (
    create_external_engine,
)
from ai_analtytics_models.services.database_inspector import (
    inspect_database,
)
from ai_analtytics_models.services.database_query import (
    execute_read_only_query,
)
from ai_analtytics_models.services.database_profiling import (
    profile_database,
    preview_database_table,
    profile_database_table,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/database-connections",
    tags=["Database Connections"],
)


async def _get_owned_connection(
    connection_id: int,
    user_id: int,
    db: AsyncSession,
) -> DatabaseConnection:
    """Fetch a connection only if it belongs to the authenticated user."""
    connection = await db.scalar(
        select(DatabaseConnection).where(
            DatabaseConnection.id == connection_id,
            DatabaseConnection.user_id == user_id,
        )
    )

    if connection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Database connection not found.",
        )

    return connection


@router.post(
    "/",
    response_model=DatabaseConnectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_database_connection(
    payload: DatabaseConnectionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DatabaseConnection:
    """Save a database connection without storing plaintext credentials."""
    try:
        encrypted_password = encrypt_password(
            payload.password.get_secret_value()
        )
    except (RuntimeError, ValueError):
        logger.exception("Could not encrypt database credentials.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Secure credential storage is not configured correctly.",
        ) from None

    connection = DatabaseConnection(
        user_id=current_user.id,
        name=payload.name,
        database_type="postgresql",
        host=payload.host,
        port=payload.port,
        database_name=payload.database_name,
        username=payload.username,
        encrypted_password=encrypted_password,
        is_active=False,
    )

    try:
        db.add(connection)
        await db.commit()
        await db.refresh(connection)
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Could not save database connection metadata.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the database connection.",
        ) from None

    return connection


@router.get(
    "/",
    response_model=list[DatabaseConnectionResponse],
)
async def list_database_connections(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[DatabaseConnection]:
    """List the authenticated user's saved database connections."""
    result = await db.scalars(
        select(DatabaseConnection)
        .where(DatabaseConnection.user_id == current_user.id)
        .order_by(DatabaseConnection.created_at.desc())
    )

    return list(result.all())


@router.post(
    "/{connection_id}/test",
    response_model=DatabaseConnectionTestResponse,
)
async def test_database_connection(
    connection_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DatabaseConnectionTestResponse:
    """Test connectivity to a saved PostgreSQL database."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    tested_at = datetime.now(timezone.utc)
    engine: AsyncEngine | None = None
    success = False

    try:
        engine = create_external_engine(connection)

        async with engine.connect() as external_db:
            result = await external_db.execute(text("SELECT 1"))
            success = result.scalar_one() == 1

    except Exception:
        logger.warning(
            "Connection test failed for database connection ID %s.",
            connection.id,
            exc_info=True,
        )

    finally:
        if engine is not None:
            await engine.dispose()

    connection.is_active = success
    connection.last_tested_at = tested_at

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Could not persist connection test result.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the connection test result.",
        ) from None

    return DatabaseConnectionTestResponse(
        connection_id=connection.id,
        success=success,
        message=(
            "Connection successful."
            if success
            else (
                "Connection failed. Check the host, port, database, "
                "credentials, network access, and PostgreSQL permissions."
            )
        ),
        tested_at=tested_at,
    )


@router.delete(
    "/{connection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_database_connection(
    connection_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a saved connection owned by the authenticated user."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    try:
        await db.delete(connection)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        logger.exception("Could not delete database connection.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete the database connection.",
        ) from None


@router.get("/{connection_id}/schema")
async def get_database_schema(
    connection_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Return table and column metadata for an owned connection."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    engine: AsyncEngine | None = None

    try:
        engine = create_external_engine(connection)
        schema = await inspect_database(engine)

        return {
            "connection_id": connection.id,
            "connection_name": connection.name,
            **schema,
        }
        
    
    except HTTPException:
        raise
    
    except Exception:
        logger.exception(
            "Failed to inspect database connection ID %s.",
            connection.id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Could not inspect the connected database. "
                "Check the connection and database permissions."
            ),
        ) from None

    finally:
        if engine is not None:
            await engine.dispose()


@router.post("/{connection_id}/query")
async def query_connected_database(
    connection_id: int,
    payload: DatabaseQueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Execute a query against an owned database connection."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    engine: AsyncEngine | None = None

    try:
        engine = create_external_engine(connection)

        result = await execute_read_only_query(
            engine=engine,
            query=payload.query,
        )

        return {
            "connection_id": connection.id,
            "connection_name": connection.name,
            **result,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None

    except Exception:
        logger.exception(
            "Query failed for database connection ID %s.",
            connection.id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The query could not be completed. "
                "Check the SQL and database permissions."
            ),
        ) from None

    finally:
        if engine is not None:
            await engine.dispose()
            
            

@router.post(
    "/{connection_id}/ask",
    response_model=DatabaseAnalysisResponse,
)
async def ask_database_question(
    connection_id: int,
    request: DatabaseQueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Answer a natural-language question using an owned PostgreSQL connection."""

    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    try:
        return await answer_database_question(
            connection=connection,
            question=request.query,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Database question failed for connection_id=%s",
            connection_id,
        )

        # Do not return raw database or provider exceptions to the client.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The database analysis could not be completed. "
                "Check the connection and query, then try again."
            ),
        ) from exc


@router.get("/{connection_id}/profile")
async def get_database_profile(
    connection_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Return table and column metadata for an owned database connection."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    engine = create_external_engine(connection)
    try:
        return await profile_database(engine)
    except Exception as exc:
        logger.exception(
            "Database profiling failed for connection_id=%s",
            connection_id,
        )
        raise HTTPException(
            status_code=502,
            detail="Could not profile the connected database.",
        ) from exc
    finally:
        await engine.dispose()


@router.get("/{connection_id}/tables/{schema_name}/{table_name}/preview")
async def get_database_table_preview(
    connection_id: int,
    schema_name: str,
    table_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 10,
):
    """Preview rows from a table in an owned database connection."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    if not 1 <= limit <= 100:
        raise HTTPException(
            status_code=422,
            detail="Preview limit must be between 1 and 100.",
        )

    engine = create_external_engine(connection)
    try:
        return await preview_database_table(
            engine=engine,
            schema_name=schema_name,
            table_name=table_name,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Database table preview failed for connection_id=%s",
            connection_id,
        )
        raise HTTPException(
            status_code=502,
            detail="Could not preview the requested database table.",
        ) from exc
    finally:
        await engine.dispose()


@router.get(
    "/{connection_id}/tables/{schema_name}/{table_name}/profile"
)
async def get_database_table_profile(
    connection_id: int,
    schema_name: str,
    table_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Return statistical profiling information for a database table."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    engine = create_external_engine(connection)

    try:
        return await profile_database_table(
            engine=engine,
            schema_name=schema_name,
            table_name=table_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Database table profiling failed for connection_id=%s",
            connection_id,
        )
        raise HTTPException(
            status_code=502,
            detail="Could not profile the requested database table.",
        ) from exc
    finally:
        await engine.dispose()
        
        

@router.post("/{connection_id}/export/csv")
async def export_database_query_csv(
    connection_id: int,
    request: DatabaseSQLExportRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Export the results of a validated read-only SQL query as CSV."""
    connection = await _get_owned_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        db=db,
    )

    engine = create_external_engine(connection)

    try:
        export_result = await export_query_to_csv(
            engine=engine,
            sql_query=request.sql_query,
        )

        filename = f"database_connection_{connection_id}_export.csv"

        return Response(
            content=export_result["csv_content"],
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Export-Row-Count": str(export_result["row_count"]),
                "X-Export-Truncated": str(export_result["truncated"]).lower(),
            },
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Database CSV export failed for connection_id=%s",
            connection_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The database query could not be exported.",
        ) from exc

    finally:
        await engine.dispose()

