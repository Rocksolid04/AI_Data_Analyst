
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_analtytics_models.database import get_db
from ai_analtytics_models.dependencies import get_current_user
from ai_analtytics_models.models.dataset import Dataset
from ai_analtytics_models.models.user import User
from ai_analtytics_models.schemas.dataset import DatasetResponse
from ai_analtytics_models.services.agent import answer_dataset_question
from ai_analtytics_models.services.analytics import load_dataset


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])

UPLOAD_DIR = Path("uploads/datasets")
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_SIZE = 10 * 1024 * 1024


class AnalyticsQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)


def _get_dataset_path(stored_filename: str) -> Path:
    """Build a path from the application's generated stored filename."""
    filename = Path(stored_filename)

    # Stored filenames should always be generated UUID filenames.
    if filename.name != stored_filename or filename.is_absolute():
        raise ValueError("Invalid stored dataset filename.")

    return UPLOAD_DIR / filename


async def _get_owned_dataset(
    dataset_id: int,
    user_id: int,
    db: AsyncSession,
) -> Dataset:
    """Retrieve a dataset only when it belongs to the current user."""
    dataset = await db.scalar(
        select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.user_id == user_id,
        )
    )

    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    return dataset


def _read_dataset_preview(
    file_path: Path,
    file_type: str,
    rows: int = 10,
) -> pd.DataFrame:
    """Read a limited number of rows for dataset previews."""
    if file_type == "csv":
        return pd.read_csv(file_path, nrows=rows)

    if file_type == "xlsx":
        return pd.read_excel(file_path, nrows=rows)

    raise ValueError("Unsupported dataset type.")


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Validate and store an uploaded CSV or XLSX dataset."""
    original_filename = (file.filename or "").replace("\\", "/")
    safe_original_filename = Path(original_filename).name
    extension = Path(safe_original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and XLSX files are supported.",
        )

    try:
        # Read at most one byte over the limit to detect oversized uploads.
        contents = await file.read(MAX_FILE_SIZE + 1)
    except Exception as exc:
        logger.exception("Failed to read uploaded dataset.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the uploaded file.",
        ) from exc

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size must not exceed 10 MB.",
        )

    # Validate that the file can be read before storing it.
    try:
        if extension == ".csv":
            dataframe = pd.read_csv(BytesIO(contents), nrows=5)
        else:
            dataframe = pd.read_excel(BytesIO(contents), nrows=5)

        if len(dataframe.columns) == 0:
            raise ValueError("No columns found.")

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file could not be read as a valid CSV or XLSX dataset.",
        ) from exc

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = _get_dataset_path(stored_filename)

    try:
        file_path.write_bytes(contents)

        dataset = Dataset(
            user_id=current_user.id,
            original_filename=safe_original_filename or "dataset",
            stored_filename=stored_filename,
            file_type=extension.lstrip("."),
            file_size_bytes=len(contents),
        )

        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

    except Exception as exc:
        await db.rollback()
        file_path.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded dataset.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the uploaded dataset.",
        ) from exc

    return {
        "message": "Dataset uploaded successfully.",
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "file_type": dataset.file_type,
        "file_size_bytes": dataset.file_size_bytes,
        "columns": [str(column) for column in dataframe.columns],
    }


@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Dataset]:
    """List datasets belonging to the authenticated user."""
    result = await db.scalars(
        select(Dataset)
        .where(Dataset.user_id == current_user.id)
        .order_by(Dataset.created_at.desc())
    )

    return list(result.all())


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Dataset:
    """Retrieve one dataset owned by the authenticated user."""
    return await _get_owned_dataset(
        dataset_id=dataset_id,
        user_id=current_user.id,
        db=db,
    )


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Return a small preview of an owned dataset."""
    dataset = await _get_owned_dataset(
        dataset_id=dataset_id,
        user_id=current_user.id,
        db=db,
    )

    file_path = _get_dataset_path(dataset.stored_filename)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on disk.",
        )

    try:
        dataframe = _read_dataset_preview(
            file_path=file_path,
            file_type=dataset.file_type,
            rows=10,
        )
    except Exception as exc:
        logger.exception("Could not read dataset preview.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the dataset.",
        ) from exc

    # Replace missing values with None for JSON-compatible output.
    dataframe = dataframe.astype(object).where(dataframe.notna(), None)

    return {
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "columns": [str(column) for column in dataframe.columns],
        "row_count": len(dataframe),
        "preview": dataframe.to_dict(orient="records"),
    }


@router.get("/{dataset_id}/profile")
async def profile_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Return dataset dimensions, column profiles, and numeric statistics."""
    dataset = await _get_owned_dataset(
        dataset_id=dataset_id,
        user_id=current_user.id,
        db=db,
    )

    file_path = _get_dataset_path(dataset.stored_filename)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on disk.",
        )

    try:
        dataframe = load_dataset(
            file_path=file_path,
            file_type=dataset.file_type,
        )
    except Exception as exc:
        logger.exception("Could not load dataset for profiling.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the dataset.",
        ) from exc

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    numeric_summary = {}

    if len(numeric_columns) > 0:
        summary = dataframe[numeric_columns].describe()
        numeric_summary = json.loads(summary.to_json())

    column_profiles = []

    for column in dataframe.columns:
        column_profiles.append(
            {
                "column": str(column),
                "data_type": str(dataframe[column].dtype),
                "missing_values": int(dataframe[column].isna().sum()),
                "unique_values": int(dataframe[column].nunique()),
            }
        )

    return {
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "total_rows": len(dataframe),
        "total_columns": len(dataframe.columns),
        "columns": column_profiles,
        "numeric_summary": numeric_summary,
    }


@router.post("/{dataset_id}/ask")
async def ask_dataset_question(
    dataset_id: int,
    payload: AnalyticsQuestion,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, Any]:
    """Answer a natural-language question about an owned dataset."""
    dataset = await _get_owned_dataset(
        dataset_id=dataset_id,
        user_id=current_user.id,
        db=db,
    )

    file_path = _get_dataset_path(dataset.stored_filename)

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on disk.",
        )

    try:
        dataframe = load_dataset(
            file_path=file_path,
            file_type=dataset.file_type,
        )
    except Exception as exc:
        logger.exception("Could not load dataset for analysis.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the dataset.",
        ) from exc

    try:
        result = await answer_dataset_question(
            dataframe=dataframe,
            question=payload.question,
        )

        # Validate the agent response without discarding report fields.
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("answer"), str)
            or not isinstance(result.get("tool_calls"), list)
        ):
            logger.error("Analytics agent returned an invalid response.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI analysis returned an invalid response.",
            )

        # Keep the chart field consistent for frontend consumers.
        result.setdefault("chart", None)

        if result["chart"] is not None and not isinstance(
            result["chart"], dict
        ):
            logger.error("Analytics agent returned an invalid chart.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI analysis returned an invalid chart.",
            )

        if "report" in result and result["report"] is not None:
            if not isinstance(result["report"], dict):
                logger.error("Analytics agent returned an invalid report.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="The AI analysis returned an invalid report.",
                )

        if (
            "report_error" in result
            and not isinstance(result["report_error"], str)
        ):
            logger.error("Analytics agent returned an invalid report error.")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI analysis returned an invalid report status.",
            )

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception("Analytics agent execution failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI analysis could not be completed. Please try again.",
        ) from exc

    except Exception as exc:
        logger.exception("Dataset question request failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI analysis request failed. Please try again.",
        ) from exc
