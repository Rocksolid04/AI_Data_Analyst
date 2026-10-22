
import json

import logging

import uuid
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from ai_analtytics_models.services.analytics import load_dataset
from ai_analtytics_models.services.agent import answer_dataset_question


import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from io import BytesIO

from ai_analtytics_models.schemas.dataset import DatasetResponse

from ai_analtytics_models.database import get_db
from ai_analtytics_models.dependencies import get_current_user
from ai_analtytics_models.models.dataset import Dataset
from ai_analtytics_models.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])

class AnalyticsQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=500)

UPLOAD_DIR = Path("uploads/datasets")
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and XLSX files are supported.",
        )

    contents = await file.read()

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

    try:
        if extension == ".csv":
            from io import BytesIO
            dataframe = pd.read_csv(BytesIO(contents), nrows=5)
        else:
            from io import BytesIO
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
    file_path = UPLOAD_DIR / stored_filename

    try:
        file_path.write_bytes(contents)

        dataset = Dataset(
            user_id=current_user.id,
            original_filename=Path(file.filename or "dataset").name,
            stored_filename=stored_filename,
            file_type=extension.lstrip("."),
            file_size_bytes=len(contents),
        )

        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

    except Exception:
        await db.rollback()
        file_path.unlink(missing_ok=True)
        raise

    return {
        "message": "Dataset uploaded successfully.",
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "file_type": dataset.file_type,
        "file_size_bytes": dataset.file_size_bytes,
        "columns": dataframe.columns.tolist(),
    }
    

@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.scalars(
        select(Dataset)
        .where(Dataset.user_id == current_user.id)
        .order_by(Dataset.created_at.desc())
    )

    return result.all()


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.scalar(
        select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        )
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return result


from io import BytesIO

@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    dataset = await db.scalar(
        select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        )
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    file_path = UPLOAD_DIR / dataset.stored_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found on disk.",
        )

    try:
        if dataset.file_type == "csv":
            dataframe = pd.read_csv(file_path, nrows=10)
        elif dataset.file_type == "xlsx":
            dataframe = pd.read_excel(file_path, nrows=10)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported dataset type.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the dataset.",
        ) from exc

    # Convert missing values and NumPy values into JSON-compatible data.
    dataframe = dataframe.astype(object).where(dataframe.notna(), None)

    return {
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "columns": dataframe.columns.tolist(),
        "row_count": len(dataframe),
        "preview": dataframe.to_dict(orient="records"),
    }
    

@router.get("/{dataset_id}/profile")
async def profile_dataset(
    dataset_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    dataset = await db.scalar(
        select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        )
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    file_path = UPLOAD_DIR / dataset.stored_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found on disk.",
        )

    try:
        if dataset.file_type == "csv":
            dataframe = pd.read_csv(file_path)
        elif dataset.file_type == "xlsx":
            dataframe = pd.read_excel(file_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported dataset type.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
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
):
    # Verify dataset ownership.
    dataset = await db.scalar(
        select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.user_id == current_user.id,
        )
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    file_path = UPLOAD_DIR / dataset.stored_filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found on disk.",
        )

    try:
        dataframe = load_dataset(
            file_path,
            dataset.file_type,
        )

        result = await answer_dataset_question(
            dataframe=dataframe,
            question=payload.question,
        )

        # Ensure the agent returns the expected response structure.
        if not isinstance(result, dict) or not isinstance(
            result.get("answer"), str
        ):
            logger.error("Analytics agent returned an invalid response.")
            raise HTTPException(
                status_code=502,
                detail="The AI analysis returned an invalid response.",
            )

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception("Analytics agent execution failed")
        raise HTTPException(
            status_code=502,
            detail="The AI analysis could not be completed. Please try again.",
        ) from exc

    except Exception as exc:
        logger.exception("Dataset question request failed")
        raise HTTPException(
            status_code=502,
            detail="The AI analysis request failed. Please try again.",
        ) from exc
