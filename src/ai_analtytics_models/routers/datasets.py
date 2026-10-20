
import uuid
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ai_analtytics_models.database import get_db
from ai_analtytics_models.dependencies import get_current_user
from ai_analtytics_models.models.dataset import Dataset
from ai_analtytics_models.models.user import User

router = APIRouter(prefix="/datasets", tags=["Datasets"])

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
