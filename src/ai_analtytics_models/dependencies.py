
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_analtytics_models.auth import decode_access_token
from ai_analtytics_models.database import get_db
from ai_analtytics_models.models.user import User


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        subject = payload.get("sub")

        if not isinstance(subject, str) or not subject.isdecimal():
            raise credentials_exception

        user_id = int(subject)
    except (jwt.InvalidTokenError, ValueError):
        raise credentials_exception from None

    user = await db.scalar(select(User).where(User.id == user_id))

    if user is None:
        raise credentials_exception

    return user
