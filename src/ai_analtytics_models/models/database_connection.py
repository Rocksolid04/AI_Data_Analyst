
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ai_analtytics_models.database import Base


class DatabaseConnection(Base):
    __tablename__ = "database_connections"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    database_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="postgresql",
        server_default="postgresql",
    )

    host: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    port: Mapped[int] = mapped_column(
        nullable=False,
        default=5432,
        server_default="5432",
    )

    database_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    username: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # Store an encrypted password, never plaintext credentials.
    encrypted_password: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Track whether the latest connection check succeeded.
    is_active: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="false",
    )

    last_tested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
