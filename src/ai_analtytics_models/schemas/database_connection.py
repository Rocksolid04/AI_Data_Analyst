
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class DatabaseConnectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=1, max_length=255)
    password: SecretStr

    @field_validator("name", "host", "database_name", "username")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("Password cannot be empty.")
        return value


class DatabaseConnectionResponse(BaseModel):
    id: int
    name: str
    database_type: str
    host: str
    port: int
    database_name: str
    username: str
    is_active: bool
    last_tested_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatabaseConnectionTestResponse(BaseModel):
    connection_id: int
    success: bool
    message: str
    tested_at: datetime


class DatabaseQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
