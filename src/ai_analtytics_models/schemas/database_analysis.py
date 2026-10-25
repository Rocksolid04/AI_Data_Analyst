
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatabaseToolResult(BaseModel):
    tool: str
    result: Any = None


class DatabaseAnalysisResponse(BaseModel):
    connection_id: int
    question: str
    answer: str
    tool_calls: list[DatabaseToolResult] = Field(default_factory=list)
    chart: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    report_error: str | None = None

    model_config = ConfigDict(extra="forbid")
