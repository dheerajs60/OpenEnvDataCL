from typing import Any
from openenv.core.env_server.types import Action as BaseAction, Observation as BaseObservation
from pydantic import BaseModel, Field

class Observation(BaseObservation):
    rows_preview: list[dict] = Field(default_factory=list)
    table_schema: list[dict] = Field(default_factory=list)
    detected_issues: list[dict] = Field(default_factory=list)
    step_count: int = Field(default=0)

class Action(BaseAction):
    operation: str = Field(..., description="Action to perform")
    column: str | None = Field(None, description="Column to operate on")
    value: Any = Field(None, description="Value for the operation")

class Reward(BaseModel):
    score: float
    reason: str
