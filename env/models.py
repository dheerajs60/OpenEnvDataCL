from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action as BaseAction, Observation as BaseObservation

class Observation(BaseObservation):
    rows_preview: list = Field(default_factory=list)
    table_schema: dict = Field(default_factory=dict)
    detected_issues: list = Field(default_factory=list)
    step_count: int = 0
    metadata: dict = Field(default_factory=dict)

class Action(BaseAction):
    operation: str
    column: str | None = None
    value: str | None = None

class Reward(BaseModel):
    score: float
    reason: str
