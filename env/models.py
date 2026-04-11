from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server.types import Action as BaseAction, Observation as BaseObservation

class Observation(BaseObservation):
    rows_preview: list[dict]
    table_schema: dict
    detected_issues: list[str]
    step_count: int

class Action(BaseAction):
    operation: str
    column: str | None = None
    value: str | None = None

class Reward(BaseModel):
    score: float
    reason: str
