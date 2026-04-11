from pydantic import BaseModel, ConfigDict, Field

class Observation(BaseModel):
    rows_preview: list[dict]
    table_schema: dict
    detected_issues: list[str]
    step_count: int
class Action(BaseModel):
    operation: str
    column: str | None = None
    value: str | None = None

class Reward(BaseModel):
    score: float
    reason: str
