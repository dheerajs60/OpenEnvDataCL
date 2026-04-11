import random
import pandas as pd
from typing import Dict, Any, Optional

from env.tasks import TASKS
from env.grader import Grader
from env.models import Observation, Action
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata


class DataCleanerEnv(Environment[Action, Observation, Any]):
    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame | None = None
        self.orig_df: pd.DataFrame | None = None
        self.task_difficulty: str | None = None
        self.grader: Grader | None = None
        self.step_count: int = 0
        self.max_steps: int = 20
        self.done: bool = False
        self.last_action: Any | None = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="data_cleaner_env",
            description="Real-world data cleaning benchmark for AI agents",
            version="1.0.0"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str = None,
        **kwargs: Any,
    ) -> Observation:
        if difficulty not in TASKS:
            difficulty = random.choice(list(TASKS.keys()))

        self.task_difficulty = difficulty
        self.orig_df = TASKS[difficulty]()
        self.df = self.orig_df.copy()

        self.grader = Grader(self.task_difficulty)
        self.step_count = 0
        self.done = False
        self.last_action = None

        return self._get_obs()

    def _get_obs(self) -> Observation:
        schema = self.grader.generate_schema(self.df)
        issues = self.grader.detect_issues(self.df)

        preview_df = self.df.head(5).where(pd.notnull(self.df), None)
        rows_preview = preview_df.to_dict(orient="records")

        return Observation(
            rows_preview=rows_preview,
            table_schema=schema,
            detected_issues=issues,
            step_count=self.step_count
        )

    @property
    def state(self) -> Any:
        score = 0.5
        if (
            self.orig_df is not None
            and self.df is not None
            and self.grader is not None
        ):
            orig_issues = len(self.grader.detect_issues(self.orig_df))
            curr_issues = len(self.grader.detect_issues(self.df))

            if orig_issues > 0:
                progress = 1 - (curr_issues / orig_issues)
                score = 0.1 + 0.8 * progress
            else:
                score = 0.9

        score = float(max(0.01, min(0.99, score)))

        return {
            "task_difficulty": self.task_difficulty,
            "step_count": self.step_count,
            "done": self.done,
            "df_shape": self.df.shape if self.df is not None else None,
            "score": score,
        }

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if self.done:
            obs = self._get_obs()
            obs.reward = 0.01
            obs.done = True
            return obs

        prev_df = self.df.copy()
        self.step_count += 1

        op = action.operation
        col = action.column
        val = action.value

        current_action = action.model_dump()
        self.last_action = current_action

        try:
            if op == "fill_missing" and col in self.df.columns and val is not None:
                col_dtype = self.df[col].dtype
                try:
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        cast_val = float(val)
                    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                        cast_val = pd.to_datetime(val, errors="coerce")
                    else:
                        cast_val = str(val)
                    self.df[col] = self.df[col].fillna(cast_val)
                except:
                    self.df[col] = self.df[col].fillna(str(val))

            elif op == "remove_duplicates":
                self.df = self.df.drop_duplicates()

            elif op == "standardize_date" and col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce").dt.strftime("%Y-%m-%d")

            elif op == "normalize_text" and col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.title()

            elif op == "rename_column" and col in self.df.columns and val is not None:
                self.df = self.df.rename(columns={col: val})

            elif op == "fix_category" and col in self.df.columns and val is not None:
                if ":" in val:
                    old_c, new_c = val.split(":", 1)
                    self.df[col] = self.df[col].replace(old_c, new_c)

            elif op == "stop":
                self.done = True

        except Exception:
            pass

        if self.step_count >= self.max_steps:
            self.done = True

        reward_val, reason = self.grader.grade_step(prev_df, self.df, current_action)
        reward_val = float(max(0.01, min(0.99, reward_val)))

        if self.done:
            final_score = self.grader.calculate_final_score(self.orig_df, self.df)
            # Maybe store final score in info if we had it, but for now we rely on reward
            # Rubric protocol typically uses the reward field for current step evaluation

        obs = self._get_obs()
        obs.reward = reward_val
        obs.done = self.done
        return obs