import random
import pandas as pd
from typing import Dict, Any

from server.tasks import TASKS
from server.grader import Grader
from models import Observation, Action, Reward


class DataCleanerEnv:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.orig_df: pd.DataFrame | None = None
        self.task_difficulty: str | None = None
        self.grader: Grader | None = None
        self.step_count: int = 0
        self.max_steps: int = 20
        self.done: bool = False
        self.last_action: dict | None = None

    def reset(self, difficulty: str = None) -> Observation:
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

    def state(self) -> Dict[str, Any]:
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

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            obs = self._get_obs()
            return (
                obs,
                Reward(score=0.01, reason="Episode already done"),
                True,
                {
                    "msg": "Episode already done",
                    "score": 0.5
                }
            )

        prev_df = self.df.copy()
        self.step_count += 1

        op = action.operation
        col = action.column
        val = action.value

        invalid_action = False

        # repeated action penalty
        repeat_penalty = 0.0
        current_action = action.model_dump()

        if self.last_action == current_action:
            repeat_penalty = 0.0  # penalties handled via small positive rewards in grader

        self.last_action = current_action

        try:
            # ✅ improved dtype-aware fill_missing
            if op == "fill_missing" and col in self.df.columns and val is not None:
                try:
                    col_dtype = self.df[col].dtype

                    if pd.api.types.is_numeric_dtype(col_dtype):
                        cast_val = float(val)
                    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                        cast_val = pd.to_datetime(val, errors="coerce")
                    else:
                        cast_val = str(val)

                    self.df[col] = self.df[col].fillna(cast_val)

                except Exception:
                    self.df[col] = self.df[col].fillna(str(val))

            elif op == "remove_duplicates":
                self.df = self.df.drop_duplicates()

            elif op == "standardize_date" and col in self.df.columns:
                self.df[col] = (
                    pd.to_datetime(self.df[col], errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                )

            elif op == "normalize_text" and col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.title()
                )

            elif op == "rename_column" and col in self.df.columns and val is not None:
                self.df = self.df.rename(columns={col: val})

            elif op == "fix_category" and col in self.df.columns and val is not None:
                if ":" in val:
                    old_c, new_c = val.split(":", 1)
                    self.df[col] = self.df[col].replace(old_c, new_c)
                else:
                    invalid_action = True

            elif op == "stop":
                self.done = True

            else:
                invalid_action = True

        except Exception:
            invalid_action = True

        if self.step_count >= self.max_steps:
            self.done = True

        reward_val, reason = self.grader.grade_step(
            prev_df, self.df, current_action
        )

        reward_val += repeat_penalty
        
        # ensure reward is always in valid (0, 1) range
        reward_val = float(max(0.01, min(0.99, reward_val)))

        reward = Reward(score=reward_val, reason=reason)

        info = {"score": 0.5}

        if self.done:
            final_score = self.grader.calculate_final_score(
                self.orig_df,
                self.df
            )
            info["score"] = float(max(0.01, min(0.99, final_score)))

        obs = self._get_obs()
        return obs, reward, self.done, info