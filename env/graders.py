import pandas as pd
import numpy as np


try:
    from openenv.core.rubrics.base import Rubric
except ImportError:
    # Fallback for environments where openenv-core is not installed
    class Rubric:
        def __init__(self): pass
        def __call__(self, action, observation): return self.forward(action, observation)

class Grader(Rubric):
    def __init__(self, task_difficulty: str = "easy"):
        super().__init__()
        self.task_difficulty = task_difficulty

    def forward(self, action, observation) -> float:
        """Standard OpenEnv rubric entry point."""
        # Derive a basic score from detected issues in the observation
        issues = observation.detected_issues if hasattr(observation, "detected_issues") else []
        if not issues:
            return 0.99
        # Basic inverse relationship: more issues = lower score
        score = max(0.01, 0.99 - (len(issues) * 0.1))
        return float(score)

    def grade(self, task_id: str, state: dict) -> float:
        score = state.get("score", 0.5)
        return max(0.01, min(0.99, float(score)))

    def generate_schema(self, df: pd.DataFrame) -> dict:
        return {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    def detect_issues(self, df: pd.DataFrame) -> list[str]:
        issues = []

        if df.isnull().sum().sum() > 0:
            issues.append("Contains missing values")

        if df.duplicated().sum() > 0:
            issues.append("Contains duplicate rows")

        for col in df.columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                try:
                    pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")
                except Exception:
                    issues.append(
                        f"Column '{col}' has invalid or mixed date formats"
                    )

        for col in df.columns:
            col_str = str(col)
            if " " in col_str or "-" in col_str:
                issues.append(
                    f"Column '{col}' has invalid naming conventions"
                )

        if "UNKNOWN_STATUS" in df.values.flatten().tolist():
            issues.append("Contains invalid categorical data")

        return list(set(issues))

    def grade_step(
        self,
        prev_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        action: dict
    ) -> tuple[float, str]:

        # duplicate-safe deletion reward
        if len(curr_df) < len(prev_df):
            prev_valid = len(prev_df) - prev_df.duplicated().sum()
            curr_valid = len(curr_df) - curr_df.duplicated().sum()

            if prev_valid == curr_valid:
                return 0.20, "Removed exact duplicates"
            else:
                if len(curr_df) < len(prev_df) * 0.5:
                    return 0.01, "Destructive row deletion (>50%)"
                return 0.05, "Destructive row deletion"

        # repeated no-op
        if prev_df.equals(curr_df) and action.get("operation") != "stop":
            return 0.02, "Repeated/useless action"

        prev_issues = len(self.detect_issues(prev_df))
        curr_issues = len(self.detect_issues(curr_df))

        if curr_issues < prev_issues:
            return 0.20 * (prev_issues - curr_issues), "Fixed data issue(s)"

        elif curr_issues > prev_issues:
            return 0.02, "Introduced new data issues"

        if action.get("operation") == "stop":
            if curr_issues == 0:
                return 0.15, "Valid stop action"
            else:
                return 0.01, "Premature stop with unresolved issues"

        return 0.01, "Minor state change"

    def calculate_final_score(
        self,
        orig_df: pd.DataFrame,
        curr_df: pd.DataFrame
    ) -> float:

        # ===== issue sub-scores =====
        null_score = 1.0 if curr_df.isnull().sum().sum() == 0 else 0.0
        duplicate_score = 1.0 if curr_df.duplicated().sum() == 0 else 0.0

        schema_score = (
            1.0
            if sum(
                1
                for c in curr_df.columns
                if " " in str(c) or "-" in str(c)
            ) == 0
            else 0.0
        )

        date_issues_curr = 0
        for col in curr_df.columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                try:
                    pd.to_datetime(
                        curr_df[col],
                        format="%Y-%m-%d",
                        errors="raise"
                    )
                except Exception:
                    date_issues_curr += 1

        date_score = 1.0 if date_issues_curr == 0 else 0.0

        category_score = (
            1.0
            if "UNKNOWN_STATUS" not in curr_df.values.flatten().tolist()
            else 0.0
        )

        # ===== difficulty-aware grading =====
        if self.task_difficulty == "easy":
            weights = {
                "null": 0.60,
                "duplicate": 0.20,
                "schema": 0.10,
                "date": 0.05,
                "category": 0.05,
            }

        elif self.task_difficulty == "medium":
            weights = {
                "null": 0.20,
                "duplicate": 0.35,
                "schema": 0.10,
                "date": 0.25,
                "category": 0.10,
            }

        else:  # hard
            weights = {
                "null": 0.25,
                "duplicate": 0.25,
                "schema": 0.20,
                "date": 0.15,
                "category": 0.15,
            }

        score = (
            null_score * weights["null"]
            + duplicate_score * weights["duplicate"]
            + schema_score * weights["schema"]
            + date_score * weights["date"]
            + category_score * weights["category"]
        )

        # destructive final penalty
        orig_valid = len(orig_df) - orig_df.duplicated().sum()
        curr_valid = len(curr_df) - curr_df.duplicated().sum()

        if curr_valid < orig_valid:
            score -= 0.50

        # NaN safety
        score = float(score)
        if np.isnan(score):
            score = 0.5

        # strict validator-safe open interval
        score = max(0.01, min(0.99, score))

        return score