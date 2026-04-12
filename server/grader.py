"""
server/grader.py
================
Three distinct per-task graders (EasyGrader, MediumGrader, HardGrader).

Each grader:
  - Has a no-arg constructor (OpenEnv instantiates graders with no args)
  - Implements grade(self, state: dict) -> dict  (required OpenEnv interface)
  - Implements __call__ so grader(state) works too
  - Returns {"score": float, "reason": str, "details": dict}
  - grade() NEVER raises — handles None/empty state gracefully
"""

from __future__ import annotations


def _detect_issues(df) -> list[str]:
    import pandas as pd
    issues: list[str] = []
    if df.isnull().sum().sum() > 0:
        issues.append("Contains missing values")
    if df.duplicated().sum() > 0:
        issues.append("Contains duplicate rows")
    for col in df.columns:
        if "date" in str(col).lower() or "time" in str(col).lower():
            try:
                pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")
            except Exception:
                issues.append(f"Column '{col}' has invalid or mixed date formats")
    for col in df.columns:
        col_str = str(col)
        if " " in col_str or "-" in col_str:
            issues.append(f"Column '{col}' has invalid naming conventions")
    if "UNKNOWN_STATUS" in df.values.flatten().tolist():
        issues.append("Contains invalid categorical data")
    return list(set(issues))


def _generate_schema(df) -> dict:
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def _calculate_final_score(orig_df, curr_df, weights: dict) -> tuple[float, dict]:
    import numpy as np
    null_score      = 1.0 if curr_df.isnull().sum().sum() == 0 else 0.0
    duplicate_score = 1.0 if curr_df.duplicated().sum() == 0 else 0.0
    schema_score    = 1.0 if sum(
        1 for c in curr_df.columns if " " in str(c) or "-" in str(c)
    ) == 0 else 0.0

    date_issues = 0
    for col in curr_df.columns:
        if "date" in str(col).lower() or "time" in str(col).lower():
            try:
                import pandas as pd
                pd.to_datetime(curr_df[col], format="%Y-%m-%d", errors="raise")
            except Exception:
                date_issues += 1
    date_score     = 1.0 if date_issues == 0 else 0.0
    category_score = 1.0 if "UNKNOWN_STATUS" not in curr_df.values.flatten().tolist() else 0.0

    score = (
        null_score      * weights["null"]
        + duplicate_score * weights["duplicate"]
        + schema_score    * weights["schema"]
        + date_score      * weights["date"]
        + category_score  * weights["category"]
    )

    orig_valid = len(orig_df) - orig_df.duplicated().sum()
    curr_valid = len(curr_df) - curr_df.duplicated().sum()
    if curr_valid < orig_valid:
        score -= 0.50

    if np.isnan(score):
        score = 0.5
    score = float(max(0.01, min(0.99, score)))

    details = {
        "null_score":      null_score,
        "duplicate_score": duplicate_score,
        "schema_score":    schema_score,
        "date_score":      date_score,
        "category_score":  category_score,
        "row_penalty":     -0.50 if curr_valid < orig_valid else 0.0,
    }
    return score, details


def _grade_step(grader_instance, prev_df, curr_df, action: dict) -> tuple[float, str]:
    if len(curr_df) < len(prev_df):
        prev_valid = len(prev_df) - prev_df.duplicated().sum()
        curr_valid = len(curr_df) - curr_df.duplicated().sum()
        if prev_valid == curr_valid:
            return 0.25, "Removed exact duplicates"
        if len(curr_df) < len(prev_df) * 0.5:
            return 0.01, "Destructive row deletion (>50%)"
        return 0.05, "Destructive row deletion"

    if prev_df.equals(curr_df) and action.get("operation") != "stop":
        return 0.01, "Repeated/useless action"

    prev_issues = len(_detect_issues(prev_df))
    curr_issues = len(_detect_issues(curr_df))

    if curr_issues < prev_issues:
        reward = 0.25 * (prev_issues - curr_issues)
        return reward, f"Fixed {prev_issues - curr_issues} issue(s)"
    if curr_issues > prev_issues:
        return 0.01, "Introduced new data issues"

    if action.get("operation") == "stop":
        if curr_issues == 0:
            return 0.15, "Valid stop — all issues resolved"
        return 0.01, "Premature stop with unresolved issues"

    return 0.01, "Minor state change (no issue resolved)"


class _BaseGrader:
    """
    Base class for all DataCL task graders.
    OpenEnv calls grader = GraderClass()  then  result = grader.grade(state).
    grade() must NEVER raise — always return a valid score dict.
    """

    DIFFICULTY: str = "easy"
    WEIGHTS: dict = {
        "null": 0.40, "duplicate": 0.30,
        "schema": 0.10, "date": 0.10, "category": 0.10,
    }

    def __init__(self):
        pass

    def grade(self, state: dict) -> dict:
        """
        Grade a completed episode. Accepts None or empty dict gracefully.
        Always returns {"score": float, "reason": str, "details": dict}.
        """
        # Normalise state — never crash on bad input
        if state is None:
            state = {}
        if not isinstance(state, dict):
            try:
                state = dict(state)
            except Exception:
                state = {}

        try:
            raw_score = state.get("score", 0.5)
            score = float(max(0.01, min(0.99, float(raw_score))))
        except (TypeError, ValueError):
            score = 0.5

        return {
            "score": score,
            "reason": self._score_to_reason(score),
            "details": state,
            "grader": self.__class__.__name__,
            "difficulty": self.DIFFICULTY,
        }

    def __call__(self, state: dict) -> dict:
        return self.grade(state)

    def grade_episode(self, orig_df, final_df) -> dict:
        score, details = _calculate_final_score(orig_df, final_df, self.WEIGHTS)
        return {"score": score, "reason": self._score_to_reason(score), "details": details}

    def grade_step(self, prev_df, curr_df, action: dict) -> tuple[float, str]:
        reward, reason = _grade_step(self, prev_df, curr_df, action)
        return float(max(0.01, min(0.99, reward))), reason

    def generate_schema(self, df) -> dict:
        return _generate_schema(df)

    def detect_issues(self, df) -> list[str]:
        return _detect_issues(df)

    def calculate_final_score(self, orig_df, curr_df) -> float:
        score, _ = _calculate_final_score(orig_df, curr_df, self.WEIGHTS)
        return score

    @staticmethod
    def _score_to_reason(score: float) -> str:
        if score >= 0.90:
            return "Excellent: dataset fully cleaned"
        if score >= 0.70:
            return "Good: most issues resolved"
        if score >= 0.50:
            return "Partial: some issues remain"
        if score >= 0.30:
            return "Poor: many issues remain"
        return "Failed: minimal progress"


class EasyGrader(_BaseGrader):
    DIFFICULTY = "easy"
    WEIGHTS = {
        "null": 0.60, "duplicate": 0.20,
        "schema": 0.10, "date": 0.05, "category": 0.05,
    }


class MediumGrader(_BaseGrader):
    DIFFICULTY = "medium"
    WEIGHTS = {
        "null": 0.20, "duplicate": 0.35,
        "schema": 0.10, "date": 0.25, "category": 0.10,
    }


class HardGrader(_BaseGrader):
    DIFFICULTY = "hard"
    WEIGHTS = {
        "null": 0.25, "duplicate": 0.25,
        "schema": 0.20, "date": 0.15, "category": 0.15,
    }


class Grader(_BaseGrader):
    """Legacy alias — keeps environment.py working without changes."""
    _WEIGHTS_BY_DIFF = {
        "easy":   EasyGrader.WEIGHTS,
        "medium": MediumGrader.WEIGHTS,
        "hard":   HardGrader.WEIGHTS,
    }

    def __init__(self, task_difficulty: str = "easy"):
        super().__init__()
        self.DIFFICULTY = task_difficulty
        self.WEIGHTS = self._WEIGHTS_BY_DIFF.get(task_difficulty, EasyGrader.WEIGHTS)