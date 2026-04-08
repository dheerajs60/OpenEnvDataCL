import pandas as pd
import numpy as np


class Grader:
    def __init__(self, task_difficulty: str):
        self.task_difficulty = task_difficulty

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
                return 0.25, "Removed exact duplicates"
            else:
                if len(curr_df) < len(prev_df) * 0.5:
                    return -0.50, "Destructive row deletion (>50%)"
                return -0.30, "Destructive row deletion"

        # repeated no-op
        if prev_df.equals(curr_df) and action.get("operation") != "stop":
            return -0.10, "Repeated/useless action"

        prev_issues = len(self.detect_issues(prev_df))
        curr_issues = len(self.detect_issues(curr_df))

        if curr_issues < prev_issues:
            return 0.25 * (prev_issues - curr_issues), "Fixed data issue(s)"

        elif curr_issues > prev_issues:
            return -0.20, "Introduced new data issues"

        if action.get("operation") == "stop":
            if curr_issues == 0:
                return 0.10, "Valid stop action"
            else:
                return -0.20, "Premature stop with unresolved issues"

        return 0.01, "Minor state change"

    def calculate_final_score(
        self,
        orig_df: pd.DataFrame,
        curr_df: pd.DataFrame
    ) -> float:

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

        score = (
            null_score * 0.25
            + duplicate_score * 0.25
            + schema_score * 0.20
            + date_score * 0.15
            + category_score * 0.15
        )

        # destructive final penalty
        orig_valid = len(orig_df) - orig_df.duplicated().sum()
        curr_valid = len(curr_df) - curr_df.duplicated().sum()

        if curr_valid < orig_valid:
            score -= 0.50

        # ✅ strict open interval required by validator
        score = max(0.01, min(0.99, float(score)))

        return score