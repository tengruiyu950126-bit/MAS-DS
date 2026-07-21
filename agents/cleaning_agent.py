"""Cleaning-plan experts."""

from __future__ import annotations

from typing import Protocol

import pandas as pd
from pandas.api.types import is_numeric_dtype

from models.cleaning_plan import CleaningPlan, CleaningStep
from models.policy import PreprocessingPolicy
from models.profile import DataProfile
from tools.policy import column_is_protected, operation_allowed, outlier_action_for_column
from tools.profiler import profile_dataframe
from tools.quality import (
    can_parse_numeric_text_losslessly,
    count_iqr_outliers,
    has_case_inconsistency,
    has_category_typo_issue,
    has_whitespace_issue,
    needs_robust_numeric_parsing,
    parse_numeric_text_series,
)


class CleaningPlanner(Protocol):
    """Interface shared by rule-based and model-backed cleaning experts."""

    name: str

    def propose(
        self,
        dataframe: pd.DataFrame,
        profile: DataProfile | None = None,
    ) -> CleaningPlan: ...


class RuleBasedCleaningAgent:
    """Free and reproducible baseline that proposes safe cleaning steps.

    This expert intentionally handles only high-confidence operations. A local
    LLM expert can later be compared against it without changing the plan
    contract or executor.
    """

    name = "rule_based_cleaning_expert"

    def __init__(self, policy: PreprocessingPolicy | None = None) -> None:
        self._policy = policy or PreprocessingPolicy()

    def _append_if_allowed(self, steps: list[CleaningStep], step: CleaningStep) -> None:
        if operation_allowed(step, self._policy):
            steps.append(step)

    def propose(
        self,
        dataframe: pd.DataFrame,
        profile: DataProfile | None = None,
    ) -> CleaningPlan:
        profile = profile or profile_dataframe(dataframe)
        steps: list[CleaningStep] = []

        if profile.duplicate_rows > 0:
            self._append_if_allowed(
                steps,
                CleaningStep(
                    operation="drop_duplicates",
                    reason=(
                        f"Detected {profile.duplicate_rows} exact duplicate rows."
                    ),
                    confidence=1.0,
                )
            )

        for column, column_profile in profile.column_profiles.items():
            if column_is_protected(column, self._policy):
                if column_profile.missing_count > 0:
                    steps.append(
                        CleaningStep(
                            column=column,
                            operation="leave_unchanged",
                            reason=(
                                "Column is protected by preprocessing policy; "
                                "automatic cleaning is disabled for this column."
                            ),
                            confidence=1.0,
                        )
                    )
                continue

            parsed_as_numeric = False
            if can_parse_numeric_text_losslessly(dataframe[column]):
                parsed_as_numeric = True
                operation = (
                    "parse_numeric_text"
                    if needs_robust_numeric_parsing(dataframe[column])
                    else "convert_numeric"
                )
                self._append_if_allowed(
                    steps,
                    CleaningStep(
                        column=column,
                        operation=operation,
                        reason=(
                            f"Column {column!r} contains numeric values stored "
                            "as text; convert it before downstream cleaning."
                        ),
                        confidence=0.95,
                    )
                )
            else:
                if has_whitespace_issue(dataframe[column]):
                    self._append_if_allowed(
                        steps,
                        CleaningStep(
                            column=column,
                            operation="strip_whitespace",
                            reason=(
                                f"Column {column!r} contains leading or trailing "
                                "whitespace in text values."
                            ),
                            confidence=0.9,
                        )
                    )
                if has_case_inconsistency(dataframe[column]):
                    self._append_if_allowed(
                        steps,
                        CleaningStep(
                            column=column,
                            operation="normalize_case",
                            reason=(
                                f"Column {column!r} contains case-inconsistent "
                                "categorical labels."
                            ),
                            confidence=0.85,
                        )
                    )
                if has_category_typo_issue(dataframe[column]):
                    self._append_if_allowed(
                        steps,
                        CleaningStep(
                            column=column,
                            operation="normalize_category_typos",
                            reason=(
                                f"Column {column!r} contains rare category labels "
                                "that closely match more frequent labels."
                            ),
                            confidence=0.78,
                        )
                    )

            numeric_for_outliers = (
                parse_numeric_text_series(dataframe[column])
                if parsed_as_numeric
                else dataframe[column]
            )
            outlier_count = count_iqr_outliers(numeric_for_outliers)
            if (
                outlier_count > 0
                and outlier_action_for_column(column, self._policy) == "flag"
            ):
                self._append_if_allowed(
                    steps,
                    CleaningStep(
                        column=column,
                        operation="flag_outliers_iqr",
                        reason=(
                            f"Detected {outlier_count} potential numeric outlier"
                            f"{'s' if outlier_count != 1 else ''} using the IQR "
                            "rule. This step is advisory and does not modify values."
                        ),
                        confidence=0.75,
                    )
                )

            if column_profile.missing_count == 0:
                continue

            if column_profile.unique_count == 0:
                self._append_if_allowed(
                    steps,
                    CleaningStep(
                        column=column,
                        operation="leave_unchanged",
                        reason=(
                            "Column is entirely missing; automatic imputation "
                            "would invent unsupported data."
                        ),
                        confidence=1.0,
                    )
                )
                continue

            if parsed_as_numeric or is_numeric_dtype(dataframe[column]):
                operation = "fill_median"
                reason = (
                    f"Fill {column_profile.missing_count} missing numeric values "
                    "with the median, which is robust to outliers."
                )
                confidence = 0.95
            else:
                operation = "fill_mode"
                reason = (
                    f"Fill {column_profile.missing_count} missing categorical "
                    "values with the most frequent observed value."
                )
                confidence = 0.85

            self._append_if_allowed(
                steps,
                CleaningStep(
                    column=column,
                    operation=operation,
                    reason=reason,
                    confidence=confidence,
                )
            )

        return CleaningPlan(steps=steps)
