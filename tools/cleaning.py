"""Whitelisted, deterministic dataframe transformations.

The executor never evaluates model-generated code. Agents may only select an
operation declared in :mod:`models.cleaning_plan`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype

from models.cleaning_plan import CleaningPlan, CleaningStep
from tools.quality import (
    has_iqr_outliers,
    normalize_case_series,
    normalize_category_typos_series,
    parse_numeric_text_series,
    strip_whitespace_series,
)


class CleaningExecutionError(ValueError):
    """Raised when a cleaning plan cannot be executed safely."""


@dataclass(frozen=True)
class ExecutionRecord:
    """Auditable before/after measurements for one cleaning step."""

    step_index: int
    operation: str
    column: str | None
    rows_before: int
    rows_after: int
    missing_before: int | None
    missing_after: int | None


def _require_column(dataframe: pd.DataFrame, step: CleaningStep) -> str:
    if not step.column:
        raise CleaningExecutionError(
            f"Operation '{step.operation}' requires a column."
        )
    if step.column not in dataframe.columns:
        raise CleaningExecutionError(f"Unknown column: {step.column!r}.")
    return step.column


def _column_missing(dataframe: pd.DataFrame, column: str | None) -> int | None:
    if column is None or column not in dataframe.columns:
        return None
    return int(dataframe[column].isna().sum())


def execute_step(dataframe: pd.DataFrame, step: CleaningStep) -> pd.DataFrame:
    """Execute one whitelisted step on a defensive copy of ``dataframe``."""
    result = dataframe.copy(deep=True)

    if step.operation == "leave_unchanged":
        return result

    if step.operation == "drop_duplicates":
        return result.drop_duplicates(ignore_index=True)

    column = _require_column(result, step)

    if step.operation in {"fill_mean", "fill_median"}:
        if not is_numeric_dtype(result[column]):
            raise CleaningExecutionError(
                f"Operation '{step.operation}' requires a numeric column; "
                f"{column!r} has dtype {result[column].dtype}."
            )
        value = (
            result[column].mean()
            if step.operation == "fill_mean"
            else result[column].median()
        )
        if pd.isna(value):
            raise CleaningExecutionError(
                f"Cannot calculate a fill value for all-missing column {column!r}."
            )
        result[column] = result[column].fillna(value)
        return result

    if step.operation == "fill_mode":
        modes = result[column].mode(dropna=True)
        if modes.empty:
            raise CleaningExecutionError(
                f"Cannot calculate a mode for all-missing column {column!r}."
            )
        result[column] = result[column].fillna(modes.iloc[0])
        return result

    if step.operation == "convert_numeric":
        result[column] = pd.to_numeric(result[column], errors="coerce")
        return result

    if step.operation == "convert_datetime":
        result[column] = pd.to_datetime(result[column], errors="coerce")
        return result

    if step.operation == "parse_numeric_text":
        result[column] = parse_numeric_text_series(result[column])
        return result

    if step.operation == "strip_whitespace":
        result[column] = strip_whitespace_series(result[column])
        return result

    if step.operation == "normalize_case":
        result[column] = normalize_case_series(result[column])
        return result

    if step.operation == "normalize_category_typos":
        result[column] = normalize_category_typos_series(result[column])
        return result

    if step.operation == "flag_outliers_iqr":
        if not is_numeric_dtype(result[column]):
            raise CleaningExecutionError(
                f"Operation '{step.operation}' requires a numeric column; "
                f"{column!r} has dtype {result[column].dtype}."
            )
        # This is intentionally non-mutating. It records an auditable warning
        # in the plan without changing potentially valid extreme values.
        has_iqr_outliers(result[column])
        return result

    # Pydantic normally prevents this branch; it remains as a defensive guard.
    raise CleaningExecutionError(f"Unsupported operation: {step.operation!r}.")


def execute_plan(
    dataframe: pd.DataFrame,
    plan: CleaningPlan,
) -> tuple[pd.DataFrame, list[ExecutionRecord]]:
    """Execute a complete plan atomically and return its audit records.

    The caller receives a new dataframe only after every step succeeds. If a
    step fails, an exception is raised and the caller's original dataframe is
    unchanged.
    """
    working = dataframe.copy(deep=True)
    records: list[ExecutionRecord] = []

    for index, step in enumerate(plan.steps):
        rows_before = len(working)
        missing_before = _column_missing(working, step.column)
        updated = execute_step(working, step)
        records.append(
            ExecutionRecord(
                step_index=index,
                operation=step.operation,
                column=step.column,
                rows_before=rows_before,
                rows_after=len(updated),
                missing_before=missing_before,
                missing_after=_column_missing(updated, step.column),
            )
        )
        working = updated

    return working, records
