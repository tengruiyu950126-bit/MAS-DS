"""Before/after change reporting for cleaning plans.

The functions in this module produce human-readable audit rows. They do not
change the execution path; instead, they replay the already approved plan with
the same whitelisted executor and record the visible changes caused by each
step.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from models.cleaning_plan import CleaningPlan, CleaningStep
from tools.cleaning import execute_step


MISSING_DISPLAY = "<missing>"
NO_COLUMN_DISPLAY = "*"


def _is_missing(value: Any) -> bool:
    """Return True for scalar missing values without treating lists as missing."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _values_equal(left: Any, right: Any) -> bool:
    """Compare values while treating two missing values as equal."""
    if _is_missing(left) and _is_missing(right):
        return True
    return left == right


def _format_value(value: Any, max_length: int = 120) -> str:
    """Format a dataframe value for compact display in the UI."""
    if _is_missing(value):
        return MISSING_DISPLAY
    text = str(value)
    if len(text) > max_length:
        return f"{text[: max_length - 1]}…"
    return text


def _format_row(row: pd.Series, max_columns: int = 8) -> str:
    """Format a removed/added row without making the audit table too wide."""
    items = []
    for column, value in row.iloc[:max_columns].items():
        items.append(f"{column}={_format_value(value, max_length=40)}")
    if len(row) > max_columns:
        items.append("...")
    return "{ " + ", ".join(items) + " }"


def _base_audit_row(
    step_index: int,
    step: CleaningStep,
    change_type: str,
    row: int | str,
    column: str | None,
    before: str,
    after: str,
) -> dict[str, object]:
    return {
        "step": step_index + 1,
        "operation": step.operation,
        "column": column or NO_COLUMN_DISPLAY,
        "change_type": change_type,
        "row": row,
        "before": before,
        "after": after,
        "reason": step.reason,
        "confidence": round(step.confidence, 3),
    }


def _removed_duplicate_rows(
    before: pd.DataFrame,
    after: pd.DataFrame,
    step_index: int,
    step: CleaningStep,
    max_changes: int,
) -> list[dict[str, object]]:
    """Return row-removal audit entries for a drop-duplicates step."""
    if len(after) >= len(before):
        return []

    duplicate_mask = before.duplicated(keep="first")
    removed_indices = list(before.index[duplicate_mask])

    # Defensive fallback: if pandas duplicate detection cannot explain all row
    # removals, still report the row-count delta without guessing individual
    # cell edits caused by positional shifts.
    expected_removed = len(before) - len(after)
    if len(removed_indices) < expected_removed:
        rows = [
            _base_audit_row(
                step_index,
                step,
                "row_count_decreased",
                "*",
                None,
                str(len(before)),
                str(len(after)),
            )
        ]
        return rows[:max_changes]

    rows = []
    for row_index in removed_indices[:max_changes]:
        rows.append(
            _base_audit_row(
                step_index,
                step,
                "row_removed",
                int(row_index),
                None,
                _format_row(before.loc[row_index]),
                "<removed>",
            )
        )
    return rows


def _dtype_change_row(
    before: pd.DataFrame,
    after: pd.DataFrame,
    step_index: int,
    step: CleaningStep,
) -> dict[str, object] | None:
    if not step.column or step.column not in before.columns or step.column not in after.columns:
        return None
    before_dtype = str(before[step.column].dtype)
    after_dtype = str(after[step.column].dtype)
    if before_dtype == after_dtype:
        return None
    return _base_audit_row(
        step_index,
        step,
        "dtype_changed",
        "*",
        step.column,
        before_dtype,
        after_dtype,
    )


def _cell_change_rows(
    before: pd.DataFrame,
    after: pd.DataFrame,
    step_index: int,
    step: CleaningStep,
    max_changes: int,
) -> Iterable[dict[str, object]]:
    """Yield cell-level changes for non-row-changing operations."""
    if step.column:
        columns = [step.column] if step.column in before.columns and step.column in after.columns else []
    else:
        columns = [column for column in before.columns if column in after.columns]

    comparable_rows = min(len(before), len(after))
    emitted = 0
    for column in columns:
        before_series = before[column].reset_index(drop=True)
        after_series = after[column].reset_index(drop=True)
        for row_position in range(comparable_rows):
            before_value = before_series.iloc[row_position]
            after_value = after_series.iloc[row_position]
            if _values_equal(before_value, after_value):
                continue
            yield _base_audit_row(
                step_index,
                step,
                "value_changed",
                row_position,
                column,
                _format_value(before_value),
                _format_value(after_value),
            )
            emitted += 1
            if emitted >= max_changes:
                return


def plan_diff_frame(
    dataframe: pd.DataFrame,
    plan: CleaningPlan,
    *,
    max_changes_per_step: int = 500,
) -> pd.DataFrame:
    """Replay ``plan`` and return a row-level/cell-level audit dataframe.

    The returned table is designed for user inspection:

    - `value_changed` rows show exact cell before/after values.
    - `dtype_changed` rows show type conversions even when values look similar.
    - `row_removed` rows explain duplicate removals without creating misleading
      positional cell diffs after rows shift.

    Non-mutating operations such as `flag_outliers_iqr` naturally return no
    rows unless they change dtype or values, which they should not.
    """
    working = dataframe.copy(deep=True)
    rows: list[dict[str, object]] = []

    for step_index, step in enumerate(plan.steps):
        before_step = working
        after_step = execute_step(before_step, step)

        dtype_row = _dtype_change_row(before_step, after_step, step_index, step)
        if dtype_row is not None:
            rows.append(dtype_row)

        if step.operation == "drop_duplicates":
            rows.extend(
                _removed_duplicate_rows(
                    before_step,
                    after_step,
                    step_index,
                    step,
                    max_changes_per_step,
                )
            )
        else:
            rows.extend(
                _cell_change_rows(
                    before_step,
                    after_step,
                    step_index,
                    step,
                    max_changes_per_step,
                )
            )

        working = after_step

    return pd.DataFrame(
        rows,
        columns=[
            "step",
            "operation",
            "column",
            "change_type",
            "row",
            "before",
            "after",
            "reason",
            "confidence",
        ],
    )


def diff_summary_frame(diff: pd.DataFrame) -> pd.DataFrame:
    """Summarize a detailed diff table by operation, column, and change type."""
    if diff.empty:
        return pd.DataFrame(
            columns=["operation", "column", "change_type", "changes"]
        )
    return (
        diff.groupby(["operation", "column", "change_type"], dropna=False)
        .size()
        .reset_index(name="changes")
        .sort_values(["operation", "column", "change_type"])
        .reset_index(drop=True)
    )
