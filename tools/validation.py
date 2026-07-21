"""Deterministic before/after validation and rollback policy."""

from __future__ import annotations

import pandas as pd

from models.cleaning_plan import CleaningPlan
from models.policy import PreprocessingPolicy
from models.validation import ValidationIssue, ValidationResult
from tools.policy import column_is_protected


def _total_missing(dataframe: pd.DataFrame) -> int:
    return int(dataframe.isna().sum().sum())


def _duplicate_count(dataframe: pd.DataFrame) -> int:
    return int(dataframe.duplicated().sum())


def _stable_value_counts(series: pd.Series) -> dict[object, int]:
    marker = ("__MAS_MISSING__",)
    values = [marker if pd.isna(value) else value for value in series.tolist()]
    return pd.Series(values, dtype="object").value_counts(dropna=False).to_dict()


def validate_preprocessing(
    before: pd.DataFrame,
    after: pd.DataFrame,
    plan: CleaningPlan,
    policy: PreprocessingPolicy | None = None,
) -> ValidationResult:
    """Validate a candidate dataframe against conservative safety rules.

    A result is invalid when preprocessing unexpectedly changes the schema,
    removes non-duplicate rows, adds rows, or introduces unresolved missing
    values. Warnings describe ineffective but non-destructive steps.
    """
    issues: list[ValidationIssue] = []
    operations = {step.operation for step in plan.steps}
    has_drop_duplicates = "drop_duplicates" in operations

    rows_before = len(before)
    rows_after = len(after)
    missing_before = _total_missing(before)
    missing_after = _total_missing(after)
    duplicates_before = _duplicate_count(before)
    duplicates_after = _duplicate_count(after)

    if list(before.columns) != list(after.columns):
        issues.append(
            ValidationIssue(
                code="schema_changed",
                severity="error",
                message="Column names or order changed unexpectedly.",
            )
        )

    if rows_after > rows_before:
        issues.append(
            ValidationIssue(
                code="rows_added",
                severity="error",
                message=f"Preprocessing added {rows_after - rows_before} rows.",
            )
        )

    removed_rows = rows_before - rows_after
    if removed_rows > 0 and not has_drop_duplicates:
        issues.append(
            ValidationIssue(
                code="unexpected_row_loss",
                severity="error",
                message=f"Preprocessing removed {removed_rows} rows without approval.",
            )
        )
    elif removed_rows > duplicates_before:
        issues.append(
            ValidationIssue(
                code="excessive_row_loss",
                severity="error",
                message=(
                    f"Preprocessing removed {removed_rows} rows although only "
                    f"{duplicates_before} duplicate rows existed."
                ),
            )
        )

    if rows_before > 0 and rows_after == 0:
        issues.append(
            ValidationIssue(
                code="empty_result",
                severity="error",
                message="Preprocessing produced an empty dataset.",
            )
        )

    if missing_after > missing_before:
        issues.append(
            ValidationIssue(
                code="new_missing_values",
                severity="error",
                message=(
                    f"Preprocessing introduced {missing_after - missing_before} "
                    "unresolved missing values."
                ),
            )
        )

    for column in before.columns:
        if not column_is_protected(str(column), policy):
            continue
        if column not in after.columns:
            continue
        before_counts = _stable_value_counts(before[column])
        after_counts = _stable_value_counts(after[column])
        modified = any(
            count > before_counts.get(value, 0)
            for value, count in after_counts.items()
        )
        if modified:
            issues.append(
                ValidationIssue(
                    code="protected_column_modified",
                    severity="error",
                    column=column,
                    message=(
                        f"Protected identifier column {column!r} was modified. "
                        "Identifier-like columns are not automatically cleaned."
                    ),
                )
            )

    if has_drop_duplicates and duplicates_after > 0:
        issues.append(
            ValidationIssue(
                code="duplicates_remain",
                severity="warning",
                message=f"{duplicates_after} duplicate rows remain after cleaning.",
            )
        )

    for step in plan.steps:
        if step.operation not in {"fill_mean", "fill_median", "fill_mode"}:
            continue
        if not step.column or step.column not in before.columns:
            continue
        before_count = int(before[step.column].isna().sum())
        after_count = int(after[step.column].isna().sum())
        if before_count > 0 and after_count >= before_count:
            issues.append(
                ValidationIssue(
                    code="fill_ineffective",
                    severity="warning",
                    column=step.column,
                    message=(
                        f"Fill operation did not reduce missing values in "
                        f"{step.column!r}."
                    ),
                )
            )

    has_error = any(issue.severity == "error" for issue in issues)
    return ValidationResult(
        valid=not has_error,
        recommend_rollback=has_error,
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=len(before.columns),
        columns_after=len(after.columns),
        missing_before=missing_before,
        missing_after=missing_after,
        duplicates_before=duplicates_before,
        duplicates_after=duplicates_after,
        issues=issues,
    )


def apply_rollback_policy(
    original: pd.DataFrame,
    candidate: pd.DataFrame,
    validation: ValidationResult,
) -> tuple[pd.DataFrame, bool]:
    """Return the committed candidate or a safe copy of the original.

    Returns ``(dataframe, rolled_back)``.
    """
    if validation.recommend_rollback:
        return original.copy(deep=True), True
    return candidate.copy(deep=True), False
