"""Reproducible corruption injection with cell-level ground truth."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


ROW_ID_COLUMN = "__mas_row_id__"
CorruptionType = Literal[
    "missing_value",
    "duplicate_row",
    "numeric_type",
    "datetime_type",
]


@dataclass(frozen=True)
class CorruptionRecord:
    error_type: CorruptionType
    row_id: int
    column: str | None
    original_value: Any
    corrupted_value: Any
    expected_behavior: Literal["restore_value", "remove_duplicate"]


@dataclass(frozen=True)
class CorruptionResult:
    clean: pd.DataFrame
    corrupted: pd.DataFrame
    records: list[CorruptionRecord]
    error_type: CorruptionType
    seed: int


def _validate_fraction(fraction: float) -> None:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1.")


def _prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError("Cannot corrupt an empty dataframe.")
    if ROW_ID_COLUMN in dataframe.columns:
        raise ValueError(f"Reserved column already exists: {ROW_ID_COLUMN!r}.")
    result = dataframe.copy(deep=True).reset_index(drop=True)
    result.insert(0, ROW_ID_COLUMN, range(len(result)))
    return result


def _sample_cells(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    fraction: float,
    rng: random.Random,
) -> list[tuple[int, str]]:
    candidates = [
        (row_index, column)
        for column in columns
        for row_index in dataframe.index
        if pd.notna(dataframe.at[row_index, column])
    ]
    if not candidates:
        raise ValueError("No eligible non-missing cells for this corruption.")
    count = max(1, round(len(candidates) * fraction))
    return rng.sample(candidates, min(count, len(candidates)))


def inject_corruption(
    dataframe: pd.DataFrame,
    error_type: CorruptionType,
    *,
    fraction: float = 0.1,
    seed: int = 42,
    columns: Sequence[str] | None = None,
) -> CorruptionResult:
    """Inject one corruption type while preserving a clean reference copy."""
    _validate_fraction(fraction)
    clean = _prepare_dataframe(dataframe)
    corrupted = clean.copy(deep=True)
    rng = random.Random(seed)
    records: list[CorruptionRecord] = []

    requested_columns = list(columns) if columns is not None else None
    if requested_columns is not None:
        unknown = set(requested_columns) - set(clean.columns)
        if unknown:
            raise ValueError(f"Unknown corruption columns: {sorted(unknown)!r}.")
        if ROW_ID_COLUMN in requested_columns:
            raise ValueError("The internal row-id column cannot be corrupted.")

    if error_type == "duplicate_row":
        count = max(1, round(len(clean) * fraction))
        selected = rng.sample(list(clean.index), min(count, len(clean)))
        duplicates = clean.loc[selected].copy(deep=True)
        corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
        for row_index in selected:
            records.append(
                CorruptionRecord(
                    error_type=error_type,
                    row_id=int(clean.at[row_index, ROW_ID_COLUMN]),
                    column=None,
                    original_value=clean.loc[row_index].to_dict(),
                    corrupted_value=clean.loc[row_index].to_dict(),
                    expected_behavior="remove_duplicate",
                )
            )
    elif error_type == "missing_value":
        eligible = requested_columns or [
            str(column) for column in clean.columns if column != ROW_ID_COLUMN
        ]
        for row_index, column in _sample_cells(clean, eligible, fraction, rng):
            original = clean.at[row_index, column]
            corrupted.at[row_index, column] = None
            records.append(
                CorruptionRecord(
                    error_type=error_type,
                    row_id=int(clean.at[row_index, ROW_ID_COLUMN]),
                    column=column,
                    original_value=original,
                    corrupted_value=None,
                    expected_behavior="restore_value",
                )
            )
    elif error_type == "numeric_type":
        eligible = requested_columns or [
            str(column)
            for column in clean.columns
            if column != ROW_ID_COLUMN and is_numeric_dtype(clean[column])
        ]
        if not eligible:
            raise ValueError("No numeric columns are available for corruption.")
        selected_cells = _sample_cells(clean, eligible, fraction, rng)
        for column in {column for _, column in selected_cells}:
            corrupted[column] = corrupted[column].astype("object")
        for row_index, column in selected_cells:
            original = clean.at[row_index, column]
            # Represent a valid number with the wrong storage type. This is a
            # reversible type corruption; an unparseable token would destroy
            # the original value and cannot be fairly scored as repairable.
            corrupted_value = str(original)
            corrupted.at[row_index, column] = corrupted_value
            records.append(
                CorruptionRecord(
                    error_type=error_type,
                    row_id=int(clean.at[row_index, ROW_ID_COLUMN]),
                    column=column,
                    original_value=original,
                    corrupted_value=corrupted_value,
                    expected_behavior="restore_value",
                )
            )
    elif error_type == "datetime_type":
        eligible = requested_columns or [
            str(column)
            for column in clean.columns
            if column != ROW_ID_COLUMN and is_datetime64_any_dtype(clean[column])
        ]
        if not eligible:
            raise ValueError("No datetime columns are available for corruption.")
        selected_cells = _sample_cells(clean, eligible, fraction, rng)
        for column in {column for _, column in selected_cells}:
            corrupted[column] = corrupted[column].astype("object")
        for row_index, column in selected_cells:
            original = clean.at[row_index, column]
            # Use an alternate but parseable representation so conversion can
            # recover the clean reference value without guessing.
            corrupted_value = pd.Timestamp(original).strftime("%Y/%m/%d")
            corrupted.at[row_index, column] = corrupted_value
            records.append(
                CorruptionRecord(
                    error_type=error_type,
                    row_id=int(clean.at[row_index, ROW_ID_COLUMN]),
                    column=column,
                    original_value=original,
                    corrupted_value=corrupted_value,
                    expected_behavior="restore_value",
                )
            )
    else:
        raise ValueError(f"Unsupported corruption type: {error_type!r}.")

    return CorruptionResult(
        clean=clean,
        corrupted=corrupted,
        records=records,
        error_type=error_type,
        seed=seed,
    )
