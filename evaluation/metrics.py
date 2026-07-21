"""Objective detection, repair, and preservation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from evaluation.corrupt_dataset import ROW_ID_COLUMN, CorruptionRecord
from models.cleaning_plan import CleaningPlan


IssueKey = tuple[str, str | None]


@dataclass(frozen=True)
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class RepairMetrics:
    repaired: int
    total_corruptions: int
    repair_success_rate: float
    preserved_cells: int
    total_unaffected_cells: int
    data_preservation_rate: float


def expected_issues(records: list[CorruptionRecord]) -> set[IssueKey]:
    mapping = {
        "missing_value": "fill",
        "duplicate_row": "drop_duplicates",
        "numeric_type": "convert_numeric",
        "datetime_type": "convert_datetime",
    }
    return {(mapping[record.error_type], record.column) for record in records}


def predicted_issues(plan: CleaningPlan) -> set[IssueKey]:
    issues: set[IssueKey] = set()
    for step in plan.steps:
        if step.operation in {"fill_mean", "fill_median", "fill_mode"}:
            issues.add(("fill", step.column))
        elif step.operation == "parse_numeric_text":
            issues.add(("convert_numeric", step.column))
        elif step.operation in {
            "strip_whitespace",
            "normalize_case",
            "normalize_category_typos",
            "flag_outliers_iqr",
        }:
            issues.add((step.operation, step.column))
        elif step.operation != "leave_unchanged":
            issues.add((step.operation, step.column))
    return issues


def calculate_detection_metrics(
    records: list[CorruptionRecord],
    plan: CleaningPlan,
) -> DetectionMetrics:
    expected = expected_issues(records)
    predicted = predicted_issues(plan)
    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = (
        true_positives / (true_positives + false_positives)
        if predicted
        else (1.0 if not expected else 0.0)
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if expected
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, pd.Timestamp) or isinstance(right, pd.Timestamp):
        try:
            return pd.Timestamp(left) == pd.Timestamp(right)
        except (TypeError, ValueError):
            return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _rows_for_id(dataframe: pd.DataFrame, row_id: int) -> pd.DataFrame:
    if ROW_ID_COLUMN not in dataframe.columns:
        return dataframe.iloc[0:0]
    return dataframe[dataframe[ROW_ID_COLUMN] == row_id]


def _record_repaired(repaired: pd.DataFrame, record: CorruptionRecord) -> bool:
    matches = _rows_for_id(repaired, record.row_id)
    if record.error_type == "duplicate_row":
        return len(matches) == 1
    if matches.empty or record.column is None:
        return False
    value = matches.iloc[0][record.column]
    if record.error_type == "missing_value":
        return pd.notna(value)
    if record.error_type == "numeric_type":
        if not is_numeric_dtype(repaired[record.column]):
            return False
        converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return pd.notna(converted) and float(converted) == float(record.original_value)
    if record.error_type == "datetime_type":
        if not is_datetime64_any_dtype(repaired[record.column]):
            return False
        converted = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
        return pd.notna(converted) and pd.Timestamp(converted) == pd.Timestamp(
            record.original_value
        )
    return False


def calculate_repair_metrics(
    clean: pd.DataFrame,
    repaired: pd.DataFrame,
    records: list[CorruptionRecord],
) -> RepairMetrics:
    repaired_count = sum(_record_repaired(repaired, record) for record in records)
    affected_cells = {
        (record.row_id, record.column)
        for record in records
        if record.column is not None
    }

    preserved_cells = 0
    total_unaffected_cells = 0
    for _, clean_row in clean.iterrows():
        row_id = int(clean_row[ROW_ID_COLUMN])
        repaired_rows = _rows_for_id(repaired, row_id)
        for column in clean.columns:
            if column == ROW_ID_COLUMN or (row_id, column) in affected_cells:
                continue
            total_unaffected_cells += 1
            if not repaired_rows.empty and _equal(
                clean_row[column],
                repaired_rows.iloc[0][column],
            ):
                preserved_cells += 1

    repair_rate = repaired_count / len(records) if records else 1.0
    preservation_rate = (
        preserved_cells / total_unaffected_cells
        if total_unaffected_cells
        else 1.0
    )
    return RepairMetrics(
        repaired=repaired_count,
        total_corruptions=len(records),
        repair_success_rate=repair_rate,
        preserved_cells=preserved_cells,
        total_unaffected_cells=total_unaffected_cells,
        data_preservation_rate=preservation_rate,
    )
