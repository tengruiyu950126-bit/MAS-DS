"""Validated cleaning-plan contracts."""

from typing import Literal

from pydantic import BaseModel, Field


Operation = Literal[
    "drop_duplicates",
    "fill_mean",
    "fill_median",
    "fill_mode",
    "convert_numeric",
    "convert_datetime",
    "parse_numeric_text",
    "strip_whitespace",
    "normalize_case",
    "normalize_category_typos",
    "flag_outliers_iqr",
    "leave_unchanged",
]


class CleaningStep(BaseModel):
    column: str | None = None
    operation: Operation
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class CleaningPlan(BaseModel):
    steps: list[CleaningStep]
