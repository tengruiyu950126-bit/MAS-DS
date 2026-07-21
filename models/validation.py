"""Structured validation results for preprocessing runs."""

from typing import Literal

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    code: str = Field(min_length=1)
    severity: Literal["warning", "error"]
    message: str = Field(min_length=1)
    column: str | None = None


class ValidationResult(BaseModel):
    valid: bool
    recommend_rollback: bool
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    columns_before: int = Field(ge=0)
    columns_after: int = Field(ge=0)
    missing_before: int = Field(ge=0)
    missing_after: int = Field(ge=0)
    duplicates_before: int = Field(ge=0)
    duplicates_after: int = Field(ge=0)
    issues: list[ValidationIssue]
