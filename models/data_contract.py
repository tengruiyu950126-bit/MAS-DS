"""Typed, JSON-serializable dataset contract and finding models."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


Severity = Literal["warning", "error"]
RuleStatus = Literal["failed", "not_evaluated"]
SemanticType = Literal[
    "any", "string", "number", "integer", "float", "boolean", "datetime", "category"
]
ColumnRuleName = Literal[
    "dtype", "nullable", "missing_ratio", "numeric_min", "numeric_max",
    "allowed_values", "datetime_min", "datetime_max", "unique", "regex",
    "text_min_length", "text_max_length",
]


class ColumnContract(BaseModel):
    description: str | None = None
    severity: Severity = "error"
    rule_severities: dict[ColumnRuleName, Severity] = Field(default_factory=dict)
    allowed_dtypes: list[SemanticType] | None = None
    nullable: bool | None = None
    max_missing_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    numeric_min: float | None = None
    numeric_max: float | None = None
    allowed_values: list[str | int | float | bool] | None = None
    datetime_min: str | None = None
    datetime_max: str | None = None
    unique: bool | None = None
    regex: str | None = None
    text_min_length: int | None = Field(default=None, ge=0)
    text_max_length: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bounds_and_pattern(self) -> "ColumnContract":
        if self.numeric_min is not None and self.numeric_max is not None and self.numeric_min > self.numeric_max:
            raise ValueError("numeric_min must be less than or equal to numeric_max")
        if self.text_min_length is not None and self.text_max_length is not None and self.text_min_length > self.text_max_length:
            raise ValueError("text_min_length must be less than or equal to text_max_length")
        if self.regex is not None:
            try:
                re.compile(self.regex)
            except re.error as exc:
                raise ValueError(f"regex is invalid: {exc}") from exc
        parsed_dates: dict[str, datetime] = {}
        for label, value in (("datetime_min", self.datetime_min), ("datetime_max", self.datetime_max)):
            if value is None:
                continue
            try:
                parsed_dates[label] = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"{label} must be an ISO-8601 datetime") from exc
        if set(parsed_dates) == {"datetime_min", "datetime_max"}:
            minimum = parsed_dates["datetime_min"]
            maximum = parsed_dates["datetime_max"]
            if minimum.tzinfo is None and maximum.tzinfo is not None:
                minimum = minimum.replace(tzinfo=maximum.tzinfo)
            if maximum.tzinfo is None and minimum.tzinfo is not None:
                maximum = maximum.replace(tzinfo=minimum.tzinfo)
            if minimum > maximum:
                raise ValueError("datetime_min must be earlier than or equal to datetime_max")
        return self

    def severity_for(self, rule: ColumnRuleName) -> Severity:
        return self.rule_severities.get(rule, self.severity)


class DataContract(BaseModel):
    name: str = Field(min_length=1)
    schema_version: str = Field(default="1.0", min_length=1)
    description: str | None = None
    required_columns: list[str] = Field(default_factory=list)
    optional_columns: list[str] = Field(default_factory=list)
    protected_columns: list[str] = Field(default_factory=list)
    allow_extra_columns: bool = True
    required_columns_severity: Severity = "error"
    unexpected_columns_severity: Severity = "warning"
    columns: dict[str, ColumnContract] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_column_sets(self) -> "DataContract":
        groups = {
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
            "protected_columns": self.protected_columns,
        }
        for label, values in groups.items():
            if any(not str(value).strip() for value in values):
                raise ValueError(f"{label} cannot contain blank column names")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} cannot contain duplicate column names")
        overlap = set(self.required_columns) & set(self.optional_columns)
        if overlap:
            raise ValueError(f"required_columns and optional_columns overlap: {sorted(overlap)}")
        return self


class ContractFinding(BaseModel):
    rule_id: str = Field(min_length=1)
    column: str | None = None
    severity: Severity
    status: RuleStatus = "failed"
    expected: str = Field(min_length=1)
    observed: str = Field(min_length=1)
    message: str = Field(min_length=1)
    structural: bool = False


class ContractValidationResult(BaseModel):
    valid: bool
    contract_name: str
    schema_version: str
    dataset_rows: int = Field(ge=0)
    dataset_columns: int = Field(ge=0)
    findings: list[ContractFinding] = Field(default_factory=list)
    error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    not_evaluated_count: int = Field(ge=0)
    block_execution: bool = False


BUILTIN_EXAMPLE_CONTRACT = DataContract(
    name="MAS-DS safe example",
    schema_version="1.0",
    description="Non-blocking example with protected identifiers and common semantic checks.",
    protected_columns=["id", "customer_id", "order_id", "student_id"],
    columns={
        "city": ColumnContract(allowed_dtypes=["string", "category"], nullable=True, severity="warning"),
        "amount": ColumnContract(allowed_dtypes=["number"], numeric_min=0, severity="warning"),
    },
)
