"""User-configurable preprocessing safety policy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.cleaning_plan import Operation


OutlierAction = Literal["flag", "ignore"]


class ColumnPolicy(BaseModel):
    """Per-column overrides for preprocessing decisions."""

    protected: bool = False
    allowed_operations: list[Operation] | None = None
    denied_operations: list[Operation] = Field(default_factory=list)
    outlier_action: OutlierAction | None = None


class PreprocessingPolicy(BaseModel):
    """Global and per-column rules for safe preprocessing.

    The default policy is conservative: identifier-like columns are protected
    automatically and numeric outliers are flagged, not modified.
    """

    protect_identifier_columns: bool = True
    protected_columns: list[str] = Field(default_factory=list)
    allowed_operations: list[Operation] | None = None
    denied_operations: list[Operation] = Field(default_factory=list)
    outlier_action: OutlierAction = "flag"
    columns: dict[str, ColumnPolicy] = Field(default_factory=dict)
