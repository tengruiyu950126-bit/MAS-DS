"""Dataframe profiling contracts."""

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    dtype: str
    missing_count: int = Field(ge=0)
    missing_ratio: float = Field(ge=0.0, le=1.0)
    unique_count: int = Field(ge=0)


class DataProfile(BaseModel):
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    column_profiles: dict[str, ColumnProfile]
