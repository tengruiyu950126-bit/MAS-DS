"""Deterministic dataframe profiling utilities."""

from __future__ import annotations

import pandas as pd

from models.profile import ColumnProfile, DataProfile


def profile_dataframe(dataframe: pd.DataFrame) -> DataProfile:
    """Return a JSON-serializable profile without modifying the dataframe."""
    columns = {
        str(column): ColumnProfile(
            dtype=str(dataframe[column].dtype),
            missing_count=int(dataframe[column].isna().sum()),
            missing_ratio=float(dataframe[column].isna().mean()),
            unique_count=int(dataframe[column].nunique(dropna=True)),
        )
        for column in dataframe.columns
    }
    return DataProfile(
        rows=int(len(dataframe)),
        columns=int(len(dataframe.columns)),
        duplicate_rows=int(dataframe.duplicated().sum()),
        column_profiles=columns,
    )
