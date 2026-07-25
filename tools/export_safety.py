"""Safety helpers for spreadsheet-compatible audit exports."""

from __future__ import annotations

import pandas as pd


FORMULA_PREFIXES = ("=", "+", "-", "@")


def neutralize_spreadsheet_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an audit-export copy whose text cells cannot start formulas.

    Cleaned-data downloads intentionally retain exact values; this helper is
    only for audit/report exports where neutralization does not alter results.
    """
    safe = frame.copy(deep=True)
    for column in safe.columns:
        safe[column] = safe[column].map(_neutralize_value)
    return safe


def _neutralize_value(value: object) -> object:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value
