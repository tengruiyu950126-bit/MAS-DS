"""Reusable data-quality heuristics for safe preprocessing."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype


_CURRENCY_PATTERN = re.compile(r"(?i)\b(?:usd|sgd|myr|rm)\b|[$€£¥]")
_NUMERIC_TEXT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_MISSING_TEXT = {"", "na", "n/a", "null", "none", "nan", "-"}


def is_text_like(series: pd.Series) -> bool:
    """Return whether a column can reasonably contain text cleanup issues."""
    return is_object_dtype(series) or is_string_dtype(series)


def is_protected_column(column: str) -> bool:
    """Detect identifier-like columns that should not be automatically changed."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    if normalized in {"id", "uuid", "guid", "identifier"}:
        return True
    return normalized.endswith(("_id", "_uuid", "_guid"))


def protected_columns(dataframe: pd.DataFrame) -> list[str]:
    return [str(column) for column in dataframe.columns if is_protected_column(str(column))]


def strip_whitespace_series(series: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace from string cells only."""
    return series.map(lambda value: value.strip() if isinstance(value, str) else value)


def has_whitespace_issue(series: pd.Series) -> bool:
    if not is_text_like(series):
        return False
    return any(
        isinstance(value, str) and value != value.strip()
        for value in series.dropna()
    )


def _style_score(value: str) -> int:
    if len(value) <= 5 and value.isupper():
        return 4
    if value.istitle():
        return 3
    if not value.islower() and not value.isupper():
        return 2
    if value.isupper():
        return 1
    return 0


def _preferred_variant(values: list[str]) -> str:
    counts = Counter(values)
    return max(
        counts,
        key=lambda value: (counts[value], _style_score(value), -len(value), value),
    )


def normalize_case_series(series: pd.Series) -> pd.Series:
    """Normalize case variants by reusing the strongest observed spelling.

    Example: ``Singapore``, ``singapore``, and ``SINGAPORE`` become
    ``Singapore`` if that is the preferred observed style. The function also
    strips surrounding whitespace before grouping variants.
    """
    stripped = strip_whitespace_series(series)
    groups: dict[str, list[str]] = {}
    for value in stripped.dropna():
        if isinstance(value, str):
            groups.setdefault(value.casefold(), []).append(value)
    mapping = {key: _preferred_variant(values) for key, values in groups.items()}
    return stripped.map(
        lambda value: mapping.get(value.casefold(), value)
        if isinstance(value, str)
        else value
    )


def has_case_inconsistency(series: pd.Series) -> bool:
    if not is_text_like(series):
        return False
    groups: dict[str, set[str]] = {}
    for value in strip_whitespace_series(series).dropna():
        if isinstance(value, str):
            groups.setdefault(value.casefold(), set()).add(value)
    return any(len(variants) > 1 for variants in groups.values())


def _category_typo_mapping(series: pd.Series) -> dict[str, str]:
    """Build a conservative typo-to-canonical mapping for categorical text."""
    if not is_text_like(series):
        return {}
    normalized = normalize_case_series(series)
    values = [
        value
        for value in normalized.dropna().tolist()
        if isinstance(value, str) and len(value) >= 5
    ]
    counts = Counter(values)
    if len(counts) < 2:
        return {}

    common = sorted(counts, key=lambda value: (-counts[value], value))
    mapping: dict[str, str] = {}
    for candidate, candidate_count in counts.items():
        if candidate_count > 1:
            continue
        for target in common:
            if target == candidate:
                continue
            if counts[target] < 2:
                continue
            if len(target) < 5:
                continue
            if target[0].casefold() != candidate[0].casefold():
                continue
            if abs(len(target) - len(candidate)) > 2:
                continue
            similarity = SequenceMatcher(
                None,
                target.casefold(),
                candidate.casefold(),
            ).ratio()
            if similarity >= 0.88:
                mapping[candidate] = target
                break
    return mapping


def normalize_category_typos_series(series: pd.Series) -> pd.Series:
    """Map rare near-duplicate category spellings to frequent canonical labels."""
    normalized = normalize_case_series(series)
    mapping = _category_typo_mapping(normalized)
    if not mapping:
        return normalized
    return normalized.map(
        lambda value: mapping.get(value, value) if isinstance(value, str) else value
    )


def has_category_typo_issue(series: pd.Series) -> bool:
    if not is_text_like(series):
        return False
    normalized = normalize_case_series(series)
    corrected = normalize_category_typos_series(series)
    return not normalized.equals(corrected)


def _parse_numeric_text_value(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if text.casefold() in _MISSING_TEXT:
        return None

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()

    text = _CURRENCY_PATTERN.sub("", text)
    text = text.replace(",", "").replace(" ", "")

    if not _NUMERIC_TEXT_PATTERN.match(text):
        return None

    number = float(text)
    if negative:
        number *= -1
    if is_percent:
        number /= 100
    return int(number) if number.is_integer() and not is_percent else number


def parse_numeric_text_series(series: pd.Series) -> pd.Series:
    """Parse numeric text such as ``$1,200`` or ``30%`` into numbers."""
    return pd.to_numeric(series.map(_parse_numeric_text_value), errors="coerce")


def can_parse_numeric_text_losslessly(series: pd.Series) -> bool:
    """Return true when all non-missing values can be parsed as numbers."""
    if is_numeric_dtype(series) or not is_text_like(series):
        return False
    non_missing = int(series.notna().sum())
    if non_missing == 0:
        return False
    parsed = parse_numeric_text_series(series)
    return int(parsed.notna().sum()) == non_missing


def needs_robust_numeric_parsing(series: pd.Series) -> bool:
    """Detect numeric text that pandas cannot parse without preprocessing."""
    if not can_parse_numeric_text_losslessly(series):
        return False
    standard = pd.to_numeric(series, errors="coerce")
    return int(standard.notna().sum()) < int(series.notna().sum())


def iqr_outlier_bounds(series: pd.Series) -> tuple[float, float] | None:
    """Return Tukey IQR bounds for numeric outlier detection."""
    if not is_numeric_dtype(series):
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 8:
        return None
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask for numeric values outside Tukey IQR bounds."""
    bounds = iqr_outlier_bounds(series)
    if bounds is None:
        return pd.Series(False, index=series.index)
    lower, upper = bounds
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric < lower) | (numeric > upper)


def count_iqr_outliers(series: pd.Series) -> int:
    return int(iqr_outlier_mask(series).sum())


def has_iqr_outliers(series: pd.Series) -> bool:
    return count_iqr_outliers(series) > 0
