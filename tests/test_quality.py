import pandas as pd

from tools.quality import (
    count_iqr_outliers,
    has_category_typo_issue,
    normalize_category_typos_series,
)


def test_normalize_category_typos_maps_rare_near_duplicate_to_common_label() -> None:
    series = pd.Series(["Singapore", "Singapore", "Singaproe", "Kuala Lumpur"])

    normalized = normalize_category_typos_series(series)

    assert normalized.tolist() == [
        "Singapore",
        "Singapore",
        "Singapore",
        "Kuala Lumpur",
    ]
    assert has_category_typo_issue(series)


def test_normalize_category_typos_avoids_short_or_weak_matches() -> None:
    series = pd.Series(["NY", "NY", "NJ", "LA"])

    normalized = normalize_category_typos_series(series)

    assert normalized.tolist() == ["NY", "NY", "NJ", "LA"]
    assert not has_category_typo_issue(series)


def test_count_iqr_outliers_detects_extreme_numeric_values() -> None:
    series = pd.Series([10, 11, 9, 10, 12, 11, 10, 9, 500])

    assert count_iqr_outliers(series) == 1
