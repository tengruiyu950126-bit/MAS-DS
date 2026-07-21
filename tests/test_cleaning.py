import pandas as pd
import pytest

from models.cleaning_plan import CleaningPlan, CleaningStep
from tools.cleaning import CleaningExecutionError, execute_plan, execute_step


def make_step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Test operation",
        confidence=1.0,
    )


def test_drop_duplicates_does_not_mutate_original() -> None:
    original = pd.DataFrame({"name": ["Alice", "Alice", "Bob"]})

    result = execute_step(original, make_step("drop_duplicates"))

    assert len(result) == 2
    assert len(original) == 3


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("fill_mean", 20.0), ("fill_median", 20.0)],
)
def test_numeric_fill_operations(operation: str, expected: float) -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})

    result = execute_step(dataframe, make_step(operation, "age"))

    assert result.loc[1, "age"] == expected
    assert dataframe["age"].isna().sum() == 1


def test_fill_mode_uses_most_common_non_missing_value() -> None:
    dataframe = pd.DataFrame({"city": ["SG", "SG", "KL", None]})

    result = execute_step(dataframe, make_step("fill_mode", "city"))

    assert result.loc[3, "city"] == "SG"


def test_convert_numeric_coerces_invalid_values() -> None:
    dataframe = pd.DataFrame({"age": ["10", "unknown", "30"]})

    result = execute_step(dataframe, make_step("convert_numeric", "age"))

    assert result["age"].tolist()[0] == 10.0
    assert pd.isna(result.loc[1, "age"])


def test_convert_datetime_coerces_invalid_values() -> None:
    dataframe = pd.DataFrame({"date": ["2025-01-02", "invalid"]})

    result = execute_step(dataframe, make_step("convert_datetime", "date"))

    assert str(result["date"].dtype).startswith("datetime64")
    assert pd.isna(result.loc[1, "date"])


def test_parse_numeric_text_handles_currency_commas_and_percentages() -> None:
    dataframe = pd.DataFrame({"amount": ["$1,200.50", "30%", "(50)"]})

    result = execute_step(dataframe, make_step("parse_numeric_text", "amount"))

    assert result["amount"].tolist() == [1200.5, 0.3, -50.0]


def test_strip_whitespace_cleans_text_cells_only() -> None:
    dataframe = pd.DataFrame({"city": [" Singapore ", "KL", None]})

    result = execute_step(dataframe, make_step("strip_whitespace", "city"))

    assert result["city"].iloc[:2].tolist() == ["Singapore", "KL"]
    assert pd.isna(result.loc[2, "city"])


def test_normalize_case_reuses_preferred_observed_variant() -> None:
    dataframe = pd.DataFrame(
        {"city": ["Singapore", "singapore", "SINGAPORE", "KL"]}
    )

    result = execute_step(dataframe, make_step("normalize_case", "city"))

    assert result["city"].tolist() == ["Singapore", "Singapore", "Singapore", "KL"]


def test_normalize_category_typos_repairs_rare_near_duplicate() -> None:
    dataframe = pd.DataFrame(
        {"city": ["Singapore", "Singapore", "Singaproe", "Bangkok"]}
    )

    result = execute_step(dataframe, make_step("normalize_category_typos", "city"))

    assert result["city"].tolist() == [
        "Singapore",
        "Singapore",
        "Singapore",
        "Bangkok",
    ]


def test_flag_outliers_iqr_is_advisory_and_non_mutating() -> None:
    dataframe = pd.DataFrame({"value": [10, 11, 9, 10, 12, 11, 10, 9, 500]})

    result = execute_step(dataframe, make_step("flag_outliers_iqr", "value"))

    assert result.equals(dataframe)


def test_missing_column_is_rejected() -> None:
    dataframe = pd.DataFrame({"age": [1, 2]})

    with pytest.raises(CleaningExecutionError, match="Unknown column"):
        execute_step(dataframe, make_step("fill_median", "salary"))


def test_numeric_fill_rejects_text_column() -> None:
    dataframe = pd.DataFrame({"city": ["SG", None]})

    with pytest.raises(CleaningExecutionError, match="numeric column"):
        execute_step(dataframe, make_step("fill_mean", "city"))


def test_execute_plan_returns_audit_records() -> None:
    original = pd.DataFrame(
        {"age": [10.0, None, 10.0], "name": ["A", "B", "A"]}
    )
    plan = CleaningPlan(
        steps=[
            make_step("fill_median", "age"),
            make_step("drop_duplicates"),
        ]
    )

    result, records = execute_plan(original, plan)

    assert result["age"].isna().sum() == 0
    assert len(result) == 2
    assert len(records) == 2
    assert records[0].missing_before == 1
    assert records[0].missing_after == 0
    assert records[1].rows_before == 3
    assert records[1].rows_after == 2
    assert original["age"].isna().sum() == 1
