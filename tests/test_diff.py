import pandas as pd

from models.cleaning_plan import CleaningPlan, CleaningStep
from tools.diff import diff_summary_frame, plan_diff_frame


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="diff test",
        confidence=0.9,
    )


def test_plan_diff_reports_cell_value_changes() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    plan = CleaningPlan(steps=[step("fill_median", "age")])

    diff = plan_diff_frame(dataframe, plan)

    assert len(diff) == 1
    row = diff.iloc[0]
    assert row["operation"] == "fill_median"
    assert row["column"] == "age"
    assert row["change_type"] == "value_changed"
    assert row["row"] == 1
    assert row["before"] == "<missing>"
    assert row["after"] == "20.0"


def test_plan_diff_reports_dtype_changes() -> None:
    dataframe = pd.DataFrame({"amount": ["10", "20", "30"]})
    plan = CleaningPlan(steps=[step("convert_numeric", "amount")])

    diff = plan_diff_frame(dataframe, plan)

    dtype_rows = diff[diff["change_type"] == "dtype_changed"]
    assert len(dtype_rows) == 1
    assert dtype_rows.iloc[0]["before"] in {"object", "str", "string"}
    assert dtype_rows.iloc[0]["after"] in {"int64", "Int64"}


def test_plan_diff_reports_duplicate_row_removal_without_shift_noise() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "city": ["A", "A", "B"],
        }
    )
    plan = CleaningPlan(steps=[step("drop_duplicates")])

    diff = plan_diff_frame(dataframe, plan)

    assert len(diff) == 1
    row = diff.iloc[0]
    assert row["change_type"] == "row_removed"
    assert row["column"] == "*"
    assert row["row"] == 1
    assert "customer_id=1" in row["before"]
    assert row["after"] == "<removed>"


def test_plan_diff_omits_non_mutating_outlier_flags() -> None:
    dataframe = pd.DataFrame({"amount": [10, 11, 12, 500]})
    plan = CleaningPlan(steps=[step("flag_outliers_iqr", "amount")])

    diff = plan_diff_frame(dataframe, plan)

    assert diff.empty


def test_diff_summary_counts_changes_by_operation() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    plan = CleaningPlan(steps=[step("fill_median", "age")])

    summary = diff_summary_frame(plan_diff_frame(dataframe, plan))

    assert summary.iloc[0]["operation"] == "fill_median"
    assert summary.iloc[0]["column"] == "age"
    assert summary.iloc[0]["change_type"] == "value_changed"
    assert summary.iloc[0]["changes"] == 1


def test_plan_diff_caps_total_audit_rows() -> None:
    dataframe = pd.DataFrame({"value": [None] * 20 + [1.0]})
    plan = CleaningPlan(steps=[step("fill_median", "value")])

    diff = plan_diff_frame(dataframe, plan, max_total_changes=5)

    assert len(diff) == 5
