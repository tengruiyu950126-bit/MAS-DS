import pandas as pd

from models.cleaning_plan import CleaningPlan, CleaningStep
from tools.cleaning import execute_plan
from tools.validation import apply_rollback_policy, validate_preprocessing


def make_step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Test operation",
        confidence=1.0,
    )


def test_valid_fill_is_committed() -> None:
    before = pd.DataFrame({"age": [10.0, None, 30.0]})
    plan = CleaningPlan(steps=[make_step("fill_median", "age")])
    candidate, _ = execute_plan(before, plan)

    validation = validate_preprocessing(before, candidate, plan)
    result, rolled_back = apply_rollback_policy(before, candidate, validation)

    assert validation.valid
    assert not validation.recommend_rollback
    assert not rolled_back
    assert result["age"].isna().sum() == 0


def test_unapproved_row_loss_is_rolled_back() -> None:
    before = pd.DataFrame({"value": [1, 2, 3]})
    candidate = before.iloc[:2].copy()
    plan = CleaningPlan(steps=[])

    validation = validate_preprocessing(before, candidate, plan)
    result, rolled_back = apply_rollback_policy(before, candidate, validation)

    assert not validation.valid
    assert rolled_back
    assert result.equals(before)
    assert any(issue.code == "unexpected_row_loss" for issue in validation.issues)


def test_new_missing_values_trigger_rollback() -> None:
    before = pd.DataFrame({"age": ["10", "unknown", "30"]})
    plan = CleaningPlan(steps=[make_step("convert_numeric", "age")])
    candidate, _ = execute_plan(before, plan)

    validation = validate_preprocessing(before, candidate, plan)

    assert validation.recommend_rollback
    assert any(issue.code == "new_missing_values" for issue in validation.issues)


def test_conversion_followed_by_fill_can_pass_validation() -> None:
    before = pd.DataFrame({"age": ["10", "unknown", "30"]})
    plan = CleaningPlan(
        steps=[
            make_step("convert_numeric", "age"),
            make_step("fill_median", "age"),
        ]
    )
    candidate, _ = execute_plan(before, plan)

    validation = validate_preprocessing(before, candidate, plan)

    assert validation.valid
    assert candidate["age"].tolist() == [10.0, 20.0, 30.0]


def test_approved_duplicate_removal_passes_validation() -> None:
    before = pd.DataFrame({"name": ["A", "A", "B"]})
    plan = CleaningPlan(steps=[make_step("drop_duplicates")])
    candidate, _ = execute_plan(before, plan)

    validation = validate_preprocessing(before, candidate, plan)

    assert validation.valid
    assert validation.rows_before == 3
    assert validation.rows_after == 2
    assert validation.duplicates_after == 0


def test_schema_change_triggers_rollback() -> None:
    before = pd.DataFrame({"name": ["A", "B"]})
    candidate = before.rename(columns={"name": "full_name"})
    plan = CleaningPlan(steps=[])

    validation = validate_preprocessing(before, candidate, plan)

    assert validation.recommend_rollback
    assert any(issue.code == "schema_changed" for issue in validation.issues)


def test_protected_identifier_modification_triggers_rollback() -> None:
    before = pd.DataFrame({"customer_id": [" C-1 ", "C-2"], "city": ["SG", "KL"]})
    plan = CleaningPlan(steps=[make_step("strip_whitespace", "customer_id")])
    candidate, _ = execute_plan(before, plan)

    validation = validate_preprocessing(before, candidate, plan)

    assert validation.recommend_rollback
    assert any(
        issue.code == "protected_column_modified"
        for issue in validation.issues
    )


def test_duplicate_removal_can_keep_protected_identifier_safe() -> None:
    before = pd.DataFrame({"order_id": ["A-1", "A-1", "A-2"]})
    plan = CleaningPlan(steps=[make_step("drop_duplicates")])
    candidate, _ = execute_plan(before, plan)

    validation = validate_preprocessing(before, candidate, plan)

    assert validation.valid


def test_rollback_returns_defensive_copy() -> None:
    before = pd.DataFrame({"value": [1, 2]})
    candidate = pd.DataFrame({"value": [1]})
    validation = validate_preprocessing(before, candidate, CleaningPlan(steps=[]))

    result, rolled_back = apply_rollback_policy(before, candidate, validation)
    result.loc[0, "value"] = 999

    assert rolled_back
    assert before.loc[0, "value"] == 1
