import pandas as pd
import pytest

from agents.cleaning_agent import RuleBasedCleaningAgent
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.policy import PreprocessingPolicy
from tools.cleaning import execute_plan
from tools.policy import (
    PolicyLoadError,
    column_is_protected,
    filter_plan_by_policy,
    load_policy_file,
    normalize_column_list,
    operation_allowed,
    policy_from_json,
    policy_to_json,
    write_policy_file,
)
from tools.validation import validate_preprocessing


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Policy test",
        confidence=1.0,
    )


def test_normalize_column_list_accepts_commas_and_newlines() -> None:
    assert normalize_column_list("email, phone\ncustomer_id") == [
        "email",
        "phone",
        "customer_id",
    ]


def test_policy_protects_extra_columns() -> None:
    policy = PreprocessingPolicy(protected_columns=["city"])

    assert column_is_protected("city", policy)
    assert not operation_allowed(step("strip_whitespace", "city"), policy)
    assert operation_allowed(step("leave_unchanged", "city"), policy)


def test_filter_plan_by_policy_removes_denied_operations() -> None:
    policy = PreprocessingPolicy(denied_operations=["normalize_category_typos"])
    plan = CleaningPlan(
        steps=[
            step("normalize_category_typos", "city"),
            step("strip_whitespace", "city"),
        ]
    )

    filtered = filter_plan_by_policy(plan, policy)

    assert [(item.operation, item.column) for item in filtered.steps] == [
        ("strip_whitespace", "city")
    ]


def test_rule_agent_respects_extra_protected_columns() -> None:
    dataframe = pd.DataFrame({"city": [" Singapore ", None, "KL"]})
    policy = PreprocessingPolicy(protected_columns=["city"])

    plan = RuleBasedCleaningAgent(policy=policy).propose(dataframe)

    assert [(item.operation, item.column) for item in plan.steps] == [
        ("leave_unchanged", "city")
    ]


def test_rule_agent_can_disable_outlier_flagging() -> None:
    dataframe = pd.DataFrame({"amount": [10, 11, 9, 10, 12, 11, 10, 9, 500]})
    policy = PreprocessingPolicy(outlier_action="ignore")

    plan = RuleBasedCleaningAgent(policy=policy).propose(dataframe)

    assert all(item.operation != "flag_outliers_iqr" for item in plan.steps)


def test_validation_uses_policy_for_identifier_protection() -> None:
    dataframe = pd.DataFrame({"customer_id": [" C-1 ", "C-2"]})
    plan = CleaningPlan(steps=[step("strip_whitespace", "customer_id")])
    candidate, _ = execute_plan(dataframe, plan)

    default_validation = validate_preprocessing(dataframe, candidate, plan)
    relaxed_validation = validate_preprocessing(
        dataframe,
        candidate,
        plan,
        policy=PreprocessingPolicy(protect_identifier_columns=False),
    )

    assert default_validation.recommend_rollback
    assert relaxed_validation.valid


def test_policy_json_roundtrip() -> None:
    policy = PreprocessingPolicy(
        protect_identifier_columns=False,
        protected_columns=["email"],
        denied_operations=["normalize_category_typos"],
        outlier_action="ignore",
    )

    restored = policy_from_json(policy_to_json(policy))

    assert restored == policy


def test_invalid_policy_json_is_rejected() -> None:
    with pytest.raises(PolicyLoadError, match="not valid JSON"):
        policy_from_json("{not json")


def test_unknown_policy_operation_is_rejected() -> None:
    with pytest.raises(PolicyLoadError, match="schema"):
        policy_from_json('{"denied_operations": ["delete_everything"]}')


def test_policy_file_helpers_roundtrip(tmp_path) -> None:
    path = tmp_path / "policy.json"
    policy = PreprocessingPolicy(protected_columns=["phone_number"])

    write_policy_file(policy, path)
    restored = load_policy_file(path)

    assert restored == policy


@pytest.mark.parametrize(
    "path",
    [
        "configs/strict_policy.json",
        "configs/permissive_policy.json",
        "configs/no_text_normalization_policy.json",
    ],
)
def test_sample_policy_configs_are_valid(path: str) -> None:
    assert isinstance(load_policy_file(path), PreprocessingPolicy)
