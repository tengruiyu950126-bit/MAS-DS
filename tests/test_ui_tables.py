import pandas as pd

from agents.multi_expert import MultiExpertCleaningAgent
from agents.planner_factory import build_offline_planner
from tools.profiler import profile_dataframe
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.policy import PreprocessingPolicy
from tools.cleaning import execute_plan
from tools.ui_tables import (
    ARBITER_TRACE_COLUMNS,
    CRITIC_TRACE_COLUMNS,
    ROUTER_TRACE_COLUMNS,
    SPECIALIST_TRACE_COLUMNS,
    arbiter_trace_frame,
    changed_columns_frame,
    critic_trace_frame,
    dataset_metrics_frame,
    execution_records_frame,
    operation_summary_frame,
    plan_frame,
    plan_summary_frame,
    policy_summary_frame,
    profile_frame,
    router_trace_frame,
    specialist_trace_frame,
    validation_issues_frame,
    validation_summary_frame,
)
from tools.validation import validate_preprocessing


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="UI table test",
        confidence=0.9,
    )


def test_dataset_and_profile_tables_are_readable() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0], "city": ["A", "B", "A"]})
    profile = profile_dataframe(dataframe)

    metrics = dataset_metrics_frame(dataframe)
    profile_table = profile_frame(profile)

    assert metrics.iloc[0]["rows"] == 3
    assert metrics.iloc[0]["missing_values"] == 1
    assert set(profile_table["column"]) == {"age", "city"}
    assert "missing_ratio" in profile_table.columns


def test_plan_and_execution_tables_are_readable() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    plan = CleaningPlan(steps=[step("fill_median", "age")])
    candidate, records = execute_plan(dataframe, plan)
    validation = validate_preprocessing(dataframe, candidate, plan)

    assert plan_frame(plan).iloc[0]["operation"] == "fill_median"
    assert execution_records_frame(records).iloc[0]["missing_after"] == 0
    assert validation_summary_frame(validation).iloc[0]["metric"] == "rows"
    assert validation_issues_frame(validation).empty


def test_plan_summary_counts_mutating_and_advisory_steps() -> None:
    plan = CleaningPlan(
        steps=[
            step("fill_median", "age"),
            step("flag_outliers_iqr", "score"),
            step("leave_unchanged", "customer_id"),
        ]
    )

    summary = plan_summary_frame(plan).iloc[0]

    assert summary["planned_steps"] == 3
    assert summary["mutating_steps"] == 1
    assert summary["advisory_steps"] == 2
    assert summary["targeted_columns"] == 3
    assert summary["avg_confidence"] == 0.9


def test_operation_summary_groups_plan_steps() -> None:
    plan = CleaningPlan(
        steps=[
            step("fill_mode", "city"),
            step("fill_mode", "segment"),
            step("drop_duplicates"),
        ]
    )

    summary = operation_summary_frame(plan)

    assert summary.iloc[0]["operation"] == "fill_mode"
    assert summary.iloc[0]["steps"] == 2
    assert summary.iloc[0]["columns"] == "city, segment"


def test_policy_summary_frame_explains_active_policy() -> None:
    policy = PreprocessingPolicy(
        protected_columns=["email"],
        denied_operations=["normalize_category_typos"],
        outlier_action="ignore",
    )

    summary = policy_summary_frame(policy)

    assert set(summary["setting"]) >= {
        "Auto-protect ID columns",
        "Protected columns",
        "Disabled operations",
        "Outlier handling",
    }
    assert "email" in summary.loc[
        summary["setting"] == "Protected columns",
        "value",
    ].iloc[0]
    assert "normalize_category_typos" in summary.loc[
        summary["setting"] == "Disabled operations",
        "value",
    ].iloc[0]


def test_changed_columns_frame_detects_dtype_and_value_changes() -> None:
    before = pd.DataFrame({"age": ["10", "20"], "city": ["A", "B"]})
    after = before.copy()
    after["age"] = pd.to_numeric(after["age"])
    after.loc[1, "city"] = "C"

    changed = changed_columns_frame(before, after)

    assert set(changed["column"]) == {"age", "city"}
    assert changed.loc[changed["column"] == "age", "dtype_changed"].iloc[0]


def test_offline_planner_factory_selects_multi_expert_agent() -> None:
    planner = build_offline_planner(
        "Deterministic routed planner",
        PreprocessingPolicy(),
    )

    assert isinstance(planner, MultiExpertCleaningAgent)
    assert planner.name == "multi_expert_cleaning_agent"


def test_trace_tables_handle_missing_trace_with_stable_empty_schemas() -> None:
    router = router_trace_frame(None)
    specialists = specialist_trace_frame(None)
    critic = critic_trace_frame(None)
    arbiter = arbiter_trace_frame(None)

    assert router.empty and list(router.columns) == ROUTER_TRACE_COLUMNS
    assert specialists.empty and list(specialists.columns) == SPECIALIST_TRACE_COLUMNS
    assert critic.empty and list(critic.columns) == CRITIC_TRACE_COLUMNS
    assert arbiter.empty and list(arbiter.columns) == ARBITER_TRACE_COLUMNS


def test_trace_tables_include_all_orchestration_stages() -> None:
    dataframe = pd.DataFrame(
        {
            "row_id": [1, 2, 2, 4],
            "amount": ["10", None, None, "30"],
            "city": ["Singapore", "singapore", "singapore", None],
        }
    )
    planner = MultiExpertCleaningAgent()
    plan = planner.propose(dataframe)
    trace = planner.last_trace

    router = router_trace_frame(trace)
    specialists = specialist_trace_frame(trace)
    critic = critic_trace_frame(trace)
    arbiter = arbiter_trace_frame(trace)

    assert trace is not None
    assert set(router["issue_type"]) >= {"protected_identifier", "missing_values"}
    assert set(specialists["expert"]) >= {
        "DuplicateExpert",
        "MissingValueExpert",
        "NumericExpert",
        "CategoricalExpert",
    }
    assert "approved" in set(critic["decision"])
    assert list(arbiter["operation"]) == [step.operation for step in plan.steps]
