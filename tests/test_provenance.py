from datetime import datetime, timezone

import pandas as pd

from agents.multi_expert import MultiExpertCleaningAgent
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.data_contract import DataContract
from models.policy import PreprocessingPolicy
from tools.provenance import (
    build_audit_provenance,
    cleaning_plan_fingerprint,
    dataframe_fingerprint,
    orchestration_trace_fingerprint,
    policy_fingerprint,
)


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="provenance test",
        confidence=0.9,
    )


def test_same_dataframe_has_same_index_independent_fingerprint() -> None:
    first = pd.DataFrame({"name": ["A", "B"], "value": [1, 2]}, index=[10, 20])
    second = first.copy()
    second.index = [100, 200]

    assert dataframe_fingerprint(first) == dataframe_fingerprint(second)
    assert len(dataframe_fingerprint(first)) == 64


def test_different_dataframe_has_different_fingerprint() -> None:
    first = pd.DataFrame({"value": [1, 2]})
    second = pd.DataFrame({"value": [1, 3]})

    assert dataframe_fingerprint(first) != dataframe_fingerprint(second)


def test_plan_fingerprint_is_stable_and_step_order_sensitive() -> None:
    first = CleaningPlan(steps=[step("strip_whitespace", "city"), step("fill_mode", "city")])
    same = CleaningPlan(steps=[step("strip_whitespace", "city"), step("fill_mode", "city")])
    reordered = CleaningPlan(steps=[step("fill_mode", "city"), step("strip_whitespace", "city")])

    assert cleaning_plan_fingerprint(first) == cleaning_plan_fingerprint(same)
    assert cleaning_plan_fingerprint(first) != cleaning_plan_fingerprint(reordered)


def test_policy_fingerprint_is_deterministic() -> None:
    first = PreprocessingPolicy(
        protected_columns=["email"],
        denied_operations=["normalize_category_typos"],
    )
    second = first.model_copy(deep=True)

    assert policy_fingerprint(first) == policy_fingerprint(second)


def test_trace_fingerprint_is_deterministic() -> None:
    dataframe = pd.DataFrame({"amount": ["10", None, "30"]})
    first_agent = MultiExpertCleaningAgent()
    second_agent = MultiExpertCleaningAgent()
    first_agent.propose(dataframe)
    second_agent.propose(dataframe.copy())

    assert first_agent.last_trace is not None
    assert second_agent.last_trace is not None
    assert orchestration_trace_fingerprint(
        first_agent.last_trace
    ) == orchestration_trace_fingerprint(second_agent.last_trace)


def test_build_provenance_links_dataset_plan_policy_and_trace_without_raw_values() -> None:
    dataframe = pd.DataFrame({"secret": ["private-value"], "amount": [10]})
    policy = PreprocessingPolicy()
    agent = MultiExpertCleaningAgent(policy=policy)
    plan = agent.propose(dataframe)

    provenance = build_audit_provenance(
        dataframe=dataframe,
        plan=plan,
        policy=policy,
        trace=agent.last_trace,
        source_label="upload:private.csv",
        planner_name="Multi-Expert",
        planner_mode="Multi-Expert",
        created_at_utc=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )
    serialized = provenance.model_dump_json()

    assert provenance.created_at_utc == "2026-07-20T12:00:00Z"
    assert provenance.dataset_rows == 1
    assert provenance.dataset_columns == 2
    assert provenance.dataset_fingerprint == dataframe_fingerprint(dataframe)
    assert provenance.cleaning_plan_fingerprint == cleaning_plan_fingerprint(plan)
    assert provenance.policy_fingerprint == policy_fingerprint(policy)
    assert provenance.orchestration_trace_fingerprint is not None
    assert provenance.final_plan_steps == len(plan.steps)
    assert "private-value" not in serialized


def test_provenance_includes_deterministic_contract_fingerprint() -> None:
    dataframe = pd.DataFrame({"id": [1]})
    plan = CleaningPlan(steps=[])
    policy = PreprocessingPolicy()
    contract = DataContract(name="audit", required_columns=["id"])

    first = build_audit_provenance(
        dataframe=dataframe, plan=plan, policy=policy, contract=contract
    )
    second = build_audit_provenance(
        dataframe=dataframe, plan=plan, policy=policy, contract=contract
    )

    assert first.contract_fingerprint == second.contract_fingerprint
    assert first.contract_fingerprint is not None
    assert len(first.contract_fingerprint) == 64
