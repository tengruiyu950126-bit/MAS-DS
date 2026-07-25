import pandas as pd

from agents.orchestrator import PreprocessingOrchestrator
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.policy import PreprocessingPolicy


class CountingPlanner:
    name = "counting_planner"

    def __init__(self, plan: CleaningPlan) -> None:
        self.plan = plan
        self.calls = 0

    def propose(self, dataframe, profile=None) -> CleaningPlan:
        self.calls += 1
        return self.plan


class MutatingPlanner:
    name = "mutating_planner"

    def propose(self, dataframe, profile=None) -> CleaningPlan:
        dataframe.loc[0, "value"] = "mutated without approval"
        return CleaningPlan(steps=[])


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Pipeline test",
        confidence=1.0,
    )


def test_pipeline_proposal_does_not_execute_or_mutate() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    original = dataframe.copy(deep=True)
    orchestrator = PreprocessingOrchestrator()

    proposal = orchestrator.propose(dataframe)

    assert proposal.profile.rows == 3
    assert proposal.plan.steps[0].operation == "fill_median"
    pd.testing.assert_frame_equal(dataframe, original)


def test_pipeline_isolates_caller_dataframe_from_planner_mutation() -> None:
    dataframe = pd.DataFrame({"value": ["original"]})
    original = dataframe.copy(deep=True)

    PreprocessingOrchestrator(cleaning_agent=MutatingPlanner()).propose(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_approved_pipeline_does_not_replan_and_commits() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    planner = CountingPlanner(CleaningPlan(steps=[step("fill_median", "age")]))
    orchestrator = PreprocessingOrchestrator(cleaning_agent=planner)

    proposal = orchestrator.propose(dataframe)
    outcome = orchestrator.execute_approved(
        dataframe,
        proposal.plan,
        run_id=proposal.run_id,
    )

    assert planner.calls == 1
    assert outcome.validation.valid
    assert not outcome.rolled_back
    assert outcome.dataframe["age"].isna().sum() == 0
    assert outcome.run_id == proposal.run_id


def test_pipeline_logs_value_free_correlated_events(caplog) -> None:
    dataframe = pd.DataFrame(
        {"secret": ["do-not-log", "safe"], "age": [1.0, None]}
    )
    orchestrator = PreprocessingOrchestrator()

    with caplog.at_level("INFO", logger="mas_ds"):
        proposal = orchestrator.propose(dataframe)
        orchestrator.execute_approved(
            dataframe,
            proposal.plan,
            run_id=proposal.run_id,
        )

    combined = "\n".join(caplog.messages)
    assert proposal.run_id in combined
    assert "planning_started" in combined
    assert "execution_completed" in combined
    assert "do-not-log" not in combined


def test_validation_failure_returns_defensive_original_copy() -> None:
    dataframe = pd.DataFrame({"age": ["10", "unknown", "30"]})
    plan = CleaningPlan(steps=[step("convert_numeric", "age")])
    orchestrator = PreprocessingOrchestrator()

    outcome = orchestrator.execute_approved(dataframe, plan)

    assert outcome.rolled_back
    assert not outcome.validation.valid
    assert outcome.dataframe.equals(dataframe)
    assert outcome.dataframe is not dataframe


def test_executor_rejects_plan_denied_by_active_policy_before_mutation() -> None:
    dataframe = pd.DataFrame({"name": [" Alice "]})
    plan = CleaningPlan(steps=[step("strip_whitespace", "name")])
    policy = PreprocessingPolicy(denied_operations=["strip_whitespace"])

    outcome = PreprocessingOrchestrator(policy=policy).execute_approved(
        dataframe,
        plan,
    )

    assert outcome.rolled_back
    assert not outcome.validation.valid
    assert outcome.execution_records == []
    assert outcome.dataframe.equals(dataframe)
    assert outcome.dataframe is not dataframe
    assert any(
        issue.code == "policy_rejected_operation"
        for issue in outcome.validation.issues
    )
