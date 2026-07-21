import pandas as pd

from models.cleaning_plan import CleaningPlan, CleaningStep
from workflow.graph import PreprocessingGraphOrchestrator


class CountingPlanner:
    name = "counting_planner"

    def __init__(self, plan: CleaningPlan) -> None:
        self.plan = plan
        self.calls = 0

    def propose(self, dataframe, profile=None) -> CleaningPlan:
        self.calls += 1
        return self.plan


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Graph test",
        confidence=1.0,
    )


def test_graph_stops_for_approval_after_planning() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    orchestrator = PreprocessingGraphOrchestrator()

    state = orchestrator.graph.invoke({"dataframe": dataframe, "approved": None})

    assert state["status"] == "awaiting_approval"
    assert state["profile"].rows == 3
    assert state["cleaning_plan"].steps[0].operation == "fill_median"
    assert "candidate" not in state


def test_approved_graph_skips_replanning_and_commits() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None, 30.0]})
    planner = CountingPlanner(CleaningPlan(steps=[step("fill_median", "age")]))
    orchestrator = PreprocessingGraphOrchestrator(cleaning_agent=planner)

    proposal = orchestrator.propose(dataframe)
    outcome = orchestrator.execute_approved(dataframe, proposal.plan)

    assert planner.calls == 1
    assert outcome.validation.valid
    assert not outcome.rolled_back
    assert outcome.dataframe["age"].isna().sum() == 0


def test_rejected_graph_returns_untouched_data() -> None:
    dataframe = pd.DataFrame({"age": [10.0, None]})
    plan = CleaningPlan(steps=[step("fill_median", "age")])
    orchestrator = PreprocessingGraphOrchestrator()

    state = orchestrator.graph.invoke(
        {"dataframe": dataframe, "cleaning_plan": plan, "approved": False}
    )

    assert state["status"] == "rejected"
    assert state["result"].equals(dataframe)
    assert dataframe["age"].isna().sum() == 1


def test_validation_failure_routes_to_rollback() -> None:
    dataframe = pd.DataFrame({"age": ["10", "unknown", "30"]})
    plan = CleaningPlan(steps=[step("convert_numeric", "age")])
    orchestrator = PreprocessingGraphOrchestrator()

    outcome = orchestrator.execute_approved(dataframe, plan)

    assert outcome.rolled_back
    assert not outcome.validation.valid
    assert outcome.dataframe.equals(dataframe)
