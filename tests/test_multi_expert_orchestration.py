import pandas as pd

from agents.multi_expert import (
    Arbiter,
    CriticAgent,
    DuplicateExpert,
    MissingValueExpert,
    MultiExpertCleaningAgent,
    RouterAgent,
)
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.orchestration import CriticResult
from tools.profiler import profile_dataframe
from agents.orchestrator import PreprocessingOrchestrator


def make_step(operation: str, column: str | None = None, confidence: float = 1.0) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Multi-expert test proposal",
        confidence=confidence,
    )


def test_router_sends_numeric_columns_to_numeric_expert() -> None:
    dataframe = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})

    routes = RouterAgent().route(dataframe)

    assert any(route.column == "amount" and route.selected_expert == "NumericExpert" for route in routes)


def test_router_sends_categorical_and_text_columns_to_suitable_experts() -> None:
    dataframe = pd.DataFrame({
        "city": ["SG", "KL", "SG"],
        "comment": [" first note ", "second note", "third note"],
    })

    routes = RouterAgent().route(dataframe)

    assert any(route.column == "city" and route.selected_expert == "CategoricalExpert" for route in routes)
    assert any(route.column == "comment" and route.selected_expert == "TextExpert" for route in routes)


def test_router_avoids_specialist_routing_for_protected_id_columns() -> None:
    dataframe = pd.DataFrame({"customer_id": [" A-1 ", None], "value": [1, 2]})

    routes = RouterAgent().route(dataframe)
    id_routes = [route for route in routes if route.column == "customer_id"]

    assert len(id_routes) == 1
    assert id_routes[0].issue_type == "protected_identifier"
    assert id_routes[0].selected_expert is None


def test_duplicate_expert_proposes_drop_duplicates_when_evidence_exists() -> None:
    dataframe = pd.DataFrame({"value": [1, 1, 2]})
    profile = profile_dataframe(dataframe)
    routes = RouterAgent().route(dataframe, profile)

    proposals = DuplicateExpert().propose(dataframe, profile, routes)

    assert [(step.operation, step.column) for step in proposals] == [("drop_duplicates", None)]


def test_missing_value_expert_proposes_safe_fill_and_leave_unchanged() -> None:
    dataframe = pd.DataFrame({
        "amount": [10.0, None, 30.0],
        "city": ["SG", None, "SG"],
        "unknown": [None, None, None],
    })
    profile = profile_dataframe(dataframe)
    routes = RouterAgent().route(dataframe, profile)

    proposals = MissingValueExpert().propose(dataframe, profile, routes)
    operations = {(step.column, step.operation) for step in proposals}

    assert ("amount", "fill_median") in operations
    assert ("city", "fill_mode") in operations
    assert ("unknown", "leave_unchanged") in operations


def test_critic_rejects_protected_column_modification() -> None:
    dataframe = pd.DataFrame({"customer_id": [" A-1 ", "A-2"]})
    unsafe = make_step("strip_whitespace", "customer_id")

    result = CriticAgent().review(dataframe, [unsafe])

    assert result.approved_steps == []
    assert result.rejected_steps == [unsafe]
    assert any("protected" in reason for reason in result.reasons)


def test_critic_rejects_conflicting_fill_operations() -> None:
    dataframe = pd.DataFrame({"amount": [1.0, None, 3.0]})
    mean = make_step("fill_mean", "amount")
    median = make_step("fill_median", "amount")

    result = CriticAgent().review(dataframe, [mean, median])

    assert result.approved_steps == []
    assert set(step.operation for step in result.rejected_steps) == {"fill_mean", "fill_median"}
    assert any("conflicting operations" in warning for warning in result.warnings)


def test_critic_rejects_unsafe_conversion_and_all_missing_fill() -> None:
    dataframe = pd.DataFrame({"label": ["one", "two"], "unknown": [None, None]})

    result = CriticAgent().review(dataframe, [
        make_step("convert_numeric", "label"),
        make_step("fill_mode", "unknown"),
    ])

    assert len(result.rejected_steps) == 2
    assert any("unsafe numeric conversion" in reason for reason in result.reasons)
    assert any("entirely missing" in reason for reason in result.reasons)


def test_arbiter_deduplicates_repeated_steps_using_highest_confidence() -> None:
    lower = make_step("strip_whitespace", "city", confidence=0.7)
    higher = make_step("strip_whitespace", "city", confidence=0.9)
    review = CriticResult(approved_steps=[lower, higher])

    plan = Arbiter().resolve(review)

    assert len(plan.steps) == 1
    assert plan.steps[0].confidence == 0.9


def test_arbiter_resolves_unreviewed_conflicts_conservatively() -> None:
    review = CriticResult(approved_steps=[
        make_step("fill_mean", "amount"),
        make_step("fill_median", "amount"),
    ])

    plan = Arbiter().resolve(review)

    assert plan.steps == []


def test_multi_expert_agent_returns_valid_plan_and_visible_trace() -> None:
    dataframe = pd.DataFrame({
        "customer_id": ["A-1", "A-2", "A-3"],
        "amount": ["10", None, "30"],
        "city": ["Singapore", "singapore", None],
    })
    agent = MultiExpertCleaningAgent()

    plan = agent.propose(dataframe)

    assert isinstance(plan, CleaningPlan)
    assert agent.last_trace is not None
    assert agent.last_trace.routes
    assert "NumericExpert" in agent.last_trace.specialist_proposals
    assert agent.last_trace.final_plan == plan
    assert all(step.column != "customer_id" for step in plan.steps)


def test_final_plan_executes_through_existing_graph_and_validation_pipeline() -> None:
    dataframe = pd.DataFrame({
        "row_id": [1, 2, 2, 4],
        "amount": ["10", None, None, "30"],
        "city": ["Singapore", " singapore ", " singapore ", "SINGAPORE"],
    })
    planner = MultiExpertCleaningAgent()
    orchestrator = PreprocessingOrchestrator(cleaning_agent=planner)

    proposal = orchestrator.propose(dataframe)
    outcome = orchestrator.execute_approved(dataframe, proposal.plan)

    assert outcome.validation.valid
    assert not outcome.rolled_back
    assert len(outcome.dataframe) == 3
    assert outcome.dataframe["amount"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(outcome.dataframe["amount"])
    assert dataframe["amount"].isna().sum() == 2
