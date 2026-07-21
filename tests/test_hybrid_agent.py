import pandas as pd

from agents.hybrid_cleaning_agent import HybridCleaningAgent
from agents.local_llm_cleaning_agent import LocalLLMPlanningError
from models.cleaning_plan import CleaningPlan, CleaningStep


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Test plan",
        confidence=0.9,
    )


class FixedPlanner:
    name = "fixed_llm"

    def __init__(self, plan: CleaningPlan) -> None:
        self.plan = plan

    def propose(self, dataframe, profile=None) -> CleaningPlan:
        return self.plan


class FailingPlanner:
    name = "failing_llm"

    def propose(self, dataframe, profile=None) -> CleaningPlan:
        raise LocalLLMPlanningError("invalid local plan")


def test_hybrid_restores_duplicate_step_missed_by_llm() -> None:
    dataframe = pd.DataFrame(
        {"row_id": [1, 2, 2], "age": [10.0, None, None]}
    )
    llm_plan = CleaningPlan(steps=[step("fill_mean", "age")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)
    operations = [(item.operation, item.column) for item in plan.steps]

    assert operations[0] == ("drop_duplicates", None)
    assert ("fill_mean", "age") in operations


def test_hybrid_prefers_valid_llm_fill_over_rule_fill() -> None:
    dataframe = pd.DataFrame({"row_id": [1, 2, 3], "age": [10.0, None, 30.0]})
    llm_plan = CleaningPlan(steps=[step("fill_mean", "age")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)
    age_operations = [item.operation for item in plan.steps if item.column == "age"]

    assert age_operations == ["fill_mean"]


def test_hybrid_uses_rule_fill_when_llm_omits_missing_column() -> None:
    dataframe = pd.DataFrame({"row_id": [1, 2, 3], "age": [10.0, None, 30.0]})

    plan = HybridCleaningAgent(FixedPlanner(CleaningPlan(steps=[]))).propose(
        dataframe
    )

    assert any(
        item.operation == "fill_median" and item.column == "age"
        for item in plan.steps
    )


def test_hybrid_never_imputes_entirely_missing_column() -> None:
    dataframe = pd.DataFrame({"row_id": [1, 2], "unknown": [None, None]})
    llm_plan = CleaningPlan(steps=[step("fill_mode", "unknown")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)
    unknown_steps = [item for item in plan.steps if item.column == "unknown"]

    assert len(unknown_steps) == 1
    assert unknown_steps[0].operation == "leave_unchanged"


def test_hybrid_falls_back_when_llm_fails() -> None:
    dataframe = pd.DataFrame({"row_id": [1, 2, 3], "age": [10.0, None, 30.0]})
    agent = HybridCleaningAgent(FailingPlanner())

    plan = agent.propose(dataframe)

    assert plan.steps[0].operation == "fill_median"
    assert agent.last_llm_error == "invalid local plan"


def test_hybrid_rejects_numeric_fill_for_categorical_column() -> None:
    dataframe = pd.DataFrame({"city": ["A", None, "B"]})
    llm_plan = CleaningPlan(steps=[step("fill_mean", "city")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)

    assert [item.operation for item in plan.steps] == ["fill_mode"]


def test_hybrid_accepts_lossless_numeric_conversion() -> None:
    dataframe = pd.DataFrame({"age": ["10", "20", "30"]})
    llm_plan = CleaningPlan(steps=[step("convert_numeric", "age")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)

    assert [item.operation for item in plan.steps] == ["convert_numeric"]


def test_hybrid_rejects_lossy_numeric_conversion() -> None:
    dataframe = pd.DataFrame({"age": ["10", "unknown", "30"]})
    llm_plan = CleaningPlan(steps=[step("convert_numeric", "age")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)

    assert plan.steps == []


def test_hybrid_accepts_robust_numeric_text_parsing() -> None:
    dataframe = pd.DataFrame({"amount": ["$1,000", "$2,000"]})
    llm_plan = CleaningPlan(steps=[step("parse_numeric_text", "amount")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)

    assert [item.operation for item in plan.steps] == ["parse_numeric_text"]


def test_hybrid_rejects_operations_on_protected_columns() -> None:
    dataframe = pd.DataFrame({"order_id": [" 1 ", " 2 "]})
    llm_plan = CleaningPlan(steps=[step("strip_whitespace", "order_id")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)

    assert plan.steps == []


def test_hybrid_keeps_rule_text_cleanup_when_llm_omits_it() -> None:
    dataframe = pd.DataFrame({"city": ["Singapore", " singapore "]})

    plan = HybridCleaningAgent(FixedPlanner(CleaningPlan(steps=[]))).propose(
        dataframe
    )

    assert any(item.operation == "strip_whitespace" for item in plan.steps)


def test_hybrid_accepts_category_typo_normalization_when_supported() -> None:
    dataframe = pd.DataFrame(
        {"city": ["Singapore", "Singapore", "Singaproe", "Bangkok"]}
    )
    llm_plan = CleaningPlan(steps=[step("normalize_category_typos", "city")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)

    assert any(item.operation == "normalize_category_typos" for item in plan.steps)


def test_hybrid_rejects_outlier_flag_for_non_numeric_column() -> None:
    dataframe = pd.DataFrame({"city": ["A", "B", "C"]})
    llm_plan = CleaningPlan(steps=[step("flag_outliers_iqr", "city")])

    plan = HybridCleaningAgent(FixedPlanner(llm_plan)).propose(dataframe)

    assert plan.steps == []
