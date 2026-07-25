import pandas as pd

from agents.cleaning_agent import RuleBasedCleaningAgent
from agents.orchestrator import PreprocessingOrchestrator
from tools.profiler import profile_dataframe


def test_profiler_returns_structured_profile() -> None:
    dataframe = pd.DataFrame({"value": [1, None]})

    profile = profile_dataframe(dataframe)

    assert profile.rows == 2
    assert profile.column_profiles["value"].missing_count == 1


def test_cleaning_agent_proposes_duplicate_and_missing_value_steps() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [10.0, None, 10.0, 10.0],
            "city": ["SG", "KL", "SG", "SG"],
        }
    )

    plan = RuleBasedCleaningAgent().propose(dataframe)
    operations = [(step.operation, step.column) for step in plan.steps]

    assert ("drop_duplicates", None) in operations
    assert ("fill_median", "age") in operations


def test_cleaning_agent_uses_mode_for_text() -> None:
    dataframe = pd.DataFrame(
        {"row_id": [1, 2, 3], "city": ["SG", "SG", None]}
    )

    plan = RuleBasedCleaningAgent().propose(dataframe)
    fill_step = next(step for step in plan.steps if step.column == "city")

    assert fill_step.operation == "fill_mode"


def test_cleaning_agent_does_not_invent_all_missing_values() -> None:
    dataframe = pd.DataFrame(
        {"row_id": [1, 2], "unknown": [None, None]}
    )

    plan = RuleBasedCleaningAgent().propose(dataframe)
    unchanged_step = next(step for step in plan.steps if step.column == "unknown")

    assert unchanged_step.operation == "leave_unchanged"
    assert unchanged_step.confidence == 1.0


def test_cleaning_agent_proposes_text_standardization_steps() -> None:
    dataframe = pd.DataFrame(
        {
            "city": ["Singapore", " singapore ", "SINGAPORE"],
            "segment": ["growth", " growth", "enterprise"],
        }
    )

    plan = RuleBasedCleaningAgent().propose(dataframe)
    operations = [(step.operation, step.column) for step in plan.steps]

    assert ("strip_whitespace", "city") in operations
    assert ("normalize_case", "city") in operations
    assert ("strip_whitespace", "segment") in operations


def test_cleaning_agent_proposes_category_typo_normalization() -> None:
    dataframe = pd.DataFrame(
        {"city": ["Singapore", "Singapore", "Singaproe", "Bangkok"]}
    )

    plan = RuleBasedCleaningAgent().propose(dataframe)
    operations = [(step.operation, step.column) for step in plan.steps]

    assert ("normalize_category_typos", "city") in operations


def test_cleaning_agent_flags_outliers_without_imputing_them() -> None:
    dataframe = pd.DataFrame({"amount": [10, 11, 9, 10, 12, 11, 10, 9, 500]})

    plan = RuleBasedCleaningAgent().propose(dataframe)
    operations = [(step.operation, step.column) for step in plan.steps]

    assert ("flag_outliers_iqr", "amount") in operations
    assert ("fill_median", "amount") not in operations


def test_cleaning_agent_parses_formatted_numeric_text() -> None:
    dataframe = pd.DataFrame({"amount": ["$1,200", "$2,500", None]})

    plan = RuleBasedCleaningAgent().propose(dataframe)
    operations = [(step.operation, step.column) for step in plan.steps]

    assert ("parse_numeric_text", "amount") in operations
    assert ("fill_median", "amount") in operations


def test_cleaning_agent_protects_identifier_columns() -> None:
    dataframe = pd.DataFrame({"customer_id": [" C-1 ", None], "city": ["SG", None]})

    plan = RuleBasedCleaningAgent().propose(dataframe)
    operations = [(step.operation, step.column) for step in plan.steps]

    assert ("leave_unchanged", "customer_id") in operations
    assert ("strip_whitespace", "customer_id") not in operations
    assert ("fill_mode", "customer_id") not in operations
    assert ("fill_mode", "city") in operations


def test_clean_dataframe_produces_empty_plan() -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    plan = RuleBasedCleaningAgent().propose(dataframe)

    assert plan.steps == []


def test_orchestrator_runs_end_to_end_after_approval() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [10.0, None, 20.0, 20.0],
            "city": ["US", "KL", "SG", "SG"],
        }
    )
    orchestrator = PreprocessingOrchestrator()

    proposal = orchestrator.propose(dataframe)
    outcome = orchestrator.execute_approved(dataframe, proposal.plan)

    assert not outcome.rolled_back
    assert outcome.validation.valid
    assert len(outcome.dataframe) == 3
    assert outcome.dataframe["age"].isna().sum() == 0
    assert dataframe["age"].isna().sum() == 1
    assert len(outcome.execution_records) == 2
