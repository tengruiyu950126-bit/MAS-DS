import pandas as pd
import pytest

from agents.cleaning_agent import RuleBasedCleaningAgent
from evaluation.corrupt_dataset import ROW_ID_COLUMN, inject_corruption
from evaluation.datasets import available_dataset_names, load_dataset, make_dirty_copy
from evaluation.metrics import calculate_detection_metrics, calculate_repair_metrics
from evaluation.run_experiment import build_planner, evaluate_once, make_demo_dataset
from models.cleaning_plan import CleaningPlan, CleaningStep
from models.policy import PreprocessingPolicy


def step(operation: str, column: str | None = None) -> CleaningStep:
    return CleaningStep(
        operation=operation,
        column=column,
        reason="Evaluation test",
        confidence=1.0,
    )


def test_default_llm_experiment_timeout_satisfies_provider_policy() -> None:
    planner = build_planner("llm", "local-test")

    assert planner._client.timeout_seconds == 30


def test_missing_corruption_is_reproducible_and_non_mutating() -> None:
    clean = make_demo_dataset(12)
    original = clean.copy(deep=True)

    first = inject_corruption(clean, "missing_value", fraction=0.1, seed=7)
    second = inject_corruption(clean, "missing_value", fraction=0.1, seed=7)

    assert first.corrupted.equals(second.corrupted)
    assert first.records == second.records
    assert clean.equals(original)
    assert ROW_ID_COLUMN in first.corrupted.columns


def test_duplicate_corruption_adds_exact_rows() -> None:
    clean = make_demo_dataset(10)

    result = inject_corruption(clean, "duplicate_row", fraction=0.2, seed=1)

    assert len(result.corrupted) == 12
    assert result.corrupted.duplicated().sum() == 2
    assert len(result.records) == 2


def test_numeric_and_datetime_corruptions_target_valid_types() -> None:
    clean = make_demo_dataset(10)

    numeric = inject_corruption(clean, "numeric_type", fraction=0.1, seed=2)
    dates = inject_corruption(clean, "datetime_type", fraction=0.1, seed=2)

    assert all(record.column in {"age", "score"} for record in numeric.records)
    assert all(record.column == "signup_date" for record in dates.records)
    assert all(isinstance(record.corrupted_value, str) for record in numeric.records)
    assert all(isinstance(record.corrupted_value, str) for record in dates.records)
    assert all(
        pd.notna(pd.to_numeric(record.corrupted_value, errors="coerce"))
        for record in numeric.records
    )
    assert all(
        pd.notna(pd.to_datetime(record.corrupted_value, errors="coerce"))
        for record in dates.records
    )


@pytest.mark.parametrize("dataset_name", available_dataset_names())
def test_builtin_datasets_support_all_corruptions(dataset_name: str) -> None:
    dataset = load_dataset(dataset_name, rows=16)

    assert len(dataset.dataframe) == 16
    assert dataset.dataframe.select_dtypes(include="number").shape[1] >= 1
    assert dataset.dataframe.select_dtypes(include="datetime").shape[1] >= 1

    for corruption in [
        "missing_value",
        "duplicate_row",
        "numeric_type",
        "datetime_type",
    ]:
        case = inject_corruption(
            dataset.dataframe,
            corruption,
            fraction=0.1,
            seed=3,
        )
        assert case.records
        assert ROW_ID_COLUMN in case.corrupted.columns


def test_dirty_sample_copy_adds_demo_issues_without_internal_ids() -> None:
    dataset = load_dataset("sales", rows=30)

    dirty = make_dirty_copy(dataset.dataframe, seed=4)

    assert len(dirty) > len(dataset.dataframe)
    assert dirty.isna().sum().sum() > 0
    assert dirty.duplicated().sum() > 0
    assert ROW_ID_COLUMN not in dirty.columns


def test_detection_metrics_match_column_level_plan() -> None:
    clean = pd.DataFrame({"age": [10, 20, 30]})
    case = inject_corruption(
        clean,
        "missing_value",
        fraction=0.34,
        seed=1,
        columns=["age"],
    )
    plan = CleaningPlan(steps=[step("fill_median", "age")])

    metrics = calculate_detection_metrics(case.records, plan)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_parse_numeric_text_counts_as_numeric_detection() -> None:
    clean = pd.DataFrame({"amount": [10, 20, 30]})
    case = inject_corruption(
        clean,
        "numeric_type",
        fraction=1 / 3,
        seed=1,
        columns=["amount"],
    )
    plan = CleaningPlan(steps=[step("parse_numeric_text", "amount")])

    metrics = calculate_detection_metrics(case.records, plan)

    assert metrics.f1 == 1.0


def test_repair_and_preservation_metrics() -> None:
    clean = pd.DataFrame({"age": [10.0, 20.0, 30.0], "city": ["A", "B", "C"]})
    case = inject_corruption(
        clean,
        "missing_value",
        fraction=1 / 3,
        seed=3,
        columns=["age"],
    )
    repaired = case.corrupted.copy(deep=True)
    for record in case.records:
        repaired.loc[
            repaired[ROW_ID_COLUMN] == record.row_id,
            record.column,
        ] = record.original_value

    metrics = calculate_repair_metrics(case.clean, repaired, case.records)

    assert metrics.repair_success_rate == 1.0
    assert metrics.data_preservation_rate == 1.0
    assert metrics.false_modification_rate == 0.0
    assert metrics.schema_preserved


def test_rule_experiment_runs_without_llm() -> None:
    record = evaluate_once(
        make_demo_dataset(12),
        dataset_name="demo",
        method="rule",
        corruption="duplicate_row",
        seed=0,
        fraction=0.1,
        model="unused",
    )

    assert record.method == "rule"
    assert record.dataset == "demo"
    assert record.detection_f1 == 1.0
    assert record.repair_success_rate == 1.0
    assert not record.invalid_plan
    assert record.latency_seconds >= 0


def test_experiment_planner_respects_policy() -> None:
    dataframe = pd.DataFrame({"city": [" Singapore ", None, "KL"]})
    planner = build_planner(
        "rule",
        model="unused",
        policy=PreprocessingPolicy(protected_columns=["city"]),
    )

    plan = planner.propose(dataframe)

    assert [(step.operation, step.column) for step in plan.steps] == [
        ("leave_unchanged", "city")
    ]


def test_parseable_numeric_string_is_not_repaired_until_dtype_is_restored() -> None:
    clean = pd.DataFrame({"age": [10, 20, 30]})
    case = inject_corruption(
        clean,
        "numeric_type",
        fraction=1 / 3,
        seed=1,
        columns=["age"],
    )

    unchanged = calculate_repair_metrics(case.clean, case.corrupted, case.records)
    converted = case.corrupted.copy(deep=True)
    converted["age"] = pd.to_numeric(converted["age"])
    repaired = calculate_repair_metrics(case.clean, converted, case.records)

    assert unchanged.repair_success_rate == 0.0
    assert repaired.repair_success_rate == 1.0


def test_expected_operation_with_wrong_value_fails_repair_evaluation() -> None:
    clean = pd.DataFrame({"age": [10.0, 20.0, 30.0]})
    case = inject_corruption(
        clean,
        "missing_value",
        fraction=1 / 3,
        seed=3,
        columns=["age"],
    )
    wrong = case.corrupted.copy(deep=True)
    for record in case.records:
        wrong.loc[wrong[ROW_ID_COLUMN] == record.row_id, "age"] = 999.0

    metrics = calculate_repair_metrics(case.clean, wrong, case.records)

    assert metrics.repair_success_rate == 0.0
    assert metrics.exact_recovery_rate == 0.0


def test_unaffected_and_protected_cell_damage_is_measured() -> None:
    clean = pd.DataFrame(
        {"account_id": ["A", "B"], "age": [10.0, 20.0], "city": ["X", "Y"]}
    )
    case = inject_corruption(
        clean,
        "missing_value",
        fraction=0.5,
        seed=1,
        columns=["age"],
    )
    damaged = case.clean.copy(deep=True)
    damaged.loc[0, "city"] = "DAMAGED"
    damaged.loc[0, "account_id"] = "CHANGED"

    metrics = calculate_repair_metrics(
        case.clean,
        damaged,
        case.records,
        protected_columns={"account_id"},
    )

    assert metrics.false_modifications == 2
    assert metrics.protected_column_modifications == 1
    assert metrics.data_preservation_rate < 1.0


def test_schema_damage_and_missing_rows_are_measured() -> None:
    clean = pd.DataFrame({"age": [10.0, 20.0], "city": ["X", "Y"]})
    case = inject_corruption(
        clean,
        "missing_value",
        fraction=0.5,
        seed=1,
        columns=["age"],
    )
    damaged = case.clean.drop(columns=["city"]).iloc[:1].copy()

    metrics = calculate_repair_metrics(case.clean, damaged, case.records)

    assert not metrics.schema_preserved
    assert metrics.unexpected_column_changes == 1
    assert metrics.unexpected_row_deletions == 1
