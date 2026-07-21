from __future__ import annotations

import pandas as pd

from agents.cleaning_agent import RuleBasedCleaningAgent
from tools.chunked import execute_chunked_csv
from workflow.graph import PreprocessingGraphOrchestrator


def test_chunked_preprocessing_matches_in_memory_rule_pipeline(tmp_path) -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C2"],
            "age": [10.0, None, 30.0, None, 50.0, None],
            "city": [
                " Singapore ",
                "singapore",
                "Singaproe",
                "Bangkok",
                "bangkok",
                "singapore",
            ],
            "amount_text": ["$10.00", None, "$30.00", "$40.00", "$50.00", None],
        }
    )
    # Exact duplicate placed across chunk boundaries.
    dataframe = pd.concat([dataframe, dataframe.iloc[[1]]], ignore_index=True)
    input_csv = tmp_path / "dirty.csv"
    output_csv = tmp_path / "cleaned.csv"
    dataframe.to_csv(input_csv, index=False)

    rule_agent = RuleBasedCleaningAgent()
    plan = rule_agent.propose(dataframe)
    expected = PreprocessingGraphOrchestrator(
        cleaning_agent=rule_agent
    ).execute_approved(dataframe, plan).dataframe

    summary = execute_chunked_csv(
        input_csv,
        output_csv,
        plan=plan,
        chunk_size=2,
        planning_sample_rows=7,
    )
    actual = pd.read_csv(output_csv)

    assert summary.rows_input == 7
    assert summary.rows_output == 5
    assert summary.rows_removed_as_duplicates == 2
    assert summary.missing_after == 0
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_chunked_fill_uses_global_median_not_per_chunk_median(tmp_path) -> None:
    dataframe = pd.DataFrame(
        {
            "age": [1.0, None, 100.0, 101.0, None, 102.0],
            "city": ["A", "A", "B", "B", "B", "B"],
        }
    )
    input_csv = tmp_path / "median_dirty.csv"
    output_csv = tmp_path / "median_cleaned.csv"
    dataframe.to_csv(input_csv, index=False)

    plan = RuleBasedCleaningAgent().propose(dataframe)
    execute_chunked_csv(
        input_csv,
        output_csv,
        plan=plan,
        chunk_size=2,
        planning_sample_rows=6,
    )
    cleaned = pd.read_csv(output_csv)

    # Global median of [1, 100, 101, 102] is 100.5. A per-chunk strategy would
    # produce different fill values in different chunks.
    assert cleaned["age"].tolist() == [1.0, 100.5, 100.0, 101.0, 100.5, 102.0]


def test_chunked_auto_adds_drop_duplicates_when_sample_misses_duplicates(tmp_path) -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C1"],
            "age": [20, 21, 22, 23, 20],
        }
    )
    input_csv = tmp_path / "dupes_dirty.csv"
    output_csv = tmp_path / "dupes_cleaned.csv"
    summary_csv = tmp_path / "dupes_summary.csv"
    plan_json = tmp_path / "dupes_plan.json"
    dataframe.to_csv(input_csv, index=False)

    summary = execute_chunked_csv(
        input_csv,
        output_csv,
        chunk_size=2,
        planning_sample_rows=2,
        summary_csv=summary_csv,
        plan_json=plan_json,
    )
    cleaned = pd.read_csv(output_csv)

    assert summary.rows_input == 5
    assert summary.rows_output == 4
    assert "drop_duplicates:*" in summary.operations
    assert summary_csv.exists()
    assert plan_json.exists()
    assert cleaned["customer_id"].tolist() == ["C1", "C2", "C3", "C4"]
